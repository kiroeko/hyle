"""Original DeepSeek/SGLang/ds4 tensor sequence source qualification."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'third_party/iree_vulkan_runtime/programs'))
from generate_mlx import wide, narrow


def decode_value(code, name):
    return [f'%{name}_byte = arith.trunci {code} : i32 to i8',
            f'%{name}_encoded = arith.bitcast %{name}_byte : i8 to f8E4M3FN',
            f'%{name}_original = arith.extf %{name}_encoded : f8E4M3FN to f32',
            f'%{name}_mask = arith.constant 127 : i32',
            f'%{name}_magnitude = arith.andi {code}, %{name}_mask : i32',
            f'%{name}_is_nan = arith.cmpi eq, %{name}_magnitude, %{name}_mask : i32',
            f'%{name}_nan_bits = arith.constant 2143289344 : i32',
            f'%{name}_nan = arith.bitcast %{name}_nan_bits : i32 to f32',
            f'%{name} = arith.select %{name}_is_nan, %{name}_nan, %{name}_original : f32']


def finite(dtype):
    t = 'f32' if dtype == 'f32' else 'i16'
    bits = ['%bits = arith.bitcast %v : f32 to i32'] if dtype == 'f32' else ['%bits = arith.extui %v : i16 to i32']
    return '\n'.join([f'func.func private @finite_{dtype}(%x: tensor<?x?x{t}>) -> tensor<i32> {{',
        '%zero = arith.constant 0 : i32', '%empty = tensor.empty() : tensor<i32>',
        '%init = linalg.fill ins(%zero : i32) outs(%empty : tensor<i32>) -> tensor<i32>',
        '%flag = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>], iterator_types = ["reduction","reduction"]}',
        f'ins(%x : tensor<?x?x{t}>) outs(%init : tensor<i32>) {{', f'^bb0(%v: {t}, %old: i32):', *bits,
        f'%mask = arith.constant {2139095040 if dtype == "f32" else 32640} : i32',
        '%exp = arith.andi %bits, %mask : i32', '%invalid = arith.cmpi eq, %exp, %mask : i32',
        '%flag_bit = arith.extui %invalid : i1 to i32', '%value = arith.ori %old, %flag_bit : i32',
        'linalg.yield %value : i32', '} -> tensor<i32>', 'return %flag : tensor<i32>', '}'])


def source_fp8_encode():
    # Direct original ds4 dsv4_e4m3fn_dequant_cpu search/even-code sequence.
    # Values use its unchanged dsv4_e4m3fn_value_cpu definition.
    values = [code / 512 if code < 8 else (1 + (code & 7) / 8) * 2 ** ((code >> 3) - 7)
              for code in range(127)]
    table = ', '.join(str(float(x)) for x in values)
    return '\n'.join(['func.func private @source_fp8_encode(%input: f32) -> i8 {',
        f'%table = arith.constant dense<[{table}]> : tensor<127xf32>',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index', '%c2 = arith.constant 2 : index',
        '%c126 = arith.constant 126 : index', '%cap = arith.constant 448.0 : f32',
        '%abs = math.absf %input : f32', '%ax = arith.minimumf %abs, %cap : f32',
        '%bounds:2 = scf.while (%lo = %c0, %hi = %c126) : (index, index) -> (index, index) {',
        '%condition = arith.cmpi ult, %lo, %hi : index', 'scf.condition(%condition) %lo, %hi : index, index',
        '} do {', '^bb0(%lo: index, %hi: index):', '%sum = arith.addi %lo, %hi : index',
        '%raised = arith.addi %sum, %c1 : index', '%mid = arith.divui %raised, %c2 : index',
        '%value = tensor.extract %table[%mid] : tensor<127xf32>', '%lower = arith.cmpf ole, %value, %ax : f32',
        '%before = arith.subi %mid, %c1 : index', '%new_lo = arith.select %lower, %mid, %lo : index',
        '%new_hi = arith.select %lower, %hi, %before : index', 'scf.yield %new_lo, %new_hi : index, index', '}',
        '%next0 = arith.addi %bounds#0, %c1 : index', '%next = arith.minui %next0, %c126 : index',
        '%a = tensor.extract %table[%bounds#0] : tensor<127xf32>', '%b = tensor.extract %table[%next] : tensor<127xf32>',
        '%ad = arith.subf %ax, %a : f32', '%bd = arith.subf %ax, %b : f32',
        '%da = math.absf %ad : f32', '%db = math.absf %bd : f32', '%closer = arith.cmpf olt, %db, %da : f32',
        '%tie = arith.cmpf oeq, %db, %da : f32', '%best32 = arith.index_cast %bounds#0 : index to i32',
        '%next32 = arith.index_cast %next : index to i32', '%one = arith.constant 1 : i32', '%zero = arith.constant 0 : i32',
        '%bestodd = arith.andi %best32, %one : i32', '%nextodd = arith.andi %next32, %one : i32',
        '%odd = arith.cmpi ne, %bestodd, %zero : i32', '%even = arith.cmpi eq, %nextodd, %zero : i32',
        '%even_tie = arith.andi %tie, %even : i1', '%choose_tie = arith.andi %even_tie, %odd : i1',
        '%choose = arith.ori %closer, %choose_tie : i1', '%selected = arith.select %choose, %next32, %best32 : i32',
        '%bits = arith.bitcast %input : f32 to i32', '%sign_shift = arith.constant 24 : i32',
        '%sign_mask = arith.constant 128 : i32', '%shifted = arith.shrui %bits, %sign_shift : i32',
        '%sign = arith.andi %shifted, %sign_mask : i32', '%signed = arith.ori %selected, %sign : i32',
        '%result = arith.trunci %signed : i32 to i8', 'return %result : i8', '}'])


def divide_448_rne():
    # Quantizer floors guarantee a positive normal F32 quotient. SPIR-V FDiv
    # permits a few ulps of error, which can flip the final BF16 halfway case.
    # Divide the integer significand by 7, then adjust its exponent for 64.
    # Nonfinite source data still follows the original finite-input failure.
    return '''func.func private @divide_448_rne(%x: f32) -> f32 {
%c2 = arith.constant 2 : i32
%c3 = arith.constant 3 : i32
%c4 = arith.constant 4 : i32
%c7 = arith.constant 7 : i32
%c8 = arith.constant 8 : i32
%c9 = arith.constant 9 : i32
%c23 = arith.constant 23 : i32
%c255 = arith.constant 255 : i32
%mask = arith.constant 8388607 : i32
%hidden = arith.constant 8388608 : i32
%threshold = arith.constant 14680064 : i32
%bits = arith.bitcast %x : f32 to i32
%fraction = arith.andi %bits, %mask : i32
%mantissa = arith.ori %fraction, %hidden : i32
%upper = arith.cmpi uge, %mantissa, %threshold : i32
%shift = arith.select %upper, %c2, %c3 : i32
%numerator = arith.shli %mantissa, %shift : i32
%quotient = arith.divui %numerator, %c7 : i32
%remainder = arith.remui %numerator, %c7 : i32
%round = arith.cmpi uge, %remainder, %c4 : i32
%extra = arith.extui %round : i1 to i32
%rounded = arith.addi %quotient, %extra : i32
%biased = arith.shrui %bits, %c23 : i32
%exponent = arith.andi %biased, %c255 : i32
%offset = arith.select %upper, %c8, %c9 : i32
%adjusted = arith.subi %exponent, %offset : i32
%exponent_bits = arith.shli %adjusted, %c23 : i32
%stored_fraction = arith.andi %rounded, %mask : i32
%result_bits = arith.ori %exponent_bits, %stored_fraction : i32
%normal = arith.bitcast %result_bits : i32 to f32
%nonfinite = arith.cmpi eq, %exponent, %c255 : i32
%result = arith.select %nonfinite, %x, %normal : f32
return %result : f32
}'''


def reciprocal_normal_rne():
    # Existing positive-normal SA floor and finite input contract keep 1/SA
    # normal too. Divide significands in U32: no approximate SPIR-V FDiv,
    # I64 capability or reciprocal work repeated for every activation element.
    return """func.func private @reciprocal_normal_rne(%x: f32) -> f32 {
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%c23 = arith.constant 23 : i32
%c253 = arith.constant 253 : i32
%c254 = arith.constant 254 : i32
%mask = arith.constant 8388607 : i32
%hidden = arith.constant 8388608 : i32
%double_hidden = arith.constant 16777216 : i32
%bits = arith.bitcast %x : f32 to i32
%fraction = arith.andi %bits, %mask : i32
%m = arith.ori %fraction, %hidden : i32
%e = arith.shrui %bits, %c23 : i32
%exact = arith.cmpi eq, %m, %hidden : i32
%numerator = arith.select %exact, %hidden, %double_hidden : i32
%base_exp = arith.select %exact, %c254, %c253 : i32
%out_exp = arith.subi %base_exp, %e : i32
%initial_r = arith.subi %numerator, %m : i32
%start = arith.constant 0 : index
%end = arith.constant 23 : index
%step = arith.constant 1 : index
%divided:2 = scf.for %i = %start to %end step %step iter_args(%q = %c1, %r = %initial_r) -> (i32, i32) {
  %shifted_r = arith.shli %r, %c1 : i32
  %take = arith.cmpi uge, %shifted_r, %m : i32
  %subtract = arith.select %take, %m, %c0 : i32
  %next_r = arith.subi %shifted_r, %subtract : i32
  %shifted_q = arith.shli %q, %c1 : i32
  %bit = arith.extui %take : i1 to i32
  %next_q = arith.ori %shifted_q, %bit : i32
  scf.yield %next_q, %next_r : i32, i32
}
%twice_r = arith.shli %divided#1, %c1 : i32
%above = arith.cmpi ugt, %twice_r, %m : i32
%tie = arith.cmpi eq, %twice_r, %m : i32
%odd_bits = arith.andi %divided#0, %c1 : i32
%odd = arith.cmpi ne, %odd_bits, %c0 : i32
%tie_up = arith.andi %tie, %odd : i1
%round = arith.ori %above, %tie_up : i1
%extra = arith.extui %round : i1 to i32
%rounded = arith.addi %divided#0, %extra : i32
%carry = arith.cmpi uge, %rounded, %double_hidden : i32
%carried_q = arith.shrui %rounded, %c1 : i32
%stored_q = arith.select %carry, %carried_q, %rounded : i32
%carry_bit = arith.extui %carry : i1 to i32
%final_exp = arith.addi %out_exp, %carry_bit : i32
%exponent_bits = arith.shli %final_exp, %c23 : i32
%stored_fraction = arith.andi %stored_q, %mask : i32
%result_bits = arith.ori %exponent_bits, %stored_fraction : i32
%result = arith.bitcast %result_bits : i32 to f32
return %result : f32
}"""


def normalize_divide_rne():
    # The public byte output cannot inherit SPIR-V FDiv's allowed ulp error:
    # an adjacent F32 at an E4 midpoint may select a different code. This is
    # the same significand long division as the strict Metal/Vulkan reference
    # helper. The positive scale is normal; a subnormal quotient is strictly
    # below E4's zero midpoint, so its original sign is the only stored bit.
    return """func.func private @normalize_divide_rne(%a: f32, %b: f32) -> f32 {
%zero = arith.constant 0 : i32
%one = arith.constant 1 : i32
%three = arith.constant 3 : i32
%four = arith.constant 4 : i32
%seven = arith.constant 7 : i32
%eight = arith.constant 8 : i32
%c23 = arith.constant 23 : i32
%c127 = arith.constant 127 : i32
%c255 = arith.constant 255 : i32
%abs_mask = arith.constant 2147483647 : i32
%sign_mask = arith.constant -2147483648 : i32
%mask = arith.constant 8388607 : i32
%hidden = arith.constant 8388608 : i32
%double_hidden = arith.constant 16777216 : i32
%inf_bits = arith.constant 2139095040 : i32
%a_bits = arith.bitcast %a : f32 to i32
%b_bits = arith.bitcast %b : f32 to i32
%sign = arith.andi %a_bits, %sign_mask : i32
%aa = arith.andi %a_bits, %abs_mask : i32
%bb = arith.andi %b_bits, %abs_mask : i32
%az = arith.cmpi eq, %aa, %zero : i32
%an = arith.cmpi uge, %aa, %inf_bits : i32
%bn = arith.cmpi uge, %bb, %inf_bits : i32
%be = arith.shrui %bb, %c23 : i32
%bz = arith.cmpi eq, %be, %zero : i32
%invalid0 = arith.ori %az, %an : i1
%invalid1 = arith.ori %bn, %bz : i1
%invalid = arith.ori %invalid0, %invalid1 : i1
%result_bits = scf.if %invalid -> (i32) {
  scf.yield %sign : i32
} else {
  %ae0 = arith.shrui %aa, %c23 : i32
  %af = arith.andi %aa, %mask : i32
  %a_normal = arith.cmpi ne, %ae0, %zero : i32
  %a_hidden = arith.select %a_normal, %hidden, %zero : i32
  %as0 = arith.ori %af, %a_hidden : i32
  %ae1 = arith.select %a_normal, %ae0, %one : i32
  %parts:2 = scf.while (%sig = %as0, %exp = %ae1) : (i32, i32) -> (i32, i32) {
    %small = arith.cmpi ult, %sig, %hidden : i32
    scf.condition(%small) %sig, %exp : i32, i32
  } do {
    ^bb0(%sig: i32, %exp: i32):
    %shifted = arith.shli %sig, %one : i32
    %lower = arith.subi %exp, %one : i32
    scf.yield %shifted, %lower : i32, i32
  }
  %bf = arith.andi %bb, %mask : i32
  %bs = arith.ori %bf, %hidden : i32
  %difference = arith.subi %parts#1, %be : i32
  %base = arith.addi %difference, %c127 : i32
  %small = arith.cmpi ult, %parts#0, %bs : i32
  %shifted_a = arith.shli %parts#0, %one : i32
  %initial_r = arith.select %small, %shifted_a, %parts#0 : i32
  %decrement = arith.extui %small : i1 to i32
  %exponent = arith.subi %base, %decrement : i32
  %start = arith.constant 0 : index
  %end = arith.constant 27 : index
  %step = arith.constant 1 : index
  %last = arith.constant 26 : index
  %division:2 = scf.for %i = %start to %end step %step iter_args(%q = %zero, %r = %initial_r) -> (i32, i32) {
    %take = arith.cmpi uge, %r, %bs : i32
    %subtract = arith.select %take, %bs, %zero : i32
    %remainder = arith.subi %r, %subtract : i32
    %shifted_q = arith.shli %q, %one : i32
    %bit = arith.extui %take : i1 to i32
    %next_q = arith.ori %shifted_q, %bit : i32
    %shifted_r = arith.shli %remainder, %one : i32
    %finished = arith.cmpi eq, %i, %last : index
    %next_r = arith.select %finished, %remainder, %shifted_r : i32
    scf.yield %next_q, %next_r : i32, i32
  }
  %has_remainder = arith.cmpi ne, %division#1, %zero : i32
  %sticky = arith.extui %has_remainder : i1 to i32
  %q = arith.ori %division#0, %sticky : i32
  %tail = arith.andi %q, %seven : i32
  %above = arith.cmpi ugt, %tail, %four : i32
  %tie = arith.cmpi eq, %tail, %four : i32
  %odd_bits = arith.andi %q, %eight : i32
  %odd = arith.cmpi ne, %odd_bits, %zero : i32
  %even_up = arith.andi %tie, %odd : i1
  %round = arith.ori %above, %even_up : i1
  %increment = arith.extui %round : i1 to i32
  %high = arith.shrui %q, %three : i32
  %rounded = arith.addi %high, %increment : i32
  %carry = arith.cmpi uge, %rounded, %double_hidden : i32
  %carried = arith.shrui %rounded, %one : i32
  %stored = arith.select %carry, %carried, %rounded : i32
  %carry_bit = arith.extui %carry : i1 to i32
  %final_exp = arith.addi %exponent, %carry_bit : i32
  %fraction = arith.andi %stored, %mask : i32
  %exp_bits = arith.shli %final_exp, %c23 : i32
  %magnitude = arith.ori %exp_bits, %fraction : i32
  %underflow = arith.cmpi slt, %final_exp, %one : i32
  %overflow = arith.cmpi sge, %final_exp, %c255 : i32
  %bounded = arith.select %overflow, %inf_bits, %magnitude : i32
  %finite = arith.select %underflow, %zero, %bounded : i32
  %signed = arith.ori %sign, %finite : i32
  scf.yield %signed : i32
}
%result = arith.bitcast %result_bits : i32 to f32
return %result : f32
}"""


def normalize_multiply_rne():
    # U16 partial products retain the complete 24x24 significand without an
    # Int64 capability. Guard/round/sticky packing is the shared reference's
    # strict F32 multiply, including tiny input normalization and zero sign.
    return """func.func private @normalize_multiply_rne(%a: f32, %b: f32) -> f32 {
%zero = arith.constant 0 : i32
%one = arith.constant 1 : i32
%three = arith.constant 3 : i32
%four = arith.constant 4 : i32
%seven = arith.constant 7 : i32
%eight = arith.constant 8 : i32
%c12 = arith.constant 12 : i32
%c16 = arith.constant 16 : i32
%c20 = arith.constant 20 : i32
%c23 = arith.constant 23 : i32
%c127 = arith.constant 127 : i32
%c255 = arith.constant 255 : i32
%abs_mask = arith.constant 2147483647 : i32
%sign_mask = arith.constant -2147483648 : i32
%mask = arith.constant 8388607 : i32
%low_mask = arith.constant 65535 : i32
%tail_mask = arith.constant 1048575 : i32
%hidden = arith.constant 8388608 : i32
%double_hidden = arith.constant 16777216 : i32
%extended_carry = arith.constant 134217728 : i32
%inf_bits = arith.constant 2139095040 : i32
%a_bits = arith.bitcast %a : f32 to i32
%b_bits = arith.bitcast %b : f32 to i32
%sign = arith.andi %a_bits, %sign_mask : i32
%aa = arith.andi %a_bits, %abs_mask : i32
%bb = arith.andi %b_bits, %abs_mask : i32
%az = arith.cmpi eq, %aa, %zero : i32
%an = arith.cmpi uge, %aa, %inf_bits : i32
%bn = arith.cmpi uge, %bb, %inf_bits : i32
%be = arith.shrui %bb, %c23 : i32
%bz = arith.cmpi eq, %be, %zero : i32
%invalid0 = arith.ori %az, %an : i1
%invalid1 = arith.ori %bn, %bz : i1
%invalid = arith.ori %invalid0, %invalid1 : i1
%result_bits = scf.if %invalid -> (i32) {
  scf.yield %sign : i32
} else {
  %ae0 = arith.shrui %aa, %c23 : i32
  %af = arith.andi %aa, %mask : i32
  %a_normal = arith.cmpi ne, %ae0, %zero : i32
  %a_hidden = arith.select %a_normal, %hidden, %zero : i32
  %as0 = arith.ori %af, %a_hidden : i32
  %ae1 = arith.select %a_normal, %ae0, %one : i32
  %parts:2 = scf.while (%sig = %as0, %exp = %ae1) : (i32, i32) -> (i32, i32) {
    %small = arith.cmpi ult, %sig, %hidden : i32
    scf.condition(%small) %sig, %exp : i32, i32
  } do {
    ^bb0(%sig: i32, %exp: i32):
    %shifted = arith.shli %sig, %one : i32
    %lower = arith.subi %exp, %one : i32
    scf.yield %shifted, %lower : i32, i32
  }
  %bf = arith.andi %bb, %mask : i32
  %bs = arith.ori %bf, %hidden : i32
  %a0 = arith.andi %parts#0, %low_mask : i32
  %a1 = arith.shrui %parts#0, %c16 : i32
  %b0 = arith.andi %bs, %low_mask : i32
  %b1 = arith.shrui %bs, %c16 : i32
  %p0 = arith.muli %a0, %b0 : i32
  %p1 = arith.muli %a1, %b0 : i32
  %p2 = arith.muli %a0, %b1 : i32
  %p3 = arith.muli %a1, %b1 : i32
  %p0_high = arith.shrui %p0, %c16 : i32
  %p1_low = arith.andi %p1, %low_mask : i32
  %p2_low = arith.andi %p2, %low_mask : i32
  %cross0 = arith.addi %p0_high, %p1_low : i32
  %cross = arith.addi %cross0, %p2_low : i32
  %cross_high = arith.shrui %cross, %c16 : i32
  %p1_high = arith.shrui %p1, %c16 : i32
  %p2_high = arith.shrui %p2, %c16 : i32
  %high0 = arith.addi %p3, %p1_high : i32
  %high1 = arith.addi %high0, %p2_high : i32
  %high = arith.addi %high1, %cross_high : i32
  %p0_low = arith.andi %p0, %low_mask : i32
  %cross_shifted = arith.shli %cross, %c16 : i32
  %low = arith.ori %p0_low, %cross_shifted : i32
  %head = arith.shli %high, %c12 : i32
  %body = arith.shrui %low, %c20 : i32
  %discarded = arith.andi %low, %tail_mask : i32
  %has_discarded = arith.cmpi ne, %discarded, %zero : i32
  %sticky = arith.extui %has_discarded : i1 to i32
  %head_body = arith.ori %head, %body : i32
  %extended = arith.ori %head_body, %sticky : i32
  %exp_sum = arith.addi %parts#1, %be : i32
  %base_exp = arith.subi %exp_sum, %c127 : i32
  %needs_shift = arith.cmpi uge, %extended, %extended_carry : i32
  %lsb = arith.andi %extended, %one : i32
  %right = arith.shrui %extended, %one : i32
  %jammed = arith.ori %right, %lsb : i32
  %q = arith.select %needs_shift, %jammed, %extended : i32
  %exp_extra = arith.extui %needs_shift : i1 to i32
  %exponent = arith.addi %base_exp, %exp_extra : i32
  %tail = arith.andi %q, %seven : i32
  %above = arith.cmpi ugt, %tail, %four : i32
  %tie = arith.cmpi eq, %tail, %four : i32
  %odd_bits = arith.andi %q, %eight : i32
  %odd = arith.cmpi ne, %odd_bits, %zero : i32
  %even_up = arith.andi %tie, %odd : i1
  %round = arith.ori %above, %even_up : i1
  %increment = arith.extui %round : i1 to i32
  %high_q = arith.shrui %q, %three : i32
  %rounded = arith.addi %high_q, %increment : i32
  %carry = arith.cmpi uge, %rounded, %double_hidden : i32
  %carried = arith.shrui %rounded, %one : i32
  %stored = arith.select %carry, %carried, %rounded : i32
  %carry_bit = arith.extui %carry : i1 to i32
  %final_exp = arith.addi %exponent, %carry_bit : i32
  %fraction = arith.andi %stored, %mask : i32
  %exp_bits = arith.shli %final_exp, %c23 : i32
  %magnitude = arith.ori %exp_bits, %fraction : i32
  %underflow = arith.cmpi slt, %final_exp, %one : i32
  %overflow = arith.cmpi sge, %final_exp, %c255 : i32
  %bounded = arith.select %overflow, %inf_bits, %magnitude : i32
  %finite = arith.select %underflow, %zero, %bounded : i32
  %signed = arith.ori %sign, %finite : i32
  scf.yield %signed : i32
}
%result = arith.bitcast %result_bits : i32 to f32
return %result : f32
}"""


def quantize(dtype, exact_output=False):
    t = 'f32' if dtype == 'f32' else 'i16'
    quant_name = f'quant_output_{dtype}' if exact_output else f'quant_{dtype}'
    lines = [f'func.func private @{quant_name}(%x: tensor<?x?x{t}>, %group_arg: i32, %floor: f32, %policy: i32) -> (tensor<?x?xi8>, tensor<?x?xf32>) {{',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
        '%group = arith.index_cast %group_arg : i32 to index',
        f'%m = tensor.dim %x, %c0 : tensor<?x?x{t}>', f'%k = tensor.dim %x, %c1 : tensor<?x?x{t}>',
        '%blocks = arith.divui %k, %group : index']
    if dtype == 'bf16':
        lines += ['%wide_empty = tensor.empty(%m, %k) : tensor<?x?xf32>',
            '%wide = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%x : tensor<?x?xi16>) outs(%wide_empty : tensor<?x?xf32>) {',
            '^bb0(%v: i16, %unused: f32):', *wide('%v', dtype, 'value')[0],
            'linalg.yield %value : f32', '} -> tensor<?x?xf32>']
    else: lines.append('%wide = tensor.cast %x : tensor<?x?xf32> to tensor<?x?xf32>')
    lines += ['%groups = tensor.expand_shape %wide [[0], [1, 2]] output_shape [%m, %blocks, %group] : tensor<?x?xf32> into tensor<?x?x?xf32>',
        '%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xf32>',
        '%init_max = linalg.fill ins(%floor : f32) outs(%empty_max : tensor<?x?xf32>) -> tensor<?x?xf32>',
        '%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xf32>) outs(%init_max : tensor<?x?xf32>) {',
        '^bb0(%v: f32, %old: f32):', '%abs = math.absf %v : f32', '%value = arith.maximumf %abs, %old : f32',
        'linalg.yield %value : f32', '} -> tensor<?x?xf32>',
        '%scales = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xf32>) outs(%empty_max : tensor<?x?xf32>) {',
        '^bb0(%amax: f32, %unused: f32):', '%maxcode = arith.constant 448.0 : f32',
        '%inverse_max = arith.constant 0.00223214296 : f32', '%normal = func.call @divide_448_rne(%amax) : (f32) -> f32',
        '%scaled = arith.mulf %amax, %inverse_max : f32', '%bits = arith.bitcast %scaled : f32 to i32',
        '%c23 = arith.constant 23 : i32', '%mask = arith.constant 8388607 : i32', '%zero = arith.constant 0 : i32',
        '%exponent = arith.shrui %bits, %c23 : i32', '%fraction = arith.andi %bits, %mask : i32',
        '%round = arith.cmpi ne, %fraction, %zero : i32', '%extra = arith.extui %round : i1 to i32',
        '%next = arith.addi %exponent, %extra : i32', '%power_bits = arith.shli %next, %c23 : i32',
        '%pow = arith.bitcast %power_bits : i32 to f32', '%one = arith.constant 1 : i32', '%two = arith.constant 2 : i32',
        '%use_pow = arith.cmpi eq, %policy, %one : i32', '%use_reciprocal = arith.cmpi eq, %policy, %two : i32',
        '%ordinary_or_reciprocal = arith.select %use_reciprocal, %scaled, %normal : f32',
        '%value = arith.select %use_pow, %pow, %ordinary_or_reciprocal : f32', 'linalg.yield %value : f32', '} -> tensor<?x?xf32>',
        '%reciprocal_policy = arith.constant 2 : i32', '%use_reciprocal = arith.cmpi eq, %policy, %reciprocal_policy : i32',
        '%normalizers = scf.if %use_reciprocal -> (tensor<?x?xf32>) {',
        '%empty_inverse = tensor.empty(%m, %blocks) : tensor<?x?xf32>',
        '%inverses = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%scales : tensor<?x?xf32>) outs(%empty_inverse : tensor<?x?xf32>) {',
        '^bb0(%scale: f32, %unused: f32):', '%inverse = func.call @reciprocal_normal_rne(%scale) : (f32) -> f32',
        'linalg.yield %inverse : f32', '} -> tensor<?x?xf32>',
        'scf.yield %inverses : tensor<?x?xf32>', '} else {', 'scf.yield %scales : tensor<?x?xf32>', '}',
        '%empty_codes = tensor.empty(%m, %k) : tensor<?x?xi8>',
        '%codes = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_codes : tensor<?x?xi8>) {',
        '^bb0(%unused: i8):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index',
        '%b = arith.divui %j, %group : index', '%v = tensor.extract %wide[%r, %j] : tensor<?x?xf32>',
        '%s = tensor.extract %normalizers[%r, %b] : tensor<?x?xf32>',
        '%normalized = scf.if %use_reciprocal -> (f32) {',
        # Do not let a Vulkan float input-denormal mode erase BF16/F32 tiny A
        # before multiplication by a large, normal reciprocal. The mantissa
        # conversion is exact; both boost operands/result are normal. Final
        # underflow is below E4's zero threshold, including its signed zero.
        '%input_bits = arith.bitcast %v : f32 to i32', '%abs_mask = arith.constant 2147483647 : i32',
        '%abs_bits = arith.andi %input_bits, %abs_mask : i32', '%zero_bits = arith.constant 0 : i32',
        '%normal_bit = arith.constant 8388608 : i32', '%nonzero = arith.cmpi ne, %abs_bits, %zero_bits : i32',
        '%small = arith.cmpi ult, %abs_bits, %normal_bit : i32', '%tiny = arith.andi %nonzero, %small : i1',
        '%product = scf.if %tiny -> (f32) {', '%mantissa = arith.uitofp %abs_bits : i32 to f32',
        '%boost_factor = arith.constant 2.350988701644575e-38 : f32',
        '%unboost_factor = arith.constant 5.960464477539063e-8 : f32',
        '%unsigned_boost = arith.mulf %mantissa, %boost_factor : f32',
        '%boost_bits = arith.bitcast %unsigned_boost : f32 to i32', '%sign_mask = arith.constant -2147483648 : i32',
        '%sign = arith.andi %input_bits, %sign_mask : i32', '%signed_boost_bits = arith.ori %boost_bits, %sign : i32',
        '%signed_boost = arith.bitcast %signed_boost_bits : i32 to f32',
        '%boosted_product = arith.mulf %signed_boost, %s : f32',
        '%unboosted_product = arith.mulf %boosted_product, %unboost_factor : f32',
        'scf.yield %unboosted_product : f32', '} else {', '%ordinary_product = arith.mulf %v, %s : f32',
        'scf.yield %ordinary_product : f32', '}',
        'scf.yield %product : f32', '} else {', '%quotient = arith.divf %v, %s : f32', 'scf.yield %quotient : f32', '}',
        '%lo = arith.constant -448.0 : f32', '%hi = arith.constant 448.0 : f32',
        '%lower = arith.maximumf %normalized, %lo : f32', '%clipped = arith.minimumf %lower, %hi : f32',
        '%bits = func.call @source_fp8_encode(%clipped) : (f32) -> i8',
        'linalg.yield %bits : i8', '} -> tensor<?x?xi8>', 'return %codes, %scales : tensor<?x?xi8>, tensor<?x?xf32>', '}']
    if exact_output:
        # Reuse group/floor/policy/scale traversal; only the observable byte
        # producer requests strict scalar normalization. Existing linear and
        # roundtrip exports retain their previous instruction sequence.
        first = lines.index('%normalized = scf.if %use_reciprocal -> (f32) {')
        last = lines.index('%lo = arith.constant -448.0 : f32')
        lines[first:last] = [
            '%normalized = scf.if %use_reciprocal -> (f32) {',
            '%product = func.call @normalize_multiply_rne(%v, %s) : (f32, f32) -> f32',
            'scf.yield %product : f32', '} else {',
            '%quotient = func.call @normalize_divide_rne(%v, %s) : (f32, f32) -> f32',
            'scf.yield %quotient : f32', '}']
        lines = [line.replace('%scaled = arith.mulf %amax, %inverse_max : f32',
            '%scaled = func.call @normalize_multiply_rne(%amax, %inverse_max) : (f32, f32) -> f32')
            for line in lines]
    return '\n'.join(lines)


def quantize_output(dtype, policy):
    t = 'f32' if dtype == 'f32' else 'i16'
    name = ('normal', 'power', 'reciprocal')[policy]
    return '\n'.join([f'func.func @quantize_{name}_{dtype}(%x: tensor<?x?x{t}>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {{',
        '%group = arith.constant 128 : i32', f'%policy = arith.constant {policy} : i32',
        f'%flag = func.call @finite_{dtype}(%x) : (tensor<?x?x{t}>) -> tensor<i32>',
        f'%encoded:2 = func.call @quant_output_{dtype}(%x, %group, %floor, %policy) : (tensor<?x?x{t}>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)',
        'return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>', '}'])


def weight_decode(dtype):
    t = 'f32' if dtype == 'f32' else 'i16'
    lines = [f'func.func @weight_decode_{dtype}(%codes: tensor<?x?xi8>, %scales: tensor<?x?xf32>) -> tensor<?x?x{t}> {{',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index', '%c128 = arith.constant 128 : index',
        '%n = tensor.dim %codes, %c0 : tensor<?x?xi8>', '%k = tensor.dim %codes, %c1 : tensor<?x?xi8>',
        f'%empty = tensor.empty(%n, %k) : tensor<?x?x{t}>',
        f'%out = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} outs(%empty : tensor<?x?x{t}>) {{',
        f'^bb0(%unused: {t}):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index',
        '%rr = arith.divui %r, %c128 : index', '%jj = arith.divui %j, %c128 : index',
        '%byte = tensor.extract %codes[%r, %j] : tensor<?x?xi8>', '%code = arith.extui %byte : i8 to i32',
        *decode_value('%code', 'decoded'), '%scale = tensor.extract %scales[%rr, %jj] : tensor<?x?xf32>',
        '%value = arith.mulf %decoded, %scale : f32']
    conversions, value = narrow('%value', dtype, 'stored')
    lines += conversions + [f'linalg.yield {value} : {t}', f'}} -> tensor<?x?x{t}>',
        f'return %out : tensor<?x?x{t}>', '}']
    return '\n'.join(lines)


def fp8_roundtrip(group):
    return '\n'.join([f'func.func @fp8_roundtrip_{group}(%x: tensor<?x?xi16>) -> (tensor<?x?xi16>, tensor<i32>) {{',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index', f'%g = arith.constant {group} : i32',
        '%gi = arith.index_cast %g : i32 to index', '%floor = arith.constant 0.0001 : f32', '%power = arith.constant 1 : i32',
        '%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>',
        '%q:2 = func.call @quant_bf16(%x, %g, %floor, %power) : (tensor<?x?xi16>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)',
        '%m = tensor.dim %x, %c0 : tensor<?x?xi16>', '%k = tensor.dim %x, %c1 : tensor<?x?xi16>',
        '%empty = tensor.empty(%m, %k) : tensor<?x?xi16>',
        '%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {',
        '^bb0(%unused: i16):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index', '%b = arith.divui %j, %gi : index',
        '%byte = tensor.extract %q#0[%r, %j] : tensor<?x?xi8>', '%code = arith.extui %byte : i8 to i32',
        *decode_value('%code', 'decoded'), '%scale = tensor.extract %q#1[%r, %b] : tensor<?x?xf32>',
        '%value = arith.mulf %decoded, %scale : f32', *narrow('%value', 'bf16', 'stored')[0],
        'linalg.yield %stored : i16', '} -> tensor<?x?xi16>',
        'return %out, %flag : tensor<?x?xi16>, tensor<i32>', '}'])


def fp4_roundtrip():
    lines = ['func.func @fp4_roundtrip_32(%x: tensor<?x?xi16>) -> (tensor<?x?xi16>, tensor<i32>) {',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index', '%g = arith.constant 32 : index',
        '%m = tensor.dim %x, %c0 : tensor<?x?xi16>', '%k = tensor.dim %x, %c1 : tensor<?x?xi16>',
        '%blocks = arith.divui %k, %g : index', '%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>',
        '%groups = tensor.expand_shape %x [[0], [1, 2]] output_shape [%m, %blocks, %g] : tensor<?x?xi16> into tensor<?x?x?xi16>',
        '%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xi32>', '%floor = arith.constant 448 : i32',
        '%init_max = linalg.fill ins(%floor : i32) outs(%empty_max : tensor<?x?xi32>) -> tensor<?x?xi32>',
        '%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xi16>) outs(%init_max : tensor<?x?xi32>) {',
        '^bb0(%v: i16, %old: i32):', '%bits = arith.extui %v : i16 to i32', '%mask = arith.constant 32767 : i32',
        '%mag = arith.andi %bits, %mask : i32', '%value = arith.maxui %mag, %old : i32',
        'linalg.yield %value : i32', '} -> tensor<?x?xi32>',
        '%empty_exp = tensor.empty(%m, %blocks) : tensor<?x?xi32>',
        '%exponents = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xi32>) outs(%empty_exp : tensor<?x?xi32>) {',
        '^bb0(%bf: i32, %unused: i32):', '%c16 = arith.constant 16 : i32', '%c23 = arith.constant 23 : i32',
        '%bits = arith.shli %bf, %c16 : i32', '%amax = arith.bitcast %bits : i32 to f32',
        '%six = arith.constant 6.0 : f32', '%s = arith.divf %amax, %six : f32',
        '%sb = arith.bitcast %s : f32 to i32', '%mask = arith.constant 8388607 : i32', '%zero = arith.constant 0 : i32',
        '%base = arith.shrui %sb, %c23 : i32', '%frac = arith.andi %sb, %mask : i32',
        '%round = arith.cmpi ne, %frac, %zero : i32', '%extra = arith.extui %round : i1 to i32',
        '%value = arith.addi %base, %extra : i32', 'linalg.yield %value : i32', '} -> tensor<?x?xi32>',
        '%table = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]> : tensor<8xf32>',
        '%empty = tensor.empty(%m, %k) : tensor<?x?xi16>',
        '%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {',
        '^bb0(%unused: i16):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index', '%block = arith.divui %j, %g : index',
        '%v = tensor.extract %x[%r, %j] : tensor<?x?xi16>', '%bits = arith.extui %v : i16 to i32',
        '%c0i = arith.constant 0 : i32', '%c1i = arith.constant 1 : i32', '%c2i = arith.constant 2 : i32',
        '%c4i = arith.constant 4 : i32', '%c7 = arith.constant 7 : i32', '%c16 = arith.constant 16 : i32', '%c23 = arith.constant 23 : i32',
        '%mask_mag = arith.constant 32767 : i32', '%mask_frac = arith.constant 127 : i32', '%mask_sign = arith.constant 32768 : i32',
        '%mag = arith.andi %bits, %mask_mag : i32', '%wide_bits = arith.shli %mag, %c16 : i32',
        '%wide = arith.bitcast %wide_bits : i32 to f32', '%exp = tensor.extract %exponents[%r, %block] : tensor<?x?xi32>',
        '%scale_bits = arith.shli %exp, %c23 : i32', '%scale = arith.bitcast %scale_bits : i32 to f32',
        '%ordinary = arith.divf %wide, %scale : f32',
        '%source_exp = arith.shrui %mag, %c7 : i32', '%fraction = arith.andi %mag, %mask_frac : i32',
        '%source_zero = arith.cmpi eq, %source_exp, %c0i : i32', '%leading = arith.constant 128 : i32',
        '%hidden = arith.select %source_zero, %c0i, %leading : i32', '%sig = arith.ori %fraction, %hidden : i32',
        '%upper = arith.maxui %source_exp, %c1i : i32', '%bounded = arith.minui %upper, %c4i : i32',
        '%shift = arith.subi %bounded, %c1i : i32', '%shifted = arith.shli %sig, %shift : i32',
        '%exact = arith.uitofp %shifted : i32 to f32', '%one = arith.cmpi eq, %exp, %c1i : i32',
        '%lofactor = arith.constant 0.0078125 : f32', '%hifactor = arith.constant 0.00390625 : f32',
        '%factor = arith.select %one, %lofactor, %hifactor : f32', '%low = arith.mulf %exact, %factor : f32',
        '%use_low = arith.cmpi ule, %exp, %c2i : i32', '%selected = arith.select %use_low, %low, %ordinary : f32',
        '%six = arith.constant 6.0 : f32', '%normalized = arith.minimumf %selected, %six : f32',
        '%best0 = arith.constant 0 : index', '%distance0 = tensor.extract %table[%best0] : tensor<8xf32>',
        '%diff0 = arith.subf %normalized, %distance0 : f32', '%d0 = math.absf %diff0 : f32']
    # Original eight-entry nearest-distance/even-code rule, no new quantizer.
    previous_best, previous_distance = '%best0', '%d0'
    for code in range(1, 8):
        lines += [f'%idx{code} = arith.constant {code} : index', f'%val{code} = tensor.extract %table[%idx{code}] : tensor<8xf32>',
            f'%diff{code} = arith.subf %normalized, %val{code} : f32', f'%d{code} = math.absf %diff{code} : f32',
            f'%closer{code} = arith.cmpf olt, %d{code}, {previous_distance} : f32']
        choose = f'%closer{code}'
        if code % 2 == 0:
            lines += [f'%tie{code} = arith.cmpf oeq, %d{code}, {previous_distance} : f32',
                f'%bi{code} = arith.index_cast {previous_best} : index to i32',
                f'%odd{code} = arith.andi %bi{code}, %c1i : i32',
                f'%wasodd{code} = arith.cmpi ne, %odd{code}, %c0i : i32',
                f'%even_tie{code} = arith.andi %tie{code}, %wasodd{code} : i1',
                f'%choose{code} = arith.ori %closer{code}, %even_tie{code} : i1']
            choose = f'%choose{code}'
        lines += [f'%best{code} = arith.select {choose}, %idx{code}, {previous_best} : index',
            f'%distance{code} = arith.select {choose}, %d{code}, {previous_distance} : f32']
        previous_best, previous_distance = f'%best{code}', f'%distance{code}'
    lines += [f'%magnitude = tensor.extract %table[{previous_best}] : tensor<8xf32>',
        '%dequant = arith.mulf %magnitude, %scale : f32', '%out_bits = arith.bitcast %dequant : f32 to i32',
        '%best = arith.index_cast %best7 : index to i32', '%half = arith.cmpi eq, %best, %c1i : i32',
        '%lowhalf = arith.andi %one, %half : i1', '%half_bits = arith.constant 4194304 : i32',
        '%fixed = arith.select %lowhalf, %half_bits, %out_bits : i32',
        '%is_zero = arith.cmpi eq, %best, %c0i : i32', '%zeroed = arith.select %is_zero, %c0i, %fixed : i32',
        '%sign = arith.andi %bits, %mask_sign : i32', '%sign32 = arith.shli %sign, %c16 : i32',
        '%signed = arith.ori %zeroed, %sign32 : i32', '%hi = arith.shrui %signed, %c16 : i32',
        '%stored = arith.trunci %hi : i32 to i16', 'linalg.yield %stored : i16',
        '} -> tensor<?x?xi16>', 'return %out, %flag : tensor<?x?xi16>, tensor<i32>', '}']
    return '\n'.join(lines)


def linear(dtype, grouped=False, four=False, policy=0, encoded=False):
    t = 'f32' if dtype == 'f32' else 'i16'
    name = 'encoded_block128' if encoded else f'{"grouped4" if four else "grouped8_reciprocal" if grouped and policy == 2 else "grouped8_power" if grouped and policy == 1 else "grouped8" if grouped else "linear_reciprocal" if policy == 2 else "linear_power" if policy == 1 else "linear_normal"}_{dtype}'
    out = f'tensor<?x?x{t}>'
    # Grouped banks are word-addressed before the expert offset is formed.
    # Keep the byte payload/scale ABI; four FP8 or eight FP4 codes share a word.
    wc_type = 'tensor<?x?x?xi32>' if grouped else 'tensor<?x?x?xi8>'
    args = [f'%input: tensor<?x?x{t}>', f'%wc: {wc_type}', '%ws: tensor<?x?x?xf32>']
    if grouped: args += ['%ids: tensor<?xi32>', '%slots_arg: i32', '%activations_arg: i32']
    if encoded:
        args = ['%input: tensor<?x?xi8>', '%as: tensor<?x?xf32>', '%wc: tensor<?x?x?xi8>', '%ws: tensor<?x?x?xf32>']
    else: args += ['%floor: f32']
    returns = out if encoded else f'({out}, tensor<i32>)'
    a_type = 'tensor<?x?xi8>' if encoded else f'tensor<?x?x{t}>'
    lines = [f'func.func @{name}({", ".join(args)}) -> {returns} {{',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index', '%c2 = arith.constant 2 : index',
        '%c128 = arith.constant 128 : index', f'%group = arith.constant {32 if four else 128} : index',
        f'%g32 = arith.constant 128 : i32', f'%k = tensor.dim %input, %c1 : {a_type}',
        f'%n = tensor.dim %wc, %c1 : {wc_type}']
    if grouped:
        lines += ['%r = tensor.dim %ids, %c0 : tensor<?xi32>',
            '%slots = arith.index_cast %slots_arg : i32 to index',
            '%activations = arith.index_cast %activations_arg : i32 to index']
    else: lines += [f'%r = tensor.dim %input, %c0 : {a_type}']
    lines += ['%blocks = arith.divui %k, %group : index']
    if not encoded:
        lines += [f'%flag = func.call @finite_{dtype}(%input) : ({a_type}) -> tensor<i32>',
            f'%policy = arith.constant {1 if four else policy} : i32',
            f'%q:2 = func.call @quant_{dtype}(%input, %g32, %floor, %policy) : ({a_type}, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)']
        ac, asc = '%q#0', '%q#1'
    else: ac, asc = '%input', '%as'
    lines += ['%zero = arith.constant 0.0 : f32', '%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>',
        '%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>']
    if four: lines += ['%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>']
    # Dense tokens share one decoded [N,K-group] matrix. Grouped rows gather
    # and decode packed words inside the native F32 reduction, so no selected
    # [R,N,K-group] weight tensor is materialized for repeated expert IDs.
    block_type = 'tensor<?x?x?xf32>' if grouped else 'tensor<?x?xf32>'
    block_shape = '%r, %c1, %group' if grouped else '%r, %group'
    identity = 'affine_map<(d0,d1,d2)->(d0,d1,d2)>' if grouped else 'affine_map<(d0,d1)->(d0,d1)>'
    iterators = '"parallel","parallel","parallel"' if grouped else '"parallel","parallel"'
    lines += ['%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {',
        '%begin = arith.muli %block, %group : index', f'%empty_a = tensor.empty({block_shape}) : {block_type}',
        f'%ax = linalg.generic {{indexing_maps = [{identity}], iterator_types = [{iterators}]}} outs(%empty_a : {block_type}) {{',
        '^bb0(%unused_a: f32):', '%row = linalg.index 0 : index', f'%j = linalg.index {2 if grouped else 1} : index',
        '%column = arith.addi %begin, %j : index']
    activation_row = '%row'
    if grouped:
        lines += ['%token = arith.divui %row, %slots : index', '%selected_slot = arith.remui %row, %slots : index',
            '%shared = arith.cmpi eq, %activations, %c1 : index', '%slot = arith.select %shared, %c0, %selected_slot : index',
            '%base = arith.muli %token, %activations : index', '%activation_row = arith.addi %base, %slot : index']
        activation_row = '%activation_row'
    lines += [f'%byte = tensor.extract {ac}[{activation_row}, %column] : tensor<?x?xi8>',
        '%code = arith.extui %byte : i8 to i32', *decode_value('%code', 'value'),
        'linalg.yield %value : f32', f'}} -> {block_type}']
    if grouped:
        lines += [f'%abar = util.optimization_barrier %ax : {block_type}',
            f'%ab = flow.tensor.tie_shape %abar : {block_type}{{{block_shape}}}',
            '%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>',
            '%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>',
            '%dot = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,0,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%ab : tensor<?x?x?xf32>) outs(%init_dot : tensor<?x?xf32>) {',
            '^bb0(%a_value: f32, %acc: f32):']
    else:
        lines += [f'%empty_w = tensor.empty(%n, %group) : {block_type}',
            f'%wx = linalg.generic {{indexing_maps = [{identity}], iterator_types = [{iterators}]}} outs(%empty_w : {block_type}) {{',
            '^bb0(%unused_w: f32):']
    if grouped: lines += ['%row = linalg.index 0 : index']
    lines += [f'%col = linalg.index {1 if grouped else 0} : index', f'%j = linalg.index {2 if grouped else 1} : index',
              '%column = arith.addi %begin, %j : index']
    if grouped: lines += ['%id = tensor.extract %ids[%row] : tensor<?xi32>', '%expert = arith.index_cast %id : i32 to index']
    else: lines += ['%expert = arith.constant 0 : index']
    if grouped:
        lines += [f'%pack = arith.constant {8 if four else 4} : index',
            '%packed_col = arith.divui %column, %pack : index',
            '%lane = arith.remui %column, %pack : index',
            f'%word = tensor.extract %wc[%expert, %col, %packed_col] : {wc_type}',
            '%lane32 = arith.index_cast %lane : index to i32',
            f'%bits = arith.constant {4 if four else 8} : i32',
            f'%mask = arith.constant {15 if four else 255} : i32',
            '%shift = arith.muli %lane32, %bits : i32',
            '%shifted = arith.shrui %word, %shift : i32',
            '%code = arith.andi %shifted, %mask : i32']
        if four:
            lines += ['%idx = arith.index_castui %code : i32 to index',
                '%value = tensor.extract %fp4[%idx] : tensor<16xf32>']
        else:
            lines += decode_value('%code', 'value')
    elif four:
        lines += ['%packed_col = arith.divui %column, %c2 : index', '%lane = arith.remui %column, %c2 : index',
            '%byte = tensor.extract %wc[%expert, %col, %packed_col] : tensor<?x?x?xi8>',
            '%word = arith.extui %byte : i8 to i32', '%lane32 = arith.index_cast %lane : index to i32',
            '%four = arith.constant 4 : i32', '%mask = arith.constant 15 : i32',
            '%shift = arith.muli %lane32, %four : i32', '%shifted = arith.shrui %word, %shift : i32',
            '%code = arith.andi %shifted, %mask : i32', '%idx = arith.index_castui %code : i32 to index',
            '%value = tensor.extract %fp4[%idx] : tensor<16xf32>']
    else:
        lines += ['%byte = tensor.extract %wc[%expert, %col, %column] : tensor<?x?x?xi8>',
            '%code = arith.extui %byte : i8 to i32', *decode_value('%code', 'value')]
    if grouped:
        lines += ['%product = arith.mulf %a_value, %value : f32',
            '%accumulated = arith.addf %acc, %product : f32',
            'linalg.yield %accumulated : f32', '} -> tensor<?x?xf32>']
    else: lines += ['linalg.yield %value : f32', f'}} -> {block_type}',
        f'%abar = util.optimization_barrier %ax : {block_type}', f'%wbar = util.optimization_barrier %wx : {block_type}',
        f'%ab = flow.tensor.tie_shape %abar : {block_type}{{{block_shape}}}',
        f'%wb = flow.tensor.tie_shape %wbar : {block_type}{{%n, %group}}',
        '%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>',
        '%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>',
        '%dot = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>] ins(%ab, %wb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_dot : tensor<?x?xf32>) -> tensor<?x?xf32>']
    lines += [
        '%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {',
        '^bb0(%unused_s: f32):', '%row = linalg.index 0 : index', '%col = linalg.index 1 : index']
    activation_row = '%row'
    if grouped:
        lines += ['%token = arith.divui %row, %slots : index', '%selected_slot = arith.remui %row, %slots : index',
            '%shared = arith.cmpi eq, %activations, %c1 : index', '%slot = arith.select %shared, %c0, %selected_slot : index',
            '%base = arith.muli %token, %activations : index', '%activation_row = arith.addi %base, %slot : index',
            '%id = tensor.extract %ids[%row] : tensor<?xi32>', '%expert = arith.index_cast %id : i32 to index']
        activation_row = '%activation_row'
    else: lines += ['%expert = arith.constant 0 : index']
    if four:
        lines += ['%c4 = arith.constant 4 : index', '%activation_block = arith.divui %block, %c4 : index']
        weight_column, activation_block = '%col', '%activation_block'
    else:
        lines += ['%weight_column = arith.divui %col, %c128 : index']
        weight_column, activation_block = '%weight_column', '%block'
    lines += [f'%a_scale = tensor.extract {asc}[{activation_row}, {activation_block}] : tensor<?x?xf32>',
        f'%w_scale = tensor.extract %ws[%expert, {weight_column}, %block] : tensor<?x?x?xf32>',
        '%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>',
        '%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>',
        '%activation_scaled = arith.mulf %d, %a_scale : f32', '%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32',
        '%value = arith.addf %old, %weight_scaled : f32', 'linalg.yield %value : f32', '} -> tensor<?x?xf32>',
        'scf.yield %next : tensor<?x?xf32>', '}', f'%out_empty = tensor.empty(%r, %n) : {out}',
        f'%out = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} ins(%total : tensor<?x?xf32>) outs(%out_empty : {out}) {{',
        f'^bb0(%value: f32, %unused: {t}):']
    casts, result = narrow('%value', dtype, 'stored')
    lines += casts + [f'linalg.yield {result} : {t}', f'}} -> {out}']
    lines += [f'return %out : {out}' if encoded else f'return %out, %flag : {out}, tensor<i32>', '}']
    return '\n'.join(lines)


def generate(path):
    entries = [finite('f32'), finite('bf16'), source_fp8_encode(), divide_448_rne(), reciprocal_normal_rne(), quantize('f32'), quantize('bf16'),
               weight_decode('f32'), weight_decode('bf16'), fp8_roundtrip(64), fp8_roundtrip(128), fp4_roundtrip()]
    # BF16 input/output cross the already qualified native Cast boundaries;
    # retire unqualified fused BF16 linear variants rather than keep two tracks.
    entries += [linear('f32', policy=p) for p in (0, 1)]
    entries += [linear('f32', grouped=True), linear('f32', grouped=True, four=True), linear('f32', encoded=True)]
    entries += [linear('f32', grouped=True, policy=1)]
    entries += [linear('f32', policy=2), linear('f32', grouped=True, policy=2)]
    entries += [normalize_divide_rne(), normalize_multiply_rne(),
                quantize('f32', exact_output=True), quantize('bf16', exact_output=True)]
    entries += [quantize_output(dtype, policy) for dtype in ('f32', 'bf16') for policy in (0, 1, 2)]
    path.write_text('module {\n' + '\n'.join(entries) + '\n}\n', encoding='utf-8', newline='\n')


if __name__ == '__main__': generate(Path(__file__).with_name('official.mlir'))
