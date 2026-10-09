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


def quantize(dtype):
    t = 'f32' if dtype == 'f32' else 'i16'
    lines = [f'func.func private @quant_{dtype}(%x: tensor<?x?x{t}>, %group_arg: i32, %floor: f32, %power: i32) -> (tensor<?x?xi8>, tensor<?x?xf32>) {{',
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
        '%inverse_max = arith.constant 0.00223214296 : f32', '%normal = arith.divf %amax, %maxcode : f32',
        '%scaled = arith.mulf %amax, %inverse_max : f32', '%bits = arith.bitcast %scaled : f32 to i32',
        '%c23 = arith.constant 23 : i32', '%mask = arith.constant 8388607 : i32', '%zero = arith.constant 0 : i32',
        '%exponent = arith.shrui %bits, %c23 : i32', '%fraction = arith.andi %bits, %mask : i32',
        '%round = arith.cmpi ne, %fraction, %zero : i32', '%extra = arith.extui %round : i1 to i32',
        '%next = arith.addi %exponent, %extra : i32', '%power_bits = arith.shli %next, %c23 : i32',
        '%pow = arith.bitcast %power_bits : i32 to f32', '%use_pow = arith.cmpi ne, %power, %zero : i32',
        '%value = arith.select %use_pow, %pow, %normal : f32', 'linalg.yield %value : f32', '} -> tensor<?x?xf32>',
        '%empty_codes = tensor.empty(%m, %k) : tensor<?x?xi8>',
        '%codes = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_codes : tensor<?x?xi8>) {',
        '^bb0(%unused: i8):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index',
        '%b = arith.divui %j, %group : index', '%v = tensor.extract %wide[%r, %j] : tensor<?x?xf32>',
        '%s = tensor.extract %scales[%r, %b] : tensor<?x?xf32>', '%normalized = arith.divf %v, %s : f32',
        '%lo = arith.constant -448.0 : f32', '%hi = arith.constant 448.0 : f32',
        '%lower = arith.maximumf %normalized, %lo : f32', '%clipped = arith.minimumf %lower, %hi : f32',
        '%converted = arith.truncf %clipped : f32 to f8E4M3FN', '%bits = arith.bitcast %converted : f8E4M3FN to i8',
        'linalg.yield %bits : i8', '} -> tensor<?x?xi8>', 'return %codes, %scales : tensor<?x?xi8>, tensor<?x?xf32>', '}']
    return '\n'.join(lines)


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


def linear(dtype, grouped=False, four=False, power=False, encoded=False):
    t = 'f32' if dtype == 'f32' else 'i16'
    name = 'encoded_block128' if encoded else f'{"grouped4" if four else "grouped8" if grouped else "linear_power" if power else "linear_normal"}_{dtype}'
    out = f'tensor<?x?x{t}>'
    args = [f'%input: tensor<?x?x{t}>', '%wc: tensor<?x?x?xi8>', '%ws: tensor<?x?x?xf32>']
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
        '%n = tensor.dim %wc, %c1 : tensor<?x?x?xi8>']
    if grouped:
        lines += ['%r = tensor.dim %ids, %c0 : tensor<?xi32>',
            '%slots = arith.index_cast %slots_arg : i32 to index',
            '%activations = arith.index_cast %activations_arg : i32 to index']
    else: lines += [f'%r = tensor.dim %input, %c0 : {a_type}']
    lines += ['%blocks = arith.divui %k, %group : index']
    if not encoded:
        lines += [f'%flag = func.call @finite_{dtype}(%input) : ({a_type}) -> tensor<i32>',
            f'%power = arith.constant {1 if power or four else 0} : i32',
            f'%q:2 = func.call @quant_{dtype}(%input, %g32, %floor, %power) : ({a_type}, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)']
        ac, asc = '%q#0', '%q#1'
    else: ac, asc = '%input', '%as'
    lines += ['%zero = arith.constant 0.0 : f32', '%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>',
        '%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>']
    if four: lines += ['%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>']
    lines += ['%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {',
        '%begin = arith.muli %block, %group : index', '%empty_a = tensor.empty(%r, %c1, %group) : tensor<?x?x?xf32>',
        '%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {',
        '^bb0(%unused_a: f32):', '%row = linalg.index 0 : index', '%j = linalg.index 2 : index',
        '%column = arith.addi %begin, %j : index']
    activation_row = '%row'
    if grouped:
        lines += ['%token = arith.divui %row, %slots : index', '%selected_slot = arith.remui %row, %slots : index',
            '%shared = arith.cmpi eq, %activations, %c1 : index', '%slot = arith.select %shared, %c0, %selected_slot : index',
            '%base = arith.muli %token, %activations : index', '%activation_row = arith.addi %base, %slot : index']
        activation_row = '%activation_row'
    lines += [f'%byte = tensor.extract {ac}[{activation_row}, %column] : tensor<?x?xi8>',
        '%code = arith.extui %byte : i8 to i32', *decode_value('%code', 'value'),
        'linalg.yield %value : f32', '} -> tensor<?x?x?xf32>',
        '%empty_w = tensor.empty(%r, %n, %group) : tensor<?x?x?xf32>',
        '%wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {',
        '^bb0(%unused_w: f32):', '%row = linalg.index 0 : index', '%col = linalg.index 1 : index', '%j = linalg.index 2 : index',
        '%column = arith.addi %begin, %j : index']
    if grouped: lines += ['%id = tensor.extract %ids[%row] : tensor<?xi32>', '%expert = arith.index_cast %id : i32 to index']
    else: lines += ['%expert = arith.constant 0 : index']
    if four:
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
    lines += ['linalg.yield %value : f32', '} -> tensor<?x?x?xf32>',
        '%abar = util.optimization_barrier %ax : tensor<?x?x?xf32>', '%wbar = util.optimization_barrier %wx : tensor<?x?x?xf32>',
        '%ab = flow.tensor.tie_shape %abar : tensor<?x?x?xf32>{%r, %c1, %group}',
        '%wb = flow.tensor.tie_shape %wbar : tensor<?x?x?xf32>{%r, %n, %group}',
        '%empty_dot = tensor.empty(%r, %c1, %n) : tensor<?x?x?xf32>',
        '%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>',
        '%dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%init_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>',
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
        '%d = tensor.extract %dot[%row, %c0, %col] : tensor<?x?x?xf32>',
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
    entries = [finite('f32'), finite('bf16'), quantize('f32'), quantize('bf16'),
               weight_decode('f32'), weight_decode('bf16'), fp8_roundtrip(64), fp8_roundtrip(128), fp4_roundtrip()]
    entries += [linear(t, power=p) for t in ('f32', 'bf16') for p in (False, True)]
    entries += [linear(t, grouped=True) for t in ('f32', 'bf16')]
    entries += [linear('bf16', grouped=True, four=True), linear('f32', encoded=True)]
    path.write_text('module {\n' + '\n'.join(entries) + '\n}\n', encoding='utf-8', newline='\n')


if __name__ == '__main__': generate(Path(__file__).with_name('official.mlir'))
