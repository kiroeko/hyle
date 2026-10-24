module {
func.func private @finite_f32(%x: tensor<?x?xf32>) -> tensor<i32> {
%zero = arith.constant 0 : i32
%empty = tensor.empty() : tensor<i32>
%init = linalg.fill ins(%zero : i32) outs(%empty : tensor<i32>) -> tensor<i32>
%flag = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>], iterator_types = ["reduction","reduction"]}
ins(%x : tensor<?x?xf32>) outs(%init : tensor<i32>) {
^bb0(%v: f32, %old: i32):
%bits = arith.bitcast %v : f32 to i32
%mask = arith.constant 2139095040 : i32
%exp = arith.andi %bits, %mask : i32
%invalid = arith.cmpi eq, %exp, %mask : i32
%flag_bit = arith.extui %invalid : i1 to i32
%value = arith.ori %old, %flag_bit : i32
linalg.yield %value : i32
} -> tensor<i32>
return %flag : tensor<i32>
}
func.func private @finite_bf16(%x: tensor<?x?xi16>) -> tensor<i32> {
%zero = arith.constant 0 : i32
%empty = tensor.empty() : tensor<i32>
%init = linalg.fill ins(%zero : i32) outs(%empty : tensor<i32>) -> tensor<i32>
%flag = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>], iterator_types = ["reduction","reduction"]}
ins(%x : tensor<?x?xi16>) outs(%init : tensor<i32>) {
^bb0(%v: i16, %old: i32):
%bits = arith.extui %v : i16 to i32
%mask = arith.constant 32640 : i32
%exp = arith.andi %bits, %mask : i32
%invalid = arith.cmpi eq, %exp, %mask : i32
%flag_bit = arith.extui %invalid : i1 to i32
%value = arith.ori %old, %flag_bit : i32
linalg.yield %value : i32
} -> tensor<i32>
return %flag : tensor<i32>
}
func.func private @source_fp8_encode(%input: f32) -> i8 {
%table = arith.constant dense<[0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0]> : tensor<127xf32>
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c126 = arith.constant 126 : index
%cap = arith.constant 448.0 : f32
%abs = math.absf %input : f32
%ax = arith.minimumf %abs, %cap : f32
%bounds:2 = scf.while (%lo = %c0, %hi = %c126) : (index, index) -> (index, index) {
%condition = arith.cmpi ult, %lo, %hi : index
scf.condition(%condition) %lo, %hi : index, index
} do {
^bb0(%lo: index, %hi: index):
%sum = arith.addi %lo, %hi : index
%raised = arith.addi %sum, %c1 : index
%mid = arith.divui %raised, %c2 : index
%value = tensor.extract %table[%mid] : tensor<127xf32>
%lower = arith.cmpf ole, %value, %ax : f32
%before = arith.subi %mid, %c1 : index
%new_lo = arith.select %lower, %mid, %lo : index
%new_hi = arith.select %lower, %hi, %before : index
scf.yield %new_lo, %new_hi : index, index
}
%next0 = arith.addi %bounds#0, %c1 : index
%next = arith.minui %next0, %c126 : index
%a = tensor.extract %table[%bounds#0] : tensor<127xf32>
%b = tensor.extract %table[%next] : tensor<127xf32>
%ad = arith.subf %ax, %a : f32
%bd = arith.subf %ax, %b : f32
%da = math.absf %ad : f32
%db = math.absf %bd : f32
%closer = arith.cmpf olt, %db, %da : f32
%tie = arith.cmpf oeq, %db, %da : f32
%best32 = arith.index_cast %bounds#0 : index to i32
%next32 = arith.index_cast %next : index to i32
%one = arith.constant 1 : i32
%zero = arith.constant 0 : i32
%bestodd = arith.andi %best32, %one : i32
%nextodd = arith.andi %next32, %one : i32
%odd = arith.cmpi ne, %bestodd, %zero : i32
%even = arith.cmpi eq, %nextodd, %zero : i32
%even_tie = arith.andi %tie, %even : i1
%choose_tie = arith.andi %even_tie, %odd : i1
%choose = arith.ori %closer, %choose_tie : i1
%selected = arith.select %choose, %next32, %best32 : i32
%bits = arith.bitcast %input : f32 to i32
%sign_shift = arith.constant 24 : i32
%sign_mask = arith.constant 128 : i32
%shifted = arith.shrui %bits, %sign_shift : i32
%sign = arith.andi %shifted, %sign_mask : i32
%signed = arith.ori %selected, %sign : i32
%result = arith.trunci %signed : i32 to i8
return %result : i8
}
func.func private @divide_448_rne(%x: f32) -> f32 {
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
}
func.func private @reciprocal_normal_rne(%x: f32) -> f32 {
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
}
func.func private @quant_f32(%x: tensor<?x?xf32>, %group_arg: i32, %floor: f32, %policy: i32) -> (tensor<?x?xi8>, tensor<?x?xf32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%group = arith.index_cast %group_arg : i32 to index
%m = tensor.dim %x, %c0 : tensor<?x?xf32>
%k = tensor.dim %x, %c1 : tensor<?x?xf32>
%blocks = arith.divui %k, %group : index
%wide = tensor.cast %x : tensor<?x?xf32> to tensor<?x?xf32>
%groups = tensor.expand_shape %wide [[0], [1, 2]] output_shape [%m, %blocks, %group] : tensor<?x?xf32> into tensor<?x?x?xf32>
%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%init_max = linalg.fill ins(%floor : f32) outs(%empty_max : tensor<?x?xf32>) -> tensor<?x?xf32>
%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xf32>) outs(%init_max : tensor<?x?xf32>) {
^bb0(%v: f32, %old: f32):
%abs = math.absf %v : f32
%value = arith.maximumf %abs, %old : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%scales = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xf32>) outs(%empty_max : tensor<?x?xf32>) {
^bb0(%amax: f32, %unused: f32):
%maxcode = arith.constant 448.0 : f32
%inverse_max = arith.constant 0.00223214296 : f32
%normal = func.call @divide_448_rne(%amax) : (f32) -> f32
%scaled = arith.mulf %amax, %inverse_max : f32
%bits = arith.bitcast %scaled : f32 to i32
%c23 = arith.constant 23 : i32
%mask = arith.constant 8388607 : i32
%zero = arith.constant 0 : i32
%exponent = arith.shrui %bits, %c23 : i32
%fraction = arith.andi %bits, %mask : i32
%round = arith.cmpi ne, %fraction, %zero : i32
%extra = arith.extui %round : i1 to i32
%next = arith.addi %exponent, %extra : i32
%power_bits = arith.shli %next, %c23 : i32
%pow = arith.bitcast %power_bits : i32 to f32
%one = arith.constant 1 : i32
%two = arith.constant 2 : i32
%use_pow = arith.cmpi eq, %policy, %one : i32
%use_reciprocal = arith.cmpi eq, %policy, %two : i32
%ordinary_or_reciprocal = arith.select %use_reciprocal, %scaled, %normal : f32
%value = arith.select %use_pow, %pow, %ordinary_or_reciprocal : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%reciprocal_policy = arith.constant 2 : i32
%use_reciprocal = arith.cmpi eq, %policy, %reciprocal_policy : i32
%normalizers = scf.if %use_reciprocal -> (tensor<?x?xf32>) {
%empty_inverse = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%inverses = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%scales : tensor<?x?xf32>) outs(%empty_inverse : tensor<?x?xf32>) {
^bb0(%scale: f32, %unused: f32):
%inverse = func.call @reciprocal_normal_rne(%scale) : (f32) -> f32
linalg.yield %inverse : f32
} -> tensor<?x?xf32>
scf.yield %inverses : tensor<?x?xf32>
} else {
scf.yield %scales : tensor<?x?xf32>
}
%empty_codes = tensor.empty(%m, %k) : tensor<?x?xi8>
%codes = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_codes : tensor<?x?xi8>) {
^bb0(%unused: i8):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%b = arith.divui %j, %group : index
%v = tensor.extract %wide[%r, %j] : tensor<?x?xf32>
%s = tensor.extract %normalizers[%r, %b] : tensor<?x?xf32>
%normalized = scf.if %use_reciprocal -> (f32) {
%input_bits = arith.bitcast %v : f32 to i32
%abs_mask = arith.constant 2147483647 : i32
%abs_bits = arith.andi %input_bits, %abs_mask : i32
%zero_bits = arith.constant 0 : i32
%normal_bit = arith.constant 8388608 : i32
%nonzero = arith.cmpi ne, %abs_bits, %zero_bits : i32
%small = arith.cmpi ult, %abs_bits, %normal_bit : i32
%tiny = arith.andi %nonzero, %small : i1
%product = scf.if %tiny -> (f32) {
%mantissa = arith.uitofp %abs_bits : i32 to f32
%boost_factor = arith.constant 2.350988701644575e-38 : f32
%unboost_factor = arith.constant 5.960464477539063e-8 : f32
%unsigned_boost = arith.mulf %mantissa, %boost_factor : f32
%boost_bits = arith.bitcast %unsigned_boost : f32 to i32
%sign_mask = arith.constant -2147483648 : i32
%sign = arith.andi %input_bits, %sign_mask : i32
%signed_boost_bits = arith.ori %boost_bits, %sign : i32
%signed_boost = arith.bitcast %signed_boost_bits : i32 to f32
%boosted_product = arith.mulf %signed_boost, %s : f32
%unboosted_product = arith.mulf %boosted_product, %unboost_factor : f32
scf.yield %unboosted_product : f32
} else {
%ordinary_product = arith.mulf %v, %s : f32
scf.yield %ordinary_product : f32
}
scf.yield %product : f32
} else {
%quotient = arith.divf %v, %s : f32
scf.yield %quotient : f32
}
%lo = arith.constant -448.0 : f32
%hi = arith.constant 448.0 : f32
%lower = arith.maximumf %normalized, %lo : f32
%clipped = arith.minimumf %lower, %hi : f32
%bits = func.call @source_fp8_encode(%clipped) : (f32) -> i8
linalg.yield %bits : i8
} -> tensor<?x?xi8>
return %codes, %scales : tensor<?x?xi8>, tensor<?x?xf32>
}
func.func private @quant_bf16(%x: tensor<?x?xi16>, %group_arg: i32, %floor: f32, %policy: i32) -> (tensor<?x?xi8>, tensor<?x?xf32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%group = arith.index_cast %group_arg : i32 to index
%m = tensor.dim %x, %c0 : tensor<?x?xi16>
%k = tensor.dim %x, %c1 : tensor<?x?xi16>
%blocks = arith.divui %k, %group : index
%wide_empty = tensor.empty(%m, %k) : tensor<?x?xf32>
%wide = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%x : tensor<?x?xi16>) outs(%wide_empty : tensor<?x?xf32>) {
^bb0(%v: i16, %unused: f32):
%value_bf = arith.bitcast %v : i16 to bf16
%value = arith.extf %value_bf : bf16 to f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%groups = tensor.expand_shape %wide [[0], [1, 2]] output_shape [%m, %blocks, %group] : tensor<?x?xf32> into tensor<?x?x?xf32>
%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%init_max = linalg.fill ins(%floor : f32) outs(%empty_max : tensor<?x?xf32>) -> tensor<?x?xf32>
%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xf32>) outs(%init_max : tensor<?x?xf32>) {
^bb0(%v: f32, %old: f32):
%abs = math.absf %v : f32
%value = arith.maximumf %abs, %old : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%scales = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xf32>) outs(%empty_max : tensor<?x?xf32>) {
^bb0(%amax: f32, %unused: f32):
%maxcode = arith.constant 448.0 : f32
%inverse_max = arith.constant 0.00223214296 : f32
%normal = func.call @divide_448_rne(%amax) : (f32) -> f32
%scaled = arith.mulf %amax, %inverse_max : f32
%bits = arith.bitcast %scaled : f32 to i32
%c23 = arith.constant 23 : i32
%mask = arith.constant 8388607 : i32
%zero = arith.constant 0 : i32
%exponent = arith.shrui %bits, %c23 : i32
%fraction = arith.andi %bits, %mask : i32
%round = arith.cmpi ne, %fraction, %zero : i32
%extra = arith.extui %round : i1 to i32
%next = arith.addi %exponent, %extra : i32
%power_bits = arith.shli %next, %c23 : i32
%pow = arith.bitcast %power_bits : i32 to f32
%one = arith.constant 1 : i32
%two = arith.constant 2 : i32
%use_pow = arith.cmpi eq, %policy, %one : i32
%use_reciprocal = arith.cmpi eq, %policy, %two : i32
%ordinary_or_reciprocal = arith.select %use_reciprocal, %scaled, %normal : f32
%value = arith.select %use_pow, %pow, %ordinary_or_reciprocal : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%reciprocal_policy = arith.constant 2 : i32
%use_reciprocal = arith.cmpi eq, %policy, %reciprocal_policy : i32
%normalizers = scf.if %use_reciprocal -> (tensor<?x?xf32>) {
%empty_inverse = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%inverses = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%scales : tensor<?x?xf32>) outs(%empty_inverse : tensor<?x?xf32>) {
^bb0(%scale: f32, %unused: f32):
%inverse = func.call @reciprocal_normal_rne(%scale) : (f32) -> f32
linalg.yield %inverse : f32
} -> tensor<?x?xf32>
scf.yield %inverses : tensor<?x?xf32>
} else {
scf.yield %scales : tensor<?x?xf32>
}
%empty_codes = tensor.empty(%m, %k) : tensor<?x?xi8>
%codes = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_codes : tensor<?x?xi8>) {
^bb0(%unused: i8):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%b = arith.divui %j, %group : index
%v = tensor.extract %wide[%r, %j] : tensor<?x?xf32>
%s = tensor.extract %normalizers[%r, %b] : tensor<?x?xf32>
%normalized = scf.if %use_reciprocal -> (f32) {
%input_bits = arith.bitcast %v : f32 to i32
%abs_mask = arith.constant 2147483647 : i32
%abs_bits = arith.andi %input_bits, %abs_mask : i32
%zero_bits = arith.constant 0 : i32
%normal_bit = arith.constant 8388608 : i32
%nonzero = arith.cmpi ne, %abs_bits, %zero_bits : i32
%small = arith.cmpi ult, %abs_bits, %normal_bit : i32
%tiny = arith.andi %nonzero, %small : i1
%product = scf.if %tiny -> (f32) {
%mantissa = arith.uitofp %abs_bits : i32 to f32
%boost_factor = arith.constant 2.350988701644575e-38 : f32
%unboost_factor = arith.constant 5.960464477539063e-8 : f32
%unsigned_boost = arith.mulf %mantissa, %boost_factor : f32
%boost_bits = arith.bitcast %unsigned_boost : f32 to i32
%sign_mask = arith.constant -2147483648 : i32
%sign = arith.andi %input_bits, %sign_mask : i32
%signed_boost_bits = arith.ori %boost_bits, %sign : i32
%signed_boost = arith.bitcast %signed_boost_bits : i32 to f32
%boosted_product = arith.mulf %signed_boost, %s : f32
%unboosted_product = arith.mulf %boosted_product, %unboost_factor : f32
scf.yield %unboosted_product : f32
} else {
%ordinary_product = arith.mulf %v, %s : f32
scf.yield %ordinary_product : f32
}
scf.yield %product : f32
} else {
%quotient = arith.divf %v, %s : f32
scf.yield %quotient : f32
}
%lo = arith.constant -448.0 : f32
%hi = arith.constant 448.0 : f32
%lower = arith.maximumf %normalized, %lo : f32
%clipped = arith.minimumf %lower, %hi : f32
%bits = func.call @source_fp8_encode(%clipped) : (f32) -> i8
linalg.yield %bits : i8
} -> tensor<?x?xi8>
return %codes, %scales : tensor<?x?xi8>, tensor<?x?xf32>
}
func.func @weight_decode_f32(%codes: tensor<?x?xi8>, %scales: tensor<?x?xf32>) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c128 = arith.constant 128 : index
%n = tensor.dim %codes, %c0 : tensor<?x?xi8>
%k = tensor.dim %codes, %c1 : tensor<?x?xi8>
%empty = tensor.empty(%n, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xf32>) {
^bb0(%unused: f32):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%rr = arith.divui %r, %c128 : index
%jj = arith.divui %j, %c128 : index
%byte = tensor.extract %codes[%r, %j] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%decoded_byte = arith.trunci %code : i32 to i8
%decoded_encoded = arith.bitcast %decoded_byte : i8 to f8E4M3FN
%decoded_original = arith.extf %decoded_encoded : f8E4M3FN to f32
%decoded_mask = arith.constant 127 : i32
%decoded_magnitude = arith.andi %code, %decoded_mask : i32
%decoded_is_nan = arith.cmpi eq, %decoded_magnitude, %decoded_mask : i32
%decoded_nan_bits = arith.constant 2143289344 : i32
%decoded_nan = arith.bitcast %decoded_nan_bits : i32 to f32
%decoded = arith.select %decoded_is_nan, %decoded_nan, %decoded_original : f32
%scale = tensor.extract %scales[%rr, %jj] : tensor<?x?xf32>
%value = arith.mulf %decoded, %scale : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @weight_decode_bf16(%codes: tensor<?x?xi8>, %scales: tensor<?x?xf32>) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c128 = arith.constant 128 : index
%n = tensor.dim %codes, %c0 : tensor<?x?xi8>
%k = tensor.dim %codes, %c1 : tensor<?x?xi8>
%empty = tensor.empty(%n, %k) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {
^bb0(%unused: i16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%rr = arith.divui %r, %c128 : index
%jj = arith.divui %j, %c128 : index
%byte = tensor.extract %codes[%r, %j] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%decoded_byte = arith.trunci %code : i32 to i8
%decoded_encoded = arith.bitcast %decoded_byte : i8 to f8E4M3FN
%decoded_original = arith.extf %decoded_encoded : f8E4M3FN to f32
%decoded_mask = arith.constant 127 : i32
%decoded_magnitude = arith.andi %code, %decoded_mask : i32
%decoded_is_nan = arith.cmpi eq, %decoded_magnitude, %decoded_mask : i32
%decoded_nan_bits = arith.constant 2143289344 : i32
%decoded_nan = arith.bitcast %decoded_nan_bits : i32 to f32
%decoded = arith.select %decoded_is_nan, %decoded_nan, %decoded_original : f32
%scale = tensor.extract %scales[%rr, %jj] : tensor<?x?xf32>
%value = arith.mulf %decoded, %scale : f32
%stored_float = arith.truncf %value : f32 to bf16
%stored = arith.bitcast %stored_float : bf16 to i16
linalg.yield %stored : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @fp8_roundtrip_64(%x: tensor<?x?xi16>) -> (tensor<?x?xi16>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%g = arith.constant 64 : i32
%gi = arith.index_cast %g : i32 to index
%floor = arith.constant 0.0001 : f32
%power = arith.constant 1 : i32
%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>
%q:2 = func.call @quant_bf16(%x, %g, %floor, %power) : (tensor<?x?xi16>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%m = tensor.dim %x, %c0 : tensor<?x?xi16>
%k = tensor.dim %x, %c1 : tensor<?x?xi16>
%empty = tensor.empty(%m, %k) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {
^bb0(%unused: i16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%b = arith.divui %j, %gi : index
%byte = tensor.extract %q#0[%r, %j] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%decoded_byte = arith.trunci %code : i32 to i8
%decoded_encoded = arith.bitcast %decoded_byte : i8 to f8E4M3FN
%decoded_original = arith.extf %decoded_encoded : f8E4M3FN to f32
%decoded_mask = arith.constant 127 : i32
%decoded_magnitude = arith.andi %code, %decoded_mask : i32
%decoded_is_nan = arith.cmpi eq, %decoded_magnitude, %decoded_mask : i32
%decoded_nan_bits = arith.constant 2143289344 : i32
%decoded_nan = arith.bitcast %decoded_nan_bits : i32 to f32
%decoded = arith.select %decoded_is_nan, %decoded_nan, %decoded_original : f32
%scale = tensor.extract %q#1[%r, %b] : tensor<?x?xf32>
%value = arith.mulf %decoded, %scale : f32
%stored_float = arith.truncf %value : f32 to bf16
%stored = arith.bitcast %stored_float : bf16 to i16
linalg.yield %stored : i16
} -> tensor<?x?xi16>
return %out, %flag : tensor<?x?xi16>, tensor<i32>
}
func.func @fp8_roundtrip_128(%x: tensor<?x?xi16>) -> (tensor<?x?xi16>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%g = arith.constant 128 : i32
%gi = arith.index_cast %g : i32 to index
%floor = arith.constant 0.0001 : f32
%power = arith.constant 1 : i32
%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>
%q:2 = func.call @quant_bf16(%x, %g, %floor, %power) : (tensor<?x?xi16>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%m = tensor.dim %x, %c0 : tensor<?x?xi16>
%k = tensor.dim %x, %c1 : tensor<?x?xi16>
%empty = tensor.empty(%m, %k) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {
^bb0(%unused: i16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%b = arith.divui %j, %gi : index
%byte = tensor.extract %q#0[%r, %j] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%decoded_byte = arith.trunci %code : i32 to i8
%decoded_encoded = arith.bitcast %decoded_byte : i8 to f8E4M3FN
%decoded_original = arith.extf %decoded_encoded : f8E4M3FN to f32
%decoded_mask = arith.constant 127 : i32
%decoded_magnitude = arith.andi %code, %decoded_mask : i32
%decoded_is_nan = arith.cmpi eq, %decoded_magnitude, %decoded_mask : i32
%decoded_nan_bits = arith.constant 2143289344 : i32
%decoded_nan = arith.bitcast %decoded_nan_bits : i32 to f32
%decoded = arith.select %decoded_is_nan, %decoded_nan, %decoded_original : f32
%scale = tensor.extract %q#1[%r, %b] : tensor<?x?xf32>
%value = arith.mulf %decoded, %scale : f32
%stored_float = arith.truncf %value : f32 to bf16
%stored = arith.bitcast %stored_float : bf16 to i16
linalg.yield %stored : i16
} -> tensor<?x?xi16>
return %out, %flag : tensor<?x?xi16>, tensor<i32>
}
func.func @fp4_roundtrip_32(%x: tensor<?x?xi16>) -> (tensor<?x?xi16>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%g = arith.constant 32 : index
%m = tensor.dim %x, %c0 : tensor<?x?xi16>
%k = tensor.dim %x, %c1 : tensor<?x?xi16>
%blocks = arith.divui %k, %g : index
%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>
%groups = tensor.expand_shape %x [[0], [1, 2]] output_shape [%m, %blocks, %g] : tensor<?x?xi16> into tensor<?x?x?xi16>
%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xi32>
%floor = arith.constant 448 : i32
%init_max = linalg.fill ins(%floor : i32) outs(%empty_max : tensor<?x?xi32>) -> tensor<?x?xi32>
%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xi16>) outs(%init_max : tensor<?x?xi32>) {
^bb0(%v: i16, %old: i32):
%bits = arith.extui %v : i16 to i32
%mask = arith.constant 32767 : i32
%mag = arith.andi %bits, %mask : i32
%value = arith.maxui %mag, %old : i32
linalg.yield %value : i32
} -> tensor<?x?xi32>
%empty_exp = tensor.empty(%m, %blocks) : tensor<?x?xi32>
%exponents = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xi32>) outs(%empty_exp : tensor<?x?xi32>) {
^bb0(%bf: i32, %unused: i32):
%c16 = arith.constant 16 : i32
%c23 = arith.constant 23 : i32
%bits = arith.shli %bf, %c16 : i32
%amax = arith.bitcast %bits : i32 to f32
%six = arith.constant 6.0 : f32
%s = arith.divf %amax, %six : f32
%sb = arith.bitcast %s : f32 to i32
%mask = arith.constant 8388607 : i32
%zero = arith.constant 0 : i32
%base = arith.shrui %sb, %c23 : i32
%frac = arith.andi %sb, %mask : i32
%round = arith.cmpi ne, %frac, %zero : i32
%extra = arith.extui %round : i1 to i32
%value = arith.addi %base, %extra : i32
linalg.yield %value : i32
} -> tensor<?x?xi32>
%table = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]> : tensor<8xf32>
%empty = tensor.empty(%m, %k) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {
^bb0(%unused: i16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%block = arith.divui %j, %g : index
%v = tensor.extract %x[%r, %j] : tensor<?x?xi16>
%bits = arith.extui %v : i16 to i32
%c0i = arith.constant 0 : i32
%c1i = arith.constant 1 : i32
%c2i = arith.constant 2 : i32
%c4i = arith.constant 4 : i32
%c7 = arith.constant 7 : i32
%c16 = arith.constant 16 : i32
%c23 = arith.constant 23 : i32
%mask_mag = arith.constant 32767 : i32
%mask_frac = arith.constant 127 : i32
%mask_sign = arith.constant 32768 : i32
%mag = arith.andi %bits, %mask_mag : i32
%wide_bits = arith.shli %mag, %c16 : i32
%wide = arith.bitcast %wide_bits : i32 to f32
%exp = tensor.extract %exponents[%r, %block] : tensor<?x?xi32>
%scale_bits = arith.shli %exp, %c23 : i32
%scale = arith.bitcast %scale_bits : i32 to f32
%ordinary = arith.divf %wide, %scale : f32
%source_exp = arith.shrui %mag, %c7 : i32
%fraction = arith.andi %mag, %mask_frac : i32
%source_zero = arith.cmpi eq, %source_exp, %c0i : i32
%leading = arith.constant 128 : i32
%hidden = arith.select %source_zero, %c0i, %leading : i32
%sig = arith.ori %fraction, %hidden : i32
%upper = arith.maxui %source_exp, %c1i : i32
%bounded = arith.minui %upper, %c4i : i32
%shift = arith.subi %bounded, %c1i : i32
%shifted = arith.shli %sig, %shift : i32
%exact = arith.uitofp %shifted : i32 to f32
%one = arith.cmpi eq, %exp, %c1i : i32
%lofactor = arith.constant 0.0078125 : f32
%hifactor = arith.constant 0.00390625 : f32
%factor = arith.select %one, %lofactor, %hifactor : f32
%low = arith.mulf %exact, %factor : f32
%use_low = arith.cmpi ule, %exp, %c2i : i32
%selected = arith.select %use_low, %low, %ordinary : f32
%six = arith.constant 6.0 : f32
%normalized = arith.minimumf %selected, %six : f32
%best0 = arith.constant 0 : index
%distance0 = tensor.extract %table[%best0] : tensor<8xf32>
%diff0 = arith.subf %normalized, %distance0 : f32
%d0 = math.absf %diff0 : f32
%idx1 = arith.constant 1 : index
%val1 = tensor.extract %table[%idx1] : tensor<8xf32>
%diff1 = arith.subf %normalized, %val1 : f32
%d1 = math.absf %diff1 : f32
%closer1 = arith.cmpf olt, %d1, %d0 : f32
%best1 = arith.select %closer1, %idx1, %best0 : index
%distance1 = arith.select %closer1, %d1, %d0 : f32
%idx2 = arith.constant 2 : index
%val2 = tensor.extract %table[%idx2] : tensor<8xf32>
%diff2 = arith.subf %normalized, %val2 : f32
%d2 = math.absf %diff2 : f32
%closer2 = arith.cmpf olt, %d2, %distance1 : f32
%tie2 = arith.cmpf oeq, %d2, %distance1 : f32
%bi2 = arith.index_cast %best1 : index to i32
%odd2 = arith.andi %bi2, %c1i : i32
%wasodd2 = arith.cmpi ne, %odd2, %c0i : i32
%even_tie2 = arith.andi %tie2, %wasodd2 : i1
%choose2 = arith.ori %closer2, %even_tie2 : i1
%best2 = arith.select %choose2, %idx2, %best1 : index
%distance2 = arith.select %choose2, %d2, %distance1 : f32
%idx3 = arith.constant 3 : index
%val3 = tensor.extract %table[%idx3] : tensor<8xf32>
%diff3 = arith.subf %normalized, %val3 : f32
%d3 = math.absf %diff3 : f32
%closer3 = arith.cmpf olt, %d3, %distance2 : f32
%best3 = arith.select %closer3, %idx3, %best2 : index
%distance3 = arith.select %closer3, %d3, %distance2 : f32
%idx4 = arith.constant 4 : index
%val4 = tensor.extract %table[%idx4] : tensor<8xf32>
%diff4 = arith.subf %normalized, %val4 : f32
%d4 = math.absf %diff4 : f32
%closer4 = arith.cmpf olt, %d4, %distance3 : f32
%tie4 = arith.cmpf oeq, %d4, %distance3 : f32
%bi4 = arith.index_cast %best3 : index to i32
%odd4 = arith.andi %bi4, %c1i : i32
%wasodd4 = arith.cmpi ne, %odd4, %c0i : i32
%even_tie4 = arith.andi %tie4, %wasodd4 : i1
%choose4 = arith.ori %closer4, %even_tie4 : i1
%best4 = arith.select %choose4, %idx4, %best3 : index
%distance4 = arith.select %choose4, %d4, %distance3 : f32
%idx5 = arith.constant 5 : index
%val5 = tensor.extract %table[%idx5] : tensor<8xf32>
%diff5 = arith.subf %normalized, %val5 : f32
%d5 = math.absf %diff5 : f32
%closer5 = arith.cmpf olt, %d5, %distance4 : f32
%best5 = arith.select %closer5, %idx5, %best4 : index
%distance5 = arith.select %closer5, %d5, %distance4 : f32
%idx6 = arith.constant 6 : index
%val6 = tensor.extract %table[%idx6] : tensor<8xf32>
%diff6 = arith.subf %normalized, %val6 : f32
%d6 = math.absf %diff6 : f32
%closer6 = arith.cmpf olt, %d6, %distance5 : f32
%tie6 = arith.cmpf oeq, %d6, %distance5 : f32
%bi6 = arith.index_cast %best5 : index to i32
%odd6 = arith.andi %bi6, %c1i : i32
%wasodd6 = arith.cmpi ne, %odd6, %c0i : i32
%even_tie6 = arith.andi %tie6, %wasodd6 : i1
%choose6 = arith.ori %closer6, %even_tie6 : i1
%best6 = arith.select %choose6, %idx6, %best5 : index
%distance6 = arith.select %choose6, %d6, %distance5 : f32
%idx7 = arith.constant 7 : index
%val7 = tensor.extract %table[%idx7] : tensor<8xf32>
%diff7 = arith.subf %normalized, %val7 : f32
%d7 = math.absf %diff7 : f32
%closer7 = arith.cmpf olt, %d7, %distance6 : f32
%best7 = arith.select %closer7, %idx7, %best6 : index
%distance7 = arith.select %closer7, %d7, %distance6 : f32
%magnitude = tensor.extract %table[%best7] : tensor<8xf32>
%dequant = arith.mulf %magnitude, %scale : f32
%out_bits = arith.bitcast %dequant : f32 to i32
%best = arith.index_cast %best7 : index to i32
%half = arith.cmpi eq, %best, %c1i : i32
%lowhalf = arith.andi %one, %half : i1
%half_bits = arith.constant 4194304 : i32
%fixed = arith.select %lowhalf, %half_bits, %out_bits : i32
%is_zero = arith.cmpi eq, %best, %c0i : i32
%zeroed = arith.select %is_zero, %c0i, %fixed : i32
%sign = arith.andi %bits, %mask_sign : i32
%sign32 = arith.shli %sign, %c16 : i32
%signed = arith.ori %zeroed, %sign32 : i32
%hi = arith.shrui %signed, %c16 : i32
%stored = arith.trunci %hi : i32 to i16
linalg.yield %stored : i16
} -> tensor<?x?xi16>
return %out, %flag : tensor<?x?xi16>, tensor<i32>
}
func.func @linear_normal_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi8>, %ws: tensor<?x?x?xf32>, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi8>
%r = tensor.dim %input, %c0 : tensor<?x?xf32>
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 0 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %group) : tensor<?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_a : tensor<?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%byte = tensor.extract %q#0[%row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%empty_w = tensor.empty(%n, %group) : tensor<?x?xf32>
%wx = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_w : tensor<?x?xf32>) {
^bb0(%unused_w: f32):
%col = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%expert = arith.constant 0 : index
%byte = tensor.extract %wc[%expert, %col, %column] : tensor<?x?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?xf32>
%wbar = util.optimization_barrier %wx : tensor<?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?xf32>{%r, %group}
%wb = flow.tensor.tie_shape %wbar : tensor<?x?xf32>{%n, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>] ins(%ab, %wb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%expert = arith.constant 0 : index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %q#1[%row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func @linear_power_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi8>, %ws: tensor<?x?x?xf32>, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi8>
%r = tensor.dim %input, %c0 : tensor<?x?xf32>
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 1 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %group) : tensor<?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_a : tensor<?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%byte = tensor.extract %q#0[%row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%empty_w = tensor.empty(%n, %group) : tensor<?x?xf32>
%wx = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_w : tensor<?x?xf32>) {
^bb0(%unused_w: f32):
%col = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%expert = arith.constant 0 : index
%byte = tensor.extract %wc[%expert, %col, %column] : tensor<?x?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?xf32>
%wbar = util.optimization_barrier %wx : tensor<?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?xf32>{%r, %group}
%wb = flow.tensor.tie_shape %wbar : tensor<?x?xf32>{%n, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>] ins(%ab, %wb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%expert = arith.constant 0 : index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %q#1[%row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func @grouped8_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi32>, %ws: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %slots_arg: i32, %activations_arg: i32, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi32>
%r = tensor.dim %ids, %c0 : tensor<?xi32>
%slots = arith.index_cast %slots_arg : i32 to index
%activations = arith.index_cast %activations_arg : i32 to index
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 0 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %c1, %group) : tensor<?x?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%byte = tensor.extract %q#0[%activation_row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?x?xf32>{%r, %c1, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,0,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%ab : tensor<?x?x?xf32>) outs(%init_dot : tensor<?x?xf32>) {
^bb0(%a_value: f32, %acc: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%pack = arith.constant 4 : index
%packed_col = arith.divui %column, %pack : index
%lane = arith.remui %column, %pack : index
%word = tensor.extract %wc[%expert, %col, %packed_col] : tensor<?x?x?xi32>
%lane32 = arith.index_cast %lane : index to i32
%bits = arith.constant 8 : i32
%mask = arith.constant 255 : i32
%shift = arith.muli %lane32, %bits : i32
%shifted = arith.shrui %word, %shift : i32
%code = arith.andi %shifted, %mask : i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
%product = arith.mulf %a_value, %value : f32
%accumulated = arith.addf %acc, %product : f32
linalg.yield %accumulated : f32
} -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %q#1[%activation_row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func @grouped4_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi32>, %ws: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %slots_arg: i32, %activations_arg: i32, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 32 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi32>
%r = tensor.dim %ids, %c0 : tensor<?xi32>
%slots = arith.index_cast %slots_arg : i32 to index
%activations = arith.index_cast %activations_arg : i32 to index
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 1 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %c1, %group) : tensor<?x?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%byte = tensor.extract %q#0[%activation_row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?x?xf32>{%r, %c1, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,0,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%ab : tensor<?x?x?xf32>) outs(%init_dot : tensor<?x?xf32>) {
^bb0(%a_value: f32, %acc: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%pack = arith.constant 8 : index
%packed_col = arith.divui %column, %pack : index
%lane = arith.remui %column, %pack : index
%word = tensor.extract %wc[%expert, %col, %packed_col] : tensor<?x?x?xi32>
%lane32 = arith.index_cast %lane : index to i32
%bits = arith.constant 4 : i32
%mask = arith.constant 15 : i32
%shift = arith.muli %lane32, %bits : i32
%shifted = arith.shrui %word, %shift : i32
%code = arith.andi %shifted, %mask : i32
%idx = arith.index_castui %code : i32 to index
%value = tensor.extract %fp4[%idx] : tensor<16xf32>
%product = arith.mulf %a_value, %value : f32
%accumulated = arith.addf %acc, %product : f32
linalg.yield %accumulated : f32
} -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%c4 = arith.constant 4 : index
%activation_block = arith.divui %block, %c4 : index
%a_scale = tensor.extract %q#1[%activation_row, %activation_block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %col, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func @encoded_block128(%input: tensor<?x?xi8>, %as: tensor<?x?xf32>, %wc: tensor<?x?x?xi8>, %ws: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xi8>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi8>
%r = tensor.dim %input, %c0 : tensor<?x?xi8>
%blocks = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %group) : tensor<?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_a : tensor<?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%byte = tensor.extract %input[%row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%empty_w = tensor.empty(%n, %group) : tensor<?x?xf32>
%wx = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_w : tensor<?x?xf32>) {
^bb0(%unused_w: f32):
%col = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%expert = arith.constant 0 : index
%byte = tensor.extract %wc[%expert, %col, %column] : tensor<?x?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?xf32>
%wbar = util.optimization_barrier %wx : tensor<?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?xf32>{%r, %group}
%wb = flow.tensor.tie_shape %wbar : tensor<?x?xf32>{%n, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>] ins(%ab, %wb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%expert = arith.constant 0 : index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %as[%row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @grouped8_power_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi32>, %ws: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %slots_arg: i32, %activations_arg: i32, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi32>
%r = tensor.dim %ids, %c0 : tensor<?xi32>
%slots = arith.index_cast %slots_arg : i32 to index
%activations = arith.index_cast %activations_arg : i32 to index
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 1 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %c1, %group) : tensor<?x?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%byte = tensor.extract %q#0[%activation_row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?x?xf32>{%r, %c1, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,0,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%ab : tensor<?x?x?xf32>) outs(%init_dot : tensor<?x?xf32>) {
^bb0(%a_value: f32, %acc: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%pack = arith.constant 4 : index
%packed_col = arith.divui %column, %pack : index
%lane = arith.remui %column, %pack : index
%word = tensor.extract %wc[%expert, %col, %packed_col] : tensor<?x?x?xi32>
%lane32 = arith.index_cast %lane : index to i32
%bits = arith.constant 8 : i32
%mask = arith.constant 255 : i32
%shift = arith.muli %lane32, %bits : i32
%shifted = arith.shrui %word, %shift : i32
%code = arith.andi %shifted, %mask : i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
%product = arith.mulf %a_value, %value : f32
%accumulated = arith.addf %acc, %product : f32
linalg.yield %accumulated : f32
} -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %q#1[%activation_row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func @linear_reciprocal_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi8>, %ws: tensor<?x?x?xf32>, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi8>
%r = tensor.dim %input, %c0 : tensor<?x?xf32>
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 2 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %group) : tensor<?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_a : tensor<?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%byte = tensor.extract %q#0[%row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%empty_w = tensor.empty(%n, %group) : tensor<?x?xf32>
%wx = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_w : tensor<?x?xf32>) {
^bb0(%unused_w: f32):
%col = linalg.index 0 : index
%j = linalg.index 1 : index
%column = arith.addi %begin, %j : index
%expert = arith.constant 0 : index
%byte = tensor.extract %wc[%expert, %col, %column] : tensor<?x?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?xf32>
%wbar = util.optimization_barrier %wx : tensor<?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?xf32>{%r, %group}
%wb = flow.tensor.tie_shape %wbar : tensor<?x?xf32>{%n, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>] ins(%ab, %wb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%expert = arith.constant 0 : index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %q#1[%row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func @grouped8_reciprocal_f32(%input: tensor<?x?xf32>, %wc: tensor<?x?x?xi32>, %ws: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %slots_arg: i32, %activations_arg: i32, %floor: f32) -> (tensor<?x?xf32>, tensor<i32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c128 = arith.constant 128 : index
%group = arith.constant 128 : index
%g32 = arith.constant 128 : i32
%k = tensor.dim %input, %c1 : tensor<?x?xf32>
%n = tensor.dim %wc, %c1 : tensor<?x?x?xi32>
%r = tensor.dim %ids, %c0 : tensor<?xi32>
%slots = arith.index_cast %slots_arg : i32 to index
%activations = arith.index_cast %activations_arg : i32 to index
%blocks = arith.divui %k, %group : index
%flag = func.call @finite_f32(%input) : (tensor<?x?xf32>) -> tensor<i32>
%policy = arith.constant 2 : i32
%q:2 = func.call @quant_f32(%input, %g32, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%r, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %blocks step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
%begin = arith.muli %block, %group : index
%empty_a = tensor.empty(%r, %c1, %group) : tensor<?x?x?xf32>
%ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
^bb0(%unused_a: f32):
%row = linalg.index 0 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%byte = tensor.extract %q#0[%activation_row, %column] : tensor<?x?xi8>
%code = arith.extui %byte : i8 to i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
linalg.yield %value : f32
} -> tensor<?x?x?xf32>
%abar = util.optimization_barrier %ax : tensor<?x?x?xf32>
%ab = flow.tensor.tie_shape %abar : tensor<?x?x?xf32>{%r, %c1, %group}
%empty_dot = tensor.empty(%r, %n) : tensor<?x?xf32>
%init_dot = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?xf32>) -> tensor<?x?xf32>
%dot = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,0,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%ab : tensor<?x?x?xf32>) outs(%init_dot : tensor<?x?xf32>) {
^bb0(%a_value: f32, %acc: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%j = linalg.index 2 : index
%column = arith.addi %begin, %j : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%pack = arith.constant 4 : index
%packed_col = arith.divui %column, %pack : index
%lane = arith.remui %column, %pack : index
%word = tensor.extract %wc[%expert, %col, %packed_col] : tensor<?x?x?xi32>
%lane32 = arith.index_cast %lane : index to i32
%bits = arith.constant 8 : i32
%mask = arith.constant 255 : i32
%shift = arith.muli %lane32, %bits : i32
%shifted = arith.shrui %word, %shift : i32
%code = arith.andi %shifted, %mask : i32
%value_byte = arith.trunci %code : i32 to i8
%value_encoded = arith.bitcast %value_byte : i8 to f8E4M3FN
%value_original = arith.extf %value_encoded : f8E4M3FN to f32
%value_mask = arith.constant 127 : i32
%value_magnitude = arith.andi %code, %value_mask : i32
%value_is_nan = arith.cmpi eq, %value_magnitude, %value_mask : i32
%value_nan_bits = arith.constant 2143289344 : i32
%value_nan = arith.bitcast %value_nan_bits : i32 to f32
%value = arith.select %value_is_nan, %value_nan, %value_original : f32
%product = arith.mulf %a_value, %value : f32
%accumulated = arith.addf %acc, %product : f32
linalg.yield %accumulated : f32
} -> tensor<?x?xf32>
%next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
^bb0(%unused_s: f32):
%row = linalg.index 0 : index
%col = linalg.index 1 : index
%token = arith.divui %row, %slots : index
%selected_slot = arith.remui %row, %slots : index
%shared = arith.cmpi eq, %activations, %c1 : index
%slot = arith.select %shared, %c0, %selected_slot : index
%base = arith.muli %token, %activations : index
%activation_row = arith.addi %base, %slot : index
%id = tensor.extract %ids[%row] : tensor<?xi32>
%expert = arith.index_cast %id : i32 to index
%weight_column = arith.divui %col, %c128 : index
%a_scale = tensor.extract %q#1[%activation_row, %block] : tensor<?x?xf32>
%w_scale = tensor.extract %ws[%expert, %weight_column, %block] : tensor<?x?x?xf32>
%d = tensor.extract %dot[%row, %col] : tensor<?x?xf32>
%old = tensor.extract %sum[%row, %col] : tensor<?x?xf32>
%activation_scaled = arith.mulf %d, %a_scale : f32
%weight_scaled = arith.mulf %activation_scaled, %w_scale : f32
%value = arith.addf %old, %weight_scaled : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
^bb0(%value: f32, %unused: f32):
linalg.yield %value : f32
} -> tensor<?x?xf32>
return %out, %flag : tensor<?x?xf32>, tensor<i32>
}
func.func private @normalize_divide_rne(%a: f32, %b: f32) -> f32 {
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
}
func.func private @normalize_multiply_rne(%a: f32, %b: f32) -> f32 {
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
}
func.func private @quant_output_f32(%x: tensor<?x?xf32>, %group_arg: i32, %floor: f32, %policy: i32) -> (tensor<?x?xi8>, tensor<?x?xf32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%group = arith.index_cast %group_arg : i32 to index
%m = tensor.dim %x, %c0 : tensor<?x?xf32>
%k = tensor.dim %x, %c1 : tensor<?x?xf32>
%blocks = arith.divui %k, %group : index
%wide = tensor.cast %x : tensor<?x?xf32> to tensor<?x?xf32>
%groups = tensor.expand_shape %wide [[0], [1, 2]] output_shape [%m, %blocks, %group] : tensor<?x?xf32> into tensor<?x?x?xf32>
%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%init_max = linalg.fill ins(%floor : f32) outs(%empty_max : tensor<?x?xf32>) -> tensor<?x?xf32>
%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xf32>) outs(%init_max : tensor<?x?xf32>) {
^bb0(%v: f32, %old: f32):
%abs = math.absf %v : f32
%value = arith.maximumf %abs, %old : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%scales = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xf32>) outs(%empty_max : tensor<?x?xf32>) {
^bb0(%amax: f32, %unused: f32):
%maxcode = arith.constant 448.0 : f32
%inverse_max = arith.constant 0.00223214296 : f32
%normal = func.call @divide_448_rne(%amax) : (f32) -> f32
%scaled = func.call @normalize_multiply_rne(%amax, %inverse_max) : (f32, f32) -> f32
%bits = arith.bitcast %scaled : f32 to i32
%c23 = arith.constant 23 : i32
%mask = arith.constant 8388607 : i32
%zero = arith.constant 0 : i32
%exponent = arith.shrui %bits, %c23 : i32
%fraction = arith.andi %bits, %mask : i32
%round = arith.cmpi ne, %fraction, %zero : i32
%extra = arith.extui %round : i1 to i32
%next = arith.addi %exponent, %extra : i32
%power_bits = arith.shli %next, %c23 : i32
%pow = arith.bitcast %power_bits : i32 to f32
%one = arith.constant 1 : i32
%two = arith.constant 2 : i32
%use_pow = arith.cmpi eq, %policy, %one : i32
%use_reciprocal = arith.cmpi eq, %policy, %two : i32
%ordinary_or_reciprocal = arith.select %use_reciprocal, %scaled, %normal : f32
%value = arith.select %use_pow, %pow, %ordinary_or_reciprocal : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%reciprocal_policy = arith.constant 2 : i32
%use_reciprocal = arith.cmpi eq, %policy, %reciprocal_policy : i32
%normalizers = scf.if %use_reciprocal -> (tensor<?x?xf32>) {
%empty_inverse = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%inverses = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%scales : tensor<?x?xf32>) outs(%empty_inverse : tensor<?x?xf32>) {
^bb0(%scale: f32, %unused: f32):
%inverse = func.call @reciprocal_normal_rne(%scale) : (f32) -> f32
linalg.yield %inverse : f32
} -> tensor<?x?xf32>
scf.yield %inverses : tensor<?x?xf32>
} else {
scf.yield %scales : tensor<?x?xf32>
}
%empty_codes = tensor.empty(%m, %k) : tensor<?x?xi8>
%codes = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_codes : tensor<?x?xi8>) {
^bb0(%unused: i8):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%b = arith.divui %j, %group : index
%v = tensor.extract %wide[%r, %j] : tensor<?x?xf32>
%s = tensor.extract %normalizers[%r, %b] : tensor<?x?xf32>
%normalized = scf.if %use_reciprocal -> (f32) {
%product = func.call @normalize_multiply_rne(%v, %s) : (f32, f32) -> f32
scf.yield %product : f32
} else {
%quotient = func.call @normalize_divide_rne(%v, %s) : (f32, f32) -> f32
scf.yield %quotient : f32
}
%lo = arith.constant -448.0 : f32
%hi = arith.constant 448.0 : f32
%lower = arith.maximumf %normalized, %lo : f32
%clipped = arith.minimumf %lower, %hi : f32
%bits = func.call @source_fp8_encode(%clipped) : (f32) -> i8
linalg.yield %bits : i8
} -> tensor<?x?xi8>
return %codes, %scales : tensor<?x?xi8>, tensor<?x?xf32>
}
func.func private @quant_output_bf16(%x: tensor<?x?xi16>, %group_arg: i32, %floor: f32, %policy: i32) -> (tensor<?x?xi8>, tensor<?x?xf32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%group = arith.index_cast %group_arg : i32 to index
%m = tensor.dim %x, %c0 : tensor<?x?xi16>
%k = tensor.dim %x, %c1 : tensor<?x?xi16>
%blocks = arith.divui %k, %group : index
%wide_empty = tensor.empty(%m, %k) : tensor<?x?xf32>
%wide = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%x : tensor<?x?xi16>) outs(%wide_empty : tensor<?x?xf32>) {
^bb0(%v: i16, %unused: f32):
%value_bf = arith.bitcast %v : i16 to bf16
%value = arith.extf %value_bf : bf16 to f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%groups = tensor.expand_shape %wide [[0], [1, 2]] output_shape [%m, %blocks, %group] : tensor<?x?xf32> into tensor<?x?x?xf32>
%empty_max = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%init_max = linalg.fill ins(%floor : f32) outs(%empty_max : tensor<?x?xf32>) -> tensor<?x?xf32>
%maxima = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel","parallel","reduction"]} ins(%groups : tensor<?x?x?xf32>) outs(%init_max : tensor<?x?xf32>) {
^bb0(%v: f32, %old: f32):
%abs = math.absf %v : f32
%value = arith.maximumf %abs, %old : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%scales = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%maxima : tensor<?x?xf32>) outs(%empty_max : tensor<?x?xf32>) {
^bb0(%amax: f32, %unused: f32):
%maxcode = arith.constant 448.0 : f32
%inverse_max = arith.constant 0.00223214296 : f32
%normal = func.call @divide_448_rne(%amax) : (f32) -> f32
%scaled = func.call @normalize_multiply_rne(%amax, %inverse_max) : (f32, f32) -> f32
%bits = arith.bitcast %scaled : f32 to i32
%c23 = arith.constant 23 : i32
%mask = arith.constant 8388607 : i32
%zero = arith.constant 0 : i32
%exponent = arith.shrui %bits, %c23 : i32
%fraction = arith.andi %bits, %mask : i32
%round = arith.cmpi ne, %fraction, %zero : i32
%extra = arith.extui %round : i1 to i32
%next = arith.addi %exponent, %extra : i32
%power_bits = arith.shli %next, %c23 : i32
%pow = arith.bitcast %power_bits : i32 to f32
%one = arith.constant 1 : i32
%two = arith.constant 2 : i32
%use_pow = arith.cmpi eq, %policy, %one : i32
%use_reciprocal = arith.cmpi eq, %policy, %two : i32
%ordinary_or_reciprocal = arith.select %use_reciprocal, %scaled, %normal : f32
%value = arith.select %use_pow, %pow, %ordinary_or_reciprocal : f32
linalg.yield %value : f32
} -> tensor<?x?xf32>
%reciprocal_policy = arith.constant 2 : i32
%use_reciprocal = arith.cmpi eq, %policy, %reciprocal_policy : i32
%normalizers = scf.if %use_reciprocal -> (tensor<?x?xf32>) {
%empty_inverse = tensor.empty(%m, %blocks) : tensor<?x?xf32>
%inverses = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%scales : tensor<?x?xf32>) outs(%empty_inverse : tensor<?x?xf32>) {
^bb0(%scale: f32, %unused: f32):
%inverse = func.call @reciprocal_normal_rne(%scale) : (f32) -> f32
linalg.yield %inverse : f32
} -> tensor<?x?xf32>
scf.yield %inverses : tensor<?x?xf32>
} else {
scf.yield %scales : tensor<?x?xf32>
}
%empty_codes = tensor.empty(%m, %k) : tensor<?x?xi8>
%codes = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_codes : tensor<?x?xi8>) {
^bb0(%unused: i8):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%b = arith.divui %j, %group : index
%v = tensor.extract %wide[%r, %j] : tensor<?x?xf32>
%s = tensor.extract %normalizers[%r, %b] : tensor<?x?xf32>
%normalized = scf.if %use_reciprocal -> (f32) {
%product = func.call @normalize_multiply_rne(%v, %s) : (f32, f32) -> f32
scf.yield %product : f32
} else {
%quotient = func.call @normalize_divide_rne(%v, %s) : (f32, f32) -> f32
scf.yield %quotient : f32
}
%lo = arith.constant -448.0 : f32
%hi = arith.constant 448.0 : f32
%lower = arith.maximumf %normalized, %lo : f32
%clipped = arith.minimumf %lower, %hi : f32
%bits = func.call @source_fp8_encode(%clipped) : (f32) -> i8
linalg.yield %bits : i8
} -> tensor<?x?xi8>
return %codes, %scales : tensor<?x?xi8>, tensor<?x?xf32>
}
func.func @quantize_normal_f32(%x: tensor<?x?xf32>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {
%group = arith.constant 128 : i32
%policy = arith.constant 0 : i32
%flag = func.call @finite_f32(%x) : (tensor<?x?xf32>) -> tensor<i32>
%encoded:2 = func.call @quant_output_f32(%x, %group, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>
}
func.func @quantize_power_f32(%x: tensor<?x?xf32>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {
%group = arith.constant 128 : i32
%policy = arith.constant 1 : i32
%flag = func.call @finite_f32(%x) : (tensor<?x?xf32>) -> tensor<i32>
%encoded:2 = func.call @quant_output_f32(%x, %group, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>
}
func.func @quantize_reciprocal_f32(%x: tensor<?x?xf32>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {
%group = arith.constant 128 : i32
%policy = arith.constant 2 : i32
%flag = func.call @finite_f32(%x) : (tensor<?x?xf32>) -> tensor<i32>
%encoded:2 = func.call @quant_output_f32(%x, %group, %floor, %policy) : (tensor<?x?xf32>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>
}
func.func @quantize_normal_bf16(%x: tensor<?x?xi16>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {
%group = arith.constant 128 : i32
%policy = arith.constant 0 : i32
%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>
%encoded:2 = func.call @quant_output_bf16(%x, %group, %floor, %policy) : (tensor<?x?xi16>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>
}
func.func @quantize_power_bf16(%x: tensor<?x?xi16>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {
%group = arith.constant 128 : i32
%policy = arith.constant 1 : i32
%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>
%encoded:2 = func.call @quant_output_bf16(%x, %group, %floor, %policy) : (tensor<?x?xi16>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>
}
func.func @quantize_reciprocal_bf16(%x: tensor<?x?xi16>, %floor: f32) -> (tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>) {
%group = arith.constant 128 : i32
%policy = arith.constant 2 : i32
%flag = func.call @finite_bf16(%x) : (tensor<?x?xi16>) -> tensor<i32>
%encoded:2 = func.call @quant_output_bf16(%x, %group, %floor, %policy) : (tensor<?x?xi16>, i32, f32, i32) -> (tensor<?x?xi8>, tensor<?x?xf32>)
return %encoded#0, %encoded#1, %flag : tensor<?x?xi8>, tensor<?x?xf32>, tensor<i32>
}
}
