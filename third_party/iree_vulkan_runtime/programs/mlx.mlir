module {
func.func @affine_linear_f32(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%k = tensor.dim %a, %c1 : tensor<?x?xf32>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf32>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf32>
      %scaled = arith.mulf %cv, %sv : f32
      %value = arith.addf %scaled, %bv : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_f32(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf32>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf32>
      %scaled = arith.mulf %cv, %sv : f32
      %value = arith.addf %scaled, %bv : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_f16(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%k = tensor.dim %a, %c1 : tensor<?x?xf16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf16>
      %scale = arith.extf %sv : f16 to f32
      %bias = arith.extf %bv : f16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @affine_grouped_f16(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf16>
      %scale = arith.extf %sv : f16 to f32
      %bias = arith.extf %bv : f16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @affine_linear_bf16(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%k = tensor.dim %a, %c1 : tensor<?x?xi16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xi16>
      %scale_bf = arith.bitcast %sv : i16 to bf16
      %scale = arith.extf %scale_bf : bf16 to f32
      %bias_bf = arith.bitcast %bv : i16 to bf16
      %bias = arith.extf %bias_bf : bf16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @affine_grouped_bf16(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xi16>
      %scale_bf = arith.bitcast %sv : i16 to bf16
      %scale = arith.extf %scale_bf : bf16 to f32
      %bias_bf = arith.bitcast %bv : i16 to bf16
      %bias = arith.extf %bias_bf : bf16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @mx4_linear_f32(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %group_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 8 : index
%bits = arith.constant 4 : index
%mask = arith.constant 15 : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%k = tensor.dim %a, %c1 : tensor<?x?xf32>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %ci = arith.index_castui %code : i32 to index
      %value = tensor.extract %fp4[%ci] : tensor<16xf32>
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf32>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf32>) {
    ^bb0(%unused_scale: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %expert = arith.constant 0 : index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      linalg.yield %factor : f32
  } -> tensor<?x?xf32>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf32>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf32>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf32>
      %scaled_dot = arith.mulf %d, %sval : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @mx4_grouped_f32(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %ids: tensor<?xi32>, %group_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 8 : index
%bits = arith.constant 4 : index
%mask = arith.constant 15 : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi8>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %ci = arith.index_castui %code : i32 to index
      %value = tensor.extract %fp4[%ci] : tensor<16xf32>
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf32>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf32>) {
    ^bb0(%unused_scale: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      linalg.yield %factor : f32
  } -> tensor<?x?xf32>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf32>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf32>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf32>
      %scaled_dot = arith.mulf %d, %sval : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @mx4_linear_f16(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %group_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 8 : index
%bits = arith.constant 4 : index
%mask = arith.constant 15 : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%k = tensor.dim %a, %c1 : tensor<?x?xf16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %ci = arith.index_castui %code : i32 to index
      %value = tensor.extract %fp4[%ci] : tensor<16xf32>
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf16>) {
    ^bb0(%unused_scale: f16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %expert = arith.constant 0 : index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to f16
      linalg.yield %source_scale_float : f16
  } -> tensor<?x?xf16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf16>
      %source_factor = arith.extf %sval : f16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @mx4_grouped_f16(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %ids: tensor<?xi32>, %group_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 8 : index
%bits = arith.constant 4 : index
%mask = arith.constant 15 : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi8>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %ci = arith.index_castui %code : i32 to index
      %value = tensor.extract %fp4[%ci] : tensor<16xf32>
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf16>) {
    ^bb0(%unused_scale: f16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to f16
      linalg.yield %source_scale_float : f16
  } -> tensor<?x?xf16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf16>
      %source_factor = arith.extf %sval : f16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @mx4_linear_bf16(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %group_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 8 : index
%bits = arith.constant 4 : index
%mask = arith.constant 15 : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%k = tensor.dim %a, %c1 : tensor<?x?xi16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %ci = arith.index_castui %code : i32 to index
      %value = tensor.extract %fp4[%ci] : tensor<16xf32>
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xi16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xi16>) {
    ^bb0(%unused_scale: i16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %expert = arith.constant 0 : index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to bf16
      %source_scale = arith.bitcast %source_scale_float : bf16 to i16
      linalg.yield %source_scale : i16
  } -> tensor<?x?xi16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xi16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xi16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xi16>
      %source_factor_bf = arith.bitcast %sval : i16 to bf16
      %source_factor = arith.extf %source_factor_bf : bf16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @mx4_grouped_bf16(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %ids: tensor<?xi32>, %group_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 8 : index
%bits = arith.constant 4 : index
%mask = arith.constant 15 : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi8>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %ci = arith.index_castui %code : i32 to index
      %value = tensor.extract %fp4[%ci] : tensor<16xf32>
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xi16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xi16>) {
    ^bb0(%unused_scale: i16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to bf16
      %source_scale = arith.bitcast %source_scale_float : bf16 to i16
      linalg.yield %source_scale : i16
  } -> tensor<?x?xi16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xi16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xi16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xi16>
      %source_factor_bf = arith.bitcast %sval : i16 to bf16
      %source_factor = arith.extf %source_factor_bf : bf16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @mx8_linear_f32(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %group_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 4 : index
%bits = arith.constant 8 : index
%mask = arith.constant 255 : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%k = tensor.dim %a, %c1 : tensor<?x?xf32>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %encoded = arith.trunci %code : i32 to i8
      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN
      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32
      %nan_mask = arith.constant 127 : i32
      %magnitude = arith.andi %code, %nan_mask : i32
      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32
      %nan_bits = arith.constant 2143289344 : i32
      %nan = arith.bitcast %nan_bits : i32 to f32
      %value = arith.select %is_nan, %nan, %decoded_value : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf32>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf32>) {
    ^bb0(%unused_scale: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %expert = arith.constant 0 : index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      linalg.yield %factor : f32
  } -> tensor<?x?xf32>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf32>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf32>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf32>
      %scaled_dot = arith.mulf %d, %sval : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @mx8_grouped_f32(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %ids: tensor<?xi32>, %group_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 4 : index
%bits = arith.constant 8 : index
%mask = arith.constant 255 : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi8>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %encoded = arith.trunci %code : i32 to i8
      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN
      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32
      %nan_mask = arith.constant 127 : i32
      %magnitude = arith.andi %code, %nan_mask : i32
      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32
      %nan_bits = arith.constant 2143289344 : i32
      %nan = arith.bitcast %nan_bits : i32 to f32
      %value = arith.select %is_nan, %nan, %decoded_value : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf32>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf32>) {
    ^bb0(%unused_scale: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      linalg.yield %factor : f32
  } -> tensor<?x?xf32>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf32>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf32>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf32>
      %scaled_dot = arith.mulf %d, %sval : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @mx8_linear_f16(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %group_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 4 : index
%bits = arith.constant 8 : index
%mask = arith.constant 255 : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%k = tensor.dim %a, %c1 : tensor<?x?xf16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %encoded = arith.trunci %code : i32 to i8
      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN
      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32
      %nan_mask = arith.constant 127 : i32
      %magnitude = arith.andi %code, %nan_mask : i32
      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32
      %nan_bits = arith.constant 2143289344 : i32
      %nan = arith.bitcast %nan_bits : i32 to f32
      %value = arith.select %is_nan, %nan, %decoded_value : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf16>) {
    ^bb0(%unused_scale: f16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %expert = arith.constant 0 : index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to f16
      linalg.yield %source_scale_float : f16
  } -> tensor<?x?xf16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf16>
      %source_factor = arith.extf %sval : f16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @mx8_grouped_f16(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %ids: tensor<?xi32>, %group_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 4 : index
%bits = arith.constant 8 : index
%mask = arith.constant 255 : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi8>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %encoded = arith.trunci %code : i32 to i8
      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN
      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32
      %nan_mask = arith.constant 127 : i32
      %magnitude = arith.andi %code, %nan_mask : i32
      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32
      %nan_bits = arith.constant 2143289344 : i32
      %nan = arith.bitcast %nan_bits : i32 to f32
      %value = arith.select %is_nan, %nan, %decoded_value : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xf16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xf16>) {
    ^bb0(%unused_scale: f16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to f16
      linalg.yield %source_scale_float : f16
  } -> tensor<?x?xf16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xf16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xf16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xf16>
      %source_factor = arith.extf %sval : f16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @mx8_linear_bf16(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %group_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 4 : index
%bits = arith.constant 8 : index
%mask = arith.constant 255 : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%k = tensor.dim %a, %c1 : tensor<?x?xi16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %encoded = arith.trunci %code : i32 to i8
      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN
      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32
      %nan_mask = arith.constant 127 : i32
      %magnitude = arith.andi %code, %nan_mask : i32
      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32
      %nan_bits = arith.constant 2143289344 : i32
      %nan = arith.bitcast %nan_bits : i32 to f32
      %value = arith.select %is_nan, %nan, %decoded_value : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xi16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xi16>) {
    ^bb0(%unused_scale: i16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %expert = arith.constant 0 : index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to bf16
      %source_scale = arith.bitcast %source_scale_float : bf16 to i16
      linalg.yield %source_scale : i16
  } -> tensor<?x?xi16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xi16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xi16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xi16>
      %source_factor_bf = arith.bitcast %sval : i16 to bf16
      %source_factor = arith.extf %source_factor_bf : bf16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @mx8_grouped_bf16(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi8>, %ids: tensor<?xi32>, %group_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%pack = arith.constant 4 : index
%bits = arith.constant 8 : index
%mask = arith.constant 255 : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi8>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %wc = arith.divui %column, %pack : index
      %lane = arith.remui %column, %pack : index
      %shift_index = arith.muli %lane, %bits : index
      %shift = arith.index_cast %shift_index : index to i32
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %shifted = arith.shrui %word, %shift : i32
      %code = arith.andi %shifted, %mask : i32
      %encoded = arith.trunci %code : i32 to i8
      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN
      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32
      %nan_mask = arith.constant 127 : i32
      %magnitude = arith.andi %code, %nan_mask : i32
      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32
      %nan_bits = arith.constant 2143289344 : i32
      %nan = arith.bitcast %nan_bits : i32 to f32
      %value = arith.select %is_nan, %nan, %decoded_value : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_scale = tensor.empty(%rows, %n) : tensor<?x?xi16>
  %typed_scale = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_scale : tensor<?x?xi16>) {
    ^bb0(%unused_scale: i16):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi8>
      %u = arith.extui %sv : i8 to i32
      %z = arith.constant 0 : i32
      %shift = arith.constant 23 : i32
      %subnormal = arith.constant 4194304 : i32
      %raw = arith.shli %u, %shift : i32
      %is_zero = arith.cmpi eq, %u, %z : i32
      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32
      %factor = arith.bitcast %scale_bits : i32 to f32
      %source_scale_float = arith.truncf %factor : f32 to bf16
      %source_scale = arith.bitcast %source_scale_float : bf16 to i16
      linalg.yield %source_scale : i16
  } -> tensor<?x?xi16>
  %s_bar = util.optimization_barrier %typed_scale : tensor<?x?xi16>
  %sb = flow.tensor.tie_shape %s_bar : tensor<?x?xi16>{%rows, %n}
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %sval = tensor.extract %sb[%r, %col] : tensor<?x?xi16>
      %source_factor_bf = arith.bitcast %sval : i16 to bf16
      %source_factor = arith.extf %source_factor_bf : bf16 to f32
      %scaled_dot = arith.mulf %d, %source_factor : f32
      %value = arith.addf %old, %scaled_dot : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @affine_embedding_f32(%w: tensor<?x?xi32>, %s: tensor<?x?xf32>, %b: tensor<?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%groups = tensor.dim %s, %c1 : tensor<?x?xf32>
%k = arith.muli %groups, %group : index
%empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%product = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xf32>) {
  ^bb0(%unused: f32):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%id = tensor.extract %ids[%r] : tensor<?xi32>
%row = arith.index_cast %id : i32 to index
%word_group = arith.divui %j, %c32 : index
%lane = arith.remui %j, %c32 : index
%word_base = arith.muli %word_group, %bits : index
%bit_start = arith.muli %lane, %bits : index
%word_offset = arith.divui %bit_start, %c32 : index
%wc = arith.addi %word_base, %word_offset : index
%shift_index = arith.remui %bit_start, %c32 : index
%end_index = arith.addi %shift_index, %bits : index
%cross = arith.cmpi ugt, %end_index, %c32 : index
%next_wc = arith.addi %wc, %c1 : index
%safe_next = arith.select %cross, %next_wc, %wc : index
%word = tensor.extract %w[%row, %wc] : tensor<?x?xi32>
%next_word = tensor.extract %w[%row, %safe_next] : tensor<?x?xi32>
%shift = arith.index_cast %shift_index : index to i32
%low = arith.shrui %word, %shift : i32
%left_index = arith.subi %c32, %shift_index : index
%left_raw = arith.index_cast %left_index : index to i32
%shift_mask = arith.constant 31 : i32
%left = arith.andi %left_raw, %shift_mask : i32
%high_raw = arith.shli %next_word, %left : i32
%zero_bits = arith.constant 0 : i32
%high = arith.select %cross, %high_raw, %zero_bits : i32
%joined = arith.ori %low, %high : i32
%code = arith.andi %joined, %mask : i32
%value = arith.uitofp %code : i32 to f32
%block = arith.divui %j, %group : index
%sv = tensor.extract %s[%row, %block] : tensor<?x?xf32>
%scaled = arith.mulf %value, %sv : f32
linalg.yield %scaled : f32
} -> tensor<?x?xf32>
%barrier = util.optimization_barrier %product : tensor<?x?xf32>
%typed = flow.tensor.tie_shape %barrier : tensor<?x?xf32>{%rows, %k}
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%typed : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
  ^bb0(%p: f32, %unused_out: f32):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%id = tensor.extract %ids[%r] : tensor<?xi32>
%row = arith.index_cast %id : i32 to index
%block = arith.divui %j, %group : index
%bv = tensor.extract %b[%row, %block] : tensor<?x?xf32>
%added = arith.addf %p, %bv : f32
linalg.yield %added : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_embedding_f16(%w: tensor<?x?xi32>, %s: tensor<?x?xf16>, %b: tensor<?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%groups = tensor.dim %s, %c1 : tensor<?x?xf16>
%k = arith.muli %groups, %group : index
%empty = tensor.empty(%rows, %k) : tensor<?x?xf16>
%product = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xf16>) {
  ^bb0(%unused: f16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%id = tensor.extract %ids[%r] : tensor<?xi32>
%row = arith.index_cast %id : i32 to index
%word_group = arith.divui %j, %c32 : index
%lane = arith.remui %j, %c32 : index
%word_base = arith.muli %word_group, %bits : index
%bit_start = arith.muli %lane, %bits : index
%word_offset = arith.divui %bit_start, %c32 : index
%wc = arith.addi %word_base, %word_offset : index
%shift_index = arith.remui %bit_start, %c32 : index
%end_index = arith.addi %shift_index, %bits : index
%cross = arith.cmpi ugt, %end_index, %c32 : index
%next_wc = arith.addi %wc, %c1 : index
%safe_next = arith.select %cross, %next_wc, %wc : index
%word = tensor.extract %w[%row, %wc] : tensor<?x?xi32>
%next_word = tensor.extract %w[%row, %safe_next] : tensor<?x?xi32>
%shift = arith.index_cast %shift_index : index to i32
%low = arith.shrui %word, %shift : i32
%left_index = arith.subi %c32, %shift_index : index
%left_raw = arith.index_cast %left_index : index to i32
%shift_mask = arith.constant 31 : i32
%left = arith.andi %left_raw, %shift_mask : i32
%high_raw = arith.shli %next_word, %left : i32
%zero_bits = arith.constant 0 : i32
%high = arith.select %cross, %high_raw, %zero_bits : i32
%joined = arith.ori %low, %high : i32
%code = arith.andi %joined, %mask : i32
%value = arith.uitofp %code : i32 to f32
%block = arith.divui %j, %group : index
%sv = tensor.extract %s[%row, %block] : tensor<?x?xf16>
%scale = arith.extf %sv : f16 to f32
%scaled = arith.mulf %value, %scale : f32
%product_value_float = arith.truncf %scaled : f32 to f16
linalg.yield %product_value_float : f16
} -> tensor<?x?xf16>
%barrier = util.optimization_barrier %product : tensor<?x?xf16>
%typed = flow.tensor.tie_shape %barrier : tensor<?x?xf16>{%rows, %k}
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%typed : tensor<?x?xf16>) outs(%empty : tensor<?x?xf16>) {
  ^bb0(%p: f16, %unused_out: f16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%id = tensor.extract %ids[%r] : tensor<?xi32>
%row = arith.index_cast %id : i32 to index
%block = arith.divui %j, %group : index
%bv = tensor.extract %b[%row, %block] : tensor<?x?xf16>
%p_wide = arith.extf %p : f16 to f32
%b_wide = arith.extf %bv : f16 to f32
%added = arith.addf %p_wide, %b_wide : f32
%out_value_float = arith.truncf %added : f32 to f16
linalg.yield %out_value_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @affine_embedding_bf16(%w: tensor<?x?xi32>, %s: tensor<?x?xi16>, %b: tensor<?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%groups = tensor.dim %s, %c1 : tensor<?x?xi16>
%k = arith.muli %groups, %group : index
%empty = tensor.empty(%rows, %k) : tensor<?x?xi16>
%product = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty : tensor<?x?xi16>) {
  ^bb0(%unused: i16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%id = tensor.extract %ids[%r] : tensor<?xi32>
%row = arith.index_cast %id : i32 to index
%word_group = arith.divui %j, %c32 : index
%lane = arith.remui %j, %c32 : index
%word_base = arith.muli %word_group, %bits : index
%bit_start = arith.muli %lane, %bits : index
%word_offset = arith.divui %bit_start, %c32 : index
%wc = arith.addi %word_base, %word_offset : index
%shift_index = arith.remui %bit_start, %c32 : index
%end_index = arith.addi %shift_index, %bits : index
%cross = arith.cmpi ugt, %end_index, %c32 : index
%next_wc = arith.addi %wc, %c1 : index
%safe_next = arith.select %cross, %next_wc, %wc : index
%word = tensor.extract %w[%row, %wc] : tensor<?x?xi32>
%next_word = tensor.extract %w[%row, %safe_next] : tensor<?x?xi32>
%shift = arith.index_cast %shift_index : index to i32
%low = arith.shrui %word, %shift : i32
%left_index = arith.subi %c32, %shift_index : index
%left_raw = arith.index_cast %left_index : index to i32
%shift_mask = arith.constant 31 : i32
%left = arith.andi %left_raw, %shift_mask : i32
%high_raw = arith.shli %next_word, %left : i32
%zero_bits = arith.constant 0 : i32
%high = arith.select %cross, %high_raw, %zero_bits : i32
%joined = arith.ori %low, %high : i32
%code = arith.andi %joined, %mask : i32
%value = arith.uitofp %code : i32 to f32
%block = arith.divui %j, %group : index
%sv = tensor.extract %s[%row, %block] : tensor<?x?xi16>
%scale_bf = arith.bitcast %sv : i16 to bf16
%scale = arith.extf %scale_bf : bf16 to f32
%scaled = arith.mulf %value, %scale : f32
%product_value_float = arith.truncf %scaled : f32 to bf16
%product_value = arith.bitcast %product_value_float : bf16 to i16
linalg.yield %product_value : i16
} -> tensor<?x?xi16>
%barrier = util.optimization_barrier %product : tensor<?x?xi16>
%typed = flow.tensor.tie_shape %barrier : tensor<?x?xi16>{%rows, %k}
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%typed : tensor<?x?xi16>) outs(%empty : tensor<?x?xi16>) {
  ^bb0(%p: i16, %unused_out: i16):
%r = linalg.index 0 : index
%j = linalg.index 1 : index
%id = tensor.extract %ids[%r] : tensor<?xi32>
%row = arith.index_cast %id : i32 to index
%block = arith.divui %j, %group : index
%bv = tensor.extract %b[%row, %block] : tensor<?x?xi16>
%p_wide_bf = arith.bitcast %p : i16 to bf16
%p_wide = arith.extf %p_wide_bf : bf16 to f32
%b_wide_bf = arith.bitcast %bv : i16 to bf16
%b_wide = arith.extf %b_wide_bf : bf16 to f32
%added = arith.addf %p_wide, %b_wide : f32
%out_value_float = arith.truncf %added : f32 to bf16
%out_value = arith.bitcast %out_value_float : bf16 to i16
linalg.yield %out_value : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @affine_linear_f32_f16(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%k = tensor.dim %a, %c1 : tensor<?x?xf32>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf16>
      %scale = arith.extf %sv : f16 to f32
      %bias = arith.extf %bv : f16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_f32_f16(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf16>
      %scale = arith.extf %sv : f16 to f32
      %bias = arith.extf %bv : f16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_f32_bf16(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%k = tensor.dim %a, %c1 : tensor<?x?xf32>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xi16>
      %scale_bf = arith.bitcast %sv : i16 to bf16
      %scale = arith.extf %scale_bf : bf16 to f32
      %bias_bf = arith.bitcast %bv : i16 to bf16
      %bias = arith.extf %bias_bf : bf16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_f32_bf16(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf32>
      linalg.yield %av : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xi16>
      %scale_bf = arith.bitcast %sv : i16 to bf16
      %scale = arith.extf %scale_bf : bf16 to f32
      %bias_bf = arith.bitcast %bv : i16 to bf16
      %bias = arith.extf %bias_bf : bf16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_f16_f32(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%k = tensor.dim %a, %c1 : tensor<?x?xf16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf32>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf32>
      %scaled = arith.mulf %cv, %sv : f32
      %value = arith.addf %scaled, %bv : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_f16_f32(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf32>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf32>
      %scaled = arith.mulf %cv, %sv : f32
      %value = arith.addf %scaled, %bv : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_f16_bf16(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%k = tensor.dim %a, %c1 : tensor<?x?xf16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xi16>
      %scale_bf = arith.bitcast %sv : i16 to bf16
      %scale = arith.extf %scale_bf : bf16 to f32
      %bias_bf = arith.bitcast %bv : i16 to bf16
      %bias = arith.extf %bias_bf : bf16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_f16_bf16(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xf16>
      %activation = arith.extf %av : f16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xi16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xi16>
      %scale_bf = arith.bitcast %sv : i16 to bf16
      %scale = arith.extf %scale_bf : bf16 to f32
      %bias_bf = arith.bitcast %bv : i16 to bf16
      %bias = arith.extf %bias_bf : bf16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_bf16_f32(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%k = tensor.dim %a, %c1 : tensor<?x?xi16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf32>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf32>
      %scaled = arith.mulf %cv, %sv : f32
      %value = arith.addf %scaled, %bv : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_bf16_f32(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf32>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf32>
      %scaled = arith.mulf %cv, %sv : f32
      %value = arith.addf %scaled, %bv : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_bf16_f16(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%k = tensor.dim %a, %c1 : tensor<?x?xi16>
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %expert = arith.constant 0 : index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf16>
      %scale = arith.extf %sv : f16 to f32
      %bias = arith.extf %bv : f16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_bf16_f16(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%c2 = arith.constant 2 : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%scale_groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %scale_groups, %group : index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = arith.divui %k, %group : index
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %begin = arith.muli %block, %group : index
  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
    ^bb0(%unused_a: f32):
      %r = linalg.index 0 : index
      %j = linalg.index 2 : index
      %ac = arith.addi %begin, %j : index
      %token = arith.divui %r, %slots : index
      %slot = arith.remui %r, %slots : index
      %actual_slot = arith.select %shared, %c0, %slot : index
      %token_base = arith.muli %token, %token_stride : index
      %slot_base = arith.muli %actual_slot, %k : index
      %row_base = arith.addi %token_base, %slot_base : index
      %at = arith.addi %row_base, %ac : index
      %av = tensor.extract %a[%at] : tensor<?xi16>
      %activation_bf = arith.bitcast %av : i16 to bf16
      %activation = arith.extf %activation_bf : bf16 to f32
      linalg.yield %activation : f32
  } -> tensor<?x?x?xf32>
  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>
  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
    ^bb0(%unused_w: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %j = linalg.index 2 : index
      %id = tensor.extract %ids[%r] : tensor<?xi32>
      %expert = arith.index_cast %id : i32 to index
      %column = arith.addi %begin, %j : index
      %word_group = arith.divui %column, %c32 : index
      %lane = arith.remui %column, %c32 : index
      %word_base = arith.muli %word_group, %bits : index
      %bit_start = arith.muli %lane, %bits : index
      %word_offset = arith.divui %bit_start, %c32 : index
      %wc = arith.addi %word_base, %word_offset : index
      %shift_index = arith.remui %bit_start, %c32 : index
      %end_index = arith.addi %shift_index, %bits : index
      %cross = arith.cmpi ugt, %end_index, %c32 : index
      %next_wc = arith.addi %wc, %c1 : index
      %safe_next = arith.select %cross, %next_wc, %wc : index
      %word = tensor.extract %w[%expert, %col, %wc] : tensor<?x?x?xi32>
      %next_word = tensor.extract %w[%expert, %col, %safe_next] : tensor<?x?x?xi32>
      %shift = arith.index_cast %shift_index : index to i32
      %low = arith.shrui %word, %shift : i32
      %left_index = arith.subi %c32, %shift_index : index
      %left_raw = arith.index_cast %left_index : index to i32
      %shift_mask = arith.constant 31 : i32
      %left = arith.andi %left_raw, %shift_mask : i32
      %high_raw = arith.shli %next_word, %left : i32
      %zero_bits = arith.constant 0 : i32
      %high = arith.select %cross, %high_raw, %zero_bits : i32
      %joined = arith.ori %low, %high : i32
      %code = arith.andi %joined, %mask : i32
      %cv = arith.uitofp %code : i32 to f32
      %sv = tensor.extract %s[%expert, %col, %block] : tensor<?x?x?xf16>
      %bv = tensor.extract %b[%expert, %col, %block] : tensor<?x?x?xf16>
      %scale = arith.extf %sv : f16 to f32
      %bias = arith.extf %bv : f16 to f32
      %scaled = arith.mulf %cv, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  } -> tensor<?x?x?xf32>
  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}
  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}
  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>
  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {
    ^bb0(%unused_sum: f32):
      %r = linalg.index 0 : index
      %col = linalg.index 1 : index
      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>
      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>
      %value = arith.addf %old, %d : f32
      linalg.yield %value : f32
  } -> tensor<?x?xf32>
  scf.yield %next : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %n) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_f32(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
        linalg.yield %av : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %scaled = arith.mulf %cv, %sv : f32
        %value = arith.addf %scaled, %bv : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_f32(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xf32>
        linalg.yield %av : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %scaled = arith.mulf %cv, %sv : f32
        %value = arith.addf %scaled, %bv : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_f32_f16(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
        linalg.yield %av : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %scale = arith.extf %sv : f16 to f32
        %bias = arith.extf %bv : f16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_f32_f16(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xf32>
        linalg.yield %av : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %scale = arith.extf %sv : f16 to f32
        %bias = arith.extf %bv : f16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_f32_bf16(%a: tensor<?x?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xf32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xf32>
        linalg.yield %av : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %scale_bf = arith.bitcast %sv : i16 to bf16
        %scale = arith.extf %scale_bf : bf16 to f32
        %bias_bf = arith.bitcast %bv : i16 to bf16
        %bias = arith.extf %bias_bf : bf16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_f32_bf16(%a: tensor<?xf32>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xf32>
        linalg.yield %av : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %scale_bf = arith.bitcast %sv : i16 to bf16
        %scale = arith.extf %scale_bf : bf16 to f32
        %bias_bf = arith.bitcast %bv : i16 to bf16
        %bias = arith.extf %bias_bf : bf16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_f16_f32(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
        %activation = arith.extf %av : f16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %scaled = arith.mulf %cv, %sv : f32
        %value = arith.addf %scaled, %bv : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_f16_f32(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xf16>
        %activation = arith.extf %av : f16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %scaled = arith.mulf %cv, %sv : f32
        %value = arith.addf %scaled, %bv : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_f16(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
        %activation = arith.extf %av : f16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %scale = arith.extf %sv : f16 to f32
        %bias = arith.extf %bv : f16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @affine_grouped_as_stored_f16(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xf16>
        %activation = arith.extf %av : f16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %scale = arith.extf %sv : f16 to f32
        %bias = arith.extf %bv : f16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf16>) {
  ^bb0(%v: f32, %unused_out: f16):
    %converted_float = arith.truncf %v : f32 to f16
    linalg.yield %converted_float : f16
} -> tensor<?x?xf16>
return %out : tensor<?x?xf16>
}
func.func @affine_linear_as_stored_f16_bf16(%a: tensor<?x?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xf16>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xf16>
        %activation = arith.extf %av : f16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %scale_bf = arith.bitcast %sv : i16 to bf16
        %scale = arith.extf %scale_bf : bf16 to f32
        %bias_bf = arith.bitcast %bv : i16 to bf16
        %bias = arith.extf %bias_bf : bf16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_f16_bf16(%a: tensor<?xf16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xf16>
        %activation = arith.extf %av : f16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %scale_bf = arith.bitcast %sv : i16 to bf16
        %scale = arith.extf %scale_bf : bf16 to f32
        %bias_bf = arith.bitcast %bv : i16 to bf16
        %bias = arith.extf %bias_bf : bf16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_bf16_f32(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
        %activation_bf = arith.bitcast %av : i16 to bf16
        %activation = arith.extf %activation_bf : bf16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %scaled = arith.mulf %cv, %sv : f32
        %value = arith.addf %scaled, %bv : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_bf16_f32(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf32>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xi16>
        %activation_bf = arith.bitcast %av : i16 to bf16
        %activation = arith.extf %activation_bf : bf16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf32>
        %scaled = arith.mulf %cv, %sv : f32
        %value = arith.addf %scaled, %bv : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_bf16_f16(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
        %activation_bf = arith.bitcast %av : i16 to bf16
        %activation = arith.extf %activation_bf : bf16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %scale = arith.extf %sv : f16 to f32
        %bias = arith.extf %bv : f16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_grouped_as_stored_bf16_f16(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xf16>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xi16>
        %activation_bf = arith.bitcast %av : i16 to bf16
        %activation = arith.extf %activation_bf : bf16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xf16>
        %scale = arith.extf %sv : f16 to f32
        %bias = arith.extf %bv : f16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xf32>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused_out: f32):
    linalg.yield %v : f32
} -> tensor<?x?xf32>
return %out : tensor<?x?xf32>
}
func.func @affine_linear_as_stored_bf16(%a: tensor<?x?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %group_arg: i32, %bits_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %groups, %group : index
%rows = tensor.dim %a, %c0 : tensor<?x?xi16>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %av = tensor.extract %a[%r, %ac] : tensor<?x?xi16>
        %activation_bf = arith.bitcast %av : i16 to bf16
        %activation = arith.extf %activation_bf : bf16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %expert = arith.constant 0 : index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %scale_bf = arith.bitcast %sv : i16 to bf16
        %scale = arith.extf %scale_bf : bf16 to f32
        %bias_bf = arith.bitcast %bv : i16 to bf16
        %bias = arith.extf %bias_bf : bf16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
func.func @affine_grouped_as_stored_bf16(%a: tensor<?xi16>, %w: tensor<?x?x?xi32>, %s: tensor<?x?x?xi16>, %b: tensor<?x?x?xi16>, %ids: tensor<?xi32>, %group_arg: i32, %bits_arg: i32, %slots_arg: i32, %activation_slots_arg: i32, %token_stride_arg: i32) -> tensor<?x?xi16> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%bits = arith.index_cast %bits_arg : i32 to index
%one_bit = arith.constant 1 : i32
%mask_high = arith.shli %one_bit, %bits_arg : i32
%mask = arith.subi %mask_high, %one_bit : i32
%group = arith.index_cast %group_arg : i32 to index
%n = tensor.dim %w, %c1 : tensor<?x?x?xi32>
%groups = tensor.dim %s, %c2 : tensor<?x?x?xi16>
%k = arith.muli %groups, %group : index
%slots = arith.index_cast %slots_arg : i32 to index
%activation_slots = arith.index_cast %activation_slots_arg : i32 to index
%token_stride = arith.index_cast %token_stride_arg : i32 to index
%shared = arith.cmpi eq, %activation_slots, %c1 : index
%rows = tensor.dim %ids, %c0 : tensor<?xi32>
%zero = arith.constant 0.0 : f32
%empty_sum = tensor.empty(%rows, %k) : tensor<?x?xf32>
%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>
%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {
  %out_begin = arith.muli %block, %group : index
  %empty_part0 = tensor.empty(%rows, %group) : tensor<?x?xf32>
  %initial = linalg.fill ins(%zero : f32) outs(%empty_part0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %part = scf.for %begin = %c0 to %n step %c64 iter_args(%partial = %initial) -> (tensor<?x?xf32>) {
    %remaining = arith.subi %n, %begin : index
    %tile = arith.minui %remaining, %c64 : index
    %empty_a = tensor.empty(%rows, %c1, %tile) : tensor<?x?x?xf32>
    %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {
      ^bb0(%unused_a: f32):
        %r = linalg.index 0 : index
        %j = linalg.index 2 : index
        %ac = arith.addi %begin, %j : index
        %token = arith.divui %r, %slots : index
        %slot = arith.remui %r, %slots : index
        %actual_slot = arith.select %shared, %c0, %slot : index
        %token_base = arith.muli %token, %token_stride : index
        %slot_base = arith.muli %actual_slot, %n : index
        %row_base = arith.addi %token_base, %slot_base : index
        %at = arith.addi %row_base, %ac : index
        %av = tensor.extract %a[%at] : tensor<?xi16>
        %activation_bf = arith.bitcast %av : i16 to bf16
        %activation = arith.extf %activation_bf : bf16 to f32
        linalg.yield %activation : f32
    } -> tensor<?x?x?xf32>
    %empty_w = tensor.empty(%rows, %group, %tile) : tensor<?x?x?xf32>
    %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {
      ^bb0(%unused_w: f32):
        %r = linalg.index 0 : index
        %out_lane = linalg.index 1 : index
        %j = linalg.index 2 : index
        %source_n = arith.addi %begin, %j : index
        %column = arith.addi %out_begin, %out_lane : index
        %id = tensor.extract %ids[%r] : tensor<?xi32>
        %expert = arith.index_cast %id : i32 to index
        %word_group = arith.divui %column, %c32 : index
        %lane = arith.remui %column, %c32 : index
        %word_base = arith.muli %word_group, %bits : index
        %bit_start = arith.muli %lane, %bits : index
        %word_offset = arith.divui %bit_start, %c32 : index
        %wc = arith.addi %word_base, %word_offset : index
        %shift_index = arith.remui %bit_start, %c32 : index
        %end_index = arith.addi %shift_index, %bits : index
        %cross = arith.cmpi ugt, %end_index, %c32 : index
        %next_wc = arith.addi %wc, %c1 : index
        %safe_next = arith.select %cross, %next_wc, %wc : index
        %word = tensor.extract %w[%expert, %source_n, %wc] : tensor<?x?x?xi32>
        %next_word = tensor.extract %w[%expert, %source_n, %safe_next] : tensor<?x?x?xi32>
        %shift = arith.index_cast %shift_index : index to i32
        %low = arith.shrui %word, %shift : i32
        %left_index = arith.subi %c32, %shift_index : index
        %left_raw = arith.index_cast %left_index : index to i32
        %shift_mask = arith.constant 31 : i32
        %left = arith.andi %left_raw, %shift_mask : i32
        %high_raw = arith.shli %next_word, %left : i32
        %zero_bits = arith.constant 0 : i32
        %high = arith.select %cross, %high_raw, %zero_bits : i32
        %joined = arith.ori %low, %high : i32
        %code = arith.andi %joined, %mask : i32
        %cv = arith.uitofp %code : i32 to f32
        %sv = tensor.extract %s[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %bv = tensor.extract %b[%expert, %source_n, %block] : tensor<?x?x?xi16>
        %scale_bf = arith.bitcast %sv : i16 to bf16
        %scale = arith.extf %scale_bf : bf16 to f32
        %bias_bf = arith.bitcast %bv : i16 to bf16
        %bias = arith.extf %bias_bf : bf16 to f32
        %scaled = arith.mulf %cv, %scale : f32
        %value = arith.addf %scaled, %bias : f32
        linalg.yield %value : f32
    } -> tensor<?x?x?xf32>
    %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>
    %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>
    %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %tile}
    %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %group, %tile}
    %empty_dot = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>
    %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %empty_part = tensor.empty(%rows, %group) : tensor<?x?xf32>
    %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_part : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %r = linalg.index 0 : index
        %c = linalg.index 1 : index
        %d = tensor.extract %dot[%r, %c0, %c] : tensor<?x?x?xf32>
        %old = tensor.extract %partial[%r, %c] : tensor<?x?xf32>
        %v = arith.addf %old, %d : f32
        linalg.yield %v : f32
    } -> tensor<?x?xf32>
    scf.yield %next : tensor<?x?xf32>
  }
  %inserted = tensor.insert_slice %part into %sum[0, %out_begin] [%rows, %group] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  scf.yield %inserted : tensor<?x?xf32>
}
%out_empty = tensor.empty(%rows, %k) : tensor<?x?xi16>
%out = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%total : tensor<?x?xf32>) outs(%out_empty : tensor<?x?xi16>) {
  ^bb0(%v: f32, %unused_out: i16):
    %converted_float = arith.truncf %v : f32 to bf16
    %converted = arith.bitcast %converted_float : bf16 to i16
    linalg.yield %converted : i16
} -> tensor<?x?xi16>
return %out : tensor<?x?xi16>
}
}
