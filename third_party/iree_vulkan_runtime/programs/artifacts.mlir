// Original mature tensor sequences; see NOTICE.md.
module {
  func.func @nf4_f32(%packed: tensor<?x?xi8>, %scales: tensor<?xf32>, %book: tensor<16xf32>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi8>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi8>
    %k = arith.muli %d1, %c2 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %packed_column = arith.divui %column, %c2 : index
        %lane = arith.remui %column, %c2 : index
        %byte = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi8>
        %word = arith.extui %byte : i8 to i32
        %high = arith.cmpi eq, %lane, %c0 : index
        %four_bits = arith.constant 4 : i32
        %high_word = arith.shrui %word, %four_bits : i32
        %selected = arith.select %high, %high_word, %word : i32
        %code = arith.andi %selected, %mask : i32
        %book_index = arith.index_castui %code : i32 to index
        %value = tensor.extract %book[%book_index] : tensor<16xf32>
        %row_base = arith.muli %row, %k : index
        %flat = arith.addi %row_base, %column : index
        %group_index = arith.divui %flat, %group : index
        %scale = tensor.extract %scales[%group_index] : tensor<?xf32>
        %decoded = arith.mulf %value, %scale : f32
        linalg.yield %decoded : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @nf4_f16(%packed: tensor<?x?xi8>, %scales: tensor<?xf32>, %book: tensor<16xf32>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi8>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi8>
    %k = arith.muli %d1, %c2 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %packed_column = arith.divui %column, %c2 : index
        %lane = arith.remui %column, %c2 : index
        %byte = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi8>
        %word = arith.extui %byte : i8 to i32
        %high = arith.cmpi eq, %lane, %c0 : index
        %four_bits = arith.constant 4 : i32
        %high_word = arith.shrui %word, %four_bits : i32
        %selected = arith.select %high, %high_word, %word : i32
        %code = arith.andi %selected, %mask : i32
        %book_index = arith.index_castui %code : i32 to index
        %value = tensor.extract %book[%book_index] : tensor<16xf32>
        %row_base = arith.muli %row, %k : index
        %flat = arith.addi %row_base, %column : index
        %group_index = arith.divui %flat, %group : index
        %scale = tensor.extract %scales[%group_index] : tensor<?xf32>
        %decoded = arith.mulf %value, %scale : f32
        %output_narrow = arith.truncf %decoded : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @nf4_bf16(%packed: tensor<?x?xi8>, %scales: tensor<?xf32>, %book: tensor<16xf32>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi8>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi8>
    %k = arith.muli %d1, %c2 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %packed_column = arith.divui %column, %c2 : index
        %lane = arith.remui %column, %c2 : index
        %byte = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi8>
        %word = arith.extui %byte : i8 to i32
        %high = arith.cmpi eq, %lane, %c0 : index
        %four_bits = arith.constant 4 : i32
        %high_word = arith.shrui %word, %four_bits : i32
        %selected = arith.select %high, %high_word, %word : i32
        %code = arith.andi %selected, %mask : i32
        %book_index = arith.index_castui %code : i32 to index
        %value = tensor.extract %book[%book_index] : tensor<16xf32>
        %row_base = arith.muli %row, %k : index
        %flat = arith.addi %row_base, %column : index
        %group_index = arith.divui %flat, %group : index
        %scale = tensor.extract %scales[%group_index] : tensor<?xf32>
        %decoded = arith.mulf %value, %scale : f32
        %output_narrow = arith.truncf %decoded : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @nf4_nested_f32(%packed: tensor<?x?xi8>, %scale_codes: tensor<?xi8>, %outer_scales: tensor<?xf32>, %scale_book: tensor<256xf32>, %offset: tensor<1xf32>, %book: tensor<16xf32>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi8>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi8>
    %k = arith.muli %d1, %c2 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %packed_column = arith.divui %column, %c2 : index
        %lane = arith.remui %column, %c2 : index
        %byte = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi8>
        %word = arith.extui %byte : i8 to i32
        %high = arith.cmpi eq, %lane, %c0 : index
        %four_bits = arith.constant 4 : i32
        %high_word = arith.shrui %word, %four_bits : i32
        %selected = arith.select %high, %high_word, %word : i32
        %code = arith.andi %selected, %mask : i32
        %book_index = arith.index_castui %code : i32 to index
        %value = tensor.extract %book[%book_index] : tensor<16xf32>
        %row_base = arith.muli %row, %k : index
        %flat = arith.addi %row_base, %column : index
        %group_index = arith.divui %flat, %group : index
        %outer_index = arith.divui %group_index, %c256 : index
        %scale_code = tensor.extract %scale_codes[%group_index] : tensor<?xi8>
        %scale_signed_index = arith.index_cast %scale_code : i8 to index
        %byte_mask = arith.constant 255 : index
        %scale_index = arith.andi %scale_signed_index, %byte_mask : index
        %scale_value = tensor.extract %scale_book[%scale_index] : tensor<256xf32>
        %outer = tensor.extract %outer_scales[%outer_index] : tensor<?xf32>
        %bias = tensor.extract %offset[%c0] : tensor<1xf32>
        %scaled = arith.mulf %scale_value, %outer : f32
        %scale = arith.addf %scaled, %bias : f32
        %decoded = arith.mulf %value, %scale : f32
        linalg.yield %decoded : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @nf4_nested_f16(%packed: tensor<?x?xi8>, %scale_codes: tensor<?xi8>, %outer_scales: tensor<?xf32>, %scale_book: tensor<256xf32>, %offset: tensor<1xf32>, %book: tensor<16xf32>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi8>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi8>
    %k = arith.muli %d1, %c2 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %packed_column = arith.divui %column, %c2 : index
        %lane = arith.remui %column, %c2 : index
        %byte = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi8>
        %word = arith.extui %byte : i8 to i32
        %high = arith.cmpi eq, %lane, %c0 : index
        %four_bits = arith.constant 4 : i32
        %high_word = arith.shrui %word, %four_bits : i32
        %selected = arith.select %high, %high_word, %word : i32
        %code = arith.andi %selected, %mask : i32
        %book_index = arith.index_castui %code : i32 to index
        %value = tensor.extract %book[%book_index] : tensor<16xf32>
        %row_base = arith.muli %row, %k : index
        %flat = arith.addi %row_base, %column : index
        %group_index = arith.divui %flat, %group : index
        %outer_index = arith.divui %group_index, %c256 : index
        %scale_code = tensor.extract %scale_codes[%group_index] : tensor<?xi8>
        %scale_signed_index = arith.index_cast %scale_code : i8 to index
        %byte_mask = arith.constant 255 : index
        %scale_index = arith.andi %scale_signed_index, %byte_mask : index
        %scale_value = tensor.extract %scale_book[%scale_index] : tensor<256xf32>
        %outer = tensor.extract %outer_scales[%outer_index] : tensor<?xf32>
        %bias = tensor.extract %offset[%c0] : tensor<1xf32>
        %scaled = arith.mulf %scale_value, %outer : f32
        %scale = arith.addf %scaled, %bias : f32
        %decoded = arith.mulf %value, %scale : f32
        %output_narrow = arith.truncf %decoded : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @nf4_nested_bf16(%packed: tensor<?x?xi8>, %scale_codes: tensor<?xi8>, %outer_scales: tensor<?xf32>, %scale_book: tensor<256xf32>, %offset: tensor<1xf32>, %book: tensor<16xf32>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi8>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi8>
    %k = arith.muli %d1, %c2 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %packed_column = arith.divui %column, %c2 : index
        %lane = arith.remui %column, %c2 : index
        %byte = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi8>
        %word = arith.extui %byte : i8 to i32
        %high = arith.cmpi eq, %lane, %c0 : index
        %four_bits = arith.constant 4 : i32
        %high_word = arith.shrui %word, %four_bits : i32
        %selected = arith.select %high, %high_word, %word : i32
        %code = arith.andi %selected, %mask : i32
        %book_index = arith.index_castui %code : i32 to index
        %value = tensor.extract %book[%book_index] : tensor<16xf32>
        %row_base = arith.muli %row, %k : index
        %flat = arith.addi %row_base, %column : index
        %group_index = arith.divui %flat, %group : index
        %outer_index = arith.divui %group_index, %c256 : index
        %scale_code = tensor.extract %scale_codes[%group_index] : tensor<?xi8>
        %scale_signed_index = arith.index_cast %scale_code : i8 to index
        %byte_mask = arith.constant 255 : index
        %scale_index = arith.andi %scale_signed_index, %byte_mask : index
        %scale_value = tensor.extract %scale_book[%scale_index] : tensor<256xf32>
        %outer = tensor.extract %outer_scales[%outer_index] : tensor<?xf32>
        %bias = tensor.extract %offset[%c0] : tensor<1xf32>
        %scaled = arith.mulf %scale_value, %outer : f32
        %scale = arith.addf %scaled, %bias : f32
        %decoded = arith.mulf %value, %scale : f32
        %output_narrow = arith.truncf %decoded : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @awq_f32_f32(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xf32>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xf32>
        %decoded = arith.mulf %value, %scale_bits : f32
        linalg.yield %decoded : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @awq_f32_f16(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xf32>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xf32>
        %decoded = arith.mulf %value, %scale_bits : f32
        %output_narrow = arith.truncf %decoded : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @awq_f32_bf16(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xf32>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xf32>
        %decoded = arith.mulf %value, %scale_bits : f32
        %output_narrow = arith.truncf %decoded : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @awq_f16_f32(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xf16>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xf16>
        %scale_wide = arith.extf %scale_bits : f16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to f16
        %source_rounded_wide = arith.extf %source_product_narrow : f16 to f32
        linalg.yield %source_rounded_wide : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @awq_f16_f16(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xf16>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xf16>
        %scale_wide = arith.extf %scale_bits : f16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to f16
        %source_rounded_wide = arith.extf %source_product_narrow : f16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @awq_f16_bf16(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xf16>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xf16>
        %scale_wide = arith.extf %scale_bits : f16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to f16
        %source_rounded_wide = arith.extf %source_product_narrow : f16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @awq_bf16_f32(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xi16>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xi16>
        %scale_bf = arith.bitcast %scale_bits : i16 to bf16
        %scale_wide = arith.extf %scale_bf : bf16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to bf16
        %source_product = arith.bitcast %source_product_narrow : bf16 to i16
        %source_rounded_bf = arith.bitcast %source_product : i16 to bf16
        %source_rounded_wide = arith.extf %source_rounded_bf : bf16 to f32
        linalg.yield %source_rounded_wide : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @awq_bf16_f16(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xi16>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xi16>
        %scale_bf = arith.bitcast %scale_bits : i16 to bf16
        %scale_wide = arith.extf %scale_bf : bf16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to bf16
        %source_product = arith.bitcast %source_product_narrow : bf16 to i16
        %source_rounded_bf = arith.bitcast %source_product : i16 to bf16
        %source_rounded_wide = arith.extf %source_rounded_bf : bf16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @awq_bf16_bf16(%packed: tensor<?x?xi32>, %zeros: tensor<?x?xi32>, %scales: tensor<?x?xi16>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %n = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%n, %d0) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %row, %c8 : index
        %lane = arith.remui %row, %c8 : index
        %even = arith.divui %lane, %c2 : index
        %odd = arith.remui %lane, %c2 : index
        %odd_shift = arith.muli %odd, %c4 : index
        %physical_lane = arith.addi %even, %odd_shift : index
        %word = tensor.extract %packed[%column, %packed_column] : tensor<?x?xi32>
        %zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %zero_shifted = arith.shrui %zero_word, %shift : i32
        %zero_code = arith.andi %zero_shifted, %mask : i32
        %signed = arith.subi %code, %zero_code : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%group_index, %row] : tensor<?x?xi16>
        %scale_bf = arith.bitcast %scale_bits : i16 to bf16
        %scale_wide = arith.extf %scale_bf : bf16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to bf16
        %source_product = arith.bitcast %source_product_narrow : bf16 to i16
        %source_rounded_bf = arith.bitcast %source_product : i16 to bf16
        %source_rounded_wide = arith.extf %source_rounded_bf : bf16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @ct_f32_f32(%packed: tensor<?x?xi32>, %scales: tensor<?x?xf32>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xf32>
        %decoded = arith.mulf %value, %scale_bits : f32
        linalg.yield %decoded : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @ct_f32_f16(%packed: tensor<?x?xi32>, %scales: tensor<?x?xf32>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xf32>
        %decoded = arith.mulf %value, %scale_bits : f32
        %output_narrow = arith.truncf %decoded : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @ct_f32_bf16(%packed: tensor<?x?xi32>, %scales: tensor<?x?xf32>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xf32>
        %decoded = arith.mulf %value, %scale_bits : f32
        %output_narrow = arith.truncf %decoded : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @ct_f16_f32(%packed: tensor<?x?xi32>, %scales: tensor<?x?xf16>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xf16>
        %scale_wide = arith.extf %scale_bits : f16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to f16
        %source_rounded_wide = arith.extf %source_product_narrow : f16 to f32
        linalg.yield %source_rounded_wide : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @ct_f16_f16(%packed: tensor<?x?xi32>, %scales: tensor<?x?xf16>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xf16>
        %scale_wide = arith.extf %scale_bits : f16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to f16
        %source_rounded_wide = arith.extf %source_product_narrow : f16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @ct_f16_bf16(%packed: tensor<?x?xi32>, %scales: tensor<?x?xf16>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xf16>
        %scale_wide = arith.extf %scale_bits : f16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to f16
        %source_rounded_wide = arith.extf %source_product_narrow : f16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
  func.func @ct_bf16_f32(%packed: tensor<?x?xi32>, %scales: tensor<?x?xi16>, %group_arg: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf32>) {
      ^bb0(%unused: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xi16>
        %scale_bf = arith.bitcast %scale_bits : i16 to bf16
        %scale_wide = arith.extf %scale_bf : bf16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to bf16
        %source_product = arith.bitcast %source_product_narrow : bf16 to i16
        %source_rounded_bf = arith.bitcast %source_product : i16 to bf16
        %source_rounded_wide = arith.extf %source_rounded_bf : bf16 to f32
        linalg.yield %source_rounded_wide : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
  func.func @ct_bf16_f16(%packed: tensor<?x?xi32>, %scales: tensor<?x?xi16>, %group_arg: i32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xf16>) {
      ^bb0(%unused: f16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xi16>
        %scale_bf = arith.bitcast %scale_bits : i16 to bf16
        %scale_wide = arith.extf %scale_bf : bf16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to bf16
        %source_product = arith.bitcast %source_product_narrow : bf16 to i16
        %source_rounded_bf = arith.bitcast %source_product : i16 to bf16
        %source_rounded_wide = arith.extf %source_rounded_bf : bf16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to f16
        linalg.yield %output_narrow : f16
    } -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }
  func.func @ct_bf16_bf16(%packed: tensor<?x?xi32>, %scales: tensor<?x?xi16>, %group_arg: i32) -> tensor<?x?xi16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %mask = arith.constant 15 : i32
    %eight = arith.constant 8 : i32
    %group = arith.index_cast %group_arg : i32 to index
    %d0 = tensor.dim %packed, %c0 : tensor<?x?xi32>
    %d1 = tensor.dim %packed, %c1 : tensor<?x?xi32>
    %k = arith.muli %d1, %c8 : index
    %empty = tensor.empty(%d0, %k) : tensor<?x?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%empty : tensor<?x?xi16>) {
      ^bb0(%unused: i16):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %group_index = arith.divui %column, %group : index
        %packed_column = arith.divui %column, %c8 : index
        %physical_lane = arith.remui %column, %c8 : index
        %word = tensor.extract %packed[%row, %packed_column] : tensor<?x?xi32>
        %shift_index = arith.muli %physical_lane, %c4 : index
        %shift = arith.index_cast %shift_index : index to i32
        %shifted = arith.shrui %word, %shift : i32
        %code = arith.andi %shifted, %mask : i32
        %signed = arith.subi %code, %eight : i32
        %value = arith.sitofp %signed : i32 to f32
        %scale_bits = tensor.extract %scales[%row, %group_index] : tensor<?x?xi16>
        %scale_bf = arith.bitcast %scale_bits : i16 to bf16
        %scale_wide = arith.extf %scale_bf : bf16 to f32
        %decoded = arith.mulf %value, %scale_wide : f32
        %source_product_narrow = arith.truncf %decoded : f32 to bf16
        %source_product = arith.bitcast %source_product_narrow : bf16 to i16
        %source_rounded_bf = arith.bitcast %source_product : i16 to bf16
        %source_rounded_wide = arith.extf %source_rounded_bf : bf16 to f32
        %output_narrow = arith.truncf %source_rounded_wide : f32 to bf16
        %output = arith.bitcast %output_narrow : bf16 to i16
        linalg.yield %output : i16
    } -> tensor<?x?xi16>
    return %result : tensor<?x?xi16>
  }
}
