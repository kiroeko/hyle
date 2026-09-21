module {
  func.func @f32_to_bf16(%input: tensor<?xf32>) -> tensor<?xi16> {
    %zero = arith.constant 0 : index
    %count = tensor.dim %input, %zero : tensor<?xf32>
    %empty = tensor.empty(%count) : tensor<?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%input : tensor<?xf32>) outs(%empty : tensor<?xi16>) {
      ^bb0(%value: f32, %unused: i16):
        %converted = arith.truncf %value : f32 to bf16
        %bits = arith.bitcast %converted : bf16 to i16
        linalg.yield %bits : i16
    } -> tensor<?xi16>
    return %result : tensor<?xi16>
  }
  func.func @f16_to_bf16(%input: tensor<?xf16>) -> tensor<?xi16> {
    %zero = arith.constant 0 : index
    %count = tensor.dim %input, %zero : tensor<?xf16>
    %empty = tensor.empty(%count) : tensor<?xi16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%input : tensor<?xf16>) outs(%empty : tensor<?xi16>) {
      ^bb0(%value: f16, %unused: i16):
        %wide = arith.extf %value : f16 to f32
        %converted = arith.truncf %wide : f32 to bf16
        %bits = arith.bitcast %converted : bf16 to i16
        linalg.yield %bits : i16
    } -> tensor<?xi16>
    return %result : tensor<?xi16>
  }
}
