module {
func.func @scan_inclusive(%input: tensor<?x?xi32>, %initial: tensor<?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %input, %c0 : tensor<?x?xi32>
  %n = tensor.dim %input, %c1 : tensor<?x?xi32>
  %empty = tensor.empty(%b, %n) : tensor<?x?xi32>
  %result:2 = iree_linalg_ext.scan dimension(1) inclusive(true)
      ins(%input : tensor<?x?xi32>) outs(%empty, %initial : tensor<?x?xi32>, tensor<?xi32>) {
    ^bb0(%left: i32, %right: i32):
      %sum = arith.addi %left, %right : i32
      iree_linalg_ext.yield %sum : i32
  } -> tensor<?x?xi32>, tensor<?xi32>
  return %result#0 : tensor<?x?xi32>
}

func.func @scan_exclusive(%input: tensor<?x?xi32>, %initial: tensor<?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %input, %c0 : tensor<?x?xi32>
  %n = tensor.dim %input, %c1 : tensor<?x?xi32>
  %empty = tensor.empty(%b, %n) : tensor<?x?xi32>
  %result:2 = iree_linalg_ext.scan dimension(1) inclusive(false)
      ins(%input : tensor<?x?xi32>) outs(%empty, %initial : tensor<?x?xi32>, tensor<?xi32>) {
    ^bb0(%left: i32, %right: i32):
      %sum = arith.addi %left, %right : i32
      iree_linalg_ext.yield %sum : i32
  } -> tensor<?x?xi32>, tensor<?xi32>
  return %result#0 : tensor<?x?xi32>
}

func.func @fp8_f32_scale(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %lhs_scale: tensor<?x?xf32>, %rhs_scale: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xi8>
  %k = tensor.dim %lhs, %c1 : tensor<?x?xi8>
  %n = tensor.dim %rhs, %c0 : tensor<?x?xi8>
  %l = flow.tensor.bitcast %lhs : tensor<?x?xi8>{%m, %k} -> tensor<?x?xf8E4M3FN>{%m, %k}
  %r = flow.tensor.bitcast %rhs : tensor<?x?xi8>{%n, %k} -> tensor<?x?xf8E4M3FN>{%n, %k}
  %ls = tensor.cast %lhs_scale : tensor<?x?xf32> to tensor<?x?xf32>
  %rs = tensor.cast %rhs_scale : tensor<?x?xf32> to tensor<?x?xf32>
  %le = tensor.empty(%m, %k) : tensor<?x?xf32>
  %re = tensor.empty(%n, %k) : tensor<?x?xf32>
  %lw = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1 floordiv 128)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%l, %ls : tensor<?x?xf8E4M3FN>, tensor<?x?xf32>) outs(%le : tensor<?x?xf32>) {
    ^bb0(%value: f8E4M3FN, %scale: f32, %unused: f32):
      %wide = arith.extf %value : f8E4M3FN to f32
      %scaled = arith.mulf %wide, %scale : f32
      linalg.yield %scaled : f32
  } -> tensor<?x?xf32>
  %rw = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0 floordiv 128, d1 floordiv 128)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%r, %rs : tensor<?x?xf8E4M3FN>, tensor<?x?xf32>) outs(%re : tensor<?x?xf32>) {
    ^bb0(%value: f8E4M3FN, %scale: f32, %unused: f32):
      %wide = arith.extf %value : f8E4M3FN to f32
      %scaled = arith.mulf %wide, %scale : f32
      linalg.yield %scaled : f32
  } -> tensor<?x?xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %lbarrier = util.optimization_barrier %lw : tensor<?x?xf32>
  %rbarrier = util.optimization_barrier %rw : tensor<?x?xf32>
  %lb = flow.tensor.tie_shape %lbarrier : tensor<?x?xf32>{%m, %k}
  %rb = flow.tensor.tie_shape %rbarrier : tensor<?x?xf32>{%n, %k}
  %result = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>,affine_map<(d0,d1,d2)->(d1,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>] ins(%lb, %rb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

func.func @fp8_ue8_scale(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %lhs_scale: tensor<?x?xi8>, %rhs_scale: tensor<?x?xi8>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xi8>
  %k = tensor.dim %lhs, %c1 : tensor<?x?xi8>
  %n = tensor.dim %rhs, %c0 : tensor<?x?xi8>
  %lg = tensor.dim %lhs_scale, %c1 : tensor<?x?xi8>
  %rg = tensor.dim %rhs_scale, %c1 : tensor<?x?xi8>
  %l = flow.tensor.bitcast %lhs : tensor<?x?xi8>{%m, %k} -> tensor<?x?xf8E4M3FN>{%m, %k}
  %r = flow.tensor.bitcast %rhs : tensor<?x?xi8>{%n, %k} -> tensor<?x?xf8E4M3FN>{%n, %k}
  %ls = flow.tensor.bitcast %lhs_scale : tensor<?x?xi8>{%m, %lg} -> tensor<?x?xf8E8M0FNU>{%m, %lg}
  %rs = flow.tensor.bitcast %rhs_scale : tensor<?x?xi8>{%n, %rg} -> tensor<?x?xf8E8M0FNU>{%n, %rg}
  %le = tensor.empty(%m, %k) : tensor<?x?xf32>
  %re = tensor.empty(%n, %k) : tensor<?x?xf32>
  %lw = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1 floordiv 32)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%l, %ls : tensor<?x?xf8E4M3FN>, tensor<?x?xf8E8M0FNU>) outs(%le : tensor<?x?xf32>) {
    ^bb0(%value: f8E4M3FN, %scale: f8E8M0FNU, %unused: f32):
      %wide = arith.extf %value : f8E4M3FN to f32
      %factor = arith.extf %scale : f8E8M0FNU to f32
      %scaled = arith.mulf %wide, %factor : f32
      linalg.yield %scaled : f32
  } -> tensor<?x?xf32>
  %rw = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0, d1 floordiv 32)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%r, %rs : tensor<?x?xf8E4M3FN>, tensor<?x?xf8E8M0FNU>) outs(%re : tensor<?x?xf32>) {
    ^bb0(%value: f8E4M3FN, %scale: f8E8M0FNU, %unused: f32):
      %wide = arith.extf %value : f8E4M3FN to f32
      %factor = arith.extf %scale : f8E8M0FNU to f32
      %scaled = arith.mulf %wide, %factor : f32
      linalg.yield %scaled : f32
  } -> tensor<?x?xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %lbarrier = util.optimization_barrier %lw : tensor<?x?xf32>
  %rbarrier = util.optimization_barrier %rw : tensor<?x?xf32>
  %lb = flow.tensor.tie_shape %lbarrier : tensor<?x?xf32>{%m, %k}
  %rb = flow.tensor.tie_shape %rbarrier : tensor<?x?xf32>{%n, %k}
  %result = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>,affine_map<(d0,d1,d2)->(d1,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>] ins(%lb, %rb : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

}
