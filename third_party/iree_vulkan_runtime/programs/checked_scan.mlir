module {
func.func @scan_inclusive(%input: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = tensor.dim %input, %c0 : tensor<?x?xi32>
  %width = tensor.dim %input, %c1 : tensor<?x?xi32>
  %zero = arith.constant 0 : i32
  %zero64 = arith.constant 0 : i64
  %minimum = arith.constant -2147483648 : i64
  %maximum = arith.constant 2147483647 : i64
  %wide_empty = tensor.empty(%rows, %width) : tensor<?x?xi64>
  %wide = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%input : tensor<?x?xi32>) outs(%wide_empty : tensor<?x?xi64>) {
    ^bb0(%value: i32, %unused: i64):
      %converted = arith.extsi %value : i32 to i64
      linalg.yield %converted : i64
  } -> tensor<?x?xi64>
  %carry_empty = tensor.empty(%rows) : tensor<?xi64>
  %initial = linalg.fill ins(%zero64 : i64) outs(%carry_empty : tensor<?xi64>) -> tensor<?xi64>
  %result:2 = iree_linalg_ext.scan dimension(1) inclusive(true)
      ins(%wide : tensor<?x?xi64>) outs(%wide_empty, %initial : tensor<?x?xi64>, tensor<?xi64>) {
    ^bb0(%left: i64, %right: i64):
      %sum = arith.addi %left, %right : i64
      iree_linalg_ext.yield %sum : i64
  } -> tensor<?x?xi64>, tensor<?xi64>
  %flags_empty = tensor.empty(%rows, %width) : tensor<?x?xi32>
  %flags = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%result#0 : tensor<?x?xi64>) outs(%flags_empty : tensor<?x?xi32>) {
    ^bb0(%value: i64, %unused: i32):
      %low = arith.cmpi slt, %value, %minimum : i64
      %high = arith.cmpi sgt, %value, %maximum : i64
      %invalid = arith.ori %low, %high : i1
      %flag = arith.extui %invalid : i1 to i32
      linalg.yield %flag : i32
  } -> tensor<?x?xi32>
  %carry_flags_empty = tensor.empty(%rows) : tensor<?xi32>
  %carry_flags = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%result#1 : tensor<?xi64>) outs(%carry_flags_empty : tensor<?xi32>) {
    ^bb0(%value: i64, %unused: i32):
      %low = arith.cmpi slt, %value, %minimum : i64
      %high = arith.cmpi sgt, %value, %maximum : i64
      %invalid = arith.ori %low, %high : i1
      %flag = arith.extui %invalid : i1 to i32
      linalg.yield %flag : i32
  } -> tensor<?xi32>
  %scalar_empty = tensor.empty() : tensor<i32>
  %scalar_zero = linalg.fill ins(%zero : i32) outs(%scalar_empty : tensor<i32>) -> tensor<i32>
  %prefix_status = linalg.reduce ins(%flags : tensor<?x?xi32>) outs(%scalar_zero : tensor<i32>) dimensions = [0,1]
      (%value: i32, %acc: i32) {
        %combined = arith.ori %value, %acc : i32
        linalg.yield %combined : i32
      }
  %status = linalg.reduce ins(%carry_flags : tensor<?xi32>) outs(%prefix_status : tensor<i32>) dimensions = [0]
      (%value: i32, %acc: i32) {
        %combined = arith.ori %value, %acc : i32
        linalg.yield %combined : i32
      }
  %output = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%result#0 : tensor<?x?xi64>) outs(%flags_empty : tensor<?x?xi32>) {
    ^bb0(%value: i64, %unused: i32):
      %converted = arith.trunci %value : i64 to i32
      linalg.yield %converted : i32
  } -> tensor<?x?xi32>
  return %output, %status : tensor<?x?xi32>, tensor<i32>
}
func.func @scan_exclusive(%input: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = tensor.dim %input, %c0 : tensor<?x?xi32>
  %width = tensor.dim %input, %c1 : tensor<?x?xi32>
  %zero = arith.constant 0 : i32
  %zero64 = arith.constant 0 : i64
  %minimum = arith.constant -2147483648 : i64
  %maximum = arith.constant 2147483647 : i64
  %wide_empty = tensor.empty(%rows, %width) : tensor<?x?xi64>
  %wide = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%input : tensor<?x?xi32>) outs(%wide_empty : tensor<?x?xi64>) {
    ^bb0(%value: i32, %unused: i64):
      %converted = arith.extsi %value : i32 to i64
      linalg.yield %converted : i64
  } -> tensor<?x?xi64>
  %carry_empty = tensor.empty(%rows) : tensor<?xi64>
  %initial = linalg.fill ins(%zero64 : i64) outs(%carry_empty : tensor<?xi64>) -> tensor<?xi64>
  %result:2 = iree_linalg_ext.scan dimension(1) inclusive(false)
      ins(%wide : tensor<?x?xi64>) outs(%wide_empty, %initial : tensor<?x?xi64>, tensor<?xi64>) {
    ^bb0(%left: i64, %right: i64):
      %sum = arith.addi %left, %right : i64
      iree_linalg_ext.yield %sum : i64
  } -> tensor<?x?xi64>, tensor<?xi64>
  %flags_empty = tensor.empty(%rows, %width) : tensor<?x?xi32>
  %flags = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%result#0 : tensor<?x?xi64>) outs(%flags_empty : tensor<?x?xi32>) {
    ^bb0(%value: i64, %unused: i32):
      %low = arith.cmpi slt, %value, %minimum : i64
      %high = arith.cmpi sgt, %value, %maximum : i64
      %invalid = arith.ori %low, %high : i1
      %flag = arith.extui %invalid : i1 to i32
      linalg.yield %flag : i32
  } -> tensor<?x?xi32>
  %final_total = linalg.reduce ins(%wide : tensor<?x?xi64>) outs(%initial : tensor<?xi64>) dimensions = [1]
      (%value: i64, %acc: i64) {
        %sum = arith.addi %value, %acc : i64
        linalg.yield %sum : i64
      }
  %carry_flags_empty = tensor.empty(%rows) : tensor<?xi32>
  %carry_flags = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%final_total : tensor<?xi64>) outs(%carry_flags_empty : tensor<?xi32>) {
    ^bb0(%value: i64, %unused: i32):
      %low = arith.cmpi slt, %value, %minimum : i64
      %high = arith.cmpi sgt, %value, %maximum : i64
      %invalid = arith.ori %low, %high : i1
      %flag = arith.extui %invalid : i1 to i32
      linalg.yield %flag : i32
  } -> tensor<?xi32>
  %scalar_empty = tensor.empty() : tensor<i32>
  %scalar_zero = linalg.fill ins(%zero : i32) outs(%scalar_empty : tensor<i32>) -> tensor<i32>
  %prefix_status = linalg.reduce ins(%flags : tensor<?x?xi32>) outs(%scalar_zero : tensor<i32>) dimensions = [0,1]
      (%value: i32, %acc: i32) {
        %combined = arith.ori %value, %acc : i32
        linalg.yield %combined : i32
      }
  %status = linalg.reduce ins(%carry_flags : tensor<?xi32>) outs(%prefix_status : tensor<i32>) dimensions = [0]
      (%value: i32, %acc: i32) {
        %combined = arith.ori %value, %acc : i32
        linalg.yield %combined : i32
      }
  %output = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%result#0 : tensor<?x?xi64>) outs(%flags_empty : tensor<?x?xi32>) {
    ^bb0(%value: i64, %unused: i32):
      %converted = arith.trunci %value : i64 to i32
      linalg.yield %converted : i32
  } -> tensor<?x?xi32>
  return %output, %status : tensor<?x?xi32>, tensor<i32>
}
}
