#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 floordiv 128)>
#map2 = affine_map<(d0, d1) -> (d0 floordiv 128, d1 floordiv 128)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1 floordiv 32)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>, sym_name = "scan_inclusive"}> ({
  ^bb0(%arg34: tensor<?x?xi32>, %arg35: tensor<?xi32>):
    %79 = "arith.constant"() <{value = 0 : index}> : () -> index
    %80 = "arith.constant"() <{value = 1 : index}> : () -> index
    %81 = "tensor.dim"(%arg34, %79) : (tensor<?x?xi32>, index) -> index
    %82 = "tensor.dim"(%arg34, %80) : (tensor<?x?xi32>, index) -> index
    %83 = "tensor.empty"(%81, %82) : (index, index) -> tensor<?x?xi32>
    %84:2 = "iree_linalg_ext.scan"(%arg34, %83, %arg35) <{dimension = 1 : i64, inclusive = true}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %85 = "arith.addi"(%arg36, %arg37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "iree_linalg_ext.yield"(%85) : (i32) -> ()
    }) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?xi32>) -> (tensor<?x?xi32>, tensor<?xi32>)
    "func.return"(%84#0) : (tensor<?x?xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>, sym_name = "scan_exclusive"}> ({
  ^bb0(%arg30: tensor<?x?xi32>, %arg31: tensor<?xi32>):
    %72 = "arith.constant"() <{value = 0 : index}> : () -> index
    %73 = "arith.constant"() <{value = 1 : index}> : () -> index
    %74 = "tensor.dim"(%arg30, %72) : (tensor<?x?xi32>, index) -> index
    %75 = "tensor.dim"(%arg30, %73) : (tensor<?x?xi32>, index) -> index
    %76 = "tensor.empty"(%74, %75) : (index, index) -> tensor<?x?xi32>
    %77:2 = "iree_linalg_ext.scan"(%arg30, %76, %arg31) <{dimension = 1 : i64, inclusive = false}> ({
    ^bb0(%arg32: i32, %arg33: i32):
      %78 = "arith.addi"(%arg32, %arg33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "iree_linalg_ext.yield"(%78) : (i32) -> ()
    }) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?xi32>) -> (tensor<?x?xi32>, tensor<?xi32>)
    "func.return"(%77#0) : (tensor<?x?xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<?x?xi8>, tensor<?x?xi8>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>, sym_name = "fp8_f32_scale"}> ({
  ^bb0(%arg15: tensor<?x?xi8>, %arg16: tensor<?x?xi8>, %arg17: tensor<?x?xf32>, %arg18: tensor<?x?xf32>):
    %47 = "arith.constant"() <{value = 0 : index}> : () -> index
    %48 = "arith.constant"() <{value = 1 : index}> : () -> index
    %49 = "tensor.dim"(%arg15, %47) : (tensor<?x?xi8>, index) -> index
    %50 = "tensor.dim"(%arg15, %48) : (tensor<?x?xi8>, index) -> index
    %51 = "tensor.dim"(%arg16, %47) : (tensor<?x?xi8>, index) -> index
    %52 = "flow.tensor.bitcast"(%arg15, %49, %50, %49, %50) <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?xf8E4M3FN>
    %53 = "flow.tensor.bitcast"(%arg16, %51, %50, %51, %50) <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?xf8E4M3FN>
    %54 = "tensor.empty"(%49, %50) : (index, index) -> tensor<?x?xf32>
    %55 = "tensor.empty"(%51, %50) : (index, index) -> tensor<?x?xf32>
    %56 = "linalg.generic"(%52, %arg17, %54) <{indexing_maps = [#map, #map1, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg27: f8E4M3FN, %arg28: f32, %arg29: f32):
      %70 = "arith.extf"(%arg27) : (f8E4M3FN) -> f32
      %71 = "arith.mulf"(%70, %arg28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%71) : (f32) -> ()
    }) : (tensor<?x?xf8E4M3FN>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %57 = "linalg.generic"(%53, %arg18, %55) <{indexing_maps = [#map, #map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg24: f8E4M3FN, %arg25: f32, %arg26: f32):
      %68 = "arith.extf"(%arg24) : (f8E4M3FN) -> f32
      %69 = "arith.mulf"(%68, %arg25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%69) : (f32) -> ()
    }) : (tensor<?x?xf8E4M3FN>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %58 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %59 = "tensor.empty"(%49, %51) : (index, index) -> tensor<?x?xf32>
    %60 = "linalg.fill"(%58, %59) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg22: f32, %arg23: f32):
      "linalg.yield"(%arg22) : (f32) -> ()
    }) : (f32, tensor<?x?xf32>) -> tensor<?x?xf32>
    %61 = "util.optimization_barrier"(%56) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %62 = "util.optimization_barrier"(%57) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %63 = "flow.tensor.tie_shape"(%61, %49, %50) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %64 = "flow.tensor.tie_shape"(%62, %51, %50) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %65 = "linalg.matmul"(%63, %64, %60) <{indexing_maps = [#map3, #map4, #map5], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg19: f32, %arg20: f32, %arg21: f32):
      %66 = "arith.mulf"(%arg19, %arg20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %67 = "arith.addf"(%arg21, %66) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%67) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "func.return"(%65) : (tensor<?x?xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<?x?xi8>, tensor<?x?xi8>, tensor<?x?xi8>, tensor<?x?xi8>) -> tensor<?x?xf32>, sym_name = "fp8_ue8_scale"}> ({
  ^bb0(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi8>, %arg3: tensor<?x?xi8>):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "tensor.dim"(%arg0, %0) : (tensor<?x?xi8>, index) -> index
    %3 = "tensor.dim"(%arg0, %1) : (tensor<?x?xi8>, index) -> index
    %4 = "tensor.dim"(%arg1, %0) : (tensor<?x?xi8>, index) -> index
    %5 = "tensor.dim"(%arg2, %1) : (tensor<?x?xi8>, index) -> index
    %6 = "tensor.dim"(%arg3, %1) : (tensor<?x?xi8>, index) -> index
    %7 = "flow.tensor.bitcast"(%arg0, %2, %3, %2, %3) <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?xf8E4M3FN>
    %8 = "flow.tensor.bitcast"(%arg1, %4, %3, %4, %3) <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?xf8E4M3FN>
    %9 = "flow.tensor.bitcast"(%arg2, %2, %5, %2, %5) <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?xf8E8M0FNU>
    %10 = "flow.tensor.bitcast"(%arg3, %4, %6, %4, %6) <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<?x?xi8>, index, index, index, index) -> tensor<?x?xf8E8M0FNU>
    %11 = "tensor.empty"(%2, %3) : (index, index) -> tensor<?x?xf32>
    %12 = "tensor.empty"(%4, %3) : (index, index) -> tensor<?x?xf32>
    %13 = "linalg.generic"(%7, %9, %11) <{indexing_maps = [#map, #map6, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg12: f8E4M3FN, %arg13: f8E8M0FNU, %arg14: f32):
      %36 = "arith.extf"(%arg12) : (f8E4M3FN) -> f32
      %37 = "arith.bitcast"(%arg13) : (f8E8M0FNU) -> i8
      %38 = "arith.constant"() <{value = 23 : i32}> : () -> i32
      %39 = "arith.extui"(%37) : (i8) -> i32
      %40 = "arith.shli"(%39, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %41 = "arith.constant"() <{value = -1 : i8}> : () -> i8
      %42 = "arith.constant"() <{value = -1 : i32}> : () -> i32
      %43 = "arith.cmpi"(%37, %41) <{predicate = 0 : i64}> : (i8, i8) -> i1
      %44 = "arith.select"(%43, %42, %40) : (i1, i32, i32) -> i32
      %45 = "arith.bitcast"(%44) : (i32) -> f32
      %46 = "arith.mulf"(%36, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%46) : (f32) -> ()
    }) : (tensor<?x?xf8E4M3FN>, tensor<?x?xf8E8M0FNU>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %14 = "linalg.generic"(%8, %10, %12) <{indexing_maps = [#map, #map6, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg9: f8E4M3FN, %arg10: f8E8M0FNU, %arg11: f32):
      %25 = "arith.extf"(%arg9) : (f8E4M3FN) -> f32
      %26 = "arith.bitcast"(%arg10) : (f8E8M0FNU) -> i8
      %27 = "arith.constant"() <{value = 23 : i32}> : () -> i32
      %28 = "arith.extui"(%26) : (i8) -> i32
      %29 = "arith.shli"(%28, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %30 = "arith.constant"() <{value = -1 : i8}> : () -> i8
      %31 = "arith.constant"() <{value = -1 : i32}> : () -> i32
      %32 = "arith.cmpi"(%26, %30) <{predicate = 0 : i64}> : (i8, i8) -> i1
      %33 = "arith.select"(%32, %31, %29) : (i1, i32, i32) -> i32
      %34 = "arith.bitcast"(%33) : (i32) -> f32
      %35 = "arith.mulf"(%25, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%35) : (f32) -> ()
    }) : (tensor<?x?xf8E4M3FN>, tensor<?x?xf8E8M0FNU>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %15 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %16 = "tensor.empty"(%2, %4) : (index, index) -> tensor<?x?xf32>
    %17 = "linalg.fill"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      "linalg.yield"(%arg7) : (f32) -> ()
    }) : (f32, tensor<?x?xf32>) -> tensor<?x?xf32>
    %18 = "util.optimization_barrier"(%13) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %19 = "util.optimization_barrier"(%14) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %20 = "flow.tensor.tie_shape"(%18, %2, %3) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %21 = "flow.tensor.tie_shape"(%19, %4, %3) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %22 = "linalg.matmul"(%20, %21, %17) <{indexing_maps = [#map3, #map4, #map5], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %23 = "arith.mulf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %24 = "arith.addf"(%arg6, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%24) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "func.return"(%22) : (tensor<?x?xf32>) -> ()
  }) : () -> ()
}) : () -> ()
