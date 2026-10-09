spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, SignedZeroInfNanPreserve, Matrix], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_float_controls]> {
  spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>
  spirv.GlobalVariable @__resource_var_0_1_ bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_3_ bind(0, 3) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_5_ bind(0, 5) : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_0_ bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_2_ bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_4_ bind(0, 4) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_6_ bind(0, 6) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.func @nf4_nested_f32_dispatch_0_elementwise_broadcast_D_f32() "None" attributes {gpu.known_block_size = array<i32: 64, 1, 1>, hal.executable.target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>} {
    %cst24_i32 = spirv.Constant 24 : i32
    %cst255_i32 = spirv.Constant 255 : i32
    %cst12_i32 = spirv.Constant 12 : i32
    %cst10_i32 = spirv.Constant 10 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst0_i32 = spirv.Constant 0 : i32
    %cst64_i32 = spirv.Constant 64 : i32
    %cst256_i32 = spirv.Constant 256 : i32
    %cst15_i32 = spirv.Constant 15 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst2_i32 = spirv.Constant 2 : i32
    %__push_constant_var___addr = spirv.mlir.addressof @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>
    %0 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %1 = spirv.Load "PushConstant" %0 : i32
    %2 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst2_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %3 = spirv.Load "PushConstant" %2 : i32
    %4 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst4_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %5 = spirv.Load "PushConstant" %4 : i32
    %6 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst6_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %7 = spirv.Load "PushConstant" %6 : i32
    %8 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst8_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %9 = spirv.Load "PushConstant" %8 : i32
    %10 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst10_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %11 = spirv.Load "PushConstant" %10 : i32
    %12 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst12_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %13 = spirv.Load "PushConstant" %12 : i32
    %__resource_var_0_1__addr = spirv.mlir.addressof @__resource_var_0_1_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_3__addr = spirv.mlir.addressof @__resource_var_0_3_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_5__addr = spirv.mlir.addressof @__resource_var_0_5_ : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_0__addr = spirv.mlir.addressof @__resource_var_0_0_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_2__addr = spirv.mlir.addressof @__resource_var_0_2_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_4__addr = spirv.mlir.addressof @__resource_var_0_4_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_6__addr = spirv.mlir.addressof @__resource_var_0_6_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
    %14 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
    %15 = spirv.CompositeExtract %14[0 : i32] : vector<3xi32>
    %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %16 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
    %17 = spirv.CompositeExtract %16[0 : i32] : vector<3xi32>
    %__builtin__NumWorkgroups___addr = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
    %18 = spirv.Load "Input" %__builtin__NumWorkgroups___addr : vector<3xi32>
    %19 = spirv.CompositeExtract %18[0 : i32] : vector<3xi32>
    %20 = spirv.IMul %17, %cst64_i32 : i32
    %21 = spirv.IMul %19, %cst64_i32 : i32
    spirv.mlir.loop {
      spirv.Branch ^bb1(%20 : i32)
    ^bb1(%22: i32):  // 2 preds: ^bb0, ^bb2
      %23 = spirv.SLessThan %22, %1 : i32
      spirv.BranchConditional %23, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %24 = spirv.ISub %1, %22 : i32
      %25 = spirv.GL.SMin %24, %cst64_i32 : i32
      %26 = spirv.SGreaterThan %25, %15 : i32
      spirv.mlir.selection {
        spirv.BranchConditional %26, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %28 = spirv.IAdd %22, %15 : i32
        %29 = spirv.GL.SAbs %28 : i32
        %30 = spirv.GL.SAbs %3 : i32
        %31 = spirv.UMod %29, %30 : i32
        %32 = spirv.IEqual %28, %29 : i32
        %33 = spirv.SNegate %31 : i32
        %34 = spirv.Select %32, %31, %33 : i1, i32
        %35 = spirv.SDiv %28, %3 : i32
        %36 = spirv.UDiv %34, %cst2_i32 : i32
        %37 = spirv.UMod %34, %cst2_i32 : i32
        %38 = spirv.IMul %35, %9 : i32
        %39 = spirv.IAdd %38, %36 : i32
        %40 = spirv.SDiv %39, %cst4_i32 : i32
        %41 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %40] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %42 = spirv.Load "StorageBuffer" %41 : i32
        %43 = spirv.UMod %39, %cst4_i32 : i32
        %44 = spirv.IMul %43, %cst8_i32 : i32
        %45 = spirv.ShiftRightArithmetic %42, %44 : i32, i32
        %46 = spirv.BitwiseAnd %45, %cst255_i32 : i32
        %47 = spirv.ShiftLeftLogical %46, %cst24_i32 : i32, i32
        %48 = spirv.ShiftRightArithmetic %47, %cst24_i32 : i32, i32
        %49 = spirv.BitwiseAnd %48, %cst255_i32 : i32
        %50 = spirv.IEqual %37, %cst0_i32 : i32
        %51 = spirv.ShiftRightLogical %49, %cst4_i32 : i32, i32
        %52 = spirv.Select %50, %51, %49 : i1, i32
        %53 = spirv.BitwiseAnd %52, %cst15_i32 : i32
        %54 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %53] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        %55 = spirv.Load "StorageBuffer" %54 : f32
        %56 = spirv.IMul %35, %3 : i32
        %57 = spirv.IAdd %56, %34 : i32
        %58 = spirv.UDiv %57, %5 : i32
        %59 = spirv.UDiv %58, %cst256_i32 : i32
        %60 = spirv.SDiv %58, %cst4_i32 : i32
        %61 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %60] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %62 = spirv.Load "StorageBuffer" %61 : i32
        %63 = spirv.UMod %58, %cst4_i32 : i32
        %64 = spirv.IMul %63, %cst8_i32 : i32
        %65 = spirv.ShiftRightArithmetic %62, %64 : i32, i32
        %66 = spirv.BitwiseAnd %65, %cst255_i32 : i32
        %67 = spirv.ShiftLeftLogical %66, %cst24_i32 : i32, i32
        %68 = spirv.ShiftRightArithmetic %67, %cst24_i32 : i32, i32
        %69 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %68] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        %70 = spirv.Load "StorageBuffer" %69 : f32
        %71 = spirv.AccessChain %__resource_var_0_4__addr[%cst0_i32, %59] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        %72 = spirv.Load "StorageBuffer" %71 : f32
        %73 = spirv.InBoundsAccessChain %__resource_var_0_5__addr[%cst0_i32, %cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        %74 = spirv.Load "StorageBuffer" %73 : f32
        %75 = spirv.FMul %70, %72 : f32
        %76 = spirv.FAdd %75, %74 : f32
        %77 = spirv.FMul %55, %76 : f32
        %78 = spirv.AccessChain %__resource_var_0_6__addr[%cst0_i32, %28] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %78, %77 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %27 = spirv.IAdd %22, %21 : i32
      spirv.Branch ^bb1(%27 : i32)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @nf4_nested_f32_dispatch_0_elementwise_broadcast_D_f32, @__builtin__LocalInvocationId__, @__builtin__WorkgroupId__, @__builtin__NumWorkgroups__
  spirv.ExecutionMode @nf4_nested_f32_dispatch_0_elementwise_broadcast_D_f32 "LocalSize", 64, 1, 1
  spirv.ExecutionMode @nf4_nested_f32_dispatch_0_elementwise_broadcast_D_f32 "SignedZeroInfNanPreserve", 32
}