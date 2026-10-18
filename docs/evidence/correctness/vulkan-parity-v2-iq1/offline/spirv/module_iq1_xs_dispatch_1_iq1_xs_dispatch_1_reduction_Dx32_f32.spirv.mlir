spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, SignedZeroInfNanPreserve, Matrix], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_float_controls]> {
  spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<2 x i32, stride=4> [0]), Block>, PushConstant>
  spirv.GlobalVariable @__resource_var_0_0_ bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_1_ bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.func @iq1_xs_dispatch_1_reduction_Dx32_f32() "None" attributes {gpu.known_block_size = array<i32: 64, 1, 1>, hal.executable.target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>} {
    %cst1_i32 = spirv.Constant 1 : i32
    %cst32_i32 = spirv.Constant 32 : i32
    %cst0_i32 = spirv.Constant 0 : i32
    %cst64_i32 = spirv.Constant 64 : i32
    %cst_f32 = spirv.Constant 0.000000e+00 : f32
    %__push_constant_var___addr = spirv.mlir.addressof @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<2 x i32, stride=4> [0]), Block>, PushConstant>
    %0 = spirv.AccessChain %__push_constant_var___addr[%cst0_i32, %cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<2 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
    %1 = spirv.Load "PushConstant" %0 : i32
    %__resource_var_0_0__addr = spirv.mlir.addressof @__resource_var_0_0_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_1__addr = spirv.mlir.addressof @__resource_var_0_1_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
    %2 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
    %3 = spirv.CompositeExtract %2[0 : i32] : vector<3xi32>
    %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %4 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
    %5 = spirv.CompositeExtract %4[0 : i32] : vector<3xi32>
    %__builtin__NumWorkgroups___addr = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
    %6 = spirv.Load "Input" %__builtin__NumWorkgroups___addr : vector<3xi32>
    %7 = spirv.CompositeExtract %6[0 : i32] : vector<3xi32>
    %8 = spirv.IMul %5, %cst64_i32 : i32
    %9 = spirv.IMul %7, %cst64_i32 : i32
    spirv.mlir.loop {
      spirv.Branch ^bb1(%8 : i32)
    ^bb1(%10: i32):  // 2 preds: ^bb0, ^bb2
      %11 = spirv.SLessThan %10, %1 : i32
      spirv.BranchConditional %11, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %12 = spirv.ISub %1, %10 : i32
      %13 = spirv.GL.SMin %12, %cst64_i32 : i32
      %14 = spirv.SGreaterThan %13, %3 : i32
      spirv.mlir.selection {
        spirv.BranchConditional %14, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %16 = spirv.IAdd %10, %3 : i32
        %17 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %16] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %17, %cst_f32 : f32
        spirv.mlir.loop {
          spirv.Branch ^bb1(%cst0_i32 : i32)
        ^bb1(%18: i32):  // 2 preds: ^bb0, ^bb2
          %19 = spirv.SLessThan %18, %cst32_i32 : i32
          spirv.BranchConditional %19, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %20 = spirv.IMul %10, %cst32_i32 : i32
          %21 = spirv.IAdd %18, %20 : i32
          %22 = spirv.IMul %3, %cst32_i32 : i32
          %23 = spirv.IAdd %21, %22 : i32
          %24 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %23] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
          %25 = spirv.Load "StorageBuffer" %24 : f32
          %26 = spirv.Load "StorageBuffer" %17 : f32
          %27 = spirv.FAdd %25, %26 : f32
          spirv.Store "StorageBuffer" %17, %27 : f32
          %28 = spirv.IAdd %18, %cst1_i32 : i32
          spirv.Branch ^bb1(%28 : i32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %15 = spirv.IAdd %10, %9 : i32
      spirv.Branch ^bb1(%15 : i32)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @iq1_xs_dispatch_1_reduction_Dx32_f32, @__builtin__LocalInvocationId__, @__builtin__WorkgroupId__, @__builtin__NumWorkgroups__
  spirv.ExecutionMode @iq1_xs_dispatch_1_reduction_Dx32_f32 "LocalSize", 64, 1, 1
  spirv.ExecutionMode @iq1_xs_dispatch_1_reduction_Dx32_f32 "SignedZeroInfNanPreserve", 32
}