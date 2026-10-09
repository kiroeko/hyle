spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Int16, Float16, SignedZeroInfNanPreserve, Matrix], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_float_controls]> {
  spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<14 x i32, stride=4> [0]), Block>, PushConstant>
  spirv.GlobalVariable @__resource_var_0_3_ bind(0, 3) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_0_ bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_1_ bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_2_ bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_4_ bind(0, 4) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.func @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32() "None" attributes {gpu.known_block_size = array<i32: 64, 1, 1>, hal.executable.target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>} {
    %cst10_i32 = spirv.Constant 10 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst1536_i32 = spirv.Constant 1536 : i32
    %cst32_i32 = spirv.Constant 32 : i32
    %cst64_i32 = spirv.Constant 64 : i32
    %cst28_i32 = spirv.Constant 28 : i32
    %cst24_i32 = spirv.Constant 24 : i32
    %cst20_i32 = spirv.Constant 20 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %cst12_i32 = spirv.Constant 12 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst15_i32 = spirv.Constant 15 : i32
    %cst3_i32 = spirv.Constant 3 : i32
    %cst2_i32 = spirv.Constant 2 : i32
    %cst1_i32 = spirv.Constant 1 : i32
    %cst7_i32 = spirv.Constant 7 : i32
    %cst34_i32 = spirv.Constant 34 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst255_i32 = spirv.Constant 255 : i32
    %cst38_i32 = spirv.Constant 38 : i32
    %cst256_i32 = spirv.Constant 256 : i32
    %cst_f32 = spirv.Constant -8.750000e-01 : f32
    %cst_f32_0 = spirv.Constant -1.125000e+00 : f32
    %cst_f32_1 = spirv.Constant 0.000000e+00 : f32
    %cst0_i32 = spirv.Constant 0 : i32
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
    %__resource_var_0_3__addr = spirv.mlir.addressof @__resource_var_0_3_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_0__addr = spirv.mlir.addressof @__resource_var_0_0_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_1__addr = spirv.mlir.addressof @__resource_var_0_1_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_2__addr = spirv.mlir.addressof @__resource_var_0_2_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %__resource_var_0_4__addr = spirv.mlir.addressof @__resource_var_0_4_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
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
        %30 = spirv.GL.SAbs %cst32_i32 : i32
        %31 = spirv.UMod %29, %30 : i32
        %32 = spirv.IEqual %28, %29 : i32
        %33 = spirv.SNegate %31 : i32
        %34 = spirv.Select %32, %31, %33 : i1, i32
        %35 = spirv.SDiv %28, %cst32_i32 : i32
        %36 = spirv.GL.SAbs %35 : i32
        %37 = spirv.GL.SAbs %3 : i32
        %38 = spirv.UMod %36, %37 : i32
        %39 = spirv.IEqual %35, %36 : i32
        %40 = spirv.SNegate %38 : i32
        %41 = spirv.Select %39, %38, %40 : i1, i32
        %42 = spirv.SDiv %35, %3 : i32
        %43 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %42] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %44 = spirv.Load "StorageBuffer" %43 : i32
        %45 = spirv.IMul %44, %3 : i32
        %46 = spirv.IAdd %45, %41 : i32
        %47 = spirv.UDiv %11, %cst256_i32 : i32
        %48 = spirv.IMul %46, %47 : i32
        %49 = spirv.IMul %48, %cst38_i32 : i32
        %50 = spirv.Variable : !spirv.ptr<f32, Function>
        spirv.Store "Function" %50, %cst_f32_1 : f32
        spirv.mlir.loop {
          spirv.Branch ^bb1(%34, %cst_f32_1 : i32, f32)
        ^bb1(%53: i32, %54: f32):  // 2 preds: ^bb0, ^bb2
          %55 = spirv.SLessThan %53, %5 : i32
          spirv.BranchConditional %55, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %56 = spirv.UDiv %53, %cst8_i32 : i32
          %57 = spirv.UMod %53, %cst8_i32 : i32
          %58 = spirv.IMul %56, %cst38_i32 : i32
          %59 = spirv.IAdd %49, %58 : i32
          %60 = spirv.IMul %53, %cst32_i32 : i32
          %61 = spirv.Variable : !spirv.ptr<f32, Function>
          spirv.mlir.loop {
            spirv.Branch ^bb1(%cst0_i32, %cst_f32_1 : i32, f32)
          ^bb1(%193: i32, %194: f32):  // 2 preds: ^bb0, ^bb2
            %195 = spirv.SLessThan %193, %cst32_i32 : i32
            spirv.BranchConditional %195, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %196 = spirv.IAdd %60, %193 : i32
            %197 = spirv.IMul %42, %11 : i32
            %198 = spirv.IAdd %197, %196 : i32
            %199 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %198] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %200 = spirv.Load "StorageBuffer" %199 : f32
            %201 = spirv.FAdd %194, %200 : f32
            spirv.Store "Function" %61, %201 : f32
            %202 = spirv.IAdd %193, %cst1_i32 : i32
            spirv.Branch ^bb1(%202, %201 : i32, f32)
          ^bb3:  // pred: ^bb1
            spirv.mlir.merge
          }
          %62 = spirv.Load "Function" %61 : f32
          %63 = spirv.SDiv %59, %cst4_i32 : i32
          %64 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %63] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %65 = spirv.Load "StorageBuffer" %64 : i32
          %66 = spirv.UMod %59, %cst4_i32 : i32
          %67 = spirv.IMul %66, %cst8_i32 : i32
          %68 = spirv.ShiftRightArithmetic %65, %67 : i32, i32
          %69 = spirv.BitwiseAnd %68, %cst255_i32 : i32
          %70 = spirv.ShiftLeftLogical %69, %cst24_i32 : i32, i32
          %71 = spirv.ShiftRightArithmetic %70, %cst24_i32 : i32, i32
          %72 = spirv.IAdd %59, %cst1_i32 : i32
          %73 = spirv.SDiv %72, %cst4_i32 : i32
          %74 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %73] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %75 = spirv.Load "StorageBuffer" %74 : i32
          %76 = spirv.UMod %72, %cst4_i32 : i32
          %77 = spirv.IMul %76, %cst8_i32 : i32
          %78 = spirv.ShiftRightArithmetic %75, %77 : i32, i32
          %79 = spirv.BitwiseAnd %78, %cst255_i32 : i32
          %80 = spirv.ShiftLeftLogical %79, %cst24_i32 : i32, i32
          %81 = spirv.ShiftRightArithmetic %80, %cst24_i32 : i32, i32
          %82 = spirv.BitwiseAnd %71, %cst255_i32 : i32
          %83 = spirv.BitwiseAnd %81, %cst255_i32 : i32
          %84 = spirv.BitwiseAnd %82, %cst255_i32 : i32
          %85 = spirv.BitwiseAnd %83, %cst255_i32 : i32
          %86 = spirv.ShiftLeftLogical %85, %cst8_i32 : i32, i32
          %87 = spirv.BitwiseOr %84, %86 : i32
          %88 = spirv.SConvert %87 : i32 to i16
          %89 = spirv.Bitcast %88 : i16 to f16
          %90 = spirv.FConvert %89 : f16 to f32
          %91 = spirv.IAdd %59, %cst34_i32 : i32
          %92 = spirv.UDiv %57, %cst2_i32 : i32
          %93 = spirv.IAdd %91, %92 : i32
          %94 = spirv.SDiv %93, %cst4_i32 : i32
          %95 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %94] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %96 = spirv.Load "StorageBuffer" %95 : i32
          %97 = spirv.UMod %93, %cst4_i32 : i32
          %98 = spirv.IMul %97, %cst8_i32 : i32
          %99 = spirv.ShiftRightArithmetic %96, %98 : i32, i32
          %100 = spirv.BitwiseAnd %99, %cst255_i32 : i32
          %101 = spirv.ShiftLeftLogical %100, %cst24_i32 : i32, i32
          %102 = spirv.ShiftRightArithmetic %101, %cst24_i32 : i32, i32
          %103 = spirv.BitwiseAnd %102, %cst255_i32 : i32
          %104 = spirv.BitwiseAnd %103, %cst255_i32 : i32
          %105 = spirv.UMod %53, %cst2_i32 : i32
          %106 = spirv.IMul %105, %cst4_i32 : i32
          %107 = spirv.ShiftRightLogical %104, %106 : i32, i32
          %108 = spirv.BitwiseAnd %107, %cst7_i32 : i32
          %109 = spirv.ShiftRightLogical %107, %cst3_i32 : i32, i32
          %110 = spirv.BitwiseAnd %109, %cst1_i32 : i32
          %111 = spirv.INotEqual %110, %cst0_i32 : i32
          %112 = spirv.Select %111, %cst_f32_0, %cst_f32 : i1, f32
          %113 = spirv.IMul %108, %cst2_i32 : i32
          %114 = spirv.IAdd %113, %cst1_i32 : i32
          %115 = spirv.ConvertUToF %114 : i32 to f32
          %116 = spirv.IMul %57, %cst4_i32 : i32
          %117 = spirv.IAdd %116, %cst2_i32 : i32
          %118 = spirv.IAdd %59, %117 : i32
          %119 = spirv.SDiv %118, %cst4_i32 : i32
          %120 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %119] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %121 = spirv.Load "StorageBuffer" %120 : i32
          %122 = spirv.UMod %118, %cst4_i32 : i32
          %123 = spirv.IMul %122, %cst8_i32 : i32
          %124 = spirv.ShiftRightArithmetic %121, %123 : i32, i32
          %125 = spirv.BitwiseAnd %124, %cst255_i32 : i32
          %126 = spirv.ShiftLeftLogical %125, %cst24_i32 : i32, i32
          %127 = spirv.ShiftRightArithmetic %126, %cst24_i32 : i32, i32
          %128 = spirv.BitwiseAnd %127, %cst255_i32 : i32
          %129 = spirv.BitwiseAnd %128, %cst255_i32 : i32
          %130 = spirv.IAdd %129, %cst1536_i32 : i32
          %131 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %130] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %132 = spirv.Load "StorageBuffer" %131 : i32
          %133 = spirv.IAdd %118, %cst1_i32 : i32
          %134 = spirv.SDiv %133, %cst4_i32 : i32
          %135 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %134] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %136 = spirv.Load "StorageBuffer" %135 : i32
          %137 = spirv.UMod %133, %cst4_i32 : i32
          %138 = spirv.IMul %137, %cst8_i32 : i32
          %139 = spirv.ShiftRightArithmetic %136, %138 : i32, i32
          %140 = spirv.BitwiseAnd %139, %cst255_i32 : i32
          %141 = spirv.ShiftLeftLogical %140, %cst24_i32 : i32, i32
          %142 = spirv.ShiftRightArithmetic %141, %cst24_i32 : i32, i32
          %143 = spirv.BitwiseAnd %142, %cst255_i32 : i32
          %144 = spirv.BitwiseAnd %143, %cst255_i32 : i32
          %145 = spirv.IAdd %144, %cst1536_i32 : i32
          %146 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %145] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %147 = spirv.Load "StorageBuffer" %146 : i32
          %148 = spirv.IAdd %118, %cst2_i32 : i32
          %149 = spirv.SDiv %148, %cst4_i32 : i32
          %150 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %149] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %151 = spirv.Load "StorageBuffer" %150 : i32
          %152 = spirv.UMod %148, %cst4_i32 : i32
          %153 = spirv.IMul %152, %cst8_i32 : i32
          %154 = spirv.ShiftRightArithmetic %151, %153 : i32, i32
          %155 = spirv.BitwiseAnd %154, %cst255_i32 : i32
          %156 = spirv.ShiftLeftLogical %155, %cst24_i32 : i32, i32
          %157 = spirv.ShiftRightArithmetic %156, %cst24_i32 : i32, i32
          %158 = spirv.BitwiseAnd %157, %cst255_i32 : i32
          %159 = spirv.BitwiseAnd %158, %cst255_i32 : i32
          %160 = spirv.IAdd %159, %cst1536_i32 : i32
          %161 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %160] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %162 = spirv.Load "StorageBuffer" %161 : i32
          %163 = spirv.IAdd %118, %cst3_i32 : i32
          %164 = spirv.SDiv %163, %cst4_i32 : i32
          %165 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %164] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %166 = spirv.Load "StorageBuffer" %165 : i32
          %167 = spirv.UMod %163, %cst4_i32 : i32
          %168 = spirv.IMul %167, %cst8_i32 : i32
          %169 = spirv.ShiftRightArithmetic %166, %168 : i32, i32
          %170 = spirv.BitwiseAnd %169, %cst255_i32 : i32
          %171 = spirv.ShiftLeftLogical %170, %cst24_i32 : i32, i32
          %172 = spirv.ShiftRightArithmetic %171, %cst24_i32 : i32, i32
          %173 = spirv.BitwiseAnd %172, %cst255_i32 : i32
          %174 = spirv.BitwiseAnd %173, %cst255_i32 : i32
          %175 = spirv.IAdd %174, %cst1536_i32 : i32
          %176 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %175] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %177 = spirv.Load "StorageBuffer" %176 : i32
          %178 = spirv.IAdd %60, %cst4_i32 : i32
          %179 = spirv.IAdd %60, %cst8_i32 : i32
          %180 = spirv.IAdd %60, %cst12_i32 : i32
          %181 = spirv.IAdd %60, %cst16_i32 : i32
          %182 = spirv.IAdd %60, %cst20_i32 : i32
          %183 = spirv.IAdd %60, %cst24_i32 : i32
          %184 = spirv.IAdd %60, %cst28_i32 : i32
          %185 = spirv.Variable : !spirv.ptr<f32, Function>
          spirv.mlir.loop {
            spirv.Branch ^bb1(%cst0_i32, %cst_f32_1 : i32, f32)
          ^bb1(%193: i32, %194: f32):  // 2 preds: ^bb0, ^bb2
            %195 = spirv.SLessThan %193, %cst4_i32 : i32
            spirv.BranchConditional %195, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %196 = spirv.IMul %193, %cst8_i32 : i32
            %197 = spirv.ShiftRightLogical %132, %196 : i32, i32
            %198 = spirv.IAdd %60, %193 : i32
            %199 = spirv.IMul %42, %11 : i32
            %200 = spirv.IAdd %199, %198 : i32
            %201 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %200] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %202 = spirv.Load "StorageBuffer" %201 : f32
            %203 = spirv.BitwiseAnd %197, %cst15_i32 : i32
            %204 = spirv.ConvertUToF %203 : i32 to f32
            %205 = spirv.FMul %202, %204 : f32
            %206 = spirv.IAdd %178, %193 : i32
            %207 = spirv.IAdd %199, %206 : i32
            %208 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %207] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %209 = spirv.Load "StorageBuffer" %208 : f32
            %210 = spirv.ShiftRightLogical %197, %cst4_i32 : i32, i32
            %211 = spirv.BitwiseAnd %210, %cst15_i32 : i32
            %212 = spirv.ConvertUToF %211 : i32 to f32
            %213 = spirv.FMul %209, %212 : f32
            %214 = spirv.ShiftRightLogical %147, %196 : i32, i32
            %215 = spirv.IAdd %179, %193 : i32
            %216 = spirv.IAdd %199, %215 : i32
            %217 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %216] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %218 = spirv.Load "StorageBuffer" %217 : f32
            %219 = spirv.BitwiseAnd %214, %cst15_i32 : i32
            %220 = spirv.ConvertUToF %219 : i32 to f32
            %221 = spirv.FMul %218, %220 : f32
            %222 = spirv.IAdd %180, %193 : i32
            %223 = spirv.IAdd %199, %222 : i32
            %224 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %223] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %225 = spirv.Load "StorageBuffer" %224 : f32
            %226 = spirv.ShiftRightLogical %214, %cst4_i32 : i32, i32
            %227 = spirv.BitwiseAnd %226, %cst15_i32 : i32
            %228 = spirv.ConvertUToF %227 : i32 to f32
            %229 = spirv.FMul %225, %228 : f32
            %230 = spirv.ShiftRightLogical %162, %196 : i32, i32
            %231 = spirv.IAdd %181, %193 : i32
            %232 = spirv.IAdd %199, %231 : i32
            %233 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %232] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %234 = spirv.Load "StorageBuffer" %233 : f32
            %235 = spirv.BitwiseAnd %230, %cst15_i32 : i32
            %236 = spirv.ConvertUToF %235 : i32 to f32
            %237 = spirv.FMul %234, %236 : f32
            %238 = spirv.IAdd %182, %193 : i32
            %239 = spirv.IAdd %199, %238 : i32
            %240 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %239] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %241 = spirv.Load "StorageBuffer" %240 : f32
            %242 = spirv.ShiftRightLogical %230, %cst4_i32 : i32, i32
            %243 = spirv.BitwiseAnd %242, %cst15_i32 : i32
            %244 = spirv.ConvertUToF %243 : i32 to f32
            %245 = spirv.FMul %241, %244 : f32
            %246 = spirv.ShiftRightLogical %177, %196 : i32, i32
            %247 = spirv.IAdd %183, %193 : i32
            %248 = spirv.IAdd %199, %247 : i32
            %249 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %248] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %250 = spirv.Load "StorageBuffer" %249 : f32
            %251 = spirv.BitwiseAnd %246, %cst15_i32 : i32
            %252 = spirv.ConvertUToF %251 : i32 to f32
            %253 = spirv.FMul %250, %252 : f32
            %254 = spirv.IAdd %184, %193 : i32
            %255 = spirv.IAdd %199, %254 : i32
            %256 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %255] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %257 = spirv.Load "StorageBuffer" %256 : f32
            %258 = spirv.ShiftRightLogical %246, %cst4_i32 : i32, i32
            %259 = spirv.BitwiseAnd %258, %cst15_i32 : i32
            %260 = spirv.ConvertUToF %259 : i32 to f32
            %261 = spirv.FMul %257, %260 : f32
            %262 = spirv.FAdd %205, %213 : f32
            %263 = spirv.FAdd %262, %221 : f32
            %264 = spirv.FAdd %263, %229 : f32
            %265 = spirv.FAdd %264, %237 : f32
            %266 = spirv.FAdd %265, %245 : f32
            %267 = spirv.FAdd %266, %253 : f32
            %268 = spirv.FAdd %267, %261 : f32
            %269 = spirv.FAdd %194, %268 : f32
            spirv.Store "Function" %185, %269 : f32
            %270 = spirv.IAdd %193, %cst1_i32 : i32
            spirv.Branch ^bb1(%270, %269 : i32, f32)
          ^bb3:  // pred: ^bb1
            spirv.mlir.merge
          }
          %186 = spirv.Load "Function" %185 : f32
          %187 = spirv.FMul %62, %112 : f32
          %188 = spirv.FAdd %186, %187 : f32
          %189 = spirv.FMul %90, %188 : f32
          %190 = spirv.FMul %189, %115 : f32
          %191 = spirv.FAdd %54, %190 : f32
          spirv.Store "Function" %50, %191 : f32
          %192 = spirv.IAdd %53, %cst32_i32 : i32
          spirv.Branch ^bb1(%192, %191 : i32, f32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %51 = spirv.Load "Function" %50 : f32
        %52 = spirv.AccessChain %__resource_var_0_4__addr[%cst0_i32, %28] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %52, %51 : f32
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
  spirv.EntryPoint "GLCompute" @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32, @__builtin__LocalInvocationId__, @__builtin__WorkgroupId__, @__builtin__NumWorkgroups__
  spirv.ExecutionMode @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32 "LocalSize", 64, 1, 1
  spirv.ExecutionMode @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32 "SignedZeroInfNanPreserve", 32
}