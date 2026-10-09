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
  spirv.func @iq1_xs_dispatch_0_elementwise_broadcast_D_f32() "None" attributes {gpu.known_block_size = array<i32: 64, 1, 1>, hal.executable.target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>} {
    %cst10_i32 = spirv.Constant 10 : i32
    %cst32_i32 = spirv.Constant 32 : i32
    %cst64_i32 = spirv.Constant 64 : i32
    %cst28_i32 = spirv.Constant 28 : i32
    %cst24_i32 = spirv.Constant 24 : i32
    %cst20_i32 = spirv.Constant 20 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %cst12_i32 = spirv.Constant 12 : i32
    %cst15_i32 = spirv.Constant 15 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst3_i32 = spirv.Constant 3 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst2_i32 = spirv.Constant 2 : i32
    %cst1_i32 = spirv.Constant 1 : i32
    %cst7_i32 = spirv.Constant 7 : i32
    %cst42_i32 = spirv.Constant 42 : i32
    %cst34_i32 = spirv.Constant 34 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst255_i32 = spirv.Constant 255 : i32
    %cst46_i32 = spirv.Constant 46 : i32
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
        %49 = spirv.IMul %48, %cst46_i32 : i32
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
          %58 = spirv.IMul %56, %cst46_i32 : i32
          %59 = spirv.IAdd %49, %58 : i32
          %60 = spirv.IMul %53, %cst32_i32 : i32
          %61 = spirv.Variable : !spirv.ptr<f32, Function>
          spirv.mlir.loop {
            spirv.Branch ^bb1(%cst0_i32, %cst_f32_1 : i32, f32)
          ^bb1(%217: i32, %218: f32):  // 2 preds: ^bb0, ^bb2
            %219 = spirv.SLessThan %217, %cst32_i32 : i32
            spirv.BranchConditional %219, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %220 = spirv.IAdd %60, %217 : i32
            %221 = spirv.IMul %42, %11 : i32
            %222 = spirv.IAdd %221, %220 : i32
            %223 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %222] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %224 = spirv.Load "StorageBuffer" %223 : f32
            %225 = spirv.FAdd %218, %224 : f32
            spirv.Store "Function" %61, %225 : f32
            %226 = spirv.IAdd %217, %cst1_i32 : i32
            spirv.Branch ^bb1(%226, %225 : i32, f32)
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
          %92 = spirv.IAdd %91, %57 : i32
          %93 = spirv.SDiv %92, %cst4_i32 : i32
          %94 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %93] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %95 = spirv.Load "StorageBuffer" %94 : i32
          %96 = spirv.UMod %92, %cst4_i32 : i32
          %97 = spirv.IMul %96, %cst8_i32 : i32
          %98 = spirv.ShiftRightArithmetic %95, %97 : i32, i32
          %99 = spirv.BitwiseAnd %98, %cst255_i32 : i32
          %100 = spirv.ShiftLeftLogical %99, %cst24_i32 : i32, i32
          %101 = spirv.ShiftRightArithmetic %100, %cst24_i32 : i32, i32
          %102 = spirv.BitwiseAnd %101, %cst255_i32 : i32
          %103 = spirv.BitwiseAnd %102, %cst255_i32 : i32
          %104 = spirv.IAdd %59, %cst42_i32 : i32
          %105 = spirv.UDiv %57, %cst2_i32 : i32
          %106 = spirv.IAdd %104, %105 : i32
          %107 = spirv.SDiv %106, %cst4_i32 : i32
          %108 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %107] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %109 = spirv.Load "StorageBuffer" %108 : i32
          %110 = spirv.UMod %106, %cst4_i32 : i32
          %111 = spirv.IMul %110, %cst8_i32 : i32
          %112 = spirv.ShiftRightArithmetic %109, %111 : i32, i32
          %113 = spirv.BitwiseAnd %112, %cst255_i32 : i32
          %114 = spirv.ShiftLeftLogical %113, %cst24_i32 : i32, i32
          %115 = spirv.ShiftRightArithmetic %114, %cst24_i32 : i32, i32
          %116 = spirv.BitwiseAnd %115, %cst255_i32 : i32
          %117 = spirv.BitwiseAnd %116, %cst255_i32 : i32
          %118 = spirv.UMod %53, %cst2_i32 : i32
          %119 = spirv.IMul %118, %cst4_i32 : i32
          %120 = spirv.ShiftRightLogical %117, %119 : i32, i32
          %121 = spirv.BitwiseAnd %120, %cst7_i32 : i32
          %122 = spirv.ShiftRightLogical %120, %cst3_i32 : i32, i32
          %123 = spirv.BitwiseAnd %122, %cst1_i32 : i32
          %124 = spirv.INotEqual %123, %cst0_i32 : i32
          %125 = spirv.Select %124, %cst_f32_0, %cst_f32 : i1, f32
          %126 = spirv.IMul %121, %cst2_i32 : i32
          %127 = spirv.IAdd %126, %cst1_i32 : i32
          %128 = spirv.ConvertUToF %127 : i32 to f32
          %129 = spirv.IMul %57, %cst4_i32 : i32
          %130 = spirv.IAdd %129, %cst2_i32 : i32
          %131 = spirv.IAdd %59, %130 : i32
          %132 = spirv.SDiv %131, %cst4_i32 : i32
          %133 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %132] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %134 = spirv.Load "StorageBuffer" %133 : i32
          %135 = spirv.UMod %131, %cst4_i32 : i32
          %136 = spirv.IMul %135, %cst8_i32 : i32
          %137 = spirv.ShiftRightArithmetic %134, %136 : i32, i32
          %138 = spirv.BitwiseAnd %137, %cst255_i32 : i32
          %139 = spirv.ShiftLeftLogical %138, %cst24_i32 : i32, i32
          %140 = spirv.ShiftRightArithmetic %139, %cst24_i32 : i32, i32
          %141 = spirv.BitwiseAnd %140, %cst255_i32 : i32
          %142 = spirv.BitwiseAnd %141, %cst255_i32 : i32
          %143 = spirv.BitwiseAnd %102, %cst3_i32 : i32
          %144 = spirv.ShiftLeftLogical %143, %cst8_i32 : i32, i32
          %145 = spirv.BitwiseOr %142, %144 : i32
          %146 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %145] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %147 = spirv.Load "StorageBuffer" %146 : i32
          %148 = spirv.IAdd %131, %cst1_i32 : i32
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
          %160 = spirv.ShiftRightLogical %103, %cst2_i32 : i32, i32
          %161 = spirv.BitwiseAnd %160, %cst3_i32 : i32
          %162 = spirv.ShiftLeftLogical %161, %cst8_i32 : i32, i32
          %163 = spirv.BitwiseOr %159, %162 : i32
          %164 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %163] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %165 = spirv.Load "StorageBuffer" %164 : i32
          %166 = spirv.IAdd %131, %cst2_i32 : i32
          %167 = spirv.SDiv %166, %cst4_i32 : i32
          %168 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %167] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %169 = spirv.Load "StorageBuffer" %168 : i32
          %170 = spirv.UMod %166, %cst4_i32 : i32
          %171 = spirv.IMul %170, %cst8_i32 : i32
          %172 = spirv.ShiftRightArithmetic %169, %171 : i32, i32
          %173 = spirv.BitwiseAnd %172, %cst255_i32 : i32
          %174 = spirv.ShiftLeftLogical %173, %cst24_i32 : i32, i32
          %175 = spirv.ShiftRightArithmetic %174, %cst24_i32 : i32, i32
          %176 = spirv.BitwiseAnd %175, %cst255_i32 : i32
          %177 = spirv.BitwiseAnd %176, %cst255_i32 : i32
          %178 = spirv.ShiftRightLogical %103, %cst4_i32 : i32, i32
          %179 = spirv.BitwiseAnd %178, %cst3_i32 : i32
          %180 = spirv.ShiftLeftLogical %179, %cst8_i32 : i32, i32
          %181 = spirv.BitwiseOr %177, %180 : i32
          %182 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %181] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %183 = spirv.Load "StorageBuffer" %182 : i32
          %184 = spirv.IAdd %131, %cst3_i32 : i32
          %185 = spirv.SDiv %184, %cst4_i32 : i32
          %186 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %185] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %187 = spirv.Load "StorageBuffer" %186 : i32
          %188 = spirv.UMod %184, %cst4_i32 : i32
          %189 = spirv.IMul %188, %cst8_i32 : i32
          %190 = spirv.ShiftRightArithmetic %187, %189 : i32, i32
          %191 = spirv.BitwiseAnd %190, %cst255_i32 : i32
          %192 = spirv.ShiftLeftLogical %191, %cst24_i32 : i32, i32
          %193 = spirv.ShiftRightArithmetic %192, %cst24_i32 : i32, i32
          %194 = spirv.BitwiseAnd %193, %cst255_i32 : i32
          %195 = spirv.BitwiseAnd %194, %cst255_i32 : i32
          %196 = spirv.ShiftRightLogical %103, %cst6_i32 : i32, i32
          %197 = spirv.BitwiseAnd %196, %cst3_i32 : i32
          %198 = spirv.ShiftLeftLogical %197, %cst8_i32 : i32, i32
          %199 = spirv.BitwiseOr %195, %198 : i32
          %200 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %199] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %201 = spirv.Load "StorageBuffer" %200 : i32
          %202 = spirv.IAdd %60, %cst4_i32 : i32
          %203 = spirv.IAdd %60, %cst8_i32 : i32
          %204 = spirv.IAdd %60, %cst12_i32 : i32
          %205 = spirv.IAdd %60, %cst16_i32 : i32
          %206 = spirv.IAdd %60, %cst20_i32 : i32
          %207 = spirv.IAdd %60, %cst24_i32 : i32
          %208 = spirv.IAdd %60, %cst28_i32 : i32
          %209 = spirv.Variable : !spirv.ptr<f32, Function>
          spirv.mlir.loop {
            spirv.Branch ^bb1(%cst0_i32, %cst_f32_1 : i32, f32)
          ^bb1(%217: i32, %218: f32):  // 2 preds: ^bb0, ^bb2
            %219 = spirv.SLessThan %217, %cst4_i32 : i32
            spirv.BranchConditional %219, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %220 = spirv.IMul %217, %cst8_i32 : i32
            %221 = spirv.ShiftRightLogical %147, %220 : i32, i32
            %222 = spirv.IAdd %60, %217 : i32
            %223 = spirv.IMul %42, %11 : i32
            %224 = spirv.IAdd %223, %222 : i32
            %225 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %224] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %226 = spirv.Load "StorageBuffer" %225 : f32
            %227 = spirv.BitwiseAnd %221, %cst15_i32 : i32
            %228 = spirv.ConvertUToF %227 : i32 to f32
            %229 = spirv.FMul %226, %228 : f32
            %230 = spirv.IAdd %202, %217 : i32
            %231 = spirv.IAdd %223, %230 : i32
            %232 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %231] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %233 = spirv.Load "StorageBuffer" %232 : f32
            %234 = spirv.ShiftRightLogical %221, %cst4_i32 : i32, i32
            %235 = spirv.BitwiseAnd %234, %cst15_i32 : i32
            %236 = spirv.ConvertUToF %235 : i32 to f32
            %237 = spirv.FMul %233, %236 : f32
            %238 = spirv.ShiftRightLogical %165, %220 : i32, i32
            %239 = spirv.IAdd %203, %217 : i32
            %240 = spirv.IAdd %223, %239 : i32
            %241 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %240] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %242 = spirv.Load "StorageBuffer" %241 : f32
            %243 = spirv.BitwiseAnd %238, %cst15_i32 : i32
            %244 = spirv.ConvertUToF %243 : i32 to f32
            %245 = spirv.FMul %242, %244 : f32
            %246 = spirv.IAdd %204, %217 : i32
            %247 = spirv.IAdd %223, %246 : i32
            %248 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %247] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %249 = spirv.Load "StorageBuffer" %248 : f32
            %250 = spirv.ShiftRightLogical %238, %cst4_i32 : i32, i32
            %251 = spirv.BitwiseAnd %250, %cst15_i32 : i32
            %252 = spirv.ConvertUToF %251 : i32 to f32
            %253 = spirv.FMul %249, %252 : f32
            %254 = spirv.ShiftRightLogical %183, %220 : i32, i32
            %255 = spirv.IAdd %205, %217 : i32
            %256 = spirv.IAdd %223, %255 : i32
            %257 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %256] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %258 = spirv.Load "StorageBuffer" %257 : f32
            %259 = spirv.BitwiseAnd %254, %cst15_i32 : i32
            %260 = spirv.ConvertUToF %259 : i32 to f32
            %261 = spirv.FMul %258, %260 : f32
            %262 = spirv.IAdd %206, %217 : i32
            %263 = spirv.IAdd %223, %262 : i32
            %264 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %263] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %265 = spirv.Load "StorageBuffer" %264 : f32
            %266 = spirv.ShiftRightLogical %254, %cst4_i32 : i32, i32
            %267 = spirv.BitwiseAnd %266, %cst15_i32 : i32
            %268 = spirv.ConvertUToF %267 : i32 to f32
            %269 = spirv.FMul %265, %268 : f32
            %270 = spirv.ShiftRightLogical %201, %220 : i32, i32
            %271 = spirv.IAdd %207, %217 : i32
            %272 = spirv.IAdd %223, %271 : i32
            %273 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %272] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %274 = spirv.Load "StorageBuffer" %273 : f32
            %275 = spirv.BitwiseAnd %270, %cst15_i32 : i32
            %276 = spirv.ConvertUToF %275 : i32 to f32
            %277 = spirv.FMul %274, %276 : f32
            %278 = spirv.IAdd %208, %217 : i32
            %279 = spirv.IAdd %223, %278 : i32
            %280 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %279] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %281 = spirv.Load "StorageBuffer" %280 : f32
            %282 = spirv.ShiftRightLogical %270, %cst4_i32 : i32, i32
            %283 = spirv.BitwiseAnd %282, %cst15_i32 : i32
            %284 = spirv.ConvertUToF %283 : i32 to f32
            %285 = spirv.FMul %281, %284 : f32
            %286 = spirv.FAdd %229, %237 : f32
            %287 = spirv.FAdd %286, %245 : f32
            %288 = spirv.FAdd %287, %253 : f32
            %289 = spirv.FAdd %288, %261 : f32
            %290 = spirv.FAdd %289, %269 : f32
            %291 = spirv.FAdd %290, %277 : f32
            %292 = spirv.FAdd %291, %285 : f32
            %293 = spirv.FAdd %218, %292 : f32
            spirv.Store "Function" %209, %293 : f32
            %294 = spirv.IAdd %217, %cst1_i32 : i32
            spirv.Branch ^bb1(%294, %293 : i32, f32)
          ^bb3:  // pred: ^bb1
            spirv.mlir.merge
          }
          %210 = spirv.Load "Function" %209 : f32
          %211 = spirv.FMul %62, %125 : f32
          %212 = spirv.FAdd %210, %211 : f32
          %213 = spirv.FMul %90, %212 : f32
          %214 = spirv.FMul %213, %128 : f32
          %215 = spirv.FAdd %54, %214 : f32
          spirv.Store "Function" %50, %215 : f32
          %216 = spirv.IAdd %53, %cst32_i32 : i32
          spirv.Branch ^bb1(%216, %215 : i32, f32)
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
  spirv.EntryPoint "GLCompute" @iq1_xs_dispatch_0_elementwise_broadcast_D_f32, @__builtin__LocalInvocationId__, @__builtin__WorkgroupId__, @__builtin__NumWorkgroups__
  spirv.ExecutionMode @iq1_xs_dispatch_0_elementwise_broadcast_D_f32 "LocalSize", 64, 1, 1
  spirv.ExecutionMode @iq1_xs_dispatch_0_elementwise_broadcast_D_f32 "SignedZeroInfNanPreserve", 32
}