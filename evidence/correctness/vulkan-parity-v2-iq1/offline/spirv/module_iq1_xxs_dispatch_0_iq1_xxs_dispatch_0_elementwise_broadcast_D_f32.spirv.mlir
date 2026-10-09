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
  spirv.func @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32() "None" attributes {gpu.known_block_size = array<i32: 64, 1, 1>, hal.executable.target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>} {
    %cst10_i32 = spirv.Constant 10 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst1024_i32 = spirv.Constant 1024 : i32
    %cst32_i32 = spirv.Constant 32 : i32
    %cst64_i32 = spirv.Constant 64 : i32
    %cst28_i32 = spirv.Constant 28 : i32
    %cst24_i32 = spirv.Constant 24 : i32
    %cst20_i32 = spirv.Constant 20 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %cst12_i32 = spirv.Constant 12 : i32
    %cst15_i32 = spirv.Constant 15 : i32
    %cst3_i32 = spirv.Constant 3 : i32
    %cst2_i32 = spirv.Constant 2 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst1_i32 = spirv.Constant 1 : i32
    %cst7_i32 = spirv.Constant 7 : i32
    %cst34_i32 = spirv.Constant 34 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst255_i32 = spirv.Constant 255 : i32
    %cst42_i32 = spirv.Constant 42 : i32
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
        %49 = spirv.IMul %48, %cst42_i32 : i32
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
          %58 = spirv.IMul %56, %cst42_i32 : i32
          %59 = spirv.IAdd %49, %58 : i32
          %60 = spirv.IMul %53, %cst32_i32 : i32
          %61 = spirv.Variable : !spirv.ptr<f32, Function>
          spirv.mlir.loop {
            spirv.Branch ^bb1(%cst0_i32, %cst_f32_1 : i32, f32)
          ^bb1(%204: i32, %205: f32):  // 2 preds: ^bb0, ^bb2
            %206 = spirv.SLessThan %204, %cst32_i32 : i32
            spirv.BranchConditional %206, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %207 = spirv.IAdd %60, %204 : i32
            %208 = spirv.IMul %42, %11 : i32
            %209 = spirv.IAdd %208, %207 : i32
            %210 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %209] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %211 = spirv.Load "StorageBuffer" %210 : f32
            %212 = spirv.FAdd %205, %211 : f32
            spirv.Store "Function" %61, %212 : f32
            %213 = spirv.IAdd %204, %cst1_i32 : i32
            spirv.Branch ^bb1(%213, %212 : i32, f32)
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
          %104 = spirv.ShiftRightLogical %103, %cst4_i32 : i32, i32
          %105 = spirv.BitwiseAnd %104, %cst7_i32 : i32
          %106 = spirv.ShiftRightLogical %103, %cst7_i32 : i32, i32
          %107 = spirv.INotEqual %106, %cst0_i32 : i32
          %108 = spirv.Select %107, %cst_f32_0, %cst_f32 : i1, f32
          %109 = spirv.IMul %105, %cst2_i32 : i32
          %110 = spirv.IAdd %109, %cst1_i32 : i32
          %111 = spirv.ConvertUToF %110 : i32 to f32
          %112 = spirv.IMul %57, %cst4_i32 : i32
          %113 = spirv.IAdd %112, %cst2_i32 : i32
          %114 = spirv.IAdd %59, %113 : i32
          %115 = spirv.SDiv %114, %cst4_i32 : i32
          %116 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %115] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %117 = spirv.Load "StorageBuffer" %116 : i32
          %118 = spirv.UMod %114, %cst4_i32 : i32
          %119 = spirv.IMul %118, %cst8_i32 : i32
          %120 = spirv.ShiftRightArithmetic %117, %119 : i32, i32
          %121 = spirv.BitwiseAnd %120, %cst255_i32 : i32
          %122 = spirv.ShiftLeftLogical %121, %cst24_i32 : i32, i32
          %123 = spirv.ShiftRightArithmetic %122, %cst24_i32 : i32, i32
          %124 = spirv.BitwiseAnd %123, %cst255_i32 : i32
          %125 = spirv.BitwiseAnd %124, %cst255_i32 : i32
          %126 = spirv.BitwiseAnd %102, %cst1_i32 : i32
          %127 = spirv.ShiftLeftLogical %126, %cst8_i32 : i32, i32
          %128 = spirv.BitwiseOr %125, %127 : i32
          %129 = spirv.IAdd %128, %cst1024_i32 : i32
          %130 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %129] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %131 = spirv.Load "StorageBuffer" %130 : i32
          %132 = spirv.IAdd %114, %cst1_i32 : i32
          %133 = spirv.SDiv %132, %cst4_i32 : i32
          %134 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %133] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %135 = spirv.Load "StorageBuffer" %134 : i32
          %136 = spirv.UMod %132, %cst4_i32 : i32
          %137 = spirv.IMul %136, %cst8_i32 : i32
          %138 = spirv.ShiftRightArithmetic %135, %137 : i32, i32
          %139 = spirv.BitwiseAnd %138, %cst255_i32 : i32
          %140 = spirv.ShiftLeftLogical %139, %cst24_i32 : i32, i32
          %141 = spirv.ShiftRightArithmetic %140, %cst24_i32 : i32, i32
          %142 = spirv.BitwiseAnd %141, %cst255_i32 : i32
          %143 = spirv.BitwiseAnd %142, %cst255_i32 : i32
          %144 = spirv.ShiftRightLogical %103, %cst1_i32 : i32, i32
          %145 = spirv.BitwiseAnd %144, %cst1_i32 : i32
          %146 = spirv.ShiftLeftLogical %145, %cst8_i32 : i32, i32
          %147 = spirv.BitwiseOr %143, %146 : i32
          %148 = spirv.IAdd %147, %cst1024_i32 : i32
          %149 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %148] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %150 = spirv.Load "StorageBuffer" %149 : i32
          %151 = spirv.IAdd %114, %cst2_i32 : i32
          %152 = spirv.SDiv %151, %cst4_i32 : i32
          %153 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %152] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %154 = spirv.Load "StorageBuffer" %153 : i32
          %155 = spirv.UMod %151, %cst4_i32 : i32
          %156 = spirv.IMul %155, %cst8_i32 : i32
          %157 = spirv.ShiftRightArithmetic %154, %156 : i32, i32
          %158 = spirv.BitwiseAnd %157, %cst255_i32 : i32
          %159 = spirv.ShiftLeftLogical %158, %cst24_i32 : i32, i32
          %160 = spirv.ShiftRightArithmetic %159, %cst24_i32 : i32, i32
          %161 = spirv.BitwiseAnd %160, %cst255_i32 : i32
          %162 = spirv.BitwiseAnd %161, %cst255_i32 : i32
          %163 = spirv.ShiftRightLogical %103, %cst2_i32 : i32, i32
          %164 = spirv.BitwiseAnd %163, %cst1_i32 : i32
          %165 = spirv.ShiftLeftLogical %164, %cst8_i32 : i32, i32
          %166 = spirv.BitwiseOr %162, %165 : i32
          %167 = spirv.IAdd %166, %cst1024_i32 : i32
          %168 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %167] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %169 = spirv.Load "StorageBuffer" %168 : i32
          %170 = spirv.IAdd %114, %cst3_i32 : i32
          %171 = spirv.SDiv %170, %cst4_i32 : i32
          %172 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %171] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %173 = spirv.Load "StorageBuffer" %172 : i32
          %174 = spirv.UMod %170, %cst4_i32 : i32
          %175 = spirv.IMul %174, %cst8_i32 : i32
          %176 = spirv.ShiftRightArithmetic %173, %175 : i32, i32
          %177 = spirv.BitwiseAnd %176, %cst255_i32 : i32
          %178 = spirv.ShiftLeftLogical %177, %cst24_i32 : i32, i32
          %179 = spirv.ShiftRightArithmetic %178, %cst24_i32 : i32, i32
          %180 = spirv.BitwiseAnd %179, %cst255_i32 : i32
          %181 = spirv.BitwiseAnd %180, %cst255_i32 : i32
          %182 = spirv.ShiftRightLogical %103, %cst3_i32 : i32, i32
          %183 = spirv.BitwiseAnd %182, %cst1_i32 : i32
          %184 = spirv.ShiftLeftLogical %183, %cst8_i32 : i32, i32
          %185 = spirv.BitwiseOr %181, %184 : i32
          %186 = spirv.IAdd %185, %cst1024_i32 : i32
          %187 = spirv.AccessChain %__resource_var_0_3__addr[%cst0_i32, %186] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          %188 = spirv.Load "StorageBuffer" %187 : i32
          %189 = spirv.IAdd %60, %cst4_i32 : i32
          %190 = spirv.IAdd %60, %cst8_i32 : i32
          %191 = spirv.IAdd %60, %cst12_i32 : i32
          %192 = spirv.IAdd %60, %cst16_i32 : i32
          %193 = spirv.IAdd %60, %cst20_i32 : i32
          %194 = spirv.IAdd %60, %cst24_i32 : i32
          %195 = spirv.IAdd %60, %cst28_i32 : i32
          %196 = spirv.Variable : !spirv.ptr<f32, Function>
          spirv.mlir.loop {
            spirv.Branch ^bb1(%cst0_i32, %cst_f32_1 : i32, f32)
          ^bb1(%204: i32, %205: f32):  // 2 preds: ^bb0, ^bb2
            %206 = spirv.SLessThan %204, %cst4_i32 : i32
            spirv.BranchConditional %206, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %207 = spirv.IMul %204, %cst8_i32 : i32
            %208 = spirv.ShiftRightLogical %131, %207 : i32, i32
            %209 = spirv.IAdd %60, %204 : i32
            %210 = spirv.IMul %42, %11 : i32
            %211 = spirv.IAdd %210, %209 : i32
            %212 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %211] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %213 = spirv.Load "StorageBuffer" %212 : f32
            %214 = spirv.BitwiseAnd %208, %cst15_i32 : i32
            %215 = spirv.ConvertUToF %214 : i32 to f32
            %216 = spirv.FMul %213, %215 : f32
            %217 = spirv.IAdd %189, %204 : i32
            %218 = spirv.IAdd %210, %217 : i32
            %219 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %218] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %220 = spirv.Load "StorageBuffer" %219 : f32
            %221 = spirv.ShiftRightLogical %208, %cst4_i32 : i32, i32
            %222 = spirv.BitwiseAnd %221, %cst15_i32 : i32
            %223 = spirv.ConvertUToF %222 : i32 to f32
            %224 = spirv.FMul %220, %223 : f32
            %225 = spirv.ShiftRightLogical %150, %207 : i32, i32
            %226 = spirv.IAdd %190, %204 : i32
            %227 = spirv.IAdd %210, %226 : i32
            %228 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %227] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %229 = spirv.Load "StorageBuffer" %228 : f32
            %230 = spirv.BitwiseAnd %225, %cst15_i32 : i32
            %231 = spirv.ConvertUToF %230 : i32 to f32
            %232 = spirv.FMul %229, %231 : f32
            %233 = spirv.IAdd %191, %204 : i32
            %234 = spirv.IAdd %210, %233 : i32
            %235 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %234] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %236 = spirv.Load "StorageBuffer" %235 : f32
            %237 = spirv.ShiftRightLogical %225, %cst4_i32 : i32, i32
            %238 = spirv.BitwiseAnd %237, %cst15_i32 : i32
            %239 = spirv.ConvertUToF %238 : i32 to f32
            %240 = spirv.FMul %236, %239 : f32
            %241 = spirv.ShiftRightLogical %169, %207 : i32, i32
            %242 = spirv.IAdd %192, %204 : i32
            %243 = spirv.IAdd %210, %242 : i32
            %244 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %243] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %245 = spirv.Load "StorageBuffer" %244 : f32
            %246 = spirv.BitwiseAnd %241, %cst15_i32 : i32
            %247 = spirv.ConvertUToF %246 : i32 to f32
            %248 = spirv.FMul %245, %247 : f32
            %249 = spirv.IAdd %193, %204 : i32
            %250 = spirv.IAdd %210, %249 : i32
            %251 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %250] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %252 = spirv.Load "StorageBuffer" %251 : f32
            %253 = spirv.ShiftRightLogical %241, %cst4_i32 : i32, i32
            %254 = spirv.BitwiseAnd %253, %cst15_i32 : i32
            %255 = spirv.ConvertUToF %254 : i32 to f32
            %256 = spirv.FMul %252, %255 : f32
            %257 = spirv.ShiftRightLogical %188, %207 : i32, i32
            %258 = spirv.IAdd %194, %204 : i32
            %259 = spirv.IAdd %210, %258 : i32
            %260 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %259] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %261 = spirv.Load "StorageBuffer" %260 : f32
            %262 = spirv.BitwiseAnd %257, %cst15_i32 : i32
            %263 = spirv.ConvertUToF %262 : i32 to f32
            %264 = spirv.FMul %261, %263 : f32
            %265 = spirv.IAdd %195, %204 : i32
            %266 = spirv.IAdd %210, %265 : i32
            %267 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %266] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %268 = spirv.Load "StorageBuffer" %267 : f32
            %269 = spirv.ShiftRightLogical %257, %cst4_i32 : i32, i32
            %270 = spirv.BitwiseAnd %269, %cst15_i32 : i32
            %271 = spirv.ConvertUToF %270 : i32 to f32
            %272 = spirv.FMul %268, %271 : f32
            %273 = spirv.FAdd %216, %224 : f32
            %274 = spirv.FAdd %273, %232 : f32
            %275 = spirv.FAdd %274, %240 : f32
            %276 = spirv.FAdd %275, %248 : f32
            %277 = spirv.FAdd %276, %256 : f32
            %278 = spirv.FAdd %277, %264 : f32
            %279 = spirv.FAdd %278, %272 : f32
            %280 = spirv.FAdd %205, %279 : f32
            spirv.Store "Function" %196, %280 : f32
            %281 = spirv.IAdd %204, %cst1_i32 : i32
            spirv.Branch ^bb1(%281, %280 : i32, f32)
          ^bb3:  // pred: ^bb1
            spirv.mlir.merge
          }
          %197 = spirv.Load "Function" %196 : f32
          %198 = spirv.FMul %62, %108 : f32
          %199 = spirv.FAdd %197, %198 : f32
          %200 = spirv.FMul %90, %199 : f32
          %201 = spirv.FMul %200, %111 : f32
          %202 = spirv.FAdd %54, %201 : f32
          spirv.Store "Function" %50, %202 : f32
          %203 = spirv.IAdd %53, %cst32_i32 : i32
          spirv.Branch ^bb1(%203, %202 : i32, f32)
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
  spirv.EntryPoint "GLCompute" @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32, @__builtin__LocalInvocationId__, @__builtin__WorkgroupId__, @__builtin__NumWorkgroups__
  spirv.ExecutionMode @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32 "LocalSize", 64, 1, 1
  spirv.ExecutionMode @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32 "SignedZeroInfNanPreserve", 32
}