module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>]> : !hal.device
  util.global private @__constant_tensor_1024xi32 : !stream.resource<constant>
  util.initializer {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = stream.timepoint.immediate => !stream.timepoint
    %buffer_cst = util.buffer.constant {alignment = 64 : index} : !util.buffer = #util.composite<7168xi8, [
    dense<"0x00000000010101000001010102010101010101021100010011000001110200011200010111010101120201011001020111010202210101000011010002110100011000010112000101110101011102011011000012110000111101001111000112120001101001011110010110110101111101011211010111120101101002011210020111110201101202011110010211110102101201022011010021120100211000012111010120120101221102012111000221120102012101000221010101210102112100001122010010210200112001011121010110210102212101002121010201011200000110010000110101011101010212010100110202011102100110001000110012001100110111001002110011011001110011011001110111011101120111011102110110001201110112011101110220001001220110012102100121011101220112012100110220011102220111020110100000111100011111000112110001101200011110010211100100121001011011010011110101111101021111010112110101111201001212010112100200111102011111020112110211111000121210001110110010111100111111001211110011121100121012001111120010121200121212001110100110111001111110011211100111101101121011011011110111111101121111011112110112121101111012011011120111111201121112011112120112101002101110021111100211101102101111021111110212111102111211021212110210101202121012021111120221111100201010012111100120121001211011012111110122111101211211012110120120111201211112012112120122121201221110022111110220111202012010010021100100201101012111010222110100211201012110021020110012201100112111001120120011211001102210011120110110211101112111011221110111221101112112011220110211211102102211021221120211221202222111002021120022211001212210012020110121211101202211012221120121211002010021010102210110012100120020011101210110022101120221011001220110002102120121021101220221012100210120010211210001112001021021010111210100122101021122010110210211122000111121001010200111112001111021011011210111112101111221011210220111112201111220021011210211112102111222022111200020102100221021002112210021112200211121012011200201212100012122011021210011222100122120011121210111222201102021021221210221202101222121012121210201010010010001100200011101010111000201110001021102010211010201121001001011010110120101101002011011000210110202101101001111000111100101111101011112010111110201111202011111010211100100121201001210000112110101121200021221000110220101102100001121010111210202112102001201100010011101100112011001110011001200110212001101100111001101110111011102110111011002110111021102120211011000120010011202100112011101121111001011100110101101101111011012110110111201101010021011110210121202101110001110110011111100111211001111120011101001111110011110110111111101111211011110120111111201111110021110110211111102111211021111120211121000121111001211100112101101121111011212110112111201121111021210120212121202122111001021110110221201102111001120120011201001112110011122100111211101112211011121120111211102112011011221110112211002120120011001210111002201110122021101210012112000101221001011210110102201101221021010200011112100111120011112200111102101111121011112210111112201111020021111210211122102111122001211210112122101121021021211220212202100112221001121210111222201112021021121220211010010100201101001021010020011100101111002021110000112100000101101011011010011110001111101011111020111110102111101011211020110120102101201011112010012121101101011021010110011101001111011011110120111101102111011011210110010111001101111011011120110111102101110001111110011111200111110011111110111111201111110021111110211111202111111001211100112111101121112011211110212111200101211011012110011121001111211011112120111121102111211011212120212122101111020011210210110112100111120011111210111112201111121021111220012112101121120001112210111122102111222021112200112120111101001101110001111100111111002111110001211100112111001111210011010110011101101111011021110110112101102121011001011110110111100111111011111110211111100121111011211110212111101101211021012110011121101111211021112110112121100101012011011120011111201111112021111120112111200111212011112120212121211101010121010101011101011111010121110101112101010101110111011101210111010111110111111101211111011121110121211101110121010111210111112101211121011121210101010111110101112101011101110111111101112111011101210111112101110101111111011111210111110111111111111111211111110121111111211111212111110101211111012111011121111111211121112111012121111121211121212111110101210111012111110121112101212121012101011121110111210111112111111121211111210121112111211121110121210111212111112121211121211121212211010102011101022111010211210102110111020111110211111102012111021121110211012102111121022111210211212102110101121111011221110112212101120101111211011112210111120111111211111112211111120121111211211112212111121101211201112112111121121121211221010122111101220121012211210122110111221111112211112122211121221121212002110100221101001221010002011100121111002221210012110110120111100211111012111110221111101221111012112110220101201221012012111120022111201201212112110101120111010211110112111101221111011221110112112101120101110211011112110111221101111221011102011111120111112201111102111111121111112211111102211111122111112221111112012111021121111211211122112111122121111211012112011121021111211211112122111121122111211211212202110102121111022221110212212102121101121201111212111112221111121221111222012112121121121201012222011122121111220221112222211122120121201012010000121100101211102012111000221110001221101002012110020101001201011012110110022101102221011012011110021111101211112012111110221111101221112012211100120121200211211012112110222122201211021002011210121112002211121012012210021122101221201102110011121100112221000102011021020110111201100122011011021110011211101112111011221110212211100102211011122110111201200112112011121120211211200122112011022120112221211112010121220101110211011112110121121101112211010112011111120111211201111122011101021111110211110112111111121111211211111122111111022111011221111112211121122111112221112112012111021121111211212112112111221121212211212102212111122121012221221112110201020112111201121102111201121112111211122112111201221112212211120102211211122112111201220112112221121122112211202212110022021110121211102212212112020101121211011202210102122101221221011212011102220111120211110212111112121111221211111222111112122111220201210212012122220121121211210212212112222122120211020222110212020112122201121212111222221112021221121222211222122120101002102010121010102211001012010000121120001211101012111020121110102222101012020010121210101220011012001100121011101210011012201120122121001201111012011100021111100211012002111100121101101211111012112110121111201211212012111110221121102211012022110100122111101222112012021110220211101212110022122120221211100222212012201210021012102211021012012210021112200211121012111220122112102222120012122210121010011200102112001011121020212210100112201011222110010201201102011011120100112201201122010001021110110211100112110011121110111211201112111021121120012211101122111011122100211222100112021021120200110212201102121021021210111212201122121021221220111220011102002111020011111200211122001101121001111210111112102111121001211210112112100101221011112210011102202111022021011220111112210101020111110201012102012121020111011201011112011111120121111201112112011101220111112201212122011101021101110211111102112111021111210211010112111101121121011211011112111111121121111211012112111121121121211211110122110111221111112211211122111121221111110221110112212101122101111221111112212111122111211221212112210101222111112222010112021111120211110212110112122101121211111212211112121121121211112212012122120111122211111222011122222111222012211200121122001201021002011210121112102211121012011220122112201211222122011201121112010211220122010211121102112221021112011211021112111211121122111211122112112201221112112211222122110211022112111222121112020211021212210212020112121211121202211212120122121211122010121200101202101012122110220211101212112022121100122211000212212012122210122210112212001102021001121210111212101102221021122210012222110102120121021201111212010122120121020211111202111102121101121211111212112112121111221211111222112122221111120221111212210122122211120202211212021112220201120212211202121112121221022212111212221122122211022220122212101212122112120201122212011202021112121211222212111202221122122211122222110202122122121222021212122212121"> : tensor<1024xi32>,
    dense<"0x0000000001010100010101021102000111010101210101000110000101110201111101001111000111100101101101011211010111120101121002011111020111110102211101011122010011200101212101002121010200001101010111010100110210021100110110011100110110011101120111011102110111011201210011022001110200111100011211000110120002111001011111010012120101121002001111020112110211111000121210001110110011111100121111001111120011101001111011011210110110111101111111011112110112121101111012011211120111121201101110021110110211111102121111021212110211111202211111002010100121111001201210012111110122111101211012012011120121121201221110020120100100201101012111011121110011211001102210011021110111211101122111011122110111211201112111021001210011012101120121022101210002112100011120010012210102112201011021021111200111102101111121011112210111112201101121022112210021112101102121001221210221202101212121021101011012010110110100111100011110010111110101111202011111010211120100121101011221010111210202110111011001120110011100110012001100110111011101110211011102120211001001120210011211110010111001101011011012110110111102101110001110110011121100111112001110100111111101111211011110120111111002111011021111120211111100121110011210110112111201121111021221110011201001112210011121110111221101112112011121110211201101120121011111210110102000111121001112200111102101111121011111220111122102111121011212210112202102110100101001011110010110110100111100011111010111110201111101021111010112111101101011021010110011101001111011011110120111101101121011001011100110111201101110001111110011111200111111011111100211111102111112021111110012111001121112011211110212111101101211001112100111121101111212011112110211121101121221011011210011112001111121011111220111112101121121021112011110100110111002111110001211100111121000111011011110110212101101101111001111110111111102111111011211110210121101111211011212110111111202111112001112120111121202121212121010101011101010101110121011101111111011121110121211101011121012111210111212101010101111101011101110111111101112111011101210111112101111101111121011111011111111111111121111111112111112121111101012111011121111111211121112111012121112121211111010121111101211121012101011121111111212111112101211121110121210111212111212122110101022111010211210102011111021111110211012102211121021121210221210112010111121101111201111112111111120121111211211112212111121111211221010122111101221121012211011122111111222111212211212120121111001211011002111110221111101221111012112110121111211211010112011101021111012211110112211101121121011201011102110111221101111221011102011111120111112201111102111111121111110221111112211111222111111201211102112111122121111211012112011121021111212211112112211121121121221211110212110112120111122211111212211112121121121211112010121110001221111012110110120111100211111022111120122111001201211012112210121110110211001112110001020110210201101112111021221110111221101112012001221121111201011102110111121101211211011122110101120111211201111122011101021111011211111112111121121111110221110112211111222111211201211102112111121121112211212122112121022121111221221112011211021112111211122112111201221112111221120112112012121111121211012212210112120111120211110212111122121111122211111212211112121122121211122222111202122110101002101010221120001211102012121010120210101220011012001100121011201221111012011100021101101211111012112120121121102211111012221120120012102211122002121200121222101210102112001001122110010201101102110011121110111211201112111011221110111222100112021021021021110200111112001111121021111210112112100101221001110220211102211111020101111201111112012111120111211201110122010111021111110211211102111121021101011211110112111111121101211211112112111101221101112211111122112111221111212211210112210111122111111221212112211111222211110212210112121111121221111212112112120111222012010210121112101201122112111201121102110211121112111211221112111221121112112212121112020211021212012212121112201012120010121221101212112022121210122210011212102112221101021201012212011102121111121211211212111122121111122211111202222112120211122202011202121112122211221220122212111202021112121211221222112212122"> : tensor<512xi32>,
    dense<"0x00000000010101000101010221010100011102011111000110110101121101011112010111220100112001012121010210021100110110011001110112011101110112012100110201111101011210021111100011101100111111001110110112101101101111011111110111121101121211011110120112111201101110021111110220101001211111012011120121121201221110020120100110221001112111011121120111012101021121000012210101102102111120011111220121122100102121001221210212010110110001111101011112020111001200110111011102110111111100101110011010110110111102101010011111110111121101111012011111100211111202111111001211100112101101121112011211110212211100112210011121110111102000111121011112210211122101122021021100011111010111110201111111021010110111101101121010011011120110111100111112001111110111111102111110011211110110121101111211011212200111112101111122011111210211120111101002111110001211100111121001111011021210110110111100111111011211110210121101111211011212110111111202111112001112121010111012101110111111101112111012121110101112101110101110111011111110111211101111121011111011111011111111111111121111111112111110101211101112111111121112111211101212111110101210101112111111121211111210121112111012121112121221101010221110102011111021111110211012102211121021101111211111112012111122121111211112112112101221111112221112120121111000211111022111110121111211211010122111101121121012211011112011111220111110211111112111111122111111211012102111121122111211211212212111102121101122211111212211111101201111002111110221111201221101102110001020110210201101112111111120101111211011122011101121111111211112112111111022111112221111112112121221121111221221112011211021112111211120112112112120111120211111222111112122112121211101010021010102212101012200110120111000211111012112120121211201200121022111220021110010201001112111011121120111212102102101111121111111201110122011111021121110211010112111101121111211211111122111121221121011221011112211111122121211222111112122111121201112220120102111211120102111211221112120211021212012212121112201012120010121222101222100112121021122211012212011112121111120222211212001222121"> : tensor<256xi32>,
]>
    %c7168 = arith.constant 7168 : index
    %did_map, %result = stream.resource.try_map on(#hal.device.affinity<@__device_0>) %buffer_cst[%c0] : !util.buffer -> i1, !stream.resource<constant>{%c7168}
    cf.cond_br %did_map, ^bb2(%0, %result : !stream.timepoint, !stream.resource<constant>), ^bb1
  ^bb1:  // pred: ^bb0
    %1 = stream.resource.alloc uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<constant>{%c7168}
    %file = stream.file.constant on(#hal.device.affinity<@__device_0>) %buffer_cst[%c0 for %c7168] : !util.buffer{%c7168} -> !stream.file
    %2 = stream.file.read on(#hal.device.affinity<@__device_0>) await(%0) => %file[%c0_i64], %1[%c0], %c7168 : !stream.file -> !stream.resource<constant>{%c7168} => !stream.timepoint
    cf.br ^bb2(%2, %1 : !stream.timepoint, !stream.resource<constant>)
  ^bb2(%3: !stream.timepoint, %4: !stream.resource<constant>):  // 2 preds: ^bb0, ^bb1
    %5 = stream.timepoint.await sync %3 => %4 : !stream.resource<constant>{%c7168}
    util.global.store %5, @__constant_tensor_1024xi32 : !stream.resource<constant>
    util.return
  }
  hal.executable private @iq1_xs_dispatch_0 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>) {
      hal.executable.export public @iq1_xs_dispatch_0_elementwise_broadcast_D_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @iq1_xs_dispatch_0_elementwise_broadcast_D_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = [64, 1, 1]>} {
          %c32_i64 = arith.constant 32 : i64
          %c28 = arith.constant 28 : index
          %c24 = arith.constant 24 : index
          %c20 = arith.constant 20 : index
          %c16 = arith.constant 16 : index
          %c12 = arith.constant 12 : index
          %c15_i32 = arith.constant 15 : i32
          %c6_i32 = arith.constant 6 : i32
          %c3 = arith.constant 3 : index
          %c4_i32 = arith.constant 4 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c3_i32 = arith.constant 3 : i32
          %c7_i32 = arith.constant 7 : i32
          %c4 = arith.constant 4 : index
          %c2 = arith.constant 2 : index
          %c42 = arith.constant 42 : index
          %c34 = arith.constant 34 : index
          %c8_i32 = arith.constant 8 : i32
          %c255_i32 = arith.constant 255 : i32
          %c1 = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c46 = arith.constant 46 : index
          %c256 = arith.constant 256 : index
          %cst = arith.constant -8.750000e-01 : f32
          %cst_0 = arith.constant -1.125000e+00 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
          %2 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
          %3 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
          %4 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
          %5 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
          %6 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
          %7 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
          %8 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(8) : i32
          %9 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(9) : i32
          %10 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(10) : i32
          %11 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(11) : i32
          %12 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(12) : i32
          %13 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(13) : i32
          %14 = arith.extui %1 : i32 to i64
          %15 = arith.shli %14, %c32_i64 : i64
          %16 = arith.extui %0 : i32 to i64
          %17 = arith.ori %16, %15 : i64
          %18 = arith.index_castui %17 : i64 to index
          %19 = arith.extui %3 : i32 to i64
          %20 = arith.shli %19, %c32_i64 : i64
          %21 = arith.extui %2 : i32 to i64
          %22 = arith.ori %21, %20 : i64
          %23 = arith.index_castui %22 : i64 to index
          %24 = arith.extui %5 : i32 to i64
          %25 = arith.shli %24, %c32_i64 : i64
          %26 = arith.extui %4 : i32 to i64
          %27 = arith.ori %26, %25 : i64
          %28 = arith.index_castui %27 : i64 to index
          %29 = arith.extui %7 : i32 to i64
          %30 = arith.shli %29, %c32_i64 : i64
          %31 = arith.extui %6 : i32 to i64
          %32 = arith.ori %31, %30 : i64
          %33 = arith.index_castui %32 : i64 to index
          %34 = arith.extui %9 : i32 to i64
          %35 = arith.shli %34, %c32_i64 : i64
          %36 = arith.extui %8 : i32 to i64
          %37 = arith.ori %36, %35 : i64
          %38 = arith.index_castui %37 : i64 to index
          %39 = arith.extui %11 : i32 to i64
          %40 = arith.shli %39, %c32_i64 : i64
          %41 = arith.extui %10 : i32 to i64
          %42 = arith.ori %41, %40 : i64
          %43 = arith.index_castui %42 : i64 to index
          %44 = arith.extui %13 : i32 to i64
          %45 = arith.shli %44, %c32_i64 : i64
          %46 = arith.extui %12 : i32 to i64
          %47 = arith.ori %46, %45 : i64
          %48 = arith.index_castui %47 : i64 to index
          %49:6 = util.assume.int 
              %18<udiv = 32>, 
              %28<umin = 0, umax = 281474976710655>, 
              %33<umin = 0, umax = 9007199254740991>, 
              %38<umin = 0, umax = 9007199254740991>, 
              %43<umin = 0, umax = 9007199254740991>, 
              %48<umin = 0, umax = 9007199254740991>
            : index, index, index, index, index, index
          %50 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xi32>>
          %51 = iree_tensor_ext.dispatch.workload.ordinal %49#0, 0 : index
          %52 = iree_tensor_ext.dispatch.workload.ordinal %49#2, 3 : index
          %53 = iree_tensor_ext.dispatch.workload.ordinal %49#3, 4 : index
          %54 = iree_tensor_ext.dispatch.workload.ordinal %49#4, 5 : index
          %55 = iree_tensor_ext.dispatch.workload.ordinal %49#5, 6 : index
          %56 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>{%52}
          %57 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%53, %54}
          %58 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%55}
          %59 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%51}
          %60 = iree_tensor_ext.dispatch.workload.ordinal %23, 1 : index
          %61 = iree_tensor_ext.dispatch.workload.ordinal %49#1, 2 : index
          %62 = iree_tensor_ext.dispatch.tensor.load %56, offsets = [0], sizes = [%52], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>{%52} -> tensor<?xi32>
          %63 = iree_tensor_ext.dispatch.tensor.load %57, offsets = [0, 0], sizes = [%53, %54], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%53, %54} -> tensor<?x?xf32>
          %64 = iree_tensor_ext.dispatch.tensor.load %58, offsets = [0], sizes = [%55], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%55} -> tensor<?xi8>
          %65 = iree_tensor_ext.dispatch.tensor.load %50, offsets = [0], sizes = [1024], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xi32>> -> tensor<1024xi32>
          %66 = tensor.empty(%51) : tensor<?xf32>
          %67 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%66 : tensor<?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64], [1]]>} {
          ^bb0(%out: f32):
            %68 = linalg.index 0 : index
            %69 = arith.remsi %68, %c32 : index
            %70 = arith.divsi %68, %c32 : index
            %71 = arith.remsi %70, %60 : index
            %72 = arith.divsi %70, %60 : index
            %extracted = tensor.extract %62[%72] : tensor<?xi32>
            %73 = arith.index_cast %extracted : i32 to index
            %74 = arith.muli %73, %60 : index
            %75 = arith.addi %74, %71 : index
            %76 = arith.divui %54, %c256 : index
            %77 = arith.muli %75, %76 : index
            %78 = arith.muli %77, %c46 : index
            %79 = scf.for %arg0 = %69 to %61 step %c32 iter_args(%arg1 = %cst_1) -> (f32) {
              %80 = arith.divui %arg0, %c8 : index
              %81 = arith.remui %arg0, %c8 : index
              %82 = arith.muli %80, %c46 : index
              %83 = arith.addi %78, %82 : index
              %84 = arith.muli %arg0, %c32 : index
              %85 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst_1) -> (f32) {
                %156 = arith.addi %84, %arg2 : index
                %extracted_14 = tensor.extract %63[%72, %156] : tensor<?x?xf32>
                %157 = arith.addf %arg3, %extracted_14 : f32
                scf.yield %157 : f32
              }
              %extracted_2 = tensor.extract %64[%83] : tensor<?xi8>
              %86 = arith.addi %83, %c1 : index
              %extracted_3 = tensor.extract %64[%86] : tensor<?xi8>
              %87 = arith.extui %extracted_2 : i8 to i32
              %88 = arith.extui %extracted_3 : i8 to i32
              %89 = arith.andi %87, %c255_i32 : i32
              %90 = arith.andi %88, %c255_i32 : i32
              %91 = arith.shli %90, %c8_i32 : i32
              %92 = arith.ori %89, %91 : i32
              %93 = arith.trunci %92 : i32 to i16
              %94 = arith.bitcast %93 : i16 to f16
              %95 = arith.extf %94 : f16 to f32
              %96 = arith.addi %83, %c34 : index
              %97 = arith.addi %96, %81 : index
              %extracted_4 = tensor.extract %64[%97] : tensor<?xi8>
              %98 = arith.extui %extracted_4 : i8 to i32
              %99 = arith.andi %98, %c255_i32 : i32
              %100 = arith.addi %83, %c42 : index
              %101 = arith.divui %81, %c2 : index
              %102 = arith.addi %100, %101 : index
              %extracted_5 = tensor.extract %64[%102] : tensor<?xi8>
              %103 = arith.extui %extracted_5 : i8 to i32
              %104 = arith.andi %103, %c255_i32 : i32
              %105 = arith.remui %81, %c2 : index
              %106 = arith.muli %105, %c4 : index
              %107 = arith.index_castui %106 : index to i32
              %108 = arith.shrui %104, %107 : i32
              %109 = arith.andi %108, %c7_i32 : i32
              %110 = arith.shrui %108, %c3_i32 : i32
              %111 = arith.andi %110, %c1_i32 : i32
              %112 = arith.cmpi ne, %111, %c0_i32 : i32
              %113 = arith.select %112, %cst_0, %cst : f32
              %114 = arith.muli %109, %c2_i32 : i32
              %115 = arith.addi %114, %c1_i32 : i32
              %116 = arith.uitofp %115 : i32 to f32
              %117 = arith.muli %81, %c4 : index
              %118 = arith.addi %117, %c2 : index
              %119 = arith.addi %83, %118 : index
              %extracted_6 = tensor.extract %64[%119] : tensor<?xi8>
              %120 = arith.extui %extracted_6 : i8 to i32
              %121 = arith.andi %120, %c255_i32 : i32
              %122 = arith.andi %98, %c3_i32 : i32
              %123 = arith.shli %122, %c8_i32 : i32
              %124 = arith.ori %121, %123 : i32
              %125 = arith.index_castui %124 : i32 to index
              %extracted_7 = tensor.extract %65[%125] : tensor<1024xi32>
              %126 = arith.addi %119, %c1 : index
              %extracted_8 = tensor.extract %64[%126] : tensor<?xi8>
              %127 = arith.extui %extracted_8 : i8 to i32
              %128 = arith.andi %127, %c255_i32 : i32
              %129 = arith.shrui %99, %c2_i32 : i32
              %130 = arith.andi %129, %c3_i32 : i32
              %131 = arith.shli %130, %c8_i32 : i32
              %132 = arith.ori %128, %131 : i32
              %133 = arith.index_castui %132 : i32 to index
              %extracted_9 = tensor.extract %65[%133] : tensor<1024xi32>
              %134 = arith.addi %119, %c2 : index
              %extracted_10 = tensor.extract %64[%134] : tensor<?xi8>
              %135 = arith.extui %extracted_10 : i8 to i32
              %136 = arith.andi %135, %c255_i32 : i32
              %137 = arith.shrui %99, %c4_i32 : i32
              %138 = arith.andi %137, %c3_i32 : i32
              %139 = arith.shli %138, %c8_i32 : i32
              %140 = arith.ori %136, %139 : i32
              %141 = arith.index_castui %140 : i32 to index
              %extracted_11 = tensor.extract %65[%141] : tensor<1024xi32>
              %142 = arith.addi %119, %c3 : index
              %extracted_12 = tensor.extract %64[%142] : tensor<?xi8>
              %143 = arith.extui %extracted_12 : i8 to i32
              %144 = arith.andi %143, %c255_i32 : i32
              %145 = arith.shrui %99, %c6_i32 : i32
              %146 = arith.andi %145, %c3_i32 : i32
              %147 = arith.shli %146, %c8_i32 : i32
              %148 = arith.ori %144, %147 : i32
              %149 = arith.index_castui %148 : i32 to index
              %extracted_13 = tensor.extract %65[%149] : tensor<1024xi32>
              %150 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %cst_1) -> (f32) {
                %156 = arith.index_castui %arg2 : index to i32
                %157 = arith.muli %156, %c8_i32 : i32
                %158 = arith.shrui %extracted_7, %157 : i32
                %159 = arith.addi %84, %arg2 : index
                %extracted_14 = tensor.extract %63[%72, %159] : tensor<?x?xf32>
                %160 = arith.andi %158, %c15_i32 : i32
                %161 = arith.uitofp %160 : i32 to f32
                %162 = arith.mulf %extracted_14, %161 : f32
                %163 = arith.addi %84, %c4 : index
                %164 = arith.addi %163, %arg2 : index
                %extracted_15 = tensor.extract %63[%72, %164] : tensor<?x?xf32>
                %165 = arith.shrui %158, %c4_i32 : i32
                %166 = arith.andi %165, %c15_i32 : i32
                %167 = arith.uitofp %166 : i32 to f32
                %168 = arith.mulf %extracted_15, %167 : f32
                %169 = arith.shrui %extracted_9, %157 : i32
                %170 = arith.addi %84, %c8 : index
                %171 = arith.addi %170, %arg2 : index
                %extracted_16 = tensor.extract %63[%72, %171] : tensor<?x?xf32>
                %172 = arith.andi %169, %c15_i32 : i32
                %173 = arith.uitofp %172 : i32 to f32
                %174 = arith.mulf %extracted_16, %173 : f32
                %175 = arith.addi %84, %c12 : index
                %176 = arith.addi %175, %arg2 : index
                %extracted_17 = tensor.extract %63[%72, %176] : tensor<?x?xf32>
                %177 = arith.shrui %169, %c4_i32 : i32
                %178 = arith.andi %177, %c15_i32 : i32
                %179 = arith.uitofp %178 : i32 to f32
                %180 = arith.mulf %extracted_17, %179 : f32
                %181 = arith.shrui %extracted_11, %157 : i32
                %182 = arith.addi %84, %c16 : index
                %183 = arith.addi %182, %arg2 : index
                %extracted_18 = tensor.extract %63[%72, %183] : tensor<?x?xf32>
                %184 = arith.andi %181, %c15_i32 : i32
                %185 = arith.uitofp %184 : i32 to f32
                %186 = arith.mulf %extracted_18, %185 : f32
                %187 = arith.addi %84, %c20 : index
                %188 = arith.addi %187, %arg2 : index
                %extracted_19 = tensor.extract %63[%72, %188] : tensor<?x?xf32>
                %189 = arith.shrui %181, %c4_i32 : i32
                %190 = arith.andi %189, %c15_i32 : i32
                %191 = arith.uitofp %190 : i32 to f32
                %192 = arith.mulf %extracted_19, %191 : f32
                %193 = arith.shrui %extracted_13, %157 : i32
                %194 = arith.addi %84, %c24 : index
                %195 = arith.addi %194, %arg2 : index
                %extracted_20 = tensor.extract %63[%72, %195] : tensor<?x?xf32>
                %196 = arith.andi %193, %c15_i32 : i32
                %197 = arith.uitofp %196 : i32 to f32
                %198 = arith.mulf %extracted_20, %197 : f32
                %199 = arith.addi %84, %c28 : index
                %200 = arith.addi %199, %arg2 : index
                %extracted_21 = tensor.extract %63[%72, %200] : tensor<?x?xf32>
                %201 = arith.shrui %193, %c4_i32 : i32
                %202 = arith.andi %201, %c15_i32 : i32
                %203 = arith.uitofp %202 : i32 to f32
                %204 = arith.mulf %extracted_21, %203 : f32
                %205 = arith.addf %162, %168 : f32
                %206 = arith.addf %205, %174 : f32
                %207 = arith.addf %206, %180 : f32
                %208 = arith.addf %207, %186 : f32
                %209 = arith.addf %208, %192 : f32
                %210 = arith.addf %209, %198 : f32
                %211 = arith.addf %210, %204 : f32
                %212 = arith.addf %arg3, %211 : f32
                scf.yield %212 : f32
              }
              %151 = arith.mulf %85, %113 : f32
              %152 = arith.addf %150, %151 : f32
              %153 = arith.mulf %95, %152 : f32
              %154 = arith.mulf %153, %116 : f32
              %155 = arith.addf %arg1, %154 : f32
              scf.yield %155 : f32
            }
            linalg.yield %79 : f32
          } -> tensor<?xf32>
          iree_tensor_ext.dispatch.tensor.store %67, %59, offsets = [0], sizes = [%51], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%51}
          return
        }
        iree_codegen.dispatch_config @iq1_xs_dispatch_0_elementwise_broadcast_D_f32 {
        ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
          %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
          iree_codegen.yield %x, %y, %z : index, index, index
        }
      }
    }
  }
  hal.executable private @iq1_xs_dispatch_1 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>) {
      hal.executable.export public @iq1_xs_dispatch_1_reduction_Dx32_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1)
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @iq1_xs_dispatch_1_reduction_Dx32_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = [64, 1, 1]>} {
          %c32_i64 = arith.constant 32 : i64
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
          %2 = arith.extui %1 : i32 to i64
          %3 = arith.shli %2, %c32_i64 : i64
          %4 = arith.extui %0 : i32 to i64
          %5 = arith.ori %4, %3 : i64
          %6 = arith.index_castui %5 : i64 to index
          %7 = iree_tensor_ext.dispatch.workload.ordinal %6, 0 : index
          %8 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%7}
          %9 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%7}
          %10 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%7, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%7} -> tensor<?x32xf32>
          %11 = tensor.empty(%7) : tensor<?xf32>
          %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
          %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<?x32xf32>) outs(%12 : tensor<?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64], [1]]>} {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.addf %in, %out : f32
            linalg.yield %14 : f32
          } -> tensor<?xf32>
          iree_tensor_ext.dispatch.tensor.store %13, %9, offsets = [0], sizes = [%7], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%7}
          return
        }
        iree_codegen.dispatch_config @iq1_xs_dispatch_1_reduction_Dx32_f32 {
        ^bb0(%arg0: !hal.device, %arg1: index):
          %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1)
          iree_codegen.yield %x, %y, %z : index, index, index
        }
      }
    }
  }
  util.func public @iq1_xs(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: i32, %arg4: !hal.fence, %arg5: !hal.fence) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @iq1_xs(%input0: tensor<?x?xf32>, %input1: tensor<?xi8>, %input2: tensor<?xi32>, %input3: i32) -> (%output0: tensor<?x?xf32>)", iree.abi.model = "coarse-fences"}} {
    %c7168 = arith.constant 7168 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32_i64 = arith.constant 32 : i64
    %__constant_tensor_1024xi32 = util.global.load immutable @__constant_tensor_1024xi32 : !stream.resource<constant>
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = arith.muli %0, %c4 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf32>{%0, %1} in !stream.resource<external>{%3}
    %5 = stream.timepoint.import on(#hal.device.affinity<@__device_0>) %arg4 : (!hal.fence) => !stream.timepoint
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %element_type_i8 = hal.element_type<i8> : i32
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%6]) type(%element_type_i8) encoding(%dense_row_major)
    %7 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?xi8>{%6} in !stream.resource<external>{%6}
    %8 = hal.buffer_view.dim<%arg2 : !hal.buffer_view>[0] : index
    %element_type_i32 = hal.element_type<i32> : i32
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input2") shape([%8]) type(%element_type_i32) encoding(%dense_row_major)
    %9 = arith.muli %8, %c4 : index
    %10 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg2 : !hal.buffer_view -> tensor<?xi32>{%8} in !stream.resource<external>{%9}
    %11 = arith.index_cast %arg3 : i32 to index
    %12 = arith.divui %1, %c32 : index
    %13 = arith.muli %0, %11 overflow<nsw> : index
    %14 = arith.muli %13, %c32 overflow<nsw> : index
    %15 = arith.muli %13, %c128 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%5) => !stream.resource<transient>{%15} => !stream.timepoint
    %16 = arith.index_castui %14 : index to i64
    %17 = arith.index_castui %14 : index to i32
    %18 = arith.shrui %16, %c32_i64 : i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.index_castui %11 : index to i64
    %21 = arith.index_castui %11 : index to i32
    %22 = arith.shrui %20, %c32_i64 : i64
    %23 = arith.trunci %22 : i64 to i32
    %24 = arith.index_castui %12 : index to i64
    %25 = arith.index_castui %12 : index to i32
    %26 = arith.shrui %24, %c32_i64 : i64
    %27 = arith.trunci %26 : i64 to i32
    %28 = arith.index_castui %8 : index to i64
    %29 = arith.index_castui %8 : index to i32
    %30 = arith.shrui %28, %c32_i64 : i64
    %31 = arith.trunci %30 : i64 to i32
    %32 = arith.index_castui %0 : index to i64
    %33 = arith.index_castui %0 : index to i32
    %34 = arith.shrui %32, %c32_i64 : i64
    %35 = arith.trunci %34 : i64 to i32
    %36 = arith.index_castui %1 : index to i64
    %37 = arith.index_castui %1 : index to i32
    %38 = arith.shrui %36, %c32_i64 : i64
    %39 = arith.trunci %38 : i64 to i32
    %40 = arith.index_castui %6 : index to i64
    %41 = arith.index_castui %6 : index to i32
    %42 = arith.shrui %40, %c32_i64 : i64
    %43 = arith.trunci %42 : i64 to i32
    %44 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%10 as %arg6: !stream.resource<external>{%9}, %4 as %arg7: !stream.resource<external>{%3}, %7 as %arg8: !stream.resource<external>{%6}, %__constant_tensor_1024xi32 as %arg9: !stream.resource<constant>{%c7168}, %result as %arg10: !stream.resource<transient>{%15}) {
      stream.cmd.dispatch @iq1_xs_dispatch_0::@vulkan_spirv_fb::@iq1_xs_dispatch_0_elementwise_broadcast_D_f32[%14, %11, %12, %8, %0, %1, %6](%17, %19, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39, %41, %43 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        ro %arg6[%c0 for %9] : !stream.resource<external>{%9},
        ro %arg7[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg8[%c0 for %6] : !stream.resource<external>{%6},
        ro %arg9[%c0 for %c7168] : !stream.resource<constant>{%c7168},
        wo %arg10[%c0 for %15] : !stream.resource<transient>{%15}
      }
    } => !stream.timepoint
    %45 = stream.timepoint.await %44 => %result : !stream.resource<transient>{%15}
    %46 = util.optimization_barrier %45 : !stream.resource<transient>
    %47 = arith.muli %13, %c4 : index
    %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%47} => !stream.timepoint
    %48 = arith.index_castui %13 : index to i64
    %49 = arith.index_castui %13 : index to i32
    %50 = arith.shrui %48, %c32_i64 : i64
    %51 = arith.trunci %50 : i64 to i32
    %52 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint_1) => with(%46 as %arg6: !stream.resource<transient>{%15}, %result_0 as %arg7: !stream.resource<external>{%47}) {
      stream.cmd.dispatch @iq1_xs_dispatch_1::@vulkan_spirv_fb::@iq1_xs_dispatch_1_reduction_Dx32_f32[%13](%49, %51 : i32, i32) {
        ro %arg6[%c0 for %15] : !stream.resource<transient>{%15},
        wo %arg7[%c0 for %47] : !stream.resource<external>{%47}
      }
    } => !stream.timepoint
    %53 = arith.muli %2, %11 : index
    stream.timepoint.chain_external on(#hal.device.affinity<@__device_0>) %52 => (%arg5 : !hal.fence)
    %54 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %result_0 : tensor<?x?xf32>{%0, %11} in !stream.resource<external>{%53} -> !hal.buffer_view
    util.return %54 : !hal.buffer_view
  }
  hal.executable private @iq1_xxs_dispatch_0 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>) {
      hal.executable.export public @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = [64, 1, 1]>} {
          %c32_i64 = arith.constant 32 : i64
          %c28 = arith.constant 28 : index
          %c24 = arith.constant 24 : index
          %c20 = arith.constant 20 : index
          %c16 = arith.constant 16 : index
          %c12 = arith.constant 12 : index
          %c15_i32 = arith.constant 15 : i32
          %c3_i32 = arith.constant 3 : i32
          %c3 = arith.constant 3 : index
          %c2 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c7_i32 = arith.constant 7 : i32
          %c4_i32 = arith.constant 4 : i32
          %c34 = arith.constant 34 : index
          %c8_i32 = arith.constant 8 : i32
          %c255_i32 = arith.constant 255 : i32
          %c1 = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c42 = arith.constant 42 : index
          %c256 = arith.constant 256 : index
          %cst = arith.constant -8.750000e-01 : f32
          %cst_0 = arith.constant -1.125000e+00 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %c4096 = arith.constant 4096 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
          %2 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
          %3 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
          %4 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
          %5 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
          %6 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
          %7 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
          %8 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(8) : i32
          %9 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(9) : i32
          %10 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(10) : i32
          %11 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(11) : i32
          %12 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(12) : i32
          %13 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(13) : i32
          %14 = arith.extui %1 : i32 to i64
          %15 = arith.shli %14, %c32_i64 : i64
          %16 = arith.extui %0 : i32 to i64
          %17 = arith.ori %16, %15 : i64
          %18 = arith.index_castui %17 : i64 to index
          %19 = arith.extui %3 : i32 to i64
          %20 = arith.shli %19, %c32_i64 : i64
          %21 = arith.extui %2 : i32 to i64
          %22 = arith.ori %21, %20 : i64
          %23 = arith.index_castui %22 : i64 to index
          %24 = arith.extui %5 : i32 to i64
          %25 = arith.shli %24, %c32_i64 : i64
          %26 = arith.extui %4 : i32 to i64
          %27 = arith.ori %26, %25 : i64
          %28 = arith.index_castui %27 : i64 to index
          %29 = arith.extui %7 : i32 to i64
          %30 = arith.shli %29, %c32_i64 : i64
          %31 = arith.extui %6 : i32 to i64
          %32 = arith.ori %31, %30 : i64
          %33 = arith.index_castui %32 : i64 to index
          %34 = arith.extui %9 : i32 to i64
          %35 = arith.shli %34, %c32_i64 : i64
          %36 = arith.extui %8 : i32 to i64
          %37 = arith.ori %36, %35 : i64
          %38 = arith.index_castui %37 : i64 to index
          %39 = arith.extui %11 : i32 to i64
          %40 = arith.shli %39, %c32_i64 : i64
          %41 = arith.extui %10 : i32 to i64
          %42 = arith.ori %41, %40 : i64
          %43 = arith.index_castui %42 : i64 to index
          %44 = arith.extui %13 : i32 to i64
          %45 = arith.shli %44, %c32_i64 : i64
          %46 = arith.extui %12 : i32 to i64
          %47 = arith.ori %46, %45 : i64
          %48 = arith.index_castui %47 : i64 to index
          %49:6 = util.assume.int 
              %18<udiv = 32>, 
              %28<umin = 0, umax = 281474976710655>, 
              %33<umin = 0, umax = 9007199254740991>, 
              %38<umin = 0, umax = 9007199254740991>, 
              %43<umin = 0, umax = 9007199254740991>, 
              %48<umin = 0, umax = 9007199254740991>
            : index, index, index, index, index, index
          %50 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c4096) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xi32>>
          %51 = iree_tensor_ext.dispatch.workload.ordinal %49#0, 0 : index
          %52 = iree_tensor_ext.dispatch.workload.ordinal %49#2, 3 : index
          %53 = iree_tensor_ext.dispatch.workload.ordinal %49#3, 4 : index
          %54 = iree_tensor_ext.dispatch.workload.ordinal %49#4, 5 : index
          %55 = iree_tensor_ext.dispatch.workload.ordinal %49#5, 6 : index
          %56 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>{%52}
          %57 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%53, %54}
          %58 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%55}
          %59 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%51}
          %60 = iree_tensor_ext.dispatch.workload.ordinal %23, 1 : index
          %61 = iree_tensor_ext.dispatch.workload.ordinal %49#1, 2 : index
          %62 = iree_tensor_ext.dispatch.tensor.load %56, offsets = [0], sizes = [%52], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>{%52} -> tensor<?xi32>
          %63 = iree_tensor_ext.dispatch.tensor.load %57, offsets = [0, 0], sizes = [%53, %54], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%53, %54} -> tensor<?x?xf32>
          %64 = iree_tensor_ext.dispatch.tensor.load %58, offsets = [0], sizes = [%55], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%55} -> tensor<?xi8>
          %65 = iree_tensor_ext.dispatch.tensor.load %50, offsets = [0], sizes = [512], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xi32>> -> tensor<512xi32>
          %66 = tensor.empty(%51) : tensor<?xf32>
          %67 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%66 : tensor<?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64], [1]]>} {
          ^bb0(%out: f32):
            %68 = linalg.index 0 : index
            %69 = arith.remsi %68, %c32 : index
            %70 = arith.divsi %68, %c32 : index
            %71 = arith.remsi %70, %60 : index
            %72 = arith.divsi %70, %60 : index
            %extracted = tensor.extract %62[%72] : tensor<?xi32>
            %73 = arith.index_cast %extracted : i32 to index
            %74 = arith.muli %73, %60 : index
            %75 = arith.addi %74, %71 : index
            %76 = arith.divui %54, %c256 : index
            %77 = arith.muli %75, %76 : index
            %78 = arith.muli %77, %c42 : index
            %79 = scf.for %arg0 = %69 to %61 step %c32 iter_args(%arg1 = %cst_1) -> (f32) {
              %80 = arith.divui %arg0, %c8 : index
              %81 = arith.remui %arg0, %c8 : index
              %82 = arith.muli %80, %c42 : index
              %83 = arith.addi %78, %82 : index
              %84 = arith.muli %arg0, %c32 : index
              %85 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst_1) -> (f32) {
                %147 = arith.addi %84, %arg2 : index
                %extracted_13 = tensor.extract %63[%72, %147] : tensor<?x?xf32>
                %148 = arith.addf %arg3, %extracted_13 : f32
                scf.yield %148 : f32
              }
              %extracted_2 = tensor.extract %64[%83] : tensor<?xi8>
              %86 = arith.addi %83, %c1 : index
              %extracted_3 = tensor.extract %64[%86] : tensor<?xi8>
              %87 = arith.extui %extracted_2 : i8 to i32
              %88 = arith.extui %extracted_3 : i8 to i32
              %89 = arith.andi %87, %c255_i32 : i32
              %90 = arith.andi %88, %c255_i32 : i32
              %91 = arith.shli %90, %c8_i32 : i32
              %92 = arith.ori %89, %91 : i32
              %93 = arith.trunci %92 : i32 to i16
              %94 = arith.bitcast %93 : i16 to f16
              %95 = arith.extf %94 : f16 to f32
              %96 = arith.addi %83, %c34 : index
              %97 = arith.addi %96, %81 : index
              %extracted_4 = tensor.extract %64[%97] : tensor<?xi8>
              %98 = arith.extui %extracted_4 : i8 to i32
              %99 = arith.andi %98, %c255_i32 : i32
              %100 = arith.shrui %99, %c4_i32 : i32
              %101 = arith.andi %100, %c7_i32 : i32
              %102 = arith.shrui %99, %c7_i32 : i32
              %103 = arith.cmpi ne, %102, %c0_i32 : i32
              %104 = arith.select %103, %cst_0, %cst : f32
              %105 = arith.muli %101, %c2_i32 : i32
              %106 = arith.addi %105, %c1_i32 : i32
              %107 = arith.uitofp %106 : i32 to f32
              %108 = arith.muli %81, %c4 : index
              %109 = arith.addi %108, %c2 : index
              %110 = arith.addi %83, %109 : index
              %extracted_5 = tensor.extract %64[%110] : tensor<?xi8>
              %111 = arith.extui %extracted_5 : i8 to i32
              %112 = arith.andi %111, %c255_i32 : i32
              %113 = arith.andi %98, %c1_i32 : i32
              %114 = arith.shli %113, %c8_i32 : i32
              %115 = arith.ori %112, %114 : i32
              %116 = arith.index_castui %115 : i32 to index
              %extracted_6 = tensor.extract %65[%116] : tensor<512xi32>
              %117 = arith.addi %110, %c1 : index
              %extracted_7 = tensor.extract %64[%117] : tensor<?xi8>
              %118 = arith.extui %extracted_7 : i8 to i32
              %119 = arith.andi %118, %c255_i32 : i32
              %120 = arith.shrui %99, %c1_i32 : i32
              %121 = arith.andi %120, %c1_i32 : i32
              %122 = arith.shli %121, %c8_i32 : i32
              %123 = arith.ori %119, %122 : i32
              %124 = arith.index_castui %123 : i32 to index
              %extracted_8 = tensor.extract %65[%124] : tensor<512xi32>
              %125 = arith.addi %110, %c2 : index
              %extracted_9 = tensor.extract %64[%125] : tensor<?xi8>
              %126 = arith.extui %extracted_9 : i8 to i32
              %127 = arith.andi %126, %c255_i32 : i32
              %128 = arith.shrui %99, %c2_i32 : i32
              %129 = arith.andi %128, %c1_i32 : i32
              %130 = arith.shli %129, %c8_i32 : i32
              %131 = arith.ori %127, %130 : i32
              %132 = arith.index_castui %131 : i32 to index
              %extracted_10 = tensor.extract %65[%132] : tensor<512xi32>
              %133 = arith.addi %110, %c3 : index
              %extracted_11 = tensor.extract %64[%133] : tensor<?xi8>
              %134 = arith.extui %extracted_11 : i8 to i32
              %135 = arith.andi %134, %c255_i32 : i32
              %136 = arith.shrui %99, %c3_i32 : i32
              %137 = arith.andi %136, %c1_i32 : i32
              %138 = arith.shli %137, %c8_i32 : i32
              %139 = arith.ori %135, %138 : i32
              %140 = arith.index_castui %139 : i32 to index
              %extracted_12 = tensor.extract %65[%140] : tensor<512xi32>
              %141 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %cst_1) -> (f32) {
                %147 = arith.index_castui %arg2 : index to i32
                %148 = arith.muli %147, %c8_i32 : i32
                %149 = arith.shrui %extracted_6, %148 : i32
                %150 = arith.addi %84, %arg2 : index
                %extracted_13 = tensor.extract %63[%72, %150] : tensor<?x?xf32>
                %151 = arith.andi %149, %c15_i32 : i32
                %152 = arith.uitofp %151 : i32 to f32
                %153 = arith.mulf %extracted_13, %152 : f32
                %154 = arith.addi %84, %c4 : index
                %155 = arith.addi %154, %arg2 : index
                %extracted_14 = tensor.extract %63[%72, %155] : tensor<?x?xf32>
                %156 = arith.shrui %149, %c4_i32 : i32
                %157 = arith.andi %156, %c15_i32 : i32
                %158 = arith.uitofp %157 : i32 to f32
                %159 = arith.mulf %extracted_14, %158 : f32
                %160 = arith.shrui %extracted_8, %148 : i32
                %161 = arith.addi %84, %c8 : index
                %162 = arith.addi %161, %arg2 : index
                %extracted_15 = tensor.extract %63[%72, %162] : tensor<?x?xf32>
                %163 = arith.andi %160, %c15_i32 : i32
                %164 = arith.uitofp %163 : i32 to f32
                %165 = arith.mulf %extracted_15, %164 : f32
                %166 = arith.addi %84, %c12 : index
                %167 = arith.addi %166, %arg2 : index
                %extracted_16 = tensor.extract %63[%72, %167] : tensor<?x?xf32>
                %168 = arith.shrui %160, %c4_i32 : i32
                %169 = arith.andi %168, %c15_i32 : i32
                %170 = arith.uitofp %169 : i32 to f32
                %171 = arith.mulf %extracted_16, %170 : f32
                %172 = arith.shrui %extracted_10, %148 : i32
                %173 = arith.addi %84, %c16 : index
                %174 = arith.addi %173, %arg2 : index
                %extracted_17 = tensor.extract %63[%72, %174] : tensor<?x?xf32>
                %175 = arith.andi %172, %c15_i32 : i32
                %176 = arith.uitofp %175 : i32 to f32
                %177 = arith.mulf %extracted_17, %176 : f32
                %178 = arith.addi %84, %c20 : index
                %179 = arith.addi %178, %arg2 : index
                %extracted_18 = tensor.extract %63[%72, %179] : tensor<?x?xf32>
                %180 = arith.shrui %172, %c4_i32 : i32
                %181 = arith.andi %180, %c15_i32 : i32
                %182 = arith.uitofp %181 : i32 to f32
                %183 = arith.mulf %extracted_18, %182 : f32
                %184 = arith.shrui %extracted_12, %148 : i32
                %185 = arith.addi %84, %c24 : index
                %186 = arith.addi %185, %arg2 : index
                %extracted_19 = tensor.extract %63[%72, %186] : tensor<?x?xf32>
                %187 = arith.andi %184, %c15_i32 : i32
                %188 = arith.uitofp %187 : i32 to f32
                %189 = arith.mulf %extracted_19, %188 : f32
                %190 = arith.addi %84, %c28 : index
                %191 = arith.addi %190, %arg2 : index
                %extracted_20 = tensor.extract %63[%72, %191] : tensor<?x?xf32>
                %192 = arith.shrui %184, %c4_i32 : i32
                %193 = arith.andi %192, %c15_i32 : i32
                %194 = arith.uitofp %193 : i32 to f32
                %195 = arith.mulf %extracted_20, %194 : f32
                %196 = arith.addf %153, %159 : f32
                %197 = arith.addf %196, %165 : f32
                %198 = arith.addf %197, %171 : f32
                %199 = arith.addf %198, %177 : f32
                %200 = arith.addf %199, %183 : f32
                %201 = arith.addf %200, %189 : f32
                %202 = arith.addf %201, %195 : f32
                %203 = arith.addf %arg3, %202 : f32
                scf.yield %203 : f32
              }
              %142 = arith.mulf %85, %104 : f32
              %143 = arith.addf %141, %142 : f32
              %144 = arith.mulf %95, %143 : f32
              %145 = arith.mulf %144, %107 : f32
              %146 = arith.addf %arg1, %145 : f32
              scf.yield %146 : f32
            }
            linalg.yield %79 : f32
          } -> tensor<?xf32>
          iree_tensor_ext.dispatch.tensor.store %67, %59, offsets = [0], sizes = [%51], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%51}
          return
        }
        iree_codegen.dispatch_config @iq1_xxs_dispatch_0_elementwise_broadcast_D_f32 {
        ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
          %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
          iree_codegen.yield %x, %y, %z : index, index, index
        }
      }
    }
  }
  util.func public @iq1_xxs(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: i32, %arg4: !hal.fence, %arg5: !hal.fence) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @iq1_xxs(%input0: tensor<?x?xf32>, %input1: tensor<?xi8>, %input2: tensor<?xi32>, %input3: i32) -> (%output0: tensor<?x?xf32>)", iree.abi.model = "coarse-fences"}} {
    %c7168 = arith.constant 7168 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32_i64 = arith.constant 32 : i64
    %__constant_tensor_1024xi32 = util.global.load immutable @__constant_tensor_1024xi32 : !stream.resource<constant>
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = arith.muli %0, %c4 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf32>{%0, %1} in !stream.resource<external>{%3}
    %5 = stream.timepoint.import on(#hal.device.affinity<@__device_0>) %arg4 : (!hal.fence) => !stream.timepoint
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %element_type_i8 = hal.element_type<i8> : i32
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%6]) type(%element_type_i8) encoding(%dense_row_major)
    %7 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?xi8>{%6} in !stream.resource<external>{%6}
    %8 = hal.buffer_view.dim<%arg2 : !hal.buffer_view>[0] : index
    %element_type_i32 = hal.element_type<i32> : i32
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input2") shape([%8]) type(%element_type_i32) encoding(%dense_row_major)
    %9 = arith.muli %8, %c4 : index
    %10 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg2 : !hal.buffer_view -> tensor<?xi32>{%8} in !stream.resource<external>{%9}
    %11 = arith.index_cast %arg3 : i32 to index
    %12 = arith.divui %1, %c32 : index
    %13 = arith.muli %0, %11 overflow<nsw> : index
    %14 = arith.muli %13, %c32 overflow<nsw> : index
    %15 = arith.muli %13, %c128 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%5) => !stream.resource<transient>{%15} => !stream.timepoint
    %16 = arith.index_castui %14 : index to i64
    %17 = arith.index_castui %14 : index to i32
    %18 = arith.shrui %16, %c32_i64 : i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.index_castui %11 : index to i64
    %21 = arith.index_castui %11 : index to i32
    %22 = arith.shrui %20, %c32_i64 : i64
    %23 = arith.trunci %22 : i64 to i32
    %24 = arith.index_castui %12 : index to i64
    %25 = arith.index_castui %12 : index to i32
    %26 = arith.shrui %24, %c32_i64 : i64
    %27 = arith.trunci %26 : i64 to i32
    %28 = arith.index_castui %8 : index to i64
    %29 = arith.index_castui %8 : index to i32
    %30 = arith.shrui %28, %c32_i64 : i64
    %31 = arith.trunci %30 : i64 to i32
    %32 = arith.index_castui %0 : index to i64
    %33 = arith.index_castui %0 : index to i32
    %34 = arith.shrui %32, %c32_i64 : i64
    %35 = arith.trunci %34 : i64 to i32
    %36 = arith.index_castui %1 : index to i64
    %37 = arith.index_castui %1 : index to i32
    %38 = arith.shrui %36, %c32_i64 : i64
    %39 = arith.trunci %38 : i64 to i32
    %40 = arith.index_castui %6 : index to i64
    %41 = arith.index_castui %6 : index to i32
    %42 = arith.shrui %40, %c32_i64 : i64
    %43 = arith.trunci %42 : i64 to i32
    %44 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%10 as %arg6: !stream.resource<external>{%9}, %4 as %arg7: !stream.resource<external>{%3}, %7 as %arg8: !stream.resource<external>{%6}, %__constant_tensor_1024xi32 as %arg9: !stream.resource<constant>{%c7168}, %result as %arg10: !stream.resource<transient>{%15}) {
      stream.cmd.dispatch @iq1_xxs_dispatch_0::@vulkan_spirv_fb::@iq1_xxs_dispatch_0_elementwise_broadcast_D_f32[%14, %11, %12, %8, %0, %1, %6](%17, %19, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39, %41, %43 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        ro %arg6[%c0 for %9] : !stream.resource<external>{%9},
        ro %arg7[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg8[%c0 for %6] : !stream.resource<external>{%6},
        ro %arg9[%c0 for %c7168] : !stream.resource<constant>{%c7168},
        wo %arg10[%c0 for %15] : !stream.resource<transient>{%15}
      }
    } => !stream.timepoint
    %45 = stream.timepoint.await %44 => %result : !stream.resource<transient>{%15}
    %46 = util.optimization_barrier %45 : !stream.resource<transient>
    %47 = arith.muli %13, %c4 : index
    %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%47} => !stream.timepoint
    %48 = arith.index_castui %13 : index to i64
    %49 = arith.index_castui %13 : index to i32
    %50 = arith.shrui %48, %c32_i64 : i64
    %51 = arith.trunci %50 : i64 to i32
    %52 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint_1) => with(%46 as %arg6: !stream.resource<transient>{%15}, %result_0 as %arg7: !stream.resource<external>{%47}) {
      stream.cmd.dispatch @iq1_xs_dispatch_1::@vulkan_spirv_fb::@iq1_xs_dispatch_1_reduction_Dx32_f32[%13](%49, %51 : i32, i32) {
        ro %arg6[%c0 for %15] : !stream.resource<transient>{%15},
        wo %arg7[%c0 for %47] : !stream.resource<external>{%47}
      }
    } => !stream.timepoint
    %53 = arith.muli %2, %11 : index
    stream.timepoint.chain_external on(#hal.device.affinity<@__device_0>) %52 => (%arg5 : !hal.fence)
    %54 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %result_0 : tensor<?x?xf32>{%0, %11} in !stream.resource<external>{%53} -> !hal.buffer_view
    util.return %54 : !hal.buffer_view
  }
  hal.executable private @iq1_xxxs_dispatch_0 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>) {
      hal.executable.export public @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = [64, 1, 1]>} {
          %c32_i64 = arith.constant 32 : i64
          %c28 = arith.constant 28 : index
          %c24 = arith.constant 24 : index
          %c20 = arith.constant 20 : index
          %c16 = arith.constant 16 : index
          %c12 = arith.constant 12 : index
          %c4_i32 = arith.constant 4 : i32
          %c15_i32 = arith.constant 15 : i32
          %c3 = arith.constant 3 : index
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c3_i32 = arith.constant 3 : i32
          %c7_i32 = arith.constant 7 : i32
          %c4 = arith.constant 4 : index
          %c2 = arith.constant 2 : index
          %c34 = arith.constant 34 : index
          %c8_i32 = arith.constant 8 : i32
          %c255_i32 = arith.constant 255 : i32
          %c1 = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c38 = arith.constant 38 : index
          %c256 = arith.constant 256 : index
          %cst = arith.constant -8.750000e-01 : f32
          %cst_0 = arith.constant -1.125000e+00 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %c6144 = arith.constant 6144 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
          %2 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
          %3 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
          %4 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
          %5 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
          %6 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
          %7 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
          %8 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(8) : i32
          %9 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(9) : i32
          %10 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(10) : i32
          %11 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(11) : i32
          %12 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(12) : i32
          %13 = hal.interface.constant.load layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(13) : i32
          %14 = arith.extui %1 : i32 to i64
          %15 = arith.shli %14, %c32_i64 : i64
          %16 = arith.extui %0 : i32 to i64
          %17 = arith.ori %16, %15 : i64
          %18 = arith.index_castui %17 : i64 to index
          %19 = arith.extui %3 : i32 to i64
          %20 = arith.shli %19, %c32_i64 : i64
          %21 = arith.extui %2 : i32 to i64
          %22 = arith.ori %21, %20 : i64
          %23 = arith.index_castui %22 : i64 to index
          %24 = arith.extui %5 : i32 to i64
          %25 = arith.shli %24, %c32_i64 : i64
          %26 = arith.extui %4 : i32 to i64
          %27 = arith.ori %26, %25 : i64
          %28 = arith.index_castui %27 : i64 to index
          %29 = arith.extui %7 : i32 to i64
          %30 = arith.shli %29, %c32_i64 : i64
          %31 = arith.extui %6 : i32 to i64
          %32 = arith.ori %31, %30 : i64
          %33 = arith.index_castui %32 : i64 to index
          %34 = arith.extui %9 : i32 to i64
          %35 = arith.shli %34, %c32_i64 : i64
          %36 = arith.extui %8 : i32 to i64
          %37 = arith.ori %36, %35 : i64
          %38 = arith.index_castui %37 : i64 to index
          %39 = arith.extui %11 : i32 to i64
          %40 = arith.shli %39, %c32_i64 : i64
          %41 = arith.extui %10 : i32 to i64
          %42 = arith.ori %41, %40 : i64
          %43 = arith.index_castui %42 : i64 to index
          %44 = arith.extui %13 : i32 to i64
          %45 = arith.shli %44, %c32_i64 : i64
          %46 = arith.extui %12 : i32 to i64
          %47 = arith.ori %46, %45 : i64
          %48 = arith.index_castui %47 : i64 to index
          %49:6 = util.assume.int 
              %18<udiv = 32>, 
              %28<umin = 0, umax = 281474976710655>, 
              %33<umin = 0, umax = 9007199254740991>, 
              %38<umin = 0, umax = 9007199254740991>, 
              %43<umin = 0, umax = 9007199254740991>, 
              %48<umin = 0, umax = 9007199254740991>
            : index, index, index, index, index, index
          %50 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c6144) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xi32>>
          %51 = iree_tensor_ext.dispatch.workload.ordinal %49#0, 0 : index
          %52 = iree_tensor_ext.dispatch.workload.ordinal %49#2, 3 : index
          %53 = iree_tensor_ext.dispatch.workload.ordinal %49#3, 4 : index
          %54 = iree_tensor_ext.dispatch.workload.ordinal %49#4, 5 : index
          %55 = iree_tensor_ext.dispatch.workload.ordinal %49#5, 6 : index
          %56 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>{%52}
          %57 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%53, %54}
          %58 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%55}
          %59 = hal.interface.binding.subspan layout(<constants = 14, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%51}
          %60 = iree_tensor_ext.dispatch.workload.ordinal %23, 1 : index
          %61 = iree_tensor_ext.dispatch.workload.ordinal %49#1, 2 : index
          %62 = iree_tensor_ext.dispatch.tensor.load %56, offsets = [0], sizes = [%52], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>{%52} -> tensor<?xi32>
          %63 = iree_tensor_ext.dispatch.tensor.load %57, offsets = [0, 0], sizes = [%53, %54], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%53, %54} -> tensor<?x?xf32>
          %64 = iree_tensor_ext.dispatch.tensor.load %58, offsets = [0], sizes = [%55], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%55} -> tensor<?xi8>
          %65 = iree_tensor_ext.dispatch.tensor.load %50, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xi32>> -> tensor<256xi32>
          %66 = tensor.empty(%51) : tensor<?xf32>
          %67 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%66 : tensor<?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64], [1]]>} {
          ^bb0(%out: f32):
            %68 = linalg.index 0 : index
            %69 = arith.remsi %68, %c32 : index
            %70 = arith.divsi %68, %c32 : index
            %71 = arith.remsi %70, %60 : index
            %72 = arith.divsi %70, %60 : index
            %extracted = tensor.extract %62[%72] : tensor<?xi32>
            %73 = arith.index_cast %extracted : i32 to index
            %74 = arith.muli %73, %60 : index
            %75 = arith.addi %74, %71 : index
            %76 = arith.divui %54, %c256 : index
            %77 = arith.muli %75, %76 : index
            %78 = arith.muli %77, %c38 : index
            %79 = scf.for %arg0 = %69 to %61 step %c32 iter_args(%arg1 = %cst_1) -> (f32) {
              %80 = arith.divui %arg0, %c8 : index
              %81 = arith.remui %arg0, %c8 : index
              %82 = arith.muli %80, %c38 : index
              %83 = arith.addi %78, %82 : index
              %84 = arith.muli %arg0, %c32 : index
              %85 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst_1) -> (f32) {
                %137 = arith.addi %84, %arg2 : index
                %extracted_13 = tensor.extract %63[%72, %137] : tensor<?x?xf32>
                %138 = arith.addf %arg3, %extracted_13 : f32
                scf.yield %138 : f32
              }
              %extracted_2 = tensor.extract %64[%83] : tensor<?xi8>
              %86 = arith.addi %83, %c1 : index
              %extracted_3 = tensor.extract %64[%86] : tensor<?xi8>
              %87 = arith.extui %extracted_2 : i8 to i32
              %88 = arith.extui %extracted_3 : i8 to i32
              %89 = arith.andi %87, %c255_i32 : i32
              %90 = arith.andi %88, %c255_i32 : i32
              %91 = arith.shli %90, %c8_i32 : i32
              %92 = arith.ori %89, %91 : i32
              %93 = arith.trunci %92 : i32 to i16
              %94 = arith.bitcast %93 : i16 to f16
              %95 = arith.extf %94 : f16 to f32
              %96 = arith.addi %83, %c34 : index
              %97 = arith.divui %81, %c2 : index
              %98 = arith.addi %96, %97 : index
              %extracted_4 = tensor.extract %64[%98] : tensor<?xi8>
              %99 = arith.extui %extracted_4 : i8 to i32
              %100 = arith.andi %99, %c255_i32 : i32
              %101 = arith.remui %81, %c2 : index
              %102 = arith.muli %101, %c4 : index
              %103 = arith.index_castui %102 : index to i32
              %104 = arith.shrui %100, %103 : i32
              %105 = arith.andi %104, %c7_i32 : i32
              %106 = arith.shrui %104, %c3_i32 : i32
              %107 = arith.andi %106, %c1_i32 : i32
              %108 = arith.cmpi ne, %107, %c0_i32 : i32
              %109 = arith.select %108, %cst_0, %cst : f32
              %110 = arith.muli %105, %c2_i32 : i32
              %111 = arith.addi %110, %c1_i32 : i32
              %112 = arith.uitofp %111 : i32 to f32
              %113 = arith.muli %81, %c4 : index
              %114 = arith.addi %113, %c2 : index
              %115 = arith.addi %83, %114 : index
              %extracted_5 = tensor.extract %64[%115] : tensor<?xi8>
              %116 = arith.extui %extracted_5 : i8 to i32
              %117 = arith.andi %116, %c255_i32 : i32
              %118 = arith.index_castui %117 : i32 to index
              %extracted_6 = tensor.extract %65[%118] : tensor<256xi32>
              %119 = arith.addi %115, %c1 : index
              %extracted_7 = tensor.extract %64[%119] : tensor<?xi8>
              %120 = arith.extui %extracted_7 : i8 to i32
              %121 = arith.andi %120, %c255_i32 : i32
              %122 = arith.index_castui %121 : i32 to index
              %extracted_8 = tensor.extract %65[%122] : tensor<256xi32>
              %123 = arith.addi %115, %c2 : index
              %extracted_9 = tensor.extract %64[%123] : tensor<?xi8>
              %124 = arith.extui %extracted_9 : i8 to i32
              %125 = arith.andi %124, %c255_i32 : i32
              %126 = arith.index_castui %125 : i32 to index
              %extracted_10 = tensor.extract %65[%126] : tensor<256xi32>
              %127 = arith.addi %115, %c3 : index
              %extracted_11 = tensor.extract %64[%127] : tensor<?xi8>
              %128 = arith.extui %extracted_11 : i8 to i32
              %129 = arith.andi %128, %c255_i32 : i32
              %130 = arith.index_castui %129 : i32 to index
              %extracted_12 = tensor.extract %65[%130] : tensor<256xi32>
              %131 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %cst_1) -> (f32) {
                %137 = arith.index_castui %arg2 : index to i32
                %138 = arith.muli %137, %c8_i32 : i32
                %139 = arith.shrui %extracted_6, %138 : i32
                %140 = arith.addi %84, %arg2 : index
                %extracted_13 = tensor.extract %63[%72, %140] : tensor<?x?xf32>
                %141 = arith.andi %139, %c15_i32 : i32
                %142 = arith.uitofp %141 : i32 to f32
                %143 = arith.mulf %extracted_13, %142 : f32
                %144 = arith.addi %84, %c4 : index
                %145 = arith.addi %144, %arg2 : index
                %extracted_14 = tensor.extract %63[%72, %145] : tensor<?x?xf32>
                %146 = arith.shrui %139, %c4_i32 : i32
                %147 = arith.andi %146, %c15_i32 : i32
                %148 = arith.uitofp %147 : i32 to f32
                %149 = arith.mulf %extracted_14, %148 : f32
                %150 = arith.shrui %extracted_8, %138 : i32
                %151 = arith.addi %84, %c8 : index
                %152 = arith.addi %151, %arg2 : index
                %extracted_15 = tensor.extract %63[%72, %152] : tensor<?x?xf32>
                %153 = arith.andi %150, %c15_i32 : i32
                %154 = arith.uitofp %153 : i32 to f32
                %155 = arith.mulf %extracted_15, %154 : f32
                %156 = arith.addi %84, %c12 : index
                %157 = arith.addi %156, %arg2 : index
                %extracted_16 = tensor.extract %63[%72, %157] : tensor<?x?xf32>
                %158 = arith.shrui %150, %c4_i32 : i32
                %159 = arith.andi %158, %c15_i32 : i32
                %160 = arith.uitofp %159 : i32 to f32
                %161 = arith.mulf %extracted_16, %160 : f32
                %162 = arith.shrui %extracted_10, %138 : i32
                %163 = arith.addi %84, %c16 : index
                %164 = arith.addi %163, %arg2 : index
                %extracted_17 = tensor.extract %63[%72, %164] : tensor<?x?xf32>
                %165 = arith.andi %162, %c15_i32 : i32
                %166 = arith.uitofp %165 : i32 to f32
                %167 = arith.mulf %extracted_17, %166 : f32
                %168 = arith.addi %84, %c20 : index
                %169 = arith.addi %168, %arg2 : index
                %extracted_18 = tensor.extract %63[%72, %169] : tensor<?x?xf32>
                %170 = arith.shrui %162, %c4_i32 : i32
                %171 = arith.andi %170, %c15_i32 : i32
                %172 = arith.uitofp %171 : i32 to f32
                %173 = arith.mulf %extracted_18, %172 : f32
                %174 = arith.shrui %extracted_12, %138 : i32
                %175 = arith.addi %84, %c24 : index
                %176 = arith.addi %175, %arg2 : index
                %extracted_19 = tensor.extract %63[%72, %176] : tensor<?x?xf32>
                %177 = arith.andi %174, %c15_i32 : i32
                %178 = arith.uitofp %177 : i32 to f32
                %179 = arith.mulf %extracted_19, %178 : f32
                %180 = arith.addi %84, %c28 : index
                %181 = arith.addi %180, %arg2 : index
                %extracted_20 = tensor.extract %63[%72, %181] : tensor<?x?xf32>
                %182 = arith.shrui %174, %c4_i32 : i32
                %183 = arith.andi %182, %c15_i32 : i32
                %184 = arith.uitofp %183 : i32 to f32
                %185 = arith.mulf %extracted_20, %184 : f32
                %186 = arith.addf %143, %149 : f32
                %187 = arith.addf %186, %155 : f32
                %188 = arith.addf %187, %161 : f32
                %189 = arith.addf %188, %167 : f32
                %190 = arith.addf %189, %173 : f32
                %191 = arith.addf %190, %179 : f32
                %192 = arith.addf %191, %185 : f32
                %193 = arith.addf %arg3, %192 : f32
                scf.yield %193 : f32
              }
              %132 = arith.mulf %85, %109 : f32
              %133 = arith.addf %131, %132 : f32
              %134 = arith.mulf %95, %133 : f32
              %135 = arith.mulf %134, %112 : f32
              %136 = arith.addf %arg1, %135 : f32
              scf.yield %136 : f32
            }
            linalg.yield %79 : f32
          } -> tensor<?xf32>
          iree_tensor_ext.dispatch.tensor.store %67, %59, offsets = [0], sizes = [%51], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%51}
          return
        }
        iree_codegen.dispatch_config @iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32 {
        ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
          %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
          iree_codegen.yield %x, %y, %z : index, index, index
        }
      }
    }
  }
  util.func public @iq1_xxxs(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: i32, %arg4: !hal.fence, %arg5: !hal.fence) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @iq1_xxxs(%input0: tensor<?x?xf32>, %input1: tensor<?xi8>, %input2: tensor<?xi32>, %input3: i32) -> (%output0: tensor<?x?xf32>)", iree.abi.model = "coarse-fences"}} {
    %c7168 = arith.constant 7168 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32_i64 = arith.constant 32 : i64
    %__constant_tensor_1024xi32 = util.global.load immutable @__constant_tensor_1024xi32 : !stream.resource<constant>
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = arith.muli %0, %c4 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf32>{%0, %1} in !stream.resource<external>{%3}
    %5 = stream.timepoint.import on(#hal.device.affinity<@__device_0>) %arg4 : (!hal.fence) => !stream.timepoint
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %element_type_i8 = hal.element_type<i8> : i32
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%6]) type(%element_type_i8) encoding(%dense_row_major)
    %7 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?xi8>{%6} in !stream.resource<external>{%6}
    %8 = hal.buffer_view.dim<%arg2 : !hal.buffer_view>[0] : index
    %element_type_i32 = hal.element_type<i32> : i32
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input2") shape([%8]) type(%element_type_i32) encoding(%dense_row_major)
    %9 = arith.muli %8, %c4 : index
    %10 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg2 : !hal.buffer_view -> tensor<?xi32>{%8} in !stream.resource<external>{%9}
    %11 = arith.index_cast %arg3 : i32 to index
    %12 = arith.divui %1, %c32 : index
    %13 = arith.muli %0, %11 overflow<nsw> : index
    %14 = arith.muli %13, %c32 overflow<nsw> : index
    %15 = arith.muli %13, %c128 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%5) => !stream.resource<transient>{%15} => !stream.timepoint
    %16 = arith.index_castui %14 : index to i64
    %17 = arith.index_castui %14 : index to i32
    %18 = arith.shrui %16, %c32_i64 : i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.index_castui %11 : index to i64
    %21 = arith.index_castui %11 : index to i32
    %22 = arith.shrui %20, %c32_i64 : i64
    %23 = arith.trunci %22 : i64 to i32
    %24 = arith.index_castui %12 : index to i64
    %25 = arith.index_castui %12 : index to i32
    %26 = arith.shrui %24, %c32_i64 : i64
    %27 = arith.trunci %26 : i64 to i32
    %28 = arith.index_castui %8 : index to i64
    %29 = arith.index_castui %8 : index to i32
    %30 = arith.shrui %28, %c32_i64 : i64
    %31 = arith.trunci %30 : i64 to i32
    %32 = arith.index_castui %0 : index to i64
    %33 = arith.index_castui %0 : index to i32
    %34 = arith.shrui %32, %c32_i64 : i64
    %35 = arith.trunci %34 : i64 to i32
    %36 = arith.index_castui %1 : index to i64
    %37 = arith.index_castui %1 : index to i32
    %38 = arith.shrui %36, %c32_i64 : i64
    %39 = arith.trunci %38 : i64 to i32
    %40 = arith.index_castui %6 : index to i64
    %41 = arith.index_castui %6 : index to i32
    %42 = arith.shrui %40, %c32_i64 : i64
    %43 = arith.trunci %42 : i64 to i32
    %44 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%10 as %arg6: !stream.resource<external>{%9}, %4 as %arg7: !stream.resource<external>{%3}, %7 as %arg8: !stream.resource<external>{%6}, %__constant_tensor_1024xi32 as %arg9: !stream.resource<constant>{%c7168}, %result as %arg10: !stream.resource<transient>{%15}) {
      stream.cmd.dispatch @iq1_xxxs_dispatch_0::@vulkan_spirv_fb::@iq1_xxxs_dispatch_0_elementwise_broadcast_D_f32[%14, %11, %12, %8, %0, %1, %6](%17, %19, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39, %41, %43 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        ro %arg6[%c0 for %9] : !stream.resource<external>{%9},
        ro %arg7[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg8[%c0 for %6] : !stream.resource<external>{%6},
        ro %arg9[%c0 for %c7168] : !stream.resource<constant>{%c7168},
        wo %arg10[%c0 for %15] : !stream.resource<transient>{%15}
      }
    } => !stream.timepoint
    %45 = stream.timepoint.await %44 => %result : !stream.resource<transient>{%15}
    %46 = util.optimization_barrier %45 : !stream.resource<transient>
    %47 = arith.muli %13, %c4 : index
    %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%47} => !stream.timepoint
    %48 = arith.index_castui %13 : index to i64
    %49 = arith.index_castui %13 : index to i32
    %50 = arith.shrui %48, %c32_i64 : i64
    %51 = arith.trunci %50 : i64 to i32
    %52 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint_1) => with(%46 as %arg6: !stream.resource<transient>{%15}, %result_0 as %arg7: !stream.resource<external>{%47}) {
      stream.cmd.dispatch @iq1_xs_dispatch_1::@vulkan_spirv_fb::@iq1_xs_dispatch_1_reduction_Dx32_f32[%13](%49, %51 : i32, i32) {
        ro %arg6[%c0 for %15] : !stream.resource<transient>{%15},
        wo %arg7[%c0 for %47] : !stream.resource<external>{%47}
      }
    } => !stream.timepoint
    %53 = arith.muli %2, %11 : index
    stream.timepoint.chain_external on(#hal.device.affinity<@__device_0>) %52 => (%arg5 : !hal.fence)
    %54 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %result_0 : tensor<?x?xf32>{%0, %11} in !stream.resource<external>{%53} -> !hal.buffer_view
    util.return %54 : !hal.buffer_view
  }
}