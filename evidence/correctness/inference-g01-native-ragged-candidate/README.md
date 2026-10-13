# G01 Apple/Vulkan native ragged 候选交付与宿主风险

不可变candidate `280f6f92ea97bf3b424113346f0264fb7b20df16`，活动
[卡](../../../docs/implementation/tasks/compute/inference-g01-native-ragged-layout.md)。
[receipt](receipt.json)与[raw](raw.zip)保实际命令/耗时/产物位置、公开consumer源码和
插桩compile commands。[实现阶段](../inference-g01-native-ragged-stage/README.md)保所有
compile/run FAIL和source四/两tests；这里没有performance/CLOSED或整个G01声明。

Air Mac16,12实际relocated static/shared，Apple与Vulkan各两个public tests PASS，只有
installed headers与hyle::compute；本地测试scalar bit carriers不假装产品API/依赖。
两个与八request、三D、F16以及AppleBF16、empty/reorder/旧results/readonly/tail/zero
allocation真实消费；四个prefix实际改名staging→relocated。对应四SKU单独public batch
头object compile PASS；不制造header-only binary journey或覆盖旧prefix。

ASan/UBSan/TSan各自复用原native-host树，Apple每种两tests、Vk每种四tests全部PASS；
新产品batch TU/public测试与Vk根/原accepted prefix失败test真实fsanitize命令核对。
ASan detect_leaks=0，原平台限制保留，不签LSan/shader/driver/private GPU内存。
Windows仅原Vulkan库四jobs定向编译PASS，未运行本机GPU，没有重新执行全库矩阵。

原native数学/Shader/Work/Queue/actual retirement未改，新冷入口仅组合已有View/SDPA/
Concat。P1比较器/资格仍待；不继承旧single或CPU性能，不签paged/growth/capture。
