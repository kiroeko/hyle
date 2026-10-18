# G01 CPU resident性能对照数学预检

产品candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f` 不变；
[receipt](receipt.json)与[raw.zip](raw.zip)记录实际Windows CPU-only static编译和oracle。
普通CPU默认入口与独立原ggml BF16 flash recipe通过同样的公开PreparedOperation检查、
CPU Queue admission、input/output mapping、真实Completion与caller-owned output归还。
对照用已有typed custom kernel factory与原MetadataArena/CpuExecution，不调用候选Provider
supports/lower，不新增runtime/数学核/产品Provider SPI。

实际BF16 BHSD GQA8/2 D256、Q1/128、KV1024/容量1037、causal、源只读、两个slots重放；
独立double oracle绝对8e-3原界线和有限性检查通过，两侧最大误差约3.05e-5。
原query-only F32 cast、cold F16 mask、最终BHSD/BF16恢复均计入对应阶段；没有完整KV
cast/展开。对照metadata/payload/cplan/mask实际字节单列，原私有pool上界仍UNKNOWN。
该证据只签预检，没有A/A、统一仪器contract/controls或timing verdict，不能关闭CPU owner。

复用原CPU-only构建树，configure113.77s、47个增量依赖/TU步骤2jobs build41.11s，
oracle3.33s；没有同时运行Air heavy任务，没有全库/大模型/GPU运行。
