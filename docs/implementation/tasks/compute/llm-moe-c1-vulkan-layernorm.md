# C1-Y33：Vulkan LayerNorm 成熟实现接入

## 当前合同与审计

- owner：`/root`；baseline `5f61546cd0f09ff735b2c01f3f28e33fa3044a2c`。
- 当前 owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、
  `graph_route_vulkan`；三者现有 C1 AUDITED 重新定界为下面有限 FIX。
- 需求：[C1](llm-moe-c1-tensor-state.md) 的 F04.2，最后一轴 LayerNorm、F32 累加、
  可选 scale/bias、F32/F16/BF16、zero/one/tails、普通 Direct/Graph 与状态/结果寿命。
  本地2B张量可作为输入；LayerNorm 组合是已冻结的正交基础库消费，不冒称 Qwen 原始 forward。
- 触发事实：当前 `region_supports` 只接 RMSNorm，LayerNorm 仍走 HYLE shader；
  [Provider 复核](llm-moe-c1-provider-reuse-audit.md) 已登记此接入余项。整包性能候选尚未稳定，
  本节先完成成熟接入及数值资格，不提前启动 P1。
- 来源：干净 ggml `456172ec733a135778adcd32d00e576a58232e45` 的 `ggml_group_norm`、
  Vulkan 分派和原版 `group_norm.comp`；HYLE 既有 schema、Region/Plan、Cast/Copy/alias。
  [源码身份](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-layernorm/source-audit.json)
  仅证明审计输入，不是 GPU 验收。

原版 `norm.comp` 的 E[x²]−E[x]² 不满足已要求的大均值小方差资格。原版 GroupNorm 已有先均值、
再 centered variance 的两遍算法；把连续每行映射成 `[width,1,1,rows]`、group=1 即为逐行
LayerNorm，保持 F32 数学及原厂 shader。其固定512线程、X维每行一个工作组、descriptor 对齐
与 metadata/输出 owner 约束必须在接入层处理。未看到结果前不声明性能优势。

## SCOPED / AUDITED：有限 FIX

| ID | 固定改动与边界 |
|---|---|
| Y33-A | Provider 的 F32 dense region 增加 LayerNorm。冷期 reshape → 原版 GroupNorm → 恢复 shape → 可选原版 Mul/Add；无 affine 时用原版 Cont 建立实际输出 owner，不能把 externally bound 输出直接设为内部 view。每个 LayerNorm 预留至多5个 tensor/node 的 arena 容量。pin、shader、vendor patch 和原生运行机制不变。 |
| Y33-B | route 准备期匹配512线程、行数工作组和 F32 暂存容量；保留设备/容量不适用时的已有合法 native 路径。非连续、rank>4、F16/BF16 用既有 Cast/Copy 与无复制 dense alias 接成 F32 二维 region；scale/bias 分别转换，affine 后仅一次 RNE 写回 half。空域仍无 GPU 工作；不新增归一化算法或公共 API。 |
| Y33-C | GroupNorm 的输入必须 descriptor 对齐。沿现有 region 冷适配把动态/偏移常量输入变成已规划的对齐存储；同时覆盖普通 Direct、Graph、Recorder、带前后缀的组合与混合参数。已有 region/native Run、结果槽、失败和 quota owner 继续共用，不新增运行时选择器。 |
| Y33-D | 先用独立两遍 FP64/明确 half RNE 对真实成熟路径作资格回归；新增零方差/epsilon0、NaN/Inf、大均值小方差、宽1/127/129/257/2048、rank8、odd byte stride/origin、scale/bias 类型组合、变输入/双槽/旧结果及冷限制测试。测试必须证明 managed 路径被执行，不能只证明 fallback 数值正确。 |

本节只接 F04.2；归约/Softmax 和其余已登记 Provider 接入余项仍待各自有限审计。现有自实现
保留为必要未覆盖域和同合同比较候选；删除必须等实际覆盖及完整成本裁决，不凭本节提前淘汰。
不改公共构建系统、selector、性能统计核心、CPU 资格 blocker 或其他模块。

## 成本与适用验证

P0：F32 dense region 使用原版两遍中心化核，affine 为至多两个原版逐元素节点；无 affine 时
输出拥有独立端口的 Cont 成本显式保留。half/非连续适配计入必要的格式转换、连续化、暂存及
最终舍入；rank flatten 本身只有冷 metadata。512线程/行数/容量在准备期固定，执行不搜索、
计时或更换 Provider。region 的原厂编码/私有分配与 HYLE 边界同步仍按既有真实机制计费。

- 首先执行 W1 的新增资格、provider/Direct/Graph 适用回归；同候选 W1 static/shared、Air/Mini
  对应 factors 和 source contract。使用已有短路径/cache，build2/CTest1，重型步骤串行。
- Windows/Air 的 source/static/shared 普通消费加入本实际 LayerNorm 片段；覆盖不同参数、
  batch/尾宽、重绑定、输出保留及关闭。按 selector 执行所选门，未变且身份匹配的证据复用；
  Y32 已完成的无关工程门不因文档或本私有适配原样重跑。
- Windows DLL 邻 exe、PATH 隔离、禁错误弹窗；Air validation，Mini 只签 correctness/portability。
- 不运行 P1；API/算法与全部 C1 成熟接入固定后执行原 C1 bundle，n15/上限31、5%门及等责原则不变。
- 此具名接入来源调查到此停止；产品纠正最多两轮。首次失败完整保留；预算到点按反例裁决，
  不改门限、切换成自写归一化或扩泛算子调查。

审计时裁决：审计完成，以上 FIX 已固定，产品尚未实施，适用验收未运行。C1 整包与共同矩阵
未签收；C1 全部适用验收完成后通知用户，在 C2 前等待提问。

## 首个实现候选

审计提交`5f94bed4`之后，Y33-A–D已落代码：原版GroupNorm及输出owner恢复、F32 region、
half/stride/rank8适配、冷设备/暂存限制与descriptor对齐；新增真实managed路径、偏移常量和
独立参考回归，更新混合图的实际F32暂存账单。产品/测试首次构建和GPU验证尚未运行，
不能签数值资格、性能或三交付。后续先执行W1三个受影响factor，再沿本卡完成其余适用门。

## 首次构建与有限纠正

候选`4f730371`的W1 configure通过，build因MSVC C4456/WERROR停止：LayerNorm分支的
局部`norm`遮蔽同一if/else链的RmsNorm变量；GPU测试未启动。保留原始失败日志和receipt，
仅把新分支变量改为`layer`。同时源码复核发现新增冷限制测试直接调用未导出的私有函数，
改为经公开`prepare`入口、临时降低设备的准备期限额验证native fallback；在断言或运行前
恢复原值，不为测试导出私有符号。产品纠正预算已用1/2，算法纠正0；其余适用验收仍未运行。

第二次构建`656b6afa`已编译新的产品文件，停止于新增测试的两个编译错误：缺少直接包含的
`<algorithm>`，以及动态Argument列表调用需显式构造`Operation`。仅修正测试调用与include；
两次失败均未启动GPU，累计纠正2/2、算法纠正0。完成此固定修正后运行一次定向资格；若出现
新的实质反例，先依法重新定界，不继续盲目修补或重测。

## 固定候选验收与局部裁决（2026-09-19）

不可变候选 `4d255a5174825dc669de1684affd7729d3b430d7` 已完成本卡全部适用正确性、
资源与交付验证。[证据汇总](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-layernorm/README.md)
保留命令、首败、原始日志、候选/工具链/输入身份及各平台收据。

| 适用项 | 实际结果 |
|---|---|
| W1 static / shared | 分别132 / 131项 C++ 回归通过；shared 无既有 static-only 私有用例。Provider source contract、MSVC 组合夹具及适用头检查通过。 |
| Air `Mac16,12` | 132项 C++ 回归和 source contract 通过，Vulkan validation 开启且无错误。 |
| Mini `Mac16,10` | 131项 C++ 回归和 source contract 通过；未安装验证层，专用 `ExplicitValidationUsesItsOwnInstance` 明确 SKIPPED，由 W1/Air 覆盖该风险。Mini 仅签正确性/可移植性。 |
| selected 工程门 | 9/9通过；111个未选项及理由保留。Compute/Graph 的默认 source/static/shared 六条消费按本次实际 Vulkan 路线隔离执行。 |
| 本实际 LayerNorm 三交付 | W1/Air 各 source/static/shared，六条公开 Direct/Graph 消费共360例、1080次执行；旧结果在关闭后可读、quota 最终归零。 |

实际消费使用已验证本地2B BF16 embedding 与参数夹具，F32/BF16、0/1/2个 affine 参数、
batch1/2、宽1/129/257/2048和17/23/128/129行；每例一个快照槽、一个复用工作槽，三次变输入。
独立两遍 FP64 参考及最终 F32/BF16 RNE，沿原门限，没有以 ggml 输出自证正确。
该 LayerNorm 是正交基础能力消费，不声称为 Qwen 原始 forward 或整模型推理。

两次产品/测试编译纠正后无新产品改动。第三次构建已通过，但首轮运行因任务夹具缺少
validation-layer 环境在 Context 创建前失败；补齐既定运行环境，同一 EXE 通过。
默认交付脚本首次拉入无关 CPU/oneDNN 构建，在测试前停止并保存为 ABORTED_SCOPE；用现有
开关固定 CPU/oneDNN OFF、Vulkan ON。其后任务 toolchain 的 Vulkan SDK 查找和强制
`CMAKE_PREFIX_PATH` 分别阻止 configure/已安装消费；改为具体 Vulkan/SPIRV 查找变量，
只续跑未完成阶段，没有改公共交付工具或重复已通过的 producer。
外部 consumer 首次编译触发 double→float 的 WERROR，改为显式转换（预算1/2），GPU未启动，
oracle与产品不变。Mini 夹具最初误要求验证层专用用例，续跑仅补未运行的 Direct/Graph；
恢复脚本一次 gtest JSON 字段假设错误也保留，未由此重复执行已通过用例。

Y33-A–D 的适用验收完成，停止本批重复验证；无 P1、性能赢家或旧实现删除裁决。
三个 owner 继续 C1 AUDITED，Reduce/Softmax 等成熟接入、四路线完整成本优选/淘汰、
CPU Q2 的独立测量资格阻塞和 C1 整包验收仍未完成；不触发 C1 完成通知，不进入 C2。

统一脚本按该不可变候选统计全部 Compute/Graph 实现 **48,424行 / 196文件**，
不含测试、bench及第三方，自检通过。证据归档核验后清理 W1 重复 producer/误启用构建与
Air 两个 producer/展开源码，共 **4,207,211,641逻辑字节**；保留 SDK、已安装消费者、
当前模块构建缓存和原始证据。Windows 清理收据的汇总字段曾为空，按未变的逐目录账单纠正，
保留原收据且未重复删除。
