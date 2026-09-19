# C1-Y34：Vulkan 成熟归约与概率接入

> **2026-09-19 授权移交：本页仅保留历史实施/验证事实。** 当前施工改由
> [新总计划 M0–M6](llm-moe-coverage-completion-plan.md)与[M0 卡](llm-moe-m0-provider-scope.md)管理；
> 下文旧 FIX、下一步与 AUDITED 叙述不再授权继续实施或性能重测。
> [所有自实现冻结](llm-moe-self-operator-freeze.md)，最后一次有限比较落败即删除。
> 若 registry 仍指向本页的 BLOCKED，原 blocker/resume_condition 继续有效，新 goal 不自动解除。

## SCOPED / AUDITED

- owner `/root`；baseline `3cd981b318f16e82f975848b976e085d7954c344`。
- 当前三个 owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`，
  保持 C1 AUDITED，仅对本卡有限 FIX 重新定界。CPU Q2 的 BLOCKED 不变。
- 需求：C1/F04 的真实 head128 L2、router/attention 概率；正交任意轴归约与广播mask。
  Sum/Mean/Softmax 已有成熟实现但尚未绑定，不能凭既有 HYLE shader 继续默认自写。
- 普通入口、共同数值/空域/NaN/Inf/±0、F32累加、half最终一次RNE、Graph结果/关闭/失败
  合同不变。继续使用现有 Program/region/Run、原厂内部资源及HYLE片段边界。
- 来源仅核对干净固定 ggml `456172ec733a135778adcd32d00e576a58232e45` 的行归约、概率
  API/shader/执行分派及当前HYLE适配；[源码与反例](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/source-audit.json)
  是审计输入，不是GPU或性能验收。

已定位两项具体约束。SumRows/Mean的XYZ分派补齐行数，而shader无尾行guard；沿已有Scan
接入方式分成至多三个完整行片段，原厂Concat组合，不发出多余行。长行Softmax的large2/3
各有一处 `max(max_val, vals[tid+s])`，每轮丢失先前累计峰值；128 lane、仅lane64为1000时
源码等价循环返回0，正确值1000。宽32769、末列峰值可触发该分块位置；修正为累积shared值，
保留原厂三阶段算法、调度与scratch。

| FIX | 固定范围 |
|---|---|
| Y34-A | F32 dense typed region接入最后轴keepdim Sum/Mean及普通Softmax。Sum/Mean尾行安全分片与必要输出shape/owner恢复；不自行实现归约核。Softmax无mask或同shape additive mask进入原厂程序，additive前的scale独立F32求值，禁止融合改变已冻结舍入顺序。 |
| Y34-B | 普通route冷期适配任意axes/keepdim、三浮点、rank≤8、合法byte-stride/View：必要Cast/Copy/不复制的permute/reshape；Sum/Mean使用F32行矩阵，Softmax恢复原轴。Boolean mask在缩放后Select为负无穷，复用既有Fill/Binary/Select；additive mask按实际广播域适配。half仅最终一次RNE，所有额外转换/复制/暂存实际计费。 |
| Y34-C | 冷期匹配subgroup/workgroup/grid/F32容量及长行原厂scratch界限；空域、rank0归约、Min/Max、LogSoftmax和设备不适用域保留现有合格路径。Softmax描述符输入/输出对齐走现有适配。只修large2/3累计max两行，使用现有精确patch机制、来源hash和依赖锁条目；不改pin、公共构建/selector/性能工具。 |
| Y34-D | 独立FP64参考、原数值门、真实managed路径证明；覆盖0/1、任意/负/多轴、rank8、half/stride、row511/512/513/262145、长行16384/16385/32769及末块极值、三种mask/scale0或负/NaN/Inf、additive中间舍入。Direct/Graph/Recorder、published中间值/分叉、双槽/变输入/旧结果/close/quota及已有OOM/drain回归。 |

## 成本、验证与停止

P0：Sum/Mean与原版Softmax算术仍为O(N)，尾行最多三个原厂片段；Sum分片的Concat可能保留
原厂临时存储。长行Softmax仍由原厂三个阶段、两份每分块F32统计数组执行。非末轴、half或
广播mask可产生F32整张暂存、转换与恢复；Boolean mask需要既有缩放/Fill/Select，不用省检查
或遗漏适配成本获胜。准备匹配与绑定一次，执行不搜索/计时。没有本批性能赢家声明。

- 当前先执行W1新增反例与三个受影响factor；再同候选W1 static/shared、Air/Mini适用回归及
  source contract/公开头。Windows DLL邻exe、PATH隔离、禁弹窗；Air验证层，Mini按已有角色。
- W1/Air实际本地2B张量的L2/概率片段进入source/static/shared，独立公式、重复运行、资源归零。
  原Y31/Y33片段不能代签这次新接线；适用完整三交付仍在C1内完成。
- 依赖锁内容变化可能扩展selector工程门；保留完整selected/skipped理由，未运行不写PASS。
  未变且身份匹配的证据复用；不得按文档提交重建，不反复执行同候选已通过的门。
- low-parallel2、重型串行；复用活动缓存，归档后及时清理本批重复producer和过期私有源码副本。
- 本批不跑P1。所有C1成熟接入稳定后，依原固定bundle完成等责性能和优选/淘汰；Min/Max和
  稳定LogSoftmax保留为已具名的语义缺口候选，不从此次借用推断自有实现胜出或删除。
- 来源调查到上述具名差异停止；产品纠正最多2轮，独立consumer编译纠正最多2轮。
  新反例超预算则按协议重新定界或BLOCKED，禁止放宽门限、用fallback假冒成熟路径通过。

审计裁决：有限FIX固定，产品尚未实施，适用验证未运行；C1整包、共同矩阵与性能未签收。
C1全部适用验收完成后通知用户，并在C2前等待提问。

## 首个实现候选

审计提交`dedb80d5`后，Y34-A–D已实施：region支持Sum/Mean/Softmax，归约尾行用原厂
分片/Concat处理；route完成任意轴/F32暂存/最终half舍入、mask与缩放顺序适配，冷期
使用已存在的subgroup/设备限额。原厂长行两处累计max已按精确hash patch修正，依赖锁仅
更新此patch目录内容与NOTICE身份。未改公共API、pin或治理/性能工具。
新增真实managed反例与cold grid测试，既有图的资源账单按新增适配更新。selector因依赖锁
扩为112个selected、8个未选；此处仅登记适用集合，首次编译/GPU资格尚未运行，不签PASS。

### W1 资格与 Air 首次编译修正

候选`05581df8`的W1 static/shared分别136/135项全部通过，source contract另5项通过；
无数值、生命周期或Vulkan validation失败。这仅签三个受影响factor，不代签112项工程门、
真实2B三交付或整包性能。启动脚本首次误用Windows PowerShell5.1，在configure前因缺少
`Path.IsPathFullyQualified`退出；改用既定PowerShell7后运行，原始首败日志保留。

Air首次build在新增测试的按值range-for触发Clang `-Wrange-loop-construct` / WERROR。
具名修正仅把`Operation`循环变量改为const引用，输入、断言及产品字节不变；保留首次失败，
从Windows提交测试修正后增量验证。产品/算法纠正仍为0，未运行Air/Mini不写PASS。

### 原生模块资格：已完成，Y34其余验收继续

产品候选`05581df8`；测试编译修正候选`11804be7`仅改变上述range-for及任务说明，产品输入
不变。W1 static/shared分别136/135项；Air136项全部通过，Mini135项通过、仅验证层专用用例
1项明确跳过（该风险由W1/Air覆盖）。两台Mac各5项source contract通过；W1增量只重跑受
测试修改影响的Direct因子91项并通过。无数值、生命周期、VUID或同步风险诊断。

[原始日志及身份](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/README.md)
保留首败、编译修正、每环境结果及CRC/SHA校验。产品/算法纠正0，测试编译纠正1，独立consumer
编译纠正0；本批没有P1。当前selector仍112项selected、8项未选，以上结果不能代签全部工程门。

下一有限动作是本地2B张量的L2/概率片段source/static/shared、适用公开头及工程余门；不重复
已经通过的原生模块回归。C1其余Provider覆盖核对、完整成本/默认规则/淘汰及CPU Q2 BLOCKED
保持，三owner继续AUDITED。Compute/Graph统一实现48,737行/197文件（不含测试/第三方）。
本批清理过期build-local第三方源码及Mini上一验证源码副本191,054,277逻辑字节，保留活动
构建、当前源码、SDK与共享缓存。

### 实际张量三交付及公开头：已完成

固定源码包候选`923895c5`（产品与`05581df8`相同），W1/Air的source/static/shared共六条
消费全部通过，每条72例/216次执行，合计432例/1296次。输入只取本地2B既有embedding
fixture，F32/BF16，形状`1×1×128`、`1×17×129`、`2×257×128`；Sum/Mean、无mask/
Boolean/additive Softmax和L2，独立FP64公式及操作合同中的F32边界，原容差不变。
两个独立运行槽、三次变输入/mask、L2 published平方和、旧结果/close读取及quota归零通过。
源码构建复用为static producer；安装路径搬迁并隐藏producer后再消费两个SDK。Windows的DLL
邻exe、PATH隔离、禁弹窗沿既定约定；所有重型步骤串行，parallel2。

消费端共两次有限纠正，原始首败均保留：①显式构造`Operation(Reduce/Softmax)`，MSVC拒绝
隐式转换；②七个Direct Invocation组成的L2链在重复执行时，后续槽仍绑定并保活上一轮的
上游输出，正确触发`result tensor is still retained`。多步Direct组合改用已有Recorder与两套
固定caller buffers，单算子和Graph继续Invocation；没有放松检查、改公式/门限或产品API。
这是不同公开入口的correctness消费，没有用不同输出责任做性能比较。产品/算法纠正仍0。

W1/Air/Mini各两个Vulkan公开头检查通过，共6门。完整
[交付记录与冻结consumer](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/delivery-progress.json)
绑定源码包、输入、消费者、首败、编译/链接/运行日志和SDK清单；归档逐成员SHA/CRC核对，
回传后核对归档身份。本阶段清理两平台的重复producer和已解包源码1,868,242,674逻辑字节，
保留SDK、安装消费者、当前模块缓存和共享依赖。

本组correctness/交付停止重复。112项selected的其余工程门仍未完成，聚合资格未签收；
C1其他Provider覆盖、完整成本优选/淘汰及CPU Q2 BLOCKED保持，不签C1完成、不进入C2。
实现统计仍为48,737行/197文件；本阶段仅增加外部消费者与证据，产品没有新增算子或代码。

### Y34 所选工程门：已完成

候选`ec240ac9`的[112项所选工程门](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/engineering/README.md)
全部资格化：96项普通工程检查、1项component-contract、15项默认三交付（45条消费）。
既有Y34三个C++ suite及两个公开头按源码、产物、163条未变编译命令与机器身份复用；
Provider的源码/MASM组合另行执行，未把旧版本门直接代签为本候选。8项未选理由保留。
本次没有产品、测试、公共工具改动或P1；W1/Air此前432例/1296次实际GPU三交付不重复。

归档核验后清理45个重复producer，共9,492,226,425逻辑字节。实现仍48,737行/197文件。
本批验收到此停止重复；C1其余Provider覆盖、完整成本比较/条件规则/存量淘汰与CPU Q2
BLOCKED仍待处理，C1及共同矩阵没有签收。C1全部适用验收完成后通知用户，在C2前等待提问。
