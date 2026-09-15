# Compute/Graph：架构研究与历史决策索引

当前计划入口是[四路线基础重构](four-route-foundation-refactor.md)，依据为已完成的
[2026-09-15 六项架构复核](six-point-architecture-review.md)。长期需求只由
[Compute](../../../architecture/compute-design.md)、[Graph](../../../architecture/graph-design.md) 与
[四条原生路线](../../../architecture/compute-routes.md)持有；当前组件状态只见
[registry](../../module-status.json)。本页不是 closure card，也不授权恢复旧 goal 或产品施工。

旧页混合了研究事实、重复架构、已撤销的 CPU+GPU 联合计划和多代百分比/执行队列。本次保留必要
决策与证据入口，移除失效的操作指令和进度表；完整历史可在
[清理前版本](https://github.com/axiomates/hyle/blob/624f81df3d2672b0f222039d892f0d0b832b0c5a/docs/implementation/tasks/compute/compute-graph-route-unification-audit.md)
追溯。旧百分比不能作为新架构进度，历史 task/evidence 不自行成为当前任务。

## 研究结果入口

| 记录 | 实际范围 |
|---|---|
| [六项复核](six-point-architecture-review.md) | 路线隔离、实验覆盖、Apple 硬件/MLX 扩展、多后端 Provider、渐进路由与文档清理；本轮结果 |
| [四路线研究](four-route-architecture-decision.md) | 共同语义/独立执行与 Provider-owned 输出；MLX 内自定义 Metal、Vulkan 原生重复执行两个有限原型 |
| [初轮接入约束](foundation-architecture-admission.md) | MLX thread-local cache、managed execution、输入借用与错误终态；不是 Apple 产品交付 |
| [CUDA 压缩准入](cuda-compressed-provider-admission.md) | Original Marlin W4A16 的独立真机入口；旧 pin 不签当前 vLLM/SGLang 或 SOTA |
| [Q4 格式/载体调查](cuda-q4-foundation-carrier-preflight.md) | GGUF 与 W4A16、activation 量化和压缩权重常驻的差别 |
| [CPU executor](cpu-executor-foundation-probe.md) | 既有公开 Sched 组合在 Windows source/static/shared 成立；无需为此新建 bridge factor |

## 本轮架构分析：统一合同与独立执行

2026-09-15 的研究采用共同计算/资源/完成语义与四条独立原生执行路线；用户明确撤销模型 CPU+GPU
联合推理，不作为延期任务。host 控制/上传下载/同步和同 GPU 多 Provider 组合仍保留。旧实现只是
待裁决资产，不能锁定 compiler、执行器、类名或 CMake 拆分，也无旧接口兼容要求。

研究使用的版本：

| 参考 | commit / 版本 |
|---|---|
| llama.cpp（含 ggml） | `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` |
| standalone ggml | `456172ec733a135778adcd32d00e576a58232e45` |
| vLLM | `51918e252af4dfff381001f42b7ebe6a0454501b` |
| SGLang | `06992c92ad4494fe9f9636e75be3e35040512447` |
| ds4 | `9139e2ae58a41503968a500f36f75895c1ba63fc` |
| MLX | v0.32.2，`1f8e74e3f12f31365464a6867c6579f0e9b29d85` |

源码证明 vendor handle/stream/workspace、MLX thread-local compile cache、Vulkan command/descriptor
pool、finite CUDA graph variants 各有原生合同；不能用一种通用热循环抹平。具体入口和采用理由归
[四路线设计](../../../architecture/compute-routes.md)，本索引不再复制。未做新依赖锁升级。

## 整体架构复核与合同修正

这是早期 A1–A6 的历史 trigger 摘要，供现有 closure records 追溯；不是命令再次重开模块。

| ID | 当时发现的问题 / 形成的语义需求 |
|---|---|
| A1 | MLX compile 命中仍实例化 arrays，不能承诺所有 Provider 零分配；runtime ownership 与 prepared/managed 成本分开 |
| A2 | flatten `B*T`、concat、head split 等需要有限派生 shape，不能逼消费者维护重复中间维度 |
| A3 | custom schema identity 不等于实际执行扩展；三交付需要真实 owning factory/代码与资源寿命 |
| A4 | 一个全局 effect 链串行化独立状态；改以资源 read/write/alias 描述真实冲突 |
| A5 | finite variants 需要共享兼容 immutable constants，并计入全部常驻/并发预算，不只单变体局部值 |
| A6 | 搜索预算耗尽但已有完整合法解，不应丢掉该解；受限成功与无解失败区别处理 |

这些是语义需求，不批准当时选用的全部类型/实现。后来又确认 Provider 生成输出须有真实 owning
result，不能所有场合强制 caller output buffer；独立 runtime 也应能直接 Compute 执行。各修正与旧
代码的取舍见本次复核，不从某一历史卡通过推出整个基础架构完成。

## 2026-09-15 原始需求重新对齐与进度基线

早期基础进度曾把 GPU runtime、codec、模拟 delegated Provider 和真实 GPU 算子/region 接入混算；
后续发现 Vulkan、公开 CPU executor、真实 profile 与跨 runtime 终态并未全部覆盖。旧估计被明确撤销，
不将它转成新架构百分比，也不由文档重写增加产品完成分数。当前研究不修改 registry。

### 更贴近消费的有限纵切

llama.cpp 的 FFN 是 gate/up→activation→down；ggml MMVQ/MMQ 与 ds4 存在直接消费 compressed
weights 的 kernel；vLLM/SGLang 在加载后 repack W4A16。由此撤销把“全量 Q4 展开 F32 常驻再算”
当压缩路线主例的计划，活动示例已退役；其旧互操作/同步事实仍保留，不能倒写成从未测试过。

来源与格式分析见 [Q4 调查](cuda-q4-foundation-carrier-preflight.md)。kernel 内 tile 解码、activation
临时量化、保持压缩的冷 repack 与整份浮点权重常驻不同；数值和内存成本都需显式说明。

### 决策与纵切的对应关系

| 纵切依据 | 可回答的问题 / 不能外推的资格 |
|---|---|
| CPU Native/oneDNN + 持久 executor | 真实任务接纳、嵌套与 owner；Windows 三交付不签其它 OS 或并行 P1 |
| CUDA compressed MatMul/FFN | 实际压缩 kernel、格式/精度、冷 packing、地址/运行状态；原始 Marlin 单 kernel 不签正式多算子 carrier/capture |
| MLX GPU FFN + 同 runtime Metal 扩展 | managed arrays、线程、结果寿命与内部组合；不签外部 MTLBuffer/Queue 桥或 M5 NAX |
| Vulkan 两个依赖 pass、重复输入 | 原生录制复用和 shader memory barrier；不签多 queue、device-loss、其它 GPU/MoltenVK 或推理吞吐 |

这些旅程检验共同语义是否可表达，但不强制共用旧 Graph/Executable/Invocation 对象。具体 shape、
版本、oracle 和运行结果在各 evidence/card，不在本索引再造执行队列。

### 基础计划覆盖审计与有限补项

历史覆盖审计发现 F1–F6：Vulkan 被后移；CPU probe 的 `parallel_for` 实际串行；profile matcher
尚未走实测回灌；GPU/runtime 组合的失败链不足；Obs 事件消费不足；各 backend 的 build/三交付不能
留到完整模型之后。它们不全部构成已修复事实，也不在这里继续排队。

其中 CPU 的公开组合已有 [Windows 检查点](cpu-executor-foundation-probe.md#windows-检查点)，
采用已有 Sched，不新增 `compute_sched`。其余旧事实进入本轮复核的“正式实现/资格尚欠”，不因两次
架构原型成功而签完；后续只有绑定到具体 owner 的活动卡才执行产品修改。

## 后续工作的边界

六项研究已报告，用户已批准把七包写成执行计划并绑定首包；实际当前入口只见上面的计划。本页不恢复
旧队列、CPU+GPU 联合推理或全算子填充。新问题先归属本地 adapter/执行器；只有反例推翻共同语义
或成本责任时重开共同设计。raw evidence 与 terminal records 保留原历史事实。
