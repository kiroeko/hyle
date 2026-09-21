# M2-S：Scatter 加法与 Select 广播的公共语义

## Closure card

- component / owner：`compute_ops`；`/root`。Apple B1是首个成熟消费者，其实现仍归原B1卡；
  CPU/CUDA/Vulkan的旧数学保持冻结，不因本卡重开或重测旧自研性能。
- baseline：`86a6a37b`，Windows canonical，工作树干净。初始SCOPED基线；当前已按下述有限FIX进入AUDITED。
- reopen trigger：F02.2/F13.2真实模型需求是重复目标求和；公共Scatter add却被描述为逐次storage
  舍入的行主序fold，`operation_traits`还把其标为`kExact`。已记录的原版MLX反例和成熟MoE组合
  证明接入受到这项额外承诺约束；需要按真实需求裁决公共语义，不能反过来要求重写成熟归约。
- 主纵切：MoE专家输出的weighted combine、重复目标及共享专家和；正交纵切：无模型标签的
  indexed floating reduction，正常/empty/checked failure、旧结果和两次调用。
- 来源：M0/C0已登记且clean的llama.cpp `4c9233c0`与SGLang `06992c92`；原版MLX0.32.2
  的GPU Scatter与现有资格收据。PyTorch官方scatter_add合同作为独立成熟接口佐证。
- scope：需求→浮点求和档位、assignment与addition的precision traits、普通公开用法和四路线
  用户说明；必要公共schema测试及首个成熟消费者。无新增enum/字段/ABI或新runtime。
- non-goals：弱化索引/alias/失败/资源检查；修改固定模型精度档或端到端误差门；删需求/覆盖格；
  研究/扩张/比较/删除冻结自研；为了通过旧反例改性能仪器或重新计时。
- 预算：共享公共语义源码/设计总30分钟，前置定位计8分钟，余22分钟。它只覆盖本次新发现的
  shared contract问题；各路线原B1源码额度保持，不能借它续杯。每具名根因最多两次纠正。
- 性能：本公共变更P0，只有准备期traits与说明，不增常态资源/调度；原版计算接入P0/P1仍由
  各B1卡承担。没有资格就不签计算或性能PASS；M5前停止点不变。
- stop：有限语义/trait/文档与适用schema门完成后收口；成熟默认仍必须逐路线实际验证，
  本卡不自动解除旧库失败，不把公共语义修改当成256格覆盖。

## 待裁决的有限范围

1. `assign`仍为storage-exact，重复索引最后逻辑更新胜出；`add`应为浮点归约，所有updates必须
   参与。明确是否允许重结合、不同累加顺序和不低于存储精度的中间值；不凭某个输出点值决定。
2. 模型若需要F32中间值，应在固定模型计算图使用已有Cast/F32端口；不将F16/BF16最终storage
   自动解释为每步必须舍入，也不允许静默降低模型明确要求的精度。
3. `kExact`只用于assign，add使用既有算术precision档。无需新增运行时开关或保留双API。
4. 旧reference逐次fold只作其实现特征，不作为新默认资格或oracle。迁移的成熟consumer用
   独立高精度归约、正常范围误差门和具名病态序列的合法浮点界；禁止用宽容差掩盖丢更新。

## 预声明验证

- 先归档原版来源和现有数值反例；不执行冻结实现对照，不重跑性能。
- 公共schema检查assign/add traits、描述符与参数不变；Windows/WSL/Air/Mini的compute_ops
  适用头和普通测试。实现仅有元数据分支，没有新并发/设备内核，旧host插桩身份适用时复用。
- 公共对象/导出/依赖不变；source/static/shared实际新计算消费沿受影响B1同一合同执行，
  shared card闭包引用明确收据，未完成交付前不写CLOSED。selector的实际selected门保留。
- Apple普通Direct/Graph成熟Scatter add由原B1定界、资格、接线；不要求旧冻结数学重写。
- 每次docs/closure；registry变更明确previous-ref，文档/状态不构建、不启动P1。

## AUDITED：模型加法合同与有限FIX

`a3ce441c`的独立SCOPED之后完成原版来源核对；三份clean/pinned源的身份和具名证据见
[source audit](../../../../evidence/correctness/compute-llm-moe-m2-scatter-semantics/source-audit.json)。
SGLang的原版FusedMoE以einsum或sum聚合专家维，另一路先用top-k权重dtype相乘、sum后cast；
llama.cpp按专家视图组合add。它们没有共同提供“低精度每步舍入、跨库逐位一致”的必需语义。
[PyTorch官方Scatter add](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.scatter_add_.html)
也明确CUDA调用可能不确定；这里只用于数值档佐证，不混同其shape/错误/owner合同。

裁决是恢复模型实际需要的浮点归约档，不为某个性能点值或库名放宽模型精度。F02.2/F13.2与
C0所有明确的FP32中间值、量化规则、最终误差门继续约束路由。浮点add必须把base和每条更新
纳入归约，允许重结合与不低于存储精度的中间值；不承诺行主序fold、每项立即舍入或跨运行位
一致。需要FP32中间值的模型图使用现有Cast/F32端口。没有新的低精度近似、丢更新或省检查授权。

| FIX | 允许改动 | 约束 |
|---|---|---|
| S1 | `compute_ops`的Scatter traits区分assign的Exact与add的既有AccumulateInOutputType；公开头说明加法精度/顺序 | 不增字段/enum/ABI；`accumulation_type=Unknown`表示未另指定更宽类型，不限制合格实现内部提升；描述符、shape、alias/effect和bounds不变 |
| S2 | Compute/Apple/Vulkan指南、F02.2/F13.2的数值说明与当前接入卡引用同一裁决 | 旧顺序/舍入只作冻结路径的实现特征，历史失败不回写PASS，256分母和模型误差门不变；自研不修改/比较 |
| S3 | schema traits回归与首个原版MLX消费者按其B1卡接入并验证 | 正常范围独立高精度oracle、精确可表示的丢更新检测、非结合反例的合法数值结果；所有checked/状态/生命周期门保留 |

公共改动是准备期元数据和明确数值合同，P0无新增常态工作/资源；未引入确定性选项或第二条
公共API。原版MLX旧有序资格仍FAIL；新加法接法必须重新资格，不能直接改旧证据标签。
本共享源码/设计累计14/30分钟，余16分钟；Apple/CPU/CUDA等原B1额度没有重置。

## S1/S2 开发检查点

已实现公共traits区分：assign保留Exact，add使用既有AccumulateInOutputType；未新增字段、
enum、ABI或执行期工作。公共头、三份路线/Compute指南、F02.2/F13.2及当前CPU/Apple卡同步
上述数值合同，保留历史有序资格失败与固定模型精度。此处没有接入新的Scatter计算。

Windows复用现有Release/shared/WERROR开发构建，parallel2；两个CTest门通过，包括公开头
检查和42个Compute Ops用例。新增用例同时核对precision、arity、alias、effect及输出数量；
全部DLL邻置EXE，SetErrorMode 0x8003，无UAC/电源策略更改。源码字节、构建产物hash、日志
和selector计划见[开发收据](../../../../evidence/correctness/compute-llm-moe-m2-scatter-semantics/windows-development.json)。

WSL/Air/Mini的schema门、selected直接消费者/三交付，以及原版MLX add资格/普通消费者仍待
完成。本轮未采样P1，不签完整覆盖或CLOSED；共享状态保持AUDITED、聚合验收INCONCLUSIVE。
共享源码/设计累计16/30分钟，余14分钟，各路线原额度保持。统一脚本统计全Compute+Graph为
**56,695实现行 / 248文件**，不含测试/vendor；未新建副本或大型构建目录，复用活动开发构建。

## 首个成熟消费者检查点

Apple B1的`33a6555b`已将普通Scatter add接到原版MLX，独立求和、特殊值、checked失败与
Graph状态/旧结果开发用例通过；来源和首轮测试构造失败及修正详见
[Apple开发收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/scatter-add-development.json)。
其中Air的compute_ops公开头及42项schema测试通过，补足本卡该环境开发门。Windows已有
同组42项通过；WSL/Mini、其余selected消费者与实际新域三交付闭包仍待完成，不签CLOSED。
共享源码/设计仍16/30分钟，Apple原卡计78/120；旧有序FAIL与自研冻结保持。

## S4 AUDITED 增量：Select 的条件参与输出形状

2026-09-21，基线`45c3ee76`，oneDNN B1接入的具名反例：Bool条件为rank8的2×3逻辑张量，
两个分支都是UInt64标量，普通typed prepare在到达Provider之前返回“tensor is not
broadcast-compatible with target shape”。现有公共指南承诺Select的NumPy式广播；模型mask
选择两个常量及普通无模型标签张量都需要三输入共同确定输出形状。

[NumPy where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)明确condition、
x、y共同广播；公共schema当前只合并两个branch，再要求condition适配该形状，是漏合并
condition的准备期缺陷，和oneDNN数学、运行期调度、旧Q2无关。

本轮将当前AUDITED范围定向增加一项S4：只在`compute_ops/operation.cc`的Select输出推导
中用已有`broadcast_shape`合并condition与branch shape，保持Bool条件、同branch dtype、
rank上限、非法广播拒绝和所有执行期语义。增加标量branch/rank8/空shape/非法形状的schema
回归及本次成熟oneDNN普通Direct/Graph消费；无新增API/ABI、常态字段、数学或通用工具。

共享源码/语义核对另计4分钟，累计20/30分钟，余10分钟；CPU/Apple的B1额度不重置。P0只
增加一次有界rank≤8的准备期形状合并，无执行期成本。本卡原四环境schema及受影响消费者/
三交付门保持，先做Windows开发门，未运行项不代签；所有冻结算子继续冻结。
