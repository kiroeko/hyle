# G08 设备端输出数值选择与可定位随机数

状态：**SCOPED，2026-10-16；尚未授权产品实现**。baseline 为
`85bcfa4eb`，G07已经整项接受，新G07–G10 goal完成1/4。原G01–G07不重跑，G09/G10未开始。
本卡只处理原[G08发现](../project/inference-single-machine-foundation-audit-2026-10.md)与
[消费形状计划](../project/inference-consumer-adaptation-plan-2026-10.md)的有限输出数值域。

## 真实任务、范围与 owner

主任务是 resident 词表 logits → GPU 数值选择 → checked completion → 只读回选中I32 ID及
需要的少量数值结果。当前本地Qwen3.5-2B `text_config.vocab_size=248320`，覆盖单请求与
多请求、请求重排/增减、至少65步和两个独立slots。模型文件只读，不运行完整模型/server。
正交任务是非模型批处理分布的离散随机选择与显式counter随机字；不以token身份设计数值API。
[输入证据](../../../../evidence/architecture/inference-g08-output-selection-intake/receipt.json)
绑定实际源码、三个固定peer、模型配置及各owner原terminal。

| 能力族 | 层级与本轮任务 | 初查 / 审计必须裁决 |
|---|---|---|
| S1 确定性单选 | 原子数值机制；复用TopK(count=1)的值+I32索引 | CPU原partial_sort为O(V)但分配V个Entry；CUDA按全输入radix-sort并分配全量keys/indices；Apple/Vulkan全argsort。审成熟arg-reduction，保原axis/stride/dtype、NaN最后、首index tie和源存储bits；不另造引擎greedy façade |
| S2 可定位随机字 | 原子数值机制；显式key/counter输入、设备生成 | 现util/SIMD xoshiro为可变顺序状态，不等于device counter schema。审固定成熟counter算法/许可/known-answer、key/counter容量、word寻址、重排/失败不暗改状态 |
| S3 有限categorical数值域 | 复合数值操作；原RNG+稳定变换+row selection | 审log-weights/weights自然输入、输出与必要selected scalar，非有限/零质量/空/overflow、精度和分布。三peer支持数值抽样，但SGLang特别要求端点精度；Metal无native F64，不准直接钳制F32 uniform后宣称等价 |
| S4 持续与组合 | BORROW原Graph/资源/Queue/native完成和拥有结果 | 真实device产出/消费，只传少量结果，两个slots、修改输入、失败/取消/close/旧结果、source/static/shared；统计、模型采样政策和调度不得下沉 |

当前拟定施工owner仅：`compute_ops`（通用数值schema/私有共用数学）、
`compute_provider_native`（CPU随机数值承载）、`compute_provider_ggml`（原CPU TopK单选）、
`compute_provider_cuda`（原CUDA collectives与设备数值）、`compute_route_apple`、
`compute_route_vulkan`（各原生承载）。它们由本卡新事实从CLOSED进入SCOPED。
Graph与CPU/CUDA默认route先只读；若实际接缝需改，先纳入owner，不借本卡改已关闭产品。

拟优先借现有typed custom operation机制新增独立数值头，保原Operation variant和Graph
schema机制；该方向仍须AUDITED时冻结。不是兼容要求：必要时可直接迁移现有消费者。
复用原allocator、状态/错误、stream/worker/fence与真正退休，不另建RNG全局owner或线程池。
随机输入属于caller，prepare不读取payload、不按地址缓存；不承诺各native数学逐bit同token。

不包括temperature/penalty/top-p策略、grammar/EOS、请求seed政策、tokenizer、完整sampler链、
MTP/多模态、完整Qwen CLI、多GPU/多机或新的全库缺口扫描。选中概率/数值结果若准入，须
冻结数学含义和费用；不能以全词表D2H或由引擎重复写设备数值来绕过S3。

## 审计与验证边界

固定peer仍为SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3，复用原归档，不刷新
其checkout。另核实际锁定native数学及成熟counter算法。审计只答S1–S4的有限决定，
至多一个检查点；4h作检查点而非强行到点结束。完成API/ownership/错误/并发/冷热成本、
SELF/BORROW/ROUTED/APPLICATION与有限FIX后逐owner AUDITED，才写产品。

AUDITED必须固定自然C++草图、真实target/filter、数学oracle/质量阈值、三交付及性能合同。
当前预声明适用检查：

- 数值：独立存储bits/首index oracle；NaN/±Inf/±0/非整tile/非末axis/stride/empty；
  counter known-answer/进位/高位/寻址与前缀稳定；categorical独立高精度reference、极端
  common offset/尾部/屏蔽和确定性固定seed分布检查，避免随机flaky门。
- 旅程：单/多请求、65步、重排与两slots、Graph设备producer→selection→少量结果、
  无隐式counter前进、非法数值失败不发布、OOM/accepted prefix/close/drain和旧输出。
- 实际CPU/CUDA/Apple/Vulkan source；受影响static/shared迁址与独立公开头，真实直接
  Graph及三个joint消费者；未变数学/资源风险按输入匹配复用，不制造全库重编验证。
- CUDA实际memcheck/racecheck、Apple实际Metal、Vulkan validation及适用三host facets；
  不用宿主san证明设备/driver/private pool，也不把未跑项写PASS。
- P0列出R/V复杂度、resident/temporary、host等待、copy/readback、状态与owner成本。
  新counter/categorical缺少原同counter合同peer时只签数值/P0，不伪造速度parity。
  S1若改实际单选执行，候选后按同TopK(1)值+索引/原完整完成与owner边界，对本baseline
  各受影响路线固定至多两形状（R1/R16、V248320）的P1；原统一5%/n15与A/A/双顺序/
  placement资格不变。真正合同与CHARGED/DEFERRED账冻结前不构造benchmark。

批量实现后集中验证、按真实失败定向修复。普通最多4jobs，CUDA pool1/split3/Ninja2；
Windows/WSL/Air重型任务串行，Mac仅实际确认身份的MacBook Air。时间沿用户授权自动追加，
保原失败和同树费用；不因换日志目录重置预算，不扩建公共统计/收口工具。
每轮按selector选择影响，运行docs/closure/状态投影与diff检查；候选后聚合七门/七问、
独立阶段红队和统一LOC。全部适用证据到位才关闭G08，保持G09/G10和CLI范围不变。
