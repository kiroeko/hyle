# HYLE 性能目标与取舍

HYLE 的目标是在真实用户合同下提供可信的基础加速，不是让每个 benchmark 行都变绿。性能结论
必须限定 capability、contract、workload、hardware、delivery 和 candidate；不存在永久的“全库
SOTA”证书。

本页规定产品取舍；统一结果语义、统计和运行预算由
[统一性能仪器协议](instrument-v1.md)定义，发布级附加要求见
[性能证据标准](benchmark-evidence-standard.md)。

## 优先顺序

1. correctness、精度、ownership、failure 与资源边界；
2. 渐进复杂度、进展保证和最坏情况；
3. 真实消费者的 latency、throughput、tail、memory、copy、energy 等 material 结果；
4. allocation、syscall、contention、cache/vector utilization 和 codegen 的机制差异；
5. 只有能传播到上层时才处理的微小常数差。

不允许用更弱语义换取“同合同”胜利，也不允许用一个资源维度掩盖另一个：吞吐提升不能自动
抵偿峰值内存、碎片、tail、错误率、公平性或可移植性退化。

## 模块性能分级

模块在 `SCOPED` 时冻结性能级别，不能看完结果再改级：

| 级别 | 正常含义 | 工程收口门 |
|---|---|---|
| `P0` | 未被 profile 证明为热点的冷路径、编译/布局 primitive | 复杂度、allocation/lock/syscall/metadata 和适用 codegen/size 结构检查 |
| `P1` | 高频 runtime primitive、容器、allocator adapter、queue/timer 操作 | 锁定性能机上的正常 workload/replay 工程门；有同合同 peer 时增加 non-inferiority；默认 material margin `M=5%` |
| `P2` | profile 已证明影响端到端 KPI 的热点 | 真实 consumer/trace 的 latency/throughput/resource 门及机制归因；更小阈值必须由贡献预算证明 |

P1/P2 的工程性能 `PASS` 是模块进入 `CLOSED` 的硬门；缺资格或功效不足应写 `BLOCKED`，不能写成
平局。Formal 发布证书是独立状态轴：模块 `CLOSED` 不等于 `CERTIFIED`，Formal `PENDING` 本身也不
阻塞已经满足的工程门。完整状态规则见
[模块重构与收口协议](../review/module-closure-methodology.md)。

## 三个独立结果与统一方向

每个 workload 分别记录：

- comparability：`COMPARABLE | NOT_COMPARABLE`；
- qualification：`QUALIFIED | DIAGNOSTIC`；
- exact performance：`WIN | PARITY | LOSE | INCONCLUSIVE`。

单边 non-inferiority 另记 `PASS | FAIL | UNRESOLVED | N/A`。不能把 `NOT_COMPARABLE`、仪器
`DIAGNOSTIC` 或统计 `INCONCLUSIVE` 互相冒充。

统一 `advantage_ratio > 1` 永远有利于 HYLE：higher-is-better 用 `candidate/peer`，
lower-is-better 用 `peer/candidate`。给定预声明 margin `m`，双边等价带为
`[1/(1+m), 1+m]`。旧 runner 若输出反方向 ratio，必须在 measurement adapter 归一化后才进入
当前 summary。

## 参照与结论

`std` 是常识基线，不总是强 peer。分配器、flat hash、SIMD、并发队列、调度等领域要选择用户
真实会替换 HYLE 的成熟实现。参考库帮助发现能力、类型与算法；其声誉和源码相似度不证明 HYLE
性能。

合同无法对齐时分别报告绝对成本和必要语义税，不把功能更少的一方宣称为 SOTA。P1 默认 5% 是
普通工程行动阈值，不是所有产品、指标或发布声明的自然常数；P2 可以用消费者预算预声明不同
阈值。
完全没有同合同 peer 时，P1 必须预先声明真实 consumer 的 latency/throughput/resource 预算；通过
该预算可以签工程 `PASS`，但 comparability 保持 `NOT_COMPARABLE`，不得产生 peer/SOTA 声明。

## 时间、空间与部署

- 时间和空间分别守门；需要 Pareto 取舍时预先声明消费者预算。
- native/exact ISA、portable dispatch、static/shared、LTO/非 LTO 是不同交付合同，不能混表。
- 没有真 AVX-512、RVV、NUMA 或 GPU 硬件时只能证明可编译/正确 fallback，不能推导时序。
- 混合大小核、Apple QoS、DVFS 和 NUMA placement 属测量合同，不是事后解释。

## 防止 benchmark 扭曲

- 需求和 API 在看跑分前由真实消费者与成熟形状决定。
- candidate、peer、输入、计费边界、material margin 和停止条件在正式运行前冻结。
- microbench 只归因；产品变化需要机制证据与 material 消费影响两把钥匙。
- A/A、known-delta、timer floor 和适用的 order/placement 未通过时结果只能是 `DIAGNOSTIC`。
- 不为稳定的亚纳秒差异增加常态字段、padding、间接或非自然入口。
- P1 异常按统一协议的固定调查预算停止；模块不得现场发明新统计方法或第五种 benchmark 姿势。

模块达到签署条件后，只因真实缺口、correctness、依赖/平台变化或强 peer 的 material 证据重新
开启。工具链或 Formal 证书过期会使相应证据失效，不会自动证明产品形状错误。
