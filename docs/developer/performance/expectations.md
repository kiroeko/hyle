# HYLE 性能目标与取舍

HYLE 的目标是在真实用户合同下提供可信的基础加速，不是让每个 benchmark 行都变绿。性能结论
必须限定 capability、contract、workload、hardware、delivery 和 candidate；不存在永久的“全库
SOTA”证书。

## 优先顺序

1. correctness、精度、ownership、failure 与资源边界；
2. 渐进复杂度、进展保证和最坏情况；
3. 真实消费者的 latency、throughput、tail、memory、copy、energy 等 material 结果；
4. allocation、syscall、contention、cache/vector utilization 和 codegen 的机制差异；
5. 只有能传播到上层时才处理的微小常数差。

不允许用更弱语义换取“同合同”胜利，也不允许用一个资源维度掩盖另一个：吞吐提升不能自动
抵偿峰值内存、碎片、tail、错误率、公平性或可移植性退化。

## 参照与结论

`std` 是常识基线，不总是强 peer。分配器、flat hash、SIMD、并发队列、调度等领域要选择用户
真实会替换 HYLE 的成熟实现。参考库帮助发现能力、类型与算法；其声誉和源码相似度不证明 HYLE
性能。

同合同结果使用 `WIN/PARITY/LOSE/INCONCLUSIVE/NOT_COMPARABLE`。合同无法对齐时分别报告绝对
成本和必要语义税，不把功能更少的一方宣称为 SOTA。

## 时间、空间与部署

- 时间和空间分别守门；需要 Pareto 取舍时预先声明消费者预算。
- native/exact ISA、portable dispatch、static/shared、LTO/非 LTO 是不同交付合同，不能混表。
- 没有真 AVX-512、RVV、NUMA 或 GPU 硬件时只能证明可编译/正确 fallback，不能推导时序。
- 混合大小核、Apple QoS、DVFS 和 NUMA placement 属测量合同，不是事后解释。

## 防止 benchmark 扭曲

- 需求和 API 在看跑分前由真实消费者与成熟形状决定。
- candidate、peer、输入、计费边界和 practical margin 在正式运行前冻结。
- microbench 只归因；产品变化需要机制证据与 material 消费影响两把钥匙。
- A/A、known-delta、timer floor、order/placement 未通过时结果统一是 `DIAGNOSTIC`。
- 不为稳定的亚纳秒差异增加常态字段、padding、间接或非自然入口。

完整仪器门见 [benchmark evidence standard](benchmark-evidence-standard.md)。模块达到签署条件后，
只因真实缺口、correctness、依赖/平台变化或强 peer 的 material 证据重新开启。
