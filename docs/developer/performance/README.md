# 性能工程与测量仪器

- [性能目标与取舍](expectations.md)
- [产品性能定位](../../architecture/performance-strategy.md)
- [统一性能仪器协议 v1](instrument-v1.md)
- [性能机器合同](machines.md)
- [通用性能证据标准](benchmark-evidence-standard.md)
- [SIMD 操作级测量方法](simd-op-benchmark-methodology.md)

这里说明怎样测、怎样校准和怎样判读。某个 candidate 的实际数字和结果属于
[implementation/evidence/performance](../../implementation/evidence/performance/README.md)，原始
机器输出属于根级 [`evidence/`](../../../evidence/README.md)。

统一仪器的施工进度、error plumbing pilot 和尚未通过的验收门记录在
[模块收口与统一性能仪器 v1 实施记录](../../implementation/tasks/build/closure-system-v1.md)；该任务页
不替代这里的规范，也不自动产生模块性能结论。

统一仪器 v1 的 `exact_verdict` 是相对 practical band 的历史工程字段。当前全库竞争性总账还须从
逐层 ratio/CI 独立报告相对 `1.0` 的方向；工程 non-inferiority、方向和是否值得改产品不得合并。
