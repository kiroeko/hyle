# 性能比较

- [目标与取舍](expectations.md)：正确性、资源成本和结论范围。
- [公平比较与报告](benchmark-evidence-standard.md)：相同工作量、计时边界、结果解释与原始数据。
- [测量与统计](instrument-v1.md)：A/A、分辨率、配对样本、顺序与置信区间。
- [验证环境](machines.md)：现有硬件、WSL、亲和性与 macOS QoS 限制。
- [SIMD 专项方法](simd-op-benchmark-methodology.md)：极短循环的 codegen、放置和锚点检查。

直接运行相关 benchmark 和分析工具；不要求机器登记、证书、固定样本上限或治理审批。
报告放在最相关的实验目录，原始输出保留于根级 [evidence](../../../evidence/README.md)，避免多处复制结论。
