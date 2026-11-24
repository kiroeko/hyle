# Resident SDPA 性能适配器

这里提供有限 attention 消费者和独立参考。CPU 入口见 [cpu_p1](cpu_p1/README.md)。
各后端的 encoded/page/ragged 子目录说明各自的输入、参考实现和验收范围。

保存原始输出，用[性能比较方法](../../docs/developer/performance/README.md)检查相同工作量、A/A、
顺序、分辨率与完整样本。历史 contract 描述当时的实验，重测时记录本次配置与分析方法。

这些数据衡量所列形状的 attention 片段。完整引擎 token/s 见[推理性能](../../docs/developer/performance/decode.md)。
