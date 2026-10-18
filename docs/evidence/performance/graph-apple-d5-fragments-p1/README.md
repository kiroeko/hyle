# D5 Apple：完整片段 P1 工程验收

**测量资格 PASS；两个完整片段均为 PARITY，5% 非劣门 PASS。**
候选 `1bf0b5566a5606540779860f98838b12df4196d0`，复用已验静态 SDK；产品仍为
`9273bc71017a1febb69db240b6ea8970b88dd77c`。机器为 MacBook Air Mac16,12 / M4，
AppleClang 21、macOS 27.0、MLX 0.32.2、ggml `456172ec`，显式 user-interactive QoS。

| 片段 | 四种顺序的 Graph 中位耗时差范围 | 裁决 |
| --- | --- | --- |
| Qwen3.5-2B 官方 BF16 FFN，M1/H2048/N6144 | -0.65% ～ -0.19% | PARITY / PASS |
| DeepSeek Flash UD-Q4_K_XL hash-MoE，H4096/N2048/top6 | -3.64% ～ -3.50% | PARITY / PASS |

比较相同成熟实现的完整阻塞 Graph/Direct 调用。Direct FFN 通过公开 `Context.compile` 保持
完整 MLX 区域，MoE 合并连续 MLX 节点，只在原生 ggml Metal 区域边界等待；没有人为逐叶等待。
原输出容差和变化输入保持不变。以上中位差由公共仪器吞吐比倒数换算，不能宣称数学 SOTA。
精确区间与逐顺序结果见[原始 summary](run/summary.json)。

八个工作负载 A/A 进程通过后，执行与分配双顺序共60个正式进程，每层 n=15，另有30个公共
控制进程。四项控制全部通过；没有扩样，没有修改产品、数学、精度门或公共统计核心。
`hyle-perf.py verify` 严格重放全部引用文件通过，见[原始输出](run/strict-replay.json)。

[原 manifest](run/manifest.json)、[控制项](run/controls.json)、[工作负载 A/A](run/d5-workload-aa.json)
保留原字节；[归档索引](index.json)绑定原文件与归档 SHA-256，日志以 gzip 保存。合格构建和
原始收据保留在 Mac 验证目录，Windows 此归档为产品证据真相；验证副本不反向覆盖源码。

本结果仅签两个完整片段；DeepSeek 官方 FP4 的 Apple 冷拒绝仍保留。CUDA/Vulkan A6 与 D6
尚未由本结果完成，模型加载、整模型吞吐和其它硬件/软件版本不在本结论内。
