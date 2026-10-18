# D5 CPU：完整片段 P1 工程验收

**测量资格 PASS；两个完整片段均为 PARITY，5% 非劣门 PASS。**
候选 `1bf0b5566a5606540779860f98838b12df4196d0`，产品 SDK 仍为
`9273bc71017a1febb69db240b6ea8970b88dd77c`。普通默认 CPU Context、单 native 线程，
W1-WIN/MSVC 19.50.35728，调用者固定 P 核并保留既定 native-runtime-pair CPU 集。
比较相同成熟实现的完整 Graph/Direct 片段；不是裸算子、整模型吞吐或数学 SOTA。

| 片段 | 四种顺序的 Graph 中位耗时差范围 | 最不利顺序的区间允许退化 | 裁决 |
| --- | --- | --- | --- |
| Qwen3.5-2B 官方 BF16 FFN，M1/H2048/N6144 | -0.22% ～ +0.31% | 1.23% | PARITY / PASS |
| DeepSeek Flash UD-Q4_K_XL hash-MoE，H4096/N2048/top6 | -0.39% ～ +0.19% | 1.52% | PARITY / PASS |

以上耗时差由公共仪器的 Graph/Direct 吞吐比倒数换算；区间不是新增统计方法，精确值见
[原始 summary](run/summary.json)。八个工作负载 A/A 进程先通过；正式执行与分配双顺序共
60 个进程，每层 n=15，另有30个公共控制进程。四项控制全部通过，未触发 n=31。
输入、输出、变化输入、完整完成和所有权释放与已验消费者相同；未重建 SDK 或重跑三交付。

`hyle-perf.py verify` 对原 manifest 及全部被引用文件严格重放通过：`QUALIFIED`、
`paths_verified=true`、当前证据政策适用。公共统计核心和产品实现均未修改。
[原 manifest](run/manifest.json)、[控制项](run/controls.json)和
[工作负载 A/A](run/d5-workload-aa.json)保留原字节；[归档索引](index.json)绑定原路径、
原 SHA-256 和归档 SHA-256。原日志以 gzip 保留，合格构建与原始收据仍留在验证目录供严格重放。

初次 preflight 因 reference 的稀疏 checkout 漏了测试头而停止，A/A 和产品采样均为零；
原日志保留在 `initial-reference-preflight`。补齐同一不可变 reference 的测试目录后才执行本批，
没有改变消费者、输入、精度门或性能门。

本结果只签本合同两个完整 CPU 片段；官方 FP8/FP4 CPU 冷拒绝、其他路线 A6 和 D6 仍按当前卡处理。
