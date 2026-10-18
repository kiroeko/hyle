# Apple parity：成熟 Provider 完整成本裁决

产品候选 `326a40ee9b2f01a760d8bbdc66974685ff8cfdb0`；性能快照
`7365f0fb95d8cb83435252c5432fa466becd2d98` 的 include/src/cmake/CMakeLists
与候选无 diff。当前 Air Mac16,12 / M4 applegpu_g16g、macOS27.0.1/26A434、
AppleClang21、SDK27/deployment14.4、交流电/低功耗关闭；原 MLX1f8e74e3 与
GGML456172ec，compiled cast patches逐文件匹配，actual owner QoS21、caller
user-interactive。不冒称 P 核 hard pin，不外推未知硬件 SOTA。

[固定合同](../../../bench/contracts/compute-apple-mature-overlap-p1-v1.json)比较原
GGML Metal 与原 MLX，双方独立普通 Context、相同 ready contiguous dynamic 输入/
权重银行及两复用 slot，完整独立输出 oracle。bind/submit、全部转换/布局/checked IDs、
native terminal与结果释放同收费；cold setup另列。MLX allocator计数不是GGML/driver
总私有内存，不外推 immutable Graph 常量融合或整模型吞吐。

[正式 run](overlap-p1-format-7365f0fb9.log)资格 PASS、工程门 PASS；
[manifest](qualified/manifest.json)、[summary](qualified/summary.json)、
[controls](qualified/controls.json)、[observations](qualified/observations.jsonl)、
[A/A](qualified/apple-overlap-aa.json)、[raw index](qualified/raw-log-index.json)
保存16同路A/A、120产品（八strata×n15）、30controls。四域合并严格WIN、32 NI门PASS，
5% margin；停止n15，没有性能优化或追加n31。

| 完整动态域 | GGML Metal / MLX 分层吞吐中位比 | 选择 |
|---|---:|---|
| BF16 router M1/N512/K8192，F32 output | 2.1995–2.2091 | 保留GGML Metal |
| 同域M16 | 1.2465–1.2564 | 保留GGML Metal |
| BF16 grouped T5/S8/E8/N1024/K4096，F32 activation/output | 4.2981–4.3266 | 保留GGML Metal |
| 同域F32 weights | 1.8420–1.8472 | 保留GGML Metal |

既有 `fixed_ggml_model_operation()` 与Graph完整native-region冷绑定就是本批赢家，
不改产品、不重跑三交付。实测优选限上述硬件/OS-driver/toolchain/原Provider版本、
物理dtype/shape/layout/dynamic合同；其它组合继续同一合法成熟ENGINEERING_DEFAULT，
不宣称WIN。模型标签不能替代端口格式，replay无搜索/测时/重选。

MLX BF16候选热期另需16MiB router或128MiB grouped整权重widen数组；GGML消费原BF16。
源码成本支持部分差距，不将每微秒都归因于它；F32 grouped亦有原程序优势。
这里没有HYLE kernel竞争；独有被授权FP4精度桥接另有完整正确性资格，SIMD不动。

[verify回读](overlap-p1-format-7365f0fb9-verify.log)为QUALIFIED/EXACT/CURRENT且paths
verified。实际工程门PASS来自run/summary，不由verify单独签发。JSON原绝对验证路径保持，
`qualified/`归档完整raw副本，不静默rebasing。

[首次遥测拒绝](first-telemetry-failure.log)及`first-rejected/`保留16同路A/A PASS与首个
product process字段拒绝。仅三个cold字段名/A/A断言/provenance hash纠正；公共仪器和
产品数学未改。该次没有合格性能verdict，纠正artifact另获有限A/A及唯一完整n15批次，
没有重复模型矩阵/SDK。重复压缩传输包归档后清理，真实raw保持。
