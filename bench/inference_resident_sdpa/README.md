# G01 resident SDPA 性能适配器

本目录只适配不可变 G01 产品 candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f` 的有限
attention 消费，不改变公共性能 runner、统计语义、产品 API 或 runtime。

候选是普通 Vulkan Graph；对照从 clean ggml
`456172ec733a135778adcd32d00e576a58232e45` 独立编译原 full-F32 flash、mask、conversion、
split-K 和 mask-opt shaders，按原 native recipe 的 strides/tuning 组装。两侧共用原
Program/Invocation 的 admission、checked completion、ResultLease/Tensor 归还；对照不调用
候选 typed lowering/Graph compile。对照的私有生命周期适配要求 static source build。

当前预检仅验证 F32 BSHD GQA8/2：Q1/KV1024/D256 causal decode、Q128/KV1024/D256 causal
prefill、Q17/KV257/D64 noncausal 留出。容量多13个 token，padding=128，独立 double oracle
沿原 `6e-6*abs(expected)+2e-6` 容差。F16 helper 分支尚未资格，不据此扩大支持声明。
当前对照 tuning 只覆盖实际 NVIDIA 与 Apple 主机；其他 vendor 明确拒绝。

冷 setup 包含 pipeline/slot/input allocation 与首次 native scratch 物化；before/after/peak
单独报告，稳定重放验证后才可采样。测量包括 ready-input bind、实际 submit/wait、全部
warm mask/conversion/split-K 与输出 owner 归还，不包括 host download 或独立 oracle。
VK memory quota 不覆盖 driver/pipeline 的私有内存，两侧总成本不能只看 quota 字节。

`HYLE_BENCH_VERIFY_ONLY=1` 只跑数学和稳定重放；`HYLE_BENCH_AA_ROUTE=native|selected`
使两侧同路，保留 execution/allocation-order 参数。当前原始工程预检见
[证据](../../evidence/correctness/inference-g01-p1-adapter/README.md)。
[正式合同](contract.json)绑定三种F32形状、八个显式strata、n=15和现有v1的5%门，
以更严格门验证活动卡原10%非劣预算；[A/A检查](validate_aa.py)复用原机器/解析器与
5%factor median、10%gross instability界线。正式A/A与统一仪器timing尚未完成，
这里不能手跑点值代签P1。CPU resident P1另验，不由本Vulkan合同代签。

CPU独立对照的[源码入口](cpu-CMakeLists.txt)沿原typed custom kernel/PreparedOperation与
MetadataArena/CpuExecution承载原ggml BF16 flash；BHSD GQA8/2 D256、Q1/128、KV1024
容量1037、源只读/两个slots重放已过[数学预检](../../evidence/correctness/inference-g01-cpu-p1-adapter/README.md)。
私有pool上界仍UNKNOWN，CPU正式P1未跑；不将这个对照Provider变成产品SPI。
