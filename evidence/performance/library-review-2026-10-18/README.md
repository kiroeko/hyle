# 八之五：全模块性能排查

基线 `90e10686e`。先核对工作量/精度/生命周期，再核仪器与噪声，最后判断实现。
本次调研覆盖中央构建表的64个 factor（含退役 Marlin 占位），不是64个模块都已重新计时。
旧结果只有在当前实现仍对应、输入与完成边界一致时才作为线索；旧 PASS、5%容忍带和状态标签不决定结论。

## 当前源码明确的问题

| 项目 | 证据与影响 | 本轮处理 |
|---|---|---|
| CPU 普通路线漏注册 ggml | [owner CMake](../../../src/compute_provider_ggml/CMakeLists.txt) 用 `NO_DEFINE`，而 [Direct](../../../src/compute_route_cpu/cpu_context.cc) / [Graph](../../../src/graph_route_cpu/cpu_context.cc) 以该 presence 宏编译注册。实际库虽链接，候选却消失，旧 CPU 排名不能直接继承 | 已恢复 target 的宏传播；实际 owner 声明的配置回归通过。未测修复后的 CPU 模型倍率 |
| StateStore 重复扫描同池 roots | [stats_locked](../../../src/compute_state/state_versions.cc) 对已命中 root 仍扫描所有更早块；`replace` / `import` 等也调用它。密集单池 B 块为二次扫描 | 已在命中后停止。相同 API、计费与输入实测见下；不增加字段、分配或 backing |
| CUDA BF16/F32 attention 的平方 scratch | [ggml CUDA](../../../src/compute_provider_ggml_cuda/ggml_cuda_provider.cc) 仅 F16 走 flash，BF16/F32 物化 scores。B1/H8/Q=KV8192 时，单个 F32 scores 为 **2 GiB**，不含 softmax 等其它 scratch。Windows 当前无 FlashInfer 路线 | 已接通适用域的 BF16 融合程序，实际 Hyle 的空间结果见下；F32 和其它不受理域未改，不能外推全部 attention |
| io_uring CQE 分类的线性查表 | [Proactor::reap](../../../src/io_core/proactor.cc) 原来每个 CQE 扫描 pending accept/close/statx；N 个未决 special op 可使完成批次最坏 O(N²)，不相关 read/write 也付扫描成本 | 已改为单个 token 索引；真实 native 同工规模对照见下，accept找空位/close提交去重的扫描仍保留 |

CUDA 的 F16 flash 分支也会物化 Q×KV 的 F16 causal mask，8K方形时为128 MiB；
不能因为主 attention kernel 融合就假定整个调用的额外空间为线性。
锁定 Torch CUDA 的 Windows 二进制没有 FlashAttention，但已有 CUTLASS BF16 memory-efficient 内核。
接入使用原生 `_efficient_attention_forward`、bottom-right mask type、显式scale、F32 accumulator；
GQA按KV头分组切片，零stride view复用KV，不复制KV缓存，不物化Q×KV mask/scores。
内核概率tile使用BF16，与现有全F32中间图并非逐位等价。

Windows RTX4070 原生[探针](bf16-sdpa/sdpa_probe.cc)已验证10个BF16场景：同头、非方形bottom-right、
H8/KV2/D256零stride GQA、decode、零/负scale、非因果Q>KV，以及超F16范围的K/V。
独立FP64公式参考的最大相对L2误差0.174713%，均满足探针声明的绝对0.01＋相对1.5%预算。
[原始结果](bf16-sdpa/results.jsonl)明确 `flash_available=false`。
Q128/KV257/H8/D64 的额外 Torch allocated 峰值262144 B，包含分组输出/拼接；该形状单个F32 scores
为1052672 B。这仅证明小规模原生接法可行，不是8K总显存、Hyle接入或吞吐结果。
复现使用本机已有锁定SDK：
`python evidence/performance/library-review-2026-10-18/bf16-sdpa/run_probe.py --compile --run`，
产物写入 `.scratch/perf-review-2026-10-18/bf16-sdpa`。该原生探针的1.5%预算不替代Hyle回归测试。

### Hyle BF16 融合接入与空间

[实现](../../../src/compute_provider_torch_cuda/artifact_provider.cc)接入既有Torch CUDA provider，
不新增用户选项或依赖。准备期检查dense BSHD、BF16、无mask端口、F32 accumulation、
SM80–SM121、相同Q/K/V depth为8–256且为8的倍数，以及原生索引、grid和16-byte对齐约束。
支持同头/GQA、bottom-right causal及显式零/负scale；causal要求KV≥Q。
默认排序为适用的FlashInfer → 适用的Torch BF16融合 → ggml，其它dtype/算子排序不变。
仍声明Managed、provider私有workspace大小未知、不支持外部capture，未伪装成预分配程序。

真实[Hyle Direct空间消费者](bf16-sdpa/hyle_space_probe.cc)在RTX4070上使用CUDA MemoryResource、
实际SharedQueue和Completion；每个形状独立新进程。B1/H8/KV2/D256、Q=KV，Q/K为零、V为1.25，
逐元素核对完整输出均精确正确，provider ID与Queue wait均核对成功。

| Q=KV | Hyle端口请求 | Torch首次submit额外allocated峰值 | Torch reserved峰值 | 旧单个F32 scores理论值 |
|---:|---:|---:|---:|---:|
| 512 | 5 MiB | 4 MiB | 22 MiB | 8 MiB |
| 1024 | 10 MiB | 8 MiB | 20 MiB | 32 MiB |
| 2048 | 20 MiB | 16 MiB | 20 MiB | 128 MiB |
| 8192 | 80 MiB | 64 MiB | 80 MiB | 2048 MiB |

[原始数据](bf16-sdpa/hyle-space-results.jsonl)分别记录cold prepare前后、submit前与wait后计数。
额外峰值为submit前resetPeak后allocated_peak减submit前allocated_current，包含首次执行、GQA分组输出和拼接。
四个cold prepare后Torch allocated/reserved均为零。这里验证了该形状族额外分配的线性增长，
**不是总VRAM、旧完整程序峰值、所有head/长度的上界或模型tokens/s比较**；驱动、module与其它分配不在Torch计数中。

实际[Hyle回归](../../../tests/compute_provider_torch_cuda/attention_test.cc)共9项通过：
[数值与拒绝5项](bf16-sdpa/numeric-and-admission-final.log)，包括独立FP64参考、同头/GQA、decode、
非方形causal、零/负scale、非因果Q>KV、超F16范围和重绑定，保留现有绝对0.01误差预算；
[异步4项](bf16-sdpa/async-lifetime-final.log)覆盖Direct与generic Graph的外部stream先写后算、
pending时销毁artifact/invocation，以及原ATen allocator受控OOM后保活已接纳前缀、报告错误、同Queue恢复。
实际GPU使用显式Torch-only Context；默认排序的[42项独立检查](cuda-policy/policy_probe.cc)也通过，
两者不等于完整默认CUDA/Graph roster的GPU测试，后者及Linux CUDA本轮未运行。
空间复现：`python evidence/performance/library-review-2026-10-18/bf16-sdpa/run_hyle_space.py --compile --run`，
复用本机锁定SDK与现成基础静态库；捕获的编译/链接命令和[source/binary hash](bf16-sdpa/hyle-source-binary-sha256.json)随数据保存。

验证中还修复了Windows CUDA13.3的`bin/x64`运行库搜索，以及backend测试覆盖用户architecture设置的问题。
临时测试对象构建曾意外触发runtime staging；已先写`test.ps1`再执行，删除29个本轮新副本、回收4.51 GiB，
原SDK与9个既有缓存runtime全部保留。最终复现编译直接调用生成的cl命令，不再次触发该复制步骤。

## 有效历史小输，当前版本待复测

比值统一为吞吐 Hyle/peer，或延迟 peer/Hyle；低于1有利于 peer。

| 项目 | 原证据 | 当前边界 |
|---|---|---|
| Apple ragged F16 prefill、D256 | [summary](../inference-g01-native-ragged-apple-m1-p1/summary.json)，8组比值0.98266–0.98941，所有CI上界低于1；约1.07%–1.76%延迟差 | 相同 packed 输入与原 MLX 数学、完整 warm 调用；当前仍有对应实现。需分清 kernel、managed metadata 与完成观察成本 |
| WSL 单文件 atomic replace watch | [summary](../io-watch-p1-v1-30e133db-w1-wsl-4c06445c/summary.json)，4点0.96813–0.98638，2组CI上界低于1 | 同写入/rename/通知分类/读取generation/完成。当前保留主要机制，但新增递归路径；需重测现版本，不按“低频可接受”免查 |
| TaskScope、32 child、4 workers | 本机历史 `C:/public/project-hyle/_validation/sched-v9-task-scope-cost-audit/p1-qualified-evidence/summary.json`；一个 placement+peer-first 组合0.970796，CI[0.955245,0.989531]，其它3组跨1 | 两侧含相同 child 工作与 Scheduler::drain。方向依赖布局/顺序，当前增加了 priority 处理；先核 A/A 与布局响应，不能直接宣布当前普遍慢2.92% |
| G09 混合 GPU 完成观察 | 旧 native-wait 诊断把等待放在 Scheduler worker；当前 Vulkan 已用 BlockingPool，Apple 已有 native subscription | 旧倍率不是同线程资源方案；修后缺同输入、同线程预算的完整性能包。历史 BLOCKED 不是性能证据 |

旧报告中的摊销空 harness ns/op 不等于硬件时钟分辨率。历史CI同侧不能替代当前版本的同路径 A/A、
顺序/布局响应和完整完成边界核对。

## 已排除的旧“输项”

- G04/G05 按用户决定保持解决；[同 FP8 输入](../g04-g05-bf16-partials-2026-10-17/same-fp8/README.md)
  已纠正计时内解码与计时前解码的不同工作，不重开旧排名。
- G07 Apple/CUDA m64 的早期0.73–0.76 / 0.94–0.97 已被
  [Apple attempt3](../inference-g07-apple-packed-attempt-3/receipt.json) 与
  [CUDA attempt3](../inference-g07-cuda-packed-attempt-3/receipt.json) 覆盖，当前仍保留修复。
- SIMD旧逐操作43条负向CI涉及32个操作，当前都已切 Highway；旧 tanh 慢的是未选路线。
- CPU旧自研SDPA/MatMul、撤回的oneDNN FFN gate和大行RMS旧实现不代表当前默认路线。
- CPU paged 对手冷阶段另复制约3.67 MB连续KV缓存，候选直接读分页。热期不可变缓存的优势不能称为
  同分页输入算法胜出；需同时比较冷准备、增长、额外空间和热执行。
- Obs concurrent atomic limiter 对 single-owner scalar gate；Arena.clear 保留 backing 对 pmr.release
  释放重分配；裸 atomic pointer 对安全 reclaim；SIMD全域数学对 valid-domain ceiling，均需先对齐职责。
- CUDA/Apple/Vulkan native-only 边界少了 owner、依赖和真实退休等工作，不能直接作同责算法排名；
  额外 adapter 成本仍值得用完整消费者定位，不因“不完全可比”而放弃优化。

## StateStore 同工规模实测

Windows11 26200 / i9-13900K；MSVC19.50.35728，Release/O2、static、trace与sanitizer关闭。
固定逻辑CPU4 affinity，不宣称已证明其P/E核心类型。准备、分配、导入与销毁均在计时外，
两侧计时相同次数的公共 `StateStore::stats()`；保留同一 schema、version、原始块、root，
校验所测计费字段、versions及累计checksum。

| 块数 / pool roots | 旧实现 μs/call | 修复后 μs/call | 成对倍率中位数 |
|---|---:|---:|---:|
| 128 / 1 | 3.289 | 0.101 | 32.57× |
| 128 / 4 | 3.943 | 0.160 | 24.64× |
| 512 / 1 | 41.019 | 0.347 | 118.79× |
| 512 / 4 | 44.516 | 0.563 | 79.36× |
| 2048 / 1 | 774.131 | 1.400 | 554.25× |
| 2048 / 4 | 776.213 | 2.181 | 354.71× |

5对独立进程 baseline A/A、9对交替A/B与B/A，全部样本和离群点保留在
[raw](state-stats-raw.json)，[summary](state-stats-summary.json)记录二进制hash和范围。
A/A中位数0.992–1.012，单次范围0.834–1.549；A/B最小倍率仍为24.45。
QPC频率10 MHz、tick 100 ns；最快聚合区间22.1 μs，未用单次100 ns读数裁决差异。

[探针源码](state_stats_probe.cc)对 baseline `90e10686e` 与修复分别链接同配置的基础静态库，
编译参数与依赖见[构建命令](state-stats-build.json)。只重建 compute_state 及测试，复用基础缓存。
25项 StateVersions/StateClone 测试通过，含 A/B/A roots、独立metadata及collect后前段空洞的精确计费。

这里证明的是统计调用及其重复 root 扫描，**不是引擎 tokens/s 提高这些倍数**。
密集单池降为线性；不同 root、长空洞前缀和 clone-route 查找仍有扫描，最坏复杂度没有普遍降为线性。
`import` 的其它去重扫描也未被这项修复消除。

## io_uring 同工规模实测

WSL Ubuntu24.04/GCC14.2、kernel6.18.54，Release/O3、shared、Werror。实际 `io_uring_setup` 与
消费者 backend 检查均成功，不是epoll fallback。运行于Linux本地文件系统，taskset固定虚拟CPU2，
不宣称Hyper-V物理P核已固定。只重建相关88步基础闭包，没有GPU/协议全量构建。

[同一消费者](io-cqe/batch.cc)以不同 `LD_LIBRARY_PATH` 加载旧库和索引库。
[loader记录](io-cqe/loader-identity.json)核对了实际落点、两库hash及exe的RUNPATH。
正式AB使用同一个7形状exe；[首轮基线](io-cqe/baseline-first.jsonl)是增加pure-read前的6形状初探，
其旧exe hash不混入正式AB结论。
5对交替AB/BA，每行262144个operation，计入submit、statx native记录分配、全部CQE、转换/回收、
token/8B数据校验及inflight归零；init/reserve、fd打开/预热、用户buffers在两侧计时外。

| 负载 | batch | 旧 ns/op | 新 ns/op | 配对倍率中位数 |
|---|---:|---:|---:|---:|
| statx | 16 | 469.0 | 452.5 | 1.037× |
| statx | 256 | 508.6 | 467.3 | 1.091× |
| statx | 1024 | 740.3 | 570.5 | 1.306× |
| read＋statx | 16 | 341.8 | 344.9 | 0.993× |
| read＋statx | 256 | 285.4 | 260.7 | 1.087× |
| read＋statx | 1024 | 405.9 | 306.6 | 1.337× |
| pure read | 1024 | 103.9 | 102.1 | 1.014× |

256/1024的两种负载均5/5对获益；不计算或虚构CI，不外推引擎tokens/s。
[原始AB](io-cqe/all-pairs.json)/[摘要](io-cqe/summary.json)保留小mixed负点及pure-read的0.9047负对。
再用相同库各做5对[同路径AA](io-cqe/controls-all-pairs.json)：mixed16旧范围0.9667–1.0060、
新0.9430–1.0432；pure-read1024旧0.9281–1.0256、新0.9032–1.0148。
该运行波动可重现原小负对的量级，目前不足以认定稳定小回退，也不证明严格等价；不将其冒充时钟量化问题。

[构建与回归](io-cqe/checks.json)：25项Proactor测试23通过、2跳过；statx失败/retry、混合批次与取消ack/target分离实际通过。
当前kernel对direct accept返回-EINVAL，成功slot/批量close生命周期仍未实测。
索引保留statx的稳定kernel输出地址、原token/取消/slot和失败回滚；无special记录的普通CQE直接跳过哈希。
代价是更多冷host索引容量，并可能因tombstone回收重分配，未声称空间减小或永久无分配。

## 模块覆盖与未测范围

- Core/Error/Obs/Math/Time/TimerQueue/Platform/Thread/FS/Process/Util：查现有证据与机制；
  没有确认新的当前同责输项。许多冷查询只有结构依据，timer主要覆盖64项生命周期。
- Mem/VM/Reclaim、Container/Intrusive/Concurrent/Queue/Text：查容量、生命周期、查找/删除机制与现有数据；
  UTF、membership、Error success等轻微负点值CI跨1。hash churn、并发erase、各queue政策及其它规模仍缺现版本充分数据。
- SIMD：查原逐操作结果到当前 provider 的映射、row/dtype/activation/inference kernels；
  不能将少量dense F32 consumer外推所有dtype、ISA、布局和规模。
- Sched与IO各协议：查现有fanout/completion、HTTP/TLS/HTTP2/transfer/mask/watch及计时源码；
  当前DNS/UDP/local/signal/file-region、递归watch、priority-enabled、blocking pool等未充分测量。
- Compute/Ops/State/Batch/Stream、Graph/Batch、全部CPU/CUDA/Apple/Vulkan providers与route、
  三种CPU＋GPU bridge均纳入调研。证据包括当前原生片段、encoded/paged/ragged、搬运与owner成本；
  跨形状性能、真实常量与状态增长、混合并行完整生命周期及最终模型tokens/s仍需要实际消费者。
- Marlin factor仅历史占位，不为已退役实现恢复一套性能验证。

后续先处理已确认的平方 scratch/扫描，再复现上述小输；不为旧标签重跑全库或给所有模块补证书。
HTTP2 benchmark的私有构造接口也已改为公开 `TcpStream::adopt`，保留原计时边界；尚未在Linux编译该fixture。
