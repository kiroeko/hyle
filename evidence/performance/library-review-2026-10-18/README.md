# 八之五：全模块性能排查

基线 `90e10686e`。先核对工作量/精度/生命周期，再核仪器与噪声，最后判断实现。
本次调研覆盖中央构建表的64个 factor（含退役 Marlin 占位），不是64个模块都已重新计时。
旧结果只有在当前实现仍对应、输入与完成边界一致时才作为线索；旧 PASS、5%容忍带和状态标签不决定结论。

## 当前源码明确的问题

| 项目 | 证据与影响 | 本轮处理 |
|---|---|---|
| CPU 普通路线漏注册 ggml | [owner CMake](../../../src/compute_provider_ggml/CMakeLists.txt) 用 `NO_DEFINE`，而 [Direct](../../../src/compute_route_cpu/cpu_context.cc) / [Graph](../../../src/graph_route_cpu/cpu_context.cc) 以该 presence 宏编译注册。实际库虽链接，候选却消失，旧 CPU 排名不能直接继承 | 已恢复 target 的宏传播；实际 owner 声明的配置回归通过。未测修复后的 CPU 模型倍率 |
| StateStore 重复扫描同池 roots | [stats_locked](../../../src/compute_state/state_versions.cc) 对已命中 root 仍扫描所有更早块；回收形成的前缀空洞也被逐块重扫，`replace` / `import` 等会调用它 | 已在命中后停止，并将前缀空洞一次跳过。相同 API、计费与输入实测见下；不增加字段、分配或 backing |
| CUDA attention 的平方 scratch | [ggml CUDA](../../../src/compute_provider_ggml_cuda/ggml_cuda_provider.cc) 原BF16/F32物化scores，F16物化causal mask。B1/H8/Q=KV8192时，单个F32 scores为2 GiB | BF16适用域已融合；ggml大型独立SDPA已按Q分块、复用完整KV与临时空间。实际8K F32显式private从8368降至48.5 MiB，见下 |
| CUDA声明F32但实际缩精度 | ggml默认cuBLAS handle启用TF32；F16 flash的PV为Half累加且不读PREC_F32 | 显式F32源请求局部切DEFAULT_MATH并恢复；ggml F16改F32 math再转最终端口。未把旧低精度速度作同合同输项 |
| io_uring CQE 分类的线性查表 | [Proactor::reap](../../../src/io_core/proactor.cc) 原来每个 CQE 扫描 pending accept/close/statx；N 个未决 special op 可使完成批次最坏 O(N²)，不相关 read/write 也付扫描成本 | 已改为单个 token 索引；真实 native 同工规模对照见下，accept找空位/close提交去重的扫描仍保留 |
| Linux watch重复close-write唤醒 | [inotify mask](../../../src/io_watch/watch_linux.cc) 同时订阅IN_MODIFY和IN_CLOSE_WRITE；公开接口只提供内容变化提示，未承诺writer-close。即使最终忽略sibling名称，额外通知也会唤醒owner | 已去冗余close订阅；实际减少约1.97次上下文切换/发布。16项watch测试通过，剩余小负样本未声称全部解决 |
| TaskScope读写状态布局 | [私有State](../../../src/sched/task_scope.cc) 的cancel token与admission写入共用cache line；32-child消费者当前复现小输 | 已分隔control字段。scope-only绝对时间有改善信号，代价为本机每scope冷状态增加48 B；31项测试通过，未直接测硬件cache事件或跨平台收益 |

原CUDA F16 flash分支还会物化Q×KV的F16 causal mask，8K方形时为128 MiB；
这条分支已因实际Half累加问题替换，不能仅凭主kernel融合推断整个调用空间或精度。
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
8K的wait后仍有32 MiB allocated结果由invocation持有，直至重用或销毁；80 MiB reserved也不等于已归还驱动。

实际[Hyle回归](../../../tests/compute_provider_torch_cuda/attention_test.cc)共9项通过：
[数值与拒绝5项](bf16-sdpa/numeric-and-admission-final.log)，包括独立FP64参考、同头/GQA、decode、
非方形causal、零/负scale、非因果Q>KV、超F16范围和重绑定，保留现有绝对0.01误差预算；
[异步4项](bf16-sdpa/async-lifetime-final.log)覆盖Direct与generic Graph的外部stream先写后算、
pending时销毁artifact/invocation，以及原ATen allocator受控OOM后保活已接纳前缀、报告错误、同Queue恢复。
实际GPU使用显式Torch-only Context；默认排序的[42项独立检查](cuda-policy/policy_probe.cc)也通过，
两者不等于完整默认CUDA/Graph roster的GPU测试，后者及Linux CUDA本轮未运行。
补做单个[BF16弱概率×大V诊断](bf16-sdpa/hyle-weak-result.json)：实际V=59904，最大相对FP64误差0.3693%、
绝对误差0.02477；只做最终BF16舍入的oracle自身绝对误差已达0.01558，候选与该oracle最多差1 BF16 ULP。
它保留了弱概率贡献，也说明上述0.01只用于原回归输入，不能当作全BF16输出域的绝对误差保证。
未把这项无容差诊断写成通过1.5%门槛或模型精度不下降；完整模型精度仍待真实引擎比较。
空间复现：`python evidence/performance/library-review-2026-10-18/bf16-sdpa/run_hyle_space.py --compile --run`，
复用本机锁定SDK与现成基础静态库；捕获的编译/链接命令和[source/binary hash](bf16-sdpa/hyle-source-binary-sha256.json)随数据保存。

不能直接把该融合入口扩至F32/F16。[锁定SDK源码复核](bf16-sdpa/f32-precision-source-review.json)确认：
SM80+ F32的QK/PV固定选择CUTLASS TF32x3，虽为F32累加，却不是严格F32乘法；
该入口没有allowTF32运行时开关，关闭cuBLAS TF32也不改变编译好的内核。
F16则会把指数权重tile缩成Half后再乘V，弱概率反例已实测失败，见下。
两种Torch扩展均未应用；保留原精度的有界Q分块已在ggml路径实现。

验证中还修复了Windows CUDA13.3的`bin/x64`运行库搜索，以及backend测试覆盖用户architecture设置的问题。
临时测试对象构建曾意外触发runtime staging；已先写`test.ps1`再执行，删除29个本轮新副本、回收4.51 GiB，
原SDK与9个既有缓存runtime全部保留。最终复现编译直接调用生成的cl命令，不再次触发该复制步骤。

### ggml CUDA：保留精度并复用Q块空间

大型独立SDPA保持完整KV，只沿Q分块；因果偏移为`KV-Q总长+q0`，保持`scale→mask→softmax`。
沿用原F32 QK、概率、PV，末端转端口dtype；分块结果copy到原BSHD输出的strided view，保留batch步幅。
单图保持原root stream顺序，用既有ggml gallocr按生命周期复用。冷阶段reserve backing，
首次提交绑定真实地址后alloc_graph，以后刷新所有views；不为每个Q块重复构造KV转换，无新选项或依赖。
原score估算超过8 MiB时分块，每块最多128行；64 MiB用于选择行数，
**不是总显存上限**：至少一完整KV行、K/V转换、原生pool、端口及driver仍另计。
算术量仍为O(Q·KV·D)，host图元数据与launch数量随块数增加；本批未测吞吐。

精度修正先于对照：[原生cuBLAS因果探针](ggml-attention/summary.json)用纯F32 A/B，
同compute32F仅换TF32 math→DEFAULT，max误差从1.52e-4降至1.19e-6；SGEMM同样复现。
[锁定补丁](../../../third_party/ggml_cuda_runtime/patches.json)按显式F32源约束局部切mode，
正常与异常路径恢复私有handle，保留原kernel。仅重编native host dispatch一个TU，未重编全部CUDA实例。

该pin在SM89的F16 flash使用Half PV累加，设置PREC_F32无效。
ggml F16因此合流F32 math；原长域prefix均值误差0.00206059超过既有0.002，修后通过。
另试Torch原生Half融合：score0/-18、V0/60000的合法F16输入期望约0.0128–0.0274，实际全0；
max误差0.02741395，故未启用。promote-F32再融合的单例虽通过，但使用TF32x3，未混同严格F32排名。
F16的快速融合路线仍需补齐；本修正可能付出吞吐代价，未把精度差当性能胜出。

实际Hyle Direct、CUDA MemoryResource/SharedQueue消费者，每个形状独立进程；B1/H8/KV2/D256，
Q=KV、Q/K=0、V=1.25。两侧共用修正后的native精度模式；F16旧侧仍为原flash。
冷初始化诊断仅累计owned ggml buffers与gallocr backing：

| dtype / Q=KV | 原显式private MiB | 修后显式private MiB | 常量oracle最大误差：旧 / 新 |
|---|---:|---:|---:|
| F32 / 512 | 43 | 5 | 7.99e-6 / 7.99e-6 |
| F32 / 2048 | 556 | 17 | 3.43e-5 / 3.43e-5 |
| F32 / 8192 | 8368 | 48.5 | 1.42e-4 / 1.42e-4 |
| F16 / 2049 | 40.02 | 17.01 | 0.000977 / 0 |
| F16 / 8193 | 256.05 | 48.25 | 0.1494 / 0 |

F32的2K/8K完整输出SHA256分别相同，但两侧都超过空间探针自定的1e-5预算；
失败/exit1完整保留，**没有改容差或称这两项数值通过**。分配计数不依赖该数值判定。
CUDA mem-info的各阶段驻留快照也保留，它包含其它设备占用，不是瞬时峰值或上表private计数。

最终10项[attention回归](../../../tests/compute_provider_ggml_cuda/attention_test.cc)通过：
原输入预算不变，覆盖非零QK、GQA、decode、非方形causal、零/负scale、非因果Q>KV、
跨块/尾块、双batch、双invocation/Queue、端口重绑、BF16大范围、F16弱概率及MLA冷受理边界。
验证闭包为当前provider TU、仅dense输入的fresh helper、相同foundation缓存与同pin native库加精度overlay；
helper显式拒绝其它算子，以复用缺新IQ1实例的旧CUDA cache。不是完整默认roster、generic Graph、
全IQ1 backend重建、故障注入或Linux CUDA验证。
[摘要](ggml-attention/summary.json)列边界；[运行原始包](ggml-attention/raw-results.zip)保留失败、最终测试源码/命令/hash、
两版空间探针及Torch F16反例，[实现来源包](ggml-attention/provider-source.zip)保留native patch和provider验证闭包。
依赖tree/NOTICE锁已同步并通过局部完整性检查。

## 有效历史小输，当前版本待复测

比值统一为吞吐 Hyle/peer，或延迟 peer/Hyle；低于1有利于 peer。

| 项目 | 原证据 | 当前边界 |
|---|---|---|
| Apple ragged F16 prefill、D256 | [旧summary](../inference-g01-native-ragged-apple-m1-p1/summary.json)，8组比值0.98266–0.98941，所有CI上界低于1；约1.07%–1.76%延迟差 | 已在Air复测当前对应路径；默认fast较快但不满足同数值合同，同合同force-fused未确认当前稳定慢，见下。旧小输不直接继承 |
| WSL 单文件 atomic replace watch | [旧summary](../io-watch-p1-v1-30e133db-w1-wsl-4c06445c/summary.json)，4点0.96813–0.98638，2组CI上界低于1 | 当前已复测并修掉冗余close-write唤醒；修后仍有0.1–1.0%小负样本，未确认稳定或严格等价，见下 |
| TaskScope、32 child、4 workers | 本机历史 `C:/public/project-hyle/_validation/sched-v9-task-scope-cost-audit/p1-qualified-evidence/summary.json`；一个 placement+peer-first 组合0.970796，CI[0.955245,0.989531]，其它3组跨1 | 当前4组复现约4–7%差，随后应用私有布局优化；混合peer漂移与scope-only小样本限制均保留，见下。不能宣称所有scope负载已解决 |
| G09 混合 GPU 完成观察 | 旧 native-wait 诊断把等待放在 Scheduler worker；当前 Vulkan 已用 BlockingPool，Apple 已有 native subscription | Air与Windows Vulkan当前受限同owner控制均未确认现有观察机制更慢，见下；Windows采用不同CPU算子，旧Abs负载及完整多slot排名仍未重测。历史 BLOCKED 不是性能证据 |

旧报告中的摊销空 harness ns/op 不等于硬件时钟分辨率。历史CI同侧不能替代当前版本的同路径 A/A、
顺序/布局响应和完整完成边界核对。

### Apple D256：先排除中间精度差

Air Mac16,12/M4/16 GiB，复用现有G09 SDK的小consumer；相关Apple batch/context/Queue/ResultSlot/
Completion源码hash与Windows一致，其余已解释差异见[来源与全部结果](apple-ragged/summary.json)。
这不是整个HEAD重建，也不是未改原版MLX排名：参考为同pin的patched-MLX fast入口。
4个平衡执行/分配顺序的进程中，默认fast在D256 prefill快约9–12%；两侧各自A/A未复现此恒定偏差。
raw compiled绕过Hyle Queue/Completion后差额仍在；令fast也走融合路径后未确认Hyle当前稳定更慢。
不能把wait中含调度/native求值/GPU的时间直接当纯GPU时间。

默认fast会物化F16 scores/softmax概率。对相同Q[64,128]/KV[1024,768]、H8/KV2/D256/F16/causal，
强键score=0/V=0，其余score=-16/V=60000的合法输入，Hyle最大绝对误差**0.001995**，
默认fast为**0.411575**，原门槛为0.002。较快参考丢失的弱概率贡献已超合同，不能直接换入。
[原始压缩包](apple-ragged/raw-results.zip)保留命令、stdout/stderr、独立oracle源码、graph与噪声样本；
诊断进程只记录两侧误差，exit0不是默认fast数值通过。

实际Air使用普通Steel（非NAX），BQ32/BK16/WM4、静态threadgroup29184 B。
唯一保精度的BQ16/WM2候选把该量降为20736 B，但同融合参考下AB/BA稳定慢14–15%，已拒绝。
产品与SDK原件未改，仅新增约49 MiB小consumer产物；不再为追低精度参考扩展调参。

### Linux watch：冗余事件导致额外往返

[当前对照](io-watch/summary.json)使用相同8-byte generation写入/close→rename→通知匹配→重读确认，
每行4096次、每路5行，setup/warm/quiescence及销毁在计时外。WSL原生文件系统、guest CPU2，
两侧实际均2个用户线程；复用GCC14 Release shared缓存，libuv为锁定commit。
历史static构建不冒充本轮shared构建；初探separate-process没有保存当时exe hash，不补认最终exe身份。
所有初探、A/A及噪声负对保留在[压缩原始包](io-watch/raw-results.zip)。

同进程两顺序先复现约5.7–6.2%负差。只用诊断shim去掉IN_CLOSE_WRITE，20480次发布的
context switches从159881降至119515，约少1.97次/发布；libuv相应119384。
perf/strace只用于机制观察，不计入吞吐样本；未从无法解码的trace推断io-wq根因。
最后产品直接移除冗余订阅，保持IN_MODIFY、metadata、create/delete、rename及root失效通知。
新增write/ftruncate在fd仍打开时exact/direct均收到kContent，连同atomic replace、cancel/close、tree共16项通过。

最终同一个exe以不同LD_LIBRARY_PATH选旧/新watch库，ldd/RUNPATH/hash核对，其余io_core/sched依赖相同。
下表保留两对完整顺序结果，非profile计时：

| 对照 | 旧Hyle μs | 旧过程libuv μs | 新Hyle μs | 新过程libuv μs | 吞吐比旧→新 |
|---|---:|---:|---:|---:|---:|
| pair0、Hyle先 | 41.505 | 36.739 | 36.404 | 36.365 | 0.88518→0.99894 |
| pair1、libuv先 | 38.106 | 37.745 | 36.235 | 35.878 | 0.99054→0.99014 |

确认修掉的是无必要的native唤醒；第二对按peer归一几乎无收益，不选点夸大。
修后约0.1–1.0%负点仍需进一步判断稳定性，不能称严格等价、所有性能差已消失或时钟分辨率问题。

### TaskScope：分隔只读token与admission写入

[当前研究](task-scope/summary.json)沿用32个Status child、各yield一次、4个persistent workers、
每16个owner一次业务失败；两侧包含相同cancel root、join、drain至off-stack cleanup及owner销毁。
manual peer是公开Hyle API组合，不是另一个调度引擎。固定4个不同物理核、CPU0/2/4/6，
Windows EfficiencyClass均为1，不根据编号推断P/E。QPC为10 MHz；旧0.2 ns空harness不是硬件tick。

初始4组当前混合消费者均小输，两侧A/A未确认恒定计时偏差。push-front和调整取消检查顺序均无稳定收益，未应用。
phase诊断没有证明join/Event为根因；它包含worker重叠且会扰动调度。只将私有State的control字段按已有
destructive-interference常量对齐，分开admission写入与child读取的token；没有改变取消、完成或错误协议。
混合BA中的manual peer也从13–14 μs漂到约17 μs，故不能只凭归一化反转认定提速。

改用已有AA_ROUTE，让两臂都只运行TaskScope，按下面进程顺序记录绝对时间：

| 进程 | 实现 | 第一臂 μs | 第二臂 μs |
|---|---|---:|---:|
| 1 | 原布局 | 17.50659 | 16.43563 |
| 2 | 分隔布局 | 13.10329 | 13.50180 |
| 3 | 分隔布局 | 12.96347 | 12.84731 |
| 4 | 原布局 | 14.10180 | 14.00150 |

候选也低于后一次较稳基线；首过程8.8%/12.2%噪声完整保留，不虚构跨进程CI或精确普遍倍率。
所有同过程A/A CI均跨1。Windows/MSVC实际State由80 B/align8变为128 B/align64，
每scope增加48 B冷内存，child大小不变；Apple ARM64常量为128，未测其布局和性能。
既有31项TaskScope测试通过，测试专用HYLE_TESTING仅用于正确性target，全部性能target关闭。
[原始包](task-scope/raw-results.zip)保留28进程、失败候选、布局、命令和测试；
[源码身份](task-scope/final-source-identity.json)区分基线、测量与应用源码。
只重编TaskScope TU并复用相同foundation缓存，不是整个HEAD运行库重建，也未跑全Sched/TSan或跨系统矩阵。

### G09：当前订阅与同owner等待控制

Air上只重编当前[observer TU](../../../src/graph_route_cpu_apple/invocation.cc)，
其原始源码及Apple context与Windows hash一致，复用同一SDK/foundation和patched MLX。
诊断控制只在parallel native lane有一个pending且无ready时解开monitor、等待原Completion；
两侧保留subscription、owner、lease、checked finish、callback join及退休，不能称理论最小native成本。
同一4194304元素F32 Graph、2个Scheduler workers；两侧进程均观察到7个线程，
含调用线程、hyle-mlx及3个匿名runtime线程。快照不等于瞬时同时runnable数量。

| 顺序 | 当前订阅 ms | 直接等待控制 ms | 串行同路径：当前 / 控制 ms |
|---|---:|---:|---:|
| AB | 2.158958 | 2.122333 | 3.080542 / 3.172834 |
| BA | 2.070042 | 2.090375 | 2.995125 / 2.917584 |

每进程8次warm、31个原始样本；计时包含bind/prepare、submit、owning wait、标量核对、结果释放及drain。
每进程前后完整oracle均通过。smoke计数确认独立形状命中、serial不命中；正式两侧均关闭计数。
[全部原始包](g09/raw-results.zip)保留源码/命令/link map/hash和线程CPU增量；[摘要](g09/summary.json)列边界。
差额反向，未建立稳定当前输项，也未证明等价或serial波动是其原因；未另做A/A，不虚构跨进程CI。
产品未改；这不是原版MLX排名、整个HEAD重建、Windows Vulkan或多slot验证。

Windows Vulkan补做当前路径的受限控制，直接重编4个相关实现TU和小facade，复用已有静态Vulkan/foundation缓存。
原Abs在显式Native CPU不受理，ATen CPU也主动拒绝dense基础运算；最终两侧均采用
CPU Add(x,x) / Vulkan Neg / CPU Add，4194304个F32元素，精确oracle分别为`2*x-1.25`和`1.25-2*x`。
这是当前完成观察的因果诊断，**不是旧Abs G09排名或Air跨平台比较**；未受理与误加载旧smoke的记录全部保留。

| 顺序 | 当前BlockingPool ms | 根线程直接等待控制 ms | 串行同路径：当前 / 控制 ms |
|---|---:|---:|---:|
| AB | 3.8874 | 4.1511 | 6.7582 / 6.9004 |
| BA | 3.7491 | 3.9281 | 6.6266 / 6.6461 |

控制只改单pending、无ready、pool尚未持有该Completion时的等待位置，保留原owner、pool任务启停与退休。
两侧显式2个Scheduler workers，pool容量2、实测live/handles均1，ATen intra/inter均1；
实际OS线程前三进程19、末进程18，线程名均空，额外线程角色未识别，不能称活跃线程完全相等。
等待放到Scheduler根线程也改变其可用性，不能直接当作可替换的生产优化或理论最小native成本。
每形状8次warm、31样本、4进程AB/BA；正式计数关闭，smoke确认独立形状命中2次、serial不命中。
计时包含bind/prepare_replay、submit、完整owner wait、标量核对、所有结果/租约释放及Scheduler drain，
每进程计时前后4M完整oracle均通过。[摘要](g09-windows-vulkan/summary.json)和
[原始包](g09-windows-vulkan/raw-results.zip)保留全部样本、线程CPU快照、命令与来源限制。
该小样本没有显示当前BlockingPool更慢；不据此宣称显著性、等价、生产提速或完整HEAD/多slot覆盖，产品未改。

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
密集单池降为线性；不同 root、内部空洞和 clone-route 查找仍有扫描，最坏复杂度没有普遍降为线性。
`import` 的其它去重扫描也未被这项修复消除。

随后补查回收形成的长空洞前缀：先保留旧池全部块，再导入新池，释放旧version/member并collect。
原循环对每个存活块重扫这段空洞；现在只找一次首个有效anchor，两层循环均从这里开始，
不新增状态、排序、索引或分配。Windows同配置、CPU4，完整stats与checksum一致，25项原状态测试通过。

| 前缀空洞 / 同池存活块 | 原 μs/call，两对 | 修后 μs/call，两对 |
|---|---:|---:|
| 128 / 128 | 4.692 / 4.694 | 0.187 / 0.175 |
| 2048 / 2048 | 1063.764 / 1050.241 | 2.484 / 2.444 |
| 0 / 2048，密集control | 2.055 / 2.051 | 2.085 / 2.054 |

仅两对AB/BA、4进程，无删样本。QPC为10 MHz，最快聚合区间156.4 μs；准备/导入/回收/销毁不计时。
密集control的+1.42%/+0.11%全部保留，不据此判定稳定退化或严格等价。
[摘要](state-prefix/summary.json)与[源码/原始结果包](state-prefix/source-evidence.zip)保存源码和复用库hash、
命令及测试日志。只直接编译相关TU，未全量重建；结果仍仅说明stats调用，不外推引擎吞吐。

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
HTTP2 benchmark的私有构造接口已改为公开 `TcpStream::adopt`，保留原计时边界；
现已用Linux GCC14.2/C++23、当前Windows源码与公开头、锁定nghttp2缓存完成
[`-fsyntax-only -Wall -Wextra -Wpedantic -Werror`检查](http2/syntax-result.json)，无诊断。
未链接或运行该fixture，不据此继承旧性能排名。
