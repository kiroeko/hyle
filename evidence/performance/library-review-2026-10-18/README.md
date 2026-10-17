# 八之五：全模块性能排查

基线 `90e10686e`。先核对工作量/精度/生命周期，再核仪器与噪声，最后判断实现。
本次调研覆盖中央构建表的64个 factor（含退役 Marlin 占位），不是64个模块都已重新计时。
旧结果只有在当前实现仍对应、输入与完成边界一致时才作为线索；旧 PASS、5%容忍带和状态标签不决定结论。

**仍未解决**：HTTP2相对手工单loop的大报文慢点；Apple Sink数学fallback及部分普通未分块域的完整scores空间。
HTTP2在header/body修复后复测仍慢约15.6–17.5%；其后仅去掉一个冗余私有Task帧，未重测该微改动的吞吐。
Apple Sink纯图Q16候选峰值反增10.17%；另一个原生compact融合候选虽省90%峰值，同工却慢约7倍，均未应用。
CUDA长域F32探针的原1e-5预算仍失败，现已定位到原生PV长归约舍入；同native对照输出完全相同，没有证实Hyle额外精度损失，见下。
普通CPU及多节点CacheEncode已复用生命周期，普通/私有paged/ragged causal冷mask已移除；动态PagedRead保留旧arena边界。
Apple已知无NAX硬件的受限F32 D192及D256 causal已融合，D256旧Q2048失败形状在新路线通过；旧失败记录仍保留。
其余queue政策和旧小输仍缺当前同责数据。CPU/Vulkan math attention、Apple受限长Q fallback与CPU Hadamard冷矩阵已修；bounded、ST512、实际Folly SPSC及Sharded非满载当前有限trace均未复现输项。
已确认的源码缺口与已实测输项分列，不能据64个factor扫描称所有shape已穷尽。

## 已确认的问题与处理

| 项目 | 证据与影响 | 本轮处理 |
|---|---|---|
| CPU 普通路线漏注册 ggml | [owner CMake](../../../src/compute_provider_ggml/CMakeLists.txt) 用 `NO_DEFINE`，而 [Direct](../../../src/compute_route_cpu/cpu_context.cc) / [Graph](../../../src/graph_route_cpu/cpu_context.cc) 以该 presence 宏编译注册。实际库虽链接，候选却消失，旧 CPU 排名不能直接继承 | 已恢复 target 的宏传播；实际 owner 声明的配置回归通过。未测修复后的 CPU 模型倍率 |
| StateStore 重复扫描同池 roots | [stats_locked](../../../src/compute_state/state_versions.cc) 对已命中 root 仍扫描所有更早块；回收形成的前缀空洞也被逐块重扫，`replace` / `import` 等会调用它 | 已在命中后停止，并将前缀空洞一次跳过。相同 API、计费与输入实测见下；不增加字段、分配或 backing |
| CUDA attention 的平方 scratch | [ggml CUDA](../../../src/compute_provider_ggml_cuda/ggml_cuda_provider.cc) 原BF16/F32物化scores，F16物化causal mask。B1/H8/Q=KV8192时，单个F32 scores为2 GiB | BF16适用域已融合；ggml大型独立SDPA已按Q分块、复用完整KV与临时空间。实际8K F32显式private从8368降至48.5 MiB，见下 |
| CUDA SinkAttention的按头展开平方scratch | 同provider的独立Sink原来仍分配完整scores/probabilities；H64、Q=KV2048时两块各1 GiB，属于schema允许的prefill | 已复用Q分块并使Graph走单Compute步骤。1025×1041实际private从651.17降至27.57 MiB；调用方mask仍为O(Q·KV)，见下 |
| CUDA variadic Concat重复复制前缀 | [lowering](../../../src/compute_provider_ggml_cuda/tensor_program.cc) 线性链保留每个累积前缀；64个各1 MiB输入会额外保留2016 MiB前缀，最终输出仅64 MiB | 已改为原生平衡树，保持Graph依赖。同输入图的private分配请求2016→320 MiB；消除二次项，不宣称一次写完的最优线性算法 |
| CUDA Hadamard平方矩阵 | [准备与lowering](../../../src/compute_provider_ggml_cuda/ggml_cuda_provider.cc) 原来保存并上传W²矩阵；W4096时主机/设备各64 MiB | 1024/2048/4096已分解为两个小变换；4096常量两侧各降至16 KiB，实际private208 KiB，两对同工延迟13.26–15.36×。≤512保留原矩阵及执行路线，融合仍可能读取它 |
| CUDA声明F32但实际缩精度 | ggml默认cuBLAS handle启用TF32；F16 flash的PV为Half累加且不读PREC_F32 | 显式F32源请求局部切DEFAULT_MATH并恢复；ggml F16改F32 math再转最终端口。未把旧低精度速度作同合同输项 |
| io_uring CQE 分类的线性查表 | [Proactor::reap](../../../src/io_core/proactor.cc) 原来每个 CQE 扫描 pending accept/close/statx；N 个未决 special op 可使完成批次最坏 O(N²)，不相关 read/write 也付扫描成本 | 已改为单个 token 索引；后续accept空位计数/close提交去重扫描也已消除。fixed-slot assignment再以最小候选消除连续填表的重复前缀；一般交错单次搜索仍可能线性。计时与访问计数分开记录 |
| Linux watch重复close-write唤醒 | [inotify mask](../../../src/io_watch/watch_linux.cc) 同时订阅IN_MODIFY和IN_CLOSE_WRITE；公开接口只提供内容变化提示，未承诺writer-close。即使最终忽略sibling名称，额外通知也会唤醒owner | 已去冗余close订阅；实际减少约1.97次上下文切换/发布。16项watch测试通过，剩余小负样本未声称全部解决 |
| TaskScope读写状态布局 | [私有State](../../../src/sched/task_scope.cc) 的cancel token与admission写入共用cache line；32-child消费者当前复现小输 | 已分隔control字段。scope-only绝对时间有改善信号，代价为本机每scope冷状态增加48 B；31项测试通过，未直接测硬件cache事件或跨平台收益 |
| BlockingPool每次受理扫描所有线程 | [reap_retired](../../../src/sched/blocking.cc) 即使没有退休记录也扫描所有保留句柄，每次submit均付O(live threads)查找 | 同锁下以现有存活数/句柄数跳过无退休扫描，无新增状态；18项测试通过。32 busy-worker同工提交两对约46.5/50.6→43.7/44.5 ns，见下；不是round-trip或外部排名 |
| ConcurrentHashMap erase重复hash | [erase](../../../include/hyle/container/concurrent_hash_map.h) 先hash选shard，再由table.erase重算同key | 复用私有prehashed路径，锁/回收/析构顺序不变。计数2→1；长key形状两对改善与AA漂移见下，不声称外部赢 |
| CPU / Vulkan Concat重复复制前缀 | 两个provider仍线性拼接；CPU保留全部中间payload，Vulkan虽有gallocr复用，复制工作仍为二次项 | 已移植平衡树，保持各自dtype/stride受理域；CPU修正重复ports元数据预算。实测与空间边界见下 |
| FlatHash近满rolling churn反复重建 | [grow](../../../include/hyle/container/detail/flat_hash_table.h) 在7/8负载附近反复同容量rehash，清理后几乎没有插入余量 | 自动增长/prepare改为密集时加倍、稀疏时清理；显式reserve合同不变。工作量计数与增加容量的代价见下 |
| Apple普通SDPA启发式回落 | 锁定MLX对Q>8、D192/256默认unfused，物化scores/probabilities；私有ragged的force不影响普通入口 | F16/BF16受限域已融合，1K F16额外峰值约12→2 MiB；后补已知无NAX硬件的F32 D256 causal及D192 tile，两个短Q长KV样本分别约65.66 MiB→528 KiB、65.41 MiB→400 KiB。其余域仍待补 |
| Apple Sink数学fallback绕过分块 | 当前schema为F32/B1/H64或128/KV1/D512，独立Sink入口直接调用原生fast数学fallback，保留完整scores并追加sink列归一化 | Q16图候选峰值反增10.17%；BQ8/BK8 compact融合峰值降90%但同工慢约7倍，均未应用。下一步考虑已有owner Job内真正checked completion的分块，不引入新调度系统 |
| HTTP2大body背压holdout | 修正peer后旧轮慢5.59–8.27%，后续有漂移；header/body修复及当前proactor的单一holdout又观察到15.6–17.5%点差 | 当前残差仍在；只去掉一个无await私有Task帧，原协议/allocator/pins不变，未测其速度收益。两endpoint Task与手工双端loop差异继续分开调查 |
| Vulkan普通causal SDPA平方mask | 已融合的F32/F16分支仍用Q×KV F32 seed与F16 mask承载因果条件 | 复用既有FullF32 causal/window字段，3项定向测试通过；1025×1041 native gallocr容量6402176→0 B，未称total scratch为零或吞吐无退化 |
| Vulkan math attention平方scratch与F32 source遗漏 | wide/Sink物化全scores，部分非连续Q还会在原生dispatch转Half，只有F32 accumulation请求不足以保持乘法精度 | 先补F32 source与Q连续布局，再同精度Q分块；4项测试通过，Sink样本gallocr容量340527360→176687104 B。小KV/大输出保留普通路线，无吞吐结论 |
| CPU Hadamard冷矩阵 | 原生FWHT已是O(WlogW)，但prepare仍构造W² host矩阵，W4096为64 MiB | unary bridge复用各pin原生FWHT，移除矩阵；私有program ABI2随baseline/AVX2选择callback，无新增用户设置或数学内核 |
| CPU普通多节点保留已消费payload | manual arena逐一保存中间值；64个1 MiB Neg串联保留63 MiB中间payload | 普通多节点复用既有gallocr；实测冷私有请求约63.94→2.94 MiB。后补多节点CacheEncode，动态PagedRead边界保留 |
| CPU普通causal SDPA冷mask | Program原来构造并长期保留Q×KV Half mask，普通prefill也会支付平方冷分配/填充 | HYCA隐式边界复用原生vector/tiled路径；Q1不设tag/全零mask，split策略保留。实测prepare少64 MiB payload，见下 |
| Apple GPU长Q数学fallback | 原生unfused保留全Q scores/probabilities；逻辑分块也可能被Metal待完成command buffers同时保留 | 静态fallback域以原数学16行分块，默认设置实测额外MLX峰值80→58.81 MiB；64行版反增8 MiB的负点保留，未称覆盖全部fallback |
| CPU私有paged/ragged causal冷mask | 非boxed Q>1仍构造并保留Q×逻辑KV的Half seed，batch admission还收取旧mask下界 | 复用同一HYCA逻辑行边界；两pin各6项通过。冷plan移除64 MiB seed，generic workspace pool仍单独保留，见下 |
| TimerQueue空闲advance重复扫描 | 每次dispatch观察未到期timer，都会重扫最近bucket全部节点；相同N节点、M次空闲轮询可付O(N·M) | 缓存仍有效的最近deadline，取消最小值后只刷新一次；不改下次advance刷新取消值的合同。27项通过，节点访问计数见下 |
| CPU算术链末端编码禁用生命周期复用 | 只要含CacheEncode，普通多节点recipe仍保留所有中间payload，退回Θ(N·M) | 移除该排除，保留callback/invalid传播/退休顺序与动态PagedRead排除；冷64Neg→Q8私有请求68112224→3100744 B，见下 |
| CPU batch重复预留scratch | common pool按float及两份int上限分配实际backing，CPU native完全不用这些views，却另行分配实际workspace | 私有provider契约允许native owner省略三块scratch；同预算样本少144 MiB/slot，原native预算检查不变，见下 |
| Process环境变更重复比较 | 公共duplicate检查为M²，POSIX/Windows继承合并逐entry扫描全部M变更；Windows还重复做M²不敏感校验 | 排序名字索引改为MlogM+ElogM，保留平台大小写与序列化结果；Windows/WSL回归和同工计数见下，不称完整spawn排名 |
| HTTP2多header/trailer视图错误 | StoredHeader内嵌view有stride，却以连续HttpHeader span发布；现有性能consumer只有一条regular header而未暴露 | 同块中分开放连续refs/views，不增默认分配容量或次数；多header回归旧失败、新14项及epoll通过，没有新的性能排名 |
| HTTP2借用body被后续接收压缩覆盖 | 同一session的其它stream操作可继续接收DATA，原memmove覆盖仍承诺有效的reader span | 原容量改固定接收环，不增字段/分配；旧view保真回归失败，新版及default/epoll各15项通过。独立合同修复，不归因原大body残差 |
| io_uring direct槽位合同与重复扫描 | accept误传CLOEXEC导致EINVAL，close少编码+1会误关其它slot/进程fd；accept按table扫描、close按pending列表查重 | 先修native合同，再以free计数/slot状态消除两处扫描；保留close/accept完成乱序下的新owner，原误关stdin失败及实际访问计数见下 |

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

### SinkAttention：切mask行并保留sink语义

同一Q分块也用于独立F32 SinkAttention，沿完整KV计算；mask仅创建原始`[KV,Q]`的行view，
偏移为`q0*nb[1]`，不复制、不按head展开。每块保留`soft_max_ext(scale,mask)`及同一per-head sink向量，
sink只进入分母。QK/PV继续显式F32；Graph region像普通SDPA一样排除Sink，让它走完整单Compute程序。
没有放宽分块recipe的单节点限制，以免输出叶子与copy terminals在多节点图内失去写入依赖。

新[Sink测试](../../../tests/compute_provider_ggml_cuda/sink_attention_test.cc)在H64/Q257/KV273、
H128/Q129/KV145验证跨块、尾块、非零QK与正负mask、全遮蔽行、每头不同sink、双invocation/Queue及全端口重绑。
稀疏QK与精确可分离V的独立FP64公式逐元素检查，两侧都通过原定1e-5；不是任意dense输入的误差上界。
只格式化最终测试，保留实测/应用两版源码及hash，未为格式重复测试。

RTX4070上唯一空间对照为B1/H64/KV头1/D512、Q1025/KV1041：
显式owned scratch＋gallocr由682797568 B（651.17 MiB）降至28913152 B（27.57 MiB），
两侧同一FP64常量V公式max误差1.3918e-6，provider/Queue检查成功。
外部端口总请求277229892 B，其中dense mask4268100 B仍为公开接口必需的O(Q·KV)输入，
不能把整个调用称为线性空间；此处消除的是按head放大的平方provider scratch。
cudaMemGetInfo仅为同步边界驻留快照，不是峰值；上述private计数不含caller、native pool、cuBLAS与driver。

[摘要](sink-attention/summary.json)及[源码/命令/原始包](sink-attention/source-evidence.zip)记录限定验证闭包：
直接编译当前provider、dense rank1/2/4与Sink专用helper、现成strict host overlay及同pin native缓存，
没有SDK复制或CUDA全量构建。region TU编译通过，generic Graph实际执行、全provider/IQ1与吞吐未测。

### Concat：保留原生依赖，去掉累积前缀链

每层相邻输入两两原生concat，奇数尾项进入下一层，保留输入顺序；单个非空输入仍独立复制，
I32仍走原生CPY而非DUP，不经F32。没有拆Graph region或引入额外写依赖；冷lowering增加O(ports)临时指针vector。
每字节至多经历ceil(log2 ports)层，保留的scratch及复制量为O(total bytes·log ports)，仍非最优的一次写入。

同一64×1 MiB recipe的实际原生lowering逐根统计：旧private payload/native allocation请求
2113929216 B（2016 MiB），修后335544320 B（320 MiB），caller输出64 MiB另计。
这里调用native buft的分配尺寸查询，**没有cudaMalloc这批旧2 GiB workspace**，不是VRAM峰值或吞吐实测。
新旧同exe的3项Direct测试均通过：新增48种dtype/负轴/不同端口长度/奇偶层/单输入组合，
浮点strided输入与raw bits核对，以及原有I32重绑；另以现成静态基础库实际通过70重复ports→Neg的同region Graph测试。
[摘要](concat/summary.json)及[原始包](concat/raw-results.zip)保留loader落点、源码、命令和旧Graph链接失败。
定向helper仅将旧native缓存缺失的3个IQ1映射显式拒绝，产品映射不变；Direct与Graph使用不同基础缓存闭包，
provider/region冻结版本早于本批Sink改动且不含Sink输入，不能称全HEAD/default roster或IQ1验证。没有SDK或CUDA全量重建。

### Hadamard：分解大变换，保留合法的小矩阵

只修改W1024/2048/4096：自然索引为`low+64*high`，先做归一化H64，交换轴并cont后做
H(W/64)，再交换轴并cont恢复原顺序。符号popcount及两个归一化尺度的乘积等于原Sylvester H(W)，
无bit reversal、无额外全局scale；中间F32、末端才转端口dtype。两次mulmat显式F32累加与F32来源模式，
high16/32可用原生dense fallback，high64复用同一H64矩阵；原FWHT仅按源码域选择，未采集独立kernel profiler trace。
保留真实小矩阵供原生fallback/fusion/debug读取，≤512路径原字节不改，没有新vendor、op或用户选项。

新增[两项回归](../../../tests/compute_provider_ggml_cuda/hadamard_test.cc)验证大宽度、多个leading axes、
forward/inverse、全端口重绑及F16/BF16，独立FP64完整popcount公式逐元素核对；
F32沿用2e-6、窄端口沿用既有rounded-output预算。另一个小consumer实际验证4096→Add为一个Graph region。
候选3/3通过，基线2/3通过：旧2048 F32误差约3.87e-4–4.13e-4，失败与原预算保留，未放宽。
此组有限输入不是全数域保证；两阶段舍入一般不与旧dense逐位等价。

实际Direct的W4096、4行、F32同输入：

| 统计 | 原实现 | 修后 |
|---|---:|---:|
| host常量vector size/capacity | 64 MiB | 16 KiB |
| 实际device常量上传 | 64 MiB | 16 KiB |
| 实际owned ggml private，含常量 | 64 MiB | 208 KiB |
| AB submit+wait μs/call | 225.332 | 14.666 |
| BA submit+wait μs/call | 193.200 | 14.570 |

caller端口131072 B另计；新private含3块随rows·W增长的F32中间张量，不能声称任意row数的总scratch都下降。
仅两对独立进程、CPU4、8warm后128次相同公共Direct submit＋Completion::wait，冷prepare/分配在计时外。
固定小整数dyadic输入、4096两侧系数均可被TF32精确表示，计时前后完整FP64 oracle误差均0，
没有把2048旧精度失败当作同工速度比较。QPC10 MHz，候选聚合1.865–1.877 ms；没有CI或引擎tokens/s外推。
[摘要](hadamard/summary.json)和[源码/全部原始包](hadamard/source-evidence.zip)保存冷计数、全部4进程、失败日志与来源hash。
复用同pin原生库及strict host overlay、现成静态Graph/foundation，当前provider/tensor/region与consumer直接编译；
缺3个IQ1映射仅在scratch显式拒绝，不称全SDK/IQ1验证，无CUDA内核全量构建或SDK复制。

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
当时direct accept返回-EINVAL而跳过成功slot/批量close；后续复核确认传入了不允许的CLOEXEC标志，不能据此判断该内核不支持，见本轮direct-slot纠正。
索引保留statx的稳定kernel输出地址、原token/取消/slot和失败回滚；无special记录的普通CQE直接跳过哈希。
代价是更多冷host索引容量，并可能因tombstone回收重分配，未声称空间减小或永久无分配。

## 模块覆盖与未测范围

跨provider再查同类机制，确认以下额外范围；这是当前源码缺口，不是已完成对手实测的输项：

| 路线 | 已确认工作/空间 | 下一步 |
|---|---|---|
| CPU ggml / Vulkan Concat | 已改平衡树；CPU owned payload126→20 MiB，Vulkan真实gallocr容量11.25→7.75 MiB | 定向测试与唯一代表形状完成，见下；未测完整引擎或所有shape |
| CPU / Vulkan ordinary causal SDPA | 两者受理域内的内部平方mask已消除；CPU私有paged/ragged非boxed域也复用HYCA | HYCA保持原native vector/tiled/split及Half PV，独立于boxed reference路线；不外推其它dtype、ISA或非有限值 |
| CPU Sink与混合旧pin F32 math / 多节点 | 分块recipe、普通多节点与多节点CacheEncode已接官方gallocr与既有可恢复OOM补丁，cold reserve保留kPrepared | 两pin相关回归通过；Sink冷私有请求156.48→48.97 MB，64步Neg约67.05→3.09 MB，末端Q8约68.11→3.10 MB。含动态PagedRead的非分块recipe仍保留旧arena |
| Apple ordinary SDPA | F16/BF16受限无mask域及已知无NAX硬件F32 D192、D256 causal已融合；其它GPU长Q静态fallback以原数学16行分块 | typed入口拒绝CPU stream，不列为合法慢路径；显式mask、wide/不同Dv、causal Q>KV及未知/新架构仍需分开。短Q数学分块负候选已撤回，不能称所有完整scores空间已消除 |
| CPU Hadamard冷常量 | W²冷矩阵已由原生unary桥移除 | [前置源码复核](cpu-hadamard/source-feasibility.json)的API缺口已补；实际两pin/baseline/AVX2结果见下，不以dummy/null伪造矩阵storage |
| FlatHash近阈值rolling churn / Queue政策 | map增长策略已修；bounded、ST512、真实Folly SPSC及Sharded非满载均有当前有限对照 | 均未确认当前输项；owner/steal现补完整校验和原始epochs保留，Crossbeam batch32/动态缩容职责明确不同，不继承旧倍率 |

- Core/Error/Obs/Math/Time/TimerQueue/Platform/Thread/FS/Process/Util：查现有证据与机制；
  TimerQueue空闲advance及Process多环境变更的重复扫描已修，有相关回归和同工计数；其余许多冷查询只有结构依据，未建立新的外部排名。
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
HTTP2 benchmark已改为公开 `TcpStream::adopt`，先通过Linux严格syntax检查，随后用锁定nghttp2源码和已有基础库实际链接/运行；
最新同责结果与剩余执行策略差异见下，不继承旧性能排名。

## BlockingPool：无退休时不扫句柄

worker在同一mutex内递减live计数并标记retired，新线程也在此锁内同时登记记录/计数。
保留句柄数等于live数时即可返回；原抽取/锁外join及shutdown慢路径保留，没有额外字段、原子或分配。
18项原及新增混合live/retired测试通过，包含shutdown完成屏障、取消、OOM、cap与反复退休。

Windows逻辑CPU4、每点8192次公共spawn_blocking；先将1或32个实际线程阻塞在外部gate，
计入job/JoinState分配、enqueue与预留vector存句柄。gate释放、全部join/值与checksum、shutdown归零在计时外。
这是提交受理成本，不是offload round-trip。32-worker AB为46.484→43.713 ns，BA为50.635→44.495 ns；
同路径AA旧48.535/47.583、新44.177/43.335 ns。1-worker一正一负及AA绝对漂移均保留，不称严格等价或全形状稳定倍率。
每过程1-worker先于32-worker，allocator暖态不同，不能用两形状绝对值推导线程数扩展曲线。
QPC10 MHz，最短聚合355 μs，不将几ns/op差额当成100 ns tick以下不可测。
[摘要](blocking-pool/summary.json)与[源码/原始包](blocking-pool/source-evidence.zip)记录全部AB/AA、返回值与限定闭包。
仅fresh BlockingPool/probe/test TU及相同基础静态库，未全量Scheduler/TSan或跨平台重建。

原blocking_bench将每批创建/销毁runtime叫warm pool，现改名amortized runtime lifecycle，保留原计时边界；
补全任务失败/值检查，不能少完成工作却记为快。该TU无Marl宏编译通过，未重跑其完整bench或编译可选Marl分支。

## Concurrent erase：复用选分片时已有的hash

ConcurrentHashMap原来在选shard后，由flat table再次hash同一个query；现在走与find/emplace一致的私有prehashed入口。
原probe/比较/slot析构/tombstone及锁范围不变，helper保留原Hash/Eq的noexcept受理约束。
诊断中hit及已分配shard的miss均2→1；零容量flat-map仍不hash直接返回，不把调用次数变成新的公开API保证。
20项定向测试通过，覆盖ConstAccessor阻塞erase、并发更新、OOM锁恢复、碰撞、异构query及非平凡析构。

Windows CPU4，固定8192个512-byte owning string keys、64 shards、同seed/shuffle，以string_view公开erase。
计入hash/锁/比较/析构free，生成/refill与最终检查在外；正式无counter，所有删除数与空表状态核对。
两对AB/BA中位数为220.850→195.837 ns、221.619→194.067 ns；每进程4warm、31原始样本，未删离群点。
基线AA225.806/253.174 ns有12.1%漂移，新AA195.789/196.265 ns差0.24%，全部保留。
少一次hash独立确认，不能把这一有限trace精确倍率归为纯hash耗时或宣称外部胜出、普遍收益与跨进程CI。
QPC10 MHz，最短聚合1.1782 ms；实际波动与100 ns计时分辨率分开处理。
[摘要](concurrent-erase/summary.json)及[原始包](concurrent-erase/raw-results.zip)保留248个样本、old/new header overlays、
命令、hash与测试，只编相关probe/test TU并复用相同静态基础缓存，没有库/SDK全量重建。

## CPU / Vulkan Concat：相同顺序的原生平衡树

两边均按相邻ports配对，奇数项保留到下一层；单输入仍输出独立copy，不改Graph依赖或发布中间值。
CPU继续支持全部15种plain scalar、strided及empty ports；Vulkan保留原F32/F16/BF16/I32、dense/nonempty受理域。
CPU图元数据还按Concat端口数留出Cont/reshape/concat预算，避免70次引用同一SSA值时仅按unique values估算不足。
真实lowering census为284个tensor、283个graph nodes，原容量84、修后364；未故意执行越界native arena。

唯一计时形状为F32、64个各64 KiB端口、4 MiB输出，所有逻辑输入alias同一物理buffer。
两边完整1,048,576元素exact复制oracle在计时前后检查；cold建图/分配不计时。
Windows各两对AB/BA、无删样本：CPU公共Direct submit+wait为53.099→10.018 ms、52.894→10.056 ms；
Vulkan原生Region submit+wait为669.503→245.219 μs、738.594→321.750 μs。
保留Vulkan绝对漂移，不据两对数据计算CI或外推模型tokens/s。

CPU同图owned aligned payload请求126→20 MiB，累计copy输出130→24 MiB；前者不含caller、metadata或执行workspace。
Vulkan实际gallocr backing为11.25→7.75 MiB，不能把全部逻辑prefix payload相加称为常驻VRAM；copy工作仍从约130降至24 MiB。
平衡树为O(total bytes·log ports)，未宣称一次写完的最优线性实现。

CPU两版Direct全storage-bit/axes/empty/stride回归通过；候选公共CPU Graph的70重复ports→发布Concat→Neg通过，
断言一个region及ggml provider。Vulkan两版各3项通过，包含真实Region发布中间值与下游Neg，未额外运行公共Graph分区。
仅fresh相关provider/region/test TU并复用各自锁定native缓存；CPU与Vulkan使用不同pin，不混成同一后端对照。
[摘要](concat-cpu-vulkan/summary.json)和[源码/原始包](concat-cpu-vulkan/source-evidence.zip)保存全部命令、hash、早期harness失败及最终结果。

## FlatHash rolling keys：清理后留下增长余量

原自动grow只在live达到7/8时扩大；先erase再insert时live总是略低于该值，导致每次清完tombstone仅剩一两个empty-slot credit。
现在自动grow与try_prepare_insert统一：live达到3/4时加倍，否则原容量清理，保持7/8最大负载及显式try_reserve(count)合同。
这让一次O(capacity)清理之后至少留常数比例插入余量；没有改变探测、析构、搬移或OOM回滚。
容量加倍是明确空间代价，不能将较低负载下的运行时改善称为同容量胜出。
重分配时旧storage与新双倍storage暂时并存，也可能更早返回OOM；原表失败保留不变。

同一默认整数hash、固定32768轮erase最旧key/insert全新key、完整存活key/value oracle，无计时counter。
8192槽、7168 live：普通insert触发23736→1次重建，重hash的存活键170115912→7167；
显式prepare再insert同trace为23742→1次，170158914→7167个键；修后最终容量为16384，原为8192。
1024槽及离7/8阈值1/16/128项的原始计数均保留；稀疏control仍原容量清理。
47项FlatHashMap/Set及ConcurrentHashMap/Set测试通过，新增近满rolling trace同时覆盖普通insert和prepare。
这里只证明消除了重复整表工作，未计时或更新外部排名；[摘要](flat-churn/summary.json)和[源码/原始包](flat-churn/source-evidence.zip)记录范围。

## Apple普通attention：仅在合法窄域绕过unfused启发式

普通入口的锁定MLX在Q>8、D192/256默认物化scores与probabilities。现在仅GPU、F16/BF16、无显式mask、
Q/V depth相同且为192/256、非空batch/KV时强制已有fused kernel；causal还要求Q≤KV。
两种layout、GQA、bottom-right causal、非因果Q>KV及零/负scale保留，未新增用户配置或kernel。
当批F32、D512、decode、所有显式mask及空域保留原fallback；后续受限F32 D256 causal融合见下，其余域仍可能有平方scratch。

当时Mac16,12/M4上的默认F32 D192 tile需要40448 B线程组内存，超过该设备32768 B，未启用F32 force；后续更小的既有tile已接通，见下。
窄型融合的Boolean全屏蔽行也与原结果不同；所有显式mask静态保留原路径，不读取mask值来选择实现。
最终4项普通入口回归通过，使用原F32/F16/BF16预算；F16弱概率×V60000另按既有0.002检查。
原版正常exit1：一般F16与弱概率两项失败，未放宽预算迁就旧实现。
早期SIGSEGV来自测试下载未eval的transpose，修正为contiguous/eval后两版均正常退出；失败和最初源码均保留。

唯一空间AB各一个新进程：普通公共Context、F16 B1/H4/KV头2/D256、Q=KV1024 causal，完整1,048,576输出均满足0.002。
MLX peak减submit前active为12,591,104→2,097,152 B，总peak16,785,412→6,291,456 B；
caller输入4 MiB、输出2 MiB另列。因此新侧额外峰值在此形状仅为输出，而不是还有2 MiB私有scratch。
这是MLX allocator全局计数，未当成进程RSS、整机VRAM或严格provider-private字节。
空间测量后只补非空guard并重跑4项测试，非空实测路径相同；未测吞吐、全Apple/Graph roster或其它机器。
[摘要](apple-sdpa/summary.json)及[原始包](apple-sdpa/raw-results.zip)记录缓存来源、source/link maps、全部失败与最终结果；未重编SDK/metallib。

## HTTP2：修正peer工作量后保留大报文未解项

direct nghttp2 peer原来每轮从头扫描所有已完成response pairs，而两边仅同时激活两个stream。
现以单调游标跳过已退休的调度工作，仍保留全部header/DATA、反序response、两层流控ack、验证与会话/transport销毁。
Linux/WSL、GCC14.2、nghttp2 1.70.0，以一个Hyle worker和一个peer worker同时存在、一次仅激活一路，taskset固定逻辑CPU2。
两边计入新socketpair及H2 sessions、固定body、完整收发/清理与外部完成handoff；持久harness和body字符串生成在外。

修正peer后首轮四进程、每route九epoch：64×1 KiB中Hyle4.742–5.152 μs/stream、peer6.379–6.656 μs；
4×256 KiB强背压中Hyle197.838–205.041 μs、peer184.990–191.581 μs，四点慢5.59–8.27%。
同路径AA也观察到第二位置约5.04%/2.66%优势；这些是位置/暖态波动，不能作修正系数或称为时钟分辨率问题。
后续不同phase的baseline也出现peer绝对漂移及一个反向点，不把首轮差额外推为所有运行的稳定比例。

单次非timed计数显示双方inbound/outbound body copy各2 MiB，compaction均0；不是已证实的整body额外复制或平方compaction。
Hyle原有1032次try-read、其中709次WouldBlock。两个小候选分别去重复notify、yield后先重试write；
后者少192次空read但多63次空write，成功收发、全部字节、515次yield和1529次notify未少。
最后两对同头对照holdout为194.866→195.419 μs、197.019→194.119 μs，小body也未持续更快，故两个产品改动均已撤回。
14项协议/feature、强制epoll的duplex/关闭两项通过，新peer-close测试也在原产品baseline单独通过。

当前大body项未解决。两边协议/生命周期工作一致，但Hyle使用两个endpoint fibers及readiness，peer在一个手工loop推进两端；
下一步核该执行模型，不靠少职责或重复同一AB矩阵凑胜。后补资源探针确认backend为io_uring、实际3个用户线程均affinity[2]，
它不是正式运行现场快照，也未将kernel worker的syscall计数混成用户I/O次数。
[stage1摘要](http2/stage1-summary.json)/[raw](http2/stage1-raw.zip)及[stage2摘要](http2/stage2-summary.json)/[raw](http2/stage2-raw.zip)
保存全部epochs、候选、失败、计数与资源边界；只编相关TU和锁定nghttp2静态库，未全量构建Hyle。

进一步把peer的`read/write`改为与TcpStream一致的`recv(MSG_DONTWAIT)`、`send(MSG_DONTWAIT|MSG_NOSIGNAL)`（平台支持时）。
同一fresh HTTP2 object、固定四进程交错新旧fixture：旧holdout两点Hyle191.163/190.895、peer180.816/182.851 μs；
对齐后两点Hyle200.330/190.287、peer174.050/168.827 μs，仍慢15.10%/12.71%，小body仍更快。
Hyle控制也有漂移，不将其当作稳定普遍比例或声称旧差额已解释。[stage3摘要](http2/stage3-summary.json)/[raw](http2/stage3-raw.zip)保留所有epoch及顺序。
软件采样先因现有perf缺DWARF unwinder只能取得leaf，后用frame pointer取到部分收发/consume父调用；
缓存foundation未重编FP，Hyle2578样本有1662个无效栈尾，peer2016样本有685个，均截断保留。
这是定位线索，不能累加inclusive占比、用样本总量排名或将差额归因为fiber；[stage4摘要](http2/stage4-summary.json)/[raw](http2/stage4-raw.zip)记录限制，产品未改。
随后仅跑一次非timed完整holdout，以exe拦截现有`Fiber::resume`的PLT调用，未重编foundation。
520次dispatch中，515个adaptive yield有513个实际切到peer并见其成功I/O字节，仅2个直接调回自身；三次其它恢复原因未分类。
原payload/反序完成/cleanup全部通过；该事件计数排除“主要在无效yield自转”的猜测，不能量化时间或解决当前残差。
[stage5摘要](http2/stage5-summary.json)/[raw](http2/stage5-raw.zip)保留全部dispatch/yield及初次编译失败；没有追加计时或产品改动。

再将原生peer拆为两个`spawn_task` endpoint，仅跑一次非计时4×256 KiB episode，完整内容/反序完成/session销毁/socket关闭均通过。
本次同样520次恢复/515次yield，完整wire字节与stage5相同；普通IO不自动pin，实测只在尾部Event等待时临时pin。
与stage2旧Hyle计数相比，成功立即read/write次数同为323/514，但多130次read、64次write EAGAIN；不同轮的立即读取字节分布也有9 B差异。
因此它尚非逐操作相同的driver，不能量化纯scheduler成本或解释前述正式样本的慢点；没有追加计时或修改产品。
[stage6摘要](http2/stage6-summary.json)/[raw](http2/stage6-raw.zip)保留全部1060个phase和520个dispatch。

随后只修scratch两fiber driver的控制流：完成flush后立即重入，yield后的首次write只试一次，再进入原普通write循环。
单次非计时episode保留原协议/反序/销毁/关闭合同，empty read839→709、empty write322→258；
成功read/write仍323/514、恢复520/yield515。立即recv为2091990 B，async read8161 B，完整wire字节不变。
这解释raw诊断多出的194次空操作，不是计时收益或产品改动；[stage7摘要](http2/stage7-summary.json)/[raw](http2/stage7-raw.zip)。

再移除全部probe、拦截与热计数，固定AB/BA两个进程，各臂9个epoch，全部保留；相同单worker、两个spawn_task endpoint、IoContext及尾部Event。
每次计时均含socketpair/session、完整收发、join与cleanup；原payload/反序/关闭warmup通过，timed调用也保留完整协议验证。
AB中Hyle193.083、raw196.432 μs/stream；BA中Hyle197.469、raw199.225 μs/stream。
这两个点在相同执行模型下未复现Hyle较慢，但差值小且有波动，不证明等价、纯scheduler成本或正式manual-loop peer已赢。
raw自身headers/owner/storage及每次清零的两块128 KiB RawWire仍计入；不能用换执行模型的排名替代原手工loop慢点。
配置minEpochTime10ms，实际最短epoch9.479ms，未补样/筛选；复用旧foundation，未含本轮TimerQueue改动。
[stage8摘要](http2/stage8-summary.json)/[raw](http2/stage8-raw.zip)保留全部epoch、实际cache身份、命令和边界。

本批源码/日志/原始数据归档后，先写`test.ps1`再按精确路径、大小及hash清理可重建产物：
Windows Concat27文件197362252 B、Air普通SDPA8文件36649824 B；未删除源码、日志、数据或原有SDK/cache。

## Queue：先修测量工具的完成与数据合同

SPSC原来判断completed不等于2后，又独立读取wait值；最后worker若在两次load间完成，主线程可能wait(2)永久等待。
现只对同一个observed值判断和wait，醒后acquire-load。MPMC原来在producer最后发布前读到empty、随后看见all-done，
会用过早的empty判断退出；现确认all-done后进入局部drain，再次pop为空才结束。两项均为benchmark竞态，不当作库queue丢数据。
双方SPSC核sum；MPMC每consumer在局部累计count/sum/xor，join后汇总核1..N，无热共享计数器；moody enqueue失败不再静默少做工作。

两个当前TU直接编译通过，四路MPMC各4P/4C/200000消息、两路SPSC各1P/1C/262144消息的最小smoke通过。
未跑完整benchmark、TBB可选分支或其它平台，也未据此更新性能排名。原owner/steal/vector区域逐byte保留。
bounded Hyle/rigtorp的usable容量均1024，可按同一冷episode或同一warm边界继续比较；当前SPSC对手只是受控手写shape，不能称实际Folly成绩。
Sharded的P×1024总容量与单shard满策略，不同于moody初始1024却可扩容/implicit producer的合同；旧倍率不继承。
策略对齐后，空扫、重试及回收本身仍可能是真成本，不能拿算法不同自动豁免小输。
[摘要](queue-contract/summary.json)和[源码/命令包](queue-contract/raw.zip)保留实际peer header hash、编译与smoke范围。

随后用真实rigtorp固定ring核同一4P/4C、usable1024、200000个uint64消息的冷episode，包含分配/8线程创建、传输、join、核验和销毁。
Windows13900K固定8个实际高性能物理核各一个逻辑线程（mask0x5555），AB/BA四进程各11epoch×2episode；
Hyle/peer分别78.991/85.413、67.395/81.766、60.699/76.845、70.413/70.712 ns/传递item（push+pop），当前未复现Hyle更慢。
同路径Hyle/Hyle A/A为73.294/64.492，首臂MAPE12.2%；最后AB差仅0.424%，不声称总体胜出、等价或warm操作排名。
最短正式epoch23.45 ms、QPC100 ns，漂移不是仪器分辨率。实际线程补探针另进程峰值12（8 workers+caller之外3个未归因），不写成恰好9线程。
[摘要](queue-bounded/summary.json)/[原始包](queue-bounded/raw.zip)保留全部epochs、失败的旧Toolhelp观察及后续live-handle观察；未改queue实现。

其余queue再作只读合同复核，未确认新增当前输项或渐进复杂度缺陷。Sharded的P×1024容量、稳定producer索引，
不能直接对应Moody初始1024且可增长/implicit注册；其MAX_SUBQUEUE_SIZE还有block32准入/回收边界，不能当作精确ring满策略。
可先测不触full的相同应用trace，明确producer token、冷/热注册、线程/析构边界和单位；SPSC外部结论须使用真实peer及相同usable capacity。
当前single-thread512波次还须验证所有成功返回/payload，并统一首次注册是否计时；暂不靠旧倍率排名。
[只读审查](queue-contract/queue-policy-review.json)保留具体源行及8个文件身份，没有新增编译或计时。

## Vulkan普通causal SDPA：因果条件不再占矩阵

既有BSHD F32/F16、D64/128/256融合分支改用同一FullF32 shader的causal/cache字段，无新kernel或dtype收窄。
window=KV−1使所有合法query的左界仍为0，并裁掉全部位于未来的KV tiles；因果偏移仍为KV−Q+q。
普通固定算子显式设置runtime-request slot为−1，避免误认为batch request0；非因果、BF16、wide和Sink路径未改。

两版普通Region数值/受理回归均通过，候选3项通过；覆盖B2/GQA、padded batch pitch、非方形、Q1/3/5/17、
D64/128/256、正/零/负scale、两套真实buffer重绑定与旧结果保留，FP64参考及原F32/F16预算不变。
validate/update_batch_layout空请求也通过，未将私有paged测试替代普通入口验证。

唯一空间形状为F32 B1/H8/KV头2/D64、Q1025/KV1041，双方完整524800元素满足常量0.25的4e-6 oracle。
native gallocr容量6402176→0 B，caller四端口requested及native容量均5264384 B，输出2099200 B完整存在。
原版仅新增线性空间上界测试按预期失败并保留日志。NativeWork split-K、commands/descriptors、driver/pool仍另计，
不推断零total scratch、整机峰值或吞吐无退化；本批没有计时或公共Graph/default roster全构建。
[摘要](causal-mask/summary.json)及[源码/原始包](causal-mask/source-evidence.zip)保留既有shader字段依据、CPU defer原因、
source/link身份及全部定向结果。后续CPU普通路线使用独立HYCA，未切boxed；私有paged seed仍保留。
归档后按新的`test.ps1`精确清理本批5个可重建exe/obj，共90630052 B；原SDK/shader/source/log不动。

## Vulkan math attention：先对齐精度，再复用块空间

wide与Sink的QK/PV补显式F32 source，Q改为F32连续布局，避免原生非连续Q隐式转Half；普通融合/paged分支不变。
原版三个F32数值测试失败；修正后的未分块baseline与候选均通过原预算，未拿原版低精度作空间/速度对照。
大型math沿Q分块、完整KV只转换一次，每块保留mask原行与per-head sinks；输出用平衡concat保持多节点/发布中间值的原生依赖。
复制代价为O(output·log块数)，不是零拷贝；8 MiB阈值与64 MiB选行目标不是总VRAM上限，设备offset对齐仍须满足。
初版小KV/大输出反增空间，故仅在`(Q-blockRows)*KV > 2*Q*Dout`时分块，原小域四个对照容量均不增加，负点保留。

4项Region回归通过：F32/F16、B2/GQA、MLA不同输出depth、正零负scale、mask/sinks、跨块尾块、两次真实重绑和下游Neg。
唯一空间样本F32 Sink B1/H64/KV头1/D512、Q513/KV2049，同精度gallocr340527360→176687104 B（−48.1%）；
caller完整端口147077380 B另计。19个QK节点的地址union47724544 B证明复用，不能当作总device峰值。
无吞吐、公共generic Graph/default roster或全SDK重建结论；[摘要](vulkan-math-attention/summary.json)/[源码原始包](vulkan-math-attention/source-evidence.zip)保留原数值失败与初版空间负点。

## CPU Hadamard：去掉冷矩阵，保留原生FWHT

各pin原生FWHT仅将输入参数/元数据locals显式化，缩放、SIMD蝶形和行划分不变；真实unary CUSTOM仅接一个输入。
callback属于所选baseline/AVX2 program，私有ABI与入口同步升为2；没有W²伪存储或新增数学内核。
旧matrix hint及其`use_ref`矩阵参考回退保留，unary在两种reference设置下均执行FWHT。
CPU-only ffa4与混合456两个pin、各baseline/AVX2的4项测试均通过（共16次），覆盖三dtype、W1/4/128/4096、forward/inverse、两slot和重绑；
真实DLL导入/导出成功，故意加载旧v1 module正确返回ABI错误；依赖tree/license单项检查通过。

W4096、rows2、两独立进程仅测public prepare的successful unaligned global new请求：总量67110619→1692 B，最大67108903→368 B。
旧矩阵payload64 MiB，额外39 B为MSVC对齐请求；两臂完整8192输出误差均0。
CPU program DLL均在计数前预热，未声称首次加载成本、全部allocator流量、瞬时RSS峰值或热吞吐改善。
只重编每pin的ops.cpp baseline/AVX2及相关host TU，复用其它native对象；未configure、复制SDK、全量构建或验证非Windows。
[摘要](cpu-hadamard/summary.json)/[源码原始包](cpu-hadamard/source-results.zip)保留全部命令、link maps、初次计数修正及真实shared边界结果。

本批按先写`test.ps1`、核精确路径/size/SHA再删除的顺序，清理Windows Vulkan/queue8文件93754784 B、WSL HTTP2首批8文件5162160 B；
CPU Hadamard产物曾供后续CPU Sink闭包复用；完成后统一回收冗余，只保留两pin最小native对象/DLL，原SDK/cache、源码与日志数据保留。

## CPU math attention：官方allocator复用生命周期

只让实际分块的Sink与混合旧pin F32 math recipe使用官方gallocr；其余CPU flash/paged、causal mask和小域策略保留。
私有CPU buffer适配通过原AllocatorRef分配payload，真实容量的外部port描述器支持原有4 B偏移映射，不抬高caller对齐。
冷reserve准备backing/元数据，首次真实映射后alloc_graph绑定；固定图重绑刷新roots/views，execution先于buffer/metadata退休。
Q分块保留完整KV、scale/全Q因果偏移、原mask行与per-head sinks，平衡concat保持发布中间值和下游依赖，复制成本仍为O(output·log块数)。
既有32条可恢复OOM edits移到common资源并按CPU pin认证，Vulkan最终allocator原字节不变；公共fault fixture原字节移至runtime测试目录。

两个pin各5/5通过，原版各4/5，仅新增cold空间上界预期失败。FP64预算不变，覆盖Sink非零/全遮蔽mask、GQA/B2、布局/Dq≠Dv、
负/零scale、tail、两独立slots和重绑、发布值→Neg；CPU-only普通flash作为未分块控制也通过。
实际provider首次submit/重绑的AllocatorRef分配次数不增；独立native fixture的C malloc计数冷reserve12、first-bind仍12，逐冷失败回收通过。
后者未注入实际provider，不将两项观察合称全heap监控或并发slot压力测试。

冷Sink F32 B1/Q129/KV2049/H64/KV头1/D512：实际增量AllocatorRef私有请求156480184→48970792/48970760 B（两pin），包括arena/workspace/backing。
caller逻辑需求43266820 B未在这个冷测试中分配，也未执行此大形状；数值由前述单独场景覆盖。不是RSS、瞬时peak或吞吐数据。
[摘要](cpu-math-attention/summary.json)/[源码原始包](cpu-math-attention/source-evidence.zip)保留baseline失败、同pin host overlays、全部命令与root独立依赖锁检查。
没有configure、SDK复制或全native构建；后续普通多节点扩展见下。

## Apple长Q fallback：逻辑分块还须服从实际提交窗口

仅GPU、Q≥1024、静态确定原生会fallback且scores节省足够的域使用16行块；既有force融合和可能NAX的D256 causal无mask域优先保留。
直接复用原MLX math的dtype/GQA、scale→QK→mask→precise softmax→PV，不对子块调用会改变kernel选择的fast SDPA。
完整KV不裁短；因果行号为KV−全Q+块起点+局部行号。Bool全遮蔽保留finite-min后的uniform，Additive全−Inf保留原NaN。
depends同时序列化Q与mask生成源，仍为纯lazy graph，不在tracing中eval/synchronize；最后一次concatenate，增加图元数据与launch。

Air M4上新旧各6项回归通过，预算不变；覆盖非零QK、GQA、Q1025尾块、Q>KV因果全遮蔽、两布局/两batch、广播mask、双slot和重绑。
最终再加GPU-only策略guard及显式causal/mask准入整理后，仅相关1项在两侧重跑通过；实测GPU空间路径不变，未重新计时/测空间。
唯一空间形状F32 B1/H4/KV头2/D256、Q=KV2048、Bool dense mask：原额外MLX峰值83886080 B，初版64行块92274688 B（反增8 MiB）。
只作一次16行调整后为61669376 B（−26.49%）；总peak104857608→82641912 B。两侧完整2097152输出均为1.25、误差0。
输入QKV16 MiB、mask4 MiB；额外峰值包含8 MiB结果，submit前active差1008 B冷常量另列。不是严格provider-private、单块硬峰值或全设备VRAM。

MLX每command buffer及active-task窗口会同时保留多个块，环境设置也影响峰值；不外推吞吐、全Apple suite或其它硬件。
CPU stream、Q<1024及未知/新架构的可能NAX边界仍保留；Air的D256 causal遗漏已补，见下。
[摘要](apple-sdpa-blocks/summary.json)/[源码原始包](apple-sdpa-blocks/source-results.zip)保留64行负点、16行全结果、最终applied与measured源码差别；无SDK/metallib重建。

本轮全部按`test.ps1`核精确路径/size/SHA清理：Windows两批79文件145734659 B、WSL15文件9385152 B、Air8文件36820176 B，合计191939987 B（约183 MiB）。
当批保留两pin的4个最小CPU native对象/v2 DLL共4281057 B；后续已由新causal模块替换，见下。源码、压缩原始数据与原SDK/cache保留。

## 普通CPU多节点：不再保留所有中间payload

当批沿用同一GraphStorage，仅扩展到不含CacheEncode/PagedRead的普通多节点recipe；后续CacheEncode扩展见下，单节点及PagedRead策略保留。
INPUT、公开OUTPUT及view根的生命周期标记不变，常量仍由Program持有，NaN修复回调的原始source保留显式依赖。
两个pin各9/9通过；旧版仅新增cold空间上界按预期失败。覆盖发布第32步的中间结果、分支/常量、NaN回调、双slot重绑及OOM回收。
64个1 MiB Neg串联的实际增量AllocatorRef私有请求67049688→3086784/3086752 B，包括元数据/workspace/backing；
caller逻辑端口2 MiB未在该冷测试分配，大形状未执行；数值由独立小形状覆盖。不是RSS、全heap或吞吐数据。
[摘要](cpu-region-storage/summary.json)/[源码原始包](cpu-region-storage/source-evidence.zip)保存逐字tested/applied快照；后者仅排版差异。

## CPU普通causal：移除冷平方mask，保持原生执行路线

私有HYCA只表示普通bottom-right因果边界，独立于会切reference路线的boxed tag。
vector裁掉原本已在QK/PV前跳过的未来KV；tiled在原有固定scratch生成相同0/−Inf，不改prefill/split选择或Half PV。
Q1省去全零mask且不设tag。当批只改普通域；私有paged非boxed Q>1的后续扩展见下。
两个pin各selected5/5、baseline3/3通过，共16次最终test执行；含native显式mask逐元素精确对照与公共provider换绑/独立FP64控制。
这是同一executor内的精确对照，未声称不同ISA逐位相同；部分tile的未来NaN/Inf仍可能按原规则传播。
CPU-only pin的Q2048/KV16384/F16，各臂一次预热program后public prepare的successful unaligned global new请求：
baseline21次67112291 B、candidate20次3388 B，差67108903 B（64 MiB payload＋MSVC对齐39 B），最大请求降至736 B。
该冷测试未创建/执行invocation，未计时或测RSS/全allocator峰值。只重编相关host TU及两pin ops，未configure或全SDK构建。
[摘要](cpu-native-causal/summary.json)/[源码原始包](cpu-native-causal/raw.zip)记录实际module身份、全部命令/失败与资源单项检查。

## Apple D256 causal：按已知硬件能力补fallback遗漏

Air实查为applegpu_g16g，原生NAX不可用；锁定实现要求至少generation17，phone至少18。
仅通过公开device_info保守识别已知13–16代，让原本确定fallback的D256 causal域沿用现有Q16块；未知/新格式及异常仍保留原生选择。
非零QK的Q1025/KV1281、D256 FP64回归与小形状控制，两侧各2项通过，原rel6e-6+abs2e-6预算不变。
唯一Q=KV2048、QK0/V1.25大空间点，MLX额外peak83902464→60371584 B（−28.05%，包含8 MiB输出）；
但当批两侧max abs error均3.43323e-5，超过9.5e-6预算，**均为FAIL**。相同最大误差不证明逐元素等价；后续新融合路线同形状通过见下，不改写旧结果。
caller QKV16 MiB与冷常量另计；未测吞吐/整机VRAM，未改SDK/kernel或在tracing中eval。
[摘要](apple-nax-capability/summary.json)/[源码原始包](apple-nax-capability/source-results.zip)保留原生能力条件、全部失败与实际计数。

本批先写`test.ps1`再按精确路径/size/SHA清理Windows64文件62820427 B、Air9文件41857520 B、WSL3文件1634936 B，合计106312883 B（约101.4 MiB）。
旧4个CPUHad复用文件已由新4个native causal baseline对象/AVX2模块替换，保留4281627 B；原SDK/其它native缓存、源码、日志和压缩数据不动。

## CPU私有paged/ragged：去掉剩余causal冷mask

非boxed causal Q>1复用现有HYCA，物理pages只负责行寻址，因果边界读取逻辑KV长度；Q1与boxed优先级不变。
删除Program mask seed、外层vector、leaf及计费，也删除batch的旧mask下界，保留query下界和最终实际native workspace预算。
相同Q>1 layout owner复用、变化重编以及Q1动态更新的原合同不变；没有native资源、依赖锁、API或ABI变动。

两pin各6项通过，共12项；F16/BF16非线性page IDs/tails、GQA及Q65/63/1覆盖layout换绑、重编和原split资格。
独立FP64最大误差分别3.23e-5/6.09e-5，小于原abs预算.002/.008；不是全suite或更广非有限值证明。
Q2048/逻辑KV16384用两个physical pages及1024个重复refs，8 MiB预算可plan，实际known native1269488 B；少1 B在最终实际workspace检查拒绝。
旧版新增预算测试按预期失败，未将失败隐去。

同128 MiB预算、预建generic pool后各一次public prepare_batch，普通global new请求67111611→2684 B，少64 MiB seed及63 B vector开销；
known native68378352→1269488 B，恰少64 MiB。**当批generic float-workspace pool仍保留相同128 MiB**，后续已单独修复未使用的common scratch预留，见下；未测RSS/峰值/执行吞吐。
只重编相关host TUs，复用两pin现存HYCA模块；应用源码仅比实测快照少一行重复include，未为此重编。
[摘要](cpu-paged-causal/summary.json)/[源码原始包](cpu-paged-causal/raw.zip)保留全部失败、命令和实测/应用区别。

## TimerQueue：保留有效最小值，避免空闲轮询反复扫描

未到期且未被取消的cached min可直接复用；取消最小值后下次advance仍刷新一次，不改变公开next/cancel/rearm/到期交付语义。
一个私有dirty bool放入现有tail padding；Windows x64本次sizeof Queue1080 B、Entry48 B均未变，未外推其它平台布局。
相同N64/1024/16384个远期timer及128次idle advance，scratch仅对recompute_next访问节点计数：
旧N×128→新0；先取消最小值时旧(N−1)×128→新N−1，即只刷新一次，最大规模2097152次→0。
精确next/now/count、无提前交付与最终全部交付oracle通过，未测latency/指令数或外部peer排名。
无探针正式candidate的27项测试通过，含百万节点churn、radix、取消/重置、最大时间、已发布delivery list及death/lifetime。
只编header消费者和test TU；初次link漏SUBSYSTEM、随后只修link重试的原失败均保留，未重编scheduler或归因HTTP2收益。
[摘要](timer-idle/summary.json)/[源码原始包](timer-idle/source-results.zip)记录精确计数、源码身份与命令。

## Apple短Q/长KV：数学分块的负候选已撤回

尝试把既有Q16数学分块扩大到Q<1024但逻辑scores≥64 MiB的静态fallback域；保留原精度与完整KV，未改command-buffer设置。
F32 B1/H4/KV头2/D256、Q129/KV32768，QK0、预先确定的交替±1.25 V：两侧独立FP64均通过，max abs3.18e-12。
实际额外MLX peak68846084→69362180 B，反增516096 B；新增非零QK/控制回归两侧也通过，但内存目标失败，候选已从产品源码撤回。
输入QKV134746112 B、输出528384 B；不是单块硬峰值或吞吐证据。旧Q2048 uniform F32失败仍单独保留，不能被这个不同形状的通过取代。
[摘要](apple-short-q/summary.json)/[冻结候选与原始数据](apple-short-q/source-results.zip)保存尝试及负点；随后转查native F32融合，见下。

## Apple F32 D256 causal：复用可容纳的原生融合tile

保留NAX优先；仅普通非page、无array mask/sinks、F32、Dq=Dv256、causal且8<Q≤KV的Steel域选择现有BQ16/BK8/WM2/WN1模板。
QueryT=float、q8=false，QK/概率/在线softmax/PV仍为F32，没有Half转换、新shader或开启TF32。
Hyle仅在已知13–16代无NAX GPU上force此域，未知/新硬件继续原生选择。实际PSO静态线程组内存须≤实际device上限，超限受控报错。
Air M4实测28928≤32768 B；旧40,448 B记录是D192，不能拿它充作D256的原资源量。

3项定向测试通过，含非零QK、正/负scale、BHSD/BSHD/GQA/尾块、弱概率×大V及小域控制，原FP64 rel6e-6+abs2e-6不变。
原固定Q129/KV32768交替V样本，复用同源码/SDK的baseline而不重跑：额外MLX peak68846084→540672 B（−99.21%），包含528384 B输出。
完整132096输出最大误差3.18e-12→1.79e-12，均通过；caller QKV134746112 B不在这个额外量内，不是整机VRAM或吞吐。
同旧Q=KV2048、QK0/V1.25及原9.5e-6预算，新路线完整2097152输出max error0，额外peak8388608 B恰为输出。
旧数学路线两个FAIL保留；只说明新受限域这一个形状已通过，不外推所有F32精度。

仅5个host TUs/3links，复用原SDK/JIT与模板；原libmlx.a/metallib未重建或改写。
最终exact manifest只改native host selector，资源树/NOTICE锁同步；原materializer单host投影及两方CMake单项资源检查通过。
应用源码仅去探针输出、加说明和测试排版，实测/应用区别可查；初次编译与换行比较失败保留，未为排版再跑消费者。
[摘要](apple-steel-f32/summary.json)/[源码原始包](apple-steel-f32/source-results.zip)含实际PSO身份、maps、旧失败和未覆盖域；未测tokens/s或跨硬件吞吐。

## CPU CacheEncode：复用算术链中间值，保留失败退休

只移除普通多节点的encoders==0排除；callback全部source edges、call/state owner、invalid标记及worker退休次序不变。
公开input/output/view根仍固定到caller backing；单节点和动态PagedRead规则不变。没有新native资源、API或算法。
两pin共11项定向通过；新回归核64Neg→Q8/FP8、发布第31步中间结果、两slot各两套4字节偏移buffer重绑、原input及旧output保留、独立编码bytes。
既有Q8/FP8参考、invalid/recovery也通过。另一个真实Graph消费者在FFA pin核一native region、两slot、accepted InvalidArgument终态退休、拒绝重提、其它slot保留与新Invocation恢复。
默认CPU route及mixed Graph未重建/验证，不能外推任意cache图矩阵。

F32[256,1024]、64Neg→Q8，冷create_invocation实际增量AllocatorRef live请求68112224→3100744 B，包含backing/元数据/native workspace。
旧版新8 MiB上界断言按预期失败；同pin各一次，没有大形状执行或caller payload分配。
这是去掉随节点保留payload的Θ(N·M)项，图元数据仍为Θ(N)；不是RSS、全C/global-new、瞬时peak或吞吐数据。
[摘要](cpu-cache-encode/summary.json)/[源码原始包](cpu-cache-encode/raw.zip)保留同native身份、全部命令/失败及Graph消费者。

本批两次均先写根目录`test.ps1`，对全部目标核路径/size/SHA后清理；Windows62文件76785932 B、WSL5文件2669392 B、Air16文件64399080 B，
合计83文件143854404 B（约137.2 MiB）。期间重建的2个Python bytecode也逐次计入实际删除；原SDK/native缓存、源码、日志与压缩数据保留。
继续保留4个最小CPU native对象/DLL共4281627 B。没有全库configure、CI或为追逐名次追加计时矩阵。

## CPU batch：不为native owner重复预留上限scratch

LeasePool创建时即分配实际resource backing，CPU native plan/run却不使用common float/device-int/host-int views，实际native workspace另外受理、计费。
私有BatchProviderVTable追加uses_caller_workspace，默认true，CPU显式false；extension升为3以拒绝旧二进制table，公开API/options及CpuProgram ABI2不变。
core只为需要这些views的provider创建/登记三scratch，output/metadata/staging及owner/completion寿命保留；FlashInfer默认仍取得三块完整scratch。
不按provider名字分支，不减少传入budget或放宽最终native已知workspace检查。

两pin各4项CPU回归通过；default=true/false mock核实际backing及视图写入、旧extension拒绝、output跨控制owner释放后的寿命。
真实Graph batch wrapper也核转发false、一native region和256个外部output。未重测真实FlashInfer/CUDA capture，不能用mock替代其完整运行结论。
F16 Q2048/逻辑KV16384、两physical pages、128 MiB float/8 MiB int同预算，创建期实际backing请求151266920→271976 B，恰少144 MiB/slot。
同输入的native plan仍报告1269488 B，少1 B预算仍在最终actual check拒绝。默认64 MiB/8 MiB预算由同规则推得省80 MiB，未另采样。
旧allocation-policy core以当前ABI3头和同native table编译；计数resource只用于创建，实际upload/plan另用system_resource，未绕过physical-identity gate。
不是RSS/瞬时peak/大形状执行或模型tokens/s；两次测试fixture失败和修正均保留，仅重相关host TUs，4个native缓存原样复用。
[摘要](cpu-batch-workspace/summary.json)/[源码原始包](cpu-batch-workspace/raw.zip)记录边界及完整失败。

## Process：环境变更按名字索引，不反复扫描

公共校验用16项inline的借用string_view索引排序/相邻查重，POSIX继承entry只找一次separator再binary_search；原追加顺序不变。
Windows复用原UTF16 changes metadata排序，以原长度及CompareStringOrdinal(TRUE)比较，保留Unicode不敏感重复判断、特殊=C:entry解析和最终环境排序。
名字工作由M²+E·M降为MlogM+ElogM；默认inherit且零变更仍跳过环境构建，不能据旧“spawn线性”描述或此修复泛化所有进程启动。

Windows21/21、WSL23/23通过，含128变更、unset/replace、平台Unicode差异、重复拒绝/OOM后重试和已有管道/并发/回收。
隔离scratch源内计数M1024：公共名字比较523776→10639（MSVC）/9353（GCC）；native环境名比较Windows1444608→21942、POSIX919808→19621。
POSIX separator查找919808→1026；N16/N1024的旧新最终环境块均逐字节相同，Windows含特殊entry，未新增计时或裸spawn外部排名。
小N16公共校验仍零allocator调用；大N1024新增一次可失败索引分配，已测typed OOM/恢复。多项无效输入的错误消息优先级可能改变。
只编相关平台源/helper/test，复用实际toolchain/foundation；macOS未跑，不能因共用POSIX文件称已验证。
[摘要](process-env/summary.json)/[源码原始包](process-env/source-results.zip)保留全部命令、库身份、原始比较与环境块。

## HTTP2：先修多header视图，再保留性能残差

StoredHeader的FieldRef与HttpHeader是交错布局，原span在第二项便按错误stride读取。现在同一allocation中先放refs数组，再放对齐且连续的HttpHeader数组。
默认每块仍128×56=7168 B，header/trailer各一次分配，无新增Stream指针；部分OOM只析构已构造的数组，容量和backing不变。
新回归双向3条regular header、2条trailer，逐项迭代/find，并保留head view跨body/trailer和trailer view跨terminal read。
before同回归失败；candidate14/14及定向强制epoll通过，未修改正式bench/driver或追加计时。
只读复查未确认当前body热路径的渐进扫描：native user-data获取本身也查nghttp2 map，简单替换Hyle lookup不能凭空消掉一次hash。
explicit drain有retained-stream扫描，但正式holdout不走此路径，不能把它当作当前大body根因。旧manual-loop排名保留，修后完整吞吐未复测。
[摘要](http2/header-storage/summary.json)/[源码与只读依据](http2/header-storage/source-results.zip)记录空间布局和残差边界。

## io_uring direct槽位：先纠正native合同，再去掉受理扫描

此前的EINVAL源于direct accept误传SOCK_CLOEXEC；direct close的file_index还须编码为slot+1，零表示普通process fd。
依据[liburing accept说明](https://github.com/axboe/liburing/blob/master/man/io_uring_prep_accept.3#L182)和[原始编码实现](https://github.com/axboe/liburing/blob/master/src/include/liburing.h#L541)。
原cached测试跳过2项；仅修accept的中间版随即复现活连接recv EBADF、误关stdin。两处修正后25/25通过，不能再称该WSL内核不支持direct accept。

受理期改为free_file_slots计数及pending-close槽状态，completion索引直接保存close_slot，不再扫描/维护旧pending-close vector。
取消/失败不提前释放目标，原target CQE才更新计数；FILES_UPDATE拒绝负控制值/不可表示fd，未观察accept期间不覆写未知kernel slot。
关闭中的descriptor拒绝新recv/send；若kernel先复用该槽、新accept先被观察，则保留新owner，旧close CQE不能清掉新连接。
新owner可收发，但在旧close退休前拒绝同槽第二次close，不新增generation表、公开helper或用户设置。

同native内核/相同完整accept→close episode，baseline先含上述两处编码修正，之后只比较受理算法；没有计时。
table/batch为64/32、1024/64、4096/128，源内定义的accept扫描访问2048/65536/524288→0，close查重496/2016/8128→0。
两侧各形状的全部连接、CQE、inflight归零和槽复用通过；这不是CPU指令数、延迟或外部peer排名，map本身工作仍存在。
该计数发生在最终fd/乱序保护前，相同有效fd普通路径不变；没有为后续保护重跑矩阵。

受理初版26/26通过，负fd保护补后1项通过；最终版本只跑2项相关回归，未声称最终全26项重跑。
另一个单次scratch probe完整交换两个真实、未消费的close/accept CQE：原kernel顺序close→accept，模拟观察顺序accept→close。
核新连接在旧CQE前后payload/reply、pending第二close拒绝、其它free slot、最终两个slot复用、3个raw fd关闭及stdin保留。
这是受控CQ观察次序模拟，未声称自然复现竞态；原CQE的token/result/flags和全部探针源码保留。
只编相关host TUs，原io-watch/foundation库与SDK未覆盖；未跑Windows/macOS，未测吞吐或注入所有native失败/取消路径。
[摘要](io-direct-slots/summary.json)/[分阶段源码与原始结果](io-direct-slots/source-results.zip)记录全部阶段、资源边界和最终实测/应用区别。

本批先写`test.ps1`、对全部精确目标核路径/size/SHA后清理：Windows37文件37438277 B、WSL38文件15539424 B，
合计75文件52977701 B（约50.5 MiB）。源码、日志、原始包、原SDK/foundation及4个最小CPU native复用文件保留。

## CUDA长域F32：定位既有超差，不改实验预算

原B1/H8/Hkv2/D256、Q=KV2048/8192、QK0/V1.25的1e-5实验仍为FAIL。
单次native分阶段诊断直接输入原QK的精确0/−Inf结果，执行严格F32 softmax/PV；没有执行QK或完整Hyle provider，也未计时。
完整输出SHA与原Hyle旧新两侧一致。实际概率全行质量误差≤9.383e-8，对该常量V最多解释1.173e-7输出误差；
实际PV仍有3.433e-5/1.417e-4超差，因此主因是原生长PV归约舍入，未证实Q分块、dtype或接线引入额外误差。
原raw两个worst-row控制字段实际使用CPU RN(1/n)，并非重放所采native概率；[摘要](cuda-f32-long/summary.json)明确纠正标签，归因使用独立全行质量界。
F32 accumulation未承诺与长度无关的1e-5绝对界，不能把此探针与先前Half累加违约混为一类。
按KV分PV再合并可缩短归约，却增加launch、partial空间并改变舍入顺序；本轮仅源审，没有为满足探针自选预算新增产品路线。
[原始包](cuda-f32-long/raw.zip)保留全部失败、命令及身份；未改变预算、产品精度或声称模型精度通过。

## Queue：补实际peer与完整warm合同

正式ST512 benchmark改为检查每次push/pop、全部FIFO payload及最后empty，并将所有队列构造、32次显式warmup和析构移出callback。
Hyle Bounded1024对实际Moody隐式producer、512项不触full trace，两固定AB/BA各9epochs：
Hyle7.986/7.988、Moody9.937/9.974 ns/单次push或pop；一次额外empty检查计入时间，分母仍为1024。
只此trace当前未复现Hyle输；Moody1024不是硬容量，不能外推MPMC、full/backpressure、bulk或token策略。
实际caller仅一个queue操作线程，进程另有2个未归因辅助线程；首次错误thread-count断言的无epoch abort保留。
TBB/rigtorp分支仅源审接口，未编跑；[摘要](queue-st512/summary.json)/[原始包](queue-st512/source-results.zip)记录限制，未改队列产品。

另用真实clean Folly `65749da4` 的ProducerConsumerQueue及原header闭包，官方FOLLY_NO_CONFIG开关，替代shape推断。
Hyle1024/实际Folly1025物理槽均有1024usable；固定LP0生产、LP2消费、LP4协调，实际affinity/current CPU均读回。
每轮262144项逐值FIFO与checksum，persistent workers创建、一次warmup和join在外，协调notify/wait、传递与全部校验在内。
两固定AB/BA各9epochs：Hyle4.025/5.724、Folly4.280/7.176 ns/传递item（含push+pop，单位不同于ST）。
没有当前Hyle输项；epoch交叠及跨进程漂移明显，不能称稳定领先。全部36epochs保留，无AA、筛点或产品优化。
一小TU复用nanobench/foundation；原formal SPSC仍明确是本地shape。[摘要](queue-spsc-folly/summary.json)/[源码原始包](queue-spsc-folly/source-results.zip)。

再补Sharded4P/1C单一非full trace：每producer512项，Hyle4×1024；实际Moody4096初始storage、4个显式ProducerToken及ConsumerToken。
Moody用CannotAlloc的try_enqueue，源审128个32元素pool blocks及各token128项index足够，Hyle用对应producer索引和consumer cursor；不称内部内存成本相等。
完整cold callback含构造/token注册、5workers启动/gate、逐producer FIFO/count/sum/xor、join/empty及析构；6个actor固定不同P-core。
两固定AB/BA各9epochs，Hyle92.88/103.15、Moody108.74/111.21 ns/传递item；MAPE7.1–11.3%，只报未确认当前输项。
配置窗口12ms、实际最短9.043ms，全部保留；进程另有未归因辅助线程，不假称恰6线程，无AA/追加形状。
[摘要](queue-sharded/summary.json)/[源码原始包](queue-sharded/source-results.zip)保留容量合同与全部36epochs；未改队列产品或扩展正式benchmark框架。

## Apple F32 D192：同一受限域复用更小原生tile

普通无page/array mask/sinks、Dq=Dv192、Q>8的float Steel域复用既有BQ16/BK8/WM2/WN1；causal仍要求Q≤KV。
D256 guard literal、NAX早返、page/half及未知架构策略保留；Hyle只在已知13–16代GPU force。
实际PSO线程组内存21760 B≤Air设备32768 B，dispatch前仍核实际PSO与设备；没有新shader、Half转换或SDK/metallib重编。
两侧3项原FP64预算控制通过，覆盖BHSD/BSHD、GQA、causal/noncausal、正负scale/tail及弱概率×60000。
唯一空间点F32 B1/H4/Hkv2/D192 Q129/KV32768 causal、QK0/V交替±1.25：
额外MLX peak68583940→409600 B，包含396288 B输出；完整99072元素原FP64检查通过。
caller QKV约96.4 MiB在此前已分配；不是总VRAM、单块硬峰值或吞吐结论。
6个host TU/4links，单host补丁投影及root独立CMake资源检查通过；产品提交`706bfb748`。
[摘要](apple-steel-f32-d192/summary.json)/[源码原始包](apple-steel-f32-d192/source-results.zip)保留源身份、失败和tested→applied区别；未扩大跨硬件结论。

## HTTP2：其它流推进IO时保留body借用

公开HttpBodyChunkView允许借用到该reader下次操作；同session另一个stream的accept/写可以在这之前继续收DATA。
原接收数组在已ACK前缀可回收且尾部满时memmove未读数据，覆盖尚未ACK的借用视图；固定receive window不排除此情况。
新真实socket回归发送96 KiB非重复pattern，读并ACK前32 KiB、借用下一块，再accept仅在body发完后才发送的第二stream。
旧版仅borrow_preserved失败，完整body和两stream收发仍成功；不把它写成崩溃或性能反例。
改为原容量固定接收环，size始终含全部未ACK字节（包括borrow），callback只向free区最多两段copy，不再memmove；
read只发布环尾连续段，ACK/done/final trailers、reset/abandon流控credit按同一used数量处理。无新增字段、分配或窗口容量。
candidate单回归通过，default15/15与强制epoll15/15通过；同一测量/应用源码，提交`b8cfc70b6`。
只编HTTP2/test相关TU，原foundation/SDK保留；没有新吞吐测量，也未把此修复当作既有manual-loop差额根因。
[摘要](http2/body-borrow/summary.json)/[源码原始包](http2/body-borrow/source-results.zip)保留旧FAIL与全部协议结果。

## io_uring固定槽：连续填表不再重复扫描占用前缀

assign_fixed_slot原来每次从0搜索，连续填N个空表槽会访问N(N+1)/2项。
增加一个private最小候选：所有更低index在mirror中都非free；UPDATE成功后前进，失败不动，release/成功direct close才向下回退。
accept仅占用槽，-3/-4乱序仍reserved，不破坏下界；未观察accept期间继续禁止UPDATE，不增加freelist/分配或改变最小空槽顺序。
单N1024实际native assign/release均完成：循环访问524800→1024；两counter-only源变体只加owner-local计数，未计时或将其称为吞吐倍率。
正常产品TU另跑6项fixed/direct测试通过、无SKIP，新增8槽fill、native失败UPDATE、乱序holes及最低位复用。
只证明连续fill的二次项消除，不承诺任意hole被direct accept抢占等交错下全局摊销O(1)；提交`e427c3cc8`。
[摘要](io-fixed-slot-hint/summary.json)/[源码原始包](io-fixed-slot-hint/source-results.zip)保留正常与计数闭包身份，原SDK/foundation未改。

本批先写根目录test.ps1，精确核路径/size/SHA后清理Windows8、WSL14、Air10个可重建产物，共78820358 B（约75.17 MiB）。
源码、日志、数据、结果包、原SDK/cache及4个CPU原生ABI2文件4281627 B均保留；[执行结果](cleanup-apple-queue-http2.json)记录实际删除。

## Apple剩余空间域：源审后的具体边界

[源审与下一批设计](apple-attention-and-io-hint.json)记录当前分支和源身份，尚无这些新域的测量：

- 普通Q9–1023的确定math fallback仍可能保留完整scores：wide/不同Dv、D192/256显式mask、F32 D256非因果等。长Q未过既有收益guard也保留原路线，不自动认定输出主导的形状应分块。
- 已Q16分块域不再有单个完整scores对象，但Metal可同时保留多个块；不承诺单块物理峰值。
- 已融合域的每query/标量广播mask若末stride=0，原生可能复制展开mask；常见末KV连续padding mask不在此项。
- Q≤8的math最多8行，随KV线性，不能统称平方prefill；Apple typed入口拒绝CPU stream，不能列为合法CPU fallback或推断独立CPU route缺失。

下一批最小Sink方案是每Q16块直接复用相同原生fallback，只切Q及全局mask行，保留完整KV/sinks与F32计算；不新写softmax或更改归一化。
预先固定Q1025/KV2049/H64/D512空间点及原FP64预算，另需跨块/尾、非零QK/mask/sinks、全mask和重绑控制。
只有实际MLX峰值下降才可宣称空间改善；该源审阶段仅做设计，后续固定实验见下。

## 当前清单复核与HTTP2大body刷新

[有限源审](performance-status-refresh.json)重新对应64个factor（29个非compute），并直接复核12份历史IO summary的68个strata，未用旧PASS或5%门槛过滤。
只有旧HTTP2与WSL watch各两组CI整体低于1，其它负点跨1；这不等于各模块都已有当前peer测量。
当前TaskScope源码仍匹配已应用布局，修后三个mixed点1.028–1.053，并无当前稳定输项证据；watch修后0.1–1.0%负点仍未稳定。
该源审JSON早于以下新HTTP2运行，不能用其“待刷新”标签覆盖实际新数据。

只刷新原4×256 KiB holdout：current HTTP2/proactor与单main三TU，原peer与stage3逐字节相同，锁定nghttp2及其余foundation缓存保留。
两固定AB/BA、每臂9epochs、完整payload/反序/两层flow credit和session/socket生命周期校验：

| 顺序 | Hyle μs/stream | 手工loop peer μs/stream | Hyle点延迟差 |
|---|---:|---:|---:|
| AB | 200.3995 | 170.61375 | +17.458% |
| BA | 204.65425 | 177.00005 | +15.624% |

36epochs全保留，实际epoch约3.36–5.56ms；没有AA/补点/新shape，不虚构跨轮显著性或新旧源改动的因果收益。
Hyle为两个spawn_task+join，peer为单manual loop推进两端；这是同应用协议工作观察，不是纯算法/调度成本。
原core链接因cwd相对库路径失败，改为原cache绝对路径后重链，保留失败；ldd确认实际fresh core。
未重建全部HEAD、未采正式现场TID/backend轨迹，perf counters不可用未拿来归因。[摘要](http2/current-holdout/summary.json)/[源码原始包](http2/current-holdout/source-results.zip)。

## Apple Sink：拒绝没有实际空间收益的纯图分块

仅按已固定Q1025/KV2049/H64/D512 F32、Q/K/mask/sinks0、V逐KV交替±1.25，baseline/candidate各一个新进程，不更改输入或峰值边界。
两侧完整33587200元素均满足原FP64预算、max error5.679e-13；caller151142660 B另计，输出均134348800 B。
额外MLX峰值却从1344307200→1481031680 B（+10.17%），因此该Q16候选未应用，原问题保留。
两侧最终2/2数值/寿命控制通过，含非零QK、逐行mask/头sink、全masked行、跨块尾、两invocation与5port重绑及旧native output保活。
初版测试在尚持旧OwnedResult/ResultLease时复用slot，触发预期的FailedPrecondition，被unchecked value转成异常；两侧失败均保留。
仅修正确lease使用并重跑同2tests，未为该harness问题重测space；原F32预算不变，最大误差4.338e-7。
MLX Metal eval会把primitive输入引用保到CB completion，depends只排序，不退休。baseline各主要逻辑数组尺寸之和接近实峰，candidate多一份output frontier也与反增量相容；
没有per-allocation/CB轨迹，不能声称已唯一归因或65块全同时驻留，更不能用逻辑16行承诺物理单块空间。
只编host小闭包，原SDK/metallib未改；[摘要](apple-sink/summary.json)/[冻结候选与原始包](apple-sink/source-results.zip)保存负点和harness修复前后证据。

## Owner/steal benchmark：先补完整合同

两正式源补push/pop返回、精确FIFO/LIFO、single-steal Retry直到Empty，以及batch逐值去重/count/sum/xor/最终空检查。
两端构造/析构在外、各200个checked warm waves；Rust保留400×50原设置并输出所有未排序epoch，median只排序副本。
Crossbeam实际registry0.8.6默认batch cap32，Hyle取剩余一半；Worker::pop还会缩容，下波再grow，不能写成warm后无分配或完全同工。
两者steal trace都在单线程且时间包含owner fill，分母512个传递item；owner分母才是1024次push/pop，不称并发偷取吞吐。
MSVC一TU与Rust offline/locked本地3crates编译，四个功能smoke各200waves通过，均在计时入口前返回，无新性能排名。
队列产品未改，提交`1a601c8ee`；Cargo.toml另只纠正优化说明评论，设置未变。[摘要](queue-owner-steal/summary.json)/[源码原始包](queue-owner-steal/source-results.zip)。

## HTTP2固定body发送：删除私有空协程层

send_submitted_fixed零co_await，只同步驱动已有stackful drive_until，却再建一个Task；两个公开父Task又等待它。
该private helper现返回普通Status，保留AllocatorRef向下传递、惰性公开父Task、body借用、取消/失败和pin清理/退休顺序。
旧GCC14 O3 object有真实256B/alignment16的内部frame factory；新object没有该private actor/destroy/frame，4请求+4响应正常路径不再经过该8个frame。
这不表示所有其它分配或OOM点相同；少掉的是多余资源需求，不是协议工作。未做新alloc probe或计时，不能据此称前述15.6–17.5%已改善。
当前default15/15与forced epoll15/15通过；实测/应用仅缩进及return合行差，未为排版重测，提交`ba85438de`。
[摘要](http2/fixed-send-frame/summary.json)/[原始包](http2/fixed-send-frame/source-results.zip)保留前后源、objects静态证据及全部协议结果。
[执行策略设计](http2-execution-control-design.json)还纠正旧线索：canonical manual同样有两块128KiB RawWire，不能将其清零成本写成two-fiber独有。
拟用同caller-owned episode/setup/teardown和callback，仅切1个Task双端loop或2个endpoint Task；它衡量驱动策略组合，不能称纯fcontext成本或替换正式peer。

## Apple Sink compact：省空间但严重退速，不应用

现有Steel模板已有BQ8/BK8/WM1/WN1的零padding compact分支；仅在scratch为exact F32 Sink域接通force admission与host tile。
没有新shader、Half query/probability或SDK重编。实际PSO为32768 B，等于Air设备上限，仍检查真实PSO和线程数。
两侧原2项数值/寿命控制均通过，覆盖非零QK、全局mask、逐头sinks、全mask行、尾部、重绑和旧输出；H128/Q17此候选也走融合，不能称未改fallback对照。
原固定Q1025/KV2049/H64/D512空间点，各一个进程、完整33587200元素检查通过；额外MLX峰值1344307200→134348800 B（约−90%），后者仅为输出大小。

随后同一consumer、同输入、prepare/bind及销毁在外，移除PSO打印后固定四进程ABBA：每进程2次warm、计时前后完整FP64 oracle、9次submit至checked wait。

| 进程顺序 | 路线 | 9样本中位数 ms |
|---|---|---:|
| 1 | baseline | 166.13 |
| 2 | compact | 1166.13 |
| 3 | compact | 1170.60 |
| 4 | baseline | 167.55 |

两个顺序均约7倍慢，36样本及8次完整oracle全保留，不以空间收益掩盖时间退步，候选不应用。
AC77%前后不变、pmset无已记录thermal/performance warning；未采实际GPU频率/功耗，未调频、AA、重试或换shape。
源审另指出scale/sink先乘log2(e)的极大finite F32边界，未实测；温和测试通过不能扩为完整F32值域保证。
[摘要](apple-sink-compact/summary.json)/[源码与原始结果](apple-sink-compact/source-results.zip)保留带/不带诊断的host差异、三TU身份、负时间结果；产品与资源manifest未改。

## HTTP2执行策略：共用episode后仍有组合差异

scratch共用caller-owned session/socket/两块RawWire、callbacks及setup/teardown，只切1个manual Task或2个endpoint Task。
两臂各一次untimed完整header/body/反序及fd关闭检查通过；固定AB/BA各9epochs：

| 顺序 | manual μs/stream | dual μs/stream | dual点延迟差 |
|---|---:|---:|---:|
| AB | 165.195735 | 185.249346 | +12.139% |
| BA | 165.476750 | 184.921817 | +11.751% |

36epochs全保留；两向各1 MiB payload与consume_stream/connection职责检查通过，但S→C wire分别1050092/1050027 B，差65 B。
终端pending frames、wire索引和frame类别未采集，两臂终止条件也不同，不能断言差异来自WINDOW_UPDATE或将其当完全同wire控制。
这只支持Task数量、推进cadence、yield/readiness及终止策略的组合影响，不是纯fcontext成本；未调用公开Hyle HTTP2 API，不能从另一轮15.6–17.5%直接减去此差额归因。

初次smoke在episode开始前SIGSEGV：共用RawHarness先构造worker线程，线程可能读取尚未初始化的workload/pending/stopping字段。
scratch只移动worker成员到状态之后，随后上述完整运行通过；canonical同处应用相同顺序修复（仅注释不同），提交`0273568b6`。
这是benchmark夹具修复，不是HTTP2产品修复；没有为它重跑完整canonical benchmark或产品测试。
[摘要](http2/common-episode/summary.json)/[源码、gdb及原始epochs](http2/common-episode/source-results.zip)保留初始失败、必要fixture差异及全部工作边界。

本批先写根目录`test.ps1`，对三系统全部精确目标完成路径/size/SHA预检，再逐文件重检删除38项，逻辑大小合计82599370 B（约78.77 MiB）。
保留下一步可复用的4个Apple小对象、2个WSL当前IO对象/库、Cargo依赖cache及4个CPU native文件，合计49项10829962 B；源码、日志、数据、原始包和SDK保留。
Cargo可能hardlink，因此逻辑大小不等于实测物理回收量。[执行结果](cleanup-sink-http2-owner.json)记录实际删除及保留数。
