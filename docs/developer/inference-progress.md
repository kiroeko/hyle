# 基础推理、多会话与多任务进度

更新：2026-10-26 06:00（北京时间）。整体粗估 **86%**。四请求普通随机采样decode约＋10.8%、综合＋9.8%，已合入；单请求／MTP正式CUDA对照未刷新，原差距仍保留。CPU共享池与原生有限轮询已合入；本次调用线程配置另窗decode＋6.54%、综合＋3.36%，采用为后续同负载配置，不是新库实现。Graph自身未见大热点；host直接IO虽通过正确性检查，加载却慢约12.1%，已关闭。继续核原生图聚合接点，不把调用次数当收益。

百分比估计剩余开发工作量。范围仍为当前硬件可做的单机、多系统、CPU／单GPU／CPU＋GPU普通推理、多会话和多请求底座；发现新缺项继续补。功能完成不代表未运行的制品／平台或完整大模型质量通过。MTP／多模态不重开功能穷尽，性能包括MTP对llama.cpp的差距。

## 任务树

- 底层功能盘点与补齐：100%
  - 模型／量化／算子与融合接线：100%
  - prefill／decode／精度／上下文与完整状态：100%
  - 多会话隔离／前缀复用／淘汰恢复：100%
  - 多请求接纳／取消／错误隔离／背压／公平性：100%
  - 可控logits的自然提交前边界：100%
- 共同架构复审与调整：100%
  - 全链职责／所有权／配置合同：100%
  - 模型／codec／后端扩展的改动范围：100%
  - 已确认接缝的共同机制调整：100%
- 性能设施复用与完整消费者优化：约47%
  - Hyle内存／容器／调度IO／图重放与缓存复用：约48%
  - 本地参考引擎思路与可证伪候选：约65%
  - MTP差距归因与执行优化：约90%
  - 已确认热点实现与最新完整四项对照：约30%
- 完成前遗漏复审与收尾：0%

## 正在做

| 任务 | 所属 | 简述 | 进度 |
|---|---|---|---|
| CPU原生图聚合接点核对 | 性能设施复用 | 已核原矩阵图及参考源码，继续看已有ggml region发现／编译机制能否保精度存储点减少程序边界 | 约20% |

## 本次完成与剩余顺序

- 功能／共同架构：本轮发现的可实现缺项已补齐；[功能复查](../evidence/research/inference-functional-review-2026-10-25.json)、[架构复查](../evidence/research/inference-architecture-closure-2026-10-25.json)及[本次独立遗漏复审](../evidence/research/inference-closure-omissions-2026-10-26.json)保留有限运行和模型／平台边界，不能据此宣布整个goal完成。
- 配置接线：已补[CPU预算工厂转发](../evidence/research/inference-cpu-options-forwarding-2026-10-26.json)及[共同冷replay选择](../evidence/research/inference-replay-policy-consistency-2026-10-26.json)。默认行为不变，有限编译通过；未据此声明完整运行或吞吐收益。
- 当前基线：已按实际依赖完成[107→108对象刷新](../evidence/research/inference-current-source-build-2026-10-26.json)，完整32进程结果见下表。后续配置改动未追加全窗，旧样本不改名、不混算。
- 性能取舍：同步报告和hidden输出已合入；宽Lt、typed MMF及K2辅助强制捕获均无稳定完整收益，关闭且不补样。数字、源身份和取舍见[性能资料](performance/decode.md)。
- 准备归因：[唯一真实诊断](../evidence/performance/inference-model-prepare-attribution-2026-10-26.json)正常结束、非时间字段0差异。66次decode的布局／batch／replay为8.441／16.328／60.361ms；target原生计划366复用／30新建。整个batch桶仅占本次decode约1.64%，不能据此期待大幅收益。
- GDN取舍：[单行为＋8进程筛选](../evidence/performance/mtp-fixed-gdn-work-io-screen-2026-10-26.json)完整工作相同、录制27→22，但复制成本未换来净收益，关闭、不补样、不合入；未追加字节／故障检查。静态端口已有复用，剩余遍历量级未知；不同down形状也尚无足够收益依据，不盲开小项。
- 随机采样：[无过滤R2检查](../evidence/research/inference-batched-unfiltered-sampling-2026-10-26.json)CPU／CUDA各7项及小Model通过；[三臂正式对照](../evidence/performance/inference-random-four-request-screen-2026-10-26.json)全程工作与输出相同，批量路径有净收益，已合入。[Hybrid功能检查](../evidence/research/inference-sampling-hybrid-replay-2026-10-26.json)虽通过，完整吞吐却下降，关闭、不合入或补样。均不计作当前greedy MTP收益。
- CPU：[首对照](../evidence/performance/inference-cpu-initial-pair-2026-10-26.json)ISA偏差已用同版本15个CPU TU修正；[AVX2完整短窗](../evidence/performance/inference-cpu-avx2-window-2026-10-26.json)确认decode仍落后且有3428个OS线程。[Context共享池修复与新三臂对照](../evidence/performance/inference-cpu-context-team-2026-10-26.json)已完成：每槽保私有plan／workspace，共享团队保活、完整native compute独占，9项runtime及真实模型工作一致；结果见下表。修复已合入，未重跑其它平台。
- CPU归因：等待策略调整前的[唯一真实诊断](../evidence/performance/inference-cpu-native-attribution-2026-10-26.json)9104记录完整、0溢出、非时间工作0差异。decode每输出native84.460ms、其中矩阵图75.305ms（本次墙时77.94%）；wide投影33.032ms、tied head19.409ms为不相交组。锁接纳全15轮仅0.642ms；native外每输出12.162ms仍含其它provider／host／仪器，未归因。未测参考原生耗时，不能据此认定差距主因或可回收收益。
- CPU等待策略：[新两臂筛选](../evidence/performance/inference-cpu-polling-screen-2026-10-26.json)8新进程全成功、工作0差异，9项runtime及先行正常行为也通过，保原生默认有限轮询已采用。BF16点积／MUL_MAT相关核心与参考相同，不能归因于旧版CPU数学；参考每整图临时创建默认pool，并非跨step常驻自旋。
- CPU Graph归因：[单TU诊断](../evidence/performance/inference-cpu-graph-submission-attribution-2026-10-26.json)10944步及64个scope完整、0溢出、非时间工作0差异。同run decode墙时约99.05%在provider调用包络内，Graph其余每输出约0.919ms且含仪器；未见支撑大改Graph提交机制的量级，不跨run减去旧native耗时。
- CPU调用配置：[新两臂筛选](../evidence/performance/inference-cpu-caller-executor-screen-2026-10-26.json)六计时＋两独立内存进程全成功、工作0差异。原生4与ATen4不变，去掉测试入口的worker-only交接后decode＋6.54%、综合＋3.36%。默认executor本来为空，未改库或绕过用户回调；通用provider分工可能改变，不能把收益全算为纯交接耗时。后续该延迟负载采用调用线程配置。
- CPU矩阵源码：[有限复核](../evidence/research/inference-cpu-matmul-granularity-review-2026-10-26.json)未发现审查路径大权重重复转型／pack。Hyle中间BF16存储点与参考保留F32结果是实际数值合同差异；187矩阵／569原生小图来自旧诊断，不能由次数预测吞吐或把旧33ms包络当转换成本。现有region聚合机制仍在核对，未改精度。
- 载权取舍：[直接host IO筛选](../evidence/performance/inference-cpu-host-direct-io-screen-2026-10-26.json)五个host TU、17项原suite及独立真实行为通过，但新六计时＋两独立内存窗口加载0.895→1.004秒（慢12.12%）、startup到ready1.003→1.117秒（慢11.27%），三样本范围不重叠。prefill43.331→43.359、decode11.416→11.419、综合3.283→3.290 tok/s均范围重叠；工作集各约5.320GiB、线程峰17，GPU缺测。627次／约3.76GB显式复制虽消除，staging与backing预算不变；关闭、不合入、不补样，不猜慢因。合法只读映射导入仍为设计，未计收益。
- 剩余顺序：先核CPU原生图边界及参考同项成本，再按有量级依据的CPU／CUDA普通及MTP热点推进；只读映射加载候选与混合FFN放置另列。当前greedy MTP仍未追平。

## 最新完整性能对照

2026-10-26完整 CUDA 窗口（正式engine四TU及自有依赖刷新，image `2a94f0b2`；参考仍为`433e7051`，保留其源码／补丁闭包，不称最新上游）。四组各三个独立进程测速，内存另采。各格为 Hyle／llama.cpp，速度单位 tok/s。

| 工况 | prefill | decode | 综合 | dedicated 显存采样峰 GiB |
|---|---:|---:|---:|---:|
| 单轮普通 | 2094.269／1578.977 | 98.474／102.141 | 89.724／89.680 | 3.888／3.728 |
| 单轮 MTP | 2229.803／1462.412 | 122.475／134.439 | 109.691／111.814 | 4.048／3.896 |
| 两轮普通 | 2894.643／2453.351 | 98.605／102.124 | 92.351／94.137 | 4.023／3.728 |
| 两轮 MTP | 2797.193／2420.500 | 127.091／140.805 | 116.165／125.083 | 4.191／3.896 |

32进程均正常结束、各臂输出和工作稳定。跨引擎续文不同，不能称逐token工作相同；不与旧窗口混样或归因跨次绝对速度变化。显存为独立旁采下界；加载、RAM WS／commit、范围及公平性边界见[新窗口证据](../evidence/performance/inference-current-window-2026-10-26.json)与[完整性能资料](performance/decode.md)。不代签其它制品、合批或多用户吞吐；每次完整对照后更新这里。

本轮另测Hyle四请求普通随机生成，N4／R4、每请求两轮P154＋64、无top-k／top-p过滤、自然EOS；不含llama.cpp。每臂三个交错测速及一个独立内存进程，单位同上。

| Hyle配置 | prefill | decode合计 | 综合合计 | dedicated显存采样峰GiB |
|---|---:|---:|---:|---:|
| 原Commands | 6036.674 | 221.435 | 206.121 | 4.417 |
| 仅批量采样，已采用 | 5944.634 | 245.271 | 226.373 | 4.401 |
| 仅Hybrid，已拒绝 | 5674.337 | 214.220 | 199.074 | 4.394 |

批量采样decode＋10.76%、综合＋9.83%，三次范围不重叠；prefill−1.52%但范围重叠。所有输出、RNG、ledger、调度与工作相同。显存各仅一次采样，不将16MiB差值签为确定节省；三点不代表统计显著性或所有负载。首次清单转义错误在加载前失败，修正后的三行为及12筛选进程均成功，原失败保留。

CPU资源修复的新三臂短窗：Windows i9-13900K、原生线程T4、P154＋16、C256、N1／R1、greedy、无MTP；参考使用同版本AVX2／FMA／F16C。各臂三个交错测速进程，内存各另一个进程，不与此前窗口混样。

| CPU配置 | prefill | decode | 综合 | 工作集采样峰GiB | OS线程峰 |
|---|---:|---:|---:|---:|---:|
| Hyle原每槽池 | 42.584 | 10.061 | 3.131 | 6.548 | 3428 |
| Hyle Context共享池，poll0基线 | 43.159 | 10.323 | 3.190 | 5.320 | 17 |
| llama.cpp AVX2 | 42.981 | 12.042 | 3.312 | 3.828 | 16 |

共享池decode＋2.60%、综合＋1.91%，三次范围不重叠；仍比参考decode慢约14.28%。12进程全部成功，各臂完整非时间报告稳定，三臂16输出及计算行数相同。工作集下降约1.32GB；private commit分别7.350／6.065／0.607GiB，含映射／私有内存差异，不能直接当成权重大小；线程峰不等于忙线程数。Hyle两臂GPU计数器无PID实例，显存值为缺测，不记零。短窗不代签长上下文、多请求、其它系统或质量；原始数据与边界见[共享池证据](../evidence/performance/inference-cpu-context-team-2026-10-26.json)。

随后只改共享团队等待策略，另跑六计时＋两独立内存进程；不含参考、不混入上表。其余CPU工况相同，整进程CPU时间包含加载和退出。

| CPU共享池等待策略 | prefill | decode | 综合 | 工作集采样峰GiB | 整进程CPU时间中位秒 |
|---|---:|---:|---:|---:|---:|
| 强制休眠poll0 | 43.068 | 10.382 | 3.191 | 5.320 | 21.109 |
| 原生默认有限轮询，已采用 | 43.203 | 10.612 | 3.214 | 5.320 | 22.031 |

decode＋2.21%、综合＋0.71%，三次范围不重叠；prefill＋0.31%但范围重叠。CPU时间＋4.37%，不是decode CPU时间或能耗。两臂OS线程峰均17；内存各一次采样，不将微小差值当节省，GPU仍缺测。所有输出和非时间工作相同，未因此刷新对参考的差距或CUDA结果，也未把两次CPU收益相加。

本次再单独比较执行器配置，原生4／ATen4不变，保留原Scheduler4的创建和寿命；只替换测试入口对象。六计时＋两独立内存进程全成功，不与上面窗口拼样，也没有参考进程。

| CPU执行器配置 | prefill | decode | 综合 | 工作集采样峰GiB | 整进程CPU时间中位秒 |
|---|---:|---:|---:|---:|---:|
| 原借用Scheduler，concurrency4 | 42.247 | 10.842 | 3.182 | 5.320 | 22.328 |
| 调用线程，concurrency1 | 43.362 | 11.551 | 3.289 | 5.320 | 21.703 |

decode＋6.54%、综合＋3.36%，三次范围不重叠；prefill＋2.64%但范围重叠。两臂线程峰仍17，内存微差不称节省，GPU缺测。16输出、169 target行及完整非时间报告相同；未检查全部中间值。它是现有合法配置的选择，不是库算法或默认值改动，不据此更新对llama.cpp的差距。
