# 基础推理、多会话与多任务进度

更新：2026-10-26 10:09（北京时间）。整体粗估 **88%**。**尚未找到已证实可大幅回收的单项。** 正式普通B2仍慢9.75%；配对诊断的135.32ms根耗时差中GPU活动并集仅差12.31ms，重点仍在准备／提交／发布衔接。余量含驱动、等待和仪器，不能直接当作可删CPU时间。页视图候选新8进程未取得净收益，已撤回；不补样或扩大该候选。结果发布扫描已采用但仅＋0.16%且范围重叠，不称稳定提速。单请求／MTP正式表不变。

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
- 性能设施复用与完整消费者优化：约52%
  - Hyle内存／容器／调度IO／图重放与缓存复用：约54%
  - 本地参考引擎思路与可证伪候选：约67%
  - MTP差距归因与执行优化：约90%
  - 已确认热点实现与最新完整四项对照：约40%
- 完成前遗漏复审与收尾：约10%

## 正在做

| 任务 | 所属 | 简述 | 进度 |
|---|---|---|---|
| 双请求主机准备／提交耗时定位 | 性能设施复用 | 页视图候选关闭；replay源审暂未找到大额重复调用，下一定位点为稳态batch准备桶，不能把录制时间全算热开销 | 约35% |

## 本次完成与剩余顺序

- 功能／共同架构：本轮发现的可实现缺项已补齐；[功能复查](../evidence/research/inference-functional-review-2026-10-25.json)、[架构复查](../evidence/research/inference-architecture-closure-2026-10-25.json)及[本次独立遗漏复审](../evidence/research/inference-closure-omissions-2026-10-26.json)保留有限运行和模型／平台边界，不能据此宣布整个goal完成。
- 生命周期复查：[两条当前源码链](../evidence/research/inference-lifecycle-followup-2026-10-26.json)未新增确定缺口，涵盖full-hit frontier失败重试／真实退休和双预算拒绝／跨后端Snapshot／整批发布；8份当前源码身份已核。仅源码复查，未新增故障执行，不代签全平台或全部生命周期。
- 当前CUDA／MTP归因：[每引擎一个新诊断](../evidence/performance/inference-cuda-mtp-current-attribution-2026-10-26.json)均正常退出，完整非时间工作与各自参考相同；GPU并集、父子区间及correlation独立复核未见阻断问题。Hyle为66个合并episode，参考为65个target＋65个draft调用，实际target行441／438、draft行434／435，不能当相同计算量。Hyle27次录制包含13次Instantiate及14次成功Update，没有Update失败回退。21个有录制decode的replay准备累计71.48ms，其中30.85ms未被可见API区间覆盖；尚非纯CPU或可回收时间。两边各轮根间gap均约1.5–2.0ms，不能从日志／调度间隙期待大幅收益。正式四项表不变。
- 录制主机路径：[两Graph TU私有诊断](../evidence/performance/inference-cuda-recording-host-attribution-2026-10-26.json)严格编链、唯一模型运行及完整工作门通过。dispatch43.909ms中42.804ms位于实际step内，外部循环／仪器余量仅1.105ms；GGML桥型未覆盖18.715ms仍含原生工作和仪器，未证明可删。结束本批N1录制调查，不增加Graph调度／重绑框架。首编译的私有Windows宏冲突、隔离修正及失败日志保留；归档后按test.ps1精确清理5个可重建产物约147MB。
- 双请求：[新8进程完整窗口](../evidence/performance/inference-b2-current-window-2026-10-26.json)已完成并独立复算；198输出、1728输入行及99个R2 episode在两端对齐，所有原始非时间报告同臂稳定。decode仍慢9.75%，不能用旧17.65%跨窗差值归因优化；四请求随机采样收益也不代签该B2旅程。
- 双请求归因：[各一次新诊断](../evidence/performance/inference-b2-current-attribution-2026-10-26.json)工作核对及独立区间复算完成；96个decode墙时差135.317ms、GPU并集差12.311ms。优先查主机准备／提交／结果回收；余量不是已证明可删除CPU时间。每步72次状态retain的重复writer发布扫描已消除，[8进程筛选](../evidence/performance/inference-result-publication-screen-2026-10-26.json)未证稳定提速，不再据此项期待主要收益。归档后按test.ps1精确清理诊断二进制约196MB。
- 阶段细分：[一次12子阶段诊断](../evidence/performance/inference-b2-host-phases-2026-10-26.json)记录完整99个物理根，但工具截断stdout，完整报告／198个ID门未完成，不写通过。页导出21.130ms含take_append／ready／reframe，仅占该次decode根约1.63%。[离线拆分与源码复查](../evidence/research/inference-b2-host-next-2026-10-26.json)确认replay60.961ms中6个录制根占37.772ms，90个非录制根仅23.189ms；后者每根约258µs、占该次decode总根墙时约1.79%，也非已证实可删成本。已有绑定／录图复用未发现新的大额重复原生调用，不新增缓存或重绑框架；稳态batch准备17.362ms尚待具体归因。参考同样通过kernel搬运状态，不能把其0个memcpy事件当0搬运。
- 配置接线：已补[CPU预算工厂转发](../evidence/research/inference-cpu-options-forwarding-2026-10-26.json)及[共同冷replay选择](../evidence/research/inference-replay-policy-consistency-2026-10-26.json)。默认行为不变，有限编译通过；未据此声明完整运行或吞吐收益。
- 当前基线：已按实际依赖完成[107→108对象刷新](../evidence/research/inference-current-source-build-2026-10-26.json)，完整32进程结果见下表。后续配置改动未追加全窗，旧样本不改名、不混算。
- 性能取舍：同步报告和hidden输出已合入；宽Lt、typed MMF及K2辅助强制捕获均无稳定完整收益，关闭且不补样。数字、源身份和取舍见[性能资料](performance/decode.md)。
- 准备归因：[唯一真实诊断](../evidence/performance/inference-model-prepare-attribution-2026-10-26.json)正常结束、非时间字段0差异。66次decode的布局／batch／replay为8.441／16.328／60.361ms；target原生计划366复用／30新建。整个batch桶仅占本次decode约1.64%，不能据此期待大幅收益。
- GDN取舍：[单行为＋8进程筛选](../evidence/performance/mtp-fixed-gdn-work-io-screen-2026-10-26.json)完整工作相同、录制27→22，但复制成本未换来净收益，关闭、不补样、不合入；未追加字节／故障检查。静态端口已有复用，剩余遍历量级未知；不同down形状也尚无足够收益依据，不盲开小项。
- 状态／捕获源码：[本地参考与现有合同](../evidence/research/inference-state-capture-next-review-2026-10-26.json)确认TRT／vLLM的索引更新需要原生状态池合同，Hyle尚无可直接接通的GDN／conv路由；已有ExecUpdate仍先重录。实际GDN／history／copy路径不能提供完整原生参数角色映射，需扩展launch／原生图接口；本轮不据此开发参数重绑。旧地址变化五个包络约29.17ms，不是当前可回收收益；旧13次录制的60.25ms还包含已消除的hidden开销，不能全算作剩余空间。大块复制候选保持关闭。
- 随机采样：[无过滤R2检查](../evidence/research/inference-batched-unfiltered-sampling-2026-10-26.json)CPU／CUDA各7项及小Model通过；[三臂正式对照](../evidence/performance/inference-random-four-request-screen-2026-10-26.json)全程工作与输出相同，批量路径有净收益，已合入。[Hybrid功能检查](../evidence/research/inference-sampling-hybrid-replay-2026-10-26.json)虽通过，完整吞吐却下降，关闭、不合入或补样。均不计作当前greedy MTP收益。
- CPU：[首对照](../evidence/performance/inference-cpu-initial-pair-2026-10-26.json)ISA偏差已用同版本15个CPU TU修正；[AVX2完整短窗](../evidence/performance/inference-cpu-avx2-window-2026-10-26.json)发现3428个OS线程。[Context共享池修复](../evidence/performance/inference-cpu-context-team-2026-10-26.json)已合入：每槽保私有plan／workspace，共享团队保活、完整native compute独占，9项runtime及真实模型工作一致。最新CPU同窗结果见下表；Mac本轮仅做有限组件检查。
- CPU归因：等待策略调整前的[唯一真实诊断](../evidence/performance/inference-cpu-native-attribution-2026-10-26.json)9104记录完整、0溢出、非时间工作0差异。decode每输出native84.460ms、其中矩阵图75.305ms（本次墙时77.94%）；wide投影33.032ms、tied head19.409ms为不相交组。锁接纳全15轮仅0.642ms；native外每输出12.162ms仍含其它provider／host／仪器，未归因。未测参考原生耗时，不能据此认定差距主因或可回收收益。
- CPU等待策略：[新两臂筛选](../evidence/performance/inference-cpu-polling-screen-2026-10-26.json)8新进程全成功、工作0差异，9项runtime及先行正常行为也通过，保原生默认有限轮询已采用。BF16点积／MUL_MAT相关核心与参考相同，不能归因于旧版CPU数学；参考每整图临时创建默认pool，并非跨step常驻自旋。
- CPU Graph归因：[单TU诊断](../evidence/performance/inference-cpu-graph-submission-attribution-2026-10-26.json)10944步及64个scope完整、0溢出、非时间工作0差异。同run decode墙时约99.05%在provider调用包络内，Graph其余每输出约0.919ms且含仪器；未见支撑大改Graph提交机制的量级，不跨run减去旧native耗时。
- CPU调用配置：[新两臂筛选](../evidence/performance/inference-cpu-caller-executor-screen-2026-10-26.json)六计时＋两独立内存进程全成功、工作0差异。原生4与ATen4不变，去掉测试入口的worker-only交接后decode＋6.54%、综合＋3.36%。默认executor本来为空，未改库或绕过用户回调；通用provider分工可能改变，不能把收益全算为纯交接耗时。后续该延迟负载采用调用线程配置。
- CPU矩阵源码：[有限复核](../evidence/research/inference-cpu-matmul-granularity-review-2026-10-26.json)未发现审查路径大权重重复转型／pack。Hyle中间BF16存储点与参考保留F32结果是实际数值合同差异；187矩阵／569原生小图来自旧诊断，不能由次数预测吞吐或把旧33ms包络当转换成本。
- CPU常量View：[实现、检查与新三臂对照](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)已完成。仅安全常量View参与冷发现连通性证明，实际View仍由原Graph管理；每图1195→1039步，逐节点BF16舍入保留。Windows四项薄入口检查及正式测试TU编译通过；[Mac五个当前产品TU＋薄消费者](../evidence/correctness/inference-cpu-constant-view-macos-2026-10-26.json)严格编译、7次提交通过，沿用冻结Graph／native／SDK，不代表完整当前Mac构建或吞吐。采用为资源／prefill优化，不计decode收益。
- 载权取舍：[直接host IO筛选](../evidence/performance/inference-cpu-host-direct-io-screen-2026-10-26.json)五个host TU、17项原suite及独立真实行为通过，但新六计时＋两独立内存窗口加载0.895→1.004秒（慢12.12%）、startup到ready1.003→1.117秒（慢11.27%），三样本范围不重叠。prefill43.331→43.359、decode11.416→11.419、综合3.283→3.290 tok/s均范围重叠；工作集各约5.320GiB、线程峰17，GPU缺测。627次／约3.76GB显式复制虽消除，staging与backing预算不变；关闭、不合入、不补样，不猜慢因。合法只读映射导入仍为设计，未计收益。
- 剩余顺序：先归因当前B2差距，再做CPU原生矩阵／工作分配的参考同项对齐；混合FFN及Apple／Vulkan缺当前完整消费者性能基线，分别做有限对照。合法只读映射仍为低优先级设计候选，无已证实收益。参数重绑缺少完整映射，索引状态需要原生能力，均无当前净收益依据，不开发新机制；减少图步数也不再据次数外推。当前greedy MTP与普通B2均未追平。

## 最新完整性能对照

2026-10-26页视图候选的Hyle自身对照：沿用普通B2固定输入，六计时＋两独立内存进程；只替换Arena对象，参考引擎未重测。

| Hyle配置 | prefill tok/s | decode合计 tok/s | 综合合计 tok/s | dedicated显存采样峰GiB |
|---|---:|---:|---:|---:|
| 当前发布扫描改进基线 | 7698.766 | 159.676 | 141.231 | 3.999 |
| 单源页视图选择候选，已撤回 | 7247.741 | 142.703 | 126.806 | 3.999 |

prefill−5.86%、decode−10.63%、综合−10.21%；后两者范围重叠，候选波动大，不据此断言退步原因，但**没有净收益依据，恢复原实现和测试**。8进程完整非时间报告及198个ID不变，相关8项CPU检查通过；新增测试先后有编译歧义和错误地使用被schema拒绝的布局，修正记录与失败原件均保留。两臂dedicated／shared峰相同，RAM微差不称节省，ready均缺样。[完整数据及取舍](../evidence/performance/inference-paged-reframe-screen-2026-10-26.json)不与旧peer样本混算，不代签MTP。

2026-10-26结果发布修改的Hyle自身对照：沿用下述普通B2固定输入，六计时＋两独立内存进程；只替换一个Graph资源对象，参考引擎未重测。

| Hyle配置 | prefill tok/s | decode合计 tok/s | 综合合计 tok/s | dedicated显存采样峰GiB |
|---|---:|---:|---:|---:|
| 原发布扫描 | 7594.862 | 163.688 | 143.978 | 3.999 |
| 发布完成事实复用，已采用 | 7892.315 | 163.947 | 144.977 | 3.999 |

prefill＋3.92%、decode＋0.16%、综合＋0.69%，**三个范围均重叠，不签稳定收益**。采用原因是发布扫描由O(S×B)降到O(B＋S)，只减少已完成的重复工作，不改变首次发布、权限和退休；不能据此宣布主要瓶颈修复。显存两臂峰相同，RAM微差不称节省；ready均缺样。[完整范围／内存及独审](../evidence/performance/inference-result-publication-screen-2026-10-26.json)保留原始数据，不与下述旧peer样本混算差距。

2026-10-26普通双请求新窗口：每请求三轮P256＋32个固定decode输入，C1024、N2／R2、prefix off，无MTP。采用当前Model的未插桩CUDA制品`41ce916d`及保留B2参考`632cc09c`；两端各三个交错测速、各一个独立内存进程。

| 工况 | prefill tok/s | decode合计 tok/s | 综合合计 tok/s | dedicated显存采样峰GiB |
|---|---:|---:|---:|---:|
| Hyle B2 | 7791.635 | 163.452 | 144.200 | 3.999 |
| llama.cpp B2 | 6450.984 | 181.111 | 152.128 | 4.743 |

三项速度的新样本范围均不重叠；不据三点签统计显著性。工作集／private commit采样峰为Hyle1.081／5.630GiB、参考4.106／5.927GiB；内存各仅一次采样下界，不签稳定节省。两端ready到prefill无样本、基线未知，GPU计数缺测保留。加载／ready／startup及完整范围见[原始窗口](../evidence/performance/inference-b2-current-window-2026-10-26.json)。固定输入与198个ID一致不代签中间数值或任务质量；BF16／F32与状态失败合同仍不同，也不是自然生成、R4随机、动态服务或多平台结论。参考与下表`433`共享8个库和运行环境，消费者及两个调度头不同，不称最新上游。

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

CPU最新三臂短窗：Windows i9-13900K、原生4／ATen4、调用线程、P154＋16、C256、N1／R1、greedy、无MTP；参考为同版本AVX2／FMA／F16C。每臂三个交错测速进程，内存各另一个进程，旧窗口不参与统计。

| CPU配置 | prefill | decode | 综合 | 工作集采样峰GiB | OS线程峰 |
|---|---:|---:|---:|---:|---:|
| Hyle原常量View边界 | 43.453 | 11.479 | 3.299 | 5.320 | 17 |
| Hyle透明常量View，已采用 | 43.953 | 11.437 | 3.323 | 5.002 | 17 |
| llama.cpp AVX2 | 42.937 | 12.027 | 3.314 | 3.828 | 15 |

prefill＋1.15%，三点范围不重叠；decode−0.37%、综合＋0.72%均范围重叠，不能称稳定提速。采用后CPU decode仍比同窗参考慢约4.9%。12进程全部成功，各臂完整非时间报告稳定；Hyle仅两处预先列明的workspace字段变化，三臂16输出及工作行数相同，不能代签中间数值或任务质量。

候选工作集／private commit采样峰分别下降约341／349MB；三臂private commit6.065／5.740／0.606GiB，映射与私有内存的差异不能当成权重大小。内存各一次采样，OS线程不等于忙线程，Hyle GPU无PID计数器实例、缺测不记零。完整数据、加载口径及源码身份见[新窗口证据](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)。旧共享池、等待和执行器配置的独立窗口保留在[性能资料](performance/decode.md)，不累加其收益。
