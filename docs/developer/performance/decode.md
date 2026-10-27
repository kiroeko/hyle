# CUDA decode 性能提升计划

本页维护当前 Qwen3.5-2B BF16、纯文本、单 CUDA GPU 测试消费者的性能工作。
模型与算子范围见[覆盖说明](../../implementation/model-operators.md)，测量方法沿用[性能指南](README.md)。
这里只保留当前基线、待验证假设、执行顺序和完成条件；原始实验留在 `docs/evidence`。
每次完整对比固定报告prefill、decode、显存变化和多轮会话综合速度，定义见[统一测量口径](README.md#推理对比固定报告)。
开发基准保留内部诊断参数`--decode-overlap off`以做同步对照、定位开销与复现问题，普通帮助与使用指南不列这个开关。
正常执行自动选择流水或单步；长期收敛为同一执行机制的不同在途深度，保留必要fallback并删除被替代的实验实现。

当前任务看[推理进度](../inference-progress.md)。本页先列最近批次，再保留先前调查依据；
旧记录中的“下一步”“本轮不开发”和候选结论只描述当时的实验，不能代替当前计划。
单会话普通／MTP的最近完整表见[目标与当前基线](#目标与当前基线)；
下面的B2、CPU和局部候选属于不同窗口，各自保留版本与测量边界。

[hidden索引接线与状态检查](../../evidence/correctness/inference-hidden-indexed-2026-10-28.json)已完成：carry／prior、接受前缀和递归NextN改用原固定池Gather／Store，删除独立hidden输出复制链。普通两轮、MTP单轮／两轮及当前K2完整非计时对照差异0；CPU／CUDA不同接受前缀、lag/catch-up和CPU主机恢复通过。单轮重录11→2，旧d9两轮15→2；两者历史基线不同，不能把计数换算为提速。保留首次冷图effect依赖失败及CPU DLL缺失，分别修正依赖链和原DLL同目录硬链接后检查通过。

最新[hidden接线完整窗口](../../evidence/performance/inference-hidden-state-current-window-2026-10-28.json)于北京时间2026-10-28 **01:10:56～01:14:13**完成40进程，固定`b59fe852`／`433e7051`、原五臂与端点。普通auto比本窗off的decode快2.64%／5.95%，比参考慢1.81%／0.59%；MTP比参考慢7.38%／5.48%，综合慢2.12%／3.72%。各臂完整非计时报告稳定；跨引擎续文第18位置的既存差异、实际接受与计算行数均保留，不称逐位相同算量或跨窗单项收益。

[a/b冷分组候选](../../evidence/performance/inference-small-projection-candidate-2026-10-28.json)用原权重128KiB通过M1／M2／T154及真实Cast／Sigmoid后缀的有限检查。M1已配对，声明执行段3→3；M2／T154为4→3，但后缀选路与workspace改变，私有物化和GPU耗时未测。尚未修改产品策略，不用执行段计数推算kernel数或整模收益。

[跨轮warm提示候选](../../evidence/performance/inference-cross-turn-warm-screen-2026-10-28.json)已撤回。
只编一个Model并运行一次真实两轮：进程exit0、生成内容和其它完整非计时事实一致，但后继提交／发布110→108、拒绝0→1；
唯一拒绝在committed375／candidate376，原捕获门报告最终物理tuple无既有完整capture。
保留Model／R／sampling提示不足以证明跨轮真实载体仍可复用；没有首差端口，不能借MTP诊断把普通路径拒绝归因hidden。
原候选及筛选失败保留，产品恢复；未跑EOS或完整TPS窗，也不把这次有限拒绝写成已测吞吐回归。

前一[固定拓扑attention完整窗口](../../evidence/performance/inference-stable-attention-current-window-2026-10-27.json)于北京时间2026-10-27 **23:10:34～23:20:05**完成40进程，使用`d9a2fe2c`与保留参考`433e7051`。每场景五臂各3次计时、1次独立内存采样，全部完整非计时报告在各臂内稳定。
auto相对当前off的decode中位单轮＋1.81%、两轮＋5.30%；相对参考普通decode仍慢5.23%／0.52%，综合输出慢0.45%／快0.013%。这只是本窗口描述，三样本不签显著性，也不与旧窗口拼算代码收益。

[Review核实与MTP诊断](../../evidence/performance/inference-capture-review-2026-10-27.json)确认旧40窗口的重录位置。当前W17单次诊断复现11次MTP录制，其中2次初始、9次已有key的首失配；16个有效变体首失配均为`prior_hidden`／`carry_hidden`的池内offset轮换，无显式key失效事件。只记录首差，不排除后续端口也变；完整历史非计时报告按原单会话导出规则归一后差异0。保留首次比较器别名失败与启动后离线关联修正，不据此声称新性能。

[固定拓扑原型](../../evidence/correctness/inference-stable-attention-2026-10-27.json)仅对普通paged F16/BF16、均长T1/T2、causal全窗口、D64/128/256启用原生graph调度；补全block mask和设备总行数绑定。只读query先比较完整原生plan拓扑与偏移，共同层核原carrier和runtime后才保capture；实际metadata replan仍上传、等待并保留accepted错误退休。用已有metadata Queue规划，执行Queue不变，未加公开配置项。
BF16 D256的96次Graph检查覆盖R1/R2、T1/T2、127/128/129与383/384/385/1023、双槽、fresh和形状变化，FP64参考最大绝对误差0.000156995；这组未代签F16及其余域。三项真实模型检查除预期流水计数外完整非计时内容相同：两轮off捕获3→1，auto15→7、发布102→110、拒绝1→0；MTP单轮仍11。记录次数不是kernel数，也不能直接换算墙时收益。

随后[小工作区预算与CLI回归](../../evidence/correctness/inference-attention-budget-fallback-2026-10-27.json)补齐优化的资源边界：固定grid超显式workspace cap时只读重查原调度，实际plan消费相同模式，原调度仍不足则保留容量错误，不增加公开开关。100次组件检查通过；原3,026,432B浮点需求在1MiB cap下，KV127／128实际需0B，129及fresh需131,584B，沿用同一FP64／tail／guard检查。最终镜像`91ebdde8`包含该回退与普通帮助隐藏内部off，唯一自然两轮检查完整非计时差异0。保留辅助脚本的路径与缺参措辞错误及离线修正，不重跑成功模型。该批本身未重测完整速度；后续已测镜像与结果以上文最新窗口为准。

此前[长输入完整窗口](../../evidence/performance/inference-long-input-current-window-2026-10-27.json)于北京时间2026-10-27 **22:23:44～22:27:18**完成16进程，使用默认auto的`aa864724`与保留参考`433e7051`。
P1024／C2048与P4032／C4096均整段prefill、最多64个自然输出；实际分别46（EOS）／64个输出，两边全部ID相同。
prefill中位领先42.81%／23.36%，decode仍慢5.35%／7.81%，含prefill的综合输出速度领先4.23%／3.43%；三样本不签显著性。
两个不同输入是固定重复原prompt IDs的合成性能负载，不是质量任务；不将跨窗口绝对速度变化归因代码收益。四项与内存边界见[当前基线](#目标与当前基线)。

此前[普通off／auto与MTP完整窗口](../../evidence/performance/inference-continuous-wave-current-window-2026-10-27.json)于北京时间2026-10-27 **21:27:15～21:38:23**完成40进程。其ordinary auto相对off的decode中位单轮＋3.10%、两轮＋1.74%；单轮范围重叠，不能宣称稳定提速或反超。两轮auto仍慢参考约3.99%，MTP慢约10.7%～10.9%。中间解析器暂停后只续跑未运行项，未重跑成功模型；不与新长输入或旧窗拼算收益。

随后[流水恢复与默认选择](../../evidence/correctness/inference-wave-recovery-2026-10-27.json)用一次实际诊断确认，committed383／candidate384时的拒绝来自pending物理tuple没有完整已有捕获，而非设备错误或KV历史重算。
现先完成原A、确认真实退休，再沿ready路径重准备；同一自然两轮的实际后继发布59→102、128个输出及实际计算工作不变，仍有一次被正确处理的拒绝。
CLI改为默认auto，缓存预算不足D2则自动D1；额外stop、控制异常及CPU／CUDA普通与MTP有限回归已检查。新共同wave只有静态控制、不另造执行器。
该批本身是功能检查，不能把发布次数增长换算成提速；之后的长输入窗口已列于上文，短输入／MTP表仍对应旧冻结`fefa445f`。

随后[普通连续wave接线](../../evidence/correctness/inference-continuous-wave-2026-10-27.json)在同一RequestLoop下跨pump保留原后继。
真实Qwen3.5-2B的R1／自然R2各完成97／194个输出，auto的后继提交和发布由44增至88；逐模式完整非时间报告仅这两个顶层计数改变。
十个tiny控制臂另比较每个已提交前缀的完整状态、epoch和RNG，并覆盖信用耗尽、跨pump单成员取消及yield／恢复。
本批还消费已返回的append交易以释放被替换旧尾页，区分驻留版本N与在途candidate。5次C++编译、5链接通过，复用SDK／原生内核；
该功能批次本身没有统计吞吐／内存结论；Model2计时观察错读prefill完成点，其吞吐全部不采用。随后Model3改读真实prefill_endpoint，增加正区间门，严格编译／链接各一次；该40进程窗口使用Model3，当时默认overlap关闭，后续默认策略另见上文。

随后[自然R1两步接力筛查](../../evidence/performance/inference-generation-continuation-screen-2026-10-27.json)用同一镜像交错运行六进程，P154＋96decode的完整非时间单请求结果一致。
每个auto进程实际发布44个后继；off／auto的prefill中位2122.69／2164.08、decode90.37／90.27、综合85.48／85.49 tok/s，范围重叠，未证明净提速。
ready中位增加39.83ms，共享cache backing增加18.82MiB；未采样全显存峰值，未新跑参考或MTP，不刷新上述完整表。
子步计时会重复包含A／B重叠区间，不能从step_sum减墙时推断主机开销。新增`--decode-overlap`默认off，下一项是分开量取Source导出／受检发布与被隐藏的准备成本。

同批另一个仅Model计时的诊断镜像，off／auto各跑一次且97 IDs与完整非时间结果一致：原wait调用墙时少53.59ms，其它互不重叠的调用桶多57.35ms，decode净多3.49ms。
prepare多47.40ms，其中前8步多17.73ms、其后88步多29.67ms；Source元数据与typed导出合计14.91ms，post-wait则少13.11ms。父子桶不再相加，这些是主机调用墙时，包含原生API影响，不能称CPU忙时或GPU活动时间。
稳态没有新增捕获。引擎额外查询A未来重放资格花3.16ms，已删除：B仍由原Graph在准备和提交处核自己的完整tuple与来源，不扩大权限。这项删除未另跑吞吐窗口；下一步拆清relay prepare增加的约0.3ms/步。

[三段prepare诊断](../../evidence/performance/inference-generation-prepare-2026-10-27.json)另用冻结Model3镜像各跑一次，复用已有时钟且完整97输出事实一致。
该次prepare共多54.38ms：前8步多18.36ms，后88步多36.02ms；后者来自绑定＋26.24ms、replay＋12.56ms，而布局与residual略省。Source元数据及capture另为15.23ms，不与prepare父桶相加。
稳态均零新增捕获。这里记录主机调用墙时，不是CPU忙时或GPU时长，也不与前窗47.40ms差额混算；下一候选只提前alias原有纯input/input的Ok分支，保留真正output重叠及权限检查。

[alias角色过滤A/B](../../evidence/performance/inference-alias-role-filter-2026-10-27.json)复用224个CLI链接输入，只替换Graph executable对象；5项现有事务／别名检查通过，12个交错进程完整非时间报告逐模式一致。
关闭接力时prefill／decode／综合中位为2117.64→2063.39／91.37→91.22／86.44→85.69 tok/s；开启时为2185.21→2127.34／91.45→91.58／86.43→86.53，范围均重叠，未证明提速或稳定回退。
源码提前过滤纯input/input的无效replacement查找，未改变output／immutable Source及事务合同；编译器可能已消除部分无效工作，不将源码级复杂度变化当作实测收益。逐模式cache backing相同，未采样全显存峰、未新跑参考或MTP；保留局部简化，不补样，不刷新完整跨引擎表。

[恒等replacement筛选](../../evidence/performance/inference-identical-capture-inputs-2026-10-27.json)只在逐项确认替换等于当前完整物理输入后，跳过每个captured port对替换表的重复查找；仍核全部当前capture key、alias、Queue和private资格。该查找项从O(P×M)变为O(P＋M)，不声称整个函数线性化。5个不同定向用例取得通过结果：首镜像4过、1个旧LRU销毁计数失败，修正为现行唯一不可执行rebuild seed合同并加强失权断言，第二镜像仅复验该项通过；合计3次C++编译、3次链接。

12个交错普通R1进程完整非时间报告逐模式一致；off的prefill／decode／综合中位2169.41→2145.13／96.70→96.67／91.19→91.09，auto为2181.75→2046.83／96.68→96.78／91.36→90.78 tok/s。范围均重叠，未证明净提速，auto的prefill中位下降6.18%也如实保留。逐模式cache backing相同，未采样全显存峰，未跑新参考或MTP，完整表仍是18:01窗口。采用为恒等替换的局部简化，不追加样本或启用默认接力。测速脚本首次helper属性错误发生在模型启动前，修正后12个模型进程各仅运行一次。

[Source绑定分段](../../evidence/performance/inference-source-bind-attribution-2026-10-27.json)在泛化浮点Source后的冻结镜像上仅跑off／auto各一次，逐模式97个ID及完整非时间报告与原控制一致；auto实际发布44个后继。用嵌套扣除后的互斥桶定位：95次完整bundle绑定合计10.216ms，原子输入绑定含子桶5.461ms，其中6935次native-input扫描遍历7,087,570项但仅1.957ms，零native命中。不能因扫描次数大就把它当主因。
slot绑定的完整观察为off12.037／auto15.338ms；写effect为3.634／6.306ms，auto另有1.928ms来源投影，IDs处理含等待1.781／1.622ms，未增加。auto的prepare_outputs合计38.317ms，其中replay其余31.039ms、consumer检查4.690ms、来源投影2.451ms；它包含原完整准备，off的普通prepare_replay未按同样入口插桩，不能以零相减称额外31ms，也不能混减前一Model3窗口的26.24ms。下一步定位共同replay准备，保留每次真实权限／tuple检查。本批只有诊断墙时，包含原生调用与等待，不是CPU忙时、统计提速或最新四项对照。

[新Review的共序校正](../../evidence/research/inference-review-common90-2026-10-27.json)发现，10-26旧跟踪的原review脚本使用Hyle90／参考93个nonrecord根，参考多3个eager根。按双方同seq、各一次native launch的90／90重算，墙时12.876／11.670ms，kernel并集10.641／10.598ms，全部GPU活动并集10.784／10.688ms；GPU活动之外的差额约1.110ms／根，仍需查图外工作。Hyle132次D2D合计0.137ms／根、39.715MB，不能以次数解释1.206ms墙差。首kernel前还含GPU上传／间隙，不是CPU自耗；逻辑权重字节除kernel时间也不是实测DRAM带宽。仅离线重读旧库，不刷新TPS；后续冻结已验证镜像的8个诊断进程结果如下。

[本轮普通／MTP跟踪](../../evidence/research/inference-current-review-profile-2026-10-27.json)完成R1／自然R2的off、auto和参考，以及R1 MTP双方共8个成功Nsight进程。Hyle使用`2fe2916cf`阶段已验证Model的NVTX冻结副本，包含索引状态／KV工作帧，不含后来的Store区间修正和RequestWave；不把它称为当前HEAD性能。下表是各臂同seq、无录制且单次native launch的共同88根，排除前8步，单位ms／根：

| 工况 | Hyle off根墙时 | 参考根墙时 | Hyle off GPU并集 | 参考GPU并集 | kernel数 Hyle／参考 |
|---|---:|---:|---:|---:|---:|
| R1 | 11.209 | 10.508 | 9.591 | 9.716 | 613／558 |
| R2 | 12.522 | 11.781 | 10.547 | 10.812 | 741／709 |

GPU量级接近而根墙时仍落后，继续定位GPU活动之外的工作；差值包含等待／间隙及仪器影响，不是CPU自耗。Hyle的D2D未归零：R1每根19次／77,824B／约0.015ms，R2为60次／491,520B／约0.050ms，旧132次／39.715MB不再适用。这些小拷贝的直接时长不足以解释当前墙差，具体来源仍待逐项核实。R2双方每根均有36个grid1／block32的同名小kernel，合计约0.415ms，支持review的小投影量级；原生短名不能独立证明每项逻辑节点身份。

auto仍只发布44个后继。共同88根的root并集off→auto为R1 986.357→947.085ms、R2 1101.970→1069.553ms，GPU并集也由844.042→850.684／928.104→942.971ms增加；不能从重叠根时长之和比较或称CPU成本完全隐藏。这是各臂单次带仪器诊断，Model根结束于外层ledger／交付之前，不更新完整吞吐或显存基线，也不推翻此前普通六进程筛选的“未证明净提速”。

MTP本次BeginCapture为16／6，Instantiate为5／2，ExecUpdate为11／6；旧27／7不作当前结论。Hyle／参考生成各97个输出，但target行257／255、接受草稿44／45，续文和实际阶段并不完全相同，且Hyle合并target／draft根，不能拿53／104根当相同工作比较。seq53首次终端T1角色、seq43拒绝后角色重入可由本次记录确认，其余重录的具体失效原因尚未全部定位；不能先把多出的捕获都判为可删除。原一次错误GGUF加载失败和离线报告oracle修正均保留，没有重跑成功模型。后续优先核实录制／准备的真实失效边界，并接通原episode跨pump推进，再筛选小投影优化。

随后[在途索引状态接线](../../evidence/correctness/inference-pending-indexed-source-2026-10-27.json)完成有限GDN／conv捕获复用与真实Source旅程，并去掉同值IDs的重复上传／同Queue等待。该批尚未接入完整模型的有界流水，也未运行tokens/s窗口，不据组件结果估算提速；自然请求的后续接线及最新完整窗口分别列于上文。

此前[引擎原池索引状态A/B](../../evidence/performance/inference-engine-slot-state-2026-10-27.json)为两版Hyle的普通B2同窗对照：6计时＋2独立内存进程全部通过，1728行、99次执行、198个ID及完整非时间报告不变。prefill7668.03→7606.82（−0.80%，范围重叠）、decode158.48→161.96（＋2.20%）、综合140.55→142.67 tok/s（＋1.51%）。dedicated采样峰4294000640→4243668992B，低48MiB；ready内存基线未知，采样峰为下界。代价是ready中位1.33150→1.37582s、冷shape准备0.411183→0.464081s；三样本不签统计显著性。原池Store接线及CPU／CUDA有限功能已检查，真实CUDA消费者的GDN／conv融合已观察；其它平台字节尾部、MTP异成员与当前完整性能仍待检查。本批未新运行llama.cpp或MTP性能，不更新旧跨引擎差距。

此前[KV持续工作帧直接A/B](../../evidence/performance/inference-paged-frame-2026-10-26.json)只比较同窗Hyle控制／候选，六计时＋两独立内存进程全部成功。224个链接输入仅arena、layout、append与Model四项改变，其余220相同；1728输入行、99次执行、198个输出ID和完整非时间报告不变。prefill7509.27→7401.39（−1.44%，范围重叠）、decode156.72→158.30（＋1.01%，范围不重叠）、综合138.47→139.75 tok/s（＋0.92%，范围不重叠）。dedicated显存采样峰两臂均4294000640B，未证明节省；RAM单次微差不作收益，ready基线未知。三样本不签统计显著性。本批未新跑llama.cpp或MTP性能，不混旧窗口刷新跨引擎表。工作帧消除了每步不可变整帧导出／合拆；当时尚未包含上述原池GDN／conv与MTP接线，不能说主要状态协议成本已经消除。

[批量append取回直接A/B](../../evidence/performance/inference-bulk-append-2026-10-26.json)已完成6个新TU、7个有限功能进程及新六计时＋两独立内存进程。当前／候选Hyle除Model、Graph资源与append三个直接对象外，其余221个链接输入相同；同一普通B2工作、完整非时间报告和198个输出ID不变。prefill7320.892→7210.113、decode154.349→154.197、综合136.232→135.877 tok/s，分别−1.51%／−0.10%／−0.26%，范围均重叠，未证明提速。两臂dedicated／shared峰同为4294000640／96468992B，ready基线未知；RAM微差不记收益。采用为通用批量所有权接口与消除逐bank重复权限表压缩，不继续补样，也不混旧peer刷新对照表。下一项是KV持续工作帧及完整提交边界，不能把本批当作原页帧／状态协议成本已消除。

最新[状态复用后B2完整窗口](../../evidence/performance/inference-b2-reusable-state-2026-10-26.json)使用该批正式入口，同次六计时＋两独立内存进程全部成功。C1024、N2／R2、三轮固定输入、无MTP／prefix；1728输入行、99次物理执行、198个输出ID一致。Hyle／参考prefill7469.731／6086.119、decode154.941／170.683、综合136.981／143.737 tok/s；分别＋22.73%、−9.22%、−4.70%，三样本范围不重叠但不签统计显著性。dedicated显存峰3.999／4.743GiB，工作集峰1.080／4.106GiB、private commit5.624／5.926GiB；内存为独立有限采样下界，ready基线未知。CPU／CUDA有限消费者已确认原writer双块交替与旧快照不变，但本窗没有控制臂或新profile，不归因本批提速、重录次数或主机时间下降。主要差距仍在；随后进行了上文的批量append取回对照，未证明提速。下文旧窗及诊断作为调查依据保留，不混样。

新增[参考同设置CPU诊断](../../evidence/performance/inference-b2-peer-cpu-critical-2026-10-26.json)：参考完整工作／198个ID／99根通过。与下述Hyle诊断共同90个不重录、各含一个原生图的decode根，墙时1223.209／1085.084ms；主线程运行且在GPU／CUDA API之外99.701／14.365ms，差85.337ms。图前／图包络／图后的差额为71.758／40.123／26.243ms，精确构成墙时差；图包络包含内部空隙，不能称kernel忙碌时间。原始调度／API／GPU区间经独立SQL边界重建一致。各一次诊断、17／4个阶段标签、驱动和不同状态／数值合同仍有限制，不刷新TPS或宣称85ms均可回收。优先定位具体准备／发布函数；参考同样逐episode同步，不能从单槽直接推导根因。

管理员权限下的[一次CPU栈／线程调度诊断](../../evidence/performance/inference-b2-cpu-critical-2026-10-26.json)已正常结束：仅一个Model TU补`/Z7`并重链本地PDB，其他221输入不变，完整原报告、198个ID及99根／1782区间通过。96个decode墙时1346.646ms，导出调度重建运行1340.863ms、离CPU5.783ms；6个样本CPU编号不一致，已发现问题的运行片段与decode重叠2.706ms，保留不确定性。完成等待1144.970ms中96次`cudaEventSynchronize`占1141.378ms、GPU图区间重叠1135.763ms，不能将这些重叠时间算作独立可回收CPU成本。CPU运行且位于GPU／CUDA API之外共122.228ms仍待参考同设置对齐；稳态replay全桶26.016ms仅占稳态根约2.13%。符号覆盖有限，稀疏公共符号还会跨真实函数边界错误归名，不能从`position_index`标签判断容器热点。当前消费者每Phase单槽、Model episode串行属实，但实际peer也decode→synchronize→采样／发布后才下一episode，尚无“单槽差异直接致慢”的证据。此次无peer新profile或吞吐表，不跨新graph／旧node诊断归因收益。

本轮[12阶段诊断](../../evidence/performance/inference-b2-host-phases-2026-10-26.json)在同次96个decode中记录replay准备60.961ms、batch准备25.062ms、页导出21.130ms。页导出含take_append／ready／reframe，占根墙时约1.63%，不是可全部删掉的CPU成本。99个根及1782个NVTX区间完整，但工具截断stdio，完整报告和198个ID门未完成；仅用于区间定位，不刷新速度表或代签工作通过。现有原生图的132个D2D事件也不能直接当额外工作：参考通过kernel更新cache／状态，须按实际搬运语义对齐。

[同次离线分组与有限源审](../../evidence/research/inference-b2-host-next-2026-10-26.json)进一步确认replay中6个录制根37.772ms、90个非录制根23.189ms（均值257.658µs／根，约占该次decode根总墙时1.79%）。非录制batch准备另17.362ms；两者是独立子桶，不把总61ms当热路径成本。命中现有capture-range时会跳过原生重录；binding/span/qualification已有缓存，seal仍需当前权限，队列容量足时不重复原生分配。此限定链未找到新的可直接实现的大额冗余调用，不新增缓存／参数重绑机制。仍需定位batch等剩余成本，不据此宣称性能已穷尽。

随后[单源页视图候选完整筛选](../../evidence/performance/inference-paged-reframe-screen-2026-10-26.json)六计时＋两独立内存进程全部正常结束，完整非时间报告及198个ID不变，8项CPU检查通过。控制／候选prefill7698.766／7247.741、decode159.676／142.703、综合141.231／126.806 tok/s；分别−5.86%、−10.63%、−10.21%。decode／综合范围重叠且候选波动大，不猜退步原因；没有净收益依据，原实现和测试均已恢复，不补样。dedicated／shared峰两臂同为4294000640／96468992B，RAM微差不称节省、ready未知。新测试的编译歧义及错误使用非法布局两次失败、修正、候选源码及全部数据保留。未重测参考，不与旧peer混算，也不代签MTP。

2026-10-26当前顺序：hidden输出稳定、CPU冷配置和同步报告输出已合入，CUDA已完成32进程正式对照。宽Lt、typed BF16 MMF、K2辅助图强制捕获及GDN双向固定工作IO均无稳定完整净收益，保持关闭。四请求普通随机批量采样decode＋10.76%、综合＋9.83%，已合入；Hybrid关闭。CPU已修参考ISA、Context共享团队和等待策略，并为当前延迟负载选择调用线程；最新常量View聚合主要减少内存并改善prefill，未取得稳定decode收益，CPU decode比同窗AVX2参考慢约4.9%。录制主机调查已结束，无大块外层Graph成本；普通B2完整窗口仍慢9.75%。新B2归因优先指向准备／提交／回收，但尚无已证实可大幅回收的具体缺陷。发布重复扫描未证稳定提速，页视图候选新完整窗口无净收益已撤回，停止扩大小项；下一步按现有轨迹区分录制／稳态准备成本并核具体调用。各独立窗口不混样或累加，单请求／MTP正式表未刷新。

[普通双请求新完整窗口](../../evidence/performance/inference-b2-current-window-2026-10-26.json)复用未插桩`41ce916d`及B2参考`632cc09c`，N2／R2、C1024、prefix off、三轮P256＋32固定decode输入，无MTP。六个交错测速及两个独立内存进程均正常结束；每臂完整非时间报告稳定，1728输入行／198决策／99个R2 episode、全部198个新旧及跨引擎ID均一致。仍有BF16／F32中间数学与不可变／可变状态、失败合同差异，不能代签逐位数学或任务质量。参考和`433`共用8个库及运行环境，但消费者／两个调度头不同，未升级上游。

| B2当前中位数 | Hyle | llama.cpp |
|---|---:|---:|
| prefill tok/s | 7791.635 | 6450.984 |
| decode合计 tok/s | 163.452 | 181.111 |
| 综合合计 tok/s | 144.200 | 152.128 |
| dedicated显存采样峰GiB | 3.999 | 4.743 |

prefill＋20.78%、decode−9.75%、综合−5.21%，三组新范围均不重叠；三样本不签显著性。WS／private commit峰为Hyle1.081／5.630、参考4.106／5.927GiB；内存各一次下界，不称稳定节省。ready基线两端未知，GPU缺样32／13保留；实际采样中位约15.13／15.07ms。加载／ready／入口到末输出中位为Hyle0.838／1.322／2.700s、参考1.678／1.714／3.003s，内部加载边界不同且未清OS缓存，不排名冷盘。速度分母为原P／D wave和全旅程端点，不叠加重叠成员时间。旧17.65%只作为调查动机，不与本窗相减归因，也不代签自然生成、R4随机或服务吞吐。

[普通B2同工作量诊断](../../evidence/performance/inference-b2-current-attribution-2026-10-26.json)两端各一个新profile均正常结束，99个物理调用／1728输入行及198个输出ID与原工作核对一致。96个decode根包络Hyle／参考为1279.679／1144.362ms，其中GPU活动并集1037.307／1024.996ms：135.317ms墙时差中，GPU并集只差12.311ms，优先查主机准备／提交／结果回收。余量含等待、未观测工作、仪器和不同原发布边界，不是123.006ms可回收CPU时间，也不刷新TPS／内存表。

同次Hyle prepare103.413ms（内含replay56.691）、submit91.066ms、completion wait1051.574ms、postwait publication32.908ms；最后一段29.785ms在可见API区间外。源审确认该B2每步72次GDN状态retain都先调用`Invocation::result`，原实现每次重复扫描固定writer集合，首次成功调用已完成发布；后续局部修改的结果见下段。词表投影两端暂定签名耗时相近且唯一节点证明未完成，降优先级。Hyle6次录制、参考3次；参考96个decode仅93个GraphLaunch，90个共同无录制launch不可与全部无录制根混称热捕获。独立区间复算及摘要对照闭合；首peer私有编译失败与修正日志保留。

上述发布扫描的[局部修改及完整筛选](../../evidence/performance/inference-result-publication-screen-2026-10-26.json)已完成：一个产品TU、一个测试TU严格编链，5项CPU租约合同及8个CUDA普通B2进程通过，完整非时间报告与198个输出ID不变。prefill7594.862→7892.315、decode163.688→163.947、综合143.978→144.977 tok/s，分别＋3.92%／＋0.16%／＋0.69%，全部范围重叠，**未证明稳定提速**。两臂dedicated／shared采样峰分别同为4294000640／96468992B，ready均无样本，RAM小差不作收益。采用为消除确定重复扫描的局部复杂度改进，首次result发布及失败重试时机不动；不能将postwait整段或主要B2差距归于此处，不补样或继续扩大该候选。编译本身成功，首次工具依赖对照因Windows路径大小写误判而在link前停止；修正后仅显式续链，原记录保留。下一项仍是准备／提交衔接，不跨窗合成与参考的差距。

[录制主机有限诊断](../../evidence/performance/inference-cuda-recording-host-attribution-2026-10-26.json)只重编两个私有Graph host TU并重链，唯一Hyle进程通过原完整工作门。21个有录制decode的replay共71.512ms，全部同父siblings65.549ms、父余5.963ms；其中dispatch43.909ms内step并集42.804ms，外部循环／NVTX余量1.105ms。4446个GGML桥型step的未覆盖部分18.715ms，2015个cuBLAS型1.823ms、448个未唯一归属2.888ms，均含仪器／未记录原生工作，不能称可回收。13次Instantiate16.346ms、14次Update1.509ms全成功，未见更新失败回退。结束本批N1录制调查，不造新的Graph调度／参数重绑机制或继续细分kernel；下一项看B2实际执行。首私有编译因Windows interface宏污染失败，隔离后严格编链通过，原失败保留；归档校验后精确清理5个可重建文件147438426B，不动模型／参考／依赖缓存。诊断不刷新TPS表。

[当前CUDA／MTP双引擎诊断](../../evidence/performance/inference-cuda-mtp-current-attribution-2026-10-26.json)各仅一个新Nsight进程，均正常结束、完整非时间报告与各自参考一致；当前Model／sampling身份与冻结依赖闭包明确列出，未重编SDK或设备内核。Hyle66个合并decode episode的墙时／GPU活动并集为1043.67／895.99ms；其中21个有录制episode的replay准备71.48ms，可见API区间并集40.63ms，其余30.85ms尚未归因，不能称纯CPU或承诺可回收。45个无录制episode墙时／GPU并集为662.66／612.89ms；差值含等待和仪器，不等于Graph开销。父子区间不相加。

Hyle27次录制对应13次Instantiate及14次成功ExecUpdate，参考7次捕获、2次Instantiate、7次成功Update；没有观察到Hyle更新失败回退。后续同核签名组合补齐的5次Instantiate合5.49ms，签名不证明完整图拓扑可更新，量级也不足以支持新造重绑框架。晚段Hyle target捕获子集均值10.883ms、参考target整调用内GPU并集10.602ms；draft分别2.721／2.738ms，范围不同，不能合成完整墙时差距或收益。两边根间gap每轮约1.5–2.0ms，不支持把主要差距归为日志间隙。实际target行441／438、draft行434／435且续文／中间精度不同，不能签同工作量胜负。独立复算未发现所审计算的阻断问题；端点补充另保留源码边界说明。仅为诊断，正式prefill／decode／RAM／VRAM／综合速度表不变。

[量级排序原记录](../../evidence/research/mtp-next-priorities-2026-10-26.json)中的GDN双向工作IO现已筛选关闭，结果见下文。次级源审确认已有常量安装、alias索引和revision／span复用，实际剩余静态遍历占比未知，尚不支持实施；不同形状N2048／K6144 down也只在旧均值有小差、范围混杂。六层fresh native plan共享包络小且须新增寿命机制，暂不优先。无过滤R>1及随机Hybrid不影响当前greedy N1 MTP窗口。

[GDN固定工作IO筛选](../../evidence/performance/mtp-fixed-gdn-work-io-screen-2026-10-26.json)仅替换Model对象，控制使用新编canonical b346，候选保原独立Snapshot和接受前缀，以一个原CopyBatch控制串行处理ingress和合并hidden／GDN egress。新增逻辑固定payload约111.8MiB、每轮37.27MiB复制与真实等待全部计时。一次行为及八进程均exit0、完整非时间stdout相同；原stderr只在预测的五个位置改变录制／复用，总27／105→22／110。decode中位127.652→127.025（−0.49%，三个范围不重叠）、综合116.434→115.905（−0.45%，范围重叠），prefill中位＋1.75%但范围重叠。三点不代签稳定回归；单次独立dedicated显存采样峰多96MiB，不等于逻辑payload或精确物理分配。关闭、未合入，不补样，也未继续字节／故障检查；不刷新正式四臂表。

两项随机普通推理候选已完成[四请求正式筛选](../../evidence/performance/inference-random-four-request-screen-2026-10-26.json)。无过滤批量采样沿原Categorical接整批logits及各请求原身份，decode221.435→245.271 tok/s（＋10.76%）、综合206.121→226.373（＋9.83%），三次范围不重叠；prefill6036.674→5944.634（−1.52%，范围重叠），已采用。其[有限数值检查](../../evidence/research/inference-batched-unfiltered-sampling-2026-10-26.json)CPU／CUDA各7项及小Model通过；真实过滤继续保原逐行路径。单次独立dedicated显存采样峰4.417→4.401GiB，不能签16MiB确定节省。

[Hybrid功能检查](../../evidence/research/inference-sampling-hybrid-replay-2026-10-26.json)虽然通过，正式四请求decode却为214.220（−3.26%）、综合199.074（−3.42%）、prefill5674.337（−6.00%），三个范围均不重叠，关闭、不合入、不补样。该候选只让原Graph捕获兼容连续范围，采样原生capture=false不变。全程输出、RNG、ledger、调度和工作均与控制相同；512条成员报告为Hybrid、500条有复用，不称512次独立捕获。三行为＋九计时＋三独立内存进程均成功；首次清单转义失败在加载前退出，修正后重新进行行为检查，保留原失败。三点不代表统计显著性，不计作单请求／MTP或对参考引擎收益。

CPU另完成[AVX2公平短窗](../../evidence/performance/inference-cpu-avx2-window-2026-10-26.json)，不沿用generic参考库的虚假胜幅。发现每Invocation独持原生池带来3428个OS线程后，[共享Context团队修复](../../evidence/performance/inference-cpu-context-team-2026-10-26.json)保留每槽独立plan／workspace、原native T4和真实退休，由一个Context保活团队并串行执行完整native compute。两产品TU严格编译、9项runtime及真实模型通过，未增加用户配置，已合入。

新的九计时＋三独立内存进程均成功，Hyle原池／共享池／AVX2参考的prefill为42.584／43.159／42.981，decode10.061／10.323／12.042，综合3.131／3.190／3.312 tok/s；共享池decode＋2.60%、综合＋1.91%，三点范围不重叠。工作集采样峰6.548／5.320／3.828GiB，private commit7.350／6.065／0.607GiB，OS线程峰3428／17／16。Hyle两臂完整非时间报告相同，参考各次报告稳定，三臂16输出及计算行数相同；线程数不是忙线程数，映射内存与private commit不直接等价。Hyle GPU计数器无PID实例，缺测不记零。该T4／P154＋16／C256／N1／greedy短窗的主要收益是资源节省，CPU decode仍慢14.28%；不扩展为多请求、跨平台、长上下文或CUDA MTP结论。后续优先核真实计算与权重存储成本，不据线程数推断剩余瓶颈。

[唯一CPU原生归因](../../evidence/performance/inference-cpu-native-attribution-2026-10-26.json)在等待策略调整前仅换一个runtime对象，原工作和输出保持一致；9104事件、1144图身份及10554节点完整、0溢出。按原marker与实际冷图身份分为prefill569调用、decode8535＝15×569调用，prefill的M1 head未算入decode。decode每输出原生compute84.460ms、含单MUL_MAT的完整图包络75.305ms，占本次墙时77.94%（原生桶的89.16%）；其中66个wide投影33.032ms、tied head19.409ms、24个FFN down12.041ms为不相交组。全15轮team接纳仅0.642ms；原生外每输出12.162ms含其它provider／host及仪器，不能全称可删除开销。图包络不是纯kernel或实测DRAM流量，也尚未与参考同项对齐，不将本次诊断TPS当基线或收益。

[CPU等待策略新窗](../../evidence/performance/inference-cpu-polling-screen-2026-10-26.json)先核实际源码：mixed DLL、参考AVX2和纯CPU新版的BF16点积／MUL_MAT相关核心相同，不能凭版本名指认旧内核较慢。参考每整图临时创建默认pool，Hyle则在同Context团队上执行许多小图；原强制poll0立即休眠，候选只删这一冷覆盖、保原生默认有限轮询。1TU严格编链、原9项runtime和唯一正常行为均通过，随后另跑六计时＋两独立内存进程，全成功、完整非时间工作一致。

新窗prefill43.068→43.203（＋0.31%，范围重叠）、decode10.382→10.612（＋2.21%，范围不重叠）、综合3.191→3.214 tok/s（＋0.71%，范围不重叠）；整进程CPU时间中位21.109→22.031秒（＋4.37%），包含加载／退出，不是decode CPU时间或能耗。工作集各约5.320GiB、线程峰各17，GPU计数缺测；内存单样本微差不作节省结论。按速度优先采用此小收益，没有新用户选项；不与旧CPU或CUDA窗口拼样，也不据此认定整段余差主因。后续核其它provider／Graph流程及参考同项成本，不扫描更多poll参数。

[CPU Graph提交诊断](../../evidence/performance/inference-cpu-graph-submission-attribution-2026-10-26.json)只替换一个submission对象，10944步／64个scope完整、0溢出、原输出和非时间工作0差异。同一run的decode每输出provider调用包络96.262ms／墙时97.181ms，约99.05%；其中ggml-cpu91.417ms、ATen4.544ms、oneDNN0.282ms。其它约0.919ms包含仪器；队列begin＋finish全15轮仅0.0264ms，不支持从Graph记账期待大幅收益。provider包络含绑定、执行器及原生计算，不能称纯kernel；不与旧native诊断相减，诊断TPS不当基线。

随后[调用线程配置筛选](../../evidence/performance/inference-cpu-caller-executor-screen-2026-10-26.json)只换测试入口对象，native4／ATen4及Scheduler4寿命不变，原borrowed executor concurrency4改用默认empty executor concurrency1。一个正常行为及六计时＋两独立内存进程全成功，完整非时间报告0差异。prefill42.247→43.362（＋2.64%，范围重叠）、decode10.842→11.551（＋6.54%，范围不重叠）、综合3.182→3.289（＋3.36%，范围不重叠）；整进程CPU时间中位22.328→21.703秒，两边线程峰17、工作集约5.320GiB，GPU仍缺测。此延迟负载后续采用调用线程；默认executor原本为空，没有库实现或默认策略改动，也不能绕过用户的执行位置要求。通用provider分工可能改变，不把全部收益签成纯交接耗时。没有新参考进程，不跨窗拼算收益或剩余差距。

[CPU矩阵源码复核](../../evidence/research/inference-cpu-matmul-granularity-review-2026-10-26.json)未发现审查路径大权重重复转换／pack；旧catalog每步187个矩阵均为BF16×BF16，66个wide的权重transpose只改描述，输出转换共811,008B／步。Hyle保留BF16中间存储点，参考主要保留F32结果、在MUL_MAT内转换输入，两者数学路径并非逐项相同。569个Hyle原生小图与参考整图执行的源码差异真实存在，但尚未单独计时；旧wide33.032ms不是转换／图kickoff的可回收成本。原region接线确实在工作，常量View发现调整的首轮完整结果如下，未改变数值合同。

[CPU常量View透明发现](../../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)仅改冷规划：无副作用／控制依赖、未发布且根为Constant的View链可参与连通性证明，View本身仍走原Graph元数据、alias与租约。每图command steps1195→1039、两variant的placement attempts1368→1056；已知编译字节119552→1893344，不能只报告步数减少。T154已知workspace22920576→16029056B、T1为438464→185536B；这些不是物理RAM节省公式。Windows原discovery、四模式View及逐节点BF16精确舍入检查通过，补测默认native1后四项薄入口全通过；正式Graph测试TU只编译，未完整链接运行。Mac五当前产品TU及薄消费者严格编译、7提交通过，仍使用冻结Graph／native／SDK；[Mac范围](../../evidence/correctness/inference-cpu-constant-view-macos-2026-10-26.json)不签完整构建、并发槽或吞吐。

新的九计时＋三独立内存进程全成功，控制／View候选／同版本AVX2参考的prefill43.453／43.953／42.937，decode11.479／11.437／12.027，综合3.299／3.323／3.314 tok/s。候选prefill＋1.15%，三点范围不重叠；decode−0.37%、综合＋0.72%均范围重叠，不签稳定decode收益。工作集采样峰5.320／5.002／3.828GiB，private commit6.065／5.740／0.606GiB；候选相对控制分别下降341282816／348819456B，各臂仅一次独立采样，不能归因于单个已知workspace。Hyle两臂线程峰17、GPU缺测。各臂完整非时间报告固定，Hyle只允许两处预先列明的workspace字段变化；三臂16个ID及工作行数相同，不代表中间精度或完整任务质量等价。加载／ready范围重叠，跨引擎边界不同，不排名冷启动。候选作为资源与prefill优化采用；CPU decode仍比同窗参考慢约4.9%，不与旧窗口拼算收益。

[状态与捕获有限源码调查](../../evidence/research/inference-state-capture-next-review-2026-10-26.json)已完成。参考的persistent pool／state indices属于原生kernel与权限合同，Hyle无现成indexed GDN／conv；现有ExecUpdate也仍先重录。实际GDN／history／copy分支不能给出完整参数角色映射，补齐要扩展launch／原生图接口，本轮不开发参数重绑。旧地址变化五次29.174ms只是历史API包络；旧录制60.245ms还含已消除的hidden19.474ms及必须保留失效门的native hardclear11.597ms，不能当当前可回收时间。没有支持这项改造的当前净收益量级，也不重开大块固定工作IO；下一批先核当前GPU执行与主机准备的占比。

加载期[host直接IO候选](../../evidence/performance/inference-cpu-host-direct-io-screen-2026-10-26.json)已关闭、不合入、不补样。私有实现复用实际CPU目的映射和原晚期Queue门，IO退休、flush／close及整组发布保持；修复coherent资源缺flush时的原路回退后，五个host TU及原FileStream Windows17项通过，独立真实行为工作不变。627次、3,763,655,360B显式复制降为零，但不能据此预测加载更快。

新六计时＋两独立内存进程全成功：加载中位0.895052→1.003500秒（慢12.12%）、startup到ready1.003448→1.116549秒（慢11.27%），三样本范围均不重叠。prefill43.331→43.359、decode11.416→11.419、综合3.283→3.290 tok/s均范围重叠；工作集各约5.320GiB、private commit约6.064GiB、线程峰17、GPU缺测。staging仍16MiB、逻辑backing预算仍约3.78GB；不猜加载变慢的原因，不将整进程CPU时间微降当能耗或decode收益。正常产品仍走原staging复制；合法只读映射导入仍为设计，须有真实mapping owner及原权限／预算接点，不能将借用视图伪装成所有者。

## 目标与当前基线

**最新短输入／多轮／MTP：2026-10-28 01:10:56～01:14:13，`b59fe852`／`433e7051`。** N1／R1、C1024、BF16、CUDA greedy、prefix off；单轮P154与两轮真实pending-flush配方保持原样，各格为Hyle／参考，速度为tok/s，显存为独立采样dedicated峰GiB。两轮不是B2多用户。

| 工况 | prefill | decode | 综合输出 | dedicated峰GiB |
|---|---:|---:|---:|---:|
| 普通auto单轮 | 2182.57／1544.85 | 94.18／95.92 | 86.54／84.47 | 4.034／3.728 |
| 普通auto两轮 | 2594.92／2519.85 | 96.17／96.74 | 89.84／89.54 | 4.165／3.728 |
| MTP单轮 | 2085.45／1537.94 | 120.02／129.59 | 106.89／109.20 | 4.091／3.896 |
| MTP两轮 | 2581.58／2297.29 | 125.17／132.43 | 113.49／117.87 | 4.226／3.896 |

本窗口ordinary off的单轮／两轮decode为91.75／90.77、综合84.49／85.17tok/s；auto实际发布55／110个后继，无拒绝或丢弃。MTP的自然接受与验证行数按各引擎实际工作计入，未人为改为相同行数；其decode仍落后约7.38%／5.48%，hidden接线与原生索引Store已在镜像中。各臂完整报告稳定，跨引擎第18位置的续文分歧已存在于前一d9窗口；不据此签质量或逐位数学等价。共同执行资源层、跨turn预热、MTP设备后继及GPU热点均纳入[当前goal](../../implementation/inference-roadmap.md#当前-goal-范围全部-review-应对)，不以本批有限结果结束工作。

auto／参考WS峰为单轮1.143／4.072GiB、两轮1.174／4.075GiB，private commit为5.811／4.673及5.992／4.675GiB；各项不相加成物理RAM。十个独立内存进程的ready基线都未知，实际采样间隔中位约15.09～15.25ms（请求10ms），峰值是采样下界。auto／参考ready中位单轮1.2509／1.6656s、两轮1.3075／1.6552s；原始范围、shared、phase变化和完整端点见[完整证据](../../evidence/performance/inference-hidden-state-current-window-2026-10-28.json)。未清OS缓存，不据此排名冷盘；不把减少捕获或跨窗TPS变化宣称为确定代码收益。

**最新长输入：2026-10-27 22:23:44～22:27:18，Hyle默认auto `aa864724`，参考 `433e7051`。** N1／R1、纯CUDA、普通greedy，固定相同输入、容量及整段prefill；12个计时与4个独立内存进程全成功，无新编译、下载、重跑或带仪器测速。每臂3次的完整非时间报告与输出ID均稳定，原端点与PSAPI／PDH采样器沿用，首对暂停检查后只续未运行项。以下均为各臂中位数：

| 输入／实际输出 | prefill Hyle／参考 | decode Hyle／参考 | 综合输出 Hyle／参考 | dedicated峰GiB Hyle／参考 |
|---|---:|---:|---:|---:|
| P1024／46（EOS） | 8025.670／5620.009 | 90.176／95.270 | 73.619／70.632 | 4.052／3.823 |
| P4032／64 | 9586.713／7771.359 | 87.331／94.731 | 55.914／54.060 | 4.698／4.161 |

速度单位tok/s。综合输出速度按实际输出数除以prefill＋decode连续区间，加载另列，不求各阶段TPS平均。P1024双方输出46且最后为EOS248046；Hyle实际提交38个后继、发布37、丢弃1，target输入1070行／47调用，参考1069行／46调用，额外已接纳工作全部计时。P4032双方4095行／64调用、64个输出，Hyle55个后继全发布、无拒绝；相同输出不代签完整隐藏态逐位等价或模型质量。

Hyle／参考进程WS峰为P1024 1.128／4.091GiB、P4032 1.136／4.242GiB；private commit为5.797／5.387及6.460／8.263GiB。dedicated、shared、WS和commit不能相加成物理RAM，独立采样峰为下界，ready基线均未知。Hyle长prefill独立workspace／slot的日志值为184.219／725.361MiB；Graph已有单Invocation内寿命复用，但各Phase仍各持工作区。这不能直接解释全部GPU峰差，原生／driver私有分配未逐项归因。

ready中位Hyle／参考为P1024 1.278／1.661s、P4032 1.279／1.695s；含启动输出速度24.081／19.891及26.342／22.177tok/s。OS缓存未清，不排名冷盘。输入是原154-ID prompt的固定重复，P4032不冒充4096；容量4096为最多64输出保留63个真实decode输入位置。当前长输入decode余差继续定位，不能只凭长度趋势认定是KV或主机成本。范围、采样缺值、实际命令与输入身份见[完整证据](../../evidence/performance/inference-long-input-current-window-2026-10-27.json)。

**以下短输入／多轮／MTP是21:27窗口，早于恢复修复，与上面长输入分开解读，不合样。**

后续[设施与参考实现审阅](../../evidence/research/inference-performance-review-2026-10-25.json)未找到每token重载权重、重建CPU池或复制完整历史的问题；本次CPU问题是大量持久池同时存活。旧轨迹晚段target GPU并集均值多约0.317ms，最大正差是66个N6144／K2048投影，draft略快；它不能解释完整墙时差距，也不构成重开已关闭叶候选的依据。CPU配置可达性此前已[补齐并做有限检查](../../evidence/research/inference-cpu-cold-options-2026-10-26.json)，本次CPU吞吐与资源结果另列于上文。

[同步报告改动](../../evidence/performance/inference-synchronous-step-report-2026-10-26.json)把同一调用内的格式化结果一次写入原stream，保留原端点、成功字节与顺序；同步29→1，局部新增4次／480B分配已计入host微测，25项合同检查通过。冻结be3消费者仅换report对象，普通／MTP各跑六个交错计时进程和两个独立内存进程，全exit0、非时间工作0差异。普通prefill／decode／综合为2812.809→2797.965、92.442→93.203、86.921→87.475 tok/s；MTP为2731.586→2715.660、119.916→121.072、109.795→110.770。普通decode三样本范围未重叠，MTP重叠，不签稳定MTP收益或统计显著性；两模式各自显存采样峰相同，RAM单次差不称节省。普通首次汇总的缺字段错误已从原日志离线修正，没有追加模型运行。原失败stream的部分前缀可能随写入粒度改变，不承诺任意facet／tie副作用逐字段等价。当前CPU新接口没有编进这份冻结CUDA消费者；两项检查范围分别保留。

[剩余热点筛选](../../evidence/performance/new-hotspots-2026-10-26.json)新增一个私有provider录制诊断进程：4446条完整、零溢出、非时间字段0差异。绑定／guard／收尾累计2.121ms，桥内13.170ms；21次有录制的prepare合64.369ms，另49.079ms位于provider计时外，具体归属未知，不称Graph CPU或可回收时间。旧trace的单次大图录制约406个GGML子调用并非406个kernel，不能用数量推动接口重构；旧profiler与新诊断也不能直接相减。单侧GDN地址固定无效，双侧暂存需约111.80MiB并每轮多37.27MiB搬运，本轮不推进。

[Graph分段诊断](../../evidence/performance/mtp-graph-prepare-attribution-2026-10-26.json)随后只替换当前映像的一个Graph host对象：唯一真实两轮exit0、所有非时间字段相同，68次prepare／132个range记录完整、零溢出。27次录制中，大图体10次的分发累计22.624ms、捕获结束16.756ms，前段17次另为3.181／0.871ms；全部录制后的真实等待共0.246ms。21个有录制decode的原cold为62.451ms，Graph部分50.954ms；45个无录制decode为25.615／11.284ms，其中冷绑定9.368ms（约208µs／轮）。父子桶不相加，不与旧provider诊断相减，也不能将捕获结束直接称Instantiate或可回收时间。原cold整段98.096ms中仍有34.477ms在Graph之外，包含上层layout／batch／input包络。现有revision／span复用已在使用，本次结果不支持新造dirty-bit或删权限／退休门。首次离线prefill掩码断言失败已保留，仅修解析器，没有重跑，不刷新上表TPS。

[Model准备分段](../../evidence/performance/inference-model-prepare-attribution-2026-10-26.json)复用已有观察接口，仅替换一个当前Model对象，唯一真实两轮exit0、完整非时间字段0差异，68次观察及408份target报告完整。此次同运行cold95.061ms＝layout8.768＋batch24.650＋replay61.626＋尾部0.017ms；66次decode分别8.441／16.328／60.361ms。target计划366复用／30新建；66次共享元数据上传仅1952B、各有一次真实host等待，不是六层各上传。45个无Graph重录轮次的270份target计划全部复用，排除稳态反复执行FlashInfer scheduler的解释。整个decode batch桶也仅占本次decode约1.64%，不能将其当作主要提速空间。可先核既有queue／copy依赖能否减少host往返，须保持写入真实退休后才发布ReadLease；不能只移动wait。replay仍是较大准备项。报告快照另0.961ms在原host端点后；本图为诊断，不刷新TPS，不与上一份运行相减。

[宽Lt叶筛选](../../evidence/performance/mtp-m2-wide-lt-screen-2026-10-26.json)对照两次N6144与一次N12288，保持BF16 A／W／Y、独立数值门、四个共192MiB物理bank及原工作量。中位114.400→115.040µs／对，五次范围重叠；图节点128→64没有提速，关闭，不继续投入共享子view和权重打包。后续typed MMF也已完成下述完整消费者筛选；两种原实现本来就使用TensorCore，不将其表述为首次启用。

[typed MMF叶](../../evidence/performance/mtp-m2-typed-mmf-screen-2026-10-26.json)随后保原BF16 A／W／Y与F32累加，数值门通过，两份输出相对Lt有13／11个BF16位差。局部中位127.376→115.408µs／对，但五次范围明显重叠。其[私有完整消费者](../../evidence/performance/mtp-m2-typed-mmf-consumer-screen-2026-10-26.json)仅换一个cuBLAS provider对象、加入原typed native对象，原stream／指针／context／错误前缀边界全部保留。一次真实两轮行为和八进程配对筛选均exit0、所有非时间工作0差异；decode121.129→121.999（+0.72%）、综合111.113→111.659（+0.49%），prefill2764.530→2721.230 tok/s。全部范围重叠、显存采样峰相同，单次RAM差不作节省结论；关闭，不合入产品或追加样本，不刷新正式四臂表。

[K2调用源审](../../evidence/research/mtp-k2-source-review-2026-10-26.json)确认旧193次调用＝2prefill＋48多行验证＋95proposal＋48repair；143个辅助调用仍有真实自回归与正式KV修复依赖，不能直接删掉。T3验证／三前缀GDN状态已经接通，后续只筛选辅助图捕获，未重开批宽功能。

[辅助捕获筛选](../../evidence/performance/mtp-k2-aux-capture-screen-2026-10-26.json)的两次行为和八进程配对均exit0、完整非时间字段0差异；实际143个辅助调用有78次录制／65次复用，未减少原提案、修复或权重读取。decode中位123.582→125.032（+1.17%）、综合113.304→114.176（+0.77%），范围重叠；prefill2879.060→2726.346，三样本不代签稳定回归。单次显存采样差不归因节省。候选带少量POD计数、原时钟端点未移，不称完全无插桩；关闭，不默认启用、不追加样本或刷新四臂表。

[冷replay一致性](../../evidence/research/inference-replay-policy-consistency-2026-10-26.json)单独修复配置接缝：Phase与Draft共用冷selector，辅助Auto保原Commands，显式prefer传入Graph原fallback；采样强制Commands例外保留。当前Model严格语法检查通过，未新链接／运行或宣称提速。CPU配置亦已在原有限预算工厂[补转发](../../evidence/research/inference-cpu-options-forwarding-2026-10-26.json)，不更改默认线程策略。

目标是查清并消除可重复的额外成本，在相同权重、输入及明确的计算和完成边界下追平 llama.cpp，
再根据实际热点争取反超。同时保持数值、状态、寿命和资源限制正确，检查 prefill、启动和内存是否退化。
每批改动以真实 tokens/s 判断收益；kernel 数减少和微基准变快只用于解释。

**2026-10-27 21:27:15～21:38:23：最近完整窗口使用Model3 image `fefa445f`，main源码 `b4e72daa`；参考保留 image `433e7051`，不是最新上游。** 单轮P154＋64输出；真实两轮同一会话各追加P154、各生成64。普通off、普通auto、MTP K1、参考普通、参考MTP共五臂；每个场景15个交错计时＋5个独立内存进程，合计40进程全部exit0。逐臂完整非时间报告稳定，普通off／auto的所有输出ID及实际工作相同。

Model3仅修正跨pump统计的prefill观察点，并检查实际发生的P／D有正完成区间；另在离线解析核有限正值及P＋D≈session，不能采用修正前Model2的吞吐。只重编一个Model C++单元并链接一次，复用其余223个链接输入、SDK、原生内核及权重，无下载。无profiler、额外推理预热或旧样本混入；使用原greedy／EOS及完成端点。两轮第二进程完成后，原解析器错把末态available=false视为全程未接力，暂停测量；修正解析器，保留原输出，只顺序续跑第3～15项，未补跑前两项。时间跨度及暂停作为测量限制保留。

| 负载／模式 | Hyle prefill | Hyle decode | Hyle 综合 | llama.cpp prefill | llama.cpp decode | llama.cpp 综合 |
|---|---:|---:|---:|---:|---:|---:|
| 单轮普通 off | 2206.734 | 94.613 | 86.997 | 1558.635 | 96.830 | 85.168 |
| 单轮普通 auto | 2152.158 | 97.547 | 88.981 | 1558.635 | 96.830 | 85.168 |
| 单轮 MTP | 2088.288 | 117.616 | 105.024 | 1474.762 | 131.923 | 108.811 |
| 真实两轮普通 off | 2587.049 | 91.129 | 85.214 | 2457.669 | 96.564 | 89.393 |
| 真实两轮普通 auto | 2622.504 | 92.714 | 86.554 | 2457.669 | 96.564 | 89.393 |
| 真实两轮 MTP | 2458.208 | 118.571 | 107.751 | 2291.911 | 132.739 | 118.047 |

单位tok/s，每臂3个独立进程的中位数。单轮auto相对off的decode＋3.10%、综合＋2.28%，但decode范围92.709～98.128与off93.661～94.915重叠；参考为95.174～96.956。单轮auto中位高参考约0.74%，不能宣布稳定反超。两轮auto相对off的decode＋1.74%、综合＋1.57%，三点范围未重叠，仍不签统计显著性；相对参考decode仍慢3.99%。普通off单轮／两轮慢2.29%／5.63%；MTP慢10.85%／10.67%，本次普通流水并未覆盖MTP。

单轮auto实际提交／发布55次、拒绝0；两轮59次、拒绝1、丢弃0，末态available=false。它表示已经安全回退，不能写成全程没有流水，也不能宣称全程连续。日志把转换定位在二轮长度384→385附近，具体Unavailable原因尚未记录；捕获tuple跨页变化仅为假说。当前代码在一次拒绝后永久停用可选接力，恢复策略属于后续检查项。

| 负载／模式 | dedicated 采样峰 Hyle／参考 | 进程 WS 采样峰 Hyle／参考 | private commit 采样峰 Hyle／参考 |
|---|---:|---:|---:|
| 单轮普通 off | 3.876／3.728 | 1.056／4.071 | 5.507／4.673 |
| 单轮普通 auto | 3.899／3.728 | 1.142／4.071 | 5.678／4.673 |
| 单轮 MTP | 4.030／3.896 | 1.142／4.221 | 5.759／5.082 |
| 真实两轮普通 off | 4.009／3.728 | 1.085／4.075 | 5.672／4.675 |
| 真实两轮普通 auto | 4.034／3.728 | 1.174／4.075 | 5.864／4.675 |
| 真实两轮 MTP | 4.169／3.896 | 1.194／4.224 | 5.979／5.086 |

单位GiB。独立PSAPI／PDH旁采峰值是下界，ready缺样保持未知；WS、commit、shared和dedicated不能相加成物理RAM。单轮auto相对off的dedicated峰多24MiB，ready中位1.246→1.297s，含启动输出速度32.292→31.744tok/s；两轮ready1.270→1.314s、含启动45.961→45.831tok/s。decode小收益不等于短请求冷启动整体改善。其它加载、范围及实际采样间隔保留在[本轮证据](../../evidence/performance/inference-continuous-wave-current-window-2026-10-27.json)；OS缓存不清，不排名冷盘。

两轮prefill按309行真实target输入计速，另列308枚新增用户token；decode按126枚输出，综合按128枚输出／连续会话时间，加载分列。Hyle MTP接受60／66、验证132行、总target／draft为441／434行；参考接受61／64，另有一次无proposal尾步，总65次decode、验证129行、总target／draft为438／435行。同模式双方从第19个输出起续文不同，不能称逐token数学工作相同。

18:01旧32进程结果及其源码身份继续保留在[上一完整窗口](../../evidence/performance/inference-batch-continuation-current-window-2026-10-27.json)，不与本窗合样或据绝对速度变化归因代码收益。旧文中“四臂／最新完整”等描述仅指各自当时记录。

以下归因沿用旧 c343 窗口及其对应轨迹。原报告可进一步算出：两轮 MTP decode 墙时为 1014.980／899.352ms，除以各自 66／65 次调用后，每 episode 为 15.3785／13.8362ms，相差约 1.5423ms。总耗时比可精确分解为 `1.128568 = (66/65) × 1.111469`；次数多 1.54%，平均 episode 多 11.15%。这是代数分解，episode 还含 draft、冷准备、状态、等待与主机工作，不能据此称同一个 GPU 任务贵 11.15%，也不能把次数因子当作接受率全部因果影响的上界。后续诊断按等待完成、读回、导出、权限与发布分别计量；原完整测速结果保持不变。

[等待后分段诊断](../../evidence/performance/mtp-finish-phase-attribution-2026-10-25.json)新增一个 Hyle 插桩进程，128 个输出及所有非时间字段与原基线一致。66 次 decode 的等待后六段合计13.350560ms，即约0.202281ms／轮；包含132次 D2H API及0.149311ms可见 GPU 活动，不能全称为 CPU。等待返回尾部未见固定1ms轮询，末 GPU 活动至 EventSynchronize 返回中位19.704µs、最大48.522µs。

同66轮的 prepare 为137.692ms，其中 replay 子区间107.097ms；45个无重录轮次的墙时中位14.806ms，11个重录但无实例化轮次18.890ms，10个含实例化轮次21.120ms。此闭包已含 seed 修正，实际13次实例化、19次更新，记录的 API 返回码均0。较大的余项仍与冷准备／重录相关，完整变化已按下述诊断拆开。

[全端口诊断](../../evidence/performance/mtp-complete-binding-diff-2026-10-25.json)完整记录5005事件、80个有效 key miss及2701个变化端口；所有变化均为同一 backing 的 offset，128个输出和非时间工作字段与原基线一致。132次范围决策中重录32、变体命中90、直接保留10。前段53／55步共重录17次，其中12次在已观察的比较中只差 hidden 输入；其余5次是首次或显式失效。固定 hidden 输入不会直接消除后段1269步的15次重录，不能把反事实比较当成未来缓存轨迹。

**固定 hidden 入口候选已关闭，未合入产品。** 私有候选为 carry／prior 复用原 CopyBatch、固定 writer 与真实退休门，额外固定 payload 为12,288B，每轮复制4／8KiB。这些是源码推定载荷，未测物理 API 次数或带宽。两份当前 app 严格编译、链接通过，其余230个链接输入相同；实际小段录制17→5、总录制32→20，大段仍15次，全部非时间报告字段相同。原观察的12个候选录制及后续原生 API 窗口仅4.259603ms，不能把107ms replay 都归给它。

随后换回原无诊断 Graph，仅做三对交错独立进程：prefill 中位2878.017→2793.682、decode125.317→125.325、综合115.005→114.568 tok/s；六次均 exit0、全部非时间报告0差异。decode 范围重叠，仅+0.006%，没有吞吐收益的依据；不把 prefill 的三样本下降签为稳定回归。没有新内存旁采或参考进程，不刷新上面的完整性能表。未报告的完整设备状态和旧读者故障场景未验证，不能当作普遍正确性证书。详见[固定入口取舍](../../evidence/performance/mtp-fixed-hidden-ingress-2026-10-25.json)。

同一旧 trace 的离线细分进一步定位：17个前段录制窗口合计6.308232ms，15个后段大图体窗口合计78.431404ms，其中捕获区间58.097852ms、后续实例化／更新 API 为20.322897ms，彼此嵌套而不另加。32个窗口外还剩22.356950ms replay，共用准备与未标记间隙无法逐段分配；prepare 另有30.595468ms位于 replay 外。候选12段紧随的销图、记录事件、等待返回窗口仅0.224217ms，未观察到 eager GPU 预热。replay 可见 GPU 活动并集为零，prepare 为0.438298ms；56.628347ms属于 submit，不能算作冷准备 GPU 成本。这些是已有插桩的墙时／API归因，并非可回收收益。后续性能阶段重点为大图体状态版本与重录，需保留旧读者和真实退休，并计入稳定入口可能引入的大量搬运。

[15次图体逐窗因果复核](../../evidence/performance/mtp-body-recapture-cause-2026-10-25.json)只重读上述旧源轨迹，没有新模型／GPU／测速。它们是 CUDA Graph 重录，不是权重重新加载：2次首次录制共18.186282ms，10次有效key的地址不命中共48.647924ms，3次native计划失效共11.597198ms；这些是原捕获至后续原生API结束的窗口，不是可回收时间。18层GDN已记录的同类池offset模式同步，未发现各层分散拼出新组合。seq37／38／43均实际由六个attention native步骤清除完整key、保留非可提交seed；seq37的128行chunk数2→3有旧同work逐query证据，seq38／43分别跨轮和384行阈值的定位是源码／layout推断，不能称本trace已记录其native拒绝原因。

其中seq13／14／15／41对最近合法变体仅差1–2个hidden**输出**offset，共15.354240ms。保留全部其它完整tuple和真实clear的离线轨迹中，固定这些输出还避免seq16的替换连带重录，合计15→10次，对应五窗19.474001ms；不是已执行候选或提速预测。剩余GDN变化主要是两role第2变体的初次填充及硬失效后重填，不支持新造按层pool机制。最小待验证方向是固定工作输出、再经原CopyBatch把选定carry／prior复制到独立canonical lease，保旧reader、预算与真实退休；它与已关闭的hidden**输入**入口候选不同，也不需要先搬运整份GDN状态。

[hidden输出优化](../../evidence/performance/mtp-fixed-hidden-output-2026-10-25.json)随后按上述方向实现并合入：只在现有K1、PreferCapture和实际caller-owned输出资格下，原Graph写固定Phase工作块，接受前缀确定后由原CopyBatch复制选定carry／prior进原canonical池。额外固定payload为16KiB，每轮4／8KiB，GDN和输入不变；均为源码逻辑字节，不是物理带宽测量。六次交错测速及两次独立内存进程均exit0、完整非时间报告0差异；原stderr实际总重录32→27、复用100→105，仅五个预测位置变化，没有新增profiler。日志未单列range身份，故不把新body15→10写成直接观测。

该配对筛选decode中位118.912→120.443（+1.29%）、综合108.891→109.973（+0.99%）tok/s，三样本范围均未重叠；prefill2635.934→2704.143但范围重叠，显存采样峰相同。保留首轮严格编译的C4458遮蔽告警及改名修正，最终主干另重新编译应用入口、复用其余233项链接输入。它是有限正向筛选，不声称统计显著性；上表为随后独立完整窗口，不把旧窗口与新窗口之差全部归于本次改动。未新增故障注入、其它后端或完整隐藏张量检查。

分页与地址稳定分别处理：Hyle目标模型已经使用PagedArena／页表和PagedAttention；页数增加仍可能跨过split-KV执行分块阈值，13→24页导致128行chunk数2→3即属此例。已核源码中，llama.cpp用持久GDN bank、索引gather及快照写回，carry还经过host暂存；TensorRT-LLM用状态池、state_indices和接受状态promote；vLLM用固定输入／输出buffer与shape buckets复用CUDA Graph。它们均有明确的复制和寿命成本，不能用其可变bank或复用输出直接覆盖Hyle旧Snapshot。相关版本、源码锚点和限制并入同份优化证据。

后续候选先比较对手实际次数，再估计能影响的工作。旧同负载逻辑轨迹中，参考 target／draft 分别录制2／5次，合计7次；2次实例化与7次 Update 都属于这些录制，不能相加成9次。Hyle 修正 seed 后仍录制32次，对应13次实例化、19次 Update。参考每轮首个T2 target先eager执行，稳定后下一次才capture；draft的首形状／变化后也有eager，均保留在原decode端点内，不是免费热身。Hyle 前段／图体和参考 draft／target 的切分边界不同，实际 decode 调用为66／65、自然接受与输入行数不同，不能按次数比例预测提速。Hyle 图体的两种 range 各首次录制共18.186282ms，另外13次合计60.245122ms；只有确实消除相应录制且扣除新增成本的改动才可能影响该子集，不直接从无插桩耗时中扣除。

增加图变体容量在此路径没有依据：对已有66次图体查询逐端口重建完整2031端口 key，保留每次真实失效，容量2、3及不淘汰的理论缓存均为15次录制、51次命中，未知比较为零；原容量精确复现全部录制位置。关闭容量扩展方向，继续调查完整状态绑定与实际失效的原因；离线等价关系推断不代表候选实际运行或资源寿命证明。详见[对手次数与离线设计](../../evidence/performance/mtp-body-replay-design-2026-10-25.json)。

[状态IO离线反事实](../../evidence/performance/mtp-state-io-offline-2026-10-25.json)已先逐条复现原15次录制／51次命中。仅规范化有实际外部状态来源的GDN端口offset后为12／54，再包括独立发布的carry端口为5／61；所有真实hard-clear及其它端口保留，未合并ValueId或授予别名权限。剩五次为两个首录及三个图体hard-clear后首录。固定地址的实际搬运、额外存储、旧reader与退休仍待实现设计；此结果没有测速或收益结论。

源码实现审查表明，自然K1路径在`finish_mtp_rounds`中先选择接受前缀，再由`finish_selected_evaluations`导出选定状态，可沿现有CopyBatch和真实Completion实现固定IO。代价并非只有地址改动：18层GDN加carry的额外输入／选定输出搬运约37.27MiB／轮；若独立保留两候选则约55.91MiB／轮，两个phase的完整固定工作块可另占111.81MiB。以上是源码几何核算，未测耗时，不包含控制块和对齐等成本。

旧Snapshot仍须持有独立的canonical状态，不能指向下一轮会覆写的固定scratch；任意Graph输出reader外逃时仍保原存储合同。copy写入成功且真实退休后才能发布状态，失败／取消由原保活与退休链承载；新增工作块须经过既有资源接纳，不能暗中提高cache或workspace上限。功能／架构复审后再决定是否实现有限候选，不以15→5直接推导提速。

本次612个 NVTX 阶段完整；Nsight stderr 实际截断，后60条 POD metadata 标为未知。所有时间属于插桩诊断，不能与旧无插桩1.5423ms差额硬相减，也不把嵌套 replay 或 wall 减 GPU union 当作可回收收益；未刷新 TPS。

当前 CLI／report 与本 BF16 路径实际变化的 4 个 Graph／batch／CUDA region 对象已刷新，复用兼容的 native 数学；不是全 SDK 或其它量化组合的重新认证。V2 清单 335 个角色与旧窗口逐项匹配原 dtype、形状和源范围，普通仅选择 320 个文本角色。本窗是优化前同窗基线，与 [10 月 24 日窗口](../../evidence/performance/qwen-mtp-single-and-multiturn-2026-10-24.json)分开，不据跨日差值宣布架构收益或回归。

**同日准备诊断另列，不能混入上面测速。** 普通／MTP 各一次短入口和一次真实两轮诊断通过；私有时钟数据在 run 返回后输出。

| 末 24 次调用的诊断 | 普通 | MTP |
|---|---:|---:|
| prepare 中位，µs | 337.75 | 480.95 |
| 其中 replay 中位，µs | 153.45 | 203.95 |
| submit 中位，µs | 72.75 | 92.40 |
| prepare＋submit 总量／调用墙时总量 | 4.191% | 4.238% |
| capture recorded／reused 总计 | 0／24 | 1／47 |

普通一次 decode 与 MTP 一次验证的工作不同；这些不是每 token 成本，子项中位不能相加。finish 含 GPU 等待与发布，该比例也不涵盖全部主机工作，不是可回收收益。当前 replay 源码窄审没有发现足以直接删除的重复准备：绑定／权限、只读 seal、队列、真实退休和失败事务仍需保留，warm qualification 和 record 已复用。准备失败及实际范围见[当前诊断](../../evidence/performance/mtp-current-prepare-diagnostics-2026-10-25.json)。

**同日四臂 GPU 轨迹与默认 IO 诊断已完成。** 普通 captured target 的单图 GPU busy 中位约 Hyle 9.510ms、参考两种图 9.498／9.513ms；后 24 个 target 区间内，Hyle 无 GPU 活动的间隙较多。MTP 后 24 区间两边均含 24 target＋23 draft，未见 Hyle 有更大的稳态主机空洞。MTP target 图 busy 中位约 10.876／10.425ms，主要正文投影组约 6.734／6.387ms，head 约 2.331／2.346ms。它们只说明当前物理路线值得检查，不签同数学、因果瓶颈或可回收收益。

Hyle 早期两次 target 间隔含真实 draft，约 12.02／9.57ms 没有 GPU 活动，并伴随录图／实例化；首个间隔的 `cuMemAlloc` 合计约 0.255ms，不能把整个间隙说成分配。参考首个录图前还有 eager target／draft，双方前 24 个录图区间的真实起点没有对齐，不能据该表比较冷启动胜负。四条轨迹另有 driver 13.4／工具 13.3 警告，MTP 原图身份查询报错；活动关联可读，完整性与图谱系没有认证。保留这些限制，不重采矩阵或用 profiler 时长替换前面的无插桩 TPS。

逐步报告函数的两次私有诊断保持原文本、顺序和工作量：普通 128 次合计 17.03ms、MTP 68 次 9.75ms，中位约 130／138µs。原计时包住向 `std::cerr` 写入的完整函数，包含同步stream写入及其触发的缓冲／flush开销；只是没有单独观察OS write／flush次数，不能称为纯格式化成本。它不含函数外的入口日志，也不是全部主机成本；参考同样有逐步日志，不能仅单方关闭IO来归因底座速度。完整 trace、IO、源码窄审与口径纠正见[当前成本归因](../../evidence/performance/mtp-current-cost-attribution-2026-10-25.json)。

[控制传输复算](../../evidence/performance/inference-control-transfer-2026-10-25.json)直接复用上述四条轨迹，无新模型或 profiler 进程。Runtime 表的 `eventClass=1` 实际记录了可见 Driver API，不能因没有独立 Driver 表就认为这些上传不可观测。普通／MTP warm 窗中，直接 H2D＋显式流同步 API 的区间 union 为 1.385／1.875ms，占各自窗口 0.528%／0.531%；参考同窗的 H2D 次数及字节反而更多。逐端口同步值得简化，但目前不能列作约 10% MTP 差距的主因或承诺收益。窗口排除首 target 前上传与末 target 后读回，保留原工具诊断和实际工作差异；API 等待、GPU 活动与图内复制分别计量。

metadata 候选已关闭：当前 CUDA 路线实际共享 source，5 个 follower 的冷期多余预留在完整 MTP 四个阶段合计仅 22,560B；不是每 token 上传或分配，保留原有准备和失败边界，不为此引入懒分配机制。

[整数暂存复用](../../evidence/performance/inference-integer-scratch-reuse-2026-10-25.json)已接原 `InputPreparationScratch`／`SmallVector`，未复用不可变版本或页读者。原 R1／K1 两轮的相关构造／扩容计时累计约 0.1071ms，含计时开销、不含析构，不能解释约 10% 余差；候选的完整输出、ledger、dispatch 及全部非时间字段保持一致。合并新精度后的入口仅做严格语法检查，没有开新的前后测速窗，不报告 TPS 收益。

[按实际逻辑调用标记的两轮诊断](../../evidence/performance/mtp-logical-call-attribution-2026-10-25.json)已补齐每端一个 NVTX／CUDA trace，全部非时间输出和工作与各自基线一致。
Hyle 的 66 次 decode 在 `prepare_replay` 累计 107.037ms，该范围内实际 GPU 活动并集为零；13 个含实例化的调用占 78.249ms，8 个仅录制的调用占 16.324ms，45 个无录制的调用占 12.463ms。
同一 trace 的 32 次录制中，19 次直接实例化、没有尝试 Update，另 13 次 Update 返回 0 且没有实例化回退；因此不能称为更新失败重建。
参考对应 2 次实例化及 7 次成功 Update。原生 API 耗时嵌套在上述范围内，不相加；两个 profiler 进程不是新的 TPS 窗，也不能用 107ms 直接替换历史墙差或承诺可回收收益。
[重建来源诊断](../../evidence/performance/mtp-capture-seed-attribution-2026-10-25.json)已进一步分清：19 次无 seed 为 4 次 Invocation／range 首录、9 次第二变体填充、6 次显式清除后的首录；9 次填充中又有 5 次是显式清除后的重填。
修正后的单次诊断有 627 个事件、零溢出，全部非时间工作与原基线相同；第一次未输出事件的仪器失败保留，不能当作零事件。
[原生重建存储复用](../../evidence/performance/mtp-native-rebuild-seed-2026-10-25.json)已合入：只在 native commands 显式失效时，经既有后端授权保留已实际退休、永久不可再次提交的重建存储；稳定绑定重验只保留已有 seed，Queue／配置／import／owner 变化仍硬清除。所有 key、绑定和执行资格照常失效，不保留旧参数权限。
定向 host 检查及原 K1 两轮行为通过；独立 CUDA API 对照中，实例化 19→13、成功更新 13→19，录制仍各 32 次，输出及全部非时间工作相同。它解决了六次可避免的实例化，不能把嵌套 API 时长或上述 107ms 当作可回收墙时。
随后只做三对交错计时及每臂一个独立内存进程：prefill 2745.82→2821.49、decode 124.10→125.12、综合 113.49→114.71 tok/s；范围全部重叠，**吞吐收益未证实**。两臂 dedicated／shared VRAM 峰分别相同，单次 RAM 差不作节省结论。该候选按冷分配复用保留，停止补样；约 10% 的主要差距仍需另查。这是 Hyle 前后筛选，沿用原消费者对象，不刷新最新普通／MTP 与 llama.cpp 的完整窗口。

[单行 F32 TopK1 捕获候选](../../evidence/performance/mtp-f32-single-row-prepared-screen-2026-10-25.json)已关闭、未合入。仅替换一个 CUDA 对象，沿用上述重建存储复用后的同一消费者；数值／双队列捕获四项及原 K1 两轮行为通过，随后三对交错测速和两个独立内存进程全部正常结束，输出与实际工作相同。decode 中位 124.754→123.441 tok/s（−1.05%），综合 114.087→113.374（−0.62%），两组范围均未重叠；prefill 2759.82→2830.06 不抵消 decode 的负向结果。每臂只有三个计时样本，不声称统计显著性或新的参考引擎差距。

该候选把每次 T2 的两段 hybrid 合成一段 captured，但实际录制／复用总计是 32／100→21／45，并非简单减半：原先十次只需重录其中一段，合并后须重录整段。没有新 API trace，不能据范围数推导原生 launch 数或断言退化根因。两臂 dedicated／shared VRAM 峰分别相同；单次 RAM 差不作节省结论。保留原 F32 路线，停止补样或扩展正式对照；源码、首个私有测试编译错误及全部样本均保留。

[M2 同输入双原权重投影候选](../../evidence/performance/mtp-m2-two-original-weight-pair-screen-2026-10-25.json)也已关闭、未合入。原 BF16 A／W／Y、F32 累加下，复用既有归约主体同时计算 gate／up；两臂 direct 和首次 captured 结果均先通过独立固定数值门，再测五次交错叶图。每图 64 对投影，节点 128→64，耗时中位 116.224→118.992µs／对（+2.38%），范围重叠；两份输出相对 Lt 有 13／12 个 BF16 位差，不声称位级等价。四对 bank 共192MiB，A 为原 prefill trace 首两行，未测当前 verifier 输入或完整消费者；停止追加样本，不从节点数推断 decode 收益。

自然请求循环的单成员桶现交给 `execute_generation_calls` 原有权威门禁检查，省去外层一次相同的全请求预算查询；多成员桶仍按原规则缩批。合并接口已严格检查，未开新吞吐窗口，不把少一次扫描换算为 tok/s。

[绑定诊断](../../evidence/performance/mtp-capture-binding-diagnostics-2026-10-25.json)进一步分开两种重录：132 次范围决策中录制 32、复用 100；58 个已记录的首不等端口全部只改变同 backing 的 offset，hidden 出现至少三种地址组合，两份变体遇到淘汰后回访。另有两次 prepare 之间明确失效后重新录制，不能据相同指纹删除该边界。首次诊断溢出及一次修正均保留，修正版零溢出、全部非时间工作与基线一致；只细记各 Invocation 前 20 次准备，不把首不等端口当作全部差异，也未观测 native GraphExecUpdate 结果。

[本批候选筛选](../../evidence/performance/mtp-state-and-draft-candidate-screen-2026-10-25.json)已关闭倒序 hidden 分配：仅编译一个私有 app TU，原两轮的一次诊断结果和工作完全一致、零溢出，但重录 32→39，复用 100→93；不做 TPS 重复或合入。配对 hidden 父块现有别名 API 足以表达，但涉及独立请求的持有粒度、prefill／单行初始化及碎片、多 K 预算和诊断寿命，当前没有收益依据支持扩大改动，也不留下挂起的布局任务。原权限、预算、显式失效及两变体政策保持不变。

另一次当前 clean image 的 K2 两轮探索保留 128 个相同输出，但验证调用 66→48、总 Graph 调用 68→193、draft 行 434→571。单次 prefill／decode／综合为 2879.81／123.44／113.47 tok/s，未见足够理由替换 K1；它不是交错统计窗口，参考仍 K1，不能称为同 K 性能胜负。

[正文路线归因](../../evidence/performance/mtp-body-route-attribution-2026-10-25.json)已统一为两端各自末 24 个 captured T2 target：两组正文投影的活动均值 Hyle／参考为 6.665／6.470ms，其中 66 个 N6144、K2048 投影多 0.209ms，另一组抵消 0.014ms。target 全部活动的 busy union 均值为 10.800／10.495ms；kernel、copy、图内间隙和各组中位分别保留，不相加成可回收速度。此前仅取末单图得到相反方向，现已明确纠正；本窗也不与全程均值或中位混算。BF16 A／Y 与参考 F32 载体、不同续文及 eager／图身份限制仍在，不能据此解释全部完整 decode 余差。

同形状 whole-M2 typed MMVF 的一次叶筛选已关闭：原 gate 输入首两行、四 bank、64 leaf 和固定数值门下，Lt／候选中位 58.800／58.240µs，仅差 0.95%，五次范围明显重叠；两者数值门通过，候选有 13 个 BF16 位差。未接完整消费者，也不继续补样或扩融合。归档 runtime 表另记录 13 次 GraphExecUpdate、19 次 Instantiate，API 返回码均为 0，区间合计分别 1.657／22.441ms；driver 表及 UpdateResult 输出未记录，不能与其它时长相减或声称这些 Update 报错后重建。

[原生计划诊断](../../evidence/performance/mtp-native-plan-diagnostics-2026-10-25.json)已实测 474 次 query：427 次复用、26 次首次准备、21 次合法分块变化。第二轮首次 decode 的七个 slot 均因 KV 页数 13→24、128-row chunk 数 2→3 而重建；六个 target 的 query 仍为 2，draft 为 1，schema／generation 没有失效。另 14 次晚段拒绝的全程原因也是 chunk_count，未记录其具体序号，不补造。模型输出和所有非时间报告与基线一致，原权限、退休及 chunk 保护继续保留。

固定分块候选已关闭、未合入：短 query、单请求 BF16 paged attention 在 query 与 plan 共用一个冷策略，尝试固定 512 个逻辑 KV 行（page16 时传 32 页），保持现有复用判定和所有权协议。[有限数值检查](../../evidence/correctness/mtp-fixed-chunk-numeric-2026-10-25.json)的四个场景均通过原 double oracle；新旧 BF16 输出有小差异，不保证位级相同，也未单独认证 capture。

固定分块不能直接扩大到所有上下文：H8、D256、262144 行时将需要 64.25MiB 临时空间，超过默认 64MiB。候选若保留，须按冷期容量选择有限适用域，其余保持原自动规划；不能随当轮长度跳档、放宽预算或删除复用保护。不启用需要额外 mask／计划字段的 CUDA graph 规划模式，也不以减少重录次数冒充 tokens/s 收益。

无插桩的普通完整运行已出现输出分叉：前 18 个相同，第 19 个 271→11，随后 104／128 位置不同；两端均正常结束，所有差异均为 token 派生字段，实际行数、调用、预算和其它非时间字段相同。原严格一致性筛查保留首失败，原 12 进程计划只运行 2 个，未继续普通重复。数值门通过不证明任务质量或相同续文。

随后单独冻结的 [MTP 六进程窗口](../../evidence/performance/mtp-fixed-chunk-screen-2026-10-25.json)按 B,C,C,B,B,C 交错，全部输出与非时间工作相同。decode 中位 124.81→126.55 tok/s（+1.40%），综合 114.29→115.71（+1.24%）；prefill 2856.83→2813.53（−1.52%）。每臂只有三个样本，保留范围，不声称统计显著性；没有 peer 或内存采样，不替代上方正式四臂窗口。小幅短上下文收益不足以支持本轮增加冷容量政策及任务质量边界，候选明确关闭，不留作待验证项。当前产品继续使用原自动规划；候选的长上下文风险和续文分叉未进入产品。

[10月24日MTP节点诊断](../../evidence/performance/qwen-mtp-current-cuda-structure-2026-10-24.json)已各跑1个插桩进程。Hyle132次Graph launch对应66轮的draft／target两段；参考116次只覆盖部分调用，其余走eager，不能直接按launch数比较总工作。目标图738／727个kernel，草拟图37或41／34个；Hyle目标图另有135次D2D、载荷中位约39.73MB，拷贝活动中位约0.137ms。它是后续减少搬运的调查入口，尚未证明可删除，也不足以解释全部速度差。

该轨迹中Hyle的2458次`cuMemAlloc`有2357次在首次decode launch之前，余下101次集中在首次shape／slot及第二轮边界；最后24次decode没有新增分配或图实例化，未发现“每token重新分配”的事实。runtime与driver分配API可能嵌套，分别计数。两臂节点采样有额外开销，静态下载仍在后台，图内统计也不含eager工作；此诊断不替换上面的正式速度窗口，不把余差全部归于分配、capture或某个kernel。

[快照复制源码核对](../../evidence/performance/mtp-qwen-snapshot-copy-review-2026-10-24.json)已关闭“直接删除36次1MiB复制”的候选：18层各自把两个GDN快照物化为可独立回收的caller状态块，保留接受前缀和旧读者；这36项的GPU活动中位60.75µs，与全部135项的136.88µs分开计。共享2MiB父块会使选中一半仍持有整块，N=R=1时每层下一轮需4MiB，超过当前3MiB池；同容量下18层共需增18MiB。保留默认独立快照，不新增拆分租约抽象或扩展K1融合；本次只核既有trace与源码，没有产品改动或新吞吐测量。

[MTP准备分段诊断](../../evidence/performance/qwen-mtp-prepare-segments-2026-10-24.json)另复用现有计时点，仅增量编译一个scratch消费者。实际两轮128输出、132行verify、66轮／60接受与本窗Hyle工作一致；最后24轮prepare整段中位560.55µs，内含layout102.55、batch151.5、replay准备238.3µs，submit中位92.85µs。batch包含重置配置及append绑定，未单独测得全图扫描成本；finish含GPU等待，各段中位不可相加。后台下载和端点后的诊断输出仍可能影响调度，不与另进程trace的空闲区间相减，也不作TPS或可直接回收时长结论。

历史[AV组装窗口](../../evidence/performance/qwen-mtp-assembly-2026-10-22.json)及[29项相关检查](../../evidence/correctness/mtp-consumer-wiring-2026-10-22.json)保留输入ReadLease候选表懒建、F32 Add对齐与冷alias准入的原结论；[AU有界capture](../../evidence/performance/qwen-capture-variants-2026-10-22.json)保留两份变体及真实失效／权限／退休门。这些历史实现与结果不充当本窗热点比例或新增收益证明。

变体实现前的[每步诊断](../../evidence/performance/qwen-capture-recording-cost-2026-10-22.json)中，63步prepare总113.41ms，新record86.17ms，占75.98%；中位prepare／record／bindings／end为1.7801／1.3629／0.1677／0.2202ms。GGML当时未每步重建原生图或重分workspace；72份caller状态端口仅两组物理tuple交替，由此进一步核全参数和私有资源后实现本批变体。该历史比例不用于解释当前全部余差，带日志的4进程也不混入正式TPS。

此前热点候选调查及最后B07原制品CPU／CUDA有限流程已完成。Lt冷算法筛选与typed MMVF整消费者候选均已拒绝，没有在这些旧候选上挂起等待验证。当前普通／MTP余差继续按本页新窗口调查；有限制品检查不代签完整模型。原稳态矩阵投影占累计kernel时长92.466%，该历史GPU占比不能解释当前完整wall差距。最新prefill、decode、启动、空间和完整边界统一见[最新完整窗口](../../evidence/performance/mtp-current-single-and-multiturn-2026-10-25.json)。

[早期carrier窗](../../evidence/performance/qwen-mtp-divergence-2026-10-22.json)仅懒建未使用的T1执行载体及prefill另一槽，family冷编与数学不变。修改前／后／原peer完整decode为123.01／122.50／140.68tok/s，综合109.33／108.32／117.07；前后范围重叠，不声明速度改善。ready中位1.644→1.613s、含启动总时长2.233→2.204s，dedicated采样峰4.120→4.069GiB；原peer仍生成不同续文。当时record／reuse两条Hyle均18／50；这些历史值保留在原证据，不混入后续四臂或单槽结论。

peer首处分歧的固定控制已完成：B165 live pending与恢复同逻辑状态后T1的36份R/S、hidden／logits逐字节一致，未见该恢复分支额外误差。精确重放C171 T2复现原round9；同一RS1状态的T1[11906]使11／271均为23.793373107910156、选11，T2[11906,271]为23.788818359375／23.805431365966797、选271，GPU greedy与logits一致。该形状数值差足以翻转本次相近top2，不是sampler行错位；不推为通用算法bug、全部rollback或质量通过。Hyle相应BF16两项均23.75，不据此改dtype、容忍值或选择规则。目标T1的writer-only草拟KV取消／策略异常已实际通过6次Graph、14份权限退休与重试，策略异常不冒充设备故障。

[前批可录制写页](../../evidence/performance/qwen-capturable-append-2026-10-21.json)已实现，原Queue、Graph effect、逐页权限和whole-state提交不变。该批真实99个完成ID／96次decode核对：每步**1次Graph launch、596个kernel、0次录制外eager D2D**，90次不重录；[前一原生调查](../../evidence/performance/qwen-paged-native-cost-2026-10-21.json)分页路径为7／584及2232次eager D2D总量。descriptor动态内容可变化，实际载体变化仍先销毁旧录制；不能只翻capture标志。本批未重采物理trace。

独占尾页与Radix的[本批实现及完整对照](../../evidence/performance/qwen-exclusive-radix-2026-10-21.json)已完成相关检查：142份真实向量逐字节一致，同页失败回退、旧快照边界、真实CUDA取消退休通过；Radix目录4项host检查包括800次oracle对照。只命中已有完整状态边界，未制造检查点。

此前[执行资源复用窗口](../../evidence/performance/qwen-shared-execution-2026-10-21.json)各臂3新进程测速、另1进程显存旁采。单会话current／before／dense／llama完整decode为**101.25／101.40／105.34／105.38tok/s**，综合**90.86／90.84／93.67／91.89tok/s**，本批基本不变；对参考decode仍慢3.92%、综合慢1.13%。共享前缀多会话实际prefill为**1217.26／721.04／1201.40tok/s**，综合**91.99／86.13／95.12tok/s**；相对before中位＋68.82%／＋6.80%，当前保留一明显慢进程、范围重叠，不签稳定普遍收益。完整四项、范围及准备子桶迁移见[最新完整窗口](../../evidence/performance/qwen-mtp-single-and-multiturn-2026-10-24.json)，不混算两种负载或历史窗口。

| 剩余项 | 已知事实与下一步 |
| --- | --- |
| 宽多行TopK1 | **已修**：旧2CTA分段归约约1.2ms，新分块＋末级直接写原值／索引约32.72µs；12相关测试和90份同B向量通过。只在Provider内改算法，Model／公共接口不加专线；prepared／capture接线和14项相关检查已完成，吞吐近持平 |
| B2后续归因 | 当前GPU无活动区间仍比参考长，状态concat约0.215ms，small-M融合也有候选；已有实测和源码记录，尚不把整个余差归为某函数或状态复制 |
| 分页相对dense的成本 | 独占尾页已实现，同页只写新后缀，分支保COW。最新单会话窗比原dense完整decode慢约3.88%。此前按decode wall中位数选取各臂实际进程，分页额外0.424ms／步，其中完整准备区间多0.201ms、其余多0.223ms；后者含输入／replay／submit／device／token，不是纯GPU。该历史区间差不当作当前成本或可消除承诺 |
| 主机页组织 | 当前仍有随历史页数增长的验证、权限及发布工作。本批单尾权改固定单槽、fresh只预留实际数量；descriptor分组／发布用同Invocation的K个原selected slots，替换全计划S扫描。未缓存权限、epoch或ready事实，上传等待未删除 |
| descriptor上传 | 新路径每次真正上传并等待退休后发布，冷准备有额外成本。沿原依赖／F1机制减少不必要等待，不伪造ready或保旧权限来换速度；完整准备桶不是单函数计时 |
| Radix与共享输入 | 有界压缩树／LRU／多读者pin已接通并核真实复用。查找最坏O(L＋E)，替换旧重复token比较O(E×L)；按实际请求增加完整检查点仍须符合版本／字节预算和冷shape域。命中收益另测，不能冒充无命中decode收益 |
| 多会话执行资源 | **重复构造已修**：Phase／Invocation／IO／workspace归Model按实际执行并发复用，Session只保完整状态。142份向量和真实借用／退休／提交后异常检查通过。完整多会话窗dedicated峰少88MiB；单会话基本不变。前缀首次T1槽准备仍在首次D，旧版在P接纳内先建，因此D中位−1.60%不等于kernel退化；全请求／全程wall保留必要工作 |
| cached-prefill首用与拓扑变化 | 三轮窗口的第2／3轮prefill仍低于参考。已区分slot首用与KV分块拓扑变化造成的合法重准备，不能统称重复规划bug；资源复用后再核实际重录、使用次数与完整请求墙时，不绕过拓扑失效 |
| 多会话显存阶段采样 | **标记解析已修并实际旁采**：新窗收到各会话9phase的18边界，动态统计峰与邻样变化；实际约15ms间隔，fullhit及部分短P无样本，精确边界增量仍未知。旧窗不重写、不补造 |

上表区分已修项、已测慢项及候选方向。分页剩余墙差尚未全部归因，descriptor等待也不是可直接删除的开销。少量合法MatMul冷算法、量化融合及packed计算仍是后文的候选方向，不算已经证实的性能缺陷或收益。

顺序：**可录制写页、独占续写、Radix、执行资源复用、typed页帧、真实批次及CLI完整对照均已完成列明范围 → 宽多行TopK1慢项及已准备能力接线已修 → CUDA单会话K1 MTP有限实现及自然四臂窗口**。Conv／GDN独立B维、页帧组合／投影、完整状态／结果分流与N＋R预算已接通，8项host及实际批次／淘汰检查有结果。精确快照命中与原token重算分别报告；跨B重算允许实际数值差异，未签完整质量。单会话延迟、缓存命中收益和多人吞吐分开；其它系统原生写页与图片仍有必要实现，不扩大成重复全量验证。

此前窗口和归因留在[上一轮提交／发布证据](../../evidence/performance/qwen-retirement-publication-2026-10-21.json)及其回链；下面D1/D2是此前dense路线的已结调查，不充当当前分页性能。

当前选用的 D1 基线是 **2026-10-20 18:39:26–18:39:46 与 18:40:15–18:40:43，北京时间**两组同窗。
固定 128-token prompt、context 256；每窗修改前／当前／llama 各 3 个新进程交错运行。

| 负载与指标 | 修改前 Hyle | 当前 Hyle | llama.cpp | 当前相对修改前 |
|---|---:|---:|---:|---:|
| 完整 31 步 decode | 102.56 | 103.99 | 104.13 | +1.39% |
| 稳态第 7–31 步 | 106.34 | 107.74 | 106.28 | +1.31% |
| 完整 127 步 decode | 105.93 | 107.07 | 106.31 | +1.08% |
| 稳态第 7–127 步 | 106.86 | 108.03 | 106.85 | +1.10% |

单位均为 tok/s。短窗当前相对 llama 完整 −0.13%、稳态 +1.37%；长窗完整 +0.71%、稳态 +1.10%。
本批修复后两种长度均有约 1.1%–1.4% 的中位提升；每臂仍仅 3 个进程，不签稳定全面胜出。
短窗 prefill 为当前 2039.84／参考 1335.99，长窗为 2084.23／1347.16 tok/s。
相对修改前 prefill 分别 −1.13%／−0.008%；保留全部样本，不由此签回归或无回归。

长窗将原 31-token 固定输入带重复并截取到 127，最终可见 255 tokens；参考消费者仅放开有界阶段数，
底层 llama 实现和采样路径不变。它用于延长固定负载观察，不代表自然生成或质量验收；两窗不合池。
本次保留原 BF16 图闭包，只替换两个通用 Graph 实现对象；后续量化能力对象未全部重编，不能称为完整当前源码交付。
“无插桩”指没有新增 D1 计时或外部采样，原计时、日志和参考的粗粒度 memory snapshot 仍保留。
[源码、原始样本、独立复算与限制](../../evidence/performance/qwen-prepare-bindings-2026-10-20.json)；
[此前 16:27 窗口](../../evidence/performance/qwen-decode-rebenchmark-2026-10-20.json)保留为历史依据，不与本批合算增益。

### 尚需明确的比较边界

- 两端使用同一组 BF16 权重和固定输入；Hyle 的 activation/output 主要为 BF16，参考 decode
  多处使用 F32 中间计算。当前结果是两种实际计算路线的比较，尚非逐算子同数学比较。
- 下述旧D1普通消费者两端都计到host token可见。该固定参考路径每步还自动回传约0.99MB logits，
  Hyle通常只回传4B token；这是该历史配置的实际行为，不能沿用为当前MTP的拷贝计数。MTP另有
  target双行输出、draft和hidden carry，其实际工作分别计账。主比较保留双方自然token路线，
  不移动计时端点或从参考时间里估算扣除拷贝。
  如需诊断输出差异，另测双方都交付 host F32 logits 和 token 的负载，使用预分配缓冲并完整计时；
  这个补充结果不能替换 token 主结果，也不能抹平中间计算精度差异。
- 当前固定序列的 hidden/logits 与 Hyle 既有基线逐字节相同，用于回归；它不构成独立模型质量验收。
- 当前短序列用于保留前后可比性。较长 decode 另列结果，明确增加后的上下文长度；不能把不同
  KV 工作量的样本合成一组，也不能把进程内各 token 当作独立进程样本。

## 执行顺序

| 步骤 | 当前状态 | 做什么 | 本步结束条件 |
|---|---|---|---|
| D1：每步准备 | 已定位并修复本批重复绑定准备 | 同物理配置复用已构造的 spans，新权限和 seal 仍准备；18 CPU＋6 CUDA、64 份向量和两窗复测已完成 | 当前这条调查结束，后续仅在新证据指出冗余时重开；metadata 上传等待保持原合同 |
| D2：投影计算 | 已完成真实后缀对照，默认保留 D1 的 F32 Top1；直接 BF16 消费者候选本轮不采用，已结项 | 3 个独立进程和 1 次结构 trace 的数值、配对、空间和失败记录见下文 | 无挂起候选；不因 kernel 数减少或选择有利统计量自动采用 |
| D3：吞吐复测 | 本批 31／127 步各一组完成，仍为有限样本 | 后续有新热点改动时再比较，复用本批短长负载 | 报完整与稳态吞吐、分布和失败，确认收益及剩余可重复慢项 |
| D4：最终整机性能对比 | 进行中，初轮及auto修复后比较已完成 | 当前源码统一比较加载、prefill、cached continuation、decode、内存和显存 | 性能、资源、正确性和公平性结论均有对应实际证据；未解决项明确保留 |

Top1 的这次有限决策已完成，接下来推进总体计划中的格式／功能补全。格式补全使用独立的功能清单推进；
CT/NVFP4 的新增能力没有进入当前 BF16 Qwen 图，不计为本图的 decode 优化。

2026-10-20的[DeepInfer源码调研](../../evidence/research/deepseek-runtime-inspiration-2026-10-20.json)另提供一个有限检查方向：
D4前核一条真实投影→residual链能否保原BF16舍入、读者与alias而直接写最终输出，减少一次实体搬运。
先核实际native epilogue与数值边界，有成立的完整候选再测吞吐；否则记录具体不采用原因并关闭本轮候选。
这不是已证明的额外开销或提速，不重开已结项的Top1实验。FP8输出布局仍按实际消费者候选推进；MTP的GPU metadata、独立草拟frontier和同图验证已接通，不再列为未接线项。

[TensorRT-LLM固定源码研究](../../evidence/research/tensorrt-llm-inspiration-2026-10-20.json)另核冷准备、共享量化及执行重叠：
已读源码的SM89普通BF16投影沿F.linear／cuBLAS路径；可见的low-M／persistent特化有SM100条件，
CUDA-core GEMM的已读公开调用者则要求FP8输入。它们不能直接解释当前4070 BF16 Qwen还能快多少。
D4前沿现真实热点只检查同dtype／shape的选路、实际权重读取和host-token空隙；已有共享FP8 prologue与有限图族不重做。
只有发现新前提或可执行候选才做配对测量；保持同权重、精度、上下文与结果可见边界，
区分单请求decode、并发总吞吐和MTP，不将大batch或B200的收益挪入本基线。

### D1：查清每步准备

诊断构建先拆消费者，再只拆 replay：三个新进程中 replay 准备中位约 92–97 µs，
其中绑定准备约 91–95 µs，累计约占 replay 桶的 98%。这些插桩结果用于归因，不作吞吐排名。
实际冗余是配置撤销后重新扫描整图、构造未改变的物理 TensorView spans。
现已复用同址、同描述、同范围的配置事实；普通绑定、命令变化、失败准备和 full reset 撤销候选，
source、opaque/native owner、派生 alias 和投影等仍保守走完整准备，未知 identity hook 照常执行。

无新增插桩窗口中，稳态准备桶的“各进程中位再取中位”短窗 **165.3→64.3 µs**，长窗 **156.4→60.9 µs**。
18 CPU 与 6 CUDA 定向回归通过、0 skip；64 份 hidden/logits 和 4 份冷图 route/boundary TSV 与保留基线逐字节相同。
本批没有新测显存／内存峰值、能耗或物理 kernel trace。
`submit_seconds` 已包含在后续提交到结果可见的时间内，不能再相加。

本轮沿[消费者 evaluate](../../../engine/models/qwen35/main.cc)分段，保持原调用和等待顺序：

1. 既有输入写入，包括 token/position 两次 H2D；它们在 `cold_batch_seconds` 桶外，但在完整 step 内。
2. layout 版本创建及六个 reader 的布局装配。
3. `batch::prepare` 的配置撤销、权限/别名检查、metadata 绑定及上传。
4. `Invocation::prepare_replay` 的绑定准备、capture 检查/复用和退休查询。
5. 原提交、等待及 host token 读回，作为完整 step 的核对边界。

绑定重建假设已经确认并修复。共享 metadata 只有首个 owner 执行实际上传和等待，
其余 reader 使用共享准备，不能按六次上传计算成本。capture 已复用也不等于绑定准备已经跳过。
本轮没有额外调用 `batch::prepare_replay` 或改变现有捕获旅程。
若要复用缓存，缓存身份必须覆盖布局、来源、权限、alias 和旧结果存活，不能绕过 idle/owner 检查。

插桩只用于归因；吞吐收益由去掉插桩后的同窗确认。若剩余工作都是必要检查、实际上传或 capture
依赖，就在这里结束这条线，不把这些行为悄悄挪出完整 step 计时。

### D2：按投影角色优化

历史匹配 trace 中，投影约占双方 kernel 累计时间的 92%；head、gate/up、down 占每步逻辑权重
字节的约 75.2%。这说明它们值得优先核查，但累计 kernel 时间不是可直接相加回收的 wall 时间。

先在当前图确认热点角色，再检查：

- 同一实际矩阵形状和 operand/output dtype 下，是否有额外读写、repack、转换或中间落地。
- 同 A 投影和 FFN 融合是否保留规定的 BF16 舍入，是否重复读权重或复制权重 bank。
- 实际带宽、cache 命中、launch 和完成等待中，哪部分解释了当前差异；由此选择现有 provider、
  融合候选或实现修复。一般机制留在底座，模型角色和生成策略留在测试消费者。

10月20日该窗口的总 kernel 是 **596 对 558**；395 是参考非投影部分，不能用作总数，也不代表后续普通／MTP的当前图。
以前测过的整批 MMVF 替换、通用大投影替换、FFN 两行 CTA 未证实收益，
只有新证据改变其前提时才重开。历史图内空隙、GraphLaunch 主机区间也不能直接承诺为可回收预算。
[热点与计数](../../evidence/performance/decode-major-hotspots-2026-10-20.json)、
[busy union 与归因限制](../../evidence/performance/decode-busy-union-2026-10-20.json)。

若更大提升需要降低权重精度，单列相同量化与质量检查的比较，保留 BF16 结果；
硬件 FP8/FP4 数学也不能自动当作现有 QDQ 路径的等价融合。

#### 本轮 D2 结论

head/down 沿既有 cuBLAS BF16 输入、权重和输出及 F32 累积；gate/up 保留严格 BF16 舍入的双输出融合。
当前核查未找到额外热 repack、全权重 bank 扩宽或可确证的大投影缺陷。原来 FFN 的 DRAM store
现象也不能据逻辑 store 次数解释为隐藏中间副本或 spill；没有新前提时不重开旧负结果。

消费者另试了直接 BF16 Top1，移除 head 后整行 BF16→F32 Cast。3 项 CUDA 回归通过、0 skip，
64 份 hidden/logits 保持逐字节相同；结构 trace 是 596→595 kernels、55→54 Cast。
这证明测试范围内的数学和结构变化，不构成吞吐或模型质量结论。

| 同窗完整 decode（tok/s） | D1 | Top1 候选 | 候选相对 D1 |
|---|---:|---:|---:|
| 31 步 | 104.89 | 103.90 | −0.945% |
| 127 步 | 99.31 | 98.50 | −0.815% |

每臂各 3 个新进程，长窗 D1 样本 max/min 跨度达 8.22%；没有确证真实退化幅度或根因，不能称为单纯计时器分辨率。
原投影 kernel、路由前缀及其余 149 个链接输入均不变；未发现 D1 全图绑定重建重新出现的明确证据。

一次固定的同进程局部对照，使用原 decode7 的 248320 个实际 logits、相同 Commands 路线、
32 次／臂预热及 ABBA/BAAB 的 128 次／臂相关样本，计时含 submit、wait 和实际 4 B host token：
直接 BF16 的耗时中位 39.95 µs，Cast＋F32 Top1 为 44.05 µs，每 cycle 的配对节省中位 2.25 µs。
这次局部观察中直接 BF16 更快，不能把整模型窗口的负号归因于 BF16 Top1 更慢。
原模型 Cast 位于更大的捕获段，这个 Commands 对照并不等价于模型后缀；辅助 stream event 还包含 host
admission 空隙，不当纯 kernel 时间，也不据此换算整模型收益。

已完成的第一轮保持 D1 基线，保存候选和全部正／负结果，保留新增的大词表 Top1 测试；
未提升基线不等于证明优化方向错误。真实 head 后缀的最终选择见下文，完整模型窗口没有继续扩张寻找正号。
[源码、完整样本、局部对照和选择依据](../../evidence/performance/qwen-bf16-top1-2026-10-20.json)。
归档核验后通过根 `test.ps1` 精确回收本轮 76 个可重建镜像／链接副产物和重复向量，约 464 MiB；
保留当前 D1 程序、参考、源码、对象和输入，[清理实录](../../evidence/performance/qwen-bf16-top1-2026-10-20.cleanup.json)。

最终对照保持真实公共投影与 capture／host-token 边界：两图共享同一原始 BF16 `[248320,2048]`
权重和 decode7 的真实 hidden，经过相同 Copy 并发布 caller hidden／logits／ID，TopK value 不发布。
原臂保留 private BF16 head→Cast→caller F32 logits，候选直接写 caller BF16 logits，保留真实存储差异。
3 个独立进程各做 32 次／臂预热、64 个 ABBA/BAAB cycle，128 个相关样本／臂；全部 768 个计时提交保留。

| 进程 | 原臂中位 µs | BF16 候选中位 µs | cycle 配对节省中位 µs | 候选均值多耗时 µs |
|---|---:|---:|---:|---:|
| 1 | 2156.00 | 2151.95 | −6.20 | +50.515 |
| 2 | 2157.50 | 2151.00 | −0.55 | +8.105 |
| 3 | 2154.55 | 2147.70 | +5.00 | +2.821 |

预定的配对方向未在独立进程间重复，不能换用普通中位数较好来签收益；均值也未改善。
独立复算未发现计时不公平或可确证的周期噪声机制。另一次结构 trace 确认两臂 head 全部被捕获，
符号与 `62080×1×1 / 16×4×1` 几何和整模型相同；4 个 CUB kernel 均为 eager 尾。
Cast 本身中位约 3.04 µs，尾部确有小节省，但 head 和完成边界的变化覆盖了它。
profiler 的时间只用于归因，不混入无 profiler 的性能结论。

4 个进程的完整 logits 和 hidden 与原始向量逐字节一致、token 均为 10462，capture 均实际复用。
后缀 workspace 从 508416 降到 11778 B，caller logits 从 993280 降到 496640 B；
这些组件的空间减少是真实结果，未据此签整模型峰值或吞吐改善。

**本轮决定：默认保留原 D1／F32 Top1，直接 BF16 消费者候选不采用并结项，无挂起候选。**
保留通用 BF16 Top1 能力、新增测试和已完成实验记录；这不是对其它硬件、负载或 BF16 Top1 算法的普遍否定。
[最终决策、真实输入身份、失败编译、全部样本和结构归因](../../evidence/performance/qwen-top1-real-suffix-2026-10-20.json)。

权重流式读取占主导时，小算子或 host 调度的改善可能被总耗时和漂移掩盖；减少实际 DRAM 读写或提高复用仍有价值。
保留的历史单节点 NCU 中，LM head 为 97.55% sustained DRAM peak（实际总流量约 491.22 GB/s），
FFN 双投影为 92.71%、L2 hit 0.89%；这支持这些大投影受带宽限制，不能代签当前整步始终饱和。
不能只凭投影占比或 GPU 利用率就声称整步达到显存带宽上限。按 NVIDIA 的
[Nsight Compute 归因方法](https://docs.nvidia.com/nsight-compute/ComputeTriage/)，
分别看实际 DRAM bytes／持续带宽、L2 命中与吞吐及延迟／dispatch，再对照绝对耗时；数值正确性仍独立验证。

### D3/D4：确认实际收益并收工

复用已有权重、参考程序、构建对象和原始输入，只增量构建受影响部分。先运行相关数值/寿命回归，
再运行无 profiler 的吞吐对照；设备分析和内存旁采另开观察，不混入吞吐样本。

每轮保留当前/前版/参考的必要同窗；记录硬件、driver、工具链、源码和二进制身份。
交换顺序，保存慢进程和失败，按需要增加独立样本及同路径 A/A，先解释波动再增加重复。
已有证据未受改动影响时直接引用，不重跑跨后端、格式、shape 的笛卡尔积。

最终报告至少包含：

| 结果 | 完成边界 |
|---|---|
| 加载、ready、TTFT | 文件读取/解码、上传、准备及必要清理分别可解释；OS 热页缓存不称冷盘 |
| prefill 与 cached continuation | 相同 token 输入、context 和状态，输出实际可见 |
| 完整 decode 与稳态 decode | 首次 capture 保留在完整结果中，稳态另列；报告样本数和分布 |
| 内存、显存、workspace、搬运 | 说明采集区间与已知/未知，workspace 为零不等于总分配为零 |
| 数值与状态 | 改动相关的独立数学/生命周期检查、固定序列回归及尚未完成的质量验证 |

“追平”需要反复独立同窗不再出现超过实际测量波动的持续落后；未测出显著差异不能直接宣称等价。
若仍有可重复的小输，继续按本页具体热点调查；没有证据时保留不确定结论，不制造任意容忍值。
反超幅度由测量决定，目前不承诺额外 15% 或 50%。

完成本轮以后，将本页更新为实际结果与剩余问题；已否定的方向留简短结论和证据链接，
避免重新从旧假设开始。本轮功能与交付结论见[最终工作快照](../../evidence/planning/current-work-final-2026-10-24.md#mtp本轮结果与边界)，后续工作按用户当前指令推进。

### 2026-10-21：D4 实测与一次性 capture 修复

首次实际测128-token前缀后的16-token续算，发现811.46对999.86 tok/s、慢18.84%，三对三样本无交叠。
[首轮证据](../../evidence/performance/qwen-current-delivery-2026-10-21.json)保留全部负结果及当时源码，不由后续数据覆盖。
两次同负载Nsight诊断确认该一次性调用付出了capture准备成本；当前trace的图内kernel不可见，未将整段图span冒称busy union，或将API耗时之和当可回收收益。
现有全局Commands对照使续算更快，却使decode变慢；因此仅改测试消费者的使用政策，复用库原有ReplayMode：
按完整计划冷统计每个exact child的两个全局parity槽，一次性槽Commands、预计复用的槽PreferCapture。
没有按shape／模型／Provider特殊判断，热路径只读预选数组，没有虚假预热或移走首次执行；生成提前结束时使用数是请求上界。

新默认auto的64份hidden／logits与原基线逐字节相同，固定预测一致。以下各窗每臂3个新进程，均无profiler或外部旁采，保完整host-token step。
同源BF16权重、固定tokens、context256；中间精度及原引擎采样读回仍不同，不能推出数学或模型质量等价。
参考消费者沿原backend greedy／public sampler，不读raw logits或做CPU argmax；引擎额外sampled-logits读回留在实际计时内。

| 输入窗／指标，tok/s中位 | Hyle auto | llama.cpp |
| --- | ---: | ---: |
| 128 prefill，短窗 | 2217.97 | 1401.10 |
| 31步完整decode | 103.63 | 105.22 |
| 短窗decode7起 | 107.91 | 107.64 |
| 128 prefill，长窗 | 2300.60 | 1284.26 |
| 127步完整decode | 107.12 | 106.47 |
| 长窗decode7起 | 108.00 | 107.04 |
| 缓存后一次追加16 | 1020.63 | 977.54 |
| 续算后15步完整decode | 100.34 | 102.74 |
| 续算后decode7起 | 107.09 | 106.36 |

16-token续算在这次同窗中从旧capture策略835.94提高至1020.63，约22.09%；此前811.46属于另一窗口，不混作这一提速比例。
缓存续算输入为原31-token tape拆成16＋15，沿同一状态从128→144→159；17个可见前缀分别与各自逐token旅程预测一致。
**D4仍未结项**：完整31步与续算后15步仍小输，稳态较快不能代签完整调用。下一动作只分解最初两槽的capture启动代价，保初始调用，不靠删样本或另设任意容忍值收工。
现有样本的均值差额集中在前两次decode：31步前两次多5.84ms、全窗多3.88ms；续算后15步前两次多4.68ms、全窗多4.21ms。
既有trace中Hyle前两M1各实例化一次、参考在decode2实例化一次；原迹不是新auto实采，API耗时差不能直接当可回收收益。
[修复源码、完整同窗／旁采、独立复算与首次成本分析](../../evidence/performance/qwen-auto-replay-policy-2026-10-21.json)保留实际边界及尚未解决项。

短窗同次进程的加载／ready／TTFT中位：Hyle约0.792／1.134／1.192s，参考1.432／1.511／1.602s；加载子边界不同，OS页缓存未控制，不称冷盘加载。
独立50ms PSAPI／PDH旁采各1个新进程：当前／参考驻留working-set观测峰约1.05／4.07GiB，private commit约5.47／4.88GiB，GPU dedicated采样峰约3.80／3.95GiB。
commit不是驻留RAM，各类内存不相加；GPU峰是采样下界，启动缺失实例保留null，过短阶段没有样本不推测phase峰。

### 首次capture的资源消冗（2026-10-21）

新auto的实际node trace确认前两M1各596个kernel，参考各558个；Hyle的记录／实例化在首次执行中，不能据此合并两份固定绑定制品。
已复用Queue中未被capture引用的状态字，并在原退休链已有同context／stream完成证明、无需原生查询时省去context guard。
新迹每个首M1的push／pop由各1313次降至945次；第二次capture少一次malloc，两次各少一次free，数学kernel数不变。
5项定向回归、64份原向量逐字节检查、Linux原安装SDK的两个消费者通过；仅增量编译两份产品C++，没有重建SDK。

另一次无profiler短窗每臂3个新进程：旧auto／当前／参考完整decode为93.04／92.29／95.79 tok/s，稳态96.64／96.18／98.27；
前两步合计中位33.67／32.05／26.35ms。样本波动较大，**没有证明吞吐提升或回归**，不与上表较快窗口混算。
[原始样本、三次node trace、数值和安装消费者实录](../../evidence/performance/qwen-capture-resource-reuse-2026-10-21.json)保留源码及限制。
接下来结合本地参考引擎调查具体的计算／搬运缺口；不因减少API次数宣告D4完成。

### 本地引擎研究与下一批

[7个本地仓库的源码对照](../../evidence/research/local-inference-optimization-2026-10-21.json)将23条机制记录归并为5个方向，尚未作为提速结果。
当前BF16 Qwen已确认T1的103／187个MatMul选direct Lt，包括down／head；T128为187／187。原provider只取一个heuristic，
下一步只选一个真实热点形状，比较少量合法冷算法；保输入、累积／舍入及workspace边界，实际会话变快才采用，不建通用调优系统。
之后处理保BF16／F16舍入的norm／activation→quant融合，以及数学合同匹配的4bit packed分块MMA。
后两者服务实际量化模型，不会自动加速当前未用该格式的BF16 Qwen。

图池／资源复用已有主体机制，只调查实际重复；expert流式驻留与CPU/GPU重叠留给需要它们的真实消费者。
不将SM100／多GPU结果外推4070，不用覆写旧结果的共享输出换取显存，源码研究不代签吞吐收益。
两侧消费者现已支持同一有序多轮phase输入和显式容量，沿原缓存记录完整会话；首轮四项比较见下节。

### 三轮共享缓存的四项基线（2026-10-21）

已完成上一段的消费者扩展。双方沿一个缓存执行`(256输入＋32单token decode)×3`，上下文0→256→288→544→576→832→864，容量均为1024。
每臂3个新进程交错测速，另各1个新进程独立旁采显存；重复原固定ID带，是合成teacher-forced性能回放。
99个host预测（3个prompt首token＋96个decode输出）在全部8进程一致。

| 阶段 | Hyle tok/s | llama.cpp tok/s | Hyle显存峰 GiB | llama.cpp显存峰 GiB |
| --- | ---: | ---: | ---: | ---: |
| 第1轮输入256 | 3279.28 | 2286.93 | 3.956 | 4.191 |
| 第1轮完整decode32 | 101.69 | 102.30 | 3.972 | 4.194 |
| 第2轮追加256 | 7527.04 | 8903.17 | 4.077 | 4.194 |
| 第2轮完整decode32 | 103.55 | 103.10 | 4.081 | 4.194 |
| 第3轮追加256 | 8410.15 | 8681.64 | 4.081 | 4.194 |
| 第3轮完整decode32 | 102.88 | 103.45 | 4.085 | 4.194 |
| 综合输出 `99/session` | 91.78 | 89.67 | 4.085（全程） | 4.194（全程） |

速度为各进程值的中位数；显存列是独立旁采进程的GPU dedicated阶段观测峰，采样下界，不是精确阶段边界占用。
完整会话中位1.07872对1.10410s，描述性综合输出差约+2.35%；第二／三轮输入仍为−15.46%／−3.13%，各3样本均较慢。
本窗无独占GPU／固定频率，不签稳定胜出或统计显著性；不与过去短窗合并计算增益。
加载／ready、完整decode与稳态分布、working-set／commit及shared GPU值在[完整数据](../../evidence/performance/qwen-multiturn-current-2026-10-21.json)。
内部启动到末输出约2.25549对2.70460s，从消费者入口计时，不包括OS装载DLL；进程wall还包括销毁，两者分列。

请求旁采间隔10ms，实际中位约15.2ms；两端ready到首prefill没有样本，因此加载后基线及严格阶段边界增量均为unknown。
不能将相邻样本跨下一阶段／退出清理的变化称为本阶段释放；dedicated和shared的独立峰也不能相加。

第二轮Hyle的完整输入约34.01ms，combined batch/replay准备约5.12ms，准备后到host token约28.82ms；参考完整28.75ms。
第三轮prefill仍record1／reuse0；每轮decode均record2／reuse30。FlashInfer按实际KV分片拓扑决定是否可复用，源码存在重规划→失效原capture范围的合法链，
但本次没有记录实际拒绝gate，尚不能认定根因。下一步只细分两项准备时间并记录已有六份公开BatchPlanReport。

新单轮schedule与原消费者64份向量逐字节一致。短多轮与逐token共享端点25个ID一致，追加后32／50份向量不同；独立Torch同计划25个ID也一致、全部值有限，
相对L2误差约1%量级及其变化已完整保留，未设事后门限、不签模型质量通过。此诊断不计入速度窗口。

### 新阶段首轮与准备诊断（2026-10-21）

最新完整四项对比集中在[最新完整窗口](../../evidence/performance/qwen-mtp-single-and-multiturn-2026-10-24.json)，[原始数据与源码](../../evidence/performance/qwen-next-phase-first-2026-10-21.json)保留本次窗口及前窗限制。
本窗各3个进程测速、另各1个显存旁采；两端均比前窗慢，不将跨窗变化归因本批workspace查询消冗，也不签稳定胜出。

独立的一次host诊断已拆开准备成本：第二轮T256使用slot1，属于该槽首次native plan，batch／replay约3.926／0.559ms；
第三轮回slot0，KV256→832按固定FlashInfer分块规则必改拓扑，batch／replay约0.542／3.623ms。
每轮decode首两次native计划均不复用，随后30次复用；等同workspace／CTA尺寸不能证明整个原生计划可复用。
已在同一BatchSlot的准备事务中一次性交接精确workspace sizing facts，保原生materialize、容量与退休检查，未单独证明吞吐收益。
下一步检查capture在实际拓扑寿命内的复用与摊销，避免只有一次使用的重录；不通过放宽计划或页权限失效规则换取速度。

### MTP额外重录定因：2026-10-22

[当前镜像的一次完整P154／64诊断](../../evidence/performance/qwen-capture-cause-2026-10-22.json)已补首次失效step／port，64自然ID、34轮决策及计算量与未插桩镜像相同。18record＝4组role／slot首次各2段＋10次真实失效，另50reuse。10次首变化端口中9次是prior／carry hidden、1次是layer0 conv history；resource／handle／shape／strides／span相同，仅池内offset变化。不是旧冷图反复规划或本窗native plan／Queue／权限失效。完整物理参数改变后，当前捕获记录必须销毁；同一allocation不授权沿用旧子范围指针。

这些失效只出现在第4～15轮；其7个调用的冷准备桶合计18.1936ms，包含其它准备和插桩，不能当重录自耗或可回收收益。34轮冷桶总57.2235ms，其中4个首次调用29.6260ms；不能将整个冷桶归到额外10次。range大小也不同：53步5次、55步1次、1269步4次，不按数量等价换算耗时。本批未做新TPS窗口，最新四项仍见[最新完整窗口](../../evidence/performance/qwen-mtp-single-and-multiturn-2026-10-24.json)。

**本轮不采用强留状态块、额外影子复制或放宽失效，相关候选按本次决定结项。** 当前LeasePool没有指定offset接口；接受前缀、旧快照／前缀pin和真实退休决定哪块实际可写，不能只按role／parity宣称地址稳定。CUDA参数更新也没有现成的通用捕获合同；旧artifact与native所有者必须先正确分离，不能仅因为工具链有`cudaGraphExecUpdate`便直接复用旧记录。没有证据支持为这10次先重构36份状态池或增加公共API。

后续一次有限热点分析已完成，见下。capture定因已结束，本轮不继续扩捕获系统调查；参考引擎续文不同，仍不签同token速度胜负。

### MTP稳态投影与行批候选：2026-10-22

[完整有限排查](../../evidence/performance/qwen-mtp-steady-profile-2026-10-22.json)只对当前普通镜像采1次Nsight。34轮各2次Graph launch均与实际GPU kernel correlation关联，64ID、接受决策和实际计算量与原记录一致。第16～34轮全部reuse2；19段中位GPU读回完成锚间隔14.043566ms、kernel duration和13.004021ms、kernel／copy／memset活动并集13.149329ms、非活动区间0.886171ms。区间不等于消费者端点wall，非活动不等于CPU自耗；CUDA13.4驱动由CUPTI13.3软件trace采集，诊断不作TPS。

矩阵投影占稳态累计kernel duration92.4658%。BF16 WMMA两类普通小投影累计约94.817／47.631ms，末分数draft GEMV约42.463ms、target大WMMA约42.218ms。完整符号、grid和每段明细留在证据中；`Kernel2`短名不能单独区分公共cuBLAS Provider与GGML融合内部的cuBLAS调用。GDN／Top1／状态搬运不作为本窗主要耗时解释。

唯一候选仅将普通rank2／两行BF16 MatMul的内部native布局改为2批单行矩阵共享原权重，保原dtype、F32累积、跨度／输出不重叠校验、32MiB workspace上限和退休；scores自定义、F32／FP16、原rank>2均不改。只编Provider和测试2个host TU，重用Qwen194／195链接输入；12项测试通过，含非方阵列主权重、两行独立重放。原P154／64各1个未插桩进程的64ID／34决策／全部选定工作字段也相同。

| 初筛指标 | 默认 | 行批候选 |
| --- | ---: | ---: |
| prefill tok/s | 2348.10 | 2351.07 |
| 完整decode tok/s | 122.62 | 93.52 |
| 综合64输出／session wall tok/s | 110.46 | 86.59 |
| 入口至末输出s | 2.188 | 2.358 |

**行批候选拒绝并结项，默认产品源码不改。** 各1进程仅作有限筛选，没有显存旁采，不认定普遍23.7%退化或完整模型质量；不再追加全窗／profile来寻找有利样本。后续若再查普通小行数投影，只选少量原形状合法冷算法，保原数学、工作量、布局和资源合同；有明确候选收益才做四项完整对比。图片encoder／projector实现与底座功能继续推进，不以这次候选失败扩大运行时调优框架。

### 历史AF普通／MTP四臂完整窗口：2026-10-22

当时使用默认Hyle镜像和原native llama.cpp消费者跑12次交错测速、4次独立内存旁采。相同P154／64、C1024／N1／R1、prefix off，两个Hyle模式均保BF16输入→F32末分数政策；无新编译、无profiler。

| 模式 | Hyle prefill／decode／综合 tok/s | llama.cpp prefill／decode／综合 tok/s |
| --- | ---: | ---: |
| 普通 | 2266.01／90.95／84.14 | 1507.63／96.03／85.03 |
| MTP K1 | 2168.72／112.27／101.24 | 1454.02／126.18／105.10 |

普通decode相对慢5.29%，MTP慢11.03%；prefill分别快50.30%／49.15%。同一模式的每臂输出／工作／决策可重复，但两对跨引擎续文都从第19枚分歧，不称同token严格差距或质量验收。两端绝对速度均低于前窗，不据非固定频率的跨窗值签代码回归。该窗完整范围、加载／启动、显存／内存、原始命令与采样限制见[AF证据](../../evidence/performance/qwen-mtp-steady-profile-2026-10-22.json)；[最新完整窗口](../../evidence/performance/qwen-mtp-single-and-multiturn-2026-10-24.json)已更新为2026-10-24单轮／真实两轮窗口。只用各窗自己的4臂互比，拒绝候选的单对初筛不混入。

### 当前MTP数值与功能边界：2026-10-22

[本轮4个有限诊断进程](../../evidence/correctness/qwen-mtp-current-numerics-2026-10-22.json)保持当前默认镜像／精度政策；无新TPS窗口。MTP19与MTP20在index18对应调用分别自然选Single／Round，两者均选11，普通20选271；MTP20该Round拒绝后仍有最后一轮Single，不能把index18的Round误叫整个进程最后一轮。三者prefill公开hidden／logits逐字节相同，不据此推全部状态相同。

同一完整P154和自然L171旧Snapshot分别fork T1／T2，T2显式使用真实草拟、quota1按原政策只发布首边界，未强迫接受或替换proposal。33次真实Graph包含自然前缀、4次分支及19次allSingle调用；旧52份状态保护通过。两点各自T1／T2首ID相同，target hidden和分数不同；P154首层conv history的新qkv行已经不同，差异发生在causal conv／GDN前。沿相同P154保完整MTP状态、连续真实quota1的20输出与普通版相同，不能只把第19枚分歧归到末步形状，也不能拿ID一致代签模型质量。当前不因此改dtype／归约政策、强制T1或取消MTP。

当时源审确认的Apple／Vulkan同Graph分页writer effect接线缺口已在后续补齐；F32 scores和GDN多快照也有实现。此段保留2026-10-22数值调查背景，当前后端结果见工作计划及后续证据。功能补齐与质量评估单列，不以数值诊断的计时当吞吐。

### M2 FFN-down有限冷算法筛选：2026-10-24

[实际叶诊断](../../evidence/performance/qwen-m2-down-lt-screen-2026-10-24.json)保持原M2／K6144／N2048、BF16输入和输出、F32累积、原权重布局、32MiB workspace上限及COMPUTE_TYPE归约。只比较默认request1与两种合法配置；原24MiB权重范围只读取一次，复制成4个GPU bank共96MiB，大于实际36MiB L2的两倍。每个配置用含64次串行M2的Graph轮转bank，5次交错试验，每次仅一次host launch，event除以64。

默认、split1 stage19、splitK16的每叶中位耗时分别为63.920／63.168／64.704µs，范围分别为59.856–71.200／59.965–66.621／60.096–68.528µs。范围重叠，约1.2%的最佳中位差没有给出明确可采用收益。stage19与默认BF16逐位相同；splitK16有20／4096个BF16值不同，仍通过独立F64点积、固定F32误差界及向外BF16 RNE端点门，不能称逐位等价。

**保留默认配置，本次有限叶候选结项，不继续搜索。** 这只是有界轮转工作集近似，未测PMU cache miss，不签绝对冷带宽、真实decode收益或最新四臂差距的完整归因。compile2／link2成功；run2在环境准备阶段失败、没有启动native，run3才实际退出0。原记录的`loaded_cublasLt`路径实为import-address所在exe，DLL完整路径未知；API版本130600为实际值，同进程配置比较仍成立。产品源码未改，最新单轮／真实两轮四臂数据保持不变。

### typed MMVF双行候选：2026-10-24

[后续有限叶对照](../../evidence/performance/qwen-m2-typed-mmvf-leaf-screen-2026-10-24.json)直接实例化已有MMVF模板，一次计算完整M2，共享权重读取，保BF16输入／权重／输出、F32累积与最终一次BF16舍入。这是新算法候选，不重开行拆或Lt参数搜索。沿用同一原24MiB down权重范围、保留的两行真实prefill输入、独立数学门及4bank／64叶／5次交错协议。

默认Lt与候选中位耗时为63.4345／58.4000µs，范围60.0935–77.0240／53.2000–58.6030µs；该叶耗时减少7.94%，五次范围未重叠。各4096个输出均过原数学门，候选有22个BF16位差异，不称逐位等价。首次link缺少原GGML abort所有者，补已有base库和原CRT链接合同后，仅复用两成功对象重链，run2实际退出0。

**整消费者筛选后拒绝，本轮结项，产品源码不改。** [六进程交错对照](../../evidence/performance/qwen-m2-typed-mmvf-consumer-screen-2026-10-24.json)中，默认／候选decode中位数为125.779／124.156 tok/s，综合速度为114.849／113.474 tok/s；三对候选均更慢。两臂128个输出ID与总工作量相同，但第12–14轮局部MTP决策不同，不能把全部差值归因于该kernel。独立结构trace确认候选确已执行，两次target各24个新双行kernel，仍各有738个kernel、135次D2D／39,731,200B。本次只用于采用／拒绝筛选，不替换正式四臂速度与内存表，不追加样本或参数搜索。
