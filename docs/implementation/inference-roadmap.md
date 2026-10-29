# 推理工程范围与顺序

HYLE 的产品目标是完整推理引擎；Compute、Graph、内存、调度、I/O 等模块继续独立交付。
当前主阶段仍是底层推理支持，引擎应用是检验真实模型的驱动，随测试附带推进。
目前完整真实文本路径以 Qwen3.5-2B 为基础，官方 BF16 以及固定 GPTQ 制品已有列明范围的检查，仍需 prepared inputs 和 token IDs。
通用 loader、C++ tokenizer、服务入口和图片链路尚未完成，工程入口统一不代表这些能力已交付。

当前阶段转入基础普通推理、多会话、多请求／多任务的功能穷尽补充、架构复审调整和性能狩猎，
进度见[推理看板](../developer/inference-progress.md)。MTP 与多模态不重开功能穷尽；性能仍包括 MTP 与参考的差距及相应执行调整，共同底座保持已有合同。
2026-10-26的新架构 Review 纳入共同编排、稳态状态协议、形状政策及完整会话换出。
共同请求／分组／缓存原语、按需快照、原池索引状态及完整主机镜像已实现并取得各自有限检查；
有限尾块及冷形状政策已有列明有限检查；重叠执行及当前完整性能对照继续推进，实际平台与制品边界分别记录。
不能以往轮完成结论排除新发现。每个缺口先明确功能和共同架构，再实现并做相关有限检查及完整性能对照。
先核真实缺项和调用链，再补功能及其架构；性能重点检查既有 Hyle 加速设施的实际复用，
并从本地成熟引擎提炼可验证候选。运行中接纳、背压与取消属于请求底层，不推给后续 HTTP 服务。

## 当前 goal 范围：全部 review 应对

用户已要求将这几轮 review 的工程、功能、架构、GPU 优化和测量问题全部纳入同一 goal。
以下是工作范围；逐项百分比、正在做的事项及最新实测统一维护在[推理看板](../developer/inference-progress.md)。
不能只完成注意力重录或一个代表模型就结束，也不把 review 的收益估算当作已实现结果。

| 任务 | 包含的 review 内容 |
|---|---|
| 功能与工程边界 | 冷形状接纳、长度分桶与预热；有效长度／varlen prefill及递归状态正确性；组合矩阵与分类错误；单成员取消、完整会话主机换出换入；CPU路由中的模型形状常量、Provider命名、SDK detail边界和ATen依赖选择；初始化与公开配置、简明执行合同、文档、Apache-2.0声明及依赖许可交付 |
| 共同执行架构 | 将真实执行编排、形状族与载体管理从Qwen Model抽离；静态模型适配；基础提交／取消／credit和token／结束／分类错误输出边界；统一稳态准备与权限表达，审查在途Source路径和普通／后继双准备路径；共同请求与缓存不等于完整执行核心；统一可共用的K1／K>1 MTP流程 |
| 普通连续流水 | wave跨pump在途、先提交后继再观察前驱；有序发布、背压、EOS、取消、失败与真实退休；常数成本的稳态准备、上传／读回合并；固定拓扑attention和独立槽；拒绝后恢复、跨turn预热提示及冷准备捕获；扩展适用配置与平台 |
| MTP性能 | 首个capture失配端口及完整绑定轮换；hidden固定池索引读写；固定验证形状和catch-up有效行；设备接受／后继与连续流水；不按测试prompt选择的草稿词表子集、接受率与完整输出检查；草稿变便宜后重测K2／多草稿 |
| GPU热点 | GDN a/b小投影并入同输入投影或专用GEMV；qkv/z/a/b、q/k/v、gate/up的整模型融合候选；模型内GEMM实际DRAM带宽；norm、残差、concat等小kernel融合。已关闭候选须有适用边界与重新评价条件，不能用旧叶子结果排除不同工况 |
| 测量与回归 | 当前版本R1／B2的off／auto及MTP跟踪、capture数量和主机／GPU分解；schedule对steps的交错A/A；原始startup、ready、prefill、decode、每turn与综合端点；显存／内存；长输入、多轮、多会话、跨平台及CPU＋GPU／MoE适用基线。重叠step时长不可相加代替墙时，旧Nsight不代签当前版本 |

底层功能补全已于2026-10-29结束：当前约定范围内已发现缺项均已实现，相关有限检查通过，独立复核未发现新增机制缺项。
本轮工程入口、完整依赖装配、文档导航及许可交付整改已完成，结果见[工程检查记录](../evidence/licensing/dependency-licenses-2026-10-29.json)。
当前收敛完整执行编排、普通／MTP共同机制及SDK接线，随后集中连续后继、GPU融合及带宽等性能工作。
必要的正确性及短回归随实现进行；完整性能窗口放在实质改动之后，不以反复性能实验替代功能开发。
原看板“功能与工程边界95%”已拆为底层功能、工程边界：有效长度prefill按后续已完成实现及有限检查结项；
CPU路由剩余成本覆盖归性能，已安装Graph Source／Store／Append的detail收敛归共同架构。它们不再作为底层功能的未完比例。
2026-10-28的[ProgramIo完整窗口](../evidence/performance/inference-program-io-current-window-2026-10-28.json)
新发现两轮prefill落后；均匀R1也选择packed数学，原indexed-state融合未命中。
均匀R1的冷选择直接A/B与数值核对已完成，保留真实ragged、padding、有界块及未知suffix能力，
结果见下方prefill记录；此后的资源与准备协议迁移不改变数学，也不冒充已测吞吐收益。
随后[当前R1普通／MTP跟踪](../evidence/performance/inference-current-host-profile-2026-10-28.json)已完成四个诊断进程，
MTP捕获数为2／5，旧重录次数不再代表当前瓶颈。仍有准备／提交与GPU活动空隙，实际接受及行数差异保留；
继续做设备后继、小投影等候选，但不以单次Nsight或不同工作量替代完整端到端吞吐对照。
当前主线[新32进程窗口](../evidence/performance/inference-main-peer-window-2026-10-28.json)于15:52～15:59完成普通／MTP单轮与两轮四项及RAM测量。
普通decode相对冻结参考落后2.83%／0.75%，MTP落后6.31%／5.26%；实际输出分叉、接受与行数保留，不能全归因于协议或从旧窗口推导代码收益。
此前13:37的40进程窗口仍留原证据，当前正式表与范围见[进度](../developer/inference-progress.md)。
CPU默认路由已删除旧41点、两个特例和硬件／配方资格机制，由同一facade按原能力与算法选择；
[有限检查与Exp计时](../evidence/performance/inference-cpu-workload-routing-2026-10-28.json)覆盖普通Compute／Graph、IEEE、线程及原码Gather，
其它算子的成本覆盖和跨平台仍需继续，不从局部Exp结果推定完整推理加速。
历史ATen拆分的[本机ON／OFF有限检查](../evidence/correctness/inference-optional-aten-2026-10-28.json)保留原始事实。
用户随后要求不提供能力裁剪开关，现已删除`--aten`／`--no-aten`和`HYLE_WITH_ATEN`公开选项；
CPU／CUDA完整路线按产品依赖自动装配ATen，其它独立模块仍按实际闭包装配。初始化自动清理旧缓存项，
不让使用者选择Provider；同轮复查初始化／构建／测试参数，修复TLS AUTO保持及测试不必要的编译环境前置。
后续[脚本与开关复查](../evidence/correctness/inference-initialization-followup-2026-10-28.json)补齐产品缩减时的休眠协议配置、
完整CPU平台前置检查、已有自定义配置、内部无效选择和显式CUDA工具链约束；75项轻量检查及2项本机预览通过，
未下载依赖或重编SDK，剩余推理架构与性能范围不变。
随后[共同载体安装](../evidence/correctness/inference-program-installation-2026-10-28.json)移除Phase／NextN／prefix／frontier的重复安装回调，
统一exact、进度、固定IO及Store准备，保留各自原复用和事务退休政策。1次CXX、1次链接及6个CPU／CUDA有限检查通过，
包含实际部分安装writer回收与重试；没有新的性能结论。
随后[共同SubmissionViews](../evidence/correctness/inference-submission-views-2026-10-28.json)收拢原Source cohort准备、导出和捕获，
模型保留Session边界、epoch及发布事实。1次CXX、1次链接和3个CUDA有限检查通过，
新增提交后部分捕获失败、外部持有者阻止回收和同会话重试检查；不改变原Completion或新增执行器。
完整模型编排及其它架构、MTP设备后继、GPU热点仍未完成，没有新的性能或质量结论。
随后[exact child资源图准备](../evidence/correctness/inference-resource-graphs-2026-10-28.json)并入原PreparedProgramFamily，
资源图、输出事实及child签名对应一起成功才发布；普通／MTP target和NextN借同一child资源图，Model删除平行数组。
2次增量CXX、2次链接及5个CPU／CUDA有限检查通过，原packed数值门不变，未全部逐字节相同；
此片不增加执行器或热路径扫描，也不代签完整编排、其它平台或吞吐提升。
随后[共同输入与提交推进](../evidence/correctness/inference-input-progress-2026-10-29.json)将原输入绑定、分块、前缀导入偏移和连续提交核对并入RequestProgress，
模型删除两份独立计数，保留真实账簿、容量、采样和Session发布。12组主机合同与5个CPU／CUDA有限消费者通过，
实际部分前缀命中分别核对记录数和token数；MTP、取消、midturn恢复与R2接力保留原状态检查。
一次错夹具入口拒绝保留原记录，改用已有正确缓存后同镜像通过；共4次CXX／链接，未重编SDK或下载模型，没有新增吞吐结论。
[混合lag K1](../evidence/correctness/inference-mixed-lag-2026-10-29.json)已接同一物理draft family，
target均为两行，draft按每成员`1+lag`消费；末行、位置、hidden／token选择及Append使用同一compact坐标。
均匀／R1保留原快路，冷能力不足按原组缩批；实际生成提交保留每成员真实角色与消费行数。
1次CXX／链接和4个CPU／CUDA有限进程通过：混合路径12项完整状态逐字节相同，原ragged／分块17项在既有GDN预算内；
退休Session扫描前拒绝也有实际见证。未实跑完整RequestLoop混合重组，没有新增吞吐、质量或跨平台结论。
这不代签多草稿Verify：原PackedGDN的精确末K合同仍要求每成员实际行数覆盖K。
随后[前缀容量原语](../evidence/correctness/inference-packed-gdn-prefixes-2026-10-29.json)为CPU／CUDA补齐
固定K容量下的真实`S1..SLi`，有效数仅由offsets推导，空成员与剩余槽全正零，不冒充新前缀。
原递推共用，未增加workspace／执行器；FP64、设备中间态继续计算、全量校验与捕获退休等15项有限检查通过。
首轮夹具漏传scale的两项失败保留；修正单行传参后只重编两测试TU、重跑两项，13项既有成功不重复。
引擎仍须接入真实conv history、hidden、逐成员logits及RNG边界，不能从底层通过推定Verify或设备后继已完成。
也不代签MTP设备后继，接受前缀的Store／KV权限、pending及原RNG地址必须另行证明。
GDN a/b N32候选已完成[32进程整模型A/B](../evidence/performance/inference-small-projection-model-window-2026-10-28.json)及266导出张量逐bit核对，
有限样本未证明稳定净收益，不改默认。当前concat形状实验结束，后续原生paired／GEMV需避免额外冷银行与strided物化，不能直接挪用旧叶子收益。
普通／MTP／NextN及采样前沿的有限exact目录与输出事实已统一到共同冷程序准备；
1次增量CXX、1次链接和CPU／CUDA六个现有消费者检查通过，原数值门不变。
[本片记录](../evidence/correctness/inference-prepared-program-family-2026-10-28.json)不代签完整编排抽离或性能提升；
模型形状资格、自然形状规划与完整编排仍待收敛，MTP设备后继和其它热点继续保留。
随后使用计数、冻结上限和有界FIFO接纳政策已迁入共同`ColdShapePlan`，原数学资格及实际载体退休仍归模型。
[3次增量编译／链接及六个有限进程](../evidence/correctness/inference-cold-shape-plan-2026-10-28.json)通过；
主机退休／构造失败契约不代签原生故障，CPU／CUDA实际接纳尚未触发FIFO淘汰。
多草稿末端合图候选已完成四后端84项状态检查和K2随机8进程对照；输出、接受及RNG相同，episode减少26次，decode中位却未提升，候选撤回。
全进程跟踪还观察到kernel净增49、CUDA GraphLaunch多24，不能从引擎episode数推导原生工作减少；固定形状与设备后继仍需独立设计。
产品仅保留本批发现的目标失败后未提交账簿撤回：原工作真实退休后清理其原槽，已提交前缀、head与RNG保持，CPU／CUDA新增回归通过。
候选、最终保留版本及范围分别见[检查记录](../evidence/correctness/inference-mtp-terminal-2026-10-28.json)和[测速归因](../evidence/performance/inference-mtp-terminal-window-2026-10-28.json)。
具体候选允许因测量无净收益而不合入，但必须留下实测结论、适用边界或明确未解决项，不能默默移出任务。
核实不成立的指控也要给结论：例如`deepseek_provider()`当前表示kernel来源与codec合同，路由并不要求
ModelId，不能仅因名字含模型来源就判定存在模型白名单。基础执行接口收敛不以HTTP服务实现为前置。
普通与B2 MTP异长prefill及MTP有限分块／StateOnly已沿原行准备、KV、状态发布及退休接通；
单行中间块、仅首次lag1追赶、不采样的部分检查点、恢复与取消已有CPU／CUDA有限检查。
其它平台与适用组合仍须复查，不能因上一轮MTP完成结论排除。
Apple／Vulkan现有Q1／Q2及随机采样的原生exact回落已有限实跑；Apple冷发现已剪去确定无lowering的族，保留Custom回调、真实预算与原生错误。
组合复查发现的缺项包括MTP逐请求采样配置及logits变换。无modifier的采样配置已沿原Session／冷程序接齐，
CPU／CUDA三配置交错执行的各17项完整状态比较逐字节通过，另通过同配置真实B2／K2及自然入口三配置检查。
本片还修复验证宽度拆分漏传配置索引、完成调用继续持有q，以及按每成员候选数误配cohort概率池；原预算仍生效。
上述无modifier范围见[采样接线记录](../evidence/correctness/inference-mtp-request-sampling-2026-10-28.json)。
逐请求bias／mask现已接通K1／多草稿、greedy／随机、原B2快照及完整前缀重采样；
共用处理只作用于原logits，外部已归一化q不重复变换，p、q及bonus使用同一episode输入。
CPU／CUDA各10项完整状态对照逐字节相同，独立有限分布、拒绝残差与bonus检查通过；
元数据前置拒绝、私有q及前沿数值失败、同key重试、StateOnly和历史重建亦已检查。
上述bias／mask输入是单episode只读[1,V]快照，不表达逐假设前缀动态语法；没有保留外部modifier载荷的严格决策回放继续拒绝。
原生路线新增组合未运行，不从有限检查推导完整任务质量或速度。[本片记录](../evidence/correctness/inference-mtp-logits-2026-10-28.json)。
动态history罚分已补产品接线，并完成CPU／CUDA有限检查。唯一`LogitsPolicy.history`保存repetition／presence／frequency，
中性默认不建counts端口、节点、主机cache或设备池；非中性三项统一使用含已消费prompt的历史范围，不混称vLLM的prompt／output分域语义。
首条件C0为ledger真实`committed_ids()`加本次实际Li或pending；q_i／p_i逐proposal推进，bonus含此前所有假设，lag1catchup不重复计数，q不二次变换。
Ledger的4V主机派生cache由原metadata预算计费，仅按新增提交token同步；原ColdResources的有界C0池按实际cohort使用，NextN双块只保当前更新值。
这不消除每episode全V上传或计数输出，CUDA精确I32 Scan沿原Managed／Commands路线并有等待成本；默认路径不引入这些工作。
StateOnly、同key恢复和完整前缀先seed后采样沿原账簿／退休合同；history-only严格回放已有下述CPU／CUDA有限实测，外部bias／mask仍拒绝严格回放。
CPU／CUDA各通过MTP 17项、普通13项完整状态对照，原modifier配置各10项回归通过；独立小图验证p／q／bonus、非有限输入、重复token和I32边界。
首轮符号view接线错误、测试API和原终止错误观察错误均保留原收据；四个已成功进程未因后续测试修正而重跑。
细节见[共同采样与history合同](../architecture/inference-design.md)及[本片记录](../evidence/correctness/inference-history-logits-2026-10-28.json)。
本轮history-only R1严格回放与K1／K2续算在CPU／CUDA各通过9项完整状态逐字节对照；control各2项，并实际核只读C0别名容量拒绝／释放重试和同key HostImage换出恢复。
MTP生成入口也接入提交后正常取消：原Step确因成员停止而cancelled，须先由原abort证明实际退休，才撤未提交账簿、释放q／C0并标整cohort Cancelled；
CPU／CUDA均已检查真实B2 K2提交后stop A正常返回和原生退休：head／ledger／RNG／pending不发布，peer原key继续。未知／未退、native或其它错误仍带原Status和退休事实抛出，不影响普通多成员幸存者发布合同。
这些检查使用小型合成制品和原generation入口＋RequestWindow，不是新增完整RequestLoop／服务层验证；逐前缀动态语法、其它平台和未知native故障未由此证明，没有trained quality或性能结论。[本轮记录](../evidence/correctness/inference-history-controls-2026-10-28.json)。
E4目标KV＋原BF16 draft的MTP组合已接通：准备器提供两组实际行字节，容量按独立银行计费，
沿原schema／COW／状态及HostImage执行。CPU／CUDA各11次完整状态字节比较、真实B2受控不同接受前缀、
整episode取消后幸存请求续算及默认路径回归通过；元数据与独立容量检查通过。
没有新增draft E4校准、整模型质量、原生平台或速度结论。[本片记录](../evidence/correctness/inference-mtp-e4-2026-10-28.json)。
逐请求停止政策已接齐提案、接受、观察、缓存命中及恢复，host与CPU／CUDA有限完整状态检查通过；该旅程未运行真实模型EOS，ignoreEOS为host规则检查。
普通请求坏logits元数据的分类和原key归属已补，CPU／CUDA双请求及原数值失败回归通过。
停止政策属于请求／账簿，不进入快照或模型状态；采样与logits变换须同时核p、q和拒绝残差，不能只删除入口限制。
普通使用不暴露流水开关；根据实际后端、阶段、采样、资源及依赖事实自动回退。同步与流水应共用准备、
提交、观察和退休机制，以不同在途深度表达；必要fallback保留，被替代的实验分支删除。
内部强制off只供基准与诊断；“尚未接通”与“原理不适用”分开记录，不用模型名字维护排除名单。

共同程序载体已接入普通Phase、prefix sampler、NextN及frontier，复用原Invocation、Queue与retention，
固定绑定全部成功才发布；未知退休保留原handle，退出交接永久禁止复用。
[载体合同检查](../evidence/correctness/inference-program-carrier-2026-10-28.json)保留CPU／CUDA原生提交与失败重试依据。
Phase的一个或三个固定槽直接使用原exact child，Source提交沿同一静态保活算法；保留原交易恢复、
非阻塞退休查询及结果观察后才归还IO的时点，成功不逐步重建。两次CXX、两次链接和五个有限进程通过；
R2额度、接力、接受后取消与幸存者续算保持原完整状态、账簿和RNG，解析条件首错只离线修正。
[Phase记录](../evidence/correctness/inference-phase-carrier-2026-10-28.json)另列未编跑的probe和未验证边界，没有新速度结论。
共同Append现再统一NextN、图内draft和target KV的真实追加范围与多bank尾权预留：调用方元数据只作资格／容量提示，原`reserve_rows`仍是权限权威。
晚Unavailable须全部原预留成功rollback才允许COW，rollback失败不伪装成功；draft按实际尾复用需要准备dirty元数据容量，失败回落才补足COW容量。
三条真实路径已编译并由上述Model检查覆盖，原page／state owner保留；host Append 6项、普通CPU／CUDA各13项完整状态回归通过。不新增owner、Queue、权限表或热回调，不将代码减少视为性能收益。
共同`ProgramIo`已迁NextN、prefix sampler及frontier，收拢原冷IO、fixed bind、动态bind、read／retain及withdraw，
以强类型O(1)handle访问，不新增提交、退休、池或热分配。
独立参考已区分CPU 52bit CDF、CUDA／Apple F32 Gumbel及Vulkan F32尾部公式；先验margin只排除近边界地址，最终ID与数值门不变。
本批CPU／CUDA／Vulkan各10项modifier、17项history、9项replay和2项control通过。
Vulkan私有输出的旧总owner基线误拒绝已修，公共旧reader的3项回归通过。
Apple普通checked Gather的原生引用分类已修，modifier／replay／control共21项状态检查通过；
history的独立有限图原先在MLX冷编译时报映射缺失；随后定位融合后执行tape遗漏实际状态依赖，
现从最终输出图重新后序排序，保留trace输入边界、siblings和全部状态输出，不关闭融合或改数学。
只重编一个MLX文件、复用原消费者后，history的17项原严格状态检查通过；前21项未在新镜像重新运行，
原失败和修复范围见[本次记录](../evidence/correctness/inference-apple-history-compile-2026-10-28.json)。
验证缓存的提交ABI已局部刷新，另有两处等价测试初始化修正。
本片没有新性能结论，不以原生平台的局部通过代替全部组合完成。
合同和实际边界见[设计说明](../architecture/inference-design.md)及[本轮记录](../evidence/correctness/inference-program-io-native-2026-10-28.json)。
模型仍负责图数学、状态、Shape、family与采样键、实际layout、接受策略及Session提交；完整执行核心未抽完。
物理载体集合已迁入共同`ProgramSlots`，将每槽原IO、Store事务、载体及冷使用事实放在同一稳定对象内，
统一冷准备、安装、pending、真实退休和退出根交接；不新增执行器或权限。部分冷失败不发布，模型保留张量、布局及replay政策。
[CPU／CUDA六个有限检查](../evidence/correctness/inference-program-slots-2026-10-29.json)通过7项完整状态字节比较与普通连续流水16臂；
仅重编两个消费者，复用227项SDK输入；未重测吞吐，也不以独立AddGraph消费者代签第二个完整模型。
继续抽离共同执行编排并核实剩余边界；不以本轮有限检查代替完整执行核心或性能结论。
自然请求 wave 的整组准备、兼容分批、缩批与统一发布现已接入共同静态编排；
实际调用结果在成功、取消及异常后归还原记录，不以副本旧值决定发布。
Qwen target／NextN 的相同 FFN 及 exact／packed GDN 尾部数学另合为单份冷构图方程。
模型数学资格、真实 layout、图族、target／draft／repair 及联合发布仍归模型，完整核心尚未抽完。
K>1实际缩宽分组现复用同一成员借用／归还机制，等宽不再建立分组表；早先组提交、后组失败的真实CPU／CUDA旅程已通过，
包括原participant、q租约、退休及原key恢复重试。自然冷需求的成对库存也已合并，四个原入口的266条冷事实与前版相同。
检查范围及两次测试计数误报的修正见[本片记录](../evidence/correctness/inference-selected-cold-demand-2026-10-28.json)，没有新吞吐结论。
下一步优先核固定投影导致既有物理span缓存失效的真实链条及append各bank重复坐标证明，
复用原Invocation／绑定与权限机制；[源码调查](../evidence/research/inference-steady-physical-work-review-2026-10-28.json)只是候选依据，未实现或测得收益。

资源episode现从模型Step拆为共同`ExecutionEpisode`，普通／后继和NextN使用同一原载体提交、观察及失败回收顺序。
模型先保活参与者，后嵌入原资源；前驱alias保留原Step。Append／Store、Source、页尾和额度仍以原权限及真实退休为准，
不新增另一套执行器、权限或自动析构清理。NextN成功保留热carrier，失败真实取回交易后才reset；
正常发布与输出观察仍归原模型和请求层。此迁移不包含形状族、完整准备编排或接受策略的全部抽离。
[本片检查](../evidence/correctness/inference-execution-episode-2026-10-28.json)以3个现有消费者、6个实际CPU／CUDA旅程
核对真实失败恢复、Source、取消与状态；22个测试／probe仅机械迁移访问，不放宽原门。未知退休故障与其它平台未重测，无新吞吐结论。
共同准备进一步接入普通／NextN的batch→Append→消费者／Store→replay流程，保留Source构造及packed offsets的原间隔。
[2个消费者、7项有限旅程](../evidence/correctness/inference-execution-preparation-2026-10-28.json)通过，
32项状态比较中2项字节相同，其余只对新F32 GDN沿原3e-6预算；旧状态、其它载荷、元数据和RNG仍保持精确检查。
普通Phase的固定IO现也消费已有ProgramIo，四个readback保留唯一owner；正常Store完成沿原Graph整cohort取回，
单项／部分失败恢复与Imported Source边界保留。[本片有限检查](../evidence/correctness/inference-store-cohort-2026-10-28.json)通过，
单轮普通／MTP另完成16进程严格同事实A/B；中位小幅变化但范围重叠，不以此代签整Model抽离或稳定提速。

自然prefill入口按真实需求冷选：已知整段单请求使用原exact图，多成员、有界块或未知prefix后缀保留packed能力。
32进程直接A/B已完成，单轮prefill约提升32%～34%，两轮观测约43%；普通输出有既存数学路径差异，
共同前缀诊断与原严格差异均保留，不能用可读续文证明质量等价，也不以此签整段同算量decode收益。
MTP本窗接受与实际工作相同；两版decode变化约0%～2%，仍需后续热点与同窗完整参考判断。
真实有界普通／MTP和prefix scheduler仍走packed，原数学及预算有限检查完成；
详见[本片A/B](../evidence/performance/inference-uniform-prefill-current-window-2026-10-28.json)。

变长Verify已接原固定容量、前缀Store和NextN repair：不同Li共用真实目标批，按实际行取logits／候选及推进RNG，
等宽／R1和冷能力缺失的原分组保留。K3的CPU／CUDA各6项完整状态比较通过，另有mixed-lag和ragged prefill回归；
旧状态和非GDN载荷精确保持，新F32 GDN继续使用原3e-6预算。
[实现与有限检查](../evidence/correctness/inference-varlen-verify-2026-10-29.json)不代表设备接受、MTP连续后继、完整RequestLoop重建或吞吐完成。

初始化／构建选项复查已按同一产品闭包原则收敛：完整路线的必要Provider自动装配；完整io的协议能力不再另设关闭门。
多配置生成器的Debug缓存选择已修正，脚本／元数据及无下载依赖边界检查见[本片记录](../evidence/correctness/inference-automatic-io-2026-10-29.json)；
其范围不代替完整协议库重编或package安装检查。
随后[初始化旧缓存复查](../evidence/correctness/inference-init-options-review-2026-10-29.json)补齐其余六个旧路由／Provider键：
旧配置在build/test前提示同名重配，init自动清理并保留产品、后端及其它选择。52项脚本与CMake元数据检查通过，未下载或重编。
输入诊断另发现错误产品名可能先准备Vulkan宿主工具，仍待复用中央产品目录提前拒绝；不引入第二份硬编码目录。

递归采样前缀新增共同设备折叠与单I32结果包，完整valid、真实quota／stop、pending及RNG规则保留；
greedy modifier同步合包，但原精确主机比较保留。CPU／CUDA独立输入图与组件状态／失败恢复检查通过，
详见[本片依据](../evidence/correctness/inference-mtp-device-packet-2026-10-29.json)。尚非设备KV或连续后继，不声明速度提升。
K1完整接受的[投机后继组件](../evidence/correctness/inference-mtp-k1-source-2026-10-29.json)现已接通：四组CUDA后继在A主机完成前提交并复用捕获，部分接受、额度结束和取消时丢弃B，完整状态及原快照保持；CPU仅证明提交前Source不支持时原串行继续。定位并修正显式Store ID换址提前清除capture key，保留原容量、绑定和权限门。
[自然接入](../evidence/correctness/inference-mtp-natural-2026-10-29.json)已将CUDA greedy K1纳入普通共用continuation与RequestLoop：原权重40输出和23条完整账簿与串行一致，实际提交3个后继、发布2个并因真实部分接受丢弃1个。默认auto／N1入口通过；成功发布后释放模型私有full-prefix票据，使原5个GDN槽足够，不扩池。其它后继配置和任意接受长度的设备续算仍在范围内，无新性能结论。

多草稿账簿的替换式SmallVector.assign会丢弃冷容量，现已在13处记录写入原位resize／copy，不改容器语义。
[冷存储检查](../evidence/correctness/inference-ledger-storage-2026-10-29.json)以预留宽度8覆盖CPU／CUDA实际Verify、撤回重备和完整记录值，
另通过真实Qwen K1自然流水及CPU实际失败／主机恢复／同键重试。13项完整状态比较保持原F32 GDN预算，未全体逐字节相同；
1次CXX／链接及4个有限进程，无SDK重编或下载。默认K1使用内联容量，无相同堆分配；该成本尚未测量，不预设为当前decode主因。
随后[逐成员实际行数](../evidence/correctness/inference-batch-member-rows-2026-10-29.json)已修复自然入口按批次最大宽度逐成员计数的错误。
PreparedGenerationCall分开规划上界与实际提交行数，报告不按接受前缀缩小已执行工作；CPU／CUDA原生命令行用输出额度3／4自然形成Verify[2,3]，
5个活动行、6个物理行、顶层与逐请求账簿一致，完整输出7个token。typed BatchObserver另核对[1,3]，
13项完整状态比较保持原数值预算，真实Qwen K1接力回归通过；共1次CXX／链接和5个有限进程，无SDK重编或下载，无新性能结论。
继续复查确认history推进与episode快照不能代签逐条件动态约束，原98%估计偏高。
[逐条件输入首片](../evidence/correctness/inference-conditional-logits-2026-10-29.json)已抽出共同typed输入，冷准备独立逐行端口，真实递归提案按行采样，原terminal复用同一bias／mask和原q租约；C0唯一并按实际草稿推进。
CPU／CUDA有限检查覆盖接受、拒绝、bonus、history计数、greedy、EOS截短、完整元数据预检及非首行全mask失败／重试，head／RNG／旧检查点保持；原共享modifier和真实Qwen K1自然接力回归通过。
首片共1次CXX／链接和5个有限进程，无SDK重编或下载；当时条件目标logits为有限合成输入，不代签完整发布。
随后[真实Verify发布](../evidence/correctness/inference-conditional-verify-2026-10-29.json)已将不可变条件行沿原EvaluationRequest交给StepMember，候选创建前完整检查实际行数、元数据和冷程序，纯输入错误保原RequestKey；租约和原生错误仍由原机制处理。
原terminal读取成员条件行，完整target／draft及账簿仍沿原提交／发布／失败退休。CPU／CUDA有限R1检查实际接受3／2行，以真实目标F32 logits和原q独立核对决策；真实非首行全mask失败后同key／原q重试，旧检查点不变，全部版本和页归还。当前3项完整状态对照均逐字节相同，仍保留既有GDN数值预算而非推广逐字节保证。
本片2次CXX（首轮Result错误返回写法失败已修并保留）、1次链接、5个有限进程，无SDK重编或下载；没有新吞吐结论。
随后[自然条件输入](../evidence/correctness/inference-natural-logits-2026-10-29.json)把请求闭包接进原RequestLoop和条件K1／Verify：当前turn输出含pending，未来假设不重复pending，StateOnly不调用；q_i／p_i保留同一条件行，静态K1及默认路径不变。
CPU／CUDA有限检查覆盖K1与随机K2的两请求／两轮、N2三请求缓存完整命中使用当前新策略，以及两条真实q产生后提供失败、实际退休与原key重试；head／账簿／RNG／旧检查点保持，最终版本及页归还。
29项完整状态比较通过；随机两请求先独立核验实际seed／stream／draw，完整状态比较只在期望副本转换已知stream标识，不改实际结果。
共1次CXX／链接和5个有限进程，无SDK重编或下载。两次离线解析错误（完整JSON分行、缓存目录析构前版本数）保留并修复，原生均成功且未重跑。
随后[条件输入控制](../evidence/correctness/inference-natural-logits-control-2026-10-29.json)完成CPU／CUDA终止前缀、静态／动态同一R2 Verify、提交后整episode取消及幸存者同key主机恢复；
终止后零提供者调用，取消不发布原head／账簿／RNG，真实退休后清q与条件owner；换出时驻留版本归零，恢复后从原输出继续。
6项控制完整状态对照使用原检查点与相同实际cursor，加原自然输入和真实Qwen接力回归共33项，保留原数值门；1次CXX／链接、5个有限进程，无SDK重编或下载，无新吞吐结论。
功能独立复核确认旧Vulkan非4字节文件输入已由[对齐修复](../evidence/correctness/inference-stream-padding-2026-10-26.json)覆盖，不能沿用旧OPEN；token、终止、取消、credit／ACK及原key分类失败行为已有原共享实现和[控制依据](../evidence/correctness/inference-request-controls-2026-10-26.json)。
当前约定底层范围的已发现功能缺项均已补齐，功能阶段标100%；新条件输入实际检查CPU／CUDA小制品，不代签Apple／Vulkan最新组合、全部制品／量化／平台或未知原生故障。
共同输出入口、完整模型编排、SDK detail收敛和普通／MTP后继及吞吐余差继续属于已列架构／性能工作，整体goal不结束。职责见[架构合同](../architecture/inference-design.md)。

## 当前实现概况

以下区分底层功能、共享架构和性能，不用某项功能通过代替三者全部完成：

| 部分 | 功能 | 架构 | 性能 |
|---|---|---|---|
| 多会话／多请求底层 | 独立状态、有界缓存、前缀复用、重算恢复、真实合批与失败退休已接通；完整主机镜像、普通prefill分块、有限尾块与已计划载体准备已有列明有限检查 | 请求循环、兼容分组、缓存接纳及执行错误身份已共用；适配器保留实际形状、完整状态与恢复，普通单成员取消已有有限CPU／CUDA检查 | 已有审查和有限测量，尚未收口 |
| MTP | 约定本机功能范围已完成并复审；保留各制品有限运行与大模型硬件边界 | 与普通推理共用生成、请求与完成合同；上一轮有限收敛已完成，本轮继续调整共享底座 | 已有正式对照、定量归因与候选取舍；仍落后参考，不表示性能余差已消除 |
| 图片多模态 | 完整图片投影链尚未开工 | 尚未收口 | 未作完整对照 |
| 普通基础推理 | 新Review的状态、恢复及本轮形状政策已实现并有限检查，未运行范围单列 | 共同wave与普通CUDA跨pump连续后继已有有限检查；完整执行编排仍主要在Qwen Model，稳态准备及在途权限统一尚未完成，CUDA greedy K1自然后继已接通，其它MTP后继仍未完成 | 有正式性能窗口和主机侧归因，当前性能狩猎仍在进行 |

服务器 API 和连续服务应用属于后续方向；底层多会话、多请求、接纳、缓存与真实合批均纳入本轮全局架构，不能推给后续 server。

| 顺序 | 上一轮工作 | 已交付的有限范围 |
|---|---|---|
| 1 | 统一工程与文档 | 工程组织已完成：引擎归入 `engine/`，默认选择 `HYLE_COMPONENTS=engine`，target `hyle_engine` 输出 `hyle`；保留独立模块构建，已通过配置／目标组织检查及 util 实际构建安装；CPU 运行报告余项见下文，不代签完整运行 |
| 2 | 穷尽现有环境可做的 MTP 功能 | 已完成约定范围并复审：Qwen27 FP8 完整普通／K1 随机 CPU 已实际通过；量化、采样、多草稿、真实混合及状态边界保留各自证据 |
| 3 | 收敛整个推理架构 | 本轮有限收敛完成：生成合同、请求窗口、typed 目录、turn 输入／额度／提交观察及同步入口 IO 已共用；Qwen／GLM Pro 原状态路径和有限扩展检查已核。模型保留真实阶段、完整状态和容量证明；范围见[整体设计](../architecture/inference-design.md) |
| 4 | 架构收敛后优化性能 | 本轮有限调查完成：完整普通／MTP 四臂基线、分层归因和已有候选均已取舍；固定分块的 MTP decode 小窗 +1.40%，因收益与容量／数值边界代价未合入。未改产品，不重复相同源码的全窗；余差与后续优化仍见[性能资料](../developer/performance/decode.md) |

上表同时区分当前底层复查与上一轮 MTP 结果，下文保留上一轮 MTP 与有限全局收敛依据。当前阶段不沿用该轮完成百分比，普通推理、多会话、多任务的[底层反向复查](../evidence/research/inference-functional-review-2026-10-25.json)与剩余性能工作统一维护在[推理看板](../developer/inference-progress.md)。
本地元数据与缓存的[范围资料](mtp-local-scope.json)保留具体身份与缺失字节；旧的可选／延期标签不能替代本轮可行性调查。
完成前重新独立审查：所有已发现且现有环境可实现的功能都须完成，新增缺口进入同一 goal。
无法装载整个大模型时，仍须处理本机可运行的数学、状态、量化角色和组件缺项。

通用 loader、C++ tokenizer、server、应用 API 和聊天体验属于后续应用方向，不是当前阶段的完成条件。
准备输入接口不顺只记录其对真实模型驱动的影响；不为追求应用功能齐全阻塞底层支持。
`util` 只是独立构建的例子；所有公开库产品均应按其依赖闭包构建，不能把模块化能力限定于某个示例。

## 当前 MTP 功能范围

| 项目 | 当前实现与有限检查 | 功能完成标准 |
|---|---|---|
| CPU 完整流程 | CPU 普通／MTP 数学路径及无 CUDA 运行依赖的独立闭包已通过 | 保留有限预算与状态寿命检查，完整正确执行相关新功能 |
| 本机量化完整路径 | GPTQ／NVFP4 CPU 与 CUDA 短旅程已跑；NVFP4 差异已定位 attention 中间精度；Unsloth GGUF Q4 普通／K1／K4 随机、Q8 普通／K4 随机 CPU 与 CUDA 均已跑；CT 对称 INT4 已完成五条 CPU／CUDA 短旅程 | 全文本主干、原 MTP、词表及接受后续算真实执行；只取确实缺少的文本角色，不下载视觉 |
| 随机采样 MTP | CPU／CUDA 已补多草稿、RNG、恢复与边界；Apple／Vulkan 均新增 K4 随机和真实双用户 K2 入口，Vulkan 有效质量误拒已修 | 明确温度／过滤后的真实分布；取消、重建与批次重排不混用会话 draw 身份，检查目标分布与边界 |
| Qwen3.5 完整多草稿 | 共享机制与 CPU／CUDA／Apple／Vulkan 代表路径已完成；具体未运行制品见量化行，其它模型组件接线另列 | 任意接受长度均选取完整状态，正确处理拒绝修复、EOS 与剩余额度；不以提速作为功能成立条件 |
| GLM／Qwen24 MoE 多草稿消费者 | GLM Flash／Pro 与 Qwen24 MoE 原组件真实 K2、完整接受前缀／续算均已通过代表 CUDA 旅程；Qwen24 target 主干逐行执行，词表 head 实际 M3 | 核实际递归合同，复用公共接受／前缀规则，补本机可执行的第二 proposal、target 验证、合法前缀完整状态与拒绝修复；不扩大逐发布者／后端矩阵 |
| 草稿控制与块尾 | CPU／CUDA 完整 Qwen 已补长度／概率阈值／禁用回退；DeepSeek Pro／Flash 原组件已有 greedy 宽度 2～5 与尾部证据；GLM／DeepSeek 随机 EOS／额度／上下文组合缺口已补并完成实际有限状态旅程 | 尊重训练头与置信值合同，覆盖长度截短、全拒绝／全接受、EOS／bonus EOS、context／quota 尾部和继续运行；接纳后失败不能静默重做 |
| 配置与输入边界 | 完整库存、普通／MTP 选择、真实容量及 token／量化 block 前置拒绝已完成 | 仅加载所选角色；按真实上下文与整数边界接纳，缺失角色／专家／codec 明确失败 |
| CPU＋GPU 完整组合 | CPU＋CUDA 普通、K1／K4、随机、多会话及恢复已通过有限旅程；CPU＋Apple／Vulkan 的原 BF16 完整随机 K1、跨域数值及退休重绑也已通过；Vulkan copy 的解绑缺口已修并有旧对象负控 | 所选三类 GPU 的明确 CPU FFN 分置均实际装配，保真实搬运、状态、预算与失败语义；不要求任意自动放置 |
| FP8 完整投影接线 | block128 已接完整角色、逐组导入与固定计算策略，27B 完整普通／K1 随机 CPU 旅程已通过；CT channel／token 与 local NVFP4 保留原片段和小型完整加载范围 | 分清 block128 与 channel／token 合同，按原值冷导入，保留实际激活与输出精度、物理角色统计及有界准备内存 |
| 新路径的正确性覆盖 | 完整状态、条件分布及固定任务检查已有有限证据；跨系统、混合路径、原组件及 Qwen27 GGUF／FP8 完整入口已补齐，约定功能范围最终复审完成 | 按真实变更检查接受／拒绝、多轮、代表性 B2、多会话淘汰恢复、取消及数值／任务质量；复用旧机制，不重复全组合测试 |

共享编排和本轮跨系统入口的有限证据已齐。Qwen27 FP8 文本＋MTP 约 27.9GiB，在当前主机实际完成普通与 K1 随机 CPU 旅程，见[完整原权重记录](../evidence/correctness/mtp-qwen27-fp8-complete-cpu-2026-10-24.json)。
此前遗漏的完整入口已补齐，不能归入未来通用 loader。最终源码与证据复审未发现其它明确、本机可实现而未完成的 MTP 功能项；本轮全局架构已作有限收敛，性能也已完成正式对照、分层归因与有限候选取舍。仍有性能余差，不把本轮完成解释为已经追平参考。
这一范围包含 MTP 必需的普通公共链路，不等于普通推理引擎全部功能已穷尽。
完整 Qwen3.5 的 K2／K4 和共享接受规则，不能代替这些组件的第二 proposal、宽验证与状态提交。
先按固定版本核齐原 header 和必要文本范围，不为重复同一数学合同自动获取其它发布者的整套权重。
现有 RNG、采样、快照、分页和迁移原语继续复用，只有真实路径暴露缺少的底层能力时才新增原语。独立公共代码迁移保留冻结功能基线；数学、精度与性能改动仍待相关功能确认，
也不把小模型能力提升、通用服务或任意 speculative 算法列为收口条件。

## 本轮可用范围

Windows 当前机器有约 64GiB RAM 和 RTX 4070 12GiB 显存，优先复用已有 Qwen3.5-2B 输入和 CUDA 构建。
Apple 验证机为 MacBook Air `Mac16,12`、16GiB 统一内存、ARM64，复用已有缓存；CPU 路线单独记录实际结果。
DGX Spark 不在当前可直接调试的环境中。不能以公司设备的预期容量安排本机完整大模型验证。

本轮工程检查：7 项引擎组织、15 项路由、16 项 SDK 与 44 项脚本检查通过；`util` 离线配置、46 步构建及安装通过，未取得 GPU／推理依赖。CUDA 正式入口普通／MTP 各完成 P154＋4 输出，属于短工程检查，未刷新性能窗口。
CPU 原失败根因已修：布局只读元数据原被归为 External，在相同 host 资源中混入缓存 payload。修复保留严格库存断言和真实有限预算／退休，30 项 BatchLayout 测试及普通两种清单／MTP 各 P154＋4 完整进程通过；结果与边界见[检查记录](../evidence/correctness/qwen-mtp-cpu-layout-budget-2026-10-24.json)。这不是长旅程、任务质量或性能窗口。
本地命令与失败记录保留在 `.scratch/engine-product-20261024/`，不是新 MTP 能力或性能通过证明。

大型 GLM、DeepSeek、Kimi 的完整推理受本机容量限制。已有原角色、尾层、MTP、专家供给、
状态与量化组件结果继续保留；真实存在的缺口按其所属数学、状态或格式记录，不重新下载整模型凑覆盖数量。

## MTP 的已完成项与边界

[有限制品清单](../evidence/research/mtp-artifact-scope-2026-10-23.json)的 25 个已确认身份已完成有限原角色／词表／
专家与完整状态流程；Qwen 有真实完整模型，GLM／DeepSeek 等仍有上游输入和初始历史 fixture 边界。
这不表示全部模型、量化、后端、任务质量与吞吐已穷尽。

本轮新增的 17 个固定制品已完成身份与角色盘点：14 个具有完整 MTP 结构，3 个完整 checkpoint 清单没有 MTP；未确认项为零。结构核查不等于整模型执行。
新增已读原结构包括 Qwen3.5 CT INT4、Unsloth MTP GGUF、Qwen27 官方 FP8 与 Unsloth NVFP4 的独立 draft。
Qwen27 FP8 的 draft 含 FP8 投影与 BF16 block scale，NVFP4 命名制品的 draft 实际全为 BF16；选定 BNB 制品没有 MTP 角色或独立文件。具体来源与未确认项以范围资料为准，旧目录数量不作为能力上限。
先读缓存和固定 metadata；缺少真实角色后再汇总必要的有界获取，遵守 [AGENTS.md](../../AGENTS.md) 的下载限制。
Kimi 当前固定来源未建立 MTP 需求，不因模型名字额外创建 MTP 实现任务。
当前完整 Qwen 的 CPU／CUDA K1 已接随机采样、概率接受、拒绝残差和会话 RNG，并检查前缀重采样、取消与恢复。
有界多草稿共享机制、GLM K1／K2、Qwen24 MoE K2 及 DeepSeek 随机控制组合已有有限证据；Qwen27 GGUF／FP8 完整有限旅程已补齐。各证据的实际接受／拒绝分支、制品与硬件边界不外推。
新增范围分别见[多草稿与控制](../evidence/correctness/qwen35-mtp-multidraft-checkpoint-controls-2026-10-24.json)、
[Apple K1](../evidence/correctness/mtp-qwen35-apple-k1-complete-input-2026-10-24.json)与
[Vulkan K1](../evidence/correctness/mtp-qwen35-vulkan-k1-complete-input-2026-10-24.json)。
[Linux CPU](../evidence/correctness/mtp-qwen35-linux-cpu-v2-k1-complete-input-2026-10-24.json)已运行普通 greedy、MTP K1 greedy／random 三条完整短入口，复用原权重与 CPU 原生缓存；不是性能测量。
[macOS 纯 CPU](../evidence/correctness/mtp-qwen35-macos-pure-cpu-complete-input-2026-10-24.json)刷新必要的当前 CPU／原生对象后，普通 greedy 与 K1 随机两条原权重短旅程均正常退出；K1 实际到达 EOS，完整状态、pending 与 RNG 正常提交。原权重全部在 host，未链接 Apple／Metal／MLX 数学或 GPU 队列；保留首次参数错误和准备失败，不将有限缓存预算冒充物理内存上限。
[Linux CUDA／混合](../evidence/correctness/mtp-qwen35-wsl-cuda-mixed-complete-input-2026-10-24.json)完成当前闭包的普通纯 CUDA 与 K1 随机 CPU FFN2 两条 P154＋4 入口；MTP 实际接受 proposal 并继续单行尾部，target／draft frontier 均为 157、lag 为 0。局部刷新真实 CUDA ABI 差异，修复一个 GCC 严格告警下的成员默认初始化遗漏；保留首次编链失败、有限预算及未知原生私有开销，未刷新性能基线。
[DeepSeek 短验证](../evidence/correctness/deepseek-mtp-short-verifier-tails-2026-10-24.json)补齐 Pro／Flash 原组件宽度 2～5 的真实状态发布、拒绝后续算与尾部策略；不代替完整大模型生成。
[接受前缀状态对照](../evidence/correctness/qwen35-mtp-prefix-state-values-2026-10-24.json)在 CPU／CUDA 各检查 17 个边界与 7 对未消费后缀扰动，完整状态、lag1 追赶和同 T1 续算均与同宽独立参考一致；不要求不同批宽逐字节相等。
[多行随机采样](../evidence/correctness/qwen-mtp-multirow-sampling-2026-10-24.json)复用产品实际构图，CPU／CUDA 各检查 m0／K2／K4、逐行 p／q、拒绝残差、bonus 与坏输入恢复；独立 F64 与 RNG 参考一致。它保留供入 verifier 的过滤后条件分布，不承诺其它批宽计算出的分布或同 seed 输出相同。
[新增原值组件](../evidence/correctness/mtp-original-source-format-tiles-2026-10-24.json)与
[Quark W4A4](../evidence/correctness/mtp-qwen24-quark-original-tile-2026-10-24.json)分别检查实际编码、scale 和计算合同，不从制品名称推断精度。
Qwen27 原 GGUF 的 9 种压缩 codec 已具备 CPU 路由，64GiB 主机的压缩常驻路线可行；
复用已有原角色，只获取缺失连续范围，不展开整套 BF16／F32 权重。
[共用几何接线](../evidence/correctness/qwen-source-geometry-v2-execution-2026-10-24.json)已完成并通过 2B CPU／CUDA／混合检查。
[27B GGUF 完整旅程](../evidence/correctness/mtp-qwen27-gguf-complete-input-2026-10-24.json)已通过 CPU 普通／K1、混合普通／K1／随机 K2；原 866 角色、完整文件哈希及有限缓存均已实际核对。
混合 MTP 暴露 CUDA 融合缺少整段 checked-index 输入边界资格，已在放置前排除无效候选，保留单算子安全检查与原搜索上限；短旅程不冒充任务质量或吞吐对照。
CPU＋CUDA 的代表 BF16 FFN 分置已完成，具体版本、原权重放置、静态队列、已知与未知内存范围见
[完整分置记录](../evidence/correctness/qwen35-cpu-ffn-cuda-placement-2026-10-24.json)。其暴露的
[process 常量租约缺口](../evidence/correctness/graph-process-system-constant-lease-2026-10-24.json)
在 Graph 原有借用／异步复制流程内修复，未另建执行器或复制 GPU 权重银行。
[CPU＋Apple／Vulkan 分置](../evidence/correctness/mtp-qwen35-cpu-native-placement-2026-10-24.json)各完成原 BF16 2B 随机 K1 短旅程，CPU FFN 分别为两层／一层；跨域 BF16、资源退休和重复重绑均有定点核查。Vulkan 原 copy 缺少解绑合同，现按原 Run／Completion 真实退休释放 VkBuffer；同一永久回归在新对象上通过、仅换回旧 copy 对象时准确失败。原生 replay DMA 的四字节对齐限制单列，不把未执行的 BF16[17] 冒充通过。
输入边界补齐了[GGUF 无 NextN 的明确拒绝](../evidence/correctness/mtp-gguf-source-capability-2026-10-24.json)与
[Linux 早读 CRLF 清单](../evidence/correctness/mtp-source-geometry-newlines-2026-10-24.json)；两项只用元数据检查，不为其重跑完整推理。

[Apple／Vulkan 多草稿](../evidence/correctness/mtp-qwen35-native-multik-random-complete-input-2026-10-24.json)各补 K4 随机和真实双用户 K2；同时修复 Vulkan 的 F32 主机谓词读取及近似除法造成的合法概率质量误拒，保留坏输入拒绝。
[压缩 head 与固定投影](../evidence/correctness/mtp-fixed-width-packed-head-2026-10-24.json)补 NVFP4 独立词表 head，CPU／CUDA 原值一致；固定激活下 T1／T3／T5 的 8 个差值符合既有数值合同。
[GGUF Q4 CPU](../evidence/correctness/mtp-qwen35-original-gguf-q4-cpu-2026-10-24.json)与
[CUDA](../evidence/correctness/mtp-qwen35-original-gguf-q4-cuda-2026-10-24.json)补齐完整原权重普通／K1／K4 随机入口；修复压缩 embedding 的 checked Gather 接线及新资源作用域的正常析构顺序。上述均为有限功能证据，不是性能对比或完整任务质量结论。
Q8 已完成原文件 LFS 和 335 角色校验，[纯 CPU](../evidence/correctness/mtp-qwen35-q8-original-pure-cpu-complete-input-2026-10-24.json)与 [CUDA](../evidence/correctness/mtp-qwen35-q8-original-cuda-complete-input-2026-10-24.json)各运行普通／K4 随机两条完整短路径；K4 实际消费 4 个草稿、RNG 游标为 10，真实物理计数为 320／15。两后端此次 ID 相同，不外推跨后端数值或全部任务质量一致。

[Apple 分配控制器](../evidence/correctness/apple-memory-allocation-controller-lifetime-2026-10-24.json)在原生包装器销毁后仍由真实分配保活；两项定向检查通过，不外推为未知在途 Model 的全部边界。
CUDA 公共 `Float8BlockLinear` 的 F32 动态 Power2 组合缺项已复用既有 F32 量化和编码矩阵乘补齐；这与源制品采用何种激活精度分别记录。
[原生退休与资源作用域](../evidence/correctness/mtp-native-retirement-resource-lifetime-2026-10-24.json)补齐部分提交、未知观察、应用对象先销毁和最后回调释放；5 项公共 Graph 与 2 项实际小 Model 检查通过，构造异常与快照逃逸另经源码／编译核对，未声称完整故障矩阵。
[输入与角色错误边界](../evidence/correctness/qwen-mtp-input-role-error-boundaries-2026-10-24.json)在加载前按真实词表／容量及完整量化 block 拒绝坏输入；18 项实际输入函数检查与 303 项元数据检查通过。
[共享控制与缓存复核](../evidence/research/mtp-controls-cache-completion-review-2026-10-24.json)将长度／阈值／禁用、EOS／尾部、多会话、初始及生成后前缀复用对应到实际实现与已有证据；这些共享机制已完成，余项落到具体制品、部署闭包、任务质量和最终复审。
[CUDA F32 动态 Power2](../evidence/correctness/cuda-fp8-f32-power2-linear-2026-10-24.json)补线已通过 CPU／CUDA 定向检查，保留真实 F32 输入及输出，不插入 BF16 中间值；复用既有原生量化，不改变 kernel ABI。
[FP8 原制品接线](../evidence/correctness/mtp-qwen-fp8-block128-prepared-binding-2026-10-24.json)补齐 block128 typed roles、明确 reader 数学身份、BF16 scale 精确扩展和逐组冷导入；CPU／CUDA 的 Qwen27／Qwen24 原小片段符合独立数值参考。[27B 完整 CPU 旅程](../evidence/correctness/mtp-qwen27-fp8-complete-cpu-2026-10-24.json)进一步实际加载 1273 角色，普通与 K1 随机均完成；保留旧缓存链接遗漏的首次失败及仅补正确 mapping 对象的修正，不改数学。本轮 MTP 实际提议并接受一次，拒绝未自然观察，不冒充质量或性能对照。F16／F32 词表 head 的精度报告同时纠正，保留旧冻结 Q8 的原报告及实际计算说明。
[CT 混合制品接线](../evidence/correctness/mtp-qwen-ct-mixed-prepared-binding-2026-10-24.json)补齐 channel／token 的 BF16 QDQ、local NVFP4 原 global 乘数与 F32 量化前舍入、词表 head 和明确的 BF16 KV 辅助角色省略。CPU／CUDA 各三个原片段符合独立参考，CPU 普通／MTP 小型真实加载通过；原被省略角色的名字、形状和 hash 仍进入身份。未执行该制品整模型。
[Apple 概率质量门禁](../evidence/correctness/mtp-apple-sampling-mass-validity-2026-10-24.json)另经真实 MLX 小图检查有限非整数质量及零质量／NaN／Inf，25 项谓词符合预期，不重跑完整模型。

旧 CPU 完整运行只证明 CPU 数学路径，测试程序来自混合对象闭包并导入 CUDA 库。
[纯 CPU 部署闭包](../evidence/correctness/mtp-qwen35-pure-cpu-deployment-2026-10-24.json)现按产品 CPU factor 解析复用缓存，刷新必要的原生与 ABI 边界对象；实际 Q4 普通／K1 运行及 PE／加载模块均无 CUDA 依赖，不声称全新安装或可重定位部署已经验证。
[GPTQ／NVFP4 K4 随机](../evidence/correctness/qwen35-original-quantized-mtp-k4-random-2026-10-24.json)补齐两份完整原制品的多草稿短旅程。旧冻结程序的报告写死 BF16 张量数量，保留原日志并以真实清单计数；当前报告已改为实际加载的物理角色计数。
[CT 完整量化入口](../evidence/correctness/qwen35-original-ct-mtp-2026-10-24.json)已取得 649 个原文本／MTP 角色，完成 CPU 普通／K1／K4 随机及 CUDA 普通／K4 随机五条短旅程；不获取视觉。[随机分叉定点归因](../evidence/correctness/qwen35-ct-categorical-backend-attribution-2026-10-24.json)用两行真实 q2 交叉重放 CPU CDF 与 CUDA Gumbel 抽样：两行均分别选 1445／464，重现首块接受长度与第五个输出分叉，且远离数值临界。该差异符合既有 Categorical 跨后端合同；两行分布本身仍有差异，本例不认证全部 CT 数学或任务质量。
[固定完整任务](../evidence/correctness/qwen35-mtp-finite-task-check-2026-10-24.json)预先确定三个任务与判定器，普通／K1／K4 共九次均正常 EOS，同任务的完整输出 IDs 一致；JSON 抽取与 Python 执行检查通过，算术三臂共同答错。
[同源 llama.cpp 对照](../evidence/correctness/qwen35-mtp-finite-task-reference-2026-10-24.json)在同一 47-token 算术输入也输出相同错误答案；该例不是 Hyle／MTP 特有失败，仍保留任务失败，不外推普遍质量或模型能力结论。
[GLM Flash／Pro 随机组件](../evidence/correctness/mtp-glm-original-stochastic-state-2026-10-24.json)已将真实 NextN 的 q 提议接入 target 验证，完整状态、pending 与 RNG 在原 Head 原子发布，并实际消费 pending 续算；Flash 另实际执行接受后的 bonus，Pro 本轮自然试验为拒绝。
[GLM 有界 K2](../evidence/correctness/mtp-glm-original-stochastic-k2-state-2026-10-24.json)现将真实递归第二 proposal 接入 T3 原词表投影；Flash／Pro 两条 CUDA 原角色旅程通过 mandatory-current 之后 0／1／2 枚已消费草稿的完整状态修复、两条 journal 顺序 catchup、EOS／额度、取消／stale 与 RNG 原子发布，并实际消费残差或 bonus 继续运行。固定 reader 支持递归，默认 K1 与较短尾部分流保留；全接受使用明确的合法受控 RNG 地址，保留初始历史和上游输入 fixture，不外推自然接受率或整模型质量。
DeepSeek 官方 [Flash](../evidence/correctness/deepseek-mtp-source-stochastic-flash-2026-10-24.json) 与 [Pro](../evidence/correctness/deepseek-mtp-source-stochastic-pro-2026-10-24.json)已将真实采样的前驱 ID 接入原 Markov head，并用真实 p／q 选择残差或 bonus 后续算。[两种量化侧车](../evidence/correctness/deepseek-mtp-source-stochastic-sidecars-2026-10-24.json)已保留各自真实 F32／BF16 输入边界并实际续算。上述均为有限原角色组件，保留上游激活与初始历史 fixture 边界。
[随机块尾与 confidence](../evidence/correctness/deepseek-mtp-stochastic-controls-tails-2026-10-24.json)补齐已发现的组合缺口：接受 EOS／额度末 proposal 保留原 lease 为 pending，按真实 consumed 状态与 RNG 原子发布；Flash 实际组件及 25 项策略／缓存检查通过，最终两类 DS 入口与 Qwen 编译通过，不外推逐制品运行。
[GLM 随机输出边界](../evidence/correctness/mtp-glm-stochastic-output-boundaries-2026-10-24.json)在原 Flash／Pro 组件实际检查单行 target 抽样、接受／残差／bonus EOS 与额度提交；单行维护 mandatory-current 的完整 NextN 状态，不抽草稿、不执行 second。Qwen、GLM 与 DeepSeek 共用 `engine/inference/mtp_acceptance.h` 的纯前缀规则，继续复用原执行器和状态容器。
Qwen27／Qwen24 原组件的共用控制器发现只接 greedy 的缺项，[Q24 原 GGUF MoE 随机续算](../evidence/correctness/mtp-qwen24-original-stochastic-state-2026-10-24.json)现已实际通过；[CPU 词表混合路线](../evidence/correctness/mtp-qwen24-mixed-stochastic-state-2026-10-24.json)取得实际触发的必要专家后，随机拒绝、残差 pending 及再次消费续算均通过，完整状态和 RNG 原子发布正常。[八种制品薄入口](../evidence/correctness/mtp-qwen-stochastic-wrapper-routing-2026-10-24.json)已透传随机选项并通过 CPU／CUDA 严格编译，不冒充逐制品实跑。[原组件随机尾部](../evidence/correctness/qwen-original-mtp-stochastic-controls-tails-2026-10-24.json)现已接共用 limits／前缀规则，实际 Q24 GGUF 单行 context／quota、真实残差 EOS、完整状态及消费 pending 续算通过；保留本次自然拒绝的分支边界，不冒称观测到自然接受 EOS。GLM [两代表量化随机入口](../evidence/correctness/mtp-glm-quantized-stochastic-entry-2026-10-24.json)Q30／B09 自然随机与消费 pending 后续算现均通过，保留前期所有缺源退出。[Q24 有界 K2](../evidence/correctness/mtp-qwen24-original-stochastic-k2-state-2026-10-24.json)现进一步接通真实递归两草稿与三个 target hidden 的原词表验证，原 GGUF CUDA 旅程通过完整 0／1／2 接受前缀修复、额度／EOS、pending 与 RNG 原子提交及真实残差续算；本轮自然接受为 0，1／2 前缀为明确的状态控制分支。MoE target body 仍逐行执行，不宣称 T3 body 并行或整台 2.4T 模型运行。
[Q27 预运行审查](../evidence/research/mtp-qwen27-run-ready-offline-audit-2026-10-24.json)核对 866 个角色、9 种编码、几何、CPU／GPU 放置与已知预算，修复准备脚本未保留 BOS 元数据及旧 CUDA 映像构造异常时的释放顺序；新消费者仅单 TU 重链。该记录形成时正文未齐，离线结果不代表完整推理；后续完整旅程另见上文的新证据。

## 架构和性能的实施边界

实际接线提炼的职责与执行链见[推理编排设计](../architecture/inference-design.md)；设计约束与已实现范围分别记录。

模型层声明权重、数学、target tap 与完整续算状态，并负责已消费输入、pending／RNG 的原状态发布事务；
通用接受规则只返回合法前缀，请求窗口和缓存目录只处理各自的政策与索引。
Compute／Graph 负责既有计算、页权限、依赖、资源与真实退休。优先收拢现有重复逻辑，
不增加另一套执行器、模型白名单或每步动态 Provider 路由。

功能接线优先；冻结未完成制品的验证基线后，并行提取已验证、行为不变的公共代码。按实际共同点处理以下架构任务：

- 分离模型数学、权重角色和量化制品绑定；同一家族不同量化配置共用构图与执行流程，保留真实 norm／tap／精度差异。
- 模型头产出 hidden／logits，调用方按实际策略生成选词图。避免先生成 greedy 节点、再用随机采样覆盖端口而留下无消费者的节点；本轮 Apple MTP 已暴露并修复这一接缝。
- 冷建状态端口与合法接受前缀的输出映射；共用绑定、收集与一次提交，保留完整快照及页帧所有权。
- 复用有界接受／尾部规则和完整发布约束；Qwen／GLM／DeepSeek 保留各自真实 draft／verify／repair 顺序、状态与 RNG 地址，不新增只转发回调的统一控制器。
- 共用有限请求窗口及 typed 完整检查点目录；接纳依赖原容量与真实退休，命中仍由模型执行新请求 fork 或同请求恢复。
- 完成事实以同步借用交给入口输出；不复制大结果或把观察完成当作设备退休。
- 根据冷编译的逐端口资源事实绑定；新增后端不修改模型数学和提交循环，不让模型猜所有输出都在同一设备域。候选应在昂贵的原生准备前拒绝已知不支持的类型，并保留首个有效原因；本轮 Apple 缺少一个 I32→F32 Cast 曾触发 4096 次回溯，不能把调高上限作为修复。
- 复用执行槽的准备期临时内存，明确配置、layout、permission 与捕获失效；维护原有预算和真实退休。

用 Qwen／GLM／DeepSeek 的已有代表消费者及同家族两个量化配置检查这些边界。
新增模型或配置不应修改共用执行循环；保持热路径静态，不增加分配、kernel 或搬运来换取抽象。

功能阶段先补上述真实使用缺口，随后依据实际接线收敛架构；性能阶段再调查普通公共成本和 MTP 额外成本。
每批完整对照同时记录 prefill、decode、内存／显存和真实多轮综合速度，普通／MTP 分账；
结果与已拒绝候选只在[性能资料](../developer/performance/decode.md)维护。
不把减少复制、kernel 数或单算子收益直接当作整引擎收益，也不为整理架构反复做全库验证。
