# 推理工程范围与顺序

长期开发依据为[两项核心期许](../architecture/inference-principles.md)：跨模型模块化与默认推理性能；每个受支持模型都应尽量达到专用引擎的执行效率，尽量不因架构降低性能。具体方案须有技术依据与相应模型的实际验证。

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
共同步骤驱动及已列资源合同已有有限检查；整体跨模型模块化、接入成本与冷／热职责尚未完成复审。
当前先完成共同执行架构与普通连续流水的真实剩余项，并复审CMake、初始化脚本和工程文档；保留相关正确性、实际费用与明显退化检查。独立性能候选后置，最终性能目标仍是默认完整词表的普通／MTP。
条件后继的未验证实现已完整保存并撤出产品工作区，后续性能阶段再使用，不计为已交付能力。

## 按两项核心期许调整当前工作

2026-10-30用户明确要求将跨模型架构与默认性能实际用于开发方向。此前把共同步骤驱动的有限完成记为整体架构100%，范围判得过窄，现予纠正；已通过的输出、状态、RNG及退休检查继续保留，不重做已完成机制。

当前按以下顺序处理真实依赖，先分清已有接口可直接复用的部分与实际缺项，再实现对应调整：

| 工作 | 当前依据与具体调整 |
|---|---|
| 跨模型需求与最小接入面 | 按稠密／MoE、KV／MLA、GDN／KDA递归状态及不同MTP结构列需求矩阵；用已有Qwen、GLM、Kimi、DeepSeek原组件核接口。新模型应增加自己的模型包、数学／状态和静态适配，不能要求修改Qwen或在共同层增加模型名分支。 |
| 模型冷构图与独立编译面 | `engine/models/qwen35/main.cc`混合GraphAssembly／Recipe、冷程序构建、热Model及请求桥接，多个消费者直接包含该文件。先沿实际输入／输出／owner依赖隔离冷模型构图，消费原PreparedProgramFamily／ProgramIo等程序事实；减少私有模型修改扩散到热运行和其它消费者的重编。不能只搬文件而保留同样耦合。 |
| 冷Graph／制品装配与完整状态 | 核Qwen GraphAssembly／Recipe与GLM Native／ResidentRun、Kimi Run及test-local ColdProgram的接缝；只提升真实共用机制。用GLM完整StateVersion／journal、Kimi KDA／MLA状态及DeepSeek原组件检查共同事实，模型仍拥有数学、状态布局与合法修复／发布顺序。 |
| 请求／恢复薄适配与冷热费用 | 检查Qwen请求准备、前缀、重建与GLM目录适配需要重复哪些政策；同时核每步形状选取、child查询、候选分配及Source装配中哪些能在冷期准备。保留动态身份、权限、完整状态和真实退休检查，不缓存另一份权限。 |
| 默认性能与候选收敛 | 与上述架构改动相关的检查使用默认完整词表、普通／MTP及相关批次，记录冷准备、热路径成本和受影响编译单元。先核当前已用融合和实际dispatch，再选通用候选；显式32K成绩只列为可选实验，不代签默认decode优势。最终让适用快路径在正常默认构建／用法下生效。 |

第一片从模型冷构图、热运行和请求控制的编译依赖着手，并用现有异构状态组件核对接入事实；不新增大ModelOps、第二执行器或账簿。
当前第一片已将Qwen冷构图辅助的非模板实现隔离到独立cc，并使一个只需冷接口的原生消费者脱离main；这是编译依赖接缝，不是完整家族工厂或跨模型模块化完成。
成熟实践依据来自实际版本的llama.cpp构图／模型／上下文分工及TensorRT-LLM调用／输入准备／runner分工，接口和数值仍按HYLE原合同核验。
冷cache事实现已独立为模型私有cache_plan.h，weights／构图头不再间接解析完整会话状态；原容量公式、布局与数值保持，实际依赖及有限默认消费者检查见[本片记录](../evidence/correctness/inference-cold-cache-facts-2026-10-30.json)。
Qwen／GLM／Kimi／DeepSeek的实际接入面已经核对：有限具体图不需要人为套入Qwen的程序族，IO／资源交接复用也不等于完整引擎接入。
目标主干数学已沿显式冷事实与端口span隔离到target_graph，原Recipe／schema／常量、准备／发布合同及默认有限消费者保持，见[本片记录](../evidence/correctness/inference-target-graph-boundary-2026-10-30.json)。
本片发现H64混合CPU FFN的T1子程序在新旧基线均因Graph内部按值Tensor偏移128B／声明256B而拒绝；已由共同输入接纳沿原Copy／预算／寿命机制按需物化修复，见[本片记录](../evidence/correctness/inference-graph-alias-alignment-2026-10-30.json)。
7项新合同、18项相关回归、原H64及真实2B默认／混合／Session有限对照通过；历史编译／测试失败与合法预算回溯分别保留。新发现继续纳入，不能以较大模型成功或旧清单完成排除小形状。
完整Family冷工厂现已隔离：一次消费已准备冷事实，完成原整图、有限compile、程序与逐child检查，Model保准入、原cache／head、采样身份与成功发布。新调用只在冷准备／原idle接纳链，逐步执行未新增分派或资源权威；cache／head准备顺序、失败诊断合并及冷计时范围变化单列验证。
[本片有限检查](../evidence/correctness/inference-family-factory-boundary-2026-10-30.json)17对／34个原生进程通过；默认四对及真实CPU2的全部非时间汇总一致，完整Session向量逐字节相同，首次预算拒绝与同key重试通过。历史工厂C4457编译失败及两次解析准备事件保留，未把晚期child检查失败、完整默认CMake或独立性能窗口写成已验证。
Phase物理准备已独立，原IO注册／slots／owner保留；Model仍负责政策、可晚挂接的state layout、共享scratch、初始化与真实退休。[本片26个有限原生进程](../evidence/correctness/inference-phase-preparation-boundary-2026-10-30.json)全部通过，包括首次真实allocation预算拒绝与重建、早／晚布局、warm原Invocation、默认四对、完整Session向量和9个diagnostic probe。230旧输入全复用，仅新增一个phase对象；无SDK／NVCC／下载或新性能窗。其余冷程序、请求恢复和稳态事实接线继续复审，不机械固定偶然host emit顺序，已完成共同IO／退休机制不重复。
跨模型caller输出池现共享显式份数／alignment下的checked容量事实，保模型保留政策、域、原LeasePool与owner；实际LeasePool用例复现非对齐V65两块旧容量拒绝并验证修复，未运行真实V65模型。GLM NextN暴露的常量View对齐缺口也已由Graph冷canonical表示修复，保原borrow／发布别名及预算／真实退休，无逐token Copy。
[本片有限检查](../evidence/correctness/inference-output-pool-and-constant-view-2026-10-30.json)的11个输出池相关用例、42项Graph／缓存检查、新旧ordinary GLM、默认完整词表四对及完整Session通过；102个实际进程中的9次历史原生失败及两次scratch处理错误保留。新旧ordinary的24项属性exact，Session两臂各378份状态／150,761,472B逐字节一致。仅9个私有布局受影响SDK对象重编、222复用；无NVCC／下载或新完整性能窗。
Graph可选原地复用已按完整根offset与required组模同余修复，保合法alias与原独立输出，无额外Copy／运行时分派。
[本片检查](../evidence/correctness/inference-optional-alias-alignment-2026-10-30.json)以同新TU验证旧b8两项真实失败、新图78项通过／零跳过；默认四对非时间字段、完整Session状态一致，GLM Pro与tiny CPU batch-state通过。仅1Graph与2测试TU重编，230旧输入复用；最初NativeNeg/F32能力误选和诊断语法失败保留，单配耗时不冒称稳定性能收益。
辅助draft／prefix／frontier冷工厂与同次warm Phase事实复用已完成；Model保原准入、guard、初始化／发布与退休，不新增热owner或回调。
[本片检查](../evidence/correctness/inference-auxiliary-warm-boundary-2026-10-31.json)19进程全通过（4CXX、3link、12native），231旧输入全复用、仅新增aux对象；三真实tracked预算拒绝／重试和warm原身份通过。默认四项对保留M2结果非时间字段一致，Session各378份／150,761,472B完整字节一致；仅两个single有近窗粗退化观察，不宣称新完整性能窗。
联合复审按五项正面要求核入口、四类消费者与成熟源码，当前未确认须先新增的共同实现缺项。正常Release shared GPU产品已[构建并有限检查](../evidence/correctness/inference-shared-graph-batch-2026-10-31.json)：修复三处friend首次声明缺导出宏导致的MSVC冲突，同cache增量成功，CPU ordinary／CUDA K1各4输出和三函数外部导入通过。公开GPU route、非维护factor、离线依赖仍保；旧手工缓存闭包不代签该产品。
[默认full-V32进程对照](../evidence/performance/inference-default-product-2026-10-31.json)全部exit0／同臂非计时稳定；普通单／双轮decode落后约2.83%／0.73%，MTP仍落后约10.51%／4.97%。actual rows、两次后继丢弃、8ready显存未知和2次有限模块枚举缺口保留；不签稳定提速或相同续算工作。该基线保留为后续依据；先补普通流水的上传／读回和跨turn捕获复用、继续跨模型接入验收，独立比较后置，不缩小完整goal。
普通dense确定输入的首组合并已[实现并有限检查](../evidence/correctness/inference-input-packet-2026-10-31.json)：原Phase／slot持有对齐packet和精确端口视图，token／position一次同步写；保原carrier／Source退休与冷预算回落，无新用户开关。正常shared增量、7项CPU／CUDA和真实2B两轮旧／新检查通过；可导出trace逐字节相同，未导出的完整KV／carry／RNG blob不作证明。默认R1稳态relay原本只传position一次，不能据本片签稳态提速；其它metadata／读回和真实跨turn捕获复用继续。
较强设备用于底座成熟后的更广整模型验证，当前工作不以租设备、完整大模型或通用loader／server为前置。
按10/31用户顺序调整，已启动的一次ordinary跟踪已自然结束，完整非计时报告一致；首轮离线提取失败从原SQLite补齐，无模型重跑。其余三臂未启动，独立优化暂缓。工程入口复审保留现有产品／后端选择：修复空内部factor缓存检测，68项脚本检查及真实离线util初始化／重复初始化／构建安装／独立消费者通过；入门文档合并重复说明。正在做和估计进度只在[看板](../developer/inference-progress.md)维护。
后续实际模块构建又补齐测试归属与私有头路径，并修复重复配置删除未变SIMD输出导致的重编；
配置稳定性、离线入口与文档的范围和失败历史统一见[工程入口检查](../evidence/correctness/inference-engineering-entry-2026-10-31.json)。

## 当前 goal 范围：全部 review 应对

**状态：active。完整目标于2026-10-30按用户的两项核心期许更新。** 本节是后续接续使用的完整goal表述；当前子任务、代表模型或可选配置的完成不能替代它。

1. **完成底层功能与跨模型模块化架构。** 继续处理全部review中的普通／MTP、多会话、多请求／多任务、缓存、执行、工程和测量问题；补齐当前硬件可实现、可检查的相关底层能力。结合不同模型需求及SGLang、llama.cpp、vLLM、TensorRT-LLM、DeepInfer等成熟引擎的实际源码，形成模型无关且职责明确的模块、形状和状态合同，减少跨模块修改扩散与新增模型的重复开发。2B Qwen只是当前完整验证消费者，不能定义共同形状或完成边界。
2. **以冷期准备消化架构复杂度，保持推理性能。** 对每个受支持模型，都尽量达到专门为它准备的引擎的执行效率，尽量不因架构降低性能；保留其合法融合、布局、状态更新与执行快路径，不为统一接口增加不需要的阶段、搬运、同步或计算。复用Compute／Graph原执行、资源、Invocation和退休机制；清楚分开冷计划／加载／构图、热执行、请求政策、模型数学和完整状态发布。尽量在推理前确定实际程序、形状族、端口、资源布局与合法能力，热路径静态直接消费准备事实，不因新抽象增加运行成本。具体方案核实际模型需求、成熟源码与适用条件，以相应完整消费者验证，不能从Qwen单例推断所有模型。保留Snapshot、RNG、有序提交、背压、首错、权限和真实退休；继续覆盖会话换出换入、CPU/GPU overlap、设备token接力、稳定索引GDN／conv状态及SDK／Provider／CPU路由／依赖边界。
3. **让默认构建和正常默认用法获得有竞争力的性能。** 纯GPU decode是首要指标，查明并改善普通及默认完整词表MTP与llama.cpp的差距，复用Hyle既有加速并寻找通用优化；CUDA重点参考TensorRT-LLM。调研开关和候选最终收敛为适用时自动选择的快路径及必要回落，不让用户靠专用参数挑选实现。显式32K等特定配置只算实验／可选结果，不代签通用默认目标；显存避免以接近翻倍占用换速度，小幅差异结合实际收益权衡。
4. **以完整消费者与真实证据交付。** 架构调整做相关正确性和明显退化检查，性能改动用当前普通／MTP、单轮／多轮及相关批次完整对照，保留prefill、decode、启动、综合速度、内存／显存、实际版本、输出／算量差异与测量波动。补当前能做的组件与组合，不以完整大模型受限排除能力，也不把有限组件通过扩大为任意模型／拓扑证明；底座成熟后再用更强设备扩大整模型验证。

既有共同步骤驱动及已列功能检查继续保留，此前4个源码改动已验证提交；不重开已完成MTP／多模态功能穷尽。
当前按上一节先完成共同架构的模块接入／编译面与冷热职责、普通连续流水的真实结构缺项和工程入口复审，再集中独立默认性能优化；相关有限检查持续进行。
通用loader、tokenizer、server不是本阶段底座工作的前置。维护看板和证据，Windows产品源码为准，直接main提交推送，不下载模型；失败、未运行与硬件受限分别记录。

用户已要求将这几轮 review 的工程、功能、架构、GPU 优化和测量问题全部纳入同一 goal。
以下是工作范围；逐项百分比、正在做的事项及最新实测统一维护在[推理看板](../developer/inference-progress.md)。
不能只完成注意力重录或一个代表模型就结束，也不把 review 的收益估算当作已实现结果。

共同架构的完成条件还包括[核心期许](../architecture/inference-principles.md)中的跨模型需求、模块依赖／独立编译、低成本接入与冷／热职责证据。
请求与资源机制须可被结构不同的模型适配消费，完整步骤编排与数学／状态布局分清，
普通及MTP共用原提交、观察、权限和退休机制；模型差异及实际部署能力自然表达，热路径保持静态直接调用。
仅抽出载体、IO或请求原语不算完整执行核心完成。当前完整引擎仍只有Qwen，按实际共同机制检查其它模型组件，
不凭接口外观声明任意模型可接入，也不以缺少大显卡将Qwen的形状升级为通用合同。
多卡／多机本轮检查设计边界与单设备假设，实际分片、通信及跨端点执行仍是未来路线，未计入当前功能完成。

本轮架构交付按以下边界分解，逐项百分比统一在看板维护；下表说明职责，不表示每行仍有未实现机制：

| 交付 | 职责与边界 |
|---|---|
| 完整步骤与模型适配 | 冷选择、程序IO、资源交接、连续前后继及同步目标驱动已共享，结果／请求见证已分离；请求桥接与GLM K1候选步骤已迁移，失败及阶段边界联合有限复审完成；数学、完整状态、Head发布和合法阶段约束留在模型 |
| 稳态与后继 | 普通／relay保原Source／Store／Append权限，连续pending及前后继控制已接共同驱动，提交见证及模型阶段联合复审已完成；随后优化稳态成本与连续性 |
| MTP阶段事实与装配 | 接受、停止／额度、提交见证与退休已共享；融合及递归的提案、验证和续算数学／RNG差异保留在适配层 |
| 多模型与拓扑边界 | 原模型组件检查适配边界；Graph多端点表达、Backend装配与模型放置分开，状态归实际存储域；不代签多卡实现 |
| 输出与性能检验 | 共同输出事实已接线；后续用当前4070上的decode、MTP和含预热／换轮的综合速度检验热路径，完整大模型与多卡验证另记 |

Qwen／Kimi的完整资源阶段交接已接入原ExecutionEpisode；Head／ledger／RNG发布、私有修复及reset政策仍由原模型负责。
GLM已有模型步骤封装及实际额度接线：原场景和RequestLoop调用真实普通／K1计算与未发布候选形成，不再依赖固定base／随机身份的计算闭包。
随后[实际计算／观察分离](../evidence/correctness/inference-glm-math-boundary-2026-10-30.json)让execute_step／execute_target及GlmStepRuntime只形成实际结果，原reference／EXPECT在scenario显式观察并立即释放临时lease／version。
两个原CUDA场景通过，107项非计时属性与拆分前一致，未改原提交数或容量上限；该片当时尚未接入共同步骤驱动；后续接入及失败检查见下文，仍不代表完整GLM。
已选K1工作遇输出额度1，仍保原提案／验证及三draw合同，只限制可发布前缀；不能触发另一条target-only单draw政策。
GLM先发布已接受完整状态、再维护journal的时序区别于Qwen，已提交结果和维护后current要分别表达，晚失败只能补观察，不能重跑原数学。
普通和relay已共用同一准备入口，Source／Store绑定仍遵守真实Append及packed元数据顺序，不再增设一层总prepare或recipe回调。
私有draft阶段已接入原成功交接：原等待计数、输出观察、Append／Store取回、一次动态IO撤回及页发布顺序保持。
[四个相关CUDA消费者](../evidence/correctness/inference-private-handoff-2026-10-30.json)覆盖递归K2、低概率截断、私有失败重试及动态probability／next_counts连续执行；三份完整非计时报告保持，无性能实验。
普通／relay已共用Qwen内的同一prepare_evaluations，Source／Store／Append及replay权限按原实际执行核验；这消除了模型内部两条准备入口的重复，尚不证明实际共同步骤驱动已从Qwen抽离。不能仅包一层总prepare／recipe调用冒充完成。
[零张量阶段结果束](../evidence/correctness/inference-effect-views-2026-10-30.json)已补齐原SubmissionSource的纯页／提交结果接线；11项测试及真实递归MTP通过，原partial捕获与错误恢复保留，没有性能实验。
入口／输出及跨模型组件的资源机制已有有限复审；完整步骤驱动仍须由结构不同的原组件实际消费。稳定引擎C++ API或第二完整模型的产品交付不是此项前置，对应未交付和验证边界仍明确保留。
后续多owner检查发现原Graph冷Store adapter仅按target.default_storage装配，同Context其它真实存储域没有候选。
[显式节点域与银行装配](../evidence/correctness/inference-node-storage-2026-10-30.json)现已修复：GraphBindings传递实际节点域，
Store／Append按原端点可访问域冷装配，源合同与银行选择分离，保留原Copy、owner及失败恢复。
17项测试、真实递归K2及CPU＋CUDA普通短旅程通过；33个受影响SDK对象与三个消费者按新公开结构布局同头重编，未混旧ABI。
此前有限复审把共同原语与局部资源阶段交接误记为完整执行架构完成，2026-10-30撤销该结论。
随后[共同步骤驱动](../evidence/correctness/inference-common-step-driver-2026-10-30.json)已将Qwen连续普通／K1的实际准备／提交／观察、后继去留和异常恢复归入drive_steps，GLM真实ordinary续算也消费其同步分支。
模型适配保留数学、实际Source边界、状态与Head发布；同一原Model／Graph继续负责执行和退休。8项有限检查及同期明显退化检查通过，未报告提速。
随后[同步与递归目标步骤](../evidence/correctness/inference-synchronous-step-driver-2026-10-30.json)已将Qwen同步、packed prefill和递归验证目标步骤接入同一驱动，保留原数学、发布、取消与真实退休。
7项有限检查通过，真实K1全部非计时字段保持；没有独立性能实验。
随后[结果／提交见证分离](../evidence/correctness/inference-publication-boundary-2026-10-30.json)将账簿预检、决策记录投影、replay核对、统一计时及精确失败清理移到调用期请求对象；Step不再持这些借用，模型只消费原通用票据做Head发布。
8项有限检查通过，真实K1全部非计时字段、GLM的54项属性保持；本片没有独立性能实验。采样／历史输入桥接与GLM K1候选驱动随后继续迁移；第二完整引擎模型仍未验证，不能把本片当成完整架构完成。
随后[请求桥接与K1步骤](../evidence/correctness/inference-request-bridges-2026-10-30.json)已将两个请求桥接移出Model，GLM实际未发布候选消费共同驱动的同步分支。
保留原身份、历史条件恢复、数学、完整状态、Head发布、journal维护与RNG。
随后[候选首错及架构联合复审](../evidence/correctness/inference-glm-candidate-failure-2026-10-30.json)补齐真实未发布候选的观察首错、实际退休后清理、Head／ledger／RNG不变及同提案完整状态重试检查。
原48项请求属性保持，3项实际工作增加、2项范围说明更新，新增6项核对；没有扩大资源上限。联合复审按共同推进、跨pump身份／额度、结果见证与MTP阶段装配逐项核销原迁移中估计，该片未发现仍须新增的共同步骤控制算法；它只覆盖已列驱动，不证明整体跨模型模块化、接入成本和冷／热职责完成。
这不代签第二完整模型、GLM跨pump、多卡或全平台；原生已退休后注入观察首错不外推在途设备故障。完整goal继续按本页顶部的广义架构调整推进，并保留默认性能差距、GPU热点及完整消费者比较。

后续[预热候选与A/A](../evidence/performance/inference-warm-boundary-aa-2026-10-30.json)未证明跨cohort保提示有净收益，
并实际复现已暖R2转入新R1的未准备Queue拒绝；候选撤回，原策略及新完整状态回归保留。
单轮schedule／steps的普通、MTP各四对进程全部非计时报告一致；双轮额外形状／workspace的分配假设仍未排除。
[代码域词表](../evidence/research/inference-code-vocabulary-map-2026-10-30.json)已由本地原tokenizer和独立参考源码得到32K映射，
未使用测试prompt／输出，零下载；它不是接受率证据。词表接线尚未编译，已隔离保存并撤出产品源码；
冷导入、原ID映射、真实接受率及净收益检查均待共同执行架构收敛后继续。随机q／p／拒绝残差须保原合同。
随后[显式草稿词表](../evidence/performance/inference-draft-vocabulary-2026-10-30.json)已按当前架构实现：显式原source-bound ID文件、一次冷导入、原ID映射及预算寿命；原target／embedding保留。
CPU／CUDA原行检查、K1／双轮／K2真实消费者和23个共同前缀导出张量检查通过；仅支持无modifier的greedy／零阈值稠密head，默认全词表与随机路径保持。
完整32进程窗采用显式32K，不代替默认全词表MTP差距；同输入同64输出的K1／K2小样本复测中K2有正向信号，实际工作差异保留，未改默认K1。长输入、默认路径和其它GPU热点继续。
条件图仍未接入产品，Executed／Skipped必须先于任何结果发布表达。

以下是已列共同步骤驱动片的迁移与检查定义，保留历史实施依据；整体架构的当前工作以上述核心期许调整为准：

1. **落实共同步骤驱动。** 将前驱提交、可选后继准备／提交、前驱完成、后继保留／撤回及跨pump恢复的共同顺序归入`engine/inference`；同步路径也消费同一阶段与结果事实。继续借原episode、Invocation和退休事实，不新增执行owner。
2. **缩小静态模型适配。** 模型提供实际阶段／形状、输入与状态绑定、合法Source边界、接受／修复及Head发布；共同层处理推进、pending身份及失败收尾。不同家族的状态和数学阶段自然保留，不强制Qwen两步、K1或单KV银行，也不把整个执行循环塞进一个回调。模型结果与请求ledger的借用边界一并分清，继续复用原提交见证，不另存一份状态或账簿。
3. **接入结构不同的现有组件。** Qwen真实普通／MTP路径和至少一个已有异构状态组件消费同一实际驱动；检查停止、额度、取消、晚观察及真实退休。只检查实际可运行范围，不下载整模型、不等待更大硬件，也不把仅调用相同请求窗口算成完整驱动复用。
4. **有限检查后再恢复性能。** 核对原输出／状态／RNG和资源寿命，确认没有明显退化；再继续词表、投影融合、带宽和完整跨引擎性能对照。架构变化导致的正确性或明显退化可当场处理，独立吞吐优化后置。

多模型形状同时参考本地成熟引擎的实际实现：vLLM `276fbcff2717`的`v1/engine/core.py`、`v1/core/sched/scheduler.py`和`model_executor/models/qwen3_5.py`，
TensorRT-LLM `46d9966b54d9`的`_torch/pyexecutor/engine/model_call.py`、`py_executor.py`及`model_engine.py`。
重点借鉴模型数学／状态事实、共同调度、原资源和结果观察之间的边界，不引入其大型动态ModelEngine或第二资源管理器。
版本来自本地checkout，所引文件内容已核对各自HEAD，未声明整个工作树干净或最新上游；参考职责并不代表已复现其性能或全部失败语义。

2026-10-29已纠正将共同原语完成误算成完整编排完成的估计，功能与共同形状联合审查。
实际已通过的检查保留；百分比统一维护在看板，不把新的任务拆分计成新增实现。

| 任务 | 包含的 review 内容 |
|---|---|
| 功能与工程边界 | 冷形状接纳、长度分桶与预热；有效长度／varlen prefill及递归状态正确性；组合矩阵与分类错误；单成员取消、完整会话主机换出换入；CPU路由中的模型形状常量、Provider命名、SDK detail边界和ATen依赖选择；初始化与公开配置、简明执行合同、文档、Apache-2.0声明及依赖许可交付 |
| 共同执行架构 | 将真实执行编排、形状族与载体管理从Qwen Model抽离；静态模型适配；基础提交／取消／credit和token／结束／分类错误输出边界；统一稳态准备与权限表达，审查在途Source路径和普通／后继双准备路径；共同请求与缓存不等于完整执行核心；统一可共用的K1／K>1 MTP流程 |
| 普通连续流水 | wave跨pump在途、先提交后继再观察前驱；有序发布、背压、EOS、取消、失败与真实退休；常数成本的稳态准备、上传／读回合并；固定拓扑attention和独立槽；拒绝后恢复、跨turn预热提示及冷准备捕获；扩展适用配置与平台 |
| MTP性能 | 首个capture失配端口及完整绑定轮换；hidden固定池索引读写；固定验证形状和catch-up有效行；设备接受／后继与连续流水；不按测试prompt选择的草稿词表子集、接受率与完整输出检查；草稿变便宜后重测K2／多草稿 |
| GPU热点 | GDN a/b小投影并入同输入投影或专用GEMV；qkv/z/a/b、q/k/v、gate/up的整模型融合候选；模型内GEMM实际DRAM带宽；norm、残差、concat等小kernel融合。已关闭候选须有适用边界与重新评价条件，不能用旧叶子结果排除不同工况 |
| 测量与回归 | 当前版本R1／B2的off／auto及MTP跟踪、capture数量和主机／GPU分解；schedule对steps的交错A/A；原始startup、ready、prefill、decode、每turn与综合端点；显存／内存；长输入、多轮、多会话、跨平台及CPU＋GPU／MoE适用基线。重叠step时长不可相加代替墙时，旧Nsight不代签当前版本 |

2026-10-29前批已盘点的单机功能项已实现并有限检查；继续联合审查不同模型、状态及执行形式的组合，
共同接口不能表达或不能保留原请求语义的缺项仍补回功能工作。Qwen私有提交ledger与GLM请求组件间的共同见证已接通：
Head成功后立即无失败地记录该成员结果，后续报告／catch-up失败按原ordinal补观察，不能重新采样或重跑推理。
记录只保留原主机事实，不保活GPU历史版本；[五个有限检查](../evidence/correctness/inference-commit-observation-2026-10-29.json)涵盖普通／MTP、晚失败及原退休断言，
不能据此声明完整多模型执行编排已抽离，或其它组合已全部检查。
随后[共同冷资源与固定准备](../evidence/correctness/inference-cold-bindings-2026-10-29.json)让Qwen与真实GLM短历史组件共享常量、实际存储域和放置组装，
从实际状态池布局冷存writer字节，原物理槽复用batch准备数组并立即释放暂借layout。六项有限检查通过，数学、原lease及Graph权限不变；
普通CPU内存直接物化到CUDA的测试假设被原Graph拒绝，改用原qualified pinned-host资源，没有放宽复制资格或新增回退。
逐成员完整状态原子发布已有能力，不要求不同用户的所有Head额外组成跨用户原子事务；必要的模型差异分别表达。
多成员MTP撤销已沿普通路径统一为原成员发布资格：完整取回原Store／Append，跳过撤销成员的接受选择、repair与发布，
幸存者使用原批次结果，不重跑；本轮proposal耗时只写实际新提交，不能改撤销成员上一条记录。
单成员、全部停止、提交前停止和owner明确abort仍按原整episode退休；
[七项有限消费者](../evidence/correctness/inference-mtp-member-withdrawal-2026-10-29.json)通过prefill、混合lag K1、history／动态条件K2及E4状态，
并保留普通CUDA流水与真实模型MTP回归。三次history测试接线失败如实保留；没有新的吞吐或全平台结论。
随后[共同尾收尾与首错恢复](../evidence/correctness/inference-append-tail-return-2026-10-29.json)合并普通target、融合draft与私有NextN的原尾归还，
去掉R1／R2三处固定元数据堆分配，并保留原失败Status、participant、退休事实与逐成员观察重试；八个有限CPU／CUDA进程通过。
随后[GLM共同请求驱动](../evidence/correctness/inference-glm-request-loop-2026-10-29.json)已用既有RequestLoop／wave替换手动FIFO、接纳和turn切换；
自然K1、缓存新请求采样及两次普通续算实际在wave内执行，原数学片段由旧路径及wave复用。
一次原权重CUDA组件旅程通过，晚观察失败后仅补观察，37项原输出／RNG／提交事实不变；不等同第二个完整引擎模型或性能结论。
本次[多模型提交与错误接入](../evidence/correctness/inference-multimodel-observation-2026-10-29.json)已接DeepSeek Pro／正式Flash原角色的真实多token提交与晚观察；
发布前使用原RequestProgress完整预检，记录不保活历史状态。DS测试消费者删除重复Graph生命周期，复用GLM／Kimi原共同路径，保留Status及独立清理／退休事实。
四个当前有限进程、六项测试通过，含真实CPU资源拒绝、Gather越界、保留结果时清理拒绝及GLM CUDA回归；一项新夹具资源声明遗漏已修并保留。
随后[压缩侧车与完整页准备](../evidence/correctness/inference-sidecar-pages-2026-10-29.json)接通Q8／BF16侧车的原提交观察、晚观察恢复及实际pending续算，
并把Qwen普通／验证／后继target、批量draft及私有NextN的完整分页准备合为一份静态算法。晚获取失败保留原episode已有权限，不另建执行或清理机制。
七个当前有限进程通过，含12项真实CPU页权限检查、CPU history K2、CUDA混合lag、真实2B MTP及普通连续流水16臂；
一次旧夹具目录冲突拒绝已保留，改用本批两份元数据文件复用原权重。无SDK重编、下载或吞吐刷新；完整执行编排仍未抽完。
旧Flash四round记录不作为当前五round的同版本基线，侧车组件亦不外推完整大模型生成。
随后[共同分页布局](../evidence/correctness/inference-paged-layout-2026-10-29.json)将target、draft、私有NextN及部分接受后的元数据组装接入同一冷预留缓冲，
保留原Schema版本、R1路径、ahead及Source权限。四项布局测试及七个CPU／CUDA模型进程通过；多行JSON日志仅修收集解析，没有重跑或放宽数值门。
联合复审的[请求恢复整改](../evidence/correctness/inference-request-recovery-2026-10-29.json)修复完整前缀命中后同wave失败重试的RNG预算重复计数，
也修复普通StateOnly重建未选到packed容量时仍声明packed的错误。先后两次真实产品失败保留，同一原采样预算旅程修复后通过。
自然N1双请求另完成部分前缀、StateOnly分块、自动HostImage换出换入及ACK背压的联合旅程；真实页耗尽拒绝保留原image／key／ledger，释放后继续，
6次target提交、无ledger重算，最终版本及image均为0。仅编两个小消费者并复用SDK，没有重编全部请求测试入口或刷新吞吐。
Kimi现有KDA／MLA消费者不产生生成token，不为共用请求接口虚构生成阶段。
本轮工程入口、完整依赖装配、文档导航及许可交付整改已完成，结果见[工程检查记录](../evidence/licensing/dependency-licenses-2026-10-29.json)。
当前收敛完整执行编排、普通／MTP共同机制及SDK接线，随后集中连续后继、GPU融合及带宽等性能工作。
本次[分页序列共同生命周期](../evidence/correctness/inference-paged-sequence-2026-10-29.json)已将原库存、任意银行的Frame、冻结／分叉及提交迁入共同层，
替换Qwen普通、图内draft及私有NextN中的重复资源算法，无兼容别名。模型保留数学、长度／frontier、GDN／conv／hidden及NextN规则，原ExecutionEpisode继续承担提交和退休。
单银行真实Graph与两银行晚拒绝共2项、14项会话／前缀测试及7个有限模型旅程通过；受session_state头变化影响的weights编译单元刷新，226个SDK输入复用。
两处新消费者编译接线问题已修，一处期望测试数错误仅重核原XML，没有重跑或放宽原断言；未刷新吞吐或运行新平台。
随后[分页追加组共同编排](../evidence/correctness/inference-paged-append-cohort-2026-10-29.json)已贯通target、批量draft及私有NextN的页扩展、布局、原Source构造与成功收尾，
保原逐成员候选／取页顺序、R1快路、pending来源及接受／repair／ledger前后的归尾时序。Source缓冲冷预留，返回即清读pin，删除两处每步临时数组。
4项组件检查含5次真实Graph提交，7条CPU／CUDA模型旅程完整报告与上一片一致；零银行共享政策页的物理尾唯一性误拒绝已真实复现并修复。
五个消费者刷新，227个SDK／引擎输入复用；编译接线失败与修复前真实失败保留，没有吞吐刷新。完整执行引擎仍未完成。
GLM Flash／Pro、DeepSeek正式Flash／Pro与Kimi的原组件程序已接共同ProgramIo：输入计划、驻留银行／map映射及输出事实冷准备，
继续使用原CompiledGraph、Invocation、池、提交和退休；没有为同步消费者伪造Carrier Scope或加一次多余解绑扫描。
[冷输入schema](../evidence/correctness/inference-input-schema-2026-10-30.json)进一步统一输入事实：
按选定程序的完整ValueId取得原external根描述，删除组件平行Port库存及Qwen三处热形状重建。
实际typed metadata仍交原Graph检查；20项测试及12条有限消费者通过，未刷新吞吐，也不代签完整阶段编排完成。
[阶段参数与事务归属](../evidence/correctness/inference-stage-resources-2026-10-30.json)进一步删除GLM／DeepSeek四处热参数数组，
按原冷阶段输入ID借用live表；保留真实专家选择／上传、全部观察输出及原提交退休。旧SlotTransactions串取另一Invocation的事务已真实复现，
现按原state身份拒绝，移动及合法重绑定仍成立。Source与分页追加复用既有容器，减少小型子分配，但不保证host元数据字节下降。
61项测试与12条有限消费者通过，11份完整非计时报告保持；同窗ABBA初筛121.87→122.67tok/s，未证明提速。
这一片不完成完整执行架构；剩余模型状态／发布差异不靠包装循环消除，先继续处理真实完整接线。
随后[实际程序域读回](../evidence/correctness/inference-readback-domains-2026-10-30.json)修复冷readback按Backend预设域编号取资源的问题，
具体程序传入原存储域，非caller无需caller域表，运行时按原输出资源读取。独立编译旧helper已复现两项原Graph绑定拒绝。
12项CPU／CUDA检查及12条有限消费者通过，11份完整非计时报告保持；227个SDK／引擎对象复用，无SDK重编、下载或性能实验。
Vulkan原测试仅迁移未运行；此片只完成实际域边界整改，不代签完整共同执行编排。
随后[完整阶段交接](../evidence/correctness/inference-stage-handoff-2026-10-30.json)将Qwen／Kimi成功等待、输出观察、原事务取回和撤依赖收敛到同一episode算法，
保模型状态／发布差异及原Status、异常、退休机制。CUDA三阶段Source链已实际复现旧顺序拒绝，再验证自身导出先封闭、incoming后撤回；原Status拒绝和非std异常后仍可恢复。
7项Kimi组件测试和12条有限消费者通过，9份完整非计时报告保持，2份carrier报告增加明确检查及CUDA的3次原Graph提交。
Qwen两段模型动作和其余main逻辑均作原文核对；仅刷新4个消费者，227个SDK／引擎输入复用，无SDK重编、下载或性能实验。
这只完成一个完整资源阶段，不代签完整模型适配；随后继续补GLM真实步骤及credit接线。
本次[GLM独立模型步骤](../evidence/correctness/inference-glm-step-2026-10-30.json)分离真实NextN／target、proposal、验证、接受前缀及完整StateDraft；
普通场景与RequestLoop共用静态借用runtime，原取消、数值oracle、CAS／提交记录与发布后journal维护留在外层，没有新owner或执行器。
实际grant进入前缀选择，无效credit和不一致RNG在验证图之前拒绝；greedy真实匹配proposal62398在grant1下只消费一行、保持pending，没有额外图。
两个原权重CUDA消费者通过，49项旧请求事实逐项保持；原ordinary full-V检查观察同一次policy输出，未减少数值门或重跑图。
只刷新一个组件TU、复用227个SDK／引擎对象，无SDK重编、下载或性能实验。CPU／量化模板有编译检查，本片未原生运行；随机自然接受true、异步GLM与整模型仍未据此证明。
真实DeepSeek首次压缩检查还发现CUDA索引分数的归约漏填stride；修改前后消费者在同SDK上均失败，独立小回归在旧对象上准确失败。
已补物理[key,token,head]步长，4项索引检查、真实首次压缩／窗口环绕及GLM回归通过；
[本批有限检查](../evidence/correctness/inference-program-io-consumers-2026-10-30.json)共18项测试及2个载体进程通过，226个SDK／引擎输入复用、仅刷新1个provider CXX对象，无NVCC或下载。
随后[连续请求控制](../evidence/correctness/inference-continuation-control-2026-10-30.json)已收拢跨pump的有序身份、额度数值边界及整组后继保留决策，
复用原RequestLoop／grant及ledger，不增加每步分配、重复额度验证或第二个执行owner。实际Source、状态边界、MTP数学、slot轮转及ledger绑定仍由模型承担。
27组主机合同与5条CPU／CUDA消费者通过；普通连续16臂、mixed-lag及原权重40输出MTP保留完整状态／RNG／账簿。
新测试采用独立host依赖，原error-only请求循环测试不被迫链接container；最小链接还修正了原账簿测试漏声明的Compute依赖。
该片不证明GLM异步路径或完整执行核心完成。
随后[冷程序完整发布](../evidence/correctness/inference-cold-family-publication-2026-10-30.json)修复Model提前插空槽及3处调用方失败清理，
有限／已接纳exact形状共用family与具体child选择，保留完整前缀接纳、diagnostic／packing隔离及原ColdShapePlan退休／回滚。
5个CPU／CUDA进程通过，含真实预算拒绝后重试、原Invocation签名、普通连续16臂及原权重40输出MTP；后两份完整报告与上一片一致。
3个消费者刷新、227个SDK／引擎输入复用，无SDK重编、下载或新吞吐结论。多宽度冷调用并非整调用事务，未强制触发不支持packed路由的返回分支。
随后[发布失败收尾](../evidence/correctness/inference-publication-retirement-2026-10-30.json)真实复现普通C++异常后未提交记录残留，
将typed／未知异常、流水外层及失败退出的退休门对齐，只清本次精确current slot；调用返回前撤回裸ledger借用，已提交peer与晚观察保留。
7个当前CPU／CUDA进程通过，含9组账簿、实际R2拒绝后重试、MTP私有修复及原权重40输出；普通流水与自然MTP完整报告和上一片一致。
原失败、一次复跑目录拒绝及被后续review替代的编译记录均保留；未注入持续私有native故障，无SDK重编或新吞吐结论。
随后[预检与原候选发布配对](../evidence/correctness/inference-paired-publication-2026-10-30.json)消除普通／MTP发布时重拼target／draft编辑清单，私有NextN也消费同一预检结果。
5项分页组件检查及5条CPU／CUDA模型旅程通过，含真实第二bank拒绝后原候选重试、K3状态、私有修复及原权重40输出；普通流水和自然MTP完整报告与前片一致。
仅刷新受头文件影响的weights编译单元，226个SDK／引擎输入复用；新增测试图的控制依赖遗漏及修正保留，无NVCC、下载或吞吐刷新。
随后[Kimi真实Source链](../evidence/correctness/inference-kimi-source-lifecycle-2026-10-30.json)已采用原Carrier／Episode及完整Run保活，
KDA hidden经真实Source输入MLA，同writer冷捕获后仅替换该端口，零主机等待、零新增录制；原Head、快照、分叉、取消与完整状态检查保留。
实际MLA暴露的I32 Scan同步溢出检查已改为设备验证和条件输出，仍Managed，溢出时整份输出不变；原无排序Scatter补固定捕获绑定资格。
17项有限组件检查及普通连续16臂、原权重40输出MTP通过，后两份完整报告与前片一致；集成能力拒绝、旧runtime断言及新测试表达式编译错误均保留。
仅刷新3个provider C++和1个CUDA单元，223个SDK／引擎输入复用，无下载或吞吐刷新；原23角色不等于完整Kimi生成。
共同冷选择、完整编排及性能仍未完成；GLM真实host路由屏障与各家族Head差异继续保留。
随后[逐资源存储归属](../evidence/correctness/inference-view-storage-2026-10-30.json)删除共同Source包装额外强制的单owner，
输出、Store及页描述分别借原owner，提交metadata owner独立；Qwen直接投影原库存，无临时vector、owner复制或第二资源管理器。
同Context／单Queue的两真实存储域输出及晚owner拒绝恢复共2项通过，Kimi原7项及普通／MTP回归通过；后两份完整报告与前片一致。
未单独运行多owner Store／页组合，没有多GPU或吞吐结论；仅刷新4个消费者，227个SDK／引擎输入复用，无SDK重编或下载。
冷选择／载体装配复审确认原共同机制已足够，剩余packing、角色及私有程序复用政策留在模型；K1／递归接受与发布顺序已共享，
不为字段投影新增包装。完整执行编排、稳态准备及性能仍继续收敛。
本轮复审进一步确认：K1融合与K>1递归已共用接受前缀、成员发布资格和原Store／Append回收主干；
剩余是跨模型的完整阶段装配及适用配置的真实设备后继，不能把这两项写成接受／退休算法仍重复，也不将不同数学recipe强行合并。
随后[Source准备复用](../evidence/correctness/inference-source-preparation-2026-10-30.json)删除每bank单页查找的临时分配和后继普通输入库存的逐步扫描／复制；
10项页来源／Kimi检查及3条模型旅程通过，含真实晚解绑拒绝后保owner、原退休恢复及同会话重试。普通流水和自然MTP完整报告与前片一致。
只刷新1个GraphBatch单元，226个SDK／引擎输入复用，无NVCC、下载或吞吐刷新；本轮实际页、Source权限与完整状态门保留。
共同架构同时审查多模型、复杂阶段及未来多端点的依赖边界，避免将当前Qwen／单卡放置固化为通用合同；
本轮不据组件通过声明大模型整机、多卡或跨机实现完成。现有硬件上的decode、MTP和含预热／换轮的综合速度仍是投入价值的关键检验，
架构按实际热点与维护问题调整，不以等待更大硬件或无限整理抽象替代速度改进。
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
现已接通[原生M2双输出](../evidence/performance/inference-paired-m2-2026-10-30.json)，保持独立原权重与dense结果；
有限数值／寿命检查通过，同CLI同步MTP初筛约＋3.37%。[当前默认auto完整窗口](../evidence/performance/inference-current-auto-window-2026-10-30.json)
同时暴露MTP后继的重录与丢弃工作开销；该整改纳入现goal，先查实际capture失配与状态／端口轮换，再减少准备和重录，不能靠诊断off规避。
后续[成功投机丢弃保留载体](../evidence/performance/inference-successful-discard-2026-10-30.json)定位到正常丢弃误走销毁路径，
由共同episode在真实成功退休后回收原事务并保留固定IO和capture；Session停止及真实失败仍销毁。
5组载体／58次提交及普通16组检查通过，64输出完整非时间报告不变；默认auto交错初筛约＋11.31%。
新32进程窗口MTP单／两轮118.02／130.69tok/s，冻结参考135.69／139.78；首次捕获、预热及无效后继仍待优化。
随后[同cohort预热保留](../evidence/performance/inference-continuation-warmth-2026-10-30.json)在实际成功丢弃后只撤失效Source，
保留原载体的预热提示；失败／停止及换组仍重置，每次准备的完整tuple和权限门不变。
单轮交错初筛约＋3.03%，有效后继7→19而实际图／行数不增；真实40输出完整状态／账簿及普通16组通过。
两轮补充只有单样本，本片不更新完整跨引擎差距；跨turn、首次捕获及无效后继计算仍保留在任务中。
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
2026-10-28的MTP生成检查采用当时整episode取消政策：原Step确因成员停止而cancelled，须先由原abort证明实际退休，才撤未提交账簿、释放q／C0并标整cohort Cancelled；
CPU／CUDA当时检查B2 K2提交后stop A正常返回和原生退休：head／ledger／RNG／pending不发布，peer原key继续。当前逐成员撤回另记；未知／未退、native或其它错误仍带原Status和退休事实抛出。
这些检查使用小型合成制品和原generation入口＋RequestWindow，不是新增完整RequestLoop／服务层验证；逐前缀动态语法、其它平台和未知native故障未由此证明，没有trained quality或性能结论。[本轮记录](../evidence/correctness/inference-history-controls-2026-10-28.json)。
E4目标KV＋原BF16 draft的MTP组合已接通：准备器提供两组实际行字节，容量按独立银行计费，
沿原schema／COW／状态及HostImage执行。CPU／CUDA各11次完整状态字节比较、真实B2受控不同接受前缀、
当时整episode取消后幸存请求续算及默认路径回归通过；元数据与独立容量检查通过，当前撤回合同另记。
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
程序安装进一步删除模型重复维护的exact索引与binding子集；共同载体从固定Executable创建Invocation，
核其完整canonical签名，原Family冷选择仍负责原始形状资格。共同槽集合只对合法未安装槽求值初始化配方，
模型不再复述生命周期规则，暖调用不复制预算句柄。[同六个有限用例复查](../evidence/correctness/inference-program-identity-2026-10-29.json)通过，
保持真实静态图／finite child身份和原状态门；不是按签名重新路由，不新增缓存或兼容分支，也没有新速度结论。
共同episode现承接Source撤销与原caller输入释放，成功撤销即清除bound事实；后续解绑失败保留剩余owner，
退休恢复只撤尚存Source，再由原Invocation销毁归还caller绑定，不重新借用已关闭载体。
[四个有限原生用例](../evidence/correctness/inference-episode-release-2026-10-29.json)通过：新增真实解绑拒绝、未发布head／epoch／RNG／ledger、
原会话恢复及完整状态合法重试；普通连续流水16臂、真实MTP后继及CPU控制保持原门。没有新的性能结论。
联合功能／架构复审又确认混合Backend提前要求所有已配置域预算，会误拒绝未使用GPU或staging的合法图；
现删该重复冷预检，由原Graph对实际资源收费。[原拒绝及修复检查](../evidence/correctness/inference-backend-budget-2026-10-29.json)
包含原Backend反例、CPU／CUDA各两次执行、68值、退休归还和缺真实GPU预算的拒绝。
只刷新包含Backend的一个weights源单元，复用其余226项SDK输入；未重跑前片模型检查，未测吞吐。
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
