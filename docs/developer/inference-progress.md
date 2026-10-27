# 推理架构 Review 进度

更新：2026-10-27 08:15（北京时间）。本轮目标：**处理推理架构 Review，补齐功能、收敛架构，再改善性能**。整体约 **92%**；总数与各分支按实际交付和剩余实现工作量一起复估，不表示测试通过率。本批独立状态读凭据和真实capture生产者准入通过12项定向检查；剩余重点是Store／KV完整状态接力、持续重叠执行、迁移遗漏复查及其它工况完整性能对照。当前仍以底层支持为主，HTTP／通用 loader／tokenizer 不是前置。

## 任务树

- Review 事实核实与范围收敛：100%（已完成）
  - 源码／性能／依赖／第二消费者核查：100%（已完成）
  - 因果修正与实施顺序：100%（已完成）
- 首批功能与接线修正：100%（已完成）
  - 空闲新形状预热与真实接纳计时：100%（已完成）
  - 普通合批单成员取消与明确结果：100%（有限 CPU／CUDA 检查通过）
  - 模型 EOS 配置集中与诊断迁移：100%（已完成）
- 模型无关的共同引擎核心：100%（本轮约定共同合同已实现并有限检查）
  - 请求／兼容合批／接纳与缓存编排：100%（本轮约定共同原语已接通，有限检查通过）
  - 冷计划与运行时的共享prefill分片事实：100%（四组主机合同与原B2有限CUDA检查通过）
  - 模型适配、错误类别与输出合同：100%（取消／交付及执行错误请求身份桥接已接线，有限检查通过）
  - SDK／Provider／依赖与诊断组织：100%（公开资源与保活已补；来源身份、CPU偏好及ATen职责已核，capture说明已修）
- 稳态状态协议：100%（本轮单episode工作态与原池状态接线已实现，有限检查通过）
  - 持续工作态与按需导出不可变快照：100%（KV持续工作帧、按需封存及原池状态已接通）
  - KV持续工作帧及无失败事务发布：100%（本批实现完成，Windows／Apple有限功能检查及B2对照完成）
  - 原生 GDN／conv 的版本化双缓冲与索引状态：100%（CPU／CUDA有限功能及真实CUDA融合、Apple／Vulkan普通与K3 tiny入口通过）
  - MTP 稳定状态槽及接受／回滚接线：100%（本批CPU／CUDA异成员接受通过；性能和其它平台边界另列）
- 其它底层功能补齐：100%（本轮已发现功能及形状政策已实现，有限检查通过）
  - 有限尾块、已计划载体准备与异长分组：100%（未知prefix尾长B2冷键与真实长prefill消费者检查通过）
  - 完整会话主机换出／换入：100%（本轮实现完成；普通／MTP／lag1及受控故障有有限检查，未注入真实驱动故障）
  - 状态搬运非阻塞观察／取消：100%（CPU受控合同及CPU／Vulkan／Apple实际关闭后回收通过有限检查）
  - Vulkan非4字节权重片段接入：100%（F3／Tensor／Graph／Vulkan已接通，原tiny普通／MTP／lag1通过有限检查）
  - Vulkan精确半字同步I/O：100%（本批功能完成，三组有限GPU检查通过）
  - CUDA 小通道因果卷积承载：100%（已补，定向数值与原失败消费者通过）
- 重叠执行与完整性能对照：约45%（实现中）
  - 原Graph在途设备token与既有capture接力：100%（单Queue固定I32一次链、checked错误、完整captured生产者与消费者通过有限CUDA检查）
  - 有界槽轮换与完整状态pending权限：约30%（实现中）
    - 共同提交事实与独立块读凭据：100%（本批有限检查通过）
    - Store／KV在途读接线：约10%（设计中）
    - 有界槽轮换与候选预算：约5%（设计中）
  - CPU提前准备与引擎有序发布：约30%（普通多成员整数准备通过有限CUDA检查；有序后继及MTP修复未接）
  - CPU／GPU／混合参考与完整四项对照：约25%（CUDA普通／MTP最新四臂完成，其它工况仍待完成）
- 遗漏复查与资料收敛：约25%（进行中）
  - 文档恢复、事实校正与阅读入口：100%（本轮修正完成）
  - 剩余功能与迁移遗漏复查：约10%（进行中）

## 正在做

- Store在途读接线：约10%（所属：重叠执行；将独立块票据接原Store目的槽、受检投影与共同终结凭据，保持take／publication与真实退休；KV另核append epoch／前缀与fresh page）。
- 完整模型接力设计：约20%（所属：重叠执行；已确认token裸视图绑定、I32形状和Hybrid覆盖三处缺口，按原capture区间与Prepared gap收敛，不扩大Provider静态承诺）。

剩余大项包括完整状态接力、重叠执行、迁移遗漏复查及其它工况完整性能对照，不能以本批CUDA通过代签这些工作。真实异长请求已有按兼容T分组执行；不以全部Provider提供varlen内核作为共同编排完成条件。

## 本批结果与下一步

[独立读凭据及captured生产者](../evidence/correctness/inference-captured-submission-2026-10-27.json)通过同一镜像12项检查：原6项Source、1项真实CUDA双Concat生产者、3项Compute受控协议及2项相邻回归。实际完整capture可保留原managed属性接原Source；未捕获command仍拒绝。票据只持原block／generation／精确范围，不含writer指针或第二Completion；导出封住本写代的新写及可写地址，真实发布后全部读者归还才能try_write。3项票据检查不是GPU Store接力证明。只编4个单元、链接1次、运行1进程，无SDK／NVCC／模型或性能重跑；Store／KV及真实Hybrid流水仍待接线。

[共同提交owner](../evidence/correctness/inference-submission-source-2026-10-27.json)将一次proxy、Queue、观察锁与完整Status集中在SubmissionSource，I32视图各保实际资源权限，原单输出入口已删除。原子bundle保所有视图，两个不同输出送入同B两个输入，source／device-wait均为1；无效后项不部分替换旧绑定。poll／wait不关闭导出，view独立后显式关闭共同Source；失败已实际退休时可归还pin，原数学错误仍保留且不发布ready。一个CUDA进程六项检查全部通过，含原错误、region、capture复用和载体重用；四不同单元共五次编译、两次链接，首镜像未运行，补编仅增加失败退休断言。尚未接Store／KV pending或引擎连续流水，无新速度结论。

[既有capture接力](../evidence/correctness/inference-graph-output-capture-2026-10-27.json)已接原Graph准备／重放逻辑，不改Provider静态属性。一个真实ggml Gather→Add消费者检查通过：ready预录后原unbind释放旧读权，同块try_write交给生产者；生产者仍pending时复用原捕获，报告0次新录制／1次复用并取得精确结果。两个真实物理tuple miss均拒绝，普通submit仍不能绕过原权限。三个单元一次编译、一次链接、一个GPU进程，无SDK重建。reuse-only仍有原metadata hook／预算／权限检查，不表示零分配；未接静态阻塞的生产者、多segment、状态／KV或引擎连续流水，也未测吞吐。

[CPU提前准备](../evidence/correctness/inference-cpu-ahead-2026-10-27.json)只针对普通B>1整批可继续的旅程，在冷预算内预留整数scratch。原AfterSubmit控制先执行，再准备位置／旧页拓扑；下一次完整匹配owner、epoch、长度、顺序及采样形状才使用。布局创建、状态／tail接纳和设备绑定仍在下一真实prepare，R1／MTP跳过。一个tiny CUDA进程共10次Graph调用、18输出，9条完整非时间提交记录、IDs和RNG一致，实际命中1次，换序／暂停取消后丢弃2次。首次fixture漏begin_turn在Graph前拒绝，修正后通过；一个不同单元编译两次。未测完整速度，也不声称隐藏整段cold_batch。

[Region准备合同](../evidence/correctness/inference-graph-output-region-2026-10-27.json)明确prepare／seal只处理metadata／资源／未提交recording，不读写载荷或接纳设备工作，移除按hook存在与否拒绝的过度gate。真实CUDA Store region在生产者仍pending时完成准备，最终写入精确值并归还原权限，一项定向检查通过。首次Gather→Add实际选中managed／host-blocking ggml region，被原runtime门正确拒绝；完整保留失败，不放宽该门。两不同单元共三次编译、两次链接与两进程，无SDK重建。尚未支持该ggml路径、pending Store源／KV或完整引擎流水。

[输出与载体分离](../evidence/correctness/inference-graph-output-detach-2026-10-27.json)取得三项有限CUDA通过结果：A完整成功且实际退休后，B仍读取旧输出时，A可detach并绑定另一块真实writer提交下一代。B取得旧值，旧source的Status不随新提交改变；pending／原错误均不解除原保留。两单元、两次编译、一次链接、一个GPU进程，无载荷复制或新执行器。该检查实际使用两块输出缓冲，不代表完整状态／KV接力或引擎持续流水已接通，也没有提速结论。

[在途设备token接力](../evidence/correctness/inference-graph-output-source-2026-10-27.json)沿原Invocation、Queue、InputSource及有序读权限实现一次链。两项CUDA检查通过：host gate实际阻止生产者完成时后继仍可提交并取得正确值；生产者checked-index错误使后继保留原code／message并拒绝发布，实际native uses归零。普通submit与capture绕过均拒绝。四个不同单元、五次编译尝试、一次链接、一个GPU进程；首次漏直接include及准备脚本错误保留。只支持已列明的固定I32／单Queue合同，未接引擎持续轮换、MTP或性能；ready只读分支和真实驱动故障未单独运行。

[有限尾块政策](../evidence/correctness/inference-bounded-prefill-2026-10-27.json)让有限Q使用满块与二次幂真尾块，不做padding。可能的prefix尾块按完整B／completion／sampler键进入原冷计划；额外exact容量从初始冷键数冻结，解除N×R直接耦合。六组主机合同通过，CUDA实际复用生成后的4行前缀，剩余T2以B2执行，请求期间新增编译和淘汰均为0。第一次N4配置未命中该覆盖，记录保留；改N2后才命中。默认无界行为及MTP有限Q拒绝保留，没有吞吐或跨分块数值一致性结论。

[相邻长prefill消费者](../evidence/correctness/inference-chunked-prefill-tail-2026-10-27.json)已适配Q64的真实尾块：P154为`64+64+16+8+2`，下一轮131行为`64+64+2+1`。一条真实Qwen消费者通过取消后旧head保留、首64行精确重建、末边界prefix历史及pending／RNG检查；A10／B3次提交共304行。B输入从15改为16以保留单次短prefill后穿插decode的原检查目的，不把改变后的工作量作旧性能对照。仅增量编译一个单元并链接一次，未扩测试矩阵。

[Apple完整消费者](../evidence/correctness/inference-apple-engine-slots-2026-10-27.json)按冻结的`bfe7159d4`源码增量编译六个单元并链接一次，普通／MTP K3两条tiny入口均通过，最终无遗留版本。两臂输出一致，MTP接受草稿数为零，不证明正向多token接受或任务质量；该窗不代签之后的尾块政策。未重建SDK／shader或下载，运行墙时不作为性能结论。

稳态状态及主机镜像的100%指本轮已实现合同，真实驱动故障未注入和有限平台／制品覆盖仍如实保留，不再把未运行故障等同于尚缺实现。深度二候选、有序后继及在途读取是重叠执行的新合同，单列在对应分支，不用现有双槽或tiny通过代签。

[执行错误身份桥接](../evidence/correctness/inference-request-error-identity-2026-10-27.json)在最终缩批后的实际bucket内，将原错误关联到完整RequestKey，保留计算Status及独立退休事实；未知成员和native故障保持批次级。四组主机合同及正常B2多轮CUDA检查通过，完整非时间报告一致；只编两个单元，无真实模型错误或设备故障注入。

共同循环、取消／ACK／背压、兼容dispatch、目录、版本接纳及prefill已各有真实消费者和有限检查，旧的65／80／60估计本批已据实际余项收敛。Provider来源身份与CPU实测偏好不构成模型白名单；完整CUDA保留ATen能力覆盖，修正了MLX capture过时注释。模型完整状态恢复保留在模型适配层，未以机械搬文件或新增server API作为收口条件；其它分支仍未完成。

[Vulkan完整消费者](../evidence/correctness/inference-vulkan-engine-slots-2026-10-27.json)复用145个链接输入，只编Model／匹配weights两个单元；普通与MTP K3各一条原tiny入口通过，均8输出且最终无遗留版本。MTP实际运行6轮验证／修复，但接受草稿数为0；输出一致只是此制品观察，不证明正向多token接受或任务质量。两次纯准备脚本错误保留，未重建SDK或shader。

[共享prefill事实](../evidence/correctness/inference-prefill-facts-2026-10-27.json)已将容量估算、冷形状、query上界、modifier输出预检、prefix尾部及实际切片接到同一无状态策略。四组主机边界检查通过，包含SIZE_MAX与零行完整命中；一条原Q2／B2多轮CUDA旅程与改动前完整非时间报告一致。只编一个Model及一个轻量测试单元，复用原223链接输入，未重跑MTP或性能窗口。下一步复核共同核心剩余责任，再安排具体改动。

[已计划载体准备](../evidence/correctness/inference-warmup-shapes-2026-10-27.json)新增`--warmup-shapes lazy|planned`，默认lazy。一次Model增量编译、复用223输入；四条tiny CUDA旅程和一个参数拒绝通过。B2多轮lazy／planned完整非时间报告一致，planned的六种载体全部在ready前创建；MTP K2与fixed分支也通过有限检查。它不执行虚假推理，不表示capture已录制或吞吐提高。

[Vulkan精确同步I/O](../evidence/correctness/inference-vulkan-halfword-io-2026-10-27.json)三组GPU检查通过，只编两个单元并链接一次：精确2／6字节读写／fill、跨1MiB的双边缘与中段、原aligned／owned3B及odd partial拒绝。短ReadLease在pool／wrapper释放后仍可读，最终quota归零；边缘用独立8B逻辑scratch和原16位搬运，未读写调用者padding。未做设备故障注入或模型性能结论。

[Vulkan原码16位Gather](../evidence/correctness/inference-vulkan-raw16-gather-2026-10-27.json)本批4项通过分属两个实际镜像：BF16 E1／F16 E3的部分源到Store、越界ID阻断发布、原F32回归。仅新增1个shader、5个不同C++单元，测试因首次额外短租约读回失败补编一次；保留2通过／2失败的原进程和后续两项通过。当时短租约同步D2H缺支持，case只用已初始化完整诊断buffer核原source字节；上方独立I/O批次已补此缺口，Graph投影与通用ready导入权限没有放宽。

[普通prefill分块CLI](../evidence/correctness/inference-prefill-chunk-cli-2026-10-27.json)三条tiny CUDA旅程通过：中间块不采样、异长B2多轮合批／pending保真、prefix复用及N1淘汰重建；三个非法组合明确拒绝。只编一个Model单元、复用223输入，无SDK重建。默认保持无限块长；MTP有限块长明确拒绝，不称已实现varlen、完整预热或吞吐提升。

[新32进程四臂窗口](../evidence/performance/inference-indexed-current-window-2026-10-27.json)已完成，普通decode单轮／两轮慢5.42%／4.64%，MTP慢7.46%／6.98%；最新完整表在下方。仅更新本窗，不把跨日速度变化归因某次优化。
[MTP异成员检查](../evidence/correctness/inference-mtp-batch-slots-2026-10-27.json)两轮B2/K3分别取1／3、3／1前缀，CPU／CUDA均通过；原Store候选字节、KV／draft续算、旧快照和未接受槽回收符合原合同，第二轮两池目的ID实际不同。受控接受不是自然质量证明。
[Vulkan半字Store](../evidence/correctness/inference-vulkan-halfword-store-2026-10-27.json)五项有限检查通过：精确2／6字节、原码／gap、换序／reset及外壳先释放；Completion沿原Run保活pipeline。当时未包含上述非4字节Gather；两批合起来仍不代表任意半字同步I/O完成。

[引擎原池状态接线](../evidence/correctness/inference-engine-slot-state-2026-10-27.json)已完成本批CPU／CUDA有限检查：tiny正式CLI普通／MTP各生成8 token；两后端的镜像、lag1、状态复用共6进程通过。原N2／R1三槽容量内两块交替，旧冻结快照字节不变；真实Qwen3.5-2B另通过17个接受前缀、7组后缀扰动，五种冷子图各选中18个索引GDN及18个索引卷积融合。实际6次C++编译、4个不同TU配置、4次链接；复用SDK及原生内核，无下载。保留一次源码并发变化的不确定编译、一次参数拒绝和一次真实append取回顺序失败；后者修正后另冻源码重新编译。没有模型质量或全平台完成结论。

本批[普通B2同窗A/B](../evidence/performance/inference-engine-slot-state-2026-10-27.json)完整非时间报告及198个ID一致，1728输入行／99次执行相同。decode＋2.20%、综合＋1.51%；prefill−0.80%且范围重叠，显存采样峰低48MiB。ready中位增加44.32ms，冷形状准备增加52.90ms；收益与代价分别保留。该A/B未包含llama.cpp或MTP；本批新四臂窗口另列，不能拿普通B2收益代签MTP提速。

索引状态的三个前提已落地：[原池槽资源](../evidence/correctness/inference-lease-slots-2026-10-26.json)25项、[融合写资格](../evidence/correctness/inference-fused-writer-2026-10-26.json)4项、[CUDA浮点Gather](../evidence/correctness/inference-cuda-slot-gather-2026-10-26.json)3组有限检查通过。后者保持F32／F16／BF16原码并支持行padding；当次尚未包含下述GDN融合，conv及完整模型性能仍未完成。
模型无关的Store描述及[槽绑定事务](../evidence/correctness/inference-slot-bindings-2026-10-27.json)另有6组检查通过，覆盖元数据失败不消费writer、寿命、真实地址与预算；这些资源类型不代表实际Store执行已接通。
[读权限撤回](../evidence/correctness/inference-projected-read-release-2026-10-27.json)11项检查通过：真实退休后只撤回指定源及其alias与guard，旧源可重获写权，其它绑定仍保留；包含三项原unbind回归。该次检查尚未包含下述实际Store执行，原生索引GDN／conv仍是剩余实现的大头。

[Store实际执行](../evidence/correctness/inference-slot-store-2026-10-27.json)已接默认CPU／CUDA装配：5个新实际写入检查及8个相邻权限合同通过，覆盖固定索引载体复用、非连续目标槽、未选择槽、原结果寿命、非法重绑和reset后失权拒绝。本批只成功编译6个不同单元、链接1个镜像，保留首次测试命令缺Toolkit头路径的失败；未重建SDK、未接引擎或测性能。该fallback不支持capture、冷索引上传可能等待，下一步仍是原生索引融合。

[私有索引GDN](../evidence/correctness/inference-indexed-gdn-native-2026-10-27.json)三组原生检查通过：K1、T4>K2最后K个状态及BF16 V／beta，与原稠密F32路径逐位一致，另核源槽、未选槽及padding不变。首次BF16失败来自旧融合预检漏排除新模式，已窄修并复用成功内核／测试对象；实际只编两个不同NVCC单元和一个测试单元，dispatcher因该修正补编一次。保持原F32运算体及旧fused-cache布局；未证明公开Provider／Graph／引擎接通或完整提速。

[CUDA GDN的Graph融合](../evidence/correctness/inference-indexed-gdn-graph-2026-10-27.json)15项有限检查通过：原13项Store／权限合同，加上Y发布、仅写状态两种实际融合；后者覆盖reset后拒绝旧权限及取回原事务。共用唯一StoreBinding及一次小索引上传，原Recipe的effect roots保留无Y输出的状态写入；不复制状态载荷。另修权限预留误用旧extra计数的问题，扩容失败仍保留原投影。7个不同C++单元增量编译，5个产品对象全程复用，追加编译只修测试；各通过结果的实际镜像、DLL装配及测试预算／alignment失败均在记录中。尚未接引擎或conv，未测完整性能，也未验证全部平台与图形。

[可移植源绑定](../evidence/correctness/inference-slot-source-fallback-2026-10-27.json)本批11项有限检查通过：共同冷准备按原Graph关联Store／Gather，CPU稠密及padding源槽可取回重用，替换源ID前先确认合法准备成功、替换后拒绝；CUDA原两融合及公开Gather结果后的默认fallback均与原稠密状态逐位一致。默认CCCL叶子缺失的读取资格已补，未强制换Provider。首次CPU缺输出绑定、CUDA测试误以为无用节点会裁掉，以及缺资格的原失败均保留；通过结果分属实际镜像，不声称同一最终镜像重跑全组。ggml叶子资格已补但本批未单独命中，尚未接引擎／conv或证明提速。

[原生槽写入](../evidence/correctness/inference-native-slot-store-2026-10-27.json)窗口中，新增Apple 3项（4种载荷）与Vulkan 3项有限检查通过，共同层另有原CPU／CUDA 11项回归通过。三种实际镜像分别记录；共同层回归后的默认初始化拼写修正另经两原生镜像编译，不冒称全组在同一源码镜像重跑。原生路径复用唯一槽事务、原ExecutionSource、FIFO／Completion及真实退休，不伪造Compute Queue；普通Gather保留实际stride、原精度及选行权限。该窗口覆盖生成结果、结果保活、未选槽／padding和reset回收，只检查4字节对齐item／offset；其后CPU与Apple补充见下，conv与完整引擎仍未完成。

[CPU默认选行](../evidence/correctness/inference-cpu-storage-gather-2026-10-27.json)已补上述CPU缺口，11项有限检查通过：普通Compute／Graph按存储格式复用Native原码选行，保留padding行stride，避免整池CONT及F16／BF16经F32往返；新规则不依赖模型、硬件或Store注解，稠密F32和已有实测规则保留。检查包含默认槽投影、两种普通入口、原7项槽合同和一项既有路由回归，未测模型或速度。仅增量编译5个单元；保留测试编译失败与离线XML汇总修正，不重跑已通过进程。

[Apple精确字节范围](../evidence/correctness/inference-apple-slot-byte-ranges-2026-10-27.json)5项有限检查通过：原3项Graph回归，加生成的BF16 2／6字节行及原FIFO拷贝合同。实际对齐范围仍有2次native blit，2字节范围为精确host copy；全部实际范围和别名先验后才写，源失败不重算。保留原事务、结果保活、真实退休及reset回收，不读写额外padding。只增量编译3个单元，保留两次验证配方缺头文件／搜索路径失败；未跑模型、性能或原生故障注入。Vulkan尾部仍需真正的原码索引／copy承载，不能仅放宽4字节transfer门。

[CUDA索引卷积及共用融合](../evidence/correctness/inference-indexed-conv-graph-2026-10-27.json)7项有限GPU检查通过：4项新conv检查及3项原GDN回归。普通图的Gather／Concat／Cast／Conv与最后K个原精度history窗口共用原槽事务；实际覆盖F32多成员、BF16长chunk、F16短chunk及逃逸读者fallback。更换目标槽ID后的第二轮实际复用原CUDA录制，历史原码、未选槽及padding保持；不复制状态载荷。只编译6个不同单元，共7次编译、2次链接；首镜像未运行，最终镜像只运行一次。没有完整引擎、模型性能或原生故障注入结论；当时尚未包含下述共同批量控制。

[共同批量控制与事务复用](../evidence/correctness/inference-batched-slot-controls-2026-10-27.json)取得29项有限CPU／CUDA通过结果：按真实执行源／Queue／资源共用固定控制载体，独立池各保真实槽号；覆盖两轮换ID、未全部take拒绝重绑、reset及真实晚期权限失败回滚。clear／refill保留原元数据并重新核租约；两处权限预留已去掉重复计数。6个不同C++单元、9次编译、2次链接，原生CUDA与SDK全部复用。首镜像26项通过、3项因测试漏Store控制边而拒绝构图；只改两测试单元，第二镜像仅重跑这3项并通过，未重复原通过项。首次重载解析编译失败与两镜像结果均保留。Apple／Vulkan本批未新运行，其原生copy内部仍可临时分配；没有模型或速度结论。下一步接入真实引擎，不能用组件通过代签MTP接受／回滚及完整提速。

KV工作帧已接通Compute、Graph及引擎：41项页存储检查、23组前缀／工作态检查、7个CPU／Vulkan／CUDA功能进程和4个递归MTP对照进程通过。Apple复用缓存增量编译11个单元，tiny普通／MTP及lag1两次运行通过，状态字节、续算、旧快照及最终资源归零一致。旧快照、失败边界、封存失败与较短接受前缀均有有限覆盖；[Windows功能记录](../evidence/correctness/inference-paged-frame-2026-10-26.json)和[Apple功能记录](../evidence/correctness/inference-paged-frame-macos-2026-10-26.json)保留原失败和未验证项。本批没有新Linux运行，4个诊断probe虽已适配但未单独编译；有限检查不代表全平台与模型质量穷尽。

[同窗Hyle控制／候选B2](../evidence/performance/inference-paged-frame-2026-10-26.json)的六计时＋两独立内存进程全部成功：prefill7509.27→7401.39、decode156.72→158.30、综合138.47→139.75 tok/s；分别−1.44%（范围重叠）、＋1.01%／＋0.92%（范围不重叠），三样本不代表统计显著性。显存采样峰相同，未证明节省。198个输出ID、1728输入行、99次执行及完整非时间报告一致；本批没有新llama.cpp进程，不刷新旧跨引擎对照表。

### 此前交付依据

- 文档：补齐16份缺页并校正范围与合同；入口见[架构索引](../architecture/README.md)。
- 共同请求循环、工作态和取消交付：[共同循环](../evidence/correctness/inference-request-loop-2026-10-26.json)、[工作态迁移](../evidence/correctness/inference-working-state-2026-10-26.json)、[请求控制](../evidence/correctness/inference-request-controls-2026-10-26.json)。
- 状态块复用与批量权限撤回：[状态复用](../evidence/correctness/inference-reusable-state-2026-10-26.json)、[批量append](../evidence/performance/inference-bulk-append-2026-10-26.json)。后者未证明提速。
- 完整主机镜像、普通／MTP／lag1：[镜像](../evidence/correctness/inference-host-image-2026-10-26.json)、[lag1](../evidence/correctness/inference-host-image-lag1-2026-10-26.json)、[原生平台](../evidence/correctness/inference-native-host-image-2026-10-26.json)。固定bank归还槽不等于释放显存；模型搬运仍同步。
- 副本保活、非阻塞观察及关闭后回收：[退休](../evidence/correctness/inference-transfer-retirement-2026-10-26.json)、[公开副本接口](../evidence/correctness/inference-copy-public-retention-2026-10-26.json)。观察错误不等于退休。
- Vulkan片段对齐与CUDA小通道卷积：[片段接入](../evidence/correctness/inference-stream-padding-2026-10-26.json)、[卷积承载](../evidence/correctness/inference-cuda-causal-conv-2026-10-26.json)。
- SDK权重冻结：[所有权边界](../evidence/correctness/inference-storage-freeze-2026-10-26.json)。

单carrier不是已证实的落后主因。[同设置CPU诊断](../evidence/performance/inference-b2-peer-cpu-critical-2026-10-26.json)支持减少主机协议工作，未证明整个区间都可消除。剩余职责见[整体设计](../architecture/inference-design.md)，原始性能与候选取舍见[性能资料](performance/decode.md)。

## 最新完整性能对照

最新两版Hyle同窗普通B2对照（各3次计时，另各1次内存采样）：

| 指标 | 改动前 | 当前 | 变化 |
|---|---:|---:|---:|
| prefill tok/s | 7668.03 | 7606.82 | −0.80%，范围重叠 |
| decode tok/s | 158.48 | 161.96 | ＋2.20% |
| 多轮综合 tok/s | 140.55 | 142.67 | ＋1.51% |
| dedicated显存峰 GiB | 3.999 | 3.952 | −48MiB，采样下界 |

这是Hyle版本对照，不与下列旧双引擎窗口拼接。三样本不代表统计显著性；ready显存基线未知。

以下各格为 **Hyle／llama.cpp**。速度单位 tok/s，显存为独立采样 dedicated 峰 GiB。B2保留此前跨引擎窗口，单用户普通／MTP使用新32进程窗口；上方普通B2的两版Hyle A/B不混入跨引擎表。

| CUDA 工况 | prefill | decode | 综合速度 | 显存峰 GiB |
|---|---:|---:|---:|---:|
| 单用户单轮普通 | 2160.112／1559.773 | 91.483／96.723 | 84.200／85.325 | 3.876／3.728 |
| 单用户单轮 MTP | 2055.844／1476.649 | 120.143／129.830 | 105.853／109.071 | 4.013／3.896 |
| 单用户两轮普通 | 2613.800／2422.222 | 92.057／96.537 | 86.083／89.395 | 4.009／3.728 |
| 单用户两轮 MTP | 2481.770／2297.157 | 123.706／132.991 | 112.146／117.957 | 4.151／3.896 |
| 普通 B2 三轮固定输入 | 7469.731／6086.119 | 154.941／170.683 | 136.981／143.737 | 3.999／4.743 |

前四行来自2026-10-27的[32进程窗口](../evidence/performance/inference-indexed-current-window-2026-10-27.json)：索引状态接线镜像，不含随后CLI分块政策。各臂内稳定，16份Hyle非计时报告与此前对应窗口一致；自然续文跨引擎不同，旧时间不混算。最后一行来自此前状态复用后的[8进程B2窗口](../evidence/performance/inference-b2-reusable-state-2026-10-26.json)：C1024、N2／R2、prefix off、无MTP，每请求三轮P256＋32固定输入；1728输入行、99物理执行和198输出ID一致。prefill＋22.73%、decode−9.22%、综合−4.70%；工作集峰Hyle／参考1.080／4.106GiB，private commit5.624／5.926GiB，ready内存基线未知。参考为实际保留版本，不声称最新上游。各窗口不混算，ID一致也不代表中间数学逐位相同；不能用跨窗绝对速度变化归因工作帧收益或退步。

CPU最新[同窗对照](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)：i9-13900K、原生4／ATen4线程、调用线程、P154＋16、C256、N1／R1、无MTP；参考为同版本AVX2／FMA／F16C。

| CPU 配置 | prefill | decode | 综合速度 | 工作集峰 GiB |
|---|---:|---:|---:|---:|
| Hyle／llama.cpp | 43.953／42.937 | 11.437／12.027 | 3.323／3.314 | 5.002／3.828 |

CPU decode仍慢约4.9%，private commit为5.740／0.606GiB；映射和私有分配不能当权重大小。内存峰均为有限采样下界，缺样不记零；加载口径和OS缓存不同，不排名冷盘。CPU＋GPU、Apple／Vulkan仍缺当前完整消费者公平基线。
