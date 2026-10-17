# G09 单CPU与单GPU联合流水

状态：**BLOCKED-from-CANDIDATE，2026-10-17；产品形状冻结，选定交付及 CUDA P1 已通过，另三组 A/A 资格未通过，聚合 P1/终审未验收**。
baseline为`9e2f25c469634a6f723e76eaf1aa2db5973bca9a`。
candidate为`3524413272cb52d06af5e64d634edbfef09b9864`；产品实现自`e09f30925`未再变化。
G07/G08/G10已接受，当前G07–G10 goal完成3/4；剩余G09的三组测量资格与聚合终审，完整文本CLI未开始。
最新[有限诊断](../../../../evidence/architecture/inference-g09-completion-observation/receipt.json)
在用户授权的管理员重试中留下前两形状记录，但访问冲突中断、native readiness 归属未得；不恢复资格。
后续[具体故障现场](../../../../evidence/architecture/inference-g09-completion-observation/nsys-procdump-receipt.json)
已取得驱动 NULL-read 及经过 Nsight 拦截层的上传提交栈；根因未签，直接调试同程序则四形状完整通过。

## 需求、真实事实与owner

本卡承接[原G09](../project/inference-single-machine-foundation-audit-2026-10.md)和
[消费形状](../project/inference-consumer-adaptation-plan-2026-10.md)：CPU与一个GPU都执行真实数学，
同一DAG的无依赖支路以及两个独立microbatch不因额外host观察被串行化；必要数据/控制/别名依赖仍须等待。
独立invocation存在或API名为async都不证明该旅程完成，不把物理copy/compute并发保证成跨硬件合同。

当前新事实：Apple `drive_graph` 的fragment循环在每个CPU/native提交后无条件await；
Vulkan同样先完成fragment和需要的transfer才推进下一个。冷compile按原拓扑连续target分组，
没有由fragment实际依赖决定的ready推进。因此按GPU支路→独立CPU支路建图时，后一支路在源码上
就被等待挡住；这超出必要CPU↔GPU数据依赖，构成本轮合法reopen trigger。
[当前源码与原owner身份](../../../../evidence/architecture/inference-g09-joint-pipeline-intake/receipt.json)
保存精确hash/锚点，尚不是运行时重叠或速度证据。

初始intake只打开Apple/Vulkan；随后一次最小CUDA公开consumer反证已确认同一J1旅程的额外串行化。
当前施工owner为`graph_route_cpu_apple`、`graph_route_cpu_vulkan`、`graph`、`graph_route_cpu_cuda`，
在审计时均为AUDITED，现进入上述固定candidate。common/CUDA只纳入原cold segment边界与联合提交推进，不扩大为普通Graph全面重构。
leaf数学、native Queue/transfer/allocator继续借原已签owner，不因此自动重开。
不重新打开G01–G08已接受的数学、packed、分页或数值选择范围。

### 本轮有限审计与反证检查点

[CUDA反证原始记录](../../../../evidence/architecture/inference-g09-cuda-preaudit/receipt.json)
含两种插入顺序、原SDK输入hash、实际CPU执行区间与CUPTI `cuLaunchKernel`观察、独立数学oracle，
以及配置/运行环境失败和累计53.962330539秒。只编译一个consumer，未重建产品或依赖。
原SDK的common Graph、CUDA joint、CPU backend/provider源码hash与当前baseline逐项一致。

- CPU先插入：独立CPU、CUDA segment都无前置依赖，report额外serialization edge为0；
  实际GPU enqueue仍在整个独立CPU执行结束后才开始。这证明当前公开路线存在额外串行化。
- GPU先插入：独立CPU节点与后续join进入同一个CPU segment，段入口统一等待GPU；
  独立prefix因此被join的依赖牵连。Apple/Vulkan的连续target分片有同类边界问题。
- 两图数学oracle均PASS；这些是旧产品的diagnostic反证，不是新实现correctness/P1或正向设备重叠PASS。

同一raw包保存独立审计`independent-consumer-audit.json`：有限MUST为cold分片边界、真实ready推进、
control/effect/alias依赖与所有accepted prefixes真实退休。必要lane/预算复用边不能因没有tensor边就删除。
以下FIX冻结这些问题的owner/API/ownership/失败/并发/资源账，不再扩大缺口扫描。

## AUDITED：有限改动集与形状冻结

| ID | 能力裁决与允许改动 | 直接验收 |
|---|---|---|
| J-F1 | J1 `PARTIAL/FIX`，SELF原Graph冷依赖投影。Apple/Vulkan连续target分片在后续节点引入新的外部producer/control依赖时分开；common segment同样保留独立prefix。保已有native fusion、显式control/effect、真实lane与预算复用边 | 两插入顺序分片/依赖，纯串行链、不必拆开的同target链、control-only与effect留出；不机械逐op拆分 |
| J-F2 | J1/J2 `PARTIAL/FIX`，SELF原hybrid组合推进。私有有界ready/frontier使用BORROW原Scheduler、CPU Queue、native Program/Invocation；native/root与一个CPU coarse child，最多两个原Sched fibers，保持每实际Queue单producer。严格串行DAG沿原inline推进 | 真数学两分支/join，两slots各65步；原worker数1/2，不靠fake provider或sleep签overlap |
| J-F3 | J3 `PARTIAL/FIX`，原边界准备/搬运按实际消费者的依赖推进；native/transfer/page controller仍由原owner承载，可变page/transfer入场由native侧串行拥有。CPU repack服从同CPU Queue单producer；已可用host/native引用不增加设备等待 | host/native输入、跨域与相邻native片段、同源多consumer，原page/source旅程；记录真实copy与host observation |
| J-F4 | J3权限/结果 `ROUTED/BORROW`，只补本轮并发保护：短入场/结果发布与LeaseSubmission.protect序列化，绝不持总锁执行CPU数学或native wait。取消/首错停止未接收工作；所有已接收worker/outcome/native prefix真实退休后原owner发布terminal | 第二支路失败、取消、poll error后仍在途、旧结果阻止reuse、drop/move/close、quota/OOM与实际权限/alias |
| J-F5 | J4 `PARTIAL/FIX`，有界依赖metadata/每slot状态、原resource report和真实公开consumer；文档明确使用成本，三交付/实际硬件与适用风险补证。没有新public API、线程池、通用scheduler或新vendor | 下述固定targets、普通selector、P0账及固定candidate后P1、独立终审/LOC |

J1/J2是复合机制，模型/request/microbatch分组仍为APPLICATION；不能要求消费者拆开同一个DAG。
J3原tensor/lease/copy/native/math均KEEP/BORROW，未经资格的GPU mutable/alias域仍按原合同拒绝。
J4 provider-private/UMA成本继续允许明确UNKNOWN；不把未知成本写成0。G10线程预算不在这里提前实现。

普通调用保持`GraphBuilder`→placement/compile→原Sched/Queues→两个独立`create_invocation`→
bind/prepare_replay→两次submit→各自checked owning Results→close；调用者持有原contexts/resources/
allocator/Sched/Queues到真实退休，结果lease继续保护backing。共享只读输入允许，可变workspace与结果slot独立。
不增加调用者必填调度policy或与benchmark相关的选项；三种C++交付入口不变。

common Graph的重用接缝限定为same-build私有逐segment提交：一次begin/最终finish仍由单owner独占；
worker读冻结的plan/bindings/routes/capture和本segment私有step invocation/scratch slice，返回本segment
Completion/error outcome；短互斥发布后才解锁后继。原cursor逐次advance路线保留为同机制的串行调用者。
cursor的next/pending/tail、aggregate、phase/exclusive与最终退休不可并发裸写。原QueueSequence必须在
执行它的fiber内begin→finish，保实际Queue/lane顺序，不跨线程转交活动sequence。
`blocks_host`只描述真实宿主执行成本，不能当设备完成证明；不得为了推进将其改成false。

新增冷依赖投影/frontier准备为O(V+E)，每slot状态O(F+I+E)，空间经原metadata allocator/budget计费；
不每op创建task/fiber，不新增OS线程。并发图每次submit至多一个额外原Sched detached driver及其
原pool stack，原Sched allocation/admission失败须转Status；严格串行图没有该额外driver。
warm frontier使用已准备的依赖计数/就绪队列/短发布；反复pending观察不受该线性界约束，
原copy、必要wait、native私有费用仍单独记账。
不能预先承诺所有provider零分配或所有硬件物理copy/compute并发。

固定新增验证入口：common `hyle_graph_test` 的`NativePartition.*`/`JointPipeline.*`有限filters；
三个现有`hyle_cpu_{cuda,apple,vulkan}_contracts`扩入公开流水旅程，必要独立硬件trace fixture使用
`hyle_cpu_{cuda,apple,vulkan}_pipeline`名称。只为实际并发/lifetime风险增加测试，不为实现逐行镜像。
common影响仅扩到原Graph executable、原leased/batch与三hybrid的真实直接消费者，未变叶证据复用。

P1冻结为两自然形状：等长独立CPU/GPU数学支路+join、两个独立microbatch的CPU→GPU→CPU链；
必要纯串行CPU→GPU→CPU作holdout。相同输入/数学/输出/原contexts/queues/slots、包括实际copy、
prepare/submit/checked observation与退休；cold compile/native准备单独列账，不能从一侧省掉。
baseline仍为9e2f25c4原HYLE；平台原5%/n15–31、A/A、双顺序和适用placement合同不变。
每形状实际维度/批数/迭代预算与两侧时序静态账单在构造benchmark前绑定该具名consumer后固定，
不可计时后挑有利尺寸；prospective独立review/text检查与短diagnostic合格后才正式采样。

本轮实现/检查按J-F1–J-F5批量完成，验证失败仅定向修复；架构/API/常态成本级别变化才回退本审计。
G07/G08已接受范围、叶Provider数学、公共构建/性能仪器不重写。没有产品gate在本节被预先判PASS。

主纵切为单机推理的两microbatch：公开CPU→GPU→CPU链与一条独立CPU数值支路、持久输入和两个slots。
正交纵切为批处理矩阵/向量管线：两个独立分支、显式join和不同图插入顺序，不引入token/request身份。
使用原GraphBuilder、NodePlacement、compile、原Scheduler/Queues、create_invocation、bind、
prepare_replay、submit、checked completion、owning Results与close。两slots各自拥有可变状态，
保留旧结果时拒绝覆盖，原资源/allocator/contexts继续活到真实退休。

## 有限能力审计与停止边界

| 族 | 必须裁决 | 初始动作 |
|---|---|---|
| J1单DAG独立支路 | CPU/GPU真实数学及join，两个插入顺序；必需数据/control/effect/alias边与额外串行化区分 | 审原cold依赖与Sched/native承载；仅为已证阻塞冻结必要FIX，不建通用新scheduler |
| J2持续两microbatch | 至少两个独立slots、65步输入变化、旧结果、请求交错；共享只读源与独立workspace | 优先BORROW原invocation/Sched/资源，证明自然公开组合的完整路线和真实overlap |
| J3边界与完成 | CPU↔GPU transfer、相邻native片段、host observation和native wait；错误/取消不发布成功，accepted prefix真实排空 | 复用F1/F2/F4与原native退休；只修必要进度组合，不重写copy/Queue/allocator |
| J4有限成本与交付 | cold依赖metadata、warm分配/提交/观察/搬运；实际profile、权限和三种交付 | 原report/native工具与有限fixture取证，未知private/UMA保留；不扩公共统计/观测产品 |

AUDITED前完成一次有限源码/peer审计和至多一次针对性反证，固定能力表、public调用草图、
API/ownership/失败/并发/资源账、SELF/BORROW/ROUTED/APPLICATION和有限FIX ledger。
本卡的范围不授权为了测试行新增public API，不把引擎公平性、continuous batching、cache/offload政策、
tokenizer、多GPU/多机或G10全进程线程预算放进来。不以要求消费者拆图/重写退休状态机代签同DAG要求。

固定peer沿原SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3，不刷新checkout。
只核本题的stream/event/prefetch、阶段owner与microbatch消费；每族至多两个主要机制来源，
第三个作反偏置，不复制完整引擎调度。源码身份与原native能力先核，再设计。

审计以2小时作检查点，非到点伪造完成；慢构建/验证按用户已有授权沿同配置自动追加时间，
保留原失败与累计费用。审计结束即冻结有限FIX并批量实现、集中检查、定向修复，不追加全库扫描。
产品候选冻结后才启动一次预声明P1；达到明确停止条件须裁决，不以继续调查替代状态。

## 预声明验证与性能合同

- source/普通测试仅选本owner与真实直接消费者：`tests/graph/cpu_apple_contracts`中的
  `hyle_cpu_apple_contracts`、Vulkan对应`hyle_cpu_vulkan_contracts`；CUDA用同名目录中的
  `hyle_cpu_cuda_contracts`和原公开联合consumer核验。新增流水target/filter在AUDITED固定。
- 真数学：CPU/GPU两分支+join的独立oracle，纯串行必要链作对照，两图插入顺序；两个slots至少65步，
  不同输入、只读共享、旧结果、move/drop/close、错误/取消与accepted prefix，有限quota/OOM/alias/control。
- 原G02/G03/G06受影响joint消费只补进度/权限接缝；数学和未变叶依赖证据按身份复用，不跑全库。
- 真硬件：Windows/WSL CPU+CUDA（先核BORROW）、Air CPU+Apple、Windows/Air CPU+Vulkan。
  沿原native工具或透明有限fixture保留真实CPU区间、native执行/完成、transfer/wait链；
  自然pending数量、wall-time变小或单一host计时不单独证明设备重叠。不能取到某native时间线时如实待证。
- 改动owner的ASan/UBSan/TSan、实际Metal/Vulkan validation与必要CUDA工具、source/static/shared/
  迁址/独立公开头按实际diff和selector选择。没有产品变化的路线不重复原叶资格；新增consumer仍须运行。
- P0先核O(V+E)冷依赖准备与有界每slot状态、warm准备/观察/搬运及实际退休。单DAG依赖推进若改变
  算法/常态成本，AUDITED冻结至多两种自然并发形状及必要串行holdout的同合同完整warm P1；
  peer为本baseline原HYLE，原5%/n15–31、workload A/A/双顺序/适用placement及artifact预检，
  独立review与text/反汇编记录必须在正式采样前完成。不能以纯trace或机制计数冒充5%性能PASS。

普通最多4jobs/4logical、BelowNormal，CUDA pool1/split3/Ninja2；Windows/WSL/Air重型任务串行，
Mac只用实际确认身份的MacBook Air。使用原合格依赖/对象和SDK，不让加一个joint消费者触发全库构建。
修改前保存baseline registry，运行selector、docs/closure及状态投影/diff；最终聚合七门/七问、
独立阶段终审、明确重开条件与统一LOC。完整流水与性能门在下述检查点仍待证，不能预记PASS；
上述旧SDK反证不代签这些门。

## 实现检查点，2026-10-17

此检查点当时仍为AUDITED。J-F1–J-F4实现已接齐，三native路线首轮实际消费者已运行；当时尚未固定最终candidate，
不把局部运行结果当成G09聚合验收。G07/G08仍为本goal已验收2/4，G10未开始。

- J-F1：common segment与Apple/Vulkan冷分片保留独立prefix；真实data/control以及lane/预算复用边保留。
- J-F2：三路线均接入原Sched的root/native与最多一个CPU coarse child；common逐segment返回local
  outcome，由CUDA root独占commit/finish。实际活动阶段完全成链时保inline，不增加OS线程或每op任务。
- J-F3：原输入repack、上传、下载与数学各按真实消费者依赖推进；绑定已经ready的host/native输入
  不提交空操作。Apple/Vulkan root保留pending原生句柄后继续领取其它ready工作，复制/分页仍用原owner。
  Apple共享内存复制仍为原同步调用，不把它改写成零成本或通用copy/compute并发保证。
- J-F4：prefix protect使用短锁，数学/native wait不持总锁；取消与领取job在同一短monitor中线性化。
  CPU child在root独立引用保护下，锁内归还引用、发布stopped并notify，解锁后不再访问Run。
  取消/首错时停止新入场、汇合CPU，锁外用每个仍有效的原native/transfer句柄checked wait至真实退休，
  然后清句柄并释放原Lease body sentinel。
- J-F5尚未验收：冷依赖/每slot账已有实现，native pending观察每轮可扫描P个在途句柄，最坏O(P)，
  用原有1ms fiber等待；不能仅凭frontier O(V+E)宣称总warm观察也是O(E)或无host成本。

### 本批有限验证与定向修复

[CUDA首轮](../../../../evidence/correctness/inference-g09-cuda-implementation-checkpoint/receipt.json)
保留原预检拒绝199条重编计划的事实；未启动CUDA算子重建。复用同一既有合格叶archive，
只做本owner的C++编译与消费者链接，不将此scratch复用投影代签新的source/static/shared交付。
后续源码复核发现取消后仍可领取未claim job，已修复并在下一轮复测。

[CUDA修复后复测](../../../../evidence/correctness/inference-g09-cuda-lifecycle-recheck/receipt.json)：
81项common、289项原Graph executable/direct consumer、3项真实native retirement，合计373项PASS；
原CPU+CUDA contracts、G08 joint回归也PASS。新公开消费者两插入顺序、1/2 workers、两slots各65步、
旧结果、取消、move/drop与drain均PASS。CUPTI记录280个实际kernel区间；两顺序各8次单DAG CPU数学
区间均有真实kernel重叠，双slot两顺序也观察到重叠。这只签该硬件/旅程，不能冒充P1或所有尺寸并发保证。
HYLE_TESTING-only的第四条native-retirement故障用例在此shipping OFF profile未编译，适用风险仍待跑。
本次增量仅7个C++对象、CUDA对象0；含第一轮与两次manifest的累计224.859秒保留。

[Windows Vulkan](../../../../evidence/correctness/inference-g09-vulkan-implementation-checkpoint/receipt.json)：
原contracts/G08 joint回归及5组两slot/65步旅程PASS（两插入顺序×1/2 workers，另加native输入）；
每组取消后close与原数学oracle均PASS。原失败保留：一次fixture误用MemoryResource.upload编译失败；
物理回收观察早于原Sched回调尾部返回；两次取消诊断显式停止。权限退休的逐wait active_uses==0
断言不变，最终物理回收用原scheduler.drain作确定屏障，实测blocks 1→0。
取消卡住的可达原因是丢掉原pending句柄后，没有主动observer继续推进Vulkan原abandoned prefix；
已用上述锁外checked wait修正，没有改叶Completion/Queue。累计活动407.874秒含独立诊断，不重置费用。

[MacBook Air Apple](../../../../evidence/correctness/inference-g09-apple-implementation-checkpoint/receipt.json)：
实查Mac16,12/M4、macOS27.0.1(26A434)、AppleClang21、原Metal组件；原contracts/G08 joint和同5组
两slot/65步/取消旅程PASS，累计23.098秒。初始环境未加文档规定Metal shim，configure失败，原失败保留；
沿同树修正PATH，未重装依赖或更改全局developer directory。实际编译前已改用Windows canonical
完整build/source Git archive，3321文件逐项校验，snapshot tree c29a1201437c2c1c91c1b0697aca8f2ac8be8b31；
没有用手选delta冒充最终合格源码，也不从验证副本回写产品。

独立复核原件按源码范围保留：
[CUDA mailbox/frontier](../../../../evidence/architecture/inference-g09-implementation-review/g09-implementation-review-1.json)、
[native阶段/取消领取](../../../../evidence/architecture/inference-g09-implementation-review/g09-native-implementation-review-2.json)、
[CUDA取消领取修复](../../../../evidence/architecture/inference-g09-implementation-review/g09-cuda-cancel-claim-review-3.json)、
[真实退休与child交接](../../../../evidence/architecture/inference-g09-implementation-review/g09-native-retirement-review-4.json)。
这些源码判定不代签未运行风险或P1。此前[79项common局部检查](../../../../evidence/architecture/inference-g09-partial-common-check/receipt.json)
及其原成本继续保留，不重复计成本。

### 跨平台与真实数学时间线检查点，2026-10-17

产品源码固定为e09f30925；本批只增加验证夹具，四owner仍为AUDITED，G10未开始。

[WSL CUDA](../../../../evidence/correctness/inference-g09-wsl-cuda-implementation-checkpoint/receipt.json)：
GCC14.2/CUDA13.3/RTX4070实际运行原contracts、G08 joint、新双slot/65步与取消旅程；
61项common、289项Graph executable/direct和3项native retirement均PASS，合计353次执行，
存在fixture覆盖交叠，不能称353个独立测试。CUPTI记录280个真实kernel区间，单图两顺序分别
7/8和8/8次有CPU数学/kernel重叠，双slot分别129和129次；不代签P1。
原预检210条边、147个CUDA对象的重建计划在编译前拒绝；同树复用逐项hash核验的未变叶archive，
实际只编译本owner/直接消费者C++，不冒充全新source/static/shared交付。累计133.680秒含原预检。

[Air Vulkan](../../../../evidence/correctness/inference-g09-air-vulkan-implementation-checkpoint/receipt.json)：
Mac16,12/M4原source集成，原contracts/G08 joint与5组双slot/65步/取消close旅程均PASS，
累计34.808秒。完整canonical build/source archive绑定e09f30925，共3321文件。
此结果尚无Vulkan实际数学GPU时间线，不能以pending或host墙钟替代。

[Air Metal实际数学](../../../../evidence/correctness/inference-g09-apple-real-math-timeline/receipt.json)：
固定同4M F32元素、默认CPU Exp与原GPU Neg及CPU join，单活动slot、warm一次后每插入顺序
8次诊断。透明装饰原Metal调用，按真实MTLFunction名`vn_Negativefloat32float32`追踪
pipeline/dispatch/command buffer；只接受无blit、恰好一次该数学dispatch、完成成功的buffer。
GPUStart/End与CPU executor数学区间使用同一system mach clock；两顺序各8/8次整个GPU区间
落入独立CPU Exp task，原join均在GPU完成后。固定double oracle、abs2e-6+rel2e-6，
拒绝非有限值；默认Abs/双slot/65步以及原contracts/G08 joint也PASS。
这只签该形状的真实重叠存在性，不签全尺寸/全provider或5%性能。

保留三种原结果：首次全局启用OBJCXX令未变GGML `.m`误按C++编译而失败，后将语言配置限定到
夹具自身；原短Abs支路在GPU启动前结束，未证明重叠；补充SiLU数值PASS，但默认ATen不使用
ParallelExecutorRef，记录只覆盖join，属于CPU观测不足，不能记为SiLU性能/并发失败。
Exp是最后一次固定补充诊断，不再换尺寸或算子。以上同树累计39.006秒包含各原失败及重试。
1us仅作诊断guard，不是统计或设备时间戳置信界；实际最小左右包含余量约384us/3777us。
运行源码及三份原快照、raw日志、binary hash与独立观察复核均在receipt归档中；最后只恢复
`native_joint_pipeline.cc`原有LF，保留CRLF→LF逐字节相等证明，不为格式变化重复构建。

### Vulkan原数学时间线检查点，2026-10-17

[双机原件及独立复核](../../../../evidence/correctness/inference-g09-vulkan-real-math-timeline/receipt.json)
补齐本卡Vulkan时间线正证。仍是同4M F32默认CPU Exp、原ggml GPU Neg与CPU join，单活动slot，
warm一次后每插入顺序8次；未换尺寸或数学。Windows RTX4070与Air M4均为原Neg query
8/8、完整GPU区间包含于独立CPU Exp task 8/8；两机默认5组双slot/65步/取消close也PASS。

测试专用loader沿原public loader_path透明转发，原logical device只附加其实际支持的
calibrated-timestamps扩展，原queues/features/pNext与数学不变。完整SPIRV字节识别原module，
再追踪原pipeline/bind/dispatch和实际QueueSubmit；不用只覆盖Recorder的旧钩子冒充ggml数学。
原neg_f32_data为4692字节；原ggml按actual Vulkan12浮点控制能力冷注入FP16控制指令，
Windows RTE+Denorm得到4768字节，Air RTE得到4744字节。夹具只对参照复现这段已锁定算法，
仍按最终完整长度/字节匹配，原shader照常转发；原始及派生SHA、具体能力与源码身份均保留。

Windows用实际10MHz QPC域，Air用实际MONOTONIC_RAW ns域。逐replay保存前后DEVICE+host
校准、maxDeviation、validBits/period、实际offset drift与有符号模差；只签计入这些观测误差后
的整个GPU dispatch bracket落入CPU数学区间。1us是诊断guard，双校准模型不作普适drift或
统计置信保证；这仍不是5% P1。query按原recording分配不同pair，在原command buffer中reset，
原checked completion之后只读availability、不追加host WAIT，原device销毁时释放pool。

原失败保留：首次未加入原设备浮点控制变换，原module未命中、query为0，未证明重叠；普通
数学和生命周期仍PASS。一次过严scratch预检拒绝了8条边中的未变外部生成器metadata与staging，
未启动build；修正后只重编两个夹具C++对象，无CUDA或shader重编。Windows同树累计
142.510秒包含原135.484秒及预检拒绝；Air30.696秒。没有产品源码改动或新的聚合gate PASS。

### Host风险与P0成本账检查点，2026-10-17

[WSL CUDA原始记录](../../../../evidence/correctness/inference-g09-cuda-host-risk-checkpoint/receipt.json)
中ASan、UBSan各通过61项common、289项原Graph executable和4项native-retirement执行；
HYLE_TESTING-only的accepted-observation拒绝故障分支此次确实编译并运行。另有原contracts、
双slot/65步/取消move/drop流水及原G06 hybrid paged/pending receiver消费者均PASS。
组间有重叠，354是每个配置的执行次数，不是唯一测试数。复用身份匹配的原native CUDA archive，
只增量编译host C++与消费者，不用这一scratch配置代签完整SDK或设备instrumentation。

TSan运行了同七组：common、executable和两个G06消费者PASS；retirement、contracts、pipeline
各报一条销毁/deallocate与先前atomic引用释放的竞争，原退出66和完整栈保留。首次WSL配置没有
编入单独的既有refcount acquire-fence观测模型；仍须逐条核对实际owner/HB并作有限复测，
不能仅凭已有工具限制判PASS。首次三配置累计608.510秒，包含这次TSan失败，不重置成本。
该记录未裁决全部host风险、Metal/Vulkan validation、CUDA device tools、交付或P1。

随后同树使用既有G02/G06/G08字节相同的final-acquire观测模型复测，七组全部PASS：
61 common、289 executable、4 retirement及contracts/流水/G06两消费者。模型只作用于实际
compute runtime、Graph、CPU+CUDA owner及fixture的编译单元，未改shipping源码或添加抑制。
首次过宽的CXX include方案在dry-run拒绝289条build边，未编译；限定到上述真实owner后为
94条增量边、CUDA对象0。原预检/配置费不重置，复测累计362.561秒；该CUDA host记录现合计
971.071秒。原三报FAIL仍在raw包，不能把有模型的host结论扩大成driver/私有分配器或设备shader证明。

[Air Apple原始记录](../../../../evidence/correctness/inference-g09-apple-host-risk-checkpoint/receipt.json)
中ASan、UBSan各通过61项common与285项executable，以及原contracts、新流水5场景和原G06
pending联合消费者。TSan中common/executable、新流水5场景与G06 pending也PASS；原contracts
进程在未变`src/compute_route_apple/apple_weights.mm:345`直接Objective-C Metal回调的byref发布处
报告竞争，该进程不能记PASS。原Metal-cpp模型只覆盖C++ wrapper，增加同一header路径无法覆盖
这一selector；不在本卡内顺手扩建模型。三配置累计185.846秒，原失败、身份和实际覆盖均保留。
[逐报源码/二进制复核](../../../../evidence/architecture/inference-g09-implementation-review/g09-tsan-observation-review-8.json)
确认WSL三报分别对应原SignatureState、RunState、ResultStorage末减引用fence，仍需既有模型复测；
Air回调源码与G09 baseline完全相同，现有模型的边界明确。局部PASS不代签这些待裁决边。

该边随后由独立有限[Metal ObjC观测工具任务](../../../../evidence/architecture/inference-g09-metal-objc-tsan-tool/tool-task.json)
完成：先按原SDK合同独立审查，仅在可丢弃验证副本给原command commit前加release、completion
callback第一句用其completed参数acquire，保原数学、byref捕获、owner与GPU退休。独立Mach-O
relocation复核确认注册0x4580→release0x462c→commit0x4634；callback acquire0x49f0在首个
capture读取0x49f8/0x49fc之前，未给G09内部同步附加HB或使用抑制。相同contracts、新流水5场景、
G06 pending在此模型下均PASS。完整3325文件副本和parent已恢复，五份原测试binary哈希不变。
原FAIL仍保留；首次33边预检拒绝和未链接otool符号guard失败均计入工具累计102.712秒，后者复用
同一已链接object/binary，用nm+relocation核对而未重编。Apple host记录现累计288.558秒。
该工具任务已CLOSED；command地址复用与外部模型可见性边界仍明确，不签全进程race-free或泄漏。

[Air Vulkan host风险原件](../../../../evidence/correctness/inference-g09-vulkan-host-risk-checkpoint/receipt.json)
中ASan/UBSan及使用原final-acquire观测模型的TSan均PASS；每配置61 common、285 executable，
另有原contracts、5组双slot/65步/取消流水和G06 pending联合消费者。沿原hash合同复用已资格的
generated Vulkan输入，未重新编译shader。三配置累计263.792秒；设备validation层、交付和P1另验。

[独立有限P0账](../../../../evidence/architecture/inference-g09-implementation-review/g09-p0-cost-review-7.json)
已复核14份产品源码身份；仅frontier/CSR与具名依赖投影具有所列线性界，每slot metadata有界。
Apple/Vulkan在每次新native入场前扫描现有pending：若N个ready任务持续pending，仅入场阶段就可能
发生N(N−1)/2次重复观察，随后轮询次数还取决于实际延迟。CUDA重试逐次重扫segment依赖，成本含
Σ(retry次数×该段indegree)。这些成本不被O(V+E)冷准备或原1ms等待抹掉。
prepare_replay、原复制、Sched fiber/唤醒、native私有资源与真实退休另计；ResourceReport本身可能
分配临时scratch并排序去重，ReplayReport不是实际poll/kernel/时间线计数。没有发现本轮新增
具体MUST；不以成本结构单独授权改产品或更换P1形状，实际性能仍按固定合同待测。

### 固定candidate与设备风险，2026-10-17

J-F1–J-F5产品实现已停止变化；本轮普通消费者、真实退休、三路线host风险与有限P0账已补证。
四owner进入上述CANDIDATE，不再写产品代码；交付、P1与终审仍待验收，状态不等于CLOSED。

[具名设备工具原件](../../../../evidence/correctness/inference-g09-native-device-validation/receipt.json)：
Windows原contracts与5场景流水确实加载Khronos instance/device validation layer，并识别
sync validation配置，VUID/同步hazard为0；Air原contracts与5场景流水实际打印Metal API/GPU
Validation Enabled且错误为0；WSL原contracts与4场景CUDA流水在memcheck/racecheck均为
0 errors/0 hazards，原checked数学/取消/退休检查均PASS。三组活动累计188.869秒，无新编译。
Windows保留陈旧OBS registry JSON及配置弃用提示，未改用户注册表；这些不是产品validation错误。
设备结论限于具名平台和变动owner的提交/退休旅程，未变叶数学/barrier资格沿原身份BORROW。
不将WSL或Windows的一份工具记录改写成所有平台工具已跑，也不签私有driver/全进程泄漏。

[P1具体消费草案](../../../../evidence/architecture/inference-g09-p1-prospective/proposal.json)
复用已固定功能consumer的4M F32维度，补齐两插入顺序、两microbatch与必要串行holdout的阶段/迭代账；
已通过独立[prospective设计审查](../../../../evidence/architecture/inference-g09-p1-prospective/review-11.json)，
截至该次设计审查尚未构造benchmark或采样；实际SDK、私有符号/loader隔离、完成账与原仪器资格五组预检均待证。
两侧persistent/cache/DSO最终销毁同列DEFERRED；逐replay结果owner与accepted work仍在warm内退休。
baseline所对应的原G08静态SDK产品
src/include/cmake/root CMake与9e2f25逐字节相同；实际baseline/candidate SDK仍须逐项hash绑定，
不能只改revision标签或混链不同private Graph layout。

下一步限于本卡剩余验收：三交付/公开头；按预声明两自然形状与必要串行holdout，
先完成prospective源/阶段审查和独立artifact/text预检，再正式P1，最后终审/LOC。
尚未完成的门均待证；不追加缺口穷尽调研，不提前推进G10，不重跑已接受G01–G08数学/P1。

### 交付与实际P1消费者源码检查点，2026-10-17

[Air交付原件](../../../../evidence/delivery/inference-g09-air-sdk-checkpoint/receipt.json)中，
Apple/Vulkan各自source/static/shared共6项均PASS；每项实际运行原contracts、两slot/65步流水、
原G06 pending消费者，独立编译两个公开Graph头。SDK先安装到staging再迁至relocated，
新consumer只使用installed头；另用实际dyld记录核对Hyle加载路径和hash。累计191.010秒。
首次Vulkan static验证helper误用embedded路径，在build前失败；同树改正布局后续跑，
原失败与费用保留，152份shader DATA逐项绑定前一G09资格，未借旧Graph archive代签新交付。

[Windows交付原件](../../../../evidence/delivery/inference-g09-windows-sdk-checkpoint/receipt.json)中，
CUDA/Vulkan各自source/static/shared共6项均PASS；两路线的原contracts、公开流水及两个独立
Graph头通过，Vulkan另验原G06 pending消费者。实际重编当前common/hybrid C++ owner，
CUDA原146个native对象按原命令、实际依赖和输入身份复用，无CUDA算子重编。
迁址SDK使用installed-only头，PE imports、staged DLL hash与隔离PATH均检查；这不是actual-loader tracing。
六项累计1403.204秒，含原失败和全部同树复验，不重置费用。

Windows原清单误选两个明确要求Linux FlashInfer的G06消费者，首次消费者编译失败保留。
原provider CMake在Windows为OPTIONAL_ABSENT且不导出其split头，此两项语义上不适用Windows；
WSL仍必须运行，不将它们改记Windows PASS。source consumer父scope遗漏CUDAToolkit查找也保留为
验证CMake失败，已定向修正，产品CMake未改。Vulkan首次预检拒绝外层generator的build/install stamp；
随后确认其原inner Ninja确实no work、外层无shader DATA生成，并核152份数据在原build前后hash不变，
沿原manifest完成交付，没有使用phony archive/data替换。WSL三交付此时仍在运行，尚不记PASS。

[实际P1消费者源码检查](../../../../evidence/architecture/inference-g09-p1-consumer-source/receipt.json)
已经完成一次有限独立review：四个冻结形状、默认原CPU Context/provider、2个原Sched workers、
双submit先于任一wait、lease内4B读取、前后独立完整run oracle、owner释放与计时内drain符合合同。
唯一源码MUST是SHARED facade误调executable-only stager；已按review接受的最小修法，给同目录
未执行的private probe EXEC调用原公开stager，facade依赖其资源stage，driver仍只链接原nanobench。
不修改公共stager、数学或产品API；实际依赖/MLX资源位置仍须产物预检，源码修改不预签该门。

原instrument用`epochIterations(1)`固定每epoch一次完整逻辑replay；`epochs(11)`按原ABBA规则
实际补成12 paired rounds，草案明确记录请求11/实际12，未改仪器或按计时调迭代数。
prepared copy字段明确仅slot0的per-invocation计划，不冒充两slot合计或实际dispatch；
最终persistent状态与DSO unload两侧同为DEFERRED，单独诊断计时，private vendor cache仍UNKNOWN。
此时只有源码草案，没有benchmark构建、短diagnostic、A/A或正式P1结果。

剩余仍限本卡：完成WSL交付；实际SDK/object、私有weak/type/data/loader/原生注册与资源身份、
数学/dispatch/copy/退休账和原workload A/A等五组预检；正式P1；终审/LOC与聚合裁决。
G09保持固定CANDIDATE，G10未开始，不因此追加功能缺口扫描或重开已接受数学。

### 选定交付矩阵完成，2026-10-17

[WSL CUDA交付原件](../../../../evidence/delivery/inference-g09-wsl-sdk-checkpoint/receipt.json)
现补齐source/static/shared三项：每配置实际通过原Graph contracts、两slot/65步流水、
G06 hybrid paged Graph、G06 pending receiver，以及两个独立公开Graph头。Linux FlashInfer消费者
实际在此编译运行；没有把Windows不适用项算成PASS。原native CUDA对象仍按输入/原命令/依赖
资格复用，无CUDA算子重编；当前common/hybrid C++与消费者实际增量构建。source parent已恢复，
SDK迁址/installed-only头与ldd依赖检查通过。这里不是native Linux P1资格。

本轮选定的五个真实native profile共15项source/static/shared交付均已通过；未变公开header/export/
package seam和叶资格的复用边界不变，不改写为全库全部配置重跑。产品依然为固定candidate。
已开始Windows CUDA baseline私有P1 facade的普通构建；此时没有benchmark执行、A/A或正式样本。
后续仍按五组实际产物/同合同/仪器预检、正式P1、终审与LOC收口，不增加产品FIX或G10范围。

Windows CUDA baseline两份private facade随后已按原静态SDK完成真实编译链接，没有HYLE产品/native
重编；初始产物预检拒绝了额外的caffe2/c10 vendor自动导出符号，严格C-getter-only assert的原FAIL保留。
尚未构建candidate、加载/执行benchmark或采样。需有限独立复核其原SDK directive、PE import归属与
private HYLE隔离边界；不能只凭源码隐藏声明或额外符号属于vendor就改记PASS，也不因此改shipping源码。

### 实际P1产物与CUDA不计时检查点，2026-10-17

[实际产物与不计时检查原件](../../../../evidence/architecture/inference-g09-p1-artifact-checkpoint/receipt.json)
现记录Windows CUDA、Air Apple/Vulkan三组profile的baseline/candidate各两份private facade，
共六套、十二个真实映像。全部仅编译公开消费者并链接既有静态SDK；未改产品、数学、CUDA算子或
公共性能仪器。Air逐份初始导出检查均仅有C getter，但尚未签实际Mach-O绑定、加载和运行。

独立R13确认Windows CUDA额外77项原caffe2/c10导出逐字来自同一原ATen对象/directive，
两侧原archive/object相同；冻结精确名单而非vendor通配，HYLE私有import/export/跨映像绑定拒绝门不变。
初始严格C-only FAIL、两次inspection helper预检失败和累计费用均保留。
R14已核四份实际PE映像/关键archive/273份同一公开头、有限反汇编与text归属；只允许下一步diagnostic，
不签A/A/P1。两侧39份实际staged runtime DLL逐文件hash相同。

首个verify-only已通过四场景前后完整4M输出oracle和八次State销毁，但最终显式FreeLibrary挂起。
WCT观察主线程等待另一线程，cycle=false；原baseline/candidate相同的CUDA静态CompletionReaper
析构会join线程，与DLL-detach loader-lock风险一致，尚未由完整native栈确认具体环。
原HANG、进程身份、WCT与费用保留；没有把它记为G09新产品缺陷或性能结果。

按独立R15有限建议，只为Windows另编driver，让两侧private映像对称保留至process exit；
原driver、四份facade、SDK未覆盖，所有warm数学/prepare/submit/原生完成、结果观察、drain和State销毁
均未改变。新driver的两种加载/分配顺序均通过完整oracle、全部State销毁及正常进程退出，
实际81/82个模块的路径/hash已核对，原CPU program实际加载。最终映像/私有cache回收仍不签dynamic-unload。
以上六套构造、冻结、原失败、driver纠正和两顺序不计时检查的已记录case费用合计387.998秒，
不含编排/源码准备/审查时间；没有重置原预算或失败。

尚无Bench::compare、workload A/A或正式P1样本。剩余Windows Vulkan产物、Air实际绑定/运行、
原A1–A5与同拓扑A/A/仪器控制、四组profile正式P1和终审/LOC依旧待证。
G09保持固定CANDIDATE，G07–G10 goal仍为已接受2/4，G10未开始；已通过15项交付不重跑。

### 四组实际运行补齐与首批A/A，2026-10-17

[四组运行及CUDA A/A原件](../../../../evidence/architecture/inference-g09-p1-native-preflight/receipt.json)
现补齐Windows Vulkan baseline/candidate两套消费者，全部八套、十六份private facade均已真实构造。
独立R16签Air八映像的内部HYLE符号/无外部weak或fixup绑定、TWOLEVEL、关键archives、有限反汇编
与text；R17签Windows Vulkan四PE、精确77项原ATen导出及BigObj/directive、private binding与text。
这些静态审查分别只准入后续diagnostic，没有代签A/A/P1。

Air Apple/Vulkan和Windows Vulkan均实际通过两种加载/分配顺序：四场景前后完整4M oracle、
每进程八次State销毁、原显式dlclose/FreeLibrary返回及正常进程退出，实际模块路径/hash保留。
结合前一检查点Windows CUDA的对称process-exit映像寿命，四组native profile的基本运行检查已齐。
Windows Vulkan首个运行实际成功，随后helper错误期待`nvvulkan.dll`而FAIL；仅纠正为实际
`vulkan-1.dll`并复测本diagnostic，原失败及费用保留，未改任何compiled artifact或产品。

原hyle-perf只启动无参数可执行，因此新增的private argv封装仅向原main传四个既有facade路径，
复制原nanobench/perf_process/common-control源码，未改公共工具或热路径。原Windows CPU C-ABI
program和Air `mlx.metallib`按原查找合同放到消费者旁；相同两侧native文件身份已有冻结。
Windows使用原`native-runtime-pair`：两个真实physical cores与caller singleton，原Scheduler2
仍由runtime管理，不宣称worker singleton/count资格；Air按原profile使用UI QoS，不制造硬pin。
极短code-padding/link-order风险对固定bulk4M数学/搬运旅程不适用的理由，已在任何A/A数据前固定；
地址/owner风险仍由完整allocation/load order覆盖。没有为适配仪器改产品API或Sched默认行为。

Windows CUDA具名probe binary的完整workload A/A现已运行通过：baseline/candidate各四种
execution×allocation/load order，共八进程×四负载，原5%factor medians和10%gross-stop全部PASS；
各进程速度比范围0.968899–1.030347。原12 paired rounds/exact iteration1及4M输入不变。
原caller/parent-mask/priority canary通过；私有源冻结为Git `c6e7de71`，是benchmark source身份，
不替代产品candidate `3524413272`/实现`e09f30925`。正式新build不自动继承probe binary的A/A资格。

本检查点新case累计85.030秒，先前387.998秒及原失败保持；不含源码/编排/审查与Git准备。
两机原baseline `9e2f25c4` 的只读用途稀疏worktree已准备，未触发产品构建。
尚无新旧版本A/B正式样本：原仪器有限contract/provenance、完整dispatch/copy账、最终产物预检、
其余实际A/A与common controls、四组5%/n15–31 P1和终审/LOC仍待证。产品仍固定CANDIDATE，
goal已接受2/4、G10未开始；不重跑15项交付或追加产品FIX/缺口扫描。

### 正式仪器合同与实际入口冻结，2026-10-17

四组native profile现已各自接入未改动的原`hyle-perf`：固定四负载、四种execution×allocation/load
顺序、原n15→31/5%和原A/A、delta5、delta10、timer-floor。四份contract通过原schema/runner
校验；code-padding/link-order不适用的理由仍沿首批A/A前的冻结声明，不按样本改合同。
其余三个无参数入口各仅编译wrapper、原nanobench和原control三个C++单元；CUDA沿原具名probe
重读实际text/imports。没有重编产品、SDK、数学或native kernel。

独立R18检查实际四组primary/control的text、imports/dependencies、commands和有限反汇编，
并冻结22份输入。唯一MUST是fresh产物不仅须匹配text，还须检查`G09_BASE`确为冻结的facade根，
防止等长路径替换只改只读数据而绕过text身份；现已仅在private validator补上此检查。
正式fresh build必须重核原16份facade及所需native资源、具名text和源码，再通过自身完整verify及
两SDK各四顺序的workload A/A才允许A/B，不能继承旧probe的A/A。原公共统计/构建工具未修改。

CUDA额外一次CUPTI补账保持独立的`DIAGNOSTIC_HANG`：四场景完整前后oracle及八次State destroy
均完成，实收100条kernel记录，其中99条具有效GPU时间区间；164条16MiB memcpy均有有效区间，
HtoD合计1GiB、DtoH合计1.5625GiB，105个buffer全回收，reported drop/errors均为0。
一条kernel时间戳无效、90条kernel没有所选direct-launch API关联，逐API/逐replay完整归属仍UNKNOWN。
进程最后停在vendor teardown的PrimaryCtxRelease ENTER，未观察到EXIT；核对原exe/PID身份后
仅终止该owned child。原setup失败、挂起、96.452秒case费用及全部限制保留，不签正常退出或性能PASS，
也不推翻此前各自身份下未注入诊断的正常退出结果；不再为该可选逐API细节增加GPU运行。

原正式流程现先在Windows CUDA运行，随后按Windows Vulkan、Air Apple、Air Vulkan串行推进。
此节只记录合同/产物入口准入，不预签尚未终态的控制、统计性能、G09终审或G10。

### 四组正式入口终态与本轮停止，2026-10-17

[正式尝试及原件](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/receipt.json)
和[独立 R19](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/19.json)保存四组唯一尝试。
所有产品、16份facade、原统计核心与冻结形状未改；没有盲重复或删除失败样本。

| profile | workload A/A | 正式资格/性能 | 本次case秒 |
|---|---|---|---:|
| Windows CUDA | PASS，正式fresh binary独立8进程 | QUALIFIED；原n15 P1工程门PASS | 522.156 |
| Windows Vulkan | FAIL，candidate CPU-first 的peer-allocation因子比0.94695274低于1/1.05 | DIAGNOSTIC，NI未决/方向INCONCLUSIVE；有效A/B为0 | 32.656 |
| Air Apple | FAIL，candidate双microbatch一进程比0.89615083低于1/1.10 | DIAGNOSTIC，NI未决/方向INCONCLUSIVE；有效A/B为0 | 14.724 |
| Air Vulkan | FAIL，baseline/candidate多项gross与factor；全进程比范围0.853571–1.259887 | DIAGNOSTIC，NI未决/方向INCONCLUSIVE；有效A/B为0 | 26.921 |

CUDA四个execution×allocation/load strata各15个独立进程，四个common control与三个validator均PASS，
source/peer/artifact/host稳定性均为真，原`hyle-perf verify`已核manifest、全部路径及资格。
独立CPU-first/GPU-first两行均为WIN：各stratum中位candidate/peer速度比分别为1.1366–1.1455、
1.1168–1.1248；双microbatch和serial holdout为原5%合同PARITY，全部NI门PASS，不追加n31。
这里比较的是本卡完整warm消费者与旧HYLE，不是llama.cpp或推理token/s。

另三组完成各8个A/A进程，但未获准进入产品比较；ready guard在首个A/B入口的compare前拒绝。
R19独立重算24份日志与原op/s判据一致，完整前后oracle/8次State destroy也逐份成立。
paired relative/CI与两行各自median elapsed倒数之比是不同估计量，不是列解析错误，不能换度量消除失败。
当前没有证实可修产品回归或具名工具/宿主机制；这些A/A偏差不写成产品LOSS，更不写PASS。

依收口方法§6/§7，本卡在资格拒绝处停止，四owner记`BLOCKED-from-CANDIDATE`，聚合
engineering_performance为`INCONCLUSIVE`；阶段功能/风险/P0、15项交付和CUDA合格P1不作废。
最终聚合red-team、完整P1及CLOSED仍未完成。阻塞owner和唯一合法恢复条件由
[独立工具卡](../tooling/inference-g09-native-workload-aa.md)固定：先证明具体instrument/host/contract
机制并独立审查修正，在新测量前冻结入口，再取得原阈值/四顺序下缺失的资格和P1。
追加epochs/warmups/n、放宽带宽、换尺寸/Provider/估计量、丢样本或同配置再试均不是恢复条件。

固定candidate上的[统一LOC原件](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/loc.json)
为全库3,832,220、自实现不含测试183,364、本轮Graph与三个hybrid实现边界19,704行；
由原`loc-report.py --commit 3524413272`及显式五个路径给出，不计scratch/benchmark为产品实现。
[本轮selected验证](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/validation.json)
仅docs/closure，146项skip/reuse；带原registry revision的64组件/schema4检查、文档链接和diff检查通过。
初次生成状态投影过期的FAIL及纠正记录保留；没有触发全局构建。

本goal保持active；[G10](../compute/inference-g10-cpu-thread-budget.md)的当前阶段和验收由自身卡与
registry记录，不以它代签G09。以上为原正式尝试停止点，后续具名工具/宿主验证单列如下。

### OBS 宿主条件排除，2026-10-17

[独立工具任务](../tooling/inference-g09-native-workload-aa.md)经 R21/R22 冻结一次 Windows Vulkan
宿主排除，未改产品、SDK、原 C++ 消费者、统计核心或 5%/10% 门。
[完整终态及 R23](../../../../evidence/architecture/inference-g09-obs-host-attempt/receipt.json)
记录 verify 与原八组 A/A 的 36 个快照均无 OBS，完整 oracle/退休及正常退出通过；
但同实现比仍为 0.886063–1.192072，原 factor/gross 资格失败。排除 OBS 不足以解决资格，
不能写为产品 LOSS 或声称历史原因已找到。ready guard 拒绝首个产品 compare，有效 A/B 仍为 0。
新增 case 34.843 秒/无产品重构建，保留旧失败和费用；本次不再运行。
三组缺失的 A/A/P1 和 G09 四 owner 的 BLOCKED-from-CANDIDATE 不变；G07/G08/G10 已独立验收。
