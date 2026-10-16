# G09 单CPU与单GPU联合流水

状态：**AUDITED，2026-10-17；有限FIX及依赖/推进形状已冻结，进入批量实现；新实现尚未验收**。
baseline为`9e2f25c469634a6f723e76eaf1aa2db5973bca9a`。
G07/G08已接受，当前G07–G10 goal完成2/4；G10和完整文本CLI未开始。

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
均为AUDITED。common/CUDA只纳入原cold segment边界与联合提交推进，不扩大为普通Graph全面重构。
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
warm只做已准备的依赖计数/就绪队列/短发布，原copy、必要wait、native私有费用仍单独记账。
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

仍为AUDITED。J-F1–J-F4实现已接齐，三native路线首轮实际消费者已运行；尚未固定最终candidate，
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

下一步限于本卡剩余验收：原G02/G03/G06直接消费、适用host/device风险、
三交付/公开头、P0完整账；candidate固定后按预声明
两自然形状与必要串行holdout，先独立artifact/text预检，再正式P1，最后终审/LOC。
尚未完成的门均待证；不追加缺口穷尽调研，不提前推进G10，不重跑已接受G01–G08数学/P1。
