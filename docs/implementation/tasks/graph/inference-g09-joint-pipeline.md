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

本轮已冻结[实现前独立设计复核](../../../../evidence/architecture/inference-g09-cuda-preaudit/design-review.json)，
开始J-F1–J-F5批量实现，仍为AUDITED而非candidate。

- J-F1首版：原native analysis新增线性依赖分片投影，Apple/Vulkan接入真实data/control prerequisites；
  common segment按新增外部依赖分开独立prefix，保原lane/预算复用边。
- J-F2首版：新增私有冷依赖表/有界frontier，Apple/Vulkan使用原root/native driver和一个CPU child，
  严格串行保inline；每个child仍用原Large stack pool，未增加OS线程或每segment fiber。
  common原cursor已抽出逐segment local outcome/串行commit接缝，原advance复用该接缝；
  **CUDA联合双lane调用者尚未接入**，不能据此宣称CUDA流水已完成。
- J-F3未完成：当前native首版仍有旧的全输入边界准备阶段和producer端搬运；
  按真实consumer的准备/transfer依赖推进仍须在本批完成，不从范围移除。
- J-F4首版补短prefix-protect锁、取消的monitor wake、两driver汇合；普通borrowed并发图也冷建原
  LeaseSubmission以保全部prefix及RunState。真实故障/取消/放弃后退休测试尚待执行。

[本次common局部检查](../../../../evidence/architecture/inference-g09-partial-common-check/receipt.json)：
Windows原source integration tree中初始75项、追加frontier后79项全部PASS，含原Graph/core/effect/
lease/custom consumer与新增两顺序分片、join、control、反复frontier、quota。原目标增量构建13.078秒，
新增frontier对象1.672秒，CUDA对象0；两次既有profile configure/manifest费用163.062/52.984秒保留。
这是改变的common源码与具体测试的局部证据，**没有编译/验证Apple/Vulkan新driver，也没有签G09聚合门**。
实际common/test源码在运行后逐项hash复核，native实现同期继续修改的事实在receipt中明确记录。

下一步继续同一有限批次的CUDA caller与J-F3，集中编译三joint owner、真实两slots/65步/硬件时间线、
错误与实际退休、适用风险/交付，然后固定candidate做预声明P1。当前完整native流水、三交付、
风险与性能门均待证；不得因79项局部测试通过而提前收口。
