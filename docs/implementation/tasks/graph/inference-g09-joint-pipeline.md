# G09 单CPU与单GPU联合流水

状态：**SCOPED，2026-10-17；源码/peer审计及CUDA反证完成，依赖与推进设计待冻结，尚未授权产品实现**。
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
均为SCOPED。common/CUDA只纳入原cold segment边界与联合提交推进，不扩大为普通Graph全面重构。
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
下一步只冻结这些FIX的owner/API/ownership/失败/并发/资源账后进入AUDITED，不再扩大缺口扫描。

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
独立阶段终审、明确重开条件与统一LOC。当前新实现的运行时/性能门均为**未运行**，不能记PASS；
上述旧SDK反证不代签这些门。
