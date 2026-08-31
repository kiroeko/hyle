# sched-v2 批级 structured work 恢复轮

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只恢复 `sched-v1` 红队证明缺失的批级 non-suspending structured route，并用新版 comparison-class
> 规则重签受影响性能能力。旧 [`sched-v1`](sched-v1-rebuild.md) 永久保持 `HISTORICAL/BLOCKED`；其
> correctness、交付和未受影响的 request/completion 证据可以作为审计输入，不能自动继承为本轮 verdict。
> 当前机械状态只见 [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

- component / owner / round ID：`sched` / scheduling-runtime owner / `sched-v2-batch-structured`；
- task kind：`RESUME`；
- previous round / blocker / resume condition：`sched-v1` / `SCHED-RT-01`。普通批级 non-suspending
  fork/join 没有自然、安全且成本匹配的 public route，旧 structured pair 又把 HYLE 逐任务
  result/status/cancel/JoinHandle 与 Marl 批级 WaitGroup 错写成同合同；恢复必须先冻结四档语义账单，
  实现 batch route，并按 comparison class 分开验证；
- baseline commit：`1908f7b9bf8c86c1da9c11aa63e767577d0fa602`；未变的 v1 产品身份为
  `f358d084d17f92e64ec51ae2bd974d2a67f417d2` / tree
  `44a0d4e0de048bf93d53ec0a1febdfe6135ac6a6`；
- 进入原因与 demand ID：`SCHED-V2-BATCH-01`，来源为当前 Graph level fan-out、量化推理 tile/kernels 的
  批级并行任务、Marl `enqueue + WaitGroup` 与 oneTBB `task_group/parallel_for` 的共同成熟形状，以及
  sched-v1 qualified natural-route material risk；
- 使用原型 provenance：当前 Graph 语法 consumer + 北极星推理 journey + Marl/oneTBB 共同任务；在
  HYLE 推理引擎完成前，吞吐 replay 仍标为 `PROVISIONAL_ARCHETYPE`，不冒充生产 trace；
- 真实任务 / normal workload：一个已经进入 Scheduler 的 root 在四个 worker 上执行一批短小、
  `noexcept`、不会挂起/阻塞的 indexed 或 caller-context work；整批共享一次 completion、可选取消与
  admission/error 状态，结果写入 caller-owned stable slots；root 只等待整批完成，不观察、detach、
  abort 或 join 某个单项；normal primary 为 16 批 × 128 work items × 1024-element deterministic tile，
  与当前 Graph automatic grain floor 相同；最多保留一个不同粒度 holdout 诊断过拟合；
- API 语义档位与使用前提：
  1. fire-and-forget：`schedule`，无 result/等待/取消 owner；
  2. batch structured：本轮能力，一次 batch completion，共享 failure/cancel，caller-owned results，
     所有 work 在返回/显式 wait 前 quiescent；
  3. per-task structured：`schedule_task`，每项独立 typed result/status/cancel/JoinHandle；
  4. suspending structured：`spawn_task`，coroutine/Fiber 可以 yield/sleep/await；
  只有上述 result 粒度、ownership、control 或 suspension 合同确实不同时才保留独立入口，不建立只按
  快慢命名的 `light/heavy` API；
- 范围：batch structured 的 public C++ shape、ownership/failure/cancel/destruction、worker/external
  wait、实现与资源成本；相邻四档的选择说明；现有 Sched component/package 和受影响直接 consumer；
  一个 same-contract primary replay、一个 HYLE 内部成本阶梯与必要机制归因；
- 非目标：关闭或重构 Graph/Compute/IO；复制 oneTBB algorithms/partitioner 全集；DAG、模型、kernel、
  NUMA/affinity、priority、caller-participating SPMD 或 grain policy；逐项 future/status；修改
  `schedule_task`/`spawn_task` 未受影响语义；C/Native FFI、稳定跨工具链 C++ ABI、P2 端到端或 Formal；
- 直接依赖 / 消费者：保持 Sched 当前 `PUBLIC core,error,time,platform,mem,util_core` 与
  `PRIVATE container`；当前 Graph 只提供语法与任务 provenance，IO/Compute 只做回归；不新增第三方
  product dependency；
- tier 与理由：`P1`。批级 dispatch/completion 是推理 graph/tile 的潜在高频 runtime primitive，但尚无
  完成的 HYLE engine profile 证明 P2 贡献；
- material margin / resource budget：W1-WIN 默认 5% same-contract non-inferiority；batch route 为
  O(N) work、O(1) terminal notification，不允许 per-item heap allocation、per-item independent status/
  cancel/handle、polling、syscall 或 context switch；允许 O(1) 或 O(worker-count) batch metadata/queue
  runners，actual allocation/bytes/queue entries/atomic/wake/context-switch 必须报告；
- 总调查预算 / 性能预算：设计与审计 60 分钟，产品批次最多 150 分钟，correctness/三平台/三交付
  120 分钟，P1 资格与采样 90 分钟，红队 30 分钟；总计不超过一个工作日；
- 最多复现、机制诊断、产品修改次数：正式 P1 只允许一轮资格化；material 异常最多一次反序/复现、
  一次 allocation/queue/wake/codegen 诊断、一个机制驱动修正和一次重跑；不修改中央统计语义；
- machine profile：`W1-WIN` P1；`W1-WSL` GCC/sanitizer/ELF；`M1-MAC` AppleClang/AArch64/Mach-O；
- instrument profile：P1 `closure-perf-v1`，correctness/delivery `closure-correctness-v1`；
- strong peers：Marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e` 的 Scheduler/WaitGroup；oneTBB
  `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` 的 task_group/parallel_for；两者本轮开始时 local checkout
  clean；
- 无同合同 peer 时：batch route 仍须通过预冻结的 per-item/per-batch resource contract 和绝对 consumer
  throughput；逐任务 rich route 若没有等价 adapter/peer，只报告 HYLE 内部语义税，不产生 peer/SOTA
  claim；
- comparison plan：
  - batch primary：`SAME_CONTRACT`，两侧都是 caller-owned result slots、一个 batch completion、无逐项
    control/suspension，并逐格签 semantic-equivalence matrix；
  - 当前 v1 rich-HYLE vs batch-Marl 行：`NATURAL_ROUTE/NOT_COMPARABLE` historical input；
  - allocation/queue/group completion 分解：`MECHANISM_DIAGNOSTIC`，只归因；
- 适用 gates：normal/zero/one/capacity/overflow、invalid/OOM/admission partial failure、move/destroy/
  early-return、worker/external wait、concurrency/reentrancy、cancel/shutdown/stress、ASan/UBSan/TSan、
  public headers、IO/Graph/Compute compatibility slice、source/static/shared selected component、C/FFI
  cleanliness、API/architecture/user/performance docs、P1 与独立红队；
- 预期输出：capability/public trace、有限 FIX、公开选择表、单一 product candidate、三平台与三交付、
  comparison matrices、qualified P1、资源账单、六问、限制/reopen triggers、terminal receipt 与四项 LOC；
- 停止条件 / 可接受限制：所有本轮 `MUST` 为 `COMPLETE/ROUTED` 且全部适用门通过才 `CLOSED`；无法
  提供安全自然 batch route、资源合同被违反、same-contract P1 非劣失败或必要平台/交付无法裁决则具名
  `BLOCKED`。Graph policy、逐项 rich-route peer claim 与 Formal 可保持明确负空间。

`BLOCKED -> SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v2\module-status-before-scoped.json`，SHA-256
`9880363F9F0D5F2E7B6D9DB8EEC9ACF1A0AA63A076C9E5E685BC57DEA1E83EB1`。

## Capability-family coverage map

进入 `AUDITED` 前完成。当前只允许只读 peer/consumer/public-surface 审计，不修改产品。

## Capability / gap ledger

进入 `AUDITED` 前冻结有限 `FIX` 集。

## Owner 六问直答

进入 `AUDITED` 与 terminal 时分别填写。

## Candidate

尚未产生。

## Verification manifest

尚未进入 candidate；未运行项不得写 `PASS`。

## 独立红队

等待冻结 candidate 与全部验证后执行，先攻击 comparison class 和 semantic-equivalence matrix。

## Closure decision

当前建议机械状态：`SCOPED`。下一步只做有界双向能力审计和 peer mechanism review；冻结
capability/`FIX` ledger 并进入 `AUDITED` 前不得修改 Sched 产品。
