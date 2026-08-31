# sched-v2 批级 structured work 恢复轮

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只恢复 `sched-v1` 红队证明缺失的批级 non-suspending structured route，并用新版 comparison-class
> 规则重签受影响性能能力。旧 [`sched-v1`](sched-v1-rebuild.md) 永久保持 `HISTORICAL/BLOCKED`；其
> correctness、交付和未受影响的 request/completion 证据可以作为审计输入，不能自动继承为本轮 verdict。
> 当前机械状态只见 [`module-status.json`](../../module-status.json)。

## Closure card（`AUDITED`）

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
  首个候选已经完成一次资格化及合同内 `n=15 -> n=31` 扩样，结果为 `INCONCLUSIVE`；反序由合同的
  双顺序 strata 同轮完成，唯一 codegen 诊断已经消费。当前只剩一个机制驱动修正和一次正式重跑预算；
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

首个候选因性能未裁决而从 `CANDIDATE -> AUDITED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v2\module-status-before-audited-fix06.json`，SHA-256
`9AFB680D40FBABA7CAE5DC5BBDD0E8FC3F7533DDA48AABD633BDBA346577B34F`。

`FIX-06` candidate 因 TSan correctness finding 从 `CANDIDATE -> AUDITED` 的前态 registry 快照保存在
canonical checkout 外：`C:\hyle-scratch\sched-v2\module-status-before-audited-fix07.json`，SHA-256
`B006E41D87B386F4DE409C0BC279735D6DBE65C56660510C8F3A05E16C852A97`。

## 审计输入与设计裁决

本轮只读审计绑定以下 clean identity：

- HYLE baseline：`1908f7b9bf8c86c1da9c11aa63e767577d0fa602`；
- llama.cpp：2026-08-31 fast-forward 后的
  `662a0b0121a53c23b825a71e64ab6eff59b7f4d8`；其 CPU graph 继续把一次 graph/kernel 工作作为共享
  threadpool episode，由 `ith/nth`、barrier、abort callback 与共享 chunk cursor 协作，不为每个 chunk
  建独立 future；
- standalone ggml：`36da57138425487184aa1da2eee2cde155909c6f`；同样保留 threadpool barrier、
  `ggml_threadpool_chunk_add` 动态分块与一次 graph completion；
- Marl：`b8406ab0a82524170ead6dd6c371b579bbc28b3e`；`schedule(Task)` + 一个共享
  `WaitGroup` 是 batch completion，Task/结果 slot 不带逐项 JoinHandle；
- oneTBB：`a8216b95202f3c55f9749870f4f8cfa4df0b32e3`；`task_group` 提供 group wait/cancel，
  `parallel_for` 提供同步 range completion 与 caller-owned 数据，但 task_group 的普通 `run(F)` 仍会为
  每项准备 task storage；
- 当前 HYLE Graph 只作语法 consumer：`graph::parallel_for` 的旧实现用递归 coroutine、`spawn_task`、
  Fiber stack 与 JoinState 完成每次 split；它证明 range 任务存在，也证明该机制不能冒充 sched 的轻量
  batch route。Graph 当前为 `QUEUED`，本轮不迁移或冻结其产品。

冻结的 public shape 名为 `sched::bulk_execute`，而不是 `light_schedule`/`heavy_schedule`：

```cpp
struct BulkExecuteOptions {
  std::size_t grain_size = 1;
  unsigned max_concurrency = 0;       // 0 = Scheduler worker 数
  const CancelToken* cancel_token = nullptr;
};

using BulkBodyFn =
    Status (*)(void*, std::size_t begin, std::size_t end) noexcept;

Status bulk_execute(Scheduler&, std::size_t begin, std::size_t end,
                    BulkBodyFn, void*, BulkExecuteOptions = {}) noexcept;
// 同头提供 noexcept lambda adapter；body 可返回 void 或 Status。
```

它是 scheduler 级固定 grain bulk primitive，不拥有自适应 partitioner、reduce/scan/sort、tensor 或 graph
policy。正常 parallel 路线借用 callable/context、cancel token 与 caller-owned result slots直到函数返回；
只提交 `min(chunks, max_concurrency/workers)` 个共享 batch-state runner。`FIX-06` 后 runner 从一个 atomic
cursor 有界预领取内部 reservation，目标为每个 runner 八次 reservation；正常范围使用 `fetch_add`，只有
接近 `SIZE_MAX`、终止领取可能回绕时使用饱和 CAS。每个 reservation 内仍逐 public grain 调 body，并在
每次调用前观察 stop/cancel。调用者在普通线程等待时阻塞线程，在 Scheduler Fiber 等待时挂起 Fiber；
从 worker-stack non-suspending callback 嵌套调用时为避免单 worker 自死锁，明确退化为当前 callback 上的
串行执行。

body 的第一个 non-OK `Status` 让 runner 在下一次 body 调用前停止；已经在执行的 body 仍完成，内部已
reservation 但未开始的 grain 可以放弃。多个并发错误的 winner 不保证 index 顺序。取消同样只阻止后续
body 调用，并在没有 body error 时返回 `Cancelled`。提交部分 runner 后发生 queue pressure 不丢工作：
已经 admission 的任一 runner 都能领取剩余全部 chunk；一个 runner 都未 admission 才返回 admission
error。空 range 为成功；反向 range、零 grain、空 raw callback 或无效 Scheduler 为可恢复错误。

拒绝的替代方案：

- 不给所有 `InlineJob` 增加可选 group/completion 指针；那会让无关 fire-and-forget 常态 queue item 为
  本轮能力永久增大；
- 不用 `schedule_task`/`spawn_task` 包装每个 item；那会重新缴纳逐项 result/cancel/JoinState 或 Fiber；
- 不先增加一个每项 heap node 的 `TaskGroup::run`；它虽接近 oneTBB shape，却违反本轮已冻结的
  per-batch/O(worker) 资源合同，动态异构 addition 继续由 `schedule + WaitGroup` 组合承载；
- 不把 Graph 的 recursive split、auto grain、reduce/scan/sort 下沉到 Sched；它们属于算法/consumer
  policy，未来 Graph 审核可把本原语作为底座。

## Capability-family coverage map

| family | admitted journeys / semantic variants | lifecycle / boundary | current route | coverage | missing edge |
|---|---|---|---|---|---|
| 短 fire-and-forget | runtime callback、无结果/等待/取消 owner | raw `fn+ctx` 零分配；capturing callable 一次 record | `Scheduler::schedule` | `COMPLETE` | 无；保持独立 |
| 有限 homogeneous/indexed batch | Graph level、量化 tile、ggml chunk episode；整批 completion/首错/可选取消 | caller-owned slots；同步 quiescence；不能挂起 | 尚无自然入口 | `MISSING` | `SCHED-V2-BATCH-01` |
| 有限 caller-context/heterogeneous batch | 一组稳定 work records，整批等待 | 以 index body dispatch caller-owned records | 新 bulk route 可完整表达 | `ROUTED` | 文档给出数组 dispatch 形状 |
| 动态异构 group | work 在执行期继续 addition；group completion | caller 平衡 add/done/admission；允许每项 owning callable 成本 | `schedule + WaitGroup` | `ROUTED` | 不满足 bulk 的 O(worker) 成本，不冒充 same-contract peer |
| 逐项 typed structured | 每项独立 result/join/abort/status | 每项 JoinHandle/cell，non-suspending worker stack | `schedule_task` | `COMPLETE` | 无；不得用于 batch peer 行 |
| 可挂起 structured/detached | timer/IO/channel await，独立 Fiber lifetime | coroutine/Fiber/stack；可 join 或 detached | `spawn_task` / `spawn_detached` | `COMPLETE` | 无；保持重语义 |
| blocking offload | OS/blocking API 或不可合作长任务 | 独立 OS pool，typed completion | `BlockingPool` / `spawn_blocking` | `COMPLETE` | 无 |
| data-parallel algorithms | adaptive split、reduce/scan/sort、kernel grain policy | Graph/算法 owner；Sched 只提供 execution primitive | 当前 Graph 旧实现 | `OUT_OF_SCOPE` | Graph 后续轮决定迁移 |

## Public-surface trace

- `scheduler.h`：`schedule`、`schedule_task` 的底层 worker-stack admission，`spawn*` 的 Fiber admission，
  worker 数与 lifecycle；本轮只新增 bulk 的 direct dependency，不改现有 layout；
- `join.h/coro.h`：逐项 result/abort/when-all 与 suspending typed route，映射到相邻第三/第四档，不复用其
  per-item state；
- `sync.h`：`WaitGroup` 是动态异构组合路线，`Event/ConditionVariable` 证明 external-thread/Fiber 双模 wait；
- `cancel.h`：`CancelToken`/`CancelScope` 作为一次 batch 共享取消源，runner 内只做 O(1) poll；
- `blocking.h`：阻塞工作独立路线；`timer.h/waker.h/fiber.h` 不因 bulk 增加新公开职责；
- 用户指南和架构任务表当前没有 batch 档，旧 `structured_task_bench` 又把不同语义写成 peer 行，均映射到
  本轮文档/证据 FIX。

## Capability / gap ledger（冻结 `FIX` 集）

| ID | demand / defect | owner 与合同 | coverage / action | verification / negative space |
|---|---|---|---|---|
| `FIX-01` | `SCHED-V2-BATCH-01` 缺少有限 batch structured route | semantic/artifact/delivery=`sched`；borrow 到同步返回，caller-owned result，O(N) work + O(worker) queue entries + O(1) batch state | `MISSING -> FIX`：新增 `bulk.h/bulk.cc` 与 lambda adapter | zero/one/range/grain、exactly-once、external/Fiber/worker-stack、single-worker、public header；不做 async handle |
| `FIX-02` | batch failure/cancel/admission/shutdown 边界未定义 | first-error、cooperative token、partial admission、quiescent teardown | `MISSING -> FIX`：一次共享状态，无 per-item status/cancel/handle/allocation | first error、pre/mid cancel、CancelScope、closed/full、partial admission/stress；不做强杀与错误 index 排序 |
| `FIX-03` | 四档选择和 Graph 边界不可发现 | architecture/user owner=`sched`，Graph 只作邻接记录 | `PARTIAL -> FIX`：更新 task table、可复制 journey、陷阱与成本表 | docs checker + user compile；不宣传 Graph 已迁移 |
| `FIX-04` | 新 header/TU 的 component/package 闭包尚无证据 | delivery owner=`build/package`，产品依赖不变 | `PARTIAL -> FIX`：登记 sched source/header，验证 source/static/shared 与 FFI 清洁 | selected-component 三 journey、header self-contained、符号/import；不新增 build option/third party |
| `FIX-05` | v1 structured peer 行语义失配 | evidence owner=`sched/perf`；primary 固定为 bulk same-contract | `INVALID -> FIX`：替换为 batch primary；rich 行只保留 `NATURAL_ROUTE/NOT_COMPARABLE` | semantic matrix、W1-WIN qualification、反序/holdout 仅按预算；不据 mechanism diagnostic 改 API |
| `FIX-06` | 首个候选每个 public grain 都在四个 worker 间竞争同一 CAS cursor；qualified batch replay 对 oneTBB 中位慢 8.1%--9.3%，在 `n=31` 后仍未通过 5% NI | product owner=`sched`；公开 grain 是 callback 粒度，不要求内部每次只领取一个 grain；保持 O(N) work、O(worker) runner、O(1) batch state 与 stop contract | `INCONCLUSIVE -> FIXED`：`fc9cc652` 让 runner 有界预领取，正常范围用 `fetch_add`、回绕边界用 saturating CAS；仍逐 grain 调 body、每次调用前检查 stop/cancel，不新增 API/layout/常态外部 metadata | 既有 callback/exactly-once/error/cancel/partial-admission/stress 全部复跑；唯一正式 P1 重跑；不加入公开 partitioner/auto-grain，不改 normal workload、peer 或 5% 门 |
| `FIX-07` | 最后 runner 的 `Event::signal()` 可先唤醒 waiter、随后仍在 `notify_all()` 读取栈上 Event；caller 返回并复用 `BulkState` 时形成真实 lifetime race | correctness owner=`sched`；同步返回必须保证所有 body 与 terminal notifier 不再访问 borrowed state，不能靠“signal 已发生”冒充 signaler quiescent | `MISSING -> FIX`：last runner 在 `signal()` 返回后 release-publish `notification_complete`，waiter 在 Event 返回后 acquire 等待该 handoff；publish 后 runner 不再触碰 state | WSL TSan 原 `RepeatedBatchLifetimeStress` 必须由 FAIL 转 PASS；Release/ASan/UBSan、external/Fiber/one-worker/500-round stress 全复跑；保持 O(1) metadata、无分配、无新 API |

本轮产品修改严格限于 `FIX-01..07`；`FIX-01..06` 已实现，当前唯一未完成产品项是 correctness
`FIX-07`。动态 task-group API、Graph 迁移、caller-participating SPMD、公开自动 grain/partitioner 与
P2 推理端到端均不因本审计进入施工。

## Owner 六问直答（`AUDITED`）

1. **第三方：**Sched 产品直接使用 vendored Boost.Context fcontext 汇编；经既有公开 component closure
   间接使用 Core/Error/Time/Platform/Mem/Util-core，Container 是 private queue 支撑。Marl/oneTBB 只作
   peer，llama.cpp/ggml 只作 consumer/capability 输入；`bulk_execute` 不新增第三方 product dependency。
2. **功能：**llama.cpp/ggml 的共享 thread episode、barrier、abort 与 dynamic chunk shape 证明 batch
   route 是推理底座 `MUST`；`bulk_execute` 已补齐有限 indexed/caller-context batch、共享首错/取消、同步
   quiescence、partial admission 与 worker-stack progress fallback。动态异构 addition 由
   `schedule + WaitGroup` 路由，逐项 rich 与可挂起任务各有独立入口；Graph 的 adaptive algorithms 明确
   归 Graph。当前审计没有仍属 Sched 的能力缺口。
3. **性能：**本轮为 P1。首个候选已证明零 per-item allocation/status/cancel/handle、O(worker) queue
   entries 与 O(1) terminal wake；qualified W1-WIN bundle 的资格、A/A、5%/10% known delta、timer floor、
   artifact/build/provenance 全部通过。对 Marl 的两个顺序中位 advantage 为 `1.022/0.988`，第二个顺序
   的 NI 置信区间跨过 0.95；对 oneTBB 为 `0.919/0.907`，两顺序均跨过 0.95，因此聚合性能为
   `INCONCLUSIVE`。唯一 codegen 诊断确认热循环逐 grain 执行 `lock cmpxchg` 且竞争失败回跳；`FIX-06`
   只修该成本结构，随后使用原合同完成唯一重跑，当前不声明 SOTA 或非劣。
4. **旧 C ABI / Native FFI：**全局策略 `N/A`；当前 Sched 公开树没有 C schema/header/bridge/专用 option。
   fcontext `extern "C"` 是私有汇编边界。新能力只提供 C++ header/decorated symbol，不制造伪 C ABI。
5. **构建与导入：**首个 candidate 已在 Windows/WSL/macOS 证明 source embedding、relocated static 与
   relocated shared selected-component consumer；`bulk.h/bulk.cc` 安装和导入正确。`FIX-06` 只改同一 TU
   的私有算法，但新 candidate 仍须复跑受影响 producer/consumer 与 Windows/ELF/Mach-O symbol/import，
   不能自动继承旧二进制。
6. **C++ API 与文档：**四档选择表、`noexcept` lambda/Status 例子、caller-owned lifetime、grain、并发写、
   worker-stack serial fallback、取消/失败非事务性等陷阱已经进入 architecture/user docs 并通过用户头编译。
   `FIX-06` 进一步明确 grain 是用户 callback/停止粒度，内部 reservation 不是调参要求或公开 partitioner；
   当前 public shape 自然，terminal 仍须重跑 installed consumer 与 docs/symbol drift check。

## 首个 Candidate（本轮历史输入，已因 `FIX-06` 退阶）

以下 identity 曾是冻结产品候选；2026-08-31 的 qualified P1 未能裁决后，registry 已合法退回
`AUDITED`，所以它不再是当前候选，也不授权继续验证为 `VERIFIED`：

- commit：`7cd2ad5a0d49f3c396a74888dc79fd98ebed09d2`；
- tree：`8e15c2be0c784dc43f985119209f84df39899270`；
- public contract：新增 `hyle/sched/bulk.h` 的 `BulkExecuteOptions`、raw `BulkBodyFn`、compiled
  `bulk_execute` 与 `void/Status` noexcept lambda adapter；现有 Scheduler/InlineJob/JoinHandle/Fiber layout
  与入口不变；
- implementation：一个 caller-stack `BulkState`，至多 `min(chunk count, max concurrency, workers)` 个 raw
  `schedule` runner，一个 lock-free chunk cursor，一次 manual Event terminal notification；无 per-item state、
  allocation、queue entry、status、cancel 或 handle；
- correctness semantics：external thread 与 Scheduler Fiber 走 parallel route；worker-stack nesting serial
  fallback；first body error 优先于 cancellation；partial runner admission 只降低并发，不丢 chunk；零 runner
  admission 返回 typed scheduler error；
- 直接产品面：Sched CMake 新增一个 TU，自动 public-header projection 安装 `bulk.h`；用户选择表、可复制
  journey、陷阱、架构任务表和复杂度合同已更新；Graph 只保留 future migration seam，未改产品；
- candidate tests：W1-WIN MSVC 19.50 Release shared selected-component build 通过；9 个 BulkExecute tests 与
  全部 166 个 Sched tests 通过；Sched 10 个 public headers self-contained compile 通过；计数 allocator
  证明 warm 后 4096-item bulk 的 runner path 新增 allocation 为 0。

`AUDITED -> CANDIDATE` 前态快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v2\module-status-before-candidate.json`，SHA-256
`8A79BB632B24C52B23598A5554FF895BAC41A2011331B661FFDE616137DBC924`。

## `FIX-06` Candidate（本轮历史输入，已因 `FIX-07` 退阶）

以下 identity 曾冻结为产品候选；WSL TSan 在 terminal notifier 中发现真实 lifetime race 后，registry 已
退回 `AUDITED`，所以它不再是当前 candidate，也不能进入 `VERIFIED`：

- commit：`fc9cc65207d6d05589da89b32435a62f0733e2ad`；
- tree：`6e516ceaec2f83ce3089429b07b5bd9bd945d80e`；
- public contract：保留首个 candidate 的 `BulkExecuteOptions`、raw `BulkBodyFn`、compiled entry 与
  `void/Status` lambda adapter；只把 stop 文字从内部“claim”改写为用户可观察的“下一次 body invocation”，
  无新 symbol、option、public type/layout 或依赖；
- implementation：一个 caller-stack `BulkState`、O(worker) raw runners、O(1) manual Event completion；
  正常范围把 chunk cursor RMW 限到约 `min(chunks, worker_count * 8)` 个 reservation 加 terminal claims，
  接近 `SIZE_MAX` 时以 saturating CAS 防回绕；reservation 内逐 public grain 调 body 并逐次观察 stop/cancel；
- resource/complexity：O(N) body work、O(worker) queue entries、O(1) batch metadata/wake，仍无 per-item
  allocation/status/cancel/handle/queue entry；内部多两个 `size_t/bool` 是一次 batch 算法状态，不进入
  Scheduler/InlineJob/public layout；
- candidate tests：fresh W1-WIN MSVC 19.50 Release shared、`/W4 /WX` 通过；全部 166 个 Sched tests 与
  10 个 public headers 通过。既有 grain=7 test 现额外签 callback 数与精确边界，first-error 仍最多四个
  in-flight body，cancel/stress/zero-allocation 全部通过；尚未查看修正后的性能结果。

`AUDITED -> CANDIDATE` 前态快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v2\module-status-before-candidate-fix06.json`，SHA-256
`21161A0A8F0164DC75CA607A38879B5587224C221BF7819D8DA89369B0B92B20`。

## Verification manifest

候选前本机定向 correctness 只作为 candidate test：

| gate | W1-WIN candidate result | terminal requirement |
|---|---|---|
| new bulk semantics/resource tests | `PASS`（9） | candidate 上重复，并扩展 sanitizer/stress |
| full Sched unit suite | `PASS`（166） | 三平台适用 suite |
| public headers | `PASS`（10） | 三平台 + installed consumer |
| WSL/macOS correctness/sanitizer | `FAIL`（WSL Release/ASan/UBSan 通过；TSan 在 `RepeatedBatchLifetimeStress` 报 terminal Event vs 下一轮 `BulkState` 构造 race；macOS 尚未运行） | `FIX-07` 后 WSL 四档 + macOS Release |
| source/static/shared delivery | `BLOCKED`（新 candidate 尚未运行） | 三平台受影响 selected-component consumer |
| same-contract P1 | `BLOCKED`（新 candidate 尚未运行；首个 candidate 的 `INCONCLUSIVE` 只作历史输入） | 原合同唯一一次 W1-WIN 重跑 |
| docs/usability/FFI cleanliness | `BLOCKED`（candidate docs/header compile 已过；installed/symbol drift 尚未重跑） | checks + consumer compile + symbol scan |
| red team | `BLOCKED`（尚未运行） | candidate 验证后独立攻击 |

## 独立红队

等待 `FIX-06` 后的新 candidate 与全部验证后执行，先攻击 comparison class、semantic-equivalence matrix、
reservation 对 first-error/cancel latency 的影响，以及是否出现未声明的 public grain policy。

## Closure decision

当前机械状态：`AUDITED`。`FIX-06` 的性能成本结构成立，但该 candidate 被 mandatory correctness
`FIX-07` 否决。下一步只能实现 signal-return handoff、冻结新 candidate，再复跑受影响 correctness、
三平台、三交付、原 same-contract P1、文档/FFI cleanliness 和独立红队。不得改 public API、normal
workload、peer、5% margin 或中央统计语义；P1 若仍未通过就按卡终止为具名 `BLOCKED`。
