# Sched v4 控制与结果合同

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前唯一状态真相是 [`module-status.json`](../../module-status.json)。本轮由只读
> [`Sched 当前产品回放`](sched-current-product-replay-2026-09.md) 的具体产品事实触发，不重做 v3 已经
> 关闭的 root participation 优化，也不把推理引擎策略下沉到通用调度器。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`sched` / Sched execution owner /
  `sched-v4-control-contracts` / `REOPEN`；
- baseline：`f4da6ed4a2953b183a99532f31686852147ca09e`；previous round：
  `sched-v3-root-participation`；
- trigger：`SCHED-V4-CONTROL-CONTRACT-2026-09-07`；
- reproducible facts：`JoinHandle::abort()` 只写 cooperative flag，parked task 必须由另一个 owner 唤醒；
  raw `DetachedFn/EntryFn` 接受可抛函数却在 `noexcept` trampoline 执行；Semaphore 的 `bool` 把 closed、
  unavailable 与 timeout 合并；用户/构建文档另有可定位的实现漂移；
- preliminary affected surface：`join/cancel/waker` 的控制合同、`Scheduler/Fiber` raw callback type、
  Semaphore acquire results、直接测试/consumer、public header closure、三种 C++ 交付和相关文档；
- preliminary non-goals：完整 scheduler/queue/fiber 重写；强制销毁 stack；绕开 operation owner 直接恢复
  仍挂链的 Fiber；external-caller SPMD/gang、模型/请求优先级、continuous batching、GPU/异构策略、
  Compute/Graph/IO 重构、Native FFI/C API、公共性能仪器修改、P2/Formal 和按 peer API 求并集；
- preliminary tier：`P1`，5% material margin。只有有限 `FIX`、API/usability 和 candidate 冻结后才定义
  并运行正常 workload；本状态不授权产品实现或性能采样。

下一合法状态是 `SCOPED`。届时必须冻结真实 journey、取消 ownership/teardown 语义、Waker 裁决、
直接依赖与 consumer、预算、机器/仪器、适用 gates 和停止条件；`AUDITED` 前不得修改产品源码。

## Closure card（`SCOPED`）

- component / owner / round / kind：`sched` / Sched execution owner /
  `sched-v4-control-contracts` / `REOPEN`；
- baseline / trigger：`f4da6ed4a2953b183a99532f31686852147ca09e` /
  `SCHED-V4-CONTROL-CONTRACT-2026-09-07`；
- normal journeys：
  1. 推理请求在一个 structured task 中等待 timer、channel 或 backend completion；owner 请求取消后，
     task、等待源和 join 的责任边界明确，不发生强制 stack unwind、悬挂 waiter 或无限误等；
  2. runtime owner 用 raw function pointer 提交非挂起 callback 或构造低层 Fiber；可抛入口在编译期被
     拒绝，而不是跨 `noexcept` trampoline terminate；
  3. bounded resource gate 的调用方能区分 acquired、temporarily unavailable、timeout 与 closed，RAII
     permit 不丢失原因；
- scope：反向追踪并冻结 cooperative task-cancel 与 operation-aware cancel 的分层；修正
  `JoinHandle`/token/wait recipe 中误导的名称或组合；让 raw callback 类型表达 `noexcept`；让 Semaphore
  acquire family 返回可判别结果并保持 FIFO/O(1) unlink；有限裁决 Waker one-shot 前置条件；迁移直接
  consumer；修正相关 architecture/user/performance/build 文档和三交付；
- non-goals：完整 scheduler/run queue/Fiber/context-switch 重写；强杀 stack；不经 operation owner 就恢复
  仍挂在 timer/IO/waiter 链的 Fiber；把所有同步原语改成自动取消；external-caller gang/SPMD、公开
  partitioner/auto-grain、worker resize、模型/请求 priority、continuous batching、GPU/异构策略；
  Compute/Graph/IO 语义重构；拆分整个 `sync.h`；Native FFI/C API；公共仪器修改；P2/Formal；
- direct dependencies：PUBLIC `core,container_intrusive,error,time,platform,platform_thread,mem,mem_vm,
  util_core`；PRIVATE `container_queue,timer_queue`；vendored Boost.Context fcontext 是 private product
  mechanism，Tracy 是条件 build-only mechanism；
- direct consumers：`io,compute,graph` 公开面使用 Sched；本轮只做被 public type change 强制要求的
  compile migration，不给这些未完成模块签收口结论，也不改变其产品策略；
- reuse candidates：Error `Status/Result` 表达 closed/timeout/unavailable；Time 提供 typed deadline；
  Container-intrusive 提供 callback/waiter 链；Util-core 提供短临界区锁；Platform-thread 提供 thread
  parking；Timer-queue 提供 deadline removal；现有 `CancelToken/CancelCallback` 和 Waker rendezvous 优先
  组合，不另造第二套取消树或 OS wait；
- strong peers：取消/任务合同使用 Tokio
  `060cc4e46aedc40b0a44dcbc31deb0df39abdb4e` 与 Folly
  `65749da412d92fe27d87ae45a8ef670720712858`；Semaphore 结果与 FIFO 使用 Tokio 同 commit 和 C++20
  `std::counting_semaphore` 的定义性边界；task/bulk negative space 使用 oneTBB
  `e00c68dcac0e83947c9b54f6d5304c0616c9639f` 与 ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` consumer shape；
- inference evidence：llama.cpp `3ad1ba7336986d98592d3e28cafd1a406715351f`、ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`、vLLM
  `52358e6e192aeae73dd3764046c98fc4b156ea83`、SGLang
  `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`；只提取 request completion、cancel/shutdown 与 CPU
  execution journey，不复制服务策略或 Python/GPU surface；
- provenance：已复现 API defect 为 `REAL_CONSUMER` 可达公共路径；推理请求等待/取消为
  `NORTH_STAR_JOURNEY`；peer 只验证通用合同；
- semantic tiers / selection：短、不挂起工作继续选择 `schedule/schedule_task`；可挂起 structured work
  选择 `spawn_task`，task cancellation 是 request/cooperative，in-flight operation cancellation 必须由
  operation owner 通过 token/callback 完成；Semaphore blocking、non-blocking 和 timed 入口共享一个
  可判别 outcome vocabulary；Waker 保持低层 completion seam，不成为普通用户取消 API；
- normal cost：既有未选择取消的 schedule/spawn/wake 路线不得新增 allocation、全局查找、syscall、
  generation word 或每次 wait CAS；显式 operation cancellation 可为 control-path token/callback 支付
  O(1) 注册/摘除和既有 token ownership；raw callback 修复是纯类型约束；Semaphore outcome 不新增
  allocation、waiter metadata、lock、wake 或复杂度，fast path 仍是一次短锁内检查；
- tier / margin：`P1` / 5%。primary 是 source-build 四 worker 的 typed task fan-out/join 与 cooperative
  cancel lifecycle，和 immutable baseline 做同产品回归；holdout 只复用 v3 已资格化的四 worker
  batch workload 对 Marl/oneTBB，确认无旁路回归。取消完成延迟只作有界 user-budget，不伪装成与
  stackless Tokio 的 primitive parity；
- budgets：capability/API 审计 90 分钟；实现与 correctness 120 分钟；delivery/docs 60 分钟；P1 45
  分钟；red team 30 分钟。P1 最多一次 primary qualification、一次 holdout qualification、一次机制
  diagnostic 和一次性能导向内部修正；fixture/tool/environment 纠正另记但不能改变产品范围；
- machines / instruments：`W1-WIN` 为 P1 裁决机，`closure-perf-v1`；`W1-WSL` 与 `M1-MAC` 做
  correctness/sanitizer/source-static-shared portability，`closure-correctness-v1`；
- correctness gates：取消 before-start/running/parked/completed 和 cancel-vs-completion race；operation
  owner teardown/late completion；when-any loser recipe；raw callback positive/negative compile；Semaphore
  zero/many/FIFO/close/timeout/race/result/RAII；Waker wake-before-suspend 与选定误用合同；shutdown；现有
  sched test suite；WSL ASan/UBSan/TSan；
- delivery gates：公开头自包含；selected `sched` source build；installed relocated static/shared C++23
  consumer；Windows/Linux/macOS；fcontext package/license；旧 C/Native FFI clean；direct IO/Compute/Graph
  compile smoke 仅在可构建范围内；
- outputs：capability map、public trace、reuse ledger、有限 `FIX`、API/usability freeze、实现/测试、
  architecture/user/build/performance 文档、candidate、三平台验证、独立红队、三项 LOC、terminal receipt
  与新 `FROZEN` confirmation；
- stop：`AUDITED` 后只实施冻结 `FIX`；一个 candidate 和上述有限 P1。全部 MUST complete/routed 且适用
  correctness/delivery/docs/reuse/red-team/P1 通过才 `CLOSED/FROZEN`；安全取消 ownership 无法定界、
  sanitizer/交付失败或正常 workload 稳定 material regression 则 `BLOCKED`，不增加测试姿势或扩建仪器。

当前进入 `SCOPED`；以上范围、非目标、成本、预算和停止条件冻结。产品实现必须等待 capability map、
public reverse trace、Waker/取消 ownership 裁决与有限 `FIX` 集在 `AUDITED` 完成。

## Capability-family coverage map（`AUDITED`）

| Family | admitted journeys / evidence | semantic/lifecycle boundary | current route | coverage | action / negative space |
|---|---|---|---|---|---|
| worker runtime | 短 callback、stackful task、affinity、steal、shutdown | worker-stack 与 Fiber 分层；bounded admission；两阶段 shutdown | `Scheduler/Fiber` | `COMPLETE` | `KEEP`；resize/pause/gang policy 不由本轮准入 |
| structured result | typed short/suspending/blocking result，join/fan-out/select | handle move-only；drop detach；task/result owner 分离 | `JoinHandle`, `schedule_task`, `spawn_task`, `BlockingPool`, `when_all/any` | `PARTIAL` | `FIX` request-only cancel 命名与 loser recipe；不强杀 stack |
| operation cancellation | 请求树取消和在途 IO/backend operation | caller/token/operation owner 共同维持 callback 与 completion 生命周期 | `CancelToken/Scope/Callback` | `COMPLETE` | `KEEP/ROUTE`；不承诺任意 wait 自动可取消 |
| completion spine | foreign-thread completion、wake-before-suspend | 每 episode 单 owner、一次 wake；Fiber 仅 suspended 时迁移 | `Waker/suspend` | `PARTIAL` | `FIX` move-only consuming handle；不增加 generation metadata |
| timer/deadline | fiber sleep、timed wait、timeout composition | monotonic；per-worker queue；owner-thread cancel | `timer_queue` + `timer.h` | `COMPLETE` | `KEEP` |
| synchronization | thread/Fiber CV wait，event/latch/barrier/rwlock | Mutex 仅短且不得跨 suspension；其他长等待走 Waker/Parker | `sync.h` | `COMPLETE` | `KEEP`；修正文档，不在本轮拆头 |
| resource permits | blocking/nonblocking/timed FIFO permits、RAII、close | 结果必须区分 closed/unavailable/timeout；zero request 是 no-op unless closed | `Semaphore` | `PARTIAL` | `FIX` 为 HYLE `Result`，保持 O(1) fast path/FIFO |
| messaging | bounded channel、oneshot、watch、broadcast | typed value ownership、close/backpressure/missed message | `sync.h` messaging types | `COMPLETE` | `KEEP`；per-wait cancel 是后续具名 consumer trigger |
| finite data parallel | inference tile fan-out、batch first-error/cancel | non-suspending borrowed body；batch result；同 owner root helping | `bulk_execute` | `COMPLETE` | `KEEP`；外部 caller gang/adaptive partitioner 上层路由 |
| delivery/diagnostics | source/static/shared、trace、fcontext | C++ SDK；private assembly；无 Native FFI | component graph/package | `PARTIAL` | `FIX` 文档依赖和 vendored platform verification drift |

### Public-surface reverse trace

| Header / public family | demand | disposition |
|---|---|---|
| `scheduler.h`, `fiber.h` | execution/admission/affinity/shutdown and low-level owner | `KEEP`；两个 raw callback type 按真实 no-throw contract 修正 |
| `join.h`, `cancel.h` | typed completion、fan-out 和 cooperative/operation cancel | `KEEP`；删除误导的 `abort` 名称，改为 `request_cancel`；token/callback 保持 operation owner 路线 |
| `waker.h`, `coro.h` | one completion spine and coroutine shell | `KEEP`；Waker 改为 move-only、一次消费，raw constructor 私有 |
| `timer.h` | monotonic sleep and timeout composition | `KEEP`；本轮无产品变化 |
| `bulk.h` | finite non-suspending batch | `KEEP`；本轮无产品变化 |
| `blocking.h` | blocking syscall/CPU handoff | `KEEP`；随 JoinHandle 名称迁移 |
| `sync.h` | short lock、long waits、resource gate、message primitives | `KEEP`；只改 Semaphore result 与错误 Mutex 文档，不按审美拆头 |
| `detail/integration.h` | IO worker-loop bridge | build-only `KEEP`，不得安装 |

现有公开入口均可映射到上述用户任务，没有 benchmark-only public symbol。`schedule`、`schedule_task`、
`spawn_detached`、`spawn_task` 与 `bulk_execute` 的区别来自 result、suspension、ownership 和 batch
granularity，不是“快/慢”命名。

## Capability / gap ledger 与有限 `FIX`

| ID | class / demand | current defect | owner / cost target | action | verification |
|---|---|---|---|---|---|
| `SCHED-V4-CANCEL-01` | `MUST` / reproducible API defect | `abort` 名称不能表达只写 cooperative flag，文档把 `when_any` loser cleanup 写得过强 | Sched semantic owner；默认 task 不增 allocation/metadata/wake | `FIX`：唯一公开名 `request_cancel()`；明确 CPU checkpoint 与 operation-token 两条路线，更新全部 direct users | compile/source scan；before-start/running/parked recipe tests |
| `SCHED-V4-NOEXCEPT-01` | `MUST` / exception-boundary correctness | raw callback 接受 throwing function pointer | Sched/standard C++ type owner；零运行时成本 | `FIX`：`DetachedFn`、`EntryFn` 加 `noexcept`，负 compile assertion | header/API test、三交付 |
| `SCHED-V4-SEMAPHORE-01` | `MUST` / typed failure usability | `bool` 折叠 closed/unavailable/timeout；zero request 可被既有 waiter 阻塞 | Sched + Error；成功 fast path 不增加 allocation/状态/锁 | `FIX`：四个 acquire 返回 `Result<void>`，guard 返回 `Result<Permit>`；错误码分别 Cancelled/Unavailable/DeadlineExceeded；zero 在未关闭时立即成功 | result/zero/FIFO/close/timeout/race tests；codegen/resource P0 |
| `SCHED-V4-WAKER-01` | `MUST` / low-level lifetime misuse | 可复制 raw pair 与一次性合同矛盾，重复 wake 会污染下一 episode | Sched completion owner；保持两指针、同一次 rendezvous | `FIX`：move-only、`wake()` 消费并先失效；raw pair constructor 私有；竞争 completion 仍由 operation owner 仲裁 | type traits、wake-before-suspend、IO/Graph compile/runtime smoke |
| `SCHED-V4-DOC-BUILD-01` | `MUST` / current contract drift | Mutex fiber 行为、private factor、fcontext verification 说明错误 | Sched/build/document owner | `FIX`：校准 architecture/user/build/third-party/performance；不改 component edge | docs/closure/selected package checks |
| `SCHED-V4-POLICY-01` | `GENERAL` / inference differential | ggml caller gang、vLLM/SGLang admission/priority、GPU policy | Compute/Graph/inference runtime | `DEFER/ROUTED` | negative-space docs；不得形成 Sched metadata |
| `SCHED-V4-CANCEL-WAIT-01` | `GENERAL` / mature async runtime | 任意 Channel/CV wait 没有 per-wait token overload | future named consumer + owning primitive | `DEFER`：当前 IO 已有 operation-owned callback，global close/shutdown 已完整；无本轮 concrete consumer 时不扩大所有 wait | reopen trigger；不得宣称任意 wait 可主动取消 |

本轮冻结 `FIX` 恰为前五项。header 拆分、worker resize/pause、per-task priority、自动 grain、任意 wait
token overload 和新调度策略均不进入施工。

## API / usability freeze

- 状态：`FROZEN`；reviewer/date：Codex / 2026-09-07；
- `JoinHandle<T>::request_cancel() noexcept` 取代并删除 `abort()`；它只请求 cooperative cancellation，
  不承诺唤醒 parked task。CPU work 在 checkpoint 观察；在途 wait 必须把 caller-owned `CancelToken` 交给
  operation owner，并由该 owner 的 `CancelCallback` 安全取消/完成后 wake；
- `Scheduler::DetachedFn` 与 `Fiber::EntryFn` 精确为 `void (*)(void*) noexcept`；capturing callable 路线
  继续由 concept 拒绝可抛 body/move；
- `Semaphore::{acquire,try_acquire,try_acquire_for,try_acquire_until}` 返回 `Result<void>`：success、
  `Cancelled`(closed)、`Unavailable`(immediate no permits/FIFO predecessor)、`DeadlineExceeded`(timeout)。
  guard family 返回 `Result<Semaphore::Permit>`，不再用 invalid Permit 隐藏失败原因；
- `Waker` 只由 `current_waker()`/Sched 内部构造，move-only；`wake()` 在调用 scheduler 前消费并失效自己。
  多 completion 竞争必须先由 operation state 选出唯一 winner；不为每个 Fiber/Waker 增加 epoch word；
- normal sketches：

```cpp
auto task = hyle::sched::spawn_task(scheduler, alloc, serve(alloc, token));
token.cancel();          // operation owner callback cancels an in-flight wait
task.request_cancel();   // cooperative CPU checkpoint; does not forge a wake

auto permit = gate.try_acquire_guard();
if (!permit && permit.error().IsUnavailable()) { /* apply backpressure */ }
```

- ownership/default：既有无取消 task 路线成本不变；token/callback 是显式 opt-in control-path ownership；
  result failure 使用已冻结 Error code，不引入异常或 hidden allocation；
- dependency/delivery：public factor closure 不变；source/static/shared 都暴露相同 C++ headers；无 C ABI；
- 冻结后只允许内部实现、直接 consumer 编译迁移、测试和文档。若必须改变上述名称、结果、ownership 或
  常态成本，退回 `AUDITED`；benchmark 点值本身不能触发退回。

## Owner 七问（AUDITED）

1. **第三方：**产品直接使用 vendored Boost.Context fcontext：release `boost-1.92.0`、revision
   `cc1b9e02cd530d6607becc984e1e5bfa4f7b3600`，Boost-1.0；Boost 拥有汇编 artifact/mechanism，HYLE
   拥有 Fiber semantics、选择、锁和交付。Tracy 仅 trace-ON build mechanism；Container queue 等是 HYLE
   factor，不冒充第三方。标准 C++ type/coroutine/atomic 与 Windows/Linux/macOS OS wait/thread API 另记
   平台合同。
2. **推理能力：**四组锁定 reference 的 request completion、CPU fan-out、cancel/shutdown、blocking
   handoff 均 `COMPLETE/ROUTED`；服务 admission、continuous batching、模型 priority、GPU stream 和
   ggml gang 是上层策略。当前未处置 `MUST`：`NO`，条件是冻结五项 FIX 完成。
3. **内部复用：**Error/Time/Container-intrusive/Container-queue/Util-core/Platform-thread/Timer-queue/Mem
   均沿合法直接边复用；fcontext 本地包装是第三方 mechanism seam；Waker 与 cancel tree 是 Sched 唯一
   runtime authority。verdict：`PASS`，待实现验证无第二套结果/取消状态。
4. **性能：**P1，source-build normal typed task/cancel lifecycle + v3 bulk holdout；当前为 `PLANNED`。
   禁止全操作、全平台、推理引擎或 Formal SOTA 声明。
5. **FFI：**没有已准入 Native FFI/C API；fcontext `extern "C"` 是 private assembly ABI，shared decorated
   C++ exports 不是跨语言承诺。当前 surface scan `CLEAN`，候选后重验。
6. **交付：**中央 factor/产品均为 `sched`；source/static/shared journey 和 fcontext target 已存在，当前
   结果为 `PLANNED`，候选后重跑 selected/relocated consumer。
7. **易用性：**任务入口按 suspension/result/ownership 选择合理；本轮修复 cancel 名称、Semaphore
   failure 与 Waker ownership，并校准 Mutex/构建说明。当前为 `PLANNED`，实现与 guide 后裁决。

## Audit decision

自顶向下和自底向上审计均完成；没有新增第六项产品 FIX。`sched` 进入 `AUDITED`，只允许实施冻结的
`SCHED-V4-CANCEL-01`、`NOEXCEPT-01`、`SEMAPHORE-01`、`WAKER-01` 与 `DOC-BUILD-01`。未运行的测试、
交付和性能仍为 `PLANNED`，不能写成 PASS。

## AUDITED implementation（候选前）

冻结的五项 FIX 已实施，尚未因此授予 candidate 或 gate verdict：

- `CANCEL-01`：删除 `JoinHandle::abort()`，唯一入口改为 `request_cancel()`；公开注释与 guide 明确它只
  发布 cooperative flag。在途 timer/IO/wait 仍由 caller-owned token 与 operation owner 的 callback
  撤销，`when_any` 不伪造 loser wake；before-start、running、parked、completed 与 completion race 均有
  定向测试；
- `NOEXCEPT-01`：`Fiber::EntryFn` 与 `Scheduler::DetachedFn` 的函数指针类型包含 `noexcept`，现有 raw
  consumer 迁移；compile-time assertion 同时证明 potentially-throwing pointer 不可转换；
- `SEMAPHORE-01`：acquire family 和 guard family 使用 HYLE `Result`，分别保留 `Cancelled`、
  `Unavailable`、`DeadlineExceeded`；zero request 在未关闭时先于 FIFO 检查成功，closed 仍优先返回
  `Cancelled`；waiter 仍为 stack node、timed unlink 仍 O(1)，成功 fast path 未增加 allocation、lock 或
  metadata；
- `WAKER-01`：Waker 保持两指针，改为 move-only consuming handle，raw pair 构造收为 Sched 私有；
  notify/timeout/shutdown/Graph 直接 consumer 在锁内或 owner 状态中转移唯一 handle、锁外消费。重复
  Windows race 回归曾真实捕获 notify 在 timed helper 进入前取走 Waker 的合法交错；修复后 timeout 与
  notify 都不再借用 stack node 上的可复制 handle；
- `DOC-BUILD-01`：user/architecture 的 Mutex、取消、Semaphore、Waker 选择说明已校准；中央依赖说明
  修正为 PRIVATE `container_queue,timer_queue`；fcontext README 只声明真实执行过的 Windows x86-64、
  Linux x86-64 与 macOS arm64 变体，README 改动同步更新中央 vendored content hash。

候选前本地检查：Windows MSVC Release `sched` 公开头与 173 tests 通过，完整 suite 连续重复十轮通过；
全部 Sched benchmark target 只编译不采样并通过；Windows shared build 的 IO、Compute、Graph 直接下游
均通过编译。它们是实现回归输入，不替代 immutable candidate 后的三平台、sanitizer、三交付、P1 与
独立红队 gate。

## Candidate

- 状态：`CANDIDATE`；candidate commit：
  `dbc9634455176450fb86d68b373c6cc4a4156f24`；candidate tree：
  `5e39d8ab76b6da83e7dd267584f9592c59930532`；
- candidate 包含全部五项 FIX、直接 consumer 编译迁移、语义/竞态测试、当前 user/architecture/build
  文档和 fcontext provenance hash；API/usability freeze 未变化；
- 从现在起产品源码、公开 API、常态 layout/cost 和 candidate tree 不再修改。后续只允许执行 closure
  card 已声明的正确性、交付、P1、红队与记录；出现必须修改候选的真实问题时退回 `AUDITED`，不得在
  candidate 上静默补丁。

## Candidate 后的 fixture 与直接消费者校正

候选后的三项改动都在 Sched 产品边界之外：

- `8d423bdf` 把 IO/Compute/Graph 私有测试和 benchmark 的 raw callback fixture 补成 `noexcept`；
- `d460ad40` 修正 Graph 原有的多 finisher Waker 所有权：只有 `pending` 的最后完成者移动并消费一次
  Waker。该 O(1)、零分配修复属于尚未冻结的 Graph consumer，不改变 Sched API/实现；
- `e91fc36b` 把 P1 completion fixture 的 Waker 循环从 `const` 借用改为一次消费。首次 qualification 在
  编译期停止，没有产生样本；这是冻结卡允许的 fixture correction。

执行证据使用 clean `e91fc36b52075f8d36a5d4cb2f375e4c967082b1`；
`git diff dbc96344..e91fc36b -- include/hyle/sched src/sched` 为空，所以产品 candidate/tree 保持不变。

## Candidate verification（`VERIFIED`）

| Gate | W1-WIN | W1-WSL | M1-MAC | Verdict |
|---|---|---|---|---|
| public headers + Sched suite | MSVC 19.50，2/2 CTest；suite 内 173 cases | GCC 14.2，2/2 | AppleClang 21，2/2 | `PASS` |
| WSL sanitizers | 本卡指定 WSL | ASan、UBSan、TSan（`setarch x86_64 -R`）均 2/2 | 本卡指定 WSL | `PASS` |
| selected source / relocated shared / relocated static | 三条均通过 | 三条均通过 | 三条均通过 | `PASS` |
| direct consumers | IO file/net/http、Compute GPU await、Graph parallel/data-parallel 六项通过 | 同六项通过 | 同六项通过 | `PASS` |
| compiled surface | 无旧 HYLE C/Native FFI；DLL 无 `SchedulerAccess` | 无公开旧 FFI/fcontext symbol | 无公开旧 FFI/fcontext symbol | `PASS` |

Windows DLL 可见 `jump_fcontext/make_fcontext/ontop_fcontext` 三个原始符号；它们是目标内 vendored
assembly 必需的 private ABI，不是 HYLE C API，公开 package/guide 也没有把它们声明成消费面。macOS
下游 aggregate static link 对重复的 `hyle_time/hyle_mem` archive 给出 linker warning，但没有缺失/重复
定义、三种交付与六项消费者全部通过；这是下游静态 link-line 的构建维护输入，不是 Sched 产品失败，
且本轮未越界重写公共 build system。

### P1 性能裁决与苹果/橘子纠正

`sched-batch-p1-v4` holdout 在 W1-WIN 完成 31 对双顺序样本，qualification/performance、A/A、5%/10%
known delta、timer floor、artifact/build/provenance 与独立 verifier 全部 `PASS`。HYLE 对 Marl 的
advantage 为 `1.153/1.129`，双顺序 `WIN`；对 oneTBB 为 `1.049/1.050`，exact `INCONCLUSIVE`、5% NI
双顺序 `PASS`。精确 bundle/hash 见
[`benchmarks-sched`](../../evidence/performance/benchmarks-sched.md)。

旧 `sched-p1-v1` primary 也在相同机器完成 qualification，但它仍把已在 sched-v1 红队判定为
`NATURAL_ROUTE / NOT_COMPARABLE` 的 typed `schedule_task` 对 Marl caller-owned batch slot 标成
`COMPARABLE`，因此机器 aggregate 如实返回 `FAIL`。这不是可用于改产品的同合同门：

- request fan-out/collect 当前双顺序 `WIN`、5% NI `PASS`（`1.617/1.632`）；
- completion wake 当前双顺序 `PARITY`、5% NI `PASS`（`1.024/1.015`）；
- typed structured row 的 `0.927/0.927` 只保留为 ownership-cost diagnostic；需要 batch-only 语义的消费
  者已有本轮 holdout 验证的 `bulk_execute`；
- 三条当前 HYLE absolute median 对 sched-v1 同机 qualified bundle 的双顺序比值分别为
  `1.132/1.088`、`0.987/1.074`、`1.015/1.009`，没有固定方向的 5% material same-product regression。

原始 FAIL manifest 没有删除、改写或伪装成 PASS；当前 `sched-p1-v1` contract/consumer manifest 已把
typed row 改成 non-gating diagnostic，避免未来再次浪费 31 对样本。跨 bundle 比值只作方向诊断，不
冒充 paired certificate。本轮工程 P1 `PASS` 由两条当前同合同 replay、P0 成本检查和完整
Marl/oneTBB batch holdout 共同给出，不宣称全操作或 Formal SOTA。

## Candidate 后红队

分离的 adversarial pass 逆向检查了：cancel flag 与 operation owner 的 teardown 顺序、Waker move/drop/
double-wake/wake-before-suspend、timed CV notify-vs-timeout、Semaphore zero/FIFO/close/timeout/overflow、
raw exception boundary、shutdown、allocator/result 生命周期、公开 include/dependency closure 和三交付。
没有新的 correctness、ownership、concurrency、复杂度、易用性或交付 finding：

- 默认 schedule/spawn/join 路线没有新增 allocation、全局查找、syscall 或常态 metadata；
- Waker 仍恰为两指针，move-only/consuming 只把已有 one-shot 前置条件变成类型合同；当前 completion
  replay 与 Marl 同合同平；
- Semaphore fast path 仍一次短锁检查，slow waiter 为 stack node，timed unlink O(1)，结果类型不增加
  waiter/lock/wake；
- `schedule`、`schedule_task`、`spawn_task`、`bulk_execute` 的选择来自 suspension/result/ownership/
  batch granularity，不是 benchmark 的 fast/slow 档位；guide 已写明选择和陷阱；
- Error/Time/Container-intrusive/Container-queue/Util-core/Platform-thread/Timer-queue/Mem 均沿中央真实边
  复用，没有第二套错误、时间、wait、queue、allocator 或 cancel authority。

### Aggregate gate verdict

| gate | verdict | basis |
|---|---|---|
| scope / architecture | `PASS` | 五项冻结 FIX 完成，推理策略和下游重构保持负空间 |
| correctness | `PASS` | 三平台 suite/direct consumers 与 WSL ASan/UBSan/TSan |
| delivery | `PASS` | selected source + relocated shared/static × 三平台；FFI 边界诚实 |
| usability / docs | `PASS` | 取消、结果、Waker ownership 和任务选择已进入公开 guide/architecture |
| reuse / red team | `PASS` | 直接边与内部 authority 正确；候选后无 material product finding |
| engineering performance | `PASS` | 当前同合同 request/wake + qualified Marl/oneTBB bulk holdout；旧苹果/橘子行不参与 gate |

不可变 product candidate LOC：全库代码 `481377`；自实现不含测试 `77120`；Sched 实现 `5626`。
全部适用工程门已经完成，当前进入 `VERIFIED`；下一合法状态是补齐限制、reopen triggers 与 terminal
receipt 后进入 `CLOSED/FROZEN`。
