# sched-v1 当前源码重构与收口

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: BLOCKED -->

> **终态：`BLOCKED`。** 产品候选仍为
> `f358d084d17f92e64ec51ae2bd974d2a67f417d2` / tree
> `44a0d4e0de048bf93d53ec0a1febdfe6135ac6a6`。三平台 correctness/sanitizer、三种 C++ 交付和直接
> consumer compatibility 均通过；但是独立性能红队推翻了 `structured-spawn-join` 的同合同声明：
> HYLE `schedule_task` 承担逐任务 allocation/result/status/cancel/handle，Marl 只承担 caller-owned result
> slots 与一个批级 `WaitGroup`。该行只能作为 `NATURAL_ROUTE` 证据，证明 HYLE 缺少自然的批级轻量
> structured route，不能签 scheduler 实现级 non-inferiority。本轮按预算停止，不回写产品；恢复条件见
> Closure decision。

> 本轮是 `sched` 第一次按当前模块收口协议审查。2026-07 的 `sched-01`--`sched-11`、旧
> `sched-module-closure.md`、架构叙事和 benchmark 只作历史输入，不继承功能、性能、交付或 API
> verdict。当前机械状态只看 [`module-status.json`](../../module-status.json)。进入 `AUDITED` 并冻结有限
> `FIX` 集前，不修改 sched 产品实现。

## Closure card（terminal `BLOCKED`）

- component / owner / round ID：`sched` / scheduling-runtime owner / `sched-v1`；
- task kind：`REFACTOR`；baseline commit：
  `63406dd0bf37ed2eb78baef8008820c64294fd81`；
- 进入原因与 demand ID：`SCHED-V1-NORTHSTAR-01`。HYLE 的首席消费者将是直接源码集成的推理引擎；
  当前 IO completion、Compute GPU/ggml bridge 与 Graph parallel 已真实暴露 sched 类型，但现有 sched 从未
  按当前 consumer-shaped 六问、能力完整性和交付方法签署，旧实现/测试可能由 benchmark 形状驱动；
- 真实 consumer / normal workload：
  1. 线程池上执行短 CPU callback 与可派生任务，包含 fork/join、work stealing、placement 和 shutdown；
  2. IO/timer/GPU completion 挂起并唤醒 fiber/coroutine，包含 timeout 与 cooperative cancellation；
  3. 推理后端或阻塞系统/库调用显式卸载到 blocking pool，再以 typed result/join 回到 scheduler；
  4. fiber 与外部线程之间使用 bounded channel、notification、semaphore 和结构化等待；
- 范围：当前 `include/hyle/sched`、`src/sched` 与 vendored fcontext 产品面；所有公开类型/函数、算法、
  ownership/lifetime/failure/concurrency、直接 HYLE 依赖与 IO/Compute/Graph consumer edge；C++23
  source/static/shared 交付、文档、旧 C/Native FFI 清洁度和当前 build/package 接线；
- 非目标：收口 IO/Compute/Graph；把 graph bulk/parallel algorithm、IO reactor、GPU stream policy 或模型
  执行策略下沉 sched；distributed/actor/process scheduler、抢占式 kill、自动 sysmon、稳定跨工具链 C++
  ABI、Native FFI/C API、全操作/全平台 SOTA、P2 端到端声明和 Formal 证书；
- 直接依赖：中央图 `PUBLIC core,error,time,platform,mem,util_core`，`PRIVATE container`；编译实现还使用
  compile-time Obs trace seam，必须在审计中裁决是否由 trace infrastructure 完整接线。产品第三方输入是
  vendored Boost.Context fcontext `cc1b9e02cd530d6607becc984e1e5bfa4f7b3600`；
- 直接消费者：`io`、`compute`、`graph`，以及 `samples/03_sched_fibers`；这些未收口模块只提供真实
  consumer slice，不借此取得自身 verdict；
- tier 与理由：`P1`。ready dispatch、spawn/join、suspend/wake、work stealing、sync 与 timer/cancel 可能
  高频，但尚无完成的 HYLE 推理引擎 profile 证明端到端贡献，故不得升为 `P2`；
- material margin / resource budget：默认 5% 同合同 non-inferiority；yield/wake/frozen-ready 常态不得新增
  allocation、syscall、全局锁或 per-operation metadata；work-stealing/queue/registration 的复杂度与 bounded
  footprint 在 audit ledger 逐项冻结；blocking/timer 只计其语义要求的 OS wait/clock/ownership 成本；
- 总预算：能力与 public trace 150 分钟；一个冻结产品批次最多 180 分钟；三平台 correctness/sanitizer、
  三交付和文档 150 分钟；P1 资格化与采样最多 150 分钟；红队 45 分钟。总工程不超过一个工作日；超出
  则按能力族拆出新 task，不扩写本轮；
- 异常预算：每个 P1 workload 最多一次反序复现、一次 codegen/profile 机制诊断、一个产品修正和一次
  完整重跑；不为亚纳秒、地址/顺序翻转或未资格化点值改变 API/layout；
- machine profile：`W1-WIN` 为 P1 锁定性能机；`W1-WSL` 为 GCC/sanitizer/ELF；`M1-MAC` 为
  AppleClang/AArch64/Mach-O correctness、交付和结构回放；
- instrument profile：P1 `closure-perf-v1`；correctness/delivery 使用 `closure-correctness-v1`；
- strong peers / current clean identities（2026-08-31 remote checked）：
  - stackful scheduler/fiber/sync：Marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e`；
  - async runtime/wake/blocking/sync：Tokio `ea91b33ca57ff0581b38e735cc108f831bccbdaa`；
  - CPU task/work-stealing/bulk demand：Taskflow `83f90a279aaab33fd4ce1deed00ca07819cbc8f8` 与 oneTBB
    `a8216b95202f3c55f9749870f4f8cfa4df0b32e3`；
  - north-star consumer：llama.cpp `9723942adc518b43c4b95dc4dce6906903eb5e09`、standalone ggml
    `36da57138425487184aa1da2eee2cde155909c6f`；随仓 ggml identity 在 audit 中单列；
- 无同合同 peer 时：以锁定 source consumer 的 throughput、tail latency、allocation/syscall/worker
  utilization 和 shutdown/cancel correctness 预算裁决；禁止把 stackful 与 stackless、caller-participating 与
  asynchronous、弱 failure/lifetime 合同混成胜负；
- 适用 gates：完整 capability-family map 与 public trace；normal/empty/boundary/OOM/move-destroy/
  concurrency/reentrancy/cancel-close/stress；ASan/UBSan/TSan；真实 x86-64 与 AArch64 fcontext；直接
  IO/Compute/Graph consumer slice；source/static/shared selected-component；C++ API/guide；C/FFI/symbol
  清洁度；有限 P1 workload 与独立红队；
- 预期输出：六问直答、能力/缺口 ledger、有限 `FIX` 集、单一 candidate identity、三平台/三交付与
  sanitizer 证据、资格化 P1 结论、architecture/user/performance/implementation 文档、限制/reopen trigger、
  terminal receipt 与四项 LOC；
- 停止条件：所有 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，全部适用门有真实 verdict 且 P1 聚合为
  `PASS` 才可 `CLOSED`；否则以具名 blocker 进入 `BLOCKED`。不把 actor/distributed/preemption、下游模块
  未完成或 Formal `NONE/PENDING` 误写成本轮产品缺口。

`QUEUED -> SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v1\module-status-before-scoped.json`，SHA-256
`F6B7F6D035EC94E693E361B837EC89963E228EE1BFCDCBBBC575637594BD8A9B`。

`SCOPED -> AUDITED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v1\module-status-before-audited.json`，SHA-256
`2966168581C074437E787398E60753865281F7FD0C1E19BEFCAEA1124C0EF0ED`。

`AUDITED -> CANDIDATE` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v1\module-status-before-candidate.json`，SHA-256
`58FC3725C673FE98204DD28D4958EFD0B6C9A3D91120566D7F372436DF0277DE`。

原候选在 P1 验证中退回 `AUDITED` 前的 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v1\module-status-before-performance-return-audited.json`，SHA-256
`5449D000DC825947E90E25735C901DFD25A05BD5D137B27C526E8F90C3125359`。

唯一产品修正完成后再次进入 `CANDIDATE` 前的 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v1\module-status-before-worker-task-candidate.json`，SHA-256
`59D1FF73E40757D33E8BB6C0ACBF0DC60FB14196D862B7ED6D026AC81E38A330`。

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | Coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| stackful execution context | low-level callback/fiber；IO/GPU completion 冻结深调用栈；Boost.Context/Marl | guarded small/default/large/one-off stack；pre-start move；resume/yield/finish | stack owner、guard、pool return、finish-after-switch、真实 x86-64/AArch64 | `Fiber` + Mem `FiberStack/FiberStackPool` + vendored fcontext | `COMPLETE` | started Fiber 不可 move；dispatch mutation 只由内部 accessor/friend 使用；平台证据待本候选重签 |
| general scheduler / placement / lifecycle | 短 CPU callback、可挂起任务、work stealing、外部 inject、IO pin、drain/shutdown；ggml cpumask/poll/priority，Marl affinity/initializer | unpinned default；显式 CPU ids 或 ordered pin；best-effort/required；balanced/latency/power idle；worker hooks | bounded admission、steal/pin、fairness、thread start failure、concurrent shutdown | M:N worker + sharded inject/local FIFO/LIFO + pinned queue | `COMPLETE` | 通用 priority 仍按 SCHED-GEN-01 等真实 serving demand；不保留 benchmark proof 类型 |
| lightweight / owned / structured task | raw zero-allocation callback、自然 lambda、批级 fork/join、逐任务 typed join、detached suspend、blocking syscall/backend | worker-stack callback；批级完成 + caller-owned results；逐任务 worker-stack typed join；stackful detached；`Task<T>`；blocking offload | closure ownership、OOM/throw、batch/individual result、detach/abort/join、shutdown | `schedule`、手写 `schedule + WaitGroup`、`schedule_task`、`spawn_detached`、`spawn_task`、`JoinHandle`、`BlockingPool` | `PARTIAL` | 普通批级完成只能让用户手写 context/result lifetime；`schedule_task` 强制每项购买独立 result/status/cancel/handle，缺少自然且安全的 batch route |
| wake / timer / deadline integration | IO、GPU、timer completion 单次唤醒；sleep/timeout；当前 IO consumer | wake-before-suspend rendezvous；relative duration/absolute monotonic deadline；arm/cancel | timer entry owner、pin episode、timeout-vs-completion winner、shutdown | `Waker` + per-worker Time `TimerQueue` | `COMPLETE` | public 时间边界为 typed Time；IO poll/wake 通过不安装的 detail integration seam |
| cooperative cancellation | request fan-out、deadline/client disconnect、JoinHandle abort、in-flight IO callback；Tokio/Folly | shareable token；child subtree；cross-thread cancel/register/drop；nested sources；O(1) poll | allocator/OOM、token/callback lifetime、reentrancy、destruction、migration | cancel tree + Fiber/worker-stack execution-local flag chain | `COMPLETE` | poll 仍为 O(k) active-source atomic loads；tree/control path 与 callback 生命周期单独串行化 |
| synchronization / notification | fiber 与外部线程的 monitor、countdown、phase barrier、signal、permits、RW access；Marl Event、Tokio/standard barriers | notify one/all；auto/manual event；one-shot latch；reusable barrier；timed semaphore | fiber/thread dual wait、FIFO/fairness、timeout unlink、close | CV、Mutex、WaitGroup、Event、Latch、Barrier、Semaphore、RwLock | `COMPLETE` | timeout unlink 为 O(1)；平台实现仍须本候选 sanitizer/correctness 重签 |
| bounded / latest / broadcast messaging | producer/consumer queue、reply slot、config/watch、fan-out broadcast；Tokio channels | blocking/try/timed/close；single-shot；latest value；lag detection | value exception contract、unsent value、OOM、sender/receiver drop、lag count | `Channel`、oneshot、watch、broadcast | `COMPLETE` | nothrow value 与 recoverable OOM/unsent ownership 已成为显式合同；watch/broadcast copy 由 RAII 保持异常安全 |
| observability / integration | 零成本关闭 trace；IO/Compute/Graph 直接消费 | trace off 零依赖；on 时单一 backend；private integration access | callback/context lifetime、registration/teardown、组件方向 | `HYLE_TRACE` foundation + private raw integration callbacks | `COMPLETE` | trace off 不引入 runtime backend；普通 public surface 与内部 mutation seam 已分层 |
| specialized inference execution | ggml caller-participating SPMD、hot spin barrier、pause/resume；llama serving priority/cancel lane | graph plan、kernel chunking、backend/NUMA/model policy | caller participation、scratch、tensor/backend lifetime | Graph/Compute 或未来 inference runtime | `ROUTED` | 不把 ggml 专用 threadpool、模型 priority 或 GPU policy复制到通用 sched；真实 consumer 证明需要通用 priority 后重开 |
| product delivery / dependencies | direct-source inference engine；installed static/shared C++ SDK | selected closure、private fcontext C/asm seam、same-toolchain C++ import | lock/license/relocation/export/header ownership | central component graph + package registry | `COMPLETE` | 产品接线已完成；本候选 source/static/shared journey 仍须重新执行才取得 gate verdict；无 Native FFI |

## Capability / gap ledger

| ID | class | exact demand / source | normal shape and target | current | action | severity / verification / negative space |
|---|---|---|---|---|---|---|
| SCHED-FIX-01 | `MUST` | 现有跨线程 cancel tests、IO callback、Tokio/Folly cancellation | 可复制 token；tree control path 可跨线程；`is_cancelled` O(1) atomic；所有 active cancel source 可组合；OOM 为 `Result` | `COMPLETE` | `KEEP` | correctness；并发/重入/树 drop/TSan；不得给 poll hot path 加锁或 allocation |
| SCHED-FIX-02 | `MUST` | ggml CPU set/poll/priority；Marl affinity/initializer；通用组合 | unpinned default；显式 ordered/CPU-id placement；required policy；semantic idle policy；noexcept worker hooks | `COMPLETE` | `KEEP` | API/latency/energy；默认无隐式 pin，专用 SPMD/pause 留 Graph/Compute |
| SCHED-FIX-03 | `MUST` | `std::chrono`/HYLE Time、Marl Event、standard/Tokio barrier、现有 timer consumer | `Duration/MonotonicTimePoint`；Event；Barrier；Latch arrive-and-wait；timed Semaphore | `COMPLETE` | `KEEP` | common primitive completeness；O(1) state transitions、timeout O(1) unlink，不复制 peer 独有组合器 |
| SCHED-FIX-04 | `MUST` | 普通 C++ user、Graph 动态 fan-out、Marl/Tokio worker tasks | raw hot seam保留；非挂起 callable 有 typed worker-stack join；可挂起 callable 走默认 stackful route；span join | `COMPLETE` | `KEEP` | `schedule_task` 用单 allocator cell 保存 callable/result state，直接在 worker stack 执行；不会为 benchmark 下沉 graph algorithms |
| SCHED-FIX-05 | `MUST` | 现有公开 messaging 承诺、Tokio ownership shape、C++ exception safety | nothrow value contract或异常安全 RAII；oneshot unsent value；自然 receive/lag diagnostics | `COMPLETE` | `KEEP` | correctness/usability；不因 API 数量重造 actor/stream 系统 |
| SCHED-FIX-06 | `MUST` | allocator charter、Fiber 可复现 move defect、benchmark-only public trace | Scheduler/BlockingPool bookkeeping 走注入 allocator；Fiber 仅 pre-start move；删除 proof RunQueue/Worker；detail integration access | `COMPLETE` | `KEEP` | lifetime/OOM/boundary；OS Thread 自身冷路径 allocation 属 Platform 合同，不在本轮越界 |
| SCHED-FIX-07 | `MUST` | build/package 六问与当前无 guide | Time 改为真实 PUBLIC edge；source/static/shared；private fcontext C/asm；architecture + scenario guide | `COMPLETE` | `KEEP` | delivery/docs；不恢复旧 C ABI 或 Native FFI |
| SCHED-FIX-08 | `MUST` | P1 closure card | request dispatch、Graph-shaped non-suspending typed fan-out/join、completion wake 的有限 consumer replay | `BLOCKED` | `STOP` | 仪器资格通过，但红队判定 structured pair 语义不等价；自然路线还暴露缺少批级 structured route，必须由新 round 重新 scope，不能在本轮继续换形状 |
| SCHED-GEN-01 | `GENERAL` | llama server priority cancel lane；Taskflow/oneTBB priority | 通用 task priority | `OUT_OF_SCOPE` | `DEFER` | owner=Sched+serving；真实 HYLE consumer证明默认/opt-in metadata与 starvation合同后重开 |
| SCHED-GEN-02 | `GENERAL` | ggml paused persistent SPMD pool | pause/resume/caller participation/hot spin barrier | `ROUTED` | `REJECT` | owner=Graph/Compute/inference；通用 scheduler pause 会阻塞 IO/timer progress |

- 现有 public trace：`Fiber/Waker` -> stackful execution/wake；`Scheduler/SchedulerOptions` -> general
  scheduler；`Task/JoinHandle/BlockingPool` -> structured task；timer functions -> wake/timer；
  `CancelToken/Scope/Callback` -> cancellation；`ConditionVariable/Mutex/WaitGroup/Latch/Channel/Semaphore/
  RwLock/oneshot/watch/broadcast` -> sync/messaging。`RunQueue/Worker` 只映射旧 benchmark proof，删除。
- 语法消费者：IO 使用 Waker/Task/CancelToken/timer/BlockingPool；Compute 使用 Waker/BlockingPool；Graph
  使用 spawn/join 并手写动态同型 handle 数组。只迁移这些直接边，不替它们收口。
- 本轮冻结 `FIX` 集仅为 SCHED-FIX-01--08。priority、actor/distributed/preemption、generic reactor、
  graph bulk/SPMD、GPU/model policy 均不顺手加入。
- Windows shared baseline（产品改动前）：VS 2026 / MSVC 19.50，`HYLE_COMPONENTS=sched` 解析为
  `core,container_intrusive,util_core,error,time,platform,mem,container,sched`；build 成功，
  `hyle_sched_test` 1/1 通过（6.13s）。这只是 baseline，不是 candidate verdict。

## Owner 六问直答（`AUDITED` 快照）

### 1. 第三方产品路径

| Dependency / identity | route / class | owner split / public leak |
|---|---|---|
| Boost.Context fcontext `cc1b9e02cd530d6607becc984e1e5bfa4f7b3600` / Boost 1.92 | direct vendored product mechanism | Sched owns Fiber semantics；Boost owns asm context switch；Build owns lock/install/license；private C/asm symbols only |
| mimalloc `18b08671c9302247bfb682286e6bf3cc1773f801` / v3.5.0 | indirect product through Mem | Mem owns allocator semantics；no `mi_*` public leak；Sched still owns correct use of injected allocator |
| Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c`、simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`、xxHash `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53`、wyhash `e4764a0b637d34d3421a7760affada9288b625a8` | indirect product through Container | Container owns public queue/value semantics与 dispatch；Sched只消费 HYLE queue types；无 vendor public type leak |
| Tracy `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c` / v0.14.1 | conditional product when `HYLE_TRACE=ON` | cross-cutting trace foundation owns backend；off path compiles away；不成为 Sched API |
| GoogleTest / nanobench / Marl / Tokio / Taskflow / oneTBB | test / benchmark / peer only | 不进入 shipping sched product |

标准库 atomics/mutex/CV/coroutine 与 OS thread/wait API 通过 Platform/standard contract；不是第三方模块。

### 2. 推理引擎能力完整度

- llama.cpp：clean `9723942adc518b43c4b95dc4dce6906903eb5e09`，2026-08-31 remote confirmed；其随仓
  ggml slice 绑定同一 commit。
- standalone ggml：clean `36da57138425487184aa1da2eee2cde155909c6f`，2026-08-31 remote confirmed。
- ggml 的 CPU mask、priority、poll 和 reusable barrier 形成 SCHED-FIX-02/03；caller-participating graph
  compute、pause/resume、chunk counter 与 scratch ownership 路由 Graph/Compute/inference runtime。
- llama server 的前置取消/高优先队列目前是 serving policy；未给所有 Fiber 增加常态 priority metadata。
- 当前仍有 SCHED-FIX-01--07 的未处置 `MUST`，所以审计时答案是 **YES**；candidate 完成前不能声称功能
  完整。不是按 ggml API 求并集，也不承诺未来永不新增。

### 3. 拟真、公平性能

- tier `P1`；normal tasks 已冻结为短 callback、spawn/join、suspend/wake、外部 inject/steal、常用 sync、
  cancellation poll/control 与 bounded channel。
- 当前只有产品改动前 baseline 和历史 benchmark；旧结果未按本轮 candidate、相同语义/计费边界、
  `closure-perf-v1` 资格化，因此 verdict 为 **INCONCLUSIVE（尚未进入 candidate 验证）**。
- 目标是 W1-WIN、5% non-inferiority；只能对实际测量 workload 作有限 claim，不声明全操作、全平台或
  端到端 inference SOTA。

### 4. 旧 C ABI / Native FFI 清洁度

- 全局 Native FFI policy 为 `N/A`；当前 public tree 无 sched C header/schema/bridge/sample/option/target。
- `src/sched/fcontext.h` 的 `extern "C"` 与 vendored asm 是 private Boost.Context ABI 边界；shared library 的
  decorated C++ exports 不是 C ABI。旧任务里 114 个 C symbol 的文字只属历史。
- 审计 verdict：`CLEAN`；candidate 后以 symbol/header/package scan 重签。

### 5. Source / static / shared 构建与导入

- 当前 Windows shared selected-component baseline build/test 通过；这不是三交付最终 verdict。
- source embedding、installed static、installed shared 都在 SCHED-FIX-07 后重跑；Time typed API 将使
  `time` 从 PRIVATE 改为真实 PUBLIC edge，删掉 proof headers 后核对 selected install。
- 当前 verdict：`BLOCKED` 于尚未产生 candidate，而不是把 producer build 冒充 consumer import。

### 6. C++ API 易用性与文档

- raw low-level hot seam 有价值，但普通用户当前仍需手工 callback/context、无单位整数 deadline、动态 join
  循环和公开 internal mutation；消息模板还有失败/异常合同错位。
- `docs/user` 没有 sched 场景指南；architecture 仍是旧 ticket/benchmark 叙事。
- 当前 verdict：`FAIL`；SCHED-FIX-02--07 修复后必须由“何时选 schedule/detached/task/blocking、何时选
  Event/Barrier/channel、ownership/取消/关闭陷阱”旅程重签。

## 原候选与性能审计退回

原产品候选曾冻结为 commit `9f24714190c7247abe46a4f030f27a60101fe90d`、tree
`01f20256f201a366bc094a078e2348dee670bb89`。其公开合同、算法、normal workload 与 P1 tier 不再扩张；
但冻结 replay 已证明 SCHED-FIX-04 的 typed non-suspending 路线没有达到同成本档，因此本轮按协议由
`CANDIDATE` 退回 `AUDITED`。以下是原候选已经完成、可在新候选复验后继承结论的实现批：

- SCHED-FIX-01 实现批：取消 token 改为可共享、tree control path 跨线程串行化，保留 O(1) atomic
  `is_cancelled()`；新增 recoverable OOM；嵌套 cancel source 与 `JoinHandle::abort` 组合；Fiber 禁止启动后
  move。Windows selected sched build 与 `hyle_sched_test` 1/1 通过（6.21s）；
- SCHED-FIX-02 与 SCHED-FIX-06 allocator 子项实现批：默认 Scheduler 不再隐式硬绑 CPU；显式 ordered
  CPU/CPU-id set、best-effort/required、balanced/low-latency/power-saving idle policy 和 worker lifecycle
  hooks 已进入 public C++ shape；Scheduler Pimpl、Worker 与 allocator-aware bookkeeping 使用注入的
  `mem::Allocator`。invalid option、strict placement、三种 idle policy 与 hook lifecycle 已补边界测试；
  Windows selected sched build 与 `hyle_sched_test` 1/1 通过（6.17s）。
- SCHED-FIX-03 实现批：公开 timer、sleep、drain、ConditionVariable、Channel 与 BlockingPool 的时间边界
  改用 `time::Duration/MonotonicTimePoint`；新增 fiber-aware auto/manual-reset Event、可复用 Barrier、
  `Latch::arrive_and_wait()` 与定时 Semaphore，timeout waiter 维持 O(1) unlink。直接 IO deadline consumer
  已迁移；Windows selected sched 与 IO build 成功，`hyle_sched_test`、`hyle_io_test` 分别 1/1 通过；
- SCHED-FIX-04 实现批：保留 raw callback 零分配 hot seam，同时新增 allocator-owned、`noexcept`、可捕获
  callable 的自然 `schedule/spawn_detached/spawn_detached_on` 路线与默认 fiber stack；同型动态任务组可用
  `std::span<JoinHandle<T>>` 做 `when_all/when_any`，不再要求 Graph consumer 手写 parameter-pack 形状。
  Windows selected sched build 与 `hyle_sched_test` 1/1 通过；
- SCHED-FIX-06 BlockingPool allocator 子项实现批：Pimpl、thread record、allocator-aware bookkeeping 与
  shutdown 暂存均使用注入 allocator；构造参数与 OOM 可经 `Status` 恢复，保留显式 fail-fast convenience。
  invalid/OOM 与 allocation/free 对账测试通过。OS Thread 自身的冷路径分配仍属于 Platform 合同。
- SCHED-FIX-05 实现批：Channel 新增 `Result<T>` 的 blocking/try/timed receive，同时保留可复用调用方
  storage 的低层 overload；bounded send 的 full/timeout/closed/invalid error 返还未发送值，普通构造
  fail-fast、显式 `Status*` 构造提供 recoverable validation/OOM；oneshot send failure 同样返还原值；
  broadcast receive error 暴露精确 missed count。ownership move/destruction 采用明确 nothrow value
  contract；watch/broadcast 可能抛出的 copy 由 RAII monitor guard 保证异常后可继续使用且 cursor 不前移，
  mutation callback 则显式要求 `noexcept`。Windows selected sched build 与 `hyle_sched_test` 1/1 通过。
- SCHED-FIX-06/07 边界实现批：删除不属于真实 Scheduler 实现的 public `RunQueue/Worker` proof、对应
  自身 benchmark 与重复测试；IO poll/wake mutation 移入不安装的 `sched/detail` build-only bridge，普通
  C++ 用户只通过 `io::IoContext` 组合。删除 proof header 后，中央图暴露出 `container` 只是真实 PRIVATE
  实现边，已同步修正；Fiber 的 wait/pin/finish/rendezvous mutation 同样改为 private，sched 源码经
  source-only accessor、structured task 经具名 friend 使用，用户仍保留 trace label。Windows selected
  sched/IO 重新 configure/build，`hyle_sched_test` 与 `hyle_io_test` 分别 1/1 通过。
- SCHED-FIX-07 API/文档与 source journey 实现批：architecture 已从旧 ticket/点值叙事收敛为长期
  runtime、任务、wake、placement、lifetime、依赖和复杂度决策；新增场景化 C++ 用户指南，覆盖
  schedule/detached/task/blocking、取消/shutdown、同步/消息 ownership 与常见陷阱；公开头移除过期
  milestone 叙事，`samples/03_sched_fibers` 改走默认栈和 owning lambda 正常入口。新增的独立 package
  consumer 实际覆盖 callback、Fiber yield、typed sleep、Task/join、Channel 与 BlockingPool；Windows
  selected-component smoke 的 source embedding、搬迁后 installed shared 和搬迁后 installed static
  configure/build/run 全部通过。最终 candidate 若不再改变产品 tree，可继承该批 delivery 证据；若改变
  公开头、依赖或 artifact 则必须重跑。

上述单测是实现批回归，不是 SCHED-FIX-01--06 的 terminal `PASS`，也没有把尚未运行的 Linux
CPU-set 语义、TSan/ASan 或三交付写成通过。

当前 IO、Compute 与 Graph 的 Windows CPU library targets 已在关闭昂贵可选 backend 后全部编译通过，
只证明候选 sched API 没有打断现有直接 consumer slice；三个下游模块的功能、正确性和性能状态不因此
改变，Compute/Graph 仍为 `QUEUED` 半成品。

### P1 机制裁决（2026-08-31）

- 冻结 workload 是四个固定物理核上的 request dispatch、当前 Graph 自动 grain floor（1024 elements）
  对应的非挂起 typed fan-out/join，以及外部 completion wake；candidate 与 Marl 分进程、双顺序执行。
- probe 的 request row 中 HYLE/Marl 吞吐比为 1.63--1.86，completion row 为 1.00--1.06；这两项没有
  material 劣势。structured row 双顺序分别为 0.7075 与 0.7052，稳定超出 5% non-inferiority margin。
- Marl 普通 `Task` 直接在已有 worker/fiber 上调用 closure，仅在任务阻塞时才取得另一 Fiber；HYLE
  `spawn_task` 对每个即刻完成的叶任务仍分配 coroutine frame、`JoinState`、`TaskDriver`，取得 guarded
  stack、构造 fcontext 并切换上下文。旧 raw `schedule` 已证明 queue/dispatch 本身不是原因。
- 这不是为 benchmark 添加捷径：公共指南此前迫使任何需要 typed result/join 的非挂起工作走完整
  stackful route，而 Graph 当前恰好是该常见消费形状。冻结修正仅新增 worker-stack typed join 路线；
  `spawn_task` 继续拥有可挂起 coroutine/Fiber 语义，benchmark 不取得专用入口。
- 反序复现与一次机制诊断预算已用完。只允许一次上述产品修正和一次完整 P1 重跑；若仍稳定 material
  `FAIL`，本轮以具名性能 blocker 进入 `BLOCKED`，不再改变 workload 或扩建中央仪器。

### 新候选冻结（2026-08-31）

- 产品 commit `f358d084d17f92e64ec51ae2bd974d2a67f417d2`，tree
  `44a0d4e0de048bf93d53ec0a1febdfe6135ac6a6`；公开合同、算法、normal workload 与 P1 tier 自此冻结。
- `schedule_task` 对 `noexcept`、不挂起的 callable 返回与 `spawn_task` 共用的 `JoinHandle<T>`，callable、
  result state 与协作取消 flag 合并为一个调用方 allocator cell；worker 直接调用，不创建 coroutine frame、
  guarded stack 或 fcontext。会挂起/阻塞的工作仍分别走 `spawn_task`/`spawn_blocking`。
- completion group 使用 atomic target + manual-reset Event，只在目标完成时唤醒；直接 join 仅在确有 waiter 时
  notify。算法仍为 O(n) install/clear、O(1) completion transition，等待期间无 polling。
- cancellation flag chain 在 Fiber 上随执行迁移，在非 Fiber 的 Scheduler worker callback 中使用
  thread-local execution scope；外部普通线程不会被误判成 sched execution context。
- Windows Release：`hyle_sched_test` 1/1 通过（6.34s），`hyle_io_test` 1/1 通过（0.07s），直接源码
  selected-component package consumer configure/build/run 通过；P1 candidate target 编译通过。上述是候选
  定向回归，不替代 WSL/macOS、sanitizer、static/shared 与正式 P1 verdict。
- harness/provenance commit 为 `67a3619a`；consumer provenance SHA-256
  `76a572e0e653f05f610b8b0d411b06ec850e2e0ca16cd202a3242fc1bfe17c98`，P1 contract SHA-256
  `dae15db50195b0906d741bb8b35df623e2d0f5d18e129351b60c3bcfe132ca52`。

## Verification manifest

冻结产品身份从 `f358d084d17f92e64ec51ae2bd974d2a67f417d2` 到 terminal task 没有任何
`include/hyle/sched`、`src/sched`、Sched package sample 或 Sched 测试漂移。验证结果如下：

| Gate / risk | Verdict | Evidence | Limit |
|---|---|---|---|
| Windows Release sched correctness | `PASS` | selected `hyle_sched_test` 1/1（6.34s） | W1-WIN / MSVC 19.50 |
| direct IO consumer | `PASS` | selected `hyle_io_test` 1/1（0.07s） | 只证明 Sched edge，不关闭 IO |
| WSL Release / ASan / UBSan / TSan | `PASS` | 1/1（1.24s）/ 1/1（3.37s）/ 1/1（1.75s）/ 161 tests（5.617s） | GCC 14.2；TSan 由 `setarch -R` 运行 |
| M1-MAC correctness | `PASS` | AppleClang 21.0 arm64 Mach-O fcontext，1/1（4.16s） | disposable canonical sync；Unix Makefiles |
| public headers | `PASS` | 9 个 Sched public headers self-contained | 当前 selected closure |
| source/static/shared C++ delivery | `PASS` | `C:\hyle-scratch\sched-v1\delivery-candidate-0829` source embedding、relocated static/shared consumer 全部运行通过 | Windows selected component |
| Graph/Compute syntax consumer | `PASS` | Windows selected CPU targets 编译通过 | 两者仍 `QUEUED`，不继承功能 verdict |
| C/Native FFI cleanliness | `PASS` | public tree 无 C/FFI surface；仅 `src/sched/fcontext.h` 保留 private Boost.Context `extern "C"` | decorated C++ shared exports 不是 C ABI |
| docs/usability | `PASS` | architecture、scenario user guide 和 package sample 已覆盖四类既有路线 | batch route 缺口成为新 round 输入 |
| P1 engineering performance | `INCONCLUSIVE` | 仪器合格；两行同合同结论有效，一行在红队中变为 `NATURAL_ROUTE/NOT_COMPARABLE` | 不能聚合为 same-contract P1 PASS |
| independent red team | `FAIL` | `SCHED-RT-01` 发现 structured 语义账单不等价与产品档位缺口 | 当前 round 的 terminal blocker |

正式 bundle 位于 `C:\hyle-scratch\sched-v1\p1-final-evidence`：contract SHA-256
`dae15db50195b0906d741bb8b35df623e2d0f5d18e129351b60c3bcfe132ca52`，consumer provenance
SHA-256 `76a572e0e653f05f610b8b0d411b06ec850e2e0ca16cd202a3242fc1bfe17c98`；`manifest.json`
`658F7F31A6D91B35322AF80FDD8E395D410F6C417F2039870274ACC8B71BC915`，`summary.json`
`70F2906196341D47E1E448156057E8FAE2E81E75C9AA8DFA2A657A7AFDFD2C56`，`controls.json`
`F622C3AE9B7290FD789D831FF9A99000B15121C5B1625ED96FB6D020DEC6BC93`，`raw-log-index.json`
`1C442CD43D1BB62573D4D4A80FEEBEB6D41A70421AA0A241886882FDB00482EA`。deep verifier、artifact、
build-contract、A/A、5%/10% known delta 与 timer floor 全部通过；这些事实证明计时器能辨别 material
差异，不证明 benchmark 两侧语义相同。

| Workload | Revised class / comparability | Qualified observation | Valid conclusion |
|---|---|---|---|
| request-fanout-collect | `SAME_CONTRACT / COMPARABLE` | HYLE 双顺序 paired median advantage 1.489 / 1.522 | 5% NI `PASS`，双顺序 `WIN` |
| completion-wake | `SAME_CONTRACT / COMPARABLE` | 1.013 / 1.017 | 5% NI `PASS`，双顺序 `PARITY` |
| structured-spawn-join | `NATURAL_ROUTE / NOT_COMPARABLE` | candidate-first paired 0.962，peer-first 0.910；后者绝对中位 1.598M vs 1.830M op/s | 当前自然 HYLE route 有 material 风险并缺产品档位；不得归因成 scheduler 同合同慢 9%--12.7% |

候选 LOC 由 `python scripts/loc-report.py --module sched --commit
f358d084d17f92e64ec51ae2bd974d2a67f417d2 --self-check --json` 从 immutable tree 复算：全库代码
`408169`，自实现包含测试 `98581`，自实现不包含测试 `61337`，Sched 实现 `5262`。

## 独立红队

- reviewer / date / budget：独立 terminal review / 2026-08-31 / 30 分钟；
- `SCHED-RT-01`（high / contract + product）：`structured-spawn-join` 把 HYLE 每任务独立
  allocator cell、typed result、Status、cancel flag、JoinHandle 与 group install/clear，同 Marl caller-owned
  slots + 单批 WaitGroup 写成 `COMPARABLE`。裁决：`accepted`；仪器结果重分类，不删除 raw evidence；
- 真实使用复核：当前 Graph level route 确实需要一批 non-suspending node work 和 caller-owned outcome，
  但 HYLE 只提供手写 `schedule + WaitGroup` 或更重的 `schedule_task`。该缺口同时由 Marl WaitGroup、
  oneTBB task_group/parallel family 和推理 graph fan-out 证明，不是 benchmark-only API；
- correctness/lifetime/build findings：未发现冻结产品新增 blocker；三平台 sanitizer、private fcontext ABI、
  selected component 和三交付证据保持有效；
- 性能有效性：request 与 completion 的同合同结论保留；structured 的 runner `FAIL` 只保留为原合同输出，
  聚合 engineering verdict 因语义前置门失败改为 `INCONCLUSIVE`。

## Terminal owner 六问

1. **第三方：**Boost.Context fcontext 是直接 private product mechanism；mimalloc 经 Mem、Highway/
   simdutf/xxHash/wyhash 经 Container 间接进入；Marl 只作 peer。所有锁定身份与 owner 仍见上方审计表。
2. **功能：**`NO`。既有 fiber、wake、cancel、sync、messaging、blocking 与逐任务 structured route 已完成；
   但普通批级 non-suspending fork/join 缺自然 public route，当前能力族为 `PARTIAL`。
3. **性能：**request dispatch 为 qualified `WIN`，completion wake 为 qualified `PARITY`；批级 structured
   路线只有 qualified `NATURAL_ROUTE` material 风险，没有有效 same-contract verdict；聚合为
   `INCONCLUSIVE/BLOCKED`，不声明全 sched SOTA。
4. **旧 C ABI / Native FFI：**`CLEAN`。private vendored fcontext C/asm seam 不构成产品 FFI。
5. **三交付：**source embedding、relocated installed static/shared C++ consumer 均 `PASS`；直接
   IO/Compute/Graph compatibility slice 通过但不关闭下游。
6. **C++ API/文档：**既有路线文档与接口可用；“无结果短任务、逐任务 result/control、可挂起任务”
   边界清楚，但批级完成的安全自然形状缺失，因此整体 verdict 为 `FAIL/BLOCKED`。

## Closure decision

- 各门：scope/architecture/correctness/delivery/docs 为 `PASS`；engineering performance 为
  `INCONCLUSIVE`；red team 为 `FAIL`；因此不能进入 `VERIFIED/CLOSED`；
- 唯一 blocker：`SCHED-RT-01`，owner 为 Sched。普通批级 non-suspending fork/join 没有安全、自然且
  与其语义成本相称的 public route，旧 structured peer pair 又不能提供 same-contract verdict；
- resume condition：创建新的 `RESUME` round，先冻结 fire-and-forget、批级 structured、逐任务 typed
  control 与可挂起任务四个语义账单；实现 batch-level route 后，以 HYLE batch 对 Marl batch 做
  `SAME_CONTRACT`，逐任务 route 只与等价 adapter/peer 比较或使用预冻结 resource budget；
- 已知限制：冻结 v1 候选可正确使用，但 `schedule_task` 不应被推荐给只需批级完成与 caller-owned result
  的大量短任务；旧 structured 点值不是 scheduler 算法证书；
- reopen triggers：上述 resume condition；或公开任务/lifetime、dependency/delivery 变化；或同合同正常
  workload 暴露新的 material correctness/performance 事实；
- terminal raw evidence 与 LOC 如上；完整 end/registry anchor 由后续 registry 提交记录；
- 建议机械状态：`BLOCKED`。
