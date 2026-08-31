# sched-v1 当前源码重构与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮是 `sched` 第一次按当前模块收口协议审查。2026-07 的 `sched-01`--`sched-11`、旧
> `sched-module-closure.md`、架构叙事和 benchmark 只作历史输入，不继承功能、性能、交付或 API
> verdict。当前机械状态只看 [`module-status.json`](../../module-status.json)。进入 `AUDITED` 并冻结有限
> `FIX` 集前，不修改 sched 产品实现。

## Closure card（`AUDITED`）

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
- 直接依赖：中央图 `PUBLIC core,error,platform,mem,container,util_core`，`PRIVATE time`；编译实现还使用
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

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | Coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| stackful execution context | low-level callback/fiber；IO/GPU completion 冻结深调用栈；Boost.Context/Marl | guarded small/default/large/one-off stack；pre-start move；resume/yield/finish | stack owner、guard、pool return、finish-after-switch、真实 x86-64/AArch64 | `Fiber` + Mem `FiberStack/FiberStackPool` + vendored fcontext | `PARTIAL` | 已启动 Fiber 仍可公开 move，fcontext trampoline 保留旧 `this`，是可复现 lifetime 缺陷；大量 dispatch mutation seam 暴露为普通 public API |
| general scheduler / placement / lifecycle | 短 CPU callback、可挂起任务、work stealing、外部 inject、IO pin、drain/shutdown；ggml cpumask/poll/priority，Marl affinity/initializer | unpinned default；显式 CPU ids 或 ordered pin；best-effort/required；balanced/latency/power idle；worker hooks | bounded admission、steal/pin、fairness、thread start failure、concurrent shutdown | M:N worker + sharded inject/local FIFO/LIFO + pinned queue | `PARTIAL` | 当前默认悄悄单 CPU 硬绑；不能表达显式 CPU set/严格策略；固定 4096 spin；没有 worker start/stop hook；`RunQueue/Worker` 是 benchmark proof 而非真实实现 |
| lightweight / owned / structured task | raw zero-allocation callback、自然 lambda、detached suspend、typed join、dynamic fan-out、blocking syscall/backend | worker-stack callback；stackful detached；`Task<T>`；homogeneous range/heterogeneous pack；blocking offload | closure ownership、OOM/throw、detach/abort/join、shutdown | `schedule`、`spawn_detached`、`spawn_task`、`JoinHandle`、`BlockingPool` | `PARTIAL` | 普通 C++ lambda 无自然 route；detached 无默认栈；`when_all/any` 只有 parameter pack，Graph 手写动态 handle 数组；BlockingPool construction/metadata 绕过注入 allocator |
| wake / timer / deadline integration | IO、GPU、timer completion 单次唤醒；sleep/timeout；当前 IO consumer | wake-before-suspend rendezvous；relative duration/absolute monotonic deadline；arm/cancel | timer entry owner、pin episode、timeout-vs-completion winner、shutdown | `Waker` + per-worker Time `TimerQueue` | `PARTIAL` | C++ public API 仍暴露无单位 `uint64_t ns`；IO poll/wake 与 stack-pool wiring 作为普通 Scheduler API 泄漏，而不是具名 integration seam |
| cooperative cancellation | request fan-out、deadline/client disconnect、JoinHandle abort、in-flight IO callback；Tokio/Folly | shareable token；child subtree；cross-thread cancel/register/drop；nested sources；O(1) poll | allocator/OOM、token/callback lifetime、reentrancy、destruction、migration | cancel tree + callback + fiber-local flag | `PARTIAL` | tree/link/callback 写操作宣称 same-worker，但公开 API、IO 和现有测试跨线程使用；token 不可共享；invalid-on-OOM；单 cancel slot 使 `CancelScope` 屏蔽 `JoinHandle::abort` |
| synchronization / notification | fiber 与外部线程的 monitor、countdown、phase barrier、signal、permits、RW access；Marl Event、Tokio/standard barriers | notify one/all；auto/manual event；one-shot latch；reusable barrier；timed semaphore | fiber/thread dual wait、FIFO/fairness、timeout unlink、close | CV、Mutex、WaitGroup、Latch、Semaphore、RwLock | `PARTIAL` | 通用 Event 与 reusable Barrier 缺失；Latch 缺 arrive-and-wait；Semaphore 缺 timed acquire；历史注释仍把未验证 macOS 结论写进当前头 |
| bounded / latest / broadcast messaging | producer/consumer queue、reply slot、config/watch、fan-out broadcast；Tokio channels | blocking/try/timed/close；single-shot；latest value；lag detection | value exception contract、unsent value、OOM、sender/receiver drop、lag count | `Channel`、oneshot、watch、broadcast | `PARTIAL` | 多个 `noexcept` 模板没有 nothrow 类型约束；手写 lock 在抛出时会永久锁住；oneshot 文档承诺失败返还原值但 `Result<void>` 丢值；Channel 只给 output 参数且 OOM 路线不醒目；broadcast 丢失可编程 lag count |
| observability / integration | 零成本关闭 trace；IO/Compute/Graph 直接消费 | trace off 零依赖；on 时单一 backend；private integration access | callback/context lifetime、registration/teardown、组件方向 | `HYLE_TRACE` foundation + raw integration callbacks | `PARTIAL` | trace 接线本身正确；普通 public surface 与 internal seam 未分层，旧 milestone 注释掩盖当前合同 |
| specialized inference execution | ggml caller-participating SPMD、hot spin barrier、pause/resume；llama serving priority/cancel lane | graph plan、kernel chunking、backend/NUMA/model policy | caller participation、scratch、tensor/backend lifetime | Graph/Compute 或未来 inference runtime | `ROUTED` | 不把 ggml 专用 threadpool、模型 priority 或 GPU policy复制到通用 sched；真实 consumer 证明需要通用 priority 后重开 |
| product delivery / dependencies | direct-source inference engine；installed static/shared C++ SDK | selected closure、private fcontext C/asm seam、same-toolchain C++ import | lock/license/relocation/export/header ownership | central component graph + package registry | `PARTIAL` | 当前 build 基线成立；删改 public seam、Time public edge和 allocator artifact 后必须重新验证三种 journey；无 Native FFI |

## Capability / gap ledger

| ID | class | exact demand / source | normal shape and target | current | action | severity / verification / negative space |
|---|---|---|---|---|---|---|
| SCHED-FIX-01 | `MUST` | 现有跨线程 cancel tests、IO callback、Tokio/Folly cancellation | 可复制 token；tree control path 可跨线程；`is_cancelled` O(1) atomic；所有 active cancel source 可组合；OOM 为 `Result` | `PARTIAL` | `FIX` | correctness；并发/重入/树 drop/TSan；不得给 poll hot path 加锁或 allocation |
| SCHED-FIX-02 | `MUST` | ggml CPU set/poll/priority；Marl affinity/initializer；通用组合 | unpinned default；显式 ordered/CPU-id placement；required policy；semantic idle policy；noexcept worker hooks | `PARTIAL` | `FIX` | API/latency/energy；默认无隐式 pin，专用 SPMD/pause 留 Graph/Compute |
| SCHED-FIX-03 | `MUST` | `std::chrono`/HYLE Time、Marl Event、standard/Tokio barrier、现有 timer consumer | `Duration/MonotonicTimePoint`；Event；Barrier；Latch arrive-and-wait；timed Semaphore | `PARTIAL` | `FIX` | common primitive completeness；O(1) state transitions、timeout O(1) unlink，不复制 peer 独有组合器 |
| SCHED-FIX-04 | `MUST` | 普通 C++ user、Graph 动态 fan-out、Marl/Tokio worker tasks | raw hot seam 保留；allocator-owned noexcept callable convenience；默认 detached stack；span join | `PARTIAL` | `FIX` | 易用性；raw path零 allocation，convenience 明示一次 allocation；不下沉 graph algorithms |
| SCHED-FIX-05 | `MUST` | 现有公开 messaging 承诺、Tokio ownership shape、C++ exception safety | nothrow value contract或异常安全 RAII；oneshot unsent value；自然 receive/lag diagnostics | `PARTIAL` | `FIX` | correctness/usability；不因 API 数量重造 actor/stream 系统 |
| SCHED-FIX-06 | `MUST` | allocator charter、Fiber 可复现 move defect、benchmark-only public trace | Scheduler/BlockingPool bookkeeping 走注入 allocator；Fiber 仅 pre-start move；删除 proof RunQueue/Worker；detail integration access | `PARTIAL` | `FIX` | lifetime/OOM/boundary；OS Thread 自身冷路径 allocation 属 Platform 合同，不在本轮越界 |
| SCHED-FIX-07 | `MUST` | build/package 六问与当前无 guide | Time 改为真实 PUBLIC edge；source/static/shared；private fcontext C/asm；architecture + scenario guide | `PARTIAL` | `FIX` | delivery/docs；不恢复旧 C ABI 或 Native FFI |
| SCHED-FIX-08 | `MUST` | P1 closure card | callback、spawn/join、yield/wake、external inject/steal、sync/cancel/channel 的有限 consumer-shaped workload | `MISSING` | `FIX` | `closure-perf-v1`、5% NI、W1-WIN；旧 milestone 点值不继承，不追亚纳秒/地址幽灵 |
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

## Owner 六问直答

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

## Candidate

尚未冻结完整 candidate；`AUDITED` 已冻结 SCHED-FIX-01--08，当前只按该集合实施。已完成但仍须随
最终 candidate 做跨平台、sanitizer、交付和红队复签的批次如下：

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
  storage 的低层 overload；oneshot send failure 返还原始未发送值；broadcast receive error 暴露精确
  missed count。ownership move/destruction 采用明确 nothrow value contract；watch/broadcast 可能抛出的
  copy 由 RAII monitor guard 保证异常后可继续使用且 cursor 不前移，mutation callback 则显式要求
  `noexcept`。Windows selected sched build 与 `hyle_sched_test` 1/1 通过（6.28s）。

上述单测是实现批回归，不是 SCHED-FIX-01--06 的 terminal `PASS`，也没有把尚未运行的 Linux
CPU-set 语义、TSan/ASan 或三交付写成通过。

## Verification manifest

尚未进入候选验证；未运行项不得写为 `PASS`。

## 独立红队

等待冻结 candidate。

## Closure decision

当前状态为 `AUDITED`；能力地图、public trace 与有限 FIX ledger 已冻结。下一合法状态是在全部产品
修复和直接 consumer 迁移后进入 `CANDIDATE`，或以具名 blocker 停止。
