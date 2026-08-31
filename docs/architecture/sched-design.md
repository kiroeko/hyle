# HYLE sched 架构

`sched` 是单进程 CPU runtime：用少量 OS worker 承载大量可协作挂起的工作，并把 timer、I/O、GPU
等完成事件统一成一次 `Waker` 唤醒。它服务推理引擎的请求并发、异步 completion、后台阻塞工作和
动态 fan-out，也服务普通工程程序；它不拥有模型、tensor、kernel、网络协议或 serving policy。

本页只记录长期需求、边界和设计理由。当前施工状态与证据以
[module-status](../implementation/status.md)及其 closure task 为准；用户应从
[sched 使用指南](../user/sched.md)选择公开入口。

## 产品边界

Sched 拥有：

- stackful Fiber、M:N worker、work stealing、显式 worker placement 和 idle policy；
- 短 callback、固定 grain 的有限 bulk execution、worker-stack typed task、可挂起 detached/structured
  task 与 blocking offload；
- suspend/wake、typed monotonic sleep/deadline、协作取消和两阶段 shutdown；
- fiber/thread 双模同步原语，以及 bounded、oneshot、watch、broadcast 消息形状；
- scheduler 需要的 allocator、错误和生命周期合同。

Sched 不拥有：

- graph/DAG 拆分、自适应 partitioner、kernel grain policy、caller-participating SPMD、模型 priority 或
  KV/scratch policy；
- IO reactor/proactor、协议状态机、GPU stream/event policy；
- distributed/actor/process scheduler、抢占式 kill 或自动检测阻塞调用的 sysmon；
- Native FFI、稳定跨工具链 C++ ABI，或把所有标准/peer API 求并集。

专用推理执行策略留给 Graph/Compute/未来 inference runtime。只有真实消费者证明某种机制可以脱离
模型和 backend 独立复用时，才把它提升为 sched 的通用能力；不能为了复刻 ggml threadpool 或一行
benchmark 给每个普通任务增加常态 metadata。

## 一种执行实体，一张唤醒网

Sched 不把 thread、fiber、coroutine 和 process 假装成同一种对象：

| 概念 | 角色 |
|---|---|
| OS worker thread | 执行引擎；一个 worker 依次运行多个 Fiber |
| stackful Fiber | 可调度实体；可在任意调用深度协作挂起，并在挂起期间被迁移 |
| C++ coroutine | 需要挂起的 Fiber 内 typed result/await 写法；不是每个 typed task 的必缴成本 |
| process | 隔离边界；跨进程工作通过 IO/共享内存形成 completion，不共享 Fiber 指针或 run queue |

统一的是 completion：

```text
timer / IO / GPU / channel notification
                  |
                  v
          Waker rendezvous
                  |
                  v
        ready queue -> worker -> Fiber resumes
```

`suspend()` 与 `Waker::wake()` 使用对称 rendezvous，解决 wake-before-suspend：无论 completion 先到还是
worker 先离开 Fiber，恰好一方负责重新注入，不能丢唤醒或并发 resume 同一栈。一个 Fiber 同时只能在
一个线程上运行；仅在 suspended 状态才允许迁移。

## 任务形状

公共任务入口按工作语义分层，而不是按 benchmark 分层：

| 形状 | 运行位置 | ownership / 成本 | 适用情况 |
|---|---|---|---|
| `schedule` | worker 自身栈 | raw 入口零分配；owning callable 一次 allocator record | 很短、不会挂起或阻塞、无需结果 |
| `bulk_execute` | 至多 worker 数个 worker-stack runner | callable/results 借用到同步返回；O(1) batch state、O(worker) queue entries、无逐项 allocation/handle | 有限 indexed/chunk work，只需要 batch completion/首错/取消 |
| `schedule_task` + `JoinHandle<T>` | worker 自身栈 | callable/result 合并为一个 allocator cell；协作 abort | 很短、不会挂起或阻塞、需要 typed result/join |
| `spawn_detached` | 独立 guarded Fiber 栈 | runtime 持有并回收 callable/Fiber/stack | 会 yield/sleep/等待，但调用方不 join |
| `spawn_task` + `JoinHandle<T>` | 独立 Fiber + coroutine frame | allocator-backed frame/driver/result/stack；协作 abort | 会挂起且需要结果、join/select/fan-out |
| `spawn_blocking` + `BlockingPool` | 专用 OS thread pool | 独立有界 runtime metadata；结果走 JoinHandle | 阻塞 syscall、外部同步 API、长 CPU 工作 |

Raw function-pointer/context seam 保留给已经拥有稳定 storage、需要避免 callable record 的内部热路；普通
C++ 用户优先传捕获 lambda。`spawn(Fiber*)` 是显式 lifetime 的低层入口，Scheduler 不取得 Fiber
ownership；一般 fire-and-forget 不应手工管理它。

`bulk_execute` 不是 Graph algorithm：Sched 只按 caller 给定 grain，用一个 atomic cursor 把有限半开
range 交给至多 worker 数个 runner，并提供一次 quiescent completion、共享首错和可选 cooperative
cancel。adaptive split、reduce/scan/sort、tensor tiling 和模型策略留给 Graph/Compute/消费者。正常外部
线程/Fiber 路线不让 caller 参与执行；从 non-suspending worker-stack callback 嵌套时串行退化，避免一个
worker 同时成为 waiter 和唯一 executor。

## Worker 与队列

每个 worker 有 owner FIFO、本地 run-next LIFO slot、pinned queue 和 timer queue；外部生产者使用分片
global injection queue，冷路径还有不丢 ready Fiber 的 overflow。空闲 worker 随机选择 victim 批量 steal，
并周期检查 global queue，避免持续本地生产饿死外部注入。

LIFO slot 只容纳刚在本 worker 变 ready 的一个 Fiber，使其栈和数据仍热时立即继续；连续使用有固定
公平预算，超过预算就回到 FIFO。它不是另一套公开 RunQueue 类型，也不能成为用户调度 policy。

默认 worker 不绑核，让嵌入 HYLE 的 host runtime 和 OS 保留调度权。需要 per-core IO、NUMA 或专用
runtime 时，用户可显式提供 CPU 顺序/集合和 best-effort/required policy。`spawn_on` 的 Fiber 始终回到
owner worker，且不能被偷；通用任务仍默认可迁移。大小核偏好由 Platform 的 CPU 事实与 ThreadPriority
表达，Sched 不根据 CPU id 或 benchmark 猜测核心类型。

Idle policy 是语义选择：balanced 做有限搜索后 park，low-latency 延长搜索，power-saving 尽快 park。
防惊群只允许有限 searching worker，并用单 worker 唤醒/接力扩张并行度。不得把固定自旋次数解释为
跨机器最优；它属于需要 P1 证据约束的 implementation policy。

## Fiber stack 与第三方边界

上下文切换使用 vendored Boost.Context `fcontext` 的 `make/jump/ontop` 汇编原语，不引入 Boost.Fiber 的
scheduler。理由是 fcontext 只保存 ABI 要求的上下文，不修改信号掩码，同时已有 x86-64/AArch64、
PE/ELF/Mach-O 实现。Sched 拥有 Fiber 语义，Boost 只拥有私有汇编机制；fcontext C symbols 和 shim
不是 HYLE C API。

Fiber stack 由 Mem 的 `FiberStack`/`FiberStackPool` 提供 guarded small/default/large tier。常用 tier
池化，非 tier 大小可一次性分配。Fiber 只允许在第一次 resume 前 move，因为 fcontext trampoline 持有
原对象地址。固定 guarded stack 是可预测的基础形状；segmented/growing stack 没有真实消费者前不加入。

## 时间、取消与生命周期

所有公开 duration/deadline 使用 Time 的 `Duration` 和 `MonotonicTimePoint`。每 worker 自持
`TimerQueue`，arm/cancel 在 owner worker 完成；timer 到期只触发 Waker，不在 timer 容器里执行任意
用户 callback。墙钟不能进入 timeout 合同。

取消是协作式：

- `CancelToken` 可共享；`derive()` 建立子树；`is_cancelled()` 是 O(1) atomic load；
- tree mutation、callback register/drop 与 cancel 是 control path，可跨线程并按树串行化；
- `CancelScope` 把 token 加入当前 sched execution context 的 active source 栈：Fiber 上随 Fiber 迁移，
  worker-stack task 中按 callback/thread 生命周期约束；它与 `JoinHandle::abort()` 组合而非覆盖；
- callback 只能请求底层操作取消或唤醒等待者，不能强行摧毁正在运行的 stack。

Sched shutdown 分成关闭 admission/广播 shutdown 与等待 in-flight 完成。stackful work 无法安全抢占；
deadline 到达后，owner 必须取消 IO、请求 cooperative cancellation 或选择进程级终止，不能让 Scheduler
遗弃仍在执行的栈。

## 同步与消息

长等待原语必须 fiber-aware：Fiber waiter 挂起并由 Waker 重入，外部 OS thread waiter 走 Platform
Parker。`ConditionVariable` 是共同地基，Mutex、WaitGroup、Latch、Event、Barrier、Semaphore 和
RwLock 复用同一唤醒协议。Semaphore 使用严格 FIFO，避免大 permit 请求或 writer 被小请求永久饿死；
timed waiter 以 intrusive 双链 O(1) 摘除。

消息按消费语义分开：

- `Channel<T>`：bounded MPMC，保留每条消息并提供 blocking/try/timed/backpressure/close；失败发送返还
  原值；
- oneshot：一个 producer/consumer 的 reply slot，失败发送返还原值；
- watch：只保留最新值，适合配置、状态和 shutdown flag；
- broadcast：bounded overwrite ring，每个 receiver 有 cursor，lag error 携带精确 missed count。

持锁 value 操作必须有明确异常合同。ownership move/destructor 采用 nothrow 约束；watch/broadcast 的
copy 使用 RAII monitor guard，失败不锁死 channel 且不推进版本/cursor。Mutation callback 要求
`noexcept`，否则“值已部分修改但是否发布”没有可维护的语义。

## Allocator、Error 与依赖方向

Sched 的 runtime metadata、worker-stack task cell、task driver、join state、blocking job 和 pooled stack
都走显式 Mem allocator；
OS Thread 自身的系统资源属于 Platform 合同。便利构造可对可信不变量 fail-fast，面向运营失败的构造和
提交必须返回 `Status`/`Result`。队列 full、timeout、shutdown 和 OOM 不能被压成同一个 bool，涉及
未消费 value 时还必须返还 ownership。

公开依赖是 Core、Error、Time、Platform、Mem 和 Util-core；Container 只支撑 compiled queue
implementation，因此是 PRIVATE edge。IO 依赖 Sched，不允许反向 include IO。IoContext 通过不安装的
build-only detail bridge 注册 worker poll/wake hook；普通用户不直接改 Scheduler worker loop。

## 复杂度与性能合同

| 操作 | 目标成本结构 |
|---|---|
| cancellation poll | O(1)，无锁、无分配、无 syscall |
| local ready/run-next | O(1)；常态 bounded lock-free queue |
| external injection | O(1) amortized，按 shard 分散 contention |
| steal | 一次 victim probe + bounded batch；不得全局扫描每个任务 |
| wake/suspend | 常态无分配；至多一次 ready routing/worker wake |
| worker-stack typed task | O(1) admission；callable/result state 合并为一次 allocation；无 Fiber/context switch |
| finite bulk execution | O(N) body work；O(worker) raw queue entries；O(1) batch state/terminal notification；无 per-item allocation/status/cancel/handle |
| timer arm/cancel | 继承 Time queue 的 O(1) node operation |
| semaphore timeout unlink | O(1) |
| join group | O(n) install/clear，等待期间无 polling |
| shutdown | O(workers + retained work/resources) |

P0 先证明复杂度、allocation、syscall、锁和常态 metadata；P1 才测 callback、spawn/join、yield/wake、
external inject/steal、sync/cancel/channel 等正常 workload。资格化对比必须相同语义、ownership、
线程拓扑和计费边界。亚纳秒点值、地址/顺序翻转、自身空白对照或 stackful 对 stackless 的混测不得改变
API/layout。没有同合同 peer 时，只声明本 workload 的绝对成本和回归预算，不宣称全平台 SOTA。

## 交付与演进

Sched 以 C++23 source、installed static 和 installed shared 三种形状交付。公开头只能包含已声明 PUBLIC
依赖；`sched/detail`、fcontext shim 和汇编是 build/private 资产，不进入安装 SDK。当前没有 C API 或
Native FFI；decorated C++ symbol 和私有 fcontext `extern "C"` 都不构成跨语言承诺。

可触发新架构决策的事实包括：真实推理 consumer 需要通用 task priority、多个 completion source 共存、
caller-participating SPMD 或不同 stack strategy；成熟 peer 证明更好的复杂度级别；支持平台/ABI 需要替换
fcontext；或现有 lifetime/backpressure 合同无法表达可复现需求。此类变化先进入 capability decision，
不能从 benchmark 行直接长成公共类型。
