# HYLE sched/ 设计

> 统一调度器：把「线程 / 进程 / 纤程 / 协程」收敛成一张调度表，是 `io/`、`compute/` completion 机制的共同地基。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`engineering-charter.md`](./engineering-charter.md)。
> clay 现状：thread_pool 真实现但单全局锁（长不出统一调度器，须重写），catomics 原子层扎实可继承，fiber 零实现。详见 [`clay-asset-liability.md`](./clay-asset-liability.md) 第五节。

本设计分四块推进：①fiber 上下文切换选型（本文）②统一调度模型 ③completion 接入 ④协程与 fiber 统一。后三块依赖①的结论。

---

## 第一节：fiber 上下文切换选型

### 候选与裁决

| 方案 | 性质 | 裁决 |
|---|---|---|
| **ucontext**（getcontext/swapcontext）| POSIX API | ❌ **出局**：macOS 已弃用；`ucontext_t` 在 aarch64 macOS 非 FFI-safe（128-bit 整数）；每次切换默认 `sigprocmask` 系统调用，比 fcontext 慢约**两个数量级** |
| **Windows Fibers**（CreateFiber）| OS 原生 | ❌ **出局**：仅 Windows；有 FLS/TLS 开销，慢于 fcontext |
| **boost.context fcontext**（裸汇编）| 手写汇编原语 | ✅ **采用**：只存被调用者保存寄存器（按调用约定），不碰信号掩码，无系统调用；跨平台成熟（x64/arm64 × Win/Linux/mac） |
| 自写汇编 | 完全自控 | ⚠️ 等于重造 fcontext，仅当需要 fcontext 给不了的能力才考虑 |
| boost.fiber / PhotonLibOS | 完整 fiber 框架 | ❌ **不整体引入**：自带调度器，会与 HYLE 的 `sched/` 抢线程管理权（与 io 选型拒 asio 同理）|

### 决策：只借 fcontext 裸原语，调度自己写

**采用 boost.context 的 `fcontext`**——但**只取那三个汇编例程**（`jump_fcontext` / `make_fcontext` / `ontop_fcontext`），不取 boost.fiber。

这与 [`io-design.md`](./io-design.md) 的判断同构：**借 syscall/原语薄层（fcontext / liburing），调度器自己造**。fcontext 本身无调度器、无框架，就是一段切换栈和寄存器的汇编，正是我们要的最小原语。

**WHY fcontext 快**：它只按调用约定保存「被调用者保存寄存器」（System V AMD64 / AAPCS64 / Windows x64），不保存信号掩码——而 ucontext 每次切换为了信号掩码走一次系统调用。这是两个数量级差距的根源。

**未来可换**：WG21 P0876 `fiber_context`（"fibers without scheduler"）是标准化裸上下文切换的提案，尚未进 C++23/26。HYLE 把切换封在自有接口 `fiber_context` 之后，将来标准落地可平滑替换，不影响上层调度器。

### 跨平台矩阵（fcontext 汇编 = 平台×架构 的小矩阵）

| 平台 × 架构 | 汇编/格式 | 调用约定 |
|---|---|---|
| Linux x64 | `*_x86_64_sysv_elf` | System V AMD64 |
| Linux arm64 | `*_arm64_aapcs_elf` | AAPCS64 |
| Windows x64 | `*_x86_64_ms_pe` | Windows x64（含 SEH/栈信息）|
| Windows arm64 | `*_arm64_aapcs_pe` | AAPCS64 + MS |
| macOS x64 | `*_x86_64_sysv_macho` | System V AMD64 |
| macOS arm64 | `*_arm64_aapcs_macho` | AAPCS64（Apple 变体）|

整个矩阵 = 3 例程 × 6 (平台×架构) 的汇编，量小且稳定。Windows 还需处理 SEH 链与栈边界登记。

### fiber 与 C++20 协程的关系（不是二选一，是分工）

| | 有栈 fiber（fcontext）| 无栈协程（C++20 `co_await`）|
|---|---|---|
| 栈 | 自带独立栈，任意调用深度可挂起 | 无独立栈，仅 `co_await` 点可挂起 |
| 开销 | 重（栈分配）| 轻（编译器生成 frame）|
| 侵入性 | 透明，旧代码无需改 | 病毒式（调用链上一路标 `co_await`）|

**HYLE 用法**：fiber 作**调度单元**（M:N，多 fiber 映射到少量 OS 线程）；`co_await` 作**异步操作的人体工学外壳**（io/gpu completion）。一个 fiber 里可以跑用 `co_await` 的代码，awaitable 挂起即让出该 fiber。二者互补——fiber 解决"任意处可让出 + 透明"，协程解决"异步写法顺滑"。

### 栈管理

- **固定大小栈 + guard page**：每 fiber 一段栈，尾部 `mmap`(`PROT_NONE`) 守护页捕获溢出（Win 用 `PAGE_GUARD`）。
- **栈池化**：栈通过对象池复用（接 `mem/`），避免频繁 `mmap`。
- **分级栈大小**：按 fiber 类别配置——io 任务小栈（省内存、高并发），compute 任务大栈。
- 暂不做分段栈/按需增长（复杂、有热点开销），固定栈 + guard page 是性价比最优。

### 落地清单

1. `fiber_context` 抽象接口（封 fcontext，预留 P0876 替换位）。
2. 6 份 fcontext 汇编 + ~10 行自写 `fcontext.h`。**抽取边界（深读 boost.context 后定）**：`fcontext_t`=void*、`transfer_t`={void*,void*} 是纯 POD、零 boost 依赖；唯一 boost 味是 `BOOST_CONTEXT_DECL`/`CALLDECL` 两宏（缺省即空/x64 无关），自定义两行即切断 boost/config。**只取 `jump/make/ontop_fcontext` 三例程的 `.S/.asm`（从 boost 上游 `libs/context/src/asm/` 拉——本地 context 是空 submodule），不要 config/continuation/fiber/stack 封装层。**
3. 栈分配器（固定栈 + guard page + 池化，接 mem/）。**自造，不借 boost `stack_context`/`protected_fixedsize_stack`**（避免栈管理耦合进 boost 抽象）。
4. 单测：百万次切换基准（对标"十亿次/秒"量级）、guard page 溢出捕获、跨平台 CI。

---

## 第二节：统一调度模型

分三小块推进：①`time/` 最小接口 + run-queue 结构（本节）②work-stealing 协议 ③线程/进程/纤程/协程落到一张表。

### 2.1 `time/` 最小接口（sched 的前置）

sched 的 deadline/超时只需要三样，先钉这个最小面，full timer 留后：

- **单调钟** `mono_now() -> u64`（纳秒，`CLOCK_MONOTONIC`/`QueryPerformanceCounter`，不受墙钟跳变影响——deadline 绝不能用墙钟）。
- **timer wheel**：分层时间轮（hierarchical timing wheel），`add(deadline, fiber)` O(1) 摊还、`tick()` 批量到期。比 `priority_queue`（clay thread_pool 用的，O(log n)）更适合海量短超时（io timeout/fiber sleep）。
- **到期动作**：超时不直接执行回调，而是**把对应 fiber 标记就绪、塞回 run-queue**——与 completion 脊同一条唤醒路径（见 ③）。

> 为什么单列 time/ 而非塞进 sched：timer wheel 有独立设计分量，且 io/ 的 timeout 也直接用它。但它是 sched 的仆从（T2），不反向依赖 sched。

### 2.2 run-queue 结构（调度单元 = fiber handle）

**关键差异**：Tokio/Go 调度无栈 task，**HYLE 的调度单元是 fiber handle（有栈）**。队列里装的是"可运行的 fiber 指针"，fiber 内部可 `co_await`（见 ④）。

借鉴 Tokio/Go 的成熟结构，按 HYLE 需求定型：

| 部件 | 设计 | WHY |
|---|---|---|
| **per-core 本地队列** | 有界 ring（256 槽），owner 一端 LIFO push/pop、thief 另一端 FIFO steal | 有界=无锁+cache 友好；本地 LIFO=热栈复用；偷 FIFO=公平、偷老任务 |
| **LIFO slot（单槽）** | 每核一个"下一个就跑"单槽，刚被唤醒的 fiber 放这里、优先于本地队列 | **completion 脊的接入点**：io/gpu 完成唤醒的 fiber 落唤醒核 LIFO slot，热 cache 立即续跑 |
| **global 注入队列** | 加锁 MPMC，接本地队列溢出 + 跨核 spawn + 外部线程提交 | 兜底；本地满了甩这里，不丢任务 |
| **steal 协议** | 本地空 → 偶发查 global（防饿死）→ 随机选 victim 偷一半 | Tokio 实证：纯偷会饿死 global，需周期性查 global |

**调度循环（每个 worker 线程）**：
```
loop:
  f = lifo_slot.take()           // 1. 刚唤醒的，热 cache 优先
      ?? local_queue.pop()       // 2. 本地 LIFO
      ?? (every Nth) global.pop() // 3. 周期查 global 防饿死
      ?? steal_from_random_victim() // 4. 偷一半
      ?? park()                  // 5. 都没有→挂起（见 2.3 防惊群）
  switch_to_fiber(f)             // fcontext 切栈（第一节）
```

**防惊群**（park/unpark）：维护 `searching_thieves` + `sleeping_workers` 两个原子计数；有新任务且当前无 thief 在搜时才 `notify_one`，避免一个任务唤醒全部核。（参考 work-stealing scheduler 论文的 actives/thieves 协议。）

**基于 clay catomics 重写**：丢弃 clay thread_pool 的单全局锁 + 单 priority_queue（长不出来），per-core 队列的 push/pop/steal 用 catomics 的 CAS + 指针 CAS 实现无锁。

### 2.3 work-stealing 精细化：spin-then-park + 防惊群

无工作的 worker 既不能空转烧 CPU，也不能太急着睡（work 来了有延迟），更不能 work 一来就唤醒全部核（惊群）。三招解决：

**① spin-then-park（睡前先转几圈）**：worker 找不到活时，先做有限次 steal 尝试（轮询若干随机 victim）+ 指数 backoff，再 park。WHY：park 要系统调用（futex/WaitOnAddress），而 work 常在微秒内出现，先自旋能避开大量 park/unpark 往返。

**② 限制同时搜索的 thief 数**：原子计数 `num_searching`，封顶（如 ≤ 核数一半）。已超额则无工作的 worker 直接 park、不加入偷窃风暴。WHY：N 个 worker 同时偷 = N² 次 steal 尝试 + cache 行争抢，吞吐反降。

**③ unpark 接力（防惊群核心）**：
- 新任务入队且 `num_searching == 0` 时，**只** `notify_one`（唤醒一个）。
- 被唤醒的 worker 置为 searcher；它找到活、从 searching 转 running 时，**若它是最后一个 searcher，再唤醒一个**（baton-passing 接力）。
- 净效果：醒着的 worker 数恰好匹配可用工作量，既不惊群也不漏唤醒。（参考 Go `wakep`/Tokio `transition_from_searching` 与 actives/thieves 论文。）

**park 机制跨平台**：futex(Linux) / `WaitOnAddress`(Windows) / `__ulock` 或 dispatch semaphore(macOS)，封在 `platform/` 后。

**两个补充约束**：
- **steal 偷一半**（非一个）：摊薄 steal 成本，有界 deque（st3/Chase-Lev）支持批量窃取。
- **LIFO slot 预算**：LIFO slot 里的 fiber 不可被偷，会饿死其他核——给它一个轮询预算上限（Tokio 实证），超了踢回本地队列尾、可被偷。

### 2.4 线程 / 进程 / 纤程 / 协程：一张表的真实含义

"统一成一张表"不是把四者塞进同一个队列——它们**不是同一种东西**。诚实的模型是：**一种可调度实体（fiber）+ 一个调度器 + 一张唤醒网（completion 脊），四者各就其位。**

| 概念 | 角色 | 在调度模型里是什么 |
|---|---|---|
| **纤程 fiber** | **调度单元（work）** | run-queue 里的 `Task`：有栈、可 steal、可让出。**这才是"表"里的行** |
| **线程 thread** | **执行引擎（engine）** | 不是被调度物，是跑 fiber 的 worker；N worker = 调度底座（M:N，M fiber 映射 N worker）|
| **协程 coroutine** | **fiber 内的挂起点** | 不是独立实体；`co_await` 让当前 fiber 在 await 点让出（详见 ④）|
| **进程 process** | **隔离 / 并行边界** | 各自一个 scheduler 实例；**跨进程不共享 run-queue**（地址空间不同，fiber 栈指针跨进程无意义）|

**真正统一的是"唤醒网"，不是队列**。凡"会阻塞、之后变就绪"的东西，都产出一个 completion、唤醒一个 fiber、落进 run-queue（2.2 的 LIFO slot）：

```
        timer 到期 ─┐
        io 完成 ────┤
        gpu event ─┼─→ [唤醒网/completion 脊] ─→ 标记 fiber 就绪 ─→ 入 run-queue
        fiber 让出 ─┤
        跨进程消息 ─┘   （shmem ring / io 完成，对 fiber 而言就是又一个 awaitable）
```

四者由此插进**同一张唤醒网**，而不是各跑各的事件循环——这才是"一张表"的实质，也是 HYLE 区别于"std::thread + asio + 各自的 GPU 流"拼装货的根本。

**两条必须讲清的边界（否则就是错的）**：
- **进程是联邦，不是共享表**：单进程内 thread+fiber+coroutine 真正共享一个调度器；跨进程只能靠 `io/` + 共享内存做消息传递，从 fiber 视角"等另一个进程"就是又一个 awaitable。**别假装进程是同一队列里的行**。
- **阻塞系统调用会卡住 worker**：某 fiber 跑了阻塞 syscall，它占用的 worker 线程被卡住、连累该核其他 fiber。缓解：(a) 阻塞调用走独立 blocking 线程池；(b) 类 Go sysmon 的监控线程检测长阻塞、临时补一个 worker 顶上。这是"线程"在模型里必须处理的现实，留待实现期定策略。

### 第二节小结

调度单元 = fiber handle（有栈）；执行底座 = M:N worker；协程是 fiber 内挂起点；进程是联邦边界。run-queue = per-core 有界 ring + LIFO slot + global 兜底，spin-then-park + unpark 接力防惊群，全部基于 catomics 无锁。**统一的本质是唤醒网（completion 脊），下一节就接它。**

> **归属澄清（深读 marl 后）**：本节的"无锁 per-core 队列 / LIFO slot / unpark 接力防惊群"是 **Tokio/Go 血统，不是 marl**。marl 实际是 **有锁 per-worker deque（`work.mutex` 保护）+ 偷 task 不偷 fiber（裸汇编 fiber 不存 TLS，锁死在创建 worker）+ 仅 spin+notify_one（无 thief 计数封顶/无 baton 接力）**。HYLE 在三处**超越** marl：①catomics 无锁队列 ②fiber 可跨核迁移（靠 Waker 唤回 origin_core + global 兜底，见 3.3）③actives/thieves 防惊群。marl 印证了 fcontext 选型（它自写等价汇编、ucontext 仅回退）和 fiber-aware CV 收敛，但调度内核是 HYLE 自有差异点。

> **缺口补记（完整性评估发现）**：sched/ 必须提供 **fiber-aware 同步原语**——fiber mutex/condvar/channel/waitgroup/latch。理由：`std::mutex` 阻塞的是 worker 线程，会连累该核所有 fiber、破坏 M:N 调度。正确做法是"竞争时让出 fiber、入等待队列，释放时把等待 fiber 重新入 run-queue"——本质又是一次 completion/唤醒（与第三节 Waker 同机制）。这套原语与 run-queue 强耦合，归 sched/，留待实现期与 ②③ 一并落地。
>
> **深读 marl 的收敛方案（采纳）**：marl 证明不必为 mutex/WaitGroup/Event/Ticket 各写一套唤醒——**只做一个 fiber-aware ConditionVariable（维护 waiter fiber 链表），其余全建在它上面**。且要**双模**：waiter 既可是 fiber（让出+重入 run-queue）也可是裸线程（回退 `std::condition_variable`）——HYLE 跨进程联邦/外部线程提交场景需要这个，别假设所有 waiter 都是 fiber。HYLE fiber-sync 照此收敛：一个 CV 地基 + 其上薄封装。

## 第三节：completion 接入（整个设计的收口）

第二节反复指向一句话：**统一的本质是唤醒网（completion 脊）**。本节把它做实——所有异步源（io/gpu/timer）如何映射到"唤醒哪个 fiber、落哪个核"。

### 3.1 Waker：统一接缝

每个会阻塞的异步操作（io 读写、gpu kernel、sleep）提交时关联一个 **`Waker`**：

```
struct Waker {
  Fiber*   fiber;        // 等这个操作的 fiber
  uint32_t origin_core;  // 提交它的核（用于局部性唤回）
};
// 操作完成 → backend 拿回对应 Waker → wake()：标记 fiber 就绪 → 入队
```

fiber 提交操作后**让出**（fcontext 切回 worker，栈原地保留）；完成时 `Waker::wake()` 把 fiber 重新入队。io/gpu/timer 三源共用这一个 `Waker`，这就是"唤醒网"的物理形态。

### 3.2 谁 poll、怎么 poll（各后端同构）

worker 调度循环（2.2）的第 0 步插入"收割 completion"：

| 后端 | poll 方式 | 归属 |
|---|---|---|
| **io_uring**（Linux）| **每核一个 io_uring 实例**，worker 在循环里 peek 自己的 CQ（无锁、无跨核争抢）| io/ |
| **IOCP**（Windows）| worker 线程 `GetQueuedCompletionStatusEx` 批量取（IOCP 本就多线程完成端口）| io/ |
| **GPU**（CUDA/Vulkan）| 二选一：(a) `cudaLaunchHostFunc` 回调里**只做入队、不调 CUDA**；(b) GPU poller 线程批量 `cudaEventQuery`(非阻塞)/Vulkan fence 轮询 | compute/ |

**per-core io_uring 是关键选择**：每个 worker 自己的 ring，提交和收割都在本核，零跨核同步——和 run-queue 的 per-core 哲学一致。GPU 因为 host 通知机制弱（不能每个 op 阻塞一线程），单设一个 completion 源翻译成 fiber 唤醒。

### 3.3 completion → 唤醒哪个 fiber、落哪个核

- **唤醒哪个 fiber**：靠 `user_data` 关联——io_uring 的 SQE 带 64-bit `user_data`、IOCP 带 `OVERLAPPED*`、CUDA host func 带 `void*`，三者都塞 `Waker` 的指针/ID，完成时原样取回 → 定位 fiber。这是映射的物理机制。
- **落哪个核**：优先唤回 `origin_core` 的 **LIFO slot**（2.2）。WHY：该 fiber 的栈、它操作的 buffer 大概率还在原核 cache，热数据立即续跑。origin_core 在睡 → unpark 它；负载失衡 → 退而落 global 让任意核捡。

### 3.4 与 io/、compute/ 的分工

- **sched/ 提供**：`Waker` 抽象 + run-queue 入队 + LIFO slot。
- **io/ 提供**：per-core io_uring/IOCP 的提交与 CQ 收割，完成时调 `Waker::wake()`。
- **compute/ 提供**：GPU 提交 + event/回调，完成时调 `Waker::wake()`。

三方只通过 `Waker` 这一个接缝耦合——io/compute 不知道调度细节，sched 不知道 io/gpu 细节。这正是 charter 里"completion 脊是 Tier 0"的落地。

## 第四节：协程与 fiber 统一

第一节定了 fiber（有栈）作调度单元，C++20 协程（无栈）作异步外壳。本节是二者的胶水。

### 4.1 awaitable 落到 fiber 让出

`co_await some_async_op()` 的 awaitable，其 `await_suspend` 做三件事：
1. 构造 `Waker{ current_fiber, this_core }`，随操作提交给 io/compute（3.1）。
2. 调 `fiber_yield()`——fcontext 切回 worker 调度循环，**当前 fiber 栈原地冻结**。
3. worker 继续跑别的 fiber。

操作完成 → `Waker::wake()` → fiber 重新入队 → 被调度 → fcontext 切回，`await_resume()` 返回结果，`co_await` 之后的代码接着跑。**对写代码的人，就是一句同步写法的 `co_await`；底层是 fiber 让出+唤醒。**

### 4.2 为什么要两层（不是冗余）

| 层 | 解决什么 | 不能少的理由 |
|---|---|---|
| **有栈 fiber** | 任意调用深度让出、旧代码透明接入 | 无栈协程要求调用链一路 `co_await`（病毒式）；深层库函数想让出就只能靠 fiber |
| **无栈协程** | 异步写法顺滑、单 fiber 内零栈切换的挂起 | 纯 fiber 每次让出要切栈；协程在 fiber 内挂起更轻 |

**分工**：fiber 是"能在任何地方让出"的粗粒度调度单元；协程是 fiber 内部"写起来顺、挂起更轻"的细粒度异步。一个 fiber 里可以有很多 `co_await` 点。二者不打架，因为它们解决的是正交问题（调度粒度 vs 写法/挂起开销）。

### 4.3 边界

- 协程的 `promise_type` 用 HYLE 的 `mem/` 分配 frame（不走默认 `operator new`，贯彻显式 allocator）。
- 不强迫所有异步都用协程：底层热路径可直接用 fiber + Waker 裸接口，协程是上层人体工学糖。

---

## 全节总结：sched/ 设计定稿

**一句话**：HYLE 的调度器 = **一种调度单元（有栈 fiber）+ 一个 M:N 无锁调度器 + 一张唤醒网（completion 脊）**，让 线程/进程/纤程/协程 + io/gpu/timer 全部插进同一套底座，而非各跑各的事件循环。

| 块 | 决策 | 关键依据 |
|---|---|---|
| ① fiber 上下文切换 | **boost.context fcontext 裸原语**（不要 boost.fiber），ucontext 出局 | 比 ucontext 快两个数量级；ucontext macOS 已弃用、非 FFI-safe |
| ② run-queue | per-core 有界 ring（owner LIFO/thief FIFO 偷一半）+ LIFO slot + global 兜底 | Tokio/Go 实证；LIFO slot = completion 接入点 |
| ② work-stealing | spin-then-park + 限制 thief 数 + unpark 接力 | 防惊群、防 N² 偷窃风暴 |
| ② 四类统一 | fiber=调度单元、thread=执行引擎、coroutine=fiber 内挂起点、process=联邦边界 | 进程不共享 run-queue（地址空间不同），靠 io+shmem 联邦 |
| ③ completion | 统一 `Waker{fiber, origin_core}`；per-core io_uring / IOCP / GPU 回调或轮询 | user_data 关联 fiber；唤回 origin_core 的 LIFO slot 保 cache 局部性 |
| ④ 协程/fiber | `co_await` 的 await_suspend 构造 Waker + fiber_yield；两层正交（调度粒度 vs 挂起开销）| 深层让出靠 fiber，写法顺滑靠协程 |

**全部基于 clay catomics 无锁重写，丢弃 clay 单全局锁 thread_pool。** time/ 提供单调钟+时间轮（deadline 不用墙钟），platform/ 提供 park 原语（futex/WaitOnAddress），二者是 sched 的 T2 仆从。

**待实现期定的策略**（非设计悬念，是落地选择）：阻塞 syscall 的 blocking 池 + sysmon 补线程；LIFO slot 轮询预算上限；steal 的 backoff 曲线；GPU completion 选回调还是轮询线程。

## 参考

- [boost.context: fcontext_t vs ucontext_t vs WinFiber（性能差两个数量级）](https://www.boost.org/doc/libs/latest/libs/fiber/doc/html/fiber/overview/implementations__fcontext_t__ucontext_t_and_winfiber.html)
- [WG21 P0876 fiber_context — fibers without scheduler](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p0876r15.pdf)
- [PhotonLibOS：现代有栈协程实现](https://www.reddit.com/r/cpp/comments/1jwlur9/stackful_coroutines_faster_than_stackless)
- [Making the Tokio scheduler 10x faster（per-core 队列 + LIFO slot）](https://tokio.rs/blog/2019-10-scheduler)
- [st3：现代有界 work-stealing 队列（FIFO steal / LIFO|FIFO worker）](https://github.com/asynchronics/st3)
- [An Efficient Work-Stealing Scheduler for Task Dependency Graph（actives/thieves 防惊群）](https://tsung-wei-huang.github.io/papers/icpads20.pdf)
- [CUDA Events / cudaLaunchHostFunc（GPU host 侧完成通知）](https://nvidia.github.io/cccl/unstable/libcudacxx/runtime/event.html)
- [x86-64 / ARM64 调用约定速查（2026）](https://www.codermusings.com/resources/calling-conventions-cheat-sheet)
