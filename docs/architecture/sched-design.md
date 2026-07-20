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

- **单调钟** `mono_now() -> u64`（纳秒，`CLOCK_MONOTONIC`/`QueryPerformanceCounter`，不受墙钟跳变影响——deadline 绝不能用墙钟）。**已落地**=`time::monotonic_ns()`（time-01）。
- **timer wheel**：分层时间轮（hierarchical timing wheel），`add(deadline, fiber)` O(1) 摊还、`tick()` 批量到期。比 `priority_queue`（clay thread_pool 用的，O(log n)）更适合海量短超时（io timeout/fiber sleep）。**已落地**=`time::TimerWheel`（time-02，tokio 6×64 位图层级扩为 8×64，覆盖完整 `uint64_t` ns deadline；对 indexed eager-erase heap 的三类负载双平台均赢）。
- **到期动作**：超时不直接执行回调，而是**把对应 fiber 标记就绪、塞回 run-queue**——与 completion 脊同一条唤醒路径（见 ③）。**已落地**=sched-04b：每 worker 持一个 `TimerWheel`，`sched::sleep_for_ns` arm 到本 worker 轮 + suspend；worker 循环顶 `fire_due_timers`（`wheel.empty()` 门=无 timer 零税）+ 用 `next_deadline_ns` 喂 `Parker::park_timeout` 不空转；到期经 sched-04a Waker rendezvous `reinject`——与 io 完成同一 `wake_rendezvous`。thread-mode `wait_for`/`wait_until` 已落地=sched-04c；fiber-mode(需跨 worker cancel)=sched-04d。

> 为什么单列 time/ 而非塞进 sched：timer wheel 有独立设计分量，且 io/ 的 timeout 也直接用它。但它是 sched 的仆从（T2），不反向依赖 sched。

### 2.2 run-queue 结构（调度单元 = fiber handle）

**关键差异**：Tokio/Go 调度无栈 task，**HYLE 的调度单元是 fiber handle（有栈）**。队列里装的是"可运行的 fiber 指针"，fiber 内部可 `co_await`（见 ④）。

借鉴 Tokio/Go 的成熟结构，按 HYLE 需求定型：

| 部件 | 设计 | WHY |
|---|---|---|
| **per-core 本地队列** | 有界 ring（256 槽，tokio `LOCAL_QUEUE_CAPACITY`），**owner FIFO（push_back tail / pop head）**、thief 从最老端 FIFO steal 偷一半（向上取整）；满时后半 128 个甩 global | 有界=无锁+cache 友好；偷老任务=公平。**注意：LIFO 性不来自 run_queue 本身**（深读 tokio `queue.rs` 校正：run_queue 是 FIFO，owner-LIFO 是 st3/Chase-Lev 语义、非 tokio）|
| **LIFO slot（单槽）** | 每核一个"下一个就跑"单槽，刚被唤醒的 fiber 放这里、优先于本地队列；**不可被偷**，连续轮询预算上限（tokio `MAX_LIFO_POLLS_PER_TICK=3`）超了关 slot 改走队尾（可被偷）| **completion 脊的接入点**：io/gpu 完成唤醒的 fiber 落唤醒核 LIFO slot，热 cache 立即续跑。**LIFO 热栈复用全靠这个单槽，不靠 run_queue** |
| **global 注入队列** | **无锁分片 MPMC（`ShardedQueue`,N 子环=每 worker 一环）**,接本地队列溢出 + 跨核 spawn + 外部线程提交 | 兜底不丢任务;分片消单点 CAS 争用(bench-02b:反超 moodycamel 2.6-3x,随生产者数平坦)。放弃全局 FIFO——work-stealing 本就无全局序 |
| **steal 协议** | 本地空 → 随机选 victim 偷一半;**且每 61 次 find_task 先查 global 一次**(`kGlobalQueueInterval`,tokio 同款质数)防本地 firehose 饿死全局注入 | Tokio 实证:纯偷会饿死 global,需周期查。**固定 61 非 EWMA 自调**:tokio 的 EWMA 自调(200µs/均值、clamp 2-127)只对超长 poll workload 有意义,固定质数已堵住饥饿洞(=正确性点);饥饿真实存在——探针实证 1-worker yield-hog 转 1700 万次而全局 victim 从未运行,加周期查后 49ms 内跑 |

**调度循环（每个 worker 线程,实际 find_task 顺序）**：
```
loop:
  f = pinned.pop()                    // 0. 钉在本核的 fiber(shard-per-core io),永不被偷,io 延迟优先
      ?? (tick%61==0 ? global.pop())  // 1. 每 61 次先查 global 防本地 firehose 饿死全局注入(机会③)
      ?? lifo_slot.take()             // 2. 刚在本核唤醒的,热 cache 优先(预算 kMaxLifoPolls=3)
      ?? local_queue.pop()            // 3. 本地 FIFO ring(owner pop head)
      ?? global.pop()                 // 4. 全局分片注入(常态兜底)
      ?? steal_from_random_victim()   // 5. 随机 victim 偷一半
      ?? park()                       // 6. 都没有→挂起(见 2.3 防惊群)
  switch_to_fiber(f)                  // fcontext 切栈（第一节）
```

**防惊群**（park/unpark）：维护 `searching_thieves` + `sleeping_workers` 两个原子计数；有新任务且当前无 thief 在搜时才 `notify_one`，避免一个任务唤醒全部核。（参考 work-stealing scheduler 论文的 actives/thieves 协议。）

**基于 clay catomics 重写**：丢弃 clay thread_pool 的单全局锁 + 单 priority_queue（长不出来），per-core 队列的 push/pop/steal 用 catomics 的 CAS + 指针 CAS 实现无锁。

### 2.3 work-stealing 精细化：spin-then-park + 防惊群

无工作的 worker 既不能空转烧 CPU，也不能太急着睡（work 来了有延迟），更不能 work 一来就唤醒全部核（惊群）。三招解决：

**① spin-then-park（睡前先转几圈）**：worker 找不到活时，先做有限次 steal 尝试（轮询若干随机 victim）+ 指数 backoff，再 park。WHY：park 要系统调用（futex/WaitOnAddress），而 work 常在微秒内出现，先自旋能避开大量 park/unpark 往返。**归属注意（深读 tokio 校正）**：当前 tokio multi_thread **无** spin-then-park 指数 backoff（它靠"park 前末位 searcher 最后一次全队列复查"+ 0-timeout park_yield 替代）；带 spin 的是已删除的 `multi_thread_alt`。所以**这条是 HYLE 自有设计（或借 Go/alt），别归功当前 tokio**。

**② 限制同时搜索的 thief 数**：原子计数 `num_searching`，封顶 ≤ 核数一半（**tokio `idle.rs` 印证：`2*num_searching >= num_workers` 即拒绝转 searching**）。已超额则无工作的 worker 直接 park、不加入偷窃风暴。WHY：N 个 worker 同时偷 = N² 次 steal 尝试 + cache 行争抢，吞吐反降。

**③ unpark 接力（防惊群核心，tokio `idle.rs` 一字不差印证）**：
- 新任务入队且 `num_searching == 0`（且 `num_unparked < num_workers`）时，**只** `notify_one`（唤醒一个）。
- 被唤醒的 worker 置为 searcher；它找到活、从 searching 转 running 时，**若它是最后一个 searcher，再唤醒一个**（baton-passing 接力，tokio `transition_worker_from_searching`）。
- 净效果：醒着的 worker 数恰好匹配可用工作量，既不惊群也不漏唤醒。（Go `wakep` / tokio `idle.rs` / actives/thieves 论文同款。）
- **sleeper 集 O(1) 维护（复杂度机会②，HYLE 对 tokio 的偏离）**：parked worker id 存一个紧凑栈 `sleepers` + 一个 per-worker `sleeper_pos`(在栈中的下标或"未睡")。`remove_sleeper` 按 `sleeper_pos` 直接 swap-remove(O(1))，不扫描。**为何偏离 tokio**：tokio 的 `unpark_worker_by_id` 容忍 O(worker数)扫描(它只在罕见的"自唤醒且本地有活"路径触发)；HYLE 在**每次 park 退出**都调 `remove_sleeper`，且在 `idle_mu` 锁内——O(1) 化同时缩短了 park/unpark 抖动下的临界区。同锁保护，无新原子，TSan 净。

> **HYLE 超越点（fiber vs task）**：tokio LIFO slot 仅省一次入队；HYLE 偷 fiber = 迁移其栈（跨核后栈+buffer cache 失效，比偷无栈 future 代价更高），所以"唤回 origin_core LIFO slot"（见 3.3）对 HYLE **更关键**——还省一次跨核栈迁移。这是 HYLE 相对 tokio 的强化点，非简单对齐。

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
- **阻塞系统调用会卡住 worker**：某 fiber 跑了阻塞 syscall，它占用的 worker 线程被卡住、连累该核其他 fiber。**✅ 已落地(sched-11)=显式 `spawn_blocking`**：把阻塞调用扔到独立 blocking 线程池(懒创建、上限 512 远超核数、10s keep-alive 空闲自退、单 MPMC 队列+condvar),worker 保持自由;结果经现有 `JoinHandle<R>`(sched-10)回传,fiber-aware CV 唤醒 join 方。**sysmon(监控线程检测长阻塞、临时补 worker)不做**——tokio 也没有,阻塞卸载是**显式**的(调用方决定什么会阻塞、主动 `spawn_blocking`);自动检测卡住 worker 补线程是 Go runtime 风格,无消费者驱动前不投机做(轴 C 纪律)。这是"线程"在模型里必须处理的现实,已由显式卸载入口覆盖。tokio `spawn_blocking`/marl blockingcall/seastar thread 同此形状。

### 第二节小结

调度单元 = fiber handle（有栈）；执行底座 = M:N worker；协程是 fiber 内挂起点；进程是联邦边界。run-queue = per-core 有界 ring + LIFO slot + global 兜底，spin-then-park + unpark 接力防惊群，全部基于 catomics 无锁。**统一的本质是唤醒网（completion 脊），下一节就接它。**

> **归属澄清（深读 marl 后）**：本节的"无锁 per-core 队列 / LIFO slot / unpark 接力防惊群"是 **Tokio/Go 血统，不是 marl**。marl 实际是 **有锁 per-worker deque（`work.mutex` 保护）+ 偷 task 不偷 fiber（裸汇编 fiber 不存 TLS，锁死在创建 worker）+ 仅 spin+notify_one（无 thief 计数封顶/无 baton 接力）**。HYLE 在三处**超越** marl：①catomics 无锁队列 ②fiber 可跨核迁移（靠 Waker 唤回 origin_core + global 兜底，见 3.3）③actives/thieves 防惊群。marl 印证了 fcontext 选型（它自写等价汇编、ucontext 仅回退）和 fiber-aware CV 收敛，但调度内核是 HYLE 自有差异点。

> **缺口补记（完整性评估发现 → sched-05 已落地）**：sched/ 提供 fiber 同步原语——CV + mutex/waitgroup/channel/latch（`include/hyle/sched/sync.h` + `src/sched/sync.cc`）。核心洞察:`std::mutex`/`std::condition_variable` 阻塞 worker 线程,长等待会连累该核所有 fiber、破坏 M:N。但**不是所有原语都该 fiber-aware**——这取决于等待时长(见下分工)。**已实测**：双平台真编真测,WSL TSan 8 个并发用例零 data race,且 SOTA 对标 marl(Mutex 打平、见 performance.md)。
>
> **深读 marl 的收敛方案（采纳,sched-05 已落地+实测纠错）**：fiber-awareness 按**等待时长**分工,不是一刀切——
> - **长等待 → fiber-aware**:`ConditionVariable` 是唯一 fiber-aware 地基(FIFO waiter 链表 + SpinLock),`WaitGroup`/`Latch`/`Channel` 全建其上。**双模**:waiter 是 fiber 则 `suspend()` 让出+重入 run-queue(复用第三节 Waker rendezvous,`current_waker()` 判定);是裸线程则回退 `platform::Parker`(跨进程联邦/外部线程提交,**非** `std::condition_variable`,HYLE 自有 Parker 无 std 依赖)。
> - **短等待(mutex)→ 不 fiber-aware,用 futex**:`Mutex` 是建在 `platform::wait_on_address` 上的 Drepper 3 态 futex(无争用单 CAS、争用 park worker),**不**让出 fiber。**实测纠错**:初版把 Mutex 也做成 fiber-aware(争用让出),benchmark 慢 marl **5.8x**——短临界区让出 fiber(~100ns)远贵于就地 park 几 ns。改 futex 后**打平 marl**(marl 同理:其 mutex 就是 std::mutex=futex,只 CV 才 fiber-aware)。契约:**别跨 fiber 让出/阻塞调用持 Mutex,那种场景用 CV**。
> - `wait_for`/`wait_until` 定时变体已接 time-02 时间轮。2026-07 收口把任务入口拆成**同合同的两条腿**：不可挂起的 `schedule(fn,ctx)` 直接在 worker 栈运行；可 yield/sleep/wait 的 `spawn_detached`/`spawn_task` 使用池化有栈 fiber。最终同形状 15 遍中，2000 noop 的 HYLE/marl 配对吞吐比为 **Windows 1.534×、Linux 1.480×、M4 1.007×（统计平局）**。M4 原先 0.904× 的真输已定位并修复为全局完成计数 cache-line 争用，不归咎异构核心或噪声。有栈腿承担 marl 该 benchmark 没承担的可挂起合同，单列绝对成本而不冒充同形状输赢。callback 的 `{fn,ctx}` payload 按值入队，去掉每任务 malloc/free；per-worker 完成计数、最多 8 个 inline batch 与 ShardedQueue 除法消除补平 ARM；conditional-notify 仍是把初版 17× 唤醒风暴根治到 SOTA 档的关键。
>
> **同步/结构化并发缺口补记(2026-07-03 参考库审计——tokio/seastar/folly 对照)**:sched-05 补齐了 CV/Mutex/WaitGroup/Latch/Channel,但对照 SOTA 仍缺几件真原语,登记为 sched-09/10(能力 C7 驱动,未排期前不做):
> - **Semaphore(计数信号量)**:异步限流标准件(限在途连接/请求、buffer 池)。tokio/seastar/folly 全有,HYLE 无。建在 CV 地基上成本低。
> - **RwLock**:读多写少共享态;现只有独占 Mutex。**oneshot**(单值回传)/ **broadcast·watch**(多消费者 fan-out / 最新值传播,用于关停信号·配置热载)同缺。→ sched-09
> - **JoinHandle + when_all/when_any(select)** ✅ **已落地(sched-10)**:`spawn(Fiber*)` 仍返 void(契约不变),新增 **`spawn_task<T>(sched,alloc,Task<T>)→JoinHandle<T>`**——`JoinHandle<T>.join()` 拿 `Result<T>`(fiber-aware 阻塞,复用 sched-05 CV)、`abort()` 协作式取消(置 flag,task 在让出点查 `is_cancel_requested()` 后 RAII 退栈——有栈 fiber 不可抢占,folly/fibers/marl 同此;**cancel flag 存在 `Fiber` 对象上而非 thread_local**——task fiber 在 `co_await` 挂起后经 work-stealing 会在**另一 worker 线程**恢复,thread_local 的 cancel scope 设在原线程、迁移后新线程读不到→漏 abort,故 flag 必须随 fiber 迁移。这是 M:N 调度器里**任何 per-fiber 状态都不能放 thread_local** 的通例,同 fiber.cc 的 TSan fiber 句柄)、drop=detach。`when_all`/`when_any(select)` 共用栈上 GroupSignal(目标计数),`when_any` 记 winner=超时(C2)/可取消 io(C3)的底层。**参考库通读定型(tokio/folly/seastar/marl)**:JoinHandle 不拥有 Fiber(marl 证明可薄如 WaitGroup(1)+slot);spawn_task 的 TaskDriver 拥有 Fiber,经 Fiber `on_finish` 钩子在 worker dispatch(控制已离开 fiber)析构=唯一 sched 拥有 fiber 处;GroupSignal 销毁前卸载=folly WhenN UAF 防护。真 io 层取消(io_uring ASYNC_CANCEL/IOCP CancelIoEx)留 io-06。

## 第三节：completion 接入（整个设计的收口）

第二节反复指向一句话：**统一的本质是唤醒网（completion 脊）**。本节把它做实——所有异步源（io/gpu/timer）如何映射到"唤醒哪个 fiber、落哪个核"。

### 3.1 Waker：统一接缝

每个会阻塞的异步操作（io 读写、gpu kernel、sleep）提交时关联一个 **`Waker`**：

```
class Waker {                 // 现状(sched-04a 已实现)
  Scheduler* sched_;          // 唤醒回哪个调度器
  Fiber*     fiber_;          // 等这个操作的 fiber
};
// 操作完成 → backend 拿回对应 Waker → wake()：标记 fiber 就绪 → 经 sched 重入全局队列
```
> **现状 vs 目标**:`wake()` 经 `reinject` 重入。**当唤醒方本身在某 worker 线程上跑**(常态:fiber A 唤 fiber B),已实现**投进本 worker 的 LIFO fast-slot**——B 下轮在本核 cache-hot 立即恢复,跳过 global CAS + `notify()` futex(复杂度机会①,ping-pong 延迟 −32%/−35% Win/Lin,见性能文档)。槽是 worker 私有纯字段(steal-exempt+无原子),`kMaxLifoPolls=3` 反饥饿。**仍未做**:Waker 带"提交核"字段以便**跨线程**唤醒也能按 origin_core 定向唤回热核(现在跨线程 wake 仍走 global 兜底);pinned fiber 已经能唤回 owner(shard-per-core),但通用非 pinned fiber 的跨核 origin 亲和还没有。

fiber 提交操作后**让出**（fcontext 切回 worker，栈原地保留）；完成时 `Waker::wake()` 把 fiber 重新入队。io/gpu/timer 三源共用这一个 `Waker`，这就是"唤醒网"的物理形态。

### 3.2 谁 poll、怎么 poll（各后端同构）

worker 调度循环（2.2）的第 0 步插入"收割 completion"：

| 后端 | poll 方式 | 归属 |
|---|---|---|
| **io_uring**（Linux）| **每核一个 io_uring 实例**，worker 在循环里 peek 自己的 CQ（无锁、无跨核争抢）| io/ |
| **IOCP**（Windows）| worker 线程 `GetQueuedCompletionStatusEx` 批量取（IOCP 本就多线程完成端口）| io/ |
| **GPU**（CUDA/Vulkan）| 二选一：(a) `cudaLaunchHostFunc` 回调里**只做入队、不调 CUDA**；(b) GPU poller 线程批量 `cudaEventQuery`(非阻塞)/Vulkan fence 轮询 | compute/ |

**per-core io_uring 是关键选择**：每个 worker 自己的 ring，提交和收割都在本核，零跨核同步——和 run-queue 的 per-core 哲学一致。GPU 因为 host 通知机制弱（不能每个 op 阻塞一线程），单设一个 completion 源翻译成 fiber 唤醒。

> **落地状态(io-04, shard-per-core)**：第 0 步收割已接线，走**接缝解耦**（sched 不 include io/）。sched 暴露**纯函数指针 hook** `Scheduler::set_io_poll(fn, ctx)`（`fn` 非 io 类型，故 sched **不**依赖 io——依赖方向 io→sched，io 用 sched 的 Waker）；io/ 的 `IoContext` **每 worker 持一个 proactor**，注册 hook + reap **本 worker 自己的 ring** 后经 `net::complete()` 路由回等待 fiber。worker 循环顶(fire_due_timers 后)非空 hook 才 reap(0)，无 io 注册=零成本。
>
> **shard-per-core（share-nothing，性能天花板）已落地**：连接 fd `bind_fd(worker)` 绑定 owner 核（满足 IOCP 一 socket 一端口的硬约束），其 fiber 用 `Scheduler::spawn_on(worker, f)` **钉在该核**——submit/reap/resume 全程单核，无锁、无跨核同步、cache 热。fiber 亲和的实现**不碰防惊群 Dekker/steal CAS**：每 worker 加一条**专属 MPMC 队列**（`BoundedQueue`，任意线程 push、仅 owner pop、steal 路径不读），`spawn_on`/唤醒走它 + `unpark_specific` 定向唤醒（绕过 `num_searching_` 门，因 pinned fiber 只能在 owner 跑）。**park 合并 ✅ 已落地**：空闲 worker 以完整 timeout(timer deadline/INFINITE)阻塞 GETEVENTS reap(不再 1ms 轮询),跨线程 spawn/steal 经 `wake_worker`(`parker.unpark` + `set_io_wake` hook→`Proactor::wake()` 写 ring 的 eventfd)即时打断——两路都打(worker 可能睡在 futex 或 reap),各自幂等。实测空闲 500ms 的 `io_uring_enter` 从 421→3。co_await send/recv + 两 fiber echo + 32 连接分片跨核 + 真 loopback TCP/UDP echo 双平台(IOCP/io_uring/epoll)真测 + TSan 净(affinity 44/44)。

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

> **已落地（sched-06,`include/hyle/sched/coro.h` header-only）**:`Task<T>`(coroutine frame 走 mem::Allocator 首参惯用法)+ `SuspendFiber`(co_await 让出 fiber:await_suspend 调 suspend() fiber-yield,**返 false** 使 wake 后协程续跑而非二次挂起——这是 §4.1 "co_await 落 fiber 让出" 的正确实现关键)+ `CaptureWaker`(两步 register-then-suspend)。frame 由显式 allocator 背书，执行它的 fiber 栈在挂起期间冻结；fiber 恢复后继续 drive frame。双平台 + WSL TSan(协程帧随 fiber 跨 worker 续跑)净。`co_await` 的三源(io/gpu/timer)统一走 Waker,与裸 fiber 路径共存。

---

## 全节总结：sched/ 设计定稿

**一句话**：HYLE 的调度器 = **一种调度单元（有栈 fiber）+ 一个 M:N 无锁调度器 + 一张唤醒网（completion 脊）**，让 线程/进程/纤程/协程 + io/gpu/timer 全部插进同一套底座，而非各跑各的事件循环。

| 块 | 决策 | 关键依据 |
|---|---|---|
| ① fiber 上下文切换 | **boost.context fcontext 裸原语**（不要 boost.fiber），ucontext 出局 | 比 ucontext 快两个数量级；ucontext macOS 已弃用、非 FFI-safe |
| ② run-queue | per-core 有界 FIFO ring（thief 偷一半）+ 独立 LIFO fast slot + global/overflow 兜底 | Tokio/Go 实证；LIFO slot = completion 接入点 |
| ② work-stealing | spin-then-park + 限制 thief 数 + unpark 接力 | 防惊群、防 N² 偷窃风暴 |
| ② 四类统一 | fiber=调度单元、thread=执行引擎、coroutine=fiber 内挂起点、process=联邦边界 | 进程不共享 run-queue（地址空间不同），靠 io+shmem 联邦 |
| ③ completion | 统一 `Waker{fiber, origin_core}`；per-core io_uring / IOCP / GPU 回调或轮询 | user_data 关联 fiber；唤回 origin_core 的 LIFO slot 保 cache 局部性 |
| ④ 协程/fiber | `co_await` 的 await_suspend 构造 Waker + fiber_yield；两层正交（调度粒度 vs 挂起开销）| 深层让出靠 fiber，写法顺滑靠协程 |

**全部基于 clay catomics 无锁重写，丢弃 clay 单全局锁 thread_pool。** time/ 提供单调钟+时间轮（deadline 不用墙钟），platform/ 提供 park 原语（futex/WaitOnAddress），二者是 sched 的 T2 仆从。

**曾"待实现期定"、现多数已落地**（非设计悬念,是落地选择）：阻塞 syscall 的 blocking 池 ✅ sched-11（懒创建池；sysmon 明确不做，卸载显式）；LIFO slot 轮询预算上限 ✅ 已落地 `kMaxLifoPolls=3`（反饥饿）；GPU completion ✅ 定为**轮询**（`cudaEventQuery` 走 worker reap 模型，同 io，见 compute-design 开工前研究结论）；steal 的 backoff 曲线仍按需微调。

## 2026-07 收口后的维护架构

早期章节保留了设计演进。维护生产代码时以本节为当前真相：

```text
 C++ typed surface                           pure C / FFI surface
 ┌─────────────────────────────┐            ┌─────────────────────────────┐
 │ schedule(fn,ctx)            │            │ hyle_sched_schedule         │
 │ spawn_detached[_on]         │            │ hyle_sched_spawn[_on]       │
 │ spawn_task<T> / JoinHandle  │            │ opaque handles + callbacks  │
 │ sync / channel / cancel     │            │ hyle/c/sched.h (114 funcs)  │
 └──────────────┬──────────────┘            └──────────────┬──────────────┘
                └────────────────────┬──────────────────────┘
                                     v
               ┌──────────────── admission + lifecycle ────────────────┐
               │ Fiber: closed + count combined CAS state              │
               │ InlineJob: open -> per-worker count++ -> recheck      │
               │ drain scans fiber state + per-worker inline counters  │
               │ ordinary shutdown admits descendants                  │
               │ begin_shutdown -> drain[_for] -> stop -> join          │
               └────────────────────┬───────────────────────────────────┘
                                     v
 ┌──────────────────────────── worker i ────────────────────────────────┐
 │ pinned overflow -> pinned bounded queue                              │
 │ periodic global -> LIFO fast slot -> local WsFifoQueue -> steal      │
 │ timer wheel / io poll -> Waker rendezvous -> ready enqueue           │
 │ <=8 by-value InlineJobs, then stackful Fiber/timer/io fairness poll   │
 └──────────────────────┬────────────────────────────────────────────────┘
                        │ queue saturation (cold path, no allocation)
                        v
          ┌─────────────────────────────────────────────┐
          │ intrusive ReadyOverflow FIFO                │
          │ global unpinned + one pinned FIFO / worker  │
          │ mutex only when bounded queues are full     │
          └─────────────────────────────────────────────┘

 blocking syscall / CPU closure
          |
          v
 ┌──────────────── BlockingPool ────────────────────────┐
 │ lazy OS threads + intrusive FIFO + predicate CV      │
 │ idle retirement/reaping; shutdown completion barrier │
 └──────────────────────────────────────────────────────┘
```

### 两条任务腿

- `schedule` 是**固定式轻任务腿**：worker-stack callback，无 fiber 分配/切栈，
  callback 的 `{fn,ctx}` payload 直接按值进入分片队列，不分配 owning record；
  内部完成计数按 worker 分片并独占 cache line，不让 producer 与所有 consumer
  抢一个全局计数；不可挂起或阻塞。它不是“少做 CPUID 的静态腿”那类 SIMD
  概念，而是调度合同选择。
- `spawn_detached`/`spawn_task` 是**完整有栈腿**：可 yield、timer、Waker、同步等待
  和 coroutine drive；代价包含 guarded stack 池和 fcontext。
- 调度循环每次最多消费 8 个短 callback 后必查另一条腿，避免 firehose 饿死 fiber；
  ready bounded
  queue 满时进入 allocation-free intrusive overflow，生产者尤其是本 worker 不自旋
  等待自己来清空队列。

### 完成脊和下游

`Waker` 的 wake-before-suspend rendezvous 是唯一恢复脊。time 的 TimerWheel、
io 的 IOCP/io_uring/epoll completion、compute 的 GPU poller、graph 的
parallel evaluation 和 compute/io 的 BlockingPool 都已直接消费 sched；不是孤立
模块。取消状态随 Fiber 迁移，不使用 thread-local scope。

### Highway thread_pool 与 SIMD 的边界

Highway `hwy/contrib/thread_pool` 是高效的 bulk parallel-for：任务集合预先已知、
worker task 不递归提交、caller 参与 barrier、可利用更丰富的 cache/topology
映射。HYLE sched 是通用异步 M:N runtime：任务可派生、fiber 可挂起、Waker 接
timer/io/gpu、支持取消/优雅关闭/blocking offload。拿 Highway 一次 `Run()` 与
HYLE stackful spawn 逐项比，是不同合同。

可吸收的 Highway 思想中，worker-stack 快任务已落到 `schedule`；caller 参与的
bulk fork-join 和 cluster-aware partition 属于 `graph::parallel_for`/topology
层，而不应污染通用 scheduler。未来只有真实 graph benchmark 显示 barrier 或
NUMA/cache-cluster 落后时才扩这一层。

SIMD 对 scheduler 控制面没有合理收益：队列元素是指针，瓶颈是原子所有权、
cache-line、park/unpark 和 fcontext；批量 SIMD 扫队列既改变并发语义又增加
依赖。SIMD 应在被调度的数据 kernel 内使用，sched 本身保持不依赖 `hyle::simd`。

### 红队维护红线

1. stackful `spawn` 的 admission 和 outstanding 必须保持同一个线性化 CAS
   状态。InlineJob 例外采用已证明的
   `open -> per-worker count++ -> recheck`：close 后竞态提交要么回滚，要么已被
   drain 扫描计入；删掉第二次检查、合并/移动计数 shard 或取消其独占 cache line
   都会分别重新引入 shutdown 漏单或 ARM cache-line 回归。
2. worker 不能在满 ready queue 上自旋等自己消费；所有 reschedule/reinject/
   pinned 路径都必须有 overflow 兜底。
3. shutdown 是完成屏障；并发调用只能有一个 join owner，其余调用等完成。
4. timer/CV 的 waiter 数据必须先完整发布再 arm，notify-vs-timeout 只有一个
   胜者负责 unlink/wake。
5. `Waker` 每 episode 恰消费一次；外部完成源先发布 Waker，再允许 completion。
6. C ABI 只能输出 POD/opaque/callback；外部计数/容量错误必须可恢复，不把恶意
   FFI 输入升级成进程 abort。

## 参考

- [boost.context: fcontext_t vs ucontext_t vs WinFiber（性能差两个数量级）](https://www.boost.org/doc/libs/latest/libs/fiber/doc/html/fiber/overview/implementations__fcontext_t__ucontext_t_and_winfiber.html)
- [WG21 P0876 fiber_context — fibers without scheduler](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p0876r15.pdf)
- [PhotonLibOS：现代有栈协程实现](https://www.reddit.com/r/cpp/comments/1jwlur9/stackful_coroutines_faster_than_stackless)
- [Making the Tokio scheduler 10x faster（per-core 队列 + LIFO slot）](https://tokio.rs/blog/2019-10-scheduler)
- [st3：现代有界 work-stealing 队列（FIFO steal / LIFO|FIFO worker）](https://github.com/asynchronics/st3)
- [An Efficient Work-Stealing Scheduler for Task Dependency Graph（actives/thieves 防惊群）](https://tsung-wei-huang.github.io/papers/icpads20.pdf)
- [CUDA Events / cudaLaunchHostFunc（GPU host 侧完成通知）](https://nvidia.github.io/cccl/unstable/libcudacxx/runtime/event.html)
- [x86-64 / ARM64 调用约定速查（2026）](https://www.codermusings.com/resources/calling-conventions-cheat-sheet)
