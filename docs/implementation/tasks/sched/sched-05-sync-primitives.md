# sched-05：fiber-aware 同步原语（CV + Mutex/WaitGroup/Latch/Channel）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | sched-04a（Waker 脊）、util-05（SpinLock）、platform-01（Parker/wait_on_address）、mem-01（Allocator） |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台 28/28（原 20 + 新 8 Sync）+ **WSL TSan 零 data race**；SOTA 对标 marl） |

> **范围**：sched-design §2.4 缺口补记 + §3 spine 的落地——fiber 感知的阻塞原语。**untimed 核心**（无限等待）本步交付；**定时变体 `wait_for`/`wait_until` 依赖 time-02 时间轮（→ sched-04b），登记延后**，untimed 核心不依赖 `time/`，现在就发。
> **cloc**：生产 290（`sync.h` + `sync.cc`）+ 测试 297（`sync_test.cc`）。

## 为什么是它（M:N 的必然缺口）

`std::mutex`/`std::condition_variable` 阻塞的是 **worker OS 线程**。在 M:N 调度器上，那会连累钉在该核上的**每一个其他 fiber**——fiber 的意义被摧毁。修法：**争用时让出 fiber（离开运行循环），释放时重新注入**——这正是又一次 completion/wake，即 **sched-04a Waker 脊的复用**，不是新机制。

## 核心决策：收敛到一个 CV（marl 的教训）

**不为 mutex/waitgroup/channel/latch 各写一套唤醒机制。** 造**一个** fiber-aware `ConditionVariable`，其余全部薄封装在它之上。这是 marl 的收敛策略，直接采纳。

## API（5 个原语）

- **`ConditionVariable`**：唯一的 fiber-aware 阻塞原语，其余都建在它上。镜像 `std::condition_variable_any`——`wait(lock)` 吃任意 BasicLockable（monitor 锁），阻塞时释放、返回前重取（Mesa 语义，谓词循环 recheck）。`notify_one`/`notify_all` 任意线程/fiber 可调。
- **`Mutex`**：**futex（Drepper 3 态）**，建在 `platform::wait_on_address` 上——**不是** fiber-aware。无争用 lock/unlock = 单 CAS（对齐 std::mutex/marl）；争用时 worker 线程 park 在 futex 字上，**不让出 fiber**。
- **`WaitGroup`**：Go 式"等 N 个任务完成"。`add(n)`/`done()`/`wait()`（fiber-aware，建在 CV 上）。
- **`Latch`**：一次性倒数（`std::latch` 形状）。`count_down`/`try_wait`/`wait`。只减不加，区别于 WaitGroup。
- **`Channel<T>`**：有界 MPMC，两端 fiber-aware。满则 producer 让出、空则 consumer 让出（两个 CV 共享一个 SpinLock + ring buffer）。`close()` 唤醒所有等待者。存储走注入的 allocator（charter：无隐藏堆）。

## 双模（必需，非可选）

waiter 可能是 **fiber**（yield + re-inject）**或裸 OS 线程**（回退 `platform::Parker`）。HYLE 的跨进程联邦 + 外部线程提交路径会从**非 fiber 的线程**把活交给调度器；假设"所有 waiter 都是 fiber"会死锁它们。CV 按"是否运行在 worker fiber 内"**逐次等待**决定模式（`prepare_wait_` 里 `current_waker().valid()` 判定）。

- **fiber 模式**：`finish_wait_` 调 `suspend()` 离开运行循环，notify 驱动的 `Waker::wake()` 重新注入。
- **thread 模式**：block 在 thread_local 的 `ParkToken`（`Parker` + `notified` 原子标志，防跨 episode 的陈旧 unpark）。

## 核心难点：lost-wakeup 窗口 + futex 3 态

1. **lost-wakeup（CV）**：waiter 把自己链入 FIFO 尾 + `waiters_++` 是在**仍持 monitor 锁**时做的（`prepare_wait_`）——notifier 要改谓词必须先拿同一把 monitor 锁，故无法在 waiter 的"检查谓词→入队"之间插入它的"改状态→notify"。
2. **notify 快路径**：无人 park 时 `waiters_==0`，notify 直接返回**不碰内部锁**（waiters_ 由 parking waiter 在 monitor 锁下发布，持同锁的 notifier 必看见非零，不丢唤醒）——这是无争用 `Mutex::unlock` 保持廉价的关键。
3. **node 生命周期**：Waiter 分配在阻塞方自己的栈上，notifier 在锁下**先拷出** waker/token 字段再唤醒——node 可能在 owner resume 的瞬间销毁。thread 模式的 token 放 thread_local（非栈），先 flag 再 unpark，唤醒后绝不再碰 node。
4. **Mutex futex 3 态**（0=空闲/1=持有无 waiter/2=持有有 waiter）：`unlock` fast-path `fetch_sub`，前值非 1（=2）才走 `unlock_slow`（store 0 + wake_one）；被唤醒者 re-exchange 到 2，故仍争用的锁保住 waiter 标志，无 lost wakeup。

## 对抗式自审（六类）

- **① 正确性**：8 测——Mutex 8 fiber×1000 增非原子计数总和精确（丢临界区→错和+TSan race）；WaitGroup 16 worker + 1 coordinator，wait() 后必见满计数；Latch 8 count_down 释放 8 waiter；Channel 单产单消（cap=4 逼满/空等待）+ 4P4C（cap=8，WaitGroup 驱动 close）；CV notify_all 释放 200 fiber；**双模两向**（thread waiter × fiber notify、fiber waiter × thread notify，都赶 suspend 竞态）。
- **② 性能**：CV 快路径无 waiter 零锁；Channel notify 无 waiter 时零锁（63ns/15.9M/s）。**WaitGroup fork-join 与 spawn-2000 对标 marl 双平台统计平局**（15 遍 bench-stat 复核；旧"反超 1.49x/2.17x"是单遍噪声，已纠正）——conditional-notify 消除 futex 唤醒风暴把原本慢 17x 修到平级（非栈池）。**争用 Mutex：Win 改 SRWLOCK 平台分叉反超 marl 1.23x**（旧 futex-over-WaitOnAddress 输 2.66x）、**Lin futex 平局**；无争用单 CAS。见 performance.md 与 sched-03。
- **③ 规范**：`sync.h` 公开头零 std 拥有型容器；API 宏放方法不放类（SpinLock 含 atomic，整类导出触 MSVC C4251，同 Fiber/Parker/Scheduler）；Channel 存储走注入 allocator。
- **④ TSan**：**WSL TSan 零 data race**（含 Mutex 非原子计数、Channel MPMC、CV 200 fiber notify_all）；fiber 跨 worker 迁移撞出的 TSan×fcontext 盲区已由 `fiber.cc` 的 `__tsan_switch_to_fiber` 注解处理（release 零成本）。
- **⑤ 残留**：定时变体 `wait_for`/`wait_until` 显式拆 sched-04b（依赖 time-02 时间轮），非略过。clay `async_value`/`barrier` 语义由 WaitGroup/Latch 覆盖。
- **⑥ 过度复杂**：只造一个 CV，其余薄封装（收敛，非四套唤醒）；未投机做协程胶水（sched-06）、取消（sched-07）。

## 实测纠错（这轮"回头攻性能"的核心）

- **Mutex 初版做成 fiber-aware（争用让出），benchmark 慢 marl 5.8x**——短临界区让出 fiber（~100ns 上下文切换）远贵于就地 park 几 ns。**改 futex 后打平 marl**。学到 marl 的分工：**mutex 用 futex、只 CV 才 fiber-aware**（按等待时长分，不一刀切）。契约：别跨 fiber 让出/阻塞持 Mutex，那种场景用 CV。
- **WaitGroup fork-join 起初慢 marl 17x**，诊断推翻"无栈池"猜想——真根因是**调度器急泊车的 futex 唤醒风暴**，落地 conditional-notify（sched-03 §2.3）把 17x 落后修到**与 marl 双平台统计平局**（15 遍 bench-stat 复核;此前一度写"反超 1.49x"是 nanobench 单遍噪声,已纠正）。这就是"做到 sched 就回头攻性能"的那一轮。

## 交付物

- `include/hyle/sched/sync.h`（ConditionVariable + Mutex + WaitGroup + Latch + Channel<T>）
- `src/sched/sync.cc`（CV prepare_wait_/finish_wait_/notify + wake_one 双模 + Mutex lock_slow/unlock_slow futex 尾）
- `tests/sched/sync_test.cc`（8：Mutex/WaitGroup/Latch/Channel×2/CV notify_all/双模×2）
- benchmark：`sync_bench.cc` + `sync_marl_bench.cc`（gated，对标 marl）

## 参考

- sched-design §2.4（同步原语缺口补记）、§3（Waker 脊复用）。marl（收敛方案：一个 CV + futex mutex）。sched-04a（Waker rendezvous，wake-before-suspend race-free 复用）、platform-01（Parker/wait_on_address）、util-05（SpinLock）。performance.md sched 节（marl 对标数据）。
