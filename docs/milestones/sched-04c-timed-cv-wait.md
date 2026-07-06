# sched-04c：定时 CV 等待（wait_for / wait_until）+ 跨 worker timer cancel

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | sched-04b（fiber sleep / per-worker 轮）|
| **状态** | ✅ 完成（双平台 5+8 测试 + TSan×3 净；红队 6 类 lost-wakeup/UAF 审计无 bug；对标 folly fibers Baton）|

> **范围**：sched-05 显式延后的定时 CV 变体的 **thread-mode 半边**——裸线程 waiter 的 `ConditionVariable::wait_for`/`wait_until`（bare + predicate，返回 `CvStatus`）。覆盖外部提交者与跨进程联邦线程。**fiber-mode 定时等待**（需跨 worker timer cancel）由 sched-04d 落地——本里程碑收口 ✅。
> **cloc**：生产 113（sync.cc 定时路径 + sync.h 定时 API）+ 测试 140。

## 为什么是它（sched-05 登记延后的定时半边）

sched-05 交付了 untimed 核心（无限等待）；`wait_for`/`wait_until` 因依赖 time-02 时间轮被显式拆出。本步先落 **thread-mode**：HYLE 的跨进程联邦 + 外部线程提交路径会从**非 fiber 的线程**把活交给调度器，它们的定时等待不需要时间轮，只需一个有界 park + 超时——简单且正确，可以先发。

## 核心决策：thread-mode 先行，用 park_timeout 不碰轮

- **thread-mode**：裸（非 fiber）waiter 用有界 `Parker::park_timeout` 阻塞，deadline 到了**自己摘链**。没有时间轮、没有跨 worker cancel，因此简单且正确。
- **fiber-mode 拆出**：一个 fiber waiter 要 arm time-02 轮的 timer，其 fire 与 notify 竞态，败者要**跨 worker cancel** 对方——但被唤醒的 fiber 可能在**不同于**持有它 timer 的 worker 上恢复，那需要 per-worker 跨核 cancel 命令队列（time-02 的 headline 分叉点）。故拆 sched-04d；在它落地前，从 worker fiber 调定时 CV 会 assert。

## API（sync.h）

- `CvStatus wait_for(lock, timeout_ns)` / `wait_until(lock, deadline_ns)`：原子释放 `lock`、阻塞到被 notify **或** `timeout_ns` 流逝，然后重取 `lock`。返回 `CvStatus{kNoTimeout, kTimeout}`。deadline key 在 `monotonic_ns`。
- **predicate 形式**：`wait_for(lock, timeout_ns, stop)` / `wait_until(lock, deadline_ns, stop)`——循环 recheck `stop()` 直到成立或 deadline 过；每轮重算剩余时间，故 spurious wake 不延长总等待（deadline 从 now + timeout 捕获一次）；wire 上 predicate 恰好变真时最后一次 check 也不漏。

## 核心难点：notify-vs-timeout 竞态仲裁

waiter 链升级为**双向链表**（`Waiter{next, prev, waker, token, linked}`），使超时的 waiter 能 O(1) 自摘（它持自己的 node 但不持前驱）。竞态由 monitor 锁下的 `linked` 标志仲裁：

- **胜者先动**：notify 先 pop（置 `linked=false`）则 waiter 醒来见已摘，报 `kNoTimeout`；timeout 先摘则 notify 见 node 已不在链，转去 pop 下一个——**不丢唤醒**。
- **node 生命周期**：`finish_wait_timed_` 超时后调 `remove_waiter_`（持 `mu_`，幂等）把自己从链摘掉，故后到的 notify 不会 pop 一个已销毁的栈 node。

## 对抗式自审（挑适用类）

- **① 正确性**：双平台 5+8 测试。bare + predicate 两形；notify-早于-deadline、纯超时、**NotifyVsTimeoutRace** 压力用例（赶胜者/败者两向）。
- **② 性能**：benchmark（15 遍）——notify-早于-deadline Win 赢 `std::cv` **2.0x**、Lin 平；纯超时双平台平（OS 定时器地板，无可超越空间）。
- **④ TSan**：TSan×3 净，含 NotifyVsTimeoutRace 压力。
- **⑤ 残留**：fiber-mode 定时等待显式拆 sched-04d，非略过——本里程碑随 04d 落地后收口。
- **红队 6 类**：lost-wakeup / UAF（超时摘链 vs notify pop、双向链 O(1) 自摘的幂等）逐条审计无 bug。对标 folly fibers Baton（atomic-arbitrated timeout-vs-post）。

## 交付物

- `include/hyle/sched/sync.h`（ConditionVariable::wait_for/wait_until bare + predicate + CvStatus；双向 Waiter 链）
- `src/sched/sync.cc`（finish_wait_timed_ 有界 park + 超时自摘、remove_waiter_、deadline_from_now_/remaining_ns_）
- `tests/sched/`（5+8：bare/predicate × notify-early/pure-timeout + NotifyVsTimeoutRace）

## 参考

- sched-design §2.1（deadline 用单调钟）、§2.4（定时变体登记延后）。sched-05（untimed CV 地基）、sched-04b（per-worker 轮 + park_timeout）。folly fibers Baton（timeout-vs-post 仲裁）。performance.md sched 节。
