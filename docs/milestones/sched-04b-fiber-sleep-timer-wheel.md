# sched-04b：fiber sleep via timer wheel（deadline 到期 wake）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | sched-04a（Waker 脊）、time-02（分层时间轮）|
| **状态** | ✅ 完成（双平台 33/33 + TSan×3 净；红队 7 类竞态审计无 bug）|

> **范围**：sched-04 的 deadline 半边——fiber 睡一段时间而**不烧 worker**。这是 time-02 时间轮经 sched-04a Waker 脊的第一个消费者：sched-design §3 里"timer 到期"funnel 进与 io/gpu 完成同一条唤醒路径。`wait_for`/`wait_until` 定时 CV 等待不在本步（→ sched-04c）。
> **cloc**：生产 113（`timer.h` + scheduler.cc 接线）+ 测试 156。

## 为什么是它（脊的第一个真实定时消费者）

sched-04a 只做了 Waker 接缝本体，deadline 半边被显式拆出等 time-02。time-02 落地后（tokio 位图层级扩为 8 层×64 槽，完整覆盖 `uint64_t` ns deadline），本步把两者接起来：**每个 worker 持一个 `TimerWheel`**，一个要睡的 fiber 在**本 worker** 的轮上 arm 一个 entry（O(1) 无分配）+ `suspend()`；worker 在它本会 park 的时机推进自己的轮、fire 到期项 → `Waker::wake()` 重新注入那个 fiber。deadline 永远 key 在 `monotonic_ns`（绝不用墙钟——墙钟跳变会毁掉 deadline 语义，sched-design §2.1）。

## 核心决策：per-worker 轮 + 冷路径才碰时钟

- **per-worker TimerWheel**（非全局）：sleeping fiber arm 在**本 worker** 的轮，唤醒也走本 worker 的 Waker reinject——与 run-queue 的 per-core 哲学一致，无跨核同步。
- **成本模型（SOTA 纪律）**：timer 只在 park 冷路径被触碰。有可运行 fiber 的 worker **从不读轮、从不读时钟**——一个没有 sleeping fiber 的 workload 为这个 feature 的存在付出**零**。arm 是 O(1)；pre-park advance 是 O(#到期) + O(8 层) 的 next-deadline 扫描。
- **`wheel.empty()` 门**：worker 循环顶的 `fire_due_timers` 先查空——无 timer 直接跳过，不读时钟。
- **park 用 next_deadline 喂 park_timeout**：worker 要 park 时用轮的 `next_deadline_ns` 喂 `Parker::park_timeout`，**不空转**（不做 1ms 轮询）。

## API

- `sleep_for_ns(ns)`：挂起当前 fiber **至少** `ns` 纳秒（向上取整到轮的 1ms tick），然后恢复。必须在 Scheduler worker 上的 fiber 内调用（否则 assert）。`ns == 0` 在下一 tick 恢复。
- `sleep_until_ns(deadline_ns)`：绝对 deadline 变体——挂起到 `monotonic_ns()` 至少达到 `deadline_ns`。已过去的 deadline 在下一 tick 恢复。
- **timed-io 积木**（供 io-06 / C2，见交付物 `timer.h`）：`arm_timeout`/`cancel_timeout` 是**不 suspend** 的低层 arm/cancel，用于把一个超时与并发 io op 组合（recv/accept/connect + deadline，先到者赢）。留给 sched-04d/io-06 编排。

## 唤醒路径（与 io 完成同构）

到期不直接执行回调，而是走 sched-04a 的 Waker rendezvous `reinject`——与 io 完成同一个 `wake_rendezvous`。这正是 sched-design §2.1 的"到期动作"：**把对应 fiber 标记就绪、塞回 run-queue**，不在 timer fire 上下文里跑用户代码。

## 契约：sleeping fiber 仍是在途工作

一个 sleeping fiber 仍计入 `outstanding`：`Scheduler::drain()`/`shutdown()` 会**阻塞到它醒来并结束**。一个睡 N 秒的 fiber 因此把 teardown 拖住 N 秒——若在意，关闭前先 cancel 长睡眠（`timer.h` 明确此契约）。

## 对抗式自审（挑适用类）

- **① 正确性**：双平台 33/33。sleep_for/sleep_until 到期恢复、`ns==0` 下一 tick、多 fiber 交错睡不同时长、已过去 deadline 立即醒。
- **② 性能**：`wheel.empty()` 门保证无 timer workload 零税；arm O(1) 无分配；park 用 next_deadline 不空转。
- **④ TSan**：TSan×3 净（sleep 跨 worker reinject、多 fiber 交错）。
- **⑤ 残留**：`wait_for`/`wait_until` 显式拆 sched-04c（需 notify-vs-timeout 仲裁）；fiber-mode 定时等待的跨核 cancel → sched-04d。非略过。
- **红队 7 类竞态审计**：arm/fire/cancel/reinject 时序、wheel.empty 门、drain 与 sleeping fiber 交互均无 bug。

## 交付物

- `include/hyle/sched/timer.h`（sleep_for_ns / sleep_until_ns + arm_timeout / cancel_timeout 积木）
- `src/sched/scheduler.cc`（per-worker TimerWheel + fire_due_timers + park 用 next_deadline）
- `tests/sched/`（33：sleep 变体 + 多 fiber 交错 + 边界）

## 参考

- sched-design §2.1（timer wheel = sched 前置，到期动作走唤醒网）、§3（Waker 脊复用）。time-02（分层时间轮，本步的时基）。sched-04a（Waker rendezvous reinject）、platform-01（park_timeout）。
