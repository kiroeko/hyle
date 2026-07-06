# sched-04d：fiber-mode 定时等待 + per-worker 跨核 timer cancel

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | 能力 C2（带超时 io）驱动 |
| **状态** | ✅ 随 io-06 落地（能力 C2 反向拉动，如轴 C 预测）|

> **范围**：fiber（非裸线程）带超时等待——sched-04c 拆出的 fiber-mode 半边。**关键：登记时最硬的"per-worker 跨核 cancel 命令队列"被 shard-per-core 消掉了**，故本步能在无那条命令队列的情况下落地。当初推迟是对的（等 C2 定形接口），现由 C2 驱动一次做对。
> **cloc**：随 io-06（`arm_timeout`/`cancel_timeout` ~20 + `TimerEntry::on_fire` 分支）。

## 为什么是它（C2 反向拉动，登记的难点被架构消掉）

sched-04c 落了 thread-mode 定时等待，fiber-mode 因"跨 worker timer cancel"的难度拆出等接口定形。sched-design §2.1 与 sched-04c 都点名过它的难点：一个 fiber waiter arm time-02 轮的 timer，其 fire 与 io 完成/notify 竞态，败者要**跨 worker cancel** 对方；而被唤醒的 fiber 可能在**不同于**持有它 timer 的 worker 上恢复——那本需要 per-worker 的跨核 cancel 命令队列（time-02 的 headline 分叉点，真正 load-bearing）。

**这个最硬的部分被 shard-per-core 消掉了**：连接 fiber 钉核（io-04 的 `spawn_on`），timer arm/fire/cancel + io submit/reap 全在 **owner worker 单线程串行**（经 subagent 核实四条路径），无跨线程，故非原子仲裁 + **无需 MPSC cancel 队列**。当初推迟等 C2 定形是对的判断。

## 核心决策：复用 per-worker 轮，单线程串行仲裁

- **不 suspend 的 arm/cancel**：`sched::arm_timeout`/`cancel_timeout`（scheduler.cc，复用 sched-04b 的 per-worker TimerWheel）不挂起——调用方 arm 一个 timer **并**submit 一个 io op，然后**只 suspend 一次**；resume 时仲裁哪个先 fire。
- **`TimerEntry::on_fire` 回调**：fire_due_timers 分支——sleep 直接 wake / timed-io 走仲裁。
- **单线程免原子仲裁**：fiber 钉核 → arm/fire/cancel/submit/reap/resume 全在 owner worker 串行，用普通标志仲裁（无原子、无 cancel 队列）。

## io 完成 vs timer 到期竞态

io 完成 vs timer 到期，败者取消对方——**io-06 的 net_timeout.h 是它的仲裁编排层**：对称 gate（io 侧仅 `!timer_fired` 时 wake、timer 侧 `on_fire` 仅 `!io_done` 时 wake），每 episode 恰 1 次 wake_rendezvous，不污染下个 episode 的 wake_state_。迟到 CQE 由 io-06 的 heap TimedSlot + refcount 续命处理（见 io-06 篇）。

## 对抗式自审（挑适用类）

- **⑤ 残留 → 已消**：登记时的 per-worker 跨核 cancel 命令队列被 shard-per-core 消掉，非略过——是架构让它不再需要。
- **⑥ 过度复杂**：未投机造 MPSC cancel 队列（等 C2 证明单线程串行足够后，一次做对）；仲裁逻辑落在 io-06 编排层，sched 只提供 arm/cancel/on_fire 三个积木。
- 竞态正确性（首到者赢、败者取消、迟到 CQE 续命）在 io-06 的红队审计中一并覆盖（见 io-06 篇）。

## 交付物

- `include/hyle/sched/timer.h`（arm_timeout / cancel_timeout：不 suspend 的低层 arm/cancel，same-worker only）
- `src/sched/scheduler.cc`（复用 per-worker 轮 + TimerEntry::on_fire 分支）
- 仲裁编排 + 真测在 io-06（net_timeout.h，recv/accept 超时三后端）

## 参考

- sched-design §2.1（跨核 cancel 是 time-02 headline 分叉点）、§2.4。sched-04b（per-worker 轮 + Waker reinject）、sched-04c（thread-mode 定时等待的另一半）。io-06（带超时 io，本步的驱动者 + 仲裁编排层）。io-04（shard-per-core / spawn_on，消掉跨核 cancel 的根因）。
