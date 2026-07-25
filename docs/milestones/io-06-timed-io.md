# io-06：带超时的 io（recv/accept/connect + deadline + 取消在途内核 op）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-04（net）、time-02（时间轮）、sched-10（select）|
| **状态** | ✅ 完成（能力 C2）。三后端 × recv/accept 超时 × 泄漏检测 4/4 双平台 + WSL TSan 零 race |

> **范围**：给 net op 加 deadline——`TimedRecvOp`/`TimedAcceptOp`/`TimedConnectOp`。op 与一个 timer 赛跑，先到者赢，输者被取消。调研 folly Baton / seastar with_timeout / tokio-uring 后落地。**顺带落地 sched-04d**（fiber 定时等待 + 竞态仲裁）。
> **cloc**：`net_timeout.h` 167 + proactor/sched/io_context 改 ~130 = ~300（+ `timeout_test` 210）。

## 为什么是它
「几乎所有网络库的生存能力」（tokio timeout / Go SetReadDeadline / asio cancel_after / seastar with_timeout）——没有它，一个慢/死的 peer 会永久占住 fiber + fd + buffer。roadmap 原本漏了，参考库审计揪出。

## 核心决策与两个硬点
### ① 仲裁免原子，但必须对称
连接 fiber 钉 owner worker（shard-per-core）：io ring、timer queue、resume 全在**同一 worker 线程串行**（io completion 在 `io_context::poll` reap、timer 在 `fire_due_timers` fire）。故 `io_done`/`timer_fired` 用**普通标志**（seastar 式免原子；folly 的原子 CAS 只为跨线程 post，不适用）。

**但** `Fiber::wake_rendezvous()` 是 2 方握手（worker-dispatch + 一个 waker）。timed op 有**两个** waker（io + timer）。若两条路径都 wake，第二个会把 `wake_state_` 翻回去并**泄漏到 fiber 下次 suspend** → 丢/伪唤醒。所以**每一侧只在自己赢时 wake**：io 侧 `!timer_fired` 才 wake，timer 侧 `!io_done` 才 wake——保证每 episode **恰一次 wake_rendezvous**。timer 侧作 `TimerEntry::on_fire` 回调（`fire_due_timers` 走它而非盲目 sleep wake），能先查 io_done。

### ② 迟到 completion 生命周期（选项 2：续命）
timer 赢 → 提交 proactor cancel，但那是**异步**的：被取消的内核 op **仍会送一个最终 CQE**（io_uring -ECANCELED / IOCP ABORTED），也可能与 op 正常完成 race。若 slot 在 fiber 栈，fiber 已 resume（-ETIMEDOUT）栈复用 → UAF。故 timed op 用 **heap `TimedSlot` + 非原子 refcount = 2**{kernel op, fiber}（tokio-uring `Lifecycle::Ignored` 同思想），各持有者用完各 release 一次，迟到 CQE 到才归零释放；**普通 op 栈 slot 零分配不变**，只 timed op 付堆分配。

**submit_cancel 返回值被故意忽略且正确**（红队 U5 核对 tokio-uring + liburing + seastar）：kernel ref 由**原 op 的 completion** 而非 cancel 释放。io_uring 保证每个内核消费的 SQE 都有 CQE，cancel 只是让那个 completion **来得更早**的优化，从不是释放 slot 的东西。seastar 干脆不发内核 cancel，纯靠「每 SQE 必有 completion」。

### token 路由
低位 tag 区分：普通 `&IoOperation`（bit0 clear）、timed `&TimedSlot | 1`。cancel op 自身 CQE 用 `kCancelToken`（tagged 但 ≠ kWakeToken）→ 路由器丢弃。`IoContext::~IoContext` **drain orphan**（shutdown 后迟到 CQE 排空，tokio-uring Driver::drop 同款）。

## proactor `submit_cancel` 三后端
io_uring `IORING_OP_ASYNC_CANCEL`（addr=user_data）/ IOCP `CancelIoEx`（扫 inflight 按 tagged token 匹配）/ epoll `EPOLL_CTL_DEL` + **合成 -ECANCELED**（epoll 无迟到 CQE，合成之以对称放 kernel-ref）。

## sched-04d 顺带落地（C2 反向拉动）
`sched::arm_timeout`/`cancel_timeout`（复用 per-worker queue，不 suspend）+ `TimerEntry` callback。当初登记时最硬的「per-worker 跨核 cancel 命令队列」**被 shard-per-core 消掉了**——连接 fiber 钉核，timer arm/fire/cancel + io submit/reap 全在 owner worker 单线程串行，无跨线程，故非原子仲裁 + 无需 MPSC cancel 队列。当初推迟是对的（等 C2 定形接口）。

## 对抗式自审（正确性 / 生命周期 / TSan / 残留）
- **红队审**：核心仲裁 / refcount / 迟到 CQE 时序均正确无 double-free，揪出 2 个泄漏（epoll 超时路径 + shutdown 不 drain，均已修）。
- **计数分配器泄漏检测**（功能断言 + TSan 都抓不到泄漏，专门加）。
- 三后端 × recv/accept 超时 × 泄漏检测 4/4 双平台 + WSL TSan 零 race。
- **⑤ 残留**：`LINK_TIMEOUT` 作 io_uring 专属快路径优化留 backlog（不可移植；主线走应用层 queue+cancel 三后端对称）。

## 交付物
- `include/hyle/io/net_timeout.h`（TimedSlot + complete_timed/timed_on_fire + make_timed_slot/resume_timed/on_submit_failed + Timed{Recv,Accept,Connect}Op）
- `src/io/proactor.cc`（submit_cancel 三后端）；`src/sched/scheduler.cc`（arm_timeout/cancel_timeout）；`src/io/io_context.cc`（timed 路由 + orphan drain）
- `tests/io/timeout_test.cc`

## 参考
- io-design「② 参考库审计真缺口」io-06 节。folly Baton / seastar with_timeout / tokio-uring Lifecycle::Ignored；liburing `io_uring_prep_cancel`（「a CQE is always generated」）。相关里程碑：time-02（时间轮）、sched-04d（fiber 定时等待）、io-04（net 底座）、net_cancel（sched-07 的取消版）。详见 [performance/benchmarks-io.md](../performance/benchmarks-io.md)。
