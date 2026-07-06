# sched-10：JoinHandle + when_all / when_any（结构化并发）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | 能力 C7 驱动；C2 的前置 |
| **状态** | ✅ 完成（Win 53/53 + WSL Release 53/53 + WSL TSan 53/53 零 race）|

> **范围**：结构化并发层（能力 C7，且是 C2 带超时 io / C3 取消的 load-bearing 前置）。参考库通读 tokio/folly/seastar/marl 后落地。`spawn(Fiber*)` 仍返 void（契约不变），新增 `spawn_task<T>(sched, alloc, Task<T>) → JoinHandle<T>`。
> **cloc**：生产 579（`join.h` 367 + `join.cc`/test 212）。

## 为什么是它（拿回任务结果 + select 底层）

sched 之前只能 `spawn(Fiber*)` 发射后不管。要 await 一个任务的结果、abort 它、或"等第一个完成"（超时 io 的 io-vs-timer select），需要一个不拥有 Fiber 的结果句柄 + 一个组等待原语。这是 C2（timeout io：`when_any(io, timer)`）与 C3（取消）的地基。

## 核心决策（参考库共识驱动每一处）

- **JoinHandle 不拥有 Fiber**——只持 refcount'd `JoinState`（结果 slot，走 mem::Allocator）。marl 证明 join 可薄如 WaitGroup(1)+slot；tokio 的 6-bit 状态机是"复用 task cell 兼当 future 和 output"生的复杂度，HYLE 独立 slot 绕开它。refcount 起 2（running task + JoinHandle），最后 drop 释放。
- **完成通知复用 sched-05 CV**（双模：join 的 fiber `suspend()`、裸线程 park）——已 TSan-proven，无手写 rendezvous 状态机。
- **`spawn_task` 拥有 Fiber** 经 heap `TaskDriver`，跑完经 Fiber `on_finish` 钩子在 worker dispatch（控制已离开 fiber、栈不执行）析构=**唯一 sched 拥有 fiber 处**；`spawn(Fiber*)` 的 caller-keeps-alive 契约不变。
- **when_all/when_any 共用栈上 `GroupSignal`**（目标计数 latch，每 task 完成时 signal）：when_all 目标 N；when_any 目标 1 + 记 winner。

## abort 协作式（有栈 fiber 不可抢占）

`abort()` 置 flag，task 在让出点查 `is_cancel_requested()` 后 RAII 退栈——有栈 fiber 不可抢占（folly/fibers 无外部 cancel，marl 无），when_any 的 loser cancellation 同机制。真 io 层取消（IORING_ASYNC_CANCEL / CancelIoEx）是 io-06/C2 的活。drop = detach（task 跑完，结果丢弃）。

## 关键 bug 修复：cancel flag 必须在 Fiber 上，不在 thread_local

task fiber 在 co_await 挂起后经 work-stealing 会在**另一 worker 线程**恢复。若 cancel scope 设在 thread_local，迁移后新线程读不到 → 漏 abort（`Join.AbortIsCooperative` flake 的 root cause，2026-07-03）。修法：flag 存在 **Fiber 对象**上（`push_cancel_scope`/`pop_cancel_scope` 写 `Fiber::cancel_flag_`），随执行迁移（同 fiber.cc 的 TSan fiber 句柄）。这是 M:N 调度器里**任何 per-fiber 状态都不能放 thread_local** 的通例——也是 sched-07 CancelScope 复用同一 seam 的原因。

## 核心难点：GroupSignal UAF 防护 + 自捕获 UAF

- **loser-vs-dead-GroupSignal UAF**（folly WhenN-inl.h:59）：一个败者可能读 `group`、unlock、被延迟；同时 winner fire 组、waiter 醒来 `clear_group()` 每个 state 并返回——**销毁栈上 GroupSignal**——延迟的败者才 `g->on_complete()` 踩已释放内存。修法：`complete_()` 在**仍持 state.mu** 时调 `on_complete`；`clear_group()` 也取 state.mu，故持 mu 跨 on_complete 强制 waiter 的 clear_group 阻塞到我们不再碰 g。锁序 state.mu → g.mu（绝不反向），无死锁。waiter 在 GroupSignal 销毁前从每个 state **卸载** group。
- **自捕获 UAF（TaskDriver）**：TaskDriver 曾在 fiber 自己 entry 内自析构 → fcontext trampoline 读已释放 Fiber（TSan SEGV 抓出）。修法：entry 只产出结果，scheduler dispatch 经 Fiber finish 钩子在**控制离开 fiber 后**调 destroy（安全 free 点）。
- **mimalloc×TSan 地址复用误报**（同 KI-2 族）：churn 压测换 TSan 可见分配器隔离（malloc 版全净证明逻辑无 race）。

## 对抗式自审（挑适用类）

- **① 正确性**：Win/WSL 53/53——join 拿结果、abort 协作退、when_all 等全、when_any 记 winner + 败者可 abort/join。
- **④ TSan**：WSL TSan 53/53 零 race（协程/fiber 跨 worker 迁移 + GroupSignal 卸载 + cancel flag 迁移）。
- **红队/自审**：修一个自捕获 UAF（dispatch 收割）+ 一个 abort thread_local flake（flag 移上 Fiber）+ GroupSignal loser UAF（持 mu 跨 on_complete）。
- **⑥ 过度复杂**：JoinHandle 不拥有 Fiber（marl 薄 slot 证可行），绕开 tokio 6-bit 状态机；真 io 取消留 io-06 不投机。

## 交付物

- `include/hyle/sched/join.h`（JoinHandle<T>/<void> + spawn_task + when_all/when_any + detail::JoinState/GroupSignal/TaskDriver + is_cancel_requested）
- `src/sched/join.cc`（push/pop_cancel_scope 写 Fiber::cancel_flag_ + is_cancel_requested）
- `tests/sched/`（53，含 AbortIsCooperative）

## 参考

- sched-design §2.4（JoinHandle + when_all/when_any 补记：不拥有 Fiber、cancel flag 不放 thread_local、GroupSignal 卸载 folly UAF 防护）。tokio JoinHandle（drop==detach）、folly WhenN（UAF 防护）、seastar、marl（薄如 WaitGroup(1)）。sched-05（CV 双模阻塞）、sched-06（Task<T>）、sched-07（复用 cancel_flag_ seam）、io-06（真 io 取消）。
