# sched-07：取消——CancelToken 树 + CancelScope + CancelCallback + HYLE_CHECK_CANCEL

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | 能力 C3（结构化取消）驱动 |
| **状态** | ✅ 完成（2026-07-05；Win MSVC + WSL io_uring/forced-epoll + WSL TSan 全绿：sched Cancel 16/16 + net_cancel 2/2 三后端 + http_cancel 2/2，TSan 零 race）|

> **范围**：结构化取消（lifecycle-design §1，能力 C3）。相比 sched-10 已发的**平的协作 flag**（JoinHandle::abort + is_cancel_requested），本步的增量价值是**树传播**：cancel 一个 token 连带 cancel 从它派生的每一个 token。真实消费者是 HTTP 请求 fan-out（io-10）：一个请求 handler 派生几个子 task/上游调用，cancel 请求（deadline / 客户端挂断）必须连带 cancel 它派生的整棵子树。
> **cloc**：生产 333（`cancel.h` + `cancel.cc`，含 `io/net_cancel.h`）+ 测试 482（cancel/net_cancel/http_cancel）。**LOC 远低于 2-3k 预测**=复用现有 seam（Waker/Fiber::cancel_flag_/is_cancel_requested/JoinHandle/proactor submit_cancel）不重复造 + C++23 精简，非 stub（20 真测 + TSan + 三后端为证）。

## 为什么是它（C3 驱动，树传播是增量价值）

sched-10 已发平的协作 flag。本步的价值是**树**：`cancel_subtree` 把取消从一个节点级联到整棵后代。不变量（walk 依赖）：**cancel 一节点 = cancel 全后代**——一个已 `cancelled` 的节点蕴含它当前整个子树都已 cancelled，故 walk 可剪枝。

## 核心决策：shard-per-core ⇒ 树上无锁

一棵树钉一个 owner worker（请求的 fiber 用 `Scheduler::spawn_on` 钉核），故一个 token 及其派生的每个 token **只被那一个 worker 线程触碰**。树链（parent*/child list）、回调链、derive/cancel/drop **全非原子、单线程串行**——刻意**不付** tokio 的 `Mutex<Inner>` + try-lock-创建序代价（那是它多线程 token 共享才需的）。

**唯一的原子**是 per-node `cancelled`（`std::atomic<bool>`），且只为兼容 `is_cancel_requested()` seam：`Fiber::cancel_flag_` 是 `atomic<bool>*`，task fiber 跨 await 迁移 worker 时读它需跨线程可见（sched-10 的 root-cause）。x86 上 relaxed/acquire bool load 是普通 mov，零成本；**写只发生在 owner worker**。

借 **tokio-util tree_node** 拓扑 + 析构过继 + 迭代传播（**去 Mutex<Inner> + try-lock-创建序**）+ **folly CancellationCallback** 侵入链 + 注册即查已取消 + 先摘再调（**去 atomic 状态字 + 跨线程注销握手**）。

## API（四件 + 一宏）

- **`CancelToken`**：move-only 单 owner，**无 refcount**——每 `derive()` 新建 node，无 token 克隆/共享（这正是让整棵树能无锁的原因）。`root(alloc)`/`derive()`/`cancel()`/`is_cancelled()`。derive 时若父已 cancelled，子**born-cancelled**（父-cancel-然后-新子 race，同 shard 无锁处理）。drop 一个中间 token 时先把活着的子过继给祖父再释放。
- **`CancelScope`**：RAII 把 token 的 flag 发布到当前 fiber 的 `cancel_flag_` seam，嵌套安全（保存/恢复前一 flag），**零改 join.cc/Fiber**——`is_cancel_requested()`/`abort()` 不变可用。task body 顶构造一个 CancelScope 即 opt-in 该 token。
- **`CancelCallback`**：RAII 侵入式（记录 lives 在注册方的栈帧上，无分配），io 取消注册点。注册即查已取消（构造时已 cancelled 则**立即**跑 fn，否则永远漏）。析构自动 unregister（正常完成的 io op 拆掉它的 cancel 钩子）。
- **`HYLE_CHECK_CANCEL()`**：纯 CPU 循环里的手动取消点——无 await 可搭便车的深 CPU 工作否则不可取消。set 则从 enclosing 函数返回 `Status::Cancelled`（要求返回 Status/Result<T>，采用与当前 `HYLE_ASSIGN_OR_RETURN` 一致的早返回风格）。

## 核心机制：迭代传播 + pop-then-invoke + 析构过继

- **`cancel_subtree` 迭代深度优先**（非递归，防深 fan-out 树爆栈）：`exchange(true)` 标记，只下降进**新 cancel** 的节点，已 cancelled 的子树剪枝；沿 parent 爬回，止于 root（不碰 root 自身的兄弟/父）。
- **`fire_callbacks` pop-then-invoke**：detach 头回调再跑，重复到空。一次 pop 一个（非缓存 next 再 invoke）使 walk 对"回调体注销另一个仍 pending 的回调"健壮——unregister 经 `prev_next` 拼接活链，总是重读当前头，绝不解引用悬垂指针（folly unlink-before-invoke 纪律，做到对兄弟拆卸健壮）。
- **`unlink_node` 析构过继**：中间 token drop 时，把它的子**过继给它的 parent**（tokio move_children_to_parent），防漏取消；若 p 已 cancelled，刚过继的子错过了原级联 → 逐个 `cancel_subtree`。

## io 取消（io/net_cancel.h）

`CancelRecvOp`/`CancelAcceptOp`：**栈 slot 无 heap**（比 net_timeout 的 refcount orphan 简单）——单 waker：cancel 只 `submit_cancel` 不 wake fiber，fiber 只被 op 单一 CQE 唤醒，栈全程冻结有效（经 **tokio-uring Lifecycle** 核实：oneshot op cancel 后恰好一个 CQE，无 double/no-CQE，故无迟到-CQE-UAF）。`await_ready` 已取消短路（不碰 ring）。归一化 **kCancelled = -125**（io_uring -125 / IOCP -995 / epoll 合成 → 统一，消费者不辨后端）。

## 消费者：HTTP 请求 fan-out（io-10 驱动）

请求 root token 派生子 token，一次 cancel 连带取消整棵子 task 树（`http_cancel_test` 4 分支全观测 + 分支隔离）。

## 对抗式自审（挑适用类）

- **① 正确性**：sched Cancel 16/16（析构过继 / born-cancelled / 宽深树 / 回调健壮性）+ net_cancel 2/2 三后端 + http_cancel 2/2。
- **③ 规范**：`cancel.h` 公开头零 std 拥有型容器；node 走注入 allocator（无隐藏 new）；侵入链无 per-op 容器分配。
- **④ TSan**：WSL TSan 零 race（唯一原子 cancelled，其余单线程串行）。
- **红队审 5 面**（遍历 / 过继 / 回调生命周期 / 短路 / 宏卫生）：揪修 `fire_callbacks` 相邻注销 UAF 隐患（改 pop-then-invoke + detach-before-invoke）。
- **⑥ 过度复杂**：刻意去 tokio 多线程锁 + folly 跨线程握手（shard-per-core 让它们不必要），非重复造。**sched 支柱 01~11 全 ✅ 真收口**。

## 交付物

- `include/hyle/sched/cancel.h`（CancelToken + CancelScope + CancelCallback + HYLE_CHECK_CANCEL + detail::CancelNode）
- `src/sched/cancel.cc`（cancel_subtree 迭代 DFS + fire_callbacks pop-then-invoke + unlink_node 析构过继）
- `include/hyle/io/net_cancel.h`（CancelRecvOp/CancelAcceptOp，栈 slot 单 waker，kCancelled=-125 归一）
- `tests/sched/`（cancel/net_cancel/http_cancel，共 20 真测）

## 参考

- lifecycle-design §1（结构化取消，能力 C3）。tokio-util tree_node.rs（拓扑 + 析构过继 + 迭代 cancel，借之**去锁**）、folly CancellationCallback（侵入链 + 注册即查 + 先摘再调，借之**去原子握手**）、tokio-uring Lifecycle（oneshot op cancel 恰一 CQE）。sched-10（JoinHandle::abort / is_cancel_requested seam / Fiber::cancel_flag_）、io-04（shard-per-core / spawn_on）、io-10（HTTP fan-out 消费者）。
