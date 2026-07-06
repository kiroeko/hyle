# sched-08：优雅关闭（两阶段 drain + 逆序清理）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | 能力 C4（优雅关闭）驱动；触发靠 signal_set（C10 ✅）|
| **状态** | ✅ 完成（2026-07-05；双平台 sched 85/85（+3 shutdown）+ WSL TSan 零 race）|

> **范围**：runtime 优雅关闭（能力 C4）——为 server（长命 fiber：accept 循环、连接循环，永不自然结束，故 plain `drain()` 会永远阻塞）做两阶段排空。调研 seastar gate/abort_source + tokio TaskTracker/runtime shutdown + folly + marl 后落地。
> **cloc**：生产 ~60（scheduler accepting/begin_shutdown/drain_for/shutdown_requested）+ 测试 130（shutdown_test）。

## 为什么是它（server 场景 plain drain 会死等）

批处理 `drain()` 等每个 fiber 自然结束——但 server 的 accept/连接循环**永不自然结束**，plain drain 会永远阻塞。需要一个"停止接受新活 + 让长命 fiber 观测到并收尾"的协作协议。触发器已就位=io-09 的 signal_set（SIGTERM/SIGINT）。

## 关键结论：优雅关闭只需平信号，不需取消树

调研后的核心判断：**优雅关闭只需平的关闭信号广播 + 在途计数归零，不需 sched-07 取消树**（树是正交的"cancel 一个 task 子树"问题）。所有参考库关闭路径都用平信号——seastar `abort_source` / tokio shutdown token 皆平的。这条判断避免了把关闭错误地耦合到取消树。

## 核心决策：两阶段

- **PHASE 1 · `begin_shutdown()`**：置 `accepting=false`（**≠ `stop`**：accepting=false 拒新 spawn 但 worker 继续跑排空；stop=true 才结束 worker 循环）+ `wake_worker` 广播全 worker（唤醒 park/reap 的，让钉核长命 fiber 观测）。`spawn`/`spawn_on` 在 `!accepting` 时**拒绝**（不计数不入队，seastar gate / tokio closed OwnedTasks 模式）。
- **PHASE 2 · `drain_for(timeout_ns)`**：等 outstanding 归零，超时返 false（`0` = 永远 = `drain()`）。然后调 `shutdown()` 停 + join worker。
- **`shutdown_requested()`**：自由函数——长命 fiber（accept/连接循环）在让出点查（经 tls_owner），观测到即 break 循环自然收尾（读一个 scheduler-wide 平 flag，一次 relaxed load；非 worker fiber 调返 false）。典型用法 `when_any(shutdown_wait(), accept())`。

## 有栈 fiber 不可抢占 ⇒ 超时只能协作退

有栈 fiber **不可抢占** → 超时只能协作退 / detach 泄漏栈（进程退出可接受，tokio 对阻塞线程同样放弃 join 而非强杀）。**绝不强杀**。现有 `shutdown()`（= drain + stop + wake + join）保留给批处理路径。

## 逆序清理

seastar at_exit 式回调栈。shard 有序关闭当前靠 `shutdown()` 的 wake-all + join；子系统逆序释放由 RAII 覆盖（IoContext 析构 drain orphan 已就位，见 io-06）。复杂 at_exit 栈待真实多子系统消费者再补（charter：无投机复杂度）。

## 对抗式自审（挑适用类）

- **① 正确性**：双平台 sched 85/85（+3 shutdown：长命 fiber 观测信号退出 / spawn 拒绝 / drain_for 超时）。
- **④ TSan**：WSL TSan 零 race（accepting 原子 + 跨线程广播）。
- **⑤ 残留**：复杂 at_exit 逆序栈待多子系统消费者，非略过——当前 RAII + shutdown wake-all/join 已覆盖单/双子系统。
- **⑥ 过度复杂**：刻意不引入取消树（正交问题）；不强杀 fiber（不可抢占，协作退 + detach 泄漏是 tokio 同款取舍）。

## 交付物

- `include/hyle/sched/scheduler.h`（accepting / begin_shutdown / drain_for + shutdown_requested 自由函数）
- `src/sched/scheduler.cc`（两阶段 + wake_worker 广播 + spawn 拒绝 + tls_owner flag 读）
- `tests/sched/shutdown_test.cc`（130：长命 fiber 观测退出 / spawn 拒绝 / drain_for 超时）

## 参考

- sched-design §2.4（阻塞现实 / 关闭）、全节总结（优雅关闭已落地）。seastar gate/abort_source（平信号 + gate 拒新活）、tokio TaskTracker/runtime shutdown（closed OwnedTasks + 放弃 join 阻塞线程）、folly/marl。io-09（signal_set，C4 触发器）。sched-07（取消树，正交——刻意不用于关闭）。
