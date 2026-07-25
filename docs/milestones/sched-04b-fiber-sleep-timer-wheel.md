# sched-04b：fiber sleep（历史文件名保留）

| 字段 | 值 |
|---|---|
| 阶段 | P4 |
| 模块 | sched/ |
| 前置 | sched-04a（Waker 脊）、time-02 |
| 状态 | ✅ 完成；2026-07-25 随 time 复审迁移到 `TimerQueue` |

> 文件名记录最初的 timer-wheel 施工任务；当前源码不存在 `TimerWheel`。迁移没有改变 sched
> 公开 sleep API，但从 1ms tick 改为 exact-ns radix deadline。

## 当前交付

- 每个 worker 持一个 single-owner `time::TimerQueue`；sleeping fiber 在本 worker arm 一个
  stable-address `TimerEntry` 后 suspend，无 heap allocation、无跨核锁。
- worker loop 先用 `timers.empty()` 门跳过无 timer 路径；有 timer 时读取 monotonic now、
  `advance` due entries，再经 sched-04a `Waker::wake/reinject` 回到统一 completion spine。
- park 前读取 `next_deadline_ns` 并计算剩余时长交给 `Parker::park_timeout`，不做 periodic tick。
- `sleep_for_ns` 使用饱和 absolute deadline；`sleep_until_ns` 接受 monotonic deadline。过去或
  零时长 deadline 在下一次 scheduler advance 变为 ready，不阻塞 OS worker。
- `arm_timeout`/`cancel_timeout` 是不 suspend 的 same-worker 积木，供 io timer-vs-operation
  仲裁使用；queue 不直接运行用户 callback。

sleeping fiber 仍计入 outstanding，shutdown/drain 会等它恢复并结束。需要有界关闭的用户应
先用上层 cancellation/timeout policy 结束长睡眠，不能从底层强杀 fiber。

迁移验收包含 sched timer/sleep/CV、io timeout、time 生命周期，Windows/WSL/macOS Release
及 WSL ASan/UBSan。time 的性能证据只见
[`benchmarks-time`](../performance/benchmarks-time.md)。
