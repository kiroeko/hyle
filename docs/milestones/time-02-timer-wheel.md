# time-02：定时器数据结构（历史任务名保留）

| 字段 | 值 |
|---|---|
| 阶段 | P2 |
| 模块 | time/ |
| 前置 | time-01；`core/intrusive_list.h` |
| 状态 | ✅ 2026-07-25 重新收口 |

> 文件名保留 `timer-wheel` 是为了不破坏 milestone 历史链接；当前产品已经没有
> `TimerWheel`/`TimerSource`。2026-07-25 的真实使用复审用 exact-ns radix
> [`TimerQueue`](../../include/hyle/time/timer_queue.h) 替换了早期 8×64、1ms tick wheel。
> 当前架构只见 [`time-design`](../architecture/time-design.md)，数字只见
> [`benchmarks-time`](../performance/benchmarks-time.md)。

## 原任务与复审结果

原任务要为 sched/io 提供大量可取消 deadline，并为固定 duration 场景提供特化结构。早期
实现选了 Tokio/Linux 风格 hierarchical wheel；它正确解决了 heap cancel O(log n)，但把
1ms tick、8×64 horizon 和一个人为的 `TimerSource` 同形接口写进了公开形状。

真实使用复审发现：

- HYLE 对 scheduler/I/O 的需求是 arbitrary exact deadline + 高频取消，不需要 tick policy；
- Seastar `timer_set` 的 radix 形状同样 O(1) arm/cancel，常驻 bucket 更小且无量化；
- Common queue 的 non-decreasing deadline 前置条件和 general queue 不同，强行共用 concept
  反而掩盖真实合同；
- benchmark 必须同时对 faithful same-algorithm rival，而不能只因赢 heap 就停止设计审查。

因此当前交付为：

| 负载 | 当前结构 | 复杂度 / 合同 |
|---|---|---|
| 任意/混合 deadline，大量取消 | `TimerQueue`（65 radix buckets） | arm/cancel O(1)，exact ns，single-owner |
| 同一 duration 的海量 timeout | `CommonTimeoutQueue` | ordered tail append/cancel O(1)，prefix delivery |

## 复审验收

- 23 个 TimerQueue + 16 个 CommonTimeoutQueue C++ tests，两组 million-operation oracle；
- C ABI general/common queue、Status、visitor re-arm、cancel_all 与 surface manifest；
- sched fiber sleep、condition-variable timeout、io timer-vs-completion 全部迁移；
- Windows/WSL/macOS 正确性与性能，M4 双 QoS，WSL ASan/UBSan；
- early benchmark 的 null-ended Seastar 假形和旧 heap/common 假对手均已纠正。

完整任务汇总与提交见 [`time-module-closure`](./time-module-closure.md)。
