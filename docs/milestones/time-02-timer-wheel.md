# time-02：双定时器结构（TimerWheel + CommonTimeoutQueue）

| 字段 | 值 |
|---|---|
| 阶段 | P2 |
| 模块 | time/ |
| 前置 | time-01；`core/intrusive_list.h` |
| 状态 | ✅ 2026-07-19 收口；C++/C ABI、红队生命周期与真实性能基线齐全 |

## 两种结构，各吃一种负载

| 负载 | 结构 | 成本 |
|---|---|---|
| 任意/混合 deadline、arm/cancel 密集 | `TimerWheel` | 8×64、1ms tick；arm 摊还 O(1)，cancel O(1)，无 tombstone |
| 大量 timer 共享一个固定 duration | `CommonTimeoutQueue` | deadline 天然递增；尾插/队首到期/摘链均 O(1) |

两者满足同一个静态 `TimerSource` concept，不用 vtable。source 只把到期节点
移入调用方的 ready list，不执行回调；sched/io 在自己的 completion/wake
脊上消费。

## 为什么 wheel 是 8×64

Tokio 的位图/slot/cascade 结构是 6×64。HYLE 保留同一算法，但把层数扩为
8：旧 6 层只有 `2^36 ms ≈ 2.18 年` horizon，超过范围的公开
`uint64_t` deadline 会被顶层绕回并提前触发，违反“永不早触发”。8 层
`2^48 ms` 覆盖完整 `uint64_t ns` 域，代价是每 worker 多 2KiB slot
sentinel 与最多两个冷路径 bitmap probe。

wheel 将 deadline 向上取整到 1ms tick，advance 将 now 向下取整，所以永不
早触发、最多晚一个 tick。`next_deadline_ns` 返回下一次到期或 cascade
边界，让 worker 精确 park，不做 1ms 空转。

## 所有权与生命周期收口

- `TimerEntry` copy/move 均删除：侵入节点地址必须稳定。
- 节点记录 owner source，而不是一个 bool。跨 source cancel/schedule 会在
  破坏两边链表/count 前 fail-fast。
- source 析构必须为空；entry 析构必须既未调度也未挂在 fired list。
- `schedule_after` 与 common queue 的 duration 算术均饱和。
- `CommonTimeoutQueue::try_schedule` 是事务性的：顺序拒绝不会丢掉旧 arm。
- common expire 使用 core intrusive list 的 prefix bulk splice；ready list
  通过 `consume_all` 一次 detach 后寄存器遍历，节点在 callback 前完成摘链，
  避免每个节点重复维护 sentinel。

## 真实性能基线

旧 benchmark 有两处不诚实，已删除：

1. wheel 对手曾是 lazy tombstone heap，但 libevent `min_heap_erase_` 和
   libuv `heap_remove` 都在节点里存 index 并 eager O(log n) erase。
2. common queue 对手曾是 `std::list + unordered_map`，人为收费分配和 hash；
   libevent 实际把 `TAILQ_ENTRY` 内嵌在 event。

新基线预分配所有节点，并让两边都把 expired 节点先转到 ready intrusive
list 再 drain，工作形状一致。节点在 epoch 外预分配并复用，arm/consume 之间
有 compiler memory barrier。common 另有相反注册顺序的 M4 order-swap，以及
带 owner/order/link 检查的合同匹配 TAILQ。2026-07-19、N=4096、每平台
15 个独立进程：

| 负载 | Windows MSVC | WSL GCC14 | Apple M4 Clang |
|---|---:|---:|---:|
| wheel arm+cancel / indexed heap | 6.86 / 10.42 ns（1.52x） | 5.95 / 6.73 ns（1.13x） | 4.91 / 8.36 ns（1.70x） |
| wheel arm+expire / indexed heap | 21.34 / 49.80 ns（2.33x） | 22.05 / 40.64 ns（1.84x） | 11.83 / 23.74 ns（2.01x） |
| wheel 70% cancel churn / indexed heap | 12.87 / 20.85 ns（1.62x） | 13.63 / 16.01 ns（1.17x） | 7.62 / 11.03 ns（1.45x） |
| common arm+expire / 裸 TAILQ | 3.28 / 4.22 ns（1.29x） | 3.14 / 3.95 ns（1.29x） | 3.52 / 2.73 ns（**0.77x**） |
| common arm+cancel / 裸 TAILQ | 2.70 / 3.13 ns（1.14x） | 1.99 / 2.59 ns（1.30x） | 3.06 / 3.06 ns（**统计平局**） |
| common arm+expire / checked TAILQ | 3.28 / 3.61 ns（1.10x） | 3.14 / 3.21 ns（统计平局） | 3.52 / 4.21 ns（**1.16x**） |

> 表中括号是 HYLE throughput / rival throughput，即 `rival ns / HYLE ns`。

TimerWheel 三平台三负载全赢。Common 的 M4 arm+expire 从 4.86ns 优化到
3.52ns：专用 front-pop、一次 detach 的 `consume_all`、hook-first 状态判别
去掉了实现浪费。相反顺序仍为 3.62/2.75ns，排除“后测者睿频获益”。对完整
owner/order/link 合同，M4 14/15 同向赢 checked TAILQ；对完全不检查的裸
TAILQ 下界仍输约 23%，这是公开安全语义成本，不能宣传成裸操作跨架构 SOTA。

这些数字只比较数据结构核心，不等同于完整 libevent/libuv event loop 吞吐。

## 验收

- 19 个 TimerWheel 测试：边界/cascade/FIFO/cancel-after-cascade/reschedule/
  完整 uint64 horizon/wrong-owner/linked-output/百万 oracle。
- 13 个 CommonTimeoutQueue 测试：顺序/FIFO/front-back cancel/事务重排/
  duration 饱和/wrong-owner/百万 oracle。
- 纯 C handle 生命周期、visitor、NULL、跨 source 和顺序失败覆盖。
- Windows / WSL / Apple M4：time、container intrusive、sched timer/sleep、
  io timeout 回归通过；WSL ASan 与 UBSan 也通过。

## 参考

- Tokio hierarchical wheel（level/slot/occupied bitmap/cascade）
- Linux kernel timer wheel
- libevent `minheap-internal.h`、`event.c` common timeout TAILQ
- libuv `src/timer.c` indexed heap removal
