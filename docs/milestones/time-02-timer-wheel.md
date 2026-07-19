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
- common expire 使用 core intrusive list 的 prefix bulk splice；扫描仍为
  O(fired)，整个区间只做一次链表接合。

## 真实性能基线

旧 benchmark 有两处不诚实，已删除：

1. wheel 对手曾是 lazy tombstone heap，但 libevent `min_heap_erase_` 和
   libuv `heap_remove` 都在节点里存 index 并 eager O(log n) erase。
2. common queue 对手曾是 `std::list + unordered_map`，人为收费分配和 hash；
   libevent 实际把 `TAILQ_ENTRY` 内嵌在 event。

新基线预分配所有节点，并让两边都把 expired 节点先转到 ready intrusive
list 再 drain，工作形状一致。节点在 epoch 外预分配并复用，arm/consume 之间
有 compiler memory barrier。2026-07-19、N=4096、每平台 15 个独立进程：

| 负载 | Windows MSVC | WSL GCC14 | Apple M4 Clang |
|---|---:|---:|---:|
| wheel arm+cancel / indexed heap | 7.24 / 16.04 ns（2.23x） | 5.81 / 6.72 ns（1.14x） | 4.92 / 8.13 ns（1.65x） |
| wheel arm+expire / indexed heap | 15.37 / 48.27 ns（3.16x） | 23.37 / 41.36 ns（1.76x） | 13.54 / 22.51 ns（1.66x） |
| wheel 70% cancel churn / indexed heap | 11.44 / 20.16 ns（1.75x） | 14.74 / 15.34 ns（1.05x） | 8.02 / 10.57 ns（1.31x） |
| common arm+expire / intrusive TAILQ | 3.60 / 4.25 ns（1.18x） | 3.13 / 3.94 ns（1.25x） | 4.86 / 2.76 ns（**0.57x**） |
| common arm+cancel / intrusive TAILQ | 2.88 / 3.14 ns（1.10x） | 1.90 / 2.59 ns（1.34x） | 3.17 / 3.09 ns（**0.98x**） |

> 表中括号是 HYLE throughput / rival throughput，即 `rival ns / HYLE ns`。

TimerWheel 三平台三负载全赢。Common 在 Windows/WSL 赢，但 Apple M4 的
expire 明确慢 42.9%，cancel 小幅慢 2.5%；因此它是语义完整的固定时长
specialization，不是跨架构 SOTA 证书。低位标记 null-ended TAILQ 实验没有改善
expire 且伤害 cancel，已撤回；不删除 owner/link/order 安全合同换数字。

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
