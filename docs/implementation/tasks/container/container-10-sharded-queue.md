# container-10：ShardedQueue 分片有界 MPMC

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **前置** | container-01（BoundedQueue Vyukov 单环）、mem-01（Allocator） |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台 6/6 + **WSL TSan 零 data race**；**已接入 sched global 注入**） |

> **范围**：N 个 Vyukov 子环（`BoundedQueue` 分片），生产者按 `producer_hint` 分片，消解单环的单点 CAS 争用。**保留有界、放弃全局 FIFO**——这是符合 sched global 注入用途的取舍（work-stealing 本就无全局序）。
> **cloc**：生产 128（`sharded_queue.h`）+ 测试 173（`sharded_queue_test.cc`）。

## 为什么是它（bench-02b 揪出的单点争用）

单个 Vyukov 环有**一个**争用的 `enqueue_pos_`/`dequeue_pos_`；多生产者下这个原子成单点 serialize 瓶颈（CAS cache line 在核间 ping-pong）。bench-02b 实测单环 4P 输 moody——纯是这个单点 CAS。**分片根治**：每生产者一个自己的环 → 生产者热路径**零跨生产者争用**，吞吐随生产者数线性扩展。

## 核心决策：纯组合，不是新无锁协议

每个 shard **就是**已审计的 Vyukov `BoundedQueue`（同 memory order、TSan-clean）。分片是纯组合——**不引入任何新原子协议**，正确性归约为"每 shard 正确" + "shard 挑选/round-robin 抽取碰的是不相交或各自独立正确的环"。这是它能安全落地的根本：没有新的 lock-free 推理负担。

## API（caller-supplied identity，无 thread_local）

- **`try_push(producer_hint, x)`**：`producer_hint` 是调用方稳定的并发身份（worker index / tid hash），选 shard。两个不同 hint 的生产者命中不同环。**无隐藏 thread_local**（fiber 跨 worker 迁移下 thread_local 是 footgun）。THAT shard 满则返 false（有界语义，倾斜下调用方重试/spill）。
- **`try_pop(cursor, out)`**：`cursor` 是每消费者自己的 round-robin 游标；从它开始扫 shard，找到后停在产出的 shard（让喂它的生产者保持该消费者热）。每 shard 检查时都空才返 false。
- **有界保留**：总容量 = num_shards × cap_per_shard。真正需求（背压、无无界增长）保留。

## 与单环的分工（生产走这条）

- **ShardedQueue**：sched `global` 注入——无全局序需求，走这条。**双平台 15 遍实测：4P 反超 rigtorp Win 3.27x / Lin 5.3x**（gap»spread；spread 21-43% 偏大是采样不足，倍数取量级铁赢，见 performance.md queue 节）。
- **单环 BoundedQueue**：真需严格全局 FIFO 时用。单线程赢 rigtorp 1.5x、多线程统计平局（同约束 SOTA）。

## 实现要点

- **allocator-backed 环数组**（非 `std::vector`——charter §1 禁 std 拥有型出可能跨 ABI 的头）。`BoundedQueue` 不可移动，故对每 shard **placement-new** 就地构造（共享本队列的 Alloc）。
- **默认编译期 `SystemAllocator`**（`[[no_unique_address]]` → 0 字节），Alloc 选择传播到每个 shard `BoundedQueue<T,Alloc>`。sched 注入运行时 `mem::Allocator`。
- T 须 trivially copyable（同底层 BoundedQueue 约束）。

## 对抗式自审（六类）

- **① 正确性**：6 测——容量=shard 和、同 hint 同 shard FIFO、单 cursor 扫全 shard、shard 满其他有空、num_shards==1 退化为单 BoundedQueue（FIFO+有界）、**4P/4C×100k MpmcStressConservation**（并集恰好=推入集，无丢无重）。
- **② 性能**：生产者热路径零跨生产者争用；4P 反超 rigtorp Win 3.27x/Lin 5.3x、moody 同量级，线性扩展。
- **③ 规范**：allocator-backed 数组（无 std 拥有容器）；复用 BoundedQueue placement-new 避 std 拥有型成员；默认编译期 SystemAllocator 零字节。
- **④ TSan**：**WSL TSan 零 data race**（含 MpmcStressConservation 4P/4C 守恒压力测试）。
- **⑤ 残留**：无。
- **⑥ 过度复杂**：分片是标准做法，纯组合无新协议；未投机做动态 shard 数/work-stealing（那是 WsFifoQueue 的活）。

## 交付物

- `include/hyle/container/sharded_queue.h`（N 子环 + producer_hint 分片 + cursor round-robin 抽取）
- `tests/container/sharded_queue_test.cc`（6，含 MpmcStressConservation）
- benchmark：`queue_sota_bench.cc`（对标 rigtorp/moodycamel，gated）

## 参考

- container-01（BoundedQueue Vyukov 单环，每 shard 复用它）。sched-design §2（global 注入队列用途）。performance.md queue 节（分片 vs 单环双平台 15 遍对标）。bench-02b（揪出单环单点 CAS 争用的动机）。
