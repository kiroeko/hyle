# bench-02：对标 SOTA(不止 std)

| 字段 | 值 |
|---|---|
| **阶段** | P3+(随各模块成熟生长)|
| **模块** | bench/ |
| **状态** | 📋 规划(用户 2026-06-28 提出:"应对比 project-hyle 下其他 C++ 高性能仓库和 std 的区别")|

> **动机**：bench-01 只对标 `std`,但 "比 std 快" 是**低标杆**——std 容器/分配器以慢著称,赢它说明不了 HYLE 世界级。真标杆是 project-hyle 下 clone 的业界最强实现。bench-02 把每个有对应物的 HYLE 组件,**直接对标 SOTA**,诚实暴露差距(打平/接近就是成功;远输就是要修的信号)。

## 对标矩阵(库已在 `C:\public\project-hyle\` clone)

| HYLE 组件 | 主对标 | 次对标 | gate(可接受线)|
|---|---|---|---|
| `SwissTable` | **abseil `absl::flat_hash_map`**(SwissTable 鼻祖)| folly `F14ValueMap` | find/insert 打平 absl ±15% |
| mem `Allocator`(mimalloc backing)| 裸 mimalloc | — | 封装层 <5% 损耗(已部分:mem-02)|
| `bounded_queue`(Vyukov MPMC)| **moodycamel `ConcurrentQueue`** | oneTBB `concurrent_queue` | 吞吐打平 ±20% |
| `ws_fifo_queue`(tokio 式)| tokio(语义参照)| oneTBB | owner push 免同步路径最快 |
| `Hash`(XXH3)| xxHash 原版 | **wyhash**(小 key 备选评估)| 封装 ≈ 原版;小 key 评估 wyhash |
| `StringBuf` intern | folly `FBString` | absl `Cord` | intern 命中快路径对标 |
| (future)`simd/` | **highway** / xsimd / simdutf | (数学核 vendored)| 每后端对标 |
| (future)`sched/` | **marl**(fiber)| tokio(work-stealing 语义)| 切换/偷取对标 |

## 接入方式(增量,不一次接全部)

- **不整体引入**(违"不引入第二调度器/event loop");只在 bench 目标 link 对标库的**头/静态档**,隔离在 `bench/` 不入生产。
- 用 clone 的本地副本(FetchContent `SOURCE_DIR` 指向 `C:\public\project-hyle\<lib>`),不联网拉取。
- 每个对标随对应 HYLE 模块成熟逐个加:container 簇先(absl/F14/moodycamel 已可),simd/sched 待模块落地。
- nanobench `.relative(true)`,对标库设 baseline=1.0,HYLE 相对数=倍率;**诚实打印,不挑数据**(沿用 bench-01 纪律)。

## 待办拆分(子里程碑)

- **bench-02a**：SwissTable vs absl::flat_hash_map + folly F14(container 簇,最高优先——hashmap 是 HYLE 自造焦点 T0)。
- **bench-02b** ✅ **完成**：四队列簇(bounded_queue/ws_fifo/work_stealing_deque/concurrent_vector)对标 moodycamel + oneTBB + **crossbeam-deque(Rust,work-stealing 同族)** + **rigtorp(同约束严格有界 MPMC)**。`bench/container/queue_sota_bench.cc`(gated)+ `bench/rust/crossbeam_queue_bench`。**揪出真"自以为 SOTA":单环 BoundedQueue 4P MPMC 曾输 rigtorp 1.33x**(读 rigtorp 源码查实:根因非"理论墙"也非"fetch_add 票号",而是 ① cell 未对齐 cache line 伪共享 ② CAS 失败 over-eager backoff)。**两个真修**:① **`ShardedQueue`**(N 个 Vyukov 子环消单点,4P 反超 moody 2.6-3.0x / rigtorp 3.4x、线性扩展,弃 sched 用不上的全局 FIFO);② **单环 cache-line pad cell + 去 backoff**(4P 反超 rigtorp 1.09x;原 cpu_relax backoff 是 net-negative,已移除)。双平台 + WSL TSan 零 race。数字见 performance.md 队列簇节。**sched global 注入已切换 ShardedQueue**(28/28 sched 测试 + WSL TSan 零 race,fiber switch 无退化)。
- **bench-02c**：Hash vs xxHash 原版 + wyhash(含小 key wyhash 评估,回答"不同情况不同 hash")。
- **bench-02d+**：simd/sched 对标(模块落地后)。

## 参考

- bench-01(对标 std 设施 + 诚实打印纪律)。performance.md(drop-in 可比 vs std 没有的能力两桶)。conventions §1(不引入第二调度器/不整体引入带 event loop 的库)。
