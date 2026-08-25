# container-07：有界 Chase-Lev work-stealing deque

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | (sched-03 前置) |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，13 case 全过含并发守恒）|

> **用途校正（2026-06）**：本 Chase-Lev deque 是 **owner-LIFO（深度优先 / fork-join 语义）**。原打算做 sched-03 调度器本地队列，但 sched-design §2.2 明确调度器要 **owner-FIFO 公平轮转**（LIFO 会让 yield 的 fiber 在单 worker 上独占、推翻 round-robin）。故调度器本地队列改用 **container-08**（tokio 式 FIFO work-stealing ring）；**本 deque 保留给 `graph-01` 的 taskgraph fork-join 执行器**（DFS 对 fork-join 反而是优点：cache 友好）。非废弃，是各就其位。

> **cloc 留档（`cloc work_stealing_deque.h work_stealing_deque_test.cc`）**：**249 行** = 98 头 + 151 测试。header-only 模板，纯算法 + 强并发测试，非 stub。
> **对抗式自审记录**：① **slot 用 `std::atomic<T>`(relaxed)** 而非裸 `T`——Lê 论文用裸数组 + fence 携带序，但 TSan 不模型化裸访问的 fence 序。当前本地 Crossbeam 为泛型 T 使用 volatile，并在源码明确承认同槽 read/write 技术上是 data race/UB；HYLE 不复制这个泛型假象，改为只接受 always-lock-free atomic 的 pointer/word-sized handle，保持 C++ memory model 正确且 TSan-clean。② **空判用有符号差** `(int64_t)(b-t)`——`try_pop_bottom` 把 bottom 减到可能回绕低于 top（空 deque），裸无符号 `t<=b` 会误判非空；crossbeam/st3 同款有符号转换。③ 编译期发现 `StealStatus` 嵌套模板导致 `<int>`/`<uint64_t>` 枚举类型不同、无法比较——**提到 namespace 作用域**（更好的 API，单一枚举）。
> **关键诚实缺口（必须显著记下）**：**TSan 不验 fence。** GCC `-Wtsan` 明告 `atomic_thread_fence` "is not supported with -fsanitize=thread"——TSan **不模型化独立 fence**，所以"TSan 零 data race"**没有**验证本 deque 最难的那部分（Lê 2013 的 seq_cst fence 携带的 StoreLoad 序）。且 x86 是 **TSO 强序**、本机**无 ARM**——fence 的正确性恰恰在最需要验证处（弱内存模型）**未经任何机器验证**。当前 fence 正确性仅靠：① 忠实移植已证明的 Lê 2013/crossbeam 算法；② 20 万项 1-owner-4-thief 守恒压力（能抓丢/重，抓不全 race）。**真闸门是模型检查器(loom/CDSChecker)或真 ARM 跑**——记为 sched-03/CI 的待办，不假装已验。`-Wno-tsan` 仅消警告噪音（否则刷屏 + 破 WERROR），**不消除此缺口**。
> **实现注记**：有界 pow2、`Allocator` 注入、T 限 trivially-copyable；top/bottom 各 `alignas(64)` 防伪共享；不做动态扩容（Chase-Lev 原论文的 CAS 换数组 + 回收是另一难题，HYLE 不需无界）；批量 steal-half 留 sched-03。

## 前置依赖

- 硬前置：`mem-01`（buffer 走 `Allocator` 注入）。
- 决策依据：`container-design.md` 行60-62（lock-free 队列 borrow/移植不自造）、`sched-design.md` §2.2（"有界 deque st3/Chase-Lev 支持批量窃取"）、§2.3（steal 偷一半）。
- 用户拍板：sched-03 队列选型走**方案 2（Chase-Lev/st3 窃取 deque）**而非复用 MPMC——性能优先，owner 快路径近免同步。

## 目的（为什么做这一步）

work-stealing 的本地队列：**owner 单端 push/pop 几乎免同步**（快路径仅 relaxed + 一个 acquire fence），**thief 从另一端 CAS 偷**。这是 tokio/Go 调度器的核心数据结构，比 MPMC ring（每次 push/pop 都 CAS）在 owner 热路径上快得多。把它**单独做出来 + TSan 验透**，再给 sched-03 搭多 worker steal 协议——隔离最难的内存序。

## 在架构中的位置

container（T0）的 lock-free 原语，sched-03 的直接前置。sched-02 的 `RunQueue` 本地 FIFO（Vyukov MPMC）将来可换成本 deque（owner 视角），但本步**只做 deque 原语 + 隔离测试**，不碰 sched 集成（那是 sched-03）。

## 关键设计决策

### D1：移植 Chase-Lev，内存序照 Lê et al. 2013（不自创）
- **不是 HYLE 发明**：Chase-Lev（2005）+ Lê/Pop/Cohen/Nardelli 2013《Correct and Efficient Work-Stealing for Weak Memory Models》给出 C11 原子的**精确内存序**。lock-free deque 是出名难写对的（ABA、ARM 弱序），照搬已证明的 order，不"简化"。
- 三操作、三视角：
  - `try_push_bottom`（**仅 owner**）：`b=bottom(relaxed)`；写 `buf[b%cap]`；`bottom.store(b+1, release)`。
  - `try_pop_bottom`（**仅 owner**）：`b=bottom-1`；`bottom.store(b, relaxed)`；**`seq_cst fence`**；`t=top(relaxed)`；空/单元素分支（单元素与 thief 争用一次 `top` CAS）。
  - `try_steal_top`（**thief**）：`t=top(acquire)`；**`seq_cst fence`**；`b=bottom(acquire)`；若 `t<b` 读 `buf[t%cap]` 然后 `top.compare_exchange(t,t+1, seq_cst,relaxed)`，失败=被抢、返回 EMPTY/RETRY。

### D2：有界（固定 pow2 容量），不做动态增长
- tokio 本地队列也是有界（256）。满了由 sched-03 甩 global。**不做** Chase-Lev 原论文的动态扩容数组（CAS 换数组 + 旧数组回收是额外难点，无界非 HYLE 所需）。容量 pow2、mask 取模、`Allocator` 注入 buffer。

### D3：T 限 trivially-copyable，返回用显式状态
- T = fiber handle/指针/POD，`static_assert`。`try_pop_bottom(out)` 返回 bool，
  `try_steal_top(out)` 返回 `StealStatus::{kOk,kEmpty,kRetry}`；只有成功才写 out，
  不用异常或 `std::optional`，header-only 模板保持内联。

## 具体事务

- [x] `include/hyle/container/work_stealing_deque.h`：`WorkStealingDeque<T>`，
  `try_push_bottom(T)` / `try_pop_bottom(out)` / `try_steal_top(out)` / `capacity()`。
  non-copyable/non-movable。
- [x] **明确不做**：动态扩容、批量 steal-half；这不是未完成项，owner FIFO +
  batch-share 由独立的 `WsFifoQueue` 承担。

## 交付物

- work_stealing_deque.h（header-only 模板）+ 测试（含 TSan）。

## 验收标准

- **编译**：Win x64（MSVC）+ Linux x64（GCC14.2）。
- **测试**：① 单线程 owner LIFO（push 三个、pop 逆序）、空 pop、满 push 拒绝、环绕。② **并发压力**：1 owner push/pop + N thief steal，验**总数守恒 + 无重复 + 无丢失**（每个值恰好被 owner-pop 或某 thief-steal 之一取走一次）。③ pop/steal 争用最后一个元素：只有一方得手。
- **正确性闸门（关键，现在能做了）**：**WSL TSan 零 data race**（`setarch -R`）——这正是选方案 2 的底气，Chase-Lev 的弱序正确性靠 TSan 把关。
- **硬指标**：1 owner（10 万 push/pop 交错）+ 4 thief，取走的并集 = 推入集合，无重无漏；TSan 干净。

## 参考

- Chase & Lev 2005《Dynamic Circular Work-Stealing Deque》。
- Lê, Pop, Cohen, Nardelli 2013《Correct and Efficient Work-Stealing for Weak Memory Models》（C11 内存序权威）。
- tokio `queue.rs`、Go runq、st3 crate（有界 Chase-Lev 实证）。
