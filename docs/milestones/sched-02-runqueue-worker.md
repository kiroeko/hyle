# sched-02：per-core run-queue + LIFO slot + 单 worker 驱动

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **Tier** | 0 |
| **LOC 估** | 2-3k |
| **骨架?** | (skeleton 之后第一步加厚) |
| **状态** | ✅ 完成（MSVC 19.50 Debug+Release，12 sched case 全过含 8 个 sched-02） |

> **cloc 留档（`cloc run_queue.h worker.h worker.cc run_queue_test.cc`）**：**205 行** = 75 头（run_queue.h + worker.h）+ 130 cc（worker.cc + run_queue_test）。纯组合层（复用 container BoundedQueue + sched-01 Fiber），无新外部依赖，精瘦合理、非 stub。
> **对抗式自审抓修（真 bug）**：`schedule_lifo` 原写法在 FIFO 满时**丢失被驱逐的 fiber**（先覆盖 slot 再 push 旧的，push 失败旧的已丢）。改为先腾位、腾不动则原状不动并拒绝（调用方保留 f，sched-03 spill global），**绝不丢**；加 `LifoEvictRejectedWhenFifoFull` 测试证明无丢失。
> **实现注记**：① RunQueue=FIFO ring（container BoundedQueue）+ 单 LIFO slot + `kMaxLifoPolls=3` 预算（tokio）；LIFO 性只在 slot、ring 是 FIFO（sched-design §2.2）。② 删掉一个半成品 `empty()`（曾无条件 return true）——worker 循环以 `next()==nullptr` 为唯一终止信号，不造会撒谎的谓词。③ `Worker::run()` 在调用线程驱动（非 OS 线程，留 sched-03）；yield 的 fiber 回 FIFO 尾轮转。④ `Worker` 的 `HYLE_SCHED_API` 放方法非类（RunQueue 成员非 POD，避 C4251——第三次同款，已补进 coding-conventions）。

## 前置依赖

- 硬前置：`sched-01`（Fiber 切换）、`container-01`（BoundedQueue 做本地 ring）。
- 决策依据：`sched-design.md` §2.2（run-queue 结构：per-core 有界 FIFO ring + LIFO slot + 预算）、§2.2 调度循环。

## 目的（为什么做这一步）

sched-01 只证了"一个 fiber 能切"。本步给出**调度结构**：一个 worker 反复从队列取 fiber、切入、让出后放回，多个 fiber 在单线程上**轮转**。这是 work-stealing（sched-03）之前必须先立的本地结构。

## 在架构中的位置

调度脊的本地层。本步**单 worker、单本地队列**，不碰跨核。global 注入队列 + work-stealing + steal + park 全留 sched-03；completion/waker 留 sched-04。

## 关键设计决策

### D1：RunQueue = 有界 FIFO ring（复用 container）+ 单 LIFO slot
- 本地队列：`container::BoundedQueue<Fiber*>`，**owner FIFO**（push tail / pop head）。容量 256（tokio `LOCAL_QUEUE_CAPACITY`）。
- **LIFO slot**：每核一个单槽，存"下一个就跑"的 fiber（刚被唤醒的，热 cache 优先于本地队列）。sched-04 completion 唤醒落这里；本步先做结构 + 预算。
- **LIFO 预算**：连续从 LIFO slot 取的次数封顶 `MAX_LIFO_POLLS=3`（tokio `MAX_LIFO_POLLS_PER_TICK`），超了把 slot 里的踢回 FIFO 尾，防它饿死队列。
- **校正记忆**（sched-design §2.2）：run_queue 本身是 FIFO，LIFO 性来自这个单槽，不是队列；不要把队列做成 LIFO。

### D2：单 worker 驱动循环（本步在调用线程跑，不起 OS 线程）
- `next()`：LIFO slot（含预算判断）→ 本地 FIFO pop。取不到返回空。
- `run()`：循环 `next()` → `fiber->resume()` → 若 `!done()` 放回 FIFO 尾（让出=可被重新调度），直到队列空 + slot 空。
- **不起 platform::Thread**（留 sched-03 多 worker）——本步在当前线程驱动，先验结构正确，降风险（维护者纪律：不一次堆多处风险）。

### D3：调度器持非拥有 Fiber 指针
- Fiber 生命周期归调用方（测试/将来的 spawn 池）。RunQueue/worker 只持 `Fiber*`。本步不做 fiber 池化（留 mem fiber_stack 里程碑 + spawn）。

## 具体事务

- [ ] `include/hyle/sched/run_queue.h`：`RunQueue`（schedule_fifo / schedule_lifo / next，含 LIFO 预算）。
- [ ] `include/hyle/sched/worker.h` + `src/sched/worker.cc`：`Worker`（持 RunQueue，`run()` 驱动循环）。
- [ ] **不做**：global 队列、work-stealing/steal、park/unpark、completion/waker、多 worker、fiber 池化、spawn API。

## 交付物

- run_queue.h + worker.h/.cc + 测试。演示：往队列塞几个会 yield N 次的 fiber，单 worker `run()` 把它们全部轮转跑完。

## 验收标准

- **编译**：Win x64 真编（MSVC 19.50）。
- **测试**：FIFO 顺序取；LIFO slot 优先于 FIFO；LIFO 预算超限踢回；多 fiber 轮转（各 yield 数次、最终全 done、观察到交替推进）；空队列 next() 返回空。
- **硬指标**：3 个各 yield 3 次的 fiber 入队，单 worker run() 后全部 done，且推进顺序体现 FIFO 轮转（非某个 fiber 独占跑完）。

## 参考

- `sched-design.md` §2.2。
- tokio `queue.rs`（本地 FIFO ring）、`MAX_LIFO_POLLS_PER_TICK`；Go runq。
