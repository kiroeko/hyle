# sched-03：work-stealing 多 worker 调度器 + spin-then-park + unpark 接力

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **Tier** | 0 |
| **LOC 估** | 2-3k |
| **骨架?** | |
| **状态** | 进行中（step1 ✅ 完成；step2 park/unpark 待做） |

> **Step 1 cloc 留档（`cloc scheduler.h scheduler.cc scheduler_test.cc`）**：**259 行** = 28 头 + 231 cc(scheduler.cc + 测试)。组合层(复用 container-08/01 + Fiber),非 stub。
> **Step 1 状态**：✅ 完成。Win x64 MSVC + Linux x64 GCC14.2 双平台 15 sched case 全过；**WSL TSan 零 data race**(8-worker×1000-fiber 含 fiber 跨线程迁移,TSan 61ms 干净)。
> **Step 1 实现/自审**：① **Pimpl**——Scheduler 公开头只一个 `Impl*`,std::vector/thread/queue 全在 .cc(避 C4251 + 公开头零 std)。② **fiber 跨线程迁移正确**:队列存 `Fiber*`,A yield→B 偷→B resume,fcontext 栈在堆、线程无关;同一时刻仅一个 worker 持有/resume 某 fiber(队列所有权),TSan 背书无并发 resume。③ outstanding 计数 spawn 先 add 再 publish、done 后 sub,drain 等 0;动态 spawn(seed 内 spawn 500 子)也正确等到。④ 析构调 shutdown 幂等安全(已 join 则 drain 立即返回)。⑤ idle **busy-spin**(yield)——step2 换 park。
> **Step 2（待做,本里程碑下半）**：spin-then-park + Parker + num_searching 封顶 + unpark 接力防惊群(§2.3)。

## 前置依赖

- 硬前置：`sched-02`（Worker/RunQueue 概念 + Fiber 驱动）、`container-08`（每 worker FIFO 本地队列）、`container-01`（global MPMC 注入）、`platform-01`（Thread + Parker）。
- 决策依据：`sched-design.md` §2.2（调度循环、global 注入、steal 协议）、§2.3（spin-then-park + 限 thief 数 + unpark 接力防惊群）。

## 目的（为什么做这一步）

把单 worker（sched-02）升级为 **M:N work-stealing 调度器**：多个 OS 线程各跑一个本地队列，空了去偷别人的，全空了挂起。这是 HYLE 并发执行的核心引擎，io/compute 的 completion 最终都唤醒到这里。

## 在架构中的位置

调度脊的主体。本步完成后，fiber 能在多核上被 spawn、steal、运行。completion/waker 接入留 sched-04，取消留 sched-07，优雅关闭留 sched-08。

## 分两步（风险分解，两提交）

### Step 1（提交 A）：多 worker + work-stealing + busy-spin
- N 个 `platform::Thread` worker，每个持一个 `container::WsFifoQueue<Fiber*>` 本地队列。
- 一个 global `container::BoundedQueue<Fiber*>`（Vyukov MPMC）注入：跨核 spawn / 外部线程提交 / 本地满溢出。
- worker 循环（§2.2）：`local.pop()` → 周期查 `global` → `steal_into` 随机 victim → 都空则 **busy `yield()`**（先不 park）。
- `spawn(Fiber*)`、`shutdown()`（置停止标志 + drain + join）。`thread_local` 当前 worker 指针（spawn 落本地队列）。
- **目的**：先验多线程真跑 fiber + 偷取正确 + 关闭干净，不引入 park 的复杂度。

### Step 2（提交 B）：spin-then-park + unpark 接力防惊群
- 把 busy-spin 换成 **spin-then-park**：找不到活先有限次 steal 尝试 + backoff，再 `Parker::park()`。
- `num_searching` 原子计数封顶 ≤ 核数一半（§2.3 ②）。
- **unpark 接力**（§2.3 ③，tokio idle.rs）：新任务入队且 `num_searching==0` 时只 `notify_one`；被唤醒 worker 转 searcher，找到活且是末位 searcher 时再唤醒一个。
- `sleeping_workers`/`num_searching` 维护，spawn/inject 触发 unpark。

## 关键设计点

- **fiber 跨线程迁移**：队列里是 `Fiber*`。fiber 在 worker A yield、被 worker B 偷走 resume——fcontext 栈在堆上、线程无关，切换正确。**同一时刻只有一个 worker 持有并 resume 某 fiber**（队列所有权保证），不会并发 resume 同一 fiber。
- **busy-spin 终止（step1）**：`stop` 原子标志 + 待完成 fiber 计数；fiber 全 done 且 stop 置位时 worker 退出循环。
- **不做**：completion/waker（sched-04）、取消（sched-07）、关闭两阶段 drain 精细化（sched-08，本步只做能 join 干净的简单版）、fiber 池化（mem fiber_stack 里程碑）。

## 验收标准

- **编译**：Win x64（MSVC）+ Linux x64（GCC14.2）。
- **测试 step1**：N worker 跑 M fiber（各 yield 数次）全部 done；跨 worker 偷取**真发生**（单 worker 塞满、其余靠偷拿到活）；计数守恒（每 fiber 恰好跑完一次）；shutdown 后所有线程 join、无泄漏。
- **正确性闸门**：**WSL TSan 零 data race**——多 worker 调度器是 race 高发区，TSan 是硬闸门。本地队列(container-08)/global(Vyukov) 都已单独 TSan 过，本步验组合 + worker 循环。
- **硬指标 step1**：8 worker、1000 fiber 各 yield 3 次，全 done、计数==1000、TSan 干净。
- **测试 step2**：park/unpark 正确（空闲 worker 真挂起、新任务唤醒）；无丢失唤醒（spawn 后任务必被某 worker 取走）；无惊群（不是每个任务都唤醒全部 worker——可用唤醒计数近似验）。

## 参考

- `sched-design.md` §2.2 / §2.3。
- tokio `worker.rs` / `idle.rs`（unpark 接力、num_searching）、Go `findrunnable`/`wakep`。
