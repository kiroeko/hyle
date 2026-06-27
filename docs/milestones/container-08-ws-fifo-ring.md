# container-08：有界 work-stealing FIFO ring（tokio queue.rs 移植）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | (sched-03 真前置) |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，19 case 全过含并发守恒；**TSan 零 data race 且覆盖可靠**）|

> **cloc 留档（`cloc ws_fifo_queue.h ws_fifo_queue_test.cc`）**：**279 行** = 128 头 + 151 测试。header-only 模板 + 强并发测试，非 stub。
> **对抗式自审抓修（动手中真错）**：初版我把 steal 的 reserve/commit **写反了**——以为"reserve 推 steal、real 不动"，会导致 owner pop 与 thief copy **读同一槽 → 同一任务执行两次**。读 tokio `queue.rs` 真源后修正：reserve 推 **`real`**、`steal` 滞后（owner pop 从推进后的 real 继续、不撞批次；滞后的 steal 挡住 push 复用正在拷贝的槽——所以容量按 steal 算）；copy 源是**滞后的 steal**（= 旧 head）；commit 把 steal 推到**当前 real**（owner 可能又 pop 了）。这正是"famously hard"的地方,差点 ship 重复执行 bug。另:测试 thief 重试块漏接 `steal_into` 直接返回值 → 丢 id 假失败,已修(统一 absorb helper)。
> **vs container-07 的关键优势（TSan 可靠性）**：本结构用 **packed-head 的 atomic CAS**（非独立 `atomic_thread_fence`），所以 **TSan 真能模型化它的同步**——"TSan 零 race"在这里是**实打实的覆盖**，没有 Chase-Lev 那个 fence 盲区。这也是调度器本地队列选它的额外收益。
> **实现注记**：tokio queue.rs 移植——head=`pack(steal<<32 | real)` 单 atomic、tail 单独；owner push_back 仅 release store(无CAS)、pop CAS head、thief steal_into 偷半(reserve real→拷→commit steal)。有界 pow2、Allocator 注入、slot atomic<T>、head/tail alignas(64)。容量上限 2^31(索引 u32)。

## 前置依赖

- 硬前置：`mem-01`（buffer 走 `Allocator` 注入）。
- 决策依据：`sched-design.md` §2.2（per-core 有界 ring、**owner FIFO push_back/pop head**、thief 偷一半）、§2.3（steal 偷一半）；container-07 用途校正（Chase-Lev=LIFO 不适合调度器，改 FIFO）。
- 用户拍板（2026-06）：sched 本地队列用 **tokio 式 FIFO work-stealing ring**，不用 Chase-Lev LIFO（公平轮转 > 深度优先；性能收益二者皆有）。

## 目的（为什么做这一步）

调度器的 per-core 本地队列：**owner 在 tail 端 push 几乎免同步**（仅 release store），owner 从 head pop、thief 从 head 偷——二者竞争 head 用 CAS。FIFO=**公平轮转**（偷最老的，不饿死），这是 tokio/Go 调度器的本地队列真实结构（区别于 container-07 的 Chase-Lev LIFO/fork-join）。

## 在架构中的位置

container（T0）lock-free 原语，**sched-03 的直接本地队列**。sched-03 每 worker 一个本队列 + 一个 global 注入队列（用 container-01 的 Vyukov MPMC）。本步只做 FIFO ring 原语 + 隔离测试 + TSan。

## 关键设计决策

### D1：tokio queue.rs 的 packed-head（不自创）
- **head 是一个 atomic，打包两个 32 位索引**：高 32 = `steal`（偷取进行中的起点）、低 32 = `real`（真实消费头）。owner pop 与 thief steal 都 CAS 这个 packed head；二者一致时无偷取在进行。tail 单独一个 atomic。
- 三操作：
  - `push_back`（**仅 owner**）：读 head(acquire) 判满；写 `buf[tail%cap]`；`tail.store(tail+1, release)`。**无 CAS**。
  - `pop`（**仅 owner**）：CAS head 推进 real（若无 steal 在进行）；读 `buf[head%cap]`。
  - `steal_into(dst)`（**thief**）：CAS head 把 steal 标记前移"偷一半"的量 → 把那半**拷进调用方的目标队列** → 第二次 CAS 把 real 跟上（提交偷取）。偷半摊薄 steal 成本（§2.3）。
- 内存序严格照 tokio `queue.rs`（owner pop `Acquire/AcqRel`、steal `Acquire`+`Release` 提交、tail `Acquire/Release`）。**不自简化**。

### D2：有界 pow2 + Allocator 注入 + T trivially-copyable
- 容量 256（tokio `LOCAL_QUEUE_CAPACITY`）级、pow2、mask 取模。满了由 sched-03 甩 global。`Allocator` 注入 buffer。T=fiber handle/指针，`static_assert`，slot 用 `atomic<T>`（同 container-07：TSan-clean + 正确）。

### D3：steal_into 偷进"目标队列"（非返回单个）
- tokio 语义：thief 把受害者一半任务**搬进自己的本地队列**，再从中取一个。所以接口是 `steal_into(WsFifoQueue& dst) -> {ok, first}`：偷半进 dst、返回其中一个直接跑。比"偷一个返回一个"摊薄竞争。

## 具体事务

- [ ] `include/hyle/container/ws_fifo_queue.h`：`WsFifoQueue<T>`，`push_back(T)->bool` / `pop()->{ok,val}` / `steal_into(WsFifoQueue& dst)->{ok,val}` / `capacity()`。non-copyable/non-movable。
- [ ] **不做**：与 sched 集成（sched-03）、global 队列（用 container-01）、动态扩容。

## 交付物

- ws_fifo_queue.h（header-only 模板）+ 测试（含 TSan）。

## 验收标准

- **编译**：Win x64（MSVC）+ Linux x64（GCC14.2）。
- **测试**：① 单线程 FIFO（push 10/20/30、pop 得 10/20/30）、空 pop、满 push 拒绝、环绕。② **并发守恒**：1 owner push/pop + N thief steal_into，验总数守恒 + 无重 + 无漏。③ 偷半：受害者有 8 个、一次 steal_into 搬走约 4 个。
- **正确性闸门**：**WSL TSan 零 data race**（`setarch -R`）。注意 packed-head 用的是 atomic CAS（非独立 fence），所以 **TSan 对本结构的覆盖比 Chase-Lev 更实**（无 container-07 那种 fence 盲区）——这点比 07 强。
- **硬指标**：1 owner（10 万 push/pop）+ 4 thief steal_into，并集 = 推入集合，无重无漏；TSan 干净。

## 参考

- tokio `tokio/src/runtime/scheduler/multi_thread/queue.rs`（packed-head、偷半、内存序权威）。
- Go runq（`runqsteal` 偷半同思路）。
- `sched-design.md` §2.2 / §2.3。
