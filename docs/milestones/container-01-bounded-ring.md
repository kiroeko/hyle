# container-01：有界 MPMC ring（Vyukov 移植）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | ✅ |
| **状态** | ✅ 完成（MSVC 19.50 Debug+Release，6 case 全过含 40 万项 MPMC 压力；**Linux/GCC14.2 + TSan 已验，零 data race**）|

> **cloc 留档（`cloc include/hyle/container src/container tests/container`）**：**220 行** = 93 头（bounded_queue.h）+ 126 测试 + 1 CMake。
> **低于估 1-2k，非 stub**——骨架 container-01 只做一个有界 MPMC ring（忠实移植 Vyukov，~95 行算法）；container 重头 hashmap/SwissTable/carray/string/SmallVector/intrusive 全在 container-02..06。MPMC 压力 4 生产者×4 消费者×10 万项总数守恒已验。
> **实现注记**：① queue 内存序严格对齐 canonical Vyukov（per-slot seq、`seq-pos`/`seq-(pos+1)` 判满空、acquire/release/relaxed、pop 后 `seq=pos+mask+1`），非自创。② 控制变量 `enqueue_pos_`/`dequeue_pos_` 各 `alignas(64)` 分独立 cache line 防伪共享——触发 MSVC C4324（"alignas 导致填充"），已在 HyleWarnings 全局 `/wd4324` 抑制（cache-line 对齐是全库设计，C4324 是预期非 bug，abseil/folly 同样抑制）。③ 构造期 OOM 走 `HYLE_FATAL_ASSERT`（非 Status 降级）——run-queue 是启动期一次性构造，分配不到=调度器起不来，startup 不变量 fail-fast 合理（区别于 error-01 的运行期消息 OOM 降级）。④ TSan 是 lock-free 真闸门但 Windows 跑不了，留 Linux CI——诚实缺口。

> **2026-06 更新（WSL 补验，缺口已闭）**：本机 WSL Ubuntu 24.04 + g++-14.2 跑 **TSan** —— MpmcStressConservation（4 生产者×4 消费者×10 万项）+ 全 6 case **零 data race**（`setarch -R` 绕过新内核 ASLR）。④ 的"TSan 缺口"已实测闭合：Vyukov 移植的内存序在真 TSan 下成立。

## 前置依赖

- 硬前置：`mem-01`（buffer 走 `Allocator` 注入）。
- 决策依据：`container-design.md` 行60-62（lock-free 队列范围收缩：**borrow/vendoring，不自造**）、行64-66（container 是 sched 的原语供应商）；`coding-conventions.md` §6（显式 allocator 注入）。

## 目的（为什么做这一步）

sched 的 per-core run-queue 与 global 注入队列都是**有界无锁环**。先把这个原语立起来，sched-01 才有队列可用。

## 在架构中的位置

walking skeleton 第五块（`...→ mem-01 → container-01 → sched-01`）。container（T0）供原语 → sched（T0）组合成调度器。本步只做有界 MPMC ring；hashmap/SwissTable/carray/string/SmallVector/intrusive 是后续 container-02..06。

## 关键设计决策

### D1：vendor/移植 Vyukov，不自造（文档已解张力）
- container 总则是"自造不借"，但 **lock-free 队列是明确例外**（container-design 行60-62）：① MPMC/SPSC ring 出名难写对（ABA/内存序/TSan）；② 它装 POD/指针，**不跨 C-ABI 的 C++ 类型边界，没有自造理由**；③ 自造预算留给差异焦点 hashmap。
- **做法：忠实移植 Dmitry Vyukov 已发表的有界 MPMC 算法**（1024cores.net，rigtorp/folly 同款），内存序严格对齐 canonical。**非 HYLE 自创**——注出处。可选的 FetchContent rigtorp 留作将来（同 mem mimalloc 思路，先内联移植验接口）。

### D2：每槽 sequence 协议 + 控制变量 cache-line 分离
- 每槽 `{atomic<u64> seq; T data}`；`enqueue_pos_`/`dequeue_pos_` 各自 `alignas(64)` 分到独立 cache line（防生产/消费侧伪共享）。**槽本身不逐个 padding**（对齐 canonical：seq 协议容忍槽间共享行，省内存）。
- 内存序：`seq.load(acquire)` / `pos CAS(relaxed)` / `data` 读写 / `seq.store(release)`——一字不差照 canonical。

### D3：Allocator 注入 + T 限 trivially-copyable
- buffer 走注入的 `mem::Allocator`（铁律）。骨架期 T 限 `trivially_copyable`（fiber handle=指针），`static_assert` 钉死，避免在无锁路径上掺构造/析构语义。

## 具体事务

- [ ] `include/hyle/container/bounded_queue.h`：`BoundedQueue<T>`，构造 `(Allocator, capacity=pow2)`，`try_push(T)->bool` / `try_pop(T&)->bool` / `capacity()`。non-copyable/non-movable。
- [ ] **不做**：SPSC 特化、hashmap/SwissTable（container-02/04）、carray/string/SmallVector/intrusive（container-03/05/06）、无界队列。

## 交付物

- bounded_queue.h（header-only 模板）+ 测试。最小演示：单线程 push 满/pop 空/环绕；多线程总数守恒。

## 验收标准

- **编译**：Windows 真编（MSVC 19.50）；模板 header-only 跨平台无平台分支。
- **测试**（T0 高要求）：单线程 FIFO 顺序、容量满拒绝、空拒绝、环绕（push/pop 超过 capacity 轮次）；**多生产者多消费者压力**：N 生产者各 push M 项、N 消费者 pop，验**总数守恒 + 无重复 + 无丢失**。
- **正确性闸门（诚实缺口）**：lock-free 正确性的真闸门是 **TSan + 线性化**——TSan 在 Linux/Clang，**本 Windows 机跑不了**，留 CI。本步只能靠 Windows 多线程压力（能抓丢/重，未必抓全部 race）。**这是 lock-free 代码的已知验证缺口，记在案。**
- **硬指标**：4 生产者×4 消费者各 10万项，pop 出的总数 == push 总数，集合无重复。

## 参考

- `container-design.md` 行60-62、行64-66。
- Dmitry Vyukov，Bounded MPMC queue（https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue）。
- rigtorp/MPMCQueue（同算法的维护实现，可选 FetchContent）。
