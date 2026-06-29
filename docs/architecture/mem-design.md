# HYLE mem/ 设计

> 极致内存管理（Tier 0，与 sched/container 并列核心）。继承 clay 分配器家族 + MemoryArena，修 5 个 bug，加 NUMA/大页/pinned/fiber 栈池/防伪共享。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`coding-conventions.md`](./coding-conventions.md)。

## clay 现状：好地基 + 5 个确认 bug

- **`Allocator` 单函数指针 vtable**（`allocator.h:24`）：所有 alloc/free/realloc 归一到一个 `realloc_func(inst, ptr, old, new)`，C-ABI 安全、跨 DLL 稳定。**clay 最成熟资产，继承作为注入接缝。**
- **`MemoryArena` 无锁定长块池**（`memory_arena.cc`）：tagged-pointer CAS 防 ABA，最亮的资产，直接继承。
- **显式 `Allocator*` 注入贯穿**：无隐藏全局堆。继承为铁律。
- **5 个 bug**（资产负债表已定位，重写依据）：
  1. `allocate(size, alignment)` **静默丢弃 alignment**（`allocator.h:37-40`）——**对 simd/ 是硬伤**（AVX512 要 64B 对齐），必修。
  2. 统计计数 32/64 位下溢（`allocator.cc:277`）。
  3. `ObjectPool<T,false>` alloc/free 不匹配（`object_pool.h:66` vs `:79`）——UB，推倒重写。
  4. `StackAllocator` 拷贝构造丢 `allocator_`（`stack_allocator.h:28`）——推倒重写。
  5. arena allocator 静态 64 槽上限（`memory_arena.cc:211`）。

## 核心决策：通用 malloc 借，专用分配器造

与 io（借 liburing）、compute（借 ggml）、simd（借 sleef 核）同构——**借多年解决的通用基底，造贴 HYLE 需求的专用层**：

| 部件 | 造/借 | 理由 |
|---|---|---|
| 通用 malloc（system allocator 的 backing）| ✅ **借 mimalloc** | 通用分配器是多年解决的难题（mimalloc/jemalloc/tcmalloc）；自造是浪费 + 必然不如。mimalloc：现代标杆、THP/大页内建、MIT、可嵌入 |
| `Allocator` vtable 注入接缝 | 🔧 **继承 clay** | 最成熟资产，C-ABI 边界契约，但**修 alignment 丢弃 bug** |
| arena / region 分配器 | 🔧 继承 MemoryArena + 修 64 槽上限 | 无锁块池是真资产 |
| object pool / stack allocator | 🔧 **推倒重写**（bug 3/4）| clay 实现有 UB |
| fiber 栈池 | 🔧 **自造** | sched/ 的 fiber 栈（定长 + guard page + 池化）|
| pinned memory | 🔧 **自造**（封 cudaHostAlloc）| compute/ GPU DMA 用 |
| NUMA / 大页 / cache-line 对齐 | 🔧 自造（部分借 mimalloc 能力）| 极致内存焦点 |

**为什么不自造 malloc 却自造这些**：通用 malloc 要处理任意大小/任意生命周期/碎片，是通用难题；而 arena（一次性释放）、pool（定长）、stack（LIFO）、fiber 栈（定长+guard）是**特化场景**，特化恰恰能比通用 malloc 快——这才是 HYLE "极致内存"的发力点。

## 分层结构

```
mem/
├─ space.h         MemorySpace 抽象（内存空间/显存专区）：一个物理内存域 + 拓扑属性
│                  （谁可访问 cpu/gpu/both、是否 coherent、带宽级、对齐）；mem/ 实现 SystemRAM space
├─ allocator.h     Allocator vtable（注入接缝，继承 clay，修 alignment bug）
├─ system.cc       SystemRAM space 的 backing = mimalloc（替换 clay 裸 malloc/posix_memalign）
├─ arena.h         Arena<Space> bump 分配器（组合在 space 上）+ MemoryArena 无锁块池（继承）
├─ pool.h          Pool<Space> 定长池（推倒重写，修 alloc/free 不匹配）
├─ stack.h         StackAllocator LIFO（推倒重写，修拷贝 bug）
├─ fiber_stack.h   fiber 栈池（定长 + guard page + 池化）→ 给 sched/
├─ numa.h          NUMA node 作为 MemorySpace 的一种（mbind/numa_alloc / VirtualAllocExNuma）
├─ hugepage.h      大页（madvise(MADV_HUGEPAGE) / MEM_LARGE_PAGES / mimalloc THP）
└─ stats.h         分配统计 + 泄漏追踪（修 32/64 下溢）

（device/pinned/unified 的 MemorySpace 由 compute/ 实现并注册——见下「MemorySpace 抽象」）
```

## MemorySpace 抽象（device 内存方案 D：组合优于继承）

**问题**：异构计算里 host RAM / 显存 VRAM / pinned / unified 是不同内存域、指针不能混用；但统一内存架构（DGX Spark GB10、Apple Silicon）下 host/device 是**同一块物理 coherent 内存**，"两种指针+拷贝"的前提不成立。

**方案 D**：把"是不是显存"从**类型分支**（每个分配器 `switch(MemKind)`）改成**组合一个 `MemorySpace` 对象**：

- **`MemorySpace`** = 一个物理内存域 + 运行时可查属性（cpu/gpu/both 可访问、是否 coherent、带宽级、对齐）。raw alloc/free/map。
- **离散 GPU** 枚举多 space：`{SystemRAM, VRAM, PinnedHost}`，跨 space `copy` = 真 DMA。
- **统一内存（DGX/Mac）** 枚举**一个** coherent space：`{Unified}`，"给 GPU"和"给 CPU"来自同一 space，`copy(同一 coherent space)` **塌缩为 no-op**（抽象自然塌缩，无需特例）。
- **分配器组合在 space 上**：`Arena<Space>` / `Pool<Space>`——同一套机器、任意 space。**「显存专区」= compute/ 提供的一个 device MemorySpace + 在其上建的 arena/pool**。

**零开销保证（热/冷分层）**：
- **热路径**（arena/pool 内 bump，每秒百万次）：`Arena<Space>` 编译期模板、单态化、纯指针算术，**零间接**。
- **冷路径**（选 space / 向 OS/驱动要新 slab，很少）：运行时 `MemorySpace` vtable，一次间接，摊薄≈0。
- clay 的 `Allocator` 本就是函数指针 vtable，D 的 space 选择**不比 clay 现状多一次间接**；统一内存上 copy 塌缩反而**比 MemKind 枚举方案更快**。

**红利**：NUMA node + device memory + unified **统一进同一个 `MemorySpace` 概念**——`numa.h` 不再是独立特例，是 space 的一种。

**tier 干净**：`mem/`（T0）定义 `MemorySpace` 接口 + 通用分配器 + 实现 SystemRAM，**零 GPU 代码、随处可测**；`compute/`（T1）实现 device/pinned/unified space 并注册，依赖 `compute→mem` 方向正确。`copy(src,dst,stream)` 归 compute/（要绑 stream），同 coherent space 时 no-op。

## 新增能力（clay 没有的"极致")

| 能力 | 做法 | 为什么 |
|---|---|---|
| **NUMA 感知** | 按 node 分配（Linux `mbind`/`numa_alloc_onnode`，Win `VirtualAllocExNuma`）| 多路服务器跨 node 访存慢 2-3×；worker 线程的内存就近分配 |
| **大页/THP** | 2MB/1GB 页（mimalloc `MIMALLOC_LARGE_OS_PAGES` + 显式 madvise）| 大张量/大 arena 减 TLB miss |
| **pinned（page-locked）** | `cudaHostAlloc` / `cudaHostRegister` | host↔device DMA 快得多；不锁页 GPU 拷贝走可分页内存慢 |
| **cache-line 对齐 + 防伪共享** | `alignas(64)`，per-core 结构按 cache-line padding | 并发结构（run-queue、计数器）伪共享杀性能 |
| **64B 对齐契约** | `allocate` 真正传递 alignment 到 backing | simd/ AVX512 load/store 要求；修 clay 的 alignment 丢弃 bug |
| **跨核 free（xcpu_freelist）** | 跨核释放推到属主核的无锁 freelist，属主周期 drain（照镜 seastar `memory.cc` 的 `cross_cpu_free_item`）| **work-stealing 必需**：fiber 被偷到别的核运行后，其栈/buffer 在非分配核释放；per-core 分配器不能假设"同核 free"，否则数据竞争或泄漏 |

> **为什么必须加 xcpu_freelist**：mem/ 的 per-core 分配器（arena/pool）天然假设"谁分配谁释放、同核"。但 sched/ 是 work-stealing——fiber 连同它的栈/buffer 会被偷到别的核、在那里结束并释放。seastar 是 share-nothing 无此问题；HYLE 选了 work-stealing，就必须补这块（seastar 已验证的方案）。这是 work-stealing 对 mem/ 的硬约束，不是可选优化。

## 与其他模块的耦合

- **→ sched/**：`fiber_stack.h` 供 fiber 栈（定长 + 尾部 guard page 捕溢出 + 池化复用，避免频繁 mmap）。sched 第一节的栈管理就落在这。
- **→ compute/**：`pinned.h` 供 GPU DMA 的 page-locked 内存；device 内存分配也走 mem/ 抽象（host/device/unified 统一接口，呼应 compute-design 的 buffer 层）。
- **→ container/**：所有容器显式注入 `Allocator*`（继承 clay 铁律），flat hashmap/carray 的分配走 mem/。
- **→ simd/**：64B 对齐契约——`mem/` 保证对齐，`simd/` 才能安全做对齐 load/store。**修 alignment 丢弃 bug 是这条契约的前提。**
- **std::pmr 互操作**：提供薄 `pmr::memory_resource` adapter，让 std 容器也能用 HYLE 分配器（低成本、高互操作价值）。

## 落地清单

1. `Allocator` vtable 继承 + **修 alignment 丢弃**（最优先，simd 依赖）。
2. system allocator 换 mimalloc backing（封在 vtable 后，不外露 mimalloc 类型——同 ggml 解耦）。
3. arena 继承 + 修 64 槽上限；pool/stack 推倒重写修 UB。
4. fiber 栈池（给 sched）、pinned（给 compute）。
5. NUMA/大页/cache-line 对齐。
6. stats 修下溢 + 泄漏追踪（debug 构建）。
7. 单测：对齐保证、并发分配 TSan、泄漏检测、NUMA 就近性验证、benchmark 对标裸 mimalloc（专用分配器应更快）。

## 决策（已定）

1. **通用 backing = mimalloc**（现代、THP 内建、MIT、易嵌入、v3 2026）。jemalloc 多线程老牌但偏大；tcmalloc 强但 Google 生态绑定。
2. **mimalloc 封在 `Allocator` vtable 后、不外露 mimalloc 类型**（同 compute 的 ggml_bridge 解耦），换 backing 只动 system.cc。
3. **device 内存抽象 = 方案 D（MemorySpace 组合式）**——mem/ 定义 `MemorySpace` 接口 + 通用分配器 + SystemRAM 实现（GPU-free）；compute/ 实现并注册 device/pinned/unified space；分配器组合在 space 上；统一内存自然塌缩。详见上方「MemorySpace 抽象」。

## 参考

- [mimalloc：现代高性能可嵌入分配器（v3 2026）](https://github.com/microsoft/mimalloc)
- [mimalloc 性能基准](https://microsoft.github.io/mimalloc/bench.html)
- [jemalloc vs tcmalloc vs malloc：P99 延迟（2026）](https://www.kunalganglani.com/blog/jemalloc-vs-malloc-tcmalloc-p99-latency)
