# HYLE mem/ 设计

> 极致内存管理（Tier 0，与 sched/container 并列核心）。继承 clay 分配器家族 + MemoryArena，修 5 个 bug，加 NUMA/大页/pinned/fiber 栈池/防伪共享。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`coding-conventions.md`](./coding-conventions.md)。

## clay 现状：好地基 + 5 个确认 bug

- **`Allocator` 单函数指针 vtable**（`allocator.h:24`）：所有 alloc/free/realloc 归一到一个 `realloc_func(inst, ptr, old, new)`，C-ABI 安全、跨 DLL 稳定。**clay 最成熟资产，继承作为注入接缝。**
- **`MemoryArena` 分片无锁定长块池**（`memory_arena.h`）：tagged-pointer CAS 防 ABA，lock-free、ABA 安全。clay 原版是**单** `head_`——8 线程 CAS 同一 cache line,O(threads) 串行化,实测慢 mimalloc per-thread heap 278x(performance.md mem 节)。**已分片根治**:内部 N 片(默认 16)各占独立 cache line,线程 homing 自己片(同线程无争用),free 按地址路由回属主片——278x→1.95x,O(N)争用→O(1) per-shard。这落地了 mem-design 列为"硬约束"却久欠的 per-core 池。残留 1.95x 是原子 CAS 常数因子(vs mimalloc 非原子 local list),进一步逼近需 xcpu_freelist(见下)。单线程仍首选 `StackAllocator`/`GrowablePool`(零原子,超 SOTA)。
- **无隐藏全局堆**:分配显式可控。默认走编译期 `SystemAllocator`(零开销);需要自定义分配器(arena/mock/跨 DLL)时**显式 opt-in 注入** `Allocator` 句柄。注入是**能力,不是强制默认**——曾把"注入贯穿"当铁律,导致每个容器都吃运行时间接调用税(实测 8%),按性能优先纠正:默认零开销,注入按需。
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
| fiber 栈池 | 🔧 **自造**(池化部分✅) | sched/ 的 fiber 栈。**池化已落地**(`MemoryArena::as_allocator` 复用定长块,fork-join +13% 实测);**guard page + 分级栈大小仍待** |
| pinned memory | 🔧 **自造**（封 cudaHostAlloc）| compute/ GPU DMA 用 |
| NUMA / 大页 / cache-line 对齐 | 🔧 自造（部分借 mimalloc 能力）| 极致内存焦点 |

**为什么不自造 malloc 却自造这些**：通用 malloc 要处理任意大小/任意生命周期/碎片，是通用难题；而 arena（一次性释放）、pool（定长）、stack（LIFO）、fiber 栈（定长+guard）是**特化场景**，特化恰恰能比通用 malloc 快——这才是 HYLE "极致内存"的发力点。

## 分层结构

```
mem/
├─ allocator.h     Allocator vtable(运行时注入接缝,opt-in;默认走编译期 SystemAllocator)
├─ system.cc       host RAM backing = mimalloc(替换 clay 裸 malloc/posix_memalign)
├─ arena.h         Arena bump 分配器 + MemoryArena 无锁块池(分片,片数随核数)
├─ pool.h          定长池(推倒重写,修 alloc/free 不匹配)
├─ stack.h         StackAllocator LIFO(推倒重写,修拷贝 bug)
├─ fiber_stack.h   fiber 栈池(定长 + guard page + 池化)→ 给 sched/
├─ numa.h          NUMA 就近放置 hint(OS 分配层:mbind/numa_alloc / VirtualAllocExNuma)
├─ hugepage.h      大页(madvise(MADV_HUGEPAGE) / MEM_LARGE_PAGES / mimalloc THP）
└─ stats.h         分配统计 + 泄漏追踪(修 32/64 下溢)
```

（device/pinned/unified 内存归 compute/ 的 stream-bound buffer,不是 mem/ 的 space——见下「device 内存」节）

## device 内存:边界是显式 copy/命令提交,不是注入 CPU 容器的 space

**纠正一个早期错误设计(用户 2026-06 证伪)**:曾设想"组合一个 `MemorySpace` 对象,同一套 `Arena<Space>` 容器代码透明跑 host/VRAM/unified,跨 space copy 在统一内存上塌缩为 no-op"(旧方案 D)。**这建模了一个不存在的用例**:

- **离散 GPU**:VRAM 指针 CPU 不可解引用;device 由**独立 kernel/shader** operate;host↔device 边界**永远是显式 DMA / 命令提交**。不存在"同一套 CPU 容器代码换个 space 就跑 device"——你不会拿 VRAM-backed 的 `SwissTable` 在 CPU 上 `.find()`。
- **统一内存(DGX/Mac)**:单一 coherent 地址空间,**无传输、无两种指针**。"copy 塌缩 no-op"是在为**不存在的问题**炫技——只有一个 space,没东西可注入/可换。

所以 `MemorySpace` 运行时多态 + `Arena<Space>` 泛化是**过度设计**(代码从未实现,幸而)。正确模型:

- **device 内存 = compute/ 的 buffer,绑 device/stream**(ggml buft 式:枚举/能力查询薄包,内存所有权+stream 绑定自造)。分配走 compute 自己的 device API,**不是**注入 CPU 容器的 allocator。
- **host↔device = 显式 `copy(src, dst, stream)` / 命令提交**(归 compute/,要绑 stream)。边界本就显式,不靠容器透明传递。
- **pinned(page-locked)** = compute 侧为 DMA staging 选的 host allocator,在 buffer 分配点选定,不是容器注入的 space。
- **NUMA** = host-RAM 在 **OS 分配层**的就近放置 hint(mbind/VirtualAllocExNuma 作用在 system allocator backing),**不是**一种 space。
- **mem/ 不需要 MemorySpace 抽象**:它只管 host RAM 的通用/专用分配器。

**tier 干净**:`mem/`(T0)只管 host RAM——通用分配器(mimalloc backing)+ 专用分配器(arena/pool/stack/fiber 栈),**零 GPU 代码、随处可测**;`compute/`(T1)自己管 device/pinned/unified buffer(绑 stream),`copy(src,dst,stream)` 归 compute/(同 coherent 内存时硬件本就无需拷贝)。方向 `compute→mem` 正确,但 mem/ 不向上暴露任何 device 概念。

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
- **→ compute/**:`pinned.h` 供 GPU DMA 的 page-locked host 内存(compute 在 buffer 分配点选用)。device 内存是 compute/ 自己的 stream-bound buffer,**不走 mem/ 的 allocator 抽象**(host↔device 边界显式 copy,见上「device 内存」节)。
- **→ container/**:容器**默认编译期 `SystemAllocator`**(零开销);需 arena/mock 时显式 opt-in 注入 `Allocator`。flat hashmap/carray 的分配走 mem/。
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
2. **mimalloc 封在 `Allocator` vtable 后、不外露 mimalloc 类型**（同 compute 的 ggml_bridge 解耦），换 backing 只动 system.cc。默认路径走编译期 `SystemAllocator`(可内联,零间接);`Allocator` 运行时 vtable 是 opt-in(per-request arena / mock / 跨 DLL)。
3. **device 内存归 compute/ 的 stream-bound buffer,不是 mem/ 的 space**(旧"方案 D MemorySpace 组合式"已证伪并废弃——见上方「device 内存」节)。host↔device 边界是显式 copy/命令提交;mem/ 只管 host RAM。NUMA 是 OS 分配层就近 hint,非 space。

## 参考

- [mimalloc：现代高性能可嵌入分配器（v3 2026）](https://github.com/microsoft/mimalloc)
- [mimalloc 性能基准](https://microsoft.github.io/mimalloc/bench.html)
- [jemalloc vs tcmalloc vs malloc：P99 延迟（2026）](https://www.kunalganglani.com/blog/jemalloc-vs-malloc-tcmalloc-p99-latency)
