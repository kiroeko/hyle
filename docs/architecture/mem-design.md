# HYLE mem/ 设计

> 极致内存管理（Tier 0，与 sched/container 并列核心）。继承 clay 分配器家族 + MemoryArena，修 5 个 bug，加 NUMA/大页/pinned/fiber 栈池/防伪共享。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`coding-conventions.md`](./coding-conventions.md)。

## clay 现状：好地基 + 5 个确认 bug

- **`Allocator` 注入 vtable**（`allocator.h`）：clay 原是**单**函数指针 `realloc_func(inst,ptr,old,new)`(挤 alloc/free/realloc 进一个、丢弃 alignment)。**HYLE 已纠正**:拆成 3 个 C 函数指针 `{alloc,free,realloc}`(各自寄存器友好、带 alignment、无派发分支,仍 C-ABI 跨 DLL 稳),且**降为 opt-in**——默认走编译期 `SystemAllocator`(内联 mimalloc,零接缝税),运行时注入只给真要多态的场景(arena/mock/跨 DLL)。
- **`MemoryArena` 分片无锁定长块池 + TLS freelist**（`memory_arena.h`）：tagged-pointer CAS 防 ABA，lock-free、ABA 安全。clay 原版是**单** `head_`——8 线程 CAS 同一 cache line,O(threads) 串行化,实测慢 mimalloc 278x。**三层组成（不同场景不同路径）**:① **TLS freelist 缓存**（tbb/mimalloc 形，同线程 free→alloc 纯指针栈、无原子）= 同线程 churn 热路径；② **magazine 批量转移**（`kRefillBatch`=16）——TLS miss 一次 CAS 从属主片 refill K 块（bounded-walk pop）、TLS 溢出按片一次 CAS flush 一段（chain-prepend push），工作集>TLS cap 时摊销 ~1/K CAS；③ **分片 Treiber 共享后备**（片数=核数向上取 2 幂≤128，free 按地址路由属主片）= 批量转移的底层栈。**性能定位（2026-07-21 审计+magazine，见 `docs/performance/benchmarks-mem.md` §2b）**：旧"同线程 ping-pong 对 tbb 赢 1.4–1.7x"是**虚高**（深度 1、只测 TLS 快路 vs tbb tcache，从不回落 shard）。真压共享路径的 section D（工作集>cap 的 BURST + 跨线程 handoff）：对**同契约** `pmr::synchronized_pool` **16–27x sign 稳赢**；对 **oneTBB scalable**——跨线程 **打平**（0.90x TIE）、BURST **输 2.3x**（magazine 前分别输 2.75x/9.4x，批量化后追平/大幅缩小）。BURST 残差=契约差价（tbb/mi per-thread heap 同线程非原子本地链 vs HYLE 真共享 slab 每 16 块 1 次 CAS）。**选型**：单线程/per-thread 无界 → `GrowablePool`；真共享有界定长、对手是 std 池 → `MemoryArena`（现含 TLS+magazine，碾 pmr、跨线程平 tbb）；要极致 MT 通用 malloc → mi/tbb。
- **无隐藏全局堆**:分配显式可控。默认走编译期 `SystemAllocator`(零开销);需要自定义分配器(arena/mock/跨 DLL)时**显式 opt-in 注入** `Allocator` 句柄。注入是**能力,不是强制默认**——曾把"注入贯穿"当铁律,导致每个容器都吃运行时间接调用税(实测 8%),按性能优先纠正:默认零开销,注入按需。
- **5 个 bug**（资产负债表已定位，重写依据）：
  1. `allocate(size, alignment)` **静默丢弃 alignment**（`allocator.h:37-40`）——**对 simd/ 是硬伤**（AVX512 要 64B 对齐），必修。
  2. 统计计数 32/64 位下溢（`allocator.cc:277`）。
  3. `ObjectPool<T,false>` alloc/free 不匹配（`object_pool.h:66` vs `:79`）——UB，推倒重写。
  4. `StackAllocator` 拷贝构造丢 `allocator_`（`stack_allocator.h:28`）——推倒重写。
  5. arena allocator 静态 64 槽上限（`memory_arena.cc:211`）。

## 核心决策：通用 malloc 借，专用分配器造

与 io（借 liburing）、compute（借 ggml）、simd（借 Highway/xsimd 数学核）同构——**借多年解决的通用基底，造贴 HYLE 需求的专用层**：

| 部件 | 造/借 | 理由 |
|---|---|---|
| 通用 malloc（system allocator 的 backing）| ✅ **借 mimalloc** | 通用分配器是多年解决的难题（mimalloc/jemalloc/tcmalloc）；自造是浪费 + 必然不如。mimalloc：现代标杆、THP/大页内建、MIT、可嵌入 |
| `Allocator` 注入接缝 | 🔧 **改造 clay** | C-ABI 边界契约;但拆单函数→3 指针(消 Win64 压栈+派发分支)、修 alignment 丢弃、降 opt-in(默认编译期 SystemAllocator) |
| 默认分配器 `SystemAllocator` | ✅ **自造(编译期内联)** | 无状态、内联 mimalloc、零接缝税(实测 97.1% 平裸 mi);容器默认参,注入是 opt-in |
| arena / region 分配器 | 🔧 继承 MemoryArena + 修 64 槽上限 | 无锁块池是真资产 |
| object pool / stack allocator | 🔧 **推倒重写**（bug 3/4）| clay 实现有 UB |
| fiber 栈池 | ✅ **自造**(mem-08) | sched/ 的 fiber 栈。`mem::FiberStack` = `VmRegion::reserve_guarded`（低 1 页永久 guard）+ 分级 `Small/Default/Large` + `FiberStackPool` 回收；`Fiber` 只有 `(size,…)` 与 `(FiberStack,…,pool*)` 两个构造，`spawn_task` 走 `Scheduler` 持有的 per-tier `FiberStackPool`（foreign acquire / worker-dispatch release，SpinLock 往返）复用栈，省每次 3 次 mmap/VirtualAlloc syscall |
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
├─ fiber_stack.h   fiber 栈(定长 usable + guard + 分级 + FiberStackPool)→ sched::Fiber
├─ numa.h          NUMA 就近放置 hint(OS 分配层:mbind/numa_alloc / VirtualAllocExNuma)
├─ hugepage.h      大页(madvise(MADV_HUGEPAGE) / MEM_LARGE_PAGES / mimalloc THP）
└─ stats.h         分配统计 + 泄漏追踪(✅ mem-06a：opt-in StatsAllocator，修 clay 无符号下溢)
```

（device/pinned/unified 内存归 compute/ 的 stream-bound buffer,不是 mem/ 的 space——见下「device 内存」节）

## device 内存:边界是显式 copy/命令提交,不是注入 CPU 容器的 space

device/pinned/unified 内存归 compute/ 的 stream-bound buffer,**不经 mem/ 的 CPU allocator 抽象**——host↔device 边界本就是显式的,不存在"同一套 CPU 容器代码换块内存就跑 device"的用例:

- **离散 GPU**:VRAM 指针 CPU 不可解引用;device 由独立 kernel/shader operate;host↔device **永远是显式 DMA / 命令提交**。不会拿 VRAM-backed 的 `SwissTable` 在 CPU 上 `.find()`。
- **统一内存(DGX/Mac)**:单一 coherent 地址空间,**无传输、无两种指针**——没有可"塌缩"的拷贝,也没东西可在 space 间换。

所以 mem/ 不引入"内存空间"抽象去统一 host/device(那建模了一个不存在的用例)。正确模型:

- **device 内存 = compute/ 的 buffer,绑 device/stream**(ggml buft 式:枚举/能力查询薄包,内存所有权+stream 绑定自造)。分配走 compute 自己的 device API,**不是**注入 CPU 容器的 allocator。
- **host↔device = 显式 `copy(src, dst, stream)` / 命令提交**(归 compute/,要绑 stream)。边界本就显式,不靠容器透明传递。
- **pinned(page-locked)** = compute 侧为 DMA staging 选的 host allocator,在 buffer 分配点选定,不是容器注入的 space。
- **NUMA** = host-RAM 在 **OS 分配层**的就近放置 hint(mbind/VirtualAllocExNuma 作用在 system allocator backing),**不是**一种 space。
- **mem/ 只管 host RAM** 的通用/专用分配器,不引入"内存空间"抽象去跨 host/device 统一。

**tier 干净**:`mem/`(T0)只管 host RAM——通用分配器(mimalloc backing)+ 专用分配器(arena/pool/stack/fiber 栈),**零 GPU 代码、随处可测**;`compute/`(T1)自己管 device/pinned/unified buffer(绑 stream),`copy(src,dst,stream)` 归 compute/(同 coherent 内存时硬件本就无需拷贝)。方向 `compute→mem` 正确,但 mem/ 不向上暴露任何 device 概念。

## 分配器分层:默认编译期绑定(零开销),注入是 opt-in

**问题**:把运行时 `Allocator` vtable(函数指针 `{alloc,free,realloc,state}`)当**强制默认**,让每个容器每次分配都付一次不可内联的间接调用——实测 churn +8%/touch +12%(vtable 拆 3 指针 + fast-path 后的地板,原 +21%)。这 8% 是"为没用到的运行时多态买单":绝大多数分配就是 host RAM 走 mimalloc,编译期已知。

**方案:两层,std `vector` vs `pmr::vector` 同款分层**

- **默认 = 无状态 `SystemAllocator`(编译期绑定,可内联)**。容器模板 `Container<T, Alloc = SystemAllocator>`。`SystemAllocator::allocate` 内联**直接 `call mi_malloc`**,无函数指针、无双跳、无派发分支 → 与裸 mimalloc 同(merged/static build 下 8% 地板归零)。
- **`Allocator` 运行时 vtable = opt-in**,给真要多态的少数场景(per-request arena、mock 测试、跨 DLL 注入)。显式声明 allocator 类型才付间接。

**两个技术点(解决"内联 vs 不暴露 mimalloc.h"的假两难,用户 2026-06 点破)**:

1. **前向声明替代暴露头**:不需要 `#include <mimalloc.h>`(数百行声明)。只需在 `system_inline.h` 里**前向声明用到的 5 个 C 函数**——`extern "C" void* mi_malloc(size_t) noexcept;` 等(`mi_decl_*`/`mi_attr_*` 是优化提示,不影响 ABI;签名只用 `void*`/`size_t`,无 mi_* 类型泄漏)。`SystemAllocator::allocate` 即可内联直调,**mimalloc.h 不进任何头**。原以为"内联=必暴露头"是假绑定。
2. **模板元编程做编译期派发**:容器默认模板参 `SystemAllocator`(无状态、空基类优化、`allocate` 全 inline)→ 单态化后调用点直接是 `mi_malloc`。要注入时换类型参为运行时 `Allocator`,同一份容器代码两条路径,编译期选定零运行时开销。

**DLL 边界的诚实权衡(不回避)**:
- **merged / 静态链接 / 单 binary**:`mi_malloc` 符号在场 → `SystemAllocator::allocate` **真内联**,8% 全消。**SOTA-parity 的判据就锚在这个 build**(也是部署默认)。
- **多 DLL(consumer 在别的 .so/.dll)**:跨模块调用 `mi_malloc` 经 IAT/PLT,**物理上必有一次间接**(任何跨 DLL 函数调用都如此,非本设计缺陷)。但仍比现状好:**去掉了函数指针层 + 双跳**,退化为一次直接 import 调用。且每个 DLL 各自静态链 mimalloc 会**多堆**(分配/释放跨堆=损坏),故跨 DLL 用单一共享 mimalloc 或显式 `Allocator` 注入——这是 opt-in 路径的正当场景。
- 结论:**默认部署(merged/static)达 SOTA;跨 DLL 是次要场景,接受一次 import 间接**,且调用规约写明"默认零开销,跨边界/异构注入需显式声明 allocator 类型"——用户已确认"调用方可严格、文档写清即可"。


## 新增能力（clay 没有的"极致")

| 能力 | 做法 | 为什么 |
|---|---|---|
| **NUMA 感知** | ✅ mem-06c：Preferred/Bind/Interleave + `HugeRegion::{local,interleaved}` + `reserve_huge_1g_pages` + 大 MemoryArena 自动 HugeRegion；无 libnuma | 多路远程 2-3×；标准库能力面完整 |
| **大页/THP** | ✅ mem-06b：`platform::vm_advise_hugepage` + `mem::HugeRegion`（2MiB 对齐+THP hint+可选 prefault）+ `enable_allocator_hugepages` 封 mimalloc；特权 1GB/hugetlb 后置 | 大张量/大 arena 减 TLB miss（访存 walk \(\sim N/\text{page}\)） |
| **pinned（page-locked）** | `cudaHostAlloc` / `cudaHostRegister` | host↔device DMA 快得多；不锁页 GPU 拷贝走可分页内存慢 |
| **cache-line 对齐 + 防伪共享** | `alignas(64)`，per-core 结构按 cache-line padding | 并发结构（run-queue、计数器）伪共享杀性能 |
| **64B 对齐契约** | `allocate` 真正传递 alignment 到 backing | simd/ AVX512 load/store 要求；修 clay 的 alignment 丢弃 bug |
| **跨核 free + 同线程 TLS 快路** | free 默认进 **本线程 TLS freelist**（owner-only 零原子）；溢出/`publish_thread_cache`/线程退出/`~MemoryArena` 按地址 **flush 回属主片 CAS**。**进程级全局锁**（`detail::tls_registry_lock`，`memory_arena.cc` 单一定义，跨 DLL 唯一）管 attach/detach/reset/dtor；节点 `arena` 字段 `atomic`（热路 acquire-load 匹配，冷路 release-store 置空）；节点由 **`system_allocator()`** 分配（与注入 backing 解耦，backing 先死不 UAF）；TLS 链表回收 `arena==nullptr` 僵尸节点 | 同线程 churn 赢 tbb；**软容量**（他线程 TLS 内 free 不可见）是有意 trade-off |

> **TLS freelist（2026-07-20 落地 + 红队）**：always-CAS 对 tbb 三平台 sign 输 0.72–0.91x 的根因是同线程仍付 `lock` CAS。现实现与红队结论：
> 1. **热路径**：TLS push/pop **零原子**（owner-only）→ Win 对 tbb **~1.7–2.0x sign 赢**。
> 2. **冷路径**：TLS 空 → 分片 Treiber；满 → flush 半量；`publish_thread_cache()` 显式回流；exit/dtor flush。
> 3. **生命周期（2026-07-21 审计加固，ASan+TSan 坐实）**：节点由进程稳定的 `system_allocator()` 分配/释放（**不**走 arena 注入 backing——节点可比 arena/backing 长寿，尤其 C ABI wrapper 析构即释放 backing，否则线程退出 flush 时 UAF）；attach/detach/reset/dtor 全在**进程级全局锁**下（原 per-arena `atomic_flag` 随 arena 先死会让线程退出方锁/读已释放对象——dtor vs 线程退出 data race，已改）；节点 `arena` 字段 `atomic`，冷路 release-store 置空、热路+僵尸回收 acquire-load 匹配；dtor detach + **僵尸回收**（长寿命线程 churn 多 arena 不泄漏 ThreadCache）。回归测试 `InjectedBackingFreedBeforeWorkerExitNoUaf`（注入 backing 先死 + 线程后退出）。
> 4. **软容量（R1，保留）**：他线程 TLS 内 free 对 `allocate` 不可见，窗口 ≤32×活线程。`publish_thread_cache` / join / overflow 恢复硬容量。红队试过 **steal + 热路径锁** → 硬容量回来但 vs tbb 退成 **TIE**（~0.98x），拒绝。与 tbb/jemalloc tcache 同类。
> 5. **组成选型**：`GrowablePool` = ST/per-thread 无界；`MemoryArena` = MT 共享有界 + TLS（接受软窗口或显式 publish）；通用 → `SystemAllocator`。

## 与其他模块的耦合

- **→ sched/**：`fiber_stack.h` 供 fiber 栈（定长 + 尾部 guard page 捕溢出 + 池化复用，避免频繁 mmap）。sched 第一节的栈管理就落在这。
- **→ compute/**:`pinned.h` 供 GPU DMA 的 page-locked host 内存(compute 在 buffer 分配点选用)。device 内存是 compute/ 自己的 stream-bound buffer,**不走 mem/ 的 allocator 抽象**(host↔device 边界显式 copy,见上「device 内存」节)。
- **→ container/**:容器**默认编译期 `SystemAllocator`**(零开销);需 arena/mock 时显式 opt-in 注入 `Allocator`。flat hashmap/carray 的分配走 mem/。
- **→ simd/**:**无依赖边(澄清)**。曾设想 64B 对齐契约(mem 保证对齐 → simd 做对齐 load/store),但 simd kernel 实现走 `LoadU/StoreU` 非对齐 load,**不 include 任何 mem 头**——simd 是零 HYLE 依赖根,与 mem 无耦合。alignment 丢弃 bug 的修复对 mem 自身的对齐保证仍有效(arena/pool 返回对齐块),只是 simd 不通过这条契约消费它。
- **std::pmr 互操作**：提供薄 `pmr::memory_resource` adapter，让 std 容器也能用 HYLE 分配器（低成本、高互操作价值）。

## 落地清单

1. `Allocator` 注入接缝改造:拆单函数→3 指针 `{alloc,free,realloc}`(消 Win64 压栈+派发分支)+ **修 alignment 丢弃** + 默认编译期 `SystemAllocator`(内联,注入降 opt-in)。
2. system allocator 换 mimalloc backing（封在 vtable 后，不外露 mimalloc 类型——同 ggml 解耦）。
3. arena 继承 + 修 64 槽上限；pool/stack 推倒重写修 UB。
4. fiber 栈（给 sched）✅ mem-08：guard + 分级 + 池；pinned 归 compute/。
5. 大页/THP ✅ mem-06b；NUMA ✅ mem-06c；cache-line 对齐已部分落地。**mem-06 主轴收口**。
6. stats 修下溢 + 泄漏追踪 ✅ mem-06a（`StatsAllocator`/`LeakTracker` opt-in；system 热路径无税）。
7. 单测：对齐保证、并发分配 TSan、泄漏检测、NUMA 就近性验证、benchmark 对标裸 mimalloc（专用分配器应更快）；fiber_stack 布局/池/Fiber 接线；stats 下溢回归。

## 决策（已定）

1. **通用 backing = mimalloc**（现代、THP 内建、MIT、易嵌入、v3 2026）。jemalloc 多线程老牌但偏大；tcmalloc 强但 Google 生态绑定。
2. **mimalloc 封在 `Allocator` vtable 后、不外露 mimalloc 类型**（同 compute 的 ggml_bridge 解耦），换 backing 只动 system.cc。默认路径走编译期 `SystemAllocator`(可内联,零间接);`Allocator` 运行时 vtable 是 opt-in(per-request arena / mock / 跨 DLL)。
3. **device 内存归 compute/ 的 stream-bound buffer,不是 mem/ 的 space**(见上方「device 内存」节)。host↔device 边界是显式 copy/命令提交;mem/ 只管 host RAM。NUMA 是 OS 分配层就近 hint,非 space。

## 参考

- [mimalloc：现代高性能可嵌入分配器（v3 2026）](https://github.com/microsoft/mimalloc)
- [mimalloc 性能基准](https://microsoft.github.io/mimalloc/bench.html)
- [jemalloc vs tcmalloc vs malloc：P99 延迟（2026）](https://www.kunalganglani.com/blog/jemalloc-vs-malloc-tcmalloc-p99-latency)
