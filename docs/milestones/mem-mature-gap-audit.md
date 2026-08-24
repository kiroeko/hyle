# mem 成熟库能力缺口复审（2026-07-26）

> **历史证据提示（2026-08-25）：**本页的绿色结论绑定文中日期/commit，不是 mimalloc 3.5.0
> 升级后 working tree 的当前证书。现行状态只看 [`pre-release-readiness`](./pre-release-readiness.md)。

| 字段 | 结论 |
|---|---|
| 范围 | `include/hyle/mem`、`include/hyle/c/mem.h`、`src/mem`、测试、guide / architecture / performance |
| 基线 | `c1f2e394`（独立 worktree：`audit/mem-mature-gap`） |
| 方法 | 从真实基础加速库用户的 lifetime、ownership、并发、失败和组合需求出发，对照工作区 oneTBB、Folly、marl、mimalloc；不按 benchmark 名称凑类型 |
| 总结 | ✅ **彻底收口。**能力面宽且边界合理；P0 契约、standard adapter 默认静态/注入分型、fiber stack 冷回收与 shared distribution 三平台证书均已落地。对照 oneTBB 实现后推翻旧 non-goal：selected-node/chunk 与 weighted placement 已由 platform 机制层和 mem HugeRegion 的 C++/C owner 面完整提供，Linux 极细布局也已关闭 VMA 资源复杂度风险。2026-07-29 严格回审又关闭 PMR 零字节、runtime adapter 隐式默认和 installed mem consumer 三项真实产品合同，当前实现绑定 `8c85331`。 |

> **实现进展（2026-07-26，`implementation/mem-strict`）：**P0 alignment/stats 已在
> `2cb1a10` 落地；`ac73fc7` 增加 `StdAllocator<T>`、FiberStackPool 精确 capacity、
> `trim/clear` 与 Scheduler shutdown owner-level 回收。Windows shared/static correctness 各 6/6，
> Windows ASan 6/6；原生 `HYLE_SANITIZER=asan` 尚有 gtest flag 未传播的构建系统缺口，本次用同构
> 全局 sanitizer flag 完成验证。WSL normal mem 4/4 + sched 2/2；GCC 14 TSan 在 `setarch -R`
> 下通过 mem 非 death 187 项与 sched 142 项。`80afa42` 进一步统一显式 deallocate alignment 并在
> Windows mem 4/4 复验；随后由下面的最终证书在当前产品 commit 上完整接住。

> **最终 standard-adapter follow-up（`ab92b49`）：**shared same-link 审核证明旧默认
> `StdAllocator(system)` / `PmrResource(system)` 为普通 system heap 也保留运行时 state 和额外
> dispatch。现补 `SystemStdAllocator<T>`（empty、always-equal、静态绑定）与进程寿命
> `system_memory_resource()`；原 `StdAllocator<T>` / `PmrResource` 专注真实运行时 policy 注入。
> 该提交的 Windows 先行表现已被下面 clean `c640072` 的三平台最终证书取代。

> **最终证书（2026-07-26，`c640072`）：**section L 扩展到 ordinary alloc/free、默认/运行时
> standard adapter 与 direct/runtime realloc；每 variant 独立新进程、正反交替，Win/WSL/M4
> interactive+utility 均 n=15、shared shipping、dirty=0。证书发现并修复 DSO 内部再次调用
> exported backend seam 的 ELF PLT 真成本；库内 runtime/PMR 现走 private helper，外部
> `SystemAllocator` 仍走公开 seam。默认 heap、SystemStd、system PMR 与 realloc 三平台无
> material LOSE；Windows runtime `PmrResource` 的双动态层约 8.5% 是显式 policy 成本，不冒充
> 默认形状。Windows mem 4/4；WSL mem 非 death 191/191 + sched 142/142、纯 C、107/107 surface，
> GCC14 TSan 同组零报告；M4 完整 6/6 + 107/107 surface。mem 恢复严格证书并退出主动重构队列。

> **严格回审补证（2026-07-29，`8c85331`）：**PMR 零字节请求不再因底层 size-0 NULL 合同
> 伪报 `bad_alloc`；runtime `StdAllocator<T>` 删除会静默选择 system backing 的默认构造，默认堆
> 继续由 empty `SystemStdAllocator<T>` 承担。新增 installed `mem` 纯 C/C++ 默认+高级组合样例，
> Win/WSL/M4 shared/static 十二组 C/C++ consumer 全绿；三平台 mem 正确性、WSL ASan/UBSan 与受影响 PMR
> n=15 证书全绿。没有新增 allocator 类型、常态 metadata 或 benchmark-only API。

> **NUMA 最终复杂度复审（本轮候选）：**实际 node membership 改为缓存位图，查询 O(1)、输入
> validation O(S)；重复 node 不再误判非法，而是与 oneTBB 一致表达 weighted placement。Linux
> 单 node / 递增逐页普通形状保持一次惰性 `mbind`；自定义条带在 K≤1024 时保留 O(K) 惰性
> policy，K>1024 时改为 O(P) 固定批次 first-touch/move、O(P/256) syscall、常量 scratch 和
> 有界 policy，消除 page-sized arbitrary order 对 `vm.max_map_count` 的架构风险。Windows/WSL
> 普通构建与纯 C 全绿，
> WSL UBSan NUMA/VM/大页全绿；C surface 保持 platform 120/120、mem 109/109，无 ABI 增长。

本文是 2026-07-25 `mem-close-plan.md` 之后的红队补充，不复制稳定设计和点值。稳定能力仍以
[`mem-design.md`](../architecture/mem-design.md) 为真源，用户选择以
[`mem.md`](../guide/mem.md) / [`c-mem.md`](../guide/c-mem.md) 为真源，性能只以
[`benchmarks-mem.md`](../performance/benchmarks-mem.md) 为真源。

## 1. 当前产品面

| 用户问题 | C++ 形状 | C 形状 | 现状 |
|---|---|---|---|
| 普通 host heap、注入自定义 backing | `SystemAllocator`、`Allocator` | 三函数指针 + state、system handle | 完整；默认编译期路径与运行时注入分开 |
| PMR / allocator-template 容器、显式首地址 alignment | 默认 `system_memory_resource()` / `SystemStdAllocator<T>`；注入 `PmrResource` / `StdAllocator<T>`；`AlignedPmrResource` | 不适用 | 默认静态形状与 runtime policy 分开，避免普通容器无理由叠加 dispatch/state |
| 限额、计数、泄漏 inventory | `QuotaAllocator`、`StatsAllocator`、`LeakTracker` | opaque decorator + POD/copy-out | 完整；并发 peak、snapshot 与销毁合同已修复并测试 |
| request/phase bulk lifetime | `Region`、`FixedRegion`、`PooledRegion` | 对等 owned/value/borrowed 形状 | 完整；epoch 复杂度已收口 |
| external variable-size bounded heap | `FixedPool` | caller-owned/in-place 或 created control | 完整；TLSF 风格 FL/SL，查找/合并 O(1) |
| hard fixed-block capacity | `SharedPool` | opaque handle | 完整；completed free 全局可见 |
| owner-local fixed-block cache | `SharedPoolCache` | 显式 cache handle | 完整；soft capacity、无隐藏 TLS |
| typed object lifetime / preserving return / loan | `ObjectPool<T>`、cache、`PreservingObjectPool<T>`、`PoolLoan` | raw slot + typed macros | C++ 自然；C 不伪造构造析构 |
| 单线程按需增长的固定 slot | `GrowablePool<T>` | runtime slot core | 完整 |
| LIFO scratch | `StackAllocator` | opaque handle | 完整 |
| 大页、NUMA host RAM | `HugeRegion` + placement policy / selected nodes + chunk | opaque handle + 同能力 | all-node/local/preferred/selected-node 能力完整；runtime capability 与结构化失败避免伪可移植 |
| guard-page fiber stack/cache | `FiberStack`、`FiberStackPool` | 有意不暴露 raw C mem API | 已补真实 capacity/trim/clear；Scheduler shutdown 自动回收 |

生产消费也支持这些形状并非只为 benchmark 存在：allocator 被 container、graph、sched、I/O、
compute bridge 广泛使用；`Region` 被 graph/HTTP request 使用；`FiberStackPool` 被 scheduler 使用。
其余池/诊断类虽暂时主要是公开能力与测试消费者，但 ownership 合同清楚，不应只按“仓内未消费”删除。

## 2. 成熟库对照后的缺口矩阵

### P0：冻结前必须修

实现状态：**3/3 已完成**。alignment 验证位于真正消费请求的 C++ 显式 overload / C convenience，
默认 overload 无热路分支；Stats peak 使用原子 add 的返回值；snapshot 选择 per-field observation +
quiescent coherence；destroy/reset/move overwrite 执行 balanced fail-fast。

| 问题 | 证据 | 影响 | 验收 |
|---|---|---|---|
| `SystemAllocator` 接受低于默认值的 0/非 2 次幂 alignment | `system_inline.h:39-58,77+` 对 `alignment <= default` 直接进 plain mimalloc；shared seam 的 `system.cc:21,35-49` 同样只检查 over-aligned；其他 region/pool/stack 均用 `is_pow2`，guide 又声明 invalid alignment fail-fast | 同一个 mem 模块有两套调用合同；错误请求静默成功，之后可能把 bug 带进任意 allocator-aware 类型 | native static/shared 与 runtime `Allocator` 的显式 allocate/deallocate/reallocate 统一 fail-fast；0 只在 C convenience 层保留“默认 alignment”语义；加 0/3/6 death test，并重跑 system hot-path 同合同 benchmark |
| `AllocStats::peak_live_bytes` 并发下可能漏掉真实高水位 | `stats.h:43-49,67-88,122-128` 先 `live_bytes_.fetch_add`，随后重新 load；期间另一线程可 free。`QuotaAllocator::commit_growth` 已展示用 fetch-add 返回值更新 peak 的正确形状 | 文档称计数精确，但高水位可能偏低；这是诊断正确性，不是 benchmark 细节 | 用本次 fetch-add 的 `next` 做 CAS peak；覆盖 allocate 与 grow-realloc；增加能制造 add/free 交错的并发测试 |
| Stats snapshot / lifetime 合同过强或未执行 | C 头 `mem.h:248` 称 “point-in-time POD snapshot”，实现 `stats.h:94-103` 是多个 relaxed load，跨字段不形成一致时点；`StatsAllocator::destroy()`（`stats.cc:238-245`）不检查 `balanced()`，而 `QuotaAllocator` 析构与 C guide 均要求无 live allocation | 用户可能把互相不一致的并发字段当作一个原子快照；带 live allocation 销毁后借出的 allocator/state 悬空而未 fail-fast | 明确选择：低成本 per-field snapshot（推荐，quiescence 后精确）并修文档，或付出序列锁成本提供 coherent snapshot；销毁/移动覆盖前按项目合同 fail-fast，补 death test |

这些修复不要求新增常态 per-allocation metadata。peak 修复甚至减少一次 `live_bytes_` load；alignment
验证应由分支可预测、可内联的 `is_pow2` 门完成，但仍须用 static/shared 同链接模型重新签热路径。

### P1：真实常用能力，冻结前应裁决

实现状态：`SystemStdAllocator<T>` / `StdAllocator<T>`、`system_memory_resource()` / `PmrResource`
的默认/注入分型与 FiberStackPool 回收 **已完成**；selected-node/chunk NUMA 随后按
platform-first 原则完成，mem 只承载 HugeRegion owner，不复制系统实现。

| 候选 | 成熟形状 | HYLE 建议 | C ABI |
|---|---|---|---|
| 传统 C++ Allocator named-requirement adapter | oneTBB `memory_pool_allocator<T>` / `cache_aligned_allocator<T>`；Folly `CxxAllocatorAdaptor<T,Inner>`；marl `StlAllocator<T>` | 默认 system heap 用 empty/always-equal 的 `SystemStdAllocator<T>`；运行时 backing 用 `StdAllocator<T>`，含 rebind/完整 backing equality/非传播 traits；覆盖 `vector`、`allocate_shared`、over-alignment 与失败路径 | 无；模板天然只属于 C++ |
| `FiberStackPool` 主动冷回收 | 原实现默认每 tier 缓存 64、内部固定 256 slots，只有析构释放；传入 cap>256 的有效上限不可见 | 已增加准确 `capacity()`、0-disable、cap clamp、`trim(keep=0)` / `clear()`；锁内摘取、锁外释放 VM，scheduler 在 worker join 后回收三档 cache | mem C 无需导出；scheduler C 用户由 runtime 自动管理 |
| 选择 NUMA nodes + interleave chunk | oneTBB `numa_allocation.h:40-48` 接收 node vector 与 `bytes_per_chunk` | **已实现**：platform 暴露实际稀疏 id、runtime capability 与连续 committed allocation；Linux first-touch 前 mbind，Windows 单 reserve 分条带 NUMA commit，macOS 单节点正常 no-op；mem 仅包装为 HugeRegion | platform raw/managed + mem HugeRegion 同步导出；C pointer/count/Status 合同与 C++ 等价 |

传统 allocator adapter 是本轮最明确的功能缺口：它在 oneTBB、Folly、marl、mimalloc C++ facade
中反复出现，解决的是常见第三方/标准容器消费，不是为 benchmark 增加一种 allocator。实现时优先让
`std::allocate_shared` 复用标准机制，不另造 HYLE shared_ptr 控制块 API。`SystemAllocator` 的 typed
construct/destroy/unique-array 便利性可以随 adapter 一并审，但不应因此再长出一族公共 allocator。

### P2：先组合、测真实消费者，再决定是否上架

1. **HYLE-owned growable variable-size pool。** 这里不能再写成笼统的 “variable-size pool” 缺口：
   `FixedPool` 已在 external fixed-capacity backing 上完整提供 variable-size individual
   allocate/free/reallocate + bulk `reset()`，而 `SystemAllocator` 接受任意大小、`Region` 提供 owned
   growable + bulk lifetime。尚未上架的仅是 oneTBB `memory_pool` 那种“产品自己取得/增长 backing，
   同时 individual free 与 bulk release”的 HYLE-native 类型。C++ 默认已经能把
   `system_memory_resource()` 交给 `std::pmr::unsynchronized_pool_resource` / synchronized 版本；
   运行时 quota/stats/custom backing 才使用 `PmrResource`。只有出现 PMR/mimalloc 无法满足的真实
   C/FFI 或专用 ownership 消费者，才设计新 opaque heap，不能把现有 FixedPool 能力误报成缺失。
2. **preserving object 的 owner-local cache。** 当前 preserving pool 每次走 hard core CAS。对昂贵
   codec/context 对象可能值得，但对象工作通常淹没 CAS；先在真实 consumer 测量，不凭微基准加类型。
3. **Arena merge / hidden TLS arena。** Folly `Arena::merge` 主要服务
   `ThreadCachedArena` 的 thread-exit zombie 回收。HYLE runtime 已明确拥有 worker 生命周期；在没有
   runtime-owner 协议前不应把隐藏 TLS registry 塞回基础 mem。

## 3. 明确非目标

- 不自写任意大小通用 malloc 与 mimalloc/标准 PMR 重复竞争；也不把 `mi_heap_t`、vendor options
  原样泄漏成 HYLE ABI。该轮审计所面对的 mimalloc v2 first-class heap 带
  allocation-thread 约束；当前 v3 backing 的具体合同与重认证状态以 `mem-design`、
  `dependency-upgrade-2026-08` 和 `pre-release-readiness` 为准。无论 vendor 版本，错误包装
  first-class heap 的风险都比少一个 HYLE 类型更大。
- 不在 mem 实现 blocking borrow；等待、fiber semaphore 与 cancellation 属 sched wrapper。
- 不把 GPU/device/pinned/staging 抽象成泛化 `MemorySpace`；native handle、mapped pointer、event 与
  completion lifetime 属 compute。
- 不收 graph liveness allocator 或 io_uring provided-buffer ring；mem 提供底层 region/backing，领域
  owner 决定注册、复用和完成语义。
- 不因 Folly 提供 no-dump/guard allocator 就直接复制；等 security/VM consumer 给出跨平台合同。

## 4. API、依赖和文档结论

**C++ API。** ownership 类型总体清楚，默认静态 `SystemAllocator` 与显式动态 `Allocator` 分层正确；
传统 allocator-template API 由 `SystemStdAllocator<T>` / `StdAllocator<T>` 补齐，PMR 由
`system_memory_resource()` / `PmrResource` 分开承载默认和注入形状；typed helper 保持在现有
对称集合中，不再散落 convenience。

**C ABI。** 109 项当前已覆盖 allocator、诊断、region、variable/fixed/growable pool、stack 与
HugeRegion。PMR、RAII loan、模板构造析构和 raw fiber stack 是合理的 C++ only。上述 P0 不增加符号；
allocator adapter/stack trim 也不要求 mem C ABI。选-node NUMA 已同步扩展 platform C 与
HugeRegion options/chunk query。HYLE C++ 内部继续使用原生 `Status` / `Result` / `Allocator`，不绕 C ABI。

**error / platform / util-core。** VM/NUMA/fiber 冷路径使用共享 error；platform 负责
page/VM/NUMA，归属正确。中央 component graph 与真实 target 边均为
`mem -> error + platform + util_core`，其中 fiber stack pool 消费 SpinLock。`core/cache.h` 是
零链接的 cache-layout header seam，`obs/trace.h` 是 `HYLE_TRACE` 跨切 seam；两者不形成普通
mem runtime component 边。HYLE 内部使用 C++ API，只有 `c_mem.cc` 翻译 C 边界。

**guide / performance。** 本轮已同步：

- C/C++ guide 解释 snapshot 是并发时 per-field 观察、quiescence 后才可作一致结论；记录 Stats
  destroy 与 reset 的 quiescence 合同。
- C++ guide 已增加标准 PMR pool-resource 组合，并给默认静态/运行时注入的 `std::vector`、
  `allocate_shared` 最短 recipe；C guide 明确 FixedPool 已是 external bounded variable-size 池。
- guide 说明 FiberStackPool 的 retention 上限、实际 capacity 与 trim 控制点。
- performance 已把 static/source 与 shared SDK 分表；clean `c640072` 的四张 raw 让同一 DSO
  backend seam 对拍默认/运行时 allocator、standard adapter 与 realloc。默认形状三平台无
  material LOSE，显式双动态层成本单列，不再遗留“待补录”或拿 exe 内第二份 mimalloc 冒充对手。

## 5. 推荐收口顺序与退出条件

1. [x] 修 3 个 P0，补静态/shared/runtime 与并发合同测试，更新 C/C++ guide。
2. [x] 实现传统 allocator adapter；补 std 容器、`allocate_shared`、over-alignment、equality/lifetime、
   OOM 测试。只对比同合同 peer，不为它另造孤立 benchmark。
3. [x] 给 FiberStackPool 补 capacity + trim，接 scheduler 的真实 shutdown reclaim 控制点；VM release
   位于锁外，shutdown 回收在 worker join 后执行。
4. [x] selected-node/chunk NUMA 定向复开并完成：OS 差异由 runtime capability + Status 表达，
   platform 拥有机制，mem/C ABI 提供对应 owner 形状；weighted placement 与 Linux 有界 VMA
   策略完成，没有把 unsupported 冒充 applied。
5. [x] 完成 shared-distribution 三平台同链接模型证书，并复跑 correctness/sanitizer/surface gates。

退出条件不是“类型数量追平成熟库”，而是：P0 为零；P1 每项有实现或有证据的拒绝；C++ 常见消费无需
绕路；C ABI 没有可表达能力的缺口；默认与常用组合在同合同、同链接模型、三平台下无 material LOSE；
architecture / guide / performance / milestone 不互相冒充真源。达到这些条件后，mem 才能再次标记
**彻底收口**。原 5 项与退出条件已在 clean `c640072` 满足；NUMA 复开又补齐 selected
nodes/chunk、weighted placement、有界 kernel resource complexity、C++/C guide 与 platform
同合同性能证据，未改变其它 allocator/pool 证书；后续只按
[`mem-design.md`](../architecture/mem-design.md) 的明确触发器重开。

## 6. 对照源码定位

- oneTBB：`include/oneapi/tbb/memory_pool.h:80,177,190`；
  `cache_aligned_allocator.h:42,111`；`numa_allocation.h:40,48`。
- Folly：`folly/memory/Allocator.h:188`；`Arena.h:62,122`；`ThreadCachedArena.h:40`。
- marl：`include/marl/memory.h:138-147,325+`。
- mimalloc：`include/mimalloc.h:213-237`（first-class heap/new/destroy/collect/aligned-at）。

这些是设计证据而不是待复制清单；最终 HYLE 类型仍须服从自己的 owner、failure、threading 与 ABI
合同。

## 7. 2026-07-29 严格回审交付

本轮按 [`module-closure-methodology.md`](../architecture/module-closure-methodology.md) 重建审查单元，
覆盖公开 C++/C 头、实现、target、成熟库、直接上下游、安装消费、四层文档和受影响性能。结论：

- 功能面继续覆盖 default/runtime heap、标准 adapter、quota/stats/leaks、四类 region/stack、
  external variable-size TLSF pool、hard/shared/cache/object/growable pool、fiber stack、HugeRegion
  selected-node/chunk/weighted NUMA。HYLE-owned 自动增长 variable heap 仍由标准 PMR pool 与
  mimalloc 更自然地承担，不是遗漏；GPU/pinned/registered buffer 仍归其 completion owner。
- oneTBB `memory_pool`/NUMA/cache-aligned resource、Folly Arena/allocator adaptor、marl pool/loan 与
  mimalloc 再对账，没有发现 HYLE 用户场景中尚未解释的能力或算法落档。FixedPool FL/SL、
  SharedPool 分片/tag 饱和、Region retained blocks、NUMA placement 的复杂度证据继续有效。
- 修复两个实际 API 缺陷：PMR size 0 使用 1B backing 成对释放；`StdAllocator<T>` 必须显式给
  runtime backing，普通 system 容器选 `SystemStdAllocator<T>`。C ABI 功能范围不变，109/109
  surface 保持；C 不机械镜像模板/PMR/RAII。
- 依赖回审确认 mem 正确使用 error/platform/util-core，未直调 OS 或绕 C ABI；component closure
  `mem -> error + platform + util_core` 在 Windows、Linux、macOS 的 focused configure 与
  shared/static installed consumer 中都成立。新增 `samples/package/mem/{c,cpp}` 让默认与高级
  组合成为外部可执行合同。
- 正确性：Windows mem 4/4（含纯 C、109-symbol surface、独立头、221 项 C++）；WSL GCC 14
  196 个受影响非 death 测试、ASan 197、UBSan 196；M4 AppleClang 21 完整 4/4。历史未改 death/
  TSan/NUMA/pool 证书保留，不用无关全库重跑冒充新增证据。
- 性能：三平台 shared shipping section L n=15 局部复证正常非零 PMR；runtime/direct 比为
  Win 0.934、WSL 0.988、M4 interactive 1.031、utility 1.035，默认 PMR 四组均 WIN。零字节
  正确性修复不增加对象状态且没有正常用法退化，详见 performance 真源。

四项 LOC（实现 `8c85331` + 本证书文档工作树，统一 cloc 口径）：

| 口径 | code 行 |
|---|---:|
| 全库 tracked（含 docs / in-tree third-party） | 279,585 |
| HYLE 自实现含测试（`include/hyle + src + tests`） | 135,519 |
| HYLE 自实现不含测试（`include/hyle + src`） | 92,914 |
| mem 当前实现（`include/hyle/mem + include/hyle/c/mem.h + src/mem`） | 5,414 |

停止条件成立：没有未解释的功能、复杂度、C/C++ 易用性、依赖、安装或文档缺口。此后只按
correctness/security、真实新消费者、平台/工具链失效或同合同可复现 material LOSE 局部重开。
