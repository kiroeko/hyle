# HYLE mem/ 设计

> Tier 0 host-RAM 基础设施。本页是 mem 当前边界、公共能力形状和长期合同的设计真源。
> 用户写法见 [`../guide/mem.md`](../guide/mem.md) 与 [`../guide/c-mem.md`](../guide/c-mem.md)，
> 实测只见 [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)，当前收口任务只见
> [`../milestones/mem-close-plan.md`](../milestones/mem-close-plan.md)。

## 1. 边界

`mem/` 只拥有普通 CPU-addressable host RAM 的分配与生命周期原语：通用 allocator、region/stack、固定块池、对象池、fiber stack、NUMA/大页和诊断适配。

以下能力由使用其完成语义的模块拥有：

- device VRAM、unified/mapped/ReBAR memory 归 `compute/`；可见性、coherence、flush/invalidate、stream/event 生命周期不能伪装成 `mem::Allocator`。
- pinned host buffer 的所有权与 DMA completion 也归 `compute/`。真正 staging slot 必须同时持有 backend handle、CPU mapped pointer、Device/ordinal、Event 和状态。
- io_uring provided-buffer ring 归 `io/`；其注册、buffer id、worker issuer 和 completion 生命周期不是通用 mem 合同。
- graph 的 liveness/worst-case reserve allocator 归 `graph/`；通用 mem 只提供它需要的底层 region/backing。

mem 不提供 pinned allocator 或单块 staging convenience。compute `Buffer` 分开保存 opaque `native_handle()` 与可选 `host_data()`；CUDA/host 两值可相同，Vulkan 分别是 `VkMem*` 与 `VkMem::mapped`。CPU 直接访问用显式 `flush_host()` / `invalidate_host()` 表达非 coherent Vulkan 合同；同步 `copy_from()` 自动包住所需 visibility。未来 staging ring 只能组合 HostBuffer、Event 与 backpressure，不上架裸 host allocator 或 `SharedPool(cudaMallocHost)` 组合。

## 2. 需求基线

### 2.1 HYLE 内部真实消费者

| 能力 | 当前生产消费者 | 真实合同 |
|---|---|---|
| `Allocator` / `system_allocator()` | sched 队列与 driver、HTTP/2 session/stream、IO pending 表、StringBuf、graph 临时容器、ggml bridge job/frame | 按值注入、alignment、sized-free、backing 生命周期；allocator 本身以 NULL 报资源失败，消费者当前混用 Status/失败 handle/fail-fast（graph parallel 临时数组仍 fail-fast），不能伪称已有统一 OOM 策略 |
| `SystemAllocator` 类型 | HYLE containers 的默认模板参数 | 外部默认容器零 vtable；内部关键路径多显式用 `Allocator` |
| `FiberStackPool` | sched worker/fiber | guarded stack、三档尺寸、跨 worker release、Scheduler 覆盖全部 fiber 生命周期 |
| `HugeRegion` | 显式大块内存调用方 | THP/NUMA 是显式策略，不由 pool 根据 allocator identity 偷换 backing |
| graph evaluation 临时区 | `parallel_eval` 的 level/handle/outcome 与 wavefront CSR/state | 明确的 phase lifetime，是通用 owned Region 的内部 workload；graph-aware liveness/reserve 仍归 graph |
| `PooledRegion` / `StackAllocator` / `GrowablePool` | 无生产调用 | 面向外部 infra 用户的公开能力；已有 C++/C 产品面、测试与 benchmark |
| `SharedPool` / `SharedPoolCache` | 无生产实例 | hard core + 显式 owner-local cache；当前仅公开能力、测试与 benchmark |
| `ObjectPool<T>` / `ObjectPoolCache<T>` | 无生产实例 | typed hard core + 显式 cache，已实现并测试 |
| compute `Buffer` / `DevicePool` | 当前无生产创建点 | 不归 mem；Buffer 已显式分开 native handle、host mapping 与 coherence 操作 |

内部系统已经坐实“通用 allocator + fiber stack”，但 HYLE 是公开 infra 库，其他能力是否保留必须再看成熟库的广泛用户形状，不能只按内部调用计数裁剪。

### 2.2 成熟库反复出现的用户形状

来自 oneTBB、mimalloc、Folly、marl、ggml 的公开 API、测试和真实调用：

1. aligned 通用 C allocator；C 返回空，C++ typed/PMR/RAII 适配边界可抛 `bad_alloc`。
2. remote free：分配线程与释放线程不同，不能假设 thread-affine。
3. owned request/frame region：大量小对象，individual free 无效，clear/destroy 整批释放；Folly `Arena` 的 min-block growth、usage 与 hard size limit 是直接参考，HYLE graph evaluation 已提供内部 workload。
4. 固定容量共享池：耗尽必须可观察；hard quota 与 thread-cache soft limit 必须是不同类型/名称。
5. typed pool 的 RAII Loan/return 是稳定易用性需求；Reconstruct/Preserve 在 marl 与 Folly expensive-context pool 中都有实证，但并不自动证明每个 bounded pool 都必须同时支持两种 policy。
6. external fixed buffer：oneTBB `fixed_pool` 证明 embedded/realtime/调用方自有 storage 的需求，但它是可 individual free 的 variable-size pool，不能与 bump Region 混为一谈。
7. strict byte limit 的强证据首先是 `Region(size_limit)`；通用 sized-free `QuotaAllocator` 是进一步外推，不能冒充成熟库共同基线。
8. explicit NUMA/hugepage policy：capability query、失败降级或显式报错，不承诺自动迁移。
9. graph-aware reserve/liveness allocator、GPU staging ring 都有真实需求，但属于拥有图或 completion 生命周期的模块，不进入通用 mem。

### 2.3 参考能力逐项处置

成熟库用于发现需求，不用于机械增加同名类型；差异必须能落到已有合同或明确的模块边界：

| 成熟形状 | HYLE 处置 | 理由 |
|---|---|---|
| oneTBB scalable/cache-aligned allocator/resource | `SystemAllocator` + `PmrResource` / `AlignedPmrResource` | 通用堆借成熟 mimalloc；高对齐是显式 PMR policy，不污染默认容器 |
| oneTBB `memory_pool`（growable variable-size） | 不另造公共通用堆 | 与 mimalloc 的职责重叠；HYLE 只保留有不同 ownership 合同的 external `FixedPool` |
| oneTBB `fixed_pool` | `FixedPool` | caller-owned storage、individual alloc/free/realloc、无 fallback；内部 two-level bitmap 给出 bounded fit |
| Folly `Arena` | `Region` | owned growable bulk lifetime、size limit、usage/reserved、large-block release；clear 后同样单调走 retained blocks |
| Folly `ThreadCachedArena` | 不内建隐藏 TLS registry | 用户可按 worker/thread 持有 `Region`；线程退出合并/全局销毁属于 runtime owner（如 sched），基础 mem 不猜 TLS lifetime |
| mimalloc heap/arena/local+remote free | `SystemAllocator` 与 `SharedPool`/显式 cache | 任意大小 heap 直接借核；严格固定容量和 soft owner cache 用不同公开类型表达 |
| marl bounded pool / policy / loan | `SharedPool`、`ObjectPool<T>`、`PreservingObjectPool<T>`、`PoolLoan` | hard capacity、Reconstruct/Preserve 和 RAII return 分型，不塞进一个 policy benchmark 类型 |

Folly `Arena::merge` 服务其 `ThreadCachedArena` 的 thread-exit zombie 收集；在 HYLE 没有对应
runtime owner/consumer 前不单独上架，否则还需定义不同 backing、size limit、并发移交和 source
失效的完整合同。真实消费者出现后应由拥有线程生命周期的模块先给出需求，而不是因参考库有
一个方法就扩面。

## 3. 公开类型必须表达的五个合同

任何 pool/region 不能只用“更快 allocator”描述；类型、构造参数和方法名必须让用户看见：

| 维度 | 必须显式区分 |
|---|---|
| size model | 固定 block / 固定 object / variable-size |
| storage ownership | owned / external buffer / borrowed block |
| concurrency | single-owner / thread-safe / thread-cached |
| capacity | hard bounded / soft cached / growable |
| lifetime与失败 | individual destroy / preserve-return / bulk reset；try(NULL) / throw adapter / blocking wrapper |

blocking borrow 不进入 `mem/` 基础池。等待策略依赖 fiber/condition/semaphore，应由 `sched/` wrapper 组合；mem 提供 nonblocking `try` 原语。

## 4. 通用 allocator

| 层 | 当前形状 | 产品处置 |
|---|---|---|
| `SystemAllocator` | 编译期无状态、header-inline 直调 mimalloc | 保留；修 typed count 乘法回绕 |
| `Allocator` | C++ 值句柄 `{alloc, free, realloc, state}` | 保留；内部运行时注入脊；C ABI 只在对外 shim 映射 |
| `PmrResource` | `Allocator` → `std::pmr::memory_resource` | 保留；构造 assert valid，equality 比较完整三函数 + state |
| `AlignedPmrResource` | 可选 minimum base alignment 的 PMR adapter | 保留；对应 oneTBB `cache_aligned_resource` 的用户需求，不给普通 PMR 强加 cache policy |
| typed/RAII helpers | `allocate_n/zeroed/reallocate_n`、construct/destroy、make_unique | 保留；乘法溢出和显式 alignment 在 System/运行时两层一致 |

通用 malloc 借 mimalloc，不自造任意大小通用堆。sized-free 必须带回原请求的 size/alignment；backing identity 与 ownership 不能丢失。

### 4.1 C/FFI 产品面

`include/hyle/c/mem.h` 按 C 用户任务独立设计；发布前仍可破坏性改形，但不能用符号数量
代替能力审查。设计参考工作区中固定的 oneTBB、mimalloc、Folly 与 marl 源码快照：

这只是**对外边界**。HYLE 源码与模块间调用全部使用 C++ `Allocator`、`Status`、`Result<T>`、
`VmRegion` 等类型；`src/mem/c_mem.cc` 才把它们翻译成 `hyle_allocator` / `hyle_status`。内部
模块不得为了“复用 C ABI”绕回 C handle。

1. `hyle_allocator` 的 POD vtable 保留；它是 C/FFI 的注入脊，不要求 C++ object layout。
2. 有独立 runtime state 的能力使用 opaque handle；C 不机械镜像模板对象和 C++ policy 类型。
3. 构造函数收到单一 `backing` 时，control handle 与内部 storage 都必须服从该 backing，不能
   绕到 global `new`。明确区分 payload/metadata 的组合产品分别服从两个域：StatsAllocator
   的 backing 是被测数据域，稳定 state、LeakTracker hash 与 C opaque control 服从 metadata，
   后者默认 system。这样 fixed-block/strict-quota backing 不会被任意尺寸诊断对象破坏合同。
   两域的构造 OOM 均经 `Result`/`hyle_status` 返回并原域回滚。FixedRegion 是三字按值 cursor，无 control allocation；FixedPool 的 owning
   factory 显式接收 control allocator，external payload 始终不 fallback。HugeRegion 的 payload
   由 OS VM 管理，小 control 当前使用 HYLE system allocator。
4. C 传入的 size/layout/alignment 是可校验数据：冷构造/初始化返回 `hyle_status + out`，
   非法参数不让 C++ contract assert 穿过 ABI；out 参数进入时清空。热 allocate/pop/acquire
   保留指针/NULL 值合同。foreign/double return、错误 marker 和 live object 下销毁仍是已建立
   的调用合同违反，可 fail-fast。
5. `PooledRegion` 在 C/C++ 使用同一名字与 `clear()`；未发布阶段不保留 `Arena/reset` 兼容壳。
   adapter 统一用 `*_as_allocator`，不混用 `get`。
6. `c_surface_symbols.txt` 只保证声明、纯 C 消费和实际导出同步；有意的产品改形必须更新它，
   它不是 ABI freeze。

已收口的决定：mem 不维护私有 TLS error；C++ 冷路径直接复用 P0 `Status/Result`，只有
对外 C shim 转成 `hyle_status`。HugeRegion 直接传递 platform VM/NUMA 的 C++ Status，成功
零分配，热分配路径不构造 Status。FixedRegion 按值、FixedPool
提供 caller-owned in-place control；StatsAllocator 作为 opt-in 产品能力导出 POD snapshot
与返回 Status 的 bounded leak copy；metadata OOM 不阻断被测 allocation，但 dropped count /
complete bit 使不完整 inventory 可观察。PMR、typed ctor/dtor 与 fiber RAII 没有
自然 C 形状，不为追求符号 1:1 而导出。

## 5. 固定块池

`SharedPool` 独占 `detail::ShardedSlab`；`SharedPoolCache` 通过 narrow batch hooks 组合在 hard core 上，不复制 slab，也不另建 TLS registry。

### 5.1 `SharedPool`：hard bounded

- 固定 block size + runtime capacity。
- completed `deallocate()` 立即进入 shared core；remote free 可见。
- 两次稳定全-shard 快照消除 block 在扫描背后迁移造成的假耗尽。
- home-shard 成功路径只做一次 CAS，保持 expected O(1)；O(shards) observation 数组与稳定空
  快照被隔离到 noinline exhaustion 冷路，不让正常成功调用保留约 2 KiB 栈帧。
- 64-bit packed head；32-bit version 饱和后永久进入该 shard 的 serialized mode，不回绕。
- 用于严格 object/block quota、跨线程 handoff；本地 request churn 不是它的吞吐强项。

### 5.2 `SharedPoolCache`：显式 owner-local soft cache

- 构造时绑定 `SharedPool`；caller-owned `span<uint32_t>` 可做到 cache metadata 零隐藏分配，也可显式注入 metadata allocator。
- capacity 与 transfer batch 是每个 cache 的公开参数；128/32 只是 convenience 默认值。
- owner-local 命中零原子；其他 cache 中的 free block 暂不可见；`publish()`/析构是显式可见性边界。
- cache 本身 single-owner，不维护 TLS node/registry；多个 cache 可并发绑定同一 hard core。
- 只用于池容量高于并发 working set、owner-local 复用为主的场景；不得宣传为 hard capacity。

两层保留。默认 shipping 热路径不增加 per-slot bitmap/atomic live counter；foreign/misaligned pointer 仍做无状态 membership fail-fast。double free、outstanding object 属调用方契约违反，`HYLE_TESTING` 用 per-slot state fail-fast；额外状态和指令在 shipping 构建完全剥离，不破坏 cache 零原子合同。

`SharedPool` 始终服从 injected backing。需要 THP/NUMA 的调用方应显式提供相应 allocator/provider；pool 不根据 allocator identity 偷换 storage domain。

## 6. typed object pool

`ObjectPool<T>` 与附着其上的 `ObjectPoolCache<T>` 已实现：placement-new、显式析构、构造异常退槽、OOM/耗尽返回 NULL；后者暴露 `publish()`。当前策略固定为每次 create/destroy 都 Reconstruct。

产品处置：保留 hard core / soft cache typed shell，并补齐独立的生命周期能力：

1. move-only `PoolLoan<T>` 已实现，作用域退出自动 destroy/return，减少异常/早退漏还；Pool 必须比 Loan 长寿。
2. 当前手工 `create/destroy` 保持 `Reconstruct` 合同：borrow 时构造、return 时析构。pool 析构前必须已归还并析构全部 live object；这是调用方合同，不得暗藏。
3. `PreservingObjectPool<T>` 已作为独立类型实现：池构造时一次性构造全部对象，return 保留状态，池析构时一次性析构；不与 slot cache 混名。
4. 默认 API 仍是 nonblocking try；需要“等到有对象”的调用方在 sched 层用 Semaphore/Channel 包装。
5. 当前 shipping 对 foreign/misaligned pointer 做 membership fail-fast；`HYLE_TESTING` 已补 per-slot liveness、double return/destroy 与析构时 outstanding allocation，shipping 不加原子计数税。

`ObjectPoolCache<T>` 不是 Preserve object cache；Cache 只表示 free slot 的 owner-local 可见性。Preserve 必须由独立 policy/name 表达。

## 7. region、stack 与增长池

### 7.1 `PooledRegion`

`PooledRegion` 只借 `SharedPool` 的一个 hard block，固定容量、单线程、bulk `clear()`。库尚未发布，旧 `Arena/reset` 双命名已删除，Region 家族只保留一套公开说法。

### 7.2 owned `Region` 与 external `FixedRegion`

owned growable `Region` 已提供：

- 按 total backing-byte block target 增长（包含内部 header/alignment，单个大请求可突破）；
- `clear()` 复用普通 block，大型专用 block 可释放；
- 当前 ordinary block 的 cursor/end 是内联热路径，搜索、增长与 backing 分配在慢路径；
- clear 后用单调 retained-block cursor 前进：一个 epoch 内每个普通 block 至多考察一次，
  总遍历 O(blocks)、每次 block transition 摊还 O(1)，不从链头重扫形成 O(blocks²)；
- 热分配只累计 current `used`；completed-epoch peak 在 clear/release 折叠，查询时取
  `max(completed_peak,current_used)`，保持任何时刻精确而不为每笔 allocation 加 peak 依赖链；
- `bytes_used()` / `reserved()` / hard `size_limit`；
- individual deallocate 明确 no-op；
- 单线程默认，不偷偷加锁。

`FixedRegion(span<byte>)` 是独立的 external-storage bump 类型：无隐藏 payload 分配、无增长、无 fallback。HYLE graph evaluation 的逐次 evaluate 临时数组是 phase-lifetime workload；graph-aware `measure/reserve/liveness` 仍归 graph。

### 7.3 `StackAllocator`

保留固定 owned slab + mark/rewind 的 LIFO scratch 合同。对齐计算必须防 uintptr 加法回绕；极端 alignment 是调用方契约边界，不得静默产生错误指针。

### 7.4 `GrowablePool<T>`

保留单线程、同尺寸、按需增长的 Reconstruct pool。release 必须来自同一 pool 的 live slot；默认 shipping 可保持零元数据，diagnostics 模式检查 membership/double release/live underflow。

## 8. strict limit 与 external storage

当前 hard `SharedPool` 解决固定块数量；`Region(size_limit)` 解决 region reserved-byte 上限；`QuotaAllocator` 解决任意 sized-free backing 的 live requested-byte 上限。三者不能互相冒充。

external 能力按 allocation model 分成两个类型：

1. `FixedRegion`：调用方借入 storage、bump + bulk reset、无 individual free。
2. `FixedPool`：调用方借入 storage、TLSF 风格的 two-level segregated-fit variable-size
   allocate/free/realloc；一级 logarithmic range、每级 8 个线性 subdivision 与两层 bitmap
   把空闲块定位限制为 O(1)，不会随 free-block 数量退化为扫描。侵入式 free list 和 boundary
   metadata 使插入、删除与邻块合并也是 O(1)。一个 recent-free cursor 只作为时间局部性快路，
   未命中后至多取得一个 bitmap-selected list head。realloc 优先复用 retained span 或后一空闲
   邻块，常见 shrink/regrow 与 adjacent grow 均为 O(1)；只有必须搬移时复制
   O(min(old,new))，并留在 noinline 冷路。它单线程、无 fallback；约 4 KiB 的显式 control
   是长寿命通用固定堆为 bounded lookup 支付的常量成本，不适合作为每个 request 的临时对象。

## 9. NUMA、大页与专用模块

- `HugeRegion`、显式 NUMA bind/interleave、THP/prefault 保留；bind/interleave 互斥，interleave
  只能搭配 NUMA-any 且不能走 local factory；策略失败是可观察的降级或错误，不保证自动 NUMA 最优。
- `reserve_huge_1g_pages` 是进程启动期冷控制面，原生 C++ 返回 `Status`，C ABI 原样转移同一
  representation；不得在 C 层把 bool 重新猜成错误原因。mimalloc 当前只暴露 ENOMEM，映射
  `ResourceExhausted` 并在 message 保留 native value，非法 node/溢出为 `OutOfRange`。
- `FiberStackPool` 保留且是当前内部刚需。
- graph 当前可直接消费通用 owned Region；`GraphArena::measure/reserve/allocate` 这类 graph-aware worst-case/liveness 策略等到对应消费者落地，归 `graph/`。
- `Buffer(kHostPinned)` 的 native-handle/mapped-pointer 合同先在 P0 纠正；`StagingRing` 等到至少两个异步传输在途、slot 复用受 Event 约束时再落地，归 `compute/`。
- io provided-buffer ring 归 `io/`。
- generic `MemorySpace`、把 unified/device memory 塞进 mem allocator、自动 NUMA 迁移、soft limit 冒充 hard quota 均否决。

## 10. 公共能力矩阵

| 用户形状 | C++ 原生产品面 | C/FFI 产品面 | 合同 |
|---|---|---|---|
| 默认/注入分配 | `SystemAllocator`、`Allocator`、typed helpers | `hyle_system_allocator`、`hyle_allocator`、inline alloc/free/realloc | 默认 direct path；运行时注入保留 backing identity |
| heap size/pressure | `system_good_size/usable_size/collect` | 同名 `hyle_system_*` | 显式查询与回收，不隐藏周期线程 |
| byte admission | `QuotaAllocator` | `hyle_quota_allocator_*` | strict live requested-byte budget |
| 统计/泄漏诊断 | `StatsAllocator::create(backing,track,metadata)`、`LeakTracker::create(metadata)` | `hyle_stats_allocator_*` + POD snapshot/bounded copy | opt-in；被测/诊断域分离；metadata OOM 的 dropped/complete 可观察；默认 system 热路零统计税 |
| owned request scratch | `Region` | `hyle_region_*` + options | growable、bulk clear、可设 reserved-byte hard limit |
| caller-buffer scratch | `FixedRegion` | 三字按值 `hyle_fixed_region` | no heap、no fallback、bulk reset |
| borrowed/LIFO scratch | `PooledRegion`、`StackAllocator` | 对应 opaque handle | 单线程；borrowed block 或 owned slab |
| external variable heap | `FixedPool` | in-place control 或显式 control allocator | caller storage、individual free/realloc、two-level bitmap O(1) fit/merge、no fallback；约 4 KiB 长寿命 control |
| hard shared slots | `SharedPool`、`ObjectPool<T>` | `hyle_shared_pool_*` + typed raw-slot 宏 | remote free 立即可见、精确 hard capacity |
| owner-local soft cache | `SharedPoolCache`、`ObjectPoolCache<T>` | `hyle_shared_pool_cache_*` + typed raw-slot 宏 | 显式 owner/cache/publish，容量是 soft visibility window |
| growable owner-local slots | `GrowablePool<T>` | runtime size/alignment `hyle_growable_pool_*` | 单线程、几何增长、Reconstruct |
| preserve/RAII loan | `PreservingObjectPool<T>`、`PoolLoan` | raw slot acquire/release，由调用方管理 init/cleanup | 能力可组合；C 不伪造 ctor/dtor/RAII |
| PMR | `PmrResource` / `AlignedPmrResource` | 不适用 | C++ 标准库适配；后者只提高 container buffer 的 base alignment，不保证 element 间隔 |
| guarded fiber stack | `FiberStack`、`FiberStackPool` | sched C ABI 内部消费 | scheduler-owned runtime specialization，不暴露危险 raw stack control |
| THP/NUMA span | `HugeRegion` + policy、`reserve_huge_1g_pages` | `hyle_huge_region_*` + policy | 显式请求、可观察 accepted/applied 状态；冷预留失败返回同一 Status |

C ABI 的“完整”按用户能力可达判断，不按 C++ 符号 1:1 判断。模板 ctor/dtor、RAII 和 PMR
没有自然 C 对象形状；raw slot 与显式 cleanup 是 C 的惯用表达。属于 C 用户范围的标量状态
查询则直接提供：Quota `remaining`、FixedRegion `remaining`、FixedPool `peak` 与 GrowablePool
runtime `slot_stride` 均不要求用户重演内部布局。

## 11. 用户选型

```text
通用 host RAM / 默认容器             → SystemAllocator
运行时注入、FFI、mock                → Allocator
std::pmr 容器                         → PmrResource
PMR buffer 需要显式 cache/page 对齐   → AlignedPmrResource
request/frame 整批释放                → Region
调用方外部 buffer 的 bulk scratch     → FixedRegion
调用方外部 buffer 的变长 alloc/free   → FixedPool
固定 slab 上 LIFO scratch             → StackAllocator
单线程同尺寸对象、按需增长            → GrowablePool<T>
多线程固定 block/object、严格容量      → SharedPool / ObjectPool<T>
owner-local 复用热点、容量有余量         → SharedPoolCache / ObjectPoolCache<T> over hard core
昂贵对象保留状态                       → PreservingObjectPool
phase 临时区严格 reserved-byte limit  → Region(size_limit)
任意 backing 的严格 live-byte budget  → QuotaAllocator
fiber guarded stack                   → FiberStackPool
NUMA/THP 大块 host RAM                → HugeRegion + 显式 policy
GPU pinned/mapped host RAM             → compute HostBuffer（native handle + host_data + flush/invalidate）
异步 staging                          → compute StagingRing（待真实 async overlap 消费者）
```

## 12. 收口定义与重开边界

mem 进入预发布冻结必须同时满足：

1. 常见 host-RAM ownership/concurrency/capacity/lifetime 形状已由上表覆盖，专用生命周期明确
   归 compute/graph/io/sched；不再因 benchmark 参数新增 allocator 类型。
2. C++ 与 C/FFI 分别通过用户任务审查；C 产品范围内的能力有直接接口或可复制组合，
   ownership、销毁顺序和失败语义无需理解内部实现才能写对。
3. 最终候选 commit 通过 Windows/Linux/macOS、纯 C、导出清单、OOM/overflow 和适用的
   ASan/UBSan/TSan。
4. throughput-oriented 能力有正常 workload、同合同成熟 peer、throughput 与 whole-operation
   latency；空间型能力有 peak/retention/fragmentation 证据。没有同合同 peer 时只报告绝对成本
   和较弱合同参照，不制造 SOTA 结论。
5. architecture、guide、performance 和 milestone 各守自己的文档职责，不复制相互的状态表。

五项通过即冻结。性能稳定性必须由多进程重复、paired sign/spread、长寿命 epoch 与结构复杂度
门禁共同建立，不能靠一次很短的 microbenchmark；通过已定义证据后也不无限等待。冻结后只有 correctness/security、真实消费者
暴露能力缺失、新平台使合同失效或既定 harness 确认的持续性能回归可以重开；LOC、命名偏好、
亚纳秒波动和无 workload 的猜测不能重开。

## 13. 验证边界

- allocator：typed count overflow、完整 backing identity、alignment、OOM、PMR throw boundary；下游消费者的 NULL/Status/fail-fast 另按各自公开合同审计。
- Buffer host path：native handle 不得冒充 mapped pointer；CUDA/Vulkan/host 三后端都验证 `host_data()` 可写、kernel 参数仍拿正确 native handle。test-only 选择器穷举后确认 Win/Mac 物理设备只提供 coherent host-visible type，non-coherent driver round-trip 明确 skip；范围算术用真实 `nonCoherentAtomSize` synthetic 测试覆盖，不冒充硬件未提供的 driver 路径已执行。
- bump/region：alignment 回绕、owned growth、bulk clear、large block、hard size limit、bytes used/reserved、external storage 无 fallback；多 block 长寿命复用用结构门禁证明每 epoch 至多线性考察，并用 32→2048 blocks 的跨进程 scaling 证书防回归。
- fixed pools：耗尽/回收、remote free、hard quota、soft publish、稳定空快照、tag 饱和、authoritative backing、lifetime/TSan/ASan；FixedPool 另验证大 alignment/overflow、随机 realloc/碎片回收和与 free-block 数量无关的至多两候选查找。
- object pools：Reconstruct + move-only Loan、早退/异常自动归还、hard/soft handoff；Preserve 一次构析与状态保留。
- diagnostics：membership fail-fast 保留；`HYLE_TESTING` 已覆盖 shared/cached/object/growable pool 的 double return/destroy、foreign pointer 与 outstanding allocation，shipping 构建完全剥离额外状态和指令；LeakTracker metadata OOM 必须留下 dropped/complete 证据，不能静默伪称完整。
- performance workload 必须描述 request batch、producer/consumer handoff、quota、expensive-context reuse 等用户行为，并扫描线程、工作集、容量和批次；不能围绕内部阈值造场景。

当前固定池实测见 [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)。

## 参考

- oneTBB scalable allocator / memory_pool / fixed_pool / IndexedBins / `cache_aligned_resource`
- mimalloc local + concurrent free lists / segment span queues
- Folly Arena / ThreadCachedArena
- marl BoundedPool / PoolPolicy / Loan
- TLSF two-level bitmap mapping；Boost.Interprocess `rbtree_best_fit` 作为 O(log N) 备选形状对照
- ggml gallocr 与 llama.cpp pinned upload ring
