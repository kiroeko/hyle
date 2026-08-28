# HYLE mem/ 设计

> 本页是 `mem-v1` 的当前设计边界。mem 只提供 C++23 source/static/shared，Native FFI 为
> `N/A`；BUILD-03 已删除旧 C schema、bridge、动态导出和 fixture。机械收口状态见
> [`module-status.json`](../implementation/module-status.json)，本文不另造状态轴。

> 用户当前选型见 [user/mem](../user/mem.md)，性能证据见
> [mem evidence](../implementation/evidence/performance/benchmarks-mem.md)，施工历史见
> [mem tasks](../implementation/tasks/mem/)。

## 1. 边界

`mem/` 只拥有普通 CPU-addressable host RAM 的分配与生命周期原语：通用 allocator、region/stack、固定块池、对象池、fiber stack、NUMA/大页和诊断适配。

以下能力由使用其完成语义的模块拥有：

- device VRAM、unified/mapped/ReBAR memory 归 `compute/`；可见性、coherence、flush/invalidate、stream/event 生命周期不能伪装成 `mem::Allocator`。
- pinned host buffer 的所有权与 DMA completion 也归 `compute/`。真正 staging slot 必须同时持有 backend handle、CPU mapped pointer、Device/ordinal、Event 和状态。
- io_uring provided-buffer ring 归 `io/`；其注册、buffer id、worker issuer 和 completion 生命周期不是通用 mem 合同。
- graph 的 liveness/worst-case reserve allocator 归 `graph/`；通用 mem 只提供它需要的底层 region/backing。

mem 不提供 pinned allocator 或单块 staging convenience。compute `Buffer` 分开保存 opaque `native_handle()` 与可选 `host_data()`；CUDA/host 两值可相同，Vulkan 分别是 `VkMem*` 与 `VkMem::mapped`。CPU 直接访问用显式 `flush_host()` / `invalidate_host()` 表达非 coherent Vulkan 合同；同步 `copy_from()` 自动包住所需 visibility。未来 staging ring 只能组合 HostBuffer、Event 与 backpressure，不上架裸 host allocator 或 `SharedPool(cudaMallocHost)` 组合。

### 1.1 依赖与构建边界

中央 component graph 与真实 target 链接边一致：
`mem -> PUBLIC {core, error, platform, util_core}`。`core` 提供 cache-layout/compiler 头合同，
`error` 承担构造/VM/NUMA 冷失败，`platform` 承担 CPU、VM 和 NUMA 机制，`util_core` 承担
fiber stack pool 的 SpinLock；mem 不直接散落 OS 分支，也不自造第二套错误。`obs/trace.h` 只是由
`HYLE_TRACE` 控制的跨切 trace seam，不让普通 mem binary 反向依赖 obs runtime。HYLE 的模块和同
工具链消费者直接使用 C++ `Allocator`/`Result`/owner 类型。

安装后消费由 `find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)` + `hyle::mem` 表达；selected-component 的
C++23 shared/static SDK consumer 不能用源码树 include 或测试 target 遮蔽导出/传递依赖缺口。

## 2. 需求基线

### 2.1 HYLE 内部真实消费者

| 能力 | 当前生产消费者 | 真实合同 |
|---|---|---|
| `Allocator` / `system_allocator()` | sched 队列与 driver、HTTP/2 session/stream、IO pending 表、StringInterner、graph 临时容器、ggml bridge job/frame | 按值注入、alignment、sized-free、backing 生命周期；allocator 本身以 NULL 报资源失败，消费者当前混用 Status/失败 handle/fail-fast（graph parallel 临时数组仍 fail-fast），不能伪称已有统一 OOM 策略 |
| `SystemAllocator` 类型 | HYLE containers 的默认模板参数 | 外部默认容器零 vtable；内部关键路径多显式用 `Allocator` |
| `FiberStackPool` | sched worker/fiber | guarded stack、三档尺寸、跨 worker release、Scheduler 覆盖全部 fiber 生命周期 |
| `HugeRegion` | 显式大块内存调用方 | THP/NUMA 是显式策略，不由 pool 根据 allocator identity 偷换 backing |
| graph evaluation 临时区 | `parallel_eval` 的 level/handle/outcome 与 wavefront CSR/state | 明确的 phase lifetime，是通用 owned Region 的内部 workload；graph-aware liveness/reserve 仍归 graph |
| `PooledRegion` / `StackAllocator` / `GrowablePool` | 无生产调用 | 面向外部 infra 用户的 C++ 公开能力；Native FFI 为 `N/A` |
| `SharedPool` / `SharedPoolCache` | 无生产实例 | hard core + 显式 owner-local cache；当前仅公开能力、测试与 benchmark |
| `ObjectPool<T>` / `ObjectPoolCache<T>` | 无生产实例 | typed hard core + 显式 cache，已实现并测试 |
| compute `Buffer` / `DevicePool` | 当前无生产创建点 | 不归 mem；Buffer 已显式分开 native handle、host mapping 与 coherence 操作 |

内部系统已经坐实“通用 allocator + fiber stack”，但 HYLE 是公开 infra 库，其他能力是否保留必须再看成熟库的广泛用户形状，不能只按内部调用计数裁剪。

### 2.2 成熟库反复出现的用户形状

来自 oneTBB、mimalloc、Folly、marl、ggml 的公开 API、测试和真实调用：

1. aligned 通用 C allocator；C 返回空，C++ typed/PMR/allocator-template/RAII 适配边界可抛 `bad_alloc`。
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
| oneTBB scalable/cache-aligned allocator/resource | `SystemAllocator` + `SystemStdAllocator<T>` / `StdAllocator<T>` + `system_memory_resource()` / `PmrResource` / `AlignedPmrResource` | 通用堆借成熟 mimalloc；默认静态绑定与运行时注入分型，高对齐 policy 不污染默认容器 |
| oneTBB `memory_pool`（owned growable variable-size） | 默认组合 `system_memory_resource()` + 标准 PMR pool；运行时 backing 用 `PmrResource` | `FixedPool` 已覆盖 external bounded 版本；没有真实消费者前，不再复制一个与 mimalloc/标准 PMR 重叠的 HYLE-owned growable heap |
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
| `SystemAllocator` | 编译期无状态；static/source 直调 mimalloc，shared 经 mem direct-import seam | 保留；修 typed count 乘法回绕且不向 shared consumer 泄漏 vendor 符号 |
| `Allocator` | C++ 值句柄 `{alloc, free, realloc, state}` | 保留；内部运行时注入脊，不为 Native FFI 预埋另一套形状 |
| `SystemStdAllocator<T>` / `StdAllocator<T>` | 默认静态 system adapter / 借用 `Allocator` 的运行时 adapter | 分型保留；普通容器保持 empty/EBO/always-equal；运行时 adapter 删除默认构造，必须显式给 backing，避免静默选择较慢形状；标准边界 OOM 抛异常 |
| `system_memory_resource()` / `PmrResource` | 默认 system PMR singleton / `Allocator` → PMR | 分型保留；默认只付 PMR virtual + shared seam，策略注入才叠加 allocator dispatch；PMR 零字节请求以 1B backing 成对处理，不把健康资源误报为 OOM |
| `AlignedPmrResource` | 可选 minimum base alignment 的 PMR adapter | 保留；对应 oneTBB `cache_aligned_resource` 的用户需求，不给普通 PMR 强加 cache policy |
| typed/RAII helpers | `allocate_n/zeroed/reallocate_n`、construct/destroy、make_unique | 保留；乘法溢出和显式 alignment 在 System/运行时两层一致 |

通用 malloc 借 mimalloc，不自造任意大小通用堆。sized-free 必须带回原请求的 size/alignment；backing identity 与 ownership 不能丢失。C++ 默认 alignment overload 保持零验证分支；显式动态 allocate/reallocate 在 dispatch 前验证非零 2 次幂。deallocate 的 size/alignment 必须与成功分配匹配，和 `std::pmr::memory_resource`/allocator-template 一样是调用方前置条件：Debug 诊断，Release 不在每次 free 上重复验证。mimalloc v3 会让小于 `MI_MAX_ALIGN_SIZE` 的 size class 使用更弱的自然对齐；HYLE 对默认/更弱 alignment 的 `<16B` 请求只把 backing size 提升到 16 后继续走 plain `mi_malloc/realloc`，trace 与 sized-free 仍记录原请求，over-aligned 才走 `mi_*_aligned`。这条实现假设由 `MI_MALLOC_VERSION==30500` 编译门和 1/8/37B alloc+realloc 测试共同锁定。

### 4.1 交付与 Native FFI 边界

mem 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 C-syntax allocator/status
schema、bridge、动态导出和 fixture；Native FFI 为 `N/A`，没有 C ABI 兼容面。HYLE 模块直接使用
`Allocator`、`Status`、`Result<T>`、`VmRegion` 和各类 RAII owner，不得为所谓“复用 ABI”绕回
opaque C handle。未来只有已命名的 foreign-runtime allocator/buffer 任务通过全局
[Native FFI 准入策略](native-ffi-policy.md)后，才另行设计窄的 runtime 类型擦除边界。

mem 不维护私有 TLS error；C++ 冷路径直接复用 P0 `Status/Result`。HugeRegion 直接传递 platform
VM/NUMA 的 C++ Status，成功零分配，热分配路径不构造 Status。FixedRegion 按值、FixedPool
提供 caller-owned in-place control；StatsAllocator 作为 opt-in 产品能力导出 POD snapshot
与返回 Status 的 bounded leak copy；metadata OOM 不阻断被测 allocation，但 dropped count /
complete bit 使不完整 inventory 可观察。PMR、typed ctor/dtor 与 fiber RAII 没有
自然 C 形状，不为追求符号 1:1 而导出。

进程级 pressure observation 由 `system_memory_snapshot()` 提供稳定 POD：allocator
reserved/committed 与 process RSS/commit 各有 current/peak。它在实现内部读取锁定的 mimalloc
统计/进程查询，不导出 vendor struct、option registry 或 reset；这是冷 O(number of allocator
heaps) 查询，默认 allocation/free 热路不增加 requested-byte 计数。需要精确 owner-local
requested-live 时显式组合 `StatsAllocator`，两种统计不能互相冒充。

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
   O(min(old,new))，并留在 noinline 冷路。`reset()` 在 quiescent phase 边界一次失效全部旧指针，
   不要求先逐块 free；调用后旧指针不得再访问或归还，
   因而这已经是 **variable-size + individual free/realloc + bulk reset**。它单线程、无 fallback；约 4 KiB 的显式 control
   是长寿命通用固定堆为 bounded lookup 支付的常量成本，不适合作为每个 request 的临时对象。

这里 `FixedPool` 的 “fixed” 指 **payload backing 的地址与容量固定**，不是 allocation 大小固定。
如果用户需要池自己取得并增长 backing，C++ 默认组合
`std::pmr::unsynchronized_pool_resource(system_memory_resource())`，跨线程则用 synchronized 版本；
需要 quota/stats/region 等运行时 upstream 才构造 `PmrResource`。Native FFI 当前为 `N/A`；是否
需要面向 foreign runtime 的 growable variable-size pool，只能由未来已命名的消费任务裁决。

## 9. NUMA、大页与专用模块

- `HugeRegion`、显式 NUMA bind/interleave、THP/prefault 保留；bind/interleave 互斥，interleave
  只能搭配 NUMA-any 且不能走 local factory；策略失败是可观察的降级或错误，不保证自动 NUMA 最优。
- system allocator 的 mimalloc backing 默认自动探测 NUMA。它在 loader 阶段、进入用户代码前缓存
  node count；因此运行期 `enable_allocator_numa` 形状无法履行并已移除。部署级 node cap 只能由
  进程启动前的 `MIMALLOC_USE_NUMA_NODES` 提供，运行期 placement 必须走 HugeRegion/platform，
  不能用一个调用成功但实际无效的 setter 冒充能力。
- system allocator 的 THP/large-page 默认同样由启动前的 `MIMALLOC_ALLOW_THP` /
  `MIMALLOC_ALLOW_LARGE_OS_PAGES` 配置。large-page capability 在 Windows loader 初始化时缓存，
  而 Linux 会对未来 mmap 动态读 option；旧的组合 runtime setter 因而跨平台语义不一致并已移除。
  HugeRegion 继续拥有可移植的 per-span policy，1GiB 显式 reserve 继续是真 runtime API。
- selected-node set + `bytes_per_chunk` 由 platform 统一拥有跨 OS 机制，mem 只以
  `HugeRegion::allocate_interleaved(bytes,nodes,chunk,flags)` 提供大块 owner 形状。空 node span
  表示全部可用节点；顺序、重复 id 的 weighted placement、page 粒度和能力失败沿用 platform
  合同；`interleave_chunk_size()` 可观察实际条带。typed node span 保持 platform→mem 的单一表示，
  调用方不需要手写有符号 cast。
  要保持自定义条带 policy 粒度时默认不请求 THP；显式叠加 THP 仍只是 advice，物理大页粒度
  可能覆盖更小条带。Linux 极细自定义布局会在构造期 first-touch，以常量 scratch 和有界 VMA
  policy 换掉无界 kernel metadata；普通大 chunk 仍保留 demand paging。
- `reserve_huge_1g_pages` 是进程启动期冷控制面，返回 `Status`，不得把 bool 重新猜成错误原因。
  mimalloc 当前只暴露 ENOMEM，映射
  `ResourceExhausted` 并通过 `PosixErrno` detail 保留 native value，非法 node/溢出为 `OutOfRange`。
- `FiberStackPool` 保留且是当前内部刚需；capacity 是真实 clamp 后的 retention 上限，0 禁用
  cache，`trim/clear` 是冷回收。Scheduler 在全部 fiber drain、worker join 后 clear 三个 tier，
  这是不会与 release 竞争的 owner-level 回收点；不在普通 idle 热路周期扫描。
- graph 当前可直接消费通用 owned Region；`GraphArena::measure/reserve/allocate` 这类 graph-aware worst-case/liveness 策略等到对应消费者落地，归 `graph/`。
- `Buffer(kHostPinned)` 的 native-handle/mapped-pointer 合同先在 P0 纠正；`StagingRing` 等到至少两个异步传输在途、slot 复用受 Event 约束时再落地，归 `compute/`。
- io provided-buffer ring 归 `io/`。
- generic `MemorySpace`、把 unified/device memory 塞进 mem allocator、自动 NUMA 迁移、soft limit 冒充 hard quota 均否决。

## 10. 公共能力矩阵

| 用户形状 | C++23 产品面 | Native FFI | 合同 |
|---|---|---|---|
| 默认/注入分配 | `SystemAllocator`、`Allocator`、typed helpers | `N/A` | 默认 direct path；运行时注入保留 backing identity |
| allocator-template adapter | `SystemStdAllocator<T>` / `StdAllocator<T>` | `N/A` | 默认 empty/always-equal 静态 system 形状；注入形状无默认构造，支持 rebind、完整 backing equality、借用生命周期；标准 OOM exception boundary |
| heap size/pressure | `system_good_size/usable_size/collect`、`system_memory_snapshot()` | `N/A` | 显式查询/回收；冷 snapshot 报 allocator reserved/committed 与 process RSS/commit current+peak；不隐藏周期线程、不污染默认热路 |
| byte admission | `QuotaAllocator` | `N/A` | strict live requested-byte budget |
| 统计/泄漏诊断 | `StatsAllocator::create(backing,track,metadata)`、`LeakTracker::create(metadata)` | `N/A` | opt-in；被测/诊断域分离；metadata OOM 的 dropped/complete 可观察；默认 system 热路零统计税 |
| owned request scratch | `Region` | `N/A` | growable、bulk clear、可设 reserved-byte hard limit |
| caller-buffer scratch | `FixedRegion` | `N/A` | no heap、no fallback、bulk reset |
| borrowed/LIFO scratch | `PooledRegion`、`StackAllocator` | `N/A` | 单线程；borrowed block 或 owned slab |
| external variable heap | `FixedPool` | `N/A` | caller storage、individual free/realloc + bulk reset、two-level bitmap O(1) fit/merge、no fallback；约 4 KiB 长寿命 control |
| hard shared slots | `SharedPool`、`ObjectPool<T>` | `N/A` | remote free 立即可见、精确 hard capacity |
| owner-local soft cache | `SharedPoolCache`、`ObjectPoolCache<T>` | `N/A` | 显式 owner/cache/publish，容量是 soft visibility window |
| growable owner-local slots | `GrowablePool<T>` | `N/A` | 单线程、几何增长、Reconstruct |
| preserve/RAII loan | `PreservingObjectPool<T>`、`PoolLoan` | `N/A` | 能力可组合；不把 ctor/dtor/RAII 伪造成动态 ABI |
| PMR | `system_memory_resource()` / `PmrResource` / `AlignedPmrResource` | `N/A` | 默认 system singleton 与运行时注入分型；零字节请求仍返回可成对释放的有效 allocation；Aligned 只提高 buffer base alignment，不保证 element 间隔 |
| guarded fiber stack | `FiberStack`、`FiberStackPool` | `N/A` | scheduler-owned runtime specialization；有界 retention + cold trim；shutdown 自动归还，不暴露危险 raw stack control |
| THP/NUMA span | `HugeRegion` + policy/selected nodes/chunk、`reserve_huge_1g_pages` | `N/A` | system allocator 自动拓扑；启动前环境限制；显式 span 请求可观察 accepted/applied/chunk，冷预留失败返回同一 Status |

## 11. 用户选型

```text
通用 host RAM                         → SystemAllocator
运行时注入、mock                    → Allocator
默认 system 的 std::pmr 容器          → system_memory_resource()
运行时 backing 的 std::pmr 容器       → PmrResource
默认 system 的 allocator 模板参数     → SystemStdAllocator<T>
运行时 backing 的 allocator 模板参数  → StdAllocator<T>
PMR buffer 需要显式 cache/page 对齐   → AlignedPmrResource
request/frame 整批释放                → Region
调用方外部 buffer 的 bulk scratch     → FixedRegion
调用方外部 buffer 的变长 alloc/free/reset → FixedPool
自增长 backing 的变长 alloc/free/release  → std::pmr pool_resource over system_memory_resource()
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
2. 保持 `Native FFI=N/A`；若出现已命名的 foreign-runtime 任务，先按全局准入策略单独立项，
   不能在 mem 修改中顺手恢复旧 C schema。
3. 最终候选 commit 通过 Windows/Linux/macOS、OOM/overflow 和适用的 ASan/UBSan/TSan，并验证
   source/static/shared C++23 消费路径。
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

当前固定池实测见 [`../performance/benchmarks-mem.md`](../implementation/evidence/performance/benchmarks-mem.md)。

## 参考

- oneTBB scalable allocator / memory_pool / fixed_pool / IndexedBins / `cache_aligned_resource`
- mimalloc local + concurrent free lists / segment span queues
- Folly Arena / ThreadCachedArena
- marl BoundedPool / PoolPolicy / Loan
- TLSF two-level bitmap mapping；Boost.Interprocess `rbtree_best_fit` 作为 O(log N) 备选形状对照
- ggml gallocr 与 llama.cpp pinned upload ring
