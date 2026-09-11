# HYLE Mem：统一 storage foundation

> 本页记录 Mem 的长期产品边界与设计理由；当前已可依赖的实现以
> [user/mem](../user/mem.md) 与中央 registry 为准，不会因 architecture 写了目标就提前变成
> shipping 能力。`mem_reclaim` 由普通 Mem façade 聚合，而不是公开内部 factor target；它的机械
> 收口状态仍只读中央 registry。Mem 只提供 C++23 source/static/shared，Native FFI 为 `N/A`；BUILD-03 已删除旧
> C schema、bridge、动态导出和 fixture。机械收口状态见
> [`module-status.json`](../implementation/module-status.json)，本文不另造状态轴。

> 用户当前选型见 [user/mem](../user/mem.md)；当前 closure record 与性能证据只从
> [implementation/status](../implementation/status.md) 进入。

## 1. 边界

Mem 是 HYLE 唯一 storage-semantics owner，目标产品面同时覆盖：

- host raw allocation、owned heap、arena、fixed/object/block pool、quota/stats；
- 不接触 payload 的 64-bit aligned logical-range suballocation；
- anonymous/file/named shared mapping、range protection/advice/residency、NUMA/大页和 guarded stack；
- host、pinned、device、managed/unified 与 custom storage 的开放 `MemoryResourceRef`；
- move-only `Buffer`、mutable/const view、import/composite、host mapping 与 sync range operation；
- resource-aware transient pool/stats 与不含 Tensor 语义的 lifetime packing plan。

语义拥有不等于 Mem 自己实现每个平台机制。Platform 保留 raw VM/NUMA/page、同步
`FileHandle` 与其它 OS mechanism；Mem 唯一拥有 anonymous/file/shared storage mapping 及统一
owner/view/resource 语义。需要 `BufferView`、跨 provider 组合或 residency 的用户走 Mem；自建
allocator/JIT owner 才直接组合 Platform `VmRegion`，已有 `FileHandle` 可 move 给 Mem
`FileMapping`。两层不是两套 storage policy。未来 Compute 保留 Device/Context/Queue/Completion、
backend/provider allocation artifact 和有序 async operation；IO 保留 namespace/handle transfer/async
protocol；Graph 把 DAG/tensor 事实翻译成 Mem planner 不理解业务语义的 interval/resource/alias 请求。
长期依赖方向固定为：

```text
compute | graph | io  --->  Mem product
                              |-- mem ----> platform
                              |-- mem_vm -> mem + platform + platform_fs + util_core
                              +-- mem_reclaim -> core
```

Mem 不包含 Device/Stream/Event/Tensor/KV/model policy，也不把 device memory 塞进 host
`AllocatorRef`。未来 Compute 向 Mem 注入长寿命 provider，并以类型化 adapter 对 Mem `BufferView`
做 async copy/submit；Mem 不用 `void* stream`、函数注册表或隐藏 staging 反向模拟 completion。
`mem_reclaim` 不分配 payload，也不是并发容器；它负责 read-mostly publication 的 reader lifetime，
Container 负责把该机制组合进自己的旧 table 生命周期。

### 1.1 依赖与构建边界

中央 component graph 与真实 target 链接边一致。base factor
`mem -> PUBLIC {core, error, platform}` 拥有 allocator/resource/buffer/pool/planner；storage factor
`mem_vm -> PUBLIC {core, error, mem, platform, platform_fs, util_core}` 拥有 mapping、LargeMapping 与
FiberStack。面向用户的 `hyle::mem` product facade 聚合三者；内部 consumer 直接链接实际 factor。
factor `mem_reclaim -> PUBLIC {core}` 由该 facade 聚合。`core` 提供 cache-layout/compiler 头合同，
`error` 承担构造/VM/NUMA 冷失败，`platform/platform_fs` 承担 OS 机制与 file handle，`util_core`
承担 fiber stack pool 的 SpinLock；mem 不自造第二套错误。`obs/trace.h` 只是由
`HYLE_TRACE` 控制的跨切 trace seam，不让普通 mem binary 反向依赖 obs runtime。HYLE 的模块和同
工具链消费者直接使用 C++ `AllocatorRef`/`Result`/owner 类型。

安装后消费由 `find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)` + `hyle::mem` 表达；selected-component 的
C++23 shared/static SDK consumer 不能用源码树 include 或测试 target 遮蔽导出/传递依赖缺口。

### 1.2 `mem_reclaim` 的产品边界

该 factor 提供显式 `ReclamationDomain`、move-only read guard、caller-owned intrusive retire record、
非阻塞 collection、grace-period synchronize/drain 和 pressure observation。普通读路径不分配；前 64 个
并发参与线程使用每 domain 独占 cache-line slot，超出后进入固定 fallback stripe。typed retire 只生成
编译期 raw callback thunk，不改变 record/domain layout。

准入的首个真实产品组合是 `ReclaimingTableConcurrentNodeHashMap` 的旧 bucket table hand-off。普通
`ConcurrentNodeHashMap` 保持零 guard 的有界 retained 路线；持续增长 consumer 显式选择服务期回收，
避免把内存与 latency 取舍伪装成唯一合同。RCU 与 Container 分别拥有 reader lifetime 和 table/node
语义，形成真实直接依赖而不互相复制。

明确负空间包括 hazard pointer、executor/GPU-fence affine callback、自动分配 retire node 和业务 snapshot
schema。一个 stalled reader 必须不能阻塞无关回收时，才以真实 consumer 另开 hazard-pointer 能力轮；
GPU completion 属 Compute/Sched owner 组合，不反向塞进 Mem。历史恢复过程保留在 closure task，当前状态
只读 [`module-status.json`](../implementation/module-status.json)。

## 2. 需求基线

### 2.1 HYLE 内部真实消费者

| 能力 | 当前生产消费者 | 真实合同 |
|---|---|---|
| `AllocatorRef` / `system_allocator()` | sched 队列与 driver、HTTP/2 session/stream、IO pending 表、StringInterner 等已实现消费者 | 按值注入、alignment、sized-free、backing 生命周期；allocator 本身以 NULL 报资源失败；QUEUED 的 Compute/Graph 旧调用不作为当前能力证明 |
| `SystemAllocator` 类型 | HYLE containers 的默认模板参数 | 外部默认容器零 vtable；内部关键路径多显式用 `AllocatorRef` |
| `FiberStackPool` | sched worker/fiber | guarded stack、三档尺寸、跨 worker release、Scheduler 覆盖全部 fiber 生命周期 |
| `LargeMapping` | 显式大块内存调用方 | THP/NUMA 是显式策略，不由 pool 根据 allocator identity 偷换 backing |
| 未来 Graph compilation/invocation 临时区 | 当前无已验收生产创建点 | compiler scratch 可用 arena；Graph 产生 interval/alias/resource facts，通用 packing/placement 语义归 Mem planner |
| `PooledArena` / `StackArena` / `GrowableObjectPool` | 无生产调用 | 面向外部 infra 用户的 C++ 公开能力；Native FFI 为 `N/A` |
| `ConcurrentBlockPool` / `BlockPoolCache` | 无生产实例 | hard core + 显式 owner-local cache；当前仅公开能力、测试与 benchmark |
| `ObjectPool<T>` / `ObjectPoolCache<T>` | 无生产实例 | typed hard core + 显式 cache，已实现并测试 |
| 未来 compute host-visible/device/managed provider | 当前无已验收创建点 | Mem 已拥有 Buffer/View/pool 语义；Compute 提供 context-owned provider artifact、device identity、kernel binding 与 async ordering；不预留旧 `DevicePool` 形状 |

内部系统已经坐实“通用 allocator + fiber stack”，但 HYLE 是公开 infra 库，其他能力是否保留必须再看成熟库的广泛用户形状，不能只按内部调用计数裁剪。

### 2.2 成熟库反复出现的用户形状

来自 oneTBB、mimalloc、Folly、marl、ggml 的公开 API、测试和真实调用：

1. aligned 通用 C allocator；C 返回空，C++ typed/PMR/allocator-template/RAII 适配边界可抛 `bad_alloc`。
2. remote free：分配线程与释放线程不同，不能假设 thread-affine。
3. owned request/frame/compiler phase region：大量小对象，individual free 无效，clear/destroy 整批释放；Folly `Arena` 的 min-block growth、usage 与 hard size limit 是直接参考。
4. 固定容量共享池：耗尽必须可观察；hard quota 与 thread-cache soft limit 必须是不同类型/名称。
5. typed pool 的 RAII Loan/return 是稳定易用性需求；Reconstruct/Preserve 在 marl 与 Folly expensive-context pool 中都有实证，但并不自动证明每个 bounded pool 都必须同时支持两种 policy。
6. external fixed buffer：oneTBB `fixed_pool` 证明 embedded/realtime/调用方自有 storage 的需求，但它是可 individual free 的 variable-size pool，不能与 bump Arena 混为一谈。
7. strict byte limit 的强证据首先是 `Arena(size_limit)`；通用 sized-free `QuotaAllocator` 是进一步外推，不能冒充成熟库共同基线。
8. explicit NUMA/hugepage policy：capability query、失败降级或显式报错，不承诺自动迁移。
9. tensor-free lifetime packing 进入 Mem，Graph 仍拥有 DAG 推导；GPU staging owner 属拥有 Completion 生命周期的 Compute，不进入 Mem。

### 2.3 参考能力逐项处置

成熟库用于发现需求，不用于机械增加同名类型；差异必须能落到已有合同或明确的模块边界：

| 成熟形状 | HYLE 处置 | 理由 |
|---|---|---|
| oneTBB scalable/cache-aligned allocator/resource | `SystemAllocator` + `SystemStdAllocator<T>` / `StdAllocator<T>` + `system_memory_resource()` / `PmrResource` / `AlignedPmrResource` | 通用堆借成熟 mimalloc；默认静态绑定与运行时注入分型，高对齐 policy 不污染默认容器 |
| oneTBB `memory_pool`（owned growable variable-size） | isolated owner 用 `Heap`；按 size class 缓存的 PMR consumer 组合标准 pool resource，运行时 upstream 用 `PmrResource` | `Heap` 提供 individual free、remote free 与 bulk invalidation；`FixedHeap` 覆盖 external bounded 版本；不再复制另一套通用 malloc/pool 算法 |
| oneTBB `fixed_pool` | `FixedHeap` | caller-owned storage、individual alloc/free/realloc、无 fallback；内部 two-level bitmap 给出 bounded fit |
| VMA/D3D12MA virtual block、OffsetAllocator | `RangeAllocator` / `RangeAllocation` | metadata 与 opaque backing 分离；64-bit aligned range、individual RAII release、bulk invalidation 和显式 metadata 上限 |
| Folly `Arena` | `Arena` | owned growable bulk lifetime、size limit、usage/reserved、large-block release；clear 后同样单调走 retained blocks |
| Folly `ThreadCachedArena` | 不内建隐藏 TLS registry | 用户可按 worker/thread 持有 `Arena`；线程退出合并/全局销毁属于 runtime owner（如 sched），基础 mem 不猜 TLS lifetime |
| mimalloc heap/arena/local+remote free | `SystemAllocator` 与 `ConcurrentBlockPool`/显式 cache | 任意大小 heap 直接借核；严格固定容量和 soft owner cache 用不同公开类型表达 |
| marl bounded pool / policy / loan | `ConcurrentBlockPool`、`ObjectPool<T>`、`PreservingObjectPool<T>`、`PoolLoan` | hard capacity、Reconstruct/Preserve 和 RAII return 分型，不塞进一个 policy benchmark 类型 |

Folly `Arena::merge` 服务其 `ThreadCachedArena` 的 thread-exit zombie 收集；在 HYLE 没有对应
runtime owner/consumer 前不单独上架，否则还需定义不同 backing、size limit、并发移交和 source
失效的完整合同。真实消费者出现后应由拥有线程生命周期的模块先给出需求，而不是因参考库有
一个方法就扩面。

## 3. 公开类型必须表达的五个合同

任何 pool/arena 不能只用“更快 allocator”描述；类型、构造参数和方法名必须让用户看见：

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
| `AllocatorRef` | C++ 值句柄 `{alloc, free, realloc, state}` | 保留；内部运行时注入脊，不为 Native FFI 预埋另一套形状 |
| `SystemStdAllocator<T>` / `StdAllocator<T>` | 默认静态 system adapter / 借用 `AllocatorRef` 的运行时 adapter | 分型保留；普通容器保持 empty/EBO/always-equal；运行时 adapter 删除默认构造，必须显式给 backing，避免静默选择较慢形状；标准边界 OOM 抛异常 |
| `system_memory_resource()` / `PmrResource` | 默认 system PMR singleton / `AllocatorRef` → PMR | 分型保留；默认只付 PMR virtual + shared seam，策略注入才叠加 allocator dispatch；PMR 零字节请求以 1B backing 成对处理，不把健康资源误报为 OOM |
| `AlignedPmrResource` | 可选 minimum base alignment 的 PMR adapter | 保留；对应 oneTBB `cache_aligned_resource` 的用户需求，不给普通 PMR 强加 cache policy |
| typed/RAII helpers | `allocate_n/zeroed/reallocate_n`、construct/destroy、make_unique | 保留；乘法溢出和显式 alignment 在 System/运行时两层一致 |

`Heap` 是独立、move-only 的 growable host heap owner。它直接持有一个 mimalloc heap，不为 HYLE
wrapper 再建 control block；`allocator()` 借用 owner，因此 `clear()` 成功换入新 native heap 后原
borrow 仍可用，move/destruction 则使 borrow 失效。allocate/reallocate 由 owner thread 发起，
deallocate 允许 remote free；`collect()` 只回收空闲页，`clear()/release()` 是必须先 quiesce 所有
allocation 的 bulk invalidation。它与 PMR pool 的区别是 ownership/lifetime，不以 benchmark 档位命名。

通用 malloc 借 mimalloc，不自造任意大小通用堆。sized-free 必须带回原请求的 size/alignment；backing identity 与 ownership 不能丢失。C++ 默认 alignment overload 保持零验证分支；显式动态 allocate/reallocate 在 dispatch 前验证非零 2 次幂。deallocate 的 size/alignment 必须与成功分配匹配，和 `std::pmr::memory_resource`/allocator-template 一样是调用方前置条件：Debug 诊断，Release 不在每次 free 上重复验证。mimalloc v3 会让小于 `MI_MAX_ALIGN_SIZE` 的 size class 使用更弱的自然对齐；HYLE 对默认/更弱 alignment 的 `<16B` 请求把实际 backing size 提升到 16 后继续走 plain `mi_malloc/realloc`，trace 保留原请求，传给 `mi_free_size` 的 size 则与实际 backing 请求一致；over-aligned 路线成对使用 `mi_*_aligned` / `mi_free_size_aligned`。这条实现假设由 `MI_MALLOC_VERSION==30501` 编译门和 1/8/37B alloc+realloc 测试共同锁定。

mimalloc v3.5.1 把非 Android 的 `MI_DEFAULT_ALLOW_THP` 从 1 改为 2；后者同时把最小 purge
粒度保持在 2 MiB，改变普通服务的 RSS retention。HYLE custom target 在非 Android 明确编译为 1，
保留已审计的普通 THP baseline；Android 保持上游 0。部署仍可在进程启动前通过
`MIMALLOC_ALLOW_THP=0/2` 选择禁用或 FULL retention。HYLE 不复制上游 CMake 的 host-native ISA
开关，发布二进制继续服从 HYLE/toolchain 的可迁移 ISA 基线。

### 4.1 交付与 Native FFI 边界

mem 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 C-syntax allocator/status
schema、bridge、动态导出和 fixture；Native FFI 为 `N/A`，没有 C ABI 兼容面。HYLE 模块直接使用
`AllocatorRef`、`Status`、`Result<T>`、`VmRegion` 和各类 RAII owner，不得为所谓“复用 ABI”绕回
opaque C handle。未来只有已命名的 foreign-runtime allocator/buffer 任务通过全局
[Native FFI 准入策略](native-ffi-policy.md)后，才另行设计窄的 runtime 类型擦除边界。

mem 不维护私有 TLS error；C++ 冷路径直接复用 P0 `Status/Result`。LargeMapping 直接传递 platform
VM/NUMA 的 C++ Status，成功零分配，热分配路径不构造 Status。FixedArena 按值、FixedHeap
提供 caller-owned in-place control；StatsAllocator 作为 opt-in 产品能力导出 POD snapshot
与返回 Status 的 bounded leak copy；metadata OOM 不阻断被测 allocation，但 dropped count /
complete bit 使不完整 inventory 可观察。PMR、typed ctor/dtor 与 fiber RAII 没有
自然 C 形状，不为追求符号 1:1 而导出。

进程级 pressure observation 由 `system_memory_snapshot()` 提供稳定 POD：allocator
reserved/committed 与 process RSS 各有 current/peak；private commit 只在 OS 能证明时出现为 optional。
它在实现内部读取锁定的 mimalloc
统计/进程查询，不导出 vendor struct、option registry 或 reset；这是冷 O(number of allocator
heaps) 查询，默认 allocation/free 热路不增加 requested-byte 计数。需要精确 owner-local
requested-live 时显式组合 `StatsAllocator`，两种统计不能互相冒充。

## 5. 固定块池

`ConcurrentBlockPool` 独占 `detail::ShardedSlab`；`BlockPoolCache` 通过 narrow batch hooks 组合在 hard core 上，不复制 slab，也不另建 TLS registry。

### 5.1 `ConcurrentBlockPool`：hard bounded

- 固定 block size + runtime capacity。
- completed `deallocate()` 立即进入 shared core；remote free 可见。
- 两次稳定全-shard 快照消除 block 在扫描背后迁移造成的假耗尽。
- home-shard 成功路径只做一次 CAS，保持 expected O(1)；O(shards) observation 数组与稳定空
  快照被隔离到 noinline exhaustion 冷路，不让正常成功调用保留约 2 KiB 栈帧。
- 64-bit packed head；32-bit version 饱和后永久进入该 shard 的 serialized mode，不回绕。
- 用于严格 object/block quota、跨线程 handoff；本地 request churn 不是它的吞吐强项。

### 5.2 `BlockPoolCache`：显式 owner-local soft cache

- 构造时绑定 `ConcurrentBlockPool`；caller-owned `span<uint32_t>` 可做到 cache metadata 零隐藏分配，也可显式注入 metadata allocator。
- capacity 与 transfer batch 是每个 cache 的公开参数；128/32 只是 convenience 默认值。
- owner-local 命中零原子；其他 cache 中的 free block 暂不可见；`publish()`/析构是显式可见性边界。
- cache 本身 single-owner，不维护 TLS node/registry；多个 cache 可并发绑定同一 hard core。
- 只用于池容量高于并发 working set、owner-local 复用为主的场景；不得宣传为 hard capacity。

两层保留。默认 shipping 热路径不增加 per-slot bitmap/atomic live counter；foreign/misaligned pointer 仍做无状态 membership fail-fast。double free、outstanding object 属调用方契约违反，`HYLE_TESTING` 用 per-slot state fail-fast；额外状态和指令在 shipping 构建完全剥离，不破坏 cache 零原子合同。

`ConcurrentBlockPool` 始终服从 injected backing。需要 THP/NUMA 的调用方应显式提供相应 allocator/provider；pool 不根据 allocator identity 偷换 storage domain。

## 6. typed object pool

`ObjectPool<T>` 与附着其上的 `ObjectPoolCache<T>` 已实现：placement-new、显式析构、构造异常退槽、OOM/耗尽返回 NULL；后者暴露 `publish()`。当前策略固定为每次 create/destroy 都 Reconstruct。

产品处置：保留 hard core / soft cache typed shell，并补齐独立的生命周期能力：

1. move-only `PoolLoan<T>` 已实现，作用域退出自动 destroy/return，减少异常/早退漏还；Pool 必须比 Loan 长寿。
2. 当前手工 `create/destroy` 保持 `Reconstruct` 合同：borrow 时构造、return 时析构。pool 析构前必须已归还并析构全部 live object；这是调用方合同，不得暗藏。
3. `PreservingObjectPool<T>` 已作为独立类型实现：池构造时一次性构造全部对象，return 保留状态，池析构时一次性析构；不与 slot cache 混名。
4. 默认 API 仍是 nonblocking try；需要“等到有对象”的调用方在 sched 层用 Semaphore/Channel 包装。
5. 当前 shipping 对 foreign/misaligned pointer 做 membership fail-fast；`HYLE_TESTING` 已补 per-slot liveness、double return/destroy 与析构时 outstanding allocation，shipping 不加原子计数税。

`ObjectPoolCache<T>` 不是 Preserve object cache；Cache 只表示 free slot 的 owner-local 可见性。Preserve 必须由独立 policy/name 表达。

## 7. arena、stack 与增长池

### 7.1 `PooledArena`

`PooledArena` 只借 `ConcurrentBlockPool` 的一个 hard block，固定容量、单线程、bulk `clear()`。库尚未发布，旧 `Arena/reset` 双命名已删除，Arena 家族只保留一套公开说法。

### 7.2 owned `Arena` 与 external `FixedArena`

owned growable `Arena` 已提供：

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

`FixedArena(span<byte>)` 是 external-storage bump 机制：无隐藏 payload 分配、无增长、无 fallback。
compiler/evaluation 的逐阶段临时数组是典型 phase-lifetime workload；未来 Graph 生成
liveness/alias/resource facts，Mem planner 拥有通用 measure/placement。

### 7.3 `StackArena`

保留固定 owned slab + mark/rewind 的 LIFO scratch 合同。对齐计算必须防 uintptr 加法回绕；极端 alignment 是调用方契约边界，不得静默产生错误指针。

### 7.4 `GrowableObjectPool<T>`

保留单线程、同尺寸、按需增长的 Reconstruct pool。release 必须来自同一 pool 的 live slot；默认 shipping 可保持零元数据，diagnostics 模式检查 membership/double release/live underflow。

## 8. strict limit 与 external storage

当前 hard `ConcurrentBlockPool` 解决固定块数量；`Arena(size_limit)` 解决 region reserved-byte 上限；`QuotaAllocator` 解决任意 sized-free backing 的 live requested-byte 上限。三者不能互相冒充。

external 能力按 allocation model 分成两个类型：

1. `FixedArena`：调用方借入 storage、bump + bulk reset、无 individual free。
2. `FixedHeap`：调用方借入 storage、TLSF 风格的 two-level segregated-fit variable-size
   allocate/free/realloc；一级 logarithmic range、每级 8 个线性 subdivision 与两层 bitmap
   把空闲块定位限制为 O(1)，不会随 free-block 数量退化为扫描。侵入式 free list 和 boundary
   metadata 使插入、删除与邻块合并也是 O(1)。一个 recent-free cursor 只作为时间局部性快路，
   未命中后至多取得一个 bitmap-selected list head。realloc 优先复用 retained span 或后一空闲
   邻块，常见 shrink/regrow 与 adjacent grow 均为 O(1)；只有必须搬移时复制
   O(min(old,new))，并留在 noinline 冷路。`reset()` 在 quiescent phase 边界一次失效全部旧指针，
   不要求先逐块 free；调用后旧指针不得再访问或归还，
   因而这已经是 **variable-size + individual free/realloc + bulk reset**。它单线程、无 fallback；约 4 KiB 的显式 control
   是长寿命通用固定堆为 bounded lookup 支付的常量成本，不适合作为每个 request 的临时对象。

这里 `FixedHeap` 的 “fixed” 指 **payload backing 的地址与容量固定**，不是 allocation 大小固定。
如果用户需要池自己取得并增长 backing，C++ 默认组合
`std::pmr::unsynchronized_pool_resource(system_memory_resource())`，跨线程则用 synchronized 版本；
需要 quota/stats/region 等运行时 upstream 才构造 `PmrResource`。Native FFI 当前为 `N/A`；是否
需要面向 foreign runtime 的 growable variable-size pool，只能由未来已命名的消费任务裁决。

### 8.1 Opaque backing 的逻辑区间

`RangeAllocator` 管理 `[0, capacity)` 中的 64-bit `[offset,size)`，但不分配、映射或触碰 payload。
它服务大 device buffer/heap、descriptor backing、pinned staging slab 或 page backing：真正的资源 owner
先取得 backing，再把 returned offset 交给 provider 创建 view/resource。它不能替代 `FixedHeap`——后者
从 caller storage 返回可解引用 host pointer，并把 block metadata 放进该 storage；也不能替代
`LifetimePlanner`——后者要求整张图的 interval 已知，而 range allocator 处理运行期 churn。

公开形状刻意只有一个默认策略。create 时调用者必须给出 `max_allocations`，实现预留最多
`2 * max_allocations + 1` 个 partition node；后续 allocate/free/reset 不增长 metadata、不锁、不 syscall。
`RangeAllocation` 是 move-only RAII identity，因而不存在把 token 交给错误 owner 的公开 free 入口。
owner-wide `reset()` 增加 generation 并重建一个完整 free range；旧 lease 立即失效，稍后析构不会误释放
已经复用的 node。控制块通过 caller-synchronized 非原子引用计数活到最后一个 lease，允许 owner 先析构；
若注入自定义 metadata allocator，其 backing 仍须活到所有 handle 消失。

索引使用 64-bit TLSF-like 两级 size classes。`size + alignment - 1` 可表示且存在相应 free range 时，
bitmap lookup、split、free 和相邻合并均为 O(1)；为避免明明有一个地址已经对齐的小洞却虚假 OOM，
保守 lookup 不命中时会做地址相关 free-range fallback，allocate 最坏 O(F)。这与成熟 aligned virtual
allocator 的真实成本一致；不能为了写“硬 O(1)”永久吞掉 alignment padding，或把 benchmark 内部 bin/
search limit 暴露成用户策略。summary stats O(1)，largest-free/free-range detailed stats 明示 O(N)。

该类型 caller-synchronized。GPU completion 应把 lease move 给拥有 stream/fence 的 pinned completion
owner；并发 wrapper、defragment/move callback、buffer-image granularity 与 KV eviction policy 分别属于
有真实同步或 device policy 的上层能力，不塞进 Mem 默认对象。

### 8.2 Mapping factor 与 warm-up 语义

`mem_vm` 是 file/shared storage mapping 的唯一 public owner；Platform 提供 raw anonymous VM 和
`FileHandle` mechanism。`VirtualMapping` 是 `VmRegion` 的零开销别名，避免两套 enum、wrapper 和
OS decision authority；`FileMapping` 与 `SharedMapping` 直接拥有 native mapping state。所有 owner
都 move-only，不增加 pimpl allocation、payload copy 或常态 dispatch。普通 Mem product 自动聚合
该 factor，用户不需要认识 target 拆分。

`FileMapping::prefetch` 是 `WillNeed` hint，成功不证明 page 已 resident；`populate` 是同步触页，工作
与覆盖页数成正比。open 不默认执行任何一个，避免让只需 lazy mapping 的消费者支付全文件 fault。
residency 只保存一个单调 public prefix，不为任意重叠 interval 增加常态树/refcount metadata。
连续 `FileMapping` 不分配 fragment metadata；POSIX/macOS 第一次有效 partial release 才建立惰性
survivor 表，checked range/view 不能跨 hole。已知一批 loader ranges 的路线先排序、合并，再与 live
fragments 一次相交，bookkeeping 为 `O(R log R + F)`；逐项入口保留便利性，但不宣称一串逐项调用也有
同样总复杂度。Windows whole-view primitive 没有安全同合同，明确返回 `Unimplemented`，调用方改用
多个精确 mapping owners。所有 range 先做 overflow/bounds 检查，unsupported advice/mode 返回共享
`Status`，不隐藏 fallback。
FileMapping 还必须按 backing 裁决 advice：Windows `DiscardVirtualMemory` 的 PAGE_READWRITE/内容未定义
合同不等价于 file-backed `MADV_DONTNEED`，所以该组合明确 `Unimplemented`，绝不能因 Platform 提供
anonymous/private mechanism 就直接套用到 RO/RW/COW 文件视图。

`SharedMapping` 不再在 payload 前嵌入 size header：这样数据从 native mapping base 开始，满足 page/
cache-line alignment，也消除 opener 观察半初始化 header 的窗口。size/access/readiness 因而明确属于
IPC protocol。实现用 native capacity 拒绝超界：Linux shm 的 `fstat` 保留精确 `ftruncate` size；
Darwin shm 与 Windows section capacity 按 native page boundary 向上取整（Windows allocation
granularity 只约束 view offset/base）。后两者不能恢复精确 logical size，双方必须外部协商；不能用
第二个隐藏 header 假装解决。
shared name 同样保留 native namespace identity：POSIX 仅规范化前导 `/`，长度和字符约束由 syscall
权威裁决并翻译为 HYLE Status；实现不能通过静默截断或 hash 引入碰撞/跨进程不一致。

## 9. NUMA、大页与专用模块

- `LargeMapping`、显式 NUMA bind/interleave、THP/prefault 保留；bind/interleave 互斥，interleave
  只能搭配 NUMA-any 且不能走 local factory；策略失败是可观察的降级或错误，不保证自动 NUMA 最优。
- system allocator 的 mimalloc backing 默认自动探测 NUMA。它在 loader 阶段、进入用户代码前缓存
  node count；因此运行期 `enable_allocator_numa` 形状无法履行并已移除。部署级 node cap 只能由
  进程启动前的 `MIMALLOC_USE_NUMA_NODES` 提供，运行期 placement 必须走 LargeMapping/platform，
  不能用一个调用成功但实际无效的 setter 冒充能力。
- system allocator 的 THP/large-page 默认同样由启动前的 `MIMALLOC_ALLOW_THP` /
  `MIMALLOC_ALLOW_LARGE_OS_PAGES` 配置。large-page capability 在 Windows loader 初始化时缓存，
  而 Linux 会对未来 mmap 动态读 option；旧的组合 runtime setter 因而跨平台语义不一致并已移除。
  LargeMapping 继续拥有可移植的 per-span policy；process-global allocator arena reserve 仍是真
  runtime API，但其 GiB 请求单位不再冒充物理页粒度。
- selected-node set + `bytes_per_chunk` 由 platform 统一拥有跨 OS 机制，mem 只以
  `LargeMapping::allocate_interleaved(bytes,nodes,chunk,flags)` 提供大块 owner 形状。空 node span
  表示全部可用节点；顺序、重复 id 的 weighted placement、page 粒度和能力失败沿用 platform
  合同；`interleave_chunk_size()` 可观察实际条带。typed node span 保持 platform→mem 的单一表示，
  调用方不需要手写有符号 cast。
  要保持自定义条带 policy 粒度时默认不请求 THP；显式叠加 THP 仍只是 advice，物理大页粒度
  可能覆盖更小条带。Linux 极细自定义布局会在构造期 first-touch，以常量 scratch 和有界 VMA
  policy 换掉无界 kernel metadata；普通大 chunk 仍保留 demand paging。
- `reserve_large_page_arena` 是进程启动期冷控制面，返回 GiB-sized arena 请求/实际成功容量、placement
  可信度、不可验证的物理页粒度与 outcome。mimalloc 可能从 1 GiB huge pages 回退到 2 MiB/native
  large pages，而其 arena API 只证明容量，不能证明最终 page size；参数/溢出在 vendor call 前走
  `Result` error，partial success 无法 rollback，不能把 vendor bool、请求单位或 arena bytes 猜成
  1 GiB OS page 成功。
- `FiberStackPool` 保留且是当前内部刚需；capacity 是真实 clamp 后的 retention 上限，0 禁用
  cache，`trim/clear` 是冷回收。Scheduler 在全部 fiber drain、worker join 后 clear 三个 tier，
  这是不会与 release 竞争的 owner-level 回收点；不在普通 idle 热路周期扫描。
- Graph compiler 可直接消费通用 arena；它把 worst-case/liveness/alias/resource 翻译为
  `LifetimeRequest`，并在 build 时借给 planner 稠密 `MemoryResourceRef` 以读取 alignment 与单次 allocation
  上限。Mem planner 负责 tensor-free measure/placement/instantiate，模型/KV 策略仍在上层；一个 alias
  root 不跨 physical backing，placement 使用 resource-local backing index + local offset。
- pinned/device/managed 通过开放 `MemoryResourceRef` 进入 Mem `Buffer/View` 语义；
  native-handle/mapped-pointer 不混同，non-coherent 路线显式 flush/invalidate。持有 Queue/Completion 与
  backpressure 的 staging owner 仍归未来 Compute。
- io provided-buffer ring 归 `io/`。
- 闭合 `MemorySpace/BufferKind`、把 unified/device memory 塞进 host `AllocatorRef`、自动 NUMA
  迁移、soft limit 冒充 hard quota 均否决。

## 10. 目标公共能力矩阵

| 用户形状 | Mem C++23 产品面 | 关键合同 |
|---|---|---|
| 默认/注入 host allocation | `SystemAllocator`、`AllocatorRef`、`Heap` 与 std/PMR adapters | direct 与 runtime borrow 分型；owned heap 可 bulk release；backing 长寿 |
| phase/external/LIFO scratch | `Arena`、`FixedArena`、`PooledArena`、`StackArena` | 单 owner；bulk/reset/rewind 失效化明示；无隐式 fallback |
| variable/fixed/object pool | `FixedHeap`、`ConcurrentBlockPool`、`BlockPoolCache`、object-pool 家族 | size model、storage ownership、hard/soft/growable、reconstruct/preserve 都由类型表达 |
| opaque backing 动态分段 | `RangeAllocator` / `RangeAllocation` | 64-bit aligned offset；显式 metadata/live 上限；RAII individual free + generation bulk reset；不拥有 payload |
| mapping/residency | `VirtualMapping`、`FileMapping`、`SharedMapping` | owner 在 Mem；anonymous/range control 与 `FileHandle` 复用 Platform，file/shared native mapping 由 `mem_vm` 实现；不默认 prefault/lock/grow |
| host/device/custom storage | `MemoryResource`、`MemoryResourceRef`、properties/domain/stats | provider 开放；不用闭合 kind；ref 一字且借用 provider |
| persistent/imported/composite bytes | `Buffer`、`BufferView`、`ConstBufferView`、`HostMapping`、`CompositeBuffer` | owner/view 分开；view 四字 O(1) subview；opaque handle 不冒充 host pointer |
| sync data/coherence | view 上 `read/write/fill/copy/map/flush/invalidate` | 无 hidden staging；非 coherent 明示；async ordering 由 IO 或未来 Compute typed adapter 组合 |
| transient storage reuse | `BufferPool` / `BufferLease` | resource-aware reuse/trim/stats；lease 归还前已等待 completion |
| worst-case lifetime packing | `LifetimeRequest/Planner/Plan/PlannedBuffers` | `[begin,end)` interval；resource/backing/local-offset 三层；有限 provider allocation 上限可拆多个 backing；plan/instantiate 分开；O((N+E) log N) / O(N) scratch+metadata；无 Tensor |
| quota/pressure/diagnostics | allocator/resource stats、quota、leak tracker、system snapshot | opt-in 热路税；partial inventory 可观察；soft retention 不冒充 hard admission |
| guarded/THP/NUMA | `FiberStack`、`LargeMapping`、显式 placement policy | requested/accepted/applied 分开；失败可观察；不自动迁移 |

当前 shipping 名称与可用路线由 [user/mem](../user/mem.md) 列出。当前产品已把
`AllocatorRef/Arena/FixedArena/PooledArena/StackArena/FixedHeap/ConcurrentBlockPool/BlockPoolCache/
GrowableObjectPool/LargeMapping` 固定到上表的借用、arena、heap、block-pool、object-pool 与
mapping 心智；旧类型名已经删除，不留 deprecated alias。

## 11. 收口定义与重开边界

本节裁决完整 Mem façade；`mem_reclaim` 仍由独立 factor/card 对自己的算法与性能负责，Mem 聚合轮只证明
用户入口和 source/static/shared 交付，不替它代签。

Mem 进入预发布冻结必须同时满足：

1. host allocator/heap/arena/pool、mapping/residency、开放 resource、Buffer/View/import/
   composite、sync operation、pool/stats 与 lifetime plan 全部有自然的用户路线；
   Compute/Graph/IO 只保留本文明确的 mechanism/policy/completion 语义，不因 benchmark
   参数新增 `fast/light` 类型。
2. 保持 `Native FFI=N/A`；若出现已命名的 foreign-runtime 任务，先按全局准入策略单独立项，
   不能在 mem 修改中顺手恢复旧 C schema。
3. 最终候选通过 Windows/Linux/macOS、OOM/overflow 和适用的 ASan/UBSan/TSan，并验证
   source/static/shared C++23 消费路径。
4. throughput-oriented 能力有正常 workload、同合同成熟 peer、throughput 与 whole-operation
   latency；空间型能力有 peak/retention/fragmentation 证据。没有同合同 peer 时只报告绝对成本
   和较弱合同参照，不制造 SOTA 结论。
5. architecture、user guide、performance 与 implementation task 各守自己的文档职责，
   不复制状态表；旧 host-only/Compute-owned Buffer/Graph-owned planner 文字已清理。

五项通过即冻结。性能稳定性必须由多进程重复、paired sign/spread、长寿命 epoch 与结构复杂度
门禁共同建立，不能靠一次很短的 microbenchmark；通过已定义证据后也不无限等待。冻结后只有 correctness/security、真实消费者
暴露能力缺失、新平台使合同失效或既定 harness 确认的持续性能回归可以重开；LOC、命名偏好、
亚纳秒波动和无 workload 的猜测不能重开。

## 12. 验证边界

- allocator：typed count overflow、完整 backing identity、alignment、OOM、PMR throw boundary；下游消费者的 NULL/Status/fail-fast 另按各自公开合同审计。
- resource/Buffer：provider/ref 生命、property/domain identity、alignment/max/OOM、零长度、
  move/reset/import 与 range overflow；native handle 不得冒充 mapped pointer；host direct 与
  non-coherent flush/invalidate 差分；CUDA/Vulkan/host 真实路线各自验证，硬件未提供的
  driver 路径明确 skip，不用 synthetic 冒充硬件 PASS。
- bump/arena：alignment 回绕、owned growth、bulk clear、large block、hard size limit、bytes used/reserved、external storage 无 fallback；多 block 长寿命复用用结构门禁证明每 epoch 至多线性考察，并用 32→2048 blocks 的跨进程 scaling 证书防回归。
- fixed pools：耗尽/回收、remote free、hard quota、soft publish、稳定空快照、tag 饱和、authoritative backing、lifetime/TSan/ASan；FixedHeap 另验证大 alignment/overflow、随机 realloc/碎片回收和与 free-block 数量无关的至多两候选查找。
- logical ranges：zero/overflow/>4 GiB/alignment、metadata OOM 回滚与上限、随机 non-overlap、split/coalesce、owner/lease move、bulk generation reset 和 stale lease；P1 只能比较 peer 也拥有的相同 offset/ownership/alignment 子合同。
- object pools：Reconstruct + move-only Loan、早退/异常自动归还、hard/soft handoff；Preserve 一次构析与状态保留。
- diagnostics：membership fail-fast 保留；`HYLE_TESTING` 已覆盖 shared/cached/object/growable pool 的 double return/destroy、foreign pointer 与 outstanding allocation，shipping 构建完全剥离额外状态和指令；LeakTracker metadata OOM 必须留下 dropped/complete 证据，不能静默伪称完整。
- performance workload 必须描述 request batch、producer/consumer handoff、quota、expensive-context reuse 等用户行为，并扫描线程、工作集、容量和批次；不能围绕内部阈值造场景。

当前工程性能证据从 [status](../implementation/status.md) 的链接进入；未进入当前 P1/P2 合同的固定池
能力只签复杂度、资源、correctness 与交付门，不冒充数值 SOTA。历史候选与性能轮只留在
implementation task/evidence 和 Git 中。

## 参考

- oneTBB scalable allocator / memory_pool / fixed_pool / IndexedBins / `cache_aligned_resource`
- mimalloc local + concurrent free lists / segment span queues
- Folly Arena / ThreadCachedArena
- marl BoundedPool / PoolPolicy / Loan
- TLSF two-level bitmap mapping；Boost.Interprocess `rbtree_best_fit` 作为 O(log N) 备选形状对照
- AMD VMA/D3D12MA virtual allocator；Sebastian Aaltonen OffsetAllocator
- ggml backend allocator/graph plan 与 llama.cpp upload buffer 生命周期
