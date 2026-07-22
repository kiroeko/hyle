# HYLE mem/ 设计

> Tier 0 host-RAM 基础设施。上游约束见 [`hyle-scope.md`](./hyle-scope.md) 与 [`coding-conventions.md`](./coding-conventions.md)。本页同时记录当前实现与需求驱动的产品处置；没有生产消费者不等于自动删除，也不等于自动上架。

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
| `Arena` / `StackAllocator` / `GrowablePool` | 无生产调用 | 当前仅公开能力、C ABI、测试与 benchmark |
| `SharedPool` / `SharedPoolCached` | 无生产实例 | 当前仅公开能力、测试与 benchmark |
| `ObjectPool<T>` / `ObjectPoolCached<T>` | 无生产实例 | 已实现并测试，不是占位符 |
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
| `Allocator` | `{alloc, free, realloc, state}` C vtable | 保留；内部运行时注入脊 |
| `PmrResource` | `Allocator` → `std::pmr::memory_resource` | 保留；构造 assert valid，equality 比较完整三函数 + state |
| typed/RAII helpers | `allocate_n`、construct/destroy、make_unique | 保留；System/运行时两层合同必须一致 |

通用 malloc 借 mimalloc，不自造任意大小通用堆。sized-free 必须带回原请求的 size/alignment；backing identity 与 ownership 不能丢失。

## 5. 固定块池

`SharedPool` 与 `SharedPoolCached` 共用 `detail::ShardedSlab`，但公开合同不同。

### 5.1 `SharedPool`：hard bounded

- 固定 block size + runtime capacity。
- completed `deallocate()` 立即进入 shared core；remote free 可见。
- 两次稳定全-shard 快照消除 block 在扫描背后迁移造成的假耗尽。
- 64-bit packed head；32-bit version 饱和后永久进入该 shard 的 serialized mode，不回绕。
- 用于严格 object/block quota、跨线程 handoff；本地 request churn 不是它的吞吐强项。

### 5.2 `SharedPoolCached`：soft cached

- 每 `(thread,pool)` TLS 128 slots，32-block magazine。
- 同线程命中零原子；其他活线程 TLS 中的 free block 暂不可见。
- `publish_thread_cache()` 是显式可见性边界。
- 只用于池容量高于并发 working set、同线程复用为主的场景；不得宣传为 hard capacity。

两者保留。默认 shipping 热路径不增加 per-slot bitmap/atomic live counter；foreign/double free、outstanding object 属调用方契约违反。但这些限制必须写入公开 API，并提供 opt-in diagnostics 构建，用 per-slot state/live count fail-fast，不能让默认检查破坏 cached 零原子合同。

`SharedPool` 始终服从 injected backing。需要 THP/NUMA 的调用方应显式提供相应 allocator/provider；pool 不根据 allocator identity 偷换 storage domain。

## 6. typed object pool

`ObjectPool<T>` 与 `ObjectPoolCached<T>` 已实现：placement-new、显式析构、构造异常退槽、OOM/耗尽返回 NULL；后者暴露 publish。当前策略固定为每次 create/destroy 都 Reconstruct。

产品处置：保留二者的 hard/soft typed shell，并补齐独立的生命周期能力：

1. move-only `PoolLoan<T>` 已实现，作用域退出自动 destroy/return，减少异常/早退漏还；Pool 必须比 Loan 长寿。
2. 当前手工 `create/destroy` 保持 `Reconstruct` 合同：borrow 时构造、return 时析构。pool 析构前必须已归还并析构全部 live object；这是调用方合同，不得暗藏。
3. `PreservingObjectPool<T>` 已作为独立类型实现：池构造时一次性构造全部对象，return 保留状态，池析构时一次性析构；不与 TLS Cached 混名。
4. 默认 API 仍是 nonblocking try；需要“等到有对象”的调用方在 sched 层用 Semaphore/Channel 包装。
5. 当前 shipping 已对 foreign/misaligned pointer 做 membership fail-fast；P1 diagnostics 只补 per-slot liveness、double return 与 outstanding loan，shipping 默认不加原子计数税。

`ObjectPoolCached<T>` 不是待实现项，也不是 Preserve object cache；“cached”只表示 free slot 的 TLS 可见性。Preserve 必须由独立 policy/name 表达。

## 7. region、stack 与增长池

### 7.1 `PooledRegion` 与兼容名 `Arena`

`PooledRegion` 只借 `SharedPool` 的一个 hard block，固定容量、单线程、bulk reset。旧 `Arena` 保留为 source-compatible alias，新增代码使用精确名称。

### 7.2 owned `Region` 与 external `FixedRegion`

owned growable `Region` 已提供：

- 按 min block size 增长；
- `clear()` 复用普通 block，大型专用 block 可释放；
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
2. `FixedPool`：调用方借入 storage、size-bin segregated-fit variable-size allocate/free/realloc、boundary metadata O(1) 邻块合并；单线程且无 fallback。

## 9. NUMA、大页与专用模块

- `HugeRegion`、显式 NUMA bind/interleave、THP/prefault 保留；策略失败是可观察的降级或错误，不保证自动 NUMA 最优。
- `FiberStackPool` 保留且是当前内部刚需。
- graph 当前可直接消费通用 owned Region；`GraphArena::measure/reserve/allocate` 这类 graph-aware worst-case/liveness 策略等到对应消费者落地，归 `graph/`。
- `Buffer(kHostPinned)` 的 native-handle/mapped-pointer 合同先在 P0 纠正；`StagingRing` 等到至少两个异步传输在途、slot 复用受 Event 约束时再落地，归 `compute/`。
- io provided-buffer ring 归 `io/`。
- generic `MemorySpace`、把 unified/device memory 塞进 mem allocator、自动 NUMA 迁移、soft limit 冒充 hard quota 均否决。

## 10. 能力矩阵与优先级

| 能力 | 处置 | 原因 |
|---|---|---|
| `SystemAllocator` / `Allocator` / typed helpers | 保留；typed count 回绕已修 | 内部广泛真实消费者，成熟库共同基础 |
| `PmrResource` | 保留；validity/full-vtable equality 已修 | std 生态适配边界 |
| `FiberStackPool` | 保留 | sched 真实热路径 |
| `SharedPool` / `SharedPoolCached` | 保留；injected backing 始终 authoritative | hard capacity 与 TLS soft visibility 是不同真实合同 |
| `ObjectPool` / `ObjectPoolCached` | Reconstruct core + `PoolLoan` 已实现 | typed bounded reuse、hard/soft visibility 与 RAII return |
| `PreservingObjectPool` | 已实现独立 Preserve lifetime | 昂贵对象只构析一次，不与 TLS Cached 混淆 |
| `StackAllocator` | 保留；alignment 加法回绕已修 | 常见 LIFO scratch |
| `GrowablePool` | 保留；P1 diagnostics | 常见 single-thread typed reuse；当前无生产消费者 |
| `PooledRegion` / `Arena` alias | 精确命名已完成 | borrowed single hard-pool block |
| owned `Region(size_limit)` | 已实现 | request/frame/graph phase 的 growable bulk lifetime |
| `FixedRegion` / `FixedPool` | 已实现为两个正交 external 类型 | bump bulk reset 与 variable individual free 不混合 |
| `QuotaAllocator` | 已实现 | 任意 sized-free backing 的严格 live-byte budget |
| HugeRegion/NUMA policy | 保留显式能力 | 大数据/HPC 真实但平台受限 |
| compute HostBuffer | native handle / host mapping / coherence 已显式分层 | Vulkan 不再把 `VkMem*` 冒充可写地址 |
| compute `StagingRing` | 至少两个 async transfer 在途的消费者出现后实现 | ownership 由 HostBuffer/Event/backpressure 决定 |
| graph-aware allocator / io buffer ring | 下沉所属模块 | 专用生命周期不属于通用 mem |

2026-07-22 收口执行新增的是成熟、正交的基础形状，不以 benchmark 参数命名：

1. `SystemAllocator` typed count 防回绕；`PmrResource` 构造验证完整 vtable，equality 比较三函数 + state。
2. `StackAllocator` / 当前 `Arena` 通过可合成地址测试的 checked-align helper 防 `uintptr_t` 回绕。
3. `SharedPool` 不再根据 system allocator identity 隐式进入 HugeRegion；THP/NUMA 由显式 backing/provider 表达。
4. compute Buffer 分开 native handle / mapped host pointer；范围 flush/invalidate 按 Vulkan `nonCoherentAtomSize` 扩边；零长度公共 no-op；async Kernel/Buffer 存活和 producer completion 责任显式。错误 pinned allocator 与单块 staging API 已删除。
5. `Region`/`FixedRegion`、`FixedPool`、`QuotaAllocator`、`PoolLoan` 与 Preserve 类型均有独立 ownership/capacity/lifetime 合同，并已接入 C ABI（模板对象 policy 除外）。

## 11. 用户选型

```text
通用 host RAM / 默认容器             → SystemAllocator
运行时注入、FFI、mock                → Allocator
std::pmr 容器                         → PmrResource
request/frame 整批释放                → Region
调用方外部 buffer 的 bulk scratch     → FixedRegion
调用方外部 buffer 的变长 alloc/free   → FixedPool
固定 slab 上 LIFO scratch             → StackAllocator
单线程同尺寸对象、按需增长            → GrowablePool<T>
多线程固定 block/object、严格容量      → SharedPool / ObjectPool<T>
同线程复用热点、容量有余量             → SharedPoolCached / ObjectPoolCached<T>
昂贵对象保留状态                       → PreservingObjectPool
phase 临时区严格 reserved-byte limit  → Region(size_limit)
任意 backing 的严格 live-byte budget  → QuotaAllocator
fiber guarded stack                   → FiberStackPool
NUMA/THP 大块 host RAM                → HugeRegion + 显式 policy
GPU pinned/mapped host RAM             → compute HostBuffer（native handle + host_data + flush/invalidate）
异步 staging                          → compute StagingRing（待真实 async overlap 消费者）
```

## 12. 验证边界

- allocator：typed count overflow、完整 backing identity、alignment、OOM、PMR throw boundary；下游消费者的 NULL/Status/fail-fast 另按各自公开合同审计。
- Buffer host path：native handle 不得冒充 mapped pointer；CUDA/Vulkan/host 三后端都验证 `host_data()` 可写、kernel 参数仍拿正确 native handle。test-only 选择器穷举后确认 Win/Mac 物理设备只提供 coherent host-visible type，non-coherent driver round-trip 明确 skip；范围算术用真实 `nonCoherentAtomSize` synthetic 测试覆盖，不冒充硬件未提供的 driver 路径已执行。
- bump/region：alignment 回绕、owned growth、bulk clear、large block、hard size limit、bytes used/reserved、external storage 无 fallback。
- fixed pools：耗尽/回收、remote free、hard quota、soft publish、稳定空快照、tag 饱和、authoritative backing、lifetime/TSan/ASan。
- object pools：Reconstruct + move-only Loan、早退/异常自动归还、hard/soft handoff；Preserve 一次构析与状态保留。
- diagnostics：当前 membership fail-fast 保留；P1 补 double free、outstanding loan、GrowablePool liveness，shipping 构建验证额外状态完全剥离。
- performance workload 必须描述 request batch、producer/consumer handoff、quota、expensive-context reuse 等用户行为，并扫描线程、工作集、容量和批次；不能围绕内部阈值造场景。

当前固定池实测见 [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)。

## 参考

- oneTBB scalable allocator / memory_pool / fixed_pool
- mimalloc local + concurrent free lists
- Folly Arena / ThreadCachedArena
- marl BoundedPool / PoolPolicy / Loan
- ggml gallocr 与 llama.cpp pinned upload ring
