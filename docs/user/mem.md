# HYLE mem：什么情况选择什么

> 本页描述当前可编译的 C++ 产品面：已收口的 host allocator/pool 机制，以及当前
> provider-neutral Buffer、mapping、transient BufferPool 和 tensor-free lifetime planner 路线；
> Compute 还提供 host、pinned、device 与可选 managed provider。Host/CUDA/Vulkan 的实际支持
> 以所选 `Device` 返回的 resource/property 为准，不把 synthetic provider 验证写成硬件承诺。
> 适用于 C++23 source/static/shared 三种交付；不描述 C API 或跨语言调用层。机械收口状态见
> [`module-status.json`](../implementation/module-status.json)。

mem 是 HYLE 的 storage foundation：host raw allocation 只是其中一层，跨 provider 的
Buffer ownership/view/mapping/reuse/placement 也由它表达。它不替业务猜生命周期，也不把某个 benchmark 的最优
形状包装成通用 allocator；使用者先确定 ownership、并发、容量和释放方式，再选择类型。
本文是 C++ 消费者的选型入口。

## 一页选择表

| 需求 | C++ 形状 | 关键合同 |
|---|---|---|
| 普通动态分配 / 依赖注入 | `system_allocator()` / `AllocatorRef` | sized-free；free/realloc 带回原 size/alignment |
| 独立 host heap，既要 individual free 又要整批作废 | `Heap::create()` 后取 `heap.allocator()` | owner-thread allocate/reallocate；允许 remote free；`clear/release` 使所有 live allocation 失效 |
| 拥有一块可安全切 view 的 host bytes | `Buffer::create(system_resource(), bytes, alignment)` | move-only owner；view 不拥有；读写/填充/复制不隐藏 staging |
| 为 Compute kernel 创建 device/pinned/managed bytes | 从 `compute::device_memory_resource` / `host_memory_resource` / `managed_memory_resource` 取得 provider，再创建 `Buffer` | resource 与 Device identity 必须匹配；managed 可能明确返回 `Unimplemented`；owner 活到 in-flight completion 之后 |
| 把多个独立 allocation/shard 暴露为一段逻辑 bytes | `CompositeBuffer::create(segments)` | 只 composite 支付 segment metadata；不 flatten、不隐藏 staging；输入在成功后才被 move |
| 反复申请短寿命 host/device scratch | `BufferPool::create(resource)` + `acquire()` | thread-safe bounded best-fit；lease 只在 completion 后归还；retention/stats/trim 可见 |
| 已知整张执行图的 storage lifetime | `LifetimePlanner::build(requests, resources)` -> `instantiate(resources)` | tensor-free；先查 resource/backing/placement，再按 provider 单次分配上限建立一个或多个 backing；alias/output facts 显式 |
| 占地址后分段提交 / 释放物理页 | `VirtualMapping::reserve` + `commit/decommit` | `VirtualMapping` 是 Platform `VmRegion` 的零开销别名；页粒度，无隐式 commit、prefault 或 fallback |
| 映射模型权重/数据集 | `FileMapping::open` | RO/RW/COW；checked range/view；advice/lock/flush 显式；POSIX/macOS 可批量释放已知页区间 |
| 命名跨进程共享 bytes | `SharedMapping::create/open(name,size,access)` | payload 保持原生页对齐；size/access 是外部协议；同步由调用方/IO 提供 |
| 接入传统 allocator-template API | 默认 system heap 用 `SystemStdAllocator<T>`；运行时 backing 用 `StdAllocator<T>` | 前者空类型/静态绑定；后者借用 backing；STL OOM 转 `bad_alloc` |
| 接入 STL PMR | 默认 system heap 用 `system_memory_resource()`；运行时 backing 用 `PmrResource`；显式高对齐用 `AlignedPmrResource` | 默认形状单层 PMR dispatch；注入形状借用 `AllocatorRef` |
| 任意 backing 的严格 live-byte admission | `QuotaAllocator` | 线程安全；按 requested bytes 精确计量 |
| opt-in allocation 计数 / 泄漏诊断 | `StatsAllocator::create(backing,track,metadata)` | 被测 backing 与诊断 metadata 分域；计数线程安全，leak inventory 是慢路 |
| 进程级 system heap / RSS 压力快照 | `system_memory_snapshot()` | 冷查询；allocator reserved/committed 与 process RSS；private commit 仅在 OS 可证明时为 optional；不冒充精确 requested-live |
| 极热读、低频替换的不可变 snapshot | `ReclamationDomain` + `ReclamationGuard` + caller-owned `RetireRecord` | 读侧不分配；短 guard；旧对象在先前读者退出后由 collector 线程同步回调 |
| request/frame/graph phase 临时对象 | `Arena` | owned、单线程、growable、bulk `clear()` |
| 调用方 buffer 中的临时对象 | `FixedArena` | external、单线程、bump、无 fallback |
| 从共享硬池借一个 scratch block | `PooledArena` | 单线程、固定容量、整体归还 |
| LIFO checkpoint/rewind scratch | `StackArena` | owned fixed slab、单线程 |
| 外部 buffer 内变长 alloc/free/realloc + 整池 reset | `FixedHeap` | 单线程、O(1) indexed fit/merge、无 fallback；约 4 KiB control；fixed 指 backing 容量固定 |
| 自己增长 backing 的变长 alloc/free + bulk release | `std::pmr::unsynchronized_pool_resource(system_memory_resource())`；并发用 synchronized 版本 | C++ 复用标准池；运行时 upstream 才包 `PmrResource` |
| 多线程固定块、严格容量 | `ConcurrentBlockPool` | hard capacity；free 完成后立即全局可见 |
| worker-local 热点固定块复用 | `BlockPoolCache` over `ConcurrentBlockPool` | 显式 owner/capacity/batch；可见性是 soft，需 publish |
| 固定数量 typed 对象，每次重建 | `ObjectPool<T>` | 每次取得运行 ctor，归还运行 dtor |
| 昂贵对象保留内部状态 | `PreservingObjectPool<T>` | return 不销毁内部状态；pool 析构时统一 cleanup |
| 单线程 typed 对象、容量可增长 | `GrowableObjectPool<T>` | 指针稳定，按需增加 chunks |
| 自动归还 pooled object | `PoolLoan<T, Owner>` / `try_loan()` | owner 必须比 loan/slot 长寿 |
| 大块、THP/NUMA placement | `LargeMapping` | large 表示 owner 尺寸，不把 THP hint 冒充显式 huge-page guarantee；requested 与 accepted policy 可分别查询 |
| fiber stack | `FiberStack` / `FiberStackPool` | guard page + 有界 runtime 缓存；shutdown/trim 可归还 VM |

GPU staging、submission/completion lifetime 属于 `compute/transfer`，不由 mem 用“通用
pinned pool”猜测。Graph 负责把 tensor/DAG 语义转换成 interval、alias 与 resource facts；
不理解 Tensor 的通用 placement/instantiate 机制归 Mem。

Platform 的 `VmRegion` 仍是公开的低层 anonymous-address owner，适合自定义 allocator/JIT 或其它
自建 storage owner；file/shared/anonymous storage mapping 则只有本表中的 Mem owner。已有
Platform `FileHandle` 可直接 move 给 `FileMapping::map`，不做第二次 path lookup、payload allocation
或 copy。普通用户因此不需要在两套 mapping 类型之间选择。

## Read-mostly snapshot：何时使用 reclamation

模型、backend、路由表或运行配置若满足“读非常频繁、更新很少、读者只需一个很短的不可变 view”，可用
`ReclamationDomain` 避免每次读取都修改 `shared_ptr` 引用计数：

```cpp
#include "hyle/mem/reclamation.h"

struct Snapshot {
  ModelRoutes routes;
  hyle::mem::RetireRecord retirement;
};

void delete_snapshot(Snapshot* value) noexcept { delete value; }

hyle::mem::ReclamationDomain domain;
std::atomic<Snapshot*> published{load_initial_snapshot()};

{
  auto guard = domain.read();
  const Snapshot* current = guard.protect(published);
  dispatch(current->routes, request);
}  // guard 必须在同一线程、尽快退出

Snapshot* old = published.exchange(build_snapshot(),
                                   std::memory_order_acq_rel);
domain.retire<&delete_snapshot>(old->retirement, old);
```

普通短读区可共享 `default_reclamation_domain()`；一个长读者不能拖延无关 owner，或 owner 需要可预测
teardown 时，应使用显式 domain。`try_reclaim()` 从不等待，`synchronize()` 等待调用前已经进入的读者，
`drain()` 只可在没有新 guard/retire 进入的外部静止点调用。callback 在执行 collection 的线程同步运行，
所以只能做 `noexcept` 的短释放工作，不能在同一 domain 的 guard 内调用 blocking collection。

它不替代所有生命周期方案：对象更新同样频繁时用 mutex/并发容器；读者需要独立延长对象生命时用
`shared_ptr`；停顿读者绝不能阻塞其它对象回收时需要 hazard-pointer family。domain 必须长寿于所有 guard
和 pending record；record 在 callback 开始前不能销毁或再次 retire，callback 也必须理解对象真正使用的
allocator。`ConcurrentNodeHashMap` 已把这种 table hand-off 封装起来，普通 map 用户无需手写 record。

## Resource 与 Buffer：什么时候不只用 allocator

`AllocatorRef` 回答“从 host backing 取多少 raw bytes”；`MemoryResourceRef` + `Buffer`
回答“这块 storage 属于哪个 provider/domain，谁释放，可以怎样读写和切 view”。
普通小对象和容器仍优先 allocator；在需要 move-only byte owner、稳定 provider identity、
opaque native handle 或跨 host/device provider 路由时使用 Buffer。

```cpp
#include "hyle/mem/buffer.h"

auto result = hyle::mem::Buffer::create(
    hyle::mem::system_resource(), 4096, 64);
if (!result) return result.error();
hyle::mem::Buffer storage = std::move(*result);

auto header = storage.view().subview(0, 64);
if (!header) return header.error();
HYLE_RETURN_IF_ERROR(header->fill(std::byte{0}));
```

`MemoryResourceRef` 是一字、不拥有的 provider borrow；provider/backend/device state
必须比 ref、Buffer、view 和所有 in-flight use 长寿。`Buffer` 和 `BufferView` 都是四字
inline 形状；view 保存 provider + opaque allocation + offset/size，所以 owner move 不会让
view 失效，owner reset/析构会。`native_handle()` 只能由已检查 resource/domain 的 typed
provider adapter 解释，它不是通用 host pointer，不能解引用或做 pointer arithmetic。

`subview` 越界返 `OutOfRange`。view 上的 `read/write/copy` 要求两端 span/range 等长；
要操作部分数据先取 subview。`map_host()` 返回 move-only `HostMapping`；只读 mapping
不会通过 `writable_bytes()` 泄漏可写 span。对 non-coherent provider，读 device 写入前显式
`invalidate()`，host 写入后显式 `flush()`；`system_resource()` 是 coherent host route，两者均
为 no-op。跨 resource copy 只走 provider native route 或两端可 host-map 的直接路线，不会暗中
分配 staging buffer。

`system_resource()` 返回 HYLE generic Buffer resource；`system_memory_resource()` 返回
`std::pmr::memory_resource*`。两者故意不是同一类型：前者承载 Buffer/provider 语义，后者只
适配标准 PMR 容器。

设备 storage 由 Compute 提供 provider，但 ownership/view 仍只用 Mem 类型：

```cpp
#include "hyle/compute/memory.h"
#include "hyle/mem/buffer.h"

HYLE_ASSIGN_OR_RETURN(auto resource,
                      hyle::compute::device_memory_resource(device));
HYLE_ASSIGN_OR_RETURN(auto weights,
                      hyle::mem::Buffer::create(resource, weight_bytes));
HYLE_ASSIGN_OR_RETURN(auto layer,
                      weights.view().subview(layer_offset, layer_bytes));
HYLE_RETURN_IF_ERROR(hyle::compute::launch(stream, kernel, config, layer));
// weights 必须活到 stream/event 表示该次使用完成以后。
```

`host_memory_resource(device)` 表示可被该 backend 使用的 host/pinned storage；
`managed_memory_resource(device)` 在 backend 没有统一内存时返回 `Unimplemented`，不会偷偷退回普通
host RAM。`memory_resource_belongs_to(resource, device)` 可在自定义 adapter 边界检查 identity。
typed kernel shell 直接接收 `Buffer`、`BufferView` 或 `ConstBufferView`，并原样保留 view 的
native handle、offset 与 size；不会额外分配、复制或把 subview 扩成整个 allocation。

模型分片、超过单次 provider allocation 上限的 storage，或必须保留每段 provider identity 时，
使用 `CompositeBuffer`，不要先拼成一个临时连续 buffer：

```cpp
std::array<hyle::mem::Buffer, 2> shards{
    hyle::mem::Buffer::create(resource_a, bytes_a, alignment).value(),
    hyle::mem::Buffer::create(resource_b, bytes_b, alignment).value(),
};
HYLE_ASSIGN_OR_RETURN(auto model,
                      hyle::mem::CompositeBuffer::create(shards));
HYLE_ASSIGN_OR_RETURN(auto window, model.view().subview(offset, length));
for (std::size_t i = 0; i < window.piece_count(); ++i) {
  HYLE_ASSIGN_OR_RETURN(auto physical_piece, window.piece(i));
  consume(physical_piece);  // 保留原 resource/native handle 和精确 clipped range
}
```

每个输入 segment 必须有效且非空；空 segment span 会生成一个有效的空 composite。factory 先验证并
分配 metadata，只有成功时才 move 输入，因此失败不会半搬走 owner。`subview()` 是 O(1)，
`piece_count()`/`piece()` 是 O(log S)，整段 read/write/fill/copy 是
O(log S + touched segments)，不会 flatten 或分配 staging。普通 `Buffer` 的大小和热路径完全不受
segment 支持影响。

Composite view 与普通 view 一样不拥有 storage：owner move 后仍有效，owner reset/析构后失效。
同一 composite 内重叠 copy 会拒绝并要求调用者显式 staging；跨 composite 的底层别名无法通用探测，
仍遵守 provider 的 overlap 合同。跨多个 provider 的 range operation 逐 piece 执行；若后续 provider
失败，前面已完成的 piece 不会回滚，因此需要事务语义的调用方必须自行预检或 staging。

短寿命 buffer 反复创建/销毁时再加 `BufferPool`，不要让每个算子自己保留一组匿名缓存：

```cpp
HYLE_ASSIGN_OR_RETURN(
    auto pool,
    hyle::mem::BufferPool::create(device_resource, {
        .max_cached_bytes = 512u << 20,
        .max_cached_buffers = 128,
    }));
HYLE_ASSIGN_OR_RETURN(auto scratch, pool.acquire(bytes, alignment));
submit_kernel(scratch.view());
wait_for_completion();  // completion 之前绝不能 reset/析构 lease
scratch.reset();
```

默认策略按 size bucket 做有界 best-fit，并给新分配保留 5% headroom；它是 inference backend
已有实践的通用形状，不是 benchmark 档位。`stats()` 区分 live/cached bytes、provider allocation、reuse
和 OOM trim retry；`trim()` 只释放 cached buffer，不等待或取消 in-flight operation。需要把 allocation
永久带走时用 `lease.take()`。

## Lifetime planner：已知整张执行图时先规划再分配

`BufferPool` 适合运行时逐次 acquire/return；如果 consumer 已经知道各 temporary 的完整
生命周期，应把业务语义擦除成 `LifetimeRequest`，把将要使用的 provider 作为能力事实传给
planner，一次得到可查询的 placement、每个 resource 的总 peak 和每个 physical backing 计划，再实例化：

```cpp
#include <array>
#include "hyle/mem/lifetime_plan.h"

std::array requests{
    hyle::mem::LifetimeRequest{
        .resource_slot = 0, .size = q_bytes, .alignment = 64,
        .begin = 0, .end = 3},
    hyle::mem::LifetimeRequest{
        .resource_slot = 0, .size = logits_bytes, .alignment = 64,
        .begin = 3, .end = 5,
        .flags = hyle::mem::LifetimeFlag::kOutput},
};
std::array providers{device_resource};
HYLE_ASSIGN_OR_RETURN(
    auto plan,
    hyle::mem::LifetimePlanner::build(requests, providers));
HYLE_ASSIGN_OR_RETURN(auto resource_plan, plan.resource_plan(0));
report_peak(resource_plan.peak_bytes);
for (std::size_t backing = 0; backing < resource_plan.backing_count;
     ++backing) {
  HYLE_ASSIGN_OR_RETURN(auto physical, plan.backing_plan(0, backing));
  reserve_or_report(physical.size, physical.alignment);
}

HYLE_ASSIGN_OR_RETURN(auto storage, plan.instantiate(providers));
submit_q(storage.view(0).value());
```

request ID 就是输入 span 的稠密下标；resource slot 也必须从 0 稠密排列。interval 是
`[begin,end)`，所以 `end == next.begin` 可以复用。`alias_of` 可指向任意 request，planner 会拒绝
cycle、跨 resource alias、overflow 和无法满足的 alias alignment；`alias_offset` 是相对父 alias 的
偏移。`kPersistent`、`kOutput` 和 `kNoReuse` 都表示该 placement 在本次 interval 后不再回收给
后续 request，但它仍可安全继承更早已经结束的 storage。

build 只在调用期间借用 provider 并读取 `minimum_alignment` / `maximum_allocation`，plan 不持有
provider，也不认识 Tensor/Graph。它用活动 interval heap 与 indexed best-fit slot 结构，复杂度
O((N+E) log N)、scratch/metadata O(N)，不会按 byte/page 建表。新 physical slot 按确定性的
next-fit backing 边界打包；单个 alias root 永不跨 backing，每个 backing 都不超过 provider 的
单次分配上限。`peak_bytes` 是该 resource 所有 backing 的精确计划大小之和，不宣称全局最少
backing 或最优 bin packing。

`MemoryPlacement::backing_index` 是 resource-local 索引，`offset` 只相对此 backing；不要把多个
opaque native handle 当成连续虚拟地址。`instantiate` 要求 provider span 与 resource slot 一一对应，
但可以使用同一个或语义等价的 provider：更大的单次分配上限和更强 minimum alignment 合法，较小
上限会在任何 backing 分配前返回 `OutOfRange`。实例化中途 OOM 会回滚已经创建的全部 backing。
provider 必须长寿于 `PlannedBuffers`、views 和 in-flight operation；plan 销毁后 owner 仍可独立使用。
零大小、但被显式选择的 resource 会保留一个 size-0 backing，从而使零长 view 仍带有 provider identity。

## Mapping：模型加载、稀疏提交和跨进程共享

`VirtualMapping` 是 `platform::VmRegion` 的零开销别名：`reserve()` 只占地址，随后显式
`commit/decommit/protect`；`allocate_rw()` 才是一步得到可写页。页级 advice 和 lock 使用
`platform::vm_advise/vm_lock/vm_unlock`。Mem 不复制类型、enum 或 OS decision authority；这些调用
不隐藏逐页表、payload copy 或后台线程，不受支持的机制返回 `Unimplemented`。

```cpp
#include "hyle/mem/mapping.h"

HYLE_ASSIGN_OR_RETURN(
    auto weights,
    hyle::mem::FileMapping::open("model.gguf"));
HYLE_RETURN_IF_ERROR(
    weights.advise(hyle::mem::MappingAdvice::kSequential));
// prefetch 是异步/提示语义；启动阶段必须完成 warm-up 时显式同步触页。
HYLE_RETURN_IF_ERROR(weights.populate());
HYLE_ASSIGN_OR_RETURN(auto model_bytes, weights.view());
// owner 必须长寿于 view；释放与 view 重叠的页之前先让该 view 失效。
```

`FileMapping` 不会自动触碰或锁定全文件，也不会在 RW open 时扩展文件。只读路线返回
`ConstBufferView`；修改文件必须先通过 `writable_bytes()`，持久化再显式调用
`FileMapping::flush()`。`BufferView::flush()` 表达 host/device coherence，不等价于文件 durability。
`prefetch()`/`MappingAdvice::kWillNeed` 是 OS hint，成功不保证页面已经 resident；`populate()` 则在
返回前同步触碰请求范围覆盖的每一页，成本与覆盖页数成正比。`open/map` 的 `length == 0`
表示映射到当前 EOF；已经建立 mapping 后的 `bytes/view/advise/populate/flush` 默认 length 是
`std::dynamic_extent`（直到 mapping 末尾），显式传 `0` 是空操作/空 view。并发 truncate 或替换
backing 不在合同内，loader 必须先稳定文件生命周期。

residency 是更强且可能受 quota 限制的合同。`lock()` 锁整段；loader 逐步确认 CPU-resident 前缀时
用 monotonic `lock_prefix()`，失败后 `locked_size()` 仍报告此前成功覆盖。默认
`ResidencyLimitPolicy::kPreserve` 不修改进程级 limit；只有应用明确接受该副作用时才选择
`kAdjustIfNeeded`。`ResidencyMode::kOnFault` 仅 Linux 有真实对等机制，其它平台返回
`Unimplemented`，不会静默变成 eager。

POSIX/macOS 的模型加载器可用 `release_range`，已知一批 range 时应优先
`release_ranges(span<MappingRange>)`。它向内取整，只释放完全落在请求内且仍存活的原生页；重复、重叠
或只覆盖半页的输入是安全的，返回实际释放的 logical bytes。连续 mapping 不分配 fragment metadata，
第一次有效 release 才建立片段表；批量路线的 bookkeeping 为 `O(R log R + F)`，避免逐个释放时反复
重建片段列表。片段表按地址有序，checked access 为 `O(log F)`，局部 advice/populate/flush 为
`O(log F + K)`（`K` 是相交片段数），不会为每个 tensor 线扫整张 survivor 表。Windows 的
whole-view primitive 没有同合同的安全 partial release；有效整页请求在 `O(R)` 校验后直接返回
`Unimplemented` 且 owner 不变，不分配 fragment metadata，调用方可改用多个精确 subrange owner。

release 后 `size()` 仍是稳定 logical extent，`mapped_size()` 是仍存活的 logical bytes，
`fragment_count()` 是 native survivor 数。`address/bytes/view` 只在所请求范围完全落于一个 live
fragment 时成功；跨 hole 返回 `FailedPrecondition`。先前取得且与 release 重叠的 pointer/span/view
立即失效，不能靠 checked accessor 追踪借用。advice/populate/flush 会对请求与所有 live fragment 的
交集执行；完全落在 hole 中则失败。锁定前缀与 release 互斥：先 `unlock()` 再释放其覆盖页。
Windows 的 `DiscardVirtualMemory` 会让 PAGE_READWRITE 内容未定义，并不是 file-backed cache eviction；
因此 `FileMapping::advise(kDontNeed)` 在 Windows 对 RO/RW/COW 都明确返回 `Unimplemented`。Linux/macOS
才沿用各自 file-backed `madvise` 语义，不能跨平台把 `kDontNeed` 理解为同一 persistence 保证。

`SharedMapping` 只拥有命名共享 storage。创建者得到 RW owner；打开者必须显式传相同 payload size，
默认是 RO，也可请求 RW。实现不放私有 header，因此 payload 从 OS mapping base 开始并保持页对齐；
代价是 size/readiness 必须属于外部 IPC protocol。HYLE 会用 OS 报告的 native capacity 拒绝超界：
Linux shm 保留精确 `ftruncate` size；Darwin shm 与 Windows section capacity 按 native page boundary
向上取整（Windows allocation granularity 约束的是 view offset/base，不是这里的 capacity）。后两者
无法恢复精确 logical size，所以错误的、但仍落在同一 page rounding 区间内的较大 size 可能映射成功。
双方必须先协商 size，并在读取前用自己的
ready/version 字段或同步原语发布初始化完成。它不内置 mutex、notification、crash recovery 或跨进程
指针修复；共享数据结构保存 offset/POD，进程间同步和 handle-transfer protocol 由 IO/调用方组合。
名称直接进入 native namespace：HYLE 只为 POSIX 补前导 `/`，不会截断或哈希。为了让同一协议跨
Linux/macOS/Windows 工作，应使用短 basename；interior `/` 或超过本机 POSIX shm 上限会返回结构化
`InvalidArgument`，不会创建一个“近似名称”的对象。

进程启动期若要给 system allocator 显式预留 pinned large-page arena，包含
`hyle/mem/large_page_reservation.h` 并调用 `reserve_large_page_arena(gib, ...)`。`gib` 是 arena
容量的 GiB 请求单位，**不是**物理页粒度承诺。mimalloc 会先尝试 1 GiB huge pages，但可回退到
2 MiB/native large pages；它的公开 API 不能区分最终粒度，因此 report 的
`page_granularity` 固定为 `kUnverified`。入口返回
`Result<LargePageArenaReservationReport>`：参数错误发生在调用 vendor 以前；开始尝试后，即使只保留
部分容量也返回 report，调用方必须同时看 `reserved_gib`、`outcome` 和 `complete()`。成功部分会立即
发布为 process-global arena，无法 rollback；NUMA node 也只是不可验证的 placement preference。
不要在请求热路调用，更不能把 arena request unit 写成“已获得 1 GiB OS pages”。

## 通用 allocator、typed array、标准 adapter 与 PMR

`AllocatorRef` 是 HYLE 内部和 C++ 用户的按值注入句柄；模块之间直接传这个 C++ 类型，不绕经
foreign ABI adapter。`allocate_n<T>` / `allocate_zeroed<T>` / `reallocate_n<T>` 会检查 count
乘法溢出，
并允许显式 alignment；用同一个 alignment 调 `deallocate_n`。`reallocate_n` 失败或溢出时
旧 allocation 仍然有效；因为底层 resize 可能按字节搬移，它只接受 trivially-copyable `T`，
带构析/自引用状态的对象应新建 storage 后逐对象 move/销毁。
`AllocatorRef{}` 是刻意保留的 **invalid sentinel**，只能先检查 `valid()`、赋入一个有效 backing 或直接
销毁；不能对它调用 allocate/deallocate/reallocate。普通默认 backing 应显式取
`system_allocator()`。句柄还借用 callback state，所有经它取得的 allocation 都必须在该 state
失效前归还。运行时注入会支付一次间接分派；不需要注入时，热点代码优先用
`SystemAllocator`、`SystemStdAllocator<T>` 或 `system_memory_resource()` 的直接形状。
原生 C++ allocate/deallocate/reallocate 的 alignment 必须是非零 2 次幂；allocate/reallocate 的
非法值会 fail-fast。deallocate 还必须带回成功分配时相同的 size/alignment，和标准 allocator/PMR
一样属于调用方前置条件：Debug 会诊断，Release 热路不重复检查。未来 binding wrapper 若接收
不可信外部参数，必须在进入这条 C++ 热路前完成校验，不能依赖未来某个 foreign-runtime
binding 的偶然形状。

```cpp
hyle::mem::AllocatorRef alloc = hyle::mem::system_allocator();
auto* samples = alloc.allocate_zeroed<float>(1024, 64);
if (samples == nullptr) return resource_exhausted;
if (auto* grown = alloc.reallocate_n(samples, 1024, 4096, 64)) {
  samples = grown;
  alloc.deallocate_n(samples, 4096, 64);
} else {
  alloc.deallocate_n(samples, 1024, 64);  // realloc failure preserved it
}
```

第三方或标准容器把 allocator 类型写进模板参数、无法接收 `memory_resource*` 时，普通 system
heap 优先用空类型、always-equal 的 `SystemStdAllocator<T>`；只有确实要在运行时注入 quota、stats、
region 或用户 backing 时才用 `StdAllocator<T>`。后者按完整 `{alloc,free,realloc,state}` identity
判断相等并借用 backing；adapter、容器与 `allocate_shared` control block 都必须先于 backing
销毁。`StdAllocator<T>` 有意没有默认构造：运行时 backing 必须写出来；默认构造容器若只需要
system heap，应把 allocator 类型写成 `SystemStdAllocator<T>`。两者在标准库边界按要求把 OOM
转为异常，HYLE 内部仍使用 null/`Result` 合同。
adapter 使用普通 `T*`（不提供 fancy pointer）；`allocate(0)` 可返回 `nullptr`，该返回值仍可用
`deallocate(pointer, 0)` 配对。C++23 下 `!=` 由跨类型 `operator==` 的 rewritten candidate 提供。

```cpp
#include "hyle/mem/std_allocator.h"

hyle::mem::StatsAllocator stats =
    hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
{
  hyle::mem::StdAllocator<Node> alloc(stats.get());
  std::vector<Node, hyle::mem::StdAllocator<Node>> nodes(alloc);
  auto root = std::allocate_shared<Node>(alloc, initial_value);
  // root 和 nodes 都在 stats 前销毁。
}
```

普通 system-backed PMR 容器直接传 `system_memory_resource()`；它是进程寿命 singleton，
不需要用户管理 resource 对象，也不会在 PMR 虚调用后再叠一层 `AllocatorRef` 函数指针。只有
运行时 backing 注入才构造 `PmrResource(alloc)`。若 SIMD/cache/page-sensitive 容器明确需要更高的**底层
buffer 起始地址**，使用 `AlignedPmrResource(alloc, alignment)`；这对应 oneTBB
`cache_aligned_resource` 的需求形状，但 alignment 由用户选择，不把平台 cache policy 塞进
所有容器。三个 resource 对合法的零字节 PMR 请求都会返回可配对 `deallocate(...,0,...)` 的
有效 allocation；内部只取 1B backing，不会把底层 allocator 的“size 0 返回 NULL”合同误翻译成
`bad_alloc`。它不会给相邻元素自动加 padding；防 false sharing 仍需 padded/aligned element。

```cpp
hyle::mem::SystemStdAllocator<Node> system_nodes;
std::vector<Node, hyle::mem::SystemStdAllocator<Node>> fast_nodes(system_nodes);
std::pmr::vector<Node> pmr_nodes(hyle::mem::system_memory_resource());

hyle::mem::AlignedPmrResource cache_buffer(
    hyle::mem::system_allocator(), 64);
std::pmr::vector<std::byte> bytes(&cache_buffer);
bytes.resize(4096);  // bytes.data() 至少 64-byte aligned
```

HYLE **并不缺可变大小内存**：通用 allocator 接受任意请求大小，Arena 接受变长请求并 bulk
clear/release，`FixedHeap` 则在调用方固定容量 buffer 内提供变长 allocate/free/reallocate + bulk
`reset()`。这里唯一没有单独上架的，是“由 HYLE 自己拥有并自动增长 backing”的又一个通用 heap。

需要独立 owned growable、individual free + 一次 bulk invalidation 时，直接使用
`Heap::create()`；它仍可把 `heap.allocator()` 注入 HYLE 容器。需要的是按 size class 缓存大量
反复分配、且消费方接受 PMR 时，再组合标准 pool resource。默认 system upstream 直接使用静态
singleton：

```cpp
std::pmr::unsynchronized_pool_resource local_pool(
    hyle::mem::system_memory_resource()); // 单 owner
std::pmr::vector<Record> records(&local_pool);
// records 先销毁；local_pool.release() 可提前归还全部 pooled blocks。
```

跨线程并发分配改用 `std::pmr::synchronized_pool_resource`。只有需要 quota、stats、region 或
用户 backing 时才创建 `PmrResource upstream(alloc)` 并把 `&upstream` 传给标准 pool；此时 pool
及其所有 allocation 不能比 upstream/backing 长寿。

## Arena 家族

### Owned request scratch

```cpp
#include "hyle/mem/arena.h"

hyle::mem::ArenaOptions options;
options.initial_block_size = 64 * 1024;
options.max_block_size = 4 * 1024 * 1024;
options.large_allocation_threshold = 256 * 1024;
options.size_limit = 8 * 1024 * 1024;  // reserved bytes，包含 header/padding

hyle::mem::Arena scratch(hyle::mem::system_allocator(), options);
for (;;) {
  auto* nodes = static_cast<Node*>(
      scratch.allocate(sizeof(Node) * node_count, alignof(Node)));
  if (nodes == nullptr) break;          // quota/backing OOM 可恢复
  run_one_request(nodes);
  scratch.clear();                      // 普通 blocks 保留给下一次 request
}
scratch.release();                      // 确定归还全部 backing capacity
```

`bytes_used()` 是成功请求的 payload 总数；`reserved()` 是当前 backing bytes；
`peak_used()` 在当前 epoch 中也精确，并且不因 clear/release 清零（实现把已完成 epoch 的峰值
延迟到边界折叠，避免每次 allocation 更新 peak）。individual deallocate 是 no-op。大于 threshold 的
dedicated block 在 `clear()` 时归还，普通 block 保留复用。`initial_block_size` / `max_block_size`
是 ordinary block 的总 backing-byte target，已经包含 Arena header 与最坏 alignment padding；
单个请求放不下时允许为该 block 突破 target。这样 4 MiB target 不会只因内部 header 变成
`4 MiB + metadata`，`size_limit` 与 allocator size class 也更可预测。

长寿命、多 block Arena 在每次 `clear()` 后用单调 cursor 依次复用 retained blocks；一个
epoch 内每个 block 至多考察一次，所以总过渡 O(blocks)、分摊到 block transition 为 O(1)。
某个 block 的尾部不能满足当前请求后，本 epoch 不再回头尝试更小请求；这是 bounded bump
arena 用少量 tail slack 换取可预测复杂度的合同。若小请求尺寸高度交错且必须回收每个洞，
应选 `FixedHeap`，而不是把 Arena 当 variable-size heap。

### External workspace

```cpp
alignas(64) std::byte storage[64 * 1024];
hyle::mem::FixedArena scratch(storage, sizeof(storage));
void* header = scratch.allocate(96, 32);
void* table = scratch.allocate(4096, 64);
if (header == nullptr || table == nullptr) return workspace_too_small;
scratch.clear();
```

`FixedArena` 不拥有 storage、不增长、不 fallback；storage 必须比所有切片长寿。它不可
copy/move，避免两个 cursor 无意间在同一 buffer 上重叠分配。

`PooledArena` 语义同样是固定容量 bump，但 storage 来自一个 hard `ConcurrentBlockPool` block；统一使用 `clear()`。
析构时整块归还。`StackArena` 增加 mark/rewind，适合嵌套 LIFO 临时区。

## Pool 家族

### Hard core 与显式 cache 不能混称

- `ConcurrentBlockPool` 是共享 admission primitive。容量 N 就意味着最多 N 个 live block；任何线程
  完成 deallocate 后，其他线程能立即取得该 block。
  默认共享 shard 数按当前进程可用 CPU（Linux 为调用线程可继承的 affinity）取 2 的幂，
  而不是按容器外的整机核数浪费 metadata；仍受 128 与 block count 上限约束。
- `BlockPoolCache` 是附着到 `ConcurrentBlockPool` 的单 owner L1。storage、容量、transfer batch 和
  生命周期都由用户显式选择；本地 free slot 只有在 `publish()`/析构后才回到共享 core。

`ConcurrentBlockPool::reset()` 是另一条 quiescent bulk-invalidation 路径：先停止所有 core/cache 操作，
再对每个仍附着的 cache 调 `publish()`（或销毁 cache），最后 reset core。否则 cache 中仍保存的
局部 free index 会与 reset 后重新可见的 slot 重复。reset 后任何旧 block 都不能再访问或归还。

typed `ObjectPoolCache<T>` 同样附着到 `ObjectPool<T>`，只增加 placement-new/显式析构的
typed shell。Cache 指 slot visibility，不代表对象状态被保留。

```cpp
hyle::mem::ConcurrentBlockPool core(hyle::mem::system_allocator(), 64, 4096);
if (!core.valid()) { /* propagate slab/metadata OOM before use */ }
std::array<std::uint32_t, 128> local_slots{};
hyle::mem::BlockPoolCache local(core, local_slots, 32);
if (!local.valid()) { /* reject unusable cache before use */ }
void* p = local.allocate();
if (p != nullptr) local.deallocate(p);
local.publish();  // handoff / idle / owner transfer boundary
```

### Reconstruct 与 Preserve

```cpp
hyle::mem::PreservingObjectPool<CodecContext> contexts(
    hyle::mem::system_allocator(), 32, dictionary);
if (!contexts.valid()) { /* propagate construction OOM */ }

if (auto context = contexts.try_loan()) {
  context->reset_stream();
  context->compress(input, output);
}  // 自动 return；CodecContext 不析构，内部 table 可继续复用
```

普通 `ObjectPool<T>::try_loan(args...)` 在每次取得时构造、归还时析构。Preserving pool 在
池构造/析构时各执行一次对象构造/析构。两者都要求 pool 比所有 loan 长寿。

### External variable-size pool

`FixedHeap` 已经是 **variable-size + individual free/reallocate + bulk reset**；名称中的 fixed
只指调用方提供的 payload backing 地址和容量固定，不是只能分配固定大小对象。

```cpp
alignas(64) std::array<std::byte, 64 * 1024> storage{};
hyle::mem::FixedHeap pool(storage);
void* header = pool.allocate(96, 32);
void* payload = pool.allocate(1536, 64);  // 同一池可同时服务不同大小
if (payload != nullptr) {
  if (void* grown = pool.reallocate(payload, 1536, 4096, 64)) payload = grown;
}
// 任一返回 nullptr 都表示 bounded capacity exhausted；reallocate 失败保留旧 allocation。
// 路径 A：逐块归还，指针随各自 deallocate 失效。
pool.deallocate(header);
pool.deallocate(payload);

// 路径 B：在 phase 边界直接 reset；它会批量使所有尚存旧指针失效，
// 不要求先逐块 deallocate。二者不要对同一指针重复使用。
pool.reset();
```

`FixedHeap` 把 block/allocation metadata 放进调用方 storage；two-level segregated free lists、
两层 bitmap、边界和统计属于 `FixedHeap` 控制对象，不占用 payload backing。一级按数量级定位，
二级把每个范围
分成 8 段；分配未命中 recent-free 快路后直接取得一个满足大小的链表头，不扫描同级空闲块。
因此 allocate/free/邻块合并的索引工作都是 O(1)，不会因长期碎片化退化。它支持变长
allocate/deallocate/reallocate、精确 requested `used/peak`；realloc 在当前物理 span 足够或后一
邻块空闲时 O(1) 原地 grow，shrink 保留 span 供后续 O(1) regrow；只有搬移时复制
O(min(old,new))。没有隐藏 payload heap，也不会 fallback。`reset()` 是 quiescent bulk
invalidation：调用前必须停止并发访问；调用后所有旧指针都失效，既不能再访问，也不能再
`deallocate`。它不要求先逐块 free，否则就失去了 bulk-release 的意义。

这个 bounded lookup 需要约 4 KiB 的显式 control。把它放进 arena/cache/device owner 等
长寿命对象，或通过 control allocator 创建一次并复用；不要为每个 request 在小栈帧上构造。
如果 storage 本来就很小、生命周期按 phase 结束，`FixedArena` 的三字 bump cursor 更合适。

它偏向“可诊断、低碎片、固定 backing”的基础设施合同，不应替代：

- request scratch（用 `FixedArena`，更快、control 更小且没有 individual-free metadata）；
- 同尺寸对象池（用 `ConcurrentBlockPool` / `ObjectPool`）；
- 通用 growable malloc（用 system allocator）。

## Quota、统计与失败

长寿推理进程若只需要回答“system allocator 预留/提交了多少地址空间、整个进程的 RSS/private commit
是多少”，使用冷路径 `system_memory_snapshot()`：

```cpp
auto observed = hyle::mem::system_memory_snapshot();
if (!observed) return observed.error();
const auto& memory = *observed;
report(memory.allocator_committed_bytes,
       memory.allocator_peak_committed_bytes,
       memory.process_resident_bytes,
       memory.process_peak_resident_bytes);
```

这是进程级时间切片：并发分配时 current 字段不保证来自同一个原子瞬间；process 字段还包括
非 HYLE allocation domain。`process_private_committed_bytes` 及其 peak 是 optional：OS 没有公开
准确概念时保持空值，不能拿某个 allocator committed、virtual size 或 host commit 代填。它刻意不
报告精确 requested-live，因为在默认分配热路维护该数会
增加常态计数成本。需要 owner-local 精确 live bytes 时，显式给该 owner 套
`StatsAllocator`。

```cpp
hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                64 * 1024 * 1024);
hyle::mem::AllocatorRef bounded = quota.get();
void* p = bounded.allocate(bytes, alignment);
if (p == nullptr) return overload;
// quota.used(), quota.peak(), quota.remaining()
bounded.deallocate(p, bytes, alignment);
```

Quota 在进入 backing 前原子 reserve，因此并发成功调用不会突破 limit；backing OOM 会回滚
reservation。它计 requested live bytes，不计 allocator usable size/metadata。返回的
`AllocatorRef` 借用 Quota 对象。

资源耗尽返回 null；invalid alignment、foreign/double return、错误 sized-free、销毁时仍有
loan 等调用合同错误允许 fail-fast。`StatsAllocator`/`LeakTracker` 可以与 Quota 叠加，
但 wrapper 顺序决定各层看到 requested attempt 还是 backing success。它们的冷构造返回
`Result`，不会从 `new` 抛出 OOM；Stats 的第三个参数是独立 metadata allocator，默认 system。
被测 backing 即使是 fixed-block/strict-quota，诊断 control/hash 也不会偷占其容量。

启用 leak map 时，metadata OOM 不应让真正的用户 allocation 失败。此时 allocation counters
仍精确，`leaks()->dropped_records()` 增加且 `complete()==false`；已知 inventory 仍可读取，
但不能宣称它穷尽了全部 leak。确认没有 live allocation 后 `clear()`/Stats reset 才会恢复
complete。只要计数、不需要逐指针报告时不要启用 leak tracking。
并发调用期间 `snapshot()` 的每个字段都是 race-free atomic observation，但多个字段不是一个
加锁/序列锁保护的共同瞬间；调用方 quiesce 所有借用 allocator 后，整份 snapshot 才可作为一致
结论。并发 peak 使用本次 `fetch_add` 的线性化结果更新，不会因另一线程随后 free 而漏掉高水位。
`StatsAllocator`/`LeakTracker` 的 reset、销毁或 move 覆盖目标要求 quiescent 且 balanced，违反时
fail-fast。不要把这个诊断 wrapper 合同误套到 `FixedHeap::reset()` 的 bulk invalidation 上。

## 三份最短组合

这些形状由 `MemUsageRecipe` 测试直接编译执行。

```cpp
// request bulk lifetime
hyle::mem::Arena request(hyle::mem::system_allocator());
void* scratch = request.allocate(4096, 64);
request.clear();
```

```cpp
// typed owner-local cache over a hard pool
hyle::mem::ObjectPool<int> core(hyle::mem::system_allocator(), 32);
if (!core.valid()) { /* propagate slab/metadata OOM */ }
hyle::mem::ObjectPoolCache<int> local(core, 8, 2);
if (!local.valid()) { /* propagate cache metadata OOM */ }
{
  auto value = local.try_loan(7);
  if (value) *value += 1;
} // loan returns before the cache is published
local.publish();
```

```cpp
// quota is the call-side outer layer; rejected requests do not reach stats
hyle::mem::StatsAllocator stats =
    hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
hyle::mem::QuotaAllocator quota(stats.get(), 1u << 20);
hyle::mem::AllocatorRef bounded = quota.get();
void* p = bounded.allocate(4096, 64);
if (p != nullptr) bounded.deallocate(p, 4096, 64);
```

声明顺序同时表达销毁顺序：调用外层 `QuotaAllocator` 先析构，backing `StatsAllocator` 后析构；
所有借用 allocator 和 allocation 必须先失效。交换 wrapper 顺序会改变“被 quota 拒绝的尝试”
是否进入统计，必须由调用者按观测目标选择。

## LargeMapping：请求不等于承诺

`LargeMapping::requested_flags()` 记录调用者策略；`accepted_flags()` 记录 OS-facing 调用已
接受的 THP/prefault/placement policy。THP advice 被接受不代表每一物理页已经提升；同样，
interleave policy 被接受也不是每页永久驻留目标节点的证明。
`numa_placement_applied()` 单独说明 placement 调用结果。`allocate*` 返回
`Result<LargeMapping>`：hard bind 或 interleave policy 无法安装时返回带原因的 Status；preferred
可以返回可用 region 但 `numa_placement_applied()==false`，THP 仍是 best-effort。
bind 与 interleave 互斥；interleave 必须搭配 `kNumaAny`，不能传给 `allocate_local`。

`SystemAllocator` 的 mimalloc backing 默认在支持的平台自动探测并使用 NUMA 拓扑。若部署方需要
限制它可见的 node 数，只能在**启动进程之前**设置 `MIMALLOC_USE_NUMA_NODES=N`；embedded
mimalloc 会在进入 `main` / 用户构造器之前由 loader 初始化并缓存该值，因此 HYLE 有意不提供一个
实际上来不及生效的运行期 setter。这个环境变量属于 system allocator 的进程启动配置，不是某次
分配的 placement。运行中需要对特定 span 显式选择 node、bind 或 interleave 时使用
`LargeMapping` / platform NUMA API。

system allocator 的 THP / large-page 默认策略也必须在启动进程前通过
`MIMALLOC_ALLOW_THP` / `MIMALLOC_ALLOW_LARGE_OS_PAGES` 配置。Windows 会在 loader 阶段缓存
large-page capability；因此 HYLE 不提供一个跨平台语义不一致、在 Windows 用户代码中实际太晚的
runtime setter。运行中对特定 span 的 THP/prefault/NUMA 策略用 `LargeMapping` flags；进程启动期
large-page arena 预留用 `reserve_large_page_arena`，其 GiB request unit 不证明物理页粒度。

需要一个连续的大块 owner、但只在选定节点间按自定义 chunk 条带时，使用 selected overload：

```cpp
constexpr std::size_t kMiB = 1024u * 1024u;
const auto available = hyle::platform::numa_nodes();
const unsigned chosen[] = {available.front(), available.back()};
HYLE_ASSIGN_OR_RETURN(
    hyle::mem::LargeMapping region,
    hyle::mem::LargeMapping::allocate_interleaved(
        256u * kMiB, chosen, 2u * kMiB,
        hyle::mem::LargeMappingFlags::kPrefault));
assert(region.interleave_chunk_size() == 2u * kMiB);
```

只在 `available.size()>1` 时构造上面的两端列表；单节点直接传空 span 或一个 node。空 span
选择所有节点，id 可以稀疏且不能用 `[0,count)` 猜；列表按期望条带顺序排列，重复 id 可表达
权重（例如 `{n0,n0,n1}`）。Linux 普通大 chunk 保持惰性 first-touch；极细布局为限制 VMA
policy 数可能在构造期触页，调用方不能把未传 `kPrefault` 当作绝对惰性保证。
selected overload 的 flags 默认 `kNone`：精确小条带与 THP 粒度可能冲突；只有确认 workload
接受大页合并时才显式叠加 `kThpHint`。底层能力与失败语义见 [`platform.md`](platform.md)。

`FiberStack::allocate` 与 `FiberStackPool::acquire` 同样返回 `Result<FiberStack>`，使 sched
能保留原生 VM 失败原因；可信固定配置仍可用 `Fiber` 便利构造，用户输入应走
`Fiber::create` 的可恢复接口。

`FiberStackPool::capacity()` 返回真实 retention 上限：请求超过固定 256 slots 时 clamp，传 0
明确禁用缓存。长寿 runtime 在已 quiesce 的 memory-pressure/idle 控制点可用 `trim(keep)` 或
`clear()` 主动释放 guarded VM；并发 release 允许在 trim 之后重新填充。Scheduler 的确定安全
回收点是 shutdown：全部 fiber 完成且 worker join 后自动 clear 三个标准 tier，不等对象析构。

## 如何看性能

Mem v6 的 host hot-path 只用同一 mimalloc binary 上的 isolated `Heap` 与 runtime-erased PMR owner
比较；Mem-VM v2 另用 Linux model-loader 的相同 mmap、64 次 lazy advice、64 次 partial munmap 和
相同 survivor teardown 比较批量片段 bookkeeping。两者统一经 `scripts/hyle-perf.py` 运行，合同分别在
`bench/contracts/mem-v6-host-allocation-p1-v1.json` 与
`bench/contracts/mem-vm-model-loader-p1-v2.json`。后者的 WSL 结果只属于 Linux guest engineering
证据，不能冒充 native-Linux 正式认证；Windows progressive residency 是另一合同，也不与 POSIX
partial release 数字池化。性能结果只回答列出的 ownership/charged-phase 语义，不外推 planner、所有
OS 或端到端推理 SOTA，更不能用更少 syscall/更弱持久化保证做苹果对橘子排名。

当前设计边界见 [`../architecture/mem-design.md`](../architecture/mem-design.md)，当前签署状态只看
[`module-status.json`](../implementation/module-status.json)。
[`mem-close-plan`](../implementation/tasks/mem/mem-close-plan.md) 与 2026-07-26 的
[`mem-mature-gap-audit`](../implementation/tasks/mem/mem-mature-gap-audit.md) 都是上一轮历史
验收输入，不是依赖升级后工作树的自动分发证书。
安装包的最小默认/组合写法也作为可执行 C++ 消费者保存在
[`../../samples/package/mem`](../../samples/package/mem)。
