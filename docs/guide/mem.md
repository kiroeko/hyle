# HYLE mem：什么情况选择什么

mem 是 host RAM 的基础原语层。它不替业务猜生命周期，也不把某个 benchmark 的最优
形状包装成通用 allocator；使用者先确定 ownership、并发、容量和释放方式，再选择类型。
本文是**语言无关的选型入口**，正文示例用最自然的 C++ RAII 写法；纯 C/FFI 用户先在这里
选能力，再到 [`c-mem.md`](./c-mem.md) 查对应的 Status/out、opaque handle 和 typed 宏。

## 一页选择表

| 需求 | C++ 形状 | C / FFI 形状 | 关键合同 |
|---|---|---|---|
| 普通动态分配 / 依赖注入 | `system_allocator()` / `Allocator` | `hyle_system_allocator` / `hyle_allocator` | sized-free；free/realloc 带回原 size/alignment |
| 接入传统 allocator-template API | 默认 system heap 用 `SystemStdAllocator<T>`；运行时 backing 用 `StdAllocator<T>` | 不适用：C 直接持有 allocator vtable | 前者空类型/静态绑定；后者借用 backing；STL OOM 转 `bad_alloc` |
| 接入 STL PMR | 默认 system heap 用 `system_memory_resource()`；运行时 backing 用 `PmrResource`；显式高对齐用 `AlignedPmrResource` | 不需要：C 直接持有 allocator vtable | 默认形状单层 PMR dispatch；注入形状借用 `Allocator` |
| 任意 backing 的严格 live-byte admission | `QuotaAllocator` | `hyle_quota_allocator` | 线程安全；按 requested bytes 精确计量 |
| opt-in allocation 计数 / 泄漏诊断 | `StatsAllocator::create(backing,track,metadata)` | `hyle_stats_allocator` | 被测 backing 与诊断 metadata 分域；计数线程安全，leak inventory 是慢路 |
| request/frame/graph phase 临时对象 | `Region` | `hyle_region` | owned、单线程、growable、bulk `clear()` |
| 调用方 buffer 中的临时对象 | `FixedRegion` | `hyle_fixed_region` 值对象 | external、单线程、bump、无 fallback |
| 从共享硬池借一个 scratch block | `PooledRegion` | `hyle_pooled_region` | 单线程、固定容量、整体归还 |
| LIFO checkpoint/rewind scratch | `StackAllocator` | `hyle_stack_allocator` | owned fixed slab、单线程 |
| 外部 buffer 内变长 alloc/free/realloc + 整池 reset | `FixedPool` | `hyle_fixed_pool` | 单线程、O(1) indexed fit/merge、无 fallback；约 4 KiB control；fixed 指 backing 容量固定 |
| 自己增长 backing 的变长 alloc/free + bulk release | `std::pmr::unsynchronized_pool_resource(system_memory_resource())`；并发用 synchronized 版本 | 暂无单个同形 C 对象；见下文组合边界 | C++ 复用标准池；运行时 upstream 才包 `PmrResource` |
| 多线程固定块、严格容量 | `SharedPool` | `hyle_shared_pool` | hard capacity；free 完成后立即全局可见 |
| worker-local 热点固定块复用 | `SharedPoolCache` over `SharedPool` | `hyle_shared_pool_cache` | 显式 owner/capacity/batch；可见性是 soft，需 publish |
| 固定数量 typed 对象，每次重建 | `ObjectPool<T>` | `HYLE_OBJECT_POOL_*` 裸槽宏 | C 自行 init/cleanup；C++ 跑 ctor/dtor |
| 昂贵对象保留内部状态 | `PreservingObjectPool<T>` | SharedPool 裸槽 + 调用方状态协议 | return 不销毁内部状态；最终 cleanup 由 C 调用方负责 |
| 单线程 typed 对象、容量可增长 | `GrowablePool<T>` | `hyle_growable_pool` / typed 宏 | 指针稳定，按需增加 chunks |
| 自动归还 pooled object | `PoolLoan<T, Owner>` / `try_loan()` | 手动 acquire/release | owner 必须比 loan/slot 长寿 |
| 大块、THP/NUMA placement | `HugeRegion` | `hyle_huge_region` | requested 与 accepted policy 可分别查询 |
| fiber stack | `FiberStack` / `FiberStackPool` | 不独立导出；由 sched C ABI 管理 | guard page + 有界 runtime 缓存；shutdown/trim 可归还 VM |

GPU staging、submission/completion lifetime 属于 `compute/transfer`，不由 mem 用“通用
pinned pool”猜测。graph tensor liveness planner 属于 graph；mem 只提供它消费的 Region。

进程启动期若需要显式预留 1 GiB OS 大页，调用 `reserve_huge_1g_pages`。它是冷策略入口，
返回 `Status`：非法 node/尺寸是 `OutOfRange`，mimalloc 无法取得连续 host RAM 是
`ResourceExhausted`；失败后仍可继续使用普通页。不要在请求热路反复调用。

## 通用 allocator、typed array、标准 adapter 与 PMR

`Allocator` 是 HYLE 内部和 C++ 用户的按值注入句柄；模块之间直接传这个 C++ 类型，不经
C ABI。`allocate_n<T>` / `allocate_zeroed<T>` / `reallocate_n<T>` 会检查 count 乘法溢出，
并允许显式 alignment；用同一个 alignment 调 `deallocate_n`。`reallocate_n` 失败或溢出时
旧 allocation 仍然有效；因为底层 resize 可能按字节搬移，它只接受 trivially-copyable `T`，
带构析/自引用状态的对象应新建 storage 后逐对象 move/销毁。
原生 C++ allocate/deallocate/reallocate 的 alignment 必须是非零 2 次幂；allocate/reallocate 的
非法值会 fail-fast。deallocate 还必须带回成功分配时相同的 size/alignment，和标准 allocator/PMR
一样属于调用方前置条件：Debug 会诊断，Release 热路不重复检查。纯 C convenience 作为 FFI 边界
仍在 Release 校验，并把 `alignment=0` 在头内折叠成默认对齐。

```cpp
hyle::mem::Allocator alloc = hyle::mem::system_allocator();
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
不需要用户管理 resource 对象，也不会在 PMR 虚调用后再叠一层 `Allocator` 函数指针。只有
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

HYLE **并不缺可变大小内存**：通用 allocator 接受任意请求大小，Region 接受变长请求并 bulk
clear/release，`FixedPool` 则在调用方固定容量 buffer 内提供变长 allocate/free/reallocate + bulk
`reset()`。这里唯一没有单独上架的，是“由 HYLE 自己拥有并自动增长 backing”的又一个通用 heap。

需要 owned growable、individual free + 一次 bulk release 且消费方接受 PMR 时，组合标准 pool
resource。默认 system upstream 直接使用静态 singleton：

```cpp
std::pmr::unsynchronized_pool_resource local_pool(
    hyle::mem::system_memory_resource()); // 单 owner
std::pmr::vector<Record> records(&local_pool);
// records 先销毁；local_pool.release() 可提前归还全部 pooled blocks。
```

跨线程并发分配改用 `std::pmr::synchronized_pool_resource`。只有需要 quota、stats、region 或
用户 backing 时才创建 `PmrResource upstream(alloc)` 并把 `&upstream` 传给标准 pool；此时 pool
及其所有 allocation 不能比 upstream/backing 长寿。

## Region 家族

### Owned request scratch

```cpp
#include "hyle/mem/region.h"

hyle::mem::RegionOptions options;
options.initial_block_size = 64 * 1024;
options.max_block_size = 4 * 1024 * 1024;
options.large_allocation_threshold = 256 * 1024;
options.size_limit = 8 * 1024 * 1024;  // reserved bytes，包含 header/padding

hyle::mem::Region scratch(hyle::mem::system_allocator(), options);
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
是 ordinary block 的总 backing-byte target，已经包含 Region header 与最坏 alignment padding；
单个请求放不下时允许为该 block 突破 target。这样 4 MiB target 不会只因内部 header 变成
`4 MiB + metadata`，`size_limit` 与 allocator size class 也更可预测。

长寿命、多 block Region 在每次 `clear()` 后用单调 cursor 依次复用 retained blocks；一个
epoch 内每个 block 至多考察一次，所以总过渡 O(blocks)、分摊到 block transition 为 O(1)。
某个 block 的尾部不能满足当前请求后，本 epoch 不再回头尝试更小请求；这是 bounded bump
arena 用少量 tail slack 换取可预测复杂度的合同。若小请求尺寸高度交错且必须回收每个洞，
应选 `FixedPool`，而不是把 Region 当 variable-size heap。

### External workspace

```cpp
alignas(64) std::byte storage[64 * 1024];
hyle::mem::FixedRegion scratch(storage, sizeof(storage));
void* header = scratch.allocate(96, 32);
void* table = scratch.allocate(4096, 64);
if (header == nullptr || table == nullptr) return workspace_too_small;
scratch.clear();
```

`FixedRegion` 不拥有 storage、不增长、不 fallback；storage 必须比所有切片长寿。它不可
copy/move，避免两个 cursor 无意间在同一 buffer 上重叠分配。

`PooledRegion` 语义同样是固定容量 bump，但 storage 来自一个 hard `SharedPool` block；统一使用 `clear()`。
析构时整块归还。`StackAllocator` 增加 mark/rewind，适合嵌套 LIFO 临时区。

## Pool 家族

### Hard core 与显式 cache 不能混称

- `SharedPool` 是共享 admission primitive。容量 N 就意味着最多 N 个 live block；任何线程
  完成 deallocate 后，其他线程能立即取得该 block。
  默认共享 shard 数按当前进程可用 CPU（Linux 为调用线程可继承的 affinity）取 2 的幂，
  而不是按容器外的整机核数浪费 metadata；仍受 128 与 block count 上限约束。
- `SharedPoolCache` 是附着到 `SharedPool` 的单 owner L1。storage、容量、transfer batch 和
  生命周期都由用户显式选择；本地 free slot 只有在 `publish()`/析构后才回到共享 core。

typed `ObjectPoolCache<T>` 同样附着到 `ObjectPool<T>`，只增加 placement-new/显式析构的
typed shell。Cache 指 slot visibility，不代表对象状态被保留。

```cpp
hyle::mem::SharedPool core(hyle::mem::system_allocator(), 64, 4096);
if (!core.valid()) { /* propagate slab/metadata OOM before use */ }
std::array<std::uint32_t, 128> local_slots{};
hyle::mem::SharedPoolCache local(core, local_slots, 32);
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

`FixedPool` 已经是 **variable-size + individual free/reallocate + bulk reset**；名称中的 fixed
只指调用方提供的 payload backing 地址和容量固定，不是只能分配固定大小对象。

```cpp
alignas(64) std::array<std::byte, 64 * 1024> storage{};
hyle::mem::FixedPool pool(storage);
void* header = pool.allocate(96, 32);
void* payload = pool.allocate(1536, 64);  // 同一池可同时服务不同大小
if (payload != nullptr) {
  if (void* grown = pool.reallocate(payload, 1536, 4096, 64)) payload = grown;
}
// 任一返回 nullptr 都表示 bounded capacity exhausted；reallocate 失败保留旧 allocation。
pool.deallocate(header);
pool.deallocate(payload);
pool.reset();  // 此时无 live allocation；整池恢复
```

`FixedPool` 把 block/allocation metadata 放进调用方 storage；two-level segregated free lists、
两层 bitmap、边界和统计属于 `FixedPool` 控制对象（C surface 可由 caller-owned control 原地
构造，或从显式 control allocator 创建 opaque handle）。一级按数量级定位，二级把每个范围
分成 8 段；分配未命中 recent-free 快路后直接取得一个满足大小的链表头，不扫描同级空闲块。
因此 allocate/free/邻块合并的索引工作都是 O(1)，不会因长期碎片化退化。它支持变长
allocate/deallocate/reallocate、精确 requested `used/peak`；realloc 在当前物理 span 足够或后一
邻块空闲时 O(1) 原地 grow，shrink 保留 span 供后续 O(1) regrow；只有搬移时复制
O(min(old,new))。没有隐藏 payload heap，也不会 fallback。`reset()` 会一次性使所有旧指针
失效，调用前必须保证无 live allocation。

这个 bounded lookup 需要约 4 KiB 的显式 control。把它放进 arena/cache/device owner 等
长寿命对象，或通过 control allocator 创建一次并复用；不要为每个 request 在小栈帧上构造。
如果 storage 本来就很小、生命周期按 phase 结束，`FixedRegion` 的三字 bump cursor 更合适。

它偏向“可诊断、低碎片、固定 backing”的基础设施合同，不应替代：

- request scratch（用 `FixedRegion`，更快、control 更小且没有 individual-free metadata）；
- 同尺寸对象池（用 `SharedPool` / `ObjectPool`）；
- 通用 growable malloc（用 system allocator）。

## Quota、统计与失败

```cpp
hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                64 * 1024 * 1024);
hyle::mem::Allocator bounded = quota.get();
void* p = bounded.allocate(bytes, alignment);
if (p == nullptr) return overload;
// quota.used(), quota.peak(), quota.remaining()
bounded.deallocate(p, bytes, alignment);
```

Quota 在进入 backing 前原子 reserve，因此并发成功调用不会突破 limit；backing OOM 会回滚
reservation。它计 requested live bytes，不计 allocator usable size/metadata。返回的
`Allocator` 借用 Quota 对象。

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
`reset`、销毁或 move 覆盖目标都要求 quiescent 且 balanced，违反时 fail-fast。

## 三份最短组合

这些形状由 `MemUsageRecipe` 测试直接编译执行。

```cpp
// request bulk lifetime
hyle::mem::Region request(hyle::mem::system_allocator());
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
hyle::mem::Allocator bounded = quota.get();
void* p = bounded.allocate(4096, 64);
if (p != nullptr) bounded.deallocate(p, 4096, 64);
```

声明顺序同时表达销毁顺序：调用外层 `QuotaAllocator` 先析构，backing `StatsAllocator` 后析构；
所有借用 allocator 和 allocation 必须先失效。交换 wrapper 顺序会改变“被 quota 拒绝的尝试”
是否进入统计，必须由调用者按观测目标选择。

## HugeRegion：请求不等于承诺

`HugeRegion::requested_flags()` 记录调用者策略；`accepted_flags()` 记录 OS-facing 调用已
接受的 THP/prefault/placement policy。THP advice 被接受不代表每一物理页已经提升；同样，
interleave policy 被接受也不是每页永久驻留目标节点的证明。
`numa_placement_applied()` 单独说明 placement 调用结果。`allocate*` 返回
`Result<HugeRegion>`：hard bind 或 interleave policy 无法安装时返回带原因的 Status；preferred
可以返回可用 region 但 `numa_placement_applied()==false`，THP 仍是 best-effort。
bind 与 interleave 互斥；interleave 必须搭配 `kNumaAny`，不能传给 `allocate_local`。

`SystemAllocator` 的 mimalloc backing 默认在支持的平台自动探测并使用 NUMA 拓扑。若部署方需要
限制它可见的 node 数，只能在**启动进程之前**设置 `MIMALLOC_USE_NUMA_NODES=N`；embedded
mimalloc 会在进入 `main` / 用户构造器之前由 loader 初始化并缓存该值，因此 HYLE 有意不提供一个
实际上来不及生效的运行期 setter。这个环境变量属于 system allocator 的进程启动配置，不是某次
分配的 placement。运行中需要对特定 span 显式选择 node、bind 或 interleave 时使用
`HugeRegion` / platform NUMA API。

system allocator 的 THP / large-page 默认策略也必须在启动进程前通过
`MIMALLOC_ALLOW_THP` / `MIMALLOC_ALLOW_LARGE_OS_PAGES` 配置。Windows 会在 loader 阶段缓存
large-page capability；因此 HYLE 不提供一个跨平台语义不一致、在 Windows 用户代码中实际太晚的
runtime setter。运行中对特定 span 的 THP/prefault/NUMA 策略用 `HugeRegion` flags，显式 1GiB
arena 预留用 `reserve_huge_1g_pages`。

需要一个连续的大块 owner、但只在选定节点间按自定义 chunk 条带时，使用 selected overload：

```cpp
constexpr std::size_t kMiB = 1024u * 1024u;
const auto available = hyle::platform::numa_nodes();
const unsigned chosen[] = {available.front(), available.back()};
HYLE_ASSIGN_OR_RETURN(
    hyle::mem::HugeRegion region,
    hyle::mem::HugeRegion::allocate_interleaved(
        256u * kMiB, chosen, 2u * kMiB,
        hyle::mem::HugeFlags::kPrefault));
assert(region.interleave_chunk_size() == 2u * kMiB);
```

只在 `available.size()>1` 时构造上面的两端列表；单节点直接传空 span 或一个 node。空 span
选择所有节点，id 可以稀疏且不能用 `[0,count)` 猜；列表按期望条带顺序排列，重复 id 可表达
权重（例如 `{n0,n0,n1}`）。Linux 普通大 chunk 保持惰性 first-touch；极细布局为限制 VMA
policy 数可能在构造期触页，调用方不能把未传 `kPrefault` 当作绝对惰性保证。
selected overload 的 flags 默认 `kNone`：精确小条带与 THP 粒度可能冲突；只有确认 workload
接受大页合并时才显式叠加 `kThpHint`。底层能力与失败语义见 [`platform.md`](./platform.md)。

`FiberStack::allocate` 与 `FiberStackPool::acquire` 同样返回 `Result<FiberStack>`，使 sched
能保留原生 VM 失败原因；可信固定配置仍可用 `Fiber` 便利构造，用户输入应走
`Fiber::create` 的可恢复接口。

`FiberStackPool::capacity()` 返回真实 retention 上限：请求超过固定 256 slots 时 clamp，传 0
明确禁用缓存。长寿 runtime 在已 quiesce 的 memory-pressure/idle 控制点可用 `trim(keep)` 或
`clear()` 主动释放 guarded VM；并发 release 允许在 trim 之后重新填充。Scheduler 的确定安全
回收点是 shutdown：全部 fiber 完成且 worker join 后自动 clear 三个标准 tier，不等对象析构。

## 如何看性能

benchmark 以场景和合同分组，入口是 `hyle_mem_scenarios_bench`；用
`HYLE_BENCH_SECTIONS=F` 跑正常 request workload，用 `HYLE_BENCH_SECTIONS=J` 跑长寿命
Region retained-block scaling。正式性能表只维护在
[`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)，本指南不复制点值、
win/loss 或待补证据。性能结果用来帮助选型，不改变前述 ownership/capacity 合同。

当前设计边界见 [`../architecture/mem-design.md`](../architecture/mem-design.md)，当前签署状态只看
[`../milestones/pre-release-readiness.md`](../milestones/pre-release-readiness.md)。
[`../milestones/mem-close-plan.md`](../milestones/mem-close-plan.md) 与 2026-07-26 的
[`../milestones/mem-mature-gap-audit.md`](../milestones/mem-mature-gap-audit.md) 都是上一轮历史
验收输入，不是依赖升级后工作树的自动分发证书。
安装包的最小默认/组合写法也作为可执行消费者保存在
[`../../samples/package/mem`](../../samples/package/mem)：C 与 C++ 两份样例都只通过
`find_package(Hyle COMPONENTS mem)` 使用 `hyle::mem`，持续覆盖 shared/static 交付。
