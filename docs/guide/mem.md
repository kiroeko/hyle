# HYLE mem：什么情况选择什么

mem 是 host RAM 的基础原语层。它不替业务猜生命周期，也不把某个 benchmark 的最优
形状包装成通用 allocator；使用者先确定 ownership、并发、容量和释放方式，再选择类型。
本文是**语言无关的选型入口**，正文示例用最自然的 C++ RAII 写法；纯 C/FFI 用户先在这里
选能力，再到 [`c-mem.md`](./c-mem.md) 查对应的 Status/out、opaque handle 和 typed 宏。

## 一页选择表

| 需求 | C++ 形状 | C / FFI 形状 | 关键合同 |
|---|---|---|---|
| 普通动态分配 / 依赖注入 | `system_allocator()` / `Allocator` | `hyle_system_allocator` / `hyle_allocator` | sized-free；free/realloc 带回原 size/alignment |
| 接入 STL PMR | `PmrResource` | 不需要：C 直接持有 allocator vtable | 借用 `Allocator` |
| 任意 backing 的严格 live-byte admission | `QuotaAllocator` | `hyle_quota_allocator` | 线程安全；按 requested bytes 精确计量 |
| opt-in allocation 计数 / 泄漏诊断 | `StatsAllocator::create(backing,track,metadata)` | `hyle_stats_allocator` | 被测 backing 与诊断 metadata 分域；计数线程安全，leak inventory 是慢路 |
| request/frame/graph phase 临时对象 | `Region` | `hyle_region` | owned、单线程、growable、bulk `clear()` |
| 调用方 buffer 中的临时对象 | `FixedRegion` | `hyle_fixed_region` 值对象 | external、单线程、bump、无 fallback |
| 从共享硬池借一个 scratch block | `PooledRegion` | `hyle_pooled_region` | 单线程、固定容量、整体归还 |
| LIFO checkpoint/rewind scratch | `StackAllocator` | `hyle_stack_allocator` | owned fixed slab、单线程 |
| 外部 buffer 内变长 alloc/free | `FixedPool` | `hyle_fixed_pool` | 单线程、O(1) indexed fit/merge、individual free、无 fallback；约 4 KiB control |
| 多线程固定块、严格容量 | `SharedPool` | `hyle_shared_pool` | hard capacity；free 完成后立即全局可见 |
| worker-local 热点固定块复用 | `SharedPoolCache` over `SharedPool` | `hyle_shared_pool_cache` | 显式 owner/capacity/batch；可见性是 soft，需 publish |
| 固定数量 typed 对象，每次重建 | `ObjectPool<T>` | `HYLE_OBJECT_POOL_*` 裸槽宏 | C 自行 init/cleanup；C++ 跑 ctor/dtor |
| 昂贵对象保留内部状态 | `PreservingObjectPool<T>` | SharedPool 裸槽 + 调用方状态协议 | return 不销毁内部状态；最终 cleanup 由 C 调用方负责 |
| 单线程 typed 对象、容量可增长 | `GrowablePool<T>` | `hyle_growable_pool` / typed 宏 | 指针稳定，按需增加 chunks |
| 自动归还 pooled object | `PoolLoan<T, Owner>` / `try_loan()` | 手动 acquire/release | owner 必须比 loan/slot 长寿 |
| 大块、THP/NUMA placement | `HugeRegion` | `hyle_huge_region` | requested 与 accepted policy 可分别查询 |
| fiber stack | `FiberStack` / `FiberStackPool` | 不独立导出；由 sched C ABI 管理 | guard page + runtime 专用缓存 |

GPU staging、submission/completion lifetime 属于 `compute/transfer`，不由 mem 用“通用
pinned pool”猜测。graph tensor liveness planner 属于 graph；mem 只提供它消费的 Region。

进程启动期若需要显式预留 1 GiB OS 大页，调用 `reserve_huge_1g_pages`。它是冷策略入口，
返回 `Status`：非法 node/尺寸是 `OutOfRange`，mimalloc 无法取得连续 host RAM 是
`ResourceExhausted`；失败后仍可继续使用普通页。不要在请求热路反复调用。

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
`peak_used()` 不因 clear/release 清零。individual deallocate 是 no-op。大于 threshold 的
dedicated block 在 `clear()` 时归还，普通 block 保留复用。`initial_block_size` / `max_block_size`
是 ordinary block 的总 backing-byte target，已经包含 Region header 与最坏 alignment padding；
单个请求放不下时允许为该 block 突破 target。这样 4 MiB target 不会只因内部 header 变成
`4 MiB + metadata`，`size_limit` 与 allocator size class 也更可预测。

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
接受的 THP/prefault/strict placement。THP advice 被接受不代表每一物理页已经提升。
`numa_placement_applied()` 单独说明 placement 调用结果。`allocate*` 返回
`Result<HugeRegion>`：hard bind/interleave 不能满足时返回带原因的 Status；preferred
可以返回可用 region 但 `numa_placement_applied()==false`，THP 仍是 best-effort。
bind 与 interleave 互斥；interleave 必须搭配 `kNumaAny`，不能传给 `allocate_local`。

`FiberStack::allocate` 与 `FiberStackPool::acquire` 同样返回 `Result<FiberStack>`，使 sched
能保留原生 VM 失败原因；可信固定配置仍可用 `Fiber` 便利构造，用户输入应走
`Fiber::create` 的可恢复接口。

## 如何看性能

benchmark 以场景和合同分组，入口是 `hyle_mem_scenarios_bench`；用
`HYLE_BENCH_SECTIONS=F` 跑收口 workload。正式性能表只维护在
[`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)，本指南不复制点值、
win/loss 或待补证据。性能结果用来帮助选型，不改变前述 ownership/capacity 合同。

当前设计边界见 [`../architecture/mem-design.md`](../architecture/mem-design.md)；预发布冻结的
验收记录见 [`../milestones/mem-close-plan.md`](../milestones/mem-close-plan.md)。
