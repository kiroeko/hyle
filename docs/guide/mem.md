# HYLE mem：按真实内存形状选择能力

mem 是 host RAM 的基础原语层。它不替业务猜生命周期，也不把某个 benchmark 的最优
形状包装成通用 allocator；使用者先确定 ownership、并发、容量和释放方式，再选择类型。

## 一页选择表

| 需求 | 使用 | 关键合同 |
|---|---|---|
| 普通动态分配 / 依赖注入 | `system_allocator()` / `Allocator` | sized-free；free/realloc 带回原 size/alignment |
| 接入 STL PMR | `PmrResource` | 借用 `Allocator` |
| 任意 backing 的严格 live-byte admission | `QuotaAllocator` | 线程安全；按 requested bytes 精确计量 |
| request/frame/graph phase 临时对象 | `Region` | owned、单线程、growable、bulk `clear()` |
| 调用方 buffer 中的临时对象 | `FixedRegion` | external、单线程、bump、无 fallback |
| 从共享硬池借一个 scratch block | `PooledRegion` | 单线程、固定容量；旧名 `Arena` 仅为兼容 alias |
| LIFO checkpoint/rewind scratch | `StackAllocator` | owned fixed slab、单线程 |
| 外部 buffer 内变长 alloc/free | `FixedPool` | 单线程、individual free、无 fallback、立即合并邻块 |
| 多线程固定块、严格容量 | `SharedPool` | hard capacity；free 完成后立即全局可见 |
| 同线程热点固定块复用 | `SharedPoolCached` | TLS 快路；跨线程可见性是 soft，需 publish |
| 固定数量 typed 对象，每次重建 | `ObjectPool<T>` | create/destroy；构造失败退槽 |
| 昂贵对象保留内部状态 | `PreservingObjectPool<T>` | 池建时构造一次，return 不析构 |
| 单线程 typed 对象、容量可增长 | `GrowablePool<T>` | 指针稳定，按需增加 chunks |
| 自动归还 pooled object | `PoolLoan<T, Owner>` / `try_loan()` | move-only；pool 必须比 loan 长寿 |
| 大块、THP/NUMA placement | `HugeRegion` | requested 与 accepted policy 可分别查询 |
| fiber stack | `FiberStack` / `FiberStackPool` | guard page + runtime 专用缓存 |

GPU staging、submission/completion lifetime 属于 `compute/transfer`，不由 mem 用“通用
pinned pool”猜测。graph tensor liveness planner 属于 graph；mem 只提供它消费的 Region。

## Region 家族

### Owned request scratch

```cpp
#include "hyle/mem/region.h"

hyle::mem::RegionOptions options;
options.initial_block_size = 64 * 1024;
options.max_block_size = 1024 * 1024;
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
dedicated block 在 `clear()` 时归还，普通 block 保留复用。

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

`PooledRegion` 语义同样是固定容量 bump，但 storage 来自一个 hard `SharedPool` block；主接口同样用 `clear()`，旧 `reset()` 仅保留给 `Arena` 兼容代码。
析构时整块归还。`StackAllocator` 增加 mark/rewind，适合嵌套 LIFO 临时区。

## Pool 家族

### Hard 与 cached 不能混称

- `SharedPool` 是共享 admission primitive。容量 N 就意味着最多 N 个 live block；任何线程
  完成 deallocate 后，其他线程能立即取得该 block。
- `SharedPoolCached` 优化同线程 free→allocate，把 free slot 暂存在 TLS。它的物理容量固定，
  但跨线程可见容量是 soft；handoff 前由释放线程调用 `publish_thread_cache()`。

typed `ObjectPool<T>` / `ObjectPoolCached<T>` 只是对应 slot pool 加 placement-new 和显式
析构。名称中的 Cached 指 slot visibility，不代表对象状态被保留。

### Reconstruct 与 Preserve

```cpp
hyle::mem::PreservingObjectPool<CodecContext> contexts(
    hyle::mem::system_allocator(), 32, dictionary);

if (auto context = contexts.try_loan()) {
  context->reset_stream();
  context->compress(input, output);
}  // 自动 return；CodecContext 不析构，内部 table 可继续复用
```

普通 `ObjectPool<T>::try_loan(args...)` 在每次取得时构造、归还时析构。Preserving pool 在
池构造/析构时各执行一次对象构造/析构。两者都要求 pool 比所有 loan 长寿。

### External variable-size pool

`FixedPool` 把 block/allocation metadata 放进调用方 storage；free bins、边界和统计属于
`FixedPool` 控制对象（C ABI 的 opaque handle 也单独分配）。它支持
变长 allocate/deallocate/reallocate、精确 requested `used/peak` 和 O(1) 邻块合并；没有
隐藏 payload heap，也不会 fallback。`reset()` 会一次性使所有旧指针失效，调用前必须
保证无 live allocation。

它偏向“可诊断、低碎片、固定 backing”的基础设施合同，不应替代：

- request scratch（用 `FixedRegion`，更快且没有 individual-free metadata）；
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
但 wrapper 顺序决定各层看到 requested attempt 还是 backing success。

## HugeRegion：请求不等于承诺

`HugeRegion::requested_flags()` 记录调用者策略；`accepted_flags()` 记录 OS-facing 调用已
接受的 THP/prefault/strict placement。THP advice 被接受不代表每一物理页已经提升。
`numa_placement_applied()` 单独说明 placement 调用结果。hard bind/interleave 不能满足时
返回 invalid region；preferred/THP 是 best-effort。

## 如何看性能

benchmark 以场景和合同分组，入口是 `hyle_arena_pool_bench`；用
`HYLE_BENCH_SECTIONS=F` 跑收口 workload。2026-07-22 的 Windows/MSVC 本机开发跑（单次，
不是跨平台证书）显示：

| workload | HYLE | 对照 | 结论 |
|---|---:|---:|---|
| external mixed request scratch | FixedRegion 1.21 ns/op | PMR reconstruct 2.05 ns/op | HYLE 约 1.7× throughput |
| owned mixed request scratch | Region 2.08 ns/op | PMR release 1.87 ns/op | HYLE 约慢 11%，需继续解释/优化 |
| external variable-size window | FixedPool 20.45 ns/op | oneTBB fixed_pool 8.66 ns/op | 未达到性能 freeze 门槛 |
| strict quota decoration | 23.0 ns/op | direct system 9.19 ns/op | 约 13.8 ns/op 原子 admission 成本 |
| expensive context loan | Preserve 21.5 ns/op | reconstruct 260 ns/op | Preserve 约 12.1× throughput |

这些数据用来暴露 trade-off，不用来改写 API。尤其 FixedPool 当前提供 oneTBB 对照不具备的
精确统计、release-surviving 检查和立即邻块合并，但约 2.36× throughput 差距仍必须在正式
freeze 前解决或用真实用户 SLA 证明可以接受。完整收口状态见
[`../milestones/mem-close-plan.md`](../milestones/mem-close-plan.md)。
