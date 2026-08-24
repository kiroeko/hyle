# HYLE mem C surface

> **发布面**：`include/hyle/c/mem.h`  
> **实现**：`src/mem/c_mem.cc`（链 `hyle_mem`）  
> **稳定性**：当前 C/FFI 能力与 manifest 已机械审查，但模块认证仍以 [`pre-release-readiness`](../milestones/pre-release-readiness.md) 为准；正式发布前可直接重塑，不承诺 ABI 兼容
> **命名**：全部 `hyle_*` / `HYLE_*`

本文只负责把**已经选定的 mem 能力**准确映射到 C/FFI：签名、Status/out、所有权和
生命周期。还没决定该用 Region、FixedPool、SharedPool 还是 cache 时，先看语言无关的
[`mem.md` 一页选择表](./mem.md)；这里不另立一套选型标准。

跨语言 / 跨编译器只绑定本头，不要 include `hyle/mem/*.h`。有 `backing` 参数的 opaque
control 与内部 storage 都由该 backing 分配；显式区分 payload/metadata 的产品则分别服从
对应 allocator。FixedRegion 是可按值持有的 3-word cursor；
FixedPool 同时提供 caller-owned `*_storage + *_init`（control/payload 都不触堆）和
`*_create(control_allocator, ...)`。`hyle_fixed_pool_storage` 是保证足够的 4096-byte 长寿命
索引 control；动态 FFI 可先查 exact runtime size/alignment。两种构造的 payload 始终只来自
调用方 storage，绝不 fallback。
所有 opaque handle 对外使用 `hyle_region *`、`hyle_shared_pool *` 等产品名；尾缀 `_o`
只存在于不可解引用的内部 struct tag，不进入用户变量和函数签名。

所有可能失败的冷构造/初始化入口统一采用 `hyle_status + out`：out 参数在前，函数进入时
先清空 `*out`，成功返回零值 OK，失败返回可拥有、可传播的共享 error Status。调用方始终用
`hyle_status_free` 结束 Status 生命周期。`alloc/realloc`、region bump、pool pop/acquire
仍在热路直接返回指针或 NULL；容量耗尽/OOM 是其值合同，不为每次调用制造 Status。

## 1. 链接与所有权

```c
#include "hyle/c/mem.h"
// link: hyle_mem
```

纯 C 验收目标：`hyle_mem_c_test`。
安装 SDK 的独立消费者见 [`../../samples/package/mem/c`](../../samples/package/mem/c)：它只请求
`COMPONENTS mem`，先走默认 system allocator，再演示 `QuotaAllocator -> Region` 组合与
`hyle_status` 所有权；shared/static 都必须能直接构建运行。

| 块 | 入口 | 所有权 / 容量合同 |
|---|---|---|
| Allocator 句柄 | `hyle_allocator` | POD 值类型；按值传 |
| system 分配器 | `hyle_system_allocator` | 进程级；alloc/free 携带相同 size/alignment |
| QuotaAllocator | `hyle_quota_allocator_create(&out,...)` → `destroy` | 任意 backing 的严格 requested live-byte 上限 |
| StatsAllocator | `hyle_stats_allocator_create(&out,backing,metadata,track)` → `destroy` | 测量 backing；control/诊断状态服从 metadata；可选慢速 leak inventory |
| Region | `hyle_region_create(&out,...)` → `destroy` | owned growable bump；clear 保留普通 blocks |
| FixedRegion | `hyle_fixed_region_init(&region, storage, size)` | 返回 Status；按值 external bump；无 create/destroy、严格 no-heap |
| FixedPool | `hyle_fixed_pool_init(&pool,control,control_bytes,storage,size)` 或 `create(&pool,control_allocator,...)` → `destroy` | external variable-size alloc/free/realloc + bulk reset；O(1) two-level fit/merge；init 严格 no-heap；约 4 KiB control |
| SharedPool | `hyle_shared_pool_create(&out,...)` → `destroy` | **硬容量**；completed free 立即对所有线程可见 |
| SharedPoolCache | `hyle_shared_pool_cache_create(&out,pool,...)` → `destroy` | **软视图**；显式 owner/capacity/batch，publish 后回 hard core |
| StackAllocator | `hyle_stack_allocator_create(&out,...)` → `destroy` | 自有 slab，单线程 LIFO bump |
| PooledRegion | `hyle_pooled_region_create(&out,pool)` → `destroy` | 借 `SharedPool` 一个块 |
| GrowablePool | `hyle_growable_pool_create(&out,...)` → `destroy` | 单线程、按需增长的裸槽池 |
| HugeRegion | `hyle_huge_region_open*(&out,...)` → `destroy` | 大页 / NUMA host RAM 区 |
| 进程维护/策略 | `hyle_system_collect` / `hyle_enable_allocator_*` / `hyle_reserve_huge_1g_pages` | cold path 与全局软策略 |

**sized-free 合同**：`free` / `deallocate` 必须带回分配请求时相同的 size 与 alignment。pool/region/stack 不保存每指针尺寸；传错是调用方契约违反。只有 system 分配器提供物理容量查询 `hyle_system_usable_size`。

## 2. Allocator vtable

```c
typedef struct hyle_allocator {
  void* (*alloc)(void* state, size_t size, size_t alignment);
  void  (*free)(void* state, void* ptr, size_t size, size_t alignment);
  void* (*realloc)(void* state, void* ptr, size_t old_size,
                   size_t new_size, size_t alignment);
  void* state;
} hyle_allocator;
```

| 符号 | 说明 |
|---|---|
| `hyle_default_alignment()` | header-inline `alignof(max_align_t)`；常量折叠，不是 DLL 导出 |
| `hyle_system_allocator()` | mimalloc-backed 进程级句柄 |
| `hyle_system_usable_size(ptr)` | system 块真实可用字节；NULL→0 |
| `hyle_system_good_size(size)` | 不分配地给出推荐 request size（≥size）；不承诺等于 alloc(size) 的 usable |
| `hyle_system_collect(force)` | 收集调用线程的 mimalloc heap；force 非零可能达到毫秒级；worker pool 需在每个目标 worker 上调用 |
| `hyle_alloc` / `hyle_free` / `hyle_realloc` | header-inline vtable 转发；alignment==0 使用默认对齐 |
| `hyle_calloc` / `hyle_realloc_array` | header-inline、count 乘法防溢出；resize 失败/溢出时旧 allocation 保持有效 |
| `hyle_allocator_is_valid(a)` | header-inline 检查三个函数指针，供配置/FFI 边界验证 |

C/C++ 两面通过 trampoline 显式构造，不 reinterpret 布局。
`hyle_alloc/free/realloc(..., alignment=0)` 在头内直接折叠默认对齐，不再先跨 DLL 查询。
0 是这三个 C convenience 的专用默认值；非零 alignment 必须是 2 次幂，alloc/free/realloc 的
非法值会 fail-fast。直接手调 vtable function pointer 属低层路径，调用方必须先满足同一合同。
这些是纯 C header helper，所以使用标准 C 的 `<stddef.h>/<stdint.h>/<string.h>`；HYLE 的
C++ 实现文件继续用 `<cstddef>/<cstdint>/<cstring>`，两侧不混写语言头。

拥有 backing 的构造器同时提供 header-inline `*_create_default(&out, ...)`，等价于传
`hyle_system_allocator()`；常用用户不必为了默认堆手写 allocator injection。显式
`create(backing, ...)` 继续用于 quota/region/pool 组合与故障注入。

### 2.1 QuotaAllocator

`hyle_quota_allocator_as_allocator()` 返回借用 wrapper 的 allocator。并发分配先原子 reserve，
成功 live bytes 不会超过 limit；backing OOM 回滚。destroy 前必须释放全部 allocation 并
静默并发访问。`used/peak/limit/remaining` 都按 requested bytes 计量；`remaining` 是当前仍可
reserve 的严格预算，不是 backing 的物理空闲字节。

### 2.2 StatsAllocator

`hyle_stats_allocator_as_allocator()` 返回借用 wrapper 的 allocator；成功 alloc/free/realloc
更新 64-bit 并发计数。`hyle_stats_allocator_snapshot(s, &out)` 用出参返回 POD，避免 FFI
绑定大型 struct-return ABI。并发更新期间每个字段都是 race-free observation，但多个 relaxed
load 不形成一个共同瞬间；quiesce 全部借用 allocator 后 snapshot 才是跨字段一致的精确结论。
并发 peak 使用 allocation/grow 的原子 add 结果更新，不漏掉随后被其他线程 free 的高水位。
`track_leaks!=0` 额外维护带锁的 pointer→size map，并可用
`copy_leaks(&total,...)` 分页/截断读取并返回 Status（复制时会从 metadata 取得临时数组）；
这是诊断慢路，生产热路只开 counters 或直接不用 wrapper。
`reset` 要求没有 live allocation，并同时清空 leak inventory。destroy 与 move 覆盖同样要求
quiescent/balanced；违反时 fail-fast，避免已借出的 allocator 静默变成悬空 state。

若记录某次 allocation 时 metadata OOM，用户 allocation 仍成功、计数仍精确，但
`hyle_stats_allocator_leak_records_dropped()!=0` 且 `leaks_complete()==0`。`copy_leaks` 仍复制
已知条目并写已知 total，同时返回 `HYLE_STATUS_DATA_LOSS`，提醒调用方不能把已知条目当完整
清单。释放全部 allocation 后 reset 才清掉历史 dropped 状态。

`backing` 是真正被观察的数据域；opaque control、稳定计数状态和 leak map 使用独立
`metadata`。因此 Stats 可以安全包住 fixed-block 或 strict-quota allocator，而不会在第一笔
用户 allocation 前用诊断状态污染其容量。普通用法选 `create_default`；测量自定义 backing
但不想自管诊断内存时用 `create_with_default_metadata`；只有需要 arena ownership、故障注入
或隔离诊断开销时才同时显式传两者。所有构造阶段 OOM 都直接返回 RESOURCE_EXHAUSTED，
已完成的 metadata 分配会原域回滚。

### 2.3 Region、FixedRegion 与 FixedPool

C surface 也**不是没有可变大小内存**：`hyle_allocator` 接受任意请求大小，`hyle_region` 是
owned growable 的变长请求 + bulk release，`hyle_fixed_pool` 是 external bounded 的变长
allocate/free/reallocate + bulk reset。当前没有的是 HYLE-owned、自动增长、同时 individual free
和 bulk release 的单个 C pool handle；没有真实 C/FFI 消费者前，不复制一套与通用 heap 重叠的
所有权协议。需要硬 buffer 边界选 FixedPool，需要 owned phase lifetime 选 Region。

- `hyle_region_create(&out,...)` 的 options 以 `HYLE_REGION_OPTIONS_INIT` 初始化。`struct_size`
  是固定 `uint32_t`，随后固定 `uint32_t reserved==0`；64-bit ABI 上首个 size 字段仍在 offset 8，
  整体仍为 40 bytes。NULL options 或只有 `HYLE_REGION_OPTIONS_REQUIRED_SIZE`（8 bytes）的前缀
  都选全部 INIT 默认；每个完整出现的 size 字段依次覆盖默认，缺失/部分尾字段保持默认，故
  缺失 `size_limit` 明确为 `SIZE_MAX`。未知尾部忽略；调用方必须保证声明的 struct_size 字节
  可读。小于 8 或 reserved 非零返回 INVALID_ARGUMENT。`clear` 保留普通 blocks、释放
  dedicated large blocks；`release` 归还全部。
- `hyle_fixed_region_init(&region,storage,size)` 直接初始化 3-word 值对象，是严格 no-heap
  bulk-reset bump；没有需要管理的 control allocation，也无需 destroy。旧切片在 clear 后全部失效。
- `hyle_fixed_pool_init(&out,&control,sizeof control,storage,size)` 的 control/payload 都由调用方持有；
  `hyle_fixed_pool_create(&out,control_allocator,storage,size)` 只从显式 allocator 创建 control，
  `create_default` 才选择 system allocator。两者都支持 individual allocate/free/realloc；realloc
  可 O(1) 原地利用当前 span/后一空闲邻块，shrink 保留 span 让后续 regrow 仍为 O(1)。reset 前必须
  无 live allocation；foreign/double return 是合同错误并 fail-fast。两级 bitmap 索引使
   allocate/free/邻块合并的查找工作为 O(1)，不会随空闲块数量增长。

`FixedPool` 名称中的 fixed 指 payload backing 的地址与容量固定，不是 allocation 大小固定。
`create` 只拥有 control；payload 仍完全来自调用方 storage，所以两种构造都不自动增长也不 fallback。

`hyle_fixed_pool_storage` 保证 4096 bytes/alignment 足以容纳当前实现：把它嵌入拥有 payload 的
长寿命 owner、放在 static storage，或用 `create` 从合适的 control allocator 分配一次。动态
FFI/插件若自己分配 control，使用 `hyle_fixed_pool_control_size/alignment()` 的 runtime 值；
`init` 同时接收 `control_bytes`，旧 header 的小 buffer 会返回 INVALID_ARGUMENT 而不是被覆盖。
不要在每个 request 的小栈帧里反复创建；小 buffer 的 phase scratch 应选三字的
`hyle_fixed_region`。

三者都单线程。它们的 `as_allocator` 句柄借用原对象；Region/FixedPool destroy 后失效，
FixedRegion 则在值对象或 payload storage 失效后失效。

观察量：FixedRegion 提供 `used/capacity/remaining`；FixedPool 提供
`used/peak/capacity`。FixedPool 的 peak 是自创建以来 requested live bytes 的高水位，reset
不会把历史峰值伪装成零。

```c
unsigned char payload[16 * 1024];
hyle_fixed_pool_storage control;
hyle_fixed_pool *pool = NULL;
hyle_status status =
    hyle_fixed_pool_init(&pool, &control, sizeof control,
                         payload, sizeof payload);
if (!hyle_status_is_ok(status)) {
  /* inspect/propagate status */
  hyle_status_free(&status);
  return;
}
hyle_status_free(&status);
void *header = hyle_fixed_pool_allocate(pool, 96, 32);
void *payload_ptr = hyle_fixed_pool_allocate(pool, 1536, 64); /* 大小可不同 */
if (payload_ptr != NULL) {
  void *grown = hyle_fixed_pool_reallocate(pool, payload_ptr, 1536, 4096, 64);
  if (grown != NULL) payload_ptr = grown; /* 失败时旧 allocation 仍有效 */
}
/* 任一 allocate 返回 NULL 都表示 bounded capacity exhausted。 */
hyle_fixed_pool_deallocate(pool, header);
hyle_fixed_pool_deallocate(pool, payload_ptr);
hyle_fixed_pool_reset(pool); /* 已无 live allocation；整池恢复 */
hyle_fixed_pool_destroy(pool); /* 不释放 control 或 payload */
```

## 3. SharedPool：硬容量共享定长块池

`SharedPool` 没有 TLS 隐藏。`deallocate()` 完成后，块立即进入共享 shard；`allocate()` 在返回 NULL 前用稳定双快照确认所有 shard 在同一公共区间内都为空，避免块跨 shard 移动造成分片假耗尽。

内部默认 shard 数使用 platform 的动态进程可用 CPU 数（向上取 2 的幂，受 128 与 block
count 限制），容器/cpuset 内不会按整机核数多建无意义的 cache-line head；C 调用者无需配置。

正常路径使用 32-bit tag 的 64-bit CAS。tag 到饱和值的 shard 会**永久**切换到 serialized fallback，并用独立 64-bit change counter 参与空快照校验；tag 绝不回绕，因此长寿命池不会重新接受陈旧 CAS。

| 符号 | 说明 |
|---|---|
| `hyle_shared_pool_create(&out, backing, block_size, block_count, alignment)` | 建硬容量池；alignment==0→默认；资源失败或非法/溢出 layout→非 OK Status，out 保持 NULL |
| `hyle_shared_pool_destroy(p)` | 销毁池；调用方保证无块存活、无并发访问 |
| `hyle_shared_pool_allocate(p)` | 取一块；稳定观察整池为空才返 NULL |
| `hyle_shared_pool_deallocate(p, ptr)` | 立即发布到共享 core；NULL no-op |
| `hyle_shared_pool_reset(p)` | 重建所有 shard；要求无活块且并发已静默 |
| `hyle_shared_pool_block_size` / `capacity` | 查询 |
| `hyle_shared_pool_as_allocator(p)` | alloc=定长 pop、free=push；realloc 在 block 内原地 resize，超限失败且保留旧块，size=0 归还；句柄借用 pool |

### 3.1 typed 裸槽宏（默认硬容量）

| 宏 | 说明 |
|---|---|
| `HYLE_OBJECT_POOL_CREATE(&out, backing, T, count)` | 用 `sizeof(T)` / `alignof(T)` 创建 `SharedPool`，返回 Status |
| `HYLE_OBJECT_POOL_ACQUIRE(pool, T)` | 返回裸、未初始化的 `T*` 槽；耗尽→NULL |
| `HYLE_OBJECT_POOL_RELEASE(pool, p)` | 调用方先清理对象，再归还槽 |
| `HYLE_ALIGNOF(T)` | C++/C23 `alignof`、C11 `_Alignof`，旧 C 则为 0 |

C 宏不运行 ctor/dtor。需要构造/析构语义时用 C++ `hyle::mem::ObjectPool<T>`；它默认同样是硬容量。

## 4. SharedPoolCache：显式 owner-local 快路

`SharedPoolCache` 附着到已有的 hard `SharedPool`。cache handle 与 `uint32_t` index array 来自显式 metadata allocator；capacity 与 transfer batch 都是构造参数。一个 cache 同时只由一个 owner 使用，不再有隐藏 TLS node 或跨 DLL registry。

代价是软容量：cache 中已 free 的块在 `publish()`/destroy 前，hard core 和其他 cache 看不到。容量紧贴工作集、或跨线程 free 后必须立即复用时，应直接使用 `SharedPool`。

| 符号 | 说明 |
|---|---|
| `hyle_shared_pool_cache_create(&out, pool, metadata, capacity, batch)` / `destroy` | 返回 Status；绑定 hard pool；destroy 自动 publish；pool 必须更长寿 |
| `hyle_shared_pool_cache_allocate` / `deallocate` | owner-local 快路；其他 cache 中的 free 块不可见 |
| `hyle_shared_pool_cache_publish(p)` | 把本地 free indices 发布回 shared core |
| `capacity` / `cached` / `transfer_batch` / `block_size` | 查询显式 policy 与当前软窗口 |
| `hyle_shared_pool_cache_as_allocator(p)` | 借用型 allocator 视图 |

对应 typed 裸槽宏：

- `HYLE_OBJECT_POOL_CACHE_CREATE`
- `HYLE_OBJECT_POOL_CACHE_ACQUIRE`
- `HYLE_OBJECT_POOL_CACHE_RELEASE`
- `HYLE_OBJECT_POOL_CACHE_PUBLISH`

C++ 对应类型是附着到 `ObjectPool<T>` 的 `ObjectPoolCache<T>`。

## 5. PooledRegion 与 StackAllocator

### 5.1 PooledRegion：借一个硬池块做 bump

| 符号 | 说明 |
|---|---|
| `hyle_pooled_region_create(&out, pool)` | 从硬容量 `SharedPool` 借一块；pool 为空→INVALID_ARGUMENT，control OOM/池耗尽→RESOURCE_EXHAUSTED |
| `hyle_pooled_region_destroy(r)` | 把整块还给 pool |
| `hyle_pooled_region_allocate(r, bytes, alignment)` | 单线程 pointer bump；放不下→NULL |
| `hyle_pooled_region_clear(r)` | 游标回块头；要求此前切片不再使用 |
| `hyle_pooled_region_used` / `remaining` / `block_size` | 查询 |
| `hyle_pooled_region_as_allocator(r)` | alloc=bump、free=no-op、realloc=bump+copy |

C/C++ 使用同一产品名，未发布阶段不保留 `Arena/reset` 兼容壳。它只借 hard
`SharedPool`，单块 borrow 不从 owner-local cache 获益。

### 5.2 StackAllocator：自有 slab 的 LIFO bump

| 符号 | 说明 |
|---|---|
| `hyle_stack_allocator_create(&out, backing, size)` / `destroy` | 自持 slab；资源失败或 size==0→非 OK Status |
| `hyle_stack_allocator_allocate(s, bytes, alignment)` | 单线程 pointer bump；放不下→NULL |
| `hyle_stack_allocator_mark` / `rewind` | 保存并恢复 LIFO 游标 |
| `hyle_stack_allocator_reset` | 整体复用 |
| `hyle_stack_allocator_used` / `capacity` / `remaining` | 查询 |
| `hyle_stack_allocator_as_allocator` | alloc=bump、free=no-op、realloc=bump+copy |

## 6. GrowablePool：单线程可增长裸槽池

| 符号 | 说明 |
|---|---|
| `hyle_growable_pool_create(&out, backing, slot_size, slot_align, initial_slots)` | 首块惰性分配，此后几何增长；构造失败→非 OK Status |
| `hyle_growable_pool_destroy` | 释放全部块 |
| `hyle_growable_pool_acquire` / `release` | 取/还裸槽，不运行 ctor/dtor |
| `hyle_growable_pool_reset` | trim 全部块；要求无活槽 |
| `hyle_growable_pool_capacity` / `live` | 当前总槽数 / 已借出槽数 |
| `hyle_growable_pool_slot_stride` | 对齐后的实际槽间距，可能大于请求的 slot_size |

typed 宏为 `HYLE_GROWABLE_POOL_CREATE/ACQUIRE/RELEASE`。需要 ctor/dtor 的 C++ 用户使用 `GrowablePool<T>`。

## 7. HugeRegion、错误与进程策略

| 符号 | 说明 |
|---|---|
| `hyle_huge_region_open(&out, bytes, flags, numa_node)` | reserve+commit，按 flags/node 放置；直接返回 VM/NUMA 的结构化 Status |
| `hyle_huge_region_open_local` | 偏好调用线程 home node |
| `hyle_huge_region_open_interleaved` | 跨节点条带 + THP |
| `hyle_huge_region_open_interleaved_nodes(&out, bytes, flags, nodes, count, chunk)` | selected-node/chunk 条带；NULL/0 选择全部，顺序有意义，重复 id 表达权重；保持条带 policy 粒度通常用 `HYLE_HUGE_NONE` |
| `hyle_huge_region_destroy` / `data` / `size` | 生命周期与查询；`open*` 成功只交付有效 handle，因此不导出恒真的 `valid` 镜像 |
| `hyle_huge_region_requested_flags` / `accepted_flags` | 区分请求策略与 OS-facing call 已接受策略 |
| `hyle_huge_region_requested_numa_node` / `numa_placement_applied` | NUMA 请求与 policy 是否被 OS 接受；不是逐页 residency 证明 |
| `hyle_huge_region_interleave_chunk_size` | interleaved region 的实际条带大小；其它 region 为 0 |
| `hyle_enable_allocator_hugepages` | 启用 mimalloc THP / large OS page 策略 |
| `hyle_enable_allocator_numa` | 配置 mimalloc NUMA node 数 |
| `hyle_reserve_huge_1g_pages` | 预留 1GiB 大页；非法 node/溢出返回 OUT_OF_RANGE，mimalloc 无法取得连续 host RAM 时返回 RESOURCE_EXHAUSTED，并保留 native error |

mem 不再维护私有 TLS error code/string。冷路径与全库共用 `hyle_status`；成功 Status 是 8B
零值、零分配，错误消息分配失败时仍保留 code。热分配路径则完全不构造 Status。

flags：`HYLE_HUGE_THP_HINT`、`HYLE_HUGE_PREFAULT`、`HYLE_HUGE_NUMA_BIND`、`HYLE_HUGE_NUMA_INTERLEAVE`；bind/interleave 互斥，interleave 只能搭配 `HYLE_NUMA_ANY`，也不能传给 `open_local`。无 NUMA 约束用 `HYLE_NUMA_ANY`。
`hyle_huge_flags` 固定为 `uint32_t`；NUMA 参数直接复用 platform 的固定宽 `hyle_numa_node`，单 node 参数、
requested-node 查询和 selected-node 数组都使用同一个 type，所以 platform 的
`hyle_numa_nodes` 结果可逐项传入而不需要 `(int)`/`(unsigned)` cast。显式数组只接受非负
实际 node ID，`HYLE_NUMA_ANY` 只用于单 node 参数；数组中出现负值返回
`INVALID_ARGUMENT`。常量仍可用于
`switch`/位运算，但不要把它们当成由编译器自行选宽度的 C enum。这样调用方即使启用
`-fshort-enums`，flags、返回值和 `-1` sentinel 的 ABI 仍与 HYLE 二进制一致。
selected-node 调用前用 platform 的 `hyle_numa_nodes` 取得实际 id，不要从 count 猜 `[0,count)`；
用 `hyle_numa_capabilities` 可预检，多节点 policy 是否成功安装仍以返回 Status 为准；目标节点
压力下 OS 可以回退。`count==0` 选择全部节点，
`chunk==0` 表示一页。`open_interleaved` 是“全部节点 + THP”的便利形状；需要自定义集合或条带
时用 `open_interleaved_nodes`。THP 可能把物理粒度扩大；需保持请求的条带 policy 粒度时传
`HYLE_HUGE_NONE`，仅在 workload 接受时显式叠加 THP/PREFAULT。Linux 极细自定义布局为限制
VMA policy 数可能在 open 内首次触页；需要严格惰性时选择较粗 chunk 或 whole-range policy。

## 8. 常用写法

下面三段分别覆盖 request bulk lifetime、typed owner cache 和 adapter composition；同样代码形状
由 `hyle_mem_c_test` 以纯 C 编译器持续编译执行。

### Request Region

```c
hyle_region *request = NULL;
hyle_status status = hyle_region_create_default(&request, NULL);
if (hyle_status_is_ok(status)) {
  void *scratch = hyle_region_allocate(request, 4096, 64);
  if (scratch != NULL) {
    /* process one request */
    hyle_region_clear(request);
  }
  hyle_region_destroy(request);
}
hyle_status_free(&status);
```

### Typed cached pool

```c
typedef struct { int id; double value; } Node;
hyle_status use_nodes(void) {
  hyle_shared_pool *core = NULL;
  HYLE_C_RETURN_IF_ERROR(
      HYLE_OBJECT_POOL_CREATE_DEFAULT(&core, Node, 1024));

  hyle_shared_pool_cache *local = NULL;
  hyle_status status =
      HYLE_OBJECT_POOL_CACHE_CREATE_DEFAULT(&local, core, 64, 16);
  if (!hyle_status_is_ok(status)) {
    hyle_shared_pool_destroy(core);
    return status;
  }

  Node *node = HYLE_OBJECT_POOL_CACHE_ACQUIRE(local, Node);
  if (node != NULL) {
    node->id = 7; /* C 负责初始化/清理裸槽 */
    HYLE_OBJECT_POOL_CACHE_RELEASE(local, node);
  }
  HYLE_OBJECT_POOL_CACHE_PUBLISH(local);
  hyle_shared_pool_cache_destroy(local);
  hyle_shared_pool_destroy(core);
  return hyle_status_ok();
}
```

### Quota + Stats

```c
hyle_status run_with_budget(void) {
  hyle_stats_allocator *stats = NULL;
  HYLE_C_RETURN_IF_ERROR(hyle_stats_allocator_create_default(&stats, 0));

  hyle_allocator measured = hyle_stats_allocator_as_allocator(stats);
  hyle_quota_allocator *quota = NULL;
  hyle_status status =
      hyle_quota_allocator_create(&quota, measured, 1u << 20);
  if (!hyle_status_is_ok(status)) {
    hyle_stats_allocator_destroy(stats);
    return status;
  }

  hyle_allocator bounded = hyle_quota_allocator_as_allocator(quota);
  void *p = hyle_alloc(&bounded, 4096, 64);
  if (p != NULL) hyle_free(&bounded, p, 4096, 64);
  hyle_quota_allocator_destroy(quota);
  hyle_stats_allocator_destroy(stats);
  return hyle_status_ok();
}
```

这里 Quota 是调用外层：被 quota 拒绝的请求不会进入 Stats backing。C opaque quota control
本身由 `measured` 分配，因此先 destroy quota，再读取最终 balanced snapshot 或 destroy stats。

### 硬容量共享池

```c
hyle_shared_pool* pool = NULL;
hyle_status status = hyle_shared_pool_create_default(&pool, 256, 1024, 0);
if (hyle_status_is_ok(status)) {
  void* block = hyle_shared_pool_allocate(pool);
  if (block != NULL) hyle_shared_pool_deallocate(pool, block);
  hyle_shared_pool_destroy(pool);
}
hyle_status_free(&status);
```

### 显式 cache 的容量交接

```c
hyle_status use_cached_blocks(void) {
  hyle_shared_pool* pool = NULL;
  HYLE_C_RETURN_IF_ERROR(
      hyle_shared_pool_create_default(&pool, 256, 4096, 0));

  hyle_shared_pool_cache* cache = NULL;
  hyle_status status =
      hyle_shared_pool_cache_create_default(&cache, pool, 128, 32);
  if (!hyle_status_is_ok(status)) {
    hyle_shared_pool_destroy(pool);
    return status;
  }

  void* block = hyle_shared_pool_cache_allocate(cache);
  hyle_shared_pool_cache_deallocate(cache, block);
  hyle_shared_pool_cache_publish(cache);
  hyle_shared_pool_cache_destroy(cache);
  hyle_shared_pool_destroy(pool);
  return hyle_status_ok();
}
```

### typed 默认硬池

```c
typedef struct { int a; double b; } Node;
hyle_shared_pool* pool = NULL;
hyle_status status = HYLE_OBJECT_POOL_CREATE_DEFAULT(&pool, Node, 1024);
if (hyle_status_is_ok(status)) {
  Node* n = HYLE_OBJECT_POOL_ACQUIRE(pool, Node);
  if (n != NULL) {
    n->a = 7;
    n->b = 1.5;
    HYLE_OBJECT_POOL_RELEASE(pool, n);
  }
  hyle_shared_pool_destroy(pool);
}
hyle_status_free(&status);
```

## 9. 导出边界

当前 mem C surface 是纯 `extern "C"` 自由函数 + opaque 句柄，没有模块 API 表或版本协商。模板、RAII 与 `std::pmr` 不跨边界：

- `ObjectPool<T>` → 默认硬容量 typed 宏；`ObjectPoolCache<T>` → 显式 cache typed 宏 + publish；
- `GrowablePool<T>` → runtime size/align + typed 宏；
- `Allocator` typed helpers → 裸 `hyle_alloc/free`；
- `QuotaAllocator` / `StatsAllocator` → opaque adapter；统计通过 POD snapshot，泄漏通过 bounded copy。

`tests/mem/c_surface_symbols.txt` 只检查声明、纯 C 引用和 DLL/ELF 导出一致。发布前仍可直接
重塑，不把 manifest 当兼容承诺。当前审查保留能映射真实 C/FFI 场景的 allocator、region、
pool、instrumentation 与 topology 能力族，但不机械镜像方法；opaque `_o` 名隐藏、默认
backing 走 header-inline convenience、默认 alignment 不跨 DLL。mem 冷路径使用共享
`hyle_status`，不再另造 TLS error；FixedRegion 用自然的 3-word C 值类型，复杂的 FixedPool 才保留 opaque
no-heap control。模板 ctor/dtor 与 PMR 仍属于原生 C++ 形状。`FiberStack`/`FiberStackPool`
是 scheduler-owned runtime specialization：sched 的 C ABI 在内部消费它们；没有独立 C
consumer 时不导出可绕过 scheduler 生命周期的 raw stack control。这是明确边界，不是遗漏。
发布后的冻结策略见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)。
