# HYLE mem C surface

> **发布面**：`include/hyle/c/mem.h`  
> **实现**：`src/mem/c_mem.cc`（链 `hyle_mem`）  
> **稳定性**：首次产品形状审查已完成；正式发布前仍可改，当前 manifest 不承诺 ABI 兼容
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑定本头，不要 include `hyle/mem/*.h`。有 `backing` 参数的 opaque
control 与内部 storage 都由该 backing 分配。FixedRegion/FixedPool 同时提供 caller-owned
`*_storage + *_init`（control/payload 都不触堆）和较方便的 `*_create`（control 使用 HYLE
system allocator）；payload 始终只来自调用方 storage，绝不隐藏 fallback。
所有 opaque handle 对外使用 `hyle_region *`、`hyle_shared_pool *` 等产品名；尾缀 `_o`
只存在于不可解引用的内部 struct tag，不进入用户变量和函数签名。

## 1. 链接与所有权

```c
#include "hyle/c/mem.h"
// link: hyle_mem
```

纯 C 验收目标：`hyle_mem_c_test`。

| 块 | 入口 | 所有权 / 容量合同 |
|---|---|---|
| Allocator 句柄 | `hyle_allocator` | POD 值类型；按值传 |
| system 分配器 | `hyle_system_allocator` | 进程级；alloc/free 携带相同 size/alignment |
| QuotaAllocator | `hyle_quota_allocator_create` → `destroy` | 任意 backing 的严格 requested live-byte 上限 |
| Region | `hyle_region_create` → `destroy` | owned growable bump；clear 保留普通 blocks |
| FixedRegion | `hyle_fixed_region_init(control,storage)` 或 `create` → `destroy` | external bump；init 严格 no-heap |
| FixedPool | `hyle_fixed_pool_init(control,storage)` 或 `create` → `destroy` | external variable-size alloc/free；init 严格 no-heap |
| SharedPool | `hyle_shared_pool_create` → `destroy` | **硬容量**；completed free 立即对所有线程可见 |
| SharedPoolCache | `hyle_shared_pool_cache_create(pool, metadata, ...)` → `destroy` | **软视图**；显式 owner/capacity/batch，publish 后回 hard core |
| StackAllocator | `hyle_stack_allocator_create` → `destroy` | 自有 slab，单线程 LIFO bump |
| PooledRegion | `hyle_pooled_region_create(pool)` → `destroy` | 借 `SharedPool` 一个块 |
| GrowablePool | `hyle_growable_pool_create` → `destroy` | 单线程、按需增长的裸槽池 |
| HugeRegion | `hyle_huge_region_open*` → `destroy` | 大页 / NUMA host RAM 区 |
| 进程策略 | `hyle_enable_allocator_*` / `hyle_reserve_huge_1g_pages` | 无句柄，全局软策略 |

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
| `hyle_alloc` / `hyle_free` / `hyle_realloc` | header-inline vtable 转发；alignment==0 使用默认对齐 |

C/C++ 两面通过 trampoline 显式构造，不 reinterpret 布局。
`hyle_alloc/free/realloc(..., alignment=0)` 在头内直接折叠默认对齐，不再先跨 DLL 查询。

拥有 backing 的构造器同时提供 header-inline `*_create_default(...)`，等价于传
`hyle_system_allocator()`；常用用户不必为了默认堆手写 allocator injection。显式
`create(backing, ...)` 继续用于 quota/region/pool 组合与故障注入。

### 2.1 QuotaAllocator

`hyle_quota_allocator_as_allocator()` 返回借用 wrapper 的 allocator。并发分配先原子 reserve，
成功 live bytes 不会超过 limit；backing OOM 回滚。destroy 前必须释放全部 allocation 并
静默并发访问。`used/peak/limit` 都按 requested bytes 计量。

### 2.2 Region、FixedRegion 与 FixedPool

- `hyle_region_create` 的 options 以 `HYLE_REGION_OPTIONS_INIT` 初始化，并带
  `struct_size`。`clear` 保留普通 blocks、释放 dedicated large blocks；`release` 归还全部。
- `hyle_fixed_region_init(&control,storage,size)` 是严格 no-heap bulk-reset bump；
  `create` 只是为愿意接受 system control allocation 的用户保留的便利工厂。旧切片在 clear 后全部失效。
- `hyle_fixed_pool_init(&control,storage,size)` 的 control/payload 都由调用方持有；`create` 是
  control-owning 便利工厂。两者都支持 individual allocate/free/realloc。reset 前必须
  无 live allocation；foreign/double return 是合同错误并 fail-fast。

三者都单线程。它们的 `as_allocator` 句柄借用 opaque object；destroy 后失效。

```c
unsigned char payload[4096];
hyle_fixed_pool_storage control;
hyle_fixed_pool *pool =
    hyle_fixed_pool_init(&control, payload, sizeof payload);
/* allocate/free ... */
hyle_fixed_pool_destroy(pool); /* 不释放 control 或 payload */
```

## 3. SharedPool：硬容量共享定长块池

`SharedPool` 没有 TLS 隐藏。`deallocate()` 完成后，块立即进入共享 shard；`allocate()` 在返回 NULL 前用稳定双快照确认所有 shard 在同一公共区间内都为空，避免块跨 shard 移动造成分片假耗尽。

正常路径使用 32-bit tag 的 64-bit CAS。tag 到饱和值的 shard 会**永久**切换到 serialized fallback，并用独立 64-bit change counter 参与空快照校验；tag 绝不回绕，因此长寿命池不会重新接受陈旧 CAS。

| 符号 | 说明 |
|---|---|
| `hyle_shared_pool_create(backing, block_size, block_count, alignment)` | 建硬容量池；alignment==0→默认；资源失败或非法/溢出 layout→NULL + `hyle_mem_last_error` |
| `hyle_shared_pool_destroy(p)` | 销毁池；调用方保证无块存活、无并发访问 |
| `hyle_shared_pool_allocate(p)` | 取一块；稳定观察整池为空才返 NULL |
| `hyle_shared_pool_deallocate(p, ptr)` | 立即发布到共享 core；NULL no-op |
| `hyle_shared_pool_reset(p)` | 重建所有 shard；要求无活块且并发已静默 |
| `hyle_shared_pool_block_size` / `capacity` | 查询 |
| `hyle_shared_pool_as_allocator(p)` | alloc=定长 pop、free=push；realloc 在 block 内原地 resize，超限失败且保留旧块，size=0 归还；句柄借用 pool |

### 3.1 typed 裸槽宏（默认硬容量）

| 宏 | 说明 |
|---|---|
| `HYLE_OBJECT_POOL_CREATE(backing, T, count)` | 用 `sizeof(T)` / `alignof(T)` 创建 `SharedPool` |
| `HYLE_OBJECT_POOL_ACQUIRE(pool, T)` | 返回裸、未初始化的 `T*` 槽；耗尽→NULL |
| `HYLE_OBJECT_POOL_RELEASE(pool, p)` | 调用方先清理对象，再归还槽 |
| `HYLE_ALIGNOF(T)` | C++/C23 `alignof`、C11 `_Alignof`，旧 C 则为 0 |

C 宏不运行 ctor/dtor。需要构造/析构语义时用 C++ `hyle::mem::ObjectPool<T>`；它默认同样是硬容量。

## 4. SharedPoolCache：显式 owner-local 快路

`SharedPoolCache` 附着到已有的 hard `SharedPool`。cache handle 与 `uint32_t` index array 来自显式 metadata allocator；capacity 与 transfer batch 都是构造参数。一个 cache 同时只由一个 owner 使用，不再有隐藏 TLS node 或跨 DLL registry。

代价是软容量：cache 中已 free 的块在 `publish()`/destroy 前，hard core 和其他 cache 看不到。容量紧贴工作集、或跨线程 free 后必须立即复用时，应直接使用 `SharedPool`。

| 符号 | 说明 |
|---|---|
| `hyle_shared_pool_cache_create(pool, metadata, capacity, batch)` / `destroy` | 绑定 hard pool；destroy 自动 publish；pool 必须更长寿 |
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
| `hyle_pooled_region_create(hyle_shared_pool* pool)` | 从硬容量 `SharedPool` 借一块；pool 为空、control OOM 或池耗尽→NULL + last-error |
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
| `hyle_stack_allocator_create(backing, size)` / `destroy` | 自持 slab；资源失败或 size==0→NULL + last-error |
| `hyle_stack_allocator_allocate(s, bytes, alignment)` | 单线程 pointer bump；放不下→NULL |
| `hyle_stack_allocator_mark` / `rewind` | 保存并恢复 LIFO 游标 |
| `hyle_stack_allocator_reset` | 整体复用 |
| `hyle_stack_allocator_used` / `capacity` / `remaining` | 查询 |
| `hyle_stack_allocator_as_allocator` | alloc=bump、free=no-op、realloc=bump+copy |

## 6. GrowablePool：单线程可增长裸槽池

| 符号 | 说明 |
|---|---|
| `hyle_growable_pool_create(backing, slot_size, slot_align, initial_slots)` | 首块惰性分配，此后几何增长；资源失败→NULL + last-error |
| `hyle_growable_pool_destroy` | 释放全部块 |
| `hyle_growable_pool_acquire` / `release` | 取/还裸槽，不运行 ctor/dtor |
| `hyle_growable_pool_reset` | trim 全部块；要求无活槽 |
| `hyle_growable_pool_capacity` / `live` | 查询 |

typed 宏为 `HYLE_GROWABLE_POOL_CREATE/ACQUIRE/RELEASE`。需要 ctor/dtor 的 C++ 用户使用 `GrowablePool<T>`。

## 7. HugeRegion、诊断与进程策略

| 符号 | 说明 |
|---|---|
| `hyle_huge_region_open(bytes, flags, numa_node)` | reserve+commit，按 flags/node 放置；失败→NULL |
| `hyle_huge_region_open_local` | 偏好调用线程 home node |
| `hyle_huge_region_open_interleaved` | 跨节点条带 + THP |
| `hyle_huge_region_destroy` / `data` / `size` | 生命周期与查询；`open*` 只返回有效 handle 或 NULL，因此不导出恒真的 `valid` 镜像 |
| `hyle_huge_region_requested_flags` / `accepted_flags` | 区分请求策略与 OS-facing call 已接受策略 |
| `hyle_huge_region_requested_numa_node` / `numa_placement_applied` | NUMA 请求与实际 placement 结果 |
| `hyle_mem_last_error_code()` | factory/policy TLS typed code：NONE / INVALID_ARGUMENT / OUT_OF_MEMORY / EXHAUSTED / PLATFORM；供控制流使用，hot allocate/pop 不为清码写 TLS |
| `hyle_mem_last_error()` | TLS 诊断字符串；不用于程序解析。mem 工厂不污染 platform VM error，HugeRegion 捕获 VM 原因后恢复 platform 通道 |
| `hyle_enable_allocator_hugepages` | 启用 mimalloc THP / large OS page 策略 |
| `hyle_enable_allocator_numa` | 配置 mimalloc NUMA node 数 |
| `hyle_reserve_huge_1g_pages` | 预留 1GiB 大页；无权限/连续 host RAM 时软失败返 0 |

flags：`HYLE_HUGE_THP_HINT`、`HYLE_HUGE_PREFAULT`、`HYLE_HUGE_NUMA_BIND`、`HYLE_HUGE_NUMA_INTERLEAVE`；无 NUMA 约束用 `HYLE_MEM_NUMA_ANY`。

## 8. 常用写法

### 硬容量共享池

```c
hyle_shared_pool* pool = hyle_shared_pool_create_default(256, 1024, 0);
void* block = hyle_shared_pool_allocate(pool);
if (block != NULL) hyle_shared_pool_deallocate(pool, block);
hyle_shared_pool_destroy(pool);
```

### 显式 cache 的容量交接

```c
hyle_shared_pool* pool = hyle_shared_pool_create_default(256, 4096, 0);
hyle_shared_pool_cache* cache =
    hyle_shared_pool_cache_create_default(pool, 128, 32);
void* block = hyle_shared_pool_cache_allocate(cache);
hyle_shared_pool_cache_deallocate(cache, block);
hyle_shared_pool_cache_publish(cache); /* 本地 free 块现在对其他 owner 可见 */
hyle_shared_pool_cache_destroy(cache);
hyle_shared_pool_destroy(pool);
```

### typed 默认硬池

```c
typedef struct { int a; double b; } Node;
hyle_shared_pool* pool = HYLE_OBJECT_POOL_CREATE_DEFAULT(Node, 1024);
Node* n = HYLE_OBJECT_POOL_ACQUIRE(pool, Node);
if (n != NULL) {
  n->a = 7;
  n->b = 1.5;
  HYLE_OBJECT_POOL_RELEASE(pool, n);
}
hyle_shared_pool_destroy(pool);
```

## 9. 导出边界

当前 mem C surface 是纯 `extern "C"` 自由函数 + opaque 句柄，没有模块 API 表或版本协商。模板、RAII 与 `std::pmr` 不跨边界：

- `ObjectPool<T>` → 默认硬容量 typed 宏；`ObjectPoolCache<T>` → 显式 cache typed 宏 + publish；
- `GrowablePool<T>` → runtime size/align + typed 宏；
- `Allocator` typed helpers → 裸 `hyle_alloc/free`；
- `QuotaAllocator` → opaque product adapter；`StatsAllocator` / `LeakTracker` 当前仍是 C++ 诊断设施。

`tests/mem/c_surface_symbols.txt` 只检查声明、纯 C 引用和 DLL/ELF 导出一致。发布前仍可直接
重塑，不把 manifest 当兼容承诺。首次产品审查决定：保留能映射真实 C/FFI 场景的 allocator、
region、pool 与 topology 能力族，但不机械镜像方法；opaque `_o` 名隐藏、默认 backing 走
header-inline convenience、默认 alignment 不跨 DLL、恒真的 HugeRegion `valid` 删除。mem
使用独立 typed error code + 诊断字符串；external FixedRegion/FixedPool 提供 strict no-heap
control；`StatsAllocator`/`LeakTracker` 没有真实 C 消费者，继续只留 C++。
发布后的冻结策略见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)。
