# HYLE C ABI：mem

> **发布面**：`include/hyle/c/mem.h`  
> **实现**：`src/mem/c_mem.cc`（链 `hyle_mem`）  
> **稳定性**：未发布，无 ABI 版本门
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑定本头，不要 include `hyle/mem/*.h`。固定块池只有 `SharedPool` 与 `SharedPoolCached` 两套现行 API；没有兼容别名。

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
| SharedPool | `hyle_shared_pool_create` → `destroy` | **硬容量**；completed free 立即对所有线程可见 |
| SharedPoolCached | `hyle_shared_pool_cached_create` → `destroy` | **软容量**；每线程 TLS cache，跨线程交接前 publish |
| StackAllocator | `hyle_stack_allocator_create` → `destroy` | 自有 slab，单线程 LIFO bump |
| Arena | `hyle_arena_create(pool)` → `destroy` | 借 `SharedPool` 的一个块；pool 必须更久活 |
| GrowablePool | `hyle_growable_pool_create` → `destroy` | 单线程、按需增长的裸槽池 |
| HugeRegion | `hyle_huge_region_open*` → `destroy` | 大页 / NUMA host RAM 区 |
| 进程策略 | `hyle_enable_allocator_*` / `hyle_reserve_huge_1g_pages` | 无句柄，全局软策略 |

**sized-free 合同**：`free` / `deallocate` 必须带回分配请求时相同的 size 与 alignment。pool/arena/stack 不保存每指针尺寸；传错是调用方契约违反。只有 system 分配器提供物理容量查询 `hyle_system_usable_size`。

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
| `hyle_default_alignment()` | `alignof(max_align_t)` |
| `hyle_system_allocator()` | mimalloc-backed 进程级句柄 |
| `hyle_system_usable_size(ptr)` | system 块真实可用字节；NULL→0 |
| `hyle_system_good_size(size)` | 不分配地预览 size class；0→0 |
| `hyle_alloc` / `hyle_free` / `hyle_realloc` | header-inline vtable 转发；alignment==0 使用默认对齐 |

C/C++ 两面通过 trampoline 显式构造，不 reinterpret 布局。

## 3. SharedPool：硬容量共享定长块池

`SharedPool` 没有 TLS 隐藏。`deallocate()` 完成后，块立即进入共享 shard；`allocate()` 在返回 NULL 前用稳定双快照确认所有 shard 在同一公共区间内都为空，避免块跨 shard 移动造成分片假耗尽。

正常路径使用 32-bit tag 的 64-bit CAS。tag 到饱和值的 shard 会**永久**切换到 serialized fallback，并用独立 64-bit change counter 参与空快照校验；tag 绝不回绕，因此长寿命池不会重新接受陈旧 CAS。

| 符号 | 说明 |
|---|---|
| `hyle_shared_pool_create(backing, block_size, block_count, alignment)` | 建硬容量池；alignment==0→默认；资源失败→NULL + `hyle_mem_last_error`；零尺寸、计数非法或乘法溢出为契约违反并 fail-fast |
| `hyle_shared_pool_destroy(p)` | 销毁池；调用方保证无块存活、无并发访问 |
| `hyle_shared_pool_allocate(p)` | 取一块；稳定观察整池为空才返 NULL |
| `hyle_shared_pool_deallocate(p, ptr)` | 立即发布到共享 core；NULL no-op |
| `hyle_shared_pool_reset(p)` | 重建所有 shard；要求无活块且并发已静默 |
| `hyle_shared_pool_block_size` / `capacity` | 查询 |
| `hyle_shared_pool_as_allocator(p)` | alloc=定长 pop、free=push、realloc=NULL；句柄借用 pool |

### 3.1 typed 裸槽宏（默认硬容量）

| 宏 | 说明 |
|---|---|
| `HYLE_OBJECT_POOL_CREATE(backing, T, count)` | 用 `sizeof(T)` / `alignof(T)` 创建 `SharedPool` |
| `HYLE_OBJECT_POOL_ACQUIRE(pool, T)` | 返回裸、未初始化的 `T*` 槽；耗尽→NULL |
| `HYLE_OBJECT_POOL_RELEASE(pool, p)` | 调用方先清理对象，再归还槽 |
| `HYLE_ALIGNOF(T)` | C++/C23 `alignof`、C11 `_Alignof`，旧 C 则为 0 |

C 宏不运行 ctor/dtor。需要构造/析构语义时用 C++ `hyle::mem::ObjectPool<T>`；它默认同样是硬容量。

## 4. SharedPoolCached：软容量 TLS 快路

`SharedPoolCached` 与 `SharedPool` 共用同一个 sharded core，但增加每 `(thread,pool)` 的 TLS freelist：每线程最多 **128 slots**，shared core 与 TLS 之间按 **32-block magazine** 批量转移。同线程 free→alloc 热路零原子。

代价是软容量：A 线程 TLS 中已 free 的块，在 A publish、cache overflow 或线程退出前，B 看不到。容量紧贴工作集、或跨线程 free 后必须立即复用时，应直接使用 `SharedPool`。

| 符号 | 说明 |
|---|---|
| `hyle_shared_pool_cached_create` / `destroy` | 建/销软容量池；资源与契约错误模型同 `SharedPool` |
| `hyle_shared_pool_cached_allocate` / `deallocate` | TLS 快路 + 32-block magazine；其他线程 TLS 中的 free 块不可见 |
| `hyle_shared_pool_cached_publish(p)` | 把**当前线程**的 TLS cache 发布到 shared core |
| `hyle_shared_pool_cached_reset(p)` | 清所有注册 cache 并重建 shard；要求无活块且并发静默 |
| `hyle_shared_pool_cached_block_size` / `capacity` | 查询 |
| `hyle_shared_pool_cached_as_allocator(p)` | 借用型 allocator 视图 |

对应 typed 裸槽宏：

- `HYLE_OBJECT_POOL_CACHED_CREATE`
- `HYLE_OBJECT_POOL_CACHED_ACQUIRE`
- `HYLE_OBJECT_POOL_CACHED_RELEASE`
- `HYLE_OBJECT_POOL_CACHED_PUBLISH`

C++ 对应类型是 `ObjectPoolCached<T>`；它是软容量，不是默认对象池。

## 5. Arena 与 StackAllocator

### 5.1 Arena：借一个硬池块做 bump

| 符号 | 说明 |
|---|---|
| `hyle_arena_create(hyle_shared_pool_o* pool)` | 从硬容量 `SharedPool` 借一块；pool 为空、wrapper OOM 或池耗尽→NULL + last-error |
| `hyle_arena_destroy(a)` | 把整块还给 pool |
| `hyle_arena_allocate(a, bytes, alignment)` | 单线程 pointer bump；放不下→NULL |
| `hyle_arena_reset(a)` | 游标回块头；要求此前切片不再使用 |
| `hyle_arena_used` / `remaining` / `block_size` | 查询 |
| `hyle_arena_as_allocator(a)` | alloc=bump、free=no-op、realloc=bump+copy |

Arena 只借 `SharedPool`。单块 borrow 不从 TLS cache 获益，硬池的立即可见容量使生命周期更直接。

### 5.2 StackAllocator：自有 slab 的 LIFO bump

| 符号 | 说明 |
|---|---|
| `hyle_stack_allocator_create(backing, size)` / `destroy` | 自持 slab；资源失败→NULL + last-error；size==0 fail-fast |
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
| `hyle_huge_region_destroy` / `valid` / `data` / `size` | 生命周期与查询 |
| `hyle_mem_last_error()` | 当前线程最近一次 mem 资源失败原因；成功工厂/acquire 清旧错误 |
| `hyle_enable_allocator_hugepages` | 启用 mimalloc THP / large OS page 策略 |
| `hyle_enable_allocator_numa` | 配置 mimalloc NUMA node 数 |
| `hyle_reserve_huge_1g_pages` | 预留 1GiB 大页；无权限/连续 host RAM 时软失败返 0 |

flags：`HYLE_HUGE_THP_HINT`、`HYLE_HUGE_PREFAULT`、`HYLE_HUGE_NUMA_BIND`、`HYLE_HUGE_NUMA_INTERLEAVE`；无 NUMA 约束用 `HYLE_MEM_NUMA_ANY`。

## 8. 常用写法

### 硬容量共享池

```c
hyle_allocator sys = hyle_system_allocator();
hyle_shared_pool_o* pool = hyle_shared_pool_create(sys, 256, 1024, 0);
void* block = hyle_shared_pool_allocate(pool);
if (block != NULL) hyle_shared_pool_deallocate(pool, block);
hyle_shared_pool_destroy(pool);
```

### 软容量 cached 池的跨线程交接

```c
hyle_shared_pool_cached_o* pool =
    hyle_shared_pool_cached_create(sys, 256, 4096, 0);
void* block = hyle_shared_pool_cached_allocate(pool);
hyle_shared_pool_cached_deallocate(pool, block);
hyle_shared_pool_cached_publish(pool); /* 当前线程已 free 的块现在对其他线程可见 */
hyle_shared_pool_cached_destroy(pool);
```

### typed 默认硬池

```c
typedef struct { int a; double b; } Node;
hyle_shared_pool_o* pool = HYLE_OBJECT_POOL_CREATE(sys, Node, 1024);
Node* n = HYLE_OBJECT_POOL_ACQUIRE(pool, Node);
if (n != NULL) {
  n->a = 7;
  n->b = 1.5;
  HYLE_OBJECT_POOL_RELEASE(pool, n);
}
hyle_shared_pool_destroy(pool);
```

## 9. 导出边界

mem C ABI 是纯 `extern "C"` 自由函数 + opaque 句柄，没有模块 API 表或版本协商。模板、RAII 与 `std::pmr` 不跨边界：

- `ObjectPool<T>` → 默认硬容量 typed 宏；`ObjectPoolCached<T>` → cached typed 宏 + publish；
- `GrowablePool<T>` → runtime size/align + typed 宏；
- `Allocator` typed helpers → 裸 `hyle_alloc/free`；
- `StatsAllocator` / `LeakTracker` 当前是内部 debug/test 设施，不是已承诺产品对象。

发布前 C ABI 可自由演进；发布后的冻结策略见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)。
