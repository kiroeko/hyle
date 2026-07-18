# HYLE C ABI：mem

> **发布面**：`include/hyle/c/mem.h`  
> **实现**：`src/mem/c_mem.cc`（链 `hyle_mem`）  
> **ABI 版本**：`HYLE_MEM_C_ABI_VERSION`（当前 **1**）  
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑本头。不要 include `hyle/mem/*.h`。

---

## 1. 链什么

```text
#include "hyle/c/mem.h"
// link: hyle_mem
```

纯 C 验收：`hyle_mem_c_test`。

---

## 2. 子系统与所有权

| 块 | 入口 | 所有权 |
|---|---|---|
| Allocator 句柄 | `hyle_allocator`（`{alloc, free, realloc, state}`） | 值类型 POD；随处按值传 |
| system 分配器 | `hyle_system_allocator` / `hyle_system_usable_size` / `hyle_system_good_size` | 进程级；`alloc` 配对 `free`（**带 size**） |
| MemoryArena | `hyle_memory_arena_create` → `destroy` | create→destroy |
| StackAllocator | `hyle_stack_allocator_create` → `destroy` | create→destroy |
| Arena | `hyle_arena_create(pool)` → `destroy` | create→destroy；借 pool 的块，pool 必须更久活 |
| HugeRegion | `hyle_huge_region_open*` → `destroy` | open→destroy |
| 进程级策略 | `hyle_enable_allocator_hugepages` / `numa` / `reserve_huge_1g_pages` | 无句柄，全局软策略 |
| 表 | `hyle_mem_api_get` | 可选 FFI 函数指针包 |

**sized-free 合同**：`free` / `deallocate` **携带 size 与 alignment**（对标 Rust `GlobalAlloc` / C++17 sized delete / jemalloc `sdallocx`）。传入的 size 必须等于分配时的请求——arena/pool/stack **不存每指针 size**，传错即 UB。唯一的物理容量查询是 `hyle_system_usable_size`（仅 system 分配器有，mimalloc 知道 size-class）。

---

## 3. 导出符号 1:1

与 `include/hyle/c/mem.h` **逐符号对齐**。改头必须同步改本节。

### 3.1 Allocator vtable

`hyle_allocator` 是**天然 C ABI 形状**（3 个 C 函数指针 + `state`），可直接调，也可交给本模块的句柄工厂当 backing。

```c
typedef struct hyle_allocator {
  void* (*alloc)(void* state, size_t size, size_t alignment);
  void  (*free)(void* state, void* ptr, size_t size, size_t alignment);
  void* (*realloc)(void* state, void* ptr, size_t old_size, size_t new_size, size_t alignment);
  void* state;
} hyle_allocator;
```

| 符号 | 说明 |
|---|---|
| `hyle_default_alignment()` | 默认对齐（`alignof(max_align_t)`）；便利 helper 传 0 时用它 |
| `hyle_system_allocator()` | mimalloc-backed 进程级句柄（同 C++ `system_allocator()`） |
| `hyle_system_usable_size(ptr)` | system 块的真实可用字节（≥ 请求）；NULL→0 |
| `hyle_system_good_size(size)` | 不分配预览 size-class 取整（nallocx）；0→0 |
| `hyle_alloc(a, sz, align)` | **inline**：经句柄调；`a`/`a->alloc` 为空→NULL；align==0→默认 |
| `hyle_free(a, p, sz, align)` | **inline**：经句柄调；空句柄安全 no-op |
| `hyle_realloc(a, p, osz, nsz, align)` | **inline**：经句柄调 |

> 桥接**从不 reinterpret**：C→C++ 经 noexcept trampoline 转发到调用方存下的句柄；C++→C 逐字段填（noexcept 函数指针隐式转普通指针）。`hyle_allocator` 与 C++ `mem::Allocator` 恰好同形是设计使然，但代码走安全构造。

### 3.2 MemoryArena —— 分片无锁定长块池

| 符号 | 说明 |
|---|---|
| `hyle_memory_arena_create(backing, block_size, block_count, alignment)` | alignment==0→默认；backing 三指针缺一→NULL（边界校验）；`block_size`/`block_count` 为 0 或 slab 分配失败→C++ ctor fail-fast abort（契约违反/资源耗尽，同 C++）；wrapper OOM→NULL |
| `hyle_memory_arena_destroy(a)` | |
| `hyle_memory_arena_allocate(a)` | pop 一块；**整池**耗尽→NULL |
| `hyle_memory_arena_deallocate(a, p)` | 归还；p==NULL no-op |
| `hyle_memory_arena_reset(a)` | 全部块归还。合同：**无块存活**时才可调 |
| `hyle_memory_arena_block_size(a)` / `capacity(a)` | 查询 |
| `hyle_memory_arena_as_allocator(a)` | vtable 视图（alloc=pop 定长块，free=push，realloc=NULL）；句柄归 a，activ至 destroy |

### 3.3 StackAllocator —— LIFO bump

| 符号 | 说明 |
|---|---|
| `hyle_stack_allocator_create(backing, size)` | 自持 size 字节 slab；backing 无效→NULL；size==0→C++ 侧 fail-fast 断言 |
| `hyle_stack_allocator_destroy(s)` | |
| `hyle_stack_allocator_allocate(s, bytes, alignment)` | bump；alignment==0→默认；放不下→NULL（无堆回退） |
| `hyle_stack_allocator_reset(s)` | 全部释放 |
| `hyle_stack_allocator_mark(s)` | 存游标 |
| `hyle_stack_allocator_rewind(s, marker)` | LIFO 回退到 mark |
| `hyle_stack_allocator_used` / `capacity` / `remaining` | 查询 |
| `hyle_stack_allocator_as_allocator(s)` | vtable 视图（alloc=bump，free=no-op，realloc=bump+copy） |

### 3.4 Arena —— 借一块 MemoryArena 块做 bump

| 符号 | 说明 |
|---|---|
| `hyle_arena_create(pool)` | 借 pool 一块；pool 须更久活；pool==NULL→NULL；**池耗尽→C++ ctor fail-fast abort**（借不到块=契约违反，同 C++ `Arena`）；wrapper OOM→NULL |
| `hyle_arena_destroy(a)` | 把块还给池 |
| `hyle_arena_allocate(a, bytes, alignment)` | bump；放不下→NULL |
| `hyle_arena_reset(a)` | 从块头重用 |
| `hyle_arena_used` / `remaining` / `block_size` | 查询 |
| `hyle_arena_as_allocator(a)` | vtable 视图（alloc=bump，free=no-op，realloc=bump+copy） |

### 3.5 HugeRegion —— 大页 / NUMA 区

soft 提示，绝不影响正确性。flags 用 `HYLE_HUGE_*` 位。

| 符号 | 说明 |
|---|---|
| `HYLE_HUGE_NONE` / `THP_HINT` / `PREFAULT` / `NUMA_BIND` / `NUMA_INTERLEAVE` | flags 位 |
| `HYLE_MEM_NUMA_ANY` | `(-1)`：无 NUMA 约束 |
| `hyle_huge_region_open(bytes, flags, numa_node)` | reserve+commit（大页对齐）按 flags/node 放置；失败→NULL（见 `hyle_vm_last_error`） |
| `hyle_huge_region_open_local(bytes, flags)` | 偏好调用线程 home 节点 |
| `hyle_huge_region_open_interleaved(bytes)` | 全节点条带 + THP |
| `hyle_huge_region_destroy(r)` | NULL 安全 |
| `hyle_huge_region_valid` / `data` / `size` | 查询 |

### 3.6 进程级 mimalloc 策略（自由函数）

| 符号 | 说明 |
|---|---|
| `hyle_enable_allocator_hugepages(allow_thp, allow_large_os_pages)` | 进程启动期调；THP + 大 OS 页 |
| `hyle_enable_allocator_numa(max_nodes)` | max_nodes==0→全节点 |
| `hyle_reserve_huge_1g_pages(pages, numa_node, timeout_ms)` | 预留 1GiB 大页；ANY→交错；成功/`pages==0`→1；软失败（无权限/无连续 RAM）→0 |

### 3.7 版本 / API 表

| 符号 | 说明 |
|---|---|
| `hyle_mem_c_abi_version()` | 等于宏 |
| `hyle_mem_api_get(out, expected_abi)` | 0 成功；-1 失败（NULL 或版本不符） |

`hyle_mem_api` 字段 = **全部 DLL 自由函数**：allocator×4 + memory_arena×8 + stack×10 + arena×8 + huge×7 + policy×3 + `c_abi_version`，外加 `abi_version` 整型。破坏性变更升版本、新字段追加表尾（保前缀布局）。

**不进表**：inline 便利 `hyle_alloc` / `hyle_free` / `hyle_realloc`（无 DLL 符号）。

### 3.8 有意不导出（仅 C++）

| C++ | 原因 |
|---|---|
| `ObjectPool<T>` / `GrowablePool<T>` / `pmr::*` | 模板，C 无泛型。C 侧用 `MemoryArena`（定长）+ 自己的 typed 包装 |
| `Allocator` typed 层（`allocate_n` / `construct` / `make_unique`） | C++ 模板便利，C 侧用裸 `alloc` + 手动 size |
| `AllocStats` / `LeakTracker` / `StatsAllocator` | 有状态调试类，C ABI 化成本高且无 FFI 需求（同 error 的 `ErrorRecord`） |

---

## 4. 常用写法

### 直接用 system 分配器

```c
hyle_allocator a = hyle_system_allocator();
void* p = hyle_alloc(&a, 1024, 64);   /* 64 对齐 */
/* ... */
hyle_free(&a, p, 1024, 64);           /* 带回同样的 size / align */
```

### 定长块池

```c
hyle_allocator sys = hyle_system_allocator();
hyle_memory_arena_o* pool = hyle_memory_arena_create(sys, 256, 1024, 0);
void* blk = hyle_memory_arena_allocate(pool);   /* NULL = 耗尽 */
hyle_memory_arena_deallocate(pool, blk);
hyle_memory_arena_destroy(pool);
```

### 帧 bump（LIFO）

```c
hyle_stack_allocator_o* sa = hyle_stack_allocator_create(sys, 1 << 20);
void* m = hyle_stack_allocator_mark(sa);
void* scratch = hyle_stack_allocator_allocate(sa, 4096, 16);
/* ... 用完一帧 ... */
hyle_stack_allocator_rewind(sa, m);
hyle_stack_allocator_destroy(sa);
```

### 大页区

```c
hyle_huge_region_o* r =
    hyle_huge_region_open(64u << 20, HYLE_HUGE_THP_HINT, HYLE_MEM_NUMA_ANY);
void* d = hyle_huge_region_data(r);
hyle_huge_region_destroy(r);
```

### 用自己的 C 分配器当 backing

```c
static void* my_alloc(void* st, size_t n, size_t a) { /* ... */ }
static void  my_free(void* st, void* p, size_t n, size_t a) { /* ... */ }
static void* my_realloc(void* st, void* p, size_t o, size_t n, size_t a) { /* ... */ }

hyle_allocator mine = { my_alloc, my_free, my_realloc, my_state };
hyle_memory_arena_o* pool = hyle_memory_arena_create(mine, 512, 64, 0);
```

---

## 5. 版本

- v1：allocator vtable（`hyle_allocator` + system + usable/good size）、MemoryArena / StackAllocator / Arena / HugeRegion 不透明句柄、进程级 hugepage/NUMA 策略、`hyle_mem_api` 表 + 版本门。
- 破坏性变更升 `HYLE_MEM_C_ABI_VERSION`；新字段追加 `hyle_mem_api` 表尾，旧调用方前缀读取不受影响。

---

## 6. 验收

```bash
cmake --build <build> --target hyle_mem_c_test
./hyle_mem_c_test   # all checks passed
```
