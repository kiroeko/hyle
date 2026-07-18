# HYLE C ABI：mem

> **发布面**：`include/hyle/c/mem.h`  
> **实现**：`src/mem/c_mem.cc`（链 `hyle_mem`）  
> **稳定性**：未发布，无 ABI 版本门（见 §3.8）  
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
| `hyle_memory_arena_create(backing, block_size, block_count, alignment)` | alignment==0→默认；**资源失败**（backing 三指针缺一 / wrapper OOM / slab·link·heads OOM）→NULL + 设 `hyle_mem_last_error`（followup #2 档 B，不再 abort）；**契约违反**（`block_size`/`block_count` 为 0，或 `block_size`×`block_count` 溢出 `size_t`）→C++ ctor fail-fast abort（同 C++） |
| `hyle_memory_arena_destroy(a)` | |
| `hyle_memory_arena_allocate(a)` | pop 一块；**整池**耗尽→NULL |
| `hyle_memory_arena_deallocate(a, p)` | 归还；p==NULL no-op |
| `hyle_memory_arena_reset(a)` | 全部块归还。合同：**无块存活**时才可调 |
| `hyle_memory_arena_block_size(a)` / `capacity(a)` | 查询 |
| `hyle_memory_arena_as_allocator(a)` | vtable 视图（alloc=pop 定长块，free=push，realloc=NULL）；句柄归 a，activ至 destroy |

**typed 对象池便利宏**（header-inline，无 DLL 符号）：`block_size == sizeof(T)` 的 MemoryArena 就是 T 的定长对象池，这几个宏只是替你填 `sizeof`/`alignof` 并 cast。

| 宏 | 说明 |
|---|---|
| `HYLE_OBJECT_POOL_CREATE(backing, T, count)` | = `hyle_memory_arena_create(backing, sizeof(T), count, alignof(T))` |
| `HYLE_OBJECT_POOL_ACQUIRE(pool, T)` | 取一个 `T*` 裸槽（耗尽→NULL） |
| `HYLE_OBJECT_POOL_RELEASE(pool, p)` | 归还槽 |
| `HYLE_ALIGNOF(T)` | 内部用：C++/C23 `alignof`、C11 `_Alignof`、否则 0（create 归一到默认对齐） |

**裸槽 + O(1)，构造留调用方**（对齐 ggml：分配器只给内存，`allocate` 热路径无构造回调——跨 DLL 的间接 `init()` 调用会给每次 pop 加分派开销，正是三函数指针 vtable 要避开的）。C 无 ctor/dtor，自己初始化返回的槽、release 前自己清理。**要真 construct/destroy 语义（create 跑 ctor、destroy 跑 ~T）用 C++ `hyle::mem::ObjectPool<T>`**——那是 C++ 专有便利，不是本 C ABI 的缺口。

### 3.3 StackAllocator —— LIFO bump

| 符号 | 说明 |
|---|---|
| `hyle_stack_allocator_create(backing, size)` | 自持 size 字节 slab；**资源失败**（backing 无效 / wrapper OOM / slab OOM）→NULL + 设 `hyle_mem_last_error`（followup #2 档 B，不再 abort）；size==0→C++ 侧 fail-fast 断言（契约违反） |
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
| `hyle_arena_create(pool)` | 借 pool 一块；pool 须更久活；**资源失败**（pool==NULL / wrapper OOM / **池耗尽借不到块**）→NULL + 设 `hyle_mem_last_error`（followup #2：池耗尽是资源失败非契约违反，C++ ctor 不再 abort） |
| `hyle_arena_destroy(a)` | 把块还给池 |
| `hyle_arena_allocate(a, bytes, alignment)` | bump；放不下→NULL |
| `hyle_arena_reset(a)` | 从块头重用 |
| `hyle_arena_used` / `remaining` / `block_size` | 查询 |
| `hyle_arena_as_allocator(a)` | vtable 视图（alloc=bump，free=no-op，realloc=bump+copy） |

### 3.5 GrowablePool —— 单线程可增长裸槽池

C++ `GrowablePool<T>` **类型擦除导出**（与 `ObjectPool<T>`↔`MemoryArena` 同构：非模板运行时核心 `detail::GrowablePoolCore` + 模板薄包装；C ABI 导出的是**核心**——裸槽 acquire/release，无 ctor/dtor）。不同于 `hyle_memory_arena_*` 定长，它**按需几何增长**——不"满"，除非 backing 自己 OOM。**单线程**（无原子，uncontended 更快；勿跨线程共享）。

| 符号 | 说明 |
|---|---|
| `hyle_growable_pool_create(backing, slot_size, slot_align, initial_slots)` | slot_align==0→默认；`initial_slots` 首块槽数（0→1），之后块几何翻倍；**资源失败**（backing 无效 / wrapper OOM）→NULL + 设 `hyle_mem_last_error`；`slot_size`==0→C++ 核心 fail-fast abort（契约违反）。首块 backing 分配是懒的（首次 acquire 才取），故 create 无 slab-OOM 分支 |
| `hyle_growable_pool_destroy(p)` | |
| `hyle_growable_pool_acquire(p)` | 取一个**裸未初始化**槽（复用 free-list，否则块内 bump，否则增长）；增长时 backing OOM→NULL（graceful，同 `ObjectPool`；acquire 不设 last-error） |
| `hyle_growable_pool_release(p, slot)` | 归还槽到 free-list，**不跑析构**；slot==NULL no-op |
| `hyle_growable_pool_reset(p)` | 把所有块还给 backing（trim）。合同：**无槽存活**时才可调 |
| `hyle_growable_pool_capacity(p)` | 当前所有块的总槽数（几何增长） |
| `hyle_growable_pool_live(p)` | 已取未还的槽数 |

**typed 便利宏**（header-inline，无 DLL 符号）：`slot_size == sizeof(T)` 的 GrowablePool 就是 T 的可增长裸槽池。

| 宏 | 说明 |
|---|---|
| `HYLE_GROWABLE_POOL_CREATE(backing, T, initial_slots)` | = `hyle_growable_pool_create(backing, sizeof(T), alignof(T), initial_slots)` |
| `HYLE_GROWABLE_POOL_ACQUIRE(pool, T)` | 取一个裸 `T*` 槽（未初始化） |
| `HYLE_GROWABLE_POOL_RELEASE(pool, p)` | 归还槽 |

**裸槽模型，同 object pool**（`HYLE_OBJECT_POOL_*`，§3.2）：acquire 给未构造 storage、release 不跑析构，C 用户自己 placement-init 返回的槽、release 前自己清理。**要 create 跑 ctor / destroy 跑 ~T 用 C++ `hyle::mem::GrowablePool<T>`**——那是 C++ 专有便利，不是本 C ABI 的缺口。

### 3.6 HugeRegion —— 大页 / NUMA 区

soft 提示，绝不影响正确性。flags 用 `HYLE_HUGE_*` 位。

| 符号 | 说明 |
|---|---|
| `hyle_huge_flags`：`HYLE_HUGE_NONE` / `THP_HINT` / `PREFAULT` / `NUMA_BIND` / `NUMA_INTERLEAVE` | 具名 enum；flags 位 |
| `hyle_mem_numa_node`：`HYLE_MEM_NUMA_ANY` | 具名 enum；`(-1)` 无 NUMA 约束 |
| `hyle_huge_region_open(bytes, flags, numa_node)` | reserve+commit（大页对齐）按 flags/node 放置；失败→NULL（见 `hyle_mem_last_error`） |
| `hyle_huge_region_open_local(bytes, flags)` | 偏好调用线程 home 节点 |
| `hyle_huge_region_open_interleaved(bytes)` | 全节点条带 + THP |
| `hyle_huge_region_destroy(r)` | NULL 安全 |
| `hyle_huge_region_valid` / `data` / `size` | 查询 |
| `hyle_mem_last_error()` | 线程局部：最近一次失败原因串，`""`=无。覆盖 HugeRegion open 的 VM 失败**与所有 create 的资源失败**（`memory_arena`/`stack`/`arena`/`growable` create 的 invalid backing·wrapper OOM·slab OOM、`arena_create` 池耗尽）；转发 platform 的 `vm_last_error`，只绑 `mem.h` 也能读失败原因 |

### 3.7 进程级 mimalloc 策略（自由函数）

| 符号 | 说明 |
|---|---|
| `hyle_enable_allocator_hugepages(allow_thp, allow_large_os_pages)` | 进程启动期调；THP + 大 OS 页 |
| `hyle_enable_allocator_numa(max_nodes)` | max_nodes==0→全节点 |
| `hyle_reserve_huge_1g_pages(pages, numa_node, timeout_ms)` | 预留 1GiB 大页；ANY→交错；成功/`pages==0`→1；软失败（无权限/无连续 RAM）→0 |

### 3.8 导出形态

纯 `extern "C"` 自由函数 + opaque 句柄，FFI 直接绑符号（`dlsym`/bindgen/ctypes 都认）。**无 `hyle_mem_api` 函数指针表、无 `api_get` 版本协商**——版本协商是"分发独立编译插件"才需要的件，HYLE 单一 `.so` 一起编译、未发布，不背这个负担。设计推理见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

**header-inline（无 DLL 符号）**：便利转发 `hyle_alloc` / `hyle_free` / `hyle_realloc`；typed 池宏 `HYLE_OBJECT_POOL_CREATE/ACQUIRE/RELEASE`、`HYLE_GROWABLE_POOL_CREATE/ACQUIRE/RELEASE` + `HYLE_ALIGNOF`。

### 3.9 有意不导出（仅 C++）

| C++ | 原因 |
|---|---|
| `ObjectPool<T>` | **泛型语法**过不了 C ABI（`<T>` 无法表达），非能力缺失。**能力已可达**：C 侧用 `HYLE_OBJECT_POOL_CREATE/ACQUIRE/RELEASE` 宏（§3.2）在 `MemoryArena` 上得到定长对象池，裸槽自 init。需 create 跑 ctor / destroy 跑 ~T 才用 C++ `ObjectPool<T>` |
| `GrowablePool<T>` | 同 `ObjectPool<T>`：**泛型语法**过不了 C ABI，非能力缺失。**能力已可达**：C 侧用 `HYLE_GROWABLE_POOL_CREATE/ACQUIRE/RELEASE` 宏（§3.5）在类型擦除的核心上得到可增长裸槽池，裸槽自 init。需 create 跑 ctor / destroy 跑 ~T 才用 C++ `GrowablePool<T>` |
| `pmr::*` | 模板，C 无泛型。专为桥接 `std::pmr` 容器，纯 C++ 语境 |
| `Allocator` typed 层（`allocate_n` / `construct` / `make_unique`） | C++ 模板便利，C 侧用裸 `alloc` + 手动 size |
| `AllocStats` / `LeakTracker` / `StatsAllocator` | 有状态调试/内省类，是 debug/test 工具，跨语言消费者自带（ASan/valgrind/各 runtime profiler），无 FFI 需求（同 error 的 `ErrorRecord`）。对照 ggml：它只把「用了多少内存」降维成标量查询 `ggml_used_mem`，也不导出统计器类 |

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

### typed 对象池（宏，裸槽自 init）

```c
typedef struct { int a; double b; } Node;
hyle_memory_arena_o* pool = HYLE_OBJECT_POOL_CREATE(sys, Node, 1024);
Node* n = HYLE_OBJECT_POOL_ACQUIRE(pool, Node);  /* 类型化，无需 cast；NULL=耗尽 */
n->a = 7; n->b = 1.5;                            /* C 无 ctor，自己初始化 */
/* ... 用完，release 前自己清理 ... */
HYLE_OBJECT_POOL_RELEASE(pool, n);
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

## 5. 稳定性

HYLE 未发布，无版本门/协商机制（见 §3.8）。发布前 C ABI 面按需自由演进——追求干净正确，不背向后兼容包袱。发布后如何冻结 ABI 见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

导出面：`hyle_allocator` vtable（+ system + usable/good size）、MemoryArena / StackAllocator / Arena / GrowablePool / HugeRegion 不透明句柄、进程级 hugepage/NUMA 策略；typed 池便利宏（`HYLE_OBJECT_POOL_*` / `HYLE_GROWABLE_POOL_*`，header-inline）。

---

## 6. 验收

```bash
cmake --build <build> --target hyle_mem_c_test
./hyle_mem_c_test   # all checks passed
```
