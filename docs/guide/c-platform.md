# HYLE C ABI：platform

> **发布面**：`include/hyle/c/platform.h`  
> **实现**：`src/platform/c_platform.cc`（链 `hyle_platform` + `hyle_error`）
> **稳定性**：未发布，无 ABI 版本门（纯 `extern "C"` 自由函数导出）  
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑本头（及 `hyle/c/error.h`）。不要 include `hyle/platform/*.h`。

资源构造、动态加载、文件映射和 VM/NUMA 的可恢复失败直接返回拥有型
`hyle_status`：

| 返回面 | 诊断 |
|---|---|
| Parker / Thread / DynLib / MappedFile / VM / NUMA / 打开的文件资源 | 返回 `hyle_status`，用 `hyle_status_message_cstr` |
| 尚未迁移的 path/env 标量函数 | `hyle_platform_last_error()`（过渡接口） |

`hyle_status` 由调用方用 `hyle_status_free` 释放；剩余 last-error 指针借用到同线程
下一次相关调用。

---

## 1. 链什么

```text
#include "hyle/c/platform.h"   /* pulls hyle/c/error.h for hyle_status */
// link: hyle_platform (+ hyle_error)
```

纯 C 验收：`hyle_platform_c_test`。

---

## 2. 子系统与所有权

| 块 | 入口 | 所有权 |
|---|---|---|
| wait | `hyle_wait_on_address*` / `hyle_wake_*` | 调用方提供 `uint32_t` 字 |
| Parker | create / destroy / park / unpark / park_timeout | create→destroy |
| Thread | spawn → join | join **释放**句柄 |
| DynLib | open / get / current / close / symbol / open_symbol | close 释放 |
| VM 自由函数 | reserve/commit/decommit/release/protect | reserve 配对 release |
| VM region | `hyle_vm_region_*` | destroy 释放 |
| hugepage/THP | `hyle_vm_advise_hugepage` 等 | soft 建议，无句柄 |
| NUMA | `hyle_numa_*` / `hyle_vm_reserve_numa` / `hyle_vm_bind_numa*` | reserve 配对 release |
| fs/env | buffer-fill / visitor | 无堆句柄 |
| fs 权限/链接/临时文件/env 写 | `hyle_fs_get_permissions` / `hyle_fs_*_symlink` / `hyle_fs_create_temp_file` / `hyle_fs_set_env_var` | temp file 返回 `hyle_file_handle`（`hyle_fs_close` 释放） |
| MappedFile | `hyle_mapped_file_*` | destroy 释放 |
| cpu | `hyle_cpu_features_get` 等 | 值返回 POD |

---

## 3. 导出符号 1:1

与 `include/hyle/c/platform.h` **逐符号对齐**。改头必须同步改本节。

### 3.1 wait / Parker（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_platform_last_error` | 尚未迁移的 path/env 标量函数的线程局部失败串；`""`=无；不得用于 Parker/打开文件资源 |
| `hyle_wait_on_address(addr, expected)` | 地址等待；允许伪唤醒 |
| `hyle_wait_on_address_for(addr, expected, timeout_ns)` | 0=超时，1=唤醒/变化/伪唤醒 |
| `hyle_wake_one` / `hyle_wake_all` | 唤醒 |
| `hyle_parker_create(out)` / `hyle_parker_destroy` | 不透明 Parker；create 返回 Status 并先清空 `*out` |
| `hyle_parker_park` / `unpark` | token 模型（Rust std） |
| `hyle_parker_park_timeout` | 1=消费 token，0=超时/无 token |

### 3.2 Thread（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_thread_spawn(out, entry, ctx, opts)` | 成功设 `*out`；失败返 `hyle_status`（调用方 free） |
| `hyle_thread_spawn_opts(out, entry, ctx, name, stack, affinity, priority)` | FFI 便利入口；不用构造 opts/has 字段，0/-1 是默认 sentinel |
| `hyle_thread_join` | join + **destroy** 句柄 |
| `hyle_thread_joinable` | 1/0 |
| `hyle_hardware_concurrency` | 逻辑 CPU 数 ≥1 |
| `hyle_this_thread_id` | 稳定线程 id |
| `hyle_this_thread_yield` | |
| `hyle_this_thread_set_name` | name 可 NULL→空串 |

**头内 inline / 宏**（无 DLL）：

| 符号 | 说明 |
|---|---|
| `hyle_thread_opts` | 首字段 `struct_size`，其后为 `name` / `has_stack_size`+`stack_size` / `has_affinity_cpu`+`affinity_cpu` / `has_priority`+`priority` |
| `hyle_thread_priority`（v4） | `LOWEST`/`LOW`/`NORMAL`/`HIGH`/`HIGHEST`（-2..2）；best-effort，POSIX 提权需权限 |
| `HYLE_THREAD_OPTS_INIT` | 默认初始化宏；会写 `struct_size`，不是全零字节 |
| `hyle_thread_opts_default` | 返回默认 opts |
| `hyle_thread_opts_set_name` / `hyle_thread_opts_set_stack_size` / `hyle_thread_opts_set_affinity` / `hyle_thread_opts_set_priority` | setter |

### 3.3 DynLib（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_dynlib_open(out, path)` | 加载；成功写拥有句柄，失败返 `hyle_status` |
| `hyle_dynlib_get(out, path)` | **仅**附着已加载，不加载缺失模块；返回 Status |
| `hyle_dynlib_current(out)` | 当前映像；返回 Status |
| `hyle_dynlib_close` | 释放句柄 |
| `hyle_dynlib_valid` | 1/0 |
| `hyle_dynlib_symbol(out, lib, name)` | 成功写符号地址；Status 的成功态可表达合法的 NULL 值符号 |
| `hyle_dynlib_extension` | `"dll"` / `"so"` / `"dylib"` |
| `hyle_dynlib_open_symbol(out_lib, out_sym, path, name)` | open+resolve；返回 Status，成功后 `*out_lib` 由调用方 close |

### 3.4 Virtual memory（DLL + inline）

| 符号 | 说明 |
|---|---|
| `hyle_protect` | `NONE` / `READ` / `READ_WRITE` |
| `hyle_page_size` | 页大小 |
| `hyle_page_align_up` | **inline**：上取整；0→0 |
| `hyle_vm_reserve(out, size)` / `commit` / `decommit` / `release` / `protect` | 全部返回 `hyle_status`；reserve 成功写 `*out` |
| `hyle_vm_alloc_rw(out, size)` | 一次 OS 操作得到可立即使用的 RW mapping（size 上取整）；返回 Status |
| `hyle_vm_free_rw(p, size)` | 释放 `alloc_rw` 的块（size 为原始请求）；返回 Status |
| `hyle_vm_region_reserve(out, size)` / `reserve_guarded` | Status+out；guarded 低 1 页 NOACCESS |
| `hyle_vm_region_reserve_numa(out, size, node)` | 偏好 NUMA `node`（`HYLE_NUMA_ANY`==普通 reserve）的托管 region |
| `hyle_vm_region_adopt(out, base, size)` | 收编裸预留；只有成功后才转移所有权，wrapper OOM/参数错误时调用方仍拥有 base |
| `hyle_vm_region_destroy` | |
| `hyle_vm_region_valid` / `data` / `size` / `usable` / `usable_size` / `has_guard` / `numa_placement` | 查询 |
| `hyle_vm_region_commit` / `commit_usable` / `decommit` / `protect` | 子区间，返回 Status |

### 3.4b Hugepage / THP（DLL + inline，v3）

soft 建议，绝不影响正确性。Linux 用 madvise；Windows/macOS 匿名区无等价时返回
`HYLE_STATUS_UNIMPLEMENTED`，映射保持可用。

| 符号 | 说明 |
|---|---|
| `hyle_large_page_size` | 大页字节（未知/不支持→0） |
| `hyle_large_page_align_up(n)` | **inline**：上取整到大页（0 回退普通页）；n==0→0 |
| `hyle_vm_advise_hugepage(ptr, size)` | 空范围→OK；ptr 须页对齐；返回 Status |
| `hyle_vm_advise_no_hugepage(ptr, size)` | 同上，反向建议 |

### 3.4c NUMA 拓扑 + 放置（DLL + 常量，v3）

| 符号 | 说明 |
|---|---|
| `hyle_numa_node`：`HYLE_NUMA_ANY` | 具名 enum；`(-1)` 无 NUMA 约束 |
| `hyle_numa_policy` | `PREFERRED`(软) / `BIND`(硬,Linux) / `INTERLEAVE`(全节点条带) |
| `hyle_numa_placement` | `APPLIED` / `NOT_NEEDED` / `UNAVAILABLE`（软 preferred 降级） |
| `hyle_numa_node_count` | 节点数 ≥1 |
| `hyle_numa_current_node` | 调用线程 home 节点（< count） |
| `hyle_numa_node_of_cpu(cpu)` | 逻辑 CPU 所属节点（< count；越界/单节点/失败→0）；配 `affinity_cpu` 把 worker 的 arena 放到其核所在节点 |
| `hyle_vm_bind_numa(out_placement, ptr, size, node, policy)` | 返回 Status；placement 出参可 NULL；硬绑定/交错无法满足是错误 |
| `hyle_vm_bind_numa_interleave(out_placement, ptr, size)` | 全节点条带（INTERLEAVE 别名） |
| `hyle_vm_reserve_numa(out, out_placement, size, node)` | Status+out；preferred 失败仍可成功分配并报告 `UNAVAILABLE` |

### 3.5 fs / env（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_file_info` | `exists` / `is_directory` / `size` / `modified_time` / `is_symlink`（v4） |
| `hyle_fs_stat` | 整包 |
| `hyle_fs_exists` / `hyle_fs_is_directory` / `hyle_fs_file_size` | |
| `hyle_fs_make_directory` / `hyle_fs_make_directory_recursive` | |
| `hyle_fs_remove_directory` / `hyle_fs_remove_directory_recursive` / `hyle_fs_remove_file` | |
| `hyle_fs_rename` / `hyle_fs_copy_file` / `hyle_fs_touch` | |
| `hyle_fs_current_directory` / `hyle_fs_temp_directory` / `hyle_fs_home_directory` | buffer-fill：返回完整长，`>=cap` 截断；`dst==NULL` 可只查长度 |
| `hyle_fs_change_directory` | |
| `HYLE_ENV_NOT_SET` | `(size_t)-1`；size_t sentinel，保留 `#define`（enum 成员在 C23 前装不下 size_t 值） |
| `hyle_env_get(name, dst, cap)` | 未设置→`HYLE_ENV_NOT_SET` |
| `hyle_fs_visit_fn` | `(name, is_dir, ctx)→int`；0 停止 |
| `hyle_fs_directory_iterate` | 打开失败→0 |

### 3.5b fs 权限 / 符号链接 / 临时文件 / env 写（DLL，v4）

| 符号 | 说明 |
|---|---|
| `hyle_fs_get_permissions(path)` | 低 9 位权限；Windows 由 READONLY 合成（0444/0666）；0 也可能是合法的 POSIX mode，先用 `hyle_fs_stat` 区分不存在 |
| `hyle_fs_set_permissions(path, mode)` | POSIX chmod；Windows 只认写位→READONLY |
| `hyle_fs_is_readonly(path)` / `hyle_fs_set_readonly(path, ro)` | 跨平台可靠子集 |
| `hyle_fs_create_symlink(target, link)` | Windows 需开发者模式/管理员，无则→0 |
| `hyle_fs_create_hard_link(target, link)` | 硬链接（同卷、仅文件）；POSIX link / Win CreateHardLinkW |
| `hyle_fs_is_symlink(path)` | 自身是否符号链接（不跟随） |
| `hyle_fs_read_symlink(path, dst, cap)` | buffer-fill；非链接/失败→0 |
| `hyle_file_handle` / `HYLE_FILE_HANDLE_INVALID` | 一字宽 native file resource；可容纳 POSIX fd / Windows HANDLE，不暗示 POSIX 语义 |
| `hyle_fs_create_temp_file(out, dir, prefix, dst, cap)` | Status+out；原子创建并返回**打开的** `hyle_file_handle`，路径写 dst，避免 TOCTOU |
| `hyle_fs_open(out, path, flags)` | Status+out；`flags`=`HYLE_OPEN_READ/WRITE/CREATE/TRUNCATE/APPEND` OR；Windows 用 UTF-16 API；无需 mmap 的普通 I/O |
| `hyle_fs_read(out_moved, file, buf, n)` / `hyle_fs_write(out_moved, file, buf, n)` | Status+out；读 0=EOF，短读/短写是成功，调用方可继续 |
| `hyle_fs_close(file)` | 返回 Status；关闭 create_temp_file / open 返回的 handle；INVALID 安全 |
| `hyle_fs_set_env_var(name, value)` / `hyle_fs_unset_env_var(name)` | 进程级；与并发读非线程安全 |

### 3.5c MappedFile —— 文件内存映射（DLL，v4）

| 符号 | 说明 |
|---|---|
| `hyle_map_mode` | `READ_ONLY` / `READ_WRITE` / `COPY_ON_WRITE`（对标 memmap2） |
| `hyle_mapped_file_open(out, path, mode, offset, length)` | length=0→到 EOF；支持非页对齐 offset（内部扩大/对齐 OS view）；返回 Status |
| `hyle_mapped_file_destroy(m)` | unmap + 关句柄 |
| `hyle_mapped_file_valid` / `hyle_mapped_file_data` / `hyle_mapped_file_size` | 查询 |
| `hyle_mapped_file_flush(m)` | 回写脏页（RW；RO/COW 为 OK no-op）；返回 Status |

### 3.6 CPU（DLL + 常量）

| 符号 | 说明 |
|---|---|
| `hyle_cpu_features` | 与 C++ `CpuFeatures` **全 bool 字段 1:1**（C 用 int 0/1）+ `logical_cpus` + `best_isa` |
| x86 字段 | `sse2`…`avx10_2`（见头） |
| arm 字段 | `neon`…`arm_bf16` |
| RISC-V 字段 | `rvv`…`rvv_zvfhmin` |
| `hyle_isa`：`HYLE_ISA_*` | 具名 enum；`SCALAR`/`SSE2`/…/`RVV` 粗粒度档（与 `Isa` 数值一致） |
| `hyle_cpu_features_get` | 值返回 |
| `hyle_cpu_supports_isa(isa)` | 本进程能否跑该档（同架构比较） |
| `hyle_cpu_isa_name(isa)` | 稳定英文名（日志） |
| `hyle_cpu_cache_line_size` | L1 数据缓存行字节数（主流 64；对齐/填充用；探测失败回退 64） |
| `hyle_cpu_cache_info` | 值返回 `hyle_cache_info`：L1D/L1I/L2/L3 容量（字节；未知级为 0） |
| `hyle_cpu_topology_get` | 值返回逻辑 CPU、performance/efficiency CPU 数与 `is_hybrid` |

拓扑只采用 OS/CPU 的明确分类（Windows `EfficiencyClass`、macOS
`hw.perflevelN`、Linux `cpu_capacity`）。若平台没有可靠信息，会保守返回
`performance_cpus == logical_cpus`、`efficiency_cpus == 0`、`is_hybrid == 0`；
这表示“未确认异构”，不会根据最高频率猜核型。

### 3.7 导出形态

纯 `extern "C"` 自由函数 + opaque 句柄（Parker/Thread/DynLib/VmRegion/MappedFile），FFI 直接绑符号。**无 `hyle_platform_api` 函数指针表、无 `api_get` 版本协商**（版本协商是分发独立编译插件才需要的件；HYLE 单一 `.so`、未发布，不背此负担）。设计推理见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

**header-inline（无 DLL 符号）**：`hyle_page_align_up`、`hyle_large_page_align_up`、thread opts helpers / `HYLE_THREAD_OPTS_INIT`。

### 3.8 有意不导出

| C++ | 原因 |
|---|---|
| `Unparker` 类型 | 用 `hyle_parker_unpark` 同一句柄 |
| `DynLib::symbol_as<Fn>`（模板） | 用 `hyle_dynlib_symbol` + 调用方 cast |
| `FileHandle`（RAII） | C 走 `hyle_file_handle` + `hyle_fs_close`；不把 Windows HANDLE 伪称 POSIX fd |
| `VmRegion::release` / `MappedFile::reset`（提前释放） | 用 `hyle_vm_region_destroy` / `hyle_mapped_file_destroy` |
| path 词法 / container 字符串 | 不在 platform C 面 |
| 模板 `for_each_entry` | C 用 `hyle_fs_directory_iterate` |
| 测试钩子 | 非发布面 |

---

## 4. 常用写法

### Parker

```c
hyle_parker_o* p = NULL;
hyle_status st = hyle_parker_create(&p);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
hyle_parker_unpark(p);
hyle_parker_park(p);
hyle_parker_destroy(p);
```

### 普通文件 I/O

```c
hyle_file_handle file = HYLE_FILE_HANDLE_INVALID;
hyle_status st = hyle_fs_open(
    &file, "config.bin", HYLE_OPEN_READ);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);

unsigned char buf[4096];
size_t got = 0;
st = hyle_fs_read(&got, file, buf, sizeof buf);
if (hyle_status_is_ok(st)) { /* consume buf[0..got), got==0 means EOF */ }
hyle_status_free(&st);
st = hyle_fs_close(file); /* report close/flush failures */
hyle_status_free(&st);
```

### Thread

```c
void worker(void* ctx) { *(int*)ctx = 1; }
int flag = 0;
hyle_thread_o* t = NULL;
hyle_thread_opts opts = HYLE_THREAD_OPTS_INIT;
hyle_thread_opts_set_name(&opts, "worker");
hyle_thread_opts_set_stack_size(&opts, 2u * 1024u * 1024u);
hyle_status st = hyle_thread_spawn(&t, worker, &flag, &opts);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
hyle_thread_join(t);
```

FFI 不方便构造结构体时可直接：

```c
hyle_status st = hyle_thread_spawn_opts(
    &t, worker, &flag, "worker", 0, -1, HYLE_THREAD_PRIORITY_NORMAL);
```

### DynLib / open_symbol

```c
hyle_dynlib_o* lib = NULL;
hyle_status st = hyle_dynlib_get(&lib, "kernel32.dll"); /* or libm.so.6 */
if (hyle_status_is_ok(st)) {
  void* sym = NULL;
  hyle_status sym_st = hyle_dynlib_symbol(&sym, lib, "GetCurrentProcessId");
  if (hyle_status_is_ok(sym_st)) { /* use sym */ }
  hyle_status_free(&sym_st);
  hyle_dynlib_close(lib);
}
hyle_status_free(&st);

hyle_dynlib_o* plug = NULL;
void* entry = NULL;
hyle_status plug_st = hyle_dynlib_open_symbol(&plug, &entry, "plugin.so", "entry");
if (hyle_status_is_ok(plug_st)) {
  /* use entry */
  hyle_dynlib_close(plug);
}
hyle_status_free(&plug_st);
```

### VM

```c
void* p = NULL;
hyle_status st = hyle_vm_alloc_rw(&p, 4096);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
/* ... */
hyle_status free_st = hyle_vm_free_rw(p, 4096);
hyle_status_free(&free_st);

hyle_vm_region_o* r = NULL;
st = hyle_vm_region_reserve_guarded(&r, 64 * 1024);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
char* lo = (char*)hyle_vm_region_usable(r);
size_t n = hyle_vm_region_usable_size(r);
hyle_vm_region_destroy(r);
```

### fs

```c
char tmp[512];
if (hyle_fs_temp_directory(tmp, sizeof tmp) == 0) {
  fprintf(stderr, "platform: %s\n", hyle_platform_last_error());
}
```

---

## 5. 稳定性

HYLE 未发布，C ABI 面无版本门/协商——发布前按需自由演进，追求干净正确。发布后如何冻结 ABI 见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

导出面覆盖：wait/Parker、Thread（priority 5 档）、DynLib、VM 自由函数 + VmRegion（含 `reserve_numa`/`adopt`）、hugepage·THP、NUMA（node 查询 + preferred/bind/interleave + `reserve_numa`）、fs/env（权限/符号链接/硬链接/原子临时文件/env 写）、MappedFile（RO/RW/COW）、cpu 全字段。

> mem 的 `StatsAllocator` / `LeakTracker` 不属于 platform 导出面；其 C ABI
> 位于 [`c-mem.md`](./c-mem.md) §2.2。
> NUMA/hugepage 是 platform 用户能力，故在本模块直接导出。

---

## 6. 验收

```bash
cmake --build <build> --target hyle_platform_c_test
./hyle_platform_c_test   # all checks passed
```
