# HYLE C ABI：platform

> **发布面**：`include/hyle/c/platform.h`  
> **实现**：`src/platform/c_platform.cc`（链 `hyle_platform`；spawn 失败依赖 `hyle_error`）  
> **ABI 版本**：`HYLE_PLATFORM_C_ABI_VERSION`（当前 **4**）  
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑本头（及 `hyle/c/error.h`）。不要 include `hyle/platform/*.h`。

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
| VM 自由函数 | reserve/commit/…/alloc_rw | reserve 配对 release |
| VM region | `hyle_vm_region_*` | destroy 释放 |
| hugepage/THP | `hyle_vm_advise_hugepage` 等 | soft 建议，无句柄 |
| NUMA | `hyle_numa_*` / `hyle_vm_reserve_numa` / `hyle_vm_bind_numa*` | reserve 配对 release |
| fs/env | buffer-fill / visitor | 无堆句柄 |
| fs 权限/链接/临时文件/env 写 | `hyle_fs_get_permissions` / `hyle_fs_*_symlink` / `hyle_fs_create_temp_file` / `hyle_fs_set_env_var` | temp file 返回 fd（`hyle_fs_close_fd` 释放） |
| MappedFile | `hyle_mapped_file_*` | destroy 释放 |
| cpu | `hyle_cpu_features_get` 等 | 值返回 POD |
| 表 | `hyle_platform_api_get` | 可选 FFI 函数指针包 |

---

## 3. 导出符号 1:1

与 `include/hyle/c/platform.h` **逐符号对齐**。改头必须同步改本节。

### 3.1 wait / Parker（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_wait_on_address(addr, expected)` | 地址等待；允许伪唤醒 |
| `hyle_wait_on_address_for(addr, expected, timeout_ns)` | 0=超时，1=唤醒/变化/伪唤醒 |
| `hyle_wake_one` / `hyle_wake_all` | 唤醒 |
| `hyle_parker_create` / `destroy` | 不透明 Parker |
| `hyle_parker_park` / `unpark` | token 模型（Rust std） |
| `hyle_parker_park_timeout` | 1=消费 token，0=超时/无 token |

### 3.2 Thread（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_thread_spawn(out, entry, ctx, opts)` | 成功设 `*out`；失败返 `hyle_status`（调用方 free） |
| `hyle_thread_join` | join + **destroy** 句柄 |
| `hyle_thread_joinable` | 1/0 |
| `hyle_hardware_concurrency` | 逻辑 CPU 数 ≥1 |
| `hyle_this_thread_id` | 稳定线程 id |
| `hyle_this_thread_yield` | |
| `hyle_this_thread_set_name` | name 可 NULL→空串 |

**头内 inline / 宏**（无 DLL）：

| 符号 | 说明 |
|---|---|
| `hyle_thread_opts` | `name` / `has_stack_size`+`stack_size` / `has_affinity_cpu`+`affinity_cpu` / `has_priority`+`priority`（v4） |
| `hyle_thread_priority`（v4） | `LOWEST`/`LOW`/`NORMAL`/`HIGH`/`HIGHEST`（-2..2）；best-effort，POSIX 提权需权限 |
| `HYLE_THREAD_OPTS_INIT` | 零初始化宏 |
| `hyle_thread_opts_default` | 返回默认 opts |
| `hyle_thread_opts_set_name` / `set_stack_size` / `set_affinity` / `set_priority`（v4） | setter |

### 3.3 DynLib（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_dynlib_open(path)` | 加载；失败 NULL |
| `hyle_dynlib_get(path)` | **仅**附着已加载，不加载缺失模块 |
| `hyle_dynlib_current` | 当前映像 |
| `hyle_dynlib_close` | 释放句柄 |
| `hyle_dynlib_valid` | 1/0 |
| `hyle_dynlib_symbol(lib, name)` | 符号地址或 NULL |
| `hyle_dynlib_last_error` | 线程局部错误串 |
| `hyle_dynlib_extension` | `"dll"` / `"so"` / `"dylib"` |
| `hyle_dynlib_open_symbol(path, name, out_lib, out_sym)` | open+resolve；成功 1，`*out_lib` 由调用方 close |

### 3.4 Virtual memory（DLL + inline）

| 符号 | 说明 |
|---|---|
| `hyle_protect` | `NONE` / `READ` / `READ_WRITE` |
| `hyle_page_size` | 页大小 |
| `hyle_page_align_up` | **inline**：上取整；0→0 |
| `hyle_vm_reserve` / `commit` / `decommit` / `release` / `protect` | 原语；bool→int 1/0 |
| `hyle_vm_last_error` | |
| `hyle_vm_set_last_error(msg)` | 设线程局部错误串；NULL 清空（v3） |
| `hyle_vm_alloc_rw(size)` | reserve+commit RW（size 上取整） |
| `hyle_vm_free_rw(p, size)` | **inline**：`release(p, page_align_up(size))` |
| `hyle_vm_region_reserve` / `reserve_guarded` | 不透明 region；guarded 低 1 页 NOACCESS |
| `hyle_vm_region_destroy` | |
| `hyle_vm_region_valid` / `data` / `size` / `usable` / `usable_size` / `has_guard` | 查询 |
| `hyle_vm_region_commit` / `commit_usable` / `decommit` / `protect` | 子区间 |

### 3.4b Hugepage / THP（DLL + inline，v3）

soft 建议，绝不影响正确性。Linux 用 madvise；Windows/macOS 匿名区无等价 → 返回 0 + `vm_last_error`。

| 符号 | 说明 |
|---|---|
| `hyle_large_page_size` | 大页字节（未知/不支持→0） |
| `hyle_large_page_align_up(n)` | **inline**：上取整到大页（0 回退普通页）；n==0→0 |
| `hyle_vm_advise_hugepage(ptr, size)` | 空范围→1；ptr 须页对齐；bool→int |
| `hyle_vm_advise_no_hugepage(ptr, size)` | 同上，反向建议 |

### 3.4c NUMA 拓扑 + 放置（DLL + 常量，v3）

| 符号 | 说明 |
|---|---|
| `HYLE_NUMA_ANY` | `(-1)`：无 NUMA 约束 |
| `hyle_numa_policy` | `PREFERRED`(软) / `BIND`(硬,Linux) / `INTERLEAVE`(全节点条带) |
| `hyle_numa_node_count` | 节点数 ≥1 |
| `hyle_numa_current_node` | 调用线程 home 节点（< count） |
| `hyle_vm_bind_numa(ptr, size, node, policy)` | 绑定已有映射；空范围→1；单节点→1(no-op)；多节点无 OS API 的硬绑定/交错→0(+`vm_last_error`) |
| `hyle_vm_bind_numa_interleave(ptr, size)` | 全节点条带（INTERLEAVE 别名） |
| `hyle_vm_reserve_numa(size, node)` | 预留偏好 `node`（ANY==普通 reserve）；失败 NULL；用 `hyle_vm_*` commit/release |

### 3.5 fs / env（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_file_info` | `exists` / `is_directory` / `size` / `modified_time` / `is_symlink`（v4） |
| `hyle_fs_stat` | 整包 |
| `hyle_fs_exists` / `is_directory` / `file_size` | |
| `hyle_fs_make_directory` / `make_directory_recursive` | |
| `hyle_fs_remove_directory` / `remove_directory_recursive` / `remove_file` | |
| `hyle_fs_rename` / `copy_file` / `touch` | |
| `hyle_fs_current_directory` / `temp_directory` / `home_directory` | buffer-fill：返回完整长，`>=cap` 截断 |
| `hyle_fs_change_directory` | |
| `HYLE_ENV_NOT_SET` | `(size_t)-1` |
| `hyle_env_get(name, dst, cap)` | 未设置→`HYLE_ENV_NOT_SET` |
| `hyle_fs_visit_fn` | `(name, is_dir, ctx)→int`；0 停止 |
| `hyle_fs_directory_iterate` | 打开失败→0 |

### 3.5b fs 权限 / 符号链接 / 临时文件 / env 写（DLL，v4）

| 符号 | 说明 |
|---|---|
| `hyle_fs_get_permissions(path)` | 低 9 位权限；Windows 由 READONLY 合成（0444/0666）；0=失败 |
| `hyle_fs_set_permissions(path, mode)` | POSIX chmod；Windows 只认写位→READONLY |
| `hyle_fs_is_readonly(path)` / `set_readonly(path, ro)` | 跨平台可靠子集 |
| `hyle_fs_create_symlink(target, link)` | Windows 需开发者模式/管理员，无则→0 |
| `hyle_fs_create_hard_link(target, link)` | 硬链接（同卷、仅文件）；POSIX link / Win CreateHardLinkW |
| `hyle_fs_is_symlink(path)` | 自身是否符号链接（不跟随） |
| `hyle_fs_read_symlink(path, dst, cap)` | buffer-fill；非链接/失败→0 |
| `hyle_fs_create_temp_file(dir, prefix, dst, cap)` | 原子创建；返回**打开的** fd（`intptr_t`，-1 失败，TOCTOU 安全）；路径写 dst |
| `hyle_fs_close_fd(fd)` | 关闭 create_temp_file 返回的 fd；-1 安全 |
| `hyle_fs_set_env_var(name, value)` / `unset_env_var(name)` | 进程级；与并发读非线程安全 |

### 3.5c MappedFile —— 文件内存映射（DLL，v4）

| 符号 | 说明 |
|---|---|
| `hyle_map_mode` | `READ_ONLY` / `READ_WRITE` / `COPY_ON_WRITE`（对标 memmap2） |
| `hyle_mapped_file_open(path, mode, offset, length)` | length=0→到 EOF；offset 页对齐；失败 NULL（见 `hyle_vm_last_error`） |
| `hyle_mapped_file_destroy(m)` | unmap + 关句柄 |
| `hyle_mapped_file_valid/data/size(m)` | 查询 |
| `hyle_mapped_file_flush(m)` | 回写脏页（RW；RO/COW 为 no-op 返 1） |

### 3.6 CPU（DLL + 常量）

| 符号 | 说明 |
|---|---|
| `hyle_cpu_features` | 与 C++ `CpuFeatures` **全 bool 字段 1:1**（C 用 int 0/1）+ `logical_cpus` + `best_isa` |
| x86 字段 | `sse2`…`avx10_2`（见头） |
| arm 字段 | `neon`…`arm_bf16` |
| RISC-V 字段 | `rvv`…`rvv_zvfhmin` |
| `HYLE_ISA_*` | `SCALAR`/`SSE2`/…/`RVV` 粗粒度档（与 `Isa` 数值一致） |
| `hyle_cpu_features_get` | 值返回 |
| `hyle_cpu_supports_isa(isa)` | 本进程能否跑该档（同架构比较） |
| `hyle_cpu_isa_name(isa)` | 稳定英文名（日志） |

### 3.7 版本 / API 表（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_platform_c_abi_version` | 等于宏 |
| `hyle_platform_api_get(out, expected_abi)` | 0 成功；-1 失败 |

`hyle_platform_api` 字段 = **全部 DLL 自由函数**（无 header-inline）：wait×4 + parker×5 + thread×7 + dynlib×9 + vm 自由×8 + region×13 + fs/env×18 + cpu×3 + `c_abi_version`，外加 `abi_version` 整型。**v3 追加（列在表尾，保 v2 前缀布局不变）**：`vm_set_last_error` + hugepage×3（`large_page_size` / `vm_advise_hugepage` / `vm_advise_no_hugepage`）+ numa×5（`numa_node_count` / `numa_current_node` / `vm_bind_numa` / `vm_bind_numa_interleave` / `vm_reserve_numa`）= 9 个。**v4 追加（同样表尾）**：fs×12（`fs_get_permissions` / `set_permissions` / `is_readonly` / `set_readonly` / `create_symlink` / `create_hard_link` / `is_symlink` / `read_symlink` / `create_temp_file` / `close_fd` / `set_env_var` / `unset_env_var`）+ mapped_file×6（`open` / `destroy` / `valid` / `data` / `size` / `flush`）= 18 个。

**不进表**：`hyle_page_align_up`、`hyle_large_page_align_up`、`hyle_vm_free_rw`、thread opts helpers / `HYLE_THREAD_OPTS_INIT`。

### 3.8 有意不导出

| C++ | 原因 |
|---|---|
| `Unparker` 类型 | 用 `hyle_parker_unpark` 同一句柄 |
| path 词法 / container 字符串 | 不在 platform C 面 |
| 模板 `for_each_entry` | C 用 `hyle_fs_directory_iterate` |
| 测试钩子 | 非发布面 |

---

## 4. 常用写法

### Parker

```c
hyle_parker_o* p = hyle_parker_create();
hyle_parker_unpark(p);
hyle_parker_park(p);
hyle_parker_destroy(p);
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

### DynLib / open_symbol

```c
hyle_dynlib_o* lib = hyle_dynlib_get("kernel32.dll"); /* or libm.so.6 */
void* sym = hyle_dynlib_symbol(lib, "GetCurrentProcessId");
hyle_dynlib_close(lib);

hyle_dynlib_o* plug = NULL;
void* entry = NULL;
if (hyle_dynlib_open_symbol("plugin.so", "entry", &plug, &entry)) {
  /* use entry */
  hyle_dynlib_close(plug);
}
```

### VM

```c
void* p = hyle_vm_alloc_rw(4096);
/* ... */
hyle_vm_free_rw(p, 4096);

hyle_vm_region_o* r = hyle_vm_region_reserve_guarded(64 * 1024);
char* lo = (char*)hyle_vm_region_usable(r);
size_t n = hyle_vm_region_usable_size(r);
hyle_vm_region_destroy(r);
```

### fs

```c
char tmp[512];
hyle_fs_temp_directory(tmp, sizeof tmp);
```

---

## 5. 版本

- v2：完整 `hyle_platform_api` 表、`open_symbol`、`vm_alloc_rw`、thread opts 糖、CPU 全字段（v1 已有）
- v3：NUMA（node 查询 / preferred·bind·interleave / `reserve_numa`）+ hugepage·THP hint + `vm_set_last_error`。新字段一律追加在 `hyle_platform_api` 表尾，v2 调用方的前缀读取不受影响；仅版本门 `hyle_platform_api_get` 拒绝旧号。
- v4：fs 权限（get/set + is/set_readonly）、符号链接（create/is/read）、原子临时文件（`create_temp_file` 返回打开的 fd，TOCTOU 安全）+ `close_fd`、env 写（set/unset）；`hyle_file_info` 加 `is_symlink`；thread `priority`（5 档）；MappedFile（RO/RW/COW 文件内存映射）。同样表尾追加，v2/v3 前缀布局不变。
- 破坏性变更升 `HYLE_PLATFORM_C_ABI_VERSION`

> **有意不导出**：mem 的 `StatsAllocator` / `LeakTracker`（AllocStats 计数 + 泄漏追踪）是**有状态 C++ 类**，属 mem 模块而非 platform；C ABI 化成本高且无 FFI 需求 → 仅 C++。NUMA/hugepage 是 platform 层自由函数原语，故进本表。

---

## 6. 验收

```bash
cmake --build <build> --target hyle_platform_c_test
./hyle_platform_c_test   # all checks passed
```
