# HYLE C ABI：platform

> **发布面**：`include/hyle/c/platform.h`  
> **实现**：`src/platform/c_platform.cc`（链 `hyle_platform`，spawn 失败依赖 `hyle_error`）  
> **ABI 版本**：`HYLE_PLATFORM_C_ABI_VERSION`（当前 **1**）  
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑本头（及 `hyle/c/error.h`）。不要 include `hyle/platform/*.h`。

---

## 1. 链什么

```text
#include "hyle/c/platform.h"   /* pulls hyle/c/error.h for hyle_status */
// link: hyle_platform (+ hyle_error)
```

纯 C 编译器可编：`hyle_platform_c_test`。

---

## 2. 子系统一览

| 块 | 类型 / 入口 | 所有权 |
|---|---|---|
| **wait** | `hyle_wait_on_address*` / `hyle_wake_*` | 调用方提供 `uint32_t` 字 |
| **Parker** | `hyle_parker_create` / `destroy` / `park` / `unpark` / `park_timeout` | create→destroy |
| **Thread** | `hyle_thread_spawn` → `hyle_thread_join` | join **释放**句柄 |
| **DynLib** | `open` / `get` / `current` / `close` / `symbol` | close 释放 |
| **VM** | 自由函数 + `hyle_vm_region_*` | region destroy；reserve 配对 release |
| **fs/env** | `hyle_fs_*` / `hyle_env_get` | buffer-fill，无堆句柄 |
| **cpu** | `hyle_cpu_features_get` | 值返回 POD |
| **表** | `hyle_platform_api_get` | 可选 FFI 函数指针包 |

---

## 3. 常用写法

### Parker

```c
hyle_parker_o* p = hyle_parker_create();
hyle_parker_unpark(p);
hyle_parker_park(p);                 /* 立即返回（token 已在） */
hyle_parker_destroy(p);
```

### Thread

```c
void worker(void* ctx) { *(int*)ctx = 1; }

int flag = 0;
hyle_thread_o* t = NULL;
hyle_thread_opts opts = {0};
opts.name = "worker";
opts.has_stack_size = 1;
opts.stack_size = 2 * 1024 * 1024;

hyle_status st = hyle_thread_spawn(&t, worker, &flag, &opts);
if (!hyle_status_is_ok(st)) { /* ... */ hyle_status_free(&st); return; }
hyle_status_free(&st);
hyle_thread_join(t);   /* join + free */
```

### DynLib

```c
hyle_dynlib_o* lib = hyle_dynlib_open("plugin.so");
if (!hyle_dynlib_valid(lib)) {
  fprintf(stderr, "%s\n", hyle_dynlib_last_error());
  return;
}
void* sym = hyle_dynlib_symbol(lib, "entry");
hyle_dynlib_close(lib);

/* 已加载、不加载： */
hyle_dynlib_o* k = hyle_dynlib_get("kernel32.dll"); /* or libm.so.6 */
hyle_dynlib_close(k);
```

### 带 guard 的栈区（fiber 友好）

```c
hyle_vm_region_o* r = hyle_vm_region_reserve_guarded(64 * 1024);
/* usable 可写；低 1 页 guard */
char* stack_lo = (char*)hyle_vm_region_usable(r);
size_t usable = hyle_vm_region_usable_size(r);
char* stack_hi = stack_lo + usable;  /* 栈向下增长时的 top */
hyle_vm_region_destroy(r);
```

### fs

```c
char tmp[512];
hyle_fs_temp_directory(tmp, sizeof tmp);
char dir[600];
snprintf(dir, sizeof dir, "%s/myjob", tmp);
hyle_fs_make_directory_recursive(dir);
hyle_fs_directory_iterate(dir, my_visit, NULL);
```

---

## 4. 导出是否齐全（对照 C++）

| C++ | C |
|---|---|
| `detail::wait_on_address/for`, `wake_one/all` | ✅ `hyle_wait_*` / `hyle_wake_*` |
| `Parker` / `park_timeout` | ✅ opaque create/destroy |
| `Unparker` | 用 `hyle_parker_unpark`（同一句柄） |
| `Thread::spawn/join`, opts | ✅ + `hyle_status` 错误 |
| `this_thread::*`, `hardware_concurrency` | ✅ |
| `DynLib::open/get/current/symbol` | ✅ |
| `vm_*` + `VmRegion` + guarded | ✅ |
| `fs` / env / iterate | ✅ 主表面 |
| `CpuFeatures` 全字段 | **子集**（常用 x86/arm + logical_cpus）；完整字段仍在 C++ |
| path 字符串（container） | 不在 platform C 面 |

有意不导出：C++ 模板 `for_each_entry`（C 用 `hyle_fs_directory_iterate`）、path 词法（container）、测试专用钩子。

---

## 5. 版本

- `hyle_platform_c_abi_version()` / `hyle_platform_api_get(out, HYLE_PLATFORM_C_ABI_VERSION)`
- 破坏性变更升 `HYLE_PLATFORM_C_ABI_VERSION`

---

## 6. 验收

```bash
cmake --build <build> --target hyle_platform_c_test
./hyle_platform_c_test   # all checks passed
```
