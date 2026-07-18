# c-core：core 模块 C ABI（`include/hyle/c/core.h`）

> 逐符号 1:1。纯 `extern "C"`，FFI 直接绑符号。设计推理见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)。

## 1. 符号

| 符号 | 签名 | 说明 |
|---|---|---|
| `hyle_version` | `() → uint32_t` | packed 版本 `(major<<16)｜(minor<<8)｜patch`，与 C++ `hyle::version()` 同值；跨语言用来确认绑的是哪个版本 |
| `hyle_version_string` | `() → const char*` | 人读 `"major.minor.patch"`（静态存储，不用 free） |
| `hyle_version_major` / `minor` / `patch` | `(packed) → uint32_t` | **inline**：拆 packed 版本（无 DLL 符号） |

## 2. 用法

```c
#include "hyle/c/core.h"

uint32_t v = hyle_version();
printf("HYLE %u.%u.%u\n", hyle_version_major(v), hyle_version_minor(v),
       hyle_version_patch(v));
/* 或直接拿字符串 */
printf("HYLE %s\n", hyle_version_string());   /* "0.1.0" */
```

## 3. 导出形态

纯 `extern "C"` 自由函数（`hyle_version` / `hyle_version_string`）+ header-inline 拆解器（`hyle_version_major/minor/patch`，不进 DLL 符号）。桥接 `src/core/c_core.cc` 编进 `hyle_core`，`hyle_version` 直接转发 `hyle::version()`，`hyle_version_string` 从 packed 值构造一次静态串。**无版本协商 / api 表**（设计见 design §3）。
