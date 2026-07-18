# HYLE C ABI：error（Status + 诊断）

> **发布面**：`include/hyle/c/error.h`  
> **实现**：`src/error/c_error.cc`（链 `hyle_error`）  
> **ABI 版本**：`HYLE_ERROR_C_ABI_VERSION`（当前 **3**）  
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器 / FFI **只许绑本头**。不要 include `hyle/error/*.h`。

---

## 1. 链什么

```text
#include "hyle/c/error.h"
// link: hyle_error
```

纯 C 验收目标：`hyle_error_c_test`。

---

## 2. 类型

| 类型 | 含义 |
|---|---|
| `hyle_status_code` | 稳定整型码：`HYLE_STATUS_OK`…`HYLE_STATUS_DATA_LOSS`（0…15） |
| `hyle_status` | **拥有型** 8B 句柄（`uintptr_t rep`）；禁止手改 / 禁止 `t=s` 当拷贝 |
| `hyle_severity` | `HYLE_SEVERITY_WARNING` / `ERROR` / `FATAL` |
| `hyle_error_report_fn` | 诊断回调：`(user, severity, file, line, message)` |
| `hyle_error_api` | 动态 FFI 函数指针表（DLL 符号；不含 header-inline） |

---

## 3. 所有权

| 操作 | API |
|---|---|
| 创建 | `hyle_status_ok` / `hyle_status_new` / 工厂 |
| 拷贝 | **`hyle_status_clone`** |
| 转移 | **`hyle_status_move(&dst, &src)`**（dst 先 free，src 置 OK） |
| 释放 | **`hyle_status_free(&s)`**（幂等；之后 s 为 OK） |

带消息的 status 堆上 refcount；OK / 无消息错误零堆。

---

## 4. 导出符号 1:1

下列与 `include/hyle/c/error.h` **逐符号对齐**。改头文件必须同步改本节。

### 4.1 DLL 导出（`HYLE_ERROR_C_API`）

| 符号 | 签名要点 | 说明 |
|---|---|---|
| `hyle_status_ok` | `() → hyle_status` | OK |
| `hyle_status_new` | `(code, msg) → hyle_status` | 通用构造；`code==OK` 忽略 msg |
| `hyle_status_free` | `(hyle_status*)` | 释放；`NULL` 安全 |
| `hyle_status_clone` | `(hyle_status) → hyle_status` | 深拷贝句柄 |
| `hyle_status_move` | `(dst*, src*)` | 移动 |
| `hyle_status_is_ok` | `(s) → int` | 1/0 |
| `hyle_status_get_code` | `(s) → hyle_status_code` | |
| `hyle_status_has_message` | `(s) → int` | |
| `hyle_status_equals` | `(a,b) → int` | 码+消息比较 |
| `hyle_status_code_name` | `(code) → const char*` | 静态字面量，如 `"NotFound"` |
| `hyle_status_message` | `(s, buf, cap) → size_t` | 仅消息；返回完整长度，`>=cap` 截断 |
| `hyle_status_to_string` | `(s, buf, cap) → size_t` | 如 `"Internal: boom"` |
| `hyle_error_set_handler` | `(fn, user)` | 简单单槽：`fn==NULL` 恢复默认 stderr；二次调用覆盖前一个（被替换的丢失）|
| `hyle_error_push_handler`（v3） | `(fn, user) → hyle_error_saved_handler_o*` | 嵌套安全：装 `fn` 并返回 opaque token（捕获被替换的 handler）；分配失败→NULL（handler 仍已装）|
| `hyle_error_pop_handler`（v3） | `(saved)` | 重装 token 记录的 handler 并释放 token；严格 LIFO（配对 push）；`saved==NULL` no-op |
| `hyle_error_report` | `(sev, file, line, message)` | 已成型字符串；FATAL trap |
| `hyle_error_reportf` | `(sev, file, line, fmt, ...)` | printf 风格（内部栈缓冲） |
| `hyle_error_api_get` | `(out*, expected_abi) → int` | 0 成功；不匹配/`out==NULL` → -1 |
| `hyle_error_c_abi_version` | `() → int` | 等于宏 |

### 4.2 头内 inline 工厂（无 DLL 符号）

均 `(const char* msg) → hyle_status`，msg 可 NULL：

| 符号 | 码 |
|---|---|
| `hyle_status_cancelled` | `HYLE_STATUS_CANCELLED` |
| `hyle_status_unknown` | `UNKNOWN` |
| `hyle_status_invalid_argument` | `INVALID_ARGUMENT` |
| `hyle_status_deadline_exceeded` | `DEADLINE_EXCEEDED` |
| `hyle_status_not_found` | `NOT_FOUND` |
| `hyle_status_already_exists` | `ALREADY_EXISTS` |
| `hyle_status_permission_denied` | `PERMISSION_DENIED` |
| `hyle_status_resource_exhausted` | `RESOURCE_EXHAUSTED` |
| `hyle_status_failed_precondition` | `FAILED_PRECONDITION` |
| `hyle_status_aborted` | `ABORTED` |
| `hyle_status_out_of_range` | `OUT_OF_RANGE` |
| `hyle_status_unimplemented` | `UNIMPLEMENTED` |
| `hyle_status_internal` | `INTERNAL` |
| `hyle_status_unavailable` | `UNAVAILABLE` |
| `hyle_status_data_loss` | `DATA_LOSS` |

### 4.3 头内 inline 谓词

| 符号 | 说明 |
|---|---|
| `hyle_status_code_is(s, code)` | 通用 |
| `hyle_status_is_cancelled` … `hyle_status_is_data_loss` | 与上表全码 1:1（**不含** `is_ok`，ok 走 DLL） |

### 4.4 源码位宏（展开为 report / reportf）

| 宏 | 展开 |
|---|---|
| `HYLE_C_WARN(msg)` | `hyle_error_report(WARNING, __FILE__, __LINE__, msg)` |
| `HYLE_C_ERROR(msg)` | ERROR |
| `HYLE_C_FATAL(msg)` | FATAL（handler 后 trap） |
| `HYLE_C_WARNF(...)` / `ERRORF` / `FATALF` | `hyle_error_reportf` |
| `HYLE_C_RETURN_IF_ERROR(st)` | 非 OK 则 `return st`（按值转移） |

### 4.5 `hyle_error_api` 字段（与 4.1 DLL 一一对应）

`abi_version`，`status_ok`，`status_new`，`status_free`，`status_clone`，`status_move`，`status_is_ok`，`status_get_code`，`status_has_message`，`status_equals`，`status_code_name`，`status_message`，`status_to_string`，`error_set_handler`，`error_report`，`error_reportf`，`c_abi_version`。**v3 表尾追加**：`error_push_handler`，`error_pop_handler`（保 v2 前缀布局不变）。

**不进表**：4.2–4.4 的 inline / 宏。

### 4.6 有意不导出

| C++ | 原因 |
|---|---|
| `Result<T>` / `ASSIGN_OR_RAISE` | 模板；C 用 `hyle_status` + 分支 / `HYLE_C_RETURN_IF_ERROR` |
| `HYLE_ASSERT` / `HYLE_FATAL` | C++ 源码面；C 用 `HYLE_C_*` |
| `ErrorRecord` | 测试设施 |
| `Status::release_rep` / `adopt_rep` | 内部桥接 |

---

## 5. 常用写法

```c
hyle_status s = hyle_status_not_found("user");
if (hyle_status_is_not_found(s)) { /* ... */ }
char buf[128];
hyle_status_to_string(s, buf, sizeof buf);
hyle_status_free(&s);

hyle_error_set_handler(my_log, NULL);
HYLE_C_ERRORF("code=%d", 42);
hyle_error_set_handler(NULL, NULL);

hyle_error_api api;
if (hyle_error_api_get(&api, HYLE_ERROR_C_ABI_VERSION) != 0) { /* ABI */ }
```

---

## 6. 版本

- `hyle_error_c_abi_version()` / `hyle_error_api_get(out, HYLE_ERROR_C_ABI_VERSION)`
- 破坏性变更（表字段删改、句柄布局）升 `HYLE_ERROR_C_ABI_VERSION`
- v2：全码谓词、`reportf`、C 宏、api 表加 `error_reportf`/`c_abi_version`
- v3：嵌套安全的诊断 handler `push`/`pop`（opaque saved-handler token，LIFO 恢复被替换的 handler，对标 signal/sigaction old-handler 惯用法）。新字段追加 api 表尾，v2 前缀布局不变；仅版本门拒绝旧号。

---

## 7. 验收

```bash
cmake --build <build> --target hyle_error_c_test
./hyle_error_c_test   # all checks passed
```
