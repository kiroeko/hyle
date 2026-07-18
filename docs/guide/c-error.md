# HYLE C ABI：error（Status + 诊断）

> **发布面**：`include/hyle/c/error.h`  
> **实现**：`src/error/c_error.cc`（链 `hyle_error`）  
> **稳定性**：未发布，无 ABI 版本门（纯 `extern "C"` 自由函数导出）  
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
| `hyle_status_code` | 稳定整型码：`HYLE_STATUS_OK`…`HYLE_STATUS_UNAUTHENTICATED`（0…16，abseil/gRPC canonical；值与 C++ `StatusCode` 恒等，`c_error.cc` static_assert 锁死） |
| `hyle_status` | **拥有型** 8B 句柄（`uintptr_t rep`）；禁止手改 / 禁止 `t=s` 当拷贝 |
| `hyle_severity` | `HYLE_SEVERITY_WARNING` / `ERROR` / `FATAL` |
| `hyle_error_report_fn` | 诊断回调：`(user, severity, file, line, message)` |
| `hyle_error_saved_handler_o` | opaque saved-handler token（`push` 返回，`pop` 释放） |

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
| `hyle_status_message_cstr` | `(s) → const char*` | 借用指针（NUL 终止），随 handle 存活到 free/move；无消息→`""`；零拷贝，FFI 友好 |
| `hyle_error_set_handler` | `(fn, user)` | 简单单槽：`fn==NULL` 恢复默认 stderr；二次调用覆盖前一个（被替换的丢失）|
| `hyle_error_push_handler` | `(fn, user) → hyle_error_saved_handler_o*` | 嵌套安全：装 `fn` 并返回 opaque token（捕获被替换的 handler）；分配失败→NULL，原 handler 不变 |
| `hyle_error_pop_handler` | `(saved)` | 重装 token 记录的 handler 并释放 token；进程级严格 LIFO（配对 push）；乱序或与另一作用域交错会 fail-fast，避免恢复悬空 handler；`saved==NULL` no-op |
| `hyle_error_last_error` | `() → const char*` | 当前线程 push-token OOM 错误；成功 push 清空 |
| `hyle_error_report` | `(sev, file, line, message)` | 已成型字符串；FATAL trap |
| `hyle_error_reportf` | `(sev, file, line, fmt, ...)` | printf 风格（内部栈缓冲） |

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
| `hyle_status_unauthenticated` | `UNAUTHENTICATED` |

### 4.3 头内 inline 谓词

| 符号 | 说明 |
|---|---|
| `hyle_status_code_is(s, code)` | 通用 |
| `hyle_status_is_cancelled` / `hyle_status_is_unknown` / `hyle_status_is_invalid_argument` / `hyle_status_is_deadline_exceeded` | 对应状态码谓词 |
| `hyle_status_is_not_found` / `hyle_status_is_already_exists` / `hyle_status_is_permission_denied` / `hyle_status_is_resource_exhausted` | 对应状态码谓词 |
| `hyle_status_is_failed_precondition` / `hyle_status_is_aborted` / `hyle_status_is_out_of_range` / `hyle_status_is_unimplemented` | 对应状态码谓词 |
| `hyle_status_is_internal` / `hyle_status_is_unavailable` / `hyle_status_is_data_loss` / `hyle_status_is_unauthenticated` | 对应状态码谓词（`is_ok` 单独是 DLL 符号） |

### 4.4 源码位宏（展开为 report / reportf）

| 宏 | 展开 |
|---|---|
| `HYLE_C_WARN(msg)` | `hyle_error_report(WARNING, __FILE__, __LINE__, msg)` |
| `HYLE_C_ERROR(msg)` | ERROR |
| `HYLE_C_FATAL(msg)` | FATAL（handler 后 trap） |
| `HYLE_C_WARNF(...)` / `ERRORF` / `FATALF` | `hyle_error_reportf` |
| `HYLE_C_RETURN_IF_ERROR(st)` | 非 OK 则 `return st`（按值转移） |

### 4.5 导出形态

纯 `extern "C"` 自由函数 + opaque `hyle_status` / `hyle_error_saved_handler_o` 句柄，FFI 直接绑符号。**无 `hyle_error_api` 函数指针表、无 `api_get` 版本协商**（版本协商是分发独立编译插件才需要的件；HYLE 单一 `.so`、未发布，不背此负担）。设计推理见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。header-inline（4.2–4.4 的工厂/谓词/宏）无 DLL 符号。

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

/* 嵌套安全：作用域内临时接管诊断，pop 精确还原前一个 handler */
hyle_error_saved_handler_o* tok = hyle_error_push_handler(my_scoped_log, ctx);
/* ... 作用域内的 report 走 my_scoped_log ... */
hyle_error_pop_handler(tok);
```

`set_handler` 是单槽便利 API；`push`/`pop` 是作用域 API。一个 pushed
作用域存活时不要调用 `set_handler`，也不要从另一线程交叉 pop。C 与 C++
`ErrorRecord` 共用同一个严格 LIFO 还原门，生命周期次序错误会立即 trap，
不会把已释放的回调重新发布。

---

## 6. 稳定性

HYLE 未发布，C ABI 面无版本门/协商——发布前按需自由演进，追求干净正确。发布后如何冻结 ABI 见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

诊断 handler 有两条路：`set_handler`（简单单槽）与 `push`/`pop`（嵌套安全，opaque saved-handler token，严格 LIFO 恢复，对标 signal/sigaction old-handler 惯用法）。非法 C severity 会归一为 `ERROR`；只有显式 `FATAL` 才触发 trap。

---

## 7. 验收

```bash
cmake --build <build> --target hyle_error_c_test
./hyle_error_c_test   # all checks passed
```
