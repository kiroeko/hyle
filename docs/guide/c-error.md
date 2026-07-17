# HYLE C ABI：error（Status + 诊断）

> **发布面**：`include/hyle/c/error.h`  
> **实现**：`src/error/c_error.cc`（链 `hyle_error`）  
> **ABI 版本**：`HYLE_ERROR_C_ABI_VERSION`（当前 **1**）  
> **命名**：全部 `hyle_*` / `HYLE_*`（无 clay 前缀）

这是 close-01 的第一块纯 C 样板。C++/同工具链开发仍可用 `hyle/error/*.h`；**跨语言 / 跨编译器 / FFI 只许绑本头**。

---

## 1. 链什么

```text
#include "hyle/c/error.h"
// 链接: hyle_error  (及统一 hyle 聚合目标，若有)
```

- 纯 C 编译器即可（`c_error_test.c` 即验收）。
- 不要 `#include "hyle/error/status.h"` 进 C 或跨编译器 FFI。

---

## 2. 核心类型

| 类型 | 含义 |
|---|---|
| `hyle_status_code` | 稳定整型码表（0=OK … 15=DATA_LOSS），与内部 `StatusCode` 数值一致 |
| `hyle_status` | **拥有型** 8 字节句柄；内含不透明 `rep`，禁止手改 |
| `hyle_severity` | 诊断级别：WARNING / ERROR / FATAL |
| `hyle_error_api` | 可选：整表函数指针，方便动态绑定 |

---

## 3. 所有权（必读）

```c
hyle_status s = hyle_status_not_found("id");
/* ... */
hyle_status_free(&s);   /* 用完必 free；OK 也安全 */
```

| 操作 | API | 说明 |
|---|---|---|
| 创建 | `hyle_status_ok` / `hyle_status_new` / `hyle_status_*` 工厂 | 得到拥有权 |
| 拷贝 | **`hyle_status_clone`** | 禁止 `t = s` 当深拷贝（带消息会双释放） |
| 转移 | **`hyle_status_move(&dst, &src)`** | dst 先被 free，src 置 OK |
| 释放 | **`hyle_status_free(&s)`** | 幂等；之后 s 为 OK |

带消息的 status 在堆上 refcount；OK 与无消息错误零堆。

---

## 4. 常用写法

### 4.1 构造与判断

```c
hyle_status s = hyle_status_ok();
if (!hyle_status_is_ok(s)) { /* ... */ }

hyle_status e = hyle_status_not_found("user");
if (hyle_status_is_not_found(e)) { /* ... */ }
if (hyle_status_code_is(e, HYLE_STATUS_NOT_FOUND)) { /* 等价 */ }

char buf[128];
hyle_status_to_string(e, buf, sizeof buf);  /* "NotFound: user" */
hyle_status_free(&e);
```

### 4.2 全码表工厂（头内 inline，无额外 DLL 符号）

`hyle_status_cancelled` / `unknown` / `invalid_argument` / `deadline_exceeded` /
`not_found` / `already_exists` / `permission_denied` / `resource_exhausted` /
`failed_precondition` / `aborted` / `out_of_range` / `unimplemented` /
`internal` / `unavailable` / `data_loss` — 均 `(const char* msg)`，msg 可 NULL。

通用入口：`hyle_status_new(code, msg)`。

### 4.3 传播习惯（C 无宏版 `?`）

```c
hyle_status do_work(void) {
  hyle_status st = open_thing();
  if (!hyle_status_is_ok(st)) return st;   /* 转移所有权给调用方 */
  /* ... */
  hyle_status_free(&st);
  return hyle_status_ok();
}
```

### 4.4 诊断回调

```c
void my_log(void* user, hyle_severity sev, const char* file,
            uint32_t line, const char* message) {
  (void)user; (void)sev; (void)file; (void)line;
  fputs(message, stderr);
}

hyle_error_set_handler(my_log, NULL);
hyle_error_report(HYLE_SEVERITY_ERROR, __FILE__, __LINE__, "oops");
hyle_error_set_handler(NULL, NULL);  /* 恢复默认 stderr */
```

`HYLE_SEVERITY_FATAL`：handler 返回后进程 trap（不返回）。

### 4.5 函数指针表（动态 FFI）

```c
hyle_error_api api;
if (hyle_error_api_get(&api, HYLE_ERROR_C_ABI_VERSION) != 0) {
  /* ABI 不匹配或 out==NULL */
}
hyle_status s = api.status_new(HYLE_STATUS_INTERNAL, "x");
api.status_free(&s);
```

---

## 5. 导出清单（是否齐全）

### Status 值通道（对标 C++ `Status`）

| C API | C++ 对应 |
|---|---|
| `hyle_status_ok` | `Status()` / `Status::Ok()` |
| `hyle_status_new` + 各工厂 | `Status(code, msg)` / `Status::NotFound(msg)` … |
| `hyle_status_free` / `clone` / `move` | 析构 / 拷贝 / 移动 |
| `hyle_status_is_ok` / `get_code` / `has_message` | `ok()` / `code()` / `has_message()` |
| `hyle_status_equals` | `operator==` |
| `hyle_status_code_name` | `status_code_name` |
| `hyle_status_message` / `to_string` | `message()` / `ToString` |
| 码表 enum | `StatusCode` X-macro |

### 诊断通道（对标 `error_handler`）

| C API | C++ 对应 |
|---|---|
| `hyle_error_set_handler` | `error_set_default_handler` |
| `hyle_error_report` | `error_report`（已成型字符串） |

### 有意不导出（C 侧用别的方式）

| C++ | 原因 |
|---|---|
| `Result<T>` / `ASSIGN_OR_RAISE` | 模板 + 语言糖；C 用 `hyle_status` 返回值 + 手动分支 |
| `HYLE_ASSERT` / `HYLE_FATAL` 宏 | 属 C++ 源码面；C 用 `hyle_error_report` + 自行 abort |
| `ErrorRecord` | 测试设施，非发布面 |
| `Status::release_rep` | 内部桥接，不进 C |

---

## 6. 版本与兼容

- 读 `hyle_error_c_abi_version()` 或 `hyle_error_api_get(..., expected)`。
- 破坏性变更：升 `HYLE_ERROR_C_ABI_VERSION`，旧 expected 返回 -1。
- 新增工厂 inline / 非破坏性函数指针可在同主版本追加（文档注明）；删改符号升版本。

---

## 7. 验收

```bash
# Windows / Linux
cmake --build <build> --target hyle_error_c_test
./hyle_error_c_test   # 纯 C 编链，应打印 all checks passed
```

C++ 回归：`hyle_error_test`（Status / handler / 宏）。

---

## 8. 与 C++ 头的边界

| | `hyle/c/error.h` | `hyle/error/*.h` |
|---|---|---|
| 语言 | 纯 C | C++23 |
| ABI 承诺 | **是（发布）** | 否（同工具链） |
| FFI | 是 | 否 |
| 性能 | 与 C++ 同路径（薄转发） | 默认可内联更多 |
