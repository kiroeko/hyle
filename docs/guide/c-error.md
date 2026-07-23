# HYLE C ABI：error（Status）

> **发布面**：`include/hyle/c/error.h`  
> **实现**：`src/error/c_error.cc`（链接 `hyle_error`）
> **稳定性**：未发布，无 ABI 版本门
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言、跨编译器和纯 C 消费者只绑定本头，不 include `hyle/error/*.h`。error C 面
只传递可恢复失败；日志和 sink 使用 `hyle/c/obs.h`。

## 1. 类型与所有权

```c
typedef enum hyle_status_code { /* canonical 0..16 */ } hyle_status_code;
typedef struct hyle_status { uintptr_t rep; } hyle_status;
```

`hyle_status` 是 8 字节拥有型句柄：

| 操作 | API |
|---|---|
| 创建 | `hyle_status_ok`、`hyle_status_new`、header-inline 每码工厂 |
| 拷贝 | `hyle_status_clone`，禁止用 `dst = src` 伪拷贝 owning handle |
| 转移 | `hyle_status_move(&dst, &src)`，先释放 dst，再把 src 置 OK |
| 释放 | `hyle_status_free(&s)`，NULL 安全、幂等、释放后为 OK |
| 增加上下文 | `hyle_status_add_context(&s, context)`，原地替换 owning handle |

OK 与无消息错误零分配；带消息错误共享引用计数 representation。非法 C enum 输入归一为
`HYLE_STATUS_UNKNOWN`。

## 2. DLL 导出符号

| 符号 | 签名要点 | 说明 |
|---|---|---|
| `hyle_status_ok` | `() -> hyle_status` | OK |
| `hyle_status_new` | `(code, msg) -> hyle_status` | 通用构造；NULL msg 表示无消息 |
| `hyle_status_free` | `(status*)` | 释放并置 OK；NULL 安全 |
| `hyle_status_clone` | `(status) -> status` | 新的独立 owning handle |
| `hyle_status_move` | `(dst*, src*)` | 转移所有权 |
| `hyle_status_add_context` | `(status*, context)` | 前置 `context: `；OK/NULL/空串 no-op；OOM 保留原状态 |
| `hyle_status_is_ok` | `(status) -> int` | 1/0 |
| `hyle_status_get_code` | `(status) -> code` | canonical code |
| `hyle_status_has_message` | `(status) -> int` | 1/0 |
| `hyle_status_equals` | `(a, b) -> int` | 比较 code 和完整消息 |
| `hyle_status_code_name` | `(code) -> const char*` | 静态字符串；非法 code 返回 `"Unknown"` |
| `hyle_status_message` | `(status, buf, cap) -> size_t` | 仅消息；snprintf 式返回所需长度 |
| `hyle_status_to_string` | `(status, buf, cap) -> size_t` | `"OK"`、`"Code"` 或 `"Code: message"` |
| `hyle_status_message_cstr` | `(status) -> const char*` | 零拷贝借用，随 handle 存活；无消息返回 `""` |

## 3. Header-inline 便利面

每个非 OK canonical code 都有工厂，例如：

```c
hyle_status s = hyle_status_not_found("shader file");
hyle_status oom = hyle_status_resource_exhausted(NULL);
```

每码也有谓词：`hyle_status_is_not_found`、`hyle_status_is_unavailable` 等；通用入口是
`hyle_status_code_is(status, code)`。

返回 `hyle_status` 的 C 函数可以使用：

```c
#define HYLE_C_RETURN_IF_ERROR(expr) /* 非 OK 则按值返回 owning handle */
```

表达式只求值一次。OK 是零 representation，无需额外释放；错误 handle 的所有权转移给
调用者。

## 4. 常用写法

```c
hyle_status load_model(void) {
  hyle_status s = open_weights();
  if (!hyle_status_is_ok(s)) {
    hyle_status_add_context(&s, "loading model weights");
    return s;
  }
  return hyle_status_ok();
}

hyle_status s = load_model();
if (!hyle_status_is_ok(s)) {
  char text[256];
  hyle_status_to_string(s, text, sizeof text);
  /* 应用自行决定是否交给 obs/宿主 logger */
}
hyle_status_free(&s);
```

## 5. 有意不导出

| C++ 能力 | C 路径 |
|---|---|
| `Result<T>` | `hyle_status` 返回值 + 成功出参 |
| `HYLE_RETURN_IF_ERROR` / `HYLE_ASSIGN_OR_RETURN` | `HYLE_C_RETURN_IF_ERROR` 或普通分支 |
| `Status::WithContext` | `hyle_status_add_context` |
| `Status::release_rep` / `adopt_rep` | 仅内部桥接，不是用户 ABI |
| 日志、sink | 独立的 `hyle/c/obs.h` 产品面 |

## 6. 验收

```bash
cmake --build <build> --target hyle_error_c_test
ctest --test-dir <build> -R hyle_error_c_test --output-on-failure
```

纯 C 测试覆盖创建、查询、clone/move/free、非法 enum、借用消息、上下文和早返回。
