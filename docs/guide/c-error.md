# HYLE C ABI：error（Status）

> **发布面**：`include/hyle/c/error.h`  
> **实现**：`src/error/c_error.cc`（链接 `hyle_error`）
> **稳定性**：未发布，无 ABI 版本门
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言、跨编译器和纯 C 消费者只绑定本头，不 include `hyle/error/*.h`。error C 面
只传递可恢复失败；日志和 sink 使用 `hyle/c/obs.h`。如果还没决定应该用 Status、
Result/out、空值合同还是 fail-fast，先看语言无关的 [`error.md`](./error.md)；本文只说明
选定 Status 后怎样准确映射到 C/FFI。

## 1. 类型与所有权

```c
typedef enum hyle_status_code { /* canonical 0..16 */ } hyle_status_code;
typedef struct hyle_status { uintptr_t rep; } hyle_status;
typedef uint16_t hyle_native_error_domain;
```

`hyle_status` 是一个 machine word 的拥有型句柄（64 位平台为 8B）：

| 操作 | API |
|---|---|
| 初始化/创建 | `HYLE_STATUS_INIT`、`hyle_status_ok`、`hyle_status_new[_n]`、header-inline 每码工厂 |
| 拷贝 | `hyle_status_clone`，禁止用 `dst = src` 伪拷贝 owning handle |
| 转移 | `hyle_status_move(&dst, &src)`；dst 必须已初始化，先释放 dst，再把 src 置 OK |
| 首错聚合 | `hyle_status_update(&overall, &next)`；消费 next 并置 OK |
| 释放 | `hyle_status_free(&s)`，NULL 安全、幂等、释放后为 OK |
| 增加上下文 | `hyle_status_add_context(&s, context)`，原地替换 owning handle |

OK 与无消息、无 detail 错误零分配；带消息或 detail 的错误共享引用计数 representation。
非法 C enum 输入归一为 `HYLE_STATUS_UNKNOWN`。

native detail 是可选的固定 `{domain, int64 code}`。domain 有 POSIX errno、Win32、Vulkan、
CUDA runtime、CUDA driver、NVRTC 和 Unknown。它不改变 one-word handle；只有实际携带 detail
的错误才在 heap rep 中多存一个 8B code。跨平台逻辑先判断 canonical status code，只有
后端专用恢复/遥测才查询 native detail。

每个 `hyle_status` 在传给 `free`、作为 `move` 的 dst 或传给 `add_context` 前，都必须由
`HYLE_STATUS_INIT`、`hyle_status_ok()`、工厂返回值或另一次合法 move 初始化。不要把未初始化
的栈变量交给这些函数：`move` 会先释放 dst 中原有的 owning handle。

## 2. DLL 导出符号

| 符号 | 签名要点 | 说明 |
|---|---|---|
| `hyle_status_ok` | `() -> hyle_status` | OK |
| `hyle_status_new` | `(code, msg) -> hyle_status` | 通用构造；NULL msg 表示无消息 |
| `hyle_status_new_n` | `(code, msg, len) -> hyle_status` | length-aware；支持嵌入 NUL、无 `strlen` |
| `hyle_status_new_native[_n]` | `(code, domain, native_code, msg[,len])` | 附加定长原生 detail |
| `hyle_status_code_from_errno` | `(errno) -> code` | O(1) canonical 分类；0 返回 OK |
| `hyle_status_from_errno` | `(errno, context) -> status` | 线程安全 native description；NULL context 可用 |
| `hyle_status_from_errno_n` | `(errno, context, len) -> status` | length-aware context |
| `hyle_status_code_from_win32` | `(uint32 code) -> code` | O(1) Win32 canonical 分类；任意 host 可用 |
| `hyle_status_from_win32` | `(code, context) -> status` | Windows 用 `FormatMessage`；保留 Win32 detail |
| `hyle_status_from_win32_n` | `(code, context, len) -> status` | length-aware context |
| `hyle_status_free` | `(status*)` | 释放并置 OK；NULL 安全 |
| `hyle_status_clone` | `(status) -> status` | 新的独立 owning handle |
| `hyle_status_move` | `(dst*, src*)` | 转移所有权 |
| `hyle_status_update` | `(overall*, next*)` | 保留首个错误；消费 next 并置 OK；NULL/self no-op |
| `hyle_status_add_context` | `(status*, context)` | 前置 `context: `；OK/NULL/空串 no-op；OOM 保留原状态 |
| `hyle_status_add_context_n` | `(status*, context, len)` | length-aware 原地 context |
| `hyle_status_is_ok` | `(status) -> int` | 1/0 |
| `hyle_status_get_code` | `(status) -> code` | canonical code |
| `hyle_status_has_message` | `(status) -> int` | 1/0 |
| `hyle_status_equals` | `(a, b) -> int` | 比较 code、native detail 和完整消息 |
| `hyle_status_code_name` | `(code) -> const char*` | 静态字符串；非法 code 返回 `"Unknown"` |
| `hyle_native_error_domain_name` | `(domain) -> const char*` | 静态 domain 名；非法输入为 `"Unknown"` |
| `hyle_status_get_native_error` | `(status, domain*, code*) -> int` | 有 detail 返回 1；输出指针可 NULL |
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
#define HYLE_C_UPDATE_STATUS(overall, expr) /* 表达式求值一次，消费并聚合 */
```

表达式只求值一次。OK 是零 representation，宏在头内直接检查 `rep != 0`，不会为热成功
路径调用 DLL；错误 handle 的所有权转移给调用者。

## 4. 常用写法

### 4.1 普通传播与语义 context

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

原生 C/POSIX 叶子直接保存 errno 并构造同一个 owning Status：

```c
int fd = open(path, O_RDONLY);
if (fd < 0) {
  int error_number = errno;
  return hyle_status_from_errno(error_number, "opening model weights");
}
```

`error_number == 0` 返回 OK。`context == NULL` 只保留 native description；返回值仍按普通
`hyle_status` clone/move/free，并自动携带 `HYLE_NATIVE_ERROR_POSIX_ERRNO` 与原始数值。若
操作合同需要更具体分类，用
`hyle_status_code_from_errno` 取得默认 code 后再做少量明确覆盖，不复制整张映射表。

Win32 叶子也必须立即保存 `GetLastError()`，再调用 `hyle_status_from_win32`。不要在保存前调用
日志、格式化或另一项 Win32 API；它们可能覆盖同一 thread-local 值。该入口在任意 host 都能
对稳定的 32-bit Win32 code 分类，Windows 上补系统描述，非 Windows 上使用数字描述；两种
情况都保留 `HYLE_NATIVE_ERROR_WIN32` detail。`hyle_status_code_from_win32` 只用于调用合同更
具体时的少量 override，不用于复制另一张大 switch。

后端原生码需要程序化处理时，创建和查询都不必解析 message：

```c
hyle_status s = hyle_status_new_native(
    HYLE_STATUS_UNAVAILABLE, HYLE_NATIVE_ERROR_VULKAN, -4, "device lost");
hyle_native_error_domain domain;
int64_t native_code;
if (hyle_status_get_native_error(s, &domain, &native_code) &&
    domain == HYLE_NATIVE_ERROR_VULKAN && native_code == -4) {
  /* recreate the Vulkan device */
}
hyle_status_free(&s);
```

跨平台策略仍先按 `HYLE_STATUS_UNAVAILABLE` 分支；native 只是更窄的二级决策面。

### 4.2 `Status + out` 冷工厂

```c
#include "hyle/c/mem.h"

hyle_status make_request_pool(hyle_shared_pool **out) {
  /* 工厂进入时会清空 *out；成功后调用方拥有 *out。 */
  return hyle_shared_pool_create_default(out, 256, 1024, 0);
}

hyle_status run(void) {
  hyle_shared_pool *pool = NULL;
  HYLE_C_RETURN_IF_ERROR(make_request_pool(&pool));

  void *block = hyle_shared_pool_allocate(pool); /* 热路：耗尽就是 NULL */
  if (block != NULL) hyle_shared_pool_deallocate(pool, block);
  hyle_shared_pool_destroy(pool);
  return hyle_status_ok();
}
```

`HYLE_C_RETURN_IF_ERROR` 对非 OK Status 做所有权转移；调用者最终负责 free。成功 Status 是零值，
宏直接继续。已有资源需要 cleanup 时使用普通分支，在 return 前先释放资源，不要用宏跳过清理。

### 4.3 首错聚合

批量执行彼此独立的步骤、需要保留第一个失败时，不要让后续错误覆盖最初原因：

```c
hyle_status overall = HYLE_STATUS_INIT;
HYLE_C_UPDATE_STATUS(overall, validate_input());
HYLE_C_UPDATE_STATUS(overall, flush_device());
HYLE_C_UPDATE_STATUS(overall, close_output());
return overall; /* ownership 交给 caller */
```

宏会把表达式求值一次并管理临时 handle。需要保存候选变量时使用
`hyle_status_update(&overall, &next)`；正常调用后 `next` 一定变为 OK，因此不要再把它当错误
读取，也不需要为其 heap backing 再 free。NULL/self 调用是 no-op；普通代码不要依赖这个
防御行为掩盖未初始化变量。

### 4.4 clone、move 与借用消息

- 两个变量都要独立拥有同一错误：`hyle_status_clone`。
- 只把所有权交给另一个变量：先用 `HYLE_STATUS_INIT` 初始化 dst，再调用
  `hyle_status_move`。
- `hyle_status_message_cstr` 只借用；原 Status 被 free/move/原地加 context 后指针失效。
- `hyle_status_add_context` 消费并原地更新唯一 owning handle：第一层精确分配，第二层起进入
  几何 prefix storage，连续 k 次的总复制为 O(最终消息字节)。每次调用后此前借用的
  `message_cstr` 都失效；不要拿 context 记录调用栈或结构化 breadcrumb，大量事件交给 obs。
- `*_n` 可保存嵌入 NUL 的文本；此时 `message_cstr` 只能看到第一个 NUL，完整字节必须用
  `hyle_status_message` 的返回长度与 buffer-fill。
- 直接 `dst = src` 只是复制 tagged word，会造成 heap Status double free；禁止。
- `hyle_status_update` 消费 `next`；不要在调用前用裸赋值另留一个同 rep 的“备份”。需要保留
  候选错误时先 `clone`。
- 每个 owning handle 最终恰好 free 一次；`hyle_status_free` 会把它重置为 OK，因此对同一
  已正确初始化的变量重复 free 是安全的。

```c
hyle_status src = hyle_status_unavailable("device lost");
hyle_status dst = HYLE_STATUS_INIT;
hyle_status_move(&dst, &src); /* src 现在是 OK */
hyle_status_free(&dst);
```

传播宏只适用于当前函数可以立即 return 的位置。如果前面已经取得文件、锁或 opaque handle，
先用普通分支完成 cleanup，再返回 Status；宏不会自动展开资源清理。

## 5. C++ 到 C 的映射

| C++ 能力 | C 路径 |
|---|---|
| `Result<T>` | `hyle_status` 返回值 + 成功出参 |
| `HYLE_RETURN_IF_ERROR` / `HYLE_ASSIGN_OR_RETURN` | `HYLE_C_RETURN_IF_ERROR` 或普通分支 |
| `Status::WithContext` | `hyle_status_add_context` |
| `Status::Update` | `hyle_status_update` / `HYLE_C_UPDATE_STATUS` |
| `Status::FromNative` / `native_error()` | `hyle_status_new_native[_n]` / `hyle_status_get_native_error` |
| C++/C representation 转移 | 仅由 HYLE 源码内部 bridge 完成；公开 C++ API 不暴露 raw rep |
| 日志、sink | 独立的 `hyle/c/obs.h` 产品面 |

## 6. 验收

```bash
cmake --build <build> --target hyle_error_c_test
ctest --test-dir <build> -R hyle_error_c_test --output-on-failure
```

纯 C 测试覆盖创建、查询、clone/move/free、非法 enum、errno/Win32/native detail、嵌入 NUL、借用
消息、上下文、首错聚合和早返回；机械 surface test 同时锁定 27 个 DLL 导出，独立头测试确保
`hyle/c/error.h` 单独以 C 编译。

二进制 SDK 的外部消费证书位于 `samples/package/error/c`：工程只启用 C，执行
`find_package(Hyle REQUIRED COMPONENTS error)` 并只链接 `hyle::error`。同一个工程必须同时通过
shared 和 static SDK；static package 会启用 C++ linker driver，但不会把 `.c` 翻译单元改成 C++。
