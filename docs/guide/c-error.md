# error direct Native FFI：托管端接收与释放失败

`hyle/c/error.h` 是 Java FFM、.NET P/Invoke 等直接绑定 `hyle_error` shared artifact 的 C-syntax
schema。它不是原生 C 产品 API，也不追求 C++ `Status` 的功能 parity。HYLE 尚未发布 binding，
当前 ABI 可继续破坏性改进；一旦某个 binding 发布，其 bootstrap 兼容规则才开始生效。

## 1. 这个 FFI 解决什么

真实旅程是：foreign runtime 调用另一个 HYLE artifact；成功收到 NULL，失败收到一个 owning
`hyle_status`；调用 `hyle_error` 查询 code/message/native detail，最后释放。foreign runtime
不创建、clone、move、加 context 或聚合 Status，这些属于 native implementation。

```text
downstream HYLE call
        │
        ├── NULL ───────────────► success
        │
        └── opaque hyle_status ─► query through hyle_error ─► destroy once
```

## 2. artifact bootstrap

精确加载实际将要调用的 `hyle_error.dll` / `libhyle_error.so` / `libhyle_error.dylib`，只解析并
调用：

```c
uint32_t hyle_error_ffi_abi_version(void);
```

high 16 bits 是 major，low 16 bits 是 minor。binding 接受相同 major 且 runtime minor 不小于它
要求的 minor。版本不兼容时，必须在解析/调用任何其他 error symbol 前拒绝。

下游 producer artifact 也必须验证它自己的 bootstrap；core 的版本查询不能替任何其他 artifact
背书。所有 artifact 必须来自一个一致 SDK。不要依赖 PATH 搜索碰巧找到同名旧 DLL。

## 3. 类型与所有权

```c
typedef struct hyle_status_handle *hyle_status;
```

- NULL 是完整 OK 表示；查询 NULL 安全，destroy NULL 安全；
- 非 NULL 是一个 immutable owning handle，最终恰好调用一次 `hyle_status_destroy(status)`；
- 不能解引用、做 pointer arithmetic、序列化或复制成第二个 owner；
- code-only handle 可能是 process-lifetime sentinel，两个 handle 可能数值相同；仍按各次返回的
  逻辑 ownership 调 destroy，不能用 pointer identity 推断 error identity；
- message/native handle 由 `hyle_error` 管理，query/destroy 期间该 artifact 必须保持加载；
- owner lifetime 被固定时可从多个线程并发 query；不能让 destroy/close 与任何 query 竞争；
- 先释放所有 status，再卸载 producer 和 `hyle_error`。

.NET 应包装为 zero-invalid `SafeHandle`；Java 应使用不可复制的 owner/`AutoCloseable`，或确保每条
路径都在 `finally` destroy。不要向业务层暴露裸 `IntPtr` / `MemorySegment` ownership。

## 4. 六个动态导出

`tests/error/c_surface_symbols.txt` 是机械清单：

| export | signature 语义 |
|---|---|
| `hyle_error_ffi_abi_version` | `uint32_t(void)` artifact bootstrap |
| `hyle_status_destroy` | 消费一个 owning handle；NULL 安全 |
| `hyle_status_get_code` | fixed-width `uint16_t` canonical code；NULL→OK |
| `hyle_status_get_native_error` | 返回 1/0；可选输出 fixed-width domain/code |
| `hyle_status_message` | 精确 UTF-8 message bytes caller-buffer copy |
| `hyle_status_to_string` | UTF-8 `Code[: message]` caller-buffer copy |

Windows 全部显式 `__cdecl`。`hyle_status_code` 与 `hyle_native_error_domain` 都是 `uint16_t` constants，
不是 compiler-sized enum。length/capacity 是 `uint64_t`；HYLE FFI 只支持 64 位目标。

没有导出的 factory、clone、move、predicate、context、Update、borrowed cstr、code-name 或 OS
adapter。它们不属于当前 receiving-binding 任务；不要在 binding 中假定旧的 raw-word C 候选仍
存在。

## 5. caller-buffer 规则

两个文本函数采用同一规则：

```c
uint64_t hyle_status_message(hyle_status status,
                             char *buffer, uint64_t capacity);
uint64_t hyle_status_to_string(hyle_status status,
                               char *buffer, uint64_t capacity);
```

1. 返回完整所需 UTF-8 byte 数，不包含结尾 NUL；
2. `buffer == NULL` 是 size-only query；
3. `buffer != NULL && capacity > 0` 时最多写 `capacity-1` bytes，再写 NUL；
4. 缓冲不足仍返回完整长度；
5. message 允许嵌入 NUL，必须按返回长度解码，不能 `strlen`；
6. Status immutable，所以 size-query 后用 `required + 1` 重试不会被该 handle 自身改变。

小 buffer 得到的是 byte prefix，可能恰好截在多字节 UTF-8 code point 中间；不要解码截断结果。
它只用于诊断/探测 truncation，拿 `required + 1` 重试成功后再解码完整 bytes。

通用读取流程：

```text
required = message(status, NULL, 0)
bytes = allocate(required + 1)
actual = message(status, bytes, required + 1)
assert actual == required
decode bytes[0..required) as UTF-8
```

调用方提供给 native API 的 message/context 由调用方保证 UTF-8；error 不在热路径验证。HYLE
产生的 Win32/HRESULT system description 经 UTF-16→UTF-8 转换；`errno` 的 locale-sensitive
description 会在冷路径验证，非法 UTF-8 退化为确定性的 ASCII `errno N`，native errno 不丢失。

## 6. code 与 native detail

先按 canonical code 恢复：

```c
hyle_status_code code = hyle_status_get_code(status);
```

只有 backend-specific recovery/telemetry 才查 detail：

```c
hyle_native_error_domain domain;
int64_t native_code;
int32_t present =
    hyle_status_get_native_error(status, &domain, &native_code);
```

无 detail 时返回 0，并把提供的 outputs 写为 `HYLE_NATIVE_ERROR_NONE` / 0；任一 output 可为 NULL。
message 是人读文本，不要从中解析 code。code/domain constants 的数字定义以 header 为准，binding
可以映射成自己的 enum，但跨 ABI 仍以 `uint16` 传递。

## 7. .NET 绑定形状

推荐结构：

1. `NativeLibrary.Load(exactErrorPath)`；只 `GetExport` bootstrap 并验证；
2. 精确加载 producer 并验证它自己的 bootstrap；
3. 为 source-generated `LibraryImport` 安装只接受这两个逻辑名的 resolver；
4. producer 返回的非零 pointer 立即包进 zero-invalid `SafeHandle`；
5. query 期间用 `DangerousAddRef`/生成器支持的 SafeHandle 参数固定 lifetime；
6. `ReleaseHandle` 只调用 `hyle_status_destroy(handle)`；
7. 先 dispose 所有 SafeHandle，再 free producer/error libraries。

不要让多个 SafeHandle 包装同一个返回值，不要靠 finalizer 作为正常生命周期。可运行样例见
[`samples/ffi/error/dotnet`](../../samples/ffi/error/dotnet)。

## 8. Java FFM 绑定形状

推荐结构：

1. 用 confined/shared `Arena` 对 exact path 建立 `SymbolLookup.libraryLookup`；
2. lookup/call 每个 artifact 的 bootstrap，兼容后才 lookup 普通 symbols；
3. `MemorySegment.NULL` 表示成功；非 NULL 放进不可复制 owner；
4. caller-buffer 使用 arena 分配 `required + 1`，按 byte length 解码 UTF-8；
5. owner 的 `close()` 或 `finally` 调用 destroy，防止异常路径泄漏；
6. 所有 handle 释放后再关闭承载 library lookup 的 Arena。

不要用 `getUtf8String` 读取可能含嵌入 NUL 的 message。可运行样例见
[`samples/ffi/error/java`](../../samples/ffi/error/java)。

## 9. native-C source helper 不是 FFI

仓库自己的 C compiler/package fixture 可额外包含：

```c
#include "hyle/c/error_helpers.h"
```

它只提供 NULL initializer、code predicates 和 `destroy-and-clear` 等 `static inline` spelling。
这些函数不会出现在 DLL/so/dylib export table，不能被 Java/.NET lookup，也不代表 HYLE 面向 C
客户提供 source API。业务 binding 应直接映射六个动态导出。

## 10. 验收与常见错误

error FFI 的最低证据包含：

- C11 schema/header 自包含、六符号 manifest 与真实 export table 一致；
- test-only producer 返回 NULL、code-only、含嵌入 NUL message + native detail 三种状态；
- Windows Java 25 与 .NET 8 exact-path 正向旅程；
- incompatible artifact 只含坏 bootstrap，binding 在 lookup 普通 symbol 前拒绝；
- Linux/macOS 对真实 shared artifact 做 bootstrap、最小 query/destroy；
- shared package 中 producer/error runtime dependencies 可解析，static/source 只作 native 交付验证。

常见错误：

- 把 `hyle_status` 当结构体或 C++ `Status` layout；
- 裸复制 owning pointer，再 destroy 两次；
- 未固定 owner lifetime 就并发 query，或让 query 与 destroy/close 竞争；
- 忘记 NULL 是 OK，先创建一个多余“OK object”；
- 在 bootstrap 前预绑定所有 symbols；
- 只验证 core artifact；
- 用 PATH/`java.library.path` 搜索而不是 exact artifact；
- 用 C-string API截断嵌入 NUL message；
- 卸载 `hyle_error` 后再 query/finalize SafeHandle；
- 把 `error_helpers.h` 的 inline 函数列为动态能力。
