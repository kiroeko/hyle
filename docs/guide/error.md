# HYLE error：什么情况用什么

error 是 HYLE 的共享失败模型。C++ 用户从 `Status` / `Result<T>` 开始；Java/.NET 等只在调用
shared artifact 时使用 [`c-error.md`](./c-error.md) 的 opaque receiver handle。HYLE 没有原生 C
产品 API。

## 一页选择表

| 你的情况 | 选择 | 不要选择 |
|---|---|---|
| 操作只成功或失败，失败可恢复 | `Status` | bool + TLS last-error |
| 成功还要返回一个值 | `Result<T>` | output 没初始化却返回 bool |
| “没有值”本来就是正常结果 | `optional`、nullable/view 或模块枚举 | `NotFound`，除非合同真的把它定义为失败 |
| poll 未就绪、队列空/满、try-lock 失败 | 模块的紧凑值/枚举合同 | 每次构造 Status |
| 调用者违反不可恢复合同 | `HYLE_FATAL_ASSERT` | `Internal` 后继续运行 |
| 给运营人员记录事件 | obs log/trace | 在 Status 构造/传播中自动打印 |
| 需要按后端原生值恢复 | 先看 canonical code，再看 `native_error()` | 解析 message |
| foreign runtime 收到下游失败 | opaque `hyle_status`，查询后 destroy | 复制 raw handle 或绑定 C++ layout |

## 最小 C++ 路径

只返回失败：

```cpp
#include "hyle/error/status.h"

hyle::Status open_model(std::string_view path) {
  if (path.empty())
    return hyle::Status::InvalidArgument("model path is empty");
  return hyle::Status{};
}
```

返回值或失败：

```cpp
#include "hyle/error/result.h"

hyle::Result<Model> load_model(std::string_view path) {
  HYLE_RETURN_IF_ERROR(validate_path(path));
  HYLE_ASSIGN_OR_RETURN(auto bytes, read_model(path));
  return decode_model(bytes);
}
```

`Result<T>` 有标准 `std::expected` 的 observers 和 monadic 操作。错误槽是 Status-compatible 的
受检类型：任何 `std::unexpected(Status::Ok())` 都会在构造或赋值 Result 时 fail-fast，而不会
形成“既是 error 又是 OK”的非法值。`result.error()` 直接提供常用只读观察和 `WithContext`，但
刻意不允许取得可变 `Status&`；需要独立 Status 时按值复制或移动：

```cpp
hyle::Status error = std::move(result).error();
```

## `Status`、`Result` 与正常状态

判断标准是调用合同，不是“有没有返回值”：

- `Status`：成功没有额外 value；
- `Result<T>`：成功必须交付一个 `T`；
- `optional<T>`/NULL：缺失就是正常分支，例如 probe 没命中；
- 枚举：操作协议有多个正常状态，例如 ready/pending/closed；
- fatal assert：继续执行会违反内存安全或内部不变量。

不要为了统一表面把热路径的 empty/full/pending 变成 owning error。反过来，冷 factory 的 OOM、
非法 options、OS 初始化失败不能压成同一个 NULL 后再查私有字符串。

## code 怎么选

恢复逻辑只按 `StatusCode` 分支。最容易混淆的几组：

- `InvalidArgument`：输入值不合法；`FailedPrecondition`：当前对象状态不允许；
- `NotFound`：请求的资源必须存在但不存在；正常 probe miss 应是空值合同；
- `ResourceExhausted`：内存/容量/句柄确实耗尽；`Unavailable`：外部资源当前不可用；
- `Unimplemented`：当前构建/平台不提供明确请求的能力；
- `Internal`：外部依赖给出不可细分的运营失败；HYLE 自身损坏应 fail-fast；
- `Cancelled`：显式取消；`DeadlineExceeded`：时间预算耗尽。

未知外部 code 保留为 `Unknown`，不要为了“可能重试”猜成 `Unavailable`。

## OS 与后端边界

在捕获原生错误后立刻转换，避免中间调用覆盖 `errno`/`GetLastError`：

```cpp
int fd = ::open(path, flags);
if (fd < 0) {
  const int error_number = errno;
  return hyle::status_from_errno(error_number, "opening model");
}
```

Win32 使用 `status_from_win32`，COM/WinRT HRESULT 使用 `status_from_hresult`；不要互换。
Win32/HRESULT 系统描述会从 UTF-16 转为 UTF-8。`status_from_errno` 会验证 locale-sensitive
`strerror` 文本；若它不是合法 UTF-8，则退化为确定性的 ASCII `errno N`，同时完整保留 native
errno。CUDA/Vulkan 等 SDK-specific translator 由对应模块提供，再通过 `Status::FromNative`
保存稳定 domain/code。

普通调用方：

```cpp
if (status.IsUnavailable()) retry_later();
```

只有确实要做 backend-specific 恢复时才读取 detail：

```cpp
const hyle::NativeError native = status.native_error();
if (native.domain == hyle::NativeErrorDomain::Vulkan && native.code == -4)
  rebuild_device();
```

message 是人读 UTF-8，不能作为机器协议或从中解析 error number。

## 什么时候增加 context

叶子描述失败操作和原因；每个真正增加用户语义的边界前置一层：

```cpp
auto result = allocate_device_memory(bytes);
if (!result) {
  return hyle::make_unexpected(
      std::move(result).error().WithContext("creating tensor storage"));
}
```

最终文本类似：

```text
loading model: creating tensor storage: vulkan vkAllocateMemory failed
```

正常传播用 rvalue/`std::move`，连续多层总复制是最终文本的 O(n)。对共享 lvalue 调用 context 会
copy-on-write；反复保留每层快照再从 lvalue 生成下一层可累计 O(k²)，只有确实需要历史值时这样做。

不要机械附加函数名、源文件行号或每层相同文本。需要 stack/backtrace 属于崩溃诊断，不属于
普通 Status。

## 聚合、忽略与格式化

保留第一项失败：

```cpp
hyle::Status overall;
overall.Update(close_input());
overall.Update(close_output());
return overall;
```

正常写法传 prvalue/rvalue；传 lvalue 表示你仍要保留原 Status。若析构/rollback 已无调用方可接收
次要失败，写 `cleanup().IgnoreError()`，让忽略意图可审查；不要裸 `(void)`。

无分配格式化：

```cpp
char text[256];
const std::size_t required = status.ToString(text, sizeof(text));
```

它总返回完整所需长度，缓冲区不足会截断并 NUL 结尾。需要 owning string 时才包含
`hyle/error/format.h`：

```cpp
std::string text = hyle::to_string(status);
```

截断单位是 byte，可能停在 UTF-8 多字节 code point 中间；不要解码截断结果。需要完整文本时按
返回的 `required + 1` 重新分配，或直接使用 `to_string`。

日志仍由调用方把 code/message/native detail 交给 obs；Status 自身不会输出。

## 成本速查

| 形状 | 正常成本 |
|---|---|
| OK / code-only | 8B，零分配、零原子 |
| message/native | 一个 heap block；复制 O(1) retain，移动一个 word |
| `swap` | 交换一个 word；零 retain/release |
| unique heap destroy | acquire load + free；共享时才 atomic decrement |
| rvalue context chain | 累计 O(最终文本字节)，稳定空间 O(最终字节) |
| caller-buffer format | 零分配；按实际写入字节线性 |
| Result 成功传播 | 标准 expected-shaped 分支/移动，无 error allocation |

message/context OOM 时 HYLE 保留 code，可能丢可选诊断。因此恢复逻辑永远不能依赖 message
存在；需要机器恢复的数据应进入 canonical code 或一个 native detail。

## 常见坑

- 不要对尚未检查的 `Result<T>` 直接调用 `.value()`；失败时标准 `std::expected` 会抛
  `std::bad_expected_access`，绕开 HYLE 的值式传播。使用传播宏，或先检查再访问 value/error。
- `Status::message()` 是 borrowed view；原 Status 被销毁/赋值后不要继续使用。
- moved-from `Status`/`Result` 仍可销毁或重新赋值，但其旧业务值不可再使用；HYLE 将 moved-from
  error 置为 code-only `Unknown`，以免 Result 的 error arm 变成 OK。
- 不要并发读写同一个 Status 对象；复制出独立 Status 后可跨线程只读/销毁。
- caller-supplied message/context 必须是 UTF-8；库不在错误热路替你校验。
- message 可包含嵌入 NUL；需要精确 bytes 时使用 `string_view::size()`，不要 `strlen`。
- `HYLE_ASSIGN_OR_RETURN` 可声明变量但展开为多条语句；裸 `if/else/loop` 必须加花括号。
- `HYLE_RETURN_IF_ERROR` 不自动加 context；在有新语义的边界显式 `WithContext`。
- `Status::Ok()` 不能放进 Result error arm；这是程序错误，会 fail-fast。
- 不要用异常承载 HYLE operational failure；标准 allocator/用户 callback 的标准异常合同应在所属
  模块边界处理，而不是污染 Status。
- 不要为了日志长期保存 borrowed message；保存整个 Status 或复制所需文本。

## Direct Native FFI

Java/.NET 等 receiving binding 只需要 NULL/opaque handle、查询和 destroy，不需要复制 C++ API。
六个真实动态导出、buffer/ownership、bootstrap 与托管写法全部见
[`c-error.md`](./c-error.md)。`hyle/c/error_helpers.h` 是仓库 native-C fixture 的 source-inline
便利层，不是 managed ABI，也不是面向 C 客户的产品。
