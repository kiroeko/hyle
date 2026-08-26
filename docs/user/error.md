# HYLE error：什么情况用什么

> error 是 HYLE 当前的共享失败模型；本页跟随 error-v3 冻结的 C++ 产品合同。完整源码、static
> 与 shared SDK 使用相同的 `Status` / `Result<T>` 语义。项目没有原生 C 产品或受支持的
> Native FFI，早期 error C schema/bridge 已移除。

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

传播宏保留值类别：prvalue/rvalue 会移动，lvalue 会复制。若你先把错误放进一个命名局部，而失败时
函数立即返回、之后不会再观察该局部，可显式转移它：

```cpp
hyle::Status status = run_step();
HYLE_RETURN_IF_ERROR(std::move(status));
```

这会把 owning message backing 直接交给调用者；不要对成员、引用参数或失败返回后仍需由其他对象观察
的 Status 使用 `std::move`。

`Result<T>` 有标准 `std::expected` 的 observers 和 monadic 操作。错误槽是 Status-compatible 的
受检类型：任何 `std::unexpected(Status::Ok())` 都会在构造或赋值 Result 时 fail-fast，而不会
形成“既是 error 又是 OK”的非法值。`result.error()` 直接提供常用只读观察和 `WithContext`，但
刻意不允许取得可变 `Status&`；需要独立 Status 时按值复制或移动：

```cpp
hyle::Status error = std::move(result).error();
```

一个容易漏掉的标准语义是：只要 `T` 可默认构造，`Result<T>{}` 就是**成功**，其 value 是
value-initialized `T`；它不表示 empty、unknown 或“尚未设置”。需要正常缺失状态时把
`optional<T>` 放在成功侧，需要失败时显式返回 `make_unexpected(Status)`。若 `T` 不可默认构造，
`Result<T>{}` 本身不可用。

## `Status`、`Result` 与正常状态

判断标准是调用合同，不是“有没有返回值”：

- `Status`：成功没有额外 value；
- `Result<T>`：成功必须交付一个 `T`；
- `optional<T>`/NULL：缺失就是正常分支，例如 probe 没命中；
- 枚举：操作协议有多个正常状态，例如 ready/pending/closed；
- fatal assert：继续执行会违反内存安全或内部不变量。

不要为了统一表面把热路径的 empty/full/pending 变成 owning error。反过来，冷 factory 的 OOM、
非法 options、OS 初始化失败不能压成同一个 NULL 后再查私有字符串。

### 借用值怎么返回

`Result<T>` 直接采用 C++23 `std::expected` 的 value 形状，因此 `Result<T&>` **不能实例化**。
按成功值的真实合同选择：

| 成功值合同 | 形状 |
|---|---|
| 必须非空的单对象借用 | `Result<std::reference_wrapper<T>>`，返回 `std::ref(object)` |
| 指针本来允许为空 | `Result<T*>`，并明确成功 NULL 的含义 |
| 连续范围借用 | `Result<std::span<T>>` |
| 文本借用 | `Result<std::string_view>` |

这些 `Result` 只拥有失败侧的 Status，**不拥有**成功侧指向的对象或存储；owner 必须活过
`reference_wrapper`、pointer、span 或 string view 的全部使用。不要用 `Result<T*>` 偷偷表达
“一定非空”，也不要从临时容器返回 span/string view。若多个真实 API 反复需要引用专用的
monadic/lifetime 语义，再重开统一 borrowed-result 设计，而不是在调用点各造一种隐式所有权。

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

### `std::error_code` 与 custom category

HYLE 不保存 `std::error_category*`，也没有一个假装能无损接收任意 category 的通用构造器。
category identity 可能属于某个动态库，不能成为跨 DLL、卸载边界或未来外语运行时边界的长期状态。
在拥有外部调用的边界、category 仍然有效时完成转换：

- `std::generic_category()` / POSIX system error 映到 `status_from_errno`；
- Windows `std::system_category()` 的 Win32 值映到 `status_from_win32`；
- Boost.System、Asio 或第三方 custom category 由所属模块把其 condition/code 映成 canonical
  `StatusCode`，只有已经登记为稳定 HYLE domain 的原生数值才进入 `NativeError`；
- 若调用者的恢复逻辑必须保留 custom category 的类型和值，就暂时保留
  `std::error_code` 或模块自己的 `std::expected<T, E>`，不要先擦成 Status 再解析 message。

`error_code::message()` 可能分配，返回的本地化字节也不保证满足 HYLE 的 UTF-8 合同；不要在
`noexcept` 边界盲目调用并直接塞入 Status。抛出 `std::system_error` 的 API 应在拥有该调用的
异常边界捕获并按上面规则转换，异常绝不能越过 HYLE `noexcept` 或模块边界。若两个以上真实
模块反复需要同一组 std/Boost 映射，再把它收成独立的冷路径 adapter header；不要增加普通
Status 的常态字段。

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

`Update` 只回答“哪一个失败代表整个操作”，不会保存全部失败。批量校验或并行任务真的需要把
每项独立失败都交给调用者时，由所属模块使用 caller-owned `std::vector<Status>`、固定容量报告
或它已有的容器，并明确排序、上限和收集过程中 OOM 的合同；结构化遥测字段交给 obs。不要把
message 拼接后当作可解析的 multi-error，也不要让所有 Status 为少数聚合场景承担 list 元数据。
若多个模块重复需要同一套 bounded collection/ordering 语义，再重开公共 `MultiError` 设计。

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

- `Result<T>{}` 在 `T` 可默认构造时是持有 `T{}` 的成功值，不是空 Result；不要把默认构造当作
  延迟初始化或未知失败。
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
- `HYLE_RETURN_IF_ERROR(named_status)` 会复制 lvalue；只有该局部在失败时生命周期确实结束，才写
  `HYLE_RETURN_IF_ERROR(std::move(named_status))` 转移所有权。
- `Status::Ok()` 不能放进 Result error arm；这是程序错误，会 fail-fast。
- 不要用异常承载 HYLE operational failure；标准 allocator/用户 callback 的标准异常合同应在所属
  模块边界处理，而不是污染 Status。
- 不要为了日志长期保存 borrowed message；保存整个 Status 或复制所需文本。
