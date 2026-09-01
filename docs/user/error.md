# HYLE error：什么情况用什么

> error 是 HYLE 当前的共享失败模型；本页跟随 error-v5 的 C++ 产品合同。完整源码、static
> 与 shared SDK 使用相同的 `Status` / `Result<T>` 语义。项目没有原生 C 产品或受支持的
> Native FFI，早期 error C schema/bridge 已移除。

## 一页选择表

| 你的情况 | 选择 | 不要选择 |
|---|---|---|
| 操作只成功或失败，失败可恢复 | `Status` | bool + TLS last-error |
| 成功还要返回一个值 | `Result<T>` | output 没初始化却返回 bool |
| 成功必须返回非空、非 owning 单对象借用 | `Result<T&>` | nullable pointer 或暴露 `reference_wrapper` 细节 |
| “没有值”本来就是正常结果 | `optional`、nullable/view 或模块枚举 | `NotFound`，除非合同真的把它定义为失败 |
| poll 未就绪、队列空/满、try-lock 失败 | 模块的紧凑值/枚举合同 | 每次构造 Status |
| 调用者违反不可恢复合同 | `HYLE_FATAL_ASSERT` | `Internal` 后继续运行 |
| 给运营人员记录事件 | obs log/trace | 在 Status 构造/传播中自动打印 |
| 需要按后端原生值恢复 | 先看 canonical code，再看 `native_error()` | 解析 message |
| failure text 要加入路径/尺寸/设备号 | `format_status` | 为拼字符串引入 logger/stream builder |

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

非引用 `Result<T>` 有标准 `std::expected` 的 observers 和 monadic 操作。错误槽是 Status-compatible 的
受检类型：任何 `std::unexpected(Status::Ok())` 都会在构造或赋值 Result 时 fail-fast，而不会
形成“既是 error 又是 OK”的非法值。`result.error()` 直接提供常用只读观察和 `WithContext`，但
刻意不允许取得可变 `Status&`；需要独立 Status 时按值复制或移动：

```cpp
hyle::Status error = std::move(result).error();
```

`Result` 是统一的 class template，因此 generic helper 可以自然推导 owning 或 borrowed value：

```cpp
template<class T>
bool succeeded(const hyle::Result<T>& result) {
  return result.has_value();
}
```

owning route 与底层 `std::expected` 具有相同 size/alignment 和运行成本，但 HYLE 不把 exact C++ type
identity 当作交付合同；需要标准 carrier 的外部 API 应在自己的边界显式选择 `std::expected`。

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

按成功值的真实合同选择：

| 成功值合同 | 形状 |
|---|---|
| 必须非空的单对象借用 | `Result<T&>`，直接返回 owner 中的 lvalue |
| 指针本来允许为空 | `Result<T*>`，并明确成功 NULL 的含义 |
| 连续范围借用 | `Result<std::span<T>>` |
| 文本借用 | `Result<std::string_view>` |

```cpp
hyle::Result<Model&> find_model(ModelRegistry& registry, ModelId id) {
  if (Model* model = registry.find(id)) return *model;
  return hyle::make_unexpected(hyle::Status::NotFound("model id"));
}
```

`Result<T&>` 提供 `has_value`、`*`、`->`、`value`、`error`、`and_then/transform/or_else` 和两个传播宏；
它内部只保存一个非空 pointer 或普通 Result error，不分配。它只拥有失败侧 Status，**不拥有或延长**
成功对象生命周期；不能从 temporary 构造。const 是 deep-const：`const Result<T&>` 只能得到
`const T&`，也不能复制回一个可写 borrow；需要可普通复制的只读结果时直接返回 `Result<const T&>`。
引用 Result 的显式默认构造是 code-only `Unknown`，不是成功，因为 reference 没有可 value-initialize
的对象。

pointer/span/string view 同样不拥有存储。不要用 `Result<T*>` 偷偷表达“一定非空”，也不要从临时
容器返回 borrow/span/string view。只有确实需要浅 const/rebinding 语义时，才把
`std::reference_wrapper<T>` 作为普通 owning Result value；它不再是非空 borrow 的默认写法。

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

Win32 使用 `status_from_win32`，WinSock/IOCP 的 `WSAGetLastError()` 使用
`status_from_winsock`，COM/WinRT HRESULT 使用 `status_from_hresult`；三个 numeric namespace 不可
互换。必须紧跟失败调用保存 raw code，不能先做一次日志/分配再读取 last error。
Win32/WinSock/HRESULT 系统描述会从 UTF-16 转为 UTF-8。`status_from_errno` 会验证 locale-sensitive
`strerror` 文本；若它不是合法 UTF-8，则退化为确定性的 ASCII `errno N`，同时完整保留 native
errno。CUDA/Vulkan 等 SDK-specific translator 由对应模块提供，再通过 `Status::FromNative`
保存稳定 domain/code。

### `std::error_code` 与 custom category

HYLE 不保存 `std::error_category*`，但提供单向 cold adapter；包含
`hyle/error/system_error.h`：

```cpp
std::error_code error;
std::filesystem::rename(from, to, error);
if (error) return hyle::status_from_error_code(error, "renaming model cache");
```

category identity 可能属于某个动态库，不能成为跨 DLL、卸载边界或未来外语运行时边界的长期状态。
adapter 在调用期间读取它，并按以下规则转换：

- `std::generic_category()` / POSIX system error 映到 `status_from_errno`；
- Windows `std::system_category()` 的 Win32 值映到 `status_from_win32`；
- Boost.System、Asio 或第三方 custom category 默认得到 `Unknown` 和人读诊断；所属模块知道准确语义
  时使用 `status_from_error_code_as(StatusCode::..., error, operation)`；custom numeric 不进入
  `NativeError`，以免伪装成稳定 domain；
- 若调用者的恢复逻辑必须保留 custom category 的类型和值，就暂时保留
  `std::error_code` 或模块自己的 `std::expected<T, E>`，不要先擦成 Status 再解析 message。

adapter 是 header-inline caller boundary：它先在创建/拥有 `std::error_code` 的 C++ 模块识别本地
generic/system singleton，再跨 HYLE shared-library seam。不要把 error code 保存到插件卸载之后再转换；
第三方 DSO 自有的 category 应在该 DSO 仍存活的 owner 边界用 `_as` 明确 canonical 语义。

`error_code::message()` 可能分配、抛异常或产生非 UTF-8 字节；adapter 捕获 custom `message()` 异常，
非法文本退化为确定性 numeric 诊断，且不改变 8B Status。抛出 `std::system_error` 的 API 仍应在拥有
该调用的异常边界捕获，再把其中的 `code()` 交给 adapter；异常不能越过 HYLE `noexcept`/模块边界。

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

failure text 需要运行时值时，同一 header 提供 no-throw helper：

```cpp
return hyle::format_status(
    hyle::StatusCode::NotFound,
    "model {} shard {} is missing", model_name, shard);
```

它使用 1 KiB stack scratch，最多创建一个 Status backing；超长输出以 `...[truncated]` 结束。编译期
format string 会拒绝参数不匹配；启用异常时，抛异常的 custom formatter 被降为
`[status formatting failed]`。它不打印日志、不附 source location。需要完整的超长文本时先用项目的
owning formatter 构造 string，再显式传给 Status；不要依赖 bounded helper 保存全部内容。

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
| `Result<T&>` | pointer + Status error arm + discriminator；零分配、不拥有 referent |
| `format_status` | 1 KiB stack + 至多一个 Status block；O(min(输出, 1023)) |
| errno/Win32/WinSock classifier | 固定 O(1) switch；零分配 |

message/context OOM 时 HYLE 保留 code，可能丢可选诊断。因此恢复逻辑永远不能依赖 message
存在；需要机器恢复的数据应进入 canonical code 或一个 native detail。

## 常见坑

- `Result<T>{}` 在 `T` 可默认构造时是持有 `T{}` 的成功值，不是空 Result；不要把默认构造当作
  延迟初始化或未知失败。
- `Result<T&>{}` 是显式 code-only `Unknown`；它与 owning Result 的默认成功不同。不要从 temporary
  构造 borrow，也不要让 owner 先于 borrow 销毁。
- `const Result<T&>` 是 deep-const；若要跨 const carrier 保留可写权限，重新检查 ownership，而不是
  `const_cast`。浅 const 是显式 pointer/reference-wrapper 合同。
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
