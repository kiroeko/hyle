# HYLE error：什么情况选择什么

error 是 HYLE 的统一失败值层。它帮助调用方把可恢复失败跨函数、模块、线程和语言边界
传递；它不代替日志、异常层级或程序不变量检查。先判断失败属于哪一类，再选择形状。
本文是**语言无关的选型入口**，示例使用自然的 C++ 写法；纯 C/FFI 用户完成选型后，
到 [`c-error.md`](./c-error.md) 查 `hyle_status` 的所有权与精确 ABI。

## 一页选择表

| 用户场景 | C++ 形状 | C / FFI 形状 | 不应该用什么 |
|---|---|---|---|
| 操作只有成功/失败，调用者需要原因 | `Status` | 返回 `hyle_status` | `bool + last_error` |
| 成功还要交付值或拥有型资源 | `Result<T>` | 返回 `hyle_status`，首位 out 参数交付值 | 失败指针 + TLS errno |
| 批量独立步骤，只保留第一个失败 | `Status::Update` | `hyle_status_update` / `HYLE_C_UPDATE_STATUS` | 覆盖最初原因 |
| 明确接受 cleanup 失败 | `IgnoreError` | `hyle_status_free` 前显式注释/处理 | 无意图地丢弃返回值 |
| allocator、pool pop、poll 的“暂时无值”本来就是正常热路结果 | 指针、`optional` 或小 enum | 指针/标量值合同 | 每次构造 owning Status |
| 参数、文件、权限、OOM、超时、设备暂不可用等运营失败 | canonical `Status` code | 同一 `hyle_status_code` | abort 或异常跨 ABI |
| C/POSIX 调用刚失败，手里有 errno | `status_from_errno(errno, "operation")` | `hyle_status_from_errno` | 各模块复制 `strerror` + switch |
| Win32 调用刚失败，手里有 `GetLastError()` | `status_from_win32(code, "operation")` | `hyle_status_from_win32` | 各文件复制 `FormatMessage` + switch |
| COM/WinRT 调用返回失败 `HRESULT` | `status_from_hresult(hr, "operation")` | `hyle_status_from_hresult` | 丢掉 HRESULT 或只把十六进制写进 message |
| 必须按 Vulkan/CUDA 等 SDK 原生码恢复 | `Status::FromNative` / `native_error()` | `hyle_status_new_native` / `hyle_status_get_native_error` | 解析 message 或再造私有码表 |
| HYLE 内部不变量损坏、double free、错误生命周期 | `HYLE_FATAL_ASSERT` | fail-fast 合同 | 返回 Status 后继续运行 |
| 需要写日志或 trace | 最终消费边界交给 `obs`/宿主 logger | `hyle/c/obs.h` | error 内部自动日志或第二套 handler |

判断关键不是“失败罕不罕见”，而是它是不是一个调用方需要理解和传播的**运营失败**。
例如 mem 句柄工厂、HugeRegion 和 external init 是冷控制面，返回 Status；同一对象的
`allocate/acquire` 是热数据面，容量耗尽就是 NULL。成功 Status 是一个 machine word 的
零值且零分配，因此冷控制面不需要为了性能退回 TLS。

## Status、Result 还是值合同

只有成功/失败时直接返回 `Status`：

```cpp
hyle::Status flush_index();
```

成功还要交付值时使用 `Result<T>`。它就是 `std::expected<T, Status>`，不另造容器协议：

```cpp
#include "hyle/error/result.h"

hyle::Result<Model> load_model(std::string_view path);

hyle::Result<int> run() {
  HYLE_ASSIGN_OR_RETURN(Model model, load_model("weights.bin"));
  HYLE_RETURN_IF_ERROR(model.prepare());
  return model.execute();
}
```

“没有取到东西”若属于正常状态，不应伪装成错误。例如 non-blocking poll、bounded pool
耗尽和 allocator OOM 可以按接口合同返回空值；只有调用者需要区分非法参数、后端失败等
原因的控制面才返回 Status。不要仅因某条 benchmark 路径方便测量就改变用户形状。

## canonical code 怎么选

code 表达调用者应如何处理，程序只按 code 分支，不解析 message：

| 情况 | code |
|---|---|
| 参数值或组合非法 | `InvalidArgument` |
| 索引、尺寸或算术范围越界 | `OutOfRange` |
| 参数本身合法，但对象当前状态不允许操作 | `FailedPrecondition` |
| 文件、键或设备对象不存在 | `NotFound` |
| 创建目标已存在 | `AlreadyExists` |
| OOM、配额、池或句柄耗尽 | `ResourceExhausted` |
| 后端、服务或设备暂不可用，可由上层重试 | `Unavailable` |
| 超时或显式取消 | `DeadlineExceeded` / `Cancelled` |
| 权限或身份失败 | `PermissionDenied` / `Unauthenticated` |
| 当前构建永久不支持请求的能力 | `Unimplemented` |
| 事务或并发冲突，需要重试 | `Aborted` |
| 外部依赖内部失败，但进程仍可安全继续 | `Internal` |
| 已确认数据损坏 | `DataLoss` |
| 无法可靠分类的外来失败 | `Unknown` |

`Internal` 不是掩盖 HYLE bug 的桶。若内部不变量已经损坏，调用方不能安全恢复，应
fail-fast；Status 只表示程序状态仍然有效、调用方可以决定下一步的失败。

## OS 错误边界不要重复造轮子

系统调用失败后要立刻保存 errno，再交给统一工厂；不要在其它可能修改 errno 的调用之后
才读取：

```cpp
int fd = ::open(path, O_RDONLY);
if (fd < 0) {
  const int error = errno;
  return hyle::status_from_errno(error, "opening model weights");
}
```

`status_code_from_errno` 适合已经自行构造完整 native message、只需要共享分类的 platform/
backend adapter。操作合同确实更具体时可以覆盖通用 code，例如线程创建的 `EAGAIN` 应报
`ResourceExhausted`。不要为同一个 errno 在 fs、VM、NUMA 中各维护一套近似 switch。

Win32 同理：在任何可能覆盖 thread-local last-error 的调用之前保存 `GetLastError()`，再调用
`status_from_win32(code, "operation")`。它用一张跨平台稳定的 O(1) taxonomy 分类，在 Windows
上由 `FormatMessage` 取得描述，并始终附带 `NativeErrorDomain::Win32` 和原始 32-bit code；
非 Windows 工具也能分类日志/远端传来的 Win32 code，只是描述退为确定性的数字文本。
`status_code_from_win32` 供确有更具体操作语义的 adapter 少量覆盖，不应复制整张映射表。

COM/WinRT 返回的是 signed 32-bit `HRESULT`，不是 `GetLastError()`。应直接调用
`status_from_hresult(hr, "operation")`：非负的 `S_OK`/`S_FALSE` 都是成功，
`HRESULT_FROM_WIN32` 复用同一 Win32 taxonomy，失败保留
`NativeErrorDomain::HResult` 和完整原值。不要把 HRESULT 误传给 `status_from_win32`，也不要
先调用其它 COM/Win32 API 后只留下一个手写 `Internal` 文本。

## 什么时候查看 native error

大多数代码只看 canonical code。只有动作确实依赖后端原生值时才查看 detail，例如 Vulkan
`DEVICE_LOST` 需要重建 device，而同为 `Unavailable` 的 incompatible driver 更适合直接
降级：

```cpp
hyle::Status status = hyle::Status::FromNative(
    hyle::StatusCode::Unavailable,
    {hyle::NativeErrorDomain::Vulkan, -4}, "device lost");

if (status.IsUnavailable()) {
  const hyle::NativeError native = status.native_error();
  if (native.domain == hyle::NativeErrorDomain::Vulkan && native.code == -4) {
    // destroy/recreate the affected Vulkan device
  }
}
```

不要先查 native 再忽略 canonical code；domain-specific 分支会降低可移植性。不要解析
message 取数值。errno/Win32/HRESULT 工厂及 HYLE 的 platform/CUDA/Vulkan translator 已自动
附带 detail；
用户包装其它 C library 时才需要直接调用 `FromNative`。一个错误若需要多组结构化字段，
请在上层结果/遥测对象建模，不把 Status 当任意 metadata bag。

## message、context 与日志

叶子诊断回答“底层到底发生了什么”：message 保留 operation、人读描述和必要参数，
`native_error()` 保留数值；SDK 提供稳定 symbolic name 时 message 也应保留它：

```text
vulkan vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)
```

中间层 context 回答“用户正在做什么”，只在增加新语义时添加：

```cpp
return std::move(status).WithContext("creating device buffer");
```

不要为每一层函数名机械加壳，也不要在传播途中自动写日志。最终消费边界决定重试、降级、
返回用户或交给 obs；否则同一错误经过多层时会被重复记录。

## 成本与所有权

`Status` 是一个 machine word：OK 和无消息、无 detail 的错误都零分配；带消息或 native
detail 的错误分配一块共享引用计数 representation。native domain 复用内部 padding，只有
实际 detail 才多存 8B code；普通 Status 不承担常驻 metadata。复制 heap Status 只增加
引用计数，移动转移所有权。需要 C++ owning 文本时显式包含 `hyle/error/format.h`。

```cpp
hyle::Status status = hyle::Status::Unavailable("device lost");
if (status.IsUnavailable()) {
  status = std::move(status).WithContext("submitting batch 42");
}
```

批量操作使用 `overall.Update(next)` 保留第一个错误；明确忽略 cleanup 失败时调用
`cleanup().IgnoreError()`。`HYLE_ASSIGN_OR_RETURN` 会展开成多条语句，放在
`if/else/for/while` 下必须加 `{}`。

## 常见坑

- 不要对来自用户输入、I/O 或设备操作的 `Result<T>` 直接调用 `.value()`；错误分支会抛
  `std::bad_expected_access`，绕开 HYLE 的值式失败协议。用传播宏，或先检查再移动
  `result.error()`。只有测试和已由不变量证明成功的初始化代码适合 `.value()`。
- `Result<T>` 是 `std::expected` alias。用 `hyle::make_unexpected(non_ok_status)` 构造错误；
  `Status::Ok()` 作为 error arm 是合同违反，受控构造/传播入口会 fail-fast。不要直接写
  `std::unexpected(Status::Ok())`。
- `Status::message()` 返回借用 `string_view`。原 Status 被销毁、赋值，或作为 rvalue 加
  context 后，不要继续使用先前 view；保留一份 Status copy 可以延长共享消息寿命。
- `status.WithContext(text)` 对 lvalue 保留原值并返回新 Status；
  `std::move(status).WithContext(text)` 消费原值，适合传播边界。不要机械地给每层函数名加壳。
- message 始终扁平连续。第一层 context 精确分配；从第二个 rvalue context 起使用几何增长
  的 prefix storage，连续 `std::move(status).WithContext(...)` 的总复制是 O(最终消息字节)，
  不是 O(k²)。反复对 lvalue 调用会因值语义/COW 每次复制当前消息，仍可能 O(k²)；传播链要
  移动，确实需要保留旧值时才用 lvalue。context 仍是少量用户语义，不是机器栈、breadcrumb
  或任意字段收集器；大量结构化事件交给 obs，多个同时已知的文本片段在上层一次组合。
- 不同 Status 副本可以跨线程传递；同一个 Status 对象不能在一边 `Update`/赋值的同时被
  另一线程读取。线程安全的是共享的不可变 message representation，不是对同一 C++ 对象的
  并发读写。
- `HYLE_ASSIGN_OR_RETURN` 支持声明，但会展开成多条语句；控制流下必须使用花括号。宏不会
  自动增加 context，也不会替你清理此前获得的裸资源。传入临时 Result 会移动；传入命名
  lvalue Result 会复制值/错误，不会暗中把它消费掉。

## 语言边界

C++ 的 `Result<T>` 在 C/FFI 中映射为“返回 `hyle_status` + 首位 out 参数”；C++ 的
`Status::WithContext` 映射为原地 `hyle_status_add_context`。`hyle_status` 是 owning handle，
native detail 映射为固定 domain/code；不能用普通结构体赋值伪造复制。完整的
clone/move/free、length-aware 文本、借用消息、buffer-fill 和传播宏见
[`c-error.md`](./c-error.md)。

安装后 C++ 参考工程位于 `samples/package/error/cpp`，只请求并链接 `hyle::error`；对应的真正
C-only 工程位于 `samples/package/error/c`。两者由 shared/static package smoke 自动发现和运行，
不是只在 HYLE 源码树内成立的示例。

C handle 的 `rep` 虽然为按值 ABI 而公开在结构体中，但它是 owning token，不是可复制整数：
初始化后才能作为 destination，复制必须 `clone`，转移必须 `move`，首错聚合优先使用
`HYLE_C_UPDATE_STATUS`。裸结构体赋值是最危险的 C/FFI 陷阱，会让两个变量释放同一 backing。
