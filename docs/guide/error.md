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
| 批量独立步骤，只保留第一个失败 | `Status::Update` | 显式保留第一个非 OK Status | 覆盖最初原因 |
| 明确接受 cleanup 失败 | `IgnoreError` | `hyle_status_free` 前显式注释/处理 | 无意图地丢弃返回值 |
| allocator、pool pop、poll 的“暂时无值”本来就是正常热路结果 | 指针、`optional` 或小 enum | 指针/标量值合同 | 每次构造 owning Status |
| 参数、文件、权限、OOM、超时、设备暂不可用等运营失败 | canonical `Status` code | 同一 `hyle_status_code` | abort 或异常跨 ABI |
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

## message、context 与日志

叶子 message 回答“底层到底发生了什么”，应保留 operation、原生错误名/数值和必要参数：

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

`Status` 是一个 machine word：OK 和无消息错误都零分配；只有带消息错误分配一块共享
引用计数 representation。复制带消息 Status 只增加引用计数，移动转移所有权。需要 C++
拥有型文本或 stream 输出时显式包含 `hyle/error/format.h`，避免让便利依赖进入所有热路径 TU。

```cpp
hyle::Status status = hyle::Status::Unavailable("device lost");
if (status.IsUnavailable()) {
  status = std::move(status).WithContext("submitting batch 42");
}
```

批量操作使用 `overall.Update(next)` 保留第一个错误；明确忽略 cleanup 失败时调用
`cleanup().IgnoreError()`。`HYLE_ASSIGN_OR_RETURN` 会展开成多条语句，放在
`if/else/for/while` 下必须加 `{}`。

## 语言边界

C++ 的 `Result<T>` 在 C/FFI 中映射为“返回 `hyle_status` + 首位 out 参数”；C++ 的
`Status::WithContext` 映射为原地 `hyle_status_add_context`。`hyle_status` 是 owning handle，
不能用普通结构体赋值伪造复制。完整的 clone/move/free、借用消息、buffer-fill 和传播宏见
[`c-error.md`](./c-error.md)。
