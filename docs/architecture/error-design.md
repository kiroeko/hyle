# error 设计：值语义失败与模块接缝

> **error-v3 当前设计合同。** 本页描述本轮 `AUDITED` 后冻结的产品边界，不提前充当候选或
> 收口证书。当前只提供 C++23 source/static/shared；Native FFI 为 `N/A`。BUILD-03 已删除旧
> C schema、bridge、C-shaped 动态导出和 managed fixture。用户写法见
> [user/error](../user/error.md)，施工状态只看 [implementation/status](../implementation/status.md)。

> error-v3 的待资格合同与上一轮历史输入分账见
> [performance evidence](../implementation/evidence/performance/benchmarks-error.md)。

## 0. 职责与依赖位置

error 位于运行时依赖图底部，负责把**可恢复的运营失败**作为值在函数、模块和线程之间传递。
其公开头直接使用 Core 编译期合同，因此中央图声明
`error -> PUBLIC core`；`hyle::core` 是 INTERFACE，没有 Core DLL 或 runtime state。

真实场景只有三类：

1. OS、文件、内存、设备或协议叶子把原生失败转成稳定错误码和诊断；
2. 中间层在传播时增加“正在做什么”的语义上下文；
3. 上层按稳定错误码恢复，并在需要时读取一个原生 domain/code。

error 不负责日志、trace、异常层级、崩溃报告、取消状态机、RPC 序列化或任意 payload registry。

## 1. 三条失败通道

| 情况 | 唯一通道 | 例子 |
|---|---|---|
| 可恢复运营失败 | `Status` / `Result<T>` | OOM、文件不存在、设备暂不可用 |
| 人读运行诊断 | `obs` | 应用记录一次重试或任务失败 |
| 程序错误/合同违反 | `HYLE_FATAL_ASSERT` | 越界、错误线程、损坏的不变量 |

`Status` 是数据，不产生 I/O；error 不持全局 handler、锁或 TLS last-error。正常的空、未就绪、
队列满/空、poll 无结果等若属于操作协议，应使用值/枚举合同，不能为统一表面而强制创建 Status。

## 2. C++ 产品面

### 2.1 `Status` 的值与所有权

`Status` 是一个 machine word：

```cpp
Status ok;
Status missing = Status::NotFound();            // code-only，零分配
Status failed = Status::Unavailable("device lost");
```

长期合同：

- 64 位目标上 `sizeof(Status) == 8`；OK 为零；
- code-only error 以内联 tagged word 表示，构造、复制、销毁均不分配、不做原子操作；
- message/native error 使用一个不可变、单块分配、intrusive-refcounted representation；
- 复制 heap Status 是 O(1) retain，移动只转移一个 word；moved-from 对象成为 code-only
  `Unknown`，不会伪装成 OK；可把不同 Status 对象跨线程传递；
- 所有 Status 自身操作 `noexcept`；OOM 至少保留 canonical code，可丢可选 message/detail；
- `StatusCode::Ok` 不携带消息；非法 code/domain 在受信边界归一为 `Unknown`；
- message 是借用 `std::string_view`，随 owning Status 存活；不把 owning STL 类型跨 DLL 接缝。

为了让常见 heap copy 与成熟 intrusive carrier 同档，consumer header 内联一次 relaxed retain。
因此 `StatusRepHeader::refcount` 位于 allocation offset 0 是明确的**已编译代码前缀合同**；不是
“representation 完全不暴露”。allocation、flags、message/native/context layout 和 final free
仍只在 `hyle_error` 内。匹配版本的 header 与 artifact 是一个交付单元。

唯一 owner 的销毁先 acquire-load refcount；为 1 时直接释放，避免常见路径的 atomic RMW。
共享 backing 才走 `fetch_sub`。同 backing copy-assignment 不做 release/retain；同 backing equality
直接返回 true。

### 2.2 code 与原生 detail

canonical `StatusCode` 使用 Abseil/gRPC 的稳定集合，回答跨平台恢复动作。典型选择：

- `InvalidArgument`：输入值本身非法；
- `FailedPrecondition`：值可合法，但当前状态不允许操作；
- `ResourceExhausted`：内存、句柄、队列或设备资源耗尽；
- `Unavailable`：外部服务、驱动或设备当前不可用；
- `Unimplemented`：当前构建或后端不支持明确请求的能力；
- `Internal`：外部依赖失败且无法更精确分类；HYLE 自身不变量损坏应 fail-fast；
- `Unknown`：不可信输入或确实无法分类的外部错误。

少数恢复路径还需要原生数值：

```cpp
Status status = Status::FromNative(
    StatusCode::Unavailable,
    {NativeErrorDomain::Vulkan, -4},
    "device lost");
NativeError native = status.native_error();
```

`NativeError` 只保存一个固定宽 `{domain, int64 code}`。domain 覆盖现有 POSIX errno、Win32、
HRESULT、Vulkan、CUDA runtime/driver 与 NVRTC 边界。它不是动态 category 或多 payload map。
普通恢复只看 canonical code；backend-specific recovery/遥测才看 native detail。禁止解析 message
取得机器值。

`std::error_code` / Boost.System custom category 必须在拥有第三方调用的模块边界转换，error 不把
`error_category*` 或 category registry 放进 Status。generic/POSIX 与 Windows system category 可分别
落到 errno/Win32 adapter；custom category 由所属模块映 canonical code，只有稳定登记的 HYLE
domain 才保留原生数值。category identity 若仍参与机器恢复，就继续保留模块 typed error，不能
经诊断 message 伪造 round-trip。重复的真实 std/Boost 消费可触发独立 cold adapter header，但不
授权扩大 8B carrier 或普通构造热路。

### 2.3 message、UTF-8 与 context

公共文本合同是 UTF-8。调用方传入的 message/context 必须已经是 UTF-8；error 不在热错误路径
重新验证。Windows 原生描述使用 `FormatMessageW` 后转换为 UTF-8，不能把当前 ANSI code page
字节送入 managed runtime。`strerror_r`/Windows CRT errno 文本受进程 locale 影响：error 在该
冷路径验证 UTF-8，非法时退化为确定性 ASCII `errno N`，native errno 仍完整保留。C++
`string_view` 与 caller-buffer 格式化都保留精确长度，允许嵌入 NUL；只有显式需要 C-string 的
消费方才可把它当 NUL 结尾文本。caller-buffer 截断是 byte prefix，可能切断多字节 code point，完整解码必须
按返回长度重试。

叶子 message 回答“底层为什么失败”，中间层只在增加用户可理解语义时前置 context：

```cpp
return std::move(status).WithContext("creating device buffer");
```

合同：OK/空 context 零分配原样返回；code/native detail 不变；最终消息按
`outer: inner: leaf` 扁平连续存储；OOM 保留原 Status。不要机械添加函数名或 source location。

第一层 context 精确分配。第二层 rvalue propagation 才升级为几何 prefix storage；唯一 owner
且 prefix 足够时原地 prepend，共享/lvalue 走 copy-on-write。由此连续 rvalue 链的累计复制与
稳定空间均为 `O(final bytes)`，扩容次数 `O(log final bytes)`。单次 lvalue context 是
`O(current + context)`；反复保留共享快照再从 lvalue 生成新值，必然可能累计 O(k²)，这是显式
共享历史的成本，不是正常传播形状。

### 2.4 OS 适配

error 提供不依赖 OS public headers 的三组基础入口：

```cpp
StatusCode status_code_from_errno(int) noexcept;
Status status_from_errno(int, std::string_view operation = {}) noexcept;
StatusCode status_code_from_win32(std::uint32_t) noexcept;
Status status_from_win32(std::uint32_t, std::string_view operation = {}) noexcept;
StatusCode status_code_from_hresult(std::int32_t) noexcept;
Status status_from_hresult(std::int32_t, std::string_view operation = {}) noexcept;
```

classification 是固定 switch 的 O(1)；成功值返回零值 OK。factory 在错误路径取得线程安全/OS
描述并形成一次最终 Status allocation；OOM 保留 code。非 Windows 主机仍能确定性分类持久化的
Win32/HRESULT 值并产生数值文本。operation-specific 语义可覆盖 canonical code，例如
`pthread_create(EAGAIN)` 比通用 I/O EAGAIN 更准确地表示资源耗尽；但不能复制整套 taxonomy。

WinSock 当前不并入 Win32 domain：`WSAGetLastError` 是不同错误域，而已关闭模块没有 Status
消费者需要它。io 的冷控制面迁移到 Status 时，若真实恢复逻辑需要，再增加独立 WinSock
domain/factory；不得把 WSA code 猜成普通 GetLastError。

### 2.5 `Result<T>` 与传播

`Result<T>` 使用标准 `std::expected` 的 value/monadic surface，但错误槽是与 `Status` 同尺寸的
受检 `detail::ResultError`：

```cpp
template<class T>
using Result = std::expected<T, detail::ResultError>;
```

它组合一个 `Status`，能隐式接收/按值交付 Status，却不暴露可变 `Status&`；因此任何 error-arm
construction/assignment 都会 fail-fast 拒绝 `Status::Ok()`，调用方也不能经基类 upcast 绕过检查。
常用的 `message/native_error/IsXxx/WithContext` 仍直接可用。这样保持标准 expected API/codegen，
同时把“错误槽不得为 OK”从传播宏的迟到检查提升为不可绕过的类型不变量。HYLE 未发布，不为此前
“直接以 `Status` 作为 error type”的 type identity 保留兼容层。

它也保留标准 `expected` 的默认构造语义：当 `T` 可默认构造时，`Result<T>{}` 是**成功**并包含
value-initialized `T`，不是 empty、unknown 或未设置的错误；当 `T` 不可默认构造时该写法不可用。
这属于 C++23 value contract，不能为了减少一个易用性陷阱而给 carrier 增加第三状态。

C++23 `std::expected` 禁止 reference value，因此 `Result<T&>` 不是当前产品形状。非空对象借用用
`Result<std::reference_wrapper<T>>`，nullable 对象用有明确 NULL 合同的 `Result<T*>`，连续范围/
文本用 `Result<std::span<T>>` / `Result<std::string_view>`；这些成功值都不延长 owner 生命周期。
Abseil `StatusOr<T&>` 的 lifetime-bound reference specialization 已作为成熟对照审查，但当前树没有
一个引用 Result 消费者。只有多个真实 API 证明 wrapper 明显破坏 monadic/lifetime 易用性时，才
重开统一 borrowed-result 形状，不能为 API 数量脱离消费另造 expected 实现。

普通失败返回使用 `make_unexpected(Status)`；也允许直接 `std::unexpected(Status)`。两个宏只
补 C++ 缺少的 `?` 语法：

```cpp
HYLE_RETURN_IF_ERROR(expr);               // Status 或 Result<void>
HYLE_ASSIGN_OR_RETURN(auto value, expr);  // Result<T>
```

表达式只求值一次；rvalue 移动、命名 lvalue 复制；外层可返回 Status 或 Result。宏局部名由
`__COUNTER__` 唯一化。声明型 `ASSIGN_OR_RETURN` 展开为多条语句，在无花括号 if/loop 下不得使用。
宏不自动加 context、日志或 source location。

### 2.6 组合与格式化

- `overall.Update(next)` 保留第一项非 OK；prvalue/rvalue 是正常路径，lvalue 表示保留原错误；
- `status.IgnoreError()` 明确标记无法再报告的清理失败；
- `Status::ToString(buffer, cap)` 为 allocation-free caller-buffer 格式化；
- opt-in `hyle/error/format.h::to_string` 提供 owning `std::string`，长输出用
  `resize_and_overwrite` 避免先零填充再覆盖。

需要收集全部错误、并行错误排序或结构化日志时，所属模块使用 caller-owned
`std::vector<Status>`、固定容量报告或已有容器，并显式定义顺序、上限与收集 OOM；结构化遥测交给
obs。只有多个真实模块重复同一 bounded collection/ordering 合同时才重开公共 `MultiError`，不要
把 Status 变成全局 error list 或自动 logger。

## 3. 交付与 Native FFI 边界

error 当前只提供 C++23 source/static/shared 交付。HYLE 模块和同工具链消费者直接使用
`Status`/`Result<T>`；source/header 与对应 artifact 是一个版本化交付单元。

Native FFI 当前为 `N/A`。BUILD-03 已删除早期 `hyle_status` schema、helper、bridge、动态导出以及
C/Java/.NET fixture；它们不是兼容面，也不能作为新模块绕回 C handle 的理由。将来只有已命名的
foreign-runtime consumer 通过全局 [Native FFI 准入策略](native-ffi-policy.md) 后，才设计满足该
任务的窄 runtime 类型擦除边界；不预埋 Status parity exports。

## 4. 算法与复杂度合同

设 `m` 为消息长度，`F` 为最终 context 文本字节数，`N` 为聚合项数。

| 操作 | 时间 | 分配/常态成本 |
|---|---:|---|
| OK/code-only construct/check/copy/destroy | O(1) | 0 分配、0 原子 |
| message/native construct | O(m) | 一个 heap block |
| heap copy/move | O(1) | copy 一次 relaxed retain；move 无原子 |
| swap | O(1) | 交换一个 word；无 retain/release |
| heap destroy | O(1) | unique load+free；shared atomic decrement |
| shared-rep equality | O(1) | pointer identity shortcut |
| 不同 rep equality | O(m) worst | code/native/message compare |
| unique rvalue context chain | O(F) total | O(F) space，O(log F) growth allocations |
| 单次 lvalue/shared context | O(current+context) | COW；反复保留历史可累计 O(k²) |
| caller-buffer format | O(min(F, cap)) | 0 分配；size-only O(1) |
| owning format | O(F) | 通常一次 string allocation |
| N 项 first-error Update | O(N) | 仅首错发生一次 move/copy |
| 深度 d Result propagation | O(d) | 每层常数分支/移动 |

不允许为 benchmark 给普通 Status 增字段、TLS cache、payload map或虚表。性能对拍必须按
code-only 与 carrying-diagnostics 两种同信息合同分账；Abseil arbitrary payload 与 HYLE 单个
native detail 不同合同，只能作机制诊断，不能宣称产品输赢。

## 5. 成熟实现裁决

error-v3 的可复查 peer 身份固定为 Abseil commit
`81c993af4fb88b3a51829a796df02251fa27102f`（本地
`C:/public/project-hyle/abseil-cpp`）与 Folly commit
`423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`（本地
`C:/public/project-hyle/folly`）；C++23 `std::expected` 是标准合同，不算第三个性能 peer。
本轮没有把未初始化的 Boost.System/Outcome/LEAF 源码冒充已读 peer 证据。

| 参考 | 借鉴 | 明确差异 |
|---|---|---|
| 上述固定 Abseil `Status/StatusOr` | canonical taxonomy、intrusive shared rep、unique-release fast path、非 OK error-arm invariant、errno mapping；审查 `StatusOr<T&>` lifetime guards | 不引入任意 payload/StatusBuilder；context 保留扁平渐进表示；当前借用值保持 C++23 expected wrapper 形状 |
| 上述固定 Folly `Expected` | 标准 expected-shaped value/error 组合、零成功分配 | 不引入 exception_wrapper 作为 HYLE operational channel |
| C++23 `std::expected` | observers 与 monadic surface | 用零状态 `ResultError` 补 Status 的非 OK 不变量 |

本轮负空间固定为：多独立 machine detail、可遍历/序列化的 structured cause、公共 MultiError、
跨进程序列化、WinSock/任意第三方 category adapter、`Result<T&>` 专用实现、source location/
backtrace、异常层级与 Native FFI construction/callback。只有具名消费者证明当前 code/native/
context、caller-owned aggregate 或 borrowed-result 组合无法表达需求时才重开；参考库 API 更多、
LOC 更大或某个微基准容易赢都不是功能授权。

## 6. 采用与模块边界

已严格审查且存在可恢复失败面的底座消费者（platform、mem、obs、util）必须直接使用 C++ Status/Result；
time 当前没有需要 error carrier 的公开失败面，不增加伪依赖。
不得保留私有 TLS last-error 或 bool+诊断。CUDA/Vulkan translator 归 compute，
syscall operation override 归 platform/io，error 只拥有公共 taxonomy 和 value carrier。

下游冷 factory/control path 应直接使用共享 `Status`/`Result<T>`；queue empty/full、poll pending、
async completion 等正常协议状态继续用紧凑值/枚举。旧模块私有 TLS last-error 或 bool+诊断若在
模块复审中被发现，应迁移到这套 C++ 错误系统；这不授权扩大 error API。

## 7. 冻结与重开

error 签署必须同时具备：C++ correctness/三平台、同合同性能 non-inferiority、
shared/static/source C++ 交付、已审底座 adoption、正确文档和可追踪 raw evidence。

签署后只因 correctness/security/ABI bug、真实 consumer 能力缺口、同合同强 peer 证明实质算法/
性能落后，或发布形状改变而局部重开。
