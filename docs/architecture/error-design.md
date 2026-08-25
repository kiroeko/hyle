# error 设计：值语义失败与跨运行时接缝

> **V2 状态：`NOT_REVIEWED`。** 本页是上一轮 error 设计输入，不是当前模块证书。当前产品只
> 提供 C++ source/static/shared；Native FFI 为 `N/A`。正文中的旧 FFI、验收状态与性能叙事在
> error 轮次必须删除或重新裁决。用户当前写法见 [user/error](../user/error.md)，施工状态见
> [implementation/status](../implementation/status.md)。

> 上一轮性能输入见 [historical error evidence](../implementation/evidence/performance/benchmarks-error.md)。
> 正文尚未按新的四类文档职责拆净。

## 0. 职责与依赖位置

error 位于运行时依赖图底部，负责把**可恢复的运营失败**作为值在函数、模块、线程以及被选中的
foreign-runtime 边界上传递。它没有 binary upstream；`result.h` 使用 core 拥有的编译期宏，
但不会因此依赖 `hyle_core` DLL。

真实场景只有四类：

1. OS、文件、内存、设备或协议叶子把原生失败转成稳定错误码和诊断；
2. 中间层在传播时增加“正在做什么”的语义上下文；
3. 上层按稳定错误码恢复，并在需要时读取一个原生 domain/code；
4. Java/.NET 等调用下游 HYLE shared artifact 后，直接接收同一个 owning error handle，查询并释放。

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
冷路径验证 UTF-8，非法时退化为确定性 ASCII `errno N`，native errno 仍完整保留。C++ `string_view` 与
Native FFI caller-buffer 都保留精确长度，允许嵌入 NUL；只有显式需要 C-string 的消费方才可把
它当 NUL 结尾文本。caller-buffer 截断是 byte prefix，可能切断多字节 code point，完整解码必须
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

## 3. Direct Native FFI

### 3.1 真实任务与边界

HYLE 没有 native-C 产品。C header 只描述让 Java/.NET 等直接调用 shared artifacts 的稳定
calling convention。error 的真实任务不是让 foreign runtime 重建整套 C++ Status API，而是：

1. 下游模块调用成功时收到 NULL；失败时收到一个 owning handle；
2. 查询 code、精确 UTF-8 message 和可选 native detail；
3. 必要时取得统一的人读 `to_string`；
4. 恰好释放一次；卸载下游与 error artifact 前不再持有 handle。

因此 `hyle_status` 是 opaque pointer，NULL 是完整 OK 表示。code-only error 映射到
`hyle_error` 内 process-lifetime sentinel，仍为零分配；message/native error 直接转移内部 rep
所有权，不再公开 raw tagged word。非 NULL handle 不可解引用、做指针运算或复制成第二个 owner。

### 3.2 唯一导出面

| export | 任务 |
|---|---|
| `hyle_error_ffi_abi_version` | 在解析其他 error symbol 前验证这个 artifact 的 ABI generation |
| `hyle_status_destroy` | 释放一个 owning handle；NULL 安全 |
| `hyle_status_get_code` | NULL→OK；其余返回 fixed-width canonical code |
| `hyle_status_get_native_error` | caller-owned POD outputs；无 detail 时写 NONE/0 |
| `hyle_status_message` | caller-buffer 精确 UTF-8 bytes；返回所需长度 |
| `hyle_status_to_string` | caller-buffer `Code[: message]`；返回所需长度 |

所有符号显式使用 C linkage、artifact export macro 与 Windows `__cdecl`。length/capacity 使用
`uint64_t`；HYLE 产品 ABI 只支持 64 位目标。buffer 规则等同 snprintf：capacity>0 时最多写
`capacity-1` bytes 并写 NUL；返回值不含终止 NUL。message 可以包含嵌入 NUL，绑定必须按长度
解码，不能先 `strlen`。

不导出 factory、clone、move、context、Update、predicate、code-name、borrowed cstr 或 OS
adapter：当前 managed receiver 任务不需要它们，暴露会制造 owning raw-handle 误用和无消费者
兼容负担。将来 managed callback 真要向 HYLE 返回 Status 时，按那条回调任务另设计，不预埋
通用构造器。

当前未收口的 util Native FFI 候选已有一个具体触发点：Registry creator 接收
`hyle_status* out_status`，但 receiving-only error FFI 只能让 foreign callback 转移另一个 HYLE
producer 给出的 owner，不能构造任意 managed failure。util review 若保留 Java/.NET callback，
必须在 `FFI=N/A`、模块级 fixed code + borrowed UTF-8 bytes（native 同步复制）与重开 error
factory/必要 ownership 操作之间作真实任务裁决，并补 OOM、异常、同步/异步生命周期和 managed
负向证据；候选头与 native-C fixture 不能冒充这项能力已经完成。

`include/hyle/c/error_helpers.h` 只是仓库 C 编译/package fixture 可选的 source-inline 便利层，
提供 NULL/predicate/`destroy-and-clear` spelling；它没有动态符号，不能计作 direct FFI 或 C 产品。
HYLE C++ 模块之间始终传 `Status/Result`，只在各模块 FFI bridge 最后一层调用内部
`status_to_ffi`。

### 3.3 artifact 与托管生命周期

binding 必须精确加载同一 SDK 中的下游 artifact 与 `hyle_error`，分别先验证各自 bootstrap；
一个 core 版本查询不能替 error 或下游 DLL/so/dylib 背书。.NET 用 zero-invalid `SafeHandle`，
Java 用受控 owner/`finally` 或 `AutoCloseable`，并确保 query 与 destroy 期间 artifact 保持加载。
ABI mismatch 必须在查找/调用普通业务 symbol 前拒绝。static/source 交付只能证明 native link；
direct managed FFI 的产品证据来自真实 shared artifact。

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

| 参考 | 借鉴 | 明确差异 |
|---|---|---|
| Abseil `Status/StatusOr` | canonical taxonomy、intrusive shared rep、unique-release fast path、非 OK error-arm invariant、errno mapping；审查 `StatusOr<T&>` lifetime guards | 不引入任意 payload/StatusBuilder；context 保留扁平渐进表示；当前借用值保持 C++23 expected wrapper 形状 |
| Folly `Expected` | 标准 expected-shaped value/error 组合、零成功分配 | 不引入 exception_wrapper 作为 HYLE operational channel |
| Boost.System/Outcome/LEAF | domain+code、checked failure 与动态 typed detail 的取舍 | HYLE 当前只需一个固定 native detail，不做动态 category/slot |
| C++23 `std::expected` | observers 与 monadic surface | 用零状态 `ResultError` 补 Status 的非 OK 不变量 |

以下只有真实消费者出现才重开：多独立机器 detail、可遍历/序列化的 structured cause、公共
multi-error、跨进程序列化、WinSock Status 冷路径、`Result<T&>` 专用 monadic 形状、重复的
std/Boost error-code adapter、managed-to-native error construction/callback。当前 util Registry
FFI 候选是最后一项的显式待裁决触发点，不是已交付能力。参考库 API 更多、LOC 更大或某个微基准
容易赢都不是功能授权。

## 6. 采用与模块边界

已严格审查的底座消费者（platform、mem、time、obs、util）必须直接使用 C++ Status/Result，
不得保留私有 TLS last-error、bool+诊断或 C FFI 内绕。CUDA/Vulkan translator 归 compute，
syscall operation override 归 platform/io，error 只拥有公共 taxonomy 和 value carrier。

尚未严格收口的 sched/io C schema 中仍有历史 TLS last-error；它们不是 error 的第二个产品面，
也不能凭现状取得 managed FFI 证书。其冷 factory/control path 必须在各自模块复审时迁移到共享
opaque `hyle_status + out`；queue empty/full、poll pending、async completion 等正常协议状态继续
用紧凑值/枚举。这个已知下游债务不授权扩大 error API。

## 7. 冻结与重开

error 签署必须同时具备：C++ correctness/三平台、同合同性能 non-inferiority、六符号 export
manifest、真实 Java/.NET exact-artifact 正负旅程、shared/static/source 交付、已审底座 adoption、
正确文档和可追踪 raw evidence。纯 C 编译或旧点值不能替代上述门。

签署后只因 correctness/security/ABI bug、真实 consumer 能力缺口、同合同强 peer 证明实质算法/
性能落后，或发布形状改变而局部重开。
