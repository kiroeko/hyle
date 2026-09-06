# error 设计：值语义失败与模块接缝

> 本页描述 consumer-driven 的 Error 长期产品边界；候选、验证或终态只由 machine-readable registry
> 裁决。当前只提供 C++23 source/static/shared；Native FFI 为 `N/A`。公开 C schema、bridge、C-shaped
> 动态导出和 managed fixture 不属于产品。用户写法见
> [user/error](../user/error.md)，施工状态只看 [implementation/status](../implementation/status.md)。

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

- 64 位目标上 `sizeof(Status) == 8`；OK 与 code-only error 都是 low-bit tagged inline word，
  构造、复制、销毁均不分配、不做原子操作；
- message/native error 使用一个不可变、单块分配、intrusive-refcounted representation；
- 复制 heap Status 是 O(1) retain，移动只转移一个 word；moved-from 对象成为 code-only
  `Unknown`，不会伪装成 OK；可把不同 Status 对象跨线程传递；
- 所有 Status 自身操作 `noexcept`；OOM 至少保留 canonical code，可丢可选 message/detail；
- `StatusCode::Ok` 不携带消息；非法 code/domain 在受信边界归一为 `Unknown`；
- message 是借用 `std::string_view`，随 owning Status 存活；不把 owning STL 类型跨 DLL 接缝。

inline representation 的 low bit 为 1（包括 OK），aligned heap pointer 的 low bit 为 0；这与成熟
`Status` carrier 的 tagged-pointer 形状一致，使 copy/destroy 只需一次 bit-test，不依赖进程地址范围。
live `Status` 的内部 word 永不为 0；0 只供 allocation helper 表示 OOM/overflow，不能被 adopt 成
对象。原始 tag 值不是公开 API 或稳定 ABI。为了让常见 heap copy 与成熟 intrusive carrier 同档，
consumer header 内联一次 relaxed retain。
因此更换 tag 编码必须原子更新匹配版本的 header/artifact，并全量重编直接消费者；不得混用旧 header
与新 shared library。项目尚未承诺稳定 C++ ABI，ExactVersion 交付允许这次预发布迁移。
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
WinSock、HRESULT、Vulkan、CUDA runtime/driver 与 NVRTC 边界。它不是动态 category 或多 payload map。
普通恢复只看 canonical code；backend-specific recovery/遥测才看 native detail。禁止解析 message
取得机器值。

`<hyle/error/system_error.h>` 在拥有第三方调用的模块边界把 `std::error_code` 单向降为 Status，error
仍不把 `error_category*` 或 category registry 放进 Status。generic/POSIX 与 Windows system category
分别落到 errno/Win32 domain；custom category 默认 canonical `Unknown`，或由 owner 通过
`status_from_error_code_as` 提供 operation-specific code，且不保留不稳定的 category/numeric identity。
category identity 若仍参与机器恢复，就继续保留模块 typed error，不能经诊断 message 伪造 round-trip。
generic/system route 在 `system_error.h` 的 caller-side inline adapter 中识别，再把闭合 route 交给 Error
artifact；不能在 HYLE DLL 内拿 consumer 的 `error_category*` 与 DLL 自己的标准库 singleton 比地址。
这保证 source/static/shared 的标准 category 语义一致，同时 custom category 仍由实际 owner 明确降级。

### 2.3 message、UTF-8 与 context

公共文本合同是 UTF-8。调用方传入的 message/context 必须已经是 UTF-8；error 不在热错误路径
重新验证。Windows 原生描述使用 `FormatMessageW` 后转换为 UTF-8，不能把当前 ANSI code page
字节送入 managed runtime。`strerror_r`/Windows CRT errno 文本受进程 locale 影响：error 在该
冷路径验证 UTF-8，非法时退化为确定性 ASCII `errno N`，native errno 仍完整保留。C++
`string_view` 与 caller-buffer 格式化都保留精确长度，允许嵌入 NUL；只有显式需要 C-string 的
消费方才可把它当 NUL 结尾文本。caller-buffer 截断是 byte prefix，可能切断多字节 code point，完整解码必须
按返回长度重试。

运行时路径、尺寸和设备号可由 opt-in `<hyle/error/format.h>::format_status` 直接形成 Status。它使用
1 KiB caller-side stack scratch，超长输出带 `...[truncated]`，custom formatter exception 被降为确定性
诊断，Status OOM 仍只丢 message。它不记录日志、source location 或 payload，也不进入最小
`status.h` 编译闭包。

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

error 提供不依赖 OS public headers 的四组基础入口：

```cpp
StatusCode status_code_from_errno(int) noexcept;
Status status_from_errno(int, std::string_view operation = {}) noexcept;
StatusCode status_code_from_win32(std::uint32_t) noexcept;
Status status_from_win32(std::uint32_t, std::string_view operation = {}) noexcept;
StatusCode status_code_from_winsock(int) noexcept;
Status status_from_winsock(int, std::string_view operation = {}) noexcept;
StatusCode status_code_from_hresult(std::int32_t) noexcept;
Status status_from_hresult(std::int32_t, std::string_view operation = {}) noexcept;
```

classification 是固定 switch 的 O(1)；成功值返回零值 OK。factory 在错误路径取得线程安全/OS
描述并形成一次最终 Status allocation；OOM 保留 code。非 Windows 主机仍能确定性分类持久化的
Win32/WinSock/HRESULT 值并产生数值文本。operation-specific 语义可覆盖 canonical code，例如
`pthread_create(EAGAIN)` 比通用 I/O EAGAIN 更准确地表示资源耗尽；但不能复制整套 taxonomy。

WinSock 明确不并入 Win32 domain：`WSAGetLastError` 是不同错误域。IOCP 已构成真实消费者，因而
Error 提供独立 O(1) taxonomy/factory；IO 在自己的轮次决定 raw completion 到 owning Status 的冷边界。
未来 fixed domain 只有在具名 consumer、稳定 numeric contract 和 Error review 同时存在时才追加；
不建立运行时 registration 或依赖 DSO lifetime 的 category pointer。

### 2.5 `Result<T>` 与传播

非引用 `Result<T>` 使用标准 `std::expected` 的 representation 与 value/monadic surface，但错误槽是
与 `Status` 同尺寸的受检 `detail::ResultError`。公开层是统一 class-template family，而不是 selector
alias：

```cpp
template<class T>
class Result;      // owning/void: zero-extra-state std::expected route

template<class T>
class Result<T&>;  // non-null caller-owned borrow route
```

它组合一个 `Status`，能隐式接收/按值交付 Status，却不暴露可变 `Status&`；因此任何 error-arm
construction/assignment 都会 fail-fast 拒绝 `Status::Ok()`，调用方也不能经基类 upcast 绕过检查。
常用的 `message/native_error/IsXxx/WithContext` 仍直接可用。统一 class template 让
`template<class T> f(const Result<T>&)` 对 owning 与 borrow 都能正常推导；owning specialization 与底层
`std::expected` 保持相同 size/alignment、allocation 和 hot-path code shape，但不承诺 exact C++ type
identity。这样同时保留标准 expected 操作，并把“错误槽不得为 OK”从传播宏的迟到检查提升为不可绕过
的类型不变量。HYLE 未发布，不为此前任何 type identity 保留兼容层。

它也保留标准 `expected` 的默认构造语义：当 `T` 可默认构造时，`Result<T>{}` 是**成功**并包含
value-initialized `T`，不是 empty、unknown 或未设置的错误；当 `T` 不可默认构造时该写法不可用。
这属于 C++23 value contract，不能为了减少一个易用性陷阱而给 carrier 增加第三状态。

C++23 `std::expected` 禁止 reference value；HYLE 因而只对 `Result<T&>` 选择一个 header-only、内部以
非空 pointer/error 存储的 borrow carrier。它从 compatible lvalue 或 unexpected error 构造，拒绝
temporary/null，提供 expected-shaped observer、`and_then/transform/or_else` 与传播宏；不分配，也不
拥有或延长 referent lifetime。const carrier 是 deep-const：`const Result<T&>` 只能观察 `const T&`，
mutable borrow 也不能从 const carrier 复制出来。`Result<const T&>` 正常可复制；显式 default 是
code-only `Unknown`，因为 reference 没有 value-initialized success。

nullable 对象仍用有明确 NULL 合同的 `Result<T*>`，连续范围/文本用 `Result<std::span<T>>` /
`Result<std::string_view>`。浅 const 或必须显式表现 rebinding 的高级场景仍可把
`std::reference_wrapper<T>` 作为普通 value；不能用 pointer 偷偷表达“成功必非空”。非引用
`Result<T>` 不因 borrow specialization 增加状态、allocation 或 runtime route。

普通失败返回使用 `make_unexpected(Status)`；也允许直接 `std::unexpected(Status)`。两个宏只
补 C++ 缺少的 `?` 语法：

```cpp
HYLE_RETURN_IF_ERROR(expr);               // Status 或 Result<void>
HYLE_ASSIGN_OR_RETURN(auto value, expr);  // Result<T>
```

表达式只求值一次；rvalue 移动、命名 lvalue 复制；外层可返回 Status 或 Result。宏局部名由
`__COUNTER__` 唯一化。声明型 `ASSIGN_OR_RETURN` 展开为多条语句，在无花括号 if/loop 下不得使用。
宏不自动加 context、日志或 source location。若一个命名局部在失败分支随函数返回而结束生命周期，
调用方应显式写 `HYLE_RETURN_IF_ERROR(std::move(local_status))`：这表达所有权转移并避免 heap error 的
retain/release；成员、引用参数或仍需保留的 lvalue 不得这样移动。

### 2.6 组合与格式化

- `overall.Update(next)` 保留第一项非 OK；prvalue/rvalue 是正常路径，lvalue 表示保留原错误；
- `status.IgnoreError()` 明确标记无法再报告的清理失败；
- `Status::ToString(buffer, cap)` 为 allocation-free caller-buffer 格式化；
- opt-in `hyle/error/format.h::to_string` 提供 owning `std::string`，长输出用
  `resize_and_overwrite` 避免先零填充再覆盖；
- 同一 opt-in header 的 `format_status(code, fmt, args...)` 在 no-throw failure boundary 直接创建包含
  运行时值的有界诊断，不引入 StatusBuilder/logger。

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
| `format_status` | O(min(F, 1023)) | 1 KiB stack + 至多一个 Status block；超长显式截断 |
| `Result<T&>` construct/check/propagate | O(1) | pointer + Status error arm + discriminator；0 allocation |
| errno/Win32/WinSock/HRESULT classify | O(1) | 固定 switch；0 allocation |
| `std::error_code` adapter | O(message bytes) cold | generic/system 一个 Status block；custom message 可临时分配 |
| N 项 first-error Update | O(N) | 仅首错发生一次 move/copy |
| 深度 d Result propagation | O(d) | 每层常数分支/移动 |

不允许为 benchmark 给普通 Status 增字段、TLS cache、payload map或虚表。性能对拍必须按
code-only 与 carrying-diagnostics 两种同信息合同分账；Abseil arbitrary payload 与 HYLE 单个
native detail 不同合同，只能作机制诊断，不能宣称产品输赢。

## 5. 成熟实现裁决

能力/API 形状主要参考 Abseil `Status/StatusOr`、Folly `result` 与 C++23 `std::expected`；具体锁定
revision、candidate 和结果属于 current closure/performance evidence，不写入长期架构。

| 参考 | 借鉴 | 明确差异 |
|---|---|---|
| Abseil `Status/StatusOr` | canonical taxonomy、intrusive shared rep、unique-release fast path、非 OK error-arm invariant、errno mapping、reference-result temporary guards | 不引入任意 payload/有日志副作用的 StatusBuilder；context 保留扁平渐进表示 |
| Folly `result` | deep-const reference result、reference lifetime/copy 边界 | 不引入 rich exception hierarchy、coroutine stopped state或 exception_wrapper 作为 HYLE operational channel |
| C++23 `std::expected` | observers 与 monadic surface | 用零状态 `ResultError` 补 Status 的非 OK 不变量 |

负空间固定为：多独立 machine detail、可遍历/序列化的 structured cause、公共 MultiError、
跨进程序列化、任意 runtime category registry、反向 error-code round-trip、source location/
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

error 签署必须同时具备：C++ correctness/三平台、适用的同合同性能门、
shared/static/source C++ 交付、已审底座 adoption、正确文档和可追踪 raw evidence。

当前资格、原始证据身份与有限结论从 [status](../implementation/status.md) 的 Engineering performance
链接进入。任何公开面、正常成本或消费者 provenance 的 material 改变都必须产生新的 qualification，
不能从旧证据外推；mechanism diagnostic 不授权改变 Status 的正常产品形状。

签署后只因 correctness/security/ABI bug、真实 consumer 能力缺口、同合同强 peer 证明实质算法/
性能落后，或发布形状改变而局部重开。
