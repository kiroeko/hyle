# Error：可恢复失败的值语义

Error 将 OS、文件、资源、设备和协议失败作为值跨函数、模块与线程传递。它回答三件事：失败应如何
恢复、底层原生错误是什么、当前操作增加了什么有用上下文。使用方法见 [Error 指南](../user/error.md)。

## 三种通道不合并

| 情况 | 通道 | 理由 |
|---|---|---|
| 可恢复运营失败 | `Status` / `Result<T>` | 调用方决定重试、拒绝或降级 |
| 人读运行诊断 | Obs | 调用方决定何时输出与输出到哪里 |
| 继续执行已不可信的程序不变量 | Core fatal assertion | 不能将损坏状态包装成可恢复错误 |

`Status` 不产生 I/O，不读钟，不保存全局 handler 或 TLS last-error。队列空/满、poll pending 等若属于
正常操作协议，可以直接用值/枚举；不能为了接口统一而全部转成错误。

## 小值与诊断 owner

`Status` 在 64-bit 目标是一个 machine word。OK 与 code-only error 内联编码，构造、复制、销毁
不分配、不做 refcount 原子操作。message/native detail 使用单块、不可变、intrusive-refcounted
representation：复制 O(1) retain，移动转移 word，唯一 owner 的销毁避开原子 RMW。

所有 Status 自身操作 `noexcept`。诊断分配失败时至少保留 canonical code，可丢弃可选文本/detail；
不能因为构造错误说明再次 OOM 而丢失原来的恢复动作。moved-from Status 是 code-only `Unknown`，
不伪装成成功。

不同 Status 对象可以共享 backing 并跨线程使用；这不授权无同步地同时修改同一个 Status 对象。
`message()` 返回借用 view，寿命受 owning Status 约束。文本使用 UTF-8、精确长度，可含 NUL，
不是始终可直接交给 C-string API。

tag 值不是公开协议。consumer header 内联 retain，因此 refcount 前缀是已编译代码合同，不能声称
representation 完全不暴露。匹配 headers 与 artifact 是同一个交付单元；替换 shared library
不保证旧 inline consumer 自动兼容，当前不承诺稳定 C++ ABI。

## Canonical code 与单个原生 detail

canonical code 使用 Abseil/gRPC 的稳定分类，供跨平台恢复。`InvalidArgument` 是输入本身非法；
`FailedPrecondition` 是当前状态不允许；`ResourceExhausted` 是资源不足；`Unimplemented` 是明确请求
尚不支持；`Unavailable` 是外部能力暂不可用。Hyle 自身不变量损坏不能一律降成 `Internal` 后继续。

少数恢复或诊断需要一个 `{NativeErrorDomain, int64 code}`。当前固定 domain 覆盖 errno、Win32、
WinSock、HRESULT、Vulkan、CUDA runtime/driver 与 NVRTC。它不是任意 payload map 或运行时 category
registry。普通恢复看 canonical code；需要原生动作时才看 native detail，不解析 message 获取机器值。

固定 errno/Win32/WinSock/HRESULT classifier 不引入 OS 头到 public surface，按常量集合分类；
factory 只在错误路径获取原生描述。WinSock 与 Win32 是不同 domain，不能合并。具体操作的 owner
可以按语义覆盖 generic 分类，例如创建线程的 `EAGAIN` 可表示资源耗尽。

`std::error_code` 是单向适配：caller-side inline adapter 识别 generic/system category，再交给
Error artifact。不能在 Hyle DLL 里拿 consumer 的 `error_category*` 与自己的标准库 singleton 比
地址。custom category 可降为 `Unknown` 或 owner 指定的 code，但不保留不稳定 category identity；
若业务仍依赖该 identity，就保留 typed error，不能假装经文本 round-trip。

## Context 为什么是扁平渐进存储

叶子文本说明底层失败；中间层只在增加操作语义时添加 context，不机械堆叠函数名或 source location。
最终文本为 `outer: inner: leaf`，code/native detail 不变。OK、空 context 原样返回，OOM 保留原 Status。

第一层 context 精确分配；后续连续 rvalue 传播可升级为几何 prefix storage。唯一 owner 且空间足够
时原地 prepend；共享或 lvalue 路线 copy-on-write。这使正常连续移动链的累计复制为 O(最终文本长度)，
扩容次数 O(log 最终长度)。保留每层历史再从 lvalue 派生新值可能累计平方复制，这是共享历史的真实
成本，不应与移动传播一起宣称“零开销”。

Windows 原生描述先从 wide text 转 UTF-8；受 locale 影响的 errno 文本若不是有效 UTF-8，退为确定的
数值说明。调用方自己传入的 UTF-8 不在每次构造重新验证，避免把诊断 carrier 变成文本处理模块。

## `Result<T>` 只有成功或失败

owning/void Result 使用 C++23 `std::expected` 的基本表示与 monadic 形状，并以受检 `ResultError`
维持“不允许 OK 进入错误槽”的不变量。错误 arm 可交付 Status，却不暴露可变 `Status&`，所以不能
通过赋值或 upcast 绕过检查。它不是完全相同的 C++ 类型，也没有额外第三种“尚未设置”状态。

当 `T` 可默认构造，`Result<T>{}` 成功并含 value-initialized `T`；不能把默认值误认为空错误。
`Result<T&>` 单独表达非空借用，因为标准 expected 不接受 reference value：

- 只接受 compatible lvalue，拒绝 temporary/null，不拥有或延长 referent 寿命；
- const carrier 只允许观察 const referent，不能从它复制出 mutable borrow；
- `Result<const T&>` 可以复制；显式默认为 code-only `Unknown`；
- 可空成功使用 `Result<T*>`，范围/文本使用 span/string_view，不用空指针暗中破坏非空借用合同。

传播宏仅补语言缺少的简洁语法：`HYLE_RETURN_IF_ERROR` 与 `HYLE_ASSIGN_OR_RETURN`。表达式只求值
一次；rvalue 移动、lvalue 复制；不自动加日志、context 或 source location。声明型 ASSIGN 展开为多条
语句，必须注意 if/loop 花括号。命名局部在返回时可明确移动，但不能随意移动仍由外部 owner 使用的
成员或引用参数。

## 聚合与格式化保持显式

`Update` 保留第一项非 OK，`IgnoreError` 表明已显式放弃清理错误。caller-buffer `ToString` 不分配；
opt-in `error/format.h` 才引入 owning string 和有界 `format_status`。公共 Status 不自动携带全部错误、
调用栈或异常对象。

需要所有错误时，所属模块选择 caller-owned vector/固定报告，自己定义排序、上限和收集 OOM。
需要结构化遥测时交给 Obs。公共 MultiError、cause graph、跨进程序列化、任意 payload registry 和
自动 logger 都需要新的真实语义依据，不能因参考库存在就加入。

## 成本与模块边界

| 操作 | 时间/资源 |
|---|---|
| OK/code-only 构造、检查、复制、销毁 | O(1)，零分配、零原子 |
| 带文本/native 的构造 | O(文本长度)，单块分配 |
| heap copy / move | O(1)，copy retain；move 不 retain |
| 连续 unique-rvalue context 链 | O(最终文本长度) 累计复制与空间 |
| 单次 shared/lvalue context | O(已有文本 + 新 context)，copy-on-write |
| caller-buffer 格式化 | 与实际输出长度成正比，零分配 |
| Result 检查/传播 | O(1) 加所含 value 的真实移动/析构成本 |

性能比较须对齐 code-only、携带诊断和原生 detail 的语义；不能用无诊断值击败保存任意 payload 的
carrier 后宣称同等。改变共享/生命周期语义也不是正常的局部优化。

CUDA/Vulkan 翻译归 Compute，syscall operation override 归 Platform/IO；Error 只拥有 taxonomy 与
carrier。异步错误不自动证明设备工作已经退休，取消也不等于资源可立即复用，这些属于执行 owner。
当前交付 C++ source/static/shared，没有 C ABI。构建见[构建架构](build-system.md)，源合同见
[Status](../../include/hyle/error/status.h) 与 [Result](../../include/hyle/error/result.h)。
