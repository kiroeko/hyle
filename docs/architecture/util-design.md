# Util：基础工具与启动期注册

Util 提供几种可独立使用的 C++ 工具，而不是通用 helper 的收纳目录。它不拥有设备、模型、
调度或业务政策；已有领域 owner 能表达资源语义时，应直接使用那个 owner。用户入口统一为
`hyle::util`，接口选择与示例见 [Util 指南](../user/util.md)。

## 两个依赖层，一个公开产品

内部拆分解决依赖方向，用户不需要为每个工具选择新的产品：

| 内部层 | 能力 | 依赖与作用 |
|---|---|---|
| `util_core` | scope guard、UniqueResource、Ref、Rng、FunctionRef、NotNull、类型名、SpinLock | header-only 基础工具，供 Mem、Sched 等较低层直接复用 |
| `util_runtime` | Registry、demangle | Registry 公开使用 Error、Mem、Container；demangle 需要平台实现 |

Registry 虽然是模板，却不能据此放进 foundation：它公开使用字符串、hash table 和 allocator。
若把它塞回 `util_core`，Container 再复用基础工具时就会形成反向依赖。demangle 的二进制代码很小，
但为它再建立一个用户产品并没有独立能力边界，因此与 Registry 共用 runtime 层。
公开产品与内部依赖的实际声明见 [HyleComponents.cmake](../../cmake/HyleComponents.cmake)。

## 工具的所有权与失败语义

ScopeGuard 处理本地控制流中的临时回滚，move 后仍只有一份执行责任；`dismiss()` 放弃责任，
`invoke()` 提前消费它。构造 callable 失败不能留下已取得但无人释放的资源，转移 guard 也不能先
disarm 源对象再冒险移动。ScopeFail/ScopeSuccess 观察语言异常计数，不把 `Status` 的错误返回当异常。
通常 cleanup 应为 `noexcept`：析构时抛异常会终止程序。

UniqueResource 只服务尚无 typed owner 的 value-like native token。普通构造表示确实取得所有权；
checked factory 才比较 invalid sentinel。deleter 的转移必须有无异常的移动或复制路线，sentinel 的
完整布尔判定也不得抛异常，避免资源取得与 owner 建立之间的泄漏窗口。Platform、Mem、IO 已提供
文件、映射或 socket owner 时，不应再降为通用整数 token。

`Ref<T>` 是对象内计数、单指针的侵入式共享所有权，没有 weak reference、aliasing control block 或
环回收。`adopt_ref` 接管已有的 +1，`retain_ref` 新增一份引用，二者不能互换。普通 `make_ref` 使用
C++ `new`，可能抛出；需要 allocator 或可恢复 OOM 时，由领域 factory 返回 `Result<Ref<T>>`，并用
匹配的 `destroy()` 释放。`ThreadUnsafeRefCounted` 只适用于完整引用生命周期都被限制在同一线程的
类型；它是类型级前置条件，不是运行时快模式。`is_unique()` 只是并发快照，不授予后续独占修改权。

FunctionRef 借用一次同步调用的 callable，不存储、不排队、不延长寿命。它保持地址和 invoker 两个
机器字，支持 mutable/const 与 throwing/noexcept 签名；类型擦除层不应为大参数增加一次中间 move。
跨调用或异步任务需要 owning callable 或其 owner 定义的 fn+context record。NotNull 只表达 required
pointer，保持单指针布局；它不能修复悬空指针，也不替代普通引用、optional pointer 或 `Result<T&>`。

## Registry：同一张表的两个阶段

Registry 用于启动期组合命名 factory，然后在服务期反复查找。它不是模型算子的热路径路由器，
也不是动态库卸载管理器。

```text
mutable startup
  锁保护 register / reserve / 表读取
  factory 与 visitor 在锁外调用
             |
             | freeze：发布既有表，不复制或重建
             v
immutable service
  Registry 查询：一次 acquire + 同一张表
  FrozenView 查询：直接使用已发布的表
```

索引保存稳定 entry 指针。每个 entry 拥有 key、help、factory callable 及其 capture；Arena 把它们
作为一笔存储建立，hash capacity 在提交 entry 前准备好。若扩容失败，不应在单调 Arena 留下不可达
字符串。freeze 是单向、幂等的 O(1) 发布，不额外分配第二张表，因此不会产生随注册规模增长的
冻结期峰值或 OOM 窗口。

三档 priority 让组合顺序显式：高档覆盖，低档忽略，同档冲突报错。被覆盖的 entry 保留到 Registry
析构，旧 Creator 和正在执行的 factory 仍有效；每个 key 只可能保留有限档位的成功注册。
`find()` 返回 lookup-only EntryView，允许启动 owner 解析一次 typed factory 后缓存，不必先构造对象
或 O(n) 枚举。Creator 和 key/help 都仍借用 Registry，FrozenView 也不拥有表。

可变阶段枚举在锁内复制时间点快照，再在锁外执行 visitor；冻结阶段直接遍历不可变表。
factory/visitor 可重入查询，但 callable 的构造、销毁与 allocator 回调不能重入 Registry。
构造/销毁期重入不是普通查询并发的延伸。

factory 的结果形状决定失败语义：`Registry<T*>` 让 missing 和返回空都保持空指针；需要区分设备、
驱动或配置失败时，使用 `Registry<Result<T*>, Args...>`，missing 成为 NotFound，factory 错误原样
传播。capture 保存注册项绑定状态，`Args...` 保存每次调用输入；转发层保持值类别，不额外移动参数。

注册必须由明确的启动代码调用并处理 Status。静态初始化副作用既可能被 archive dead-strip，
也难以把 collision/OOM 交还启动 owner。Registry 不提供 unregister：安全卸载还要排空创建的对象、
借用的函数指针和并发 reader，必须由真正拥有这些资源的模块管理器承担。

## 随机数、短锁与诊断

Rng 提供显式 seed 的 xoshiro256++ 可复现流、无偏整数范围和 jump 拆流。单次 scalar draw 为 O(1)，
bounded draw 为期望 O(1)，`discard(n)` 为 O(n)。`thread_rng()` 是 best-effort 的线程便利入口，
不保证默认序列可复现、跨动态库 stream ID 唯一或区间不重叠；安全材料必须使用 OS/密码学随机源。

SpinLock 是紧凑 TTAS/backoff 锁，只适合几条指令、低争用的临界区。锁内可能分配、rehash、执行
未知 callback 或等待时，应选择 mutex；Fiber 长等待使用 Sched 原语。默认紧凑布局不强制每个对象
承担 cache-line padding，只有实际 owner 发现伪共享后才加。`cpu_relax()` 只是自旋 hint，不建立同步。

`type_name/type_hash` 来自编译器类型拼写，适合受控构建内诊断；hash 可碰撞，拼写可随工具链改变，
不能作为 wire format、数据库或 plugin 的持久身份。

demangle 是冷路径诊断。owning overload 在 caller TU 构造 `std::string`，通过窄同步 sink 接收平台
结果，避免把 DLL-owned STL 布局与堆所有权交给调用方；caller-buffer overload 返回 required size，
支持截断和输入输出重叠。两者都不保证内部零分配或无锁：Itanium demangler 可能 malloc，Windows
DbgHelp 需要串行。HYLE 的锁只协调同一 runtime image，宿主直接使用 DbgHelp 或多个静态副本时仍须
统一协调。它们均不能用于受限 signal/exception handler。

## 成本与维护边界

Ref、borrow、scope/resource owner 和短锁不增加动态分派表；Registry 查询为 O(|key|) hash 加期望
O(1) probes，注册另复制 key/help，枚举为 O(n)。可变枚举快照需要 O(n) 临时空间，冻结枚举为 O(1)
额外空间。demangle 为输入/输出长度相关的冷成本，不应移入每 token 或每元素路径。

修改时重点保持 callable 异常边界、资源转移责任、freeze 发布顺序及借用寿命；布局和并发变化再检查
对应直接消费者。公开交付是 C++ SDK，不承诺稳定跨工具链 C++ ABI，也没有单独的 C/FFI 产品面。
共同原则见 [基础机制](foundation-design.md)，构建与交付见 [构建架构](build-system.md)。
