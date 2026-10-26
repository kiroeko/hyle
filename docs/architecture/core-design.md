# Core：编译事实、程序不变量与布局

Core 是没有独立二进制的 C++ header-only 产品。它解决 runtime 建立前就需要统一表达的问题：
编译目标、编译器合同、程序不变量和静态 cache 布局。使用示例见 [Core 指南](../user/core.md)；
跨模块共同原则见[基础机制](foundation-design.md)。

## 为什么保持很小

“所有模块都可能使用”不足以成为 Core 的归属理由。分配、时钟、运行时 CPU 探测、错误值、日志和
同步都有自己的 owner、失败与并发语义，分别归对应模块。Core 不初始化 runtime，也不拥有线程、
allocator、注册表或设备状态。

| 能力 | 公开头 | 设计原因 |
|---|---|---|
| fatal/debug assertion、typed context、fatal observer | `hyle/core/assert.h` | 所有模块共享程序不变量的失败语义 |
| 当前 TU 的编译环境 | `hyle/core/environment.h` | 集中处理编译器/平台预定义宏，避免到处复制探测 |
| compiler attribute、alias/branch/loop hint | `hyle/core/compiler.h` | 只封装标准 C++ 尚不能自然表达的可移植缺口 |
| cache spacing 与 prefetch intent | `hyle/core/cache.h` | 将静态类型布局和运行时 topology 分开 |
| 编译期产品版本 | `hyle/version.h` | build/package 元数据，经 Core include contract 交付 |

根级 `hyle/visibility.h` 属 build/package 的符号交付合同，不是 Core runtime API。

## 编译事实不是运行能力

OS、architecture、compiler、endianness、build、sanitizer、analyzer、exceptions 和 RTTI facts 描述
**当前 translation unit**。布尔 facts 始终是 `0/1`，因此使用 `#if HYLE_OS_WINDOWS`，不能用
`#ifdef` 判断真假。`HYLE_BITNESS` 单独表达位数；generic feature probe 供预处理条件使用。
clang-cl 是 Clang frontend 加 MSVC ABI mode，不同时伪装成两个 primary compiler。

这些 facts 不证明运行 CPU 能执行某条指令。运行时 CPU、OS 许可、cache、NUMA 归
[Platform](platform-design.md)，多 ISA 实现选择归 [SIMD](simd-design.md)。一个 TU 的编译选项
可能允许编译器在整个 TU 使用指令集，因此不能只检查某个显式 intrinsic 对应的 feature。

## 宏只补语法缺口

已有自然标准形状时直接使用：`[[nodiscard]]`、`= delete`、`std::assume_aligned` 等不再包装成
另一套语言。拼接等预处理辅助收在 `detail/`，不作为一般用户 API。

inline、cold、branch、loop 和 prefetch 是 hint，是否采用由工具链决定。`HYLE_ASSUME`、
`HYLE_RESTRICT` 与 release `HYLE_UNREACHABLE` 则是语义承诺；违约可能产生 undefined behavior，
不能代替输入检查。assumption 表达式不能带必须发生的副作用：不同 build/compiler 的求值行为不同。
新 attribute 应解决已有消费者的可移植问题，不能仅为“宏更全”扩面。

## 程序不变量的失败路径

`HYLE_FATAL_ASSERT` 在 Debug/Release 都检查；`HYLE_DEBUG_ASSERT` 在 Debug、sanitizer 和 analyzer
TU 启用。普通 Release 的 debug family 不求值，但仍检查表达式、message 和 context field 的类型，
避免两种 build 接受不同的合法源码。需要 Release 执行的安全检查不得藏入 debug family。

条件只求值一次，message/context 仅失败时构造。context 是最多 8 个 typed field，不是任意对象的
formatting 框架；支持标量、指针、C string 与 `string_view`。事件同步借用数据，不复制 payload；
默认 formatter 对每段文本限制输出长度，避免失败路径上的无界格式化与分配。

可恢复的输入、OOM、文件、设备和协议失败归 [Error](error-design.md)。fatal 表示继续执行已不可信，
最后仍会 `abort()`。失败 sink 不承诺 async-signal-safe、backtrace、minidump 或恢复执行。

`set_fatal_observer()` 是宿主已有 crash 管线的窄接缝：

- callback 在失败线程同步运行，必须 `noexcept`，不能保留事件里的借用 view；
- 多个失败线程可能并发调用它；递归 fatal 绕过 observer，避免无限递归；
- slot 只保证同一 linked image 内一致，不构成跨 DSO 的全进程 crash manager；
- 原子替换 callback 不等待旧 callback 退出。卸载 callback 所在模块前，宿主仍须使旧调用静止。

因此 observer 不能用来构造一个隐藏的日志 runtime 或 plugin 卸载屏障。

## 静态布局与运行时 cache

cache spacing 常量进入公开类型的 `sizeof/alignof`，只能由编译目标决定，不能随运行机器探测改变。
当前布局取舍是 Apple arm64 使用 128-byte constructive/destructive spacing，其他 AArch64 与 x86-64
通常使用 64；max-destructive 保守上界为 128。实际常量以 `cache.h` 为准。

默认 destructive spacing 用于隔开独立写字段；constructive spacing 用于确有共同访问关系的数据；
max-destructive 用于少量控制字。不能把最大 spacing 机械用成密集容器 stride，否则会放大空间与
cache 压力。运行时 cache hierarchy 可帮助算法选 tile/分片，但不修改已经编译好的对象布局。

read、one-shot read 与 write intent prefetch 都是单地址、O(1)、best-effort hint。不建立同步、
不延长对象寿命、不保证 residency，也不允许访问已失效地址。具体 hint 可退化为空操作。

## 为什么没有 Core DLL 或 runtime version

`hyle::core` 在源码、静态 SDK、共享 SDK 中都是 `INTERFACE` target；它传播 include 与 C++ build
contract，没有需要 stage 的 Core runtime。改变 inline header 后必须重编消费者。

`hyle/version.h` 标识编译当前 TU 的 headers；单独链接一个版本库只能证明该库自己的版本，不能
证明进程内其他模块、inline 类型或编译选项兼容，所以不提供 linked version query。显式取得
`inline constexpr` 常量地址时工具链仍可物化 storage，这不等于存在独立 runtime。

Core 没有 native C/FFI 面；动态调用不能替代调用方 TU 的 attribute、类型布局或 inline prefetch。
构建和安装关系见[构建架构](build-system.md)，具体接口以[公开头](../../include/hyle/core/)为准。
