# Core 架构：header-only 编译与布局地基

> 本页描述当前长期设计，不是验收证书。当前产品状态只见
> [`module-status.json`](../implementation/module-status.json)，并由其中的 `record` 指向当前轮
> task；历史施工、三平台、codegen、package 与红队记录只作追溯。性能结论边界见
> [`benchmarks-core`](../implementation/evidence/performance/benchmarks-core.md)，用法见
> [`user/core`](../user/core.md)。

## 定位

Core 是 HYLE 组件图最底层的 C++ header-only 组件。它只承载两类在任何 runtime 建立前就必须
可用、没有 owning state、且需要跨模块保持一致的 primitive：

1. 程序不变量与诊断失败：always-on fatal、debug/instrumented assertion、有界 typed context，
   以及宿主在终止前观察失败事件的窄 seam；
2. 当前 translation unit 的编译环境事实和编译器合同，包括 OS/arch/compiler/build/sanitizer、
   inline/noinline、aliasing、branch、loop/layout hint 与 optimizer promise；
3. 编译期 cache layout recommendation 与三种 best-effort prefetch intent。

产品版本位于根级 `<hyle/version.h>`。它是 build/package 拥有的 checked-in 产品元数据，经
`hyle::core` 的 include contract 交付，但不是第三个 Core runtime 能力。symbol export/import
策略位于根级 `<hyle/visibility.h>`，由 build/package 拥有；它也不是 Core API。

这个边界刻意很小。Core 不因为“很多模块都可能用”而吸收容器、分配、时钟、CPU 探测、错误值、
日志或同步；这些能力需要各自完整表达 ownership、failure、concurrency 与复杂度。

## 公开面与归属

| 用户任务 | 公开头 | 当前形状 | 语义 owner |
|---|---|---|---|
| 违反程序不变量时立即终止并携带有界上下文 | `<hyle/core/assert.h>` | fatal/debug 两族 assertion、typed fields、failure observer | core |
| 查询当前 TU 的编译目标与 instrumentation | `<hyle/core/environment.h>` | 始终为 `0/1` 的 OS/arch/compiler/build/sanitizer/endianness 等事实与 generic feature probes | core |
| 在已测 native hot path 表达编译器合同 | `<hyle/core/compiler.h>` | inline/cold/alias/branch/loop/layout/lifetime/allocator-return/optimizer contracts | core |
| 设计 false-sharing/constructive layout | `<hyle/core/cache.h>` | constructive/destructive/max-destructive 常量 | core |
| 为已知访问模式给出 cache hint | `<hyle/core/cache.h>` | read、one-shot read、write intent prefetch | core |
| 识别编译当前 TU 的 HYLE headers | `<hyle/version.h>` | major/minor/patch、packed、string 编译期常量 | build/package |
| 为 compiled target 选择 export/import decoration | `<hyle/visibility.h>` | build-owned symbol visibility contract | build/package |

旧的 `<hyle/macros.h>`、`compiler_defs.h`、`cache_defs.h`、`<hyle/core/version.h>`、
含义容易与 runtime Platform 混淆的 `<hyle/platform.h>`、linked
`version()/version_string()` 与 version-only `hyle_core` binary 已退出当前产品面。项目尚未发布，
不保留转发头或兼容别名，避免让两个 owner 长期并存。

## 依赖与组件图

```text
<hyle/core/assert.h> ──────────────┐
                                  ├──► hyle::core (INTERFACE, no artifact)
<hyle/core/compiler.h> ───────────┤
<hyle/core/environment.h> ────────┤
<hyle/core/cache.h> ──────────────┤
<hyle/version.h> ─────────────────┘

hyle::core ──PUBLIC──► every component whose public headers consume Core
```

`core` 自身没有 HYLE 依赖。中央组件图中当前登记的公开头消费者——`container_intrusive`、`util_core`、
`error`、`obs`、`math`、`time`、`platform`、`mem`、`simd`、`container`、`util`、`sched`、`io`，以及仍为
`QUEUED` 的 `compute`/`graph` 旧资产——都直接声明 `PUBLIC core`。例如 `math` 的 unchecked scalar helpers 公开使用
`HYLE_FATAL_ASSERT`，因此 `math -> PUBLIC core` 是真实 source/package 合同，不是为了构造顺序保留的
伪依赖。

这条边不是为了链接一个 DLL，而是让 include directory、C++23/build contract、安装组件闭包和
公开源码依赖成为 target 的机器事实。中央组件图负责唯一声明，真实 target link 从同一记录
生成；配置合同还扫描公开头，拒绝“直接 include Core 却没有直接 `PUBLIC core`”以及没有实际
include 的冗余边。安装只投影 selected closure 的 owned headers，不再掩盖缺失依赖。

Core 只传播 build-owned C++/platform compile contract，不链接 `hyle_trace` 或 `hyle_testing`：它的
源码与头都不消费这些 gate。即使工程打开 profiling，链接 `hyle::core` 也不能凭空给消费者增加
Tracy runtime；测试专用 layout definition 同样不能污染这个公共地基。

## 设计裁决

### 编译器合同：标准 C++ 优先，wrapper 只补可移植缺口

Highway、Abseil、Folly 等成熟底层库都把 compiler-specific attribute/pragma 收在很小的 portability
层，而不会让业务模块到处复制 builtin。HYLE 采用相同原则，但不为已有自然标准形状制造别名：

- must-use 直接写 `[[nodiscard]]`；
- 禁止 copy/move 直接写 `= delete`；
- 已知对齐使用 `std::assume_aligned<N>`；
- token 拼接只保留在 `<hyle/core/detail/preprocessor.h>`，不是公共用户能力。

保留的 macro 都处理受支持编译器之间仍存在的语法或退化差异。hint 在未知 compiler 上可以安全
退化；`ASSUME`、`RESTRICT` 和 release `UNREACHABLE` 则是语义承诺，违约为 undefined behavior，
不能被当成 validation。

只有已经存在真实 consumer 的窄 attribute 才准入。`HYLE_COLD`、`HYLE_NO_TAIL_CALL` 用于 Core
failure sink，`HYLE_LIFETIME_BOUND` 标注同步借用，`HYLE_ALLOCATOR_RETURN(size_index)` 统一 Mem
对 fresh allocation 的声明。没有 consumer 的 `HOT`、blanket no-sanitize 或 nonnull matrix 不进入
Core；不为“宏更全”制造第二套注解语言。

### 编译环境事实：只描述当前 TU

`<hyle/core/environment.h>` 的 OS、architecture、compiler、pointer width、endianness、build mode、
sanitizer、static analyzer、exception 与 RTTI facts 始终展开为整数 `0` 或 `1`，可安全写成
`#if HYLE_OS_WINDOWS`。`HYLE_BITNESS` 为 `32/64`，未知表示为 `0`；generic `HYLE_HAS_*` probe 只用于
预处理条件。Clang 与 clang-cl 通过 primary compiler fact 和 `HYLE_COMPILER_CLANG_CL` secondary fact
区分，避免把 ABI mode 当作另一个 frontend。

这些值描述“正在编译这个 translation unit 的目标和选项”，不探测运行机器，也不证明某条 ISA
可执行。runtime OS/CPU/cache/NUMA 归 Platform，fat-binary ISA dispatch 归 SIMD。Core 不再为单个
consumer 发布 `HYLE_HAVE_SSSE3` 一类特例；Container 的 fixed-target probe 自己根据 current-TU
compiler target 选择实现。

### Fatal assertion 与 error 的边界

`HYLE_FATAL_ASSERT[_MSG/_CONTEXT]` 在 Debug 和 Release 都检查；它处理继续执行已不可信的 invariant
或 hardening 条件。`HYLE_DEBUG_ASSERT[_MSG/_CONTEXT]` 只在 Debug、sanitizer 或 static-analyzer TU
启用，普通 Release 中使用 C++ unevaluated expression 检查与启用态相同的 condition、message 和
1–8 `FatalField` 合同，但不求值、没有 runtime 指令。两族条件都只求值一次，message 与 fields
只在失败时构造。context 的 field array 在调用者完整表达式中形成，不由宏在调用者作用域声明固定
局部名，因此合法的用户标识符不会被宏捕获。

context 不是 printf 格式串：调用方显式提供最多 8 个 `FatalField`，值只接受 bool、最多 64-bit 的
integral/enum、floating、object pointer、C string 或 `std::string_view`。事件全部同步借用；Core
不分配，默认 formatter 对每段 text 最多写 256 bytes。`set_fatal_observer()` 原子交换一个
`noexcept` callback；callback 可把事件交给宿主已有的 crash/log pipeline，但返回后 Core 仍然
`abort()`。递归 fatal 跳过 observer，走默认诊断并终止，避免无限递归。

失败 sink 是 header-inline、ODR-safe、noinline/cold/no-tail-call，并使用 build-owned hidden
visibility，不能成为下游 shared module 的公开实现符号。observer slot 只承诺同一 linked image 内
一致，不冒充跨 DSO、跨语言或 signal-safe crash manager。失败诊断使用 stdio/`abort()`，不承诺
async-signal-safe、backtrace、minidump、logging policy 或恢复执行。OOM、输入、文件、设备和协议失败
必须用 error 模块的 `Status/Result`。

`HYLE_ASSUME` 在 Debug 借 fatal assertion 尽早暴露错误，在 Release 才向 optimizer 作承诺；
`HYLE_UNREACHABLE` 同理。assumption 表达式必须无副作用：Debug/GCC fallback 会求值，而其他
Release spelling 可以丢弃求值。这不改变“错误条件必须先由正常控制流验证”的规则。

### Cache layout 与 runtime topology 的边界

Core 常量是按编译 target family 选择的稳定 recommendation，会进入公开对象的 `alignof/sizeof`
和 ABI。runtime CPU/cache observation 属 platform；它可以帮助算法选择或分片，但不能替换一个
已经参与类型布局的编译期常量。

`kDestructiveInterferenceSize` 用于常态 false-sharing spacing；
`kConstructiveInterferenceSize` 用于确有证据的共同访问 packing；
`kMaxDestructiveInterferenceSize` 是稀疏、独立写控制字的跨 target 保守上界，不应成为密集容器
的默认 stride。当前编译期表是 Apple arm64 `128/128`、其他 AArch64 与 x86-64 `64/64`、未知
target `64/128`；独立的 max-destructive 常量始终为 128。这个区别保留 Apple 的真实 128-byte
coherency line，同时避免在 Windows/Linux AArch64 的普通密集布局上无依据地翻倍 footprint。
Core 不公开 L1/L2/L3 selector：prefetch 层级不是可移植合同，runtime cache
hierarchy 也不是 Core 的静态类型属性。

三种 prefetch 都是单地址、O(1)、best-effort hint。它们不读取值、不延长对象 lifetime、不建立
同步、不保证 residency；编译器或目标不支持时可以改变具体 hint 或成为 no-op。是否调用只能由
拥有访问模式的 consumer 决定。

### Version、visibility 与 runtime artifact

`<hyle/version.h>` 描述“这些 headers 声称自己是哪一版”，供 root CMake、package metadata、
编译期诊断和 release manifest 使用。HYLE 的 Core 没有真实 runtime state；为版本查询单独制造
library 只能确认 version library 自己，而不能证明进程内其他 HYLE artifact、inline 类型或编译
选项兼容。因此 Core 不提供 linked version query。

常态 constant-evaluated use 没有 runtime call；若用户显式取得某个 `inline constexpr` 变量的地址，
compiler 可以按 C++ ODR 规则物化 inline/weak storage。这不等于独立 Core runtime，也不承诺“目标
文件绝对没有任何 version symbol”。

同理，`HYLE_*_API`/hidden visibility 决定 compiled module 如何导出符号，必须随 static/shared、
platform loader 和 package 一起维护。把它放进 Core 会混淆 source semantics 与 binary delivery。

### Native FFI

Core 的 Native FFI 为 `N/A`。foreign runtime 无法通过一次动态调用复用调用方 TU 的 compiler
attribute、类型 layout 或 inline prefetch；版本常量也不足以构成真实外语任务。当前没有
`<hyle/c/core.h>`、Core C exports、managed Core smoke 或 Core ABI generation。

未来若出现真实 loader/bootstrap 需求，应按
[Native FFI 准入规则](native-ffi-policy.md)建立窄的 runtime 类型擦除面，由拥有实际 runtime
artifact 的模块或 package loader 持有；不恢复“C++ header parity”式 Core ABI。

## 三种 C++ 交付

| 交付 | `hyle::core` 的实际形状 | 用户需要知道的事 |
|---|---|---|
| 整体源码 | `add_subdirectory()` 产生 `INTERFACE` target | 链接 target 取得 include/build contract；没有 Core object library |
| 静态 SDK | installed imported `INTERFACE` target | 没有 `hyle_core.a/.lib`；能力编进 consumer TU |
| 共享 SDK | installed imported `INTERFACE` target | 没有 `hyle_core.so/.dylib/.dll`，也没有 Core runtime 需要 stage |

三种形状使用同一公开头。改变 compiler/cache/version header 后必须重编消费者；替换其他 shared
module 不能更新已经内联进 consumer 的 Core 代码。

## 明确负空间与重开条件

Core 不拥有 runtime CPU/cache/NUMA/OS discovery、allocator、container、clock、同步、日志、
recoverable error、backtrace/signal/minidump crash policy、ABI compatibility probe 或
foreign-language binding。它不承诺
具体 cache residency、精确机器 cache-line size、force-inline 必然发生，或对失效地址安全 prefetch。

只有以下事实足以重开这个边界：真实 consumer 需要一种在所有 runtime 之前可用的跨模块 primitive；
受支持 compiler/target 使现有合同错误；直接消费者出现 material codegen/correctness 问题；或成熟
peer 在相同语义下证明更好的复杂度/可移植形状。LOC、macro 数量、单条 hint benchmark 或亚纳秒
差异不是重开理由。
