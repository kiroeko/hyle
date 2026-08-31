# 全库能力展开与组件因子化 v1

> 记录性质：`ACTIVE PROJECT ARCHITECTURE REPLAY`。它不改变任何模块的机械状态，也不授权修改产品
> 代码；当前状态仍只以 [`module-status.json`](../../module-status.json) 为准。

## 目标与基线

- 基线 commit：`b59db00d8d7662adadb5af8b7af03eb631aa91f0`；
- 基线 tree：`e7b3a5f2f6bc90b194821c1777542c3c47a27c7b`；
- 目标：先完整统计 13 个基础产品域的应有功能和自然下游形状，再跨域确定 owner、不可再有意义拆分的
  组件因子、依赖图和实施拓扑；
- 冻结：Compute、Graph 保持 `QUEUED`；其他域在能力统计期间不修改产品实现；Mem v2 已有 FIX 集只作
  输入，在全局因子图完成前暂停候选实现；
- 非目标：本轮不跑 benchmark、不签性能结论、不按当前 CMake 图修 API、不按代码行补功能，也不重开
  `CLOSED` 模块。发现真实产品缺口时只登记 `PRODUCT_TRIGGER`，待全局图完成后按拓扑恢复。

## 锁定参考输入

| reference | identity | role |
|---|---|---|
| llama.cpp | `e4b9af007beae34abba094cefd187658e89bbac8` | 真实推理 consumer journey |
| llama.cpp vendored ggml tree | `4660c26f69a7c1ab6c8309934214bde5622483ea` | 与 consumer 同版本的基础实现 |
| standalone ggml | `d4716378882593333721eb33f153144b6885caf2` | 最新独立基础能力差异 |
| Folly | `23ebefdf96c116700d3e134f04d95a2b60a509de` | 成熟基础库 peer |
| oneTBB | `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` | 成熟并行/基础库 peer |
| Boost | `78505eab9b203e3e26e83893d0685f160133fa0f` | 成熟 C++ 组件 peer |

其他域可增加至多两个直接 strong peers，但必须锁定完整身份并写明发现职责。参考更新后先更新身份和
差异，不把“最新版”当功能准入理由。

## 阶段门

### A. 功能超集与自然形状

按 Core、Error、Container-intrusive、Util-core、Time、Platform、Mem、Obs、SIMD + Math、Container、
Util-runtime、Sched、IO 的顺序逐域填写：

| domain | capability family | exact source / journey | semantic variants and lifecycle | natural C++ downstream shape | current coverage | admit/reject reason |
|---|---|---|---|---|---|---|

每域必须覆盖五种输入：ggml、llama.cpp、成熟库、定义性工程能力、HYLE public trace。当前依赖只记录，
不能据此拒绝能力。13 域全部填完、跨域查漏且没有未解释的 `MUST` 后，才冻结 A。

SIMD 在本轮有绑定例外：无论内部后端和 ISA 怎样组织，都保持一个不可拆分的公开产品组件；Math 单独
裁决边界。不能用组件因子化把 SIMD 重新拆成 Highway、xsimd 或 HYLE 多套用户产品。

#### A1. Core 产品域（第一遍完成，尚未跨域冻结）

本节的 `coverage` 只描述基线实现对该能力行的覆盖，不继承旧 `core-v2` 的模块关闭结论，也不在此时
决定最终 owner 或依赖。直接推理证据来自锁定 llama.cpp 随仓 ggml 的 `ggml/include/ggml.h`、
`ggml/src/ggml.c`、CPU kernels 与 `src/llama.cpp`；strong peers 按能力行限于 Highway
`hwy/base.h`/`detect_compiler_arch.h`/`cache_control.h`、Abseil `absl/base/{attributes,optimization,
prefetch,macros}.h`、Folly `folly/{CPortability,lang/SafeAssert,lang/Hint}.h` 和 oneTBB
`detail/{_config,_assert}.h` 中至多两个。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `CORE-A-01` | 低层代码发现不可恢复的程序不变量后立即终止 | 条件只求值一次；Debug/Release 均检查；表达式、文件、行和可选静态消息在终止前有效；成功路径不分配、不锁 | 一个 always-on invariant check；recoverable 输入/OOM/系统失败不走此路 | `COMPLETE`：`HYLE_FATAL_ASSERT(_MSG)` 覆盖最小旅程 | ggml/llama 大量 `GGML_ASSERT/ABORT`，Highway always-on assert；定义性 `MUST` |
| `CORE-A-02` | fatal 时携带实际对象/尺寸等有限上下文，并让宿主 crash/telemetry 层观察后仍终止 | bounded typed/formatted context；可选 observer 必须线程安全且不能恢复；默认 stderr；backtrace 与 signal-safe 另行裁决 | 成功路径仍为一条冷分支；failure event 与 termination 分离，observer 返回后必然 abort | `PARTIAL`：只能借用一个 NUL 字符串；没有值上下文或宿主 observer | ggml 有 formatted abort + callback，Highway有 formatted abort + thread-safe handler；普通库嵌入诊断为 `MUST`，但 backtrace/signal policy 不自动进入 Core |
| `CORE-A-03` | 昂贵的不变量只在 debug/instrumented build 检查；安全边界可选择 always-on hardening | debug-only、instrumented、always-on 三种编译语义；被禁用表达式不执行副作用 | 按语义命名 debug assertion 与 invariant/hardening assertion，不按“fast/slow”命名 | `PARTIAL`：标准 `assert` 可完成 debug-only；HYLE 只有 always-on 路线，instrumented/hardening 模式未统一 | Abseil assert/hardening、Highway assert/dassert；通用 `MUST`，可由标准路线承载但 HYLE 内部语义需统一 |
| `CORE-A-04` | 已测 hot path 向 MSVC/GCC/Clang 表达 optimizer 和布局合同 | inline/noinline、restrict、flatten/unroll、expression branch prior、assume/unreachable、no-unique-address；退化和 UB 边界明确 | 标准属性/函数优先；只有跨 compiler 无自然标准形状时才使用窄 macro | `COMPLETE`（现有档位）：`compiler.h` 的 11 个入口；未知 compiler 的 hint 可退化 | ggml `RESTRICT/UNREACHABLE`、Highway/Abseil 同类 portability；定义性 `MUST` |
| `CORE-A-05` | Mem、容器、错误值和低层 runtime 表达编译器可验证的 ownership/codegen 合同 | cold/hot、no-tail-call、lifetime-bound、allocator/nonnull、sanitizer detection/最窄 suppression；每项可独立退化 | 标准属性优先；compiler wrapper 只暴露已经有真实 consumer 的语义，不公开“万能 attribute”或 blanket no-sanitize | `PARTIAL`：fatal 内部私有 cold；Mem 私造 allocator-return macro，Sched 私探 sanitizer；无统一 lifetime-bound/sanitizer contract | Abseil attributes + Folly portability；HYLE 已有重复/局部 consumer，作为 `MUST` 能力族准入，具体条目待相邻域反查 |
| `CORE-A-06` | header/source 在编译期识别受支持环境并选择可编译实现 | OS、arch、compiler、pointer width、endianness、build/sanitizer、exception/RTTI 与 generic feature probe；只描述当前 TU，不冒充 runtime CPU | 所有支持维度为一致的 0/1 compile-time facts；未知值显式可判；标准 `<bit>` 等能表达的直接用标准 | `PARTIAL`：有 OS/arch/compiler；bitness 依赖未自行 include 的 `INTPTR_MAX`；缺 endianness/sanitizer 等统一事实 | Highway detect + Folly portability；ggml 多 backend/ISA 编译分支；基础可移植构建 `MUST` |
| `CORE-A-07` | kernel 判断“当前 TU 编译了什么 ISA”，运行时再由 SIMD 唯一裁决可执行 target | compile-target 与 runtime-supported 严格分离；fixed/fat target 均适用；不能把一条 ISA 特例当通用平台事实 | 用户从一个不可拆分 SIMD 产品取得能力；底层只提供 generic compile-target facts，不能公开 `HAVE_SSSE3` 式逐消费者补丁 | `PARTIAL`：根 `platform.h` 只特设 `HYLE_HAVE_SSSE3`，形状错误；SIMD 另有完整 vendor route | ggml arch TUs + Highway target model；推理 kernel `MUST`，最终边界留给 SIMD/Build 因子阶段 |
| `CORE-A-08` | 并发控制字隔离 false sharing，或将共同访问的小状态放在同一 line | constructive/destructive recommendation 与跨 target 保守上界；compile-time layout 与 runtime topology 分离 | `constexpr size_t`，直接用于 `alignas`；值进入 layout，用户必须重编 | `COMPLETE`：三项 interference 常量，当前 Container/Mem/Sched 有真实消费 | ggml CPU scratch/cache-line padding + Abseil cache-line contract；定义性 `MUST` |
| `CORE-A-09` | 已知未来访问的 kernel 发出 cache hint | temporal read、one-shot/NTA read、write intent；O(1) no-op fallback；不建立 lifetime/sync/residency | 三个按访问意图命名的 inline 函数；不暴露 L1/L2/L3 selector | `COMPLETE`：三项 `prefetch_*` 与 Abseil 同形；SIMD/Container 有真实消费 | llama.cpp/ggml quant kernels + Abseil prefetch；推理 hot path `MUST` |
| `CORE-A-10` | cache flush、streaming store/fence、spin pause 等低层机器动作 | cache coherence/persistence、memory ordering、spin progress 各自语义不同 | 不塞入通用 prefetch API；分别带入 Platform/Mem/Util/SIMD 域查 owner | `OUT_OF_SCOPE`（Core 基线） | Highway cache-control 用于发现，但不因同在一个 peer header 就准入 Core；已转交后续域清单 |
| `CORE-A-11` | consumer、package 与诊断识别所编译 headers/实际 artifact | header semver、package version/commit/build manifest；runtime query 只有真实 compiled-artifact 任务才成立 | 编译期 constants + package manifest；不能用单一 runtime number 假装证明 inline C++ ABI 兼容 | `PARTIAL`：header semver 完整；无 runtime commit/build identity，visibility 由 build contract 承载 | ggml/llama 有 runtime version/commit；HYLE source-primary 无当前 runtime task，保留为 Build/package `GENERAL` 而非伪 Core DLL |
| `CORE-A-12` | compiled 模块在 source/static/shared 下正确导入导出，普通 C++ API 可标废弃/必须使用 | export/import/hidden 与 component build mode；deprecated/nodiscard/fallthrough/source-location 等语言属性 | build 生成/模块拥有 `*_API`；标准 C++ 属性直接使用，不为数量造包装 | `ROUTED`：根 visibility contract + C++23 标准库；Core 无 binary | ggml/llama visibility 宏 + Abseil 标准优先；交付 `MUST`，但不是 Core runtime 功能 |
| `CORE-A-13` | byteswap/endian/bit ops、source location、assume-aligned、compiler fence 等基础表达 | 标准 C++23 语义；不复制旧编译器兼容层 | `std::byteswap/endian/countl_zero/source_location/assume_aligned` 与 `std::atomic_signal_fence` | `ROUTED`：标准库可完成；HYLE 个别 raw builtin 留待所属域实现审计 | Highway/Folly 用于旧工具链的补位不构成 HYLE 新 API；当前 C++23 合同下 `ROUTED` |

当前 public-surface 反向追踪已覆盖 `<hyle/core/{assert,compiler,cache}.h>`、`<hyle/version.h>`、
`<hyle/visibility.h>` 和 `<hyle/platform.h>`。第一遍得到三个跨域候选触发而不是施工票：

1. fatal 的 bounded context/observer 路线需要 Error/Obs 一起反查，不能由 Core 自行吸收 logging/crash policy；
2. compile-environment 与 compiler-annotation 家族不完整，且已有 include-order/局部重复事实；
3. `HYLE_HAVE_SSSE3` 是单一消费者特例，必须在 SIMD 不可拆分约束下统一回审，不能继续扩成一串全局
   `HAVE_*`。

这表示 Core 产品域的阶段 A 第一遍已经完成，但不表示 Core 模块需要立刻重开，也不表示上述能力最终
都属于一个 Core target。只有 13 域完成并做跨域反查以后才产生 `PRODUCT_TRIGGER` 和 owner。

### B. 跨域归属与质因数分解

只使用已经冻结的 A 输入，建立全局账本：

| capability family | semantic owner | owning product domain | irreducible component | independent consumers | cost/delivery closure | duplicate/conflict resolution |
|---|---|---|---|---|---|---|

`container_intrusive` 作为正例校准：属于 Container 产品域，但因 Time 等消费者只需要零分配侵入式链接，
可独立闭环并避免完整 Container 的 Mem/Platform 闭包，所以是独立因子。没有独立消费、依赖缩减或成本
隔离收益的文件/函数不得制造 target。

### C. 依赖图与施工顺序

为每个因子记录 public semantic/type、private mechanism、build-only vendor、consumer 与唯一 decision
authority 五类边。消除 owner 错位、机制类型泄漏和重复决策后，输出无环中央组件图、旧到新迁移表及
拓扑施工顺序。只有此后才为首个组件建立 `SCOPED/AUDITED` 的有限 FIX 集并修改产品代码。

## 停止与裁决

- A 阶段每个能力行有来源、完整旅程、自然形状、覆盖状态和准入/拒绝理由后停止，不求 peer API 并集；
- A 未完成时，不讨论最终依赖；B 未完成时，不决定实现顺序；C 未完成时，不开始模块重构；
- 发现 correctness/security/build blocker 时建立独立有限任务，不借机提前改产品架构；
- 本记录完成时应给出 A/B/C 三项明确产物；缺少 material 输入则记录具名 blocker，不以“继续研究”结案。
