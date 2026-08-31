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
| Abseil | `2c004366e983c5be8334ac1ea3d4420e8fbcbea7` | Core/Error 等基础能力 strong peer |

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

#### A2. Error 产品域（第一遍完成，尚未跨域冻结）

本节重新从真实失败旅程展开能力，不继承旧 `error-v4` 的“已关闭所以功能完整”结论，也不把 llama.cpp
面向 C 的 bool、null、整数返回值原样复制成 HYLE C++ API。直接推理证据来自锁定 ggml 的
`ggml_status`、graph/backend 返回、fatal/compute-abort callback，llama.cpp 的模型/backend 加载失败与
异常边界，以及 HYLE 当前 Platform/Mem/Obs/Sched/IO 消费。strong peers 限于本轮锁定的 Abseil
`Status/StatusOr/StatusBuilder` 和 Folly `Expected/result/exception_wrapper`；C++23 `std::expected` 是
语言合同，不计作第三个 peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `ERROR-A-01` | 叶子失败跨平台分类，上层按稳定语义恢复 | OK + 完整 canonical code 集；名称、predicate、factory；未知外来值 fail closed 为 `Unknown` | 无返回值时返回 `Status`；恢复只比较 typed code，不解析文本 | `COMPLETE`：gRPC/Abseil 17-code taxonomy、name/factory/predicate 已闭环 | ggml 的四态 status 证明最小需求，Abseil 与多 HYLE consumer 证明完整 taxonomy；基础 `MUST` |
| `ERROR-A-02` | operational failure 作为便宜值跨函数、模块和线程传递 | OK/code-only、message/detail、copy/move/swap/equality/destroy；共享不可变；OOM 仍保留 code；同对象并发写不承诺 | 小型 `noexcept` value；成功和 code-only 不分配，诊断走冷 backing；不使用 TLS last-error 或隐式日志 | `COMPLETE`：one-word tagged `Status`、immutable refcount backing、OOM degradation | Abseil tagged carrier + HYLE Platform/Mem/Sched handoff；基础与推理控制面 `MUST` |
| `ERROR-A-03` | 操作成功时还交付 owning value，或只交付成功/失败 | value、move-only value、`void`、error、default construction、observer、monadic composition；error arm 永不含 OK | `Result<T>`/`Result<void>` 与标准 expected surface；失败显式 `unexpected`，不增加第三个 empty 状态 | `COMPLETE`：`std::expected<T, ResultError>` 保持标准 surface 并补非 OK invariant | Abseil/Folly + HYLE 35-file historical adoption；基础 `MUST` |
| `ERROR-A-04` | registry/backend/map lookup 成功时返回一个保证非空、但不拥有对象的 borrow | mutable/const borrow、temporary rejection、deep-const、lifetime-bound、转换和 monadic propagation；nullable 与 range/text view 分开 | 与 `Result<T>` 一致的 `Result<T&>`-like borrow，或语义同等的具名 non-null borrow；nullable 仍用 pointer，range/text 用 span/view | `PARTIAL`：`reference_wrapper`/pointer/span/view 能表达 ownership，但没有直接 borrow carrier，且 wrapper 改变 observer/monadic 形状 | 当前 Abseil `StatusOr<T&>` 与 Folly `result<T&>` 都完整提供；通用 foundation lookup `GENERAL`，待 Container/Util/Platform 反查真实频度后定 API |
| `ERROR-A-05` | 中间层只在增加用户可理解语义时给失败加 context | empty/OK no-op、lvalue snapshot、unique rvalue chain、native detail 保真、OOM；正常链总复制 O(final bytes) | 显式 `WithContext`，输出保持 `outer: inner: leaf`；不自动塞函数名/source location/log | `COMPLETE`：扁平渐进 prefix storage；共享历史的 lvalue 分支成本已显式 | llama/ggml 的分层加载诊断 + Abseil builder practice；基础 `MUST` |
| `ERROR-A-06` | 失败文本需要包含路径、尺寸、设备号等运行时值 | UTF-8、嵌入 NUL、格式失败/OOM、临时对象 lifetime；文本 convenience 不绑定 logging/source policy | 普通 C++ 可先用标准/项目 formatter 产生 owning text；若 noexcept 边界反复需要，则是独立的 bounded/format-to-Status 冷 helper，而非有副作用 StatusBuilder | `PARTIAL/ROUTED`：Status 接受任意 `string_view`，但没有统一 no-throw 动态格式化路线 | llama/ggml formatted diagnostics + Abseil StatusBuilder；`GENERAL`，带入 Util/Obs/Core 反查，不能仅为 API 数量把 logger 塞入 Error |
| `ERROR-A-07` | OS/backend 失败同时保留 portable recovery code 与一个机器可读原生值 | POSIX errno、Win32、WinSock、HRESULT、Vulkan、CUDA 及后续 backend；立即捕获；跨 host 可命名；不能保存有 DSO lifetime 的 category pointer | `StatusCode` + 可选稳定 `{domain-id, int64 code}`；热 completion 可先保留紧凑 native outcome，在 ownership/API 冷边界转换 | `PARTIAL`：errno/Win32/HRESULT factory 和若干固定 backend domain 完整；缺 WinSock，固定 enum 对新增 backend 的登记策略尚未统一 | HYLE IOCP 已真实调用 `WSAGetLastError`，Compute 有 Vulkan/CUDA；推理多 backend `MUST`，旧“WinSock 无消费者”前提已失效 |
| `ERROR-A-08` | `std::error_code`、filesystem、第三方 SDK 或自定义 category 在拥有它的边界进入共享失败流 | generic/system category、owner custom category、operation-specific override、未知 category；避免伪 round-trip | generic/POSIX 与 Windows system 走稳定 adapter；custom category 由 owner 映 canonical code，只有已登记稳定 domain 才保留 numeric detail | `PARTIAL/ROUTED`：规则已文档化但无公共 cold adapter；ggml filesystem 与 HYLE 各模块仍各自转换 | ggml backend registry/OpenCL cache 使用 `std::error_code`，Abseil errno 与 Folly typed failure；基础集成 `GENERAL`，待跨域去重 |
| `ERROR-A-09` | 多层同步函数短路传播失败，成功路径不构造诊断 | Status/`Result<void>`、`Result<T>`、lvalue copy/rvalue move、表达式一次求值、外层 Status/Result、标准 monadic use | 两个窄传播宏 + 标准 expected operations；context/logging 仍显式；宏陷阱可见 | `COMPLETE`：`RETURN_IF_ERROR`、`ASSIGN_OR_RETURN` 与标准 monadic surface | Abseil/Folly propagation + 大量 HYLE consumer；易用性 `MUST` |
| `ERROR-A-10` | cleanup/批处理需要保留首错、显式忽略，或有界收集全部错误 | first error、discard、all-error ordering/cap/OOM/parallel merge；聚合 storage 由业务 owner 定义 | `Update`/`IgnoreError`；全部错误先使用 caller-owned collection，只有两个消费者共享同一 bounded contract 才提公共 helper | `COMPLETE + ROUTED`：first/discard 完整，all-error 可组合 | foundation teardown/parallel journey；`MUST`，但不准入无界 `MultiError` |
| `ERROR-A-11` | 最终边界把失败交给人读日志、测试或 UI | code/message/native 分片、caller-buffer size query、truncation、embedded NUL/UTF-8、可选 owning string | allocation-free `ToString` + opt-in owning `to_string`；structured observation 交给 Obs | `COMPLETE`：caller buffer 与 owning convenience 均有 | Abseil formatting + HYLE user/Obs boundary；基础 `MUST` |
| `ERROR-A-12` | async 工作跨线程完成、取消或超时，上层区分失败与正常 pending/stopped | transport/lifetime、cancel/deadline canonical outcome、pending/empty/closed/stopped protocol states | 已完成 operational failure 可用 Status；pending/stopped/closed 用 owning Sched/IO 状态机或 enum，不能给每次 poll 分配 Status | `COMPLETE + ROUTED`：Status 可跨线程且有 Cancelled/DeadlineExceeded；协议状态不由 Error 承载 | ggml compute-abort + Folly `stopped_result` + HYLE Sched/IO；能力 `MUST`，状态机 owner 留到因子阶段 |
| `ERROR-A-13` | parser/backend 需要多个 typed fields、cause 或诊断 provenance，边界仍需共享恢复语义 | schema、typed matching、cause traversal、serialization、source/backtrace、OOM/隐私/DSO lifetime | owner 可用 `std::expected<T, E>` 保真 typed error，再显式降为 canonical/native；trace/log/crash 信息走 Obs/Core 对应机制 | `ROUTED`：单 native detail + caller typed error route 能表达当前恢复；通用 Status 无 payload/cause | Abseil payload 与 Folly rich error 用于发现；没有真实共享 schema，不准入常驻 payload map，但必须在 Obs/Compute/IO 域反查是否确实可达 |
| `ERROR-A-14` | 第三方 callback、标准容器或 backend 在明确边界抛异常，HYLE noexcept/value API 必须给出确定结果 | `bad_alloc`、`system_error`、known exception、unknown exception；只在 owning boundary 捕获；不跨模块传播 RTTI/exception ABI | owner adapter 映成精确 Status 或保留 typed exception；只有重复且完全同语义的 mapping 才抽冷 helper | `PARTIAL`：Mem/Sched/Obs/Compute 等已有多处手写 catch/mapping，尚未证明一致或穷尽 | llama.cpp 大量 throw/catch + Folly exception wrapper；边界处理 `MUST`，但“所有异常进入 Status”明确拒绝 |
| `ERROR-A-15` | 不可恢复 invariant/fatal 与可恢复 operational failure 不混淆 | fatal context/observer 后必终止；Status 不恢复内存损坏或合同违反 | Core fatal route；Error 只处理可恢复值 | `ROUTED` 到 `CORE-A-01/02` | ggml assert/abort；基础 `MUST`，不复制 fatal handler 到 Error |
| `ERROR-A-16` | foreign runtime/RPC 需要稳定 wire/handle/error schema | create/call/error/cancel/destroy/unload、schema/version、serialization 与 ownership | 只有具名 consumer 才设计窄 runtime type erasure；不是机械导出全部 C++ Status/Result | `OUT_OF_SCOPE`（当前产品） | 当前 source/static/shared C++ 是唯一交付；无 named FFI/RPC consumer，明确拒绝伪 C ABI |

当前 public-surface 反向追踪覆盖 `<hyle/error/{status,result,format}.h>`、compiled `hyle_error` seam、
现行 architecture/user guide，以及 Platform/Mem/Obs/Sched/IO/Compute 的生产调用。第一遍得到四个跨域候选
触发，而不是 Error 施工票：

1. WinSock 已有真实 IO consumer，旧 negative-space 需要在 IO 域清点后确定 domain/factory 与热 completion
   转换边界；
2. 直接 borrowed result 已由两个当前 strong peer 提供，需由 Container/Util/Platform 的非空 lookup
   journey 判断 `reference_wrapper` 路线是否仍然自然；
3. 动态格式化、rich typed error 和 exception adapter 的最终 owner 只能与 Core/Util/Obs/IO/Compute 一起
   去重，不能把 Abseil StatusBuilder 的 logging/payload 副作用整体复制进来；
4. 固定 native-domain enum 的新增登记和 binary/version 成本必须在 Build/package 横向合同中裁决，不能让
   每个 backend 自造第二套错误系统。

这表示 Error 产品域的阶段 A 第一遍已经完成，但不重开当前 `CLOSED` Error，也不授权实现上述候选。
最终 `MUST`、owner、组件形状和依赖只能在 13 域完成后的跨域账本中冻结。

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
