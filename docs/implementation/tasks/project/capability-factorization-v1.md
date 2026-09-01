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
| `ERROR-A-06` | 失败文本需要包含路径、尺寸、设备号等运行时值 | UTF-8、嵌入 NUL、格式失败/OOM、临时对象 lifetime；文本 convenience 不绑定 logging/source policy | 普通 C++ 可先用标准/项目 formatter 产生 owning text；若 noexcept 边界反复需要，则是独立的 bounded/format-to-Status 冷 helper，而非有副作用 StatusBuilder | `PARTIAL`：Status 接受任意 `string_view`，但没有统一 no-throw 动态格式化路线 | llama/ggml formatted diagnostics + Abseil StatusBuilder；`GENERAL`，带入 Util/Obs/Core 反查，不能仅为 API 数量把 logger 塞入 Error |
| `ERROR-A-07` | OS/backend 失败同时保留 portable recovery code 与一个机器可读原生值 | POSIX errno、Win32、WinSock、HRESULT、Vulkan、CUDA 及后续 backend；立即捕获；跨 host 可命名；不能保存有 DSO lifetime 的 category pointer | `StatusCode` + 可选稳定 `{domain-id, int64 code}`；热 completion 可先保留紧凑 native outcome，在 ownership/API 冷边界转换 | `PARTIAL`：errno/Win32/HRESULT factory 和若干固定 backend domain 完整；缺 WinSock，固定 enum 对新增 backend 的登记策略尚未统一 | HYLE IOCP 已真实调用 `WSAGetLastError`，Compute 有 Vulkan/CUDA；推理多 backend `MUST`，旧“WinSock 无消费者”前提已失效 |
| `ERROR-A-08` | `std::error_code`、filesystem、第三方 SDK 或自定义 category 在拥有它的边界进入共享失败流 | generic/system category、owner custom category、operation-specific override、未知 category；避免伪 round-trip | generic/POSIX 与 Windows system 走稳定 adapter；custom category 由 owner 映 canonical code，只有已登记稳定 domain 才保留 numeric detail | `PARTIAL`：规则已文档化但无公共 cold adapter；ggml filesystem 与 HYLE 各模块仍各自转换 | ggml backend registry/OpenCL cache 使用 `std::error_code`，Abseil errno 与 Folly typed failure；基础集成 `GENERAL`，待跨域去重 |
| `ERROR-A-09` | 多层同步函数短路传播失败，成功路径不构造诊断 | Status/`Result<void>`、`Result<T>`、lvalue copy/rvalue move、表达式一次求值、外层 Status/Result、标准 monadic use | 两个窄传播宏 + 标准 expected operations；context/logging 仍显式；宏陷阱可见 | `COMPLETE`：`RETURN_IF_ERROR`、`ASSIGN_OR_RETURN` 与标准 monadic surface | Abseil/Folly propagation + 大量 HYLE consumer；易用性 `MUST` |
| `ERROR-A-10` | cleanup/批处理需要保留首错、显式忽略，或有界收集全部错误 | first error、discard、all-error ordering/cap/OOM/parallel merge；聚合 storage 由业务 owner 定义 | `Update`/`IgnoreError`；全部错误先使用 caller-owned collection，只有两个消费者共享同一 bounded contract 才提公共 helper | `COMPLETE`：first/discard 直接提供，all-error 由 caller-owned collection 完整组合 | foundation teardown/parallel journey；`MUST`，但不准入无界 `MultiError` |
| `ERROR-A-11` | 最终边界把失败交给人读日志、测试或 UI | code/message/native 分片、caller-buffer size query、truncation、embedded NUL/UTF-8、可选 owning string | allocation-free `ToString` + opt-in owning `to_string`；structured observation 交给 Obs | `COMPLETE`：caller buffer 与 owning convenience 均有 | Abseil formatting + HYLE user/Obs boundary；基础 `MUST` |
| `ERROR-A-12` | async 工作跨线程完成、取消或超时，上层区分失败与正常 pending/stopped | transport/lifetime、cancel/deadline canonical outcome、pending/empty/closed/stopped protocol states | 已完成 operational failure 可用 Status；pending/stopped/closed 用 owning Sched/IO 状态机或 enum，不能给每次 poll 分配 Status | `ROUTED`：Status 可跨线程且有 Cancelled/DeadlineExceeded；协议状态由 Sched/IO 承载 | ggml compute-abort + Folly `stopped_result` + HYLE Sched/IO；能力 `MUST`，状态机 owner 留到因子阶段 |
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

#### A3. Container-intrusive 产品域（第一遍完成，尚未跨域冻结）

本节不把当前 `intrusive_list.h` 等同于整个问题域。直接推理输入包括 HYLE Time 的 timer membership 与
批量 ready handoff、Sched 的 timeout-unlink/wake-chain/remote-ready 形状、IOCP 的 in-flight/free-list
复用，以及 llama.cpp tokenizer 的 index-linked symbol chain 和 ggml CANN 的 owning LRU 反例。strong
peers 限于 Folly 的 uncounted/counted list、atomic intrusive list 与 intrusive heap，以及 oneTBB runtime
内部 counted member/base intrusive list；Folly 公开 list alias 的底层 Boost.Intrusive 只算同一条 peer
证据，不重复计票。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `INTR-A-01` | caller 已拥有稳定地址 node，希望零分配建立 membership | list 不拥有/分配/释放 node；node/list 任一先销毁的规则明确；linked 期间地址稳定；同一对象可用多个 hook 加入多条结构 | hook 是显式 member/base；container 以 `T` 和 hook identity 编译期绑定；普通 owning sequence 不走 intrusive | `COMPLETE`（当前 member-list 路线）：两指针 hook、sentinel-only list、multiple membership | HYLE TimerEntry + Folly/oneTBB runtime；基础 `MUST` |
| `INTR-A-02` | node 可自动撤销 membership，或 owner state machine 必须显式退休 | auto-unlink 与 safe-link；copy/move 不偷转 membership；析构、clear、list-first/node-first、外部同步完整 | 两个按生命周期命名的 hook/list family，而不是 fast/safe 性能档 | `COMPLETE`：auto/safe 两套同算法 surface，safe Debug 诊断 | Folly/Boost 两种 link mode + HYLE Time safe owner；基础 `MUST` |
| `INTR-A-03` | FIFO/LRU/timer/waiter 在已知节点上双向遍历并 O(1) 摘链 | empty/front/back、push/pop、insert/erase/remove、iterator/reverse iterator、iterator invalidation、const/ranges；空结果为 pointer/null | 标准 bidirectional range-like list；已证明前置条件走窄操作，nullable/already-linked 可走 `try_*` | `COMPLETE`：C++23 bidirectional range、基本操作与窄 recoverable try 入口完整 | Folly/oneTBB + Time；基础 `MUST` |
| `INTR-A-04` | timer bucket、ready batch 或 LRU 在不搬 node 时转移一个节点、整链或连续 range | same/cross-list、single/all/range splice、append、swap/list move、iterator stability；range transfer 不扫描 node | `splice`/`append`/`swap` 与 `iterator_to`；所有结构转移保持 O(1)，不暴露 benchmark-only batch API | `COMPLETE`：single/all/range splice、append、move/swap 与 O(1) iterator recovery | HYLE TimerQueue/CommonTimeoutQueue 是真实 consumer；复杂度档 `MUST` |
| `INTR-A-05` | owner 批量接管所有 node，逐项销毁、重链或唤醒 | list 先逻辑 detach；callback 前当前 hook 已 unlinked；callback exception/noexcept 与残余 ownership 明确 | `consume_all(noexcept fn)` 返回数量；只遍历不改结构用 `for_each`；allocator/disposer 留给 owner | `COMPLETE`：bulk detach/consume 与 visit 已覆盖 | Time ready handoff、Folly atomic sweep 形状；基础 `MUST` |
| `INTR-A-06` | 同步对象/LRU 需要频繁 O(1) `size()`，但普通 timer bucket 不该为计数付费 | counted 与 uncounted 是类型选择；single mutation O(1) 更新；range splice 必须扫描、接受 caller count 或限制为 whole-list，不能伪称 O(1) | 独立 counted safe-list/policy；默认 uncounted 仍不是 `sized_range` | `MISSING`：Time 在 owner 跨 bucket 计数；公共 list 没有可选 counted variant | Folly `CountedIntrusiveList`、oneTBB runtime list 与 Folly TimedMutex/sem consumer；通用 `GENERAL`，模块化准入而非给默认 list 加 word |
| `INTR-A-07` | 用户把 intrusive list 当完整非 owning sequence 做 reverse、merge、sort、unique 或 predicate removal | comparator/predicate exception、stability、iterator validity、partial progress、O(n)/O(n log n)；不接管 storage | STL-like 算法只在其语义不能由 ranges 组合自然完成时提供；range erase/dispose 不复制同复杂度 convenience | `PARTIAL`：循环 erase/for_each/consume 可组合 removal；没有 reverse/merge/sort/unique | Boost/Folly full-list surface 与通用 list journey；`GENERAL`，待 Container/Time consumer 反查，不因 peer API 数量一次性全抄 |
| `INTR-A-08` | 既有对象不能公开 member hook，或用继承/tag/traits 表达多种 membership/pointer model | direct member、base hook、tag、多 hook、private access、custom/value traits、fancy/offset pointer；object recovery 必须对 ABI fail closed | 常用 member/base 形成短 shape；高级 traits/pointer model 可直接使用成熟依赖，不造万能 policy matrix | `PARTIAL`：direct member、private friend、多 hook 与支持编译器 ABI gate 完整；无 base/value traits/fancy pointer | oneTBB 同时有 base/member，Boost/Folly 提供成熟 member/policy route；`GENERAL`，需在 Build/Platform/shared-memory 域裁决是否 HYLE 自有 |
| `INTR-A-09` | owner 保持业务 identity 时用新 node 原位替换旧 node，或交换两个 auto membership | same/cross-list、one/both unlinked、safe owner mirror 同步、iterator invalidation；不移动 payload | auto list 可 typed `swap_nodes`；safe list 若准入应是由 owner 明确调用的 list-level replace，而非裸 hook 手术 | `PARTIAL`：auto typed swap 完整，safe replace 缺失 | 通用 intrusive state-machine journey；`GENERAL`，只有 Container/Time/Sched 出现真实 safe replacement 才升 `MUST` |
| `INTR-A-10` | free-list/cancellation/wake chain 只需一指针遍历，或要 `next + prev_next` 的 O(1) 自摘链 | forward slist、hlist、LIFO/FIFO、tail、splice-after、safe/auto、link storage 复用；footprint 与双向 list 不同 | 单链/hlist 必须是独立类型合同，不能把双链 hook当“通用 next”；owner 可保留更窄特化 | `PARTIAL`：Sched cancel/wake、Mem pool 与 IO free-list 有多套私有形状；无公共 slist/hlist | Boost slist、Folly atomic link 与多个 HYLE consumer；能力 `MUST` 进入跨域去重，但不预判公共化 |
| `INTR-A-11` | 多线程 producer 无分配发布已有 node，单 consumer 原子摘走一个 batch 后按 FIFO/LIFO sweep | MPSC/MPMC 范围、memory order、single-consumer、ABA/reinsert、destruction-empty、callback/reclamation；不承诺任意并发 erase | 一个原子 head + one-link hook 的 publish/detach-batch primitive；ordering 在 API 中点名；reclamation 由 node owner | `MISSING`：当前 Sched 选择其他 queue，未提供通用原子 intrusive batch list | Folly FiberManager/AtomicIntrusiveLinkedList 是成熟真实消费；推理 runtime `GENERAL`，待 Sched/Obs/IO 工作量确认 |
| `INTR-A-12` | timer/executor 需要按 key 取 top，并在既有 node 上 erase/update/meld 而不另分配 handle | min/max、comparator、push/top/pop、arbitrary erase/update、meld、stable ties、node membership；复杂度按 heap 算法明确 | intrusive heap 是独立结构，不伪装 list policy；owner 决定 deadline/task 语义 | `MISSING`：Time 有专用 timer containers，Container 有 owning heaps route；无 intrusive heap | Folly timekeeper/epoll/executor 的 `IntrusiveHeap` 为真实 peer 消费；`GENERAL`，带入 Time/Sched/Container 比较现有复杂度 |
| `INTR-A-13` | 同一稳定对象需要 ordered/unordered intrusive index、multi-index 或共享内存 offset links | key mutation、unique/multi、rehash、iterator/order、hook count、pointer model、owner lifetime | 优先直接使用成熟 Boost.Intrusive/领域 index；只有两个 HYLE consumer 共享窄语义才形成产品能力 | `ROUTED`：当前用户可选择成熟依赖，HYLE 没有对应 surface | llama/ggml 没有证明通用需求；peer breadth 单独不足，明确不因“Boost 有”复制树/哈希全集 |
| `INTR-A-14` | concurrent owner 在锁内重链，锁外 wake/destroy，或需要 lock-free reclamation | thread-compatible vs thread-safe、callback reentry、node destruction race、ABA/hazard/epoch、owner validation | 普通 list 只承诺 external synchronization；并发 publication/reclamation 分成具名结构，不给每个 hook 增 owner/atomic metadata | `ROUTED`：当前 list 的 external-sync 边界明确；并发 waiter/IO 状态机由各 owner 承载 | HYLE Sched/IO + Folly atomic peer；并发语义 `MUST`，但 blanket thread-safe list 明确拒绝 |
| `INTR-A-15` | source/static/shared consumer 独立选择零 runtime 链接能力 | header identity、component selection、no binary journey、pointer-to-member compiler gate；模板对象不跨 Native FFI | header-only C++ target；FFI 由拥有 opaque runtime object 的上层 adapter | `COMPLETE`：独立 `container_intrusive` INTERFACE target；Native FFI `N/A` | 当前 package/source journeys；交付 `MUST` |

public-surface 反向追踪覆盖 `<hyle/container/intrusive_list.h>`、Time 的 `TimerEntry/TimerList`、现行
architecture/user guide 和独立 component target。llama.cpp 的 tokenizer `prev/next` 是可 relocation
vector 中的 index graph，CANN LRU 又拥有并删除对象；两者都不是把 HYLE raw-hook list 强塞进推理代码的
理由。第一遍得到五个跨域候选，而不是立即扩张这个小组件：

1. counted list 是明确可模块化的常用缺口，不能再用“默认 list 不应付计数税”把整个能力删除；
2. HYLE 已有多套私有 slist/hlist，必须在 Mem/Sched/IO 清点后区分可复用质因数与合法状态机特化；
3. atomic producer-batch list 与 intrusive heap 有成熟 runtime 实例，但是否属于本组件只能由 Sched/Time/
   Container 的生命周期和复杂度共同裁决；
4. base/value-traits、高级 pointer model 与完整 list algorithms 可以走成熟依赖，不能仅因 HYLE 自实现较窄
   就宣称功能不存在，也不能未经真实旅程复制 Boost 全部 policy；
5. `container_intrusive` 仍是“质因数”正例的候选，不是先验结论；如果最终出现多个互不需要的语义闭环，
   第二阶段仍应因真实消费拆分，但现有双向 list 自身已经能独立交付。

这表示 Container-intrusive 产品域的阶段 A 第一遍已经完成；当前 `CLOSED` 组件不重开，新增候选只有在
13 域跨查后才形成有限 `PRODUCT_TRIGGER`。

#### A4. Util-core 产品域（第一遍完成，尚未跨域冻结）

本节把 `util_core` 当作低层 utility 产品域，而不是把当前五个公开 header 反推成完整定义。直接推理
输入包括 llama.cpp sampler 的可复制/事务式 RNG 状态、标准 uniform/discrete distribution 与 ggml-opt
shuffle，ggml 的大量同步 callback + context 边界，以及 HYLE 当前 Mem/Sched/Graph 的短锁、Platform
目录 visitor、Sched/IO/Graph 的多套 callback record 和 Container 私有 adaptive lock。strong peers 按
能力行限于 Folly 的 Random/ScopeGuard/FunctionRef/SpinLock，Abseil 的 Cleanup/FunctionRef/nullability，
Boost 的 intrusive pointer counter policy，以及 oneTBB 的 spin mutex/rw mutex；C++23 标准库路线不计作
额外 peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `UTILC-A-01` | 推理采样、训练 shuffle、仿真或测试需要快速且可复现的统计 bit stream | 显式 seed、copy/move、完整 state checkpoint/restore、非法全零 state、固定算法 identity；常态不分配、不锁 | 一个满足 `uniform_random_bit_generator` 的具体 engine；显式对象承载可复现性，不藏在线程全局状态 | `COMPLETE`：xoshiro256++、SplitMix64 seed、`RngState`、copy/state/from-state 与 URBG surface 完整 | llama sampler 复制主/事务 RNG，ggml-opt 保存 engine；Folly xoshiro；推理与通用计算 `MUST` |
| `UTILC-A-02` | 用户从 bit stream 取得整数/浮点/概率值，或无偏选择/打乱区间 | u32/u64 全域与有界域、闭/半开边界、bool、float/double、rejection、random-access shuffle；极端输入不溢出 | engine 提供少量高频 typed draw；通用 distribution 继续组合标准 `<random>`，不复制一套 HYLE 名字 | `COMPLETE`：Lemire u32/u64、unit/range float、bool、callback/range shuffle；标准 distribution 可直接消费 `Rng` | llama uniform/discrete + ggml `std::shuffle`，Folly Random；普通旅程 `MUST` |
| `UTILC-A-03` | token categorical、Bernoulli/Gaussian/Zipf 等领域分布，或要求与既有 engine 逐 draw 对齐 | 权重归一化、数值稳定性、空/退化分布、算法版本、CPU/backend draw commit/rollback；结果可复现合同必须点名 engine 与 distribution | 标准分布能满足时直接接 `Rng&`；token top-k/top-p/categorical 和 backend transactional policy 属推理/Math owner | `ROUTED`：`std::*_distribution` 可组合；HYLE 没有把 sampler policy 塞进 RNG | llama sampler 对 double uniform、discrete 和 backend committed draws 有真实语义；这是 `MUST` journey，但不是 Util-core API 并集 |
| `UTILC-A-04` | worker/thread 获得独立统计流，或普通代码取得无需复现的 convenience stream | 显式 jump/long-jump 分区、TLS 首次 seed 失败、跨 DSO/线程 identity、稳态无锁；“大概率不同”不能冒充数学不重叠 | 可复现并行用 owner 持有的 engine + 明确分流；best-effort convenience 单独命名并暴露限制 | `COMPLETE`：2^128/2^192 jump 与 best-effort `thread_rng()` 两条路线分离 | Folly ThreadLocalPRNG + HYLE Sched 潜在 worker journey；通用 `MUST` |
| `UTILC-A-05` | 安全 token/key/nonce、ASLR salt 或可信 seed material 需要操作系统熵 | blocking/failure、fork、sandbox、平台 provider、buffer fill、zeroization；不能从统计 PRNG 或 `random_device` 推断安全 | Platform/安全 owner 的 `secure_random(span<byte>) -> Status`，再显式 seed 统计 engine；不自写 crypto PRNG | `OUT_OF_SCOPE`：`thread_rng` 明确非安全，当前无 secure bytes API | Folly secureRandom 用于发现，HYLE IO/security 边界决定准入；能力本身 `GENERAL`，最终 owner 交给 Platform/IO，不污染 Util-core |
| `UTILC-A-06` | 资源取得后有多早退，需要一次 lexical rollback、cleanup 或显式 commit | once、dismiss/invoke、move transfer、callable storage 失败、逆序析构、callback throw；borrow 必须活到 guard | owning callable 的 `ScopeGuard`/scope-exit value；普通 typed handle 仍优先自己的 RAII | `COMPLETE`：CTAD、nodiscard、move、dismiss/invoke、constructor failsafe 与 defer macro | Abseil Cleanup + Folly ScopeGuard；通用 `MUST` |
| `UTILC-A-07` | 只在 exception failure/success 执行 cleanup，或一个 handle/deleter 组成 typed unique resource | `uncaught_exceptions` 边界、deleter state、empty/release/reset、move、invalid handle、析构 no-throw | 工具链 `<scope>` 的 `scope_fail/scope_success`；资源用 `unique_ptr`/owner RAII/可用的 `unique_resource`，不扩成万能 guard | `ROUTED`：status-return 代码可用 guard + commit；exception/typed-resource 由标准和 owner 完整承载 | Abseil/Folly breadth 用于选择边界；无 HYLE consumer 证明需要第二套类型，明确不因 peer API 数量复制 |
| `UTILC-A-08` | 热对象把 strong count 放在对象内，以单指针 owner 跨函数/线程传递 | initial ownership、adopt/retain、copy/move/reset/release、const/upcast/casts、custom destroy、final handoff ordering、identity hash/order | `RefCounted<T>` + one-word `Ref<T>`；adopt 与 retain 在调用点显式，释放永不抛 | `COMPLETE`：强所有权 surface 与 atomic final-release ordering 完整，duck-typed counter 也可接入 | Boost intrusive_ptr + Folly intrusive ownership；foundation hot object `GENERAL`，不因当前仓内零调用删除公开能力 |
| `UTILC-A-09` | 确定 thread-confined 的 AST/plan/object graph 不应为每次 retain/release 支付原子 RMW | thread-safe 与 thread-unsafe 是类型级选择；跨线程误用是前置条件；计数宽度/溢出、copy base、custom teardown 一致 | 同一 `Ref<T>` handle 接 counter-policy base，或一个明显命名的 non-atomic base；不能运行时分支 | `PARTIAL`：`Ref` 可接用户 duck type，但内置 `RefCounted<T>` 只有 atomic policy，普通用户没有自然的 non-atomic shape | Boost `thread_safe_counter/thread_unsafe_counter` 是成熟直接证据；性能基础库 `GENERAL`，待 Graph/Container/Sched 反查后定 owner/API |
| `UTILC-A-10` | 对象需要 weak observer、环打破、aliasing、非侵入 ownership 或 lock-free publication | strong/weak control block、expired/lock、cycle policy、alias subobject、allocator、atomic load/store 与 reclamation | 直接使用 `std::shared_ptr/weak_ptr`、`atomic<shared_ptr>` 或领域 generation/hazard handle；不能给每个 intrusive object 常驻 weak metadata | `ROUTED`：HYLE `Ref` 刻意只承载 strong intrusive；标准库路线完整 | llama Vulkan/adapter/cache 大量 shared/weak/unique ownership；真实 `MUST` journey 证明路线，而不是要求 HYLE 复制 control block |
| `UTILC-A-11` | intrusive object 要从 injected allocator fallibly 构造，并用同一 allocator 销毁 | size/alignment、constructor failure、OOM、partial construction、deleter state、Result propagation；owner lifetime | owning module 的 `Result<Ref<T>>` factory 完成 allocate/construct/adopt，自定义 `destroy` 回到同一 allocator | `ROUTED`：`make_ref` 是 throwing `new` convenience；Mem/Error/owner 可完整组合 fallible 路线 | HYLE Mem 与 Ref public contract；反向让 Util-core 依赖 Mem/Error 会破坏基础闭包，明确留给 owner |
| `UTILC-A-12` | 极短低争用状态用紧凑 exclusive lock；原子轮询需要目标 ISA 的低功耗/progress hint | Lockable/try/unlock、TTAS/backoff、尺寸、memory order、x86/Arm/RISC-V hint、unknown fallback、公平/递归/hold-time 边界 | compact `SpinLock` + `std::lock_guard`；单独的 `cpu_relax` 只表达 hint，不建立同步；支持目标有可用 pause feature 时不能悄悄退成空操作 | `PARTIAL`：锁与 x86/Arm hint 完整，但当前 RISC-V 即使编译目标有 Zihintpause 也落入 no-op；owner 也无法询问 pause feature | ggml CPU threadpool 明确含 RISC-V Zihintpause 分支，oneTBB/Folly 提供同类短锁；跨支持平台一致性为 `MUST` 候选 |
| `UTILC-A-13` | 读多写少或中等争用小状态需要 shared locking、短 spin 后 parking，而不是纯烧核或重量 mutex | SharedLockable、writer starvation/fairness、try/upgrade、spin budget、atomic wait/notify、thread/fiber 区分、false sharing | 按语义命名的 compact RW spin 或 adaptive thread mutex；fiber-aware wait 仍归 Sched；不提供“fast/slow lock”档位 | `MISSING`：Container 已有私有 AdaptiveMutex/AdaptiveRwLock，ggml 有 OS shared mutex；Util-core 只有 exclusive spinner | oneTBB `spin_rw_mutex` + Folly RW/Micro spin families，以及 HYLE 重复实现；`GENERAL` 进入跨域因子化，不预判公开还是私有 foundation |
| `UTILC-A-14` | 多个 lock-free/parking state machine 都要做 bounded pause、yield、park，而各自仍拥有 wake condition | reset、iteration budget、ISA hint、thread yield、deadline/cancel、park token；不同算法不能共享一个错误的固定常数 | 可复用部分至多是 tiny backoff/hint policy；何时 park/wake 继续由 Container/Sched/Mem owner 决定 | `PARTIAL`：`SpinBackoff` 私有，Container 两份 pause/backoff，Sched 又手写 pause/yield；标准 `atomic::wait` 只覆盖 park primitive | oneTBB internal atomic-backoff + Folly spin families用于机制对照；HYLE 重复是 `GENERAL` 去重输入，不据此强造公共类 |
| `UTILC-A-15` | 同步 API 在一次调用期间借用任意 callable，不拥有、不复制、不分配，同时避免每个 API 重写 trampoline | void/value/noexcept signature、function/member/lambda、const/mutable、nullable 与 lifetime-bound；绝不能存入异步 record | two-word `FunctionRef<R(Args...)>`-like borrowed view，按值参数；存储/返回/异步队列使用 owning callable 或显式 fn+ctx | `MISSING`：Platform visitor 有自制 trampoline，HYLE 有多套同步模板与 raw pair，但没有统一 borrowed callable vocabulary | Abseil/Folly FunctionRef + HYLE/ggml callback journeys；通用 `GENERAL`，需逐 consumer 区分 call-only 与 stored callback 后再准入 |
| `UTILC-A-16` | scheduler/thread/IO 要把 callable 保存到稍后、跨线程或重复调用 | copy/move-only、SBO/allocator、exception/noexcept、cancel/destroy、module lifetime、one-shot/repeating；ownership 必须显式 | 普通冷路径用 `std::function/std::move_only_function`；allocator/stack/fiber 约束强的热 record 由 owning module 类型擦除 | `ROUTED`：Platform/Sched 已有自然 lambda convenience 与窄 fn+ctx core；没有通用 owning callable | llama 大量 `std::function`，Folly Function 提供更宽选择；当前无两个 HYLE owner 共享完全相同 storage 合同，拒绝先造万能 wrapper |
| `UTILC-A-17` | API 区分 required borrow、nullable borrow、optional reference 与“失败但成功时非空”的 borrow | temporary rejection、const conversion、pointer-like smart handle、lifetime-bound、null check、Result composition；不拥有对象 | `T&`/annotated or wrapped non-null pointer 表达 required；`T*`/optional-ref 表达 nullable；fallible non-null borrow 与 Error carrier 对齐 | `PARTIAL`：语义可由 raw pointer/reference/reference_wrapper 组合，但全库大量非空注释，Error 也缺统一 direct borrow carrier | Abseil nullability/optional_ref + Folly not_null；已与 `ERROR-A-04`、`CORE-A-05` 形成跨域 `GENERAL` 候选，最终只保留一个 vocabulary owner |
| `UTILC-A-18` | 日志、测试和调试在 compile time 取得 static type spelling/label，不依赖 RTTI | cv/ref/array/function/incomplete/local type、unsupported compiler、static storage、hash collision、TU/plugin/compiler/build 边界 | `constexpr string_view type_name<T>()` + diagnostic hash；correctness dispatch 绝不只比较 hash | `COMPLETE`：probe-sliced compiler name 与 FNV diagnostic hash 已明确边界 | Folly TypeInfo + Boost type-name 路线；定义性诊断 `GENERAL`，ggml 的显式 tensor/op enum name 是领域 identity 而非替代品 |
| `UTILC-A-19` | runtime type erasure/registry 需要进程内无碰撞 token，或持久/plugin/wire 协议需要稳定 ID | RTTI on/off、DSO unload、address uniqueness、collision、version/schema、serialization | build-local runtime dispatch 可用 `std::type_index`/显式 address token；跨 DSO/持久协议必须用 owner enum/schema/UUID | `ROUTED`：runtime demangle 在 Util-runtime，correctness identity 走显式 owner schema；`type_hash` 不承担此合同 | Abseil FastTypeId + Folly TypeInfo 用于发现；llama/ggml 用显式 enum/vtable identity，未证明 HYLE 需要第二个全局 type registry |
| `UTILC-A-20` | 用户需要 span/optional/variant/expected/bit/source-location/RAII lock/owning function 等通用语言积木 | C++23 标准语义、constexpr/noexcept、range/iterator、compiler floor 与 feature availability | 能由当前 C++23 合同自然表达的直接使用 `std::*`；HYLE 只补真实缺口或跨工具链不一致，不建兼容层大全 | `ROUTED`：当前 Core/Error/Container 只对有独立语义的能力拥有类型 | 定义性工程能力检查；peer 为旧标准提供的 polyfill 不构成 HYLE 新 API 准入理由 |
| `UTILC-A-21` | source/static/shared 用户只选低层 header 能力，不拉入 allocator/container/runtime artifact | header self-containment、component selection、直接 Core edge、模板/inline ODR、无虚构 binary journey；Native FFI 不镜像模板 | header-only C++ component 或更小因子的聚合 target；安装只交付所选 closure 的 headers | `COMPLETE`：`hyle::util_core -> core` INTERFACE target 与五头交付完整 | 当前 package/source journey；交付 `MUST`，但当前五族是否仍应捆成一个因子留到 B |

public-surface 反向追踪覆盖 `<hyle/util/{random,ref_counted,scope_guard,spin_lock,type_name}.h>`、独立
`hyle::util_core` target、当前 architecture/user guide 和生产树调用。当前仓内只有 SpinLock/cpu_relax 有
真实生产采用：Mem、Sched 与冻结但未完成的 Graph；Rng、Ref、ScopeGuard、type-name 的仓内调用为零，
但各自有标准/成熟 peer 和自然 SDK journey，不能因“内部没用”删除。llama.cpp 的 ownership 以
`unique_ptr/shared_ptr/weak_ptr` 为主，证明 weak/non-intrusive 路线应保持标准库可选，并不证明单指针
intrusive strong owner 无效。

第一遍得到六个跨域候选，而不是立即扩写这个 header-only 聚合包：

1. 当前 Rng 能直接组合 llama.cpp 使用的 uniform/discrete/shuffle，copy/state 也能表达 speculative
   checkpoint；token sampling 数值算法和 backend commit policy 留给 Math/推理 owner；
2. `cpu_relax` 在已支持的 RISC-V 产品目标上静默 no-op，是支持矩阵缺口；最终修复位置要与 Core 的
   compile-target facts、Platform runtime feature 和不可拆分 SIMD 路由共同决定；
3. thread-confined intrusive counter 是成熟的零原子语义档，当前 duck-type escape hatch 不等于自然用户
   shape；是否补 counter policy 要由 Graph/Container/Sched 的真实 ownership 反查决定；
4. Container 私有 adaptive locks、Sched backoff 和 Util-core spinner 已形成同步机制重复；第二阶段要区分
   可复用 thread primitive、fiber policy 与 owner-specific state machine，不能把所有锁堆进 `util`；
5. borrowed callable 与 non-null/optional borrow 是成熟的低成本 vocabulary，并有 HYLE 重复形状；但
   stored async callback、fallible borrow 和 compiler annotation 各有不同生命周期，必须分别与 Sched/IO、
   Error、Core 去重后才决定是否产生公共类型；
6. 当前 `util_core` 把五个语义无关但同为 header-only/Core-only 的能力聚合在一个 target。它可能只是低
   成本交付聚合，也可能包含多个可独立消费的因子；B 阶段按依赖缩减、成本隔离和用户选择收益裁决，
   不能因名字带 `core` 先验认定不可再分。

这表示 Util-core 产品域的阶段 A 第一遍已经完成；当前 `CLOSED` 组件不重开，任何缺口只有在 13 域
完成跨查、确定唯一 owner 和有限 FIX 集后才形成 `PRODUCT_TRIGGER`。

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
