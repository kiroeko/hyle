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

#### A1. Core 产品域（能力账本已随 A 阶段冻结）

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

#### A2. Error 产品域（能力账本已随 A 阶段冻结）

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

#### A3. Container-intrusive 产品域（能力账本已随 A 阶段冻结）

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

#### A4. Util-core 产品域（能力账本已随 A 阶段冻结）

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

#### A5. Time 产品域（能力账本已随 A 阶段冻结）

本节把 Time 分成“时钟/时间值”“阻塞等待”和“只管理节点、不执行回调的 timer data structure”，不把
完整 scheduler、calendar 数据库或设备 profiling 强塞进同一 target。直接推理输入包括 llama.cpp 的随机
seed、下载重试/休眠和时间文本，ggml CPU/Vulkan/WebGPU/设备 backend 的 elapsed/profiling journey，
以及 HYLE Platform/Sched/IO/Obs 的 deadline、timestamp 与 timer-node 消费。新增 strong peers 锁定为
Seastar `fd71a5b4c71b91abd4e93c9d20f88e1fd7465976` 的 `timer/manual_clock/lowres_clock`，以及 libevent
`e1f0335d2f3af1235421c939c62943bb99b3d652` 的 min-heap/common-timeout；可注入时钟形状另用已锁定
Abseil `Clock/SimulatedClock`，每一能力行仍不超过两个直接 peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `TIME-A-01` | 超时、排队与 elapsed 计算取得不回退的 host 时间 | steady identity、ns 精度、跨线程可比、进程生命周期；不能与 realtime 数值混用 | `MonotonicClock/TimePoint/Duration` 与 `monotonic_now/ns()`；typed 路线优先，raw ns 只给紧凑状态机 | `COMPLETE`：标准 steady clock identity 与 typed/raw 两条路线齐全 | ggml timers、llama timeout、Seastar steady timers；基础 `MUST` |
| `TIME-A-02` | 文件元数据、协议和用户事件取得 Unix wall time | realtime 可校时/回退，1970 前后、负秒、ns normalization、系统时钟跳变；不能用来算 deadline | `RealtimeTimePoint` + `UnixTimestamp{seconds,nanoseconds}`，转换显式 fallible | `COMPLETE`：realtime typed/raw/timestamp 和合法性合同完整 | llama metadata/log timestamps + HYLE Platform/IO；基础 `MUST` |
| `TIME-A-03` | 大量统计、日志限流或粗粒度 housekeeping 避免每次读取最贵时钟 | realtime/monotonic 两域分离、availability、平台 fallback、精度不承诺 | `coarse_*_now/ns/timestamp()` + capability predicate；语义不适用时显式退化到 precise | `COMPLETE`：Windows/Linux/macOS route 与 availability 均有 | Seastar lowres clock + HYLE Obs rate-limit journey；高频通用 `GENERAL` |
| `TIME-A-04` | benchmark、tracing 与 kernel instrumentation 读取最低开销 host profiling counter | raw unordered read、ordered start/stop pair、counter stability/frequency/calibration、unsupported fallback | `profiling_clock_ns()` 与成对 measurement API；availability 可查，结果仍是 ns duration domain | `COMPLETE`：x86 invariant TSC、Arm virtual counter 与 monotonic fallback | ggml/Vulkan profiling + Seastar/定义性测量 journey；性能库 `MUST`，但它不替代公平仪器 |
| `TIME-A-05` | scheduler/diagnostics 区分 wall elapsed 与当前线程或整个进程 CPU 消耗 | current-thread/process、平台失败的确定值、暂停/迁移、累计不回退；arbitrary-thread/children 另行 | chrono-compatible `ThreadCpuClock`/`ProcessCpuClock` 和 raw ns | `COMPLETE`：Windows/POSIX 两类累计 clock 已提供 | HYLE performance/Obs + Seastar thread-cputime clock；基础诊断 `GENERAL` |
| `TIME-A-06` | backend/device event 需要 CUDA/Vulkan/Metal/accelerator 自己的时间域与相关性 | device tick、queue event、host-device calibration、wrap、异步 completion、跨设备不可比 | owning Compute backend 的 event/timestamp query；需要关联时产出显式 correlation sample | `ROUTED`：Time 只拥有 host clocks，Compute backend 自有 profiling | ggml CUDA/Vulkan/WebGPU journey；推理 `MUST`，但 host Time 不应伪装设备 clock |
| `TIME-A-07` | 任意 chrono duration 转紧凑 ns deadline，极值输入不能溢出或变成超长等待 | signed/unsigned/floating、负值、NaN/inf、ratio scaling、saturating add/sub、past deadline | `duration_ns_saturated`、typed deadline/remaining 与显式 raw helpers | `COMPLETE`：wide multiply/divide、饱和和 typed conversion 完整 | llama retry/sleep + HYLE Sched/IO；基础 `MUST` |
| `TIME-A-08` | 同步宿主线程相对睡眠或睡到绝对 monotonic deadline | zero/past/huge duration、interrupt/spurious wake、oversleep、OS timer lifetime；不是 fiber wait | `sleep_for(duration)` / `sleep_until(MonotonicTimePoint)`，`noexcept` 且饱和 | `COMPLETE`：Windows waitable timer、POSIX nanosleep/clock sleep route | llama download delay + 标准线程 journey；通用 `MUST` |
| `TIME-A-09` | fiber/coroutine 等待 deadline、取消或 timeout，同时不阻塞 worker | ready/cancel/deadline race、timer ownership、wake exactly once、structured task lifetime | Sched 的 `sleep_until/for` 与 deadline/cancellation state；Time 只交付 clock/queue primitive | `ROUTED`：Sched 已消费 `MonotonicTimePoint` 和 `TimerQueue` | llama async/server 与 HYLE Sched；推理 runtime `MUST`，owner 不是 Time facade |
| `TIME-A-10` | POSIX/Windows/filesystem/protocol 在边界交换 Unix timestamp | `time_t` signedness/range、`timespec` normalization、FILETIME epoch/ticks、system-clock range | `optional` conversions，不用 magic sentinel；canonical value 保留 ns | `COMPLETE`：time point、time_t、timespec、FILETIME 双向覆盖 | HYLE Platform fs + IO protocols；跨平台 `MUST` |
| `TIME-A-11` | 日历日期、时区/DST、locale 和文本 parse/format | civil arithmetic、tzdb update、ambiguous/skipped time、RFC/ISO format、locale/error | C++23 chrono/tzdb 或协议 owner 的 formatter/parser；运行时 tz database 是独立交付决策 | `ROUTED`：Time 保持 machine-time primitive，当前 llama 文本由标准路线完成 | llama chat/server text + Abseil civil/timezone 用于发现；真实 journey `MUST`，不要求 HYLE 复制 tzdb |
| `TIME-A-12` | 用户方便地做一次 elapsed/lap 比较且不误用 realtime | snapshot、elapsed/until、回退 clamp、极值、copy/order；统计记录由调用者拥有 | 小值 `Instant`，可取 monotonic time point；不隐式日志或注册 metric | `COMPLETE`：`Instant::now/elapsed/until` 和安全非负差完整 | oneTBB tick-count/标准 stopwatch journey；易用性 `GENERAL` |
| `TIME-A-13` | 单 event-loop 管理大量任意 deadline，常见路径 arm/cancel，advance 批量交付 | stable caller-owned node、single owner、reschedule、past deadline、cancel-all、queue-first/entry-first teardown、总序不承诺 | intrusive `TimerEntry` + allocation-free `TimerQueue`；回调/command queue 不进入节点 | `COMPLETE`：65-bucket exact radix queue，arm/cancel/next O(1)，advance 与触达/交付成正比 | Seastar timer-set + HYLE Sched production journey；runtime `MUST` |
| `TIME-A-14` | 数百万连接共享同一 duration，按 arm 顺序形成自然 deadline 序列 | one duration per queue、nondecreasing arms、transactional reject/reschedule、FIFO equal deadline、bulk detach | `CommonTimeoutQueue(duration)`；duration-to-queue router 由 IO owner 按真实档位拥有 | `COMPLETE`：append/pop/unlink/next 均 O(1)，ordering precondition 可恢复检查 | libevent common timeout + server idle-timeout journey；IO 加速 `GENERAL` |
| `TIME-A-15` | timer 批次必须按 deadline/priority 稳定全序，或需要 indexed update | equal-key stability、erase/update handle、allocation、callback reentrancy、batch size | 默认 queue 交付 batch 后由 caller sort；反复需要 handle heap 时用 Container 的 intrusive/indexed heap factor | `ROUTED`：TimerQueue 明示不保证跨 bucket 总序，现有需求可组合排序 | libevent heap + Seastar timer-set；能力 `GENERAL`，不能为罕见排序给每个 timer 常驻 metadata |
| `TIME-A-16` | 周期 housekeeping、metrics flush、网络重传按固定延迟或固定速率重复 | fixed-delay/fixed-rate、skip/catch-up/bounded burst、drift、cancel/rearm race、callback duration | Sched/IO 的 repeating task/timer policy，内部复用一次性 timer node；必须让语义在 API 名字可见 | `MISSING`：全库只有一次性 sleep/timer，Obs 的周期日志不是 scheduler | Seastar periodic timer + llama server housekeeping journey；跨域 `GENERAL` 候选，最终 owner 为 Sched/IO 而非低层 Time queue |
| `TIME-A-17` | scheduler、timeout 算法和重试测试在不睡真实时间的情况下确定性推进 | manual now/set/advance、多个 waiter、回退策略、clock lifetime、真实/虚拟选择在类型或注入点固定 | clock concept/小型 provider 注入 owning runtime；单元测试可用 `ManualClock`，生产默认 system monotonic | `MISSING`：queue 可显式传 now，但 Sched 和 deadline helpers 固定读取全局 clock | Seastar `manual_clock` + Abseil `SimulatedClock`；确定性 runtime 测试 `GENERAL`，带入 Sched 因子化 |
| `TIME-A-18` | 一个 executor 跨线程 arm/cancel timer，并在到期执行 callback | command serialization、callback ownership/exception、cancel race、shutdown、periodic、worker affinity | Sched/IO timer service 或 executor handle；Time data structure 只返回 due nodes | `ROUTED`：TimerQueue 明确不执行回调，Sched 已拥有 worker/fiber lifecycle | Seastar/libevent runtime journey；基础 runtime `MUST`，拒绝把锁和 owning callable塞入 `TimerQueue` |
| `TIME-A-19` | deadline/cancellation budget 跨多层 RPC、IO、queue 与 compute 传播 | absolute vs relative、clock domain、child budget、cancel reason、timeout result、serialization | typed monotonic deadline + Sched/IO cancellation/deadline context；跨机器协议传相对 budget 或 wall metadata | `ROUTED`：Time 的 deadline value 完整，传播状态机尚由各 owner 组合 | llama RPC/server 与通用 structured concurrency；`GENERAL`，在 Sched/IO 清点后裁决公共 vocabulary |
| `TIME-A-20` | source/static/shared C++ 用户只选择 host time primitive 和 timer data structures | header/compiled split、platform implementation、component visibility、模板/inline ODR；无伪 C/Native FFI mirror | `hyle::time` compiled component + 自包含 C++ headers；FFI 只在未来 runtime object consumer task 中类型擦除 | `COMPLETE`：component target、visibility 和三种 C++ delivery journey 已存在 | 当前 build/package contract；交付 `MUST` |

public-surface 反向追踪覆盖 `<hyle/time/{clock,unix_timestamp,timer_entry,timer_queue,
common_timeout_queue}.h>`、`hyle::time` target，以及 Platform filesystem、Sched timer/sleep、IO timeout 和
Obs timestamp/rate-control 调用。第一遍得到四个跨域候选：

1. host machine-time、blocking thread sleep 与 callback-free timer data structure 已形成完整低层闭环；日历/
   时区和 device clocks 有明确 owner，不是 Time API 数量缺口；
2. manual/injected clock 是成熟 peer 与确定性 scheduler 测试共同证明的真实缺口；仅让 `TimerQueue` 接受
   显式 `now` 不能替代 runtime 的可注入时间源；
3. periodic policy、cross-thread timer service 与 deadline/cancellation propagation 属 Sched/IO 生命周期，
   但全库能力超集必须承认这些 journey，不能因 Time 本身刻意 callback-free 就从账本消失；
4. arbitrary-deadline radix queue、same-duration queue 和 total-order/indexed heap 是三种不同成本结构；前两者
   已有自然形状，第三者应优先复用 Container factor，不能为 benchmark 统一成一个万能 timer 类型。

这表示 Time 产品域的阶段 A 第一遍已经完成；当前 `CLOSED` 模块不重开，manual clock、periodic policy
等只有在 13 域跨查并确定 owner 后才形成有限 `PRODUCT_TRIGGER`。

#### A6. Platform 产品域（能力账本已随 A 阶段冻结）

本节从“跨 OS 的 host 事实与资源机制”展开，而不是从当前九个 header 反推边界。直接推理输入包括
llama.cpp/ggml 的 CPU feature/physical-core 选线程、affinity/priority、NUMA、CPU-memory sizing、模型
mmap/mlock/prefault/fragment release、临时文件原子替换和动态 backend 加载，以及 HYLE
Mem/SIMD/Container/Sched/IO/Compute 的全部 public reverse trace。strong peers 按能力行使用已锁定的
Folly `MemoryMapping/File/Random/portability`、oneTBB `info/numa/scalable allocator` 与 Boost
Filesystem/Interprocess 中至多两个；标准 OS/C++23 合同不计作额外 peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `PLAT-A-01` | fat binary 在执行 kernel 前发现当前进程可安全使用的 CPU 指令 | x86 CPU bit + OS XSAVE、Arm auxv、RISC-V hwprobe/auxv、未知 false、baseline probe TU/LTO 隔离；hardware 与 current-thread enable 分开 | immutable `CpuFeatures` snapshot；每个 feature 是事实，不让 consumer 自跑 CPUID | `COMPLETE`（现有字段）：x86/Arm/RVV feature 和 OS state gate 已有 | ggml CPU dispatch + Folly/定义性 portability；推理 `MUST` |
| `PLAT-A-02` | scalable/matrix kernel 还需要向量长度和每线程 AMX/SME/SVE state | SVE/RVV bytes、VL 可按线程改变、SME/SME2、AMX tile permission、enable/restore、kernel/OS support 分离 | Platform 提供窄的 current-thread capability/enable mechanism；SIMD 唯一决定 target 和调用 route | `PARTIAL`：能探 SVE/RVV feature，但无 VL、SME/AMX 或 thread enable；Highway 内部 `Lanes()` 只覆盖其 own target | 当前 ggml 明确读取 SVE/RVV width 并启用 AMX/SME kernels；推理 `MUST` 候选，带入 SIMD 边界而非扩散 `HAVE_*` 宏 |
| `PLAT-A-03` | 普通模块只需选择一个 coarse host ISA tier | fixed/fat build、同 arch ordering、compiled-target 与 runtime-target 分离、override/testing | `best_isa/supports/name` 是轻量 platform fact；完整 routing manifest 只由不可拆分 SIMD product 拥有 | `COMPLETE`：scalar/x86/Neon/SVE/RVV coarse tier 齐全 | HYLE SIMD/Container production trace；基础 `MUST` |
| `PLAT-A-04` | allocator/kernel 取得 cache line 与 L1/L2/L3 容量提示 | instruction/data/unified、unknown、异构核差异、sharing domain、runtime 事实不能进入 stable ABI layout | conservative Core 常量做 layout；Platform snapshot 做容量/策略提示 | `PARTIAL`：line + 单份 L1d/L1i/L2/L3 容量有，缺 per-CPU/share/associativity facts | ggml cache-sized tiling + Folly/oneTBB topology；优化 `GENERAL`，与下一行统一审计 |
| `PLAT-A-05` | inference runtime 选择物理核、避开 SMT sibling、按 LLC/cluster 分 worker | package/die/NUMA/core/thread/cache-domain IDs、稀疏 CPU、P/E kind、cpuset 动态交集、unknown 不猜 | buffer-fill `CpuLocation/CpuTopologySnapshot` keyed by available CPU id；Sched 再决定 worker policy | `MISSING`：只有 system logical/P/E count，没有 physical core、SMT sibling 或 cache sharing | llama `common_cpu_get_num_physical_cores` 和 lockstep 跳 SMT 是直接 consumer；oneTBB info；推理 `MUST` 候选 |
| `PLAT-A-06` | runtime 按当前进程真正可用 CPU sizing，并可优先强核但不丢集合 | dynamic affinity/cgroup/cpuset/Windows CPU Sets、sparse ids、OS order、可信 performance rank、query churn | count-only + buffer-fill ids/info/order；策略输入是值，不在 Platform 自动删核 | `COMPLETE`：available count/ids/info 与 P/E ordering 均有 | llama cpumask/thread count + HYLE Sched；推理 `MUST` |
| `PLAT-A-07` | runtime 把 available CPU 分 NUMA node 并询问当前线程 node | sparse nodes、CPU-to-node、current node、single-node floor、query failure；distance/latency 另行 | stable node span + fallible current/cpu lookup | `COMPLETE`：node list/membership/count/current/cpu mapping 完整 | ggml NUMA initialization + oneTBB NUMA；推理 `MUST` |
| `PLAT-A-08` | 大块 host arena 选择 preferred、strict bind、all/selected weighted interleave | existing/reserve/committed range、soft fallback、selected order/chunk、first-touch、whole-owner rollback、placement 不是 residency proof | `VmRegion` RAII + typed policy/result；调用者显式选 node/order | `COMPLETE`：Linux/Windows/macOS capability、selected-node/chunk 与 O(nodes)/O(pages) 边界完整 | ggml NUMA modes + oneTBB allocator constraints；推理 `MUST` |
| `PLAT-A-09` | model planner 在分配前取得 host 物理内存、当前可用预算与进程/container 限制 | total/available、commit/cgroup/job limit、swap/pressure、snapshot race、unified-memory 与 discrete device budget 分开 | fallible `HostMemoryInfo` snapshot，字段可 unknown；Compute 把 host fact 与 device heap 聚合成 planner view | `MISSING`：只有 page geometry，没有 host memory capacity/available query | ggml CPU device `get_memory` 和 llama layer fitting 是直接 consumer；OS 定义性能力；推理 `MUST` 候选 |
| `PLAT-A-10` | VM owner 做页对齐和 mapping granularity 计算 | base page、allocation granularity、large-page candidates、overflow、平台值缓存 | `page_size` + checked align helpers；需要 file offset granularity 时显式字段 | `COMPLETE`（当前 journey）：page/large page size 与 saturating align 已有 | llama mmap alignment + Mem/Stack production trace；基础 `MUST` |
| `PLAT-A-11` | arena/JIT stack owner reserve address、分段 commit/decommit、protect、整段 release | zero/overflow、page rounding、reserve vs committed、subrange、move/adopt、failure preserving owner | raw `vm_*` mechanism + move-only `VmRegion`; normal RW 一步 convenience | `COMPLETE`（非 executable）：reserve/RW/commit/decommit/protect/release/adopt 完整 | Folly/Boost VM + HYLE Mem/Sched stack；基础 `MUST` |
| `PLAT-A-12` | fiber/native stack 需要 no-access guard 并保持一个清晰 owner | guard direction/count、usable span、commit failure、move/release、stack growth model | `VmRegion::reserve_guarded` 返回 owner + usable view | `COMPLETE`：低 guard page + committed usable span 已有 | Boost Context/Folly fibers + HYLE Sched；runtime `MUST` |
| `PLAT-A-13` | CPU JIT/code cache 需要 W^X 转换、instruction-cache flush 与 dual mapping | RW build/RX publish、never-RWX policy、icache coherence、code lifetime、platform entitlements | 独立 executable-memory owner，不给通用 `Protect` 随手加 RWX；Compile/Graph consumer 决定是否准入 | `OUT_OF_SCOPE`：`Protect` 只有 none/read/RW | 没有当前 HYLE/locked inference consumer；peer existence 不足，具名 CPU-JIT task 才触发 |
| `PLAT-A-14` | 大 arena 选择 THP/large/hugetlb 页并可 prefault | hint vs guaranteed page、size inventory、privilege/reserve、NUMA、fallback、touch cost | Platform 只给 page facts/OS advice；Mem 的 `HugeRegion`/mimalloc reserve 拥有 allocation policy | `ROUTED`：Platform hint/size + Mem HugeRegion/prefault/1GiB route 完整；无通用 guaranteed huge-page owner | Folly HugePages/oneTBB allocator + HYLE Mem；大模型 `GENERAL` |
| `PLAT-A-15` | 创建有栈、名字、affinity、priority 的长寿命 OS thread，并明确 join | raw fn+ctx、owning move-only callable、config before entry、best-effort/required、exception boundary、move/join | move-only `Thread` + `ThreadOpts`; 大量短任务仍进 Sched | `COMPLETE`（join lifecycle）：allocation-minimal 与 ergonomic spawn 两条路线齐全 | ggml threadpool + oneTBB/标准线程；基础 runtime `MUST` |
| `PLAT-A-16` | 明确 fire-and-forget native service thread，或希望 stop-on-destruction | detach 后 handle/lifetime、process teardown、callable ownership、cooperative stop、join failure | 普通 stop-aware owner 优先 `std::jthread`/Sched task；若必须 native opts，提供显式 `detach` 而非析构暗 detach | `PARTIAL`：custom-config Thread 只支持 join，标准 `jthread` 可完成无自定义 stack/affinity 的路线 | 标准线程 lifecycle；通用 `GENERAL`，需 IO/Obs backend 反查后决定是否扩 Platform owner |
| `PLAT-A-17` | 主线程临时改 affinity，完成探测/工作后恢复原 mask | get/set、sparse/large mask、inherited state、scope rollback、macOS unavailable、query/set race | fallible `this_thread::affinity()` snapshot + set，必要时上层 ScopeGuard 恢复 | `MISSING`：只能 set，不能查询/恢复 | llama hybrid-core probe 明确 save/set/restore，ggml main thread cleanup；推理 `MUST` 候选 |
| `PLAT-A-18` | 服务进程或 worker 调整 scheduling priority/QoS | current thread vs process class、portable ladder、permission、realtime hazard、restore、required/best effort | thread priority 留在 Thread API；process-wide side effect 用独立显式 function/value，不藏进 spawn | `PARTIAL`：thread priority 完整，process priority/class 不存在 | llama `set_process_priority` 是直接 consumer；`GENERAL`，高风险 realtime 必须显式 |
| `PLAT-A-19` | 自有原子状态机在值未变时无轮询阻塞，并定点 wake | 32-bit wait、spurious wake、timeout、wake-one/all、address lifetime、memory order由状态机拥有 | low-level detail wait/wake；普通 C++ 首选 `atomic::wait`，OS timed wait 由窄 wrapper 补 | `COMPLETE`：futex/WaitOnAddress/Apple address wait 完整 | ggml threadpool + HYLE Sched/Container; runtime `MUST` |
| `PLAT-A-20` | 单 worker park/unpark，允许通知先到且合并重复 token | one waiter、token state、timed single-shot、spurious wake、owner lifetime、not semaphore | `Parker` + cheap borrowed `Unparker`; multi-waiter/cancel queue属 Sched | `COMPLETE`：三态 token protocol 已有 | Rust/Tokio shape + HYLE Scheduler；runtime `MUST` |
| `PLAT-A-21` | 配置/model/cache 需要同步文件 owner 与普通 cursor I/O | open flags、move/close、short read/write、exact/all、seek/position、sync、EOF/zero、UTF-8 path | one-word `FileHandle` RAII + span overloads；失败走 Status/Result | `COMPLETE`：常用同步 owner journey 完整 | llama file loader + Folly File；基础 `MUST` |
| `PLAT-A-22` | 并发 loader/IO 需要 positional、vectored、direct、async/registered file 操作 | offset independent、scatter/gather、alignment、cancel/completion、shared owner、backpressure | IO 的 file/proactor handle；Platform 不给单 cursor owner 加锁或 OVERLAPPED state | `ROUTED`：Platform 明确把这些交给 IO | ggml multi-file/model loading + Folly async IO；推理 `MUST`，待 IO 清点验证覆盖 |
| `PLAT-A-23` | 用户查询和变更基本 filesystem 对象 | stat/exists/type/size/mtime、mkdir/remove/rename/copy/touch、permission、link/symlink、missing vs error | UTF-8 path + typed snapshot/Status；纯词法 path 不在此 owner | `COMPLETE`（当前 portable subset）：基本 query/mutation 全有 | llama cache/download + Boost Filesystem；基础 `MUST` |
| `PLAT-A-24` | 遍历目录或安全删除 caller-owned tree，数据量不可先完整 materialize | early-stop visitor、entry kind、不跟 link、concurrent replacement threat model、depth/path/handle O(depth) | allocation-free visitor core + C++ callable wrapper；recursive delete 明示 destructive/lifetime contract | `COMPLETE`：native iterator + bounded growth/错误语义完整 | ggml backend discovery + Boost/Folly FS；基础 `MUST` |
| `PLAT-A-25` | cache/download 原子创建临时文件，写/flush 后替换目标 | same-directory unique create、non-inherit、path ownership、durability、rename overwrite/race、cleanup on failure | `CreatedTempFile` + FileHandle sync + `rename_path` 可组合；只有共享事务合同才再造 `AtomicFileWriter` | `COMPLETE`（机制组合）：安全 temp owner/rename/sync 均有 | llama download/OpenCL cache journey；基础 `MUST`，不为 convenience 重复 owner |
| `PLAT-A-26` | cache/database 协调多进程 writer，或预分配/稀疏/resize 文件 | advisory/mandatory lock、lock range、crash release、truncate/preallocate/sparse、mapped-view invalidation | owning IO/storage layer 的 file lock/resize operation；若只是 open-time truncate，Platform 已覆盖 | `MISSING`（advanced journey）：无 lock/resize/preallocate；open truncate only | mature filesystem/IO practice；`GENERAL`，在 IO/real cache consumer 后定 owner，不塞进每次 FileHandle |
| `PLAT-A-27` | 大模型/数据集以 path 或已打开 handle 建 read-only/RW/COW subrange mapping | arbitrary offset、EOF bound、move/reset、const/writable view、flush、file rename while open | move-only `MappedFile`，mode 在 type state/query 中明确 | `COMPLETE`：三 mode、path/consumed handle、checked range/view/flush 完整 | llama mmap + Folly MemoryMapping；推理 `MUST` |
| `PLAT-A-28` | loader 告知 sequential/random/will-need/dont-need，并渐进锁住 CPU-resident prefix | hints 不保证、whole/prefix、eager/on-fault、quota adjust opt-in、failure preserves prefix、unlock/reset | `advise/prefetch` + `lock_prefix/locked_size`; policy 显式 options | `COMPLETE`：hint 与 residency lifecycle 分离且完整 | llama mmap/mlock + Folly MemoryMapping；推理 `MUST` |
| `PLAT-A-29` | 首次推理前同步 prefault/populate，或 lazy tensor offload 后释放 mapping fragments | hint vs guaranteed touch、range list、RSS/fault latency、holes destroy contiguous view、Windows portability、outstanding views | prefault 可由 Mem/model loader touch policy或独立 populate op；fragment mapping需要新的 segmented owner，不能破坏 `MappedFile::bytes()` | `PARTIAL`：anonymous HugeRegion 可 prefault，MappedFile 只有 hint；partial unmap 只能 separate mapping/`DontNeed` route | 当前 llama 使用 `MAP_POPULATE` 与 `unmap_fragment`；推理 `MUST` journey，具体 owner/shape在 Mem/IO/loader 因子阶段裁决 |
| `PLAT-A-30` | 多进程/remote backend 共享 host pages、handle 与 notification | named/anonymous shared memory、fd/HANDLE transfer、permissions、resize/map、cross-process synchronization、crash cleanup | IO/IPC 的 `SharedMemory` owner + Platform VM/file mechanisms；普通 `MappedFile` 不冒充 channel | `ROUTED`（规划边界）：现有 IO 是否完整待 A13 | ggml virtgpu shared-memory transport + Boost Interprocess；推理 backend `GENERAL` |
| `PLAT-A-31` | 动态发现并固定 optional backend/plugin symbol | load/by-existing/current、search/name/canonical path、reference pin、typed signature assertion、explicit close、ABI/version policy | move-only `DynLib`; plugin ABI/registry 由 consumer owner | `COMPLETE`：open/get/current/symbol/typed lookup/lifetime 完整 | ggml backend dynamic loader + Folly portability；推理 `MUST` |
| `PLAT-A-32` | 取得 cwd/temp/home/env 并在 startup 变更环境 | buffer-fill/owning convenience、unset vs empty、encoding、process-global concurrency、chdir race | low-level buffer-fill + opt-in `std::string` convenience；global mutations 明示 startup-only | `COMPLETE`：目录和 env read/set/unset 完整 | llama config/cache/plugin discovery + Boost FS；基础 `MUST` |
| `PLAT-A-33` | seed、nonce、session id 或 hash salt 取得操作系统安全随机 bytes | getrandom/BCrypt/SecRandom、partial/EINTR、sandbox/fork、blocking、failure、caller span | `secure_random(span<byte>) -> Status` 的 OS mechanism；统计 RNG 由 Util-core 用显式 seed | `MISSING`：Util `thread_rng` 明示非安全，Platform 无 entropy API | `UTILC-A-05` 反查 + Folly secureRandom/Abseil seed material；基础安全 `GENERAL` 候选 |
| `PLAT-A-34` | 设备枚举、VRAM/UMA budget、driver/queue capability 与 host-device memory relation | backend-specific IDs/heaps、dynamic budget、integrated/discrete、peer access、driver lifetime | Compute backend registry/device properties；Platform 只提供 host OS facts | `ROUTED`：冻结 Compute 拥有设备 SDK 边界 | ggml backend device properties；推理 `MUST`，不能把 Vulkan/CUDA structs 泄入 Platform |
| `PLAT-A-35` | subprocess、signal/crash、sandbox、process handle 与 IPC lifecycle | spawn/exec quoting/env/fd inheritance、wait/kill/cancel、signal safety、privilege/sandbox | 独立 Process/IO/Obs factor，在具名 consumer 下设计；不建巨型 Platform facade | `OUT_OF_SCOPE`（当前 Platform） | llama tools 有 subproc，但 HYLE 基础加速闭包无当前 consumer；`GENERAL` journey保留 trigger |
| `PLAT-A-36` | source/static/shared C++ consumer 选择 Platform mechanism 且不接受伪 C ABI | component closure、CPU probe baseline/LTO isolation、visibility、OS libraries、header self-containment、relocation | compiled `hyle::platform` + typed C++ owners；未来 foreign runtime只为具名 opaque runtime adapter | `COMPLETE`：三种 C++ delivery 和 probe isolation 已闭合 | build/package contract；交付 `MUST` |

public-surface 反向追踪覆盖 `<hyle/platform/{cpu,thread,wait,virtual_memory,numa,fs,mapped_file,
dynlib,convenience}.h>`、`hyle::platform` target 和所有直接产品消费者。第一遍得到七个跨域候选：

1. 当前 CPU feature 不是“完整 inference hardware contract”：physical core/SMT/cache-domain facts、host
   memory budget、scalable-vector width 与 AMX/SME thread state 都有当前 llama/ggml 直接旅程；
2. affinity 的 set-only 形状无法表达 llama 已使用的 save/set/restore，process priority 也不能被 thread priority
   假装覆盖；两项都属于明确但副作用可见的机制，不应藏进 scheduler policy；
3. NUMA selected-node/chunk、VM、guarded region、OS wait/Parker 和普通 file/mapping/dynlib 已形成自然闭环，
   没有因内部 consumer 特化而缺少普通入口；
4. mapped-file synchronous populate 与 fragment release 是当前 inference loader 的真实 journey。前者不能用
   will-need hint冒充，后者不能破坏连续 `MappedFile` owner；需要与 Mem/IO/model-loader 生命周期共同定形；
5. secure OS entropy 从 Util-core 反查后确认是全库基础能力缺口，最自然机制 owner 是 Platform，统计 engine
   仍留 Util-core；
6. positional/direct/async file、shared memory、device facts、fiber synchronization 和 process lifecycle都有明确
   owner route；A13/Compute 以后必须验证 route 真存在，不能在此表把“计划交给别人”写成 `COMPLETE`；
7. `platform` 当前九个头本身可能在 B 阶段形成 CPU facts、thread/wait、VM/NUMA、sync file/mapping、dynlib
   多个因子；是否拆 target 由独立消费和依赖缩减决定，不因都调用 OS 就先验捆绑。

这表示 Platform 产品域的阶段 A 第一遍已经完成；当前 `CLOSED` 模块不重开。physical topology、host
memory facts、affinity snapshot、extended ISA state、secure entropy 与 mapping population/fragment lifecycle
只有在 13 域跨查并选定 owner 后才形成有限 `PRODUCT_TRIGGER`。

#### A7. Mem 产品域（能力账本已随 A 阶段冻结）

本节采用已经触发 `mem-v2` 的语义边界：Mem 拥有 storage resource、allocation ownership、view、
mapping/residency、pool/trim/stats 与 tensor-free lifetime packing；Platform/Compute/IO 只提供 OS、device
或 completion mechanism。现有 `mem-v2` 九票在本项目阶段只作审计输入，不授权施工。直接推理证据来自
llama.cpp/ggml 的 mmap/mlock、host/pinned/device/managed buffer type、import/multi-buffer、copy/coherence、
dynamic tallocr/gallocr/reserve，以及 HYLE 全库 allocator/fiber/Container/IO/Compute 消费。additional locked
peers 为 CCCL `b7aaea69a2b07e50f09e67f2962da0243e0b7c5d`、oneDNN
`5dc67b7f37b1215d8c1f054e99c1b7cfd40317a9` 和产品 mechanism mimalloc v3.5.0
`18b08671c9302247bfb682286e6bf3cc1773f801`；每个能力族至多直接使用两个 peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `MEM-A-01` | allocator/arena/kernel 做 size/address 对齐且不在极值回绕 | pow2 validation、up/down/check、size/pointer、checked overflow、default/SIMD/cache/page alignment 分开 | constexpr size helpers + pointer helpers；动态不可信输入走 `try_*` | `COMPLETE`：alignment family 和 overflow route 完整 | HYLE Mem/SIMD/Container trace；定义性 `MUST` |
| `MEM-A-02` | 普通 host byte allocation 需要高质量通用 heap、alignment、realloc 与 remote free | zero/OOM、sized free、over-alignment、realloc failure preserves old、cross-thread free、thread cache/collect | stateless `SystemAllocator` direct route；普通 caller 不付 vtable/owner allocation | `COMPLETE`：mimalloc-backed direct allocator 与 static/shared seam 已有 | ggml host allocation + mimalloc/oneTBB；基础 `MUST` |
| `MEM-A-03` | owner 把任意 host backing 注入 Container/Sched/IO 而不模板化整个图 | borrowed provider identity、alloc/free/realloc、alignment/sizes、null OOM、copy handle、backing outlives allocations | 明确命名的 `AllocatorRef`-like small value；default invalid，按值传递 | `PARTIAL`：四字 `Allocator` 机制完整，但名字/`.get()`/lifetime 不能自然表达 borrow | HYLE 大量直接 consumer + CCCL resource-ref；基础 `MUST`，属于 mem-v2 taxonomy trigger |
| `MEM-A-04` | C++ 用户分配 typed raw array、构造 object 或取得 allocator-aware unique owner | count overflow、zeroed trivial、ctor throw rollback、destroy、move-only deleter、alignment | inline `allocate_n/reallocate_n/construct/destroy/make_unique`，失败保持 null/exception边界合同 | `COMPLETE`：typed/RAII helpers 齐全 | 标准 allocator journey + HYLE consumers；易用性 `MUST` |
| `MEM-A-05` | 长寿进程按 backing size class 预估容量并在 idle/pressure 点归还 cache | usable/good size、normal/force collect、per-thread heap、global vs owner-local、query precondition | system-specific cold queries/collect，不扩大所有 allocator vtable | `COMPLETE`：usable/good/collect 已提供 | mimalloc controls + server runtime journey；`GENERAL` |
| `MEM-A-06` | 一个 owner 需要任意大小 individual free、remote-free policy与一次 bulk release/trim，并能回注 HYLE allocator consumer | single-owner/thread-safe choice、grow/collect/abandon、owner destruction、borrowed ref、stats/quota、allocation outlive rule | move-only `Heap` owner + borrowed allocator/resource ref；mechanism 可直接复用 mimalloc heap/arena | `MISSING`：只能组合标准 PMR pool，不能自然回注非 PMR HYLE consumer | oneTBB `memory_pool` + mimalloc heap + current user journey；基础库 `MUST` 候选，已在 MEM2-02 |
| `MEM-A-07` | STL allocator-aware container 选择 system fast path或运行时 backing | propagate/copy/move/swap/equality、allocate(0)、OOM exception、EBO/always-equal、backing lifetime | `SystemStdAllocator<T>` 与显式 `StdAllocator<T>(ref)` 两型；不隐藏 system fallback | `COMPLETE`：两 adapter 及 allocator traits 路线完整 | oneTBB allocators + HYLE Container；基础 `MUST` |
| `MEM-A-08` | PMR graph/container 在 runtime 选择 system、injected 或高对齐 backing | memory_resource equality、zero-byte pair、alignment、virtual dispatch、upstream lifetime、sync/unsync pool composition | process singleton `system_memory_resource` + borrowed `PmrResource` + opt-in `AlignedPmrResource` | `COMPLETE`：三种自然 PMR route 已有 | standard PMR + oneTBB cache-aligned resource；基础 `MUST` |
| `MEM-A-09` | request/frame/ggml-context 批量分配变长对象并在 phase clear/release | owned growth、block target、large block、hard reserved limit、clear retains normal blocks、release all、usage/peak、single owner | move-only/growable `Region`；individual free 明确 no-op | `COMPLETE`：owned Region 的 growth/limit/stats/large release 完整 | ggml context + Folly Arena；推理与通用 `MUST` |
| `MEM-A-10` | caller-owned workspace/stack buffer 做零 backing allocation 的 bump slice | external lifetime、fixed hard capacity、alignment/OOM、bulk clear、copy/move forbidden | `FixedRegion(span<byte>)` small value | `COMPLETE`：external bounded bump route 完整 | embedded/kernel workspace + oneTBB fixed storage；`MUST` |
| `MEM-A-11` | nested scratch 按 LIFO mark/rewind，越过 scope 后批量回退 | mark identity/order、alignment、OOM、rewind invalidation、external backing | `StackAllocator` over caller storage，mark token显式 | `COMPLETE`：fixed LIFO allocator 已有 | parser/kernel nested scratch；`GENERAL` |
| `MEM-A-12` | 从 hard shared block pool 借一块作为 owner-local fixed-capacity region | pool exhausted、one block ownership、bulk clear、return-on-destroy、pool outlives region | `PooledRegion` 是借 block 的 bump owner，不与 owned Region 同名 | `COMPLETE`：borrow/clear/return lifecycle 已有 | HYLE pool composition；`GENERAL` |
| `MEM-A-13` | embedded/realtime caller 在固定 payload buffer 内分配不同大小并 individual free/realloc 或 bulk reset | external storage、variable size/alignment、split/coalesce/fragmentation、failed realloc preserves old、quiescent reset invalidates all | `FixedPool`/更直白的 external variable-pool name；显式 ~4 KiB control 与 hard capacity | `COMPLETE`（机制）：two-level bitmap/TLSF-like O(1) fit、free/coalesce/realloc/reset 完整；taxonomy 待修 | oneTBB fixed_pool + normal external-workspace journey；`MUST` |
| `MEM-A-14` | 多线程共享固定大小 block，容量耗尽必须严格可观察，remote free 立即可见 | block size/capacity、hard bound、sharding、ABA/version saturation、foreign/double free、reset quiescence | concurrent fixed-block core `SharedPool`，constructor failure 用 `Result<Owner>` 而不是半有效 object | `PARTIAL`：hard algorithm/remote free 完整，当前 post-construction `valid()` 与名称不统一 | marl/Folly pool + HYLE handoff journey；`MUST` taxonomy/fallible-construction trigger |
| `MEM-A-15` | 每 worker 在 hard shared pool 前放 owner-local free-slot cache | explicit cache capacity/batch/storage、single owner、publish/destructor visibility、soft capacity、core outlives cache | `SharedPoolCache` attached owner；caller storage或显式 metadata allocator | `COMPLETE`（机制）：local zero-atomic hit 与 publish route完整 | mimalloc local/remote split + HYLE scheduler-shaped use；`GENERAL` |
| `MEM-A-16` | 固定容量 typed object churn，每次 borrow 构造、return 析构 | ctor args/throw rollback、hard core/cache、live object teardown、foreign/double return | `ObjectPool<T>` + optional `ObjectPoolCache<T>`，hot try returns nullable/loan | `COMPLETE`：Reconstruct typed shell/core/cache 完整 | marl/object pools + HYLE C++ journey；`MUST` |
| `MEM-A-17` | codec/context 等昂贵 object 在 borrow 之间保留内部 state | construct all/failure rollback、try borrow/return、不析构 on return、pool teardown、state reset由用户 | 独立 `PreservingObjectPool<T>`，不与 slot cache policy 混名 | `COMPLETE`：preserve policy 与 lifecycle 已有 | Folly expensive context pool + marl policy；`GENERAL` |
| `MEM-A-18` | thread/fiber-local typed object pool 容量未知、按 block增长且无需原子 | geometric growth、free-list reuse、ctor/dtor、trim/reset、move、live objects、allocator injection | `GrowablePool<T>` single-owner；若需要 arbitrary bytes 用 owned Heap 而非模板 pool | `COMPLETE`（typed mechanism）：grow/reuse/release完整，命名仍纳入 taxonomy | clay/mature pool pattern；`GENERAL` |
| `MEM-A-19` | 早退/异常时保证 pool object 归还 | move-only、empty、release/reset、owner outlives loan、Reconstruct/Preserve destroy route | `PoolLoan<T,Owner>` returned by `try_loan` | `COMPLETE`：typed RAII return 已有 | C++ usability + marl loan；`MUST` |
| `MEM-A-20` | owner 对 requested live bytes 做严格 admission，跨线程不越 quota | atomic reserve/rollback、size overflow、realloc delta、peak/remaining、backing OOM、borrow lifetime | `QuotaAllocator` owner exposes borrowed allocator ref；hot reject null | `COMPLETE`：hard byte limit 与 statistics 完整 | server admission + oneTBB/allocator practice；`GENERAL` |
| `MEM-A-21` | opt-in owner 观测 alloc/free/realloc、peak、known leaks，而不污染默认 heap hot path | atomic snapshot一致性边界、metadata allocator/OOM+dropped、reset quiescence、stack traces另归 Obs | `StatsAllocator` + bounded `LeakTracker`/snapshot；不完整 inventory 可见 | `COMPLETE`：counters/leak map/dropped/clear contracts 完整 | Folly/mimalloc diagnostics + HYLE owner journey；`GENERAL` |
| `MEM-A-22` | runtime 读取 allocator reserved/committed 与 process RSS/commit，触发 trim/pressure policy | current/peak、allocator vs process、concurrent snapshot、resource capacity/free/used/reserved、owner trim | system cold snapshot + per-resource stats/trim interface；Platform提供 host OS facts，Mem聚合 storage view | `PARTIAL`：`system_memory_snapshot` 和 system collect 有，device/custom resource stats/trim 不统一 | llama backend memory fitting + CCCL/ggml pool stats；推理 `MUST`，已在 MEM2-06 |
| `MEM-A-23` | 大 host span 选择 THP/prefault/NUMA preferred/bind/selected interleave | request vs accepted、single/selected/all nodes、chunk order/weight、hard/soft placement、fallback、whole release | Mem `HugeRegion` policy owner over Platform `VmRegion` mechanism | `COMPLETE`（现有 host journey）：THP/prefault/NUMA/1GiB reserve完整；owner重叠待 taxonomy 收敛 | ggml NUMA + Folly/oneTBB；推理 `MUST` |
| `MEM-A-24` | fiber runtime 获取 guarded stacks，并在 worker-local/remote release 后 bounded retain/trim | size tiers、guard、Result failure、cross-worker release、capacity 0、trim/clear、shutdown quiescence | `FiberStack` owner + `FiberStackPool`; scheduling policy仍在 Sched | `COMPLETE`：guarded VM、pool、remote release、trim完整 | marl/Folly fibers + HYLE Sched production；runtime `MUST` |
| `MEM-A-25` | model weights/data 用 anonymous/file/named mapping owner，统一 range/view与 residency | reserve/commit/protect、RO/RW/COW、subrange、advice/populate、lock prefix/range、sparse release、flush、move | Mem mapping family owning semantics；Platform/IO提供 syscall/namespace mechanism | `PARTIAL`：Platform `VmRegion/MappedFile`、IO `SharedMemory` 分裂，Mem only HugeRegion；populate/fragment edge缺失 | llama mmap/mlock/lazy fragments + Folly/Boost mapping；推理 `MUST`，MEM2-03 |
| `MEM-A-26` | host、pinned、device、managed/unified、registered 或 custom storage 用同一选择/查询 vocabulary | open provider、domain/device identity、host/device accessibility、coherence、alignment/max/capacity、allocation properties | small property-bearing `MemoryResourceRef` + owning provider elsewhere；不能用闭合 `BufferKind` enum | `MISSING`：Mem 只有 host Allocator，Compute 只有局部二值 resource | ggml buffer type + CCCL memory-resource model；推理 `MUST`，MEM2-04/05 |
| `MEM-A-27` | persistent weights/KV/output 或 transient scratch 获得 move-only byte owner与零成本 subview | size/alignment/resource identity、const/mutable view、O(1) subview、host mapping optional、move/reset、in-flight lifetime | `Buffer` owner + `BufferView/ConstBufferView`; view 不分配、不拥有 | `PARTIAL`：Compute `Buffer` 有 owner/host pointer，但 Mem 无公共 view/resource model | ggml backend buffer/tensor views + CCCL buffer；推理 `MUST`，MEM2-04 |
| `MEM-A-28` | 外部 pointer/native allocation 被安全 adopt/borrow/import，或一个 logical buffer 跨多个 physical resources | borrowed vs adopted deleter、provider pin、offset/alignment、composite segment order、partial failure/teardown | explicit `import/adopt` factory and segmented/composite owner/view；不把 raw handle塞进普通 constructor | `MISSING`：Compute 能暴露 native handle但没有通用 import/composite | ggml buffer-from-host-ptr/meta multi-buffer + Boost.Compute；推理 `MUST` |
| `MEM-A-29` | caller 对 buffer 做 read/write/fill/copy/map 与非 coherent flush/invalidate | sync range bounds、same/cross resource、overlap、host map lifetime、coherence、failure/partial、no hidden staging | sync operation在 Mem resource/buffer seam；async ordered extension接受 Compute Stream/Event而不让 Mem反向依赖 | `PARTIAL`：Compute Buffer 有局部 sync methods，跨 provider/composite/managed route不统一 | ggml buffer interface + CCCL async resource；推理 `MUST`，MEM2-04/05 |
| `MEM-A-30` | transient buffers/blocks 在同一 resource 上复用，压力点可 trim并观测 retained bytes | arbitrary/LIFO/best-fit、release threshold、size/alignment key、stream safety、capacity stats、OOM、owner shutdown | `PooledResource/BufferPool` semantics over injected resource；provider可用driver pool实现而不改公共合同 | `PARTIAL`：host多种 pool与 Compute `DevicePool` 分裂，无跨 resource shape | ggml CUDA/SYCL/Vulkan pools + CCCL pools；推理 `MUST`，MEM2-06 |
| `MEM-A-31` | graph/engine 给出 size/alignment/live interval/alias约束，先 measure peak/reserve再复用 buffer | multiple resources、persistent/output、in-place/alias groups、determinism、plan/instantiate split、overflow/OOM | tensor-free `LifetimePlanner` requests -> placements；Graph只把 DAG 翻译为事实 | `MISSING`：没有通用 planner，unfinished Graph自有局部 temporary logic | ggml gallocr + oneDNN planner；推理 `MUST`，MEM2-07 |
| `MEM-A-32` | lock-free/read-mostly container 删除节点/旧 table 后，在最后 reader 离开时安全回收 | hazard pointer、epoch/RCU/QSBR tradeoff、thread/domain registration、retire threshold、deleter/allocator、thread exit、bounded grace period | 独立 reclamation domain + guard/token + `retire`; Container选择机制但不各自私藏无法回收 list | `MISSING`：ConcurrentNodeHashMap 把 retired tables留到析构，Obs另有私有 reader epoch | Folly Hazptr/RCU + HYLE两个直接消费者；并发基础 `GENERAL`，是现有 MEM2 表遗漏的跨域候选 |
| `MEM-A-33` | secret/key/model-license material 要求不可优化掉的清零、锁页或 dump policy | secure erase、compiler optimization、swap/dump、move/copy、exception、platform support | 安全 owner组合标准/OS secure-zero与locked mapping；不让普通 Buffer 默认支付清零/锁页 | `OUT_OF_SCOPE`（当前 Mem） | 无当前具名 HYLE/llama secret-storage consumer；安全任务出现时单独准入，不因“memory”一词扩面 |
| `MEM-A-34` | tensor/KV/model placement、eviction/defrag/offload/paging 决定“哪块数据放哪里多久” | dtype/layout/model semantics、sequence state、quality/latency policy、backend scheduling | Engine/Graph policy产生 Mem resource/lifetime requests；Mem不包含 Tensor/Model/Stream | `OUT_OF_SCOPE`：只保留 backing/view/planner边界 | llama KV/model loader；推理 `MUST` journey但 owner明确不是 Mem |
| `MEM-A-35` | source/static/shared C++ 用户选择 Mem，而 provider adapters仍保持无环 | mimalloc source ownership、visibility、resource callback location、selected components、header/ODR、三交付 relocation | `hyle::mem` semantic core；Compute/IO adapters向下实现 provider，不让 Mem link上层；无伪 C ABI | `PARTIAL`：mem-v1 delivery完整，v2跨域 owner/provider seam尚未实现/验证 | build/package contract + MEM2-08/09；重构交付 `MUST` |

public-surface 反向追踪覆盖 17 个 `<hyle/mem/*.h>`、`hyle::mem` target、当前
Platform `VmRegion/MappedFile`、IO `SharedMemory`、Compute `Buffer/BufferType/DevicePool`、Graph allocator
逻辑和所有 allocator consumers。第一遍得到六个关键结论：

1. mem-v1 的 host allocator、arena、external variable pool、hard/shared/typed/growable pool、quota/stats、
   HugeRegion 与 fiber stack不是“都没用”；这些机制相当完整，主要问题是 taxonomy/fallible owner形状；
2. 推理 storage lifetime 不能按 syscall/provider 所在文件夹分裂。Resource/Buffer/View/import/composite、
   mapping、copy/coherence、resource pool与planner仍是实质 `MISSING/PARTIAL`，因此 mem-v2 trigger成立；
3. `Platform-A-09/29` 的 host capacity、populate/fragment journey在 Mem 中分别落到 resource capacity与
   mapping owner；Platform继续只提供可移植机制，不能各自维护两套用户 owner；
4. device/pinned/managed 语义进入 Mem不意味着 Mem依赖 Compute。上层 provider实现向下满足 Mem callback/
   concept，Device/Stream/Event和async ordering仍归 Compute；
5. concurrent reclamation 是现有 mem-v2 capability map漏掉的成熟基础能力：HYLE concurrent container与
   Obs都已有私有/延迟回收问题，B阶段必须裁决 Hazptr/epoch factor属于 Mem、Util还是Container；
6. 现有 MEM2-01～09仍是候选输入，但在全局 A/B/C 完成后必须根据上述 reclamation 与 Platform/IO/
   Container交叉结果显式更新 active card，不能直接恢复旧 FIX 集施工。

这表示 Mem 产品域的阶段 A 第一遍已经完成；`mem-v2` 保持 `AUDITED` 且暂停产品实现。最终
PRODUCT_TRIGGER/owner/FIX 集只在 13 域跨查和因子化完成后重新冻结。

#### A8. Obs 产品域（能力账本已随 A 阶段冻结）

Obs 的产品边界不是一套应用 logger，也不是把 profiler、metrics backend 和 crash reporter 全部绑成一个
service。它拥有低成本 observation 事件、禁用/采样控制、同步 host-adapter seam，以及可由最终 image
选择的 profiling seam；队列、持久化、网络和业务 schema 仍由具有相应生命周期的 owner 负责。直接推理
证据来自 llama.cpp/ggml 的 process log callback、server slot/progress/timing、context/sampler perf snapshot、
Prometheus metrics 和 backend profiling；仓内反向追踪覆盖 Mem allocation、Sched fiber/timer、IO/Compute
completion trace。strong peers 锁定为 Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7`、Folly
`23ebefdf96c116700d3e134f04d95a2b60a509de`，profiling mechanism 为 Tracy
`30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`；每个能力族至多直接使用两个 peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `OBS-A-01` | library call site 按严重度记录诊断，关闭时不计算昂贵参数 | compile-time floor、process runtime threshold、invalid level、并发读写、OFF 无求值 | `HYLE_LOG_<LEVEL>` + `should_log/set_level`；宏只捕获 caller source | `COMPLETE`：五级门控、compile floor 和 no-evaluation 已覆盖 | Abseil log + llama/ggml macros；基础 `MUST` |
| `OBS-A-02` | 已格式化 bytes、source、时间和少量字段作为一个同步 event 交给宿主 | embedded NUL、UTF-8/任意 bytes、borrow lifetime、可选 source/time、空字段、字段顺序 | borrowed `LogRecord` + span fields；同步返回前有效 | `PARTIAL`：message/source/time 与 string fields 完整；缺 event/category identity、typed value 和 correlation attributes | llama callback + Folly `LogMessage`; logger-neutral structured seam `MUST`，具体字段待 B 阶段去重 |
| `OBS-A-03` | 普通 C++ caller 类型安全格式化诊断且日志失败不改变业务流 | format exception、OOM、oversize/truncation、source location、disabled args | `std::format_string` convenience 写 bounded stack buffer；owning formatting 留给 sink | `COMPLETE`：1 KiB bounded format、marker 和 failure record 已有 | Abseil/Folly call-site logging；易用性 `MUST` |
| `OBS-A-04` | 高频 warning/progress 只记录条件、前 N、每 N、2 的幂或时间间隔 | approximate/exact contention、N<=0、reset/move、call-site static 与 per-request object、注入 monotonic time | 宏控制常见 call site；`LogIntervalLimiter` 等显式 state 服务 request/slot | `COMPLETE`：全部控制族及精确/近似语义齐全 | Abseil/Folly rate controls + llama slot progress；推理 `MUST` |
| `OBS-A-05` | 一个库进程把同步 record 接入宿主 logger，启动/关闭时安全注册和摘除 | callback 并发、reentry、self/cross remove、quiescence、flush、duplicate/full、state lifetime | borrowed `LogSink` + move-only `SinkRegistration`；一个正常 host adapter | `COMPLETE`：固定 64 slot、无分配 emit、immutable snapshot 与 quiescent RAII remove | Abseil/Folly sink registry；基础 `MUST` |
| `OBS-A-06` | CLI/tool 没装宿主 logger 时仍得到安全的 stderr 诊断 | TTY color、Windows debugger mirror、控制字符/换行注入、concurrent write、显式 takeover/restore | process default sink，可幂等启停；不是应用配置树 | `COMPLETE`：转义、串行输出、颜色和 takeover 路径齐全 | llama default callback + Abseil stderr；工具 `GENERAL` |
| `OBS-A-07` | application 需要 async writer、bounded queue、drop/block policy、file rotation 或 remote upload | record deep copy、backpressure、ordering、flush/ack、shutdown、disk/network failure、fork | 宿主 sink callback 内复制后进入应用/IO owner；Obs core 不暗建线程 | `ROUTED`：同步 adapter 合同完整，队列和 persistence 明确外置 | llama `common_log` worker + Folly handlers；能力 `GENERAL`，owner 为 application/IO，不是基础 emit seam |
| `OBS-A-08` | 大应用按 module/category 与 verbosity 局部过滤，同时仍统一进入一个 sink | hierarchical category、numeric verbosity、dynamic config、unknown category、DSO unload、disabled cost | stable category/event name 作为 record metadata；filter/config tree 可由宿主 adapter 拥有 | `PARTIAL`：severity 和显式 `_IF` 可表达简单情况；record 无 category/event identity，也无 per-category gate | Folly categories + llama verbosity；元数据 `GENERAL`，配置树不自动准入 HYLE |
| `OBS-A-09` | tests 临时捕获并断言 library diagnostics，不接触全局 stderr | concurrent producers、scope teardown、nested capture、exact bytes/fields、no dangling state | 普通 typed sink + RAII registration；专用 mock 只在重复痛点出现时提供 | `COMPLETE`：现有 adapter 足以自然实现 capture，仓内测试已使用 | Abseil mock log 用于 journey；不复制专用 API 的拒绝性结论 |
| `OBS-A-10` | profiling build 标记同步 CPU scope 并附 name/text/value/color | compile OFF 零求值、dynamic text bound、nesting、exception return、thread migration 限制 | `HYLE_TRACE_ZONE*` lexical macros；最终 image 选择 backend | `COMPLETE`：Tracy zone 与 bounded annotations 已覆盖 | Tracy + HYLE IO/Compute hot zones；profiling `MUST` |
| `OBS-A-11` | runtime 时间线记录 plot/counter、message、frame、thread、fiber 和 allocation lifecycle | integer width、literal/dynamic text、alloc/free pairing、fiber migration、frame naming、OFF no-eval | 语义命名的 trace macros，不给普通业务对象增加 metadata | `COMPLETE`：全部列举 seam 已有且 Mem/Sched/IO/Compute 真实消费 | Tracy + HYLE consumers；推理 profiling `MUST` |
| `OBS-A-12` | IO request、task 或 backend launch 在不同线程开始/结束并在 timeline 上保留因果 flow | 64-bit id、begin/step/end、reuse/overflow、cancel/error、fan-out/fan-in、disabled path | cheap explicit flow token/events；不把 lexical zone 跨线程析构 | `MISSING`：fiber identity 只覆盖 stackful migration，没有普通 async flow seam | Tracy/Perfetto async journey + HYLE IO/Sched/Compute；推理 runtime `GENERAL` |
| `OBS-A-13` | GPU command、stream/event 与 host launch 在统一 profile 中关联 | backend context、device/queue、timestamp calibration、command lifetime、multi-device、trace OFF | Compute-owned typed GPU tracing adapter，向 Obs profiling seam 发事件 | `ROUTED`：CPU launch/reap zone 已有；无 HYLE GPU zone/context seam | Tracy GPU + ggml backend profiling；推理 `MUST`，实现 owner 为 Compute |
| `OBS-A-14` | scheduler/container 调查 mutex wait/hold、shared lock 和 contention | lock lifetime、try/shared modes、recursive lock、owner thread、OFF layout/code cost | opt-in lock annotation/wrapper，只对真实 lock owner；不替换同步原语 | `MISSING`：Tracy backend有 lock API，HYLE seam未暴露 | Tracy lock tracing + HYLE Sched/Container；诊断 `GENERAL`，待 consumer 反查 |
| `OBS-A-15` | profiler 用户选择 callstack depth、sampling、on-demand/connected 和 plot presentation | build-wide consistency、runtime connection、sampling permission、overhead、static/shared single client | build/package options + 极窄 query/control；高级 viewer policy可直接归 backend | `PARTIAL`：trace ON/OFF 和 single-client交付完整；无 sampling/connectivity/callstack seam | Tracy controls；高级 profiling `GENERAL`，不能把所有 vendor knobs 复制进 HYLE |
| `OBS-A-16` | inference/runtime 累计可查询 counter 与 gauge，而不是只向在线 profiler 发瞬时 plot | relaxed update、signed/unsigned、set/add、snapshot consistency、reset epoch、overflow、cache-line placement | owner-held typed `Counter/Gauge` primitive；无默认全局注册和字符串查找 | `MISSING`：`HYLE_TRACE_COUNTER` 不可查询且 trace OFF 消失 | llama server metrics + Folly counters；推理服务 `MUST` 候选 |
| `OBS-A-17` | token latency、queue wait、kernel duration 和 size distribution 需要 count/sum/min/max/histogram | units、bucket layout、merge、overflow、concurrent record/snapshot、reset/window、percentile误差 | unit-aware owner-held accumulator/histogram，可 merge snapshot；hot record 不分配 | `MISSING`：各模块可手写 atomics，Obs 无可复用 distribution primitive | llama perf/server buckets + Folly histograms；推理 `MUST` 候选 |
| `OBS-A-18` | 多 worker/slot 把 local metrics 合并为 process snapshot，避免每 event 全局原子争用 | sharding、thread exit、epoch/reset、merge consistency、bounded cardinality、allocator failure | explicit local shard/merge owner；registry 仅登记静态 descriptor 时才存在 | `MISSING`：无 metrics storage/merge contract | Folly timeseries + llama per-slot/global metrics；性能基础 `GENERAL` |
| `OBS-A-19` | metric 带 model/backend/device/status 等低基数 labels 并导出 JSON/Prometheus/OTLP | label lifetime、cardinality bound、escaping、schema/version、pull/push、backpressure、security | core 提供 typed snapshot/visitor；具体 exposition/exporter 在 application/IO adapter | `ROUTED`：低层 metric primitives 缺失；export 本身正确归宿主 | llama Prometheus endpoint + telemetry practice；snapshot `GENERAL`，网络 backend 非 Obs core |
| `OBS-A-20` | context/sampler/graph owner 返回稳定 perf snapshot，并可 reset/print | owner lifecycle、unit、count=0、concurrent query、reset race、derived rate、ABI/version | owner-specific stats struct built from common primitives；格式化/打印在 user boundary | `ROUTED`：Time 可计时，但当前模块各自决定 stats；Obs 无公共 primitives | llama `llama_perf_*` + ggml node perf fields；推理 `MUST` journey，owner保持在 Compute/Graph/Sched |
| `OBS-A-21` | request、trace、model 和 slot identity 随 task/fiber/IO callback 传播并自动进入 log/trace | explicit parent、TLS/fiber migration、async handoff、nested restore、sampling、ID generation、privacy | small explicit observation context + scoped attach；Sched负责捕获/恢复，record只借用属性 | `MISSING`：现有 log/trace 没有 correlation context | Folly `RequestContext` + llama slot tags；推理服务 `GENERAL`，需与 Sched/IO 因子化 |
| `OBS-A-22` | telemetry snapshot读取 RSS、host/device memory、queue depth、worker utilization | query cost/failure、sampling cadence、units、NUMA/device、stale value、privilege | Platform/Mem/Compute/Sched提供值，Obs metrics只记录/导出；不重复 OS 探测 | `ROUTED`：各 owner 能力分别待其域裁决 | llama system/backend metrics + HYLE owner boundaries；推理 `GENERAL` |
| `OBS-A-23` | invariant/fatal/terminate 路径在 abort 前输出 bounded context，且不递归或分配失控 | signal safety、recursive failure、OOM、other threads、stacktrace/minidump、observer lifetime、privacy | Core保留 fail-fast；可选 process-lifetime bounded crash observer 由 Obs/Platform 提供 | `PARTIAL`：Core fatal直接写 stderr，普通 logger非 signal-safe；没有统一 bounded observer/crash artifact | Abseil failure signal + HYLE Core/Error trace；基础诊断 `GENERAL`，是跨域候选而非日志宏扩张 |
| `OBS-A-24` | Status/native failure 被记录时保留 code/domain/message，而不是先拼成不可解析文本 | OK policy、message bytes、native domain/code、source、redaction、allocation failure | explicit `LogField`/formatter adapter 或 structured error event；logging仍无控制流副作用 | `PARTIAL`：Error有完整访问器/`to_string`，Obs没有 typed adapter且字段只能 string | HYLE Error + Abseil status logging journey；内部易用性 `GENERAL` |
| `OBS-A-25` | secret/token/prompt/PII 不应因默认 telemetry 被意外记录 | classification、redaction/hash、structured field、binary payload、debug override、sink trust boundary | caller/schema显式选择可记录字段；sink可执行 deployment policy；不做猜测式自动扫描 | `ROUTED`：默认 sink只做控制字符安全转义，不承诺 secret redaction | server security journey；策略归 consumer/sink，Obs应记录合同与安全坑 |
| `OBS-A-26` | source/static/shared C++ 用户获得相同 log/trace seam，关闭 trace 不携带 client | selected target、visibility、header macro一致性、single Tracy client、final image、无伪 C ABI | `hyle::obs` + optional trace build feature；只公开 C++ surface | `COMPLETE`：三交付和 trace OFF/ON 已验证 | build/package contract；交付 `MUST` |

public-surface 反向追踪覆盖三个 public header、`src/obs/log.cc`、全部仓内 trace consumer、现行 architecture/
user guide 与 `obs-v3` 证据。第一遍得到六个关键结论：

1. 同步 library logging 的门控、格式化、rate control、host adapter、quiescent lifecycle 与默认 CLI sink
   已经闭环，不能因 llama 有异步 worker 就把应用 logger 重新塞进 Obs；
2. 当前 `LogRecord` 的 string-only fields 足以承载旧能力，但对 category/event identity、typed value、Status
   和 correlation 的自然形状只是 `PARTIAL`，B阶段需决定一个最小 attribute vocabulary；
3. Tracy CPU/fiber/memory/counter seam 已有真实仓内消费；普通 async flow、lock contention 和 GPU tracing仍需
   分别与 Sched/IO、Container和Compute owner组合，不能靠跨线程析构 lexical zone；
4. profiler plot 不是可查询 metrics。llama server/perf 提供了直接 consumer，counter/gauge、distribution与
   shard/merge snapshot 是实质缺口；网络 exporter 与应用 registry不因此进入低层 hot path；
5. Core fatal、Error structured status 与 Obs crash/diagnostic bridge仍有跨域空隙，但 signal-safe crash path不能
   重用任意同步 sink graph；
6. Obs 私有双 snapshot reader epoch 是固定 registry 的合法实现，同时也是 `MEM-A-32` reclamation 因子输入；
   A阶段只记录重复机制，不提前替换已经关闭的实现。

这表示 Obs 产品域的阶段 A 第一遍已经完成；当前 `CLOSED` 模块不重开。metrics、async correlation、typed
attributes、GPU/lock trace 与 crash observer 只有在 13 域跨查并确定 owner 后才形成有限
`PRODUCT_TRIGGER`。

#### A9. SIMD + Math 产品域（能力账本已随 A 阶段冻结）

本节保持绑定例外：SIMD 是一个不可拆分的公开产品，不按 Highway、xsimd、HYLE body、ISA 或算法族拆成
多个用户组件。其两层产品面分别是普通 pointer/count/POD whole-buffer API，以及在 consumer TU 中原样使用
锁定 Highway/xsimd 的 expert SDK；387 operations / 422 route scopes 是旧 HYLE register surface 的迁移账本，
不是两家 provider 的能力上限。Math 则单独核对为极小 scalar numeric 组件，不能成为 SIMD scalar fallback
或杂项数学伞头。直接推理证据来自 ggml CPU 的 row algebra、dtype、quant/dequant、mixed dot、GEMV/GEMM、
activation/norm/softmax、sampling top-k 和多 ISA kernels。strong implementation peers 锁定 Highway
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`、xsimd
`3b3151ff665514a00a437cc985f47b7de99b772b`；SLEEF
`7623d6cfa2712462880fa63a4d0f0b5f775d1a83` 只用于 transcendental discovery，不是当前产品依赖。

##### A9.1 SIMD capability-family map

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `SIMD-A-01` | 普通用户对连续 buffer 调用一次语义明确的加速 operation | null-on-zero、alias、tail、NaN/overflow/rounding、allocation、dispatch frequency逐族声明 | pointer/count core + 少量 span convenience；vendor type 不越过边界 | `COMPLETE`：13 个 ordinary headers覆盖现有 row/math/search/scan/order/codec/RNG族 | HYLE public surface + ggml row journey；产品形状 `MUST` |
| `SIMD-A-02` | kernel 作者需要 HYLE 未包装的 register/mask/load/store/arithmetic/permute/reduction primitive | lane type、fixed/scalable vector、mask、tail、alignment、target namespace、升级重编 | 显式链接 `hyle::simd_expert` 后原样使用 `hwy::` / `xsimd::` | `COMPLETE`：两家锁定版本完整 header-only surface 在三种交付可用；422 不限制它 | Highway+xsimd；广面能力 `MUST`，拒绝建立第二套 HYLE register DSL |
| `SIMD-A-03` | 已知部署 ISA 的单个 custom kernel 要模板/宏单态化并完全内联 | scalar/SSE4/AVX2/AVX512/NEON/SVE/RVV、compiler flags、exact fail-closed、provider native type只在 TU | 独立 OBJECT target + `hyle_target_simd()` | `COMPLETE`：source/static/shared 同一 helper；classic cl.exe AVX512 明确拒绝 | Highway target model + HYLE delivery；专家路径 `MUST` |
| `SIMD-A-04` | 未知客户 CPU 的 custom kernel 需要一个 binary 内含多 ISA 且只在整批边界选择 | canonical target set、symbol suffix、feature/OS state、exact/fat、cached selector、no per-lane callback | pointer/count/POD entry + `hyle_add_fat_kernel` / select macro | `COMPLETE`：每 target TU 单态化，默认一次 cached indirect，exact 零 selector | Highway dispatch + HYLE helper；推理 kernel `MUST` |
| `SIMD-A-05` | 普通 API 在 x86/ARM/RISC-V 选择安全 broad target并能报告实际 family route | compile/runtime分离、OS vector state、scalar floor、per-family lower tier、override/diagnostic | 私有 Runtime table；公开只给 `SelectedTarget` diagnostics | `COMPLETE`：scalar、SSE4、AVX2、AVX512、NEON、SVE、RVV 与 family facts 齐全 | Highway + Platform facts；基础 `MUST` |
| `SIMD-A-06` | specialty inference kernel 使用 VNNI、BF16、FP16/FP8、SVE2/dotprod/i8mm、AMX/SME | CPU+OS+per-thread enable、compiler support、feature orthogonality、fallback、matrix state restore | Platform 提供 facts/state；SIMD/Compute kernel单独 feature-gated，不抬高 broad tier | `PARTIAL`：F16C、AVX/AVX512-VNNI、AVX512-BF16、AVX10.2已有；缺 ARM dotprod/i8mm/SVE2/SME 和 AMX | ggml ISA kernels + Highway targets；推理 `MUST` 候选，与 `PLAT-A-02` 联查 |
| `SIMD-A-07` | f16、bf16、E4M3、E5M2 作为稳定 storage value 跨 buffer/API 保存 | layout/size/triviality、RNE、subnormal、NaN/Inf、saturate、signed zero | HYLE POD bit carrier + scalar conversions | `COMPLETE`：四种窄 storage 与 f32 scalar conversion合同完整 | ggml dtypes + Highway/CUDA semantics；推理 `MUST` |
| `SIMD-A-08` | 五种公开 floating storage 任意异类型批量转换，不建立 f32 临时数组 | 20 directed edges、same-width exact alias、unequal disjoint、rounding/special values、target feature | overloaded `convert(src,dst,n)` | `COMPLETE`：全转换图，F16C/BF16/AVX10.2 specialty与 portable fallback已有 | Highway + inference storage journey；推理 `MUST` |
| `SIMD-A-09` | 普通连续 f32/f64 row 做 elementwise algebra 和 scalar transform | add/sub/mul/div、scale/bias/affine、abs/neg/square/sqrt、exact alias、NaN/zero | 语义函数而非 runtime functor；一次 whole-buffer dispatch | `PARTIAL`：列举族完整；常用 min/max/clamp、FMA/mul-add、reciprocal/rsqrt、fill/copy/compare-select无 ordinary 入口 | ggml vec + Highway/xsimd；`GENERAL`，需由 Container/Compute真实重复决定最小扩面 |
| `SIMD-A-10` | row 计算 sum、dot、min/max/argmin/argmax 与组合统计 | empty、NaN、first tie、accumulator width、reproducibility、one-pass、overflow | 分族 typed reduction；组合统计只在减少真实遍历时准入 | `PARTIAL`：sum、dot、extrema/index/minmax完整；缺 sum-of-squares/L1/L2、mean/variance 等组合 reduction | ggml norm/cvar + Highway reductions；推理 `GENERAL` 候选 |
| `SIMD-A-11` | storage/mixed precision dot 不先全量 convert | f32/f64、bf16×bf16、f32×bf16对称、f16、i8/i16；accumulation与overflow显式 | overload by storage types；target feature specialization私有 | `PARTIAL`：上述常用族完整；没有 fp8 dot、u8×i8 或 block-quant dot | ggml mixed/quant dot；基础浮点 `COMPLETE`，新推理 dtype为 `MUST` 跨域候选 |
| `SIMD-A-12` | model weights 使用 Q/IQ/TQ/MXFP/NVFP 等 block format 做 quantize/dequantize | block layout/version、scale/min/zero、rounding、tail、reference codec、alignment、quality policy | format-owning quant component + row codec；不能把 opaque block塞入通用 fp8 carrier | `ROUTED`：SIMD只有通用 bitpack/fp8；HYLE 当前无 block-quant产品 | ggml quants 是直接 consumer；推理 `MUST`，最终 owner在 SIMD/Compute/tensor-format 因子阶段裁决 |
| `SIMD-A-13` | quantized weights 与 activation直接 row dot/GEMV/GEMM | format pair、tile layout、K multiple、packing、ISA/matrix extension、accumulator、thread partition | format-specific kernel table由 Compute 调度，SIMD提供 expert primitive/target substrate | `ROUTED`：expert可写 kernel，但没有 HYLE quant kernel family | ggml CPU quant kernels；推理 `MUST`，不能把矩阵/tensor policy倒灌普通 row API |
| `SIMD-A-14` | 常用 f32/f64 transcendental 作用于整段并明确 accuracy/domain | exp/log/trig/hyperbolic/gamma/error/pow/atan2/hypot/remainder/nextafter、special values、alias | `<name>_f32/f64_batch`; safe ordinary 与 caller-proven in-range入口分离 | `COMPLETE`：real unary/binary/sincos及窄合同覆盖广，scalar oracle兜底 | Highway/xsimd/SLEEF discovery；通用数值 `GENERAL` |
| `SIMD-A-15` | complex SoA buffer 做常见 elementary/transcendental 运算 | f32/f64、branch cuts、NaN/Inf、exact in-place、两个输出 alias约束 | separate real/imag pointer batches，不固化 `std::complex` ABI/layout | `COMPLETE`：exp/log/sqrt/trig/hyperbolic/pow齐全 | xsimd/Highway math +科学计算 journey；`GENERAL` |
| `SIMD-A-16` | 推理 row 运行 GELU、sigmoid、SiLU、ReLU family | exact/approx variant、model参数、NaN、f32/f16/bf16 storage、in-place | 以数值语义命名 whole-row kernel；不能把模型默认值藏在名字后 | `PARTIAL`：f32 GELU exact/tanh、sigmoid/SiLU/ReLU/leaky/ELU/hard variants完整；缺 f16/bf16与 quick GELU | ggml activation rows；推理 `MUST` 候选 |
| `SIMD-A-17` | gated MLP 对两段输入做 ReGLU/GeGLU/SwiGLU及可选 backward | split/binary layout、gate variant、f16/f32、output alias、backward、fusion boundary | model-independent row primitive可接受两 input spans；tensor stride/thread在 Compute | `MISSING`：普通 SIMD 无 GLU family | ggml `reglu/geglu/swiglu` 是直接 consumer；推理 `MUST` 候选 |
| `SIMD-A-18` | 一行 logits 做 stable softmax/log-softmax | max-shift、NaN/+Inf/all -Inf、exact alias、empty、f32 | `src,dst,n` core + in-place convenience | `COMPLETE`：普通连续 f32 合同完整 | ggml softmax + Highway math；推理 `MUST` |
| `SIMD-A-19` | attention/logits softmax 同时应用 scale、bias/mask、causal window或 online block | mask dtype/stride、-Inf、ALiBi、rows、online combine、precision、thread/backend fusion | Compute operator/kernel；可在固定 TU组合 SIMD primitives，不能堆 ordinary tiny calls | `ROUTED`：基础 softmax已有，tensor/fused variants无 | ggml softmax/flash-attention；推理 `MUST`，owner为 Compute |
| `SIMD-A-20` | row 做 RMSNorm/LayerNorm、mean/variance 和 normalize | epsilon、f32 accumulate、weights/bias、one/two-pass、NaN、in-place、f16/bf16 I/O | model-independent reduction+transform core或 Compute fused op；必须先裁决复用粒度 | `MISSING`：只有可组合 sum/algebra，没有自然一次/两次遍历 norm API | ggml norm/rmsnorm；推理 `MUST` 候选，SIMD/Compute owner待 B 阶段 |
| `SIMD-A-21` | scalar numeric buffer 做 equality search、extrema/index 和 sortedness | i/u8..64、f32/f64、empty、first tie、NaN、asc/desc、single-pass minmax | typed pointer/count family | `COMPLETE`：十类型与全部列举操作齐全 | Highway contrib + sampling/index journeys；基础 `GENERAL` |
| `SIMD-A-22` | f16/bf16 buffer 做 numeric/bit equality、f32 sum 与 IEEE total order | signed zero、NaN payload、encoding equality、reversible key、asc/desc | narrow-specific whole-buffer API + constexpr key transform | `COMPLETE`：两种窄类型合同齐全 | Highway narrow + inference storage inspection；`GENERAL` |
| `SIMD-A-23` | contiguous numeric buffer 做 inclusive/exclusive prefix sum | i/u32/64、f32/f64、init、exact alias、integer modulo、floating association | const input/output core + in-place convenience | `COMPLETE`：六类型、两 scan语义完整 | Highway scan + offset construction；`GENERAL` |
| `SIMD-A-24` | segmented/keyed/custom scan、parallel scan 或 GPU scan | segment flags/keys、associativity、identity、partition combine、scratch、executor/backend | expert custom kernel、Container algorithm或 Compute operator按 owner承担 | `ROUTED`：基础 sum scan完整，无万能 runtime callback | Highway/oneDPL style + graph journeys；按真实 owner准入，拒绝逐元素 callback |
| `SIMD-A-25` | numeric keys 做 full sort、partial sorted prefix 和 nth selection | i/u16..64、f32/f64、u128、key-value、asc/desc、NaN、O(1) hidden workspace | typed in-place operations | `COMPLETE`：11 类 key/POD 和三 operation齐全 | Highway VQSort + HYLE semantics；基础 `GENERAL` |
| `SIMD-A-26` | 需要稳定 numeric order或保留原 keys输出 permutation | integer/float order、NaN-last、signed zero/payload stability、caller scratch、OOM、32-bit index bound | explicit radix/argsort scratch core + allocating convenience | `COMPLETE`：base-256 stable radix和argsort覆盖公开 numeric types | Highway origin + sampling/data-prep journey；`GENERAL` |
| `SIMD-A-27` | sampling/beam search 只取 float score 的 top-k并保留 token index | descending score、stable/deterministic ties、NaN、k<<n、scratch、non-destructive input | score+index partial selection/sort或 indices top-k；不能强迫完整 argsort | `MISSING`：float partial_sort会丢 index，argsort必须完整排序；整数 KeyValue不能自然表达 float score | llama top-k sampling + ggml score arrays；推理 `MUST` 候选，和 Container algorithm联查 |
| `SIMD-A-28` | unsigned values 按固定位宽压缩/解压并跨 ISA得到同一 payload | u8/u16/u32、bits range、tail/padding、endianness/version、capacity、no overlap | `packed_size` + pack/unpack；framing由 owner记录 | `COMPLETE`：canonical v1 layout和全部类型齐全 | Highway bit ops + model metadata/storage；`GENERAL` |
| `SIMD-A-29` | 大批生成可复现非加密 u64/f32/f64 uniform values并划分 worker stream | seed/copy、jump/split、advance、跨 ISA位一致、thread confinement、range | mutable `RandomState` + fill/split；安全熵另归 Platform | `COMPLETE`（uniform）：xoshiro state lifecycle和三输出完整 | llama sampling RNG + HYLE Util RNG；推理 `GENERAL` |
| `SIMD-A-30` | 批量 normal/exponential/bernoulli或 categorical sampling | distribution contract、quality、determinism、tail、state consumption、alias、top-k/top-p policy | 通用数值 distribution与模型 sampling policy分开；可在 expert kernel组合 | `ROUTED`：uniform substrate已有；标准 distribution可用，categorical owner不在 SIMD基础层 | llama sampler journey；能力由 sampling/Compute owner按真实算法准入 |
| `SIMD-A-31` | hash、CRC、UTF、base64、compression、crypto 等算法内部需要 SIMD | wire semantics、security、state/streaming、error、ISA specialty、dispatch | 各语义 owner使用同一 expert/fat substrate；不把所有“用了向量”的算法迁入 SIMD | `ROUTED`：Container/IO等可直接消费锁定 provider或自己的 dependency | Highway contrib + HYLE Container；质因数是 target substrate，不是巨型 SIMD facade |
| `SIMD-A-32` | GEMM/GEMV/conv/attention/RoPE 等 tensor kernel需要 shape、stride、packing、thread/backend策略 | dtype/quant format、layout、batch、broadcast、scratch、fusion、parallelism、device | Compute operator/kernel registry消费 SIMD expert/ordinary primitives | `ROUTED`：SIMD不实现 tensor operator；Compute/Graph当前冻结且覆盖不作断言 | ggml CPU operators；推理 `MUST`，保持依赖方向 `Compute -> SIMD` |
| `SIMD-A-33` | provider升级或 HYLE新优化能逐 operation替换实现且不增加路由成本 | semantic equivalence、compiler/target/variant、material evidence、fallback、origin/license | compile-time route manifest；无法稳定分胜负默认当前 Highway | `COMPLETE`：387/387 inventory、422 scopes、HYLE winners与provider route已机械化 | Highway+xsimd/HYLE ledger；渐进强化架构 `MUST` |
| `SIMD-A-34` | source/static/shared 用户获得同一 ordinary/expert/fixed/fat 形状 | relocatable headers/helpers、offline locked sources、visibility、toolchain/ISA、no vendor binary ABI | `hyle::simd` + opt-in `hyle::simd_expert` | `COMPLETE`：三平台三交付已签署；无旧 register façade或伪 C ABI | build/package + simd-v4；交付 `MUST` |

##### A9.2 Math capability-family map

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `MATH-A-01` | 不可信 shape/stride/offset/byte count 做 checked add/sub/mul | same-type non-bool signed/unsigned、overflow无 UB、constexpr、精确成功值 | `try_add/try_subtract/try_multiply -> optional<T>` | `COMPLETE`：三基本 checked operation和 native/portable lowering完整 | CCCL/Folly + ggml shape arithmetic；基础 `MUST` |
| `MATH-A-02` | 非负 count 做 ceil division和 representable multiple round-up | divisor>0、non-negative、overflow、trusted/untrusted、constexpr | `try_*` 返回 optional；已验证 invariant convenience fail-fast | `COMPLETE`：ceil/round四入口完整 | tensor/block sizing；基础 `MUST` |
| `MATH-A-03` | mixed-width/signedness value进入统一 shape type前检查可表示性 | enum/integral、negative-to-unsigned、narrowing、bool、target type choice | 直接使用 C++23 `std::in_range` + explicit cast；复杂 conversion由owner schema处理 | `ROUTED`：标准库已自然覆盖，不复制 `checked_cast` | 标准定义性能力；拒绝无消费者 wrapper |
| `MATH-A-04` | bit ceil/floor/width、rotate/popcount、gcd/lcm、limits与普通 scalar cmath | 标准整数/浮点语义、constexpr availability、NaN/domain | 直接使用 `<bit>/<numeric>/<cmath>/<numbers>/<limits>` | `ROUTED`：C++23标准库 owner | 标准能力；Math不做改名伞头 |
| `MATH-A-05` | saturating/clamped/fixed-point或 mixed-result arithmetic | result type、promotion、rounding、saturation bound、overflow flag、constexpr | 只有具名 consumer共享同一政策时才增加窄 helper；lane版本走 SIMD expert | `OUT_OF_SCOPE`：当前无统一 HYLE scalar journey | CCCL扩展用于发现；peer API数量不构成准入 |
| `MATH-A-06` | floating approximation、tolerance、sign/dead-zone、polynomial与transcendental | abs/relative/ULP、NaN/Inf/zero、accuracy profile、domain | scalar普通用标准库；buffer用 SIMD；领域 tolerance由consumer命名 | `ROUTED`：旧含糊 `sgn/approx_eq` 已正确移除 | 标准库 + SIMD；拒绝把策略伪装通用事实 |
| `MATH-A-07` | linear algebra、FFT、geometry、statistics、quant scale或模型算子 | shape/layout、precision、scratch、backend、parallelism、domain policy | 成熟依赖或 Compute/Graph/consumer组件；Math不成为算法垃圾桶 | `OUT_OF_SCOPE`：依赖方向和owner边界明确 | ggml/oneDNN真实形状；能力保留给上层，不是 Math缺口 |
| `MATH-A-08` | header-only scalar primitive在 source/static/shared 中同形使用 | inline/ODR、Core invariant dependency、无 binary/FFI、零动态资源 | `hyle::math` INTERFACE target + `<hyle/math/scalar.h>` | `COMPLETE`：三交付、P0 code shape与文档已签署 | build/package + math-v2；交付 `MUST` |

public-surface 反向追踪覆盖 14 个 SIMD public headers（含两个 expert helper header）、全部普通 Runtime
family、provider manifest、锁定 provider交付、Math scalar header，以及 ggml CPU/quant/operator和 llama sampling
消费。第一遍得到八个关键结论：

1. SIMD 的 breadth 已通过 expert SDK取得 Highway/xsimd完整 header-only 能力；不能再把422误写为用户能用
   的全部 SIMD，也不能为了“API数量”重新制造 HYLE register façade；
2. 普通 whole-buffer 层已有 dtype、real/complex math、基础 activation/softmax、search/scan/order/bitpack/
   uniform RNG 的宽能力，但 row min/max/clamp/FMA、组合 reduction、GLU/norm与带索引 top-k存在真实自然缺口；
3. fp8/block-quant dot与量化 GEMV/GEMM是推理主路径，不得因为 expert可以手写就算产品已覆盖；其 format、
   packing和tensor调度必须与 Compute边界共同裁决；
4. broad target完整不代表 specialty feature完整。AMX/SME、SVE2/dotprod/i8mm和每线程matrix state与
   `PLAT-A-02` 是同一个跨域缺口；
5. attention mask/fusion、RoPE、matrix和tensor stride属于 Compute，依赖方向必须保持
   `Compute -> SIMD`，不能让不可拆分 SIMD产品依赖半成品 Compute；
6. hash/UTF/compression/crypto等“内部用了 SIMD”的能力仍归其语义owner，公共质因数是 expert/fat target
   substrate；这不违反 SIMD不可拆分的用户产品约束；
7. Math 的 checked shape arithmetic和non-negative rounding已经完整，标准数值功能直接走 C++23，复杂
   数值/模型算法有明确owner；本轮未发现应扩大 Math 的准入能力；
8. A阶段只登记上述候选。当前 `CLOSED` 的 SIMD/Math不重开，不执行逐 op benchmark，也不恢复任何旧实现；
   最终 trigger必须等 Container/Sched/IO和全局因子化反查。

这表示 SIMD + Math 产品域的阶段 A 第一遍已经完成；SIMD继续是一个不可拆分产品，Math继续是独立极小
scalar组件。新候选只有在 B 阶段确定语义owner和最小自然API后才可形成有限施工卡。

#### A10. Container 产品域（能力账本已随 A 阶段冻结）

本节只统计非侵入式 Container 产品；`container_intrusive` 已在 A3 单独展开，不能因为头文件仍位于同一
目录就重复计入。public trace 覆盖 34 个非侵入式公开头、全部 `src/container/**` 实现以及当前
Util/Sched/IO/Graph consumer。直接推理证据来自锁定 ggml 的 graph hash + side arrays、allocator/backend
registries、llama.cpp 的 recent-token history、token/name maps、trie/Aho、n-gram direct index、server
queues与 backend graph cache。strong peers 按行从 Abseil containers、Boost.Container/Heap/DynamicBitset、
Folly F14/queues/EvictingCacheMap 与 oneTBB concurrent containers/LRU 中最多取两个；实现输入另锁定
Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c`、simdutf
`2d4e249dcfd978b39dd567b156f20a43db97377b`、xxHash
`c0b5ea995d66691734b1a79ad89e73a0d2fd5a53` 与 wyhash
`e4764a0b637d34d3421a7760affada9288b625a8`。标准容器能自然完成的旅程明确路由标准库，不用 HYLE
类名数量伪装能力完整度。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `CONT-A-01` | 普通动态连续 sequence 保存 tensor、token、shape 或 metadata | allocator、range、move/copy、iterator invalidation、exception/OOM按标准合同 | `std::vector<T>`；只有不同 storage/cost 合同才选 HYLE 类型 | `ROUTED`：不复制 general vector 表面 | llama/ggml 大量 vector + 标准定义；完整路由，不是缺口 |
| `CONT-A-02` | 通常很短、偶尔 spill 的 sequence 避免常态 heap allocation | inline count、grow/OOM、non-trivial/move-only、self alias、allocator、invalidation | `SmallVector<T,N,Alloc>` | `COMPLETE`：range/insert/erase/reserve/resize 与 fallible grow齐全 | Abseil InlinedVector + Boost small_vector；基础 `MUST` |
| `CONT-A-03` | 上限编译期已知的 sequence 在对象内只构造实际元素 | zero/full、overflow policy、non-trivial lifetime、move/copy、no allocation | `StaticVector<T,N>` / `InplaceVector<T,N>` | `COMPLETE`：固定容量与 fail-fast overflow合同完整 | Boost static_vector + C++26 inplace_vector形状；基础 `MUST` |
| `CONT-A-04` | trivially-copyable metadata需要低开销 bulk append/resize和可注入 allocator | relocation、zeroing、overlap、OOM、alignment、capacity growth | `PodVector<T>`，类型约束显式 | `COMPLETE`：POD bulk journey完整 | ggml plain arrays + Folly fbvector机制对照；通用 `GENERAL` |
| `CONT-A-05` | 双端 staging queue 需要 grow、两段连续访问和稳定 O(1) 端点操作 | wrap、reserve/shrink、reject-on-full、non-trivial lifetime、allocator failure | `RingDeque<T>`；普通 deque语义仍可用 `std::deque` | `COMPLETE`：grow/bounded模式与 segments齐全 | Boost devector/circular buffer + loader journeys；`GENERAL` |
| `CONT-A-06` | sampler/repetition history只保留精确最近 N 项，满时覆盖最旧 | capacity 0/1/任意值、wrap、resize保留 newest、overwrite alias/lifetime | `CircularBuffer<T>`，不把 overwrite policy塞进普通 deque | `COMPLETE`：exact-capacity overwrite合同完整 | llama recent tokens + Boost circular_buffer；推理 `MUST` |
| `CONT-A-07` | parser/socket staging需要 byte ring 的 reserve/commit、peek/consume和两段 I/O | exact committed bytes、wrap、partial producer/consumer、growth/OOM、borrow validity | `ByteRingBuffer` 的 reservation/segments；blocking归 IO/Sched | `COMPLETE`：byte lifecycle与两段访问齐全 | HYLE IO consumer + Folly IOBuf用途对照；基础 `MUST` |
| `CONT-A-08` | append/grow 后元素地址必须稳定，但不需要并发 writer | append-only或erase、iterator/reference稳定范围、segmented locality、allocator | 普通需求先用 `std::deque`；若需要 index-stable append则使用一个具名 segmented sequence | `ROUTED`：标准 deque覆盖端点增长；`ConcurrentVector`不应为单线程强制原子成本 | Boost stable_vector + oneTBB concurrent_vector；无独立消费时不复制类型 |
| `CONT-A-09` | token/tensor/name metadata做 cache-local expected-O(1) map/set lookup | map/set、heterogeneous lookup、insert/assign、erase/filter、reserve、move-only、OOM | `FlatHashMap` / `FlatHashSet` | `COMPLETE`：Swiss probing、语义 hash、过滤和 allocator路线齐全 | Abseil flat hash + Folly F14；基础 `MUST` |
| `CONT-A-10` | read-mostly小表需要连续排序存储、range query和低常数迭代 | map/set、bulk sorted-unique、heterogeneous lookup、mutation、invalidation | `FlatMap` / `FlatSet` | `COMPLETE`：SoA map、set、bulk/mutation合同完整 | Boost flat containers + sorted metadata journey；`GENERAL` |
| `CONT-A-11` | 持续 mutation 的 ordered index 需要 logarithmic lookup/range/erase | map/set、heterogeneous compare、node split/merge、allocator/OOM、iterator lifetime | `BTreeMap` / `BTreeSet` | `COMPLETE`：ordered unique-key family完整 | Abseil btree + Boost ordered containers；`GENERAL` |
| `CONT-A-12` | multi-key、node extraction/merge或标准 node-address合同是业务语义 | unique/multi、node handle、splice/merge、allocator compatibility | 直接使用 `std::map/set/multimap/multiset/unordered_*` | `ROUTED`：HYLE不复制标准 node表面 | 标准定义 + llama ordinary maps；无不同加速合同不准入 |
| `CONT-A-13` | graph pass把 pointer/key映射成 dense ID，并以 ID 索引多组 SoA side arrays | build/reset epoch、duplicate、capacity、ID稳定范围、side-array同步、no erase或显式 compaction | Graph owner组合 `FlatHashMap<Key,Index>` 与 dense vectors；ID是逻辑 insertion ID，绝不能暴露 hash物理槽 | `ROUTED`：通用积木齐全，但尚无 graph-owned组合类型 | ggml `ggml_hash_set` 在 graph/backend/allocator多次真实消费；推理 `MUST`，带入 Graph而非新 hash-table API |
| `CONT-A-14` | 多线程 registry需要 shard-local guard期间访问 flat value | const/mutable accessor、insert/assign/erase、reserve/clear quiescence、OOM | `ConcurrentFlatHashMap`，borrow由 accessor lifetime表达 | `COMPLETE`：sharded map与锁内引用合同完整 | oneTBB concurrent_hash_map + Folly concurrent map；运行时 `MUST` |
| `CONT-A-15` | 多线程 registry需要 value地址跨 rehash稳定 | accessor、erase safety、rehash publication、reader lifetime、table reclamation、OOM | `ConcurrentNodeHashMap`；stable node与 reclamation domain分离 | `PARTIAL`：node/accessor语义完整，旧 table直到 map析构才释放 | oneTBB concurrent_hash_map + Folly reclamation；`GENERAL`，与 `MEM-A-32` 同一资源缺口 |
| `CONT-A-16` | 多线程只维护 membership，不应为 dummy mapped value支付不自然的 API/slot成本 | flat/node-stable、heterogeneous lookup、erase、accessor需求、reclamation | `ConcurrentFlatHashSet` / `ConcurrentNodeHashSet` 或能证明零额外成本的 set policy | `MISSING`：只有 concurrent map；`map<K,unit>`形状和成本未签 | oneTBB concurrent sets + Folly concurrent set形状；`GENERAL`，待 Sched/IO/registry反查 |
| `CONT-A-17` | 多 writer并行 reserve互不重叠 index并 append，读者在 publication后稳定迭代 | grow、partial construction、OOM、clear quiescence、index/reference稳定、exception policy | `ConcurrentVector<T>` segmented append | `COMPLETE`：并发 append/grow与稳定地址合同完整 | oneTBB concurrent_vector + parallel metadata build；`GENERAL` |
| `CONT-A-18` | 单 producer/单 consumer以固定内存无锁传递 typed payload | exact capacity、peek/consume、move-only、full/empty、thread roles、destruction quiescence | `SpscQueue<T>` try-only mechanism | `COMPLETE`：cache-line隔离和精确容量齐全 | Folly ProducerConsumerQueue + streaming journey；运行时 `MUST` |
| `CONT-A-19` | 多 producer/consumer在固定预算内传递任务并显式处理 full | bounded MPMC、shard hint/fair pop、move-only、progress、wrap、allocator | `BoundedQueue<T>`；争用需要分片时 `ShardedQueue<T>` | `COMPLETE`：Vyukov ring与分片策略齐全 | Folly MPMC + oneTBB bounded queue；运行时 `MUST` |
| `CONT-A-20` | scheduler owner fast-path push/pop，thief偷取单项或 FIFO批次 | owner/thief角色、fixed capacity、ABA/wrap、steal race、spill owner、quiescence | `WorkStealingDeque<T>` / `WsFifoQueue<T>`；调度政策不进入类型 | `COMPLETE`：两种窄角色机制齐全 | Folly/oneTBB scheduler mechanisms + HYLE Sched；运行时 `MUST` |
| `CONT-A-21` | burst不能安全预估小上限、但 producer不能因 full失败的 typed handoff | SPSC/MPSC/MPMC、segmented growth、OOM、try-only core、reclamation、size estimate | unbounded segmented queue mechanism；wait/timeout/close作为 Sched组合层 | `MISSING`：现有五种 queue全部固定容量 | Folly UnboundedQueue + llama server mutex/deque burst；`GENERAL`，必须保留无界内存风险 |
| `CONT-A-22` | consumer需要阻塞到 item/close/deadline/cancel，producer需要 backpressure | close drain/abort、timeout、stop token、lost wake、executor/fiber/thread、fairness | Sched channel/service组合 queue mechanism与 wait primitive；不是每个 container重复一套 CV | `ROUTED`：Container只有 try语义 | oneTBB blocking queue + llama server queue；运行时 `MUST`，owner带入 Sched |
| `CONT-A-23` | 多 producer/consumer按 priority取任务，允许近似 size但不需要 key update | comparator、equal-key、concurrent push/try_pop、allocation、starvation、close在外层 | `ConcurrentPriorityQueue<T>` mechanism，service生命周期仍由 Sched拥有 | `MISSING`：只有 FIFO/steal queues | oneTBB concurrent_priority_queue + runtime priority journey；`GENERAL`，需 Sched反查是否真实重复 |
| `CONT-A-24` | BPE、timer或规划器在已入 heap对象上 decrease/increase key、erase或 meld | stable handle、invalid handle、comparator、stable ties、allocator、O(log n) update/erase | indexed/mutable heap；普通只 push/pop继续用 `std::priority_queue`/heap algorithms | `MISSING`（mutable部分）；普通 heap完整路由标准库 | Boost.Heap + llama BPE/Time candidate；`GENERAL`，必须有 update/erase consumer才准入 |
| `CONT-A-25` | runtime-sized dense mask做 bit algebra、count、find first/next和 set-bit iteration | resize/bounds、word tail、and/or/xor/not、population、allocator/OOM | `DynamicBitSet` | `COMPLETE`：dense bitset常用操作与 iterator齐全 | Boost dynamic_bitset + ggml graph masks；基础 `MUST` |
| `CONT-A-26` | 小 active-ID集合在较大 universe内做 O(1) add/remove/contains和 dense iteration | ID bounds/grow、swap-remove invalidation、clear、duplicate | `SparseSet<Id>` | `COMPLETE`：dense+sparse membership合同完整 | scheduler active slots + sparse-set成熟形状；`GENERAL` |
| `CONT-A-27` | ID到 payload 的 sparse map、generation-safe slot或 compressed rank/select bitmap | stale handle、generation wrap、payload lifetime、compaction，或 bitmap codec/rank/select | `SparseSet + vector`可组合简单 map；generation arena由 owner持有；compressed bitmap需独立格式证据 | `ROUTED`（简单形状）/`OUT_OF_SCOPE`（其余） | 当前 ggml/llama无独立通用消费；peer数量不足以准入新类型 |
| `CONT-A-28` | API低成本借用 UTF-8/byte text并与标准 string view互操作 | null/empty、embedded NUL、known terminator、<=4 GiB界限、substring、lifetime | `StringView` method-shaped trivial borrow | `COMPLETE`：标准互操作、search/substr/NUL事实齐全 | Abseil StringView + llama metadata；基础 `MUST` |
| `CONT-A-29` | parser/loader做 ASCII case/trim/split/search、hex/C escape与 UTF-8校验/计数 | malformed offset、streaming partial code point、UTF-8/16/32转换、output capacity、replacement policy | byte/ASCII helpers留 Container；Unicode诊断/转码应为具名 text-codec factor，不堆 bool overload | `PARTIAL`：ASCII/hex/escape、UTF-8 decoder/validate/count齐全；缺错误位置和 UTF转码 | simdutf完整能力 + llama UTF/tokenizer输入；`GENERAL`，与 IO/text owner联查 |
| `CONT-A-30` | 普通拥有型 text、small string或超大分段 immutable text | allocator、SSO、NUL、mutation、sharing、slice、flatten、encoding | 普通走 `std::string`；短二进制可组合 SmallVector；rope/Cord/IOBuf只有真实跨层传递才单立 factor | `ROUTED`：不复制 owning string；无 Cord产品 | llama大量 std::string + Abseil Cord/Folly IOBuf仅作发现；当前无准入缺口 |
| `CONT-A-31` | 用户做跨平台 lexical path join/normalize/extension/absolute判断 | separator、root/drive/UNC、`.`/`..`、UTF/bytes、no filesystem access、bounded output | `PathView` + `PathBuffer` lexical helpers；实际 I/O归 Platform/IO | `COMPLETE`：Windows/POSIX lexical family完整 | loader/path journeys；基础 `GENERAL` |
| `CONT-A-32` | token/op/name重复字符串变成稳定小 ID并按 generation释放 | permanent/ref-counted/batch、OOM prefix、stale ID、const lookup、thread safety、view lifetime | `StringInterner` / `StringId` | `COMPLETE`：sharded intern、generation/refcount/batch与语义 hash齐全 | llama registries + foundation interning；`GENERAL` |
| `CONT-A-33` | map/set按值语义 hash整数、浮点、pointer、text与 composite key | Eq兼容、NaN/zero、padding、seed/secret、pair/tuple/array/custom、heterogeneous | `Hash<T>`、StringHash/HardenedStringHash、显式 bytewise opt-in | `COMPLETE`：xxHash/wyhash私有 lowering与组合桥接齐全 | Abseil Hash + llama BPE/composite keys；基础 `MUST` |
| `CONT-A-34` | 文件/model/protocol需要跨进程稳定 fingerprint、checksum或 wire hash | algorithm/version、endianness、streaming、collision/security、持久化兼容 | IO/format owner暴露具名 checksum/fingerprint；Container hash不承诺持久稳定 | `ROUTED`：当前 hash只服务运行期容器 | ggml model/cache metadata + xxHash API；`GENERAL`，不能误用 semantic hash |
| `CONT-A-35` | tokenizer/parser做 prefix trie、多 pattern Aho-Corasick或 token automaton | alphabet/UTF、longest/all match、streaming state、build/freeze、memory layout、serialization | text/tokenizer algorithm factor组合 Container storage；不把 alphabet/model policy塞入 map | `ROUTED`：通用 storage齐全，无 HYLE matcher产品 | llama `common/trie` 是直接消费；推理 `MUST`，B阶段必须确定 Text/Util-runtime/IO owner |
| `CONT-A-36` | graph/kernel/backend cache做 O(1) key lookup、命中提升和容量淘汰 | entry-count/weight budget、peek vs touch、erase/clear、eviction callback、ownership、thread safety、in-flight pin | 通用 cache只拥有 hash+recency+entry lifetime；device bytes、compile policy与异步销毁由调用者注入 | `MISSING`：没有 LRU/evicting storage mechanism | ggml CANN graph cache + Folly EvictingCacheMap + oneTBB concurrent_lru_cache；推理 `GENERAL`，旧“全部是 policy”结论过宽 |
| `CONT-A-37` | source/static/shared C++消费者取得相同模板、compiled kernel和私有依赖行为 | allocator/public edge、vendor type不泄漏、visibility、offline pin/license、header self-containment | HYLE-owned C++ types；无 Container Native FFI或稳定 C++ ABI承诺 | `COMPLETE`（基线交付）：public `core/mem`、private `platform`与 build-only/private providers | build/package + container-v2；交付 `MUST`，依赖图待 C阶段重算 |

public-surface 与 consumer 反向追踪得到九个关键结论：

1. sequence、flat/sorted/B-tree、dense masks、exact history、hash、borrowed text、path和 interner 的已有
   常用形状很宽；普通 vector/string/node containers继续走标准库，不因基础库定位复制全部标准表面；
2. ggml 的 physical hash slot 是 C 实现用来耦合 side arrays 的技巧，不是自然公共容器 identity。HYLE应由
   Graph owner组合 `FlatHashMap<Key,Index>` 与 dense storage，保留 insertion ID而不泄漏 rehash槽位；
3. 旧审计把全部 LRU/cache能力都归为 domain policy过宽。设备/编译预算仍属 consumer，但 O(1) lookup、
   recency promotion、容量淘汰与 callback是可复用机制；当前为真实 `MISSING` 候选；
4. HYLE现有 queue全部有界。固定预算、无 steady allocation的语义必须保留，但 Folly与 llama server证明
   “burst不可预估且 producer不能因 full失败”是不同的通用旅程；应调查无界 segmented mechanism，
   close/wait/deadline/cancel仍归 Sched；
5. concurrent membership和 concurrent priority是成熟 peer已提供的自然形状，当前 HYLE缺失；只有
   Sched/IO/registry反查出真实消费才升级为 `MUST`，不能为了容器类名大全直接施工；
6. mutable/indexed heap与普通 priority queue不同；后者完整路由标准库，前者只有真实 decrease-key/
   erase-handle任务才准入，并与 A3 intrusive heap、Time/Sched共同裁决；
7. `ConcurrentNodeHashMap`的 stable node合同完整，但 retired table直到析构是资源复杂度缺口，必须与
   `MEM-A-32` reclamation因子一起处理，不能由各容器继续私藏回收机制；
8. ASCII/UTF-8基础处理已有，但 Unicode错误定位和转码没有自然公开面；trie/Aho也是真实推理消费。
   两者需要 B阶段确定 text-codec/tokenizer factor，而不是继续把所有字符串算法堆入 Container；
9. 本节只登记候选且不继承旧 benchmark verdict。当前 `CLOSED` Container不重开；A13完成后由跨域
   consumer数量、成本闭环和owner边界形成有限 `PRODUCT_TRIGGER`。

这表示 Container 产品域的阶段 A 第一遍已经完成；新增实质输入为 LRU/recency、unbounded queue、
concurrent set/priority、mutable heap、Unicode codec、text matcher和 shared reclamation，而不是再次扩写
现有容器的 convenience API。

#### A11. Util-runtime 产品域（能力账本已随 A 阶段冻结）

Util-runtime 不是剩余功能的杂项桶。当前产品只有两个运行时能力族：显式的名字→factory Registry，以及
普通线程诊断用的 C++ symbol demangle；RNG、intrusive ownership、scope guard、spin lock和 compile-time
type name已经在 A4 Util-core展开。public trace覆盖 `registry.h`、`demangle.h`和唯一 compiled
`demangle.cc`，当前仓内没有 Registry 产品消费者。推理证据来自锁定 ggml/llama.cpp 的 backend
registration、按名/类型/设备枚举、dynamic load/unload和 proc-address；这些 API用于发现完整 journey，
不按函数数目全部下沉到通用 Registry。demangle strong peers锁定 Folly
`23ebefdf96c116700d3e134f04d95a2b60a509de` 与 Boost
`78505eab9b203e3e26e83893d0685f160133fa0f`。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `UTILR-A-01` | owner在启动期把显式名字绑定到 typed factory | nullable或`Result<T>` factory、typed Args、invalid/OOM、owned key/help、noexcept调用边界 | `Registry<Result, Args...>::register_creator`，无 capture函数指针；状态通过显式 Args/descriptor传入 | `COMPLETE`：两种失败形状、字符串 ownership与 transactional OOM齐全 | ggml backend registration +通用 startup composition；推理 `MUST` |
| `UTILR-A-02` | 多来源为同名实现提供 fallback/default/preferred并得到确定冲突结果 | lower no-op、equal conflict、higher replace、旧 view lifetime、registration order不决定优先级 | 显式 `RegistryPriority` + `Status` | `COMPLETE`：三档与并发替换 snapshot合同完整 | backend fallback/override journey；通用 `GENERAL` |
| `UTILR-A-03` | owner按名解析 factory/metadata一次并缓存，而不提前构造对象 | present/missing、mutable/frozen、creator identity、borrowed help/key、replacement snapshot | `find(StringView) -> optional<EntryView>`；缓存 typed creator而不是 opaque `void*` | `COMPLETE`：mutable/frozen expected-O(1) lookup齐全 | ggml `backend_reg_by_name` + registry lookup；推理 `MUST` |
| `UTILR-A-04` | owner按名直接构造并保留 factory自身失败原因 | missing与creator failure分离、move-only Args、reentry、result ownership | nullable Registry或推荐 `Registry<Result<T>,Args...>::create` | `COMPLETE`：creator锁外执行并原样传播 `Status` | backend init journey + Error contract；推理 `MUST` |
| `UTILR-A-05` | 诊断/UI枚举所有登记项、名称和帮助并可提前停止 | mutable snapshot OOM、frozen borrow、unspecified order、visitor reentry、size/has/help | `for_each` / `for_each_key` / `size` / `has` / `help` | `COMPLETE`：mutable snapshot解锁回调，frozen零分配 | ggml reg count/get +普通服务发现；`GENERAL` |
| `UTILR-A-06` | 初始化完成后一次发布，服务期反复只读 lookup/create | idempotent one-way freeze、publication memory order、view lifetime、post-freeze mutation failure | `freeze() -> FrozenView`，consumer缓存 view | `COMPLETE`：freeze O(1)/零分配；frozen read无 mutex/生命周期 atomic | init-heavy/read-heavy engine journey；推理 `MUST` |
| `UTILR-A-07` | 登记已存在 descriptor/object而不是每次运行 creator | owned/borrowed/shared value、replacement、destruction、mutable/frozen、enumeration | domain owner组合 `FlatHashMap<Name,Value>`；只有多个 owner重复 priority/publication才扩 Registry value policy | `ROUTED`：通用 Container积木完整，Registry只拥有 factory语义 | ggml直接登记 backend reg/device pointer；推理 `MUST`，由 Compute/backend catalogue拥有 |
| `UTILR-A-08` | factory需要每个 registration独有 context/capture | context lifetime、copy/move/destroy、allocation、module image、exception/noexcept | 优先登记 typed descriptor `{context, function table}`；通用 callable只有两个真实消费者共享合同才准入 | `ROUTED`：现有 Creator无 capture；descriptor可作为 factory结果/Args | Folly stateful factory差异 + plugin journey；`GENERAL`，不为 convenience引入 `std::function` |
| `UTILR-A-09` | 用户输入的 backend name支持 ASCII case-insensitive、alias或旧名迁移 | canonicalization、Unicode/ASCII、collision、deprecated alias、diagnostics | backend owner在登记/查询边界显式 canonicalize并登记 alias；generic Registry保持 exact bytes | `ROUTED`：StringView exact lookup完整 | ggml `striequals` + CLI backend names；推理 `MUST`，政策不属 Util |
| `UTILR-A-10` | 注册表按 type、capability、priority score或“best device”选择 | typed enum/schema、multi-device、capability query、ordering/tie、live resources | Compute/backend catalogue提供 typed query；Registry只做 exact name→factory | `ROUTED`（机制）/上层尚未实现 | ggml by-type/best/device APIs；推理 `MUST`，不能伪称 Util缺一个万能 predicate lookup |
| `UTILR-A-11` | 动态发现/加载 backend library并验证 ABI/version/symbol | search path、duplicate、platform suffix、signature/version、failure diagnostics、security | module manager组合 Platform `DynLib`、typed module manifest与 Registry/catalogue | `ROUTED`（Platform loader）/产品 owner `MISSING` | ggml load/load-all + llama plugin deployment；推理 `MUST`，B阶段需形成 module factor |
| `UTILR-A-12` | 卸载动态 backend前注销 factory并等待 object/thread/callback/reader排空 | registration token、remove visibility、in-flight pin、DSO code/data lifetime、timeout/failure | module manager拥有 registration leases与quiescence；不能只给 Registry加 `erase(name)`制造假安全 | `MISSING`（全局产品）；Util one-way Registry明确不支持 | ggml unload +其生命周期风险、Obs sink quiescence；推理 `GENERAL`，跨 Platform/Compute/Util/Sched裁决 |
| `UTILR-A-13` | backend提供可选扩展操作而不让核心接口每次扩张 | typed extension ID/schema、version、signature安全、missing、module lifetime | backend descriptor的 typed extension table；裸 proc-address只留 module ABI边 | `ROUTED`：generic Registry不承载 extension schema | ggml `reg_get_proc_address`真实消费；推理 `MUST`，owner为 Compute/module |
| `UTILR-A-14` | archive/static link中的实现自动登记 | dead-strip、static-init order、duplicate、error propagation、test isolation | 默认显式 composition root处理 `Status`；必要时 build-generated manifest，而非注册 macro | `OUT_OF_SCOPE`：有意不提供 auto-registration | 静态库定义性陷阱；拒绝把链接副作用当易用性 |
| `UTILR-A-15` | 普通线程把 ABI mangled C++ name转成完整可读 owning text | null/plain fallback、OOM、type_info、platform ABI、C++/Rust语言边界 | `demangle(const char*) -> std::string` 与 type_info overload | `COMPLETE`：Itanium/MSVC与 caller-owned string齐全 | Folly + Boost demangle；诊断 `MUST` |
| `UTILR-A-16` | 日志/诊断写入固定 caller buffer并知道所需长度 | cap 0、truncation、NUL、input/output overlap、platform temporary allocation/failure | snprintf-shaped `demangle(name,out,cap)` + array overload | `COMPLETE`：返回 full length、fallback与 overlap合同齐全 | Folly buffer demangle + embedded diagnostics；`MUST` |
| `UTILR-A-17` | 调试工具判断当前 binary是否真的支持某种 demangler，或区分 raw fallback与成功 | build capability、per-call parse failure、language、raw==demangled歧义 | 只有 consumer需要 machine-readable verdict时增加 `try_demangle`/capability fact；普通显示继续 fallback | `PARTIAL`：当前只返回最终文本，不能辨识成功原因 | Folly build capability接口；`GENERAL`，无当前 HYLE consumer不扩 API |
| `UTILR-A-18` | crash/signal handler在无 heap、无 lock、无 loader调用下渲染 stack symbol | async-signal safety、bounded parser、reentrancy、preloaded symbol table、truncation | Obs crash/symbolization factor使用 caller storage；普通 Util demangle不能承诺此语义 | `ROUTED`：当前 demangle明确允许 malloc/Windows mutex | Folly optional libiberty + crash journey；诊断 `MUST`，与 `OBS-A-23`联查 |
| `UTILR-A-19` | Rust/Swift或其他 foreign runtime symbol需要相应 parser | language detection、parser availability/version、安全边界、fallback | foreign-runtime/Obs owner按真实栈引入 parser；不把所有语言塞进 C++ utility默认依赖 | `OUT_OF_SCOPE` | Folly Rust demangle用于发现；无 HYLE consumer，不准入 |
| `UTILR-A-20` | 极长或不可信 symbol不能让 demangler栈/时间/内存无界增长 | input length ceiling、NUL scan、fallback、stack recursion、allocation cap、platform差异 | 明示 trusted diagnostic name；若 crash/untrusted输入准入则使用 length-bounded API/limit | `PARTIAL`：Windows动态输出限1 MiB；输入本身和 Itanium parser无统一上限 | Folly max-symbol防护 +安全定义；`GENERAL`，交给 Obs/Util边界裁决 |
| `UTILR-A-21` | source/static/shared C++用户得到相同 Registry模板与 compiled demangle | component closure、visibility、DbgHelp/ABI link、same-toolchain、无 vendor/public FFI泄漏 | `hyle::util` C++ target；Native FFI保持 N/A | `COMPLETE`（基线交付）：public Core/Error/Mem/Container，compiled demangle已三形状签署 | build/package + util-runtime-v3；交付 `MUST`，C阶段重算边 |

public-surface 与推理 consumer 反查得到七个关键结论：

1. Registry 的 factory registration、priority、lookup/create、enumeration和 one-way publication已经形成
   自然完整闭环；没有理由为了 peer API数量加入 captured callable、global static-init或 predicate大全；
2. ggml登记的是既有 backend descriptor，HYLE未来应由 Compute/backend catalogue拥有 typed values、
   device enumeration和 best-selection；用 Container组合即可，不该扭曲 factory Registry；
3. exact byte name与用户可见的 case-insensitive/alias政策是两层合同。generic Registry保持精确，backend
   owner canonicalize，避免把 ASCII/Unicode/兼容政策固化到所有 Registry；
4. dynamic load、typed manifest/proc extension与 safe unload是推理引擎必需的完整 module journey。Platform
   已有 loader机制，但 module manager和 quiescence尚缺；只增加 `Registry::erase`不能解决悬空代码；
5. ordinary-thread C++ demangle的 owning/buffer形状完整。成功原因可观察、输入上限是低优先级缺口；
   async-signal-safe symbolization则是 Obs/crash产品，不是现有 Util overload的“更快版本”；
6. Util-runtime无直接第三方产品依赖；Container的 Highway/xxHash/wyhash/simdutf仅经组件闭包间接出现，
   Registry语义不能归给这些实现机制；
7. 本节不重开 `CLOSED` util。A13之后只有 module lifecycle、Obs symbolization或两个以上真实 Registry
   consumer证明公共缺口时，才建立有限 trigger。

这表示 Util-runtime 产品域的阶段 A 第一遍已经完成；自身没有新的确定 `MUST` API缺口，新增全局输入是
module manager/quiescence，另有 demangle verdict/input-bound 两个低优先级候选。

#### A12. Sched 产品域（能力账本已随 A 阶段冻结）

Sched 的产品边界是进程内 CPU execution runtime：拥有 worker、stackful Fiber、C++ coroutine task、唤醒、
取消、定时挂起、同步原语和阻塞工作移交；不拥有 tensor/DAG 分区、模型 serving policy、网络协议或 GPU
kernel。public trace覆盖 `scheduler/bulk/blocking/fiber/coro/join/cancel/waker/timer/sync`，并反查当前 IO
的 worker-loop integration和 Compute的独立 GPU poller。推理证据来自锁定 ggml CPU threadpool的 affinity、
priority、poll、barrier、pause/resume和 caller participation，以及 llama server的 priority/cancel journey。
strong peers使用已锁定 Folly、oneTBB，另锁定 Marl
`b8406ab0a82524170ead6dd6c371b579bbc28b3e` 与 Seastar
`fd71a5b4c71b91abd4e93c9d20f88e1fd7465976`；peer的 API数量不直接构成准入。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `SCHED-A-01` | runtime在固定 worker集合上启动、接收工作、停止接收、排空并析构 | zero/auto/explicit workers、startup rollback、recursive submit、drain deadline、cooperative shutdown | move-only `Scheduler` + `create/options` + `stop_accepting/drain` | `COMPLETE`：两阶段 shutdown、定时 drain和失败回滚已有 | Folly/Marl executors + server lifecycle；基础 `MUST` |
| `SCHED-A-02` | 部署按可用 CPU顺序/集合、名称和优先级建立 worker | best-effort/required、P/E偏好、稀疏 CPU id、permission failure、worker hook | `SchedulerOptions`引用 Platform typed facts；不接收裸 OS mask | `COMPLETE`（显式 placement）：CPU order/set、priority、name、policy齐全 | ggml threadpool params + Platform；推理 `MUST` |
| `SCHED-A-03` | runnable work低争用入队、本地执行并在 worker空闲时窃取 | local FIFO/LIFO、foreign injection、pinned work、fairness、wake/park、bounded admission | 私有 per-worker deque + sharded injection；公开只表达任务语义 | `COMPLETE`：work stealing、pinned queue、idle policy和有界 admission已有 | Folly/Marl/oneTBB；基础 `MUST` |
| `SCHED-A-04` | 短、不挂起、无返回值 callback在 worker stack执行 | context lifetime、noexcept/status、queue full、shutdown、affinity | `schedule(fn, context)`或无 capture callable convenience | `COMPLETE`：借用期和失败合同完整 | executor normal journey；基础 `MUST` |
| `SCHED-A-05` | 短 typed callable在 worker stack执行并异步取得值或错误 | move-only callable/result、exception boundary、allocation/OOM、cancel-before-run | `schedule_task(F) -> Result<JoinHandle<R>>` | `COMPLETE`：typed cell和 `Result`传播已有 | Folly SemiFuture/executor journey；`GENERAL` |
| `SCHED-A-06` | 可挂起 stackful任务拥有独立 stack并在完成后释放 | stack budget/overflow、pool reuse、detached lifetime、cancel、worker migration | `spawn_detached` +显式 stack size/allocator；内部 `FiberStackPool` | `COMPLETE`（支持平台）：Mem-backed stack和 recycle已有 | Marl/Folly fibers；基础 `MUST` |
| `SCHED-A-07` | C++ coroutine以 Task值表达挂起计算并由 scheduler驱动 | lazy ownership、single consume、void/value、frame OOM、fiber bridge、cancel context | `Task<T>` + `spawn_task` + awaiter | `COMPLETE`：promise/frame lifecycle和 JoinHandle bridge已有 | C++ coroutine runtime journeys；推理 `GENERAL` |
| `SCHED-A-08` | 调用方等待、轮询或取得一个异步任务结果 | value/error、void、single take、external thread/fiber wait、abandon | move-only `JoinHandle<T>`，`join/ready/get`语义分离 | `COMPLETE`：状态拥有任务结果和等待通知 | Folly futures + Marl wait groups；基础 `MUST` |
| `SCHED-A-09` | 一个 lexical operation启动若干子任务，失败/取消/离开作用域时保证全部收束 | cancel-on-error、join-on-exit、child error aggregation、nesting、detached escape prohibition | RAII `TaskGroup`/nursery持有 child handles和 CancelScope | `MISSING`：JoinHandle和 CancelScope可手工组合，但没有结构化生命周期 | inference request fan-out + structured-concurrency peers；推理 `MUST`候选 |
| `SCHED-A-10` | 等待一组已知 task全部完成或任一完成 | homogeneous span/heterogeneous pack、empty、winner index、error/cancel、lifetime | `when_all/when_any(JoinHandle...)` | `COMPLETE`（JoinHandle域）：pack/span和 one-winner仲裁已有 | futures/select journey；基础 `GENERAL` |
| `SCHED-A-11` | 在 channel、timer、cancel、IO completion等不同 awaitable间做 first-ready race | registration rollback、multiple-ready tie、winner ownership、lost wake、cancel losers、allocation | 有限 typed `select`/registration token协议；不是 runtime `std::function`数组 | `MISSING`：IO timeout/cancel各自实现仲裁，公共层只有 JoinHandle race | IO重复机制 + Seastar/Folly async composition；推理 `MUST`候选 |
| `SCHED-A-12` | CPU密集、系统调用或外部库阻塞工作离开 scheduler worker | result/error、dynamic threads、idle retirement、shutdown、caller cancellation | `spawn_blocking(F) -> JoinHandle<R>` + process/runtime-owned `BlockingPool` | `COMPLETE`（基础 handoff）：独立线程、最大值和 keepalive已有 | ggml外部 pool隔离 + async runtimes；推理 `MUST` |
| `SCHED-A-13` | 阻塞突发不能无限累积 pending jobs或线程 | capacity、try/async submit、fairness、deadline、queue metrics、shutdown | `BlockingPoolOptions{max_threads,max_pending}` + explicit admission result | `PARTIAL`：线程上限已有，pending queue无界且无 admission/telemetry | service overload journey；`GENERAL`，需真实 burst profile后准入 |
| `SCHED-A-14` | 外部线程/设备完成时恢复一个已挂起 fiber且最多恢复一次 | copy/drop、wake-before-suspend、duplicate wake、scheduler shutdown、foreign thread | small owning `Waker` + capture/suspend awaiter | `COMPLETE`：one-shot episode和 reinjection合同已有 | IO/GPU completion；推理 `MUST` |
| `SCHED-A-15` | worker park边界同时管理 IO、GPU及其他 completion source | register/remove、source lifetime、poll budget、next deadline、wake、deep-sleep veto、shutdown quiescence | 私有 typed completion-source registry；public owner提供 lease | `PARTIAL`：当前仅一个 IO poll/wake槽；GPU另起 poller线程 | Seastar reactor +仓内 IO/Compute；推理 `MUST`候选 |
| `SCHED-A-16` | fiber按相对时长或单调绝对 deadline挂起 | zero/past/saturation、cancel race、worker timer queue、no busy wait | `sleep_for/sleep_until`；低层 `arm_timeout/cancel_timeout`供组合 | `COMPLETE`：复用 Time timer entry和 O(1) cancel | async runtime timer journey；基础 `MUST` |
| `SCHED-A-17` | 高基数任务在未来某时执行 callback或按周期运行 | one-shot/periodic、missed tick、drift、cancel handle、callback stack/allocation | `ScheduledExecutor`或 owner组合 timer+waker；周期政策显式 | `MISSING`（公共产品）：当前 `spawn + sleep`每项占 Fiber stack | Folly ScheduledExecutor + maintenance/batch window；`GENERAL`，需消费者决定是否单立因子 |
| `SCHED-A-18` | 请求/任务形成协作取消树并注册 race-safe callback | parent/child、already-cancelled、callback unlink、scope restore、thread/fiber origin | copyable `CancelToken`、`derive`、`CancelScope`、RAII callback | `COMPLETE`：树传播、当前作用域与 unlink lifecycle已有 | llama request cancel + standard stop token；推理 `MUST` |
| `SCHED-A-19` | 所有可能长期等待的原语能一致接收 deadline和取消 | absolute vs relative、reason、winner race、resource return、late completion | generic select/deadline wrapper优先；只有值所有权不同才加具名 overload | `PARTIAL`：CV/Event/Channel/Semaphore有 timed形状；Join/RwLock/oneshot/watch/broadcast与 cancellation不一致 | server SLO + IO已有重复；推理 `MUST`候选 |
| `SCHED-A-20` | trace/request/error correlation随 task/fiber迁移并在 child继承 | typed key、inherit/override、destruction、thread bridge、detached lifetime、no hidden heap on hot path | execution-local context frame + explicit propagation token；Obs只消费 view | `MISSING`：只有 CancelScope有私有 execution-local stack | Folly RequestContext + async trace journey；推理 `MUST`候选，跨 Obs/Error |
| `SCHED-A-21` | condition predicate等待能由普通线程或 fiber通知，并支持绝对/相对期限 | spurious wake、unlock/relock、notify race、timeout unlink、destruction precondition | `ConditionVariable`配 BasicLockable | `COMPLETE`：双路径和 O(1) timed unlink已有 | std/Folly Baton semantics；基础 `MUST` |
| `SCHED-A-22` | 极短、不跨挂起点的互斥临界区保护 runtime metadata | contention、worker blocking、try lock、ownership、yield prohibition | 明确命名/文档的 OS/futex `Mutex`; standard lock guard互操作 | `COMPLETE`（窄合同）：故意可能阻塞 worker且禁止跨 yield | runtime metadata journey；基础 `MUST` |
| `SCHED-A-23` | async/fiber任务持锁时可挂起而不阻塞 worker线程 | fairness、cancel/deadline、owner task、poisoning、guard跨 resume、upgrade | 独立 `AsyncMutex`/fiber mutex；不能悄悄改变现有 short Mutex成本 | `MISSING`：可用 RwLock write guard绕行但形状和成本不自然 | Folly fibers/Seastar mutex + async state；推理 `GENERAL`候选 |
| `SCHED-A-24` | 固定工作数完成、一次 latch和重复 barrier同步 | add/done race、underflow、thread/fiber wait、reuse/generation、drop | `WaitGroup`、`Latch`、`Barrier`各自表达生命周期 | `COMPLETE` | ggml barrier + standard runtime primitives；基础 `MUST` |
| `SCHED-A-25` | 单事件按 auto/manual reset通知一个或全部 waiter | set/reset race、timed wait、thread/fiber、destruction | `Event(reset_mode,initial)` | `COMPLETE` | Windows/Folly event journeys；`GENERAL` |
| `SCHED-A-26` | N-token资源按 FIFO获取、超时并用 RAII归还 | multi-permit、close、overflow、fairness、cancel、guard move | `Semaphore` + move-only `Permit` | `COMPLETE`（无 CancelToken）：FIFO、timed、close和 guard已有 | async capacity control；基础 `MUST` |
| `SCHED-A-27` | 多读单写 async锁支持 try和 read→write政策 | fairness、writer starvation、downgrade/upgrade、deadline/cancel、guard lifetime | `RwLock` + typed guards | `PARTIAL`：read/write/try/downgrade完整；无 timed/cancel和 safe upgrade | state/cache journey；`GENERAL` |
| `SCHED-A-28` | 有界 MPMC消息队列提供背压、关闭和 typed失败值 | capacity>0、try/block/timed、close/drain、move-only T、sender value return | `Channel<T>` | `COMPLETE`（bounded）：send/recv/try/relative deadline/close齐全 | service pipeline + async peers；基础 `MUST` |
| `SCHED-A-29` | producer burst不可预估或需要 capacity=0 rendezvous | unbounded segmented growth、OOM、backpressure absence、close；zero-capacity handoff公平性 | storage复用 Container queue，Sched拥有 wait/cancel/close service；分开 named factory/type | `MISSING`：当前 Channel拒绝 zero且只有 bounded ring | llama server burst + `CONT-A-21`；`GENERAL`，B阶段组合而非复制 storage |
| `SCHED-A-30` | 单次值从一个 sender交给一个 receiver | sender/receiver drop、value ownership、already sent、thread/fiber wait | `oneshot<T>() -> {Sender,Receiver}` | `COMPLETE`（基础）：drop/error和 move-only值齐全 | async result handoff；`GENERAL` |
| `SCHED-A-31` | 多 receiver观察最新配置/状态并等待版本变化 | clone receiver、sender drop、version wrap、borrow/copy、slow reader | `watch<T>()` with changed/borrow/value semantics | `COMPLETE`（基础） | async config publication；`GENERAL` |
| `SCHED-A-32` | 多 receiver取得容量有限的消息流并辨认 lag | subscriber lifecycle、lag count、close、capacity、slow reader、clone cost | `broadcast<T>()` + typed recv error | `COMPLETE`（基础） | telemetry/control fan-out；`GENERAL` |
| `SCHED-A-33` | 多来源提交的 callback按一个逻辑序列严格串行但可迁移 worker | order、reentrancy、failure、shutdown、queue growth、lifetime | `SerialExecutor`/strand包裹现有 Scheduler | `MISSING`：可手工用 Channel+单 consumer组合，但重复且占常驻任务 | Folly SerialExecutor + protocol/state machine；`GENERAL`候选 |
| `SCHED-A-34` | latency-sensitive/control/background任务共享 pool时有有限优先级/公平预算 | priority classes、starvation bound、inheritance、queue full、pinned work、dynamic policy | 小而固定的 priority lane + explicit option；模型 score仍由 serving owner持有 | `MISSING`：worker priority仅设置 OS线程，不影响 task次序 | llama priority queue + Folly/oneTBB executor priority；推理 `MUST`候选 |
| `SCHED-A-35` | 多 NUMA node机器让 worker、ready queue和主要 allocation保持本地 | node arena、CPU set、cross-node steal threshold、fallback、hotplug、memory policy | `SchedulerTopology`由 Platform facts构造，Mem resource按 arena注入 | `PARTIAL`：可显式 pin CPU，但没有 node arena或 locality-aware steal | ggml affinity + Platform/Mem NUMA；推理 `MUST`候选 |
| `SCHED-A-36` | 一段数据并行工作做 adaptive partition、parallel for/reduce/scan/sort | grain、cost、nested parallelism、cancel、determinism、scratch、backend | 独立 algorithm/Compute factor消费 executor；Sched只提供 execution substrate | `ROUTED`：`bulk_execute`仅 finite index batch，不伪装算法全集 | oneTBB algorithms + inference kernels；推理 `MUST`但 owner不在 Sched |
| `SCHED-A-37` | ggml式一次建立常驻 CPU team后反复执行 graph plan、barrier、pause/resume | dynamic chunk、caller participant、plan lifetime、nested external pool、NUMA | Compute CPU executor adapter或专用 operator runtime；不污染通用 Scheduler API | `ROUTED`：finite bulk和基础 primitives可组合，未承诺 ggml pool替换 | ggml真实 threadpool；推理 `MUST`，owner为 Compute |
| `SCHED-A-38` | runtime检测长时间未让出 worker、队列延迟和阻塞误用 | threshold、sampling、false positive、task identity、hook cost、disabled fast path | Obs sink/hook消费 scheduler events；诊断默认可关闭 | `MISSING`（产品诊断）：worker hooks存在但无 stall/queue-latency语义 | Seastar stall detector + tail latency；推理 `GENERAL`候选，跨 Obs |
| `SCHED-A-39` | 单元测试确定性推进 ready queue和虚拟时间而不依赖 wall clock | manual run-one/run-all、fake clock、timer advance、random steal、deadlock report | developer-only `ManualExecutor`/test scheduler，不进入普通 runtime成本 | `MISSING`：现有测试依赖真实 scheduler/clock | Folly ManualExecutor +定义性测试；开发 `GENERAL`候选 |
| `SCHED-A-40` | stackful context在所有宣称的 CPU/OS target上构建和切换 | ABI unwind、sanitizer hooks、stack alignment、Windows x64/ARM64、x86-64/AArch64/RISC-V | private Boost.Context fcontext port；public Fiber不暴露 vendor type | `PARTIAL`：Windows/Linux/macOS x86-64/AArch64覆盖；RISC-V与Windows ARM64无产品路径 | HYLE platform matrix + Boost.Context；跨平台 `MUST`候选 |
| `SCHED-A-41` | source/static/shared C++消费者获得同一 scheduler/fiber语义 | assembly selection、Boost pin、visibility、same-toolchain、thread libs、no private C ABI export | `hyle::sched` C++ target；fcontext C/asm仅私有机制 | `COMPLETE`（当前支持平台）：三交付已签署，无 Native FFI或稳定 C++ ABI承诺 | build/package + sched-v3；交付 `MUST` |

public-surface、仓内 consumer与推理 journey反查得到九个关键结论：

1. Sched 已经是完整 runtime而非一个 thread pool：worker、Fiber/coroutine、typed result、取消、定时、阻塞
   移交和主要同步原语都有自然闭环；不能因 peer表面更大就重写这些能力；
2. 当前 `Mutex`的窄语义是短临界区且可能阻塞 worker，它本身完整；“允许跨挂起点”的需求应形成独立
   AsyncMutex，而不是让同一个类在 benchmark中偷偷改变成本；
3. JoinHandle的 `when_all/when_any`不能替代异构 select。IO已重复实现 timer/cancel/completion仲裁，
   通用 registration/select协议是最强的跨域质因数候选之一；
4. `JoinHandle + CancelScope`提供积木但没有 lexical nursery。推理请求 fan-out要可靠处理异常退出、取消和
   child lifetime，结构化 TaskGroup是直接产品候选；
5. CPU affinity完整但 NUMA scheduling不完整：Platform知道 node/CPU，Mem能提供 node资源，Sched仍缺
   node arena与 locality-aware stealing；该缺口不能由调用方传一个更复杂 CPU数组假装解决；
6. OS worker priority与 task priority是不同能力。llama serving和成熟 executor证明有限 priority class有真实
   journey，模型 score/tenant policy仍留上层；
7. 当前 worker park只有单个 IO source，GPU因此另起 poller。多 completion source、task-local correlation、
   stall/queue-latency hook都应和 IO/Compute/Obs在 B阶段共同裁决，避免三处私有实现；
8. `bulk_execute`是有限同步批次，不是 oneTBB算法层，也不是 ggml常驻 graph pool；parallel algorithms和
   CPU operator team按依赖方向路由给上层，不扭曲 Scheduler；
9. 本节只登记候选，不继承旧 benchmark verdict，也不重开 `CLOSED` Sched。A13后按独立 consumer、
   complexity/cost闭环与 delivery边界形成有限 trigger。

这表示 Sched 产品域的阶段 A 第一遍已经完成；最实质的新增输入是 structured TaskGroup、heterogeneous
select、AsyncMutex、priority lanes、NUMA arenas、multi completion sources和 task context，而非再为单项
benchmark添加一套 queue或 worker模式。

#### A13. IO 产品域（能力账本已随 A 阶段冻结）

IO 当前约 1.13 万行 public/private实现，是 completion runtime、socket/file、TLS和 HTTP协议产品的组合，
不能用 echo benchmark或“有一个 Proactor”代替功能完整性。public trace覆盖 `capabilities/proactor/io_context/
net/tcp/udp/file/shared_memory/dns/signal/tls/http/http2`，并反查 Platform的同步 filesystem、Sched的 Waker/
cancel/timer、Mem allocator/Region和 Container pending table。直接推理 journey来自锁定 llama.cpp server的
timeout、reuse-port、Unix socket、multipart、SSE/disconnect/static file，model download的 URL/TLS/custom
headers/redirect/HEAD/ETag/range-resume/cancel/atomic replacement，以及 ggml RPC的 framed TCP与 Linux/Apple
RDMA transport、llama model loader的 Direct I/O。两个 direct strong peers锁定 Asio
`bd500f0a018db9a845ebaaed5c0318343ae9f497` 与 Seastar
`fd71a5b4c71b91abd4e93c9d20f88e1fd7465976`；liburing/OS APIs、picohttpparser、nghttp2和 OpenSSL只作为
机制/协议/依赖证据，不重复计作 strong peer。

| ID | capability family / normal journey | admitted semantic variants and lifecycle | natural C++ downstream shape | HYLE baseline coverage | source and admission result |
|---|---|---|---|---|---|
| `IO-A-01` | runtime在当前 OS选择最高可用 completion backend并诚实报告能力 | IOCP、io_uring、epoll fallback、kqueue；kernel/permission failure、queue depth、native/emulated/optional capability | move-only per-worker `Proactor` + typed backend/capability snapshot | `COMPLETE`（当前平台）：SOTA-first fallback和 capability bitset已有 | Asio/Seastar backend practice + HYLE public trace；基础 `MUST` |
| `IO-A-02` | Scheduler每个 worker拥有一个 submit/reap shard并把 park期限与 IO wait合并 | exact worker count、single issuer、fiber migration between ops、init rollback、context-before-scheduler teardown | `IoContext(Scheduler&, workers)`持有 per-worker Proactor和 blocking pool | `COMPLETE`（单 IO source）：提交/收割 ownership和 cross-thread wake合同明确 | Seastar shard-per-core + HYLE Sched；推理 `MUST` |
| `IO-A-03` | worker批量收割完成、保留 user token/result/flags/buffer id并被外部工作唤醒 | max batch、timeout 0/infinite、software post、multishot terminal、provided buffer、in-flight accounting | caller-owned completion span + `reap/post/wake`；不在机制层知道 Fiber类型 | `COMPLETE`：四后端 completion seam和 batched reap已有 | io_uring/IOCP定义 + mature loops；基础 `MUST` |
| `IO-A-04` | 普通用户在 IPv4/IPv6间构造、保存、比较、格式化并查询 local/peer socket address | numeric/hostname、scope id、port 0、Unix endpoint、sockaddr length、invalid input、no raw padding leak | owning trivial `SocketAddress/Endpoint`值 + parse/format `Result`; native view只在 submit边界 | `PARTIAL`：`ResolvedAddr`有128-byte值；其他入口仍用 `void*`、手算 length和 bool parser | Asio endpoint + llama IPv6/Unix deployment；基础 `MUST`候选 |
| `IO-A-05` | 高层 socket/file/TLS/HTTP调用保留 portable recovery语义和原生诊断 | submit failure、EOF、timeout、cancel、DNS/TLS/protocol/OS error、partial bytes、invalid object；WinSock立即捕获 | ordinary facade返回 `Result<T>`/typed transfer result；raw negative native result只留 Proactor/expert边 | `MISSING`（统一产品形状）：当前 invalid object、bool、null和多套负整数混用，除 cancel外未接 Error | `ERROR-A-07/12` + Asio error_code +真实恢复；基础易用性 `MUST` |
| `IO-A-06` | 已连接 stream做一次 read/write，listener accept，client connect | zero/partial/EOF、flags、full duplex、one outstanding per direction、borrowed buffer、submit queue full | RAII stream/listener上的 awaitable；raw op slot不暴露给普通用户 | `COMPLETE`（base one-shot）：TCP六类 awaitable和四后端实现已有 | Asio/Seastar stream journey；基础 `MUST` |
| `IO-A-07` | 不可信对端的 recv/send/connect/accept能由绝对 deadline或 CancelToken终止 | already expired/cancelled、completion race、late CQE、buffer/token lifetime、canonical outcome | base、`*_until`、`*_cancel`，后续由 generic select统一组合 token+deadline | `COMPLETE`（单腿）：TCP/UDP六操作 deadline与 cancel齐全；两者组合和通用 wait一致性见 `SCHED-A-11/19` | server shutdown/SLO；推理 `MUST` |
| `IO-A-08` | TCP server/client创建、bind/listen、accept、connect并用 RAII关闭 | IPv4/6、backlog、port 0、nonblock/CLOEXEC、accepted owner assignment、connect failure、half close | `TcpListener/TcpStream` + typed Endpoint + `Result`; connection可显式绑定 worker | `PARTIAL`：基础 RAII和 owner sharding完整；setup只返回 invalid object，地址/port观察不完整 | llama HTTP/RPC + Asio sockets；推理 `MUST` |
| `IO-A-09` | framed protocol完整读取 N bytes或发送全部 bytes且保留统一 deadline/cancel | EOF partial count、zero progress、>4 GiB chunking、absolute budget、cancel、scatter segments | `read_exact/write_all` over async byte stream；结果区分 partial与 error | `COMPLETE`（单 buffer <=u32每调用）：TCP base/deadline/cancel loops已有 | ggml RPC length framing + common protocol journey；推理 `MUST` |
| `IO-A-10` | 服务配置 socket行为并观察 endpoints | nodelay、keepalive及 probes、reuse-address/port、dual-stack、buffer sizes、DSCP、linger/reset、shutdown read/write、local/peer name | typed `SocketOptions`/narrow setters，unsupported返回 Status；不暴露平台 magic int | `PARTIAL`：nodelay、boolean keepalive、write-half-close和内部 reuse-address已有；其余无公开面 | llama reuse-port/timeouts + Asio socket options；推理 `MUST`候选 |
| `IO-A-11` | hostname/service异步解析为 IPv4/IPv6 endpoint列表且不阻塞 worker | copied host、IDN/ASCII policy、service、flags、address count、temporary/permanent failure、deadline/cancel、resolver shutdown | `resolve(...) -> Task<Result<SmallVector<Endpoint>>>`; blocking fallback是实现而非用户错误模型 | `PARTIAL`：getaddrinfo已卸载；固定255-byte host、8地址和 custom error，不能取消在途 resolver | Asio/Seastar resolver + llama remote client；推理 `MUST` |
| `IO-A-12` | client从 hostname建立连接时并发/错峰尝试 IPv6/IPv4并保留一个总期限 | RFC 8305 ordering/delay、per-attempt cleanup、proxy、local bind、winner、all-error diagnostics、cancel | `async_connect(endpoints, ConnectOptions)`或 HTTP/RPC owner组合 resolver+Scheduler select | `MISSING`：注释称可逐项 fallback，但没有 Happy Eyeballs或完整 hostname connect | llama model download/RPC + mature clients；推理 `MUST` |
| `IO-A-13` | 高建连 server一次 arm连续取得连接，并在停止时安全回收 arm/accepted handles | native/emulated、backpressure、owner distribution、terminal error、cancel/deadline、listener close | RAII `MultishotAcceptStream`或 accept-batch；low-level completion token私有 | `PARTIAL`：Proactor native/emulated multishot完整，但普通 `TcpListener`只有 one-shot | liburing mechanism + server accept journey；`GENERAL`候选 |
| `IO-A-14` | portable server连续按 consumer速度读取网络数据 | one-shot loop、buffer ownership、EOF、deadline/cancel、backpressure；不能伪造 kernel multishot | repeated `read`/buffered stream，由 consumer每次交下一块 | `COMPLETE`：四后端 one-shot是可移植基线 | Asio async_read_some +定义性；基础 `MUST` |
| `IO-A-15` | Linux高吞吐 receive一次 arm并由 kernel从 registered ring选择 buffer | buffer group lifecycle、kMore/kBuffer、pending burst、RAII return、drop/cancel、old kernel fallback | 成功取得 `IoUringExtension`后创建 `MultishotRecvStream/RecvBuf` | `COMPLETE`（显式 Linux extension）：provided ring、multishot和late completion已有 | liburing proxy shape；高吞吐 `GENERAL` |
| `IO-A-16` | Linux高频 socket避免每 op fd lookup或 direct accept到 fixed slot | registration/update/release、slot exhaustion、direct close、fallback、teardown | `IoUringExtension` borrowed capability；ordinary TcpStream保持 raw handle | `COMPLETE`（机制层）：fixed recv/send和 direct accept/close已有 | io_uring +既有实测；可选 accelerator `GENERAL` |
| `IO-A-17` | RPC/TLS/HTTP framing用 scatter/gather一次提交 header+payload或把数据读入多段 | iovec lifetime/count/bytes overflow、partial segment progress、platform max、cancel/deadline | span of byte spans + typed progress cursor；single-buffer API保留 | `MISSING`：无 readv/writev/sendmsg/WSASend普通产品腿 | ggml framed RPC + Asio buffers + `PLAT-A-22`；推理 `MUST` |
| `IO-A-18` | static/model/data server在 file与socket间做高吞吐 bulk transfer | sendfile/splice/copy_file_range/TransmitFile、offset/count/partial、TLS fallback、cancel/backpressure、sparse file | `send_file`/bulk transfer在 plain transport快腿；TLS或不支持时 bounded-buffer fallback | `MISSING`：当前只能手写 File read + Tcp write copy loop | llama static/model serving + Seastar zero-copy streams；推理 `MUST`候选 |
| `IO-A-19` | UDP endpoint发送/接收一个 datagram并得到 peer地址 | v4/v6、zero-length、truncation、message boundary、deadline/cancel、address length | move-only `UdpSocket` + typed endpoint/datagram result | `COMPLETE`（基础）：send/recv、timed/cancel和 family-aware setup已有，地址形状仍受 `IO-A-04`限制 | Asio datagram + backend discovery；基础 `MUST` |
| `IO-A-20` | telemetry/discovery/QUIC substrate使用 connected UDP、multicast、broadcast或 batch datagrams | join/leave/source group、TTL/interface、pktinfo/ECN/truncation、recvmmsg/sendmmsg、queue depth | 分族 socket options + datagram metadata/batch spans | `MISSING`：只有 unconnected one-shot sendto/recvfrom | Asio/Seastar datagram breadth；`GENERAL`，无当前 llama核心需求不一次性全抄 |
| `IO-A-21` | 同机 server/client通过 Unix-domain socket或 Windows named pipe避免 TCP栈 | path/abstract namespace、permissions、cleanup、credentials、message/byte mode、cross-platform endpoint | `LocalStream/LocalListener`，Windows/POSIX各自真实实现而非假同构 flags | `MISSING` | llama server直接支持 Unix socket + Asio local/pipe；推理 `MUST` |
| `IO-A-22` | host启动工具/MCP/backend子进程并异步读写 stdin/stdout/stderr | anonymous pipe、inheritance、half-close、process exit/cancel、deadlock、Windows quoting/job/POSIX signals | Process owner管理 child；IO拥有 adopted async pipe endpoints，Sched select等待 output/exit | `ROUTED`（process lifecycle）/`MISSING`（async pipe endpoint） | llama MCP subprocess + Asio pipes；推理 `GENERAL`，与 Platform future process因子联查 |
| `IO-A-23` | regular file异步 open、positional read/write和 close而不阻塞 scheduler worker | short/EOF、offset overflow、append、path lifetime、native async vs blocking offload、destructor fallback | move-only async File消费/持有唯一 native owner，返回 typed outcome | `PARTIAL`：功能操作完整且 fallback不堵 worker；与 Platform FileHandle重复 owner且失败只变 invalid/negative | llama split model files + Seastar file + `PLAT-A-22`；推理 `MUST` |
| `IO-A-24` | 文件查询 size/type/mtime、sync并预分配空间 | async metadata、datasync/full、mode 0/unsupported mode、offset+len overflow、close durability | File methods + Error结果；namespace mutation仍走 Platform | `COMPLETE`（机制）：io_uring原生或 blocking offload；nonzero unsupported诚实失败 | model/cache writer + Asio/Seastar；基础 `MUST` |
| `IO-A-25` | 大模型选择 Direct I/O绕页缓存并满足设备/OS alignment | O_DIRECT/NO_BUFFERING、sector/block alignment、tail bounce、fallback policy、queue depth、buffer registration、telemetry | `OpenOptions{direct}` + queried alignment + Mem aligned buffer/resource；不能静默回 buffered | `MISSING`：open flags和 File API没有 direct/alignment能力 | llama.cpp `LLAMA_LOAD_MODE_DIRECT_IO` + `PLAT-A-22`；推理 `MUST` |
| `IO-A-26` | 慢/远程 filesystem操作按请求期限取消且 late completion不悬空 | open/read/write/sync、backend cancellability、blocking job无法强杀、buffer lifetime、partial result | cancel/deadline awaitable，或明确不可取消的 blocking operation handle | `MISSING`：File无 deadline/CancelToken；注释把它排除但真实 network filesystem可无界 | Asio cancellation + service shutdown；`GENERAL`候选 |
| `IO-A-27` | 顺序模型/数据流做 read-ahead、buffered input/output和 flush | buffer ownership、seek、EOF、backpressure、prefetch window、direct interaction、allocator | 独立 async stream adapter组合 File，不把 cursor状态塞进 positional primitive | `MISSING`（product convenience）但可由 File+Mem组合 | Seastar fstream + model streaming；`GENERAL`，需要 loader profile后准入 |
| `IO-A-28` | filesystem namespace做 stat/exists/mkdir/remove/rename/temp/atomic replace和同步 cursor I/O | UTF-8、symlink、TOCTOU、durability、visitor、error | 直接消费 Platform `fs/FileHandle`；IO只在需要异步 completion时适配 owner | `ROUTED`：`PLAT-A-21/23/25`已完整，不能在 IO复制第二套 path API | llama cache/download；推理 `MUST`由 Platform承载 |
| `IO-A-29` | model/config/cache变化触发 reload而不轮询 whole directory | file/dir watch、rename pairs、overflow/rescan、recursive、close/cancel、platform differences | async `FileWatcher`输出 typed events；路径 ownership由 caller/Platform | `MISSING`：Platform明确路由 IO，当前无 watcher | Seastar fsnotify + deployment reload journey；`GENERAL`候选 |
| `IO-A-30` | 两进程按名字共享一段 host memory并明确 namespace生命周期 | create-exclusive/open、size、permissions、flush/unlink、resize、fd/HANDLE transfer、sync/crash cleanup、offset-only data | mapping owner归 Mem/IPC factor；IO只拥有命名/handle-transfer和 async notification机制 | `PARTIAL`：Named mapping create/open/data/flush完整；权限、resize、handle transfer和 channel/sync缺失 | ggml virtgpu/shared backend + `PLAT-A-30`/`MEM-A-25`；推理 `GENERAL`，B阶段重定 owner |
| `IO-A-31` | 进程在 SIGINT/SIGTERM/Ctrl-C后以普通 async completion启动优雅关闭 | handler safety、host handler restore、single process owner、repeat events、close race、fan-out | one `SignalSet` waiter，随后用 Sched channel/cancel tree广播 | `COMPLETE`（当前 graceful journey）：POSIX self-pipe与 Windows console handler已有 | Asio/Seastar signal + server shutdown；推理 `MUST` |
| `IO-A-32` | client/server把已连接 TCP升级为验证的 TLS stream并异步 handshake/read/write/shutdown | TLS1.2/1.3、SNI、DNS/IP identity、system trust、close_notify、partial records、allocator/OOM | move-only `TlsContext/TlsStream` + typed async operations；OpenSSL type不泄漏 | `COMPLETE`（基础 optional capability）：memory BIO transport与身份验证已有 | Asio/Seastar TLS + llama HTTPS；推理 `MUST` |
| `IO-A-33` | production TLS配置 custom trust、server chain、mTLS、protocol/cipher、通用 ALPN和 session reuse | CA file/memory/system、client cert/key、verify callback/policy、ticket/cache lifetime、resumption、key reload、ALPN list | immutable/shared `TlsConfig` builder + per-connection state；HTTP/2只消费通用 ALPN结果 | `PARTIAL`：system-trust client、file server cert和专用 h2 ALPN；缺 custom CA/mTLS/general ALPN/session reuse | llama TLS deployment + Asio SSL context；推理 `MUST`候选 |
| `IO-A-34` | TLS handshake/read/write/shutdown遵守总 deadline和 request cancellation | WANT_READ/WRITE循环、partial plaintext、timeout race、close_notify budget、late TCP completion | operation接受 deadline/token或 generic select wrapper贯穿整个 pump | `MISSING`：TLS pump只调用 base TCP；shutdown虽有限策略但无 caller deadline/token | hostile peer + server SLO；推理 `MUST` |
| `IO-A-35` | HTTP/1.0/1.1连接安全解析/生成 request/response并保持 keep-alive/pipeline leftover | header/body caps、CL/chunked/close-delimited、TE+CL、duplicate CL、injection、HEAD/1xx/204/304 | borrowed request/response view + explicit caps + transport-neutral framing core | `COMPLETE`（fixed-body framing）：pico parser集成和安全边界已有 | llama HTTP server/client + protocol定义；推理 `MUST` |
| `IO-A-36` | server在一个 accepted connection上循环请求、调用 handler并为每请求复用有界 arena | plain/TLS、keepalive、handler allocation/OOM、response lifetime、connection close、request retention cap | low-level `HttpServerConn` + allocator-first per-connection serve helper | `COMPLETE`（单 connection）：request Region和 retained cap已有 | llama handler journey + HYLE Mem；推理 `MUST` |
| `IO-A-37` | token生成器尚未产出下一块时异步等待，随后无忙等地流式发送 HTTP/1响应/SSE | async producer、empty heartbeat、disconnect、backpressure、deadline/cancel、trailers、chunk lifetime | `AsyncBodyReader/Stream<OwnedOrBorrowedChunk>` awaitable；writer每块 await producer再写 | `PARTIAL`且当前形状不自然：`NextChunkFn`同步返回，不能挂起；等待会堵 worker，空块循环会 busy-spin | llama SSE/token stream直接消费；推理 `MUST` |
| `IO-A-38` | server流式接收大 request/upload并可安全解析 multipart而不先缓冲全体 | content-length/chunked、body cap、incremental consumer、backpressure、disconnect、multipart boundaries/file metadata/temp spill | headers先交付，再以 async body reader流动；multipart是可选 streaming codec | `MISSING`：request body固定缓冲且上限8 MiB，无 multipart | llama multimodal multipart/file upload；推理 `MUST`候选 |
| `IO-A-39` | 整个 HTTP request从 headers到 handler/body/response共享一个 deadline、CancelToken与 disconnect信号 | slowloris、handler timeout、client gone、stream producer、shutdown、late op、status mapping | `RequestContext{deadline,cancel,peer}`；底层每次 transport op继承同一预算 | `MISSING`：只可在裸 Tcp层手工选 timed/cancel，HTTP convenience无法贯穿 | llama `is_connection_closed` + configurable timeout；推理 `MUST` |
| `IO-A-40` | 用户启动一个可用 HTTP server：accept、分配连接、并发限制、停止接收、排空 | bind/reuse-port/Unix/TLS/ALPN、per-IP/global admission、handler task group、graceful deadline、overload response | `HttpServer` host组合 Listener+Scheduler TaskGroup；routing/business仍由 caller | `MISSING`（host lifecycle）：`http_serve`只服务一条已接受连接 | llama server main + Sched structured-concurrency gap；推理 `MUST` |
| `IO-A-41` | 已连接 HTTP/1 client发送一个基础 request并读取安全有界 response | method/path/host/body、HEAD、chunked/CL/close、keepalive、borrowed view | `HttpClientConn::request` on existing transport | `COMPLETE`（窄连接级）：basic fixed request/response已有 | llama health/API calls；基础 `GENERAL` |
| `IO-A-42` | 普通 client从 URL完成 parse→resolve→Happy-Eyeballs→TLS→custom request并复用连接 | URI/IPv6/userinfo、headers/auth、proxy、redirect policy、pool/idle timeout、DNS/TLS errors、HTTP/2 negotiation | `HttpClient` owning resolver/connector/pool + request builder；connection primitive仍可直接用 | `MISSING`：当前 caller必须自组 transport，request无法添加任意 headers或连接策略 | llama model download + Asio composition；推理 `MUST` |
| `IO-A-43` | 大模型从 HTTP(S)可靠下载到本地缓存并可恢复 | HEAD/ETag/Last-Modified、Range/206、resume、redirect、bearer/basic auth、progress/cancel/retry、size/checksum、temp+fsync+atomic rename | 独立 download helper组合 Http async body、Platform atomic-file mechanisms和 Obs callback | `MISSING`：HTTP client固定8 MiB缓冲，不能承载模型下载 | llama `common/download.cpp`真实 journey；推理 `MUST`，不属于应用“可选 convenience” |
| `IO-A-44` | HTTP/2 client/server在一个 connection复用多 stream并正确处理 HPACK/flow control/RST/GOAWAY | h2c/TLS ALPN、interleaving、stream cap/body cap、header validation、graceful shutdown、allocator | HYLE transport驱动 private locked nghttp2 core；public只暴露 HYLE messages/stream IDs | `COMPLETE`（fixed-body optional capability）：多 stream和 flow-control已有 | nghttp2 protocol core + modern serving；推理 `GENERAL` |
| `IO-A-45` | HTTP/2 request/response body边生成边发送/接收，按 stream window施加背压 | DATA provider, deferred resume、trailers、per-stream cancel/deadline、large body、connection scheduling、buffer lifetime | per-stream async body reader/writer；nghttp2 deferred data callback与 HYLE Waker桥接 | `MISSING`：submit/respond只接受最多8 MiB fixed body，next只返回完整 body | token streaming与model transfer；推理 `MUST`候选 |
| `IO-A-46` | routing、CORS/auth/middleware、JSON schema、static mount与业务 endpoint | route params、policy order、security、content type、cache headers、framework ownership | inference host/application组合 HTTP primitives；static bytes可消费 `IO-A-18` | `ROUTED`：不把 llama server framework复制入基础 IO | llama server大量真实消费但语义属应用；明确拒绝巨型 web framework |
| `IO-A-47` | SSE或 WebSocket承载长连接双向/单向增量消息 | SSE retry/event framing、heartbeat/disconnect；WebSocket upgrade/frame/mask/ping/close/backpressure | SSE组合 async HTTP body；WebSocket只有具名双向 consumer时成为独立 protocol factor | `PARTIAL`（SSE依赖 `IO-A-37`）/`OUT_OF_SCOPE`（WebSocket） | llama SSE是直接需求；无当前 WebSocket consumer，不因 peer breadth准入 |
| `IO-A-48` | QUIC/HTTP/3、gRPC或完整 RPC协议栈 | congestion/crypto/multiplex/schema/retry/load balance、dependency/runtime ownership | 使用成熟协议库并通过 HYLE completion/executor adapter；不在 IO手写协议全集 | `OUT_OF_SCOPE`（当前 base）：无已命名首版 engine consumer | industry discovery；待真实 protocol decision再准入 |
| `IO-A-49` | ggml式 remote backend在可靠 byte stream上交换 length-prefixed typed messages | frame size/endian/version、partial transfer、capability handshake、shutdown、auth、cancel | Compute/RPC owner定义 codec；IO提供 exact/scatter transport和 endpoint | `ROUTED`：`IO-A-09`可承载基础 TCP framing，协议不属 IO | ggml RPC直接 journey；推理 `MUST`由 Compute/remote factor拥有 |
| `IO-A-50` | distributed inference在同机/局域高速链路用 RDMA传大 tensor并协商 TCP fallback | verbs/Apple RDMA、registration/pinned memory、queue depth、segmentation/flush、completion/cancel、capability handshake | 独立 transport factor组合 Mem registered regions、IO completion和 Compute RPC；不藏在 TcpStream bool开关 | `MISSING`（全局产品） | ggml Linux verbs + Apple Thunderbolt RDMA直接消费；推理 `GENERAL`，B阶段定 IO/Compute/Mem owner |
| `IO-A-51` | 运维观察 pending/in-flight、queue saturation、bytes/latency/error/cancel和 connection lifecycle | disabled fast path、per-shard counters、high-cardinality labels、trace correlation、backend queue depth、drop | low-cost hooks/counters交 Obs汇总；不在 Proactor内建 exporter | `PARTIAL`：capability/in-flight query和 benchmark hooks零散，无产品级 typed events/metrics | inference tail latency + `OBS-A`/`SCHED-A-38`；推理 `GENERAL`候选 |
| `IO-A-52` | source/static/shared用户在支持 target得到同一 C++语义和可选 TLS/H2 capability | x64/ARM64 OS matrix、fcontext assembly、optional OpenSSL/nghttp2、private pico、offline pin、visibility、same-toolchain | `hyle::io` + explicit feature options/capability probes；vendor types不穿 public API | `PARTIAL`（产品矩阵）：三交付机制已有；coroutine transport不覆盖 RISC-V/Windows ARM64，OpenSSL是系统依赖 | build/package + `SCHED-A-40`；交付 `MUST`候选 |
| `IO-A-53` | C++消费者从一个入口学会选择 base/timed/cancel、portable/extension和 protocol层 | header grouping、typed failure、lifetime/worker preconditions、feature-off、examples、no stale FFI promise | layer-oriented guide + compilable journeys；普通用户不需要理解 completion token tags | `PARTIAL`：C++ guide覆盖主要腿，但 Error/async stream/client/server host缺口使易用性未闭环；历史 task仍含已删除 C ABI叙事 | HYLE public docs trace；易用性 `MUST` |
| `IO-A-54` | foreign runtime需要网络/file服务时获得稳定 runtime-owned handle ABI | create/context/op/cancel/poll/callback/destroy、buffer ownership、thread attach、schema/version | 只有具名 Java/C#等 consumer后设计 type-erased runtime；不机械导出 awaitable/模板 | `OUT_OF_SCOPE`（当前）：Native FFI为 N/A，旧 C bridge已删除 | global Native FFI policy；拒绝恢复无人消费的伪 C ABI |

public-surface、HYLE相邻模块、locked inference consumer与两个 strong peer反查得到十二个关键结论：

1. Proactor、per-worker shard、completion batching、park/wake和 portable one-shot网络机制是完整的；这只证明
   IO substrate成立，不证明 HTTP client、server host、Direct I/O或 distributed transport已经完成；
2. 普通 C++层把 invalid object、bool、null、custom DNS error和负 native integer混用，是明确的 Error集成和
   易用性缺口。raw negative CQE可留 expert层，但 ordinary File/TCP/TLS/HTTP必须有 typed结果；
3. 当前地址面仍大量使用 `void* + length`；`ResolvedAddr`证明稳定 POD可行。一个 typed Endpoint是 DNS、
   TCP、UDP、Unix/local stream和错误诊断共同需要的质因数候选；
4. 网络单操作 deadline/cancel已经扎实，HTTP/TLS/DNS/File却没有把同一 budget/token贯穿完整 operation。
   这与 `SCHED-A-11/19` 的 select/deadline组合直接相交，不能继续逐协议复制 race state machine；
5. HTTP/1固定 body framing和 per-connection arena完整，但 `NextChunkFn`是同步 callback，无法等待下一 token。
   因此当前“流式响应/SSE”只适合预先可得 chunks，不是推理引擎需要的异步 backpressured stream；
6. HTTP client只是已连接 transport上的窄 request。它不能携带任意 headers，也没有 URL、DNS、Happy
   Eyeballs、TLS、redirect或 pool；更关键的是8 MiB整包 response无法下载模型。llama download直接证明
   async body + range/resume + atomic-file组合是 `MUST` journey；
7. HTTP/2正确借 nghttp2承担协议状态，但 body仍固定整包；异步 data provider、per-stream cancel和 flow-
   controlled stream是其真实剩余边界，不能靠“支持多 stream”掩盖；
8. `File`已避免在 worker上做同步磁盘操作，但缺 llama当前正式提供的 Direct I/O、vectored I/O和 bulk
   file-to-socket；同时它与 Platform `FileHandle`重复 native owner，B阶段必须消除双所有权模型；
9. basic UDP完整；advanced UDP、watcher、local IPC和 async pipe不能因“peer有很多 API”全量施工。Unix
   socket已有 llama server直接需求，watcher/pipe分别需配置 reload与 MCP/process journey形成 owner；
10. `SharedMemory`完成命名 mapping而非 IPC channel；mapping owner与 residency归 Mem，namespace/handle
    transfer/notification归 Platform/IO机制。RDMA同样必须把 registered memory、completion与 RPC政策拆开；
11. routing/CORS/auth/JSON/static policy和 ggml RPC codec属于应用/Compute owner；SSE substrate、send-file和
    exact/scatter transport仍属 IO。这个边界避免把 IO做成 web framework，也不丢真实加速积木；
12. 本节是 `QUEUED` IO的 Stage A输入，不继承2026-07旧 benchmark或“C ABI已完整”结论；当前没有产品
    实现改动。A完成后由跨域 owner和有限 trigger决定 IO首轮 scope。

这表示 IO 产品域以及 13 个产品域的阶段 A第一遍已经完成。IO的确定高价值输入不是更多低层 submit
重载，而是 typed Endpoint/Error、async body与 whole-operation cancellation、完整 HTTP client/server host、
Direct/vectored/bulk file IO、local stream，以及 RDMA/shared-memory等跨域 transport的正确因子化。

#### A 阶段全局交叉核对与冻结（完成）

本节是 A1--A13 的 terminal record。机械扫描得到 13 个产品域、14 张能力表（SIMD 与 Math 按绑定规则
分别审计边界）、377 个连续且唯一的 ID；没有重复、断号、未知覆盖状态或未填写的表行。覆盖分布只描述
基线事实，不是完成率或施工优先级：

| product ledger | ID range | rows | complete | partial | missing | routed | out of scope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core | `CORE-A-01..13` | 13 | 4 | 6 | 0 | 2 | 1 |
| Error | `ERROR-A-01..16` | 16 | 7 | 5 | 0 | 3 | 1 |
| Container-intrusive | `INTR-A-01..15` | 15 | 6 | 4 | 3 | 2 | 0 |
| Util-core | `UTILC-A-01..21` | 21 | 7 | 4 | 2 | 7 | 1 |
| Time | `TIME-A-01..20` | 20 | 12 | 0 | 2 | 6 | 0 |
| Platform | `PLAT-A-01..36` | 36 | 20 | 5 | 5 | 4 | 2 |
| Mem | `MEM-A-01..35` | 35 | 20 | 8 | 5 | 0 | 2 |
| Obs | `OBS-A-01..26` | 26 | 9 | 5 | 6 | 6 | 0 |
| SIMD | `SIMD-A-01..34` | 34 | 19 | 5 | 3 | 7 | 0 |
| Math boundary | `MATH-A-01..08` | 8 | 3 | 0 | 0 | 3 | 2 |
| Container | `CONT-A-01..37` | 37 | 21 | 2 | 5 | 9 | 0 |
| Util-runtime | `UTILR-A-01..21` | 21 | 9 | 2 | 1 | 7 | 2 |
| Sched | `SCHED-A-01..41` | 41 | 23 | 6 | 10 | 2 | 0 |
| IO | `IO-A-01..54` | 54 | 17 | 13 | 18 | 4 | 2 |
| **total** | -- | **377** | **177** | **65** | **60** | **62** | **13** |

`PARTIAL` 或 `MISSING` 只表示 HYLE 基线没有完整自有实现；它可能是已准入的基础能力、需要 B 阶段
定 owner 的候选、需要更多真实 consumer 才能准入的 `GENERAL`，也可能已有标准库或成熟依赖路线。
因此 125 行未完全覆盖不能机械转换成 125 个 API、FIX 或模块 reopen trigger。

跨域反查把所有 `PARTIAL/MISSING` 且准入文字提到直接、候选或条件式 `MUST` 的 66 行收敛到以下有限
reconciliation clusters；这张表只证明遗漏已被解释，不提前决定 owner：

| cluster | explained capability rows | B-stage ownership question |
|---|---|---|
| R1. 基础合同、fatal 与错误边界 | `CORE-A-02/03/05`、`ERROR-A-07/14`、`OBS-A-02`、`IO-A-05` | compiler contract、fatal observer、native error、exception adapter 与 structured event 各自放在哪个最小因子 |
| R2. 编译事实、ISA 与三交付合同 | `CORE-A-06/07`、`UTILC-A-12`、`PLAT-A-02`、`MEM-A-35`、`SIMD-A-06`、`SCHED-A-40`、`IO-A-52/53` | Core/Platform/SIMD 的 compile/runtime facts 如何唯一裁决，Build/package 如何只承载交付而不拥有产品语义 |
| R3. intrusive links 与并发 storage 机制 | `INTR-A-09/10` | 私有 slist/hlist 是否已有两个独立 consumer；safe replace 仍是条件项，不能因一行 `MUST` 字样自动公开化 |
| R4. allocator、buffer、backend 与 planner | `MEM-A-03/06/14/26/27/28/29/30/31` | resource taxonomy、stable-address pool、device buffer/view、async transfer、pool 与 graph planner 应拆成哪些可独立消费因子 |
| R5. topology、residency、priority 与 locality | `PLAT-A-05/09/17/29`、`MEM-A-22/25`、`SCHED-A-34/35` | machine facts、mapping/residency policy、NUMA allocation 与 scheduler policy 怎样单向依赖 |
| R6. structured async composition | `SCHED-A-09/11/15/19/20`、`IO-A-11/12/34/39/40` | task group、heterogeneous select、completion registry、deadline/cancel、task/request context 是否形成共享 runtime factors |
| R7. endpoint、socket、file 与 local transport | `IO-A-04/08/10/17/18/21/23/25` | typed endpoint/error、vectored/bulk/direct file I/O 和 local stream 的 ownership 如何避免 Platform/IO 双 owner |
| R8. TLS、HTTP streaming 与 model transfer | `IO-A-33/37/38/42/43/45` | transport-neutral async body、client/server host、TLS config 与 download helper 怎样组合而不把 IO 变成 web framework |
| R9. metrics 与 distribution | `OBS-A-16/17` | counter/histogram 的低成本原语、aggregation 和 exporter policy 如何分层 |
| R10. 推理数值原语 | `SIMD-A-11/16/17/20/27` | dtype conversion、activation/gating、normalization 与 top-k 应由不可拆分 SIMD 公开产品还是未来 Compute 算子层拥有 |

其余 `PARTIAL/MISSING` 行的 admission 已明确为 `GENERAL` 或成熟路线比较项；主要相交面是 counted/atomic
intrusive 与 reclamation（Container-intrusive/Mem/Container）、borrow/nullability/callback vocabulary
（Error/Util-core）、periodic/manual time 与 async primitives（Time/Sched）、process/module quiescence
（Platform/Util-runtime/IO）、Unicode/text（Container/IO）、runtime trace/context（Obs/Sched/IO）以及
IPC/shared-memory/RDMA（Platform/Mem/IO/Compute）。它们进入 B 的同一全局账本，不因 peer surface 大而
自动获准实现。已有 62 个 `ROUTED` 行保留标准库、成熟依赖或具名上层 owner；13 个 `OUT_OF_SCOPE` 行
都有显式拒绝理由。其中唯一在 admission 中同时写明推理 `MUST` 的 `MEM-A-34` 已明确路由给 Compute，
不是被删除的需求。

最后一次纵向查漏覆盖 model load/placement、CPU kernel、sampling、request scheduling、network serving、
model download、telemetry、shutdown/plugin lifecycle 和 distributed transport；横向查漏覆盖 fatal/error、
compile/runtime facts、time、topology、filesystem/process、memory、concurrency、text、delivery 与 Native FFI
政策。Compute/Graph 仍冻结，但它们的直接消费要求已作为 `ROUTED` 行或上述 cluster 输入保留；
Build/package 仍是横向开发单元，不伪造第 14 个产品能力域。

据此裁决：**阶段 A 已冻结，阶段 B/C 尚未开始**。冻结的是能力输入和拒绝理由，不是 177 行以外的实现
承诺，也不改变 `module-status.json`、不重开任何 `CLOSED` 模块、不授权产品改动。以后只有新的具名
consumer、锁定 peer 差异、可复现产品缺陷或标准变化，才能以增量 evidence 修订 A；不能用 benchmark
行、代码行数或“再看一遍”恢复无限清点。

### B. 跨域归属与质因数分解

只使用已经冻结的 A 输入建立全局账本。下列 `B-Fxx` 是本 task 内的 semantic factor ID，不是已经登记的
component ID、CMake target 或模块状态；真实 target 名、公开/私有边和迁移顺序只在 C 阶段登记。

factor verdict 的含义如下：

- `KEEP_PRIME`：现有边界的语义闭环可以保留；不表示当前实现自动覆盖新增能力行；
- `SPLIT_PRIME`：应从现有聚合边界分出，因而能减少真实闭包或隔离可选成本；
- `NEW_PRIME`：当前没有相应产品因子，但已准入能力、独立 consumer 和生命周期足以建立一个；
- `ROUTE`：标准库、成熟依赖、上层或宿主才是自然完成路线，不制造 HYLE component。

`source A rows` 使用闭区间，例如 `CORE-A-01..06`。每个 A 行只在下面一张主表出现一次；若一行本来混合
了多个 owner，则整体进入 `B-R06`，再由后面的 decomposition table 切开语义，不能让同一个 A ID 在
两个 factor 中重复计票。

#### B1. 全局 factor 与 owner 账本

| ID / verdict | semantic factor and owning product domain | source A rows | independent consumers | cost and delivery closure | duplicate/conflict resolution |
|---|---|---|---|---|---|
| `B-F01 KEEP_PRIME` | `core` / Core | `CORE-A-01..06, CORE-A-08..09, ERROR-A-15` | Error、Platform、Mem 及所有低层 header | header-only；成功路径无 allocation/lock/runtime registry；source/static/shared 只传播 header contract | Core 只拥有 invariant、compiler/layout facts 与 hints；fatal observer 是无反向依赖的 bounded callback seam，日志/crash policy不进入 Core |
| `B-F02 KEEP_PRIME` | `error` / Error | `ERROR-A-01..11` | Platform、Mem、Sched、IO 及宿主边界 | code-only/success 不分配；rich text 留冷 backing；一个 compiled representation seam | Error 唯一拥有 canonical recovery vocabulary；协议 pending/closed、fatal、日志、exception boundary和 owner-specific typed schema 分别外置 |
| `B-F03 KEEP_PRIME` | `container_intrusive` / Container | `INTR-A-01..13` | Time timer nodes、Sched/IO state machines、普通 stable-address owner | header-only、caller-owned storage；counted/atomic/heap 是独立类型选择，不给默认 hook 加 metadata | list/slist/atomic publish/intrusive heap 都共享 hook ownership 边界；高级 policy 可直接借 Boost，不复制 policy matrix |
| `B-F04 KEEP_PRIME` | `util_core` / Util-core | `UTILC-A-01..02, UTILC-A-04, UTILC-A-06, UTILC-A-08..09, UTILC-A-12, UTILC-A-14..15, UTILC-A-17..18` | sampler/test RNG、owner RAII、短锁状态、同步 callback API | header-only，只依赖 Core；每个 header 按需编译，继续拆 target 不减少依赖或常态成本 | 只保留 statistical RNG、scope guard、intrusive strong ref、spin/backoff、FunctionRef、borrow vocabulary 和 diagnostic type name；park/fiber/crypto 外置 |
| `B-F05 SPLIT_PRIME` | `time` host clocks / Time | `TIME-A-01..05, TIME-A-07..08, TIME-A-10, TIME-A-12, TIME-A-17` | Platform thread/fs、Sched、Obs、IO、独立 elapsed/deadline 用户 | typed host clocks + 很小 OS artifact；不带 intrusive node、callback executor 或 timezone database | Time 唯一拥有 host time domains/conversion 与 injectable clock vocabulary；device time和 async policy外置 |
| `B-F06 SPLIT_PRIME` | `timer_queue` / Time | `TIME-A-13..15` | Sched timer service、IO common timeout、独立 event loop | header/data-structure factor，依赖 host time value与 intrusive hook；无 thread、callback、allocation | 从 Time clocks 分离，使只需 timestamp 的 Platform 不取得 intrusive closure；全序/执行 policy 仍归 consumer |
| `B-F07 SPLIT_PRIME` | `platform_host` / Platform | `UTILC-A-05, PLAT-A-01, PLAT-A-03..12, PLAT-A-31..33` | SIMD target facts、Mem VM/NUMA、backend loader、secure seed caller | Core/Error 上的 OS facts与冷机制；不携带 thread wait 的 Time type或 filesystem surface | 拥有 CPU/topology/NUMA/raw VM、DynLib、environment和 secure entropy事实；SIMD只消费事实，不能在这里建立第二个 kernel dispatcher |
| `B-F08 SPLIT_PRIME` | `platform_thread` / Platform | `PLAT-A-02, PLAT-A-15..20` | Sched workers、Compute poller、长寿命 native service和 extended-ISA kernel thread | native Thread、current-thread ISA state、wait/wake/Parker 与 priority；time dependency只由 timed wait带入 | host factor报告静态 facts，本 factor拥有每线程 VL/AMX/SME enable/restore；fiber-aware wait属于 Sched，process lifecycle不塞入 Thread facade |
| `B-F09 SPLIT_PRIME` | `platform_fs` / Platform | `PLAT-A-21, PLAT-A-23..26, IO-A-28` | model/cache loader、IO download/async adapter、工具与配置读取 | synchronous FileHandle、namespace/temp/atomic replacement；不携带 Proactor、TLS或HTTP | Platform唯一拥有同步 namespace与cursor机制；IO只适配异步 operation，mapping/residency移交 Mem VM |
| `B-F10 SPLIT_PRIME` | `mem_core` / Mem | `MEM-A-01..05, MEM-A-07..08, MEM-A-20..22` | Container、Registry、Sched/IO allocator injection、普通 PMR/STL 用户 | alignment、AllocatorRef、system heap、PMR、quota/stats；mimalloc是 build-only mechanism；不依赖 Platform VM | 唯一拥有 host allocation/resource vocabulary；decorator 只由 opt-in owner支付，不把 pool/device/mapping塞进最小 allocator闭包 |
| `B-F11 SPLIT_PRIME` | `mem_pool` / Mem | `MEM-A-06, MEM-A-09..19` | request/parser arenas、object churn、worker-local cache、embedded fixed storage | Heap/Region/fixed/shared/object/growable pool；每种 metadata和 concurrency由具名类型承担 | variable/fixed、individual/bulk、reconstruct/preserve 语义按类型分开；不是 benchmark 的 fast/light 档位 |
| `B-F12 SPLIT_PRIME` | `mem_vm` / Mem | `PLAT-A-14, PLAT-A-27..29, MEM-A-23..25` | model mapping/residency、Sched fiber stacks、shared-memory transport | over Platform raw VM/fs mechanisms；HugeRegion、mapping family、guarded stack和 residency policy独立交付 | 消除 Platform MappedFile、Mem HugeRegion与 IO SharedMemory 的三个 storage owner；namespace/async notification仍由 IO |
| `B-F13 NEW_PRIME` | `mem_buffer` / Mem | `MEM-A-26..31` | Compute backend provider、IO transfer、inference loader/planner、工程 accelerator | provider-neutral ResourceRef、Buffer/View、sync/async seam、pool和 tensor-free LifetimePlanner；不得反向依赖 Compute | Compute/IO只实现 provider adapter；device/tensor placement policy不进入 Mem，闭合 enum不成为扩展点 |
| `B-F14 NEW_PRIME` | `mem_reclaim` / Mem | `MEM-A-32` | concurrent node/table Container 与 Obs sink snapshot | 独立 domain/guard/retire 生命周期；只有使用者支付 reader/retire metadata | intrusive link不负责 reclamation，Container不再各自保留到析构；具体 hazard/epoch机制由同一 factor裁决 |
| `B-F15 KEEP_PRIME` | `obs` log/trace seam / Obs | `OBS-A-01..15, OBS-A-24` | 所有库 call site、宿主 logger、CPU/GPU runtime tracing adapter | synchronous borrowed record；trace OFF为零产品依赖，Tracy是可选 build-only backend | log与trace共享 event/correlation schema且 disabled cost已闭合，不为 viewer knob拆产品；metrics和crash artifact另行处理 |
| `B-F16 NEW_PRIME` | `metrics` / Obs | `OBS-A-16..20, OBS-A-22, IO-A-51` | inference server、Sched/IO/runtime telemetry、普通工程服务 | owner-held counter/gauge/distribution + local merge/snapshot；无默认全局 registry，exporter由宿主拥有 | 与瞬时 trace plot分离，防止 trace OFF时统计消失；Platform/Mem/Sched/IO只提供事实，不重复存储/导出 |
| `B-F17 KEEP_PRIME` | `simd` / SIMD | `CORE-A-07, SIMD-A-01..11, SIMD-A-14..18, SIMD-A-20..23, SIMD-A-25..29, SIMD-A-33` | ordinary buffer用户、custom kernel作者、Container/Compute实现 | 一个不可拆分公开产品；whole-buffer边界最多一次 dispatch，expert/fixed/fat编译期路线无额外层 | Platform只给 facts；Highway/xsimd/HYLE是内部 mechanism；SIMD是自身 target/kernel route唯一 decision authority |
| `B-F18 KEEP_PRIME` | `math` checked scalar / Math | `MATH-A-01..02` | shape/stride/size schema与任意低层 checked arithmetic caller | 极小 header-only O(1) helper；标准可完整表达的能力不包装 | 不吸收 SIMD、linear algebra、统计、tolerance或模型运算；小不是并入 Util 的理由，独立 failure语义和消费者已闭合 |
| `B-F19 SPLIT_PRIME` | `container` sequential storage/algorithms / Container | `CONT-A-02..06, CONT-A-09..11, CONT-A-24..26, CONT-A-33, CONT-A-36` | tensor/token metadata、ordered/hash index、bitset、cache storage | allocator-aware owning structures；无并发/reclamation/wait metadata；compiled hash/algorithm mechanism可私有 | 标准容器足够的旅程直接路由；mutable heap和LRU只拥有通用 storage/lifetime，不吸收 timer/backend policy |
| `B-F20 SPLIT_PRIME` | `container_concurrent` / Container | `CONT-A-14..17` | backend registry、runtime maps/sets、并发 append state | accessor/publication合同 + Mem reclamation；不携带 scheduler wait/cancel | 与 sequential容器分开，stable-node和table retirement由 Mem reclamation统一，不为 set伪造 mapped值 |
| `B-F21 SPLIT_PRIME` | `container_queue` / Container | `CONT-A-07, CONT-A-18..23` | Sched ready/work-steal、IO/pipeline handoff、普通 bounded/unbounded queue用户 | try-only storage mechanisms；bounded/unbounded/role/priority为真实类型差异，无 CV/deadline/cancel | Sched拥有 blocking/backpressure/close service；队列不再迫使 Sched取得完整 text/map/bitset Container闭包 |
| `B-F22 SPLIT_PRIME` | `text` / Container | `CONT-A-28..32` | IO protocol/header、tokenizer/parser、path与intern metadata | borrowed UTF-8/byte view、codec、lexical path和interner；simdutf/Highway可作 build-only mechanism | 文本语义归 Text而非“任何用了SIMD的算法”；不复制 owning string，Unicode错误与转码保持 typed shape |
| `B-F23 SPLIT_PRIME` | `registry` / Util-runtime | `UTILR-A-01..09, UTILR-A-14` | backend/op/factory catalogue和一般 composition root | startup→freeze→read one-way factory publication；header/template交付，不携带 demangler artifact | exact name→factory是唯一语义；device scoring、module unload、auto registration与业务 alias policy不上浮进 Registry |
| `B-F24 SPLIT_PRIME` | `symbol` ordinary C++ demangle / Util-runtime | `UTILR-A-15..17, UTILR-A-20` | logs、tests、debug UI和普通 diagnostic tools | compiled owning/caller-buffer demangle；不依赖 Registry、Mem或Container，不承诺 signal safety | 从 Util aggregate分离，避免只要 demangle就拉入 registry closure；crash symbolization是另一生命周期 |
| `B-F25 KEEP_PRIME` | `sched` execution runtime / Sched | `ERROR-A-12, TIME-A-09, TIME-A-16, TIME-A-18..19, OBS-A-21, SCHED-A-01..35, SCHED-A-38..40` | inference request runtime、IO coroutine/timeout、一般 CPU async用户 | worker/Fiber/coroutine/task/cancel/select/timer/sync/channel同一 suspension生命周期；策略metadata按 opt-in类型 | Time只给 clock/queue，Container只给 storage，Obs只消费 context；sync primitives不能脱离 scheduler progress合同假拆 target |
| `B-F26 SPLIT_PRIME` | `io_core` completion/file/net/IPC / IO | `PLAT-A-22, IO-A-01..21, IO-A-23..27, IO-A-29, IO-A-31` | async file、TCP/UDP/local endpoint、RPC transport、server runtime | one Proactor/IoContext completion runtime；typed Endpoint/Error/operation budget；不携带 TLS/HTTP依赖 | Platform保留同步 fs/native mechanism，Mem保留 storage；socket/file共享 completion ownership所以不按 syscall拆组件 |
| `B-F27 SPLIT_PRIME` | `io_tls` / IO | `IO-A-32..34` | HTTPS/RPC TLS和普通 secure stream用户 | transport adapter + immutable config/session；OpenSSL是 mechanism，feature关闭时不构建/探测 | TLS不泄漏 OpenSSL类型；deadline/cancel沿 IO/Sched协议，不让 HTTP专有 ALPN成为通用 TLS合同 |
| `B-F28 SPLIT_PRIME` | `io_http` HTTP/1 + host/client/async body / IO | `IO-A-35..42` | inference HTTP/SSE server、普通 HTTP client/server | transport-neutral framing + async body + client/server host；pico parser为私有 mechanism；不默认拉入 H2 | routing/auth/JSON是应用；TLS通过 stream adapter组合；whole-request deadline/context不逐层复制 state machine |
| `B-F29 SPLIT_PRIME` | `io_http2` / IO | `IO-A-44..45` | 需要 multiplex/flow control 的 server/client | nghttp2私有协议状态 + HYLE async body；独立可选依赖和 capability | 不用一个 IO_HTTP2 bool改变整个 IO artifact；HTTP message/body语义复用 HTTP factor，nghttp2类型不外泄 |
| `B-F30 NEW_PRIME` | `io_transfer` reliable HTTP download/cache / IO | `IO-A-43` | model downloader、数据/asset cache与普通 large-file fetcher | 组合 HTTP async body、Platform atomic file、resume/ETag/range/checksum；独立隔离 filesystem/retry metadata | 不把模型格式或 cache policy塞入 HTTP connection；一个直接推理 journey加通用独立 consumer足以形成高层因子 |

`B-F01..30` 共覆盖 303 个 A 行。`container_intrusive` 仍是校准正例，但不是先验模板：TimerQueue、
Platform、Mem、Container、Util 和 IO 的上述拆分都必须同时产生独立 consumer、依赖/可选成本缩减和完整
lifecycle；相反，单个文件、函数或同依赖的 header 不因此成为 target。

#### B2. 完整 route 账本

以下 74 行不再制造 product factor，或必须先按语义拆开：

| ID / verdict | source A rows | final route |
|---|---|---|
| `B-R01 ROUTE_STANDARD` | `CORE-A-13, UTILC-A-03, UTILC-A-07, UTILC-A-10, UTILC-A-16, UTILC-A-19..20, TIME-A-11, SIMD-A-30, MATH-A-03..04, MATH-A-06, CONT-A-01, CONT-A-08, CONT-A-12` | C++23/标准容器/标准 distribution 已给出自然完整路线；HYLE不制造兼容层或同义 wrapper |
| `B-R02 DELIVERY_PROJECTION` | `CORE-A-11..12, ERROR-A-16, INTR-A-15, UTILC-A-21, TIME-A-20, PLAT-A-36, MEM-A-35, OBS-A-26, SIMD-A-34, MATH-A-08, CONT-A-37, UTILR-A-21, SCHED-A-41, IO-A-52, IO-A-54` | Build/package 是所有 factor 的唯一 delivery owner；版本 manifest、visibility、source/static/shared、feature和未来具名 Native FFI由横向合同投影，不拥有运行时语义 |
| `B-R03 ROUTE_UPPER_OWNER` | `TIME-A-06, PLAT-A-34, MEM-A-34, SIMD-A-12..13, SIMD-A-19, SIMD-A-32, MATH-A-07, CONT-A-13, UTILR-A-10, UTILR-A-13, SCHED-A-36..37, IO-A-49` | device event、tensor/quant/operator、graph dense-ID、backend catalogue、parallel algorithm与 RPC codec 属 Compute/Graph/engine；需求保留但这些 owner 仍冻结 |
| `B-R04 ROUTE_HOST_POLICY` | `OBS-A-25, IO-A-46` | secret/redaction、routing、auth、CORS、JSON和业务 middleware由宿主/application拥有；基础 seam 保留但不下沉 policy |
| `B-R05 DEFERRED_FACTOR` | `PLAT-A-13, PLAT-A-35, MEM-A-33, OBS-A-23, MATH-A-05, CONT-A-35, UTILR-A-11..12, UTILR-A-18..19, IO-A-48, IO-A-50` | executable memory、Process/module manager、secure memory、crash symbolization、consumer-specific arithmetic、matcher、foreign symbols、QUIC/gRPC/Web protocol和 RDMA 均有具名 owner/触发条件，但当前证据未准入公开 component |
| `B-R06 DECOMPOSED` | `CORE-A-10, ERROR-A-13..14, INTR-A-14, UTILC-A-11, UTILC-A-13, PLAT-A-30, SIMD-A-24, SIMD-A-31, CONT-A-27, CONT-A-34, IO-A-22, IO-A-30, IO-A-47` | A 行包含多个语义 owner；按下表拆开，不把 composite family 强塞给一个模块 |
| `B-R07 USER_GUIDE_PROJECTION` | `IO-A-53` | 每个 IO factor拥有自己的场景、lifetime、失败和陷阱说明；跨 factor选择页属于 IO产品指南，不归 Build/package或某一个 binary owner |

`B-R06` 的最终唯一权威如下：

| source row | decomposition and unique semantic owner |
|---|---|
| `CORE-A-10` | spin pause/backoff 归 `util_core`；streaming store与 kernel fence归 `simd`/owning kernel；persistence cache flush归 `mem_vm` 或持久化 owner；Core 不拥有万能 machine-action API |
| `ERROR-A-13` | owning parser/backend 保留 typed `E`；`error` 只拥有 canonical/native projection；`obs` 只拥有 structured observation，不给 Status加常驻 payload map |
| `ERROR-A-14` | 每个 owning exception boundary负责捕获和精确 mapping；只有完全同语义在两个 owner重复后，`error`才提供 cold no-throw helper，不建立 blanket “所有 exception 转 Status” policy |
| `INTR-A-14` | `container_intrusive` 只承诺 external synchronization与 atomic publish类型；reader-safe retirement归 `mem_reclaim` |
| `UTILC-A-11` | `mem_core`提供 fallible allocate/deallocate，`error`提供 Result，最终对象模块拥有 construct/destroy factory；Util不造跨 owner allocator factory |
| `UTILC-A-13` | 普通 thread mutex/RW路线优先标准库，OS wait mechanism归 `platform_thread`，可挂起锁归 `sched`；`util_core`只留短 spin/backoff |
| `PLAT-A-30` | shared mapping/residency owner归 `mem_vm`，name/handle transfer/async notification归 `io_core`；Platform只保留 raw OS mechanism |
| `SIMD-A-24` | 普通 contiguous sum scan留 `simd`；keyed/segmented storage algorithm归 Container，parallel/device scan归 Compute |
| `SIMD-A-31` | hash/text/codec/compression/crypto由各自语义 factor拥有并可直接借 vendor SIMD；不能因实现用了向量就依赖或迁入 `simd` |
| `CONT-A-27` | 简单 sparse map组合归 `container`；generation arena归对象 owner；compressed rank/select只有新 consumer准入后才建立格式 factor |
| `CONT-A-34` | runtime hash归 `container`；稳定 checksum/fingerprint归 IO/format owner，且必须具名 wire/format版本 |
| `IO-A-22` | adopted async pipe endpoint归 `io_core`；child process/exit/job/signal生命周期留在 `B-R05` 的 Process候选 |
| `IO-A-30` | mapping owner归 `mem_vm`；namespace、handle transfer和 notification归 `io_core`，进程间同步由协议 owner组合 |
| `IO-A-47` | SSE归 `io_http` async body；WebSocket只有具名双向 consumer后才从 `B-R05` 建独立 protocol factor |

#### B3. mechanism、decision authority 与交付权威

| semantic factors | mechanism/artifact owner | sole decision authority | delivery owner |
|---|---|---|---|
| `mem_core`, `mem_pool` | locked mimalloc + HYLE typed owners | Mem 对 allocator/resource/pool shape、fallback和 trim作唯一决定 | Build/package 投影锁、许可证、offline/source/static/shared |
| `obs` | HYLE seam + optional locked Tracy | Obs 对 event schema与 trace enablement作决定；viewer policy不进入 HYLE | Build/package |
| `simd` | locked Highway、xsimd、HYLE winners与 ISA TUs | SIMD 对自身 operation/target route唯一裁决；Platform只报告 facts | Build/package，包括 expert/fixed/fat assets |
| `container`, `text` | HYLE algorithms及 locked Highway/simdutf/xxHash/wyhash等 build-only mechanism | 各语义 factor只裁决自己的 hash/text/container algorithm，不复用 SIMD产品 dispatcher | Build/package |
| `sched` | HYLE runtime + locked private fcontext C/assembly | Sched 对 worker、suspension、wake、timer和cancel state machine唯一裁决 | Build/package按平台选择私有 artifact |
| `io_tls` | OpenSSL transport mechanism | IO TLS 对 config/session/operation outcome唯一裁决；OpenSSL type不公开 | Build/package对 optional system dependency作显式 availability/failure投影 |
| `io_http` | HYLE message/body/host + locked private pico parser | IO HTTP 对 framing、limits、body/backpressure唯一裁决 | Build/package |
| `io_http2` | locked nghttp2 protocol core | IO HTTP2 adapter对 HYLE stream/body lifecycle唯一裁决，nghttp2维护 wire state | Build/package |
| standard/direct routes | C++23标准库或用户显式选择的成熟依赖 | 标准/owning consumer | 用户自己的交付；HYLE文档只说明 route |

GoogleTest、nanobench和 peer checkout 只属于测试/证据，不是产品 mechanism；Folly、oneTBB、Abseil、
Seastar、Marl 与 Boost 只有在某行明确选择 `BORROW` 后才成为交付输入，不能因参与发现就自动进入依赖图。
任一锁定 dependency 升级只重开上述受影响 semantic factor 与直接 consumer，不继承旧性能或交付证书。

#### B4. 跨域冲突和拒绝过度拆分

| conflict | frozen resolution |
|---|---|
| CPU/ISA discovery | `platform_host`拥有 OS/CPU/topology facts，`platform_thread`拥有 current-thread VL/extended-state enable，`simd`拥有 SIMD product dispatch；Container/Text 自己的算法 route 只消费 facts或编译期 vendor mechanism，不建立第二个 SIMD authority |
| clocks/timers/deadlines | `time`拥有时间值和时钟；`timer_queue`只管理 node；`sched`拥有等待、取消、周期和传播；IO复用同一 operation budget |
| file/mapping/shared memory | `platform_fs`拥有同步 namespace/FileHandle；`mem_vm`拥有 mapping/residency/storage；`io_core`拥有 async operation、namespace transfer和 notification；三者不得各自持有同一 native handle owner |
| allocator/resource/buffer | `mem_core`拥有 host allocator vocabulary，`mem_buffer`拥有 provider-neutral storage，Compute只实现 provider并拥有 tensor placement policy |
| links/reclamation/concurrent container | `container_intrusive`拥有 membership，`mem_reclaim`拥有 reader-safe retirement，`container_concurrent`拥有 map/accessor semantics |
| spin/native/fiber synchronization | `util_core`拥有短 spin/backoff，`platform_thread`拥有 OS thread/wait，`sched`拥有可挂起同步与 progress |
| diagnostics/context/metrics | Core只 fail-fast，Error只传失败，Sched传播 task context，Obs记录 event，Metrics聚合值；没有一个“全能 telemetry/error context”对象 |
| Registry/module/plugin | `registry`只做 one-way name→factory；`platform_host`只做 DynLib mechanism；卸载/quiescence等待未来 Module factor，不给 Registry增加不安全 erase |
| network/protocol/application | `io_core`、TLS、HTTP、HTTP2、transfer分别闭环；routing/auth/model policy在宿主，QUIC/gRPC/RDMA等按具名 consumer准入 |

以下拆分被明确拒绝：Core 每个 attribute、Util-core 每个 header、Container 每个 class和 Platform 每个 OS
function 都没有独立 target收益；Error 的 Status/Result共享同一 failure representation；Obs 的 log/trace共享
零成本 seam；Sched 的 sync/cancel/timer共享 suspension lifecycle；IO 的 socket/file共享 completion runtime；SIMD
受不可拆分产品决策约束。它们可有私有 source/object分区，但不是公开组件。

#### B5. Stage B 机械裁决与冻结

主账本共有 30 个 factor、303 个来源行；route账本共有 7 类、74 个来源行。按上述闭区间展开后，
`CORE/ERROR/INTR/UTILC/TIME/PLAT/MEM/OBS/SIMD/MATH/CONT/UTILR/SCHED/IO` 的 377 个 A ID
全部恰好出现一次：缺失 0、重复 0、未知 owner 0。14 个 composite A 行又在 `B-R06` 中完成语义拆分。

据此裁决：**阶段 B 已冻结，阶段 C 尚未开始**。冻结的是 semantic owner、30 个候选质因数、route 与
negative space；不是 CMake target、依赖可见性、目录迁移、FIX 集或模块 reopen。现有 umbrella 名称是否
保留为纯 convenience target、每个 factor 的 component ID、公开/私有/build-only 边、旧 API迁移和真实
施工拓扑都必须由 C 阶段从本账本导出，不能回头用当前图删除已经冻结的能力。

### C. 依赖图与施工顺序

为每个因子记录 public semantic/type、private mechanism、build-only vendor、consumer 与唯一 decision
authority 五类边。消除 owner 错位、机制类型泄漏和重复决策后，输出无环中央组件图、旧到新迁移表及
拓扑施工顺序。只有此后才为首个组件建立 `SCOPED/AUDITED` 的有限 FIX 集并修改产品代码。

## 停止与裁决

- A 阶段每个能力行有来源、完整旅程、自然形状、覆盖状态和准入/拒绝理由后停止，不求 peer API 并集；
- A 未完成时，不讨论最终依赖；B 未完成时，不决定实现顺序；C 未完成时，不开始模块重构；
- 发现 correctness/security/build blocker 时建立独立有限任务，不借机提前改产品架构；
- 本记录完成时应给出 A/B/C 三项明确产物；缺少 material 输入则记录具名 blocker，不以“继续研究”结案。
