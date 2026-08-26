# Core v2 重建与收口记录

> **状态：`SIGNED`。** 本任务不继承已删除旧 closure 文档的绿色状态。建立账本时检查的基线为
> `87c831d228e4e2f7844e60a3e9fd167cef4992b9`；最终产品候选为
> `b241a228d3311f0faa65cde68123de7da3688c2f`，证据提交为
> `b9231054b8eaa2f46850cb8b203b6056ae71b6a7`。

稳定设计见 [`core-design`](../../../architecture/core-design.md)，用户写法见
[`user/core`](../../../user/core.md)，性能资格见
[`benchmarks-core`](../../evidence/performance/benchmarks-core.md)。

## 目标与范围

本轮从推理引擎/工程计算消费者反推 Core，不按原目录或旧 benchmark 形状冻结 API。审查范围包括：

- 旧 product identity、compiler macro、fatal assertion、cache layout/prefetch 与 visibility；
- `src/core` runtime、Core C/managed fixtures、tests、sample、component/package contract；
- 全库公开头的语法消费者，以及 raw builtin、alignment、prefetch、fail-fast 的语义重复；
- source/static/shared C++ 交付和 Native FFI 准入；
- 被边界变化直接触碰的 container_intrusive、util_core、error、obs、platform、time、mem、simd、
  container、util、sched、io、compute、graph。

Build/package 是相邻 owner：版本、visibility、组件图、安装和 runtime staging 改变时共同回审，
但不把这些职责塞回 Core。其他模块只修直接依赖/编译阻断，不在本任务宣称重新收口。

## Demand/capability ledger

| ID | Consumer task / normal shape | 处置与公开路线 | Owner / 复杂度 | 强 peer 与裁决 |
|---|---|---|---|---|
| CORE-INV-01 | HYLE hot/cold path 发现不可能继续的程序不变量，并可附静态上下文 | `MUST`：`HYLE_FATAL_ASSERT` / `HYLE_FATAL_ASSERT_MSG`，Debug/Release 都 fail-fast | core；成功 O(1)，失败 terminal | Abseil/Folly/标准 assert practice；可恢复失败仍走 error；消息只在失败时求值且借用 NTS |
| CORE-COMP-01 | 跨 MSVC/GCC/Clang 表达 inline/noinline/restrict/branch/loop/layout hint | `MUST`：单一 `<hyle/core/compiler.h>` | core；compile-time，hint 可退化 | Highway/Abseil/Folly 与 compiler docs；只保留标准 C++ 尚不能自然覆盖的 portability seam |
| CORE-OPT-01 | 已证明条件供 optimizer 消除不可能路径 | `MUST`：Debug 诊断、Release `HYLE_ASSUME/UNREACHABLE` | core；Release 预期零 validation cost | 违约 UB；表达式必须无副作用；拒绝把 optimizer promise 当输入 validation |
| CORE-LAYOUT-01 | 并发结构隔离独立写字段或打包共同读字段 | `GENERAL`：三项 interference 常量 | core；compile-time，无 metadata | oneTBB/标准 interference practice；runtime probe 归 platform |
| CORE-PREFETCH-01 | Swiss/vqsort 等拥有明确未来访问的 hot loop 发出 hint | `GENERAL`：read/one-shot-read/write intent | core primitive、consumer policy；O(1) | compiler builtin/Highway/Folly practice；不公开 cache-level selector |
| PKG-VERSION-01 | build、release manifest 与用户识别编译 headers | `BORROW/ADJACENT`：根 `<hyle/version.h>` 常量 | build/package；compile-time | runtime peer 有真实 artifact 才提供 linked query；Core version-only DSO 被拒绝 |
| BUILD-VIS-01 | compiled module 选择 static/shared export/import | `ADJACENT`：根 `<hyle/visibility.h>` | build/package | 与 compiler source semantics 分离 |
| CORE-FFI-01 | foreign runtime 调用 Core compiler/layout/prefetch | `REJECT/N/A`：没有可成立的 runtime task | 无 Core FFI | 动态调用不能改变调用方 TU codegen/layout；版本号不足以制造 ABI |
| CORE-INTR-01 | time 需要零分配侵入链表但不能依赖完整 container | `MOVED`：`hyle::container_intrusive` | container owner | 独立 INTERFACE target，已从 Core 迁出 |

### 公开入口反向追踪

| 公开入口 | Demand ID | 保留理由 |
|---|---|---|
| `HYLE_FATAL_ASSERT` / `HYLE_FATAL_ASSERT_MSG` | CORE-INV-01 | 全库程序不变量统一 fail-fast；消息版本提供零成功路径求值的静态上下文，不依赖 runtime error/logging |
| compiler.h 中 10 类 portability macro | CORE-COMP-01 / CORE-OPT-01 | 处理受支持 compiler 的真实语法/退化差异 |
| 三项 interference constants | CORE-LAYOUT-01 | 公开对象布局需要稳定 compile-target contract |
| 三项 `prefetch_*` | CORE-PREFETCH-01 | 多个真实 hot consumer 使用相同单地址 primitive |
| `hyle::kVersion*` / `HYLE_VERSION_*` | PKG-VERSION-01 | build/package/header identity，不是 runtime API |

删除项也完成反向追踪：`HYLE_NODISCARD` 可被 `[[nodiscard]]` 覆盖；copy-disable macro 可被显式
`= delete` 覆盖；public concat 只有宏实现使用；`HYLE_TRAP` 没有独立于 fatal invariant 的用户任务；
assume-aligned 使用 `std::assume_aligned<N>`；linked version 和 Core FFI 没有真实 runtime demand。

## 依赖与消费者账本

公开头扫描确认 14 个实际 consumer 直接使用 Core/root version contract：

| Consumer group | 直接 `PUBLIC core` | 代表语义 |
|---|---|---|
| 轻量叶子 | container_intrusive、util_core、error | compiler/invariant/detail contract |
| runtime 底层 | obs、platform、time、mem | compiler/invariant/cache contract |
| data/compute | simd、container、util | compiler/cache/prefetch contract |
| 组合层 | sched、io、compute、graph | 公开 header inline/compiler contract |

`math` 没有 Core public include，保留零边。中央 `HyleComponents.cmake` 既声明唯一依赖又生成真实
target link，并扫描公开头拒绝漏边或多余边。旧安装整棵 include tree、传递 include、所谓
target-neutral macro 例外都不再充当依赖机制。

语义重复扫描的裁决：

- 已确认 false-sharing layout 的真实消费者使用 Core 常量，没有以 raw 64/128 复制同一语义；
- Swiss/vqsort 的相同 prefetch 任务走 Core helper；vendor port 内部机制不冒充 HYLE public API；
- 19 个依赖旧 transitive macro include、14 个依赖 transitive type visibility 的公开头改为直接
  include，header-self-contained/组件合同负责防回归；
- 多处 `cpu_relax` 是 util-core 后续审查事项，不扩张 Core；
- linked version 没有 production consumer，删除而不是为 sample/benchmark 维持 runtime。

## 架构与 API 裁决

- [x] Core 改为零源码、零 artifact 的 `INTERFACE` component；
- [x] 合并旧 `macros.h`/`compiler_defs.h` 为 C++-only `<hyle/core/compiler.h>`；
- [x] 新增 `<hyle/core/assert.h>`，用 ODR-safe noinline、可用时 cold 且 hidden 的 sink 保持成功路径小；
- [x] 提供失败时才求值的 borrowed-message assertion，并迁移 Error `Result` 与 Container B-tree 的
  两处手写 `false && "message"`；
- [x] 删除可由标准 C++ 自然覆盖或无独立 demand 的 public macro；
- [x] `cache.h` 直接持有 target-family 常量与三个 intent，不再维护 C defs 镜像；
- [x] 产品版本迁到 `<hyle/version.h>`，删除 linked C++ query 与 version-only runtime；
- [x] visibility 明确留在 build/package，不算 Core capability；
- [x] Core Native FFI 定为 `N/A`，删除 Core C header、exports、tests、package/Java/.NET fixture；
- [x] legacy Error FFI 的 schema version 暂由 Error 自己拥有；本任务不借机宣称 Error FFI 已清理；
- [x] source/static/shared Core sample 统一为编译期版本、compiler 与 cache 的正常 C++ 旅程；
- [x] 所有真实 consumer 建立直接 `PUBLIC core`，并补公开头的 self-contained include。
- [x] Core target 从无实际消费者的 trace/testing 横切边退出；source 与 installed contract 均拒绝
  profiler/test gate 经 `hyle::core` 传递。

项目尚未发布，不保留旧路径/符号兼容层。HYLE 自身仍直接使用 C++ API，不绕经任何 FFI。

## 正确性、交付与性能门

以下表格均绑定同一 candidate；原始日志和哈希见
[`manifest-b241a22`](../../../../evidence/core/manifest-b241a22.md)。

| Gate | 必须覆盖 | 当前记录 |
|---|---|---|
| Windows | MSVC `/WX` Core-only Release/Debug、clang-cl Release、header self-contained、component contract | `PASS`；MSVC/clang-cl 各 6/6，Core label 5 项；Debug optimized codegen 按设计 skip |
| Linux | GCC Release + ASan/UBSan、Core-only；AArch64 cross compile/codegen | `PASS`；Release 6/6，ASan/UBSan 各 5/5；cross 不冒充真机运行 |
| macOS | AppleClang arm64 Release/Debug、header/package consumer | `PASS`；Release/Debug 各 Core 5/5 |
| source delivery | Core-only parent `add_subdirectory`，唯一产品组件 target 为 INTERFACE Core | `PASS` |
| static SDK | relocated installed consumer；没有 Core archive/runtime | `PASS` |
| shared SDK | relocated installed consumer；没有 Core DSO/runtime staging | `PASS` |
| codegen/symbol | inline/noinline/assume/prefetch/fatal hot path；shared export 不得泄漏 Core detail；普通 version value use 不链接 getter/runtime | `PASS`；9 functions、3 exact prefetch intents，parser 红队 14/14 |
| consumer performance | 正常 workload 按两把钥匙裁决 | Core 只签署 primitive 结构；具体收益留给 owning module，不制造纳秒榜 |
| docs | stale symbol/path scan、relative-link contract、四类文档职责 | `PASS`；设计、用户、性能、任务与 manifest 分工明确 |
| red team | API/能力/复杂度/依赖/交付/性能反证 | `PASS`；无剩余 MEDIUM+，限制与 out-of-scope 已登记 |

## 成熟库对账结论

- Highway、Abseil、Folly 与 compiler 文档支持“小 portability layer + 标准 C++ 优先”的形状；
  HYLE 不复制各库全部宏，只覆盖当前 compiler/consumer 需求。
- oneTBB/标准库的 interference/cache-aligned practice 支持稳定 layout recommendation；runtime
  topology 与静态 ABI layout 必须分开；Abseil/Folly 的 AArch64 64-byte 裁决也证明不能把 Apple
  arm64 的 128-byte coherency line 泛化到 Windows/Linux AArch64。
- oneTBB、ggml、llama.cpp 等 runtime version query 都附着于真实 runtime artifact。HYLE Core 没有
  runtime state，因此只为 query 造 binary 不能诊断整个产品。
- peer 的 prefetch API 都不能替拥有访问模式的 consumer 做策略决定；SOTA 证据必须回到正常
  workload，而不是单 hint benchmark。

## LOC、candidate 与签署字段

| 字段 | 值 |
|---|---|
| Candidate commit/tree | `b241a228d3311f0faa65cde68123de7da3688c2f` / `e73472e2cefedd70e41eca6243871b2343db7d74` |
| 全库代码 | 297,471（含本轮已跟踪 raw evidence） |
| 自实现包含测试 | 141,867 |
| 自实现不包含测试 | 96,105 |
| Core 实现代码 | 233；边界为 `include/hyle/core/**`、`include/hyle/version.h`、`src/core/**` |
| Raw evidence | [`evidence/core/manifest-b241a22.md`](../../../../evidence/core/manifest-b241a22.md)；证据提交 `b923105` |
| Red-team verdict | high/medium findings fixed；缺口反证完成；限制/out-of-scope 已接受；剩余 MEDIUM+ = 0 |
| End commit / push | `main`；最终 handoff 报告远端提交 |

## 剩余限制与停止规则

Core 不保证某 hint 被 compiler 接受、某 cache 层级命中、runtime 精确 cache-line size，或跨 compiler/
stdlib/build-mode ABI。`HYLE_ASSUME/RESTRICT/UNREACHABLE` 仍要求调用者证明前置条件，且 assume
表达式必须无副作用。fatal sink 使用 stdio/`abort()`，不承诺 async-signal-safe；出现真实 crash
reporter 消费者时由正确 owner 重开，而不是给常态断言热路径预付复杂度。

上述门已完成，Core v2 进入 `SIGNED`。此后只有真实 consumer 缺口、correctness/security、直接依赖
语义改变、新 compiler/target，或强 peer 在同合同正常 workload 证明 material advantage 才重开；
LOC、feature 数、命名偏好和亚纳秒 benchmark 不构成理由。
