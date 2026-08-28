# Core 能力完整性存量回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)回放到已经 `CLOSED` 的
> `core-v2`。这是 product-non-mutating 的存量审计，不把新版方法伪造成旧轮当时已经执行过，也不
> 替代 [`module-status.json`](../../module-status.json) 的当前状态。

## 回放合同

- component / replay：`core` / `core-capability-replay-v2`；
- reviewed state / round：`CLOSED` / `core-v2`；
- closure task：[`core-rebuild`](core-rebuild.md)；
- candidate / end / registry anchor：
  `b241a228d3311f0faa65cde68123de7da3688c2f` /
  `49512b77434c4f15fec355e4bd6d6ab5a8b9b650` /
  `1a465e23ea69a6e613840cf8d30d33054463ae1c`；
- actual replay commit：`2cc6ec26d80af9d7e69f36555f0ac0c9cf5dc6b1`；日期：2026-08-28；
- normal consumer：HYLE native C++ 模块和源码/静态/共享 SDK 用户，在自己的 TU 中使用 fail-fast、
  compiler/optimizer、cache-layout 和 prefetch primitive；
- tier / instrument：`P0 / closure-correctness-v1`。这些 primitive 没有可独立计时的用户任务，
  本轮 timing budget 为 0；
- 调查预算：30 分钟；一个 fresh Windows Core-only build、一次当前 source/static/shared selected-
  component smoke、两个预钉死 strong peers/能力族；
- strong peers：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 的
  `absl/base/{optimization,prefetch,attributes}.h`，Highway
  `239071390b18ca70f8fadde90ebefa6d8b12cb69` 的 `hwy/{base,cache_control}.h`；Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 只用于反向检查 HYLE 现有 consumer 的 portability
  拼写，不在同一裁决中充当第三个 peer；
- 范围：自顶向下能力族、全部现有 public surface、真实 consumer/重复实现、复杂度、三种交付、
  易用性与负空间；
- 非目标：新增宏、重跑亚纳秒 benchmark、签署 owning consumer 的收益、复审完整 build/package、
  `util-core` 或 container；发现相邻问题只登记下一 owner。

候选到 replay commit 的 Core 产品路径
`include/hyle/core/**`、根 `platform/version/visibility.h` 与 `src/core/**` 为零 diff。只有 harness 变化：
Core header test 删除已经不存在的 C-product 例外，package sample 改成 exact-version 查找。由于中央
component/package 系统在候选后演进，本轮仍重新跑了当前交付 journey，而没有只靠路径零 diff 推断。

## 自顶向下 capability-family map

| Family | 已准入用户旅程与来源 | 必须覆盖的变体、生命周期和边界 | 当前路线 | Coverage / action | 缺失边或负空间 |
|---|---|---|---|---|---|
| release-surviving invariant failure | Error/Container/Mem/SIMD 等真实 consumer；Abseil hardening | success/failure、无消息/借用 NTS、Debug/Release；条件一次求值，消息只在失败时求值，最终 `abort()` | `<hyle/core/assert.h>` 两个 fatal macro 与 hidden cold sink | `COMPLETE / KEEP` | 可恢复失败走 error；动态格式可由调用者预先形成 NTS；不是 logging/crash reporter，也不保证 signal-safe |
| portable compiler source contract | 真实 hot headers/TU；Highway/Abseil | inline/noinline、restrict、flatten、显式/default unroll、空成员布局、likely/unlikely；unsupported compiler 必须安全退化 | `<hyle/core/compiler.h>` | `COMPLETE / KEEP` | `[[nodiscard]]`、`= delete`、`[[maybe_unused]]`、`std::assume_aligned` 等已有标准形状不再包装；单库属性并不自动准入 |
| optimizer promise and impossible control flow | Mem/Container/SIMD/Error 真实 consumer；Highway/Abseil | Debug 可诊断；Release 提供 assume/unreachable；表达式无副作用，违约 UB，不能代替输入校验 | `HYLE_ASSUME`、`HYLE_UNREACHABLE` | `COMPLETE / KEEP` | 不承诺 compiler 接受 hint；没有运行时 validation 或 recoverable 结果 |
| compile-target cache layout | 并发 container、mem、sched 真实布局；成熟 interference practice | constructive/destructive/跨 family 保守上界；支持 x64、Apple/其他 arm64 和 unknown 64-bit fallback；进入 ABI | 三个 `inline constexpr` recommendation | `COMPLETE / KEEP` | runtime cache hierarchy/line observation 归 platform；不提供 L1/L2/L3 selector，不把 runtime 值放进静态类型布局 |
| single-address prefetch intent | Swiss/vqsort 真实 temporal-read consumer；Abseil 的相同三意图族及 compiler/ISA primitive | temporal read、one-shot/NTA read、write intent；任意意图都可 no-op，不读值、不延寿、不建立同步 | 三个 `prefetch_*` inline function | `COMPLETE / KEEP` | NTA/write 当前没有仓库生产调用，但其成熟、零状态、完整三意图形状不是 benchmark API；距离、收益和污染归 owning workload |
| build identity and binary decoration | source/static/shared package journey | target OS/arch/compiler、64-bit contract、header version、export/import/hidden；选择、安装、重定位 | 根 `<hyle/platform.h>`、`<hyle/version.h>`、`<hyle/visibility.h>`，语义和 delivery owner 均为 build/package | `ROUTED / KEEP`（就 Core 所需路线完整） | build/package 全域能力将在自己的 development-unit replay 裁决；Core 不制造 runtime version artifact |
| spin-wait hint and backoff policy | `util::SpinLock`、sched 真实 consumer；Highway `Pause` 与成熟锁实现 | 单指令 relax、bounded backoff、OS/fiber park 是不同层级；hint 不建立同步 | public `hyle::util::cpu_relax()` 和 owning synchronization 类型 | `ROUTED / BORROW-ADJACENT` | 不下沉 Core。container 两个 private adaptive lock 仍复制 ISA spelling，交给 `util-core/container` 回放裁决 owner/依赖，不据此重开 Core |
| streaming store/fence/cache flush | SIMD kernel、time counter serialization、mapped-device memory 等具名但不同语义任务 | vector width/alignment、memory ordering、device coherency、persistence 均由 owner 决定 | SIMD `Stream`、time/platform/compute 各自的受限实现 | `ROUTED / REJECT from Core` | 不把 Highway `cache_control` 的并列函数名误当成一个可移植 Core 用户旅程 |
| recoverable error、logging/crash、runtime topology、FFI | 相邻问题域或尚无外语 runtime task | ownership、格式、并发、取消、unload 都需要 runtime owner | error、obs、platform；Native FFI 按 accepted policy 为 `N/A` | `OUT_OF_SCOPE / REJECT` | 不为“最底层”把 runtime state、allocation、registry、锁或动态符号塞入 Core |

所有 `MUST` 都为 `COMPLETE` 或具名 `ROUTED`，不存在 `PARTIAL + KEEP`。表中 routed family 的完整性
只签署 Core 用户走到正确 owner 的路线，不提前替 build/package、util-core、SIMD 或 platform 收口。

## 自底向上 public-surface trace

| 当前入口 | Capability / demand | 生产反向证据 | 裁决 |
|---|---|---|---|
| `HYLE_FATAL_ASSERT[_MSG]` | invariant failure | 前者 90 个非 Core production 文件，后者 2 个；success/failure、消息惰性和 death path 有测试 | `KEEP` |
| `HYLE_FORCE_INLINE` / `HYLE_NOINLINE` | compiler source contract | 分别 59 / 6 个非 Core production 文件 | `KEEP` |
| `HYLE_RESTRICT` / `HYLE_FLATTEN` / 两种 `UNROLL` | compiler source contract | vqsort 正常 kernel；每项 1 个 production 文件 | `KEEP`；低调用数不等于 benchmark-only，属于同一成熟 portability seam |
| `HYLE_NO_UNIQUE_ADDRESS` | compiler source contract | 14 个 util/container production headers | `KEEP` |
| `HYLE_LIKELY` / `HYLE_UNLIKELY` | compiler source contract | 各 5 个 production 文件 | `KEEP` |
| `HYLE_ASSUME` / `HYLE_UNREACHABLE` | optimizer promise | 6 / 2 个 production 文件 | `KEEP` |
| constructive/destructive/max-destructive 常量 | cache layout | 0 / 5 / 5 个非 Core production 文件 | `KEEP`；constructive 是完整对偶 recommendation，且无常态 metadata，不因当前零调用删除 |
| temporal/NTA/write `prefetch_*` | prefetch intent | 2 / 0 / 0 个非 Core production 文件；三者有 public correctness 和 optimized codegen probe | `KEEP`；同一完整三意图族，不以 benchmark 数量保留 |
| `HYLE_VERSION_*` 与 `hyle::kVersion*` | build identity | root build、exact package sample、compile-time test | `ROUTED/KEEP` |
| `HYLE_OS/ARCH/COMPILER/BITNESS/HAVE_SSSE3` | build target identity | Platform/Time/Mem/SIMD/IO 等直接 compile-time consumers | `ROUTED/KEEP`；当前 product 只支持 64-bit target ABI |
| `HYLE_EXPORT/IMPORT/HIDDEN` | binary decoration | compiled modules 与 Core fatal sink；shared symbol probe | `ROUTED/KEEP` |
| `<hyle/core/detail/preprocessor.h>` | public-header implementation seam | Error result macro 与 util ScopeGuard header | 保持 installed `detail` dependency，不是受支持的用户 API |

仓库还扫描了 Core 之外的 vendor compiler spelling。唯一非算法/ISA分派的例外是 Mem 为了与 mimalloc
v3 的 MSVC public prototype 类型兼容而私有声明的 `HYLE_MEM_MI_ALLOCATOR_RETURN`；它是第三方
redeclaration seam，不构成一个跨模块 Core attribute family。SIMD vendored vqsort 的局部 `HWY_*`
拼写同样不扩张 HYLE public API。没有发现第二套 public assume/prefetch/layout/assert 实现。

## 当前验证

| Gate | 当前结果 | 证据边界 |
|---|---|---|
| fresh Windows Core-only Release/Werror | `PASS`，16/16 | VS 18 2026 / MSVC 19.50；含 Core correctness、C++ header self-contained、codegen 与 parser red team |
| source consumer | `PASS`，1/1 | generated parent `add_subdirectory()`，只链接 `hyle::core` |
| relocated shared SDK consumer | `PASS`，1/1 | installed `hyle::core` 为 INTERFACE，没有 Core DSO/runtime |
| relocated static SDK consumer | `PASS`，1/1 | installed `hyle::core` 为 INTERFACE，没有 Core archive/runtime |
| P0 cost structure | `PASS` | compile-time/inline O(1)；success path 无 allocation、lock、syscall、registry 或常态 metadata；fatal path 终止，prefetch 可 no-op |
| nanobenchmark | `N/A` | primitive 的收益/损害只能由 owning consumer workload 证明；不为亚纳秒点值改变 Core |

本回放没有重跑 Linux/macOS；候选的三平台证据仍是历史事实，本轮只证明当前 Windows 产品和当前
三种 delivery contract。Core 产品路径自候选未变，未出现新 toolchain/target 反例。

## 裁决与下一动作

- 产品 verdict：`CONFIRMED`；`core` 保持 `CLOSED`，实现改动为 0，没有 `PRODUCT_TRIGGER`；
- 方法 verdict：旧 task 的能力来源和 public trace 很强，但旧 `KEEP` 没有显式证明“完整”。本页补上
  family/coverage 两轴，不改写旧过程；
- 性能 verdict：仍为 P0 structural `PASS`；不新增 Core-wide SOTA 或 timing 声明；
- 下一动作：项目采用清单进入 `error`；到 `util-core`/container 回放时必须裁决 `cpu_relax` 的两份
  private duplication，不能无视，也不能借此把同步策略塞进 Core；
- 重开条件：supported compiler/target 使现有合同错误；真实 consumer 出现缺失的 runtime-before-
  runtime primitive；直接 consumer 的 correctness/codegen 失败；或 strong peer 在同语义正常 workload
  证明 material 优势。宏数量、peer API 并集、零调用互补项或亚纳秒差异不触发重开。
