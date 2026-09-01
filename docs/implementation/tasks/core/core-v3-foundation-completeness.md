# Core v3 基础能力补全

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本轮由全库能力展开中的 `CORE-A-02/03/05/06` 新事实定向重开 Core。它不是对 `core-v2`
> 的无证据复查；旧轮仍是历史证据。本页保存本轮冻结 card、施工事实与 terminal receipt。

## Closure card

- component / owner / round ID：`core` / Core / `core-v3-foundation`；
- task kind：`REOPEN`；
- baseline commit：`0fc5fbc0cc8a86ef043788400e6305e5e6afa0e4`；
- 进入原因与 demand/defect ID：`CORE-FOUNDATION-COMPLETENESS-01`；全库能力展开证明
  `CORE-A-02/03/05/06` 仍为 `PARTIAL`，且 HYLE 已存在 Sched sanitizer 与 Mem allocator-return
  的局部重复；
- 真实 consumer task / normal workload：HYLE 的 Mem、Container、Sched、Platform 等 native C++
  translation unit 在成功热路径使用 invariant/compiler/environment primitive；量化推理宿主在 fatal
  时携带尺寸、索引或对象地址，并可在终止前接收一个有界事件；
- 范围：补齐 bounded typed fatal context 与 failure-only observer seam；补齐 debug/instrumented 与
  always-on assertion 语义；只增加已有真实消费者的 compiler annotation；修正并统一当前 TU 的
  compile-environment facts；迁移本轮直接发现的 HYLE 重复与错误 owner；更新 Core tests、P0 codegen、
  三种交付和文档；
- 非目标：日志、回溯、signal handler、minidump 或 crash policy；可恢复 error；runtime CPU/topology；
  ISA runtime dispatch；`HYLE_HAVE_*` ISA 列表；spin/cache flush/streaming store；runtime version artifact；
  Native FFI/C API；稳定 C++ ABI；P1/P2 纳秒 benchmark；重审 Mem/Sched/Container 算法；
- 直接依赖：无 HYLE 依赖；标准库、compiler builtin 与 C stdio/abort 是平台合同；
- 直接消费者：所有公开头使用 Core 的 factor；本轮语义迁移重点为 Mem、Sched、Container 以及
  compile-environment 宏的现有 HYLE consumers；
- tier 与理由：`P0`。新增能力是 compile-time/failure-only foundation，正常成功路径没有可独立计时的
  用户任务；
- material margin / resource budget：timing `NOT_APPLICABLE`；always-on success path 仍只允许一次冷
  branch，debug-disabled path 必须没有 runtime 指令；成功路径不得 allocation、lock、syscall、observer
  atomic load 或常态 metadata；fatal 默认输出与 typed fields 有明确上限；
- 总调查预算 / 性能预算：4 小时 / 45 分钟；超出即按现有事实 `CLOSED` 或写出单一 blocker 后
  `BLOCKED`；
- 最多复现、机制诊断、产品修改次数：一次 release codegen replay、一次机制诊断、一次产品修正与重跑；
- machine profile：`W1-WIN`、`W1-WSL`、`M1-MAC`；
- instrument profile：`closure-correctness-v1`；
- strong peers：
  - fatal context/observer：Highway `63f4b16cda024e2adff8a18dee81e067f8a87d61` 与 standalone ggml
    `d4716378882593333721eb33f153144b6885caf2`；
  - assertion modes：Highway `63f4b16cda024e2adff8a18dee81e067f8a87d61` 与 Abseil
    `b7adf83dccc73cd897fa6938978d96dce63a5d52`；
  - compiler/environment：Highway `63f4b16cda024e2adff8a18dee81e067f8a87d61` 与 Folly
    `eeac00030b8408cf5de8f7123967db86582b2857`；
- 北极星身份：llama.cpp `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0`，随仓 ggml tree
  `2e3f246882361e111128191d95ec3b6151d640a0`，standalone ggml 如上；均在 2026-09-01 fetch 后为 clean；
- 无同合同 peer 时的预算及禁止声明：P0 只裁决结构成本；禁止 Core-wide timing、全平台 SOTA 或
  “比 Highway/Abseil 更快”的声明；
- 使用原型 provenance：HYLE 内部为 `REAL_CONSUMER`；llama.cpp/ggml fatal 旅程为
  `NORTH_STAR_JOURNEY`；两库共同的 portability shape 为成熟 peer demand；
- API 语义档位与选择前提：recoverable failure 继续走 Error；昂贵内部检查走 debug/instrumented
  assertion；继续执行不可信的 invariant/hardening 走 always-on fatal；需要值上下文时显式附加少量
  typed fields；宿主 observer 只观察并返回，Core 随后必然终止；
- 每档常态成本合同：普通编译事实为零 runtime；禁用 assertion 为零 runtime 且不求值；always-on
  assertion 成功时一个预测冷 branch；fields、observer atomic 与 stderr 仅在 failure path；
- 适用 gates：normal、zero/boundary、borrow lifetime、observer concurrency/reentrancy、release codegen、
  public-header self-containment、Core source/static/shared consumer、Mem/Sched/Container 直接 consumer、
  Windows/WSL/macOS correctness、docs/usability、red team；OOM/cancel/backend timing 不适用；
- 预期输出：一个仍为 header-only、无 HYLE 依赖、无 binary 的 Core factor；有限 FIX ledger、候选、
  P0 manifest、六问、四项 LOC 与 terminal receipt；
- 停止条件 / 可接受限制：所有 `MUST` 为 `COMPLETE/ROUTED` 且 gates 通过即停止。Core observer 是
  failure-only callback seam，不承诺 signal safety、回溯、日志或跨动态装载单元的 crash-manager policy。

基线创建时工作树已有 Container/Mem 性能证据实验的无关修改；它们不属于本轮，不作为 baseline
产品输入，也不得随 Core 提交。

## Capability-family coverage map（双向审计完成）

| Family | admitted journey / evidence | required variants / lifecycle | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| invariant and fatal context | HYLE invariant users；llama.cpp/ggml formatted abort；Highway handler | always-on；static message/typed values；observer returns then termination；bounded failure work | frozen `FatalEvent/Field/Value` + context macro + image-local observer seam | `MISSING -> FIX` | logging/backtrace/signal/cross-DSO crash-manager policy不进入 Core |
| diagnostic assertion modes | HYLE expensive internal invariants；Highway DASSERT；Abseil hardening | debug、sanitizer/instrumented、always-on；disabled expression no side effect | frozen debug family + existing fatal family | `MISSING -> FIX` | 没有按 benchmark 命名的 fast/slow assertion；recoverable failure仍走 Error |
| compiler source contracts | Core fatal、Mem allocator redeclaration、borrowed failure fields | cold/no-tail、lifetime-bound、allocator-return；unsupported compiler safe fallback | frozen four narrow annotations in `compiler.h` | `PARTIAL -> FIX` | 没有真实 consumer 的 hot、nonnull、blanket sanitizer suppression明确不准入 |
| compile-environment facts | all platform-specific TUs；Sched TSan；multi-backend inference source | 0/1 OS/arch/compiler/build/sanitizer/exception/RTTI/endianness；pointer width；generic probes | frozen `<hyle/core/environment.h>` truth table | `PARTIAL -> FIX` | runtime CPU/ISA support不在此处；single-ISA SSSE3 特例移回 Container compile route |
| cache layout and prefetch | existing Container/Mem/SIMD users | current three layout constants and three intents | Core v2 | `COMPLETE` | runtime topology/residency仍在 owner module |

## Capability / gap ledger（`AUDITED` 冻结）

| ID | class | source | normal shape | owner | cost target | state | action | verification / negative space |
|---|---|---|---|---|---|---|---|---|
| `CORE3-01` | `MUST` | `CORE-A-02`、ggml/Highway | failure event with borrowed typed fields and optional observer | Core | success one branch；最多 8 fields、每段默认输出最多 256 bytes、failure no heap | `MISSING` | `FIX` | death/context/reentrancy/concurrency + codegen；无 backtrace/logging |
| `CORE3-02` | `MUST` | `CORE-A-03`、Highway/Abseil | debug/instrumented assertion plus existing always-on fatal | Core | disabled zero runtime/evaluation | `MISSING` | `FIX` | mode-specific compile/runtime probes |
| `CORE3-03` | `MUST` | `CORE-A-05`、Mem/Sched scan | cold/no-tail/lifetime-bound/allocator-return and sanitizer facts | Core | compile-only；no metadata | `PARTIAL` | `FIX` | compile/codegen；不加未消费 attribute |
| `CORE3-04` | `MUST` | `CORE-A-06`、include-order defect | consistent current-TU facts in the Core namespace/header | Core | preprocessor-only | `PARTIAL` | `FIX` | truth-table + three-toolchain compile |
| `CORE3-05` | `MUST` | `CORE-A-07` owner decision | remove Core SSSE3 special case; Container owns its compile route | Container semantic / Core delivery | no dispatch/cost change | `PARTIAL` | `FIX` | Container compile/tests；不扩成 ISA macro matrix |
| `CORE3-06` | `MUST` | Core v2 | layout/prefetch | Core | unchanged P0 contract | `COMPLETE` | `KEEP` | existing correctness/codegen |

### 现有 public surface 与重复扫描

- 现有 invariant、compiler、layout、prefetch 和 version/visibility 入口全部反向映射到上述 family；
  非 Core production/test/sample consumer file 数中，fatal 100、force-inline 36、noinline 12、
  no-unique-address 15、assume 11，其他窄 hint 也都有真实 consumer 或作为同族必要对偶保留；
- compile facts 当前跨 53 个左右的 HYLE source/header/test/sample 文件使用；旧宏只在 true 时定义，
  因此所有既有 `defined(HYLE_*)` consumer 必须与 0/1 合同一起机械迁移，不能只改 header；
- `<hyle/platform.h>` 的名字把 Core current-TU facts 与 runtime Platform 产品混在一起。本轮不保留
  compatibility forwarding header，统一迁到 `<hyle/core/environment.h>`；历史 task 仍记录旧事实；
- `HYLE_HAVE_SSSE3` 只有 Container probe 一个产品 consumer。它不是通用 current-TU fact，由
  Container 使用编译器 target fact在自己的 header route 中裁决；Core 删除该单项特例；
- Sched `fiber.cc` 私探 TSan，迁到 `HYLE_SANITIZER_THREAD`；Mem `system_inline.h` 私造
  allocator-return declaration，迁到 `HYLE_ALLOCATOR_RETURN(size_index)`；这两项不改变所属模块
  的算法、layout、failure 或 normal workload；
- 未发现第二套 public fatal、prefetch、layout 或 assume API。`HYLE_HOT`、returns-nonnull、通用
  alloc-size 双参数、no-sanitize suppression 没有当前准入 consumer，保持负空间。

### 本轮有限 FIX 集

1. `CORE3-01`：实现 borrowed `FatalEvent/FatalField/FatalValue`、最多 8 项 typed context、默认 bounded
   stderr formatter、thread-safe failure-only `set_fatal_observer()`；observer 返回后必然 `abort()`，递归
   fatal 跳过 observer 后走默认终止；
2. `CORE3-02`：实现 `HYLE_DEBUG_ASSERT[_MSG/_CONTEXT]`。Debug、sanitizer 或 analyzer build 启用；
   disabled build 不求值；`HYLE_FATAL_ASSERT[_MSG/_CONTEXT]` 继续作为 always-on hardening/invariant；
3. `CORE3-03`：在 compiler portability seam 增加且只增加 `HYLE_COLD`、
   `HYLE_NO_TAIL_CALL`、`HYLE_LIFETIME_BOUND`、`HYLE_ALLOCATOR_RETURN(size_index)`，Core 自己先消费前三项，
   Mem 消费最后一项；Sched 使用统一 sanitizer fact；
4. `CORE3-04`：以 `<hyle/core/environment.h>` 提供 generic feature probe，以及严格 0/1 的
   OS/arch/compiler/pointer/endian/build/sanitizer/analyzer/exception/RTTI facts；删除旧根 header并迁移
   直接 HYLE consumers、header ownership、LOC 与 package fixtures；
5. `CORE3-05`：删除 Core 的 SSSE3 特例，由 Container 保留同一编译目标条件；补齐 correctness、
   release codegen、三交付、文档与负向检查。除此五项不再增加产品能力。

## API / usability freeze

- 状态：`FROZEN`；
- fatal types：`FatalKind`、`FatalValueKind`、`FatalValue`、`FatalField`、`FatalEvent`、
  `FatalObserver`；事件和字段都是只借用到 callback 返回，不允许 observer 保留 view；
- fatal construction：`fatal_field(name, value)`、`HYLE_FATAL_FIELD(expr)`、
  `HYLE_FATAL_NAMED_FIELD(name, expr)`；支持 bool、有/无符号 integral、enum、floating、object pointer、
  `nullptr`、C string 与 `std::string_view`，其他类型需调用者显式投影；
- fatal operations：`set_fatal_observer(observer)` 原子交换并返回前值；observer slot 是 header-only
  linked-image seam，不冒充跨 DSO crash manager。普通无 observer 路线仍输出 stderr；
- assertion families：`HYLE_FATAL_ASSERT`、`HYLE_FATAL_ASSERT_MSG`、
  `HYLE_FATAL_ASSERT_CONTEXT`；`HYLE_DEBUG_ASSERT`、`HYLE_DEBUG_ASSERT_MSG`、
  `HYLE_DEBUG_ASSERT_CONTEXT`；条件一次求值，context 只在失败时求值；
- compile environment：`<hyle/core/environment.h>`；现有 OS/arch/compiler 名称改为始终 0/1，并增加
  unknown/clang-cl、pointer width、endianness、debug/release、四 sanitizer/any、analyzer、exceptions、
  RTTI 与四个 generic feature probes；不再有 `<hyle/platform.h>` 或 `HYLE_HAVE_SSSE3`；
- compiler annotations：上述四个窄 macro；allocator-return 允许 null，只表达 fresh/nonalias 与可用的
  size 信息，不得用于 borrowed/realloc 返回；
- 场景选择：可恢复失败选 Error；昂贵内部一致性检查选 debug family；安全/状态不可继续选 fatal；
  需要动态值才选 context；runtime OS/CPU/topology 选 Platform；
- public dependency / delivery：`hyle::core` 仍是无 artifact 的 INTERFACE；三个交付形状安装同一组
  headers，shared 形状不制造 Core DLL；
- misuse diagnostics：超过 8 fields 编译失败；unsupported field type 编译失败；observer 声明
  `noexcept` 且不得恢复；临时 text 只在同步 failure callback 内借用；
- freeze identity：本 task 的 `AUDITED` 版本 / 2026-09-01；
- 冻结后允许的内部优化边界：failure formatter/observer implementation、compiler spelling 与
  target-specific no-op fallback；不得由 codegen 结果增加新 public macro、字段或 runtime artifact。

## Owner 六问直答

### 1. 第三方产品路径

| Dependency / locked identity | direct / indirect | class | owner / disposition |
|---|---|---|---|
| 无 | — | product | Core 的 selected product closure 只有 HYLE headers；semantic/mechanism/decision/delivery owner 均为 Core/build-package |
| GoogleTest `063de7e9578f82b369302001269680b4b1553359` | test configure only | test | test mechanism；tests-off source/static/shared consumer 不取得它，不泄漏 public type |
| Highway `63f4b16cda024e2adff8a18dee81e067f8a87d61`、Abseil `b7adf83dccc73cd897fa6938978d96dce63a5d52`、Folly `eeac00030b8408cf5de8f7123967db86582b2857` | no link | peer | 只用于 fatal/assert/compiler shape 对账；不是 Core mechanism/artifact |
| llama.cpp `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0`、随仓 ggml tree `2e3f246882361e111128191d95ec3b6151d640a0`、standalone ggml `d4716378882593333721eb33f153144b6885caf2` | no link | north star / peer | 只提供推理宿主真实旅程与 failure shape |

平台合同是 C++ standard library、compiler builtin/attribute、C stdio 与 `abort()`；它们不冒充第三方
HYLE 模块。`git ls-tree`、selected `core` configure 和三交付安装清单共同反证 Core 没有 vendored、
integrated 或 FetchContent 产品输入。第三方升级不会改变 Core，除非新的 peer/consumer 事实触发重开。

### 2. 推理引擎能力完整度

llama.cpp、随仓 ggml 与 standalone ggml 均在 2026-09-01 fetch 后以本卡冻结的 clean identity 审计：

| Real journey / foundation difference | admitted demand | HYLE route / owner | coverage |
|---|---|---|---|
| tensor/kernel invariant 带 index、size、pointer 后终止 | bounded typed fatal event + host observation | Core fatal context/observer | `COMPLETE` |
| expensive internal validation 与 release hard invariant 分级 | debug/instrumented + always-on family | Core assertion modes | `COMPLETE` |
| 多 backend/target TU 的 compiler、OS、arch、sanitizer 与 build facts | numeric current-TU facts；consumer-backed annotations | Core environment/compiler | `COMPLETE` |
| cache-aware container/kernel 的单地址提示与静态 layout | three intent prefetch + interference constants | Core primitive；收益由 consumer owner | `COMPLETE` |
| recoverable backend/input/resource failure | 不能用 fatal 冒充返回值 | Error | `ROUTED` |
| runtime CPU/topology/NUMA 与 kernel dispatch | 非 compile-site fact | Platform / SIMD | `ROUTED` |
| logging、trace、backtrace、signal/minidump/crash policy | 不属于 prime foundation | Obs / future crash owner | `OUT_OF_SCOPE` |

当前证据下未处置 `MUST = NO`。这只表示本轮准入的推理基础旅程全部 `COMPLETE/ROUTED`；不承诺未来
消费者永不提出新能力，也不按 ggml/peer API 数量求并集。

### 3. 拟真、公平性能

- tier / normal task：`P0`；成功 invariant、禁用 debug check、compile environment/annotation 与单地址
  cache primitive，交付形状为实际 source/static/shared C++ consumer；
- candidate / machines / instrument：`cdd6b1c612c8e318220972fa0936f23e28045447`；W1-WIN、W1-WSL、
  M1-MAC；`closure-correctness-v1` 的 optimized codegen、symbol、sanitizer 与真实 direct consumer；
- verdict：`P0 STRUCTURAL PASS + TIMING NOT_APPLICABLE`；success context 是一个冷 branch，disabled
  debug 是零 runtime，compile facts/attributes 是零 registry/metadata，failure work 有 8 fields/256-byte
  segment 上限；
- 有限 claim：冻结工具链/交付上的结构成本满足上述合同；具体 hint 收益由 owning workload 裁决；
- 禁止外推：不声称比 Highway/Abseil/Folly 更快，不声称所有 compiler lowering、cache hit、下游吞吐或
  全平台 SOTA。完整证据见
  [`benchmarks-core`](../../evidence/performance/benchmarks-core.md)与
  [`manifest-cdd6b1c`](../../../../evidence/core/manifest-cdd6b1c.md)。

### 4. 旧 C ABI / Native FFI 清洁度

- 全局 policy 为 `N/A`，Core 没有准入 FFI slice；
- candidate 的 Core boundary 没有 public C header/schema、bridge source、未装饰 C export、C/managed
  sample、专用 option/target/package metadata或只为 FFI 支付的 layout/indirection；
- C stdio/`abort()` 是 private platform boundary，不是 HYLE C ABI；shared symbol audit 证明 detail fatal
  helper 不进入 COFF/ELF/Mach-O public export；
- shared package 只导入 C++ INTERFACE target，没有 Core DSO 或 decorated runtime API；
- verdict：`CLEAN`。

### 5. Source / static / shared 构建与导入

| Journey | verdict | component / artifact / relocation evidence |
|---|---|---|
| source embedding consumer | `PASS` | 三平台 parent `add_subdirectory` 只选 `core` 并链接 `hyle::core`，各 `1/1` |
| installed static consumer | `PASS` | 三平台 relocated exact-package consumer 各 `1/1`；header-only，无 Core archive |
| installed shared consumer | `PASS` | 三平台 relocated exact-package consumer 各 `1/1`；header-only，无 Core DSO/staging |

中央 graph 把 `core` 定义为无 HYLE dependency 的 prime factor，selected header 清单安装 8 个当前头；
static/shared 不制造虚构 compile definition、link closure 或 runtime loader 任务。三种 journey 都由
`HyleCiSelectedComponentSmoke.cmake` 验证，不以 producer configure 成功代替 consumer 导入。

### 6. C++ API 易用性与文档

| User journey | natural route | ownership / failure / concurrency | misuse guidance | verdict |
|---|---|---|---|---|
| hard invariant | `HYLE_FATAL_ASSERT[_MSG/_CONTEXT]` | failure-only borrowed values；observer 原子替换、同步返回后终止 | recoverable failure 走 Error；observer 不保留 view | `PASS` |
| expensive diagnostic | `HYLE_DEBUG_ASSERT[_MSG/_CONTEXT]` | Release disabled 且不求值；debug/sanitizer/analyzer active | 不把必要工作放在表达式/fields | `PASS` |
| compile target/config | `<hyle/core/environment.h>` numeric facts | 当前 TU，不代表 runtime machine | clang-cl frontend 与 Windows ABI 分开判断 | `PASS` |
| compiler source contract | `<hyle/core/compiler.h>` narrow macros | 前置条件由 caller 证明；unsupported safe fallback | 不为“也许更快”遍撒 hint | `PASS` |
| cache/layout primitive | `<hyle/core/cache.h>` | compile-time layout；prefetch 只是一地址 hint | runtime topology 走 Platform；策略归 consumer | `PASS` |

默认静态消息最简单，只有需要动态诊断值才 opt in typed fields/observer；没有 internal/benchmark seam
泄漏。architecture 解释边界，user guide 按“什么情况选什么”给出可复制 journey 与陷阱，本 task 保存
事实，performance 页只保存有限 claim。stale scan 只在历史 task/evidence 找到旧 `<hyle/platform.h>`/
C ABI，当前 architecture/user/sample/package 没有宣传已删能力。verdict：`PASS`。

## Candidate / verification / closure

### Candidate 1 construction

冻结的五项 FIX 已全部落地，未增加第六项产品能力：

- `CORE3-01/02`：`assert.h` 提供最多 8 个 typed fields、每段 256-byte 默认输出上限、同步
  failure-only observer，以及 always-on/debug-instrumented 两族 assertion；成功 codegen probe 直接使用
  context 形状，证明 fields 与 observer 不进入 hot path；
- `CORE3-03`：compiler portability seam 增加四个有 consumer 的窄 annotation。Mem 的 mimalloc fresh
  allocation redeclaration 和 Sched 的 TSan fiber route 已消费统一合同；
- `CORE3-04`：所有当前 HYLE source/header/test/sample consumer 已从删除的根 `<hyle/platform.h>` 迁到
  `<hyle/core/environment.h>` 的数值事实；CMake header owner、partial-checkout fixture、LOC 边界与 Core
  package sample 同步更新；
- `CORE3-05`：Container 的 fixed-target probe 私有地拥有 SSSE3 compile condition；Core 不再发布一条
  consumer-specific ISA macro。迁移触及的 Highway VQSort adaptation 仅改变 environment include/
  preprocessor spelling，故中央 integrated-source 内容锁同步刷新为
  `ce2a4185775362f4617caf5fe67dd358d61517e596641e89c9553beffb59ac1f`；origin、算法、route 与 license
  均未变化；
- closure contract 的单测不再假定 Core 永久为 `CLOSED`，而是从 registry 选取任一 terminal receipt
  验证 CLOSED-only 字段；checker 和状态机语义没有改变。

候选前诊断已在 W1-WIN/MSVC 与 W1-WSL/GCC 14 编译运行 Core 及 Mem/Container/Sched 直接消费者；
Windows 还编译运行迁移涉及的 Obs/SIMD/IO。GCC shared-symbol audit 曾发现
`write_fatal_event` 可见性遗漏，修正后通过；这属于候选冻结前的真实产品缺陷，不保留为限制。

首个候选 identity：commit `cbc2951ca34ffa961d9db8660b45f17eac2a0104`，tree
`21697748299939378a258a5c5aee02ef02582c15`。冻结时约定所有产品 gate 只验证这一不可变 identity；
下节记录它被拒绝并按状态机退回 `AUDITED`，没有把修复后的新 tree 冒充同一候选。

### Rejected candidate 1

`cbc2951ca34ffa961d9db8660b45f17eac2a0104` / `21697748299939378a258a5c5aee02ef02582c15`
在 W1-WIN 的 MSVC Release/Debug WERROR 全部通过，但 clang-cl 20.1.8 WERROR 在编译阶段拒绝：

1. `[[clang::lifetimebound]]` 放在 array-reference declarator 之后，被解释为作用于 array type 而非
   parameter；普通 `std::string_view`/pointer 参数没有该问题；
2. formatter bound test 用 `HYLE_COMPILER_MSVC` 选择 `tmpfile_s`，但 clang-cl 是 Clang frontend +
   Windows CRT ABI，错误进入 deprecated `tmpfile()` 分支。

这两项是 `CORE3-03/04` 内的可移植性缺陷，不改变 frozen API、算法、normal workload 或成本合同。
按 `CANDIDATE -> AUDITED` 退回后，只允许修正 attribute placement 与 test ABI condition，再建立新的
commit/tree；首个候选不得复用或写成 PASS。

### Candidate 2 remediation

退回 `AUDITED` 后只改动两处已授权的可移植性缺陷：char-array overload 不再对编译器已知生命周期的
array reference 重复施加 `lifetimebound`，formatter test 则按 `HYLE_OS_WINDOWS` 选择 Windows CRT。
修复后的 mutable tree 已通过 W1-WIN clang-cl 20.1.8、MSVC 19.50 和 W1-WSL GCC 14 的 Core
Release WERROR 全套测试；公共 API、failure formatter、success path codegen 与候选 1 的成本合同均未
改变。candidate 2 冻结为 commit `cdd6b1c612c8e318220972fa0936f23e28045447`、tree
`3855da3095dcd52417aeb46b8822346755b28abd`；所有正式 gate 只验证这一不可变 identity 并从头重放。

## Verification manifest

完整平台/命令边界记录在
[`manifest-cdd6b1c`](../../../../evidence/core/manifest-cdd6b1c.md)；下表不把未运行项写成 PASS。

### Correctness risk matrix

| Risk | verdict | command / evidence | limitation / reason |
|---|---|---|---|
| normal | `PASS` | immutable candidate：Windows MSVC/clang-cl、WSL GCC、macOS AppleClang Core `18/18` | 只覆盖冻结 public contract |
| empty / zero | `PASS` | zero fields/message、null pointer、disabled expression 与 header tests | empty observer 仍走默认 terminal formatter |
| capacity / boundary / overflow | `PASS` | 256-byte formatter bound；scratch compile：8 fields WERROR 成功、9 fields 精确 static-assert 失败；unsupported type 精确 compile 失败 | field/text 上限是 diagnostic resource contract |
| OOM / system failure / invalid input | `NOT_APPLICABLE` | success path 不分配；failure 必然终止；可恢复输入/资源失败由 Error | 不制造假的 OOM recovery |
| move / destroy / lifetime | `PASS` | borrowed string/pointer field tests、同步 observer/death path、lifetime-bound 三前端 compile | views 只活到 callback 返回 |
| concurrency / reentrancy | `PASS` | 2 threads × 20,000 observer exchanges；recursive observer fallback；WSL TSan | observer 是 linked-image seam，不是跨 DSO manager |
| cancellation / close | `NOT_APPLICABLE` | invariant terminates；observer set/exchange/null reset 已覆盖 | 无异步 service lifecycle |
| property / differential / fuzz / stress | `PASS` | deterministic typed-value/formatter cases与 40,000 次 atomic exchange under TSan | 未运行 coverage-guided fuzz，不作该 claim |
| ASan / UBSan / TSan | `PASS` | WSL GCC 14：ASan `5/5`、UBSan `5/5`；TSan Core `14/14` + release assertion `1/1` | TSan 按文档用 `setarch x86_64 -R` |
| real ISA / backend | `PASS` | x64 MSVC/clang-cl/GCC 与 Apple M4 arm64 AppleClang optimized codegen/environment truth table | runtime ISA/backend dispatch 不属于 Core |

### Aggregate gates

| Aggregate gate | verdict | evidence | limitation |
|---|---|---|---|
| correctness / sanitizer | `PASS` | 三平台 immutable Core、negative compile、ASan/UBSan/TSan | 无 signal-safety claim |
| component / source / static / shared | `PASS` | 三平台 selected Core smoke；每种 journey 各 `1/1` | Core 正确地没有 binary artifact |
| direct consumer | `PASS` | Windows/macOS WERROR 与 WSL normal policy 的 Mem/Container/Sched 各 `38/38` | 不签下游算法；WSL Container test-only GCC warning 单独披露 |
| P0 engineering performance | `PASS` | optimized codegen + symbol audit；`closure-correctness-v1` | timing `NOT_APPLICABLE`；无 SOTA claim |
| usability / docs | `PASS` | architecture/user/sample/package/stale API scan + `check-docs.py` | 历史 task 可保留旧名但不能冒充当前入口 |
| independent red team | `PASS` | candidate 1 被真实拒绝；candidate 2 重放全部 gates | 剩余 MEDIUM+ = 0 |

### Red-team findings

| Finding | severity / class | disposition |
|---|---|---|
| array-reference `lifetimebound` 在 clang-cl 被解释到错误 declarator | high / portability | `FIXED`；删除冗余 annotation，candidate 2 clang-cl WERROR `18/18` |
| formatter test 把 MSVC frontend 与 Windows CRT ABI 混为一谈 | high / test portability | `FIXED`；按 `HYLE_OS_WINDOWS` 选 `tmpfile_s` |
| hidden fatal helper 曾泄漏 ELF shared symbol | high / delivery | 候选 1 前 `FIXED`；三平台 symbol audit PASS |
| Windows worktree 的 `.git` pointer 在 WSL LOC test 不可解析 | medium / fixture | `DISPROVED` 为产品问题；WSL native candidate clone 完整 `18/18` |
| WSL GCC WERROR 在 Container `SmallVector` test-only 发出既存 array-bounds diagnostic | adjacent / consumer | Core 三个产品 target 已编译；normal consumer suite `38/38`；不越界修改 Container，也不写成其新证书 |

### Candidate LOC

命令：

```text
python scripts/loc-report.py --module core --commit cdd6b1c612c8e318220972fa0936f23e28045447 --self-check --json
```

| Metric | code |
|---|---:|
| 全库代码 | 413,981 |
| 自实现包含测试 | 99,891 |
| 自实现不包含测试 | 62,143 |
| Core 实现 | 763 |

`cloc 2.08` 对 1,339 个 tracked files 的 immutable tree self-check 为 `pass`；Core canonical boundary 是
`include/hyle/core/**`、`include/hyle/version.h`、`include/hyle/visibility.h` 与 `src/core/**`。

## Verification verdict

不可变 candidate `cdd6b1c612c8e318220972fa0936f23e28045447` / `3855da3095dcd52417aeb46b8822346755b28abd`
的 scope、architecture、correctness、delivery、usability/docs、red-team 与 P0 engineering-performance
七项 aggregate gate 均为 `PASS`。正式验证没有再改产品代码、公开合同、正常成本或 candidate identity；
本轮没有未处置的 `MUST` 能力、correctness/security blocker 或需要继续研究的性能点。

## 接受限制与重开条件

接受限制：fatal text/fields 为同步 borrowed views；observer 仅 linked-image、failure-only，不承诺
async-signal safety、backtrace、日志或跨 DSO crash policy；compiler hints/attributes 可在不支持的 target
退化，且 semantic promise 仍要求 caller 证明前置；静态 cache/layout 不等于 runtime topology；P0 证据
不签任何 consumer throughput 或全平台 SOTA。

只有下列新事实重开 Core：

1. 真实消费者暴露 Core 边界内 correctness/security、缺失 foundation 或难用 API；
2. public API、正常成本、layout、compiler/environment 语义、组件/交付边界改变；
3. 支持的 compiler/target 使冻结 codegen、attribute、truth table 或 symbol contract 失效；
4. 强 peer 证明更优复杂度档，或同合同正常 consumer 证明 material regression/advantage；
5. 当前明确路由到 Error/Platform/SIMD/Obs 的能力改变 ownership，形成新的具名 Core demand。

新 benchmark、API/LOC 数量、命名偏好、亚纳秒点值或历史任务命令句本身不构成重开。

## Closure decision

本轮裁决为 `CLOSED`：能力 ledger 没有未处置的 `MUST`，七项 aggregate gate 全部 `PASS`，没有
correctness/security、交付或本 tier 性能 blocker。候选仍是
`cdd6b1c612c8e318220972fa0936f23e28045447` / `3855da3095dcd52417aeb46b8822346755b28abd`；
终态文档提交只保存 receipt、限制、重开条件与 LOC，不冒充新的产品 candidate。后续 registry anchor
提交只回填本 receipt 的真实 Git identity，不授权继续修改 Core。
