# Core v3 基础能力补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由全库能力展开中的 `CORE-A-02/03/05/06` 新事实定向重开 Core。它不是对 `core-v2`
> 的无证据复查；旧轮仍是历史证据。本页是本轮唯一活动 closure card 与施工记录。

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
`21697748299939378a258a5c5aee02ef02582c15`。后续所有产品 gate 只验证这一不可变 identity；若公共
合同、算法或正常成本发生变化则退回 `AUDITED`，不得把新 tree 冒充同一候选。不可变 replay、gate
matrix、LOC 和 terminal receipt 在验证后填写。

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
