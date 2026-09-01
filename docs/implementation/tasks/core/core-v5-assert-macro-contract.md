# Core v5 断言宏合同修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮不是无新事实复查。Core v4 关闭后，针对公开开发者接口的独立红队得到两个可复现产品缺陷，
> 命中“真实消费者暴露 correctness/API-usability defect”和“支持编译器使冻结语义失效”重开条件。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`core` / Core / `core-v5-assert-macro-contract` / `REOPEN`；
- baseline：`b94b451e3209dc65a6887f0888dc0688b8c86391`；前轮：
  `core-v4-build-contract-projection`；
- trigger：`CORE-ASSERT-MACRO-CONTRACT-01`；
- `CORE5-R1`：`HYLE_FATAL_ASSERT_CONTEXT` 与启用态 `HYLE_DEBUG_ASSERT_CONTEXT` 在宏展开内部声明固定
  名称 `hyle_fatal_fields`。字段表达式若合法引用调用者的同名变量，会被新声明遮蔽并读到内部数组，
  已在 GCC 14 observer 复现为“期望 signed 7，实际得到 pointer”；
- `CORE5-R2`：禁用态 Debug 宏以 `if constexpr (false)` 包住表达式。在模板中它不实例化 dependent
  condition；简单版本只丢弃 `cond`，没有检查启用态真实使用的 `!(cond)`；context 版本也没有强制
  1–8 个字段。GCC 14 已复现 Release 接受而 Debug 拒绝 non-Boolean condition、无效 dependent
  expression 和 9 fields，AppleClang 21 同样接受禁用态 0 fields；
- peer mechanism：本地锁定 Abseil `b7adf83d` 使用 `decltype((void)(expr))()` 形成零运行时的
  unevaluated semantic check；只借用这一语言机制，不复制 CHECK 产品面；
- 初步受影响面：`include/hyle/core/assert.h`、Core assertion correctness/negative-compile/codegen 合同，
  以及 observer lifetime 与 disabled-expression 陷阱说明；
- 暂定 tier：`P0 STRUCTURAL`。缺陷是编译语义与宏卫生，不是计时问题；正常成功路径和禁用态必须不
  增加指令、allocation、lock、syscall、dispatch 或常态 metadata；
- 暂定 non-goals：新增 Core capability、日志/crash manager、recoverable error、任意格式化、C/Native
  FFI、纳秒 benchmark、相邻模块复审和 build-system 重写。

## Closure card（`SCOPED`）

- component / owner / round / kind：`core` / Core / `core-v5-assert-macro-contract` / `REOPEN`；
- baseline：`b94b451e3209dc65a6887f0888dc0688b8c86391`；trigger / evidence：
  `CORE-ASSERT-MACRO-CONTRACT-01` / 上述两组可复现反例；
- normal journey：C++ consumer 在任意合法局部命名下声明带 1–8 个 typed fields 的 invariant；Release
  consumer 写一次 Debug assertion，禁用时不求值但仍与启用态接受/拒绝同一表达式；失败 observer 在
  同步借用期内读取 event；
- scope：修复 context 宏卫生；统一 enabled/disabled Debug condition、message 与 fields 的编译语义；
  增加正/负 compile、运行时 observer 与 optimized codegen 回归；补充 observer callback 生命周期和
  disabled assertion 表达式陷阱；
- non-goals：新增 capability、比较式 CHECK 大全、日志/crash manager、recoverable error、任意格式化、
  observer RAII、跨 DSO 管理、C/Native FFI、相邻模块 review、CMake 产品图重写与纳秒 benchmark；
- dependencies / consumers：Core 产品仍无 HYLE/第三方产品依赖；全部包含 `assert.h` 的 C++ factors 是
  语法消费者，本轮以 Core 直接 journey、public-header closure 和中央 component compile contract
  覆盖传播风险，不改变任何依赖边；
- tier / cost contract：`P0 STRUCTURAL`；timing peer comparison `NOT_APPLICABLE`。成功态 Fatal 仍只付
  一个冷分支；disabled Debug 生成零运行时求值/指令；不增加 allocation、lock、syscall、dispatch、对象
  metadata 或 ABI/layout；
- budgets：总调查与实施 60 分钟、验证 60 分钟；既有复现之外最多一次机制确认、一次产品修改和一次
  完整重跑；不得扩建公共性能仪器；
- machines / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC` / `closure-correctness-v1`；
- strong peer / standard：Abseil `b7adf83dccc73cd897fa6938978d96dce63a5d52` 的 C++20+
  unevaluated-expression mechanism；C++23 `decltype`/requires 语言语义。peer 只证明机制，不准入新 API；
- provenance：两个 finding 为 `REAL_REPRODUCIBLE_DEFECT`；调用 journey 来自现有公开合同和遍布 HYLE
  factors 的真实语法消费者，不由 benchmark 生成；
- correctness gates：正常 success/failure、1/8 fields、0/9 fields 负编译、同名调用者标识符、非 Boolean
  condition、dependent invalid expression、message/field 单次或零次求值、observer borrow/reentrancy、
  public header self-contained、三平台 WERROR；不拥有资源，OOM/cancel/move/real ISA 均不适用；
- delivery gates：source embedding 与 installed static/shared Core consumers；Core 是 header-only，后两者
  验证同名 imported INTERFACE journey而不制造 archive/DSO；中央 component contract 重放；
- P0 gate：optimized object 中 disabled Debug probe 无执行指令，Fatal success path 保持冷分支且失败
  helper 不内联进热路径；不采纳纳秒点值；
- stop：有限 `FIX` 集全部完成并在不可变 candidate 上通过 correctness、delivery、docs/usability、
  independent red team 与 P0 structural gate即关闭；受支持 compiler 不能同时满足语义等价与零运行时
  成本时写单一 blocker，不继续发明宏姿势。

## Capability-family coverage map

| Family | admitted journey / evidence | semantic and lifecycle boundary | current route | Coverage | missing edge / negative space |
|---|---|---|---|---|---|
| always-on invariant + typed context | HYLE public consumers；llama.cpp/ggml 高频 invariant | condition once；message/1–8 typed fields failure-only；sync borrow；abort | fatal macro family | `PARTIAL` | fixed macro local can capture caller identifier；fix hygiene，不加入 printf formatting |
| debug/instrumented invariant | HYLE expensive internal consistency；Abseil assertion mechanism | enabled 与 disabled 接受同一 C++ expression；disabled zero evaluation/code | debug macro family | `PARTIAL` | disabled branch skips dependent/context semantics；fix unevaluated checking |
| host fatal observation | embedding host forwards bounded event before termination | image-local atomic slot；sync borrowed callback；recursive fallback；always abort | `set_fatal_observer` | `PARTIAL` | implementation complete；guide 未明确 plugin unload/quiescence responsibility |
| compiler/environment/cache/version foundations | Core v3/v4 admitted families and current public consumers | unchanged ownership、complexity、layout and delivery | existing headers | `COMPLETE` | runtime topology/ISA、logging、error、allocator remain adjacent owners |
| formatted abort / crash policy | ggml formatted abort and host crash integration | arbitrary formatting、backtrace、signal/minidump and process-wide policy | Obs/host composition | `OUT_OF_SCOPE` | typed bounded context remains Core；不复制 ggml varargs surface |

Public-surface reverse trace found no benchmark/test-only entry. `HYLE_FATAL_ASSERT[_MSG/_CONTEXT]` maps to the
first family; `HYLE_DEBUG_ASSERT[_MSG/_CONTEXT]` and `HYLE_DEBUG_ASSERT_ENABLED` map to the second;
`FatalValue/Field/Event`、`fatal_field`、field macros and `set_fatal_observer` map to typed failure observation.
All other `assert.h` details are implementation seams. `compiler.h`、`environment.h`、`cache.h`、`version.h`
and visibility ownership are unchanged and remain covered by the v3/v4 candidate evidence; this directed reopen
does not silently rescope them.

## Capability / gap ledger

| ID | Class / source | normal shape | owner / cost target | Coverage | Action | verification / negative space |
|---|---|---|---|---|---|---|
| `CORE5-01` | `MUST` reproducible API defect | caller field expression may use any legal local name | Core / failure-only array construction | `PARTIAL` | `FIX` | same-name observer death test；macro expansion has no caller-visible local declaration |
| `CORE5-02` | `MUST` reproducible compiler defect + C++23/Abseil mechanism | Release-disabled Debug expression is unevaluated but semantically checked like enabled form | Core / zero runtime code | `PARTIAL` | `FIX` | reject non-Boolean/dependent-invalid/message-invalid/field-invalid and 0/9 fields；accept valid templates |
| `CORE5-03` | `MUST` frozen P0 contract | successful Fatal remains one cold edge；disabled Debug contributes no work | Core / no metadata, allocation, lock, syscall or instructions | `PARTIAL` | `FIX` | optimized object audit plus existing single/zero evaluation tests |
| `CORE5-04` | `MUST` usability defect | host installs/restores callback across plugin/image lifetime | Core docs / no runtime addition | `PARTIAL` | `FIX` | guide names restore-before-unload and external quiescence；architecture stops pointing only at v3 task |

Every `MUST` has one finite `FIX`; there is no `PARTIAL + KEEP`. Arbitrary formatting, CHECK comparison families,
observer RAII/quiescence machinery, cross-DSO process registry and Native FFI remain rejected until a named consumer
proves their distinct task. No adjacent-module backlog or dependency change was found.

## API / usability freeze

- 状态：`FROZEN`。所有 public names、signatures、field types、1–8 bound、borrow/abort contract and include
  route remain unchanged；the repair is an internal array factory plus unevaluated compile semantics；
- normal recipe remains one macro for invariant, `_MSG` for static/borrowed explanation, `_CONTEXT` for 1–8 typed
  values, and explicit observer install/restore for host forwarding；
- defaults/cost：always-on condition once；debug disabled zero evaluation/code；message/fields failure-only；
  observer opt-in and failure-only；
- misuse diagnostics：zero/nine fields、invalid field/message and invalid condition must fail in every build mode；
  valid dependent code must compile in every mode；
- source/static/shared discovery remains `hyle::core` INTERFACE with no artifact。No public dependency、layout、
  error shape or build option changes；
- freeze identity：本 `AUDITED` task；performance cannot reopen API from a timing point because timing is N/A。

## Owner 六问（`AUDITED`）

1. **第三方：**Core product path has no direct or indirect third-party implementation。`std::array/span/
   string_view/atomic`、compiler attributes and C stdio/`abort` are standard/compiler/runtime contracts, not a
   third-party module。Abseil `b7adf83dccc73cd897fa6938978d96dce63a5d52` is `peer` only：semantic/artifact/
   decision owner remains Core；build/package remains delivery owner；no public third-party type leaks。
2. **功能：**2026-09-01 remote check records llama.cpp
   `518b76236b1f72a186deb5fcabc7b6e59e21fbc4` (only docs changed from the clean local
   `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0` across `src/common/include`), bundled ggml tree
   `2e3f246882361e111128191d95ec3b6151d640a0`, and standalone ggml
   `d4716378882593333721eb33f153144b6885caf2`。Their frequent invariant journey is admitted and becomes
   `COMPLETE` after CORE5-01/02；formatted varargs is intentionally Obs/host policy。No other unhandled Core `MUST`。
3. **性能：**`P0 STRUCTURAL`；normal consumer is source-compiled assertion code。Required verdict is structural
   `PASS/FAIL`, timing peer comparison `NOT_APPLICABLE`；no SOTA/non-inferiority claim。Candidate must prove one
   cold failure edge and zero disabled runtime instructions。
4. **旧 C ABI / Native FFI：**global policy `N/A`；no public C header/schema、bridge、undecorated export、managed
   sample、FFI option/target/package metadata or FFI-paid layout。Maintenance-only C linkage in codegen probes is
   test-private；shared decorated C++ symbols do not create C ABI。Expected verdict `CLEAN`。
5. **构建与导入：**Core stays header-only。Source embedding and installed static/shared consumers all compile
   through `hyle::core`；the installed shapes are imported INTERFACE journeys with no fabricated archive/DSO。
   Selected headers、exact package and relocation are rechecked；dependency graph is unchanged。
6. **C++ API 与文档：**task-shaped macros remain natural and do not expose benchmark/internal seams。The fix makes
   Release/Debug diagnostics predictable without adding user syntax。Guide must state field count/type rules、
   unevaluated semantics、borrow and observer unload/quiescence trap；architecture retains only long-term reasons。

## 冻结 FIX 集（`AUDITED`）

1. `CORE5-01`：replace caller-scope fixed array identifiers with a constrained C++23 internal field-array factory;
   both Fatal and enabled Debug context use it without changing public syntax or failure lifecycle。
2. `CORE5-02`：disabled Debug uses unevaluated C++ expressions that check the exact condition context、message route
   and same field factory；1–8 count is constrained in all modes。
3. `CORE5-03`：add one same-name runtime regression、valid/non-evaluated Release tests、five focused negative compile
   contracts and a disabled-Debug optimized probe；extend only the existing Core codegen verdict。
4. `CORE5-04`：update Core architecture/user text for current evidence ownership、semantic parity and observer
   callback image lifetime/quiescence。

当前为 `AUDITED`。以上四项是本轮全部授权修改；任何新 public API、capability、layout、dependency、
normal metadata or generic test-framework work must start another scope, not enter this implementation。
