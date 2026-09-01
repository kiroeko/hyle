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

当前为 `SCOPED`：closure card 已冻结；产品修改必须等待 capability/public-surface 审计与有限 `FIX`
ledger 进入 `AUDITED`。
