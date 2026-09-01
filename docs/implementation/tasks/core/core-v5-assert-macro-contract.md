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

当前仅登记 `REOPENED` 事实；尚未冻结 closure card 或 `FIX` ledger，也未授权产品修改。
