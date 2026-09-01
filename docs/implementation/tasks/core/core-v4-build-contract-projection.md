# Core v4 编译合同投影重签

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由 Core v3 关闭后的真实 delivery 审计定向重开。Core headers、API、算法与布局仍冻结；本页只
> 重签 `hyle::core` 向 C++ consumer 传播的 build-owned compile contract。

## Closure card（`REOPENED`）

- component / owner / round / kind：`core` / Core + build-package / `core-v4-build-contract-projection` /
  `REOPEN`；
- baseline：`fe944155e6ada4e68997aaa3fa7d52f27b0769e0`；前轮：`core-v3-foundation`；
- trigger：`CORE-BUILD-CONTRACT-01`。Core-only target 仍传播由 Container 需求产生的 SSSE3 许可和
  MSVC `/EHsc`；这击中 v3 的“delivery shape 或 compiler semantics 改变”重开条件；
- evidence：[`Build/package v8`](../build/build-system-v8-core-contract-isolation.md)；
- scope：保持同一 Core 产品实现，只把公开 build usage requirements 缩回 Core 的真实最小集合；重放
  source/static/shared、编译环境事实、无二进制/FFI 和直接 consumer delivery；
- non-goals：新增或删除 Core capability/API；改变 fatal/cache/compiler/environment 实现；重审
  Container 算法/性能；全库 no-exception 支持；Native FFI；纳秒 benchmark；
- tier：`P0 STRUCTURAL`，timing 不适用；正常路径不得因 CMake target 增加 runtime dispatch、调用、
  metadata 或 ISA requirement；
- machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；instrument：`closure-correctness-v1`；
- required gates：Core-only target graph/compile-options、normal 与 exception-disabled compile、三平台
  Core tests、source/static/shared relocated consumer、无 Core binary/symbol、Container effective
  SSSE3 ownership、文档/易用性、独立 red team；
- stop：上述门全部通过则建立不可变 candidate 并关闭；任一受支持平台不能保持 Core 最小合同或
  Container 等价交付则写出单一 blocker，不扩大为全库构建重写。

当前为 `REOPENED`：尚未冻结 FIX ledger，也未修改产品或 build 实现。

