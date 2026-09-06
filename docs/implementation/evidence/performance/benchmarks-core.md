# Core 当前工程性能证据

<!-- PERFORMANCE EVIDENCE: CURRENT -->

> 当前状态只见 [`module-status.json`](../../module-status.json)。本页绑定 Core v5 candidate
> `4b7f28c55e9ca118cf87b80058ce291261033ec5` 的 P0 结构证据；完整三平台结果见
> [verification manifest](../../../../evidence/core/manifest-4b7f28c5.md)，任务事实见
> [Core v5 closure card](../../tasks/core/core-v5-assert-macro-contract.md)。

## 当前有限结论

Core v5 只修复 assertion 宏的 C++ 编译语义，没有新增 runtime state、layout、依赖或正常路径工作，
因此 timing peer comparison 为 `NOT_APPLICABLE`，工程 verdict 为 `P0 / STRUCTURAL PASS`：

- context 宏不声明固定 caller-scope 名称，合法字段表达式不会被宏内部捕获；
- Release-disabled Debug condition、message 和 1--8 fields 由 unevaluated expression 保持启用态语义检查，
  同时保证零求值；
- optimized disabled-Debug probe 在 MSVC x64、GCC x86-64、AppleClang arm64 上逐指令等同同签名
  identity control；
- fatal assertion 的 success path 只有通往 cold failure sink 的条件边，没有 allocation、lock、syscall、
  runtime dispatch 或常态 metadata；
- 三平台 Release/Debug WERROR、正/负编译、适用 sanitizer、源码/static/shared relocated consumer、
  public-header closure 和中央 component contract 全部通过。

公开结论仅是：冻结候选和受检工具链/交付形状上的 assertion 正常路径结构成本符合合同。终止路径
observer、格式化和 I/O 不作纳秒排名；本页不宣称“整个 Core SOTA”。

## 结构证据边界

Core 的 value/compiler/layout helpers 通过 `constexpr`、`static_assert`、object-size/alignment、公开头
self-containment 和目标 codegen 验证。只有具名 consumer 将某个 primitive 提升为热点并证明存在可归因
的 material 差距，才建立 P1/P2 timing 合同；单条指令、亚纳秒点值或代码地址差异不授权改变 API。

Core v1--v4 的候选、实验和已替换结论只从各自 implementation task 与 Git 历史追溯，不在当前性能页
重复展开，也不能续签到 v5。
