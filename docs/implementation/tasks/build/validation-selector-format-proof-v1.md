# Validation selector：行内格式证明

<!-- PROJECT UNIT REVIEW: AUDITED -->

横向工具修正，不改变任何产品 registry 或实现。

- baseline：`e61dfd1d3b9769993af4c5b97ae223ad42a9cce2`。
- 可复现输入：`int x=0;` → `int x = 0;`。v1 只自动证明行首/行尾格式，其他空白重排进入源码
  gate，与本 session「格式变化不得触发构建」要求不符。
- 有限 FIX：未证明的空白变化明确 `BLOCKED`、不选 build/P1；接受逐文件 old/new Git blob hash
  绑定、带 reviewer/reason 的 `FORMAT_ONLY/SEMANTIC` 证明。CI 自动读取约定的候选内证明路径。
- 非目标：C++ 语义解释器、formatter 实现、产品变更、P1 采样或全库构建。
- 验收：普通行内重排无构建；精确格式证明只选轻量门；过期/未声明的证明不能跳过变化；
  字符串空白变化可以由精确 `SEMANTIC` 记录进入正常 owning-factor gate。
- 预算/停止：一次修正、四类回归场景、选择器测试一轮与 docs/closure/diff；通过即 CLOSED。
- 原 [v1 task](change-impact-validation-selector-v1.md)及其 candidate/证据保留历史事实，
  此处仅修复具名工具漏判，不重写原 receipt。
