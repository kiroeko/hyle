# Validation selector：行内格式证明

<!-- PROJECT UNIT REVIEW: CLOSED -->

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

候选：`89c719af1f9e7c7d2c63e69e934acefa3ef8fab9`，tree
`d435e2477b2e10d639a465018f39c9578fda9188`。机器证明由候选内
`docs/implementation/validation-format-proof.json` 提供；同一入口供本地/CI 使用，精确验证 blob pair、
文件模式、classification 与 reviewer/reason，过期记录不能忽略新的变化。

验收 `PASS`：27 tests / 50.977 秒，覆盖行内重排阻塞、精确 FORMAT_ONLY、过期证明拒绝和真实
字符串空白 SEMANTIC 路线；本轮只实际运行 docs、closure、selector 三门，65 门带理由跳过。
docs 检查 452 Markdown / 1215 links、closure 28 components/schema 1 均通过。未重跑原已通过
LOC boundary test，本库 native build、交付与 P1 采样仍为零。

机器输出：[format fix result](../../../../evidence/validation-selector-v1/format-fix-result.json)及同目录
`receipts/`。其 JSON 使用 LF，raw logs 保留原字节；归档后 hash 再核验。前一集成提交
`07289cbd7b0f66fc8f66d6049eb1bcf04f072855` 的
[GitHub Actions](https://github.com/axiomates/hyle/actions/runs/34087668301) 也实际通过：compiler provisioning、
dependency restore 及所有 full jobs 均 `skipped`，selected gate 执行成功。这证明 CI 消费 selector，
不把该 run 冒充本修正候选的远端证据。

同一 immutable candidate 的 LOC self-check `PASS`：

| Metric | code |
|---|---:|
| 全库代码 | 467533 |
| 自实现含测试 | 121658 |
| 自实现不含测试 | 77122 |
| 当前工程单元 build-package | 10275 |

终态 `CLOSED`，无 blocker。格式证明是具名审查证据，不是通用 C++ 语义证明或无条件路径白名单。
只有相同任务出现可复现误选、错误 hash 绑定或绕过已冻结 P1 限制时，才另立有限工具修正。
