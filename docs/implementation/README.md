# 实施材料与追溯

这里保留仍被测试或调查使用的结构化输入、来源与原始资产，不维护逐轮任务卡、资格表或施工计划。
当前用法见[用户手册](../user/README.md)，设计理由见[架构](../architecture/README.md)。

## 仍在使用的材料

- [现行模型、量化制品与算子需求](model-operators.md)：当前支持范围、计算子图和反向依赖审查入口；
  与历史实现、运行结果分开，不将需求清单视为已通过的支持表。
- [Qwen3.5-2B 本地输入身份](tasks/compute/qwen35-2b-local-input.json)，对应
  [独立文本参考](../../tests/compute_models/qwen35_reference.py)。这是输入记录，不是完整 Hyle 引擎通过证明。
- [模型来源](tasks/compute/llm-moe-model-sources.json)、
  [历史片段映射](tasks/compute/llm-moe-coverage-matrix.json)、
  [原验收数据](tasks/compute/llm-moe-d5-acceptance-ledger.json)：部分现有测试读取这些 JSON，
  字段中的阶段名和历史结果不自动代表当前支持范围。
- `tasks/compute/` 的其余 JSON：原始需求、格式、handoff 与实验索引；不作为新任务或开发门禁。
- [evidence](../evidence/README.md)：历史源码、日志、输入、数据与复现资产；JSON/ZIP 保留旧实验参数、结果与归档。

## 追溯旧文档

清理前的任务与过程文档在提交 `d12f2204d` 中，例如：

```text
git show d12f2204d:docs/implementation/tasks/project/qwen35-2b-command-consumer-scope-2026-10.md
```

旧报告中的审批、状态、预算、截止时间、命令和“下一步”不是当前规则；
旧路径、硬件、依赖版本和通过数量也需要重新核实。
有用的设计取舍与故障模式已归入现行架构、开发或用户指南，不再要求维护者同时更新这些过程记录。
