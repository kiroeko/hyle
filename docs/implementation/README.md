# 工程范围与模型参考

这里区分**整体开发范围**、**局部模型需求**和**按需追溯输入**；不维护逐轮任务卡或验收报告。

| 层次 | 入口 | 用途 |
|---|---|---|
| 整体重点 | [推理工程范围](inference-roadmap.md) | 本阶段完整目标、范围与边界；具体任务和进度只在[看板](../developer/inference-progress.md) |
| 局部重点 | [模型与算子地图](model-operators.md) | 模型、格式、计算和状态需求；需求不等于支持 |
| 细节参考 | `tasks/compute/`现有结构化输入 | 固定源身份、需求编号与原测试资产；部分JSON仍被测试读取 |
| 历史查证 | [原始证据](../evidence/README.md)与Git | 旧调查、旧计划及失败记录；其中命令和“下一步”不是当前指令 |

## 仍被使用的输入

- [Qwen3.5-2B本地输入身份](tasks/compute/qwen35-2b-local-input.json)与[独立文本参考](../../tests/compute_models/qwen35_reference.py)。
- [模型来源](tasks/compute/llm-moe-model-sources.json)、[需求目录](tasks/compute/llm-moe-provider-demand-matrix.json)、[历史片段映射](tasks/compute/llm-moe-coverage-matrix.json)及[原验收数据](tasks/compute/llm-moe-d5-acceptance-ledger.json)。

输入记录和历史状态不代表当前完整引擎通过；保留仍被代码读取的字段，不为整理阅读入口改写原资产。

## 追溯旧文档

更早的过程文档可从`git show d12f2204d:<path>`读取。当前用法见[用户文档](../user/README.md)，当前合同与取舍见[架构](../architecture/README.md)，不从旧记录恢复治理流程。
