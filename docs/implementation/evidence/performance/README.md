# 性能结果档案

本目录保存人可读的性能调查与历史结果。它回答“在什么候选、合同、机器和统计规则下观察到
什么”，不保存性能方法，也不授予整个模块一个永久的 `SOTA` 标签。

- 仪器与判定规则见
  [benchmark evidence standard](../../../developer/performance/benchmark-evidence-standard.md)。
- 原始 CSV/JSON/log 放在仓库根级 [`evidence/performance/`](../../../../evidence/performance/)。
- 当前模块证书与可信度见 [施工状态](../../status.md)。

每份有效报告必须绑定实现 commit、构建/依赖 manifest、公开或内部合同卡、平台与 affinity、
输入分布、强 peer 版本、计费边界、原始数据和 verdict。缺少任一关键项的旧报告仍可用于定位
历史问题，但统一标为 `HISTORICAL` 或 `DIAGNOSTIC`，不能驱动产品形状或当前宣传。

“同合同不劣”“复杂度同档”“某个 workload 获胜”和“模块整体 SOTA”不是同一个命题。最后一项
通常没有可证定义，文档不得使用它代替逐能力结论。
