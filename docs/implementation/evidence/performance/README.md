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

历史 receipt、今天能否用当前 CLI 回放、当前产品状态是三件事。后来发生的 verifier 或环境策略漂移
若有显式版本兼容规则、且 frozen contract 能证明与 artifact 无关，可继续深回放并报告 compatibility；
无法证明时才形成方法兼容缺口，不能追溯证明产品失败。两种情况都不得修改旧 manifest。若
consumer 文件只发生经独立 review 证明不触及 ownership、计费、贡献、算法或交付的字节变化，旧
engineering bundle 可通过精确 hash renewal 链回放并报告 `REVIEWED_NON_MATERIAL`；任何真实消费者
语义/source artifact 变化仍不能续签新声明，Formal 也不继承 renewal。完整判断规则见
[统一性能仪器](../../../developer/performance/instrument-v1.md)。

“同合同不劣”“复杂度同档”“某个 workload 获胜”和“模块整体 SOTA”不是同一个命题。最后一项
通常没有可证定义，文档不得使用它代替逐能力结论。

## 当前证据怎样查

不要在本目录维护第二份容易漂移的“当前版本列表”。当前模块、工程性能 verdict 和精确证据
链接统一由 [`module-status.json`](../../module-status.json) 投影到
[`status.md`](../../status.md)；状态表中的 Engineering performance 单元格可直接进入该模块当前证据。
同一个报告可以服务多个拆分 factor，例如 Platform 三个 factor；这不会创造第二个产品状态。

registry 的 `performance.evidence` 路径是当前工程证据的唯一选择器。旧候选页面应写 `HISTORICAL`
或 `DIAGNOSTIC`，不得靠标题中的“当前”、任务中的 `PASS`、文件名版本号或额外 marker 取得权威性。

跨模块的固定方向点值复核保存在
[模块固定方向历史审计](../../tasks/performance/frozen-module-fixed-direction-audit-2026-09-05.md)；
它是历史审计输入，不替代每个 current evidence 页面，也不把一次无机制的微小固定方向改写成产品
缺陷。
