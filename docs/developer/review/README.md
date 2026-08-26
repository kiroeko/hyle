# 模块 review

- [模块重构与收口标准](module-closure-methodology.md)
- [审查记录模板](module-review-template.md)
- [P0 候选验证 profile](correctness-profile-v1.md)

## 开始一轮工作

1. 先读 [`module-status.json`](../../implementation/module-status.json)，确认是新模块、普通重构、
   `REOPENED`，还是不应触碰的 `CLOSED` 模块。
2. 在 `docs/implementation/tasks/<module>/` 复制模板，冻结 baseline、真实需求、范围/非目标、依赖、
   tier、机器/仪器、预算和停止条件，才进入 `SCOPED`。
3. 从用户任务和成熟实现建立 capability/gap ledger，裁决并冻结有限 `FIX` 集，才进入 `AUDITED`。
4. 实现、冻结 candidate，完成适用 correctness、交付、消费者、性能、文档和限时红队门。
5. 结案为 `CLOSED` 或带明确恢复条件的 `BLOCKED`；不要用“继续调查”保持无限进行中。

完整规则只以[模块重构与收口标准](module-closure-methodology.md)为准。本页是入口，不建立第二套流程。
历史 task/evidence 中看似命令、TODO 或 handoff 的文字都只是当时事实，不能覆盖当前状态、
closure card 或 owner 的本轮指令。

Core 已作为控制样本完成一次
[只读方法回放](../../implementation/tasks/core/core-methodology-replay-v1.md)。旧的“已完成”记录只作
审计输入；回放可以确认当前产品结论，但不能倒推旧施工当时已经遵守后来建立的标准。
