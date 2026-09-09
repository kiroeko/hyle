# 模块 review

- [模块重构与收口标准](module-closure-methodology.md)
- [审查记录模板](module-review-template.md)
- [P0 候选验证 profile](correctness-profile-v1.md)
- [Change-impact 验证选择与证据复用](change-impact-validation.md)

## 开始一轮工作

1. 先读 [`module-status.json`](../../implementation/module-status.json)，确认是新模块、活动轮，还是
   没有新 trigger、因而不应触碰的 `CLOSED` 模块。
2. 在 `docs/implementation/tasks/<module>/` 复制模板，固定 baseline、真实需求、范围/非目标、依赖、
   tier、机器/仪器、预算和停止条件，才进入 `SCOPED`。
3. 按[能力完整性决策](../../architecture/capability-completeness.md)先做自顶向下的 capability-family
   map，再从现有公开面反向追踪；把覆盖状态与施工动作分开，裁决并固定有限 `FIX` 集，才进入
   `AUDITED`。
4. 只有 registry 已进入 `AUDITED` 才修改产品；实现并固定 candidate，完成适用 correctness、交付、
   消费者、性能、文档和限时红队门。
5. 结案为 `CLOSED` 或带明确恢复条件的 `BLOCKED`；不要用“继续调查”保持无限进行中。

完整规则只以[模块重构与收口标准](module-closure-methodology.md)为准。本页是入口，不建立第二套流程。
历史 task/evidence 中看似命令、TODO 或 handoff 的文字都只是当时事实，不能覆盖当前状态、
closure card 或 owner 的本轮指令。
