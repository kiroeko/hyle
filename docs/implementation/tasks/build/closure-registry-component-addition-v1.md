# Closure registry：新 factor 的 QUEUED 登记

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 问题与边界

全局 factorization 已要求新 factor 先同时进入中央 component manifest 与
`module-status.json`，初始状态只能是 `QUEUED`，并把修改前 registry 传给
`check-module-closure.py --previous`。旧检查器却要求前后 component 集合完全相同，使这条受文档要求的
登记路径机械上必然失败。

这是独立 build/tool blocker，不属于 Time 产品实现。本任务只修 transition comparison；不改变 module
状态语义、CMake 图、产品 API、性能仪器或历史 registry。

## 实现合同

- 既有 component 可以继续按原状态图迁移；
- 新 component 只能以 `QUEUED` 首次出现；
- 删除、重命名或重排既有 component 继续失败；
- 当前 registry 仍必须与当前 `HyleComponents.cmake` 的完整有序 component 集严格一致；
- 检查器不解释 CMake、Git 历史或 factor 产品语义。

## 验证与裁决

tooling regression 覆盖合法 `QUEUED` 插入，以及新增为 `SCOPED`、删除和重排三个负例。规范 registry、
状态投影、文档链接和 diff hygiene 同时通过后，本独立工具任务关闭。后续 Time factor migration 必须保存
修改前快照并实际使用 `--previous`，不能因本工具修复省略状态证据。
