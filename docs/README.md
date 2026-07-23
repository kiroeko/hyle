# HYLE 文档地图与信源规则

HYLE 的文档按读者任务分层，同一事实只在一个目录维护。发现冲突时，先按下表找到
该事实的责任目录，再修责任文档；不要在其他目录复制一份“同步后的答案”。

| 目录 | 只负责 | 不负责 |
|---|---|---|
| [`architecture/`](./architecture/README.md) | 当前整体设计、模块边界、公共能力形状、长期约束与决策理由 | 施工进度、某次测试数、benchmark 点值 |
| [`guide/`](./guide/README.md) | 用户如何构建、选型、调用和正确管理生命周期 | 设计演进史、性能总表、项目排期 |
| [`performance/`](./performance/README.md) | benchmark 方法、环境、原始结果、同合同对照、win/loss 与待补证据 | API 设计、实现任务排期 |
| [`milestones/`](./milestones/README.md) | Jira 式 work item：目的、依赖、改动、验收、提交和剩余任务 | 当前架构真相、长期用户手册、跨任务性能总账 |
| [`debug/`](./debug/) | 一次故障或调查的证据链 | 当前产品说明 |
| [`setup/`](./setup/README.md) | 贡献者环境与构建依赖 | 产品能力和进度 |

## 冲突优先级

1. 实际公开头和实现决定“代码现在做什么”；测试与导出清单决定机械事实。
2. `architecture/` 决定“产品为什么这样设计、边界在哪里”。
3. `performance/` 决定“在哪个 commit、平台和 workload 下快慢如何”。
4. `guide/` 必须与前三者一致，但只保留用户完成任务所需的信息。
5. `milestones/` 和 `debug/` 是带日期的工作记录。旧名、旧数字和当时判断可以作为历史
   留存，但不得被引用为当前产品事实。

## 更新纪律

- 公共类型、所有权或模块边界改变：更新对应 `architecture/*-design.md`，再更新 guide。
- C/FFI 产品面改变：更新 `architecture/c-abi-design.md`、对应 `guide/c-*.md` 和机械清单；
  milestone 只记录这次 work item 做了什么。
- benchmark 改变：结果、方法和未关闭输项只写入 `performance/`；guide 只链接，不抄数字。
- task 完成：milestone 写验收和 commit，稳定结论合并回 architecture/performance/guide 后停止扩写。
- 项目状态只在 milestone 索引/预发布任务单维护；architecture 不写每日完成百分比。

