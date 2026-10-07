# HYLE 实施记录

本目录保存已经发生或正在发生的工程事实：任务范围、调查过程、候选版本、验证结果、提交和未决项。
它相当于项目的 Jira 与实验档案，不定义长期架构，也不自动证明当前实现已经收口。

## 入口

- [当前施工状态](status.md)：新一轮逐模块重审的唯一状态表。
- [任务入口](tasks/README.md)：进入当前 record；被替代的精确历史从 Git 恢复。
- [性能结果](evidence/performance/README.md)：具备上下文的人读结果；原始数据位于仓库根级
  [`evidence/`](../../evidence/README.md)。
- [正确性资格证据](evidence/correctness/README.md)：跨组件逐能力语义回映与 coverage discovery
  解释；不创建第二套模块状态。

## 记录规则

每个新任务至少记录：目标、范围与非目标、起始 commit、需求或缺陷证据、变更、验证、结果、
遗留风险和结束 commit。进行中的假设必须标为假设；失败实验也应保留结论，避免重复踩坑。

任务结果与当前组件状态是两回事。历史 task 只保留当时事实；组件状态只按
[模块审查方法](../developer/review/module-closure-methodology.md)和当前 registry 推进。旧记录中的“完成”“SOTA”或
百分比只描述当时采用的标准，除非当前状态页与新的候选证书明确引用，否则均视为历史事实。
同理，旧记录里的“下一步”“继续”“必须做”“用户已决定”、handoff 或命令行只说明当时上下文；
新会话不得把这些文字当成当前提示、授权或待办，除非当前 closure card 明确重新纳入。

任务结束后，只把仍然有效的内容分别沉淀到：

- architecture：长期需求、边界与决策；
- developer：可复用的维护方法和故障经验；
- user：当前用户可以依赖的用法与陷阱。

不要把整份任务日志复制过去。

- [CPU SDPA、Copy、RMSNorm 当前成熟实现复比](tasks/compute/cpu-three-operator-recomparison.md)：
  有界性能证据任务，固定新版 peer、同合同和一次批量采样，产品状态不变。
- [CPU 三项成熟接缝、规则绑定与有限清退](tasks/compute/cpu-three-operator-route-closure.md)：
  基于实际复比修复 F32 fused、绑定适用规则并裁决重复实现。

被当前状态链替代的旧 task/evidence 会从工作树删除，以免搜索结果冒充待办。需要追溯时使用
`git log --all --name-status -- <path>` 定位 revision，再用 `git show <revision>:<path>` 读取；删除不等于
抹除历史。
