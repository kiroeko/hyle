# 任务与调查索引

这里保存类似 Jira task 的施工事实：当时要解决什么、改了什么、怎样验证、得出了什么结论。记录可以
解释历史，但不会因为仍在仓库中就自动成为当前架构、API 或性能合同。

| 范围 | 任务记录 |
|---|---|
| 构建、安装、发布与公共仪器 | [`build/README.md`](build/README.md) |
| 异构计算 | [`compute/`](compute/) |
| 容器与数据结构 | [`container/`](container/) |
| 最小基础层 | [`core/`](core/) |
| 错误系统 | [`error/`](error/) |
| 计算图 | [`graph/`](graph/) |
| 异步 I/O | [`io/`](io/) |
| 标量数学补位 | [`math/`](math/) |
| 内存系统 | [`mem/`](mem/) |
| 可观测性 | [`obs/`](obs/) |
| 平台抽象 | [`platform/`](platform/) |
| 全项目与依赖升级 | [`project/`](project/) |
| 调度器 | [`sched/`](sched/) |
| SIMD | [`simd/`](simd/) |
| 时间与定时器 | [`time/`](time/) |
| 通用工具 | [`util/`](util/) |

## 新记录的最低内容

- 唯一标题或任务标识、记录性质、范围与不在范围内的事项；当前产品状态只写入
  [`module-status.json`](../module-status.json)；
- 变更前的事实和可复现问题，而不是从结论倒推的叙述；
- 实现选择、被否决的替代方案及理由；
- 精确的候选提交、命令、平台和证据路径；
- 结论、限制、未完成项与重新打开条件。

完成任务后，把仍然有效的内容分别沉淀到对应权威位置：设计和决策进入
[`architecture/`](../../architecture/README.md)，可重复维护方法进入
[`developer/`](../../developer/README.md)，外部 C++ 用法进入 [`user/`](../../user/README.md)。旧任务中的
退役名称、旧性能数字和阶段性判断只保留历史含义。
