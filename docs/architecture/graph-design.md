# Graph 待设计边界

> `graph` 当前是 `QUEUED` 身份预留，唯一状态见
> [`module-status.json`](../implementation/module-status.json)。旧的 graph-01/02 里程碑、行数估算和
> “已完成”叙述属于历史任务，不是当前产品合同。

## 候选问题域

Graph 最终可能承载两类相关但未证明必须同属一个组件的任务：

1. 依赖 DAG 的建模、增量失效、缓存、拓扑执行和结构化错误传播；
2. range 上的 parallel-for/reduce/scan/sort 与 partitioning。

推理引擎可能需要静态或重复执行的算子 DAG，工程计算可能需要增量图和通用数据并行。两者共享
调度/取消机制不等于必须共享公开类型。开工时应先从真实 consumer 操作序列展开能力，再决定是一个
产品、多个内部 factor，还是直接组合现有 Sched/Container/SIMD。

## 已确定的边界

- Graph 不拥有线程池、fiber、wait 或 cancellation runtime；这些归 Sched。
- Graph 不拥有底层 kernel、device buffer 或 stream；这些归未来 Compute。
- Graph 不重写容器、allocator、timer 或 SIMD primitive，只通过真实依赖组合。
- 节点 callable、结果和错误的 ownership 必须显式；不能把 `std::future` 或裸线程池焊进公共面。
- 模型格式、operator schema、tensor planner、KV policy 和 serving orchestration 属于推理引擎。

## 开工门

1. 选取一个版本化推理 DAG journey 和一个工程增量/数据并行 holdout。
2. 对 oneTBB、Taskflow、Folly/成熟图执行器与真实引擎形状做能力族审计。
3. 决定 graph 与 data-parallel 是否同 owner，并据此重新审查中央依赖图。
4. 冻结 node/result/cancel/lifetime、重复执行、并发 mutation 和失败传播合同。
5. 把现有源码逐入口映回需求；旧 API 不需要兼容，benchmark-only 形状应删除。
6. 先验证复杂度、allocation 和 scheduling 结构，再在真实 DAG/range 上做同合同性能比较。

历史方案和实验只保存在 Git 历史中；当前没有活动 `tasks/graph` 记录。它们可以避免重复调查，
但不能替代 registry 中的新 `SCOPED/AUDITED` 轮。
