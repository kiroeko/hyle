# Clay 迁移资产的权威边界

HYLE 的一部分早期代码来自 clay。来源解释历史，不赋予代码持续存在、公开或兼容的权利。迁移资产
必须和新实现一样通过当前能力、形状、依赖、correctness、性能和交付审查。

## 决策

- 以用户任务和当前组件 owner 重新设计，不按 clay 目录或类名恢复模块。
- 能直接证明正确、自然且符合依赖层级的算法可以保留；只因“已经有代码”不能进入能力超集。
- C API、跨 DLL factory、旧线程池、平台 shim、数据模型或 benchmark 形状不会因历史来源而兼容。
- SIMD/数学、容器、内存、调度和图能力分别进入自己的 owner；重复实现和第二 runtime 应删除。
- 旧代码行数、迁移百分比和当时缺陷清单是 implementation 事实，不属于长期架构。

2026-07 的逐文件资产盘点已迁到
[历史任务记录](../implementation/tasks/project/clay-asset-liability-2026-07.md)；更大范围的覆盖调查见
[clay coverage map](../implementation/tasks/project/clay-coverage-map.md)。它们可用于追溯来源和避免重复
调查，但不得覆盖当前 [能力完整性决策](capability-completeness.md)、[组件路线](roadmap.md)或 registry。
