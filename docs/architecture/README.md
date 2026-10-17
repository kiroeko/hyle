# 架构：少数原则与必要差异

这些文档解释为何这样分层，以及哪些原生差异不能被统一外观抹掉。
API与用法见[用户文档](../user/README.md)，操作与性能方法见[开发文档](../developer/README.md)，
当前任务见[工作主线](../developer/current-work.md)。这里集中说明长期边界和设计取舍。

从[产品范围](hyle-scope.md)进入：先区分事实、机制、应用政策和资源owner，再看需要深入的设计。

| 设计 | 要回答的问题 |
|---|---|
| [基础机制](foundation-design.md) | 编译/运行事实、错误值、allocator、容器、时间与观测为何各有边界 |
| [运行时](runtime-design.md) | 谁负责进展、取消、wake、排空、IO参数和child reap |
| [SIMD](simd-design.md) | 编译期实现选择与运行时ISA选择如何分开，为什么不维护第三套register库 |
| [Compute](compute-design.md) | 数值/资源合同、Provider选择、异步保活、状态发布与分页准备 |
| [原生路线](compute-routes.md) | CPU、CUDA、MLX/Metal、Vulkan及CPU＋GPU哪些条件不能互换 |
| [Graph](graph-design.md) | DAG、region、shape/effects、并发内存与固定执行计划如何衔接 |
| [构建与交付](build-system.md) | 用户意图如何变成target、依赖、宏和可迁移SDK |
| [许可](licensing.md) | 原创、移植和第三方材料的授权与分发边界 |
| [参考实现](references.md) | 相关问题该参考谁，哪些机制不能直接照搬 |
