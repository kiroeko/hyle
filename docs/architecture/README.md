# 架构与设计

HYLE 提供可组合的基础机制；模型和业务策略留在消费者。
设计页说明产品边界和实现理由，具体用法与支持限制见[用户文档](../user/README.md)，
当前安排见[工作主线](../developer/current-work.md)。

## 产品与系统

- [产品范围](hyle-scope.md)、[动机](motivation.md)、[推理引擎需求](inference-engine-north-star.md)
- [能力完整性](capability-completeness.md)、[组件与依赖](capability-factorization.md)、[分层](roadmap.md)
- [性能比较](performance-strategy.md)、[生命周期和失败](lifecycle-design.md)
- [构建与交付](build-system.md)、[许可](licensing.md)、[FFI 边界](native-ffi-policy.md)
- [参考实现](references.md)、[模型与格式需求](model-compute-catalogue.md)

## 模块

- [Core](core-design.md)、[Error](error-design.md)、[Platform](platform-design.md)
- [Mem](mem-design.md)、[Time](time-design.md)、[Obs](obs-design.md)、[Util](util-design.md)
- [Container](container-design.md)、[侵入式容器](container-intrusive-design.md)
- [SIMD](simd-design.md)、[SIMD 第三方边界](simd-vendor-surface.md)、[Math](math-design.md)
- [Sched](sched-design.md)、[IO](io-design.md)、[Process](process-design.md)
- [Compute](compute-design.md)、[Graph](graph-design.md)、[后端路线](compute-routes.md)
- [冷路由](compute-routing-system.md)、[CPU/GPU 组合](compute-next-stage.md)
- [资源与流送](compute-resource-streaming-design.md)、[版本状态与分页](compute-state-paged-design.md)

已有范围讨论还包括[基础能力](foundation-capability-expansion.md)和
[候选通用机制](inference-engine-capability-watchlist.md)。它们记录设计需求，不构成完成声明或自动待办。
