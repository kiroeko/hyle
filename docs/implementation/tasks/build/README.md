# 构建、交付与项目级仪器任务

本目录记录构建/安装/发布系统，以及跨模块公共收口工具的施工事实。长期构建架构以
[`docs/architecture/build-system.md`](../../../architecture/build-system.md)为准；当前模块状态以
[`module-status.json`](../../module-status.json)为准。

- [构建与包系统收口](build-package-system-closure.md)
- [构建验证迭代](build-validation-iteration.md)
- [Build/package system v2 review（历史 VERIFIED 输入）](build-system-v2-review.md)
- [Build/package system v3 current-tree 收口（CLOSED）](build-system-v3-current-tree.md)
- [Build/package system v4 util 依赖边现树复签（CLOSED）](build-system-v4-util-edge-replay.md)
- [Build/package 能力完整性存量回放（PRODUCT_TRIGGER）](build-package-capability-replay-v1.md)
- [Build/package v5 selected-closure 修复（CLOSED）](build-system-v5-selected-closure.md)
- [Build/package v6 共享锁定源码缓存（CLOSED）](build-system-v6-shared-source-cache.md)
- [Build/package v7 内部 factor / 外部产品 façade 投影（CLOSED）](build-system-v7-factor-facade-projection.md)
- [Build/package v8 Core 编译合同隔离（CLOSED）](build-system-v8-core-contract-isolation.md)
- [Build/package v9 factor 迁移后 current-tree 复检（CLOSED）](build-system-v9-current-tree-replay.md)
- [Build/package v10 CUDA 语言作用域修复（CLOSED）](build-system-v10-cuda-language-scope.md)
- [Build/package v11 IO 因子身份登记（CLOSED）](build-system-v11-io-factor-registration.md)
- [Build/package v12 只追加身份与拓扑构造解耦（ACTIVE）](build-system-v12-append-only-topological-construction.md)
- [模块收口状态机与统一性能仪器 v1](closure-system-v1.md)
- [Formal 性能证据保留层 v1](perf-evidence-retention-v1.md)

任务记录可以解释一次实现或失败实验，但不能单独充当当前用户指南、模块 `CLOSED` 结论或 Formal
性能证书。
