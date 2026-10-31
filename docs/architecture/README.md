# 架构文档

架构文档回答职责如何分开、关键合同为什么存在。用法查[用户手册](../user/README.md)，
当前任务查[进度看板](../developer/inference-progress.md)，测量数据查[性能资料](../developer/performance/README.md)。

## 阅读地图

| 层次 | 读什么 | 读到什么程度 |
|---|---|---|
| 整体重点 | [两项核心期许](inference-principles.md) → [引擎概览](engine-overview.md) → [推理整体架构](inference-design.md) | 知道目标、依赖方向、冷／热分工、提交与资源寿命 |
| 局部重点 | 下面对应模块设计 | 只读本次要改的模块及其依赖合同 |
| 按需细节 | [用户手册](../user/README.md)、公开头、设计页链接的源码 | 查具体参数、完整API、算法或后端限制，不要求顺序通读 |
| 历史证据 | [实验记录](../evidence/README.md)及Git历史 | 查某个结论的版本、输入、失败或复现，不作为当前工作指令 |

项目关系是`engine → Graph → Compute／原生后端`。内存、容器、调度和I/O等基础模块按需复用，
可独立构建使用；模型格式、tokenizer、生成及会话／请求政策归引擎，底层不依赖模型名单。

## 整体与交付

| 文档 | 唯一主题 |
|---|---|
| [项目动机](motivation.md)、[产品范围](hyle-scope.md) | 为什么做、当前产品边界及基础库与引擎的关系 |
| [两项核心期许](inference-principles.md) | 跨模型模块化、冷期准备及每个受支持模型的默认性能目标 |
| [引擎概览](engine-overview.md)、[整体架构](inference-design.md) | 源码地图；模型、请求、缓存、执行和完整状态如何衔接 |
| [构建](build-system.md)、[许可](licensing.md) | 产品／依赖与SDK交付；原创及第三方材料边界 |
| [参考实现](references.md) | 哪类机制参考谁及不可照搬的条件 |

<a id="基础模块"></a>

## 局部设计

| 要修改的范围 | 文档 |
|---|---|
| 共同所有权、任务、取消与关闭 | [基础机制](foundation-design.md)、[运行时](runtime-design.md) |
| 计算语义、选择与资源 | [Compute](compute-design.md)；涉及具体后端时查[原生路线](compute-routes.md) |
| 逻辑DAG、形状、放置与执行计划 | [Graph](graph-design.md) |
| 编译事实与错误 | [core](core-design.md)、[error](error-design.md) |
| OS／硬件、分配与存储 | [platform](platform-design.md)、[mem](mem-design.md) |
| 数据结构 | [container](container-design.md)、[侵入式容器](container-intrusive-design.md) |
| 任务与I/O | [sched](sched-design.md)、[io](io-design.md)、[process](process-design.md) |
| 时间、观测及工具 | [time](time-design.md)、[obs](obs-design.md)、[util](util-design.md) |
| 标量与SIMD | [math](math-design.md)、[SIMD分派](simd-design.md)、[SIMD实现来源](simd-vendor-surface.md) |

## 维护边界

只在职责、公开合同或关键取舍改变时更新对应设计，不把每片实现、检查数量、失败重试和百分比追加到架构页。
当前事实在所属入口维护，其它页面链接引用；旧版本细节可用Git查证，不要求为每轮修改复制新报告或归档。
