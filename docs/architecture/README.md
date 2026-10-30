# 架构：职责、决策与取舍

HYLE 的产品目标是完整的单机推理引擎，各基础模块仍独立交付。当前主阶段是底层推理支持，
以真实模型驱动功能、架构和性能的收敛；现有 Qwen 入口还不是通用模型服务。

主要计算关系如下：

```text
engine/   模型接入、生成／MTP、会话、请求与缓存政策
   ↓ 组成模型的计算图
Graph     图编译、放置、资源规划与执行依赖
   ↓ 绑定计算实现
Compute   算子语义、设备资源与原生执行
   ↓
CPU / CUDA / Apple Metal / Vulkan
```

内存、容器、SIMD、调度、I/O 等基础模块由各层按需复用。Compute 可直接使用，Graph 可在独立
C++ 程序中使用；两者都不解释模型文件或生成策略。具体模型与推理政策属于 `engine/`。

本目录解释这些边界为什么存在、关键决策及取舍。第一次阅读可以按自己的目标选一条：

1. **了解项目**：[项目动机](motivation.md) → [产品范围](hyle-scope.md)。
2. **理解推理路径**：[两项核心期许](inference-principles.md) → [引擎概览](engine-overview.md) → [推理系统整体架构](inference-design.md)，
   涉及计算图和设备执行时再看 [Graph](graph-design.md) 与 [Compute](compute-design.md)。
3. **修改某个基础模块**：先看下面对应的设计；涉及所有权、失败或关闭时查
   [基础机制](foundation-design.md)与[运行时](runtime-design.md)的共同原则。

API 与用法见[用户文档](../user/README.md)，构建、测试与测量方法见[开发文档](../developer/README.md)。
当前实现位置和后续工作统一看[推理进度](../developer/inference-progress.md)。

## 总体

| 文档 | 要回答的问题 |
|---|---|
| [推理引擎的两项核心期许](inference-principles.md) | 跨模型模块化、冷热职责与默认推理性能如何指导开发取舍 |
| [项目动机](motivation.md) | 为什么做 HYLE，为什么需要真实消费者，为什么先做本地推理引擎 |
| [产品范围](hyle-scope.md) | 做什么、不做什么，模块与引擎的分工 |
| [构建与交付](build-system.md) | 用户意图如何变成 target、依赖、宏和可迁移 SDK |
| [许可](licensing.md) | 原创与第三方材料的许可、归属和交付 |
| [参考实现](references.md) | 各类问题参考谁，哪些机制不能照搬 |

## 基础模块

共同原则（事实与政策分开、owner 决定寿命、发布与退休分开、按使用付费）在
[基础机制](foundation-design.md)；任务、取消与关闭的共同顺序在[运行时](runtime-design.md)。
下面是各模块的设计文档。

| 模块 | 文档 | 要点 |
|---|---|---|
| core | [core](core-design.md) | 编译与布局事实、断言、可见性宏、版本 |
| error | [error](error-design.md) | `Status`／`Result` 的表示、错误域、跨 DSO 与 OOM 行为 |
| platform | [platform](platform-design.md) | CPU／ISA／拓扑、线程与等待、虚拟内存、文件系统、动态库在三个系统上的取舍 |
| mem | [mem](mem-design.md) | mimalloc 后端、arena 与池、Buffer／映射、配额、fiber 栈、读者回收 |
| container | [container](container-design.md)、[侵入式容器](container-intrusive-design.md) | 何时自建容器、哈希与 UTF-8 的第三方选择、并发队列、侵入式成员关系 |
| time | [time](time-design.md) | 时钟域、deadline、timer queue |
| obs | [obs](obs-design.md) | 日志、指标、trace 的成本模型与 Tracy 接入 |
| util | [util](util-design.md) | 借用与拥有的 callable、registry 的冻结、作用域工具 |
| math | [math](math-design.md) | checked 标量运算的边界 |
| simd | [SIMD 分派](simd-design.md)、[实现来源](simd-vendor-surface.md) | 编译期与运行期 ISA 选择；Highway、xsimd 与 HYLE 自有实现如何分工 |
| sched | [sched](sched-design.md) | fiber 与 work stealing、上下文切换选型（fcontext 而非 ucontext／Windows Fiber）、取消与 scope |
| io | [io](io-design.md) | IOCP／io_uring／kqueue 的完成模型、HTTP/1、HTTP/2、TLS、WebSocket、文件监视 |
| process | [process](process-design.md) | 子进程、进程树与 reap |

## 计算与推理

| 文档 | 要回答的问题 |
|---|---|
| [Compute](compute-design.md) | 数值与资源合同、Provider 选择、异步保活、状态发布与分页准备 |
| [原生路线](compute-routes.md) | CPU、CUDA、MLX/Metal、Vulkan 及 CPU＋GPU 组合中哪些条件不能互换 |
| [Graph](graph-design.md) | DAG、region、shape／effects、并发内存与执行计划如何衔接 |
| [推理引擎概览](engine-overview.md) | `engine/` 的组成、一次 decode 步的路径、当前结构限制 |
| [推理系统整体架构](inference-design.md) | 普通／MTP、多会话／多请求、缓存、模型状态、冷装配与执行如何分工 |

更早版本的设计文档（2026-10-18 之前）可以从提交 `d12f2204d` 读取，例如
`git show d12f2204d:docs/architecture/compute-routing-system.md`。其中的类型名和流程可能已经变化，
以当前代码和上面的文档为准。
