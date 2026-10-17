# HYLE C++ 用户文档

从[使用指南](usage-guide.md)开始：准备环境、构建库、嵌入源码或消费 SDK，再选择模块。
HYLE 面向 C++23，源码、静态 SDK 和动态 SDK 使用相同公开 API；当前没有产品 C API 或跨语言 binding。

## 接入与运行

- [使用指南](usage-guide.md)：首次构建、最小消费者和模块选择。
- [交付与集成](distribution.md)：SDK 安装、版本、动态库、后端资源和部署边界。
- [调用安全边界](usage-contract.md)：借用、异步完成、fiber、分配器和 ISA。
- [可编译样例](../../samples/README.md)：完整 C++ 工程和运行说明。

## 推理计算

- [Compute](compute.md)：准备算子、绑定存储、提交与等待。
- [Graph](graph.md)：多节点图、编译、独立执行槽与重复执行。
- [Apple MLX/Metal](apple-compute.md)、[Vulkan](vulkan-compute.md)：后端条件、原生接口和限制。
- [批次 attention](batch-attention.md)：ragged/paged 请求、KV 编码和页面更新。
- [状态版本](state-versions.md)：只读快照、dirty clone 和成功发布。

## 通用模块

| 领域 | 指南 |
|---|---|
| 基础与错误 | [core](core.md)、[error](error.md)、[math](math.md) |
| 系统、内存与时间 | [platform](platform.md)、[mem](mem.md)、[time](time.md) |
| 数据结构与工具 | [container](container.md)、[侵入式容器](container-intrusive.md)、[util](util.md) |
| 向量运算 | [simd](simd-usage.md) |
| 执行与通信 | [sched](sched.md)、[io](io.md)、[process](process.md) |
| 观测 | [obs](obs.md) |

支持范围以具体接口的系统、设备、数值和布局条件为准。模型片段或单个平台的结果不代表整模型、
所有硬件或任意编码均可用。
