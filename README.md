# HYLE

HYLE（HYbrid LEvels）是面向单机、多系统的 C++23 推理引擎项目，同时提供可独立构建和使用的
内存、数据结构、SIMD、调度、I/O、Compute 与 Graph 基础模块。
引擎层负责模型接入、生成、会话与请求编排，底层模块保留各自的计算和资源边界。
当前开发重点仍是底层推理支持；引擎路径随真实模型测试推进，server 与应用 API 属于后续工作。

当前推理目标是多系统上的单机 CPU、CPU＋一张 GPU、单 GPU。
项目仍处于预发布阶段，支持范围和限制以对应接口、后端及实际验证环境为准。
当前完整文本路径来自 Qwen3.5-2B 官方 BF16 实现，包含普通推理、greedy K1 MTP 和多会话；
仍使用准备后的本地权重清单和 token IDs，通用模型加载、C++ tokenizer、服务入口与图片链路尚未完成。

## 主要能力

- **真实模型路径**：用 Qwen 文本、会话和 MTP 执行检验底层功能、架构与端到端性能，积累引擎实现。
- **Compute 与 Graph**：直接算子执行、计算图编译、可复用计划、执行槽及跨设备资源依赖。
- **内存与数据处理**：分配器、资源预算、buffer、容器、标量数学与 SIMD 整批运算。
- **任务与通信**：CPU task/fiber 调度、取消与关闭、异步文件和网络 I/O、子进程管理。
- **基础设施**：错误表达、平台能力、时钟、日志、指标与跟踪。

项目面向 Windows、Linux 和 macOS，计算后端包括 CPU、CUDA、Metal 与 Vulkan，具体组合由平台和
接口支持范围决定。提供完整源码、静态 SDK 和共享 SDK 三种 C++ 交付，不承诺跨编译器/STL 的二进制 ABI。

## 文档与引导

| 目的 | 入口 |
|---|---|
| 构建和运行当前推理引擎 | [推理引擎指南](docs/user/inference.md) |
| 独立使用 C++ 基础模块 | [用户手册](docs/user/README.md) |
| 修改、构建、测试与排障 | [开发与维护](docs/developer/README.md) |
| 查询已验证模型、算子与边界 | [模型与算子覆盖](docs/implementation/model-operators.md) |
| 理解产品边界与设计取舍 | [架构](docs/architecture/README.md) |
| 查询实施材料与历史依据 | [实施材料](docs/implementation/README.md) |
| 查看当前实现范围与后续顺序 | [推理工程计划](docs/implementation/inference-roadmap.md) |

[文档索引](docs/README.md)说明文档分工；[样例](samples/README.md)提供独立 C++ 消费者工程。
工作区约定见 [AGENTS.md](AGENTS.md)。

## 仓库结构

```text
include/hyle/   C++ 公开头
src/            模块实现
engine/         推理引擎与模型接入
tests/          正确性、并发与构建测试
bench/          性能实验
samples/        C++ 消费者工程
cmake/          组件、依赖和安装规则
scripts/        构建、依赖和诊断工具
docs/           用户、开发、架构和历史记录
docs/evidence/  原始实验数据
third_party/    第三方源码与适配
```

## 许可

项目授权以 [LICENSE](LICENSE) 为准；第三方材料保留各自许可，见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
