# HYLE

HYLE（HYbrid LEvels）是面向本地 AI 推理、异构计算、资源流送和低延迟服务的 C++23 基础加速库。
它提供可组合的内存、数据结构、SIMD、调度、I/O、Compute 和 Graph；模型格式、tokenizer、
生成与采样策略、KV 缓存策略和 serving 政策由推理引擎拥有。

当前推理目标是多系统上的单机 CPU、CPU＋一张 GPU、单 GPU。
项目仍处于预发布阶段，支持范围和限制以对应接口、后端及实际验证环境为准。

## 主要能力

- **Compute 与 Graph**：直接算子执行、计算图编译、可复用计划、执行槽及跨设备资源依赖。
- **内存与数据处理**：分配器、资源预算、buffer、容器、标量数学与 SIMD 整批运算。
- **任务与通信**：CPU task/fiber 调度、取消与关闭、异步文件和网络 I/O、子进程管理。
- **基础设施**：错误表达、平台能力、时钟、日志、指标与跟踪。

项目面向 Windows、Linux 和 macOS，计算后端包括 CPU、CUDA、Metal 与 Vulkan，具体组合由平台和
接口支持范围决定。提供完整源码、静态 SDK 和共享 SDK 三种 C++ 交付，不承诺跨编译器/STL 的二进制 ABI。

## 文档与引导

| 目的 | 入口 |
|---|---|
| 使用 HYLE，包括首次构建与 C++ 接入 | [用户手册](docs/user/README.md) |
| 修改、构建、测试与排障 | [开发与维护](docs/developer/README.md) |
| 理解产品边界与设计取舍 | [架构](docs/architecture/README.md) |
| 查询实施材料与历史依据 | [实施材料](docs/implementation/README.md) |

[文档索引](docs/README.md)说明文档分工；[样例](samples/README.md)提供独立 C++ 消费者工程。
工作区约定见 [AGENTS.md](AGENTS.md)，当前任务见[工作主线](docs/developer/current-work.md)。

## 仓库结构

```text
include/hyle/   C++ 公开头
src/            模块实现
tests/          正确性、并发与构建测试
bench/          性能实验
samples/        C++ 消费者工程
cmake/          组件、依赖和安装规则
scripts/        构建、依赖和诊断工具
docs/           用户、开发、架构和历史记录
evidence/       原始实验数据
third_party/    第三方源码与适配
```

## 许可

项目授权以 [LICENSE](LICENSE) 为准；第三方材料保留各自许可，见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
