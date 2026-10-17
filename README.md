# HYLE

HYLE（HYbrid LEvels）是面向本地 AI 推理、异构计算、资源流送和低延迟服务的 C++23 基础加速库。
它提供可组合的内存、数据结构、SIMD、调度、I/O、Compute 和 Graph；模型格式、tokenizer、
采样、KV 缓存策略和 serving 政策由推理引擎拥有。

当前推理目标是多系统上的单机 CPU、CPU＋一张 GPU、单 GPU。
项目仍处于预发布阶段，支持范围和限制以对应接口、后端及实际验证环境为准。

## 开始使用

- [C++ 使用指南](docs/user/usage-guide.md)
- [源码、静态库与共享库集成](docs/user/distribution.md)
- [可运行样例](samples/README.md)
- [开发环境与离线依赖](docs/developer/setup/README.md)

三种形式均为 C++ 交付，不承诺跨编译器/STL 的二进制 ABI；当前没有独立 C API 或外语绑定。

## 开发与文档

- [开发约定](AGENTS.md)
- [当前工作主线](docs/developer/current-work.md)
- [文档入口](docs/README.md)
- [架构与产品边界](docs/architecture/README.md)

旧治理流程、状态机和审批工具已删除。直接实现任务，运行相关构建、测试或性能实验，
如实说明结果与未验证项。历史实验和原始数据保留用于追溯。

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
