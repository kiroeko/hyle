# HYLE

HYLE（HYbrid LEvels）是面向 AI 推理与工程计算的 C++23 基础加速库。它希望让内存、数据结构、
SIMD、调度、I/O 和异构计算在同一套依赖与生命周期合同下组合，而不是让每个子系统各自维护
线程池、allocator 和事件循环。

当前首要需求锚是可执行的本地量化推理引擎；工程计算 workload 用作泛化留出。HYLE 只提供
可复用机制，模型格式、token/KV 语义、batching 和 serving 策略留在消费方。项目仍处于预发布
重审期，不能把旧 benchmark 或历史“已收口”记录当成当前证书。

## 产品边界

HYLE 面向 C++ 用户提供三种交付形式：

- 完整源码；
- headers + static libraries；
- headers + shared libraries。

三者都是 C++ 交付，不承诺跨编译器/STL 的 C++ 二进制 ABI。项目没有独立的原生 C 产品，也
暂不提供 Native FFI；真实外语消费者出现后，再用窄的运行时类型擦除边界单独设计和验证。

性能目标不是“每个微基准都赢”，而是逐能力达到合理的算法复杂度，在真实合同、正常 workload
和强 peer 下排除有意义的劣势，并明确延迟、吞吐、内存和可移植性的取舍。未通过仪器资格与
消费者相关性门的亚纳秒差异只用于诊断，不能塑造公共 API。

## 开始使用

- [C++ 使用指南](docs/user/usage-guide.md)
- [三种交付与集成](docs/user/distribution.md)
- [可运行样例](samples/README.md)
- [开发环境与离线依赖](docs/developer/setup/README.md)

## 项目文档

所有人读文档按职责分为四类，详见 [文档入口](docs/README.md)：

- [architecture](docs/architecture/README.md)：背景、愿景、架构和决策依据；
- [implementation](docs/implementation/README.md)：任务、调查、结果与当前施工状态；
- [developer](docs/developer/README.md)：维护规则、review、构建测试、性能仪器和排障；
- [user](docs/user/README.md)：C++ 用户的选型、用法、生命周期与陷阱。

自动化维护者先读仓库级 [agent 入口](AGENTS.md)。模块当前状态只以
[implementation/status](docs/implementation/status.md) 及其机器源为准；不要从本页推断当前施工模块。
模块审查方法见
[module closure methodology](docs/developer/review/module-closure-methodology.md)，真实消费者边界见
[inference engine north star](docs/architecture/inference-engine-north-star.md)。

## 仓库结构

```text
include/hyle/   C++ 公开头
src/            按模块组织的实现
tests/          正确性、并发与构建合同测试
bench/          归因与同合同性能实验
samples/        独立 C++ 消费工程及内部 conformance fixtures
cmake/          组件图、依赖与发布机器合同
scripts/        构建、审查、性能和文档工具
docs/           四类人读文档
evidence/       与候选绑定的原始 CSV/JSON/log
third_party/    已审查的 vendored/ported 资产
```
