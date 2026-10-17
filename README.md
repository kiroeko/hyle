# HYLE

HYLE（HYbrid LEvels）是面向本地 AI 推理、异构计算、资源流送和低延迟服务的 C++23 基础加速库。
它提供可组合的内存、数据结构、SIMD、调度、I/O、Compute 和 Graph；模型格式、tokenizer、
生成与采样策略、KV 缓存策略和 serving 政策由推理引擎拥有。

当前推理目标是多系统上的单机 CPU、CPU＋一张 GPU、单 GPU。
项目仍处于预发布阶段，支持范围和限制以对应接口、后端及实际验证环境为准。

## 开始使用

安装 Python 3.10+、CMake 3.26+ 与受支持的 C++ 工具链后，三平台使用同一入口：

```text
python scripts/init.py
python scripts/build.py
```

默认 CPU、Release、共享库，测试关闭。用 `init.py --backends cpu,cuda` 选择所需编译后端；
可选 `cpu`、`cuda`、`metal`、`vulkan` 或逗号组合，运行时的设备和混合执行策略由引擎决定。
开启测试时先 `init.py --tests`，构建后运行 `python scripts/test.py`。
各脚本均提供 `--help`，Unix 可使用 `python3`。

接下来按一条路径阅读，不必先通读架构和历史记录：

1. [环境与构建](docs/developer/setup/README.md)：准备工具、选择后端、构建和测试。
2. [C++ 起步](docs/user/usage-guide.md)：接入自己的工程，运行最小消费者。
3. [模块手册](docs/user/README.md)：按实际任务查 Compute、Graph 或基础模块。

三种形式均为 C++ 交付，不承诺跨编译器/STL 的二进制 ABI；当前没有独立 C API 或外语绑定。

## 维护与设计

修改代码见[开发与维护](docs/developer/README.md)，理解取舍见[架构](docs/architecture/README.md)。
[文档导航](docs/README.md)区分使用、维护、设计和实施背景；
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
