# HYLE

**HYLE (HYbrid LEvels)** is a pre-release C++23 inference engine for CPU, one GPU, or CPU + one GPU.
Its foundation libraries can also be built, installed and used independently. Windows, Linux and
macOS are supported; available compute backends are CPU, CUDA, Apple Metal and Vulkan.

HYLE 是面向单机的推理引擎。引擎负责模型、生成、会话与请求；Compute、Graph、内存、调度、I/O
等基础库保持模型无关，可单独构建和使用。项目目前集中完善推理底座。

| 想做什么 | 从这里开始 |
|---|---|
| 构建并运行模型 | [推理引擎指南](docs/user/inference.md) |
| 在 C++ 工程中使用基础库 | [使用指南](docs/user/usage-guide.md)与[样例](samples/README.md) |
| 看架构与开发方向 | [架构入口](docs/architecture/README.md)、[两项核心期许](docs/architecture/inference-principles.md) |
| 看当前进度及下一项 | [推理看板](docs/developer/inference-progress.md) |
| 查构建、测试或排障方法 | [开发指南](docs/developer/README.md) |

## 现状

- **完整文本入口是 Qwen**：普通生成、greedy／随机采样、MTP、多轮／多会话、共享前缀及有界缓存恢复。
  当前日常完整验证模型为 Qwen3.5-2B；其它制品与较大模型的实际运行范围见[模型覆盖](docs/implementation/model-operators.md)。
- **单机运行**：CPU、单 GPU 或 CPU＋一张 GPU；支持范围由实际后端、制品、布局和设备能力决定。
- **尚未交付**：通用模型目录加载、C++ tokenizer、应用／HTTP 服务 API、图片输入及多卡／多机。
  GLM、DeepSeek、Kimi 等目前是组件级实现，没有完整引擎入口。

当前输入需要用 Python 准备本地权重清单和 token IDs；初始化 C++ 构建不会下载模型或安装 tokenizer 环境。

## 快速开始

先按[构建指南](docs/developer/setup/README.md)准备工具链，在仓库根目录运行：

```text
python scripts/init.py
python scripts/build.py --jobs 2
```

新配置默认为 **CPU、Release、共享库、engine，测试和 trace 关闭**；已有配置保留选择。
CUDA、混合后端、独立基础库和 SDK 安装均在构建指南中选择。

准备本地模型并运行普通生成：

```text
python engine/models/qwen35/prepare.py --model LOCAL_MODEL --output .scratch/qwen-input --prompt "Please calculate 2 + 3. Answer briefly."
<build-dir>/bin/hyle --inputs .scratch/qwen-input --mtp off --steps 64 --capacity 1024
```

将 `<build-dir>` 换成构建脚本打印的目录，Windows 使用 `hyle.exe`。
MTP、采样、后端及会话配置继续看[推理引擎指南](docs/user/inference.md)。

## 性能与进度

最新完整普通／MTP、单轮／多轮速度、内存及测量限制见[decode 性能资料](docs/developer/performance/decode.md)；
当前任务与粗估进度只在[看板](docs/developer/inference-progress.md)维护。
结果限于记录的模型、平台、配置及参考版本，不能从组件通过推定整模型性能。
基础库对照见[模块性能](docs/developer/performance/modules.md)。

## 平台与工具链

Windows、Linux 使用 x86_64；macOS 使用 Apple Silicon。需要 C++23、Python 和 CMake，
编译器、CUDA／Metal／Vulkan 版本及系统条件统一见[环境准备](docs/developer/setup/README.md)。
源码、静态 SDK 和动态 SDK 使用相同 C++ API；部署与 ABI 条件见[交付与集成](docs/user/distribution.md)。

## 文档

[文档索引](docs/README.md)按使用、开发、架构和原始证据组织。首次使用无需通读实验记录。
[用户手册](docs/user/README.md)按任务导航；接口签名与局部约束以公开头文件为准。

## 仓库结构

```text
include/hyle/  公开 C++ 头文件      src/       基础模块实现
engine/        引擎与模型接入       tests/     正确性与构建测试
samples/       C++ 消费者样例       bench/     性能实验
scripts/       初始化与开发工具     cmake/     构建与安装
docs/         使用、开发和架构     docs/evidence/  原始实验记录
```

## 许可

[Apache License 2.0](LICENSE)。第三方组件保留各自许可证，见[NOTICE](NOTICE)、
[第三方清单](THIRD_PARTY_NOTICES.md)及[许可边界](docs/architecture/licensing.md)。
