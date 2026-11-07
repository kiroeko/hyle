# HYLE

**[Official website / 官网 →](https://axiomate.net/hyle)**

**HYLE (HYbrid LEvels)** is a pre-release C++23 inference engine for a single machine.
Its foundation libraries can also be built, installed and used independently. Builds target Windows,
Linux and macOS; backend components include CPU, CUDA, Apple Metal and Vulkan. Actual model,
format and backend coverage depends on the configuration and validation described below.

HYLE 是面向单机的推理引擎。推理引擎内核负责已接纳工作的准备、模型计算、生成与状态推进；
引擎的外围部分负责模型制品、请求、会话、调度和结果交付。内核包含模型专属实现，并使用可独立构建的
Compute、Graph、内存、调度、I/O 等基础库；推理内核也不同于单个 CUDA／Metal kernel。
当前完整入口仍以 Qwen 为主，内核与引擎外围尚未按这个职责边界完整拆开，见[引擎概览](docs/architecture/engine-overview.md)。

当前开发顺序是先审计、修正并冻结推理内核，再设计、开发并冻结内核之外的引擎部分，完成后进入 M2。
冻结确定职责与执行合同，不冻结具体 kernel 或性能实现；现有多模型、多平台和量化组件的验证范围按实际结果说明。

## 与llama.cpp相比

Qwen3.5-2B、RTX4070、C32768、默认Release。2026-11-05与官方CUDA版llama.cpp **b12063**同窗交错，各臂3个独立进程；下表为完整八轮decode，包含请求内的冷开销。

| 格式／Hyle模式 | decode tok/s Hyle／llama.cpp | Hyle速度÷llama.cpp |
|---|---|---|
| BF16／普通 | 104.81／100.62 | 104.2% |
| BF16／MTP K1 | 141.94／116.52 | 121.8% |
| Q8_0／普通 | 175.04／164.33 | 106.5% |
| Q8_0／MTP K1 | 208.77／181.37 | 115.1%（数值仅诊断） |
| Q4_K_M／普通 | 247.70／229.51 | 107.9% |
| Q4_K_M／MTP K1 | 254.12／236.62 | 107.4% |

BF16 MTP使用准备好的32K草稿子集，量化目标MTP使用完整草稿词表。普通路径另做8192位置数值回归；该评估不覆盖MTP。†表示三进程范围重叠。标准吞吐、长prompt、逐轮落后项、启动与资源代价见下页；这些速度比不等于本次修复对旧main的提速。
**[查看完整性能对比：逐轮结果、长prompt、启动、显存与复现方法 →](docs/developer/performance/decode.md)**

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

完整结果见首页上方的[性能摘要及详细对照](docs/developer/performance/decode.md)；
当前任务由[看板](docs/developer/inference-progress.md)维护。
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
