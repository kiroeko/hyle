# HYLE

**[Official website / 官网 →](https://axiomate.net/hyle)**

**HYLE (HYbrid LEvels)** is a pre-release C++23 inference engine for a single machine.
Its foundation libraries can also be built, installed and used independently. Builds target Windows,
Linux and macOS; backend components include CPU, CUDA, Apple Metal and Vulkan. Actual model,
format and backend coverage depends on the configuration and validation described below.

HYLE 是面向单机的推理引擎。推理引擎内核负责已接纳工作的准备、模型计算、生成与状态推进；
引擎的外围部分负责模型制品、请求、会话、调度和结果交付。内核包含模型专属实现，并使用可独立构建的
Compute、Graph、内存、调度、I/O 等基础库；推理内核也不同于单个 CUDA／Metal kernel。
公共 CLI、token 工作进程与文本服务通过模型提供者装配；当前完整运行适配器是 Qwen3.5。
模型保留数学、权重解释与完整状态，公共层负责请求选择、交付和会话，见[引擎概览](docs/architecture/engine-overview.md)。

通用装配的 R1–R4 职责与执行合同已冻结；已接入共享权重的多执行方案、服务会话与冷期制品复用，见[通用引擎设计](docs/architecture/engine-design.md)。
冻结确定职责与执行合同，不冻结具体 kernel 或性能实现；现有多模型、多平台和量化组件的验证范围按实际结果说明。

## 与llama.cpp相比

Qwen3.5-2B、RTX 4070、C32768、默认 Release。2026-11-08，Hyle 公共 token 工作进程 `206e59182` 与官方 CUDA 版 llama.cpp **b12063** 同窗交错，每臂 3 个独立进程。八轮短提示，表中排除首轮；综合速度包含后七轮 prefill 与 decode。

| 格式／模式 | decode tok/s Hyle／llama.cpp | 综合 tok/s Hyle／llama.cpp |
|---|---:|---:|
| BF16／普通 | 102.88／98.59 | 100.94／94.54 |
| Q8_0／普通 | 175.62／160.56 | 169.97／150.84 |
| Q4_K_M／普通 | 249.96／225.01 | 236.88／207.04 |
| BF16／默认 MTP | 160.88／116.82 | 154.68／110.01 |
| Q8_0／默认 MTP | 233.54／180.66 | 222.00／166.53 |
| Q4_K_M／默认 MTP | 321.58／244.25 | 298.32／218.58 |

Hyle 默认 BF16 为 K3，量化格式为子集头 K1；上游显式 K1，本窗口未扫描上游其他深度。Hyle 使用 token 管道的交付时间，上游使用 HTTP 服务的内部计时，两者端点并不完全相同。完整进程、A/A、首轮和其他负载见下页；结果不代表其他设备或模型。
**[查看完整性能对比：逐轮结果、长prompt、启动、显存与复现方法 →](docs/developer/performance/decode.md)**

| 想做什么 | 从这里开始 |
|---|---|
| 构建并运行模型 | [推理引擎指南](docs/user/inference.md)；[文本 CLI 与 Responses 接口](docs/user/text-api.md) |
| 在 C++ 工程中使用基础库 | [使用指南](docs/user/usage-guide.md)与[样例](samples/README.md) |
| 看架构与开发方向 | [架构入口](docs/architecture/README.md)、[两项核心期许](docs/architecture/inference-principles.md) |
| 看当前进度及下一项 | [推理看板](docs/developer/inference-progress.md) |
| 查构建、测试或排障方法 | [开发指南](docs/developer/README.md) |

## 现状

- **完整文本入口是 Qwen**：普通生成、greedy／随机采样、MTP、多轮／多会话、共享前缀及有界缓存恢复。
  当前日常完整验证模型为 Qwen3.5-2B；其它制品与较大模型的实际运行范围见[模型覆盖](docs/implementation/model-operators.md)。
- **单机运行**：本阶段完整引擎验证使用 Windows、CUDA 和 Qwen3.5-2B；CPU、Apple Metal 及其他模型的完整引擎路径尚未在本阶段验证。已有后端组件不等于完整路径已验证。
- **文本服务**：本地 tokenizer／Jinja，具名执行与采样选择、Responses 文本与 SSE、进程内续接／分叉、受限 KV 驻留和可信前缀域。冷期制品可独立生成或在加载时按配方生成，之后复用；功能与限制见[接口指南](docs/user/text-api.md)。
- **尚未交付**：通用模型目录加载、C++ tokenizer、完整服务 API、图片输入及多卡／多机。
  GLM、DeepSeek、Kimi 等目前是组件级实现，没有完整引擎入口。

运行需要先准备本地权重清单。token CLI 与文本工作进程共用模型提供者和请求入口；文本 CLI 和 Responses 可用本地 tokenizer 处理文本。Qwen 重放、探针及 JSON 报告使用独立诊断目标。初始化 C++ 构建不会下载模型或安装 tokenizer 环境。

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
<build-dir>/bin/hyle generate --inputs .scratch/qwen-input --tokens .scratch/qwen-input/prompt.ids --draft-count 0 --max-output-tokens 64 --capacity 1024
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
