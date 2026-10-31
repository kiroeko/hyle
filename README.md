# HYLE

**HYLE (HYbrid LEvels)** is a C++23 inference engine for a single machine: CPU only, one GPU, or
CPU + one GPU, on Windows, Linux and macOS, with CPU, CUDA, Apple Metal (MLX) and Vulkan backends.
Its foundation libraries (memory, containers, SIMD, fiber scheduling, async I/O, Compute and
Graph) can also be built, installed and used on their own. The project is pre-release: the full
text path runs Qwen3.5 with ordinary decoding, multi-token prediction and multiple sessions, while
development currently focuses on the inference foundation. The real-model engine drives that
work; a general model loader, a C++ tokenizer and a server API are not built yet. Licensed under
Apache-2.0.

---

HYLE 是面向单机的 C++23 推理引擎，目标是在 Windows、Linux、macOS 上做纯 CPU、单 GPU 和
CPU＋一张 GPU 的推理。引擎之下是一组可以单独构建和使用的基础库，覆盖内存、容器、SIMD、
调度、I/O、Compute 与 Graph。引擎负责模型接入、生成、会话和请求编排；基础库不依赖任何模型。

第一次使用从[用户手册](docs/user/README.md)开始；理解模块分工看[架构入口](docs/architecture/README.md)。
当前正在做什么、接下来做什么，统一看[推理进度](docs/developer/inference-progress.md)。

## 现状

项目处于预发布阶段，当前主阶段是完善底层推理支持，收敛普通推理、MTP、多会话和多请求的
共同执行架构，再继续优化整条推理路径。项目内引擎用真实 Qwen 模型检验这些能力与性能，
通用模型应用和服务接口随后推进。

已经可用：

- **完整文本推理**：Qwen3.5-2B 官方 BF16，以及 GPTQ、NVFP4、GGUF Q4/Q8、compressed-tensors
  INT4 等量化制品。Qwen3.8-27B 的官方 FP8 制品在 CPU 上、Unsloth GGUF Q4 制品在 CPU 和 CPU＋CUDA
  上跑通过完整短流程。
- **生成**：greedy 与随机采样（温度、top-k、top-p），MTP 多草稿投机解码。
- **多会话**：多轮对话、共享前缀复用、有界缓存与淘汰后恢复（重算，或从主机内存镜像换回）、
  多请求合批；请求层支持运行中接纳、取消和输出背压。
- **后端**：CPU、CUDA、Apple Metal、Vulkan；CPU＋GPU 混合时可把前 N 层的 FFN 放到 CPU。

尚未完成：

- 通用模型目录加载、C++ tokenizer、chat 模板。现在需要先用 Python 脚本把模型和 prompt
  准备成权重清单与 token ID。
- HTTP 服务与应用 API、图片输入。
- 其它模型家族（GLM、DeepSeek、Kimi 等）目前只有组件级实现和检查，还没有完整引擎入口。

各制品、平台的实际检查范围见[模型与算子覆盖](docs/implementation/model-operators.md)和
[推理工程计划](docs/implementation/inference-roadmap.md)。

## 快速开始

先按[环境准备](docs/developer/setup/README.md)装好工具链，再在仓库根目录运行：

```text
python scripts/init.py
python scripts/build.py --jobs 2
```

新配置默认是 **CPU、Release、共享库、engine 及其依赖，测试和 trace 关闭**。
已有配置保留原选择；初始化按所选产品准备依赖并复用缓存，不安装系统编译器、驱动或模型。

常用选择如下；完整配置、切换、测试和安装方法见[构建指南](docs/developer/setup/README.md)。

| 需要什么 | 传给 `init.py` 的参数 |
|---|---|
| CUDA 引擎 | `--name qwen-cuda --components engine --backends cuda` |
| CPU＋CUDA 能力 | `--name qwen-mixed --components engine --backends cpu,cuda` |
| 只使用基础库 | `--name libraries --components mem,container` |

具名初始化后用同一名称构建，例如 CUDA：

```text
python scripts/init.py --name qwen-cuda --components engine --backends cuda
python scripts/build.py --name qwen-cuda --target hyle_engine --jobs 2
```

准备本地输入并运行普通生成。分词需要另行准备兼容的 Transformers 环境；脚本核对本地模型文件，
只生成权重偏移和 token ID，不下载也不复制权重：

```text
python engine/models/qwen35/prepare.py --model LOCAL_MODEL --output .scratch/qwen-input --prompt "Please calculate 2 + 3. Answer briefly."
<build-dir>/bin/hyle --inputs .scratch/qwen-input --mtp off --steps 64 --capacity 1024
```

将 `<build-dir>` 替换为构建脚本打印的目录，Windows 使用实际 `hyle.exe` 路径。
MTP 需要额外原权重，准备时加 `--mtp`，运行时用 `--mtp on`；后端选择、混合执行、采样与多会话见
[推理引擎指南](docs/user/inference.md)。基础库的 C++ 集成从[使用指南](docs/user/usage-guide.md)开始。

## 性能与进度

最新普通／MTP、单轮／多轮的正式对照、内存和测量边界统一见
[decode 性能资料](docs/developer/performance/decode.md)，当前任务与未完成项见
[推理进度](docs/developer/inference-progress.md)。结果限于记录中的模型、平台、配置及固定参考版本，
不代表任意模型或硬件已达标。基础模块与 Abseil、Folly、oneTBB、nghttp2 等的对照见
[模块性能数据](docs/developer/performance/modules.md)。

## 平台与工具链

| 系统 | 架构 | 计算后端 |
|---|---|---|
| Windows | x86_64 | CPU、CUDA、Vulkan |
| Linux | x86_64 | CPU、CUDA、Vulkan |
| macOS | ARM64（Apple Silicon） | CPU、Metal、Vulkan |

需要 Python 3.10+、CMake 3.25+，以及 MSVC 19.43+ / clang-cl 18+、GCC 14+ / Clang 18+ 或
AppleClang 18+。CUDA 需要 CUDA Toolkit 13.3+。Linux AArch64 可以构建部分基础模块。
各平台细节见[环境准备](docs/developer/setup/README.md)。

交付形式有完整源码、头文件＋静态库、头文件＋动态库三种，API 相同；不承诺跨编译器或
跨标准库的 C++ 二进制兼容，详见[交付与集成](docs/user/distribution.md)。

## 文档

| 想做的事 | 入口 |
|---|---|
| 运行推理引擎 | [推理引擎指南](docs/user/inference.md) |
| 在自己的程序中使用基础库 | [用户手册](docs/user/README.md) |
| 理解模块职责和设计取舍 | [架构](docs/architecture/README.md) |
| 构建、测试、排障、测性能 | [开发与维护](docs/developer/README.md) |
| 查看引擎当前进度 | [推理进度](docs/developer/inference-progress.md) |

全部文档的分工见[文档索引](docs/README.md)，可编译的消费者工程见[样例](samples/README.md)。

## 仓库结构

```text
include/hyle/   公开 C++ 头文件
src/            模块实现
engine/         推理引擎与模型接入
tests/          正确性、并发与构建测试
bench/          性能实验
samples/        独立的 C++ 消费者工程
cmake/          组件、依赖锁与安装规则
scripts/        初始化、构建、测试与诊断脚本
docs/           用户、开发、架构文档
docs/evidence/  实验的原始数据与日志
third_party/    第三方源码与补丁
```

## 许可

HYLE 采用 [Apache License 2.0](LICENSE)，归属说明见 [NOTICE](NOTICE)。第三方组件保留各自的
许可证，清单见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，许可政策见
[许可与第三方代码边界](docs/architecture/licensing.md)。
