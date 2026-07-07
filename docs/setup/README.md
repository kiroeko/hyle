# HYLE 环境准备（面向贡献者）

> 这个文件夹只回答一个问题：**从零把一台机器准备成能编译 / 测试 / 性能测试 HYLE 的开发环境。**
> 面向**开发 HYLE 本身**的贡献者（不是把 HYLE 当依赖用的库使用者——那看 [`../guide/usage-guide.md`](../guide/usage-guide.md)）。
> 内容以**实测**为准，慢变；命令 / 版本变了要回填。

## 按平台分文件

三个平台各有独立文件，把「构建 / 测试 / 性能测试」三件事一次讲清（含各自的坑）：

| 平台 | 文件 | 角色 |
|---|---|---|
| Windows（MSVC） | [`windows.md`](./windows.md) | 主开发平台之一；GPU（CUDA）真机 |
| Linux / WSL（GCC14） | [`linux-wsl.md`](./linux-wsl.md) | **lock-free TSan 唯一闸门**；GPU（CUDA）真机；9p 铁律 |
| macOS（Apple Silicon） | [`macos.md`](./macos.md) | ARM/NEON + kqueue 真机；无 CUDA（CPU-only） |

依赖（跨平台，可选能力）单独一文件：

| 主题 | 文件 |
|---|---|
| 外部依赖（OpenSSL / CUDA / cuDNN / TensorRT / ggml / googletest）怎么装、哪些必需哪些可选 | [`dependencies.md`](./dependencies.md) |

## 最小心智模型（读任何平台文件前先懂这个）

HYLE 的依赖分三类，决定了「新机器 `cmake` 会不会失败」：

1. **硬门槛（必需，缺了 CMake 直接 FATAL_ERROR）**：C++ 编译器（GCC14+ / Clang18+ / MSVC19.43+）、CMake 3.25+、Ninja、汇编器（MASM/GAS）。这些**不自动装**，人得先装好——见各平台文件第一节。
2. **自动拉取（联网自动，无需人工）**：`ggml`、`googletest` 走 **FetchContent**——新机器首次 `cmake` 会**自动联网 clone** 到 `.deps/`（[`dependencies.md` §FetchContent](./dependencies.md)）。**有网就开箱即用**；有备份的机器可预填缓存免下载（见该节）。
3. **可选能力（找到才启用，找不到静默跳过）**：OpenSSL（TLS）、CUDA（GPU 后端）。**缺了不报错**，只是对应功能不编——`cmake` 照样成功、其余全绿。cuDNN/TensorRT 更靠后（当前里程碑不依赖）。

> **一句话**：硬门槛靠人装（第 1 类），ggml/gtest 自动下（第 2 类），GPU/TLS 可有可无（第 3 类）。所以「新机器直接 `cmake` 失败」几乎总是第 1 类没装齐，或第 2 类**没网**。

## 「能不能自动化装好一切？」——诚实回答

**部分能，部分不能，现状是「不自动、靠文档」：**

- **第 2 类（ggml/gtest）已经自动**：FetchContent 就是自动化，联网即备齐，无需人工。
- **第 1 类（编译器/CMake/Ninja）不自动、也不该由本项目自动**：装系统级工具链需要 admin/包管理器权限，跨发行版差异大（apt/brew/VS installer/vcpkg 各不同），一个 CMake 脚本去装它们既不安全也不可移植。**业界惯例也是文档 + 手动**（或 CI 镜像预装）。本项目选择：**把前提写成人类可照抄执行的命令**（各平台文件第一节就是可复制的安装命令块），而不是藏一个魔法脚本。
- **第 3 类（CUDA/OpenSSL）不自动**：CUDA Toolkit 几 GB、要配驱动，OpenSSL 在 Windows 要 vcpkg——都是重量级、机器相关，只在你**要**那个能力时才装。文档给出每个的确切安装步骤（[`dependencies.md`](./dependencies.md)）。

**可改进方向（未做，非阻塞）**：可以提供一个 `bootstrap.ps1`/`bootstrap.sh` 每平台脚本，把「第 1 类工具链 + 可选第 3 类」的安装命令封装起来（本质是把各平台文件第一节的命令块脚本化）。目前没做，因为三平台维护成本 vs 收益不划算——文档命令块已经是「人能看懂、能逐条执行」的，且贡献者机器通常已有工具链。**若未来贡献者增多，值得做。**

## 验证一切就绪（各平台通用终点）

配置 + 构建 + 跑几个测试，绿了就能开工：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # 平台专属 flag 见各文件
cmake --build build
cd build/bin && ./hyle_sched_test   # 换任意模块测试
```

- 看配置日志确认可选能力状态：`OpenSSL ... found` / `CUDA Toolkit ... found` / `ggml via FetchContent`。
- lock-free 改动**必须**过 WSL TSan（[`linux-wsl.md`](./linux-wsl.md)）。
- 性能数字**必须**在原生文件系统（非 9p / 非网络盘）跑，见各平台「性能测试」节。
