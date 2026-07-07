# 外部依赖：怎么装、哪些必需、哪些可选

> 跨平台通用。平台专属的安装命令在各平台文件里，这里讲**每个依赖是什么角色、怎么被 CMake 发现、装不装的后果**。

## 总览

| 依赖 | 类别 | 缺了会怎样 | 怎么被发现 |
|---|---|---|---|
| C++ 编译器 / CMake / Ninja / 汇编器 | **硬门槛** | CMake FATAL_ERROR，配置失败 | CMake 自带检查 |
| **ggml** | **自动拉取**（FetchContent） | 有网自动下；无网且无缓存则配置失败 | `FetchContent`（pin commit） |
| **googletest** | **自动拉取**（FetchContent） | 同上（仅影响测试，`-DHYLE_BUILD_TESTS=OFF` 可跳） | `FetchContent`（pin tag v1.15.2） |
| **OpenSSL** | 可选能力 | TLS（io-08）不编，其余全绿 | `find_package(OpenSSL QUIET)` |
| **CUDA Toolkit** | 可选能力 | GPU 后端不编，host fallback 顶上，其余全绿 | `find_package(CUDAToolkit QUIET)` |
| **cuDNN / TensorRT** | 可选能力（更后） | 当前里程碑不依赖，完全可不装 | 手动 `-I/-L`（未来算子/推理路径） |

**"可选能力 = 可选依赖"铁律**：OpenSSL/CUDA 找到才启用对应代码，找不到静默跳过、配置照样成功。所以一台**只装了硬门槛**的机器 `cmake` 就能过（ggml/gtest 联网自动下），只是没有 TLS/GPU。

---

## FetchContent：ggml 与 googletest 的自动拉取（cil2cpp 形式）

**机制**：`ggml`（根 `CMakeLists.txt`）和 `googletest`（`tests/CMakeLists.txt`）都用 CMake 的 `FetchContent`，**pin 到确定版本**：

- ggml → `GIT_TAG eced84c86f8b012c752c016f7fe789adea168e1e`（v0.15.3）
- googletest → `GIT_TAG v1.15.2`

**默认行为（开箱即用，需联网）**：新机器 clone HYLE → 跑 `cmake` → FetchContent **自动联网 clone** 这两个依赖到 `.deps/<dep>-src`，编进 `.deps/<dep>-build`。**你什么都不用准备**（除了有网）。

**持久缓存（删 build 不重下）**：根 `CMakeLists.txt` 把 `FETCHCONTENT_BASE_DIR` 钉到源码树内的 `.deps/`（`.gitignored`）。所以：
- `rm -rf build` 重新配置 → **不重新下载**（源码还在 `.deps/<dep>-src`）
- 同一机器开第二个 build 目录 → 复用同一 `.deps`

> 这套「pin + 持久 `.deps`」借鉴 cil2cpp 的 `FETCHCONTENT_BASE_DIR` 用法。`.deps` 是**每台机器各自的**（Windows `C:\public\project-hyle\hyle\.deps`、WSL `/root/hyle/.deps`、mac `~/hyle/.deps`），不跨机共享。

### 有备份的机器：预填缓存免下载（可选加速）

若机器上已有一份**该 pin 版本**的 ggml 源码（本仓库开发机上 `C:\public\project-hyle\ggml` 就是 ggml pin commit 的一份备份），可以让 CMake 直接用它、**跳过联网下载**——用 CMake 官方的 `FETCHCONTENT_SOURCE_DIR_<NAME>` 命令行参数：

```bash
cmake -S . -B build -G Ninja \
  -DFETCHCONTENT_SOURCE_DIR_GGML=C:/public/project-hyle/ggml
```

- **只是命令行参数，不写进 CMakeLists**——CMakeLists 保持纯 GIT 形式（开箱即用），是否用本地源码是**每台机器构建时**的选择。
- `<NAME>` 是 `FetchContent_Declare` 里的名字大写：`GGML`、`GOOGLETEST`。
- 那份源码**必须是对应的 pin 版本**（ggml 备份已 checkout 到 `eced84c8`）。版本不符会编译出错或行为不一致。
- 这不改变构建产物——只是源码来源从「网络」变成「本地目录」，pin 版本一致故等价。

> **为什么不把本地源码写死进 CMakeLists（用 SOURCE_DIR）**：那样每台机器都得先手动把源码铺到某个固定路径才能构建，违背「clone 完直接 cmake 就行」的开箱即用意图。默认联网下载、有备份的机器用命令行 override 免下载——两全。

**ggml 的构建开关（HYLE 强制设定，见根 CMakeLists）**：`GGML_OPENMP=OFF`（compute-05 发现 10：n_threads=1 叶子，不让 ggml 起 OpenMP 线程）、`GGML_BUILD_TESTS/EXAMPLES=OFF`、`GGML_CUDA_FA=OFF`（跳 ~120 个 FlashAttention kernel，桥只用 GEMM，省首编时间）、`GGML_CUDA` 跟随是否找到 CUDA。

**关掉 ggml（无 compute 需求时）**：`-DHYLE_COMPUTE_GGML=OFF` → 不拉 ggml、不编 ggml_bridge，compute 其余（01–04）照编。

---

## OpenSSL（io-08 TLS，可选）

TLS 是可选能力：`find_package(OpenSSL)` 找到就编 `tls.cc` + 打开 `HYLE_IO_TLS`；找不到完全不编，**零 OpenSSL 依赖**。**动态链**（安全库应跟随系统 CVE 更新，不静态焊死）。

- **Linux/WSL**：`sudo apt install libssl-dev`（系统包，零配置，装完重配即启用）。
- **Windows**：无系统级 MSVC OpenSSL 开发库，用 **vcpkg**（详见 [`windows.md`](./windows.md) 依赖节）。
- 运行 TLS 测试要 OpenSSL 的 DLL 在 PATH（Windows）——见平台文件。

---

## CUDA Toolkit（compute GPU 后端，可选）

`find_package(CUDAToolkit QUIET)` 找到 → `HYLE_COMPUTE_CUDA=ON` + `enable_language(CUDA)` + 编 `cuda_backend.cu`；找不到 → 只编 host fallback，**零 CUDA 依赖**，其余全绁。

- **本仓库统一 CUDA 13.3**（RTX 4070 / sm_89）。Windows toolkit 在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`；WSL 在 `/usr/local/cuda-13.3`（软链 `/usr/local/cuda`）。装法见平台文件。
- **编译/链接不需要 PATH**（CMake 走 `find_package`）；**运行 GPU 测试**要 CUDA runtime DLL 在 PATH（Windows）。
- **mac 无 NVIDIA GPU** → CUDA 后端 `#if` 门控编不进，走 host / 未来 Metal。

## cuDNN / TensorRT（更后，当前可不装）

**compute-01～05 不依赖 cuDNN/TensorRT**——它们是未来算子/推理图优化路径的加速档。当前完全可以不装。装齐的机器信息见 `reference_cuda_env`（cuDNN 9.24 / TensorRT 11.1，Win + WSL 均已装并端到端验证）。要用时按那里的 `-I/-L/-l` 指路。
