# 外部依赖：怎么装、哪些必需、哪些可选

> 跨平台通用。平台专属的安装命令在各平台文件里，这里讲**每个依赖是什么角色、怎么被 CMake 发现、装不装的后果**。

## 总览

| 依赖 | 类别 | 缺了会怎样 | 怎么被发现 |
|---|---|---|---|
| C++ 编译器 / CMake / generator / 汇编器 | **硬门槛** | CMake FATAL_ERROR，配置失败 | CMake 自带检查；Ninja 不是唯一 generator |
| **ggml** | **自动拉取**（FetchContent） | 有网自动下；无网且无缓存则配置失败 | `FetchContent`（pin commit） |
| **googletest** | **自动拉取**（FetchContent） | 同上（仅影响测试，`-DHYLE_BUILD_TESTS=OFF` 可跳） | `FetchContent`（pin tag v1.15.2） |
| **nghttp2** | **自动拉取**（FetchContent） | 仅选择 io 时取得；提供 HTTP/2 协议核 | `FetchContent`（pin commit v1.69.0） |
| **volk + Vulkan-Headers** | **自动拉取**（FetchContent，compute-06） | 全 pin `vulkan-sdk-1.4.341.0`，落 build tree `_deps`；缺网首配失败 | `FetchContent`（两个 pin commit） |
| **glslang** | 测试/benchmark shader 工具 | 只在 compute tests/bench 打开时拉取；生产 SDK 不需要 | `FetchContent`（pin commit） |
| **Vulkan loader + ICD/driver** | 可选能力（compute-06 运行时） | 无则 Vulkan 后端运行时禁用（`vulkan_vtable()` 返 null），其余全绿 | volk 运行时 dlopen `vulkan-1.dll`/`libvulkan.so`/`libvulkan.dylib` |
| **OpenSSL** | 可选能力 | `HYLE_IO_TLS=AUTO/OFF` 可无 TLS；ON 时缺失即配置失败 | `find_package(OpenSSL QUIET)` |
| **CUDA Toolkit** | 可选能力 | `HYLE_COMPUTE_CUDA=AUTO/OFF` 可只用 host；ON 时缺失即配置失败 | `find_package(CUDAToolkit QUIET)` |
| **cuDNN / TensorRT** | 可选能力（更后） | 当前里程碑不依赖，完全可不装 | 手动 `-I/-L`（未来算子/推理路径） |

**"可选能力 = 可选依赖"铁律**：OpenSSL/CUDA 都是 `AUTO|ON|OFF`。AUTO 找到就启用、找不到降级；
ON 要求能力存在，否则配置失败；OFF 连依赖都不探测。只有本地探索适合 AUTO，正式 SDK/CI 应写死 ON/OFF，
避免构建机环境暗中改变产物。

---

## FetchContent：固定版本源码依赖

**机制**：重型依赖由所属模块的 dependency 入口取得，测试依赖由 tests 入口取得；都用 CMake
`FetchContent` **pin 到确定版本**：

- ggml → `GIT_TAG eced84c86f8b012c752c016f7fe789adea168e1e`（v0.15.3）
- nghttp2 → `GIT_TAG 68cb6900fde14c77f0cd7add0e094a862960eb99`（v1.69.0）
- Vulkan-Headers / volk → `vulkan-sdk-1.4.341.0` 对应固定 commit
- glslang → compute tests/benchmark 需要 shader 工具时的固定 commit
- googletest → `GIT_TAG v1.15.2`

**默认行为（开箱即用，需联网）**：新机器 clone HYLE → 跑 `cmake` → FetchContent 自动联网 clone
到 `<hyle-build>/_deps/<dep>-src`，生成物进入同一 build tree 的 `_deps/<dep>-build`。删除 build tree
会一并删除这些内容，这是有意的：不同编译器、架构、sanitizer 或 CMake option 不能误共享依赖生成物。

**离线/已审查源码**：用 FetchContent 标准 override 给每个依赖注入源码，而不是共享另一个 build tree：

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GGML=/approved/ggml
```

其他名字同理：`GOOGLETEST`、`NGHTTP2`、`VOLK`、`VULKAN_HEADERS`、`GLSLANG`。`HYLE_DEPS_DIR`
可以移动**当前 build tree 自己的**依赖区，但不能指向源码树，也不应被不同 toolchain/build tree 共用。
仓库根 `.deps/` 只是旧开发树兼容的 ignored 路径，不是现行默认契约。

源码树已提供根文件 `hyle-offline-deps.cmake`。制作 offline source bundle 时，把依赖分别放在
`deps/<FetchContent-name>/`；文件会扫描这些目录并自动设置对应的
`FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>`，自身不再维护一份供应商名单。例如：

```text
hyle-offline-deps.cmake
deps/
  ggml/
  nghttp2/
  vulkan_headers/
  volk/
  googletest/       # 构建 tests 时需要
  glslang/          # 构建 compute tests/bench 时需要
```

```bash
cmake -C hyle-offline-deps.cmake \
  -S hyle-<version>-source -B build -DHYLE_COMPONENTS=all
```

目录名必须使用 `FetchContent_Declare()` 的名字，而不是上游仓库展示名，例如
`vulkan_headers` 而不是 `Vulkan-Headers`。bundle manifest 仍应记录每项上游 URL、HYLE pin
commit/tag、校验值和许可证。只复制 FetchContent 生成的 `<build>/_deps` 不够：里面混有特定
toolchain 的 build/stamp 状态，目录布局也不是发布 API。

逐项 source override 已保证该依赖不执行 download/update；不要把
`FETCHCONTENT_FULLY_DISCONNECTED=ON` 单独当成首次离线配置开关。若发行方需要机械保证首次配置绝不
访问网络，应让 bundle 提供 dependency provider；普通 HYLE 离线交付只要完整映射本次 component/
feature 会触达的全部 FetchContent 项即可。OpenSSL、CUDA Toolkit 这类 `find_package()` SDK 不在
FetchContent bundle 内，仍需由离线 toolchain/sysroot/package prefix 提供。

从零准备 Windows/Linux/macOS 工具链、系统 SDK、依赖介质并做真实断网验收，见
[`offline.md`](./offline.md)。

**ggml 的构建开关（HYLE 强制设定，见 `HyleComputeDependencies.cmake`）**：只构建私有 PIC 静态 CPU +
可选 CUDA
闭包；关闭 OpenMP、BLAS/Accelerate、Metal/Vulkan/RPC、tests/examples/server、backend dynamic load、
FA/NCCL/ccache，避免引入 HYLE 不消费的运行时。`GGML_CUDA` 跟随最终启用的 CUDA 能力。
普通源码构建 `HYLE_COMPUTE_GGML_NATIVE=ON` 偏向本机性能；`HYLE_INSTALL=ON` 默认 OFF，x86 使用
SSE4.2 baseline。已知部署 CPU 用 `HYLE_SIMD_EXACT_TARGET=AVX2/AVX512` 同步提高 HYLE 与 ggml ISA。

**关掉 ggml（无 compute 需求时）**：`-DHYLE_COMPUTE_GGML=OFF` → 不拉 ggml、不编 ggml_bridge，compute 其余（01–04）照编。

---

## OpenSSL（io-08 TLS，可选）

TLS 是三态能力：`HYLE_IO_TLS=AUTO` 自动发现，`ON` 要求 OpenSSL，`OFF` 不探测且保持零 OpenSSL
依赖。启用后编 `tls.cc` 并对消费者公开同名编译宏。OpenSSL 使用动态链（安全库应跟随系统 CVE
更新，不静态焊死）。

- **Linux/WSL**：`sudo apt install libssl-dev`（系统包，零配置，装完重配即启用）。
- **Windows**：无系统级 MSVC OpenSSL 开发库，用 **vcpkg**（详见 [`windows.md`](./windows.md) 依赖节）。
- 运行 TLS 测试要 OpenSSL 的 DLL 在 PATH（Windows）——见平台文件。

---

## CUDA Toolkit（compute GPU 后端，可选）

`HYLE_COMPUTE_CUDA=AUTO` 时 CMake 自动发现 Toolkit；`ON` 要求找到；`OFF` 不运行依赖探测、不启用
CUDA language，只编 host fallback。最终启用时才编 `cuda_backend.cu` 并链接 CUDA runtime/driver/NVRTC。
普通本机构建未指定时使用 `CMAKE_CUDA_ARCHITECTURES=native`；`HYLE_INSTALL=ON` 的 CUDA SDK 必须
显式给出目标，例如 `-DCMAKE_CUDA_ARCHITECTURES='80;86;89'` 或 `all-major`，否则配置拒绝。

- **本仓库统一 CUDA 13.3**（RTX 4070 / sm_89）。Windows toolkit 在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`；WSL 在 `/usr/local/cuda-13.3`（软链 `/usr/local/cuda`）。装法见平台文件。
- **编译/链接不需要 PATH**（CMake 走 `find_package`）；**运行 GPU 测试**要 CUDA runtime DLL 在 PATH（Windows）。
- **mac 无 NVIDIA GPU** → CUDA 后端 `#if` 门控编不进，走 host / 未来 Metal。

## cuDNN / TensorRT（更后，当前可不装）

**compute-01～05 不依赖 cuDNN/TensorRT**——它们是未来算子/推理图优化路径的加速档。当前完全可以不装。装齐的机器信息见 `reference_cuda_env`（cuDNN 9.24 / TensorRT 11.1，Win + WSL 均已装并端到端验证）。要用时按那里的 `-I/-L/-l` 指路。
