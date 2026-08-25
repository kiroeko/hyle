# 外部依赖：怎么装、哪些必需、哪些可选

> 跨平台通用。平台专属的安装命令在各平台文件里，这里讲**每个依赖是什么角色、怎么被 CMake 发现、装不装的后果**。

## 总览

| 依赖 | 类别 | 缺了会怎样 | 怎么被发现 |
|---|---|---|---|
| C++ 编译器 / CMake / generator / 汇编器 | **硬门槛** | CMake FATAL_ERROR，配置失败 | CMake 自带检查；Ninja 不是唯一 generator |
| **ggml** | **自动拉取**（FetchContent） | compute + sched/fcontext 且启用 bridge 时取得；严格离线 bundle 缺源码则配置失败 | lock pin `v0.21.0` |
| **googletest** | **自动拉取**（FetchContent） | 同上（仅影响测试，`-DHYLE_BUILD_TESTS=OFF` 可跳） | lock pin `v1.18.0` |
| **nghttp2** | **自动拉取**（FetchContent） | io + sched/fcontext 且启用 HTTP/2 时取得 | lock pin `v1.70.0` |
| **volk + Vulkan-Headers** | **自动拉取**（FetchContent，compute-06） | 同属 `vulkan-sdk-1.4.357.0`；缺网首配失败 | lock 内两个精确 commit |
| **glslang** | 测试/benchmark shader 工具 | 只在 compute tests/bench 打开时拉取；生产 SDK 不需要 | 同属 `vulkan-sdk-1.4.357.0` |
| **Tracy** | 启用 trace 时的外部源码树 | `HYLE_TRACE=ON` 才编译；任意/脏/被改写的源码会配置失败 | lock pin `v0.14.1`，`HYLE_TRACY_DIR` |
| **Vulkan loader + ICD/driver** | 可选能力（compute-06 运行时） | 无则 Vulkan 后端运行时禁用（`vulkan_vtable()` 返 null），不改变其他已配置能力 | volk 运行时 dlopen `vulkan-1.dll`/`libvulkan.so`/`libvulkan.dylib` |
| **OpenSSL** | 可选能力 | `HYLE_IO_TLS=AUTO/OFF` 可无 TLS；ON 时缺失即配置失败 | `find_package(OpenSSL QUIET)` |
| **CUDA Toolkit** | 可选能力 | `HYLE_COMPUTE_CUDA=AUTO/OFF` 可只用 host；ON 时缺失即配置失败 | `find_package(CUDAToolkit QUIET)` |
| **cuDNN / TensorRT** | 可选能力（更后） | 当前里程碑不依赖，完全可不装 | 手动 `-I/-L`（未来算子/推理路径） |

**"可选能力 = 可选依赖"铁律**：OpenSSL/CUDA 都是 `AUTO|ON|OFF`。AUTO 找到就启用、找不到降级；
ON 要求能力存在，否则配置失败；OFF 连依赖都不探测。只有本地探索适合 AUTO，正式 SDK/CI 应写死 ON/OFF，
避免构建机环境暗中改变产物。

---

## FetchContent：固定版本源码依赖

**唯一版本源**是 [`../../cmake/HyleDependencyLock.json`](../../../cmake/HyleDependencyLock.json)。每项同时
登记 class、展示版本、40 位 commit、Git root tree、官方 codeload URL、archive SHA-256、
SPDX/自定义许可证表达式和许可证文件哈希。FetchContent/external checkout 另登记无 `.git` 展开树
SHA-256，in-tree snapshot 登记 vendored content tree，修改后并入产品的 port 则登记 HYLE integration path
集合及其 adapted-content SHA-256。`HyleDependencyVersions.cmake` 只负责验证 JSON 并投影
`HYLE_DEP_<NAME>_*` 变量；不得在模块 CMake、脚本或文档再手写第二份 pin。

当前产品/测试 FetchContent 快照为：

- ggml `v0.21.0`；
- nghttp2 `v1.70.0`；
- Vulkan-Headers、volk、glslang 同一 `vulkan-sdk-1.4.357.0` train；
- googletest `v1.18.0`。

在线取得使用 lock 中的官方 codeload archive，并由 CMake `URL_HASH SHA256=...` 在展开前验证；commit
仍是源码身份，archive SHA-256 是传输身份，两者不能互相代替。in-tree 快照（mimalloc `v3.5.0`、
simdutf `v9.1.0`、xxHash dev HEAD `c0b5ea995d66`、wyhash snapshot、nanobench `v4.6.0`、picohttpparser snapshot、
Boost.Context `boost-1.92.0` fcontext）会在其实际 component/benchmark 闭包被选择时审计规范化内容树和
许可证；core-only 不因此要求未消费 vendor。`cmake -P cmake/HyleDependencyVersions.cmake` 与完整源码包
target 仍强制全量审计，防止发布介质夹带未复证快照。
普通 `-DHYLE_DEPENDENCY_LOCK_FILE=...` 不能替换这份中央账本；否则攻击者可同时重写 URL、hash 与预期
源码身份，使后续校验只证明一份自带的恶意账本。刻意损坏 lock 的合同夹具只能由已执行的受信 CMake
代码以 GLOBAL property 证明一个精确测试路径；能注入 toolchain/top-level include 的代码本来就在构建
信任边界内。

Highway/xsimd 有两个不能混为一谈的身份：`highway_port`/`xsimd_port` 是已经迁入并修改为 HYLE
D-tag/API 的产品源码 origin，分别绑定 `83eff393...`/`5d2490f5...`，同时锁住当前 adapted paths 的内容
hash 和独立许可证 payload；`highway`/`xsimd` 是只用于当前 benchmark/reference 的 2026-08-22
上游 peer。更新 peer 不会自动“升级”已大幅适配的产品 port；必须先做上游差异审计、逐项回移候选、
正确性对拍和性能 rebaseline。Highway `math-inl.h` 中注明来自 JPEG XL 的 `Cbrt` 段也属于
`highway_port` 这条精确 origin，故该 record 与安装 registry 同时要求 Highway Apache-2.0 和 JPEG XL
BSD-3-Clause 两份文本，不伪造另一个 libjxl revision，也不留 unlocked 例外。

**默认行为（开箱即用，需联网）**：新机器 clone HYLE → 跑 `cmake` → FetchContent 下载已校验 archive
到 `<hyle-build>/_deps/<dep>-src`，生成物进入同一 build tree 的 `_deps/<dep>-build`。删除 build tree
会一并删除这些内容，这是有意的：不同编译器、架构、sanitizer 或 CMake option 不能误共享依赖生成物。

**离线/已审查源码**：用 FetchContent 标准 override 给每个依赖注入源码，而不是共享另一个 build tree：

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GGML=/approved/ggml
```

这个标准变量不是绕过 lock 的后门：六个 consumer 都经统一 wrapper，在 `MakeAvailable` 前接受两种且仅两种
身份——精确 commit/tree/clean 的 Git checkout，或与 lock 的官方 archive 展开树哈希完全一致、无需 `.git`
的目录。文件修改、增加、删除、空目录变化、链接拼写变化或逃逸链接都会配置失败。FetchContent 的
“第一次声明胜出”也不能绕过：wrapper 会复验 `MakeAvailable` 实际使用的 source，父工程同名预声明若
不是 locked identity，整个 HYLE configure 失败。父 CMake/toolchain 是可执行信任根；这里保证最终产品
不能带着错误身份成功生成，并不把父脚本当成可沙箱执行的数据。其他名字同理：
`GOOGLETEST`、`NGHTTP2`、`VOLK`、`VULKAN_HEADERS`、`GLSLANG`。`HYLE_DEPS_DIR`
可以移动**当前 build tree 自己的**依赖区，但不能指向源码树，也不应被不同 toolchain/build tree 共用。
benchmark 对手的 sibling checkout 根使用环境变量 `HYLE_BENCH_DEPS_DIR`；它刻意不是
`HYLE_DEPS_DIR`，也不参与 FetchContent population。
仓库根 `.deps/` 只是旧开发树兼容的 ignored 路径，不是现行默认契约。

源码树已提供根文件 `hyle-offline-deps.cmake`。制作 offline source bundle 时，把依赖分别放在
`deps/<lock-name>/`。目录可以是 lock commit 的**干净 Git checkout**，也可以是 lock 中官方 archive
未经编辑的展开根目录；前者验证 HEAD/root tree、零 tracked/untracked/ignored 文件（ignored build/cache
同样可能被上游 glob/include 消费），后者验证包含目录、文件原始字节和安全相对
symlink 目标的完整 manifest，两者都重验许可证。通过后才设置
`FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>`。Tracy 同样可用两种形状，映射为 `HYLE_TRACY_DIR`。允许的名字
直接来自 lock，不另维护供应商名单。例如：

```text
hyle-offline-deps.cmake
deps/
  ggml/
  nghttp2/
  vulkan_headers/
  volk/
  googletest/       # 构建 tests 时需要
  glslang/          # 构建 compute tests/bench 时需要
  tracy/            # 仅 HYLE_TRACE=ON 时需要
```

```bash
cmake -C hyle-offline-deps.cmake \
  -S hyle-<version>-source -B build -DHYLE_COMPONENTS=all
```

目录名必须使用 lock 名，例如 `vulkan_headers` 而不是 `Vulkan-Headers`。脚本强制
`FETCHCONTENT_FULLY_DISCONNECTED=ON` 与 `FETCHCONTENT_UPDATES_DISCONNECTED=ON`；缺少已启用 feature 的
源码会直接失败，不会回退联网。lock 的 expanded-tree hash 使官方 codeload 解压目录无需携带 `.git`；
传输 archive 仍先核对其独立 SHA-256。只有组织使用不同但由外部系统独立证明的展开源码时，才可在 preload
显式使用 `-DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON`；该 mismatch 逃生形状核对许可证并拒绝
**任何** symlink（即便目标看似是安全相对路径），避免多级链接把词法根内目标重新导向根外。普通 direct
override 也不能自我声明为已证明。
完整准备命令和风险边界见 [`offline.md`](offline.md)。
不触达 FetchContent/Tracy 的 core/error-only 选择允许没有 `deps/`；这不是降级，所需源码集合为空且
断网门仍为 ON。若后来启用 tests/io/compute/trace，对应预期本地路径缺失会立即失败。

OpenSSL、CUDA Toolkit、Vulkan loader/driver 属于宿主 SDK/运行时：它们由 `find_package()`、toolchain、
sysroot 或 driver 提供，不能用某个源码 commit 冒充机器环境 pin。正式构建应固定 feature ON/OFF，并在
构建证明中记录实际发现版本；这类 SDK 不放进 FetchContent source bundle。

从零准备 Windows/Linux/macOS 工具链、系统 SDK、依赖介质并做真实断网验收，见
[`offline.md`](offline.md)。

**ggml 的构建开关（HYLE 强制设定，见 `HyleComputeDependencies.cmake`）**：只构建私有 PIC 静态 CPU +
可选 CUDA
闭包；关闭 OpenMP、BLAS/Accelerate、Metal/Vulkan/RPC、tests/examples、backend dynamic load、
FA/NCCL/ccache，避免引入 HYLE 不消费的运行时。`GGML_CUDA` 跟随最终启用的 CUDA 能力。
普通源码构建 `HYLE_COMPUTE_GGML_NATIVE=ON` 偏向本机性能；`HYLE_INSTALL=ON` 默认 OFF，x86 使用
SSE4.2 baseline。已知部署 CPU 用 `HYLE_SIMD_EXACT_TARGET=AVX2/AVX512` 同步提高 HYLE 与 ggml ISA。

**关掉 ggml（无 compute 需求时）**：`-DHYLE_COMPUTE_GGML=OFF` → 不拉 ggml、不编 ggml_bridge，compute 其余（01–04）照编。
ggml bridge 与 HTTP/2/TLS transport 都依赖 sched/fcontext。无 fcontext 目标上 HTTP/2/ggml 默认 OFF、
显式 ON 失败；TLS AUTO 不探测 OpenSSL、TLS ON 失败。稳定 stub 继续返回 unavailable，但不会向安装包
登记虚假的 feature。compute host/CUDA/Vulkan 仍可独立构建。

---

## OpenSSL（io-08 TLS，可选）

TLS 是三态能力：有 sched/fcontext 时 `HYLE_IO_TLS=AUTO` 自动发现，`ON` 要求 OpenSSL，`OFF` 不探测且保持零 OpenSSL
依赖。启用后编 `tls.cc` 并对消费者公开同名编译宏。OpenSSL 使用动态链（安全库应跟随系统 CVE
更新，不静态焊死）。无 fcontext 时 AUTO 直接禁用且不探测，显式 ON 是不可能合同并失败。

- **Linux/WSL**：`sudo apt install libssl-dev`（系统包，零配置，装完重配即启用）。
- **Windows**：无系统级 MSVC OpenSSL 开发库，用 **vcpkg**（详见 [`windows.md`](windows.md) 依赖节）。
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
