# 外部依赖：怎么装、哪些必需、哪些可选

> 跨平台通用。平台专属的安装命令在各平台文件里，这里讲**每个依赖是什么角色、怎么被 CMake 发现、装不装的后果**。

## 总览

| 依赖 | 类别 | 缺了会怎样 | 怎么被发现 |
|---|---|---|---|
| C++ 编译器 / CMake / generator / 汇编器 | **硬门槛** | CMake FATAL_ERROR，配置失败 | CMake 自带检查；Ninja 不是唯一 generator |
| **Highway** | **自动拉取**（FetchContent） | 选择 `simd` 或 `container` 时取得；严格离线 bundle 缺源码则配置失败 | lock pin `f503c485…`；只执行 HYLE 自己的 target/package 集成 |
| **xsimd** | **自动拉取**（FetchContent） | 选择 `simd` 时与 Highway 一起取得；严格离线 cache/bundle 缺源码则配置失败 | lock pin `3b3151ff…`；用于锁定的 SIMD provider 路由 |
| **ggml** | **自动拉取**（FetchContent） | compute + sched/fcontext 且启用 bridge 时取得；严格离线 bundle 缺源码则配置失败 | lock pin `v0.21.0` |
| **googletest** | **自动拉取**（FetchContent） | 同上（仅影响测试，`-DHYLE_BUILD_TESTS=OFF` 可跳） | lock pin `v1.18.0` |
| **nghttp2** | **自动拉取**（FetchContent） | io + sched/fcontext 且启用 HTTP/2 时取得 | lock pin `v1.70.0` |
| **volk + Vulkan-Headers** | **按需拉取**（FetchContent，compute-06） | 仅 `HYLE_COMPUTE_VULKAN=ON` 时取得；此时缺网且无离线源码会配置失败 | 同属 `vulkan-sdk-1.4.357.0`，lock 内两个精确 commit |
| **glslang** | 测试/benchmark shader 工具 | Vulkan 已开启且 compute tests/bench 打开时才拉取；生产 SDK 不需要 | 同属 `vulkan-sdk-1.4.357.0` |
| **Tracy** | 启用 trace 时自动取得的外部源码树 | `HYLE_TRACE=ON` 才取得并编译；任意/脏/被改写的覆盖源码会配置失败 | lock pin `v0.14.1`；`HYLE_TRACY_DIR` 仍可显式覆盖 |
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

- Highway `f503c485…`（`simd` 与选定 `container` kernel 的直接产品源码）；
- xsimd `3b3151ff…`（`simd` 的锁定 provider 源码）；
- ggml `v0.21.0`；
- nghttp2 `v1.70.0`；
- Vulkan-Headers、volk、glslang 同一 `vulkan-sdk-1.4.357.0` train；
- googletest `v1.18.0`。

在线取得使用 lock 中的官方 codeload archive，并由 CMake `URL_HASH SHA256=...` 在展开前验证；commit
仍是源码身份，archive SHA-256 是传输身份，两者不能互相代替。in-tree 快照（mimalloc `v3.5.1`、
simdutf `v9.1.0`、xxHash dev HEAD `c0b5ea995d66`、wyhash snapshot、nanobench `v4.6.0`、picohttpparser snapshot、
Boost.Context `boost-1.92.0` fcontext）会在其实际 component/benchmark 闭包被选择时审计规范化内容树和
许可证；core-only 不因此要求未消费 vendor。`cmake -P cmake/HyleDependencyVersions.cmake` 与完整源码包
target 仍强制全量审计，防止发布介质夹带未复证快照。
普通 `-DHYLE_DEPENDENCY_LOCK_FILE=...` 不能替换这份中央账本；否则攻击者可同时重写 URL、hash 与预期
源码身份，使后续校验只证明一份自带的恶意账本。刻意损坏 lock 的合同夹具只能由已执行的受信 CMake
代码以 GLOBAL property 证明一个精确测试路径；能注入 toolchain/top-level include 的代码本来就在构建
信任边界内。

Highway/xsimd 的身份不能混为一谈：`highway` 是选择 `simd` 或 `container` 时取得的直接产品源码，
`xsimd` 是选择 `simd` 时取得的另一条锁定 provider 源码。HYLE 只编译自己的 target-bound TU，不执行
上游 project，也不把任一供应商类型泄漏进 ordinary batch API；benchmark 使用独立、干净的 sibling
checkout 作同 identity peer。
`highway_port` 只登记 SIMD-11 有界裁决后保留的最小 VQSort adaptation，绑定旧 origin、adapted-path
内容 hash 和 Apache-2.0 payload；它保留 compiler/scalable target glue、HYLE public POD/static entry 与
产品语义，不能被误记成原创实现。数学与 bitpack port 已被 direct Highway 替代并从账本删除。新增 checkout 本身不
等于完成迁移，必须同时删除等价产品 route、旧 source audit 与过期许可证投影。

**默认行为（开箱即用，首次需联网）**：新机器 clone HYLE → 跑 `cmake`。所选闭包缺少锁定源码时，
configure 自动下载官方 archive，在共享 `HYLE_DEP_SOURCE_CACHE` 内校验、展开并原子落盘；依赖的
binary/subbuild 只进入当前 build 的 `HYLE_DEP_BUILD_DIR`。删除 build tree 不删除源码，后续新 build
复用同一份源码且不联网；不同编译器、架构、sanitizer 或 CMake option 仍各自生成 binary/subbuild。

默认共享源码位置为：

- Windows：`%LOCALAPPDATA%/hyle/dependencies/v1`；
- macOS：`$HOME/Library/Caches/hyle/dependencies/v1`；
- 其他 Unix：`$XDG_CACHE_HOME/hyle/dependencies/v1`，未设置时为
  `$HOME/.cache/hyle/dependencies/v1`。

`-DHYLE_DEP_SOURCE_CACHE=<absolute-path>` 优先于同名环境变量，可把它统一放到例如
`C:\public\project-hyle\.deps`。`HYLE_DEP_BUILD_DIR` 默认是当前 Hyle binary tree 的 `_deps`，只能由
该 build 使用。源码 cache 不得放进源码树或 build tree，binary/subbuild 也不得反过来放进共享 cache。

**离线/已审查源码**：用 FetchContent 标准 override 给每个依赖注入源码，而不是共享另一个 build tree：

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GGML=/approved/ggml
```

这个标准变量不是绕过 lock 的后门：全部 locked consumer 都经统一 wrapper，在 `MakeAvailable` 前接受两种且仅两种
身份——精确 commit/tree/clean 的 Git checkout，或与 lock 的官方 archive 展开树哈希完全一致、无需 `.git`
的目录。文件修改、增加、删除、空目录变化、链接拼写变化或逃逸链接都会配置失败。FetchContent 的
“第一次声明胜出”也不能绕过：wrapper 会复验 `MakeAvailable` 实际使用的 source，父工程同名预声明若
不是 locked identity，整个 HYLE configure 失败。父 CMake/toolchain 是可执行信任根；这里保证最终产品
不能带着错误身份成功生成，并不把父脚本当成可沙箱执行的数据。其他名字同理：
`HIGHWAY`、`XSIMD`、`GOOGLETEST`、`NGHTTP2`、`VOLK`、`VULKAN_HEADERS`、`GLSLANG`。
benchmark 对手的 sibling checkout 根使用环境变量 `HYLE_BENCH_DEPS_DIR`；它刻意不是
`HYLE_DEP_SOURCE_CACHE` 或 `HYLE_DEP_BUILD_DIR`，也不参与产品 dependency population。仓库内旧
`.deps/` 已删除；不要恢复一个同时装 source/build/subbuild 的模糊目录。

共享 cache 的管理入口不会修改中央 lock：

```bash
cmake -DHYLE_DEP_ACTION=status -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=sync -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=sync -DHYLE_DEP_NAMES='highway;xsimd' \
  -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=sync -DHYLE_DEP_NAMES=xsimd -DHYLE_DEP_REPAIR=YES \
  -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=check-updates -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=prune -DHYLE_DEP_PRUNE_CONFIRM=YES \
  -P cmake/HyleDependencyCacheTool.cmake
```

`sync` 与首次 configure 使用同一下载/校验实现；`status` 复验当前 identity。损坏时普通 configure
fail closed，只有具名依赖加 `HYLE_DEP_REPAIR=YES` 才删除其损坏展开树，并优先从仍通过 SHA-256 的
cache archive 重建。`check-updates` 只给维护者 upstream 差异输入；`prune` 只删除带 HYLE cache marker
且不由当前 lock 引用的旧 identity，并要求明确确认。版本升级仍必须原子修改
lock/provenance/license 并重跑受影响模块门。

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
  highway/         # 选择 simd 或 container 时需要
  xsimd/           # 选择 simd 时需要
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
`HYLE_BUILD_TESTS=OFF` 且不触达 Tracy 的 core/error-only 选择允许没有 `deps/`；这不是降级，所需
源码集合为空且断网门仍为 ON。后来启用 tests、trace，或显式开启 HTTP/2、ggml、Vulkan 等
FetchContent feature 时，对应预期本地路径缺失会立即失败；只选择默认 feature-OFF 的 io/compute
根不会取得这些可选依赖。

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
io 与 compute 的当前公开头都暴露 sched 类型，因此中央组件图无条件把 sched/fcontext 纳入这两个
产品根。目标平台没有已接线的 fcontext 时，选择 io、compute 或 graph 会在 feature 探测前直接配置
失败；`OFF` 只关闭相应可选 backend，不能生成删掉 sched 公开依赖的 host-only/stub SDK。未来若要
交付 raw proactor 或 host-only compute，必须先拆出拥有独立公开头与 consumer journey 的真实组件。

---

## OpenSSL（io-08 TLS，可选）

TLS 是三态能力，当前默认 `OFF`：`AUTO` 自动发现，`ON` 要求 OpenSSL，`OFF` 不探测且保持零 OpenSSL
依赖。启用后编 `tls.cc` 并对消费者公开同名编译宏。OpenSSL 使用动态链（安全库应跟随系统 CVE
更新，不静态焊死）。这项 option 只在可构建的 io 闭包中存在；无 fcontext 目标会拒绝 io 根，
不会退化成 TLS stub。

- **Linux/WSL**：`sudo apt install libssl-dev`，再显式配置 `-DHYLE_IO_TLS=ON`；本地探索也可用
  `AUTO`，但正式 SDK/CI 应固定为 `ON` 或 `OFF`。
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
