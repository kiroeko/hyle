# 外部依赖：怎么装、哪些必需、哪些可选

> 跨平台通用。平台专属的安装命令在各平台文件里，这里讲**每个依赖是什么角色、怎么被 CMake 发现、装不装的后果**。

## 总览

| 依赖 | 类别 | 缺了会怎样 | 怎么被发现 |
|---|---|---|---|
| C++ 编译器 / CMake / generator / 汇编器 | **硬门槛** | CMake FATAL_ERROR，配置失败 | CMake 自带检查；Ninja 不是唯一 generator |
| **Highway** | **自动拉取**（FetchContent） | 选择 `simd` 或 `container` 时取得；严格离线 bundle 缺源码则配置失败 | lock pin `18e449b6…`；只执行 HYLE 自己的 target/package 集成 |
| **xsimd** | **自动拉取**（FetchContent） | 选择 `simd` 时与 Highway 一起取得；严格离线 cache/bundle 缺源码则配置失败 | lock pin `3b3151ff…`；用于锁定的 SIMD provider 路由 |
| **ggml** | **QUEUED Compute 的锁定候选源码** | 只有旧 compute gate 或未来经审计的 provider 选择它时取得；不是当前受支持产品要求 | lock pin `v0.21.0`，新 provider 开工前必须重验版本/API |
| **googletest** | **自动拉取**（FetchContent） | 同上（仅影响测试，`-DHYLE_BUILD_TESTS=OFF` 可跳） | lock pin `v1.18.0` |
| **nghttp2** | **自动拉取**（FetchContent） | io + sched/fcontext 且启用 HTTP/2 时取得 | lock pin `v1.70.0` |
| **volk + Vulkan-Headers** | **QUEUED Compute 的锁定候选源码** | 旧 Vulkan gate 开启时取得；正式 Vulkan backend 重构后仍须按新 factor/版本重新裁决 | 同属 `vulkan-sdk-1.4.357.0`，lock 内两个精确 commit |
| **glslang** | 旧 Compute 测试/benchmark shader 工具 | 旧 Vulkan tests/bench 打开时才拉取；不证明未来生产 SDK 需要它 | 同属 `vulkan-sdk-1.4.357.0` |
| **Tracy** | 启用 trace 时自动取得的外部源码树 | `HYLE_TRACE=ON` 才取得并编译；任意/脏/被改写的覆盖源码会配置失败 | lock pin `v0.14.1`；`HYLE_TRACY_DIR` 仍可显式覆盖 |
| **Vulkan loader + ICD/driver** | 未来正式 Vulkan backend 的宿主运行时 | 旧实现无 loader 时不可用；新合同要求明确 capability/失败，不继承旧 null-vtable 语义 | 由宿主 SDK/driver 提供，不进入源码 lock |
| **OpenSSL 3.0+** | 可选能力 | `HYLE_IO_TLS=AUTO/OFF` 可无 TLS；ON 时缺失或低于 3.0 即配置失败 | `find_package(OpenSSL 3.0 QUIET)` |
| **CUDA Toolkit** | 未来主要 CUDA backend 的宿主 SDK | 旧 gate 可探测它；新 CUDA factor 尚未签发，普通已支持组件不需要 | `find_package(CUDAToolkit)` 候选，精确合同待审 |
| **Metal SDK + Apple providers** | 未来 Apple 主要路线 | Metal SDK 随 Apple toolchain；MLX 是首个待资格化 provider，当前尚未签发；MPSGraph/MPS 后续逐能力审计，Core ML 暂属模型级 routed runtime | 宿主 SDK + 可能的锁定 MLX provider source |
| **ROCm / HIP SDK** | 必须补齐的 AMD 路线 | 当前无合格实现；不能用 Vulkan 或 CUDA 宏替换冒充 | 宿主 SDK/toolchain，精确发现合同待审 |

**“可选能力 = 可选依赖”铁律**：已签发 feature 的 ON 要求依赖存在，OFF 不探测；正式 SDK/CI 写死
能力集合，避免环境暗中改变产物。QUEUED Compute 的现有选项只是待迁移 baseline，不为新架构冻结名称、
默认值或降级语义。

---

## FetchContent：固定版本源码依赖

**唯一版本源**是 [`../../cmake/HyleDependencyLock.json`](../../../cmake/HyleDependencyLock.json)。每项同时
登记 class、展示版本、40 位 commit、Git root tree、官方 codeload URL、archive SHA-256、
SPDX/自定义许可证表达式和许可证文件哈希。FetchContent/external checkout 另登记无 `.git` 展开树
SHA-256，in-tree snapshot 登记 vendored content tree，修改后并入产品的 port 则登记 HYLE integration path
集合及其 adapted-content SHA-256。`HyleDependencyVersions.cmake` 只负责验证 JSON 并投影
`HYLE_DEP_<NAME>_*` 变量；不得在模块 CMake、脚本或文档再手写第二份 pin。

当前产品/测试 FetchContent 快照为：

- Highway `18e449b6…`（`simd` 与选定 `container` kernel 的直接产品源码）；
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
已经改写并并入 HYLE 源码的部分同样进入中央 lock，而不是因为“不下载”就消失：
`platform_thread` 的 Parker 记录 Rust std 1.94.1 commit `e408947b…`、原始 tree/archive、adapted HYLE
content hash 和选择的 MIT 文本；安装包同时携带 `rust-parker-LICENSE-MIT` 与第三方 notices。
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
源码集合为空且断网门仍为 ON。后来启用 tests、trace 或 HTTP/2 时，对应预期本地路径缺失会立即失败；
ggml/Vulkan 只有具名 QUEUED Compute baseline 才应开启。

OpenSSL、CUDA Toolkit、Vulkan loader/driver 属于宿主 SDK/运行时：它们由 `find_package()`、toolchain、
sysroot 或 driver 提供，不能用某个源码 commit 冒充机器环境 pin。正式构建应固定 feature ON/OFF，并在
构建证明中记录实际发现版本；这类 SDK 不放进 FetchContent source bundle。

从零准备 Windows/Linux/macOS 工具链、系统 SDK、依赖介质并做真实断网验收，见
[`offline.md`](offline.md)。

### QUEUED Compute 的旧依赖门

`HyleComputeDependencies.cmake`、`HYLE_COMPUTE_GGML[_NATIVE]`、`HYLE_COMPUTE_CUDA` 与
`HYLE_COMPUTE_VULKAN` 当前描述旧实现的可复现 baseline。它们可以在资产审计中显式打开，但不能用于
宣传当前 Compute 支持，也不冻结未来 factor 或 option 形状。旧 graph/compute target 对 Sched/fcontext
的公开依赖同样待重构；新设计要求 direct Compute core 不因 coroutine convenience 强制取得 Sched。

新架构将 Apple 的 MLX-first provider + 原生 Metal backend/custom-kernel escape hatch 与 NVIDIA 的
CUDA backend + providers 作为主要路线；Vulkan 是正式消费级保底路线，Apple 经 MoltenVK 只作
portability；另有明确的
HIP/ROCm 缺口。每个 backend/provider factor 只有在 closure card 完成 capability、ownership、threading、
source/static/shared 与真机资格后，才会在本页升级为正常安装说明。

---

## OpenSSL（io-08 TLS，可选）

TLS 是三态能力，当前默认 `OFF`：`AUTO` 自动发现 OpenSSL 3.0+，`ON` 要求 OpenSSL 3.0+，`OFF` 不探测且保持零 OpenSSL
依赖。启用后编 `tls.cc` 并对消费者公开同名编译宏。OpenSSL 使用动态链（安全库应跟随系统 CVE
更新，不静态焊死）。这项 option 只在可构建的 io 闭包中存在；无 fcontext 目标会拒绝 io 根，
不会退化成 TLS stub。

- **Linux/WSL**：`sudo apt install libssl-dev`，再显式配置 `-DHYLE_IO_TLS=ON`；本地探索也可用
  `AUTO`，但正式 SDK/CI 应固定为 `ON` 或 `OFF`。
- **Windows**：无系统级 MSVC OpenSSL 开发库，用 **vcpkg**（详见 [`windows.md`](windows.md) 依赖节）。
- 运行 TLS 测试要 OpenSSL 的 DLL 在 PATH（Windows）——见平台文件。

---

## CUDA Toolkit（QUEUED Compute baseline）

旧 `HYLE_COMPUTE_CUDA=AUTO|ON|OFF` 仍可用于复现待审源码；它不是新 CUDA backend 的已接受配置
合同。运行旧资产需要匹配的 CUDA runtime/driver/NVRTC，安装型 artifact 仍必须显式记录 architecture。

- **本仓库统一 CUDA 13.3**（RTX 4070 / sm_89）。Windows toolkit 在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`；WSL 在 `/usr/local/cuda-13.3`（软链 `/usr/local/cuda`）。装法见平台文件。
- **编译/链接不需要 PATH**（CMake 走 `find_package`）；**运行 GPU 测试**要 CUDA runtime DLL 在 PATH（Windows）。
- **mac 无 NVIDIA GPU**；未来主要路线是 MLX-first provider + 原生 Metal backend/custom-kernel
  escape hatch。

## cuDNN / TensorRT（当前不属于产品依赖）

当前已支持 HYLE 组件不需要 cuDNN/TensorRT。未来只有具名 CUDA provider task 裁决为 `BORROW` 后，
才增加版本、发现、license、artifact 和 runtime 说明；本机装有 SDK 不是准入证据。
