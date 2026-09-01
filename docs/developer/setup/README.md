# HYLE 环境准备（面向贡献者）

> 这个文件夹只回答一个问题：**从零把一台机器准备成能编译 / 测试 / 性能测试 HYLE 的开发环境。**
> 面向**开发 HYLE 本身**的贡献者（不是把 HYLE 当依赖用的库使用者——那看 [`../guide/usage-guide.md`](../../user/usage-guide.md)）。
> 内容以**实测**为准，慢变；命令 / 版本变了要回填。

## 按平台分文件

三个平台各有独立文件，把「构建 / 测试 / 性能测试」三件事一次讲清（含各自的坑）：

| 平台 | 文件 | 角色 |
|---|---|---|
| Windows（MSVC / clang-cl） | [`windows.md`](windows.md) | MSVC 主构建 + clang-cl 可移植性门；GPU（CUDA）真机 |
| Linux / WSL（GCC14） | [`linux-wsl.md`](linux-wsl.md) | **lock-free TSan 唯一闸门**；Linux 执行环境与 GPU 路径；不是独立物理性能机；9p 铁律 |
| macOS（Apple Silicon） | [`macos.md`](macos.md) | ARM/NEON + kqueue 真机；CUDA-free，仍可构建 Vulkan backend |

依赖（跨平台，可选能力）单独一文件：

| 主题 | 文件 |
|---|---|
| 外部依赖（OpenSSL / CUDA / ggml / nghttp2 / Vulkan / googletest）怎么装、哪些必需哪些可选 | [`dependencies.md`](dependencies.md) |
| 无网/隔离网机器怎样准备工具链、SDK 与完整依赖源码包 | [`offline.md`](offline.md) |

## 最小心智模型（读任何平台文件前先懂这个）

HYLE 的依赖分三类，决定了「新机器 `cmake` 会不会失败」：

1. **硬门槛（必需，缺了 CMake 直接 FATAL_ERROR）**：C++ 编译器（GCC14+ / Clang18+ / MSVC19.43+）、CMake 3.25+、一个 CMake generator，以及所选组件需要的汇编器（MASM/GAS）。根 `project()` 只启用 CXX；`HYLE_BUILD_TESTS=OFF` 时只有所选闭包含 mem/mimalloc 才按需要求 C 编译器。开启测试会进入 GoogleTest 自己的 CMake project，并可能同时探测 C frontend。Ninja 只是可选 generator；Visual Studio/Unix Makefiles/Xcode 也可用。这些**不自动装**，人得先装好——见各平台文件第一节。
2. **自动拉取（联网自动，无需人工）**：Highway、xsimd、`googletest`、`nghttp2`、`ggml`、
   Vulkan-Headers、volk、glslang 和按需 Tracy 由所选组件/feature 触发——新机器首次 `cmake` 会自动
   取得到跨 build 的 `HYLE_DEP_SOURCE_CACHE`；当前 build 的 `_deps` 只保存 binary/subbuild
   （[`dependencies.md` §FetchContent](dependencies.md)）。离线时搬运同一工具生成的完整 cache，或用
   根目录 `hyle-offline-deps.cmake` 注入逐项已审查源码。
3. **可选能力（显式契约）**：TLS、CUDA 取值为 `AUTO|ON|OFF`；HTTP/2、ggml、Vulkan 为 `ON|OFF`。这些昂贵能力当前都默认 `OFF`，并只在 io/compute owner 被所选闭包包含时声明；AUTO 可降级，ON 缺依赖即失败，OFF 不探测/取得依赖。`HYLE_COMPUTE_GGML_NATIVE` 只在 ggml 开启时出现；`HYLE_TRACE` 只在所选闭包有 trace consumer 时出现，`HYLE_TRACY_DIR` 是 trace 开启后才出现的可选受审源码覆盖，默认走共享 cache；单一 Tracy 实现不再有 `HYLE_TRACE_BACKEND`。正式 CI/交付必须写死能力集合。cuDNN/TensorRT 更靠后（当前里程碑不依赖）。

> **一句话**：硬门槛靠人装，FetchContent 依赖联网下载或显式注入源码，GPU/TLS 则由 AUTO/ON/OFF 明确选择。

## 「能不能自动化装好一切？」——诚实回答

**部分能，部分不能，现状是「不自动、靠文档」：**

- **第 2 类（源码依赖）已经自动**：首次在线 configure 按需填充共享、锁定的 source cache；后续
  build tree 复用且不重新下载。隔离网交付由发行方搬运该受管 cache，或准备可审计的
  `deps/<FetchContent-name>/` 源码闭包。
- **第 1 类（编译器/CMake/generator）不自动安装**：装系统级工具链需要 admin/包管理器权限，跨发行版差异大（apt/brew/VS installer/vcpkg 各不同），项目不能安全接管。Windows 的 `scripts/hyle-windows.ps1` 只用 `vswhere` 选择已安装 SDK/MSVC ABI 环境；Linux/WSL/macOS 的 `scripts/hyle-unix.sh` 只从已安装 compiler 中选择并预检 floor，再原样执行给它的命令。两者都不安装或持久修改系统工具链。
- **第 3 类中的系统 SDK 不自动**：CUDA Toolkit/driver、OpenSSL 等重量级且机器相关，只在你**要**那个能力时才装；HTTP/2、ggml、Vulkan 的源码部分仍按第 2 类取得。文档给出确切边界（[`dependencies.md`](dependencies.md)）。

## 验证一切就绪（各平台通用终点）

配置 + 构建 + 跑几个测试，绿了就能开工：

```bash
# Linux / WSL / macOS：统一选择已安装的受支持 compiler。
scripts/hyle-unix.sh cmake --preset <linux|macos>
scripts/hyle-unix.sh cmake --build --preset <linux|macos> --parallel 2
scripts/hyle-unix.sh ctest --preset <linux|macos>

# Windows 使用 hyle-windows.ps1 或已初始化的 VS shell。
cmake --preset <windows-vs2026|windows>
```

这些 preset 的开发 feature contract 相同：顶层默认选择 `HYLE_COMPONENTS=all`，昂贵的
TLS/HTTP2/CUDA/ggml/Vulkan/trace 能力保持默认 OFF，所以是 CPU baseline；它们不强迫 Unix 使用
Ninja。父工程若通过 `add_subdirectory()` 嵌入 HYLE，则必须在调用前显式设置
`HYLE_COMPONENTS`，不会继承顶层的隐式 `all`。

普通测试默认 `HYLE_TEST_INSTRUMENTATION=OFF`，因此验证 shipping layout/hot path；只有要覆盖
fault injection 或结构诊断 seam 时才在独占 build tree 显式设为 `ON`。两种配置不能复用同一
cache 轮流改值来冒充两份证据。
完整一步入口是 `cmake --workflow --preset <windows-vs2026|windows|linux|macos>-check`。VS 2026
原生入口要求 CMake 4.2+；`windows` 是 VS 2022 回退。Windows 若要让 Ninja 使用已选择的最新
MSVC，改用 `scripts/hyle-windows.ps1` + `windows-ninja`；安装 clang-cl 18+ 后可用同一 wrapper +
`windows-clangcl` 在独立 build tree 做可移植性复核。

- 看配置日志确认已选择 owner 的可选能力状态；未选择 io/compute 时，相应 cache 项不会出现。
- lock-free 改动**必须**过 WSL TSan（[`linux-wsl.md`](linux-wsl.md)）。
- 性能数字**必须**在原生文件系统（非 9p / 非网络盘）跑，见各平台「性能测试」节。
- correctness、engineering 与 formal 的机器角色以
  [性能机器合同](../performance/machines.md)为准；“三个执行环境”不等于“三台独立性能机器”。

## 并行开发、factor 构建与产品交付门

每个 agent/session 使用独占、绝对 scratch build 路径，不得复用 preset 的固定 build tree 或另一
会话的 CMake cache。Unix 用 `scripts/hyle-unix.sh cmake ... -B <session-scratch>`；Windows wrapper 使用
`-BinaryDir <absolute-path>`。普通源码/SDK journey 用 `-Component <product>`；只验证一个内部施工
单元时用 `-Factor <factor>`，等价于顶层 fresh tree 的 `HYLE_DEV_FACTORS`。factor selector 不能用于
父工程嵌入或 `HYLE_INSTALL=ON`，也不会制造用户 `hyle::<factor>` target。

```powershell
scripts/hyle-windows.ps1 -Action Configure `
  -Preset windows-ninja `
  -BinaryDir C:\hyle-scratch\container-intrusive-factor `
  -Factor container_intrusive
```

产品交付前运行 `cmake/HyleCiSelectedComponentSmoke.cmake`。它对一个产品根分别验证 source
embedding、shared SDK、static SDK，以及搬迁后的 installed C++ consumer；这比“全库刚好能编”更能
发现闭包、owned headers 和 package 边错误。Windows 入口例如：

```powershell
scripts/hyle-windows.ps1 -Action SelectedComponentSmoke `
  -Component mem `
  -SmokeRoot C:\hyle-scratch\mem-<session-id>
```

Unix 入口例如：

```bash
scripts/hyle-unix.sh cmake -DHYLE_SOURCE_DIR="$PWD" \
  -DHYLE_SMOKE_ROOT=/tmp/hyle-mem-agent-a \
  -DHYLE_SMOKE_COMPONENT=mem \
  -P cmake/HyleCiSelectedComponentSmoke.cmake
```

GitHub Actions 的 PR/main push 只跑高信号 Linux CPU/合同门以控制额度；Windows、macOS、交叉编译、
完整 package/source 和 locked-feature 矩阵只由手工 full dispatch 展开。
