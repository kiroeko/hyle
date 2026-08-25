# HYLE 环境准备（面向贡献者）

> 这个文件夹只回答一个问题：**从零把一台机器准备成能编译 / 测试 / 性能测试 HYLE 的开发环境。**
> 面向**开发 HYLE 本身**的贡献者（不是把 HYLE 当依赖用的库使用者——那看 [`../guide/usage-guide.md`](../guide/usage-guide.md)）。
> 内容以**实测**为准，慢变；命令 / 版本变了要回填。

## 按平台分文件

三个平台各有独立文件，把「构建 / 测试 / 性能测试」三件事一次讲清（含各自的坑）：

| 平台 | 文件 | 角色 |
|---|---|---|
| Windows（MSVC / clang-cl） | [`windows.md`](./windows.md) | MSVC 主构建 + clang-cl 可移植性门；GPU（CUDA）真机 |
| Linux / WSL（GCC14） | [`linux-wsl.md`](./linux-wsl.md) | **lock-free TSan 唯一闸门**；GPU（CUDA）真机；9p 铁律 |
| macOS（Apple Silicon） | [`macos.md`](./macos.md) | ARM/NEON + kqueue 真机；CUDA-free，仍可构建 Vulkan backend |

依赖（跨平台，可选能力）单独一文件：

| 主题 | 文件 |
|---|---|
| 外部依赖（OpenSSL / CUDA / ggml / nghttp2 / Vulkan / googletest）怎么装、哪些必需哪些可选 | [`dependencies.md`](./dependencies.md) |
| 无网/隔离网机器怎样准备工具链、SDK 与完整依赖源码包 | [`offline.md`](./offline.md) |

## 最小心智模型（读任何平台文件前先懂这个）

HYLE 的依赖分三类，决定了「新机器 `cmake` 会不会失败」：

1. **硬门槛（必需，缺了 CMake 直接 FATAL_ERROR）**：C++ 编译器（GCC14+ / Clang18+ / MSVC19.43+）、CMake 3.25+、一个 CMake generator，以及所选组件需要的汇编器（MASM/GAS）。Ninja 只是可选 generator；Visual Studio/Unix Makefiles/Xcode 也可用。这些**不自动装**，人得先装好——见各平台文件第一节。
2. **自动拉取（联网自动，无需人工）**：`googletest`、`nghttp2`、`ggml`、Vulkan-Headers、volk、glslang 按所选组件/feature 走 **FetchContent**——新机器首次 `cmake` 会自动取得并放到当前 build tree 的 `_deps/`（[`dependencies.md` §FetchContent](./dependencies.md)）。离线时用根目录的 `hyle-offline-deps.cmake` 注入已审查源码目录。
3. **可选能力（显式契约）**：TLS、CUDA 取值为 `AUTO|ON|OFF`；HTTP/2、ggml、Vulkan 为 `ON|OFF`。AUTO 可降级，ON 缺依赖即失败，OFF 不探测/取得依赖；正式 CI/交付必须写死能力集合。cuDNN/TensorRT 更靠后（当前里程碑不依赖）。

> **一句话**：硬门槛靠人装，FetchContent 依赖联网下载或显式注入源码，GPU/TLS 则由 AUTO/ON/OFF 明确选择。

## 「能不能自动化装好一切？」——诚实回答

**部分能，部分不能，现状是「不自动、靠文档」：**

- **第 2 类（源码依赖）已经自动**：FetchContent 联网时按需备齐；隔离网交付则由发行方准备可审计的 `deps/<FetchContent-name>/` 源码闭包。
- **第 1 类（编译器/CMake/generator）不自动安装**：装系统级工具链需要 admin/包管理器权限，跨发行版差异大（apt/brew/VS installer/vcpkg 各不同），项目不能安全接管。Windows 的 `scripts/hyle-windows.ps1` 只用 `vswhere` 选择已安装 SDK/MSVC ABI 环境，并按 preset 调用已安装的 cl.exe 或 PATH 中的 clang-cl；它不安装或修改 Visual Studio/LLVM。
- **第 3 类中的系统 SDK 不自动**：CUDA Toolkit/driver、OpenSSL 等重量级且机器相关，只在你**要**那个能力时才装；HTTP/2、ggml、Vulkan 的源码部分仍按第 2 类取得。文档给出确切边界（[`dependencies.md`](./dependencies.md)）。

## 验证一切就绪（各平台通用终点）

配置 + 构建 + 跑几个测试，绿了就能开工：

```bash
cmake --preset <windows-vs2026|windows|linux|macos>
cmake --build --preset <windows-vs2026|windows|linux|macos> --parallel 2
ctest --preset <windows-vs2026|windows|linux|macos>
```

这些 preset 的开发 feature contract 相同（TLS/CUDA 显式 OFF，但 HTTP/2/ggml/Vulkan 仍按默认能力构建），
因此不能称为 CPU-only；它们不强迫 Unix 使用 Ninja。
完整一步入口是 `cmake --workflow --preset <windows-vs2026|windows|linux|macos>-check`。VS 2026
原生入口要求 CMake 4.2+；`windows` 是 VS 2022 回退。Windows 若要让 Ninja 使用已选择的最新
MSVC，改用 `scripts/hyle-windows.ps1` + `windows-ninja`；安装 clang-cl 18+ 后可用同一 wrapper +
`windows-clangcl` 在独立 build tree 做可移植性复核。

- 看配置日志确认可选能力状态：`io TLS <AUTO|ON|OFF>` / `compute CUDA <AUTO|ON|OFF>` / `ggml via FetchContent`。
- lock-free 改动**必须**过 WSL TSan（[`linux-wsl.md`](./linux-wsl.md)）。
- 性能数字**必须**在原生文件系统（非 9p / 非网络盘）跑，见各平台「性能测试」节。
