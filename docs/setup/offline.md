# HYLE 离线构建环境

> 本页面向无外网、隔离网或要求依赖先审查后进入构建区的机器。它回答“怎样准备一台能从干净目录
> 编译 HYLE 的机器”；三种交付形状与授权边界见 [`../guide/distribution.md`](../guide/distribution.md)。

## 先闭合四层，不要复制一棵在线 build 目录

一份可复现的离线环境由四部分组成：

1. **HYLE 源码**：带版本/commit、SHA-256、`LICENSE` 和 `THIRD_PARTY_NOTICES.md` 的源码包；
2. **宿主工具链**：C/C++ 编译器、平台 SDK、CMake 3.25+、generator；完整测试还需要 Python 3，
   制作 source package 或跑部分 benchmark 工具还需要 Git；
3. **FetchContent 源码闭包**：放在 HYLE 根目录内的 `deps/<FetchContent-name>/`，由
   `hyle-offline-deps.cmake` 映射；
4. **系统 SDK/运行时**：OpenSSL、CUDA Toolkit/driver、Vulkan loader/ICD 等。这些由
   `find_package()`、toolchain/sysroot 或运行环境提供，不属于 FetchContent bundle。

不要交付 `<build>/_deps`、`CMakeCache.txt`、对象文件或整棵 build tree：它们包含生成器、绝对路径、
编译器和宿主机状态，不是可迁移依赖包。在线准备机应与离线目标的 OS、架构、发行版/SDK 大版本匹配；
所有安装包和源码目录记录来源版本、上游 commit/tag、SHA-256 与许可证。

## 按功能准备源码依赖

只需携带本次 component/feature 会触达的闭包：

| 构建能力 | `deps/` 目录 |
|---|---|
| `HYLE_BUILD_TESTS=ON` | `googletest/` |
| `HYLE_IO_HTTP2=ON` | `nghttp2/` |
| `HYLE_COMPUTE_GGML=ON` | `ggml/` |
| `HYLE_COMPUTE_VULKAN=ON` 的产品库 | `vulkan_headers/`、`volk/` |
| Vulkan 且 tests/bench 打开 | 再加 `glslang/` |

目录名必须与源码中的 `FetchContent_Declare()` 名字一致。一个含全量可选源码依赖的交付可采用：

```text
hyle/
  hyle-offline-deps.cmake
  deps/
    googletest/
    nghttp2/
    ggml/
    vulkan_headers/
    volk/
    glslang/
```

`hyle-offline-deps.cmake` 自动扫描一级子目录，并设置标准
`FETCHCONTENT_SOURCE_DIR_<UPPER_NAME>` cache 变量；它不复制依赖清单，也不下载内容。某个已启用能力
对应的目录缺失时，FetchContent 仍会尝试正常取得依赖，在真正断网的验证机上应因此失败，不能把
已有在线 cache 当成“离线成功”。pin、许可证和 bundle 清单要求详见 [`dependencies.md`](./dependencies.md)。

## 准备宿主工具链

### Windows

在联网 Windows 机器下载 Visual Studio Build Tools bootstrapper，然后建立可搬运的 C++ workload
layout。Build Tools 的 C++ workload ID 是 `Microsoft.VisualStudio.Workload.VCTools`：

```powershell
vs_BuildTools.exe --layout D:\hyle-offline\vs-layout `
  --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --lang en-US
```

在离线机器从同一 layout 安装，并保持相同 workload/options；`--noWeb` 防止安装器补取未打包组件：

```powershell
D:\hyle-offline\vs-layout\vs_BuildTools.exe --noWeb --wait `
  --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
```

另外携带 CMake 官方二进制包；使用 `windows-ninja` 时再带 Ninja，完整测试带 Python 3，制作源码包
带 Git。Visual Studio layout 路径保持较短并保留在稳定位置，安装器后续维护会记住该位置。官方入口：
[Visual Studio 离线 layout](https://learn.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=visualstudio)、
[Build Tools workload IDs](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=visualstudio)、
[CMake downloads](https://cmake.org/download/)。

安装后用 `scripts/hyle-windows.ps1`；脚本只发现/选择已经安装的 MSVC，不替你安装环境。

### Linux / WSL（Ubuntu 24.04 基线）

CPU build/test 基线包为 `gcc-14 g++-14 cmake ninja-build make binutils python3`；制作源码包或运行需要
Git 的工具时加 `git`，TLS 加 `libssl-dev`，AArch64 交叉编译再加
`gcc-14-aarch64-linux-gnu g++-14-aarch64-linux-gnu`。

长期或多机隔离环境应从受控的 Ubuntu repository mirror/snapshot 安装；一次性介质也必须在与目标
完全匹配的干净在线 VM 中下载包及其传递 `.deb`，再在断网副本验证，不能只复制在线机当前恰好
缓存的几个顶层包。Canonical 提供
[repository mirror](https://documentation.ubuntu.com/landscape/how-to-guides/repository-mirrors/) 与
[air-gapped mirror](https://documentation.ubuntu.com/landscape/how-to-guides/repository-mirrors/manage-repositories-in-an-air-gapped-or-offline-environment/)
流程。WSL 仍须在原生 ext4 镜像构建，不在 `/mnt/c` 上复用 Windows build tree。

### macOS

在 Apple Developer 的 More Downloads 取得与目标 macOS 匹配的 Command Line Tools 离线安装包；
它提供 clang、系统 SDK、汇编器及 Git。再携带 CMake 官方 macOS binary；可使用系统 Unix Makefiles，
因此 Ninja 可选。完整测试另需可审计的 Python 3 安装包。官方入口：
[Apple Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools)、
[CMake downloads](https://cmake.org/download/)。

Homebrew cellar 目录不是可迁移 SDK 包。若要 TLS/Vulkan 等可选能力，应使用组织维护的 bottle mirror
或已经签名、记录版本的离线包，并通过 toolchain/`CMAKE_PREFIX_PATH` 显式暴露。

## 最小、确定的离线验收

下面验证全部 14 个 HYLE 模块的 CPU 基线，不启用需要系统 SDK 或额外协议/计算后端的 feature；
因为 tests 打开，bundle 只需 `deps/googletest/`：

```bash
cmake -C hyle-offline-deps.cmake --preset linux \
  -DHYLE_COMPONENTS=all -DHYLE_BUILD_TESTS=ON \
  -DHYLE_BUILD_BENCH=OFF -DHYLE_INSTALL=OFF -DHYLE_WERROR=ON \
  -DHYLE_IO_TLS=OFF -DHYLE_IO_HTTP2=OFF \
  -DHYLE_COMPUTE_CUDA=OFF -DHYLE_COMPUTE_GGML=OFF \
  -DHYLE_COMPUTE_VULKAN=OFF
cmake --build --preset linux --target hyle_tests --parallel 2
ctest --preset linux --output-on-failure
```

Windows Ninja 的等价入口：

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset windows-ninja `
  -C hyle-offline-deps.cmake -DHYLE_COMPONENTS=all -DHYLE_BUILD_TESTS=ON `
  -DHYLE_BUILD_BENCH=OFF -DHYLE_INSTALL=OFF -DHYLE_WERROR=ON `
  -DHYLE_IO_TLS=OFF -DHYLE_IO_HTTP2=OFF `
  -DHYLE_COMPUTE_CUDA=OFF -DHYLE_COMPUTE_GGML=OFF -DHYLE_COMPUTE_VULKAN=OFF
scripts/hyle-windows.ps1 -Action Build -Preset windows-ninja --target hyle_tests --parallel 2
scripts/hyle-windows.ps1 -Action Test -Preset windows-ninja --output-on-failure
```

制作不含测试的 CPU SDK 时可设 `HYLE_BUILD_TESTS=OFF`，此时连 googletest/Python 也不需要。若启用
HTTP/2、ggml、Vulkan、TLS 或 CUDA，按上表补齐源码目录和系统 SDK，并把相应开关明确设为 `ON`；
正式离线产品不要依赖 `AUTO` 随机器偶然降级。

最终验收必须在**全新 build 目录、网络实际断开或被构建沙箱禁止**的机器上执行，并核对：

- `cmake --version`、编译器版本、Python/Git（如需要）与平台 SDK；
- 传输前后的 SHA-256、bundle manifest、许可证；
- configure 日志中的 component/feature contract 与预期完全一致；
- build/test 没有读取旧 `_deps`、用户 package registry 或其他开发机 cache。
