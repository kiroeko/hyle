# HYLE 离线构建环境

> 本页面向无外网、隔离网或要求依赖先审查后进入构建区的机器。它回答“怎样准备一台能从干净目录
> 编译 HYLE 的机器”；三种交付形状与授权边界见 [`../guide/distribution.md`](../../user/distribution.md)。

## 先闭合四层，不要复制一棵在线 build 目录

一份可复现的离线环境由四部分组成：

1. **HYLE 源码**：带版本/commit、SHA-256、`LICENSE` 和 `THIRD_PARTY_NOTICES.md` 的源码包；
2. **宿主工具链**：C/C++ 编译器、平台 SDK、CMake 3.25+、generator；完整测试还需要 Python 3，
   制作 source package 或跑部分 benchmark 工具还需要 Git；
3. **受锁源码闭包**：首选搬运联网准备机用 `HyleDependencyCacheTool.cmake sync` 生成的完整共享源码
   cache；也可把 lock commit 的干净 Git checkout，或 lock 内官方 archive 的原样展开树，放在 HYLE
   根目录内的 `deps/<lock-name>/`，由 `hyle-offline-deps.cmake` 验证并映射；
4. **系统 SDK/运行时**：OpenSSL、CUDA Toolkit/driver、Vulkan loader/ICD 等。这些由
   `find_package()`、toolchain/sysroot 或运行环境提供，不属于 FetchContent bundle。

选择 `simd` 或 `container` 时必须提供锁定 Highway；选择 `simd` 还必须提供锁定 xsimd。它们是由
HYLE target TU 直接编译的产品源码，而不是任意系统安装。SIMD bitpack 与 Container 的选定
string/bitset kernel 直接编译 Highway，SIMD 的少量已裁决 route 使用 xsimd；VQSort integrated port
已是 HYLE 源码包
内的修改代码，不另放 checkout。完整源码审计会验证其 origin、integration paths、adapted-content hash
和许可证副本。
benchmark 使用独立干净的 Highway/xsimd sibling checkout，避免产品输入与对手共用 include path；
这些 peer checkout 不代替产品 cache 内的 locked sources。

不要交付 `<build>/_deps`、`CMakeCache.txt`、对象文件或整棵 build tree：它们包含生成器、绝对路径、
编译器和宿主机状态，不是可迁移依赖包。在线准备机应与离线目标的 OS、架构、发行版/SDK 大版本匹配；
所有安装包和源码目录记录来源版本、上游 commit/tag、SHA-256 与许可证。

## 按功能准备源码依赖

只需携带本次 component/feature 会触达的闭包：

| 构建能力 | `deps/` 目录 |
|---|---|
| `container`，或任何传递依赖它的组件 | `highway/` |
| `simd`，或任何传递依赖它的组件 | `highway/`、`xsimd/` |
| `HYLE_BUILD_TESTS=ON` | `googletest/` |
| 有 sched/fcontext 且 `HYLE_IO_HTTP2=ON` | `nghttp2/` |
| QUEUED Compute 旧 `HYLE_COMPUTE_GGML=ON` baseline | `ggml/` |
| QUEUED Compute 旧 `HYLE_COMPUTE_VULKAN=ON` baseline | `vulkan_headers/`、`volk/` |
| 上述旧 Vulkan baseline 且 tests/bench 打开 | 再加 `glslang/` |
| `HYLE_TRACE=ON` | `tracy/`（映射为 `HYLE_TRACY_DIR`） |

目录名必须使用中央 lock 的 `name`；FetchContent 项与声明名一致，Tracy 则是映射到
`HYLE_TRACY_DIR` 的 external checkout。一个含全量可选源码依赖的交付可采用：

```text
hyle/
  hyle-offline-deps.cmake
  deps/
    highway/
    xsimd/
    googletest/
    nghttp2/
    ggml/
    vulkan_headers/
    volk/
    glslang/
    tracy/            # 仅 trace 构建
```

### 推荐：搬运 HYLE 管理的共享源码 cache

在联网准备机指定一个空目录，让管理工具取得中央 lock 中的全部可搬运源码：

```powershell
cmake -DHYLE_DEP_SOURCE_CACHE=D:\hyle-media\deps-cache `
  -DHYLE_DEP_ACTION=sync `
  -P cmake/HyleDependencyCacheTool.cmake
```

把整个 `deps-cache` 连同根 marker、`archives/`、`sources/` 和 identity metadata 一起做介质 hash 后
搬运。离线机配置时显式指定它并禁止网络：

```powershell
cmake -S . -B build-offline `
  -DHYLE_DEP_SOURCE_CACHE=D:\hyle-media\deps-cache `
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON `
  -DHYLE_COMPONENTS=all
```

configure 会重新验证所选 source tree 与许可证；目标机只在自己的
`build-offline/_deps`（或 `HYLE_DEP_BUILD_DIR`）生成平台/工具链相关状态。不要只复制某个在线 build 的
`_deps`，也不要漏掉 cache marker 后把任意目录冒充受管 cache。

### 备选：逐项受审源码 override

`hyle-offline-deps.cmake` 只接受
[`cmake/HyleDependencyLock.json`](../../../cmake/HyleDependencyLock.json) 中的产品源码名。Git checkout
验证精确 HEAD、`HEAD^{tree}`、零 tracked/untracked/ignored 文件；无 `.git` 的展开树验证 lock 中的完整内容
manifest。两种形状都核对许可证；通过后才设置标准
`FETCHCONTENT_SOURCE_DIR_<UPPER_NAME>`，Tracy 则设置 `HYLE_TRACY_DIR`。脚本同时强制
`FETCHCONTENT_FULLY_DISCONNECTED=ON` 和 `FETCHCONTENT_UPDATES_DISCONNECTED=ON`，因此缺少所选 feature
需要的目录会在首次干净配置中失败，不会访问网络或借共享 cache 蒙混过关。显式 override 的优先级
高于 cache，因此这个入口适合组织已经按依赖逐项审批的介质。
若所选 component/feature 完全不触达外部源码（例如 tests 关闭的 core/error-only 形状），`deps/` 可不存在
或为空；断网门仍开启。预载脚本会为未携带项登记预期本地路径，只有真正选择该项时才清晰报 missing source。

## 在线准备机：制作可证明的源码树

### 首选：校验并展开官方 archive（离线机无需 Git）

从 lock 读取 `archive.url`、`archive.sha256` 和 `expanded_tree_sha256`。先下载 archive 并核对传输
SHA-256，再把唯一的顶层目录剥掉，原样展开到 `deps/<lock-name>/`。下面 ggml 命令只展示
2026-08-25 当前 lock 的介质制作形状；lock 改动后必须重新读取值，不能把文档示例当第二份 pin：

```bash
curl -L -o ggml.tar.gz \
  https://codeload.github.com/ggml-org/ggml/tar.gz/8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b
echo '3623ae4195577c11795f035baa602fcb282b80aa9d91be024874b5023eb1d0b5  ggml.tar.gz' \
  | sha256sum --check
mkdir -p deps/ggml
tar -xzf ggml.tar.gz --strip-components=1 -C deps/ggml
```

不要在展开后运行格式化、换行转换、补文件或删“无用”目录。manifest 以排序后的三类记录计算：目录；
`path-hex + raw-file-SHA256`；`path-hex + original-relative-link-target-hex`。路径和链接目标拒绝控制字符，
链接先解析并拒绝绝对/逃逸目标；链接目标的原始拼写参与 hash，所以 `dir/file` 与 `dir/./file` 不等价。
增加/删除/改写文件、增加空目录或改变 symlink 都会失败。mtime、owner 和 Unix mode 不进入展开树 hash，
以便同一官方 archive 可在 Windows/Unix 复证；这些传输元数据仍由 archive SHA-256 保护。

ggml 当前官方 archive 含安全相对链接 `.pi/SYSTEM.md -> gg/SYSTEM.md`。Windows 上须使用能保留 symlink
的文件系统/解包工具；若安全策略把它扁平化或禁止创建，展开树会按设计不匹配，此时改用下节精确 Git
checkout，不要手工伪造。传输前后还应对整个介质 manifest 再做 SHA-256。

### 备选：精确干净 Git checkout

从 lock 读取每项 `repository` 和 `revision`，克隆后 detach 到精确 commit并保留 `.git`。例如：

```bash
git clone https://github.com/ggml-org/ggml.git deps/ggml
git -C deps/ggml checkout --detach 8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b
git -C deps/ggml rev-parse HEAD
git -C deps/ggml rev-parse 'HEAD^{tree}'
git -C deps/ggml status --porcelain=v1 --untracked-files=all
git -C deps/ggml ls-files --others --ignored --exclude-standard
```

后两条必须都输出空字符串。ignored generated/cache 文件仍可能被上游 CMake glob 或 include 优先级
消费，不能作为“干净”源码介质隐藏。前两条必须分别等于 lock 的 `revision`、`git_tree`。其他依赖同样按 lock 准备；
可使用组织内 Git mirror，只要最终对象树一致。传输前后再对整个介质/manifest 做 SHA-256。这样离线机
不必信任目录名或准备机缓存，配置本身仍会重验源码身份。

上述官方 archive 的原样展开树是默认安全合同，不需要逃生开关。只有组织刻意使用与官方展开树不同、
但已由 SBOM、内部签名 manifest 或其他外部系统独立证明的源码时，才可显式加：

```bash
cmake -DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON \
  -C hyle-offline-deps.cmake -S . -B build ...
```

这是逃生开关，不是等价证明：preload 会核对许可证、计算并打印实际树 hash，并拒绝 mismatch 树中的
**任何** symlink（安全相对链接也只允许出现在 digest 完全匹配的官方展开树中，防止多级链接经另一个
symlink component 实际逃出根目录），再把**本次已检查路径**作为当前 configure 进程的 GLOBAL property
attestation 交给 consumer；普通
`-D` cache 参数不能伪造该 property。`FETCHCONTENT_SOURCE_DIR_<NAME>` 即使同时设置开关或同名 cache
变量，也不能自行绕过 lock。release/CI 应使用官方
archive 的 exact expansion 或精确干净 Git checkout，并保留 archive/commit、lock 和介质校验证据。

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

另外携带 CMake 官方二进制包；使用 `windows-ninja` 时再带 Ninja。若离线机还要运行
`windows-clangcl` 可移植性门，准备 layout 时必须把 Visual Studio 的「C++ Clang tools for Windows」
一并纳入（或携带组织批准的 LLVM 18+ 离线包），安装后确认 `clang-cl --version` 与 PATH；普通
MSVC workload 不保证包含该 frontend。完整测试带 Python 3，制作源码包带 Git。Visual Studio layout
路径保持较短并保留在稳定位置，安装器后续维护会记住该位置。官方入口：
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

下面验证中央 manifest 当前登记的全部 target 及其 factor 闭包（其中包含 header-only
`util_core` 与 `container_intrusive` 内部 factor）的 CPU 基线，不启用需要系统 SDK
或额外协议/计算后端的 feature；
因为全组件包含 `simd` 且 tests 打开，逐项 override bundle 需要 `deps/highway/`、`deps/xsimd/` 与
`deps/googletest/`；若使用共享 cache 介质，则其中需要有这三个 locked identity：

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
HTTP/2 或 TLS 时，按上表补齐依赖并把开关明确设为 `ON`。ggml/Vulkan/CUDA 目前只为具名 Compute
baseline task 开启；未来 backend/provider 的离线合同以其新 factor closure card 为准。

OpenSSL、CUDA Toolkit、Vulkan loader/ICD/driver 是环境 SDK 或运行时，不是会被 HYLE 编译进产品的
源码 checkout，因此不写伪 commit、也不放进 `deps/`。它们必须由已审查的离线安装介质、toolchain/
sysroot/`CMAKE_PREFIX_PATH` 提供，并在验收记录实际版本。Tracy 不同：`TracyClient.cpp` 会直接编入
HYLE，所以属于 lock 管理的源码依赖，不能接受任意 `HYLE_TRACY_DIR`。

最终验收必须在**全新 build 目录、网络实际断开或被构建沙箱禁止**的机器上执行，并核对：

- `cmake --version`、编译器版本、Python/Git（如需要）与平台 SDK；
- 传输前后的 SHA-256、bundle manifest、许可证；
- configure 日志中的 component/feature contract 与预期完全一致；
- build/test 只读取本次明确指定并已验真的共享 source cache/override，不读取其他开发机 cache、旧
  build `_deps` 或用户 package registry。
