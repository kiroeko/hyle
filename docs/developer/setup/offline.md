# 离线构建

离线机器使用同样的 `init.py`、`build.py`、`test.py`，通过 `init.py --offline` 禁止依赖下载。
需要准备基础工具、HYLE 源码、所选闭包的锁定依赖，以及相应系统 SDK/runtime。
不要搬运对象、`CMakeCache.txt` 或整棵在线 build 目录。

## 准备基础工具

- Windows：Python 3.10+、CMake（最低版本见[初始化说明](README.md)）、包含 C++/Windows SDK/MASM 的 VS 或 Build Tools；
  clang-cl 构建另带该 frontend。离线安装介质按 [Microsoft layout 说明](https://learn.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=visualstudio)
  准备，与目标 workload 保持一致。
- Linux/WSL：通过对应发行版的受控镜像或离线包准备 Python、CMake、GCC 14+/Clang 18+、
  Ninja/Make、系统头和汇编工具。依赖包必须匹配目标发行版，不能只复制准备机恰好缓存的顶层包。
- macOS：准备匹配系统的 [Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/)、
  Python、CMake，以及所选 Apple 后端需要的 Metal 组件。

CUDA Toolkit、驱动和 OpenSSL 等另按目标系统准备。它们不属于 FetchContent 源码介质。
基础工具和系统 SDK 的准备方法也见[平台说明](README.md)。

## 准备所选依赖

在联网准备机，用相同产品和后端完成一次初始化，使共享 cache 含有该闭包的资产。
例如准备 CPU Compute/Graph 测试所需依赖：

```text
python scripts/init.py --backends cpu --components compute,graph --tests --cache-dir <shared-cache>
```

共享 cache 的源码部分也可由 `cmake -DHYLE_DEP_ACTION=sync -P cmake/HyleDependencyCacheTool.cmake`
预取，但该命令不替代各后端的平台 SDK 或宿主工具准备。需要 ATen 的 CPU/CUDA 产品闭包会
自动装配其 SDK，离线介质须提前包含匹配目标平台/架构的资产；独立模块和不依赖 ATen 的其它
后端不额外要求该 SDK。Vulkan 的宿主 glslang 也须提前准备。准备机与目标机不匹配时，
使用目标平台的联网准备环境或经过核实的相应 SDK 介质。

搬运完整共享 cache 时保留根 marker、身份 metadata、archives、sources 和所需平台 SDK 资产。
核对传输前后的 hash，并保留随附许可证。不要把任意下载目录冒充 HYLE 管理的 cache。
目标机的 `_deps` 仍由本地构建生成，不复制准备机的二进制子构建。

离线初始化使用搬入后的实际目录：

```text
python scripts/init.py --backends cpu --components compute,graph --tests --cache-dir <shared-cache> --offline
python scripts/build.py --target hyle_tests --jobs 2
python scripts/test.py --jobs 2
```

初始化会重验实际使用的锁定输入；缺少依赖、SDK 或宿主工具时明确失败。
`--offline` 不等于安装操作系统包，也不替代对实际断网环境的运行检查。

## 逐项源码介质（维护者入口）

已有受审源码包时，可以保留原生 CMake preload：

```text
hyle/
  hyle-offline-deps.cmake
  deps/<lock-name>/
```

目录名和所需源码以 [中央 lock](../../../cmake/HyleDependencyLock.json) 及实际产品闭包为准。
原样官方 archive 展开树验证完整内容身份；Git checkout 验证精确 revision/tree，且不包含
tracked、untracked 或 ignored 修改。两者均验证许可证，不能通过修改 URL 与预期 hash 自证。
普通测试需要 GoogleTest；HTTP/2 和 trace 另有所需源码；完整 Compute 后端还包含自己的依赖。
只准备 Highway、xsimd、GoogleTest 并不能代表全产品离线闭包已经完备。

直接从已初始化的工具链 shell 运行维护配置：

```text
cmake -C hyle-offline-deps.cmake -S . -B .scratch/build/offline-core -DHYLE_COMPONENTS=core -DHYLE_BUILD_TESTS=OFF
```

此入口强制断网，所选源码缺失时不会借网络补齐。平台 SDK 资产仍需由合格的 cache 或标准
CMake 查找输入提供。普通用户无需为共享 cache 逐项编写 override。

源码、静态库和共享库的交付边界见[交付说明](../../user/distribution.md)。
