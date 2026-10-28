# macOS 基础环境

普通初始化、构建、测试和安装使用[三个 Python 入口](README.md)。当前完整 CPU 与 Apple Compute
产品支持 Apple Silicon；选择 `--backends metal` 使用 MLX/Metal，`--backends cpu,metal` 同时构建 CPU 与 Apple GPU 能力。

## 已安装工具

需要 Python 3.10+、CMake（最低版本见[初始化说明](README.md)）、AppleClang/Clang 18+、匹配的 macOS SDK 和 Ninja 或 Make。
Command Line Tools 的安装方式见 [Apple 官方说明](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/)，
Python 和 CMake 可从[Python](https://www.python.org/downloads/macos/)与[CMake](https://cmake.org/download/)取得。
初始化发现已有工具；不替换全局 developer directory 或修改 Shell 启动文件。
系统自带的 `python3` 可能低于要求；使用已安装的 Python 3.10+ 运行三个入口，版本不足时会给出提示。

包含原生地址等待实现的 `platform_thread` 要求 macOS deployment target 14.4+。
顶层构建有对应默认值；父工程源码嵌入应在 `add_subdirectory(hyle)` 前设置合格的
`CMAKE_OSX_DEPLOYMENT_TARGET`。显式 `CMAKE_OSX_ARCHITECTURES` 每个 build 只接受一个架构，
不能用单一 Hyle 依赖/ISA 图直接构建 universal binary。

## Metal 与其他依赖

Apple 后端需要可执行的 `metal` 和 `metallib`。Command Line Tools 存在不等于 Metal 组件已经安装；
缺失时按 [Apple 额外组件说明](https://developer.apple.com/documentation/Xcode/downloading-and-installing-additional-xcode-components)
在 Xcode Settings → Components 中安装 Metal Toolchain。初始化会检查并给出指引，
不把某台验证机的工具目录、签名组件或临时 shim 当成用户安装步骤。

Vulkan 源码构建还需要宿主 shader 编译器；在线初始化可按 lock 准备缺少的 glslang。
TLS 需要 OpenSSL 3 开发包；`init.py --components io --tls` 自动查找常见安装位置，
非标准位置可用高级参数 `-D OPENSSL_ROOT_DIR=<prefix>` 指定。
非登录 SSH 中若找不到已安装的 Python/CMake，应使用实际可执行文件路径或临时进程 PATH。

## 验证副本与测量

按 [AGENTS.md](../../../AGENTS.md) 使用当前可用验证机。验证副本不反向覆盖产品源码，
也不把远端工具、SDK、旧 build 或整套依赖复制回工作区；优先复用固定 Hyle 工作目录里的缓存。

ARM 与 x86 的吞吐不能直接解释为库实现胜负。macOS 的 QoS 不等于固定到特定 P/E 核；
测量时记录实际电源、QoS 和低功耗状态，见[验证环境](../performance/README.md#环境与解释限制)。
离线工具与依赖介质见[离线构建](offline.md)。
