# Windows 基础环境

普通初始化、构建、测试和安装统一使用[三个 Python 入口](README.md)，无需每次手工初始化 VS shell。
脚本按已保存的编译器和 VS 实例恢复子进程环境，不永久修改当前用户的 PATH、INCLUDE 或 LIB。

## 已安装工具

| 工具 | 要求 |
|---|---|
| Python | 3.10+，安装入口见 [Python](https://www.python.org/downloads/windows/) |
| CMake | 最低版本见[初始化说明](README.md)，见 [CMake 下载](https://cmake.org/download/) |
| Visual Studio / Build Tools | MSVC 19.43+，包含 C++ 桌面工作负载、Windows SDK 和 MASM |
| clang-cl | 可选，18+；仍需 Windows SDK 与 MSVC 头、库 |
| Ninja | 安装后可使用；初始化会发现已有的构建工具 |

Visual Studio 的安装步骤见 [Microsoft C++ 安装说明](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)。
已有多个 VS/toolset 时，初始化选择合格实例；后续 build/test 使用相同工具链。
切换 MSVC 与 clang-cl 应重新初始化相应配置，不在原 CMake cache 中原地换编译器。

CPU 与 CUDA 完整产品当前支持 Windows x86_64；CUDA 另需 CUDA Toolkit 13.3+ 和可用驱动。
Vulkan 源码构建需要宿主 shader 编译器；缺少时初始化可从中央 lock 准备它。
脚本不会替用户安装 CUDA Toolkit 或显卡驱动。

## 平台注意事项

- 从普通 PowerShell 运行 `python scripts/test.py` 时，独立头测试也会取得对应 MSVC 环境。
- `--sanitizer asan` 支持 Windows；TSan/UBSan 使用受支持的 Unix 工具链。ASan runtime 必须与编译器匹配。
- 构建目录过长时，使用仓库内较短的 `--build-dir .scratch/b/<label>`。
  系统长路径开关不能解除每个编译器或第三方子工具的路径限制，见
  [Windows 长路径说明](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)。
- CMake preset 仍可供 IDE 或直接 CMake 使用；直接维护命令需要相应 Developer Shell。
  VS 2026 generator 本身需要 CMake 4.2+，普通 Python 入口无需用户选择 generator/preset。

TLS 需要与 MSVC ABI 匹配的 OpenSSL 3 开发包；可按组织既有方式安装，例如
[vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-msbuild)。
`init.py --tls` 会自动查找常见安装位置；非标准位置可用高级参数
`-D OPENSSL_ROOT_DIR=<prefix>` 指定。
不要使用 Git for Windows 附带的 MinGW OpenSSL 代替 MSVC 开发库。

离线介质准备见[离线构建](offline.md)；性能采样要求见[验证环境](../performance/README.md#环境与解释限制)。
