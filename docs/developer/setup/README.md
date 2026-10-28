# 初始化、构建与测试

Windows、Linux 和 macOS 共用三个入口，只依赖 Python 标准库：

```text
python scripts/init.py
python scripts/build.py
python scripts/test.py
```

准备 Python 3.10+、CMake（最低版本以[根 CMakeLists.txt](../../../CMakeLists.txt)为准，当前为 3.25） 和受支持的 C/C++ 工具链；Unix 上若命令名是 `python3`，
将示例中的 `python` 替换为 `python3`。基础工具的安装要求见
[Windows](windows.md)、[Linux / WSL](linux-wsl.md)、[macOS](macos.md)。
初始化会发现已装工具、准备所选后端的依赖并运行 CMake；不会安装系统编译器、驱动或更改全局环境。

新配置默认为 **CPU、Release、共享库、推理引擎及其依赖、tests OFF**，TLS、HTTP/2、trace 均关闭。
Provider 和第三方实现由 HYLE 自动装配，不需要用户选择。
已有配置保留其明确选择；只构建基础模块时传 `--components util` 或所需模块集合。

## 常见用法

```text
# 默认引擎工程
python scripts/init.py
python scripts/build.py --jobs 2

# CUDA 引擎，启用测试环境
python scripts/init.py --backends cuda --components engine --tests --name inference
python scripts/build.py --target hyle_engine --jobs 2

# 只构建工具模块，不准备 Compute/GPU 依赖
python scripts/init.py --components util --name utility
python scripts/build.py --jobs 2

# 为上面已启用测试的配置构建并运行相关测试
python scripts/build.py --name inference --target hyle_tests --jobs 2
python scripts/test.py --name inference --filter compute --jobs 2

# 将引擎配置安装到指定目录
python scripts/build.py --name inference --install .scratch/install/hyle --jobs 2
```

`init.py` 保存当前选中的初始化目录。`build.py` 和 `test.py` 无参使用当前配置；
也可通过 `--name inference` 或高级参数 `--build-dir <path>` 选择已有配置。
`test.py` 只运行 CTest，不隐式构建；未启用测试时会给出重新初始化的命令。
初始化未完成或失败时，默认 build/test 拒绝沿用旧产物。修复原因后可无参运行 `init.py` 重试，
也可用 `--name` / `--build-dir` 明确选择另一份已经成功的配置。

切换 Debug、编译器或后端时再次运行 `init.py`。默认构建目录位于
`.scratch/build/` 中与实际配置匹配的目录；相同配置复用已有目录和共享依赖，
不兼容的工具链不会混进原 cache。并行会话应选择不同的 `--name` 或 `--build-dir`。
同一标签重配测试开关时可用 `--tests` / `--no-tests`。初始化失败不冒充新的可用配置。
缩小产品集合时，只准备新集合实际需要的依赖；例如从 `all` 切到 `util`，旧 TLS／HTTP/2 设置
不会要求继续构建协议模块。显式请求 `--tls`／`--http2` 时应同时选择 `io`。

## 选择配置

| 参数 | 含义 |
|---|---|
| `--backends cpu` | `cpu`、`cuda`、`metal`、`vulkan` 或逗号组合，默认 CPU |
| `--backends all` | 本平台支持的全部完整后端；所需系统 SDK 缺少时明确失败 |
| `--components engine` | 默认选择引擎，自动带出所需基础模块；可构建 `hyle_engine`，输出 `hyle` |
| `--components util` | 只选择既有工具模块及其依赖；也可传 `mem,container`、`compute,graph` 等产品集合 |
| `--components all` | 顶层完整项目，包含全部库与引擎 |
| `--config Debug` | `Release`、`Debug`、`RelWithDebInfo` 或 `MinSizeRel` |
| `--compiler clang` | `auto`、`gcc`、`clang`、`msvc` 或 `clang-cl` |
| `--tests`、`--static` | 开启测试、选择静态库；均有对应 `--no-...` 形式 |
| `--sanitizer asan` | `off`、`asan`、`tsan` 或 `ubsan`，受平台支持范围约束 |
| `--tls`、`--http2`、`--trace` | 开启所需协议或诊断能力 |
| `--name <label>` | 为常用配置设置简短标签 |
| `--cache-dir <path>` | 高级：指定 checkout 外的共享源码 cache |
| `--offline` | 禁止联网，使用已准备好的依赖与 SDK |
| `--dry-run` | 检查并打印计划，不配置或取得依赖 |

系统自动检测。Windows/Linux x86_64 的 `all` 包含 CPU、CUDA、Vulkan；macOS ARM64 包含
CPU、Metal、Vulkan。其它架构只接受实际支持的后端。`metal` 对应 Apple MLX/Metal 实现。
这些参数选择编译进库的可用能力；引擎运行时使用 CPU、GPU 或混合执行，由具体 Context 和引擎策略决定。
请求能力缺少系统 SDK 时明确失败，不静默裁剪。

需要 ATen 的 CPU/CUDA 产品闭包会自动准备、链接和安装匹配的 SDK，不提供用户 ATen 开关。
只构建 `util` 等独立模块或不依赖 ATen 的其它后端，不会额外取得该依赖。
算子仍按真实 dtype、布局、设备及 replay 能力准备；没有合法实现时明确拒绝，不通过降低精度
或隐式更换设备补齐。应检查实际 Provider 和资源需求。

完整参数及默认值以各脚本 `--help` 为准。`build.py --target <name>` 可重复；
`test.py --filter <regex>` 使用 CTest 名称过滤，`--list` 只列出测试。
`init.py --config` 使用表中的四种标准配置；消费已有 CMake 构建树时，build/test 也接受该树
实际提供的自定义配置名称，不会把单配置构建树切成另一种配置。
`build.py --install` 未指定 prefix 时写入 `.scratch/install/<build-name>`；
安装会构建完整的所选产品，不能与 `--target` 混用；选择引擎时还安装 `bin/hyle`（Windows 为 `hyle.exe`）。
构建树内程序位于 `<build-dir>/bin/`，多配置生成器位于 `bin/<config>/`。
引擎的模型输入准备和当前运行限制见[引擎指南](../../user/inference.md)。高级的额外 CMake 输入使用
`init.py -D NAME=VALUE`，可重复；脚本已有的选项使用对应参数。
产品与后端参数已经决定 Provider，旧 Provider 开关和内部 factor 选择不属于这个入口。
显式指定的 CUDA 编译器或 Toolkit 路径不可用时会报告错误；只有未指定时才自动寻找已安装版本。
普通日志保留 CMake 输出，`--verbose` 补充完整命令。安装入口使用标准 CMake 导出，
不修改系统 PATH、Shell 配置或用户的 CMake 搜索环境。
依赖 cache 与离线介质见[依赖说明](dependencies.md)和[离线构建](offline.md)。

## 维护检查

专项构建检查仍直接运行相应 CMake 脚本，不进入普通初始化流程。例如：

```text
cmake -DHYLE_SOURCE_DIR=. -DHYLE_SMOKE_ROOT=.scratch/smoke/core -DHYLE_SMOKE_COMPONENT=core -DHYLE_SMOKE_GENERATOR=Ninja -P cmake/HyleCiSelectedComponentSmoke.cmake
```

Windows 上这些直接调用应从对应的 Visual Studio Developer Shell 执行。
日常只运行与改动相关的测试和实验；手动 CI 使用同样的三个 Python 入口，不自动触发。
benchmark、代码生成和模拟器工具见 [scripts](../../../scripts/README.md)。
