# 初始化、构建与测试

新配置默认是 **CPU、Release、共享库、engine 及其依赖，tests OFF、trace OFF**。
在仓库根目录运行：

```text
python scripts/init.py
python scripts/build.py --jobs 2
```

这三个入口 `init.py`、`build.py`、`test.py` 在 Windows、Linux、macOS 上共用，只依赖 Python 标准库。
先安装 Python 3.10+、CMake（最低版本以[根 CMakeLists.txt](../../../CMakeLists.txt)为准，当前为 3.25）
和受支持的 C/C++ 工具链，平台要求见[Windows](windows.md)、[Linux / WSL](linux-wsl.md)、[macOS](macos.md)。
Unix 若使用 `python3`，替换示例中的 `python` 即可。

初始化会发现已装工具、按产品／后端闭包准备依赖并运行 CMake，复用已有共享缓存。
CPU/CUDA 闭包可能包含 ATen 等 SDK；首次取得依赖需要网络，`--offline` 缺少资产时明确失败。
它不安装系统编译器／驱动、不改变全局环境，也不取得模型或安装 Python tokenizer 环境。
Provider 自动装配，不需要用户选择；`io` 包含 TLS／HTTP/2，需要已安装的 OpenSSL 3 开发包，
nghttp2 按 lock 准备。依赖细节见[依赖说明](dependencies.md)和[离线构建](offline.md)。

## 常见用法

```text
# CUDA 引擎
python scripts/init.py --name inference --components engine --backends cuda
python scripts/build.py --name inference --target hyle_engine --jobs 2

# 只构建工具模块，不准备 Compute/GPU 依赖
python scripts/init.py --name utility --components util
python scripts/build.py --name utility --jobs 2
```

CPU＋CUDA 能力将上例的 `--backends cuda` 改为 `--backends cpu,cuda`；Apple Silicon 用 `metal`，
Vulkan 用 `vulkan`，混合能力用 `cpu,metal` 或 `cpu,vulkan`。这些选择表示编译能力，
实际设备与 CPU/GPU 分置仍由应用决定。只构建或安装基础库 SDK 时，可选择 `--components mem,container`
或 `compute,graph`；C++ 链接方式见[使用指南](../../user/usage-guide.md)。

## 测试与安装

对上面的 `inference` 配置显式启用测试，再构建并运行 CTest：

```text
python scripts/init.py --name inference --tests
python scripts/build.py --name inference --target hyle_tests --jobs 2
python scripts/test.py --name inference --filter compute --jobs 2
python scripts/build.py --name inference --install .scratch/install/hyle --jobs 2
```

这次初始化保留该配置的组件和后端。`test.py` 不隐式构建；未启用测试时会给出重新初始化的命令。
安装会构建完整的所选产品，不能与 `--target` 混用；选择引擎时还安装 `bin/hyle`（Windows为`hyle.exe`）。

## 切换与恢复

`init.py` 保存当前选中的配置；build/test 无参使用它，也可用 `--name` 或 `--build-dir` 显式选择。
已有配置保留未指定的组件、后端、配置和测试等选项。初始化未完成或失败时，默认 build/test 拒绝沿用旧产物；
修复原因后无参重试 `init.py`，或明确选择另一份已成功的配置。

切换 Debug、后端或测试开关时重新初始化对应配置，测试开关使用 `--tests`／`--no-tests`。
更换编译器需使用另一 `--name`／`--build-dir`，避免混用不兼容 cache。
默认构建目录在 `.scratch/build/`，相同配置复用目录与共享依赖；并行会话选择不同标签或目录。
缩小产品集合后只准备新集合所需依赖，例如 `all` 切到 `util` 不继续准备协议依赖。

旧配置先对同一标签或目录运行 `init.py`，清理已废弃的路由／Provider／协议开关并保留产品等选择；
build/test 遇到旧开关会提示重新初始化，不暗中重配或运行旧配置。迁移细节见[依赖说明](dependencies.md)。

## 选择配置

| 参数 | 含义 |
|---|---|
| `--backends cpu` | `cpu`、`cuda`、`metal`、`vulkan` 或逗号组合，默认 CPU |
| `--backends all` | 本平台支持的全部完整后端；所需系统 SDK 缺少时明确失败 |
| `--components engine` | 默认选择引擎，自动带出所需基础模块；可构建 `hyle_engine`，输出 `hyle` |
| `--components util` | 只选择既有工具模块及其依赖；也可传 `mem,container`、`compute,graph` 等产品集合 |
| `--components io` | 包含 TLS、HTTP、HTTP/2、transfer、watch 和 WebSocket 及其依赖，无额外协议开关 |
| `--components all` | 顶层完整项目，包含全部库与引擎 |
| `--config Debug` | `Release`、`Debug`、`RelWithDebInfo` 或 `MinSizeRel` |
| `--compiler clang` | `auto`、`gcc`、`clang`、`msvc` 或 `clang-cl` |
| `--tests`、`--static` | 开启测试、选择静态库；均有对应 `--no-...` 形式 |
| `--sanitizer asan` | `off`、`asan`、`tsan` 或 `ubsan`，受平台支持范围约束 |
| `--trace` | 开启诊断跟踪；选择实际包含跟踪点的产品 |
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
构建树内程序位于 `<build-dir>/bin/`，多配置生成器位于 `bin/<config>/`。
引擎的模型输入准备和当前运行限制见[引擎指南](../../user/inference.md)。高级的额外 CMake 输入使用
`init.py -D NAME=VALUE`，可重复；脚本已有的选项使用对应参数。
产品与后端参数已经决定 Provider，`io` 产品也决定协议依赖；旧 Provider／协议开关和内部 factor 选择
不属于这个入口。OpenSSL 位于非标准目录时仍可使用 `-D OPENSSL_ROOT_DIR=<prefix>`，
锁定源码也保留标准 `FETCHCONTENT_SOURCE_DIR_<NAME>` 覆盖方式。
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
