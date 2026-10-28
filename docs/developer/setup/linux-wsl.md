# Linux / WSL 基础环境

普通初始化、构建、测试和安装使用[三个 Python 入口](README.md)。脚本发现已安装的受支持
GCC/Clang，不把发行版默认的旧编译器自动当成合格环境，也不修改系统编译器链接。

## 已安装工具

需要 Python 3.10+、CMake（最低版本见[初始化说明](README.md)）、GCC 14+ 或 Clang 18+，以及 Ninja 或 Make 和系统汇编工具。
按实际发行版的软件源安装；例如 Ubuntu 24.04 可以使用其 GCC 14 包。
其他发行版应使用自己的包管理文档，不照搬不匹配的软件包命令。

选择 Clang 时使用 `python scripts/init.py --compiler clang`；完整 CPU 产品当前要求 Linux x86_64。
Linux AArch64 可构建受支持的基础组件，完整 CPU/CUDA 产品未提供该架构的依赖配方。

CUDA 需要 Linux CUDA Toolkit 13.3+ 和驱动。WSL 中应发现 Linux `nvcc` 与 Linux runtime，
不能把 Windows PATH 中的 `nvcc.exe` 当成 Linux 编译器。已有 cache 选错工具链时应重新初始化，
不把 Windows 的对象文件、SDK 或 CMake cache 搬进 Linux 构建目录。
Vulkan 缺少宿主 glslang 时，在线初始化可按中央 lock 准备它；离线时须提前准备相应工具。

## WSL 目录与缓存

源码副本、构建目录和依赖 cache 放在 WSL 自己的原生文件系统，避免在 `/mnt/c` 上做构建、
压力或性能测试。Windows 与 WSL 共享同一物理主机，不能当作两台独立性能机并行采样。

验证副本只同步源码和所需改动；排除 `.scratch/`、`build*/`、`_deps/`、对象和 SDK 安装目录。
复用同一固定 Hyle 工作目录中的对应 cache，不为每次实验复制整套依赖或 build。
项目当前源与验证机约定见 [AGENTS.md](../../../AGENTS.md)。

## Sanitizer 与可选能力

`init.py --sanitizer tsan --tests` 选择线程检查；ASan/UBSan 同样由 `--sanitizer` 指定。
若运行时因高熵 ASLR 报 `unexpected memory mapping`，按实际内核及 sanitizer 文档诊断；
需要关闭 ASLR 时可从 `setarch -R` 子进程运行测试。不要自动忽略新的 sanitizer 报告。

安装 OpenSSL 3 开发包后运行 `init.py --components io`，即包含 TLS 与 HTTP/2；nghttp2 源码按 lock
自动准备，无额外协议开关。非标准 OpenSSL 路径可用 `-D OPENSSL_ROOT_DIR=<prefix>` 指定，
缺少依赖时明确报错。需要引擎及协议库时可选择 `--components engine,io`。
更多依赖角色见[依赖说明](dependencies.md)，断网介质见[离线构建](offline.md)。
性能工具与比较方法见[性能比较](../performance/README.md)。
