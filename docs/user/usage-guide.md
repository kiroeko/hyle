# HYLE 使用指南

HYLE 是供 C++23 程序使用的基础库。普通集成只需要选择产品、计算后端并链接
`hyle::<product>`；内部实现和第三方库由 CMake 自动带入。它提供推理计算与资源管理能力，
模型格式解析、tokenizer、请求循环和 CPU/GPU 放置策略仍由推理引擎负责。

## 第一次构建并使用

先按[环境准备](../developer/setup/README.md)安装 Python、CMake 和对应编译器。
在 HYLE 仓库根目录运行：

```sh
python scripts/init.py --name first --components mem,container
python scripts/build.py --install
```

默认是 CPU、Release、shared，测试关闭。此例安装到 `.scratch/install/first`，不修改全局环境。
需要静态库时在初始化命令加 `--static`。重复执行无参数脚本会使用当前选中的构建；
切换配置、测试或后端的方法见[配置与构建](../developer/setup/README.md)。

在仓库旁创建 `my_app` 目录，将下面两个文件放进去。先写 `my_app/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.26)
project(my_app LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem container)
add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE hyle::mem hyle::container)
hyle_stage_runtime(my_app)
```

对应的 `main.cc`：

```cpp
#include <hyle/container/small_vector.h>

int main() {
  hyle::container::SmallVector<int, 4> values;
  return values.try_push_back(42) && values[0] == 42 ? 0 : 1;
}
```

从包含 `my_app` 的目录配置和构建应用：

```sh
cmake -S my_app -B my_app/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<HYLE安装目录的绝对路径>"
cmake --build my_app/build --config Release
```

安装目录是前面生成的 HYLE `.scratch/install/first`，不是其 `lib` 子目录。
消费者使用与 SDK 相同的编译器、标准库和配置。`init.py` 会打印 Compiler 路径，工具发现与 VS
环境恢复只作用于子进程，不修改当前终端的全局 `PATH`。Linux/macOS 上普通 CMake 可能选到另一版
默认编译器；必要时在上面的配置命令加 `-DCMAKE_CXX_COMPILER="<init 打印的 Compiler 路径>"`。
Windows 普通终端可使用对应 Visual Studio/toolset 的生成器；使用 Ninja 时需在相应开发工具环境中
配置消费者。工具链与配置兼容要求见[交付说明](distribution.md#制作与消费-sdk)。

构建完成后运行：

```sh
./my_app/build/my_app             # Linux/macOS，单配置生成器
```

Windows PowerShell 按生成器选择路径：

```powershell
.\my_app\build\my_app.exe          # Ninja 等单配置生成器
.\my_app\build\Release\my_app.exe  # Visual Studio 等多配置生成器
```

程序退出码为 0 表示此最小样例成功。
`hyle_stage_runtime()` 为本地运行处理动态库与后端资源；应用的正式安装布局由应用设置。
SDK 版本、工具链兼容性和部署要求见[交付与集成](distribution.md)。

## 直接嵌入源码

不制作 SDK 时，在父工程中选择所需产品：

```cmake
set(HYLE_COMPONENTS "mem;container")
add_subdirectory(path/to/hyle hyle-build)
target_link_libraries(my_app PRIVATE hyle::mem hyle::container)
hyle_stage_runtime(my_app)
```

父工程必须显式设置 `HYLE_COMPONENTS`，以免依赖意外构建整库；顶层独立构建省略时为 `all`。
只链接应用直接使用的产品，头路径、C++23 要求和传递依赖会自动传播。不要手工追加第三方库或
链接内部 `_factor_*` target。

## 开始使用 Compute / Graph

先选择编译进库的后端集合。例如：

```sh
python scripts/init.py --name inference --components compute,graph --backends cpu,cuda
python scripts/build.py
```

CLI 后端名为 `cpu`、`cuda`、`metal`、`vulkan`，可逗号组合。所选 SDK 缺失时明确失败。
源码嵌入使用对应的 `HYLE_COMPUTE_ROUTES` 集合，Metal 的 CMake 值为 `apple`。
构建多个后端只表示库具备这些能力，运行时使用 CPU、GPU 或混合图仍由应用明确决定。

| 任务 | 从哪里开始 | 可运行示例 |
|---|---|---|
| 准备一个算子，并对变化输入重复执行 | [Compute：第一次 Direct 计算](compute.md#第一次-direct-计算) | [Direct CPU/GPU](../../samples/package/compute/cpp/) |
| 构建多节点计算、规划中间存储并重复执行 | [Graph：第一次图执行](graph.md#第一次图执行) | [多节点 CPU/GPU 图](../../samples/package/graph/cpp/) |
| 使用原生 Metal 或 Vulkan 程序 | [Apple](apple-compute.md)、[Vulkan](vulkan-compute.md) | 各指南链接的后端示例 |
| 显式组合 CPU 与一个 GPU 的片段 | [Graph：CPU 与 GPU 组合](graph.md#cpu-与-gpu-组合) | [CPU+Metal](../../samples/package/graph/cpu_apple/)、[CPU+Vulkan](../../samples/package/graph/cpu_vulkan/) |
| 复用外部资源、分块装入权重 | [Compute 资源租约](compute.md#外部资源租约与-direct-提交)、[文件管线](compute.md#有界文件与资源管线) | [资源租约](../../samples/package/graph/resource_leases/)、[文件流](../../samples/package/graph/file_streaming/) |
| 分页 KV、多请求 attention 或状态快照 | [批次 attention](batch-attention.md)、[状态版本](state-versions.md) | 按指南选择对应路线 |

支持由实际设备、dtype、编码、形状、布局和资源决定。描述合法不等于后端可执行；
检查 `prepare()` / `compile()` 的返回值，并等待成功 `Completion` 后再读取输出。

## 按任务选择模块

| 需求 | 模块 |
|---|---|
| 编译环境、程序不变量、cache layout | [core](core.md) |
| 可恢复失败与带上下文的返回值 | [error](error.md) |
| OS、CPU、线程、虚拟内存、同步文件机制 | [platform](platform.md) |
| allocator、arena、pool、buffer、映射与 NUMA placement | [mem](mem.md) |
| 时钟、deadline、线程 sleep 与 timer 容器 | [time](time.md) |
| 日志、trace 与指标 | [obs](obs.md) |
| RAII、借用 callback、引用计数、随机数与 registry | [util](util.md) |
| 顺序、hash、并发、队列、文本与侵入式数据结构 | [container](container.md)、[侵入式容器](container-intrusive.md) |
| shape/byte-size 的 checked 整数运算 | [math](math.md) |
| 数组运算、类型转换、packing 与 ISA 分派 | [simd](simd-usage.md) |
| CPU task/fiber、同步、取消与消息 | [sched](sched.md) |
| 异步文件、网络、HTTP、WebSocket 与可靠传输 | [io](io.md) |
| 子进程、stdio、等待与进程树回收 | [process](process.md) |

使用[公开样例](../../samples/README.md)查看完整错误处理与关闭顺序。跨线程、fiber 挂起、
借用 buffer 或异步设备工作前，阅读[调用安全边界](usage-contract.md)及对应模块的生命周期说明。
