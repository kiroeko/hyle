# HYLE 交付与集成

首次构建和最小消费者见[使用指南](usage-guide.md)，工具链与初始化参数见
[环境与构建](../developer/setup/README.md)。本页说明源码、静态 SDK 和动态 SDK 的交付合同。
推理引擎的 `hyle` 程序、当前模型和运行限制见[引擎指南](inference.md)；SDK 模块可以独立交付。

## 选择交付方式

| 方式 | 适用场景 | 集成入口 |
|---|---|---|
| 完整源码 | 修改实现、审计、整体编译或定制目标 | `add_subdirectory()` |
| 公开头 + 静态库 | 更新时重新链接，固定部署工具链 | `find_package(Hyle ...)` |
| 公开头 + 动态库 | 多个同工具链 C++ 程序共享库 | `find_package(Hyle ...)` |

三种方式使用相同的 `hyle::<product>` target 和 C++23 API。静态 HYLE 不表示全部第三方运行库
也静态链接；例如 CPU Compute 仍使用外部 shared CPU runtime。当前没有产品 C API 或跨语言 FFI。
许可证和再分发权限按仓库法律文本与第三方 notice 处理。

## 源码嵌入

```cmake
set(HYLE_COMPONENTS "mem;container")
add_subdirectory(path/to/hyle hyle-build)
target_link_libraries(my_app PRIVATE hyle::mem hyle::container)
hyle_stage_runtime(my_app)
```

`HYLE_COMPONENTS` 接受产品名的逗号或分号集合；嵌入时必须显式给出，顶层新配置默认 `engine`。
选择 `util`、`mem;container` 或 `compute;graph` 等模块只构建相应库闭包；
顶层 `all` 包含所有库与引擎，父工程显式 `all` 只选库，构建应用须明确包含 `engine`。
CMake 自动解析依赖。只链接代码直接使用的产品；不要手工链接内部 `_factor_*` target、
复制第三方 include 路径或枚举库顺序。

HYLE 根工程使用 C++；所选闭包含 mem 时会按需启用 mimalloc 所需的 C 编译器。
源码和依赖的离线准备见[离线说明](../developer/setup/offline.md)。交付源码时不要附带本机 build、
`_deps` 或依赖二进制缓存。仓库提供 `hyle_source_package` 构建目标，将 Git 跟踪的源码制作成
`<build-dir>/dist/hyle-<version>-source.zip` 并附 SHA-256；未提交的文件不会进入该包。

## 制作与消费 SDK

例如制作静态 mem/container SDK：

```sh
python scripts/init.py --name sdk-static --components mem,container --static
python scripts/build.py --install <安装目录>
```

shared SDK 省略 `--static`，使用不同的 build 名称和安装目录。`--install` 构建并安装所选产品的
完整闭包，不能与 `--target` 混用；省略安装目录时写入 `.scratch/install/<build-name>`。

消费者在自己的工程中：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem container)
add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE hyle::mem hyle::container)
hyle_stage_runtime(my_app)
```

用 `CMAKE_PREFIX_PATH` 指向安装 prefix，或将 `Hyle_DIR` 指向其中的 CMake package 目录。
安装只包含所选产品闭包的公开头和必需依赖。`Hyle_AVAILABLE_COMPONENTS` 列出包内产品；
请求未安装产品会失败。消费者不应依赖 producer build tree 中的绝对路径。

预发布 `0.y.z` 包要求完整精确版本，包括 patch；头、库和配置应来自同一次 SDK。
不承诺不同编译器、标准库、runtime、架构或 Debug/Release 之间的 C++ ABI 兼容。
不要把 static/shared 或不同配置覆盖安装到同一个 prefix 后依赖搜索顺序。

## 动态库和后端资源

源码和 SDK 消费者都可调用 `hyle_stage_runtime(my_app)`，应用 target 必须已经创建：

- Windows：把传递 DLL 闭包放到 executable 旁，后续构建也检查并更新已变化文件。
- Linux/macOS：补充本地 `BUILD_RPATH`；不把 SDK 路径写入应用的 `INSTALL_RPATH`。
- Metal：同时处理程序需要的原生资源，细节见 [Apple 指南](apple-compute.md)。

该 helper 用于本地构建运行。应用正式安装时，动态库、资源目录和 loader 路径由应用工程设置。
它不会安装 GPU 驱动、VC runtime 或其他机器组件。

在创建 executable 的同一 CMake 目录调用 helper。将 `find_package(Hyle)` 放在应用目录或公共祖先，
使 imported targets 可见；也可使用标准 `find_package(Hyle GLOBAL ...)`。普通跨目录 wrapper 可用。
若 wrapper 隐藏了仅对子目录可见的 Hyle SDK，或隐藏了登记附加资源的私有 SDK owner，所选配置的
构建会报可见性错误，避免漏拷贝。未启用的条件分支不触发该检查；源码嵌入不需要重新查找内部依赖。

Linux 静态 SDK 要链接进 shared library 时，制作 SDK 前设置
`-D CMAKE_POSITION_INDEPENDENT_CODE=ON`。消费者上的 PIC 设置不会重编译已安装的 archive。
内部测试插桩不能进入 SDK；普通测试可与安装同时启用。

## Compute / Graph 后端

初始化脚本使用 `--backends cpu,cuda,metal,vulkan` 的所需子集。直接 CMake 或源码嵌入使用：

```cmake
set(HYLE_COMPONENTS "compute;graph")
set(HYLE_COMPUTE_ROUTES "cpu;cuda")
add_subdirectory(path/to/hyle hyle-build)
target_link_libraries(my_app PRIVATE hyle::compute hyle::graph)
```

CMake 值为 `cpu`、`cuda`、`apple`、`vulkan`；CLI 的 `metal` 对应 `apple`。
默认 CPU，集合顺序不是运行时偏好。后端所需数学实现自动接入，不需要选择 oneDNN、ATen、ggml
等内部 Provider，也不需要手写其存在性宏。CPU/GPU 混合 SDK 与混合图的实际执行策略分开：
后者见 [Graph](graph.md#cpu-与-gpu-组合)。HIP/ROCm 当前没有产品配置项。

| 后端 | 交付时需要注意 |
|---|---|
| CPU | 使用锁定的 CPU 数学依赖；静态 HYLE 仍有外部 shared runtime，迁移时携带匹配的依赖资产 |
| CUDA | 完整产品支持 Windows/Linux x86_64；制作与消费 SDK 都需要匹配的系统 CUDA Toolkit，运行需要驱动 |
| Metal | macOS Apple Silicon；原生资源随 SDK 交付，具体布局与设备限制见 [Apple](apple-compute.md) |
| Vulkan | 运行机器需要 loader 与适用设备；SDK 内带已编译 shader，二进制消费者不需要 glslang；见 [Vulkan](vulkan-compute.md) |

系统工具链、Toolkit 版本和缓存准备统一见[依赖说明](../developer/setup/dependencies.md)。
macOS 每个构建目录只选择一个 `CMAKE_OSX_ARCHITECTURES` 架构，不支持同一树制作 universal 库。

已安装 SDK 的后端不会因消费者配置而改变。可查询 `Hyle_AVAILABLE_COMPUTE_ROUTES`，或公开 target 的
`HYLE_COMPUTE_ROUTES` 属性；需要缺少的后端就重新制作 SDK。消费者可在 `find_package` 前设置
`HYLE_COMPUTE_ROUTES` 断言所需子集；它不会下载或重编 Provider。

## SIMD 专家接口

普通数组运算只链接 `hyle::simd`。需要固定 ISA 的自定义 kernel 或锁定 Highway/xsimd 头时：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS simd)
add_library(my_kernel OBJECT my_kernel.cc)
hyle_target_simd(TARGET my_kernel ISA AVX2)
target_sources(my_app PRIVATE $<TARGET_OBJECTS:my_kernel>)
target_link_libraries(my_app PRIVATE hyle::simd)
```

helper 对整个 target 施加 ISA flags，因此为 kernel 建单独 OBJECT target，不要给整个应用加 AVX2。
SDK 自带所需开发头，消费者 configure 不取依赖；vendor vector 类型留在该编译单元，跨库接口使用
pointer/count/POD。更多固定、多 ISA 与运行时分派用法见 [SIMD 指南](simd-usage.md)。

## 发布应用

保留 HYLE 版本、目标平台、编译器与标准库、配置、linkage、后端集合、依赖和许可证信息。
对完整安装 prefix 归档，搬迁后用独立消费者确认 `find_package`、链接和本地运行。
单独复制几份头或库文件不构成完整 SDK。

HYLE 不维护 pkg-config、Bazel 或 Meson 的第二套依赖图。非 CMake 主工程可用小型 CMake bridge
生成所需 native artifact，再接入自己的发布流程。代码签名、包格式与应用升级策略由发行方管理。
