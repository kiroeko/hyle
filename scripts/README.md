# 构建与诊断脚本

Windows、Linux、macOS 使用同样的三个入口：`python scripts/init.py`、
`python scripts/build.py`、`python scripts/test.py`。完整参数以各脚本的 `--help` 为准。[环境准备](../docs/developer/setup/README.md)说明平台与依赖，
[性能比较](../docs/developer/performance/README.md)说明计时、统计和公平比较。

| 任务 | 入口 |
|---|---|
| 发现工具链、选择可用后端、准备依赖并保存 CMake 配置 | `init.py` |
| 增量构建、选择目标、安装到指定 prefix | `build.py` |
| 运行已构建的 CTest，可按名称过滤 | `test.py` |
| 取得 benchmark 的外部参考库 | `fetch-bench-vendors.{sh,ps1}`、`setup-bench-libs.sh` |
| 汇总 nanobench 的重复运行与配对数据 | `bench-stat.py` |
| 分析 SIMD 操作比较与校准输出 | `op-parity-report.py` |
| 隔离测量 RSS | `mem-footprint-stat.py` |
| 检查公开头自包含 | `header-self-contained.py` |
| 检查实际共享库符号导出 | `check-shared-export-hygiene.py` |
| 检查特定优化的生成代码 | `*-codegen-audit.py`、`simd-exact-artifact-check.py` |
| 在模拟器中验证 scalable ISA 执行与正确性 | `rvv-qemu-test.sh`、`sve-qemu-test.sh`、`simd-scalable-exact-qemu-test.sh` |
| 生成并检查 SIMD 操作/Provider 路由 | `simd-op-coverage.py` |
| 检查 SIMD 公共边界的第三方类型/include | `check-no-vendor.sh` |

新配置默认选择 `engine`，构建 `hyle` 推理程序及其所需库。下面先准备 CUDA Release 开发与测试环境，
然后增量构建和运行测试；CPU 使用 `--backends cpu`，Apple Silicon 使用 `metal`，Vulkan 使用 `vulkan`。
编译器、CUDA Toolkit 等系统工具需先安装，项目依赖会复用共享缓存。

```sh
python scripts/init.py --name engine-cuda --backends cuda --config Release --tests
python scripts/build.py --name engine-cuda
python scripts/test.py --name engine-cuda
```

`--components` 是项目选择入口：`engine` 自动包含 compute、graph、执行进度和资源传输等所需库；
`util`、`compute,graph` 等选择只构建对应库；`all` 选择完整项目。库使用者无需开启引擎，
例如 `init.py --name util --components util --tests` 不准备 GPU 或推理 Provider 的依赖。
`--backends all` 准备当前系统支持的后端集合，`cpu,cuda` 等组合按需选择。
这些参数选择编译路线，具体模型与运行设备以程序实际支持为准；同时编译 CPU 和 CUDA 不等于启用混合执行。

调试配置可另建 `init.py --name engine-debug --backends cuda --config Debug --tests`。
多个配置共享依赖缓存，各自保留产物；重复初始化已有名称时，未指定的组件、配置和测试选项保持原值。
`build.py` 默认构建选定项目及已启用的测试，只需要程序时可用 `--target hyle_engine`；
之后运行测试仍需先构建相应测试目标。`build.py --install [prefix]` 安装完整选定项目，
包含引擎的选择提供 `bin/hyle`（Windows 为 `hyle.exe`）及库；只选择库则安装该 SDK 子集。
所需运行库由 CMake 暂存和安装，不需要手动拼接 Provider DLL 路径。

`hyle_perf/` 保留可复用的统计、数据模型和日志解析函数。专项 `validate-*-bench.py` 用于检查对应
benchmark 的结构、输出或资源事实，不决定是否可以开发，也不代替数值和实际运行验证。

`build.py` 和 `test.py` 默认使用当前选中的初始化目录，也可用 `--name` 或 `--build-dir` 选择其它配置。
初始化失败后默认操作会拒绝旧产物，修复后运行 `init.py` 重试。
`test.py` 不隐式构建；安装使用 `build.py --install [prefix]`，不更改用户环境。
专项维护检查仍直接使用 CMake target、CTest 或 `cmake -P`；按改动运行相关检查，报告失败与未运行项。

产品依赖由 [HyleDependencyLock.json](../cmake/HyleDependencyLock.json) 管理，benchmark fetcher
只准备比较对象，不能替代产品依赖解析。SIMD route JSON 是代码生成输入，修改后使用生成器同步
对应生成物；方法见[SIMD 维护指南](../docs/developer/maintenance/simd-dispatch-guide.md)。

SIMD 配置检查使用 `python scripts/simd-op-coverage.py --check`；修改配置后用
`--write-provider-manifest --write-provider-routes` 生成头。`op-parity-report.py --check-orders`
检查实际采样输入和执行顺序，具体统计解释见[SIMD 测量方法](../docs/developer/performance/README.md#分辨率与极短循环)。
