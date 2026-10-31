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

默认值、常用 components／backends、具名配置、构建／测试／安装及失败恢复，统一见
[初始化与构建指南](../docs/developer/setup/README.md)。`test.py` 只运行已构建的 CTest；
所需运行库由 CMake 暂存和安装，无需手动拼接 Provider DLL 路径。
模型输入与运行参数见[推理引擎指南](../docs/user/inference.md)，C++ 消费者见[使用指南](../docs/user/usage-guide.md)。

`hyle_perf/` 保留可复用的统计、数据模型和日志解析函数。专项 `validate-*-bench.py` 用于检查对应
benchmark 的结构、输出或资源事实，不决定是否可以开发，也不代替数值和实际运行验证。

产品依赖由 [HyleDependencyLock.json](../cmake/HyleDependencyLock.json) 管理，benchmark fetcher
只准备比较对象，不能替代产品依赖解析。SIMD route JSON 是代码生成输入，修改后使用生成器同步
对应生成物；方法见[SIMD 维护指南](../docs/developer/maintenance/simd-dispatch-guide.md)。

SIMD 配置检查使用 `python scripts/simd-op-coverage.py --check`；修改配置后用
`--write-provider-manifest --write-provider-routes` 生成头。`op-parity-report.py --check-orders`
检查实际采样输入和执行顺序，具体统计解释见[SIMD 测量方法](../docs/developer/performance/README.md#分辨率与极短循环)。
