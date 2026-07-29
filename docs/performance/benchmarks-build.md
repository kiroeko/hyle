# build/package system 性能证书

构建系统没有运行时 ns/op，但它仍有用户可感知的正常路径性能：clean configure、无源码变化的
reconfigure/no-op build、改一个模块后的失效范围，以及依赖获取。这里单独记录点值；architecture
只解释 target 图，milestone 只记录施工结论。

## 方法

- 源码：2026-07-29 clean commit `75a8d3f`，Release，`HYLE_COMPONENTS=time`；Windows 为 MSVC 19.50 +
  Ninja，WSL 为 GCC 14.2 + Ninja，M4 为 AppleClang 21 + Unix Makefiles；
- product 图：tests/bench/TLS/CUDA OFF；
- test 图：tests ON、`HYLE_TEST_INSTRUMENTATION=OFF`，GoogleTest 源码用
  `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` 注入已有 pin，排除网络；
- clean configure 从新的系统临时目录开始；reconfigure 复用同一目录；
- initial/no-op build 构建 `hyle_time` 或 `hyle_time_tests`，并发度 2（与 package CI 的有界负载一致）；
- 数值是一次结构诊断，不冒充跨机器统计分布；三平台使用相同组件、依赖输入和并发合同。

## 三平台结果

| 平台 / 图形 | clean configure | reconfigure | initial build | no-op build |
|---|---:|---:|---:|---:|
| Windows product：`error,time` | 2.018s | 0.157s | 1.597s | 0.047s |
| Windows tests + 本地 pinned GoogleTest | 2.654s | 0.504s | 8.266s | 0.048s |
| WSL product：`error,time` | 15.810s | 0.020s | 2.330s | ≤0.010s |
| WSL tests + 本地 pinned GoogleTest | 17.330s | 0.150s | 10.900s | ≤0.010s |
| Apple M4 product：`error,time` | 0.340s | 0.040s | 0.570s | 0.060s |
| Apple M4 tests + 本地 pinned GoogleTest | 0.540s | 0.150s | 4.890s | 0.120s |

结论：三端 no-op 都在 10--120ms 以内，且 Ninja 明确报告 `no work to do`；最新把 root/tests/bench
CMake 下沉到模块目录后没有常态重编或全局 target 泄漏。Windows 单次 cold 点比旧证书高约
7--13%、M4 基本不变，WSL clean configure 从旧 18--23s 降到 15.8--17.3s；一次样本不足以把这些
冷启动波动判成输赢，但足以排除数量级回归。WSL 仍明显慢于 Windows/M4，product 与本地注入测试图
处在同一档，说明瓶颈在该镜像的 CMake compiler/feature configure，而不是 GoogleTest 下载或 HYLE
无关组件。测试首编增加的是 GoogleTest 与 test TU 的真实编译成本，不冒充产品图成本。

## 聚合 target 调度

shipping-layout CI 过去把 `hyle_error_tests hyle_platform_tests hyle_time_tests hyle_mem_tests` 作为四个
显式根交给 `cmake --build`；统一 `hyle_tests` 本来已经准确表达当前 component 闭包。Windows
VS 2026/MSBuild 在所有输出已存在后的交替 no-op 复测如下：

| 入口 | 三次 wall time | 稳态中心 |
|---|---|---:|
| `--target hyle_tests` | 17.226s（补齐此前未构建的 time 输出）、2.221s、2.192s | **2.21s** |
| 四个 module aggregate | 4.507s、4.440s、4.315s | **4.44s** |

首个 17.226s 不是 no-op，故不进入中心值。统一 aggregate 在 VS 生成器上的稳态调度约省一半；Ninja
本来就能把多根合成一张 DAG，主要收益是让本地、CI 和模块闭包使用同一个机器合同，而不是宣称编译器
本身加速。CI 已改为一次构建 `hyle_tests`。

## 异常与边界

既有全量 `build` 树曾每次为一个 io HTTP/2 test 重编约 100 个对象，并提示
`.ninja_deps: premature end of file`。移走这一个损坏数据库、让 Ninja 重建后，第一次重编 102
个对象，第二次为 70ms / `no work to do`。这是被中断或损坏的生成器状态，不是源码 target
持续失效；用户遇到同提示应重建对应 build tree 的依赖数据库，不能用反复全编作为正常基线。

普通 PowerShell 直接运行一个由 Developer Shell 配置的 MSVC/Ninja tree 会因缺少 `INCLUDE/LIB`
而报标准头不存在。受支持入口 `scripts/hyle-windows.ps1` 每次通过 vswhere 导入环境；这项属于
入口正确性，不计入 configure 性能。

更早的 WSL fresh `time,platform`/`io` 与 Apple M4 fresh `time,platform` 曾观察约
44s/49s/112s，但这些
运行包含 FetchContent 获取/配置，不能与上表的本地 pinned-source 图直接比较。后续证书必须同时
保留 dependency acquisition cold start 与 injected/offline graph 的分层，避免把网络速度写成
HYLE CMake 算法，也避免因 target 图快就否认用户第一次构建很慢。网络点值只作用户体验诊断，
不设为 HYLE 可优化的 SOTA 判据；正式/离线构建使用 pinned source override。

## 收口门

- [x] Windows、WSL、Apple 同组件、同依赖提供方式的 clean/reconfigure/no-op 表；
- [x] `time` 子集 target 图只含 `error,time`，不存在 compute/io/ggml 等无关失效对象；
- [x] shared/static/source-embed/install 既有三平台迁移矩阵保持有效，本轮 package 冷树重签；
- [x] clean commit 的 source zip 由 CI 核对 SHA-256/manifest/license，解包后脱离 checkout 构建；
- [x] tests 的 production-layout 与 instrumented-layout 分开，后者不能代替发布形状正确性；
- [x] 依赖下载失败有 SYSTEM/本地 override/离线说明，网络获取与本地图成本分层。
