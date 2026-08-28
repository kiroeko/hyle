# Build/package 能力完整性存量回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)回放到已经关闭的
> `build-v4-01`。Build/package 是贯穿全部组件的开发单元，不是可链接产品组件，因此不伪造
> `module-status.json` 条目。本页只读审计既有候选；发现产品触发后，修复必须进入独立 round。

## 回放合同

- unit / replay：`build-package` / `build-package-capability-replay-v1`；
- reviewed round / task：`build-v4-01` /
  [`build-system-v4-util-edge-replay`](build-system-v4-util-edge-replay.md)；
- reviewed candidate / tree / terminal end：
  `469356394a185b5201675fd76ff1a10a4050528b` /
  `eabcba1fe1cffb484b20c377af0423028e2f9062` /
  `cb46d1183b7ed11757ab0661db893cdfd98b9f65`；
- replay base：`9559784feac490cf97715725739b57f7a88314d8`；reviewer：Codex；日期：
  2026-08-28；
- normal consumers：获得授权源码并用 `add_subdirectory()` 选择一个或多个产品根的 C++ 父工程；
  通过 relocatable CMake package 消费 selected static/shared SDK 的 C++ 工程；多个模块维护者在独占
  scratch 中并行 configure/build/test；离线、受锁、可审计的源码交付；
- north-star journey：本地量化推理引擎从源码组合 mem/container/simd/sched/io/compute 等模块，明确
  ISA/backend，能做全程序 LTO/PGO；同一产品图也能形成固定工具链的 static/shared SDK。模型资产、
  kernel 选择和服务部署策略不下沉到本单元；
- tier / instrument：`P0 / closure-correctness-v1`。没有 runtime 热路径或 ns/op；复杂度门是选取闭包、
  configure/invalidation、依赖取得和交付图，不产生 build-system SOTA 点值；
- 调查预算：45 分钟；每族最多两个已钉死 strong peers；一个隔离的 selected-root 负向复现；本回放
  不修改产品、不开托管 CI、不重跑 optional hardware matrix；
- strong peers：oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 的 target/install/package/
  CPack 与 toolchain/test 形状；Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f` 的 source embedding、target dependency、install/export、
  C++ standard propagation 和 pre-release ABI 边界；CMake 3.25+ 官方 target/package/FetchContent/
  preset/toolchain 模型作为语言标准合同；
- 范围：当前 component/target graph、source/static/shared、package/relocation、option/toolchain、
  online/offline dependency、license/source bundle、测试/benchmark/CI、版本/ABI/FFI 负空间、全部用户可见
  build entry 和正常构建复杂度；
- 非目标：替 compute/graph 补产品能力；运行 CUDA/Vulkan/TLS/HTTP2 设备证书；新增 Native FFI、
  Bazel/Meson/pkg-config/CPack 入口；以一轮 cold configure 点值重写 CMake；发布签名、SBOM 或商业
  artifact repository。

候选到 replay base 的 canonical 43 文件 build/package 实现边界为零 diff；
`cmake/HyleComponents.cmake` blob 在两端均为
`75604571c701f376509c6fe4eabc218d94f2872e`。候选以后 util owning-demangle 的 sample/产品发生了受审
变化，但 component manifest、package、option、lock、offline、workflow 和工具入口没有漂移。

## 自顶向下 capability-family map

| Family | 已准入用户旅程与来源 | 必须覆盖的生命周期、边界与成本 | 当前路线 | Coverage / action | 缺失边或负空间 |
|---|---|---|---|---|---|
| component identity and dependency projection | 模块消费者、并行维护者、CMake/Abseil/oneTBB | 唯一 ID、direct PUBLIC/PRIVATE/build-only edge、无环、header owner、selected closure、真实 target link、添加/删除模块 | `HyleComponents.cmake` 单一 manifest + `hyle_add_module()` + final graph contract | `COMPLETE / KEEP` | graph schema 全量验证是 O(V+E) 元数据任务；不能由目录顺序或共享 include root 替代 |
| selected source assembly | 授权源码父工程、北极星推理引擎 | 显式 roots、只建立闭包 target/依赖/feature；父工程拥有 generator/toolchain/output；未选模块的施工中状态不能阻断 | `HYLE_COMPONENTS` + `add_subdirectory()` + `hyle::<module>` | `PARTIAL / PRODUCT_TRIGGER` | target/依赖取得已 selected-only，但 include-owner/source-edge audit 在解析 roots 前遍历全库；见 `BUILD-CAP-PARALLEL-01` |
| installed static SDK | 固定部署、oneTBB/Abseil/CMake package | selected headers、archives、公开/私有静态闭包、系统/第三方 target、relocation、精确版本、法律资产 | `HYLE_INSTALL=ON,HYLE_STATIC=ON` + `find_package(Hyle ... COMPONENTS ...)` | `COMPLETE / KEEP` | 非 CMake consumer 通过薄 CMake bridge；无具名消费者前不发布可能不完整的 `.pc` |
| installed shared SDK and loading | 多程序共享 artifact、Windows/Linux/macOS | selected headers/DSO、transitive runtime、RPATH/DLL staging、relocation、loader identity、header/artifact 同构 | shared package + sibling RPATH + `hyle_stage_runtime_dependencies()` | `COMPLETE / KEEP` | shared 仍是同工具链 C++ SDK，不是稳定 ABI/FFI；跨 allocator/runtime 由匹配合同约束 |
| product version, configuration and binary decoration | source/static/shared 三种交付 | checked-in version 单源、0.x exact package/loader identity、static/shared export/import、C++23/compiler/platform fail-fast | `<hyle/version.h>`、visibility/build contract、exact `HyleConfigVersion` | `COMPLETE / KEEP` | 1.0 后才切 conventional major compatibility；不制造 version-only DSO |
| dependency acquisition and provenance | 在线开发、隔离网客户、可选 backend | selected-only obtain、版本/commit/archive/tree/license hash、用户 override、无网 fail closed、系统 SDK 分界 | dependency lock + FetchContent wrapper + `hyle-offline-deps.cmake` | `COMPLETE / KEEP` | 系统 compiler/SDK 不由项目安装；镜像可替代网络但最终内容 identity 必须相同 |
| source delivery and legal closure | 获授权源码的客户 | clean tracked tree、commit manifest、SHA-256、LICENSE/notices、in-tree/integrated dependency audit、脱离 Git 可构建 | `hyle_source_package` + 独立 offline dependency bundle | `COMPLETE / KEEP` | source zip 不携带 FetchContent cache；binary archive/sign/SBOM 属发布流水线，可从受检 install prefix 形成 |
| options, ISA/backend and toolchain composition | 普通 CPU 用户、专家 exact ISA、cross/toolchain consumer | owner-scoped option、确定默认、AUTO/ON/OFF、invalid fail closed、fresh cache、multi/single config、parent ownership | 少量产品/验证 option + standard CMake generator/toolchain variables + presets | `COMPLETE / KEEP` | LTO/PGO/unity/compile cache 使用标准 CMake/toolchain，不另造 HYLE 开关；真实后端由 owning module 验收 |
| module development and validation | 多 agent/session、CI、模块收口 | 独占 scratch、selected source/test/static/shared/relocation、aggregate tests/bench、header surface、shipping/test layouts | Windows wrapper + selected/component/package/source contracts + module aggregates | `PARTIAL / FIX` | scratch 隔离本身完整；全库源码扫描仍让未选模块影响 selected configure，破坏语义隔离 |
| incremental build and CI layering | 日常提交、有限 hosted 额度 | no-op 零重编、leaf invalidation 不扩散、PR 高信号、手工三平台/cross/feature/package，新增组件自动进基线 | generator DAG + one manifest + layered Actions | `COMPLETE / KEEP` | hosted/full 和真实设备未在本回放运行；CI 只证明 job 明示形状，不签模块能力 |
| alternate build/package ecosystems | 非 CMake 父工程、发行系统 | 不要求复制 transitive C++/CUDA/system closure；可重定位 install tree；归档/签名由部署 owner | installed CMake package 或薄 CMake bridge；标准 archive/sign 工具处理 install prefix | `ROUTED / DEFER` | oneTBB/Abseil 的 pkg-config/CPack 用于发现路线，不自动成为 HYLE MUST；具名 consumer + static/shared 全闭包 smoke 才触发新增入口 |
| Native FFI / C product | 尚无已命名外语 consumer | runtime type erasure、create/call/error/cancel/destroy/unload、独立 versioning | accepted Native FFI policy 为 `N/A` | `OUT_OF_SCOPE / REJECT` | 内部 C/assembly/OS callbacks 不建立产品 C ABI |

除 selected-source/parallel-audit family 外，当前 `MUST` 都为 `COMPLETE` 或拥有自然 `ROUTED` 路线。
该 family 是常用 source journey，不允许以“全库目前只有约三十万行”降为 GENERAL。

## 自底向上 public-surface trace

| 当前入口 | Capability / journey | 裁决 |
|---|---|---|
| `HYLE_COMPONENTS`、`HYLE_RESOLVED_COMPONENTS`、`hyle::<module>` | 选择产品根、观察闭包、链接真实组件 | `KEEP`；修复只限制审计作用域，不改变 target 名或闭包 |
| `HYLE_STATIC`、`HYLE_INSTALL` | static/shared 产品形状与 install/package 生成 | `KEEP`；不继承或改写父工程 `BUILD_SHARED_LIBS` |
| `HYLE_BUILD_TESTS/BENCH`、`HYLE_TEST_INSTRUMENTATION`、`HYLE_WERROR`、`HYLE_SANITIZER` | 正确性、诊断、benchmark 和维护强度 | `KEEP`；不是产品 feature，shipping/test layout 分开 |
| owner-scoped TLS/HTTP2/CUDA/ggml/Vulkan/trace 与 exact-SIMD controls | 显式 backend/ISA 合同 | `KEEP`；owner 未选择时不探测或广告 |
| standard CMake generator/toolchain/install-prefix/FetchContent source variables | 迁移、交叉编译、离线/source override | `ROUTED/KEEP standard`；不包装第二套拼写 |
| `hyle_stage_runtime_dependencies(target)` | Windows source/installed shared runtime staging | `KEEP`；static/Unix/header-only 安全 no-op |
| `Hyle_AVAILABLE_COMPONENTS/FEATURES`、`Hyle_STATIC` 与 `find_package` components | installed SDK 发现、拒绝缺失组件、检查产品形状 | `KEEP` |
| `hyle_source_package` | 干净授权源码归档 | `KEEP`；只在顶层 Git checkout 可用 |
| `hyle_<module>_tests/benches`、`hyle_tests/benches` | 模块/整图验证聚合 | `KEEP`；benchmark target 不形成产品能力 |
| `hyle-windows.ps1` actions 与跨平台 presets/workflows | 选择受支持 MSVC 环境、独占 scratch、常用验证入口 | `KEEP`；preset 固定树只供单会话开发，不供并行证据 |

没有 public C/managed target、`HYLE_ENABLE_NATIVE_FFI`、version-query DSO、全局输出目录劫持或
benchmark-only 产品开关。所有现有用户入口都能映射到具名 family。

## Strong-peer 差异裁决

- oneTBB 与 Abseil 都用 target/export package 支撑源码和安装消费；HYLE 的 selected component、
  direct-edge/header-owner、offline lock 与三种交付是自身真实模块化需求，不为追赶 peer 文件数而删除；
- 两个 peer 都提供 pkg-config，oneTBB 还提供 CPack。它们证明非 CMake/成品归档任务存在，但不能证明
  一个 `.pc` 能完整表达 HYLE static C++ link driver、CUDA/imported targets、runtime staging 和 feature
  闭包。当前完整 route 是用 CMake 产生/消费 install prefix，再由发布系统归档或由非 CMake 工程使用
  薄 bridge；具名消费者出现后按该消费者的 static/shared journey单独准入；
- peer 不在每次用户 configure 时扫描全部未选源码来证明自有 manifest。HYLE 可以保留比 peer 更强的
  direct-edge fail-closed 合同，但它必须只作用于正在组装的 selected closure；全量审计属于 all-component
  CI，不应把并行无关模块重新耦合。

## `BUILD-CAP-PARALLEL-01` 复现与机制

在 replay base 的隔离 detached worktree 中，只给未选择的 `graph` 实现增加一条未声明
`#include <hyle/obs/log.h>`，然后配置：

```text
cmake -S <probe> -B <probe-build> -G "Visual Studio 18 2026" -A x64
  -DHYLE_COMPONENTS=core -DHYLE_BUILD_TESTS=OFF -DHYLE_BUILD_BENCH=OFF
```

配置在建立 `core` target 前失败：

```text
component 'graph' implementation 'src/graph/taskgraph.cc' includes
'hyle/obs/log.h' owned by 'obs', but the component has no direct product dependency
```

根因不是 target graph：`HyleComponents.cmake` 在解析 `HYLE_COMPONENTS` 前，对
`HYLE_KNOWN_COMPONENTS` 全量执行 public/source include audit。于是 source consumer 的正常配置成本为
`O(V+E+S_all)`，且任一未选模块的施工中 include 会阻断别的模块；文档声称的 selected-only 审计与
独立并行 scratch 只完成了一半。

修复边界冻结为：先仅用 manifest 元数据解析/验证 selected closure，再只对该 closure 执行 public/source
audit；all-component CI 仍扫描全库。必须增加正负 fixture，证明未选模块 drift 不影响 `core`，而 selected
模块的同类漏边仍 fail closed。不得删掉 direct-edge audit、放宽真实 target 复核或借此重写 package。

## 裁决与下一动作

- 产品 verdict：`PRODUCT_TRIGGER`；不是证据不足，也不是纳秒/冷启动波动。它击中 v4 明示的并行
  独占 cache、selected-root 和更优复杂度重开条件；
- 保持事实：v4 candidate 的 source/static/shared、package/offline/license/source-bundle、option、CI
  与 version/FFI 结论没有被推翻；canonical build boundary 自候选零漂移；
- 下一动作：创建 `build-v5-selected-closure` 独立 round，只修 `BUILD-CAP-PARALLEL-01`、相应 regression
  和用户路由说明；修复前不得把本回放描述成 `CONFIRMED`；
- 性能边界：结构目标从 source configure 的 `O(S_all)` 收束为 `O(S_selected)`，不运行跨机器 cold-start
  排名，不因毫秒点值再改算法；
- 重开/停止：v5 适用 selected/component/source/static/shared 和文档门通过即可停止；若仍有未选源码
  耦合或 selected 漏边不再 fail closed，则明确 `BLOCKED`。

回放 base 的 canonical LOC self-check（snapshot
`9559784feac490cf97715725739b57f7a88314d8`）：全库代码 **298,788**；自实现包含测试
**122,321**；自实现不包含测试 **82,176**；build/package 实现 **7,726** 行。LOC 只用于规模复盘，
不作为触发或完整性证据。
