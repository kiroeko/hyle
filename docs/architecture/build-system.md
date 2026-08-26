# 构建与组装架构

> 状态：当前设计。本文描述仓库中实际生效的 CMake 契约，而不是发布承诺。
> Hyle 当前交付面是 C++：源码组装、静态 SDK、共享 SDK。历史 C/跨语言表面不属于当前产品设计。

## 设计目标

构建系统必须同时满足四件事：

- 模块边界可见：每个组件拥有明确 target，内部依赖不能藏在目录顺序或手工链接里。
- 组装可裁剪：源码使用者、静态 SDK 和共享 SDK 使用同一组件图。
- 外部依赖可复现：身份、版本、内容和许可证来自一份锁定记录，并投影到在线、离线、安装和 CI 流程。
- 回归可持续：新模块加入后自动进入基本构建测试；更昂贵的产品形态和硬件路径分层验证。

构建系统证明“这些源码能按声明的契约组装和消费”，不代替模块的功能、API 或性能收口。

目标架构由 `HyleTargetPlatform.cmake` 在编译器识别后归一化为
`HYLE_TARGET_PROCESSOR`，源码、ISA 与组件能力选择只读取这一事实。不能直接依赖
`CMAKE_SYSTEM_PROCESSOR`：Visual Studio 多平台生成器在 `-A ARM64` 时仍可能把它保留为宿主
`AMD64`，从而让 ARM64 产物错误选择 x86 源码。

## Target 与组件所有权

一个公开组件对应一个 CMake target：

- 构建树名称：`hyle_<module>`。
- 构建树和安装树的消费名称：`hyle::<module>`。
- `core`、`util_core` 与 `container_intrusive` 是可选择、可安装的 `INTERFACE` 组件；其余组件
  通常是编译型 target。
- `hyle_build_contract`、`hyle_testing`、`hyle_trace` 是横切构建 target，不是产品组件根。
- 横切 target 也必须遵守真实消费边：模块只有在源码或公开头读取 trace/testing gate 时才传播
  对应 target，并通过 `TRACE`/`TESTING` 显式 opt-in；配置会把声明与当前 selected sources/公开头
  反向对账。纯编译/布局地基 `core` 不声明这两个 gate，不能因全局开关无故携带 profiler runtime
  或 test-only ABI definition。
- Trace backend 在模块图声明后才建立；`HYLE_TRACE=ON` 而所选闭包没有 trace consumer 会配置失败，
  不会把 Tracy 悄悄变成第二个产品根。

各层只拥有一类事实：

| 位置 | 唯一职责 |
|---|---|
| `cmake/HyleComponents.cmake` | 组件全集、构造顺序、Hyle 内部 `PUBLIC`/`PRIVATE` 依赖 |
| `src/<module>/CMakeLists.txt` | 本模块源码、公开头、编译属性和外部实现 target |
| `tests/<module>/CMakeLists.txt` | 本模块测试；被选组件缺少该文件会配置失败 |
| `bench/<module>/CMakeLists.txt` | 可选 benchmark，不参与正确性证明 |
| `samples/package/<module>/` | 安装包消费样例，由 package smoke 自动发现 |

模块目录不得重新声明 Hyle 内部依赖。跨模块源码调用与 target 图不一致时，应修图或修调用，不能靠 `add_subdirectory()` 顺序偶然通过。

### 单一内部依赖 manifest

`HyleComponents.cmake` 是内部组件关系的唯一 manifest，并产生两种投影：

1. 配置开始时解析 `HYLE_COMPONENTS` 的传递闭包，决定实际加入哪些模块目录。
2. `hyle_add_module()` 从同一记录生成真实的 `target_link_libraries(... PUBLIC|PRIVATE ...)` 边。

配置结束时 `hyle_validate_final_module_graph()` 再核对 target 的 `LINK_LIBRARIES` 和
`INTERFACE_LINK_LIBRARIES`，防止后续手工链接绕过 manifest。

当前图的关键关系是：

```text
core, math  leaves
container_intrusive -> PUBLIC core
util_core   -> PUBLIC core
error       -> PUBLIC core
obs         -> PUBLIC core error
platform    -> PUBLIC core error
time        -> PUBLIC core error container_intrusive
mem         -> PUBLIC core error platform util_core
simd        -> PUBLIC core error platform
container   -> PUBLIC core error mem container_intrusive; PRIVATE platform
util        -> PUBLIC core error mem container util_core
sched       -> PUBLIC core error platform mem container util_core; PRIVATE time
io          -> PUBLIC core error mem container; optional sched
compute     -> PUBLIC core error mem; PRIVATE platform simd; optional sched
graph       -> PUBLIC core error mem container; PRIVATE util_core; optional sched
```

`PUBLIC` 表示消费者编译或链接本模块时也需要该依赖；`PRIVATE` 表示只服务本模块实现。这里记录的
是实际 target/link 契约。`container_intrusive` 证明 header-only 能力也应在公开类型穿过模块边界
时拥有真实 target：`time` 暴露 `TimerList`，故不能把这条边藏成 include-path 偶然可见；完整
`container` 则公开聚合该轻量目标。

Core 的 source contract 不再是 target-neutral 例外。公开头直接 include `<hyle/core/**>` 或
`<hyle/version.h>` 的组件必须直接声明 `PUBLIC core`，即使 `hyle::core` 是 `INTERFACE`、没有
binary symbol。配置阶段会扫描各组件公开头并与中央 manifest 对账：漏边和没有真实 include 的
多余边都失败。`container_intrusive`/`util_core` 这样的 header-only target 同样遵守这条规则；
复制整棵 include tree 不能使未登记依赖合法。

`<hyle/version.h>` 是 build/package 拥有的 checked-in 产品元数据，并由 Core include contract
交付；Core 本身不拥有 linked version query。`<hyle/visibility.h>` 的 export/import policy 也由
build/package 拥有，随各 compiled module 的 static/shared 形状维护，而不是 Core capability。

## 外部依赖：单点登记，多路投影

外部依赖分成两个正交登记层，不能相互替代：

| 登记层 | 文件/接口 | 回答的问题 |
|---|---|---|
| 来源与法律锁 | `cmake/HyleDependencyLock.json` | “究竟是哪份源码，内容和许可证是什么？” |
| 产品使用登记 | `hyle_register_package_dependency()` | “哪个模块使用它，安装包需要导出哪些 target、资产和下游依赖？” |

锁文件由 `HyleDependencyVersions.cmake` 读取，记录仓库或归档身份、revision、归档与展开树哈希、集成源码哈希、许可证表达式及许可证文件哈希。模块依赖脚本可以决定适配器的编译选项，但不得复制版本、URL、哈希或许可证事实。

同一登记事实投影到以下路径：

| 路径 | 契约 |
|---|---|
| 在线配置 | `FetchContent` 使用锁文件投影出的 URL/SHA256；统一 wrapper 在使用前后校验覆盖源和实际展开源 |
| 离线配置 | `hyle-offline-deps.cmake` 从同一锁读取 `deps/<name>/`，设置标准 `FETCHCONTENT_SOURCE_DIR_*`，并禁止下载 |
| 安装导出 | package registry 汇总 exported targets、静态闭包、`find_dependency()`、运行时/开发资产和工具 |
| 许可证 | 配置、打包和 CI 均按锁文件校验许可证文件及哈希；安装 `LICENSE` 与第三方 notices |
| CI/cache | lock 哈希参与缓存键；locked-feature job 使用官方归档验证适配器和安装包 |
| 源码包 | 打包前审计所有 in-tree/integrated-source 锁记录，但不把 FetchContent 下载物塞进源码包 |

`HYLE_DEPS_DIR` 默认位于当前构建树 `_deps`。嵌入父工程时只设置 Hyle 目录范围内的 `FETCHCONTENT_BASE_DIR`，不抢占父工程 cache。较重的可选后端发现留在相应模块脚本，例如 IO、compute 和 trace。

## Option 语义

用户可理解的 option 应表达产品选择或验证强度；计算得到的 `HYLE_BUILD_<COMPONENT>` 只是内部状态。

| 类别 | 选项 | 语义 |
|---|---|---|
| 组件 | `HYLE_COMPONENTS` | `all` 或逗号/分号列表；空值、未知名、`all` 与显式名混用均失败 |
| 链接形态 | `HYLE_STATIC` | `OFF` 明确构建 shared，`ON` 明确构建 static；不改父工程 `BUILD_SHARED_LIBS` |
| 安装 | `HYLE_INSTALL` | 生成可重定位 SDK；当前要求 `HYLE_BUILD_TESTS=OFF` |
| 验证 | `HYLE_BUILD_TESTS`, `HYLE_TEST_INSTRUMENTATION`, `HYLE_WERROR`, `HYLE_BUILD_BENCH` | 分别控制测试、测试插桩、警告门和 benchmark |
| Sanitizer | `HYLE_SANITIZER` | `off|asan|tsan|ubsan`；非法值配置失败 |
| SIMD | `HYLE_SIMD_EXACT_TARGET` | 空值或 `SCALAR/SSE4/AVX2/AVX512/NEON/SVE/RVV` |
| 自动特性 | `HYLE_IO_TLS`, `HYLE_COMPUTE_CUDA` | `AUTO` 探测、`ON` 必须存在、`OFF` 不探测 |
| 布尔特性 | `HYLE_IO_HTTP2`, `HYLE_COMPUTE_GGML`, `HYLE_COMPUTE_GGML_NATIVE`, `HYLE_COMPUTE_VULKAN` | 显式启停当前模块能力 |
| Trace | `HYLE_TRACE`, `HYLE_TRACE_BACKEND`, `HYLE_TRACY_DIR` | 启用并选择 trace 实现或预置源码 |

生成器、工具链、配置类型、安装前缀、CUDA architectures 和 `FETCHCONTENT_SOURCE_DIR_*` 使用标准 CMake 变量。正式证据与发布配置应把会改变产物的 `AUTO` 固定为 `ON` 或 `OFF`。安装包消费者通常不需要任何 `HYLE_*` 构建 option。

## 三种 C++ 交付形态

### 源码组装

获得授权源码的使用者通过 `add_subdirectory()` 组装所需组件，并链接 `hyle::<module>`。父工程拥有生成器、工具链、输出目录和标准 CMake 策略；Hyle option 只描述 Hyle 自身。嵌入模式不创建源码发行 target。

请求 `core` 时唯一的产品组件 target 是 `hyle::core` INTERFACE target；构建树仍有无 artifact 的
内部 build/trace/testing foundation targets，但 Core 不链接 trace/testing，也不产生
object/archive/runtime。请求其他组件会通过中央图取得它所需的 Core source contract，不能靠父工程
的 include path 偶然可见。

### 静态 SDK

以 `HYLE_INSTALL=ON`、`HYLE_STATIC=ON` 安装后，消费者使用：

```cmake
find_package(Hyle CONFIG REQUIRED COMPONENTS error mem)
target_link_libraries(app PRIVATE hyle::mem)
```

导出 target 必须传播静态链接所需的私有闭包、系统库和已登记的 `find_dependency()`。
若只安装/消费 `core`，package 只提供 headers 与 imported INTERFACE target，不存在 Core archive。

### 共享 SDK

以 `HYLE_INSTALL=ON`、`HYLE_STATIC=OFF` 安装，消费语法相同。Unix 安装 target 使用相对同级 RPATH；Windows 消费 target 可调用 `hyle_stage_runtime_dependencies(target)` 把所需 DLL 放到运行目录。

安装前缀和 GNUInstallDirs 必须保持相对、可搬迁；绝对安装子目录会失败。由于现有公开头存在跨组件引用，SDK 目前安装整个 `include/hyle`，可用组件以导入 target 和 `Hyle_AVAILABLE_COMPONENTS` 为准。源码授权与静态/共享产物形态是两个独立维度。

Core-only shared SDK 同样没有 Core DSO/DLL 和 runtime staging；`HYLE_STATIC` 只影响请求到的编译型
模块。Core helper 已内联进 consumer，header 变化后必须重编，不能靠替换其他 shared artifact
更新。

## 源码包与离线构建

顶层 Git checkout 提供 `hyle_source_package`：它审计全部 in-tree/integrated dependency，拒绝 dirty 或 untracked 工作树，以 `git archive` 打包当前提交，并生成源码清单和 SHA256。产物包含已跟踪源码与法律材料，不包含构建缓存或 FetchContent checkout。

离线依赖包是独立输入。准备与锁文件匹配的 `deps/<lock-name>/` 后使用：

```powershell
cmake -S . -B build-offline `
  -C hyle-offline-deps.cmake `
  -DHYLE_COMPONENTS=all
```

离线初始化脚本只接受精确、干净的 Git checkout 或能通过展开树与许可证校验的官方归档；缺少被选组件所需依赖时立即失败。`HYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES` 只用于明确承担风险的异常恢复，不是发布流程。

## CI 分层与持续回归

CI 不把所有平台、后端和性能门塞进一次任务：

| 层级 | 当前职责 |
|---|---|
| Windows/Linux/macOS 基线 | all-component CPU 配置、构建和测试；Linux 同时检查文档合同；昂贵可选后端关闭 |
| Linux 合同与交付 | 组件图检查、production shape、安装包消费者、完整源码包 |
| Windows clang-cl foundation | 一组底层组件及多个 exact-SIMD artifact |
| Linux AArch64 cross | 全组件交叉编译；不宣称目标机运行正确性 |
| Locked feature/package | 锁定归档、可选适配器、static/shared SDK 和 trace smoke |
| 手工/发布证据 | sanitizer、真实 TLS/CUDA/Vulkan 设备、SVE/RVV 运行和性能认证 |

基线遍历 `HYLE_KNOWN_COMPONENTS`，所以新组件自动进入 all-component build；开启测试时，缺少模块测试目录会机械失败。package smoke 自动发现安装消费样例。更严格的 production roots 仍是显式列表，只能在对应模块通过收口后扩展。CI 绿灯只覆盖该 job 明示的形状，不能外推硬件、性能或未运行配置。

### 模块持续回归

模块审查后的每次依赖、公开头或 option 改动，都应至少重跑：

1. 本模块及其传递依赖的 source build/test。
2. 受影响的 static/shared package consumer。
3. 涉及外部依赖时的锁、离线和许可证投影。
4. 涉及 ABI 或运行时装载时的 relocated consumer。
5. 涉及性能时的独立、可复现性能门；普通 CI 不替代性能证据。

## 新模块接入协议

1. 在 `HyleComponents.cmake` 登记组件、构造顺序和全部内部 `PUBLIC`/`PRIVATE` 边。
2. 用 `hyle_add_module()` 创建 `src/<module>` target，只列本模块源码和外部实现 target。
3. 增加 `tests/<module>/CMakeLists.txt` 及模块聚合测试；需要时再增加 benchmark。
4. 若引入外部源码，先更新 dependency lock，再在模块包登记中声明唯一 owner、feature、导出 target、资产和许可证投影。
5. 增加 C++ 源码消费和 static/shared 安装消费样例；确认 relocation。
6. 让 all-component、依赖图、离线、源码包和适用的 feature CI 失败一次，再修到通过，证明门确实覆盖新模块。
7. 更新 architecture、user guide、task/evidence；构建通过不等于模块收口。

## 已知遗留

- `include/hyle/c/**`、相应 `c_*.cc`、C surface/export fixtures、package C/FFI fixtures，以及 managed 样例和脚本，是逐模块待删除或重新分类的历史债；CI 暂时编译它们不构成 C 或跨语言产品承诺。
- 根工程仍启用 `LANGUAGES C CXX`，因为 vendored C、`.c` 实现和汇编桥接仍需要 C 编译器；这不代表存在 C 产品 API。
- `SOURCE-MANIFEST.txt`、部分 CMake 注释和组件依赖理由仍有旧 C ABI 叙事，需随模块复审清理。
- 历史 `hyle/macros.h`/compiler-definition target-neutral 例外已经删除；后续新 header-only 公共
  能力必须有明确 owner target 和真实依赖边。公开头扫描当前只机械覆盖 Core contract，其他模块
  的同类自动化仍应随各模块 v2 review 扩展。
- production-shape 根列表仍含已重新打开审查的模块；它是回归集合，不是“已冻结”声明。
- 可选特性语义尚未统一：TLS/CUDA 使用 `AUTO/ON/OFF`，其他后端仍是布尔值；应由所属模块复审后统一。
- 托管 CI 尚未覆盖 sanitizer、真实加速设备及 SVE/RVV 运行；Windows/macOS 的完整交付矩阵也不是每个 PR 都运行。
- 安装当前复制整个公开头树；模块边界稳定后才可评估 component-filtered headers。
- 源码包有意不携带 FetchContent 依赖；需要全离线构建时必须同时准备独立离线依赖目录。
- `compute` 与 `graph` 仍是未完成模块。all-component 绿灯只证明现有构建与测试契约，不证明其功能已经完成。
