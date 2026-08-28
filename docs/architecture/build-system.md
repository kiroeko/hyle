# 构建与组装架构

> 状态：当前设计。本文描述仓库中实际生效的 CMake 契约，而不是发布承诺。
> Hyle 当前交付面是 C++：源码组装、静态 SDK、共享 SDK。早期公开 C/跨语言表面已经移除，
> 不属于当前产品设计。

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
- `core`、`util_core`、`container_intrusive` 与 `math` 是可选择、可安装的 `INTERFACE`
  组件；其余组件
  通常是编译型 target。
- `hyle_build_contract`、`hyle_testing`、`hyle_trace` 是横切构建 target，不是产品组件根。
- 横切 target 也必须遵守真实消费边：模块只有在源码或公开头读取 trace/testing gate 时才传播
  对应 target，并通过 `TRACE`/`TESTING` 显式 opt-in；配置会把声明与当前 selected sources/公开头
  反向对账。纯编译/布局地基 `core` 不声明这两个 gate，不能因全局开关无故携带 profiler runtime
  或 test-only ABI definition。
- Tracy 实现在模块图声明后才建立；`HYLE_TRACE=ON` 而所选闭包没有 trace consumer 会配置失败，
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
`INTERFACE_LINK_LIBRARIES`，防止后续手工链接绕过 manifest。组件 schema、名称与拓扑始终按完整
manifest 做轻量元数据验证，依赖闭包按已验证的拓扑单次反向传播；随后只对本次 selected closure
读取公开头和实现源码。这样 all-component CI 仍审完整树，而单模块源码消费者不会被另一个未选模块
的施工中状态阻断，include audit 成本也从全库 `S_all` 收束到 `S_selected`。

当前图的关键关系是：

```text
core        leaf
math        -> PUBLIC core
container_intrusive -> PUBLIC core
util_core   -> PUBLIC core
error       -> PUBLIC core
obs         -> PUBLIC core error
time        -> PUBLIC core container_intrusive
platform    -> PUBLIC core error time
mem         -> PUBLIC core error platform util_core
simd        -> PUBLIC core platform
container   -> PUBLIC core mem; PRIVATE platform; BUILD_PRIVATE simd_vec_headers
util        -> PUBLIC core error mem container
sched       -> PUBLIC core error platform mem container util_core; PRIVATE time
io          -> PUBLIC core error mem container time sched
compute     -> PUBLIC core error mem sched; PRIVATE platform simd
graph       -> PUBLIC core error mem container sched; PRIVATE util_core
```

`PUBLIC` 表示消费者编译或链接本模块时也需要该产品依赖；`PRIVATE` 表示只服务本模块实现。
`BUILD_PRIVATE` 是有真实 `$<BUILD_INTERFACE:...>` target 边、但不扩大安装闭包的内部 header seam；
当前 container 的 per-ISA TU 只借 `simd_vec_headers`，不会因此构建/交付完整 simd runtime。这里记录的
都是实际 target/link 契约。`container_intrusive` 证明 header-only 能力也应在公开类型穿过模块边界
时拥有真实 target：`time` 暴露 `TimerList`，故不能把这条边藏成 include-path 偶然可见；完整
`container` 不聚合这项独立能力。`platform/fs.h` 公开暴露 `time::UnixTimestamp`，所以
`platform -> PUBLIC time` 是真实 SDK 边；sched 的 waiter 链因 detached wake-link 复用而保留私有特化，
不伪造一条未直接使用的依赖。io、compute 与 graph 的当前公开面都暴露 sched 类型，因此 sched
是无条件 `PUBLIC` 边；无 fcontext 平台会诚实拒绝这三个产品根。未来若要交付 raw proactor 或
host-only compute，必须先拆成拥有独立公开头与 journey 的真实组件，不能只从图上删边。

Core 的 source contract 不再是 target-neutral 例外。公开头直接 include `<hyle/core/**>` 或
`<hyle/version.h>` 的组件必须直接声明 `PUBLIC core`，即使 `hyle::core` 是 `INTERFACE`、没有
binary symbol。配置阶段会扫描 owned public headers 与实现源码并同中央 manifest 对账：公开
include 必须有直接 `PUBLIC` 边，实现 include 必须有直接 `PUBLIC/PRIVATE` 边；只通过 forward
declaration 暴露外部类型的边用显式 `PUBLIC_TYPE_DEPS` 注释；不进入产品闭包的实现头借用必须匹配
`PRIVATE_BUILD_DEPS` seam。漏边、多余公开边和未注释 type/build edge 都失败。
`container_intrusive`/`util_core` 这样的 header-only target 同样遵守这条规则；共享源码
include root 或复制整棵 include tree 都不能使未登记依赖合法。

该 include-owner 检查只裁决正在组装的 selected closure，不把未选模块重新耦合进 source journey。
选择某组件时，它和全部传递依赖都会被检查；CI 的 `HYLE_COMPONENTS=all` 因而仍机械覆盖每个组件。
组件 manifest 自身的 unknown edge、重复、PUBLIC/PRIVATE 冲突和非拓扑顺序无论选择什么 root 都会
失败，因为这些属于轻量全局 schema，而不是未选源码内容。

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
| 验证 | `HYLE_BUILD_TESTS`, `HYLE_TEST_INSTRUMENTATION`, `HYLE_WERROR`, `HYLE_BUILD_BENCH` | 分别控制测试、测试插桩、警告门和 benchmark；普通测试默认 shipping layout，诊断插桩须显式开启 |
| Sanitizer | `HYLE_SANITIZER` | `off|asan|tsan|ubsan`；非法值配置失败 |
| SIMD | `HYLE_SIMD_EXACT_TARGET` | 在 fresh cache 中只在 simd 被选时声明；空值或 `SCALAR/SSE4/AVX2/AVX512/NEON/SVE/RVV` |
| 自动特性 | `HYLE_IO_TLS`, `HYLE_COMPUTE_CUDA` | 在 fresh cache 中只由所选 owner 组件声明；类型为 `AUTO|ON|OFF`，当前默认 `OFF`；`AUTO` 探测、`ON` 必须存在、`OFF` 不探测 |
| 布尔特性 | `HYLE_IO_HTTP2`, `HYLE_COMPUTE_GGML`, `HYLE_COMPUTE_VULKAN` | 在 fresh cache 中只由所选 owner 组件声明，昂贵能力默认 `OFF`；`HYLE_COMPUTE_GGML_NATIVE` 仅在 ggml 开启时声明 |
| Trace | `HYLE_TRACE`, `HYLE_TRACY_DIR` | 在 fresh cache 中仅闭包有 trace consumer 时声明，默认 `OFF`；路径只在开启时声明。当前只有 Tracy，已删除无选择意义的 `HYLE_TRACE_BACKEND` |

Sanitizer 是整个所选 target 图的构建形状，不是只给测试可执行文件加一个编译参数。对象库和静态库
携带插桩；最终 executable/shared/module target 负责链接相应 runtime。Windows 的 cl.exe 与
clang-cl 目前只支持 `asan`：HYLE 显式链接编译器自带的 dynamic-CRT runtime 和 thunk、保留
SEH interceptor 与调试信息，并把匹配的 ASan DLL 放到最终产物旁边；请求 `tsan`/`ubsan` 会直接
配置失败并要求转到 GCC/Clang 环境。HYLE 自己取得的 GoogleTest 会使用同一 ASan 形状，嵌入父工程
提供的 `GTest::gtest_main` 则由父工程保证 ABI/插桩一致。

顶层配置省略 `HYLE_COMPONENTS` 时默认 `all`；作为子目录嵌入时，父工程必须在
`add_subdirectory()` 前显式设置根组件，防止一次普通依赖引入意外构建整库。在独占的 fresh build
tree 中，测试插桩、SIMD exact、IO/compute feature 和 trace 路径等 cache 控制只在其 owner/gate
成立时声明。CMake 会保留旧配置曾写入的 cache key；改变组件闭包或产品形状时应使用新的独占
scratch，或明确 `--fresh` 后重新配置，不能把旧 key 仍可见误认为当前闭包消费了该能力。生成器、工具链、配置类型、安装前缀、CUDA architectures 和
`FETCHCONTENT_SOURCE_DIR_*` 使用标准 CMake 变量。正式证据与发布配置应把会改变产物的 `AUTO`
固定为 `ON` 或 `OFF`。安装包消费者通常不需要任何 `HYLE_*` 构建 option。

## 三种 C++ 交付形态

### 源码组装

获得授权源码的使用者在 `add_subdirectory()` 前显式设置 `HYLE_COMPONENTS`，组装所需组件并链接
`hyle::<module>`。只有 Hyle 自身作为顶层工程时，省略该变量才表示 `all`。父工程拥有生成器、
工具链、输出目录和标准 CMake 策略；Hyle option 只描述 Hyle 自身。嵌入模式不创建源码发行 target。

根 `project()` 只声明 `LANGUAGES CXX`。C 是内部实现语言而非产品消费者前置条件；在
`HYLE_BUILD_TESTS=OFF` 的产品/source 形状中，当前只有所选闭包含 mem/mimalloc 时才
`enable_language(C)`。测试构建还会进入 GoogleTest 自己的 CMake project，并可能探测 C frontend；
这属于测试工具链需求，不建立 C 产品 surface。

请求 `core` 时唯一的产品组件 target 是 `hyle::core` INTERFACE target；构建树仍有无 artifact 的
内部 build/trace/testing foundation targets，但 Core 不链接 trace/testing，也不产生
object/archive/runtime。请求其他组件会通过中央图取得它所需的 Core source contract，不能靠父工程
的 include path 偶然可见。

### 静态 SDK

以 `HYLE_INSTALL=ON`、`HYLE_STATIC=ON` 安装后，消费者使用：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS error mem)
target_link_libraries(app PRIVATE hyle::mem)
```

导出 target 必须传播静态链接所需的私有闭包、系统库和已登记的 `find_dependency()`。
若只安装/消费 `core`，package 只提供 headers 与 imported INTERFACE target，不存在 Core archive。

### 共享 SDK

以 `HYLE_INSTALL=ON`、`HYLE_STATIC=OFF` 安装，消费语法相同。Unix 安装 target 使用相对同级 RPATH；Windows 消费 target 可调用 `hyle_stage_runtime_dependencies(target)` 把所需 DLL 放到运行目录。

安装前缀和 GNUInstallDirs 必须保持相对、可搬迁；绝对安装子目录会失败。SDK 只安装所选组件闭包
拥有的公开头；只有闭包含 `TRACE` consumer 时才额外安装 trace foundation 的 `obs/trace.h`，不会
复制整棵 `include/hyle`。
公开头 include-owner 审计要求每个跨组件 include 都有对应直接 `PUBLIC` 边，防止缺失 target 被整树
安装偶然遮蔽。可用组件仍以导入 target 和 `Hyle_AVAILABLE_COMPONENTS` 为准。源码授权与静态/共享
产物形态是两个独立维度。

预发布 `0.y.z` 的 package version 和 shared-object loader/SOVERSION identity 都使用完整版本并要求
`ExactVersion`；1.0 后才切换为 conventional major compatibility。匹配版本的 C++ headers、artifact、
工具链和配置始终是同一交付合同。

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
| PR / main push | 只跑 Linux all-component shipping-layout CPU 基线、文档和组件/依赖合同；昂贵可选后端关闭 |
| 手工 full dispatch | Linux/Windows/macOS CPU 基线，并在适用主机跑 production shape、package consumer 与源码包 |
| 手工 Windows clang-cl | 显式 test instrumentation 的一组底层组件及多个 exact-SIMD artifact |
| 手工 Linux AArch64 cross | 全组件交叉编译；不宣称目标机运行正确性 |
| 手工 locked feature/package | 锁定归档、可选适配器、static/shared SDK 和 trace smoke |
| 手工/发布证据 | sanitizer、真实 TLS/CUDA/Vulkan 设备、SVE/RVV 运行和性能认证 |

基线遍历 `HYLE_KNOWN_COMPONENTS`，所以新组件自动进入 all-component build；开启测试时，缺少模块
测试目录会机械失败。PR 与 main push 都故意只保留高信号 Linux 门以控制额度；Windows、macOS、
交叉编译、完整交付和 locked feature 由 `workflow_dispatch` 展开。package smoke 自动发现安装消费
样例。更严格的 production roots 仍是显式列表，只能在对应模块通过收口后扩展。CI 绿灯只覆盖
该 job 明示的形状，不能外推硬件、性能或未运行配置。

### 模块持续回归

模块审查后的每次依赖、公开头或 option 改动，都应至少重跑：

1. 本模块及其传递依赖的 source build/test。
2. 用独占 scratch 跑该根的 `HyleCiSelectedComponentSmoke.cmake`：source embedding、shared、static
   及搬迁后 installed consumer；Windows 可用 `hyle-windows.ps1 -Action SelectedComponentSmoke`。
   smoke 生成的 Darwin 父工程从中央 platform requirement 取得 deployment target；不得让 source 与
   SDK 两种形状偶然使用不同的 macOS loader floor。
3. 涉及外部依赖时的锁、离线和许可证投影。
4. 涉及 ABI 或运行时装载时的 relocated consumer。
5. 涉及性能时的独立、可复现性能门；普通 CI 不替代性能证据。

并行 agent/session 不得复用 preset 的固定 build tree，也不得在另一会话的 cache 上重配。Windows
wrapper 的 `-BinaryDir <absolute-session-scratch>` 为 Configure/Build/Test 提供独占目录；配置单个
根时同时传 `-Component <name>`。临时 scratch 应位于明确的会话目录，完成后由拥有它的会话清理。

## 新模块接入协议

1. 在 `HyleComponents.cmake` 登记组件、构造顺序和全部产品 `PUBLIC`/`PRIVATE` 边；若仅借用另一
   模块的实现 header 且不形成交付需求，登记受审的 `PRIVATE_BUILD_DEPS` seam，不得伪造成产品边。
2. 用 `hyle_add_module()` 创建 `src/<module>` target，只列本模块源码和外部实现 target。
3. 增加 `tests/<module>/CMakeLists.txt` 及模块聚合测试；需要时再增加 benchmark。
4. 若引入外部源码，先更新 dependency lock，再在模块包登记中声明唯一 owner、feature、导出 target、资产和许可证投影。
5. 增加 C++ 源码消费和 static/shared 安装消费样例；确认 relocation。
6. 运行 all-component、依赖图、离线、源码包和适用的 feature 合同并修到通过；新增合同测试时，
   用隔离的负向 fixture 证明它能拒绝缺边/缺资产，不能故意破坏产品树或共享 CI 来“失败一次”。
7. 更新 architecture、user guide、task/evidence；构建通过不等于模块收口。

## 边界与已知遗留

- 当前公开头、模块 target、安装包 consumer 和样例只定义 C++ 产品面；早期 `include/hyle/c/**`、
  公开 `c_*.cc` bridge、C surface/export fixture 与 managed FFI probe 已移除。没有通过
  [Native FFI 准入](native-ffi-policy.md)前不得恢复第二套镜像 API。
- 根工程只启用 `LANGUAGES CXX`；所选闭包需要 mem/mimalloc 时才按需启用内部 C。内部
  `extern "C"`、C 编译单元或汇编桥接不代表存在 C 产品 API。
- 历史 `hyle/macros.h`/compiler-definition target-neutral 例外已经删除；后续新 header-only 公共
  能力必须有明确 owner target 和真实依赖边。配置阶段对 selected closure 的全部 owned public
  headers 做 include-owner / direct-`PUBLIC` 对账；`obs/trace.h` 是仅随 `TRACE` consumer 闭包交付的
  明确 foundation seam。
- production-shape 根列表仍含已重新打开审查的模块；它是回归集合，不是“已冻结”声明。
- 可选特性都由 owner 模块声明，昂贵能力默认 OFF；TLS/CUDA 保留 `AUTO|ON|OFF` 的显式探测语义，
  其他后端为布尔值。没有第二个 trace backend 时不恢复 backend selector。
- 托管 CI 尚未覆盖 sanitizer、真实加速设备及 SVE/RVV 运行；Windows/macOS 只在手工
  `workflow_dispatch` 运行，不属于 PR/main push 门。
- 安装只复制 selected closure 的 owned public headers；缺少 owner 或 `PUBLIC` 边会在配置阶段失败。
- 源码包有意不携带 FetchContent 依赖；需要全离线构建时必须同时准备独立离线依赖目录。
- `compute` 与 `graph` 仍是未完成模块。all-component 绿灯只证明现有构建与测试契约，不证明其功能已经完成。
