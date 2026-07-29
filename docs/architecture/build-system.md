# 构建与组装架构

## 定位

HYLE 的 CMake target 图是项目组织架构的可执行版本，不只是“把当前仓库编过”的脚本。
源码、公开头、C ABI、直接依赖、可选后端和验证面都必须归属到 target；文档解释怎样选择，
不能用目录顺序、全局变量或维护者记忆补全隐藏依赖。

## 设计目标与不可变原则

构建系统同时服务仓库维护者、源码嵌入方、二进制 SDK 使用者和经授权的源码客户。四类人入口不同，
但必须落到同一张产品 target 图；不能为某一种交付形状重建第二套源文件、依赖或 feature 清单。

1. **target 是事实，文件名不是接口。**公开能力统一由 `hyle::<module>` 表达；头路径、编译合同、
   直接依赖、私有 link-only archive 和 runtime staging 随 target 传播。用户不得靠手写库文件顺序
   补全构建系统遗漏的信息。
2. **component 表达用户意图。**`HYLE_COMPONENTS` 的值是用户想取得的产品根，不是要编译的目录清单；
   构建系统负责计算闭包，并只激活闭包内源码、测试、benchmark 和第三方发现。
3. **模块拥有实现，manifest 拥有产品图。**模块自己的源码、外部实现依赖、测试和 benchmark 归模块
   目录；声明式 component manifest 唯一拥有内部 HYLE 边及可见性。根 CMake 只维护工具链合同、
   横切 helper 和最终组装，不回到巨型源文件清单。
4. **选项描述产品决定，不暴露脚本偶然。**公开 cache 变量回答“要什么组件、能力、交付物或验证”，
   不回答“某段内部 CMake 是否执行”。标准 CMake 已有准确语义时直接使用标准变量。
5. **父工程拥有父工程。**source embed 不改写父工程的 `BUILD_SHARED_LIBS`、generator、toolchain、
   configuration types、输出目录、安装根或发布 target；HYLE 的局部决定必须落在自己的 target 上。
6. **可复现优先于便利探测。**`AUTO` 可以帮助本机开发，正式 SDK/CI 必须把会改变产品能力的选项
   固定成 `ON` 或 `OFF`，并把结果写入 package feature manifest。
7. **错误配置立即失败。**未知/空 component、非法枚举、缺少硬能力、测试布局进入安装包、未声明
   link 边和不可迁移安装路径都在 configure 阶段拒绝；不允许静默退回更小或更大的产品。
8. **授权与 artifact 正交。**shared、static、complete source 只决定文件闭包和消费入口，不自行扩大
   使用、修改、披露或再分发权利。

这些原则比当前 helper 名称稳定。未来可以重写 CMake 实现，但只要仍满足同一组用户合同，就不属于
产品架构变化；反之，即使只是增加一个布尔开关，只要破坏上述合同也必须按架构变更审查。

## 选项的信息模型

公开选项按“它描述什么”分层，而不是按最初在哪个 CMake 文件出现分类：

| 信息层 | 命名与值域 | 例子 | 约束 |
|---|---|---|---|
| 产品根 | `HYLE_COMPONENTS=<list>` | `mem;simd` | 依赖自动闭包；显式空/未知值失败 |
| artifact 形状 | `HYLE_STATIC`、`HYLE_INSTALL` | static installed SDK | 不借用或改写父工程 `BUILD_SHARED_LIBS` |
| 产品能力 | `HYLE_<MODULE>_<FEATURE>` | `HYLE_IO_TLS`、`HYLE_COMPUTE_CUDA` | 可探测能力用 `AUTO/ON/OFF`；ON 是硬合同 |
| 硬件特化 | HYLE 语义或标准 toolchain 变量 | `HYLE_SIMD_EXACT_TARGET`、`CMAKE_CUDA_ARCHITECTURES` | 进入 artifact/ISA 合同，分发包不得偷用本机 native |
| 验证形状 | `HYLE_BUILD_*`、`HYLE_TEST_*`、`HYLE_SANITIZER` | tests、bench、shipping layout、TSan | 不改变正常产品 API；测试插桩不得进入 install |
| 对标开发 | `HYLE_BENCH_*` | `HYLE_BENCH_HIGHWAY` | 只给 benchmark 加对手，永不改变产品 target |
| 构建环境 | 标准 CMake/toolchain/FetchContent 变量 | `CMAKE_BUILD_TYPE`、`FETCHCONTENT_SOURCE_DIR_GGML` | 不再包一层同义 HYLE 开关 |

新增 cache 变量前必须回答以下问题：

- 它是否是用户或制包者需要选择的稳定决定？如果只是函数之间传状态，使用普通局部变量或 target
  property；计算结果采用内部 `*_ENABLED`，不得再暴露为第二个可写 cache 开关。
- 它是否改变安装包能力、ABI、ISA 或依赖闭包？若改变，必须可被 package manifest 审计，并有
  source-embed/install smoke；不能只在 configure 日志打印一行。
- 它是否仅属于一个模块？使用 `HYLE_<MODULE>_<FEATURE>`，并只在 component 闭包含该模块时探测或
 取得依赖。跨模块产品决定才允许根级 `HYLE_<NAME>`。
- 它是否存在三种真实状态？只有“便利探测/硬要求/明确禁止”才用 `AUTO/ON/OFF`；普通二元能力使用
  `ON/OFF`，有限集合使用严格校验的 cache string，拼写错误不得退回默认值。
- CMake 是否已有同一语义？build type、install prefix、CUDA architectures、toolchain file 和
  FetchContent source override 均使用标准名字，不创造 HYLE 方言。

因此“选项比较多”本身不是失败；让普通用户必须理解无关选项、同一决定出现两个名字、或一个选项在
不同入口表达不同产品，才是失败。安装包消费者通常不设置任何 HYLE 构建选项；源码用户通常只设置
component、linkage、tests，再为实际使用的可选后端补一个明确能力合同。

授权方式与交付形状正交。HYLE 当前是 proprietary，不代表客户只能拿二进制；经单独书面授权的
客户可以取得完整源码。构建系统把交付形状明确分为三类：

| 交付形状 | 构建入口 | 内容与用途 |
|---|---|---|
| headers + shared libraries | `HYLE_INSTALL=ON; HYLE_STATIC=OFF` | 动态 SDK；安装 target、运行库、头、manifest 和 license closure |
| headers + static libraries | `HYLE_INSTALL=ON; HYLE_STATIC=ON` | 静态 SDK；额外传播 link-only 私有 archive/系统依赖 |
| complete source | 顶层构建的 `hyle_source_package` | 干净 Git commit 的完整 tracked tree；客户用同一 component/feature 图自行组装 |

三者改变的是 artifact closure，不改变许可证。源码包不能从工作目录盲拷贝：target 遇到 tracked
修改或未跟踪文件会拒绝，产物带 export-substituted commit manifest 与 SHA-256。
HYLE 被父工程 `add_subdirectory()` 嵌入时不创建源码分发 target，避免一个依赖暗中接管父工程打包。

## 产品组件

每个二进制模块有 build-tree target `hyle_<module>` 和消费别名 `hyle::<module>`。用户通过
`HYLE_COMPONENTS` 给出所需的产品根，CMake 计算依赖闭包，只把闭包内模块加入构建：

```text
roots: core util_core error obs math platform time mem simd container util sched io compute graph

util_core-> (header-only leaf)
obs      -> error
platform -> error
time     -> error
mem      -> error + platform + util_core
simd     -> error + platform
container-> error + mem + platform
util     -> container + util_core
sched    -> error + platform + time + mem + container + util_core
io       -> error + mem + container [+ sched]
compute  -> error + platform + mem + simd [+ sched]
graph    -> error + mem + container + util_core [+ sched]
```

箭头右侧是 component 解析闭包；`util_core` 虽无 binary artifact，仍是可选择、可安装并被真实
target link 表达的 INTERFACE component，不能从图里省略。只提供 OS/架构/编译器宏的
`hyle/platform.h` 与 `hyle_trace` 横切 gate 不作为产品 root。包含 `hyle/platform.h` 只是零链接代价的编译环境
依赖，不等于链接 `hyle::platform`；例如 time 的 native clock 实现用它选择 OS 分支，但 time 的
二进制闭包仍然只有 error。未设置 `HYLE_COMPONENTS` 时默认 `all`；逗号与分号都可分隔多个根。
显式空值会失败，避免丢失的 shell/CI 变量意外触发全库与重型依赖。父工程在
`add_subdirectory()` 前设置普通变量时，CMake cache 会镜像同一有效值，GUI、问题报告与实际 target
图不会分别显示 `all` 和子集。未知组件同样在 configure 阶段直接失败，配置日志打印
requested/resolved 清单，因此一次构建“组装了什么”可审计。
方括号内的 sched 边只在 HYLE 已接入 fcontext 的目标组合上存在（当前为 x86-64
Windows/Linux/macOS 与 arm64 Linux/macOS）；其他平台仍可组装 io proactor、compute host/backend
基础层和 graph serial core。显式请求不受支持的 sched 会在 configure 阶段失败，不会静默降级。

组件选择同时控制依赖发现：不选 `io` 就不探测 OpenSSL、不取得 nghttp2；不选 `compute`
就不探测 CUDA，也不取得 ggml/Vulkan-Headers/volk/glslang。根构建把 `HYLE_DEPS_DIR` 一次投影为
HYLE 目录作用域内的 `FETCHCONTENT_BASE_DIR`，因此产品依赖和 test-only GoogleTest 都写入同一
位置，又不会改写 source-embedding 父工程的 cache。默认位置是当前 HYLE build tree 的 `_deps`；
源码与生成物一起归该 build tree 所有，不同 toolchain/build tree 不共享这个目录。`_deps` 是
FetchContent 的 population 实现区，不是可交付的 offline bundle 格式。离线或
经审查的 bundle 保存独立依赖源码，并通过标准 `FETCHCONTENT_SOURCE_DIR_<NAME>` 逐项注入；依赖
生成物仍进入本 build tree，不能写入或依赖源码树。仅预填一个仓库根 `.deps`、或只打开
`FETCHCONTENT_FULLY_DISCONNECTED`，都不是首次配置的稳定合同。源码根的
`hyle-offline-deps.cmake` 可作为 initial cache 扫描 bundle 的
`deps/<FetchContent-name>/`，自动生成标准 override；它不维护另一份依赖名字清单。

TLS 与 CUDA 使用 `AUTO|ON|OFF` 三态：AUTO 仅供便利探测，ON 是硬能力契约，OFF 不运行依赖探测。
正式 SDK/CI 必须显式选择 ON/OFF，因此构建机偶然安装了某个 SDK 不会改变发布形状。

## 模块所有权与并行施工

一个产品模块的构建定义不应散落在中央巨型清单中。现有模块按以下固定位置拥有自己的三块接线：

| 内容 | 模块所有位置 | 中央文件只负责 |
|---|---|---|
| 内部 HYLE 直接依赖及 PUBLIC/PRIVATE | `cmake/HyleComponents.cmake` 的声明式 manifest | 同一声明同时生成 component 闭包和真实 target link |
| 产品源码、外部实现依赖 | `src/<module>/CMakeLists.txt` | `hyle_add_module()` 按 manifest 接入内部边；根只按闭包加入模块 |
| 重型可选能力发现/取得 | `cmake/Hyle<Module>Dependencies.cmake` | 根只在解析闭包包含该模块时调用其配置入口 |
| 正确性、C ABI、header/surface 测试 | `tests/<module>/CMakeLists.txt` | `tests/CMakeLists.txt` 提供 GoogleTest、pure-C/surface/header helper 和 aggregate |
| 正常 workload 与 SOTA 对拍 | `bench/<module>/CMakeLists.txt` | `bench/CMakeLists.txt` 提供 nanobench、aggregate 和逐模块装载 |

测试片段是必需所有权：选中的产品组件若没有 `tests/<module>/CMakeLists.txt`，configure 直接失败，
不能静默漏测。benchmark 片段可以不存在，因为不是每个基础模块都有独立可测 workload；存在时必须把
每个 executable 显式登记到 `hyle_<module>_benches`。`bench/peers/` 只配置多个模块共享的外部
对手，target 接线仍归消费模块；真正横跨多个产品且不能归一方所有的 workload 放
`bench/integration/`，登记到独立的 `hyle_integration_benches`，不冒充任一产品模块。中央入口用
`add_subdirectory()` 装载这些所有权目录，
因此每个模块有独立的 source、binary 和普通变量作用域；target 名、CTest 名与统一 runtime 输出目录
保持稳定。模块生成文件默认归自己的 binary 子目录，测试默认工作目录也随模块目录走；需要跨目录的
源码、生成物或工作目录时必须显式声明，不能依赖兄弟片段泄漏。共享 benchmark peer 应在父目录创建
普通 target；未来若改用 `find_package()` 的非 `GLOBAL` imported target，也必须在共同父目录发现或
显式提升可见性，不能让 sibling 偶然可见。

`hyle_add_module()` 自动读取 manifest 并建立内部 target link；模块 CMakeLists 不再重复列出内部
`DEPS/PUBLIC_DEPS`。所有模块目录返回后仍会读取 target 的 `LINK_LIBRARIES`/
`INTERFACE_LINK_LIBRARIES` 做终态审计，因此声明之后手写
`target_link_libraries(hyle_x ... hyle_y)` 也不能绕过组件闭包。外部实现库不进入 HYLE-to-HYLE 图，
仍由所属模块负责发现、链接与 package/export 语义。

### 一个依赖 manifest，两种自动投影

内部依赖只声明一次，但要在两个 CMake 阶段发挥作用：

| manifest 投影 | 发生时间 | 生成结果 |
|---|---|---|
| component closure | 任何模块 target 创建前 | 用户请求 `mem` 时解析加载集合和拓扑顺序 |
| target link | `hyle_add_module(mem ...)` 创建 target 时 | 生成真实 PUBLIC/PRIVATE linker edge 与 usage requirement |

例如 mem 的完整内部依赖事实只有：

```cmake
_hyle_declare_component(mem PUBLIC_DEPS error platform)
```

模块所有权目录只声明实现：

```cmake
hyle_add_module(mem SOURCES system.cc stats.cc c_mem.cc)
```

manifest 中 PUBLIC/PRIVATE 的并集让 `HYLE_COMPONENTS=mem` 在进入模块目录前解析成
`error;platform;mem`；同一份可见性信息随后让 helper 自动链接 `hyle_error`、`hyle_platform`。这样既
不会面对尚未创建的 target 反向猜闭包，也不把源码、外部库和条件实现清单搬回中央脚本。

终态审计不是第二份声明：它从生成后的 target properties 反查结果是否仍等于 manifest，用来捕获模块
创建后偷偷追加的内部边。mimalloc、OpenSSL、ggml 等非 HYLE 实现依赖只属于模块 target，不伪装成
用户 component。若将来需要改变内部依赖，只改 manifest 一处；模块 CMakeLists 无同步工作。

### 外部依赖一次登记，多种 package 投影

外部实现依赖的取得、编译和真实链接仍归消费模块，因为这些动作必须发生在 target 创建前后；但同一
依赖不再到中央 `HylePackage.cmake` 重写一遍名字分支。模块在确认依赖 target/资产后调用一次
`hyle_register_package_dependency()`，登记 owner、可导出 archive、package feature、静态下游
`find_dependency()`、公开头/工具和许可证。package 层只遍历 registry，并自动投影为：

| 登记事实 | 自动生成的交付结果 |
|---|---|
| `EXPORT_TARGETS` / `STATIC_EXPORT_TARGETS` | shared/static 对应的 `HyleTargets.cmake` 闭包 |
| `STATIC_FIND_DEPENDENCIES` | 静态包 `HyleConfig.cmake` 的下游 SDK 发现 |
| `FEATURES` | `Hyle_AVAILABLE_FEATURES` |
| runtime/development asset specs | 所选实现对应的 license、header 和 tool 安装规则 |

registry 拒绝重复名字、未知 owner、缺失 export target、非法下游依赖格式、缺失/绝对安装资产路径。
因此增加一种外部实现只修改模块所有权位置；中央 package 文件不知道 mimalloc、nghttp2、ggml、volk、
OpenSSL、CUDA 或 Tracy 的名字。`THIRD_PARTY_NOTICES.md` 仍是需要人工审查的法律说明，不拿机器 registry
替代许可证判断；但实际编入 package 的文件闭包由同一登记机械生成。

### 新增模块或能力的固定协议

新增产品模块不是“在根目录多写一个 `add_library`”。维护者按以下顺序扩展：

1. 在 `HyleComponents.cmake` 用一次声明登记稳定 component 名、PUBLIC/PRIVATE 内部直接依赖，保持无环
   拓扑；manifest 同时生成加载前闭包和加载后的真实 link，模块目录不得重复内部边。
2. 在 `src/<module>/CMakeLists.txt` 用统一 helper 声明产品 target 和源文件；外部实现库仍由模块用普通
   target 命令接入，并在可用后调用一次 `hyle_register_package_dependency()` 登记 export、feature、
   下游 SDK 与资产闭包，不把源码/供应商清单写回根 CMake 或中央 package 分支。
3. 若有重型可选依赖，将发现/取得逻辑放入 `cmake/Hyle<Module>Dependencies.cmake`，根只在闭包选中
   该模块时调用一个配置入口；生成文件进入模块/build tree，不进入源码树。
4. 提供 `tests/<module>/CMakeLists.txt`，登记 C++ 正确性、适用的纯 C/C surface/header 检查和
   `hyle_<module>_tests` aggregate。没有测试所有权的产品模块 configure 失败。
5. 有真实 workload 时再提供 `bench/<module>/CMakeLists.txt`；benchmark 可缺席，但不能以 benchmark
   方便为理由改变产品 API、常态 metadata 或依赖形状。
6. 公共模块 target 自动参与 build-tree alias 和 install export；不要另维护一份安装 target 清单。
   新 feature 若改变交付物，在模块的同一 registry 声明补 feature/license/find-dependency，并加外部
   consumer 验证；不得直接修改 `HylePackage.cmake` 增加依赖名字分支。
7. 更新对应 architecture（边界/原则）、guide（用户选型）、C guide（FFI 映射）、performance（证据）
   与 milestone（施工记录），各文档不互相复制职责。
8. 至少验证最小 component 子集、依赖闭包 aggregate、source embed 和相关 static/shared package；发布前
   再进入完整平台/后端矩阵。终态 link audit 必须通过，不能靠扩大 component 图掩盖错误依赖。

新增模块通常会修改“一处中央图 + 模块自己的 source/tests/bench 目录”；新增模块能力通常只改模块依赖
入口和该模块所有权目录。若一次普通能力扩展必须修改多个无关模块的 CMakeLists，先判断是否缺少合理
的公共 target/helper，而不是继续传播全局变量。

这里的“独立”指所有权目录、变量/生成目录隔离、可选择的 target 图和独立验证入口，不是把每个
目录做成另一个 `project()`。开发者仍从仓库根配置，由根统一执行工具链门槛和依赖闭包。例如只施工
container 时使用独立 build tree：

```powershell
cmake -S . -B build-container -DHYLE_COMPONENTS=container -DHYLE_BUILD_BENCH=OFF
cmake --build build-container --target hyle_container_tests --config Release
ctest --test-dir build-container -C Release -L container --output-on-failure
```

这只组装 `error + platform + mem + container` 产品闭包，不探测 io/compute 的第三方。需要同时回验
依赖模块时构建总聚合 `hyle_tests`；只验证当前模块时用 `hyle_<module>_tests` 与 CTest label。

这也是多人/多 AI 客户端并行施工的合并边界：大家必须从同一 clean main SHA 建独立 branch/worktree，
一个施工单元只拥有 `include/hyle/<module>`、`src/<module>`、`tests/<module>`、`bench/<module>`、
对应 guide/architecture/performance 文档；不要同时改根 component 图、公共 helper、共享 peer 配置和
总索引。若任务证明必须改 `core`、公共宏、模块依赖或根 CMake，先停下并交由基线维护者串行落地，
其他分支 rebase 到新 SHA 后继续。这样是减少文本冲突和隐式合同漂移，不是允许模块绕开向下依赖复审。

## 四条消费路径

1. **仓库顶层构建**：默认 `all`，产物集中到 `<build>/bin`，tests 默认 ON、bench 默认 OFF。
2. **源码嵌入**：父工程在 `add_subdirectory(hyle)` 前设置 `HYLE_COMPONENTS`；tests 在 subproject
   默认 OFF，HYLE 不改父工程的 `BUILD_SHARED_LIBS`、configuration types 或全局输出目录。
3. **安装包消费**：生产构建执行 `cmake --install`，下游用
   `find_package(Hyle COMPONENTS ...)` 后链接同名 `hyle::<module>`；config 会拒绝安装中不存在的
   component，并传播 C++23、static/shared、模块依赖和内部静态后端。`Hyle_AVAILABLE_COMPONENTS`
   是本次组装的产品清单，`Hyle_AVAILABLE_FEATURES` 是 trace/TLS/CUDA/Vulkan 等能力清单。
4. **直接预构建文件**：仅是诊断/ABI smoke 路径；缺少传递依赖建模，不是推荐发布面。

0.1 不另装 pkg-config 文件。oneTBB/Abseil/libevent/mimalloc 的 `.pc` 证明它适合简单 Unix link
closure，但 HYLE 的 static 包还要表达 C++ link driver、CMake imported archive、可选
OpenSSL/CUDA/ggml/volk 与平台依赖；只输出 `-lhyle_<leaf>` 会让 shared 看似可用而 static/feature
包静默缺依赖。relocatable `HyleConfig.cmake` 因而是 headers+shared/static 两种二进制 SDK 的
唯一机器合同；完整源码交付继续以根 component/feature target 图为合同。其他构建系统应导入
CMake package，或用一个薄 CMake bridge 产出其本地 target。若真实消费者要求 `.pc`，
实现门是 shared/static 与 feature closure 全部 source-independent smoke，不接受半覆盖。

Windows 源码嵌入时没有 rpath。consumer executable 链接共享 HYLE 后调用
`hyle_stage_runtime_dependencies(target)`，会把其传递 DLL 放到 executable 同目录；这比让 HYLE
篡改父工程输出目录更局部、也更容易迁移。安装包提供同名 helper；Unix build/install tree
不要求用户设置全局 loader 环境：安装的每个 HYLE shared object 自带 sibling-relative lookup
（ELF `$ORIGIN`、Mach-O `@loader_path`），因此传递模块依赖不错误依赖 executable RUNPATH 的继承。
这只设置在 HYLE 的 install target，源码嵌入时仍由父工程拥有其 RPATH 策略。

install export 直接遍历已解析产品 target，不另造一份依赖表；build-tree alias 与 imported target
都叫 `hyle::<module>`。`HYLE_INSTALL=ON` 只允许 `HYLE_BUILD_TESTS=OFF`。开发测试分两门：默认
instrumented 门定义 `HYLE_TESTING` 以覆盖 fault/结构诊断；production-shape 门关闭 instrumentation，
用与出厂库相同的 layout/热路跑其余主体测试。前者不能代替后者。公开头整体安装（header-only foundation 有跨二进制
组件引用），可用二进制组件由 `Hyle_AVAILABLE_COMPONENTS` 和 imported targets 决定。安装布局服从
GNUInstallDirs；bin/lib/include/data 子目录必须保持相对路径，迁移根由 `CMAKE_INSTALL_PREFIX` 决定，
避免 export 把第一次安装机器的绝对目录烙进 consumer。

## 静态、共享与 ABI

`HYLE_STATIC=OFF` 默认让每个已选 HYLE 模块成为 SHARED target；ON 时显式成为 STATIC target。
实现不改父工程 `BUILD_SHARED_LIBS`。`hyle_add_module` 同时传播 C++23 floor、公开 include、直接
target 依赖、trace/testing ODR gate 和正确的 `HYLE_<MODULE>_BUILD`/`HYLE_STATIC` 可见性宏。
公开 inline 代码还通过可导出的 `hyle_build_contract` 传播 MSVC 语言/UTF-8 语义与 x86-64-v2
SSSE3 floor；源码树和安装包消费者因此不会把同一个 header-inline 函数编成两种 ODR 形状。
MSVC C++ 还显式传播 `/EHsc`：公开 `noexcept` convenience adapter 会捕获 STL allocation
exception 并转换成 `Status`，因此消费 TU 必须使用匹配的标准 C++ exception 语义；该选项用
language generator expression 隔离，不污染纯 C 或 MASM。platform 的 x86 CPU feature probe
则独立进入关闭 IPO 的 private object target，防止 LTO 把高级 ISA 探测内联进 baseline 路径；
其余 platform target 仍继承父工程 IPO/LTO，安全隔离不得退化成全模块禁用优化。

macOS platform component 的 deployment floor 是 14.4，因为公开 Parker/address-wait 后端使用
14.4 才公开的 `os_sync_wait_on_address*`。HYLE 顶层原生 macOS 构建在用户未指定时设置 14.4；
source embedding 时父工程拥有 `CMAKE_OSX_DEPLOYMENT_TARGET`，若选择 platform 且值更低则
configure 立即失败。只构建不依赖 platform 的 component 不应被这个 floor 无故抬高。

公开 `include/hyle/core/version_defs.h` 是唯一版本源；CMake 在 `project()` 前读取它，随后让
共享库、source archive 和 package 使用同一个项目版本，即使组件 SDK 没有构建 `hyle_core`
也会一致。ABI 未冻结的 0.x 阶段以 `major.minor` 作为
loader identity（例如 `libhyle_time.so.0.1`），package 也只把同 minor 版本判为兼容；到 1.0
以后才切换为常规的 major-only SONAME / `SameMajorVersion`。package smoke 同时检查当前 full
version exact、同 minor compatible 和下一 ABI generation rejected。

C surface gate 在共享构建检查动态导出表，在静态构建检查 archive 中定义的全局 C 符号；
二者都还要对齐 reviewed manifest、声明和 pure-C 引用，不能用“静态库没有 exports”跳过 ABI
完整性验证。

## 验证产物归属

`hyle_add_test(module ...)` 和 pure-C helper 只在对应产品 target 存在时创建测试。每个已选模块
有 `hyle_<module>_tests` aggregate，统一 `hyle_tests` 依赖全部已解析组件的 aggregate。因此：

- 构建 `hyle_core_tests` 会同时产出 core C++/C binary，不再留下 CTest 指向旧或缺失 binary；
- 构建 `hyle_tests` 后运行 CTest，登记项与当前组件闭包一致；
- 切换同一 build tree 的 `HYLE_COMPONENTS` 会重新生成 target/test 集，而不是沿用旧模块。

benchmark 使用同一套组织边界：只有已解析组件的 benchmark 会进入 target 图；
`hyle_<module>_benches` 构建一个模块的完整 benchmark 集，`hyle_benches` 构建所有已选组件的集合。
Highway/xsimd checkout 只在 `simd` 已选时检查，因此打开 benchmark 不会让 core、mem、container
等组装意外依赖 SIMD 对手。benchmark executable 的模块归属显式登记在 CMake 中，不能靠文件名
或开发者记忆推断。

io 的 pinned nghttp2 作为私有 PIC archive：shared io 将其吸收，不产生第二个运行时 ABI；static io
通过 `LINK_ONLY` 迁移 archive、生成头和 MIT 条款。compute 对 ggml/volk 使用同一闭包纪律：只构建
HYLE 实际消费的 CPU、可选 CUDA 与 Vulkan meta-loader PIC 静态实现；shared compute 将它们吸收进
`hyle_compute`，static compute 则导出 `LINK_ONLY` archive、所需头和 license。glslang 只服务 tests/
benchmark 的离线 shader 生成，不进入生产 SDK。CUDA SDK 必须显式给出
`CMAKE_CUDA_ARCHITECTURES`，不能把构建机的 `native` 烙成误称可分发的二进制。

ggml 的 CPU ISA 是交付物契约：通用安装包默认关闭 `GGML_NATIVE` 并使用 x86 SSE4.2 baseline；已知
部署机器可用 `HYLE_SIMD_EXACT_TARGET=AVX2/AVX512` 将 HYLE 与 ggml 提升到同一 ISA floor，源码自用
也可显式打开 `HYLE_COMPUTE_GGML_NATIVE`。这不冒充 ggml 的多后端动态变体分发；若要一个包在未知
CPU 上同时得到每台机器的 ggml 最优 ISA，需另做每 ISA backend + runtime score/load 的产品形状。
cross compile 和三平台完整迁移矩阵的最终证书只见 build/package milestone。

平台 preset 只组合开发入口，不重新定义产品：Linux/macOS 不锁 generator，Windows 明分
CMake 4.2+ 原生 VS 2026、VS 2022 回退与“已选择 MSVC + Ninja”三条路径；后者由
`scripts/hyle-windows.ps1` 经
vswhere 发现现有工具链并临时导入环境。VS generator 不保证在并存的 v143 中选择数值最高工具集，
因此同一脚本在 `-Preset windows` 时枚举 VS 2022 的 MSVC 目录并显式传入最高的 14.43+ `-T version=`；
这生成可由 Visual Studio 打开的 `hyle.sln`。`windows-vs2026` 使用 CMake 4.2 新增的
`Visual Studio 18 2026` generator 和默认 v145；较旧 CMake 会明确拒绝该 preset，不静默降级。
它生成 VS 2026 的原生 `hyle.slnx`。
当已安装的 Visual Studio 比当前 CMake generator 更新时，Ninja 路径仍可使用它的最新编译器。
CI 对 Windows/Linux/macOS 运行全部 14 个模块的确定性 CPU 验证面：TLS、HTTP/2、CUDA、ggml 与
Vulkan 显式关闭，构建聚合 `hyle_tests` 并运行全部 C++、pure-C、header/surface/ABI 门。它不冒充
带外部 SDK/设备的发布矩阵；这些 feature 仍需独立 release check。已经严格收口的底层 roots 另以
`HYLE_TEST_INSTRUMENTATION=OFF` 运行 shipping-layout 验证面；这个 root 清单随模块验收推进而显式
增加，未验收模块即使在 all graph 通过也不会被误写成产品证书。

安装门从空目录制作 shared/static 的 all-component CPU SDK。外部消费者从 package 公布的
`Hyle_AVAILABLE_COMPONENTS` 枚举，为每个 `hyle::<component>` 建立独立最小链接目标，机械守住
export/usage-requirement 图，又不把彼此重叠的 static 闭包硬塞进一个非正常可执行；同时保留真正调用
mem C API 与 C++ `SystemAllocator` 的 general probes，覆盖
`mem -> platform -> error` loader、mimalloc 私有符号和 static C++ link-driver 闭包。严格收口模块在
`samples/package/<module>/...` 拥有 source-independent focused project；package smoke 递归发现这些
`CMakeLists.txt`，分别对 shared/static SDK configure、build、run，不维护第二份中央模块名单。error
当前同时提供从 `LANGUAGES C` 开始的 owning-handle probe 和 `Status`/`Result<T>` C++ probe。只有
“枚举 target 可链接”不能证明每个模块 API 可用，所以新模块收口时必须提交自己的外部 API probe。
Linux 还执行 component 空值、嵌入 cache、error-only dependency cache、晚期依赖成员绕过、PUBLIC/PRIVATE
可见性扩大、package registry 正向/重复/漏 target，以及 offline bundle 映射的 configure contract；
并从 clean checkout 生成完整 source zip、核对 SHA-256/commit/legal payload、解压后独立构建当前
all-component CPU 图。Linux AArch64 也交叉编译同一图，而不再只抽查 core。

### 持续回归与模块验收生命周期

其他模块仍会改变，因此构建门按“发现回归的频率”分层，不能把一张越来越大的矩阵塞进每次提交：

| 层 | 触发频率 | 当前合同 | 结论边界 |
|---|---|---|---|
| CPU baseline | 每个 PR、每次 main 提交，三平台 | all components、tests、Werror、所有外部 feature OFF | 当前图可编译/测试；不冻结模块 API |
| distribution | PR 至少 Linux；main/手动三平台 | shared/static install、外部 consumers、complete source、source embed | 三种交付形状仍完整；focused probe 才证明具体 API |
| portability | PR/main 的 AArch64 compile；周期门逐步增加 | all-component cross + Werror，后续 minimum CMake、ASan/UBSan/TSan、SVE/RVV QEMU | 目标图/工具链可迁移；cross compile 不冒充真机运行 |
| feature/release | 手动、nightly 或 release | TLS/HTTP2/ggml/Vulkan/CUDA 的真实 SDK、loader 和适用硬件 | 只认证实际启用并运行的能力组合 |

`all` 的含义始终是“仓库在该提交声明的全部当前模块”，所以新增组件自动进入 baseline、安装包、
源码包和 AArch64 compile。模块验收则是单独的状态迁移：施工中只要求全图不破；严格收口后，把该
模块的 production-layout roots、module-owned installed API probe、适用 sanitizer/ISA 门和文档合同
加入长期集合。预发布阶段允许直接改善 C/C++ API，不维护虚假的历史兼容快照；一旦公开 ABI 政策
冻结，再增设跨版本 consumer/ABI snapshot，而不是用当前编译成功替代兼容性证明。

为控制 CI 成本，PR 的昂贵 distribution 只在 Linux 跑，main 与手动触发再补 Windows/macOS；三平台
all-module baseline 永不因此降级。每个 job 有明确超时，不能用缩小组件集掩盖死锁或依赖下载失控。
build tree 不跨 toolchain 缓存；未来只缓存带版本/校验和的下载源码或编译器缓存。若 GitHub 在 runner
分配前因额度/账单拒绝任务，这属于 CI 基础设施不可用，不能记为代码失败，也不能替代本地三平台证据。
当前迭代任务、缺口与升级条件记录在
[`../milestones/build-validation-iteration.md`](../milestones/build-validation-iteration.md)。

`cmake/toolchains/linux-aarch64-gcc.cmake` 是整项目交叉编译的参考入口：compiler/sysroot 属工具链，
host program 与 target library/header/package 搜索严格分开，`CMAKE_TRY_COMPILE_TARGET_TYPE` 设为
`STATIC_LIBRARY`，配置过程不得执行 target binary。专用 SVE/RVV QEMU 脚本是 ISA 产品测试，不能
替代这条标准 CMake target-graph 迁移证书。

## 维护规则

- 新模块必须登记一次 component manifest 并创建 module target；configure 会验证依赖名、拓扑顺序、
  target 已存在，以及所有模块装载后的直接 HYLE link 边与 manifest 完全一致。
- 新模块自动进入 all-component baseline/distribution/cross compile；严格收口时在
  `samples/package/<module>/...` 另交 module-owned installed API probe（自动进入 shared/static smoke），
  并把适用的 production-layout、sanitizer、ISA/后端证书加入长期集合。
- 新可选第三方只能在其消费组件已选且 feature 已启用时发现/取得。
- 外部依赖的 export target、package feature、下游 SDK 和安装资产只在所属模块 registry 登记一次；
  中央 package 层只做通用投影，不能按供应商名分支。
- 不在 CMake 中写开发机绝对路径；preset 是可选入口包，不是编译成功的隐藏前置。
- parent-owned 的 generator、toolchain、配置、输出目录和 `BUILD_SHARED_LIBS` 不得被 subproject 改写。
- source tree 只存源码与审查过的 vendored 资产；cache、下载、生成头和制品一律进 build tree。
- HYLE 当前 proprietary notice、第三方总清单，以及编入所选组件的 vendored/ported 许可证随
  install graph 安装；许可证 payload 与 binary/archive 一样属于可迁移闭包。完整边界见
  [`licensing.md`](./licensing.md)。
