# 构建与组装架构

## 定位

HYLE 的 CMake target 图是项目组织架构的可执行版本，不只是“把当前仓库编过”的脚本。
源码、公开头、C ABI、直接依赖、可选后端和验证面都必须归属到 target；文档解释怎样选择，
不能用目录顺序、全局变量或维护者记忆补全隐藏依赖。

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
roots: core error obs math platform time mem simd container util sched io compute graph

platform -> error
time     -> error
mem      -> error + platform
simd     -> error + platform
container-> mem + platform
util     -> container
sched    -> error + platform + time + mem + container
io       -> error + mem + container [+ sched]
compute  -> error + platform + mem + simd [+ sched]
graph    -> error + mem + container [+ sched]
```

箭头右侧是 target 依赖闭包，不重复 core/util-core 纯头、只提供 OS/架构/编译器宏的
`hyle/platform.h` 和 `hyle_trace` 横切 gate。包含 `hyle/platform.h` 只是零链接代价的编译环境
依赖，不等于链接 `hyle::platform`；例如 time 的 native clock 实现用它选择 OS 分支，但 time 的
二进制闭包仍然只有 error。`all` 是完整产品默认值；逗号与分号都可分隔多个根。未知组件在
configure 阶段直接失败，配置日志打印 requested/resolved 清单，因此一次构建“组装了什么”可审计。
方括号内的 sched 边只在 HYLE 已接入 fcontext 的目标组合上存在（当前为 x86-64
Windows/Linux/macOS 与 arm64 Linux/macOS）；其他平台仍可组装 io proactor、compute host/backend
基础层和 graph serial core。显式请求不受支持的 sched 会在 configure 阶段失败，不会静默降级。

组件选择同时控制依赖发现：不选 `io` 就不探测 OpenSSL、不取得 nghttp2；不选 `compute`
就不探测 CUDA，也不取得 ggml/Vulkan-Headers/volk/glslang。第三方默认写入当前 HYLE build
tree 的 `_deps`，源码与生成物一起归该 build tree 所有；不同 toolchain/build tree 不共享这个
目录。离线或经审查的依赖源码通过标准 `FETCHCONTENT_SOURCE_DIR_<NAME>` 注入，生成物仍留在本
build tree，不能写入或依赖源码树。

TLS 与 CUDA 使用 `AUTO|ON|OFF` 三态：AUTO 仅供便利探测，ON 是硬能力契约，OFF 不运行依赖探测。
正式 SDK/CI 必须显式选择 ON/OFF，因此构建机偶然安装了某个 SDK 不会改变发布形状。

## 模块所有权与并行施工

一个产品模块的构建定义不应散落在中央巨型清单中。现有模块按以下固定位置拥有自己的三块接线：

| 内容 | 模块所有位置 | 中央文件只负责 |
|---|---|---|
| 产品源码、公开/私有依赖 | `src/<module>/CMakeLists.txt` | 根 target 图计算 component 依赖闭包并按序加入模块 |
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
CI 对 Windows/Linux/macOS 运行 core 验证面，并在每个平台
额外以 `HYLE_TEST_INSTRUMENTATION=OFF` 运行 error/platform/time/mem 的 shipping-layout 验证面；
随后从空目录制作 shared/static mem SDK，再由只链接 `hyle::mem` 的 pure-C error+mem consumer 与 C++
`SystemAllocator` consumer 真运行；shared 形状机械覆盖 `mem -> platform -> error` 的传递
shared-object lookup，并证明公开 inline 形状不泄漏 mimalloc 符号，static 形状覆盖 mimalloc
link-only archive 与 C++ link driver。simd、CUDA、compute/io 全矩阵不塞进每提交 smoke，仍属发布检查。

`cmake/toolchains/linux-aarch64-gcc.cmake` 是整项目交叉编译的参考入口：compiler/sysroot 属工具链，
host program 与 target library/header/package 搜索严格分开，`CMAKE_TRY_COMPILE_TARGET_TYPE` 设为
`STATIC_LIBRARY`，配置过程不得执行 target binary。专用 SVE/RVV QEMU 脚本是 ISA 产品测试，不能
替代这条标准 CMake target-graph 迁移证书。

## 维护规则

- 新模块必须同时登记 component、直接依赖和 module target；configure 会验证依赖名、拓扑顺序、
  target 已存在，以及 module 声明的直接 HYLE link 边与 component 图完全一致。
- 新可选第三方只能在其消费组件已选且 feature 已启用时发现/取得。
- 不在 CMake 中写开发机绝对路径；preset 是可选入口包，不是编译成功的隐藏前置。
- parent-owned 的 generator、toolchain、配置、输出目录和 `BUILD_SHARED_LIBS` 不得被 subproject 改写。
- source tree 只存源码与审查过的 vendored 资产；cache、下载、生成头和制品一律进 build tree。
- HYLE 当前 proprietary notice、第三方总清单，以及编入所选组件的 vendored/ported 许可证随
  install graph 安装；许可证 payload 与 binary/archive 一样属于可迁移闭包。完整边界见
  [`licensing.md`](./licensing.md)。
