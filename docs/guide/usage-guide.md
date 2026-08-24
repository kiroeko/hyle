# HYLE 使用指南（面向库使用者）

构建并消费 HYLE 见本页；向客户制作 headers+shared、headers+static 或完整源码交付物，见
[`distribution.md`](./distribution.md)。交付形状不决定授权范围。

> 这份文档面向**集成 HYLE 的开发者**——把 HYLE 当依赖用的人。它回答"怎么构建、有哪些开关、每个模块怎么调"。
>
> 区别于另外两份：[`architecture/`](../architecture/) 是设计内幕（为什么这么造，面向贡献者）；[`usage-contract.md`](./usage-contract.md)（同目录）是**为 SOTA 性能保留的调用安全边界**（哪些用法会触发 UB，必读但只讲"红线"）。本指南讲"正常怎么用"。
>
> 项目施工状态见 [`../milestones/README.md`](../milestones/README.md)。本指南不复制就绪度，
> 只列当前源码中的稳定入口和正常用法；绑定 commit 的性能数字只见
> [`../performance/`](../performance/README.md)。

## 0. 模块入口

| 模块 | 主要头文件入口 | 用途 |
|---|---|---|
| `error/` | `hyle/error/{status,result}.h` | Status/Result 与分层错误上下文；选型见 [`error.md`](./error.md) |
| `mem/` | `hyle/mem/{allocator,region,shared_pool,fixed_pool}.h` | allocator、region/pool、诊断与 NUMA/大页；选型见 [`mem.md`](./mem.md) |
| `platform/` | `hyle/platform/{thread,wait,fs,virtual_memory,numa,mapped_file,cpu}.h` | OS 线程、等待、文件系统、VM 与拓扑 |
| `container/` | `hyle/container/{swiss_table,small_vector,concurrent_hash_map,concurrent_node_hash_map}.h` | flat/node 并发 map、连续/内联容器、bit/ring、字符串和调度队列；选型见 [`container.md`](./container.md) |
| `time/` | `hyle/time/{clock,timer_queue,common_timeout_queue}.h` | 时钟、deadline、线程 sleep 与两类 timer；选型见 [`time.md`](./time.md) |
| `util/` | `hyle/util/{ref_counted,registry,scope_guard,random}.h` | 引用所有权、registry、scope guard 与 RNG |
| `obs/` | `hyle/obs/{log,trace}.h` | 结构化日志与可选 trace |
| `sched/` | `hyle/sched/{scheduler,fiber,coro,sync}.h` | M:N fiber/coroutine、work stealing、同步、取消与 blocking offload |
| `simd/` | `hyle/simd/{dispatch,ops,math,native}.h` | runtime/native/exact SIMD；层级选择见 [`simd-usage.md`](./simd-usage.md) |
| `io/` | `hyle/io/{io_context,tcp,udp,file,shared_memory,tls,http,http2}.h` | proactor 网络/文件/共享内存和协议适配 |
| `compute/` | `hyle/compute/{device,buffer,stream,kernel,device_pool}.h` | host/CUDA/Vulkan device、buffer、completion 与 kernel |
| `graph/` | `hyle/graph/{taskgraph,parallel,data_parallel}.h` | 增量任务图和数据并行编排 |

## 1. 构建与集成

### 1.0 先选消费形状

普通用户先看自己拿到什么，不需要从 HYLE 的全部 CMake 选项开始阅读：

| 你拿到的内容 | 入口 | 需要设置的 HYLE 构建选项 |
|---|---|---|
| headers + shared SDK | `find_package(Hyle COMPONENTS ...)` | **无**；linkage 和 feature 已由包固定 |
| headers + static SDK | `find_package(Hyle COMPONENTS ...)` | **无**；同样只链接 `hyle::<module>` |
| complete source / source checkout | `add_subdirectory()` 或 FetchContent | 通常只设 `HYLE_COMPONENTS`、`HYLE_STATIC`、`HYLE_BUILD_TESTS=OFF` |

installed SDK 用户用 `CMAKE_PREFIX_PATH` 告诉 CMake 包在哪里，不再设置 `HYLE_STATIC` 或重新选择
backend；可通过 `Hyle_STATIC`、`Hyle_AVAILABLE_COMPONENTS` 和 `Hyle_AVAILABLE_FEATURES` 查询收到的
产品。源码用户只有在确实需要 TLS/CUDA 等可选能力时才设置对应模块 feature；正式、可复现的产物写
明确 `ON`/`OFF`，本机试用才保留 `AUTO`。

三种形状的完整可复制工程、Windows DLL 处理、源码 FetchContent 和自行制作内部 SDK 见
[`distribution.md`](./distribution.md)。无论哪种形状，用户代码都链接相同的 `hyle::<module>`；不要
手写 include path、库文件名或传递依赖顺序。

### 1.1 用 CMake 集成（推荐）

HYLE 是标准 CMake 工程，每个模块导出 `hyle::<module>` 别名 target。先声明所需产品根，
CMake 会自动加入依赖闭包，不会配置无关 GPU/IO 后端：

```cmake
set(HYLE_COMPONENTS "container;simd")
add_subdirectory(hyle)            # 或 FetchContent；已安装 SDK 使用 find_package
target_link_libraries(your_app PRIVATE hyle::container hyle::mem hyle::simd)
hyle_stage_runtime_dependencies(your_app) # Windows shared build 的传递 DLL
```

链接 `hyle::<module>` 自动带上 `include/` 头路径与依赖模块（如 `hyle::container` 自动拉
`hyle::mem`）。包含头一律 `#include "hyle/<module>/<header>.h"`。源码嵌入默认不构建 HYLE
自己的测试，也不修改父工程的 `BUILD_SHARED_LIBS`、输出目录或 configuration types。

### 1.2 构建选项全集（这是"开关太多"的答案）

HYLE 的构建开关分**产品/组装参数**与**benchmark 对照门**两个范畴，别混。安装包消费者不使用
这些构建开关；以下全集只供源码集成、HYLE 开发和制包时查阅。构建系统审查期间
不再维护容易漂移的“总共有几个开关”手工数字，以本表与 `cmake -LH` 为准：

- **A. 产品/组装参数**（`HYLE_<名>` cache 变量）——决定选哪些模块、static/shared、可选
  backend、测试与工程严格度；普通集成最常用的是 `HYLE_COMPONENTS` 和 `HYLE_STATIC`。
- **B. 开发/基准对照门**（`bench-all-sota` 管理 **10 个共享 gate**；IO 另有 3 个平台/依赖专用 gate）——**只往 benchmark 加对照行，对出厂库零影响**。只有已选组件的 benchmark 会创建；仅当组件闭包含 `simd` 时才检查 `HYLE_BENCH_VENDOR_DIR` 下固定版 Highway/xsimd checkout。集成方永远用不到（见 §1.2.1）。
- **CMake preset** 是开发机入口组合，不是产品参数或构建前置；集成方不需要 preset。

> **命名约定**：凡产品参数都是 `HYLE_<名>` cache 变量，用 `-D` 设；bench 对照门统一带 `HYLE_BENCH_` 前缀，自我标记为 B 类。CMakePresets 里的 `bench` / `bench-all-sota` 是**预设包**（不是参数本身），`--preset` 选用——它们只是"把一组 `-D` 一次设好"的快捷方式，`bench-` 前缀表明展开的全是 B 类 bench 门。

**A 类——产品构建参数**（顶层 `cmake -S . -B build -D<OPT>=<VAL>` 设）。**默认值已为"生产使用者"调好**——通常你只需默认 + 选 Release/静态。

| 选项 | 默认 | 取值 | 何时改 |
|---|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`/`Release`/`RelWithDebInfo` | 性能库默认 Release(-O3)；要符号调试用 RelWithDebInfo |
| `HYLE_COMPONENTS` | `all` | 组件名的 `;`/`,` 列表 | 只组装所需根；支持 core/util_core/error/obs/math/platform/time/mem/simd/container/util/sched/io/compute/graph，依赖自动闭包；util_core 是无 binary artifact 的可安装 C++ target |
| `HYLE_STATIC` | `OFF`(SHARED) | `ON`/`OFF` | 要静态链进你的二进制（无 HYLE DLL 分发）设 ON；不改变父工程 `BUILD_SHARED_LIBS` |
| `HYLE_BUILD_TESTS` | 顶层 `ON`、嵌入 `OFF` | `ON`/`OFF` | ON 时只创建已选组件测试；`hyle_<module>_tests` 同时构建该模块 C++/C binary |
| `HYLE_TEST_INSTRUMENTATION` | `ON` | `ON`/`OFF` | 仅 tests ON 时生效；ON 定义 `HYLE_TESTING` 以测 fault/结构诊断，OFF 运行 shipping layout/hot path 主体测试 |
| `HYLE_BUILD_BENCH` | `OFF` | `ON`/`OFF` | 只有想跑微基准才开；仅创建组件闭包内的 benchmark。构建 `hyle_<module>_benches` 或全部已选组件的 `hyle_benches` |
| `HYLE_INSTALL` | `OFF` | `ON`/`OFF` | 显式生成 relocatable `HyleConfig.cmake`、组件 targets 和 install rules；与 `HYLE_BUILD_TESTS=ON` 互斥，避免发布 test-only ABI |
| `HYLE_WERROR` | `OFF` | `ON`/`OFF` | CI 用 ON；本地开发 OFF |
| `HYLE_SANITIZER` | `off` | `off`/`asan`/`tsan`/`ubsan` | 查内存/数据竞争/UB 时开（配 Debug/RelWithDebInfo）。**严格校验**：非法值直接 `FATAL_ERROR`（不静默降级为 off）；认编译器原生别名 `address`→asan、`thread`→tsan、`undefined`→ubsan、`none`→off。**MSVC 只支持 asan**（tsan/ubsan 在 MSVC 会 `FATAL_ERROR`，需 GCC/Clang，如 WSL） |
| `HYLE_DEPS_DIR` | `<hyle-build>/_deps` | path | 当前 build tree 的 FetchContent 源码/生成物；不要让不同 toolchain/build tree 共用。离线源码用标准 `FETCHCONTENT_SOURCE_DIR_<NAME>` 注入 |
| `HYLE_SIMD_EXACT_TARGET` | 空（runtime fat） | SCALAR/SSE4/AVX2/AVX512/NEON/SVE/RVV | 构建整库 exact ISA；普通分发不要改，见 §2 |
| `HYLE_TRACE` | `OFF` | `ON`/`OFF` | 编译期 instrumentation；ON 还需 `HYLE_TRACY_DIR` |
| `HYLE_IO_HTTP2` | 有 fcontext `ON`，否则 `OFF` | `ON`/`OFF` | 仅在选中 io 且 sched 可用时取得 nghttp2；无 fcontext 显式 ON 失败，OFF 仍保留 unavailable stub C ABI |
| `HYLE_IO_TLS` | `AUTO` | `AUTO`/`ON`/`OFF` | 有 fcontext 时 AUTO 有 OpenSSL 就启用；无 fcontext 时 AUTO 不探测、ON 失败；OFF 始终不探测 OpenSSL |
| `HYLE_COMPUTE_CUDA` | `AUTO` | `AUTO`/`ON`/`OFF` | AUTO 有 Toolkit 就启用；ON 要求 CUDA；OFF 不探测 Toolkit、也不启用 CUDA language |
| `HYLE_COMPUTE_GGML` | 有 fcontext `ON`，否则 `OFF` | `ON`/`OFF` | ggml bridge 依赖 sched；无 fcontext 显式 ON 失败，host/CUDA/Vulkan 不受影响 |
| `HYLE_COMPUTE_VULKAN` | `ON` | `ON`/`OFF` | 仅在选中 compute 时取得 Vulkan-Headers/volk；不要求 fcontext |
| `HYLE_COMPUTE_GGML_NATIVE` | 普通构建 `ON`；install `OFF` | `ON`/`OFF` | ON 为当前 CPU 编译 ggml；可迁移包保持 OFF，已知 CPU 可配合 exact target 提升 ISA floor |
| `CMAKE_CUDA_ARCHITECTURES` | 普通 CUDA 构建 `native` | CMake architecture 列表/`all-major` | CUDA SDK 必须显式给出，如 `80;86;89`；`HYLE_INSTALL=ON` 拒绝 `native` |

`AUTO` 适合本地尝试；正式 SDK/CI 必须显式写 `ON` 或 `OFF`，否则同一命令可能因构建机安装的 SDK
不同而得到不同 feature manifest。`ON` 的意义不是“尽量开启”，而是“这个产物没有该能力就拒绝产生”。

#### 测试专用接口（`HYLE_TESTING` 门控——集成方零残留）

有的接口**只为测试存在**：让测试够到一条生产调用方永不需要的内部路径（例：一个"可控 Event"——先报 N 次 NotReady 再 Ready，让 host 后端无需真 GPU 就能驱动 GpuPoller 的 suspend→wake 全链路）。这类接口**绝不进出厂库**，用 `HYLE_TESTING` 宏门控。

- **宏来源**：`HYLE_TESTING` 只在 `HYLE_BUILD_TESTS=ON` 且
  `HYLE_TEST_INSTRUMENTATION=ON` 时由 `hyle_testing` INTERFACE target 统一传播，保证库与 test TU
  无 ODR 分裂。默认开发门测难达 fault/结构路径；另用 instrumentation OFF 跑 shipping
  layout/hot-path 主体测试。集成方 tests OFF 时门控块整个消失，**出厂库零 test hook**。
- **写法契约**（写 test-only 接口必守）：
  1. 声明**与**定义都包在 `#if defined(HYLE_TESTING)` 里；
  2. 函数/成员注释**必须以 `TEST-ONLY:` 开头**，说明它为何不是生产 API。一个没有此标注、却只被测试调用的 raw/辅助接口 = bug（要么补此门控 + 注释，要么它其实是该删的遗留、或该补注释的公开 API）。

#### 1.2.1 B 类——开发/基准对照门（集成方忽略）

**仅当 `HYLE_BUILD_BENCH=ON` 时额外可用**（拿 HYLE 跟外部 SOTA 参考库赛跑，给 benchmark 加对照行；**不影响出厂库**）。最常用两个:
| 选项 | 默认 | 作用 |
|---|---|---|
| `HYLE_BENCH_ABSL` + `HYLE_ABSL_DIR` | `OFF` | SwissTable vs `absl::flat_hash_map`;SmallVector vs `absl::InlinedVector` |
| `HYLE_BENCH_HIGHWAY` | `OFF` | simd vec / dtype / math vs 固定版外部 Google Highway（只 toggle 对比行） |

> 还有 `HYLE_BENCH_{MOODYCAMEL,TBB,RIGTORP,XSIMD,SIMDUTF,BOOST,WYHASH,MARL}`(队列/分配器/math/utf8/侵入容器/hash 对标)。`HYLE_BENCH_{HIGHWAY,XSIMD}` 都使用 `HYLE_BENCH_VENDOR_DIR` 下的固定外部 checkout；SIMDUTF 仍在 `third_party/`。其余是 opt-in sibling clone。**一键全开**：先运行 `scripts/fetch-bench-vendors.{sh,ps1}`，再运行 `scripts/setup-bench-libs.sh`，最后 `cmake --preset bench-all-sota`。全清单与运行法见 [`scripts/README.md`](../../scripts/README.md)。
>
> 上述 ABSL/Highway 加 8 个补充项正好是 preset 管理的 **10 个共享 gate**。IO benchmark 另有
> `HYLE_BENCH_{LIBURING,ASIO,LIBUV}`；它们受 OS、头/库和 backend 形状约束，不由
> `bench-all-sota` 隐式打开，需要按 IO 性能文档显式选择。
>
> 产品 SIMD math 不再依赖 in-tree Highway/xsimd；固定外部 checkout 只用于 benchmark 对照。
> `simdutf` 仍作为 UTF-8 产品核 vendored。`HYLE_BENCH_*` 只切换对照行，不改变产品路径。

**集成方最小命令**（静态链、不编测试）：
```bash
cmake -S hyle -B build -DCMAKE_BUILD_TYPE=Release -DHYLE_COMPONENTS="mem;container" \
  -DHYLE_STATIC=ON -DHYLE_BUILD_TESTS=OFF
cmake --build build
```

#### 1.2.2 开发用 CMake preset

`CMakePresets.json` 当前提供日常三平台入口与 benchmark 入口；它们是仓库维护便利，不是
source embed/install consumer 必须继承的产品合同：

| preset | 作用 |
|---|---|
| `windows-vs2026` | CMake 4.2+ 的原生 VS 2026/v145 solution（`.slnx`） |
| `windows` | VS 2022 generator 回退；包装脚本显式选择最高的合格 v143 toolset |
| `windows-ninja` / `win` | 使用已选择的 MSVC + Ninja；`scripts/hyle-windows.ps1` 用 vswhere 自动导入最新已安装环境 |
| `linux` / `macos` | 不指定 generator；通过 CC/CXX 或 CMake 正常规则选择受支持工具链 |
| `wsl` | `linux` 的 gcc-14/g++-14 便利形状；重压/benchmark 使用 native ext4 checkout |
| `wsl-tsan` | 继承 wsl，独立 build tree 启用 ThreadSanitizer |
| `package-cpu-shared/static` | all-component、install ON 的确定性 feature-enabled CPU SDK 输入；TLS/CUDA OFF，HTTP/2/ggml/Vulkan ON，ggml native OFF |
| `linux-aarch64-cross` | 标准 toolchain-file 驱动的 all-component CPU AArch64 cross smoke；Werror 下验证全部当前模块可编译/链接，不执行 target binary，也不代表真机或模块验收 |
| `bench` | 默认 all-components + `HYLE_BUILD_BENCH=ON` + Release，独立 `build-bench` cache；因包含 SIMD，先准备 `HYLE_BENCH_VENDOR_DIR` 中固定版 Highway/xsimd。自选非 SIMD 组件不需要它 |
| `bench-all-sota` | 继承 `bench`，再把 10 个共享 `HYLE_BENCH_*` gate 一次设 ON（含 WYHASH；先 fetch SIMD vendors，clone 类再跑 `setup-bench-libs.sh`）；不隐式打开 3 个 IO 专用 gate |

所有 preset/source-embed/install 形状都要求 64-bit target ABI；32-bit toolchain 会在 configure
阶段收到明确错误。x86-64、AArch64、RV64 有架构专用路径；其他 64-bit 架构只取得可用的
generic/scalar 能力，不能假定 SIMD variant 或 fcontext 可用。

```bash
# Windows（PowerShell；其他 clone 型 rival 按 scripts/README 准备）
scripts/fetch-bench-vendors.ps1
cmake --preset bench-all-sota && cmake --build --preset bench-all-sota

# Linux/WSL（vendor 在 ext4；generic preset 不替你选择 GCC14）
scripts/fetch-bench-vendors.sh --dir "$HOME/vendor"
CC=gcc-14 CXX=g++-14 cmake --preset bench-all-sota -DHYLE_BENCH_VENDOR_DIR="$HOME/vendor"
cmake --build --preset bench-all-sota
```

> 集成方不需要继承这些 preset；它们是维护者的开发、CI、交叉编译和制包入口。嵌入方仍只用
> §1.2 的 A 类 `-D` 参数，安装包消费者只用 imported targets。

### 1.3 工具链要求

C++23。GCC 14+ / Clang 18+ / MSVC 19.43+（VS 17.13+）。CMake 3.25+。低于此 CMake 直接报错（`std::expected` 等需要）。

### 1.4 把 HYLE 编成什么库（动态 vs 静态）+ 怎么发布

HYLE 默认编成**共享库**（`.so`/`.dll`，每个已选模块一个 `hyle_<mod>`）；
`-DHYLE_STATIC=ON` 编成**静态库**（`.a`/`.lib`）。HYLE 作为顶层工程时产物集中在
`<build>/bin/`；作为 subproject 时尊重父工程布局。怎么选：

| 你的场景 | 选 | 命令 | 为什么 |
|---|---|---|---|
| **多个进程/插件共用一份 HYLE**；想热替换实现；想锁住 ABI 边界 | **动态**（默认）| `cmake -B build -DCMAKE_BUILD_TYPE=Release` | 一份 `.so`/`.dll` 多处加载；`HYLE_<MOD>_API` 的 dllexport/import 在此生效，跨边界 ABI 稳定 |
| **单一可执行、想零运行时依赖**；嵌入式/容器精简；想让链接器删未用符号 | **静态** | `cmake -B build -DHYLE_STATIC=ON -DCMAKE_BUILD_TYPE=Release` | 折进你的二进制，没有"找不到 .so"问题；`HYLE_<MOD>_API` 退化为空 |

**分发型 vs 自用型构建（与 §2 SIMD 模型直接相关）**：
- **分发给未知 CPU** → 用**默认 baseline**（别加全局 `-march=native`）。这样 SIMD 的 baseline dispatcher 不会发出 AVX2 指令，老 CPU 不 SIGILL；热路径由运行期 CPUID dispatch 选变体（§2-B）。这是发布 `.so`/`.dll` 的正确姿势。
- **自编自用（已知就这台机器）** → 加 `-march=native`（或 `/arch:AVX2`），见 §2-A，零 dispatch、极致。**但这种二进制别发给更老的 CPU。**

最小发布命令（动态、分发型、不带测试）：
```bash
cmake -S hyle -B build -DCMAKE_BUILD_TYPE=Release \
  -DHYLE_COMPONENTS='simd;mem' -DHYLE_BUILD_TESTS=OFF -DHYLE_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/hyle
cmake --build build -j2
cmake --install build
```

### 1.5 消费安装后的 HYLE package（推荐二进制路径）

下游只需要安装前缀，不需要 HYLE 源码路径，也不应 `find_library` 猜文件名/静态链接顺序。
完整 shared/static 可运行示例见 [`samples/04_prebuilt_lib/`](../../samples/04_prebuilt_lib/)：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS simd mem)

add_executable(app main.cc)
target_link_libraries(app PRIVATE hyle::simd hyle::mem)
hyle_stage_runtime_dependencies(app) # Windows shared DLL closure；其他形状 no-op
```
- `CMAKE_PREFIX_PATH=/opt/hyle`（或 `Hyle_DIR=<prefix>/lib/cmake/Hyle`）告诉 CMake 去哪里找包。
- Unix shared SDK 的 HYLE 模块用 loader-relative 路径寻找同一 `lib/` 中的其他 HYLE 模块；整体
  移动 prefix 后不需要为 HYLE 内部传递依赖另设 `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`。系统部署
  到非默认位置时，应用自身怎样找到第一个 HYLE 库仍由应用的 RPATH、loader 配置或打包器决定。
- 自定义安装根只设 `CMAKE_INSTALL_PREFIX`；可定制相对的 GNUInstallDirs 子目录，不要把
  `CMAKE_INSTALL_LIBDIR`/`INCLUDEDIR` 等直接设成绝对路径，否则无法整体搬迁，configure 会拒绝。
- imported target 自动传播 C++23、`HYLE_STATIC`、公开头与依赖模块；静态 mem 的 mimalloc 后端也作为
  内部 `LINK_ONLY` target 随包安装，用户不手写它。
- 纯 C 工程消费 static package 时，`HyleConfig` 只为最终链接启用 CXX language，让 CMake 选择当前
  toolchain 的 C++ ABI/runtime；`.c` 源仍按 C17 编译，不硬编码 `stdc++` 或 `c++`。
- package 只承诺它实际安装的组件；请求缺失 component 会在 configure 阶段列出 available 清单。
- `Hyle_AVAILABLE_COMPONENTS` 可审计模块闭包，`Hyle_AVAILABLE_FEATURES` 可审计 trace、TLS、CUDA、
  Vulkan 等可选能力；不要从本机恰好装了什么反推包的形状。
- 有 sched/fcontext 的 io package 可保留默认 `HYLE_IO_HTTP2=ON`：pinned nghttp2 静态协议核被 shared
  io 吸收，static io package 自动迁移其 link-only archive、生成头和 MIT 条款。无 fcontext 目标默认
  OFF，显式 ON 失败；stub 仍提供稳定 availability=0，但不会伪写 `io_http2` feature。
  该最小 io 形状只承诺 proactor/shared-memory 和 C HTTP/2 probe/stub；头树虽然统一安装，
  `hyle::io` 会导出 `HYLE_IO_HAS_COROUTINE_TRANSPORT=0`，sched-dependent C++ 头会立即拒绝编译。
- compute package 同样完整：shared `hyle_compute` 吸收 ggml/volk，static package 自动传播其
  link-only archive、头与 Threads/CUDA 依赖；可运行 consumer 见
  [`samples/07_installed_compute/`](../../samples/07_installed_compute/)。正式构建显式选择
  `HYLE_COMPUTE_CUDA=ON/OFF`；ON 时还要给 `CMAKE_CUDA_ARCHITECTURES`，不能用 `native` 出 SDK。
- 安装前缀始终带 HYLE proprietary notice 和第三方总清单；实际编入的 mimalloc、xxHash、
  wyhash、simdutf、fcontext、picohttpparser、Highway/xsimd/JPEG XL port 以及可选 Tracy 的
  许可证按 target 图迁移。拿到源码或 SDK 不等于取得 HYLE 原创部分的外部使用许可。
- **当前边界（诚实）**：core/error/platform/mem/time/obs/simd/util/container/sched/io 已有纯
  `extern "C"` C ABI；compute/graph 尚无 C 产品面，math 要先完成与 simd/std 的边界审查再决定是否
  需要独立 C ABI。统一 install/export 已完成，不再属于 C ABI 待办。SIMD 外部调用必须用
  `hyle/c/simd.h`，不能把 C++ mangled symbol 当 FFI。

## 2. SIMD 的编译变体模型（`HYLE_SIMD_TARGET_*` 是什么）

常用 batch 能力、稳定 C ABI 示例和 exact/native source 路径集中在
[`simd-usage.md`](./simd-usage.md)。本节只解释如何构建和选择 ISA 变体。

> 这是"编译选项多"的另一处来源，单列澄清——**普通使用者基本不用碰它**。

`simd/` 有两套用法，按你的部署方式选：

**(A) 自编自用 / 已知硬件（最简，零 dispatch，极致）**：为你自己的机器加 `-march=native`（GCC/Clang）或 `/arch:AVX2`（MSVC）即可。`vec.h` 的 `ScalableTag<T>` 会按编译期 baseline 自动取最宽向量。**不需要任何 `HYLE_SIMD_TARGET_*`**——它们是给下面 (B) 的。

```cpp
#include "hyle/simd/vec.h"
namespace hs = hyle::simd;
float dot(const float* x, const float* y, size_t n) {
  const hs::ScalableTag<float> d;           // 满宽向量（-march=native 下 = AVX2 8 lane）
  const size_t w = hs::Lanes(d);
  auto acc = hs::Zero(d);
  size_t i = 0;
  for (; i + w <= n; i += w)
    acc = hs::MulAdd(hs::LoadU(d, x+i), hs::LoadU(d, y+i), acc);  // AVX2 下编成 vfmadd
  float s = hs::ReduceSum(d, acc);
  for (; i < n; ++i) s += x[i]*y[i];
  return s;
}
```
同一份源码，目标 ISA 决定它编成 SSE4.2(4 lane) 还是 AVX2(8 lane)。只有在你明确控制部署 CPU
或正在编写 HYLE 的变体 TU 时才使用这条路径；普通应用应优先使用下面的自动派发 API。

**(B) 分发单一二进制到未知 CPU（运行期选 ISA）**：若你分发一个要在各种 CPU 上跑的二进制，不能 `-march=native`（老 CPU 会 SIGILL）。模型是"每 ISA 编一份变体 + 启动时 CPUID 选最优"。`HYLE_SIMD_TARGET_SSE4`/`_AVX2`/`_AVX512` 宏就是给**变体 TU** 用的——同一份 kernel 源码编 N 份，每份配一个宏 + 对应 `-march`。**运行期探测用 `dispatch.h`**：

```cpp
#include "hyle/simd/dispatch.h"
if (hyle::simd::supports(hyle::simd::Isa::kAvx2)) { /* 调 AVX2 变体 */ }
// best_isa() 给最高可用档；isa_name() 打印
```
> **现状（simd-22 起）**：普通应用不定义任何 SIMD 宏或 `-march`。对 HYLE 自带的派发型
> batch op，直接调用普通 API；共享的 `Runtime` 表只选择一次，之后每个 batch 经过一次间接调用：
> ```cpp
> #include "hyle/simd/ops.h"
> #include "hyle/simd/runtime.h"
> float d = hyle::simd::dot(x, y, n);              // 自动选 scalar/SSE4.2/AVX2/AVX512
> auto selected = hyle::simd::runtime().isa;
> ```
> 若部署 ISA 已知，用 `hyle_target_simd(TARGET app ISA AVX2)` 配置整个 target，并调用
> `hyle::simd::native::*`；这条路径没有 CPUID 或函数指针调用。若要让 HYLE 的全部普通 batch API
> 也固定目标，配置 HYLE 时传 `-DHYLE_SIMD_EXACT_TARGET=AVX2`。完整边界和示例见
> [`simd-dispatch-guide.md`](../architecture/simd-dispatch-guide.md)。
>
> **AVX512 验证边界**：`dot`/`saxpy` 的 AVX512 变体已接进派发，Intel SDE 已执行正确性验证。
> SDE 是正确性与派发证据，不代表真实硬件时序；AVX512 性能结论仍只接受真机 benchmark。

### 2.1 哪些行为是编译期定的、哪些是运行期探测的（三条轴，别混）

平台相关行为分三类，**只有第 2 类走上面的"编译变体 + 运行期 dispatch"双重模型**：

| 轴 | 例 | 何时定 | 说明 |
|---|---|---|---|
| **平台/架构身份** | `HYLE_OS_*`/`HYLE_ARCH_*`/`HYLE_COMPILER_*`（`platform.h`）| **纯编译期** | 你为哪个 OS/架构编译就是哪个，无从运行期切换（syscall/ABI 在链接期已定）。不属 ISA dispatch 范畴。 |
| **`simd/` 的 ISA 后端** | vec 后端 / `ops.h` / `dtype_ops.h` | **双重**（编译变体 + 运行期选）| 完全符合 §2 双模型：`-march` 自编(A) 或 运行期 CPUID 选变体(B)。**探测档 ≠ kernel 档**：`best_isa()` 如实报 CPU 能力(含纯 AVX 档 `kAvx`,Sandy/Ivy Bridge 的 256-bit float),但若该档无专属 kernel(如无 avx-only kernel),派发会向下 clamp 到有 kernel 的最近档(kAvx→sse4)——绝不 SIGILL。|
| **`container/` SwissTable 的 SIMD 探测** | `probe.h` 的 SSSE3 `_mm_sign_epi8` 快路径 | **纯编译门控**（`HYLE_HAVE_SSSE3` 宏 + SSE2 兜底）| **故意的例外**：container 是 Tier 0,不依赖 `simd/`(Tier 1) 以免架构倒置,故内联一小段自包含探测并编译门控。**SSSE3 现为全局 baseline**(x86-64-v2)：CMake 顶层对 x86 GCC/Clang 全局加 `-mssse3`,MSVC x64 无条件,故 `HYLE_HAVE_SSSE3` 三平台默认成立、SSSE3 快路径默认生效。**无 SIGILL 风险**(SSE2 基线恒在兜底)。将来可选复用 `simd/ops` 做运行期探测,但不强耦合。|

**一句话**：`simd/` 的双重探测设计是真的、自洽的；`platform.h` 是天然纯编译期的另一根轴；`container/probe.h` 是**有意的纯编译门控例外**(Tier 0 不倒挂依赖)——SSSE3 现为全局 baseline,快路径默认生效,正确安全,但不做运行期 dispatch。

### 2.2 `hypot`：完整语义与受限极速版显式二选一

```cpp
#include "hyle/simd/math.h"

// 与 std::hypot 一样做指数缩放：极大/极小量级也保持 overflow/underflow safety。
hyle::simd::hypot_f32_batch(a, b, dst, n);

// 直接计算 sqrt(a*a + b*b)：不做逐 lane 检查或缩放，由调用方保证输入域。
hyle::simd::hypot_in_range_f32_batch(a, b, dst, n);
```

| API | 选择条件 | 契约与代价 |
|---|---|---|
| `hypot_f32_batch` | 输入可能跨极端数量级，或需要完整 `std::hypot` 语义 | exponent scaling，避免中间平方溢出及不必要的下溢 |
| `hypot_in_range_f32_batch` | 调用方已知数据处于普通范围，吞吐优先 | direct square/add/sqrt，允许实现融合FMA；`a`、`b` 必须有限，且中间和必须可表示；不保护极小输入的平方下溢，末位舍入可随ISA不同 |

`hypot_in_range_f32_batch` 不在热路径逐元素检查契约。一个简单的**防溢出充分条件**是每个输入都满足 `|x| <= sqrt(FLT_MAX/2) ≈ 1.30438e19`；若输入可能接近上限、包含无穷/NaN，或必须保留极小非正规数，请用完整版本。`dst` 可与 `a` 或 `b` **完全同址**原位写回，但其他部分重叠不支持。f64 direct-formula 没有稳定收益，因此不提供 `hypot_in_range_f64_batch`；实测见 [`../performance/benchmarks-simd.md`](../performance/benchmarks-simd.md)。

## 3. 各就绪模块最小用法

### 3.1 mem —— 两层分配器模型（默认零开销，注入 opt-in）
**默认不用传 allocator**:头内联容器(`SmallVector`/`BoundedQueue`/`ShardedQueue`/`WsFifoQueue`/`WorkStealingDeque`/`ConcurrentVector`)默认参 `mem::SystemAllocator`——编译期绑定、无函数指针接缝(`[[no_unique_address]]` 不占字节)；static/source 形状内联直达 mimalloc，shared SDK 经 `hyle_mem` 一次直接 import 且不泄漏 vendor ABI。要 arena/mock/跨 DLL 才**显式 opt-in** 传运行时 `Allocator` 句柄。
直接用 mem/ API 或 type-erased-core 容器（`FlatHashMap`/`PodVector`/`StringInterner`）并需要指定 backing 时，传 `system_allocator()`（mimalloc backing）：
```cpp
#include "hyle/mem/allocator.h"
using hyle::mem::system_allocator;
auto a = system_allocator();
int* p = a.allocate_n<int>(16);        // 16 个 int 的裸存储
a.release(p, 16);
auto obj = a.make_unique<MyType>(args);  // 经此 allocator 析构的 unique_ptr

// 头内联容器:默认无需 allocator 参数
hyle::container::SmallVector<int, 16> v;          // 默认 SystemAllocator,零开销
hyle::container::BoundedQueue<int> q(1024);       // 只需容量
```

#### 3.1.1 专用分配器选型（选对工具——选错会比通用 malloc 还慢）

mem/ 提供一组**专用分配器**,各自为一种分配模式设计。**选对了大幅赢通用 malloc;选错了反而更慢**(例:单线程却用为多线程共享设计的无锁池,白付每次原子 CAS)。按"分配模式"对号入座:

| 你的场景 | 用 | 为什么 / 别用啥 |
|---|---|---|
| **通用大小、单纯想要快 malloc** | `SystemAllocator` / `system_allocator()` | 默认前者：编译期 direct binding；需要运行时注入形状才取后者句柄。不确定用啥就用 `SystemAllocator`。 |
| **一个阶段内分配很多、整批一次性释放**(一次 taskgraph pass、一个请求的临时对象) | `Region` / `PooledRegion` / `StackAllocator` | 纯指针 bump，**无 per-object free**，clear/reset/析构整批释放；分别是 growable owned、借一个 SharedPool block、owned slab + mark/rewind。 |
| **单线程、同一种固定大小对象反复 alloc/free**(对象池) | `GrowablePool<T>` | 单线程 free-list、无原子、按需增长。这是单线程对象池的正解。 |
| **多线程共享有界固定池，容量必须准确** | `SharedPool` / `ObjectPool<T>` | **硬容量默认**：无 TLS 隐藏，completed free 立即共享；稳定双快照避免分片扫描假耗尽。正常 32-bit tagged CAS，tag 饱和的 shard 永久切入 serialized fallback，绝不回绕。 |
| **worker-local 复用热点，池有充足余量** | `SharedPoolCache` / `ObjectPoolCache<T>` over hard pool | **软容量 opt-in**：cache storage/capacity/batch 显式；本地 free→alloc 零原子。handoff/idle 前由 owner `publish()`。 |

**一句话决策树**：单纯 malloc→`SystemAllocator`；request/frame 整批丢→`Region`；借池中固定 scratch block→`PooledRegion`；LIFO scratch→`StackAllocator`；单线程对象池→`GrowablePool`；多线程共享且 quota 必须准确→`SharedPool`/`ObjectPool`；测得 owner-local churn 热点且能接受软容量→在 hard pool 上显式挂 cache。

hard pool 与 cache 是组合层，但合同不同：`SharedPool` 不隐藏 block；`SharedPoolCache` 可能在其他 owner cache 仍有 free block 时返回 `nullptr`。不要把 cache 性能冒充 hard pool 性能。最新数字与对手形状见 [`performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)；调用红线见 [`usage-contract.md`](./usage-contract.md) §2.1。

### 3.2 container —— 先按形状选
```cpp
#include "hyle/container/swiss_table.h"
using hyle::container::FlatHashMap;
FlatHashMap<uint64_t, uint64_t> m;
m.insert(42, 100);
if (uint64_t* v = m.find(42)) { /* *v == 100 */ }
```
`SmallVector<T,N>` 是通用 owning inline vector；`PodVector<T>` 才是按字节 relocation 的
C-friendly 连续数组。字符串优先名为 `StringView` / `StringInterner`，并发队列必须按
MPMC、sharded、owner-LIFO stealing、owner-FIFO sharing 的角色选择，不能只看 benchmark。
完整选择表和坑点见 [`container.md`](./container.md)。

**哈希器（`hyle/container/hash.h`）**：默认 `Hash<T>` 只覆盖整数/枚举、指针、浮点、
`string_view` 和 `std::string`；`+0.0` / `-0.0` 会按相等语义归一。owning string 若需
用 `string_view` / `string_ref` / C literal 异构查询，显式选透明 `StringHash` +
`StringEqual`；helper 不保存借用指针且调用为 `noexcept`。C pointer/array 按 NUL 结尾
解释，含嵌入 NUL 的 key 必须传显式长度的 view。所有未特化的用户自定义类型默认都
不可哈希，必须提供逐字段语义 hasher 或显式特化。`BytewiseHash<T>` 只是调用方 opt-in：
它要求 trivially-copyable 且具有唯一对象表示，但这个 trait 只能排除 padding/多重编码，
不能证明自定义 `Eq` 比较了每个 byte；调用方仍须保证 `Eq(a,b)` 蕴含对象 bytes 完全相同。
较长字节串由 `hash_bytes` 按长度和安全 ISA 选择库内 XXH3 或 wyhash。它是
进程内 table hash，不是跨版本/构建/CPU 稳定的 fingerprint。公开 API 不再提供
把私有 vendor 头拉入用户 TU 的 `hash_bytes_inline`：它只改善一个微基准调用 seam，
却破坏 header/static/shared/source 三种交付的一致性。真实热点优先缓存 hash 或批处理。

### 3.3 error —— Status / Result / 上下文

完整的语言无关选型、canonical code 和诊断边界见 [`error.md`](./error.md)；这里仅给集成时的
最短 C++ 路径：

| 情况 | 使用 |
|---|---|
| 只有成功/失败，调用者需要原因 | `Status` |
| 成功返回一个值/RAII 资源 | `Result<T>` |
| 批量独立步骤，保留第一个失败 | `Status::Update` |
| 必须按 CUDA/Vulkan/Win32 原生码恢复 | canonical code 后再查 `native_error()` |
| 明确接受 cleanup 失败 | `IgnoreError`（让丢弃意图可见） |
| allocator/pool pop/poll 的“暂时无值”是正常热路结果 | 指针/optional/小 enum，不强塞 Status |
| HYLE 自身不变量破坏、double free、错误生命周期 | `HYLE_FATAL_ASSERT`，不允许恢复 |
| 需要日志或 trace | 在最终消费边界交给 obs/宿主 logger，Status 自身不记录 |

code 选择遵循调用者该做什么：坏参数用 `InvalidArgument`，合法参数但状态不允许用
`FailedPrecondition`，OOM/配额/池耗尽用 `ResourceExhausted`，临时后端故障用
`Unavailable`，永久缺能力用 `Unimplemented`。叶子 message 保留原生原因；中间层只有
增加“用户正在做什么”的新语义时才加 context。不要解析 message 做程序分支。

**`Status`** 是 8 字节 tagged-word 错误载体（`sizeof(Status)==sizeof(void*)`）：
- OK（`rep_==0`，最常见）与无消息、无 detail 错误（内联码）都不碰堆；
- 带消息或 machine-readable native detail 的错误 malloc 一次，拷贝共享 backing；
- 普通逻辑按 canonical code 分支；只有后端专用恢复才读 domain/code，不解析 message。

```cpp
#include "hyle/error/status.h"
using hyle::Status; using hyle::StatusCode;

Status a = Status::Ok();                      // OK，零分配
Status b = Status::NotFound();                // 无消息，内联码，零分配
Status c = Status::Internal("shard 7 gone");  // 带消息，堆 rep（refcount）
Status d = std::move(c).WithContext("loading model");
                                               // "loading model: shard 7 gone"
Status vk = Status::FromNative(
    StatusCode::Unavailable, {hyle::NativeErrorDomain::Vulkan, -4},
    "device lost");

if (!d.ok()) d.code();          // StatusCode::Internal（16 个 canonical 非 OK 码）
d.IsInternal();                 // 每码一个谓词：IsNotFound() / IsCancelled() / ...
d.message();                    // string_view（无消息时为空），不拥有
char buf[128];
d.ToString(buf, sizeof buf);    // "Internal: loading model: shard 7 gone"

#include "hyle/error/format.h"  // 仅需要 C++ owning 文本的 TU 包含
std::string text = hyle::to_string(d);
```

**`Result<T>` = `std::expected<T, Status>`**（C++23，monadic `and_then`/`transform` 免费）。两个传播宏做 Rust `?`-式早返回，在返回 `Status` 或 `Result<T>` 的函数里都能用：

```cpp
#include "hyle/error/result.h"   // Result<T> + 两个早返回宏
hyle::Result<int> parse(std::string_view s);
hyle::Result<void> finish_step();

hyle::Result<int> do_thing() {
  HYLE_RETURN_IF_ERROR(step1());              // Status 非 OK 立即返回
  HYLE_RETURN_IF_ERROR(finish_step());        // Result<void> 也可直接传播
  HYLE_ASSIGN_OR_RETURN(int v, parse("42"));  // Result 失败则返回其 Status
  return v + 1;
}

hyle::Result<int> load() {
  return hyle::make_unexpected(hyle::Status::NotFound("model"));
}
```

`make_unexpected` 只接受非 OK Status；把 `Status::Ok()` 放进 error arm 是合同违反并会
fail-fast。宏对临时 Result 移动、对命名 lvalue Result 复制，不会暗中消费一个还要复用的
结果对象。

批量操作用 `overall.Update(next)` 保留第一个错误；明确接受清理失败时调用
`cleanup().IgnoreError()`。`HYLE_ASSIGN_OR_RETURN` 可以声明变量，因此会展开成多条语句；
放在 `if/else/for/while` 下时必须加 `{}`。

**约定（重要）**：`Status` 只表达**运营失败**（找不到、超时、权限、OOM…）。
自家 bug 不返 Status——用 `HYLE_FATAL_ASSERT`。外部后端 translator 在叶子保留
操作名、原生错误名和数值；中间层仅在增加用户语义时调用 `WithContext`。带消息构造
OOM 时保留错误码；增加 context OOM 时保留原消息。Status 传播不自动写日志，人读
诊断统一使用 §3.5 的 obs。

跨 DLL / C-FFI 消费者用纯 C 面（`hyle/c/error.h`）：普通失败返回 `hyle_status`，成功还要
交付句柄时使用首位 out 参数。mem 冷工厂就是这一形状，热 allocate/acquire 仍返回 NULL。
完成选型后，精确的 owning 生命周期与 ABI 映射见 [`c-error.md`](./c-error.md)。

### 3.4 sched —— 纤程 + 调度器
```cpp
#include "hyle/sched/scheduler.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
using namespace hyle::sched;

Scheduler sched(8, system_allocator(), /*queue_capacity=*/1024);   // 8 worker, 队列容量须 2 的幂
auto fast = sched.schedule(fn, arg);   // worker 栈：不可 yield/sleep/wait/block
auto full = sched.spawn_detached(fn, arg, kDefaultStackSize);  // 自持 guarded fiber：可挂起
sched.shutdown();                       // 排空后 join worker
```
需要返回值/结构化组合时用
`spawn_task(sched, alloc, Task<T>) -> JoinHandle<T>`；阻塞 syscall 用
`spawn_blocking`，不要占住 worker。`shutdown()` 允许已接纳任务派生后代并排空；
服务关闭先 `begin_shutdown()` + `drain_for()`。完整选择、生命周期和纯 C/FFI
用法见 [`c-sched.md`](./c-sched.md)。

⚠️ 见 `usage-contract.md`：fiber 体勿跨 `yield()` 持有 thread_local 派生值（work-stealing 会迁移线程）。

### 3.5 obs —— 日志

```cpp
#include "hyle/obs/log.h"
HYLE_LOG_INFO("loaded {} items", count);   // caller-side std::format，非流

hyle::obs::LogField fields[] = {{"request_id", "a7f2"}};
hyle::obs::LogRecord record{
    .level = hyle::obs::LogLevel::Warn,
    .message = "slow request",
    .fields = fields,
    .field_count = 1,
};
hyle::obs::log_record(record);             // 同步结构化 fan-out
```

`HYLE_LOG_*` 自动附带调用点 file/function/line；门控失败时参数不格式化。sink
callback 可能并发执行且不得抛异常。`remove_sink` 是 quiescent unregister：
返回后其他线程已不再使用 sink state，因此可以安全释放；callback 自摘除也不会
死锁，但 state 仍须活到当前 callback 返回。

trace 是另一条编译期面：

```cpp
#include "hyle/obs/trace.h"
HYLE_TRACE_ZONE_NAMED("parse");
HYLE_TRACE_COUNTER("requests", total);
HYLE_TRACE_ALLOC_NAMED(ptr, bytes, "arena");
```

默认 `HYLE_TRACE=OFF` 时宏参数不求值、生成零指令；`-DHYLE_TRACE=ON
-DHYLE_TRACY_DIR=...` 接 Tracy。纯 C/FFI 使用稳定 runtime trace 函数及
opaque sink，见 [`c-obs.md`](./c-obs.md)，不要直接绑定 C++ `LogSink` 或 Tracy 类型。

### 3.6 time —— 时钟、deadline 与定时器

```cpp
#include "hyle/time/clock.h"
#include "hyle/time/timer_queue.h"

using namespace std::chrono_literals;
auto started = hyle::time::Instant::now();
auto deadline = hyle::time::deadline_after(5ms);
auto elapsed = started.elapsed();
```

正常 deadline 永远使用 monotonic；realtime 只做墙钟时间戳；`fast_*` 是显式 profiling/
coarse 日志选项，不能用于 timeout。任意或混合 deadline 用 exact-ns `TimerQueue`；海量相同
duration 才用 `CommonTimeoutQueue`。二者都是 stable-address、single-owner 的侵入式数据
结构，只交付节点而不执行 callback。OS 线程 sleep 用 `time::sleep_*`，HYLE fiber 必须使用
`sched::sleep_*`，否则会阻塞 worker。

完整选择表、代码、周期 timer 策略和生命周期陷阱见 [`time.md`](./time.md)；纯 C/FFI 的
Status/opaque handle/visitor 映射见 [`c-time.md`](./c-time.md)；性能数字只见
[`../performance/benchmarks-time.md`](../performance/benchmarks-time.md)。

### 3.7 util —— Rng / Ref / Registry / DEFER
```cpp
#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/registry.h"
#include "hyle/util/scope_guard.h"
using hyle::util::RefCounted; using hyle::util::Ref; using hyle::util::make_ref;

hyle::util::Rng rng{12345};
auto id = rng.range_u64(0, 99);      // 无偏闭区间；PRNG，不可用于 token/key/nonce

struct Node : RefCounted<Node> {     // CRTP：计数内嵌对象，Ref<T> 就一个指针、无控制块
  int v; explicit Node(int v) : v(v) {}
};
Ref<Node> a = make_ref<Node>(42);    // 起始 +1，离开作用域自动 decref
Ref<Node> b = a;                     // 共享所有权（retain），析构各自 -1

FILE* fp = std::fopen("x", "r");
HYLE_DEFER(std::fclose(fp));         // 作用域退出时执行（C++23 仍无 std defer）
```
`make_ref` 收下初始 +1（adopt 语义）；要"保留已有裸指针"用 `retain_ref`，二者在调用点显式区分。
可复现并行随机流用一个确定 seed，然后每个 worker 取一份状态并调用一次
`jump()`；无需可复现时用 `thread_rng()`。Registry 适合启动期按名字注册 factory；它拥有
key/help，factory 与 visitor 都在锁外调用并可重入。初始化结束后调用 `freeze()`，高频 dispatch
缓存返回的 `FrozenView`，即可绕过普通入口的一次生命周期 atomic。不会失败或 NULL 已足够表达
结果时用 `Registry<T*>`；backend 初始化需要设备/驱动/配置诊断时用
`Registry<Result<T*>>`，missing 会得到 `NotFound`，creator error 原样传播。

纯 C、FFI 或跨编译器消费者不要绑模板/类布局，改用
[`hyle/c/util.h`](../../include/hyle/c/util.h) 的 RNG POD 和
Ref/Registry/SpinLock opaque handle；当前 47 个动态符号 + 10 个 inline convenience 的生命周期、
共享 Status/allocator 与 callback 合同见 [`c-util.md`](./c-util.md)。C 没有词法析构，旧 heap
ScopeGuard 已删除；用 goto-cleanup 或宿主语言 defer/finally。`type_name<T>`/`type_hash<T>`
依赖 C++ 静态类型，也有意不做运行时擦除。

**性能须知（何时 Ref 划算、何处别用）**：`Ref` 拷贝与析构需要原子引用计数；跨核共享时
cache-line ownership 转移会放大成本。具体点值见 performance，据此：
- **传参优先 `const Ref<T>&` 或裸 `T*`（借用），别按值传** —— 每次按值传拷贝都是一对 incref/decref。函数只是"看一眼"对象时收借用引用，只有真正要"延长生命周期/存起来"时才按值收 `Ref<T>`。这和 `shared_ptr` 的最佳实践一致。
- **别在紧循环里反复拷贝+析构同一个 `Ref`** —— 背靠背的两个原子(同 cache line)会串行化(第二个卡在第一个排空 store buffer 之后)，把原子延迟暴露到极致。需要在循环内多次访问同一对象时，循环外持一个 `Ref`、循环内用 `.get()` 或 `const&`。（性能文档那条 microbench 正是这个反模式，不代表真实开销。）
- **单线程、生命周期明确**时，直接裸指针 / 值语义 / `unique_ptr` 往往比任何引用计数都快 —— 引用计数是为"共享所有权 + 不确定谁最后释放"付的税。`Ref` 的价值在侵入式对象模型和省一次控制块分配，不在让引用计数本身变快；跨 ABI 使用的是独立的 type-erased `hyle_util_ref`，不是冻结 `Ref<T>` 布局。

### 3.8 io —— proactor 完成式 IO

典型用法是创建 `Scheduler` 与 `IoContext`，把连接绑定 owner worker，再用 `tcp_listen` /
`tcp_socket`、`UdpSocket`、`File` 或 `SharedMemory` 的 RAII/awaitable 接口。网络、文件、TLS、
HTTP 的完整 C++ recipe 与生命周期见 [`io.md`](./io.md)；本总指南只保留执行模型。

**前提（模型不同于传统，必须知道）**：HYLE io 是 **proactor（完成式）**，不是 reactor（就绪式）。传统写法是 `read()` 阻塞、或 `epoll` 等"可读"后自己 `read`；HYLE 是"**提交 IO → fiber 让出 → 内核做完 → 唤醒 fiber**"。你不"等可读"，你"提交读、完成时拿到已读好的数据"。这与 fiber 天然契合（提交即让出、完成即续跑），也是 C++ 标准网络的方向（WG21 P2300/P4100）。std 无异步 IO 等价物，故非"替换 std"，是净新增机制。

**后端取舍（自动选，SOTA-first；调用方通常无需关心）**：

| 后端 | 何时用 | 性质 |
|---|---|---|
| **io_uring**（Linux）| 默认，`init()` 总先试 | 真 proactor，最高性能；submit 纯用户态填 SQE + 一次 enter（SQPOLL 下零 syscall） |
| **IOCP**（Windows）| Windows 唯一 | OS 原生真 proactor |
| **epoll**（Linux）| **仅当 io_uring 不可用**（老内核 / `kernel.io_uring_disabled` sysctl / 容器 seccomp 挡） | reactor，内部模拟完成（就绪→内联 IO→completion），保住"io_uring 被禁的硬化容器/k8s"仍能跑 |

正式运行优先 io_uring/IOCP；epoll 是 Linux 上 io_uring 不可用时的兼容后端，
`HYLE_IO_FORCE_EPOLL` 仅用于测试。绑定 commit 的吞吐、扩展性和 syscall 结果见
[`../performance/benchmarks-io.md`](../performance/benchmarks-io.md)。


## 4. C-ABI 边界（跨 DLL / FFI）

HYLE 内部 C++23；稳定边界是 `include/hyle/c/*.h` 的纯 `extern "C"` **命名自由
函数 + POD/vtable 回调 + 不透明句柄**，不使用模块级 API 函数表，也不把 STL、
模板或异常跨 DLL。core/error/platform/mem/time/obs/simd/util/container/sched/io 已落地；
compute/graph 尚待各自模块收口，math 先审产品边界。统一 install/export 已完成。sched 的完整 FFI
合同见 [`c-sched.md`](./c-sched.md)，总规范见
[`c-abi-design.md`](../architecture/c-abi-design.md)。

## 5. 安全边界（必读补充）

### 3.9 platform —— OS 抽象（线程/VM/文件映射/fs/动态库/CPU 探测）

最底层 OS 封装，其它模块的地基；也可直接用。公开 owner 均为 move-only RAII；Win/Linux/mac
三平台的上一轮真机结果是历史输入，当前工作树是否重新签署只看
[`pre-release-readiness`](../milestones/pre-release-readiness.md)。

```cpp
// 虚拟内存：保留/提交分离 + guard page（fiber 栈、大缓冲的地基）
#include "hyle/platform/virtual_memory.h"
using namespace hyle::platform;
HYLE_ASSIGN_OR_RETURN(
    VmRegion r,
    VmRegion::reserve_guarded(64 * 1024));           // 低 1 页永久 no-access 守护页
HYLE_RETURN_IF_ERROR(r.commit_usable());             // 提交可用区为 RW
void* p = r.usable();                                // 首个可写字节（越界写→守护页 fault）

// 文件内存映射：RO / RW(回盘) / COW(私有)，对标 Rust memmap2
#include "hyle/platform/mapped_file.h"
HYLE_ASSIGN_OR_RETURN(
    MappedFile m,
    MappedFile::open("weights.bin", MappedFile::Mode::kReadOnly));
if (m.valid()) { const float* w = static_cast<const float*>(m.data()); /* 零拷贝读 */ }

// 文件系统 + 原子临时文件（返回打开的句柄，TOCTOU 安全）
#include "hyle/platform/fs.h"
#include "hyle/platform/convenience.h" // opt-in owning string/vector 查询
char path[1024];
HYLE_ASSIGN_OR_RETURN(
    CreatedTempFile temporary,
    create_temp_file(nullptr, "myapp", path, sizeof path)); // mkstemp/CREATE_NEW
FileHandle& file = temporary.file;
if (temporary.path_size >= sizeof path) { /* 路径被截断：关闭 handle 后处理 */ }
const char payload[] = "ready";
HYLE_ASSIGN_OR_RETURN(
    std::size_t wrote,
    file.write(payload, sizeof(payload) - 1));
HYLE_RETURN_IF_ERROR(file.close()); // 显式 close 才能观察延迟写回/关闭失败
HYLE_RETURN_IF_ERROR(set_permissions("f.txt", 0644));
HYLE_RETURN_IF_ERROR(create_hard_link("a.txt", "b.txt"));
// Windows 必须知道 target 类型；dangling directory link 要显式给 kind。
HYLE_RETURN_IF_ERROR(create_symlink(
    "future-dir", "dir-link", SymlinkKind::kDirectory));
HYLE_ASSIGN_OR_RETURN(std::string target, read_symlink("link"));
HYLE_ASSIGN_OR_RETURN(std::string temp_dir, temp_directory());
auto env = get_env_var("MYAPP_CACHE");
if (!env) {
  if (env.error().IsNotFound()) { /* 未设置：选择默认目录 */ }
  else { return env.error(); }
}
HYLE_RETURN_IF_ERROR(set_env_var("MYAPP_MODE", "fast"));

// 线程：ABI-clean 句柄 + name/affinity/stack/priority；默认 best-effort
#include "hyle/platform/thread.h"
ThreadOpts opts; opts.name = "worker"; opts.priority = ThreadPriority::kHigh;
auto t = Thread::spawn([state = std::move(state)] { run_worker(*state); }, opts);

// 动态库（RTLD_NOLOAD 附着已加载；macOS 传 install-name）+ CPU 特性
#include "hyle/platform/dynlib.h"
#include "hyle/platform/cpu.h"
HYLE_ASSIGN_OR_RETURN(DynLib lib, DynLib::open("plugin.so"));
HYLE_ASSIGN_OR_RETURN(auto fn, lib.symbol_as<int(*)()>("entry"));
if (cpu_features().avx2) { /* 运行期安全走 AVX2 */ }  // best_isa() 是 simd dispatch 的依据
const auto& cache = cache_info();        // L1D/L1I/L2/L3 bytes；未知级为 0
const auto& topo = cpu_topology();       // 缓存的系统 CPU 总量与可信 P/E 分类
HYLE_ASSIGN_OR_RETURN(
    std::vector<unsigned> available,
    available_cpu_ids());                // 完整进程配额；真实 id 可稀疏
ThreadOpts pinned_opts;
if (!available.empty()) pinned_opts.affinity = available.front();
pinned_opts.config_policy = ThreadConfigPolicy::kRequired;
auto pinned = Thread::spawn([&] { run_pinned(); }, pinned_opts); // 设置失败则不运行
```

需要少量 worker 优先使用强核时，用
`available_cpu_ids(CpuOrder::kPerformanceFirst, ...)` 或
`SchedulerOptions::worker_cpu_order`；该偏好保留所有可用 CPU，强核信息未知时保持 OS 顺序。
需要“只用强核”的硬合同则过滤 `available_cpu_info()` 的 `CpuKind::kPerformance`，不能把
unknown 当 performance。macOS 用 `SchedulerOptions::worker_priority = ThreadPriority::kHigh`
走 QoS，不猜 logical CPU 编号；两个 scheduler 选项都默认关闭。

**跨平台语义提醒**：`create_symlink` 在 Windows 需开发者模式/管理员（否则返回 `PermissionDenied`）；`set_env_var(name,"")` 在 Windows 删除变量、POSIX 保留空值；thread 配置默认 best-effort，需要保证时选 `ThreadConfigPolicy::kRequired` 或检查 `this_thread::set_*` 的 Status（macOS 硬 affinity 明确 `Unimplemented`，priority 走 QoS）。`cpu_topology()` 只报告系统明确提供的 P/E 分类；信息不足时把全部逻辑 CPU 归入 performance 并令 `hybrid=false`，不做频率启发式猜测。worker 数用动态 `available_cpu_count()`，绑核 id 用 `available_cpu_ids()`；二者都遵守进程 affinity/cpuset。纯 C / FFI 消费者用 `hyle/c/platform.h`（见 [`c-platform.md`](./c-platform.md)）。

本指南讲"正常用法"。**为 SOTA 性能保留的调用约束**（违反 = UB，非 HYLE bug）单独列在 [`usage-contract.md`](./usage-contract.md)（同目录）：fiber 跨 yield 的 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离等。集成前过一遍。

## 6. 反馈

问题报 https://github.com/user/axiomate/issues。
