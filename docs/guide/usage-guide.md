# HYLE 使用指南（面向库使用者）

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
| `container/` | `hyle/container/{swiss_table,carray,small_vector,string_ref}.h` | flat map、连续/内联容器、字符串和并发队列 |
| `time/` | `hyle/time/{clock,timer_wheel,common_timeout_queue}.h` | 时钟、deadline、通用与固定时长 timer source |
| `util/` | `hyle/util/{ref_counted,registry,scope_guard,random}.h` | 引用所有权、registry、scope guard 与 RNG |
| `obs/` | `hyle/obs/{log,trace}.h` | 结构化日志与可选 trace |
| `sched/` | `hyle/sched/{scheduler,fiber,coro,sync}.h` | M:N fiber/coroutine、work stealing、同步、取消与 blocking offload |
| `simd/` | `hyle/simd/{dispatch,ops,math,native}.h` | runtime/native/exact SIMD；层级选择见 [`simd-usage.md`](./simd-usage.md) |
| `io/` | `hyle/io/{io_context,tcp,udp,file,shared_memory,tls,http,http2}.h` | proactor 网络/文件/共享内存和协议适配 |
| `compute/` | `hyle/compute/{device,buffer,stream,kernel,device_pool}.h` | host/CUDA/Vulkan device、buffer、completion 与 kernel |
| `graph/` | `hyle/graph/{taskgraph,parallel,data_parallel}.h` | 增量任务图和数据并行编排 |

## 1. 构建与集成

### 1.1 用 CMake 集成（推荐）

HYLE 是标准 CMake 工程，每个模块导出 `hyle::<module>` 别名 target：

```cmake
add_subdirectory(hyle)            # 或 FetchContent / find_package(未来)
target_link_libraries(your_app PRIVATE hyle::container hyle::mem hyle::simd)
```

链接 `hyle::<module>` 自动带上 `include/` 头路径与依赖模块（如 `hyle::container` 自动拉 `hyle::mem`）。包含头一律 `#include "hyle/<module>/<header>.h"`。

### 1.2 构建选项全集（这是"开关太多"的答案）

HYLE 的构建开关分**两个范畴**，别混（速查：**A 类产品参数 5 个 + B 类 bench 门 10 个 + 2 个 preset**）：

- **A. 产品构建参数**（`HYLE_<名>`，全大写下划线 = CMake cache 变量，共 **5 个**：`HYLE_STATIC` / `HYLE_BUILD_TESTS` / `HYLE_BUILD_BENCH` / `HYLE_WERROR` / `HYLE_SANITIZER`；外加 CMake 内建 `CMAKE_BUILD_TYPE`）——影响**出厂的库**：静态/动态、sanitizer、CI 严格度。这是你集成时唯一要管的。（注：SIMD 不暴露硬件开关——默认编全部 ISA 变体、运行期 CPUID 选最优，见 §2。）
- **B. 开发/基准对照门**（`HYLE_BENCH_<名>`，共 **10 个 gate**；其中 7 个 clone 类各配一个 `HYLE_<X>_DIR` 路径变量）——**只往 benchmark 加对照行，对出厂库零影响**。此外，任何 `-DHYLE_BUILD_BENCH=ON` 都须有 `HYLE_BENCH_VENDOR_DIR` 指向固定版 Highway/xsimd 对照 checkout；Windows/mac 有默认，Linux/WSL 显式设置。集成方永远用不到（见 §1.2.1）。
- **2 个 CMake preset**（`bench` / `bench-all-sota`，唯一的 `CMakePresets.json`）——是"一组 `-D` 一次设好"的**预设包**，不是参数本身，`--preset` 选用。

> **命名约定**：凡产品参数都是 `HYLE_<名>` cache 变量，用 `-D` 设；bench 对照门统一带 `HYLE_BENCH_` 前缀，自我标记为 B 类。CMakePresets 里的 `bench` / `bench-all-sota` 是**预设包**（不是参数本身），`--preset` 选用——它们只是"把一组 `-D` 一次设好"的快捷方式，`bench-` 前缀表明展开的全是 B 类 bench 门。

**A 类——产品构建参数**（顶层 `cmake -S . -B build -D<OPT>=<VAL>` 设）。**默认值已为"生产使用者"调好**——通常你只需默认 + 选 Release/静态。

| 选项 | 默认 | 取值 | 何时改 |
|---|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`/`Release`/`RelWithDebInfo` | 性能库默认 Release(-O3)；要符号调试用 RelWithDebInfo |
| `HYLE_STATIC` | `OFF`(SHARED) | `ON`/`OFF` | 要静态链进你的二进制（无 DLL 分发）设 ON |
| `HYLE_BUILD_TESTS` | `ON` | `ON`/`OFF` | 当依赖集成时设 `OFF`（不编 GoogleTest，省时间）。**同时派生编译宏 `HYLE_TESTING`**：ON 时注入所有模块库(经 `hyle_testing` INTERFACE target)，解锁 `#if defined(HYLE_TESTING)` 门控的**测试专用接口**；OFF(集成/出厂)时这些接口不编译、出厂库零残留(见下"测试专用接口")|
| `HYLE_BUILD_BENCH` | `OFF` | `ON`/`OFF` | 只有想跑微基准才开（拉 nanobench）。开了才解锁 B 类门 |
| `HYLE_WERROR` | `OFF` | `ON`/`OFF` | CI 用 ON；本地开发 OFF |
| `HYLE_SANITIZER` | `off` | `off`/`asan`/`tsan`/`ubsan` | 查内存/数据竞争/UB 时开（配 Debug/RelWithDebInfo）。**严格校验**：非法值直接 `FATAL_ERROR`（不静默降级为 off）；认编译器原生别名 `address`→asan、`thread`→tsan、`undefined`→ubsan、`none`→off。**MSVC 只支持 asan**（tsan/ubsan 在 MSVC 会 `FATAL_ERROR`，需 GCC/Clang，如 WSL） |

#### 测试专用接口（`HYLE_TESTING` 门控——集成方零残留）

有的接口**只为测试存在**：让测试够到一条生产调用方永不需要的内部路径（例：一个"可控 Event"——先报 N 次 NotReady 再 Ready，让 host 后端无需真 GPU 就能驱动 GpuPoller 的 suspend→wake 全链路）。这类接口**绝不进出厂库**，用 `HYLE_TESTING` 宏门控。

- **宏来源**：`HYLE_TESTING` 由 `HYLE_BUILD_TESTS` 派生（`cmake/HyleTesting.cmake` 的 `hyle_testing` INTERFACE target，每个模块 PUBLIC 链接，同 `hyle_trace` 的 gate 模式）。`HYLE_BUILD_TESTS=ON` → 定义（生产库源码 + 测试 exe + 消费者对头文件的编译一致，无 ODR 隐患）；集成方 `-DHYLE_BUILD_TESTS=OFF` → 不定义，门控块整个消失，**出厂库零 test hook**。
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
> 产品 SIMD math 不再依赖 in-tree Highway/xsimd；固定外部 checkout 只用于 benchmark 对照。
> `simdutf` 仍作为 UTF-8 产品核 vendored。`HYLE_BENCH_*` 只切换对照行，不改变产品路径。

**集成方最小命令**（静态链、不编测试）：
```bash
cmake -S hyle -B build -DCMAKE_BUILD_TYPE=Release -DHYLE_STATIC=ON -DHYLE_BUILD_TESTS=OFF
cmake --build build
```

#### 1.2.2 CMake preset（共 2 个，都是 bench 预设包）

`CMakePresets.json` 只有 2 个 preset，**都属 B 类**（开发期跑基准用，不是产品参数）：

| preset | 作用 |
|---|---|
| `bench` | `HYLE_BUILD_BENCH=ON` + Release，独立 `build-bench` cache；先有 `HYLE_BENCH_VENDOR_DIR` 中固定版 Highway/xsimd |
| `bench-all-sota` | 继承 `bench`，再把全部 10 个 `HYLE_BENCH_*` gate 一次设 ON（先 fetch SIMD vendors，clone 类再跑 `setup-bench-libs.sh`） |

```bash
# Windows/macOS（vendor 默认目录）
scripts/fetch-bench-vendors.sh
scripts/setup-bench-libs.sh
cmake --preset bench-all-sota && cmake --build --preset bench-all-sota

# Linux/WSL（vendor 在 ext4；generic preset 不替你选择 GCC14）
scripts/fetch-bench-vendors.sh --dir "$HOME/vendor"
CC=gcc-14 CXX=g++-14 cmake --preset bench-all-sota -DHYLE_BENCH_VENDOR_DIR="$HOME/vendor"
cmake --build --preset bench-all-sota
```

> 集成方不需要任何 preset——它们纯为开发期跑基准服务。产品构建只用 §1.2 的 A 类 `-D` 参数。

### 1.3 工具链要求

C++23。GCC 14+ / Clang 18+ / MSVC 19.43+（VS 17.13+）。CMake 3.25+。低于此 CMake 直接报错（`std::expected` 等需要）。

### 1.4 把 HYLE 编成什么库（动态 vs 静态）+ 怎么发布

HYLE 默认编成**共享库**（`.so`/`.dll`，每模块一个 `hyle_<mod>`）；`-DHYLE_STATIC=ON` 编成**静态库**（`.a`/`.lib`）。两种产物都落在 `<build>/bin/`（运行时库、导入库、静态档统一在那）。怎么选：

| 你的场景 | 选 | 命令 | 为什么 |
|---|---|---|---|
| **多个进程/插件共用一份 HYLE**；想热替换实现；想锁住 ABI 边界 | **动态**（默认）| `cmake -B build -DCMAKE_BUILD_TYPE=Release` | 一份 `.so`/`.dll` 多处加载；`HYLE_<MOD>_API` 的 dllexport/import 在此生效，跨边界 ABI 稳定 |
| **单一可执行、想零运行时依赖**；嵌入式/容器精简；想让链接器删未用符号 | **静态** | `cmake -B build -DHYLE_STATIC=ON -DCMAKE_BUILD_TYPE=Release` | 折进你的二进制，没有"找不到 .so"问题；`HYLE_<MOD>_API` 退化为空 |

**分发型 vs 自用型构建（与 §2 SIMD 模型直接相关）**：
- **分发给未知 CPU** → 用**默认 baseline**（别加全局 `-march=native`）。这样 SIMD 的 baseline dispatcher 不会发出 AVX2 指令，老 CPU 不 SIGILL；热路径由运行期 CPUID dispatch 选变体（§2-B）。这是发布 `.so`/`.dll` 的正确姿势。
- **自编自用（已知就这台机器）** → 加 `-march=native`（或 `/arch:AVX2`），见 §2-A，零 dispatch、极致。**但这种二进制别发给更老的 CPU。**

最小发布命令（动态、分发型、不带测试）：
```bash
cmake -S hyle -B build -DCMAKE_BUILD_TYPE=Release -DHYLE_BUILD_TESTS=OFF
cmake --build build           # 产物在 build/bin/  (libhyle_*.so / hyle_*.dll + .lib)
```

### 1.5 消费 HYLE 发布的二进制（链接预构建库）

如果 HYLE 已被编好（上面那步），下游工程**不必再从源码 add_subdirectory**——直接链接预构建产物即可。这条路也是验证 `HYLE_<MOD>_API` 导出边界的方式。完整可跑示例见 [`samples/04_prebuilt_lib/`](../../samples/04_prebuilt_lib/)，要点：

```cmake
# 1) 找到预构建库（动态的导入库 / 静态档都在 <hyle-build>/bin）
find_library(HYLE_SIMD_LIB NAMES hyle_simd PATHS "<hyle-build>/bin" REQUIRED NO_DEFAULT_PATH)

add_executable(app main.cc)
target_compile_features(app PRIVATE cxx_std_23)          # HYLE 头要 C++23
target_include_directories(app PRIVATE "<hyle-repo>/include")
target_link_libraries(app PRIVATE "${HYLE_SIMD_LIB}")
```
- **静态消费**：HYLE 用 `-DHYLE_STATIC=ON` 编的话，下游也要 `target_compile_definitions(app PRIVATE HYLE_STATIC)`，让 `HYLE_<MOD>_API` 同样退化为空（对上库的编译方式）。**注意传递依赖**：静态链 `hyle_mem` 还需链 `mimalloc_static`（mem 的私有后端）；多模块静态消费要按依赖序把各模块 + 其依赖都列上——这类自动化是未来 `install()`+`find_package(hyle)`（P6 打包）要做的。`hyle_simd` 无此传递依赖，最干净。
- **动态消费运行期**：Linux 让 loader 找到 `.so`（`RPATH` 指向 `<hyle-build>/bin` 或装进系统库路径）；Windows 把 `hyle_*.dll` 放在 exe 同目录或 `PATH`。
- **当前边界（诚实）**：error/platform/mem/time/obs/simd/util/container 已有纯 `extern "C"` C ABI；其他模块仍在
  close-01 队列。SIMD 外部调用必须用 `hyle/c/simd.h`，不能把 C++ mangled symbol 当 FFI。

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
**默认不用传 allocator**:头内联容器(`SmallVector`/`BoundedQueue`/`ShardedQueue`/`WsFifoQueue`/`WorkStealingDeque`/`ConcurrentVector`)默认参 `mem::SystemAllocator`——编译期绑定、内联直调 mimalloc、零接缝(`[[no_unique_address]]` 不占字节)。要 arena/mock/跨 DLL 才**显式 opt-in** 传运行时 `Allocator` 句柄。
直接用 mem/ API 或 type-erased 容器(`SwissTable`/`CArray`/`string_buf`,出线核服务所有 T,按设计收运行时句柄)时,传 `system_allocator()`(mimalloc backing):
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
| **通用大小、单纯想要快 malloc** | `system_allocator()`(SystemAllocator) | 内联直调 mimalloc,零接缝。不确定用啥就用它。 |
| **一个阶段内分配很多、整批一次性释放**(一次 taskgraph pass、一个请求的临时对象) | `Region` / `PooledRegion` / `StackAllocator` | 纯指针 bump，**无 per-object free**，clear/reset/析构整批释放；分别是 growable owned、借一个 SharedPool block、owned slab + mark/rewind。 |
| **单线程、同一种固定大小对象反复 alloc/free**(对象池) | `GrowablePool<T>` | 单线程 free-list、无原子、按需增长。这是单线程对象池的正解。 |
| **多线程共享有界固定池，容量必须准确** | `SharedPool` / `ObjectPool<T>` | **硬容量默认**：无 TLS 隐藏，completed free 立即共享；稳定双快照避免分片扫描假耗尽。正常 32-bit tagged CAS，tag 饱和的 shard 永久切入 serialized fallback，绝不回绕。 |
| **worker-local 复用热点，池有充足余量** | `SharedPoolCache` / `ObjectPoolCache<T>` over hard pool | **软容量 opt-in**：cache storage/capacity/batch 显式；本地 free→alloc 零原子。handoff/idle 前由 owner `publish()`。 |

**一句话决策树**：单纯 malloc→`SystemAllocator`；request/frame 整批丢→`Region`；借池中固定 scratch block→`PooledRegion`；LIFO scratch→`StackAllocator`；单线程对象池→`GrowablePool`；多线程共享且 quota 必须准确→`SharedPool`/`ObjectPool`；测得 owner-local churn 热点且能接受软容量→在 hard pool 上显式挂 cache。

hard pool 与 cache 是组合层，但合同不同：`SharedPool` 不隐藏 block；`SharedPoolCache` 可能在其他 owner cache 仍有 free block 时返回 `nullptr`。不要把 cache 性能冒充 hard pool 性能。最新数字与对手形状见 [`performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)；调用红线见 [`usage-contract.md`](./usage-contract.md) §2.1。

### 3.2 container —— SwissTable
```cpp
#include "hyle/container/swiss_table.h"
using hyle::container::SwissTable;
SwissTable<uint64_t, uint64_t> m(system_allocator());
m.insert(42, 100);
if (uint64_t* v = m.find(42)) { /* *v == 100 */ }
```
其余：`CArray<T>`（侵入式数组）、`SmallVector<T,N>`、`string_ref`（视图字符串）、`BoundedQueue`（lock-free MPMC）。

**哈希器（`hyle/container/hash.h`）**：默认 `Hash<T>` / `hash_bytes` 让整数/指针/≤16B POD 走内联 mix，字节串按长度选择 XXH3 或 wyhash。**绝大多数 map key 用法无需关心**。`hash_bytes` 对 >16B 字节串使用 out-of-line kernel，避免让大型 vendor 头进入用户 TU；若 profiler 显示短字节串哈希的调用边界是热点，可选择 inline 入口，以编译时间换运行时调用消除。具体阈值和实测只见 performance：
```cpp
#include "hyle/container/hash_inline.h"   // 仅此 TU 拉入 xxhash/wyhash（编译期代价）
uint64_t h = hyle::container::hash_bytes_inline(p, n);  // 与 hash_bytes 结果 bit-identical，无 seam
```
`hash_bytes_inline` 覆盖**全部长度**（≤16 内联 mix / 17-128 XXH3 / >128 wyhash，与 `hash_bytes` 同分派）——**刻意只做一个全长度函数，不按长度拆成多个 API**：≤16B 段默认 `hash_bytes` 本就内联（inline 变体与它字节级同码、零差别），长串段 seam 已摊平（省的 ~1ns 是噪声），**真有收益的只有 17~128B 短串这一段**。为"对称"给每个长度带都配一个 inline 入口只会制造和默认完全相同的冗余 API + 选择困惑，违背"不为对称加复杂度"——所以就这一个函数，用户按"要不要消 seam"而非"按长度"决策。**适用边界（重要）**：`hash_bytes_inline` 是 **header-inline、非 C-ABI 导出符号**，要求你**源码内嵌 / 静态链接 HYLE**（它把 xxhash.h/wyhash.h 编进你的 TU）；**纯 DLL / C-FFI 消费者用不了它，用默认 `hash_bytes`**（那才是 C-ABI 稳定、DLL 可调的）。默认 `hash.h` 不含它——不 include 就零代价。不确定就用 `hash_bytes`。

### 3.3 error —— Status / Result / 上下文

完整的语言无关选型、canonical code 和诊断边界见 [`error.md`](./error.md)；这里仅给集成时的
最短 C++ 路径：

| 情况 | 使用 |
|---|---|
| 只有成功/失败，调用者需要原因 | `Status` |
| 成功返回一个值/RAII 资源 | `Result<T>` |
| 批量独立步骤，保留第一个失败 | `Status::Update` |
| 明确接受 cleanup 失败 | `IgnoreError`（让丢弃意图可见） |
| allocator/pool pop/poll 的“暂时无值”是正常热路结果 | 指针/optional/小 enum，不强塞 Status |
| HYLE 自身不变量破坏、double free、错误生命周期 | `HYLE_FATAL_ASSERT`，不允许恢复 |
| 需要日志或 trace | 在最终消费边界交给 obs/宿主 logger，Status 自身不记录 |

code 选择遵循调用者该做什么：坏参数用 `InvalidArgument`，合法参数但状态不允许用
`FailedPrecondition`，OOM/配额/池耗尽用 `ResourceExhausted`，临时后端故障用
`Unavailable`，永久缺能力用 `Unimplemented`。叶子 message 保留原生原因；中间层只有
增加“用户正在做什么”的新语义时才加 context。不要解析 message 做程序分支。

**`Status`** 是 8 字节 tagged-word 错误载体（`sizeof(Status)==sizeof(void*)`），三态零开销分级：
- OK（`rep_==0`，最常见）与**无消息**错误（内联码）——**都不碰堆**；
- 只有**带消息**的错误才 malloc 一次（引用计数，拷贝共享同一 backing）。

```cpp
#include "hyle/error/status.h"
using hyle::Status; using hyle::StatusCode;

Status a = Status::Ok();                      // OK，零分配
Status b = Status::NotFound();                // 无消息，内联码，零分配
Status c = Status::Internal("shard 7 gone");  // 带消息，堆 rep（refcount）
Status d = std::move(c).WithContext("loading model");
                                               // "loading model: shard 7 gone"

if (!d.ok()) d.code();          // StatusCode::Internal（16 个 canonical 非 OK 码）
d.IsInternal();                 // 每码一个谓词：IsNotFound() / IsCancelled() / ...
d.message();                    // string_view（无消息时为空），不拥有
char buf[128];
d.ToString(buf, sizeof buf);    // "Internal: loading model: shard 7 gone"

#include "hyle/error/format.h"  // 仅需要 C++ owning/stream 输出的 TU 包含
std::string text = hyle::to_string(d);
std::cerr << d;                 // 应用侧 opt-in；HYLE 核心不会自动写日志
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
#include "hyle/sched/join.h"
using namespace hyle::sched;

Scheduler sched(8, system_allocator(), /*queue_capacity=*/1024);   // 8 worker, 队列容量须 2 的幂
auto fast = sched.schedule(fn, arg);   // worker 栈：不可 yield/sleep/wait/block
auto full = sched.spawn_detached(fn, arg);  // 自持 guarded fiber：可 yield/sleep/wait
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
using hyle::time::Instant;
auto t0 = Instant::now();                         // 单调钟快照
auto deadline = hyle::time::deadline_after_ns(5'000'000);
// ... 干活 ...
uint64_t ns = t0.elapsed_ns();                    // 将来值安全钳到 0
uint64_t left = hyle::time::remaining_ns(deadline);
hyle::time::sleep_until_ns(deadline);             // 绝对单调 deadline
```

`monotonic_ns()`、`realtime_ns()` 和纯算术 deadline helper 都是 header-inline；普通
C++ 调用以及 sched/io 热路径不付 DLL 跳转。加法、`sleep_for_us/ms` 的单位换算采用
饱和语义，极大超时不会回绕成已经过期。

**两个单调钟,按需选(默认用 `monotonic_ns`)**:

| | `monotonic_ns()`(默认) | `fast_monotonic_ns()`(opt-in) |
|---|---|---|
| 实现 | QPC / clock_gettime(OS 原语) | rdtsc 直读 + 定点换算 |
| 延迟 | 与同工作量 `std::chrono` 同档 | 通常比默认钟更低 |
| 正确性 | ✅ 永远对、唤醒安全、跨核严格单调 | ⚠️ sleep 唤醒/VM 迁移可能跳变;跨核迁移可能小幅回退(无 max 钳制,差值会下溢) |
| 可用于 deadline/超时 | ✅ **是**(sched 定时器键它) | ❌ **否** |
| 硬件要求 | 无 | invariant TSC(CPUID 探测);无则透明回退默认钟 |
| 首次调用 | 无 | 一次性校准 ~几 ms |

**决策**:
- **要算超时/deadline/sched 定时 → 永远 `monotonic_ns()`**(fast 版的跳变会让超时灾难性出错)。
- **高频采样/profiling 打点,且偶发唤醒后毛刺可容忍 → `fast_monotonic_ns()`**;最好把线程钉核,或只对"同核近距离两次读"求差。
- 拿不准 → `monotonic_ns()`。它与 `std::chrono` 做同样工作，且是 sched/io deadline 的统一时基。

```cpp
// 例:profiling 热路径打点(可容忍偶发毛刺,要最低读取成本)
uint64_t a = hyle::time::fast_monotonic_ns();
hot_path();
uint64_t span = hyle::time::fast_monotonic_ns() - a;  // 同核近距离两读,安全
```

#### 3.6.1 定时器结构选型（海量超时:选对结构,别都塞一个堆）

管理海量超时(io 超时、连接空闲超时、fiber sleep)时,`time/` 提供**两种定时器数据结构**,都满足同一 `TimerSource` 接口(`schedule`/`cancel`/`advance`/`next_deadline_ns`),**按负载选、编译期定、零虚调用开销**。这不是重复造轮子——SOTA 的 libevent 同样是两套并存(min-heap + `common_timeout` 队列),因为**一个结构吃不下所有定时负载**。

| 你的场景 | 用 | 为什么 / 别用啥 |
|---|---|---|
| **任意/混合超时时长,海量 arm/cancel**(fiber sleep 任意时长、通用定时) | `TimerWheel`(分层时间轮) | 8 层×64 槽覆盖完整 `uint64_t` 纳秒域，arm/cancel 摊还 O(1)，取消立即摘链。**通用默认，拿不准用它**。对真实 indexed eager-erase min-heap，当前双平台 arm/cancel、expire、churn 均领先。 |
| **海量定时器共享同一固定时长**(百万连接都是 30s 超时) | `CommonTimeoutQueue` | 单条有序侵入链：同一时长的 deadline 天然递增，arm/expire/cancel 都是 O(1)。一个队列只管一种时长；混合时长必须用 wheel。跨架构性能边界见 performance 总账。 |
| **定时器较少或已有 heap 生态** | indexed min-heap | HYLE 不提供；heap 可以做到 eager O(log n) keyed erase，不能把对手建模成惰性墓碑。轮的价值是大量任意 deadline 下更低的常数和 O(1) cancel，不是声称 heap 必然有墓碑。 |

**一句话决策树**:任意时长 → `TimerWheel`(默认);海量**相同**时长 → `CommonTimeoutQueue`。多种固定时长并存 = 建多个 `CommonTimeoutQueue`,由**调用方**按时长路由(像 libevent 的 `common_timeout_queues[]`——路由是消费者的事,不在数据结构里)。

`TimerWheel` 适合任意 deadline；`CommonTimeoutQueue` 只适合同一 duration 且 deadline
非递减的 workload。实现与消费者状态见 milestones，性能数据见
[`../performance/benchmarks-time.md`](../performance/benchmarks-time.md)。

`TimerEntry` 是侵入式节点，地址必须稳定；不能复制/移动，且析构前必须从 source 或
到期输出链中移除。source 只移动到期节点，不调用回调：

```cpp
#include "hyle/time/timer_wheel.h"

hyle::time::TimerWheel wheel{hyle::time::monotonic_ns()};
hyle::time::TimerEntry timer{my_context};
hyle::time::TimerList fired;

wheel.schedule_after(&timer, 25'000'000);  // 饱和 now + 25 ms
wheel.advance(hyle::time::monotonic_ns(), fired);
while (auto* entry = fired.pop_front()) {
  consume(entry->data);
}
```

纯 C、FFI 或跨编译器消费者使用 opaque handle、visitor 和 pointer/POD 接口，见
[`c-time.md`](./c-time.md)；不要把上述 C++ 类布局当作稳定 ABI。

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
`jump()`；无需可复现时用 `thread_rng()`。Registry 适合按名字注册 factory，所有
表访问线程安全，factory 在锁外调用而可重入；遍历 visitor 在锁内，不能重入。

纯 C、FFI 或跨编译器消费者不要绑模板/类布局，改用
[`hyle/c/util.h`](../../include/hyle/c/util.h) 的 RNG POD 和
Ref/ScopeGuard/Registry/SpinLock opaque handle；46 个可调用入口、生命周期和
callback 合同见 [`c-util.md`](./c-util.md)。`type_name<T>`/`type_hash<T>` 依赖
C++ 静态类型，是唯一有意不做运行时擦除的 util 能力。

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
模板或异常跨 DLL。error/platform/mem/time/obs/simd/util/container/sched 已落地；
上层 io/compute/graph 和统一 install/export 仍在 close-01 收口。sched 的完整 FFI
合同见 [`c-sched.md`](./c-sched.md)，总规范见
[`c-abi-design.md`](../architecture/c-abi-design.md)。

## 5. 安全边界（必读补充）

### 3.9 platform —— OS 抽象（线程/VM/文件映射/fs/动态库/CPU 探测）

最底层 OS 封装，其它模块的地基；也可直接用。全部 move-only RAII、跨平台（Win/Linux/mac 三平台真机验证）。

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
char tgt[4096];
HYLE_ASSIGN_OR_RETURN(std::size_t target_size,
                      read_symlink("link", tgt, sizeof tgt));
char temp_dir[1024];
HYLE_ASSIGN_OR_RETURN(std::size_t temp_size,
                      temp_directory(temp_dir, sizeof temp_dir));
char value[256];
auto env = get_env_var("MYAPP_CACHE", value, sizeof value);
if (!env) {
  if (env.error().IsNotFound()) { /* 未设置：选择默认目录 */ }
  else { return env.error(); }
}
HYLE_RETURN_IF_ERROR(set_env_var("MYAPP_MODE", "fast"));

// 线程：ABI-clean 句柄 + name/affinity/stack/priority；默认 best-effort
#include "hyle/platform/thread.h"
ThreadOpts opts; opts.name = "worker"; opts.priority = ThreadPriority::kHigh;
auto t = Thread::spawn(&entry, ctx, opts);

// 动态库（RTLD_NOLOAD 附着已加载；macOS 传 install-name）+ CPU 特性
#include "hyle/platform/dynlib.h"
#include "hyle/platform/cpu.h"
HYLE_ASSIGN_OR_RETURN(DynLib lib, DynLib::open("plugin.so"));
HYLE_ASSIGN_OR_RETURN(auto fn, lib.symbol_as<int(*)()>("entry"));
if (cpu_features().avx2) { /* 运行期安全走 AVX2 */ }  // best_isa() 是 simd dispatch 的依据
const auto& cache = cache_info();        // L1D/L1I/L2/L3 bytes；未知级为 0
const auto& topo = cpu_topology();       // 缓存的系统 CPU 总量与可信 P/E 分类
HYLE_ASSIGN_OR_RETURN(std::size_t worker_count, available_cpu_count());
unsigned first_cpu;
HYLE_ASSIGN_OR_RETURN(
    std::size_t available,
    available_cpu_ids(&first_cpu, 1));   // 完整进程配额；写入首个真实 id
ThreadOpts pinned_opts;
if (available != 0) pinned_opts.affinity_cpu = first_cpu; // id 可稀疏
pinned_opts.config_policy = ThreadConfigPolicy::kRequired;
auto pinned = Thread::spawn(&entry, ctx, pinned_opts); // 设置失败则 entry 不运行
```

**跨平台语义提醒**：`create_symlink` 在 Windows 需开发者模式/管理员（否则返回 `PermissionDenied`）；`set_env_var(name,"")` 在 Windows 删除变量、POSIX 保留空值；thread 配置默认 best-effort，需要保证时选 `ThreadConfigPolicy::kRequired` 或检查 `this_thread::set_*` 的 Status（macOS 硬 affinity 明确 `Unimplemented`，priority 走 QoS）。`cpu_topology()` 只报告系统明确提供的 P/E 分类；信息不足时把全部逻辑 CPU 归入 performance 并令 `hybrid=false`，不做频率启发式猜测。worker 数用动态 `available_cpu_count()`，绑核 id 用 `available_cpu_ids()`；二者都遵守进程 affinity/cpuset。纯 C / FFI 消费者用 `hyle/c/platform.h`（见 [`c-platform.md`](./c-platform.md)）。

本指南讲"正常用法"。**为 SOTA 性能保留的调用约束**（违反 = UB，非 HYLE bug）单独列在 [`usage-contract.md`](./usage-contract.md)（同目录）：fiber 跨 yield 的 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离等。集成前过一遍。

## 6. 反馈

问题报 https://github.com/user/axiomate/issues。
