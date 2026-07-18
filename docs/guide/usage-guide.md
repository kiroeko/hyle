# HYLE 使用指南（面向库使用者）

> 这份文档面向**集成 HYLE 的开发者**——把 HYLE 当依赖用的人。它回答"怎么构建、有哪些开关、每个模块怎么调"。
>
> 区别于另外两份：[`architecture/`](../architecture/) 是设计内幕（为什么这么造，面向贡献者）；[`usage-contract.md`](./usage-contract.md)（同目录）是**为 SOTA 性能保留的调用安全边界**（哪些用法会触发 UB，必读但只讲"红线"）。本指南讲"正常怎么用"。
>
> **诚实状态**：HYLE 在建设中。下表标注每个模块就绪度；标 📋 的尚未实现，**别依赖**。

## 0. 模块就绪度（先看这张，别用没就绪的）

| 模块 | 头文件入口 | 状态 | 一句话能力 |
|---|---|---|---|
| `error/` | `hyle/error/status.h` `result.h` `error_handler.h` | ✅ | `Status` / `std::expected` 错误模型 + `ASSIGN_OR_RAISE` |
| `mem/` | `hyle/mem/allocator.h` 等 | ✅ | `Allocator` vtable（mimalloc backing）+ arena/pool/stack |
| `platform/` | `hyle/platform/{thread,wait,dynlib,fs,virtual_memory,numa,mapped_file,cpu}.h` | ✅ | thread(name/affinity/priority)+park、dynlib、fs(权限/symlink/hardlink/tempfile/env)、虚拟内存+guard、NUMA/大页、文件内存映射、CPU 特性探测。三平台真机验证 |
| `container/` | `hyle/container/{swiss_table,carray,small_vector,string_ref,...}.h` | ✅ | SwissTable、CArray、SmallVector、lock-free 队列、string_ref |
| `time/` | `hyle/time/clock.h` | ✅ | 单调钟/墙钟/sleep（ns） |
| `util/` | `hyle/util/{ref_counted,registry,scope_guard,...}.h` | ✅ | `Ref<T>`、Registry、`DEFER`、type_name |
| `obs/` | `hyle/obs/log.h` | ✅ | level 门控 + sink 注册表日志 |
| `sched/` | `hyle/sched/{scheduler,fiber,coro}.h` | 🔶 | fcontext 纤程 + work-stealing 调度 + fiber-aware 同步 + fiber sleep + 协程胶水(`Task<T>`/co_await 让出 fiber);取消/优雅关闭 待 |
| `simd/` | `hyle/simd/{dispatch,vec,dtype,math}.h` | 🔶 | 运行期 ISA 探测（含纯 AVX 档 kAvx）+ 可移植 `vec<T,N>`；scalar/SSE4.2/AVX2/AVX512/NEON 定宽后端与 SVE/RVV scalable 后端均已执行对拍，AVX512 由 Intel SDE 验证正确性和 kAvx512 派发，性能结论仍只接受真机 benchmark |
| `io/` | `hyle/io/{proactor,net,io_context,tcp,udp}.h` | ✅ | proactor 三后端 + read/write/socket/datagram 真 op(**无 stub**);co_await socket 层 + **shard-per-core**(每 worker 一 ring + fiber 亲和 spawn_on 绑核,share-nothing)+ **TcpListener/TcpStream + UdpSocket** RAII 封装;真 loopback TCP/UDP echo 三后端真测+TSan;echo+pipelined benchmark;park 合并(空闲真睡眠、eventfd 唤醒)。待 file/shmem(io-05)|
| `compute/` `graph/` | — | 📋 | 尚未开工 |

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
> **向量化数学核(Highway/xsimd)与 utf8 SIMD(simdutf)当前仍在 `third_party/` 无条件编进生产、默认全速**——这是 simd-20 的迁移中状态；外部 checkout 仅供 unchanged benchmark 对照。无产品 `-D` 开关，运行期 `cpu_features().avx2` 门控(老 CPU 落 scalar 不 SIGILL)。上面的 `HYLE_BENCH_*` 仅 toggle bench 的 SOTA 对比行,不影响产品默认。

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
- **当前边界（诚实）**：这验证的是 HYLE 的 **C++ API 跨 dllexport/import 边界**。**纯 `extern "C"` C-ABI shim（扁平 C 头、无名字修饰）是 P6,尚未做**——所以现在是"C++ ABI 跨 DLL",不是"C ABI"。要把 HYLE 当纯 C 库给非 C++ 调用方,等 P6。

## 2. SIMD 的编译变体模型（`HYLE_SIMD_TARGET_*` 是什么）

常用 batch 能力、开发期调用示例和 close-01 C ABI 迁移边界集中在
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

`hypot_in_range_f32_batch` 不在热路径逐元素检查契约。一个简单的**防溢出充分条件**是每个输入都满足 `|x| <= sqrt(FLT_MAX/2) ≈ 1.30438e19`；若输入可能接近上限、包含无穷/NaN，或必须保留极小非正规数，请用完整版本。`dst` 可与 `a` 或 `b` **完全同址**原位写回，但其他部分重叠不支持。真机 `bench-stat -n15`：AVX2为0.140 vs 0.281 ns/op（约 **2.03×**），Apple NEON为0.220 vs 0.454 ns/op（约 **2.07×**）。f64 direct-formula 试验仅统计打平，因此没有发布名不副实的 `hypot_in_range_f64_batch`。

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

mem/ 提供一组**专用分配器**,各自为一种分配模式设计。**选对了 5-10x 赢通用 malloc;选错了反而更慢**(例:单线程却用为多线程共享设计的无锁池,白付每次原子 CAS)。按"分配模式"对号入座:

| 你的场景 | 用 | 为什么 / 别用啥 |
|---|---|---|
| **通用大小、单纯想要快 malloc** | `system_allocator()`(SystemAllocator) | 内联直调 mimalloc,零接缝。不确定用啥就用它。 |
| **一个阶段内分配很多、整批一次性释放**(一次 taskgraph pass、一个请求的临时对象) | `StackAllocator` / `Arena` | 纯指针 bump,**无 per-object free**,reset()/析构整体释放。**单线程**(一个 bump 游标天生串行)。`StackAllocator` 自有 slab + `mark/rewind`(LIFO 嵌套作用域回滚);`Arena` 借 `MemoryArena` 的一块池块。 |
| **单线程、同一种固定大小对象反复 alloc/free**(对象池) | `GrowablePool<T>` | 单线程 free-list,无原子,3.5x 赢裸 mi。**这是单线程对象池的正解**。 |
| **多线程共享一个有界固定池**(每线程都从同一池借/还,如 fiber 栈池) | `MemoryArena` / `ObjectPool<T>` | 无锁分片 Treiber 栈,跨线程安全。**⚠️ 只在真需要跨线程共享时用**——它每次 alloc/free 各付一次原子 CAS,**单线程用它就是选错工具**(慢裸 mi ~3x,该换 `GrowablePool`)。`ObjectPool<T>` = `MemoryArena` + typed placement-new/dtor。 |

**一句话决策树**:单纯 malloc→`SystemAllocator`;批量分配整批丢→`Arena`/`StackAllocator`;单线程对象池→`GrowablePool`;**多线程共享**有界池→`MemoryArena`/`ObjectPool`。无锁池(`MemoryArena`/`ObjectPool`)的原子开销只有在真跨线程共享时才划算,单线程场景务必用 `GrowablePool` 或 bump 分配器。详细性能数据见 [`performance/benchmarks-mem-container.md`](../performance/benchmarks-mem-container.md) mem 节。

### 3.2 container —— SwissTable（对标 absl 追平/超，快 std 2.4-5.3x）
```cpp
#include "hyle/container/swiss_table.h"
using hyle::container::SwissTable;
SwissTable<uint64_t, uint64_t> m(system_allocator());
m.insert(42, 100);
if (uint64_t* v = m.find(42)) { /* *v == 100 */ }
```
其余：`CArray<T>`（侵入式数组）、`SmallVector<T,N>`、`string_ref`（视图字符串）、`BoundedQueue`（lock-free MPMC）。

**哈希器（`hyle/container/hash.h`）**：默认 `Hash<T>` / `hash_bytes` 已 best-of-breed——整数/指针/≤16B POD 走内联 mix（无调用），字节串按长度选最快核（17-128B→XXH3、>128B→wyhash）。**绝大多数用法（map key）无需关心**。仅一个进阶开关：`hash_bytes` 对 **>16B 字节串**走出线核（把 7490 行 xxhash.h 挡在你的 TU 外），代价是一次 ~1ns 调用（在 **短串** ~64B 上占比可见,长串摊平、≤16B key 不走此路）。**若** profiler 显示你有一个热循环在 hash 大量 **短字节串（17~128B）**、且你愿意让该 TU 吃 xxhash.h 的编译期开销,用 opt-in 头消掉那 1ns:
```cpp
#include "hyle/container/hash_inline.h"   // 仅此 TU 拉入 xxhash/wyhash（编译期代价）
uint64_t h = hyle::container::hash_bytes_inline(p, n);  // 与 hash_bytes 结果 bit-identical，无 seam
```
`hash_bytes_inline` 覆盖**全部长度**（≤16 内联 mix / 17-128 XXH3 / >128 wyhash，与 `hash_bytes` 同分派）——**刻意只做一个全长度函数，不按长度拆成多个 API**：≤16B 段默认 `hash_bytes` 本就内联（inline 变体与它字节级同码、零差别），长串段 seam 已摊平（省的 ~1ns 是噪声），**真有收益的只有 17~128B 短串这一段**。为"对称"给每个长度带都配一个 inline 入口只会制造和默认完全相同的冗余 API + 选择困惑，违背"不为对称加复杂度"——所以就这一个函数，用户按"要不要消 seam"而非"按长度"决策。**适用边界（重要）**：`hash_bytes_inline` 是 **header-inline、非 C-ABI 导出符号**，要求你**源码内嵌 / 静态链接 HYLE**（它把 xxhash.h/wyhash.h 编进你的 TU）；**纯 DLL / C-FFI 消费者用不了它，用默认 `hash_bytes`**（那才是 C-ABI 稳定、DLL 可调的）。默认 `hash.h` 不含它——不 include 就零代价。不确定就用 `hash_bytes`。

### 3.3 error —— Status / Result / 诊断

**`Status`** 是 8 字节 tagged-word 错误载体（`sizeof(Status)==sizeof(void*)`），三态零开销分级：
- OK（`rep_==0`，最常见）与**无消息**错误（内联码）——**都不碰堆**；
- 只有**带消息**的错误才 malloc 一次（引用计数，拷贝共享同一 backing）。

```cpp
#include "hyle/error/status.h"
using hyle::Status; using hyle::StatusCode;

Status a = Status::Ok();                      // OK，零分配
Status b = Status::NotFound();                // 无消息，内联码，零分配
Status c = Status::Internal("shard 7 gone");  // 带消息，堆 rep（refcount）

if (!c.ok()) c.code();          // StatusCode::Internal（16 个 abseil/gRPC canonical 码）
c.IsInternal();                 // 每码一个谓词：IsNotFound() / IsCancelled() / ...
c.message();                    // string_view（无消息时为空），不拥有
char buf[128];
c.ToString(buf, sizeof buf);    // "Internal: shard 7 gone"；snprintf 式，返回完整长度
```

**`Result<T>` = `std::expected<T, Status>`**（C++23，monadic `and_then`/`transform` 免费）。两个传播宏做 Rust `?`-式早返回，在返回 `Status` 或 `Result<T>` 的函数里都能用：

```cpp
#include "hyle/error/result.h"   // Result<T> + HYLE_RETURN_NOT_OK / HYLE_ASSIGN_OR_RAISE
hyle::Result<int> parse(std::string_view s);

hyle::Result<int> do_thing() {
  HYLE_RETURN_NOT_OK(step1());               // step1() 返回 Status；非 OK 立即返回
  HYLE_ASSIGN_OR_RAISE(int v, parse("42"));  // 成功绑定 v，失败提前返回其 Status
  return v + 1;
}
```

**约定（重要）**：`Status` 只表达**运营失败**（找不到、超时、权限、OOM…）。**自家 bug 不返 Status**——用 `HYLE_FATAL_ASSERT`（fail-fast，见 `hyle/macros.h`）。区分外部边界（该返 Status）vs 调用方契约违反（该 assert）。带消息构造在 OOM 时优雅降级：保留码、丢消息，绝不 abort。

**诊断通道**（人读的日志/告警，区别于机读的 Status 返回值）：

```cpp
#include "hyle/error/error_handler.h"
HYLE_WARN("retry %d/%d", i, n);      // 格式化后送当前 handler
HYLE_ERROR("connect failed: %s", e);
HYLE_FATAL("unrecoverable");         // 报告后 trap（永不返回）

// 软断言：release 存活（不像 <cassert> 被 NDEBUG 编掉），失败时报告而非 UB，求值为 bool
if (!HYLE_ASSERT(p != nullptr)) return Status::InvalidArgument("null p");

// 安装自定义 sink（默认写 stderr）：把 HYLE 诊断接进宿主 logger。
hyle::ErrorRecord rec;   // 零分配计数器：测试里捕获诊断
rec.install();
// ... 触发一些 HYLE_WARN/ERROR ...
rec.count(hyle::Severity::Error);   // 统计
rec.restore();
```

跨 DLL / C-FFI 消费者用纯 C 面（`hyle/c/error.h`），见 [`c-error.md`](./c-error.md)。

### 3.4 sched —— 纤程 + 调度器
```cpp
#include "hyle/sched/scheduler.h"
#include "hyle/sched/fiber.h"
using namespace hyle::sched;
void task(void* arg) { /* ... */ Fiber::yield(); /* ... */ }

Scheduler sched(8, system_allocator(), /*queue_capacity=*/1024);   // 8 worker, 队列容量须 2 的幂
auto f = std::make_unique<Fiber>(kDefaultStackSize, &task, arg);  // 栈=guarded FiberStack
sched.spawn(f.get());
sched.shutdown();   // 排空（等所有 fiber 完成）再 join worker
```
⚠️ 见 `usage-contract.md`：fiber 体勿跨 `yield()` 持有 thread_local 派生值（work-stealing 会迁移线程）。

### 3.5 obs —— 日志
```cpp
#include "hyle/obs/log.h"
HYLE_LOG_INFO("loaded {} items", count);   // caller-side std::format，非流
```

### 3.6 time —— 单调钟 / 计时
```cpp
#include "hyle/time/clock.h"
using hyle::time::Instant;
auto t0 = Instant::now();           // 单调钟快照（不受系统墙钟调整影响）
// ... 干活 ...
uint64_t ns = t0.elapsed_ns();      // 自 t0 起经过的纳秒
hyle::time::sleep_for_ms(10);       // 睡 10ms（另有 sleep_for_us / sleep_for_ns）
```
`Instant` 全 header-inline（算术不过 DLL），适合热路径打点。

**两个单调钟,按需选(默认用 `monotonic_ns`)**:

| | `monotonic_ns()`(默认) | `fast_monotonic_ns()`(opt-in) |
|---|---|---|
| 实现 | QPC / clock_gettime(OS 原语) | rdtsc 直读 + 定点换算 |
| 延迟 | ~12 ns | **~7 ns(Win)/~6.8 ns(Linux)** 快 1.7-2.0x |
| 正确性 | ✅ 永远对、唤醒安全、跨核严格单调 | ⚠️ sleep 唤醒/VM 迁移可能跳变;跨核迁移可能小幅回退(无 max 钳制,差值会下溢) |
| 可用于 deadline/超时 | ✅ **是**(sched 定时器键它) | ❌ **否** |
| 硬件要求 | 无 | invariant TSC(CPUID 探测);无则透明回退默认钟 |
| 首次调用 | 无 | 一次性校准 ~几 ms |

**决策**:
- **要算超时/deadline/sched 定时 → 永远 `monotonic_ns()`**(fast 版的跳变会让超时灾难性出错)。
- **高频采样/profiling 打点,且偶发唤醒后毛刺可容忍 → `fast_monotonic_ns()`**;最好把线程钉核,或只对"同核近距离两次读"求差。
- 拿不准 → `monotonic_ns()`。它已和 std::chrono 持平(~12ns),够快且零心智负担。

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
| **任意/混合超时时长,海量 arm/cancel**(fiber sleep 任意时长、通用定时) | `TimerWheel`(分层时间轮) | 6 层×64 槽 tokio 移植,任意时长 O(1) 摊还 arm/cancel、无墓碑。**通用默认,拿不准用它**。churn 负载赢 min-heap 2-3x(双平台)。 |
| **海量定时器共享同一固定时长**(百万连接都是 30s 超时) | `CommonTimeoutQueue` | 单条有序侵入链:相同时长 deadline 天然递增 → arm=O(1) 尾插、到期=O(1) 头摘、cancel=O(1) 摘链。**这类负载赢 TimerWheel arm+expire 3.0-3.7x / arm+cancel 1.6-1.75x(双平台)**——省掉轮的 6KB 槽数组 + 每次 level/slot 数学。**一个队列只管一种时长**。 |
| ~~定时器少、需精确"最近到期"、取消少~~(**非 HYLE 负载**) | ~~min-heap~~ | HYLE 未提供。堆的 O(log n) arm/cancel + 惰性墓碑污染 expire,对"海量 arm/cancel≫真到期"的 HYLE 负载是错的工具(churn 实测输轮 2-3x)。列此仅为说明为何不选。 |

**一句话决策树**:任意时长 → `TimerWheel`(默认);海量**相同**时长 → `CommonTimeoutQueue`。多种固定时长并存 = 建多个 `CommonTimeoutQueue`,由**调用方**按时长路由(像 libevent 的 `common_timeout_queues[]`——路由是消费者的事,不在数据结构里)。

> **就绪度**:`TimerWheel` 已接入 sched(fiber `sleep_for_ns`/`sleep_until_ns` 用它)。`CommonTimeoutQueue` 已实现 + 双平台测试 + benchmark 验证,**但其目标消费者(io reactor 的连接超时)尚未落地**——它是"就位待接入"的结构,不是已在服役。性能数据见 [`performance/benchmarks-sched.md`](../performance/benchmarks-sched.md) 时间轮节。

### 3.7 util —— Ref<T> 侵入式引用计数 + DEFER
```cpp
#include "hyle/util/ref_counted.h"
#include "hyle/util/scope_guard.h"
using hyle::util::RefCounted; using hyle::util::Ref; using hyle::util::make_ref;

struct Node : RefCounted<Node> {     // CRTP：计数内嵌对象，Ref<T> 就一个指针、无控制块
  int v; explicit Node(int v) : v(v) {}
};
Ref<Node> a = make_ref<Node>(42);    // 起始 +1，离开作用域自动 decref
Ref<Node> b = a;                     // 共享所有权（retain），析构各自 -1

FILE* fp = std::fopen("x", "r");
HYLE_DEFER(std::fclose(fp));         // 作用域退出时执行（C++23 仍无 std defer）
```
`make_ref` 收下初始 +1（adopt 语义）；要"保留已有裸指针"用 `retain_ref`，二者在调用点显式区分。

**性能须知（何时 Ref 划算、何处别用）**：`Ref` 拷贝 = 1 次 incref（原子加），析构 = 1 次 decref（原子减），与 `std::shared_ptr` 算法同工（各 2 个 locked RMW）。原子 RMW 不是免费的（~几 ns，且跨核共享时更贵）。据此：
- **传参优先 `const Ref<T>&` 或裸 `T*`（借用），别按值传** —— 每次按值传拷贝都是一对 incref/decref。函数只是"看一眼"对象时收借用引用，只有真正要"延长生命周期/存起来"时才按值收 `Ref<T>`。这和 `shared_ptr` 的最佳实践一致。
- **别在紧循环里反复拷贝+析构同一个 `Ref`** —— 背靠背的两个原子(同 cache line)会串行化(第二个卡在第一个排空 store buffer 之后)，把原子延迟暴露到极致。需要在循环内多次访问同一对象时，循环外持一个 `Ref`、循环内用 `.get()` 或 `const&`。（性能文档那条 microbench 正是这个反模式，不代表真实开销。）
- **单线程、生命周期明确**时，直接裸指针 / 值语义 / `unique_ptr` 往往比任何引用计数都快 —— 引用计数是为"共享所有权 + 不确定谁最后释放"付的税。`Ref` 的价值在跨 DLL ABI 稳定（无控制块）和侵入式省一次分配，不在让引用计数本身变快。

### 3.8 io —— proactor 完成式 IO（🔶 co_await + shard-per-core + TcpListener/Stream）

**⚠️ 现状:真 TCP/UDP echo 已能跑通,走 shard-per-core（性能天花板架构）。** 典型用法:起 `Scheduler` + `IoContext io(sched, num_workers)`;`tcp_listen(io, owner, "0.0.0.0", port)` 得 `TcpListener`,`co_await ls.accept(owner)` 得 `TcpStream`,再 `co_await conn.read(buf,len)` / `co_await conn.write(buf,len)`;客户端 `tcp_socket` + `co_await s.connect(&ss,len)`。UDP:`udp_bind(io, owner, host, port)` / `udp_socket(io, owner)` 得 `UdpSocket`,`co_await sock.recv_from(buf,len,&peer)` / `co_await sock.send_to(buf,len,&dest,dlen)`。**四层**:① `proactor.h` 收割接缝 + io_uring/IOCP/epoll 三后端 + 全 submit 真 op(含 recvfrom/sendto);② `net.h` co_await awaitable(`IoOperation` slot 承 result + Waker 唤醒);③ `io_context.h` **每 worker 一 proactor**,`bind_fd(worker,fd)` 绑核 + hook 接 worker 循环收割本核 ring;④ `tcp.h`/`udp.h` RAII `TcpStream/TcpListener/UdpSocket`。**shard-per-core 用法要点**:连接 fd `bind_fd` 到 owner 核,服务它的 fiber 用 `Scheduler::spawn_on(owner, fiber)` **钉该核**——submit/reap/resume 全在一个核,零跨核同步。**取舍**:连接分片跨核=负载可能倾斜(热连接压一个核),这是 share-nothing 的固有取舍(seastar 同)。空闲 worker 走 park 合并(真睡眠 + eventfd 跨线程唤醒,非 1ms 轮询)。**未做**:file/shmem(io-05)。这里讲清模型与取舍。

**前提（模型不同于传统，必须知道）**：HYLE io 是 **proactor（完成式）**，不是 reactor（就绪式）。传统写法是 `read()` 阻塞、或 `epoll` 等"可读"后自己 `read`；HYLE 是"**提交 IO → fiber 让出 → 内核做完 → 唤醒 fiber**"。你不"等可读"，你"提交读、完成时拿到已读好的数据"。这与 fiber 天然契合（提交即让出、完成即续跑），也是 C++ 标准网络的方向（WG21 P2300/P4100）。std 无异步 IO 等价物，故非"替换 std"，是净新增机制。

**后端取舍（自动选，SOTA-first；调用方通常无需关心）**：

| 后端 | 何时用 | 性质 |
|---|---|---|
| **io_uring**（Linux）| 默认，`init()` 总先试 | 真 proactor，最高性能；submit 纯用户态填 SQE + 一次 enter（SQPOLL 下零 syscall） |
| **IOCP**（Windows）| Windows 唯一 | OS 原生真 proactor |
| **epoll**（Linux）| **仅当 io_uring 不可用**（老内核 / `kernel.io_uring_disabled` sysctl / 容器 seccomp 挡） | reactor，内部模拟完成（就绪→内联 IO→completion），保住"io_uring 被禁的硬化容器/k8s"仍能跑 |

**一句话**：正式运行走 io_uring/IOCP（SOTA），epoll 是"io_uring 被关掉时"的兜底降级，非默认——`init()` 自动按此顺序选，`HYLE_IO_FORCE_EPOLL` 仅测试用。**性能对标已补**（io-04 echo 64 连接×64 round-trip + pipelined 512 流高扇出：reinject+COOP 优化后单 worker **平裸 epoll、单核赢 asio 1.25x/libuv 1.52x**、shard-per-core 扩展 4w 赢 epoll 2.25x、park 合并空闲 syscall 421→3）——见 [`performance/benchmarks-io.md`](../performance/benchmarks-io.md)。


## 4. C-ABI 边界（跨 DLL / FFI）

HYLE 内部 C++23，对外稳定边界走 **vtable 结构体 `xxx_api_t` + 不透明实例 + `make_X()/destroy_X()` 工厂**（分配责任锁在 DLL 内，禁客户端 `new/delete` HYLE 对象）。若你跨 DLL 边界用 HYLE 类型或做 FFI 绑定，遵此范式；纯 `extern "C"` C 头 shim 正在逐模块封（P6）。细节见 [`architecture/coding-conventions.md` §4](../architecture/coding-conventions.md)。

## 5. 安全边界（必读补充）

### 3.9 platform —— OS 抽象（线程/VM/文件映射/fs/动态库/CPU 探测）

最底层 OS 封装，其它模块的地基；也可直接用。全部 move-only RAII、跨平台（Win/Linux/mac 三平台真机验证）。

```cpp
// 虚拟内存：保留/提交分离 + guard page（fiber 栈、大缓冲的地基）
#include "hyle/platform/virtual_memory.h"
using namespace hyle::platform;
VmRegion r = VmRegion::reserve_guarded(64 * 1024);  // 低 1 页永久 no-access 守护页
r.commit_usable();                                   // 提交可用区为 RW
void* p = r.usable();                                // 首个可写字节（越界写→守护页 fault）

// 文件内存映射：RO / RW(回盘) / COW(私有)，对标 Rust memmap2
#include "hyle/platform/mapped_file.h"
MappedFile m = MappedFile::open("weights.bin", MappedFile::Mode::kReadOnly);
if (m.valid()) { const float* w = static_cast<const float*>(m.data()); /* 零拷贝读 */ }

// 文件系统 + 原子临时文件（返回打开的句柄，TOCTOU 安全）
#include "hyle/platform/fs.h"
char path[1024];
OwnedFd fd = create_temp_file(nullptr, "myapp", path, sizeof path);  // mkstemp/CREATE_NEW
set_permissions("f.txt", 0644);  create_hard_link("a.txt", "b.txt");
char tgt[4096];  read_symlink("link", tgt, sizeof tgt);  // buffer-fill：返回完整长度

// 线程：ABI-clean 句柄 + name/affinity/stack/priority（best-effort，需权限时软降级）
#include "hyle/platform/thread.h"
ThreadOpts opts; opts.name = "worker"; opts.affinity_cpu = 0; opts.priority = ThreadPriority::kHigh;
auto t = Thread::spawn(&entry, ctx, opts);

// 动态库（RTLD_NOLOAD 附着已加载；macOS 传 install-name）+ CPU 特性
#include "hyle/platform/dynlib.h"
#include "hyle/platform/cpu.h"
DynLib lib = DynLib::open("plugin.so");  auto fn = lib.symbol_as<int(*)()>("entry");
if (cpu_features().avx2) { /* 运行期安全走 AVX2 */ }  // best_isa() 是 simd dispatch 的依据
const auto& cache = cache_info();        // L1D/L1I/L2/L3 bytes；未知级为 0
const auto& topo = cpu_topology();       // 逻辑 CPU 与可信的 P/E 分类
```

**跨平台语义提醒**：`create_symlink` 在 Windows 需开发者模式/管理员（无则软失败）；`set_env_var(name,"")` 在 Windows 删除变量、POSIX 保留空值；thread priority/affinity 是 best-effort（无权限软降级）。`cpu_topology()` 只报告系统明确提供的 P/E 分类；信息不足时把全部逻辑 CPU 归入 performance 并令 `hybrid=false`，不做频率启发式猜测。纯 C / FFI 消费者用 `hyle/c/platform.h`（见 [`c-platform.md`](./c-platform.md)）。

本指南讲"正常用法"。**为 SOTA 性能保留的调用约束**（违反 = UB，非 HYLE bug）单独列在 [`usage-contract.md`](./usage-contract.md)（同目录）：fiber 跨 yield 的 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离等。集成前过一遍。

## 6. 反馈

问题报 https://github.com/user/axiomate/issues。
