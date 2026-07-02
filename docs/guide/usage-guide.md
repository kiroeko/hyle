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
| `platform/` | `hyle/platform/{thread,wait,dynlib,fs}.h` | 🔶 | thread+park、dynlib、fs/env（vm 待） |
| `container/` | `hyle/container/{swiss_table,carray,small_vector,string_ref,...}.h` | ✅ | SwissTable、CArray、SmallVector、lock-free 队列、string_ref |
| `time/` | `hyle/time/clock.h` | ✅ | 单调钟/墙钟/sleep（ns） |
| `util/` | `hyle/util/{ref_counted,registry,scope_guard,...}.h` | ✅ | `Ref<T>`、Registry、`DEFER`、type_name |
| `obs/` | `hyle/obs/log.h` | ✅ | level 门控 + sink 注册表日志 |
| `sched/` | `hyle/sched/{scheduler,fiber}.h` | 🔶 | fcontext 纤程 + work-stealing 调度（协程/取消 待） |
| `simd/` | `hyle/simd/{dispatch,vec,dtype,math}.h` | 🔶 | 运行期 ISA 探测（含纯 AVX 档 kAvx）+ 可移植 `vec<T,N>`（scalar/SSE4.2/AVX2 实跑；fp16/bf16 dtype + 向量化数学核（Highway/xsimd）done；NEON 写+编+objdump 验未跑存疑；AVX512 后端+ops kernel 已默认派发，objdump 双平台验真发 zmm，但本箱 AVX2-only 未实跑=存疑） |
| `io/` | `hyle/io/proactor.h` | 🔶 | proactor 完成收割接缝 + io_uring/IOCP 双后端;io_uring read/write 真 op + epoll 降级(SOTA-first)done;IOCP 真 op / net socket 待(io-02 IOCP/io-04) |
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
- **B. 开发/基准对照门**（`HYLE_BENCH_<名>`，共 **10 个 gate**；其中 7 个 clone 类各配一个 `HYLE_<X>_DIR` 路径变量，3 个 vendored 类无路径变量）——**只往 benchmark 加对照行，对出厂库零影响**，仅 `-DHYLE_BUILD_BENCH=ON` 时存在。集成方永远用不到（见 §1.2.1）。
- **2 个 CMake preset**（`bench` / `bench-all-sota`，唯一的 `CMakePresets.json`）——是"一组 `-D` 一次设好"的**预设包**，不是参数本身，`--preset` 选用。

> **命名约定**：凡产品参数都是 `HYLE_<名>` cache 变量，用 `-D` 设；bench 对照门统一带 `HYLE_BENCH_` 前缀，自我标记为 B 类。CMakePresets 里的 `bench` / `bench-all-sota` 是**预设包**（不是参数本身），`--preset` 选用——它们只是"把一组 `-D` 一次设好"的快捷方式，`bench-` 前缀表明展开的全是 B 类 bench 门。

**A 类——产品构建参数**（顶层 `cmake -S . -B build -D<OPT>=<VAL>` 设）。**默认值已为"生产使用者"调好**——通常你只需默认 + 选 Release/静态。

| 选项 | 默认 | 取值 | 何时改 |
|---|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`/`Release`/`RelWithDebInfo` | 性能库默认 Release(-O3)；要符号调试用 RelWithDebInfo |
| `HYLE_STATIC` | `OFF`(SHARED) | `ON`/`OFF` | 要静态链进你的二进制（无 DLL 分发）设 ON |
| `HYLE_BUILD_TESTS` | `ON` | `ON`/`OFF` | 当依赖集成时设 `OFF`（不编 GoogleTest，省时间） |
| `HYLE_BUILD_BENCH` | `OFF` | `ON`/`OFF` | 只有想跑微基准才开（拉 nanobench）。开了才解锁 B 类门 |
| `HYLE_WERROR` | `OFF` | `ON`/`OFF` | CI 用 ON；本地开发 OFF |
| `HYLE_SANITIZER` | `off` | `off`/`asan`/`tsan`/`ubsan` | 查内存/数据竞争/UB 时开（配 Debug/RelWithDebInfo）。**严格校验**：非法值直接 `FATAL_ERROR`（不静默降级为 off）；认编译器原生别名 `address`→asan、`thread`→tsan、`undefined`→ubsan、`none`→off。**MSVC 只支持 asan**（tsan/ubsan 在 MSVC 会 `FATAL_ERROR`，需 GCC/Clang，如 WSL） |

#### 1.2.1 B 类——开发/基准对照门（集成方忽略）

**仅当 `HYLE_BUILD_BENCH=ON` 时额外可用**（拿 HYLE 跟外部 SOTA 参考库赛跑，给 benchmark 加对照行；**不影响出厂库**）。最常用两个:
| 选项 | 默认 | 作用 |
|---|---|---|
| `HYLE_BENCH_ABSL` + `HYLE_ABSL_DIR` | `OFF` | SwissTable vs `absl::flat_hash_map`;SmallVector vs `absl::InlinedVector` |
| `HYLE_BENCH_HIGHWAY` | `OFF` | simd vec / dtype / math vs Google Highway（指向 `third_party/`，已 vendored，仅 toggle 对比行） |

> 还有 `HYLE_BENCH_{MOODYCAMEL,TBB,RIGTORP,XSIMD,SIMDUTF,BOOST,WYHASH,MARL}`(队列/分配器/math/utf8/侵入容器/hash 对标)。其中 `HYLE_BENCH_{HIGHWAY,XSIMD,SIMDUTF}` 指向 `third_party/` 已 vendored 库(不再需 sibling clone),其余仍是 opt-in sibling clone。**一键全开**:`scripts/setup-bench-libs.sh` clone 需要 sibling 的那些后 `cmake --preset bench-all-sota`。全清单与运行法见 [`scripts/README.md`](../../scripts/README.md)。
>
> **向量化数学核(Highway/xsimd)与 utf8 SIMD(simdutf)已 vendored 在 `third_party/`、无条件编进生产、默认全速**——无 `-D` 开关,运行期 `cpu_features().avx2` 门控(老 CPU 落 scalar 不 SIGILL)。上面的 `HYLE_BENCH_*` 仅 toggle bench 的 SOTA 对比行,不影响产品默认。

**集成方最小命令**（静态链、不编测试）：
```bash
cmake -S hyle -B build -DCMAKE_BUILD_TYPE=Release -DHYLE_STATIC=ON -DHYLE_BUILD_TESTS=OFF
cmake --build build
```

#### 1.2.2 CMake preset（共 2 个，都是 bench 预设包）

`CMakePresets.json` 只有 2 个 preset，**都属 B 类**（开发期跑基准用，不是产品参数）：

| preset | 作用 |
|---|---|
| `bench` | `HYLE_BUILD_BENCH=ON` + Release，HYLE 自身 + std/vendored 对照，无外部 clone |
| `bench-all-sota` | 继承 `bench`，再把全部 10 个 `HYLE_BENCH_*` gate 一次设 ON（clone 类需先 `setup-bench-libs.sh`） |

```bash
cmake --preset bench-all-sota && cmake --build --preset bench-all-sota
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
同一份源码，`-march` 决定它编成 SSE4.2(4 lane) 还是 AVX2(8 lane)。**这是你 99% 时候要用的方式。**

**(B) 分发单一二进制到未知 CPU（运行期选 ISA）**：若你分发一个要在各种 CPU 上跑的二进制，不能 `-march=native`（老 CPU 会 SIGILL）。模型是"每 ISA 编一份变体 + 启动时 CPUID 选最优"。`HYLE_SIMD_TARGET_SSE4`/`_AVX2`/`_AVX512` 宏就是给**变体 TU** 用的——同一份 kernel 源码编 N 份，每份配一个宏 + 对应 `-march`。**运行期探测用 `dispatch.h`**：

```cpp
#include "hyle/simd/dispatch.h"
if (hyle::simd::supports(hyle::simd::Isa::kAvx2)) { /* 调 AVX2 变体 */ }
// best_isa() 给最高可用档；isa_name() 打印
```
> **现状（simd-04 起）**：运行期变体选择**已就绪**。对 HYLE 自带的派发型 op（`hyle/simd/ops.h` 的 `dot_f32`/`saxpy_f32`、`hyle/simd/dtype_ops.h` 的 fp16/bf16 批量转换），**你什么都不用做**——它们内部按 `best_isa()` 选 fn-ptr（首次调用定一次、缓存），自动跑到最优 ISA：
> ```cpp
> #include "hyle/simd/ops.h"
> float d = hyle::simd::dot_f32(x, y, n);          // 自动选 scalar/SSE4.2/AVX2/AVX512
> // hyle::simd::active_variant() 报实际选中的档
> ```
> `dispatch.h` 的 `supports()`/`best_isa()` 是给你**自己的** kernel 做同样分发用的。模型是 fn-ptr 表（每调用一次指针间接；零间接的变体-DLL 模型是更重的后续里程碑）。**自动选择对 HYLE 自带 op 已可用,不再需要手动分发。**
>
> **AVX512 档存疑（诚实警示）**：`dot_f32`/`saxpy_f32` 的 AVX512 变体已接进派发——一台真 AVX512 CPU 会**默认**跑它。但 HYLE 开发箱是 AVX2-only(无 AVX512 硬件/SDE),该路径**只经 objdump 双平台验证真发 zmm 指令、未经执行对拍**(标 `存疑`)。这是平台优先库的取舍(详见 [`architecture/simd-design.md`](../architecture/simd-design.md) AVX512 节)。若你在 AVX512 机上用 HYLE 且要绝对保险,可暂以 `-march=native` 自编(走编译期路径、同样吃 AVX512 但避开运行期变体),或等 HYLE 在真 AVX512 硬件跑通对拍后摘掉存疑标。

### 2.1 哪些行为是编译期定的、哪些是运行期探测的（三条轴，别混）

平台相关行为分三类，**只有第 2 类走上面的"编译变体 + 运行期 dispatch"双重模型**：

| 轴 | 例 | 何时定 | 说明 |
|---|---|---|---|
| **平台/架构身份** | `HYLE_OS_*`/`HYLE_ARCH_*`/`HYLE_COMPILER_*`（`platform.h`）| **纯编译期** | 你为哪个 OS/架构编译就是哪个，无从运行期切换（syscall/ABI 在链接期已定）。不属 ISA dispatch 范畴。 |
| **`simd/` 的 ISA 后端** | vec 后端 / `ops.h` / `dtype_ops.h` | **双重**（编译变体 + 运行期选）| 完全符合 §2 双模型：`-march` 自编(A) 或 运行期 CPUID 选变体(B)。**探测档 ≠ kernel 档**：`best_isa()` 如实报 CPU 能力(含纯 AVX 档 `kAvx`,Sandy/Ivy Bridge 的 256-bit float),但若该档无专属 kernel(如无 avx-only kernel),派发会向下 clamp 到有 kernel 的最近档(kAvx→sse4)——绝不 SIGILL。|
| **`container/` SwissTable 的 SIMD 探测** | `probe.h` 的 SSSE3 `_mm_sign_epi8` 快路径 | **纯编译门控**（`HYLE_HAVE_SSSE3` 宏 + SSE2 兜底）| **故意的例外**：container 是 Tier 0,不依赖 `simd/`(Tier 1) 以免架构倒置,故内联一小段自包含探测并编译门控。**SSSE3 现为全局 baseline**(x86-64-v2)：CMake 顶层对 x86 GCC/Clang 全局加 `-mssse3`,MSVC x64 无条件,故 `HYLE_HAVE_SSSE3` 三平台默认成立、SSSE3 快路径默认生效。**无 SIGILL 风险**(SSE2 基线恒在兜底)。将来可选复用 `simd/ops` 做运行期探测,但不强耦合。|

**一句话**：`simd/` 的双重探测设计是真的、自洽的；`platform.h` 是天然纯编译期的另一根轴；`container/probe.h` 是**有意的纯编译门控例外**(Tier 0 不倒挂依赖)——SSSE3 现为全局 baseline,快路径默认生效,正确安全,但不做运行期 dispatch。

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

### 3.3 error —— Status / expected
```cpp
#include "hyle/error/status.h"
#include "hyle/error/result.h"   // ASSIGN_OR_RAISE / RETURN_NOT_OK
hyle::Status do_thing() {
  HYLE_RETURN_NOT_OK(step1());
  HYLE_ASSIGN_OR_RAISE(auto val, step2());   // 失败即提前返回其 Status
  return hyle::Status::Ok();
}
```
**约定**：自家 bug 不返 Status——用 `HYLE_FATAL_ASSERT`（fail-fast，见 `hyle/macros.h`）。Status 只表达运营失败。

### 3.4 sched —— 纤程 + 调度器
```cpp
#include "hyle/sched/scheduler.h"
#include "hyle/sched/fiber.h"
using namespace hyle::sched;
void task(void* arg) { /* ... */ Fiber::yield(); /* ... */ }

Scheduler sched(8, system_allocator(), /*queue_capacity=*/1024);   // 8 worker, 队列容量须 2 的幂
auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize, &task, arg);
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

### 3.8 io —— proactor 完成式 IO（🔶 收割接缝就绪，socket 待 io-04）

**⚠️ 现状：还不能用来做真网络 IO。** 已就绪的是 `proactor.h` 的**完成收割接缝** + io_uring/IOCP/epoll 后端 + io_uring `submit_read/write`；**net socket（accept/recv/send）是 io-04，未做**。这里只讲清模型与取舍，等 io-04 补完整用法。

**前提（模型不同于传统，必须知道）**：HYLE io 是 **proactor（完成式）**，不是 reactor（就绪式）。传统写法是 `read()` 阻塞、或 `epoll` 等"可读"后自己 `read`；HYLE 是"**提交 IO → fiber 让出 → 内核做完 → 唤醒 fiber**"。你不"等可读"，你"提交读、完成时拿到已读好的数据"。这与 fiber 天然契合（提交即让出、完成即续跑），也是 C++ 标准网络的方向（WG21 P2300/P4100）。std 无异步 IO 等价物，故非"替换 std"，是净新增机制。

**后端取舍（自动选，SOTA-first；调用方通常无需关心）**：

| 后端 | 何时用 | 性质 |
|---|---|---|
| **io_uring**（Linux）| 默认，`init()` 总先试 | 真 proactor，最高性能；submit 纯用户态填 SQE + 一次 enter（SQPOLL 下零 syscall） |
| **IOCP**（Windows）| Windows 唯一 | OS 原生真 proactor |
| **epoll**（Linux）| **仅当 io_uring 不可用**（老内核 / `kernel.io_uring_disabled` sysctl / 容器 seccomp 挡） | reactor，内部模拟完成（就绪→内联 IO→completion），保住"io_uring 被禁的硬化容器/k8s"仍能跑 |

**一句话**：正式运行走 io_uring/IOCP（SOTA），epoll 是"io_uring 被关掉时"的兜底降级，非默认——`init()` 自动按此顺序选，`HYLE_IO_FORCE_EPOLL` 仅测试用。**性能对标待 io-04 net 真实并发负载**（孤立测单 read/write 是测内核非测封装）。


## 4. C-ABI 边界（跨 DLL / FFI）

HYLE 内部 C++23，对外稳定边界走 **vtable 结构体 `xxx_api_t` + 不透明实例 + `make_X()/destroy_X()` 工厂**（分配责任锁在 DLL 内，禁客户端 `new/delete` HYLE 对象）。若你跨 DLL 边界用 HYLE 类型或做 FFI 绑定，遵此范式；纯 `extern "C"` C 头 shim 正在逐模块封（P6）。细节见 [`architecture/coding-conventions.md` §4](../architecture/coding-conventions.md)。

## 5. 安全边界（必读补充）

本指南讲"正常用法"。**为 SOTA 性能保留的调用约束**（违反 = UB，非 HYLE bug）单独列在 [`usage-contract.md`](./usage-contract.md)（同目录）：fiber 跨 yield 的 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离等。集成前过一遍。

## 6. 反馈

问题报 https://github.com/user/axiomate/issues。
