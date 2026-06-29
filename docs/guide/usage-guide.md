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
| `simd/` | `hyle/simd/{dispatch,vec,dtype,math}.h` | 🔶 | 运行期 ISA 探测 + 可移植 `vec<T,N>`（scalar/SSE4.2/AVX2 实跑；fp16/bf16 dtype + sleef 数学核 done；NEON/AVX512 写+编+objdump 验、未跑存疑） |
| `io/` `compute/` `graph/` | — | 📋 | 尚未开工 |

## 1. 构建与集成

### 1.1 用 CMake 集成（推荐）

HYLE 是标准 CMake 工程，每个模块导出 `hyle::<module>` 别名 target：

```cmake
add_subdirectory(hyle)            # 或 FetchContent / find_package(未来)
target_link_libraries(your_app PRIVATE hyle::container hyle::mem hyle::simd)
```

链接 `hyle::<module>` 自动带上 `include/` 头路径与依赖模块（如 `hyle::container` 自动拉 `hyle::mem`）。包含头一律 `#include "hyle/<module>/<header>.h"`。

### 1.2 构建选项全集（这是"开关太多"的答案）

所有开关在顶层 `cmake -S . -B build -D<OPT>=<VAL>` 设。**默认值已为"生产使用者"调好**——通常你只需默认 + 选 Release/静态。

| 选项 | 默认 | 取值 | 何时改 |
|---|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`/`Release`/`RelWithDebInfo` | 性能库默认 Release(-O3)；要符号调试用 RelWithDebInfo |
| `HYLE_STATIC` | `OFF`(SHARED) | `ON`/`OFF` | 要静态链进你的二进制（无 DLL 分发）设 ON |
| `HYLE_BUILD_TESTS` | `ON` | `ON`/`OFF` | 当依赖集成时设 `OFF`（不编 GoogleTest，省时间） |
| `HYLE_BUILD_BENCH` | `OFF` | `ON`/`OFF` | 只有想跑微基准才开（拉 nanobench） |
| `HYLE_WERROR` | `OFF` | `ON`/`OFF` | CI 用 ON；本地开发 OFF |
| `HYLE_SANITIZER` | `off` | `off`/`asan`/`tsan`/`ubsan` | 查内存/数据竞争/UB 时开（配 Debug/RelWithDebInfo） |

**仅当 `HYLE_BUILD_BENCH=ON` 时额外可用**（对标 SOTA 参考库，需本地 clone）：
| 选项 | 默认 | 作用 |
|---|---|---|
| `HYLE_BENCH_ABSL` + `HYLE_ABSL_DIR` | `OFF` | SwissTable vs `absl::flat_hash_map` 对标 |
| `HYLE_BENCH_HIGHWAY` + `HYLE_HIGHWAY_DIR` | `OFF` | simd vec vs Google Highway 对标 |

**集成方最小命令**（静态链、不编测试）：
```bash
cmake -S hyle -B build -DCMAKE_BUILD_TYPE=Release -DHYLE_STATIC=ON -DHYLE_BUILD_TESTS=OFF
cmake --build build
```

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
> float d = hyle::simd::dot_f32(x, y, n);          // 自动选 scalar/SSE4.2/AVX2
> // hyle::simd::active_variant() 报实际选中的档
> ```
> `dispatch.h` 的 `supports()`/`best_isa()` 是给你**自己的** kernel 做同样分发用的。模型是 fn-ptr 表（每调用一次指针间接；零间接的变体-DLL 模型是更重的后续里程碑）。**自动选择对 HYLE 自带 op 已可用,不再需要手动分发。**

### 2.1 哪些行为是编译期定的、哪些是运行期探测的（三条轴，别混）

平台相关行为分三类，**只有第 2 类走上面的"编译变体 + 运行期 dispatch"双重模型**：

| 轴 | 例 | 何时定 | 说明 |
|---|---|---|---|
| **平台/架构身份** | `HYLE_OS_*`/`HYLE_ARCH_*`/`HYLE_COMPILER_*`（`platform.h`）| **纯编译期** | 你为哪个 OS/架构编译就是哪个，无从运行期切换（syscall/ABI 在链接期已定）。不属 ISA dispatch 范畴。 |
| **`simd/` 的 ISA 后端** | vec 后端 / `ops.h` / `dtype_ops.h` | **双重**（编译变体 + 运行期选）| 完全符合 §2 双模型：`-march` 自编(A) 或 运行期 CPUID 选变体(B)。|
| **`container/` SwissTable 的 SIMD 探测** | `probe.h` 的 SSSE3 `_mm_sign_epi8` 快路径 | **纯编译门控**（`#ifdef __SSSE3__` + SSE2 兜底）| **故意的例外**：container 是 Tier 0,不依赖 `simd/`(Tier 1) 以免架构倒置,故内联一小段自包含探测并编译门控。**无 SIGILL 风险**(SSSE3 仅在 `-mssse3` 时编出,SSE2 基线恒在),但分发型二进制若没开 `-mssse3` 则即便 CPU 支持也不会走 SSSE3 快路径。要让 SwissTable 吃满本机 ISA,自编时加 `-march=native`。将来可选复用 `simd/ops` 做运行期探测,但不强耦合。|

**一句话**：`simd/` 的双重探测设计是真的、自洽的；`platform.h` 是天然纯编译期的另一根轴；`container/probe.h` 是**有意的纯编译门控例外**(Tier 0 不倒挂依赖)——正确安全,但不做运行期 dispatch。

## 3. 各就绪模块最小用法

### 3.1 mem —— 显式 allocator 注入
所有分配型 API 收一个 `Allocator`（值传递，无隐藏全局堆）。进程级默认堆是 `system_allocator()`（mimalloc backing）：
```cpp
#include "hyle/mem/allocator.h"
using hyle::mem::system_allocator;
auto a = system_allocator();
int* p = a.allocate_n<int>(16);        // 16 个 int 的裸存储
a.release(p, 16);
auto obj = a.make_unique<MyType>(args);  // 经此 allocator 析构的 unique_ptr
```

### 3.2 container —— SwissTable（对标 absl，快 std 2-6x）
```cpp
#include "hyle/container/swiss_table.h"
using hyle::container::SwissTable;
SwissTable<uint64_t, uint64_t> m(system_allocator());
m.insert(42, 100);
if (uint64_t* v = m.find(42)) { /* *v == 100 */ }
```
其余：`CArray<T>`（侵入式数组）、`SmallVector<T,N>`、`string_ref`（视图字符串）、`BoundedQueue`（lock-free MPMC）。

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

## 4. C-ABI 边界（跨 DLL / FFI）

HYLE 内部 C++23，对外稳定边界走 **vtable 结构体 `xxx_api_t` + 不透明实例 + `make_X()/destroy_X()` 工厂**（分配责任锁在 DLL 内，禁客户端 `new/delete` HYLE 对象）。若你跨 DLL 边界用 HYLE 类型或做 FFI 绑定，遵此范式；纯 `extern "C"` C 头 shim 正在逐模块封（P6）。细节见 [`architecture/coding-conventions.md` §4](../architecture/coding-conventions.md)。

## 5. 安全边界（必读补充）

本指南讲"正常用法"。**为 SOTA 性能保留的调用约束**（违反 = UB，非 HYLE bug）单独列在 [`usage-contract.md`](./usage-contract.md)（同目录）：fiber 跨 yield 的 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离等。集成前过一遍。

## 6. 反馈

问题报 https://github.com/user/axiomate/issues。
