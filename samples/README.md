# HYLE samples

这里同时包含两类工程，不能混读：

- C++ 用户 sample：演示把 HYLE 当 infra 集成进 C++23 工程；
- C 语法 package/schema fixture：只验证候选 FFI 声明、compiler/link/export 与安装闭包卫生，
  **不是 native-C 产品示例，也不是 Java/.NET 易用性证书**。

每个工程都是独立 CMake project。源码 sample 用 `add_subdirectory`，安装包 sample/fixture 用
`find_package`；C++ 消费者均链接同名 `hyle::<module>` target。

## 接入模式（每个 sample 的 CMakeLists 都这么做）

```cmake
# 集成方在 add_subdirectory 之前关掉 HYLE 自己的测试/基准，避免拖入 GoogleTest 等
set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(<path-to-hyle> hyle_build)

target_link_libraries(your_app PRIVATE hyle::container hyle::mem)  # 按需选模块
```

链接 `hyle::<module>` 自动带上头路径与依赖模块（如 `hyle::container` 自动拉 `hyle::mem`）。
头一律 `#include "hyle/<module>/<header>.h"`。

## C++ 用户示例

| 目录 | 模块 | 演示什么 |
|---|---|---|
| [`01_mem_container/`](./01_mem_container/) | mem + container | 默认 `SystemAllocator` + 公开形状 `FlatHashMap`；运行时 allocator 只在 quota/stats/自定义 backing 时显式 opt-in |
| [`02_simd/`](./02_simd/) | simd | C++ 运行期 ISA 派发的 `dot`/`saxpy` + 直接用 `vec<T,N>` D-tag 写宽度无关 kernel + 打印选中的 ISA 变体；同目录 `simd_c_sample` 只是旧 SIMD FFI schema fixture |
| [`03_sched_fibers/`](./03_sched_fibers/) | sched | `Scheduler::spawn_detached` 自持 guarded fiber：`yield()` 协作、stack 回收、`shutdown()` 排空 |
| [`04_prebuilt_lib/`](./04_prebuilt_lib/) | 构建/集成 | install + `find_package(Hyle COMPONENTS simd mem)`；同一 consumer 验证 shared/static、传递依赖、mimalloc 私有后端和 Windows DLL staging |
| [`07_installed_compute/`](./07_installed_compute/) | compute C++ API | 消费 CPU/CUDA、shared/static compute package；枚举真实 backend，并验证 ggml/volk/CUDA 传递闭包无需源码 checkout |

> 01-03 用源码 `add_subdirectory` 集成；04-07 用安装包集成。

## Package/schema conformance fixtures（非用户产品示例）

| 目录 | 验证内容 |
|---|---|
| [`05_installed_c/`](./05_installed_c/) | 旧 error/mem C-shaped 候选可由 C compiler 表达，且 shared/static 安装闭包完整；同目录 C++ `SystemAllocator` probe 仍是正常 package consumer |
| [`06_installed_io_http2/`](./06_installed_io_http2/) | 旧 io C-shaped 候选、HTTP/2 feature manifest 与 nghttp2 安装闭包；不构成 io Native FFI 或 C 产品证书 |
| [`package/`](./package/) | 各组件的 source-independent C++ consumer，以及以 C 语法编写的 schema/compiler/link fixtures |

## 构建单个示例

```bash
cd samples/01_mem_container
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/mem_container_sample      # Windows: build\Release\mem_container_sample.exe
```

工具链同 HYLE：C++23（GCC 14+ / Clang 18+ / MSVC 19.43+）、CMake 3.25+。

## 集成前必读

- [`../docs/guide/usage-guide.md`](../docs/guide/usage-guide.md)：构建选项 + 各模块用法。
- [`../docs/guide/usage-contract.md`](../docs/guide/usage-contract.md)：热路径调用安全红线（违反 = UB）。
