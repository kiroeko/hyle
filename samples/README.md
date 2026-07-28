# HYLE samples

可运行的 C/C++ 示例工程，演示把 HYLE 各模块**当 infra 集成进自己工程**的推荐姿势。
每个示例是一个**独立 CMake 工程**——它通过 `add_subdirectory` 把 HYLE 仓库当子目录拉进来，
前三个示例使用源码嵌入；其余示例使用安装后的 CMake package。两条路径都链接同名
`hyle::<module>` target，subdir 集成也会长期保留。

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

## 示例清单（覆盖已就绪模块）

| 目录 | 模块 | 演示什么 |
|---|---|---|
| [`01_mem_container/`](./01_mem_container/) | mem + container | 默认 `SystemAllocator` + 公开形状 `FlatHashMap`；运行时 allocator 只在 quota/stats/自定义 backing 时显式 opt-in |
| [`02_simd/`](./02_simd/) | simd | 运行期 ISA 派发的 `dot`/`saxpy` + 直接用 `vec<T,N>` D-tag 写宽度无关 kernel + 打印选中的 ISA 变体 |
| [`03_sched_fibers/`](./03_sched_fibers/) | sched | `Scheduler::spawn_detached` 自持 guarded fiber：`yield()` 协作、stack 回收、`shutdown()` 排空 |
| [`04_prebuilt_lib/`](./04_prebuilt_lib/) | 构建/集成 | install + `find_package(Hyle COMPONENTS simd mem)`；同一 consumer 验证 shared/static、传递依赖、mimalloc 私有后端和 Windows DLL staging |
| [`05_installed_c/`](./05_installed_c/) | error + mem C ABI、mem C++ API | 只链接 `hyle::mem` 消费 shared/static 安装包；纯 C 验证传递模块闭包，C++ `SystemAllocator` 验证 shared 不泄漏 mimalloc 符号且 static 自动携带实现 archive |
| [`06_installed_io_http2/`](./06_installed_io_http2/) | io C ABI + HTTP/2 | 纯 C 工程消费 shared/static io package；审计 `io_http2` feature，并证明 pinned nghttp2 闭包不依赖源码 checkout |
| [`07_installed_compute/`](./07_installed_compute/) | compute C++ API | 消费 CPU/CUDA、shared/static compute package；枚举真实 backend，并验证 ggml/volk/CUDA 传递闭包无需源码 checkout |

> 01-03 用源码 `add_subdirectory` 集成；04-07 用安装包集成。

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
