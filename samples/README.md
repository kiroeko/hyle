# HYLE samples

可运行的示例 C++ 工程，演示把 HYLE 各模块**当 infra 集成进自己工程**的推荐姿势。
每个示例是一个**独立 CMake 工程**——它通过 `add_subdirectory` 把 HYLE 仓库当子目录拉进来，
这正是外部使用者最朴素的集成方式（未来 HYLE 提供 install/`find_package` 后会更简，但 subdir
集成永远可用）。

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
| [`01_mem_container/`](./01_mem_container/) | mem + container | 显式 `Allocator` 注入 + `SwissTable`（对标 absl 的 flat hashmap）——HYLE 的"无隐藏全局堆"范式 + 旗舰容器 |
| [`02_simd/`](./02_simd/) | simd | 运行期 ISA 派发的 `dot_f32`/`saxpy_f32` + 直接用 `vec<T,N>` D-tag 写宽度无关 kernel + 打印选中的 ISA 变体 |
| [`03_sched_fibers/`](./03_sched_fibers/) | sched | `Scheduler` + `Fiber`：spawn 多个纤程、`yield()` 协作、`shutdown()` 排空 |
| [`04_prebuilt_lib/`](./04_prebuilt_lib/) | 构建/集成 | 把 HYLE 编成**预构建库**(动态/静态)再链接的集成方式——验证 `HYLE_<MOD>_API` 导出/导入边界(对比 01-03 的源码 `add_subdirectory` 集成) |

> 只列**已就绪**模块（见 [`../docs/guide/usage-guide.md`](../docs/guide/usage-guide.md) §0 就绪度表）。io/compute/graph 尚未实现，无示例。
> 01-03 用源码 `add_subdirectory` 集成；04 用**预构建库**集成（两种姿势都演示）。

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
