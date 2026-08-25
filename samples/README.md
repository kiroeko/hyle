# HYLE samples

当前用户面只支持 C++23，交付形式有三种：完整源码、headers + static libraries、headers +
shared libraries。这里的产品示例只演示这三种 C++ 接入；shared library 不等于 Native FFI，
当前 Native FFI 状态是 `N/A`。

仓库中仍残留少量早期 C 语法 package/schema fixture 和 managed probe。它们是**待删除的历史
conformance 债**，不是 native-C 产品示例，也不是 Java/.NET 可用性证书，不能据此推断任何
外语能力。

每个 C++ 示例都是独立 CMake project。源码 sample 用 `add_subdirectory`，安装包 sample 用
`find_package`；消费者均链接同名 `hyle::<module>` target。

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
| [`01_mem_container/`](01_mem_container/) | mem + container | 默认 `SystemAllocator` + 公开形状 `FlatHashMap`；运行时 allocator 只在 quota/stats/自定义 backing 时显式 opt-in |
| [`02_simd/`](02_simd/) | simd | C++ 运行期 ISA 派发的 `dot`/`saxpy` + 直接用 `vec<T,N>` D-tag 写宽度无关 kernel + 打印选中的 ISA 变体；同目录旧 `simd_c_sample` 属待删除历史债，不是用户入口 |
| [`03_sched_fibers/`](03_sched_fibers/) | sched | `Scheduler::spawn_detached` 自持 guarded fiber：`yield()` 协作、stack 回收、`shutdown()` 排空 |
| [`04_prebuilt_lib/`](04_prebuilt_lib/) | 构建/集成 | install + `find_package(Hyle COMPONENTS simd mem)`；同一 consumer 验证 shared/static、传递依赖、mimalloc 私有后端和 Windows DLL staging |
| [`07_installed_compute/`](07_installed_compute/) | compute C++ API | 消费 CPU/CUDA、shared/static compute package；枚举真实 backend，并验证 ggml/volk/CUDA 传递闭包无需源码 checkout |

> 01-03 用源码 `add_subdirectory` 集成；04 与 07 用安装包集成。

## 待删除的历史 conformance 债（非用户产品示例）

| 目录 | 验证内容 |
|---|---|
| [`05_installed_c/`](05_installed_c/) | 旧 error/mem C-shaped fixture 待删除；同目录 C++ `SystemAllocator` probe 仍是正常 package consumer |
| [`06_installed_io_http2/`](06_installed_io_http2/) | 旧 io C-shaped fixture 待删除；HTTP/2 feature manifest 与 C++ package 闭包验证仍可保留 |
| [`package/`](package/) | source-independent C++ consumer 是有效测试；其中 C/managed schema/compiler/link fixture 属待删除历史债 |

## 构建单个示例

```bash
cd samples/01_mem_container
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/mem_container_sample      # Windows: build\Release\mem_container_sample.exe
```

工具链同 HYLE：C++23（GCC 14+ / Clang 18+ / MSVC 19.43+）、CMake 3.25+。

## 集成前必读

- [使用指南](../docs/user/usage-guide.md)：构建选项 + 各模块用法。
- [调用安全边界](../docs/user/usage-contract.md)：热路径调用安全红线（违反 = UB）。
