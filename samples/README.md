# HYLE samples

当前用户面只支持 C++23，交付形式有三种：完整源码、headers + static libraries、headers +
shared libraries。这里的产品示例只演示这三种 C++ 接入；shared library 不等于 Native FFI，
当前 Native FFI 状态是 `N/A`。
早期 C schema/package fixture 与 managed probe 已移除；本目录没有 native-C 或外语产品示例。

每个 C++ 示例都是独立 CMake project。源码 sample 用 `add_subdirectory`，安装包 sample 用
`find_package`；消费者均链接同名 `hyle::<product>` façade target。

## 接入模式（每个 sample 的 CMakeLists 都这么做）

```cmake
# 集成方在 add_subdirectory 之前关掉 HYLE 自己的测试/基准，避免拖入 GoogleTest 等
set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(HYLE_COMPONENTS "container;mem")
add_subdirectory(<path-to-hyle> hyle_build)

target_link_libraries(your_app PRIVATE hyle::container hyle::mem)  # 按需选产品
```

链接 `hyle::<product>` 自动带上头路径与内部 factor 闭包（如 `hyle::container` 自动拉所需 mem）。
不要直接链接包内的 `hyle::_factor_*` detail target。
头一律 `#include "hyle/<module>/<header>.h"`。

## C++ 用户示例

| 目录 | 模块 | 演示什么 |
|---|---|---|
| [`01_mem_container/`](01_mem_container/) | mem + container | 默认 `SystemAllocator` + 公开形状 `FlatHashMap`；运行时 allocator 只在 quota/stats/自定义 backing 时显式 opt-in |
| [`02_simd/`](02_simd/) | simd | C++ 运行期 ISA 派发的 `dot`/`saxpy`；可用 `HYLE_SAMPLE_SIMD_ISA` 绑定一个 expert TU 并直接使用锁定 Highway；打印选中的 ISA 变体 |
| [`03_sched_fibers/`](03_sched_fibers/) | sched | `Scheduler::spawn_detached` 自持 guarded fiber：`yield()` 协作、stack 回收、`shutdown()` 排空 |
| [`04_prebuilt_lib/`](04_prebuilt_lib/) | 构建/集成 | install + `find_package(Hyle COMPONENTS simd mem)`；同一 consumer 验证 shared/static、传递依赖、mimalloc 私有后端和 Windows DLL staging |
| [`05_installed_sdk/`](05_installed_sdk/) | 构建/集成 | source-independent C++ package consumer：普通 mem/STL/PMR journey 与全部 advertised imported targets |
| [`06_installed_io_http2/`](06_installed_io_http2/) | io + HTTP/2 | source-independent C++ feature consumer：原生 HTTP/2 seam 与 static/shared 私有协议依赖闭包 |
| [`08_inference_service/`](08_inference_service/) | platform + mem + util + sched + io + simd + obs | 公开 C++ API 的真实 HTTP 服务纵切：模型映射、backend 选择、REST/SSE、指标与关闭；用假计算暴露仍需用户手写的通用胶水 |
| [`package/graph/custom_kernel/`](package/graph/custom_kernel/) | compute + graph | owning typed factory、跨 shared image 的 schema/执行、direct 和 Graph 重放；支持源码或安装包接入 |
| [`package/container/intrusive/cpp/`](package/container/intrusive/cpp/) | container / intrusive journey | source-independent C++ package consumer：通过 `hyle::container` 使用 auto/safe member hook、显式 retirement、insert/erase、整链 splice 与 caller-owned lifetime |
| [`package/math/cpp/`](package/math/cpp/) | math | source/static/shared 中的 checked/fail-fast integer ceil/round 与 header-only target 闭包 |
| [`package/simd/cpp/`](package/simd/cpp/) | simd | source/static/shared 同一 `hyle::simd_expert`、fixed-target direct Highway/xsimd 与 custom fat kernel |
| [`package/process/cpp/`](package/process/cpp/) | process | source/static/shared 中启动当前 probe、取得 typed exit，并验证 runtime shutdown 与传递依赖 |
| [`package/compute/cpp/`](package/compute/cpp/) | compute | source/static/shared 中显式 CPU Context/Queue、Mem tensor、prepared RMSNorm 与 typed host kernel |
| [`package/graph/cpp/`](package/graph/cpp/) | graph | source/static/shared 中多节点 GraphBuilder、workspace/report、Executable/Invocation 与绑定提交 |

> 01-03 用源码 `add_subdirectory` 集成；04 及之后的编号示例与 `package/*/cpp` 用安装包集成。
> `package/*/cpp` 是构建回归 fixture，不是额外的产品 API。

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
