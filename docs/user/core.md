# Core C++ 使用指南

Core 是 HYLE 的 header-only C++ 地基。普通业务代码通常不会单独“调用 Core”；当你在 native hot
path 里需要编译器合同、设计并发对象布局、表达程序不变量，或读取编译期产品版本时才直接使用。

| 我现在要做什么 | 选什么 | 不要选什么 |
|---|---|---|
| 可恢复地返回 OOM、文件、设备或输入失败 | [error](error.md) 的 `hyle::Status` / `hyle::Result<T>` | `HYLE_FATAL_ASSERT`、`HYLE_ASSUME` |
| 证明永远不应失败的程序不变量 | `HYLE_FATAL_ASSERT`；需要静态上下文时用 `HYLE_FATAL_ASSERT_MSG` | 用 error 值继续运行一个已经损坏的进程状态 |
| 给已测 native kernel 表达 alias/branch/loop/inlining 合同 | `<hyle/core/compiler.h>` | 为“也许更快”遍撒 hint |
| 隔开高争用独立写字段 | `kDestructiveInterferenceSize` | 把 runtime cache-line probe 放进公开类型布局 |
| 对明确的未来访问给出 cache hint | 对应 `prefetch_*` | 把 prefetch 当读取、同步或 residency 保证 |
| 判断编译当前 TU 的 header 版本 | `<hyle/version.h>` 常量 | 查找不存在的 Core runtime version/DLL |
| 获取当前机器 cache/CPU/NUMA 信息 | [platform](platform.md) 模块 | Core 的编译期 layout recommendation |

Core 没有 umbrella header；只 include 当前任务需要的头。

## 加入构建

安装后的 SDK：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core)
target_link_libraries(my_target PRIVATE hyle::core)
```

整体源码组装：

```cmake
set(HYLE_COMPONENTS core)
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
add_subdirectory(path/to/hyle)
target_link_libraries(my_target PRIVATE hyle::core)
```

`hyle::core` 是 `INTERFACE` target：source、static SDK、shared SDK 三种交付都没有 Core binary，
也没有 `hyle_core.dll/.so/.dylib` 需要 stage。target 传播 include path 与 HYLE 的 C++23/build
contract，实际 helper 编进你的 translation unit。不要只手写 include path；那会绕过组件闭包。
Core 没有 trace site 或 test-only ABI，因此不会传播相应横切 target；仅选择 Core 却设置
`HYLE_TRACE=ON` 会明确配置失败，而不会生成一个无人使用的 profiler runtime。

如果同一 target 还使用其他 shared HYLE 模块，仍按
[distribution guide](distribution.md) 为那些模块 stage runtime dependency；这个动作与 Core
本身无关。

## 编译期版本

```cpp
#include <hyle/version.h>

#include <string_view>

static_assert(hyle::kVersionMajor == HYLE_VERSION_MAJOR);
static_assert(hyle::kVersionPacked >= 0x000100u);
static_assert(std::string_view(hyle::kVersionString) == HYLE_VERSION_STRING);
```

这些常量说明编译当前 TU 的 headers。它们不是 binary compatibility 证明，也不能查询进程里实际
加载了哪些其他 HYLE shared modules。Core 没有 linked `version()`/`version_string()`。

## 程序不变量

```cpp
#include <hyle/core/assert.h>

void consume(unsigned index, unsigned capacity) {
  HYLE_FATAL_ASSERT_MSG(index < capacity, "index outside arena");
  // 失败时在 Debug/Release 都 abort
  // ...
}
```

只在继续执行意味着程序状态已经不可信时使用 fatal assertion。用户输入、容量不足、系统调用、
OOM、设备和协议失败都应返回 error 模块的值。两个 macro 都只求值一次条件；消息版本只在失败时
求值一次，并在终止前借用 NUL-terminated `const char*`。它不复制、不接管 lifetime，也不是格式化
或日志 API，也不保证 signal handler 中安全。不要依赖条件中的副作用；若消息是动态存储，必须
保证它在失败调用期间仍有效。

## 编译器合同

```cpp
#include <hyle/core/compiler.h>

#include <cstddef>

HYLE_NOINLINE int sum(const int* HYLE_RESTRICT values,
                      std::size_t count) {
  HYLE_ASSUME(values != nullptr);
  int total = 0;
  HYLE_UNROLL(4)
  for (std::size_t i = 0; i < count; ++i) total += values[i];
  return total;
}
```

| 入口 | 语义 | 主要陷阱 |
|---|---|---|
| `HYLE_FORCE_INLINE` | 请求 hot body 内联 | 是 hint，不是保证；可能增加 code size/I-cache 压力 |
| `HYLE_NOINLINE` | 把 cold/独立 body 留作调用 | 仍受 compiler/linker 规则影响 |
| `HYLE_RESTRICT` | pointer ranges 不 alias | 违约是 UB；不能凭希望添加 |
| `HYLE_LIKELY/UNLIKELY` | 提供稳定 branch 概率先验 | 数据分布变化会适得其反；条件只写可安全求值的表达式 |
| `HYLE_FLATTEN` | 请求展开调用链 | code-size 税高，只用于测过的 kernel |
| `HYLE_UNROLL(n)` / `HYLE_DEFAULT_UNROLL` | loop-transform hint | Debug/sanitizer/不支持 compiler 可退化为空 |
| `HYLE_NO_UNIQUE_ADDRESS` | 跨 MSVC/其他 compiler 表达空成员布局 | 影响类型布局；跨边界必须用同一 build contract |
| `HYLE_ASSUME(expr)` | 声明无副作用表达式必真 | Debug 会求值并在 false 时 abort；Release 为 false 是 UB，且不同 compiler 可能不求值；绝不能借它执行函数调用、递增或其他必要工作 |
| `HYLE_UNREACHABLE()` | 声明控制流不可抵达 | Debug 会 abort；Release 抵达是 UB |

优先写标准 C++：must-use 用 `[[nodiscard]]`，禁止 copy/move 用显式 `= delete`，已知对齐用
`std::assume_aligned<N>`。Core 没有 `HYLE_NODISCARD`、copy-disable macro、public token concat 或
assume-aligned wrapper。

## Cache layout

```cpp
#include <hyle/core/cache.h>

#include <atomic>
#include <cstdint>

struct alignas(hyle::core::kDestructiveInterferenceSize) Counter {
  std::atomic<std::uint64_t> value{0};
};
```

- `kConstructiveInterferenceSize`：确实一起访问的小状态可用于 packing；
- `kDestructiveInterferenceSize`：常态独立写字段的 false-sharing spacing；
- `kMaxDestructiveInterferenceSize`：跨 target 的保守最大 spacing，只适合稀疏控制字等明确场景。

它们是编译 target-family recommendation，不是当前 CPU 探测结果，并会改变 `alignof`、`sizeof`、
allocation alignment 和 ABI。不要仅为了“cache friendly”给所有对象加 128-byte padding；先量真实
contention 与 footprint。需要运行时 cache/CPU/NUMA 信息时使用 platform。

当前值为 Apple arm64 `128/128`、其他 AArch64 和 x86-64 `64/64`、未知 target `64/128`
（constructive/destructive）；`kMaxDestructiveInterferenceSize` 为 128。不要把这些值序列化为跨
build ABI，或假定所有 ARM 机器都与 Apple Silicon 相同。

## Prefetch

```cpp
#include <hyle/core/cache.h>

void use(int);

void visit_reused(const int* current, const int* likely_next) {
  hyle::core::prefetch_read(likely_next);  // 预计很快读取且还会复用
  use(*current);
}

void visit_once(const int* current, const int* one_shot_next) {
  hyle::core::prefetch_read_non_temporal(one_shot_next);  // 预计只读一次
  use(*current);
}

void update_soon(int* target) {
  hyle::core::prefetch_write(target);  // 即将修改，需比 read hint 更谨慎
}
```

prefetch 只是 O(1) best-effort hint；某些 compiler/target 会替换 hint 或成为 no-op。它不会验证地址、
延长对象 lifetime、产生 C++ load、取得锁、建立 happens-before 或保证任何 cache 层级。只传入来自
仍存活对象/分配的可形成地址，并确保即使 hint 完全消失，程序也同样正确。

硬件 prefetcher 已能处理很多线性模式。只有正常 consumer workload 证明 latency 能被未来工作
覆盖且没有额外带宽/cache pollution 时才保留手工 hint；不要根据单条 prefetch 纳秒 benchmark
修改 API 或对象 metadata。

## Native FFI 与 C

Core 没有 C 产品 API，也没有 Java/.NET/Python/Rust 的 Core binding。compiler attribute、layout
与 inline prefetch 必须发生在 C++ consumer 的编译单元里，动态函数调用无法提供等价能力。若你
只需要其他 HYLE runtime 功能，应查看该模块是否通过 Native FFI 准入；不要依赖历史
`hyle/c/core.h` 或猜测 Core export。
