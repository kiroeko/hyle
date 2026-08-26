# Core C++ 使用指南

> **API 暂定。** Core 正在进行 v2 `OPEN` 复审。下列例子只描述当前确实存在的代码，不构成
> compatibility 或 SOTA 承诺；正式发布前，归属、命名和组件位置都可能改变。

Core 目前给 C++ 用户提供三组底层工具：

| 需求 | include | 当前入口 |
|---|---|---|
| 编译期或 linked 产品身份 | `<hyle/core/version.h>` | `kVersion*`、`version()`、`version_string()` |
| cache layout 常量或 inline prefetch hint | `<hyle/core/cache.h>` | `k*InterferenceSize`、`prefetch_*` |
| HYLE/compiler 合同 | `<hyle/macros.h>` | `HYLE_NODISCARD`、`HYLE_ASSUME`、`HYLE_RESTRICT`、inline/unroll/fatal helpers |

Core 没有 umbrella header；只 include 当前任务需要的头。

侵入式链表属于 container 语义，已经迁到独立的轻量
[`hyle::container_intrusive`](container.md#侵入式链表独立轻量组件) target。需要它的用户不应再通过
Core 获取，也不必因此链接完整 `hyle::container`。

## 把 Core 加入 C++ 构建

安装后的静态或动态 SDK：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS core)
target_link_libraries(my_target PRIVATE hyle::core)
hyle_stage_runtime_dependencies(my_target)
```

动态 package 下，`hyle_stage_runtime_dependencies()` 负责 stage DLL/DSO；静态 package 下它是安全
no-op。不要为 installed package 手工定义 `HYLE_STATIC`，应使用该 package 导出的 target。

整体源码组装：

```cmake
set(HYLE_COMPONENTS core)
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
add_subdirectory(path/to/hyle)
target_link_libraries(my_target PRIVATE hyle::core)
```

如果 enclosing build 需要静态库，在配置 HYLE 时一致地选择 `HYLE_STATIC`。三种交付下，cache/
compiler 逻辑都由 header inline 编进 consumer；只更新 shared library 不能替换这些代码，
相应 header 或常量变化后必须重编消费者。target 会传播当前 C++23 要求。

Core 当前没有受支持的 Native FFI 产品。源码树里的 C-linkage version 候选只是未发布实现历史，
不是 Java/.NET/Rust/Python 合同；Core rebuild 仍为 open 时，不要据此制作 binding。

## 产品身份

只需要“编译时用了哪版 header”时用常量；需要诊断实际加载了哪个 `hyle_core` 时才调用 linked
query：

```cpp
#include <cstdio>

#include <hyle/core/version.h>

static_assert(hyle::kVersion >= HYLE_VERSION_PACK(0, 1, 0));

int main() {
  std::printf("headers=%s runtime=%s\n", hyle::kVersionString,
              hyle::version_string());
  return hyle::version() == hyle::kVersion ? 0 : 1;
}
```

`version_string()` 返回 module-owned 借用存储，绝不能 free；卸载动态 `hyle_core` 前先复制字符。
产品版本相等只是一项诊断，不是通用 binary-compatibility 证明，也不能验证进程里每个 inline 类型
是否由同一套 header 编译。

## Cache layout 与 prefetch

只有 false sharing 或 constructive packing 真的是对象设计一部分时，才使用 layout 常量：

```cpp
#include <atomic>
#include <cstdint>

#include <hyle/core/cache.h>

struct alignas(hyle::core::kDestructiveInterferenceSize) Counter {
  std::atomic<std::uint64_t> value{0};
};
```

这些常量是编译期 target-family recommendation，不是当前 CPU 的测量值。它们影响 `alignof`、
`sizeof`、allocation alignment 和 ABI；不要混用由不同 header 构建的对象。runtime CPU/cache
observation 属 platform，使用时需另外请求该组件。

当前 hint 为：

- `prefetch_read(p)`：预计会重复使用的数据；
- `prefetch_read_non_temporal(p)`：可能只读取一次的数据；
- `prefetch_write(p)`：capable target 可能受益于提前取得 write ownership 的地址。

它们允许退化成其他 hint 或 no-op，不会验证 `p`、延长对象生命、读取值、建立同步或保证 cache
residency。先依赖普通访问和硬件 prefetcher；只有拥有该访问模式的正常 workload 证明有稳定
收益时才加 hint。永久 cache-line padding 也会消耗内存，不能默认加到所有对象上。

## 编译器合同

`HYLE_NODISCARD`、`HYLE_FORCE_INLINE`、`HYLE_NOINLINE`、branch hint 与 unroll/flatten 都是
source-level compiler 工具。以下危险合同尤其要谨慎：

- `HYLE_ASSUME(expr)` 要求表达式无副作用且已被证明；Release 中为 false 是 undefined behavior；
- `HYLE_RESTRICT` 承诺相关 pointer range 不 alias，违约是 undefined behavior；
- Release `HYLE_UNREACHABLE()` 告诉 optimizer 控制流不可能抵达，不是输入验证；
- `HYLE_FATAL_ASSERT` 终止进程，只用于 invariant 被破坏，不能处理可恢复 user/I/O error；
- force-inline、flatten、unroll 和 prefetch 都有 code-size/I-cache 代价，只用于测过的 native hot path。

可恢复失败使用 error 模块。普通应用代码优先使用标准 C++；只有需要匹配 HYLE header 合同或
经过测量的 kernel 时，才选择对应 HYLE macro。
