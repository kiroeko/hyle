# C ABI 设计决策（C++ 库的独立 C/FFI 产品面）

> **信源**：本文件是 HYLE **对外 C ABI 面的设计推理与决策**。落地细节看 `include/hyle/c/*.h` 与各模块 `docs/guide/c-*.md`（逐符号 1:1）；本文件只管"为什么这么设计"。
> 上位依据：[`engineering-charter.md`](./engineering-charter.md) §6（FFI 决策）、[`coding-conventions.md`](./coding-conventions.md) §4（C-ABI 范式）。本文件在其之上补齐**具体做法与实证对照**。

## 0. 目标与边界

**HYLE 是 C++ 基础加速库；`include/hyle/c/*.h` 是另行设计的 C/FFI 产品面，
不是把原生 C++ 类自动翻译一遍。** C ABI 覆盖适合跨语言使用的运行时能力，模板、
RAII 和 C++ 标准库适配只在存在自然 C 用户任务时才下沉。

- **C ABI 是唯一承诺跨编译器/跨 STL/跨语言稳定的接缝**——几乎所有 OS/语言都认 C 调用约定 + C 结构体布局。
- **两种消费者、两条独立线路**（关键区分）：
  - **C++ 用户**：直接用 `include/hyle/<mod>/*.h` 的原生 C++ API（模板、RAII、`Allocator`/`VmRegion`/`ObjectPool<T>`）。不关心跨语言。**这条线的"完整"独立评估**——C++ 面本身完整即可。
  - **跨语言用户**：只绑 `include/hyle/c/*.h`。**这条线的"完整"也独立评估**——每个能力有没有 C ABI 通路。
  - 两条线互不牵扯：C++ 有模板 ≠ C ABI 缺能力；C ABI 没导出某类 ≠ C++ 用户拿不到。
  - 但“符号不必 1:1”不等于任意删能力：属于 C/FFI 产品范围而不能直接导出的
    模板、RAII、协程或回调组合，应在对应 `docs/guide/c-*.md` 给出可执行的 C 组合
    路径。明确只服务原生 C++ 的能力则应写清边界，不制造生硬的 C 壳。

## 1. 核心手法：类型擦除（C++ → C ABI 的关键结论）

C ABI 要当一等公民，数据模型就得是**运行时参数化**，而非**编译期参数化**。C 没有泛型，编译期的 `<T>` 过不了边界；把类型信息**擦除成运行时参数**就过得去。

| 编译期（C++，过不了 C） | 运行时擦除（C ABI 通路） |
|---|---|
| `ObjectPool<T>` / `ObjectPoolCache<T>` | hard pool 用 `hyle_shared_pool_create(sizeof(T), count, alignof(T))`；soft 层用 `hyle_shared_pool_cache_create(pool, metadata, capacity, batch)` 显式附着并 publish；`<T>` 降为 runtime size/alignment |
| `GrowablePool<T>` | `hyle_growable_pool_create(slot_size, slot_align, initial_slots)` —— 非模板运行时核心（`detail::GrowablePoolCore`）承载擦除后的裸槽逻辑，`<T>` 模板只是跑 ctor/dtor 的薄包装 |
| `Allocator::construct<T>()` | 裸 `alloc` 返回内存，构造由调用方语言做 |
| `Tensor<float, N>`（ggml 的对照） | `ggml_tensor { enum type; void* data; int64_t ne[]; }` —— 类型降成 `enum` 标签 |

**判据**：C ABI 用户"不缺能力" ≠ "每个 C++ 符号都有 C 对应"。判据是**能力有没有 C ABI 通路**：
- `ObjectPool<T>` / `ObjectPoolCache<T>` 不直接导出模板，但定长对象池能力分别经 `hyle_shared_pool_*` / `hyle_shared_pool_cache_*` + typed 宏可达 → **不缺**。
- 泛型语法不可达（`<T>`），但能力经 size 擦除全部可达。

## 2. 双产品面与共享实现（ggml/oneDNN 实证）

```
┌─ 原生 C++ 产品面                         ← include/hyle/<mod>/*.h
│   模板、RAII、强类型与 std::pmr；是 C++ 用户的首选 source API
├─ C/FFI 产品面                            ← include/hyle/c/*.h
│   opaque 句柄 + 自由函数 + POD；是跨编译器/跨语言的 ABI 稳定面
└─ 共享实现                                ← src/*.cc + header-inline fast path
    两个产品面复用算法与合同，但分别按语言习惯设计，不要求符号或对象布局 1:1
```

**实证（源码在 `C:\public\project-hyle\` 下）**：
- **ggml**：21 个 C 头 `extern "C"` + **唯一 1 个** C++ only 头 `ggml-cpp.h`（`#ifndef __cplusplus #error`，仅 `unique_ptr`+deleter 包 C 句柄，deleter 调 `ggml_free`）。内部实现是 C++（8 `.cpp` vs 3 `.c`，`ggml-impl.h` 有 template/std::），但**对外零 C++ 泄漏**。ggml 不是"避免了模板"，是**领域决定其公开数据模型天然是 C 形状**（`ggml_tensor` = 运行时 `enum type` 标签，非模板）——类型多态放运行时，C ABI 自然一等公民。
- **oneDNN**：26 C 头 + 16 C++ 头；`dnnl.hpp` `#include "dnnl.h"`，用 `handle<T, traits>` 模板（`shared_ptr` + `handle_traits<T>::destructor` 调 C 的 `dnnl_*_destroy`）包 opaque C 句柄，`error::wrap_c_api` 把 C 返回码转 C++ 异常。**重量级 C++ 便利层，但仍在 C ABI 之上**。

HYLE 不在 C ABI 上再包一层冗余 C++ facade：原生 `include/hyle/<mod>/*.h` 本身就是
第一等 C++ 产品面。C ABI 只承担自然的 C 用户任务与跨语言稳定接缝。

## 3. 版本协商：删除（场景驱动，非默认标配）

**决策（2026-07-18）：删掉整套 `hyle_*_api` 函数指针表 + `hyle_*_api_get(out, expected_abi)` 版本门 + `HYLE_*_C_ABI_VERSION` 宏。** C ABI 面回归**纯 `extern "C"` 自由函数导出**（ggml 核心的形态）。

**实证对照**：
- **clay**（HYLE 前身）：C-ABI vtable（`allocator_api_t`/`os_api_t`）**没有任何 version 字段**；`api_registry_api_t` 仅前向声明、从未定义。ABI 稳定纯靠 opaque 句柄 + vtable + make/destroy。**零版本协商。**
- **ggml**：核心 C API 只有 `ggml_version()`/`ggml_commit()` 返**字符串**（人看的），无数字版本门。**唯一**的数字版本校验 `GGML_BACKEND_API_VERSION` 在 `ggml-backend-reg.cpp:239`——**只在 dlopen 一个独立编译的 backend 插件时**校验 `reg->api_version` 拒绝错配。
- **HYLE 曾经**：每个 c 头都上了 `api` 表 + `api_get(expected)` 拒旧号 + "表尾追加保前缀布局"——**比 clay（零）和 ggml（仅插件处）都重**，是核心 API 面就上全套。

**为什么删**：
1. **场景不成立**。版本门保护的是"host 与独立编译的插件二进制错配"（ggml backend 插件的场景）。HYLE 目前是**单一 `.so` 一起编译发布**，没有"独立编译的插件"这个场景——版本门保护的错配根本不会发生。
2. **未发布、无外部调用方**。"向后兼容"（表尾追加、拒绝旧号）保护的老调用方不存在，是纯负担。
3. **干净优先**。未发布阶段应做尽量干净正确的版本，不背兼容包袱。字段该怎么排怎么排，该改就改。

**结论**：版本协商是**场景驱动的件（分发独立编译插件才需要）**，不是 C++→C ABI 的默认标配。将来 HYLE 真做"可独立编译加载的后端插件"时，再在**那个插件边界**引入版本握手（学 ggml backend 的 `api_version`），而不是核心 API 面。

> **函数指针表本身**也一并删了：它是为"运行时动态发现/版本谈判"设计的，与版本门同源。纯 `extern "C"` 导出 + 头文件声明已足够 FFI 直接绑（`dlsym`/bindgen/ctypes 都认符号），不需要把所有函数塞进一个结构体再 `api_get` 填表。

## 4. 有状态诊断：先判断是不是产品能力

“有状态”不是免除 C ABI 的理由。先按用户场景分类：

| 类 | 模块 | 当前判定与 C 用户路径 |
|---|---|---|
| `StatsAllocator` / `LeakTracker` / `AllocStats` | mem | 已认定为 opt-in 用户能力：opaque stats allocator + POD out-snapshot + bounded leak copy；默认 system allocator 不埋统计成本 |
| `ErrorRecord` | error | 当前是错误处理器的测试记录器，不是产品对象。C 用户能力由 `hyle_error_set_handler` / `push_handler` / `pop_handler` 覆盖，可在自己的 callback state 中累计同样的信息 |

**升级规则**：只要上述统计/诊断被提升为面向用户的产品能力，就必须导出；
典型形态是 opaque handle + POD snapshot/visitor，或者在所属资源句柄上提供标量
查询。不能继续以“C 没有类”或“它是诊断功能”为由省略。

**对照 ggml**：ggml 没有导出统计器类，而是把产品需要的“用了多少内存”
降维成 `ggml_used_mem(ctx)` / `ggml_backend_buffer_get_usage` 标量查询。要点是
**用户能力可达**，不是必须照搬 C++ 对象形状。

**注意区分**（措辞要精确）：
- `ObjectPool<T>`：**泛型语法不可达，但能力可达**（经 arena + 宏）——不是"不需要"。
- `StatsAllocator`：已通过 `hyle_stats_allocator_*` 提供正式 C 通路；计数是 opt-in
  wrapper，逐地址 leak inventory 还需显式开启，因此不污染默认分配热路。
两者目前都没有同名 C 对象，但前者是直接类型擦除，后者是明确的产品边界判定；
都必须在用户 guide 中交代可达路径，不能只写“C++ only”。

## 5. 错误跨 C 边界

- **返回值通道**：`hyle_status`（opaque owning handle，`uintptr_t rep`）值返回 + 分支；`HYLE_C_RETURN_IF_ERROR` 传播。`Result<T>`（`std::expected`）是模板，C 侧用 `hyle_status` + 出参表达。
- **诊断通道**：`hyle_error_report` + 可换 handler（`hyle_error_set_handler` 简单单槽 / `hyle_error_push_handler`+`pop` 嵌套安全，对标 signal old-handler）。
- **绝不让 C++ 异常跨 C 边界**（UB）。catch 只在 adapter 边界做"外来异常→Status"。
- **分配器层不走 `hyle_status`**（mem 的刻意分层）：C 句柄工厂的资源失败和可由 FFI 数据表达的非法值/溢出 layout 均直接返 NULL；`hyle_mem_last_error_code()` 以独立 TLS 小枚举区分 INVALID_ARGUMENT、OUT_OF_MEMORY、EXHAUSTED 与 PLATFORM，字符串只作诊断。生命周期、并发静默、foreign/double free 等时序契约违反仍 fail-fast。分配器是**最底层基元**，让它依赖上层 error 模块的 `Status` 会**倒置依赖**。HugeRegion 在调用 VM 原语时只复制本次失败原因到 mem 通道，并恢复调用者原有的 platform VM error。
- **platform 按失败形状分通道**：Thread 构造返回 `hyle_status`；VM/NUMA/
  MappedFile 用 `hyle_vm_last_error`；DynLib 用 `hyle_dynlib_last_error`；
  Parker wrapper 与 fs/fd 用 `hyle_platform_last_error`。所有字符串均线程
  局部，成功清空 stale 值。这样不把 100+ 个布尔/指针原语机械改成
  Status，同时保证每条资源失败路径可诊断。
- 对照：ggml 用 `ggml_status` 返回码 + `ggml_abort` 回调；oneDNN 用 `dnnl_status_t` + C++ 层 `wrap_c_api` 转异常。HYLE 与 ggml 同构（返回码 + fail-fast）。

## 6. C++ API 易用性（第 1 层，给 C++ 用户）

C++ 用户拿原生 API，易用性靠 C++ 语言特性本身：
- **RAII**：`VmRegion`/`MappedFile`/`OwnedFd`/`StackAllocator` 析构自动释放，move-only。
- **模板便利**：`Allocator::allocate_n<T>`/`construct<T>`/`make_unique<T>`、硬容量 `ObjectPool<T>`、显式软层 `ObjectPoolCache<T>`、`GrowablePool<T>`。
- **类型安全**：强类型句柄、`Status::Is##Code()` 谓词、`Result<T>` 单子式传播（`HYLE_ASSIGN_OR_RAISE`）。

C ABI 用户（第 2 层）的易用性则靠 **header-inline 便利层**（零新导出符号，不进 ABI）：
- error：全码工厂 `hyle_status_not_found()` 等 + 谓词 `hyle_status_is_*()` + `HYLE_C_RETURN_IF_ERROR`。
- platform：`hyle_thread_opts_set_*` + `HYLE_THREAD_OPTS_INIT` + `hyle_page_align_up` inline。
- mem：`hyle_alloc`/`free`/`realloc` 便利转发 + `*_create_default` system-backing 构造 + 默认硬容量 `HYLE_OBJECT_POOL_*`、显式软容量 `HYLE_OBJECT_POOL_CACHE_*` typed 宏（`HYLE_ALIGNOF` 跨 C/C++ shim）。默认 alignment 在头内常量折叠，不先跨 DLL 查询。

**原则**：两条线各自把易用性做到位——C++ 靠语言特性，C 靠 header-inline 糖 + opaque 句柄。inline 糖不进 DLL 导出、零 ABI 风险，是"给 C 用户便利但不污染稳定面"的标准手法（HYLE 与 ggml/oneDNN 一致）。

## 7. 落地纪律

- HYLE 主体是 C++23：`include/hyle/<mod>/*.h` 用 `<cstddef>` 与
  `std::size_t`/`std::byte`。只有语言中立的 `include/hyle/c/*.h` 用 `<stddef.h>` 与
  全局 `size_t`；这些头必须继续能由 C11/C17 编译器直接消费。
- 每个对外模块一层 `include/hyle/c/<mod>.h`：纯 C 可编译、`extern "C"`、opaque `_o` 句柄、create/destroy 配对、POD 值类型、显式 allocator 注入。
- 可扩展输入 POD 以 `struct_size` 开头；值返回 snapshot POD 只用于小而
  稳定的输出。桥接编译单元用 standard-layout/trivially-copyable/offset
  与 enum-value `static_assert` 把 C/C++ 两面锁在一起。
- 分平台导出宏三态（`HYLE_<MOD>_C_API`：dllexport/dllimport/visibility）。
- 桥接实现 `src/<mod>/c_<mod>.cc` 编进对应 `hyle_<mod>`，**显式构造** C++ 对象，不 reinterpret 布局。
- 纯 C smoke test `tests/<mod>/c_<mod>_test.c`（C 编译器编，只 include C 头）证明无 mangling/模板/STL 泄漏。
- 文档 `docs/guide/c-<mod>.md` 逐符号 1:1；对模板/RAII/协程等非
  1:1 形状，另给“C++ 能力 → C 组合”表和可复制 recipe。不得只列
  “C++ 专属”而不说明 C 用户如何取得等价能力或为什么它明确不是产品能力。
- 三平台真编真测（Win MSVC + WSL GCC + mac clang），mac 的严格 clang 常揪出隐式 include 依赖（如 `std::nothrow` 需 `<new>`）。
