# C ABI 设计决策（C++ 库如何全面导出 C ABI）

> **信源**：本文件是 HYLE **对外 C ABI 面的设计推理与决策**。落地细节看 `include/hyle/c/*.h` 与各模块 `docs/guide/c-*.md`（逐符号 1:1）；本文件只管"为什么这么设计"。
> 上位依据：[`engineering-charter.md`](./engineering-charter.md) §6（FFI 决策）、[`coding-conventions.md`](./coding-conventions.md) §4（C-ABI 范式）。本文件在其之上补齐**具体做法与实证对照**。

## 0. 定调（用户拍板）

**HYLE 从一开始就设计为"C++ 实现、全面导出 C ABI"。** 对外发布面尽可能全部走纯 `extern "C"` 的 `include/hyle/c/*.h`，让任意语言（Python/Rust/Go/TS/…）FFI 直接绑。

- **C ABI 是唯一承诺跨编译器/跨 STL/跨语言稳定的接缝**——几乎所有 OS/语言都认 C 调用约定 + C 结构体布局。
- **两种消费者、两条独立线路**（关键区分）：
  - **C++ 用户**：直接用 `include/hyle/<mod>/*.h` 的原生 C++ API（模板、RAII、`Allocator`/`VmRegion`/`ObjectPool<T>`）。不关心跨语言。**这条线的"完整"独立评估**——C++ 面本身完整即可。
  - **跨语言用户**：只绑 `include/hyle/c/*.h`。**这条线的"完整"也独立评估**——每个能力有没有 C ABI 通路。
  - 两条线互不牵扯：C++ 有模板 ≠ C ABI 缺能力；C ABI 没导出某类 ≠ C++ 用户拿不到。

## 1. 核心手法：类型擦除（C++ → C ABI 的关键结论）

C ABI 要当一等公民，数据模型就得是**运行时参数化**，而非**编译期参数化**。C 没有泛型，编译期的 `<T>` 过不了边界；把类型信息**擦除成运行时参数**就过得去。

| 编译期（C++，过不了 C） | 运行时擦除（C ABI 通路） |
|---|---|
| `ObjectPool<T>` | `hyle_memory_arena_create(sizeof(T), count, alignof(T))` —— `<T>` 降解成 `size_t block_size`/`alignment` |
| `GrowablePool<T>` | `hyle_growable_pool_create(slot_size, slot_align, initial_slots)` —— 与 `ObjectPool<T>`↔`MemoryArena` 同构：非模板运行时核心（`detail::GrowablePoolCore`）承载擦除后的裸槽逻辑，`<T>` 模板只是跑 ctor/dtor 的薄包装 |
| `Allocator::construct<T>()` | 裸 `alloc` 返回内存，构造由调用方语言做 |
| `Tensor<float, N>`（ggml 的对照） | `ggml_tensor { enum type; void* data; int64_t ne[]; }` —— 类型降成 `enum` 标签 |

**判据**：C ABI 用户"不缺能力" ≠ "每个 C++ 符号都有 C 对应"。判据是**能力有没有 C ABI 通路**：
- `ObjectPool<T>` 不导出，但定长对象池能力经 `hyle_memory_arena_*` + `HYLE_OBJECT_POOL_*` 宏可达 → **不缺**。
- 泛型语法不可达（`<T>`），但能力经 size 擦除全部可达。

## 2. 三层架构（业界通用解，ggml/oneDNN 实证）

```
┌─ 第 3 层：C++ 便利层（可选，header-only，在 C ABI 之上）
│   ggml-cpp.h / oneDNN dnnl.hpp — RAII/模板/类型安全，包 C 句柄，零新导出符号
├─ 第 2 层：C ABI（发布契约，extern "C"）        ← HYLE 的 include/hyle/c/*.h
│   opaque 句柄 + 自由函数 + POD + 返回码。唯一 ABI 稳定面，所有语言绑这层
└─ 第 1 层：C++ 实现（内部，不发布）              ← HYLE 的 src/*.cc + include/hyle/<mod>/*.h
    模板、类、std:: 自由用；C++23 的 ABI 风险被第 2 层完全隔离
```

**实证（源码在 `C:\public\project-hyle\` 下）**：
- **ggml**：21 个 C 头 `extern "C"` + **唯一 1 个** C++ only 头 `ggml-cpp.h`（`#ifndef __cplusplus #error`，仅 `unique_ptr`+deleter 包 C 句柄，deleter 调 `ggml_free`）。内部实现是 C++（8 `.cpp` vs 3 `.c`，`ggml-impl.h` 有 template/std::），但**对外零 C++ 泄漏**。ggml 不是"避免了模板"，是**领域决定其公开数据模型天然是 C 形状**（`ggml_tensor` = 运行时 `enum type` 标签，非模板）——类型多态放运行时，C ABI 自然一等公民。
- **oneDNN**：26 C 头 + 16 C++ 头；`dnnl.hpp` `#include "dnnl.h"`，用 `handle<T, traits>` 模板（`shared_ptr` + `handle_traits<T>::destructor` 调 C 的 `dnnl_*_destroy`）包 opaque C 句柄，`error::wrap_c_api` 把 C 返回码转 C++ 异常。**重量级 C++ 便利层，但仍在 C ABI 之上**。

**HYLE 现状**：第 1、2 层齐全（error/platform/mem）。**第 3 层（对标 ggml-cpp.h/dnnl.hpp 的 C++ 便利包装）目前不做**——因为 HYLE 的 C++ 用户直接用原生第一等的 `include/hyle/<mod>/*.h`（本就是 RAII/模板），比"包 C 句柄的壳"更好；只有当对外主接口是 C（ggml/oneDNN 那样）才需要那层壳补 C++ RAII。**HYLE 不是那种库，故不需要。**

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

## 4. 有状态调试类：留 C++（无 FFI 需求）

有状态的调试/内省类**故意只在 C++**，不 C ABI 化：

| 类 | 模块 | 为什么留 C++ |
|---|---|---|
| `StatsAllocator` / `LeakTracker` / `AllocStats` | mem | 内存记账 + 泄漏追踪。debug/test 工具，跨语言用户自带（ASan/valgrind/各 runtime profiler），无 FFI 需求 |
| `ErrorRecord` | error | 诊断记录器（per-severity 计数 + last message）。同上，测试设施 |

**对照 ggml**：ggml 也**不导出统计器类**——它只把"用了多少内存"降维成**标量查询** `ggml_used_mem(ctx)`/`ggml_backend_buffer_get_usage`。即"可观测性经标量查询，不经有状态对象"。HYLE 若将来要给 FFI 用户内存可观测性，也应走标量查询而非导出 `StatsAllocator` 句柄。

**注意区分**（措辞要精确）：
- `ObjectPool<T>`：**泛型语法不可达，但能力可达**（经 arena + 宏）——不是"不需要"。
- `StatsAllocator`：**能力也可 C ABI 化（opaque 句柄 + snapshot POD），但无 FFI 需求**——是"不需要"。
两者都留 C++，但理由不同，别混。

## 5. 错误跨 C 边界

- **返回值通道**：`hyle_status`（opaque owning handle，`uintptr_t rep`）值返回 + 分支；`HYLE_C_RETURN_IF_ERROR` 传播。`Result<T>`（`std::expected`）是模板，C 侧用 `hyle_status` + 出参表达。
- **诊断通道**：`hyle_error_report` + 可换 handler（`hyle_error_set_handler` 简单单槽 / `hyle_error_push_handler`+`pop` 嵌套安全，对标 signal old-handler）。
- **绝不让 C++ 异常跨 C 边界**（UB）。catch 只在 adapter 边界做"外来异常→Status"。
- **分配器层不走 `hyle_status`**（mem 的刻意分层）：句柄工厂 `hyle_*_create` **资源失败返 NULL + 设 `hyle_mem_last_error` 字符串**（invalid backing / wrapper·slab OOM / 池耗尽）；**契约违反仍 fail-fast abort**（ctor 的 `block_size==0`、`block_size*block_count` 溢出 `size_t`；`allocate` 收到非 2 的幂对齐）。分配器是**最底层基元**，让它依赖上层 error 模块的 `Status` 会**倒置依赖**；故 mem 自足用"NULL + thread-local last-error 串"通道（与 ggml 分配器返 NULL / 返回码同理），不引入 `hyle_status`。
- **platform 按失败形状分通道**：Thread 构造返回 `hyle_status`；VM/NUMA/
  MappedFile 用 `hyle_vm_last_error`；DynLib 用 `hyle_dynlib_last_error`；
  Parker wrapper 与 fs/fd 用 `hyle_platform_last_error`。所有字符串均线程
  局部，成功清空 stale 值。这样不把 100+ 个布尔/指针原语机械改成
  Status，同时保证每条资源失败路径可诊断。
- 对照：ggml 用 `ggml_status` 返回码 + `ggml_abort` 回调；oneDNN 用 `dnnl_status_t` + C++ 层 `wrap_c_api` 转异常。HYLE 与 ggml 同构（返回码 + fail-fast）。

## 6. C++ API 易用性（第 1 层，给 C++ 用户）

C++ 用户拿原生 API，易用性靠 C++ 语言特性本身：
- **RAII**：`VmRegion`/`MappedFile`/`OwnedFd`/`StackAllocator` 析构自动释放，move-only。
- **模板便利**：`Allocator::allocate_n<T>`/`construct<T>`/`make_unique<T>`、`ObjectPool<T>`/`GrowablePool<T>`。
- **类型安全**：强类型句柄、`Status::Is##Code()` 谓词、`Result<T>` 单子式传播（`HYLE_ASSIGN_OR_RAISE`）。

C ABI 用户（第 2 层）的易用性则靠 **header-inline 便利层**（零新导出符号，不进 ABI）：
- error：全码工厂 `hyle_status_not_found()` 等 + 谓词 `hyle_status_is_*()` + `HYLE_C_RETURN_IF_ERROR`。
- platform：`hyle_thread_opts_set_*` + `HYLE_THREAD_OPTS_INIT` + `hyle_page_align_up` inline。
- mem：`hyle_alloc`/`free`/`realloc` 便利转发 + `HYLE_OBJECT_POOL_CREATE/ACQUIRE/RELEASE` typed 宏（`HYLE_ALIGNOF` 跨 C/C++ shim）。

**原则**：两条线各自把易用性做到位——C++ 靠语言特性，C 靠 header-inline 糖 + opaque 句柄。inline 糖不进 DLL 导出、零 ABI 风险，是"给 C 用户便利但不污染稳定面"的标准手法（HYLE 与 ggml/oneDNN 一致）。

## 7. 落地纪律

- 每个对外模块一层 `include/hyle/c/<mod>.h`：纯 C 可编译、`extern "C"`、opaque `_o` 句柄、create/destroy 配对、POD 值类型、显式 allocator 注入。
- 可扩展输入 POD 以 `struct_size` 开头；值返回 snapshot POD 只用于小而
  稳定的输出。桥接编译单元用 standard-layout/trivially-copyable/offset
  与 enum-value `static_assert` 把 C/C++ 两面锁在一起。
- 分平台导出宏三态（`HYLE_<MOD>_C_API`：dllexport/dllimport/visibility）。
- 桥接实现 `src/<mod>/c_<mod>.cc` 编进对应 `hyle_<mod>`，**显式构造** C++ 对象，不 reinterpret 布局。
- 纯 C smoke test `tests/<mod>/c_<mod>_test.c`（C 编译器编，只 include C 头）证明无 mangling/模板/STL 泄漏。
- 文档 `docs/guide/c-<mod>.md` 逐符号 1:1。
- 三平台真编真测（Win MSVC + WSL GCC + mac clang），mac 的严格 clang 常揪出隐式 include 依赖（如 `std::nothrow` 需 `<new>`）。
