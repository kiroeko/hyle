# C ABI 施工参考（给 HYLE 开发者：如何为一个模块导出 C ABI）

> **读者**：HYLE 开发者。收口某个模块（container / simd / sched / io / compute / graph）需要给它做 C ABI 导出时，照这份走——就像 mem/error/platform 已经做的那样。
> **配套**：本文件是 **how（照着做的步骤 + 模式 + checklist）**；**why（类型擦除、三层架构、为什么删版本协商）** 见 [`c-abi-design.md`](./c-abi-design.md)。上位依据 [`engineering-charter.md`](./engineering-charter.md) §6、[`coding-conventions.md`](./coding-conventions.md) §4。
> **样板**：`include/hyle/c/mem.h` + `src/mem/c_mem.cc` + `tests/mem/c_mem_test.c` + `docs/guide/c-mem.md` 是**最完整的活样板**，遇到本文没覆盖的情况以它为准。

## 0. 先判断：这个模块要导出什么

对模块的每个公开 C++ 面，按下表归类，决定它进不进 C ABI：

| C++ 面 | 判断 | 做法 |
|---|---|---|
| 无状态自由函数（`page_size()`、`vm_reserve()`） | **直接导出** | 一对一包一个 `hyle_*` 自由函数 |
| RAII 类（`VmRegion`/`MappedFile`/`StackAllocator`） | **导出为 opaque 句柄** | `_o` 句柄 + create/destroy + 方法转发 |
| 值语义 vtable（`Allocator`） | **导出为 POD struct** | 天然 C 形状（fn 指针 + state），显式桥接 |
| 模板 `Foo<T>` | **不导出类型；导出能力** | 类型擦除：`<T>`→运行时 `size/align/enum`（见 design §1）；typed 便利用 header-inline 宏 |
| 有状态调试/内省类（`StatsAllocator`/`ErrorRecord`） | **不导出** | 仅 C++；无 FFI 需求（可观测性走标量查询，学 ggml `ggml_used_mem`） |
| 用 `std::` 拥有型 / `std::function` / `std::span` 的签名 | **改形态再导出** | 换成 ptr+len、fn 指针+ctx、buffer-fill/visitor |

**判据不是"每个 C++ 符号都要有 C 对应"，是"能力有没有 C ABI 通路"**（design §1）。模板不可达≠能力缺失。

## 1. 文件与接线（六件套）

为模块 `<mod>` 做 C ABI，动这些：

1. **新建 `include/hyle/c/<mod>.h`** —— 纯 C 头（见 §2 骨架）。
2. **新建 `src/<mod>/c_<mod>.cc`** —— 桥接实现（见 §3）。
3. **改 `src/<mod>/CMakeLists.txt`** —— 把 `c_<mod>.cc` 加进 `hyle_add_module(<mod> SOURCES ... c_<mod>.cc)`。导出宏 `HYLE_<MOD>_BUILD` 由 `hyle_add_module` 自动定义（`cmake/HyleModule.cmake`），无需手动。
4. **新建 `tests/<mod>/c_<mod>_test.c`** —— 纯 C smoke（见 §4）。
5. **改 `tests/CMakeLists.txt`** —— 仿 `hyle_mem_c_test` 加一个 C 可执行：`add_executable` + `target_link_libraries(... PRIVATE hyle_<mod>)` + `set_source_files_properties(... LANGUAGE C)` + `hyle_apply_warnings`/`hyle_apply_sanitizer` + `add_test`。
6. **新建 `docs/guide/c-<mod>.md`** + 在 `docs/guide/README.md` 加索引行 —— 逐符号 1:1。

## 2. 头文件骨架（照抄 mem.h 顶部，替换 MOD/mod）

```c
/* SPDX: same as HYLE. Pure C ABI for the HYLE <mod> module (close-01).
 * Naming: every symbol is hyle_* / HYLE_*. Do NOT include hyle/<mod>/ from C.
 * Docs (1:1 with this header): docs/guide/c-<mod>.md */
#ifndef HYLE_C_<MOD>_H_
#define HYLE_C_<MOD>_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* export (same module as C++ <mod>: HYLE_<MOD>_BUILD) — three-state macro */
#if defined(HYLE_STATIC)
#  define HYLE_<MOD>_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_<MOD>_BUILD)
#    define HYLE_<MOD>_C_API __declspec(dllexport)
#  else
#    define HYLE_<MOD>_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_<MOD>_C_API __attribute__((visibility("default")))
#else
#  define HYLE_<MOD>_C_API
#endif

/* ... opaque handles + free functions ... */

#ifdef __cplusplus
}
#endif
#endif
```

**不要**加：`hyle_<mod>_api` 函数指针表、`hyle_<mod>_api_get`、`HYLE_<MOD>_C_ABI_VERSION` 宏。未发布阶段版本协商是负担（design §3）。纯 `extern "C"` 自由函数 + opaque 句柄即可，FFI 直接绑符号。

## 3. 桥接实现的三条铁律（src/c_<mod>.cc）

1. **opaque 句柄 = wrapper struct 内嵌 C++ 对象**，`_o` 类型**在 anon namespace 之外**定义（同 `hyle_vm_region_o`）：
   ```cpp
   struct hyle_foo_o { hyle::mod::Foo obj; /* ...ctor... */ };
   hyle_foo_o* hyle_foo_create(...) { return new (std::nothrow) hyle_foo_o{...}; }
   void hyle_foo_destroy(hyle_foo_o* f) { delete f; }  // delete on nullptr is safe
   ```
   句柄方法一律 `return f ? f->obj.method() : <默认>`（NULL 句柄安全）。

2. **绝不 reinterpret_cast 跨 ABI 布局**。C↔C++ 转换走**显式构造**或**逐字段拷贝**：
   - C→C++（把 C 回调交给 C++）：C 的 fn 指针是普通 `void*(*)(...)`，C++ 侧要 `noexcept` fn 类型——普通指针**不能隐式转 noexcept**，所以要 **noexcept trampoline** 转发（`c_backing_alloc` 那样），不是 cast。
   - C++→C（把 C++ vtable 交给 C）：`noexcept` fn 指针**能隐式转**普通 C 字段，逐字段填即可（`from_cpp` 那样）。
   - wrapper 若存了调用方的 C 回调结构体，**存在前、C++ 对象在后**（析构逆序，C++ dtor 可能回调 backing）。

3. **契约违反 fail-fast，边界故障返 NULL/错误码**（区分二者）：
   - 调用方契约违反（如 StackAllocator size==0）→ 让 C++ ctor 的 `HYLE_FATAL_ASSERT` 触发 abort，**不要**在 C 层静默返 NULL 掩盖（对齐 charter fail-fast）。
   - 真边界故障（backing 缺 fn 指针、`new` OOM、`base==NULL`、池耗尽）→ 返 NULL。
   - **句柄返回型 API 返 NULL 时，配一条 thread-local last-error 串**（如 mem 的 `hyle_mem_last_error`，转发 platform 的 `vm_set_last_error`）：NULL 只说"失败了"，C 用户还需知道"为什么失败"（invalid backing？OOM？池耗尽？）——与 error 模块的 `hyle_status_message_cstr` 呼应，跨 C 边界的失败都要能取到人可读原因。**每条** NULL 返回路径都要设串，别让 C 用户读到上一次的过时串。
   - 系统边界输入才校验（FFI 调用方可能传零值结构体）；内部/框架保证不重复校验。

4. **跨 ABI POD 要钉布局，输入 options 要能演进**：
   - 桥接 `.cc` 对公开 POD 至少
     `static_assert(std::is_standard_layout_v<T>)` 与
     `std::is_trivially_copyable_v<T>`；关键首字段/固定大小再 assert。
   - 可能追加字段的输入 options 以 `uint32_t struct_size` 为首字段，初始化
     宏/helper 写 `sizeof(T)`；库在读字段前先验证可读范围。不要把“全零
     内存”误写成可长期演进的 ABI 协议。
   - 小型输出 snapshot 可以值返回 POD，但发布后字段和布局即冻结；预期
     高频扩展的输出改为 `struct_size + out*`，或新增独立查询函数。
   - C enum 与 C++ enum/位值在桥接 `.cc` 用 `static_assert` 锁定，不能靠
     两份手抄表默契一致。

## 4. 易用性：两条线，各自做到位

**C ABI 用户（跨语言/FFI）的易用性靠 header-inline 便利层**——零新导出符号、不进 ABI、纯语法糖。这是必做的一个维度，不是可选。mem/error/platform 已立的模式，照抄：

- **便利转发**：`hyle_alloc/free/realloc`（经句柄调，省手解 fn 指针；NULL 句柄安全）。
- **typed 宏**（类型擦除的用户侧糖）：`HYLE_OBJECT_POOL_CREATE(backing,T,n)` 替用户填 `sizeof(T)`/`HYLE_ALIGNOF(T)`；`HYLE_ALIGNOF` 是跨 C/C++ 的 alignof shim（C++/C23 `alignof`、C11 `_Alignof`、否则 0→默认对齐）。
- **工厂 + 谓词**（error 模式）：`hyle_status_not_found()` 等全码工厂 + `hyle_status_is_*()` 谓词 + `HYLE_C_RETURN_IF_ERROR`。
- **opts 建造 + 尺寸初始化宏**（platform 模式）：`HYLE_THREAD_OPTS_INIT`
  + `hyle_thread_opts_set_*`；宏写 `struct_size`，因此应称“默认初始化”，
  不应承诺其字节全零。
- **buffer-fill / visitor**（避免返回拥有型内存）：`size_t f(char* dst, size_t cap)` 返完整长（`>=cap` 表截断）；迭代用 `int (*visit)(..., void* ctx)` 返 0 停。

原则：inline 糖给 C 用户便利、但不污染稳定导出面（不进 DLL 符号，零 ABI 风险）。**C++ 用户不管这层**——他们直接用原生 `include/hyle/<mod>/*.h`（RAII/模板本就好用），HYLE 不做"包 C 句柄的 C++ 壳"（design §2：只有对外主接口是 C 的库如 ggml/oneDNN 才需要那层）。

## 5. 纯 C smoke test（tests/c_<mod>_test.c）

- **纯 C 编译**（`LANGUAGE C`），只 include `hyle/c/<mod>.h`——证明无 mangling/模板/STL 泄漏。
- Windows 加 `#define _CRT_SECURE_NO_WARNINGS 1`；用一个 `CHECK` 宏累计失败、结尾打印 `<mod>_test: all checks passed`。
- 覆盖：每个句柄 create→用→destroy（含耗尽/边界）、每个自由函数、vtable 视图（`as_allocator` 那种）、便利宏、NULL 安全路径、系统边界校验（零值结构体→NULL）。

## 6. 三平台真编真测（收口硬门）

改完必须三平台真跑（不是"应该能过"）：
- **Win MSVC**（`build/`，vcvars + `cmake --build --target hyle_<mod>_c_test`）。
- **WSL Linux GCC-14**（`/root/hyle` 镜像，`CXX=g++-14`；`setarch -R` 跑，避 ASLR TSan 陷阱）。
- **mac arm64 clang**（`~/hyle` 镜像，`/opt/homebrew/bin/cmake`）——**clang 最严，常揪出隐式 include 依赖**（如 `std::nothrow` 需显式 `<new>`），别跳过。
- 都要：C 测 `all checks passed` + 对应 C++ 测试无回归 + `-Werror` 干净。
- **wsl.exe 陷阱**：`bash -lc` 里的 `$var` 会被外层 git-bash 吃掉——多变量逻辑写成 `.sh` 文件再跑（见 reference 记忆）。

## 7. 收口 checklist（机械跑一遍）

- [ ] 每个 C++ 公开面按 §0 表归类，缺口补或显式登记"有意仅 C++"（附理由：泛型不可达 vs 无 FFI 需求，别混）。
- [ ] 六件套齐（§1）；导出宏三态；`extern "C"` 包全。
- [ ] 桥接三铁律（§3）：opaque 句柄 anon-ns 外定义 + 零 reinterpret + fail-fast/边界二分。
- [ ] C POD 有 layout/trivial/关键 offset assert；可扩展输入 options 首字段为 `struct_size`；C/C++ enum 值有编译期锁。
- [ ] 易用性 header-inline 层（§4）：便利转发 + typed 宏 + 工厂/谓词按模块需要补。
- [ ] **无** api 表 / api_get / 版本宏（design §3）。
- [ ] 纯 C smoke test 覆盖句柄生命周期 + 便利 + NULL 安全 + 边界校验。
- [ ] 机械核对 C 头每个 `hyle_*` 入口在纯 C smoke 与 guide 中至少出现一次；聚合写法不能掩盖漏测符号。
- [ ] 三平台真编真测 all passed + C++ 无回归 + `-Werror` 干净。
- [ ] `docs/guide/c-<mod>.md` 逐符号 1:1 + README 索引；close-01 里程碑登记该模块 C ABI 落地。
- [ ] `docs/guide/c-<mod>.md` 的"有意不导出"表说清每个仅-C++ 项的理由。
