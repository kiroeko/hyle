# util-06：demangle 运行期符号反修饰

> **2026-07-19 收口复审**：Windows 对超过 1 KiB 的合法 `?` decorated symbol
> 现会有界扩容重试到 1 MiB；buffer 合同统一 fail-fast，并提供纯 C buffer-fill
> 入口。完整归档见 [`util-module-closure.md`](./util-module-closure.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | util/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；util 46→52 test）|

> **范围**：吸纳 clay `sys/demangle.h`+`.cc`（28+45 行）。运行期把 mangled 符号（`_Z1gv` → `g()`）转人类可读，供诊断/backtrace。是 util-02 `type_name<T>`（编译期、静态类型、无 RTTI）的**运行期补集**——处理只有运行时才知道的名字（backtrace 帧、多态 `typeid(x).name()`）。
> **cloc**：**生产 65 行**(demangle.h 14 + demangle.cc 51) + **测试 46 行**。

## clay API diff + 改进

- **覆盖**：clay `demangle(name, Allocator*)` → HYLE `demangle(mangled, out, cap)`。
- **改进①(无分配 API)**：clay 返回 `Allocator*` 分配的 `char*`（隐藏所有权），且把 `__cxa_demangle` 的 malloc 结果包进 `std::function`-deleter 的 `unique_ptr`（为一次 `free()` 建堆闭包）。HYLE buffer-fill（snprintf 语义），libc 临时块在内部 free，**API 表面零分配、零 std::function**。
- **改进②(MSVC 真做)**：clay 非 Itanium 路径直接回显 mangled 名。HYLE 用 `UnDecorateSymbolName`(dbghelp) 真反修饰；DbgHelp 单线程，静态 mutex 串行化（非热路径）。
- **gap 登记**：clay `demangle_type<T>()`(模板，需 RTTI) **不移植**——`type_name<T>()`(util-02) 覆盖编译期静态类型且无需 RTTI 更优；运行期多态类型走 `demangle(typeid(x).name(), ...)`。非缺口，是更优替代。

## 对抗式自审（六类）

- **① 正确性**：`fill` cap>0 复制 min(n,cap-1)+NUL；cap==0 不写只回长度（snprintf 语义真测）；`__cxa_demangle` status==0&&非空才用、否则 `free(null)` 安全回退原名；空串/"main"/截断/null/零容量五边界全测，双平台往返 `typeid` 名含简名。
- **② 性能**：非热路径(诊断/backtrace)。`__cxa_demangle` 的 malloc 是 Itanium ABI 强制(无 in-place 变体)；MSVC mutex 是 DbgHelp 要求。常见路径用栈 1KB buffer，超长 MSVC symbol 才有界分配重试。
- **③ 规范**：公开头只 `<cstddef>`+visibility；无 std 拥有型出签名；buffer-fill；导出宏在函数上；无 iostream/std 垃圾容器。`.cc` 内部用 `std::mutex`/`std::lock_guard`(MSVC 路径，不出头)合规。
- **④ clay 覆盖**：demangle 主体✅ + 无分配/MSVC 真做两改进；demangle_type gap 已登记为更优替代。
- **⑤ 残留脚手架**：无新增。
- **⑥ 过度复杂**：单函数 + 一个 fill 助手；比 clay unique_ptr+std::function 简洁。
- **"比 std 快"**：N/A——std 无运行期 demangle（编译器 ABI 设施），这是 std 没有的**能力**而非可对标项。

## 交付物

- `include/hyle/util/demangle.h`（`demangle` buffer-fill + `HYLE_UTIL_API` 导出宏）
- `src/util/demangle.cc`（GCC/Clang `__cxa_demangle` / MSVC `UnDecorateSymbolName`）
- `tests/util/demangle_test.cc`（6）
- CMake：util 从 INTERFACE 改编译库（`hyle_add_module(util SOURCES demangle.cc)`）；Win dbghelp 经 `#pragma comment(lib)` 链接。

## 参考

- clay `sys/demangle.h`+`src/clay/sys/demangle.cc`。conventions §1(buffer-fill 替 std 拥有型)、§8(type_name 编译期补集)。
