# HYLE 编码规范（Coding Conventions）

> 这份管「每一行怎么写才不破坏范式」，与 [`engineering-charter.md`](./engineering-charter.md)（管"用什么"）分工。每条规则都带 **WHY**——人懂了原因才不会绕过。规则源自 clay 调研：取其为 ABI/性能而生、C++23 仍无更优解的写法，弃其只为补旧编译器坑的包袱。

## 0. 总纲：边界从严，内部从新

HYLE 的每条纪律都回溯到四个**约束源**：① ABI 稳定 ② 编译期成本 ③ 极致性能 ④ 跨平台。

**一条主轴**：跨 C-ABI 边界 / 公开头从严（沿用 clay 的硬纪律，因为这里 ABI 和性能承重）；库内部实现从新（自由用 C++23）。下面的"继承"几乎都在边界，"取代"几乎都在内部——这与 charter 的「内部 C++23 / 外部纯 C-ABI」二分是同一件事。

---

## 1. std 使用纪律（分层，非一刀切）

| 位置 | 规则 | 替代品 |
|---|---|---|
| 公开头 / 跨 DLL / C-ABI 边界 | **禁** std 拥有型容器（`std::string`/`vector`/`unordered_map`/`shared_ptr`/`function`）出现在签名或导出结构体 | `string_ref`(视图)、`carray`、flat hashmap、侵入式 `Ref<T>`、函数指针 |
| 库内部 `.cc` / 非导出类 | **自由用** std | — |

**WHY**：`std::string` 的 SSO 布局、`std::vector` 的三指针布局、allocator、控制块都是**未跨编译器/STL 版本稳定的 ABI**——泄进 C 接缝即废掉 ABI 承诺。clay 实证：`thread_pool.h` 内部用 `std::vector<std::thread>`/`priority_queue`，但跨边界一律走 `string_ref`/`carray`。

**硬线（可 lint）**：公开头里出现 `#include <string>`/`<vector>`/`<unordered_map>` 且用于导出符号 = 违规。

## 2. 头文件 / include 纪律

- **每模块一个 `decls.h`**：只放前向声明（`struct xxx_api_t;`、`struct Allocator;`）+ 导出宏，自身只 include `macros/platform/visibility/types` 这几个最小头（clay `sys/decls.h` 实证）。
- **公开头 include 最小化**：能前向声明就不 `#include`；重实现头不进公开头。
- 全部 `#pragma once`。

**WHY**：切断 include 传递链——降编译期、防依赖环、保证改一个实现头不引发全库重编。

## 3. 宏纪律

**何时才用宏**：仅当 C++23 的 `constexpr`/`inline`/`concept`/模板**做不到**时（代码生成、编译期分支、ABI 布局断言、跨平台属性）。能用语言特性就不用宏。

**铁律**：
1. **属性/平台宏必带 `#else` 兜底**——未知编译器要退化为安全空定义，绝不编译失败（clay `CLAY_NODISCARD`/`FORCE_INLINE`/`LIKELY` 三分支 + 兜底）。
2. **ABI 布局假设用 `static_assert` 固化**——`CLAY_PAD` 显式声明每处 padding（配 `-Wpadded`）、`static_assert(sizeof(T)==N)` 钉死跨边界结构体大小。
3. **警惕宏求值陷阱**——如 `LIKELY` 必须 `static_cast<bool>`（`__builtin_expect` 收 long），宏参数必须加括号。
4. **前缀 `HYLE_`**，全大写。

**取代**：`MANUALLY_ALIGNED_STRUCT` 的 `pragma pack` → `alignas` + 独立 `static_assert`；`debugbreak()` 的 `asm("int $3")` → `__builtin_debugtrap()`/`__debugbreak()`；`CLAY_INHERITS` 伪继承 → 纯 C++ 真继承（仅 C 兼容 POD 处才保留）。

## 4. C-ABI 边界范式（对外沟通的落地）

跨 DLL / 对外暴露能力，**统一用这套**（clay 最成熟资产）：

- **vtable 结构体 `xxx_api_t`**：纯函数指针结构体，`struct { inst* ; ret(*fn)(handle*, ...); }`；用 designated initializer 填表。
- **不透明实例**：只前向声明，后缀约定 **`_o`**（实例）/ **`_t`**（vtable/值类型）；调用方持指针、不知布局。
- **make/destroy 工厂**：`make_X()/destroy_X()` 配对，分配责任锁在 DLL 内（防 CRT 堆不匹配），禁客户端 `new/delete`。可返回带自定义 deleter 的 `unique_ptr`。
- **显式 `Allocator*` 注入**贯穿所有分配型 API，无隐藏全局堆。
- **`extern "C"` 只在真需要 C 链接的自由函数上用**——别误以为"包个 extern C 就算合规"，clay 的 ABI 稳定主要靠 vtable+不透明指针，不是 extern "C"。

**WHY**：C++ 无稳定 ABI（mangling/模板/异常跨边界）；vtable+不透明指针是同时满足"跨编译器稳定 + 可运行时替换实现 + 纯 C 可调"的唯一解。

## 5. 错误处理

- **零异常**：错误路径不 throw（继承 clay 哲学）。
- **内部用 `std::expected<T,E>`** 取代 clay 手写 `Result`（省 `aligned_storage`+`launder`）。
- **保留 `HYLE_ASSIGN_OR_RAISE` / `HYLE_RETURN_NOT_OK` 宏**——标准无 Rust `?` 等价，这是真增量。
- **状态码用 X-macro 单一数据源**（clay `CLAY_FOREACH_STATUS_CODE`）：加错误码只改一行，自动生成 enum + 工厂 + 判定。
- **零开销 OK 路径**：错误载体 OK 态零分配（参考 clay Status `rep_==0` 指针标签内联）。
- **`HYLE_ASSERT` 求值为 bool 且 release 不消失**：`if (!HYLE_ASSERT(x)) return ...;`，release 下记录而非 UB。

## 6. 内存与对象生命周期

- **侵入式 `RefCounted` + `Ref<T>`**（CRTP 内嵌 `atomic refcount`）：跨边界比 `shared_ptr` 省控制块、可定制 `destroy()`。`incref` 用 `relaxed`、`decref` 用 `acq_rel`（教科书级正确，照搬）。**改进**：转换运算符设 `explicit`，避免隐式误用。
- **显式 allocator 注入**（见 §4）。
- **单例默认 Meyers**（`static T inst` C++11 起线程安全），leaky 仅按需——丢掉 clay 的 leaky 默认。

## 7. 容器与数据布局

- **header-before-data 侵入式数组**（carray）：指针即数据、`a[i]` 零间接、cache 友好、可直传 C API。**改进**：C++23 包一层带 `[[nodiscard]]`/迭代器的薄封装补类型安全。
- **type-erased core + typed shell**：内联模板按类型在头里展开，重逻辑走 `HYLE_API` 的 `detail::*_internal(void*, item_size, ...)` 只编一份进 DLL。**保留为核心范式**（跨 DLL 防代码膨胀 + allocator 一致）。
- **`string_ref` 的 `null_terminated` 位**：相对 `std::string_view` 的真实增量——标志为真才能零拷贝当 C 串传。保留。
- **cache-line 对齐**：`HYLE_CACHE_LINE_SIZE 64`、`HYLE_ALIGNED`；并发结构体按 cache-line padding 防伪共享。
- **显式 padding**：`HYLE_PAD(n)` + `-Wpadded`，禁隐式填充。

## 8. 编译期设施

- **concepts/`requires` 取代 SFINAE**：clay 的 `detector`/`void_t`/`enable_if` 链 → C++23 concept，公开 API 必用 concept（可读 + 错误信息友好）。`detector` 仅留底层兼容。
- **编译期 `type_name<T>()` + 类型哈希**（`__PRETTY_FUNCTION__`/`__FUNCSIG__` 切片 + murmur）：无 RTTI 的稳定类型 ID。**保留但集中维护 + 单测**（每编译器前缀串脆弱）。reflection 要等 C++26。
- **`ScopeGuard` / `HYLE_DEFER`**：Go 式 defer，RAII 收尾。C++23 仍无标准 defer，保留（用 CTAD 简化）。
- **比较用 `operator<=>` + `=default`**，丢掉 clay 的 Comparable CRTP。
- **字符串拼接用 `std::format`**，丢掉 clay 的 `str()`。

## 9. 自注册 / 插件

- **Registry + `HYLE_DECLARE/DEFINE` 宏对**：静态初始化期自注册、跨库 dllimport/export、优先级覆盖（插件体系刚需）。**改进**：内部 `std::unordered_map` 换 HYLE flat hashmap（与"回到底层"一致）；禁 `std::exit`，走错误返回。

## 10. 速查表

**继承（风格基因，照搬/微调）**：C-ABI vtable+`_o/_t`+make/destroy+显式 allocator、Status 指针标签内联、X-macro 码表、`ASSIGN_OR_RAISE`、侵入式 RefCounted/Ref、编译期 type_name+哈希、ScopeGuard/DEFER、header-before-data carray、type-erased core+typed shell、string_ref null_terminated 位、`HYLE_PAD`+对齐 static_assert、cache-line 对齐、三态可见性宏、Registry 自注册、std 分层纪律、decls.h include 纪律。

**取代（C++23 吸收，留意图换实现）**：Comparable→`<=>`、SFINAE→concepts、Result 核心→`std::expected`、`str()`→`std::format`、catomics→`std::atomic`(留自旋锁 padding)、hashmap 宏 temp/哨兵→concept 模板(+SwissTable SIMD)、pragma pack→`alignas`、`same_as_t`→`std::type_identity_t`、Singleton leaky→Meyers。

**淘汰（补旧编译器坑的包袱）**：整份 `C++17.h` polyfill、`asm("int $3")`→`__builtin_debugtrap`、`CLAY_INHERITS` 伪继承→真继承、MSVC<1900/GCC5 退化分支、编译期 `sizeof(char[-1])` 类型校验技巧→`static_assert`/concept。
