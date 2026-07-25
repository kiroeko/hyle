# HYLE 编码规范（Coding Conventions）

> 这份管「每一行怎么写才不破坏范式」，与 [`engineering-charter.md`](./engineering-charter.md)（管"用什么"）分工。每条规则都带 **WHY**——人懂了原因才不会绕过。规则源自 clay 调研：取其为 ABI/性能而生、C++23 仍无更优解的写法，弃其只为补旧编译器坑的包袱。

## 0. 总纲：边界从严，内部从新

HYLE 的每条纪律都回溯到四个**约束源**：① ABI 稳定 ② 编译期成本 ③ 极致性能 ④ 跨平台。

**一条主轴**：跨 C-ABI 边界 / 公开头从严（沿用 clay 的硬纪律，因为这里 ABI 和性能承重）；库内部实现从新（自由用 C++23）。下面的"继承"几乎都在边界，"取代"几乎都在内部——这与 charter 的「内部 C++23 / 外部纯 C-ABI」二分是同一件事。

### 两条贯穿性工作纪律（用户定，统领全局）

1. **自家 bug 绝不 catch（fail-fast）；throw 可用但谨慎、用数据说话**：bug 类失败响亮地崩（`HYLE_FATAL_ASSERT`/abort + 诊断），**绝不 catch 吞掉自己的 bug**；运营失败显式返 `Status`/`expected`。**throw 不是教条禁止**——默认 no-throw，要破例须满足"不伤热路径 + 为隔壁稳定 + catch 的是外来异常非自家 bug"且**拿测量数据/后果撑**。catch 只允许在 adapter 边界做"外来异常→Status"转换、绝不外泄。`-fno-exceptions` 旗标本身=实测决定（zero-cost 异常不伤热路径运行时，但旗标买二进制/优化，量了再定）。**WHY**：catch 自家 bug = 把崩溃推迟到更难查处；异常跨 C-ABI 是 UB。详见 [`lifecycle-design.md`](./lifecycle-design.md) §0。
2. **没想透的硬问题标"待调研"，绝不用半成品方案带偏后续实现**：拿不准的设计点（如 GPU 在飞取消的更细粒度、-fno-exceptions 下借入 C++ 库的隔离 TU 机制），**只写"已知事实 + 参考实证 + 待调研标记"，不写处方**。**WHY**：过早锁死的半成品方案会污染后续实现、比留白更难纠偏；留到该模块开工前带专业调查+实验再定。**性能优先**——拿不准时倾向更激进的零开销选择，但标注为待验证而非既定。

---

## 1. std 使用纪律（分层，非一刀切）

| 位置 | 规则 | 替代品 |
|---|---|---|
| 公开头 / 跨 DLL / C-ABI 边界 | **禁** std 拥有型容器（`std::string`/`vector`/`unordered_map`/`shared_ptr`/`function`）出现在签名或导出结构体 | `string_ref`(视图)、`carray`、flat hashmap、侵入式 `Ref<T>`、函数指针 |
| 库内部 `.cc` / 非导出类 | **自由用** std | — |

**WHY**：`std::string` 的 SSO 布局、`std::vector` 的三指针布局、allocator、控制块都是**未跨编译器/STL 版本稳定的 ABI**——泄进 C 接缝即废掉 ABI 承诺。clay 实证：`thread_pool.h` 内部用 `std::vector<std::thread>`/`priority_queue`，但跨边界一律走 `string_ref`/`carray`。

**硬线（可 lint）**：公开头里出现 `#include <string>`/`<vector>`/`<unordered_map>` 且用于导出符号 = 违规。

**禁 std 流（铁律）**：**全库不用 `<iostream>`/`std::cout`/`std::cerr`/`std::endl`/`<sstream>`/`std::ostream`**。理由：std 流体积大、慢（同步开销/locale/虚函数链）、ABI 不稳、错误处理糟糕——对极致性能 infra 是负担。诊断输出用 C 的 `std::fprintf(stderr, ...)`（printf 风格，如 `HYLE_FATAL_ASSERT`）；将来结构化输出走 HYLE 自己的 log（obs-01）。格式化用 `std::format`（值语义、无流），不用 `<<`。

**禁 std 性能垃圾容器/哈希（铁律,用户 2026-06）**：生产代码**不用 `std::unordered_map`/`unordered_set`/`map`/`set`/`std::hash`**——它们慢(节点分配/指针追逐/弱哈希)、ABI 不稳。**有 HYLE 自己的就用自己的**:hashmap→`SwissTable`(container-04)、哈希→XXH3 `Hash<T>`(container-02)、动态数组→`CArray`/`SmallVector`、字符串→`string_ref`/`string_buf`(container-06)。
- **唯一例外:测试**可用 std 容器/哈希做**对拍 oracle**(参照基准),非生产路径。
- `std::vector`/`std::unique_ptr` 在内部 `.cc` 做平凡局部 scratch 仍可(非性能热点、不出公开头);但凡 HYLE 有等价高性能物的场景,优先 HYLE 的。
- **这条与"红队攻击残留脚手架"联动**:早期因依赖顺序临时用了 std 的,等 HYLE 等价物落地后**必须替换**(见红队纪律 + clay-coverage-map 脚手架债登记)。

## 2. 头文件 / include 纪律

- **每模块一个 `decls.h`**：只放前向声明（`struct xxx_api_t;`、`struct Allocator;`）+ 导出宏，自身只 include `macros/platform/visibility/types` 这几个最小头（clay `sys/decls.h` 实证）。
- **公开头 include 最小化**：能前向声明就不 `#include`；重实现头不进公开头。
- **按语言边界选择 size 头和名字**：原生 C++ 头 `include/hyle/<mod>/*.h` 使用
  `<cstddef>`、`std::size_t`、`std::byte`；只有必须被 C11/C17 编译器直接消费的
  `include/hyle/c/*.h` 使用 `<stddef.h>` 和全局 `size_t`。不要因为 HYLE 有 C ABI 就把
  C 写法带进主体，也不要把 C ABI 头机械改成 `<cstddef>`（标准只保证
  `std::size_t`，且 C 编译器不能解析该头）。
- 全部 `#pragma once`。

**WHY**：切断 include 传递链——降编译期、防依赖环、保证改一个实现头不引发全库重编。

## 3. 宏纪律

**何时才用宏**：仅当 C++23 的 `constexpr`/`inline`/`concept`/模板**做不到**时（代码生成、编译期分支、ABI 布局断言、跨平台属性）。能用语言特性就不用宏。

**铁律**：
1. **属性/平台宏必带 `#else` 兜底**——未知编译器要退化为安全空定义，绝不编译失败（clay `CLAY_NODISCARD`/`FORCE_INLINE`/`LIKELY` 三分支 + 兜底）。
2. **ABI 布局假设用 `static_assert` 固化**——`CLAY_PAD` 显式声明每处 padding（配 `-Wpadded`）、`static_assert(sizeof(T)==N)` 钉死跨边界结构体大小。
3. **警惕宏求值陷阱**——如 `LIKELY` 必须 `static_cast<bool>`（`__builtin_expect` 收 long），宏参数必须加括号。
4. **前缀 `HYLE_`**，全大写。
5. **导出宏放方法、不放类（当类有非 POD 成员时）**——`HYLE_<MOD>_API` 标在**成员函数声明**上，而非 `class X` 前。若导出整个类、而类有 `std::atomic`/`Allocator`/其它非平凡成员，MSVC 报 **C4251**（成员类型未导出）。把宏放在需跨 DLL 调用的方法上，类布局留客户端实例化即可消除。实证：Parker/Fiber/Worker 三处同款。纯 POD 成员的类（如 `Status` 单 `uintptr_t`）不受此限。

**取代**：`MANUALLY_ALIGNED_STRUCT` 的 `pragma pack` → `alignas` + 独立 `static_assert`；`debugbreak()` 的 `asm("int $3")` → `__builtin_debugtrap()`/`__debugbreak()`；`CLAY_INHERITS` 伪继承 → 纯 C++ 真继承（仅 C 兼容 POD 处才保留）。

## 4. C-ABI 边界范式（对外沟通的落地）

跨 DLL / 对外暴露能力，**统一用这套**（clay 最成熟资产）：

- **vtable 结构体 `xxx_api_t`**：纯函数指针结构体，`struct { inst* ; ret(*fn)(handle*, ...); }`；用 designated initializer 填表。
- **不透明实例**：只前向声明，后缀约定 **`_o`**（实例）/ **`_t`**（vtable/值类型）；调用方持指针、不知布局。
- **make/destroy 工厂**：`make_X()/destroy_X()` 配对，分配责任锁在 DLL 内（防 CRT 堆不匹配），禁客户端 `new/delete`。可返回带自定义 deleter 的 `unique_ptr`。
- **分配显式可控,默认零间接**:默认走编译期 `SystemAllocator`（static/source 内联直达 mimalloc；shared 经 `hyle_mem` 一次 direct import，不泄漏 vendor ABI）；需自定义分配器(arena/mock/跨 DLL)时**显式 opt-in 注入** `Allocator` 句柄。无隐藏全局堆。注入是能力非强制默认(详见 mem-design「分配器分层」)。
- **`extern "C"` 只在真需要 C 链接的自由函数上用**——别误以为"包个 extern C 就算合规"，clay 的 ABI 稳定主要靠 vtable+不透明指针，不是 extern "C"。

**WHY**：C++ 无稳定 ABI（mangling/模板/异常跨边界）；vtable+不透明指针是同时满足"跨编译器稳定 + 可运行时替换实现 + 纯 C 可调"的唯一解。

## 5. 错误处理

- **零异常**：错误路径不 throw（继承 clay 哲学）。
- **内部用 `std::expected<T,E>`** 取代 clay 手写 `Result`（省 `aligned_storage`+`launder`）。
- **只保留 `HYLE_ASSIGN_OR_RETURN` / `HYLE_RETURN_IF_ERROR` 两个传播宏**——标准无 Rust `?` 等价；命名与成熟库一致，并由真实产品调用验证。
- **状态码用 X-macro 单一数据源**（clay `CLAY_FOREACH_STATUS_CODE`）：加错误码只改一行，自动生成 enum + 工厂 + 判定。
- **零开销 OK 路径**：错误载体 OK 态零分配（参考 clay Status `rep_==0` 指针标签内联）。
- **语义边界显式增加上下文**：外部 translator 保留 operation/原生错误名/数值；中间层仅在增加用户语义时调用 `Status::WithContext`。
- **失败通道不重叠**：运营失败返 Status，程序错误 `HYLE_FATAL_ASSERT`，人读日志归 obs；不存在 soft assert 或 error 私有 handler。

## 6. 内存与对象生命周期

- **侵入式 `RefCounted` + `Ref<T>`**（CRTP 内嵌 `atomic refcount`）：跨边界比 `shared_ptr` 省控制块、可定制 `destroy()`。`incref` 用 `relaxed`；`decref` 用 release，只有末次释放补 acquire fence。批量计数拒绝零、溢出和下溢；`RefCompatible` 拒绝会经非虚基类错误析构的形状。转换运算符显式，避免隐式误用。
- **默认编译期 `SystemAllocator`,注入 opt-in**（见 §4）。
- **单例默认 Meyers**（`static T inst` C++11 起线程安全），leaky 仅按需——丢掉 clay 的 leaky 默认。

## 7. 容器与数据布局

- **header-before-data 侵入式数组**（carray）：指针即数据、`a[i]` 零间接、cache 友好、可直传 C API。**改进**：C++23 包一层带 `[[nodiscard]]`/迭代器的薄封装补类型安全。
- **type-erased core + typed shell**：内联模板按类型在头里展开，重逻辑走 `HYLE_API` 的 `detail::*_internal(void*, item_size, ...)` 只编一份进 DLL。**保留为核心范式**（跨 DLL 防代码膨胀 + allocator 一致）。
- **`string_ref` 的 `null_terminated` 位**：相对 `std::string_view` 的真实增量——标志为真才能零拷贝当 C 串传。保留。
- **cache-line 对齐**：C++ 使用 `core::kDestructiveInterferenceSize`，C/公共布局使用
  `HYLE_STATIC_CACHE_LINE_SIZE`；Apple Silicon 为 128B，其余当前目标 64B。运行时算法事实用
  `platform::cache_line_size()`，不得拿 runtime 值形成公共类型 ABI。
- **padding**：优先 `alignas` + 真实字段布局与 `static_assert`；不保留只为压制 benchmark/
  `-Wpadded` 的匿名 padding 宏。

## 8. 编译期设施

- **concepts/`requires` 取代 SFINAE**：clay 的 `detector`/`void_t`/`enable_if` 链 → C++23 concept，公开 API 必用 concept（可读 + 错误信息友好）。`detector` 仅留底层兼容。
- **编译期 `type_name<T>()` + 类型哈希**（`__PRETTY_FUNCTION__`/`__FUNCSIG__` 切片 + murmur）：无 RTTI 的稳定类型 ID。**保留但集中维护 + 单测**（每编译器前缀串脆弱）。reflection 要等 C++26。
- **`ScopeGuard` / `HYLE_DEFER`**：Go 式 defer，RAII 收尾。C++23 仍无标准 defer，保留（用 CTAD 简化）。
- **比较用 `operator<=>` + `=default`**，丢掉 clay 的 Comparable CRTP。
- **字符串拼接用 `std::format`**，丢掉 clay 的 `str()`。

## 9. 自注册 / 插件

- **Registry + `HYLE_DECLARE/DEFINE` 宏对**：静态初始化期自注册、跨库 dllimport/export、优先级覆盖。内部用 HYLE SwissTable；所有表访问同步，creator 在锁外调用以允许重入；同优先级接线冲突 fail-fast，不用 `std::exit`。

## 10. 能力 dispatch（编译期消变体，运行期最多冷跳一次）

吃各家硬件独家加速（AMX/VNNI、Vulkan cooperative matrix、Blackwell FP4、DSA…）时，**绝不在热路径放运行时间接**。按"何时知道硬件"分层，**热路径任何一层都零间接**：

> **前置结论（2026-06-28，纠"全用编译选项"的误解）**：硬件特性代码（SIMD/CUDA/Vulkan）**不是、也不该全用编译选项搞定**。编译选项（`-march`/`#ifdef`）只覆盖 **SIMD 的"自编自用"半场**；**分发二进制（HYLE 是 C-ABI `.so`，正属此类）必须走运行期 dispatch**——否则 `-march=native` 编出的 `.so` 在老 CPU 上直接 SIGILL。这也是**普遍做法**：glibc IFUNC / NumPy / FFmpeg / OpenCV / llama.cpp 官方包**全是运行期选 ISA**，纯编译选项只见于 Gentoo/HPC 自编。**GPU 更不能用编译选项**：CUDA = fat-binary（多架构 PTX）+ 驱动运行期 JIT + dlopen（无卡不能硬链）；Vulkan = SPIR-V 可移植中间码 + 驱动运行期编。**结论：编译门控可作 SIMD 起步（如 SwissTable match_empty 的 `HYLE_HAVE_SSSE3` 门控，SSSE3 现为全局 baseline），但不能停在那——simd 一开张就要搭 `dispatch.h`（运行期 CPUID 选变体）骨架；GPU 后端一律 dlopen，核心 `.so` 不硬依赖 CUDA/Vulkan 在场。** 详见下表三层模型。

| 层 | 何时 | 机制 | 热路径开销 | 地位 |
|---|---|---|---|---|
| **构建期** | 为已知硬件编 | 宏/feature flag + 每架构独立 TU + `-march=native`（llama.cpp `GGML_NATIVE` 本地自编默认）| **零**（错误路径不编出来）| 一等公民（自编/已知硬件最优）|
| **编译期** | 变体内部 | `if constexpr` + concepts + 模板单态化 | **零**（死分支消失、全内联）| 始终用 |
| **运行期** | 一个二进制发到未知 CPU | 同源码按 ISA 编 N 份（**每变体独立 DLL**）+ 启动 CPUID **打分选最高分变体** + dlopen，**冷入口跳一次** | **进去后零间接**（满速跑该变体）| 一等公民（**分发二进制的常态**）|

**层级关系（校正自真实 llama.cpp）**：两种**都是一等公民，按部署模型选，谁都不是"罕见兜底"**：
- **自编 / 已知硬件**（你为自己机器 `-march=native`）→ 构建期，运行期零 dispatch、极致。
- **分发单一二进制吃异构硬件** → 运行期多变体。**llama.cpp 官方分发恰是这条**（CI 用 `GGML_NATIVE=OFF` + `GGML_CPU_ALL_VARIANTS=ON` + `GGML_BACKEND_DL=ON`，把 ~13 个 x86 变体各编成独立 DLL，启动 `ggml_backend_score()` CPUID 打分选优）。**不是 IFUNC/全局指针，是"每变体独立 DLL + load 时 score"**——这个粒度反而正合下面"组合优于包含"。

**探测器隔离（llama.cpp 实证的坑，必须照做）**：选变体用的 CPUID 探测代码**必须编在 baseline ISA、独立 TU、禁 LTO**。否则 LTO 把高级 arch 指令内联进探测函数，老 CPU 上"用来探测的代码"自己先 SIGILL。HYLE `simd/dispatch.h` 照此。

**组合优于包含（铁律）**：每个特化是独立组合单元（一个 TU / 一个 tag 实现 / **一个注册的变体 DLL/后端**），**不是**上帝类内部 `switch(caps)` 或 vtable 分支。验收标准：**加一个厂商特化 = 加一个组合单元、改 1 文件、0 调用点改动、0 热路径开销**。

**统一探测**：CPUID(AMX/VNNI/AVX512)、GPU caps(cooperative matrix/FP4)、加速器在场(DSA/QAT)——一处探测、各模块查。`simd/`(指令集)、`compute/`(设备/扩展)、`io/`(SPDK 等后端)、`mem/`(大页/NUMA 拓扑) 全走这同一套模型，只是粒度不同。可照镜 ggml 的 `dev_caps`/`get_proc_address`（取可选扩展函数、不破 vtable）。

**HYLE 不写厂商 kernel**：框架负责「插得进、选得对、有地板兜底」；AMX GEMM/FP4 kernel 本身多数**借 ggml**（已有 ggml-amx/cuda/vulkan），少数 SIMD 级自写。kernel 不因框架变免费——这是诚实边界。

## 11. 速查表

**继承（风格基因，照搬/微调）**：C-ABI vtable+`_o/_t`+make/destroy+显式 allocator、Status 指针标签内联、X-macro 码表、传播宏（现名 `ASSIGN_OR_RETURN`）、侵入式 RefCounted/Ref、编译期 type_name+哈希、ScopeGuard/DEFER、header-before-data carray、type-erased core+typed shell、string_ref null_terminated 位、显式布局 static_assert、类型化 cache-line 对齐、三态可见性宏、Registry 自注册、std 分层纪律、decls.h include 纪律。

**取代（C++23 吸收，留意图换实现）**：Comparable→`<=>`、SFINAE→concepts、Result 核心→`std::expected`、`str()`→`std::format`、catomics→`std::atomic`(留自旋锁 padding)、hashmap 宏 temp/哨兵→concept 模板(+SwissTable SIMD)、pragma pack→`alignas`、`same_as_t`→`std::type_identity_t`、Singleton leaky→Meyers。

**淘汰（补旧编译器坑的包袱）**：整份 `C++17.h` polyfill、`asm("int $3")`→`__builtin_debugtrap`、`CLAY_INHERITS` 伪继承→真继承、MSVC<1900/GCC5 退化分支、编译期 `sizeof(char[-1])` 类型校验技巧→`static_assert`/concept。
