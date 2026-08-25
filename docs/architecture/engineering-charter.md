# HYLE 工程章程（Engineering Charter）

> 动手前的最终工程依据。覆盖：scope/形状、标准、开发范式、测试方案、边界、对外沟通。
> scope 细节见 [`hyle-scope.md`](./hyle-scope.md)，模块划分与设计原则见 [`README.md`](./README.md)。
> 本章程的决策基于两份调研：clay 既成约定盘点 + 2026 外部最佳实践，凡决策附依据。

## 1. Scope 与形状（收口）

通用高性能 C++ infra：基础基质 + `mem / container / simd / math / sched / io / compute / graph`
等支柱模块。地基继承 clay 并重新收口、支柱自造；这是设计来源，不再用
「1/3·2/3」当 LOC 预算。当前四项实测与终态纪律只见
[`hyle-scope.md`](./hyle-scope.md)；实时任务进度以 milestones/README 为唯一信源。不特化、不往应用层叠。

## 2. 标准

### C++ 标准：**C++23**
- 目标工具链基线：**GCC 14+ / Clang 18+ / MSVC 19.43+**（≈2024 主流）。
- 依据（cppreference 实测）：`std::expected` GCC12/Clang16/MSVC19.33、monadic ops GCC13/Clang17/MSVC19.36、deducing this GCC14/Clang18/MSVC19.43——核心特性 2022–2023 即全支持。
- 不依赖过新特性：`std::flat_map`(Clang20)、`std::mdspan`(GCC16) 不纳入硬依赖（HYLE 自造容器替代）。
- **modules 暂缓**（2026 仍不适合做构建基础）；**executors 用 [NVIDIA/stdexec](https://github.com/nvidia/stdexec)**，不等 C++26 STL。
- clay 是 C++17（手写 SFINAE、已废弃的 `aligned_storage`/`result_of`）——继承代码迁移时一并现代化。

### 错误模型：**零异常 + `std::expected`**
- 沿用 clay 的零异常哲学（错误路径不 throw）。
- **用 `std::expected<T,E>` 替代 clay 手写的 `Result<T>`**（C++23 原生，省去 `aligned_storage` 联合 + SFINAE 链）。
- 状态码沿用 clay 的 `enum class StatusCode`（X-macro 生成）思路。
- 传播宏（类比 clay `CLAY_ASSIGN_OR_RAISE`）保留，改名 `HYLE_*`。

### 命名 / 宏 / 可见性（继承 clay，改前缀 `HYLE_`）
- `namespace hyle` + 内部 `hyle::detail`。
- C++ 重类型大驼峰（`Allocator`/`SharedPool`/`SharedPoolCache`）；Native FFI schema 使用
  `hyle_*` snake_case、opaque handle 与固定布局 POD，不暴露 C++ 类型。
- 导出宏三态（static/dllexport/dllimport，类比 `CLAY_SYS_API`）；`HYLE_FORCE_INLINE`、
  `HYLE_LIKELY/UNLIKELY`、类型化 compile-time cache layout、`HYLE_DISABLE_COPY*`。
- 删除 clay 里的老编译器退化分支（MSVC<1900、GCC5 等）。

## 3. 开发范式

- **协程驱动 fiber**：`co_await` 作为 fiber 让出/唤醒的语言机制（C++20 协程三编译器已稳）。
- **分配显式可控、默认零间接**：容器默认编译期 `SystemAllocator`（静态/源码形状内联直达 mimalloc；shared SDK 经 `hyle_mem` 一次直接 import，vendor ABI 不外泄），运行时 `Allocator` 句柄（按值传的 `{alloc/free/realloc fn, state}`）按需 opt-in 注入（arena/mock/跨 DLL）。无隐藏全局堆。
- **内部完整 C++23**（模板/SIMD/容器尽情用）；C++ 头是正式 source API，跨语言 binary 边界才走
  direct Native FFI（见 §6），所以 C++ 对象布局不进入外语 ABI。
- **贴硬件**：cache-line 对齐、分支预测标注、无锁优先（基于 clay 的 catomics）。
- **构建**：CMake；修掉 clay 的反模式（Clang 硬编码 `-g -O0`、`-march=native` 损害可分发性）。默认编译为共享库（见 §6）。

## 4. 测试方案

> clay infra **零测试**是其最大负债。HYLE 从第一天建设施，新代码必带测试。

- **单测框架：GoogleTest**（folly/abseil/Chromium 同款，mock 完整，iron 已 vendored googletest 可直接复用）。
- **微基准：[nanobench](https://github.com/martinus/nanobench)**（单头、精度高、贴硬件热点）；必要时 Google Benchmark 出对外可比报告。
- **并发正确性**：TSan（主查 data race）/ ASan / UBSan **分独立 CI 构建**互斥跑；自研无锁原语配 TSan annotation 避免误报。
- **Fuzzing**：解析/序列化输入面用 libFuzzer。
- **CI 矩阵**：三编译器（GCC/Clang/MSVC）× 三平台（Linux/Windows/macOS）× sanitizer 构建。

### 覆盖率政策（分 Tier，核心接近 100%）

**核心要点：覆盖率是"卫生地板"，不是"正确性证明"。** 对 HYLE 最难的无锁/并发代码，100% 行覆盖照样能藏 ABA / memory-order / race bug——真正的正确性闸门是 sanitizer + 压力测试 + 模型检查，优先级高于覆盖率数字。

| Tier | 行/分支覆盖 | 更高优先级的正确性闸门 |
|---|---|---|
| **T0**（mem/container/sched）| **接近 100%（硬要求）** | TSan/ASan/UBSan 必过 + 无锁结构压力测试/模型检查（loom/CDSChecker 式枚举交错）+ 故障注入测 OOM/错误路径 |
| **T1**（simd/io/compute）| **≥90%** | 每后端对拍参考实现 + 跨平台矩阵聚合；AVX512/GPU 需真机 runner |
| **T2/T3** | **80%+（务实）** | 边界测试为主，不为数字测 trivial getter |

**三个必须认清的现实**：
1. **覆盖率 ≠ 正确性**：并发 bug 活在线程交错里、不在代码行里。无锁代码的闸门是 TSan + 压力测试，不是百分比。
2. **"100%" = 跨平台矩阵聚合**：单机跑不出全部——fcontext 6 平台汇编、io 四后端、SIMD 多 ISA、park 原语互斥；AVX512/GPU 需真实硬件 runner，没有就是覆盖硬缺口。按平台分别测、CI 矩阵聚合。
3. **防御性/不可达分支**：OOM、"不可能" assert 守卫——靠**故障注入**够到（好），**不为数字删守卫**（坏），实在不可达则显式标注豁免。
4. **成本曲线**：90→100% 陡峭，最后几个 % 别耗在扭曲代码够错误路径上——那份精力投到 TSan/fuzzing 回报高得多。

## 5. 边界

- **模块公开/内部二分**：每模块有 `include/`（公开）与 `src/`（内部）。公开头不泄漏内部实现类型。
- **binary ABI 承诺只在 Native FFI 接缝**：原生 C++ 头是正式 source API，但不承诺跨编译器/
  跨 STL 二进制兼容；跨语言稳定边界由 shared artifact 的 C calling convention、真实导出和
  明确 layout/ownership 共同构成（§6），不是“存在一个 C 头”单独决定。
- **依赖方向**：`sched/` 是 `io/`、`compute/` 的 completion 共同地基，单向依赖；`graph/` 跑在 `sched/` 上。无环。
- **第三方解耦**：ggml 等通过 adapter 对接上游 pin 版，HYLE 抽象不暴露第三方类型（见 compute-design.md）。

## 6. 对外沟通 / FFI（关键决策）

### 决策：**C++23 source API + shared library 的 direct Native FFI**

- HYLE 没有 native-C 客户，也不维护另一套面向 C 程序员的产品 API。
- `include/hyle/c/*.h` 使用 C 语法，是为了让 Java FFM、.NET P/Invoke 以及未来正式承诺的
  Rust/Python binding 描述稳定 calling convention、整数宽度、POD 和 opaque handle。
- direct FFI 能力必须是 shared artifact 中真实导出的 unmangled symbol；`static inline`、macro、
  透明自引用结构和调用方另编译的 C recipe 都不是动态能力。
- 标准形状是 exported free function、opaque handle + create/destroy、可演进 fixed-width POD、
  callback + context、ptr + len/明确 buffer ownership，以及能摊薄边界成本的 batch operation。
- 每模块先从真实 foreign-runtime 任务裁决是否需要 direct FFI；没有任务可明确记 `FFI=N/A`，
  不为追求 C++ parity 制造导出。只有保留 direct FFI 的模块，最低验收才是 Java 与 .NET 从精确
  `.dll/.so/.dylib` 路径无 shim 直绑；正式发布其他 binding package 时，才把其支持平台加入
  发布矩阵。C schema 编译测试只是卫生门。
- headers + static 与完整源码是 C++ 交付形式；headers + shared 同时承载 C++ 动态 SDK 和 Native
  FFI。源码客户当然可以自行构建 shared artifact，但源码包本身不是可加载 ABI。

### 与 clay 的关系（回答"clay 是不是也这样保 ABI 稳定"）
- **方向一致**：clay 已用「函数指针 vtable + 不透明指针 + make/destroy 工厂 + 显式 Allocator
  注入」穿透 DLL 边界、避免导出 C++ 类布局/STL。
- **但 clay 没到纯 C 头**：它仍是 C++ 头（`namespace clay`、模板 `carray`、`string_buf_t` 是纯虚接口依赖 C++ ABI），保的是"同一 C++ 工具链下的跨 DLL 稳定"，**非跨语言/跨编译器**。
- **HYLE 收紧为真实 direct FFI**：语言中立 schema 必须与 export manifest、calling convention、
  ownership 和 managed-runtime consumer 同时成立；不能把 header-inline helper 当导出。
- **协同点**：外语边界不暴露 C++ 类型，所以内部 C++23 的 STL/layout 风险被 ABI bridge 隔离。
  稳定性仍限定在声明支持的 target ABI/triple；C 语法不会神奇消除架构、位宽或 calling-convention 差异。

## 7. 决策速查表

| 维度 | 决策 | 关键依据 |
|---|---|---|
| C++ 标准 | C++23（GCC14+/Clang18+/MSVC19.43+）| std::expected 全家 2022–2023 即全支持 |
| 错误模型 | 零异常 + std::expected | 继承 clay 零异常，原生替代手写 Result |
| 命名/宏 | 继承 clay，前缀 HYLE_ | clay 最成熟资产 |
| 协程/调度 | co_await 驱动 fiber | C++20 协程三编译器稳 |
| executors | NVIDIA/stdexec | C++26 STL 未就绪 |
| modules | 暂缓 | 2026 不适合做构建基础 |
| 单测 | GoogleTest | 行业标准 + iron 已 vendored |
| 微基准 | nanobench | 贴硬件、单头 |
| 并发查错 | TSan/ASan/UBSan 分构建 + libFuzzer | 无锁代码必需 |
| 对外 ABI | shared library direct Native FFI；C 语法只作 schema | Java/.NET 无 shim 直绑 + 明确 layout/ownership |
| 内部实现 | 完整 C++23；另有正式 C++ source API | FFI bridge 不暴露 C++ 对象布局 |

## 8. 下一步

> ⚠️ **本节是项目开工时的起点记录(历史)**,非当前下一步。当前真实进度见 [`../milestones/README.md`](../milestones/README.md)——sched 已大幅推进(fcontext 切换/work-stealing/Waker/同步原语 done),simd 支柱基本完成,地基层立住。

动手前奠基已完成（scope/标准/范式/测试/边界/外部沟通定稿）。实现起点：深挖 `sched/`——它是 `io/`、`compute/` completion 机制的共同地基。第一块是 fiber 上下文切换选型（ucontext / boost.context / 自写汇编），结论会约束后续。
