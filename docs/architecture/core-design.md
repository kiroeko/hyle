# Core 架构：待裁决边界

> **状态：`DRAFT / OPEN`。** 本页只描述仓库中当前存在的实现，不是收口证书。v2 复审施工见
> [`core-rebuild`](../implementation/tasks/core/core-rebuild.md)，历史性能输入见
> [`benchmarks-core`](../implementation/evidence/performance/benchmarks-core.md)，当前 C++ 用法见
> [`user/core`](../user/core.md)。

## 定位

Core 是 HYLE 最底层的源码基础。它目前没有 HYLE runtime 上游，承载 error、platform、mem、
container、sched 和 SIMD 建立前就要可用的机制。这个位置要求 Core 尽量小、无状态、合同稳定；
但“位于最底层”本身不能证明当前放在 Core 的每项能力都归属正确。

公开实现目前包含四个能力族：

| 能力族 | 当前文件与形状 | 已观察到的消费者 | 尚未裁决的归属问题 |
|---|---|---|---|
| 产品身份 | `version_defs.h`、`version.h`、`version.cc`；编译期常量，加 linked `version()` 与借用的 `version_string()` | 根 CMake 与 Core package sample；普通产品模块没有调用 linked query | linked 产品身份应归 Core 还是 build/package；FFI generation 常量尤其像 build/package 职责 |
| 编译器与二进制形状合同 | `compiler_defs.h`、根级 `macros.h`、根级 `visibility.h`；attribute、optimizer promise、fail-fast 与 symbol visibility | 全库广泛使用，SIMD 和其他热 header 尤其多 | compiler hint 很可能属于地基；export/import policy 属 build/package；`HYLE_FATAL_ASSERT` 需明确属于 Core 还是 error/diagnostics |
| cache 合同 | `cache_defs.h`、`cache.h`；目标家族 layout 常量与三种 inline prefetch intent | container、mem、sched、SIMD/vqsort | 稳定 layout 常量可归 Core、runtime topology 归 platform；prefetch 是 Core primitive 还是 SIMD/消费者内部机制 |
| 侵入式稳定节点链表 | `intrusive_list.h`；header-only member hook、非 owning 双向链表 | time timer queue；sched 经 time 间接消费 | 应归 Core、util-core、container 还是 time-private；仅仅避免依赖环不能作为公开归属证据 |

这四项裁决完成前，模块边界都是暂定的。

## 当前依赖与交付形状

```text
compiler/cache/list headers ──► 编进每个 C++ consumer TU
version.h ────────────────────► hyle_core (version.cc)

cache 消费者：container、mem、sched、simd/vqsort
list 消费者： time ──► sched/io
macro 消费者：几乎全部 HYLE 模块
```

中央组件图把 `core` 声明为零依赖组件。多数下游虽不链接 `hyle_core`，却直接消费 Core 的
header-only 能力。重建时必须裁决组件图究竟只表达 link closure，还是也应表达 public source
dependency；当前文档不能默认后者已经得到证明。

需要分别验收三种 C++ 交付：

| 交付 | 当前行为 | 复审要求 |
|---|---|---|
| 整体源码 | `add_subdirectory(hyle)` 产生 `hyle::core`；inline 能力编进消费者，linked identity 来自 `version.cc` | parent options、compile features 与 component selection 必须可组合 |
| 静态 SDK | 消费者链接 `hyle::core`；`HYLE_STATIC` 去掉 import/export decoration | installed headers 与 archive 必须来自同一个一致 build，包含全部 inline 合同 |
| 动态 SDK | 消费者链接 import library/DSO 并 stage `hyle_core`；只有 linked identity 随 DSO 替换 | inline macro、常量或 list 改变时必须重编消费者；只替换 DSO 不够 |

**当前 Core 产品复审把 Native FFI 裁决为 `N/A`。** foreign runtime 没有合理任务需要为
compiler attribute、单地址 prefetch、layout 常量或 intrusive hook 支付动态调用。仓库里仍实际
存在预发布候选 `include/hyle/c/core.h`、`src/core/c_core.cc`、三个 C exports、测试与 FFI package
probe；它们是待处理的实现历史，不是已接受产品面。Core 签署前必须删除这些内容，或把真实的
全局 loader/bootstrap 任务迁给正确 owner；不能仅因符号已经存在就宣称 Java/.NET/Rust/Python
能力完整。

## 暂定合同

### 产品身份

当前 C++ header 提供 packed/text 产品版本常量；linked 函数返回编入 `hyle_core` 的版本，以及
指向模块静态存储的借用指针。动态模块卸载后该指针失效。这些都是当前实现事实，还不是稳定
ABI 承诺。

待裁决问题包括：runtime version query 是否在 package mismatch 诊断之外有真实价值；source
provenance 是否更应属于 release manifest；产品版本是否与 build/package compatibility 混为一谈。

### 编译器与二进制形状合同

当前 macro 覆盖 force/no-inline、must-use、aliasing、flatten/unroll、branch hint、empty-member
layout、copy suppression、fail-fast、assume 与 unreachable。它们没有 runtime 状态。有些只是可
安全降级的 hint；`HYLE_ASSUME`、`HYLE_RESTRICT` 和 Release `HYLE_UNREACHABLE()` 是 optimizer
合同，违约会进入 undefined behavior。`HYLE_FATAL_ASSERT` 终止进程，不是可恢复输入验证。

v2 复审必须拿当前 Highway、Abseil、Folly 和受支持 compiler 文档复核命名、Debug/sanitizer
退化及 codegen，还要把 compiler semantics 与 CMake 所有的 symbol visibility 分开。全库已经
大量使用只能证明迁移成本，不能证明模块归属正确。

### Cache 合同

当前常量是编译期 target-family recommendation。它们进入公开对象 layout，因此成为 consumer
ABI；它们不是 runtime observation，不能与 platform cache probe 混用。prefetch 函数是
best-effort inline hint，不建立地址有效性、cache residency 或同步关系，且允许退化为 no-op。

签署前，每个 layout 使用都要区分 false-sharing spacing、constructive packing 与无关的 SIMD/
对象 alignment；永久 padding 必须由 consumer footprint 与 contention 证据支撑。prefetch 必须
在拥有访问模式的正常 workload 中验证，不能靠单独计时一条 hint 签证书。

### 侵入式稳定节点链表

当前 list 不拥有节点，也不分配。一个节点按每条同时 membership 嵌一个 hook，linked 期间地址
必须稳定；并发操作要求外部同步。实现不存 owner/size；unchecked 操作依赖 membership 前置
条件，`try_push_*`/`try_insert` 只检查 null/already-linked。hook 和 list 析构都会摘链。

当前源码的算法界为：

| 操作 | 当前复杂度 |
|---|---:|
| front/back、push/pop、insert/erase/remove | O(1) |
| 单节点或整链 splice、append、swap、list move | O(1) |
| iteration、clear、`consume_all`、`for_each` | O(n) |
| `move_prefix_to` | O(k) 扫描 + O(1) 重链 |

这些是代码性质，不是性能证书。归属复审必须以真实 time/scheduler 任务对比 Boost.Intrusive 与
Folly 同类形状，判断是否真的需要公开通用 list。若保留公开面，还要在受支持工具链上，以同一
candidate 验证 member-pointer/container-of 可移植性、iterator 失效、callback mutation、
self-splice、异常合同与析构顺序。

## 暂定负空间

下列能力当前不属于 Core，但在 open review 结束前仍只是边界假设：

- runtime CPU/cache/NUMA/OS discovery 归 platform；
- owning container、同步、分配、clock、logging 与可恢复错误归各自模块；
- Core 不承诺 cache residency、portable exact cache-line size，或对无效地址安全 prefetch；
- Core 当前不提供 managed-language container 或 compiler-hint ABI；
- intrusive list 不提供 ownership、O(1) size、随机访问、owner check 或内部同步。

真实消费者、成熟 peer 的更好算法/形状，或依赖环审计都可能在 `core-rebuild` 中改变这些裁决；
因此它们还不是签署后的 non-goal。

## 签署条件

Core 保持 `OPEN`，直到下列内容绑定同一 candidate commit/tree：

- 外部场景、语法消费者与语义重复账本；
- 四个能力族的成熟库对账与明确归属；
- C++ source/static/shared 的默认、高级与组合旅程；
- Native FFI 的最终 `N/A` 或真实任务裁决，以及相应删除/迁移或完整 direct-binding 证据；
- Windows、Linux、macOS 的 correctness/toolchain/package 证据；
- complexity 与合格正常 workload 性能证据；旧 Core benchmark 在重签前只作 historical input；
- 同一 candidate 的文档、依赖图、测试、sample、LOC、commit 与 push。
