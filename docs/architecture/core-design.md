# Core 架构：待裁决边界

> **状态：`DRAFT / OPEN`。** 本页只描述仓库中当前存在的实现，不是收口证书。v2 复审施工见
> [`core-rebuild`](../implementation/tasks/core/core-rebuild.md)，历史性能输入见
> [`benchmarks-core`](../implementation/evidence/performance/benchmarks-core.md)，当前 C++ 用法见
> [`user/core`](../user/core.md)。

## 定位

Core 是 HYLE 最底层的源码基础。它目前没有 HYLE runtime 上游，承载 error、platform、mem、
container、sched 和 SIMD 建立前就要可用的机制。这个位置要求 Core 尽量小、无状态、合同稳定；
但“位于最底层”本身不能证明当前放在 Core 的每项能力都归属正确。

公开实现目前包含三个能力族：

| 能力族 | 当前文件与形状 | 已观察到的消费者 | 尚未裁决的归属问题 |
|---|---|---|---|
| 产品身份 | `version_defs.h`、`version.h`、`version.cc`；编译期常量，加 linked `version()` 与借用的 `version_string()` | 根 CMake 与 Core package sample；普通产品模块没有调用 linked query | linked 产品身份应归 Core 还是 build/package；FFI generation 常量尤其像 build/package 职责 |
| 编译器与二进制形状合同 | `compiler_defs.h`、根级 `macros.h`、根级 `visibility.h`；attribute、optimizer promise、fail-fast 与 symbol visibility | 全库广泛使用，SIMD 和其他热 header 尤其多 | compiler hint 很可能属于地基；export/import policy 属 build/package；`HYLE_FATAL_ASSERT` 需明确属于 Core 还是 error/diagnostics |
| cache 合同 | `cache_defs.h`、`cache.h`；目标家族 layout 常量与三种 inline prefetch intent | container、mem、sched、SIMD/vqsort | 稳定 layout 常量可归 Core、runtime topology 归 platform；prefetch 是 Core primitive 还是 SIMD/消费者内部机制 |

这三项裁决完成前，Core 模块边界仍是暂定的。

侵入式稳定节点链表的归属已经裁决：公开头迁到
`<hyle/container/intrusive_list.h>`，类型位于 `hyle::container`，并由独立 header-only target
`hyle::container_intrusive` 承载。`time` 公开依赖这个轻量 target，完整 `hyle::container` 公开
聚合它；二者都不要求 `time` 依赖完整 container。项目尚未发布，因此不保留
`hyle::core::IntrusiveList` 别名或旧路径转发头，避免形成两个 owner。

## 当前依赖与交付形状

```text
compiler/cache headers ───────► 编进每个 C++ consumer TU
version.h ────────────────────► hyle_core (version.cc)

cache 消费者：container、mem、sched、simd/vqsort
macro 消费者：几乎全部 HYLE 模块

container/intrusive_list.h ──► hyle::container_intrusive ──► time
                                                └──────────► container
```

中央组件图把 `core` 声明为零依赖组件。多数下游虽不链接 `hyle_core`，却直接消费
`hyle/macros.h`、compiler definitions 或 cache headers。当前构建合同把根级宏及其 compiler
definition 支撑头视为 target-neutral foundation headers；包含它们不形成 `hyle::core` 链接边。
这是待 Core/build 全局复审的明确边界，不是允许普通模块隐藏任意 header dependency 的通例。
cache headers 当前同样被多个未链接 `hyle::core` 的模块包含，但它们是否也属于 target-neutral
合同仍未裁决，不能从 macro 例外自动推出。
相比之下，侵入式链表是独立公开数据结构，已经用真实 `container_intrusive` target 和
`PUBLIC` 边建模。

需要分别验收三种 C++ 交付：

| 交付 | 当前行为 | 复审要求 |
|---|---|---|
| 整体源码 | `add_subdirectory(hyle)` 产生 `hyle::core`；inline 能力编进消费者，linked identity 来自 `version.cc` | parent options、compile features 与 component selection 必须可组合 |
| 静态 SDK | 消费者链接 `hyle::core`；`HYLE_STATIC` 去掉 import/export decoration | installed headers 与 archive 必须来自同一个一致 build，包含全部 inline 合同 |
| 动态 SDK | 消费者链接 import library/DSO 并 stage `hyle_core`；只有 linked identity 随 DSO 替换 | inline macro 或常量改变时必须重编消费者；只替换 DSO 不够 |

**当前 Core 产品复审把 Native FFI 裁决为 `N/A`。** foreign runtime 没有合理任务需要为
compiler attribute、单地址 prefetch 或 layout 常量支付动态调用。仓库里仍实际
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

## 暂定负空间

下列能力当前不属于 Core，但在 open review 结束前仍只是边界假设：

- runtime CPU/cache/NUMA/OS discovery 归 platform；
- owning container、侵入式容器、同步、分配、clock、logging 与可恢复错误归各自模块；
- Core 不承诺 cache residency、portable exact cache-line size，或对无效地址安全 prefetch；
- Core 当前不提供 managed-language container 或 compiler-hint ABI；

真实消费者、成熟 peer 的更好算法/形状，或依赖环审计都可能在 `core-rebuild` 中改变这些裁决；
因此它们还不是签署后的 non-goal。

## 签署条件

Core 保持 `OPEN`，直到下列内容绑定同一 candidate commit/tree：

- 外部场景、语法消费者与语义重复账本；
- 三个能力族的成熟库对账与明确归属；
- C++ source/static/shared 的默认、高级与组合旅程；
- Native FFI 的最终 `N/A` 或真实任务裁决，以及相应删除/迁移或完整 direct-binding 证据；
- Windows、Linux、macOS 的 correctness/toolchain/package 证据；
- complexity 与合格正常 workload 性能证据；旧 Core benchmark 在重签前只作 historical input；
- 同一 candidate 的文档、依赖图、测试、sample、LOC、commit 与 push。
