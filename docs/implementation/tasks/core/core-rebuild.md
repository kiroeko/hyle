# Core v2 重建与收口记录

> **状态：`OPEN` / 黄色。** 本任务取代已经删除的历史
> `core-module-closure.md`，不继承该文档的绿色状态。建立本轮初始账本时检查的产品基线为
> `87c831d228e4e2f7844e60a3e9fd167cef4992b9`；目前没有冻结 v2 candidate commit/tree。

## 目标

从真实 C++ 消费者重新建立 Core 产品边界，再按当前模块收口方法审查功能、形状、复杂度、易用
性、依赖、三种交付和合格性能。现有代码与历史证据只能作为输入；全部门绑定同一 candidate 前，
Core 必须保持 open。

最终稳定设计归 [`core-design`](../../../architecture/core-design.md)，当前暂定 C++ 用法归
[`user/core`](../../../user/core.md)，性能证据归
[`benchmarks-core`](../../evidence/performance/benchmarks-core.md)。

## 审查单元

目录名不是产品边界。本轮包括：

- `include/hyle/core/{version_defs,version,cache_defs,cache,compiler_defs,intrusive_list}.h`；
- 逻辑上由 Core 持有的根级 `include/hyle/{macros,visibility}.h`；
- `src/core/{version,c_core}.cc` 与 `hyle_core` target；
- Core tests、package samples、managed samples、export manifest 与 benchmark；
- Core source/static/shared 消费需要的 component graph 与 package 行为；
- time、sched、container、mem、SIMD 与 build system 中的直接调用者和语义重复实现。

Build/package 是相邻模块，不是 Core 的实现细节。symbol visibility、export ownership、package
component、runtime staging 或源码组装发生变化时，必须共同回审 build/package。

## 开工时实现账本

仓库当前有四个能力族。下表只登记事实与问题，不批准现有位置：

| 能力族 | 当前实现 | 当前直接或语义消费者 | 必须作出的裁决 |
|---|---|---|---|
| 产品身份 | 编译期 product/FFI 常量与 linked C++ 产品版本查询 | 根 CMake、package probe；普通 HYLE 模块不调用 linked query | Core 与 build/package 的归属；linked identity 是否是真实产品能力 |
| compiler/binary 合同 | compiler detail macros、根级 public macros、symbol visibility、fail-fast | 全库广泛使用；SIMD 与热 container 是主要 codegen 消费者 | 分开 compiler semantics 与 build-owned visibility；裁决 fatal assertion 及 public/internal macro |
| cache 合同 | target-family layout 常量与 inline local/NTA/write prefetch | container、mem、sched、SIMD/vqsort | Core/platform/consumer-private 归属；按语义分类全部 64/128 alignment |
| intrusive list | header-only、非 owning member-hook list，含 iterator 与 batch 操作 | time timer entry/queue；sched/io 间接消费 | Core、util-core、container 或 time-private；不能只因规避依赖环就保留公开通用类型 |

### 初始语法消费者账本

- `time/timer_entry.h` 嵌入 `IntrusiveListHook`，并定义 timer queue 使用的 list；
- container public headers 用 Core destructive-interference 常量布局 queue state、map shard、vector
  与 work-stealing structures；compiled container 也用于 raw shard 和 string-buffer shard；
- `mem/shared_pool.h` 在 public header-defined layout 中使用 destructive-interference 常量；
- scheduler compiled code 使用 maximum destructive-interference 常量；
- SwissTable 与 SIMD/vqsort 调用 `prefetch_read`；
- 根 CMake 读取 `version_defs.h`；installed C++ sample 调用 linked `version()` 并覆盖其他能力族；
- compiler/fatal/visibility macro 遍布全库，必须生成完整账本，不能靠人工挑几处代表。

### 初始语义重复问题

- raw compiler builtin/attribute 与 ported vendor macro 都要分类，包括没有 include `hyle/macros.h`
  的复制实现；
- `alignas(64/128)` 要分成 false-sharing layout、SIMD/object alignment、lookup-table alignment 与
  偶然 stack layout；
- platform runtime cache observation 不能误算成 Core 稳定 layout 常量的重复；
- 当前没有明显第二套 intrusive doubly-list，但仍要比较 timer/scheduler 私有链接和 container
  structure 的真实职责；
- package/version manifest 可能已经能解决 linked-version 任务，无需 Core runtime API。

两份账本目前都只是初稿；完成归属迁移后还要重新扫描。

## 产品面与交付裁决

本轮产品是三种交付形状下的 C++：

| 形状 | 必须跑通的旅程 | 当前状态 |
|---|---|---|
| 整体源码 | 配置 Core-only source subtree，链接 `hyle::core`，编译运行默认/高级/组合 C++ 用户 | 尚未按 v2 重签 |
| 静态 SDK | relocate installed static package，只消费其 exported target | 尚未按 v2 重签 |
| 动态 SDK | relocate installed shared package，stage runtime dependency，同时消费 inline 与 linked Core 行为 | 尚未按 v2 重签 |

Native FFI 目前明确为 **Core 产品要求的 `N/A`**：尚未发现 foreign runtime 对 compiler hint、cache
常量、单地址 prefetch 或 native intrusive hook 的真实任务。只有 product identity 并不足以制造
一套 Core FFI surface。

这个裁决暴露的是清理任务，而不是给当前树签证书。树中仍有 `include/hyle/c/core.h`、
`src/core/c_core.cc`、三个 exports、C/export tests、installed C probe 与 Java/.NET Core smoke。
candidate 冻结前必须删除它们，或把它们迁给已经证明真实需求的全局 loader/bootstrap owner。
`HYLE_FFI_ABI_VERSION` 还被其他模块 schema 消费，因此要一致迁移 ownership，不能只在 Core
局部删除。完成前，这些符号只是 legacy pre-release candidate，不是受支持 API。

## 功能与架构审计计划

### Gate A — 场景与归属

- [ ] 为四个能力族分别写外部/内部场景，包含 ownership、lifetime、concurrency、failure、正常
      scale 与 platform；
- [ ] 完成语法消费者和语义重复账本；
- [ ] 裁决每个能力族和每个根级文件的 owning module；
- [ ] 写清负空间与 reopen trigger；
- [ ] 更新中央组件图，或明确证明为何 header/source dependency 应与 link dependency 不同。

### Gate B — 成熟库对账

- [ ] 用 oneTBB 及当前 package/loader practice 对比 version/build identity，而不只比较版本号格式；
- [ ] 用当前 Highway、Abseil、Folly 与受支持 compiler 合同对比 compiler、assume、visibility、
      prefetch 形状；
- [ ] 用当前 Boost.Intrusive 与 Folly 消费者对比 intrusive ownership、hook policy、iterator、
      splice、teardown 与复杂度；
- [ ] 先调查更好的算法、metadata trade-off 或 toolchain strategy，再做纳秒优化。

### Gate C — API 与正确性

- [ ] 分别审查 C++ 名称、include、默认路径、高级路径与组合路径；删除 benchmark-shaped 或只供
      维护者使用的 public entry；
- [ ] 从合同重新推导 intrusive-list lifetime、member-pointer portability、callback、iterator、
      self-operation、exception 与 concurrency 测试；
- [ ] 在 MSVC、GCC、AppleClang 上重建 compiler/cache codegen 与 sanitizer/Debug 退化测试；
- [ ] 在 source、tests、samples、exports、build/package 与文档中一致解决 legacy Native FFI。

### Gate D — 交付与集成

- [ ] fresh Windows、Linux、macOS Core-only correctness/header build；
- [ ] relocated C++ source/static/shared consumers；若保留 linked identity，还要覆盖 shared-runtime
      mismatch 与 borrowed lifetime；
- [ ] 只回审被触碰的 time、container、mem、sched、SIMD 依赖边，不无故重开无关模块；
- [ ] 验证 component selection、installed target closure、visibility、runtime staging、offline/source
      bundle 与 license/package metadata。

## 复杂度与性能门

当前 intrusive-list 源码看起来提供 O(1) single-node/whole-list relink、O(n) traversal/clear 与
O(k) prefix scan；需要以合同和强 peer 复核。version query 及 compiler/cache helper 都是 O(1)
机制，但不能为了表格完整制造单指令/getter SOTA benchmark。

- [ ] 冻结 candidate 与 claim-bearing source/harness identity；
- [ ] 只用同合同的正常 intrusive workload 和真实 time/scheduler consumer；把语义成本归因分开；
- [ ] 以 codegen 加 owning consumer workload 验证 compiler/prefetch，不单独计时 hint；
- [ ] 覆盖代表规模、default/peer-first、必要 placement/code-shape perturbation、三平台、逐进程 raw
      evidence、exact 五态与独立 non-inferiority gate；
- [ ] 调查每个 `LOSE`、`UNRESOLVED` 或 shape-sensitive 结果；不得为赢 benchmark 改 public
      metadata 或 API。

上述门完成前，所有旧 Core benchmark 都是 `HISTORICAL/UNQUALIFIED`。

## 文档与签署清单

- [x] 用本 open rebuild 记录取代继承绿色的旧 closure 文档；
- [x] architecture 与 C++ user guide 标为 provisional；
- [x] 旧性能 claim 降级为 historical/unqualified；
- [ ] 归属裁决后，修复相邻 build/FFI methodology 与 samples 中过时 Core FFI claim；这些文件不在
      本文档 patch 获准的范围内；
- [ ] 把 Windows/Linux/macOS evidence、package consumer、performance 与 docs 绑定同一 candidate；
- [ ] 报告四项 LOC 并解释边界变化；
- [ ] 红队复核最终 API/dependency/performance 证据；
- [ ] commit、push，并记录 required CI 或明确 waiver。

只有四项归属裁决与两份消费者账本都完成，任务才可从 `OPEN` 进入 `AUDITED`；只有剩余 checkbox
全部完成，才可进入 `SIGNED`。历史绿色文字不能作为状态迁移证据。
