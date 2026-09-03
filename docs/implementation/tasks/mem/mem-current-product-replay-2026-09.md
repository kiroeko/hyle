# Mem 当前全产品能力回放 — 2026-09

<!-- READ-ONLY CAPABILITY REPLAY: PRODUCT_TRIGGER -->

> 本页是 product-non-mutating replay，不是第二套模块状态。Mem 在发现满足收口协议的具体
> `PRODUCT_TRIGGER` 前仍以 [`module-status.json`](../../module-status.json) 中的 `CLOSED` 为准；本页
> 不授权修改公开 API、算法、布局、依赖或正常 workload。

## 回放合同

- 对象：普通用户看到的 `hyle::mem` 产品，以及内部 `mem`、`mem_vm`、`mem_reclaim` 三个 factor 的
  累积合同；
- reviewed rounds：`mem-v2-rebuild`、`mem-v3-vm-factorization`、
  `mem-reclaim-v2-inference-resume`、`mem-v4-reclamation-facade`；
- reviewed current registry identity：Mem v4 candidate
  `0f6ef8e83ee290ab565c0d53081a0acb75a46675`，end
  `5a51b909031a3efd18288f84aa573549f5a4acaf`，anchor
  `b9fdec1a404aa05f0e63cd214896b05c09c59444`；
- replay baseline：HYLE `911e4ce61adc924e2261b622ca98461b2734a4d5`；
- reviewer / date：Codex / 2026-09-03；
- owner request：停止过早启动的 IO 审计，从 Mem 重新验证当前累积产品，而不是把历史窄
  `CLOSED` round 自动当成对全部 Mem 能力的今日回答；
- 总预算：只读能力/API/依赖/证据审计最多 4 engineering hours；性能计时预算为 0。若发现产品触发器，
  后续新 closure card 单独冻结实现与 P1 预算；
- verdict：`PRODUCT_TRIGGER`。本页只证明并定界 trigger，不授权产品修改；Mem 在 registry 完成合法
  `CLOSED -> REOPENED -> SCOPED -> AUDITED` 以前仍保持原 `CLOSED` 状态。

## 锁定输入

| Source | Clean identity at replay start | Role |
|---|---|---|
| llama.cpp | `c7bda030e7faee594dbe7550185e857351ad405d`；随仓 ggml tree `25e2241ed891c98cd26af5c47246c79b430a49cd` | 当前模型加载、buffer、KV/推理生命周期真实消费形状 |
| standalone ggml | `d4716378882593333721eb33f153144b6885caf2` | 当前基础 buffer/resource/allocation planning 能力 |
| mimalloc | HYLE 产品锁 `v3.5.0` / `18b08671c9302247bfb682286e6bf3cc1773f801`；上游 clean `c683b7c60c91cb2809977e37d3f4d084d7497180`，含 `v3.5.1` | host heap mechanism、版本影响与交付 |
| oneTBB | `843162db8cf4b2531a4a6147b6ea3fd860d44668` | scalable allocator、memory pool 与并发生命周期 peer |
| Folly | `80071a1f79e110b34a2cf7a3a4d01a59e823d697` | arena、mapping、reclamation peer |
| CCCL | `b7aaea69a2b07e50f09e67f2962da0243e0b7c5d` | property-bearing host/device/pinned/managed resource 与 buffer peer |

每个 capability family 最多使用上述两个 strong peers；llama.cpp/ggml 是强制 consumer/foundation
差异输入，不因“peer 数”被排除。Boost 工作树有未提交修改，本轮不把它登记为干净 peer identity。

## 必须回答的范围

1. **功能超集：**从最新 llama.cpp 真实 journey、standalone/随仓 ggml、定义性 C++/OS 能力和成熟 peer
   建立 family map；逐族审取得/构造、正常操作、组合/互操作、释放/trim/回收、边界、ownership、
   concurrency、failure、复杂度与交付。不得按 API 数量求并集。
2. **自然产品形状：**反向追踪全部公开 Mem 入口，识别 benchmark-only、半生命周期、错误 owner、
   命名/默认值/失败形状不自然和需要用户跨内部 factor 拼装的路线。
3. **质因数与依赖：**检查 `mem/mem_vm/mem_reclaim` 拆分是否足够小且无环，是否正确复用
   Core/Error/Platform/Platform-fs/Util-core，直接消费者是否绕过 Mem 私造 allocator、mapping、buffer、
   reclamation 或 placement 机制。
4. **第三方：**区分 HYLE semantic owner 与 mimalloc/OS/peer mechanism；检查 v3.5.1 相对产品锁是否
   改变实际源码、语义、算法、codegen、许可证或交付，不能因“有新版”自动升级或继承旧证据。
5. **性能：**先审复杂度、allocation/copy/syscall/atomic/lock、常态 metadata 和旧 comparison class。
   本 replay 不运行 benchmark；若有触发器，新轮先冻结 API/usability，再定义一个拟真 primary 与至多
   一个 holdout，同合同 P1 和 natural-route 分开。
6. **交付与文档：**核对 source、installed static/shared、selected roots、外部依赖锁/离线/许可证，清理
   旧 C ABI/Native FFI 叙事；architecture 解释边界，user guide 按场景告诉用户选什么，performance 只
   保留候选绑定的有限结论。

## 初始 capability-family 审计队列

- host system/aligned allocation、reallocation、STL/PMR adapter 与独立 growable heap；
- monotonic/stack/request scratch、fixed/variable/typed/concurrent pool、quota/stats/trim；
- virtual/file/shared mapping、reserve/commit/protect/advice/residency/huge/NUMA/guarded stack；
- resource properties、Buffer owner/view/import/composite、host/pinned/device/managed provider seam；
- sync/async copy/fill/map/flush/invalidate 的 owner 边界；
- transient pooling、capacity query、interval/lifetime packing 与 worst-case reserve；
- read-mostly publication/reclamation 及其与 Container 的直接 consumer；
- OOM、overflow、alignment、zero-size、move/destroy、跨线程 free、callback/quiescence 和 provider unload；
- 三种 C++ 交付、公开头闭包、第三方类型泄漏、旧 FFI 残留和用户选型可发现性。

## 停止与下一状态

- 全部当前 `MUST` 已自然 `COMPLETE/ROUTED`，产品 identity 与适用证据未被击穿：`CONFIRMED`；
- 没有产品缺陷，但旧任务/仪器不足以回答当前问题：`METHOD_GAP`，Mem 仍 `CLOSED`；
- 存在可复现缺失 journey、错误 owner/生命周期/复杂度/交付，或受影响第三方升级事实：
  `PRODUCT_TRIGGER`，记录 trigger ID 与最小影响面，然后才建立 `REOPENED -> SCOPED -> AUDITED` 新轮。

当前只允许完成 public-surface trace、consumer/peer 差异、依赖/交付扫描和 capability ledger；不得先写
产品代码，也不得因为旧 benchmark 某行输赢改变 API。

## 回放结果

### 当前基线复现

在未修改产品的 `28cd723bef71bbd1378940d39f31fde4c527761f` 工作树上，用 W1-WSL、GCC/G++ 14、
Release、`HYLE_COMPONENTS=mem`、tests on、bench off 建立仓外 fresh build；`mem|mem_vm|mem_reclaim`
标签 6/6 通过，总用时 39.19 s，其中 `hyle_mem_test` 32.15 s。这个 receipt 证明当前 Linux
correctness/header smoke 可复现，不替代三交付、sanitizer、macOS、Windows 或性能门。

当前产品 façade 共安装 28 个 `include/hyle/mem/*.h` 公开头；内部为 `mem`、`mem_vm`、
`mem_reclaim` 三个无环 factor。当前工作树统计为：全库代码 437,250 行、自实现含测试 111,204 行、
自实现不含测试 70,416 行、Mem 三 factor 实现 7,403 行（37 个实现文件）。这些数值只描述工作树规模，
不是 terminal LOC receipt。

### Capability-family coverage map

| Family | admitted journey / evidence | current route | coverage | missing edge / negative space |
|---|---|---|---|---|
| host system allocation、owned heap、STL/PMR interop | 普通 infra allocation；mimalloc、oneTBB | `SystemAllocator/AllocatorRef/Heap` 与 std/PMR adapters | `PARTIAL` | API/生命周期完整，但 shipping mechanism 仍硬锁 mimalloc v3.5.0；v3.5.1 含 correctness/codegen/平台实质修复，旧 dependency qualification 不能继承 |
| phase/LIFO scratch 与 fixed/variable/typed pool | request/frame scratch、remote free、hard/soft/growable pool；Folly/oneTBB/marl | Arena、StackArena、FixedHeap、ConcurrentBlockPool/cache、object pool、标准 PMR pool route | `COMPLETE` | blocking borrow 属 Sched；隐藏 TLS pool 无具名 lifecycle owner，拒绝下沉 |
| VM/file/shared/huge/NUMA/guarded stack | llama model mmap/mlock、显式 locality 与 fiber stack | `VirtualMapping/FileMapping/SharedMapping/HugeMapping/FiberStack`；OS mechanism 走 Platform | `COMPLETE` | FileMapping 故意不 partial-unmap；RW/COW 以 checked `writable_bytes()` 暴露修改，没有证据要求再复制一个 mutable BufferView 入口 |
| property-bearing resource 与 owned/borrowed/composite bytes | ggml buffer/resource、CCCL；host/device/pinned/managed provider seam | `MemoryResource/Ref`、`Buffer/View/HostMapping/CompositeBuffer` | `COMPLETE` | backend 从 host pointer 注册、buffer usage/reset 是 Compute/Graph typed policy；Mem 保留 provider-neutral storage vocabulary，不把模型/backend hint 塞进每个 Buffer |
| sync range data/coherence | CPU/provider read/write/fill/copy、non-coherent mapping | Buffer view inline range operations + provider dispatch | `COMPLETE` | async ordering、staging 和 completion 由 Compute/IO 组合，不在 Mem 伪造通用 event |
| transient backing reuse | inference temporary buffer acquire/return/trim | `BufferPool/BufferLease` | `COMPLETE` | completion 前归还仍是调用方/Compute lifecycle 合同 |
| whole-graph lifetime packing 与实例化 | ggml gallocr；已知 interval/alias/resource facts | `LifetimePlanner/Plan/PlannedBuffers` | `PARTIAL` | planner 把每个 resource 压成一个线性 peak 并只创建一个 Buffer，无法满足 provider `maximum_allocation` 下的 multi-chunk graph allocation |
| read-mostly publication reclamation | Container concurrent-node snapshot consumer；Folly peer | `ReclamationDomain/Guard/RetireRecord` | `COMPLETE` | stalled-reader-independent hazard pointer 是不同进展/metadata 合同；当前无具名 consumer，保持 GENERAL backlog |
| quota/pressure/stats/leak diagnostics | infra OOM、retention、debug | QuotaAllocator、resource/pool stats、snapshot、LeakTracker/StatsAllocator | `COMPLETE` | opt-in accounting 不进入默认 allocator 热路 |
| source/static/shared、selected roots、FFI 清洁 | C++ embedding/package consumer | Mem façade 聚合三 factor，mimalloc 私有静态 artifact | `PARTIAL` | 当前三种 C++ 交付和 Native FFI=N/A 边界正确；但 dependency 与 planner 产品变化后必须重签受影响交付，而不能沿用旧 candidate |

### Product triggers

#### `MEM-LIFETIME-MULTI-CHUNK-01`

- HYLE `MemoryResource::properties()` 已公开 `maximum_allocation`，`Buffer::create()` 对超过上限的单笔
  allocation 正确返回 `OutOfRange`；
- `LifetimeResourcePlan` 只有一个 `peak_bytes/alignment`，`MemoryPlacement` 没有 physical allocation ID，
  `LifetimePlan::instantiate()` 对每个 resource 恰好调用一次 `Buffer::create(peak_bytes)`；
- 因此，只要同一 resource 的合法 graph peak 超过其单 allocation 上限，planner 就无法实例化，即使每个
  root request 都能放入多个合法 backing。用户手工构造 `CompositeBuffer` 不能补全这条路线，因为它要求
  consumer 自己重做 placement/chunk ownership；
- 锁定 standalone ggml `d4716378882593333721eb33f153144b6885caf2` 的 `ggml_dyn_tallocr` 读取
  backend max size，把 placement 表示为 `{chunk, offset}`，再按 chunk 建 buffer。该机制来自
  `254ea3e588432e6b2814b82f324fe6bb25d78a25`（2025-09-24），早已包含在 Mem v2 当时锁定的同一
  ggml identity 中；这是旧审计漏项，
  不是本次 peer 漂移或 benchmark 驱动的新需求；
- 最小影响面：lifetime-plan API/实现/测试、guide/architecture、Mem 三交付；不修改 Tensor、Graph、
  Compute backend 或普通 Buffer layout。目标仍为 O((N+E) log N) build、O(N) metadata，request view
  O(1)，不建立按 byte/page 的表。

#### `MEM-MIMALLOC-351-01`

- shipping Mem 唯一直接第三方 product mechanism 被 lock、vendored tree、源码 `static_assert` 和测试同时
  固定为 mimalloc v3.5.0 / `18b08671c9302247bfb682286e6bf3cc1773f801`；
- 当前 stable v3.5.1 / `34fbd7e7cd4627424490afe19b20f8066bfc537d` 改变了实际源码与 hot-path
  codegen，并包含 arena natural-alignment、aarch64/RISC-V/clang-cl build、double-free checking、large
  zalloc branch 与 free codegen 修复。它不是只改下载元数据，按 dependency policy 必须定向重开 host
  allocator route；
- HYLE 有意私有编译 `src/static.c`，不直接采用 upstream CMake。升级必须显式裁决 v3.5.1 的 Linux
  `MI_ALLOW_THP=FULL` 和 ARM `MI_OPT_ARCH` 新默认，不能盲继承会改变 HYLE baseline/retention 的选项；
- 小于 `alignof(max_align_t)` 的 HYLE 请求继续 padding 到 16B：mimalloc 的 natural alignment 只在 size
  已达到相应边界时成立，v3.5.1 的 arena 修复没有取消 HYLE 对弱显式 alignment 仍返回
  `max_align_t` 对齐的公开合同。

### 差异处置与 public-surface trace

- 28 个公开头全部能映射到上表 family；未发现 shipping benchmark-only 类型、`fast/light` 命名档或旧
  C API。`HYLE_TESTING` 下的 fit/membership diagnostics 不进入产品 layout。
- llama.cpp 的 mmap range -> device `buffer_from_host_ptr`、WEIGHTS/COMPUTE usage，以及 ggml buffer reset
  都需要 backend/tensor/execution facts，登记到 Compute/Graph，而不是给所有 provider-neutral Mem Buffer
  增加常态字段。`FileMapping::view()` 已自然提供 mmap 的 const byte view；未来 typed Compute import
  adapter 应消费该 view 并拥有注册/解除注册生命周期。
- Folly 当前 hazptr 改动用于 stalled-reader-independent reclamation 和 prefetch；它不击穿当前 epoch/RCU
  family 的已声明进展合同。oneTBB 相对旧 identity 没有形成新的 Mem public capability trigger。
- `FileMapping` 对 RW/COW 只提供 checked mutable span 而不提供 mutable `BufferView`，当前 normal journey
  完整；若未来 provider-generic writable mapped output 确实需要它，以具名 consumer 触发，不在本轮为 API
  对称性扩面。

### 六问直答

1. **第三方：未通过。**直接 product dependency 只有私有静态编入的 mimalloc；semantic owner 为 Mem，
   mechanism/artifact owner 为 mimalloc，version/交付 owner 为 Build。v3.5.1 是 material upgrade trigger。
   OS API、C++ 标准 allocator/PMR 与 compiler builtin 不是第三方产品模块。
2. **功能：未通过。**除 lifetime multi-chunk 外，当前准入 Mem family 为 `COMPLETE` 或具名
   `ROUTED/OUT_OF_SCOPE`；`MEM-LIFETIME-MULTI-CHUNK-01` 是未处置 `MUST`。不能再以“手工 CompositeBuffer”
   代替 planner 的正常 graph journey。
3. **性能：有限通过，整模块不作 SOTA 签名。**当前 v2 证据只签 Heap parity、FixedArena win、checked
   BufferView parity、BufferPool win 四个冻结 workload；旧 v1 多行经当前 comparison class 重放为
   `NATURAL_ROUTE/MECHANISM_DIAGNOSTIC`，不能冒充 same-contract。新轮对 multi-chunk 只做 P0
   complexity/resource gate，对 mimalloc 3.5.1 只重放受影响的 normal host-allocation P1；不重跑整个 Mem。
4. **旧 C ABI / Native FFI：通过。**公开树无 C schema/header/bridge/专用 option/managed sample/未装饰 HYLE
   C export；`system_inline.h` 的 `extern "C"` 只声明私有 mimalloc mechanism。shared library 仍只提供 C++
   decorated exports，Native FFI 为 `N/A`。
5. **构建与导入：当前旧 candidate 有证据，本轮尚不能最终通过。**factor 图与 selected header 闭包无环，
   replay 的 WSL source build/test 通过；但 dependency 与 planner candidate 改变后必须重新运行 source、
   installed static/shared consumer，不能继承旧 artifact receipt。
6. **C++ API / 文档：未通过。**大多数 ownership/concurrency/失败形状和按场景 guide 自然；planner API
   无处输入/查询 physical-allocation constraint/id，guide 仍把一次 backing 写成完整路线，并把超上限问题
   推给手工 composite。mimalloc 版本说明也停在 v3.5.0，须随新 candidate 校正。

## 终止裁决

本回放在两个可复现且有限的产品 trigger 上停止，不再继续扫描可选 peer feature。下一轮只能覆盖：

1. max-aware multi-chunk lifetime plan/instantiate 的自然 C++ 形状与 P0 复杂度；
2. mimalloc v3.5.1 vendored/lock/custom-build 升级及受影响 host-allocation correctness、P1 和交付；
3. 由前两项直接改变的 architecture/user/performance/task 事实、三交付与红队。

mutable FileMapping view、hazard pointer、backend buffer usage/reset、host-pointer device registration、async
completion 和 Tensor/Graph policy 不进入本轮 `FIX` 集。先以本页两个 trigger 合法改 registry 为
`REOPENED`；随后新建 closure card 并逐级进入 `SCOPED/AUDITED`，不得在本 replay 页直接施工。
