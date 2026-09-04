# Container-intrusive 冻结确认 — 2026-09-04

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` 产品的只读确认，不是一次无触发重开，不建立新的产品 candidate，也不替换
> [`container-intrusive-v4-capability-extension`](container-intrusive-v4-capability-extension.md) terminal
> record。当前状态真相仍是 [`module-status.json`](../../module-status.json)；其中的
> `freeze_confirmation` 只把本次确认绑定到明确的仓库快照和本文证据。

## 绑定身份与范围

- confirmation：`container-intrusive-freeze-2026-09-04`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `5f39c898b6e6680ec0ea8208ad27f125ee51552c` /
  `4982a0a87bdabd7ef6bdcaa99db3542d5d343b08`；
- frozen product candidate / tree：
  `58431ddbef6f54f9d4275b0fea2d317cba00b9f3` /
  `8df2b7d7806918d03b5bb5c31611adb196f30c7f`；
- candidate 到 reviewed commit 的四个 public implementation header、模块 tests、benchmark 与专门
  package sample：zero diff；
- product modifications in this confirmation：none。

冻结后的中央 factor/product graph 与构建合同发生过重排，因此本次重新执行当前精确 factor、当前
Container 聚合产品的 source/static/shared/relocated journeys 和全图组件合同，不从旧 receipt 推断交付。
两个冻结 workload consumer 的 provenance 也逐项复核：

- `samples/package/container/cpp/main.cc` 当前 SHA-256 为
  `47522da4f537d07f75dc82cfba42050f946126fcab1c4b84e4afcb00c1a0b1cd`；相对 candidate 只在已经完成的
  intrusive journey 后增加 `ConcurrentNodeHashMap` 检查，三种 topology 的操作、检查与计费未变；
- `src/sched/cancel.cc` 当前 SHA-256 为
  `634465e979b8978dc2c54a5d8b2c2e93ec302d2652f406d0aaf88aa98f559b8e`；相对 candidate 只把
  `mem::Allocator` handle 迁移为 `mem::AllocatorRef`，safe-hlist record layout、注册/注销、锁、
  detach-before-callback、递归 fire 与 retirement 路径均未变。

这些是 current-tree 的非 material provenance 复核，不改写 retained bundle 的 frozen path identity，也不把
不同 consumer 伪装成 `EXACT`。

## 能力、接口与真实消费确认

当前能力是一个按消费合同分型、caller-owned、稳定地址、零分配的 intrusive membership 集合：

| 任务 | 公共形状 | 常态成本与关键保证 |
|---|---|---|
| 双向 queue/LRU/timer、已知 node 摘链 | auto/safe `IntrusiveList` | node/list 各 2 pointer；已知 mutation、single/whole transfer O(1) |
| safe membership 且 mutation 后频繁读数量 | `CountedIntrusiveList` | 只给选择者的 list 增 1 word；size/single/whole O(1)，普通 range splice O(k)，verified-count 路线 O(1) |
| arena/object/deferred FIFO chain | `IntrusiveForwardList` | node 1 pointer、list 2 pointer；cached-tail append 与 whole handoff O(1) |
| callback/bucket、紧凑 head、已知 node 注销 | auto/safe `IntrusiveHList` | node 2 pointer、head 1 pointer；known-node unlink/replace O(1) |
| 非 owning sequence 处理 | 双向/forward 的 reverse、remove、unique、merge、sort | 零分配；线性算法 O(n)，stable sort O(n log n)，异常后不遗失 membership |

这不是 benchmark 导出的 fast/light 分层。不同类型只为其真实语义支付 count、tail、previous-link 或
auto-unlink metadata。TimerQueue 使用 safe 双链，Sched cancellation 使用 safe hlist；ggml arena/object
chain、wait-set/LRU/cache 与推理 runtime callback 是对应的真实或 provisional consumer archetype。

当前 clean reference identities 为 Boost
`cb94f33fcb13e90922905c2d8105508e71fa6db4`、Folly
`80071a1f79e110b34a2cf7a3a4d01a59e823d697`、oneTBB
`843162db8cf4b2531a4a6147b6ea3fd860d44668`、standalone ggml
`d4716378882593333721eb33f153144b6885caf2` 与 llama.cpp
`c7bda030e7faee594dbe7550185e857351ad405d`。Boost 的 relevant intrusive submodule tree 与冻结 peer 相同；
Folly 的 list/Arena/atomic-list 路径没有相关产品变化，hazard-pointer 增量属于 Mem-reclaim；oneTBB 与
ggml/llama.cpp 增量也没有形成新的 common intrusive `MUST`。

Atomic publication/ABA/reclamation 由 Container/Sched/Mem-reclaim 的 owning protocol 持有；intrusive
tree/hash/heap、base/value/function traits、fancy/offset pointer 与 shared-memory policy 继续路由到
Boost.Intrusive，直到至少两个真实 HYLE journey 证明该依赖路线不足。Waiter、pool free-slot 和 IOCP link
因复用 storage、publication 或 kernel-state 合同保留专用结构。这些边界不是“漏实现 peer 的符号并集”，
也不允许仅凭新 benchmark 扩大 prime factor。

公开 API 仍按标准容器心智提供 iterator/range、front/back、push/pop、insert/erase、splice、clear/consume
与适用 sequence algorithms；safe/auto、counted/uncounted、forward/hlist 的选择和错误选型均在
[`user/container-intrusive`](../../../user/container-intrusive.md) 就近说明。红队没有发现内部 allocation、
lock、atomic、syscall、dynamic dispatch、常态 benchmark metadata、fast/slow seam、旧 C header/schema、
Native FFI option 或隐藏 recoverable failure。Wrong owner/range/count 与失效 lifetime 是程序合同，模块没有
需要 `Status/Result` 的运行时失败面。

## 现有模块与加速设施复用裁决

> 2026-09-05 按通用 closure 第 3 问增补；复核基线为
> `59d8046b38d580081a716f166a496c4a13c5986d`，frozen candidate 到该提交的四个 product headers、模块
> tests/benchmark 与专门 package sample 仍为 zero diff。本节不修改产品或替换原冻结身份。

| Needed mechanism | Existing candidate | Verdict | Reason |
|---|---|---|---|
| hot-path compiler assumption | Core | `REUSE` | 全部 topology 直接使用 `HYLE_ASSUME`/Core compiler contract |
| pointer-to-member owner recovery | Core | `LOCAL` | Core 没有标准 C++ 所缺少的 member inverse；实现留在唯一需要它的 factor-private helper，并对不支持 ABI 编译期 fail closed |
| iterator、invoke、comparison、pointer traits | C++ standard library | `REUSE` | 复用标准 iterator/ranges 心智、`std::invoke` 与 comparator；没有自造 callable wrapper |
| callback delivery | Util-core `FunctionRef` | `NOT_APPLICABLE` | algorithms 是可内联、保留具体 predicate/comparator 类型与约束的模板；类型擦除会增加依赖和间接调用，且没有存储 callback 的需求 |
| allocation与 owning storage | Mem/Container | `NOT_APPLICABLE` | factor 的定义性任务就是 caller-owned、零分配 membership；引入 allocator 会改变语义和常态成本 |
| recoverable error | Error | `NOT_APPLICABLE` | null/already-linked 有显式 try 路线；wrong owner/range/lifetime 是程序合同，不是运行时失败 |
| atomic publication/ABA/reclamation | Mem-reclaim/Container/Sched | `BORROW/ROUTED` | 并发 owner 必须共同拥有同步与 retirement；普通 hook 不重复这套 protocol |
| tree/hash/heap、base/fancy/offset policy | Boost.Intrusive | `BORROW/ROUTED` | 成熟 policy 路线完整，未出现两个真实 HYLE journey 证明需要复制 framework |
| SIMD、CPU/topology/NUMA dispatch | SIMD/Platform | `NOT_APPLICABLE` | link surgery 是依赖地址的 pointer-chasing，比较器由用户定义；没有可跨任务向量化的数据布局或 runtime target 决策 |

源码/include、中央闭包与现有 HYLE factor 反查没有发现重复 allocator、error、clock、CPU probe、wait、
reclamation 或 dispatch。Factor-private member recovery 是 C++ pointer-to-member ABI 边界，Core 没有同义
primitive；把它上移不会减少任何依赖。当前只依赖 Core 是语义/成本审计的结果，不是追求最少引用；没有
遗漏的 material 加速设施或第二 runtime decision authority。该项 verdict 为 `PASS`。

## 质因数与构建裁决

Container-intrusive 是当前内部依赖图中的**质因数 factor**。中央图声明
`_hyle_declare_component(container_intrusive PUBLIC_DEPS core ...)`；精确选择它得到且只得到
`core,container_intrusive`。普通用户不选择内部 factor，而是链接聚合它的 `hyle::container` 产品；四个
split public headers 随 Container 的 source、installed static 与 installed shared SDK 交付。

该 factor 是 header-only，没有 binary、第三方实现依赖或可导出的窄 Native FFI。Pointer-to-member、inline
iterator、caller-owned node storage/lifetime 本身也不是自然的跨语言运行时边界；未来 managed runtime 若需要
timer/callback，应由上层 opaque owning handle API 承担。把四种 topology 再拆 target 不会解除新的依赖环、
隔离可选依赖或形成独立生命周期，反而会把同一 hook/member-recovery 合同碎片化。因此当前不再分解。

## 当前树验证

Windows 权威工作树中的无关既有修改没有触及本 factor；WSL/macOS 使用 reviewed commit 的 disposable
native copy，远端内容没有回写产品。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact `container_intrusive` factor | resolved=`core,container_intrusive` | same | same | `PASS` |
| Release correctness/tool contract | `24/24 PASS`；factor label `6/6` | `24/24 PASS`；factor label `6/6` | `24/24 PASS`；factor label `6/6` | `PASS` |
| headers/no-exceptions/codegen | every gate `PASS` | every gate `PASS` | every gate `PASS` | `PASS` |
| Container source + relocated shared/static consumers | every journey `PASS` | every journey `PASS` | every journey `PASS` | `PASS` |
| current central component contract | `PASS` | selected journey | selected journey | `PASS` |
| docs / closure registry | `check-docs PASS`；`check-module-closure PASS` | repository contract | repository contract | `PASS` |

## 性能与冻结裁决

模块保持 `P1 ENGINEERING PASS`，material margin 为 5%。保留的 `container-intrusive-p1-v2` bundle 已执行
deep verifier：manifest、summary 与 normalized contract-lock 的 SHA-256 分别为
`eb3bf0a4bfed3cb39bf61a239487963fe6d889b13721cc5d996e7057f8dd4283`、
`b9ca233cf0a3be1d50c469e545997c7b871487a63a5b593a1fa277a14294e369` 与
`c10b5e062cb944f06ccb880a7ac1ff454f810cf28906d2b3afad84a0a3efe415`；结果为
`qualification=QUALIFIED`、`paths_verified=true`、`provenance_compatibility=EXACT`、
`current_certification_policy_eligible=true`。当前四个实现 header、bench 与 relevant locked peer 实现无
material 变化；consumer 差异已在上文按语义/计费复核，因此无需用同一 workload 重采样来制造新点值。

冻结的四个 N=4096 consumer workload × 八个 order/placement/allocation strata 全部通过 5% non-inferiority：
counted exact 为 `INCONCLUSIVE` 但非劣门全过，forward 为 `PARITY`，hlist 与 sequence sort+merge 为
`WIN`。A/A、5%/10% known-delta、timer floor、placement、artifact 与 build controls 全部通过。该结论只签
W1-WIN、MSVC 19.50、冻结的 Boost/pre-v4 Sched peer 与这四种正常 shape；不声称跨平台 timing、全部操作和
规模、端到端推理、Formal 或 universal SOTA。

没有发现新的准入能力缺口、复杂度/性能缺陷、benchmark-shaped API、易用性缺陷、旧 C ABI、文档事实错误
或 build/package integration 缺陷。Container-intrusive 因此保持 `CLOSED`，并在 reviewed repository
snapshot 上附加 `FROZEN` 标记。只有 registry 中既有 `reopen_triggers` 的新事实才能解除；peer API 数量、
LOC、新 benchmark 或无新事实的“再看一遍”本身不能。

## Snapshot LOC

`python scripts/loc-report.py --module container_intrusive --commit
5f39c898b6e6680ec0ea8208ad27f125ee51552c --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,255 |
| 自实现不包含测试 | 70,651 |
| Container-intrusive 实现 | 1,590 |
