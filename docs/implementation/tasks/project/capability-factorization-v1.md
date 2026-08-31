# 全库能力展开与组件因子化 v1

> 记录性质：`ACTIVE PROJECT ARCHITECTURE REPLAY`。它不改变任何模块的机械状态，也不授权修改产品
> 代码；当前状态仍只以 [`module-status.json`](../../module-status.json) 为准。

## 目标与基线

- 基线 commit：`b59db00d8d7662adadb5af8b7af03eb631aa91f0`；
- 基线 tree：`e7b3a5f2f6bc90b194821c1777542c3c47a27c7b`；
- 目标：先完整统计 13 个基础产品域的应有功能和自然下游形状，再跨域确定 owner、不可再有意义拆分的
  组件因子、依赖图和实施拓扑；
- 冻结：Compute、Graph 保持 `QUEUED`；其他域在能力统计期间不修改产品实现；Mem v2 已有 FIX 集只作
  输入，在全局因子图完成前暂停候选实现；
- 非目标：本轮不跑 benchmark、不签性能结论、不按当前 CMake 图修 API、不按代码行补功能，也不重开
  `CLOSED` 模块。发现真实产品缺口时只登记 `PRODUCT_TRIGGER`，待全局图完成后按拓扑恢复。

## 锁定参考输入

| reference | identity | role |
|---|---|---|
| llama.cpp | `e4b9af007beae34abba094cefd187658e89bbac8` | 真实推理 consumer journey |
| llama.cpp vendored ggml tree | `4660c26f69a7c1ab6c8309934214bde5622483ea` | 与 consumer 同版本的基础实现 |
| standalone ggml | `d4716378882593333721eb33f153144b6885caf2` | 最新独立基础能力差异 |
| Folly | `23ebefdf96c116700d3e134f04d95a2b60a509de` | 成熟基础库 peer |
| oneTBB | `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` | 成熟并行/基础库 peer |
| Boost | `78505eab9b203e3e26e83893d0685f160133fa0f` | 成熟 C++ 组件 peer |

其他域可增加至多两个直接 strong peers，但必须锁定完整身份并写明发现职责。参考更新后先更新身份和
差异，不把“最新版”当功能准入理由。

## 阶段门

### A. 功能超集与自然形状

按 Core、Error、Container-intrusive、Util-core、Time、Platform、Mem、Obs、SIMD + Math、Container、
Util-runtime、Sched、IO 的顺序逐域填写：

| domain | capability family | exact source / journey | semantic variants and lifecycle | natural C++ downstream shape | current coverage | admit/reject reason |
|---|---|---|---|---|---|---|

每域必须覆盖五种输入：ggml、llama.cpp、成熟库、定义性工程能力、HYLE public trace。当前依赖只记录，
不能据此拒绝能力。13 域全部填完、跨域查漏且没有未解释的 `MUST` 后，才冻结 A。

SIMD 在本轮有绑定例外：无论内部后端和 ISA 怎样组织，都保持一个不可拆分的公开产品组件；Math 单独
裁决边界。不能用组件因子化把 SIMD 重新拆成 Highway、xsimd 或 HYLE 多套用户产品。

### B. 跨域归属与质因数分解

只使用已经冻结的 A 输入，建立全局账本：

| capability family | semantic owner | owning product domain | irreducible component | independent consumers | cost/delivery closure | duplicate/conflict resolution |
|---|---|---|---|---|---|---|

`container_intrusive` 作为正例校准：属于 Container 产品域，但因 Time 等消费者只需要零分配侵入式链接，
可独立闭环并避免完整 Container 的 Mem/Platform 闭包，所以是独立因子。没有独立消费、依赖缩减或成本
隔离收益的文件/函数不得制造 target。

### C. 依赖图与施工顺序

为每个因子记录 public semantic/type、private mechanism、build-only vendor、consumer 与唯一 decision
authority 五类边。消除 owner 错位、机制类型泄漏和重复决策后，输出无环中央组件图、旧到新迁移表及
拓扑施工顺序。只有此后才为首个组件建立 `SCOPED/AUDITED` 的有限 FIX 集并修改产品代码。

## 停止与裁决

- A 阶段每个能力行有来源、完整旅程、自然形状、覆盖状态和准入/拒绝理由后停止，不求 peer API 并集；
- A 未完成时，不讨论最终依赖；B 未完成时，不决定实现顺序；C 未完成时，不开始模块重构；
- 发现 correctness/security/build blocker 时建立独立有限任务，不借机提前改产品架构；
- 本记录完成时应给出 A/B/C 三项明确产物；缺少 material 输入则记录具名 blocker，不以“继续研究”结案。
