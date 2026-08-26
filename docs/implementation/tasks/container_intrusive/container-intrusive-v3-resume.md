# container_intrusive v3 收口恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是当前 `RESUME` 轮的施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；v2 的 `HISTORICAL` 记录只提供旧候选和阻塞事实，
> 不授权本轮修改，也不产生当前结论。

## Closure card（`SCOPED`）

- component / owner / round ID：`container_intrusive` / container owner / `container-intrusive-v3`
- task kind：`RESUME`
- 前一 round：`container-intrusive-v2`
- 前一 blocker：冻结候选没有绑定 `closure-perf-v1` 的 W1-WIN P1 engineering `PASS`，最终独立
  red-team 也没有签署 candidate identity。
- 前一 resume condition：在冻结的 5% margin 下生成一份 candidate-bound W1-WIN P1 manifest，随后在
  不迎合 benchmark 改产品形状的前提下取得最终 identity/red-team verdict。
- baseline commit：`a5c74e2be31d258c67e3d6244c8a24d3abeb21c0`
- 进入原因与 demand/defect ID：`INTR-RESUME-PERF-01`（旧工程性能 blocker）、
  `INTR-CURRENT-DRIFT-01`（依赖升级和 build/package v2 后重新绑定当前产品/交付 identity）、
  `INTR-CONSUMER-REVIEW-01`（按真实消费者与成熟库形状复查能力、API 和 negative space）。
- 真实 consumer task / normal workload：`time::TimerEntry` 的显式 safe retirement；task/timer/
  completion/object-pool 中 stable-address、caller-owned node 的 FIFO drain、已知节点取消、遍历、整链/
  range handoff 和 owner/node retirement。
- 范围：公开 auto/safe member-hook 双向链表的能力、C++ API、ownership/lifetime/concurrency/failure
  合同、算法复杂度与常态 layout；`core` 直接依赖、`time` 直接消费者和独立 component 的 source/
  static/shared 交付；现有统一性能合同及用户/架构/证据文档。
- 非目标：完整 owning/concurrent `container`、time 算法、sched/io 私有 link 状态机；node reclamation、
  内部同步、offset pointer、完整 Boost.Intrusive 策略/算法矩阵；C API、Native FFI；Formal release
  certification。
- 直接依赖 / 直接消费者：`PUBLIC core` / `PUBLIC time`；完整 `container` 不聚合本组件。
- tier 与理由：`P1`；公开容器 primitive 的 mutation/traversal/retirement 会在 runtime 高频执行。
- material margin / resource budget：同合同 Boost peer 的每个 required workload 采用冻结的 5%
  material/non-inferiority margin；64-bit auto/safe hook 和 list 各不超过两个 pointer；热操作零分配、
  无锁、无 syscall，已知节点 mutation/range relink 保持 O(1)，遍历/clear/consume 保持 O(n)。
- 总调查预算 / 性能预算：人工审计与归因最多 120 分钟；机器采样只执行 contract 固定阶段，不以墙钟
  截断必需样本。性能最多一轮 `probe`、一轮 fresh n=15 `qualify-platform`；只有统一仪器停止规则允许
  时自动进行全新 n=31。
- 最多复现、机制诊断、产品修改次数：一次正常运行；合同内 order/placement strata；一次 codegen/
  profile 机制诊断；至多一个有机制与消费双证据的产品假设及重跑。不得发明额外 benchmark 姿势。
- machine profile：`W1-WIN` 是 P1 engineering 裁决机；`W1-WSL`、`M1-MAC` 只承担适用 correctness/
  portability 复核，不池化性能样本。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用本轮冻结的风险矩阵和现有
  component/package 合同。
- strong peers：Boost.Intrusive commit `df8eeb90846648464e3c8e378e312f0cec772e83`（同合同能力、API、
  算法与 P1 peer，local source `C:/public/project-hyle/boost`）；Folly commit
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`（消费者形状和能力查漏，local source
  `C:/public/project-hyle/folly`）。每项 capability 最多使用这两个预选 peer。
- 无同合同 peer 时的预算及禁止声明：不适用；Boost 提供 auto/safe member-hook 同合同 peer。Folly
  只用于形状查漏，不把不可对齐入口计入胜负。
- correctness gates：normal、empty、range/same-list boundary、move/swap/destroy/lifetime、callback
  reentrancy contract、random differential、Debug misuse、ASan/UBSan、supported compiler ABI/codegen、
  public-header self-containment 和 `time` 代表消费者。
- delivery gates：当前 component graph/direct target edge；source standalone；relocated static/shared
  SDK consumer；header-only 组件不得制造 binary artifact；不存在 Native FFI surface。
- performance gates：`bench/contracts/container-intrusive-p1-v1.json` 的全部 required comparable
  workloads/strata/controls；contract、peer、candidate、build 和 artifact identity 必须绑定。
- 预期输出：有限 capability/gap ledger 与 `FIX` 集；当前 candidate commit/tree；correctness、delivery、
  docs/usability、red-team 与 P1 manifest；四项 LOC、限制和 reopen triggers。
- 停止条件 / 可接受限制：所有适用工程门通过则 `CLOSED`；仪器/机器/功效仍不能产生 P1 `PASS` 则写
  一个可点名 blocker 后 `BLOCKED`。可接受限制是 64-bit、standard-layout、stable-address、外部同步、
  caller-owned raw-pointer member-hook；没有真实独立需求时不扩成完整 intrusive framework。

## Capability / gap ledger

本节在 `AUDITED` 前保持为空；本轮只允许实现这里冻结为 `FIX` 的项目。

| ID | requirement class | demand kind | exact source | normal shape | owner | ownership/failure/concurrency | complexity/resource target | verdict | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|

- 公开入口反向追踪：审计中。
- 语法消费者 / 语义重复扫描：审计中。
- public/internal/higher-layer 裁决：审计中。
- 本轮冻结 `FIX` 集：尚未进入 `AUDITED`，不得改产品实现。
- 相邻模块 backlog：审计中。

## Candidate

尚未进入 `CANDIDATE`。v2 product identity 只作为审计输入，本轮必须重新绑定当前 commit/tree。

## Verification manifest

尚未进入 `VERIFIED`；不得继承 v2 的点值或 dirty 平台结果。

## 独立红队

在 capability ledger 和有限 `FIX` 集冻结后，由独立 reviewer 攻击当前 round/candidate identity。

## Closure decision

当前为 `SCOPED`，没有 terminal receipt。
