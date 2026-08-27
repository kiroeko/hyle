# error v4 收口恢复记录

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本页是 `error-v4` 的 `RESUME` 施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；已终止的
> [`error-v3`](error-v3-closure.md) 只保留历史事实，不在本轮改写。

## Closure card（`CLOSED`）

- component / owner / round ID：`error` / error owner / `error-v4`；task kind：`RESUME`。
- 前一 round / blocker：`error-v3` / `ERR-PERF-CONSUMER-KEY-01`。旧合同把 fixed-backing atomic
  microbenchmark 误标成 replay，且一个 `Result` peer ownership 不对称，因此不能产生产品性能裁决。
- 前一 resume condition：用同一父构建图、自然且对称的 ownership、真实或忠实 source consumer
  trace 建立新合同；mechanism microbenchmark 只能诊断，不能裁决产品。
- baseline commit：`d9644548bf1690a4a1c86f205936cd857b1b92e8`。
- 范围：仅验证上述恢复条件，并把有效的 source-consumer performance evidence 重新绑定到未变化的
  error 产品候选；同步 architecture、performance evidence、registry 和状态投影。
- 非目标：新增 error 能力、改公开 API/layout/ownership、按 benchmark 改产品；logging、exceptions、
  arbitrary payload/cause/source-location、MultiError、完整 foreign-category adapter、下游模块收口、
  Native FFI/C API、Formal release certification。
- 真实 consumer task：`graph::TaskGraph::evaluate` 64-node success/fresh owning last failure、
  `graph::ParallelEvaluator` first-error owning handoff、`sched::JoinState` failed owning handoff。
- 直接依赖 / 直接消费者：`PUBLIC core`；graph/sched 是本轮性能来源消费者，模块自身收口仍归各自 owner。
- tier / margin：`P1` / 冻结的 5% engineering non-inferiority margin。
- machine / instrument：`W1-WIN` / `closure-perf-v1`，合同
  `bench/contracts/error-source-consumer-v1.json`；`W1-WSL`、`M1-MAC` 沿用候选正确性/可移植性证据，
  不池化性能样本。
- strong peer：Abseil commit `81c993af4fb88b3a51829a796df02251fa27102f`，只比较同 consumer
  algorithm、同 ownership topology、同观察结果的 `absl::Status`。
- 调查与采样预算：不改产品；一次 clean W1 qualification，n=15，只有统一仪器停止规则允许时才 fresh
  n=31；不得增加姿势或追逐 fixed-backing 旧点值。
- correctness / delivery / API / docs：产品树若与 v3 candidate 完全相同，则重验 identity 和现行门，
  不伪造重复产品施工；任一 drift 都退回 `AUDITED` 重新评估。
- 停止条件：恢复条件、identity、全部适用门和独立复核通过则依次进入
  `AUDITED -> CANDIDATE -> VERIFIED -> CLOSED`；有效合同发现 material regression 则进入有限产品诊断，
  无有效裁决则重新 `BLOCKED`。

## 恢复 ledger

| ID | requirement | evidence owner | required verdict | current disposition |
|---|---|---|---|---|
| ERR-V4-IDENTITY-01 | v3 error 产品实现与 public surface 未漂移 | git candidate/tree + current diff | exact match | `PASS` |
| ERR-V4-CONSUMER-01 | source-primary、自然且对称的 consumer 合同 | contract/source validator | PASS | `PASS` |
| ERR-V4-PERF-01 | 四个 source consumer workload 的全部 order/placement strata | unified W1 qualification | 5% NI PASS | `PASS` |
| ERR-V4-DOC-01 | 不再把旧 fixed-backing 8% 或不对称 Result 行当产品结论 | architecture/performance/task docs | PASS | `PASS` |
| ERR-V4-GATES-01 | correctness、delivery、API/usability、red-team 仍绑定相同产品 | v3 evidence + current identity checks | PASS | `PASS` |

除以上五项外无本轮 `FIX` 集。旧 benchmark 仍可保留为 mechanism diagnostic，但不能触发产品修改或
“SOTA/全面胜出”声明。

## Audit result

- `git diff 9595724a -- include/hyle/error src/error` 为空；error 产品实现和公开 surface 仍是 candidate
  `9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9` / tree
  `a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d`。本轮 benchmark、合同、validator 和文档不进入产品树。
- 新合同 ID/hash 为 `error-source-consumer-v1` /
  `dde5b9a2e8c85d11014d2bb8312a948da422af1b1d339f09ef1046f9e1ec6f97`；benchmark source hash 为
  `ebb8c919a1d5236138cbd865d442e87d8f7f9167b02946b795df463259c96466`。
- source provenance 固定为 `src/graph/taskgraph.cc`
  `9adb3f42e9cf9fd11c4edf961f2d121fe06341f6b6417b79ee4c8ecb9a2078f5`、
  `src/graph/parallel_eval.cc` `5a4888eab4a46f670f81f8a112587cc5c2f5b0872592b5b132f1a2e8ee0f7fb4`、
  `include/hyle/sched/join.h` `29d1ac248c8ee3fbf9e9e8ba8f7cd3c22f8b900c7182508f12f471b88065eb3e`。
- HYLE 与 Abseil 共用同一模板 consumer algorithm；fresh owning failure、copy/move handoff、source
  location、最终 observable 与顺序/placement strata 都由合同锁定。pre-sampling artifact/build/source
  validators 全部 `PASS`。
- 旧 fixed-backing shared-copy 约 8% 慢只保留机制诊断；旧 `Result` rvalue/lvalue 非对称行的 derived
  product verdict 继续拒绝。审计没有产生产品 `FIX`，也没有发现恢复 ledger 外的新功能/API 缺口。

## Candidate

本轮复用且仅复用未变化的 error 产品候选：

- commit：`9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9`；
- tree：`a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d`；
- current identity check：`include/hyle/error` 与 `src/error` 相对该 commit 的 diff 为空。

`d9644548` 及以后新增的 source-consumer benchmark/contract/validator 只负责证明候选，不属于候选产品
实现，也不改变 API、layout、ownership、复杂度或交付 closure。故本轮不制造一个仅含证据文件的伪产品
candidate；若后续 identity check 发现产品 drift，必须退回 `AUDITED`。

## Verification manifest

所有产品门重新绑定 candidate `9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9` / tree
`a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d`。产品路径 diff 为空，所以 v3 已通过的三平台
correctness、public-header、source/static/shared delivery、API/usability 和底座 adoption 证据仍证明完全
相同的机器码来源；v4 没有用新文档冒充新产品验证。

P1 engineering bundle：

- W1-WIN，fresh MSVC x64 Release/static、portable ISA、LTO off；Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f`；n=15，未触发 n=31；
- contract SHA-256：`dde5b9a2e8c85d11014d2bb8312a948da422af1b1d339f09ef1046f9e1ec6f97`；
- retained manifest：
  `C:/hyle-scratch/error-source-consumer-v1-d9644548-win-rerun02-evidence/manifest.json`，SHA-256
  `0afe6d46813cab7c6b949b1f4478544c99b7f040054d579f6839bc19307b7c13`；summary/observations SHA-256
  为 `97873ce6a55d96792a3a269984488f95afa4233f5b2d1976a6259bd9b566d39f` /
  `6cc1921740edf57feecda19f4c0e15c96533cfa724ad6c133a1b63414611c408`；
- artifact/build/source-consumer validators、A/A、known delta、timer floor、placement controls 全
  `PASS`；4 workload × 4 strata 全部 comparable、5% NI `PASS`；
- TaskGraph success combined `PARITY`（ratio 0.9984–1.0018）；fresh owning last failure `WIN`
 （1.1950–1.2160）；parallel first-error handoff `WIN`（1.4083–1.4181）；Join failed handoff `WIN`
 （1.5030–1.5221）；
- deep verifier：`qualification=QUALIFIED`、`paths_verified=true`、
  `current_certification_policy_eligible=true`。

当前复核：70 个 tooling tests、module closure contract、264 Markdown/744 relative-link docs contract、
source-consumer deep replay 和 product-path identity check 全部 `PASS`。独立于旧 v3 computed verdict 的
post-terminal red-team finding 已被 fail-closed contract policy、source provenance 和 ownership validator
覆盖；旧合同的 fixed-backing 8% 与不对称 Result 比较仍明确拒绝为产品裁决。

aggregate gates：scope、architecture、correctness、delivery、usability/docs、red-team、engineering
performance 均为 `PASS`。Formal 为独立 `PENDING`，不属于本轮工程验证。

## Closure decision

本轮 `CLOSED`。旧 `ERR-PERF-CONSUMER-KEY-01` 的恢复条件已由新的 source-primary contract 完整满足；
没有改产品实现，也没有把 fixed-backing microbenchmark 的约 8% 差距转成产品 shape。四个真实消费者
workload 的所有 order/placement strata 都通过冻结的 5% P1 engineering non-inferiority gate，候选的
correctness、delivery、API/usability、architecture/docs 与 red-team 门因 exact product identity 不变而
重新绑定并通过。

可接受限制：

- P1 `PASS` 只覆盖 W1-WIN、固定 Abseil peer 和四个 source consumer workload；Formal、跨平台性能和
  “所有 error 操作都更快/SOTA”均未签署；
- fixed-fixture shared owning-message copy microbenchmark 仍可复现约 8% 慢，但 consumer materiality 与
  原因未成立，只是 mechanism diagnostic；
- low-bit header-inline representation 要求 header/artifact 完全匹配和 full consumer rebuild；当前不承诺
  稳定 C++ ABI 或 Native FFI；
- graph/sched consumer source hash、合同、peer、machine、instrument 或交付形状变化时，当前工程声明必须
  重新资格化；io 与其他未收口模块的冷失败通道 adoption 仍归其 owner。

重开条件：真实消费者发现 correctness/security/ownership/API-usability/交付缺陷或本边界内缺失能力；
Status/Result 算法、layout、公开 surface、直接依赖或受支持编译器语义变化；成熟同合同 peer 证明更优
复杂度或正常 consumer workload 上可复现的 material regression；retained bundle 不再可深重放，或合同、
peer、machine、instrument identity 改变而仍需续签声明。

终态 identity 绑定 VERIFIED commit/anchor
`26679f1b2a1987b39cdf302eb95ed3e5fefeb168`；产品候选仍为 `9595724a` / `a828fb4c`，终态文档提交
不冒充产品 candidate。四项 LOC 从 immutable candidate 运行
`python scripts/loc-report.py --module error --commit <candidate> --self-check --json`，self-check `PASS`：

1. 全库代码：293,909；
2. 自实现包含测试：122,027；
3. 自实现不包含测试：82,135；
4. error 实现：1,350（`include/hyle/error` + `src/error`）。
