# error v4 收口恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `error-v4` 的 `RESUME` 施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；已终止的
> [`error-v3`](error-v3-closure.md) 只保留历史事实，不在本轮改写。

## Closure card（`CANDIDATE`）

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
| ERR-V4-PERF-01 | 四个 source consumer workload 的全部 order/placement strata | unified W1 qualification | 5% NI PASS | verification pending |
| ERR-V4-DOC-01 | 不再把旧 fixed-backing 8% 或不对称 Result 行当产品结论 | architecture/performance/task docs | PASS | `FIX`（只改证据状态） |
| ERR-V4-GATES-01 | correctness、delivery、API/usability、red-team 仍绑定相同产品 | v3 evidence + current identity checks | PASS | audit `PASS`，candidate 绑定待签 |

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
