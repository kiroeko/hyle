# error v4 收口恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `error-v4` 的 `RESUME` 施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；已终止的
> [`error-v3`](error-v3-closure.md) 只保留历史事实，不在本轮改写。

## Closure card（`SCOPED`）

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
| ERR-V4-IDENTITY-01 | v3 error 产品实现与 public surface 未漂移 | git candidate/tree + current diff | exact match | audit pending |
| ERR-V4-CONSUMER-01 | source-primary、自然且对称的 consumer 合同 | contract/source validator | PASS | audit pending |
| ERR-V4-PERF-01 | 四个 source consumer workload 的全部 order/placement strata | unified W1 qualification | 5% NI PASS | verification pending |
| ERR-V4-DOC-01 | 不再把旧 fixed-backing 8% 或不对称 Result 行当产品结论 | architecture/performance/task docs | PASS | verification pending |
| ERR-V4-GATES-01 | correctness、delivery、API/usability、red-team 仍绑定相同产品 | v3 evidence + current identity checks | PASS | verification pending |

除以上五项外无本轮 `FIX` 集。旧 benchmark 仍可保留为 mechanism diagnostic，但不能触发产品修改或
“SOTA/全面胜出”声明。
