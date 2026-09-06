# closure-perf-v1：control precision escalation

<!-- INSTRUMENT TASK: ACTIVE -->

## 触发与边界

- baseline：`26da9d3dc24e78b920d60e5743da5726e0fe5ca0`；owner：公共性能仪器；日期：2026-09-06；
- trigger：Platform v10 的唯一 fresh holdout 在 n=15 时 4/4 product non-inferiority、A/A、10% delta、
  timer floor 与 placement 均通过；5% positive control 的两种顺序 median 都在预注册容忍带内且 10% 响应
  继续单调，但一个 99.26% CI 下界尚未证明大于 1。合同已预声明 `[15,31]`，runner 却只允许 A/A 的
  precision-only uncertainty 进入 n=31，把同类的 positive-control precision uncertainty 当成 terminal
  failure；
- 这是通用 stop-rule 不完备，不是 Platform workload 或结果特例。修改不得读取产品 ratio 决定规则，
  不改变 control 阈值、统计量、confidence level、产品 margin、peer、row、执行顺序或 qualification gate；
- 非目标：让任意 failed control 重跑、按结果追加样本、修改 n=31 结果、放宽 known-delta、重写 benchmark、
  重跑其它模块或把 diagnostic bundle 升格。

## 预冻结规则

`control_allows_sample_escalation()` 对 required common control 只接受三种状态：

1. `PASS`；
2. A/A 的既有 precision-only 状态：两个固定 order 仅为 `PARITY`，或因 CI 穿过已冻结等价边界而
   `INCONCLUSIVE`，且至少一个 order 需要更多样本；
3. known-delta 的 precision-only 状态：
   - kind、两个固定 order、accepted median band 和 estimate 结构完整；
   - `monotonic_5_to_10 == true`；
   - 每个 order 至少 15 个独立样本，median 均落在既有 accepted band；
   - interval 有限、有序且包含 median；
   - gate 唯一未满足条件是至少一个 order 的 CI lower `<= 1.0`，即尚不能证明注入的正差异，而不是
     median 错位、响应反向、数据损坏或其它 control failure。

只有合同预先声明 `[15,31]`、全部 required controls 为上述状态、全部 product NI 为 `PASS/UNRESOLVED`
且至少一个 product/control 确实 precision-unresolved 时，才完整替换运行 n=31。n=31 仍以原 gate 终裁；
失败就是 terminal failure，不再扩样。runner 与 evidence verifier 必须调用同一纯函数并重放同一决定。

## 验证与停止

- unit canary：接受 median-in-band + monotonic + CI-crosses-one；拒绝 median out-of-band、non-monotonic、
  insufficient/malformed/non-finite interval、无 unresolved order 与其它 control kind；
- runner/evidence parity：同一 synthetic controls + product summary 的 live/replay escalation 一致；
- tooling suite、docs contract 和 module closure contract 必须通过；
- 规则实现并经 synthetic canary 证明后停止。Platform 只能另建 `RESUME` round，使用 clean candidate、
  clean peer 和 fresh paths 运行一次；本任务不裁决 Platform。
