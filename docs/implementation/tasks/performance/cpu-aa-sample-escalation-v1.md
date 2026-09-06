# CPU P1 A/A 预冻结扩样修复

<!-- INSTRUMENT TASK: COMPLETE -->

## 复现事实

SIMD v6 候选 `fde6fc9c...` 的 W1-WIN P1 回放在预冻结 `n=15 -> n=31` 合同下完成 15 个独立进程样本。
构建、来源、artifact、host、5%/10% known-delta、timer floor 与 placement 均通过；A/A 的
candidate-first 方向为 `0.97667`，98.5% 区间 `[0.95046, 1.00837]`，仅因区间下界穿过既有 5%
等价边界 `0.95238` 而成为 `INCONCLUSIVE`。runner 却把所有非 `PASS` control 当成硬失败，因此没有走
合同已经冻结的 31-sample 第二阶段。manifest
`C:\public\project-hyle\_validation\simd-v6-provider-consumer-replay\perf-evidence-v4\manifest.json`
正确保持 `DIAGNOSTIC/INCONCLUSIVE`；它不是 SIMD 性能失败，也不能用于产品 verdict。

## 有限修复

扩样判定继续要求 product evidence、pre-sampling validators、全部非 A/A controls 和全部产品门没有
`FAIL`。唯一新增许可是：若 A/A 两个固定 order 只包含 `PARITY`，或
`INCONCLUSIVE/median-ci-crosses-decision-boundary`，则把它识别为统计精度不足，允许执行合同原已声明的
31-sample stage。以下情况仍立即停止：A/A `WIN/LOSE`、其他 A/A 原因、known-delta/timer/placement/
delegated failure、产品 `FAIL`、未知或畸形 control。

runner 与 deep evidence replay 共用 `control_allows_sample_escalation()` 语义；单元测试同时证明 A/A
边界未决可以扩样，以及方向性 A/A/known-delta failure 不能借此放行。样本数、5% margin、置信度、
执行顺序、placement、workload 与产品代码均未改变。这是独立公共仪器修复，不计入 SIMD candidate
identity，也不授权其他模块改写历史 evidence。

## 验证

- `python -m unittest tests.tooling.hyle_perf_controls_test tests.tooling.hyle_perf_runner_test tests.tooling.hyle_perf_evidence_runner_test`
- `python scripts/hyle-perf.py verify` 对修复后新 manifest 做 raw n=15 stop-rule 与最终 stage 深回放；
- 旧 v4 manifest 保持不可改写；新 verifier 会以
  `raw n=15 evidence disagrees with the frozen n=31 escalation decision` 拒绝把它续签为当前证据，这正是
  被修复的旧 stop-rule 缺陷，不允许原地补样或改写旧 manifest。

实现 verdict：`FIXED`。48 项 controls/runner/deep-replay 单元测试通过；修复后的 SIMD v6
`perf-evidence-v5` qualification 与 deep verify 均为 `QUALIFIED/CURRENT/EXACT`。该次 A/A 在首个
15-sample stage 已得到 `PARITY`，因此无需实际扩到 31；单元测试覆盖了仅因 CI 穿越边界时扩样以及
方向性 A/A/其他 control 失败时拒绝扩样。公共 schedule 只按“统计精度是唯一未决项”的原意同时
处理产品与 A/A，不会用扩样掩盖真实 control 或产品失败。
