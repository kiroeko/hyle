# Structured pair placement validator 修复

## Task card

- trigger：Sched v6 的首个合法 `structured-paired-process-v1` × placement 合同，在正式采样前被
  `_validate_variant_artifacts` 拒绝；实际四个二进制的路径、长度与 SHA-256 均不同。
- root cause：校验器已经要求每个 `target_variant` 恰有 candidate/peer 两个产物，随后却又要求
  `target_variant` 数等于产物数。两角色 × 两 variant 必然得到 `2 != 4`，因此这种合同不可能通过。
- scope：仅修正公共性能仪器的产物身份检查，并添加结构化 pair 回归 canary。
- non-goals：修改 Sched 产品候选、benchmark workload、margin、样本数、统计方法、机器合同或历史证据。
- stop：若修复会允许同一路径/同一 binary 充当 placement、破坏普通 nanobench 合同，或需要改变
  产品 API/算法，则停止。

## 决策与结果

placement 路由身份是 `(target_variant, pair_role)`。普通 nanobench 每个 variant 仍只允许一个产物；
structured pair 每个 variant 必须正好有 candidate 与 peer 两个不同产物，且所有已登记路径与内容仍须
互异。Sched 合同只是暴露缺陷的首个真实 canary，不获得例外。

验证只运行 `hyle_perf_runner_test.py`、完整 tooling tests、合同 validate、文档/closure checker 与
`git diff --check`。修复通过后重新创建全新的 P1 build/evidence 目录；失败运行不作为性能样本。
