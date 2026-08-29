# 性能消费者 provenance 精确续接任务

<!-- PERFORMANCE INSTRUMENT TASK: COMPLETED -->

## 问题

统一仪器把真实 consumer 源文件整文件 SHA-256 作为产品 replay 的 provenance。严格 hash 能阻止旧
证据静默套到新消费者上，但同一文件内与 workload 无关的注释或相邻 API 迁移也会让历史 bundle
停止。Time 的 retained bundle 正是这种情况；util-core 当前 provenance 完全精确，实际没有失败。

不能用以下方式处理：忽略所有 source drift、按文本启发式猜测相关代码、修改旧 manifest，或为一个
模块硬编码例外。它们会让真正的 ownership、计费边界或消费者频率变化漏过。

## 冻结方案

新增仓库级 `hyle.perf.provenance-renewals/v1` 账本。默认仍要求 byte-exact；只有同时满足以下条件才
接受历史 engineering replay：

1. contract ID 与 semantic SHA-256、workload ID、repository-relative path 全部精确匹配；
2. frozen hash 到 current hash 存在唯一、无环的显式迁移链；
3. 每条边固定为 `NON_MATERIAL / ENGINEERING_REPLAY_ONLY`；
4. 每条边绑定一份 repository-relative review record 及其 SHA-256；
5. review 逐项证明 semantic contract、ownership topology、material contribution、charged phases、
   algorithm 与 delivery 未变；
6. Formal evidence 不接受 renewal；任何未知、分叉、循环、缺失或被修改的审查记录继续 fail closed。

verifier 输出 `EXACT`、`REVIEWED_NON_MATERIAL` 或 `NOT_APPLICABLE`，并在使用迁移时输出账本 hash 与
实际迁移边。runner 生成新资格证据时仍必须与合同 provenance byte-exact；renewal 只解决历史证据的
当前工程回放，不成为采样捷径。

## 本轮边界

- 修改 owner：`scripts/hyle_perf/provenance.py`、evidence verifier、CLI、tooling tests、统一仪器文档；
- 登记 Time `scheduler-timer-lifecycle-64` 的一条精确非实质迁移；
- 不修改 Time、util-core、sched 或 platform 产品代码，不重跑性能采样，不重写任何 retained bundle；
- 不裁决 Time candidate 之后的公开头产品变化；该事实留给 Time 产品回审。

## 验证计划

- exact provenance 无需账本且继续通过；
- 精确单边与链式 renewal 通过，并报告兼容等级；
- 错 contract/hash/path/workload、分叉、循环、review record 漂移与 Formal 使用全部拒绝；
- util-core retained bundle 保持 `EXACT / QUALIFIED`；
- Time retained bundle 得到 `REVIEWED_NON_MATERIAL / QUALIFIED`；
- 其他 retained error/platform/mem/container-intrusive/obs 包不得回归；
- tooling tests、`check-docs.py`、`check-module-closure.py` 与 diff hygiene 通过。

## 实现与结果

- 新增 `scripts/hyle_perf/provenance.py`：严格 schema、重复 key/未知字段拒绝、唯一无环 hash 链、
  hashed review record 与 engineering-only scope；
- `verify` 新增 `--provenance-renewals`，默认读取
  `bench/performance-provenance-renewals.json`，并输出兼容等级、账本 SHA-256 和实际迁移边；
- manifest、contract lock、summary、raw logs 和旧 provenance hash 均保持原字节；runner 的新采样路径
  仍要求 contract 与当前 consumer byte-exact；
- Time 精确迁移的 review record 位于
  [`time scheduler provenance review`](../../evidence/performance/provenance-reviews/time-scheduler-timer-lifecycle-f3db-to-d7ff.md)。

2026-08-29 验证结果：

- tooling `77/77 PASS`；精确、两边链式、错合同、Formal 拒绝、review 篡改和 branching canary 均通过；
- Time retained bundle：`QUALIFIED / REVIEWED_NON_MATERIAL /
  LEGACY_COMPATIBLE_TRACE_DISABLED / paths_verified=true`；
- util-core：`QUALIFIED / EXACT`，确认它不含 scheduler/Time provenance；
- error、platform、mem owner、mem handoff、obs trace、container-intrusive 与上述两包共 8 份 retained
  bundle 全部深验通过，没有公共放宽回归；
- ledger 当前 SHA-256：
  `eaa83115e82d40bb001f1eeacd43f5c11c4f375ba65e6cbd95165896bcfc25d9`。

本任务完成后停止。Time 公开头 candidate drift 仍由 Time 产品回审独立裁决，不能借 provenance
renewal 消除。
