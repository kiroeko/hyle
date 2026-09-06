# Frozen 模块文档与记录校准（2026-09）

<!-- PROJECT MAINTENANCE RECORD: CLOSED -->

本轮主要校准已经带 `freeze_confirmation.marker=FROZEN` 的 15 个模块及公共文档合同；同时清理
工作树中未成证据的 Container/Mem/Util 性能残留，并让 current-evidence 规则覆盖 registry 中全部
`CLOSED` 记录。不改变产品代码、模块状态、冻结身份或 reopen trigger。当前状态仍只由
[`module-status.json`](../../module-status.json) 给出。

## 固定范围与身份

- canonical checkout：`C:\public\project-hyle\hyle`；
- baseline commit / tree：`1cfb3b1d8f41a746afa02e24d0d98e3e42b27cba` /
  `bb245b90d69df581462b66411a6a9e188a9e09be`；
- 日期：2026-09-07；
- 范围：`core`、`error`、`container_intrusive`、`util_core`、`time`、`timer_queue`、
  `mem_reclaim`、`obs`、`math`、`platform`、`platform_thread`、`platform_fs`、`mem`、
  `mem_vm`、`simd`。

## 校准结果

- `status.md` 恢复为 registry 的机械投影，明确区分 `CLOSED` 与可选的 `FROZEN`
  confirmation，并把工程性能 verdict 链接到 registry 指定的精确 evidence；
- closure checker 现在验证 freeze confirmation 的状态、marker、ID、日期、完整
  commit/tree 与 evidence；文档 checker 要求每个当前 `CLOSED` 性能页恰有一个 current marker，
  并拒绝不再被 registry 引用的伪 current 页；
- architecture、user guide、performance index 和 task index 中混入当前合同的旧轮次、旧 commit、
  旧任务链接及 Mem v1/v5、Platform v3-v10、Time v4 等叙述已删除或降为明确历史证据；
- 未提交的 Container queue 性能实验依赖三个已经漂移的 consumer source hash，不能作为当前仪器或
  当前结论；其临时源码、合同、validator 和 replay 草稿已移除。旧 Container/Mem 性能页只保留
  对比较类别仍有用且明确标为 historical 的内容；
- registry 只校准 `updated_at` 和 IO 暂停摘要中的过时施工顺序；所有 `product_state`、gate、
  closure/freeze identity 与 reopen trigger 保持不变。改动前快照保存在
  `C:\public\project-hyle\_validation\registry-snapshots\module-status-pre-doc-calibration-2026-09-07.json`
  （SHA-256 `B5FF7DA3BFCBB42B3FF978CB6CD50387FF7852CCE8CA8BCF0E7388E073F56EE9`）；
- 本轮没有把旧 `CLOSED` 自动升级为 `FROZEN`，也没有重写既有不可变 closure evidence。

## 自实现实现代码行数

口径：排除测试、benchmark、文档、脚本、生成物和第三方源码；按 registry 的 factor ownership
列示。15 个 factor 的实现文件物理并集另外做了去重统计，结果同为 **44,992** 行、205 个文件，
所以表内合计没有重复计算公共头。

| factor | implementation LOC |
|---|---:|
| Core | 762 |
| Error | 2,178 |
| Container-intrusive | 1,590 |
| Util-core | 1,549 |
| Time | 1,420 |
| Timer-queue | 438 |
| Mem-reclaim | 227 |
| Obs | 1,829 |
| Math | 420 |
| Platform | 6,964 |
| Platform-thread | 857 |
| Platform-fs | 2,712 |
| Mem | 7,631 |
| Mem-VM | 1,318 |
| SIMD | 15,097 |
| **合计** | **44,992** |

这是本 baseline 的工程规模快照，不是能力、质量或性能得分；后续产品变更必须重新统计，不能把本表
当作实时 registry。

## 验证

- `python scripts/check-module-closure.py --previous <改动前 registry 快照>`：`PASS`；
- `python tests/tooling/check_module_closure_test.py`：`PASS`；
- `python scripts/check-docs.py`：`PASS`；
- Python compile check 与 `git diff --check`：`PASS`。
