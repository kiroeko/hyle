# 模块重构与收口任务模板 v1

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 复制到 `docs/implementation/tasks/<module>/`。权威状态文件是
> `docs/implementation/module-status.json`，完整判据是
> `docs/developer/review/module-closure-methodology.md`。本页记录该轮事实，不复制当前状态；真正进入
> `CLOSED/BLOCKED` 的 terminal end commit 才把顶部 `ACTIVE` 改为 `HISTORICAL` 并写 receipt marker。

## Closure card（进入 `SCOPED` 前冻结）

- component / owner / round ID：
- task kind：`NEW / REFACTOR / REOPEN / RESUME`
- `RESUME` 前一 round / blocker / resume condition：
- baseline commit：
- 进入原因与 demand/defect ID：
- 真实 consumer task / normal workload：
- 范围 / 非目标：
- 直接依赖 / 直接消费者：
- tier 与理由：`P0 / P1 / P2`
- material margin / resource budget：
- 总调查预算 / 性能预算：
- 最多复现、机制诊断、产品修改次数：
- machine profile：
- instrument profile：
- strong peers（每项 capability 最多两个，绑定版本/commit/source）：
- 无同合同 peer 时的 consumer latency/throughput/resource 预算及禁止声明：
- 适用 correctness / delivery / performance gates：
- 预期输出：
- 停止条件 / 可接受限制：

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE` | missing edge / negative space |
|---|---|---|---|---|---|---|

先从用户任务、定义性能力、当前/北极星消费者和 mature peers 自顶向下填写；再枚举全部现有公开入口
反向追踪。一个函数存在不等于能力完整；单向互操作、缺 teardown、关键边界未定义或 benchmark-only
lifecycle 必须记为 `PARTIAL`。

## Capability / gap ledger

| ID | requirement class | demand kind | exact consumer / peer / standard source | normal shape | semantic / artifact / delivery owner | ownership/failure/concurrency | complexity/resource target | coverage state | `KEEP/FIX/BORROW/DEFER/REJECT` | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|---|

`requirement class` 只用 `MUST/GENERAL`：`MUST` 属本轮冻结范围，必须有实现、自然组合、成熟依赖或
明确裁决；`GENERAL` 只用于成熟库查漏，不会自动扩大本轮。

`coverage state` 只用 `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE`。覆盖状态和施工动作是两个轴；
`KEEP` 只能用于 `COMPLETE` 或已证明完整的 `ROUTED`，不得出现 `PARTIAL/MISSING + KEEP`。

- 现有公开入口反向追踪：
- 已有能力完整性检查（语义变体 / lifecycle / boundary / ownership / interop / delivery / usability）：
- 语法消费者 / 语义重复扫描：
- public/internal/higher-layer 裁决：
- 本轮冻结 `FIX` 集：
- 相邻模块 backlog：

## Candidate

- product commit / tree：
- 公开合同与 user journey 变化：
- ownership / lifetime / concurrency / failure 变化：
- 算法 / 复杂度 / 常态 metadata 与 layout 变化：
- 直接消费者迁移：
- candidate tests：
- harness / tool identity（与 product identity 分开）：

所有 commit/tree/anchor 字段使用完整 40 位 Git object ID；不得在不可变证据中混用短 hash。

## Verification manifest

### Correctness risk matrix

| Risk | `PASS/FAIL/NOT_APPLICABLE/BLOCKED` | 命令 / 证据 | 不适用理由 / 限制 |
|---|---|---|---|
| normal |  |  |  |
| empty / zero |  |  |  |
| capacity / boundary / overflow |  |  |  |
| OOM / system failure / invalid input |  |  |  |
| move / destroy / lifetime |  |  |  |
| concurrency / reentrancy |  |  |  |
| cancellation / close |  |  |  |
| property / differential / fuzz / stress |  |  |  |
| ASan / UBSan / TSan |  |  |  |
| real ISA / backend |  |  |  |

| Aggregate gate | `PASS/FAIL/INCONCLUSIVE` | 命令 / 证据 | 限制 |
|---|---|---|---|
| correctness / sanitizer |  |  |  |
| component / source / static / shared |  |  |  |
| direct consumer |  |  |  |
| P0/P1/P2 engineering performance |  |  |  |
| usability / docs |  |  |  |
| independent red team |  |  |  |

P1/P2 补充：

- frozen workload / peer /计费边界 / margin：
- machine / instrument profile：
- A/A 与 known-delta qualification：
- 正常 consumer/replay verdict：
- 异常后唯一复现、机制诊断和产品重跑：
- Formal 状态（不阻塞本轮）：`NONE/PENDING/CERTIFIED/FAILED/EXPIRED`

## 独立红队

- 审查者 / 日期 / 时间预算：
- capability / contract / complexity：
- correctness / lifetime / concurrency：
- dependency / build / delivery：
- engineering performance validity：

| 问题 | severity | `fixed/disproved/accepted/out-of-scope` | 证据 |
|---|---|---|---|

## Closure decision

- 各适用门最终 verdict：
- blocker，或无 blocker 的依据：
- 已知限制：
- reopen triggers：
- terminal receipt 内容（完整 end commit ID 只由后续 registry 提交记录）：
- terminal receipt marker：只在真正终止的 end commit 把顶部 marker 改为 `HISTORICAL`，并把下方
  `<STATE>` 替换为 `CLOSED` 或 `BLOCKED`：
  `<!-- MODULE CLOSURE RECEIPT: <STATE> -->`
- registry anchor commit：end commit 保存 receipt 内容，不能在该提交内写入自己的 hash；提交产生后，
  由后续 registry 提交记录完整 40 位 end/anchor identity。
- raw evidence：
- 四项 LOC snapshot commit：
- 全库代码：
- 自实现包含测试：
- 自实现不包含测试：
- 当前模块实现代码：
- 建议机械状态：`VERIFIED / CLOSED / BLOCKED`

最后运行：

```text
python scripts/check-module-closure.py
python scripts/check-docs.py
python scripts/loc-report.py --module <component> --commit <full-snapshot-commit> --self-check --json
```

不带 `--commit` 的命令只用于查看当前工作树规模，不能回填 terminal candidate 记录。
