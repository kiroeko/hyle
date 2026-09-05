# 模块重构与收口任务模板 v1

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 使用前确认：本轮对象是边界已经由全局能力/因子审计确定的组件。若功能归属或依赖图仍有争议，先按
> [`capability-factorization`](../../architecture/capability-factorization.md)完成全局只读阶段，不填写
> 本模板并提前修改产品。

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
- 待复用的现有 HYLE/standard/peer mechanisms 与待判定加速设施：
- tier 与理由：`P0 / P1 / P2`
- material margin / resource budget：
- 总调查预算 / 性能预算：
- 最多复现、机制诊断、产品修改次数：
- machine profile：
- instrument profile：
- strong peers（每项 capability 最多两个，绑定版本/commit/source）：
- 无同合同 peer 时的 consumer latency/throughput/resource 预算及禁止声明：
- 使用原型 provenance：`REAL_CONSUMER / NORTH_STAR_JOURNEY / PROVISIONAL_ARCHETYPE / STANDARD / DIAGNOSTIC_SYNTHETIC`
- API 语义档位与选择前提（按任务命名，不使用只有快慢含义的档位）：
- 每档常态成本合同（per-item/per-batch metadata、allocation、queue、copy、atomic、wake/context-switch、syscall、working set）：
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

## API / usability freeze（性能前硬门）

- 状态：`DRAFT / FROZEN`
- 覆盖全部 `MUST` 的普通 C++ journey / compile-only sketch：
- 类型关系、ownership/lifetime/failure/concurrency/default 决策：
- 场景选型表、误用诊断和 guide recipe：
- public dependency / source-static-shared 可发现性：
- freeze identity / reviewer：
- 冻结后允许的内部优化边界：
- 若性能要求退回 `AUDITED`，对应 normal-consumer material evidence 与新增 `FIX`：

P1/P2 计时只能在本节为 `FROZEN`、correctness 实现已绑定 candidate 后开始。benchmark 不得成为新增
public 入口或常态 metadata 的需求来源。

## Owner 七问直答

### 1. 第三方产品路径

| Dependency / locked identity | direct / indirect | `product/test/benchmark/peer` | semantic owner | mechanism / artifact owner | decision authority | delivery owner | public-type leak / upgrade trigger |
|---|---|---|---|---|---|---|---|

- 标准库 / compiler builtin / OS API 合同：
- 无第三方时的 target、源码与锁文件反证：

### 2. 推理引擎能力完整度

- llama.cpp reference / clean locked commit / last remote check：
- standalone ggml reference / clean locked commit / last remote check：
- llama.cpp 随仓 ggml identity（若与 standalone 不同）：

| llama.cpp real journey / ggml foundation difference | admitted demand | HYLE route / owner | `COMPLETE/ROUTED/OUT_OF_SCOPE/MISSING` | evidence / remaining trigger |
|---|---|---|---|---|

- 当前证据下是否存在未处置 `MUST`：`YES / NO / BLOCKED`
- 结论边界：这里只回答已准入常用旅程，不承诺未来永不新增，也不按 ggml API 求并集。

### 3. 内部复用与加速设施

| Needed mechanism / task | existing HYLE / standard / locked-peer candidate | actual route | semantic/cost/dependency equivalence | `REUSE/BORROW/LOCAL/NOT_APPLICABLE` | evidence / reason / owner |
|---|---|---|---|---|---|

- semantic duplicate / private hand-roll scan：
- SIMD、allocator/pool、Container、reclamation、CPU/topology/NUMA、clock、wait/synchronization、hash/text
  等设施适用性：
- 单一 runtime decision authority / dispatch ownership：
- 合法但未采用的候选及其具体语义、成本、codegen、平台、依赖或交付理由：
- verdict：`PASS / FAIL / BLOCKED`

### 4. 拟真、公平性能

- tier / normal consumer task / delivery shape：
- candidate / machine / instrument / peer / resource accounting / margin：
- verdict：`P0 STRUCTURAL PASS|FAIL + TIMING NOT_APPLICABLE`，或
  `P1/P2 NON_INFERIOR|BETTER|FAIL|INCONCLUSIVE`；
- 可以公开的有限 claim：
- 明确禁止外推的 SOTA / 全平台 / 全操作 claim：

### 5. 旧 C ABI / Native FFI 清洁度

- 当前 FFI policy / 是否有已准入 slice：
- public C headers/schema、bridge sources、未装饰 C exports：
- C/managed samples、options、targets、package metadata：
- private `extern "C"` / C / assembly 边界及 owner：
- shared C++ decorated exports 与允许的 symbol evidence：
- verdict：`CLEAN / ADMITTED_FFI_PASS / FAIL / BLOCKED`

### 6. Source / static / shared 构建与导入

| Journey | `PASS/FAIL/NOT_APPLICABLE/BLOCKED` | component/artifact/import evidence | relocation/runtime/known limit |
|---|---|---|---|
| source embedding consumer |  |  |  |
| installed static consumer |  |  |  |
| installed shared consumer |  |  |  |

- 中央组件闭包 / selected headers：
- static compile definitions / transitive link closure：
- shared export/import / Windows staging 或 Unix loader lookup：
- exact package discovery / relocation：

### 7. C++ API 易用性与文档

| User journey | natural public route | ownership/lifetime/failure/concurrency | misuse/compile diagnostics | guide/example evidence | verdict |
|---|---|---|---|---|---|

- 默认路径与显式 opt-in 成本：
- public/internal/benchmark seam 泄漏检查：
- architecture / user / implementation / performance 职责检查：
- 过时 API、C ABI、状态或构建说明扫描：
- verdict：`PASS / FAIL / BLOCKED`

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
| existing-capability reuse / acceleration applicability |  |  |  |
| P0/P1/P2 engineering performance |  |  |  |
| usability / docs |  |  |  |
| independent red team |  |  |  |

P1/P2 补充：

- frozen workload / peer /计费边界 / margin：
- comparison class：`NATURAL_ROUTE / SAME_CONTRACT / MECHANISM_DIAGNOSTIC`
- 语义等价矩阵（`SAME_CONTRACT` 必填；其他类型用于解释边界）：

| Dimension | candidate | peer / adapter | `EQUAL/ADAPTER_EQUALIZED/INTENTIONALLY_DIFFERENT` | charged/resource note |
|---|---|---|---|---|
| work / result / result granularity |  |  |  |  |
| ownership / lifetime / teardown |  |  |  |  |
| failure / OOM / cancel / close |  |  |  |  |
| ordering / capacity / backpressure |  |  |  |  |
| thread-safety / progress / caller participation |  |  |  |  |
| blocking / suspension / wake |  |  |  |  |
| affinity / stealing / topology |  |  |  |  |
| setup / steady-state / teardown charging |  |  |  |  |
| memory-resource / delivery / codegen |  |  |  |  |

- 额外工作/保证方向（`none / candidate / peer / both-different`）及对应矩阵格：
- “苹果对橘子”前置门：若上项不是 `none`，是否已撤销 `SAME_CONTRACT` 与 primitive
  `PARITY/non-inferiority/SOTA`，只保留自然路线用户预算，或已用计费 adapter 真正对齐：
- comparison class 独立 reviewer / 日期（必须在采样前）：

- 资源结果：per-item/per-batch allocation/bytes、queue entries、atomic/lock、wake/context-switch、syscall、working set：
- machine / instrument profile：
- A/A 与 known-delta qualification：
- 正常 consumer/replay verdict：
- 固定方向点值复核：逐项列出全部必需 strata 均 `advantage_ratio < 1` 的 HYLE 行（不论幅度或
  `PARITY/PASS`），并记录 comparison class、A/A/order/placement、一次机制检查、历史/holdout 方向、
  “产品机制 / 额外语义 / 分辨率内无机制 / 未归因 diagnostic”分类及是否满足产品两把钥匙：
- 异常后唯一复现、机制诊断和产品重跑：
- Formal 状态（不阻塞本轮）：`NONE/PENDING/CERTIFIED/FAILED/EXPIRED`

## 独立红队

- 审查者 / 日期 / 时间预算：
- capability / contract / complexity：
- correctness / lifetime / concurrency：
- dependency / build / delivery：
- existing-capability reuse / acceleration applicability / decision authority：
- engineering performance validity：
- comparison class / semantic-equivalence validity：

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
