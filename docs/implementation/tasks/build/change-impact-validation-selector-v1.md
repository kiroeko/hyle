# Change-impact validation selector v1

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 冻结任务卡

- unit / owner / kind：`build-package` 横向工程单元 / HYLE maintainers / `TOOL`。
  本卡不进入产品 registry，不重开任何产品模块。
- baseline commit：`ff408680036a8a9c344fbe97b911a09f2b265e1c`；
  baseline tree：`95e777fa65e1add4089b8f9f099be79ddf7df7f9`。
  开始时 canonical Windows checkout 的 `main` 与 `origin/main` 一致且工作树干净。
- demand：`VALIDATION-COST-2026-09-07`，owner 本 session 明确要求纠正频繁全量验证。
  当前 `.github/workflows/build-system.yml` 在每次 push/PR 无条件配置全产品、构建
  `hyle_tests` 并运行全部 CTest；文档、状态、格式更新同样付费。
- real consumer：本地模块维护者与 PR/main CI；普通旅程是给定两份 Git 快照，得到可执行、
  可解释的有限验证计划，执行一次并复用身份匹配的证据。
- 范围：变更分类、中央图投影、有限 gate 选择、身份/证据复用、冻结候选 P1 单次执行保护、
  本地/CI 入口、维护文档、selector 合同测试与 build-package LOC 边界登记。
- 非目标：所有 `include/hyle/**`、`src/**` 产品改动；模块 registry/status 变化；
  Container 性能盘点；util/sched/io/compute/graph 实现；真实消费能力审计；
  closure checker、CMake 产品构建语义和性能统计核心重写；运行本库 C++ 构建或 P1。
- tier：`P0` 维护工具；只做 diff、图遍历、hash、进程调度，没有产品运行时成本。
- machine / instrument：canonical Windows，Python 3 + Git + CMake script mode；
  原有 checker、LOC reporter、factor test targets/labels 和 selected-component smoke。
- 预算：审计与实现最多一个工作日；一次有限实现及至多两轮缺陷修复；
  合同测试目标小于两分钟，本库 build/P1/跨平台机器预算为零。
- 停止：下列验收全部完成则 `CLOSED`；若必须改冻结产品、重写统计核心、
  隐藏未知变更或运行全库验证才能满足，则记录具名 `BLOCKED`，不扩项。

## AUDITED：有限 FIX 与验收

| ID | 现有路线 / 缺口 | 冻结动作与验收 |
|---|---|---|
| VI-1 | Git 有 immutable tree；缺少 impact 判定 | 比较完整快照，处理增删/重命名、混合改动；docs/status/可证明格式变化不选 build/P1；未知 ownership 明确阻塞 |
| VI-2 | `HyleComponents.cmake` 已有唯一直接边与 split header owner | 用 CMake script mode 投影，禁止复制依赖表；普通源码只选 owning factor 与一跳消费者，公共头/API/图/package seam 才扩大；只构建所选测试 targets、运行所选 labels |
| VI-3 | 性能仪器已有 contract/资格裁决 | selector 不做性能统计；P1 必须绑定 immutable commit/tree、API/algorithm freeze、contract、toolchain、machine；持久原子 claim，重复/失败/中断不自动重跑 |
| VI-4 | 历史 logs 不能安全自动复用 | gate 证据键包含 candidate tree、contract hash、toolchain、machine profile、工具/命令身份；只有完整成功且原始 log hash 匹配才复用，漂移和损坏拒绝 |
| VI-5 | push/PR 无条件 all-product | CI 常规入口消费同一 selector，上传 selected/skipped/reason 报告；显式手工 full 保留；selector 自身变更只跑工具合同 |
| VI-6 | 维护指南与 LOC 未登记新工具 | 更新 agent/review/build 入口，说明 skip 不等于 PASS、适用未完成为 BLOCKED；固定工程单元计数包含 selector，native 自实现计数不变 |

复用裁决：Git 负责对象身份和 diff，CMake 负责解析其自身 manifest，现有 factor aggregate/CTest
labels 负责测试，现有 selected-component smoke 负责三交付，现有 P1 仪器负责资格与统计。
新工具只拥有选择、执行调度和证据键。无第三方 Python 包、全库 include 扫描或第二份产品状态表。
格式自动判定采取保守可证明范围；不能证明等价时输出需审定的具体原因，不能用 `git diff -w`
吞掉字符串、宏续行或 Python 缩进语义。

## 预声明验证

- 隔离临时 Git 仓库的端到端 selector 合同：轻量变更、因子归属、一跳/公共边界扩展、
  split headers、图删除边、未知路径、缓存命中/身份漂移/log 损坏、P1 冻结拒绝与单次 claim。
- 真实中央图 script-mode 导出及全已跟踪产品路径 ownership 检查；不 configure/build 产品。
- CI YAML/selector 调用、factor targets/labels、零构建路径的静态及进程 spy 检查。
- `python tests/tooling/loc_report_test.py`（LOC 边界变更适用）。
- `python scripts/check-module-closure.py`、`python scripts/check-docs.py`、`git diff --check`。
- `python scripts/loc-report.py --module build-package --commit <full-candidate-commit> --self-check --json`。
- 无 C++ 产品/ABI/ISA 改动，C++ sanitizer、source/static/shared 实跑与 timing comparison
  均 `NOT_APPLICABLE`。测试用无成本子进程证明调度，不产生产品性能结论。

## 结果

工具实现已冻结，后续提交记录其完整 candidate identity。首轮隔离 fixture 的 22 项合同测试用时
23.281 秒通过；据边界复查补充了持久 P1 预算、profile hash、重复执行 lease 与构建前置缓存检查。
最终验收由选择器本身执行本轮所选门后记录，不把首轮测试冒充最终候选证据。

CMake 原 manifest 会在 metadata 后直接进入 configure-only 的 `CONFIGURE_DEPENDS` 审计；
首次 script-mode 诊断因此拒绝执行。最终 adapter 只提取具名 metadata 前缀交给 CMake 原生执行，
并检查边界与后置声明，未改变 `HyleComponents.cmake` 或任何产品构建语义。

## 最终候选与验收

- candidate commit：`b14d680d15a181072f0a18528f761f599b5c29e4`；
  candidate tree：`8e29fa98a96cfc626afb48110215557f4d8429e7`。
- scope card commit：`e97cec9687b69b0c0808960f0e43e94babd83c3c`，先于实现建立。
- 最终工具为 618 行 Python selector、15 行 CMake metadata adapter、集中工具 policy、
  Git fixture 合同测试与 CI/维护入口；不增加产品 factor 或状态轴。
- baseline→candidate 的 `include/hyle/**`、`src/**`、`module-status.json`、`status.md` diff 为空。
  Container 五个因子的 `CLOSED/FROZEN` 和其他模块状态完整保留。

| Gate | Verdict | Evidence |
|---|---|---|
| selector 合同 | `PASS` | 25 tests / 37.369 秒；隔离 Git fixtures，所有 C++ configure/build/CTest/P1 子进程均用 spy 替代 |
| 精确 factor/消费者/公共边界 | `PASS` | queue 源码只选 queue+sched；queue 公共头扩至其反向消费者；text split header 不误认 sequential factor；删边与重命名保留旧、新影响 |
| 无构建的轻量路径 | `PASS` | 本轮自身只执行 docs、closure、LOC test、selector test 四门；65 门带原因跳过 |
| 身份与证据复用 | `PASS` | 同一候选第二次执行四门全部 `REUSED`、零测试重跑；tree/contract/toolchain/machine/log 漂移不能误复用 |
| P1 单次与负向路径 | `PASS` | API/algorithm/profile/contract/immutable identity 拒绝用例、成功复用、失败/中断/并发 claim，以及改日志目录/追加 docs 不能重置额度 |
| LOC boundary | `PASS` | `python tests/tooling/loc_report_test.py`；新工具进入 build-package，native 自实现口径不变 |
| docs / closure | `PASS` | 451 Markdown、1211 relative links；28 components / schema 1 |
| CI 调度 | `PASS` | YAML 解析、selector 调用与 manual-full guard；未声明远端 CI 已运行 |
| diff / product isolation | `PASS` | `git diff --check`；产品实现与 registry/status 无改动 |
| native build / 三交付 / sanitizer / P1 timing | `NOT_APPLICABLE` | 本轮只有维护工具，无产品变动；本库实际编译与采样次数均为 0 |

原始机器输出：[candidate selection](../../../../evidence/validation-selector-v1/candidate-result.json)、
[reuse selection](../../../../evidence/validation-selector-v1/reused-result.json)，四个 gate 的原始 logs 与
hash receipts 在同目录 `receipts/`。两个报告只做无损 JSON compact，不删 selected/skipped reason。
receipt 中的原始绝对路径保留执行事实，归档目录保存同名 receipt/log；可用 `--cache` 指向该目录
只读核验相同 candidate 的复用。P1 原始 manifest 验证仍由现有仪器负责，本轮没有产品性能证书。

四项 LOC 同一 immutable candidate snapshot，命令为
`python scripts/loc-report.py --module build-package --commit b14d680d15a181072f0a18528f761f599b5c29e4 --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 467288 |
| 自实现含测试 | 121658 |
| 自实现不含测试 | 77122 |
| 当前工程单元 build-package 实现 | 10249 |

终态为 `CLOSED`，VI-1～VI-6 有限 FIX 完成，无 blocker。最后 receipt/归档提交只补文档与证据，
仅运行 docs/closure/diff 门，不重复上述已通过工具测试或本库构建。适用范围不包含复杂 formatter
的普遍语义证明、跨 clone 防篡改、模块额外 sanitizer/ISA 风险或所有未知脚本的自动归属。
只有可复现漏选/误扩范围、证据身份误复用、P1 重复执行或真实新增工具消费者，才另开有限工具任务。
