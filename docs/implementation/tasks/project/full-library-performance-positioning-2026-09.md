# 全库性能定位与差距消除（2026-09）

<!-- PROJECT PERFORMANCE AUDIT: CLOSED -->

## 任务卡

- owner / kind / demand：HYLE maintainers / `BOUNDED_PROJECT_PERFORMANCE_AUDIT` /
  `PROJECT-PERFORMANCE-POSITIONING-2026-09-11`。目标是用可复核事实回答“已有更完整的成熟库时，
  为什么选择 HYLE”，而不是继续用“5% 门已通过”代替性能方向或宣称全库绝对 SOTA。
- baseline：`bbd1eebada1be72d5b33c07e8c0763b16196d3fb`；canonical checkout 为
  `C:\public\project-hyle\hyle`。
- 状态输入：[`module-status.json`](../../module-status.json) 中除 `compute`、`graph` 外的全部
  `CLOSED` factor。横向 build/package 只审交付成本，不伪造 runtime workload。
- 输入：当前 registry 选择的性能 evidence、仍可机械重放的 raw/summary/manifest、P0 结构记录、
  当前产品源码与强 peer 源码；历史失效证据只作 provenance，不自动续签。
- 真实 workload 来源：ggml/llama.cpp、vLLM、SGLang、ds4 的推理路径，以及异构渲染/资源流送、
  批处理数值管线和低延迟服务中的相关正交形状。
- scope：校准性能 verdict 与产品行动阈值；建立唯一去重总账；清点全部当前胜、平、未知和固定输项；
  对高权重固定输项先做源码/复杂度/成本归因；只有满足合法 trigger 时才定向重开、修复和回归；形成
  用户可读的 HYLE 差异化价值与严格声明边界。
- non-goals：修改 `compute`/`graph`；为凑“胜率”新增 benchmark-only API、layout 或常态 metadata；
  把不同语义、交付、ISA、冷热、完成边界或资源预算混成排名；按模块机械重跑全部 P1；把未测 API
  猜成 PARITY；承诺永久、跨平台、跨 workload 的“全库 SOTA”。
- 调查预算：先完成零采样 inventory；每个疑似固定输项最多一次合同复核和一次静态机制归因。
  只有仍为高权重、同合同且无法由现有证据裁决的项，才建立独立、预冻结的 P1/P2 子卡。每个有效
  product loss 最多一轮产品纠正；fixture/tool/environment 问题进入独立任务，不能在模块内扩建仪器。
- 运行预算：第一阶段不运行 P1/P2。后续只运行子卡预声明 workload，使用物理机互斥；GitHub hosted
  CI 额度耗尽期间保持 hosted 资格 `UNPROVEN`，不触发无关 workflow。
- 适用验证：本轮文档/账本/工具单测、`check-module-closure.py`、`check-docs.py`、`git diff --check`；
  只有产品或仪器实现变化时才由 change-impact selector 增加 native、交付或性能门。
- 停止：29 个 CLOSED factor 均有性能去向；独立 workload 已去重；每项有合同、peer、平台、方向、
  practical materiality、根因/未知与消费权重；全部固定输项均已裁决为定向 FIX、诚实取舍或具名未知；
  没有无界“继续研究”；最终文档可直接回答 HYLE 已证明的优势、劣势和未证明范围。

## 判别合同

同一条证据同时回答四个不同问题，禁止互相覆盖：

1. `comparability`：是否为相同语义、生命周期、进展与计费边界；
2. `qualification`：当前机器、artifact、controls 与统计是否有资格；
3. `direction`：在 ratio `1.0` 两侧是否得到 `WIN / LOSS / INCONCLUSIVE`；
4. `engineering action`：差距是否超过预声明消费者预算，是否具备机制与消费两把钥匙。

总账的可读 verdict 使用 `WIN / PARITY / INCONCLUSIVE / LOSS / N/A`：全部适用 strata 合格且置信
区间固定在 1 上方为 `WIN`，固定在 1 下方为 `LOSS`；方向翻转、区间跨 1 或证据不足为
`INCONCLUSIVE`；只有没有固定方向且通过预声明双边 practical-equivalence 门，或产品热路/生成代码
已证明等价时才写 `PARITY`；无合理 peer 或只适用 P0 时为 `N/A` 并保留结构成本结论。稳定 1% 落后
仍是 `LOSS`，只是可能低于产品纠正优先级；5% 默认 margin 只属于工程 non-inferiority / 行动预算，
不能把方向性 `LOSS` 改名为 `PARITY`。

## 阶段账本

| 阶段 | 输出 | 状态 |
|---|---|---|
| P-A 合同校准 | architecture/developer 规则和机器可检查字段一致 | CLOSED |
| P-B 零采样清点 | 29 factor、52 个去重 workload、历史/缺失边界 | CLOSED |
| P-C 固定输项归因 | 同合同、源码成本、顺序/地址/core、消费权重与根因 | CLOSED |
| P-D 定向修复 | 没有同时具备同合同固定聚合 LOSS、可行机制和 material 消费影响的合法 reopen | CLOSED / NO REOPEN |
| P-E 真实纵切 | 已保留公开 transport + HTTP/1 流式路线；端到端推理 P2 明确未证明 | CLOSED |
| P-F 产品结论 | 当前全库 scorecard、Why HYLE、限制与后续机会 | CLOSED |

## 最终输出与裁决

- registry 起点：29 `CLOSED`、2 `QUEUED`；本项目审计本身不创建第二套模块生命周期状态。
- 机器总账：[`full-library-scorecard.json`](../../evidence/performance/full-library-scorecard.json)，
  SHA-256 `1b6b62eb4ac0eddfa4cc7f0d4ae46cb618a07034e813c097e7943835e8dd50f4`。
- 人读结论：[`full-library-scorecard.md`](../../evidence/performance/full-library-scorecard.md)。当前 52 个独立合同组为
  `23 WIN / 2 PARITY / 16 INCONCLUSIVE / 11 N/A / 0 聚合 LOSS`；计数不是胜率。
- 唯一保留的固定负向分层是 WSL exact-file watcher 的两个 cell；其 generic multi-target、UTF-8
  validation、coalescing 与 fiber wake 是必要产品成本，当前低权重 reload 路线裁决为
  `ACCEPT_TRADEOFF`。其他小幅负向 median 均未得到固定方向，如实保留为 `INCONCLUSIVE`。
- 本轮没有修改产品源码、性能仪器或 `module-status.json`，也没有重开任何 factor；因此没有
  为文档/账本变化重跑 P1/P2，也没有消耗 hosted CI 额度。
- 结论：HYLE 有可复核的局部 SOTA routes，但不具备“全库 SOTA”或“端到端推理更快”证据。
  用户当前选择 HYLE 的可辩护理由是成熟 provider 广度、有限已证赢家和跨模块低税组合；
  compute/graph 和首个真实推理消费者完成后，再依法触发 P2。
- terminal verdict：**`CLOSED`**。完整性检查器
  [`check-performance-scorecard.py`](../../../../scripts/check-performance-scorecard.py) 的 SHA-256 为
  `0878a41dadb09d5d267e84828b4ae820c6e1a803b6ed3f4f1d47a76cdb68d436`；最终验证命令与结果见下方。

## 最终验证

| 验证 | 结果 |
|---|---|
| `python scripts/check-performance-scorecard.py` | PASS：29 CLOSED factor / 52 claim / 2 excluded |
| `python -m py_compile scripts/check-performance-scorecard.py` | PASS |
| `python scripts/check-module-closure.py` | PASS：31 component / schema 4 |
| `python scripts/check-docs.py` | PASS：240 Markdown / 906 relative link |
| `git diff --check` | PASS |
