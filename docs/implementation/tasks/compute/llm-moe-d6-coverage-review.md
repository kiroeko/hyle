# M3-D6：覆盖对账与终审

<!-- MODULE REVIEW: AUDITED -->

## 当前范围

- baseline：`09bf94c7960ba436713103e884433000da9d3c74`，Windows canonical，工作树干净。
- owner：承接 `graph` 当前 [D5 卡](llm-moe-d5-real-fragment-acceptance.md)的文档与证据汇总；
  registry 仍绑定 D5，未完成的 A6 不转移到本卡首次接线或验收。
- 固定目标：八模型、22个固定制品画像、四路线、普通 Direct/Graph，共176格。
- 本批只做文档/状态摘要与已有证据对账；产品、测试消费者、构建和公共性能仪器 FIX 均为零。
  不重开 D1–D4，不新增数学或 Provider，不替换模型格式，不追加 GPU 测量。
- 停止：有限文档集完成后保留 A6 与残余能力裁决，不以“所有格已有结果”声称目标完成。

## 审计发现与有限 FIX

| ID | 已核对的问题 | 本批动作 | 验收 |
|---|---|---|---|
| DOC1 | 22画像当前结果分散在D5 ledger/多份收据，用户难以区分格式、路线与运行范围 | 从ledger逐格投影用户覆盖表，提供固定制品链接、原收据、容量/精度/硬件限制 | 176格与两入口配对核对；不得把拒绝或数值失败标通过 |
| DOC2 | D1/D2需求矩阵仍含baseline的0通过、Marlin/FIX与未接线计数，缺少当前状态指向 | 标明哪些字段是baseline审计，指向D5当前执行账本及D4退场记录；保留原事实 | 固定profile/region需求不改；原计数不伪造为新结论 |
| DOC3 | Compute/Graph用户页仍称GDN/hash-router/sink片段待接入，当前registry摘要也有152格/A5未验等旧数字 | 只校正已有收据证明的阶段、能力归属和摘要；公开语义与数值门不变 | 文档链接/closure/既有ledger结构检查；registry保持原状态与owner |

## 关闭前仍必须成立的事项

| 要求 | 已有证据与当前结论 | 剩余动作 |
|---|---|---|
| 固定画像需求 | [模型事实](llm-moe-model-weight-operator-audit.md)、[需求矩阵](llm-moe-provider-demand-matrix.json) | 文档对账，不重新调研所有模型 |
| 普通入口完整片段、状态、资源/失败 | [D5 ledger](llm-moe-d5-acceptance-ledger.json)：154运行通过、20冷拒绝、2数值失败、0未验 | 22个非通过格仍是不支持域，不能当作通过 |
| source/static/shared | [CPU](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-delivery/README.md)、[CUDA](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-delivery/README.md)、[Apple](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-delivery/README.md)、[Vulkan](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-delivery/README.md)均完成代表消费者验证 | 不把拒绝测试通过称为该画像可执行，不把代表消费者代签任意硬件 |
| 适用完整成本P1 | [CPU](../../../../evidence/performance/graph-cpu-d5-fragments-p1/README.md)、[Apple](../../../../evidence/performance/graph-apple-d5-fragments-p1/README.md)通过；[Vulkan中断](../../../../evidence/performance/graph-vulkan-d5-fragments-interrupted/README.md)；[CUDA预检失败](../../../../evidence/performance/graph-cuda-d5-fragments-preflight/README.md) | GPU两路线尚未完成，维持INCONCLUSIVE；不要求用户提供绝对平稳桌面 |
| 成熟覆盖闭合 | 当前拒绝证明所接路线不能完成，未证明所有合格成熟来源均缺失；[CPU](llm-moe-d5-cpu-official-boundary.md)、[Apple](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-fp4-native-boundary/README.md)均明确此边界 | 不能只把20格改名UPSTREAM_MISSING后关闭；后续若有具名成熟候选，按owner另定有限接线，禁止恢复自研数学 |
| 红队、最终owner状态与停止 | 本批只处理DOC1–3 | A6及残余能力边界未解决前不签CLOSED，不发“基础架构重构完成” |

## 验证预算

一次已有ledger结构检查、覆盖表逐格比对、原收据链接核对、change-impact selector、docs与closure。
registry只更新摘要，不改状态；closure使用上述baseline为`--previous-ref`并生成status页。
统一LOC脚本统计Compute+Graph实现，不含测试；文档更新不构建、不运行P1。

## DOC1–DOC3 完成记录

- [用户覆盖表](../../../user/model-fragment-coverage.md)的22行与176个ledger单元、固定制品链接逐项相符；
  两入口状态相同才合并成一格。所有RUN_PASS单元的region集合完整且逐region为RUN_PASS，
  11份当前主收据的路径与声明范围已对账。该检查不重签原始运行或扩大其数值域。
- D1/D2矩阵只增加baseline/当前运行事实的边界指针，不改变profile_catalog、旧计数、历史FIX或画像verdict。
- 当前registry摘要和生成status页已更新；57个owner的state、record、candidate及performance身份均不变。
  没有用历史CLOSED或这次文档更新给D5新能力补签资格。
- Compute指南纠正了“全部转置MatMul/门控走Native”的过时说明：
  `src/compute_route_cpu/cpu_context.cc`默认先oneDNN、再ggml、最后Native；
  `src/compute_provider_ggml/ggml_cpu_provider.cc`已经包含转置F32 MatMul和SiLU gated原版程序；
  `src/compute_provider_onednn/onednn_provider.cc`的单任务M<16拒绝不代表跳过ggml。
  不修改这些规则或以文档修正宣称新的性能优势。
- `tests/compute_models/d5_acceptance_test.py`通过；selector只选择docs/closure，130项跳过或复用。
  `check-module-closure.py --previous-ref 09bf94c7960ba436713103e884433000da9d3c74`通过，
  docs链接检查与`git diff --check`通过；没有构建或GPU测试。
- 统一`scripts/loc-report.py --module compute-graph`沿既有模块路径统计：**60,373实现行 / 241文件**，
  不含测试。产品`include/src/cmake/CMakeLists.txt`相对`9273bc71`无变化。

DOC1–DOC3到此停止。本卡保持AUDITED；完整D6未完成，GPU A6和上表残余能力裁决仍待处理。
