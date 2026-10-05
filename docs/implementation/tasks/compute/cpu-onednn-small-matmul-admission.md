# CPU oneDNN：恢复小矩阵的真实准入

<!-- MODULE REVIEW: CLOSED -->

## Closure card

- owner：`compute_provider_onednn`；baseline `a1a1de89`，Windows canonical。
- trigger：[实际准入反例](../../../../evidence/correctness/cpu-native-domain-admission/README.md)。
  普通 M1 prepare 被旧“Native 更便宜”条件拒绝，而同版原 BRGEMM carrier 正确完成 M1/K64/N64。
  这是已接成熟能力的错误准入，不是 benchmark 排名触发的重开。
- 原 terminal：[模型画像冷路由](llm-moe-model-profile-routing-refactor.md)。其 API/画像/规则能力保持；
  本轮只改 BRGEMM 小矩阵准入，不继承该 terminal 授权改其它域。
- 主 consumer：LLM decode 的 dense F32 M1/M4 投影，用普通 CPU factory 或专家 oneDNN Context
  prepare、caller workspace、独立 invocation、submit/wait 后重复执行。
- 正交 consumer：批处理数值中的 M15/M16、奇数 K/N 矩阵乘法，同一 C++ 生命周期。
- scope：非空 exact dense row-major F32、F32 accumulation、现有 BRGEMM 支持的维度。
  空维度、转置权重、encoded dtype、bounded shape 仍按原真实能力拒绝；不新增 kernel/线程池。
- 依赖：既有 pinned oneDNN 3.13.2 BRGEMM、Compute/Mem/ParallelExecutorRef；直接消费者 CPU Direct
  及 Graph leaf。完整 ggml region 仍由原规划绑定，不为此拆分。
- public API、资源 owner、完成/错误、并发与构建因子冻结：没有新增或改变类型；只扩大真实能力的准入。
- tier：P0，本轮纠正 capability filter，不声明 Native/oneDNN 胜负；自研裁决仍由
  [CPU 逐域卡](cpu-native-provider-domain-decision.md)以不可变候选启动适用 P1。
- 预算：一次源码/实际准入批次、一次产品修改与一次批量验证；最多一次由具体失败触发的纠正。
  不重写性能仪器或公共治理，不优化 Native，不增加 benchmark 目录。
- 预声明验证：Windows/Linux oneDNN provider/normalization/tensor/composition/Graph 定向测试；
  CPU ordinary factory 的小矩阵 Provider 身份和实际输出/重放；失败/shape/resource 已有相关测试；
  两平台 source/static/shared 独立普通小矩阵 consumer（复用现有 producer/SDK 和 DLL 邻置）；
  change-impact plan、closure/docs/diff。未运行项保持 UNPROVEN。
- stop：适用验证通过后冻结候选/terminal；若原 carrier 在真实普通生命周期失败，则记录具体事实，
  退回有限纠正或 BLOCKED，不恢复未经比较的 Native 优势假设。

## 能力审计与有限改动集（已固定）

| 能力 | 现状/语义 | 承载 | 动作 |
|---|---|---|---|
| exact 非空 dense F32 MatMul，M<16 serial | 同版 carrier 可运行，adapter 人为拒绝 | BORROW oneDNN | FIX 小矩阵准入 |
| M≥16 或 caller executor 分片 | 已接 BRGEMM、caller scratch、真实 wait | BORROW，现有路径 | KEEP |
| 空维度、非 plain/非 dense、非 F32、bounded shape | 本 carrier 不能兑现这些完整合同 | 已有其它合格 Provider 或明确拒绝 | KEEP 真实边界 |

审计：`onednn_provider.cc::select_route` 仅在 task_count=1/M<16 按旧 Native 成本推断拒绝。
`brgemm_internal.cc::create_brgemm_plan` 验证正维度、int64 上界和任务数，没有 M16 能力门；
实际 M1 carrier 正确性证据已保存。因此只删除推断过滤，保留全部语义、资源、workspace 检查。

| FIX | 允许改动 | 验收 |
|---|---|---|
| OM1 | 删除 `kNativeSerialRowCutoff` 与 serial M<16 冷拒绝；不改 BRGEMM 数学/plan/submit | M1/M4/M15/M16，奇数 K17/N65 和 K64/N64 实际输出、两次重放 |
| OM2 | 旧小矩阵拒绝测试改为准入；composition 回落用真实不支持的转置布局；增加 ordinary CPU 小矩阵身份测试及独立三交付 consumer | 冷 propose/prepare 同语义，真实 unsupported 不受影响，输出正确 |

原审计阶段有限 FIX 已冻结，owner 先绑定 AUDITED 再改产品；不借此更改 Native、ggml、Graph planner、路由器 API 或性能工具。

## 七问

1. 原版版本、许可证和交付由已有 oneDNN factor/lock 持有，数学源不改。
2. 小矩阵属于普通 MatMul 同一能力，不让用户选择 Provider 顺序或伪装更多行。
3. 复用原 BRGEMM plan、scratch 验证、Compute invocation 与闭合资源生命周期。
4. 冷过滤删除一条分支；没有新热 dispatch/search/计时。公平性能裁决在后续 CPU 逐域卡。
5. 无新 C/FFI 或公共私有 carrier；诊断 carrier 调用不进入用户指南。
6. ABI/组件图/package 输入不变；三交付实际小矩阵调用仍须验证。
7. 普通 `cpu::create_context()` 按固定优先级匹配，专家 oneDNN 也能自然 prepare 小矩阵。

## 验证与裁决

固定产品候选为 `65a02982ac108d57b13769029f56a6a6903895be`；OM1/OM2 已实现，公开合同、数学、
资源路径及冷准入停止变化。Windows 定向 7/7 和三交付已通过；Linux 批量 8/8、普通 Direct/Graph
与三交付消费者已通过。Linux 428 个注册 GTest 中 414 通过、14 因旧模型 fixture
未部署跳过；这些旧测试不冒充本次通过，也不将本次数值准入写成全模型或 Native 性能裁决。

本卡已先进入 CANDIDATE，再依据实际收据完成以下 terminal 裁决。
文档/状态提交不改变上述不可变产品候选，不据此重构建或重跑 P1；本卡不签性能 WIN，也不关闭 CPU 逐域目标。

## CLOSED

上述候选全部适用有限验收通过，接受 OM1/OM2；
[实际收据、失败纠正与限制](../../../../evidence/correctness/cpu-mature-admission-batch-v1/README.md)。

| 门 | 裁决 |
|---|---|
| scope | PASS：只恢复成熟 BRGEMM 已存在的小矩阵能力，未优化或裁决 Native |
| architecture | PASS：沿原 plan/workspace/执行/完成路径，无新增公共 API、kernel 或执行器 |
| correctness | PASS：Windows 定向 7/7，Linux 共同批次 8/8，小矩阵数值、两次变化输入、默认身份；未部署旧 fixture 的跳过不计 PASS |
| delivery | PASS：两平台 source/static/shared，独立普通及专家 Direct、普通 Graph，实际重复执行 |
| usability_docs | PASS：普通 CPU factory 和默认小矩阵准入文档，不要求用户排序 Provider |
| red_team | PASS：真实不支持的转置布局仍回落；原 shape/resource/workspace/失败检查相关测试通过 |
| engineering_performance | PASS/P0：删除冷过滤，原数学与常态执行成本不变；不签算法 WIN、不启动 P1 |

停止本卡施工；Native 重叠域与固定优选规则继续由逐域卡负责。重开条件为实际小矩阵准入/数值、
资源与 completion 或三交付失效，不因计时点值或旧卡命令重开。
统一脚本：Compute + Graph **61,035 实现行 / 248 文件**，不含测试与 vendor。
