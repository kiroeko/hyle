# R3 CPU：具名性能例外与终局验收

## Closure card

- 状态：`CLOSED`；baseline `42ae3005899cbd6117494b2a609afae6133eae6b`。
- 恢复 owner：`graph`、`graph_route_cpu`、`graph_provider_onednn`。原
  [Graph 停止卡](cpu-graph-replay-cost-audit.md) 的用户裁决恢复条件已满足。
- 同步收口：原 [R3 卡](cpu-v1-four-route-foundation.md) 的 `compute`、`compute_backend_cpu`、
  `compute_provider_native`、`compute_provider_onednn`、`compute_route_cpu`，其候选与验收证据不变。
- trigger：2026-09-16 用户针对已报告的唯一未决项，明确选择：
  **“记录验收例外，继续 R4（推荐）”**。授权只限此处已量化的 CPU 小片段，不是未来性能门的普遍豁免。
- scope：登记这一验收取舍、核对已有证据/候选、按协议结束 R3，继续已启动的 R4。
- non-goals：无产品/API/规则/算法修改，无性能重测，无追加样本，无放宽统计门，无新优化队列。
- 预算：本轮只做有限记录及证据身份核对；0 次构建、0 次 P1。范围外新反例另按原 owner 处理。
- 真实 consumer、依赖、能力表、七问、成本与普通 C++ 形状：完整继承原 R3 卡及 Graph 修正卡；
  这次不另造 consumer 或 benchmark。GPU backend 已在独立 R4 卡 `AUDITED`，不在此关闭它的新范围。

## 有限 FIX 与验收取舍

| ID | 动作 | 验收 |
|---|---|---|
| A1 | 登记用户批准的唯一 decode-fused 小片段验收例外 | 原始 CI、5% 门、LOSS/INCONCLUSIVE 保持原样 |
| A2 | 核对候选与既有 receipts，结束 R3 状态/性能账本 | 不重建、不重测，不把未运行项记 PASS，不关闭 R4 owner |

唯一例外：`graph-cpu-route-p1-v3` 的 `decode-fused-boundary-1x256x512`，Graph 对比同一 oneDNN
BRGEMM + HYLE SIMD 的直接融合 FFN。约 12 微秒，paired median ratio 为 0.966131–0.974263，
即约 2.6%–3.5% 包装耗时；一个 stratum 的 98.9326% CI 为
`[0.9517370300377059, 0.9726927370864213]`，原门仍为 `1/1.05=0.9523809523809523`。
其约 5.07% 的退化上界未能满足原 5% 非劣门；**统计结果仍是 INCONCLUSIVE，严格方向仍是 LOSS**。

用户接受这项已量化的小片段限制，允许以 `ACCEPT_TRADEOFF` 完成 R3 工程验收。其依据是完整 CPU
正常入口、图/片段生命周期、资源规划与重复执行已经实现，Direct 全部四项和 Graph 其余十一项
非劣门已通过，继续为该小片段追加采样的既定预算已用尽。不得称为 parity、全部性能 PASS 或 SOTA。
其余三个 Graph 严格 LOSS 均已通过原非劣门，继续明确保留方向；用户例外只改变上述一个未决项的
工程处置，不修改另外十一项统计裁决。

## Candidate 与复用证据

| owner | 不变产品候选 | 适用结果 |
|---|---|---|
| Compute / CPU backend / Native / oneDNN | `79e96cebeb176c48d66f6b6a156fa9d910b89d82` | R3 正常入口接入与交付范围；相应产品路径到当前树无后续差异 |
| CPU Direct route | `d9712e68d2b812449a170f474d71a347a27b0ba1` | Direct 工程 P1 4/4 非劣 PASS |
| Graph / CPU Graph route / Graph oneDNN | `1a3a8d0e40a2a17250119369331c712718a4d880` | replay 修正与全部普通门；Graph P1 11/12 非劣 PASS + 1 个用户接受的未决项 |

既有 receipts：

- [R3 普通验收](../../../../evidence/correctness/compute-cpu-r3-foundation/README.md)：Windows/WSL
  source 各 23/23，Air/Mini 各 19/19；四环境 relocated static/shared 的 Direct/Graph/Sched 各 3/3；
  WSL ASan/UBSan/TSan 各 3/3；CPU OFF 与 oneDNN OFF 的实际构建/消费隔离。
- [Graph 修正普通验收](../../../../evidence/correctness/graph-cpu-replay-cost/README.md)：Windows/WSL
  source 各 7/7，Air/Mini 各 6/6；四环境 relocated static/shared Graph/Sched 各 2/2；
  WSL ASan/UBSan/TSan 各 5/5。外部 oneDNN JIT 未插桩，不扩大 sanitizer 声明。
- [Direct P1](../../../../evidence/performance/compute-cpu-route-p1-v3-d9712e68-w1-win/README.md)：
  engineering、资格/controls/deep verify PASS，n=15；1 WIN / 3 INCONCLUSIVE，4/4 非劣 PASS。
- [Graph P1 v3](../../../../evidence/performance/graph-cpu-route-p1-v3-7fa6dc0a-w1-win/README.md)：
  engineering、资格/controls/deep verify PASS，n=31；4 LOSS / 2 WIN / 6 INCONCLUSIVE。
  实际 runner source `7fa6dc0a` 与产品 `1a3a8d0e` 的产品投影相同；原始证据不覆写。

本轮仅核对上述已经运行的证据及产品路径；未启动任何新 correctness、交付或性能运行。

## 七门与最终处置

| Gate | 工程验收依据 / 限制 | Verdict |
|---|---|---|
| scope | 原 R3 C41–C46、Graph RC1/RC2 完成；本轮只处理 A1/A2 | PASS |
| architecture | 原生 CPU/Provider region、cold binding、真实资源与完成 owner；源码复核已完成 | 继承 PASS |
| correctness | 四环境、真实 FFN/Sched、两槽/变化输入、关闭/OOM/失败与适用 sanitizer | 继承 PASS |
| delivery | source/static/shared、搬迁消费、CPU/oneDNN 裁剪与一致默认选择 | 继承 PASS |
| usability_docs | 普通用户不排列 Provider、不交 timing profile；寿命与能力限制明确 | 继承 PASS |
| red_team | 已完成 cold proof、Queue owner 替换、异步/accepted-prefix 寿命与候选源码复核 | 继承 PASS |
| engineering_performance | Direct 4/4；Graph 11/12；唯一未决项由用户批准 `ACCEPT_TRADEOFF` | 工程接受；统计聚合仍 INCONCLUSIVE |

停止条件：本轮状态流与账本同步完成后关闭 R3，不追加 CPU 性能工作。reopen 仅限新真实 consumer
证明该已接受开销不可接受、受支持域 material 回归、错误/寿命缺陷、相关依赖升级或公开合同变化；
不能因重复查看同一 CI、重新计数 LOC 或目标百分比而重开。

当前全实现 LOC：Compute + Graph **31,034**，不含测试；统一脚本 self-check PASS（`42ae3005`）。
其中包含既有资产和 R4 刚开始的私有 context bridge，不能将它们当作 R3 新增算子或完成百分比。
基础架构 goal 仍 ACTIVE；R4 CUDA、R5 Apple、R6 Vulkan、R7 终审均尚未完成。

## 审计固定

A1/A2 已冻结。逐路径 Git 对比确认三个候选组的当前产品 bytes 均未变化；原始资格、普通验收与
用户裁决记录齐全。没有新产品 FIX；只继续相同候选的状态流。身份对比存于本轮 validation receipt。

## Terminal

A1/A2 完成，八个 CPU/共同层/Graph owner 结案。Graph 的恢复状态流为 `41878390` SCOPED →
`ad4d5cff` AUDITED → `a437dc28` 绑定不变产品 CANDIDATE → 本次 CLOSED。
[验收记录及产品身份核对](../../../../evidence/correctness/cpu-r3-acceptance/README.md)保留用户原话、
三个候选组的逐路径一致性和统一 LOC。机器性能账本纳入完整的四项 Direct、十二项 Graph 与 P0，
原始 margin、CI、方向及原始包不变；唯一工程例外不改写为 5% 非劣 PASS。

本次只运行 selector、带 previous-ref 的 closure、docs、scorecard 和 diff 检查，没有构建或 P1。
CUDA backend 的 R3 交付范围已经满足，但其当前 R4 C6 范围仍为 AUDITED，由 R4 卡继续承担。
R3 到此停止；基础架构 goal 继续 R4–R7。
