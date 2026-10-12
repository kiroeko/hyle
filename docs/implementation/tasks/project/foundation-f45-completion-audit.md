# A/F4/F5完整goal完成核对

状态：COMPLETION_UNPROVEN，固定product25e；检查阶段记录，不改变原目标或授权新产品。
以[设计](../../../architecture/compute-state-paged-design.md)、[固定peer研究](foundation-f45-reference-research-2026-10.md)
和A/F4/F5各原有限ledger为完成定义。未完成/间接证据不能计为完成。

| 原编号 | 直接权威证据 | 当前裁决 |
|---|---|---|
| A-01 | A terminal/shared exact/sanitizer，原shape/identity/View/guards | 原有限范围CLOSED；F4→25e相关variant/三hybrid source无diff，不重开 |
| A-02 | A finite native/hybrid factories、CPUCUDA family配额/旧结果及18 SDK | 原有限范围CLOSED；不签任意general/相同native code |
| A-03 | A actual native前驱错误/跨fragment/hostwait拒绝/原barrier | 原有限范围CLOSED；F5新pure节点仍沿原共同compiler，终审核回映 |
| A-04 | A qualified CPU effects/footprints/whole-graph alias/失败prefix | 原有限范围CLOSED；未知GPU mutable/effect carrier拒绝保持，不假报对称 |
| F4-01 | F4 terminal/common24/三san：schema/目录/owned snapshot/OOM | 原范围CLOSED；29d3→25e State源码/三个公头无diff |
| F4-02 | F4 share/dirty完整成员F2 clone、readonly替换、原资源拒绝 | 原范围CLOSED；F5组合另有真实append/read证据 |
| F4-03 | F4全部ready/expected-base/并发single winner/旧native版本 | 原范围CLOSED；不签业务token/RNG/任意rollback |
| F4-04 | F4 actual native error/cancel/unknown/suffix/retirement | 原范围CLOSED；F5新增组合不能借此代签 |
| F4-05 | 五成员recurrent/conv/KV两step、正交资产集合、四route oracle | 原范围CLOSED；完整真实模型engine尚未实现 |
| F4-06 | F4普通Compute façade/source/18 SDK/guide/P0 | 原范围CLOSED；F5新package公开FI link另有反证修复与复验 |
| F5-01 | common33/CPU三san/native同版本、九disabled头/四负API、18 SDK | 实现/多数门已证明，native剩余san和最终风险裁决未完整 |
| F5-02 | 原FI12域/long split、公开installed D256四域Direct/Graph | finite native资格已证明；仅Linux CUDA/SM89，Windows具名unsupported |
| F5-03 | 原native prepare/run/unknown/close/failure与actual guards、各route风险/ASan；CUDA UBSan/TSan数据竞争 | Linux宿主有限facet通过；Air Apple/Vulkan UBSan/TSan补验中，整进程LSan/外库线程生命周期非PASS |
| F5-04 | 原v19同slot/共享metadata与页权限、六submit/字段P0 | 默认tax/已声明失败同slot及分离源码终审已证；完整聚合仍待裁决 |
| F5-05 | 七profile72 ragged、四路线paged/F2/F4、private UNKNOWN成本账 | bounded组合已证；guide正交已核，不签全平台同吞吐/capture |
| F5-06 | 18 SDK raw/hash/candidate映射、source头/缺符号、模块测试/guide/P0/LOC及分离红队 | delivery矩阵/分离审查已证明；剩余Air san、组件聚合及最终terminal回映未完成，固定实现后engine评估已补但尚非goal完成后裁决 |

权威旧范围分别为[A terminal](../graph/foundation-a-final-closure.md)、
[F4 terminal](../compute/foundation-f4-final-closure.md)。F5当前状态仍唯一由registry/活动
[候选卡](../compute/foundation-f5-candidate-validation.md)给出；此表不把任何未验gate改成PASS。
原F1–F3/R1–R7不因项目审计重跑。18 SDK逐stage/hash/候选映射校验见
[完整矩阵](../../../../evidence/correctness/foundation-f5-sdk-complete-matrix/README.md)。

## 分离审查的当前发现

本阶段只读固定candidate与原证据，未实现新产品。已独立核对schema/generation饱和前
拒绝、slot epoch/old token与output gating、prepare清旧plan、Graph完整node/layout集与
共享metadata同一version、页frame准确权限、failure清全部token、原actual use/drain。
未在这些固定合同发现需改产品的新反证；这不等于全红队完成，余下入口/证据/平台限制
必须核齐。调用只读原签范围，不凭“再看一遍”重开closed模块。

随后已完成本轮分离源码/API/证据边界审查：
[有限结论](../../../../evidence/correctness/foundation-f5-separated-redteam/README.md)。没有新产品
缺陷，两个文档阶段/失效恢复遗漏已补；该审查不替代尚未运行的Tsan/组件聚合或最终评估。

Linux native UBSan与TSan宿主数据竞争三消费者现已实际通过：
[UBSan](../../../../evidence/correctness/foundation-f5-native-cuda-host-ubsan/README.md)、
[TSan数据竞争](../../../../evidence/correctness/foundation-f5-native-cuda-host-tsan/README.md)。
TSan原退出66为NVIDIA诊断线程finished/not joined，无HYLE生命周期对照精确复现；原FAIL
保留，线程生命周期facet仍INCONCLUSIVE。数据竞争门没有suppression或改exitcode。

明确未完成：全平台sanitizer facet最终适用性/范围裁决；整进程CUDA LSan仍NVIDIA库报告且无抑制，
HYLE-owned地址/权限与quota证据不冒充GPU/driver总量；公共component all聚合因为环境/
原剩27.860秒预算未PASS，追加300秒问题待用户答复。相同阻因没有导致当前impasse，
因为san和终审仍有可推进动作，goal保持active。

随后Air Apple UBSan已实际两consumer PASS；Apple TSan公开consumer两次FAIL/-6，
报告MLX Data deleter的GCD读/冷线程写。无HYLE匹配模板对照复现同报告，简单两个对照
未复现的差异也保留，实际vendor Darwin USE_TSAN不支持；不能直接定为假阳性或签PASS。
[完整诊断](../../../../evidence/correctness/foundation-f5-native-apple-host-tsan-diagnostic/README.md)
保持INCONCLUSIVE_MLX_METAL_HOST_RACE_REPORT。Air Vulkan UBSan正在原有限配置构建，
其TSan尚未启动。产品25e未改、20 owner仍CANDIDATE，无新产品缺陷归因或虚假terminal。
固定实现后的剩余基础库工作量和engine估计已补入
[评估](foundation-post-goal-engine-effort-2026-10.md#2026-10-13固定实现后的核算)，但完整goal
后验收与terminal仍待上述门真实裁决，不能以评估写完代替产品完成。

最终必须补完全部原编号/七门的真实范围裁决、reopen triggers、20 owner合法terminal与
用户要求的goal后基础库剩余缺口/单机单多请求engine自有LOC评估。现在不签CLOSED或complete。
