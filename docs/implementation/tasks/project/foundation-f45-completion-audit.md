# A/F4/F5完整goal完成核对

状态：COMPLETE，最终固定product11c348e40（F5数学/API25e，IREE仅TSan配置两行）。
以下保完整调查阶段与原未验项历史；其当前裁决由本页最终核对表和
[F5 terminal](../compute/foundation-f5-final-closure.md)更新，不删除原FAIL/限制。
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
| F5-03 | 原native prepare/run/unknown/close/failure与actual guards、各route风险/ASan；CUDA UBSan/TSan数据竞争；Air两route UBSan/TSan | HYLE-owned宿主有限facet通过；Apple native DSO资格/IREE TSAN ABI修复各有直接证据，整进程LSan/外库线程生命周期非PASS |
| F5-04 | 原v19同slot/共享metadata与页权限、六submit/字段P0 | 默认tax/已声明失败同slot及分离源码终审已证；完整聚合仍待裁决 |
| F5-05 | 七profile72 ragged、四路线paged/F2/F4、private UNKNOWN成本账 | bounded组合已证；guide正交已核，不签全平台同吞吐/capture |
| F5-06 | 18 SDK raw/hash/candidate映射、source头/缺符号、模块测试/guide/P0/LOC及分离红队，全部已声明宿主san facet | delivery矩阵/分离审查已证明；组件聚合及最终terminal回映未完成，固定实现后engine评估已补但尚非goal完成后裁决 |

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

后续有限对照：无HYLE matching-template程序只换同候选共享MLX后24个GPU cold worker
退出0。Apple原两消费者沿同root/预算、同已插桩HYLE objects只重链接vendor DSO也均
通过，宿主并发facet见
[Apple TSan](../../../../evidence/correctness/foundation-f5-native-apple-host-tsan/README.md)。
原静态MLX失败及vendor whole-process未插桩/INCONCLUSIVE保留，未用suppression。
[Air Vulkan UBSan](../../../../evidence/correctness/foundation-f5-native-vulkan-host-ubsan/README.md)
也已两consumer PASS；Vulkan TSan正在独立配置，组件合同预算问题仍待答复。

最终必须补完全部原编号/七门的真实范围裁决、reopen triggers、20 owner合法terminal与
用户要求的goal后基础库剩余缺口/单机单多请求engine自有LOC评估。现在不签CLOSED或complete。

最后Air Vulkan TSan实际wait count异常由IREE sanitizer ABI不一致精确反证。独立
[IREE配置卡](../compute/foundation-f5-iree-tsan-contract.md)合法SCOPED/AUDITED后仅同步
原TSan开关，固定11c348e40；原20个F5 candidate25e的数学/API/普通源码输入不改，当前
活动owner为21个。修复后同root/原预算两consumer全部PASS，
[原18 SDK条件影响](../../../../evidence/architecture/foundation-f5-iree-tsan-impact/README.md)
逐原configure日志证明OFF输入完全相同，保原资格不假新run。当前native宿主适用facet
已经具证；完整风险/七门回映见[最终矩阵](../compute/foundation-f5-final-risk-matrix.md)。
剩余组件合同为原272.140/300秒FAIL/INCONCLUSIVE，追加300秒问题仍待答复。最终须把
21 owner各有限范围合法terminal与最终评估一并完成，不能因为其他门已过而跳过这一项。

## 原完整目标的最终核对

| 原编号/交付 | 当前权威证明 | 最终裁决 |
|---|---|---|
| A-01 | A terminal共享静态约束/owner与各自然native/hybrid artifact，相关variant/hybrid源码F4→最终无diff | CLOSED，保原语义档 |
| A-02 | A有限exact variants/coverage/overflow/OOM/独立slots及18 SDK，表外拒绝 | CLOSED；不签虚假general |
| A-03 | A实际纯control前驱错误/跨fragment/禁止host wait拒绝与原barrier | CLOSED；未丢控制边 |
| A-04 | A合格CPU effects/footprints/全图alias与actual失败prefix，GPU未资格carrier拒绝 | CLOSED；未假报任意GPU effects |
| F4-01 | F4 schema/目录/owned snapshots/OOM、common24/三san，原State源码/公头到最终无diff | CLOSED |
| F4-02 | F4 share/pure replace/dirty原F2 clone/readonly约束，F5新组合实际精确页旅程 | CLOSED |
| F4-03 | F4全部ready/expected-base并发single winner/旧版本native保活 | CLOSED；不解释模型checkpoint/RNG |
| F4-04 | F4实际producer错误/cancel/unknown/suffix/实际退休，F5组合失败不发布另证 | CLOSED |
| F4-05 | F4五成员recurrent/conv/KV两step与资产正交四路线oracle；F5分段矩阵/多request版本闭环 | CLOSED，本有限consumer |
| F4-06 | F4 ordinary source/18 SDK/guide/P0，F5新façade/provider link及installed消费分别复验 | CLOSED |
| F5-01 | common33/九头/source四负API/owned同版本metadata，CPU三san/native source/18 SDK | CLOSED |
| F5-02 | 原FI C++12域/long split merge，public installed D256四域Direct/Graph，Win实际unsupported | CLOSED，仅原native资格 |
| F5-03 | 原actual H2D/prepare/run/suffix/unknown/open失败、F1/F2退休，CPU失败ready/F4不发布，所有声明宿主facet | CLOSED；外库未插桩/LSan/线程报告保限制 |
| F5-04 | 原region同slot/coordinator完整layout集/shared metadata、普通两state/对象/六submit P0 | CLOSED，原warm合同不变 |
| F5-05 | 四路线paged+七profile72 ragged append/F2/F4 old-new-old及非uniform/GQA/tail oracle/明确copy/wait账 | CLOSED；reference成本不冒充高性能 |
| F5-06 | 18 SDK raw/stage/hash、source/新头/迁址/loader/缺路线、适用san/guide/P0/LOC/分离审查、完整component脚本 | CLOSED |
| IRE-TSAN-01 | 实际ABI反证合法reopen、唯一两行配置FIX、同tree原两个consumer及C runtime/HYLE flags | CLOSED；原其他scope未扩大 |
| 预算/状态/命令 | component原三FAIL+一次TIMEOUT保留，用户standing授权有限续时后PASS；21合法CANDIDATE→CLOSED，previous-ref7335bd069 | PASS，原失败/累计不重置 |
| goal后缺口/工作量核算 | 最终candidate/21 owner统一LOC与[最终engine评估](foundation-post-goal-engine-effort-2026-10.md#goal收口后的最终评估) | 完成评估；不是已实现/验收整引擎 |

A/F4的原完整terminal分别在上文链接；F5六FIX/全部逐能力风险/七门/reopen/机器/预算/LOC
由最终terminal给出，原21范围与新IREE只san配置的特殊边界保存。各raw/receipt/stage
与候选映射在[最终收据](../../../../evidence/correctness/foundation-f5-final/receipt.json)实际校验；
原48份风险archive/18 SDK和后续san/组件全部原失败未改名为PASS。所有产物来自canonical
或一次性验证副本，源码未逆向覆盖，重型串行，无全算子/P1/新engine/F6。

最后component原脚本到最终成功消息，最新PASS256.047秒；累计856.297/1200，扩时分别
300及600、原总300/原耗时/所有终态保存。当前没有剩余required gate/施工项；常态
元数据/默认codegen仍P0，未伪造吞吐/capture或Windows FI数学。
21 owner只签当前有限范围；全库registry均CLOSED不等于未来基础需求永远完整。
