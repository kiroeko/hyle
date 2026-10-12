# F5分页/变长批次与四路线消费收口

状态：CLOSED，2026-10-13。只关闭F5-01..06、原有限package/权限桥及IRE-TSAN-01；
完整A/F4/F5 goal的16项回映见[完成审计](../project/foundation-f45-completion-audit.md)。
不扩完整引擎/F6、模型策略、新数学/copy kernel、通用Queue/runtime或公共验证仪器。

## 固定输入与owner范围

整体固定candidate11c348e404df2c3fc404056ff3a9e885d4d990bc；F5数学/API固定25e4fb398e2a782ee60f916c3c6351c4736b6eda，
最后产品差分只有IREE内部TSan配置两行。此后canonical include/src/cmake/scripts/tests与
candidate无diff。Windows canonical单向给WSL/Air验证副本，Apple只actual Mac16,12 Air。

原[六FIX主卡](foundation-f5-paged-ragged-batch.md)、[Provider/Graph桥](foundation-f5-owner-bridges.md)、
[native reference桥](foundation-f5-native-reference-bridges.md)、[package卡](foundation-f5-package-seam.md)
及[候选卡](foundation-f5-candidate-validation.md)拥有baseline/准入/预算。原20 owner范围由
[初始scope](../../../../evidence/architecture/foundation-f5-candidate-scope/owners.json)及
[CUDA修复scope](../../../../evidence/architecture/foundation-f5-cuda-link-final-scope/owners.json)保留。
第21 owner compute_provider_iree_vulkan仅由[新ABI反证窄卡](foundation-f5-iree-tsan-contract.md)
合法SCOPED→AUDITED→CANDIDATE，仅补原IREE TSan开关；未重审该Provider旧数学。

compute_batch拥有typed容量/有效区、owned layout/version、实际页frame与opt-in Direct控制；
graph_batch拥有共同recipe/原region同slot协调。原compute/graph/provider/route权限桥各保
原owner，package-only owner只签自然产品发现/依赖/导出。compute_state只签F5组合/产品seam，
F4实现不改。A/F4/F1–F3/R1–R7原范围保留，不因新terminal重签整个算子目录。

## 六FIX与真实消费

| FIX | 实际完成及精确资格 | Verdict |
|---|---|---|
| F5-01 | 独立Paged/Ragged schema，checked capacity/active区/GQA/indptr/page/tail/empty，owned host快照与schema/generation；原F2上传同版本device镜像 | PASS |
| F5-02 | 锁定FI原C++ plan/update/run与独立slots/workspace，head64/128/256与原long split/merge；ordinary installed D256四域Direct/Graph独立oracle | PASS，仅Linux CUDA原准入域 |
| F5-03 | 显式prepare/update失效旧epoch/plan，actual H2D/prepare/run失败、suffix、unknown及原use/pending真实退休；旧token/result拒覆写，失败open slot不可复用 | PASS |
| F5-04 | opt-in coordinator访问原Graph region同一slot/workspace；完整node/layout集合/共享metadata同version，原权限与failure cleanup；普通warm无隐式plan/JIT/新字段税 | PASS |
| F5-05 | native与显式参考各自然Direct/Graph，精确页writer/read→原F2 dirty范围→F4成功发布→old/new/old；七profile72 ragged、实际第二request重排与成本 | PASS，保各route费用/资格差异 |
| F5-06 | source新头/缺API、18 installed static/shared格、actual rename/only-prefix consumer/loader、原故障/OOM/san、guide/正交/P0/LOC/分离审查、完整component合同 | PASS，按原适用facet |
| IRE-TSAN-01 | 原IREE C proactor与HYLE adapter同时TSan，修复operation结构56/64字节ABI错位，原Air Vulkan两消费者同tree/预算复验 | PASS；普通OFF条件路径不变 |

逐能力/边界/OOM/ownership/取消/并发/主正交/平台回映在
[最终风险矩阵](foundation-f5-final-risk-matrix.md)，其具体episode与命令在
[原审计](foundation-f5-closure-audit.md)的原receipt，不由测试数量代签。
共同33项/九头、caller冷OOM802点（CPU118/native684）、实际CUDA prepare/run/unknown、
Apple八queued、Vulkan16driver错误、CPU failed-ready→F5/F4不提交/不发布分别直接取证。
控制API externally serialized，immutable版本与原native回调/页权限退休的并发另验。
F2不成功不产生ready lease；consumed page writer须retain/complete，abandon按原close/recreate，
不猜ready或用readonly whole-arena假造独占页权限。

## 三交付与sanitizer范围

18格SDK为Win cpu,cuda / cpu,vulkan / vulkan，WSL cpu,cuda / vulkan，Air cpu,apple /
cpu,vulkan / apple / vulkan，各static/shared。原[总矩阵](../../../../evidence/correctness/foundation-f5-sdk-complete-matrix/README.md)
逐raw/stage/hash核验，最后IREE san-only修复按[OFF条件投影](../../../../evidence/architecture/foundation-f5-iree-tsan-impact/README.md)
保原SDK输入资格，不伪造新的run。native-only不借CPU factor/数学；Windows CUDA native batch
只签具名Unimplemented及Graph执行前拒绝，WSL Vulkan只软件包/loader，不签hardware数学。
实际供应新头独立TU，disabled路线源码九头可include且四组API具名缺席；SDK缺未选route头。

common与CPU实际Direct/Graph分别ASan/LSan、UBSan、TSan；Linux native CUDA宿主地址ASan/
UBSan/TSan数据竞争三consumer；Win Vulkan宿主MSVC ASan；Air Apple ASan/UBSan和同原
HYLE已插桩objects/native MLX DSO的TSan；Air Vulkan UBSan与修复后TSan均真实执行。
compile commands覆盖对应controller/coordinator/native import/retirement/消费者，Vulkan TSan
还核IREE C proactor flags。实际GPU数学仍成熟Provider，未用CPU替代或未插桩binary假报。

宿主san不签GPU kernel/driver/未插桩vendor整体：CUDA原cuInit shadow-gap FAIL、NVIDIA
LSan2320/8、1000/5、8920/35及诊断线程exit66、Apple静态MLX/Data两FAIL与全部no-HYLE
对照均保留。CUDA仅按原文档恢复地址门/分离thread-leak门；Apple仅分离native DSO，没有
suppression/关闭data-race检查/改exitcode。vendor whole-process仍INCONCLUSIVE，不改成语义NA。
MSVC没有UBSan/TSan/LSan，Apple实际runtime不支持LSan；相应平台已用实际故障/guard/退休/
quota证据证明本scope，不称整进程零race/零泄漏/私有GPU总量。详见[native资格卡](foundation-f5-native-host-sanitizers.md)。

## 七门、成本、终审与预算

| Gate | 直接证据 / 限制 | Verdict |
|---|---|---|
| scope | 原六FIX、21 owner合法范围，主/正交旅程和non-goals冻结 | PASS |
| architecture | 低层F1/F2复用，无Compute反依赖Graph；原Provider/slot/native owner，无第二runtime/kernel | PASS |
| correctness | 上述逐能力/风险矩阵、各适用宿主san与实际GPU权限/故障/退休 | PASS，本有限scope |
| delivery | ordinary source、18 SDK/头/缺席/迁址/loader、两实际package反证修复、原完整component脚本最终成功 | PASS |
| usability_docs | [公开guide](../../../user/batch-attention.md)、实际public consumers及[正交旅程](../../../../evidence/correctness/foundation-f5-usability-orthogonal/README.md) | PASS |
| red_team | [分离源码/API审查](../../../../evidence/correctness/foundation-f5-separated-redteam/README.md)，后续san/组件反证与修复、最终证据资格重新回映 | PASS |
| engineering_performance | [结构P0](../../../../evidence/performance/foundation-f5-cost-structure/README.md)、[普通Graph codegen](../../../../evidence/performance/foundation-f5-ordinary-graph-p0/README.md)、固定LOC；无自动P1 | PASS P0 |

P0保所有费用：layout校验/active复制O(R+P)，固定metadata清零O(M)；arena read排序去重
O(P log P)/O(P)guards，backing与cold槽有硬容量；native同完整indptr key才reuse plan，
metadata M字节H2D/原planner分配与wait显式。参考indices4NHL/mask4NL、gather K/V4NHLD，
GQA/F32/Copy/等待是额外O(KV)费用，logical总和含alias重复，不是假resident上界。
Apple/Vulkan镜像3B+378，CPU Direct11个实际stage/waits；私有GPU/driver仍UNKNOWN。
普通Graph两state声明/匹配GCC shipping对象及六submit无新增默认字段/指令；该配置P0
不外推所有编译器吞吐或capture。无timed WIN/PARITY/SOTA、engine high-performance声明。

最终[收据](../../../../evidence/correctness/foundation-f5-final/receipt.json)绑定候选、21 owner/
原证据hash、18 SDK映射、七门、component时间与[源码/头LOC](../../../../evidence/correctness/foundation-f5-final/owners-loc.json)。
全库自有native174338，21 owner完整union58672/280files；compute_batch1437/graph_batch400，
F5对F4净增2610；tracked native tests净增1862，测试登记/工具净增5，IREE CMake净增2。
不把整个owner的旧代码、证据/文档或vendor算作F5新增，更不直接从engine LOC减去2610。

保全部原FAIL/TIMEOUT/elapsed：Direct故障原180超时与原唯一追加180、SDK shared link/
owned-header、各fixture编译/配置/环境/native fatal与san诊断均按原root。component原三
FAIL累计272.140/300，用户新授权追加300，续跑TIMEOUT328.110后再有限追加600；总上限1200，原时间
仍计入。库按实际内存j4/j2/j1、consumer1、跨机器重型串行，不因candidate/目录换预算。

重开只由可复现身份/权限/寿命/UB/交付缺陷、真实consumer准入的新常用机制、相关依赖
升级或合格同合同material成本回归触发。历史TODO/benchmark点值/再看一遍不授权。
本goal完成后按[最终基础库/引擎评估](../project/foundation-post-goal-engine-effort-2026-10.md)
记录单机单/多请求高性能残留；当前共同语义/资源接缝完成不等于所有route native性能/
capture/general或真实整模型完成。此轮停止，不启动完整engine/F6。
