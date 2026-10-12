# F5固定实现的最终风险回映

状态：最终收口风险回映，F5数学25e与IREE配置11c348e40。原六FIX、20 owner及独立IREE
窄FIX、平台资格和累计预算保持；terminal见[最终记录](foundation-f5-final-closure.md)。
原历史尝试由各receipt保留，此页不授权重开或新产品范围。

## 原六FIX与逐能力证明

| 能力族 / FIX | 正常与边界 | failure / OOM / overflow | ownership / teardown | cancel / concurrency | public / 正交 / 交付 |
|---|---|---|---|---|---|
| typed schema/owned layout，F5-01 | common33：paged/ragged、GQA/active区、错表/尾/empty/checked容量；inactive读取真实页修复反例 | 元数据OOM/饱和身份/checked乘积，预检后复制；旧版本不被失败覆盖 | schema owns/readonly F1 export，host spans不跨调用借用 | immutable version分享/生成由common三san证明；冷create不接纳异步工作，取消不能自动撤销已拥有版本 | 18 SDK公开四域；source九头/四负API；opaque分段矩阵同容量/有效区旅程 |
| native plan/update/run，F5-02/03 | Linux FI原12域、long split/merge；公开installed D256四域Direct/Graph；原同indptr key复用plan | 实际四prepare失败/H2D/run错误、suffix停止、失效旧token、open失败拒复用；caller冷OOM与原limits | 独立slot/workspace/mirror、旧result/use阻止覆写；原pending/use实际退休后drain | actual CUDA正常放弃/取消/unknown Gate与late use；宿主地址ASan/UBSan/TSan数据竞争；vendor私有成本/整进程报告另限 | 自然Linux cpu,cuda static/shared；Windows native batch具名Unimplemented与Graph预拒绝，未准入数学不算PASS |
| Graph同slot协调，F5-04 | 完整node/layout集合、shared metadata同version、原region实际slot/workspace；普通dense未新增规划 | 错NodeId/foreign/default/plan失败清token、actual region错误/suffix、open failed slot | 原leased Graph及F1/F2保持；不存在另一个native plan或可写whole-arena假alias | independent invocations和原retirement；控制对象外部串行；相应CPU/CUDA宿主三san及原实际故障 | 普通Compute/Graph消费，不补内部target；GCC字段/对象/六submit P0；其他工具链吞吐未由此证明 |
| exact-page arena与append/F4，F5-05 | same-schema/layout/backing与精确页集合，乱页/tail；七profile72 ragged与四路线paged old/new/old | 错页/foreign/OOM、F2失败不产ready writer，F5不submit/F4不发布；used页不可取得writer | 原F1独立blocks/native guards，weak ready事实避免arena/state互持；旧页/版本保活、末quota0 | consumed writer须retain/complete；abandon不伪ready，未知实际退休不猜free；close/recreate合同明示 | public页资源→原F2→F4→Direct/Graph；分段矩阵/资产集合接口不解释token/model/eviction |
| explicit CPU/Apple/Vulkan reference，F5-05/06 | 原Gather/View/Cast/MatMul/Softmax/Copy，非uniform/GQA/causal/NaN padding，same-generation两bank/双slots | caller cold OOM802总点中native684/CPU118；CPU actual error/failed-ready；Apple八queued与Vulkan16实际driver错误 | 原native Program/readonly Results/page imports，旧result/escaped owner/actual drain；不跑CPU fallback | CPU三san、Win Vulkan ASan；Air Apple ASan/UBSan与native DSO宿主TSan、Air Vulkan UBSan/修复后TSan | actual native-only与CPU+GPU SDK独立供头/迁址/loader；WSL软件Vulkan只包；参考O(KV)费用不签高性能/capture |

该表引用的具体命令、输入hash、失败预算和实际episode在
[原收口审计](foundation-f5-closure-audit.md)、[native sanitizer卡](foundation-f5-native-host-sanitizers.md)
及[18格矩阵](../../../../evidence/correctness/foundation-f5-sdk-complete-matrix/README.md)逐项链接。
数字不是通过率，不由一个happy-path或一台机器代签另一个风险。语义边界：所有controller
控制API为thread-compatible、由外部串行；新native回调/use退休实际跨线程，已单独验而非NA。
F2没有成功输出时不存在可绑定F5的ready lease，实际failed-ready旅程证明不提交/不发布，
不是把producer全类删为NA。GPU权限依实际原native/use/故障证明，宿主san不查看kernel。

## Sanitizer精确资格和限制

| facet | 当前可依赖结论 | 不由此得出的结论 |
|---|---|---|
| common ownership/layout并发及CPU实际Direct/Graph | ASan/LSan、UBSan、TSan分别实际通过 | 不代签任何原生GPU/driver |
| Linux CUDA新宿主callback/retirement | 地址ASan、UBSan、TSan数据竞争三consumer实际通过 | NVIDIA库LSan与诊断线程生命周期仍INCONCLUSIVE；cuRAND对照差异保留，不称整进程零泄漏 |
| Win actual Vulkan宿主 | MSVC ASan两个consumer实际通过 | MSVC没有UBSan/TSan/LSan；不可把不支持写成语义NA，不代签driver |
| Air Apple宿主 | ASan/UBSan实际通过；原已插桩HYLE objects仅vendor MLX DSO分离后两个TSan consumer通过 | 原静态coalescing/GCD/MLX Data报告由无HYLE对照复现但仍保存FAIL；vendor未插桩整体INCONCLUSIVE，Apple LSan runtime不支持 |
| Air actual Vulkan宿主 | UBSan与IREE配置修复后TSan两个consumer实际通过；Win ASan独立证明访问facet | MoltenVK/driver/private vendor未插桩，不外推Windows并发；原IREE ABI失败保留 |

本轮必须证明HYLE-owned宿主和实际native权限/失败/退役，私有GPU storage/外库未插桩
边界沿原卡已声明，不扩成driver内部审计。外库报告保INCONCLUSIVE，不伪NA或整套san
PASS；有可归因HYLE源码或跨边界使用错误的证据须退AUDITED，不能以外库名称免责。

## 七门当前裁决

| Gate | 证据 | 当前裁决 |
|---|---|---|
| scope | 六FIX、21 owner各合法有限范围，原A/F4/F1–F3/R1–R7保留，无模型策略/新kernel/runtime | PASS |
| architecture | compute_batch/state高于F1/F2且不反依赖Graph；Graph原region同slot、native原owner；package-only各保范围 | PASS |
| correctness | 上述风险、实际native故障/各宿主san；IREE ABI反证依法两行修复后两consumer通过，vendor限制保留 | PASS，本scope |
| delivery | source、18 SDK/rename/loader，CUDA public FI link修复、IREE OFF条件映射；完整component脚本最终通过 | PASS |
| usability_docs | public guide/实际隔离SDK消费者/正交映射，borrow/consumed writer/费用/unknown明确 | PASS |
| red_team | 固定源码/入口/风险分离核查、文档finding纠正；后续反证/全部最终raw资格与49文件hash再核 | PASS |
| engineering_performance | P0结构/ordinary字段与codegen/固定产品测试工具LOC，不签引擎吞吐或capture | PASS P0 |

reopen triggers沿原协议：可复现身份/权限/寿命/UB/交付缺陷、真实消费者证明的缺失常用
机制、相关依赖升级或合格同合同material成本退化；历史TODO/重新看一遍/参考库多一个
feature不授权重开。本表没有改变任何owner状态或把未裁决gate变为PASS。

后续Air Vulkan TSan的原C runtime/adapter配置ABI反证已由独立
[IREE窄FIX](foundation-f5-iree-tsan-contract.md)修复并实际复验通过；旧表中的“尚在运行”是
该阶段事实。21个活动owner为原20个F5 + 仅TSan配置的IREE owner，各候选范围明确保留。
普通18 SDK的[条件影响映射](../../../../evidence/architecture/foundation-f5-iree-tsan-impact/README.md)
证明OFF产品资格不变；组件聚合仍待原预算答复，不以san完成代签交付工具门。

用户2026-10-13授权类似验证直接有限续时；component原三FAIL、追加300后的TIMEOUT均
计入，再追加600后同原脚本完整PASS256.047秒，累计856.297/1200。
[完整组件证据](../../../../evidence/correctness/foundation-f5-component-facade-correction/README.md)
与[最终聚合收据](../../../../evidence/correctness/foundation-f5-final/receipt.json)给出当前裁决，
原FAIL/限制作历史保留，不把缺少native/全模型/P1工作纳入本goal完成量。
