# F5固定实现的最终风险回映

状态：CANDIDATE审计补充，product25e4fb398。原六FIX、20 owner、平台资格和预算不变；
不是terminal record，不签CLOSED。当前授权仍由[候选卡](foundation-f5-candidate-validation.md)
及registry给出，原历史尝试由各receipt保留。

## 原六FIX与逐能力证明

| 能力族 / FIX | 正常与边界 | failure / OOM / overflow | ownership / teardown | cancel / concurrency | public / 正交 / 交付 |
|---|---|---|---|---|---|
| typed schema/owned layout，F5-01 | common33：paged/ragged、GQA/active区、错表/尾/empty/checked容量；inactive读取真实页修复反例 | 元数据OOM/饱和身份/checked乘积，预检后复制；旧版本不被失败覆盖 | schema owns/readonly F1 export，host spans不跨调用借用 | immutable version分享/生成由common三san证明；冷create不接纳异步工作，取消不能自动撤销已拥有版本 | 18 SDK公开四域；source九头/四负API；opaque分段矩阵同容量/有效区旅程 |
| native plan/update/run，F5-02/03 | Linux FI原12域、long split/merge；公开installed D256四域Direct/Graph；原同indptr key复用plan | 实际四prepare失败/H2D/run错误、suffix停止、失效旧token、open失败拒复用；caller冷OOM与原limits | 独立slot/workspace/mirror、旧result/use阻止覆写；原pending/use实际退休后drain | actual CUDA正常放弃/取消/unknown Gate与late use；宿主地址ASan/UBSan/TSan数据竞争；vendor私有成本/整进程报告另限 | 自然Linux cpu,cuda static/shared；Windows native batch具名Unimplemented与Graph预拒绝，未准入数学不算PASS |
| Graph同slot协调，F5-04 | 完整node/layout集合、shared metadata同version、原region实际slot/workspace；普通dense未新增规划 | 错NodeId/foreign/default/plan失败清token、actual region错误/suffix、open failed slot | 原leased Graph及F1/F2保持；不存在另一个native plan或可写whole-arena假alias | independent invocations和原retirement；控制对象外部串行；相应CPU/CUDA宿主三san及原实际故障 | 普通Compute/Graph消费，不补内部target；GCC字段/对象/六submit P0；其他工具链吞吐未由此证明 |
| exact-page arena与append/F4，F5-05 | same-schema/layout/backing与精确页集合，乱页/tail；七profile72 ragged与四路线paged old/new/old | 错页/foreign/OOM、F2失败不产ready writer，F5不submit/F4不发布；used页不可取得writer | 原F1独立blocks/native guards，weak ready事实避免arena/state互持；旧页/版本保活、末quota0 | consumed writer须retain/complete；abandon不伪ready，未知实际退休不猜free；close/recreate合同明示 | public页资源→原F2→F4→Direct/Graph；分段矩阵/资产集合接口不解释token/model/eviction |
| explicit CPU/Apple/Vulkan reference，F5-05/06 | 原Gather/View/Cast/MatMul/Softmax/Copy，非uniform/GQA/causal/NaN padding，same-generation两bank/双slots | caller cold OOM802总点中native684/CPU118；CPU actual error/failed-ready；Apple八queued与Vulkan16实际driver错误 | 原native Program/readonly Results/page imports，旧result/escaped owner/actual drain；不跑CPU fallback | CPU三san、Win Vulkan ASan；Air Apple ASan/UBSan与native DSO宿主TSan、Air Vulkan UBSan；Vulkan TSan待裁决 | actual native-only与CPU+GPU SDK独立供头/迁址/loader；WSL软件Vulkan只包；参考O(KV)费用不签高性能/capture |

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
| Air actual Vulkan宿主 | UBSan两个consumer实际通过；同代码模板Win ASan独立证明访问facet | TSan独立配置尚在运行；MoltenVK/driver/private vendor未插桩，不外推Windows并发 |

本轮必须证明HYLE-owned宿主和实际native权限/失败/退役，私有GPU storage/外库未插桩
边界沿原卡已声明，不扩成driver内部审计。外库报告保INCONCLUSIVE，不伪NA或整套san
PASS；有可归因HYLE源码或跨边界使用错误的证据须退AUDITED，不能以外库名称免责。

## 七门当前裁决

| Gate | 证据 | 当前裁决 |
|---|---|---|
| scope | 六FIX与20 owner合法候选，原A/F4/F1–F3/R1–R7范围保留，无模型策略/新kernel/runtime | 已证明有限实现范围；最终terminal待聚合 |
| architecture | compute_batch/state高于F1/F2且不反依赖Graph；Graph原region同slot、native原owner；package-only owners各保范围 | 已证明原固定合同；没有新接口/常态字段 |
| correctness | 上述风险、原actual native故障和逐facet san | PENDING：Air Vulkan TSan尚未裁决，不能提前PASS |
| delivery | source资格、18 installed矩阵/实际rename/loader，CUDA公共FI link反例已修复验 | PENDING：独立component全配置合同原272.140/300未PASS，追加预算答复待定 |
| usability_docs | public guide/实际隔离SDK消费者/正交映射，borrow/consumed writer/费用/unknown明确 | 已证明；不将guide存在签未运行API |
| red_team | 固定源码/入口/风险/资格分离核查，原两个文档finding已纠正；后续san反证独立保留 | 源码有限审查完成；最终新增证据资格仍需回映 |
| engineering_performance | P0成本结构/ordinary字段和codegen/独立产品测试工具LOC | 已证明P0结构；未自动P1，不签引擎吞吐或capture |

reopen triggers沿原协议：可复现身份/权限/寿命/UB/交付缺陷、真实消费者证明的缺失常用
机制、相关依赖升级或合格同合同material成本退化；历史TODO/重新看一遍/参考库多一个
feature不授权重开。本表没有改变任何owner状态或把未裁决gate变为PASS。
