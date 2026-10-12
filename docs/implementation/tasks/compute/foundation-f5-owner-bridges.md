# F5 原Provider与Graph同slot接缝

状态：SCOPED。baseline ab3d94125；沿[F5活动卡](foundation-f5-paged-ragged-batch.md)同一六FIX/
原型及产品验证预算，不因本卡重置。原F1–F4/A/R1–R7已签功能保留；只以下实际新接缝准入。

## Closure card

- owners：compute、graph、compute_provider_flashinfer。新compute_batch/graph_batch已SCOPED。
- trigger：F5原型已证明锁定paged/ragged C++ planning/run；现FlashInfer Provider只提供dense
  prepare，Graph v19 PreparedRegion/RegionInvocation真实拥有slot/workspace而无显式layout
  preparation入口。另造Direct slot规划不能证明Graph实际提交slot已有同版本metadata。
- consumer：相同host-authoritative metadata版本和readonly Q/KV，Direct原批次与Graph原region
  同slot prepare/update→原native submit→旧结果/native-use退休；主单/多请求与正交分段矩阵。
- scope：F5-BRIDGE-C（同一原Provider instance的冷named extension查找/保活与admission），
  F5-BRIDGE-G（opt-in取得原Graph region的prepared/invocation/workspace/关联节点，以便批次
  协调器在submit前准备该同一slot），F5-BRIDGE-FI（锁定原FlashInfer批次C++承载扩展）。
- 非目标：普通19版region vtable/layout/Invocation常态字段、submit隐式plan、第二executor/
  Queue/runtime、重写原dense数学/所有Provider、升级锁、新math/copy kernel、公共工具。
- 依赖方向：compute只保存具名extension owner/admission，不包含Graph或batch数学；graph
  窄private访问桥不解释F5策略，批次协调器由更高graph_batch拥有。原native数学由原
  FlashInfer owning factor实现，不让common compute_batch反依赖路线Provider造成环。
- 生命周期：compiled原ProviderExtensionRef保exact Context/provider instance；extension查找
  冷且版本精确，unsupported不伪造成功。Graph访问限idle原invocation，observer/permission
  由原owner裁决；实际batch preparation不能持旧caller host spans，不能替换另一slot。
  active/failed/closed/in-flight/old output与资源别名仍沿原Graph/F1/F2边界，绝不绕过。
- 语义/API：独立BatchPlanning extension；返回同provider instance的typed capability，不改变
  原Region19合同。规划表只保存在显式opt-in batch协调器，普通调用者不付无条件字段/
  分配/查表/每步DAG遍历。正式签名在AUDITED前固定，SCOPED不写产品。
- 原native Apple/Vulkan仍是其typed Program/Invocation及readonly native owner，显式
  reference可冷lower成原Gather/View/SDPA等既有数学；不得为了common签名偷跑CPU或新shader。
  若原route API事实要求改owner，先独立SCOPED纳入，不能本卡顺手改其CLOSED产品。
- 性能：P0，新feature opt-in；原dense/普通Invocation代码与layout成本不变。native planner
  临时vectors/occupancy/H2D/workspace/合并拓扑计账，UNKNOWN保留；不自动P1/capture资格。
- 预算/机器：F5主卡两轮owner/API审计、同configure300/build1800/test120/consumer180；
  Win/WSL/Air实际路线及package/san适用门，库j4/j2/j1、test1、重型跨机器串行。原型16.807/
  0.365秒已计同根，后续shape/split风险不重置。到达预算按主卡BLOCKED/INCONCLUSIVE。
- 预声明：原instance/foreign/default/wrong版本/closed查找拒绝；Graph同region/slot/workspace
  与错误NodeId、busy/old result/reset失效；accepted planning/upload failure及prefix/owner/alias/
  cancel/abandon/unknown/close/drain；native paged/ragged与独立oracle、Direct+Graph实际一致；
  普通dense没有新增plan/JIT路径，源码/API/新头/三交付/迁址/loader/缺路线、guide/P0/红队。
  仅新batch与真实直接消费者/必需旧接缝，不回放全算子或已签F1–F4全库。每轮docs/closure/
  diff/previous-ref，source/static/shared适用矩阵在AUDITED按真实组件图固定。
- stop：owner/lifetime/normal costs新事实退明确阶段；固定scope外新能力不加入。对F5 MUST
  未完成不能写NA；原typed runtime/capture差异如实保留，整个goal仍须完整F5与后续差额评估。

## 原范围保留

compute原F4 package-only与F1–F3机制、graph原A-01..04、FlashInfer原dense/head256/Linux资格
保留在[原状态快照](../../../../evidence/architecture/foundation-f5-owner-scope/old-scope.json)。
当前只SCOPED，未写这三owner产品，也未修改19版vtable或普通Invocation layout。
