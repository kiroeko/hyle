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

## AUDITED Direct batch与原Provider冻结

compute与compute_provider_flashinfer进入AUDITED；graph仍SCOPED，未授权改Graph产品。
compute_batch沿已AUDITED合同增加以下opt-in执行shell，数学和native slot仍由原Provider拥有。

- BatchProgram::prepare(Context,schema,options)只在冷路径按原Provider binding查精确版本的
  BatchPlanning extension；原instance owner由ProviderExtensionRef保活。同实例冷query的
  overload不改普通submit/Provider vtable，Compute不含Graph/batch数学。
- BatchProgramOptions冷固定device/host workspace资源、float/int工作区硬上限、最大独立
  invocation数、scale/causal、allocator/ResourceBudget。每invocation有自己的原native plan、
  F1镜像/输出/工作区和F2上传槽；不共享可变FlashInfer plan，不建自动bucket/global cache。
- BatchInvocation::create(program,borrowedQueue)创建实际原native slot。prepare_batch(layout)
  要same-schema、没有旧ticket/结果/native-use；版本预检后才失效旧epoch，沿原F2上传整个
  owned snapshot到本slot镜像，再经原Queue提交原native planning及int H2D并显式等实际退休。
  返回move-only PreparedBatch，持本slot/epoch/layout；错误已接受prefix不复用旧token。
- PreparedBatch::run只用既有plan、本slot/mirror generation与readonly Q/K/V，沿原named
  extension/QueueSequence提交原kernel，无submit隐式plan/JIT/Provider搜索。原Completion、
  LeaseUseBatch保活payload/mirror/所有workspaces，取消只停发布，不假称GPU抢占/rollback。
  输出成功实际退休后经原WriteLease.publish接成ReadLease；旧结果使同slot写许可不可重获。
- Native planning report给实际float/int使用、layout generation、split-KV/CTA/native拓扑和
  copy/host等待事实；原scheduler临时vectors/occupancy/private bytes仍UNKNOWN，capture=false。
  未证明capture不因此卡变成NA/PASS；只有本F5第一档managed preparation资格。
- 原FlashInfer native domain先F16/BF16、NHD、head64/128/256、checked GQA/right-aligned causal；
  原12场景包括乱页/尾与split-KV合并对oracle。其他domain明确decline，不升级锁/新kernel。
  native callback只在原Context/Queue admission下运行，C++17边界保存原PlanInfo/模板kernel，
  C++23 owning adapter转换Status/预算并保持原dense路径不变。
- 同一typed BatchInvocation供后续Graph原region create_invocation拥有；Graph规划必须操作
  该同一对象，不能规划Direct A后提交Graph B。新Graph协调器仍需其AUDITED再实现。

三项有限FIX：C-01冷同instance query；FI-01原native prepare/plan/run+report；COMMON-EXEC-01
opt-in原F1/F2/Completion shell、epochs/old outputs/close/cancel。它们是F5-02/03/04的实现分解，
F5全部六FIX与四路线明确reference/append闭环/三交付未缩减。native Direct不是Graph/全F5证书。
两纵切仅输入/布局不同；owner/alias/metadata版本/硬容量/生命周期相同，无默认Tensor字段税。
