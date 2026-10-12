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

本轮实现反证确认F1 retain_resource_owner位于整个LeaseDomain，不是单block：把仍持该pool
root的owner挂回domain会成环。F5固定块集合因此复用ReadLease::adopt已有的冷root释放语义：
新增private LeaseAccess::detach_pool_root，仅在全部固定writer块已分配且真实resource owner
已接入后，放弃allocation root而不关闭已分配lease域。每个block/native use保domain，domain
保不回指pool的实际provider/resource owner，最后引用释放。常规LeasePool析构/close、所有
原F1–F4 public行为不变，不新增常态字段或fence。此窄C-02是同F5 owner生命周期FIX，仍
AUDITED；必须追加escaped output/native admission及allocator归零反证，不以结果能读掩盖环。

## AUDITED 首批执行事实

[真实HYLE native Direct](../../../../evidence/correctness/foundation-f5-native-direct-first/README.md)
四个域（paged/ragged各F16 D64短KV、BF16 D128长KV）各三layout episode对独立oracle通过。
schema→owned host layout→原Context pinned staging→原F2同slot device mirror→原FlashInfer
planning/int H2D→run/实际完成→原F1 readonly output。长KV实际split-KV/原merge，metadata
staging bytes/上传bytes/host wait单列，原planner临时vectors/occupancy/private UNKNOWN不隐去。

显式update_batch只有Provider证明完整native key（同schema、query/KV indptr和固定spec）
不变时复用；page IDs/tails不参与原scheduler拓扑但更新镜像/版本。indptr变更在update中
重新plan，run不重规划。原native slot保owned layout，token/报告/镜像generation一致。
旧output禁止覆写，释放invocation/Program后结果仍接原F1 native read-use，escaped output
仍占冷slot容量；实际resource/domain最后释放才归还额度，metadata quota最终0。

[Win共同16项/四头](../../../../evidence/correctness/foundation-f5-common-execution/README.md)
同时通过，新增fixed-block root转交原F1生存规则反证。保留原两次Direct源资源拒绝、
update lambda编译FAIL/原输入历史与累计预算；原typed host/source合同修正用pinned staging，
没有放宽旧CUDA validate/copy算法。只此新shell/原FI/真实直接消费者，不回放算子全集。
产品仍AUDITED；Graph同slot、四路线显式reference/append闭环、完整故障/SDK/san/P0/终审
尚未完成，不能CANDIDATE/CLOSED，也不能据四个Direct域代签整个F5或单机吞吐。

## AUDITED G-PAGE：原Graph精确页权限绑定

baseline 9d9ab550b。ARENA-03 Direct已证明整块容量视图与actual page guards必须分立；
原LeasedInvocation::bind(ReadLease)要求完整TensorDesc覆盖，不能承载这项真实消费者。
graph/graph_batch沿本卡AUDITED有限FIX增加G-PAGE，不重开F1/F4产品、不改v19或普通对象字段。

- graph private RegionAccess冷create显式预留额外readonly guards容量。沿原bindings容器
  保存实际F1 page guards，尾部guard-only项用无效ValueId表示，不参与公开binding查找。
  原submit不增加分支/第二executor：同一原LeaseUseBatch读取所有项，原prefix retirement
  统一保活。reset冷移除guard-only项；普通调用无额外项、无扩容及use容量变化。
- private projected bind仅input、不可直接published、不可有physical alias descendant；
  扫描实际physical plan所有读者，必须是调用者给定exact ProviderId下的selected region
  step且在给定allow-list中。普通operation/copy/其他region或无读者拒绝。再沿原Invocation
  bind检查描述/resource/alias；实际page guards必须同resource/native handle且在容量范围。
  trusted graph_batch负责从不可伪造PagedArenaRead取得容量/guards，并核每个消费slot的
  K/V端口、same-schema与完全相同layout版本；query/metadata或混合layout不能借此绑定。
- graph_batch Slot保immutable K/V frames到原Invocation/Completion/result退休；shared_submit
  核实际输入storage精确身份，prevent raw容量或同slot版本替换。prepare只有原控制门通过
  后清frames/guards/token，部分失败不能提交混合generation。一次prepare内projected端口
  不接受覆绑；prepare/reset是显式更新边界，ordinary dense路径不自动升级资格。
- 验证：原native Graph同树/累计预算新增F16/BF16实际页append→F4发布→旧新旧oracle，
  used-page/free-page分离；错误schema/layout/port/公开whole-arena输出/非batch读者拒绝，
  旧result阻止更新、close/escaped result/最后metadata归零。Win真实owner/WX与新头检查。
  源码/SDK/san及其他全F5风险继续按主卡，不能用此checkpoint代签整个F5。

G-PAGE已实现，见[逐页Graph收据](../../../../evidence/correctness/foundation-f5-graph-page-first/README.md)。
实际CUDA Graph F16/BF16旧→新→旧六oracle、原两域双节点12oracle、CPU实际Copy/
alias/published容量三拒绝、原result/close/escaped result/最后quota0通过。原F2 clone1024+
append512bytes与F4脏成员/同版本metadata成功发布，whole arena4096未复制。Win原产品
source/static消费、modified Graph owner TU/WX和两个独立Graph头通过。新test错误View
构造器编译FAIL保留，按原slice工厂修正。六FIX及四路线reference/完整控制器/ragged/
其他风险/SDK/san/P0/guide/终审未缩减，仍AUDITED；本次不签candidate/完整F5。

## AUDITED Graph opt-in同slot冻结

graph与graph_batch进入AUDITED。Graph只增加cold private region访问桥：Executable访问实际
编译region identity，Invocation/LeasedInvocation访问同一实际opaque slot，保留原busy/failed/
old-result/closed裁决；不改v19 table、普通对象layout或submit路径。

graph_batch提供具名batch region Provider wrapper，借原Provider的native BatchPlanning
extension和原backend session，不起第二Context/Queue/runtime。cold region只承载已定义的
paged/ragged pure node，输入/输出原IR身份与control ordering不删；原native资格不满足就拒绝。
原region invocation持BatchInvocation及PreparedBatch。opt-in协调器冷create按实际provider
instance初始化它，prepare/update前通过原Graph控制门检查，再给该同一slot上传/规划及绑定
其真实metadata mirror；原Graph执行器shared_submit直接把原native kernel写入Graph已分配
output，没有shadow result copy或另造executor。原Graph Completion保整个region及私有资源
到真实退休，lease-aware外部输入/输出沿原LeasedInvocation跟踪权限/别名。

冷immutable capacity/kernel与每slot可变plan分立。GraphBatch的compiled region可在显式cold
invocation初始化时建立一次共享native BatchProgram，受本region冷monitor保护；不在run做
JIT/规划/动态bucket或全图扫描。显式opt-in元数据/初始化状态只在新batch factor，普通Graph
节点/Invocation无新增字段。private workspace/原plannerUNKNOWN如实报告，capture=false。
同schema、layout generation、原region/slot、mirror身份和队列均检查；错误原instance与旧
结果挡住prepare/update，原Graph失败prefix退休后不能假恢复，必须重新创建原invocation。

graph_batch依赖graph/compute_batch/compute/error/mem及private core/util，不让Compute
依赖Graph；原Provider wrapper的C++配置有独立owner与allocator，底层base Provider可拷贝
但不借caller局部指针。Native Apple/Vulkan后续explicit reference仍走其原typed Program与
既有math，不能借此CUDA shared接口代签它们。当前有限G-01访问桥/G-02原region Provider与
coordinator是F5-04分解，六FIX/四路线/append闭环及全部交付风险未缩减。

## AUDITED Graph首批实际同slot消费

[原Graph native旅程](../../../../evidence/correctness/foundation-f5-native-graph-first/README.md)
实际通过：paged F16 D64短KV、ragged BF16 D128长KV，各两个串联节点共享一个IR metadata
输入、两个独立invocation/计划/工作区；初始双slot与拓扑改变后共12次独立dense oracle。
原region shared_submit运行原FlashInfer kernel并直接写原Graph外部F1输出，未建立第二
executor/shadow输出。旧结果阻止prepare，第三slot受冷容量拒绝，metadata quota最后归零。

同一个metadata ValueId必须对应完全相同的owned layout版本与device资源域，完整集合
预检后才reset。每region仍有自身镜像/规划/上传且报告其真实费用；cold bridge持leader
readonly mirror供同输入消费，绑定不会被后节点覆盖成另一generation。更新先释放所有
shared readonly引用再重新取writer；部分准备失败清掉全部token/bindings，不能提交混合
批次。重复node/同输入不同版本/caller metadata注入明确拒绝。metadata必须是显式Graph
input，constant/device-produced页表未准入。普通v19/Invocation layout/submit合同不变。

证据仍是AUDITED有限checkpoint，未签candidate/全F5。其余故障/取消/unknown、四路线
reference及append/F4完整闭环、三交付/san/P0/guide/终审继续，Windows CUDA native paged
仍未准入，不拿此WSL GPU结果代签其他route。

[Windows首批编译](../../../../evidence/correctness/foundation-f5-graph-windows-first/README.md)
新batch /WX、两Graph头普通TU与自然产品链接/运行通过；原MSVC move-only token ternary
编译FAIL保留，显式if/else修正后沿同树/预算成功。[修正后定向复核](../../../../evidence/correctness/foundation-f5-graph-rechecks/README.md)
补Win16项/五compute头独立TU及WSL同两域双节点12oracle再次PASS。Windows没有native
paged数学资格；新Graph头独立TU/全部SDK格仍在F5-06。所有产品保持AUDITED。
