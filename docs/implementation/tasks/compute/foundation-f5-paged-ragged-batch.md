# F5 paged/ragged 与显式批次规划

## Closure card

- 状态：AUDITED，按下文各有限冻结段实施；尚未candidate/最终资格。新compute_batch与
  graph_batch已在0cc357684以QUEUED登记；本卡绑定同一F5六FIX，不缩成单次read。
- baseline：db8861394（F4已CLOSED，candidate29d3c7a3e）；A/F1–F4与R1–R7原范围不重开。
- trigger：用户活动goal、[固定peer研究](../project/foundation-f45-reference-research-2026-10.md)、
  [F5设计](../../../architecture/compute-state-paged-design.md)，原dense FlashInfer没有真实batch
  page端口、原普通Graph Invocation没有per-layout preparation接缝。
- consumer：单机单/多请求的packed Q、显式paged/ragged KV→host-authoritative layout版本→
  原成熟native prepare/update/run→独立slot/输出保活；append/读取→F4 branch/成功发布。
  正交纵切为不含token/模型策略的分段矩阵/资源批次，相同capacity/有效区/metadata lifetime。
- owners：compute_batch负责typed容量/有效区/schema、owned metadata版本和批次生命周期；
  graph_batch负责opt-in规划协调，绑定原Graph region的同一invocation/workspace/token。
  Compute不得反依赖Graph；普通19版region/Invocation无新增无条件字段或warm规划。
  原Provider若确需增BatchPlanning extension、Graph若需窄private访问桥，必须独立把其
  实际owner按本trigger先SCOPED/AUDITED，不凭本卡改CLOSED产品。路线资源/执行仍原owner。
- 承载：Linux CUDA先借锁定FlashInfer0.6.18/69ff11fc原C++页结构/scheduler/kernel，不带
  Python/Torch runtime或SGLang私有猴补。CPU/Apple/Vulkan显式选择原native gather/materialize
  参考档，单独计copy/KV/mask/workspace/host wait，不叫同吞吐native paged，不偷跑CPU。
  Windows CUDA native paged当前未准入，保持具名pending/unsupported，不拿WSL代签。
- 固定语义：Paged与Ragged为独立typed schema；F16/BF16、F32 accumulation、checked GQA/
  head/page/layout/causal域。物理capacity与active counts分离，host快照/device mirror/plan
  token同generation，接纳前校验indptr/page ID/tail/shape/资源/overflow。空域明确原数学定义
  或拒绝；padding不是KV。layout更新不能凭shape/地址复用旧plan。
- 冷/热：容量/算法域与有限slots冷准备，每layout显式prepare_batch核对/复制metadata及原
  native planning/H2D，然后run只执行已准备批次。原native host临时vectors/occupancy/private
  bytes如实计账，未知不写0；无submit隐式plan/JIT/fallback/无限bucket缓存。
- 生命周期：PreparedBatch不可伪造且绑定原slot/generation；in-flight/旧输出阻止覆写。已
  接受H2D/native prefix失败或取消不发布旧token；原F1/F2/native证明退休后才能reset/reuse。
  shared page写前由F4显式clone/私有化，原append/scatter需实际writer/effect权限。
- 非目标：模型格式/token/RNG/scheduler/admission/page/LRU/eviction策略、MLA/压缩稀疏/
  所有KV量化、任意rollback、通用增量图、万能Queue/runtime、新math/copy kernel、公共
  build/selector/perf工具重写、自动P1、完整引擎/F6。
- 预算：两轮有限owner/API审计；六FIX。先一个固定原C++ paged/ragged lowering原型，
  native build1800/run120秒；各产品gate configure300/build1800/test120/consumer180，
  sanitizer build1800/test120。每gate原失败/时间累计，不因目录/候选/提交重置；到达预算
  BLOCKED/INCONCLUSIVE，不追加新kernel或无限外库research。owner/生命周期/常态成本
  新事实改变则退SCOPED/AUDITED，AUDITED冻结接口后才写产品。
- 机器/范围：Windows CPU/CUDA/Vulkan、WSL CPU/CUDA与san、实际Air CPU/Apple/Vulkan；
  Mac16,12先核对，Mini不可用；WSL软件Vk只包。只新batch/真实直接消费者/必要新package
  seams与实际需改的旧owner，不回放原算子全集。重型跨机器串行，库j4≥12GiB/j2≥8GiB/
  j1≥4GiB、test1，复用身份匹配source/cache/独立配置binary tree。
- 性能：P0先核capacity/metadata/plan/gather费用和常态路径；仅不可变候选+真实hotspot+
  同合同才另定有限P1。稳定地址/固定batch size不能签capture，topology需原native独立证明。
- 预声明风险：乱序physical页/非整尾/非uniform Q/GQA/causal右对齐；同shape不同IDs/layout
  generation/错indptr/坏页/尾/overflow/empty拒绝；host snapshot/device mirror同版本，payload/
  metadata/plan native use保活、两slots/旧输出、pending producer、OOM/partial upload/native
  error/cancel/abandon/unknown/suffix/close/drain；实际append/read/F4branch/发布/旧版本。
  Direct与Graph同schema/同slot真正运行，对独立dense oracle；完整计入显式参考成本。
  三交付/新头/缺路线/迁址/loader、适用san/并发、guide、P0/LOC/分离红队。每轮docs/closure/
  diff，registry明确previous-ref；仅适用风险可NA，未完成MUST不以NA隐藏。
- stop：必须按活动预算终局或退明确阶段；整个goal完成后再做用户要求的剩余基础库/引擎
  自实现代码量评估，scope不扩完整engine。F4/CLOSED证据不代签F5数学或整引擎吞吐。

## 本轮有限FIX

| ID | 完成定义 |
|---|---|
| F5-01 | typed paged/ragged数学端口、物理容量/有效区、owned host layout快照/同版本device mirror |
| F5-02 | 原FlashInfer C++真正native plan/run、独立slots/workspaces，实际平台/资源资格 |
| F5-03 | prepare/update失效/错误/实际prefix/metadata H2D/退休，旧token不能误run，capture单独资格 |
| F5-04 | opt-in版本化BatchPlanning与Graph原region同slot协调，普通19版region/Invocation不改warm合同 |
| F5-05 | 明确native/显式materialize/pending，actual append→read→F4branch/成功发布及独立oracle/成本 |
| F5-06 | 四自然route Direct/Graph、source/static/shared/新头/缺席/迁址/运行库，风险/guide/P0/红队 |

当前先核原C++参数/计划owner和公开C++自然草图，固定API/依赖/权限/slot失效后才进入AUDITED。
existing Operation::custom的64-byte typed schema可避免改旧variant；原ProviderExtensionRef可借
原Context/Provider保活与admission，但它们的存在不能代签batch完整journey或Graph规划资格。
直接修改旧owner前单独登记其合法scope；不为方便把新batch和所有旧route造成反向环。

QUEUED登记只加两identity，没有源码/头/target/产品root/依赖边。selector保守选全图的原plan
在.scratch/foundation-f4/f5-queued-plan.json；本轮只执行中央身份/状态/docs，不构建未影响
模块/全算子，不修改公共selector缩表。后续真实product seam扩展按本卡再选适用门。

## SCOPED native原型事实

[锁定C++ lowering原型](../../../../evidence/architecture/foundation-f5-native-probe/README.md)
实际PASS：直接原FlashInfer PrefillPlanWorkspaceSize/PrefillPlan与paged/ragged原kernel，
F16/BF16四场景对独立double dense oracle。head64/GQA2，两request Q长1/2、KV长19/10，
page16、物理页2/0/3、尾3/10，padding填64，causal右对齐；最大误差F16约1.15e-4、
BF16约1.01e-3。原planner选CTA16/no split，float workspace0/int64bytes只适用于这个域。
WSL RTX4070/GCC14.2/NVCC13.3，build16.807/run0.365秒计预算，不是性能计时资格。
原代码/锁定输入hash/精确身份/命令与日志封存；没有Python/Torch runtime或新kernel。

原型只证明承载，不签HYLE lifecycle、Graph同slot、其他head/layout/capture/私有allocation。
Graph现存v19 PreparedRegion/RegionInvocation确实拥有opaque slot和workspace，外部规划一个
独立Direct slot不能冒充它；opt-in桥必须访问这同一原slot。Apple/Vulkan的原typed Program
同样保留自身native owner，不能为统一签名转成共同Queue/CPU数学。原native原型后继续
有限owner/API审计，状态仍SCOPED，六个FIX产品未开始。

同原型根与累计预算核对head64/128/256、F16/BF16、paged/ragged共12场景PASS。
head128长KV8193/4097实际native split-KV及merge执行：paged float workspace1,453,056字节/
int604，ragged float1,486,080/int653；不能由batch shape或短域workspace0猜测plan相同。
原planner临时vectors/occupancy/H2D与合并拓扑保留，未强制关闭split、未写新merge kernel。
同根累计build70.455/run0.989秒，原4场景archive与精确输入历史仍保留。
需变更的原Compute/Graph/FlashInfer接缝已按[独立卡](foundation-f5-owner-bridges.md)SCOPED，
只允许新named extension/原region同slot桥/native批次承载，原19版/普通dense范围保留。

## AUDITED 共同端口与metadata冻结

compute_batch进入AUDITED：本factor先拥有F5-01的typed容量/schema/不可变metadata与执行方
共同消费的身份；F5-02..06及整个goal不缩减，原Provider/route执行controller与graph_batch
同slot协调仍按其活动卡SCOPED审计，未经AUDITED不改它们的产品。

| 能力族 | 现状/承载 | 有限动作与验收 |
|---|---|---|
| paged/ragged数学端口/容量与有效区 | MISSING；原dense schema不能表达page表 | SELF BatchCapacity+独立PagedAttention/RaggedAttention typed custom schema，不改旧Operation variant |
| immutable host layout/schema identity | MISSING；两peer均host-known元数据 | SELF BatchSchema::create+version，BatchLayoutVersion保schema/owned F1 readonly blob，same-schema generation |
| metadata/payload readonly/native保活 | 原F1/F2 COMPLETE | BORROW ReadLease/CopyBatch及原native actual retirement；本factor不再做fence/Queue |
| plan/run与Graph同slot | PARTIAL，原C++ probe已验，HYLE尚缺桥 | 原Provider/Graph活动卡FIX，不能以本metadata实现代签 |
| scheduler/token/page eviction/模型checkpoint | APPLICATION | 不解释业务policy；F4仍由consumer显式列完整state |

冻结API：BatchCapacity含storage(kPaged/kRagged)、dtype(F16/BF16)、requests/query rows/
logical KV rows/page arena/page refs硬上限、Q/KV heads/depth/page size；整数界限为原native
int32/uint32端口，extent乘积checked。BatchSchema为immutable owned identity+线程安全cold
generation allocator；BatchLayoutVersion是不可变共享handle。version接HostBatchLayout四个
borrowed spans(query indptr/KV indptr/page IDs/tails)，先完整预检，再复制成owned固定容量
host readonly blob；spans不保留。generation不由caller伪造，same-shape不同layout仍是新版本。
每个版本保schema owner，schema ID跨实例不混用；计数耗尽拒绝，不复用identity/旧token。

定义：query为packed[N,Hq,D]、K/V为paged[Pages,page,Hkv,D]或ragged[KVrows,Hkv,D]；
输出capacity[Ncap,Hq,D]，active rows在layout。KV indptr为page refs或packed rows，page ID
是物理页；非empty Q要求非empty KV且Qlen≤KVlen（right-aligned causal有有效位置）。
empty batch与empty request只描述空有效区，不制造零byte F1资源；native数学是否empty
run按原Provider验。重复page IDs可readonly分享，mutable append必须F4私有化/原writer。
无填充KV进入有效区，末tail必须1..page，zero pages只能tail0。ragged不接page IDs/tails。
snapshot固定容量host blob可原F2上传；export ReadLease用于F4显式资源成员分享/替换，opaque
metadata编码不准raw dirty-clone。不是device生成页表验证或隐式D2H。

依赖冻结：compute_batch PUBLIC compute/compute_ops/error/mem，PRIVATE core/util_core；
不依赖Graph、路线Provider或compute_state，不产生反向环。公头batch_layout.h、batch_ops.h
归新factor。版本冷分配计入inject allocator/ResourceBudget，blob bytes另计原mem资源；
验证O(requests+page refs)，版本复制O(1)，常态Tensor/Invocation无新增字段或每步查表。
新factor先由维护入口验证，只在真实façade接缝合法纳入后进入普通Compute产品；该接缝不能
偷偷改CLOSED owner。两纵切同样只消费owned分段offset/index有效区，不把模型token语义下沉。

本阶段有限FIX：F5-COMMON-01 checked schema/typed output推导，02 owned host layout版本/
generation及F1 export，03界限/错表/alias共享/OOM/旧版本并发/empty测试与普通原型消费者。
三项是F5六FIX的共同端口实现分解，不是替代目标。原native C++12场景与上游peer作为数学
承载输入；snapshot runtime尚未验收，产品实现后仍需全部适用风险/交付及终审才能CANDIDATE。

## AUDITED 共同端口首批实现

[首批证据](../../../../evidence/correctness/foundation-f5-common-first/README.md)：Win MSVC19.50/
shipping/WERROR，实际15项及两新公头独立TU PASS。typed容量与paged/ragged输出推导、完整
host布局预检、fixed-capacity owned只读blob/schema generation、foreign schema/错表/页尾/
GQA/整数与stride overflow/empty/OOM/并发版本/晚于schema的F1导出；真实F4分支替换/发布后
旧layout值保持。首fixture假设F1截断read/写拒绝码错误修正，原FAIL与累计预算保留。
只新compute_batch factor及F4 TEST_DEPS；没有原Provider/Graph/全算子验证或产品改动。

现产品仍AUDITED，不是candidate或F5六FIX完成：device mirror、真正HYLE native规划/执行、
Graph同slot、四路线明确参考/append闭环、三交付/san/P0/终审继续按活动卡。新factor未加入
普通Compute façade，此发现接缝后续须独立合法scope，不在CLOSED旧owner顺手改产品根。

原native执行首批已按[owner卡](foundation-f5-owner-bridges.md)落地：BatchProgram/
BatchInvocation/PreparedBatch opt-in shell、原named Provider capability、原F2 pinned metadata
上传/原planner与kernel/explicit update、fixed slot/旧结果/actual-retirement及容量额度。
Win共同16项/四头与WSL实际Direct四域各三布局oracle通过，原失败/预算保留。共同层仍
AUDITED；F5六FIX未全部完成，尤其Graph实际same-slot与四路线reference/append完整消费
旅程/三交付/san/P0/终审继续。本轮不改原Graph v19/普通Invocation形状，不把native Direct
证据替代Graph或Apple/Vulkan，也不声称Windows已有FlashInfer native paged。

Graph原region同slot首批已实际通过，见[owner卡](foundation-f5-owner-bridges.md)和
[native Graph证据](../../../../evidence/correctness/foundation-f5-native-graph-first/README.md)：
两域各两节点串联共享一个metadata输入、两个独立计划slot及拓扑改变后12次oracle。
同输入不同版本/重复node/metadata注入/第三slot/旧结果明确拒绝；共享readonly镜像在原
Graph控制门之后释放/回收，未改变普通v19/Invocation常态字段。每region自己的显式
metadata上传和规划费用保留。F5-04由接口落地推进到有限实际消费，仍不签完整风险，
产品保持AUDITED；四路线reference/append/F4闭环及交付/san/P0/终审继续。

## AUDITED 显式materialize参考档冻结

baseline 9d7041ce6；这是F5-05/06已准入reference的有限分解，不新增数学/kernel/runtime。
已核现有Apple MLX Gather/MatMul/Softmax、Vulkan原ggml Gather/MatMul/additive Softmax与
CPU原Provider。Vulkan原SDPA不接外部mask，不能把固定capacity padding算作真实KV。
本档明确展开原typed操作，qualification仍由实际route冷prepare/compile裁决；不改旧route
产品源码或默认选择。缺数学域明确拒绝，不偷跑CPU，不称native paged/capture。

- REF-01 compute_batch拥有MaterializedBatchProgram：冷固定same-schema、每request KV硬上限、
  host metadata与全部逻辑intermediate byte上限、scale/causal、allocator/ResourceBudget。
  原typed recipe以五输入Q/K/V/owned I32 gather indices/F32 additive mask承载，输出同Q容量。
  recipe只由原View/Gather/Cast/Copy/MatMul/Softmax构成，cold一次推导每value描述；每query
  作为独立batch，gather显式展开KV heads至Q heads，F32 score/softmax/value累积后转回原dtype。
  物理索引把(page,row,KV head)映到原arena，right-aligned causal mask不让padding进入有效区。
- REF-02 version(layout)先完整same-schema/每request上限预检，再产生owned readonly host
  index/mask快照并保layout/program。每layout更新生成新同版本快照；原F1导出、F2显式上传，
  不接受caller/device raw表冒充layout。空active batch明确拒绝；inactive capacity输出仍未定义，
  其mask只保一个dummy可读位置，避免全mask softmax，无业务有效区或额外有效KV。
- REF-03 graph_batch从同recipe建立普通原Graph input/nodes/published output；CPU/Apple/Vulkan
  原编译器与Direct原operation Program均消费同typed序列，原native owner/Completion继续自身
  生存规则。此阶段先共享recipe/metadata/Graph构造；实际route Direct组合控制器和完整native
  运行必须后续取证，不能由graph构造API代签全部旅程或要求用户重写生命周期状态机。

成本：host生成O(Qcap*Hq*KVbound)，index I32及mask F32，两个gather输出明确
O(Qcap*Hq*KVbound*D)字节，GQA重复、F32转换/permute materialize、scores/probabilities均
计入logical intermediate保守上界；该上界不是native resident精确值，Provider/driver仍UNKNOWN。
硬上限不满足在接纳前拒绝。metadata上传、native scratch、host wait与每层重复费用分别报告，
不从native Direct workspace小值推导本档免费。常态非batch对象无新增字段/路径。

有限预声明：乱页/非整尾/非uniform Q/GQA/causal及noncausal，foreign schema、超per-request/
host/intermediate cap、整数乘积/flatten I32 overflow、empty拒绝、只读导出/旧版本/allocator OOM；
共同schema测试/公头，四实际route Direct+Graph独立oracle、F2同版本上传及F4 append/branch/
旧版本/native保活；原source/static/shared/缺路线/迁址/loader及san/P0/guide/红队继续主卡。
预算、失败历史与串行机器规则沿原卡，不因reference档另起累计预算。旧route真实接口若需
修改，先独立owner scope；本冻结不授权重写CLOSED数学/图编译器或公共构建/验证工具。

## AUDITED 显式reference共同层与CPU首批

[本轮证据](../../../../evidence/correctness/foundation-f5-materialized-first/README.md)：
MaterializedBatchProgram/Version与普通Graph构造已实现；Win共同23项/六头，实际Win/WSL
CPU paged/ragged ×F16/BF16 ×causal/noncausal各8域、Direct/Graph各16次独立oracle通过。
未用物理页/尾padding填NaN，active结果仍正确。masked容量gather重复当前request真正
有效行，避免读padding再以0*NaN污染；原checked Gather保留native界限拒绝，不以host
预检代替用户直接使用普通recipe时的native检查。

固定typed序列由原View/Gather/Cast/Copy/MatMul/additive Softmax承载；最终Copy为独立
输出显式计费，原Graph不导出private reshape alias作为外部F1输出。示例host1920bytes、
gather98304、logical保守上界paged553984/ragged529408；含GQA重复/F32转换/permute/
alias view及output Copy，不称精确resident，原Provider/private仍UNKNOWN。冷explicit byte
caps、same-schema/每request bound/overflow/empty拒绝、owned旧版本/只读export/OOM已测。

这是共同能力与CPU数学承载证明；Direct样例先冷prepare全部step，再沿原leased operation
逐步submit/wait，不在sample运行时规划，也不把它当完整生产batch controller。路线Direct
组合控制器、Apple/Vulkan实际native Direct/Graph、F2同版本上传、append/F4完整闭环及
其他故障/SDK/san/P0/guide/终审继续。产品保持AUDITED，不签F5-05/06完整完成。

Apple/Vulkan Direct冷适配已按[独立实际owner卡](foundation-f5-native-reference-bridges.md)
合法纳入并实现。actual Win Vulkan、Air Apple/Vulkan各八域/双slots/metadata更新后的
48次Direct+Graph oracle及两个route新头独立TU通过，见
[原生reference证据](../../../../evidence/correctness/foundation-f5-native-reference-first/README.md)。
共享typed recipe在GPU由原MLX/ggml/IREE/native Program承载，原F2镜像上传与版本更新
明确发生，旧result保护仍原native owner；没有CPU fallback或capture/native paged假报。
F5-05/06向四route实际数学消费推进，batch控制器、append/F4完整闭环和其他最终门仍待验。

## AUDITED 逐页权限与append接缝冻结

baseline 780c666e2。本轮真实反证：F1整块arena reader的subview不能单独变成free-page
writer；同backing的两个独立F1 block可以分别native-read和write，但单页ReadLease不能
绑定当前整个arena的数学TensorDesc。F1合同正确，缺的是F5物理视图与实际page footprint
的接缝；不得用raw alias绕过readonly，也不得把全arena clone当有成本优势的页CoW。
原25项测试保存反证；首新增fixture以整个view读1byte的尺寸假设错误保留，修正显式subview。

有限实现分解（仍属F5-05/06，不扩页淘汰/模型策略或新kernel）：

- ARENA-01 compute_batch的PagedArena冷分配原resource一个连续arena；固定物理page slots
  各自原F1 block，另有cold lifetime sentinel。acquire(page ID)只给没有旧reader/native use
  的原WriteLease，PagedPageWriter::publish/complete在成功实际退休后登记ready page。
  old-state分享的是page lease，不是整arena reader。页选择/free/LRU策略留consumer；借用
  native resource/allocator的原生命周期保留，无普通对象字段变化。
- ARENA-02 PagedArena::read(layout,readonly page leases)先核same-schema、每个actual page
  ID/原block/完整page范围、集合完整且无foreign，再给不可伪造的PagedArenaRead快照。
  它保layout、arena和去重的actual页permissions。whole native TensorView只供trusted private
  adapter；公开API不导出一个虚假的whole-arena ReadLease或可写raw alias。
- ARENA-03 原native Direct增加query lease+两PagedArenaRead的显式run档；核same layout/
  resource/容量，原LeaseUseBatch持全部actual页到原Completion实际退休。Graph/native
  reference接同footprint，仍原region/slot/Program/Completion；private projected binding只许
  已资格batch readers，禁止给普通dense节点或直接发布整个arena。需要实际旧owner bridge时
  在其AUDITED卡补固定FIX后才改产品，不在common factor反依赖Graph或native route。
- ARENA-04 actual append：旧page只读→原F2/原pure Scatter产生private新page→原arena新slot
  writer成功ready→F4 draft替换dirty page与layout→expected-base成功发布；旧page/版本仍
  可原数学读，未改成员保持共享。计实际页copy/append/materialize bytes，失败prefix/取消/
  unknown不发布或误复用。具体F4目录/schema由consumer完整列明，不新建模型事务。

P0：cold O(max physical pages) F1 slots/metadata，read预检/保活O(unique referenced pages)；
数学仍用原连续arena地址与page表，未使用页没有read权限，因此可在独立slot写入。原F1
block计数保证old reader/native use阻止reuse。mutable writer弃用/accepted copy未结束时
不得假空闲；无法证明ready者保留Unavailable/失败状态。无自动grow/页cache或热DAG遍历。
只有本显式paged档改变，ragged既有域与本goal的完整消费验收仍保留，不能冒充general。

适用验证：原共同factor的错page/schema/backing/尺寸/duplicate/empty/OOM/overflow/旧reader/
native read-use/abandon/close、真实F4 branch与普通single/multi request append/read；原CUDA
Direct/Graph以及CPU/Apple/Vulkan explicit reference完整闭环/预算/数学oracle；原SDK/san/
P0/guide/终审。全部预算与失败沿原根累计，不以新文件/提交重置。owner仍AUDITED。
