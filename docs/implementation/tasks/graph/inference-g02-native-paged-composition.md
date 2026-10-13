# G02 CPU 与 Apple/Vulkan 精确页联合消费

状态：AUDITED，2026-10-14。当前 owner 为 graph_route_cpu_apple、
graph_route_cpu_vulkan、graph_batch；baseline `66af56852d0602ce14cdfde9aba3de373a7472c6`。
授权来自[十项 goal](../project/inference-gaps-goal-execution-2026-10.md)。原
[Foundation A](foundation-a-final-closure.md)有限域与失败证据保留；本卡不重开旧数学、
effects、普通 native Graph、scheduler 或资源桥。

## 触发、消费与边界

真实触发是已准入 G02 的 CPU→原生 page16 attention→CPU 旅程。G01 新 native
PagedBatchInvocation 已实现精确页读取并完成 actual Air 数值、原生风险和三交付；
其产品候选为 `520027a57`，性能仍在独立固定副本验收。本轮不修改该候选。共同
LeaseSubmission::bind_pages 已在[原 G02 卡](inference-g02-multi-queue-batch.md)验证。

当前两个联合 Invocation 公开面只有 ConstTensorView、单 ReadLease、native tensor
和 weight binding，不能传入 PagedArenaRead。编译器仅将普通节点切成原 CPU/native
fragments，原 native fragment 不接 owned layout/页表。把整 arena 投影为普通 readonly
tensor 会扩大权限，且不能证明数学只读取获准页。这是新消费需要与现有形状之间的
可复核缺口，不是对已关闭普通域重看一遍。

主消费者为 Qwen3.5 GQA8/2 D256 attention 片段：CPU query 数值预处理→实际 Apple/
Vulkan 两请求 page16 attention→CPU Cast/Neg，含单请求、混合 prefill/decode、乱序/
尾页/共享页和旧结果。正交消费者为非模型的两个只读分页数值域、独立 slots 与真实
CPU 前后段，防止以单一路线、raw pointer 或仅 GPU 运算冒充联合。

完整 public journey 必须让用户沿原 GraphBuilder、compile、create_invocation、绑定
确切 frames、prepare_replay、submit/wait、结果 owner 与 close 完成。图节点只表达
数学/固定形状，冷 binding 拥有 schema/recipe/layout；用户不生成 raw metadata tensor、
不写第二套提交循环、权限退休或 CPU/GPU scheduler。请求、cache 调度与模型政策不下沉。

范围是 owned 精确 layout 的 native fragment、公开 page binding、实际 Q/输出跨 CPU/GPU
边界、guard/generation/失败/取消/旧结果和实际退休。至少沿 G01 当前 Apple F16/BF16、
Vulkan F16、D64/128/256/page16 数学域验收。Vulkan BF16 全链继续由原 G05/G07 处理；
CUDA/CPU 与原 G02 未完成门保留，不能由本卡缩减全 goal。一般 shape、capture、producer、
delta planning、量化 KV 与重叠流水仍对应 G03–G10，不冒称本轮已经兑现。

## 承载审计与下一步

BORROW 原 CPU/native fragment、Context/Queue/Copy、G01 单 table/两 bank 共同 controller、
原 ResultSlot 和 MQ-LP guard retirement。SELF 只冷 graph node/owned layout 映射、准确
frame 资格与原 fragment 接纳，禁止新增数学、controller、executor、queue 或 whole-arena
permission。常态普通 nonpaged 路线的 binding、native output topology 与 alias/effect
预检保持其原合同；页专用承载应 opt-in，不为每个 ordinary value 分配最大页表。

原 CPU+Vulkan 在 prepare_replay 绑定并 seal 原 native 输出 backing，drive 检查结果
backing 不变；接入页 controller 必须核清该输出与后继的所有权，不能删掉原检查来
取得 PASS。Apple 原 drive 在 CPU prefix 实际完成后再绑定原 native inputs；两者保
不同的合法原生行为。失败或取消只有实际 prefix 退休后才释放所有 referenced guards；
未引用页始终独立可写，页输入不得发布或流向普通 dense/CPU 操作。

图内页节点/冷绑定的共同 owner 预计为 graph_batch。其原 MQ-CPU candidate `a0c047ea`
当前未重开；必须先以新事实合法回到 AUDITED、明确该共同 API/依赖和有限 FIX，才可
修改产品。两个本卡 owner 先进入 SCOPED；公共形状、每段 owner 与有限 FIX 还没有
最终冻结，不写产品代码。下一步只完成该接缝的有限源审计并进入 AUDITED。

## 预声明验证、预算与停止

P0 核实际 CPU→GPU→CPU 原 math/copy/完成事件，精确页集合和 host wait/table/native
private UNKNOWN；不重跑 G01 已关闭 CPU/ragged P1。若 API/算法固定后发现独立的常态
成本问题，须明确同层同合同片段再冻结定向 P1，不能从 benchmark 行改变公共形状。

源/relocated static/shared、逐公开头、actual Air 两路线联合数学与对应宿主 ASan/UBSan/
TSan；沿已声明 Metal TSan 同步边界模型与 ASan 无 LSan 限制，不把宿主插桩当 shader/
driver 资格。Windows Vulkan 定向编译/链接；本机重型 GPU run 仍不恢复。适用 unit 与
原 cpu_apple/cpu_vulkan contracts 直接负空间，selector 保 selected/skipped reasons。

必须覆盖 foreign schema/resource/generation、重复/共享 pages、空域、允许的 topology
变化、两 slots/旧 output、每个引用页防写、未引用页可写、冷预算/逐分配 OOM、CPU/native
失败、accepted prefix/未知完成、取消/放弃与 close。所有未运行门 UNPROVEN；不能
用共同 helper 或 G01 direct 输出代签新联合消费。CPU prefix 与 GPU fragment 都须实际
运算并由独立 oracle 核输出，必须检查原图其它消费者拒绝扩大页权限。

源审计/实现/验证各2小时检查点，按用户常设授权有限延长并保所有 FAIL/elapsed/累计；
重型串行、四 jobs 上限、不运行本机全模型。停止按固定 FIX 与适用门终态，或真实
外部阻断/协议回退；不因短观察超时重启工作。docs/closure 每轮适用检查，无产品修改
的本 SCOPED 登记不构建、不跑性能。

## 有限接缝审计与 API 冻结

SCOPED 两 hybrid 登记提交 `2e9444109`。本节完成三 owner 的有限审计；graph_batch 原
MQ-CPU candidate `a0c047ea`及全部原 common/CPU+CUDA证据与未决 Windows 门保留。
该 owner 仅为新图节点/冷 recipe-layout 映射依法回到 AUDITED；原 graph candidate5ce、
五个 G01 candidate520 与两个普通 native Graph owners 均不改产品。

三 owner 归属：graph_batch 拥有可被 GraphBuilder 描述的三 port native page 节点及冷
node binding；两个 hybrid 拥有原 CPU/native fragments 中的页消费和 public invocation
接纳。这里只给两个 hybrid 增加对 graph_batch 的真实 public 边，不增 component 或
重排图；Compute 不反向依赖 Graph。固定三 engine 的准确 block-table/valid-region
消费需求沿原审计，不再扩成 peer API 并集。

固定 API：graph::batch::NativePagedAttention 是原 custom Operation 的小型 trivially-copyable
descriptor，三 ports 为原 recipe 的 Q、真实物理 K、真实物理 V；输出是 active packed Q。
factory `native_paged_attention(recipe, layout)` 验同 schema 并只保存 capacity/active rows/
scale/causal；冷 `NativePagedNodeBinding` 保存 NodeId、owned recipe、初始 owned layout
和显式 NativePagedInvocationOptions。hybrid CompileOptions 增加 paged_nodes span，仅借
compile 调用；plan 拥有这些配置。普通调用沿 builder.add_node、compile(paged_nodes)、
create_invocation、prepare_batch(span<NodeLayout>)、bind(Q/source)、bind(K/V,frame)、
prepare_replay、submit/wait。所有 raw metadata 仍由原共同 controller 持有和 F2 上传。

每次 prepare_batch 必须给出完整 live 页节点集合。新 layout 同 schema且 active Q
总行数保持该固定图输出形状；request 拆分、page order/共享/tail/logical长度可以在
已资格域中变化，冷准备原 G01 native Program 并上传 owned I32 table。active Q 总形状
变化沿原 compile/有限 exact variants，不能冒称一般动态形状。caller spans 不保留；
拒绝不允许混用新旧 prepared tokens，旧结果/实际 uses 阻止 prepare、bind、reset。

每个页节点保独立 native fragment 边界，前后 ordinary fragments 沿原编译器和 runtime。
BORROW 同一个 MaterializedNativeState/PagedMetadataPolicy 与原 native Program、Invocation、
F2 banks、exact frames、native retirement；只提供薄 traits，不能复制 controller。
初始 native Program 在 compile 经原 private page factory 资格；相同初始 layout 可借
这一 immutable Program，其它同形状 topology 在显式 cold prepare 调原 factory。
页专用 plan/slot/input 映射 opt-in，普通 value 不增每值最大页表或 frame；记录新增冷
plan/slot metadata，普通数学/提交循环/queue不替换。

K/V 只能是 Graph inputs且所有读者都是本轮 qualified 页节点的 K/V ports，同一输入
关联的各 layout 必须完全同代；禁止 publisher、CPU/dense/View/query/未资格 reader
消费整 capacity。bind(frame) 验 node/schema/layout/generation、真实 native resource与
capacity，再用 MQ-LP bind_pages 复制 exact guards；所有拒绝保旧 input。普通 tensor/
ReadLease/weight bind 不得绕过这些页 input 限制。全 Q0/page_refs0 只需真 Q，K/V 无
frame或假 root；原 G01 empty native Program保持。共同 guards 从 CPU prefix admission
开始保护到全部 accepted native prefixes实际退休，不依赖 tensor 的地址可见性授权。

CPU query prefix 成功并完成原 upload后才把 ready Q 交给页 controller；错误不得启动
页 math 或后继。Apple 页结果沿原 checked completion与owning result传给后继。Vulkan
页输出在显式 cold prepare 接入原 CompositionAccess::output/seal 的确切 reference
baseline，必要 seal 在 bind完成且任何 native admission 前；保 ordinary fragment 的
backing不变检查。原 transfer retain 与 CPU后继仍用真实 native buffer。旧 native result
只在 outer idle/result gate通过后释放；取消/放弃仍由原 coarse drive和实际 prefix退休。

| FIX | 允许改动 | 验收 |
|---|---|---|
| NJ-01 | graph_batch 三port descriptor/factory/owned cold binding，共同 recipe与active输出资格 | schema/shape/empty/foreign/非法node与common public source/SDK/header |
| NJ-02 | 两hybrid 编译器冷owned映射、页节点独立原fragment、初始original Program与两真实public依赖边 | 真实CPU前后段、原node/control/ordinary negative space、无CPU fallback |
| NJ-03 | opt-in slots借原共同page controller，完整prepare_batch/layout version与原F2单table/两banks | 同形状topology演化/完整集合/双slot/old results、显式cap与冷成本 |
| NJ-04 | public exact-frame bind与MQ-LP guard组；限制所有page input readers/publishers | foreign resource/generation、shared/unreferenced pages、拒绝保旧绑定/权限不扩大 |
| NJ-05 | 原drive中的ready Q/page math/原Vulkan output seal与后继数据/真实retirement接缝 | CPU/native prefix错误、unknown/cancel/abandon、实际page guards与结果生命周期 |
| NJ-06 | 公开指南与同一个主/正交消费者，三交付/逐头、适用宿主风险与原consumer回归 | 实际两平台/math/oracle/逐cold OOM/预算、原所有FAIL与限制保留 |

P0 收费包括 graph node/plan/slot 冷 metadata、实际 table upload/host waits、Q/输出边界
copy与native private UNKNOWN。共同 LeaseUsePool 仍可能 O(guards²)去重，G06不代签。
native query readonly wrapper需要的metadata按真实生命周期收费；不声明零warm allocation。
这是组合资格轮，没有新的吞吐优势宣称，不启动新 P1或重跑G01数学P1。适用门仍全部
UNPROVEN；发现改变上述shape、owner、permissions或成本级别的新事实先回本卡裁决，
不能删除原 Vulkan seal/guard来得到PASS。七问沿上述已锁依赖/owner/BORROW/无FFI、
P0、三交付与自然公开journey裁决，candidate以后只补实际结果。


NJ-01草稿已写common三port descriptor/factory与owned NodeBinding声明；同一graph_batch
目标新增native_paged_node.cc，不改产品root/全局构建或G01候选。独立fixture描述active5
而capacity11、logical95而physical64的重复/乱序页，并由真实GraphBuilder核三inputs/
active output；foreign同capacity schema、空输出与伪造physical root反例已登记。当前
尚未compile/run，不能写PASS或candidate。三owner保持AUDITED，NJ02–06未实施。
selector NJ01原plan为READY，只选择实际graph_batch与其真实direct消费者/逐头/Graph交付；
按活动卡先编单CPU common consumer及原materialized负空间，待当前Apple正式P1实际
terminal后串行运行，未扩大成全部算子或重跑closed数学。


NJ01 source阶段五项PASS：active5/capacity11、logical95/physical64、三真实Graph inputs、
foreign同capacity schema、empty/伪root拒绝及原三materialized负空间，见
[阶段](../../../../evidence/correctness/inference-g02-native-page-node-stage/README.md)。
首次fixture误用Operation(descriptor) compile FAIL，修为原Operation::custom后通过；
产品不因fixture失败改形状。累计84.031s，原FAIL/log与标明重建的首test输入保留。
只是common GraphBuilder/schema源资格，不代签native math/frame guard/SDK/san。
NJ02–06未实现，三owner继续AUDITED，不能candidate或签整个G02/goal。


NJ02–05联合草稿已写入两个AUDITED hybrid与共同graph_batch：冷编译校验全部页读者、
隔离原native fragment并保owned recipe/layout；opt-in slots借同一page controller；
公开prepare_batch与exact-frame bind接原MQ-LP guards；实际CPU完成后的Q沿原资源桥进入
page math，Vulkan仍预接原输出owner并保seal/backing检查。中央只增加两个真实public
依赖边，普通native Graph/G01固定520产品与不可变P1源码没有修改。

公开联合消费者已登记实际CPU ABS/Cast→原page attention→CPU Cast/Neg，page16三D、
Apple两dtype/Vulkan F16、causal/noncausal、双slots与12次同输出形状topology演化，
用独立double核全输出；未引用sentinel页仍独立可写。当前尚未编译或运行，所有新的
联合资格均UNPROVEN。Air原cached两route树只编公开hyle::graph消费者与原ordinary
contracts，fourjobs、重型串行，等原Vulkan P1真正终态后执行；原FAIL与累计保留。

本草稿P0包括每plan一冷paged vector、每Invocation一个可空owner与仅opt-in页slots；
每页slot复用原table/banks/native准备，额外query readonly wrapper如实收费。没有每
普通value最大页表或逐值frame，但不能声称普通owner成本绝对为零。selector当前READY，
中央依赖文件触发保守全factor选择；原plan及独立selector工具卡保留，仅执行本活动卡
实际owner与直接消费门，不重开无变化模块。closure64/docs1195检查PASS不代签产品。


## NJ-Q01 真实 ready Q 只读资源接缝

原三owner仍AUDITED；compute_batch新接缝SCOPED，baseline `8cf6be8d3`，原native五owner
finite terminal与candidate520不撤销。首次actual Apple public联合consumer配置6.162s/
编译6.390s PASS，数学2.469s FAIL：native batch Q resource differs。原19 SHA输入/原日志
保留于本轮scratch固定stage；失败未启动page数学。Vulkan本新联合尚未编译/运行。

源码事实：Apple MemoryResource::import_readonly 保活真实ready MLX array，以派生于原
MemoryState的ImportedArrayStorage导出永久readonly lease，资源identity有意不同于
分配资源。原MemoryResource::binding已经资格化这类真实Metal carrier及dense形状、
origin/size，但MaterializedNativeState在调用原binding前先要求query.resource==device.ref。
Vulkan原readonly import仍采用exact device.ref与真实AllocationToken，没有这个差异。
这是真实引擎ready producer结果消费的适配缺口，不能伪造资源identity、删除所有resource
校验、增加整Q冗余copy或复制共同controller来规避。

scope只允许compute_batch共同controller一个private query-resource资格traits接缝，
供本轮hybrid Apple traits明确准入原readonly native authority；随后仍必须调用原
MemoryResource::binding核真实MemoryState/native handle/description/origin。未声明该
traits的原G01 direct/reference与Vulkan保持原exact resource校验，API/layout/metadata/
权限池/submit/retirement不变化。三个原G02 owner沿NJ02–06，不扩大owner或数学域。
必须以失败原consumer通过与原direct/reference负空间、foreign真实资源/伪resource
拒绝反证该桥；source/static/shared/头/宿主风险沿本卡，P0无额外payload copy/队列。
