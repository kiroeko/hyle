# G03 已接纳页更新到原 attention 的组合范围

状态：CANDIDATE，2026-10-14；当前owner compute_batch；续接baseline `1dcc26cc0af4c6c68a4c9ab330fbd8088dc6aaa9`（初始baseline `c8ea4cc31`保留）。
授权与全项范围仍为[十项goal](../project/inference-gaps-goal-execution-2026-10.md)。
GP01/GP02已按[有限terminal](inference-g03-dirty-page-terminal.md)关闭，原数学/性能/失败
和private UNKNOWN保留；只因原goal尚未完成的真实producer→attention旅程启动新轮。
未固定AUDITED/FIX前不改产品，不能由历史卡授权修改其它CLOSED owner。

真实主旅程：QKV/RoPE或普通数值producer生成本步值→精确dirty page update→同设备
原attention→数值输出，Q1/17、GQA8/2 D256、64步增长/跨页、两request/两slots/共享页。
反偏置旅程是普通数值页producer→range更新→readonly reduction。模型slot/request/evict/
cache政策不下沉；公共机制必须保成功、actual retirement、取消/放弃/unknown与费用。

[actual pending producer公共反证](../../../../evidence/architecture/inference-g03-pending-producer-public-probe/README.md)
已证明CUDA F1 writer→原F2 bind_mutable(ordered_view)+bind_producer有自然路线，不能重造。
F2未完成时只保原exclusive permission，successful output与Arena.read/batch bind仍须ready；
GP01/GP02解决成功后回接，不提供pending-page或attention native dependency。当前接缝
须用最小complete consumer与实际copy/wait/plan/allocation账裁决，不能先设计第二controller。

四路线/三个CPU+GPU自然public序列及原F1/F2/native对象回映；特别核原Direct/native
output何时具真实backing、accepted prefix依赖、页初始化/dirty范围、native inputs保活、
旧输出和cold metadata顺序。原MF controller、PagedBatch/Graph slots、原native completion/
dependency/lease机制优先；不伪造ready/whole-arena权限、host callback结果成功或CPUfallback。
有真实下层新机制才登记对应owner scope/audit，先compute_batch核组合owner边界。

预声明验证：64步非零独立oracle、producer→dirty update→attention真实完整生命周期，
所有referenced/nonreferenced页权限；pending无output/unsafe reuse，producer/prefix错误、
unknown、cancel/abandon/close与actual retirement；shared/old state/两slots，foreign/layout/
版本拒绝、cold逐allocation OOM/明确预算；source/static/shared/独立头与宿主san/真实直接
consumer。CPU/CUDA/Apple/Vulkan真实资格分别记录，不拿atom或一个平台代签完整ROUTED。

P0先做纵切费用账，finite支持域/API/owner/ownership/cold-warm/failure/并发与FIX固定后
才实现；有新normal算法再对不可变candidate冻结必要同合同P1，不重跑原G01/GP01/02。
G06 metadata/capture、G09流水和G04/05/07数学/编码仍原项目，不扩出新全局调研项。
主张无新API必须给完整实跑、寿命与费用，而不是只说F1/F2已有。

2026-10-14 的 [Apple 原生输出公开探针](../../../../evidence/architecture/inference-g03-native-output-copy-shape/README.md)
已用同一 Abs Program/Invocation 的两个真实 F16 输出证明：readonly 导出 resource
身份逐次不同，固定设备 ref 的 F2 source binding 拒绝；按每次导出重建原 copy
program/slot 后 1024 bytes 数值复制正确。固定产品未改，全部 fixture FAIL 与累计
42.421 秒保留。此结果定位 ready producer 来源绑定的 cold 成本接缝，不代签 pending
page/attention，不据此提前固定 API；若需改变 F2/native owner，先登记对应 scope/audit。

该 ready 来源接缝随后按[Apple 有限 terminal](inference-g03-apple-copy-source-terminal.md)
完成 AS01/AS02：固定原 program/slots 接 trusted readonly exports；实际 F16/BF16
producer→dirty页→attention、三hostsan、两SDK全部通过。不能重复该已验范围或把其
显式producer-ready/copy观察当成pending组合。compute_batch仍SCOPED，未固定完整FIX。

下一次有限审计只裁决以下原接缝：

1. 原 CPU/CUDA caller-owned writer 已能pending producer→F2；原 native OwnedResult 的
   completion为const且输出只在成功结果中可见。Apple真实backing在原worker eval时
   产生，不能在host预绑定时假定存在；Vulkan冷输出buffer不等于成功读取权限。
2. 原 F2 submission保原 mutable binding/phase/native use，unbind_mutable须成功实际
   退休且没有保留submission；read frame接ready原页。若组合需pending页，必须显式
   保原页身份/初始化范围、被接纳copy的精确写范围、producer/prefix成功与actual
   retirement；不能伪造ReadLease或只把writer裸span交给attention。
3. 后继attention须依原queue/Job/Completion实际排序，失败/unknown只阻止suffix，不
   发布页成功；保旧reader/result和原native input寿命。公共组合优先借原F2及MF/Paged
   controller，不能再建scheduler或在consumer重复host continuation状态机。

必须先写四原路线/三联合路线具体调用与cold/warm账，再固定owner与有限API/FIX。
原ready版本可作为数值对照，pending验收要单列中间host waits，不能借上面的PASS代签。

本轮已完成[五场景实际所有权handoff反证](../../../../evidence/architecture/inference-g03-pending-copy-owner/README.md)：
原invocation与submission一起移动后只留单一观察者，原guards/actual retirement/失败
与cold slot可回接，16-byte局部组合不修改普通F2字段/锁/原子/queue。后继source seam
按[三owner新scope](inference-g03-copy-update-owner.md)固定typed身份、原prefix及受控读
admission；本主卡不因此进入AUDITED，不把fixture线程/wait当产品执行owner或完整路线。

源审计/实现/验证各2h检查点，沿用户常设授权有限追加，保全部FAIL/累计；Windows
canonical，Air only实查身份，WSL/Air可丢弃单向副本。所有heavy serial、最多四jobs，
不跑本地重型GPU/full model、不重启旧Windows大套件、不改公共构建/性能/治理工具。
停止依据有限FIX和实际门裁决，普通一次失败/未知/短观察timeout不作为停止或重启理由。

## 当前已完成的 source 接缝与有限下一步

原F2的[CopyBatchUpdate有限terminal](inference-g03-copy-update-terminal.md)现已关闭
compute/Apple/Vulkan三source owner CU01–CU03：两原controls移动到唯一观察owner，
trusted mutable geometry/root/last-write、保原native prefix、原ordered read admission；
四source/生命周期/六hostsan/六SDK与定向Windows证据成立。这里是当前主卡重新纳入
的已验输入，不授权重新修改CLOSED source产品。

接下来本主卡的有限审计只固定两个仍真实缺失的receiver组合：

1. 原Paged/Batch controller消费移交的update，精确核页原root/初始化/布局/写范围与
   全部referenced页权限，借原native endpoint排序和F1 use；最终合并所有producer/
   acceptedprefix错误，actual suffix退休前保原source owner，不伪造ReadLease或新队列。
2. 原Direct/native producer到update的实际输出寿命：CPU/CUDA已存在caller-owned
   writer/producer自然序列；Apple original backing在原worker eval之后才产生，Vulkan
   冷buffer不是成功读取权限。需逐原owner确定late backing/已接纳native输出的受控
   消费入口、cold/warm账与失败/取消/unknown规则，再固定scope/audit/FIX。不得由ready
   显式hostwait版本代签；新source机制如确需修改CLOSED owner，另注册真实新轮。

仍须四路线/三CPU+GPU联合、主attention与正交数值reduction、64步两request/两slot/
旧结果、actual数值oracle及完整风险/交付。当前仅SCOPED，未固定receiver公开形状，
不允许compute_batch产品实现。原G01/G02未决门及G04–G10目标全部保留。

## receiver 纵切审计的已实测事实

2026-10-14，[原 pending copy→attention 反证](../../../../evidence/architecture/inference-g03-pending-attention-receiver/README.md)
已在同一 Apple/Vulkan 原 Runtime/queue 实测：source update 已请求取消，而原后继
attention 仍返回 SUCCESS。原 guards 保持到 actual retirement，最终只 wait attention；
这证明 native 排序不能独自承担来源错误合并。原型为 trusted 组合，不是公开 receiver
验收，未给独立数值 oracle/64步/late producer/全七路线签 PASS。Apple 错误 target、
Vulkan fixture compile/shared private symbol link FAIL 与累计时间完整保留。

本轮据此固定接纳与退休边界：接纳前已知来源失败或取消必须拒绝 suffix；接纳后才
确定的失败不承诺 native work 抢占，必须退休所有已接纳 work、保 guards/source owner
并拒绝发布成功结果。unknown retirement 不能当成功、不能提前返还 writer/reuse。

| 路线 | 原对象与实际调用 | 已有接缝 | 尚须本轮裁决 |
|---|---|---|---|
| CPU | caller writer→原 producer completion→F2 bind_mutable/bind_producer→CopyBatchUpdate | 原 queue sequence / F1 ordered permission，可复用，无需第二执行器 | Batch controller 的精确 pending 页接纳、suffix 与来源最终状态合并 |
| CUDA | 同一 F1 writer/原 CUDA producer→原 F2→CopyBatchUpdate | 1024-byte 实际 gated producer 已证明自然路线；原 event/queue ordering | receiver 原 endpoint import、所有页 guards、实际退休与来源错误 |
| Apple | 原 CopyBatchUpdate→typed input/prefix→原 PagedBatchAccess.submit | 实际 FIFO＋guard 原型可以执行；取消来源不传播，已复现 | 原 worker 接纳/执行前检查与最终结果合并；原 native output 在 eval 后才有 backing |
| Vulkan | 原 CopyBatchUpdate→typed input/prefix→原 PagedBatchAccess.submit | 实际 queue/fence＋guard 原型可以执行；取消来源不传播，已复现 | 接纳前拒绝与已接纳 suffix 结果合并；冷 output buffer 不提供 ready permission |
| CPU+CUDA | 原 graph multi-queue region/slot→上述 source/receiver | 原公共 region/completion bridge 保留 | 同样的 pending permission/错误合同如何由原 bridge 承接，不能用 CPU fallback |
| CPU+Apple | 原 hybrid bindings/region→原 Apple receiver | 已关闭的 ready 联合分页输入/资源桥保留 | late output、pending receiver与跨域最终状态，不能复制 hybrid scheduler |
| CPU+Vulkan | 原 hybrid bindings/region→原 Vulkan receiver | 已关闭的 ready 联合分页输入/资源桥保留 | 原 endpoint/unknown fence、pending receiver及跨域错误传播 |

冷成本：原 layout version/native program/invocation、metadata upload、F1 use pool/页
frame 的 cold 分配逐项归原 owner；metadata cold host wait 明确计入。warm 接缝要求
保原 update 两控制和原 worker/queue、精确引用页，不复制完整 KV、不新增默认对象
锁/原子/队列；尚未有实现，不把这些目标写成已证明 zero-cost。G06 的 topology/
metadata/capture 优化仍留 G06，不在 G03 重开已关闭的数学与性能轮。

下一步仅固定上述 receiver 与 late output 的有限 API/ownership/FIX；实际必要的下层
改动先按本反证登记新 scope。主 owner 仍 SCOPED，不由此自动进入 AUDITED。

已请求取消的原 dependency 接纳检查按
[两 native owner 有限新轮](inference-g03-native-dependency-admission.md)单独登记；
不把这一步检查当完整 receiver 聚合或 pending 页形状完成。

该有限轮现以 `9404af694` CLOSED：两 actual route 取消接纳/晚取消成功保持、原
public65/生命周期、六 host 风险、四 SDK/12 头、定向 Win Vulkan 编译全部完成，
原 fixture/字段机械 FAIL 保留。后继接纳之后才出现的来源错误仍须由本主卡聚合，
pending 页权限/late output 与完整七路线旅程仍未实现，本主卡继续 SCOPED。

原结果槽的聚合观察按[有限新 scope](inference-g03-owned-outcome-observation.md)固定：
源 controls 必须保到 actual suffix 退休，结果 payload/slot 不复制；原 Graph 还消费
completion()，因此只改 façade 的 wait/result 不足。实际 move-owner 原型已通过
Apple 三/Vulkan 五场景，包含接纳后取消、abandon、Vulkan unknown/final fence wait；
完整公开形状/Source CPP retirement/四 source families仍须本新轮冻结并实现。

来源聚合与生产cold nodes已在 f307/cdb8 检查点实现、逐分配OOM、实际四 source/lifecycle/
SDK/宿主风险通过，原控制可回接。当前 compute_batch 有限施工入口转到
[原page初始化前缀](inference-g03-page-initialization-prefix.md)：它固定本主卡需要的初始化
事实，不把ready frame当pending资格。主卡的完整receiver、late output、Graph与七路线
验收继续保留；后继采用这些实际初始化事实与原来源协议，不再另造来源owner。

2026-10-14续接：初始化前缀有限轮已以`15c1a5a36`候选、`1dcc26cc0`terminal CLOSED。
原PagedPageWriter现在必须显式publish(rows)/complete(ready,rows)，新页/回收代际为0，
原unique ready页保committed prefix，read在分配之前验证每个引用。14项/64步、3hostsan、
三交付、原CPU/Vulkan数学消费及专用调用迁移已验，不重复这些门。当前新事实是原controller
仍只接成功ready frame，不能消费已接纳F2 update并证明prefix与actual write coverage合并。
继续既有goal中未交付的组合范围，不重开有限IP轮；尚须固定receiver公开形状/有限FIX，
不得从本段直接开始产品实现。来源聚合和cold source节点仍沿f307/cdb8原实体。

## 当前接纳合同实施批（PF01–PF03，AUDITED）

baseline `503a98344`。compute_batch负责原页/布局组合，compute及两native route只负责
读取原F2 accepted逻辑写覆盖；三source owner原OO审计阶段保留、当前有限增补绑定本卡。
本批固定全部页接纳事实，不先修改数学、原native控制器或Graph产品；那些后续接入仍是
完整G03的必需项，不由本批代签。普通numeric page→reduction与主KV追加共用合同。

PF01：公开PagedPageUpdate{const PagedPageWriter* page, CopyResourceId resource}描述
源update中的目的页；arena.pending_read(layout, ready_pages, updates)产生独立
PagedArenaPendingRead。它保原arena/schema/layout、ready guards及pending页root/generation/
原initialized prefix与布局实际required rows；不把pending变ReadLease，不能传入原ready
bind/run。构造时原页writer必须同arena且已消费原lease，exact referenced set、ready/pending
不相交、pending物理页不重复；重复/shared引用合并最大行需求。descriptor只在构造时借用，
记录复制identity而不借PageWriter地址；原source controls/实际writer寿命由receiver冷槽保留。

PF02：三个原CopyStreamAccess增加covers(update,id,offset,bytes)，复用原input资格与
copy_output_coverage、最后accepted write，拒绝known source/tail错误、不观察/分配/消费。
这是逻辑coverage检查，不签source成功；suffix仍须原native dependency与OutcomeSources。
原Apple observations同步合同保持，无新默认锁/原子/元数据。

PF03：trusted PagedArenaAccess.validate逐原input核root/原writer/generation/resource/
native handle/全page offset与extent、仍acquired且无新代际；按每个pending页required rows
减原prefix求新增byte gap，仅用上述accepted coverage覆盖它。继承前缀不要求本次重写；
不连续写洞、typed subrange/外来root不合格。admit_read先验证全部，再对ready页用原read、
pending页用原ordered read；receiver必须在此前资格化本源的实际native prefix/endpoint。
source observe/reset与resolve/admission之间禁止交错。tensor只对trusted math返回容量，
frame没有public host字节读取、ready转换或另外execution owner。

自然序列：旧页acquire→F2 bind_mutable/producer→submit/take原Update→pending_read（也可
在bind之后、submit之前冷构造）→receiver冷source槽take→validate/原prefix资格→原F1
admit→原suffix→原DependentCompletion合并；失败不发布页/结果，完成后沿原take_mutable与
显式publish(rows)回接。两request、shared page、old frames及多目的资源同样按exact域。
新页prefix0只读实际copy覆盖，不能沿旧探针用“其它布局的ready页frame”提供假权限。

冷热/并发：新frame是显式冷owned layout metadata，保原ready路径和结构不变；最多一个
frame对象、两个实际输入vector和一个exact-domain scratch，不拷贝payload。排序/合并
O((refs+inputs)log(refs+inputs))，validate/admit为每pending一次原input+coverage扫描；
记录真实成本，不称整步零分配。source input/coverage原借用串行；各frame句柄独立可复制。
无依赖/FFI/factor扩张，public ABI允许直接改，不做旧签名兼容。

有限FIX仅PF01–03，冻结后允许实现；不增加第五种copy执行器或在本批修元数据planning。
预声明：exact foreign/root/gen/subrange/重复/superset、shared非尾页、初始化洞与partial row、
多phase/gap accepted coverage、known拒绝不消费、旧ready/native guard及64步追加独立oracle、
每cold allocation OOM与资源预算、原ready回归；Windows对应common/CPU直接consumer，
Air实际Apple/Vulkan原Update接纳并核no poll、3hostsan、source/static/shared和独立头；
CUDA共用source编译/小payload资格沿现有树、禁止nvcc/旧模型重构。当前PF新frame不能由
这些测试签原controller/Graph pending消费完成。普通docs/closure/selector保skip，P0不做P1。
沿原2h检查点和用户常设延长记录累计，heavy串行/4jobs。必要下一接入在本主卡固定原
controller/source槽与Graph生命周期后再授权，完整G03与全部G01–G10范围不变。

PF02审计增补（实现时真实反证）：Windows第2次测试在原ordered_read仍保页时，
CopyBatchUpdate.take_mutable返回了writer。原F1拒绝其立即view/publish，但原接口声明
“无实际native use才归还”没有在移交前检查。PF02固定增加共同/Apple/Vulkan三个原
take_mutable入口的原WriteLease.view唯一性/成功资格检查；拒绝不移动原binding。复用
既有F1，不改F1布局/锁/原子或另建controller，显式withdraw增加一次原常数检查。
第1次测试失败为32KiB fixture遗漏sentinel的1字节，已补齐；两次原FAIL/日志/elapsed保留。

## PF01–PF03 已验证实施检查点（2026-10-14）

本批产品和三个专用测试已实现；[输入资格验证记录](../../../../evidence/correctness/inference-g03-pending-page-inputs/validation-result.json)
保留原日志、每次 FAIL/累计时间、最终 source hash、native artifact receipt、两原 TSan
模型与 SDK consumer 编译命令。四 owner 仍 AUDITED，未把整个 G03 转入 CANDIDATE/CLOSED。

Windows 20 项与原 public64；Air 两 actual native、六 host sanitizer（每组 page20、
native 两 episode、原 CPU copy8）；两 relocated static/shared 各 page20 与三 standalone
头；CUDA actual256-byte 原 gated producer/F2/download＋byte/sum oracle 均 PASS。
初始化洞、shared/non-tail、root/generation/extent、每冷分配 OOM、两原 copy slot/64步、
旧 guard 与原 controls 回收得到定向验证。Windows 最终 receipt 后只有 common detail
头合同注释变化；Air/CUDA final manifest 匹配产品字节，未伪造 Windows 新编译身份。

真实 withdrawal 反证已修于三个原 take_mutable：先用既有 F1 view 的唯一性和成功检查，
拒绝不移动 binding。冷 frame 最多四分配，payload 不复制；coverage/admission 不观察
producer、不新增队列。原 source native fixture 不算 installed native receiver 交付；
host sanitizer 不签 shader/driver，CUDA 沿原 source HYLE_TESTING 树且无 nvcc/kernel。
所有 fixture FAIL、原 worker teardown ASan、autonomous counter 误判和 WSL 输入扫描
中断均保留。没有重型 run 存活；不重复本批、IP、cold source 与原数学已证明门。

下一步继续本主卡的真实 native/Batch receiver、原 Graph completion 聚合与 late output；
在此固定下一有限 FIX/公开消费形状后实现。完整七路线64步/两请求/两slot/独立 reduction、
原 G01/G02 未决门与 G04–G10 均未代签。整体粗估18%–20%，完整 G 项仍0/10。

## 当前原 native receiver 接入批（RC01–RC04，AUDITED）

baseline `9b3cd6217`，沿本卡 compute_batch/compute/Apple/Vulkan 的 AUDITED 实施。真实消费入口
是原 PagedBatchInvocation；原 Graph 接纳原 OwnedResult.completion()，所以最终结果
必须映射在同一个原 result storage 上。公共 source owner 不新增产品机制。

RC01：两路线增加显式冷 PagedBatchReceiver::create(PagedBatchInvocation&&,
max_source_completions)。它只拥有移交的原 controller、一原 CopyOutcomeSource 和
DependentCompletion；不另建 native invocation、worker、queue 或 result slot，不给
原 ready controller 增加字段/分配。prepare_batch 委托原 controller。调用
submit(query, pending_keys, pending_values, CopyBatchUpdate&&)组合实际更新；一个
原 Update 可以含多个原 producer binding/目的资源/phase。take_update 在原聚合观察
引用和全部 actual retirement 结束后回接原 Update，writer/invocation 沿原 API 复用。

RC02：同一个原 controller 的 page binding 重用原 address importer，新增 pending
frame 重载；native view 保原 arena/layout，所有 permission 由原 use pool 精确接纳。
先验证 schema/layout/root/init/accepted coverage 与原 runtime endpoint，再绑定，
最后转移原 source。已知拒绝/OOM/容量不足不消费 caller Update；source 转移后若
原 suffix 尚未接纳，归还同一 Update。不得用其它布局的 ready frame 提供假权限。

RC03：原 suffix 接纳后借既有 DependentCompletion 替换原 OwnedResult completion；
最终 wait/result/completion 都合并 source 状态。接纳后失败/取消/unknown 保所有
guards 和 source 至 actual retirement；不承诺抢占。旧公共结果/聚合观察与尚未回接
的原 source 阻止 slot reuse，close/drain 沿原 controller 并在 suffix 退休后处理 source。

RC04：自然 C++ source/static/shared consumer 与 guide 覆盖真实 pending K/V→原
attention→同一结果/原 control 回接；两请求、两原 source slot、64步数值 oracle，
hole/foreign/cancel-before/cancel-after/unknown/abandon/old-result/OOM 定向风险。
Air actual Apple/Vulkan、三 host sanitizer 与两交付独立头沿匹配缓存树，所有 heavy
串行/4jobs；Windows 只对应公共 common/定向 Vulkan C++ 编译。Graph 真实组合、
CPU/CUDA receiver、late native output及完整七路线仍由本主卡继续，不由单纯
completion 形状或本批两 native PASS 代签。现阶段 P0 只核冷/暖成本和复杂度，
保原 metadata upload host wait 与 native private unknown，不重跑原数学/P1。

有限 FIX 冻结为上述四项。新 opt-in receiver 冷分配绑定原 allocator/budget，
binding 沿原 native import 冷 metadata；submit 中 source take、资格与聚合 publish
不增加 metadata 分配，原 native submit 的已有成本如实保留。原 Update 借用访问
与 receiver 控制外部串行，资源/allocator/context 延续原寿命合同。验证失败累计
在本卡保留，沿原2h检查点及用户常设延长，不新增逐函数施工卡。

RC03 实测增补：Apple 风险第1次失败证明 Completion.wait 的 settle 会释放聚合
proxy，但 OwnedResult 仍保原 storage。因而仅查 DependentCompletion.available
不足以兑现本批已定的 old-outcome/take_update 合同。修正直接借原 ResultAccess
读取同一 storage，在显式 take_update 调用原 begin_result/rollback_result gate；
只在 opt-in receiver 保一个 borrowed storage pointer，原 controller 保其寿命，
回接之前清空它。原 ready 对象/结果槽/锁/分配不变，失败后原 Update 继续保留。
保留此 FAIL，不以放宽原验收条件解决。

RC02/RC03 实测增补：Vulkan ggml native operation 接纳拒绝返回普通 error，suffix
无 actual use；receiver 若先 release 未订阅来源协议，该既有协议会 abandon Update，
再回接只得到空句柄。固定先移动回原 Update 再 release。reusable 同时核原 source
payload 已回接，不能只查协议引用已结束。这是既定拒绝回接/旧来源 gate 的实现修正，
不修改下层 source 协议。Vulkan unknown 首次 test 把既有 aggregate 的 PENDING 当
即时错误，已按原 actual-retirement 合同改测 final wait 的首个 OOM；原 FAIL 全保留。

## RC01–RC04 已验证实施检查点（2026-10-14）

两条 original PagedBatchInvocation 的 opt-in receiver 已实现并通过
[native receiver 记录](../../../../evidence/correctness/inference-g03-native-pending-receiver/validation-result.json)。
Apple F16/BF16 与 Vulkan F16 实际两请求/两原 slot/64步、独立 double 数值 oracle、
fresh prefix/dirty copy/跨页和混合 Q1/17；source 风险 Apple8/Vulkan10；六 host
sanitizer 同 journey/风险/原 ready regression；四 relocated static/shared 同 public64
及八 standalone 头；Windows MSVC 定向 native receiver 编译与 common20 均 PASS。
原 Runtime/queue/结果槽保留，最终 OwnedResult 的 completion/wait/result 同源状态。
接纳拒绝归还 Update，旧结果/尚未回接 source 阻止 slot reuse；两真实实现 FAIL 和
fixture诊断失败在原日志/累计中保留。source 更新期只有最终聚合 wait，metadata
prepare 等待/冷 view 分配/native private unknown 不隐去。宿主风险只签宿主；
SDK journey 无私有头，source风险使用既有边界仪器；两原 TSan 模型未扩张。

四 owner 继续 AUDITED；本批不签完整 G03 或所有 selector reverse-factor gate。
下一步接 common CPU/CUDA 与原 Graph，补 late native producer backing 和七路线/
独立 reduction 最终旅程；原 G01/G02 未决门、G04–G10 与所有失败/成本继续保留。
不重复 PF/IP/source/数学和本批已证明门；所有 heavy 工作按实际 terminal 收尾。

## 当前 common Batch receiver 接入批（BC01–BC04，AUDITED）

baseline `5f496b827`，owner compute_batch；沿既有 compute 的来源机制，不重开原
CPU/FlashInfer数学、native页适配或ready controller。真实入口是CPU与CUDA共享的
BatchProgram/BatchInvocation/PreparedBatch。原队列已有 enqueue_resolved_wait 和
provider enqueue_admitted，可把原来源等待和原数学放在同一个原 QueueSequence。
CPU队列只接已实际完成的inline来源，不偷偷wait异步来源；CUDA资格由原backend裁决。

BC01：公开显式冷 PagedBatchReceiver::create(BatchInvocation&&,
max_source_completions)，持原controller、原PreparedBatch ticket、原CopyOutcomeSource/
DependentCompletion。prepare_batch(layout,update)委托原plan/update；submit(query,
pending_keys,pending_values,CopyBatchUpdate&&)沿同一schema/layout与已验PF规则。
create拒绝旧ticket/外部Graph输出slot，不消费被拒controller；所有冷分配失败保原owner。
原ready对象没有新增字段/分配。一个原Update可含多producer/phase/目的页。

BC02：完整检查typed input、覆盖、原准备代际和资源，然后在原sequence记录实际
来源wait，只有排序接纳成功后才原ordered-read接纳pending页并enqueue原数学。
无接纳prefix的拒绝回接同一个Update；wait或数学已接纳的prefix必须返回其真实
Completion与首个error，保source到actual retirement，不能当普通error丢掉prefix。
借原ProviderExtensionAccess.enqueue_admitted，不新增backend/provider/public工具机制。

BC03：原BatchInvocation已有RetainedCompletion作为公共观察槽，在它内部映射原
DependentCompletion。receiver冷owner用一个既有CompletionRetention接收聚合的最终
状态并保原controller，不创建第二结果槽或另外executor。output()只在原聚合成功和
actual retirement后委托原F1发布；suffix成功/source失败不能发布。take_update检查原
公共观察、全部actual guards和来源引用均退休；旧output阻止原output复用。失败epoch
不自动洗掉错误，关闭与drain沿原Queue/guards/完整聚合，放弃不隐藏wait。

BC04：原CPU ggml page16实际两request/两slot/64步F16/BF16、独立double oracle与
正交page payload→数值消费；初始化洞、精确页/外来layout、旧output、未知异步来源
拒绝不隐式wait、每冷分配OOM/资源预算/接纳拒绝/已接纳prefix错误；Windows定向
common/CPU与source消费，Air或WSL对应宿主风险/三交付按缓存身份。CUDA沿原树小
payload与现有数学artifact资格，禁止新nvcc/kernel构建；若artifact不匹配只记录未完成，
继续真实Graph/late-output等其余原目标，不冒充CUDA math PASS。全部heavy串行/4jobs，
不启动full model/P1/全局构建，仍沿原2h检查点、常设延长和所有原失败/unknown预算。

有限FIX冻结为BC01–04。冷source四分配、observer与owner元数据显式计费；暖submit
不复制KV、不新开queue/线程、不另分配source节点，原provider/plan成本保留。原
controller/queue/context/allocator/source访问外部串行并保至真实退休。完整Graph、
native late backing与七路线仍是本主卡必需后续，不由本批或API形状代签。

BC02实测增补：Windows首次common CPU64失败为已成功退休的inline来源返回空
native prefix；原 enqueue_resolved_wait 要求真实native dependency，不能传空。既有
CopyStreamAccess已逐原phase核成功，只在prefix.pending()时记录原native wait，
inline成功不制造wait/事件/host barrier。保原失败日志和累计，未改下层等待机制。

## BC01–BC04 公共 receiver 实施检查点（2026-10-14）

共同 receiver 已实现，[定向验证记录](../../../../evidence/correctness/inference-g03-common-pending-receiver/validation-result.json)
保产品字节、每阶段输入、原失败/累计、宿主模型与 SDK receipts。Windows/Air 原生 CPU
F16/BF16 两请求/两原 slot/64步和 double oracle、七边界、三宿主 sanitizer、两迁址
static/shared SDK 与独立公共头通过；Windows 原 ready paged/ragged 五测试通过。
CUDA 512-byte 真实 gated producer→原 F2→原 FlashInfer paged 原生数学与最终聚合通过；
原 CPU 数学资格明确拒绝未声明 coherent-host 的 CUDA pinned resource；此负例不签
CPU 异步来源接纳/拒绝或跨域数值旅程。CUDA 只编译 C++ glue/fixture，原 kernel artifact
复用，无 nvcc。

CPU 首次真实 inline 空 prefix 实现失败与第二次 capacity-view fixture 失败保留。
CUDA 首次 native kernel 属性/模块加载与 fixture host gate 互等，原 111.257s 与实际
backtrace保留；同一原 controller 显式预热后通过。WSL旧CPU provider缺 BatchPlanning
的验证副本失败保留，只从canonical同步原已关闭provider，不反向覆盖产品或重开owner。

本批继续 AUDITED，不签完整 BC04/G03、CUDA 三交付/64步或所有 reverse-factor 门。
CPU异步来源拒绝、接纳后来源错误/unknown/abandon、明确资源预算边界、Graph、late native backing与
七路线/正交 reduction 仍必需。原 G01/G02未决门与 G04–G10、失败和 private UNKNOWN
全部保留；只复用本批已证明门，整体粗估20%、完整项目0/10。所有本批 heavy 已terminal。

BC04 续验固定：同一原 CPU64 数值旅程迁入 CUDA 原队列/FlashInfer，保持 GQA8/2
D256、F16/BF16、两请求/两原 slot、Q1/17 与64步跨页，不缩小最终主形状。常驻 K/V
与 query 总payload约400KiB，两个原 workspace 各8MiB；仍为有限小消费者，不运行
模型、P1或新 kernel 编译。 source/迁址static/shared均先核实际匹配缓存；无 nvcc
前置条件不成立则保未决门并继续其它真实主卡工作，不重置前轮失败与累计。


## BC04 CPU/CUDA 批量验证检查点（2026-10-14）

[本批记录](../../../../evidence/correctness/inference-g03-common-receiver-batch/validation-result.json)
证明 CUDA source/static/shared 的原 GQA8/2 D256、F16/BF16、双请求/双slot/64步完整
数值旅程；五项 actual CUDA 来源/未知/错误/放弃及原 CPU queue 拒绝边界；CPU 九项
风险（新增预算拒绝与放弃原槽回收），Windows 与三宿主 sanitizer 均通过。无产品改动，
无 nvcc/new kernel；原 source4/CPU64/SDK、两 native64 和已证明原数学门继续复用。

原159.825s诊断终止、两个 fixture lifetime fatal、错误的“未知=已知失败”判据及全部
累计保留。最终按原public Unavailable/actual retirement合同分别验 known 与unknown；
未知错误最终合并并拒绝成功 output，不改产品或放宽错误合同。CUDA shipping backend
未编入 HYLE_TESTING；source 风险只借既有观测 hook，不代签 GPU sanitizer。

模块仍 AUDITED，不签整项 G03。下一批集中处理原 Graph pending 消费与 native late
output 的剩余有限接缝，再批量验证七路线与正交 reduction；对应 owner 先完成范围/
API/寿命/成本审计，保原 G02 未决门。按用户要求一批实现、批量检查、针对失败修复，
不再按单 helper 切施工卡或重复已证明门。原十项 goal、失败/UNKNOWN保留；本批后
整体粗估约22%，完整 G 项验收仍0/10，不由receiver三交付代签完整G03。

## Graph 与 late output 集中实施批（GL01–GL04，AUDITED）

baseline `19413efdd0427cd21f4431dfa7561291e9034bde`。本批只补本主卡尚未完成的
pending producer → 原 F2 → 原 Graph 页读取 → 原数值结果，不重新验原数学、PF、RC、BC。
真实新事实是 `batch::Invocation::bind(PagedArenaRead)`、两联合 Graph 的同名绑定只认
ready 页；Apple `Results::arrays()` 在原 worker 执行前为空，而原 F2 只认 ready ReadLease。
因此两 CLOSED graph_route_cpu_apple/vulkan 按新 pending 消费事实重开；graph/graph_batch
的 G02 candidate 因新公开合同回退 AUDITED，原未决 Windows gate 与旧 candidate 证据保留。
四现有 AUDITED compute owner 续用本卡。没有新 component、数学实现或执行器。

主消费者仍是 GQA8/2 D256、F16/BF16（各路线资格单列）、Q1/17、双请求/双原 slot/64 步
增长跨页；正交消费者是普通 native 数值 producer → range update → reduction。七路线
全部保留，不以两个 receiver 或 host 显式等待版本代签。既有三 peer 固定源码与
[原 source anchors](../../../../evidence/architecture/inference-single-machine-foundation-audit/source-anchors.json)
仍作为机制分工输入：请求/cache/模型政策上移，native 执行与 backing 留在原 owner。

| ID | 能力/现状 | owner、有限改动与冻结合同 | 验收 |
|---|---|---|---|
| GL01 | Graph 原 ordered sequence 与 final result 已有，pending permission 接缝 MISSING | graph BORROW 原 SubmissionCursor/LeasedInvocation/use pool/retention；trusted projected binding 与 submit callbacks，回调在原 QueueSequence 内先接 source wait，再接 exact guards，最终聚合在原公共 completion 槽内发布；普通 ready 调用无新增 owner 字段 | rejected-prefix 回接、accepted-prefix 不丢失、source 失败无 output，原 ready 回归 |
| GL02 | common batch region slots 已有，public pending 组合 MISSING | graph_batch SELF 一个显式冷 PendingInvocation，移动原 Invocation，保一个原 CopyOutcomeSource/DependentCompletion 与有限 pending input 表；bind/prepare/report 委托原 slot，submit(Update&&)/take_update 自然回接；不另建 Graph/Batch invocation/queue/output | CPU、CUDA、CPU+CUDA 实际主/正交 journey、风险、三交付/独立头 |
| GL03 | 原两 native 联合 Graph 只接 ready，PARTIAL | 两 graph_route BORROW 原 PageSlots/MaterializedNativeState 与原 Graph ResultSlot；增加 pending frame bind、submit(Update&&)、take_update。source 只在显式 pending 冷扩展创建，一 Update 可供同图多个页节点，最终映射同一 Graph OwnedResult。exact node/layout/readers 在 replay 前固定 | Apple/Vulkan 单 GPU 与两联合路线 actual journey、两节点、旧结果、源失败/unknown/取消/放弃/预算 |
| GL04 | 原 native accepted result → F2 late source MISSING | 两 compute_route SELF typed bind_source(span<ResultCopyBinding>, OwnedResult<Results>&&)；每条 binding 固定 resource/output_index/TensorDesc，同一原结果可供多个 K/V 来源，仍只有一个 completion 观察 owner。冷持原 result storage，移动其真实 completion 到原 producer binding。Apple backing 在原 FIFO copy Job 的 preflight 成功后解析，完整物理别名/范围再验后才 blit；Vulkan 可冷解析地址但仍保 producer 排序/成功合同。绝不创建 public pending ReadLease；接纳拒绝不消费结果。复用原 recording/worker/queue/phase guard | actual非零数值 producer、多个来源/phase、foreign/shape/alias/OOM、错误与实际退休，三交付 |

自然调用为 common `PendingInvocation::create(std::move(original), capacity, leases)`，然后
prepare_batch、bind(Q ready/输出 writer/KV pending)、submit(std::move(update))、最终 wait、
output、释放旧 output/completion 后 take_update。native joint 在原 Invocation 上选择 pending
bind 与 submit(Update&&)。native F2 的 accepted output 绑定成功后才转移原 OwnedResult；
回收原 producer slot 须等原 copy/source/结果引用全部退休。各 control 对象外部串行，
原 context/resources/allocator/budget 保至 actual retirement。失败 epoch 不被 prepare 洗掉。

冷热冻结：GL01/02 不扩普通 Graph/Batch 对象；冷输入表按有限节点/输入数预留，暖 submit
不分配 source 协议节点、不拷贝 KV、不新开线程/队列。GL03/04 允许原 native 私有冷 state
保存一个 nullable 扩展指针，扩展只在选择 pending 时分配；这是明确的常数冷 metadata
成本，不能藏成零成本。native 原 tracing/import/结果/同步费用继续如实列账；不因这批
顺手做 G06 规划或 G09 流水。late physical proof 不可伪装成 ready/F1 权限。

P0；预声明 source/static/shared 同 public journey、逐头、对应 Windows common/CPU 与
native C++ 编译、WSL CUDA 既有 kernel artifact、Air 实际 Apple/Vulkan 与宿主三 sanitizer。
原 ready consumer 只跑直接受影响代表；不全局构建、不 P1、不重跑已证明数学/PF/RC/BC。
所有 heavy 串行、最多四 jobs，原 2h 检查点与用户常设延长保累计。每批集中实现再集中
验证，失败只定向修复；不逐函数另建施工卡。适用但未完成的 gate 保 UNPROVEN，不写 PASS。
七问沿本卡既有依赖/FFI/交付边界；新增公开形状须落使用指南，固定原 lifetime/owner，
工具无缺陷不扩工具。有限 GL01–04 全部实现并验后才候选；新架构事实先改本节审计。

GL03 的显式冷入口固定为 `reserve_pending_sources(max_source_completions)`，在原 joint
Invocation 上调用一次，随后 bind(PagedArenaPendingRead)/submit(Update&&)/take_update。
一次 Update 共用全图 source owner，不能按每个页节点复制观察者。GL04 采用批量 result
binding，是因为一个 native QKV producer 的同一 OwnedResult 可以同时包含多个输出；
不能要求用户为 K/V 重复执行 producer，或移动同一 move-only completion 两次。来源的
逻辑尺寸是声明，实际 backing/别名 proof 在 native owner 里兑现，不能造假 Buffer/ReadLease。

2026-10-14 实施中：GL01–GL04 主体已写入工作树，沿本批集中实现、批量检查和定向修复。
Windows 已编译 common/native Vulkan/两联合 Graph 接缝；common CPU Graph 的 F16/BF16、
双请求/双槽/64步真实 pending 消费通过，九项原来源风险与36项原结果回归通过。Air Apple
F16/BF16、Vulkan F16 的实际 Abs 多输出 producer→原 F2→pending Direct attention 同64步
及独立 double oracle通过；producer 返回后至消费间没有 host wait。这只签上述 source
journey，不签 GL03 联合 pending、CUDA Graph、正交 reduction、三交付或完整 G03。

本批真实失败保留：CPU Graph 曾提前对尚无 ready permission 的 typed pending 输入调用
普通 lease admission，现只对本次 trusted adapter 明确列出的 exact input 延后至原 sequence
回调，其余 ready 检查仍执行。Vulkan 原资格检查曾拒绝所有 managed producer；实际源代码
证明非 async 原 NativeWork 在 submit 返回前将整个 resident graph 与原 fence 提交到同一
Runtime compute queue，F2 使用同一 queue/submit lock，因此允许这个固定前缀。async、
needs_drain、IREE 未证明前缀继续拒绝；原 producer checked error/unknown/cancel 与实际
retirement 仍参与最终 source outcome，未插入 host wait 或换成更简单 producer 回避失败。
原 Windows/Apple 编译失败、fixture警告与全部累计保留。全部 owner 仍 AUDITED，待上述
剩余消费者和适用风险/交付通过后再冻结 candidate，不能把本段或 selector READY 当全门 PASS。

[本批 source 检查点](../../../../evidence/correctness/inference-g03-graph-late-source-checkpoint/validation-result.json)
保逐次输入 manifest、真实硬件身份、全部编译/消费失败与累计；Windows 九项 copy-range
测试还验证 logical-only 来源不能伪造物理坐标/写目的权限、未知来源仍不得绕过已知目的
hazard，以及 backing 解析后的真实部分别名必须拒绝。该记录明确为 partial implementation，
不代签新增生命周期/宿主 sanitizer/三交付/selector reverse-factor 或整项验收。

## GL 集中消费续验检查点（2026-10-14）

Air 的 CPU+Apple F16/BF16、CPU+Vulkan F16 已通过实际 native Abs 多输出→原 F2→联合
Graph 两个页节点共享同一 Update→必需 CPU 前后段的64步/双请求/独立 double校验。
active Q 2/18沿原固定输出合同冷编译两图，各两原 Graph槽、合计四个冷槽；两原 copy/
producer槽不变。每步释放所有冷变体的旧导入再取页 writer；不得缩成固定Q或把四槽报成两槽。
Vulkan 首次未执行变体重复prepare失败证明原 output引用基线直到driver才seal，现移到原
冷 prepare_batch 的输出/transfer桥全部固定后seal，未删除旧结果、引用数或实际退休gate。
Apple 两个 fixture错误（不存在的drain调用、equal bytes不同layout identity）及固定图换
shape拒绝、Vulkan idle cold输出FAIL全部保留。

CUDA共同 Graph 的 F16/BF16、双请求/双槽/64步actual消费通过。原 fixture错用foreign Context
资源拒绝保留，修为同一wrapper Context的原资源/queue/F2，显式 storage domain；无产品
弱化/no-op tensor。WSL首选facade预检会触发无关ggml CUDA kernel，未启动；定向维护factor
只编译原Graph/Batch/backend/FlashInfer C++，原kernel artifact复用。header CUDA传播夹具
漏链与全部configure成本保留。CPU+CUDA source尚在诊断；原完整组件/三交付未代签。

两native正交非attention source通过：原Abs四输出、共享一个completion、两phase反序重排，
首个绑定output仅在后phase使用，64步/双copy槽与独立排列和CPU sum数值oracle。最终F2
成功后显式256字节host staging算成本；不把它宣称异步GPU reduction。原遗漏公共header
编译FAIL保留。完整新风险/sanitizer/SDK、原G01/G02未决门与G04–G10仍必需，owner继续AUDITED。

[续验记录](../../../../evidence/correctness/inference-g03-graph-late-consumers-checkpoint/validation-result.json)
保上述 source输入/原失败/机器身份和累计。CPU+CUDA 的同64步现已实际通过：原公开联合
compiler源码 TU与共同 PendingInvocation，真实 CPU ggml F32 Neg/Cast前后段与原FlashInfer。
这次维护消费复用五个冻结CPU ggml archive和原AVX2 capsule；完整组件archive输入资格与
安装交付仍UNPROVEN，记录将此条明确为diagnostic，不能由数值PASS代签完整factor/SDK。
原provider子域拒绝、OpenMP/static链接/运行时capsule缺失与全部构建成本保留。当前Windows
新联合public TU/link、更新后的CPU Graph64/九来源风险/36结果回归也通过；无本机GPU执行。

审计裁决：GL01–04 作为一个有限实施批固定；所有既有 source/数学/ready controller 为 BORROW，
只补实际 pending 组合和 late backing 生命周期。拒绝第二 executor/result slot、whole-arena
权限和同步 ready 冒充。原 G02 candidate 5ce0bed8/1cd08939 与未决 gate 保历史证据，
不能用本轮状态回退擦除。source anchors 与上述当前入口逐项回映后，允许本批产品实现。

GL04 宿主 TSan 新事实：Apple late result 的原 producer Completion 由 FIFO preflight 在
原 copy slot observations 锁内观察，CopyOutcomeSource 的实际退休订阅却在锁外修改同一
Completion lifetime；64步 source 消费实际报告竞态。修正仍归 GL04：trusted source access
在同一个原观察锁内完成资格检查、非阻塞观察和 retention attachment；阻塞等待先沿原
Update/native recording 完成，不能持该锁等 worker。只增加 opt-in cold source control
callback，不改变普通 Completion 的外部串行合同，不增加 queue/结果槽/工作线程。
首个联合风险 TSan 报告另为测试 fault allocator 非原子计数，已改为并发计数夹具；该失败
与真正产品竞态分别保留，未通过的配置仍未签章。

## GL 生命周期与宿主风险检查点（2026-10-14）

[本批证据](../../../../evidence/correctness/inference-g03-graph-late-risks/validation-result.json)
保存 source/Windows/六宿主配置的所有尝试、输入、失败、累计、instrumentation 与原 TSan
模型 SHA。common CPU Graph 九风险、两 native late result 各十二风险、联合 Apple 八/Vulkan
十风险通过；原 ready joint 两路线亦通过。最后产品字节在 Apple/Vulkan 各 ASan/UBSan/TSan
下的六个定向程序全部通过，含 source/joint64 的真实数值消费者。原 Apple source tests 在
同步修复前通过，未冒充修复后的新增普通 source 资格；宿主风险不签 LSan/shader/driver。

TSan 发现的 fault allocator 夹具竞态与 GL04 真正 completion attachment 竞态分别修复，
原报告/失败/累计保留。Windows 最后新风险编译/CPU 九风险、Graph64、原 receiver 九风险与
36 outcome 回归通过。owner 仍 AUDITED；新安装 static/shared/public-header、common GPU
Graph 与 CPU+CUDA 生命周期、完整联合 factor/artifact、部分预算/非法 native alias 门和原
G01/G02 未决门均未代签。十项整体仍粗估22%，完整 G 项0/10；本批所有 heavy 已 terminal。

## 完整 CPU+CUDA 组件资格的缓存裁决（2026-10-14）

原 maintenance 数值消费者的 frozen CPU archive 仍只签 diagnostic。实际 Ninja explain 已
证明当前完整 `graph_route_cpu_cuda` 闭包中，ggml 准备源码时间戳晚于旧对象，precise-unary/
CPU capsule 部分对象缺失，另有编译命令变化；不能只复制旧 archive 冒充当前组件/交付资格。
普通小 CUDA Graph 风险仍只用匹配的原 FlashInfer kernel，并已通过八项 actual lifecycle。

因此本次原三交付门允许一次明确的完整 CPU+CUDA 组件依赖构建：复用原 source/static/shared
树，构建 graph/graph_batch/graph_route_cpu_cuda/compute_batch 及其真实依赖；允许其中
必需的原 ggml CUDA translation units 重编译，不改原算法、vendor、工具或构建系统，不跑
全模块测试/P1/full model。此例外只用于补完整组件身份，不能扩成日常无条件 nvcc；相同
input/tree/toolchain 通过后复用。各 profile 原 configure/失败/累计保留，heavy 串行/最多
四 jobs，2h 检查点与用户常设追加时间授权继续适用，期间不修改产品输入。

## GL 安装交付与剩余资格检查点（2026-10-14）

[本批记录](../../../../evidence/correctness/inference-g03-graph-late-delivery/validation-result.json)
证明 Apple/Vulkan 各 relocated static/shared 的 CPU Graph64、native late source64、联合
Graph64、正交 reduction64及四 standalone public headers；消费者只用 installed 产品头，
shipping 对象无 HYLE_TESTING。CPU+native 联合新增共享 metadata exhaustion 的 cold owner/
pending bind 拒绝在 source 与三 host sanitizer 下均通过，Apple 九/Vulkan 十一风险。

CUDA 新 Graph 八风险实际通过；known/unknown/late source 错误、abandon、旧输出、页回接
与原 CPU Queue拒绝保留原合同。原 CUDA native cancel 返回 Unimplemented；取消拒绝后
不虚构退休，真实工作成功后才能取结果。原 Update 的 request_cancel 同时关闭 slot，仍
不得成功 withdraw writer；不把这个负例写成 native preemption 支持。最初错误的取消
断言和触发无关 kernel 的 preflight失败保留。实际 readonly native result 用作 destination
的新增负例已写入本批测试，尚未运行；独立 mutable native alias import 没有公共入口，
不为测试伪造原结果 allocation 的可写权限。

完整 CPU+CUDA static SDK 正按上节缓存裁决构建；尚未通过，不由 maintenance cached CPU
archive代签。shared 及必需 CPU/GPU fragments 的剩余 lifecycle 仍需验证。owner 继续
AUDITED；原 G01/G02 未决 gate 与 G04–G10完整保留，整体粗估24%，完整G项0/10。

GL 收口复核固定两个剩余证据格：一是两个独立 accepted native OwnedResult（不能用一个
结果的多 outputs代签）经同一 F2 Update进入原 receiver；二是真 required CPU+CUDA 前后
数学 fragments 的 source error/abandon/old-output/actual retirement。两者都沿本卡原公开
形状与 GL01–04 实现，不增加能力/组件/执行器。前者小非零双 Abs source→两个 joint page
节点，独立 constant-value oracle及第二来源取消失败/放弃；后者同 gated CUDA 来源，但
真实 CPU Neg/Cast与Cast/Neg、双行 double oracle，沿完整原 factor构建，禁止借 frozen
archive或纯 GPU attention风险代签。san只定向选择改变的风险夹具。

P0冻结账：common `prepare_batch` 会清 pending 输入，随后每次 bind会为 native readers 与
lane表各请求一次 bounded metadata（实际分配至少两次/输入，另有原frame metadata）。
外层 reserve不等于内层向量冷一次后永不分配；承诺只限 warm submit 不分配 source协议
节点/不复制KV/不开新queue。重复 prepare/bind账保留，由 G06 再处理，不能本轮偷偷改
算法或签全 decode零分配。native原 metadata table上传/host wait、两形状×两slots＝四
冷 Graph slots、正交256-byte staging、private UNKNOWN均保留，不重跑原数学/P1。

## 完整 CUDA 交付与独立多来源续验（2026-10-14）

[新增证据检查点](../../../../evidence/correctness/inference-g03-full-cuda-delivery-multi-source/validation-result.json)
证明完整原 CPU+CUDA 闭包的 static/shared 构建、安装、迁址和公开消费者均通过。两交付各
执行 pure CUDA 与 required CPU+CUDA 的 F16/BF16 同64步双请求、双槽、跨页和独立 double
oracle；各四 standalone public headers。消费者没有源码/producer build include，五项
shipping object命令无 HYLE_TESTING。原 frozen archive消费仍为历史 diagnostic；本次不借
该 workaround，实际依赖重编成本与输入逐次保留，两profile累计1277.770秒。

readonly native result用作destination的新增负例在 Apple/Vulkan source及六 host sanitizer
配置均通过。两个独立 accepted native Abs OwnedResult通过同一原 F2 Update进入两个联合
page节点；实际非零oracle、第二来源取消失败、abandon在两source与六 ASan/UBSan/TSan
配置均通过。夹具曾有 macro编译警告、输出长度错误、Vulkan回收source前状态预期错误；
已定向修正，全部原失败和累计保留，未改产品成功/失败权限。回收原Update之后 failed epoch
仍必须拒绝，不能由回收前的Unavailable推断产品可复用。

真实 required CPU+CUDA 前后数学片段的八项 source lifecycle已启动，正在原 source tree
构建完整factor及必要原依赖，最多四jobs；未通过前不签该门。其他已通过产品字节一致的
数学、host风险和installed证据直接复用。owner继续AUDITED，未冻结candidate、未签完整
G03；原 G01/G02 未决验收和 G04–G10保留。整体仍以约24%粗估，完整G项仍0/10。

## GL 实现停止与候选准备（2026-10-14）

[最后 source 证据](../../../../evidence/correctness/inference-g03-source-joint-final-checkpoint/validation-result.json)
已补完整原 CPU+CUDA factor的公开 hyle::graph源码消费：F16/BF16 同64步/双请求/双槽/
跨页与double oracle通过，没有explicit compiler TU或frozen CPU archive。required CPU
Neg/Cast→CUDA paged→CPU Cast/Neg八生命周期通过；最终 pure CUDA八项回归、改动后的
MSVC native result/joint夹具编译也通过。全部重型任务terminal，无新增产品修改。

首joint夹具将 Gate释放放在submit返回后，原GPU→CPU replay允许checked host wait，
因而在实际cudaEventSynchronize处挂起。backtrace、-15/94.718秒与全源码构建551.340秒
保留；夹具现于原非source Graph event同步点先核producer guards仍live，再释放Gate。
joint放弃是在checked boundary之后放弃returned result/owner；pure CUDA另验真正pending
abandon。两种原公开语义不混报，不并发析构正执行的外部串行控制对象。独立用户授权
red-team复核确认该反证合法，未发现新产品缺陷或G03 MUST；有限实现到此停止扩张。

| 七问 | 本批结论与位置 |
|---|---|
| 第三方 | 原 dependency lock/版本/许可证和native Provider不变；FlashInfer、MLX、ggml原数学BORROW，实际backend/domain拒绝保留。原三peer源码与需求回映沿本卡及十项源码审计，不以SDK构建行新增能力。 |
| 完整度 | GL01–04全部兑现四原路线/三联合公开journey、来源错误与actual retirement；双独立native producer、两page节点同Update、正交range/reduction补齐。未准入Vulkan BF16/async/IREE等域明示，不代G04–G10数学/规划/流水。 |
| 内部复用 | 原Invocation/Graph slots、F2 controls、原ResultStorage/唯一completion、QueueSequence/原FIFO/native fence、F1 exact guards与DependentCompletion；无第二executor/queue/result slot。 |
| 性能 | 本批P0结构审计：nullable cold扩展/有限输入表；submit不新建协议节点/复制KV/新queue；bind有bounded分配，Apple late backing实际物理重验及原观察锁收费，metadata/crossCPU waits与private UNKNOWN保留。未改数学、不以计时签parity，不重复旧P1。 |
| FFI | 同名公开C++机制；无C/Native FFI、无第三方原生类型泄露到新增通用入口。 |
| 三交付 | source与Apple/Vulkan/完整CPU+CUDA迁址static/shared实际public64、独立头、shipping无HYLE_TESTING；最后完整source新factor及8档风险见上述检查点；Windows只签C++编译和CPU。 |
| 易用 | batch-attention指南列自然prepare/bind/submit/take_update、两独立producer与multioutput、拒绝不消费、原结果/旧guard/失败slot、storage identity、cancel拒绝/checked等待及bounded metadata成本。 |

scope/architecture/correctness/delivery/usability_docs/red_team/engineering_performance的
本批适用门已具实际证据。冻结commit后只补candidate身份/LOC/七门裁决，不再改产品。
原G01/G02未决整体验收不能被本批销成PASS；G04–G10保留，goal继续active。

## 固定 candidate

`11b2403e794b6eed2c89418d72b8c89aeaaac746`，八当前owner合法AUDITED→CANDIDATE；产品/API/算法/成本已固定，
上述同字节风险与三交付证据复用，只补身份/LOC/最终裁决。无存活重型任务。
