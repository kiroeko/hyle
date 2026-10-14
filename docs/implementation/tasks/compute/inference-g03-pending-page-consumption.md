# G03 已接纳页更新到原 attention 的组合范围

状态：AUDITED，2026-10-14；当前owner compute_batch；续接baseline `1dcc26cc0af4c6c68a4c9ab330fbd8088dc6aaa9`（初始baseline `c8ea4cc31`保留）。
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
