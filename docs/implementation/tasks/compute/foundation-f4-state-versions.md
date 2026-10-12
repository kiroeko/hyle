# F4 显式不可变资源集合版本

## Closure card

- component / owner：新compute_state。已在1f2bd596b登记QUEUED；共同schema/集合版本/容量/发布由
  compute_state拥有，复制计算/原native退休保持F1/F2与各原route。不让Compute/Mem反向依赖state。
- baseline：bc9a689f983bfa5bf9da9a30019ff58f033378f0；A已CLOSED，不重开其原签范围。
- trigger：[固定peer研究](../project/foundation-f45-reference-research-2026-10.md)、
  [设计F4-01..06](../../../architecture/compute-state-paged-design.md)及用户明确活动goal授权。
- 正常consumer：显式readonly资源集合→多个有界draft→未改成员分享/pure final直接replace→
  只对dirty共享成员沿原F2 clone→成功ready全部成员→expected-base发布；旧版本/reader/native
  使用者保活，失败/取消不发布，不伪rollback实际已写prefix。
- 主纵切：recurrent FP32、conv window、KV块/布局和不可变host sidecar组成的显式完整状态。
  正交纵切：异构渲染资产集合候选更新/保留旧集合及分段矩阵共享/dirty更新。
- public journey：schema/opaque entry→StateStore有限目录→StateVersion/StateDraft→原ReadLease/
  WriteLease/typed F2 invocation→可观察CloneTicket/独占writer→ready Version→StateHead。
- scope：仅F4六FIX，冷容量/元数据与readonly完整成员；不把collection版本嵌到所有Tensor。
- non-goals：模型/RNG/token/checkpoint含义、任意过去rollback、业务事务、模型格式、page/LRU/
  admission策略、F5数学、万能Queue/runtime、新copy/math kernel、worker/reaper、公共工具重写。
- 依赖：compute/F1/F2、compute_ops、mem、util_core/error。native types只从其原factory进入
  typed clone adapter；不得把MLX/Vk opaque layout raw-clone，转换由consumer显式选择/计费。
  直接消费者只有新state fixture、普通四route state consumers及其必要package seams。
- 性能：P0，目录O((V+D)E)、分享零payload copy、dirty O(bytes)、已ready token发布O(1)短monitor；
  backing根去重、clone pools、host metadata深复制与native未知成本分账。无自动P1。
- 机器：Windows CPU/CUDA/Vulkan，WSL CPU/CUDA与适用sanitizers，Air CPU/Apple/Vulkan；
  Air先核对实际Mac16,12硬件/toolchain，Mini不用。WSL软件Vulkan仅构建/包。
- 预算：两轮有限owner审计；六FIX；每gate configure300/build1800/test120/consumer180秒；
  sanitizer build1800/test120。原失败累计，不因日志/提交/remote copy重置。重型门串行，
  库j4≥12GiB/j2≥8GiB/j1≥4GiB、test1，复用source cache但binary不跨配置共享。
- 预声明验证：schema/ID/不同store/base/完整成员/描述编码/overflow/容量/metadata OOM；
  share/replace/host-owned snapshot/old versions/导出reader及native-use容量；dirty clone仅实际
  dirty bytes、原ready/reset/独占writer/producer；冲突、失败prefix/cancel/abandon/未知/
  未启动suffix、不提前回收/close/collect/drain，不关闭caller Context。
  主/正交public consumers，四route实际source/static/shared、公开头/缺路线/真实迁址/loader。
  新store-local publication/record回收并发适用ASan/UBSan/TSan；原GPU数学不由san代签。
  每轮docs/closure/diff，registry明确previous-ref/生成status；不用queued identity触发全库。
- stop：预算到达或原权限/退休不能满足该旅程则BLOCKED/INCONCLUSIVE；系统合同新事实退回
  SCOPED/AUDITED；固定scope外新feature不顺手加入。目标完成后再做用户要求的剩余基础库统计。

## 当前阶段

AUDITED。两轮owner/consumer审计已结束，下列公开对象与生命周期冻结后实施六FIX。
本checkpoint仍无target/header产品代码；实现与状态晋级分开提交。

## 能力初审与已确认复用

| 能力族 | 层级/来源 | 当前覆盖 | 有限动作 |
|---|---|---|---|
| 成员readonly/独占writer/native-last-use | 原子；F1实际闭环 | COMPLETE于原F1范围 | BORROW ReadLease/WriteLease/LeaseUse，不复制引用或fence协议 |
| 批量dirty bytes copy/producer/actual retirement | 复合；F2及peer copy source retention | COMPLETE于原四F2范围 | BORROW四typed Program/Invocation/Submission与原reset/drain |
| schema/不透明成员身份/完整immutable集合 | 复合；SGLang conv/temporal/sibling，资产集合 | MISSING | SELF有限schema与owned目录/host sidecar |
| 多branch分享/替换/发布冲突/旧版本 | 复合；vLLM CoW、llama有限snapshot，资产更新 | MISSING | SELF bounded draft/version/head，不解释模型恢复 |
| 集合容量/backing去重/clone生命周期 | 复合；两消费者重复glue | PARTIAL | SELF目录/clone记录；原F1实际identity/原native退休 |
| token/position/RNG/MTP接受/页表选择 | 应用策略 | OUT_OF_SCOPE | APPLICATION consumer显式列入成员或业务key |

第一轮owner输入：ReadLease::adopt只消费成功且禁止未跟踪writable alias；Readonly导入不可
promote为writer。WriteLease::publish要求所有admitted uses成功退休。
F2 output只覆盖成功last-write范围；取writer需要get output→drop submission→原reset→try_write。
LeasePool析构会关闭domain，新native use/try_write拒绝；clone pool owner必须随F4拥有对象保活。
LeaseAccess::backing返回真实pool根/resource/fullcapacity；同pool不同entry不能按entry重复计费。
collection版本generation不同于单F1 block generation；opaque IDs不能用裸ordinal跨schema伪造。
原native owner/alias合同不允许F4把外部仍可写的array/Vk allocation改名immutable。

## 有限FIX（实施前冻结API后补验收）

| ID | 工作与验收 | 当前 |
|---|---|---|
| F4-01 | schema/opaque身份/有限目录与完整成员；host sidecar拥有，不引用caller临时span | UNPROVEN |
| F4-02 | 分享/pure output替换/原F2 dirty clone；编码/alias/跨资源非法拒绝；不会全集合copy | UNPROVEN |
| F4-03 | 全成员成功ready、expected-base冲突不改current、旧版本/native use保活 | UNPROVEN |
| F4-04 | producer/partial error/cancel/abandon/unknown/未启动suffix，有限记录/backing真实回收及close/collect/drain | UNPROVEN |
| F4-05 | recurrent+conv+paged布局与正交资产集合public consumer，显式snapshot/恢复限制 | UNPROVEN |
| F4-06 | 四route自然typed入口、三交付/新头/缺席/迁址、P0已知/未知账/guide及红队 | UNPROVEN |

七门目前UNPROVEN。主/正交body都须沿普通public C++ API，不以private测试hook代签。

## 第二轮审计：API / owner / lifetime freeze

- StateSchema::create复制有限StateEntrySpec；resource成员有TensorDesc/实际byte extent和可选
  resource约束，metadata成员有明确byte长度。允许空schema，不接F1不能表达的零byte资源。
  StateEntryId保留schema owner与ordinal，只由schema.entry产生，不能跨schema伪造/ABA。
- StateStore::create预备V/D/retained-block目录，StateStoreOptions默认V16/D8/C4/retained-block128、
  backing128MiB/host snapshot1MiB，所有上限可冷配置并checked overflow。没有自动eviction。
  metadata通过inject allocator/ResourceBudget；外部已有backing是观察去重账，不重复reserve。
- StateVersion是共享immutable目录handle；member返回原ReadLease，不复制payload。StateIdentity
  保留原version owner/generation；复制O(1)，token仍占版本容量。导出ReadLease/native use可比
  目录更晚释放，因此store另保有每F1 block的readonly tracking anchor，collect只在该anchor
  成为唯一reader后移除；同pool根按LeaseAccess::backing完整capacity去重，不能按entry累加。
- import完整预检IDs/重复/遗漏/描述/资源/总账后复制readonly handle；host metadata深复制成
  自有readonly bytes。失败不消费caller source。Tensor/native import合法性沿F1原readonly
  carrier；caller未停writable aliases不受此API救济。旧readonly数据不会被draft改写。
- begin(base)自动分享成员；StateDraft为thread-compatible独占可变candidate。replace(id,ReadLease)
  接成功pure output，replace_metadata复制新host snapshot；不强制clone原initial/final状态。
  finalize只在全部成员ready/无pending writer/原错误成功时生成版本；未ready/容量满保留draft。
- StateHead::create(store,initial)为可选store-local头；current返回version共享owner，
  publish(expectedIdentity,candidate)检查same store/schema与open，在短monitor内切换，旧owner
  在锁外释放。冲突不改变current/candidate，publication无GPU等待/数学/IO。
- StateCloneRoute::prepare(store,resource/capacity,原CopyBatchProgram,typed create-invocation factory)
  冷创建有限原F2 slots与目标LeasePool。factory只冷调用，不保留执行回调。共同core仅保存
  对原typed objects的有限poll/cancel/drain/destroy生命周期桥，不增加native调度/Queue。
  默认每slot完整dirty成员一次原F2 phase/range；不clone全集合，不半截伪编码块。
  定向实现核对发现原F2 request_cancel关闭该copy invocation，reset不会重新开放。route因此
  保留typed Program与冷create factory，错误/取消已实际退休的槽显式StateCloneRoute::rearm
  才可重建原invocation；不在submit隐式重建、不开caller Queue/Context、不保留数学执行回调。
  此冷恢复有原F2 metadata/未知native分配成本，OOM保留原退休槽，unknown不能rearm。
  冷factory仅借用稳定native endpoint，限定trivially-destructible capture，不持StateStore/
  Version owning handle制造反向环。route.report是原每slot最近episode快照，shared program
  metadata一次，rearm切换原slot身份，不能当lifetime累计或整个状态系统总成本。
  report在native工作/显式wait observer活跃时返回Unavailable，不并发读取原thread-compatible
  F2 report。显式wait由每个原slot的host observer gate串行，close/collect不等待该gate。
- draft.clone_for_write(id,route)返回CloneTicket：保留draft/base和原native slot，非阻塞poll或
  显式wait；源必须是成功readonly member，目的fresh独占。原F2 output→drop submission→
  reset→try_write成功后才给StateWriter；此顺序不能借file-stream私有take_mutable绕开权限。
  StateWriter持draft/store owner与独占WriteLease；publish借原F1检查全部writes后接回draft。
  若原native update消费writer，可complete同原block/generation的成功ReadLease；错root拒绝。
- pending clone/writer阻止finalize；取消/abandon标记candidate未完成并沿原F2真实退休；
  registered slots没有反向持draft/store的owner环，未接纳/未开始suffix不持producer。未知完成
  仍不复用slot。store.collect非阻塞poll abandoned native slots+tracking anchors；drain仅
  等本store注册copy工作，不关闭caller Context或等待业务reader。close停止新draft/clone/head
  发布，保留旧版本readonly消费；clone池跟随store/version/writer等F4 owners，不提前关domain。
  最后F4 owner析构无隐藏等待，已接纳native backing沿原F2/F1实际退休继续保活，原resource/
  Context/allocator依然按公开borrow合同由caller保活。
- common header state_versions.h及typed clone header state_clone.h归compute_state；private
  util_core/refcount与原F1 ledger，无Graph反向边。正常用户只用hyle::compute；新factor进入
  Compute facade，native typed参数依原已选route。测试只取CPU backend以验证原F2 bytes copy，
  不因copy消费CPU数学默认Provider全目录。

## 首批实施事实（非候选/终局）

[共同层首批](../../../../evidence/correctness/foundation-f4-core-first/README.md)实际Win static
19项+三个公开头通过。已接schema/有限目录/backing ledger、host snapshot、share/replace/head、
原typed F2 clone/poll/wait/writer/rearm；dirty16bytes/一次CPU native copy，accepted CPU producer
DataLoss阻止ready版本、旧base保持。原配置/编译FAIL及首次ctest零匹配FAIL均保留并更正，
同树累计预算没有重置。不能据此签四route/完整consumer/sanitizer/SDK/P0/红队。
Compute façade新增root的授权与交付沿[独立窄卡](foundation-f4-package-seam.md)处理，原factor
cpp/已签F1–F3机制不改；活动状态仍AUDITED，有限六FIX尚未全部验收。

第二批consumer：Win CPU/CUDA的GDN/conv Direct+Graph、五成员state与资产旅程已首跑通过；
Win Vulkan与Air Apple增加原生readonly pure output实际source endpoint clone后通过，见
[Win Vulkan](../../../../evidence/correctness/foundation-f4-source-windows-cpu-vulkan/README.md)与
[Air Apple](../../../../evidence/correctness/foundation-f4-source-air-cpu-apple/README.md)。
CUDA锁定conv需channels%128==0，fixture从32改为128；native取消不可抢占返回Unimplemented
不能当提前停止；Apple array import为逐object readonly资源身份，必须用真实endpoint冷计划/
保活source，managed conv/KV域可用原F1固定池。Vulkan按原prepare/staging copy接口接线。
原fixture拒绝/编译失败保持，原SDK/Provider不升级、不改旧factor源码。页layout只是F4元数据，
F5 attention并未由此实现；这些source旅程不代签sanitizer/三交付/故障完整格/P0/终局红队。
普通指南见[State](../../../user/state-versions.md)。剩余Air Vulkan、WSL/source与额外适用风险继续按卡。

Air Vulkan与更新后的Win CUDA同一consumer完整旅程已实际PASS：
[Air Vulkan](../../../../evidence/correctness/foundation-f4-source-air-cpu-vulkan/README.md)、
[Win CUDA](../../../../evidence/correctness/foundation-f4-source-windows-cpu-cuda/README.md)。
共同risk fixture现24项实际PASS，新增pending write禁止publish、cancel拒绝未启动suffix、
释放caller Store/route后版本/native admission保持pool开放，opaque readonly可分享但raw clone
明确拒绝、不同resource约束拒绝。candidate前产品与b1e37d75c相同，仅补tests/consumer/guide。

当前实现有限FIX已接线固定：F4-01 schema/ID/完整目录+snapshot，F4-02 typed dirty clone/share/
replace，F4-03 ready版本/head冲突+old versions，F4-04原native退休/取消/producer与bounded
records，F4-05四route主/正交ordinary consumers，F4-06自然入口/指南。准备提交不可变candidate；
WSL/三sanitizer/installed交付、更多native fault适用风险、P0/LOC及candidate后红队仍待裁决，
这些未运行项不写PASS。

正常草图：schema/Store→import complete members→Head.current→begin(base)→share/replace或
clone_for_write→poll/wait取得writer→原数学/byte更新→publish/complete→finalize→
head.publish(base.identity,candidate)。recurrent/conv/KV与资产集合只有schema/body不同。

原Readonly ReadLease只表示成功导入/已publish权限，不表示后来某个只读数学任务失败能污染
payload。F4不以任意read task状态替代writer成功证明，也不解释native unknown回滚。
新store publication/record生命周期并发要独立TSan；旧F1/F2原机制只对新public journey定向验证。

七问冻结：原锁/许可KEEP，两个纵切/三peer共同owner证据准入；BORROW已签lease/copy/native；
P0固定目录与有限ledger冷扫描，warm native不加版本字段；同工具链C++无新FFI；四自然route/
三交付、新头/缺路线按实际selected seams；guide必须解释完整schema、borrow/readonly、容量、
expected-base冲突、实际prefix/unknown和opaque不能raw-clone。没有timed性能胜负结论。

QUEUED identity只新增一条未实现声明，未改变原产品闭包/头/target；selector对中央manifest
保守选择全图，原plan在.scratch/foundation-a/f4-queued-plan.json。按用户要求及本卡语义仅做
closure中央身份/顺序与docs检查；不启动未受影响模块/全目录构建，不改公共selector来缩表。
