# G03 原 F2 update 所有权与受控 pending 输入

状态：AUDITED，2026-10-14；owner `compute`、`compute_route_apple`、
`compute_route_vulkan`；baseline `f764a1053`。compute_batch仍按
[主卡](inference-g03-pending-page-consumption.md)SCOPED裁决完整receiver/页/late native输出。
本卡是必要的F2 source权限接缝，不签整个pending producer→attention或G03。

触发：原单个CopySubmission持接纳/退休记录，原CopyInvocation持完整writer与几何，
后继batch缺可安全持有并核原binding身份的typed组合。已有pending CUDA F1→F2路线
完整，ready Apple来源也已关闭；不得重造它们。真实需求是更新到原attention/reduction
时保原writer及精确范围、借原native prefix、在失败/unknown时不发布ready。

[实际move-owner反证](../../../../evidence/architecture/inference-g03-pending-copy-owner/README.md)
已证明移动原两个controls到唯一观察者可跨线程handoff、回收原cold slot，五个Apple
成功/取消/前缀失败/放弃/closed场景通过，16 bytes、不增加原对象字段/原子/锁/queue。
fixture显式wait只作寿命证明，不能代签pending读取或其它平台。普通数值producer到
readonly reduction作为反偏置，页/请求政策仍在消费者。

预声明：原source单元/真实直接consumer；CPU与tinyCUDA、actualAir Apple/Vulkan的
source权限/生命周期资格分开；foreign invocation/submission/ID/源-only或未写资源拒绝、
保成功页untouched bytes、在途无public ready、原prefix/取消/unknown/放弃/closed、
回收同一slot/固定metadata、原LeaseUse与native prefix；适用hostsan、source/static/shared
公共consumer与独立相关头、定向Windows编译。全部只选原F2/真实消费者，不跑全模型/
旧全局矩阵/已闭数学P1，不改公共构建/仪器/checker。

P0：新的opt-in owner仅组合两move-only原handles；普通F2 layout/allocation/原子/锁/
copy/wait不变。typed身份/范围/原prefix证明与权限admission复用原F2/F1 helpers，bounded
phase扫描如实计；没有新增数学算法。原native metadata/private成本与全部失败保留。
调查/实现/验证各2h检查点，沿用户常设授权追加，预算/失败不重置；Windows canonical，
Air实查Mac16,12，heavy串行/最多4jobs，禁止本地heavyGPU/full model。

## AUDITED前的固定问题

有限Source方案选择移动invocation+submission，不能只复制opaque state并把原control
留在caller，从而制造跨actor共享观察者或默认metadata。原controls的实际相同state/接纳
证明必须在factory核，不拿两个valid handles或metadata报告代替identity。

候选公开形状：三原namespace的move-only `CopyBatchUpdate`，由原invocation/submission
共同建立；非阻塞observe保原Status（不能把Unavailable错误一律解释为pending），
explicit wait/cancel、成功后原mutable return、actual retirement后原invocation回收。
source resource/native dependency/ordered read仅trusted receiver seam，不能public暴露
pending WriteLease或伪造ReadLease。需在AUDITED固定精确签名、private资格与失败/完成账。
完整receiver必须实际核native endpoint/ordering，并合并producer/accepted-prefix错误；
source atom的通过不能替代这些工作。

## 冻结合同与有限 FIX

| ID | 固定动作 | 验收 |
|---|---|---|
| CU01 | 三原namespace的CopyBatchUpdate：两原controls移动到唯一owner，factory核actual同state/原接纳；观察/等待/取消、成功mutable返回、实际退休后原invocation回收 | 不同invocation/submission拒绝且不消费；sizeof仅原两handles；pending/五生命周期/回收原coldslot与三交付 |
| CU02 | 原CopyStreamAccess提供typed update的原mutable input、原native prefix dependency、F2 ordered read admission；复用原bindings/written/phase/LeaseAccess | 非写destination/foreignID/已publish/已知failed拒绝；pending只提供trusted借用，不public ready；最后logical phase为空时仍取真实native prefix |
| CU03 | 四原路线source权限/生命周期与必要消费者验证，保原失败/unknown/budget | finite source、hostsan/SDK/headers/定向Windows；完整receiver与late native backing仍主卡，不代签 |

承载是SELF有限source seam，复用原F2/F1/native，不设计新的execution owner。API形状由
真实update→attention/reduction及原型驱动，而非benchmark。原prepare/create/bind/submit/
output/unbind/reset/close/drain入口分别回指cold capacity、原读写权限、接纳/退休、成功
coverage与生命周期，全部KEEP；新owner用于后继组合的受控持有，非旧签名兼容转发。

公开source/static/shared均同名CopyBatchUpdate，metadata模板shell可用同一vendor-free
实现，typed identity/observation重逻辑仍compiled原factor；namespace route保持真实不同
control/native types。静态take(invocation&&,submission&&)失败不移动，成功同时移空两个
caller handles，非copyable。observe返回原Status；wait显式收费；request_cancel沿原
submission；take_mutable先成功实际退休，再丢原submission并借原GP02返回整个caller
owned mutable范围（未写字节仍caller内容，不扩大F2.output逻辑coverage）。release_invocation
非阻塞要求actual in_flight=false，回原control及其failed/closed状态，不假装恢复成功。
观察后仍保原invocation的phase completions；任何borrowed native dependency录入与后继
实际退休之间，owner不得销毁/回收/另观察失效的prefix。后继必须消费owner并串行观察。

私有CopyUpdateInput含原const WriteLease借用、const buffer geometry、physical/root/last
write证明。只有actual accepted且真实written destination、原完整mutable binding可以
取；known tail/phase失败拒绝。admit_read只由已资格化ordering的receiver调用，复用F2
ordered_read_unique，原writer一直由source控制持有。不是ReadLease，不允许host读取或
公开writer。dependency先保所有known error，再取最后仍async的原phase native prefix，
不拿最后空logical phase的inline完成抹掉先前在途native工作；无async也需保原Status。
全部同一原queue/runtime，实际receiver须核endpoint/real device并在最终完成合并源错误。

失败、OOM、cancel/close、abandon及borrowedContext/Queue/resource/allocator寿命沿原
control；take失败不动原binding，unknown无ready/无提前回收。source移交前caller串行，
移交后只有receiver观察原controls，避免新增默认共享observer。普通F2对象layout/锁/
原子/分配完全不变，opt-in owner为两handles，source工厂/几何有界检查，observe扫描
原finite phases；不会创建Completion、线程、queue或新native buffer。

只允许batch_transfer.h/cc、原Apple/Vulkan memory header/copy program/private copy seam、
既有copy_stream_access.h，以及直接tests/guide/evidence。公共构建图/工具、数学与
compute_batch产品仍不得改。private原Result/Memory Native输出backing未知仍需receiver
主卡固定独立FIX，源接缝不要求其预先存在。

七问：原依赖锁/许可不变；已命名两纵切及四source资格；复用原已闭F2/F1/GP02；P0
无默认额外机制，真实source费用单列；无FFI；三交付/头直接consumer；guide解释移动、
借用、错误与回收。原peer固定归档的copy/event/resident写模式沿主卡，仅借其owner
边界，不引入模型政策。工程门仍待实跑，AUDITED不写PASS，不重跑已闭数学P1。

## 当前 source 实跑进展

[Apple source 证据](../../../../evidence/correctness/inference-g03-copy-update-apple-source/README.md)
已对 AUDITED WORKTREE 实跑 CU01 的五场景、CU02 的 typed input/坏ID/源-only/最后空
logical phase与真实native prefix。另用原CopyRecording作直接receiver，先核同Runtime，
foreign Runtime拒绝，原ordered read guard接原preflight；成功/取消/前缀失败三场景均
通过，成功256bytes全部31，失败native_calls=0且不发布页。中间无producer/copy host
wait，仅suffix.wait；没有新worker/queue/Completion。三次source输入全部日志保留，
累计46.699秒。测试fixture显式控制观察handoff，不代表完整pending attention或native
late-output方案；CU03四路线风险/三交付/Windows与其余owner资格仍待完成，不写CLOSED。

[Apple/Vulkan公共source旅程](../../../../evidence/correctness/inference-g03-copy-update-native-public/README.md)
现已通过同一公共源码的初始化+64 dirty steps、original slot/generation与全历史独立回读。
Vulkan必须沿原staging路线读取，fixture漏MemoryKind编译失败及直接device读被拒绝两
原FAIL均保留；累计111.774秒。CPU/CUDA同源码验证继续，仍不代签其风险/SDK门。

[CPU/CUDA公共source旅程](../../../../evidence/correctness/inference-g03-copy-update-cpu-cuda-public/README.md)
亦已通过相同65步、1024/16384 bytes、原generation/cold slot/control回收与完整历史比较。
实际RTX4070 tiny transfer，原cached4jobs串行；只编原compute transfer及4个CUDA host
backend文件，无nvcc数学kernel或全模型。累计74.464秒（含约69.4秒原parent configure）。
当前测试文件仅将CRLF恢复为原仓库LF，语义/产品SHA不变；各证据保实际当时输入字节，
不伪造同一编译输入。source证据不代签风险、SDK、完整pending attention或G03完成。

[CUDA pending source](../../../../evidence/correctness/inference-g03-copy-update-cuda-pending/README.md)
已实跑original gated1024bytes producer→F2→CopyBatchUpdate：typed input、源-only/坏ID
拒绝、最后inline phase保真native prefix，原writer无ready；只final update.wait，bytes49
与原guards退休通过。累计62.046秒（含约60.2秒原parent configure）；
该case不代签CUDA完整receiver与失败/unknown风险。测试/source证据已保存，CU03仍施工。
