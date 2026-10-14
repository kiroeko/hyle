# G03 已接纳 producer 到精确分页消费

状态：AUDITED，2026-10-14；owner compute_batch、compute、compute_route_apple、
compute_route_vulkan，baseline `5e80c4df0`；公共形状反证基于 `3cbbd4e2e`。
授权来自[十项goal](../project/inference-gaps-goal-execution-2026-10.md)，输入G03来自
[固定源码审计](../project/inference-single-machine-foundation-audit-2026-10.md)。
原G01 native、CPU页与G02 native联合有限terminal保留；本卡不重开旧数学/P1、原G02
shared common未决Windows门或默认executor。未完成AUDITED/固定FIX前不写产品。

## Trigger 与主消费者

真实一期旅程：QKV/RoPE或普通数值producer产生本步数据→按精确dirty行/页更新原cache→
同设备原page attention→原输出owner。两request decode/prefill混合、尾增/跨页/共享只读
页、旧结果与写权限。CPU、CUDA、Apple、Vulkan及三个CPU+GPU联合路线的既有公共
能力须逐项回映，不由一个copy或ready小模型签所有producer链。

当前F1支持LeasedInput读exclusive WriteLease，F2 bind_producer建立原native依赖；
CopyBatchSubmission.output仅成功actual retirement后产生ReadLease，PagedArena.read/
batch bind只接ready frame。原Apple/Vulkan native imports会保input views；idle新prepare
释放哪些旧view、允许何时拿到page writer，需要真实consumer核，不由接口名推断缺口。
目标是将原原子自然组合为有限通用更新→读取旅程；不得伪造ready、whole arena权限、
复制整KV历史、默认CPUfallback或新增模型cache/request事务。

成熟来源固定：SGLang7224fce 的MemoryPool.set_kv_buffer/store/reshape cache，
vLLM98f721 flash_attn.reshape_and_cache_flash按slot_mapping有效量写cache，
llama.cpp cbe28b llama_kv_cache::cpy_k/cpy_v借原ggml_set_rows。只读这些固定对象，
不刷新peer、不扩出G11；reference政策与实际storage/ordering机制分别裁决。

主消费者是Qwen3.5 GQA8/2 D256片段的64步真实增长/跨页/两请求演化；正交消费者是
设备驻留数值页的独立producer/range update→readonly reduction/consumer，检查共享
页、并发slots、取消与错误，不让机制只表达KV。编码沿现有F16/BF16资格，量化与扩
attention数学仍G04/G05/G07；G06规划/capture、G09流水和G10private预算不代签。

## owner / 审计与停止

先compute_batch SCOPED核真实Arena/batch组合需要。若原F1/F2或native route确实需要
新公开/私有承载，必须先登记其真实owner、scope与AUDITED，才改产品；不得在本卡
越过仍CLOSED/CANDIDATE的边界。Graph/调度器不是默认实现owner，不复制另一个
controller、queue、Completion、permission state machine或公共工具。

下一步只冻结：真实producer输出/精确写范围/权限、native依赖与成功读取、失败/unknown/
cancel/abandon/close、旧输入清理、borrowed Context/Queue/资源、预算/费用和public调用。
先用可复现最小consumer证明现有路线的COMPLETE/PARTIAL/MISSING，已有完整路径可
ROUTED但必须实际走完；原子存在、仅ready copy或上层自己写重复寿命glue不构成反证。

预声明验证：64步非零独立oracle与true dirty bytes、无whole-history copy、全部referenced
与unreferenced权限；已接纳producer失败/unknown阻止成功输出、actual retirement之前
不可写，取消/放弃/close；共享页/两slots/旧结果/foreign与错误映射拒绝；cold每分配OOM/
明确预算；source、relocated static/shared、公开逐头与真正直接consumer/宿主san。
实际Air only，Windows只定向compile/link及已允许轻量CPU风险，不恢复重型GPU/fullmodel。
GPU/driver/private UNKNOWN按实际限制，不冒充宿主插桩资格。

P0先记录每步dirty bytes、QKV/append/copy、native dependency/host waits、warm allocation、
retained input成本与private UNKNOWN。若必须引入新的算法/常态结构，API固定候选后才
冻结同语义完整producer+消费旅程的必要定向P1，不重跑既有G01/数学性能。
源审计/实现/验证各2h检查点，按用户常设授权有限延长保原FAIL/累计；heavy serial、四
jobs，不扩建selector/checker/性能或全局构建核心。停止依据有限FIX及实际门终态、
明确协议回退/真实外部阻断，不以短观察timeout重启或把未知门改名PASS。

## 2026-10-14 精确 dirty 更新子批 scope

[实际公共反证](../../../../evidence/architecture/inference-g03-dirty-page-consumer/README.md)：
16384字节整页成功，1024字节单行更新不能经现有output/complete回接原page；两个拒绝
都正确。source audit找到三路线原CopyStreamAccess::take_mutable，不复制其retirement。
新事实只重开compute及两native route的F2公开权限返回，原数学/P1不重开。

本子批只解决原page内容保留/独占再取得与F2精确range update后的权限回收；G03整项
还保producer→attention的native依赖、成功转换、低host边界与idle imports消费证明，
不能由本子批关闭代签。不新增pending ReadLease、whole-arena permission或引擎事务。
下一阶段固定两个公共调用：PagedArena.acquire(ReadLease&)和各CopyBatchInvocation的
unbind_mutable(CopyResourceId)，通过原unique/actual successful retirement门复用现成机制。
原CopyStream私有消费不迁移到另一状态机；F3与batch/native为直接消费者。

## 固定 API、coverage 与有限 FIX

这次owner轮只固定两个机制。ready page→exclusive update目前PARTIAL，GP01 FIX；
F2 update→原mutable权限返回已有private实现，GP02 BORROW/FIX其public入口。
原成功F2 range output、精确frame、原page IDs/identity、native uses/retirement、F3文件流、
普通batch数学与旧输出门KEEP。slot_mapping/request/KV政策REJECT为基础库机制。
完整pending producer→attention仍PARTIAL，待这两个机制的真实长期consumer回映；
本ledger不把它DEFER成完成，也不授权未审计的native ordering/Graph公开合同变化。

| FIX | owner与产品形状 | 成功/失败与成本 |
|---|---|---|
| GP01 | compute_batch：PagedArena::acquire(ReadLease& ready)→PagedPageWriter | 只接受此arena当前ready original整页，复用ReadLease::try_write；成功消费入参、同block/offset/generation、slot变acquired，失败不消费/不改slot。保old frame/read/native use的unique拒绝；无payload copy、allocation或扫描，全页内容由caller原初始化证明 |
| GP02 | compute/Apple/Vulkan：CopyBatchInvocation::unbind_mutable(CopyResourceId)→WriteLease | 借原CopyStreamAccess::observe进行nonblocking actual retirement观察，再take_mutable；无retained submission/actual uses，所有accepted phases成功retired才返原writer，或撤回尚未提交binding。失败/unknown/closed/readonly/output已publish/错误ID拒绝不消费。成功清原binding并使prepared失效；O(accepted phases)，已知ready不重复native wait，无host wait/new state/allocator/metadata |

自然调用是arena.acquire(page_ready)→F2 bind_mutable(move(page.lease()))→bind_producer/
prepare/submit→实际wait成功→drop submission→page.lease()=copy.unbind_mutable(id)→
page.publish()。未写bytes仍是caller拥有的旧内容，不称其为本次copy output；F2原output
coverage保持严格。第一次page.acquire(id)须caller先初始化所有即将被数学读的bytes；
新API不追踪模型有效区，也不宣布未初始化padding可读。两个独立consumer是KV dirty
row append和驻留数值页range replace→readonly reduction，保old/shared observers拒绝。

Context/Queue/native resources/allocator仍稳定借用；控制仍thread-compatible、外部串行。
cancel/abandon/unknown必须沿原Completion实际退休，不新增destructor wait；closed/failed
不通过unbind逃出错误。cold拒绝可withdraw未提交writer，原prepare必须重新执行。
native readonly output不能变成mutable；三交付签名与export由同owner传播，无新依赖/SDK。

预声明本子批验证：CPU public64步更新/独立数值、只copy1024byte、同generation与untouched
bytes；shared/read/use/foreign/subrange/stale/closed/retained submission/outputpublished/失败
拒绝、cold撤回/reprepare；原PagedArena/F4、F2/文件stream直接消费者；三host san、actual
Air Apple/Vulkan dirty页→原分页数学及失败/unknown；对应relocated static/shared及逐头。
MSVC定向公共链接、轻量CPU执行；原未决重型WindowsGPU门保。cold allocation/budget
行为原样，新增入口以零分配quota反证。P0够裁决无新算法的本子批，未启动新P1；若
后续异步组合改变normal cost/算法，另在此G03内冻结相应有限owner/FIX与必要P1。

产品只允许GP01/GP02与直接consumer/使用说明；不改F1权限、copy coverage、native
math、controller、Graph/scheduler或公共selector/checker。该子批完成后仍逐项核G03
pending chain与其余七实际路线，不签整个G03/goal。预算沿原2h检查点、自动有限追加
保全部FAIL/elapsed，不以本scope/audit提交重置原验证成本。

2026-10-14 GP01/GP02 draft已经实现八产品文件35行；[首个CPU源码阶段](../../../../evidence/correctness/inference-g03-dirty-page-cpu-stage/README.md)
public64与原/新增九arena/F2/F4测试PASS。两次fixture编译/范围错误保原日志/输入与累计。
产品owner仍AUDITED；native actual、san、三交付/逐头与failure/unknown门尚未签，不宣布
candidate/子批/整个G03完成。原Windows重型未决门仍不启动。

实际Air追加反证：GP02只take_mutable时，abandon成功native工作后controller无公开
retirement观察，writer仍Unavailable。按同FIX补上原nonblocking observe再take，非新
Completion/state机器；原source proof仍绑定旧draft，不冒充新bytes资格。新验证保原
abandon FAIL/input，重跑受影响的CPU/actual native入口与必要风险，不重跑原G01 P1。

[追加native source/风险阶段](../../../../evidence/correctness/inference-g03-dirty-page-native-stage/README.md)
完成：Apple F16/BF16、Vulkan F16共192 final64步mixed prefill/decode/growth与9 actual
native copy风险PASS；原CPU64/arena9复验与F3直接14tests通过。保abandon真缺口、
Vulkan fixture的UMA路径/empty accepted prefix与错误teardown FAIL及对应inputs/ips。
host san、三交付/头仍待候选资格，整个pending producer→attention和G03其它路线保留。
