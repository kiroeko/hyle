# F5 六FIX收口审计与预算裁决

状态：AUDITED（2026-10-12 用户批准原故障门一次追加180秒，有限FIX已冻结）；baseline
46b1d7e70。原[A/F4/F5目标](../../../architecture/compute-state-paged-design.md)、
[F5主卡](foundation-f5-paged-ragged-batch.md)全部保留，不把已实现部分改成完成定义。
本页是本轮核对与停止记录，不代签CANDIDATE、CLOSED或未运行项。

## 当前实现与直接证据

| 原FIX | 当前可证明事实 | 未完成的原完成定义 |
|---|---|---|
| F5-01 | common33项/九头：typed capacity/active区、immutable identity/owned host metadata；原native/controller实机同版本mirrors | sanitizer/最终四route交付与完整风险未签 |
| F5-02 | Linux CUDA真正原FI C++ plan/run、独立slot，short/long split+merge；实际HYLE adapter/Graph D256四域24独立oracle | standalone Direct D256 installed消费及其他风险/交付仍未签，不能用raw probe代签 |
| F5-03 | prepare/update、同shape page/tail更新、KV topology replan、old token/output拒复用、明确copy/wait账；Graph四个H2D/plan失败、actual run suffix/unknown，Direct open failed slot拒复用已验 | 其他route与producer/partial/abandon/错误组合等完整适用项仍未完成 |
| F5-04 | 原v19 region同slot/metadata mirror与original Graph kernel，private native/Gather qualification和CPU/native controllers；native partial prepare/region suffix/unknown actual drain已验 | 其他适用组合、普通warm成本/P0终审尚未完成 |
| F5-05 | four routes Direct/Graph真实paged与ragged F2→F4发布→old/new/old，非uniform/GQA/causal与容量padding成本 | 完整错误/关闭矩阵、guide/正交消费终审尚未签；Windows CUDA native paged仍具名unsupported |
| F5-06 | 普通source/static/shared初步消费者、新头有限TU、actual native GPU profiles已验 | 原18适用installed SDK格/缺符号/迁址/loader、san、P0/codegen/LOC、guide、分离红队、合法最终closure尚未完成 |

直接证据锚：

- [common与native页资源](../../../../evidence/correctness/foundation-f5-paged-arena-first/README.md)、
  [Graph页资源](../../../../evidence/correctness/foundation-f5-graph-page-recheck/README.md)。
- [CPU Graph controller](../../../../evidence/correctness/foundation-f5-host-controller-first/README.md)、
  [CPU Direct controller](../../../../evidence/correctness/foundation-f5-direct-host-controller-first/README.md)。
- [Apple/Vulkan native controllers](../../../../evidence/correctness/foundation-f5-native-controller-first/README.md)：
  每actual GPU profile96矩阵+24页oracle、同版本mirrors/two banks/old result、cap/OOM/close/
  escaped result/quota0；原runtime/capture/成本差异保留。
- [ragged多request闭环](../../../../evidence/correctness/foundation-f5-ragged-append-first/README.md)：
  七profile72episode，真实second-request重排；clone2560+append512+tail init1024，不伪称页CoW。

## 有限风险核对

| 适用风险 | 已有证据范围 | 本轮裁决 |
|---|---|---|
| schema/layout/页表/尾/GQA/overflow/empty/metadata OOM | common33项、actual caller错schema/layout与部分cap/OOM | 局部PASS，不签各controller所有OOM位置 |
| same-version host/device/token/metadata | 原FI同slot、CPU fixed controller与native two-bank update实机 | 正常/旧version局部PASS；partial failure未验 |
| producer未ready | 公共数学入口只ReadLease/PagedArenaRead，F2成功ready后才生产这类句柄；没有bind_producer/writer coercion | 不能因此把整个producer风险NA；仍需F5消费/失败ready路径证据 |
| alias/effect/读写权限 | 页guards/free writers、Graph extra reader/copy/published/foreign proof拒绝、F4 clean sharing | 正常接缝PASS；实际失败prefix/alias最终门未验 |
| old result/native use/close/escaped owner | Direct/Graph多数正常case、原F1晚guard与quota0；批准追加后native unknown poll/wait+cancel/close实际drain两episode通过 | 本有限unknown-retirement门PASS；不代签其他错误/abandon组合 |
| cancel/abandon/unknown/native error/suffix | 原基础机制已签，不能代签F5新增组合 | F5适用门尚未完成 |
| concurrency | immutable common版本并发生成/readonly分享、实际两个独立invocations | 控制对象原thread-compatible，不将并发control变适用；剩适用san/consumer退役未验 |
| SDK/san/P0/guide/redteam | 有限source/header/owner工具链证据 | NOT_RUN/UNVERIFIED，不能写PASS或NA |

原无证明的类别保持未验；任何未来NA必须指出具体API/语义不适用，不能以成本大或工具不足
替代验证。最终goal后用户要求的剩余foundation与单机单/多请求engine self-LOC评估仍保留。

## 原故障门与停止记录（保留）

source审查提出BatchInvocation::drain只wait Completion、未显式等pending/use的疑点，尚未
以runtime证明。新增native_fault_consumer借原CUDA流的真实kernel/event/reaper和test-only
unknown观察装饰，attempt原native Direct root同配置/树/预算；并未新增产品runtime/公用工具。

[故障TIMEOUT收据](../../../../evidence/correctness/foundation-f5-native-fault-timeout/README.md)：
build PASS，consumer在断言前无stdout并TIMEOUT；consumer累计180.030/180秒。记录PID
消失，全部本门stage终态。不能把TIMEOUT当作产品失败断言，也不能复写为PASS。
按主卡“每gate原失败/时间累计…到达预算BLOCKED/INCONCLUSIVE”与[模块方法](../../../developer/review/module-closure-methodology.md)
“达到预算仍不能裁决时必须BLOCKED”停止；compute_batch BLOCKED from AUDITED，F5
聚合INCONCLUSIVE。原已通过正常与append证据保持，A/F4不重开。

恢复条件：用户明确批准保留原TIMEOUT、同root/config/tree追加一次有限180秒后，先按
合法BLOCKED→SCOPED→AUDITED重新固定**仅本观察装饰定位/actual retirement**有限FIX。
先让fixture给出阶段锚或移到已接纳kernel之后阻塞，不能假定当前hang就是drain缺陷；
只有实际反证才改产品。若不追加则保本阻塞裁决，不转其他日志根或预算躲过停止条件。

预算审批待答时只读核对实际compiler include的锁定prefill.cuh：ragged分派launch前确有
cudaFuncSetAttribute（4278），header hash和原Graph warm-before-Gate参照保存在
[静态补查](../../../../evidence/correctness/foundation-f5-native-fault-timeout/static-followup.json)。
这不是hang根因/运行断言；获有限追加后应先加阶段锚、warm原domain，并把Gate移到provider
已接受kernel之后/sequence completion之前。没有获批就不启动本门或改blocked产品。

## 2026-10-12 有限恢复范围

用户回复“追加预算”，满足上述恢复条件。恢复baseline为9c4dca4bd，原TIMEOUT与
consumer累计180.029564707秒保留；只给同一个CUDA Direct consumer gate追加一次180秒，
总上限360秒。configure300/build1800原累计不变；同root
`/root/hyle-scratch/compute-unified/foundation-f5-native-direct`、同source/build/config/machine，
每次resume仍扣除全部历史stage elapsed。不得另建预算根、重跑全部算子或更改公共验证工具。

固定范围：fixture增加阶段stdout、先warm相同ragged domain，然后将CUDA host Gate放在
provider已接受kernel之后、原finish_sequence记录completion之前。unknown观察仍只装饰
原completion，真实CUDA event/reaper/F1 use不替换。先验actual drain与guards，再裁决是否
存在产品反证；只有实际断言失败才允许修改compute_batch实际退役等待。原六FIX、其余风险、
SDK/san/P0/guide/终审与goal后工作量评估全部保留，本次预算不代签这些门。

SCOPED恢复提交f73ae8761；本阶段AUDITED固定以下有限FIX，然后才修改或运行本门。

| FIX | 现状与准入 | 固定改动及验收 |
|---|---|---|
| OBS-01 | 原fixture在断言前TIMEOUT；锁定peer prelaunch设置属性、原Graph fixture warm均已静态核对，但未证明hang根因 | stdout阶段锚；warm同domain；finish_sequence中在已接受kernel之后插Gate再委托原finish记录event。原unknown错误转发与真实资源退役不变；必须实际到达drain断言 |
| DRAIN-01（仅runtime反证触发） | BatchInvocation::drain只等可观察Completion；unknown不保证actual退役，静态疑点不能代替失败 | 若断言证明提前返回，局部等待原LeaseUsePool和native pending actual退役、pending release通知；保留第一个错误，正常submit/plan/API/provider/runtime不变。相同fixture须证明Gate前阻塞、后实际guards释放且不发布/复用失败slot |

owner只有compute_batch（OBS-01测试、DRAIN-01局部实现），Graph/backend/provider不修改。
默认正常路径不增字段、线程、轮询器或新执行层；有条件等待只在显式drain。先运行OBS-01，
记录真实FAIL或PASS；若追加预算用尽仍无法裁决，按原方法BLOCKED/INCONCLUSIVE，不能续加。
本门只选真实CUDA Direct consumer与文档/状态检查；公头无改动时复用已匹配头证据，
其他风险/最终candidate的san及18 installed SDK格仍按原卡完成。

## 本次恢复实际裁决

[追加与修复收据](../../../../evidence/correctness/foundation-f5-native-fault-resume/README.md)：
OBS-01到达断言阶段，consumer-resume11/12在实际prefix被阻塞、三input guards仍持有时
输出drain waited=0，然后fixture资源析构fatal，原FAIL保留。反证触发DRAIN-01：显式drain
等原uses/pending actual退役并保留首错，release通知atomic waiter；没有新runtime/API/字段。
最终unknown poll/wait各一episode含cancel/close/Completion与token释放、Gate前等待/后guard
归零、失败slot不发布/复用、quota0全部PASS。首次hang根因未裁决，不改写原TIMEOUT。

同root累计configure16.403/build476.227/consumer190.072秒，总consumer上限360秒；
批准新增180秒没有重置原180.030秒。Win common33项及CPU/Vulkan Direct/Graph实际消费者、
WSL原CUDA Graph定向复核PASS；未变公头复用匹配证据。compute_batch保持AUDITED继续原
六FIX，当前预算阻塞已解除；其余表中未验项、最终goal后foundation/engine估算仍未完成。

## 2026-10-12 继续施工：有限OOM位置核对

用户要求“查，不能block”；预算阻塞已经解除，旧goal控制器状态不作为工程停止条件。
compute_batch/graph_batch保持AUDITED，不重开A/F4。本轮先执行原风险矩阵里的
OOM-POINT-01：CPU Direct与Graph固定reference controller cold create的caller可控metadata
allocation逐位置失败。先测相同配置成功路径的实际allocation次数，硬上限64，逐一注入
一次失败，检查ResourceExhausted与全部已接纳metadata释放；成功基线须有真实controller。
原provider内部不经caller allocator的未知费用不由此代签；不改产品分配协议/公开API。

只修改现有CPU实际consumer fixture，复用原Win/WSL CPU consumer root/config/tree与
configure300/build1800/consumer180累计；不占用仅OBS-01/DRAIN-01批准的CUDA追加预算。
本次只对应actual direct consumers与docs/closure/diff，公头未变继续复用匹配证据。
若发现真实产品反证，先在AUDITED卡固定局部FIX再修改；不因为审计未完成再次把goal标blocked。

[OOM位置收据](../../../../evidence/correctness/foundation-f5-cold-oom-points/README.md)：
Win MSVC/WERROR与WSL GCC各Direct14/Graph45 caller metadata位置全部ResourceExhausted且
cleanup归零，共118位置；原实际CPU完整矩阵/精确页/资格/ragged consumer继续PASS。
没有产品/API改动，原root/config/tree/累计预算保留；不代签provider私有allocation或其他route。

## AUDITED GRAPH-FAIL-01：原native Graph接缝的有限错误组合

baseline23b1e6672，goal恢复active。固定同一短ragged/F16/D64两node依赖Graph：四个
prepare sequence位置（node1 H2D/plan、node2 H2D/plan）分别在真实accepted prefix之后
给原finish_sequence追加具名错误，仍由原CUDA event/Completion退役。验证prepare返回
原错误、清所有token/metadata binding、不能submit混合episode、资源实际释放/quota0。
第五episode在第一个真实region原BatchProvider run返回accepted EnqueueResult后追加错误，
由同配置cold Provider table装饰计实际run调用，核原错误、后续region未接纳、无输出、
closed不可复用与drain。第六episode只在整个segment的region都已接受后插Gate并装饰unknown观察，
不在后续kernel的prelaunch API之前阻塞；warm相同完整Graph后，核公共观察与token释放/
close/drain不等于actual退休，Gate前guards保留、后归零、不发布失败输出。

只有test-only backend与same-build cold Provider table观察装饰，原configuration retain/release
hooks保持；全部原Provider/Graph kernel/F2/event/reaper/F1 use
保持真实，未加产品测试开关/私有runtime。只现有native Graph consumer及小fixture helper；
若有反证再固定其实际owner局部FIX，不顺手改公用runtime/工具。沿原
`foundation-f5-native-graph` root、原config/tree，configure300/build1800/consumer180历史
累计，六episode串行；不转移CUDA Direct的原TIMEOUT/追加预算。其他route的适用风险仍保留。

实际审计发现原Graph将两个shared regions合并进一个queue sequence；不能用finish次数
假充region次数。RUN episode改在原BatchProvider run的accepted EnqueueResult处追加错误，
并计真实run次数；UNKNOWN在完整segment原finish处插Gate。原两次fixture编译FAIL与最初
错误sequence计数造成的runtime FAIL保留，不因此修改产品。

GRAPH-DOMAIN-01沿同一原Graph门再补D256的paged/ragged×F16/BF16四个短域，实际同一
BatchProgram/native slots计划与Graph kernels，对原double独立oracle核双node/shared metadata/
两slot/update/old result。只fixture，原HYLE已准入D256而未签其实际消费；不以raw probe代签，
不把它写成standalone Direct D256交付证书（该形状仍在后续实际SDK Direct消费核对）。

UNKNOWN-REUSE-01属于原OBS/DRAIN批准范围里的失败slot不可复用验收：Direct现有两episode
都先cancel/close再drain，不能证明未主动关闭时unknown失败已经实际退休的slot不能prepare。
追加一个同ragged域episode：unknown poll→释放Completion/token→释放真实Gate并同步原流，
有限等待actual in_flight清零，然后在仍open Context中尝试prepare；只等Unavailable观察slot
清理，不把pending当失败证明。必须拒绝失败slot。仍原Direct root累计360秒总上限，未追加
预算。若runtime反证，固定局部FIX仅在prepare通过实际pending/use/completion门后检查原
native terminal，令control failed；不在native callback写普通failed flag，避免引入并发数据竞争。

[原生风险定向收据](../../../../evidence/correctness/foundation-f5-native-risk-recheck/README.md)：
六Graph故障与四D256域24独立oracle通过，原正常/页/ragged消费保留。Direct第三episode实际
反证consumer-resume16证明open失败slot在actual退休后可重新prepare；局部修复actual门后的
terminal检查，最终三episode通过，原FAIL/TIMEOUT与197.157/360秒consumer累计保留。
Win common33项/CPU完整消费与OOM位置/实际Vulkan、WSL Graph定向复核PASS，无API/字段变化。
本轮完成上述有限门，完整六FIX与其余未验风险/18 installed SDK/san/P0/guide/终审继续。

## AUDITED NATIVE-OOM-POINT-02：两原生route逐位置冷失败

baseline4dd42b4cf，compute_route_apple/vulkan与graph native factory保持其原AUDITED范围。
固定已有F16/BF16 paged reference recipe，Direct与Graph controller成功cold create实际计数
（硬上限128），随后每次cold create仅在一个caller allocator位置失败，必须报告
ResourceExhausted、完整释放已接纳metadata，并保留真实native资源owner。只覆盖经caller
allocator的allocation，不代签MLX/ggml/driver私有heap、GPU执行失败或partial upload。

把CPU既有逐位置FaultAllocator抽到test-only helper，行为不变，由原Win/WSL CPU门复核。
新native位置进入原Win Vulkan/Air Apple/Air Vulkan consumer root/config/tree与各原累计
configure300/build1800/consumer180；actual Air identity先读取，只有Mac16,12，机器重型串行。
产品/API/公头不改；若出现runtime反证再固定实际owner局部FIX，不回放原全算子/全库。

[两原生route逐位置收据](../../../../evidence/correctness/foundation-f5-native-cold-oom-points/README.md)：
Win Vulkan与Air Vulkan各Direct/Graph×F16/BF16×59=236位置、Air Apple各53=212位置，
原生共684全部ResourceExhausted并cleanup归零；CPU helper抽取后原118位置复核通过，
共802。三个实际GPU profile的原正常/精确页/F4/metadata/old result/close/escaped result/
ragged消费继续PASS。只签caller可控cold metadata，不签私有device heap或accepted partial。
原profile root/config/tree/预算累计与actual Air身份保留；无产品/API变化，其他风险与收口继续。

## AUDITED VK-FAIL-02：实际driver接纳/观察边界

baseline135954d68。复用原Vulkan lifecycle/copy测试已经采用的same-build native PFN观察装饰，
只在独立真实Context替换QueueSubmit/GetFenceStatus/WaitForFences，全部成功分支转发原函数。
固定F16 ragged一request/D64参考档，Direct/Graph各四episode：metadata第一次native submit
拒绝；math第一次native submit拒绝；math第二次submit拒绝（实际原第一prefix已接纳，
不凭shape猜成本）；math真正submit后一次GetFenceStatus返回非退休错误。核首错、无输出/
不可复用、真实native guards及drain；最后必须原wait/fence证明退休、guards与metadata归零。
每episode独立Context，避免故障poison污染后续正常消费者。没有产品hook/新runtime/数学kernel。

只Vulkan fixture借原private头访问实际PFNs，必要SDK header include只在私有验证parent补充，
不改变产品构建系统。沿原Win Vulkan与Air Vulkan root/config/tree/consumer180累计，实际
GPU/identity先核对，重型串行。Apple不能冒充相同PFN门，其queued原生边界另按原合同验证。
若有真实反证，再固定实际owner局部FIX；当前测试不授权改旧runtime算法或全局工具。

VK-FAIL-02首次在注入前发现合法单KV reference冷factory触发ggml transposed assert，
独立[UNIT-STRIDE卡](foundation-f5-vulkan-degenerate-matmul.md)合法SCOPED/AUDITED重开
compute_provider_ggml_vulkan；其余原owner范围不重开。借原零拷贝同形reshape修复unit
stride后，Win/Air Direct/Graph共四个nonuniform128-value oracle与16个actual native PFN
错误episode通过，[收据](../../../../evidence/correctness/foundation-f5-vulkan-failure-unit-stride/README.md)。
unknown原Vulkan合同为Pending且保三input uses，后原wait/fence退休保首错，不发布/closed
不可复用；首次fatal与fixture误读poll语义的FAIL保留。原正常/页/ragged/OOM消费与六个
source-contract测试PASS。保持AUDITED，其他原未验项与18 SDK/san/P0/guide/终审继续。

## AUDITED APPLE-QUEUE-03 / VK-OPEN-03

baselinef9c4be383。Apple复用原lifecycle/copy测试的MLX GPU Checkpoint：原square prefix后
只挡住同一个原Runtime worker或抛具名native异常，CPU路径拒绝；不加产品hook/数学kernel。
固定既有F16 ragged D64、物理Q/KV容量2、active Q1/KV2，Direct/Graph各四独立Context：
先失败的原native job挡住F5 metadata prepare（异步调用只拥有该controller控制，main不并发
操作controller），Gate前prepare等待、后传播Internal、不提交混合镜像/复用失败slot；已
prepared/bound F5 math排在Gate后，分别cancel、释放结果和controller、前序native失败。
核queued三input guards不早退、关闭/取消/放弃真实退休、失败无输出、quota0。所有原生成功
路径仍原MLX/worker/F2，Gate只属于test primitive；原API与普通字段不改。

Vulkan在原故障fixture添加不先close的已退休错误复用检查：known rejection与unknown实际
wait后释放public result，再prepare必须失败，保原错误；原unknown放弃结果后drain路径的
已有证据保留，不以closed拒绝假充open失败。只actual Win/Air Vulkan与Air Apple同root/tree/
config/consumer180累计，库j4/j2/j1、consumer1、重型串行，原FAIL/TIMEOUT不重置。
有实际反证先固定局部owner FIX；产品/公用验证工具不顺手改，原SDK/san/P0/guide/终审保留。

[queued lifetime收据](../../../../evidence/correctness/foundation-f5-native-queued-lifetime/README.md)：
actual Air Apple Direct/Graph八episode通过，metadata prepare Gate前等待/后传播前序Internal，
queued math cancel/abandon/failure三input guards保持到原worker实际退休，失败不发布，
open失败slot拒复用、quota0。Win/Air Vulkan known/unknown错误actual wait后未先close的
prepare/submit均拒绝，原首错/fence证明/guards正确；旧Pending-result放弃路径证据保留。
原正常/页/ragged/单KV与caller OOM通过，无产品/API变化，后续完整风险汇总与SDK/san等继续。

## SDK消费者准备：只公开入口的真实普通消费

baselinec092c5ba3。新增public_batch_cpu_consumer只包含公开Compute/Graph/CPU/simd头，
不含src私有头、trusted bridge或产品测试宏；普通Compute+Graph链接。paged/ragged×F16/
BF16四域，GQA4:2、active Q1/物理Q2、乱physical页2/0、非整尾、三有效KV的独立均值oracle
核Direct/Graph，并保old-output门。先沿原CPU门source原型定向执行，固定候选后把同一
consumer复制进隔离SDK目录，只从迁址prefix发现头/库。当前尚未签installed SDK或candidate。
这是原F5-06准备工作，不重定义余下risk；native公开consumer、缺席头/负符号与18格仍保留。

完整原要求与有限后续SDK/san/P0等回映见[最终验证计划](foundation-f5-final-validation-plan.md)。
public CPU原型四域Direct/Graph通过；固定产品输入c092且产品代码未变时沿原身份producer
完成Windows cpu-vulkan static的一格AUDITED provisional安装/实际rename/隔离复制消费者/
六前端头TU/具名cuda与apple缺席/普通运行。这只签该public CPU旅程的安装消费，native
Vulkan、detail/split头/负符号等未由此签，不把它算成18格最终资格或CANDIDATE。

## AUDITED CPU-ERROR-04 / FAILED-READY-04

baselineee10e6bb6。原CPU backend finish_sequence同步返回Completion::completed；本轮只装饰
真实Context的finish，在actual accepted CPU operation/copy后追加ResourceExhausted，仍
转发原finish。Direct首个完成stage失败后须停止后续stage；Graph按原合并segment错误
裁决，不以finish次数假充Graph节点suffix。Direct/Graph核无输出、open失败prepare/submit
拒复用、真实guards归零、drain保首错。只同ragged F16固定recipe，不换Provider/math/runtime。

failed ready纵切：完整F4 K/V/layout目录与expected-base head→原StateCloneRoute/F2 clone
真实已接纳copy失败→CloneTicket.wait不产生writer、draft不能finalize、head保持原identity。
同prepared F5 Direct/Graph不能bind缺失readonly K或submit；失败的实际ticket本来就不产生
可消费的lease，空lease只表示缺失，不能充当ready。原key/value/layout保持readonly及quota0。
只现CPU actual consumer与test-only小helper，原Win/WSL CPU root/config/tree/consumer180
累计；产品/公头不变。原closed F4不重开，只验证新的F5组合，不回放F4全集。

同原OBS/DRAIN actual退休范围再加一个正常Direct结果/token放弃episode：原CUDA event
仍被已接纳kernel后的Gate挡住，不装unknown、不cancel，放弃public观察后drain必须等
实际guards退休，返回成功而不制造取消错误。原Direct累计360秒总上限不变；Graph原门
在known error wait后、未close时prepare拒复用，原metadata/normal/unknown证据保留。

[补测收据](../../../../evidence/correctness/foundation-f5-failed-ready-abandon/README.md)：Win/WSL
CPU各四episode通过，actual error无输出/open失败不复用；F2 clone失败不产writer、F5
不submit、F4 head保持。CUDA正常public结果/token放弃→Gate前drain等guard、后OK；Graph
known error actual wait后的open prepare拒绝。原全部失败/同root预算保留，产品/API未改。

## 公开native SDK消费者准备

baseline905415cff。新增public_batch_native_consumer只公开Compute/Graph/Apple或Vulkan API，
由真实installed route宏选择，不forge能力，不include src/detail或调用trusted bridge。
与公开CPU同四域和独立mean oracle，原F2上传Q/K/V/页，public PagedArenaRead精确权限，
public typed Direct与native Graph factory、same-layout镜像与old result拒复用。source原型
沿原Win Vulkan/Air Apple/Vulkan root/tree/config/预算累计，只定向新的public消费者；
后续复制到隔离SDK目录，从自然Compute/Graph发现，不替代18格或candidate最终资格。

[公开native源码收据](../../../../evidence/correctness/foundation-f5-public-native-source/README.md)：
Win Vulkan/Air Apple/Vulkan各四域Direct/Graph独立oracle共24、原F2/public页权限/同镜像/
old result通过。用户guide已加四路线选择、生命周期、explicit reference成本与capture缺席，
只用公开消费者作可编译草图；仍不签installed格，完整six FIX与剩余七门不缩小。
