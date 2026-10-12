# F5 六FIX收口审计与预算裁决

状态：AUDITED（2026-10-12 用户批准原故障门一次追加180秒，有限FIX已冻结）；baseline
46b1d7e70。原[A/F4/F5目标](../../../architecture/compute-state-paged-design.md)、
[F5主卡](foundation-f5-paged-ragged-batch.md)全部保留，不把已实现部分改成完成定义。
本页是本轮核对与停止记录，不代签CANDIDATE、CLOSED或未运行项。

## 当前实现与直接证据

| 原FIX | 当前可证明事实 | 未完成的原完成定义 |
|---|---|---|
| F5-01 | common33项/九头：typed capacity/active区、immutable identity/owned host metadata；原native/controller实机同版本mirrors | sanitizer/最终四route交付与完整风险未签 |
| F5-02 | Linux CUDA真正原FI C++ plan/run、独立slot，short/long split+merge；原native probe覆盖64/128/256 | HYLE adapter/Graph D256实际消费尚需精确核对，不能用raw probe代签；故障门未签 |
| F5-03 | prepare/update、同shape page/tail更新、KV topology replan、old token/output拒复用、明确copy/wait账 | accepted H2D/plan/run prefix failure/cancel/abandon/unknown/suffix/drain适用项不由正常case代签 |
| F5-04 | 原v19 region同slot/metadata mirror与original Graph kernel，private native/Gather qualification和CPU/native controllers | 部分准备失败、多region suffix、actual unknown retirement等未验；普通warm成本/P0终审尚未完成 |
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
