# F5 六FIX收口审计与预算裁决

状态：BLOCKED/INCONCLUSIVE（compute_batch native unknown-retirement gate）；baseline
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
| old result/native use/close/escaped owner | Direct/Graph多数正常case、原F1晚guard与quota0 | 局部PASS；unknown/error下actual drain门INCONCLUSIVE |
| cancel/abandon/unknown/native error/suffix | 原基础机制已签，不能代签F5新增组合 | F5适用门尚未完成 |
| concurrency | immutable common版本并发生成/readonly分享、实际两个独立invocations | 控制对象原thread-compatible，不将并发control变适用；剩适用san/consumer退役未验 |
| SDK/san/P0/guide/redteam | 有限source/header/owner工具链证据 | NOT_RUN/UNVERIFIED，不能写PASS或NA |

原无证明的类别保持未验；任何未来NA必须指出具体API/语义不适用，不能以成本大或工具不足
替代验证。最终goal后用户要求的剩余foundation与单机单/多请求engine self-LOC评估仍保留。

## 本轮真实故障门与停止

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
