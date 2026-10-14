# G03 已接纳页更新到原 attention 的组合范围

状态：SCOPED，2026-10-14；初始owner compute_batch；baseline `c8ea4cc31`。
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
