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
