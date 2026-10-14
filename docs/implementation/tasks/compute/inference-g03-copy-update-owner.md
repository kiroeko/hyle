# G03 原 F2 update 所有权与受控 pending 输入

状态：SCOPED，2026-10-14；owner `compute`、`compute_route_apple`、
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
