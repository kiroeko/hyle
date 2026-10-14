# G03 原 copy update 来源所有权与受控 pending 输入有限终审

CLOSED，2026-10-14；owners compute、compute_route_apple、compute_route_vulkan；
baseline f764a1053，按[施工卡](inference-g03-copy-update-owner.md)固定 CU01–CU03。
不可变产品candidate `bcaec010b528e372671f737f40b4aa53eb076aaf`。
本记录只签原F2来源接缝；完整G03/十项goal保持未完成。

## 七门与完整 source 合同

| 门 | Verdict | 证据与边界 |
|---|---|---|
| scope | PASS | 合法SCOPED d345→AUDITED5eda→产品bca→CANDIDATEf942；原copy update到attention/reduction的真实source所有权需求；完整receiver留主卡 |
| architecture | PASS | 原F1/F2/实际Queue/Runtime/phase/completion/GP02复用；唯一观察owner移动两个原controls，不创建执行owner/queue/Completion或模型政策；原peer固定source归档复核沿主卡 |
| correctness | PASS | 四路线公共65步；Apple五生命周期与三个实际原native copy后继；Vulkan六生命周期与actualfence；CUDA真实在途source及四风险；CPU六source单元，typed identity、坏ID、源-only、unaccepted、published拒绝 |
| delivery | PASS | source/static/shared同一CopyBatchUpdate；六relocated SDK配置、八个公共65步run、18strict standaloneheaders；consumer/installedheader SHA核；MSVC编译/link与actualCPU65步 |
| usability_docs | PASS | [Compute指南](../../../user/compute.md)给移动、observe原Status/明确wait、取消/放弃、成功writer、control回收、borrowed资源寿命与private pending读边界；SDK只链接hyle::compute |
| red_team | PASS | 实现后独立重读控件identity/users/acceptedphase、status/actual retirement分离、last-write/root/written范围、published/empty负空间、move/回收/abandon和shared符号；实跑对应拒绝/unknown/风险和六hostsan |
| engineering_performance | PASS | P0：opt-in owner仅两handles16bytes；旧控制/常态metadata/分配/原子/锁/queue不变，factory有界身份检查；私有input按activecopies/phases扫描，prefix按原acceptedphases扫描，无新payload copy/wait/数学算法 |

公开CopyBatchUpdate::take核同一个实际state/唯一原submission且有接纳phase；拒绝不消费，
成功同时移空caller。observe保原Status，不用Unavailable猜成功/退休；wait明确收费，
request_cancel保原native语义。take_mutable只在原全组成功实际退休后回原writer；未写
内容仍caller职责，不扩大output成功copy coverage。release_invocation只回控制，不抹
failed/closed状态或暗示数据发布；同一原coldslot/generation/metadata保留。

私有typed input/prefix/admit_read只给trusted receiver：保原mutable writer/geometry/
physical/root/last-write/written/acceptedphase，不给public pending ReadLease。最后空
logical phase已inline也不能抹掉此前实际native prefix；known tail/phase失败拒绝。
receiver必须核实际endpoint/device/native ordering，并合并整个source最终错误；借用
期间保原owner到后继实际退休，不能另观察/回收/销毁prefix。caller初始bytes已初始化
和页/请求政策仍属consumer，不从dirty写推导整页成功copy coverage。

## 实跑与失败保留

- [Apple source](../../../../evidence/correctness/inference-g03-copy-update-apple-source/README.md)：五pending生命周期、三个原native copy后继，foreign Runtime拒绝；成功256bytes31，失败native_calls=0，仅suffix.wait；46.699s。
- [native公共](../../../../evidence/correctness/inference-g03-copy-update-native-public/README.md)：Apple/Vulkan公共各65步，1024/16384bytes、全历史独立staging回读，111.774s。漏MemoryKind与device直接host读两fixtureFAIL保留。
- [CPU/CUDA公共](../../../../evidence/correctness/inference-g03-copy-update-cpu-cuda-public/README.md)：同一公共65步、originalslot/metadata/版本/回收，74.464s；[CUDA pending](../../../../evidence/correctness/inference-g03-copy-update-cuda-pending/README.md)实际gate/1024bytes49、末尾inline仍保nativeprefix，62.046s。
- [Vulkan六风险](../../../../evidence/correctness/inference-g03-copy-update-vulkan-risks/README.md)：accepted inline prefix/native拒绝、unknown、cancel/closed/abandon，46.477s；最初错误期待unknown立即OOM且拒绝geometry的fixtureFAIL保留。原Run在未知退休时pending/Unavailable，只有actualfence后发布第一OOM；借用geometry不等于ready。
- [CUDA四风险](../../../../evidence/correctness/inference-g03-copy-update-cuda-risks/README.md)：成功/原Unimplemented取消/unknown/abandon，72.057s；只给一个原host TU启用已有观察hook，未改产品源码/全局flag，无nvcc/kernel或device poisoning；originalreaper实际退休后metadata/guards归零。
- [六hostsan](../../../../evidence/correctness/inference-g03-copy-update-host-sanitizers/README.md)：CPU六单元+native公共65+Apple8/Vulkan6各3mode，170.768s。原编排f-string语法FAIL0.014s保；ASan无LSan，TSan只沿原已独立证明final-acquire与同Metalcommandbuffer显式模型，不签shader/driver/privatepool/CUDA runtime。
- [六SDK](../../../../evidence/correctness/inference-g03-copy-update-sdk-stage/README.md)：source/static/shared公共同名入口；Air四SDK与WSL CPU/CUDA两SDK、18heads，139.132s，original installed source/hashes/relocation核验，所有最终PASS。
- [Windows](../../../../evidence/correctness/inference-g03-copy-update-windows/README.md)：MSVC定向编译/link和CPU65步252.547s，不运行VulkanGPU/旧全局graph矩阵。

全部heavy串行/最多4jobs，Air实际Mac16,12；Windows canonical、验证副本单向同步。
原native metadata/privatepool/driver UNKNOWN、Apple每native phase原worker同步与CUDA
不支持preemptive cancel仍保，不能称完全原生执行/capture/成本对称或全步零分配。

[LOC self-check](../../../../evidence/correctness/inference-g03-copy-update-host-sanitizers/loc-owner-union.json)
固定bca：全库2930757、自实现不含测试176777；本三owner src明确并集21146。不是
整个Compute或推理引擎接线LOC。性能P0结构PASS，不重跑已闭数学P1。
[机械与影响回映](../../../../evidence/correctness/inference-g03-copy-update-host-sanitizers/validation-result.json)
保WORKTREE无法归属既有candidate introducing commit的BLOCKED工具计划；改用immutable
HEAD计划READY，未改checker/tool或产品授权。新增metadata/compiledtraits没有改变原
数学/Graph/State/FileStream算法/对象布局，按本source卡定向验；原Graph candidate
未决门继续保留，不能把它们写成PASS或因本轮关闭而豁免。

## 停止与未决范围

CU01–CU03有限source实现/验证结束；仅可复现已验身份/权限/Status/实际退休/成本/交付
缺陷或新的真实机制需求触发新scope/audit。闭本接缝不关闭
[完整pending页→attention主卡](inference-g03-pending-page-consumption.md)：compute_batch
仍SCOPED，需固定四路线/三CPU+GPU组合的receiver、页初始化/精确范围、late native
output backing、producer/sourceprefix错误传播与完整64步数值旅程，再合法进入AUDITED。
G01/G02原未决Windows门、G04–G10、单/多请求全域与十项goal全部保留，goal active0/10。
