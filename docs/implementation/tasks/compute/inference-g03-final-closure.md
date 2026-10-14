# G03 增量 producer→更新→消费最终裁决

状态：**CLOSED，2026-10-14**。产品 candidate
`11b2403e794b6eed2c89418d72b8c89aeaaac746`；合法 CANDIDATE 状态提交
`c9290d177ba6d4424cb57f9cae54ef5d12bec6e7`。当前八 owner 为 compute、compute_batch、
compute_route_apple/vulkan、graph、graph_batch、graph_route_cpu_apple/vulkan。
[原主卡](inference-g03-pending-page-consumption.md)保存 scope/audit、有限 FIX 和全部历史。
[终局证据](../../../../evidence/correctness/inference-g03-final-qualification/validation-result.json)
绑定 candidate、最终输入、七门、LOC与未决边界；candidate后未改产品。

## 交付与公共旅程

G03 的已准入有限旅程现完整：真实 producer→原 F2 精确 dirty 更新→同设备 Direct或原
Graph attention→成功数值输出；失败/unknown/取消拒绝/放弃保原来源和 actual retirement。
不把未完成 writer当ready，不复制执行器/结果槽，不按每个页节点重复来源 owner。

共同 `PagedBatchReceiver`、`graph::batch::PendingInvocation`移交原冷 slot/controller；
两 native联合 Invocation用 reserve_pending_sources、pending frame bind、submit(Update&&)、
take_update。原 native F2的 ResultCopyBinding 接一个结果的多个输出，也接多个独立
accepted结果；每结果仍只有一个原 completion owner。Apple实际 backing在原worker
preflight后解析并完整重验物理范围/别名，Vulkan冷地址不提供ready permission。

| 实际路线 | 已验主旅程与交付 |
|---|---|
| CPU | F16/BF16双请求/双槽/64步；原Direct与共同Graph，CPU风险/三host sanitizer、源码与迁址static/shared |
| CUDA | Linux原FlashInfer F16/BF16同64步；Direct与共同Graph、真实gated来源和八Graph生命周期、迁址static/shared；不外推未验Windows GPU域 |
| Apple | 实际Air原生F16/BF16多输出producer/F2/Direct64；完整生命周期、三个host sanitizer与迁址static/shared |
| Vulkan | 实际Air原生F16同旅程/风险/三个host sanitizer/迁址static/shared；BF16与async/IREE prefix不在本期准入域 |
| CPU+CUDA | 完整原factor源码、迁址static/shared；必需CPU Neg/Cast→CUDA页数学→CPU Cast/Neg、F16/BF16同64步及八生命周期 |
| CPU+Apple | 必需CPU前后数学、两个页节点共用一个Update、F16/BF16同64步；源码、host风险与两安装交付 |
| CPU+Vulkan | 同样真实CPU前后数学/两页节点同Update、F16同64步；源码、host风险与两安装交付 |

主形状保持GQA8/2 D256、Q1/17、两请求、64步跨页增长、独立double oracle。
native固定输出行数使用Q2/18两冷图，每图两个原Graph槽，合计四槽，不冒充两槽。
两独立native Abs producer另以实际非零值验证共用Update的成功、第二来源取消失败与
放弃，在两source及六host sanitizer配置均通过。正交原Abs四输出→两phase反序范围
更新→CPU reduction的64步及安装消费者证明late source机制不由attention独占；最终
成功后显式256-byte host staging计费。共同/原F2的普通数值更新反证沿原已验输入复用。

## 七门与结构成本

| Gate | Verdict | 证据与范围 |
|---|---|---|
| scope | PASS | 原goal/消费者/三peer固定输入；合法scope→audit→不可变candidate→当前terminal；只闭G03已准入有限域 |
| architecture | PASS | 原F1 exact guards、F2 controls、CopyOutcomeSource/DependentCompletion、原native FIFO/fence、Graph QueueSequence/slots/ResultStorage复用；无第二执行owner |
| correctness | PASS | 页初始化与accepted写覆盖、referenced页/root/layout/version、旧结果/来源回接、OOM/预算、known/unknown/晚失败、取消边界和放弃；Direct/Graph七实际路线数值及六nativehost风险 |
| delivery | PASS | source/static/shared公开实际consumer；native四SDK与完整CPU+CUDA两SDK、独立公共头、installed-only命令、shipping无HYLE_TESTING；MSVC定向编译与CPU消费 |
| usability_docs | PASS | [batch指南](../../../user/batch-attention.md)给自然调用、multioutput/独立producer、拒绝不消费、旧结果/资源寿命、失败slot、native cancel拒绝与费用 |
| red_team | PASS | 独立用户授权agent复核无新产品缺陷/MUST；两独立producer和true CPU+CUDA风险补齐；joint同步边界后的放弃与pureCUDA pending放弃分别核验 |
| engineering_performance | PASS | P0结构审计；原数学与其P1不重开，新增opt-in冷owner/有限表；submit不新增协议节点、KV payload copy或queue；以下实际费用与UNKNOWN保留 |

共同 pending Graph每次prepare清输入，随后bind的reader与lane向量分别请求bounded metadata，
另有原frame费用；外层reserve不是全过程零分配。Apple晚backing物理重验及原观察锁收费，
source资格/观察/retention attachment共享原锁；阻塞wait在锁外沿原Update退休。
原metadata上传/准备等待、native四冷Graph槽、正交staging和private UNKNOWN均保留。
CPU+CUDA的GPU→CPU replay可checked host wait；不宣称完全异步或非阻塞submit，G06/G09
处理规划与流水。原CUDA native cancel返回Unimplemented；拒绝不是preemption/退休，
Update request_cancel还会关闭原slot。失败状态不通过取回控制变成成功writer。

## 证据、失败与统计

- [原native receiver](../../../../evidence/correctness/inference-g03-native-pending-receiver/validation-result.json)、[common/Direct续验](../../../../evidence/correctness/inference-g03-common-receiver-batch/validation-result.json)保RC/BC数值、风险与交付。
- [GL宿主风险](../../../../evidence/correctness/inference-g03-graph-late-risks/validation-result.json)、[native交付](../../../../evidence/correctness/inference-g03-graph-late-delivery/validation-result.json)保真实TSan修正及六配置/四SDK。
- [完整CUDA交付与独立来源](../../../../evidence/correctness/inference-g03-full-cuda-delivery-multi-source/validation-result.json)保两个完整SDK与readonly destination13/独立多来源风险。
- [最终source](../../../../evidence/correctness/inference-g03-source-joint-final-checkpoint/validation-result.json)保完整CPU+CUDA源码64/生命周期8、pureCUDA8与MSVC更新夹具。
- [终局qualification](../../../../evidence/correctness/inference-g03-final-qualification/validation-result.json)保候选后native默认Apple9/Vulkan11、输入匹配与独立红队裁决。

所有原compile/fixture/runtime失败、时间与累计、frozen archive诊断和旧G01/G02未决门保留。
Apple真实completion attachment竞态与测试fault allocator竞态分别修正；原报告未销账。
首true joint夹具在原checked wait挂起，-15/94.718秒及backtrace保留；只改测试Gate释放点，
未改产品同步合同。宿主san模型仅签宿主，不签LSan/driver/shader/CUDA runtime sanitizer。

[统一LOC self-check](../../../../evidence/correctness/inference-g03-final-qualification/loc-owner-union.json)
绑定11b2403e7：全库3373533、自实现不含测试178807；八owner src与本轮变更公共头显式
并集40780。这是选定并集，不冒充八模块所有公共头、不预测完整推理引擎LOC。

## 停止与后续

G03已准入有限FIX与适用七门完成，停止产品施工。只有真实复现的已验权限/状态/退休/
backing/成本/交付缺陷或新的实际消费者需求才触发新scope/audit。
原G01/G02整体验收仍保其未决项，G04–G10、全模型prefill/decode token/s与多GPU/多机
不由本record代签。十项goal继续active；完整项当前仅G03一项完成。
