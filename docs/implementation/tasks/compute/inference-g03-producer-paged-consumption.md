# G03 已接纳 producer 到精确分页消费

状态：SCOPED，2026-10-14；初始owner compute_batch，baseline `5e80c4df0`。
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
