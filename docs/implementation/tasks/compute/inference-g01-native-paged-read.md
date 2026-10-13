# G01 Apple/Vulkan 原生精确页读取

状态：SCOPED，2026-10-14。baseline
`9856c7c6e4f8e7f3729efbc89e43352db49b4c4e`；授权来自
[十项执行总卡](../project/inference-gaps-goal-execution-2026-10.md)。本卡先定问题和验证，
未AUDITED前不改产品；原finite resident/ragged/CPU page/default construction关闭证据保留。

## Closure card

owner为compute_route_apple、compute_route_vulkan、compute_provider_ggml_vulkan及
compute_ggml_runtime。原candidate分别a3fa6be8cf22780491e8b383ef1aafb678db1e23、
a3fa6be8cf22780491e8b383ef1aafb678db1e23、bc0ba41f714f6529cb1c583bee59d482946e7589、
42f98b94f33bdbb252b9259ece33533cf763fa41。只重开原目标需要的新page机制，旧数学与
已验CPU page仍可依赖，停止重复其P1。Graph/native联合owners不在此时改产品，G02
[原卡](../graph/inference-g02-multi-queue-batch.md)保持完整未决清单。

trigger是已准入真实两request KV arena消费：当前native MaterializedBatchInvocation每Q/
head gather整历史并走reference recipe，原prepare_ragged_batch只有stride contiguous KV，
均不能把任意physical page order/tail/share按逻辑序直接交给原native attention。
[输入审计](../project/inference-single-machine-foundation-audit-2026-10.md)G01/G02已记录该缺口。
原CPU page有限轮已CLOSED，但不构成两GPU数学的反证。

主消费者：Qwen3.5 attention片段的resident GQA cache，单请求decode及两request混合
prefill/decode，物理页反序/尾页/共享/未引用sentinel；正常用户通过原typed native Context/
MemoryResource/Program/Invocation与F1确切frame运行。正交反证为非模型的变长GQA数值
两段、非causal/空request/不同topology两slots与source alias；不把模型请求策略下沉。
至少保F16共同域、GQA8/2、D64/128/256、page16；各原已准入dtype/数学资格在AUDITED
核原source后明确，不能用一个最小点代签本卡或删除goal其它域。

机制裁决先BORROW原锁MLX SDPA/Metal、ggml Vulkan flash/shaders及原native retirement；
SELF仅不可避免的页地址/有效长度承载、冷owned topology/精确guard glue。不新增attention
公式、第二executor/queue/scheduler或每Q/head KV物化。是否改原shader读点及哪个有限
primitive/recipe承载须在AUDITED冻结，不能借benchmark输赢选择API。

已核原依赖MLX1f8e74e3f12f31365464a6867c6579f0e9b29d85和ggml456172ec733a135778adcd32d00e576a58232e45。
MLX vector/2pass与steel tiled都有独立K/V地址读点；当前入口只有array/stride/有效N，
无page table参数。ggml Vulkan当前flash标记为F32数学资格，不表明已支持页地址；需要
审计全部shader读点与原descriptor/dispatch/record假设，保实际capacity与独立logical行数。
固定三engine peer来源沿总卡，读取原block table/valid region合同，不fetch或升级依赖。

scope包括原native高效页数学、owned version/table保活、真实物理root/span与逻辑读域
资格、typed public批次入口及原normal/ragged负空间。G02之后把layout/generation和所有
page guards接入joint plan原fragments/slots；已完成LeaseSubmission页组不授权raw arena
cast，也不证明任何generic native input只读已准页。新的public shape/owner/negative
space及有限FIX未固定，当前不写实现、不启动P1。

预算：源审计/实现/验证各2h检查点，自动有限延长并保所有原FAIL/累计；重型串行，
Windows≤4jobs，不启本机重型GPU/全模型；Apple资格只actual Air。无外部阻断时继续定位，
预算不按新log重置。停止依据有限FIX/适用门终态、真实无法推进或协议要求回退，非短超时。

预声明验证：两request independent nonzero double oracle；三D与冻结dtype、causal/
noncausal、page reorder/share/tail/empty、64次复用及有限演化、双topology slots/旧output；
foreign schema/resource/generation拒绝、未引用页可写、共享页防写、quota/逐cold分配故障/
accepted native prefix失败与实际retirement。结构P0核所有实际K/V读点、无warm历史拷贝/
Q-head展开，table/metadata/host wait/private UNKNOWN如实计费。selector只真实owner与
直接native/Graph消费；source/static/shared及公开逐头、适用宿主三san；Air实际GPU数学，
Windows只定向compile/link及已允许的有限风险。P1仅不可变API/算法、同生命周期/原数学/
原16AA/controls/5%门、实际source/SDK身份固定后串行一次。普通closed CPU/ragged不重测P1。

下一步只完成本scope内原读点、资源/frame/version/submit形状与有限FIX审计，进入AUDITED
后实施。整个G01/G02、G03–G10保完整目标，不由这张scope卡或一组source结果代签完成。
