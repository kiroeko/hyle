# G04/G05 编码 KV 与有限 attention 语义实施批

状态：**SCOPED，2026-10-14**。baseline `694a3d1a4b02540cb56b6ceae35b73dd1dd93440`；
当前共同形状 owner：compute_ops、compute_batch。原 G03 已 CLOSED，不由本卡重开其权限/
退休实现。具体 native/Provider/Graph owner只在实际路径与改动确定后加入同卡；AUDITED
和有限 FIX 冻结之前不写产品代码。本批沿[十项总卡](../project/inference-gaps-goal-execution-2026-10.md)，
不取消其余 G 项，也不把本地 BF16 模型不需要量化 KV 当作完成 G04。

## 新事实与范围

主消费者是 Qwen 实物所用 Q8/KV2 D256 形状，以及留出的有限窗口/非因果/不同 V 维数值
消费者；模型层、token位置、校准/精度策略留引擎。库负责真实 codes/scales 与页面几何、
增量编码和直接读取、数学语义、精确权限与 native retirement。

当前 BatchCapacity 只有一个 data_type/head_dim，K/V/root/output一起推导；PagedArena
按同一种plain dense页分配，BatchProgram/Paged/RaggedAttention没有window参数。
FlashInfer原入口固定T/T/T、D/D、causal与window_left=-1。已有TensorDesc/StorageFormat
能描述编码，但BlockQuantizationRoundtrip只返回dense BF16，量化权重端口也不能冒充
已编码cache追加/attention。以上是当前源码可复核的消费缺口，满足两owner reopen trigger。

有限候选域如下，最终准入必须由锁定native源码和小公开消费者裁决，不先签能力PASS：

| 项 | 本批固定调查目标 | 不自动扩张 |
|---|---|---|
| G04 | 首个真实8-bit cache编码与明确scale/rounding、append→resident→直接attention；先核GGUF Q8_0跨路线与FI FP8 E4M3实际支持/成本，再决定一类统一编码或有限具名native表示 | 不求所有量化格式并集，不把roundtrip/full-history解码/fallback签成完成 |
| G05 window | causal window，bottom-right位置p=Lkv-Lq+i；允许j<=p且window_left<0或j>=max(0,p-window_left)，左端含边界；-1为unbounded、0为self-only | 不将window_left当总窗口长度，不默认bidirectional local window |
| G05 noncausal | full noncausal实际native数学；独立有效区与Q/KV长度边界 | 不默认任意custom mask/sinks/softcap/MLA |
| G05维度 | 分清Dqk/Dv、两cache页几何与输出；先核原64/128/256及留出(64,128) | 不以padding或改一个kernel模板伪造不同V维 |

三种部署和七路线继续保留，single/multi、prefill/decode与真实producer组合均须覆盖；
native表示不要求字节布局一样，但每个准入格必须有明确public carrier/编码/数学资格。
格式校准、请求/page政策、完整模型/server/token/s、多GPU/多机不是本批产品。

## 已纳入的固定输入与成本反证

三peer归档再次SHA验证均匹配原输入：SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3。
llama原cache以type_k/type_v和各自row_size建view/set_rows；SGLang原FI把Dqk/Dvo、dtype、
causal/window与k/v scale分别传入；vLLM CPU cache更新/消费也显式带dtype/scales。
本地Qwen配置仍为24层hybrid，其中6个full attention是Q8/KV2 D256；原输入只读复用。

锁定ggml CPU量化attention按K的vec_dot_type转换Q，V可逐行to_float临时O(Dv)，不等于
全历史解码；必须在精度/oracle/费用合同记录额外Q缩窄。Vulkan锁定native在部分Q>=64、
quant K/V与dense布局条件下会展开全部K/V到F16 scratch；Metal也有kv_f16选择，不能因
拥有quant FA名字就签直接压缩消费。FI原Python入口还可能在输出后应用v_scale；原生
混合Q/KV模板与scale位置必须直接查锁定源码，不能猜一个新增参数就可用。

优先复用原编码身份/TensorDesc、F1/F2、G03 typed ready/pending页、原slots/queues/
source/outcome。只修改必要真实几何与数学承载，不建立第二权限体系、planner、executor
或通用format schema；不能为躲开private成本把正常域缩成仅一个decode token。

## 本轮审计与实施入口

在一张卡中完成以下有限决定后进入AUDITED：选定编码/scale、Q与K/V真实描述、Dqk/Dv/
物理页/输出字节、metadata/port与G03 carrier回映、actual native owner、有限数学域及
cold/warm/失败/并发合同。一次汇总有限FIX，再集中实现、批量检查、定向修复。
不得先扩Capacity然后逐个模块猜接口，也不逐helper另建卡。

预声明验证：公开原schema/carrier缺口小反证；编码字节/scale独立oracle、真实delta append
与未触碰历史、旧reader/shared页/跨页、独立double attention/量化误差；窗口首尾/self-only/
offset、full noncausal与不同Dv输出；foreign/编码/layout/初始化/OOM/预算/取消/unknown/
actual retirement；四native和三联合实际资格、source/static/shared/独立头、直接consumer
及适用宿主风险。原数学/权限/交付证据按输入身份借用，未运行项不写PASS。

P0先固定分配/metadata/copy/wait/量化Q与cache解码账。新增数学必须在不可变candidate后
经预先同合同、真实native peer的有限P1裁决；不重复G01/G03已验域，不改公共仪器/构建
系统或依赖锁。必要vendor窄patch及owner只能在AUDITED后固定，不覆盖不可变source cache。

审计、实施、验证各2h检查点；沿用户常设授权有限追加，所有原FAIL/elapsed/累计保留。
停止条件是有限域/FIX和实际门可裁决，或真实外部阻断；一次失败、构建慢、未知成本不
作为goal停止。Windows canonical、Air only实查身份，验证副本单向；heavy串行、4jobs，
不启动本机重型GPU/完整模型或全局测试。当前仅审计，未启动任何产品构建。

## 当前公共反证与native费用（2026-10-15）

[本批intake与原始probe](../../../../evidence/architecture/inference-g04-g05-domain-intake/intake.json)
已再次验证三peer原归档SHA，并在缓存Windows树仅编译/执行一个公开geometry probe。
configure57.859秒、必要原SIMD生成依赖与小probe compile23.891秒、run0.047秒；无产品改动，
无native数学/GPU/模型/P1。原闭合数学不因此重跑。

公开实测：GGUF Q8_0格式确实能造17408-byte真实root描述，而当前同shapeF16 root是32768/
page16384 bytes，原infer_batch_attention拒绝该编码描述。causal=false的原recipe也不能
创建Q2/KV1的layout；layout层无条件把decoder位置约束加到full noncausal，是必须迁移到
适用数学资格而不是继续保留的限制。这个probe只签缺口实测，不签新增编码能力PASS。

锁定Metal在Q>=32的Q8_0可选择kv_f16，且extra_kv_f16还可能常态预留整套F16 K/V scratch；
只限制Q到decode不会交付本goal的prefill/存储成本目标。FI混合Q/KV模板存在，但原Python
把Kscale折入sm_scale、Vscale放在输出后，必须计入真实后处理或资格化原fused variant。
编码最终选择与各native直接读路径仍须一遍有限核对；两共同owner保持SCOPED。
