# G04/G05 编码 KV 与有限 attention 语义实施批

状态：**Apple CE14 退回 AUDITED，其余 owner 保持原 CANDIDATE；CPU provider/runtime 的 CE13 为42a4c7cdcc，其余候选为982676cb05，2026-10-15**。baseline `694a3d1a4b02540cb56b6ceae35b73dd1dd93440`；
共同批owner见下表：九个原实际组件及下文新增CUDA数值状态owner；有限CE01–07/API/成本合同已冻结。原 G03 已 CLOSED，不由本卡重开其权限/
退休实现。九owner已合法SCOPED后进入AUDITED；仅下列有限FIX授权产品实现。
本批沿[十项总卡](../project/inference-gaps-goal-execution-2026-10.md)，
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

## 原SCOPED审计入口与预算（已由下文裁决）

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
不启动本机重型GPU/完整模型或全局测试。SCOPED阶段未启动产品构建；当前按下文AUDITED实施。

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
这次probe时两共同owner为SCOPED；编码与路径现已按下文固定，实际native资格仍未验收。

## 共同域裁决与具体 owner（2026-10-15）

锁定native源码摘录及整文件SHA见
[native-source-anchors](../../../../evidence/architecture/inference-g04-g05-domain-intake/native-source-anchors.json)。
原始peer归档、公开缺口probe与原失败不变。此节冻结架构与有限FIX，不把模板/源码存在签为实际资格。

| owner | 本轮新事实与允许改动 |
|---|---|
| compute_ops | 编码spec/描述/dirty encoder与有限attention语义；现有roundtrip无真实编码输出 |
| compute_batch | schema、K/V/page/output独立几何、arena role、适用数学layout检查；原权限/controller复用 |
| compute_provider_ggml | 原native batch/region Q8 ports、不同Dv、窗口与编码producer；当前仅同dtype/plain |
| compute_ggml_runtime | 原CPU page地址patch的有限窗口/编码承载；原算法复用，不改锁/获取/构建核心 |
| compute_provider_flashinfer | 原planner/Params/kernel有限混合FP8、QK/VO、mask/window及编码producer |
| compute_route_apple | 原MLX Program/array/command buffers，原private paged attention tile decoder及dirty encoder |
| compute_route_vulkan | 原paged/ragged计划及原tensor producer；不同Q/K/V/output描述 |
| compute_provider_ggml_vulkan | 原ggml region、量化ports和原shader/page patch；排除整历史F16临时展开 |
| graph_batch | 原NativePagedAttention描述与输出活跃区；当前硬造同shape K/V和同Dv输出 |

只新增以上七个实际owner到同卡SCOPED；compute_ops/batch原已SCOPED。F1/F2、native result
source观察/退休、hybrid coordinator、queues、dependency/capture authority不重写，原G03不重新验收。
若出现必须修改该边界的新事实，再定向加入owner，不能悄悄在CLOSED路径写产品。

### 能力族与承载

| 能力族 | 层级/来源 | 当前覆盖 | 动作与唯一owner |
|---|---|---|---|
| 真实编码新增行 | 原子；llama set_rows、vLLM update、ggml Q8参考/FI FP8 | MISSING | FIX/BORROW：compute_ops语义，既有Provider/native算法；输出本轮实际carrier |
| 编码驻留页→attention | 复合；Qwen shape与量化留出消费者、三个peer | PARTIAL | FIX：compute_batch几何，原F2/G03权限和完成BORROW，native math各原owner |
| causal window/full noncausal/不同Dv | 原子；FI/llama实际参数与正交交叉attention | PARTIAL | FIX/BORROW：共同数学合同，原CPU/FI/MLX/ggml Vulkan核 |
| scale校准/精度选择/page策略 | 应用策略 | OUT_OF_SCOPE | REJECT下沉；engine选择常量，库只冻结表示与使用合同 |
| 原ready/pending/旧reader/失败/退休 | 复合；G03终审 | COMPLETE于原域 | KEEP/BORROW原owner；新增编码/几何须做直接consumer反证 |

公开面回映：BatchSchema/Version是容量/布局identity；Paged/RaggedAttention与BatchProgram是
数学recipe；NativePagedBatchProgram/Version是原native recipe/准确索引；PagedArena/Writer/
Read/PendingRead是原精确页面权限；Graph NativePagedAttention是同recipe的节点；原Materialized
reference入口保plain参考档，编码页必须明确拒绝或合格native，不能偷偷全历史decode后签直接消费。

### API、identity与费用冻结

选择共同逻辑合同与有限native表示：CPU/Apple/Vulkan **GGUF Q8_0**，CUDA **E4M3FN**。
既有plain F16/BF16资格保留；Vulkan原F16数学域不扩成无证据BF16。Q8不冒充FI原生格式，
FP8不冒充已有block128权重格式。各格式在四路线都有所选真实承载，仍须实际run才能签PASS。

compute_ops增加小值类型CacheStorage：data_type/原StorageFormat/scale的明确访问器；plain scale=1，
Q8_0 logical F32、原block identity/32/34/末D轴、scale在block且外部scale=1；E4M3FN
是plain8-bit scalar、正有限F32外部scale。工厂/描述helper复用原TensorDesc，不造format registry。
K/V的FP8 scale随同一不可变BatchSchema/cache identity固定；改变scale必须新schema/cache，
encode与消费读取同spec。校准是应用政策，不新增动态全局scale/requant生命周期。

BatchCapacity分query_type/output_type、key_storage/value_storage、qk_head_dim/value_head_dim；
默认plainF16/D64，成熟有限资格先为Q/output同F16或同BF16，K/V同选定编码，(64,64)/
(128,128)/(256,256)/(64,128)。公用description helper分别生成Q、K、V、output；不留旧签名
转发/双轨兼容。PagedArena::create显式key/value role，字节/prefix用各自真实row/page描述。
所有现有仓库consumer随同批迁移。schema结构检查不含causal规则；非空Q仍须非空KV，
causal Q<=KV在适用recipe/native admission检查，full noncausal允许Q>KV。

CacheEncode是一个readonly-input→本轮encoded-output的原子Operation，带同CacheStorage；
支持F16/BF16/F32输入的完整D行。Q8输出plain U8物理carrier（本轮row count与真实row bytes），
FP8输出plain E4M3FN carrier；typed helper给出相应logical cache描述。原NativeCopyResults
仍只消费真实plain dense carrier，F2搬准确字节，原arena admission保逻辑描述/scale identity。
不增加任意GPU原位mutation、模型slot_mapping或第二completion/controller。

Q8：每32元素F32 absmax/127，存F16 scale，zero block全零，codes roundf（half-away-from-zero）
并界定[-127,127]；本轮确定性编码使用参考规则，不能借optimized ties差异声称byte一致。
输入是有限且scale可表示的数值域（amax/127<=65504）；非有限输入不承诺有效数学，invalid
format/scale/extent在冷准备返回错误。FP8：x/scale→finite-saturating nearest-even E4M3FN；
消费数值为code*scale。CPU原flash额外Q转换是明确精度合同，不被F32累加名义掩盖。

window_left int32默认-1、合法>=-1；causal=true时按上文bottom-right/inclusive规则。
full noncausal本轮仅window=-1；不自动加bidirectional localwindow。窗口/不同Dv不制造
Q×KV mask展开。复用原CPU循环/MLX tile/FI mask variant/Vulkan shader的有效行判断。
FP8 FI用独立Q/KV模板、独立K/V stride、编译时QK/VO；原AttentionVariantBase已有v_scale
检测/OutputTransform，private derived Params可传scale，在原F32归一化后输出cast前使用；
Kscale折入sm_scale。split-KV需实际oracle反证，不猜default Params有此字段。

cold：固定schema/spec、description、native plan/slot与workspace，保持原allocator/resource预算。
warm encoder只处理dirty rows，F2只复制dirty bytes，native attention只在row/tile scratch解码；
禁止整历史K/V解码/展开或为规避它缩成decode-only。既有cold prepare等metadata wait保留G06，
native private UNKNOWN照实报告。借原owner的affinity、cancel/refusal、accepted-prefix、old-output/
实际retirement；控制外部串行，readonly schema/version按原allocator线程安全约束。

自然旅程：建CacheStorage和带K/V几何的BatchSchema→分别建key/value arena→CacheEncode本轮
K/V→原result source/F2准确range→原pending frame→原Direct/Graph receiver→成功output。
CPU/GPU联合保真实CPU前后数学，单/多请求、长prefill和跨页decode均覆盖。正交消费者是
full noncausal Q>KV、(64,128)交叉attention，与Qwen既有GQA8/KV2 D256互相反证。

### 有限FIX（一次实现批，不逐helper建卡）

| ID | 允许改动 | 验收 |
|---|---|---|
| CE01 | CacheStorage/描述与CacheEncode，明确Q8/FP8字节、scale、rounding/输出carrier | 独立字节oracle、ties/zero/saturation、invalid scale/shape、各native真实producer |
| CE02 | BatchCapacity、真实K/V/output描述、arena role与layout数学分层 | page/row bytes、Q>KV full noncausal、不同Dv、原prefix/foreign/OOM/shared reader |
| CE03 | 原CPU native batch/region与CPU页patch直接Q8、窗口、不同Dv、编码producer | double oracle+CPU Q转换误差、跨页、prefill/decode，零整历史展开 |
| CE04 | 原FI scheduler/Params/kernel混合FP8、window/noncausal、不同Dv及delta encoder | 实际CUDA源/SDK、split-KV/scale/oracle、长prefill与delta；无host fallback |
| CE05 | 原MLX native private paged/ragged tile与encoder，真实Q8 U8输入与不同Dv | 实际Air F16/BF16 Q、Q8/math/window/长prefill与delta，原MLX queues/slots |
| CE06 | 原Vulkan region/page shader/encoder，Q8与窗口/不同Dv、显式直接压缩档 | 实际Air F16 Q、长prefill触发域，禁止fullKV F16 scratch，原native lifetime |
| CE07 | 原Graph节点描述、全部直接consumer迁移、指南与三交付 | 七真实路线、两个source owners/同Update/旧reader、source/static/shared/逐头、P0/P1 |

所有FIX当前UNPROVEN。实现细节可在固定owner/API/成本等级内定向查；改变这些合同须回AUDITED。

### 预声明门与七问

scope/architecture：本节、实际源码SHA、公开缺口probe与readonly consumer audit；不签native PASS。
correctness：独立Q8/FP8编码bytes与数学oracle；主D256 GQA8/2及留出(64,128)，单/多请求，
Q1/17/65与长KV/跨页/双slots；window -1/0/15/16与offset，full noncausal Q>KV；至少两次
增量只改变新行且历史字节不变，旧reader共享页、foreign描述/scale identity、invalid/OOM/
budget、原accepted错误/cancel/abandon/实际retirement。原未变G03风险借用，新增几何/encoder
与实际consumer适用host ASan/UBSan/TSan；shader/driver不由host sanitizer签发。
delivery/usability：只对应factor与真实直接consumer、独立public headers、四native source与
static/shared、三联合；shipping无HYLE_TESTING。文档/registry轮不构建，不跑P1。
red_team：不可变candidate独立反证，当前readonly agent已建议真实carrier和固定scale合同。
engineering_performance：P0逐owner记录encoding/copy/plan/allocation/wait与scratch；冻结candidate
后一次有限同合同P1（5% material margin），与锁定原native同编码、同有效区/完成/费用比，
先语义/费用/A-A/双顺序preflight，支持原P1仪器；不是跨格式或plain-vs-quant速度排名。

七问按以上门裁决：第三方仍原ggml/FI/MLX精确锁和现有许可证，无新依赖/升级；能力完整
按CE01–07有限域；内部复用原TensorDesc/F1/F2/G03/native数学/队列；性能计入真实全部费用；
没有C/FFI；三交付仍同原组件图且必要私有源码patch留既有owner；用户按明确spec/role自然消费。
当前后三类运行证据仍UNPROVEN，不能用源码存在、模板可实例化或G03通过替代。

本批budget仍2h阶段检查点/自动追加/保累计；已做公开probe81.797秒保留。实际新增native
模板/窄patch可能必须重编本owner依赖，准许本卡对应小native TU（包括必要nvcc）和构建闭包，
heavy串行/4jobs；不借此重建全库或恢复本机heavy GPU/完整模型。Air运行前实查身份，产品
只来自Windows单向archive。当前没有运行中的构建或native资格程序。

2026-10-15：九owner的SCOPED登记与mandatory checks已提交；同卡进入AUDITED，
CE01–07开始同一实现批。新能力/运行资格仍UNPROVEN，原G03与G01/G02历史状态不回填。

2026-10-15共同层首批：CacheStorage、cache logical/carrier描述与CacheEncode schema已实现；
实际MSVC/shared compute_ops及12项低精度测试PASS（4新增、8原有），
[阶段证据](../../../../evidence/correctness/inference-g04-g05-common-storage-stage/result.json)
保存输入/产物SHA、原始日志/XML与runner。configure224.156秒、compile2.891秒、run0.062秒，
累计227.109秒；编译仅compute_ops两个TU和test，没有native/GPU/模型运行。
九owner保持AUDITED。CE01仅共同schema部分完成；实际encoder、独立BatchCapacity/page role、
native数学、F2增量消费、三交付与性能均UNPROVEN，不计G04/G05完成。selector公共头对应的
其余factor/direct consumer/SDK门将在同一实现批固定后执行，不由这12项测试冒充通过。

2026-10-15 CE02实施发现并裁决descriptor空间：原MSVC实查Operation128、View120、
CustomOperationValue96、StorageFormat40、CacheStorage56、BatchCapacity40、native节点52。
两个完整format不能放入原inline64。当前AUDITED内将CacheStorage改为8-byte immutable
具名factory值，format由有限合法data_type派生原identity，不携带第二份40-byte格式状态；
plain/Q8/FP8语义与scale identity不变，移除可变aggregate字段。inline上限改88，实查
Operation仍128、Custom120、BatchCapacity60、native节点76；无普通Operation对象空间税，
custom构造多清零24 bytes属冷成本。非MSVC布局/交付还须实际验证，不预签。

CE02集中迁移schema/K/V/output/page role、layout数学分层、Graph与原consumer。编码/窗口/
不同Dv尚未接原核时明确Unimplemented；reference明确拒绝编码cache，不以整历史decode
冒充原生消费。当前所有FIX仍按同卡，未进入candidate，不借共同geometry测试签G04/G05。

本次selector的product_authorization七个实际修改owner均AUDITED、issues为空；仅10个
既有共享bench API迁移路径为unknown，原BLOCKED plan保留。按
[独立工具记录](../build/validation-selector-shared-bench-paths.md)续记此既有条件，模块不改
selector/policy或工具core。执行本卡预声明的共同factor/直接consumer/逐头编译并留真实结果；
其余公共头对应消费者与交付等门在共同批固定后完成，未运行不签PASS。

2026-10-15 CE02共同geometry批已验证，
[阶段证据](../../../../evidence/correctness/inference-g04-g05-shared-geometry-stage/result.json)
保存74项host tests、10逐头编译、原真实CPU Direct/Graph各F16/BF16的64-step double oracle
consumer、两原pending风险program及完整输入SHA/raw。共同factor build12.953秒，真实
CPU consumer必要public closure build51.125秒，累计含configure/两批test/逐头128.844秒；
没有本机GPU执行、nvcc、全模型、sanitizer/SDK或P1。本轮使用实际新公共ABI验证直接
consumer，未重签未变化的G03数学性能。旧plain正常旅程通过，新编码数学仍未接通。

下一批按CE03–06实现原native encoder/直接压缩消费和窗口/不同Dv；CPU encoder还需产出
CUDA联合所消费的E4M3FN carrier（单CPU attention首域仍Q8_0），不能把CPU→CUDA路径
遗漏成只用CPU Q8 bytes。各原核接通后移除其当前明确unsupported边界，再实际资格；
不能将这些拒绝保留为最终G04/G05收口。九owner仍AUDITED，完整goal继续。

CE03原CPU承载已核：CPU-only使用锁定ffa4e8b，CPU/GPU联合使用456172ec加原IQ1
composition；两源码的泛型flash已经支持Q8 K/V和独立Dv。有限page patch加私有boxed
causal/window边界和原generic选择，不重写attention数学，不制造Q×KV新mask。原plain
同D/unbounded路径保原fast/mask合同。为保持同一窄patch在两core/mixed/Air/SDK输入一致，
本卡只准许更新两owner已有vendored patch树摘要；upstream revision/archive/SDK pin完全
固定，不升级依赖、不改source cache/公共build或patch composition工具。

CPU encoder在原ggml custom节点/原workers中借公开Q8 from_float_ref，FP8借已验SIMD
E4M3 scalar；声明compute_provider_ggml的直接simd依赖。每普通Invocation加一个8-byte
nullable指针；只有含encoder或Q8 attention的invocation冷分配原allocator的状态/有限calls（每encoder
一个callback值），不新增queue/thread/execution owner。每worker Q8 stack128 bytes，warm
无全量input cast或history展开、无每block quantize_init锁。非有限/超Q8 scale实际payload
只写原atomic error flag，原workers退休后返回InvalidArgument，错误输出不发布ready。

2026-10-15 CE03 Windows source阶段已通过，
[阶段证据](../../../../evidence/correctness/inference-g04-g05-cpu-encoding-math-stage/result.json)
保存最终10项新增域测试、同一产品代码的5项原paged/ragged数学与风险测试、输入SHA、全部8次
运行及原始失败；累计372.126秒。实际456 mixed CPU core/原AVX2模块、MSVC shared shipping
完成Q8/FP8真实encoder bytes、Q8直接paged/ragged、F16/BF16 Q/output、四组Dqk/Dv、
window -1/0/15/16、full noncausal Q>KV、GQA8/2和Q65/D256长prefill；paged逻辑274行
共享/重排物理256行，独立double参考消费真实编码cache，容差包含原Q8查询转换与output舍入。

只读反证修复两项真实数值边界：boxed F16 V逐行转换后用原V32累计，4096-row单worker
抵消反例通过；Q8查询的有限/half scale范围在原映射之后、output写入之前检查。新增域
接受后的InvalidArgument沿原Completion返回，旧失败slot拒绝复用，按原合同创建新slot。
新Invocation仍只有原8-byte可空指针；encoder/Q8消费者才分配同一allocator状态和有限
callback/query-root表，已知capacity计入原bytes报告。Q8暖查询检查O(Q*Hq*Dqk)，无新增
thread/queue owner、无整KV展开或boxed Q×KV mask；native未知metadata仍如实报告。

本次复用cpu+vulkan树的ggml aggregate因补丁source identity更新重生成了Vulkan依赖，
原始411-step闭包及成本保留，没有Vulkan执行或资格。其后定向修复只增量编译，4jobs/重型
串行；初次摘要排序、重复scratch target、缺factory链接和测试误假定失败slot可复用等
失败日志全部保留。原已通过plain数学只运行一次，不因测试修复反复重跑。

这只是CE03有限Windows source阶段：CPU-only ffa core仍只有补丁派生检查，实际运行、
两个encoder同Graph及F2增量页面旅程、新域sanitizer/并发/OOM、SDK和P1尚未签发；
CUDA/Apple/Vulkan新增encoder与数学也仍UNPROVEN。九owner保持AUDITED，不能据此
关闭G04/G05或回填其他G项。

### CE04 CUDA numeric-status有限补充范围（2026-10-15）

新增owner `compute_backend_cuda`，baseline `b9d8bce8221870dc8fc5af61fccfffa8cfb5f048`。
该owner先进入SCOPED，原九owner保持AUDITED。具体新事实：CacheEncode要求有限实际payload，
CUDA原 `acquire_device_status` 已有reset/首错/完成D2H/跨queue合并机制，却只有索引越界和
整数溢出码；非有限编码输入不能准确映射到InvalidArgument。这是本轮已准入真实producer
必须增加的语义域，按原F5 terminal的新能力/缺陷trigger重开，不重新验收原G03权限或退休。

有限CE08只添加private数值输入域错误码及原完成status翻译；CacheEncode核在读取NaN/Inf
时写首错并安全跳过对应payload，真实Completion报告InvalidArgument。不得增加queue/
callback/host wait/error owner，不改完成槽、reset/merge/capture布局或语义。主消费者是
GPU本轮FP8新行encoder→原F2/page receiver，正交消费者是公开one-op坏payload后的退休/
拒绝发布与fresh invocation；原索引/整数错误按原同协议保留。执行同卡预声明的实际CUDA
数值/退休/三交付及适用host风险门，先做有限source批，未运行不签PASS。该私有enum与
switch的P0为无新增布局/分配，CUDA数学性能仍属于冻结candidate后的同卡P1。预算沿本卡
2h检查点/自动追加/原失败累计，其他CLOSED后端范围不重开。

CE08已核对原 `cuda_device_status.cc::completed_device_status` 和原首错/reset/merge；
范围、owner、完成/失败/费用形状及有限FIX固定，同卡正式进入AUDITED。十owner的新数学/
风险/交付/P1仍UNPROVEN，CE08只允许上述码和原switch翻译及其真实encoder消费者。

CE04同AUDITED内的实质数值反证：原variant在split各partial内乘sV后存DTypeO；F16
partial的±65536可先变Inf，即使最终相消应有限也会NaN。原FI明确支持DTypeO=float及
F32 merge，因此FP8路径固定未缩放F32 partial/merge，最后一个有限native kernel乘sV
并转实际F16/BF16输出；QK scale仍在原softmax前。planner的output width=4、KV width=1，
输出staging为align16(Qactive*Hq*Dv*4)，计入float workspace硬预算/报告，末kernel计一次
dispatch。成本O(Q*Hq*Dv)，无整KV展开；原plain同D/causal/unbounded仍走原2-byte入口。
独立K/V paged strides使用原ctor，单位为元素，原layout/plan/queues/retirement不变。
实际sV512正负相消、split与非split/scale/window/Dv oracle仍必须运行，源码支持不签PASS。

CE04首轮实际WSL源target编译：configure51.672秒后，native_cache F32 merge的原generic
派发实例化未准入D512，触发cp_async 512-bit静态断言；compile88.359秒原FAIL保留。
当前窄修复在原F32 merge入口做D64/128/256有限specialization，调用同一原persistent
kernel/四stage/128线程/原params，不编写另一套merge数学，也不改upstream cache或pin。
真实运行继续同树、同target，仅重编受影响TU；新CUDA域仍UNPROVEN。
源码反证也核了SM89共享内存：D256长prefill由原planner选CTA64，FP8/F32约96KiB，
D64/D128/(64,128)约32/72/80KiB，均在约99KiB block上限内；不得借此先签运行PASS。
有效QK scale及其log2乘积须有限，准备时明确拒绝溢出；新encoder沿原numeric状态协议。

CE04第二次同target定向编译已terminal PASS，554.542秒；含初次configure/FAIL累计
694.573秒。实查WSL RTX4070/SM89、driver617.14、CUDA13.3.73/GCC14.2，源码来自Windows
单向manifest archive，shipping HYLE_TESTING OFF。新的encoder/FP8数学与CE08真实产物
已编译，当前没有活跃构建或GPU程序。新增公开fixture已登记factor test与simd仅测试依赖，
包含sK=.5/sV=2、全部有限shape/window与sV512 split正负大值抵消；这些fixture尚未执行，
不由compile PASS代签correctness/risk/SDK/P1。十owner仍AUDITED，产品draft尚未提交。

CE04/08有限Linux CUDA source阶段现已通过，见
[原始结果](../../../../evidence/correctness/inference-g04-g05-cuda-encoding-math-stage/result.json)。
最终12项测试全部PASS：原4项one-op数学/冷分配与新增8项FP8字节、接受后numeric错误、
paged/ragged编码消费、独立sK=.5/sV=2、F16/BF16 Q/output、有限四组Dqk/Dv、window
-1/0/15/16、full noncausal Q>KV、D256 Q1/65长prefill与共享/重排逻辑274行/物理256行。
8192-row split-KV明确断言真实split开启；BF16编码的±65536、sV512相消后F16输出有限零，
证明本反例中没有partial half溢出。独立oracle读取实际编码bytes，不借用native输出做参考。

七次attempt与全部FAIL/elapsed保留在证据；attempt3原install/tests配置冲突，attempt4
测试直接compute_batch缺链接，attempt5缺device workspace/pinned-host配置，attempt6
实际encoder route2未通过旧route1 preparation guard，均没有改写为PASS。最后修复在原
describe后按encoder/attention检查route；numeric测试新增接受提交标记，冷准备错误不能
冒充device numeric状态通过。所有重型任务串行/4jobs，同缓存树按需编译，巨大native TU
只在算法改动时编译，后续fixture修复没有重复它。当前无活跃native程序。

本阶段只签Linux RTX4070/SM89、原shared shipping源码factor资格；增量F2/Graph、
新域风险、static/shared SDK、Windows实际CUDA、Apple/Vulkan和P1仍UNPROVEN，
十owner保持AUDITED，G04/G05整个项目未关闭。

CE05 Apple批内数值承载：Q8合法编码可以从65504解出65532，不能先用half tile暂存；
原二pass未归一partial也必须F32。因此Q8私有入口只将活跃Q转换为F32，原vector/二pass/
steel以F32 tile直接解block32，结果最后转换到公开F16/BF16。额外费用为O(Q*Hq*(Dqk+Dv))
的活跃Q/output转换和原F32partial，native私有metadata照实UNKNOWN；没有全历史KV展开。
不同Dv在原steel拆开BD/BV，窗口按原effective row predicate；原plain同D/unbounded NAX
和GQA特化保留。无新executor/queue/controller，没有增加普通Program/array状态。
新dirty encoder借原MLX F32数组节点、原with_device_status/Results，block34 U8 output走原F2。
全部属于既定CE05有限数学与成本承载，当前仅draft，实际Air身份已核，尚未签数学PASS。

CE05实际Air source阶段现已通过，
[阶段证据](../../../../evidence/correctness/inference-g04-g05-apple-encoding-math-stage/result.json)
保存10项最终测试、九次attempt、全部原FAIL/elapsed和canonical输入SHA；实查Mac16,12、
arm64、macOS27.0.1、AppleClang21。Q8 encoder三种输入、zero block、signed ties与
±nextafter(.5)临界、NaN/Inf/有限F32 scale overflow均已实际验证。真实paged/ragged Q8、
F16/BF16 Q/output、有限四组Dqk/Dv、window -1/0/15/16、full noncausal Q>KV、
D256 Q1/65长prefill、逻辑274/物理256共享重排页与独立double oracle通过。

数值反例：8192行±65504经Q8真实编码后解码超half范围，Q1二pass和Q65原steel均产生
正确有限抵消；解码和未归一partial不提前缩到half。D256 F32 tile固定WM2/BQ16/BK8，
静态threadgroup28928 bytes。不同Dv的paged/ragged零域返回[0,Hq,Dv]。长8192-row
plain/Q8二pass依次切window -1/0/16/-1通过；pipeline hash含paged/q8/window常量，
有效循环按窗口和原stride对齐限制，连续plain指针同时移动相应起点。原plain同维无界
F16/BF16、D64/128/256、paged/ragged直接数学也完成一次同产品回归。

原失败保留：fixture未取CacheStorage Result值、GTest模板逗号；U8 View缺原dtype输出映射；
paged fixture漏host metadata预算与空tail记录；NOTICE更新后漏同步license digest。
readonly反证还纠正roundf临界、二pass缓存键与独立Dv空形状。所有修复在既定owner/有限FIX
内，实际共享库源码目标、原MLX编译/运行，无全局测试/模型/P1；未变原域没有重复性能验证。
当前无活跃重型任务。十owner仍AUDITED；Vulkan、增量F2/Graph、新域风险/SDK/P1仍未验收。

CE06本批具体承载已固定在原Plan/Region内：attention借原FullF32 scalar Q8 decoder及独立
HSK/HSV，原use_dequant_kv已排除FullF32；只扩原private cache op的几何、causal/window、
table/row offset。ragged保真实完整K/V roots并传physical row offset，避免Q8 D64/Hkv2
的31*136-byte prefix不能满足descriptor根地址对齐；packed Q8的2-byte pitch与根descriptor
alignment分开，原root guard不减弱。没有全KV展开、Q×KV mask或第二执行器。

dirty CacheEncode先借原checked tensor输出安全F32新行，按abs(x)<=65504*127检查payload，
原status bit4经reset/retirement翻译InvalidArgument；随后借原ggml F32→Q8 Copy，输出以
零拷贝I8/U8 view发布原block34字节。不造slot_mapping/identity rows或另一个producer owner。
原Copy push已有unused param4，只由带private exact tag的新encoder启用fraction>=.5
ties-away；普通Q8 Copy/SetRows保持原规则。费用新增O(dirty elements) F32安全区和一次
checked dispatch，encoded output与原status/work/storage计入现有allocator；不触碰历史页。
这些仍属已冻结CE06有限FIX/原owner，当前draft，实际Air数学/三交付/风险/P1尚未签发。

CE06实际Air source阶段现已通过，见
[阶段证据](../../../../evidence/correctness/inference-g04-g05-vulkan-encoding-math-stage/result.json)。
最终10项测试PASS；实查Mac16,12、arm64、macOS27.0.1、AppleClang21，普通shared源码产物。
F16 Q/output直接消费paged/ragged Q8，有限四组Dqk/Dv、window -1/0/15/16/INT32_MAX、
full noncausal Q>KV、D256 Q1/65长prefill、共享重排逻辑274/物理256页均通过独立double oracle。
8192-row Q1/Q65大值抵消、plain/Q8长窗口切换、独立Dv空输出[0,8,128]及原plain同D
64/128/256无界域完成定向回归；没有全历史解码或Q×KV mask。

encoder三种输入各128个block的实际bytes与独立参考一致，含zero、signed ties和
nextafter(.5)临界；NaN/Inf与F32 scale overflow均实际接受提交后由原Results报InvalidArgument。
原shader除法的近似会把amax=1.375、x=-.6875的参考-63.499996变成-63.5，仅precise不足；
private exact分支按两次独立F32 quotient保留舍入，并以FMA residual修正。每block增加常数算术，
不改普通Copy/SetRows规则。readonly有界复核无新增MUST；exact分支每block显式增加4次FMA，
重复division能否消除由编译器决定。此处只签实际oracle覆盖的输入，不代签所有极小scale边界；
subnormal/FTZ及极小非零scale倒数溢出仍属于本卡待验数值风险。

四次attempt累计375.797秒，全部原FAIL和canonical输入SHA保留：首次raw ggml_view_tensor
没有producer图边，导致空图，改为原ggml_view_4d；空slice保留非canonical stride，先走零
payload Copy后reshape；第二、三次仅byte临界失败。readonly反证修正INT32_MAX窗口打包的
有符号溢出，使用uint32后加1，-1仍编码为0。原错误、预算和累计没有重置。

本阶段仅签实际Air Vulkan有限source数学/编码；四后端源码阶段至此都有实际证据。十owner仍
AUDITED，增量encoder/F2/Graph、多producer和三联合consumer、新域风险/SDK/P1与CPU-only
原ffa执行继续待验；G04/G05完整项目未关闭。当前没有活跃重型任务。

### CE09 Vulkan Graph 默认编码接缝（2026-10-15，AUDITED）

新增实际owner `graph_route_vulkan`，本轮baseline `4a40ebd0fb04489d39e27736353df6d53698729d`。
原F5 CLOSED域保留；新公开消费者 `tests/compute_batch/encoded_page_native_consumer.cc`
以真实CacheEncode Graph producer接原F2与pending page receiver，Air Vulkan attempt1实际
被 `vulkan_lowering.cc` 的具名custom白名单拒绝，报“Vulkan custom schema needs a cold native
lowering”。Compute端同一CacheEncode已有CE06实测，不是新增编码算法或要求调用方再写lowering。

有限FIX仅将CacheEncode路由给已存在的原Compute Plan lowering；GraphBuilder/原Program/
结果owner/F2保持原旅程，不增加API、对象字段、常态metadata、queue或同步。主消费者是两个
独立编码Graph接同Update；正交消费者是同Graph内两个不同K/V维度encoder与CPU前后数学。
只运行已声明的新域source consumer、适用source/static/shared/风险门及文档/closure检查；
该转发本身P0，无独立P1。沿本卡预算与用户常设追加授权，不重置原FAIL与累计时间。
先SCOPED登记，再审计固定上述一项FIX后AUDITED，产品改动只在后者提交。

审计裁决：原 `compile_impl` 已把纯Graph的真实TensorDesc、SSA输入/输出与custom Operation
复制进原Plan；CacheEncode只缺 `mature_custom` 的默认路由准入。选定表示/精度/失败检查由
CE06的 `lower_cache_encoder` 负责，Graph不得重复检查或引入另一个encoder。有限一项FIX已
固定，本owner进入AUDITED；现为十一owner同批，实际组合验证仍UNPROVEN。

### CE07 Apple/Vulkan 编码 producer 到页面消费 source 阶段

[阶段证据](../../../../evidence/correctness/inference-g04-g05-encoded-native-page-source/result.json)
保存实际Air Apple四次、Vulkan两次attempt，累计109.711秒，所有原FAIL/输入SHA/工具链与
fixture版本保留。最终两路线各3项测试PASS，均使用同一个公开消费者夹具；每路线8个成功
episode各6次更新、4个接受后数值失败episode，签F16 Q/output、Q8 KV有限域。

实际覆盖原生receiver与CPU+GPU联合Graph、single activeQ1/multi activeQ18、(64,128)与
(256,256)、page16/window16。每个成功episode交替使用同Graph两个encoder和两个独立
Graph原始result，均进入同一个原F2 Update，只观察最终原聚合；move后不能重用producer，
保留result不能取回Update，实际完成并释放后才能取回原controls。joint前缀为required CPU
Abs(-.125)/Cast、后缀Cast/Neg；非零dirty K参与独立量化softmax oracle，避免全零K掩盖
未消费Q/K的问题。两套carrier分别按真实K/V row bytes绑定，不假定同维或同page bytes。

首次显式克隆旧tail页，之后只复制requests*(Krow+Vrow)字节，报告逐次断言；完整K/V页
byte oracle核dirty codes与未触碰历史/unused tail，独立旧reader跨全部新版本仍禁止原页
写入且重新运行数学正确，共享clean页保持原bytes。第二个独立encoder或paired Graph内
V encoder输入NaN时，最终result/completion均保留InvalidArgument，全部4个writer都不能
发布，旧页面bytes保持；不是只验证先观察到错误就假定退役。

首次Apple fixture误用readonly-export query，违反原native batch要求同resource的Q合同，
已用原device LeasePool准备Q，不修改权限边界。首次Vulkan实际发现CE09默认Graph路由
缺失；唯一产品修复是把CacheEncode交给已有Compute Plan。只读复核还补齐了非零K/
加权oracle、完整K页与全部writer失败检查，最终差分无新增MUST。

此处不签CPU/CUDA新域producer组合、CPU-only ffa、新域完整risk/SDK/P1或整个G04/G05。
Apple BF16新域数学已有CE05证据，本次F2/联合夹具仅签F16；其余准入格保留待验，不冒签。
十一owner仍AUDITED；当前没有活跃重型任务，Windows canonical工作树保留全部本批变更。

### CE07 CPU-only 编码 producer 到页面消费 source 阶段

[阶段证据](../../../../evidence/correctness/inference-g04-g05-cpu-only-encoded-page-source/result.json)
保存实际 Windows CPU-only static shipping 构建、五次 attempt、原始 FAIL 和累计 280.800 秒。
实际使用 CPU-only ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab-hyle`，
HYLE_TESTING OFF；i9-13900K、Windows build 26200、MSVC 19.50.35728.0。
原生 CacheEncode/paged/ragged 等 15 项在 attempt2 各自通过，但该次整体因新夹具失败仍记 FAIL；
产品输入哈希相同，因此不重复旧测试。最终 attempt5 的两项新增 public consumer 测试通过。

共享夹具 `tests/compute_batch/encoded_page_common_consumer.cc` 使用原 Graph invocation、
LeaseUsePool 写票、CopyBatchUpdate 与 Direct/Graph pending receiver，没有新增产品执行器。
八个成功 episode 各六次更新，覆盖 single activeQ1/multi activeQ18、(64,128)/(256,256)、
Q8 KV、F16 Q/output、page16/window16；Graph 前后使用原 CPU Abs/Cast 和 Cast/Neg。
同 Graph 的 K/V 编码输出是同一个真实 WriteLease 的不重叠子视图，携带一个原始 Completion；
两个独立 Graph 则分别绑定原始 WriteLease 和 Completion，不合成共享完成对象。
首次显式克隆旧尾页，此后仅复制 requests*(Krow+Vrow) 字节；完整 K/V bytes、独立量化
softmax oracle、共享干净页及跨新版本旧 reader 的权限/数学均实际检查。

四个坏输入 episode 覆盖 paired/independent producer 与 Direct/Graph receiver。
CPU 的真实数值错误可能在 Graph submit、写票附着或 F2 bind_mutable 的 poisoned-source
准入阶段同步拒绝；本阶段如实检查 InvalidArgument、producer 无 output、无 inflight/phase
completion/use guard、所有目的 writer 撤回及旧页面不变，不冒充接受后的异步传播资格。
共享夹具的异步分支另在最终 wait 报错后检查 output 仍拒绝，再取回原 Update 并逐个检查
目的 writer 不可发布；CUDA 对应分支尚待实际执行，不由 CPU 同步拒绝代签。

原失败包括 CPU wrapper 默认 resource 未配置、把持有 ready output 时仍允许取回 Update
误判为失败、以及把同步 poisoned-source 准入拒绝误当异步提交失败。修正均在夹具中，
没有修改已验产品或错误/权限边界。只读有限复核未发现新增 MUST。
本证据只签该 CPU-only 源码阶段；CUDA/CPU+CUDA、新域完整风险、SDK、其余准入 dtype 与
P1 继续待验，G04/G05 和十一 owner 的 AUDITED 状态不变。

### CE07 Apple BF16 增量消费补充

[补充证据](../../../../evidence/correctness/inference-g04-g05-apple-bf16-encoded-page-source/result.json)
签实际 Air Mac16,12/macOS27.0.1/AppleClang21 的 BF16 Q/output、Q8 KV。沿原共享夹具
将 query bits、schema、CPU Cast、原生结果描述和解码一起切到 BF16；独立 oracle 沿原
CE05 的 .008 数值容差。原生与 required CPU+Apple、single Q1/multi Q18、(64,128)/
(256,256) 共八组成功 episode 各六次更新，以及四组 paired/independent source NaN
失败均通过。原 dirty bytes、旧 reader/共享页、最终 Results/Completion 与全部 writer
失败检查保留。只读差分复核无新增 MUST；Vulkan 不启用 BF16 新测试。
本次仅夹具编译 1.436 秒、运行 .968 秒，累计 2.405 秒；原阶段所有 FAIL/109.711 秒
保留，未重跑未变产品数学/P1。CPU/CUDA BF16 增量组合与全批风险/SDK/P1仍待验。

### CE04 实际组合反例：FP8 Dqk64/Dv128 的 K tile

WSL common encoded source attempt1 编译通过后，Direct 非零 K 消费失败；attempt3/4
诊断重现 actual .0772094727、独立 weighted oracle .0800453784，完整 K/V 页面 bytes
与 Q bits 0x3000 均正确。锁定 FI prefill.cuh 的 paged/ragged K producer 因 Dv128
选择 128B swizzle，但 Dqk64 的 NUM_MMA_D=4，FP8 加载循环为 4/8=0，实际漏载 K。
这属于本卡既定 CE04 数学有限 FIX 的真实反例，owner compute_provider_flashinfer 仍 AUDITED。

有限修复裁决：只在 FP8 QK64/VO128 的原 native tile 内补足 K 的128-byte承载，并同步
shared extent/stride、padding zero-fill、producer回退、repack源stride与两dispatcher的
shared-memory账；公开D、global缓存stride、QK数学迭代保持64。原FlashInfer kernel/
planner/slots不替换，无全历史KV padding/解码。使用具名锁定patch和build-local派生源码，
不修改immutable dependency cache或公共构建工具。该方案已有独立只读核证，仍须实跑。
原 source12 数学阶段的较弱K分布不能代签这个新反例；原所有 FAIL/时间继续累计。

同批 Graph failure 检查的原要求过严：实际 aggregate wait 保留 InvalidArgument，原
Invocation::output 固定以 FailedPrecondition 拒绝失败结果；公开合同没有重复同错误码的
承诺。按原合同保留最终 wait 的严格错误断言及 wait 后 output 拒绝检查，attempt4六组
失败 episode 全部通过，没有改产品。attempt2诊断printf误用string_view::c_str导致的
编译失败也保留；不因修正夹具把 original attempt1/2/3/4 的整体 FAIL 改写为 PASS。

新增 FlashInfer patch 只登记唯一 owner 数据，没有改 selector 算法。工具检查发现两项
既有消费者闭包期望过时，换回 baseline policy 后原样重现，作为
[独立测试维护任务](../build/validation-selector-consumer-expectations-2026-10.md)处理；
仅更新两个期望集合后两项定向测试通过，原失败保留。

### CE04/07 CUDA 有限修复与实际组合 source 通过

[阶段证据](../../../../evidence/correctness/inference-g04-g05-cuda-encoded-page-source/result.json)
保存六次 attempt、全部原 FAIL 与累计 1316.471 秒。attempt5 的局部 Python 解释器未查找
造成配置失败，已在本 owner CMake 显式 find_package；没有修改公共工具。attempt6 编译
561.986 秒、测试 1.024 秒通过：F16 Q/output、FP8 sK=.5/sV=2，Direct/CUDA Graph/
required CPU+CUDA，single Q1/multi Q18、(64,128)/(256,256)，十二组成功各六次更新，
六组原始 encoder 接受后数值失败，最终 wait 保留 InvalidArgument、output 拒绝、全部
目的 writer 不可发布。完整 K/V bytes、独立加权 oracle 与旧 reader/共享页不变均通过。

有限补丁位于 third_party/flashinfer_cache/patches.json，锁定原 FI revision 和 before/after
header SHA；复用既有 exact-source preparer，不修改 immutable cache。仅 native_cache.cu
的实际编译命令使用派生 include，其余原 TU 保留原头路径。真实派生 header SHA 与 manifest
相同，原缓存 header SHA 未改变，均保存在证据。K tile 增加 CTA_TILE_KV*64 bytes、padding
列 zero-fill，repack 仍只转换真实64元素；三处 native dispatcher 的 shared-memory 账同步。
无公开 metadata/队列/权限/历史 cache 扩展，native private 费用未由本阶段代签已知。
有限独立只读复核无剩余 MUST。本阶段尚不签强权重 ragged/repack 负对照、BF16新组合、
完整risk/SDK/P1或整个G04/G05。

CPU-only BF16 新组合已另行通过，见
[补充证据](../../../../evidence/correctness/inference-g04-g05-cpu-bf16-encoded-page-source/result.json)。
原两项 F16 与新增 BF16 共3项 consumer 测试通过；每dtype八组成功各六次更新、四组同步
拒绝。BF16 query bits/Cast/描述/解码一起切换，Q8 block scale仍是F16；原oracle/权限/
dirty bytes不变。仅夹具编译4.938秒、测试.297秒，累计5.235秒，原15native测试不重复，
原五attempt/280.800秒不重置。十一owner保持AUDITED。

### CE04/07 强权重负对照与 CUDA BF16 补充

[补充证据](../../../../evidence/correctness/inference-g04-g05-cuda-bf16-strong-weight-source/result.json)
保存 attempt7/8；八次CUDA source累计1388.252秒，前六次证据不覆盖。attempt7编译通过，
但原ELF RPATH优先于LD_LIBRARY_PATH，负对照loader检查拒绝运行；实际worker退出1，
如实记录terminal FAIL，测试未运行。attempt8只在负对照设置显式LD_PRELOAD，实际ldd
确认加载修复前原动态库（链接attempt6之前已保存，SHA054f65b8…）；正向测试不设置它。

原坏库在一个具名测试的八个子组合全部失败，退出1/XML tests1 failures1且八条数值失败；
修复后同测试通过。组合为paged/ragged、F16/BF16、Q2明确CTA16/Q66明确CTA>16 repack。
K/V按同一physical row的period5相关，Q=.125；独立oracle读取实际FP8 bytes与原page IDs，
原坏库与期望差约.03，不能被原.0025容差掩盖。只对强权重helper使用ASSERT_NEAR，以便
失败时每个子组合最多输出一条；旧数学断言保持原规则。

最终12项全部PASS：共同F16/BF16 producer→同F2 Update→Direct/CUDA Graph/required
CPU+CUDA，各dtype十二组成功各六次更新、六组接受后错误；九项native数学/编码测试包含
新增强权重及原八项窗口/非因果/独立Dv/长prefill/scale/split-KV抵消回归。第二轮仅fixture
编译17.379秒，后续0.077秒复用；没有重编未变native TU。只读有限复核无剩余MUST。

CE07本批source至此具有CPU、CUDA、Apple、Vulkan及三联合真实consumer证据；CPU/CUDA/
Apple签所列F16/BF16，Vulkan仍仅F16。它不签所有模式/capture/成本完全对称；新域完整
tiny-scale/subnormal/FTZ等数值风险、OOM/cancel/unknown/concurrency/retirement、source/
static/shared/逐头与不可变candidate P1仍待验，G04/G05和十一owner未关闭。

### CE01/03/05/06 极小 Q8 block 的有限数值修正

2026-10-15，CPU numeric7 与实际Air Apple/Vulkan numeric1探针已运行；原FAIL/时间保留。
CPU在amax=FLT_MIN或2^-122时产生Inf倒数，锁定ggml参考随后将Inf/NaN直接转int8，普通
运行得到零码不能证明无UB。amax=2^-120时CPU有有限倒数，Apple/Vulkan却因subnormal/FTZ
得到零码，违反本轮原字节合同。FP8极端正scale的CPU numeric7/CUDA numeric9已实际通过。

本卡仍AUDITED，在原CE01/03/05/06内冻结一个窄数值修正：计算的F16 block scale若舍入
为零，该完整34-byte Q8 block规范为全零；非零F16 scale继续原两次F32商及ties-away参考。
这是显式编码字节合同变化，包括原已定义的2^-24/2^-120零scale块也由非零code变全零；
解码值不变，不宣称全域仍逐字节等于未修改ggml。用户已不要求兼容；不缩小有限输入域。

CPU只在原CacheEncode调用reference之前分流；Apple在求倒数/归一化前用真实F16 scale
bits识别零block，并把分母替1、输入替0，避免eager分支的非有限数值进入int8；Vulkan仅
private exact encoder分支在危险倒数残差之前分流。普通ggml量化器、Copy/SetRows和attention
核不改。无新API/owner/队列/常态metadata，费用为dirty block已有扫描内的amax/零判别与
原有归一化节点中的mask；同卡P0/P1照实计入。

有限验收为三输入dtype、零/denorm_min/FLT_MIN/2^-122/2^-120/2^-24、F16零scale舍入门槛
127*2^-25及nextafter两侧、最小非零half scale附近、最大合法scale的完整34-byte oracle；
保留正常ties/zero/NaN/Inf/overflow检查，并用小host UBSan/float-cast风险运行反证CPU转换。
源探针只作原事实，不先签新合同PASS；三路线实机及适用风险门未验仍UNPROVEN。

### CE01/03/05/06 有限数值边界实机通过

[数值阶段证据](../../../../evidence/correctness/inference-g04-g05-numeric-boundaries/result.json)
保存十次实际运行、原Apple/Vulkan numeric1 FAIL、输入版本、日志/XML、runner和累计233.419秒。
CPU numeric7普通探针PASS只作诊断，不代签无UB；numeric8正式完整字节oracle两项PASS。
CPU与实际RTX4070 CUDA均通过FP8正scale为denorm_min/FLT_MIN/FLT_MAX的独立枚举字节检查。

实际Air Mac16,12/arm64/macOS27.0.1/AppleClang21，Apple和Vulkan各numeric2一项边界测试
PASS：F16/BF16/F32三输入、14组幅值及F32最大合法值，完整34-byte oracle包括零scale舍入
门槛与最小非零half scale两侧、±0、subnormal/FTZ输入。随后各numeric3的全部11项原/新
编码与attention测试PASS；编译各.138/.137秒复用缓存，不重编未变原数学。Vulkan numeric2
因private shader新输入必要重编79.512秒，原普通Copy/SetRows量化规则保持。

实际WSL GCC14 CPU-only shipping UBSan并显式float-cast-overflow，原锁定ggml量化器以
±2^-122输入在ggml-quants.c:296报Inf转signed char，负对照如期退出1；修复后的公共编码
及原F2/Direct/Graph消费者9项全部PASS。原ggml-quants、HYLE CPU encoder与两测试TU的
实际compile command均具备undefined和float-cast-overflow，不以仅测试exe的flag冒充覆盖。
该风险运行100.422秒；不签GPU shader/driver sanitizer、LSan/TSan或全F32穷举。

有限只读复核无新增MUST；公共注释和用户指南明确零scale字节合同变化、正常非零scale的
原参考规则与边界。十一owner仍AUDITED；OOM/budget/cancel/unknown/concurrency/retirement
新增域风险、source/static/shared/逐头与冻结candidate后的P1继续待验。本批所有重型程序
已经terminal，没有残留构建；全goal粗估40%，仅G03整项CLOSED，其余状态不回填。

### 新增域风险批的固定消费形状

沿原预声明风险门，复用原故障夹具并以测试专用编译选择启用Q8 K/V、Dqk64/Dv128与
window16；原plain fixture默认不变，不增加产品故障API。CPU原pending receiver九种模式
检查成功、初始化hole、foreign layout、逐cold allocation OOM、source容量、closed、旧ticket、
原resource budget与abandon；按照各自真实encoded row/page bytes生成copy范围与output。
实际Air原native page六种模式覆盖metadata拒绝、cancel/abandon或accepted wait/unknown、
逐cold allocation、empty/foreign/旧结果。仅换数据域与独立byte几何，原native实际边界hook
不改。CPU public encoder与编码F2/Graph消费者的ASan/TSan沿现有缓存host树补齐；已通过的
UBSan/float-cast不重复。shader/driver、LSan及private pool资格不由host instrumentation代签。
本批不重跑原G03全矩阵/P1；新增域风险仍待实际运行后裁决。

CUDA同批沿原五种receiver source风险模式，换成FP8 Kscale=.5/Vscale=2、D64/128与window16，
warm/native root、dirty producer与两个copy范围均按各自描述/row bytes。原实际event observation
hook只在验证树的一个backend host TU启用；不扩写hook，不由该构建代签shipping SDK/GPU
sanitizer。两原CPU encoder invocation另以两线程/两queue同时运行Q8/FP8，检查一侧NaN失败
不污染另一侧的flags/bytes，失败原invocation按原合同重建。启动gate保证线程准备或第二次
async启动失败时不会把另一线程永久留在等待中。

### 新增域有限风险阶段通过

[编码页风险阶段证据](../../../../evidence/correctness/inference-g04-g05-encoded-risk-stage/result.json)
保留11次实际运行、两个原FAIL、全部输入版本/原日志/实际编译命令，累计441.900秒。
本阶段仅修改测试夹具，产品仍为304f7a600的实现，十一owner仍AUDITED。

Windows CPU-only新Q8形状的九种pending receiver模式及原plain九模式全部PASS；最终七项
encoder测试也通过，包含两实际线程/两queue、同prepared artifact下两原invocation的Q8/FP8
字节与NaN失败隔离、失败invocation重建。原source/dirty范围与output按K/V独立几何计算。

实际Air Mac16,12，Apple和Vulkan各六种native page风险模式PASS；逐个注入cold controller
52/55个allocator失败点无泄漏，拒绝cold prepare仍保留原generation并可运行旧recipe。
Apple实际queued prefix/cancel/abandon、Vulkan实际metadata/native拒绝、accepted wait错误和
unknown poll均检查query/page guards及实际native/fence退役，失败后所有准确页面可重新获取。
empty输出、旧result排他、foreign resource、不同字节K/V交换拒绝及原plain domain control
通过。Apple attempt1原FAIL是新断言误要求native bind失败后继续复用该slot；原合同允许
退役，故改用独立原controller检查交换拒绝，正常slot仍完成。未改产品错误/恢复边界。

实际WSL GCC14 CPU-only，ASan（detect_leaks=1）、UBSan（另含float-cast-overflow）、TSan
各10项encoder/编码F2/Direct/Graph及九种新域receiver模式全部PASS。实际compile command
核定ggml-quants、CPU encoder、两public positive TU与receiver夹具的instrumentation。
ASan1在普通运行通过后，独立只读复核发现测试barrier的准备/第二async启动异常可能死等；
改为主线程建原对象和有异常回收的单向promise gate，最终ASan2/TSan2/UBSan1均绑定新版本。
TSan1在main前unexpected memory mapping退出66，原FAIL保留；沿原已验证的进程级
setarch x86_64 -R重跑通过，不修改全局kernel配置或压制race报告。

实际RTX4070/CUDA13.3.73，FP8 D64/128独立scale与window16的原五种source风险全部PASS：
known/unknown/late source错误、abandon、foreign CPU QueueSequence拒绝。真实producer guard、
原native slot、source metadata均在实际完成后退役，不以成功suffix掩盖源错误。原观测hook
仅验证树backend host TU启用；构建日志无CUDA kernel重编，未据此签shipping SDK资格。

有限只读差分复核无剩余MUST。新域各GPU路线适用host sanitizer、source/static/shared与
逐头、不可变candidate后的P1仍待验；不签driver/shader sanitizer或private pool已知字节，
原G03与已验CPU数值负对照不重跑。当前所有重型程序terminal，无残留构建/模型程序。

### CE10：实际 Apple host UBSan 发现的原生空值生命周期

本卡compute_route_apple仍AUDITED。首次对完整MLX host施加UBSan后，Apple新域数学尚未
执行即在原metal-cpp Foundation/NSObject.hpp:112因空Referencing<ResidencySet>*成员调用
退出。独立无HYLE/MLX小程序用原NS::SharedPtr<Device>默认构造后TransferPtr赋值同样
退出-6，而直接真实objc_msgSend(nil,release)退出0。原SharedPtr的no_sanitize属性不把
C++空对象调用变为已定义；不靠关闭UBSan或新增验证模型结案。

固定CE10有限FIX归原Apple owner：在锁定metal-cpp f567ed的NSSharedPtr.hpp nullable
retain/release调用处判空，一次覆盖RetainPtr、default/null/copy/move、reset、析构；保持
原非空引用计数顺序、对象layout/API与nil空操作语义。只增加现有Apple patch bundle的
具名manifest和原Apache-2.0许可证，复用既有exact source materializer，在Apple私有依赖
接线中使用build-local派生头，并安装同一派生头。不得修改原dependency cache、公共
materializer/selector、NSObject成员函数内判断this、扩大Apple功能域或引入全局同步。

验收固定为原头独立反例保留FAIL、派生头同一反例与完整空/非空/copy/move/reset生命周期
在Clang21/O3/UBSan通过，再续既定Apple11数学、4编码native/joint及6风险；已通过的
Vulkan与CPU身份匹配证据复用。冷/暖非空retain/release多一次判空，作为本卡P0/P1的
原生桥接成本；不新增常态metadata/分配/锁。当前尚未签CE10 PASS，所有原FAIL/elapsed保留。

### CE11：实际 Apple 分页消费者的 allocator 统计读写竞争

Apple TSan attempt2的11项数学PASS，随后首个公开encoded native/joint分页消费者在
get_active_memory与主线程释放旧array的active_memory_写之间报告真实race并退出-6。
原MLX MetalAllocator写路径已持mutex_，四个统计/limit getter却未持锁；malloc/free在同一
锁内调用其中getter，因此不能简单给getter加锁而保留内部调用，否则会递归死锁。

本卡Apple owner仍AUDITED，固定CE11有限FIX为原allocator.h/cpp：四个外部getter复用
既有mutex_；原malloc/free已经持锁的统计读取直接访问相同字段/cache_size，保持原策略、
统计含义和引用/释放顺序。不增加锁对象、atomic、scheduler或整次执行串行化；不修改
原source cache，不压制TSan。成本为每次外部统计读取一次既有mutex短临界区，原分配/释放
路径不新增lock。沿既有Apple manifest接线与静态/动态安装，不改公共HYLE API。

验收为锁定原输入与TSan race负对照保留，修复后同一11数学/4分页/6风险的实际TSan通过，
再批量复验Apple ASan/UBSan含CE10独立生命周期。Vulkan与CPU不受影响的证据继续借用。
若出现不同报告先定位实际owner，不把本FIX扩成通用MLX线程安全重构。全批仍未收口。

### 实际 Apple/Vulkan 新域宿主 sanitizer 阶段结果

[本阶段结果与原始记录](../../../../evidence/correctness/inference-g04-g05-native-host-sanitizers/result.json)
保存13次实际运行、额外一次configure前TSan模型校验失败、原始日志/XML/输入与runner。
已测累计992.375秒，启动前校验失败未测得耗时单独保留unknown，不记为零或重置预算。
实际MacBook Air Mac16,12/arm64/macOS27.0.1/AppleClang21，六组Apple/Vulkan各
ASan/UBSan/TSan均PASS：每组11数学，Apple4/Vulkan3编码native/joint消费者，以及六种
原Q8独立页几何风险含52/55个cold allocation故障点。原MLX host、ggml Vulkan host、
HYLE encoder/controller/Graph/协调器及测试TU的实际sanitizer编译命令同时保存。

CE10原无HYLE/MLX空成员调用反例退出-6，真实Objective-C nil消息对照退出0；产品九函数
十二处nullable调用判空，派生SharedPtr头SHA为d93f38effcc9556aa2cd2dec13db0cdcf37cd36bfd217bcbeedeb7e6c55fd2f4。
Clang21/O3/UBSan独立完整空/非空/copy/move/reset/引用计数与真实Metal device生命周期PASS。
CE11原Apple TSan2保留真实统计竞争FAIL；修复后TSan3、ASan3、UBSan5均绑定同一
CE10+CE11产品并通过上述全部消费者。只读复核无剩余MUST，未增加allocator锁对象/状态，
外部统计getter新增原mutex短临界区，原分配/释放路径不新增lock。

最终Apple三配置实际SharedPtr和allocator派生文件SHA、compile command均核对；原缓存SHA
未变。TSan仅使用原已独立复现的final-acquire与Metal commit/completion注解，原模型SHA
不变；完整逆变换严格匹配当前canonical头。首次校验脚本漏逆变换三处既有method/call
编辑导致启动FAIL，修正校验后续跑，没有修改产品引用计数或压制报告。Metal模型只替换
派生树一个MTLCommandBuffer.hpp，Foundation保留实际产品CE10判空。Vulkan借用早先
PASS时已核对route输入与全部非Apple lock条目一致，不因Apple patch asset改变重跑。

本阶段ASan detect_leaks=0，不签LSan、GPU shader/driver、native private-pool字节上界、
SDK/逐头/P1或G04/G05全批PASS。十一owner仍AUDITED。下一批按既定门复用六棵shipping
SDK树，使用新安装并搬迁的prefix运行12独立头、新数学、CPU/native/joint编码消费者；
CUDA显式清除上一风险阶段私有CMAKE_PROJECT_INCLUDE，实际编译命令必须无HYLE_TESTING。

### 新增域静态/动态 SDK 与独立头交付结果

[六组混合路线 SDK](../../../../evidence/correctness/inference-g04-g05-encoded-sdk/result.json)
覆盖实际 Air CPU+Apple、CPU+Vulkan 和 WSL CPU+CUDA 的静态/动态配置；
[Windows CPU-only SDK](../../../../evidence/correctness/inference-g04-g05-cpu-only-encoded-sdk/result.json)
另覆盖原 ffa CPU-only 两配置。全部绑定同一产品 f772fcad2，无新产品改动。
各配置使用复用的 shipping 构建树、新安装并搬迁的 prefix，外部 consumer 仅使用安装头，
十二个改变的公开头分别独立编译；混合配置十六个 consumer TU，CPU-only 十四个。
实际 shipping 编译命令无 HYLE_TESTING，不借上一风险树的 hook 代签交付。

每组十三项新增 CPU 编码/有限数学测试和三个公开 Direct/Graph 增量 F2 consumer PASS；
首次 Apple shared 另包含一项原 plain control。Apple/Vulkan 各十一项、CUDA 十项 native
数学 PASS；Apple 四项、Vulkan/CUDA 各三项原生/联合编码消费者 PASS。
原有限 F16/BF16、D64/128 与 D256、window/noncausal、六次 dirty 更新、同 Graph/两 Graph
的 K/V producer、历史/共享页和失败不发布合同继续检查。Vulkan BF16 未因此准入。
Apple 安装派生 SharedPtr/allocator 头及许可证 SHA 匹配；安装头独立 Clang21/O3/UBSan
生命周期测试在静态/动态配置均 PASS。

六组混合配置保留九次原尝试，累计1946.982秒。原 CUDA configure 空 include 参数失败、
风险缓存 tests/install 配置不符后的定向终止，以及遗漏 compute_state 目标导致旧库安装
失败均保留；仅修改验证脚本参数/目标列表，没有重写产品构建系统或重置失败预算。
CPU-only 三次尝试累计265.423秒；原 shared attempt1 漏编 backend_cpu/provider_ggml/route_cpu 三个工厂目标，
安装混入旧静态 lib，外部消费者出现三项 dllimport 未解析；补齐 shipping 目标后使用
新 prefix 复验，保留原 FAIL。CPU-only 使用同一原树显式切换 static/shared，不宣称两棵
独立 producer 树。原始 runner、命令、输入、XML、安装 artifact SHA 与完整失败日志均归档。

本阶段只签三交付与逐头/公开消费者门，不签新增域 P1、native private-pool 上界或整个
G04/G05/G01–G10 完成。十一 owner 仍 AUDITED；下一步收敛成本账、冻结 candidate，沿本卡
5% material margin 和同编码/有效区/完成/费用合同执行有限性能验收，不重跑原 G03 P1。

### 固定产品 candidate、成本预审与 CPU peer 数值预检

产品 candidate 为 `982676cb05a6e6244e66e0a596347aa96043617b`，全部产品输入与 SDK 已验
`f772fcad2f2a6a67aa605bbaee10b68f1d540708` 相同；中间提交只增加交付证据/文档。
CE01–11 有限产品 FIX、直接消费者迁移、新域数学/有限 source 风险、CPU 与 Apple/Vulkan
宿主 sanitizer、三交付与独立头均已有上述实际证据。CUDA 新域宿主 sanitizer 矩阵尚无
完整记录，仍须按适用性取证，不能由 source-risk 或其他后端代签。十一 owner 合法 AUDITED→CANDIDATE，公开合同/算法/常态布局冻结。
本节以后只增加独立 benchmark adapter、合同、审查和证据；若新事实必须改产品则退回 AUDITED，
不得在 candidate 上改代码后继续引用旧 SDK。

[结构成本账](../../../../evidence/architecture/inference-g04-g05-cost-preflight/cost-ledger.json)
核对22个产品文件 SHA；独立只读复核未发现新的生命周期/产品 MUST。明确计入 CPU 每次 Q8
attention 对全部 Q 的 O(Q*Hq*Dqk) 有限/范围扫描，随后原 cast/Q8 dot 转换与逐 V 行 decode；
CUDA 一个 sequence 共享的4B reset/回读/event，以及 FP8 F32 输出暂存和最终 scale/cast；
Vulkan dirty encoder 的4*Ndirty安全 F32 scratch、checked dispatch、Q8 Copy、状态退休和
必要 flush/invalidate；Apple 编译 warmup 与正常 evaluate 的真实 stream/output/status 同步。
只更新 dirty rows 不等于单核/零同步；MLX 实际 fusion 数与 native pool 峰值保持未知，
不由 host sanitizer 或接口 cost report 代签 G10。当前 P0 为源码结构账，完整工程性能仍待 P1。

[peer 构造前合同](../../../../evidence/architecture/inference-g04-g05-cost-preflight/peer-preflight.json)
保留原 CPU/Metal/Vulkan/FI 源码锚点和各自限制；原 FI 64/128 错误核不能当独立合格 peer，
原 Metal 未导出 Q8 64/128 不能冒充已有 pipeline。有限 CPU adapter 首先接通独立原 ffa DLL
的 Q8 描述、K/V 独立 stride、Dv 输出与原 F16 bottom-right window mask；没有新数学核。
原 G01 plain 证据不因本轮 benchmark adapter 扩展被改写，不重跑其既有 P1。

[CPU 数值预检](../../../../evidence/correctness/inference-g04-g05-cpu-encoded-peer-preflight/result.json)
保留两次实际尝试、原 adapter 编译失败，总计37.969秒，无产品重建或计时。已验 primary/
placement 各三个形状：Q1/17 KV1024/768 D256 window16、Q64/128 同 KV/D256 causal无界、
Q17/1 KV7/47 D64/Dv128 full noncausal（含 Q>KV）。两 arm 对独立 double oracle 最大
绝对误差0.000244301088，arm 差为零，压缩页 readonly 字节核对通过；原 DLL/SDK artifact
SHA 与实际 compute 函数 image、tensor336/plan56 ABI 同时核定。

该 adapter 只比较固定 ready 历史的 resident warm attention：同原 BatchProgram/两个slots/
update/F1/Queue/Completion/output owner 返回。peer cold gather 的额外压缩缓存每个 D256
形状1949696 bytes，留出形状22032 bytes，另有原 causal/window mask；都显式报告，warm
不复制历史。不得用它代签 dirty producer、changing-layout planning 或全 G04/G05。
[有限合同](../../../../bench/inference_resident_sdpa/cpu_encoded_p1/contract.json)已通过原统一
schema/provenance evaluator，5% margin、n15/31、双执行/分配顺序、placement、同 workload
A/A 与原 controls 不变。候选后先做短 A/A preflight，不合格即停完整采样；无新增 P1 PASS。

有限 CPU adapter 的独立只读复核无新增 MUST；实际源码、原 DLL 独立性、三形状、成本/
完成边界与 retained 数值日志均匹配。下一步只允许原合同 A/A→controls→有限 P1；其余
GPU peer 的编译/oracle/性能、dirty producer 性能与 CUDA 新域 host 风险仍需实际证据，
本 candidate 不因 CPU warm 对照通过就缩小全批范围。

本轮 selector 因六个 shared benchmark/合同输入没有 owner 映射仍为 BLOCKED，原计划
`.scratch/inference-gaps/g04-g05/candidate-cost-plan.json` 保留；product_authorization 无
issues，selected docs/closure 和146 skipped reasons保留。沿既有
[工具归属记录](../build/validation-selector-shared-bench-paths.md)追加实际反例；本轮只做
上述 adapter 数值/合同与必要维护检查，不重写 selector、公共仪器或产品构建系统。


CUDA 新域 host 风险按原 F5 三棵缓存树续验：ASan 地址 facet、UBSan、TSan 数据竞争，
各运行十项 native 数学、三个 Direct/Graph/CPU+CUDA 增量 consumer 和原五种 source 风险；
核对 backend/provider、FI 的 host CUDA TU、batch/arena 与 consumer 的实际 sanitizer 命令。
仅私有 backend host TU 启用原 observation hook，不签 shipping。沿原独立 no-HYLE 对照，
ASan 用 protect_shadow_gap=0/detect_leaks=0，TSan 用 report_thread_leaks=0 与进程级
setarch；原 NVIDIA LSan/诊断线程 INCONCLUSIVE 不改名 PASS，不抑制 data-race 报告。
只构建实际 factor/consumer 闭包，先保存 dry-run；如引入无关 ggml CUDA 全目录重编则先
定位接线，不借模块任务改公共 build/core。重型任务串行，最多4 jobs，FI 模板编译单独限1。
CPU 当前 A/A16路 PASS，正式采样尚在运行；CUDA worker 仅准备，未因此签任何运行结果。

原三棵 sanitizer 树的 ggml CUDA archive 已做有限独立复核并以同一只读脚本再核：各146个
对象/成员、242个派生源码/头逐 SHA 相同；外部依赖1479/1479/1481无缺失或晚于相关对象
的更新，五个 HYLE 私有依赖与原 F5/current canonical 同 SHA。全部对象有相应 sanitizer
与 SM89 参数。历史系统头未保存哈希，借用依据明确限于原工具身份与 Ninja 依赖 freshness，
不制造历史哈希证明。验证 hook 仅把 ggml 的旧未改 native archive 链接到独立 imported
archive；CPU/base、FI 与全部改动宿主照常编，保原三类公开 consumer，不改产品构建图。
同一脚本/完整命令及聚合 SHA 将随本阶段 raw 封存；此处仍不是 CUDA 新域运行 PASS。


### 合格 CPU P1 失败与 CE12 有限性能修正

[原完整结果](../../../../evidence/performance/inference-g04-g05-cpu-encoded-w1-p1/receipt.json)
保留两次尝试累计1182.344秒、16 A/A、120产品与30 controls；qualification QUALIFIED、
verify退出0，性能 FAIL。window D256比0.95144–0.95661，5%非劣 UNRESOLVED；长prefill
D256比0.97497–0.98384，非劣 PASS；D64/Dv128 Q>KV留出比0.80083–0.80431，非劣 FAIL。
三项相对1稳定方向均 LOSS，不能把 prefill 的非劣 PASS/band PARITY 改称性能持平。
不增加样本、改数据/阈值或缩去失败留出域，原结果永久保留。

独立只读源审定位 ggml_cpu_provider.cc 的 Q8 query guard：逐元素 cache_input_value 对
BF16调用外部 GGML_API ggml_bf16_to_fp32，原头本身指出可直接位移16位；IPO OFF 无跨
TU内联。留出每episode新增9216次、window新增36864次标量调用与检查。跨strata实测
额外约19.674/15.327us；这是实质机制候选，尚不把全部差异归因给这一项。

只将 compute_provider_ggml 从 CANDIDATE退回AUDITED，baseline仍982676/f772；其它十个
owner/原CPU kernel与已验三交付不重开。CE12有限FIX：把Q8查询的已准入F16/BF16/F32
finite/range验证按dtype在循环外派发，使用类型正确的inline位表示比较/转换，保留同一
65504*127界限与输出映射前拒绝。不得删安全检查、改变错误/恢复、public API、layout、
新增常态state/分配、私造SIMD调度或改attention数学/页地址/ggml kernel/benchmark数据。
批量验证原编码/attention/Direct/Graph与数值边界、CPU三宿主风险和两安装配置；未变头/
GPU/math/原G03证据按输入与实际路径借用。重新冻结后只对同合同、同原仪器/样本/5%门
做有限P1，若仍有material差异再按实际机制裁决。当前CUDAASan只执行已归档f772输入，
不得从canonical反向覆盖或在其运行时再启动本机重型验证。

### CUDA 新域宿主 ASan 实际结果

[本阶段记录](../../../../evidence/correctness/inference-g04-g05-cuda-encoded-host-asan/receipt.json)
绑定冻结的 CE12 前输入与实际 RTX4070/SM89：十项 native 数学、三个 Direct/Graph/CPU+CUDA
增量消费者，以及五种 source 失败/unknown/晚到/放弃/foreign queue 风险全部通过地址检查。
两次尝试累计已测509.217秒；第一次编译成功，测试启动脚本误用 bin 路径，保留原 FAIL 与
启动耗时 unknown。第二次核对完整源码及实际二进制 SHA 后复用同一产物，仅续跑测试，未重编。
原146对象 ggml CUDA archive 复用证明、完整宿主/FI sanitizer 命令和原始日志已封存。
本阶段不签 LSan、UBSan/TSan、shader/driver 或 private-pool 上界；后两种宿主检查仍待运行。

CE12 查询验证使用原类型的 IEEE magnitude 位顺序，循环外按 dtype 派发；F16 只需排除
Inf/NaN，BF16/F32 保留同一65504*127界限。独立规则遍历每种16位格式全部65536种表示，
与原 finite/range 判定无差异；这是源码规则复核，不冒充编译后测试。新增公开 consumer 对
20种 F16、26种 BF16 表示各检查首尾，共92个 episode，涵盖正负边界/次正规/Inf/NaN、
实际输出及 poisoned slot 生命周期。实际 source、三宿主风险与两安装配置结果尚待收集。

CE12 第一次普通 factor 构建发现原 provider 测试 target 漏接真实 Batch/CPU route fixture：
共享链接49个 unresolved，产品 provider 与新增测试 TU 已编译。该实际验证接缝只补
provider 的 TEST_DEPS compute_route_cpu 及测试 target 的 compute_batch/compute_route_cpu；
不改任何产品依赖、公开头、shipping target 或构建核心。原 FAIL/累计180.640秒保留，
同一缓存树补接后仅续编选中的三个 factor/common target。

selector 对仅 TEST_DEPS 的中央声明变化误选全库，已登记
[独立工具反例](../build/validation-selector-test-only-component-deps.md)，保留原两份 plan；
本轮按已证明的测试接线影响执行上述有限 gates，不重写 selector 或重编无关产品。

GPU 性能 peer 准备先做独立数值预检：原始 GGML Vulkan full-F32/non-MMQ shader 已有
Q8 type8/block34 和独立 HSK/HSV，不改 shader 算术。benchmark adapter 扩独立 K/V
carrier/stride、Dv split/reduce 与有限 window 冷 F16 mask；保 query/output 转换和原生
slot/completion。预检覆盖14个编码形状（含 D256/Q65、Q>KV、split、window0/16）与
两个原 plain F16/F32 控制、两槽重放/readonly payload。仅准备数值与承载资格，不签 P1。
原 Air SPIRV 必须逐 hash 核对原456 clean peer，并从其二进制重建一致 header；借用
shipping static 缓存仅编两个 benchmark TU，dry-run 若牵连无关产品重编则先查接线。
Air Mac16,12/arm64/macOS27.0.1/AppleClang21身份已实查；仍待 CPU 批次完成后串行实跑。

### CE12 普通门完成与有限 CPU 复测候选

[实际普通验证](../../../../evidence/correctness/inference-g04-g05-ce12-cpu-validation/receipt.json)
绑定 CPU product5f9b261bfe6cfabdcbaf9fd84b693c35763b93a7：source22 PASS；provider、CPU
route、Graph CPU 与 CPU executor 四个 CTest 全部 PASS（42.56/617.96/616.47/0.24秒）；
三种宿主 sanitizer 各17测试＋9风险 PASS，ASan含LSan，UBSan含float-cast-overflow。
relocated static/shared 各14数学、3公共消费者及12独立头 PASS。已测阶段累计2078.734秒，
包含首次49 unresolved链接FAIL；封存脚本旧CTest摘要文字断言FAIL/耗时unknown另保留，
只修本次打包匹配，没有因此重跑通过项目或修改公共工具。

只将 CPU provider 冻结 CANDIDATE@5f9b261bf，其它十个 owner 仍为982676输入。公开头、
API、对象布局、错误/退休和 attention 数学未变；F32内部 guard 分支不新增 F32 Batch资格。
CPU 新 SDK 绑定见[复测来源记录](../../../../evidence/correctness/inference-g04-g05-ce12-cpu-validation/p1-rebind.json)：
仅更换 installed SDK prefix 与 supporting provenance hashes，原形状/fixtures/ownership/
16 A/A/n15/31/5%门及所有其它合同字段经结构化比较不变。artifact/provenance身份变化应由
新 manifest 记录新 hash，不冒称原 d42锁未变化；原 QUALIFIED/FAIL 证据不覆盖或改名。
复测未运行，当前不得写性能改善或全G04/G05 PASS。

### Vulkan 独立 Q8 peer 数值预检完成

[实际预检](../../../../evidence/correctness/inference-g04-g05-vulkan-encoded-peer-preflight/receipt.json)
实际 Air/MoltenVK 原456 shader，14编码＋2原plain控制 PASS，编码各两slot四次重放，
readonly K/V逐字节不变；最大绝对误差0.0000304988602。长KV case实际四dispatch包含
split reduce。已测21.318秒，原 shader generator 为 no work；实际构建两benchmark TU
及25个SIMD/State依赖TU。原 dry-run停在CMake重生成，未展示完整后续计划，实际日志
如实封存；已核两个代表依赖的源码SHA/编译命令与原shipping SDK一致，不声称只编两TU。
这里只签原始独立 shader adapter 的有限数值承载，不签P1/A/A、mask-opt、多batch、
producer/growth/capture 或 CPU CE12；量化causal固定mask的O(Q*KV)冷存储明确计费。

### Vulkan 多请求编码页比较的构造前合同

下一步仅构造 benchmark adapter，不改变冻结产品或启动完整采样。正常用户沿原
PagedBatchInvocation 两槽：prepare_batch(layout)→bind(真实 Q/K/V exact frames)→submit→
wait→取得输出并释放。peer 沿原同一 MaterializedNativeState/PagedMetadataPolicy、F1/F2、
native Program/Invocation 和 retirement，只把数学 recipe 换为独立原456 Q8 shader；
逐请求压缩行 Gather 后接已预检 dense shader，独立 Dqk/Dv 和 bottom-right window。
原 shader 算术不变，不使用 candidate attention lowerer。原 host 可能选择 MMQ/整历史
F16 展开的事实保留；本 adapter 明确选原 non-MMQ direct-Q8 pipeline，不签最佳原 host。

两侧每 operation 都计入 prepare 的 Program/Invocation 创建、完整页索引 F2 上传与
host wait、exact permissions、bind/submit、native 完成与 output owner 返回/释放；peer
额外压缩页 gather、Q/output 转换和输出拼接全计费。当前 submit 消耗 prepared/bound，
不能在 benchmark 私下恢复标志以伪造 warm replay。长期不可变历史的初次 payload 上传、
schema/layout 创建和槽建立属冷成本；按实际 prepare 创建的 mask、scratch、program
仍计入每 operation，不能因内容不变就标为冷成本。

有限形状沿 CPU 三组：Q1/17、KV1024/768、D256、window16；Q64/128、同 KV/D256、
unbounded causal；Q17/1、KV7/47、Dqk64/Dv128、full noncausal。Vulkan Q/output 为 F16，
K/V 同 Q8_0 bytes，页16、GQA8/2，重排及共享页、inactive poison，独立 stored-byte double
oracle、两槽重放和 readonly bytes。构造后先执行数值/完整生命周期预检，再提交不可变
benchmark 合同与实际源码时序账；A/A、双执行/分配顺序和原5%非劣门通过前不完整采样。
本合同不签 G06 更新/增量 producer/capture、native private-pool 上界或端到端 token/s。

三个 benchmark adapter 文件已完成源码接线；独立只读复核未发现新的 MUST，确认 stored
Q8 oracle、独立 K/V stride、child constants/alias 保活、exact frame 和两槽重放/完整计费。
这是静态复核，尚未编译运行，不签数值或 P1。原 selector 对三个共享 bench 路径仍
unknown，原 BLOCKED plan 与146 skipped reasons保留在独立工具记录，不改公共 selector。

构造后静态账发现，原先逐页三输入 Concat 在112个引用页的形状要产生76个 gather
dispatch，不能把该人为分组作为成熟 peer 的固有成本。尚未运行该 adapter，现改用已有
lower_gather 的原 GetRows I32 位复制路线：每请求共享一份 logical-row→physical-row
I32 索引，K/V 各自按编码 row bytes gather，无量化解码或新 shader。索引生成/上传、原
index normalization、生成的输出复制和所有 scratch 都在 prepare/operation 费用内；
无论页数如何只构造两个请求各两个 Gather。原 attention shader 仍独立、算术未变。
此修正发生在数值预检和任何采样之前；上述旧接线静态复核不代签新 Gather 的运行资格。

[逐阶段静态成本账](../../../../evidence/architecture/inference-g04-g05-cost-preflight/vulkan-page-preconstruction.json)
绑定实际源码 hash：两组 D256 每 operation 的 peer 有效压缩 gather 为1949696 bytes，
window/prefill mask 分别28160/327680 bytes；D64/Dv128 holdout gather22032 bytes、无mask。
这些是逻辑工作量，不是 driver/private-pool 存储上界。GetRows 原始生成器840明确
TEMP/A/D 为 uint；已有 U8 Gather 仅重解释位模式。独立复核新接线未发现 MUST，
同时明确 K/V 各有 normalization 和输出 Copy，不能只记两个 GetRows。实际生成 dispatch、
额外复制与 native 分配尚待串行预检，不由静态账签 P1 或零 allocation。

正式比较的[独立 SDK 构建入口](../../../../bench/inference_resident_sdpa/vulkan_encoded_p1/CMakeLists.txt)、
[合同](../../../../bench/inference_resident_sdpa/vulkan_encoded_p1/contract.json)及
[费用/来源](../../../../bench/inference_resident_sdpa/vulkan_encoded_p1/consumer-provenance.md)
已备齐，尚未运行。它只构建比较程序/control，直接链接已验证的 relocated static SDK，
不会重新构建产品库。29个HYLE库与原封存哈希一致；三个原生archive整文件哈希不同，
但有序对象成员152/9/15逐字节一致，原不等哈希与成员证据同时保留，不冒称整库文件相同。
A/A 必须先读取实际 numerical/lifecycle receipt，再核对五个benchmark输入、公共/Vulkan
产品输入、original SPIR-V、SDK库和installed headers；16进程与八strata、原controls、
n15→31、5%门不变。独立静态复核无新增MUST；合同schema校验通过不代表构建、A/A或P1。

数值运行前进一步移除 reference helper 每次 prepare 的 mask metadata fprintf，防止
peer 额外日志污染计时；冷形状字节账保留，数学/上传/owner未变。此前Vulkan队列仍仅在
等待、没有子进程或重型运行，已保存取消记录并换新源码快照；CPU扩样/后续CUDA不受影响。

### Apple 原始 peer 的资源与完成接线复核

原456源码与文件SHA已补入上述 peer-preflight.json：graph_compute 返回同步前的异步状态，
Metal synchronize 的晚到 CommandBuffer 失败只写内部 has_error，不能把该返回值当作本次
checked completion。原 host-pointer import 会扩大到 OS 页边界，真实 exact page lease
也不能单独证明这个扩大映射合法。这两项是独立比较器的接线约束，不是新增产品缺口。

有限 adapter 优先沿原私有 encoder 的真实 MTLBuffer/offset 桥，保活原 owner、设备与精确
范围，并检查本次实际 CommandBuffer 完成状态；保留原 general Q8 padding、mask、split/
reduce 与全部转换/分页 gather 费用。64/128 只补原模板的 benchmark 显式实例，先实机
编译和字节 oracle，再进入 P1；原 shader 算术不改，明确披露 general-kernel 选择，
不声称原 host 最佳性能。当前仅源码资格准备，未编译、未签数值或性能通过。

### CE12 CPU 同合同复测的实际裁决

[完整复测](../../../../evidence/performance/inference-g04-g05-ce12-cpu-encoded-w1-p1/receipt.json)
绑定原不可变 source993fc414e 与 CPU candidate5f9b261bf、已重绑的实际新 static SDK。
16 A/A PASS，qualification QUALIFIED、verify退出0；原自动 n15→31 策略触发第二阶段，
两阶段368产品＋92对照全部封存，最终统计仅用 n31、不合并样本。原5%门/形状/顺序/
数据不变。本次3334.563秒，含原两次尝试累计4516.907秒，原 qualified FAIL 不覆盖。

window D256 八strata比1.16564–1.17288，非劣 PASS、方向 WIN；长 prefill D256比
0.98482–0.98704，非劣 PASS/band PARITY，稳定方向仍 LOSS。64/128非因果Q>KV留出比
0.94915–0.95397，七strata非劣 UNRESOLVED、一项 FAIL，聚合 FAIL；相对1方向均 LOSS。
因此整个 CPU P1 仍 FAIL，不能把主要场景改善代签留出通过，不能继续加样本或改阈值。

provider 保持 CANDIDATE，先对实际留出残余成本作有限源码/生成代码定位，具备机制与
真实消费两把钥匙后才回 AUDITED 修正。CUDA UBSan 已在同一串行队列续跑，TSan随后；
其固定冻结输入不受 CPU 结果影响。Vulkan 数值队列继续等待两种 CUDA 宿主检查结束。

### CE13：有限 CPU 页地址与不可变描述成本修正

baseline 为原 CPU provider5f9b261bf、runtime982676；其它九 owner 的冻结输入不变。
CE12 合格留出失败与真实量化分页用户的逐请求/逐 token 路径共同构成修正依据。
实际 shipping SDK 的 AVX2 DLL SHA143054661c3c 与原 producer 一致；该 producer 的
ops.cpp 源码 SHA13ac378b7a33 匹配当前 patch manifest。MSVC14.50 反汇编确认原 generic
循环在0x0808/0x08C3各调用一次 hyle_flash_row_offset，未内联或消除重复；页16本已
shift/mask，不把它描述成两次动态除法。留出每episode1328个 KV token，共2656次调用。

只将 compute_ggml_runtime、compute_provider_ggml 退回 AUDITED，同卡固定两项 FIX：
原 generic 循环在有效 token 上只求一次 physical row，K/V 使用各自 stride；原 CPU
batch Request 冷阶段保存四个已验证 byte extent，热路径仍校验实际 buffer 范围并创建
原 TensorView，保全部 resource/输出权限检查。后者每请求增加四个 size_t（64位32B），
计入原 sizeof(Request) workspace 账；不建缓存/索引或改变公开对象。

原 Q8 query finite/range 检查及输出映射前失败、页表 identity/保活、非因果/window边界、
线程池/plan/queue/completion、数学顺序和原 peer 均保留。仅改原 CPU patch bundle、锁的
tree/license身份及 native_batch 私有实现，不改公共工具、构建图、公开头/API、GPU产品。
两项一次实现后批量验证 source 的数学/页/Direct/Graph/数值边界和短资源拒绝、CPU三类
宿主风险与 static/shared installed consumers；原 GPU/native 数学和 G03 证据按未变路径
借用，Apple 混合发行中的 CPU 另做有限运行。以原输入/失败/累计为基线，不重置预算；
冻结新 candidate 后仅做原同语义 P1，不靠改形状/阈值/采样计划取得通过。
独立只读复核未见新的 correctness MUST；具体收益与 locality贡献仍未知，不由源码签PASS。

两处实现草稿已完成：同一地址改动同时进入 ffa CPU 与 mixed456 的原 patch context，
原 mixed IQ1 19项组合器未改且实际推导通过。两个 in-tree snapshot、NOTICE/lock 原始
身份校验已通过；首次打包哈希按 Windows Path 忽略大小写排序而失败，原 FAIL/unknown
耗时保留，改回原 CMake 的 portable-string 顺序后通过，不改公共工具或产品算法。

本次 lock 仅更新这两个 bundle 的 tree 与 NOTICE hash；selector 因中央 lock 选全库的原
plan保留。依用户限定的模块/直接消费者范围，实际 source 选 runtime/provider 两factor
及原 public encoded/paged/ragged Direct/Graph，借用未变 route 和通用头证据；不运行无关
io/container/其它 provider 或整 Graph CPU 回归。CUDA/Air 当前固定输入仍单独完成，
随后串行执行本 CE13 普通门；未运行的修复数学/风险/SDK/P1仍UNPROVEN。

### CUDA 新域宿主 UBSan 实际结果

[UBSan 阶段](../../../../evidence/correctness/inference-g04-g05-cuda-encoded-host-ubsan/receipt.json)
绑定与 ASan 完全相同的冻结 archive：实际 RTX4070/SM89，13数学/Direct/Graph/联合测试与
五种 source 风险全部 PASS，已测482.431秒。未改146对象 ggml CUDA archive 按原同配置
证明复用，CPU/base/FI 与改动宿主正常编译，实际测试二进制前后 SHA 一致。TSan 随后已
在同一串行队列启动；本阶段不签 shader/driver、LSan、private pool 上界或 CE12/13 CPU。

[TSan 阶段](../../../../evidence/correctness/inference-g04-g05-cuda-encoded-host-tsan/receipt.json)
现已完成同一冻结输入的13数学/消费者与5 source风险，数据竞争检查 PASS，已测453.707秒。
ASan/UBSan/TSan 宿主三方面至此均有实际证据；原 NVIDIA 诊断线程 lifetime/LSan 的
INCONCLUSIVE、不签 shader/driver/private pool 上界和 CPU CE12/13 的限制仍保留。

### Vulkan 编码分页完整比较旅程的实际数值预检

[完整预检](../../../../evidence/correctness/inference-g04-g05-vulkan-encoded-page-preflight/receipt.json)
实际 Air/MoltenVK 三形状、双方实现各两槽四次重放全部 PASS，最大绝对误差
0.0000304865893，readonly K/V逐字节不变；四次尝试已测阶段累计24.131秒，外层观察
26.016秒另列且不重复相加。原第一次在 force-glob dry plan 截断前停止，旧阶段状态
未因观察结束改写；实际 PID已消失且无子任务。随后 SDK变量名、complete初始化行数的
configure/compile FAIL 均保留；只修比较器接线，不改产品或数学。

最终采用独立 SDK 消费工程，逐 SHA 核原已验证 static库/配方/installed headers，仅编
四个benchmark TU，无产品库/着色器重编。实际 configure 后用字节相同的外部 build.ninja
副本只读查看完整计划，真实构建仍用原 manifest；不修改公共构建工具或 producer cache。
全部物理页原本已完整上传，complete现在明确传16个初始化行；逻辑tail、poison、exact
frame与费用合同不变。这里只签数值/生命周期，不签 A/A、P1、G06 或整批 G04/G05。
CE13 已在这些任务实际结束后启动普通门，保持所有重型任务串行。

CE13 Windows source 第一轮已通过：22项原数学/分页/ragged/Direct/Graph/数值边界，
以及 provider、runtime/source合同、两种资源失败夹具共5个定向 CTest（42.65秒主测试，
全部 CTest 43.06秒）。构建129.532秒。实际 AVX2 ops派生 SHA18c32123603a 与新锁匹配，
generic 反汇编已无原两次外部 row-offset helper 调用；只把它作为已消除重复调用的生成
代码证据，不签性能收益。三类宿主 sanitizer 与两种 SDK 仍沿实际串行队列执行，
CPU provider/runtime 保持 AUDITED，尚未冻结新 P1 candidate。


### CE13 CPU 普通门完成，混合发行有限验证启动

[普通门证据](../../../../evidence/correctness/inference-g04-g05-ce13-cpu-validation/receipt.json)
已封存：22项源码消费者、5个定向 factor CTest、三类宿主 sanitizer 各17数学/消费者与9风险、
static/shared SDK 各14数学＋3共同消费者及12独立头全部 PASS。实际阶段累计815.230秒；
外层845.079秒与阶段重叠，不重复相加。原 identity 包装失败与未知时间、CE12 合格性能
FAIL 全保留；生成代码消除重复调用不直接作为性能改善。

同一批重型任务已实际结束后，Air 混合发行的 CE12/CE13 CPU 数学验证才启动。复用与当前
native 依赖逐项核对的152对象 Vulkan archive，只重编受影响 CPU 和原22项消费者；实际
计划若包含 GPU/shader 重编则先停止。验证用接线留在可丢弃副本，不修改产品构建系统。
本次不代签新 GPU、完整混合 SDK 或 P1；CPU 两 owner 仍 AUDITED，待该有限验证与新
不可变 candidate 性能裁决。


### CE13 混合 CPU 有限资格通过与新候选冻结

[Air 混合 CPU 证据](../../../../evidence/correctness/inference-g04-g05-ce13-air-mixed-cpu/receipt.json)
实际 Mac16,12/AppleClang21/arm64、mixed456 runtime、无 HYLE_TESTING 的22项原消费者全部
PASS；成功轮 configure12.950秒、CPU compile10.850秒、tests24.074秒，连同原两次
接线/计划拒绝累计73.867秒。152对象 native Vulkan archive 按依赖与 SHA 证明复用，
实际最终构建未包含 GPU/shader；测试用 archive 同字节副本避开 Ninja 原输出的重建规则。
原配置依赖可见性 FAIL、意外重编 dry-plan 的预编译拒绝全保留，不签新 GPU/完整混合 SDK。

CPU provider/runtime 以既有完整产品 commit42a4c7cdcc0822a7b162efe575909622a906de1d
进入 CANDIDATE，其余九 owner 的候选不变。普通门与本有限混合消费均完成；没有追加
API、算法、公开形状或常态成本。CPU 同合同 P1 只重新绑定已实测 CE13 static SDK 与
provenance hash，三形状、两槽、16 A/A、执行/分配顺序、n15→31、5%门及原 peer 不变，
[重绑定记录](../../../../evidence/correctness/inference-g04-g05-ce13-air-mixed-cpu/p1-rebind.json)
用规范化结构相等核验。原 qualified FAIL 不覆盖；新 P1 尚未运行，G04/G05 仍未关闭。


CPU CE13 P1 已从干净不可变 source5db68c92a136c2cdff9d0df621394d0bd85b57aa
实际启动，使用新 static SDK；16组 workload-matched A/A 全部 PASS，已进入原正式采样。
没有依据部分样本签性能结果。原 CPU 进程实际结束并封存后，才允许队列启动独立 Vulkan
同合同 P1；重型任务继续串行，编译最多4 jobs。


Apple 独立 peer 的原源码 materializer 已实现并实际运行，生成
[source-only 证明](../../../../evidence/architecture/inference-g04-g05-cost-preflight/apple-peer-source.json)：
六个原456输入 SHA 全匹配，沿原 CMake 内联，去除显式导出行后的 general/pad/blk 原文本
逐字不变，只保留三个原Q8实例及一个明确新增64/128模板实例。保留原 general 直接终态
输出；没有选择 vec/split/reduce。此处只签源码生成，不签 Metal 编译、数学或 P1。
独立只读接线审计已确认沿 MLX 原 CommandEncoder 绑定实际数组、追踪依赖并检查实际
CommandBuffer 完成；mask查询轴必须按原Q=8补齐，K/V按原head-major行布局，不能用
general尾部判断掩盖blk读取越界。后续有限桥保持原费用账，不扩大产品域。


### Apple 独立比较器实现批（尚未运行）

benchmark-only Custom、完整分页消费者及独立 SDK CMake 已集中实现；沿原 MLX encoder
登记实际 Q/K/V/mask/pad/blk/output，沿原456 general Q8 数学，保留实际 MCB 直到原
HYLE Completion 后检查。Q→F32、head-major Q8 gather/contiguous、F16 mask、原 pad/blk、
F32→F16 输出与原 prepare/F2/权限/两槽/等待均计费，double oracle 与完整 readonly 检查
不变。原 host 其它 vec/全KV展开选择不参与本次显式 general peer，不称原host最佳策略。

[一次批量源码交叉检查](../../../../evidence/architecture/inference-g04-g05-cost-preflight/apple-adapter-review.json)
发现的唯一 scratch 接纳前缀生命周期 MUST 已修正：临时 owner 与真实 MCB 在首个 dispatch
前登记，后续 pipeline/launch 异常由原 MLX/HYLE drain 持有。修正版无未解决源码 MUST；
这不是编译或数值 PASS。原 source materializer 已按原 SHA 实际生成，并可原样复用。

独立工程只链接已验证 Apple static SDK。28个 HYLE 库、6个 native 库、10个配方和
1个 metallib 的固定身份保存在
[SDK 清单](../../../../bench/inference_resident_sdpa/apple_encoded_peer/qualified-sdk-artifacts.json)；
native库与原producer整文件不同，但逐成员名称/顺序/对象字节一致。不据此猜测元数据差异
原因，CPU域仍是旧SDK，不代签CE12/CE13。首轮只编3个benchmark TU并执行原三形状、
双方两槽四次重放与readonly预检；必须等CPU与Vulkan实际重型进程终态后串行启动。


Apple 预检后的正式 P1 包现已准备并通过 contract validate，尚未编译/A-A/采样。
[合同与消费者账单](../../../../bench/inference_resident_sdpa/apple_encoded_p1/consumer-provenance.md)
固定原3形状、8strata、16 A/A、双执行/分配顺序、placement、n15→31与5%门；
实际数值预检、原 shader/header、SDK/object身份和product输入不匹配即停止。数值成功后
才封存/提交对应 receipt，再从新干净不可变源码启动 P1；没有数值 PASS 不会进入采样。
初次 device library/pipeline JIT 与缓存预热计入双方未计时数值/预热阶段；之后每episode
的 native prepare/F2/权限/实际数学/完成/output release 仍全部计费。CPU、Vulkan和Apple
重型阶段由实际进程/远程终态串行交接，不以观察超时启动下一项。

CUDA 独立 peer 的有限公平性裁决已补入[原peer账单](../../../../evidence/architecture/inference-g04-g05-cost-preflight/peer-preflight.json)。
原69ff短K/D64-Dv128 FP8 kernel已有错误，不能复用同一产品修复签独立算法。
本轮 immutable history 应允许与CPU相同的冷只读替代cache：FP8精确页→冷阶段实际GPU
解码成额外T16 snapshot，保留原frames并每次核identity，warm走原未修复T16 FI与原16-bit
split/merge；冷初始化/检查完成/释放与额外capacity bytes单列。没有真实producer/version/
tail/page/scale变化时，反复warm全历史解码会人为降级peer，因此不采用。只裁决该有限
adapter，不签实现/数学/P1，也不称原生直接FP8、动态增量或best-host比较。


### CUDA 原版 BF16 快照比较器实现批（尚未编译）

benchmark-only native TU 使用独立 namespace 包含未经修改的原69ff FI headers，
避免与产品修复模板/内核合并。原 FP8 exact frames 冷阶段按授权物理页/有效行的并集
解码到额外 BF16 snapshot，保留原 page IDs、独立 K/V stride、原 owner 和 scale；
冷 decode 通过原 Queue/F1 实际完成后才 ready，两槽共用只读副本。每个 warm episode
仍走原 Batch update、metadata/F1/F2、原 frame range/identity 检查、原 BF16 paged
attention/原16-bit split-merge、实际 Completion/output owner 返回与释放。

三组 workload 几何保持原合同，query/output 选择 BF16，K/V FP8 scale 为 .5/2；
独立 fixture 使用非零 Q 和相关 K/V，double oracle 解实际 FP8 bytes，完整 physical
readonly 与双方两槽四次重放均已接入。额外 snapshot capacity、实际 decoder read/write、
冷初始化与 teardown 秒数单列，native private-pool upper 仍 UNKNOWN。原 CUDA static
SDK 的 CPU 域早于 CE12/13；此次只验 GPU bridge，不代签新 CPU。

新增独立 SDK CMake 只编 benchmark 与一个原 FI TU（该 TU compile pool=1，整体最多4），
不重编产品或 native 依赖。当前仅实现/源码身份与普通文档检查；未编译、未执行数值，
没有 A/A/P1 或 G04/G05 PASS。真实 CUDA 预检须等待既有 CPU→Vulkan→Apple 重型阶段
实际结束，再按同一有限批次构建；首轮出现的问题集中记录后定向修正。


### CPU CE13 同合同性能通过，剩余路线串行资格

[实际封存结果](../../../../evidence/performance/inference-g04-g05-ce13-cpu-encoded-w1-p1/receipt.json)
为 QUALIFIED、performance PASS、verify0。16 A/A 通过，原自动 n15→31 完成，共368产品/
92 controls，最终只用 n31 的248产品/62 controls，不混合两阶段。当前3337.063秒，保原
两轮失败累计7853.970秒。windowD256 比率1.1716–1.1787，WIN；prefillD256 0.9899–0.9939，
5%带内 PARITY、方向 INCONCLUSIVE；noncausal Q>KV D64/Dv128 0.9654–0.9685，5%带内
PARITY、方向 LOSS。三组非劣全部 PASS。停止重复此已签 CPU 有限域，不代签 producer/
growth、G06、GPU、整个 G04/G05 或 token/s。

Vulkan 首次接力被 canonical-clean guard 在远程启动前拒绝：当时新 CUDA P1 文件正在
提交，原5.5秒失败保留，没有真实 Vulkan run/采样被重置。源完成提交后，实际首 run
从 clean cee599774 在 Air 启动的[正式结果](../../../../evidence/performance/inference-g04-g05-vulkan-encoded-m1-p1-attempt-1/receipt.json)
现已封存：QUALIFIED、performance PASS、verify0，16 A/A 与 n15 的120产品/30 controls。
windowD256 比率5.6315–5.6856、prefillD256 3.8990–4.0220、noncausal D64/Dv128
5.5225–5.5900，三组方向 WIN、5%非劣 PASS。实际远程1104.133秒；bundle/deploy另列，
外层观察与远程重叠不重复相加。只比较原456 general Q8路线，不称原host最佳策略或token/s。

CUDA [有限 P1 合同](../../../../bench/inference_resident_sdpa/cuda_encoded_p1/consumer-provenance.md)
已通过 contract validate；源码原始 archive 与整个 include 内容实际匹配，332个 GPU SDK
产品输入与当前 canonical 相同。只读源码交叉检查未发现原实现批 MUST，随后补了自有
BF16 padding 清零及强 oracle witness；这些都未冒充编译/数值 PASS。首次源码检查误把
archive extraction 当作 Git checkout 的失败与未知时间保留；已改为固定 archive SHA 和
逐 include 字节核验，真实 Git 模式禁止向祖先仓库回溯。

[CUDA 数值预检](../../../../evidence/correctness/inference-g04-g05-cuda-encoded-page-preflight/receipt.json)
现已实际通过：三形状、双方两槽四次重放共24正确性结果、3组pair与完整readonly，最大
绝对误差0.00109081301，小于原8e-3门。四轮已测阶段累计126.802秒，原源码身份误判及
未知耗时另保留；host Status 类型编译FAIL、重用对象枚举FAIL、日志字面量换行导致计数
FAIL均保留。最终只重编1个benchmark TU，另外3对象按SHA复用，产品与原生依赖未重建。
正式 P1 已从干净95276662bc2fb0103a53e6a2c8663c1042b43f91启动，尚无性能结论；
实际 W1-WSL/ext4工程资格不能签 native Linux/Formal。

Apple 首轮在配置前因验证脚本要求该SKU未安装的CUDA头而失败；脚本现仅核所选SDK头，
并核真实compiler依赖中的HYLE头。第二轮3个benchmark TU中的reference编译因原Metal
230690字节字符串触发overlength警告升级而失败；现保留原字节和警告，仅解除该警告的
Werror。两轮失败/耗时保留，未取得Apple数值或P1通过。修正后的第三轮排在实际CUDA
终态之后；重型任务串行，最多4编译jobs，不重跑已通过的CPU/Vulkan有限P1。


### CE14 Apple 量化 prefill 的合格性能反证与有限修正

baseline `89236554cf6a3efb6698fefbb013948e4ea53597`，原产品 candidate982676cb05不改写。
[Apple 实际 P1](../../../../evidence/performance/inference-g04-g05-apple-encoded-m1-p1-attempt-2/receipt.json)
为 QUALIFIED、verify0、16 A/A PASS、120产品/30 controls；窗口D256比率1.1841–1.1942与
非因果D64/Dv128比率1.0776–1.0837均非劣PASS，但D256 prefill仅0.3249–0.3255，明确LOSS/
非劣FAIL。738.364秒及原设置失败保留，不重复无变更的同一性能运行。

源码已核：Q8为保65532有限解码值而使用F32 query/tiles/partials，prefill落到旧steel
F32路径。原MLX is_nax_available按实际GPU generation判定；本机Apple M4 g16不满足
非p架构gen>=17的原条件。先前NAX试改的源测试11数学/4公开页与原风险虽通过，但没有
执行新NAX分支，不能作为性能修正证明；该试改已撤回，配置hash错误及实际60.146秒保留。
仅compute_route_apple保持AUDITED；原产品和其他owner的候选不动。

有限FIX CE14改为实际M4 STEEL承载：沿原矩阵乘、online softmax、F32 score/输出及原
Q8 tile/page16 loader检查计算精度与tile形状，直接读取驻留编码，不全历史解码。必须保
65532及长prefill/消去边界、字节stride和尾行权限；不得用half临时溢出、缩小prefill范围、
放宽5%门或强开本机不支持的NAX。内部tile固定为Q8 D256 unbounded的BQ8/BK16/WM1，共28800-byte共享内存；原BQ16/BK8的28928-byte不降精度。复用同一F32 loader/online softmax/MMA，移除Q8 V矩阵每次加载前后重复的无内存SIMD fence；原threadgroup staging barrier及QK收敛点保留。
公共API、ownership、失败、权限、退休及未变路线不改。

预声明验证：同一批已有public encoded页/Graph/联合消费者与原Apple native cache数学
（F16/BF16、D64/128/256、ragged/page16、tail/重排、Q65以上、65504编码边界及两槽），
保原double oracle门；新增断言只核新loader/选择/字节stride和有限数值边界。实际Air源、
适用三种host sanitizer及static/shared对应库/消费者定向验证；未变公共头与原G03风险按
输入身份复用。native输出/metadata/临时内存账检查无全历史decode。通过普通门并冻结新
Apple候选后才重绑定同一三形状/16 A/A/双顺序/n15→31/5%合同运行有限P1；不重跑CPU/
Vulkan/CUDA未变产品域。调查/实施/验证各2h检查点，沿用户授权追加且原成本不重置。

CE14 的 selector 原计划保留 READY，但依赖账本单一内容hash变化被保守扩成全产品。
结构对照已证明仅 mlx_cast_patches.tree_sha256 改变，17份原始/派生源码逐SHA可重演；
组件图、其它依赖、公开头与package seam均未改。按本卡只运行Apple owner及其原
页/Graph/联合消费者和三交付/适用风险，逐gate保留语义未变的skip理由；不运行全库，
不修改公共selector或构建核心。此证明只签源码/归属，不签新数学或性能PASS。
