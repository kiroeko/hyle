# G04/G05 编码 KV 与有限 attention 语义实施批

状态：**AUDITED，2026-10-15**。baseline `694a3d1a4b02540cb56b6ceae35b73dd1dd93440`；
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
