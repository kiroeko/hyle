# G04/G05 编码 KV 与有限 attention 语义实施批

状态：**Apple CE14 CLOSED3b5157532；Vulkan provider 的 CE16 已固定 CANDIDATE；其余 owner 保持原 CANDIDATE，2026-10-15**。baseline `694a3d1a4b02540cb56b6ceae35b73dd1dd93440`；
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
放宽5%门或强开本机不支持的NAX。BQ8/BK16/WM1及移除Q8 V-load无内存fence的试改已被短成本反证否决并撤回，保原F32
baseline。该试改的11项数学、4项公开消费者、原风险与static SDK虽通过，但双顺序prefill
比率仅约0.127，不能进入candidate；精度/生命周期通过不能代替性能修正。
[原始尝试与裁决](../../../../evidence/architecture/inference-g04-g05-apple-prefill-correction/receipt.json)
保留所有输入、配置hash错误、未执行NAX路径、三次shared-Custom比较桥接失败、实际
static比较与累计成本。接下来只核原STEEL的矩阵输入类型、mixed-input/F32-accumulator
机制与编译器活跃寄存器边界，先固定可保65532/小数值的承载再实施下一批；不重跑失败
配置、不放宽原数值或性能门。
公共API、ownership、失败、权限、退休及未变路线不改。

CE14 下一实施批固定为原M4 STEEL mixed-input机制：原`BaseMMAFrag<float>::mma`
已接受独立A/B/C类型，复用其half-input/F32-accumulator指令，保原BQ16/BK8/WM2及
原staging/收敛barrier。Q8解码和shared-memory数据仍为F32；每个8×8 fragment在SIMD
一致性检查确认所有输入为零或绝对值位于[2^-14,65504]后才缩窄矩阵输入。任何微小非零、
超half范围或非有限值仍走原F32矩阵调用，避免65532溢出和小概率贡献丢失。分支只使用
局部fragment，不增加公共状态、整历史转换或host读取；正常档沿原Metal Q8 peer的
half输入/F32累加精度，用原double误差门验证，不宣称逐位同F32乘法。
同批增加原公开消费者的低概率大V反例：Q65/KV8192、一个高score行和其余低score大V行，
要求原误差门保其非零贡献；与原高值消去/长prefill/独立Dv一次批量验证。数值或成本
反证失败仍否决实现，不以新增测试或调阈值保留它。

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

CE14 mixed-input 批现已否决并恢复原 patch/lock 字节，新增低概率贡献反例保留。
[实际记录](../../../../evidence/architecture/inference-g04-g05-apple-guarded-mma-trial/receipt.json)
包括首次 Metal half2 构造编译失败、修正后的 source 12数学/4公开消费者/原风险和
static SDK通过。双顺序短诊断的prefill比率仅0.1825/0.1827，比原合格0.325更差；
没有启动shared、sanitizer或正式P1，不将短诊断写为合格性能结论。父批与子阶段耗时重叠，
分别保留而不重复累计。原范围、阈值及其它路线不动。

下一有限修正只隔离原STEEL的寄存器fragment循环同步：保原F32数学、BQ16/BK8/WM2、
threadgroup staging及全部mem_threadgroup屏障；仅Q8分支跳过QK load后和P×V load前后的
mem_none SIMD执行屏障。上述循环没有shared写入，跨SIMD生产/消费由已有threadgroup屏障
覆盖，矩阵collective仍由所有lane在同一固定循环调用；plain路径保原屏障。此项与已否决
的BQ8/BK16/WM1合并试改不同，不能从旧合并结果裁决独立机制。
先一次source 12数学/4公开消费者/原风险、static SDK及原三形状双顺序短诊断；失败则
撤回，不追加无变更P1。若机制成本改善且数值通过，再执行原适用风险/shared与冻结流程。

上述独立同步试改已完成并否决：[source/静态SDK/短诊断原始记录](../../../../evidence/architecture/inference-g04-g05-apple-simd-sync-trial/receipt.json)。
12数学、4公开消费者及原风险通过，双顺序prefill仅0.3275/0.3237，未改善原损失。
原patch/lock再次恢复；不为这项无收益改动追加shared/sanitizer/P1。下一步先取实际
Metal编译管线资源与loader成本证据，区分shared容量、活跃寄存器和重复解码，暂不实施
新的精度、tile或同步试改。其它项目与资格保持原状。

CE14 下一实施批按[实际归因](../../../../evidence/architecture/inference-g04-g05-apple-loader-attribution/receipt.json)
固定两处私有修正，baseline325e964f2：对齐且每线程完整32元素块时只读一次scale，展开
该块code→F32 staging；其它loader几何保原实现。仅原始Query为F16的Q8 STEEL路径使用
F16 Query staging/fragment、BQ16/BK16/WM2；K/V staging、P、score、O与累加仍F32，保
所有原屏障。实际资源归因显示Q8原kernel与纯F32同几何占28928字节，Query降为原F16
后同样资源容量可容纳两倍BK。不是把K/V或小概率缩为half，不全历史解码或增添缓存。
原Query类型作为private primitive状态保留，进入equivalence/state和JIT key；BF16及plain/
vector路径保原数学，未增加公共参数、layout或跨模块owner。增加F16 subnormal Query
乘大K的原double-oracle反例，防止staging丢失小Query；与原12数学/4公开消费者/原风险、
static SDK和同三形状短成本一次批量执行。归因样例不代签13测试、完整生命周期或P1。
后续适用host风险、shared SDK及正式同合同资格仍按原流程；所有原失败继续保留。

本批[实际source/static结果](../../../../evidence/architecture/inference-g04-g05-apple-loader-query-stage/receipt.json)
为13数学、4公开消费者、原风险及relocated static SDK通过；短诊断prefill已升至0.8206/
0.8218，仍未达原非劣门，保持AUDITED。独立GPU探针确认每线程16元素也共享一个32元素
Q8块scale；原loader在此正常几何仍逐元素重复解码。下一有限修正将scale复用覆盖完整
块和对齐的半/子块：只读实际vec_size个code，不越过34字节块；F16 Query D256改回原
32-query宽度并用BK8/WM4，Query half + K/V float占29184字节，复用同一F32累加与屏障。
独立Q128/KV768核中该组合约0.687ms、原Q8核约3.08ms，输出一致；这仍不是完整episode/
P1。只对K加half范围guard的探针更慢、packed_char4没有收益，均未进入产品。固定这批
后再一次跑原13数学/4公开/风险/static/同三形状短诊断，不放宽数值或性能门。

CE14 当前产品冻结为`3b5157532`：[普通工程证据](../../../../evidence/correctness/inference-g04-g05-apple-ce14-ordinary/receipt.json)
已实际完成source13、public4、static/shared各13+4与CPU14+3/12headers，ASan/UBSan/TSan
各13+4及原风险，Metal shader validation实际13+4与独立enabled-state见证。ASan不签
LSan；TSan沿原两项显式模型，shader/driver/private pool不由host sanitizer代签。
逐hunk反查确认partial-block不越过34字节块、inactive行不读、private Query类型进入
equivalence/state/JIT key；默认QueryT保plain原float fragment，无公共API/schema/权限/
退休变化。原数值门和旧失败全部保留。短诊断三形状双顺序均大于1，prefill1.1429–1.1496；
这只允许冻结并进入原合同资格，不是P1通过。Apple owner合法进入CANDIDATE，其它owner
候选不动；新SDK独立peer数值重绑定、16 A/A和原n15→31/5%正式门仍待完成。

CE14 正式attempt4已终止并[封存原始结果](../../../../evidence/performance/inference-g04-g05-apple-encoded-m1-p1-attempt-4/receipt.json)：
16 A/A及5项controls全部PASS，原n15的120产品/30controls与strict verify0实际完成；三形状
诊断比率分别1.2263–1.2357、1.1420–1.1504、1.0756–1.0807。但前次SDK证据更新遗漏同步
contract内3个consumer-provenance.json SHA，product evidence因hash不同FAIL，整轮为
DIAGNOSTIC/UNPROVEN，不能改写成合格PASS。实际717.606秒与旧失败全部保留。
仅纠正这3个证据SHA为实际已提交文件236929b48aac413eac5de056c69d378918c0038486d6b1007216c89b55491b88；
产品3b5157532、SDK、peer、三形状、时序/成本、16 A/A、n15→31、5%阈值全部不变。
下一有限attempt5先核Git/worktree/远程的JSON→MD全部hash、完整合同及SDK identity，
再沿原仪器运行一次；不修改性能核心、手工提升旧manifest或重跑产品/普通测试。原fresh
合同只产生SDK比较程序，不触发native/product构建。启动前确认原parent终止与Air容量；
继续heavy串行、max4jobs、原成本不重置。工程资格仍待这次证据绑定正确的结果。

### CUDA A/A 失格后的有限测量诊断

原attempt3的4ms和attempt4的40ms均已实际16 A/A FAIL；后一轮selected prefill出现0.8863，
留出最高1.1250及factor bias。全部仍UNPROVEN，不把仪器不稳定当产品性能损失或PASS。
下一步只在Apple attempt5真实终态后，复用原CUDA attempt4 SDK、对象和三形状/16 paired
rounds，将独立诊断副本的minEpochTime分别设200ms、500ms。各只运行selected/native一对
控制，共4进程；固定原失稳placement/peer-first/peer-allocation-first层，后一个时长换route
顺序。数值oracle、两槽/4 replay、完整update→权限→原生完成→output release均不变。
只重编这1个benchmark CXX TU的两个时长副本并链接原对象；无CUDA TU、native/product
重建、无HYLE源码修改、无全库运行、无完整P1。实际原compdb/link和所有复用对象SHA先核。
记录真实机器/驱动、环境/affinity、完整raw、时长与同路线比率；这是失稳归因，不签16路A/A
资格或吞吐。若较长片段仍失稳，停止追加同配置完整采样，按实际同步/调度/地址机制进一步
定界；若显示可用分辨率，再按原5%门固定一次全16路A/A验证。不能放宽阈值或删除不利形状。
诊断预算1h检查点，沿用户常设授权追加、保留原失败和成本；heavy串行，max4jobs。

Apple CE14 attempt5已[QUALIFIED/PASS/strict verify0](../../../../evidence/performance/inference-g04-g05-apple-encoded-m1-p1-attempt-5/receipt.json)：
16 A/A、原n15的120产品/30controls完成，window1.2279–1.2381、prefill1.1443–1.1513、
留出1.0765–1.0833，三组方向WIN、原5%非劣PASS；713.801秒及原失败保留。仅Apple owner
依[有限域终审](inference-g04-g05-apple-ce14-terminal.md)合法CLOSED，其余owner/candidate
不动，不代签整个G04/G05或goal。停止重复Apple已签域。当前先按总卡补G02 Windows尾项，
其后才运行已声明CUDA测量诊断；G06不提前实施。


## CE15：G02 补验发现的 Vulkan 输出根别名（AUDITED）

baseline `54f081f52eecc64e309f46b04458e043aad42a9f`，原 provider candidate
`982676cb05a6e6244e66e0a596347aa96043617b`。G02 Windows 补验在
`VulkanGraphMoeFragments.D5DeepSeekFullAttentionChain` 触发原生分配断言。
[原失败与独立诊断](../../../../evidence/correctness/inference-g02-vulkan-port-alias-diagnostic/receipt.json)
已封存：scratch 仅增加端口 trace，正常源码/DLL/exe SHA 均未变；四输入、两输出的
同一 region 中，公开 value 5/6 的 port 4/5 指向同一 CONCAT 根，第二次导入分配必然失败。
这是已准入公开消费者的实际 correctness 缺陷，只有 compute_provider_ggml_vulkan 退回
AUDITED；其他 owner、G03 terminal 和全部原失败不变。

有限 FIX CE15 仅修原 Region::create 的公开输出导入：保留输入和每个输出的独立端口，
当某输出追溯的根已被输入或先前输出占用时，借原 ggml_dup/cont 为该输出建立实际拥有的
native producer。不能通过删除端口、共享两个 caller output binding 或绕过原生断言修复。
无冲突输出仍沿原根导入；纯 reshape 的形状恢复与内部零拷贝 view 保留。原 graph、
Imports、allocator、recording、失败回收与退休承载保持，禁止增加另一套 controller。
主纵切为上述 DeepSeek attention；正交为普通数值管线发布同一计算结果的多个 dense
reshape，另覆盖导出输入 reshape，避免模型专用判断。

冻结公开 API/identity/权限/并发/交付 target 不变；每个公开端口仍有独立 binding/owner。
P0：允许 prepare 时沿已有 ports 查重及每个冲突输出一个 native copy；对应运行费用按
实际输出字节计，不增加无冲突路径的 payload copy 或每次 bind 的查重/分配。原 finite
encoded P1 的无冲突单输出输入可按源码分支及 artifact 影响借用，不重跑已签数学性能。
如事实要求修改公开别名语义、shader 数学或常态状态结构，须回本节重新裁决。

批量实现后固定一个 candidate，再验证：provider 的根冲突反证（input/output、多个输出、
不同 dense shape、精确独立容量与重复绑定），真实 Direct/Graph 入口与原失败 case；
续跑 G02 原未运行 Vulkan 集合，既有 PASS 仅在能证明未触发该分支时复用。按 selector
选 provider 与真实直接消费者，source/static/shared 及实际 Air 的对应输出别名旅程；
原宿主 metadata/OOM/retirement 风险按输入匹配借用，新增元数据容量/清理须实测，
不以 host sanitizer 冒充 GPU driver 检查。无公共头/layout/依赖变化，不启动无关 headers、
全库、CUDA 或 P1。普通检查仍为 docs/closure/diff。实现/验证 2h 检查点，自动有限延长
并保全部失败/elapsed；停止条件为本 FIX 和受影响门可裁决，或确切外部条件无法推进。
本节只完成诊断和有限审计，尚无修复 PASS；G02 与 G04/G05 均未整体验收。


CE15 产品 candidate `1168874af05b8038b27a9e685e83a44dc4204456`：只改 provider Region::create，
冲突导出借 ggml_dup，metadata arena 为每个公开输出预留一个额外 tensor 上界；
新增一个 provider 回归覆盖 input 别名与三个 computed reshape 输出、精确容量/小一字节拒绝、
两次实际 replay。冷容量增加 O(outputs) 个 ggml tensor metadata，warm 只在冲突导出复制
对应输出字节。AUDITED 提交3d40003bb早于产品提交；当前仅冻结实现，测试尚未实跑。


CE15 首轮[实际定向结果](../../../../evidence/correctness/inference-g02-vulkan-port-alias-fix-stage/receipt.json)：
增量 build6.531s（只有 provider 与原所需测试闭包，无 CUDA TU）；根别名独立回归
0.391s PASS，涵盖输入 alias/三个 computed exports/精确容量/各两 replay。原失败
DeepSeek Graph case4.031s 已无重复分配崩溃，正常生成 XML，但输出3/11等的数值门
FAIL，主要在 V4-Flash Q8 ratio4/128。数值根因尚未裁决，不改容差或原 oracle，不签
G02/CE15 CLOSED；764个原默认 Vulkan case 尚未运行。三交付/Air及受影响消费者待验。
当前无重型进程在运行；先定位该同例数值差异，若需另改产品，先退回 AUDITED 更新
有限 FIX。不得因这一个回归 PASS 回填整个原用例或851集合为 PASS。


## CE16：G02 原消费者的大角度 RoPE 精度（AUDITED）

baseline `6303460820bc1a451bde5927e9fa6f3d7298ea17`，仅 compute_provider_ggml_vulkan
退回 AUDITED，保 CE15 产品1168874af及所有旧证据。
[同输入源码与实际诊断](../../../../evidence/correctness/inference-g02-vulkan-rope-phase-diagnostic/receipt.json)
排除 SinkAttention 操作数猜测；24个失败全在旋转尾部448–455。公开 F32 RoPE 在
position2052误差1.68e-4，131071误差8.65e-3；原 shader直接调用GLSL sin/cos。
scratch只替换两份原SPIR-V数组，用split-2pi/三次显式FMA归约后，原完整DeepSeek
case（原测试二进制/独立oracle/所有原容差）PASS，独立12组phase最大误差4.18e-7。

有限 FIX CE16 只改原 unary.comp 的 sin/cos：对 abs(F32 phase)<=1048576 使用三段
2pi常量及显式FMA，保先乘出的原F32 phase；域外/NaN/Inf沿原intrinsic，不签域外
新增精度。零符号明确保留。更新原manifest/notice与对应依赖内容hash，不升级依赖、
不改CPU/CUDA/Apple、公开API、layout、权限、recording、队列或生命周期。原schema
及F32 phase合同KEEP，原GPU dispatch/storage BORROW；无新原子或controller。
主纵切为原DeepSeek完整attention，正交为普通相位旋转的负位置/逆频率、近零和范围边界。

P0：sin/cos内增加一次round/乘法、三次FMA与有界分支，无新tensor/dispatch/allocation/
copy/wait。原G04/G05 encoded attention P1没有Sin/Cos节点，可按源码影响借用；
不复跑无关数学性能。实现后固定candidate，沿CE15三交付/风险和原未运行集合批量
验证；增加Direct独立F64方程的F32 RoPE回归，覆盖正负位置/频率、零和域边界，
实际Air对应Direct/Graph门。原精度门/状态反馈不改，不以native结果自证正确；
若归约不能覆盖本次完整实际输入，回本节裁决，不靠放宽阈值。检查点/资源/停止沿CE15。
当前仅完成诊断及有限审计，G02仍未验收。


CE16 产品 candidate `2b60d9a34be12fda9a9e3fdc26f7a61b495c3be7`，先前AUDITED提交01a8a7cf4。原pin/revision/
archive与所有其它依赖条目不变，只更新Vulkan patch内容及notice摘要。Direct回归冻结
18个正负/零/边界位置、三个head、32个独立频率（含负/零频率），对原computed-F32
phase的F64 sin/cos方程使用4e-6门。当前尚未在正常重配树实跑。

原selector WORKTREE计划已保存在scratch：依赖lock整文件变化保守选择全库；该输出
原样保留，不运行本卡明确排除的全库构建。此精确变更只含同一Vulkan in-tree pin的
内容hash/notice hash和已登记provider输入；组件图、公开头与其它依赖条目未变。
将原过宽结果作为独立验证工具限制保留，实际门仍按CE15/16的Vulkan owner、直接
消费者、三交付与实际Air执行；未运行全局gate不写PASS，不扩建selector/core。


CE16 正常重配335.906s PASS，整个派生native树2263文件仅unary.comp不同；
按实际compiler edge/依赖复用321个object，原149个vendor CUDA对象未重编，另一个
precise-unary CUDA TU因旧依赖不可取证而保守重绑定。151份生成shader头/主CXX输入
须在实际generation后再次核SHA，不能仅凭未改源假定生成物相同。首build在既有Direct
IQ1测试的double→float隐式range构造触发MSVC /WX；只改测试为显式static_cast，
保相同数值转换、原完整build FAIL与elapsed。产品candidate2b60不变，继续原树增量，
不重新配置、不放宽warning/数值门。原默认851用例将在修正后的新正常产物上整集合一次
串行复验，覆盖前缀已过但可能实际消费sin/cos的案例；不重跑CUDA804或其它模块。


后续增量实际暴露Direct catalogue测试缺ggml-quants.h的私有include路径；Graph同一
测试夹具已有正确SYSTEM PRIVATE HYLE_GGML_SOURCE_DIR/src。只为Direct测试target
补相同目录，产品target/导出不改。第二个scratch驱动曾错误要求resume仍有一个CUDA
编译（实际该TU已在首build完成），该setup断言失败单独保留，修为允许0或原单一TU；
全部既有失败/成本不覆盖。下一正常重配只纳入此测试注册纠正，沿原树复用既有native
对象与生成物；后续batch构建收集全部实际错误，不逐算子新建构建树。


CE15/16 [正常 Windows 定向验证](../../../../evidence/correctness/inference-g02-vulkan-alias-phase-windows-stage/receipt.json)
已实际 PASS：最后增量 build48.844s；provider独立输出根0.250s、Direct18位置phase0.297s、
Direct完整DeepSeek5.906s、Graph完整DeepSeek3.860s，各一实际XML、零失败/跳过。正常
重配与全部旧build/setup失败、321个native对象及151份生成输入的前后SHA、完整2263
native文件双树摘要均已封存。最终构建无CUDA TU，产品candidate仍为2b60d9a34。
这只签四项定向source结果；static/shared、Air和新增metadata容量/清理风险尚未签。

原851默认Graph集合在同一正常产物上实跑：前101 PASS，第102项
VulkanGraphMoeFragments.D5GlmFlashPooledIndexerWithTail正常完成但数值FAIL，失败保留。
其差异只见official-fp8的output5；源码显示fixture仍把F16乘数witness作为严格参考，
当前F32 MatMul显式保留F32乘数。仅作有限待证假设，不能据此改写FAIL或放宽门。
随后保持同一四产物SHA和全部原assert/oracle，逐例串行跑剩余749项并收集所有失败，
不再因首个数值失败中止整批。待批次终态后只对实际失败做定向诊断/修复与受影响复验；
不重跑CUDA804、不提前G06/P1。G02仍未整体验收。


CE16 原默认851项已全部实际执行，[原842 PASS／9 FAIL及环境诊断](../../../../evidence/correctness/inference-g02-vulkan-final-test-diagnostics/receipt.json)
原样封存，零skip。原生参考Backend明确请求validation=true，但启动环境未发现本机已缓存
Khronos layer；实际enumeration由11项缺层变为进程级VK_LAYER_PATH后的12项含层。
仅停止旧parent并排空原child，未计分的in-flight case重跑；旧PASS保留、环境失败及未运行
项在相同四产物SHA下续跑。没有关闭validation、重装系统、改产品或覆盖旧FAIL/elapsed。

最终两个测试缺陷一次修正：pooled official Flash score仍用旧coopmat2/F16乘数witness，
而当前公开F32 MatMul显式保留F32乘数；scratch保持原独立F32方程与严格门，四tail×四
反馈16组全部bad_f32=0，旧F16 witness共620个超差。两Vulkan caller仅对此原region和
F32 MatMul合同选回原F32参考。另一组八个IQ1 catalogue case因combined build的CPU扩展
枚举误入无对应encoder的ggml_quantize_chunk，编码阶段断言；Vulkan Bank改按三种公开
quant格式选择既有original raw bytes/grid与独立decoder，不由CPU枚举推断Vulkan量化器。

[正常增量与完整默认门合并裁决](../../../../evidence/correctness/inference-g02-vulkan-windows-default-acceptance/receipt.json)
PASS：仅四个测试CXX TU、两个测试exe，12.844s；九Graph原失败与九Direct对应case共18
实际PASS，原842不受两个有限predicate影响的PASS保留，总851/851。原容差、独立方程、
形状、产品candidate2b60及provider DLL均未变；321 native objects/151 generated inputs
在本次build后再次核SHA全同。普通source contract六项PASS。当前只剩本FIX的实际Air
source/static/shared与对应alias/phase/metadata资格；整项G02仍未验收，不重复CUDA804、
已通过842或P1，不扩大为全库验证。


CE15/16 [实际Air三交付](../../../../evidence/correctness/inference-g02-vulkan-alias-phase-air/receipt.json)
已PASS：source/static/shared各两项公开Direct/Graph phase与alias/旧结果/关闭，source
另过独立输出端口及metadata三项风险。原computed-F32相位的独立F64门4e-6不变。
首source构建后SHA检查识别生成unary.comp.cpp变化，未跑测试便补编该native对象；
旧构建/elapsed保留，后续static/shared预先排除该对象复用，其余依实际deps/flags/SHA
复用并做post-generation检查。当前产品2b60未再改。

原G02 common/native/精确守卫、Windows804/851与本补验域已[聚合验收](../graph/inference-g02-aggregate-acceptance.md)，
goal为3/10。此聚合不代签整个G04/G05或关闭其它当前候选；下一步按上文固定CUDA
200/500ms四控制诊断收其资格，不重复G02/Apple已签P1，不提前G06。


CUDA原声明200/500ms四控制已[实际结束并封存](../../../../evidence/performance/inference-g04-g05-cuda-duration-diagnostic/receipt.json)。
200ms两路线三形状ratio为0.9838–1.0356；500ms原生留出0.9319、selected留出1.0597，
同实现控制方向相反且超过原5%带。加长片段本身未消除不稳定；不挑200ms有利结果
启动同配置完整P1，也不把它判成产品性能损失/通过。只有两份benchmark CXX TU重编，
原SDK/其余对象SHA不变，原4ms/40ms失败及本次elapsed保留。当前无heavy进程；按原
固定分支先反查真实时序/同步/地址/调度机制，再声明有限诊断，不改公共性能核心或产品。
CUDA三宿主facet已有分别封存的ASan/UBSan/TSan结果；保原LSan和诊断线程边界，不重复。

### CUDA 完整 episode 的有限机制诊断

源码核查确认 `update_batch` 每次复制整 metadata、提交上传并等待；随后原生 run 和
Completion 再等待，两个 A/A 对象各有两个独立 slot。nanobench compare 使用共同迭代数、
块内 ABBA/BAAB 顺序，校准与计时均执行完整 episode；旧日志只有汇总，没有分段或原始轮次。
这些事实尚不能裁决是地址、slot、宿主调度还是 GPU 服务时间造成失稳。

固定下一诊断只编译一个 scratch benchmark CXX TU，链接原 attempt4 SDK/对象并核 SHA，
不修改产品或公共性能核心。保原三个形状、两 slot、四次独立 oracle replay、完整
update→run→wait→output/release。两无 profiler 进程分别运行 selected/native 同实现控制，
均在原 placement1/peer-first allocation 层：每形状先八个 ABBA/BAAB epoch（每边512个
episode），再2048个逐 episode 配对（同样平衡顺序）。记录每次 update、run submit、
wait、output/release、wall/thread CPU、slot、实际CPU与线程 context switches；缓存记录后
再输出，计时中不打印。另两进程仅对原 D64/128 留出作相同诊断，使用本机已安装的
Nsight Systems CUDA API/GPU trace；禁 CPU sampling/context-switch profiler，避免扩大采样。
profiler 与无 profiler 的结果分别保留，不作为产品吞吐或 A/A 资格。

总计四诊断进程，不新增完整 P1/16路 A/A、不删除不利原形状；Nsight 不可用时保留该失败，
仍分析无 profiler 分段。预算1h检查点，heavy串行、max4jobs，原失败与累计成本不重置。
以分段/slot/调度和真实 GPU trace 决定下一分支；若无明确机制，不据此改产品或挑选有利时长。

上述四进程[实际机制记录](../../../../evidence/performance/inference-g04-g05-cuda-episode-mechanism-diagnostic/receipt.json)
已封存，compile/link/运行共31.701s。原留出 selected blocked配对中位1.11368，逐episode
配对中位0.99912，但保留全部长尾后16×128 arithmetic组的边际ratio仍0.92292；不能用
有利的单次中位数代替原完整episode平均吞吐。两slot无持续分离，guest context switch很少；
Nsight留出GPU kernel约3.9us（original）或4.6+1.0us（selected），80B metadata copy约1.26us，
而event synchronize中位15.6–17.5us、最大1.85–1.94ms。此为宿主/API完成观察时间尺度的
直接证据，尚不细签Windows driver/hypervisor/桌面竞争的最终原因；profiler结果不作吞吐。

下一有限诊断仅替换scratch benchmark计时调度：继续借原nanobench逐epoch ABBA/BAAB，
每epoch恰好一个完整episode，固定16×2048 paired episodes。每2048个配对将两边**所有**
elapsed相加、除以2048，形成原16个算术平均组，再用原Result/CompareResult格式和原配对
统计输出。保留组sum/count与相同独立数学检查；不截尾、不剔除慢样本、不改estimand或5%门，
不修改nanobench/hyle_perf核心或产品。原4ms/40ms/200ms/500ms结果全保留。
只编译一个scratch CXX及link；同placement1/peer-allocation-first，两路线各candidate-first/
peer-first一次，共四控制进程。若四控制三形状均在原5%带内，再冻结该局部adapter及其原始
轮次/成本，声明完整16路A/A；本轮不能提前签资格或直接启动产品P1。若仍失稳，停止单纯
追加样本量。预算1h检查点，heavy串行、max4jobs，复用对象SHA再次核验，原成本不重置。

固定16×2048的[四控制预检](../../../../evidence/performance/inference-g04-g05-cuda-episode-interleave-preflight/receipt.json)
实际PASS，12个原三形状比率0.98065–1.01749，原5%门不变。首次scratch CXX缺iostream
编译FAIL9.628s保留，同树补include后compile/link及四进程完成，总86.265s；没有产品改动。
据此冻结本workload局部adapter：原nanobench单episode交错、每2048对全量求和、原16组
Result/CompareResult统计。canonical只改benchmark、原A/A telemetry/group校验和provenance
摘要，公共性能核心/产品/API/输入/oracle/门槛均不改；组记录在计时后输出并flush。

下一不可变benchmark候选按原 `cuda_encoded_p1/contract.json` 启动attempt5：原两placement、
两执行顺序、两allocation顺序和两same-implementation route完整16路A/A；只有真实PASS
才进入原n15八strata产品和controls，必要时仅原声明n31扩样。资格/产品结果与strict verify
如实封存，原attempt3/4失败不重置、不回写；四控制不冒充完整资格。既有static CUDA SDK、
原native数学、三宿主sanitizer均按身份借用，只构建该独立benchmark工程及所需pristine peer，
无HYLE产品/native依赖全局重建。heavy保持串行，无Windows/Air并发；2h检查点沿常设授权
追加，实际累计不清零。若A/A仍FAIL，按原stop分支停止同配置重跑；若资格通过而产品失败，
保留真实LOSS并以实际机制/消费者重新裁决，不能放宽5%门或改变输入来通过。

本次selector预检对四个benchmark/contract路径报告unknown mapping，未发现产品owner授权问题，
只选docs/closure；原计划与skipped reasons保存在scratch，不把BLOCKED改写PASS、不扩建selector。
这不改变产品组件状态；本次实际验证范围由上述具名独立benchmark卡与既有性能合同固定。
docs/closure/diff及Python语法已通过，下一次资格仍须绑定提交后的不可变候选和原SDK身份。

attempt5在原数值receipt的benchmark SHA检查失败，完整A/A未启动，首产品进程因ready门拒绝，
未取得任何timing样本；[原失败](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-5/receipt.json)
与72.614s原样保留。下一步直接对已构建b7b8d264b的primary/placement两artifact运行原
VERIFY_ONLY三形状/各四replay/独立double和readonly检查，签独立数值rebind补证；原数值
receipt及所有旧attempt不覆盖。借用原SDK/原header认证并重核SHA，无重新配置/编译。
两实际数值结果通过后，validator改为读取该补证，按同一冻结episode计时合同续attempt6。
原5%/16路/原n15/31和预算不变；这只是缺失的数值绑定，不把设置失败当A/A FAIL或产品LOSS。

[新数值绑定](../../../../evidence/correctness/inference-g04-g05-cuda-episode-interleave-numeric-rebind/receipt.json)
实际PASS：复用attempt5两个已建artifact，各24次独立oracle核查与三组pair/readonly，
共2.636s，零configure/compile。原23个benchmark输入只有主CXX变化，原SDK/artifact/
recipes/native archives与original headers全部核SHA未变。原数值receipt以完整SHA链接，
不覆盖旧记录；validator只切换到此新补证。合同的measurement说明同步到真实16×2048，
统计/输入/5%/n15/31不变。下一attempt6沿本节原资格序列，前次设置失败与费用继续累计。

attempt6 source `2f01f4d991d58973d1cb1f0882f2fc86bd4b9ebc` 已实际通过
[完整16路A/A](../../../../evidence/performance/inference-g04-g05-cuda-episode-aa-qualification/receipt.json)：
48个原三形状ratio为0.98730–1.02326，gross/factor median检查均PASS、零errors；两原执行/
分配顺序、两placement与两同实现route全部保留。此checkpoint不签完整P1或产品非劣。
同一parent继续原n15八strata产品/controls和必要原n31扩样，未修改候选或重跑A/A。
初始产品片段出现prefill耗时差异，当前只有进行中的样本，不据此签LOSS或改产品；待原
完整统计裁决。若确有material损失，再按原机制/真实consumer回AUDITED有限修正。

采样期间仅作[输出成本只读审计](../../../../evidence/architecture/inference-g04-g05-cuda-output-cost-audit/receipt.json)：
未启动其它GPU任务或编译，cuobjdump静态读取实际artifact耗时0.274s。FP8无论是否split
均写F32输出staging，再单独scale/cast；D256 prefill为1572864-byte staging，其write/read
额外3145728 bytes及一个dispatch。原FI variant确有private v_scale的归一化后输出变换；
它不能无条件用于16-bit split partial，否则原有限抵消会先溢出。SM89 causalD256/CTA64/
kvMMA4的static符号original240/selected246 registers、均零stack/local；这不证明运行时
选中了该variant，也不把静态SHARED:0当动态shared-memory为零。

若同一attempt6真实TERMINAL、strict verify0、QUALIFIED且有material非劣FAIL，下一有限
归因只在原primary artifact上运行一次Nsight Systems CUDA trace：原VERIFY_ONLY三形状、
双方两slot/四oracle replay及原warmup，收actual attention/merge/scale_cast、动态shared/
register/grid和关联launch API。禁CPU sampling/context-switch profiler；不改输入、阈值、
binary或SDK，不与P1并发。保全部raw/实际elapsed，区分冷decoder/数值download与各数学
kernel；profiling只作机制证据，不签吞吐。预算15min检查点沿原用户常设追加/原累计；
只有实际归因后才scope有限产品修正，当前owner仍CANDIDATE、不提前实现G06。

attempt6 [原完整记录](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-6/receipt.json)
已TERMINAL：120产品/30controls、16路workload A/A、五项controls全部PASS，strict verify0，
worker2535.381s；Windows外层2741.891s重叠另列，不相加。最终qualification为DIAGNOSTIC，
performance仍UNPROVEN：唯一未过validator是build-contract的ISA=unknown，CUDA独立合同
漏了其它三路线已有的 `HYLE_SIMD_EXACT_TARGET=""` cache声明。其余build字段、输入/
artifact/host稳定性、干净source/peer与consumer evidence均通过。原manifest/样本不得回写
或提升成QUALIFIED/产品LOSS；停止同一缺项配置的全量重采。

本次最小合同修正只加同样的空exact-target声明，表示实际portable宿主代码。先在新的
scratch树按原完整configure参数加该字段作configure-only预检，保存实际cache与compdb，
对照旧/新编译命令（仅build路径归一化）、显式ISA flags与GCC实际默认target。不编译产品/
peer、不改公共checker或旧树；后续新不可变候选须在采样前核实此字段及其它build条件。

上节单次profile的条件仅为只读P0归因作明确补充：也允许本次已封存的DIAGNOSTIC输入，
但必须再次证明只有build-contract ISA缺项、五controls/16A/A与其它validators PASS、所有
稳定性/consumer/source/peer条件有效，且原parent实际终态/消失。仍只运行同一个primary的
VERIFY_ONLY三形状CUDA trace，不重采产品性能、不把trace或diagnostic ratio当qualified LOSS。
这是检查已证实的额外staging/dispatch结构成本；任何产品修正另按真实consumer与源码机制
裁决并合法退回AUDITED，新P1必须包含正确ISA资格。预算/未知费用/原失败全部沿前节保留。

前述 configure-only 与单次 profile [实际补证](../../../../evidence/architecture/inference-g04-g05-cuda-native-resource-diagnostic/receipt.json)
已完成：37.640s configure、3.641s trace；八条归一化编译命令未变，portable声明与实际GCC
x86-64默认一致，旧cache未改。trace原三形状共24 oracle、各方8次attention，全部原数值门通过。
D256实际selected CTA64/MMA_KV4、Q32KiB+KV32KiB+FP8→BF16 repack32KiB，共96KiB；
original BF16 MMA_KV1为Q32KiB+KV16KiB，共48KiB。原FI已经直接写F32输出，不能把差异
误归因于另一个未使用的F32输出shared tile。FP8最小合法MMA_KV2仍需64KiB，当前100KiB
SM预算不能同时容纳两个CTA；不据此盲改tile/调度或承诺减少shared。
F32四stage merge实际16.5KiB，BF16为8.5KiB；selected另外有scale/cast dispatch。
以上仅P0归因，不把profiler时长、diagnostic ratios或补充ISA声明提升为合格吞吐/LOSS。


## CE19：有限最终输出适配（AUDITED）

只将 compute_provider_flashinfer 从 CANDIDATE 退回 AUDITED，baseline为原982676cb05a6，
当前证据锚3f17ffb40。其余owner、公开schema/API、对象/lifetime/OOM/cancel、编码与语义域
KEEP。真实consumer仍是本卡量化KV的单/多请求驻留decode/prefill及正交noncausal/不等Dv；
每次完成前对整个输出再读写和dispatch是源码已证实的额外线性成本，原生variant已具最终
F32缩放，原persistent merge也在F32归一化后最终cast。因此以消费与P0机制双证据准入，
不以未资格化ratio声称material LOSS，不改变attention tile或FP8 repack算法。

有限FIX只有两项，一批实现/检查：

- CE19-A：无split的FP8复用原variant私有v_scale，在F32归一化后直接写最终F16/BF16；
  去掉整个最终F32 staging与scale_cast，普通非FP8语义不变。
- CE19-B：split仍以F32保存所有partial，复用原persistent merge，在最终归一化后乘scale、
  最后才cast到Q输出类型；单partial分支也不得先cast。vendor只加私有merge回调seam和
  原merge可选scale epilogue，默认行为保留，不复制merge/softmax/attention算法。F32输入
  pipeline固定两stage，以维持D256原BF16四stage相同的字节预算；其它资源形状不调参。

删除不再使用的最终staging计费与独立scale_cast；原planner仍按F32 partial字节计算split
workspace。只允许native_cache/native_batch私有实现、声明、具名vendor patch及真实数学
回归测试；不改公共build/perf/checker、调度器或公共API，不新增fast模式。所有有限维度、
paged/ragged、F16/BF16、window/noncausal、空/单/多partial均须正确，尤其保留原溢出反例，
并将BF16/F16 partial低位抵消反例变成实际GPU回归，防止只保留文档算术证据。

验证：selector owner授权/有限FI source+既有13数学与新抵消、直接batch消费者，source和
static/shared SDK按影响复用未变依赖；三宿主sanitizer只验证实际受影响host部分，旧设备/driver
限制保留，不重复G02或其它路线已验收矩阵。无API变化不重跑全局测试。固定候选前只作原
三形状VERIFY_ONLY及一次短机制trace确认输出dispatch消失/partial正确；不签吞吐。候选固定
后按原CUDA合同（含已补portable声明）重新数值绑定及A/A/P1/strict verify，原输入、5%、
16路和n15/31不改。当前没有heavy任务；所有heavy串行、最多4jobs/CUDA pool1，2h检查点
沿用户常设授权有限追加，原失败和累计成本不重置。若实际数学/host报错则保留原失败后在
两FIX内修正；新架构/数学域需求必须重新定界，不能无限扩大CE19。G06仍不实施。

CE19首次configure在依赖审计处FAIL63.128s，数学/编译尚未启动；原日志保留。
修正同一private patch的lock tree/NOTICE身份，版本仅标识本地patch v2，不改upstream pin。
原FI CUDA标准为17，callback detection沿该依赖的void_t形式；不提高公共build/toolchain门槛。
selector因具名vendor patch/共同lock保守选全库，原READY授权只FI AUDITED/无issues保留；
实际只有FI build-local头发生数学变化。沿已声明有限FI/direct batch/source/static/shared门，
无影响路线保留旧身份证据，未运行全库项不代签PASS，不借此重写selector或触发全局构建。

CE19设备风险预声明：只读确认本机Compute Sanitizer 2026.2.1.0可用后，在本次source/math实际
通过、同一artifact SHA固定且无其它heavy任务时，各运行一次 memcheck 与 racecheck，过滤
FlashinferCache.* 真实数学cases（含paged/ragged/有限depth、单/多partial和三种split精度反例）。
memcheck不签driver/private-pool泄漏，racecheck最多4 CPU workers；保正常异步launch，不强制
blocking或改变stream/输入。固定 --error-exitcode 86，保存完整log/命令/elapsed及前后artifact
SHA；任何未运行/工具失效或真实hazard分别记UNKNOWN/FAIL，不签设备PASS。预算15min检查点
沿常设追加。此为新两stage共享内存流水及融合输出地址的实际风险验证，不替代三宿主facet，
也不触发其它路线/原G02矩阵或P1。原normal CUDA编译短暂并行两TU，已用私有进程串行化
保留19.026s暂停，不重启编译；后续配置只对FI target设compile pool1，普通CXX仍max4jobs。

CE19-B源码复核补充：patched cascade 的merge kernel新增scale参数，但其它TU仍使用pristine
头。因此同时给build-local默认merge host dispatcher私有名字，避免同一host模板名在两TU
分别绑定8/9参数kernel的不同定义；默认数学保持原样，不把该差异留给linker选择。此为
同一FIX的私有ABI隔离，不增加公共API、独立merge算法或其它owner改动。当前source2仍按
其已冻结旧snapshot编译/测试，结果只对该snapshot签发，后续实际修正版须独立rebind。
后续SDK/host风险私有profile固定本机SM89、FI compile pool1、NVCC --split-compile=3，
Ninja最多2jobs（优化线程至多3，加一个普通CXX），不启LTO或改优化等级；只省无当前机器
验收意义的SM80构建，不签其它GPU架构。新产物照常跑其真实math/SDK/风险及数值绑定，
不把编译线程数选项当bitwise或性能等价证明。公共产品CMake/工具/core不改。

CE19首轮draft48b384的[source/math实际记录](../../../../evidence/correctness/inference-g04-g05-cuda-ce19-draft-output-fusion/receipt.json)
已封存：configure68.348s、build983.199s、16 tests/零FAIL/零SKIP 3.011s。
新BF16残差64、F16残差16及原partial溢出0均通过，正常prefill/decode、mixed短K、window、
noncausal、不等Dv/主D256长prefill、OOM回滚/拒绝与编码边界亦通过。该draft尚无后续
82ddbb453私有host名字隔离，不能据此把最终CE19签PASS/CANDIDATE。原失败63.128s与
19.026s串行暂停保留；下一source3用前节SM89有限profile取最终实际绑定，复用未变依赖。

CE19 source3配置实际55.459s PASS、prepare-manifest0.052s PASS；随后dry-run只显示
VerifyGlobs强制phony→Re-running CMake，范围断言拒绝，进程exit1、产品编译/测试未启动。
同一已配置树与全部snapshot输入SHA复核后，用build.ninja逐字节相同的另名preview.ninja
展开真实对象图：47edges，仅FI四CUDA TU、无ggml CUDA。原normal cmake --build仍走原图，
不修改公共Ninja/CMake或重配；续跑记录独立source3-resume-1，原source3结果和失败原样链接。
四CUDA实际compdb均89-only/--split-compile=3，FI pool1、max2 Ninja；终态后还需核原manifest
SHA未变。该预览修正只恢复原有限验证，不签任何产品PASS，不重置原费用或新增能力。

CE19最终82ddbb453的[source17项实际证据](../../../../evidence/correctness/inference-g04-g05-cuda-ce19-final-output-source/receipt.json)
已封存：同树resume build321.543s、16 tests2.968s；原manifest SHA未变、四CUDA对象均
SM89/split-compile3且无ggml重建。追加单partial回归只编一个测试CXX3.795s、17 tests
2.916s全PASS，产品library SHA未变；同batch里64-row短请求与8192-row长请求共同强制
split，正确F16/BF16输出96.0625/96.5，先narrow再scale均误为96。原零/多partial精度及
所有原数学用例仍通过。只将FI owner进入CANDIDATE82dd，原其它owner不动；源码数学
稳定，不再扩产品形状。SDK、宿主/设备风险、原定数值/P1仍未代签PASS。


## CE20：16位输出SMEM交接同步（AUDITED）

真实[设备检查](../../../../evidence/correctness/inference-g04-g05-cuda-ce19-device-race/receipt.json)：
memcheck4.610s/零error PASS；racecheck10.144s/exit86，40errors、378warnings、100hazards
displayed，十项数学用例均通过仍不能签设备PASS。原raw完整保留，不压掉warning或放宽门。
只将FI从CANDIDATE82dd退回AUDITED，真实consumer/公开scope/owner/API及原baseline不变。
源码证据：SharedStorageQKVO把Q/K/V、跨warp归并与O staging放在union；原
threadblock_sync_mdo_states结束读取后没有屏障，16位write_o_reg_gmem会复写同一区域，
随后同warp还会从刚重排写入的SMEM加载128b。原F32直写路线不使用这段输出staging，
本轮融合首次把所选FP8最终输出接到了该原生16位writer。这里必须按真实hazard修正，
不能把数学PASS冒充同步安全，也不扩展attention数学/状态/公共编排层。

有限FIX仅两处BORROW原生同步原语：16位writer复用SMEM前统一CTA barrier，warp0完成
重排写后、读出前syncwarp；F32输出分支、partial/merge算法、tile/调度、编码和公共对象
KEEP。不新增kernel或条件fast模式。只改已登记private vendor patch/原子更新lock身份；
其它owner不变。一次集中实现后同SM89/pool1/split3/max2配置只重编实际受影响TU，先原
17项math、再同十项memcheck/racecheck。两设备门均通过才冻结下一候选，随后恢复已声明
SDK/宿主facets和原合同P1；未变域身份匹配则借用，不重跑G02或其它路线已验收矩阵。
预算2h检查点沿原常设授权追加、原失败/累计成本不清零；若仍有hazard保留实际报告并在
这两个边界内定位，不为PASS放宽print/错误语义或压制检查。新范围需求须另定界。

CE20同族反证补充：原native_attention/native_batch也包含同pin的pristine prefill writer。
当前owner仍AUDITED，设备检查在原十项cache数学之外，另对现有Flashinfer的F16/BF16公开
prefill/decode两case各走一次racecheck（无新增输入/产品算法）。它们复用同一同步边界，
用于确认私有补丁的接线范围是否完整；若有同族报告，只允许同owner私有头接线覆盖，
不复制第二份修正、不扩全局图或原G02矩阵。原补丁两barrier FIX不变。

CE20 cache-only实际检查：source4 configure54.983s、build289.730s（仅native_cache一个CUDA
TU）、17 tests3.155s PASS；memcheck4.343s/零error、原十项racecheck8.639s/零hazard/
零error/零warning PASS。两条普通prefill/decode公开consumer的补充racecheck6.406s仍exit86，
0errors/512warnings；其native_attention/native_batch仍从pristine头读取同一个未修正writer。
这证明补丁接线未覆盖完整同owner能力，不把cache PASS代签整个Provider。

仍在本轮AUDITED内补齐同一FIX的唯一私有承载：整个FI target统一读build-local verified
header set，移除cache-only include例外；同步补丁只有前述两barrier，不复制第三份实现。
统一后不再存在pristine/patched merge host ABI混用，恢复原generic host dispatcher名字，
保留optional scale epilogue与private Params回调，decode头也自然复用原名称。无公共API/
组件图/其它owner/公共build core改动。适用门仍原17math、十cache mem/race及两普通case
race，再恢复SDK/宿主/P1；两个warning用例原记录保留，不压制检查或新增输入形状。

统一私有头后的274cd9fec已取得[最终source/device证据](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-output-sync-final/receipt.json)：
17数学全PASS；十cache memcheck4.217s零error、racecheck8.637s零hazard/error/warning；
两普通F16/BF16 case racecheck4.129s亦全零。原cache-only 512warnings及CE19 40errors/
378warnings不覆盖。只有FI owner回CANDIDATE274cd9fec，产品形状/实现冻结；下一步只做
已声明static/shared公开SDK、受影响宿主facets与原合同数值/P1，未签全批CLOSED。

CE20[动态SDK实际补证](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-shared-sdk/receipt.json)
已PASS：从搬迁prefix独立编译16个consumer/header TU，无仓库头；CPU/native数学与两种公开
组合消费全部实际通过。首次配置45.013s/范围预检发现146个ggml CUDA因派生目录c41→2f41
被判重编，guard exit1阻止产品build，原记录保留。242 native源码/头与5 canonical输入逐SHA
相同，另名私有manifest只归一这些CUDA边的source/include路径和precise-unary对象引用；
Ninja自身对146原对象验证原command identity和deps freshness，实际NO WORK0.323s。
其余当前CPU/FI/product边不改；续跑原配置、不再configure，host producer80.763s、install
0.542s，全部146旧对象前后SHA及原normal manifest SHA未变。public SDK检查不把此复用
冒充当前未运行静态SDK、三宿主facet或P1；也不修改公共构建/selector核心。

CE20静态SDK首次configure53.934s PASS，原producer plan guard在产品build前拒绝146个
无关ggml CUDA对象（旧路径c41→当前2f41）。原失败/54.158s成功设置阶段保留。静态树
单独核242 native源码/头、5 canonical输入与146实际对象；Ninja对原命令与deps判NO WORK，
耗时0.266s。只在静态树自己的另名manifest映射等价CUDA输入/precise-unary引用，保留
当前全部CPU/FI边；同source archive/同configure续跑，无公共工具或产品改动，FI四CUDA
TU仍按新候选真实编译。该静态复用不是动态树复用的代签，终态再核146对象与normal图SHA。

下一P1数值/资格前先补一次原peer安全反证：同pin pristine writer也缺少已实证的同步。
静态SDK结束、无其它heavy任务后，只在冻结attempt6 primary artifact上运行一次原三形状
VERIFY_ONLY racecheck，保原双方/四oracle/warmup，不重采性能或构建。该artifact内original
namespace与HYLE namespace分开，逐类归因；其旧HYLE分支不签新候选。print-limit0、4workers、
error-exit86、保存完整log/elapsed/前后artifact SHA。若原peer有同族hazard，先修复对比语义
并明确记录本地derivation，再重新绑定数值；不能沿不安全baseline签LOSS/WIN，也不因读到
相同源码就预判工具结果。无hazard则保实际清洁证据、继续原pristine peer。该反证不改变
原5%、8strata/16AA/n15/31、输入/所有旧P1 verdict，不改公共性能核心。预算15min检查点
沿常设授权有限追加，原失败与费用全部保留。

CE20[静态SDK实际补证](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-static-sdk/receipt.json)
已PASS：CPU数学14、native数学13、CPU公开组合3、native公开组合3，以及12独立头/16个
consumer编译TU，全从搬迁prefix消费，无仓库头。静态FI四CUDA TU实际编译，producer
359.790s、install0.776s、consumer configure31.105s/build19.106s，续跑成功阶段416.825s；
首次设置/范围guard54.158s另外保留。静态146 native对象复用有自己的Ninja无工作证明，
对象前后SHA与normal图SHA均未变；不以动态模式证据代签。source/static/shared产品旅程
至此均有当前候选实际结果；三个受影响宿主facet及原P1/聚合仍待验。

[原peer安全反证](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-peer-safety-preflight/receipt.json)
实际22.578s，原24独立oracle最大误差0.001091<0.008均PASS，但racecheck exit86：192条
已显示original namespace读写warning，另有一条达到hazard上限提示，不推断未显示域清洁。
据此前节明确分支，下一独立比较改成sync-fixed native69ff：认证原Git pin/全部原头，在
build-local副本仅选用与产品同样的两处输出同步，分别签effective header与selected-edits
SHA；namespace隔离保留，原math/tile/layout/planner/BF16 partial不改，无FP8 padding/F32
merge产品特例复用。统计行/claim/provenance显式标sync-fixed，合同id升v2，新数值补证
指向独立CE20 receipt；旧receipt/manifest/样本全保留，不回写pristine旧结果。当前仅
source-only制备验证通过；新的compiled数学/race和完整资格未运行，不代签PASS。

本次只是workload-local peer安全修正、当前实际静态SDK prefix及数值绑定更新。API/产品
candidate274不变，原输入/3形状、8strata/16AA/n15/31、5%门与全量尾样本统计不变；
公共性能核心/工具不改。新数值receipt未实际形成前ready门必须拒绝采样。

本次peer/合同变更的selector计划ce20-peer-sync-plan仍因同一共享bench目录归属缺口为
BLOCKED（六个unknown路径，product_authorization无owner/issues），原146 skipped reasons保留。
沿已登记独立工具任务与本卡有限验证续跑，不借此修改selector或全库构建，未运行项不代签。

CE20[ASan宿主地址facet](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-host-asan/receipt.json)
实际PASS：16项native/common数学消费与五种原失败/未知/取消风险，累计523.003s；四个
受影响FI CUDA TU的host instrumentation均从实际compdb核验，未以GPU设备检查代签。
只借原同mode、同compiler/device/driver的不可变146-member ggml CUDA archive，当前242
native输入和5 canonical输入再核SHA，历史散对象/系统header freshness没有重新证明；两次
旧对象路径核对失败保留，未改写为PASS。当前runtime测试确实在新OS运行，不继承原F5
runtime verdict。原NVIDIA LSan INCONCLUSIVE保留，本次detect_leaks=0只签地址facet；
shader/driver/private-pool上界不由宿主ASan签发。UBSan、TSan与新数值/P1仍待验。

CE20[UBSan宿主未定义行为facet](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-host-ubsan/receipt.json)
实际PASS：同一最终产品候选的16项数学/消费测试、五种source风险均完成，累计468.508s；
四个FI CUDA TU及直接受影响host链的instrumentation由实际compdb核验。只复用已核SHA的
原同mode native archive，不代签原系统header freshness或设备/driver/private-pool上界。
ASan/UBSan当前均通过，下一门TSan；sync-fixed参考的新数值与P1仍待实际执行。

CE20[TSan宿主数据竞争facet](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-host-tsan/receipt.json)
实际PASS：16项数学/消费测试和五种source风险，累计460.325s；同候选四FI CUDA TU与
受影响host链实际instrumentation。原NVIDIA诊断线程lifetime限制与LSan INCONCLUSIVE
保留，不压制data-race检查，不把宿主结果当设备/driver/private-pool上界。当前source17、
设备同步、static/shared SDK及ASan/UBSan/TSan三个受影响facet均已有独立实际PASS。
下一步仅独立sync-fixed比较程序：两variant×三形状的原四次double oracle、readonly与
primary双方memcheck/racecheck，再沿原统计门执行新A/A/P1；未通过不签整项完成。

CE20[sync-fixed参考与当前静态SDK的实际数值/设备补证](../../../../evidence/correctness/inference-g04-g05-cuda-ce20-sync-fixed-numeric-rebind/receipt.json)
已PASS：新独立比较程序仅编译一个原生CUDA TU及直接CXX，两variant×三形状×双方四次
独立double oracle共48次，最大误差0.00109081301<0.008；每次原fixture均检查实际物理
readonly页和两slots。primary双方memcheck1.813s零error、racecheck20.889s零hazard/
error/warning。固定SDK的37个artifact、recipe、native archive与全部产品/比较输入SHA
前后核验；实际compiler依赖闭包只读sync-fixed verified headers。原native pin与唯一
两barrier derivation/effective headers均记录，不继承pristine旧参考的安全或性能结论。
阶段命令累计105.858s（另有WSL只读canonical Git检查约两分钟，未计入命令阶段和性能）；
实际公共build-contract预检PASS，portable ISA已从CMakeCache中的显式空值确认。新A/A
与P1尚未签发；下一独立attempt7沿v2合同，原5%/8strata/16AA/n15,31与所有旧结果不变。

CE20[attempt7原始结果](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-7/receipt.json)
已封存：worker2719.253s、重叠外层2721.891s，16AA/120产品/30controls及strict verify0。
artifact/build-contract/16路AA三个前后validator、五controls与source/artifact/host稳定性均
PASS，但三workload consumer evidence均FAIL：同步修正后的provenance JSON已更新，合同
仍绑定旧61995c哈希，实际为f8d7d979。唯一资格失败是该绑定，最终DIAGNOSTIC/UNPROVEN；
不提升旧manifest，也不据其prefill/窗口样本退回产品实现。原所有记录和费用不覆盖。

本次只修正三处provenance SHA，规范化合同逐字段相等（除这三SHA）；产品candidate274、
源码/SDK/全部数值、输入/统计/门槛/采样计划不变，原数值receipt继续按实际SHA复用。
canonical侧已调用原load_contract和_evaluate_product_evidence：三项全PASS，嵌套prose SHA
亦PASS。private launcher同一检查前置到configure/A-A/采样之前，任一失败立即停止；公共
性能core不改。下一独立attempt8保留原5%/8strata/16AA/n15,31，不重跑产品构建/数学/SDK/
宿主风险；新资格只由新实际结果签发。沿原常设预算追加，原失败成本不重置。

## CE21：合格成本反证与有限只读归因（CANDIDATE，未授权产品改动）

[attempt8实际结果](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-8/receipt.json)
为QUALIFIED、performance FAIL、strict verify0；worker2664.435s，重叠外层2667.750s。
16AA、三个前后validator、三consumer资格、五controls及全部身份稳定性均PASS。
D256 window比率0.953993–0.958642、方向LOSS/5%门UNRESOLVED；prefill0.861386–0.872412、
方向LOSS/非劣FAIL；留出0.980057–0.994756、方向INCONCLUSIVE/5%门PASS。原n15已出现
明确material FAIL，原停止规则不启动n31，不靠同配置重采寻找通过。原7 DIAGNOSTIC与
全部失败/费用保留；G04/G05不能关闭，G06仍不实施。

主消费者仍为Qwen真实GQA8/KV2 D256长prefill，多请求Q64/128与窗口消费；不引入新形状。
在没有其它heavy任务且原8真正terminal后，仅一次对冻结8 primary artifact的原三形状
VERIFY_ONLY Nsight CUDA trace，核24独立oracle、actual kernel/merge与host launch账、
register/shared资源及原artifact SHA前后不变；不重采P1、不修改输入/门槛/peer或公共工具。
只读阶段预算15min检查点沿常设授权追加，原cost不重置。先证明实际费用结构，再固定
有限FIX并合法回AUDITED；此前不改CANDIDATE产品或以旧982的profile代签当前274。

CE21[当前274实际归因](../../../../evidence/architecture/inference-g04-g05-cuda-ce21-attribution/receipt.json)
3.121s/24oracle PASS，artifact前后SHA未变。D256 FP8实际MMA_KV4/98304 shared/246–248
register；参考BF16 MMA_KV1/49152 shared/240register。prefill当前attention约53.2us、
参考39.3us，merge约12.0/9.0us；无最终scale kernel。以上短trace只归因，不签新吞吐。

源码核定：原compute_qk/compute_sfm_v已经有完整FP8寄存器fragment swizzle/vec_cast与
原F32 MMA路径；repack只是用额外BF16整tile暂存替代该转换。现有kMinValidMmaKV仍
保证packed布局对齐，原planner/split/workspace与F32 partial不必改变。真实Qwen GQA8/KV2
D256 prefill material FAIL与该额外暂存/转换结构两把钥匙均成立。

有限唯一FIX BORROW原寄存器转换：本provider私有已准入FP8 cache不再启用BF16 repack，
存储/KernelTraits与三dispatch预算六处判定保持一致；原occupancy/MMA_KV选择自然按真实
存储计算，不手工试tile、不改planner或benchmark形状。所有F16/BF16普通数学、F32
partial/merge最终scale+cast、两个输出同步屏障和短K补零KEEP。无公共API/对象layout/
executor、全历史解码、新fast选项或自写attention数学。只改具名private manifest/NOTICE
与该依赖lock身份，upstream pin不变；其他owner及原所有证据不动。FI以原274为baseline
合法回AUDITED后才实施，先集中一次实现，再批量验证，不先签性能改善。

适用门固定原17math、十cache mem/race与两普通case race、直接3公共consumer、static/
shared SDK以及受影响三host facet；复用未变CPU/G02/Apple/Vulkan与native ggml依赖，
不跑全库。SM89/FI pool1/split3/max2Ninja、heavy串行；每批2h检查点按常设追加保原cost。
新source数学/设备门通过才固定新candidate，随后重新数值绑定及原合同A/A/P1/strict。
若数学/设备出现真实失败仅在该六处一致性边界内修复；若与原寄存器路径本身矛盾，则
保留失败并还原该试改，不靠删反例/放宽精度或任意新算法扩张。

CE21六处一致性修改已集中实施，FI仍AUDITED，未签数学/成本改善。selector ce21-register-plan
为READY，product_authorization仅FI且无issues；共同lock使选择器保守选全库，实际只有
FI私有headers派生改变。沿已固定FI/直接consumer/source/SDK/风险门，身份未变的其它
route/native依赖复用并保留skip理由，不为全库假签PASS或修改公共selector/build核心。

CE21同一存储FIX的静态补充：原SharedStorageQKVO还在union中预留CTA_TILE_Q*Dv个
DTypeO；F32 write_o_reg_gmem明确只写global float2，不读写此output tile，16位writer
才使用shared重排。只禁repack时D256 split的该未用F32数组仍把union撑到64KiB。
因此有限FIX加一处条件容量：F32 unused output member仅保1元素，16位容量原样；Q/K/V
与跨warp状态仍各自保完整成员，不能删整个union或共享live存储。此为同owner同资源
结构问题，不改F32 partial或原merge数学/ABI。总七处存储/traits/预算判定固定。

当前source1仍运行六处版本的不可变d1 archive，结果只对该draft签发；不改其运行中源码
或重置耗时。最终七处版本必须在source1真正terminal后独立snapshot重新验证原17math
与设备/SDK/host/P1门。canonical lock另只恢复未变字段的原序列化，和source1 JSON
逐结构相等，有单独proof；该format-only变化不触发全库构建或代签runtime。

CE21[最终七处source/device实测](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-register-source-device/receipt.json)
已PASS：最终source2为9c9ab88bb不可变快照，17项数学零FAIL/SKIP，阶段累计434.876s；
十cache memcheck、十cache racecheck及两普通F16/BF16 racecheck共17.877s，全部零
error/warning/hazard，实际二进制前后SHA一致。六处draft source1另为17项PASS、462.057s，
只保留历史证据，不代签最终七处。以上阶段费用分别列账，不清零原qualified FAIL及累计。

仅FI从AUDITED进入CANDIDATE9c9ab88bb，其他owner和已签路线不动；尚未签性能改善或
整个G04/G05。后续串行执行原static/shared relocated SDK、受影响三host facets及新numeric/
原合同P1；复用经实际源码/command/dependency/object身份核验的146个未变ggml CUDA对象，
当前FI仍实际重编，不触发全库构建，也不将私有构建manifest作为产品改动。

CE21[搬迁静态SDK](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-static-sdk/receipt.json)
已实际PASS，累计514.153s：14 CPU math、13 native math、3 CPU与3 CUDA公共consumer，
零FAIL/SKIP；12个独立公共头与16个consumer编译TU均只使用搬迁后的installed headers。
shipping无HYLE_TESTING/sanitizer；146个未变native CUDA对象经源码、原command/dependency
freshness和前后object SHA核验复用，当前FI四CUDA TU实际重编，原manifest前后不变。
此项只签静态交付；动态SDK、三host facets、新numeric/P1仍待实际裁决。

CE21[搬迁动态SDK](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-shared-sdk/receipt.json)
已实际PASS：14/13/3/3测试、12公共头、16 consumer TU，零FAIL/SKIP，成功续跑105.582s。
原shared1 configure47.970s后，复用guard发现一个precise unary对象缺少Ninja deps；未启动
产品build即失败，原报告保留。只刷新该精确原对象6.265s，另外145个不重编；shared2核对
相同source inputs与compiler/config后复用原configure，146个对象freshness及SHA核验通过。
这些费用分开保留；未把缺失deps写PASS，也未重建全套native依赖。

原P1仅重绑实际CE21 static prefix、prose身份、全部三consumer SHA及新numeric receipt路径；
规范化合同其余字段逐项相等，原JSON换行保留。原三形状、peer两输出屏障派生、8strata/
16AA/n15,31/5%/controls与所有旧FAIL均不变。宿主三facet和新numeric尚未签发，不能据
SDK成功预写A/A或吞吐PASS。

本次ce21-sdk-binding-plan仍因原shared benchmark路径族的三个JSON/Python输入unknown而
BLOCKED；product_authorization无owner/issue，docs/closure及146 skipped reasons均保留。
沿[既有独立工具记录](../build/validation-selector-shared-bench-paths.md)分账，按本卡原有限
门继续，不改selector、公共性能核心或产品，也不将该工具归属状态写成验证PASS。

CE21[ASan地址facet](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-host-asan/receipt.json)
已实际PASS：16测试零FAIL/SKIP及5个原source风险模式，最终七处vendor输入逐字绑定9c9，
四FI CUDA TU及原直接host消费者均有实际instrumented command，运行前后二进制SHA一致。
阶段累计524.610s，外层launch532.000s与之重叠，不重复相加；原NVIDIA process LSan仍
INCONCLUSIVE，不由本地址facet冒充完整泄漏证明。三facet批次已串行进入UBSan，后续
TSan、新numeric/P1尚未签发；产品candidate与其它owner不变。

CE21[UBSan facet](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-host-ubsan/receipt.json)
已实际PASS：同一9c9最终七处输入，16测试零FAIL/SKIP、5个原source风险模式均通过，
实际instrumented输入与运行前后二进制SHA核验通过。阶段累计457.414s，外层launch461.016s
与之重叠，不重复相加。批次已串行进入TSan，未提前签TSan、新numeric/P1或整个G04/G05。

CE21[TSan facet](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-host-tsan/receipt.json)
已实际PASS：16测试零FAIL/SKIP、5个原source风险模式均通过，最终七处vendor9c9输入、
四FI CUDA TU的实际instrumentation及运行前后二进制身份核验通过。阶段累计454.900s，
外层launch458.640s与之重叠；原NVIDIA diagnostic-thread lifetime仍INCONCLUSIVE，
不签线程泄漏、GPU shader/driver或private pool上界。

三facet串行[原批次账](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-host-tsan/batch-result.json)
已TERMINAL_PASS，逐项run/seal与原费用全部保留；source/device、static/shared和三host
facet至此均有当前9c9实际证据。接下来仅按原合同做新numeric及A/A/P1/strict验证，未签
数值绑定、性能改善或整个G04/G05。

CE21[新数值及设备绑定](../../../../evidence/correctness/inference-g04-g05-cuda-ce21-register-numeric-rebind/receipt.json)
已实际PASS，记录阶段累计75.783s：primary/placement各24个独立double oracle，最大绝对
误差0.00109081301，小于原8e-3；三形状、双slot重放/readonly与实际SDK artifacts/recipes/
native archives前后SHA均通过。primary另跑memcheck/racecheck，零error/warning/hazard；
build-contract预检PASS。只构建比较consumer与一个original CUDA TU，未重编产品库。

原peer仍只取两处输出同步屏障：selected edits SHA47df06c7、effective prefill SHA06189719
与原CE20相同；新的product manifest身份4c7b35bc单独绑定，不把CE21七处资源策略应用到peer。
下一独立attempt9按原合同运行16AA、8strata、n15/31、5%和strict verify，三consumer/prose
资格检查仍前置到configure/A-A/采样之前；原attempt8的qualified FAIL及所有旧费用保留。

## CE22：当前9c9的合格成本反证与有限只读归因

[attempt9实际结果](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-9/receipt.json)
为QUALIFIED、performance FAIL、strict verify0；16AA、120 product与30 control进程完成，
原n15已出现material FAIL，按原规则不追加n31。worker2635.632s与外层2637.703s重叠，
不重复相加。windowD256比率0.981006–0.985242、方向LOSS/5%非劣PASS；prefillD256
0.883122–0.891172、方向LOSS/非劣FAIL；留出D64/Dv128为0.914731–0.926564、
方向LOSS/非劣FAIL。原8 FAIL、全部数学/交付/风险证据和累计成本均保留。

FI保持CANDIDATE9c9，不以新失败自动授权产品改动。原三形状与真实Qwen prefill、留出
消费者不变；只在原9实际terminal后，对同一immutable primary artifact做一次VERIFY_ONLY
Nsight CUDA trace：核24独立oracle、actual attention/merge/host launch及register/shared/
grid，SHA前后不变。不新采P1、不试tile/算法变体、不改peer/输入/5%或公共工具；15min
检查点沿常设授权追加、原预算不清零。先定位实际机制，再按原两把钥匙决定有限FIX并
合法回AUDITED；没有机制证据不能无限重试。G04/G05未关闭，G06仍不实施。

CE22[当前artifact归因及生成代码](../../../../evidence/architecture/inference-g04-g05-cuda-ce22-attribution/receipt.json)
已完成：profile3.415s，24oracle PASS/max0.00109081301，binary前后SHA相同；后续实际
codegen提取1.132s不含先前cuobjdump资源表0.097s。短trace只作机制证据，不签新吞吐。
D256产品MMA_KV2/register255/shared49152，参考MMA_KV1/register240/shared49152；
prefill attention约51.177/39.682us、merge11.987/9.051us。D64/Dv128产品MMA_KV8/
register255/shared40960，参考MMA_KV4/register177/shared32768，attention约8.691/4.345us。

实际mangled kernel逐一匹配cuobjdump/SASS：产品window/prefill/holdout栈帧40/48/120字节，
分别8/10/27条LDL及同数STL；实际参考均零栈帧、零LDL/STL。shipping compile command
绑定cf631私有头，after SHA与当前manifest一致；未用旧generated cache推断当前资源。
源码max_num_mma_kv_reg仍为原通用上界，smem按压缩payload预算，FP8 minimum valid
MMA_KV又有packed对齐约束；下一步只审计这两者与宽化fragment的实际寄存器开销。
当前不授权简单对半tile、删除对齐检查或变体扫参；F32 partial/最终scale+cast、两输出
屏障、shortK补零和已准入数学/consumer全KEEP，后续有限FIX必须先单独明确并合法AUDITED。

CE22[有限页偏移源码审计](../../../../evidence/architecture/inference-g04-g05-cuda-ce22-attribution/paged-offset-source-audit.json)
确认原69ff的page_produce_kv_on_the_fly在当前私有头中逐字未变。非shared paged内核保留
两组size_t页偏移数组，D256共32字节/线程、D64/Dv128共128字节，提前生成的下一tile
K/V地址贯穿QK/softmax/PV，与当前actual栈帧/LDL/STL反证对应。真实Qwen prefill及留出
qualified FAIL和该持久地址结构两把钥匙成立；不以峰值寄存器数本身授权任意改tile。

有限唯一FIX BORROW原按需页偏移加载器，只用于已准入FP8 paged路：复用原protective
K/V offset和原async load，移除两组持久地址数组及提前生成的下一tile地址；保存current/
next packed-page base，原K-next/V-next prefetch时序、commit/wait/barrier不动。把已签
HYLE_FP8_SHORT_K的物理128字节/上半补零移到该原helper，global64行和QK64保持原样。
tile/warp/MMA、packed最小对齐、原dispatcher、数学/scale/F32 partial与merge、权限/退休
全KEEP；普通16-bit、FP4、ragged和原shared-KV路径不变。不新增公开API、控制器或整历史
展开。只允许private manifest/NOTICE与FI lock身份，baseline9c9；FI合法回AUDITED后实施。

显式成本取舍是加载处重算protected页偏移，K/V可能重复page divmod，换取移除跨数学阶段
存活的地址数组；源码预算不等于已证明寄存器或吞吐改善。先一次集中实现，再原17math、
十cache mem/race及两普通case race；随后冻结candidate，补实际codegen、直接consumer、
static/shared、受影响三host facets和新numeric/原16AA/P1/strict。其余路线证据按身份复用，
SM89/FI pool1/split3/max2Ninja、heavy串行、每批2h检查点沿常设追加，原失败/费用不清零。

CE22十处顺序manifest edit已集中实施，完整派生diff核对了initial K/V、K-next/V-next和
current/next page base的时序；未改变commit/wait/barrier及数学顺序。FI仍AUDITED，
尚未签数学、设备或寄存器/吞吐改善。ce22-paged-offset-plan为READY，授权仅FI且无issues；
共同lock导致选择器保守选全库，保留selected/skipped理由，按本卡真实FI/直接consumer门
执行，未变native依赖和其它已签路线按身份复用，不触发全库构建或改公共selector。

CE22[17数学与设备检查](../../../../evidence/correctness/inference-g04-g05-cuda-ce22-paged-offset-source-device/receipt.json)
已实际PASS，product8d61d51f1：source阶段458.838s，十cache memcheck、十cache racecheck及
两普通F16/BF16 racecheck共17.998s，零FAIL/SKIP、error/warning/hazard，二进制前后SHA一致。
FI进入CANDIDATE8d61；此处只签数学与设备安全，不签资源、吞吐或整个G04/G05。

随后按预声明执行[相同三个内核实例的实际生成代码对比](../../../../evidence/architecture/inference-g04-g05-cuda-ce22-paged-offset-codegen/receipt.json)：
exact mangled symbols、SM89、原tile/warp/MMA相同，当前私有头SHA与manifest一致。
window/prefill/holdout栈帧从40/48/120增加到48/56/144字节，峰值寄存器仍255；
LDL/STL从8/8、10/10、27/27变为9/9、12/11、30/31。提取成功不等于资源目标通过：
移除持久地址数组没有实现预期寄存器减负；不能据此宣称吞吐改善，也不能仅由栈增加
断言实际吞吐回退。该P0反证出现后不启动长SDK/三host/numeric/P1批次，未运行门仍未裁决。
保留8d数学/device PASS、代码与全部费用，所有原qualified FAIL不变。

下一有限只读审计仅核当前内核的原FP8转换fragment、live寄存器与packed物理tile约束；
必须解释相同实际实例的成本来源，优先借成熟原加载/矩阵乘路径。无新产品FIX授权，
不试tile变体、不删除对齐保护、不改peer/统计/阈值；若形成有限修正，先合法回AUDITED。
G04/G05仍未关闭，G06不实施；完整goal仍3/10。

## CE23：原概率片段按消费顺序缩短存活范围

[当前源码审计](../../../../evidence/architecture/inference-g04-g05-cuda-ce23-score-stream/source-audit.json)
绑定8d61实际头与compute_sfm_v函数SHA。原函数先把全部NUM_MMA_KV概率转换成half数组，
再按mma_kv升序逐块做PV；D256当前数组32字节/线程，留出128字节。每次PV只消费其中
一块，每query仅需16字节；PV不读取d，原每query的rowsum KV升序可以原样保留。
原两真实消费者qualified FAIL及实际栈溢出与该源码存活结构构成两把钥匙。

本轮唯一有限FIX：仅原FP8、float QK、非repack路径将half概率数组缩为一个KV块，
把原vec_cast8和m16k16_rowsum移到已有mma_kv消费循环，消费slot0。原转换、rowsum与MMA
函数直接复用；每output的KV/MMA/Q加法顺序及每query的KV求和顺序不变。16bit、FP4、
repack与非float QK仍用原整数组流程。只改private manifest/NOTICE及FI lock身份，
baseline8d61；先合法回AUDITED再实施，不引入公开API或额外产品文件。

tile/warp/MMA、planner、packed对齐、QK/softmax、prefetch/wait/barrier、scale/F32 partial/
finalcast和权限/退休全KEEP。先原17math、十mem/十race及两普通case race，再对相同三
mangled内核核实际codegen；源数组缩小不等于寄存器或吞吐改善。资源机制未改善则在长
SDK/三host/numeric/P1前停止该修正批；若成立才冻结candidate、继续原声明门，未运行项
不记PASS。SM89/FI pool1/split3/Ninja2、heavy串行及常设追加授权不变，旧失败费用不清零。

CE23唯一函数edit已集中实施，完整原始头逐条派生和最终SHA核验通过；16bit/FP4/repack
分支保留原数组与循环，只有原FP8 float-QK PV用一个half块。FI仍AUDITED，尚未签新数学、
设备或codegen。ce23-score-stream-plan为READY，产品授权仅FI且无issues；共同lock保守
选择全库的selected/skipped理由保留，按本卡FI与真实直接consumer门执行，不全局构建。

CE23[实际数学/device](../../../../evidence/correctness/inference-g04-g05-cuda-ce23-score-stream-source-device/receipt.json)
17math、十mem/十race/两普通race全部PASS，零FAIL/SKIP及error/warning/hazard；source阶段
429.651s，device17.479s。[同实例生成代码](../../../../evidence/architecture/inference-g04-g05-cuda-ce23-score-stream/codegen-receipt.json)
显示D256两实例仍为255寄存器、48/56字节栈、9/9与12/11条LDL/STL，和CE22相同；
留出栈从144增加到160字节、LDL/STL32/33。资源目标未达，不启动长后续门，也不冻结为
新的性能candidate。源码数组的显式容量不能替代编译器实际存活/分配证据。

按[精确回退记录](../../../../evidence/architecture/inference-g04-g05-cuda-ce23-score-stream/rollback.json)
移除CE22页偏移与CE23概率片段两项未达目标的改动，三个产品输入逐字归一化恢复到9c9；
该基线原source/device、SDK、三host及numeric证据可按相同产品输入复用，qualified9 FAIL
继续有效。两次尝试的代码、实际检查、资源反证与费用全保留，不把回退写成新PASS。
FI仍AUDITED，下一步只读核对packed tile逐lane读写约束与原16bit计算预算；不盲删
最小对齐保护，不扫tile变体。G04/G05未关闭，G06不实施。

## CE24：准确 packed 行约束与原16bit计算粒度上界

[源码与逐lane范围审计](../../../../evidence/architecture/inference-g04-g05-cuda-ce24-packed-math-budget/source-audit.json)
绑定恢复后的9c9头与原mma/permuted_smem/fragment shuffle SHA。原128B producer实际要求
NUM_MMA_KV×4可整除NUM_WARPS_Q，64B为×2；统一FP8×2 guard对128B偏保守。枚举本域
64/64、64/128、128/128、256/256的四warp布局及1/2/4/8 KV块，合法producer的全部16字节
写槽恰覆盖一次；128B的K/V LDMATRIX行地址全部落在相应16行块内，shuffle只跨同一warp。
128B的MMA_KV1并不要求删除对齐保护；64B与FP4原更强约束仍需保留。

原dispatch只按FP8压缩payload算KV存储预算，却在QK/PV中宽化到16bit片段。固定两真实
消费者qualified9 FAIL、原255寄存器和栈反证，与该dtype缩存储却扩大每warp计算块的结构
形成两把钥匙。借原native16bit footprint作为FP8计算粒度上界，不把它冒充实际shared字节
或精确寄存器计数。实际W1查询每SM102400、每block opt-in101376字节；原公式投影D256
MMA_KV2→1，留出8→4，实际compressed shared分别40960/24576；这些不是吞吐PASS。

本轮唯一有限FIX包：FP8的IsInvalid与三dispatch最小tile采用原128B×4/64B×2真实行约束；
FP4保原×2。三dispatch额外以原16bit(QK+VO) footprint限制FP8的MMA_KV计算粒度，最低
夹到准确合法tile，继续与实际shared容量、原寄存器上界及原dispatch宏取min。普通16bit、
FP4和既有unsupported边界不变。baseline9c9；只准private manifest/NOTICE/FI lock身份，
FI已AUDITED并在实施前记录本包。不扫tile变体、不按benchmark行调参、不删所有guard。

原planner/CTA_Q/warp、转换与QK/PV/softmax、mask/scale/F32partial/finalcast、prefetch/
同步及权限退休全KEEP。先原17math、十mem/十race与两普通race，再核新1/1/4特化实际
codegen；生成代码只签资源，运行时选择还须由后续实际consumer核实。资源目标未改善
不启动长后续；若成立才冻结candidate，沿static/shared、受影响三host facets、新numeric、
原16AA/三形状/8strata/n15,31/5%/strict全部门。其它路线按不变输入复用；SM89/pool1/
split3/Ninja2、heavy串行与常设时间追加不变，原失败和费用全保留。

CE24四个manifest edit已集中实施：IsInvalid与三个原dispatcher，完整派生diff核对了真实
物理容量、原寄存器上限、native16工作集上界及guard的一致性。未改数学或加载器。
FI仍AUDITED，新数学/device/codegen尚未签发；按本卡有限门执行，未变路线不重跑。

CE24[当前6e57数学/device](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-packed-math-source-device/receipt.json)
已实际PASS：17math、十mem/十race及两普通race均零FAIL/SKIP与error/warning/hazard；
source阶段450.089s，device17.233s，artifact前后SHA一致。随后[新1/1/4特化生成代码](../../../../evidence/architecture/inference-g04-g05-cuda-ce24-packed-math-budget/codegen-receipt.json)
实际提取成功，和qualified9的2/2/8特化相比，三个STACK40/48/120全部变为0，LDL/STL
8/8、10/10、27/27全部变为0，寄存器分别255/248/254。只签实际静态资源改善；尚不
冒充运行时选中特化、动态shared、occupancy或吞吐通过。新符号与投影逐一绑定，未把
旧模板参数写成未变。

FI进入CANDIDATE6e57e28ad，继续串行static/shared relocated SDK、受影响三host facets、
新numeric，再一次短VERIFY_ONLY核实际runtime1/1/4身份，最后原AA/P1/strict。其它owner
不动；原qualified9 FAIL及两次未达目标的资源修正/费用全保留，G04/G05仍未关闭。

CE24[静态SDK](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-static-sdk/receipt.json)
与[动态SDK](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-shared-sdk/receipt.json)
均已实际PASS：各14 CPU math、13 native math、3 CPU与3 CUDA consumer，零FAIL/SKIP；
各12独立公共头、16 consumer TU仅用搬迁installed headers，无HYLE_TESTING/sanitizer。
static阶段495.607s，shared153.239s，分别列账。146个未变native对象经源码、原command/
Ninja依赖freshness与前后SHA核验复用；shared计划0个CUDA重编边，当前FI对象沿用本轮
已实测构建，未借用旧candidate或跳过实际消费者。

仅重绑原P1的实际CE24 static prefix、prose、全部三consumer SHA和当前numeric receipt
路径；规范化其余合同字段逐项相同、原JSON换行保留。原三consumer_evidence与嵌套prose
身份预检实际PASS；这不是numeric/A-A/吞吐PASS。三形状、原peer两屏障派生、16AA、
8strata、n15,31、5%与全部旧FAIL不变。

ce24-sdk-binding-plan沿[原独立工具事项](../build/validation-selector-shared-bench-paths.md)
因既有shared benchmark的三unknown路径BLOCKED，product_authorization无owner/issue；
docs/closure及146 skipped reasons保留，按本卡有限门继续。不在模块内改selector或公共
性能核心，也不把该工具归属状态写PASS。接下来三host facets串行批验，随后新numeric/
实际runtime身份及原AA/P1/strict；产品candidate6e57与其它owner不变。

CE24[ASan](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-host-asan/receipt.json)、
[UBSan](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-host-ubsan/receipt.json)与
[TSan](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-host-tsan/receipt.json)三项宿主
检查均已实际PASS：各16测试零FAIL/SKIP、5个原风险模式通过，6e57三vendor输入、四FI
CUDA TU实际instrumentation与运行前后二进制SHA均核验一致。[串行批次](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-host-tsan/batch-summary.json)
已TERMINAL_PASS，实际run/seal六阶段及全部费用保留。阶段分别536.579/474.641/470.187s；
外层launch分别541.985/478.063/473.266s，与各自阶段重叠，不重复相加。

原NVIDIA进程LSan及独立复现的diagnostic-thread lifetime仍INCONCLUSIVE，不由本三项
宿主检查代签GPU shader/driver/private-pool上界。当前source/device、两SDK和三host均有
6e57实际证据；接下来新numeric、实际runtime内核身份和原AA/P1/strict仍待签发。仅编译
比较consumer与原peer CUDA TU，不重编产品；产品candidate和其它owner不变。

CE24[新数值绑定](../../../../evidence/correctness/inference-g04-g05-cuda-ce24-packed-math-numeric-rebind/receipt.json)
已实际PASS：primary/placement各24个独立double oracle，最大绝对误差0.00109081301，低于
原8e-3；双slot重放/readonly、实际SDK artifacts/recipes/native archives及源码身份前后
一致。primary另跑memcheck/racecheck零error/warning/hazard，build-contract预检PASS。
只编译比较consumer与一个原peer CUDA TU；记录阶段76.610s，不包含前置WSL Git状态
核对等未计时开销。产品manifest新身份fa98f671，peer仍仅取原两输出屏障，selected SHA
47df06c7/effective prefill06189719不变，未把本轮guard/工作集规则应用到peer。

随后一次[实际运行时身份trace](../../../../evidence/architecture/inference-g04-g05-cuda-ce24-packed-math-budget/runtime-receipt.json)
已PASS：同一数值binary的24oracle通过，前后SHA相同；实际mangled kernels准确命中新
1/1/4特化，register255/248/254、dynamic shared40960/40960/24576、block32×4×1，
与本轮静态资源投影一致。capture阶段2.966s，只签实际身份与数学，不签吞吐/occupancy。

source/device、两SDK、三host、新numeric和实际runtime身份至此均有当前6e57证据。
下一独立attempt10按原16AA/8strata/n15,31/5%/五controls/三consumer/strict合同运行；
资格预检继续前置到configure、A/A与采样之前。产品仍冻结6e57，所有旧qualified FAIL、
两次未达资源目标的修正及费用保留，整个G04/G05和完整goal仍未关闭。

## CE25：6e57 合格结果与剩余 prefill 成本

[attempt10](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-10/receipt.json)
已实际TERMINAL：QUALIFIED、performance FAIL、strict verify0。原16AA、三consumer与
三validator、五controls均PASS，artifacts/inputs/host稳定；n15完成120产品、30控制进程，
因明确prefill FAIL不追加n31。worker2679.859s与外层2715.844s重叠，不重复相加。

windowD256 ratio0.977278–0.983388、方向LOSS/5%非劣PASS；prefillD256为
0.852769–0.858133、方向LOSS/非劣FAIL；noncausal D64/Dv128为0.976652–0.989723、
方向INCONCLUSIVE/非劣PASS。静态零栈读写和实际1/1/4身份成立，不能替代prefill预算。
原9与8 qualified FAIL及全部源/数学/device/SDK/host/numeric证据和费用不覆盖。

FI保持CANDIDATE6e57，不由失败自动授权新产品修改。下一有限只读审计复用已捕获的
当前6e57 runtime trace，按形状分离attention/merge成本，再核锁定69ff的原FP8 prefill
共享BF16 tile repack路径与当前MMA1预算。原源码在CTA_Q>16、有限FP8域本来启用该
路径；CE21为当时双tile/大shared成本关闭它，不能假定修正packed预算后仍应关闭。
先证明原helper、物理容量、每SM实际shared上界与数学/同步一致，再固定唯一有限FIX并
合法回AUDITED；不试多变体、不改peer/输入/5%或公共仪器、不新采相同P1。G04/G05仍
未关闭，G06不实施，完整goal仍3/10。

CE25[原prefill路径源码审计](../../../../evidence/architecture/inference-g04-g05-cuda-ce25-native-prefill/source-audit.json)
已完成：原69ff六处storage/traits/三dispatcher开关逐一核对，确为有限FP8、VO非64且不大于
256、CTA_Q>16时的成熟原路径。复用现有trace按形状分离，prefill attention均值约
54.132/38.599us、merge11.533/8.711us；这些短trace只作机制证据，不另签性能。
当前寄存器路径按Q warp重复FP8 fragment置换/转换，原helper可协作转换当前tile，再用
原16bit ldmatrix/MMA；已有short-K SOURCE_COLS修正须保留。

唯一有限FIX为恢复该原六开关的一致性，保CE24准确行约束/native16工作集上界，以及
F32 partial/最终scale+cast、两输出屏障、short-K补零/stride和原权限退休。baseline6e57，
仅FI合法回AUDITED后改private manifest/NOTICE/FI lock身份，不新增API、算法、全历史
展开或tile试验。D256仍MMA1，shared含8192字节stage共49152；留出仍MMA4，stage16384、
shared40960。按已查询shared容量均可容纳两CTA；这不是总occupancy或吞吐证明。

先一次集中实现，再原17math、十mem/十race/两普通race及新codegen；静态/动态搬迁SDK
各33测试/12公共头/16TU后，先新numeric与一次短runtime身份核对，尽早核实际路径与数学，
再串行三host风险门，最后原AA/P1/strict。numeric与host门相互独立，全部仍须在P1前完成，
未运行门不记PASS。三形状均保留，不以修prefill回退window/留出已过预算；原8/9/10FAIL、
全部旧费用及第三方UNKNOWN保留。SM89/pool1/split3/Ninja2、heavy串行/常设追加不变。

CE25原六开关已集中恢复；删除manifest中冗余的关闭层，并将后三个CE24 dispatcher补丁
重绑到原开关，完整派生diff只显示六处谓词恢复。CE24准确行约束/工作集、F32输出tile
容量、short-K stride/补零、原数学与同步均保留。FI仍AUDITED，尚未签新数学/device/
codegen或吞吐。按本卡有限门验收，不重跑未变其它路线。

CE25冻结candidate为0143102ae。新[源码与设备风险记录](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-native-prefill-source-device/receipt.json)
实际17math、十cache memcheck/十racecheck及两普通F16/BF16 racecheck全部PASS，无FAIL/SKIP、
error、warning或hazard；source419.578s、device17.155s。新[codegen记录](../../../../evidence/architecture/inference-g04-g05-cuda-ce25-native-prefill/codegen-receipt.json)
逐一核实际同1/1/4特化，STACK/LDL/STL保持全部零，register从CE24的255/248/254降为
241/241/165。该结论仅为当前source shared artifact静态资源，不能签runtime dispatch或吞吐。
FI由AUDITED合法进入CANDIDATE014；下一步两搬迁SDK、新numeric/runtime、三host及原P1，
G04/G05仍未验收，完整goal仍3/10。全部旧失败、费用与第三方未知保留。

CE25两搬迁SDK均已实际通过：[static](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-static-sdk/receipt.json)、
[shared](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-shared-sdk/receipt.json)各33测试、12公共头、
16消费TU，零FAIL/SKIP；shipping无HYLE_TESTING/sanitizer、仅从搬迁安装头消费。146未变ggml CUDA
对象逐一核源/依赖/原命令与Ninja freshness，未触发原生CUDA全量重编。实际stage费用static
437.962s、shared151.052s分别保留。
性能绑定仅将真实SDK prefix、prose身份、三个workload的provenance SHA及新numeric receipt路径
迁移到CE25，原contract其它字段与换行保持；原三consumer+nested prose preflight PASS。
下一步新numeric及一次runtime身份核对，再三host门和原P1；本次不签数值/吞吐或G04/G05完成。
本次ce25-sdk-binding-plan仍因同三shared bench JSON/Python路径unknown为BLOCKED，
product_authorization无owner/issues；docs/closure及146 skipped reasons保留，沿既有独立工具
事项与本卡有限门继续，不改公共selector或将工具状态签为PASS。

CE25真实消费数值与runtime身份已实际核对：新[numeric记录](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-native-prefill-numeric-rebind/receipt.json)
两个variant各24个独立double oracle通过，max_abs_error=0.00109081301低于8e-3；primary
memcheck/racecheck零error/warning/hazard，build-contract与source/SDK/artifact只读核验PASS。
仅编消费者及原peer一个CUDA TU，产品库不重编。实际记录stages费用78.441s；
初始WSL canonical Git状态/身份检查位于该stage计时前，未记录部分不记零。
新[runtime记录](../../../../evidence/architecture/inference-g04-g05-cuda-ce25-native-prefill/runtime-receipt.json)
为同一新static SDK消费binary的一次VERIFY_ONLY trace，3.266s，24 oracle通过、SHA不变；
实际三mangled特化与codegen完全一致，MMA1/1/4、register241/241/165、dynamic shared
49152/49152/40960，block32×4×1。此为路径/数学证据，不作吞吐或总occupancy结论。
接下来只跑原三host风险门与原P1；G04/G05未完成，所有旧FAIL/UNKNOWN与费用不覆盖。

CE25三个宿主风险facet均已实际通过：[ASan](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-host-asan/receipt.json)、
[UBSan](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-host-ubsan/receipt.json)、
[TSan](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-host-tsan/receipt.json)，
每项16测试/零SKIP及五风险模式，四FI CUDA TU均按本facet带宿主插桩，当前candidate三vendor
输入与binary SHA不变。费用逐facet保留：
asan stage481.298s/outer485.766s；
ubsan stage412.277s/outer420.796s；
tsan stage414.626s/outer425.219s；
outer与stage互相重叠，不重复相加。[批次汇总](../../../../evidence/correctness/inference-g04-g05-cuda-ce25-host-tsan/batch-summary.json)
不签shader/driver/private-pool证书，原进程LSan与diagnostic-thread lifetime INCONCLUSIVE保留。
现在只剩原同合同AA/P1/strict及整项终审；本次不签性能或G04/G05完成，不重置旧失败/成本。

## CE26：014 合格结果与剩余成本只读复核

[attempt11](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-11/receipt.json)
已实际TERMINAL，candidate014/source681a5277e，QUALIFIED、performance FAIL、strict verify0。
原16AA、五controls、三consumer和采样前后三validator均PASS，artifact/input/host稳定。
n15完成120产品与30控制，明确prefill FAIL不追加n31。worker2629.221s与outer2691.563s
互相重叠，不重复相加。windowD256 ratio0.986884–0.993310、方向LOSS/5%PASS；
prefillD256为0.913710–0.926685、方向LOSS/5%FAIL；noncausal D64/Dv128为
0.981282–1.001226、方向INCONCLUSIVE/5%PASS。CE25恢复原路径使prefill较原10改善，
仍未到原门槛，不将该改善签为性能PASS。原8/9/10及全部失败/费用不覆盖。

FI保持CANDIDATE014；失败不自动授权改产品。下一步仅复用已有CE25 runtime trace，
按三个实际形状分离attention、merge与其它GPU工作，再对照当前源/原peer的partial、
最终scale/cast与调用边界，找出剩余可复核成本。15min只读检查点沿原用户常设授权追加、
保留累计；不新采profile或同配置P1，不改peer/输入/统计/5%，不试tile变体。只有实际
消费material缺口与具体源码机制两把钥匙成立，才固定有限FIX并合法回AUDITED。
当前没有新FIX或运行资格，G04/G05未关闭，完整goal仍3/10，G06不实施。

CE26唯一有限FIX已固定：[单输出组源码审计](../../../../evidence/architecture/inference-g04-g05-cuda-ce26-single-output-merge/source-audit.json)
证明原threadblock_sync_state先同步各ty局部状态，再由每个ty按完全相同顺序读取/合并相同
共享状态；当前D256四组重复最终归约、归一化、scale和相同输出写入，单partial分支也重复
访问。实际原/产品grid一致，未发现plan分片差异；原F32 float2直写和成熟向量FP8转换保留。

baseline014，仅FI合法CANDIDATE→AUDITED后实施：只为现有SCALE_OUTPUT=true的F32
merge选择ty0承担相同顺序的最终跨组归约、归一化/scale/sole cast与输出；零/单partial
最终访问也只由ty0执行，optional LSE只由tx0写。其它ty仍提交原局部状态并参与全部原
CTA/cp.async同步；原SCALE_OUTPUT=false路径保持原行为。D64/128/256按真实bdx/bdy
逐地址验证唯一覆盖，不新增数学、算法、tile、pipeline配置或API，不缩窄partial或移除屏障。

两把钥匙为qualified11主consumer material FAIL与该可复核重复所有权成本。先集中修复
private cascade manifest/NOTICE/FI lock，再原17math、十mem/十race/两普通race及codegen
（新增实际merge特化资源/指令核对），两搬迁SDK、新numeric/runtime、三host和原P1。
默认旧FI实例及三形状均保留；未运行门不记PASS。原2h检查点/自动追加/累计费用、SM89
pool1/split3/Ninja2/heavy串行不变。当前仅审计与授权，尚未改产品或签新运行/性能。

CE26单输出组修正已集中实施，只改private cascade的三个精确manifest片段及NOTICE/FI lock；
完整派生diff保留所有CTA/cp.async同步调用数，局部state写入和归约顺序不变，默认false
特化保原行为。FI仍AUDITED，尚无新数学/device/codegen/性能结论。selector
ce26-single-output-plan为READY；共同lock变化保守选择全库因子，按本卡实际FI/direct
consumer有限门执行，保留selected/skipped理由，不启动无改动路线或全库构建。

CE26新数学与设备门已实际通过：[source/device](../../../../evidence/correctness/inference-g04-g05-cuda-ce26-single-output-source-device/receipt.json)
为17数学、十cache memcheck/十racecheck及两普通F16/BF16 racecheck，零FAIL/SKIP/error/
warning/hazard，artifact SHA稳定。实际stage费用source415.946s、device16.681s分别保留。
新[codegen](../../../../evidence/architecture/inference-g04-g05-cuda-ce26-single-output-merge/codegen-receipt.json)
逐一核当前相同MMA1/1/4 attention及F32 merge，四个kernel均STACK/LDL/STL零；实际
register分别241/241/165/48。该静态资源与opcode结论不能代签运行线程谓词或吞吐。
FI合法冻结CANDIDATE1e343ee3b；接着两搬迁SDK、新numeric/runtime（须包含实际merge
身份）、三host及原P1。新候选尚无性能结论，G04/G05未完成，原11及全部旧失败/费用保留。
