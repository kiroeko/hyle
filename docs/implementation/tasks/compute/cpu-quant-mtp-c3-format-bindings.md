# C3：CPU 格式与普通算子批量接线

<!-- MODULE REVIEW: AUDITED -->

## 计划身份

- 状态：AUDITED，已固定G1–G4、N1–N6范围；Native已绑定本卡的原CPU解码/MLX组合接缝。
  本卡整体未完成，适用资源、平台、交付和画像资格不得被跳过。
- baseline：`1efa59bd218adef71de0c3606cd2e968e3f953e3`；Windows实际HEAD核对、工作树干净，保留全部后续提交。
- owner范围：compute_ops、compute_provider_ggml；C1准入的ATen/借用原程序由compute_provider_native承接，CPU facade只在真实缺口时按owner纳入。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- scope：Q01–Q11公开block身份/描述/原字节import，现有MatMul、GroupedMatMul、Gather/GetRows在真实端口的成熟接线。
- C1若准入NF4/AWQ/GPTQ/MLX制品，添加其具名packing/成熟carrier；没有审计不能猜成GGUF4-bit。
- 不扩全量ggml类型目录、不自研decoder/GEMM、不以整个文件的Q4名替代tensor编码；按原block/role使用。
- 正常consumer是混合profile projection/expert/embedding；正交普通压缩数值矩阵/索引处理；无需模型loader。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| Q1 | 一份语义format映射和checked描述器批量扩展11格式，不自动强造所有格式×端口组合 | block bytes/alignment/rank/group/shape与原码相符，unknown/越界明确错误 |
| Q2 | 原ggml CPU import+原MatMul/MUL_MAT_ID/GetRows，共用program/参数化fixture | 直接执行压缩字节，完整精度/duplicate expert/checked index合同，必要转换冷处理 |
| Q3 | 已准入非GGUFcarrier复用原代码/成熟组合，遵守codebook/zero-point/permutation等 | 原tensor角色与排除项保留，prepare/replay资源/失败/并发真实成立 |
| Q4 | 普通Direct/Graph自然绑定、完整region/资源复用/重复执行 | 不需用户排名，热路径不编译或展开全库权重，失败不被不当fallback掩盖 |

## 验证、预算与停止

- 全批一次接线、参数化oracle与归并signature，最多一批定向纠正；不每个格式开一棵build树。
- Factor tests/header/普通入口/资源/OOM/两系统三交付按stable candidate集中执行；适用风险未运行写未验。
- 数值对照采用同编码/累加/输出和边界的原成熟程序，非同合同只记录差异。
- P0为主；若合格CPU库拒绝必需格式/端口，具名记录，不自写替代、不将拒绝算完成。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。

## 用户最新性能口径

单一合格成熟carrier完成正确接入与P0；多家成熟库具备同合同重叠域时按总计划批量比较固定默认规则。
不比较自研，不为每个算子/档位重建或计时，最多两个代表消费族共用一份稳定候选P1。

## GGUF 有限审计与 API 冻结

来源：C1完整104画像/826分片实际头；C2固定CPU core与原IQ1 MatMul/GetRows资源/数值验证。
现有`gguf_weight_description` / `gguf_grouped_weight_description`、MatMul、GroupedMatMul、Gather保留，
不增加新执行器或模型loader。公开枚举追加11项，具名StorageFormatId追加不覆盖旧ID。

| FIX | 原问题/源码 | 固定产品动作 | 验收 |
|---|---|---|---|
| G1 | low_precision.h/.cc只列10编码 | 批量追加Q4_0、Q4_1、Q2_K、IQ2_XXS/XS/S、IQ1_S/M、IQ1_XXXS/XXS/XS，原bytes/block/alignment | 原编码描述、非法/溢出/伪造格式、checked import |
| G2 | CPU native_quantized_type无11格式；embedding仅Q4K/Q6K/Q8 | 对应原ggml type；全部新增格式只在C2新CPU envelope下接受，旧GPU SDK不代签其CPU新域；embedding依实际decoder能力接受 | 一批参数化MatMul、GroupedMatMul、GetRows；重复/越界ids、changed replay |
| G3 | 三GPU helper的enum switch封闭 | 仅添加default返回COUNT，拒绝新增未验格式；不接新GPU数学/数据布局 | 编译期consumer seam/source合同；既有能力与GPU状态不变 |
| G4 | 需要普通入口真实消费 | 普通CPU Direct和Graph参数化fixture，保留原压缩bytes、F32/Q8_K内部转换边界 | 原编码参考、多个slots、资源/失败/close、批末两系统三交付 |

G1/G2 owner compute_ops、compute_provider_ggml。G3的compute_provider_ggml_cuda/metal/vulkan保持
原AUDITED状态，只将当前卡绑定为公共enum拒绝分支的补充范围；原D5资格/限制仍见
[原记录](llm-moe-d5-real-fragment-acceptance.md)，本批不恢复旧GPU队列或Graph BLOCKED。

语义：weights原block、MatMul逻辑[K,N]，grouped[E,N,K]，embedding[N,K]；quant GetRows输出F32。
原库MatMul内部activation Q8_0/Q8_K转换由其CPU trait决定，对照包括该转换；F32累加，声明输出dtype。
prepare只按descriptor建元数据/pool/workspace；payload每次绑定，可changed replay，不偷偷缓存mutable weights。
ids在写输出前checked；不同slot共享不可变bytes、独占workspace/pool，borrow owner活至最后完成。
资源/销毁/OOM复用C2已有原库边界；hot path不重新搜索/计时/编译或展开完整权重。

MatMul/Grouped/Gather从现有普通入口自动绑定mature provider，用户不选择Provider或timing profile。
新格式只签实际CPU envelope，GPU保留原锁、旧资格和明确拒绝。单一原ggml carrier无适用赢家P1。

非GGUF后续有限审计须固定packing、codebook/nested scale/zero-point/permutation端口和cold转换边界，
才能把Native从C2绑定本卡。现有Float8WeightDecode提供“显式执行一次→作为constant复用”的自然先例；
不得把每轮全库解包、未资格化的4bit等价替换或放宽精度写成完成。

## 非GGUF第一组有限审计（N1–N3）

Native仅对下列原CPU解码路径追加AUDITED范围；MLX压缩矩阵/专家仍独立审计，不以本组替代。

- N1：新增`model_ops::ArtifactWeightDecode`纯计算接口。它接受原始typed tensor，输出原数据类型的
  [N,K]权重；没有文件读写/模型解析。NF4默认BF16（本制品原quant-state dtype），AWQ/CT默认scale dtype。
  显式运行一次再绑定constant，沿已有Float8WeightDecode自然路径；不能藏到每次MatMul或解包整个模型。
- N2：borrow pinned bitsandbytes/AutoAWQ/compressed-tensors原ATen计算次序，保留codebook、offset、
  逆AWQ排列和原int4 signed offset；Native原ATen boundary扩至6输入，无新自研decoder/GEMM。
- N3：Direct/Graph cold-decode→普通成熟MatMul/Gather完整消费；checked形状/溢出、OOM、changed replay、
  共享参数/独立slot/close和精度差分；API稳定后与GGUF共用批量验证。

| 格式 | 输入端口与实际源 | 输出/成本 |
|---|---|---|
| NF4 | U8[N,K/2]，F32[N*K/group] absmax，F32[16] codebook；高nibble在前 | F32 codebook/scale乘后转原dtype；caller显式持有decoded weight |
| NF4 double quant | U8[N,K/2]，U8[G] absmax，F32[ceil(G/256)] nested absmax，F32[256] nested codebook，F32 scalar offset，F32[16] codebook；G=N*K/group | 原nested scale算式＋offset，再原NF4解码；不把U8 scales当F32 |
| AWQ GEMM int4 | I32[K,N/8]，I32[K/group,N/8] zero，floating[K/group,N] scale | 原列unpack＋[0,4,1,5,2,6,3,7]逆序，(q-zero)*scale，再转[N,K]；默认F16 |
| CT symmetric int4 | I32[N,K/8]，floating[N,K/group] scale | 原dense low-first pack减8并应用scale；默认BF16；完整头证明本制品无g_idx/zero-point，不扩未需求格式 |

NF4 group64、AWQ/CT group128为实际目标；参数只控制原数学分组与输出dtype，不控制Provider选择。
所有输入要求plain dense，K完整group/packing，invalid shape/enum/type/overflow显式失败。
Decoded weight驻留和cold临时量明确记账；这组只面向实际dense Qwen制品，不能用于MLX全专家热展开。
MLX仍须借入原压缩CPU调用/selected expert边界后才签范围。

## 第一组实施与有限验证

- GGUF G1/G2批量接入已提交，11格式的普通CPU Direct/Graph MatMul、GroupedMatMul、GetRows及
  changed replay、重复ids/越界写保护实际通过；失败后按公开合同退役slot，再由原artifact创建新slot。
- N1/N2原bitsandbytes NF4/nested、AutoAWQ GEMM、CT0.18 symmetric4 CPU序列已接入，F32格式差分通过。
  后续候选已补验N3默认BF16/F16、cold constant MatMul及changed activation的Direct/Graph消费；资源失败格仍待验。
- 两组typed consumer和factor suite共享Windows producer，完整候选/命令/失败修正边界见
  [第一组证据](../../../../evidence/correctness/cpu-quant-mtp-c3-first-batch/README.md)。
- MLX affine/mxfp4/mxfp8 linear/选中专家F32/BF16、affine BF16 embedding已接入并通过Windows有限矩阵；整体C3保持AUDITED、不签完成。
  Linux/三交付、C4全部新分支和C5资格矩阵仍未完成。
- P0：GGUF原packed byte直接执行；dense artifact解码是显式cold操作，caller支付并持有decoded存储。
  没有性能P1、自研算子扩充或SIMD删除。

后续混合格式候选与实际范围见[第二组证据](../../../../evidence/correctness/cpu-quant-mtp-c3-mixed-batch/README.md)。
统一29路径当前实现61,471行/250文件；不含测试/vendor/bench，完整LOC收据在第二组证据。

## MLX 有限审计与 N4–N6 冻结

固定MLX `2b2bb49790158ecd008d1685c94e3ef119792a00`，原`ops.cpp`与
`backend/cpu/quantized.cpp`、QuantizedEmbedding源码已读取。实际制品需要affine4/group64、
mxfp4/group32、mxfp8/group32；172份rank3压缩权重涉及专家或分组projection，不能全库解包。

- N4：`model_ops::MlxQuantized`具名纯计算接口/工厂支持linear、grouped expert、affine embedding。
  Raw U32 packed weights与float scales/bias或U8 E8M0 scales分别作为端口；A为F16/BF16/F32。
  Linear/grouped以原CPU SIMD/F32 accumulation公式执行，affine输出为A/scale共同浮点dtype，
  MX输出保持A dtype；MX scale先按原CPU转换到该dtype，再F32累加，不隐去低精度scale溢出/下溢。
- N5：保留原MLX CPU公式，ATen承载block dot/编码读入/矩阵与cast；原K group顺序累加，
  MF先group dot再scale，affine先q*scale+bias再dot。只展开当前K group和选中expert，
  不构造完整[E,N,K]浮点库。Embedding沿原take→affine_dequantize，只展开选择的rows。
- N6：普通Direct/Graph三格式＋grouped/repeated ids＋embedding原dtype、source oracle、changed replay、
  bounds/OOM/独立slot/close；与dense cold-decoder默认dtype与完整constant消费一起签下一批。

接口input顺序：linear A[M,K], U32 W[N,K/(32/bits)], scales[N,K/group], optional affine biases；
grouped A[T,S,K], W[E,N,K/(32/bits)], scales[E,N,K/group], optional bias, I32 ids[T,S]；
embedding W[N,K/8], scales/bias[N,K/group], I32 ids[M]。所有storage plain dense。
输出为linear[M,N]、grouped[T,S,N]、embedding[M,K]；浮点精度和真实storage分开记录。
无模型loader、sampler、Provider排序、新自研数学/优化或GPU能力扩充。

一次有界port；不以MLX的Apple现有资格代签CPU。同次序source reference与F32 arithmetic关联误差
分别记录，不能拿未量化matrix或不同accumulation合同签胜负。新Provider/格式全部适用门仍待验。
