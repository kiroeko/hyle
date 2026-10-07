# F5 CPU自研源码清退

<!-- MODULE REVIEW: AUDITED -->

当前owner `compute_provider_native`；`compute`仅绑定两份私有RoPE生成文件的有限删除，不重审runtime。
baseline `0d96c635040e70b1ee561e51e4cb1a74169870f0`；总合同[CPU最终收口](cpu-final-closure-plan.md)。
F4已完成当前软件的有限全候选优选／工程默认与三平台CPU-only／四mixed三交付，
由[35种源码交接](cpu-final-f4-retirement-handoff.json)驱动本轮真实冗余清退，不继承旧187/17。
自研胜出只保留有资格的必要族/子域；无完整优势且有合格成熟接替的自研实现整项源级删除。
共享kernel若有胜出域保留必要分支，劣势路由不进入默认；不能用后备1000或旧证书证明价值。
真实独有动态/资源消费先成熟替代或明确用户能力取舍，不能删后才发现Mac或通用Graph失能。
SIMD/Mem独立基础保留；仅为已退役算子增的私有SIMD扩展可按真实需求一起裁决，借用原库组合不误称自研。
清理实现/CMake/私有plan/无需求入口及旧测试，新增明确拒绝和普通成熟消费者，不新增数学优化。
稳定后只受影响correctness/两入口/三交付/P0，未改变赢家数学不重复P1。

## SCOPED：有限审计、消费者和预算

主consumer为固定104 GGUF＋官方／非GGUF／MTP／DSpark的普通CPU Direct/Graph；
正交consumer为无模型身份的tensor管线、strided/empty/alias/checked-index与具名动态／严格资源消费。
MatMul/SDPA及其它已退役能力不复活；图语义、public descriptors、runtime／完成模型和Provider接缝不重构。

| 审计项 | 固定边界／输出 |
|---|---|
| A1 冗余14种 | Abs/Sigmoid/Softplus/Multiply/Maximum/Select、Sum/Mean、LayerNorm、ScatterAssign/ScatterAdd/Concat、RoPE/LogSoftmax逐语义／精度／alias／bounds／资源核对成熟接替 |
| A2 20有优势种 | 从F4实际41精确WIN点映射必要dtype／variant和共享源码分支；不以部分WIN或未资格点扩保留，不改赢者数学 |
| A3 独有与借入 | 具名declared-domain F32 RMSNorm和Copy／严格资源；View别名、Mem/SIMD基础、TorchAO/ATen/MLX/DS4原程序与codec分清 |
| A4 私有生成与owner | `src/compute/rope_kernel_impl.h`／`rope_kernels.h`仅被Native引用；有限Compute owner覆盖其删除，不删除SIMD通用机制 |
| A5 验收迁移 | 旧Native-only tests按真实当前消费者迁成熟普通入口／明确Native拒绝；现规则43点/普通19/20、失败/OOM/alias/state/declared-domain及三平台三交付差异批量执行 |

调查预算只一批上述源码／现有资格对账；不新开性能调研／AA／P1、不扩算子／依赖，不优化self。
实现前依法AUDITED固定有限FIX；真实缺成熟语义／资源或不能合法撤回的具名能力必须说明，不用删除后skip冒充完成。
需要用户取舍时先给具体源级方案和保留／撤回能力，不把所有旧benchmark严格域自动当新的必需consumer。
stop：有限FIX与适用差异验收全部结束裁决CLOSED；真正缺实现／资源取舍按协议明确，不无限研究。

## AUDITED：成熟完整数学与有限FIX

源码对账：ggml普通supports的Unary/Multiply/Reduce/Norm/Concat/Scatter/RoPE，
ATen `select`／`numeric_program`／原byte-storage和稳定math、现有AtenNumeric/Storage/Transformer资格，
以及Native各源族/dispatch/plan。F4每个Native准入点已有成熟数值承载；未重新采样。
成熟ATen补齐ggml的stable tail、I64/odd-stride/全scalar位存储、LogSoftmax/Bool和rank/布局边界；
没有新增数学缺口，唯一准入者不造性能竞争。原NaN／signed zero／F32累积／checked写前拒绝必须保留。

| 源族 | 冻结动作 |
|---|---|
| elementwise | 源级删Abs/Sigmoid/Softplus、Multiply/Maximum和Select；保留F4资格需要的Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Relu/SiLU、Add/Subtract/Divide/Minimum必要dtype实例；Gated仅已资格F16 separate-SiLU |
| reduction | 删Sum/Mean执行／finish helpers；保留已资格F16 Min/Max、BF16 Max所需共享算术与axes/layout逻辑 |
| normalization | 删LayerNorm／bias／非RMS分支；保留F32动态／严格RMS与BF16已资格RMS必要共享分支 |
| indexing | 删ScatterAssign/ScatterAdd/Concat执行及字段；保留有资格Gather必要共享checked/I32/I64和bit-storage路径 |
| transformer | 删RoPE全部执行/准备、两私有生成文件和CMake fat target；删LogSoftmax及未资格Softmax dtype/mask分支，保留F16 Bool Softmax必要共享执行 |
| cast/transform | Cast保留4个资格dtype对的必要共享转换，Copy保留具名动态／严格资源与Mem机制，Fill保留F32 bit-write；View别名保留 |

自研数学从35种归到20有资格种＋独有Copy；分支不因为unmeasured或优先级1000永久保留。
SIMD通用abs/multiply/sum/LayerNorm/RoPE等独立基础不在删除范围；仅删Compute私有RoPE生成kernel。
Native borrowed TorchAO/ATen/MLX/DS4程序、generic schema/Graph/mature providers不改。

用户已有明确“落后／没有完整优势且成熟接替就删除，SIMD本身保留”的源级清退授权。
本批对应Native-only及其旧0-private/prepared-only数学域随冗余实现撤回，严格请求若没有合格成熟
资源承载就明确Unimplemented／ResourceExhausted，不把unknown写0、不放宽预算。
这些不是当前固定模型必需的独有数学；具名F32 declared-domain RMSNorm／Copy资源consumer仍保留并验收。

| 有限FIX | 实施／验收 |
|---|---|
| D1 | 上表14整种与未需要实例物理删除、dispatch／CMake／NativePlan字段清理；Native expert明确拒绝，不仅降优先级 |
| D2 | 原Native-only math tests迁普通成熟入口／明确拒绝；保留资源／算法必要测试，不能用skip或简单删除覆盖替代真实数学验收 |
| D3 | 文档与35种→21种／必要dtype source ledger；当前43规则保持，数学代码没有优化改写，旧P1身份按保留算术源码对账复用 |
| V1 | 批量完成后集中owning/普通两入口、43规则、14成熟替代特殊值/layout/alias/index/empty、RMS/Copy动态/预算/OOM/close/并发、固定C5完整旅程及受影响source/static/shared三平台差异；按selector保留适用与复用理由 |

预算为一批删除＋一批具名测试迁移纠正；任何成熟数值/资源新缺陷依法另定向绑定，不能在此自研补洞。
不重跑P1／AA，不增加新Provider／依赖，不把旧GPU状态或公共治理纳入。

### D1具名接线反例与D4边界补齐

删除前的正交反证：`NativeElementwise.HalfStorageSupportsOddByteStridesAndBroadcast`和
`[2,3]` F32 byte-strides`[10,4]`实际消费，旧Native scalar路径可执行；ggml native描述要求element-aligned
strides，ATen原math存在但其selected/numeric_program只接稳定Unary和Min/Max，基础Unary/Binary未接。
这是真实接线缺口，不能删除后把原合法布局签成完整替代；不因此保留一整套自研数学。
D4限定现有ATen原abs/neg/exp/log/sqrt/rsqrt/tanh/relu和add/sub/mul/div的edge布局／empty／rank0或高rank
接线与对应alias／checked稳定语义。只在非dense、empty、rank0或rank>4的原语义请求准入；
F4全部dense rank2准入集合和43 WIN候选不扩大，不新增数学loop、Provider、依赖或采样。
适用V1新增真实edge正确性／三交付差异，原失败保留；发现此反例后本轮不再宣称原14种无条件全布局接线齐全。

### D1产品删除批与V1首批

14整种执行／准备／dispatch已源级清退；Compute两私有RoPE文件与fat target移除。
保留族按F4必要dtype实例收窄，RMS/Copy具名资源、View、SIMD/Mem及借入codec／程序保持。
产品库编译通过；清理中误删的Softmax共有axis/mask helpers已恢复，原编译FAIL保留。
当前owning首批62项全部通过、无skip，包含14种明确Native拒绝、成熟odd-byte基础Unary/Binary
真实数值和变输入重放、原ATen稳定函数／storage／transformer／资源与Native RMS/Copy动态／OOM／owner边界。
UInt32 Fill的旧Native-only测试迁原ATen bit-write，domain反偏置由仍保留的F32 Exp作exact反例；
初次两项旧fixture FAIL仍保留。本轮没有P1或自研优化。
其余旧Native-only目录的迁移、保留20种全部必要算术重放、两普通入口／C5／三平台三交付差异与
source ledger／终审仍未完成，owner保持AUDITED，不能据首批PASS关闭F5。
