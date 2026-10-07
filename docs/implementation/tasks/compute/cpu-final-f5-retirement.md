# F5 CPU自研源码清退

<!-- MODULE REVIEW: CLOSED -->

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

### D2 owning测试迁移批

旧全域Cast／基础elementwise／LayerNorm／Reduce／Scatter／Concat／RoPE和宽Softmax数值测试
迁现有ATen原程序，保留独立oracle、特殊值、odd byte stride、checked index、empty与变输入重放。
被撤回的dense Select exact-alias明确在提交前拒绝且原位数据不变；没有以skip替代旧能力取舍。
Native测试限定保留dtype／variant，新增4个Cast资格对、BF16 RMS参数精度／独立invocation和
F16 Min/Max、BF16 Max的dense／odd-stride／多axes实际数学检查；RMS／Copy资源测试保留。
当前Windows owning全部144项PASS、无skip；原142项首次迁移4项FAIL、随后1项FAIL以及
补充测试首次namespace编译FAIL原log均保留，修正后全部通过。测试迁移未改产品数学／43规则。
原始记录在`.scratch/validation/cpu-final-f5/receipt.json`与owning XML，最终将集中归档。
普通两入口／C5／三平台三交付差异与source ledger仍待验，F5保持AUDITED。

## CANDIDATE：产品固定、差异验收继续

产品candidate `507bad9698f9988d371c59e5661a69e4394c580a`；D1／D4源码与43规则已停止变化。
owning144项无skip通过，逐源码族去留见[35→21账本](cpu-final-f5-source-ledger.json)。
后续只迁移测试／完善说明和集中差异验收；若发现需要改变数学、正常准入或产品成本的新事实，
依法退回AUDITED，不能在候选状态暗改产品。Compute owner仍仅覆盖两私有RoPE文件删除。
首批普通consumer已编译通过；generic Graph有70项仍固定旧Native Neg／Abs／旧MatMul等的夹具FAIL。
这些按D2迁移到真实成熟调用；严格已知资源测试保留RMS／Add残差管线和原限额／OOM断言，
没有以成熟unknown资源冒充零字节。不是新增自研数学，也不启动P1。
本机完整普通CPU／Graph模型合同串行批、Graph夹具纠正、三平台source/static/shared与受影响mixed
交付差异仍待最终记录，未完成的项不签PASS、不关闭F5／goal。

### V1本机完整consumer与Graph夹具差异

Windows完整consumer批实际串行执行1903.641秒：Compute、CPU backend、oneDNN三个owning
consumer／region、ggml、Graph logical、普通CPU Direct完整批与CPU executor均通过。
普通Graph完整851项中849项通过、无skip；两项只因旧断言仍要求oneDNN Select而FAIL，
F4当前工程默认实为ATen。精度／原bits／独立invocation／变输入重放均通过；修正两处identity断言后
定向2项通过，未重跑其余849项，也未回写最初aggregate FAIL。

Generic Graph最初70项旧Native数学依赖FAIL；本批实际迁移到ggml／ATen，保留原数字oracle。
未依赖退役数学的66项继续用Native context；strict资源测试使用保留的RMS／Add残差管线，
已撤回MatMul／宽数学strict域不借成熟unknown私有量冒充0。
异步protocol夹具的测试backend不是CPU backend，保留真实Native RMS／Copy和具名Exp尾部；
不把ggml强接到不兼容backend。mock region仅验Graph调度／完成协议，不签成熟数学或完整路线资格。
原27项／随后1项迁移FAIL均保留；最终Generic Graph283项全PASS、无skip。
本轮没有产品数学修改／新的性能采样；三平台owning及source/static/shared／mixed差异仍待。

### V1 Windows三交付差异完成

source／搬移static／搬移shared分别实际运行43规则请求×两普通context和48数值／重放检查；
新增8项是普通两入口的odd-byte Abs／Neg／Multiply及稳定Softplus。static进程只允许CPU2时
43精确规则全部命中；unrestricted／shared按成熟工程默认，不外推WIN。
DLL邻exe、installed精简PATH及shared实际出口通过，共17成功步骤；四处consumer的
显式Operation包装遗漏造成的原编译FAIL保留，仅修测试consumer，产品未变。
[实际交付记录](../../../../evidence/correctness/cpu-final-f5/README.md)已归档。
MacBook Air144项owning全通过；其三交付与Linux／mixed差异仍在串行队列，不签未完成项。

### D5 GCC具名死代码残留：退回AUDITED

Linux GCC14当前Werror编译拒绝`indexing.cc:46 same_storage`：原Scatter删除后该匿名私有
函数无调用。Windows／Mac现有测试和三交付通过；本反例不是数学／资源缺口。
候选退回AUDITED，有限D5只删除这一未调用函数及同批编译若精确定位的其它清退残留；
不新增数学、放宽warnings、调整公共构建或延长性能预算。先提交本协议再删除。
原candidate507bad96与全部成功证据保留；新输入按未调用代码等价差异复用数学／43规则／已验交付，
Linux续编译和后续平台／mixed差异绑定新不可变输入，原GCC FAIL保留，不重复完整P1／模型长批。

D5同批第二个精确反例：`reduction.cc`的identity／accumulate／dense三个私有switch已只执行
Min/Max，公共ReductionKind仍含成熟库承载的Sum/Mean，GCC `-Wswitch`要求显式默认出口。
仅将原switch后的NaN／return出口放入default；准入仍先拒绝非Min/Max，不恢复Sum/Mean执行或改变算术。
D5第三个精确位置：`transformer.cc:44 coordinates`原RoPE／宽Softmax删除后无调用，
GCC Werror拒绝；仅删除该匿名未调用helper，保留F16 Bool Softmax实际坐标路径与axis/mask helpers。

### D5完成并重新固定候选

产品candidate `64b454fa23404d26bead60d495eda9475c802e32`：相对507bad96仅indexing／transformer
两个未调用私有helper删除，以及reduction原NaN／return出口移入default，Min/Max算术／准入不变。
GCC14完整Native owning构建和144项测试通过，无skip；三个原编译FAIL和摘要编码FAIL均保留。
新候选不需要新的数学／43规则／P1资格；原Win/Mac数学与三交付按该等价差异复用。
Linux三交付与四mixed envelope差异绑定新输入，尚待完成；仍不能关闭F5／goal。

### V1 CPU-only三平台交付全部完成

Mac本包20成功步骤、Linux18成功步骤分别完成源码／搬移static／搬移shared的43×两context与
48数值／重放检查，真实loader与shared出口通过；原消耗方SDK／cached producer复用，未下载新库。
Mac交付61.329秒，Linux交付216.922秒；Linux三私有文件等价D5差异绑定64b454fa，
Win/Mac原交付身份和Math证明按上述等价边界保留，不重跑同数学P1。
三平台owning各144项通过，无skip。当前待验只剩本包四mixed envelope的CPU差异与F5终审；
F6仍需逐原目标复核、文档／统一LOC与归档清理，不能提前关闭整个goal。

### V1 Windows mixed CPU差异完成

CPU+CUDA与CPU+Vulkan各15成功步骤完成source/static/shared；每个真实source／SDK消费者
执行43规则×两context和48数值／replay检查，qualified为false，未泄露CPU-only统计WIN。
DLL邻exe／精简PATH与shared实际出口通过；产品候选64b454fa，实际Windows输入8e8ee710仅多状态说明。
两个envelope原始receipt／log／consumer集中归档；没有GPU数学、格式或性能新资格。
Mac CPU+Apple及Linux CPU+CUDA、已声明Native独立头门仍待；F5保持CANDIDATE。

## F5 CLOSED：源码清退及适用差异验收完成

candidate `64b454fa23404d26bead60d495eda9475c802e32`，已在8e8ee710合法重新进入CANDIDATE；
其后只有证据／文档／状态，没有产品变化。全部本包有限D1–D5与V1完成。
[终审收据](../../../../evidence/correctness/cpu-final-f5/terminal-receipt.json)与
[头／impact／原日志及执行身份](../../../../evidence/correctness/cpu-final-f5/terminal-delta.zip)保存实际范围。
四mixed环境Win CUDA／Win Vulkan／Mac Apple／Linux CUDA分别15／15／20／18成功步骤，
全部真实source／搬移static/shared均执行43请求×两context及48数值／重放检查，无CPU-only WIN泄露。
三平台Native独立头门实际通过；没有重新编库签doc PASS，没有新P1。

| 七门 | 本包结论与证据边界 |
|---|---|
| scope | 固定104 GGUF／官方／非GGUF／MTP／DSpark不删，14冗余种物理清退，20有资格种＋独有Copy保留；SIMD/Mem／View与原借入程序保护 |
| architecture | 冷期按完整合同与适用硬件／软件／线程绑定，原资源／执行／完成及完整成熟region保留；无新执行器、公共API／layout或热搜索 |
| correctness | 三平台owning各144无skip；普通Direct832、Graph851按原849＋两处旧identity纠正组成，无skip；GenericGraph283及实际CPU executor／OOM／close／alias／checked-index／独立槽通过 |
| delivery | CPU-only三平台三交付与四mixed CPU接缝43／48格实际通过，搬移／offline缓存／DLL邻置／真实loader／exports及三平台独立Native头通过 |
| usability_docs | 清退账本与严格旧域撤回、具名RMS／Copy保留、普通成熟默认、性能WIN条件与未测限制明确，消费者不排序Provider |
| red_team | 原编译／数学fixture／身份／环境／编码FAIL全保留；mock仅验Graph协议，unknown不写0，不以不同dtype／格式或metadata冒充数学资格；具体GCC残留有限处理 |
| engineering_performance | P0删除冗余源码／无新数学优化、资源税或submit间接层；保留分支算术及43规则不变，F4当前限定证据复用，不扩n／AA／P1或挽留输家 |

Compute owner仅覆盖两私有RoPE文件删除，不重审runtime；Native本包CLOSED，其他CPU／GPU／公共Graph旧状态保持。
实现Compute＋Graph **60,161行／249文件**，不含测试／vendor／bench；相对F4清退949行、2文件。
这不是整个CPU goal的最终通知：F6逐原目标终审、公开完成口径和归档产物清理仍待。
reopen：新的可复现正确性／生命周期／编译／交付／能力缺失反例，或已资格条件误绑定；
不因旧卡、doc提交、未知硬件无统计WIN或“再看一次”重开。
