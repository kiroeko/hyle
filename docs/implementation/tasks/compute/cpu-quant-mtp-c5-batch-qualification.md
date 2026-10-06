# C5：CPU 全批验收、文档与停止

<!-- MODULE REVIEW: AUDITED -->

## 计划身份

- 状态：AUDITED，真实角色/几何与既有承载已对账，固定下方有限F1–F4；尚未签本包执行PASS。
- baseline：`400927ff7af0b150432b830ce47b3aa7ac5fd3c9`；SCOPED提交`39ea36847d679900537375eefb2b7ac9368e9002`。
  Windows实际HEAD与干净工作树已核对，保留C1–C4全部后续提交。
- owner范围：compute_ops、compute_provider_ggml、compute_provider_native；以本卡交接其AUDITED记录，
  不伪造AUDITED→SCOPED状态跳转或重开其它CLOSED/BLOCKED。
  CPU/Graph facade、公共头与交付直接消费者按实际未变身份复用；没有新增产品owner。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- scope：全部C1准入的104档位对账结果、3sidecar、2新官版及非GGUF已准入对象，两入口、资源与两系统三交付。
- 不新增算子/Provider/优化或第一次接线；只合并真实证据并补缺验收格。
- 新版当前能力表与旧22画像/P1历史分开，不能覆盖旧失败、被拒域或缩小范围后冒称完成。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| V1 | Metadata驱动完整CPU验收矩阵：匹配实际角色/格式/state/全部required region | 每格普通Direct/Graph实际输出/完成/失败通过；未运行/拒绝/数值失败不计完成 |
| V2 | Windows/Linux owning+headers、source/static/shared、DLL/loader/relocation/offline | 候选/source/toolchain/SDK身份匹配，复用理由明确，无全模型/未知硬件外推 |
| V3 | P0与已声明的唯一代表性成本证据汇总；用户CPU构建/default规则与文档 | 没新P1主张不强制竞赛；必要未验项不伪造PASS，旧Native规则不越过peer版本 |
| V4 | closure/docs/selector reasons、统一LOC、精确owner临时目录清理、skip-ci提交推送 | 最后红队无未决必需风险后ownerCLOSED；完成即停止 |

## 验证、预算与停止

- API/算法稳定后一个不可变验收候选；复用已完成包的匹配身份，不因docs/state重建或重跑P1。
- 只核对当前已准入scope，不扩大新发布/其它路线；无必要不重复owner suite。
- 存在未裁决非GGUF范围或任何必需拒绝/失败时保留具体阻塞；不得以有结果等于全PASS。
- 成功后明确报告CPU量化与MTP扩覆盖完成、限制和全部实现LOC，停止等待下一阶段；不开新goal/算子大全。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。

## 用户最新性能口径

单一合格成熟carrier完成正确接入与P0；多家成熟库具备同合同重叠域时按总计划批量比较固定默认规则。
不比较自研，不为每个算子/档位重建或计时，最多两个代表消费族共用一份稳定候选P1。

## 本包定界与审计输入

- C1完整header/index为准：104/104 GGUF主档位、826/826 shard/sidecar、152/152 Safetensors shard。
  存储表有25种实际GGUF类型；I32 hash/索引表按整数控制端口处理，不能当MatMul权重。
- C3/C4真实收据为执行输入：11新增GGUF及原格式、四artifact decode、三MLX格式，
  新官方主干/三阶段、三sidecar的普通source两入口；两系统安装SDK40项与资源20项。
  复用必须匹配产品/软件身份、语义、实际角色/shape/state及入口；头完整不等于RUN_PASS。
- 对全部角色按实际算子和端口去重，记录制品到资格的映射；同格式不同真实消费合同不能合并。
  保存尚未执行的格，先把有限差异固定为FIX，再一次参数化接验证消费者；不新增数学核。
- 源码/公开API/Provider没有本包升级授权；若发现产品接缝缺陷，先在V1–V4边界明确反例/FIX与owner。
  新数学或必须改变长期合同的缺口须具名定界，不能用自研或缩小必需集合补齐PASS。
- 每个系统一个稳定候选批次，构建j2、测试j1；只补缺失适用格。已有SDK与资源身份相同则复用，
  不逐档位构建、重复三交付或重跑性能。最终范围裁决、用户文档与红队属于本卡，不跳CLOSED。

## 有限审计与固定FIX

[实际需求收据](../../../../evidence/correctness/cpu-quant-mtp-c5-demand-audit/README.md)保存全部制品到合同的关联。
现有原ggml、oneDNN与原ATen/codec组合能表达已查合同；没有新数学核或Provider升级授权。
BF16原F32输出使用既有bfloat_linear_f32/grouped语义；F16原乘法可由显式cold权重cast及activation
原F16舍入后F32成熟MatMul表达，转换成本必须记录，不伪装为原packed热程序或性能优势。

| FIX | 当前有限差异 | 本包动作与结束证据 |
|---|---|---|
| F1 | 新制品缺完整typed角色/几何运行资格 | 参数化559类GGUF权重/lookup合同，真实K/N、原codec/reference、changed replay与indexed失败；保留104身份映射 |
| F2 | 非GGUF小夹具不能代签真实内宽 | 参数化34类实际packed/scale/zero/bias合同；保留原dtype/group、显式cold decode消费与selected group；不全库专家热展开 |
| F3 | 参数/版本身份不能仅靠dtype推断完整region | 30类非线性/卷积参数及全部准入family required regions逐项映射；补未验形状/组合与精确身份，C4 branch复用不越界 |
| F4 | 全域聚合、文档/终审仍未裁决 | 两系统同产品因子/普通入口批次与匹配三交付/资源收据；P0和具名适用P1、七门、LOC、清理、逐状态关闭 |

预算：一份稳定候选集中构建/验收，一批定向测试/接缝纠正；新数学/算法优化0、逐格式性能竞赛0。
只有真实失败反例可以在F1–F4内新增产品接缝FIX，先写明影响owner/合同再改产品；不能默默扩大到新模型或新API。
执行consumer是测试/示例组合，允许使用原库codec作为参考，不是新的产品模型loader或推理引擎。

### F1.a：Q8_0 大词表的真实准入缺陷

候选`f618b5178fb16fa9a1c84b815a2bb51922ab4bf8`的Windows两入口各558/560通过。
`g9d504e80cbfefb7f`是实际Qwen2.4T Q8_0 embedding [248320,8192]，元素数仍小于INT32_MAX，
总编码字节约2.16GB却被ggml CPU representable的INT32_MAX总字节上界拒绝。
原固定core的get_rows_q按int64行号与size_t byte stride定位，线程分块仍有int调度变量；
HYLE已保留元素数、维度、stride、输出及checked索引边界，原库无需新数学或新接口。

本FIX只扩CPU-only新版envelope的native-encoded Gather第一个输入总字节准入；
维度/元素数/stride及其它operation边界保持。owner为compute_provider_ggml，当前AUDITED授权。
真实全跨度末端/重复行、changed replay和旧有限矩阵用于验证，GPU/旧core资格不扩大。
产品shim变更使旧安装SDK无法覆盖本格；C5须重新安装两系统static/shared并验证此真实大表，
不能仅引用C4旧SDK的代表入口PASS。
另一失败`g2fcad3816188464a`是测试误加I32→F32 Cast；实际hash/专家ids保留I32，
按既有host checker读取整数，不修改产品Cast能力或放宽数值门。

### F1/F2 第一批实际执行

候选`46a55325a465a94da3d79d2e5e92b0cb6b2cfb62`的Windows/Linux普通Direct/Graph各594项通过：
559实际GGUF合同、104精确身份的真实最小projection旅程，以及34实际非GGUF合同；
全部本批indexed消费执行越界/写保护、slot退役及新slot恢复。
[准确收据](../../../../evidence/correctness/cpu-quant-mtp-c5-matrix/README.md)不把2,413需求关联冒称完整混合模型运行。
F3完整required-region与30类参数对账、F4更新SDK和最终聚合仍未完成；owner保持AUDITED。
首次故障注入写4字节违反Buffer完整视图合同，修正为完整索引payload；没有产品错误/数值门放宽。

### F3/F4 第二批与有限存储接缝

[更新SDK](../../../../evidence/correctness/cpu-quant-mtp-c5-sdk/README.md)：两系统static/shared普通两入口64项通过，
包含修复后的2.16GB Q8_0真实表；其余包装/资产/缓存合同按未变身份复用C3/C4。
[源代码消费者](../../../../evidence/correctness/cpu-quant-mtp-c5-final/README.md)：Windows剩余两入口362项完成，
首次14项fixture环境skip保留，设置已校验的本地fixture后全部补跑通过。Linux同批Direct172/Graph190全部通过，
没有skip；两系统共724项。产品/测试身份与原数值矩阵匹配，没有重复执行594项/入口。

[参数精确对账](../../../../evidence/correctness/cpu-quant-mtp-c5-parameters/README.md)把30类shape/storage细分为
80类真实role/shape、1,354制品关联，不是80个新算子。F3发现旧完整消费者已经归一化卷积权重，
不足以证明原GGUF singleton物理形状接入。固定本批有限补验：六个实际卷积形状经公共View/Concat
进入原完整GDN/KDA消费者；独立oracle、实际state宽度及反馈不变，没有新增数学或产品改动。
这六项已在两系统普通两入口通过；其余参数完整用途/required-region仍须裁决，owner保持AUDITED。

F3另一个明确复用边界：C4新版主干签的是官方FP4/FP8/BF16程序，不能代签新版GGUF主干的F32 state/math。
固定完整header显示Pro-0813与Flash-0731各UD-Q4_K_XL/UD-Q8_K_XL的专家均MXFP4、router均BF16，
shared与普通投影分别Q8_0/BF16，q-rank为1536/1024，H为7168/4096，E为384/256。
既有原GGUF程序可完整承载。将这四个精确身份纳入同一F3批次：ratio0/4/128 attention/cache、
indexer compressor、ordinary/hash MoE、mHC后真实全词表head组合。只复用原provider/recipe，
原独立oracle、实际内宽、完整vocab跨度及state反馈保留；其它档位实际格式合同仍回指F1矩阵。
没有新增产品数学、公开API或性能赢家。这四项已在两系统普通两入口通过。
准确候选及40项收据见上述参数对账目录；两平台只差docs/evidence，产品/测试源匹配。
全词表head实际消费mHC产品输出，预期来自完整独立mHC oracle和原codec，不自我对照；误差门未改。
F3仍须完成80条角色与全部required-region最终关联，F4最终红队/状态/清理仍未关闭。

### F3有界反例：Qwen2.4T实际RoPE核心

当前旧`sem::attention`用交错RoPE、`1/(i+1)`频率与默认norm epsilon。
固定C1第一GGUF shard metadata为qwen35moe、rotary64、theta10000000、rms_eps1e-6、MRoPE sections11/11/10/0；
已登记Transformers固定源码的rotate_half/apply_rotary_pos_emb使用前后半区，原llama qwen35moe路径同属MRoPE。
文本三轴position相同可用HalfSplit表达；旧交错夹具不能签这个模型的R04真实核心资格。

固定有限补验：CPU-only两份精确C1制品身份（Q8_0/UD-IQ1_M）的F32 Q/K norm、HalfSplit64、
theta1e7、两行KV历史与四次changed replay/反馈、sigmoid gate。只组合既有成熟操作；
不改产品数学、旧GPU测试/资格或误差门。投影原格式/真实内宽仍引用F1实际矩阵。
本格尚未运行，旧交错测试保留为该布局的历史事实，不改写旧PASS来代签目标模型。

后续CPU-only实际HalfSplit/频率/epsilon与KV反馈已两系统两入口通过；旧交错历史结果不代签。

同一参数审计发现旧Qwen GGUF FFN与2.4T MTP的部分RmsNorm使用API默认1e-5，
不能代签固定模型1e-6。有限补验合并为一个CPU-only批次：H2048/5120/8192的F32 norm/残差，
以及2.4T真实Q8_0 eh_proj的embedding-first双norm/Concat/MatMul/后norm与四次carry反馈。
后者保留原Q8 activation codec/vec-dot参考、真实K16384/N8192；不以F32点积替代。
这只补实际消费者资格，不改API默认值、产品kernel或旧GPU资格。

首轮`911efc5c7e10`的Windows规范norm/RoPE两项通过，MTP反馈FAIL。
源码确认新MTP夹具漏了`.quantized=true`；原`sem::mtp`及完整GGUF反馈均按已冻结量化片段门，
原Q8 activation codec对上游浮点差异不是光滑映射，不能套非量化F32片段的门。
修正测试分类，复用已有量化片段门，不新设阈值；三真实宽度的norm仍独立按严格F32门。
保留失败raw/XML（可无损gzip），不改产品数学或原codec/reference，随后两入口补跑。

最终候选`a68f8ed7a5ea`两系统普通两入口各3项，共12项通过。准确作用域、失败归档与精度合同见
[实际Qwen资格](../../../../evidence/correctness/cpu-quant-mtp-c5-qwen-actual/README.md)。
本MTP程序只签输入/投影/carry边界，不代签完整NextN decoder；required-region最终聚合须保留该区别。
80条参数已生成源码/SHA/两系统收据索引，仍不以testcase名称匹配代替完整语义复核。

### F3完整NextN有限补验

固定原Unsloth llama `46cbf0e95786`的qwen35moe.cpp（SHA
`e449e79acb469d122addd9a9aaa2886fc99c13c3b5826877f84e9c6ab38b25bd`）确认NextN为
embedding-first双norm/eh_proj→完整GQA/cache→attention residual→post norm→routed及gated shared MoE
→FFN residual→shared-head norm→h_nextn/full vocabulary head。
C1的2.4T blk.92 Q8_0与UD-IQ1_M真实格式分别为Q8和Q5 attention、Q8和Q2 expert、
Q8和Q5/Q6 shared、F32 router/norm、Q8 eh_proj，主embedding/head为Q8与Q4_K。

固定同一批两个完整CPU-onlyNextN消费者，两入口/两系统：H8192、Q64/KV4/D256、
expert512/top10/I2048/shared2048、完整V248320，四次carry/cache反馈；普通成熟操作和原codec/reference。
按计划降低驻留expert到8，用显式I32映射表达合成重复expert权重；router仍完整512，K/N与head跨度不缩小。
没有新产品API/kernel或性能比较。此完整NextN资格尚未运行，输入边界PASS不能代签。

`61155eed2a73`完整NextN在两系统普通两入口通过；没有失败、skip或数值门变化。
required-region对账另确认旧2B GQA夹具虽为HalfSplit，却旋转完整256并用默认epsilon；
同一CPU-only实际GQA消费者按固定配置统一覆盖Q8/KV2、Q24/KV4、Q64/KV4（三者D256、rotary64），
保留theta1e7/epsilon1e-6与KV反馈。只补三个真实几何，无新kernel、GPU改动或通用API默认值修改。

后续`c1b0250ed4bb`两系统普通两入口各4项通过：三实际GQA几何、三宽度严格norm、
输入Q8 MTP与两完整NextN程序。详见实际Qwen目录expanded收据；Windows wrapper数量预期由3改4，
Direct4已通过，只补未执行Graph，没有重复build/Direct或改产品。
80条参数的有限源码/合同复核已完成，见[复核表](../../../../evidence/correctness/cpu-quant-mtp-c5-parameters/contract-review.md)。
八模型族/104档位required-region收据已建[聚合索引](../../../../evidence/correctness/cpu-quant-mtp-c5-required/README.md)，
仍须完成跨证据完整作用域裁决、两新官版/三sidecar/四非GGUF及资源/三交付/七门终审；不提前签goal完成。

### F2/F3非GGUF未压缩text合同与有限成熟接缝

聚合确认四制品未压缩text参数不是GGUF的F32域。按kind/dtype/shape去重45合同：
13 projection、10 norm、3 conv、2 head、3 embedding、3 position lookup、3 log-decay、3 time-bias，
以及HC/sink/router/I32 lookup。963个视觉参数仍在既定语言范围外；487个packing companion由F2已有端口证明。
固定同一参数化批次，实际原dtype/完整K/N、词表跨度与changed replay；不按tensor逐个建target。

真实源码反例：公开CausalConv1d可推断F16/BF16/F32，但CPU ggml仅接F32，当前ATen/native源码没有低dtype
成熟承载。C1需要BF16[6144,1,4]、BF16/F16[10240,1,4]，不能把F32 conv PASS直接代签。
固定产品FIX owner compute_provider_native（当前AUDITED）：现有Torch CPU program增加窄F16/BF16 conv接缝，
只调用原ATen conv1d，布局permute/reshape，明确F32 accumulation后回原output dtype；无新数学核。
使用原managed runtime/accepted failure/slot退役/metadata回收，转换/原库临时为真实热成本，未知私有上界不写0。
原F32 ggml路径不变，输入元素/原F32临时int64容量检查；不添加全局线程、JIT搜索或Provider排序。
新原库程序采用追加的私有implementation ID，既有ID不重排。新增45合同及原资源边界、两系统更新SDK
须完成资格后才签PASS；该dtype只有此具名成熟承载，无新赢家P1或自研对比。
