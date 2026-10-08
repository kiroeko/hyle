# Apple MLX/Metal：覆盖、接线、默认优选与清退收口

<!-- MODULE REVIEW: AUDITED -->

baseline `406cc3c23843ea9928e45970527ed2664f8f2728`，Windows工作树干净。
2026-10-08用户要求Apple按已完成CPU/CUDA相同原则开始。产品真相仅Windows；Apple副本可丢弃。
此页先是协调只读审计卡，不把旧AUDITED记录自动变成新scope产品授权。写产品前按实际owner
重新审计并独立提交有限FIX；旧预算、失败、CLOSED/BLOCKED不自动重置。

## 四个有限包

| 包 | 权重 | 完成条件 |
|---|---:|---|
| A1 当前目录/成熟承载/接线与构建审计 | 25% | 全104 GGUF/559合同、45plain/34packed、八族及官方更新/MTP/DSpark映射，真实来源、数值/资源/成本与有限FIX |
| A2 批量成熟接线与正确性 | 35% | 成熟库优先；按实际端口/编码一次批量接线后集中准入、完整输出/状态与资源验收，单承载不造竞赛 |
| A3 默认、有限best-of及自研去留 | 25% | 真实重叠且同精度/转换/同步/融合/完整终态的一次有预算比较；有条件规则及未知条件成熟默认，冗余自研物理清退 |
| A4 三交付与终审清理 | 15% | 当前Air source/static/shared/relocation/metallib/dylib、两普通入口、失配/关闭/失败、docs/closure/LOC/垃圾清理、推送后停止 |

## 需求与边界

需求仅来自[共同模型目录](../../../architecture/model-compute-catalogue.md)及
[逐制品端口](cpu-final-demand-carriers.json)，与CPU/CUDA对称，不重新研究模型/下载巨型权重。
包含DeepSeek Pro0813/Flash0731、官方混合FP4/FP8、主流已登记量化与必要MTP；不能以旧
Apple40/44或曾经的M4性能机证书代签完整新目录。HYLE不实现loader、整模型engine、CPU+GPU
模型切分、HIP或分布式；GPU的host控制/准备/同步和同GPU不同成熟程序组合仍在范围。

第一候选是原MLX GPU程序与ggml Metal codec/完整region，先查具体原版能力及接线成本；
不按Provider名字假定覆盖，不因漏接就新增HYLE算子。所有现有自研数学冻结，成熟接入缺口
排除前不扩张/优化/研究自研。确无成熟同合同承载时列具体缺口/取舍，不伪PASS。
MLX compilation/fusion、ggml原管理、MLX arrays/原生完成和Metal buffers保持，不引入
万能Queue/executor或逐kernel vtable。资源封装、语义边界修复和成熟组合不误删为自研。

## 当前真实事实

Air连接 `ssh kiro@172.22.10.198` 默认身份；2026-10-08实际为Mac16,12/Apple M4/16GiB，
macOS27.0.1 build26A434、AppleClang21.0.0、CMake4.3.4、SDK27.0，交流电，剩余约43GiB。
设备/OS/工具链与旧证书不完全相同，旧performance只保留历史，资格须重新匹配。
当前remote `~/hyle` HEAD为1bf0b556、工作树干净；产品源码不得从它反向复制。
当前route/graph owners为旧D5-F45 AUDITED，ggml Metal为旧CPU enum迁移AUDITED。
源码 `native_quantized_type`只接10种GGUF类型，default明确decline新增CPU格式；
`apple_mature_float`明确拒绝FP4小scale roundtrip。先静态逐合同映射，不把这些当新数值FAIL。

## 有界执行

A1：一批需求归并、一批原成熟源码检出、一批接线/构建审计，各具体问题最多一次follow-up；
首小时产出静态映射与有限FIX，不翻遍历史任务。A2按能力域批量接线/延迟验证，source低并行2、
测试1、重型串行；缓存复用，Mac磁盘不够先清拥有者精确临时路径，不删source/模型/共享cache。
只同步完整Git候选/archive并记录身份，禁止手选远端产品文件反覆盖Windows。
配置步骤只以CMake/参数/工具链为输入；普通代码变化不重新配置/重复全集。具名实现失败最多
一次定向纠正，再按协议裁决，不换目录/ID重置预算，不重写公共治理/仪器。

A3在实现/接口固定的不可变候选上用统一hyle-perf，5% material margin、固定n15，
Apple实际QoS/交流电/low-power/机器身份收费；不冒称P核hard pin，不升31。不合格AA项不进
正式性能，近似/噪声直接选成熟。只导入完整候选赢家，不把单kernel时间代替终态成本，
不拆成熟融合来竞速。不无限优化保自研；真正unique资源/语义具名保留，有完整成熟替代且
无完整优势的自研清退。SIMD本身不动。未测硬件/版本仍有合法成熟默认，不让用户排rank/timing。

## 预声明验证和停止

只读A1仅输入SHA/映射完备性、docs/closure/selector与统一LOC，不build/P1。
后续按有限FIX选择原算子/region、资源/lease/Completion/失败、两slot/变输入、index/encoding
边界及当前Air三交付；未变域按source/合同/工具链/机器匹配复用，不复制CPU/CUDA PASS。
Windows/Linux Apple-OFF独立公开头/构建合同按实际改动选择，不制造Apple binary journey。
每包完成/重大scope变化报告实现/验证/未完/阻塞与Compute+Graph全实现LOC，文档/状态不build/P1。
七门全部适用证据成立后关闭本轮并停止；其它路线/公共Graph旧Vulkan blocker保持。
推送采用[skip ci]，不使用Actions。2026-10-08用户明确选择建立独立持续goal，现已创建；
A1已实际产出[559合同静态映射](../../../../evidence/correctness/apple-parity-a1/gguf-static-binding.json)。
其中127合同/11类型缺ggmlMetal类型绑定，432仅静态有映射，全部runtime仍UNVERIFIED；
不把这一统计当Provider实际支持结论。下一有限动作检出现有成熟Metal codecs与完整操作的
适用精度/资源后固定批量FIX，所有自研数学冻结。

## A1 审计与有限 FIX（产品前独立提交）

[carrier/region映射](../../../../evidence/correctness/apple-parity-a1/carrier-region-map.json)覆盖
45plain/34packed/八族49关联及全部官方更新/sidecar。34中16原MLX affine/MX4/MX8请求
没有MlxQuantized接线；18个NF4/AWQ/CT没有ArtifactWeightDecode接线。这里只确认源级缺口，
不是runtime支持。先接原MLX公开quantized_matmul/gather_qmm/dequantize/take与已有Metal
dispatch，所有实际精度/scale/状态和终态资格仍由A2验收。119合同的8种漏接类型已有原
Metal分派；余8合同的IQ1_XS/XXS/XXXS须查已登记成熟来源，不能从新enum猜测GPU存在。

| FIX | 有限允许变化 | 验收边界 |
|---|---|---|
| F1 | ggml Metal只接8种已存在原分派的codec；embedding/activation逻辑精度按原支持扩接，不改vendor数学 | 对应原encoded bytes/数值/索引/别名/终态，不能只prepare |
| F2 | MLX自有16类packed，原公开qmm/gather_qmm/解码/行选择；InputSpec补必要U8/U32 carrier，共同Direct/Graph使用同一个冷lowering | 原group/mode/nibble、F32累加/最终dtype、选中专家/IDs/动态数据和Completion；不伪跨format |
| F3 | NF4/AWQ/CT及IQ1扩展只在具名成熟source/API可完整承载后借入，官方FP4严格小scale独立资格 | 无完整成熟证明仍UNVERIFIED，新增self不授权；失败具体定界 |
| F4 | 当前Apple默认冷绑定/条件、完整region边界及重复self去留，先成本/完整候选，再一次稳定批次P1 | 未匹配成熟默认；不得计时/热重选，不拆MLX/ggml成熟融合 |
| F5 | owning target三交付/metallib/dylib/离线与失配、文档/LOC/清理 | 批量实现后集中当前Air验证，精确具名失败仅一次纠正 |

前三个现有AUDITED owner依法重新绑定此有限scope，原F45/F46/CPU enum失败和预算保留；
旧范围的[F45记录](llm-moe-d5-portable-float-native-qualification.md)与
[CPU enum迁移记录](cpu-quant-mtp-c3-format-bindings.md)仅作历史证据，不恢复旧施工队列。
它们不是CLOSED reopen，不伪造AUDITED→SCOPED或BLOCKED。公共API/数据/执行仍原Apple
native形状，F1/F2是成熟接线，F3原生资格前不写数学。各有限source疑问一轮follow-up即裁决。

## A2 第一批真实验证

`d4f808614` 经完整Git bundle正向同步到Air，复用producer一次configure/parallel2构建，
三owning test target编译通过。[实测记录](../../../../evidence/correctness/apple-parity-a2/README.md)：
Direct23项中21 PASS/1旧fixture SKIP/1新增packed FAIL；Graph20项19 PASS/1旧fixture SKIP。
新增packed十二组合被旧admission guard拒绝，尚未进入原MLX数值执行；这是具名HYLE接线缺陷。
有限纠正只补该准入条件，Metal测试把8种原码从旧decline迁至完整MatMul replay，并使原
GetRows可承载当前全部已绑定native codec。修复后数字/状态待验，不把编译、旧通过或SKIP签
新增能力PASS。原失败保留；没有启动P1或重跑全部模型。

`02a573d3`增量构建通过，Metal8/8测试通过，含18codec真实MatMul/GetRows、replay/索引/资源。
MLX线性/embedding实际通过，六grouped组合另发现原`indices_or_default`遗漏stream参数的
I32→U32转换；接入处显式owned-stream cast，使原内部cast成为no-op，隔离检查不放宽。
此新具名问题仅一次定向纠正；原admission FAIL保留且已经解决。两入口用同一独立oracle验证
不同专家scale/变IDs/变activation和embedding，下一候选结果待验。18codec微型F32通过不
冒充559完整合同；官方FP4/18artifact/8IQ1扩展与默认/交付仍未完成。Windows Apple-OFF
精确factor configure通过，无SDK下载/探测，不代签Apple runtime。

`125b2209`增量构建通过；Direct新增2/2测试通过（含全部十二格式/精度/线性/grouped组合、
双embedding精度、双replay、不同专家/变IDs/变activation与失败）。Graph新测试2/2被第二处
旧custom准入名单拒绝，尚未数值执行。F2的具名定向纠正将Direct lower、Graph预检/trace的
三份名单统一为Compute-owned native predicate；仅detail seam，无数学/stream/资源/公开
用户入口改变。适用头/seam验收随selector扩展，Graph修复结果待验，原FAIL保留。

`58c49dae9`增量构建和Direct2/2、Graph2/2全部通过，原codec18种数值/资源资格仍对应
`02a573d3`未漂移实现。此首批十二packed组合+双embedding精度的三处准入/stream接线缺陷
已解决，原FAIL不抹除，不签全目录。Windows Apple-OFF头2/2通过，用户报告本机CPU峰值后
确认本地检查已结束、无HYLE编译/测试残留；后续Apple build/数值/P1只放Air，不并跑本机
编译。Native头/三交付selector门集中A4，尚未PASS。统一Compute+Graph实现61,190行/
258文件，不含tests/vendor。剩余559真实合同/16实际MLX几何/18artifact/8IQ1扩展、当前
完整模型/官方/MTP链、默认/有限优选/三交付都未被此首批结果代签。

## F3 首批 source/API 定界（产品前固定）

baseline `3b4e7a53`。18个NF4/AWQ/CT请求的数学来源沿用CPU/CUDA已登记的原tensor程序：
bitsandbytes `833649043474` 的default `_dequantize_4bit_compute`/nested block256；AutoAWQ
`88e4c76b2075` 的`unpack_awq`/八列reverse/GEMM zero与原scale dtype；compressed-tensors
`ac8e2ba82f4e` 的complete symmetric4原offset8/scale dtype程序。MLX公开GPU take、bitwise、
reshape、repeat、transpose、multiply与astype完整承载这些tensor步骤；只翻译原程序，不新增
HYLE codebook、量化算法、GEMM或shader。F32 NF4 lookup/nested offset、AWQ high-column
permutation/真正zeros、CT先cast到scale dtype的存储边界须保留，最终override另作cast。

有限实现：各原source独立借入MLX carrier目录，沿中央lock/owning Apple target登记版本、
许可证与三交付；只新增原`ArtifactWeightDecode`的native lowering，并由现有共同predicate
同时连接两入口。原API/schema不扩张，CPU/CUDA源与资格不变，Apple-OFF不消费这些源码。
冷decode生成明确[N,K]常量，MLX持有原临时数组/输出，不缓存可变payload，不按文件名猜format。
实际完整矩阵/两入口/replay/rounding/失败/OOM与准备后MatMul消费一次批量验收，尚未runtime
PASS，也不先与自研比较。各具名新失败仅一次source纠正，不换ID/目录重置预算。

本批同时把34packed/559GGUF的共享**需求数据与oracle**从CPU扩展宏耦合中解开，Apple只
消费真实存在的vendor enum；缺失的IQ1_XS/XXS/XXXS使用明确不可用标记并冷拒绝，绝不定义
假的Metal capability宏或运行COUNT类型的oracle。原Unsloth固定版本的Metal分派也没有三类
native case，继续具名成熟缺口调查；本批不签这8合同或旧官方FP4资格，不写对应新自研数学。

F3首批`516cff801`只在dependency hash审计失败（本地WindowsPath排序与已有portable字符串
排序不同）；一次元数据纠正`1db9ad96`后三新锁值匹配，无公共工具改写/原依赖变更。
Air增量构建通过，[34packed两入口记录](../../../../evidence/correctness/apple-parity-a2/README.md)
全部通过：原格式/实际N/K/完整cold decoded矩阵/后续MatMul/双replay与适用坏索引。
MoE明确原始E与已验E<=8，保留共享样本口径，不代签巨型256专家payload驻留。Input13/13、
selector59/59工具测试通过；宽selector仅因新ownership/dependency记录，保留计划并按当前
owner/实际影响选择，不触发其它CLOSED模块全集。三交付/license/offline集中A4，未PASS。
统一Compute+Graph实现61,241行/260文件（不含tests/vendor）；没有新增self kernel或P1。

## 当前559 GGUF批次与下一具名纠正

`efa1ef48`两入口各559项：544 PASS/7 FAIL/8 IQ1扩展SKIP，Direct116.212s、Graph101.567s。
[原压缩日志/XML/逐项摘要](../../../../evidence/correctness/apple-parity-a2/README.md)保留，
不能将SKIP计数或仅静态枚举当覆盖。共有两项具名问题，定向纠正后只重跑对应失败ID：

1. Q8_0 N248320/K8192 embedding/head原encoded bytes>INT32_MAX；夹具直接cast flat U8
   shape导致负值，产品input_spec也有同一1-D cap。F1追加有限carrier FIX：小payload维持
   原形状，大payload按完整encoded row映射checked二维U8，各axis在native int内；真实byte
   count/row-contiguous import/逻辑dtype/原ggml program不变，两入口统一从input_spec上传。
   当axis仍不可表达才OutOfRange；不截数据、不偷偷解成F16、更不做CPU模型fallback。
2. 五项Q4_K embedding精确输出FAIL。原Metal函数高nibble使用half `d/16.h`，CPU原函数
   先F32 widen，scale精度损失与错误位置一致；当前正式ggml原源码仍有这个表达式。需具名
   成熟精确承载，不修改容差，不在F1自造shader或数学补丁。此项尚未有合格替换。

另定位Anemll原Metal IQ1_XXXS候选`a809594fd2035979ad2b7ef23398127c1d07e891`，限定查实际
encoding/源码/API/许可证及另外两种IQ1，未准入，不扩研究面。旧官方FP4两边界/预算不变。
A2与whole goal仍未完成；已通过域不因上述问题重跑全集或P1。

大载体纠正`33de7d258`已经完成：Air owning build通过，只复验两项失败Q8_0ID及carrier
metadata边界；Direct3/3、Graph2/2通过，原2,161,377,280字节用U8[248320,8704]按完整
encoded rows承载，原packed bytes/native program不变，未重跑559/P1。结合不受改动影响
的旧通过项，当前两入口各546 numeric qualified/5 Q4_K精确失败/8 IQ1 skip；这是证据复用
与delta合并，不冒称新全集。34packed域不变。统一实现61,252行/260文件，不含tests/vendor。
阶段仍A2；Q4_K/窄IQ1/官方FP4、45plain与新官方/MTP完整片段、默认/优选/三交付未完成。

新增[成熟codec定向卡](apple-mature-codec-borrow.md)只重开Apple实际所需的runtime enum/core
接线；原CPU/CUDA/Vulkan状态不变。原Anemll源码已确认三IQ1的MV及grouped wrappers，
source/grids/ABI审计后再固定实现，不以README签GPU能力。Q4_K另借原CPU数学到MLX所选行。

当前A2新增实测：45plain两入口各45/45通过；五Q4_K精确失败在原source的MLX所选行承载
下两入口5/5通过；原Anemll IQ1在实际8个合同两入口8/8通过。结合原通过和两大载体delta，
三端口矩阵559/34/45已有成功数值/replay样本证据，限定实N/K/全输出、MoE E<=8、plain head
全行4096-row tiling，不冒充整模型驻留。新core尚需对应provider/resource回归与三交付。
[完整记录](../../../../evidence/correctness/apple-parity-a2/README.md)保留旧FAIL、SKIP与新delta，
没有全集/P1反复启动。61,328实现行/261文件，不含tests/vendor。纠正表述：34/45 runner
不自动使用invalid_index_input，坏索引暂复用同域小测试；新所选行及真实几何资源负例仍待。
完整family/新官方/MTP片段、官方FP4、默认/有限优选与A4交付仍待，未关闭A2/goal。

FP4新增[原DS4 Metal source资格](apple-ds4-fp4-source-qualification.md)：已有CPU来源对应的
原GPU group32 pass实际包含floor，配合其原host Safe math具名设置检验tiny责任边界。
只一批原source probe，产品FIX零，不恢复F46旧预算/命令；不换容差/模型数学或全局设置。

新所选行资源Direct10/10、Graph2/2通过（坏ID/空ID/failed slot恢复/保留结果/close/OOM/
晚到完成/取消），没有凭invalid_index_input元数据签PASS。既有region factor一批Direct178
PASS/7真实输入SKIP、Graph149 PASS/7 SKIP/1旧provider-name断言FAIL；Windows已有真实
117MB Qwen子集的25个hash复核后完整同步Air，给出fixture env，仅补七SKIP与一个诊断FAIL。
Direct7/7、Graph8/8通过，无SKIP，合并185/185与157/157（不冒称第二次whole factor run）。
原GPU数学没改，Q4旧松门反而收紧。新官方0813/0731与DSpark indexed完整region仍须对应
witness，旧名字不代签。FP4原DS4候选实测222 mismatch/190非零变零，已请求按goal冻结规则
决定必要GPU数值保真桥接；答复前产品FIX零。A2未关闭，A3/P1与A4交付仍待。

下一批固定九项/入口：官方Pro0813、Flash0731各target-hidden/confidence、全词表Markov、
128/129-token prefill与五token注意力/cache六次replay；Qwen Q4命名sidecar的真实Q3/Q4/Q6
embedding/decoder/完整词表MTP；Flash Q8与BF16 sidecar各原MXFP4专家与三stage DSpark、
共享embedding/cache/confidence/完整词表head。均复用共同C4模型形状、制品revision、oracle
和whole-fragment门，不增加画像、格式或产品数学。量化dot integration oracle调用独立原
ggml Metal backend并保持物理权重/F32 accumulation；不能借CPU vec-dot不同activation
codec冒充Metal oracle。primitive codec/equation证据继续保留，原backend不作为产品fallback。
两入口测试按fixture.phase绑定真实prefill/decode；本批只增测试，尚未PASS。仅新九项集中
build2/test1，既有559/34/45和185/157证据不重跑。官方FP4 main/完整官方DSpark专家依赖
待决精确保真方案，未被本批侧车MXFP4或FP8 attention替代。

`b229512d`新九项两入口均7 PASS/2 FAIL，Qwen与Flash两sidecar完整MTP全部通过。具名
测试接线漏传output ordinal，导致共享C4的inverse-RoPE检查未生效；`6aa5ca5f`只纠正
这项并复跑两个失败ID，仍有残余。原结果保留；不将正确启用checker签数值PASS。
固定源码差异：C4默认ModelLinear独立参考用了FMA，而本路线既有F45.6已经实测锁定
同一原MLX block128程序的unfused F32 scale product/sum。有限纠正把该选项显式传入C4
model，CPU/CUDA仍默认fused，Apple在构造所有reference前固定unfused；不逐值选择答案，
不改产品、不改FP8/BF16/storage/tolerance。只复验两个失败ID。inverse-RoPE的CPU trig
逐位检查若仍不符合原GPU程序，则保存具名合同问题作有限source定界，不无限测法/放宽总门。

`7da02640`具名delta仍2 FAIL/入口，Pro同4个pointwise比较、Flash同5个exact inverse-RoPE
比较。FMA选择没有解决本witness，不据此归因或签PASS，停止盲改oracle。当前cache/attention
其它输出成功不代签完整片段；后续只做具名源精度/存储边界定界。目标hidden参考受新显式
舍入选项影响，终审前定向确认；其它Markov/confidence和三sidecar证据不变。

A4允许独立先验证新成熟借入的实际SDK闭包：新公开C++ consumer仅链接hyle::graph，
源码嵌入与static/shared find_package使用同一main；真实非零Q4_K MatMul双replay、原所选行
decode/失败/保留lease及NF4成熟decode，不使用私有HYLE/vendor头或mock。先复用当前shared
producer，只有INSTALL选项改变时configure；relocation到独立prefix、metallib/许可证/dylib
与两入口验收一次。它只证明相应交付，不代签待决FP4和两个失败官方witness，不启动P1。

`8e95478c`同一公开consumer三交付样本全部PASS：
[源码/static/shared receipt与raw](../../../../evidence/correctness/apple-parity-a4/README.md)。
安装SDK搬迁、断网configure、隐藏producer bin/metallib后Direct/Graph真实Q4_K双replay/
精确所选行/坏索引/close后lease以及成熟NF4运行通过，资源hash与新borrow许可证匹配。
本批consumer声明/参数和CMake编译器cache刷新失败均保存；纠正仅消费/配置，不改产品，
shared producer不随consumer修改重编或重装。source复用build耗时58.95s。A4其它公开头/
boundary矩阵与最终清理尚未闭合；A2数值两项、官方FP4与A3资格仍待，不冒称整个交付/goal
完成。统一实现仍61,328行/261文件；本轮新增均test/evidence，不计入产品实现行数。

两项官方attention残余按[一次数值责任捕获](apple-dspark-numeric-cause.md)执行：只Direct
两ID/原六replay，原门保持，逐层原source方程/原MLX trig诊断，不继续猜测FMA或调容差。
三交付已通过consumer保持原候选；本批还独立检查三形态的三个Apple公开头，未变矩阵不跑。

一次捕获已证明CPU/GPU trig精度路径错配与前驱误差传播，非grouped/final接线错误。
`bdc60f88`按原MLX数学预先固定独立完整参考后，两个失败ID Direct2/2、Graph2/2全部PASS，
原whole pointwise、BF16/storage、cache反馈/reset门不放宽；矩阵/三交付不重跑。九新witness
两入口已有合并PASS证据；目标hidden/confidence的受影响参考两入口各2/2确认、三形态公开
头9/9通过。数值原因任务结案，不继续捕获；官方FP4与依赖它的完整官方main/DSpark专家
仍待原桥接决定，不用量化MXFP4侧车代签。A3默认/有预算成本裁决与最终A4复核仍待。

A3的[默认与有限成本范围](apple-default-cost-scope.md)已按当前产品源码定界：多数实际
packing域是单成熟carrier，不假造best-of竞赛；plain BF16/F32 linear与plain grouped的
两成熟候选保留完整转换/内存/同步成本，最终稳定candidate后才做原生准入与一次P1。
不存在本路线自研shader/GEMM/GDN/Hadamard kernel可虚报清退；语义/资源桥接不误删。

两重叠域数学准入已完成：原Metal与原MLX候选同普通Context服务，4/4及共同router几何
定向1/1通过（原N768只额外sample，不冒称目录；当前N512/K8192）。逐值精度、变输入/
变权重、checked IDs/lease/close均通过；MLX动态BF16权重widen的16MiB/128MiB额外数组
列入P0成本。不是性能WIN，不切默认，不重跑矩阵/交付/P1，正式采样等完整candidate稳定。

C1的CPU+Metal shared-core定向编译/实际分派已通过：runtime9/9、Metal provider8/8，
六个active core/codec/Metal source文件与Apple-only同hash，原三类型CPU实际codec执行和
Metal原有资源/数值边界成立。新八IQ1合同的同source旧数值复用，不反复重跑；CPU/CUDA/
Vulkan模块状态未动。user Apple页同步当前21 GGUF/Mlx/artifact能力、checked行/容量和
原FP4未决，清掉已过时“仅十类”的说明。仍不签FP4、正式成本或whole goal完成。

A4最后的资源失配负例只用已通过的source/static/shared公开consumer原binary：fresh process
分别隐藏对应app/SDK与两个producer的metallib，清 inherited DYLD，要求明确缺resource错误。
原MLX source缺library路径抛runtime_error，不以设备枚举/mock代替。try/finally恢复全部
文件并校验原hash，不改binary/系统设置，不重跑旧成功矩阵。三delivery各一次负例，未跑
前不PASS；若异常终止或意外找到外部资源，记录具名失败，不无限藏路径重试。

资源负例三形态均安全exit1、文件原hash已全部恢复，但消息仅“MLX Metal GPU is unavailable”，
不是明确缺资源诊断。具名原因：原MLX gpu::device_count catch-all把缺metallib异常吞成0，
HYLE先is_available使原错误不可见。F5固定唯一产品FIX：在同一cold owner先读取已有GPU
Environment完成实际native初始化，保留原错误，再创建stream；不添加文件扫描/额外probe/
CPU fallback，不改数学/公共API/热执行。当前compute_route_apple仍AUDITED；其它owner
状态不变。仅Context/Environment既有测试、三交付原consumer成功路径与缺资源负例、原hash
恢复一次验收。适用selector/header/交付按该风险，不重跑全模型或P1；失败最多一次纠正。

`4c3a0b53`F5诊断FIX验收通过：Context/生命周期9/9，原source/static/shared公开consumer
成功路径3/3，fresh process缺resource负例3/3均明确报metallib无法加载而非GPU不存在，
所有资源try/finally恢复后原hash一致。[原失败/修复raw](../../../../evidence/correctness/apple-parity-a4/README.md)
保留。只增量编译一处cold initializer及重链必要consumer，未重跑矩阵/P1。公共API与数学
不变，FP4取舍与A3正式成本仍未完成。

A3性能消费程序已准备：routerM1/M16、groupedBF16/F32四域完整两银行输出预检通过，
primary/placement已构建、实际owner QoS21已观测，固定合同validate通过；A/A与正式n15
P1都未运行，不签性能/default切换。待FP4取舍并固定完整candidate后只执行一次正式批次。
