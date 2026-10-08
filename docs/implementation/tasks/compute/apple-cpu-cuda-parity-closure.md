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
