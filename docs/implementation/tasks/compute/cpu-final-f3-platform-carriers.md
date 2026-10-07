# F3 三平台成熟CPU承载

<!-- MODULE REVIEW: AUDITED -->

Native原ATen/Torch程序owner的F3范围重新定界；baseline `2792b79f`（实施前记录完整revision）。
此SCOPED只冻结范围，旧F1 AUDITED不授权本批新产品；独立审计固定下表后再绑定registry至本卡。
总合同[CPU最终收口](cpu-final-closure-plan.md)，以[F2固定矩阵](cpu-final-demand-carriers.json)为输入。
优先复用ggml原CPU、可移植成熟ATen/BLAS/量化程序及已有借入组合，补Mac ARM64和实际平台缺口。
保留原BF16/F16/FP8/FP4/编码与state精度、selected expert/cold转换成本；CPU执行不调用Metal代签。
每个采用库先锁版本/hash/许可证/ABI，完整source/static/shared runtime交付；同进程core归属按F1冻结。
一次批量接线，两入口/正确性/资源/失败/OOM/并发/生命周期/三交付在本包完成；单承载P0，不造赢家P1。
没有合格成熟承载时列精确端口/反例与有限取舍，请求必要决策，不能自己扩数学或删除固定需求。
写产品前真实owner SCOPED/AUDITED，公共Graph BLOCKED不自动恢复；只改有具体反例的合法owner。

## Mac CPU有限承载批

真实trigger：平台门只允许Win/Linux x86_64，已借原ATen/codec程序未在Mac CPU交付，固定模型必需合同因此缺失。
官方ARM64 LibTorch2.8原archive已从Windows取得并传Air验证SHA；build-version为`2.8.0`，
archive SHA `be0642231d2008c7b90071fc0898d82a0c002d47f38350ebf3ed779fc01972d7`、71,787,762字节。
Air实际Mac16,12/macOS27.0.1/AppleClang21/CMake4.3.4，产品源码仍只由Windows前向同步。
otool发现CPU dylib的libomp依赖硬编码Homebrew路径：不能靠本机已安装brew冒充可搬移SDK。
原包也有weak MPS/Metal系统链接；HYLE借用程序的张量与操作全部CPU，不启用HYLE Apple路线或以GPU代签。

| FIX | 有限动作 | 本批验收 |
|---|---|---|
| T1 | 锁Darwin-aarch64官方SDK原archive/版本及许可证，标准CPU AUTO取得而非静默关闭 | CPU preset实际包含原ATen/codec，offline本地archive/hash miss明确失败 |
| T2 | 平台SDK key/精确version校验同时服务source与installed consumer；Win/Linux+cpu条件保持 | 外部root错误版本/目标拒绝，Mac2.8.0只用于ARM64 CPU |
| T3 | 只在有owner marker的managed SDK staging修正libomp为loader-relative，保留原archive/SHA与派生内容manifest | 不改用户外部SDK；搬移/无Homebrew依赖仍装载、真实CPU执行 |
| T4 | 注册torch/torch_cpu/c10/libomp真实dylib闭包，RPATH复用现helper；编译现原程序不写新数学 | 两入口、dtype/packing/状态/失败/资源/并发、source/static/shared集中验证 |

普通API/ownership/失败/并发/完成与准备/热路径不改变。tensor逻辑和执行精度按原程序保持，
全模型CPU math明示、shadercompiler/Metal context不作为本批依赖。无本批赢家P1，单承载P0与正确性/资源为主。
预算：一个锁定SDK、一次批量接线/构建及至多一批具名工具链/loader纠正；不换版本寻找通过。
C1/F2合同全部仍适用；缺数学、模板/SDK ABI无法适配时记录具体反例，不缩范围、不开自研。
本卡不等于F1已完成；已稳定F1事实按精确身份复用，最终交付在本批完成不首次留给F6。

## 审计完成与实施冻结

baseline完整revision `2792b79fabd9439341517edf3f44b1762faa22c0`，T1–T4已AUDITED。
原代码不含x86-only数学/ISA要求；既有ATen/借入量化程序按CPU Tensor执行，原SDK ARM64提供所用接口。
runtime闭包为torch/torch_cpu/c10/libomp四dylib。SDK原压缩包SHA固定，managed stage独占地修正
CPU dylib的绝对libomp依赖为`@loader_path/libomp.dylib`，必要ad-hoc重签只作用于该派生stage；
原archive永不改、派生文件重新进完整manifest。外部用户root不修改，必须符合可装载/版本/架构合同。
installed shim同样按平台key判断版本，Win/Linux2.8.0+cpu条件不放宽；CUDA Torch始终拒绝。
现runtime helper只有Linux RPATH分支，T4有限补Darwin路径，保持Linux专用linker flag不传Apple。
公开API/对象/原数字程序不改；Native owner重新审计后合法绑定本卡，不继承旧F1有限FIX授权。
F2/Mac缺口由此实际施工，shared/static/source批量验收与负空间均需运行，不以SDK获取签覆盖。

## T5组合oracle有界纠正（测试，不改数学/误差门）

Mac原C5 Direct653项中652通过；Pro0813 Q8的mHC→BF16 head失败。
定向同输入探针证明mHC所有原输出门已通过，最大f32差异4.768e-7，但一个值跨BF16 RNE阈值；
原head actual使用observed mHC、oracle却使用另一reference mHC，两个不同输入不能签head数学失败。
actual输入的原codec/dot独立对照通过；不据诊断PASS签原完整批，原失败/全部数值差异保留。
T5只修测试组合：上游mHC仍独立reference、原误差与实际state反馈全验；下游head保持原BF16转换/权重/真实K/V分片，
独立oracle消费相同observed上游f32值。没有阈值放宽、不换dtype、不预计算actual输入、不删除分支/反馈/全head。
Float32允许的实现误差不承诺跨BF16舍入阈值后与另一reference bit-identical；这是逐精度边界组合验收。
单case纠正重验+Graph全部C5首次验；Direct其它652复用实际通过XML，原aggregate FAIL不重写PASS。

## 当前实际验收（shared/source部分）

Windows不可变产品`42898c37ef90005944b73f4bf174f9b7f1070550`前向同步至Air，
CPU配置中Apple/CUDA/Vulkan关闭；managed原ARM64 SDK身份/版本与loader修正通过。
原Provider owning组通过，普通Direct12/Graph11均无skip；本地Qwen原权重投影fixture已绑定。
C5 Direct原652项通过，T5候选`6359804abc6e0543f3fef9a4b6b622f01a8b98db`具名单case完整head重验通过；
Graph全部653项首次运行通过、无skip。原Direct aggregate FAIL及诊断边界保留。
[原日志/XML/来源与receipt](../../../../evidence/correctness/cpu-final-f3/README.md)归档，不改原误差门。
installed static/shared/搬移、Linux当前版本及其它完整旅程仍待运行，本包不关闭、不签新性能WIN。
统一Compute+Graph实现61,242行/251文件，新增自研数学0。

## Mac三种C++普通交付已验

候选`9a9936e8271fc834f4b090316bc0a5d7f9b25982`由Windows现有source-package脚本生成完整带manifest/SHA的源码包，
Air只消费前向副本。两个公开consumer按每种交付共用一个producer：source嵌入2项、
shared搬移后Direct/Graph各1项、static搬移后各1项均通过，无skip；shared出口检查16库/1262出口通过。
唯一环境纠正为SSH非登录PATH缺Ninja，原失败保留，只续失败步骤，不重建已通过producer。
四原ARM64 CPU dylib的lipo/otool已审；torch_cpu的libomp边为loader-relative。
机器SDK仍按installed shim单独解析，不冒称LibTorch随HYLE SDK完整打包。
[交付receipt/日志](../../../../evidence/correctness/cpu-final-f3/README.md)有原archive、manifest、runtime与consumer身份。
此为普通两入口交付，不签全部C5在installed形态重跑；Linux当前升级候选集中验证中。

同候选的installed shared/static另复用原C3格式consumer（原文件SHA不变），每形态36个Direct/Graph数值检查通过：
十一GGUF CPU新增格式、NF4/nested/AWQ/CT原解码及真实projection、affine4/MXFP4/MXFP8选中专家。
只各编一个consumer文件约1秒，不重建producer；全模型语义/状态仍由具名C5/官方片段独立验。

## Linux当前成熟承载已验（source/shared，交付仍待）

同Windows不可变source archive候选`9a9936e8271fc834f4b090316bc0a5d7f9b25982`只前向覆盖已空闲的ext4验证副本；
Linux GCC14.2、原LibTorch2.8.0+cpu、ggml0.26和oneDNN3.13.4。oneDNN新版本低并行集中build777.711秒，
11组Provider/原runtime owning验收通过。Direct全批828项原821通过、7个本地fixture读取失败；
原因是archive有fixture/目录而env指外层。只改env到真实目录后7项1.561秒补验通过，原FAIL保留。
Graph首次全批846项通过，无skip；完整官方混合precision/新DeepSeek主干、三stage DSpark/MTP sidecar与C5端口/状态均实际运行。
[原配置/构建/失败/补验/XML](../../../../evidence/correctness/cpu-final-f3/README.md)归档；没有产品/误差/fixture字节变化，
不重复成功Direct或重编producer。Linux installed static/shared、Mac剩余官方分支与Windows当前全批/最终交付仍待验。
WSL只签Linux correctness角色，不冒称独立formal性能机；本包不关闭。

## T6 ARM FP16 MatMul累加准入纠正已审计

Mac剩余批出现OriginalScaleDtypeRoundsBeforeExplicitFloat32Output失败：原artifact decode逐值精确门通过，
随后plain F16 MatMul输出6.40234375而独立F32累加/一次F16舍入应为6.3984375。
原ggml vec.cpp的ggml_vec_dot_f16经GGML_F16_DOT映射进入ARM FP16 vfmaq_f16/vaddq_f16，
GGML_PREC_F32 metadata没有使该实际dot宽化。不能以half输出误差门掩盖FP32累加合同。
成熟ATen已有显式F32输入/MatMul/一次输出舍入和Mac owning资格，普通CPU路线不需要新数学。
有限FIX：compute_provider_ggml在ARM64 cold supports拒绝plain F16 MatMul FP32请求；
Graph完整region复用同一supports，不另写旁路，合法ATen leaf承接。x86、packed原codec与其它原op不改变。
已有ggml owner在F1 AUDITED，T6定界后绑定本卡；新增一个ggml拒绝/普通两入口精确反例和实际artifact composition重验，
迁移旧F16“必须ggml”metadata断言。单批构建/定向补验/原remaining Graph首次全批，不重跑成功C5、不跑P1。
不升级或改原vendor数学、不扩自研、不撤回标准CPU F16算子可用性；原FAIL保留。

T6改后C5只补受影响plain F16的GGUF/非GGUF端口与相关默认metadata；其他成功C5保持复用，
不是重跑整批。普通artifact composition维持原精确门与两次变输入。

Graph旧身份/融合断言也随T6迁移：ARM F16普通linear明确ATen，Hadamard仍ggml，保留实际变输入/独立slot数学；
F16全宽投影仍用该ARM用例原有max2.5e-3与relative-RMS1e-3门，门属于同输入/半精度输出而非“允许半精度累加”。
不因换provider把near-zero改成不含F32 reduction舍入预算的纯relative门；阈值不变，另由精确累加反例排除非法half累加。

## T7 sidecar片段同输入组合修正（测试）

同Mac Direct剩余159项中157通过，另一个失败为Flash Q8 sidecar完整片段；
源码明确前置prefill已经独立通过，但complete的实际输入是cache_actual，oracle初始化却是cache_expected。
这仍是两个不同输入跨量化边界的比较，不能签为同合同实现失败。
有限测试FIX：prefill/embedding各自保持独立oracle与原精度门；下一complete片段的独立数学对照
从其相同实际prefill cache/embedding初始化，完整三stage、replay反馈/reset、head与所有分支不变。
complete内部仍以独立reference推进并受原全片段误差门，不能逐node用actual替换reference来消除误差。
只运行一次具名完整sidecar Direct重验及原remaining Graph首次批；如果仍失败，保留真实数值反例再定界，
不放宽阈值、不削小词表/状态、不无限换输入重测。原159项FAIL保留。

### T8 sidecar数值原因有界调查

T7同输入修正未消除Q8 sidecar首replay state反例（归一化max4.2128861% > 原4%）；
同片段Windows/Linux已过而Mac未过，不能记工程默认PASS或删除需求。
先完成独立Graph首次批。调查预算两次定向probe：第一只暴露完整三stage的attention/post-MoE状态与独立reference，
找出首次偏离；第二只查第一probe定位的一个原库/组合接缝。probe不签资格、不改产品API/算法/误差门，
不换模型输入、样本数或量化。只有具体源码/准入反例再审计有限产品FIX；预算内无法裁决就说明精确未决边界，
不给不合格形态签完成，也不扩全库/所有算子性能调查。

### T8结果与T9参考FP32边界纠正

两probe均已用完：第一attention/post-MoE偏差逐stage增大；第二首qa/Q8投影max-scaled0.439%，
sink局部0.0763%，最终约4.21%。Direct/Graph反例完全相同；没有证据指向图调度或宽精度累加失败。
具体源码反例：sidecar norm_ref复用Double平方/mean/sqrt/两次乘积后一次F32舍入；
原GGUF/公开RMSNorm链路是F32 square、F32 mean/rsqrt结果与F32乘法边界。
量化之前多保留的Double位数改变下一Q8编码，不能用最终storage同为F32证明中间精度相同。
T9只修sidecar测试norm_ref为独立scalar原F32边界；square在F32、累加按原ggml的double sum，
mean/rsqrt/normalization/weight乘法各显式F32，不借actual中间值替代oracle、不修改product/provider。
Double高精度误差对照的旧FAIL/两个probe完整保留；四percent片段门、全词表/三stage/cache反馈/原权重/输入完全不改。
这是两probe找到的具体参考精度FIX，不增加第三次探针；一个不可变候选上完整sidecar两入口正式重验。
若仍失败按原门保留未决，不继续改参考/阈值寻找通过。现FP16准入修复与其它已过资格独立保留。

## T6–T9正式结果

产品`6bf82ba8`的ARM FP16错误准入已撤；owning旧广播断言按实际拒绝迁移后通过。
受影响23个F16合同：Direct26项中的artifact精确反例、默认metadata与23项通过；
该批唯一仍失败sidecar作为独立未决保留，不改该旧aggregate。Graph24项受影响合同通过。
参考候选`3c62c77b9b93f127fb8f71d265cbc89247f427c0`保留原FP32 norm边界后，
完整Q8/BF16两sidecar Direct1项、Graph5个失败case名（过滤命中6项）全部通过，原4%门不变。
两probe只归因不签资格；原Direct157/159与Graph171/176的成功格复用，所有旧失败/缺include编译错误仍保留。
[完整收据/原失败/两probe/正式纠正](../../../../evidence/correctness/cpu-final-f3/README.md)已归档。
当前源代码Compute+Graph61,246行/251文件；4行有效cold准入胶水，自研数学新增0。
新ARM准入修复的三交付增量、Windows全批和Linux当前安装仍待完成，F3不关闭。

## T10固定Q8 sidecar的量化前原norm规则（SCOPED）

Windows当前Direct全批唯一失败为同Flash Q8 sidecar；原FP32参考门max约4.20% > 4%，
该平台普通RMSNorm优先oneDNN，而Mac使用原ggml。不能按平台更换reference或放宽门。
范围只为已固定repo/revision/manifest/variant的Q8 sidecar、plain dense F32 RMSNorm、原epsilon1e-6/F32累加，
冷匹配优先原ggml以保持量化前原GGUF程序边界；不是性能WIN、不改变generic默认、数学、对象、热路径或公开API。
真实owner compute_route_cpu仍AUDITED于F1；先审计本具名规则，绑定本卡后才实施。
一次增量构建与Windows具名Direct/Graph正式重验；其它成功全批复用，不再开探针或换参考。
若仍失败保存具体结果，不追加新norm方案。其它平台同规则的资格按产品实际影响补验，不假称自动通过。

### T10 AUDITED

原接缝确认：Request已有不可变WorkloadContext/profile，cold默认仅decorate成功proposal，
不改变provider准入、原execution/lifetime或Graph完整region。ggml RMSNorm原F32边界与该原GGUF sidecar一致；
oneDNN generic仍是合法数学，但此既定量化敏感片段没有通过完整状态门。
固定FIX为上述repo/revision/variant/manifest四项一致、format_family=gguf与RMSNorm F32/epsilon/两plain dense端口匹配；
给此具名工程规则独立entry/version，不借旧WIN，Unknown/失配不带该entry。无新常态metadata/alloc。
现Direct827项通过、唯一sidecar FAIL，原记录保留；不重跑827。Graph原full尚未运行，rule后的首次全批不伪PASS。
风险验收含正确软件/真实profile命中、revision/manifest/variant失配、普通unknown沿generic、Direct/Graph同数值与资源。
