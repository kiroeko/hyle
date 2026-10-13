# G01 Apple/Vulkan 原生精确页读取

状态：AUDITED，2026-10-14。baseline
`9856c7c6e4f8e7f3729efbc89e43352db49b4c4e`；授权来自
[十项执行总卡](../project/inference-gaps-goal-execution-2026-10.md)。本卡先定问题和验证，
未AUDITED前不改产品；原finite resident/ragged/CPU page/default construction关闭证据保留。

## Closure card

owner为compute_route_apple、compute_route_vulkan、compute_provider_ggml_vulkan及
compute_ggml_runtime。原candidate分别a3fa6be8cf22780491e8b383ef1aafb678db1e23、
a3fa6be8cf22780491e8b383ef1aafb678db1e23、bc0ba41f714f6529cb1c583bee59d482946e7589、
42f98b94f33bdbb252b9259ece33533cf763fa41。只重开原目标需要的新page机制，旧数学与
已验CPU page仍可依赖，停止重复其P1。Graph/native联合owners不在此时改产品，G02
[原卡](../graph/inference-g02-multi-queue-batch.md)保持完整未决清单。

trigger是已准入真实两request KV arena消费：当前native MaterializedBatchInvocation每Q/
head gather整历史并走reference recipe，原prepare_ragged_batch只有stride contiguous KV，
均不能把任意physical page order/tail/share按逻辑序直接交给原native attention。
[输入审计](../project/inference-single-machine-foundation-audit-2026-10.md)G01/G02已记录该缺口。
原CPU page有限轮已CLOSED，但不构成两GPU数学的反证。

主消费者：Qwen3.5 attention片段的resident GQA cache，单请求decode及两request混合
prefill/decode，物理页反序/尾页/共享/未引用sentinel；正常用户通过原typed native Context/
MemoryResource/Program/Invocation与F1确切frame运行。正交反证为非模型的变长GQA数值
两段、非causal/空request/不同topology两slots与source alias；不把模型请求策略下沉。
至少保F16共同域、GQA8/2、D64/128/256、page16；各原已准入dtype/数学资格在AUDITED
核原source后明确，不能用一个最小点代签本卡或删除goal其它域。

机制裁决先BORROW原锁MLX SDPA/Metal、ggml Vulkan flash/shaders及原native retirement；
SELF仅不可避免的页地址/有效长度承载、冷owned topology/精确guard glue。不新增attention
公式、第二executor/queue/scheduler或每Q/head KV物化。是否改原shader读点及哪个有限
primitive/recipe承载须在AUDITED冻结，不能借benchmark输赢选择API。

已核原依赖MLX1f8e74e3f12f31365464a6867c6579f0e9b29d85和ggml456172ec733a135778adcd32d00e576a58232e45。
MLX vector/2pass与steel tiled都有独立K/V地址读点；当前入口只有array/stride/有效N，
无page table参数。ggml Vulkan当前flash标记为F32数学资格，不表明已支持页地址；需要
审计全部shader读点与原descriptor/dispatch/record假设，保实际capacity与独立logical行数。
固定三engine peer来源沿总卡，读取原block table/valid region合同，不fetch或升级依赖。

scope包括原native高效页数学、owned version/table保活、真实物理root/span与逻辑读域
资格、typed public批次入口及原normal/ragged负空间。G02之后把layout/generation和所有
page guards接入joint plan原fragments/slots；已完成LeaseSubmission页组不授权raw arena
cast，也不证明任何generic native input只读已准页。新的public shape/owner/negative
space及有限FIX未固定，当前不写实现、不启动P1。

预算：源审计/实现/验证各2h检查点，自动有限延长并保所有原FAIL/累计；重型串行，
Windows≤4jobs，不启本机重型GPU/全模型；Apple资格只actual Air。无外部阻断时继续定位，
预算不按新log重置。停止依据有限FIX/适用门终态、真实无法推进或协议要求回退，非短超时。

预声明验证：两request independent nonzero double oracle；三D与冻结dtype、causal/
noncausal、page reorder/share/tail/empty、64次复用及有限演化、双topology slots/旧output；
foreign schema/resource/generation拒绝、未引用页可写、共享页防写、quota/逐cold分配故障/
accepted native prefix失败与实际retirement。结构P0核所有实际K/V读点、无warm历史拷贝/
Q-head展开，table/metadata/host wait/private UNKNOWN如实计费。selector只真实owner与
直接native/Graph消费；source/static/shared及公开逐头、适用宿主三san；Air实际GPU数学，
Windows只定向compile/link及已允许的有限风险。P1仅不可变API/算法、同生命周期/原数学/
原16AA/controls/5%门、实际source/SDK身份固定后串行一次。普通closed CPU/ragged不重测P1。

下一步只完成本scope内原读点、资源/frame/version/submit形状与有限FIX审计，进入AUDITED
后实施。整个G01/G02、G03–G10保完整目标，不由这张scope卡或一组source结果代签完成。


## 原读点审计结果与共同owner

[有限源审计](../../../../evidence/architecture/inference-native-paged-read-audit/README.md)已
核实际原锁文件/host dispatch与HYLE现有patch：MLX在当前M4资格中的NAX路径用kU16，
Ktile<T,2,1>一次加载32逻辑rows；任意page16不能仅替换这个32-row tile的base。必须处理
每16-row原fragment的地址/边界，保原MMA/softmax/reduction；vector两路径、旧steel loader
与NAX dsplit也不能由一个decoder路径的结果代签。

Vulkan原push constant已经128bytes，普通layout不能扩张；原HYLE FullF32 tag明确走
scalar tuning/f32 shader并禁止全KV dequant/repack。因此本轮先保原已资格数学身份，
不为了pages改无关normal coopmat路径，也不把pristine coopmat存在冒充HYLE当前同数学
资格。BF16原carrier/decoder存在，但现有SDPA/tag admission及output cast并不完整接受，
其实际有限dtype与输出档位要在AUDITED明定，不能猜BF16全链已完成或偷偷drop goal。

中央图把共用MaterializedNativeState头归compute_batch；其metadata banks/frame compare/
page admission/actual retirement是两原native路线共用机制。为复用该controller的元数据/
版本/原生退休承载，compute_batch以本卡进入SCOPED，原F5 candidate11c348e404df2c3fc404056ff3a9e885d4d990bc
保留。只允许在AUDITED冻结后做必要共同承载，不扩大producer/政策/第二queue或仪器。
新native公共batch入口/metadata形状/Context借用与资格负空间尚需冻结，当前五owner均
SCOPED；不以完成source audit宣称AUDITED、CANDIDATE或数学PASS。

## API、承载和有限FIX冻结

当前五owner进入AUDITED；原SCOPED5/source audit为输入。通用机制SELF只owned metadata/
version、原固定bank与权限协调，BORROW全部原native数学、CopyBatch/Invocation/实际退休。
metadata与math admission分开：common recipe可描述F16/BF16，Apple承载F16/BF16，Vulkan
先沿已验F16/fullF32原数学域。Vulkan BF16全链不在此处冒称接受；原goal G05/G07实际
dtype/语义选择与未证资格保留，不由carrier存在取消或代签这些项。

普通调用冻结：先`NativePagedBatchProgram::prepare(schema, options)`，两route各自
`PagedBatchInvocation::create(context, recipe, device, staging, options)`；
`prepare_batch(layout)`→`bind(query_read, key_frame, value_frame)`→`submit()`返回原
OwnedResult<route::Results>，report/close/drain沿原native规则。普通用户不传第四个raw
metadata tensor，不得自行扩大arena permission。Reference MaterializedBatchInvocation
仍表达不同的显式reference算法，不新增旧签名转发或兼容层。

Common recipe只持schema/scale/causal及四个输入desc：packed Q、真实page capacity K/V、
I32 max_page_refs的owned页表。version保同schema完整owned layout/generation，冷生成
仅O(max_page_refs) I32表，unused slots零初始化；没有Q/head/KV展开、mask或payload拷贝。
version只借caller layout的创建调用；导出ReadLease保recipe/layout/allocator/budget owner。
所有大小/费用有溢出检查与显式host metadata budget；host table与metadata wrapper分账。

Native slots复用compute_batch原controller的banks、CopyBatch upload、frame compare、
原LeaseUsePool和native retirement：引用与page-native只由policy决定metadata count/recipe。
page-native一个table/staging与两mirror bank，不复制原controller。prepare先验schema/
原reuse gate/旧output/permission use，再cold准备原native Program与table；本轮topology
变化cold prepare如实收费，不声称G06增量planning/capture完成。Context/device/staging/
allocator借用且保持地址稳定，活过所有work与结果；控制外部串行，独立slots无共享mutable
bindings。接纳前失败不得发布新generation；已接纳upload/math错误或未知完成沿原failed/
close/drain，只有actual retirement释放guard；metadata/scratch/native private UNKNOWN保。

Native math每request只取active Q与其owned页表区间；物理K/V root shape/bounds保持真值，
logical KV独立由refs/tail得出，允许逻辑重复页大于physical容量，不伪造大view。只运行
实际页地址qualified原kernel，query/output/原mask或causal表示费用单列。页表origin、
dtype/count与实际root/resource身份都核；空request不读KV。普通nonpaged/ragged数学、
layout及常态Request/Program不加page fields；原native graph record/cancel退休合同保。

| FIX | 真实owner与允许改动 | 验收 |
|---|---|---|
| NP-01 | compute_batch owned NativePagedBatchProgram/Version、四input/单I32 table、预算/identity/负空间 | 非模型metadata消费、foreign schema/overflow/empty/share/tail/逐分配失败、源/SDK/逐头 |
| NP-02 | compute_batch 原MaterializedNativeState泛化metadata policy；两native route薄typed PagedBatchInvocation | exact frames/generation、两banks/slots、旧result、partial failure/actual retirement、原reference负空间 |
| NP-03 | compute_route_apple 原MLX private paged primitive与原vector/2pass/steel/NAX（含dsplit）页读地址 | F16/BF16三D/causal/两request/任意page16、独立double、无warm全KV copy与头展开 |
| NP-04 | compute_provider_ggml_vulkan 原recipe私有页op/root/table资格；compute_ggml_runtime原fullF32 scalar shader/host地址+logical KV | 不增长128byte普通push layout、不改normal coopmat、全读点/tail/重复页、原record与数值 |
| NP-05 | 两route public native批次入口与guide/直接consumer，selector真实native/Graph负空间 | 原预声明正确性/宿主风险/三交付/逐头/同合同P1，未跑门保UNPROVEN |

私有shader/packet细节可在上述原math身份下有界细化；改变public形状、owner、权限或成本
等级即退回AUDITED。所有FIX均未验收，不能CANDIDATE或签整个G01；G02新joint binding
待原page-reading数学资格后由其真实owner另行SCOPED/AUDITED，原goal域保留。


NP-01实际产品草稿197d39684已提交：owned四input/单I32表，version输出lease独立保
recipe/layout owner、无self cycle。Windows四source tests PASS，含query8192/head16
表仍32byte、logical95>physical64真实重复页、empty/foreign/budget/nonfinite/逐cold OOM。
见[阶段](../../../../evidence/correctness/inference-g01-native-page-metadata-stage/README.md)。
首Schema比较compile FAIL保，改用原owns身份函数后source2/扩case的source3 PASS；最终
19.422s配置/21.954s构建/0.031s run，四jobs。NP02–05尚未实现，五owner仍AUDITED。
原component contract首次137.157s FAIL缺验证进程SDK环境，已定位实际已装SDK并同原
工具第二次跑，保失败不改工具。中央仅增加新头的owner mapping，selector仍保守选
全库；按独立工具卡留原计划，不重开无变化owners或运行全库数学。


组件合同第二次299.531s FAIL进一步定位为工具生成shared-source-cache位于canonical
source内，违反原dependency cache规则；不是新页表产品故障。保前两次137.157/299.531s
与全部生成目录，未改公共脚本。第三次把明确验证根置于C:/h7/g01-native-page-component-gate-3
（产品仓库外）、仍原SDK/工具/fourjobs，验证子目录删除边界先核。当前实际进程运行，
未获terminal不写PASS；宿主三san入口已按最终七个canonical SHA输入准备，待重型任务
串行门完成后执行，NP02–05与全部native数学资格仍未实现/未证。


NP-01宿主风险ASan+LSan/UBSan/TSan各37tests PASS，新metadata＋原layout/materialize/
arena回归，实际flags核验；0.433/0.171/0.337s run。NP02共享controller草稿3747a1a52
已提交，metadata policy固定count1/2，复用原bank/CopyBatch/frame/permission/retirement，
未复制controller或新增queue；page policy借stable Context按layout冷准备，原reference无
该额外字段。实际Air Apple/Vulkan旧materialized consumer分别12.110s/23.145s build、
4.184s/7.339s run PASS；只签原negative space，不签新page policy/native数学。见
[风险与原controller](../../../../evidence/correctness/inference-g01-native-page-metadata-risk/README.md)。
原组件合同第三次255.328s PASS，原两FAIL与累计692.016s保留；未改公共工具/构建核心。
五owner仍AUDITED：新的typed paged invocation/实际page compiler/NP03–04数学未实现，
新头static/shared等仍未证，不由37tests或reference输出代签CANDIDATE/整个G01。

NP02/NP04 Vulkan实际草稿接入：公开PagedBatchInvocation只接受确切K/V frames，复用
共享单table policy；私有四port op把真实physical roots、logical rows和table offset交给
原FullF32 scalar flash。私有tag使用op params5–8；原binding6（不用mask-opt）承载完整
I32表，原mask-only push字段承载offset/原Q数，普通128byte packet和coopmat身份不改。
全部scalar K/V staging/direct/decode读地址映射page16，causal限制按logical位置；不新增
attention公式、不物化KV或每Q/head mask。exact patch contexts/source/result SHA已逐项
重放核验，source cache未改。新实际GPU消费者包含三D、causal/noncausal、重复/乱序/
tail/空request、logical79>physical64、64次topology演化、双slot/旧output、foreign schema/
mixed frame拒绝与32byte上传费用。2026-10-14 actual Air Mac16,12/arm64/macOS27.0.1/
AppleClang21已复核；只启动该消费者和原reference target的fourjobs复用树构建。
这是AUDITED可变草稿，编译/数学结果未获得前不写PASS；Apple原生paged仍未实现，
本批三交付/适用风险/P1亦未完成。selector因patch中央lock变动保守全选的原plan保留，
沿工具卡限制实际验证为变动owner/真正消费者，不重开全库或启动全局数学验证。

Vulkan草稿2500128ed获得actual GPU source六组PASS：F16 D64/128/256×causal/noncausal，
每组64次topology演化，independent double oracle与logical79>physical64/确切guards/旧
output/foreign schema/mixed generation/32byte费用全部通过。原materialized direct/Graph/
fault/append负空间也PASS。见[阶段证据](../../../../evidence/correctness/inference-g01-native-page-vulkan-stage/README.md)。
首83.783s compile FAIL为fixture误用half bit carrier转换，修显式encoder/独立decoder后
build0.876s/math7.144s/reference5.345s PASS，原FAIL/input/log与累计97.149s保留。
实际编译host/scalar after SHA分别核验。尚未完成新page accepted-prefix/unknown风险、
宿主风险、三交付/逐头/P1及Apple页数学，五owner仍AUDITED，整个goal仍0/10。

Apple NP02/NP03与NP05接入草稿afd2d8df7已实现：薄typed slot复用单table共同policy；
private MLX inference-only primitive借同一个原eval/dispatch，不改普通SDPA或AttnParams
layout。vector/2pass/GQA、older steel loader、NAX32-row K tile的两原16-row fragment与
D256 split读点映射page16，保原MMA/softmax/reduction。实际Air最终short12组＋long16组，
各64演化（共1792final episodes），F16/BF16三D、causal/noncausal、logical79/4111/8207>
physical64、双slot/旧result/确切guard/generation/成本/metadata零用量 independent double
全PASS。所有实际MLX patch after SHA与compiled inputs分别核验，见
[Apple阶段](../../../../evidence/correctness/inference-g01-native-page-apple-stage/README.md)。
首33.481s FAIL缺直接mlx/ops.h，修include后通过；保所有初版和final输入/日志与累计
59.438s。原reference direct/Graph/fault/append负空间PASS，不代签新page风险。两路线
普通public消费和限制已写[使用页](../../../user/batch-attention.md)。Older steel源码读点
已映射，M4实际数值只签原选中的NAX/向量路径，不外推其它硬件。五owner仍AUDITED，
新的错误/unknown/accepted-prefix风险、三交付/逐头、宿主风险和不可变同合同P1未验，
不能由source数学代签CANDIDATE/CLOSED或全goal；整个G01–G10仍0/10。

共同controller后续75885aaa5修正page准备接纳顺序：新冷Program/Invocation保临时，
upload.reset成功后才替换旧native，拒绝reset不留下旧prepared layout/table配新recipe。
不改reference顺序/API/layout。actual Air两route新短页consumer和原reference各build/run
全PASS，累计14.878s，见[接纳顺序阶段](../../../../evidence/correctness/inference-g01-native-page-admission-order-stage/README.md)。
原长域math patch不变，未重复已通过的长域。此回归未注入reset拒绝，也不代签全部新
page错误/未知/accepted-prefix风险、三交付/逐头/P1；五owner保持AUDITED，完整goal继续。

本轮risk的真实新事实：all-Q0/page-refs0 layout的schema/metadata有效，但PagedArena.read
按原非空page permission合同拒绝空集合，新native空输出分支无可达公开调用。NP02/05在
仍AUDITED时冻结零域接纳：仅Q rows与page refs均零时，typed page bind接受两个默认空
frame；私有native Program只接真Q容量port并输出原空view，不制造虚假K/V root或dummy
大tensor。common recipe四description/单table/banks报告不变。非空域仍要求完整同代
exact frames；原PagedArena非空合同、reference/Graph/CPU消费者不改，不扩大其它owner。
这是有限FIX的空消费与权限边界修正，必须由实际source/原负空间验证，不能跳过本失败。
