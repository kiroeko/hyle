# G01 Apple/Vulkan ragged 布局直接消费

状态：CANDIDATE，2026-10-13；owner `compute_route_apple`、`compute_route_vulkan`。
不可变candidate `280f6f92ea97bf3b424113346f0264fb7b20df16`。
baseline `9cfa7b50db652010da5d4d544da9ee01c10644bc`。授权来自当前十项goal；真实
reopen trigger为两request Q1/17、KV31/47、GQA8/2 D256的packed NHD消费，已有native
MaterializedBatchProgram按Q/head索引并物化历史，不能满足已准入G01结构目标。
原Apple F5与Vulkan 3d92有限CLOSED证据不变，原数学/shader/native Work不重开。

本轮只做已有host-owned ragged layout的直接消费，不代替G02精确页、G03 producer、
G04量化、G05额外语义或G06增长/decode规划复用；这些保留原总目标。主纵切为两变长
请求native SDPA和packed输出，反偏置为D64 noncausal、empty request与独立slots/旧输出。
第一资格域沿已有成熟承载：Apple F16/BF16、Vulkan F16，D64/128/256。其他原已准入
dtype/路线格仍待后续验收，不以本有限子批声称各模式对称或整个G01完成。

拟最终消费形状：`apple/vulkan::prepare_ragged_batch(Context, BatchLayoutVersion,
RaggedAttention, WorkloadContext)`冷返回原native Program；三inputs是schema固定
packed容量Q/K/V，单output是本layout实际packed Q rows。options只有scale/causal，不含
模型policy；布局版本/checked indptr只在冷阶段消费。Program/Invocation/native inputs/
Results沿原owner与actual retirement，run不扫描metadata、不建图或新Queue/controller。
布局变化调用显式冷准备；已有Program复用不受同内容新generation影响，G06另验规划费用。

拟有限FIX NR-01：共用已有BatchSchema/完整host layout校验，在各原route内按request
Slice→Reshape→原SDPA→packed输出Concat生成原Plan/MLX tracing callable。Q0跳过；
Q>0必须KV>0且causal KV>=Q。原View保持字节origin/strides；不生成Gather、GQA repeat、
全KV cont/Copy或scores。Concat和query转换只计O(QHD)，原mask仍O(Q×KV)。
NR-02：原native lowering/actual bound origin对齐冷资格与容量/tail、readonly及views反证，
不能因API名叫view就签KV零物化。NR-03：原三交付、OOM/unsupported/old result/close/
actual retirement与公开文档；无兼容wrapper、没有新factor/第三方/构建核心修改。

预声明：两请求独立double oracle、两dtype实际适用域、三depth、causal/noncausal、tail、
empty/reorder/foreign-invalid layout与资源、两slots/旧输出/原fail retirement；实际native
graph/dispatch搬运账证明KV不重复物化。source/relocated static/shared，Apple/Vulkan实际
GPU仅MacBook Air先查硬件/toolchain身份；Windows只定向编译。宿主ASan/UBSan/TSan
适用范围按原路线限制，GPU driver/private memory UNKNOWN不签0。有限同合同P1在稳定
candidate后冻结，不能借CPU或旧single P1代签multi。普通selector只对应factor/真实消费者。
调查/实现/验证各2h检查点，按用户常设授权追加保累计日志，Windows≤4jobs、重型串行。
当前CPU正式P1独立运行时不启动Air/Windows另一重型run，只允许定界/写canonical源码。

审计固定：参数直接复用已有RaggedAttention，capacity必须等于layout.schema.capacity，
scale有限；不新增common类型或改compute_batch。每request Q/K/V Slice axis0，再reshape
到BSHD{1,rows,heads,D}，原SDPA后reshape回NHD；原Concat axis0拼实际活跃rows。
所有Q空时原View slice0输出空packed张量，不做数学；foreign capacity/storage无效先拒。
Vulkan原prepare_plan已把View转为root+byte_offset别名，SDPA recipe只读取这些rank4
aliases；原resident input资格与实际origin检查仍在，没有推进未资格普通节点。Apple原
lower已有slice/reshape、BSHD transpose和MLX fast SDPA，没有新math/copy/gather公式。
两路线原Program都保调用方runtime/输入/旧Results实际寿命，layout只需冷成功消费，不
由待执行payload读出host metadata。engine重排/增长选择何时冷准备属G06后续组合成本。
两份短route组合只使用现有原子操作；平台分别降低到原Plan与MLX retained callable，
不为消除这段冷装配再新增公共recipe/controller类型。数学与ownership没有重复实现。
AUDITED后固定NR-01/02/03，实际数学/搬运/失败证据之前不签性能或CLOSED。

实现草稿已写：两原route冷组合入口，无新math/Queue/controller；Vulkan原Plan保View
alias/origin，Apple捕获owned layout给原tracing。新增共用实际native consumer测试三D/
两request+empty、重排、独立slots/旧结果、readonly/tail与foreign capacity拒绝；公开
batch头加入逐头集合。当前未编译、未运行、未签candidate，CPU独立P1仍占重型资源。
源码审阅修正输出拼接：收集全部active request后发出一次原多输入Concat，避免逐request
拼前缀的O(requests²×QHD)费用；原Vulkan大arity资格/直接分片Copy与MLX concatenate
成本仍须实跑，不新增拼接数学或shader。两请求主域算法与原SDPA不变。
新增8request输出/tail反偏置，source-only结构test通过既有private ProgramState读取
每个SDPA的真实external KV根与byte origin，必须仍指原capacity inputs 1/2，不能指
物化值。它不修改public诊断面，不等于warm/driver私有费用已过；需与实际math/retirement
和搬运门一起验收。所有新增test仍未编译/运行。

实际source阶段已[实跑并封存](../../../../evidence/correctness/inference-g01-native-ragged-stage/README.md)：
Air Mac16,12/10core16GiB/macOS27.0.1/AppleClang21.0.0，Apple两tests、Vulkan四tests
PASS。主数学F16与Apple BF16三D、noncausal/tail/readonly/empty/reorder/旧outputs，
zero metadata allocation及8request；Vk实际原KV roots/byte origins，真实managed math
前缀后拒绝原native Concat suffix，原pending/wait/无ready/error保留均PASS。
所有compile/runtime测试错误保原记录，修正测试错误没有改变原native数学/容差/失败
语义。产品两TU/公开头与2020canonical输入SHA核对；正式P1源git树不改。
进入不可变candidate后source/static/shared、适用宿主san与有限同合同P1继续，未跑项
不写PASS。保全部其它G01与G02/G03–G10，不能代签paged/growth/capture。

候选交付：Air原selected-graph static/shared trees复用，Apple与Vulkan各source＋实际
relocated static/shared的两public tests均PASS。安装工程只链接公开hyle::compute，
私有KV-root/失败结构tests只签source，未冒充SDK入口；测试scalar bit carriers随consumer
本地复制，不从SDK借未安装SIMD/private头。四prefix均staging实际改名为relocated，
原已安装single SDK和正式P1树不覆盖。宿主san和有限P1仍待，不提前CLOSED。

候选宿主风险：Air复用原F5 native-host六棵树，Apple ASan/UBSan/TSan各两tests、
Vulkan各四tests全部实际PASS；两新增产品batch TU、对应public测试以及Vk私有风险
测试真实compile commands的fsanitize flags核对。ASan detect_leaks=0按原平台限制，
不签LSan/shader/driver或vendor私有GPU内存；原native shader/runtime源码不改。
P1仍待，未继承single或CPU结果；Windows仅定向4jobs编译原Vulkan库，不运行本机GPU。
该候选[交付/宿主风险证据](../../../../evidence/correctness/inference-g01-native-ragged-candidate/README.md)
已封存四SDK、六san、四public installed header object compile与Windows affected library
compile PASS；原stage失败记录独立保留。native Vulkan私有比较器已实现并在Air预检
编译/oracle PASS：按request独立原full-F32 flash shader recipe映射到同一个原Program，
原F16 query/output适配、mask/优化/split-K承载保留，输出借原storage Concat，三实际
shape与原F16 bound2e-3/readonly/tail。它不调用candidate attention lower/math，尚未
正式资格/采样，不由诊断结果签performance PASS。Apple比较器及正式两平台门待续。

Apple比较器已实现：原MLX fast SDPA直接构图，不调用candidate lower；两边同原
Context.compile/Program/Invocation/两slots/F16 packed ready sources/checked completion/
单packed owning output。公用三shape/double oracle/readonly/tail预检已PASS；首次
编译opaque default构造与array默认构造失败、随后比较器borrow池早于slots销毁的
teardown崩溃均保留，改为原工厂直接构造/optional与显式先释放borrow者，不改产品。
当前公开MemoryResource inline default ctor暴露incomplete unique_ptr的实用接口反例
归原G10/交付接口复核，候选内不顺手重写其ABI；initializer消费现已正常。

正式前固定两合同：Apple1f8e811f、Vulkand9a54a85；实际Mac16,12/M1-MAC engineering
identity，F16/GQA8/2 Q1/17 KV1024/768、Q64/128长prefill、Q1/17 KV31/47 D64留出。
16 paired rounds/min1/4ms、双execution/allocation与placement、原A/A/controls/n15/31/
5%阈值不变。实际诊断process成本Apple2.361s、Vulkan2.454s；Vulkan短留出诊断出现
约13%差距，保留为待正式确认成本事实，不由未资格点值改API或签PASS。Apple原MLX
1f8e74e3真git peer已fetch且clean；正式配置核对所链接原fast SDPA/Metal shader源码
与该pin，原BFloat copy/compiler补丁身份另保。未开始正式采样，未签这两个性能门。

Apple正式已启动：fresh1在原MLX源码身份脚本错误路径配置FAIL（4.868s），原树/日志
保留；按真实git列表修私有脚本，fresh2 clean source c6d10f66同合同/产品280继续。
配置核对原SDPA和Metal kernel族均PASS，A/A16实际PASS。Apple fresh2现已TERMINAL：
845.543s、n15产品120/controls30，QUALIFIED、5%非劣PASS、原verify退出0；
[实际证据](../../../../evidence/performance/inference-g01-native-ragged-apple-m1-p1/README.md)
保三点方向。混合D256及D64方向INCONCLUSIVE；长prefill比0.9827–0.9894，稳定LOSS
约1–1.7%，在5%范围内但不改名为方向parity。原fresh1配置FAIL另保累计。
Vulkan同产品280、clean source c6d10f66、原合同d9a54a85已启动；fresh2后缀复用修正
后的共享adapter源，不制造一次不存在的Vulkan前置FAIL。实际PID58732、原同一任务
续跑，不因观察超时重启。只有这一重型任务；原selected native/hybrid直接消费者回归
在其终态后串行复用各source-parent工程，不能换源码home或触发全局构建。

2026-10-13进度检查：G01约12h，整个G01仍未完成。该有限NR轮现在仅剩Vulkan同合同
性能裁决、原直接消费者回归与一次候选终审；不再追加全局调研/重复CPU或Apple P1。
原已准入分页/联合权限在G02，持续metadata/growth在G06，producer与private预算分别
在G03/G10继续，不能把这些门伪签成此NR轮已通过，也不能在此轮顺手扩大FIX。

候选独立只读复检固定为新增两个公开入口和四产品文件，不重扫已关闭原math/runtime：
BatchSchema::version已检查容量/非负单调indptr/GQA/右对齐有效KV并拥有metadata；入口
再拒绝invalid Context/layout、capacity不等、非ragged、非finite scale与未资格dtype/D。
Apple retained callable捕获owned version而非caller spans，Vulkan只在冷plan消费host
metadata；warm仍是原Program，不引入第二个Queue/完成权威。全部request输出一次Concat，
全空Q用原空slice，O(QHD)转换/packed输出和原causal mask如实收费；实际KV root/offset
反证已PASS。source-only失败注入只签source，SDK只签public consumer；没有用单请求
或CPU性能代签native多请求。复检未定位此有限候选的新产品错误；Vulkan性能与直接
消费者回归仍待实际终态，不先签七门CLOSED。

原统一LOC工具绑定candidate280、两owner各自self-check PASS：全库code2,871,923，
自实现不含测试175,049；Apple模块实现4,262，Vulkan模块实现8,825。该统计描述候选
规模，不按行数折算goal完成百分比，不为文档证据提交重新统计或重跑产品性能。
