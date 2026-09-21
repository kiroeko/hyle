# oneDNN 同实现 A/A 差距：用户要求的后续原因调查

- owner：`/root`；2026-09-21用户明确要求“后面还是要查清楚为什么差距这么大”。
- 当前安排：2026-09-21用户追问后已启动现有证据/源码的有界复核；首次复核结果如下。
  随后完成两组固定诊断共八进程，已复现10%量级的运行时计时污染；历史唯一归因仍未证明。
  产品、正式benchmark和公共仪器未改。用户已接受已证实机制与历史限制，批准恢复M2；
  随后明确允许按工程证据翻案，当前裁决见末节，旧Q2不再是产品FAIL或接入阻塞。
- 输入：[Q2原终局](llm-moe-c1-normalization-aa-qualification.md#q2-终局blocked)与
  [原始证据](../../../../evidence/performance/onednn-normalization-aa-q2/README.md)。17×257 Direct
  LayerNorm两个角色实际均调用相同oneDNN/HYLE DirectRoute；吞吐比0.906388/倒数1.103281
  违反10% gross门，正式产品性能样本0。原日志接线修复不解释或消除这个数值差距。
- 目标：用可核验的实例证据定位差距来自调用责任不等价、计时/顺序、JIT/地址、运行环境等
  哪个具体因素；不能把假说当根因，不能据此判断HYLE自研算子输赢。
- 首次源码/现有证据复核上限30分钟；只查这项差距，不扩成Sched/oneDNN/公共仪器重审。
  若原记录不足，先写明确假说、观测缺口与一个有限诊断bundle，经独立SCOPED/AUDITED后
  才能增加观测。旧Q2完整重跑额度仍为0，不以改名/等空闲/调功耗来重复原实验取绿。
- 原门限、失败记录保持；不默认接受例外，不在这里裁决/优化/删除冻结数学。必要的新产品
  修正只有具名反例且对应owner依法审计后才允许；未到M5不进行自研性能比较。
- 交付：原因、证据、可复现方式与影响；若30分钟内尚不能证明，明确缺失的观测与下一项
  有限验证，不写“已查清”，不继续无预算研究。不能把其他新算子验收通过作为此项解释。
- 最终goal收口必须报告此项实际结论与未决限制，不能遗忘用户要求。M5开始前仍须停下汇报。

## 首次复核：差距真实，具体原因尚未证明

本次复核计20/30分钟，停止扩大现有源码调查；余10分钟用于下述诊断的有限接法审计，不重新
起算。只分析原包的八个`selected`进程，即oneDNN与自身比较；未研究或计时冻结Native数学。
分析脚本、原包hash/CRC、逐进程实例和比值见
[现有证据分析](../../../../evidence/performance/onednn-aa-cause-followup/README.md)。

| 已核对事实 | 能说明什么 |
|---|---|
| 17×257 Direct LayerNorm的八个比值中一个为0.906388，另七个在0.981523–1.020131；八个中位0.995955 | 没有证明此形状普遍慢10%；不能用总体中位数抹去失败进程 |
| 失败进程两实例都经同一DirectRoute、oneDNN provider、submit/wait；共享输入，独立output和392字节workspace | 已排除一个调用HYLE数学、一个调用oneDNN的“苹果对橘子”；A/A中旧显示名称`HYLE Native`具有误导性，但provider记录明确 |
| 两份相应JIT dump均3039字节且SHA256相同，八进程均如此 | 未发现生成不同机器码；dump不能给出每个角色实际JIT入口地址 |
| 仪器使用共同迭代数及ABBA/BAAB配对；失败表的两臂误差均0.3%，配对比约0.904、区间0.883–0.912 | 不是固定先A后B，也不能认定只是某一个离群样本；进程内存在较稳定偏差 |
| 失败配置的output/workspace低地址偏移在三个正常配置中也出现 | 简单低位对齐本身解释不了差距；实际代码/数据placement与缓存交互仍未排除 |
| caller始末均CPU2，亲和0x4；仅有整个比较的粗粒度thread CPU时间 | 未发现记录中的跨核，但不足以排除臂间干扰或归咎Sched |

资格脚本取两臂吞吐中位数之比0.906388；表格的0.904来自配对log-ratio中位数，两者统计量
不同，均触犯此例原门。不能把它解释成四舍五入或修一下parser就能恢复资格。独立的
`{output}`路径接线错误也不解释数值差距。

当前较值得验证的是**实例/代码/数据布局效应或该进程中的运行环境干扰**，这只是收窄后的
假说。旧日志丢弃了CompareResult已有的逐轮数据，也没有role→实际JIT入口或逐臂cycle/
context-switch记录，无法从汇总表唯一归因。原因结论仍INCONCLUSIVE，旧Q2仍FAIL，正式
产品性能样本仍0；不裁决任何HYLE算子的输赢。

## 后续诊断 SCOPED：单一形状、同实例控制与角色交换

这是独立benchmark/evidence范围，owner `/root`；不改变两个oneDNN产品owner的新能力scope。
先在剩余10分钟内核对复用源/库身份和记录接口，冻结以下有限FIX并登记AUDITED，才构建或
运行诊断。若现有接缝不能承载所需观测，就写具体缺口，不扩建公共仪器。

- 只用17×257 F32 Direct LayerNorm、oneDNN成熟实现、相同输入与完整submit/wait。保留两个
  实例A/B；两臂用相同类型的调用封装，记录实际provider、对象/output/workspace身份。
- 一个固定bundle最多4个新进程。每进程固定六组比较：A/A、B/B、A/B、B/A、A/A、B/B，
  后两个进程反转中间两组顺序；每组12个配对round，minEpochTime沿用8ms，不追加采样取绿。
  A/B与B/A交换同一对象对应的角色，不能重新构造对象冒充交换；配置在首个进程前固定。
- 保存现有CompareResult的每轮elapsed/iterations和配对统计，不把新查询放入算子热调用，
  不改nanobench/公共性能核心。原始JIT dumps、粗粒度CPU/亲和/实例事实继续记录，明确它们
  无法替代实际JIT入口和逐臂干扰观测。
- Windows锁原P核、低并行构建、DLL邻置、禁用系统错误弹窗，无UAC/功耗调整。重型验证串行。
  首个执行错误即停；正常完成固定bundle后立即裁决，不复跑原Q2的全形状/全部因子矩阵。
- 若偏差随对象移动，记录实例相关证据；若只随角色/位置移动，记录测量接线相关证据；若未
  复现，明确只排除了本bundle内的稳定偏差，不能声称已解释历史失败。新证据不自动解除旧FAIL。

本节尚未AUDITED、未运行。用户随后明确要求“必须查明这个之后继续”：从此检查点起先停下
其它M2接入，集中执行本项有限诊断和证据驱动定位；不再把原因调查挂到后续能力之后。
最后必须给出证据支持的原因或具体尚缺的关键观测，不能把“调查过”勾成“已查清”。

## AUDITED：复用原调用和现有静态 SDK 的单形状诊断

已核对CompareResult公开接口能在compare返回后导出每轮elapsed/iterations，无需改变
nanobench。原normalization benchmark、perf_process及nanobench相对Q2的`0e107a06`未改。
锁定oneDNN provider/CPU backend实现也未改；公共Compute准备设施后来有变，因此不把当前
HEAD重编译冒充历史二进制。原Q2可执行文件身份没有找到，现有同版本静态SDK可作诊断对照，
其文件hash逐一记录；新实验只定位可复现机制，不声称逐字节重放历史进程。

冻结D-A1..3，源码/设计累计28/30分钟，余2分钟；实现、构建与规定bundle执行单列：

1. D-A1：独立诊断TU以原benchmark为输入，复用Problem、DirectRoute、verify和冷observe；
   只执行`DirectRoute(problem, true)`，不调用Native或Graph。两臂用同一个BoundRoute类型。
   宿主main明确选择17×257 LayerNorm，六组比较及12round/8ms按上节固定。
2. D-A2：四个进程在创建A/B时分别采用AB、BA、AB、BA顺序；后两个进程将中间的A/B、B/A
   两组顺序反转。调用对象不重新构造，前后A/A、B/B用于识别角色偏差/时段变化。每组前后独立
   oracle验证；只在冷边界记录CPU、thread time和对象，逐轮结果由已有public accessor导出。
3. D-A3：在任务scratch建独立CMake consumer，`find_package(Hyle)`使用既有
   `C:/hyle-scratch/active-cpu/aa/selected-graph/static-sdk-relocated`；其Release、MSVC14.50、
   STATIC ON、oneDNN THREADPOOL/cache OFF/concurrent ON配置已核对。静态SDK只读，不触发
   原产品构建；诊断代码、纳入源、SDK头/库、产物与DLL全部hash绑定并在运行前固定。

本诊断不新增产品target、不改正式benchmark/公共仪器或任何自研数学；scratch源及配置随
证据归档。运行器固定进程亲和0x4/CPU2和AboveNormal、SetErrorMode(0x8003)、DLL邻置，
串行至多四进程，每进程30秒timeout。失败或预算用尽即停止并保留原始记录。

### 诊断首进程的身份检查修正

独立consumer构建成功；首进程六组正确性/测量已完成，接着因诊断脚本要求JIT跨可执行文件的
完整SHA相同而停止。已查明3039字节代码仅offset89..96的一个绝对地址不同：`49 bb imm64`
把oneDNN静态tail mask地址装入r11，后续vmovups加载该表；非地址字节全部相同。链接map将
新指针定位到`jit_io_helper_t<Ymm>::prepare_vmm_mask`的`mask_f32+24`，PE中实际32字节为
一个全1 lane与七个零lane，符合257的尾1元素。上游`src/cpu/x64/utils/jit_io_helper.cpp`相符。
这是新可执行文件/ASLR下的地址重定位，并非不同数学；首进程的两实例原始dump完全相同。

仅修正任务诊断的身份判定：保留原始dump/hash，另外检查固定mov指令、唯一8字节地址位置、
PE只读表值及其余代码逐字节相同。原脚本和失败收据不覆盖；单独resume脚本重解析首进程，
继续尚未执行的三个固定进程。诊断EXE/源/SDK不改、不重新采样首进程、不增四进程预算。
首次源码/设计额度至此30/30分钟，后续只执行已固定的观测；结果不足则明确具体缺口并按
协议重新定界，不能默默追加来源调查或整套资格重跑。

## 首个诊断终局与具体剩余缺口

固定四进程全部完成，24组/每组12配对round，输入身份未变，正确性通过；详见
[实例诊断收据](../../../../evidence/performance/onednn-aa-cause-followup/instance-diagnostic.json)。
两个实例交换的八个比值为0.991656–1.000919，没有复现固定10%实例差。同一实例对自身的
16个比值为0.971892–1.039048，15个区间包含1；3.9%只是点值，不能当稳定偏差或旧FAIL根因。
首进程身份检查修正只重解析既有数据，四进程中没有任何重跑。

现存旧benchmark的link map还显示四个Direct lambda thunk全部合并到0x140004a10，不能继续
仅凭lambda源代码不同猜测不同热代码。该现存EXE hash为`f4c6232c…`，不是Q2的`e3def077…`，
故这条只作旁证，未冒充旧Q2的逐字节证据。

首次30分钟源码/设计及一个四进程bundle均已到停止点：原因INCONCLUSIVE、旧FAIL保留。
具体剩余缺口是原失败前14个Direct/Graph构造/释放产生的布局没有还原，同时缺少每臂细分的
wall和线程cycle观测。不能把这次未复现当已解释，也不恢复其它M2接入。

## 第二段 SCOPED：前置布局与执行时段的分离

用户明确要求查明后继续；按上述新观测定界下一段，原30分钟不回填、不重置。本段源码/
接法上限20分钟，只审计两个变量：原版调用的前置分配历史、当前线程实际cycle与wall的关系。
最多一个四进程bundle，仍只有17×257被计时；不新增全形状资格或自研比较，不修改公共仪器。

- 前置布局：使用原Problem/DirectRoute/GraphRoute且只选oneDNN，构造原17×257之前的14个
  region并执行必要的原正确性预热，随后释放；不对前置形状采样或裁决性能。与无此前缀的
  两个进程成对保留，记录地址而非假定布局已相同。
- 细分观测：每个诊断operation批量执行256次原Direct run；只在批量边界读取QPC和
  QueryThreadCycleTime，保留顺序和角色，既有nanobench仍负责配对。新增观测开销双臂相同，
  不当作正式单次调用性能；不把cycle强行换算为秒或直接声称测得CPU频率。
- 原因裁决：检查偏差是否随前缀/对象布局出现，还是随采样时段与线程实际执行量分离；同实例
  控制和对象交换继续存在。前缀没有复现原地址时明确标注；任何未复现都不能写“已查明”。

先AUDITED并固定实现/运行组合后才新增观测；具名接线错误最多一次修正，正常完成即裁决。
API依据为Microsoft [QueryThreadCycleTime](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)：
线程cycle含user/kernel执行，时钟实现由CPU决定，不能转换成elapsed time。只查询自身线程，
不申请额外权限、不启用系统级trace、不弹UAC。

### 第二段 AUDITED：固定布局前缀和批量窗口观测

已有公开调用、SDK及第一段collector接缝已核对；本段源码/接法累计6/20分钟，余14分钟。
冻结D-B1..3，只新增独立scratch诊断consumer，产品和统一统计核心保持不变：

1. D-B1：四进程前缀顺序为无、有、有、无；A/B创建顺序为AB、BA、AB、BA。前缀的14个
   oneDNN region各实例执行64次及原独立oracle，不采样；随后只计时17×257 Direct。
   每进程仍六组A/A、B/B、A/B、B/A、A/A、B/B（奇数进程交换中间两组），12round/8ms。
2. D-B2：统一WindowOp每次执行256个原Direct run，双臂只差绑定对象；预分配观测空间，
   批量边界读取QPC、QueryThreadCycleTime、RDTSCP及CPU/AUX，不改内层调用。通过原
   CompareResult共同迭代数重建末尾24个实测epoch，明确剔除校准窗口，不丢弃慢样本。
   RDTSCP先检查CPUID支持；额外固定一次5ms busy与20ms Sleep观测两计数器关系，只作
   当前机器的可解释性控制，不转换cycle为秒，不据此调功耗。AUX不当作跨系统通用CPU ID。
3. D-B3：独立build和不可变输入hash；复用已审计静态SDK、DLL邻置/CPU2/无UAC设置。JIT按
   第一段已证明的唯一绝对mask地址检查，保留原始dump；前缀进程只取最后两个对应内核做
   同形状身份核对。所有源、二进制身份、原始逐批窗口/round、地址和失败原样归档。

RDTSCP语义按Microsoft [intrinsic文档](https://learn.microsoft.com/en-us/cpp/intrinsics/rdtscp?view=msvc-170)
核对。计时observer自身成本对称但非零，本段只用于因果诊断，不代签Q2或任何算子性能PASS。

## 第二段终局：污染机制已证实，旧样本唯一归因未决

四进程、24组、288对实测round已完成，独立正确性与不可变输入通过，所有实测窗口在CPU2。
前缀恢复了原失败input/output/workspace的低22位相对布局，仍没有固定10%实例中位差。
观测脚本有一次具名修正：前置Graph沿用`onednn-prefix`标签，初稿误要求全部为`onednn`；
修正仅接受前置Graph该标签，并修正空AssertionError被当作假值的退出接线。首两个进程只
重解析，剩余两个进程只运行一次，原始失败与脚本保留；EXE/库/调用源均未改。

已捕获可解释的反例：`run-1/post-AA/round-5`比较**同一个实例**，wall每次约2177.57ns和
1971.99ns，差10.425%；线程运行tick只差0.0454%，运行tick/总TSC分别83.77%和92.46%。
相同对象的主要差值来自未记账为该线程运行的时间。全288对中有17对wall相差超过9%而线程
运行tick差不足1.5%。这是实际发生的计时污染，并非注入Sleep到算子或仅凭“可能有噪声”猜测。

指标解释已核对Microsoft [x64计数器说明](https://devblogs.microsoft.com/oldnewthing/20160429-00/?p=93385)：
QueryThreadCycleTime采用RDTSC；当前机器invariant TSC与busy/Sleep控制相符。它不是PMU
指令/核心cycle计数，不把差0.0454%误称为“指令数一样”，也不把线程tick替换正式wall性能。
完整来源、逐批/逐轮记录、地址、校准与实测分界见
[窗口诊断收据](../../../../evidence/performance/onednn-aa-cause-followup/window-diagnostic.json)。

**本次不能证明旧Q2的0.906388聚合值唯一由这个机制引起**。旧记录没有逐轮线程计数，原始
EXE未保留；物理页/cache映射和实例metadata地址也不能回到历史进程。再次采样不能补回当时
丢失的观测。不能点名某个后台任务/驱动，不能据此判oneDNN慢或HYLE Sched有错。

第一段源码/设计30/30分钟及四进程预算已用完；第二段源码/接法累计12/20分钟，但固定四
进程bundle已到正常停止点，不追加第三轮取绿。结论为机制CONFIRMED、历史唯一归因
INCONCLUSIVE；旧资格FAIL、正式产品性能INCONCLUSIVE/样本0均保持。
以上是提交用户裁决前的诊断边界；当时先停止其它M2接入，没有把唯一历史原因写成已证明。

## 当前工程裁决：旧样本无效，恢复 M2

2026-09-21用户先接受“已证实机制与历史限制，继续M2”，随后明确：若大概率是测试环境问题、
没有复现，可以翻案，不能继续视为工程FAIL。本轮据此裁决：**旧Q2样本不具备性能裁决资格，
按环境干扰最可能的无效测量处理；撤销其产品失败含义及后续接入阻塞。**

依据为相同Provider/相同JIT、七个其它原配置正常、两组固定诊断未复现稳定10%聚合差，以及
同一个物理实例实际出现10.425%墙钟差而线程运行tick只差0.0454%。这是有证据的工程判断，
不是对历史进程的唯一归因证明。原始FAIL日志保持不变，当前结论不写性能PASS、不新增产品
性能样本、不放宽统计门、不启动Q3或追加原因调查。调查任务至此结束；两个oneDNN owner
按独立M2卡继续资格化和批量接入，所有新能力仍履行正常验收，M5前停止点保持。
