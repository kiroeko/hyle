# CUDA 性能优选补齐与对应清退

<!-- MODULE REVIEW: CANDIDATE -->

baseline `3e9e71d0ab6a915d3682314e9102864638d1005e`，Windows工作树干净。
2026-10-08用户要求CUDA补至CPU同级，并明确“差不多就选主流，不无限性能比较”。
旧goal完成只代表覆盖/交付/工程默认；本轮重新纳入实测优选与对应清退，旧AA FAIL不改写。

## 四个有限工作包

| 包 | 权重 | 完成条件 |
|---|---:|---|
| C1 同合同候选与入口排查 | 20% | 全部实际重叠的能力/精度/资源/完成账单；一次具名AA成本排查，冻结有限矩阵和纠正 |
| C2 批量资格/比较与裁决 | 35% | 单承载不比，合格同合同请求一次固定n15；明显差距定winner，接近/噪声选成熟，不升级n/反复换姿势 |
| C3 冷规则及冗余清退 | 30% | 合格赢家有精确条件规则；无规则成熟默认仍可用；可完整替代且无优势的self物理删除，unique明确保留 |
| C4 差分交付与终审 | 15% | 受影响Direct/Graph/source/static/shared/失配/资源检查及七门、LOC、垃圾清理、提交推送后停止 |

当前consumer/需求目录与U23原数学和交付保持；不重新下载模型/重跑559合同，不改CPU或其它GPU。
scope owner候选仅compute_route_cuda、graph_route_cuda、compute_provider_cuda，原其它成熟库
只有具体接线反例才重开。它们旧CLOSED不授权本轮产品变化，先绑定SCOPED再AUDITED有限FIX。

## CPU 相同的快速工程取舍

- 支持/接线/原precision与全成本先成立；不同编码、舍入、完整region、ownership不伪同合同。
- 单成熟carrier只正确性/资源/P0，不造性能比赛。
- 同合同明显、合格且可复现的优势才能成为WIN冷规则；近似/不确定直接选成熟工程默认。
- 不为保住self无限优化；有完整成熟替代而无明确完整优势就清退。SIMD和路线控制/资源/完成
  与原借入库不误删；unique要有具名consumer/标准需求，不能只靠旧自研测试反证新设计。
- “同CPU级别”指真实候选审查、合格比较和规则/去留，不要求复制43条WIN或每个硬件都冠军。
  不能把仅工程默认标成全域best-of，也不能为增加WIN数扩算子或改用户形状。

## 有界执行和停止

C1调查上限30分钟：一遍已有失败/源码/原成本检查、一个明确机制诊断，不重测原8轮/升级样本。
C2适配器一次批量构建（至多一次具名setup纠正），一次短AA资格批及一批适用n15/四strata，
不升级31。AA未过的单项不进入正式采样，原FAIL保留；不能因少数无资格项阻塞已资格项。
模块adapter只映射workload/遥测，统计仍统一hyle-perf；公共工具如有真实必需缺陷另建工具卡。
首次四小时内必须有C1有限矩阵、批量准入和C2裁决产物；到预算按原门裁决，不继续研究。
源代码稳定再采样；批量接线/规则后集中验证，复用既有producer和最新U5 SDK，Windows DLL邻exe，
所有子进程隐藏、build2/test1、重型串行。文档/状态不build/P1。每包结束/重大变化报告实际结果
及统一Compute+Graph实现LOC；[skip ci]提交推送，不消耗Actions。

## 当前具体反例与预审

上轮adapter只有两次verify热身，nanobench warmup=0、minEpoch8ms、约12 paired round；
完整resident叶操作约10--30微秒，原日志同实现err达41.4%，包括没有ggml复杂管理的原native。
这些是已知入口事实，**不足以唯一归因GPU/系统噪声**，不应直接转成永久无比较默认。
先看原compare循环与native event/Completion，具名检验冷缓存/时钟稳定和CPU未运行成本，
完整终态等待仍收费，不能用纯CUDA event kernel时间替代完整consumer以获得漂亮比值。
本轮只在具体机制依据上纠正adapter；不提高margin、改pair/尺寸、拿更多n掩盖偏差。

完整C1审计及有限FIX前不改产品、不正式采样。三个本轮owner已依法CLOSED→SCOPED；其余CUDA owners保持旧CLOSED。

C1一次入口诊断已完成，见[原始结果](../../../../evidence/performance/cuda-best-of-c1/README.md)。
2048对称预热不足以消除同路径11/30us和7.4%差异，不反复增加预热/round或升级n。
已证adapter缺少D5已有的AA ready guard，补此窄guard，公共runner/统计不改；其验证待执行。

C1已冻结[206个有限请求](cuda-best-of-requests.json)的候选静态矩阵：32个语义variant、
三种实际float storage、主decode/小prefill holdout、6种storage cast和2个plain Gather。
这是8个重叠能力域的有界批量准入输入，不签206项已支持/已比较；每项仍须真实prepare与输出
门，单承载/没有同合同的域明确不比。Linux attention独立同合同region，Windows不伪造该Provider。

批量admission预编译发现adapter误用显式Operation构造，旧日志保留；只修adapter为既有
显式typed Operation，不改API或产品。初始诊断build加批量build两次已用，允许原同ID最后
一次这项具名纠正（累计3次），不再加预算/新ID；product/P1预算不变。

C1首次冷准入在ggml caller-workspace检查停止：zero-byte BufferView仍有owner，
不能传给provider-owned workspace；只修adapter在size0用默认空参数。首次HYLE Copy prepared
不代签206项，原FAIL/三次build预算保留，不启动第四次C1 build；C2冻结完整slot/输出批次后
统一编译/验收此必要接线及全部请求。C1明确交接此未验入口，不假签C1运行资格完成。

## C1 审计冻结与 C2 批量验收

C1机制调查停止于原四诊断进程，未唯一归因的噪声不再循环研究。源合同和206请求有限矩阵
固定，C2一次批量adapter完成零workspace正确绑定、原native slot、真实device payload/完整
terminal输出核对，不能从prepare声明签数学资格。每项五候选实际检出，包括原Torch只接
stable Softplus、SGLang仅相应vector/norm、ggml明确支持域与CCCL Gather。

预冻结数值：两次seed5/17变化、列/行非均匀dyadic数据；norm gamma/beta、additive mask及
重复Gather IDs显式端口。独立Double方程、逐输出核对，原本库F32/F16/BF16已有测试口径
分别3e-5/0.001/0.008乘(1+abs(reference))；不把这个窄有限域外推极端值/全部public合同。
独有资源/特殊值资格按旧原测试复用。失败项具名隔离且保留，不使未运行项变PASS。

有限产品FIX冻结：R1依合格完整候选/成本裁决做私有cold规则，公共API/对象/语义和原热程序
保持；R2按真实完整替代删除冗余self数学与对应默认，独有消费明确KEEP，不扩self/优化；
R3相同两入口/资源/失配和source/static/shared差分验收。没有已测赢家的请求只工程默认，
但仍必须完成本矩阵适用审查/准入/有预算裁决，不能用文档/default假签所有性能补齐。
本AUDITED记录独立提交后才允许上述三owner产品变化；C2不改任何成熟数学或公共instrument。
C2只用现adapter/SDK一次batch build，至多一次具名纠正（独立完整payload batch，不重置旧
C1三次build/首次冷准入FAIL）。AA资格与P1之后只取实际CORRECT且同合同的候选组合。

C2批量准入/输出及独立pair-slot验证完成，见[原资格](../../../../evidence/performance/cuda-best-of-c2/README.md)：
206请求/五provider1030观察，408完整输出通过、622明确decline，零数值失败、零无承载；
192多候选/14单承载，对应212 pair program两侧各两slot/变输入均通过。Plain Gather原ggml
仅准入encoded embedding，所以这两个plain请求CCCL单承载，不能伪两家竞速。单项未做AA/P1
不签性能PASS；下一步在这固定212 pair上一次短AA资格，只对合格子集正式n15，不重跑本数学门。

C2原8进程AA批次完成：212 pair中47合格/165未资格，原误差/偏差保留；
正式一次n15仅取47，165不再采样/调参数而按用户要求成熟工程默认。使用原AA image/全部
邻exe DLL逐字复制到外部仪器项目，不能重新编译/重调数学。qualified TSV每条行与原212
对应行逐字相同，只去掉独立、已完全settled的非资格请求；每项Context/两slot/数据/
2048对称warm/compare/完成边界不变，所以不是逐pair subprocess或其它生命周期改动。
本subset只改变batch的非计费外侧项目数量，不继承不同shape/recipe/algorithm的资格。
原全212 TSV及完整AA日志封存，新47 TSV/声明哈希受validator检查，不冒称旧完整file SHA相同。

## C2 终态与 C3 有限规则

一次n15/四strata已完成，仪器qualification及canonical manifest verify PASS；47 pair为
25 WIN、2 LOSE、20 INCONCLUSIVE，聚合performance FAIL原样保留，不能因筛选赢家改成全批PASS。
206请求逐项覆盖完整候选后只有23个native完整赢家；另2个weighted RMS pair WIN缺少第三候选
资格，不导入完整winner。169请求成熟ENGINEERING_DEFAULT、14单承载；165个AA未资格pair
不再采样、升级n或调shape。原日志、逐项裁决、完整候选清单见[C2证据](../../../../evidence/performance/cuda-best-of-c2/README.md)。

R1冷规则只导入23个完整赢家的精确算子/端口dtype/shape/参数/资源及测量环境；失配按各算子
成熟默认，不能拿部分WIN外推。native context冷包装只保存一次环境资格并向原prepare/drain
转交原context；prepared执行表保持。ggml带Graph region扩展的context不包装，不破坏原region
所有权/融合。Graph完整region优先于仅叶操作有证的选择；没有完整region优势不拆片段。
R2不再优化或计时自研：按实际完整替代域删除冗余数学，独有语义/资源域具名保留；若整项删除
需要撤回有效独有能力，先形成具体可审阅撤回方案，不能套用旧CPU MatMul决定。
C4只对规则/清退影响运行两OS source/static/shared与原资源/失配/region消费者，不重跑未变
559模型数学或再做P1。当前C1/C2完成，整体约55%；C3/C4尚未完成，不冒称CUDA收口。

R1资源审计：私有proposal当前每Provider实例一个候选，描述符阶段没有实际resource。
因此不把已测device/256对齐规则外推managed或低对齐资源：同一个原native实现另登记一个
受条件约束的冷候选实例，原工程fallback实例保留。独立instance是现有Context支持的组合，
不是新executor/API/数学；仅qualified实例限制device属性/256对齐并使用实测policy，失配由
原实例保持完整能力。原ggml context/region扩展不改。只增加compute_route_cuda的私有
platform/platform_thread依赖，借用现有CPU身份/调用线程亲和查询；系统优先级只在冷准备
校验，执行不查询。CMake只在原static Release/MSVC/CUDA89-real/无LTO/原Provider配方允许
该候选，shared/WSL/未知配方沿原默认。不复制环境探测或公共路由/治理系统。

R2最终源级清单：tensor/normalization中的23胜出请求保留原kernel体、grid、数学和完成表；
未胜出且成熟库承载普通dense域的重复AOT specialization清退。tensor删除41个numeric dense
实例和4个Copy dense实例；Reduce Sum/Mean与普通/加性mask Softmax删除18个dense实例；
Concat删除16个dense实例。它们共79个重复程序，**不是79个算子，也不是整项数学删除**。
原generic程序覆盖仍有效的stride/广播/低对齐/no-private/capture等独有域；唯一Relu、
number Min/Max、NaN extrema、boolean/log Softmax、checked Scatter，以及borrowed激活尾部/
separate GELU保留。Norm各共享数学体已有完整胜出或独有mixed-parameter资源域，不复制优化。
不把旧expert显式native当普通默认；169默认域继续优先合格成熟实现。Native版本递增到v12，
23胜出kernel的兼容依据为未变源码/编译配方与实际device code对照，不重采性能。若对照出现
真实winner body变化，不能继承该项WIN；先具名修正或撤掉该规则，不追加整批P1。

## C3 当前结果与 C4 进行中

candidate `bccdf231`：23完整选择已实现，native测量实例与原fallback共享同一数学/执行表；
精确配方/硬件/调用线程/端口/shape及device256资源条件在冷路径绑定。GPU代码669→590，
净减少79个重复程序（95旧symbol去掉、16generic Concat改名）；207个保留native同名程序
SASS逐指令一致，未变成熟程序/模型算子不重采。不是删除79个算子，也不是全体自研删除。

Windows source首次数学/资源25、普通规则5、受影响Graph4与完整region3均PASS。static原29项
最后批次28 PASS/1 managed FAIL原样保留，23实测规则实际命中及变输入输出PASS。新增真实
冷选择缺陷按[独立Core卡](compute-bound-view-alignment-cuda-best-of.md)修复：Unimplemented
资源失配继续下个候选，已知view offset冷过滤，descriptor后换违规view拒绝。FIX-C第一
窄检查只签普通默认；最终正确profile下2项（managed双输入/完整输出/Completion，偏移及
descriptor违规绑定）PASS，源码与配方断言证实测量候选确实激活。原29项没有第三次。
Windows static普通Direct/Graph独立消费者PASS。其余Windows shared与WSL source/static/shared
正按38步差分批次串行运行；Core原单元门、证据归档、文档/终审/垃圾清理仍待，不签CLOSED。

## C4 验收与候选冻结

固定产品/测试候选 `cd53173946564155121819f1ccae7f6726b908ed`，之后只文档/证据维护。
Win shared、WSL static/shared各29数学/资源/默认检查及普通Direct/Graph独立消费者PASS；
WSL source25/6/7 PASS。Core原63单元在Win/WSL各PASS，FIX-C严格profile两项PASS，
原static29 FAIL原样保留且没有第三次，23规则及28未变域证据按冷热源码/合同复用。
当前[终审证据](../../../../evidence/correctness/cuda-best-of-c3/README.md)绑定原失败、语义
组件证明与完整归档；两OS/source/static/shared实际交付成立，不冒称原FAIL变PASS。

| 门 | 当前实际裁决 |
|---|---|
| C++/正确性 | Core两OS各63，原数学与23变输入完整输出、active规则、negative/FIX-C PASS |
| API/自然接口 | 普通Direct/Graph不传provider排序/timing；规则按算子/端口/资源/shape冷匹配；融合region保持 |
| 三交付 | Win/WSL source/static/shared、SDK relocation/loader/邻exe DLL及真实两入口 PASS |
| 依赖/构建 | 中央只增加两个已有闭包内private直接边；全部产品根/公共头/传递闭包相同，真实producer/SDK两OS PASS |
| 性能 | C2 qualification PASS；25 WIN/2 LOSE/20未裁决聚合FAIL保留。只导入23完整赢家；169默认/14单承载，不追加采样 |
| 文档/易用性 | 当前限制与入口文档更新；docs/closure待本候选冻结后最后检查，无新构建/P1 |
| 红队/生命周期 | managed/offset/descriptor换非法view拒绝、两slot/变输入、零热分配、关闭/失败及原特殊值资源域检查 PASS |

统一Compute+Graph非测试实现 **61,100行/258文件**。已清理Win约10.72GiB、WSL约0.14GiB
当前owner临时消费者/首次失败SDK，保留source、最新SDK、模型、缓存和P1验证输入。原公共
Graph/Vulkan BLOCKED不变，其它路线/CPU旧状态不重开；本goal只关闭本轮CUDA规则/去留。
完整模型数学104GGUF/559合同等复用原U23/U5明确不变域，不因文档收口重新全跑。
