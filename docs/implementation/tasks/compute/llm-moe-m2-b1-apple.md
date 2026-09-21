# M2-B1 Apple：原版 MLX 张量、归一化与状态批量覆盖

## Closure card

- component / owner：`compute_route_apple`；`/root`。普通Graph是直接消费者，当前没有已证实需要
  修改`graph_route_apple`产品的事实，不自动重开它或ggml Metal/runtime。
- baseline：`50189a2b`，Windows canonical，工作树干净。当前阶段SCOPED，先审计后固定有限FIX。
- trigger：M0/M2的B1要求F01.1–3、F02.1–4、F03.1–4、F04.1–6、F19.1–4共21行由成熟
  实现承载；M1-A terminal明确允许新数值、shape或region需求重开。旧有限M1验收继续有效。
- 主纵切：embedding→norm/gate→residual/index/state，两个请求、动态索引、旧结果保活。
  正交纵切：无模型注解的broadcast/strided/scalar/empty张量和多输出Graph。
- 来源：已锁MLX0.32.2 `1f8e74e3`（MIT），必要时仅用已准入原版ggml Metal组合。
  不从库名推断完整语义，不重新下载已有锁定来源。
- scope：21行需求到原版MLX operation/compiled region的成批映射与资格；复用现有native
  array、Context线程/stream、Program/Invocation、checked结果和普通Direct/Graph。
  现有纯MLX调用可按身份复用；冻结的自有Metal/scatter/check数学不算成熟覆盖。
- non-goals：研究/修改/扩张/比较/删除任何冻结自研算法；新Metal kernel或同义改写；模型loader；
  新万能执行器；公共构建/性能/closure仪器重构；解除CPU oneDNN Q2；替其它路线签覆盖。
- API/依赖：保留自然Apple typed入口与MLX原生array/region，普通用户不写Provider glue；
  Compute不依赖Graph，不为逐节点管理拆坏原版融合。当前无新增公开API/依赖factor要求。
- 预算：本M2 Apple批源码/设计总120分钟，前置合同/接口定位计6分钟，余114分钟。
  独立于已经关闭的M1资源范围；同批后续子卡共享额度，不按算子续杯。每具名根因最多两次
  源码驱动修正；正常构建/预声明验证单列，到预算固定具体结论或阻塞，不无限研究。
- 性能：先P0账单；API/算法稳定的不可变候选才启动一次必要P1，对照同一原版成熟完整调用，
  包括转换、同步、融合、checked终态与结果寿命。5%非劣门；A/A和双顺序先资格，15对起，
  合法总上限31对。不测试冻结实现的输赢，Mini不代签Air Formal profile。
- stop：21行及全部适用门完成后关闭；不兼容域保留原版反例与具名成熟后备责任，不缩小分母。
  不把接口未接通改名为算法缺口。完整goal继续；M5开始前暂停等待用户。

## 待审有限改动集

| ID | 允许范围（审计冻结后才实施） | 必须证明 |
|---|---|---|
| A1 | 21行能力表：复用原版MLX tensor/arithmetic/norm/region；按需要接原版indexed/reduction/sort及合法组合 | dtype/rank/stride/empty/axis、特殊值/舍入、ties/NaN、重复索引和checked溢出，不能进入自研数学 |
| A2 | 既有冷lowering与原版MLX检查结果接线；只作格式/布局/端口、错误语义与成熟原子的组合 | 原版计算在GPU；只读回必要验证摘要；坏索引/溢出不发布成功结果或推进后段state |
| A3 | 普通Direct/Graph默认成熟选择、compiled region与每请求原生owner | 两slot、旧结果、变化输入、取消/关闭/失败、无热Provider搜索或计时；Graph变化若必需先另定owner |
| A4 | 对应测试/真实组合consumer、所在factor CMake、用户文档和成本记录 | 两Mac、host插桩、Windows/WSL OFF、同份source/static/shared组合消费，不逐算子重跑交付 |

## 审计需回答

1. 已有原版MLX绑定与冻结数学的边界：仅以调用/来源身份确认，不调查冻结实现的算法。
2. 原版Gather/Scan/稳定TopK、Select/min/max、stable activation、mask/LogSoftmax与norm舍入
   是否满足C0精确域；索引/溢出检查也必须由原版成熟运算承担，不新增检查kernel。
3. 原版compiled图如何保留动态数据依赖、检查失败和两输出；既有checked元数据只作资源接线。
4. 原版Scatter重复add是否满足逐更新舍入与顺序；不相容时记录反例，不用atomic同名算子代签。
5. 所有准备/运行分界、原版隐式materialization/分配/host wait以及公开输入ready前提准确记账。

## 预声明验证

- 一组原版资格表，覆盖本批新增成熟调用的正常/空/尾/strided、rank0..6/8、各axis、F32/F16/BF16、
  index宽度及特殊值；独立oracle，不调用或计时冻结实现。失败最多两次具名纠正。
- 产品普通Direct与Graph的主/正交组合、两slot/旧结果/变化输入、动态非法索引与溢出、OOM/失败
  与close；不变资源设施只复用身份匹配的证据，新增调用必须实跑。
- Air和Mini的Apple原生正确性；Air适用host ASan/UBSan不代签Metal device。Windows/WSL
  按selector执行Apple-OFF。重型跨机器串行、parallel2，不修改电源/权限或弹UAC。
- 一份真实组合consumer在两Mac完成source/static/shared、prefix搬移和producer隐藏，检查原版
  MLX资源/许可/导出；复用未变交付接缝，新增域仍要实际运行。
- selector选中门、docs/closure（registry带明确previous-ref）、一次最终候选复核与必要P1；
  文档/状态不构建、不采样。统一Compute+Graph实现LOC，不包含测试/vendor。

当前仅SCOPED；未冻结产品FIX，无新增完整覆盖PASS。CPU/CUDA原卡、预算、未决域与验收保持。
oneDNN新能力按用户2026-09-21决定进入[独立scope](llm-moe-m2-b1-onednn.md)，旧Q2资格FAIL保留。

## 首组原版资格检查点

Air Mac16,12、macOS27.0/26A428上，锁定原版MLX的GPU compiled程序已执行127组资格：
19组F32/I64提升Scan及溢出摘要、48组stable argsort→take（F32/F16/BF16/I32、17/129/4097列、
两方向/变化输入）、60组原版take（六种storage、I32/I64、重复/负数/轴宽/最大索引）。
原版算子组合及独立host oracle通过，未调用冻结HYLE计算或计时。

原始源、构建、API调用首轮编译错误与修正、库与源码hash见
[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/indexed-qualification.json)。
这只证明所列原版组合；产品lowering、普通入口、资源/失败、Mini与三交付/P1尚未签。
来源/组合与checked资源接缝核对累计26/120分钟，余94分钟；独立构建与资格执行单列。
下一步提交有限AUDITED接法后方可实施，不据此填完整需求格。

## AUDITED：首批原版 indexed 计算接线

2026-09-21，在`54101d9c`干净检查点核对上述127组原版资格、公共Operation语义、
`apple_tensor.cc`冷分派与`apple_context.cc`的checked资源接缝。这里只查冻结实现的调用边界，
不调查其数学。源码/设计累计32/120分钟，余88分钟；同批额度不重置。

| 能力/需求 | 当前证据与裁决 | 有限FIX与责任 |
|---|---|---|
| Gather / F02.1 | 原版take、比较/any/where已资格；普通接线仍PARTIAL，BORROW | A-I1：六种已准入storage、I32/I64索引、任意axis/stride；checked先用原版where产生安全索引，状态随结果依赖保留；空输出不读payload，非空输出遇空源轴报OutOfRange |
| Scan / F02.4 | 原版cumsum的F32、I32提升I64与全部前缀检查已资格；PARTIAL，BORROW | A-I2：I32 inclusive前缀检查包含exclusive最终total；exclusive用原版slice/concatenate移位，结果cast回I32；F32用原版cumsum；空张量直接返回空结果 |
| TopK / F04.6 | 原版stable argsort/take_along_axis已证明ties、NaN和storage bits；PARTIAL，BORROW | A-I3：F32/F16/BF16/I32，任意轴，两输出；largest只反转排序key，I32提升I64避免INT_MIN取反和浮点精度损失；原storage取值，indices为I32；保留稳定有序结果满足可放宽的flags |
| 其余B1需求 | 原版MLX绑定已有部分能力；完整特殊值/舍入/Scatter等资格未完成 | PARTIAL；本检查点不签覆盖或授权改写冻结数学，后续仍须在同预算内明确原版接法 |

三个FIX共用私有`apple_mature_indexing.h/.cc`；只写原版MLX公开操作的组合与必要元数据分派。
在公共schema/shape检查后绑定，TopK移到浮点专用分支之前以兑现公共I32域。
旧`apple_indexing.cc`/`apple_reduction.cc`/`apple_scatter.cc`保持原文件，M5前不比较或删除。
新增源只进所属factor的CMake SOURCES；无新公开API、依赖、资源owner或Graph产品修改。

### 冻结的普通用法、失败与成本

- 普通Direct仍是`Context::prepare(Operation, InputSpec...)`后创建/绑定/提交invocation；
  Graph仍以`GraphBuilder::add`及`graph::apple::compile`自动使用相同冷lowering。
  用户不手选Provider；准备绑定锁定MLX，执行不计时或搜索。
- 复用managed MLX compiled region、独立invocation、native array结果owner与Context关闭。
  checked状态用已有`with_device_status`形成原版depends边，完成后只读已有I32隐藏摘要；
  数据错误不发布成功结果、不改输入或旧结果，失败invocation不得再次提交，fresh slot可恢复。
- 全部计算/检查在GPU。元数据遍历O(rank)只在准备期；Gather按原版gather和检查成本，
  Scan为原版线性扫描与检查，TopK采用原版稳定全排序再取k，完整排序及key/materialization计费。
  MLX管理其分配/融合/执行；本次不声称prepared零分配或TopK具备选择算法的复杂度。
- shape维度受MLX int范围限制，I32最大轴长下提升后的有符号I64前缀不溢出。
  未checked索引保持调用方有效索引前提；没有CPU模型fallback或新自有Metal检查kernel。

### 本增量验证

复用既有独立oracle用例，实际重跑Apple Direct/Graph的索引、TopK、失败/资源与直接消费者；
新增I32 TopK极值/精确排序及原版primitive来源检查，覆盖compiled调用而非仅直接MLX探针。
保留空域、rank8、非连续、非法动态索引、exclusive末项溢出、两slot/旧结果及close验证。
Air先开发验证，Mini/适用host插桩、三交付与P1仍按父卡在整批稳定后完成，不将开发检查点签为
全包PASS。selector按私有实现与factor SOURCES选择；本AUDITED元数据提交只跑docs/closure。

## indexed 接线开发检查点

产品检查点`d6fb456e`已把上述原版Gather、Scan、stable TopK接入普通Direct/Graph；I32 TopK
不再被浮点专用分派拒绝。MLX原生资源、checked完成与Graph产品源码复用；三个冻结数学文件
相对`b00b6d03`逐字节不变，未做旧算子性能比较。完整Windows source tree归档到Air，双方
Git tree核对一致；Mac结果和源码均未反向覆盖Windows。

Air完成34个Direct与11个Graph用例的开发验证，含原始storage bits、I32极值/稳定顺序、
任意轴/非连续/rank8/空域、坏索引、exclusive末项溢出、失败恢复、两slot/旧结果和关闭。
首轮44个通过；唯一来源断言误把MLX的`CumSum`节点名写成类名`Scan`，核对锁定源码后仅改
测试并单项复验通过，复用其余已通过项。原全组CTest的3/4记录保留，不改写成一次4/4。
开发汇总为45个不同用例已验证，46次执行包含该历史失败。Graph测试首次编译的Result清空
歧义也已修正并保留原日志，没有数值/产品实现修正。

Air两个公开头门通过；Windows Apple-OFF配置及两个公开头门通过。Windows首个命令误用
PowerShell5，现有helper所需Path API不可用，改用PowerShell7成功，共享工具未修改。
原始日志、源/产物hash、完整snapshot tree及未运行项见
[开发收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/indexed-development.json)。

源码/设计累计35/120分钟，余85分钟；正常实现、构建与预声明验证单列。统一脚本确认全
Compute+Graph **56,510实现行 / 246文件**，不含测试/vendor；全库代码1,081,132行，
自实现不含测试151,695行（均绑定该产品检查点）。其余B1成熟域、Mini、host插桩、新域三交付、
WSL OFF、最终selected门与P1尚未完成，整卡仍AUDITED/验收INCONCLUSIVE，完整覆盖格仍0。
后续继续本卡成熟覆盖；不能因局部测试通过提前结束M2或启动M5。

本检查点已清理Windows与Air各60,298,074字节的完整源码归档，以及Windows
1,660,940字节的临时OFF构建；先核对绝对路径与任务owner后删除。Air原生开发构建保留复用，
缓存/本地模型未改，原始必要日志已压入37,337字节证据包。

## 待审增量：原版 Scatter 赋值与 add 资格边界

`54c02a4e`后继续同一B1，原A-I1..3授权保持。本节只固定原版资格范围；新产品FIX尚未AUDITED。
F02.2需求包含unique assign和duplicate add；现公开Scatter额外承诺重复assign取最后项、add逐次
舍入。未核对的原版同名函数不能代签这些语义，也不因接库擅自放宽合同。

- 锁定MLX的`backend/metal/indexing.cpp:232–257`拒绝8字节scatter值，并从base复制输出；
  `kernels/indexing/scatter.h`以`op.atomic_update`提交不同线程的更新，未承诺逻辑行主序。
  这些是原版源码；未读取冻结HYLE Scatter算法。
- 待资格接法：原版I32 `scatter_max`把每个目标映射到最大逻辑更新位置，再原版take/where
  选择更新或base。数据不交给8字节scatter；F32/F16/BF16选择用原版同宽bit view保持storage，
  I64/Bool直接原版选择。索引/布局和checked摘要复用已资格的原版操作组合。
- 限定原版资格：六种storage×两种index宽度、scalar/多维/repeated/strided、动态坏索引与
  恢复、raw bits、空域；独立host赋值oracle。一次F32原版scatter_add非结合序列核对只验证其
  能否满足当前逐次add合同，保留失败/无保证结论，不调用旧HYLE数学、不计时。
- 成本预审：临时I32目标映射O(axis extent)与更新位置O(index count)，原版take/where为
  O(output elements)，固定数量原版调用；不建立按每条更新展开的巨大图，不新增Metal核。
  add不合格时保持具名成熟承载缺口，不转为自研授权或缩小完整B1分母。

本次合同/原版源码与接法核对累计44/120分钟，余76分钟；资格执行单列。先归档资格、冻结
具体有限FIX并提交AUDITED检查点，再改产品分派。其它成熟张量/归约域仍待整批来源/语义核对。

## AUDITED：原版 Scatter 赋值增量

`d1918255`后的原版GPU compiled资格一次完成：赋值242组通过，覆盖六storage/两index宽度、
scalar/多维/129项、noncontiguous、重复/非法/恢复与空域，独立oracle逐位一致。
原版`scatter_add`的四个F32非结合输入有三项不满足当前有序add合同：3/129/8193个更新的
顺序oracle为0，原版分别得到1/16/128。这对应原版线程间atomic_update无顺序保证，
不宣称原版算子自身错误，不调用冻结HYLE算法；原始输入/结果见
[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/scatter-qualification.json)。

据此仅冻结A-S1：在既有私有成熟lowering增加上述max-map→take/where赋值组合，普通Scatter
的assign分支冷绑定；add保持原冻结路径及成熟接入未完成记录。六storage、I32/I64、任意轴/
rank/stride与已有checked/empty行为保持；output/index/update元素数继续限制为INT32_MAX。
空indices原版copy保留base，空output不读payload；非法indices只送safe值给原版Scatter Max，
隐藏I32摘要复用现有with_device_status，失败不发布结果或改变base/旧结果。

仅改`apple_mature_indexing.cc`与`apple_tensor.cc`分派、来源断言/用户说明；不改公共API、
依赖、CMake或Graph产品，不读取/修改冻结`apple_scatter.cc`等数学文件。P0为固定数量原版
调用与O(output+axis extent+indices)临时/工作量；无逐update图展开或host payload回读。
新增普通compiled来源断言检查`Scatter Max`；既有独立赋值/边界/Graph状态用例作为实际回归，
相关数值/资源与三交付/P1仍按整卡完整门执行。源码/设计46/120分钟，余74分钟。

## Scatter 赋值接线开发检查点

产品检查点`ccaa44ee`（完整tree `e2ce79d4220d5f15a4ab231a5ac90ed9896071a4`）已落实A-S1：
普通Direct/Graph的Scatter assign冷绑定原版MLX Scatter Max→take/where组合。保留六storage
原始位、任意轴/非连续、重复索引最后赋值、空域、checked失败和旧结果寿命；原冻结数学文件
逐字节不变。add仍保留已记录的三个原版顺序反例，成熟承载尚未解决，不签该域完整覆盖。

Windows完整源码快照与Air的Git tree核对一致后，Air Release/shared/WERROR开发构建通过；
本轮4/4 CTest通过，含两个公开头门和34个Direct、11个Graph用例。这次全组45个用例均通过，
不改写前一检查点首次来源断言失败的历史。Windows Apple-OFF配置和两个公开头门通过。
来源节点断言确认本次赋值使用原版Scatter Max，原有独立oracle/状态用例完成实际产品回归。
原始日志、源码/产物hash及适用未运行项见
[开发收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/scatter-development.json)。

源码/设计仍为46/120分钟，余74分钟；正常实现与已声明开发验证单列。其余B1数值域、Mini、
host插桩、新域三交付、WSL OFF、最终selected门与不可变候选P1尚未完成。整卡保持AUDITED，
完整验收INCONCLUSIVE，新增完整需求覆盖仍为0；未将本开发检查点作为整批完成。

统一`loc-report.py --self-check`核对全Compute+Graph为**56,573实现行 / 246文件**，不含测试
或第三方源码；全库代码1,081,414行，自实现不含测试151,758行，绑定上述产品检查点。
本轮已清理Windows与Air各60,347,869字节的完整源码归档，以及Windows1,660,941字节的
临时OFF构建；删除前核对绝对路径与任务owner。保留Air活动构建、依赖缓存和本地模型；
必要日志归档为26,741字节证据包。本次记录变更仅执行docs/closure，不再构建或采样。

## 待审增量：原版归约、掩码与概率归一化

从`d1f6205b`继续同批B1，A-I1..3/A-S1保持。本增量先资格原版MLX Sum/Mean/Min/Max、
Softmax以及其官方LogSoftmax组合；不读取、修改或计时冻结`apple_reduction.cc`数学。
已有`apple_tensor.cc`的布局/Select/Unary/Binary/gate/norm均为原版调用或其组合，保持来源边界，
复用独立测试；不因迁移重写已经合格的绑定。

- 待资格域：F32/F16/BF16、F32累加、任意axis集/keep-dim、scalar/rank8/stride、空输出和空归约
  identity、NaN/Inf/正负零；Softmax覆盖Bool/additive广播mask、scale、非末轴、全mask、长行、
  大公共offset与两种输出形式。只用原版MLX与独立host oracle，不调用HYLE数学。
- LogSoftmax直接采用锁定MLX `python/mlx/nn/layers/activations.py:66–75`的公开组合：先减max，
  再减shifted输入的logsumexp；该来源明确防止大offset吞掉normalizer。C++原版ops负责计算。
- 若原版Min/Max对NaN或signed zero与公共合同不同，只资格原版比较/any/where的有限修饰；
  不增Metal核，不用与冻结实现比较替代独立oracle。空域由准备期shape确定。
- P0为固定数量原版归约/逐元素调用，工作量O(input elements)，中间数组由MLX管理；临时与
  mask/精度转换全部计费。先记录原版资格，再提交有限AUDITED接法，最后改产品分派。
- 开发验证复用普通Direct/Graph的独立归约/概率与资源用例，新增原版primitive来源断言。
  Mini/插桩/三交付/P1仍按整批预声明门；本scope记录只跑docs/closure。

Scatter顺序问题的本轮有限需求核对：F02.2/F13.2矩阵要求重复目标求和，没有逐次storage舍入
条款；现公开文档承诺更强。已登记且干净的llama.cpp `4c9233c0`在`llama-graph.cpp:2327–2353`
依专家序号组合add；SGLang `06992c92`的`fused_moe_native.py:156–162`则先按top-k权重类型相乘，
再sum后cast。两者不能证明所有模型要求统一的逐次低精度舍入，也不能仅凭此放宽现有公共合同。
保留具名公共语义审计事项，后续在共享owner依法定界；本增量不改变Scatter API或其失败记录，
不把成熟接入差异转成自研理由。源码/设计累计58/120分钟，余62分钟，同批预算不重置。

## AUDITED：原版归约与概率归一化增量

`7fdc3c6b`之后，Air锁定MLX GPU compiled组合已通过409组资格：174组归约、235组
Softmax/LogSoftmax。范围、独立oracle容差、原始探针及其一次局部命名编译修正见
[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/reduction-qualification.json)。
这些不代签普通产品或整批验收；普通测试采用原有更严格的独立oracle，不改其容差。

| FIX | 原版承载与冻结接法 | 必需语义/成本 |
|---|---|---|
| A-R1 / F04.5 | 原版Sum/Mean/Min/Max；在私有`apple_mature_reduction.h/.cc`接受已验证schema、shape和dtype，冷分派替换旧归约路径 | F32或half显式F32累加，F32默认Unknown仍合法；axes/all/keep、scalar/rank8/stride/empty；Min/Max用原版bit view/equal/any/where修饰零符号；原版自身传播NaN，源码`kernels/reduction/ops.h:175–285`与探针已核对，不再加重复NaN全量扫描 |
| A-R2 / F02.3、F04.5 | 原版precise Softmax；LogSoftmax按上游nn官方shift→logsumexp组合；原版F32 multiply/where/add处理scale与mask | 原任意axis/广播mask、NaN/+Inf/全负Inf语义及大offset精度；空output由shape直接构造；最终一次cast回storage，原生临时/转换和全部归约计费 |

两项只改Apple私有lowering及factor SOURCES；无新公开API/资源owner、依赖或Graph产品改动。
外来异常仍在既有adapter边界转Status，Context/Program/Invocation/结果lease/close合同复用。
不得读取或修改冻结归约、索引、Scatter数学；不在该增量裁决Scatter公共语义。
普通来源测试要求原版Sum/Min/Max/Or/Softmax/Log节点，无自有MetalKernel；现有Direct与Graph
独立oracle及slot/旧结果/失败用例实际回归。开发验证后继续整批其余域，不因局部通过标CLOSED。
源码/设计累计65/120分钟，余55分钟；资格运行和正常开发验证单列。

## 归约与概率接线开发检查点

`877bd869`（tree `447aecf8313c730931d6b9a6fbacdb0fe13ac4d6`）已落实A-R1/R2。普通Direct
和Graph现在使用原版MLX归约、precise Softmax与官方LogSoftmax组合；旧归约/索引/Scatter数学
文件保持原样。无新公开API、依赖或Graph产品变更，MLX仍管理compiled region与临时资源。

Air Release/shared/WERROR复验4/4 CTest通过，含35个Direct、11个Graph用例与两个公开头门。
既有更严格的独立oracle确认任意轴/布局/空域、NaN/Inf/零符号、广播mask、长行和大offset；
Graph多输出概率选择/归一化、变化输入、旧结果与生命周期用例通过。Windows Apple-OFF配置、
两个公开头门通过。完整Windows snapshot与Air tree一致，远端没有反向修改产品。

首次开发验证的44/46通过、两项失败记录保留：一项是新接线漏掉现有additive mask同storage的
准备拒绝，已恢复该检查；另一项是来源断言把原版融合`LogSumExp`期望成`Log`，核对上游源码后
只改断言。数值计算无需修正。最终46个不同用例通过；两轮共92次执行，不能混成92项新能力。
原始日志、两tree身份、产物hash、selected及未运行门见
[开发收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/reduction-development.json)。

源码/设计累计68/120分钟，余52分钟。剩余B1域及Scatter有序add问题、Mini、host插桩、新域
三交付（含selector选中的compute/graph delivery门）、WSL OFF、整批final selected与P1未完成；
整卡保持AUDITED/完整验收INCONCLUSIVE，新增完整覆盖格0。未执行性能采样。
统一LOC为**Compute+Graph 56,693实现行 / 248文件**，不含测试/vendor；全库代码1,081,802，
自实现不含测试151,878。已清理Windows和Air各60,385,031字节源码包及Windows1,661,106字节
OFF构建；保留活动native构建/缓存/模型，必要开发证据包37,267字节。本记录只跑docs/closure。

## M2-S：共享数值裁决后的 Scatter add 接入范围

`662bb3f2`的[共享审计](llm-moe-m2-scatter-numerical-semantics.md)已明确浮点add允许重结合、
不同更新顺序及存储精度或更宽的中间值；固定模型F32端口/误差门不变。此前3/129/8193项的
原版MLX有序资格FAIL保持历史结论；本批余项以新的公共合同执行，不能由文档修正取得PASS。

下一项有限资格只使用锁定MLX的原版scatter_add与reshape/moveaxis/expand_dims/Cast，以及
已资格的比较/any/where检查组合。HYLE只转换端口形状、连接隐藏状态和原生owner；不新增
Metal核或读取冻结Scatter数学。正常输入用独立高精度归约，精确可表示的重复更新检查丢项；
非结合反例单列合法数值结果，保留原来的checked失败、旧结果和Graph状态寿命要求。
原版资格完成后，仍须在本B1卡固定A-S2有限FIX，再改普通产品分派和对应oracle。
本节没有授权未资格的新分派，源码/设计仍68/120分钟；完整21行与跨环境/交付/P1门保持。

## AUDITED：原版 Scatter add 增量

`0168276b`之后，Air锁定MLX原版GPU compiled组合一次通过420组资格：396组shape/layout/
status，24组精确贡献、非结合与特殊值控制；来源、独立oracle及原始探针见
[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/scatter-add-qualification.json)。
F32的三项非结合序列得到0，属于该输入合法F32组合；F16/BF16提升后得到精确的1。旧有序
FAIL保持，不把本原版资格当普通产品或完整覆盖验收。

冻结A-S2：在`apple_mature_indexing.cc`用原版reshape/moveaxis/expand_dims把updates转换为
native索引前缀加base切片形状，原版F32 scatter_add后一次cast回storage；`apple_tensor.cc`
将普通Scatter add冷绑定此接法。F32/F16/BF16之外仍在空域分支前拒绝；空indices原版copy、
空output跳过payload、INT32_MAX元素范围及原版safe-index/隐藏状态保持。复用既有原生owner，
无新公开API、依赖或Graph产品改动；冻结数学文件不改、不删除。

成本为原版O(output+updates+indices)工作与数组，包含F32转换/临时输出和checked归约；
不按update展开图，不读回payload或引入自有核。MLX负责原生布局整理、分配与调度。
普通测试迁移为独立FP64求和和最终storage边界，单列精确贡献及合法非结合结果；Graph和Direct
分别验oracle，旧结果/input的逐位不变仍严格检查。保留所有坏索引、失败恢复、状态反馈和close
门，新增原版Scatter Sum来源断言。非确定性不授权放宽固定模型误差门。
源码/设计累计75/120分钟，余45分钟；正常实现/预声明验证单列，完整B1验收仍未完成。

## Scatter add 接线开发检查点

`33a6555b`（tree `1ca0ac214b274dd92993072899eb7a7bc2900491`）已落实A-S2，普通Direct/Graph
使用原版MLX Scatter Sum、F32累加和最终storage转换。checked隐藏状态、空域、整数add准备
拒绝、资源owner和状态反馈保持。三个冻结数学文件、Graph产品、公开API/依赖/CMake未改。

Windows完整源码快照与Air tree核对一致，Release/shared/WERROR低并行构建通过。首次6个
CTest门为5过1失败：42项compute_ops、36/37项Direct、11项Graph通过；新边界用例把rank8
base与二维indices组合成rank9 updates，超出已有公共上限。只修测试，加入合法的一维indices
组合并明确验证超限拒绝；定向重跑该用例通过，复用其它未变通过项。最终90个不同用例已验证，
91次执行包含原失败；不把首轮CTest记录改成6/6。全部原始记录与身份见
[开发收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/scatter-add-development.json)。

独立FP64 oracle、精确贡献控制、合法非结合结果和原版Scatter Sum来源断言通过；Graph/Direct
各自验证结果，输入与旧结果仍要求逐位不变。Windows Apple-OFF配置及两个公开头门通过。
共享Scatter语义的Air公开头与42项schema门也完成，WSL/Mini及其交付闭包尚未完成。
源码/设计累计78/120分钟，余42分钟；未运行P1，完整B1仍AUDITED/INCONCLUSIVE、覆盖0。

统一LOC为**Compute+Graph 56,734实现行 / 248文件**，不含测试/vendor；全库代码1,082,397，
自实现不含测试151,919。已按绝对路径和owner核对清理Windows/Air各60,489,961字节源码包、
Windows1,661,106字节OFF构建；保留活动原生构建/缓存/模型，开发证据压缩为44,388字节。

接下来按本卡21行收账，补齐F04.3的L2归一化和F04.4的norm-before-swish gate等模型组合
consumer资格，复用已有成熟原子和原生region。之后在同一整批候选执行Mini、host插桩、
新域三交付、WSL OFF、final selected与P1；不逐算子重跑全包，不启动冻结自研比较。
