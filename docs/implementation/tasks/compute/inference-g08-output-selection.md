# G08 设备端输出数值选择与可定位随机数

状态：**AUDITED，2026-10-16；八owner按S1–S4及Mem M1有限FIX施工，首批CPU/CUDA runtime已过，尚未整项验收/P1**。baseline 为
`85bcfa4eb`，G07已经整项接受，新G07–G10 goal完成1/4。原G01–G07不重跑，G09/G10未开始。
本卡只处理原[G08发现](../project/inference-single-machine-foundation-audit-2026-10.md)与
[消费形状计划](../project/inference-consumer-adaptation-plan-2026-10.md)的有限输出数值域。

## 真实任务、范围与 owner

主任务是 resident 词表 logits → GPU 数值选择 → checked completion → 只读回选中I32 ID及
需要的少量数值结果。当前本地Qwen3.5-2B `text_config.vocab_size=248320`，覆盖单请求与
多请求、请求重排/增减、至少65步和两个独立slots。模型文件只读，不运行完整模型/server。
正交任务是非模型批处理分布的离散随机选择与显式counter随机字；不以token身份设计数值API。
[输入证据](../../../../evidence/architecture/inference-g08-output-selection-intake/receipt.json)
绑定实际源码、三个固定peer、模型配置及各owner原terminal。

| 能力族 | 层级与本轮任务 | 初查 / 审计必须裁决 |
|---|---|---|
| S1 确定性单选 | 原子数值机制；复用TopK(count=1)的值+I32索引 | CPU原partial_sort为O(V)但分配V个Entry；CUDA按全输入radix-sort并分配全量keys/indices；Apple/Vulkan全argsort。审成熟arg-reduction，保原axis/stride/dtype、NaN最后、首index tie和源存储bits；不另造引擎greedy façade |
| S2 可定位随机字 | 原子数值机制；显式key/counter输入、设备生成 | 现util/SIMD xoshiro为可变顺序状态，不等于device counter schema。审固定成熟counter算法/许可/known-answer、key/counter容量、word寻址、重排/失败不暗改状态 |
| S3 有限categorical数值域 | 复合数值操作；原RNG+稳定变换+row selection | 审log-weights/weights自然输入、输出与必要selected scalar，非有限/零质量/空/overflow、精度和分布。三peer支持数值抽样，但SGLang特别要求端点精度；Metal无native F64，不准直接钳制F32 uniform后宣称等价 |
| S4 持续与组合 | BORROW原Graph/资源/Queue/native完成和拥有结果 | 真实device产出/消费，只传少量结果，两个slots、修改输入、失败/取消/close/旧结果、source/static/shared；统计、模型采样政策和调度不得下沉 |

当前拟定施工owner仅：`compute_ops`（通用数值schema/私有共用数学）、
`compute_provider_native`（CPU随机数值承载）、`compute_provider_ggml`（原CPU TopK单选）、
`compute_provider_cuda`（原CUDA collectives与设备数值）、`compute_route_apple`、
`compute_route_vulkan`（各原生承载）。它们由本卡新事实从CLOSED进入SCOPED。
Graph与CPU/CUDA默认route先只读；若实际接缝需改，先纳入owner，不借本卡改已关闭产品。

拟优先借现有typed custom operation机制新增独立数值头，保原Operation variant和Graph
schema机制；该方向仍须AUDITED时冻结。不是兼容要求：必要时可直接迁移现有消费者。
复用原allocator、状态/错误、stream/worker/fence与真正退休，不另建RNG全局owner或线程池。
随机输入属于caller，prepare不读取payload、不按地址缓存；不承诺各native数学逐bit同token。

不包括temperature/penalty/top-p策略、grammar/EOS、请求seed政策、tokenizer、完整sampler链、
MTP/多模态、完整Qwen CLI、多GPU/多机或新的全库缺口扫描。选中概率/数值结果若准入，须
冻结数学含义和费用；不能以全词表D2H或由引擎重复写设备数值来绕过S3。

## 审计与验证边界

固定peer仍为SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3，复用原归档，不刷新
其checkout。另核实际锁定native数学及成熟counter算法。审计只答S1–S4的有限决定，
至多一个检查点；4h作检查点而非强行到点结束。完成API/ownership/错误/并发/冷热成本、
SELF/BORROW/ROUTED/APPLICATION与有限FIX后逐owner AUDITED，才写产品。

AUDITED必须固定自然C++草图、真实target/filter、数学oracle/质量阈值、三交付及性能合同。
当前预声明适用检查：

- 数值：独立存储bits/首index oracle；NaN/±Inf/±0/非整tile/非末axis/stride/empty；
  counter known-answer/进位/高位/寻址与前缀稳定；categorical独立高精度reference、极端
  common offset/尾部/屏蔽和确定性固定seed分布检查，避免随机flaky门。
- 旅程：单/多请求、65步、重排与两slots、Graph设备producer→selection→少量结果、
  无隐式counter前进、非法数值失败不发布、OOM/accepted prefix/close/drain和旧输出。
- 实际CPU/CUDA/Apple/Vulkan source；受影响static/shared迁址与独立公开头，真实直接
  Graph及三个joint消费者；未变数学/资源风险按输入匹配复用，不制造全库重编验证。
- CUDA实际memcheck/racecheck、Apple实际Metal、Vulkan validation及适用三host facets；
  不用宿主san证明设备/driver/private pool，也不把未跑项写PASS。
- P0列出R/V复杂度、resident/temporary、host等待、copy/readback、状态与owner成本。
  新counter/categorical缺少原同counter合同peer时只签数值/P0，不伪造速度parity。
  S1若改实际单选执行，候选后按同TopK(1)值+索引/原完整完成与owner边界，对本baseline
  各受影响路线固定至多两形状（R1/R16、V248320）的P1；原统一5%/n15与A/A/双顺序/
  placement资格不变。真正合同与CHARGED/DEFERRED账冻结前不构造benchmark。

批量实现后集中验证、按真实失败定向修复。普通最多4jobs，CUDA pool1/split3/Ninja2；
Windows/WSL/Air重型任务串行，Mac仅实际确认身份的MacBook Air。时间沿用户授权自动追加，
保原失败和同树费用；不因换日志目录重置预算，不扩建公共统计/收口工具。
每轮按selector选择影响，运行docs/closure/状态投影与diff检查；候选后聚合七门/七问、
独立阶段红队和统一LOC。全部适用证据到位才关闭G08，保持G09/G10和CLI范围不变。

## AUDITED：接口、承载与有限改动集

[固定审计与原文](../../../../evidence/architecture/inference-g08-output-selection-audit/audit.json)
确认三peer的数值/政策边界、原native执行和错误owner；Random123 v1.14.0固定
`726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`，借其Philox4x32-10方程/官方known-answer，
BSD-3-Clause原许可/NOTICE随公共schema依赖交付，不拉入完整新runtime。原MLX提供log1p；
无log1p的GLSL用稳定补偿对数，数学与Eigen3.3.7独立参考的fallback相同，不依赖Eigen。
这是必要的整数/数学承载，不复制util/SIMD的顺序RNG或新增全局generator。

公开头冻结为 `hyle/compute/sampling_ops.h`，归compute_ops；使用现有typed custom schema，
不扩大Operation variant或新增runtime extension总API。C++自然入口：

```cpp
using namespace hyle::compute;
// Direct prepare 或 builder.add 接收同一 Operation；context 属于所选原路线。
auto bits = sampling_ops::random_bits(8);
auto selection = sampling_ops::categorical();       // I32[R]
auto scored = sampling_ops::categorical(true);      // I32[R], F32[R] log probability
```

`RandomBits{word_count=4}`读dense U32[R,4]，每行依次key低/高32位、draw低/高32位；
word j为Philox counter `{floor(j/4)低/高32位,draw低/高32位}`的lane j%4。
raw words输出dense U32[R,word_count]，跨合格路线逐bit相同、前缀不随word_count变。
word_count≤INT32_MAX，零count/零R合法；没有隐式counter加法，全部64位draw值合法。
下次draw与溢出处理属于caller，不引入内部可变状态、host seed同步或地址cache。

`Categorical{return_log_probability=false}`读dense F16/BF16/F32[R,V] log-weights和相同
U32[R,4]，V>0且≤INT32_MAX；零R合法。有限值有效，-Inf表示零质量，NaN/+Inf或任一
全-Inf行经原checked完成返回InvalidArgument，不发布可用输出。只读输入、输出不别名。
可选F32结果为原存储log-weights归一化后的选中项log probability；没有temperature/
过滤/penalty/EOS等策略。若不要该结果，省去其normalizer工作；不可读回完整词表求它。

CPU借llama的CDF数值档：原SIMD exp、F64累计/阈值，复用原executor与显式有界scratch。
GPU借SGLang/vLLM的Gumbel/arg-reduction任务分解，先减row max，column j用raw words
2j/2j+1组成开区间64位midpoint；直接从原/补码计算U或1-U较小的一侧，稳定计算指数噪声。
不先舍入U到F32再截断端点，不冒充native F64。native数学允许合格误差；同一native recipe
固定输入/key/draw须可重放且不依赖batch排列，**categorical ID不承诺CPU/GPU间逐bit相同**。
所有路线共享输入、错误、ownership、计数和质量验收，raw随机字有更强的逐bit合同。

| FIX | 冻结处理 | 保留边界 |
|---|---|---|
| S1 | CPU TopK1单趟保原key/tie/源bits、无V个Entry；CUDA借CCCL arg/segmented reduction及原Layout，不全量radix-sort；Apple借原native max/argmax和NaN有效tie组合；Vulkan为原row-wise dense F32单选使用有界分块pair reduction | 其它TopK count/axis/stride/dtype原合同与路径保持；常用dense last-axis优先，原native域不支持的dtype不因此新增；无独立greedy状态owner |
| S2 | compute_ops登记两个typed数值schema和版本化Philox合同；四native借同一固定方程、license和known-answer | CPU/CUDA/Metal/GLSL各自原编译/内存/完成owner；不建立随机库facade、依赖服务或新的总Operation分发 |
| S3 | CPU原NativePlan submit/allocator/executor；CUDA原Queue设备status/CUB/native kernel；Apple原MLX编译/array/check trace；Vulkan原dispatch/status/fence | CUDA先验证全部row再写caller输出；native拥有的中间值可以安全默认执行但失败不发布；无提交前payload D2H/控制等待。Metal/GLSL不要求F64/int64 shader feature |
| S4 | 主/正交公开Direct/Graph与三个joint消费者、65步/两个slots/重排/旧结果/失败和小结果回读 | 原F1–F3/Graph/传输/取消/关闭协议复用；不改既有packed权重/attention数学，不借本轮恢复G07或旧CUDA交付脚本 |

常态费用：raw bits为O(R×words)，无常驻generator；CPU categorical为O(RV)，有界每worker
scratch、SIMD exp和F64累计；GPU为O(RV)的随机/变换/归约，必要native张量或分块scratch
明确计费，**不承诺O(R)临时空间或零分配**。选中ID为4R字节，可选logprob再4R字节，
加原必要status；codes/counters/输入上传显式。prepare只读描述，JIT/pipeline为冷费用。
borrowed输入和allocator保活至实际退休，重绑/修改须等原完成；原取消不能撤销已接纳计算。
native独立invocations/旧结果/close仍沿各自原对象，不合并平台原生owner或伪造capture资格。

### 冻结验收与P1准入

新增正常source目标固定为`hyle_compute_ops_sampling_test`，以及CPU/CUDA/Apple/Vulkan
各`hyle_compute_route_<route>_sampling_consumer_test`和
`hyle_graph_route_<route>_sampling_consumer_test`；对应新SDK消费者使用同源码installed-only。
三个joint补实际GPU数值选择→少量结果→CPU消费，原资源桥不因新schema重复全套历史验证。
原TopK受影响用例仅按count1/NaN/tie/axis/stride/empty，保count>1代表性holdout；
原packed/attention、其它provider不作为额外测试根。host三san只选新schema/CPU与实际
修改的native adapter facet；CUDA两工具、Metal、Vulkan validation签实际设备范围。

raw bits逐bit对官方Philox4x32-10三组known-answer、独立scalar corpus和跨路线数据；
检查word前缀/高key与draw/重排/两slots/失败不前进。categorical固定seed、至少262144
样本的非均匀/均匀/屏蔽分布，bucket期望≥20时每项误差≤7×sqrt(Np(1-p))+8，
所有bucket均检查；质量不是只验证ID范围。另验大common offset、窄dtype、极端log差、
all-invalid及独立F64 logprob，通常误差门2e-5+2e-5×abs(ref)。GPU尾部变换须包含原字
0/UINT32_MAX组合与独立F64方程；宿主约105万组预检最大误差1.31e-6仅证明可行，
实际shader仍须同一2e-5误差门；初次scratch符号错误被断言拒绝，原脚本/失败保留。

P1仍仅S1、各受影响路线R1/R16×V248320，原85bc TopK(1)值+I32索引与新实现同输入/
NaN/tie/源bits/owning结果/完成边界；原统一5%/n15–31、A/A、执行/分配/placement资格。
冷prepare/JIT不进warm；submit→完整native完成/必要status→结果取得/释放计费，最后
Program/Invocation/cache析构若DEFERRED须双方明列。不得用只产ID的轻量argmax代替
原两输出peer，不以只计GPU kernel/省略结果寿命换PASS。精确源码时序账与自然peer
adapter必须在构造benchmark前审完，不改公共统计核心。新categorical/RNG只签质量与P0，
不把不同状态/分布/完成的peer硬做同合同速度门；完整CLI/token/s仍在本goal之后。

六owner本轮只从SCOPED进入AUDITED；产品实现后先完成上述批量普通门才冻结candidate。
当前0个新产品build、0个runtime用例、0个P1，没有任何G08完成或性能PASS声明。

### 首批实现检查点

已写通用sampling_ops schema、固定Philox core/许可登记、CPU native随机字/CDF数值路径，
并将CPU TopK1改为原比较合同的单趟选择、去掉V个Entry scratch；均尚未编译或验收。
继续同一实现批的CUDA/Apple/Vulkan与公开消费者，然后集中运行冻结验证；不逐函数反复建树。

新公共头在中央SPLIT_PUBLIC_HEADERS登记一项，component ID/边、原Operation variant与旧
公共头没有变化。selector因中央manifest路径保守选中全库，原计划保留在scratch，尚未执行
native gates；不能把这些未运行项写PASS。批末核实际header/依赖/包投影与输入身份，按
本卡六owner/真实消费者及适用交付执行；无相关语义变化的独立产品需明确保留跳过理由。
不在本轮重写selector、公共构建系统或恢复旧重型交付脚本。


## G08 原定Graph消费者接缝：Vulkan owner补入

2026-10-16，编写S4公开消费者时，确认`graph_route_vulkan/vulkan_lowering.cc`
在两个位置按显式custom白名单拒绝新schema：双输出仅允许原recurrent，默认native
承载也只列原model_ops。这是本卡原定Direct/Graph同行为的可复现源码缺口；不是
新增抽样能力或改变Graph资源合同。原terminal `docs/implementation/tasks/compute/inference-g04-g05-final-closure.md`、candidate
`982676cb05a6e6244e66e0a596347aa96043617b` 保持旧范围有效，新baseline `1e58e644a5dc9306b05315e8f1d679a5dcc4b179`。

仅`graph_route_vulkan`进入SCOPED；审计限两个白名单与同一头注释，准备验证原计划
Vulkan Graph sampling消费者、双结果独立发布/Copy消费、原TopK留出和static/shared
迁址。复用原PlanStep可变outputs、prepare_plan、status、owner、liveness与真实fence；
不加Graph API、对象字段、dependency edge、private pool或新shader。P0接线无新增
热路径循环/分配；G08已有四路线S1 P1范围不增。无该owner产品改动，无runtime PASS。


该接缝审计完成：KEEP原PlanStep.outputs及逐port liveness；FIX仅新增RandomBits/
Categorical到成熟custom识别，并将双输出Categorical纳入已有合格双port分支；BORROW
全部Compute数值、状态、native资源与退休。两种公开输出形状、单/多请求反偏置与三peer
数值/政策边界已由本卡S2–S4审计冻结，不增加语义。有限FIX固定，owner进入AUDITED；
下节源码修改尚未编译/运行，不构成验收。其他六owner与原失败/费用不变。


### 四路线实现与消费者批次，尚未验收

S1与S2/S3已接原CPU/CUDA/Apple/Vulkan承载，补八个同源码公开Direct/Graph
sampling消费者与schema/KAT单元测试。共享消费者仅使用公开头；原native数学的
端点风险、三joint、小形状独立port发布、host/device风险、三交付和S1 P1仍待补齐/运行。
已写262144样本×均匀/非均匀/屏蔽质量门、独立F64 selected logprob、窄dtype、真实
V248320的R1/R16各65步、变化输入/高draw/请求重排、payload失败后同key复现与旧结果。
采用两个正常replay slots并保留一个旧结果slot；Vulkan原Tensor owner禁止在旧结果
仍被消费时覆盖同slot，测试遵守该合同，不新增产品输出分配/隐藏复制。

首次WSL g++14公开CPU Direct/Graph及schema纯语法批检查通过；不代签链接或runtime。
selector首次消费批因Random123新vendor缺少独立归属映射而拒绝；按已审compute_ops
owner仅追加validation-policy的vendor data entry，算法/门/选择器实现未改，不绕过
授权。中央manifest/lock造成的保守全库选择仍保留，非相关产品不启动全局编译。


实际GPU端点检查增加三个source-only `sampling_math` target：CUDA/Apple/Vulkan都直接
编译产品使用的同一私有数学函数，输入1048772对原始U32字（196端点组合＋1048576
固定corpus），逐项签F64方程2e-5绝对误差。CUDA/GLSL仅抽出原函数为私有include，
不增加测试专用公开schema、产品字段/分支/运行费用；Apple复用原Metal字符串头。
这些仍待实际编译/设备运行，宿主probe不代签。首轮源码批checkpoint为9d581dc81，
先集中build/run数值公开旅程，后续私有probe按真实增量接入，同树失败/费用连续保留。


首轮9d581dc81集中构建真实FAIL：CPU native build_plan重复kCustom case，尚未启动
runtime。已定向合并到原custom分支并按两个sampling descriptor分派，原DS4保持原承载；
并将sampling并行work估计限定为这两个descriptor，避免影响旧custom成本。
原日志`.scratch/inference-gaps/g08-cuda-source-1`、已用113.327s
与146个native CUDA对象复用证明保留；同一build tree继续，未更换预算或宣称PASS。


第二次同树批checkpoint ba64ff2a3在CMake generate真实FAIL：新CU数学测试位于原
enable_language(CUDA)的兄弟目录，缺CMAKE_CUDA_COMPILE_OBJECT；只在该测试目录
启用原已锁定CUDA language，产品构建算法未改。此轮66.804s及
原失败完整保留。三joint公开消费者已写：GPU Copy producer→categorical双结果→CPU
Cast/Add，65步/两slots，仅迁移R个I32与R个F32；CUDA两跳staging明确收费，仍待运行。


首轮完整构建4e0bd29a5已通过（含三CUDA TU），schema三测试PASS，CPU TopK/raw RNG
旅程PASS；CPU categorical在消费者create_invocation缺显式caller workspace而FAIL，
已仅修公开测试适配器按workspace_requirement为各slot配置真实原资源workspace。
并修测试range-loop/dangling-else编译警告。原轮费用142.06s和FAIL日志
保留；Graph/CUDA/joint/device math尚未执行，不代签PASS。测试正文增量复用同tree/configure，
不为此重跑CUDA编译或无关模块。


19b6320c9测试正文增量仅14edges、无CUDA编译；CPU categorical随后发生SIGSEGV，
GDB定位在CommonSession析构的mimalloc free。尚未归因，不将前序部分PASS写为整项PASS。
先对真实caller workspace添加前后guard并输出有限阶段标识，区分数值写越界、消费寿命
与哪个dtype/shape；不修改Mem/SIMD等关闭模块，不靠重复全局构建排查。


## G08真实消费者依赖修复：Mem over-aligned sized free

2026-10-16，最小复现不执行任何Compute：system_resource Buffer 884 bytes、alignment256，
写满再销毁即SIGSEGV；208 bytes配对和prepare/close-only也保留。GDB与诊断源码在
`.scratch/inference-gaps/g08-cpu-debug-2`。原native GPU端点1048772对已PASS，CPU计算
数值/guard成功后释放失败，因此本问题不能靠改sampling shape或丢弃正常输入掩盖。

owner `mem`从原CLOSED有限重开到SCOPED；前轮 `docs/implementation/tasks/mem/mem-v10-external-byte-reservation.md`、candidate
`4e9f74a09b7218f6053038cc5c5b091e73ff0bc5` 维持原证据，新baseline
`994fc6b006990f12e32dc41b8055a391732e593c`。
仅审Mem system.cc/system_inline.h到锁定mimalloc3.5.1的over-aligned释放接线。
不改vendor、Allocator/Buffer布局、分配策略、quota或普通natural-alignment分支。
这是一项G08旅程必需的可复现correctness依赖修复，不新增G11功能族或重开缺口扫描。

验证冻结：请求size跨small-class边界且alignment16/64/256/4096的完整写入/释放、
allocate/reallocate/Buffer/SystemAllocator/跨线程释放及零/大请求留出；原Mem实际适用
correctness/公开头与三个OS source/static/shared，host sanitizer facets和G08真实CPU/
Graph/joint直接消费者。沿原树、串行、普通最多4jobs、保留已有失败/费用。停止于此一
分支修复及适用验证，不扩写allocator或公共构建/统计仪器。当前无Mem产品修改。


Mem依赖审计：锁定vendor free.c:308–348的mi_free_size_aligned直接调用mi_free_size；
后者按caller size≤MI_SMALL_SIZE_MAX进入small-page metadata路径，而aligned分配
alloc-aligned.c会过分配size+alignment-1，884+255越过small阈值，实际page类别不再由
caller size证明。成熟mi_free_aligned直接沿mi_free检查真实page，不依赖这个错误推断。
有限FIX M1：仅HYLE过对齐分支调用原mi_free_aligned，两个static/shared入口一致；
KEEP natural≤16分支mi_free_size、所有分配与realloc、对象字段/API/资源账；BORROW原
vendor真实page判断，禁止修改mimalloc或手写新的allocator/free算法。增加实际回归与
输入guard。仅此有限集合授权AUDITED。

性能按P0记录：缺陷884/256 baseline本身崩溃，不能构造有效同合同旧版本P1或把修复
声称为计时胜利。natural默认路径逐源码保持，已有有效大size的旧over-aligned路径本也
进入同一mi_free；修复仍O(1)、无新增分配/锁/状态。若实现偏离这个成本边界，先重审；
不把未测的整库性能写PASS。G08四路线TopK原P1范围、三GPU数学门保持不变。


M1实现仅改过对齐释放为现成mi_free_aligned，vendor完整未变；新增size/alignment边界、
realloc/跨线程及Buffer完整payload/移动回归。独立批结果也已收齐：CPU Direct/Graph
均同一allocation-only缺陷；CUDA两公开TopK旅程PASS，raw空形状proposal把dispatch_count
写0被原结构合同拒绝，已定向改为一次合法backend episode（零native kernel事实保留）；
CUDA实际math1048772对PASS、maxabs1.30670895e-6；joint路由尚失败，留待后续定向
裁决，不签PASS。原日志source5-remaining-tests及五次源码批费用均保留。


joint定向诊断确认测试选了现公开Cast不支持的I32→F32域；该功能不是G08验收前置。
将同一选中I32接现成CPU Gather查F32表，再与selected logprob做CPU Add，覆盖引擎
实际ID消费，仍只迁移24 bytes/3rows和两跳staging；只改消费者，不新增整数Cast能力
或修改其他owner。旧路由失败保留，正式旅程改签GPU selection→CPU lookup/add。


825e662a3源码批Mem244/244 PASS（含M1回归），同一CPU消费者已通过三浮点dtype、
独立logprob/重排/旧owner及真实R1/R16各65步。失败测试第二次沿用已失败的原invocation，
原operation.cc明确要求failed invocation recreated，故仅更新测试：重建failed slot1，
原Program/healthy slot0/key/draw/旧输出保留；按原合同检查每个InvalidArgument及同key复现。
本次不改产品，不重跑已通过且输入未变的Mem244项；保持后续平台/交付/风险待验。


21e925b5a首批普通运行：CPU/CUDA四个Direct/Graph sampling消费者全部PASS，
含三种浮点storage、R1/R16全词表各65步、每law262144样本、原失败slot重建、
独立数值/比特/旧结果；schema三测试和CUDA1048772对端点复验PASS。joint唯一剩余
测试错误是遗漏CPU table常量的host external_resource合同，导致默认storage0尝试
把普通host常量直接物化到GPU；仅补公开consumer的table→kHostStorage，不改Graph。
下一增量只构建/运行该joint，四个已过的普通消费者和Mem不重复运行。


019bf6e43 joint仅2个增量edge后进入原冷placement搜索预算上限；不增加预算或改
Graph搜索器，先将本consumer原本已知的三个GPU节点、两个CPU节点和六个中间值
storage全部用原公开required合同表达，避免让编译器枚举并不需要的CPU/GPU选择。
只有该joint继续定向验证，其他已通过数值测试、Mem和device math证据复用。


cc081788d显式target/storage后已走过cold placement；joint原生GPU→CPU队列不能
消费异步GPU prefix，consumer先前kReject按原合同拒绝该拓扑。改用原默认kAllow，
并断言replay report记录实际host wait；两跳staging和24-byte readback均收费，
不声称零host wait或异步overlap。本次仍仅consumer修复，不扩展G09或改Graph产品。


### 2026-10-16 首批CPU/CUDA源码普通门检查点

[原始记录与所有失败归档](../../../../evidence/correctness/inference-g08-implementation-checkpoint/receipt.json)：
Mem WSL source244/244、schema3/3、CPU/CUDA Direct+Graph四个公开数值消费者、实际CUDA
1048772对端点及CPU+CUDA GPU selection→24-byte CPU lookup/add各65步/两槽均PASS。
普通全词表R1/R16各65步、每law262144固定seed质量门、三浮点storage与错误/旧结果
包含在四个公开消费者中；具体source snapshot、二进制hash、命令与原FAIL均保留。
源代码最后产品改动为825e662a3，随后仅消费者/文档纠正；joint最新consumer0b4c9a421。

当前八owner仍AUDITED，G08粗估约55%、新G07–G10 goal约40%，正式仅G07完成1/4。
这些比例是施工粗估，不是验收门PASS比例。Apple/Vulkan尚未实际build/run，独立port发布、
剩余风险/三交付/TopK S1 P1和最终文档/审查尚未完成；Mem其他OS/SDK/host facets也待验。
下一步沿同一冻结范围验证Apple/Vulkan并补这些有限门，不启动G09/G10、CLI或新缺口扫描。
全部重型任务已结束，保留同树缓存、原失败与累计费用；后续仍重型串行。


Apple/Vulkan首批前已实查Air Mac16,12 arm64、macOS27.0.1、clang21与Metal32023.883，
无活动build；使用原shared-sdk-build作source普通批，INSTALL=OFF/TESTS=ON，4jobs。
补S4必要的Apple U32普通carrier映射：logical_type/input_spec原已支持U32，但native_type
遗漏它，导致RandomBits→原Copy的Graph消费者被拒绝。仅补同一既有dtype映射，不扩大
arithmetic/TopK dtype，不改Graph/runtime或增加新API。原TopK axis/stride/dtype/empty
留出使用原两份test TU的私有target，只跑适用filter，不构建整个模型测试集。
