# G08 设备端输出数值选择与可定位随机数

状态：**CANDIDATE，2026-10-17；八owner的S1–S4/M1接口与算法冻结于5103c13d8，普通数值及适用device/host风险已过，交付18/18已过，四路线正式P1与回溯artifact终审补证已过，最终LOC/状态落账待完成**。baseline 为
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


Apple首批fc15404bb使用原tree、4jobs，产品CPP已通过，测试TU在const mlx::array
调用非const wait()处集中FAIL；仅修Direct/Graph、math和joint测试局部array可变性。
原preflight VERIFY_GLOBS dry run失败及byte-identical frozen manifest恢复记录保留，
本轮累计33.741s不重置。另补原预声明四条Graph的categorical
ID-only/score-only/both独立port发布，变化输入、内部依赖、单port错误和旧owner；
提供--ports-only，只补新分支时不重跑已过的全词表/质量旅程。尚无Apple runtime PASS。


Apple a2真实普通批：Air Mem244、schema3、CPU Direct全数值/分布、Apple实际端点
1048772对(maxabs3.72330296e-6<2e-5)、CPU+Apple 24-byte lookup/add联合65步均PASS。
Apple Direct/Graph先在TopK顺序oracle失败；S1改用原IEEE storage的U32有序key原生
arg reduction，明确保subnormal/NaN-last/±0首index，原take_along_axis发布源bits，
I32及k>1不变。仍属已冻结S1，未启动P1、不改变公共形状或增加排序scratch。

Air CPU Graph数值至R1全词表时遇旧borrowed输出binding地址被allocator复用后的过渡
alias拒绝；仅修测试先reset旧binding集合再绑定新owner，不改Graph产品。旧目标在该
接口采用逐个bind会检查当前绑定集；稳定地址的引擎replay不需要每步这样重绑。
私有holdout实际输出在bin，a2 runner最后错误路径的SETUP_FAIL及全部前序测试日志保留；
只修runner定位，后续定向跑未过目标。测试增加有限phase参数用于独立补验，不重跑已过
的Mem/CPU Direct/math/joint或恢复全库测试。


### 2026-10-17 AUDITED补充：同一Apple浮点key缺陷覆盖原k>1

a3 b06bf3bf1证据：新k1的原storage整数key已通过；F32负subnormal的k3/smallest
原argsort输出index0、bits80000001，独立oracle要求index16、bits80000011。原四项
TopK holdout仍PASS，说明它们未覆盖这个旧缺陷，不能据此删掉新边界用例。Air CPU
Graph完整数值/质量及新独立port旅程已PASS。原日志/费用保留。

仍在原owner compute_route_apple/AUDITED、本卡S1既有TopK数值合同内；不准增加dtype、
schema、API或新机制。有限FIX增补A1：将k1刚修正的IEEE原storage key抽成同一私有
helper，k>1浮点也接同一key，但继续原stable mx::argsort、原slice和take_along_axis；
largest仅反转整数有序key、NaN恒置排序末端，±0归同key由原稳定排序保first index。
原I32 k1 arg reduction、k>1 widening/sort保持不变，输入/结果与所有owner合同不改。

P0：k>1仍原O(VlogV) argsort、4-byte/key（窄float原也widen为F32），无新host wait/
私有pool/metadata/额外排序索引owner；整数key的点运算沿MLX已有fusion。撤销此前
“k>1实现完全不变”这一限制，仅允许上述同因key修正。为量化合法正常数据上的额外费用，
Apple S1 P1增加且仅增加一个holdout：R16×V248320、count3、largest、同原85bc两输出/
owning完成/5%门；普通有限normal logits让旧版也满足同一oracle，不以旧版会崩的Tiny
案例做速度比较。其他三路线仍各两形状，Apple原R1/R16 count1保持。尚未构造P1。
先完成此AUDITED有限决定再改产品，产品实现后普通数值通过再冻结candidate。


447a6289a Apple Direct/Graph完整TopK、raw/categorical三dtype/full-vocab/65步/
质量均PASS，Graph独立port与原4个TopK留出也PASS。CPU Graph在a3已完整PASS。
下一Vulkan普通批补原native View输入byte origin1/3/17（旧公开typed carrier可消费）
到TopK1、raw U32与三categorical dtype；Graph用原View producer保留origin，
不由Copy物化掩盖Direct/Graph消费差异。仅是原S1/S2/S3有限dense/stride合同测试。

Vulkan v1新origin测试括号错误已修，v2产品/测试全部完成编译。Air原外置validation
layer目录已不存在，原带validation的失败保留；v3明确关闭该环境选项只签普通数值，
不签validation风险。补原compute_route_vulkan计划中Categorical双输出白名单，与已冻结
Graph接线一致，不改变SSA/outputs/status机制。v3 TopK数值与raw PASS，实际设备数学
1048772对PASS(maxabs3.72330296e-6)、CPU+Vulkan 24-byte joint65步PASS。

v4定位F32大offset score：row12选中值=max=1025.25，独立F64期望-1.832462711928077，
实际-1.83251953；按原tensor.comp数值做法以precise固定shifted与最终减法，不放宽门。
byte1原native View在未用raw scratch descriptor上被拒绝：改复用已拥有aligned输出作
未用binding，不新增scratch；shader按原tensor load_bytes方法读取跨word的U32/窄float，
仍使用原byte-origin push常量与descriptor。原17项TopK留出16项PASS，唯一失败仅要求
所有count走旧managed region；count1改验原新两dispatch，所有原独立bits断言保留。
这些属于原AUDITED S1–S4有限实现纠错；未进入CANDIDATE或启动P1。

### 2026-10-17 Apple/Vulkan源码普通门检查点

[原始结果、全部失败与身份归档](../../../../evidence/correctness/inference-g08-air-implementation-checkpoint/receipt.json)
保留a1–a4、v1–v5全部日志、manifest与二进制hash。Apple a4与Vulkan v5公开Direct/Graph
完整数值、三dtype、F64 score、全词表R1/R16各65步、每law262144样本、错误和旧owner
均PASS；两路线Graph独立port均PASS，Vulkan原生View byte1/3/17 raw/三dtype categorical/
TopK1也PASS。CPU+Apple、CPU+Vulkan 24-byte CPU lookup/add联合65步、两槽PASS；
原Apple4项、Vulkan17项TopK留出PASS。实际Metal/Vulkan1048772对数学端点均PASS，
maxabs均3.72330296e-6。Air Mem244与ARM CPU Direct/Graph已PASS。

Vulkan v3/v5明确未启validation，原validation-requested环境失败不能由这些PASS覆盖；
仍须恢复真实layer并运行适用风险门。WSL CPU/CUDA新独立port补验、剩余host/device风险、
三交付/公开头、S1精确P1及最终文档/审查仍待完成，八owner保持AUDITED。
G08施工粗估约70%，G07–G10 goal约44%，正式仍仅G07完成1/4；比例不是验收门通过率。
本批已结束，没有重型任务遗留运行；继续原有限门，不启动新缺口调研、G09/G10或CLI。

2026-10-17后续有限门：WSL四条Graph独立port补验PASS；原CPU selection两项、CUDA
四dtype/axis/stride/empty TopK及CUDA Graph留出PASS。CPU Graph私有target原先因遗漏
原test DEPS/macro选中0项，不能计PASS；补原依赖后source13实际1项PASS，原0项日志保留。
实际CUDA memcheck/racecheck各六组（TopK、raw、categorical、Graph ports、joint、端点）
全PASS，均0 errors/0 hazards。ASan静态host facet：Mem两修改TU51项、schema3项、CPU/
CUDA Direct+Graph完整数值/质量/独立port及joint均PASS；13个相关TU的实际instrumentation
与原native archive真实链接hash已核。UBSan/TSan、其他native host facets/SDK/P1仍待验。

Windows原Vulkan tree为当前五个目标补齐实际依赖，原459-edge预检失败及123.469s保留，
同树续跑，4jobs/4逻辑CPU affinity/BelowNormal，实际compiler子进程已核相同限制。
新测试Direct/Graph完整旅程、byte origins、Mem244和原TopK17项PASS，validation日志无
VUID/Validation Error/SYNC-HAZARD；独立device math仍FAIL，不签整项风险PASS。
首次定位端点i126、words ffff0000:00000000，small=1.5258789062499973e-5，期望
11.090347259516088，实际11.0932713，abs_error=.002923995977>2e-5。

仍按原AUDITED S3有限数学FIX：借同owner已有tensor.comp log1p的小参数八项级数，
small<1/8计算正的-log(1-small)，相对截断界<7.6e-9；其余upper继续补偿log，保precise
中间差值。避免native log在接近1时的相对尾部误差，不放宽原2e-5门、不钳制uniform。
仍O(RV)、每draw最多八项标量运算，无新临时buffer/owner/状态或host wait；raw Philox和
TopK算法不改。先定向实际device math，再复验同路线数值/分布与Metal portability。

3a66ab066实际Windows Vulkan端点全1048772对PASS，maxabs5.12140621e-6<2e-5；W4
Direct/Graph完整数值、质量、独立port与origin复验PASS，validation日志干净。Air Vulkan v6
新版Direct/Graph、端点与CPU+Vulkan joint全PASS，继续明确区分该机未装Khronos layer。

Apple实际Metal API/GPU validation首批：新Direct/Graph sampling、device math和joint
均PASS；旧四项TopK留出前三PASS，Graph probability→TopK→sum→divide在零行冷compile
abort。a5新增shape诊断确认typeF32、rows0、experts256的Graph compile触发dispatch
grid8×0×0。原TopK/softmax/reduce已有空域返回；普通Binary divide仍交MLX空域融合。

仍在compute_route_apple/AUDITED、原S4零行组合消费：有限A2为已通过公共schema与floating
域检查的Binary空输出返回原mx::zeros(shape,type,stream)，不发无元素的原生算术。仅改
该owner原apple_tensor.cc，不改Graph API/owner或MLX vendor。非空lowering保原式；新增
shape检查只在原冷tracing，固定形状warm不新增逐元素运算/资源/等待，S1 P1范围不扩大。
保原abort及四个已过Metal结果，定向同形状后重跑本Apple有限Metal批。

### 2026-10-17 device/host风险检查点

[全部原始结果与失败](../../../../evidence/correctness/inference-g08-device-host-checkpoint/receipt.json)
保留CPU Graph零用例误选、Windows预检/端点失败、Apple零行Metal abort及各次修正。
bba0a5a35 Apple a6在实际Metal API/GPU validation下五目标全部PASS，原四项留出的
零行Graph/Direct、三dtype均PASS。Windows Vulkan W3/W4与Air v6新版数值/数学已过；
Windows layer实枚举、实请求，接受日志无Validation Error/VUID/SYNC-HAZARD。
实际CUDA memcheck/racecheck各六组、ASan/UBSan/TSan静态host七目标均PASS；Mem修改TU
三facet分别51/50/50项，schema每facet3项，CPU/CUDA四完整消费者含新port与joint，13个相关TU
有实际san flag，native archive输入与真实链接hash相符。

CPU/CUDA三host facet原树串行已结束；Apple/Vulkan其余host facets、三交付与独立头、
S1精确P1、文档/LOC/终审仍待完成。八owner保持AUDITED，尚不冻结candidate。
G08施工粗估约75%，当前G07–G10 goal约45%，正式仅G07完成1/4；G09/G10与CLI未开始。

### 2026-10-17 冻结候选与有限交付矩阵

[Air六配置host证据](../../../../evidence/correctness/inference-g08-air-host-facets/receipt.json)：
Apple/Vulkan各ASan/UBSan/TSan，六目标共36次运行全PASS，含完整Direct/Graph、joint、
旧TopK留出与相关Mem/schema。保留原refcount/Metal completion边界模型及其原限制；
Vulkan仅按相同GLSL/命令/feature配置复用生成的C++ DATA，产品/native C++均实际san编译。

八owner由AUDITED进入CANDIDATE，固定完整commit `5103c13d8ed119b06aa64e799f9768aea45ab913`。
后续只补交付、头、成本账、P1和终审；产品/API/算法若需变化，按协议退回AUDITED，不让
旧证据隐式覆盖改动。P1精确时序账仍须在构造benchmark之前单独固定，尚未开始采样。

交付沿实际受影响平台与原CUDA两OS承载冻结六profile：Windows CPU、Windows CPU+CUDA、
Windows CPU+Vulkan、WSL CPU+CUDA、Air CPU+Apple、Air CPU+Vulkan；每个执行源码嵌入、
relocated static、relocated shared。普通CPU入口在各真实SDK包中消费，新native Direct/
Graph及对应joint使用原同源码public-only fixture；Mem复用原三个公开回归案例，禁止复制
system_internal或产品src头到SDK消费者。逐一保留编译include隔离、runtime依赖、独立
sampling_ops与各carrier/Mem头、Random123许可/NOTICE；无全库模型/性能测试根。
源/SDK构建复用对应既有producer，重型串行；Windows普通4jobs/低优先级，CUDA仍pool1/
split3/Ninja2，原flags/native输入资格保留。source/static/shared未过前不声明交付完成。

### 2026-10-17 交付检查点：7/18

[Air六交付](../../../../evidence/delivery/inference-g08-air-sdk-checkpoint/receipt.json)：Apple/Vulkan
各源码嵌入、迁址static、迁址shared全PASS。每配置原六个公开CPU/native Direct/Graph、joint与
Mem消费者、11个独立公开头；Mem仅原三个公开回归案例。static/shared实际迁址，Random123
LICENSE/NOTICE均安装；每配置六个轻量runtime probe实际打印dyld加载路径，Hyle库仅来自
对应迁址prefix或源码producer。保留首Apple shared私有脚本的头名错误及同树续验费用。

[Windows CPU shared](../../../../evidence/delivery/inference-g08-windows-cpu-sdk-checkpoint/receipt.json)：
原producer增量104步、公开CPU Direct/Graph完整数值/质量/full-vocab与Mem三回归PASS，8个
公开头独立编译、许可/NOTICE、迁址include隔离和staged DLL逐一匹配安装包hash均PASS。
执行PATH排除产品/producer；Windows此项不冒充实际module tracing。私有runner曾错误假定
ggml CPU变体DLL也安装于bin；按实际lib路径修正、保留两次setup失败，复用原prefix，未重建
或重装producer。普通4jobs/四逻辑CPU/BelowNormal已实核。

冻结产品candidate仍5103c13d8；本批仅说明与证据。G08施工粗估约80%，G07–G10 goal约45%，
正式仍仅G07完成1/4。余下11项交付、S1精确时序账/P1、最终文档/LOC/独立终审未完成；
不把当前数值/风险PASS或交付计数当整项验收，不启动G09/G10或CLI。

### 2026-10-17 交付11/18与S1计费边界冻结

[Windows CPU剩余两交付](../../../../evidence/delivery/inference-g08-windows-cpu-sdk-checkpoint/remaining.json)
证明源码嵌入与迁址static也已PASS；连同前述shared，该profile三交付完成。
保留源码运行库检查脚本的dumpbin路径失败及385.765s，同树续跑另59步，最终公开CPU
Direct/Graph、Mem三回归、八独立头与PE import/staged DLL检查通过，不把额外费用抹掉。

[CUDA首两交付](../../../../evidence/delivery/inference-g08-cuda-sdk-checkpoint/receipt.json)：Windows
shared、WSL static各六个公开CPU/native Direct/Graph、joint与Mem消费者、13独立头、迁址及
Random123许可/NOTICE全PASS。外部已验证LibTorch SDK/cache按原distribution合同保留要求。
Windows原私有profile的INTERFACE目标错误、152-CUDA-edge预检及覆盖MSVC原选项的CCCL
编译失败均保留；只修私有配置，未改产品。Windows实取旧对象记录的242个派生/5个公共输入，
逐字节及原Ninja命令/依赖freshness检查通过，复用146个未变ggml CUDA对象；六个实际需要的
产品CUDA依赖正常构建。WSL沿原同资格缓存，无跨平台对象复用。

[S1源码时序账](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-boundary.json)
在构造benchmark前固定九形状、11阶段、双方CHARGED/DEFERRED/N/A、资源账及自然baseline
承载；原85bc与candidate共30份关键源码、21份完全相同，原文与行号均归档。计费包含每次
submit、全部选择/排序、实际checked完成、两个小输出的host读取与步内owner释放；冷准备、
输入上传、末次保留结果与runtime/private cache最终析构分列，不宣称所有host尾部已退役。
原完整CPU/CCCL工厂、Apple typed lowerer与Vulkan完整route保持原算法/owner；只做私有符号/
namespace隔离。已从原Git对象物化CUDA/Vulkan参考源码草稿，未配置/编译benchmark、未跑
oracle/A/A/placement或P1，不能据此签同合同资格/性能PASS。5%、n15–31与控制门不变。

当前交付11/18：Air两profile六项、Windows CPU三项、Windows CUDA shared、WSL CUDA static。
剩Windows CUDA static/source、WSL CUDA shared/source及Windows Vulkan三项，按串行一批执行，
遇真实失败定向处理。产品candidate仍5103c13d8，G08粗估约85%、整个goal约46%，正式仅G07
完成1/4；G09/G10尚未开始，不新增缺口扫描或CLI范围。

### 2026-10-17 交付17/18与同合同基准工作区

同一串行批次新增WSL CUDA shared、Windows CUDA static/source、Windows Vulkan三交付
全部PASS，当前17/18。最后WSL CUDA source沿原parent/configure/build继续：原私有helper
把两份历史native profile误当成唯一一份；按输入、原Ninja命令和依赖freshness只接受一份后，
保留原48.571s。后续预检仅允许两项CUDA编译，实际需要十个精确Hyle依赖TU；已用Ninja
explain核明新增/缺失对象、device-status/native-batch头更新和原命令变化，再定向续跑。
146个未变ggml CUDA对象仍沿原资格复用；没有产品改动、时间戳伪造或失败预算重置。
该source尚未结束，不能把交付总门写PASS。

[源码复核](../../../../evidence/architecture/inference-g08-output-selection-audit/red-team-source-checkpoint.json)
覆盖37个本轮include/src文件及三项依赖/公开头/selector登记，未发现新的产品缺陷；
最终交付、adapter runtime及性能仍待复核，不把本次源码检查当整项red_team通过。
[统一LOC](../../../../evidence/architecture/inference-g08-output-selection-audit/loc-final.json)
在不可变8349757按八CANDIDATE owner唯一联合计数，self-check PASS：42679行/208文件；
全库3797821行、自实现不含测试182302行。产品字节与5103相同，不是新增行数或引擎LOC预测。

[基准源码](../../../../bench/compute_ops/CMakeLists.txt)固定于`df7ca0271`，
Windows/WSL/Air使用干净的同一benchmark/evidence提交；原版peer为干净的85bc，
CPU三TU、CUDA两TU、Apple两TU、Vulkan完整26TU共33个原实现TU及共用私有头均实际核对。
旧完整include树只作原文，公共声明使用同一合格SDK；Vulkan仅自有声明作namespace隔离。
[参考接线修正](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-adapter-source-corrections.json)
保留构建前发现的Git peer绑定缺失、Apple旧tensor误调用当前indexing实现的问题；
现同时编译并隔离旧tensor和indexing，正式A/A前还检查实际符号与编译命令。

[四路线工作区与实际配置](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-workspace-checkpoint.json)
均通过benchmark-only预览；配置后仅补A/A子进程使用原注册策略，
Windows实测canary从默认NORMAL不符变为原ABOVE_NORMAL/CPU2策略匹配。公共仪器、数学、
九形状、5%、n15/31未改。新JSON发布前统一LF并重算引用hash，原格式失败保留。
尚无实际benchmark workload build、数值oracle、A/A或P1样本，不能据此报速度或性能PASS。
重型任务继续串行；全部交付结束后先probe预检，再启动原正式仪器。G08粗估约88%，
整个G07–G10 goal约47%，正式仍仅G07完成1/4；G09/G10及CLI未开始。

### 2026-10-17 十八项交付完成与实际性能预检

[最终交付凭据](../../../../evidence/delivery/inference-g08-sdk-final/receipt.json)汇总六profile各
source/static/shared共18项全部PASS，归档hash与18组唯一身份已核对。最后WSL CUDA source
沿原树完成252步及六个公开消费者、13个独立头；保留两次原私有helper/预检失败与累计
570.062s，不重置预算。原LibTorch/CUDA多版本链接警告保留在日志，消费运行按既有SDK合同
通过；不把此项扩大为hermetic LibTorch或实际Windows module tracing声明。

CPU diagnostic probe已经实际启动，固定benchmark `df7ca0271`、original85 peer和合格static
SDK；注册CPU2/ABOVE_NORMAL继承canary为PASS。尚未取得workload/oracle/A/A及正式P1
结果，probe不作性能证据。后续其余三路线先按同样边界probe，再进入正式统一仪器。

普通selector发现新`bench/inference_output_selection`目录无factor ownership；将canonical
基准目录归入现有`bench/compute_ops`，23文件逐字节与df7的原目录一致，证明附在工作区
检查点。没有修改selector、组件图或产品，正在使用df7的干净P1副本与原配置不受影响。

当前G08粗估约90%，整个G07–G10 goal约48%，正式仍只接受G07一项。余四路线性能资格/
正式裁决、最终文档聚合与终审；G09/G10及CLI未开始。

[首次实际probe与定向修正](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-probe-setup-checkpoint.json)：
CPU原对照内部provider的ggml.h/ggml-cpu.h私有编译头未接入，首次build失败14.016s，
没有进入数值oracle、A/A或产品计时。18项公开SDK消费者不受影响。仅给原对照target接入
同CPU producer的e74fc9派生include，contract显式绑定，22份native头加入现有逐artifact
hash核查；不修改产品或统计仪器。原失败/output/build完整保留。新目录/私有头配置需要
新的不可变benchmark输入及明确setup-resume tree，原费用累计；不把它写成P1通过。

[CPU实际diagnostic结果](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-cpu-probe-checkpoint.json)：
修正后的干净benchmark固定`5fbc139921`，仅CPU原对照私有头/目录输入变化，所有产品路径未变。
CPU两个实际binary独立数值oracle、16次workload A/A及stock verify均PASS，运行103.015s，
连原失败累计117.031s。stock probe仍为DIAGNOSTIC/n1，generic aa/delta5/delta10因独立样本
不足写FAIL，timer-floor/placement为PASS；不能据此签正式性能。原private远程启动在sparse
checkout内寻找未物化交付凭据，三路线均在probe/build前失败，共2.875s及日志保留；按同一
receipt的SHA验证外置路径后续跑。WSL/Air干净source均已固定同一5fbc；CUDA实际probe
已经启动，其后串行Apple/Vulkan。没有启动正式qualification或G09/G10。

### 2026-10-17 GPU预检集中失败与同树修复

[三路线实际build失败及有限修正](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-gpu-build-fixes.json)：
CUDA/Apple/Vulkan分别用62.655s/5.317s/4.148s，均在build停止，未进入数值、A/A或产品采样。
一次修三处：CUDA benchmark补原公开backend CUDA头；Apple原两TU的第三个global predicate
仅作私有符号隔离，并扩实际nm检查；Vulkan补原85bc的view_alias与两IREe私有桥头，逐字节
证明和当前共用私有桥相同。原33个对照TU及数学/完成/双输出读取/owner边界不变。

三GPU路线复用各自`probe-build-setup-resume-1`及已有对象，另存新诊断output/job，保留
原失败和累计费用；产品candidate、公共仪器、5%、n15/31均未改。CPU保持已验证的5fbc
benchmark用于其后正式P1；本批只修GPU激活分支与对照构建，不重复CPU/SDK/全库验证。
四路线正式性能与最终聚合仍待完成，G09/G10未开始。

### 2026-10-17 四路线实际preflight完成，正式P1准入

[最终preflight](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-preflight-final.json)
绑定原时序账/静态成本、18项交付、不可变CPU5fbc/GPU47b与干净original85。四路线实际
build、两binary共九形状的source-bits/first-index oracle、native/selected各八strata的
workload A/A与stock verify全PASS；Apple实际nm同时证明原tensor→原indexing、第三predicate
独立隔离，Vulkan26原TU和三共用桥头均实际编译。CUDA原对象沿同树复用。

CPU/CUDA/Apple/Vulkan累计诊断分别117.031s/186.199s/66.685s/88.363s，保留全部原失败。
CUDA实际workspace R1由12039680B降到9728B，R16由192249344B降到512B；双方provider/
implementation与ordinary入口一致，persistent416B/private8B不变，结果读取仅8/128B。
这些是实际资源事实，尚非速度结论。

stock probe只有n1：generic aa/delta5/delta10仍如实FAIL/独立样本不足，计时分辨率与
placement为PASS；两执行顺序的校准中位数都在原预声明点值带，5→10单调，workload
自身A/A无固定factor偏差。当前允许开始一次四路线正式`qualify-platform`：原5%、n15–31，
各原profile、八strata/placement、两小输出及完成/owner账不变；重型仍CPU→CUDA→Apple→
Vulkan串行。新qualified-build是原仪器fresh要求，不覆盖或重置旧probe失败/树/费用。
正式资格/性能及终审未签，八owner仍CANDIDATE，G09/G10未开始。

### 2026-10-17 首路线正式P1：CPU已接受，GPU批次继续

[CPU正式证据](../../../../evidence/performance/inference-g08-output-selection/cpu/receipt.json)与
[性能分账](../../evidence/performance/inference-output-selection-g08-2026-10.md)：固定CPU
benchmark5fbc、产品5103及原85bc，八strata各n15、120产品进程与30公共控制完成，
qualification/controls/stock verify/5%非劣门全PASS，n15停止。R1与R16严格方向均WIN，
各分层吞吐ratio分别1.218660–1.255562、1.218030–1.266141，均非整引擎加速比。
CPU新旧都O(RV)，所省的是V宽Entry scratch/fill与partial_sort触达。

CPU正式阶段508.265s及原117.031s诊断分别保留；没有复位原失败。CUDA正式P1已经接上，
其后Apple、Vulkan按原批次串行，不能提前接受整项G08。八owner保持CANDIDATE；
G09/G10与CLI仍未开始。

### 2026-10-17 四路线正式P1及终审补证完成

[四路线性能分账](../../evidence/performance/inference-output-selection-g08-2026-10.md)与原始
CPU/CUDA/Apple/Vulkan receipt均已封存：每路线八strata各n15、120产品进程与30控制，
qualification、五controls、stock verify及原5%非劣门全部PASS。九形状相对1的严格方向均WIN；
Apple K3约1.043倍、原instrument exact verdict为PARITY，两口径各自保留，不互相替换。
比较对象为原HYLE85bc，没有推断整模型或llama.cpp的token/s。

[独立消费终审](../../../../evidence/architecture/inference-g08-output-selection-audit/independent-consumer-review.json)
未发现有限S1同合同的MUST；它发生于正式采样开始后，原文保留当时Apple/Vulkan尚未terminal的
状态，不回写。该历史pending由当前四路线实际terminal补齐。
[实际artifact补证](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-artifact-postreview.json)
确认四路线同优化设置、真实submit/wait/两输出读取和局部owner释放未被消除；CPU execute移动16B，
CUDA计时callback移动16B（不声称其GenericArm或GPU代码也移动），Apple/Vulkan execute与callback
移动20B，text分别仅增加32/0/20/20B。section/反汇编记录也是采样后补证；作为工程终审证据修正接受，
保留未在preflight附齐该记录的流程偏差，不声称完整Formal release预签。

四路线正式阶段合计2411.786s，连已记录诊断合计2870.065s；串行controller墙时2455.485s
另列，未吞掉原失败或冒充整轮历史总成本。CUDA172MB含binary原ZIP在receipt明确标识的本地
内容寻址缓存，live qualified artifacts保留；提交raw含全部样本/身份，工程证据不宣称Formal长期保存。
其它三路线提交归档含实际binary。最终43项产品/依赖输入hash仍匹配5103，当前只剩LOC、
terminal七门/七问和八owner的CANDIDATE→CLOSED落账，G09/G10与CLI尚未开始。
