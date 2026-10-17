# G01 驻留 SDPA：视图消费与原生 fused 路径

状态：CPU owner CANDIDATE，Vulkan有限子批CLOSED，2026-10-13。owner：`compute_provider_ggml`、
`compute_provider_ggml_vulkan`；baseline：`070609442806ebfe6a0dc8d6594b7388c494f20b`。
CPU不可变candidate：`d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`。
Vulkan RR不可变candidate：`a40003f7edb54a3da3113d0cd401b46211e0b0f0`；原Vulkan精度
candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`和全部原证据保留。
当前Vulkan Provider/Route端口修正candidate为
`3d92eee97a70f7788dfb9bbcda73f6e0f801959f`，按[端口卡](inference-g01-vulkan-resident-ports.md)
验收；runtime的a400数学/shader/Work不改。本卡继续拥有CPU原candidate及resident总合同。
授权与完整目标见[缺口执行总卡](../project/inference-gaps-goal-execution-2026-10.md)。
本卡先修 G01 中已有 SDPA 的实际消费和成本缺陷，不签整个 G01，也不替代多请求分页及
Apple/Vulkan 联合分页的后续 owner 卡。

## 真实触发与消费域

单请求 cache 按容量驻留，decode 只改变有效长度；公开 TensorDesc 已能表示字节 strides，
CPU ggml 的 tensor 导入也设置 native nb，但 SDPA qualification 逐个拒绝非 dense Q/K/V。
因此 head-first 的容量 cache 截短 seq_len 后仍须重排/复制；不能由“存在 view 类型”判断
该旅程完整。主消费者为 Qwen3.5-2B full-attention 片段，GQA8/2 D256，prefill 和单 token
decode；留出为两个 batch 的非均匀 padding、两种 attention layout、不同 Q/KV 长度和
独立 slot/旧 output。权限仍用原 ready ReadLease，跨页与 arbitrary page-table 另验。

Vulkan 当前这条 SDPA lower 显式生成 F32 scores/softmax/MatMul，即使锁定 ggml Vulkan 已
有 FLASH_ATTN_EXT。这是同数学同 owner 的承载差距，不由 benchmark 排名触发。原生核
要求 Q/output F32、mask F16、K/V 支持编码、D 为8倍数及实际 subgroup/cooperative 资格；
不能直接用“上游支持”代签本机资格。CPU 已有部分 F32/F16 fused 档，Apple 已借 MLX fast
SDPA，均保其已有实现和限制；本卡不新增第三方库或自写 attention 数学。

固定参考仍为总卡的 SGLang/vLLM/llama.cpp SHA。实际 vLLM CPU 核从 block_table 直接
读取，显式 query/cache strides、ISA dispatch 和 OpenMP workers，说明输入形状和执行预算
必须同时核对；不把整个 vLLM/Torch 运行时作为本轮借用对象。当前锁定 ggml CPU 原核
要求 depth 连续、接受外层 nb；CPU qualifier 必须与实际 lowering/allocator 一起反证。

## 本轮边界与审计输出

职责不变：原 Provider 拥有 native 支持域、lower、workspace/ports、错误与 actual retirement；
Graph/Compute 原 executor、lease 和 Queue 不改。公开 ScaledDotProductAttention/布局不新增
模型参数或兼容签名。SCOPED提交`f65616d2b`保留；下表固定后才允许产品改动。

有限审计：CPU contiguous-depth、非重叠正 strides 的 rank4 view 能否沿 native 导入和 fused
分支消费；BF16 是否原核直接支持、原 causal mask 与 query cast 的费用；Vulkan fused
替换时 mask 常量/上传、GQA、输出顺序、dtype 和设备 capability 的全链资格。若需改变
公开 shape、资源身份或依赖方向，先回本卡定界，不在 qualifier 中偷偷接受未证输入。
有限 FIX 已按实际源码固定如下；其他能力不得顺手扩张。

| FIX | 最终形状与承载 | 验收与负空间 |
|---|---|---|
| RS-01 | CPU原SDPA接受rank4 plain、depth连续、外层正的row-major padding strides；原native输入已设置nb，不再强制全dense。原同dtype F32/F16/BF16、两layout、GQA和有效长度语义不变 | 容量cache截短，padding哨兵不参与数学；拒绝depth跳跃/负或重叠strides，不公开whole paged arena |
| RS-02 | 借锁定CPU ggml flash原核直接读K/V，F16/BF16域覆盖causal/noncausal，F32保现有锁资格。query转换允许O(QHD)，不转换/展开完整KV；causal F16 mask按Q×K显式计费 | 原核depth连续、BF16原vec-dot/to-float资格已在实际锁源码核对；两layout、不同Dqk/Dv和保留原小shape域用oracle反证；不得仅靠if条件称 fused 成功 |
| RS-03 | Vulkan仅对同F32/F16、BSHD且同D64/128/256的SDPA接原FLASH_ATTN_EXT和显式full-F32 private tag；Q转F32、K/V保原编码，返回原output顺序/类型。causal mask以原native Fill→DiagMask→Cast F16生成，非causal无mask；实际device supports_op仍冷拒绝无原生资格设备 | 原核half缩窄的实际失败后按native精度卡收紧高效资格，原oracle不放宽；Qwen GQA8/2 D256与留出D64/128；其余既有传统lower域保留并明示成本，不能代签G01全部域。mask和query cast计费，避免Hq×Q×K的scores及按Q/H展开KV |
| RS-04 | Vulkan原typed region在SDPA有实际需求的rank4正row-major padding域接native nb；imports/端口范围仍按ggml_nbytes校验并保原owner。其他operation原qualifier不放宽，非dense输入若被未资格节点读取必须拒绝 | TensorDesc required_bytes/原VkBuffer范围、非零offset、padding、源只读与旧输出寿命；native graph库支持检查不绕过 |

source/static/shared仍原compute/graph产品选择，没有新factor/第三方/FFI/public类型。普通调用
仍`ScaledDotProductAttention`加TensorDesc；新view支持只为原合法数学提供自然消费，descriptor
本身不授权页capacity。冷graph/slot身份不改，只有原Provider定向资格和lower；warm native
submit仍原Queue和lease use，OOM/未支持/失败由原Result/Status路径回滚，借用和关闭合同不变。

具体审计锚点：CPU `create_tensor_input`原nb导入、`supports`1494、lower821、cold mask1646；
实际锁CPU `ggml-cpu/ops.cpp` flash的depth连续断言及K vec-dot/V to-float、BF16 traits；
Vulkan `tensor_domain`/`native_tensor`目前dense-only且未设置nb、SDPA lower857、
`native_storage_impl.h`原device supports_op/owner/range拒绝；锁Vulkan flash19420实际dtype/
D/device检查。Vulkan新增native stride导入属于RS-04，不能误称已有端到端能力。

不纳入任意负/重叠 strides、任意 attention 特例、quant KV、新 scheduler、量化格式扩展、
原生分页或全引擎吞吐。这些当期 G 项仍保总卡工作，不因本卡有限收口删除。
直接依赖/消费者沿中央 component DAG 和 change-impact selector，不新增重复名单。

## 预声明验证、成本与停止

先精确复现原 view 拒绝，再以同 oracle 比较有效 KV 与 padding；包含短 decode、长 prefill、
两布局、GQA、尾部/无效区、独立 slots、保旧输出、失败/OOM及原 cancel/close/retirement。
模块 tests/逐头与真实直接消费者按 selector；CPU 实际 Windows/WSL，Vulkan 实际 Windows，
Apple 只在影响闭包适用时用 MacBook Air，不用 CPU fallback 填 GPU 格。公开消费者要跑
source、relocated static/shared，复用现有树/锁/缓存；适用宿主 ASan/UBSan/TSan，GPU/driver
私有内存与既有外库限制仍明确。文档/closure checker 每轮照常，不重写公共工具。

P0 先核 input copy、Q/KV cast、mask、scores、scratch、native private UNKNOWN 与成本阶。
冻结 candidate 后针对新增 fused/view 档做一次同合同定向 P1：自然 native ggml flash 为 peer，
实际 admission→completion→output owner return，短 decode/长 prefill 和一组留出；A/A 与双
顺序先验噪声，预声明非劣 margin 10%，不以阈值把稳定小幅 LOSS 改成 parity。无法证明
同合同的格如实 INCONCLUSIVE，不借不同 dtype/位置结果签性能。原参考档不无故重测。

调查检查点2小时、实现检查点4小时、候选验证检查点2小时；按用户常设授权正常推进时
自动延长并保原日志、FAIL/TIMEOUT和累计 elapsed。原本机12 jobs起；用户反馈桌面卡顿后
本机改4 jobs，Air10，重型本机GPU run暂不恢复。
有限缺陷已裁决且全部适用门通过才 CLOSED；真实外部阻断按协议记录精确条件，不靠换
日志目录重置，不把“继续调研”当终局。七问/固定 FIX/最终资格集中追加本卡与一份证据。

实现阶段记录：CPU resident view/native F16/BF16 fused改动已实现；宿主模块全部unit与逐头
通过，普通CPU入口的BF16 GQA8/2 D256 resident cache实跑选择ggml且数学通过；旧installed
SDK同调用会选择ATen。Vulkan source已接view与原flash，但定向oracle失败，尚未candidate：
锁定原核的F32 accumulation不防止Q/K/V/概率内部half缩窄。原数值容差不放宽，需原native
dispatch资格补齐；此新事实按实际承载owner定界。测试另纠正独立Tensor与ResultLease的
寿命假设，聚合数值错误只减少重复输出，数学oracle/容差未改；该更新测试尚未重跑。
MSVC旧测试夹具的unused observer、显式double→float及vendored private include归属修正
仅为test-only，不改其他CLOSED产品；全部原FAIL保留。后续三交付/san/P1仍待candidate。
当前[实现阶段证据](../../../../evidence/correctness/inference-g01-resident-stage/README.md)
保留CPU实际PASS和Vulkan实际FAIL，更新后的Vulkan测试尚未重跑，不由源码推导PASS。
原锁定承载的精度选择归属已在[原生精度卡](inference-g01-vulkan-native-precision.md)
已依次SCOPED→AUDITED，原Provider有限FIX按精度反例冻结为三个D和同D；不会放宽容差
或将未资格native域签PASS。其余已准入域需要的高效消费仍保总goal，不因本子批删除。

原生修正后Vulkan同F32/F16 D64/128/256的定向public consumer全部通过原容差；准确长度/
offset/padding、source只读、旧输出与独立slots/重放均实跑。生命周期拒绝位置按原native
实现从bind修正为submit，未改产品合同；原错误断言保留。normal/strict scalar cache交错
与双顺序反例也通过。这个有限resident子批进入候选准备，三交付/san/P1未完成，不签
整个G01。CPU原模块/逐头与默认入口的已过证据仍有效，不额外重跑旧大模型矩阵。

候选验收新增：Vulkan source/static/shared与Win地址ASan/Air UBSan/TSan已经分别实跑
通过，详细收据见原生精度卡；CPU[WSL ASan/LSan](../../../../evidence/correctness/inference-g01-cpu-host-asan/README.md)
四个实际attention风险tests通过。同一CPU-only GCC14.2验证树的
[UBSan](../../../../evidence/correctness/inference-g01-cpu-host-ubsan/README.md)与
[TSan](../../../../evidence/correctness/inference-g01-cpu-host-tsan/README.md)也已实跑通过这四个
tests，两个变更TU实际插桩核对；不代签全CPU算子或Graph生命周期。
[CPU普通consumer三交付](../../../../evidence/correctness/inference-g01-cpu-resident-delivery/README.md)
也已实跑通过：同BF16 GQA8/2 D256、两layout、Q1/17、KV31/容量37、源只读及重放；source
与实际改名relocated static/shared均退出0，公开默认入口选择成熟ggml。旧树hook/symlink的
原FAIL及产物身份保留，编译最多4路且无GPU运行。P1与最终review仍待；本卡保持
CANDIDATE，完成数仍0/10，不把阶段证据算整个G01完成。

候选LOC按原统一脚本一次计量：三个resident承载owner的实现文件和CPU provider公开头
联合为4907行（不是本次净新增行数，也不是推理引擎预计行数）。
[原始报告](../../../../evidence/correctness/inference-g01-cpu-resident-delivery/resident-owner-loc.json)
绑定candidate及显式边界，self-check通过；测试/bench/third_party patch不混入产品实现数。

[性能适配器预检](../../../../evidence/correctness/inference-g01-p1-adapter/README.md)已过Windows
static编译与Air原oracle/稳定重放，覆盖decode、长prefill和留出。原生recipe保split-K、mask-opt、
相同F32精度和Program输出寿命；首用scratch增长单列。对照stride/alias及私有配置的原FAIL
保留，产品candidate不变。这里只是采样前的数学/形状预检；统一仪器合同、A/A、双顺序/
placement与正式controls仍未完成，不能据此签P1或CLOSED。

采样前合同细化：现有nanobench runner v1实际只准5%的practical/non-inferiority门；不为
本模块重写仪器。原卡10%非劣预算保留，先按更严格的5%合同验收；5% FAIL不会被模块
改名PASS，真实方向/区间和原FAIL保持。正式样本按原n=15停止/追加规则，A/A先验检查
使用现有5%偏置/10%gross instability界线；不能由原oracle预检推导性能资格。

首次正式fresh构建在原compute `LeaseUsePool::create` 的uint32容量与64位SIZE_MAX上界
比较触发Clang21 `-Wtautological-constant-out-of-range-compare`，样本尚未开始。此前Air
correctness树WERROR=OFF；本轮保WERROR=ON，只在明确M1编译profile登记
`-Wno-error=tautological-constant-out-of-range-compare`，诊断仍输出。该上界在64位恒不可达，
32位仍有意义；不为此改无关CLOSED产品或重开通用工具。原构建FAIL/100.28s保留，
产品candidate与三种数学形状不变；性能合同尚未取样就更新profile/hash，再做fresh资格。

正式Air[resident Vulkan P1](../../../../evidence/performance/inference-g01-resident-vulkan-m1-p1/README.md)
已完成原verify回放：资格PASS、performance FAIL。decode吞吐比0.687–0.710，明确超原10%
预算；长prefill通过5%非劣但仍有小幅方向LOSS，D64留出一个placement/order格未过5%。
16个A/A、120个产品process和30个controls及原FAIL/耗时完整保留，不签CLOSED。
原native graph_compute每次遍历/编码并retire cleanup，对照可冷record固定recipe；需要核
原command-buffer/descriptor/byte-origin可复用条件后固定有限纠正，不能只凭点值改公共API
或把managed费用漏计。产品改动前须退回AUDITED并重新冻结FIX；目前只读定位，candidate未改。

## 有限 native record 费用纠正（AUDITED）

触发为上述合格decode LOSS及原native源码每次重编码/cleanup事实；baseline
`49af6056b`（完整revision由Git解析），旧不可变candidate和全部PASS/FAIL/elapsed保留。
CPU产品不改，仍保持原candidate；Vulkan Provider与原native runtime回AUDITED。
不新增公共API/layout、Provider数学、shader、capture承诺或第二runtime。

原native已有`ggml_vk_begin_submission(..., one_time=false)`，每backend拥有自己的
command pool、descriptor pools/sets。以下有限FIX只复用这套owner与原shader记录，不能
把global pipeline对象当可随意共享的mutable recording。

| FIX | 固定合同与实现边界 | 验收 |
|---|---|---|
| RR-01 | 原NativeWork对同full-F32 tag的单SDPA及其Fill/DiagMask/Cast/Cont/Permute/View固定graph可保原compute记录；只在无host staging、native event/semaphore依赖且实际退休后启用重放。其他graph/模式沿原managed执行 | 首用完整数学、first-use费用、连续重放、三D/两dtype；未资格域不伪造prepared/capture |
| RR-02 | 原Region.bind在改port前按VkBuffer、byte origin、capacity与strong owner身份检查记录失效；变化后沿原native重新编码，不保旧输入到额外业务周期。固定graph/math metadata不由warm payload决定 | 两独立slots、不同backing/非零offset/别名与同buffer内容更新；旧output拒绝submit不变，readonly与writer寿命不减弱 |
| RR-03 | 保原Queue提交锁/一个真实fence/accepted-prefix保活与异常drain；cached command pool/descriptor只属该Work，退休前不reset，失败/device-loss不复用，销毁仍按原native退休 | repeated/rebind/close、partial/OOM与原风险反例；不额外每kernel marker，不假设异常代表已退休 |

冷record与失效后再record的费用明确；warm固定port只O(ports)资格检查、原native submit/
wait和output归还。常态private字段/记录buffer归原Work，真实bytes或UNKNOWN保原报告，
不让普通其他operation付新规划/数学。再冻结candidate后复用原三交付/风险的适用入口，
追加record/rebind反例及一次同合同P1；原比较门/失败预算不重置，不重跑无影响全库。

RR实现事实：原backend默认保留1024-byte Add/RmsNorm partials，并在每次graph执行时
无条件清零；Air UMA把这一步变成host memset，导致纯attention记录也不满足无host
staging资格。仅在上述固定graph已经证明没有Add/RmsNorm consumer时，执行原graph
期间将该clear的size置零，返回或异常时恢复原值；其他graph仍走原处理。数学和shader
未改，也没有忽略真实host update。第一次记录失败、持久输入导入后的失败及诊断记录
全部保留，临时诊断字段/API已经移除。128次native内容更新/owner切换与公开64步
offset/backing切换已在Air实跑通过；上述文字记录实现阶段事实。

RR候选固定为 `a40003f7edb54a3da3113d0cd401b46211e0b0f0`，两个承载owner由AUDITED
进入CANDIDATE。[实现阶段收据](../../../../evidence/correctness/inference-g01-vulkan-replay-stage/README.md)
保留所有失败和最终Air通过、Windows4路编译通过。原public API/math/shader不变，private
Work常态增加固定记录与弱graph身份；其他operation冷评估一次、warm不重扫。RR适用
三交付/sanitizer与原合同P1待补，整个G01仍未完成；CPU保持原candidate而不无故重测。

RR candidate的[实际交付与风险收据](../../../../evidence/correctness/inference-g01-vulkan-replay-candidate/README.md)
现已补齐Air source、改名搬迁static/shared SDK及宿主ASan/UBSan/TSan有限公开/native
测试，原始输入逐SHA核对candidate；Windows仅签RR4路编译，原Windows ASan仍绑定旧
candidate。Air风险不外推shader/driver，ASan不签LSan。正式RR P1已在独立clean Git
checkout、新fresh tree启动，原peer与合同canonical hash
`4ae724d684be30814e2f28d7d933f771b7e03dda56b86cb51464cef706031720`不变；
原provisional archetype来自旧candidate，其数学/输入身份继续复用，新产品身份由此次
manifest绑定。前一轮完整性能FAIL/构建tree保持原字节。尚无RR性能结论，不签CLOSED。

CPU采样前合同已按本卡主/留出形状冻结草稿：BF16 BHSD GQA8/2，Q1/128、KV1024、
D256 causal，另Q17/KV257/D64 noncausal。仍是独立原flash graph与相同公开
PreparedOperation/CPUQueue/Completion/caller output；两个slots、coherent映射、
单native thread、原8e-3 oracle、readonly；query/output与mask/cplan计费，private pool
上界仍UNKNOWN。新bench只增加上述留出、实际affinity/order/placement遥测与公共controls，
CPU产品candidate不变。16个workload A/A预检后才按八strata×n15原仪器取样，5%更严门
和本卡10%预算保留。新版三形状/placement/control现以2 jobs实际编译通过，54个依赖/
benchmark步骤160.39s，三个独立oracle/readonly/双slots预检3.42s通过；正式A/A/P1未跑。
[三形状原始预检](../../../../evidence/correctness/inference-g01-cpu-p1-three-case-adapter/README.md)
保留实际日志、源SHA、原peer真实Git pin及selector BLOCKED计划，不签正式P1。
[CPU时序/成本/资格账](../../../../bench/inference_resident_sdpa/cpu_p1/README.md)固定双方
完整完成边界、warm费用和UNKNOWN；它不替代实际A/A或全样本裁决。
共享bench目录的selector归属缺口已另记独立工具任务（已删除的历史文件：`docs/implementation/tasks/build/validation-selector-shared-bench-paths.md`），
原BLOCKED计划保留；本模块不修改公共工具，适用入口按上述有限合同直接取证。

RR[正式P1](../../../../evidence/performance/inference-g01-resident-vulkan-replay-m1-p1/README.md)
资格PASS/verify通过而性能仍FAIL，decode吞吐比0.719–0.727，prefill5%非劣
PASS但方向LOSS，D64留出非劣UNRESOLVED且方向LOSS。新源码事实是外层canonical
adapter仍会复制全KV，已按真实owner登记[驻留端口卡](inference-g01-vulkan-resident-ports.md)
SCOPED；原本次数值、交付和san PASS仍只签其有限数学/生命周期，不签零拷贝成本。

CPU正式v1在冷诊断/A-A途中主动停止，原构建/日志/851.5s保留于
[排期停止记录](../../../../evidence/correctness/inference-g01-cpu-p1-cadence-stop/README.md)。
长prefill约133ms/op，104paired rounds会给每个process约55s数学，正式八strata×n15
需约两小时；这不是产品性能FAIL。v2在正式product samples前固定16paired rounds与
实际遥测校验，5%/原10%预算、数学/peer、n15/31、原controls不改。CPU产品d91a不改，
新benchmark需独立clean source身份重新资格；公共仪器不改，不把v1未完成项签PASS。

CPU后续真实多请求缺口另起[ragged原生卡](inference-g01-cpu-native-ragged.md)；原单请求
candidate和P1独立checkout保留，只有完成该卡AUDITED后才改变Provider。单请求与新batch
职责/证据分开，不借新增入口重测无影响全库或删除尚待paged/native GPU域。

CPU单请求[实际W1 P1](../../../../evidence/performance/inference-g01-resident-cpu-w1-p1/README.md)
已完成：独立source af9、原CPU产品d91a、v2固定16rounds，16个A/A、n15的120产品/
30 controls，资格与三点5%非劣均PASS、原verify退出0，实际2598.99s。decode吞吐比
0.9977–1.0021、prefill0.9987–1.0040、留出0.9977–1.0011，方向均INCONCLUSIVE。
原STOP/锁失败/未知private成本保留。不代签新ragged多请求；其实现仍按独立活动卡验收。
