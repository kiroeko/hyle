# CUDA U3 R1：当前模型片段、状态与 MTP 接缝

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`c0150bc93ffa099ada45ac401bba15a36a9ec996`；Windows唯一产品真相。
- 当前owner：`compute_provider_torch_cuda`、`compute_provider_ggml_cuda`、`graph_provider_ggml_cuda`；
  route仍保留N2 capability／U2合同。只允许下面已固定FIX，不从旧卡取得新数学授权。
- trigger：[U3 R1](cuda-final-u3-mature-bindings.md) 要求继承CPU当前完整需求。
  CUDA测试已include `cpu_c4_model_branches.h`却未接当前Pro0813／Flash0731完整DSpark及三sidecar
  runner；旧preview测试、原矩阵32项不能代签当前state／carry／完整head。
- 固定输入：共同目录／family-witness-index的八族49个region关联、官方mixed dtype、当前四新
  branch profile、Qwen Q3/Q4/Q6 named sidecar、两个Flash Q8/BF16 sidecar、本地Qwen3.5原版。
  同时保留全GGUF／non-GGUF基础合同；不另造模型画像或release。
- 主纵切：普通CUDA Direct／Graph完整program，权重／precision／真实内宽／cache／carry／
  confidence／Markov／三stage／完整vocabulary保留；少量token／驻留expert和head分片按原合同。
- 正交纵切：变化输入／IDs、状态反馈、两个独立slot、OOM／失败输出、close后accepted work保活。
- scope：共同program与当前CUDA admission逐step对账；成熟候选真实缺口或错误接线才能FIX。
  权重准备与热program分开，Provider原融合／资源管理不因HYLE调度需要拆坏。
- non-goals：自研CUDA数学冻结；不改CPU产品／公共语义或旧Vulkan BLOCKED；不写模型engine、
  loader／tokenizer／CPU+GPU模型分担、新通用Executor／验证框架或无需求大算子目录。
- resources：复用CUDA primary／stream／Mem及Compute/Graph Completion；actualstate反馈，失败output
  不消费；unknown private allocation／capture如实声明，cold binding与hot execution边界保持。
- tier：P0与correctness；缺口首次单成熟承载不造P1，当前重复域在U4同合同全成本有限裁决。
- budget：一次program／source／admission批量审计，固定有限FIX后一次批量接线／集中双平台验证，
  至多一批具名纠正。没有真实成熟承载则具名有限决策，不扩自研或无限换probe。
- predeclare：Windows／Linux原program两入口、当前dtype／物理格式、完整state/head、失败／OOM／
  repeated execution／capture限制及直接消费者；最终U3 immutable candidate集中source/static/shared、
  CUDA-only／mixed、offline／relocation／邻exe DLL，必须在U2/U3关闭前完成，不留U6首次接线。
  docs／closure／selector、统一Compute+Graph非测试LOC。未运行不签PASS。
- stop：原输入与当前CUDA map固定后AUDITED有限FIX才能写产品；所有联合资格前不签CLOSED。

## SCOPED 审计清单

| ID | 当前事实 | 本批审计 |
|---|---|---|
| A1 | C4当前model/sidecar program未在CUDA runner注册 | 复用相同数学／profile／callback，逐step typed admission与准备／热边界 |
| A2 | 旧D5八族完整region有CUDA runner但非本候选证明 | 按49关联去重映射当前test／code，保留未验证／缺口，不复读所有历史任务 |
| A3 | 原GGUF与新版FP8／FP4／E8M0来源/codec可能精度不同 | 对照已固定原carrier与当前role／rounding，禁止同文件Q标签或同op名代签 |
| A4 | 生命周期/独立slot/软件事实/SDK组成是路线合同 | 复用N1/N2机制证据，新增state／consumer适用项具名补齐，三交付联合验收 |

旧CPU已闭数学与性能只作为当前需求／源对照；本卡不重开CPU。原源研究以U1已有固定输入为准，
只为具体step反例做有界查证。审计后KEEP/FIX/BORROW/DEFER/REJECT写在本卡与小型coverage map。

## AUDITED 有限 FIX

SCOPED提交`f3a3a697`，原common程序测试绑定候选`fd08e31a`；两平台目标仍不扩大。
current Pro/Flash的target_hidden／confidence和Markov完整词表两入口通过，cache/主干/三stage
审计暴露下面具名事实。原失败保存于本轮evidence，不由绿原子或旧CPU PASS代签。

| FIX | 证据／问题 | 有限动作 |
|---|---|---|
| F1 | 原Softplus在当前router/三stage落到hyle-cuda，ggml无通用stable precision资格 | 现Torch CUDA carrier接原ATen stable Softplus；原SDK／shape／dtype／Completion复用，不写新数学 |
| F2 | plain BF16 GroupedMatMul明确要求F32 activation；原CUDA MMF在mmf.cuh把float2 activation cast成nv_bfloat162 | owning lowering先用原ggml cast把BF16 weights变F32，再原MUL_MAT_ID；量化权重/A8原格式路径不变。当前source精度标志不能改变MMF的RHS cast，不能假设设flag已兑现 |
| F3 | CUDA runner漏传expect_output的port index，current source validator收不到真实port；硬写decode忽略prefill | 只迁移test runner，phase/observe/validate/state来自原common程序；不放宽误差或修改CPU fixture |
| F4 | CUDA未接三sidecar与当前49 family-region的必要consumer | 批量接原program与GPU codec资格；CPU AVX2/vec-dot oracle不能外推CUDA Q8路径，具名复用原GPU codec参考后才签 |
| F5 | 最终状态/资源与全部基础格式/软件事实/三交付未联合 | 同一最终U3 candidate按既定输入批量验证，Windows/Linux/source/static/shared/CUDA-only/mixed/失败/replay；未运行UNVERIFIED |

F2的原源码账单：ggml `456172ec` CUDA MMF的BF16分支用`ggml_cuda_cast<nv_bfloat162>(float2)`
构造tensor-core tile，F32 accumulation并不保留F32乘数。public GroupedMatMul的plain BF16合同
要求F32乘数。原fixture输出3在0.0013427734附近出现6.1035e-5偏差，仍保留当前门；不通过
更改全局tolerance或把CPU约定降成BF16乘数来签通过。原cast+MUL_MAT_ID的完整资源/搬运成本
属于本program，U4公平比较必须计入。这个cast不是HYLE新codec或自研GEMM。

## 联合交付批次的复用与停止规则

2026-10-08 用户要求避免小调整反复重跑全集。本轮交付执行采用下面可检查的合同，
不改公共 selector、closure checker 或性能统计核心：

- 固定候选、consumer 输入、工具链、配置、机器与实际 executable／runtime 身份；
  按 artifacts、fixed、current、families、lifecycle 分组保存原日志、XML、哈希与结果。
  分组用例并集必须与请求全集逐名相等，失败／skip 不得作为已完成组复用。
- 每次先写 RUN／REUSE／STOP 及理由。同身份已通过组复用，驱动中断从未完成组继续；
  失败组先查源码／资源／参考合同，重跑须具名因果与有限理由，不能靠重跑变绿注销原失败。
- 文档／状态不 build／P1。fixture 或驱动变化只验其受影响组；产品变更按 selector／本卡
  直接消费者扩展，并记录证据失效范围。公共 API／ABI、依赖版本或跨组资源机制变化才能
  扩大到其全部适用合同；重新配置本身不能触发无理由的全集重跑。
- source、relocated static、relocated shared、CUDA-only／mixed 的身份分别记录，不能互相
  冒充交付证明；相同合同数学资格可引用已封存 native 证据，实际 link／loader／配置边界仍须运行。

首个独立源码 consumer 整批650项中649项通过；Q3_K expert `g3560f36d25c826bd` 在
replay1第二slot出现大误差。原失败保留；相同未变 executable 的单例和560项物理编码组
均通过，尚不能唯一定位产品／reference／跨carrier状态原因，不签该原失败已解决。
当前仅做具名 carrier-transition 定向诊断，不重跑650项全集，也不放宽误差门。
该35项 transition（原N1／N2加失败单例）亦通过，仍不能唯一归因。联合交付的专用
[驱动与边界记录](../../../../evidence/correctness/cuda-final-u3-joint-delivery/README.md)
保存执行约束；整批中完全通过的组可独立引用，原失败 aggregate 不因此注销。

### Q3_K 整批失败的具名根因与有限纠正

后续原日志逐值审计：4096个失败全部是replay1的slot1，actual／expected都按8个prototype
重复。fresh原CUDA Q3_K结果与原actual八个prototype全部吻合；相同Q3_K bytes按IQ3_S
执行的原CUDA结果与原expected全部吻合（JSON打印误差分别约5.7e-11／4.7e-7）。
Q3_K与IQ3_S每256元素均为110 bytes，K=2048时shape／nb相同。原`common.cuh`
capture properties保存src data／ne／nb，未保存src type；`ggml-cuda.cu`以首node地址为key。
固定同址、IQ3_S暖机两次后将source type换Q3_K的最小原程序复现仍返回IQ3_S结果，
与旧expected吻合。因此不是HYLE输出或误差门问题，而是test-only thread_local reference
backend跨已销毁逻辑program复用capture身份。产品每Invocation独占backend／immutable recipe，
不受这个reference跨program寿命错误约束；没有授权修改原kernel或产品cache。

有限纠正只有共享CUDA reference的projection／grouped backend改为本逻辑program的局部owner，
并加等步长codec切换回归。保持原physical bytes、CUDA activation codec、点值门和原完整
consumer不变。预声明定向：该回归／IQ3_S与Q3_K原失败consumer、fixed物理编码组两入口，
既有reference直接消费者按selector；不重跑无关P1或全路线。原失败与反例封存，纠正尚未验证
前不签根因解决；本包／交付总体仍AUDITED。

纠正验收已完成：原thread_local owner回归FAIL；局部owner的回归／原IQ3_S和Q3_K
consumer3项PASS，独立static SDK的Direct／Graph fixed组各560项PASS，无skip。
原整批失败现分类为错误reference的误报，原日志保留，不把旧650项日志改成PASS。
源码指纹、同址capture反例、初版短序列PASS、旧owner FAIL、新结果和哈希见
[根因与资格](../../../../evidence/correctness/cuda-u3-reference-codec-lifetime/README.md)。
只改test reference寿命及其回归，无产品／原数学／误差门变更；未重跑其它651/649原生全集或P1。
后续SDK current组因用户要求优先排查而中断，已有完成组保留；该中断不记产品FAIL，也不记PASS。

### 最终交付续跑的有限范围

U1审计已结束，八族49个family-region关联现在与封存的两平台native用例逐项对应；
Qwen named-Q4 MTP的CPU／CUDA用例名不同，但都直接调用`cpu_c4_qwen_mtp::check`，
不是新程序。完整mapping随交付证据封存，不用测试总数替代该对应关系。

两个普通shape/provider admission用例首次独立SDK编译时无条件包含Linux专属FlashInfer头；
Windows关闭该Provider的SDK正确地没有该头。test-only纠正复用已有
`d5_optional_cuda_provider::is_flashinfer`，不把未选Provider补装进SDK。
只重编该一个TU，source／relocated static SDK各2项通过，未重跑大executable。

SDK current续跑保留原21个完整用例，每个完整program独立进程，避免多个不相关模型shape
的原库private allocator缓存叠加。已完成编码／artifact组复用；families与普通入口／生命周期
按未完成项补验。此分组不删除程序内部stage、权重精度、真实内宽、state或完整词表。

Windows CUDA-only static SDK联合资格现已完成：完整native需求651／649的用例并集逐名无缺项，
加普通入口／负空间／reference回归实际不同用例667／652项通过；49关联完成逐项映射。
漏带的五个本地Qwen Graph用例已用独立目标补齐，没有重跑通过的编码组。scope、原失败、
复用与source／offline／relocation／DLL边界见
[静态联合资格](../../../../evidence/correctness/cuda-final-u3-windows-static/README.md)。
shared／mixed／Linux shipping矩阵及U4／U5仍待，本卡继续AUDITED，不以此子格签全包完成。

## 七问与冻结

第三方依旧原ggml／ATen源、pin和license；当前模型完整需求不扩画像；backend、Graph adapter、
Mem/Completion复用；P0记F32转换／index同步／私有分配账单，单成熟补洞不跑P1；无FFI；
三交付以既有SDK/owning package规则联合验收；用户仍只选CUDA与自然Direct/Graph入口。
CPU已闭数学/性能、通用SIMD、其它GPU与public schema不改，自研CUDA继续冻结。

## 当前官方链路第一批结果

产品修复`3922e3ed`（test-only继承至`3aff1ac2`）：两当前official profile的target/confidence、
Markov完整词表、prefill/五tokencache、ratio0/4/128主干attention/MoE、完整三stage/shared
embedding/confidence/head在Windows/Linux Direct/Graph通过。Linux同候选10项native无skip，
Windows数学／输入身份匹配证据复用及最终cache consumer验证通过。范围和原失败见
[当前资格](../../../../evidence/correctness/cuda-final-u3-r1-current/README.md)。

F3具体Source误差账单：prefill引用actual cache后的projection，source dot对actual输入等于GPU输出；
相消使已admitted BF16输入误差的点值relative比值放大。GPU consumer沿common validator机制
对actual public inverse验证独立grouped方程/原BF16边界，其它完整gate保持，不改global tolerance。
F4及F5、全49关联与其它族尚未完成；本包仍AUDITED，不代签全R1或CUDA CLOSED。

## F4 三sidecar的GPU reference接缝

当前common sidecar program保持原数学／profile／head/state；只增加test-only量化projection
reference替换点。CPU默认分支逐行原样保留；CUDA fixture用原ggml CUDA backend直接MatMul，
相同packed bytes／F32 accumulation／原GPU A8路径，拒绝CPU backend。不能把CPU AVX2／
Q8_K／vec-dot程序当成CUDA Q8_1参考。此差分证明单成熟carrier的adapter接线；原codec/方程
独立资格仍由G1/G2与逐物理格式测试承担，不宣称新数学／性能peer／SOTA。
差分参考独立拥有原CUDA backend／buffer／graph，普通HYLE入口仍完整运行三sidecar和原cache/head。

具名F4准入修正：Qwen sidecar的Q3_K embedding有248320×5120逻辑元素，压缩bytes/strides仍
小于INT32；原GET_ROWS按选中row解码、row offset用size_t、内宽/int64索引，不创建全float表。
现representable统一按全F32转换预留的INT32/4逻辑上限误拒绝。只对原native quant Gather source
将logical count门调整至INT32，保留physical bytes/strides、输出、checked IDs与其它op旧上限。
不扩大其它route/格式，不通过裁小词表或copy冒充覆盖。

F4后续具名SDPA形态：Qwen MTP明确请求non-causal／无mask的BSHD cache attention。当前CUDA
route只允许causal并固定写diag mask，原ggml程序已有null-mask flash及无diag的F32路径。
仅按现operation.causal选择原mask/无mask，保留dtype/scale/GQA/index/score容量资格，不改公共
schema或写attention kernel；case仍用原query/key/cache，不篡改consumer的causal标记。

## F4 原生临时池资源接入审计

候选`22069f25`的Windows Direct/Graph均报告driver code 2，operation为
`cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0)`；Linux同候选前驱的
live backtrace停在同一原生调用。原ggml `456172ec`每个backend/stream的VMM池预留32 GiB
虚拟地址，而Direct多个prepared operation与Graph片段分别保留独立backend。完整Flash
三stage是实际消费者，不能通过缩小shape、删stage或重新排序测试签资格。

本接缝有限FIX：冷建invocation时选择原ggml `ggml_cuda_pool_leg`（原cudaMalloc缓存池），
由原backend独占各stream的池；不修改kernel/codec/融合、公共Queue、slot独立性或模型数学。
只在private native resource bridge中绑定现成原池；不全局关闭CUDA VMM、不要求用户加开关。
原同址reuse、graph warmup/capture与析构保持原程序。资源成本从每stream固定32 GiB地址预留
改为实际临时容量，原cached payload/私有分配仍为managed未知上界，不冒称无分配prepared。
Windows的首错和Linux栈已保留；历史失败不签PASS。新的适用验证为三sidecar两入口、原两slot/
close/失败/replay/capture负空间与完整region。先验证单一不可变候选；资源接入不启动P1。

该F4资源事实及原池源码已AUDITED，owner仍为当前`compute_provider_ggml_cuda`；native bridge
编入已AUDITED的`compute_ggml_runtime`原源，但不修改其composer、pin、CPU执行或公共治理。

同候选`da67204e`的当前五official程序与三sidecar已在Windows/Linux普通两入口通过，
原Provider资源、编码、两slot、重放、noncausal源方程以及Linux capture失败注入通过。
精确范围与保留原失败见[sidecar资格](../../../../evidence/correctness/cuda-final-u3-r1-sidecars/README.md)。
全固定GGUF与其余family-region仍需完成；`93792dd4`批量注册共同C5原合同/参数/完整NextN，
GPU codec reference接缝仅为test，当前尚未编译/运行，不算通过或本包完成。三交付联合项仍待。

### 固定C5批次的两项准入／布局纠正（AUDITED）

`0ba0586e`已实际编译；104制品身份的projection及原Qwen参数形状在Windows两入口通过。
完整559数值合同批次检出两类有限FIX，旧失败保留：

1. Qwen2.4T NextN的Q8_0 embedding是248320×8192，压缩source bytes约2.16 GiB，选中row
   decoder使用size_t byte offset／int64 width；现source门仍用INT32 bytes而误拒绝。
   只把已admitted quantized Gather source的physical byte上限改为UINT32，其logical/stride/
   checked IDs/输出/其它op限制不变，不裁词表。原CUDA getrows.cu的模板参数与row地址证明此域。
2. 七个plain BF16 expert合同的同token重复ID产生第三slot零输出。原ggml F32 MUL_MAT_ID
   fallback的expert/token/slot循环命中后`break`，假设token内专家不重复。HYLE GroupedMatMul
   合同允许重复ID。对plain权重采用原reshape，把[T,S,K]当[T*S,1,K]、IDs当[T*S,1]，原
   MUL_MAT_ID后reshape回[T,S,N]；每原token/slot成为独立native token，保留原F32 operands/
   checked IDs/物理权重和输出位置，避免复制selected权重或更改原kernel。原量化路径不变。

两项只在当前AUDITED ggml CUDA owner的既有lowering内修复；不是新数学或扩大公共schema。
固定批次完成后一次集中纠正验证，并补同址/重复ID的小型source方程回归；同候选再资格化当前
official/sidecar资源。其余族仍以完整需求映射为准，未运行不能PASS，不启动P1。

纠正后`374bf465`已在Windows/Linux Direct/Graph各通过581项：全559数值合同（433 packed／
126 plain）、104制品身份、当前参数／完整NextN和本候选official／三sidecar；重复expert/F32
独立方程回归亦通过。原失败已无损归档，见[固定GGUF资格](../../../../evidence/correctness/cuda-final-u3-fixed-gguf/README.md)。
其余family-region／本地真实权重与最终联合三交付仍未完成，本包保持AUDITED；不要外推为
U3或整个CUDA收口，U4/U5未因correctness通过自动完成。

F3后续reference接缝：其余family批次的Kimi MLA仍用common fixture的默认
`round_grouped_input_to_bfloat=true`，与本CUDA plain grouped F32输入合同不一致。
fixture已有false独立源方程（原Metal consumer已使用），CUDA两调用应显式传false；只改test
caller，不改CPU默认／产品数学、whole-fragment误差门或相应state/cache验证。初版失败保留，
本caller纠正未运行前不能PASS；不把这种不同乘数边界的参考差异记成性能胜负。

`7a9531ce`的必要caller修正已编译并运行，两入口仍FAIL；首差仍在replay1的attention port3。
该F32来源事实不是完整根因结论，不据此外推修正PASS。其余family批次Windows Direct36中35、
Graph34中33通过，无skip（包括本地原版程序与完整recurrent）；Kimi MLA三profile整项保留
未资格化，下一步仅对actual query/cache/state与独立源方程进行有限因果对账。原结果保存在
本轮`.scratch/validation/cuda-final-u3-r1/windows-family-programs`及`windows-family-correction`，
之后随本子批归档；不盲测、不改whole-fragment门，不把已sealed559合同批次撤回为未运行。

实际源码反例已定位到原 F32 MMF 的 TF32 指令与未消费精度要求的 dispatch；定向 runtime
接缝单独按[原 F32 分发卡](cuda-final-u3-f32-dispatch.md)审计。前述仅 reference／cache 的推断
不足以解释全部误差，原FAIL保持，禁止以近零点值／整体误差较小直接撤门签PASS。

原F32精度分发实际修复后，临时reference包装移除，Kimi原完整独立轨迹通过；两平台651/649
联合请求全集及原资源门已完成[native资格](../../../../evidence/correctness/cuda-final-u3-f32-native/README.md)。
Linux单进程资源饱和timeout没有改写为PASS；新分组case全集逐名核对后通过。当前native
成熟接线批次已取得资格；最终joint三交付/route facts负空间与U4/U5仍需完成，不自动CLOSED。
