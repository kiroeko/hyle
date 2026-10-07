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
