# 单机高性能推理基础能力审计

状态：RESEARCH_COMPLETE，2026-10-13。用户要求彻底扫描单机单GPU、单机CPU与一个GPU联合计算、
单机CPU三个部署形态，单/多请求均覆盖；GPU分别按CUDA/Apple/Vulkan原实际资格。
baseline67192441c70e8e07b1cae1b09d221114b738c9ca，64 module原有限范围CLOSED，产品工作树干净。
承接[原缺口研究](inference-foundation-gap-research-2026-10.md)与
[A/F4/F5终局](foundation-f45-completion-audit.md)，本轮只读审计，不从历史待办授权产品施工。

范围以完整旅程展开：加载/转换/放置、prefill、驻留状态增量更新、单请求decode、变长多请求、
CPU/GPU partition与流水、MoE权重选择/搬运、分支checkpoint、logits/采样/流输出、内存压力、
取消/错误/关闭以及普通C++三交付。分布式/多GPU/多socket、多模态及完整serving产品政策不纳入
必做；模型/格式/请求策略与通用机制分开。成熟dense/GQA、MoE、hybrid/recurrent、量化权重/KV、
长上下文/短decode/混合prefill作为不同消费形状，是否准入由实际源码/现有HYLE合同裁决。

固定本次远端精确源码：SGLang7224fce1b5e488c9916f20ec0556d696e595d512、
vLLM98f721116227f0f48690deb05c815c057bac0668、
llama.cpp cbe28bb3cb0758f0c566a9a3561766821aeb0748。只fetch对象并git archive到scratch，
三个既有checkout的HEAD/分支/修改不移动；source-manifest保存原身份/归档hash。

采用双向扫描：peer实际owner/调用链→机制及成本，HYLE全部相关公开头/typed ops/route/
compile/资源合同→逐族回映。每个缺口列正常对象序列、ownership/失败/退休、当前公开路线、
缺口/适配/资格边界、CPU和各GPU/hybrid域、可能owner、成熟借用/本库机制/应用策略、优先级与
具名下一验收。新增能力不从benchmark行、仓库LOC或一个API名称推出。

有限调查为一次全公开面索引、18个旅程族的三部署/七实际route能力矩阵、三peer调用链以及一遍
疑点反证。每族有可核源码出处和裁决后停止，不把未来所有feature变成当前MUST。
不构建全库、不跑P1/模型server、不改产品/API/CMake/registry/锁/公共工具，不自动reopen。
结果为有出处的功能/消费适配/高性能资格/engine策略清单和有限后续工作包；不以源码扫描签
整模型正确性、性能达标或绝对没有其他缺口。只执行selector的docs/closure与diff检查。

## 当前结论

HYLE已具备资源寿命、范围搬运、文件管线、有限图与状态版本的主要机制；不足主要集中在
**高效驻留KV消费、更新到读取的设备内组合、F5与联合图的接缝、量化KV和低成本输出选择**。
有些数学/编码已经能算，但实现形状会产生额外展开、等待或受限域；这些是高性能目标的
消费接缝或资格缺口，不能把已有API数量相加成“引擎基础已完善”。

本轮没有发现需要立即改Core/Mem/Sched/IO基础合同的具体缺陷；新要求下的机制候选与
性能疑点列在下文。原CLOSED仅证明其原范围，不等于以下新旅程已通过，也不因本审计
自动撤销旧资格或授权实现。数值正确/实际速度仍须具体模型与机器的消费者验证。

## 来源与范围

源码来自上面三个精确远端对象；HYLE输入是baseline与未变化的产品源码。Compute/Graph
97个头建立了索引，其中68个公开头、29个detail头；再沿相关实现、Platform/Mem/Sched/IO/
SIMD/RNG与当前user/terminal合同反证。哈希、API导航和所引用源码清单保存于
[输入证据](../../../../evidence/architecture/inference-single-machine-foundation-audit/README.md)。
索引是覆盖导航，不能代替语义、数学或性能证明；原模型片段/mock service不是完整engine。

关键peer机制不是同名API求并集：

| 旅程 | 实际源码与动作 | 对基础库的要求 |
|---|---|---|
| 驻留KV与有效长度 | [llama get_k/get_v/cpy_k](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-kv-cache.cpp#L1294)：按当前n_kv创建strided views，set_rows写新增位置；[vLLM CPU attention](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/attention/backends/cpu_attn.py#L419)直接接cache/block_table/seq_lens及slot_mapping写入 | 不要求各route同一种paged对象；要求数学直接消费驻留数据、只更新实际范围、有效区正确 |
| CUDA批规划与更新 | [SGLang host-known fast plan](https://github.com/sgl-project/sglang/blob/7224fce1b5e488c9916f20ec0556d696e595d512/python/sglang/srt/layers/attention/flashinfer_backend.py#L215)避免D2H并更新预分配buffer；[vLLM FlashInfer](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/attention/backends/flashinfer.py#L2623)按slot_mapping追加KV，分prefill/decode与capture资格 | metadata/更新/数学应有可组合原生依赖；私有_cached_module不是可直接照搬的稳定HYLE API |
| CPU与GPU数据流水 | [vLLM prefetch](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/model_executor/offloader/prefetch.py#L283)每层event而非总stream等待；[ggml async copy](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/ggml/src/ggml-backend.cpp#L651)也会在无原生能力时同步fallback | 原生能力、实际等待与跨阶段寿命必须报告，不把名称async或pending数量签成overlap |
| expert驻留与选择 | [llama MoE cache](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-moe-cache.cpp#L72)合并相邻miss上传，slot映射与LRU归engine；vLLM prefetch用固定buffer/event，SGLang offloader保存原资源 | 借F1/F2/weights slots表达实际搬运，缓存命中/预测/LRU不下沉；支持编码与选中expert消费才形成完整路线 |
| 输出选择与采样 | [vLLM sampler](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/sample/sampler.py#L242)区分greedy与随机；[top-k/top-p](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/sample/ops/topk_topp_sampler.py#L280)保per-request生成器域与fallback；[SGLang sampler](https://github.com/sgl-project/sglang/blob/7224fce1b5e488c9916f20ec0556d696e595d512/python/sglang/srt/layers/sampler.py#L187)greedy直接argmax | 普通数值选择/随机原语与领域采样规则分开；不强制全vocab每步D2H，不许诺所有seed跨route同token |
| 状态与错误关闭 | [llama recurrent](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-memory-recurrent.cpp#L189)不能恢复未保存的过去；[SGLang Mamba](https://github.com/sgl-project/sglang/blob/7224fce1b5e488c9916f20ec0556d696e595d512/python/sglang/srt/mem_cache/unified_cache/components/mamba.py#L576)checkpoint有flush前提；[llama析构](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-context.cpp#L574)在释放输出buffer前等待async copies；[vLLM scheduler](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/core/sched/scheduler.py#L183)仅在多inflight且KV connector consumer域延迟free | F1/F4已提供机制，engine完整列状态与业务引用；请求结束不等于native退休，未保存过去不能回滚；connector例子不扩成本轮分布式要求 |

## 三部署和七实际路线的能力矩阵

GPU主计算仍由CPU提交/IO；这里GPU-only指没有CPU数学fallback。联合路线明确CPU数学节点，
不把“GPU offload使用host内存”称CPU/GPU联合计算。CPU-only允许一颗CPU上的多个core；
并行度、NUMA、ISA与native私有线程是单独合同。

下表“可”指现有公开路径和已签有限域；“桥”指缺自然组合接缝；“热”指当前额外结构费用或
高性能资格不足；“域”指所需形状/编码/语义仍需准入；“策”指engine策略。“可”不是本轮新PASS。
联合列分别沿其真实GPU资格继承，不能用CPU回退填满。

| 旅程族 | CPU | CUDA | Apple | Vulkan | CPU与CUDA | CPU与Apple | CPU与Vulkan |
|---|---|---|---|---|---|---|---|
| 01 文件加载与有界stage | 可F3 | 可F3 | 可F3 | 可F3 | 可 | 可 | 可 |
| 02 权重编码与冷转换 | 可/域 | 可/域/热 | 可/域/热 | 可/域/热 | 可/域 | 可/域 | 可/域 |
| 03 dense prefill与基础数学 | 可/域 | 可/域 | 可/域 | 可/域/热 | 可/域 | 可/域 | 可/域 |
| 04 高效驻留KV读取 | dense可，F5热 | Linux FI可，Win域 | dense可，F5热 | dense/F5热 | 桥/热 | 桥/热 | 桥/热 |
| 05 增量写入到attention组合 | F1/F2可，F5桥 | F1/F2可，F5桥 | 桥/热 | 桥/热 | 桥 | 桥 | 桥 |
| 06 量化KV真实存储消费 | 域 | 域 | 域 | 域 | 域 | 域 | 域 |
| 07 window/sink/MLA等attention域 | 部分可/域 | 部分可/域 | 部分可/域 | 部分可/域 | 域 | 域 | 域 |
| 08 active metadata与规划复用 | 可/热 | 可/热 | 可/热 | 可/热 | 桥/热 | 桥/热 | 桥/热 |
| 09 有限shape与replay/capture | 可 | 可/域 | 可/域 | 可/域 | 可/域 | 可/域 | 可/域 |
| 10 placement与copy/compute流水 | 策 | 可 | 可 | 可 | 可/热 | 可/热 | 可/热 |
| 11 MoE grouped与专家驻留 | 可/域 | 可/域/热 | 可/域/热 | 可/域 | 可/桥 | 可/桥 | 可/桥 |
| 12 recurrent/conv与完整状态 | 可/域 | 可/域 | 可/域 | 可/域 | 可/域 | 可/域 | 可/域 |
| 13 logits选择与随机数值 | CPU可 | 桥/热 | 桥/热 | 桥/热 | 可/桥 | 可/桥 | 可/桥 |
| 14 CPU执行预算与locality | 可/桥 | host策 | host策 | host策 | 可/桥 | 可/桥 | 可/桥 |
| 15 requested与私有资源账 | 可/未知 | 可/未知 | 可/未知 | 可/未知 | 可/未知 | 可/未知 | 可/未知 |
| 16 concurrency/cancel/实际退休 | 可 | 可 | 可 | 可 | 可 | 可 | 可 |
| 17 prefix/branch/checkpoint | F1/F4可/热/策 | 可/热/策 | 可/热/策 | 可/热/策 | 可/热/策 | 可/热/策 | 可/热/策 |
| 18 public交付与真实模型闭环 | SDK可，模型待验 | SDK可，模型待验 | SDK可，模型待验 | SDK可，模型待验 | SDK可，模型待验 | SDK可，模型待验 | SDK可，模型待验 |

族01/16主要复用F1–F3与原native退休；族02/11对应G07，族03/04对应G01，族05对应G02/G03，
族06对应G04，族07对应G05，族08/09对应G06及原A合同，族10对应G09，族12/17复用F4且保
原成本边界，族13对应G08，族14对应G10，族15保原requested/private UNKNOWN边界，族18
复用已有SDK交付且要求未来真实模型闭环。不是每族都需要新增基础库功能。

## 缺口和已有实现的适配差异

### 驻留attention和参考档成本

**G01 高性能消费接缝，优先级1。** F5 CPU/Apple/Vulkan的共同recipe并非只gather一次KV：
[batch_materialize](../../../../src/compute_batch/batch_materialize.cc#L59)按物理Q容量N、query heads H、
每request最大KV长度L和D生成indices4NHL、mask4NL，K/V gathered bytes=4NHLD，随后cast/
MatMul/softmax/Copy。它复制/重复实际KV，且依物理capacity而非active量计费。
例如N32/H32/L4096/D128，仅K/V gathered即2 GiB；这是结构计算，不是峰值resident实测。
长prefill的Q维也参与复制，不能作为高性能默认。原F5明确把它作为显式reference，原资格有效。

反证已找到：CPU [ggml SDPA](../../../../src/compute_provider_ggml/ggml_cpu_provider.cc#L834)在部分F32/
F16 causal域调用原flash_attn_ext；Apple [SDPA](../../../../src/compute_route_apple/apple_tensor.cc#L317)
使用MLX fast原生接口，不能说这些路线缺所有attention。Vulkan
[ggml region](../../../../src/compute_provider_ggml_vulkan/region.cc#L857)的该SDPA分支显式构造F32 scores/
softmax/MatMul；其他IREE域另核，不能由一个分支断言整个route无优化。

最小处理：为驻留cache、真实seq_len/strides或page表接成熟native消费；单请求优先验证现有
dense/view是否足够，多请求再选择成熟paged/varlen/等价region。CPU可核ggml/vLLM原CPU
机制，Apple核MLX/ggml Metal，Vulkan核ggml/IREE。owner是各route/Provider及batch桥；
不写新通用Mem分页或默认CPUfallback。验收为同数值/有效区，记录额外copy和scratch，
避免按Q行/头重复全KV，独立低延迟decode及长prefill再做定向P1/P2。

### F5和联合图之间缺少组合接缝

**G02 API_GAP，优先级1。** CPU/GPU联合与F5单路线均已存在，但公开形状不能直接相加。
[graph batch](../../../../include/hyle/graph/batch.h#L85) create保存一个Queue，
[submit](../../../../src/graph_batch/batch.cc#L201)仅调用original.submit(queue)；
[原LeasedInvocation](../../../../src/graph/leased_invocation.cc#L436)明确required_queue_count不为1时
拒绝，虽其底层已有span<QueueBinding>档。CPU+CUDA本来就使用两个lane/完整bindings，
其wrapper也没有PagedArenaRead bind。CPU+Apple/Vulkan的
[bind](../../../../include/hyle/graph/cpu_apple.h#L166)同样只有ready lease/native tensor/weights，
没有F5精确page frame。Page frame不公开whole-arena view，这是正确权限边界，不能靠raw
pointer或src/private桥绕过。ragged/dense图可用已有操作组合，不能因此签paged联合完整。

最小处理：F5 coordinator沿原multi-queue replay接口借用完整bindings；native hybrid在原
owner中接准确page-footprint/metadata generation到原slot。保同slot、旧结果、错误suffix与
actual retirement，不建另一个scheduler/Queue。owner为graph_batch和三hybrid/原native桥。
验收为公开CPU节点→GPU实际paged attention→CPU消费者、两个活跃Queue、精确页权限和
失败/取消/旧result，三交付不引用内部类型。原拆成独立图方案可作为可用路径，但重复寿命
胶水与额外wait必须计费，不能当最佳消费形状已经具备。

### 增量更新的写权限和原生依赖

**G03 COMPOSITE_GAP候选，优先级1，需最小consumer反证。** 原F1不缺独占路径：
[LeasedInput](../../../../include/hyle/compute/leased_operation.h#L12)能读WriteLease，F2能bind_producer并
建立真实native wait。缺的是这些原子到F5的自然组合：F2
[output](../../../../src/compute/batch_transfer.cc#L506)须实际退休才产生ReadLease，PagedArena.read/
F5输入只接ready frame，F5准备还有host wait；没有公开pending-page/producer→batch完整档。
不能把读未完成writer假装ready。模型QKV/slot_mapping/编码变换/append→attention是否能全留
在设备上、少host边界，必须整体检查，而不是只演示一个copy。

Apple/Vulkan native Invocation还会保原input views到下一bind或销毁；controller prepare清
自己的reads不等于释放原native imports。若需原页独占改写，这种保活与旧reader/F4 sharing
会限制复用。原保护正确，不能为了性能提前drop仍在途owner。单路线小页copy/双buffer
可以正确运行；是否需要新复合helper取决于长期decode的实际copy/分配/等待账。

最小处理：保F1 exclusive/native use，提供已接纳producer与精确范围的受控更新依赖、明确
成功转换到reader以及idle input清理/重绑定旅程；借成熟set_rows/reshape_and_cache或F2适合
的范围copy，不默认克隆全历史。GPU effect只在真实mutable carrier/footprint可证明时准入。
owner为compute_batch、F1/F2 seam及原route/graph bridge；不造任意GPU mutation或模型事务。

### 量化KV和更广attention语义

**G04 MODEL_DOMAIN_GAP，优先级1或2，由目标内存/模型决定。**
[BatchCapacity](../../../../include/hyle/compute/batch_layout.h#L18)只有同一种F16/BF16数据域和一个D，
没有KV codes/scales、Q与KV不同编码、native cache layout端口。
[BlockQuantizationRoundtrip](../../../../include/hyle/compute/model_ops.h#L108)返回dense BF16，不代表
已存储量化KV或让attention直接消费压缩cache。llama按原cache type建view/set_rows；vLLM
CPU及CUDA update均显式cache dtype/scales，CUDA还有NVFP4路径。格式身份可以由TensorDesc
表达，不证明整个“量化追加→驻留保存→attention”有合格承载。

最小处理：按真实选定cache编码分离payload/scales/layout/范围及数学资格，借成熟
quantize-append和消费kernel；避免每步解码全部cache。不能为总显存目标补一个假的预算开关。
owner是compute_ops/batch及Provider；engine仍选择KV精度/误差容忍和压缩策略。

**G05 MODEL_DOMAIN_GAP，优先级2。** FI
[qualify](../../../../src/compute_provider_flashinfer/flashinfer_provider.cc#L126)只causal，D64/128/256与
GQA ratio1/2/4/8；[native batch](../../../../src/compute_provider_flashinfer/native_batch.cu#L42)硬设
window_left=-1、QK与V维度相同、无额外mask/sinks/softcap。标准RoPE可在engine用原算子
组合，不是缺口；但window首尾、非causal多token speculative、不同D和MLA/稀疏/压缩cache
不能只改页表/padding伪装同数学。已有SinkAttention/LightningIndexer是有限具名域，不是
任意paged window/MLA接口。先锁定目标模型实际所需参数，复用成熟Provider窄扩域，
不把所有peer特例一次并入batch schema。验收用位置/有效区/独立oracle及对应cache生命周期。

### CUDA每批规划和metadata形状

**G06 HOT_PATH_GAP，优先级1。** 原F5显式prepare/run合同正确，但
[prepare](../../../../src/compute_batch/batch_execution.cc#L181)每次完整快照staging/H2D，upload.wait，
新plan再wait；reuse仍有一个host wait。[metadata version](../../../../src/compute_batch/batch_layout.cc)
固定capacity blob清零，native key比较完整Q/KV indptr。尾增加可复用plan但仍复制整个M，
跨页/重排需plan；多个layer各region分别准备。接口没有已资格的active/delta上传或设备更新档。
当前使用原BatchPrefillPlan/kernel完成decode数学，不能因此证明已具备专用decode最佳路线。

最小处理：保generation/原slot失效/独立workspaces，分离host-known与device-produced
metadata，按真实改变部分或active部分更新，复用成熟decode/plan并显式报费用。稳定bucket
由engine选择；capture需原topology资格和错误/更新合同，不能只将capture_compatible改true。
SGLang的私有cached plan作为需求反证，不是可直接集成接口。验收为稳定tail、跨页、新请求/
结束、重排、同shape不同布局与failure，逐步统计upload bytes、plan和host-wait次数。

### 权重格式和selected expert的消费形状

**G07 NATIVE_ADAPTER或资格缺口，优先级2。** HYLE有GGUF block formats、QuantizedMatMul、
MLX grouped、FP8/FP4、ArtifactWeightDecode等，不缺一个总名叫“量化”的API。
但[ArtifactWeightDecode](../../../../include/hyle/compute/model_ops.h#L30)明确cold输出dense；某些
非native packed途径会扩张常驻权重或每次构造选中权重临时值。Apple
[block_dot](../../../../src/compute_route_apple/apple_mature_float.cc#L164)显式decode再matmul，MLX
原生affine量化matmul和其他ggml路径另有原承载，不能把全部量化路线都说成dense展开。
不同格式已有数学承载与高性能packed消费应分别记。

CUDA [selected ids](../../../../src/compute_provider_torch_cuda/artifact_provider.cc#L226)的
any().item<bool>()是源码明确的GPU→host检查；用于resident grouped计算时会增加同步。
不能删除bounds检查；借原安全device status/合格原生checked region，或明确host-known IDs
档，须保错误输出和suffix合同。CPU缓存策略本来需要知道IDs时，相应等待不应算多余。

最小处理：按目标artifact→实际物理TensorDesc→所选row/expert→native MatMul的路径锁定
repack次数、resident增量、temporary bytes与wait，不为所有格式写万能codec。两个consumer
重复的编码转换/owner桥归route/Provider；格式命名/模型权重角色/选择专家仍属engine。

### 输出选择和随机数值

**G08 NUMERICAL_SEAM候选，优先级2。** HYLE已有TopK/Softmax/Scan/Reduce、CPU RNG和SIMD
选择/排序，可以做greedy与CPU采样。GPU没有公共counter-addressed RNG/categorical等原
数值schema；部分TopK例如[Vulkan ggml](../../../../src/compute_provider_ggml_vulkan/region.cc#L911)
走完整argsort，即使count=1也不是单趟argmax。不存在现成的“只传回选中token/必要logprob”
随机GPU完整consumer；TopK(k)传回小集合可作为某些策略路径，但不覆盖任意top-p精确采样。

最小处理：评估通用row-wise argmax/selection/随机抽样原语与counter/state所有权，借成熟
数值实现，不把temperature/penalty/grammar/EOS/policy全部塞成HYLE Sampler。每request
seed与batch重排语义明确，保持质量/分布/错误，并允许CPU路径；跨Provider同seed同token
不是自动合同。新增API须由实际GPU consumers重复glue与copy/延迟反证，不从peer sampler名字准入。

### 联合路线的调度和CPU执行预算

**G09 HOT_PATH_QUALIFICATION_GAP，优先级2。** 原CPU+一个GPU placement、copy/native
wait、独立invocations都存在，不缺联合图总API。Apple/Vulkan
[执行](../../../../src/graph_route_cpu_apple/invocation.cc#L267)按coarse fragment循环，CPU和GPU
片段都先await再推进；独立invocations/原transfer queues可以overlap，不能说平台永不并行。
但同一个DAG中独立CPU/GPU支路、GPU连续片段减少中间host观察、microbatch流水的高性能
资格尚须具体consumer。必要CPU↔GPU边界wait不是缺陷；无依赖支路被串行化才是应反证的机制。

最小处理：先用原有限plan/independent slots与Sched做真实多microbatch、expert支路；若重复
手写native退休/阶段依赖或有material串行化，再在原hybrid owner增加有限准备好的依赖推进。
不为engine公平/continuous batching造第二runtime。UVA/zero-copy是可选实际硬件路线，
vLLM也明确它受PCIe带宽限制，不能因有UVA就把显式F2复制当功能错误。

**G10 CONTROL/OBSERVABILITY_GAP候选，优先级2。**
[CPU options](../../../../include/hyle/compute/cpu.h#L20)已有executor和ggml native_threads，Sched/
Platform已有core affinity/NUMA，不能说没有CPU并行或locality。
[ATen runtime](../../../../src/compute_provider_native/aten_cpu_provider.cc#L31)报告uses_private_threads，
但普通CPU options并未统一控制该私有池；多invocation与prefill/decode不同线程预算需整个
consumer核定。不能由native_threads=1保证全部Provider单线程，也不能靠增加CPU线程一定加速。
建议冻结可报告/可约束的executor与private-pool合同、必要phase专用contexts/共享池组合，
保进程全局setter/其他用户和oversubscription边界，不新增模型级scheduler政策。

## 已有机制不应重新实现

F1最后使用/租约退休、F2范围/producer/native waits、F3文件stage/背压/取消排空、F4显式
版本与成功发布、A有限shape/control/合格effects、ordinary source/static/shared均有实际
公开路线。新增glue必须复用它们，不能因为某个新shape不顺手就重写Queue、copy kernel、
allocator、文件管线或Graph compiler。完整Unicode、JSON、格式读取、tokenizer按既有
成熟库路由；HYLE不需要为缩engine行数加入模型格式或tokenizer产品。

大page目录若每token都走F4，[原P0](../../../../evidence/performance/foundation-f4-p0/README.md)
begin/finalize/末record O(E)及有限ledger扫描是已声明成本；F4适合显式checkpoint，不是
任意大cache的热路径目录。优先由engine按业务checkpoint频率使用；若实际trace必须每步
发布大目录，才成为compute_state的新成本候选。ResourceBudget只约束managed requested
bytes，private UNKNOWN/系统争用如实保留；admission/offload安全余量归engine，没有一个
统一显存限额可以证明所有驱动分配永不OOM。

## 如何处理适配差异

每一项先形成一条自然C++旅程，再做反证，不能从接口名字下结论：

| 差异类型 | 处理 | 消费方与库的边界 |
|---|---|---|
| 普通连接 | 使用现有public API，补recipe/example并记录必要借用/错误 | engine保自然调用，不制造薄wrapper |
| 模型或政策变化 | tensor命名、RoPE配置、EOS、batch/prefix/LRU/LoRA key和checkpoint完整性留engine | 数学、storage/lifetime原子仍借库 |
| 重复通用glue | 两独立consumer都重写owner/version/退役/metadata或跨route布局时补窄组合seam | 归真实机制owner，保原自然native类型 |
| 同名操作的域不匹配 | 按实际shape/dtype/encoding/active区/alias逐Provider确认，再借成熟承载或具名拒绝 | 不用padding、reinterpret或CPUfallback凑表 |
| 可运行但附加成本改变档位 | 列额外copy/expanded bytes、cold/warm分配、H2D/D2H、plan/host wait、private pools与capture限制 | 作为竞争性缺口，具体trace后定向P0/P1，不把每个便利成本都定为bug |

API新增顺序为consumer草图→ownership/producer/error/close→成本账→两个反偏置旅程→
有限owner/FIX→公开三交付。数学/shape固定以后才做同合同性能，source-only推断不签吞吐。
这里的两条旅程用于反证通用性和过拟合，不是“已有两个产品才允许修缺陷”的硬门槛；
一个可复现的真实consumer、标准或成熟peer也可构成准入依据。
G02是源码可定位的公开组合缺口；G01/G04/G05/G06是明确缺席域/成本与native资格；
G03/G07–G10需要最小消费者收窄到必须修正的接缝。未完成的推断不改名MUST/PASS。

每个适配项的后续卡应保存同一组记录：消费方实际shape/dtype/encoding/active区；原生对象
及public调用序列；当前适配代码及owner；cold/warm适配行数、额外copy bytes、temporary/
resident bytes、分配、plan和host waits；producer/别名/版本/错误/退休合同；最终裁决与
最小验收。共同封装可以减少重复行数，但内部仍展开或等待时不能算性能缺口消失。
验收同时保留“不需要新增API”的反例，以当前自然调用能否满足目标为准。

## 有限后续工作包和验收

1. **先联合分页和驻留读取**：G02与G01。同一个公开单CPU+单GPU图跑CPU节点、驻留
   attention、CPU输出；CPU-only与GPU-only用同数学/有效区，保存所有额外KV拷贝与权限。
2. **再增量写和低等待decode**：G03/G06。长上下文单request和两种长度request交错，
   QKV producer→append→attention→token，跨页/重排/结束/取消，证明只更新实际dirty域。
3. **按目标模型补编码和语义**：G04/G05/G07。选定quant KV/weights、window或特殊attention，
   原编码到原生kernel有独立数值参照和成本；不实现所有格式/所有模型。
4. **再降低输出和联合流水费用**：G08/G09/G10。设备输出选择、CPU private-pool预算、
   多microbatch CPU/GPU支路，有正确性/失败/旧结果/退役再按同合同P1/P2量TTFT/ITL/吞吐。

每包绑定实际owner与最小实现需求，估engine重复glue的变化，不用本次基础库新增LOC直接
相减。真正复用完整机制时engine可以减少平台胶水；模型/策略行仍存在。当前14–30k
engine条件估计不是“只要补10项就一定下降到某数”，需同一prototype基线测量。

验收底座是一个dense/GQA、一个MoE或recurrent留出consumer的加载→prefill→多步decode→
输出/取消/关闭；三部署形态保持同模型/精度/位置/有效区，记录适配行及每层copy/wait/
plan/allocation/native use。这构成高性能需求的证据入口，不在本只读审计中启动完整engine。
