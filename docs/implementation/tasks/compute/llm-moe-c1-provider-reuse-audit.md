# C1：成熟 Provider 承载复核

## 当前接入覆盖与后续施工纠正（2026-09-18）

owner `/root`，当前核对 baseline `6695fba0`。用户指出“Provider 接入后应有对应普通路由”，并
要求修正偏向逐项自写的任务卡；同时明确已有缺陷、为性能省略必要检查的问题也必须处理。
[C1 活动卡](llm-moe-c1-tensor-state.md)已据此重排施工顺序，下文较早来源调查仍是证据，
其中“下一项”等历史安排不覆盖该当前裁决。
2026-09-19按`aa6690c6`和当前施工卡更新下面CPU/Vulkan的接入事实；性能未资格化的域仍不写赢家。

| 路线 | 当前普通接入的具体事实 | 下一步的接入责任 |
|---|---|---|
| CPU | 已有BRGEMM及[C1-AA](llm-moe-c1-onednn-normalization.md)的RMSNorm/LayerNorm与原生Add/Mul post-ops接入；其Q2测量资格BLOCKED，其他C1域仍有Native承载；ggml-CPU仍BLOCKED | 按具名恢复条件处理资格阻塞，再完成其余成熟覆盖与等责取舍；不能把注册或局部norm接入等同完整算子集 |
| CUDA | `cuda_context.cc` 注册 cuBLAS、Marlin、SGLang 与 HYLE CUDA；当前 SGLang 是 activation 源子集，C1 多项为自有 kernel/CUB 构件 | 按片段核对已登记 CUDA 成熟库的实际覆盖、硬件/版本与 C++ 交付后成组接入；库名不代签完整 norm/index/region |
| Apple | MLX typed lowerings/region 已承载较多 C1 操作，checked 状态与有序 Scatter 含 HYLE Metal 适配 | 复用当前 MLX 接线，核对未覆盖域与必要检查；比较完整调用链，不为与其他路线对称而重写 |
| Vulkan | ggml已承载归一化、具名算术/稳定变换/门控、TopK、Gather、F32 Scan和Sum/Mean/Softmax；checked/顺序语义由受审适配组合。[Y34](llm-moe-c1-vulkan-reduction.md)归约/概率/L2三交付与工程门已完成；[Y35](llm-moe-c1-vulkan-elementwise.md)逐元素/门控新绑定已实现，W1/Air/Mini适用原生模块资格通过（Mini另有1项layer测试显式跳过） | 完成Y35其余适用验证，再处理完整成本比较、条件规则及自有重复路径取舍；已有接入不等于性能优选或C1完成 |

本次有界读取固定 ggml `456172ec` 的 `ggml.c`、`ggml-vulkan.cpp::supports_op`/执行分派与对应
shader，并对照 HYLE `region.cc`、`vulkan_program.cc` 和原需求矩阵，得到下列具体区分：

| C1 需求 | 上游现成实现与当前差异 | 分类 |
|---|---|---|
| F03 算术与稳定变换 | Y35已接Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid/SiLU/Softplus、Add/Sub/Mul/Div及packed/separate SwiGLU；冷期适配三浮点、rank≤8和合法stride/广播。Rsqrt用原厂Fill/Sqrt/Div，动态Min/Max及其他未绑定激活保留原生路径 | **[Y35已实现、部分资格通过](llm-moe-c1-vulkan-elementwise.md)**；完整三交付与性能未签收，half转换、高rank广播物化和Rsqrt三个dispatch必须计费 |
| F04 归约/Softmax | Y34已接SumRows/Mean/Softmax并适配任意轴、half、stride、Boolean/additive mask、尾行与长行；Min/Max、LogSoftmax、空域及设备不适用域保留既有路径 | **[Y34数值/实际三交付已资格化](llm-moe-c1-vulkan-reduction.md)**；selected112工程门已通过；完整成本和存量取舍未签收，不能从成熟域外推全部域均由ggml承载 |
| F04 LayerNorm | `norm.comp` 用 E[x²]−E[x]²；Y33已以原版 `group_norm.comp` 的两遍中心化算法接入，适配dtype/layout/设备与输出owner | **[Y33数值与交付已资格化](llm-moe-c1-vulkan-layernorm.md)**（2026-09-19）；大均值小方差通过原门，W1/Air实际三交付通过，完整成本性能取舍仍待C1固定bundle |
| F04 TopK | 已用原版Argsort/GetRows承载值+索引，补共同stable/NaN与对齐语义；冷Plan/Graph已支持该多输出域 | **数值与三交付已资格化**，详见C1-Y；原版全排序成本保留，完整性能取舍待裁决 |
| F02 Gather/Scatter | 有 GetRows/SetRows/GetRowsBack；类型/轴和输入要求不同，普通 HYLE 默认 checked，失败不得越界或破坏输出 | **索引/状态/失败适配**；上游有算子不能替普通入口免除检查，也不能把这些接口差异等同缺整个算子 |
| F02 Scan | 有 F32 inclusive Cumsum 及设备 subgroup 条件；本包 counts/offsets 还要求 I32 exclusive、checked overflow | **具名数值档位缺口**；未核对另一成熟候选或受审上游适配前不裁决自写完整 Scan |
| F01 View/Copy/Cast/Concat | 上游有 View/Repeat/Copy/Concat 等；HYLE 已有原生 View owner、byte-stride 与不同 dtype/空域 | **按域复用/映射**；零复制 View 的所有权仍由 HYLE 管理，额外 materialize 和舍入必须计费/验证 |

该表没有把“待绑定”写成 PASS，也不增加第二份覆盖 registry。下一批产品 FIX 必须依据这些
具体差异回到实际 owner 的卡固定；此前候选只达到局部验证的资源/状态事实可复用。
后续默认规则仅接受通过本包数值、checked、失败/资源与三交付资格的候选；省检查得到的性能
结果不参与优选。任何已有缺陷按反例/影响域定向修复，有限预算与 CLOSED/BLOCKED 协议照常。

### CPU oneDNN：已存在的能力与接入责任

以下保留接入前的源码审计输入；归一化/post-ops后续实现及当前Q2状态以C1-AA和registry为准。
本次补充只读取产品锁定的 oneDNN `3.13.2 / 71094df451909e95ceeefc8f64c275cdcbbcdff5`，
没有用本地 peer 工作树的更新 HEAD 替换依赖。`HyleComputeDependencies.cmake` 当时设置
`BUILD_GRAPH=OFF`、`CPU_RUNTIME=SEQ`、`ENABLE_PRIMITIVE=MATMUL`；`graph_provider_onednn`
当前是 HYLE 组织的 BRGEMM gated MLP。因此下列上游能力多数尚未通过 HYLE 普通入口接通。

| 已登记 C1 需求 | 锁定源的成熟实现 | 接入时必须核对 |
|---|---|---|
| F01 转换/布局 | primitive reorder，以及 Graph Reorder/TypeCast | 共同 dtype、RNE、byte stride/起点、空域与额外 materialize 成本；上游声明不代签全部 HYLE layout |
| F03 算术/激活 | `dnnl_types.h` 的 eltwise/binary，包括 soft_relu、logistic、swish；Graph 支持相应 op 和 post-op partition | 特殊值、广播、半精度中间值/最终舍入、alias；不只比较理想有限值输入 |
| F04 RMSNorm/LayerNorm | `dnnl_rms_norm` flag、`dnnl_layer_normalization_forward_primitive_desc_create[_v2]`；Graph 的 RMSNorm/LayerNorm pattern | epsilon、scale/bias 类型与稳定方差；Graph RMSNorm pattern 目前是 rank2–5、最后归一化轴，rank1 等先查无复制 shape 适配 |
| F04 归约/概率 | reduction 与 softmax/logsoftmax primitives；Graph 有 ReduceSum/Mean/Min/Max 等 | 任意轴、keep-dim、空集合、NaN、Boolean/additive mask 和 scale 的实际组合及成本 |

资源/执行已查到可用接口：`dnnl_graph_compiled_partition_execute_v2` 接 caller scratchpad，
并能查询 scratchpad logical tensor；`dnnl_threadpool_iface` 的同步 parallel_for/in_parallel/
concurrency 可对接现有 `ParallelExecutorRef`，不要求另造线程池。Graph host allocator 回调
没有 userdata，不能擅自假设它已提供每 context 的 HYLE allocator 归属。

原版 Graph 也有必须计费的具体成本：`src/graph/interface/partition.cpp::execute_v2` 每次建立
input/output vector；`kernels/layer_norm.cpp::execute_impl` 使用线程局部 execution-args cache
和 scratchpad shared owner。caller scratchpad 不等于完全无私有分配；C API 签名也不等于内部
C++ 分配不会抛异常。接入前需要定界 OOM 回收、两个 invocation、线程切换/重入、原生 cache
生命周期与首用/重复执行成本，再在原版 Graph partition 和 primitive/post-ops 两条具名候选
中选择。此处没有据此裁决弃用 oneDNN 或改写其算法。

CPU 默认工厂目前注册 oneDNN 后再注册 Native；扩大 Provider 的 `propose` 域就会影响普通
默认选择。后续接入必须同时冻结资格和条件规则，不能把“新分支能编译”当成已验证的赢家。
真实精度遗漏已经单独进入 [C1-Z 修复](llm-moe-c1-cpu-matmul-precision.md)，不等待性能比较才
处理，也不借此重开所有算子。以上来源核对不新增覆盖 PASS；下一产品批次仍须有限 AUDITED FIX。

## 范围与裁决

- 当前 owner `/root`，产品 baseline `6be8e6a7aa2ca898b8e6e89d4dd10679d1f40476`；本页是
  [C1 当前卡](llm-moe-c1-tensor-state.md)的定向审计输入，不是新实现授权或验收收据。
- 触发事实：C1 部分承载决定从共同单算子接口出发，把 rank、dtype、数值或原生对象差异直接
  转成 HYLE kernel 扩域；`BORROW` 还包含了参考算法和 SDK 原语，不能证明已复用成熟算子。
- 本次只核对既有 C1 产品的来源、七个需求模型与本地 2B 的 llama.cpp 入口，以及一个真实
  C1 region 的 ggml 接入边界。
  不重审四路线架构，不扩大模型或算子范围，不凭源码存在签正确性、性能或四路线覆盖。
- 裁决：**不得以“第三方没有所需算子”为由继续默认自写队列。** 所列常规与递归计算已有
  具名第三方实现。先核对完整 region 与窄源码接入；所需域的明确缺口或完整收益才授权自写。
  当前未新增产品 FIX；下一项在 C1 卡固定接入方案及适用验证后才实施。

## 产品来源的实际分类

| 路线 | 当前确实复用的计算实现 | 不能计为第三方算子接入的部分 |
|---|---|---|
| CPU | 默认工厂包含 oneDNN；已有 BRGEMM 直接调用 oneDNN API | Native 的 C1 tensor/index/norm 扩域是 HYLE 实现；复用自己的 SIMD 不等于复用外部算子 |
| CUDA | 默认工厂包含 cuBLAS、Marlin、SGLang；C1 TopK/Scatter/Scan 使用 CUDA SDK/CUB 算法组件 | C1 tensor/norm/reduction 的 HYLE kernel 即使使用 SDK math/collective，仍由 HYLE 维护其算法与寻址；不是 ggml CUDA Provider |
| Apple | MLX typed lowerings、编译 region 与原生算子；保留 MLX 数组/执行资源 | 必需的 checked 状态处理和有序 Scatter Metal 片段仍为自有实现，不能把整个路线都记为无修改 MLX |
| Vulkan | `build_shaders.py` 直接编译固定 ggml F32 MatMul/MatVec/Copy 源码；SiLU 保留上游 GLU shell 并有具名数学修改 | C1 T/V/W/X 的 `tensor.comp` 是 HYLE shader；借鉴归约/缓存/word 算法不等于接入 ggml runtime 或原版算子 |

这张表只分类来源，不给自写比例或完成百分比。既有自有路径保留当前正确性事实；默认优选和
是否保留全部适用域，还需本包等责比较。不能先删除有效能力，也不能因已经写完而排除成熟实现。

## 从实际模型计算链核对

使用已登记、本地干净源码，不更新产品依赖：llama.cpp
`4c9233c034fc450dcf34c7c0988aebe6da5cdf1a`；独立 ggml
`456172ec733a135778adcd32d00e576a58232e45`。文件身份和核对范围见
[源码记录](../../../../evidence/correctness/compute-llm-moe-c1/provider-reuse-source-audit.json)。

七个目标都已有可定位的 llama.cpp 实现路径。六个目标对应上述已登记 pin 中的四组模型实现；
GLM-5.3-Flash 对应现成支持分支，不把当前 pin 缺少其注册项当成需要自己发明算法。

| 需求模型 | converter / 模型计算图 | 当前来源资格 |
|---|---|---|
| Qwen3.8-27B | `conversion/qwen.py` 的 `Qwen3_5ForConditionalGeneration` → `src/models/qwen35.cpp` | 固定主线 pin；本地2B也用该族 |
| Qwen3.8-2.4T-A95B | 同 converter 的 `Qwen3_5MoeForCausalLM` → `qwen35moe.cpp` | 固定主线 pin；复用 DeltaNet 与 MoE 图构建 |
| Kimi-K3 | `conversion/kimi_k3.py` 的 `KimiK3ForConditionalGeneration` → `kimi-k3.cpp` | 固定主线 pin；有 KDA/MLA、AttnRes、latent MoE 与 SiTU 路径 |
| DeepSeek-V4-Pro / Flash | `conversion/deepseek.py` 的 `DeepseekV4ForCausalLM` → `deepseek4.cpp` | 固定主线 pin；按配置取压缩率/层数/专家等，不写两套算法 |
| GLM-5.3 | `conversion/glm.py` 的 `GlmMoeDsaForCausalLM` → `glm-dsa.cpp` | 固定主线 pin；具备 full/shared indexer 图路径 |
| GLM-5.3-Flash | 支持分支 `conversion/glm.py` 的 `Glm5NextForConditionalGeneration` → `glm5-next.cpp` | [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773)，本次绑定 head `de25343596a2c924ba62d35b126e811be0d38a49`；查询时尚未合并，不静默升级产品 pin |

GLM-5.3-Flash 支持分支复用 Kimi KDA、DeepSeek mHC/MoE；池化索引由已有图操作组合，所查询的
PR 文件清单没有 `ggml/` 后端改动。该事实进一步支持优先接入现成计算链。来源分支、模型结构
存在与 HYLE 产品四路线正确性/性能合格分别记录；本次不下载大模型权重，也不声称运行过它们。

| 真实消费 | 固定源码中的实际路径 | 对 HYLE 的结论 |
|---|---|---|
| Qwen3.5 语言主干 | [`qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/models/qwen35.cpp)：embedding、RMSNorm、QKV/FFN 投影、SiLU/Sigmoid/Softplus、残差、RoPE、attention 与 recurrent state | 先回映这些完整计算片段；不能从 HYLE 每一个原子的全 dtype/rank 域推导必须重写 |
| 卷积与 DeltaNet | 同文件调用 `ggml_ssm_conv` 与 `build_recurrent_attn`；[`delta-net-base.cpp`](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/models/delta-net-base.cpp)含 fused、autoregressive、chunking，融合入口返回 output 与 new state | 有完整成熟算法候选；C3 应先接入/适配，不把两输出与持久状态当成自写整核的理由 |
| DeltaNet 后端 | 同 pin 的 CPU `ops.cpp`、CUDA `ggml-cuda.cu`、Metal `ggml-metal-ops.cpp` 和 Vulkan `ggml-vulkan.cpp`均有实现/分派 | 证明有对应后端源码，不证明所有类型/shape/设备已通过 HYLE 验收；Apple 的 ggml-Metal 与 MLX 分别评估，不混为一个 runtime |
| Vulkan C1 常规算子 | `ggml-vulkan.cpp`实际分派 RMSNorm、Softmax、TopK、GetRows、SumRows、Cumsum；有 RMSNorm+Mul、RMSNorm+Mul+RoPE、SSMConv+激活等融合识别 | 下一项先查现成算子或融合片段；不把已有 shader 当成只能阅读的参考资料 |

Vulkan 源码也给出具体资格边界：普通 Softmax 接 contiguous F32 输入及可选 F32/F16 mask；
Gated DeltaNet 接六个 F32 输入，value head dimension 为16/32/64/128；TopK 要求连续布局并按
设备 pipeline 资格选择；Cumsum 有 subgroup 条件。这些是**适配/资格问题**，不能简写为缺算子。
例如模型片段内部维持合适的 F32 中间值可能避免逐节点转换，应在完整片段核对数值与驻留成本。
已有公开数值合同不静默放宽；真正不相容的域另外明确，并只为该域保留必要补缺。

## 有限的下一步接入裁决

下一轮只围绕 Vulkan 的一个真实 C1 region 处理接入，沿现有每机制最多60分钟/两条候选路径预算：

1. **完整 ggml Vulkan region**：检查普通 graph API、持久资源、失败回报和与 HYLE 的 buffer/
   completion 边界。该 pin 的公开 `ggml-vulkan.h` 有 backend/buffer 工厂，没有公开 VkDevice/
   VkBuffer 借用接口；这是具体接入工作，不是算法缺失，也未证明 region 不可用。若需要窄桥接
   补丁，先按依赖协议定界，比较一次解决接入边界与反复维护算子的成本。
2. **固定上游算子/融合源码接入现有 native program**：仅在完整 region 的具体边界不合格时
   采用，保留算法主体、许可和可核对的上游身份，逐项标明必要修改。不能把重新实现相似算法
   记为源码接入。准备时绑定，执行不搜索/计时。

`compute_provider_ggml` 当前 CPU runtime 的 BLOCKED/恢复条件仍成立；不能自动解除，也不能
外推为 ggml 的 CUDA/Metal/Vulkan 算子全部不可复用。若一个必要接入缺陷阻挡大量算子，应优先
对该依赖边界定向修复，而不是默认重写这些算子。没有新的必要性或收益证据前，不新增自写核。

上述核对不运行 P1、不改变验收矩阵的64行/256格，也不把已有 X 数值回归通过算作 Provider 优选。
七个需求模型的来源入口已回映；全部后端、量化格式、shape 与状态边界的产品验收仍按原
C2–C6 包逐项完成，不从模型可推理或源码存在直接写成四路线 PASS。

## 四路线 Provider 候选与接入优先级

本表是2026-09-18核对官方接口后的候选顺序，不把仓库检出、库名或上游能力表当成 HYLE 已接入。
已经存在的实现按当前版本继续使用；新增候选在对应 C1–C6 卡绑定版本/许可、owner 和有限 FIX，
再进入产品。没有共同真实缺口时不为增加 Provider 数量而引入依赖。

| 路线 | 优先的库/实现 | 可以承担的工作与当前接入事实 |
|---|---|---|
| CPU | [oneDNN](https://github.com/uxlfoundation/oneDNN)、ggml-CPU | 当前实际使用 oneDNN BRGEMM，并由 HYLE Graph region 组合；`ONEDNN_BUILD_GRAPH=OFF`、primitive 限 MATMUL，不能记为完整 oneDNN Graph API/算子集已接入。按真实缺口扩展成熟库承载；ggml 先解决具名 runtime 边界 |
| CUDA | 现有 cuBLASLt、Marlin、SGLang；新增优先核对 [cuDNN Graph](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer) | 已有矩阵/低比特与部分门控接入；cuDNN 提供融合 Attention 图；FlashInfer 有 Attention、MLA、MoE、norm/RoPE/门控等，逐项核对硬件和 C++ 入口，不把 Python 包安装当产品交付 |
| CUDA 定向补充 | [CUTLASS/CuTe C++](https://github.com/NVIDIA/cutlass)、ggml-CUDA；[FlashAttention](https://github.com/Dao-AILab/flash-attention) | 优先使用已实现的 GEMM/Grouped GEMM/Attention 核或融合程序。调用 CUTLASS 已有 kernel 与仅用其组件自写算法分别记录；FlashAttention 的 Windows/C++ 交付仍须验证 |
| Apple | [MLX](https://github.com/ml-explore/mlx)、ggml-Metal；必要时 [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph) | MLX 已有 C++ typed/compiled region 路径，优先扩复用范围；ggml-Metal 提供成熟模型片段；MPSGraph 是原生图框架候选，跨 runtime 资源与同步成本需具体核对 |
| Vulkan | ggml-Vulkan；[ncnn](https://github.com/Tencent/ncnn)、[MNN](https://github.com/alibaba/MNN)作定向备选 | ggml 与目标 LLM 计算链最直接，当前只接窄 shader 子集，完整 region 边界是优先事项；ncnn/MNN 有 Vulkan 后端及 C++/图计算入口，但不能由此推断其覆盖所有目标模型、BF16和状态合同 |
| CPU 定向备选 | [XNNPACK](https://github.com/google/XNNPACK)；平台 BLAS | 只在 oneDNN/ggml 的具体算子、布局或小批量域存在缺口/收益时选择；不新增通用推理引擎层 |

源码检出与接入按“真实模型片段 → 库中的实际实现 → 适用硬件/格式 → C++/原生资源接入 →
四路线普通消费者验证”推进。每次选择能覆盖一组能力的完整实现，保留融合；不能先把它拆成
大量单算子再逐个包 wrapper。SDK/CUB 仍可作为排序、扫描等局部构件，不与完整 LLM Provider
混记。FlashInfer 等上游的自动选择/调优机制不直接成为 HYLE 执行期裁决：研发固定具体后端和
条件规则，准备时只匹配绑定。专属 SM90/SM100 等的分支不能在本机 SM89 上冒签资格。

### 执行顺序与存量自有实现裁决

2026-09-18 用户确认本表作为 C1–C6 主流 Provider 清单，并明确先接入、后核对缺口、最后才
补自有算法。按每个真实片段执行下列有限流程，不新增一套施工包或覆盖 registry：

1. **来源准备**：优先使用已检出的固定 ggml/oneDNN/MLX/CUDA 实现；必要新增库绑定不可变
   revision、许可、工具链/硬件域和 C++ 入口。记录未检出、已固定来源、已接入和已验证的区别，
   不把候选清单写成产品支持表。
2. **实际接入**：按当前 owner 固定有限 FIX，优先保留完整算子、融合片段或 region，落实原生
   资源、状态、完成/失败和三交付。先走通普通 Direct/Graph 的真实消费，再评估覆盖；必要
   依赖桥接按协议定向修复，不把接口差异直接变成逐算子重写。
3. **剩余缺口**：将实际不支持的数值、shape、状态或硬件条件回填原64行矩阵/当前卡，附上
   具体调用或源码证据，区分接入未完成与第三方实现确实缺失。只有后者才进入自有补缺裁决；
   有合格成熟实现的域直接进入完整成本比较。
4. **存量取舍**：既有 HYLE 实现与合格 Provider 使用相同输入、数学/量化合同、状态与最终完成
   责任，计入转换、同步、融合损失、workspace/驻留和结果寿命。无完整收益时采用成熟实现；
   有可信源码优化机制才消耗当包预声明的有限纠正预算，不无限尝试或重测。
5. **替换后删除**：若 HYLE 不胜出，且没有可信优化机制或预算内不能反超，先迁移普通消费者并
   验证被替代的支持域，再删除重复产品代码、私有 API、构建/测试专用分支和过时说明。保留独立
   oracle/回归覆盖；必要未覆盖域或有证据的条件优势实现按具体域保留。不能仅降低其默认优先级
   却无限维护无需求的双轨。

**SIMD 基础库不在上述删除范围内，基于 SIMD 的算子 Provider 在内。** 用户明确要求保留其服务非推理场景的通用能力；某个推理片段的
结果不授权删除/裁剪 SIMD 模块、公共接口或非推理消费者。这里只精简 Compute/Graph 内已经
被合格 Provider 覆盖且没有独立价值的重复算子路径，不能沿依赖向下顺手删基础设施。

当前第一项仍是 ggml-Vulkan 的真实 C1 region 接入。上述顺序不要求先把候选池中所有库都链接
才能开始验证；每个库解决具名覆盖或收益需求，重叠能力择优。尚未接入/验证的库不签 PASS，
尚未比较的 HYLE 算子不预判胜负。本次记录没有新增产品 FIX、重跑 P1 或删除产品代码。

### Vulkan region 接入试验裁决（2026-09-18）

在产品 baseline `d23a3dbf` 完成上述第一条候选路径的有界试验，
[原始记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-region-admission.json)
包含命令/代码/日志归档、上游文件身份和未证明项。产品源码没有变化。

- 固定 ggml `456172ec` 的完整 Vulkan runtime 在 W1 RTX4070 实际完成 F32 RMSNorm → 权重缩放 →
  SiLU → 残差；宽2048、行数1/17/128，各三次变化输入，独立 double 公式核对通过。
- 原版 API 没有 native import/export；在验证副本加入窄设备/缓冲桥即可继续执行同一原版 region：
  导出 VkDevice/VkQueue/VkBuffer，借用同设备 buffer，一次输入由原生 Vulkan 命令写入；借用
  包装销毁后原 owner 的 buffer 继续有效。Khronos validation 请求开启后同样通过，日志无 VUID。
  算子算法未改；没有把简单 fill/copy 当作 region 的计算验收。
- 采用 **ggml Vulkan managed region + 受审窄源码桥** 进入产品接入定界，当前不转向自写 shader。
  架构本就允许 provider managed encoding；上游逐次遍历/融合/录制的成本要诚实计入，不能用现有
  HYLE recorded-only 实现反证成熟 region 不该接入，也不宣称原版 graph 是零分配 command replay。
- 仍须解决：可恢复 context OOM、部分初始化回滚、共享设备初始化/queue 并发、接纳部分工作后的
  失败与完成、私有分配计费，以及普通 HYLE Direct/Graph 与三交付。原 CPU `compute_provider_ggml`
  的 BLOCKED 不解除；新 Vulkan owner 独立承担这些具名接入责任。
- CMake 试验使用原厂 targets，关闭 CPU/backend tests/examples，parallel2、shader 内部单并行；
  复用已下载 SDK 的便携提取，没有执行 installer/UAC。产品接入须转为现有锁与包机制下的集中
  声明，验证副本脚本不会作为普通用户的构建入口。

本次只签接入路线可行性的局部事实，不签 HYLE 产品覆盖、完整生命周期、Apple/MoltenVK、三交付
或性能胜出。后续有限范围见 [ggml Vulkan 接入卡](llm-moe-c1-ggml-vulkan-region.md)。
