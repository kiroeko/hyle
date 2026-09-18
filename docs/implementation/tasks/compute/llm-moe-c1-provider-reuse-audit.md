# C1：成熟 Provider 承载复核

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
| CPU | [oneDNN](https://github.com/uxlfoundation/oneDNN)、ggml-CPU | oneDNN 已有 primitive/Graph 接入，扩展前按具体 consumer 定界；ggml 的量化与模型计算片段先解决具名 runtime 边界，不能把旧 blocker 变成永久自写理由 |
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

**SIMD 不在上述删除范围内。** 用户明确要求保留其服务非推理场景的通用能力；某个推理片段的
结果不授权删除/裁剪 SIMD 模块、公共接口或非推理消费者。这里只精简 Compute/Graph 内已经
被合格 Provider 覆盖且没有独立价值的重复算子路径，不能沿依赖向下顺手删基础设施。

当前第一项仍是 ggml-Vulkan 的真实 C1 region 接入。上述顺序不要求先把候选池中所有库都链接
才能开始验证；每个库解决具名覆盖或收益需求，重叠能力择优。尚未接入/验证的库不签 PASS，
尚未比较的 HYLE 算子不预判胜负。本次记录没有新增产品 FIX、重跑 P1 或删除产品代码。
