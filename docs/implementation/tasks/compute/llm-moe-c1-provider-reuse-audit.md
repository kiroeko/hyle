# 模型需求 → 成熟 Provider → 四路线接入目录

## 当前执行边界（2026-09-19）

本页已重写，替代旧 C1 按原子扩张的候选顺序。上位唯一队列为
[M0–M6](llm-moe-coverage-completion-plan.md)，当前 owner `/root`，产品基线 `ebe5af51`。
本页只记录需求与接入裁决输入；模块状态见 registry，实际产品覆盖见64行矩阵和 receipts。

**工作假设：所需计算主要已有成熟实现，预计新增 HYLE 算子为0。** 优先接入 ggml/llama.cpp
实际模型计算图及四后端，复用 MLX/oneDNN/成熟 CUDA 路径；不是先把 HYLE 自写核补齐再比速度。
全部准入第三方算子/片段与 Provider 接入完成前，禁止扩张、调研、优化或性能比较自研算子。
冻结文件身份登记不算算法研究，不进入其性能/优化分析。

## 模型能力决定算子和片段

模型配置、源码 pin、数学与格式继续复用[来源快照](llm-moe-model-sources.json)、
[C0 资格](llm-moe-c0-source-qualification.json)及[需求矩阵](llm-moe-coverage-matrix.json)。
下面按能力决定需要寻找的成熟实现；具体后端接受域必须由 M0 回填，不能预填 PASS。

| 模型输入 | 模型图要求 | 需求族 | 首选承载/接入方式 |
|---|---|---|---|
| Qwen3.5-2B / Qwen3.8-27B | embedding、norm/gate、dense FFN、GQA/RoPE/KV、短卷积与 DeltaNet 状态、具名 MTP | F01–F10/F14/F19，量化 F06/F18 | ggml 对应模型图中的单算子/融合/递归 region；MLX、oneDNN、CUDA 专用库按同域组合 |
| Qwen3.8-2.4T-A95B | 上述混合计算、routed/shared experts、top-10/dispatch/expert/combine | 再含 F12–F13 | ggml qwen35moe 的现成 MoE 组合；成熟 grouped/quant GEMM 按后端接入 |
| Kimi K3 | KDA、gated MLA、latent experts/SiTU、AttnRes、MXFP4/W4A8 | F11–F16/F18，加共用 F01–F08/F19 | ggml kimi-k3 图/后端；成熟 KDA/MLA/低精度片段；不从公式重写 |
| DeepSeek V4 Pro/Flash | CSA/HCA/窗口、压缩增量状态、sparse index、Hadamard、mHC/Sinkhorn、hash/score MoE、FP4/FP8 | F08/F12–F18，加共用域 | ggml deepseek4 图按两份配置实例化；已有 ds4/SGLang 仅用于具名成熟片段与独立参考 |
| GLM 5.3 | MLA/DSA、full/shared indexer、跨层索引复用、MoE、FP8 scaling/MTP | F12–F16/F18–F19，加共用域 | ggml glm-dsa 图及后端；保持状态/索引的 region 依赖 |
| GLM 5.3 Flash | KDA + pooled sparse index/tail、mHC、MoE | F11–F14/F16/F18–F19，加共用域 | 已登记 glm5-next 支持分支复用既有 KDA/mHC/MoE 图；版本来源单独资格化 |

HYLE 的操作名不强迫 Provider 逐个 kernel 接口。一次模型计算图调查产出操作/片段集合，按
Provider 的公共 C++/C API、图 partition 或完整 backend region 成组映射；原版源码摘取为最后
一种成熟接法，仍保留上游身份/算法，不把重写公式叫接入。模型加载、采样和服务留在 consumer。

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
| 卷积与 DeltaNet | 同文件调用 `ggml_ssm_conv` 与 `build_recurrent_attn`；[`delta-net-base.cpp`](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/models/delta-net-base.cpp)含 fused、autoregressive、chunking，融合入口返回 output 与 new state | 有完整成熟算法候选；M2 应先接入/适配，不把两输出与持久状态当成自写整核的理由 |
| DeltaNet 后端 | 同 pin 的 CPU `ops.cpp`、CUDA `ggml-cuda.cu`、Metal `ggml-metal-ops.cpp` 和 Vulkan `ggml-vulkan.cpp`均有实现/分派 | 证明有对应后端源码，不证明所有类型/shape/设备已通过 HYLE 验收；Apple 的 ggml-Metal 与 MLX 分别评估，不混为一个 runtime |
| Vulkan C1 常规算子 | `ggml-vulkan.cpp`实际分派 RMSNorm、Softmax、TopK、GetRows、SumRows、Cumsum；有 RMSNorm+Mul、RMSNorm+Mul+RoPE、SSMConv+激活等融合识别 | 按 M0 映射现成算子或融合片段；不把已有 shader 当成只能阅读的参考资料 |

Vulkan 源码也给出具体资格边界：普通 Softmax 接 contiguous F32 输入及可选 F32/F16 mask；
Gated DeltaNet 接六个 F32 输入，value head dimension 为16/32/64/128；TopK 要求连续布局并按
设备 pipeline 资格选择；Cumsum 有 subgroup 条件。这些是**适配/资格问题**，不能简写为缺算子。
例如模型片段内部维持合适的 F32 中间值可能避免逐节点转换，应在完整片段核对数值与驻留成本。
已有公开数值合同不静默放宽；真正不相容的域另记精确证据；全部必需第三方接入完成前，不据此调研或扩张自研。


## 四路线 Provider 候选与接入优先级

### 广覆盖基线与已有实现

| Provider / 路线 | 已有事实 | 新计划要求的接法 / 待裁决点 |
|---|---|---|
| ggml CPU | HYLE owner BLOCKED，空 factor 不是已接入 | 以 backend/graph region 承载模型计算；优先依法解决 fallible context/threadpool 与持久 pool seam。算法不重写 |
| ggml CUDA | 有已登记上游后端源码，尚无 HYLE 完整 backend 接入 | M0 映射需求域；M1/M2 用同 GPU 资源、stream/完成和 region 桥成批接入，不能拿已有 HYLE/CUB 核冒称 ggml |
| ggml Metal | 有上游后端来源，不是 MLX backend，尚无 HYLE 完整接入 | 对 MLX 以外必需域/整模型片段优先核对；冻结 native buffer/command/同步互操作后接入，不能强行共享 MLX runtime |
| ggml Vulkan | 已有 managed region 与具名 norm/逐元素/gate/TopK/Gather/F32 Scan/归约域 | 在同一后端扩展模型所需 region 映射，复用已验证资源 seam；不是另写一套 shader |
| oneDNN / CPU | BRGEMM、RMSNorm/LayerNorm、Add/Mul post-ops 已接；Q2 测量资格 BLOCKED | 完整 primitive/Graph partition 按需求批接；共享 descriptor/线程/资源边界。旧 Q2 仅按原条件恢复，不拖住无关 Provider |
| MLX / Apple | 已有较广 typed/compiled region 接入 | 批量补模型需要的公开运算与 state/quantization 组合，保留数组所有权与原生融合 |
| cuBLASLt / CUDA | 已有普通工厂与成熟 GEMM 接入 | 复用适用 GEMM/GEMV/精度域，新增需求按必要 shape/type 扩接接口；不写 GEMM 替代 |
| Marlin / CUDA | 固定上游量化 kernel 已接 | 复用准入 Q4 数值档位；其它格式先核对成熟实现，不把位宽相同当合同相同 |
| SGLang / CUDA | 已接 activation；AB 增加 half RMSNorm 与 Unary，W1 static 局部通过 | 按 C++ 可交付的具名片段复用；AB 的 shared/WSL/真实2B三交付/P1 待对应批次补齐，不优先给旧自研跑竞速 |

“所有应接入”由模型需求与证据决定。ggml 四后端是首选广覆盖方案；若某后端由已合格成熟库
完整覆盖且没有具名接入收益，可以在 M0 明确排除重复接入，但必须给需求逐域证据，不能静默
遗漏。各 Provider 可承载一组原子、融合片段或完整 region；不要求每个库接其全部无关 API。

### 必须在 M0 处置的专用候选

以下是有限候选池，**尚未逐项决定是否必需，也均不能据库名写成已接入**。M0 一次性把每项
与需求行绑定，裁决必需接入/已有覆盖可复用/后备/不准入；未裁决时阻止进入自实现阶段。

| 候选 | 对应模型需求 | 优先接入方式与定界 |
|---|---|---|
| FlashInfer | CUDA attention/KV、MoE/低精度等实际受支持域 | 可独立交付的 C++/CUDA API 或固定源码 region；核对硬件、缓存、失败及许可，不能安装 Python 就算 SDK 接入 |
| cuDNN Graph | CUDA attention/卷积/归一化及可融合图域 | execution plan/graph API；固定算法/准备成本，保持 region 完整 |
| CUTLASS / CuTe | CUDA dense/grouped/低精度 GEMM | 既有成熟实现实例化或现成 dispatch；模板工具箱本身不等于完整算子，不能借它默认自写 kernel |
| FlashAttention | CUDA GQA/MLA 等 attention 域 | 可交付成熟融合程序；若已由 ggml/FlashInfer/cuDNN 满足则记录重叠，不为数量重复接库 |
| MPSGraph | Apple 的具名 MLX/ggml Metal 未覆盖域 | Apple 原生图/资源桥；先证明真实需求与可交付性 |
| ncnn / MNN | Vulkan 已有后端仍不满足的具名算子/量化域 | 成熟 C++/Vulkan region 或完整算子，不以通用推理框架名义引入模型引擎 |
| XNNPACK / 平台 BLAS | CPU 具名实际缺口或完整消费收益 | 成熟 C API/BLAS 调用；ggml/oneDNN 已覆盖且无需求价值时排除 |

无需把全部候选都变成依赖，但**所有需求和候选的处置必须先完成**。某候选调查到预算还不清楚，
记录未决与具体事实缺口，禁止以“只查过两个库”为由准许自己实现。

## 接入方式与非算法边界

1. 能用成熟公开 API/完整 region 就直接接，保留其融合、packing、workspace 和 native 执行。
2. 缺 import/export、持久上下文、错误返回等边界时，先审计一个小范围 resource/runtime bridge；
   patch factor 有明确 pin/维护/升级/失败与三交付责任。接口缺陷不是重写其算子算法的理由。
3. 完整 runtime 有具体不可消除的边界时，才固定原版 kernel/融合源码的窄 C++ 接入，保留
   provenance、许可证与升级范围。数学改写属于自研，受冻结与全局前置门约束。
4. 状态、量化和多输出在 region 边界统一适配，表驱动绑定一组操作。额外转换/同步/物化写进
   成本与资源账单；不能省检查或让用户自己补 glue 来制造覆盖。

## 模型与设备版本共同参与固定路由

M0/M1 必须保留模型家族/模型 revision 与配置身份、prefill/decode/region 语义、量化/shape，
以及实际设备/SM/架构、driver/toolkit/OS/Provider 版本的条件。模型知识由 consumer 显式给出，
HYLE 不加载 checkpoint；条件只能缩小已有合法候选，不能仅凭名字跳过图/数值/资源检查。
规则可选择整片段专用计算方法或融合，而非只替换单个 op；普通入口和编译缓存必须消费同一身份。
规则/软件/模型配置变化使受影响计划失效；准备匹配绑定，执行不搜索或测速。

## 当前阻塞与下一动作

- ggml CPU：原非 abort context/threadpool/plan seam 或独立 source-patch factor 的恢复条件不变。
  这是高优先级接入阻塞，不是 CPU 算子缺失证据。
- oneDNN Q2：原 A/A 测量资格失败，产品采样0；需要原卡声明的独立新证据，不能因新 goal 重测。
- GLM Flash 支持分支、每后端类型/形状/量化与 Windows C++ 可交付性还需 M0 定界。
- 下一动作仅为 M0 全模型映射、候选处置和接入审计；禁止回到 Y/AB 后再逐个扩自写的旧队列。
