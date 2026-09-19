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
下表按模型定位成熟实现；M0 已交付[逐域映射](llm-moe-provider-map.json)，后端实际验收仍不能预填 PASS。

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


## M0 准入决定与施工入口

[详细施工清单](llm-moe-provider-integration-ledger.md)已冻结 Provider 的必需/已有复用/后备/
条件资格处置、精确源码身份、M1资源接口包、M2六个模型片段批次及适用验证。它与
[64行四路线映射](llm-moe-provider-map.json)共同替代原“候选待定”队列；产品状态仍只看 registry。

| 路线 | 必需广覆盖与已有复用 | 专用/条件资格 | 后备与触发条件 |
|---|---|---|---|
| CPU | ggml CPU；oneDNN primitive/partition/reorder | ggml fallible context/threadpool/plan 独立 patch factor；精确 W8A16/FP8 数值组合 | XNNPACK/平台 BLAS：主方案出现具名实际缺口时核对，不为重复 GEMM 数量接入 |
| CUDA | ggml CUDA；cuBLASLt/Marlin/SGLang | FlashInfer C++ attention/KV；CUTLASS/CuTe 只作为已有成熟程序依赖；head256/SM89/Windows 资格 | cuDNN Graph、FlashAttention：按实际适用域补位；不引入 Python/Torch runtime |
| Apple | MLX typed/compiled region + ggml Metal 完整递归/稀疏 region | MLX locked pin 的量化/FP8入口；两个 native runtime 的资源/完成桥 | MPSGraph：具名 MLX/ggml 不满足域时核对；OS/SDK版本进入资格 |
| Vulkan | 已接 ggml Vulkan managed region 的模型图批量映射 | IREE Vulkan 条件资格 Q-VK：标准整数 scan 与 FP8/E2M1/UE8类型处理；未运行不得签覆盖 | ncnn/MNN：主方案在精确格式/设备/交付上失格时先核对；不能直接转自写 |

上游源码表明可以先接成熟模型片段；也存在必须明说的边界：CPU ggml量化dot不自动满足
W8A16，ggml没有本合同FP8 dtype，Vulkan F32 Scan不等于checked I32 exclusive scan。
这些差异分别交给具名成熟组合/候选资格项，不降低原64域数值与四路线要求。

## 固定的接入边界

1. 优先公开 C++/C API 或完整 region，保留第三方的融合、packing、workspace和native执行。
2. 缺 import/export、持久上下文、失败返回时，独立审计窄resource/runtime bridge；不是改写数学。
3. 必须借用源码时采用固定的原版kernel/融合程序，保留provenance、许可与升级范围；改写公式
   或手写shader/MLIR算法属于自研，不能借第三方名字绕过冻结。
4. 状态、格式、多输出在片段边界统一适配；转换、同步、物化和检查计入完整资源成本。
   普通Direct/Graph按规则消费，不让用户补Provider glue或排序候选。

模型家族/revision/配置、prefill/decode/region语义、shape/数值/量化与GPU/SM/架构、
driver/toolkit/OS/Provider版本共同限定规则。consumer显式提供模型事实，HYLE不加载模型；
名字不能跳过图与数值检查。相关事实纳入计划/编译缓存identity，改变时重新准备；热执行不探测、
搜索或计时。M1-R用未知模型、版本不匹配、缓存失效和普通入口测试签收。

## 当前完成度和阻塞

- M0的映射与详细计划已交付；源码存在、候选资格、产品绑定和实际验收分开记录。
- ggml CPU原runtime abort与oneDNN Q2的BLOCKED保持原恢复条件，不能靠新计划解除。
- Q-VK、Q-CUDA-API及数值/状态资格是下一步有限接入工作；当前未执行，不写PASS。
- 全CPU/CUDA/Metal/Vulkan自研算子冻结，全部第三方接入完成前禁止扩张、调研、优化和性能比较。
  最后M5统一有限裁决；SIMD基础库保留，Compute SIMD算子参加同样的淘汰。
- 按用户要求先报告计划完成。后续首项为M1-G资源seam审计，绝不恢复旧Y/AB逐个自写队列。
