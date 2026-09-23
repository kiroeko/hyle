# M2 B2--B6 成熟 Provider 批次合同

## 1. 作用与状态

本页把[总计划](llm-moe-coverage-completion-plan.md)和
[Provider 清单](llm-moe-provider-integration-ledger.md)中的 B2--B6 变成可审计的施工边界。需求行、
模型和数值域仍以[覆盖矩阵](llm-moe-coverage-matrix.json)为准；四路线候选入口仍以
[Provider 映射](llm-moe-provider-map.json)为准。本页不把映射写成 PASS，也不授权任何 HYLE
自实现数学。

[M2 批次 manifest](llm-moe-m2-provider-batch.json)是本页的机器可验证投影：固定 B2--B6 的43行
需求归属、成熟 Provider 编译超集、CPU/CUDA/Apple/Vulkan 的9个稳定 source/static/shared profile、
低并行度和执行顺序。`COMPILED_ONLY` 只说明构建能力已声明，不能代签接线或覆盖。B2 审计时向同一
manifest 增加 gate group 与 residual，不重新创建逐算子 profile。

M2-B1 已于2026-09-23按[终局](llm-moe-m2-b1-terminal.md)结束。B2 必须先绑定当前 owner、
baseline 和有限 FIX 表并进入 AUDITED，才能修改产品代码。B3--B6 同理，不因本页存在而提前施工。每个批次只有一张活动父卡；路线和
Provider 是父卡内的批量接入单元，不为每个算子重复建卡、配置、构建或出具交付证据。

## 2. 共同执行合同

- 执行顺序固定为 B2→B3→B4→B5→B6；每批内按 CPU→CUDA→Apple→Vulkan 串行运行重型验证。
- 最小实现单元是“路线 × Provider 原生 region/能力族”。完整 region 可以承载多行需求；仍须在
  表驱动用例中逐行证明 dtype、layout、shape、状态和数值合同。
- 允许的产品改动只有成熟 Provider 的资源桥、描述转换、原版调用、完成/失败传播、静态规则绑定
  和必要的无损 packing。禁止新增或优化 CPU/CUDA/Metal/Vulkan 数学 kernel、shader、排序、
  归约、scan 或量化算法。
- 原版入口不接受某域时，先使用映射中已准入的成熟后备。只有精确反例进入残差账本；M2 不用
  自实现凑覆盖，M3 汇总真实覆盖，M4 才能审计最小补缺。
- 每条路线复用 B1 已建立的活动构建树、依赖缓存和环境 profile。一次增量构建覆盖本批全部产品
  改动；跨环境、host 插桩、source/static/shared、最终 selected 与必要 P1 在路线候选稳定后各
  集中运行一次。未变接缝按 tree/contract/toolchain/machine 身份复用。Windows 提交后用 manifest
  baseline 的 Git 增量 bundle 同步验证副本，不用 tar 覆盖 checkout 或破坏增量构建时间戳。
- B2 启动时根据本页 B2--B6 的既定需求一次声明每个 Provider 的 M2 primitive/feature 超集；后续
  批次只增加 HYLE 映射，不逐批改变 vendor 编译选项。具体上游不允许裁剪或超集会显著增加交付
  负担时记录例外，但不能到每个算子才重新配置和冷编译 Provider。
- 正确性使用同一份逻辑输入和独立 oracle。性能只裁决成熟默认路径的完整接入成本；冻结的 HYLE
  算子不参加竞速。准备允许固定规则匹配与绑定；热执行不搜索、不计时、不重新选择 Provider。
- 每批关闭条件：所有必需成熟入口已 BOUND 并完成适用门，或留下具名上游不支持/后端不接受/
  数值状态不相容反例及后续 owner；未运行项保持 UNPROVEN/BLOCKED，不能写 PASS。

## 3. 路线代码

| 代码 | 成熟承载 |
|---|---|
| `CPU-G` | ggml CPU retained graph/composite/fused region |
| `CPU-D` | oneDNN primitive/Graph partition、reorder 与成熟浮点 MatMul 组合 |
| `CUDA-G` | ggml CUDA retained native backend region |
| `CUDA-A` | FlashInfer 合格 attention/KV region，Windows 不合格域使用已准入 ggml CUDA 后备 |
| `CUDA-Q` | cuBLASLt、Marlin、SGLang/CUTLASS 与 ggml CUDA 的成熟 GEMM/量化程序 |
| `APPLE-M` | MLX public C++ typed/compiled region |
| `APPLE-G` | ggml Metal 完整 recurrent/sparse/MoE region |
| `APPLE-Q` | MLX quantization、quantized MatMul 或有界成熟解码组合 |
| `VK-G` | ggml Vulkan managed backend region |
| `VK-C` | 已限定资格的 IREE Vulkan 标准操作 region；失格域核对具名 ncnn/MNN 后备 |

路线代码只是首选承载，不是硬编码单一 Provider。模型/config/phase、设备型号/架构、驱动、SDK 与
Provider 版本可以参与冷绑定规则；规则必须有开发期证据并保存在准备结果中。

## 4. B2：线性、量化线性与注意力

### 需求与成熟承载

| 需求 | 模型消费 | CPU | CUDA | Apple | Vulkan |
|---|---|---|---|---|---|
| F05.1--F05.4 dense/batched/strided GEMM 与 FFN region | 八个目标模型；M=1/17/128，真实投影宽度 | `CPU-D` | `CUDA-Q` | `APPLE-M` | `VK-G` |
| F06.1 Q4-S128、F06.2 Q4-A64、F06.3 Q8-B32 W8A16 | 本地 Qwen3.5-2B 及相应目标格式 | `CPU-D` | `CUDA-Q` | `APPLE-Q` | `VK-C` |
| F07.1 causal GQA/MQA、F07.2 KV append/read、F07.3 gated attention | Qwen attention 与所有模型的 KV 状态 | `CPU-G` | `CUDA-A`/`CUDA-G` | `APPLE-M` | `VK-G` |
| F08.1 partial/interleaved RoPE、F08.2 YaRN、F08.3 Hadamard | Qwen、DeepSeek、GLM 对应位置公式 | `CPU-G` | `CUDA-G` | `APPLE-M` | `VK-G` |

### B2 有限 FIX 与验收

1. 一次绑定 dense/quantized FFN 和 attention/KV/RoPE 三个原生能力族；优先完整 FFN/attention
   region，不为 region 内部原子另造 adapter。
2. Q4/Q8 使用同一份逻辑 codes/scales；只允许无损重排或有界成熟 tile decode。W8A16 禁止把
   activation 重量化为 A8，禁止整模型浮点常驻。
3. GQA 同时覆盖 prefill/decode、head256、真实 Q/KV 比、非零 offset、不同 Q/K 长度；KV 覆盖
   单 token/chunk append、容量/有效长度、重复页索引和两个 invocation 隔离。
4. RoPE/YaRN/Hadamard 使用锁定模型参数和独立公式 oracle；NoPE、partial prefix、两种布局、位置
   边界及逆 Hadamard 都必须明确。
5. P0 记录 packing、workspace、同步、必要转换和压缩驻留。稳定后按 `C2.F05`--`C2.F08` 集中跑
   一次适用 P1；不与冻结自实现对比。

B2 关闭产物是 13 行 × 4 路线的真实 verdict 和模型片段收据，不要求整模型加载器。未满足的行
进入 M3-02 残差，不自动授权实现新算子。

## 5. B3：递归混合状态

| 需求 | 片段 | 四路线成熟承载 |
|---|---|---|
| F09.1--F09.2 | causal depthwise conv prefill/update | CPU/CUDA/Vulkan `*-G`，Apple `APPLE-G` |
| F10.1--F10.2 | Gated DeltaNet chunk/recurrent | 同上，复用上游 fused 或完整图组合 |
| F11.1--F11.2 | KDA chunk/recurrent、gate 与 state layout | 同上 |

有限 FIX 是把原版 conv/DeltaNet/KDA region 接到普通 Direct/Graph 与显式 state ports，保留
Provider 的状态布局、workspace 和融合。验收固定 prompt 1/17/128、257-token chunk 与逐 token
等价、非零初态/reset、两请求隔离和旧结果 lease；再用本地 2B 的 B16/Q4/Q8 输入跑 24 层、8 步
teacher-forced decode。B3 不引入采样、tokenizer、模型文件加载 API 或推理引擎。

## 6. B4：MoE、专家组合与 MTP 片段

| 需求 | 片段 | 四路线成熟承载 |
|---|---|---|
| F12.1--F12.3 | score/bias、grouped top-k、hash routing | 各路线 `*-G` 原版图/region |
| F13.1--F13.3 | dispatch permutation、grouped experts、weighted combine、shared/latent/SiTU | 各路线 `*-G` 原版图/region |
| F14.1--F14.5 | mHC、跨 block attention residual、Qwen/DeepSeek/GLM MTP | 各路线 `*-G` 原版图/region |

有限 FIX 是批量绑定 routing→dispatch→expert→combine 和三类模型特有 region；模型 identity 与
配置参与冷规则，普通用户不排序 Provider。验收覆盖空专家、极偏斜专家、ties/NaN 策略、重复
combine、真实投影宽度、共享状态、MTP index carry 和两个 invocation；服务端专家调度、通信、
采样和 speculative engine 不进入 Compute/Graph。

## 7. B5：稀疏注意力与压缩状态

| 需求 | 片段 | 四路线成熟承载 |
|---|---|---|
| F15.1--F15.2 | MLA absorbed forms、gated MLA/NoPE | CUDA 优先 `CUDA-A`，其余路线 `*-G` |
| F16.1--F16.3 | DSA learned index、跨层复用、pooled index 与 tail | 各路线 `*-G` |
| F17.1--F17.3 | CSA/HCA、sliding/overlap compression、增量状态 | 各路线 `*-G` |

有限 FIX 是接入原版 sparse/attention/index/compression region，并把选择索引、窗口与压缩状态作为
显式端口/资源保存。验收使用相同索引比较 full/shared/pooled 路径，覆盖跨层复用、unpooled tail、
窗口重叠、非零增量状态、Pro/Flash 配置和 MTP carry；转换、同步和压缩节省统一计费。

## 8. B6：FP8/FP4/MXFP4 数值档位

| 需求 | 精确格式 | CPU | CUDA | Apple | Vulkan |
|---|---|---|---|---|---|
| F18.1 | E4M3 FP8、block128 F32 scale | `CPU-D` | `CUDA-Q` | `APPLE-Q` | `VK-C` |
| F18.2 | E4M3 FP8、UE8M0 power-of-two scale | `CPU-D` | `CUDA-Q` | `APPLE-Q` | `VK-C` |
| F18.3 | E2M1 MXFP4 weights + E4M3 activation、group32 UE8M0 | `CPU-D` | `CUDA-Q` | `APPLE-Q` | `VK-C` |
| F18.4 | DeepSeek FP4/FP8、group32/group128 exact scale/rounding | `CPU-D` | `CUDA-Q` | `APPLE-Q` | `VK-C` |
| F18.5 | quantized attention/index state | `CPU-D` | `CUDA-Q` | `APPLE-Q` | `VK-C` |

有限 FIX 是绑定已准入成熟格式转换、dense/grouped MatMul 和 index region。五行分别资格，不用
“支持 FP8/FP4”的名称代签；每行保存 bit-level codes/scales、饱和/舍入、activation 分组、SiTU
再量化和 compressed-residency 证据。无原生指令可以使用同设备成熟后备组合，但不得整模型解压
常驻。验收包括独立 bit decoder、同量化值正确性、本地 2B FP8 24 层/8 步旅程以及完整转换成本。

## 9. 批次状态与停止

| 批次 | 当前状态 | 启动条件 | 关闭后动作 |
|---|---|---|---|
| B2 | [`CLOSED`](llm-moe-m2-b2-terminal.md) | 成熟入口已批量接线并完成四路线、资源和交付验收；精确格式/完整片段缺口已记为有限 residual | 直接进入 B3 |
| B3 | [`CLOSED`](llm-moe-m2-b3-terminal.md) | F32 causal conv 与 scalar/KDA GDN 已接四路线原版成熟入口；低精度、完整模型链和具名 backend 限制已进入 M3-03 residual | 直接进入 B4 |
| B4 | [`CLOSED`](llm-moe-m2-b4-terminal.md) | grouped expert MatMul 与 mHC 三枚 region 已接四路线 ggml 原版入口，普通 Direct/Graph、资源与两种 Apple 交付已验证 | 直接进入 B5 |
| B5 | [`CLOSED`](llm-moe-m2-b5-terminal.md) | Lightning Indexer 已接四路线 ggml 原版入口；MLA、pool/tail 与 compression 复用已有成熟图能力并保留真实模型片段残差 | 进入 B6 精确低精度格式资格化 |
| B6 | [`CLOSED`](llm-moe-m2-b6-low-precision.md#终局裁决) | 已接 IREE Vulkan F18.1/2 dense+BF16 组合完成 Windows shared 与 Air static 验收；其余精确域保留具名 residual | [M2 已汇总关闭](llm-moe-m2-terminal.md)并暂停，不进入 M3 |

`QUEUED_BY_PLAN` 只是工作顺序，不代替 `module-status.json` 的 component owner 状态。启动每批时只
登记实际受影响的既有/新增 component，并在实现前完成 SCOPED→AUDITED；不得为64行复制 owner。
