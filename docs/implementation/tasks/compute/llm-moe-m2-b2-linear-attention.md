# M2-B2 成熟 Provider 线性、量化线性与注意力批次

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- round / owner：M2-B2；`/root`。
- baseline：`ba63ca79107cd82dc21d781b51a35391839b1497`；Windows canonical 工作树干净。
- trigger：目标模型对 F05.1--F08.3 共13行的真实消费，以及本地 Qwen3.5-2B 的 FFN、GQA/KV、
  RoPE 片段。现有公共 schema 已有 `MatMul`、Q4 `QuantizedMatMul`、`RoPE` 与 GQA/MQA attention，
  但四路线尚未把本批成熟 Provider 域全部接到普通 Direct/Graph，也没有本批资源、三交付和完整片段
  验收。这是缺失常用能力与公开路线变化的合法 reopen trigger。
- 主纵切：相同 logical weights/codes/scales 的 dense/quantized FFN，随后执行 Qwen GQA prefill/decode、
  KV append/read 与 partial/interleaved RoPE；覆盖 M=1/17/128、真实 Q/KV 比、head256 和非零 position。
- 正交纵切：YaRN 与 Hadamard/位置预处理片段，验证 Provider region 不接受某形状时的成熟后备、
  两 invocation、旧结果 lease、失败不发布及重复执行不重建程序。
- 当前机器：Windows RTX 4070 SM89/CUDA 13.3/Vulkan；WSL 用于 Linux CUDA/FlashInfer 与静态/插桩；
  Air/Mini 用于 Apple 与 MoltenVK。产品真相只在 Windows，远端用 Git 增量 bundle 更新。
- 性能：先做 P0 packing/workspace/sync/copy/压缩驻留账单。API 与整片段稳定后，每路线只对完整
  FFN/attention 接入成本运行一次适用 P1；不按13行逐算子计时，不与冻结 HYLE 数学竞速。
- 调查预算：每路线成熟入口/接法核对最多45分钟，总计180分钟；正常实现、编译与预声明验证另计。
  已登记 pin 和入口直接复用。出现改变公开 identity、资源 owner 或常态成本等级的新事实即停止并
  重定界；不通过添加 Provider 或改治理工具续预算。
- stop：已准入成熟入口全部 BOUND 并完成适用门，或留下具名上游不支持/语义不相容反例与 residual
  owner；未运行保持 `UNPROVEN/BLOCKED`。新增自研 kernel/shader/算法为0。B2 关闭后直接进入 B3。

## Owner 与依赖边界

| 路线 | 本轮 owner | SCOPED 时的边界 |
|---|---|---|
| common | `compute_ops` | 先审计现有 schema 能否表达13行；只有成熟 region 确需且不能由现有 tensor/state 端口表达时才增加语义对象，不增加数学实现 |
| CPU | `compute_provider_onednn`、`compute_provider_ggml`、`graph_provider_onednn`、`compute_route_cpu`、`graph_route_cpu` | owner 已为 M2 `AUDITED`；审计 oneDNN MatMul/reorder 与 ggml retained FFN/attention/position region，复用原 runtime |
| CUDA | `compute_provider_cublas`、`compute_provider_cuda`、`compute_provider_ggml_cuda`、`graph_provider_ggml_cuda`、`compute_provider_flashinfer`、`compute_route_cuda`、`graph_route_cuda` | cuBLASLt、FlashInfer 与 Graph route 因新 batched/attention 消费从 `CLOSED` 重开；Marlin 保持 `CLOSED` 并按现有 signed-Q4/group128 factory 只读复用 |
| Apple | `compute_route_apple`、`graph_route_apple` | 继承 [Apple B1 terminal](llm-moe-m2-b1-apple.md)，因 B2 MLX typed/compiled region 与 Graph 消费从 `CLOSED` 重开；ggml Metal 保持 `CLOSED`，本批不以 recurrent provider 承载 MLX 线性/注意力 |
| Vulkan | `compute_provider_ggml_vulkan`、`compute_provider_iree_vulkan`、`compute_route_vulkan`、`graph_route_vulkan` | owner 已为 M2 `AUDITED`；审计 ggml Vulkan managed region，IREE 只承载精确合格的标准/量化补位 |

`compute_ggml_runtime`、CUDA backend、Mem 和构建公共 seam 不在当前预期 FIX；出现可复现基础缺陷时
另作定向 owner，不在 B2 顺手重写。`compute_provider_marlin`、`compute_provider_ggml_metal` 的既有
产品不改时不伪造 reopen；只增加测试不要求改变其 terminal 状态。

## SCOPED 审计清单

1. 从13行需求逐一映射 Provider 原版入口、pin、dtype/layout/shape、native resource/program、
   Direct/Graph 接法和成熟后备；`COMPILED_ONLY` 不算 BOUND。
2. 对 F05 固定 dense、batched、strided 与 FFN region；对 F06 固定 Q4-S128、Q4-A64、Q8-B32
   W8A16，禁止把 activation 重量化成 A8，禁止整模型浮点常驻。
3. 对 F07 固定 prefill/decode GQA/MQA、KV 容量/有效长度/offset、单 token/chunk append、重复页索引、
   gated attention 与两请求隔离；Provider 完整 region 优先，不拆坏上游融合和资源管理。
4. 对 F08 固定 partial/interleaved RoPE、YaRN、Hadamard 的模型参数、布局和独立公式 oracle；明确
   NoPE、边界 position 和 inverse 形状。
5. 在同一 [M2 manifest](llm-moe-m2-provider-batch.json) 中追加有限 gate group、适用性与 residual；
   不新建逐算子 profile，不改 B2--B6 已冻结 Provider feature 超集。
6. 审计完成只留下 `KEEP/FIX/BORROW/DEFER/REJECT`，然后将实际修改 owner 绑定本卡为 `AUDITED`；
   未证明需要修改的 CLOSED owner 不进入施工集合。

## SCOPED 验证计划

- 表驱动 schema/provider 资格和独立 oracle；每路线普通 Direct/Graph、资源/失败、重复执行与两 slot。
- 复用 manifest 的9个 source/static/shared profile；每路线稳定后一次增量 build，test 串行，路线按
  CPU→CUDA→Apple→Vulkan 执行，未变门只接受 selector 签署的 `REUSED`。
- change-impact selector 只运行实际受影响 factor/consumer；Windows DLL 邻置启动，不改 UAC/电源设置；
  重型验证低并行且串行。
- 路线候选稳定后才运行预声明完整片段 P1；若语义/融合/转换不同，先修正合同，不能做苹果对橘子比较。

## AUDITED：成熟入口与有限施工表

2026-09-23 在 SCOPED 检查点 `205b2644` 后核对锁定的 ggml `456172ec`、oneDNN
`v3.13.2`、MLX `0.32.2`、CUDA 13.3/cuBLASLt、Marlin `51918e25`、FlashInfer
`0.6.18` 与现有 IREE/Vulkan 接缝。只核对 F05--F08 的原版入口和产品 glue，没有研究、扩张或
比较冻结的 HYLE kernel/shader/SIMD 算子。以下是 backend-domain 审计，不是覆盖 PASS。

| 能力族 | CPU | CUDA | Apple | Vulkan |
|---|---|---|---|---|
| F05 dense/batch/stride | `KEEP` oneDNN rank2 F32 与现有完整 FFN region；`BORROW` ggml `mul_mat` 补 half/batch/stride | `KEEP` cuBLASLt rank2；`FIX` 其 batch/stride，`BORROW` ggml region 保留完整融合 | `KEEP` MLX rank2；`FIX` MLX 原生 batch/half 域，Graph 保留一个 compiled region | `BORROW` ggml Vulkan `mul_mat`；现有仅 MXFP4 的窄适配不代签 dense 域 |
| F06 Q4/Q8 W*A16 | oneDNN/ggml 只接经原版能力查询证明不把 activation 量化为 A8 的域；格式不匹配保持 `DEFER` | `KEEP` Marlin Q4-S128 M=1..64；`BORROW` ggml Q4/Q8 和合格 cuBLASLt；M=128 或格式不合保持 `DEFER` | `KEEP` MLX affine Q4-A64；`FIX` MLX 原版可接受的 4/8-bit group 域和无损冷 packing | `BORROW` ggml Vulkan Q4_0/Q4_1/Q8_0；仅精确格式/组宽匹配的格可 BOUND |
| F07 GQA/KV/完整片段 | `BORROW` ggml flash-attention/原版组合；复用 B1 Scatter/Gather state ports | `KEEP` FlashInfer 的 8Q/2KV/head256 FP16/BF16 与 ggml CUDA fallback；复用 B1 state ports | `BORROW` MLX SDPA 和已有 MLX Scatter/Gather；完整 Graph 仍是一个 compiled region | `BORROW` ggml Vulkan flash-attention；复用 B1 Scatter/Gather 和 native completion |
| F08 RoPE/YaRN/Hadamard | `BORROW` ggml `rope_ext` 与 Hadamard-hinted `mul_mat` | 同一 ggml CUDA 原版路径 | MLX 原生 `hadamard_transform`；RoPE/YaRN 由 MLX 原语在 compiled region 内执行 | 同一 ggml Vulkan 原版路径，须先过 backend support probe |

### 语义、资源与路由决定

- `MatMul` 继续表示 NumPy batch broadcast；物理转置/stride 留在 `TensorDesc`。`QuantizedMatMul`
  增加 signed-symmetric 8-bit 语义，四路线只接受具名 packing；禁止将 W8A16 偷换为 W8A8。
- 新增纯 `Hadamard` 语义，只描述最后一维的正交归一化变换和 inverse；实现调用 MLX 原语或
  ggml 的成熟 Hadamard hint，不新增 HYLE 数学 kernel。YaRN 继续由调用者提供已经按模型参数生成的
  frequency tensor；执行期不读取模型配置或重新裁决缩放规则。
- KV 容量、有效长度、position/page index 与可变 state 继续是显式 tensor/state ports。B2 复用
  B1 已接的成熟 Gather/Scatter，不另造推理引擎式 KV cache 对象。
- Provider 能力只在 prepare 时匹配。模型/config/phase、GPU 型号/SM/架构、驱动、CUDA/SDK 和
  Provider 版本可以参与固定规则；hot run 不搜索、不计时、不重建程序。
- 每个完整 FFN/attention region 由原 Provider 保有 packing、workspace、stream/thread pool、
  completion 与内部融合。HYLE 只管 region 边界、端口、状态和失败不发布。

### 有限 FIX

| ID | 允许的产品改动 | 验收 |
|---|---|---|
| B2-C0 | `compute_ops` 增加 Q8 与 Hadamard 公共语义、推导和边界；不增加数学实现 | 13 行 schema 表、错误域与独立 FP64/解码 oracle |
| B2-C1 | CPU ggml 私有 recipe 成批加入 MatMul/QMM、RoPE、SDPA、Hadamard；保留 oneDNN leaf/FFN region 和普通 CPU 固定优选 | M=1/17/128、prefill/decode、两 slot、旧结果与不支持域精确拒绝 |
| B2-C2 | cuBLASLt batch/stride；ggml CUDA 同一 recipe 的 dense/Q8/RoPE/Hadamard；复用 Marlin、FlashInfer 与现有 attention | Windows SM89 与 WSL 原版路径、native stream/completion、head256、压缩驻留 |
| B2-C3 | Apple MLX typed/compiled region 加 batch/half MatMul、合格量化、SDPA、RoPE/YaRN、Hadamard | Air/Mini 原生 array/stream、重复执行不重编译、Graph 最大 region |
| B2-C4 | ggml Vulkan region 加精确支持的 dense/Q4/Q8、SDPA、RoPE/Hadamard；IREE 只保留已证明的标准补位 | RTX 4070 与 MoltenVK backend probe、无 CPU payload fallback、失败/完成 |
| B2-C5 | 四路线共用表驱动 case、模型片段、P0 成本、manifest gate/residual、三交付与用户文档 | selector 适用门；路线稳定后各一次完整 FFN/attention P1，不逐算子计时 |

上表之外的产品改动为0。某格式、形状或设备被原版 capability probe 拒绝时，记录具名
`UNPROVEN` residual，不能用冻结自研实现或浮点全权重常驻补洞。`compute_provider_marlin`、
`compute_provider_flashinfer`、`graph_provider_onednn` 和 `compute_provider_ggml_metal` 当前产品实现
保持只读；若后续出现必须改其 owner 的可复现事实，先重新定界，不能借本表顺手修改。

本次源码/API 审计累计保守记 **35/180分钟**（SCOPED 前置10、四路线入口与格式17、Hadamard/
模型语义8）；正常实现、构建和预声明验证另计。当前仍没有新增 BOUND、覆盖或性能 PASS；现在开始
按 B2-C0→C1→C2→C3→C4→C5 批量施工。
