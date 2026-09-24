# M3-D5：真实模型片段批量验收

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owner：`compute_ops` 持有验收矩阵和公共测试 consumer。D5 默认不重开已关闭路线；只有实际片段暴露
  产品缺陷时，才以失败收据把对应 Provider/route/Graph owner 重新定界到 `SCOPED -> AUDITED`。
- baseline：`deae2ed23fd3bf9ebc30364874d94881fc8f16c3`；D4 已关闭，Windows canonical 工作树干净且与
  `origin/main` 一致。
- trigger：[M3 主卡](llm-moe-m3-provider-demand-reconciliation.md)的 D1-D4 已完成需求正规化、成熟
  Provider 接线和错误自实现退场，但 D3 candidate 只证明代表格式/程序，不等于 22 个真实 profile 的
  模型片段已运行。D5 必须消除这种证据缺口。
- 正常 consumer：普通 C++ 用户提供 exact model/artifact profile，选择 CPU、CUDA、Apple 或 Vulkan
  路线，以 Direct `prepare` 或 Graph `compile` 准备真实 region；同一 artifact 重复 submit 时不重新
  选 Provider、解码权重或建立 workspace。
- 主纵切：本地 `Qwen/Qwen3.5-2B@15852e8` 官方 BF16，直接读取用户已有 safetensors 中的真实 tensor
  slice，覆盖 dense projection、full attention、Gated DeltaNet、MTP 和状态。
- 正交纵切：Qwen3.8-2.4T IQ4 MoE、Kimi K3 MXFP4/KDA、DeepSeek V4 FP4/FP8+DSA/mHC、
  GLM-5.3 FP8+DSA 及 GLM-5.3-Flash KDA；使用固定 header/config 的真实内宽并缩小 token、expert
  capacity，不宣称整模型推理。
- scope：八个模型、22 个固定 profile、15 个语义 region；四路线各自的 Direct/Graph、两次 submit、
  profile/region 冷绑定、数值/状态、资源/完成/失败、源码/静态/动态交付，以及适用 P1。
- non-goals：模型 loader 产品化、tokenizer/采样/推理 engine、整模型常驻、下载其它巨型 checkpoint、
  新增或优化 HYLE 数学/kernel/shader、逐算子重开旧实现对比、视觉 encoder。
- 依赖与 consumer：D3 的 ggml CPU/CUDA/Metal/Vulkan native storage candidate、oneDNN/cuBLASLt/
  FlashInfer/MLX/IREE 限定域、公共 `WorkloadContext`/region signature 与 Graph compile；D6 消费本卡收据。
- performance：正确性稳定后的不可变 candidate 执行一次适用 `P1`。计入准备、必要转换、上传、同步、
  workspace 和完成；只验证自然成熟路线没有 material 工程退化，不与已删除自研算子竞速。
- machines：Windows RTX 4070 是 CPU/CUDA/Vulkan 产品真相；MacBook Air `Mac16,12` 是 Apple formal
  机，Mac mini `Mac16,10` 只签 correctness/portability。重型路线串行、构建低并行。
- budget：一个共享 profile/shape manifest、一个本地 safetensors 有界提取器、每路线一组参数化 Direct
  和 Graph consumer、每路线一次稳定后三交付与一次适用 P1。不得按 15 region × 22 profile 手写测试。
- stop：[`llm-moe-d5-acceptance-ledger.json`](llm-moe-d5-acceptance-ledger.json)的 176 个
  profile/route/entry cell 全部获得 `RUN_PASS`、`BACKEND_REJECTED` 或 `UPSTREAM_MISSING`；未运行数为
  0。若精确缺口需要新数学或 Provider，记录最小后续卡并停止，不在 D5 扩张。

## 批量验收形状

| 模型批次 | profile 数 | 真实宽度锚点 | 必验 region |
|---|---:|---|---|
| Qwen3.5-2B | 3 | H=2048、FFN=6144、GQA 8/2×256、GDN 16×128、conv K4 | R01-R05、R13-R15 |
| Qwen3.8-27B | 3 | H=5120、FFN=17408、GQA 24/4×256、GDN K16/V48×128 | R01-R05、R13-R15 |
| Qwen3.8-2.4T | 3 | H=8192、E=512/top10、expert N=2048、GQA 64/4×256 | R01-R05、R10-R11、R13-R15 |
| Kimi K3 | 3 | H=7168、E=896/top16、expert N=3584、attention 96×128 | R01-R03、R06-R07、R10-R12、R14-R15 |
| DeepSeek V4 Pro | 1 | H=7168、E=384/top6、expert N=3072、index H64×D128/top1024、hc4 | R01-R03、R07-R15 |
| DeepSeek V4 Flash | 3 | H=4096、E=256/top6、expert N=2048、index H64×D128/top512、hc4 | R01-R03、R07-R15 |
| GLM-5.3 | 3 | H=6144、E=256/top8、expert N=2048、index H32×D128/top2048 | R01-R03、R07-R11、R13-R15 |
| GLM-5.3-Flash | 3 | H=4096、E=288/top8、expert N=2048、KDA H64×D128、hc4 | R01-R03、R06-R15 |

真实宽度不缩小；只缩小 batch、token、活跃 expert 和上下文容量。量化 profile 必须保留实际 codes/scales、
block/group 与逐 tensor role 的混合格式。官方与 GGUF 是不同 profile；某条路线拒绝官方 FP4/FP8 时，
不能用同模型 GGUF 的通过结果代签。

## 能力审计与有限动作

| ID | 当前事实 | 有限动作 | 完成证据 |
|---|---|---|---|
| D5-A1 | 22 个 profile 身份和真实 shape 已固定，但尚无统一运行 ledger | 生成共享 manifest 与 176-cell ledger，测试和 evidence 只引用稳定 ID | schema/self-check 通过；每格回指模型来源和 region |
| D5-A2 | 本地 Qwen 权重已校验，现有 provider test 仍多为合成小 shape | 有界读取选定 BF16/F32 tensor slice，生成外部 scratch fixture 与独立 oracle | 文件 digest、tensor 名/offset/dtype/shape 和结果收据 |
| D5-A3 | D3 已运行真实量化 block，但没有和 exact model profile/region 绑定 | 参数化现有成熟 Direct/Graph 程序，传入 exact `WorkloadContext` 并使用真实内宽 | Provider/report、两次 submit、结果/状态、准备期拒绝 |
| D5-A4 | 生命周期测试分散在路线测试中 | 每路线批量复用现有 resource/completion/failure fixture；只补 profile 纵向组合缺口 | 无热选择/重排、失败不发布、teardown 平衡 |
| D5-A5 | D3 三交付只签代表格式，未签 profile 片段 | 候选稳定后每路线一次 source/static/shared 外部 consumer | 搬迁后运行普通入口；DLL 与 EXE 同目录 |
| D5-A6 | 完整 profile 成本尚未验收 | 每路线选一个主 profile 和一个量化/MoE 反偏置做一次适用 P1 | 完整成本、A/A 与双顺序合格；精确限制入账 |

以上六项是 D5 全部动作。A2-A4 属于测试消费者和证据；若它们揭示产品缺陷，先停止并重开精确 owner，
不得在测试内实现 fallback 或新数学。

### D5-F1：CUDA BF16 dense projection 有界修复

本地 Qwen3.5-2B 的首个真实 R03 片段暴露了一个此前合成测试没有覆盖的产品缺陷：默认 CUDA 路线的
`cuBLASLt` Provider 只接纳 FP16/FP32，`ggml CUDA` 又按既定分工把普通 rank-2 dense MatMul 留给
`cuBLASLt`，所以合法的 BF16 `[1,2048] × [2048,2048]` 在 prepare 阶段被全部成熟 Provider 拒绝，
最终只显示末位 tensor Provider 的 unsupported 原因。

本缺陷把 `compute_provider_cublas` 从既有 CLOSED 经 SCOPED 精确重开为 AUDITED，固定的 FIX 只有：

1. 让现有 cuBLASLt adapter 接纳 plain rank-2 BF16 row/column-major 输入和 dense 输出，以
   `CUDA_R_16BF` 存储、FP32 accumulation 运行；不新增 HYLE kernel、fallback 或热路径搜索；
2. 在 Provider 测试中覆盖 BF16 行/列布局、两条独立 invocation 和变化输入；
3. 让真实 Qwen R03 的普通 CUDA Direct/Graph 入口绑定 `cublasLt`，完成两次 submit 和独立 oracle；
4. 按 selector 运行 `compute_provider_cublas`、`compute_route_cuda`、`graph_route_cuda` 及公共检查。

除 BF16 数据类型映射、相应合同文字和上述测试外，不改变 cuBLASLt 算法选择、workspace、路由优先级、
其它 Provider 或任何算子域。若 CUDA 13.3 heuristic 对该真实形状仍拒绝，则记录精确 backend rejection，
不在 D5 内扩大实现。

**D5-F1 CLOSED（`67948310b`）**：cuBLASLt 13.6.0 在 RTX 4070 上接受 BF16 row/column-major
布局并以 FP32 累加运行。Provider 6/6、CUDA Direct 10/10、CUDA Graph 29/29 以及 Compute/Graph
源码、共享、静态和搬迁消费者均通过；不可变候选上的真实 R03 Direct/Graph 两次 submit 收据见
[CUDA Qwen3.5 R03](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-r03/README.md)。
两格仍是只覆盖 R03 的 partial receipt，不提前宣称完整 profile 或 P1。

### D5-F2：Vulkan BF16 decode MatMul 边界修复

同一真实 R03 片段在默认 Vulkan 路线暴露了不同的成熟 Provider 边界缺陷。固定 ggml
`456172ec733a135778adcd32d00e576a58232e45-hyle` 的 Vulkan backend 会把 rank-2、1 行 BF16
MatMul 送入 `mul_mat_vec`；该分支的公开选择条件允许 BF16 activation，但实际 pipeline 入口只接受
F32、F16 或 Q8_1 activation，因而执行时触发 `b_type` 断言。现有 rank-3 小形状测试没有覆盖这个
decode 分支。这是 HYLE 适配器对固定上游能力域限定不足，不是缺少新的 MatMul 实现。

`compute_provider_ggml_vulkan` 从 D3 的 CANDIDATE 退回 AUDITED；固定 FIX 只有：

1. 在现有 ggml Vulkan region 构图时，rank-2 BF16 MatMul 的 1–4 行 activation 先用上游 `ggml_cast`
   扩为 F32，再交给同一个上游 BF16-weight/FP32-accumulation MatMul；权重、输出语义和 Provider 不变；
2. Direct 与 Graph 用完整 Qwen3.5 `[2048,2048]` checkpoint 权重、精确 R03 workload、两次变化输入
   验证普通入口；补 1–4 行资格回归以覆盖固定上游的整个 small-matvec 分支；
3. 只运行 `compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan` 及 selector
   选中的公共检查；候选稳定后再签 Vulkan 三交付。

不修改 ggml 源码，不新增 HYLE shader、MatMul、热选择或 fallback，也不改变 F16/F32、rank-3/batched、
大于 4 行或量化 MatMul。若 F32 activation 边界仍被固定上游拒绝，则记录精确
`BACKEND_REJECTED`，不在 D5 内扩张数学实现。

**D5-F2 CANDIDATE（`958b9d69f`）**：固定 ggml Vulkan 0.24.0 在 RTX 4070 上通过 activation-only
上游 cast 承载 BF16 rank-2 decode MatMul。1–4 行边界回归、完整 `[2048,2048]` Qwen R03 的
Direct/Graph 两次 submit、ggml/IREE Provider 检查、Compute/Graph owner suite，以及源码、共享、静态和
搬迁外部 consumer 均通过；未增加 HYLE 数学、shader、fallback 或热路径选择。收据见
[Vulkan Qwen3.5 R03](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-r03/README.md)。
两格仍只覆盖 R03，不提前宣称完整 profile 或 P1；Provider 回到 D5 总验收所需的 CANDIDATE 状态。

### Apple Qwen3.5 R03 检查点

不可变 tests-only candidate `78561cec7` 在 formal Air `Mac16,12` 上直接由 MLX 0.32.2 Metal 路径
承载完整 `[2048,2048]` BF16 checkpoint projection；Direct/Graph 两次 submit、独立 oracle、两个
完整 owner suite，以及源码、共享、静态和搬迁外部 consumer 均通过。未暴露产品缺陷，也未增加 HYLE
数学、Metal kernel 或 fallback。收据见
[Apple Qwen3.5 R03](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-r03/README.md)。
两格仍只覆盖 R03，不提前宣称完整 profile 或 P1；`compute_ops` 继续持有 D5 AUDITED 总验收。

### D5-F3：Qwen3.8 Gated DeltaNet grouped-head Provider 接线

Qwen3.8-27B 官方 profile 的 R05 真实形状是 K16/V48×128。公共 `GatedDeltaNet` schema 已要求 V heads
为 K heads 的整数倍，但四个 ggml adapter 都额外要求 K/V heads 相等。固定 ggml
`456172ec733a135778adcd32d00e576a58232e45-hyle` 的 op 与 backend 将该比例作为正式能力：上游
`test_gated_delta_net` 有 `v_repeat` 维度，CUDA kernel 用独立 Q stride 和 V head dispatch，Metal/Vulkan
supports-op 也没有等头限制。Windows baseline 的精确 K16/V48×128、T=1 consumer 在 CPU prepare
阶段稳定返回 `ggml CPU operation domain is not admitted`；失败收据见
[Qwen3.8 R05 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r05-admission-audit/README.md)。

因此 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal` 与
`compute_provider_ggml_vulkan` 从各自 D3/D5 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. 四个 adapter 在既有 F32 dense/depth/token/size 边界内接纳 `V_heads % K_heads == 0` 且 K/V depth
   相同的 Gated DeltaNet；继续调用同一个上游 `ggml_gated_delta_net`，不新增 HYLE math、kernel、shader、
   fallback 或热路径搜索；
2. 共享测试 fixture 固定 `Qwen/Qwen3.8-27B@1d4bf0f`、官方 safetensors index digest、R05、
   K16/V48×128 与 decode T=1，以非零数据和独立 double oracle 覆盖头复用、state output 和两次 submit；
3. 每条路线在自己的 Direct/Graph 普通入口验证 Provider identity、workload identity、数值、状态、失败不发布
   与资源回收；Apple 在 formal Air 上运行，Windows 运行 CPU/CUDA/Vulkan；
4. 复用 D5 的每路线 source/shared/static 外部 consumer 和最终 P1 批次，不为本修复另造交付或性能仪器。

不改变公共 schema、route 顺序、GDN 状态布局或 KDA gate 语义。若某个固定上游 backend 实际拒绝该精确
比例，则该路线记录 `BACKEND_REJECTED` 和原生拒绝，不在 D5 内实现替代算子。

**D5-F3 CANDIDATE（`d7cb7e01`）**：四个 adapter 只放宽到公共 schema 已允许且固定 ggml 原生实现
支持的 `V_heads % K_heads == 0`。同一份官方 Qwen3.8-27B R05 K16/V48×128、decode T=1 fixture 在
Windows CPU/CUDA/Vulkan 和 formal Air Apple M4 的普通 Direct/Graph 入口全部通过，两次 submit 的
输出与新 state 均匹配独立 double oracle；完整受影响 owner suite 也通过。未增加 HYLE 数学、kernel、
shader、fallback 或热选择。收据见 [Qwen3.8 R05 grouped-head](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r05-grouped-head/README.md)。
该收据仍只覆盖 R05；其余 region、三交付与 P1 未完成，所以 176-cell ledger 不提前改变任何 cell。

### D5-F4：Qwen3.5 CPU BF16 RoPE 类型边界

本地 Qwen3.5-2B 官方 profile 的 R04 使用 BF16 activation、GQA 8/2×256 和 F32 accumulation。批量
Graph consumer 在普通 CPU 路线准备成功后，于固定 ggml CPU 的 `ggml_compute_forward_rope` 中触发
`fatal error`：HYLE adapter 的资格函数接受 BF16 RoPE，但该上游 CPU 入口只执行 F16/F32；同一资格
函数反而拒绝上游原生支持的 F32。源代码条件与实际失败一致，收据见
[Qwen3.5 CPU R04 RoPE admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r04-cpu-rope-admission-audit/README.md)。

因此 `compute_provider_ggml` 从 D5-F3 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. CPU adapter 的 RoPE 资格域与固定上游对齐为 F16/F32，并继续接纳 Qwen 官方 BF16；BF16 在冷准备的
   ggml 图内使用上游 `ggml_cast` 转成 F32，执行同一上游 RoPE，再由既有输出边界转回 BF16；
2. Provider 回归覆盖 BF16/F32、两种 RoPE layout 和变化输入；真实 Qwen3.5 R04 的普通 Graph 入口
   覆盖精确 8/2×256、两次 submit、Provider/workload identity 和独立 decode oracle；
3. 只运行 `compute_provider_ggml`、`graph_route_cpu` 及 selector 选择的公共检查；三交付和 P1 继续并入
   D5 路线批次，不为本缺陷新增仪器。

不修改公共 RoPE schema、ggml 源码、路由优先级或其它路线，也不新增 HYLE 数学、fallback 或热选择。

**D5-F4 CANDIDATE（`f340b0d4`）**：CPU adapter 以固定 ggml 的既有 Cast/RoPE 图执行 BF16，F16/F32
保持原生类型域；BF16/F32 Provider 回归、Qwen3.5 精确 GQA 8/2×256 Graph、ggml CPU 34/34、CPU
Direct 12/12、CPU Graph 34/34 与普通 cpu-executor 全部通过。未修改公共 schema、上游源码或路由顺序。
失败与修复边界收据见 [R04 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r04-cpu-rope-admission-audit/README.md)。

### D5-F5：Qwen3.5 CPU Direct BF16 SwiGLU 成熟接线

官方 Qwen3.5-2B R02 的普通 CPU Direct 路径按 BF16 存储边界逐算子准备：RmsNorm 与三次 MatMul
均绑定固定 ggml CPU，独立 `GatedActivation(SiLU, separate)` 却落到已冻结的 `hyle-native`。固定 ggml
0.24.0 已提供 `ggml_swiglu` 与 `ggml_swiglu_split`；其 CPU 实现原生执行 F16/F32，BF16 可在同一上游
图内 Cast 到 F32 并由既有输出边界 Cast 回 BF16。源代码与普通入口复现见
[Qwen3.5 CPU Direct R02 gate admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r02-cpu-direct-gate-admission-audit/README.md)。

因此 `compute_provider_ggml` 从 D5-F4 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. CPU adapter 接纳公共 `GatedActivation` 的 SiLU packed/separate 两种形状，lower 到固定上游
   `ggml_swiglu`/`ggml_swiglu_split`；F16/F32 保持原生，BF16 通过上游 Cast 执行；
2. Provider 回归覆盖 F16/BF16/F32、packed/separate、变化输入与非 SiLU 拒绝；真实 Qwen3.5 R02
   使用真实 2048/6144 权重、两次 Direct replay、独立 BF16 边界 oracle 和成熟 Provider identity；
3. 只运行 `compute_provider_ggml`、`compute_route_cpu` 及 selector 选择的公共检查；三交付和 P1 继续并入
   D5 路线批次。

不修改公共 schema、ggml 源码、路由优先级、其它 activation 或其它路线；不新增 HYLE 数学、fallback、
热选择或性能竞速。若固定上游不能正确执行该精确域，则记录 `BACKEND_REJECTED`，不恢复 native 作为
D5 成熟覆盖证据。

**D5-F5 CANDIDATE（`3798fd11`）**：CPU adapter 已用固定 ggml 的 `ggml_swiglu`/
`ggml_swiglu_split` 接管 F16/BF16/F32 的 packed/separate SiLU；BF16 只在同一上游图内 Cast 到 F32，
再由既有输出边界转回。Provider 35/35、CPU Direct 19/19、CPU Graph 两个测试目标和 fixture 输入工具
13/13 均通过；未修改公共 schema、上游源码、路由顺序或其它 activation。

### CPU Qwen3.5 official Graph 完整格

不可变 candidate `f340b0d4` 已用本地 checkpoint 的 embedding slice、layer-0 norm/FFN/conv/GDN/out-proj、
layer-3 Q/K norm 与 MTP 权重，完成 R01-R05、R13-R15 的普通 CPU Graph 两次 replay、独立 oracle、
成熟 Provider identity、资源与失败检查；既有 R03 收据并入同一格。完整收据见
[CPU Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-official-graph/README.md)。
ledger 首次得到 1 个 `RUN_PASS`；CPU Direct、其它路线、三交付和 P1 仍未由该格代签。

### CPU Qwen3.5 official Direct 完整格

不可变 candidate `3798fd1129323c53c373dbdf7bd628b1a1f6365d` 复用同一 checkpoint fixture，
通过普通 `compute::cpu` 上下文逐算子冷准备并执行 R01-R05、R13-R15。R02 保留真实 BF16 边界并由
ggml SwiGLU 与 MatMul 承载；每个 prepared call 两次变化输入 replay，R01 另验证越界失败不发布，
既有 R03 收据并入完整格。完整收据见
[CPU Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-official-direct/README.md)。
ledger 现有 2 个 `RUN_PASS`；其它路线、三交付和 P1 未由这两格代签。

### D5-F6：Qwen3.5 CUDA BF16 SwiGLU 成熟接线

官方 Qwen3.5-2B R02 的普通 CUDA Direct 路径对精确 BF16 `[1,6144]` separate
`GatedActivation(SiLU)` 冷准备成功，但绑定的是已经冻结的 `hyle-cuda` fallback。固定 ggml
0.24.0 CUDA backend 已提供 `ggml_swiglu`/`ggml_swiglu_split`，原生执行 F16/F32；其 adapter 当前既不
资格化也不 lower 公共 `GatedActivation`。普通入口复现和固定上游源码边界见
[Qwen3.5 CUDA R02 gate admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r02-cuda-gate-admission-audit/README.md)。

因此 `compute_provider_ggml_cuda` 从 D5-F3 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. CUDA adapter 接纳公共 `GatedActivation` 的 SiLU packed/separate 两种形状，lower 到固定上游
   `ggml_swiglu`/`ggml_swiglu_split`；F16/F32 保持原生，BF16 在同一 ggml CUDA 图内 Cast 到 F32，
   再由现有输出边界 Cast 回 BF16；
2. Provider 回归覆盖 F16/BF16/F32、packed/separate、变化输入与非 SiLU 拒绝；真实 Qwen3.5 R02
   使用精确 2048/6144 权重、两次 replay、独立 BF16 边界 oracle 和成熟 Provider identity；
3. 同一批完成 Qwen3.5 official CUDA Direct/Graph 的 R01-R05、R13-R15 普通入口验收；只运行
   `compute_provider_ggml_cuda`、`compute_route_cuda`、`graph_route_cuda` 及 selector 公共检查，三交付和
   P1 继续并入 D5 路线批次。

不修改公共 schema、ggml 源码、路由优先级、其它 activation 或其它路线；不新增 HYLE 数学、kernel、
fallback、热选择或性能竞速。若固定上游拒绝精确类型/形状，则记录 `BACKEND_REJECTED`，不恢复 native
作为 D5 成熟覆盖证据。

### D5-F7：Qwen3.5 CUDA BF16 RoPE 类型边界

F6 接线后的完整 CUDA Direct 执行批次中，R02 FFN 已通过真实权重与独立 oracle；R04 的精确 BF16
GQA 8/2×256 RoPE 则在普通入口 prepare 成功后，于 invocation 初始化/执行固定 ggml CUDA 的
`rope.cu` 触发 `src0->type == F32 || F16` 断言。adapter 的资格函数接受 BF16，但 lowering 将 BF16
原样交给只执行 F16/F32 的上游 CUDA RoPE；源码条件与真机失败一致。收据见
[Qwen3.5 CUDA R04 RoPE admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r04-cuda-rope-admission-audit/README.md)。

`compute_provider_ggml_cuda` 已因 F6 处于 AUDITED；本次在任何 RoPE 产品修改前追加的有限 FIX 只有：

1. adapter 的 RoPE 资格域与固定上游对齐为 F16/F32，并继续接纳官方 BF16；BF16 在同一 ggml CUDA
   图内使用上游 `ggml_cast` 转成 F32，执行同一上游 RoPE，再由现有输出边界转回 BF16；
2. Provider 回归覆盖 F16/BF16/F32、两种 RoPE layout、变化输入与不支持类型拒绝；完整 Qwen3.5
   CUDA Direct/Graph 批次继续验证精确 8/2×256、两次 replay、成熟 Provider/workload identity 和独立
   decode oracle；
3. 验证集合保持 F6 的 `compute_provider_ggml_cuda`、`compute_route_cuda`、`graph_route_cuda`、selector
   公共检查与 D5 路线三交付/P1，不另造仪器或测试矩阵。

不修改公共 RoPE schema、ggml 源码、路由优先级或其它路线；不新增 HYLE 数学、kernel、fallback 或热
选择。固定上游若拒绝 Cast 后的精确图，则记录 `BACKEND_REJECTED`，不绕回 native。

**D5-F6/F7 CANDIDATE（`9917438b`）**：固定 ggml CUDA adapter 已用上游
`ggml_swiglu`/`ggml_swiglu_split` 接管 F16/BF16/F32 packed/separate SiLU，并以同一 ggml CUDA 图内
Cast 衔接 BF16 RoPE。Provider 24/24、CUDA Direct 18/18、CUDA Graph 35/35 在静态与共享树通过；
共享树启用 WERROR，DLL 与 EXE 同目录。不可变 SM89 candidate 的 selector 又完成 configure、四个选中
factor 并得到 PASS；没有新增 HYLE 数学、kernel、fallback 或执行期选择。

### CUDA Qwen3.5 official Direct/Graph 完整格

不可变 candidate `9917438bb44ebc29f647f2b0ce2919a73342fa3e` 复用同一真实 checkpoint fixture，
通过普通 CUDA Direct `prepare`/submit 和 Graph compile/bind/replay 覆盖 R01-R05、R13-R15。每个
prepared call/graph 使用变化输入至少执行两次，成熟 Provider 在冷准备时固定；R02 SwiGLU 与 R04
RoPE 不再落入冻结的 `hyle-cuda`。CUDA 首次执行允许固定上游建立一次 graph metadata，稳定性从第一轮
warm replay 到第二轮 replay 验证；失败路径不发布输出。完整收据见
[CUDA Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-official-direct/README.md)
和
[CUDA Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-official-graph/README.md)。
ledger 现有 4 个 `RUN_PASS`、172 个 `VALIDATION_NOT_RUN`。本批静态/共享 owner suite 不代签 D5-A5
要求的 relocated external source/static/shared consumer；D5-A5 与 D5-A6 P1 仍随稳定 CUDA 路线批次执行。

### Vulkan Qwen3.5 official Direct/Graph 完整格

不可变 candidate `1fa67a8facf76feb78f66b5576aee45ac7d16d99` 复用同一真实 checkpoint
fixture，通过普通 Vulkan Direct `prepare`/bind/submit 和 Graph build/compile/bind/submit 覆盖
R01-R05、R13-R15。embedding/head、FFN、GQA/RoPE/attention、causal conv/GDN 及 MTP/BF16
边界均使用精确官方 shape 和真实权重；每个 prepared program 以变化输入、token 或 state 建立独立
invocation 重放，失败 Gather 不发布输出。Direct 冷准备逐项断言固定 ggml Vulkan managed region 或限定
IREE dispatch，Graph 冷编译逐项断言成熟 managed region；没有回接冻结的 HYLE 自研数学。

不可变 W1-WIN/MSVC 19.50/WERROR selector 的 docs、closure、configure、完整
`compute_route_vulkan` 与 `graph_route_vulkan` factor 均为 PASS；定向 Direct 6/6、Graph 5/5 也通过。
完整收据见
[Vulkan Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-official-direct/README.md)
和
[Vulkan Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-official-graph/README.md)。
ledger 现有 6 个 `RUN_PASS`、170 个 `VALIDATION_NOT_RUN`。本批共享 WERROR owner suite 不代签
D5-A5 要求的 relocated external source/static/shared consumer；D5-A5 与 D5-A6 P1 仍随稳定 Vulkan
路线批次执行。

### Apple Qwen3.5 official Direct/Graph 完整格

不可变 tests-only candidate `e99c2bf929bbee747dbb8a392ab3ff813299be3f` 在 formal Air
`Mac16,12` 上复用同一真实 checkpoint fixture，通过普通 Apple Direct `prepare`/bind/submit 和
Graph build/compile/bind/submit 覆盖 R01-R05、R13-R15。MLX 0.32.2 保留原生 Metal region，
Gated DeltaNet 保留完整 ggml Metal region；Graph compile report 逐项确认成熟实现策略，Program 持有
捕获常量。变化输入/token/state、独立 invocation、失败不发布和 BF16/F32 边界均实际运行。

定向 Direct 5/5、Graph 5/5，完整 Direct owner 52/52、Graph owner 25/25 在 AppleClang 21、WERROR
下通过；Windows 不可变 selector 的 docs、closure 与 Apple-OFF exact-factor configure 也通过。完整收据见
[Apple Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-official-direct/README.md)
和
[Apple Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-official-graph/README.md)。
ledger 现有 8 个 `RUN_PASS`、168 个 `VALIDATION_NOT_RUN`。已有 R03 三交付证明继续有效，但不提前代签
D5-A5 的最终稳定 Apple 路线批次；D5-A6 P1 也仍未运行。

## 验收合同

1. `RUN_PASS` 必须来自不可变 candidate 上的普通 Direct 或 Graph 入口，记录 exact profile、region、shape、
   provider/software/device identity 和输出 oracle；source symbol、target、枚举或 prepare-only 均不算。
2. 同一 prepared artifact 至少 submit 两次；第二次必须改变 activation 或 state，证明没有 stale output，且
   submit 不重新选路、计时、解码或分配持久 workspace。
3. `BACKEND_REJECTED` 需要精确 route/device/provider/version/format/shape 的 prepare/capability 失败；
   `UPSTREAM_MISSING` 需要固定成熟源码和 provider inventory 的具名缺口。不得保留模糊 `NOT_RUN`。
4. 本地 Qwen 实际 bytes 只作为测试输入复制到验证 scratch；不提交模型数据，不让远程副本覆盖 Windows
   产品源码。其它巨型模型使用固定 header/config 与格式合法的非零 payload，不冒充实际 checkpoint bytes。
5. region 可以由 Provider 融合承载，也可以由 Graph 组合成熟节点；HYLE 只检查边界、路由、状态和结果，
   不为“看见每个算子”拆坏上游融合。

## 预声明验证

- 文档/ledger 先运行 closure、docs 与 JSON self-check，不构建、不跑 P1。
- 测试 consumer 稳定后按 selector 选择 CPU、CUDA、Apple、Vulkan 的直接目标；重型路线串行。
- 候选固定后才运行三交付与 P1。未运行项保留 `VALIDATION_NOT_RUN`，绝不提前写 PASS。
- D6 只汇总红队、最终矩阵、限制、LOC 和状态，不在终审首次接线。

## AUDITED

八个模型批次、22 个 profile、真实宽度、176 个运行 cell、六项有限动作、机器、P1 与停止条件已经冻结。
D5 从 A1/A2 开始批量实现；新增模型、profile、Provider 或自研数学均须退回重新定界。
