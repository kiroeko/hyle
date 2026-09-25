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

## 接线优先的批次执行规则

依照[Compute 算子批次合同](../../../architecture/compute-design.md#61-算子接入按有限批次接线和验证)，
D5 不再以单个 op/片段作为构建和完整验证的单位。先按下表固定模型/权重角色及已接入的成熟
Provider，批量写完该组四路线 Direct/Graph 的必要接线、夹具和 oracle；批末冻结不可变候选，
一次选择受影响门并复用同一构建树。前一批完整证据可按精确身份复用。API/算法稳定后才做路线
三交付和适用 P1；未运行格及门始终保留 `VALIDATION_NOT_RUN`，不能用编译目标或单片段代签。

当前 profile 批次是 Qwen3.8-2.4T 官方 BF16。其 R10/R11 接线子组覆盖 512-way/top-10
路由、动态 grouped expert、独立 sigmoid shared expert 与最终相加，在 CPU、CUDA、Apple、
Vulkan 的普通 Direct/Graph 入口接通；F19 的四路线 F32 score 定向结果不能替代整条链。
该子组的定向测试与完整 owner suite 是 partial receipt，不能触发 profile 级全量 selector、
三交付或 P1。继续批量接通同一 profile 的 R01-R05/R13-R15 后冻结候选，集中验证四路线的
变化输入重放、近并列成员、独立数值 oracle、成熟 Provider report、资源/完成/失败和交付。
若某路线真实缺少固定上游能力，记录精确拒绝并停止该路线，不用 HYLE 自研数学补洞。之后的
Kimi、DeepSeek、GLM 和量化格式按同样有限批次推进，不逐算子重建。

**当前 BF16 full MoE 批次结果（部分收据）**：四路线各自的完整 Graph owner 与 Direct owner
suite 均已通过 512-way/top-10、动态 grouped expert、shared expert 和 final add 的三次变化输入
重放及独立 oracle。Graph 的 CUDA 256-byte workspace 对齐缺陷已按 D5-F20 修复；Apple Direct
消费者的惰性 MLX reshape 必须在下一次 bind 前求值。见
[Graph 收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-full-router-graph/README.md)
与 [Direct 收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-full-router-direct/README.md)。
这只证明 R10/R11 的完整链，不能代签官方 BF16 整个 profile 的其它 region、三交付或 P1；
R01 资格纠正后 ledger 为 0/176 `RUN_PASS`。其余模型继续按有限批次接线，到批末再选受影响门。
不可变 `c2c1f7e8d` 的提前 selector 尝试选中 13 门；docs、closure、共享模型输入工具与
W1-WIN/MSVC 配置已运行通过。它在 CPU factor 首次编译共用 ggml CUDA 对象时人为停止，
余下 factor 未获 selector verdict；这次试运行不改变上述 partial receipt 或 ledger，
完整 selector 留到官方 BF16 profile 接线稳定后的批末执行。

**R05 causal conv 批内检查点（`55ff8c517`）**：官方 2.4T 的
`(2×16+128)×128=20,480` 通道、K4 causal Conv1d 已通过四路线 Direct/Graph
八条普通入口的两次变化输入和独立 oracle；见
[定向收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-causal-conv/README.md)。
这不触发逐算子的完整 selector/三交付/P1。此前 R01 的七行词表测试不能签官方
248,320 行词表；后续共用 BF16 词表批内检查点已接通真实宽度的 embedding/head
边界，见下文。2.4T 官方八格继续保持 `VALIDATION_NOT_RUN`，不把定向测试记作全格通过。

**R01 资格纠正（`ae15979a8` 后审计）**：此前 Qwen3.5-2B 与 Qwen3.8-27B
六个 profile 的 48 个已签格，均将真实 248,320 行词表缩成七行完成 Gather/输出头；
这不符合本卡仅缩小 batch、token、活跃 expert 和上下文容量的规则。
旧收据保留为 R01 的有限接线证据，其它已运行 region 以 partial receipt 保留；
[`ledger`](llm-moe-d5-acceptance-ledger.json) 将 48 格退回 `VALIDATION_NOT_RUN`，
当前完整格为 0/176。先按共用真实词表宽度批量接通四路线 Direct/Graph，批末再
统一运行完整 profile gate；历史收据中的 48/176 是纠正前的快照，不再代表当前验收。

R01 的 248,320 行允许按硬件资源无损分片执行：每片保留该 profile 的真实 hidden、dtype、
权重格式与原始行号，全部分片的 logits 必须按全局行序拼成完整词表并检查；Gather 必须验证
跨片全局索引和越界失败。分片程序可复用，但不得只抽样几行或把分片宽度当作模型词表宽度。
这属于图/消费者的资源规划，不新增 HYLE 数学或让 Provider 热路径重新选择。

**R01 共用 BF16 词表批内检查点（`c4e2b0d0`）**：Qwen3.8-27B 与 2.4T
官方 profile 的 248,320 行输出头在四路线 Direct/Graph 上无损分片运行，保留各自
5120/8192 hidden 与 BF16 类型；两次变化输入的全部 logits 按全局行序核对独立 oracle。
Gather 已验证跨片和尾片的具名全局索引。32 个定向用例全部通过，八个 owner suite
均无失败；六个依赖本地 Qwen3.5 夹具的用例在每个 suite 中 SKIP。详见
[共用词表部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-full-vocabulary-batch/README.md)。
这仍不是 R01 或完整 profile 的验收：本批尚未新增 248,320 全局越界拒绝用例，
量化画像的精确权重格式未在这份 BF16 收据内运行，完整 selector、三交付与适用 P1
也未签署。完整 profile 接线稳定后再集中验证，ledger 保持 0/176 `RUN_PASS`。

**R01 Qwen3.5-2B 共用词表批内扩展（`ae54c66c`）**：同一分片与 oracle
消费者已复用于本地官方 BF16 profile 的 H=2048、V=248,320；四路线 Direct/Graph
各自的输出头与全局 Gather 共 16 个定向用例通过，使用既有隔离构建树增量编译。
见[2B 词表部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-full-vocabulary-batch/README.md)。
本次没有再启动完整 owner suite、全路线 selector、三交付或 P1；这些门仍在
profile 接线稳定后的统一批末执行。全局越界和真实量化格式仍需同批补足，
ledger 保持 0/176 `RUN_PASS`。

**R01 BF16 全局越界批内检查点（`d2250624`）**：上述 2B、27B、2.4T
官方 profile 的词表消费者在四路线 Direct/Graph 共 24 个定向用例上拒绝
`-1` 和 `248,320` 全局索引；绕过消费者检查提交尾片本地索引 `2560`
时，成熟 Provider 的 checked Gather 返回越界，失败不取代已发布结果。
见[共用越界部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen-full-vocabulary-bounds/README.md)。
BF16 词表边界的该项缺口已有运行证据；其它量化权重格式、完整 profile
组合、全量 selector、三交付与适用 P1 仍待稳定批末验收，ledger 保持
0/176 `RUN_PASS`。

## 批量验收形状

| 模型批次 | profile 数 | 真实宽度锚点 | 必验 region |
|---|---:|---|---|
| Qwen3.5-2B | 3 | H=2048、FFN=6144、GQA 8/2×256、GDN 16×128、conv K4 | official：R01-R05、R13-R15；两份 GGUF：R01-R05、R14-R15 |
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

### D5-F8：GGUF 量化 embedding 原生行读取接线

Qwen3.5 `UD-Q4_K_XL` 的精确 embedding 权重为 Q6_K；目标 GGUF 画像还要求 Q4_K 与 Q8_0。
Windows baseline `0c7293ff38ac078c095e623062398087a5a89ba1` 上，普通 CPU Direct 以真实隐藏宽度
`[7,2048]` 准备 Q6_K table 时被 adapter 的通用 GGUF 端口门拒绝，最终落到 native provider 的
`plain scalar tensor storage` 拒绝。固定 ggml `b9d43bc809551dde` 的 CPU、CUDA、Metal 和 Vulkan
源码均已有 Q4_K/Q6_K/Q8_0 GET_ROWS；这是 adapter 接线缺陷，不是上游算子或 HYLE 数学缺口。收据见
[GGUF embedding admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-gguf-embedding-admission-audit/README.md)。

因此 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal` 和
`compute_provider_ggml_vulkan` 从各自 candidate 精确退回本卡 AUDITED。有限 FIX 只有：

Apple 的普通 Direct 与 Graph 还各有一层 GGUF carrier/region 准入，因此本路线实施前另将
`compute_route_apple` 与 `graph_route_apple` 从旧 CLOSED 经 SCOPED 精确重开为 AUDITED；它们只负责
把同一 checked embedding 域交给 `compute_provider_ggml_metal`，不改变其它 Apple 运算或 MLX region。

1. 四个 adapter 只接纳 rank-2 Q4_K/Q6_K/Q8_0 embedding table、axis `0/-2`、I32 indices、dense
   F32 output，调用固定上游 `ggml_get_rows`；不写 HYLE 解量化器、kernel 或 shader；
2. 保留公共 checked Gather 语义：越界在结果发布前失败且输出不变；prepared program 以变化 indices
   重放两次，不重新选路、解码权重或分配持久 workspace；
3. 以 Qwen3.5 `UD-Q4_K_XL` 的 Q6_K `[7,2048]` 纵切在四路线普通 Direct/Graph 实际运行；后续
   Q4_K/Q8_0 profile 复用同一参数化 fixture，不手写路线乘画像测试；
4. 只运行四个 provider owner suite、对应 Direct/Graph factor 与 D5 最终三交付/P1。其它格式、轴、
   路由顺序和自研数学继续冻结；固定上游若在具体设备拒绝精确域，按合同记录
   `BACKEND_REJECTED`。

**D5-F8 CPU CANDIDATE（`42f0320f`）**：CPU adapter 已在固定 ggml 的
`ggml_get_rows` 上接纳 Q4_K/Q6_K/Q8_0 精确 embedding 域。Qwen3.5 primary GGUF 的 Q6_K
`[7,2048]` R01 在普通 Direct/Graph 中完成 profile 冷绑定、两次变化 indices replay、独立 packed-byte
oracle、checked 失败不发布和 Graph 零 replay metadata 分配。i9-13900K/RTX 4070、MSVC 19.50、
WERROR 的 selector 对 docs、closure、configure、Provider、CPU Direct 与 CPU Graph 全部 PASS；收据见
[CPU Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-primary-gguf-r01/README.md)。

**D5-F8 CUDA CANDIDATE（`889bb81a`）**：CUDA adapter 复用同一固定上游
`ggml_get_rows`，接纳完全相同的 Q4_K/Q6_K/Q8_0 embedding 域；checked indices 在任何 native graph
写输出前验证，失败不发布，invocation 的验证存储在准备时一次分配。Qwen3.5 primary GGUF 的 Q6_K
`[7,2048]` R01 已在 RTX 4070 上通过普通 Direct/Graph 的两次变化 indices replay、独立 packed-byte
oracle、checked 失败输出保持和 Graph 零 replay metadata 分配。CUDA 13.3.73、driver 610.62、SM89、
MSVC 19.50、WERROR 的不可变 selector 对 docs、closure、configure、Provider、CUDA Direct、
ggml CUDA Graph provider 与 CUDA Graph 全部 PASS；收据见
[CUDA Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-primary-gguf-r01/README.md)。

**D5-F8 Apple CANDIDATE（`94264ba2`）**：Apple Direct 的 GGUF carrier 与 Graph region 已把同一
checked embedding 域交给固定 ggml Metal `ggml_get_rows`。Qwen3.5 primary GGUF 的 Q6_K
`[7,2048]` R01 在 formal Air M4 上完成普通 Direct/Graph 冷绑定、两次变化 indices replay、独立
packed-byte oracle、越界提交前失败、输出保持及后续恢复；Provider owner suite 同时运行 Q4_K/Q6_K/
Q8_0。固定上游 Metal Q4_K 的独立解码资格界为实测 `0.4% + 2e-4`，Q6_K/Q8_0 保持 `1e-6`。
AppleClang 21、Release/WERROR selector 的 docs、closure、configure、Provider、Direct 与 Graph 全部
PASS；收据见 [Apple Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-primary-gguf-r01/README.md)。

#### D5-F8 Vulkan route reopen audit

Windows baseline `65138ac77` 的固定路径仍不能把量化 embedding 交给成熟实现：
`compute_provider_ggml_vulkan::region_supports` 只接纳 plain I32 unchecked Gather，导致 GGUF table 不能
成为 managed region；随后 `prepare_plan` 会在 native `lower_gather` 前以“compressed storage requires a
matching mature region”拒绝。既有 `lower_gather` 的 GPU bounds check、safe indices 和 checked publication
适用于公共失败语义，但它后续按逻辑 `data_type_size` 把 plain tensor 重排成 raw I32 rows，不能解释 GGUF
block storage。固定 ggml Vulkan 已有目标 Q4_K/Q6_K/Q8_0 GET_ROWS，因此这是精确的 route/provider
adapter seam，不是算子或 shader 缺口。

本事实满足 `compute_route_vulkan` 的 reopen trigger；该 owner 从旧 D4 CLOSED 只为 D5-F8 量化
embedding 边界进入 SCOPED。`graph_route_vulkan` 不需要新的产品 lowering，它继续复用同一 Compute plan，
只在候选上完成 Graph 普通入口验收。

有限 FIX 固定为：

1. `compute_provider_ggml_vulkan` 只新增 exact rank-2 Q4_K/Q6_K/Q8_0 F32 embedding table、axis
   `0/-2`、dense rank-1 I32 indices 和 dense F32 output 的 unchecked native region 资格，仍只 lower 到
   固定上游 `ggml_get_rows`；unchecked 是 provider 内部边界，不成为公共 API；
2. `compute_route_vulkan::lower_gather` 只对该 exact GGUF 域复用既有 GPU `lower_index_check`，把 safe
   indices 和原量化 table 交给一个 provider region，再用既有 checked publication 决定是否发布 F32
   output；不做 raw-word 重排、host readback、新 shader、新数学或 fallback；
3. Provider owner suite 运行 Q4_K/Q6_K/Q8_0、两次 replay 和错误域拒绝；Qwen3.5 primary GGUF Q6_K
   `[7,2048]` 在普通 Vulkan Direct/Graph 运行两次变化 indices、独立 packed-byte oracle、越界输出保持
   与恢复，并确认一个 managed region；
4. 只运行 `compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan` 与 selector 选中的
   公共检查。本 partial R01 不启动 D5-A5 三交付或 D5-A6 P1，也不改变其它 Gather、格式、轴或路线。

若固定上游在 RTX 4070 上拒绝任一精确格式，记录具名 `BACKEND_REJECTED`；不得回接冻结的自研
Vulkan shader 或把量化 bytes 当普通标量 storage。

实施前 ledger 仍为 8 个 `RUN_PASS`、168 个 `VALIDATION_NOT_RUN`；CPU、CUDA 与 Apple 共六个
primary GGUF cell 只新增 R01 partial receipt，其余 region 未运行，因此不提前把完整 cell 写成 PASS。

**D5-F8 Vulkan CANDIDATE（`8af3a29d`）**：Provider 只为 exact Q4_K/Q6_K/Q8_0 embedding
开放内部 unchecked `ggml_get_rows` region；普通路线先运行既有 GPU bounds check，把 safe indices 与原始
量化 table 交给固定上游，再由既有 checked publication 决定是否发布结果。Qwen3.5 primary GGUF 的
Q6_K `[7,2048]` R01 已在 RTX 4070 上通过普通 Direct/Graph 的两次变化 indices replay、独立
packed-byte oracle、越界不发布、保留既有结果及 fresh invocation 恢复；owner suite 同时运行三个目标
格式。Vulkan SDK 1.4.357.0、driver 610.62、MSVC 19.50、Release/WERROR selector 的 docs、closure、
configure、ggml/IREE Vulkan Provider、Direct 与 Graph 全部 PASS；收据见
[Vulkan Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-primary-gguf-r01/README.md)。
ledger 计数仍为 8 个 `RUN_PASS`、168 个 `VALIDATION_NOT_RUN`，但四路线共八个 primary GGUF cell
现均有 R01 partial receipt；完整 cell、三交付与 P1 仍未代签。

### D5-F9：Qwen3.5 primary GGUF 混合投影批量验收

权重审计固定的 `UD-Q4_K_XL` 不是单一 Q4 文件：FFN、full attention 与 Gated DeltaNet projection
实际混合 Q4_K、Q5_K、Q6_K、Q8_0 和 IQ4_XS。D5-F9 不新增产品实现，而是用同一参数化 consumer
批量验证每条路线既有 ggml adapter 是否真正承载这些精确 tensor role、内宽和物理格式；只有实际失败
才按本卡规则重新打开对应 owner。

**D5-F9 CPU partial（`6834a6c9`）**：普通 CPU Direct/Graph 已运行五个精确代表 tensor：
`blk.0.ffn_gate.weight` Q4_K `[2048,6144]`、`blk.0.ffn_down.weight` Q6_K
`[6144,2048]`、`blk.3.attn_k.weight` Q5_K `[2048,512]`、`blk.8.ffn_gate.weight`
IQ4_XS `[2048,6144]` 和 `blk.0.ssm_out.weight` Q8_0 `[2048,2048]`。两条入口均冷绑定
固定 ggml CPU，以两个变化 activation 重放并匹配独立 packed-weight decode oracle；Graph 每项只有一个
成熟 region。不可变 MSVC 19.50/WERROR selector 的 docs、closure、fixture 工具、Direct 与 Graph
全部 PASS，未暴露产品缺陷，也没有新增 HYLE 数学。收据见
[CPU primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-primary-gguf-projections/README.md)。
本 partial 只签 R03/R14；R02、R04、R05、R15 和完整 cell 仍未代签，ledger 总计数不变。

**D5-F9 CUDA partial（`9dc37c98`）**：同一参数化 consumer 已在 RTX 4070/SM89 的普通 CUDA
Direct/Graph 上运行全部五个精确 tensor，均冷绑定固定 `ggml-cuda`；设备 activation 变化两次，结果匹配
独立 packed-weight decode oracle，Graph 每项只有一个成熟 region。CUDA 13.3.73、MSVC 19.50、
Release/WERROR 的不可变 selector 通过单架构 configure、Direct 与 Graph，docs/closure 从相同合同复用；
没有新增 CUDA kernel 或 fallback。收据见
[CUDA primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-primary-gguf-projections/README.md)。
本 partial 同样只签 R03/R14，完整 cell 和 ledger 总计数不变。

#### D5-F9 artifact region applicability correction

固定 revision `f6d5376` 的 Qwen3.5 `UD-Q4_K_XL` 与 `UD-Q8_K_XL` 均各有 320 个 tensor；
完整 header 名单的 MTP/NextN 匹配数都是 0，两个名单完全相同，且仓库树中没有 MTP sidecar。
因此模型家族的官方 BF16 checkpoint 继续要求 R13，但这两个 GGUF artifact 不要求 R13。当前 ledger
已按具体 artifact identity 删除这两份 profile 的 R13；不能凭官方 checkpoint 的能力为 GGUF 虚造权重，
也不能用官方 R13 收据代签 GGUF。审计收据见
[Qwen3.5 GGUF region applicability](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-gguf-region-applicability-audit/README.md)。

**D5-F9 Vulkan partial（`e31a6e7d`）**：同一组五个精确 tensor 已在 RTX 4070 的普通 Vulkan
Direct/Graph 上运行，均冷绑定固定 `ggml-vulkan`；两次变化的 device activation 匹配独立
packed-weight decode oracle，Direct 每项观察到一个成熟 region，Graph 的单 MatMul 通过成熟 managed
region 编译。Vulkan SDK 1.4.357.0、driver 610.62、MSVC 19.50、Release/WERROR 的不可变 selector
通过 docs、closure、configure、Direct 与 Graph；没有新增 HYLE Vulkan shader 或 fallback。收据见
[Vulkan primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-primary-gguf-projections/README.md)。
本 partial 同样只签 R03/R14，完整 cell 和 ledger 总计数不变。

**D5-F9 Apple partial（`ef9c0562`）**：同一组五个精确 tensor 已在正式 MacBook Air M4 的普通
Apple Direct/Graph 上运行，固定 `ggml Metal` 并实际加载 Q4_K、Q5_K、Q6_K、IQ4_XS、Q8_0
`mul_mv` pipeline；两次变化的 shared-buffer activation 匹配独立 packed-weight decode oracle。
AppleClang 21、macOS SDK 27.0、Release/WERROR 的不可变 selector 通过 docs、closure、configure、
Direct 与 Graph。首次完整 owner suite 使用了旧 R03 partial 夹具，4 个既有测试因缺文件拒绝；从 Windows
canonical checkpoint 重建并同步当前 12 张量夹具后，Direct/Graph 均 2/2 PASS。收据见
[Apple primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-primary-gguf-projections/README.md)。
本 partial 同样只签 R03/R14，完整 cell 和 ledger 总计数不变。

### D5-F10：Qwen3.5 primary GGUF 完整路线格

**D5-F10 CPU complete（`b2e6a27f`）**：在既有 R01、R03/R14 精确收据上，普通 CPU
Direct/Graph 已补齐该 artifact 的全部剩余适用 region。R02 运行完整
`2048 -> 6144 -> 2048` RMSNorm/SwiGLU FFN，gate/up 保留原生 Q4_K、down 保留原生 Q6_K；
R04 运行 F32 norm、half-split RoPE 和精确 `8 query heads / 2 KV heads x 256` GQA 语义；R05
运行 K4 causal conv 与 `16 x 128` Gated DeltaNet 状态更新。每条链以两个变化 activation/state
重放，逐中间结果匹配独立 oracle；R02 Direct/Graph 在准备完成后的重放区间均为零 metadata
allocation/reallocation。所有调用和 Graph compile report 只固定到 ggml CPU 或 oneDNN，没有新增
HYLE 数学或 fallback。

固定 GGUF artifact 不含 R13；这不是跳过适用能力，具体 header/repository 证据仍由 D5-F9 的 artifact
applicability correction 约束。不可变 MSVC 19.50 Release/WERROR selector 通过 docs、closure、fixture
工具、configure、CPU Direct 与 CPU Graph，127 个无关 gate 跳过或复用；P1 未启动。完整收据见
[CPU primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-primary-gguf-complete/README.md)。
ledger 现有 10 个 `RUN_PASS`、166 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定 CPU
路线批次执行，不由本 tests-only candidate 代签。

**D5-F10 CUDA complete（`ae2b789b`）**：普通 CUDA Direct/Graph 在既有 R01、R03/R14 收据上
补齐相同 artifact 的 R02、R04、R05 与 R15。R02 保留 Q4_K gate/up 和 Q6_K down；R04 按真实
模型合同使用 BF16 activation/storage、F32 accumulation，而不是无适用成熟 attention Provider 的
F32-only 测试形状；R05 覆盖 K4 causal conv 和 `16 x 128` Gated DeltaNet 状态更新。所有准备调用与
Graph step 均固定到 ggml CUDA、cuBLAS、SGLang/CCCL native 或 FlashInfer 成熟路径，没有新增 HYLE
数学或 fallback。固定 ggml CUDA 允许在首次 submit 建立 CUDA Graph metadata，第二次变化输入重放不再
发生 metadata allocation/reallocation。

不可变 Windows RTX 4070/SM89 candidate 以 `CUDAARCHS=89-real` 通过 MSVC 19.50 Release/WERROR
selector 的 docs、closure、configure、CUDA Direct 与 CUDA Graph；127 个无关 gate 跳过或复用。
完整收据见 [CUDA primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-primary-gguf-complete/README.md)。
ledger 现有 12 个 `RUN_PASS`、164 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定
CUDA 路线批次执行，不由本 tests-only candidate 代签。

**D5-F10 Apple complete（`690d5f2c`）**：正式 MacBook Air M4 上的普通 Apple Direct/Graph
在既有 R01、R03/R14 收据上补齐相同 artifact 的 R02、R04、R05 与 R15。R02 运行完整
`2048 -> 6144 -> 2048` F32 RMSNorm/SwiGLU 链，并保留 Q4_K gate/up 与 Q6_K down；R04 以
primary GGUF 模型身份运行 BF16 norm、half-split RoPE 和精确 `8/2 x 256` GQA；R05 以同一
模型身份运行 K4 causal conv 与 `16 x 128` Gated DeltaNet 状态更新。量化路径实际加载 ggml
Metal Q/K/IQ pipeline，其余计算固定到 MLX 0.32.2 或完整 ggml Metal region；没有新增 HYLE
数学、fallback 或热路径选路。

不可变 AppleClang 21、macOS SDK 27.0、Metal 17F109、Release/WERROR selector 通过 docs、
closure、configure、Apple Direct 与 Apple Graph；127 个无关 gate 跳过或复用。完整收据见
[Apple primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-primary-gguf-complete/README.md)。
ledger 现有 14 个 `RUN_PASS`、162 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定
Apple 路线批次执行，不由本 tests-only candidate 代签。

**D5-F10 Vulkan complete（`e3b2a5fa`）**：Windows RTX 4070 上的普通 Vulkan Direct/Graph
在既有 R01、R03/R14 收据上补齐相同 artifact 的 R02、R04、R05 与 R15。R02 运行完整
`2048 -> 6144 -> 2048` F32 RMSNorm/SwiGLU 链，保留 Q4_K gate/up 与 Q6_K down；Graph 将完整
逻辑链一次编译为一个成熟 managed region。R04/R05 以 exact primary-GGUF 身份复用已验的 BF16
`8/2 x 256` GQA、K4 causal conv 与 `16 x 128` Gated DeltaNet 语义。Direct 的每项均固定到
ggml Vulkan managed region 或限定 IREE dispatch；没有新增 HYLE shader、数学或 fallback。

不可变 MSVC 19.50、Vulkan SDK 1.4.357.0、driver 610.62、Release/WERROR selector 通过 docs、
closure、configure、Vulkan Direct 与 Vulkan Graph；127 个无关 gate 跳过或复用。完整收据见
[Vulkan primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-primary-gguf-complete/README.md)。
ledger 现有 16 个 `RUN_PASS`、160 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定
Vulkan 路线批次执行，不由本 tests-only candidate 代签。

### D5-F11：Qwen3.5 fidelity GGUF 完整路线格

**D5-F11 CPU complete（`74101b6e`）**：普通 CPU Direct/Graph 已完整运行固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile 的全部适用 region。测试没有从 artifact 名称推断全局
Q8：R01 使用真实 F16 embedding；R02 使用 Q8_0 gate/up/down；R03/R14 分别运行 Q8_0 FFN/V/O
投影和 F16 GDN-output/K/Q 投影；R04/R05 保留真实 `8/2 x 256` GQA、K4 causal conv 与
`16 x 128` Gated DeltaNet 语义，state/conv/norm 为 F32。每个 prepared chain 或 Graph program
以变化输入重放，匹配独立 Q8_0 packed-weight、F16 dense 或 F32 fragment oracle；无效 embedding
索引失败不覆盖已发布结果。所有调用与 Graph compile report 只固定到 ggml CPU 或 oneDNN，没有新增
HYLE 数学、fallback 或热路径选路。固定 320-tensor artifact 不含 MTP/NextN，因此 R13 仍按已审计
applicability 证据排除。

不可变 MSVC 19.50 Release/WERROR selector 通过 docs、closure、模型输入工具、configure、CPU Direct
与 CPU Graph；Direct 可执行文件 30/30、Graph route 可执行文件 45/45。首次 selector 进程缺少
Visual Studio `rc`/`mt` 环境而在 compiler configure 停止；同一 candidate 加载 `VsDevCmd` 后通过，
期间没有源码变化。完整收据见
[CPU fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 18 个 `RUN_PASS`、158 个 `VALIDATION_NOT_RUN`。CUDA、Apple、Vulkan 六格以及 D5-A5
三交付、D5-A6 P1 仍未由本 tests-only candidate 代签。

**D5-F11 CUDA complete（`d407d9cc`）**：普通 CUDA Direct/Graph 已完整运行同一固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile。R01 的 F16 embedding 冷绑定原始 CCCL checked
Gather，并按其合同持有 workspace；R02 保留 Q8_0 gate/up/down；R03/R14 分别运行 Q8_0 与 F16
投影；R04/R05 保留精确 `8/2 x 256` GQA、K4 causal conv 和 `16 x 128` Gated DeltaNet 语义。
每条 Direct/Graph 路径准备一次并以变化输入重放，匹配独立 Q8_0、F16 或 fragment oracle；无效
embedding 索引失败不覆盖已发布结果。所有调用和 Graph report 只固定到 ggml CUDA、cuBLAS、
SGLang/CCCL native 或 FlashInfer，没有新增 HYLE 数学、fallback 或热路径选路。R13 继续按固定
artifact 的已审计缺失排除。

不可变 RTX 4070/SM89、MSVC 19.50、CUDA 13.3、Release/WERROR selector 通过 docs、closure、
configure、CUDA Direct 与 CUDA Graph；完整 Direct 29/29、Graph 46/46，127 个无关 gate 跳过或
复用。完整收据见
[CUDA fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 20 个 `RUN_PASS`、156 个 `VALIDATION_NOT_RUN`。Apple、Vulkan 四格以及 D5-A5 三交付、
D5-A6 P1 仍未由本 tests-only candidate 代签。

**D5-F11 Apple complete（`65bdb5f5`）**：普通 Apple Direct/Graph 已完整运行同一固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile。R01 的 F16 embedding 由 MLX Metal 执行；
R02 的 Q8_0 gate/up/down 以及 R03/R14 的 Q8_0 投影由原始 ggml Metal 执行，F16 投影由
MLX Metal 执行；R04/R05 保留精确 `8/2 x 256` GQA、K4 causal conv 和 `16 x 128`
Gated DeltaNet 语义。每条 Direct/Graph 路径准备一次并以变化输入重放，匹配独立 Q8_0、F16 或
fragment oracle；无效 embedding 索引失败不覆盖已发布结果。所有调用和 Graph report 只固定到
MLX 或 ggml Metal，没有新增 HYLE 数学、fallback 或热路径选路。R13 继续按固定 artifact 的已审计
缺失排除。

不可变 MacBook Air M4、AppleClang 21、macOS SDK 27.0、Release/WERROR selector 通过 docs、
closure、configure、Apple Direct 与 Apple Graph；完整 Direct 59/59、Graph 32/32，127 个无关
gate 跳过或复用。完整收据见
[Apple fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 22 个 `RUN_PASS`、154 个 `VALIDATION_NOT_RUN`。Vulkan 两格以及 D5-A5 三交付、
D5-A6 P1 仍未由本 tests-only candidate 代签。

**D5-F11 Vulkan complete（`123c87c1`）**：普通 Vulkan Direct/Graph 已完整运行同一固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile。R01 的 F16 embedding、R02 的 Q8_0
gate/up/down，以及 R03/R14 的 Q8_0 与 F16 投影均固定到成熟 ggml Vulkan region；R04/R05
保留精确 `8/2 x 256` GQA、K4 causal conv 和 `16 x 128` Gated DeltaNet 语义，并只使用已准入的
ggml 或有限 IREE Vulkan dispatch。每条 Direct/Graph 路径准备一次并以变化输入重放，匹配独立
Q8_0、F16 或 fragment oracle；无效 embedding 索引失败不覆盖已发布结果。候选没有新增 HYLE
数学、fallback、Provider 或热路径选路，R13 继续按固定 artifact 的已审计缺失排除。

不可变 Windows RTX 4070、driver 610.62、MSVC 19.50、Vulkan SDK 1.4.357.0、Release/WERROR
selector 通过 docs、closure、configure、Vulkan Direct 与 Vulkan Graph；完整 Direct 142/142、
Graph 53/53，127 个无关 gate 跳过或复用。完整收据见
[Vulkan fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 24 个 `RUN_PASS`、152 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍未由本
tests-only candidate 代签。

### D5-F12：Qwen3.8 CPU GQA 资格域修复

Qwen3.8-27B 官方 profile 的 R04 使用 BF16、24 query heads / 4 KV heads × 256 和 F32 累加。
普通 CPU Direct 在 RMSNorm/RoPE 准备成功后拒绝 SDPA；同一 executable 的 Qwen3.5 8/2×256 GQA
通过。源码定位到 `compute_provider_ggml` 的 SDPA 资格函数把成熟路径写死为 `Q=8、KV=2、D=256`，
而紧邻的固定 ggml lowering 已使用 `ggml_mul_mat` 的 head-batch 广播，没有 8/2 专用算法。失败和源码
边界见 [Qwen3.8 R04 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r04-cpu-gqa-admission-audit/README.md)。

因此 `compute_provider_ggml` 从 D5 candidate 精确退回本卡 AUDITED。有限 FIX 只有：

1. 在既有 dense BF16/F16/F32、causal、无显式 mask、F32 accumulation 域内，将 SDPA 资格从固定
   8/2 改为正 head 数、`Q_heads % KV_heads == 0`、Q/K/V depth 相同、K/V 形状相同和既有输出形状；
2. 把 Qwen3.8 `attn_output_gate=true` 所需的公共 Sigmoid 只映射到固定 ggml 已有的
   `ggml_sigmoid`；保留 F32 内部执行和既有输出转换，验证 BF16、有限尾部、±∞ 与 NaN，不开放其它 unary；
3. Provider 回归同时保留 8/2×256 并新增 24/4×256，两者继续 lower 到同一固定 ggml program；
4. Qwen3.8 官方 CPU Direct/Graph 使用 exact profile、真实内宽、两次变化输入和独立 oracle 完成
   R01-R05、R13-R15；既有 R05 K16/V48×128 收据并入完整格；
5. 只运行 `compute_provider_ggml`、`compute_route_cpu`、`graph_route_cpu` 与 selector 选择的公共检查。

不修改公共 SDPA/Unary schema、ggml 源码、算法、路由顺序、native Provider 或任何 HYLE 数学/kernel；若固定
ggml 实际执行 24/4 仍失败，则记录精确 `BACKEND_REJECTED`，不在 D5 实现替代 attention。

不可变 candidate `667dda0d6a696eab0d3f243386a73f2f54cefe55` 已完成该有限 FIX。CPU Direct 与
Graph 均在官方 H=5120、FFN=17408、GQA 24/4x256、partial interleaved RoPE 64、MTP 10240
形状上通过 R01-R04、R13-R15，并合并既有 R05 K16/V48x128 收据；每个 artifact 使用变化输入
replay，失败路径不发布结果，编译报告只含 ggml CPU/oneDNN 成熟 Provider。Release/WERROR selector
的 docs、closure、模型输入工具、configure、ggml Provider、CPU Direct 与 CPU Graph 七个 gate
全部 PASS，126 个无关 gate 跳过或复用；未启动 P1。完整收据见
[Qwen3.8 official CPU complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen38-official-complete/README.md)。
ledger 现有 26 个 `RUN_PASS`、150 个 `VALIDATION_NOT_RUN`；CUDA、Apple、Vulkan 六格以及 D5-A5
三交付和 D5-A6 P1 尚未由本批代签。

### D5-F13：Qwen3.8 CUDA GQA 资格域修复

Qwen3.8-27B 官方 profile 的 R04 使用 BF16、24 query heads / 4 KV heads × 256 和 F32 累加。
普通 CUDA Direct 的 RMSNorm、partial interleaved RoPE 和输出 Sigmoid 均可准备，但 SDPA 在准备期被
默认路线拒绝；同一 executable 的 Qwen3.5 8/2×256 GQA 已通过。源码定位到 `compute_provider_ggml_cuda`
的 SDPA 资格函数把 query/KV head 数写死为 8/2，紧邻的成熟 lowering 已使用固定 ggml CUDA
`ggml_mul_mat` head-batch 广播，没有 8/2 专用算法。失败和源码边界见
[Qwen3.8 CUDA R04 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r04-cuda-gqa-admission-audit/README.md)。
放宽 GQA 后，同一普通入口又显示输出门公共 Sigmoid 落入冻结的 `hyle-cuda`；固定 ggml CUDA
tensor program 已承载同组 unary，且上游已有 `ggml_sigmoid`，只是 adapter 的资格表和 lowering 漏接。

因此 `compute_provider_ggml_cuda` 从 D5 candidate 精确退回本卡 AUDITED。有限 FIX 只有：

1. 在既有 dense BF16/F16、causal、无显式 mask、F32 accumulation、SM89 资格域内，将 ggml CUDA
   SDPA admission 从固定 8/2 改为正 head 数、`Q_heads % KV_heads == 0`、Q/K/V depth 相同、K/V
   形状相同和既有输出形状；
2. 把公共 Sigmoid 只映射到固定 ggml 的 `ggml_sigmoid`，复用 tensor program 既有 F32 内部执行和
   输出转换；验证 BF16 有限输出，不开放其它 unary；
3. 保持 FlashInfer 的 8/2×256 已资格化原生 kernel 域不变；24/4 由固定 ggml CUDA 成熟图接管，
   不新增 HYLE kernel、fallback、热路径搜索或路由顺序；
4. Provider 回归保留 8/2×256 并新增 24/4×256，两者必须准备到成熟 Provider；Qwen3.8 CUDA
   Direct/Graph 使用 exact profile、两次变化输入、独立 oracle 和稳定 replay 完成 R01-R04、R13-R15；
5. CUDA 首次 submit 允许固定上游建立一次 graph metadata，资源稳定性从第一轮 warm replay 到第二轮
   replay 比较；只运行 `compute_provider_ggml_cuda`、`compute_route_cuda`、`graph_provider_ggml_cuda`、
   `graph_route_cuda` 与 selector 选择的公共检查。

不修改公共 SDPA/Unary schema、ggml 源码、算法、FlashInfer kernel 或任何 HYLE 数学；若固定 ggml CUDA
实际执行 24/4 仍失败，则记录精确 `BACKEND_REJECTED`，不在 D5 实现替代 attention。

**D5-F13 CUDA CANDIDATE（`80bbbba06`）**：固定 ggml CUDA adapter 已把既有 attention program
的资格域从固定 8/2 放宽到正整数倍 GQA，并把公共 Sigmoid 映射到上游 `ggml_sigmoid`；FlashInfer
仍只接管其已资格化的 8/2×256 域。Qwen3.8-27B 官方 H=5120、FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、MTP 10240 在普通 CUDA Direct/Graph 均以变化输入通过，首次 submit 后 replay
资源稳定，invalid Gather 不发布结果；既有 R05 K16/V48×128 收据并入完整格。Release/WERROR
selector 的 docs、closure、configure、ggml CUDA Provider、CUDA Direct、ggml CUDA Graph Provider 与
CUDA Graph Route 七个 gate 全部 PASS，125 个无关 gate 跳过或复用；未启动 P1。完整收据见
[Qwen3.8 official CUDA complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen38-official-complete/README.md)。
ledger 现有 28 个 `RUN_PASS`、148 个 `VALIDATION_NOT_RUN`；Apple、Vulkan 四格以及 D5-A5 三交付和
D5-A6 P1 尚未由本批代签。

### D5-F14：Qwen3.8-2.4T 官方 BF16 grouped expert 准入

固定 `Qwen/Qwen3.8-2.4T-A95B@207bd685` 官方 profile 的 1,609 个 tensor 全部为 BF16；
92 层主干中的 routed expert 按层堆叠，配置为 H=8192、512 experts、top-10、expert/shared
intermediate=2048。现有公共 `GroupedMatMul` 却把 dense expert weight 写死为 F32，四个固定 ggml
adapter 也重复这一限制，因此 R11 在进入成熟 Provider 前即被拒绝。固定 ggml `ggml_mul_mat_id`
本身创建 F32 输出，upstream backend test 明确登记 BF16 weight + F32 activation，CPU 使用 BF16
type trait 的 F32 dot 域，CUDA/Metal/Vulkan 的现有 backend admission/dispatch 均包含 BF16 weight。
完整源码边界见 [D5-F14 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-grouped-admission-audit/README.md)。

因此 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal` 与
`compute_provider_ggml_vulkan` 从各自 D5 candidate 精确退回本卡 AUDITED；`compute_ops` 已由 D5
持有 AUDITED。有限 FIX 只有：

1. 公共 `GroupedMatMul` dense schema 接纳 F32 或 plain BF16 weight；activation/output 保持 F32、ids
   保持 I32、accumulation 保持显式 F32，输出形状和现有 GGUF packed weight 的 F32 logical type 合同
   不变；
2. 四个固定 ggml adapter 只放宽 plain dense BF16 weight 的相同域，继续直接 lower 到
   `ggml_mul_mat_id`；CPU、CUDA、Vulkan 解除重复的 F32 admission，Metal 补齐其 native dtype 到
   `GGML_TYPE_BF16` 的映射；不接纳 BF16 activation/output，不增加转换、fallback、热选择或 HYLE 数学；
3. 四个 Provider 回归用 BF16 expert weight、F32 activation/output、I32 ids、变化输入 replay 和独立
   double oracle，保留既有 F32/量化 grouped expert 回归；
4. Qwen3.8-2.4T 官方 Direct/Graph batch 保留 H=8192、expert intermediate=2048、512-way router 与
   top-10 语义；允许按 D5 容量合同只实例化被选 active expert 子集，但不得缩小内宽、dtype、state
   或语义端口；
5. 只运行 `compute_ops`、四个 ggml Provider owner suite、对应 Direct/Graph route factor 与 selector
   选择的公共检查；API/算法和路线批次稳定后才进入 D5-A5/A6，不在本修复中提前启动 P1。

不修改 ggml 源码、Provider 顺序、路由键、量化格式、资源模型或任何 HYLE kernel/shader。若某后端
实际拒绝 BF16 `MUL_MAT_ID`，该路线记录精确 `BACKEND_REJECTED` 并停止，不以 F32 常驻复制伪造覆盖。

**D5-F14 CANDIDATE（`d707af1e562e5fd4c183d187238a58bb8eab6641`）**：公共 schema 和四个
固定 ggml adapter 已接纳 plain BF16 expert weight，并继续以 F32 activation/output/accumulation、
I32 ids 直接执行上游 `ggml_mul_mat_id`。Windows CPU/CUDA/Vulkan 与正式 MacBook Air M4 的
Provider、普通 Direct 和 Graph 定向用例均以变化输入通过；完整受影响套件分别通过 CPU 49/49、
64/64，CUDA 48/48、65/65，Vulkan 162/162、73/73，Apple 78/78、51/51。Metal 验收首次发现其
dtype 映射遗漏，最小补齐后实际编译运行 `kernel_mul_mv_id_bf16_f32_4`；没有转换、fallback、热选择
或 HYLE 数学。完整 partial 收据见
[Qwen3.8-2.4T BF16 grouped expert](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-grouped/README.md)。
本收据只签 R11；公共 API selector 选择的 Windows Compute/Graph 源码、共享、静态及搬迁 CPU consumer
均已通过。R01-R05、R10、R13-R15、D5-A5 各路线 profile 三交付与 P1 尚未完成，因此 ledger 仍为
48 个 `RUN_PASS`、128 个 `VALIDATION_NOT_RUN`。

### D5-F15：Qwen3.8-2.4T 官方 BF16 CPU 片段批量验收

tests-only candidate `041d9e63a` 把已验 Qwen3.8 CPU Direct/Graph 参数化 consumer 复用到官方
2.4T 的 H=8192、shared expert N=2048、GQA 64/4×256、GDN K16/V128×128 和 MTP 16384 输入。
普通入口分别运行 embedding/投影/head、shared FFN、RoPE/GQA/output gate、MTP 和 GDN 状态更新，
两次变化输入复用准备/编译结果并对照独立 oracle。现有 R11 BF16 grouped expert 收据继续独立有效。
Windows CPU Direct 全套 54/54、Graph 全套 69/69，无跳过；证据见
[Qwen3.8-2.4T CPU batch](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-cpu-batch/README.md)。
R10 路由/派发/合并和 R05 的完整模型投影/存储边界仍待验；本 partial 不改变 ledger 48/176，
不代签各路线三交付或 P1。

### D5-F16：Windows CUDA 运行库版本使 CCCL Provider 不可达

**SCOPED（baseline `0d25f9938`）**：在官方 Qwen3.8-2.4T CUDA 批次中，R01 的 BF16 Gather
准备失败；相同的 Qwen3.8-27B 既有 R01 用例在当前机器也重现。源码定位到
`compute_provider_cuda/collectives_provider.cc` 的 CCCL 资格规则：SM 8.9、编译 Toolkit 13030、
`CCCL-300304` 都匹配，但运行库要求精确 13030。当前 `v13.3` SDK 的 `CUDART_VERSION=13030`、
`CUB_VERSION=300304`，其 Windows 运行库和驱动经 API 均报告 13040；因此 CCCL 的 Gather、TopK、
Scan 在匹配阶段整体不可达，最后的 CUDA tensor Provider 只返回“未实现该算子族”。
[NVIDIA CUDA Runtime 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html)
明确说明 CUDA 13 起 Windows 使用随显示驱动提供的运行库，其版本可能与编译 Toolkit 不同；
[CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
允许 CUDA 13.x 内的兼容组合，但具体 Provider 能力仍须本机验证。

本次只重开 `compute_provider_cuda` 的 CCCL 环境资格；不修改 TopK/Gather/Scan 数学、CCCL 调用、
其它 CUDA Provider 或优先级。D4 CLOSED 的算子清退结论仍有效。先完成 AUDITED 有限 FIX 定界，
再改产品代码；现有 CUDA 测试候选的 R01 失败保持原样作为触发证据。

**AUDITED 有限 FIX**：

1. 仅将 CCCL 的 `kRuntimeVersion` 从精确 13030 改为闭区间 13030..13040；SM 8.9、编译 Toolkit
   13030、`CCCL-300304` 仍精确匹配。该区间只覆盖已验旧机运行库和本次实际 13.4 运行库；不宽放
   CUDA 14、其它架构/Toolkits 或其它 Provider。
2. 在 RTX 4070 当前 13040 环境重新运行 CCCL Gather/TopK/Scan owner suite 与 Qwen3.8 27B/2.4T
   普通 CUDA Direct/Graph；对 2.4T 保留 H=8192、K16/V128、top-10、两次变化输入和独立 oracle。
3. 只运行 selector 指向的 Provider/Direct/Graph factor、输入工具与公共检查；本规则没有更改数学或
   公开 API，本 partial 不启动 D5-A5/A6。若 13.4 库实际执行不合格，则撤销扩域并记录精确拒绝。

**CANDIDATE（`05b374e61`）**：有限 FIX 已实现，RTX 4070 / SM 8.9 上 Toolkit 13.3 编译产物
与驱动提供的 13.4 runtime 组合实际运行通过。CCCL owner 定向 10/10、全套 42/42；CUDA Direct
全套 53/53、Graph 全套 70/70，均无跳过。Qwen3.8-2.4T 官方 BF16 的 Direct/Graph 各 5/5
片段通过；既有 27B Gather 也恢复。证据见
[Qwen3.8-2.4T CUDA batch](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-cuda-batch/README.md)。
这仍是局部片段证据：R10 与 R05 完整链、各路线 profile 交付和 P1 未签，ledger 维持 48/176。

### D5-F17：Qwen3.8-2.4T 官方 BF16 Apple/Vulkan 形状复用

tests-only candidate `10c878305` 将既有 Qwen3.8 Apple/Vulkan Direct/Graph profile consumer
参数化，加入官方 2.4T 的 H=8192、shared expert N=2048、GQA 64/4×256、MTP 16384 和
GDN K16/V128×128；仍复用同一普通入口、成熟 Provider 与独立 oracle。各路线此前的 R11 grouped
expert 收据独立有效。Vulkan RTX 4070 的新形状 Direct/Graph 各 5/5，完整 factor 167/167、
78/78；MacBook Air M4 的新形状 Direct/Graph 各 5/5，完整 factor 83/83、56/56，均无跳过。
Windows selector 选择 Vulkan 两个 factor；Apple 验证副本的 selector 选择 Apple 和 Vulkan factor，
由两台适用平台分别执行。证据见
[Qwen3.8-2.4T Apple/Vulkan batch](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-apple-vulkan-batch/README.md)。
R10 router/dispatch/combine 与 R05 完整模型投影/存储链未由这批片段覆盖，ledger 仍为 48/176；
不代签 D5-A5 三交付或 D5-A6 P1。

### D5-F18：Qwen3.8-2.4T R10 router 的成熟 Softmax 接入

**AUDITED（baseline `dbf492934`，owner `compute_provider_ggml`）**：固定 llama.cpp
`4c9233c` 的 `qwen35moe.cpp` 调用 `build_moe_ffn`，其实现对 512 个 router logits 先做
Softmax，再稳定取 top-10、抽取选中权重并重新归一化；之后按选中专家执行乘法、加权合并，shared
expert 另有 sigmoid gate。当前 HYLE ggml CPU Provider 已有这些基础图节点中的 MatMul、Cast、
TopK、Reduce 与 Binary，但 F32 router Softmax 没有 admission/translation，普通 CPU 入口选到冻结的
`hyle-native`。新增的真实 H=8192、E=512/top-10 Direct consumer 在成熟 Provider 断言处定向失败，
其余数值断言通过；这不能作为 R10 PASS。

有限 FIX 只在 `compute_provider_ggml` 接入固定 ggml 原版 `ggml_soft_max`：F32 plain/dense rank-2、
最后轴、无 mask、scale=1、标准 Softmax、F32 accumulation；不扩宽其它 dtype/轴/mask，不改路由
优先级、不加 HYLE 算法。用两次变化输入和稳定 tie 检查 512→10 的概率、ID 与归一化；随后接到
已有 BF16 grouped expert 的真实 selected IDs 和权重，验证 Direct/Graph 与四路线。适用 selector
factor、公共检查和 P0 成本结构先验收；本阶段不代签 D5-A5/P1，也不强迫插入上游 ggml MoE 路径没有的
Scan/Scatter 节点。

**CANDIDATE（`66f4d80fb`，R10 router partial）**：固定 ggml CPU `ggml_soft_max` 已按有限 FIX
接入；H=8192 BF16 router projection 产生 512 个 score，经显式 F32 Cast、Softmax、稳定 TopK、
Reduce/Divide 形成 10 个 ID 和重新归一化的权重。普通 CPU Direct 与 Graph 用两次变化输入和一次
并列 tie 各通过 1/1。`oneDNN=ON` 的正式受影响完整套件为 Provider 40/40、Direct 55/55、
Graph 70/70。`oneDNN=OFF` 的 Graph 全套 64/65，失败的旧 Qwen3.5 用例明确断言必须出现 oneDNN
region，故不能把该配置误记 PASS；它不是本次 Softmax 回归。完整局部证据见
[R10 CPU router](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-r10-router-cpu/README.md)。
该 partial 没有把 top-10 ID/权重接到 grouped expert、shared gate 和最终 combine，也未裁决 BF16
router MatMul 存储边界与固定 ggml 图的 F32 score 边界是否等价；不签 R10 或完整 profile cell。

**D5-F18 后续局部收据（`34a13aab`、`0c97b284`）**：十个 resident expert 的测试权重已
区分；CPU Graph 用同一 BF16 token 完成 512-way router、稳定 top-10、成熟 Gather 复制激活、动态
ID 驱动 BF16 grouped expert、归一化权重乘法和十路合并。两次改变顺序的 replay 对照独立 oracle，
Graph 定向 1/1、完整 factor 70/70。夹具变更在 CPU/Apple 完整 Direct/Graph 和 CUDA/Vulkan 定向
Direct/Graph 均通过。详见 [R10 CPU Graph dispatch partial](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-r10-graph-dispatch-cpu/README.md)。
这仍未接共享专家分支及最终相加，Direct 与其它三路线尚未接动态路由。固定 llama.cpp
`build_lora_mm` 的 ggml MatMul 产出 F32 logits，而现有公共 BF16 MatMul 合同先存 BF16 再 Cast F32；
需先定界这个精度差异，不能以当前 synthetic logits 正确就称模型等价。ledger 仍为 48/176。

### D5-F19：官方 BF16 router 的 F32 score 存储边界

**AUDITED（baseline `f32c85ef3`，owners `compute_ops` 与四个固定 ggml Provider）**：
固定 llama.cpp `4c9233c` 的 `build_moe_ffn` 通过 `build_lora_mm(gate_inp, cur)` 产生 router
logits，所调用的 `ggml_mul_mat` 在固定 ggml 源码中明确创建 `GGML_TYPE_F32` 输出。现有 HYLE
公共 `MatMul` 要求两输入同 dtype 且输出也同 dtype；四路线官方 BF16 consumer 因此先把 score 写入
BF16，再 Cast F32。512 路 top-10 的第 10/11 名可因这一额外 BF16 舍入变成 tie，改变专家成员；
目前的单对角 synthetic router 没有暴露这个风险。此为模型语义缺口，不以 P1 或性能猜测驱动 API。

有限 FIX 固定如下：

1. 在公共 model operation 中增加明确的 rank-2 `BF16 Linear -> F32` 语义，输入分别为
   activation `[T,K]` 和模型原始输出优先存储的 weight `[E,K]`，F32 输出 `[T,E]`；不改通用
   `MatMul` 的同 dtype 合同，不隐式转换已有图；
2. CPU/CUDA/Metal/Vulkan 四个固定 ggml Provider 只在该精确域接纳并直译原版
   `ggml_mul_mat` 的 F32 输出。Vulkan 的一行 BF16 小 matvec 复用已审定的上游管线限制处理：
   只将该激活在设备图内提升 F32，权重仍 BF16；验收含此必要成本。现有 GGUF、oneDNN、MLX、
   cuBLAS 等 Provider 顺序及准入不变；不增加 HYLE 数学、F32 常驻权重副本或设备往返；
3. 先以第 10/11 名接近的 BF16 输入制造 BF16 中间存储会改变 top-10 的红灯测试，再对四路线
   Direct/Graph 验证 F32 score、ID、归一化权重与动态 expert replay；失败时只记录精确 Provider
   边界，不能用 host 排序或修改测试 oracle 代替；
4. 运行 selector 选择的公共语义、Provider、Direct/Graph factor、三交付风险和公共检查；P0 审查
   转换/同步/常驻资源，稳定不可变候选的适用 P1 留在对应路线 profile 验收，不在本精度 FIX 中
   提前多次计时。

本 FIX 不签共享 expert、完整 R05、其它模型/量化画像或完整 profile cell；ledger 保持 48/176。

**路线接缝重定界（baseline `f595c7f51`）**：`BFloatLinearF32` 的固定 ggml Vulkan
Direct 定向实际执行已通过，但 Graph 编译返回 `Vulkan custom schema needs a cold native
lowering`，在到达同一 Provider 前被成熟 custom 白名单拒绝。Apple 的 Direct 与 Graph 路线同样
以固定 `fixed_ggml_model_operation` 列表划分 ggml region。新模型 consumer 是 F19 原生 F32
router；这构成 `graph_route_vulkan` 的新事实 reopen trigger（当前 SCOPED），并要求
`compute_route_apple`、`graph_route_apple` 从既有 D5 候选重定界为 AUDITED。既有 D4 删除裁决
和 F8 候选保留为历史证据。Apple 两 owner 的有限 FIX 仅是各自在冷期的
`fixed_ggml_model_operation` 分类加入此 custom schema，交由已审定 ggml Metal Provider 执行；
Direct/Graph 各以近并列 F32 score 和变化输入验证。
本轮只接通该 operation 至已 AUDITED 的固定 ggml Provider，不改变 Graph 调度、MLX region、
CUDA 路由、资源/完成语义、旧 GGUF 路由或自研数学。

**Vulkan Graph AUDITED（baseline `a6c79badb`）**：失败点精确在
`graph_route_vulkan/vulkan_lowering.cc` 的 `mature_custom` 分类，不在固定 ggml Vulkan
Provider、Vulkan shader 或用户自定义 lowering。有限 FIX 只加入 `BFloatLinearF32` 这一已由
Direct 实际执行的 schema；以普通 Graph 准备、两次近并列变化输入、F32 score、Provider region
report 与完整受影响 factor 验证。不修改其他 custom 准入或新增后备执行器。

**F19 部分候选（`774c3a32d`）**：公共 BF16 `[T,K]` × 原生输出优先 BF16 `[E,K]`
已接到四路线固定 ggml Provider 的 F32 score 边界。第 10/11 名近并列红灯先证明旧 BF16
中间存储会改变入选专家；新路径的四路线 Direct/Graph 单算子定向结果通过，MacBook Air 四个
相关完整 suite 也通过。Apple/Vulkan 的 Graph factor 辅助函数实际以 GraphBuilder 建单节点图并
调用路线 Graph compile；尚未签多节点 R10 整链。CPU Graph 的动态专家与加权合并仍为 partial，
其他路线
尚未连接该完整链。Windows selector 的 27 个受影响 gate 尚未全部完成：修正 Vulkan SDK 后的
全架构 ggml CUDA 构建已停止，具名中断 lease 阻止旧候选继续取得 selector verdict。
[F32 score partial receipt](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-router-f32-score/README.md)
记录实际测试和未签边界；三交付与 P1 不写 PASS，ledger 维持 48/176。

### D5-F20：CUDA Graph workspace 别名根对齐

**SCOPED（baseline `6f954ef5`，owner `graph`）**：本轮完整 Qwen3.8-2.4T
BF16 MoE CUDA Graph 首次将 `Cast` 的内部 F32 输出经零偏移 `View` 交给固定 ggml CUDA region。
Region 的冷期 `ResourceRequirement.minimum_alignment` 已声明 256 字节；Graph 的内部生命周期
规划却只按 tensor 基本对齐和 64 字节默认值分配根，未把下游 port 要求传到别名根。
MSVC Release/WERROR 编译通过，普通 Graph submit 的完成状态实际报
`CUDA buffer offset does not satisfy the requested type alignment`，属于新的可复现重开事实。
同批 CPU、Vulkan Graph 整链定向已通过；CUDA 失败不能记作模型 cell PASS。

本缺陷只涉及 Graph 冷期的 workspace/alignment 规划，不修改 ggml CUDA 的既有
256 字节合同、CUDA 算子、Provider 次序或公共路由。AUDITED 前核对具体失败值、
端口 proposal 与别名偏移；有限 FIX 应使选中 Provider 的每个输入/输出最小对齐传到物理
实例及零偏移别名根，对非零偏移不能满足的组合在准备期拒绝。验收只选受影响 Graph
编译/执行、别名和四路线原有 owner 回归，候选稳定后按 selector 签署；不重跑无关 P1。

**AUDITED（baseline `49c6c63b`）**：固定 ggml CUDA region 的
`propose_resources` 对需要向量化的边界端口已提交 256 字节要求，Graph 的
`discovery.requirements` 在冷期持有该值；`prepare_placed_steps_attempt` 建物理实例时没有保存
这个要求，`plan_lifetimes` 建根请求时仅取 `max(64, TensorDesc.required_alignment())`。
因此缺陷在 Graph 消费已声明的端口约束，不在 Provider 提案或设备枚举。有限 FIX 固定为：
保存被选中 operation/region 的输入输出端口最小对齐；把零偏移 View/alias 的要求向根传播；
工作区请求使用传播值，非零偏移冲突在准备期返回错误。只改 Graph 冷期规划，不增加
新 Provider、数学、运行时搜索或公共治理逻辑。定向重放 CUDA 三组变化输入、近并列成员及
shared/final oracle，随后运行 Graph CPU/CUDA/Vulkan 受影响 owner 和 Mac Apple 回归；
完整 selector、三交付及适用 P1 仍在稳定批次签署。

**Graph 批次部分收据（candidate `825c41f8d`）**：CPU、CUDA、Vulkan 与
MacBook Air Apple 的完整多节点 R10/R11 Graph 均以三组变化输入通过独立 ID、routed、
shared、combine 和 final oracle；四路线完整 Graph owner suite 及适用 Graph 核心回归通过。
Graph 对齐 FIX 已在 CUDA 真实执行中验证。详见
[完整 MoE Graph 批次](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-full-router-graph/README.md)。
这是 Graph 子批结果，Direct 整链、完整 selector、三交付和 P1 尚未签署；ledger 保持 48/176。

### Qwen3.8 Apple official Direct/Graph 完整格

不可变 tests-only candidate `6086dd721af664a1d759bb4eea3dfd645e02c99f` 在正式 MacBook Air M4
上以普通 Apple `Context::prepare` 和 Graph build/compile/bind/submit 覆盖官方 H=5120、
FFN=17408、GQA 24/4×256、partial interleaved RoPE 64 与 MTP 10240。每个 prepared artifact
使用变化输入重放，invalid Gather 不改写已发布结果；既有完整 ggml Metal R05 K16/V48×128 收据并入
两格。现有 MLX Metal 与固定 ggml Metal 直接接受这些形状，没有修改产品实现、Provider 顺序、公共
schema 或新增 HYLE 数学。

Release/WERROR selector 的 docs、closure、configure、Apple Direct 和 Apple Graph 五个 gate 全部
PASS，127 个无关 gate 跳过；完整 Direct factor 54.37 秒、Graph factor 13.05 秒。完整收据见
[Qwen3.8 official Apple complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen38-official-complete/README.md)。
ledger 现有 30 个 `RUN_PASS`、146 个 `VALIDATION_NOT_RUN`；Vulkan 两格以及 D5-A5 三交付和 D5-A6
P1 尚未由本批代签。

### Qwen3.8 Vulkan official Direct/Graph 完整格

不可变 tests-only candidate `5fc3c2f418795e6cd89b86e3f546d5775b96758a` 在 Windows RTX 4070
上以普通 Vulkan prepare/bind/submit 和 Graph build/compile/bind/submit 覆盖官方 H=5120、
FFN=17408、GQA 24/4×256、partial interleaved RoPE 64 与 MTP 10240。每个 prepared artifact
使用变化输入重放，invalid Gather 不改写已发布结果；既有完整 ggml Vulkan R05 K16/V48×128 收据
并入两格。现有固定 ggml Vulkan managed region 与限定 IREE dispatch 直接接受这些形状，没有修改
产品实现、Provider 顺序、公共 schema 或新增 HYLE 数学。

Release/WERROR selector 的 docs、closure、configure、Vulkan Direct 和 Vulkan Graph 五个 gate 全部
PASS，127 个无关 gate 跳过；完整 Direct factor 31.98 秒、Graph factor 16.73 秒。首两次 configure
分别因未绑定 Vulkan SDK，以及 CMD 提前展开 PATH 后丢失 Windows SDK `rc/mt` 而在产品编译前停止；
最终收据显式绑定 Vulkan SDK 1.4.357.0、`rc`、`mt` 与 `glslc`。完整收据见
[Qwen3.8 official Vulkan complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen38-official-complete/README.md)。
ledger 现有 32 个 `RUN_PASS`、144 个 `VALIDATION_NOT_RUN`；D5-A5 三交付和 D5-A6 P1 仍由稳定路线
批次统一执行。

### Qwen3.8 primary GGUF CPU Direct/Graph 完整格

不可变 tests-only candidate `051b3e061627661ac020e7a6217154677e69ddae` 已在普通 CPU
Direct/Graph 入口完成固定 `qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的 R01-R05、R13-R15。
R01 保留 Q4_K embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact 实际出现的 Q3_K、
IQ3_S、IQ4_NL、IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的 Q6_K
`10240 -> 5120` 权重。语义链保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、
Gated DeltaNet K16/V48×128 及 MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding
不发布结果。所有调用与 Graph report 只绑定固定 ggml CPU 或 oneDNN，没有增加 HYLE 数学、fallback、
热选择或产品实现。

Release/WERROR selector 的 docs、closure、模型输入工具、configure、CPU Direct 与 CPU Graph 六个 gate
全部 PASS，127 个无关 gate 跳过或复用；完整 Direct owner 63.21 秒，Graph 与 executor 63.67 秒。
完整收据见
[Qwen3.8 primary GGUF CPU complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen38-primary-gguf-complete/README.md)。
ledger 现有 34 个 `RUN_PASS`、142 个 `VALIDATION_NOT_RUN`；其余六条 Qwen3.8 GGUF 路线格以及
D5-A5 三交付和 D5-A6 P1 尚未由本批代签。

### Qwen3.8 primary GGUF CUDA Direct/Graph 完整格

不可变 tests-only candidate `02b58c8df5b3e281df8d6aea967c47d4eeea08c0` 已在普通 CUDA Direct/Graph 入口完成固定
`qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的 R01-R05、R13-R15。R01 保留 Q4_K
embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact 实际出现的 Q3_K、IQ3_S、IQ4_NL、
IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的 Q6_K `10240 -> 5120` 权重。语义链
保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、Gated DeltaNet K16/V48×128 及
MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding 不发布结果。所有调用与 Graph
report 只绑定固定 ggml CUDA、cuBLASLt、SGLang/CCCL 或 FlashInfer，没有增加 HYLE 数学、fallback、
热选择或产品实现。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、CUDA Direct 与 CUDA Graph 五个 gate
全部 PASS，127 个无关 gate 跳过或复用；CUDA Toolkit 13.3.73、`CUDAARCHS=89-real`，完整 Direct
owner 63.09 秒、Graph owner 64.50 秒。完整收据见
[Qwen3.8 primary GGUF CUDA complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen38-primary-gguf-complete/README.md)。ledger 现有
36 个 `RUN_PASS`、140 个 `VALIDATION_NOT_RUN`；其余四条 Qwen3.8 GGUF Apple/Vulkan 路线格以及
D5-A5 三交付和 D5-A6 P1 尚未由本批代签。

### Qwen3.8 primary GGUF Apple Direct/Graph 完整格

不可变 tests-only candidate `dfa1423767d375154aa75cc717d327d7073d331b` 已在正式 MacBook Air M4
的普通 Apple Direct/Graph 入口完成固定 `qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的
R01-R05、R13-R15。R01 保留 Q4_K embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact
实际出现的 Q3_K、IQ3_S、IQ4_NL、IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的
Q6_K `10240 -> 5120` 权重。语义链保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、
Gated DeltaNet K16/V48×128 及 MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding
不发布结果。量化调用固定绑定 ggml Metal，其余语义片段固定绑定 MLX Metal；没有增加 HYLE 数学、
fallback、热选择或产品实现。

Q4_K `get_rows` 的独立 F32 oracle 首轮暴露了成熟 Provider 的数值边界。源码确认 ggml Metal
`dequantize_q4_K` 在高半字节分支以 half 精度执行 `xb->d / 16.h` 后发布 F32；实测最大绝对差
`4.506111145019531e-4`。仅该转换按 `5e-4` 验收，其它 embedding 格式仍保持 `1e-6`。
AppleClang 21 Release/WERROR selector 的 docs、closure、configure、Apple Direct 与 Apple Graph 五个
gate 全部 PASS，127 个无关 gate 跳过或复用；完整 Direct owner 80.02 秒、Graph owner 40.11 秒。
完整收据见
[Qwen3.8 primary GGUF Apple complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen38-primary-gguf-complete/README.md)。
ledger 现有 38 个 `RUN_PASS`、138 个 `VALIDATION_NOT_RUN`；剩余 Qwen3.8 GGUF Vulkan 两格以及
D5-A5 三交付和 D5-A6 P1 尚未由本批代签。

### Qwen3.8 primary GGUF Vulkan Direct/Graph 完整格

不可变 tests-only candidate `af6a99c450996250a3f557f1eda832ca91c7ef41` 已在正式 Windows
RTX 4070 的普通 Vulkan Direct/Graph 入口完成固定
`qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的 R01-R05、R13-R15。R01 保留 Q4_K
embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact 实际出现的 Q3_K、IQ3_S、IQ4_NL、
IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的 Q6_K `10240 -> 5120` 权重。语义链
保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、Gated DeltaNet K16/V48×128 及
MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding 不发布结果。所有调用与 Graph
report 只绑定固定 ggml Vulkan managed region 或限定 IREE Vulkan program，没有增加 HYLE 数学、
fallback、热选择或产品实现。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、Vulkan Direct 与 Vulkan Graph 五个
gate 全部 PASS，127 个无关 gate 跳过或复用；Vulkan SDK 1.4.357.0、NVIDIA driver 610.62，完整
Direct owner 86.70 秒、Graph owner 70.37 秒。完整收据见
[Qwen3.8 primary GGUF Vulkan complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen38-primary-gguf-complete/README.md)。
ledger 现有 40 个 `RUN_PASS`、136 个 `VALIDATION_NOT_RUN`；Qwen3.8-27B primary GGUF 已完成四路线
八格，D5-A5 三交付和 D5-A6 P1 仍由稳定路线批次统一执行。

### Qwen3.8 fidelity GGUF CPU Direct/Graph 完整格

不可变 tests-only candidate `5bbecf49865ffbe638d9639ac0fb051ece539e8c` 已通过普通 CPU
Direct/Graph 入口完成固定 `qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的
R01-R05、R13-R15。固定 header 为 1 shard、866 tensors、27,320,697,856 logical parameters；
embedding、dense FFN、GDN projection 与 attention output 为 Q8_0，full-attention Q/K/V、LM head
与 MTP projection 为 BF16，norm、conv 与 recurrent state 为 F32。测试没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention output、
SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact 使用变化输入
重放，越界 embedding 不发布结果；所有 Direct/Graph report 只接受固定 ggml CPU 或 oneDNN 成熟实现，
没有新增 HYLE 数学、fallback、热选择或产品代码。

MSVC 19.50 Release/WERROR selector 的 docs、closure、模型输入工具、configure、CPU Direct 与 CPU
Graph 六个 gate 全部 PASS，127 个无关 gate 跳过或复用；完整 Direct owner 66.90 秒，Graph 与 executor
67.12 秒。完整收据见
[Qwen3.8 fidelity GGUF CPU complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 42 个 `RUN_PASS`、134 个 `VALIDATION_NOT_RUN`；其余 CUDA、Apple、Vulkan 六格以及
D5-A5 三交付和 D5-A6 P1 仍待对应路线批次完成。

### Qwen3.8 fidelity GGUF CUDA Direct/Graph 完整格

不可变 tests-only candidate `e1ecacc3bd7c27c461b8cc46665180aeeacfd584` 已通过普通 CUDA
Direct/Graph 入口完成固定 `qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的
R01-R05、R13-R15。固定 header 为 1 shard、866 tensors、27,320,697,856 logical parameters；
embedding、dense FFN、GDN projection 与 attention output 为 Q8_0，full-attention Q/K/V、LM head
与 MTP projection 为 BF16，norm、conv 与 recurrent state 为 F32；测试按真实 tensor role 路由，
没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention output、
SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact 使用变化输入
重放，越界 embedding 不发布结果；所有 Direct/Graph report 只接受固定 ggml CUDA、cuBLASLt、
SGLang/CCCL 或 FlashInfer 成熟实现，没有新增 HYLE 数学、fallback、热选择或产品代码。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、CUDA Direct 与 CUDA Graph 五个 gate
全部 PASS，127 个无关 gate 跳过或复用；CUDA Toolkit 13.3.73、`CUDAARCHS=89-real`，完整 Direct
owner 66.53 秒、Graph owner 68.66 秒。完整收据见
[Qwen3.8 fidelity GGUF CUDA complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 44 个 `RUN_PASS`、132 个 `VALIDATION_NOT_RUN`；其余 Apple、Vulkan 四格以及 D5-A5
三交付和 D5-A6 P1 仍待对应路线批次完成。

### Qwen3.8 fidelity GGUF Apple Direct/Graph 完整格

不可变 tests-only candidate `1544c98061f36a8a7e5ea67b5db00a5f47bd8872` 已在正式 MacBook Air M4
的普通 Apple Direct/Graph 入口完成固定 `qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的
R01-R05、R13-R15。固定 header 为 1 shard、866 tensors、27,320,697,856 logical parameters；
embedding、dense FFN、GDN projection 与 attention output 为 Q8_0，full-attention Q/K/V、LM head
与 MTP projection 为 BF16，norm、conv 与 recurrent state 为 F32；测试按真实 tensor role 路由，
没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention output、
SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact 使用变化输入
重放，越界 embedding 不发布结果；量化调用固定绑定 ggml Metal，其余语义片段固定绑定 MLX Metal，
没有新增 HYLE 数学、fallback、热选择或产品代码。

AppleClang 21 Release/WERROR selector 的 docs、closure、configure、Apple Direct 与 Apple Graph 五个 gate
全部 PASS，127 个无关 gate 跳过或复用；MLX 0.32.2、ggml Metal 0.24.0、macOS SDK 27.0，完整
Direct owner 39.91 秒、Graph owner 37.79 秒。完整收据见
[Qwen3.8 fidelity GGUF Apple complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 46 个 `RUN_PASS`、130 个 `VALIDATION_NOT_RUN`；剩余 Vulkan 两格以及 D5-A5 三交付和
D5-A6 P1 仍待对应路线批次完成。

### Qwen3.8 fidelity GGUF Vulkan Direct/Graph 完整格

不可变 tests-only candidate `a11d9667683795950960323627532629e2d7726e` 已在正式 Windows
RTX 4070 的普通 Vulkan Direct/Graph 入口完成固定
`qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的 R01-R05、R13-R15。固定 header 为 1 shard、
866 tensors、27,320,697,856 logical parameters；embedding、dense FFN、GDN projection 与
attention output 为 Q8_0，full-attention Q/K/V、LM head 与 MTP projection 为 BF16，norm、conv
与 recurrent state 为 F32；测试按真实 tensor role 路由，没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention
output、SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、
partial interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact
使用变化输入重放，越界 embedding 不发布结果；量化调用固定绑定 ggml Vulkan，其余语义片段固定
绑定成熟 ggml Vulkan 或已准入 IREE Vulkan 程序，没有新增 HYLE 数学、fallback、热选择或产品代码。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、Vulkan Direct 与 Vulkan Graph
五个 gate 全部 PASS，127 个无关 gate 跳过或复用；Vulkan SDK 1.4.357.0、NVIDIA driver 610.62，
完整 Direct owner 90.31 秒、Graph owner 75.23 秒。完整收据见
[Qwen3.8 fidelity GGUF Vulkan complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 48 个 `RUN_PASS`、128 个 `VALIDATION_NOT_RUN`；该 profile 四路线八格已全部闭环；
D5-A5 三交付和 D5-A6 P1 仍随稳定路线批次执行。

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
