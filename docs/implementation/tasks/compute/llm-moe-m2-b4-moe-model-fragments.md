# M2-B4 成熟 Provider MoE 与模型片段批次

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- round / owner：M2-B4；`/root`。
- baseline：`d12193cab61d67eb8318d14d08f2e85f83275fe4`；[B3 已关闭](llm-moe-m2-b3-recurrent-state.md)，Windows canonical
  工作树在本卡登记前干净。
- trigger：F12.1--F14.5 的 MoE routing→dispatch→expert→combine、shared/latent/SiTU 与
  DeepSeek V4 mHC/MTP 真实消费。B1/B2 已提供 score、TopK、Gather/Scatter、MatMul 与常用组合，
  但公共层尚不能表达 grouped expert MatMul 和 mHC 三个原版 region，四路线也未接锁定 ggml 的
  `mul_mat_id`、`dsv4_hc_comb/pre/post`。
- stop：把所有已准入成熟 B4 入口按 Provider 能力族成批接入普通 Direct/Graph，或留下具名
  backend/数值反例；新增 HYLE 数学 kernel/shader/SIMD 算子为 0。服务端专家调度、通信、采样、
  speculative engine 与模型 loader 不进入本批。

## SCOPED owner 与有限边界

| owner | 本批允许审计/修改 | 本批禁止 |
|---|---|---|
| `compute_ops` | 在 `model_ops.h` 增加 grouped MatMul 与 mHC combine/pre/post 的 typed descriptor、稳定 identity 和自然 shape 推导 | 新算法、模型文件解析、把 engine routing 塞入 Operation |
| CPU | `compute_provider_ggml` 一次接 `ggml_mul_mat_id` 与三枚 DSV4 HC 原版节点；CPU/Graph route 复用 retained recipe | 按专家写 host loop、oneDNN 冒充 grouped expert |
| CUDA | `compute_provider_ggml_cuda` 复用同一 recipe 和 native stream/buffer；Graph 复用 leaf/合法 region | 新 `.cu`、CPU fallback、按运行 timing 选 Provider |
| Apple | 优先使用锁定 ggml Metal 原版 model region；若现有 Apple native region seam 不能有限泛化，则用 MLX public compiled composition并记录准确边界 | 新 Metal shader、拆成逐 token host 调度 |
| Vulkan | `compute_provider_ggml_vulkan` managed region 批量 lower 四枚节点，最终资格仍由原版 backend `supports_op` 裁决 | 新 SPIR-V、相似 shape 代签 |

共享路由 matcher、Mem、公共构建系统、性能仪器和冻结自研算子不在预期 FIX。现有 score/bias、
TopK、Gather/Scatter、activation、residual 与普通 MatMul 只作已有成熟组合消费，不重复接线或竞速。

## SCOPED 验收

1. Grouped MatMul 固定自然布局 `[E,N,K] × [T,S,K] × ids[T,S] → [T,S,N]`；ids 为 I32，
   空/偏斜专家、重复专家、越界失败和两次 replay 使用同一批表驱动输入。
2. mHC 固定 DeepSeek V4 `hc=4`：combine `[T,24]×[3]×[24]→[T,4,4]`，pre
   `[T,4,D]×[T,4]→[T,D]`，post `[T,D]×[T,4,D]×[T,4]×[T,4,4]→[T,4,D]`；
   epsilon/iteration 是 cold descriptor，热执行不搜索。
3. F12 routing 与 F13 dispatch/combine 复用既有成熟 TopK/Gather/Scatter/MatMul region；F14 的
   AttnRes/MTP 只验证可组合状态/索引边界，不引入推理引擎。
4. 四路线各一次增量构建和一组独立 oracle；P0 核对 packing、device residency、dispatch、同步、
   retained program 与重复执行。没有冻结完整 MoE fragment P1 card 时保持 `NOT_APPLICABLE`。
5. M2 最后统一执行 Compute/Graph source/static/shared 聚合交付；本批不为四枚 descriptor 复制
   九套构建。未运行门与真实模型链保持 `UNPROVEN`，归 M3-04。

完成上游入口和 backend 资格审计后，本卡须写入有限 `KEEP/FIX/BORROW/DEFER/REJECT` 表并转为
`AUDITED`，之后才能修改产品代码。

## AUDITED：原版入口与有限 FIX

2026-09-24 在 SCOPED baseline 上核对锁定 ggml `456172ec` 的 public schema、四个 backend 的
`supports_op` 与实际 dispatch。CPU、CUDA、Metal、Vulkan 都提供 `GGML_OP_MUL_MAT_ID` 和
`GGML_OP_DSV4_HC_COMB/PRE/POST`；四条路线保持各自 native buffer、stream/queue、graph 与完成
语义。Grouped MatMul 的第一批准入域固定为 dense F32，量化 weight 的准确 storage-format 映射留给
B6，不能仅凭位宽相似签署。

| ID | 有限施工 | 验收 |
|---|---|---|
| B4-C0 | 在 `model_ops.h` 增加 `GroupedMatMul` 与 mHC combine/pre/post 的稳定 typed descriptor；只表达自然布局、F32 域、固定 `hc=4` 与 combine cold 参数 | header self-contained、arity/type/shape、错误不发布 |
| B4-C1 | CPU ggml retained recipe 一次 lower 四枚原版节点，Graph 复用同一合法 leaf/region | Direct/Graph、偏斜/重复专家、两次 replay、独立 oracle |
| B4-C2 | CUDA ggml recipe 复用同一逻辑 lower 和 native stream/buffer；资格继续由原版 backend 裁决 | SM89 Direct/Graph、完成/失败、无 `.cu` 新增 |
| B4-C3 | 将现有 Apple ggml Metal whole-region seam 有限泛化到已登记 typed model op，lower 四枚原版节点；MLX 继续承担普通组合 | Air shared、Mini static、native Metal buffer/replay，无新 shader |
| B4-C4 | Vulkan managed region lower 四枚原版节点，Graph custom dispatch 识别这些 provider-owned schema | RTX 4070 与适用 MoltenVK，`ggml_backend_supports_op` 为最终资格 |
| B4-C5 | 一组 B4 fixture 同时覆盖 grouped expert 与 mHC 纵切；F12/F13 其余 routing/dispatch/combine 复用 B1/B2 的成熟组合；MTP 只验证可组合边界 | P0 resource/dispatch；M2 末尾统一三交付；P1 保持不适用 |

`KEEP`：B1/B2 已接入的 score、stable TopK、Gather/Scatter、activation、residual、MatMul 和组合图；
`BORROW`：以上四枚 ggml 原版 model op；`DEFER`：量化 grouped weights 的准确 format 映射到 B6，
完整模型层/多卡 dispatch 到 M3/M4；`REJECT`：host expert loop、自研数学、逐 token 同步及运行时竞速。

Apple 的改动只把既有 ggml Metal native-region owner 从单一 GDN 扩到这组已审计原版 op，不建立
万能执行器。组合路由仍在 prepare 阶段按平台、硬件、驱动/SDK、模型身份、路线、量化档位和
Provider 版本匹配，执行阶段不搜索或计时。有限 FIX 已冻结，可以进入产品施工。
