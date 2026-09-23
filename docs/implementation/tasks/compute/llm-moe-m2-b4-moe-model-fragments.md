# M2-B4 成熟 Provider MoE 与模型片段批次

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- round / owner：M2-B4；`/root`。
- baseline：`d12193cab61d67eb8318d14d08f2e85f83275fe4`；B3 已关闭，Windows canonical
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
