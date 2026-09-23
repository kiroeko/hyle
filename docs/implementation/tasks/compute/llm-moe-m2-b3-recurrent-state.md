# M2-B3 成熟 Provider 递归混合状态批次

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- round / owner：M2-B3；`/root`。
- baseline：`5d97716ed4b34e46491a99cb459327fab5580828`；B2 已关闭，Windows canonical
  工作树在本卡登记前干净。
- trigger：F09.1--F11.2 的 Qwen3.5、Kimi 线性注意力真实消费。公共 `GatedDeltaNet` 已能表达
  scalar gate，Apple/Vulkan 已接部分 F32 域；但 causal depthwise conv 没有自然公共语义，CPU/CUDA
  尚未把锁定 ggml 的 GDN/KDA 原版能力接入普通 Direct，KDA 的 per-channel gate 也不能通过当前推导。
- 目标纵切：`CausalConv1d → GatedDeltaNet`，覆盖 prompt 1/17/128、decode、非零初态、scalar gate
  与 per-channel KDA gate、两个输出、两次 replay 和两 invocation。输入/权重/state 由调用者提供；
  不增加模型加载器、tokenizer、采样或推理引擎。
- stop：锁定 ggml `456172ec` 的 `ssm_conv` 与 `gated_delta_net` 在其真实支持域进入 CPU、CUDA、
  Apple、Vulkan 普通 Direct/Graph，或以具名 backend `supports` 反例保留 residual。新增 HYLE 数学
  kernel/shader/SIMD 算子为 0。

## SCOPED owner 与边界

| owner | 本批允许审计/修改 | 本批禁止 |
|---|---|---|
| `compute_ops` | 增加仅含语义、推导和稳定 identity 的 `CausalConv1d` typed custom descriptor；把现有 GDN gate 从 scalar 扩到上游已经实现的 per-channel KDA | 自研 conv/GDN 数学、模型策略、运行时路由搜索 |
| CPU | `compute_provider_ggml` 的原版 `ggml_ssm_conv`、`ggml_gated_delta_net` 与现有 recipe/resource owner；`compute_route_cpu`/`graph_route_cpu` 普通消费 | oneDNN 伪造 recurrent 能力、逐 token host loop |
| CUDA | `compute_provider_ggml_cuda` 的同一原版节点与 native stream/buffer/completion；Graph 复用 leaf/最大合法 region | 新 CUDA kernel、CPU payload fallback |
| Apple | causal conv 优先使用 MLX `conv1d(groups=C)` compiled region；GDN/KDA 复用现有 ggml Metal whole region | 拆开 ggml recurrent 融合、增加 Metal shader |
| Vulkan | 扩大现有 ggml managed region 到 causal conv 与 KDA gate；设备最终由 `ggml_backend_supports_op` 裁决 | 新 SPIR-V、相似语义代签 |

`compute_ggml_runtime`、Mem、公共构建系统、路由 matcher 与冻结自研实现不在预期 FIX。若必须改变这些
边界，本批停止并重新定界；不在 B3 内扩建治理工具。

## SCOPED 审计与验收

1. 核对 public `[B,T+K-1,C] × [C,K] → [B,T,C]` 与 ggml native
   `[window,C,B] × [K,C] → [C,T,B]` 的无损冷布局映射；禁止 replay host copy。
2. 核对 GDN scalar/KDA gate、Q/K/V/state 布局、两输出与 backend 原版 support；CPU/CUDA recipe
   如需多输出，只扩其 SSA/output binding，不制造通用万能执行器。
3. 一批测试同时覆盖 Direct/Graph、prompt/decode、非零 state、scalar/KDA、旧结果和失败不发布；
   四路线只各做一次增量构建，M2 最后再做一次聚合三交付。
4. 性能只做 P0：确认 prepare 后图/程序复用、输入输出 device resident、state materialization 和同步
   次数。B3 没有冻结完整片段 P1 卡，不制造逐算子比较。

审计完成后只留下有限 `KEEP/FIX/BORROW/DEFER/REJECT` 表并把实际 owner 转为 `AUDITED`，之后才改
产品代码。
