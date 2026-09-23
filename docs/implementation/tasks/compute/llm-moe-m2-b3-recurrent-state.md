# M2-B3 成熟 Provider 递归混合状态批次

<!-- MODULE REVIEW: AUDITED -->

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

## AUDITED：原版入口与有限 FIX

2026-09-24 在 SCOPED baseline 上核对锁定 ggml 的声明、backend 实现和 upstream backend-op tests，
并核对 MLX 0.32.2 的 `conv1d` C++ API。ggml 的 CPU、CUDA、Metal、Vulkan 都登记
`GGML_OP_SSM_CONV` 与 `GGML_OP_GATED_DELTA_NET`；上游 GDN 已明确接受 scalar gate
`[1,Hv,T,B]` 或 KDA per-channel gate `[Dv,Hv,T,B]`，返回 scores 与 K 个 state snapshot 的 packed
结果。HYLE 只取 K=1 并发布两个公共输出。

| ID | 有限施工 | 验收 |
|---|---|---|
| B3-C0 | 新增 `model_ops.h` 中稳定 identity 的 `CausalConv1d` descriptor；公共布局固定为 `[B,T+K-1,C] × [C,K] → [B,T,C]`；GDN 推导接受 scalar/per-channel gate | header self-contained、错误 arity/type/shape、scalar/KDA 独立公式 |
| B3-C1 | CPU ggml recipe 的单值 `output` 改为有限 output vector；普通 leaf 接入 `ssm_conv` 与 packed GDN 的两个 owning outputs | Direct/Graph、T=1/17/128、两次 replay、非零 state、旧结果 |
| B3-C2 | CUDA 复用同一 recipe/output 模型，原版 graph 在 native stream 上运行；外部 output 各有独立 device binding | SM89 Direct/Graph、scalar/KDA、completion/failure，不增加 `.cu` |
| B3-C3 | Apple causal conv 由 MLX `conv1d(groups=C)` 放入现有 compiled region；现有 ggml Metal GDN whole region 扩到 per-channel gate | Air shared、Mini static、MLX replay 与 Metal 两输出资源 |
| B3-C4 | Vulkan managed region lower `ssm_conv`，现有 GDN 接受 KDA gate；最终资格继续调用 `ggml_backend_supports_op` | RTX 4070 与 Air MoltenVK、无 host payload fallback |
| B3-C5 | 追加一组表驱动 B3 语义/路线测试、P0 成本与 manifest residual；M2 末尾再运行一次聚合三交付 | selected gates、docs/closure；未运行 P1 保持 SKIPPED |

`DEFER`：锁定 upstream 的四个 backend 对 `ssm_conv` 资格均为 F32；GDN 本批也只对各 backend 实际
通过的 dense F32/equal-depth 域签署。F16/BF16、更多 state snapshot、融合 bias+SiLU 与完整 24 层
旅程如果没有同合同成熟路径，作为 B3 residual 进入 M3，不能用 host conversion 或自研 kernel 补写。

以上修改不会改变组合路由维度；模型族/模型名/phase、平台、设备架构、驱动/SDK、量化档位仍由既有
冷 matcher 决定候选，Provider 的真实 shape/dtype support 再完成准备期绑定。有限 FIX 已冻结，可以
进入产品施工。
