# M3-D3：ggml Vulkan 原生压缩权重接缝

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- owners：当前 `SCOPED` owner 为 `compute_route_vulkan`、`graph_route_vulkan`、
  `compute_provider_ggml_vulkan`；`compute_provider_iree_vulkan` 已处于旧卡 `AUDITED`，本卡审计完成前
  保留原记录且不授权修改。公共 GGUF storage catalog 仍由 `compute_ops` 的 M3 主卡持有。
- baseline：`92b08826168a93c04f6868de9896c0bdbb262060`；Windows canonical 工作树干净并与
  `origin/main` 一致。
- reopen trigger：固定 ggml `456172ec` 的 Vulkan `supports_op` 与 shader 生成器已含公共十格式的
  `MUL_MAT`/`MUL_MAT_ID`；但 HYLE Vulkan managed region 只识别 plain storage 与一个 MXFP4 特例，
  其余真实 GGUF K/IQ/Q8 权重从普通 Direct/Graph 入口不可达。
- 正常 consumer：已解析 GGUF tensor table 的普通 C++ 用户，把公共 checked `TensorDesc` 与原始 packed
  bytes 上传或借入同一 Vulkan device，准备 MatMul/GroupedMatMul 或 Graph region，并在 prepare 固定
  Provider 后对变化 activation 重复 submit。
- 主产品纵切：Qwen dense `[M,K] * Q4_K[K,N]`；正交反偏置纵切为 MoE
  `IQ4_XS[E,N,K] * [T,S,K] + ids[T,S]`，另用 MXFP4 检查 32/17 block 与已有路径兼容。
- scope：复用公共十格式 catalog，批量补全 ggml Vulkan native type/shape/import、MatMul 与
  GroupedMatMul、Direct/Graph 选择、资源/失败/复用和三交付；核对限定 IREE 标准/FP8/scan 域，不把它
  伪装成 GGUF Provider。
- non-goals：GGUF 文件加载、量化/重编码、HYLE decoder/kernel/shader、CPU fallback、逐 kernel vtable、
  新 Provider、自研数学、模型级推理或改变 CPU/CUDA/Apple candidate。
- 依赖与 consumer：`compute_ops -> compute_provider_ggml_vulkan -> compute_route_vulkan ->
  graph_route_vulkan`；IREE 只保留精确合格域。公开 consumer 是 Vulkan Compute/Graph 用户文档。
- performance：P0。packed bytes 必须直接进入原生 ggml Vulkan tensor，不得常驻解码、复制为浮点或在
  submit 重新选路/重排。完整 profile 的端到端 P1 留给 D5。
- machines：Windows RTX 4070 是产品、主 correctness 与交付真相；MacBook Air 的 MoltenVK 只签
  portability/capability，不替代 Windows verdict。构建低并行、重型验证串行。
- budget：一个批量 type mapper、两类 operation lowering、现有 route/Graph 判定、一个批量测试目标与
  现有交付 smoke；若上游实际不支持某格式/形状，记录精确 rejection，不增加自研 shader。
- stop：十种格式逐一取得静态及 RTX 4070 capability verdict；至少 Q4_K、IQ4_XS、MXFP4 与 grouped
  expert 通过普通 Direct/Graph、两次 submit、资源/失败和 source/static/shared，或形成上游具名拒绝。

## SCOPED

本卡只重开上述三个 ggml Vulkan/route owner。下一步审计公共 descriptor 到 ggml tensor 的 type、轴与
byte ABI，固定上游 shader/device 条件、GroupedMatMul 参数顺序、buffer alignment/lifetime、Graph 分区和
IREE 边界，再写有限 FIX。在三个 owner 进入 `AUDITED` 前不修改产品代码；旧 IREE owner 也不会在审计
完成前被本卡改动。
