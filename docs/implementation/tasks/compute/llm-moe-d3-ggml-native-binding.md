# M3-D3：ggml native storage 与 CPU/CUDA 批量接缝

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owners：`compute_provider_ggml`、`compute_provider_ggml_cuda`、
  `graph_provider_ggml_cuda`；公共格式描述由已处于 `AUDITED` 的 `compute_ops` 主卡持有。
- baseline：`8bc31b76f`；Windows canonical 工作树干净。
- reopen trigger：真实 Provider 矩阵证明固定 ggml 已含目标 GGUF K/IQ/Q8/MXFP4 程序，但 CPU/CUDA
  HYLE adapter 的 `representable()`/lowering 只接受 plain storage，普通 Direct/Graph 不可达。
- consumer：普通 C++ consumer 从已加载的 GGUF tensor table 构造 checked `TensorDesc`，在 CPU 或
  CUDA route 准备 `MatMul`/`GroupedMatMul` 或同一成熟 region；准备绑定后重复 submit。
- scope：公共 ggml/GGUF native storage 身份和描述器；CPU/CUDA native tensor import、能力查询、
  lowering、资源/完成；Direct 与 Graph ordinary entry；真实混合格式小片段。
- non-goals：读取 GGUF 文件、量化/重编码权重、实现 decoder/kernel、改变 Graph/route 架构、
  Apple/Vulkan 接线、冻结自研算子比较。
- performance：先 P0；候选固定且默认路线变化后才对包含准备/上传/同步的片段执行一次适用 P1。
- stop：精确 K/IQ/Q8/MXFP4 格式要么从 CPU/CUDA 普通入口运行通过，要么记录固定上游 capability
  rejection；不得退回人工 `Q4-S128` 或浮点全量常驻替代。

## SCOPED

本卡只对 CPU/CUDA 的共同接缝定界。下一步审计物理 block ABI、现有 MatMul/GroupedMatMul shape
翻译、provider resource import 和 Graph recipe 复用，然后固定有限 FIX；在 `AUDITED` 前不改产品。

## AUDITED

固定 ggml 物理 ABI 与现有 lowering 后，有限 FIX 为：

1. `compute_ops/low_precision` 增加 Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、IQ3_XXS、IQ3_S、
   IQ4_NL、IQ4_XS 的版本化格式 ID，以及 rank-2 `[K,N]` 和 grouped `[E,N,K]` checked descriptor；
   block 大小直接固定自 ggml/GGUF ABI，函数只描述已有字节，不量化数据。
2. CPU/CUDA adapter 以格式 ID 映射 `ggml_type`，只在 MatMul RHS 或 GroupedMatMul weight 端接受；
   native K 轴成为 ggml `ne0`，不对压缩权重做 `transpose + cont`，其它 operation 继续拒绝量化 storage。
3. CPU Direct 接受成熟 ggml quant `MUL_MAT` 的精确 F32/F16/BF16 activation 域；CUDA Direct 允许
   rank-2 quant MatMul。真正 capability 仍由 `ggml_backend_supports_op`/prepare 返回决定。
4. Graph 复用同一 recipe/compile，不新建逐 kernel Graph adapter；增加普通 Direct/Graph 的 Q4_K、
   Q8_0、IQ4_XS、MXFP4 代表片段，两次 submit，并检查错误格式在准备期拒绝。
5. 只运行 selector 选出的 `compute_ops`、ggml CPU/CUDA、对应 Graph/route consumer；候选稳定后再做
   source/static/shared 与适用 P1。本批不修改任何上游 kernel、自研 kernel 或 Marlin。

以上是本卡全部产品改动。若 grouped schema 或 native resource 无法在该集合内表达，退回主卡重新
定界；不能增加 decoder 或相近格式 fallback。
