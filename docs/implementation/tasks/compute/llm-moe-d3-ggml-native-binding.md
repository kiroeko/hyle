# M3-D3：ggml native storage 与 CPU/CUDA 批量接缝

<!-- MODULE REVIEW: SCOPED -->

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
