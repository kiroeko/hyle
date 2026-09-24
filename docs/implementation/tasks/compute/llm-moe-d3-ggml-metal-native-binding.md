# M3-D3：ggml Metal 原生压缩权重接缝

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- owners：当前 `SCOPED` owner 为 `compute_route_apple`、`graph_route_apple`；
  `compute_provider_ggml_metal` 因注册表禁止 `AUDITED -> SCOPED`，在本卡完成审计前保留旧
  `AUDITED` 记录且不授权产品修改，审计完成后再绑定本卡。公共 ggml/GGUF storage catalog 由仍处于
  `AUDITED` 的 `compute_ops` 主卡持有。
- baseline：`f18bf7b2dad9b903bea06bfe8e14b60d1f023d8c`；Windows canonical 工作树干净。
- reopen trigger：固定 ggml `456172ec` 的 Metal backend 已含目标 Q/K/IQ/Q8/MXFP4
  `MUL_MAT`/`MUL_MAT_ID`，公共十格式 descriptor 已冻结到 `08b8e2f1`；但
  `NativeProgram::validate()` 仍只接受 dense F32/F16/I32，普通 Apple `InputSpec`/binding 也不能同时
  表达逻辑压缩 tensor 与物理 packed byte carrier。因此真实 GGUF 权重从 Apple Direct/Graph 不可达。
- consumer：已解析 GGUF tensor table 的普通 C++ 用户，用公共 checked `TensorDesc` 和保有 packed
  payload 的 MLX/Metal carrier 准备 Apple MatMul/GroupedMatMul 或 Graph region；准备期固定 ggml Metal，
  同一 invocation 对变化 activation 重复 submit。
- scope：Apple route 的原生 packed input carrier；ggml Metal 精确十格式 import、MatMul/
  GroupedMatMul lowering、能力检查和原生资源；Direct/Graph 普通入口、失败/复用与三交付。
- non-goals：GGUF 文件加载、量化/重编码、HYLE decoder/kernel/shader、把 MLX affine Q4 与 GGUF 格式
  混同、逐 kernel executor、改变 CPU/CUDA candidate、扩张算子目录或模型级推理。
- performance：先 P0；packed payload 不得因 HYLE 接线被常驻解码或逐次重排。完整片段 P1 仍由 D5
  在不可变 profile candidate 上统一计算准备、上传、同步和融合成本。
- stop：十种 catalog 格式在 ggml Metal 上逐一取得上游 capability verdict；至少 Q4_K、IQ4_XS、
  MXFP4 与 grouped expert 通过普通 Direct/Graph、两次 submit、资源/失败和 source/static/shared，或
  记录固定上游 capability rejection。不得用 MLX affine Q4、F32 展开或 CPU 路线代签。

## SCOPED

本卡只重开上述三个 Apple owner。下一步审计 Apple `InputSpec`/Graph value 到原生 Metal buffer 的物理
carrier、ggml tensor 轴与 block ABI、MatMul/GroupedMatMul lowering、常量/临时值边界和两台 Mac 的
真实支持矩阵，然后固定有限 FIX；在进入 `AUDITED` 前不修改产品代码。
