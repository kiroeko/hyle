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
   IQ4_NL、IQ4_XS、MXFP4 的版本化格式身份，以及 rank-2 `[K,N]` 和 grouped `[E,N,K]` checked
   descriptor；block 大小直接固定自 ggml/GGUF ABI，函数只描述已有字节，不量化数据。MXFP4 复用
   已公开的精确 32-value/17-byte identity，不制造第二种等价格式。
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

## 实现检查点

有限 FIX 已按既定边界落地，候选尚未冻结：

- 公共描述器固定十种原生 block ABI，rank-2 权重的逻辑 `[K,N]` 映射为 ggml 连续 `[N,K]`，grouped
  权重保持 `[E,N,K]`；已知 ID 配错 block elements/bytes 会被拒绝。
- CPU/CUDA 只在 MatMul RHS 或 GroupedMatMul weight 接受原生压缩 storage；lowering 直接建立对应
  `ggml_type` tensor，不对 packed weight 执行 `transpose + cont`，其它 operation 不会误收。
- 普通 `cpu::create_context()`、`cuda::create_context()` 与 CPU/CUDA Graph 入口均已证明绑定 ggml，
  同一 prepared invocation 对变化 activation 连续 submit 两次。Q4_K、Q8_0、IQ4_XS、MXFP4 和
  grouped expert 的代表程序已实际执行；错误 ABI 在 prepare/compile 前失败。
- P0 源码账单为：描述器创建 O(1)，prepare 按十项固定表匹配；packed payload 只走路线原有的输入
  upload/import，重复 submit 不分配格式转换缓存、不重排权重、不搜索或计时选择 Provider。

Windows Release 收据（RTX 4070 / SM89、CUDA Toolkit 13.3、ggml CUDA FA 开启）：

| 形状 | 结果 |
|---|---|
| shared CPU：Compute Ops、ggml CPU、CPU Direct/Graph、逐头编译 | 9/9 PASS |
| shared CUDA：Compute Ops、ggml CPU/CUDA、CUDA Direct/Graph、逐头编译 | 12/12 PASS |
| static CPU+CUDA：上述八个标签及逐头编译 | 17/17 PASS |
| selector 全影响面：22 个 factor 的构建与所选测试（不含 header） | 30/30 PASS |
| `compute_ops` 公共头反向消费者 | 1/1 PASS |
| Compute 源码嵌入、动态/静态安装、搬迁后外部消费 | 3/3 PASS |
| Graph 源码嵌入、动态/静态安装、搬迁后外部消费 | 3/3 PASS |
| Graph 动态 SDK 导出卫生 | 18 个库、1244 个导出符号 PASS |

Graph 交付在已经通过的同候选 Compute 对象树上把产品根从 `compute` 增量扩到 `graph`，再分别安装到
全新动态/静态目录并搬迁消费；只复用相同 toolchain、源码和配置的供应商对象，不复用安装结果或
consumer verdict。selector 未选择 P1：本批固定的是原生 storage 接缝与代表程序，真实模型片段的
完整准备、上传、同步和融合成本仍由 D5 在不可变 profile candidate 上裁决，不能在此写成 PASS。

维护检查 `check-module-closure.py --previous-ref f091b21d`、`check-docs.py` 与 `git diff --check` 均已
通过。产品候选只差提交冻结；冻结前本卡和三个 Provider owner 保持 `AUDITED`，不提前写
`CANDIDATE`/`CLOSED`。
