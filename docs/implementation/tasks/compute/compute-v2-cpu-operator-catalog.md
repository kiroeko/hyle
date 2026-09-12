# Compute v2：CPU 推理算子目录与多算子执行合同

## Closure card

- component / owner：`compute` / Compute；round=`compute-v2-cpu-operator-catalog`；
- baseline：`1b32fb8e792567b23d2aae15b1174ac52e0f8ffb`；previous terminal：
  [`compute-v1-clean-foundation`](compute-v1-clean-foundation.md)；
- 合法 trigger：v1 terminal 明确把完整算子目录留给后续卡；已接受的
  [`compute-design`](../../../architecture/compute-design.md)要求以 normalization → matmul → gated
  activation 的真实 CPU 片段验证 typed op/provider 合同；ggml/llama.cpp、vLLM、SGLang 与 ds4 都有
  对应稳定推理计算族；
- 正常用户任务：C++ 推理引擎准备固定 shape 的 RMSNorm、矩阵乘与 SiLU-gated multiply，随后在同一
  CPU Context/Queue 上重复提交，不在 steady path 做 shape inference、route search、packing 或 allocation；
- 正交反偏置任务：工程计算用户不构造 Graph，直接准备普通二维矩阵乘与逐元素 gated transform；公开
  类型不得泄漏 decoder/model/provider 专名；
- scope：审计并冻结 `MatMul` 与 `SiluMul`（最终名称由 AUDITED 形状决定）的 dtype、rank、layout、
  transpose、broadcast、alias、precision、failure 与 workspace 合同；实现本轮准入的 dense f32 CPU
  reference 与 HYLE-native prepared route；让现有 Operation/PreparedOperation 可自然承载多个 operation
  kind；为 Graph v2 暴露稳定描述和执行入口；
- non-goals：完整公共 op 超集、quantized matmul、attention/RoPE/embedding、模型/KV/batching/sampling、
  外部 ggml/oneDNN provider、CUDA/Metal/MLX/Vulkan/HIP、异步 queue、Native FFI；这些必须另开有限卡；
- 直接依赖：沿用 `core/error/mem` 公开边与 `math/simd/util_core` 实现边；fat-kernel selector 直接读取
  Platform 的规范 CPU facts，因此 `platform` 也是显式实现边；不新增 Sched 或 Graph 反向边；
- 直接消费者：Graph v2 multi-op fragment 与独立 direct-Compute package sample；
- P0：逐 op 记录复杂度、读写量、SIMD route、packing/workspace、allocation、alias check 与调用边界；
- P1：候选后只对最终 shipping CPU route 做一次同合同测试。MatMul/SiluMul 选择至少一个当前成熟实现作
  peer，并把 dtype/layout/precision/threading/setup/completion 对齐；Graph 税另归 Graph 卡。固定门 5%，
  方向无论幅度都如实进入总账；
- 机器与验证：Windows/MSVC 为工程性能机；Windows、WSL/GCC 14、macOS/AppleClang 做 correctness/header；
  WSL ASan/UBSan/TSan 按风险；OOM、overflow、zero extent、wrong shape/layout/dtype、unsafe alias、move/close、
  reference differential、direct consumer、source/static/shared、red-team、closure/docs/diff；
- 调查预算：四个真实引擎与成熟 CPU provider 各扫描一遍；能力/API 纠正最多两轮；性能根因纠正最多
  两轮；每个正式 workload 最多一次 `n=15` qualification；
- stop：若标准 MatMul 形状必须依赖未建 provider/packing owner 才能达到可接受路线，或 tensor contract
  无法表达所需 layout/precision，则停止并形成精确依赖卡，不以 benchmark-only fast API 绕过。

## SCOPED 边界

本状态只授权能力和源码审计，尚未授权修改产品。AUDITED 前必须输出：

1. 四个真实引擎及 ggml/oneDNN/BLAS 类 peer 的调用形状和语义差异；
2. 每个现有与候选 Operation 入口的需求回映；
3. 有限 `KEEP/FIX/BORROW/DEFER/REJECT` ledger；
4. 自然 C++ 调用草图以及同合同性能 preflight 计划。

## AUDITED 能力与形状裁决（2026-09-12）

### 真实消费与成熟实现对账

本轮审计使用以下本地、可复现源码身份，而不是旧 HYLE benchmark：

- ggml `e91ded11bdcd` 与 llama.cpp `e71b80510c84`：`ggml_rms_norm` 沿 row、
  `ggml_mul_mat` 的推理权重按 `[out, in]` 物理布局读取，图上再组合 SiLU 与 multiply；
- vLLM `51da0ca66c80`：`hw_agnostic/layers/layernorm.py` 明确 RMSNorm 沿最后一维且
  weight 为 `[hidden]`；`activation.py::SiluAndMul` 明确 packed-halves
  `silu(x[..., :d]) * x[..., d:]`；
- SGLang `dcebe8c4733a`：CPU MoE reference 为 `a @ w1.T -> SiluAndMul -> ... @ w2.T`，
  Triton route 同样以 `[rows, 2*hidden]` packed 输入工作；
- ds4 `f62ca29a3087`：公开/内部 kernel 同时存在 rows RMSNorm、普通/量化 MatMul 与
  gate/up/SwiGLU fusion；这些证明算子族稳定，但模型、量化格式和专名不进入本轮公开合同；
- oneDNN `6b7f1fbbb18e`：Graph gated-MLP 样例以通用 MatMul、Sigmoid/Multiply 和 strided
  logical tensor 组合，并允许 combined gate/up weight 通过 stride/offset 零拷贝解释。

审计因此否决 v1 的“rank-1 且 weight 与输入完整同形”限制。它只满足单行验证，不满足正常推理
consumer。转置也不是 MatMul 的布尔属性：逻辑 shape 与 element stride 已足够表达，公开 tensor
提供 permutation convenience，provider 可在 prepare 冷路径识别布局。

### 有限能力 ledger

| ID | 能力 | 裁决 | 本轮精确合同 |
|---|---|---|---|
| C20 | tensor axis permutation | `FIX` | `TensorDesc::permuted()` 与二维 `transposed()` 只重排自含 shape/stride，O(rank)、零分配、零 copy |
| C21 | row-wise RMSNorm | `FIX` | dense f32 input/output `[..., hidden]`；可选 dense f32 weight `[hidden]`；逐 row 复用 SIMD，允许 exact in-place |
| C22 | ordinary MatMul | `FIX` | dense f32 rank-2 A `[M,K]`、C `[M,N]`；B 为逻辑 `[K,N]`，支持 dense row-major 或 dense `[N,K]` 的 transpose view；无 bias/packing/隐藏 copy |
| C23 | packed SiluAndMul | `FIX` | dense f32 input `[...,2d]`、output `[...,d]`；按最后一维两半解释并逐 row 复用 `simd::swiglu_f32_batch` |
| C24 | single-output shape inference | `FIX` | `infer_output_description(Operation, inputs)` 是 Graph 与 direct consumer 的唯一 shape 规则；失败不修改 caller 状态 |
| C25 | typed Operation / PreparedOperation | `FIX` | 三种 descriptor 都是无 provider pointer 的小值；prepare 固定 kind/layout/count/route，submit 不再假定 RMSNorm |
| C26 | f16/bf16 storage | `DEFER` | SIMD 已有 leaf，但 MatMul/provider/accumulation 合同尚未同卡闭环；不得只为扩大 enum 宣称支持 |
| C27 | fused add-RMSNorm、bias/epilogue、separate-input SwiGLU | `DEFER` | 是稳定候选，但不是证明 multi-op Graph 所必需；后续按真实 fusion/provider 卡进入 |
| C28 | quantized/indirect/batched MatMul、attention/RoPE/MoE | `DEFER` | 由后续量化、provider 与算子目录卡分别所有，不污染普通 MatMul |
| C29 | provider-specific transpose/layout flag | `REJECT` | 使用 TensorDesc stride/permutation；不得把 cuBLAS/ggml/oneDNN 选择泄漏进公共 op |

### 所有权、失败与复杂度

| 对象/阶段 | owner 与 lifetime | 可失败事实 | 复杂度/热路约束 |
|---|---|---|---|
| `TensorDesc` permutation | self-contained value；不持有 storage | invalid/duplicate axis、rank mismatch | O(rank)，零分配/复制 |
| shape inference | 借用 descriptor span，仅返回一个 value | dtype/rank/shape/layout/overflow/epsilon | O(rank)，冷路径 |
| `PreparedOperation` | retain Context；拥有 descriptor/route/预计算维度，不拥有 tensor payload | unsupported contract/OOM 只在 prepare 前或 context state | fixed metadata；prepare 可检查，submit 不搜索 route |
| submit bindings | 借用 caller Mem views 到 Completion 终态 | context/queue、shape、domain、mapping/coherence、unsafe alias | O(rank+bindings)+算子本体；零常态 heap/JIT/packing/copy |

MatMul 对转置 weight view 走连续 K dot；普通 row-major B 走连续 N accumulation。两者是同一逻辑 API 的
两种真实 layout route，不是 benchmark-only fast entry。空 extent 成功且不解引用空 storage；任一非空
dimension/byte乘法溢出必须在冷路径报告。

### 自然 C++ journey

```cpp
auto x = TensorDesc::dense(DataType::kFloat32, std::array{rows, hidden}).value();
auto stored_w =
    TensorDesc::dense(DataType::kFloat32, std::array{2 * ffn, hidden}).value();
auto logical_w = stored_w.transposed().value(); // [hidden, 2*ffn], zero-copy view
auto projected = infer_output_description(MatMul{}, std::array{x, logical_w}).value();
auto gated = infer_output_description(SiluAndMul{}, std::array{projected}).value();
auto prepared = prepare(context, MatMul{}, x, logical_w, projected).value();
```

同一 `Operation` 与 descriptor 直接交给 Graph；用户不构造 string 属性、raw kernel 参数、provider node
或输出 shape 的第二份规则。

### 性能 preflight

- RMSNorm：`rows={1,4,32}`、`hidden={256,4096}`，和直接调用同一 HYLE SIMD row kernel 对齐；
- SiluAndMul：相同 row/hidden，和直接 `simd::swiglu_f32_batch` 对齐；
- MatMul：decode `M=1/4` 与短 prefill `M=32`，`K/N` 使用 transformer 常见量级；分别测 row-major B
  与 `[N,K]` transpose view，peer 必须同为单线程 f32、同 layout、同 prepare/completion 边界；
- 不把 HYLE 的 shape/alias/storage 安全合同与 raw pointer leaf 混成胜负；leaf 与完整 prepared API 分层
  报告。正式 P1 只在 immutable candidate 后执行一次。

## Audit verdict

`C20--C25` 构成一个有限、模型无关且可由当前依赖实现的 CPU 产品增量；没有新质因数或反向依赖。
允许状态进入 `AUDITED` 并修改 Compute 产品。实现若要求隐藏 packing、引入 provider 专名或扩大到
`C26--C28`，必须停下另立卡。

## Provider seam follow-up（2026-09-12）

本卡预声明的普通 row-major MatMul stop 已由
[`compute-provider-onednn-v1`](compute-provider-onednn-v1.md) 精确触发。该卡依法接管 provider、packing/
scratchpad 与并发 run-state，不回写本卡的算子语义。为支持 provider，`PreparedOperation` 从 move-only
inline native state 调整为可复制的不可变 shared artifact，实际 submit 移到每次运行独占的
`OperationInvocation`；native route 也走同一公开形状。此项不是兼容层或 benchmark API，而是异步/第三方
provider 都需要的 ownership 修正。

## Native row-major MatMul checkpoint（2026-09-12）

旧 v1 scalar tile 没有被包装或保留。普通 row-major f32 路线现由 Compute 自己拥有矩阵语义，并复用
SIMD/Platform 已冻结的编译期 fat-kernel 与一次 ISA 选择设施：每个 ISA body 直接编入 `hyle_compute`，
`prepare()` 只选择一次固定签名函数指针；steady submit 不探测、不路由、不 packing、不申请 workspace，
也没有跨模块 vector ABI。

源码复核纠正了早先对 oneDNN AVX2 BRGEMM blocking 的误读：oneDNN 3.13.2 对普通 f32 AVX2 选择的
主 balance 是 3-row × 4-vector，而不是 6-row × 2-vector。HYLE 因此使用同为 12 accumulator 的 3×4
prefill tile、2×4 尾 tile，以及既有 4×2/1×4 decode/小矩阵路线；B 地址使用 pointer induction，prefill
路线以 64 个 f32 column 的 N block 完成所有 row tile，再前进到下一 B slab。曾尝试的强制 K×4 展开被
MSVC 生成的 accumulator stack spill 明确否决并删除；最终 AVX2 3×4 与 2×4 K 热循环反汇编均为零
vector stack access。`hyle_add_fat_kernel()` 同时把 IPO/LTO 隔离提升为中央 helper 的不变量，Compute 因
直接使用 CPU selector 在中央组件图声明 `platform` 私有边。

Windows correctness 的 deterministic differential matrix 覆盖 1/4/6-row tile、组合余数、SIMD/N-block
尾部和 `K=0`，所选 Compute 测试为 1/1 PASS。Windows i9-13900K 固定 logical CPU 2 的一次非资格化
dual-order 诊断中，同一公开 API、layout、workspace 与 completion 下，native 在 `M=1/4,K=N=1024`
两种顺序均领先 oneDNN；`M=32,K=512,N=1024` 则仍约慢 10.4%--16.1%。相比改动前约慢 27%--44%，
这只证明 tile/codegen 根因修正有效，不是 P1 或 parity 证据。正式 P1 必须由项目仪器记录 affinity/frequency
并在 immutable candidate 后执行；M=32 固定输项在此之前保持公开，不能以 5% 误差或被 BLOCKED 的
oneDNN provider 冒充通过。
