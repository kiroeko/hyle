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

## Candidate checkpoint（2026-09-12）

不可变 Compute v2 产品候选为 `d6a93086b4cdeb6b412f62c103ff9fb5d61f8902`。它固定 C20--C25 的
Tensor permutation、row-wise RMSNorm、普通/transpose-view MatMul、packed SiluAndMul、唯一 shape
inference、prepared artifact/run-state owner 和 HYLE-native SIMD MatMul route。当前 HEAD 相对该候选的
`include/hyle/compute` 与 `src/compute` 差异为空；后续提交只改变 Graph、可选 provider、测试、证据、
状态和文档，没有用 oneDNN P1 结果回写 native 产品。

Graph v2 已直接消费同一 Operation/shape-inference/prepared-invocation 合同，Windows、Linux 与两台 Apple
Silicon 的 Compute/Graph 定向 correctness/header 以及 WSL ASan/UBSan/TSan 已覆盖当前产品字节。候选
LOC 自检为全库 `563,541` 行、自实现不含测试 `99,071` 行、Compute 实现 `2,246` 行。

这只允许进入 `CANDIDATE`，不构成关闭：本卡预声明的最终 shipping native route P1、当前候选的定向
source/static/shared 复核、终局七问与红队仍须完成。若 P1 证明 M=32 native route 存在 material 固定输且
不能由已关闭 provider 的自然选择边界诚实解决，依法退回 `AUDITED`；不得放宽门或用 decode 胜利平均。

## Final CPU route P1 preflight（2026-09-12）

正式合同为 [`compute-cpu-route-p1-v1`](../../../../bench/contracts/compute-cpu-route-p1-v1.json)，真实形状
账本为 [`consumer_shapes_v1.json`](../../../../bench/compute/consumer_shapes_v1.json)。它不再把完整 Compute
安全合同与裸指针 GEMM 混成一个胜负：candidate 是显式启用已关闭 oneDNN provider 的普通 Compute
Context，peer 是 provider-empty 的同一 Compute v2 candidate；两侧使用完全相同的 Tensor/Buffer/Queue/
PreparedOperation/Invocation/Completion、串行执行和独立同值 storage。

- row-major `M=1/4,K=N=1024` 与 physical `[N,K]` transpose-view `M=1` 应在 prepare 时选择同一
  HYLE-native/SIMD body；这三行验证 provider fallback 在 steady path 没有扫描或抽象税；
- row-major `M=32,K=512,N=1024` 是此前暴露 10.4%--16.1% native loss 的精确机制行；oneDNN-enabled
  candidate 必须在 prepare 时选择已经独立证明对 direct ukernel 5% 非劣的 BRGEMM provider；
- allocation order、execution order 与函数 placement 独立交叉，共 8 strata；四行都必须通过固定 5%
  non-inferiority，方向无论大小都进入总账；
- RMSNorm 与 SiluAndMul 不重复裁决 leaf：当前 Compute candidate 直接调用已关闭 SIMD 的
  `rms_norm_f32` 与 `swiglu_f32_batch`，后者已经在锁定 Highway 同合同上通过 P1；当前 Graph v2 P1 又以
  同一 Compute PreparedOperation 对拍完整三算子 fragment。这里复用两个既有证据层，不把完整安全 API
  对裸 pointer leaf 的额外语义写成 loss 或 win。

一次非资格化 dual-order/placement 预检确认四个 operation 都选择预期 route、输出在声明容差内一致；
M=32 的 selected/native op/s 比约 `1.05--1.11`，其余同 body 行随单次顺序在约 `0.95--1.05` 内翻转。
这只证明合同可以进入一次 `n=15` qualification，不用于修改产品或签发结论。

## P1 evidence correction（2026-09-12）

`compute-cpu-route-p1-v1` 的一次采样虽通过通用 qualification controls，但工程 verdict 为
`INCONCLUSIVE`。随后按性能证据标准复核 benchmark 源码，确认它把两侧分别交给两个连续的
`Bench::run()`，而不是同 iteration count、cyclic interleave 的 `Bench::compare()`；同时它只使用通用
算术 A/A，没有运行完整 Compute workload 的同路线 A/A。v1 因此违反测量前硬门，保留为失败事实，
不得用于阻塞产品、签 non-inferiority，或靠追加样本寻找 PASS。

纠正后的新实验是
[`compute-cpu-route-p1-v2`](../../../../bench/contracts/compute-cpu-route-p1-v2.json)：

- 两侧依次覆盖同一组 caller-owned input/output tensor；Context、prepared artifact 和 invocation 仍独立，
  因而测到的是真实 route/algorithm 差异，不是两份 4 MiB 权重与输出地址的 cache/page-color 差异；
- 八个完整 warm operation 交替执行，正式行使用同一 `HarnessRunner` 类型和
  `Bench::compare()`；104 个 paired rounds、4 ms epoch 只提高进程内分辨率，不改变 5% 门；
- `HYLE_BENCH_AA_ROUTE={native,selected}` 可把两行替换为同一完整产品 route。独立 validator 在
  primary/placement、双执行顺序、双 artifact allocation 顺序共 16 个进程形状上覆盖全部四个矩阵；
  它按 route/shape 检查 overall、execution-order、allocation-order 与 placement 各因子中位数均在 5%
  band，并以 10% 拒绝单进程 gross instability，不能让某个靠近 5% 边界的短样本冒充固定偏差。当前
  短预检全部 PASS；最近一次 raw ratio 位于约 `0.972..1.053`，所有因子中位数仍在 5% 内；
- `M=1/4` 与 transpose-view M=1 在 prepare 后执行同一 native function pointer，P0 已证明 steady path
  没有 provider search；它们在 v2 中只保留为 mechanism diagnostic。唯一产品门是会实际改变算法的
  `M=32,K=512,N=1024` selected-oneDNN 与 forced-native 同合同路线。

以上只修复证据命题和仪器，不改变 Compute candidate 产品字节。

## Final CPU route P1 verdict（2026-09-12）

`compute-cpu-route-p1-v2` 在 W1-WIN、MSVC 19.50、Release、static、portable ISA、LTO off 的一次
engineering qualification 中于 `n=15` 按规则停止；`qualification_gate=PASS`、
`performance_gate=PASS`，没有运行 `n=31`，artifact identity、fresh-build contract、完整 workload A/A、
5%/10% known delta、timer floor、placement、host/input stability 全部 PASS：

- row-major `M=1` 的八个 point ratio 为 `0.99612--1.00697`，99.26% CI 总范围
  `0.97052--1.02849`，combined=`PARITY`、5% non-inferiority=`PASS`；
- row-major `M=4` 的八个 point ratio 为 `0.99780--1.00263`，CI 总范围
  `0.99476--1.00642`，combined=`PARITY/PASS`；
- short-prefill `M=32,K=512,N=1024` 的 shipping selected-oneDNN route 相对 forced-native 八个
  point ratio 为 `1.06388--1.07328`，全部 CI lower 均大于 1；两个 CI 与 5% material boundary 相交，
  因而严格 combined exact verdict 保持 `INCONCLUSIVE`，但方向稳定有利且 5% non-inferiority=`PASS`；
- transpose-view `M=1` 的八个 point ratio 为 `0.99440--0.99995`，CI 总范围
  `0.98274--1.01221`，combined=`PARITY/PASS`。该行两侧在 prepare 后执行同一 native function pointer，
  点值的小幅同向漂移没有形成可归因的产品机制。

完整 bundle 保存在
[`compute-cpu-route-p1-v2-1e9c37f5-w1-win`](../../../../evidence/performance/compute-cpu-route-p1-v2-1e9c37f5-w1-win/)；
严格 `hyle-perf verify` 返回 `QUALIFIED`、`CURRENT`、`EXACT`、`paths_verified=true`。结果证明 shipping
选择边界修复了此前 M32 native 固定输项，同时没有给 decode 或 transpose fallback 增加 material steady
tax；它不把同函数体诊断行宣传为外部 SOTA 胜负。

## 七问终局答案

1. **第三方路径**：`compute` factor 本身没有直接 vendor dependency 或 vendor public type；它通过已关闭的
   Mem/Platform/SIMD 等 HYLE factor 间接取得 mimalloc、Highway/xsimd 的合法实现。x64 oneDNN 是独立、
   可关闭的 `compute_provider_onednn` factor；关闭它时不下载、不编译、不导出恒失败 stub，启用时仍只向
   Compute 返回 type-erased `Provider` 值。
2. **功能完整**：C20--C25 完整拥有 CPU Device/Context/Queue/Completion、Mem storage bridge、rank<=8
   TensorDesc/view、row-wise RMSNorm、普通和 transpose-view MatMul、packed SiluAndMul、唯一 shape inference、
   prepared artifact/per-run invocation 与 typed host-kernel escape hatch。f16/bf16、量化、attention/RoPE、
   embedding、batched MatMul、region fusion 与 GPU backend 是明确后续纵切，不是当前 CPU 卡冒充的能力。
3. **内部复用**：公开边只指向 `core/error/mem`；实现边复用 `math/platform/simd/util_core`。错误、allocator、
   Buffer/mapping/coherence、refcount、ISA facts 与 whole-row kernels 均有既有 owner；Compute 没有复制这些
   机制，也没有反向依赖 Graph、Sched 或可选 provider。
4. **性能**：shape inference/prepare 是 O(rank) 冷路径；provider/ISA 选择、JIT 与 workspace 决策只在
   prepare。warm submit 为 O(rank+bindings+算子本体)，零常态 heap、route search、JIT、packing、隐藏 copy、
   thread handoff 或 syscall。正式 P1 的全部行通过 5% non-inferiority；同 native decode/transpose 为
   `PARITY`，M32 shipping selected-oneDNN 对 forced-native 八个 point ratio 为 `1.06388--1.07328`。
5. **接口与易用性**：公开面是自然 C++23 typed value/RAII journey；direct 用户不必构造 Graph，重复执行
   明确分开 `prepare -> create_invocation -> submit -> Completion`，昂贵阶段和 workspace owner 可见。
   没有旧 C API/Native FFI、字符串属性袋、裸 `void*` 用户入口、provider-specific transpose flag、
   benchmark knob 或固定 transformer façade。
6. **构建交付**：中央 component graph 是唯一内部依赖真相，`src/compute/CMakeLists.txt` 只登记源码和
   fat-kernel body；oneDNN 拥有自己的 target、开关、离线依赖、静态开发闭包与许可证。Windows 的 source、
   relocated installed shared、relocated installed static consumer 均实际执行通过；shared export hygiene
   扫描 6 个 HYLE libraries、682 个 exports，无私有/vendor/C ABI 泄漏。
7. **文档与状态**：用户指南说明何时用 Compute/Graph、MatMul layout、provider/executor、owner、失败边界、
   性能限制和三种交付；架构页只保存长期 backend/provider 决策，不复制易过时进度。当前状态只由 registry
   指向本 terminal record，性能总账保留完整 bundle、严格方向与未覆盖能力。

## Terminal record（2026-09-12）

不可变 Compute v2 产品候选为 `d6a93086b4cdeb6b412f62c103ff9fb5d61f8902`。候选到终局的
`include/hyle/compute`、`src/compute`、Compute tests 与中央 component/link graph 差异为空；候选后的变更
只增加已独立关闭的 optional provider、Graph consumer、package sample、benchmark 纠正、证据和文档，
没有根据 P1 回写 API、layout、算法或常态 metadata。

| Gate | 候选绑定证据 | Verdict |
|---|---|---|
| scope | C20--C25 全部完成；C26--C28 明确交给后续 dtype/量化/operator/provider 卡，C29 拒绝；direct host-kernel 正交 journey 证明产品不是三算子 benchmark façade | `PASS` |
| architecture | backend/provider 两轴、immutable prepared artifact/per-run invocation、Mem storage 与 Completion lifetime 分层成立；依赖无环且没有 Graph/Sched 反向边 | `PASS` |
| correctness | Windows、Linux、MacBook Air、Mac mini 的 Compute public-header 与功能测试通过；zero extent/overflow、shape/layout/dtype、alias、coherence/mapping failure、OOM/rollback、move/close、跨 Context、reference differential 与 native tile/tail 均覆盖；WSL ASan/UBSan/TSan 零 report | `PASS` |
| delivery | Windows source embedding、relocated shared、relocated static consumer 全过；三者实际运行 native 与 oneDNN 路线，shared 6 libraries/682 exports 通过卫生检查，static dev/license closure 完整 | `PASS` |
| usability_docs | `Device -> Context -> Queue`、Tensor/Mem bridge、direct prepare/invocation、typed host kernel、oneDNN 选择与限制均有可编译 journey；无伪 C ABI 或隐藏昂贵 convenience | `PASS` |
| red_team | 有界复核公开入口、ownership、销毁序、并发声明、provider failure、OOM、alias、storage domain、组件边、导出、旧入口、benchmark seam 和苹果/橘子风险；只发现并纠正 v1 测量仪器及过时状态文字，未发现候选产品反证 | `PASS` |
| engineering_performance | P0 成本结构通过；`compute-cpu-route-p1-v2` 为 `QUALIFIED/CURRENT/EXACT`，完整 workload A/A 与全部 controls 通过，所有产品行 5% non-inferiority=`PASS` | `PASS` |

代表性验证与结果：

```text
python scripts/hyle-perf.py verify --contract bench/contracts/compute-cpu-route-p1-v2.json --manifest evidence/performance/compute-cpu-route-p1-v2-1e9c37f5-w1-win/manifest.json
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/public/project-hyle/hyle/.scratch/compute-v2-delivery -DHYLE_SMOKE_COMPONENT=compute -P cmake/HyleCiSelectedComponentSmoke.cmake
python scripts/loc-report.py --module compute --commit d6a93086b4cdeb6b412f62c103ff9fb5d61f8902 --self-check
```

严格性能重放为 `QUALIFIED/CURRENT/EXACT/paths_verified=true`；selected-product smoke 返回
`source, shared, static, relocated consumers` 全部 PASS；LOC self-check 为全库 `563,541` 行、自实现不含
测试 `99,071` 行、Compute 实现 `2,246` 行。

公开限制：当前只有同步 CPU execution 与 f32 RMSNorm/MatMul/SiluAndMul；Queue 不拥有 worker，caller
payload、workspace、provider executor backing 与 allocator 必须满足指南寿命；provider terminal failure 后
output 可以部分写入。只有可复现 correctness/security/lifetime/alias 缺陷、真实消费者要求当前 owner 下的
新 CPU dtype/op/fusion、公开合同/算法/交付变化、相关依赖升级，或合格同合同证据显示 material regression
才重开本卡；ggml region 与每个 GPU backend/provider 必须建立自己的 closure card，不能回写本记录。

七门全部 `PASS`，Compute v2 CPU 纵切裁决为 **`CLOSED`**。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
