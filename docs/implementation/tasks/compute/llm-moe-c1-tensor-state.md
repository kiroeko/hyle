# C1：四路线基本张量与状态边界

## Closure card

- round / current owner：LLM/MoE C1，`/root`。
- baseline：`e346e8fadfce7d4d9a7d94b5b1ebb4e80f1b87d1`。
- 上位范围：[唯一计划](llm-moe-coverage-completion-plan.md)；输入/来源准备见
  [C0](llm-moe-c0-local-contracts.md)；逐需求见 [64 行矩阵](llm-moe-coverage-matrix.json)。
- trigger：实际本地 2B 语言权重/参考计算与七个模型源码要求 F16/BF16 embedding、norm、
  softplus、广播、索引及持续状态；目前 CPU 普遍限 F32，CUDA 缺普通索引/norm，Apple/Vulkan
  typed 普通入口只有窄 GEMM/SiLU 域。是具名真实 consumer 的缺失能力，不重审已验收基础设施。
- 当前施工 owner：`compute_ops`（schema/shape）、`compute`（普通查询/准备事实）、`graph`
  （必要共同 shape/effect 边界）、`compute_provider_native`、`compute_provider_cuda`、
  `compute_route_apple` / `graph_route_apple`、`compute_route_vulkan` / `graph_route_vulkan`；新增直接依赖
  `compute_backend_cuda` 只按下述 C1-L 卡施工。
  `compute_route_cpu/cuda` 和 `graph_route_cpu/cuda` 默认工厂先复用；只有具体代码缺陷才定向纳入。
  oneDNN、cuBLASLt、Marlin、Mem/Sched 本包没有已证缺陷，不自动重开。CUDA backend 只因 checked
  操作缺少异步算法状态进入 [C1-L 定向依赖卡](llm-moe-c1-cuda-device-status.md)，其余 backend 不重开。
- 状态：`compute_ops` / `compute_provider_native` 已完成下述 C1-A/B/C/D 的 AUDITED；`compute` 仅
  对 D01 的 typed prepare 和 F01–F02 的普通准备诊断进入 AUDITED；`compute_provider_cuda` 对 E01–E03、G01–G04、H01–H03、I01–I03、J01–J03、K01–K03、M01–M03、N01–N04 和 O01–O03 进入 AUDITED；其余五个 owner
  保持 SCOPED。CUDA backend 仅对 C1-L 的 L01–L04 进入 AUDITED。AUDITED 只授权具名固定子集，
  不授权本卡其他尚未裁决的草案。
- 主 consumer：本地 2B 的 embedding → Gemma RMS → gated 逐元素 → residual → index/state
  输入输出，两独立 invocation 与旧结果保留；实际 BF16 输入，不把 F32 smoke 当作通过。
- 正交 consumer：F32/F16 带 stride 的批量变换、空/尾块、共享不可变表与独立索引更新。
- scope：矩阵 C1 的 F01–F04、F19，后续能力只消费这些机制。新增 softplus、必要 scan 与已有
  schema 的完整实际执行；L2/gated norm 优先复用原子并让 provider 保留融合，不造模型层类型。
- non-goals：GEMM/QMM 扩域、attention/conv/DeltaNet/KDA/MoE 完整片段、模型加载/策略、公共
  治理/统计工具改写、全库依赖重审；它们按后续包推进，不顺手塞入本包。
- tier：P0 + 候选固定后每路线一个具名 C1 bundle 的 P1；当前不运行 P1。
- 调查预算：每个具体未决机制最多 60 分钟/两个具名来源；本包产品纠正最多两轮。超出时按具体
  反例重新定界或记录 blocker，不反复推翻四路线架构。

## 已完成的源码审计输入

| owner | 当前事实 / 源码 | C1 缺口与承载方向 |
|---|---|---|
| compute_ops | `operation.h` Unary 止于 SiLU；已有 View/Cast/Reduce/RmsNorm/LayerNorm/Gather/Scatter/TopK；operation.cc 完成共同 shape 检查 | softplus 与必要 scan 的 typed schema；保留既有数学/alias 语义，明确半精度舍入 |
| compute | operation.cc 已有 cold propose → prepare 与明确 decline/error；普通入口不能只查 schema 推断硬件可用 | 普通 capability query 与准备拒绝复用同一资格判断；不执行 payload，不变热路径 |
| Native | elementwise/normalization/indexing/reduction/transform 分文件；SIMD F32 优势与 CPU executor 已存在 | F16/BF16 所需路径和稳定数学；按实际消费融合、尾块/stride；不每元素分配或调用 vtable |
| CUDA | native_kernel.cu + SGLang SwiGLU 与 cuBLAS/Marlin 工厂组合；有自己的 native stream/capture | 成熟 CUDA kernel 承载必要 tensor/index/norm；HYLE 控制边界、失败和数值，不借 CPU 模型 fallback |
| Apple | apple_operations.cc 仅 F32 2-D GEMM/affine Q4/SiLU；apple_lowering.cc 拒绝非纯/非固定域 | MLX GPU ops/完整 region 的 typed 接线；functional state 输入输出、实际 dtype/shape；保持数组 owner |
| Vulkan | vulkan_operations.cc 的录制/packing/liveness 已完成；vulkan_lowering.cc 只接 pure fixed 图 | 扩 SPIR-V tensor/index/norm 与真实 visibility/state 边界；有界原生程序，不引入通用逐 kernel executor |
| Graph | 通用 Graph 已有 derived shape、资源 effects、alias、独立 invocation；两条原生 lowering 较窄 | 在路线本身落实所需 shape/参数/state；不因 native 不同复制一套逻辑模型图 |

审计继续只核对本表直接源码、已登记 peers 与 C0 固定输入。新依赖引入须同时有 pin/license/三交付。
CPU 通过成熟 SIMD/oneDNN 及受审局部算法；CUDA 优先具名成熟 kernel；Apple 优先 MLX 组合；
Vulkan 优先受审 ggml shaders。各自细节在 AUDITED 时固定，不能把方向表当成已经接线。

## 有限 FIX 草案（尚不授权产品实施）

| ID | 允许改动 | 必需验收 |
|---|---|---|
| C1-01 | 共同 softplus/scan 语义、必要数值/索引/alias 域及普通能力查询 | 真实请求同准备资格一致；invalid/overflow/unsupported 清楚；新增公共面回映需求 |
| C1-02 | Native F16/BF16 tensor/index/norm 与现有 F32 能力复用 | BF16 真实输入、尾块/stride、特殊值、重复索引、独立状态；无全权重浮点常驻 |
| C1-03 | CUDA 同一组能力和默认规则，保留 native stream/capture | 实际 GPU kernel、同精度 reference、准备失败与接纳后 drain；无 host payload fallback |
| C1-04 | Apple MLX typed 普通入口、原生图 lowerings 与状态结果 | 同 runtime GPU region、旧结果保活、两个状态、所需 dtype/shape；无用户 custom adapter |
| C1-05 | Vulkan 同能力的原生 pipeline/录制与数据参数 | 真 SPIR-V 算法、同步/alias/scratch、重复执行与失败；无 empty target/copy 代签 |
| C1-06 | 受影响 Graph shape/parameter/effect/查询与 C++ consumer/指南 | Direct 和 Graph 结果/状态一致；编译不改输入；稳定程序按输入位置/索引重放 |
| C1-07 | 本包逐路线 source/static/shared、正确性、P0/P1、红队与结果回映 | C1 MUST 所有格有实际 receipt，普通用户无需选 Provider；不留到 C7 首次验证 |

## 自然调用与生命周期草图

下面是沿用现有对象的设计草图，新增 schema/query 的精确签名仍待 AUDITED；不是可复制的已交付示例。

```cpp
auto context = compute::cpu::create_context(/* caller executor optional */);
auto op = compute::prepare(*context, compute::RmsNorm{.epsilon = 1e-6F}, input, weight);
// Query/preparation use actual dtype/shape/resource facts and the route's fixed rules.
// Bind native resources to independent invocations, run, observe completion, retain results.
// The Graph consumer adds the same typed operations and compiles once for its route.
```

- Compute 与 Graph 保持现有两入口；用户提供路线、计算、数据/参数，不提供排名或 timing profile。
- GPU 的不同对象/输出 owner 保留；Apple 的 functional state 是输入数组到新状态数组，native
  runtime 保活旧结果；CPU/CUDA/Vulkan 可在声明的独占、alias 和真实完成边界内更新状态资源。
- 位置、有效长度、索引内容等是运行数据；只在合法性确实变化时选择已准备的有限 variant。不把
  token 循环变为重建模型图，不把 MLX 私有内存写成零。
- 热路径只调用绑定程序与必要 native dispatch；无查询、计时、Provider 搜索、每元素 vtable。
- OOM、非法输入与预算在可证明的接纳前拒绝；接纳后错误保留 owner 至终态并 drain，不换 Provider 重跑。

## 预声明验证与性能边界

- correctness：F32/FP16/BF16、真实 2B embedding/norm/gate/residual/index 片段；batch1 prompt
  1/17/128，holdout batch2/23/129；非连续布局、空/one、tile tails、mask 全 false、重复索引、
  NaN/Inf/极端 softplus；独立标量/位级 oracle 与实际参考。
- lifecycle：两 invocation、共享常量、独立可变状态、保留旧结果、非法 alias、cold rollback、
  accepted-prefix failure、关闭/drain。host sanitizer 只签实际观测 host 域，device 工具按路线适用。
- 平台/交付：Windows/WSL CPU 和 CUDA；Air CPU/Apple/Vulkan；Mini CPU/Apple portability；
  source/installed static/installed shared 同一普通组合，每次低并行 build2/CTest1，重型串行。
- P0：逐片段 allocation/copy/launch/reduction、layout 转换、同步和常驻/临时字节。纯数据/文档无 P1。
- P1 bundle：各路线同一 C1 片段，在不可变 candidate、相同 profile、输入/状态、最终完成及输出
  责任下，对直接原生等责组合；有成熟完整融合 peer 时同时核对其语义。normal M=1/17/128、holdout
  M=23/129；冻结 A/A、A/B+B/A，n=15 上限31，5% 工程 material margin。未过 preflight 不进完整采样。
  短操作按真实完整片段聚合；不拿裸 kernel 与拥有完整结果寿命的 Graph 排名。适用快慢方向如实记录。
- 性能纠正只在机制与消费者影响都成立时启动，最多两轮；新候选重回 AUDITED，文档变化不重跑。
- 维护：selector selected gates，closure `--previous-ref`，docs，diff check；工作包交付时统一 LOC。

## 当前裁决

### C1-A：Softplus 共同语义与 Native F32（AUDITED）

SCOPED baseline `f968e6f0`；真实 consumer 是 Qwen DeltaNet 的
`g = -exp(A_log) * softplus(a.float() + dt_bias)` 与 DeepSeek 的 `sqrt(softplus(router_score))`。
两处都实际需要 F32，源于 C0 固定 Torch Qwen 实现与 SGLang
`kernels/jit/csrc/deepseek_v4/hash_topk.cuh:19-21`。正交任务是稳定逐元素数值变换。

| FIX | 固定允许改动 | API / 数值 / 成本与验收 |
|---|---|---|
| A01 | 在既有 Unary 增加 `UnaryKind::kSoftplus` 并更新共同 mode 验证 | beta=1 的 log(1+exp(x))；同 input/output shape/type、既有 floating 域；pure、alias 不变；只追加 enum 值，不加对象字段或新 runtime |
| A02 | Native F32 dense/strided 的 Softplus 实现与差分/别名测试 | 稳定 `max(x,0)+log1p(exp(-abs(x)))`；NaN 传播，-Inf→+0，+Inf→+Inf；dense 复用 SIMD exp/log1p、固定 128-float 栈 scratch，strided 走标量；无堆分配、锁、syscall 或新增 vtable |

承载裁决：`BORROW` 已有 HYLE SIMD 的数学核与 C++ scalar math；复用 Native 的 tensor/index/alias/
completion 机制，不复制 SIMD dispatch。F16/BF16 扩域、其他 Unary/scan/query 和 GPU 路线仍按本卡
后续审计推进，不把当前子集签为 F03.2、C1 或四路线完成。

自然 Direct 为 `compute::prepare(context, compute::Unary{compute::UnaryKind::kSoftplus}, desc)`；
Graph 仍消费 `Operation(Unary{...})`，无模型字段、Provider 排序或新 convenience runtime。
错误、move、并发、teardown、output owner 与既有 Unary 完全相同；空 tensor 不触摸 storage。

当前适用验证：compute_ops schema、Native Direct 的独立 FP64 `log1p(exp(x))` oracle（有限测试输入
不超 double exp 域）、极小负值/大正值/±Inf/NaN、dense128边界与非连续布局/精确 alias；selector
选中的当前 owner 和直接消费者/公开头。三交付与四路线完整片段随 C1 固定候选执行；此子集不跑 P1，
未执行项保持未证明。仅这两个 FIX 已固定，产品 commit 前 owner 必须已在独立提交进入 AUDITED。

### C1-B：Native F16/BF16 基本算术与归一化（AUDITED）

审计 baseline `0449facf`。C0 的实际 320 张量及 Torch reference 已证明：embedding 和 residual
保持 BF16，Qwen 的 norm 在 F32 计算并使用 F32 learned scale，再舍入回输入类型；Q4/Q8 验收还需
F16 激活。源码检查确认 Native 的 Gather/Select/Copy/Concat 已按标量字节宽度实现，应复用；
`elementwise.cc` 的 arithmetic、`normalization.cc` 的 LayerNorm、`native_provider.cc` 的 RMSNorm
目前硬拒绝半精度。现有 SIMD conversion、数学核和 row executor 足以承载本子集，不引入依赖。

| FIX | 固定允许改动 | 数值 / 成本 / 验收 |
|---|---|---|
| B01 | 共同 RMSNorm/LayerNorm 允许 F16/BF16 input 搭配同类型或 F32 scale/bias | 每个参数独立验证 dtype、rank、hidden；输出保持 input dtype；F64/整数及不匹配宽度仍按合同拒绝；这是实际 Qwen 参数类型，不降级权重 |
| B02 | Native Unary/Binary 扩到 F16/BF16，复用已有 dense/广播/stride/alias 机制 | 在 F32 计算、输出 RNE 一次；dense 固定 128 元素分块转换后调用既有 SIMD；strided 使用安全 byte load/store；无整张量展开、堆分配、线程或新 metadata |
| B03 | Native RMSNorm/LayerNorm 扩到 dense F16/BF16 和 B01 参数域 | 明确 F32 accumulation；固定 128 元素栈分块，RMS 两遍、LayerNorm 三遍，复用 SIMD sum/dot 与 conversion；只在最终 affine 完成后舍入；按既有 row executor 分工，exact input alias 安全，旧 F32 核保留 |

承载为 `BORROW` 既有 SIMD + `OWN` 薄的类型/分块接线。定型不新增公共对象或 NativePlan 字段。
半精度 GEMM、GatedActivation、Reduce/Softmax/TopK、Scatter-add 和 scan 尚未由本子集授权；它们
有各自的累加/舍入或状态合同，继续按 C1/C2 的具名 consumer 审计，不将当前接线算作 F01–F04 完成。

普通 Direct/Graph 沿用 Unary/Binary/RmsNorm/LayerNorm；Gemma 的 `1+gamma` 仍通过明确的普通
Binary/常量组成，不引入模型专用参数。prepare 根据 dtype 绑定模板实例，执行不重新选择实现。
生命周期和 completion 复用现有 Native；不改变关闭、并发、failure-prefix 或 allocator 合同。

适用验证：共同 mixed-parameter 推导/拒绝；每个 dtype 的独立 double 数值 oracle 和位级舍入；
全部 Unary/Binary 模式、广播和奇数字节 stride、精确 alias、空/1/127/128/129/257、极值/NaN/Inf；
RMS/LayerNorm 的 F32 与同类型 scale/bias、常量 row、零 epsilon、129 与实际 2048 hidden、两个
invocation 与参数保留；已有 F32 回归和 selector 对应门。C1-A 与 B 共用一次稳定子候选的公开面/
消费者检查，不因文档变化重建。完整模型片段、四路线三交付与 P1 仍按 C1 整包固定候选执行。

### C1-A/B 实施事实

- `0449facf`：Softplus schema + Native F32；W1 MSVC Release/WERROR，compute_ops 与 Native
  correctness 实际通过，极值/stride/alias 已运行。
- `782c575e`：Native F16/BF16 Unary/Binary、RMSNorm/LayerNorm 和 F32 norm 参数；W1 相同
  工具链两个 suite 共 127 case 实际通过。独立半精度 oracle 用数学解码和相邻可表示值搜索，未复用
  产品 conversion；覆盖 0/1/127/128/129/257、2048 hidden、混合参数、两 invocation 与旧结果保留。
- 原始 build/test 日志位于 `C:\hyle-scratch\llm-moe-c1-20260917`。Windows `782c575e` 的 23 个
  selected gates 全部实际 PASS，包括受影响 correctness/header/直接消费者、Compute 与 Graph 各自
  source/installed-static/installed-shared；[完整门清单及原始日志哈希](../../../../evidence/correctness/compute-llm-moe-c1/ab-windows-selected.json)
  保留 95 个 skipped reasons。该证据只覆盖 A/B 与 W1，不覆盖后续 C/D/E 或其他环境。
- C1-C `06d28274`、D `f626fb48`、E `3997c621` 已形成有限实现。`3997c621` W1 Release/WERROR
  编译通过，普通 CPU/CUDA Direct/Graph consumers、CUDA provider 等 6 个 test executable 实际
  PASS；Scan 的两处新测试分别错用了轴错误码和失败 slot 生命周期。`561aea36` 只纠正测试，
  未改产品，compute_ops 35 case + Native 106 case 重跑全部 PASS。联合覆盖的 8 个 test executable
  及初次失败保存在[focused receipt](../../../../evidence/correctness/compute-llm-moe-c1/cde-windows-focused.json)。
- 实际新消费包括 F16/BF16 2048-hidden embedding/norm/gate/residual/scatter/softmax 图的两
  invocation/三次改变输入/旧结果保留，Scan→Gather offsets，BF16 三门控的默认 Direct 与 CUDA
  capture 重放。合成输入只签这些机制，不称为本地完整 2B 权重片段验收。
- 验证副本只在 A/B 全部门结束后前移，复用未改变的依赖 object，产品源码仍由 Windows canonical
  提交提供。Scan 新公开接口的 full selector/其他环境与 C1 固定 bundle 仍未执行，不继承 A/B 的交付。
- `561aea36` 统一 LOC self-check：Compute+Graph 全部实现 **37,929 行 / 131 文件**，不含测试/vendor，
  与 C0 37,035 行口径一致；[原始统计](../../../../evidence/correctness/compute-llm-moe-c1/cde-implementation-loc.json)。
- 无 P1；无 C1 整包、整模型或 GPU 新覆盖 verdict。矩阵仍 UNPROVEN。
- registry 摘要曾仍写 C1-A 限定，现已同步 A/B/C。实际有限 FIX 分别已在 `ffef1a5f`、
  `885f5f3d`、`9181710f` 独立卡提交后才写各自产品；owner 持续 AUDITED，没有状态跳级。

### C1-C：Native 剩余半精度归约与门控（AUDITED）

baseline `782c575e`。真实需求沿用 F02–F04：Torch 的 half attention softmax 在 F32 求值后回到
activation dtype；SwiGLU 在乘法完成后舍入；MoE router 的 TopK/归约/重复索引回收及 residual
需要相同普通入口。已定向读取 `reduction.cc`、`native_provider.cc` 的 gated 路径、`indexing.cc`、
`transformer.cc` 的 Softmax/TopK 与当前公共精度 traits。无需新 Provider 或新依赖。

| FIX | 固定改动与语义 | P0 / 必需验证 |
|---|---|---|
| C01 | F16/BF16 Reduce 的 Sum/Mean/Min/Max，明确 `accumulation_type=F32`；Unknown 的半精度累加域保持拒绝 | dense all/last-axis 128 元素分块；general 按 output/reduced 坐标走一次输入，O(N·rank)，不按 output 分块重扫全输入；最终一次舍入；多轴/stride/empty/NaN/Inf |
| C02 | half packed/separate GatedActivation 的 SiLU/exact GELU/GELU-tanh | F32 activation 与乘法后一次输出舍入；128 元素两块 scratch；复用既有一次 host mapping 与 row executor；无整张量展开、无新增 plan 字段；真实 hidden、尾块、两种输入形状 |
| C03 | half Scatter-add | 沿既有 updates 逻辑行主序，每个加法在 F32 求值并舍入写回 storage；保留 checked indices 在任何 output 写入前拒绝、assign 与 exact input0 alias；重复索引、索引越界拒绝不改输出 |
| C04 | half Softmax/LogSoftmax，Bool 与同类型 additive mask，F32 accumulation | 每 invocation 显式一个 F32 row workspace，冷期 checked size；每 row 转换/应用 mask 后复用完整 SIMD softmax，再一次输出舍入；任意 axis/stride、全 mask、极值、alias、workspace 失败；F32 旧核不变 |
| C05 | half TopK values + i32 indices，既有 stable/NaN/tie 规则 | 复用已有 row F32 workspace 与成熟 top-k；workspace 在 prepare 明确并复用；不展平整个 tensor；任意 axis、tail、sorted/unsorted、重复分数、NaN 与两 invocation |

源码反例收回 B03 的精度检查：旧 F32 RMSNorm 分支未检查 `accumulation_type`，会接受声明 F64
而实际调用 F32 SIMD。当前同 owner 的有限修复必须在冷期拒绝该域，并补拒绝测试；没有改默认 F32
算法或依赖 owner。C04/C05 的零 output 域明确 workspace=0，不能让空工作负载分配整行 scratch。

承载仍为既有 SIMD 与 Native 的薄类型接线。F32 路径保留；精度、mask 或 scratch 条件不满足时准备
拒绝，运行不另找 Provider。当前不改 softmax mask 公共 dtype 合同，不顺手改 RoPE/attention/GEMM，
后者按对应包处理。上述五项的 finite FIX 在产品修改前独立提交；不因 enum 或测试行机械填全目录。

验证沿用 C1 卡，增加具名 double/排序/顺序舍入 oracle 和 workspace/checked-index 失败测试；本次
实现没有公开 API 改动时只跑 owning factor 与 selector 直接消费者。完整 bundle 三交付/P1 仍待 C1。

### C1-D：有界前缀求和（AUDITED）

baseline `06d28274`。F02.4 的 I32 expert counts→offsets 需要 exclusive scan；C3 已登记的
DeltaNet gate chunk 需要 F32 inclusive scan。定向源码为固定 SGLang revision 的
`python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu:103-235` 和
`python/sglang/kernels/ops/attention/fla/cumsum.py:60-70`。前者明确 i32 exclusive-prefix，后者
把 gate 提升到 F32 后沿 chunk 时间轴求和；本子集不引入 MoE packing、ragged 或模型层类型。

现有 `hyle/simd/scan.h` 已有 O(N)、O(1) scratch 的 inclusive/exclusive kernel，并保证整数
modulo 运算无 signed overflow UB。Compute 的 checked overflow 要求更强，需先验证全部整数
前缀再写任何 output，不能直接把 modulo 当作 counts 合同。浮点允许固定路线的并行重结合。

| FIX | 固定实施范围 | 数值 / 成本 / 验收 |
|---|---|---|
| D01 | `compute_ops` 增加 `Scan{axis=-1, exclusive=false}`、kind/name/traits/infer；`compute` 只增加自然 typed `prepare(context, Scan, desc)` | 单输入/同 shape/type dense 输出，rank≥1，正负 axis；本轮共同数值域 I32/F32；inclusive 含当前元素、exclusive 从零开始；整数全部前缀（含最终总和）必须可表示，失败不写输出；pure，可精确 alias input0；不新增对象或执行器 |
| D02 | Native I32/F32 dense/byte-strided Scan | dense 最后一轴复用 SIMD scan；其余按 axis row 顺序遍历，O(N·rank)、O(1) scratch；I32 两遍（先 checked，再写），F32 一遍；无堆分配/模型常驻展开；冷绑定 dtype/exclusive，复用原 invocation/completion；不改 SIMD owner |
| D03 | schema、Direct、默认 Graph consumer 和文档 | counts→offsets→Gather 的真实组合、empty/one/tails、负 axis/中轴/odd byte stride、dense exact alias、正负溢出且 output 不变；F32 独立标量 oracle、NaN/Inf；更新 family 表；四路线具体执行仍分别验收 |

本次新增 schema/自然接口不宣称 GPU 已实现。共享 API 的 selected 消费者/headers/三交付与后续 C1
接口定型候选合并执行，期间未运行项保持未证明；先运行新增 owner 的局部正确性，不运行 P1。
能力 query、CUDA checked-index 完成时错误等仍待各自有限审计，D01 不授权改这些公共运行机制。

### C1-E：CUDA 原生门控与成熟片段扩域（AUDITED）

baseline `f626fb48`。F03.2 的本地 2B BF16 SwiGLU 是实际缺口；固定七模型的激活域还要求既有
typed GELU/GELU-tanh 门控。已经 vendored 的固定 SGLang `activation.cuh` 同时拥有这三种数学核，
当前 HYLE 接线只实例化 F16/F32 SiLU。`sglang_device_compat.cuh` 缺 BF16 include；原生
`native_kernel.cu`、private `native_kernel.h` 与 `cuda_provider.cc` 也用 bool half 限定域。
这是现有片段扩域，不引入新依赖、版本、provider、执行器或 CUDA backend 修改。

| FIX | 固定实施范围 | 承载 / 成本 / 验收 |
|---|---|---|
| E01 | 私有 dtype/activation 冷绑定扩为 F32/F16/BF16 × SiLU/GELU/GELU-tanh；支持合法零 output | 不把公共 C++23 descriptor 搬进 CUDA20 TU；HYLE packed/separate kernel 复用既有逐元素索引，非空 prepare 解析函数与 launch 参数，submit 无模式搜索；零元素 submit 不取 device payload 指针、不发 kernel |
| E02 | 已固定 SGLang packed 完整 16-byte vector 的同三种激活、BF16 实例化 | `BORROW` 原片段，不改 vendor 数学；无 PDL/fast-math/提前 half activation 舍入；BF16/F16 width%8、F32 width%4 及 alignment16 仍为资格条件；默认工厂顺序不变，原 F16/F32 SwiGLU 规则保留 |
| E03 | 数值/生命周期/default factory/Graph 与 user header、指南回映 | 两 invocation、真实 completion、capture 重放、2048/6144/tails、±Inf/NaN/极值、packed/separate、F16/BF16 最终一次舍入；零 output 不需要有效 device allocation；原 tail/不安全 alias/关闭/故障保持可判定 |

现有 Program/Prepared 元数据不因新 dtype/模式增加常态字段；private prepare 参数用具名 enum。
非空路径的 dispatch 数仍为一。空路径不虚报实际发出 kernel；其结构 cost 的 dispatch 项保持未声明。
三交付、默认优选与完整等责 P1 在本包固定 CUDA bundle 执行；当前门控子集没有 CUDA 新覆盖 PASS。
普通索引的运行时越界、其他张量算子、能力查询仍另行审计，E01 不授权改公共 completion 协议。

### C1-F：普通准备即真实能力检查（AUDITED）

baseline `1c55aae7`。本地模型 consumer 需要在分配 activation/状态和执行前确认具体 dtype、shape、
resource 与已选设备可执行，而不只调用 schema inference。源码审计确认现有 descriptor/resource
版 `prepare()` 已通过同一 `propose → prepare_candidate` 完成全部资格检查；成功返回可直接复用的
程序。单独只调用 propose 的布尔查询会误报后续明确 decline 或编译失败的请求；再次 prepare 后丢弃
结果则重复编译。故复用既有普通 prepare，不新增另一套支持列表、查询对象或公共运行机制。

具体缺陷是 `prepare_for_mode` 丢弃所有合法 unsupported 原因，最终只返回通用无 Provider 错误；
用户看不到实际 dtype/layout/资源限制。源码 `operation.cc` 的共同验证、resource match、context
admission 及 explicit decline/error 语义已经足够，不需要 backend 或 Graph runtime 改动。

| FIX | 固定实施范围 | 验收 / 成本 |
|---|---|---|
| F01 | `compute` 普通 shared/delegated prepare 在全部候选不支持时保留具名 Provider 的实际拒绝原因 | 优先已有完整 proposal 的 prepare decline，否则保留固定规则顺序中最后一个 proposal unsupported；资源错误/编译失败/其他 fatal 原样返回，不增加重试；只在最终失败追加 Provider 名，成功路径不拼接文本；不新增常态对象字段或执行期开销 |
| F02 | 公开头/指南说明 descriptor prepare 是实际能力检查和程序绑定；普通 CPU/CUDA 与必要故障测试 | 成功后保留程序用于执行，不要求 payload/Queue；dtype/shape/resource/closed 真实拒绝；propose 不支持和 prepare 明确 decline 均可诊断，fatal Unimplemented 不被当作 fallback；准备不映射输入或执行 kernel |

Apple/Vulkan 继续使用各自原生编译/准备对象，在本卡对应路线审计中落实同等普通旅程；本子集不强迫
它们共享 Compute Context，不预签其能力。现有 Graph compile 已绑定真实程序，本次不改图规划。
验证只运行受影响 compute/backend/custom/Native 与默认 CPU/CUDA 消费者；共享 API 的完整交付
继续按 C1 定型候选执行。无新 API、无新依赖、无 P1，C1 整包仍未完成。

### C1-G：CUDA 基本张量与数值运算（AUDITED）

baseline `23297530`。真实 C0 模型和 F01/F02.3/F03 需要 BF16 residual、gate 广播、F32 softplus、
转换与 head-interleaved view。定向审计现有 CUDA provider/native kernel、共同 View alias helper，
以及固定 ggml 的 `src/ggml-cuda/unary.cu` / `binbcast.cu` / `cpy.cu`：成熟实现同样以 FP32 数学加
末次 half 转换、显式 stride/broadcast 实现这些机制。ggml 的四维 ggml_tensor/PDL 接口和部分数值
边界（ReLU NaN、Softplus 阈值）不等同 HYLE rank8/稳定数学合同，直接接入不能免除适配。
承载裁决为 CUDA 工具链 math/half 原语加 HYLE 薄张量接线；不移植其整套 tensor runtime，不改成熟
门控片段，也不基于尚未运行的性能宣称优于成熟实现。完整等责 bundle 仍是本包固定验收项。

| FIX | 有限实施范围 | 数值 / 生命周期 / P0 / 验证 |
|---|---|---|
| G01 | HYLE CUDA provider 的 View/Copy/Fill/浮点 Cast | plain View 复用共同 byte-offset，submit 无 device payload；Copy/Fill 按1/2/4/8字节精确搬运，Cast 仅 F32/F16/BF16，RNE 一次；整数 Cast 和 FP8 运算另按已冻结需求资格；dense 输出，rank≤8，非负/zero/奇数 byte stride、empty/one/tail；Copy/Cast 不允许重叠 |
| G02 | 所有已公开 Unary 与 Binary 模式、Select 的实际 F32/F16/BF16 消费域 | FP32 数学、最后一次舍入；stable sigmoid/softplus、ReLU/GELU/SiLU 特殊值与 Native 独立数学合同一致；Min/Max 为 fmin/fmax；Select 按 Bool mask 保留所选值原始位；rank0 和尾维广播，Select/Copy/Fill 同时支持 I32 等 plain storage 位搬运 |
| G03 | 私有 CUDA tensor 程序、冷期 dtype/mode/layout kernel 绑定与默认接入 | 不扩大既有 gated Prepared/Program，不增加执行器/Provider；每个非空原子一次 GPU launch、零 workspace/slot私有状态，无 host payload fallback/copy；dense 直接索引，general O(N·rank) byte-safe 索引；仅逐元素且 input desc 精确等于 output 时允许原位；空输出不取 device pointer；CUDA capture 与真实 Completion 保留 |
| G04 | Direct/default CUDA Graph 重放、数值/边界/alias/资源/文档 | 三 dtype 的独立 double/位级 oracle，真实2048与6144宽度、0/1/127/129/奇数stride/广播/NaN/Inf；two slots/三次变更输入/旧结果保留，非法重叠拒绝、关闭、default factory binding 和 capture；已有门控回归；三交付/P1 随固定 C1 bundle |

CUDA tensor count 冷期限定 ≤INT32_MAX（View 不读取 payload）；所有 byte span 仍由 TensorDesc/共同
binding 验证。input metadata 只在准备阶段参与索引计划，执行直接调用已绑定 native function；不传
公共 C++23 descriptor 到 CUDA20 TU。新源码仅加入既有 owner 的 source list，不改公共构建治理。
Concat、Reduce/norm、Softmax/TopK、checked indices/Scan 不由本子集授权，仍保留明确缺口。

### C1-F 实施事实

`23297530` 已实现准备拒绝诊断和 descriptor 能力旅程。W1 MSVC Release/WERROR 编译及 21 个
selected-factor test executable 全部实际 PASS，包含普通 CPU/CUDA、Graph、Native/custom/provider、
CUDA accepted-prefix/late-failure 与 Vulkan 直接消费者；[原始命令及日志哈希](../../../../evidence/correctness/compute-llm-moe-c1/f-windows-focused.json)。
无 payload 的真实 resource 资格、关闭拒绝和 fatal Unimplemented 不被 fallback 均已运行。
未跑其他环境、public header/三交付增量或 P1；本证据不签 C1 整包。

### C1-G 实施事实

- `0408da5d`：CUDA View/Copy/Fill/Cast/Unary/Binary/Select 的 native 程序、默认 Provider 接线，
  独立半精度 oracle、资源/别名/metadata OOM、普通 Direct 与 Graph 两槽 capture consumer。
  最初编译仅测试 helper 的 `Status/ResultError` 三元表达式有歧义，`23e4da3d` 显式返回纠正。
- 实际差分发现 Unary GELU/GELU-tanh 在负尾部的 `1+erf/tanh` FP32 相消；`7232228b` 改用等价
  erfc/stable sigmoid 公式，原数值门限不变。没有改原有成熟 gated 片段、公共语义或运行对象。
- `7232228b` W1 Release/WERROR 编译及 provider 13 case、普通 Direct 4 case、Graph 6 case
  全部 PASS。包括 BF16 交错 slice→Copy→F32 Softplus→BF16→residual→广播 mask 的
  129/2048/6144 宽度，两 invocation、三次变更输入、保留另一槽旧结果、原生 capture；Direct
  复用同一 byte-strided Copy 程序，热重放无 metadata allocation；在途 owner 保留/关闭、冷期
  allocation failure rollback 均实际运行。[原始记录与两次问题](../../../../evidence/correctness/compute-llm-moe-c1/g-windows-focused.json)。
- 统一 LOC `--self-check`：Compute+Graph 全部实现 **38,582 行 / 135 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/g-implementation-loc.json)。
- 新程序不扩大既有 gated Prepared/Program；每个非空 tensor 一次 native launch，无 workspace/
  invocation device storage/host payload fallback；View 不发 kernel。新增 kernel/模板 code storage
  由 CUDA driver 管理，不写成零 resident cost。完整 P1/三交付/其他环境与真实权重片段仍待 C1。
- 下一有限审计为 CUDA norms/reduction/index 及各自 native 失败边界，不把当前 pointwise 证据
  扩张为 F01–F04 或 C1 整包完成。

### C1-H：CUDA RMSNorm 与 LayerNorm（AUDITED）

baseline `326cd5f4`，owner `/root` / `compute_provider_cuda`。F04.1/F04.2 及本地 Qwen 权重要求
BF16 activation 与 F32 learned scale，Gemma 的 `1+gamma` 已能由 G 的 Binary 构成；FP16 量化激活
需要相同接口。已核对固定 SGLang `kernels/jit/csrc/elementwise/rmsnorm.cuh` 的同 dtype weight
约束和 ggml `src/ggml-cuda/norm.cu` 的 F32 路径；它们都按行并行归约，不能直接签 mixed F32 参数。
ggml 的 LayerNorm 使用 E[x²]-E[x]²，在大均值小方差域不满足本卡稳定数值验收。

本子集 `BORROW` 已依赖 CUDA Toolkit 13.3 SDK 的 Cooperative Groups tile reduction 与 CUDA
浮点转换，`OWN` 薄的 norm 公式和 descriptor/resource 接线；不复制 SDK 源码、不增加供应商 runtime。
具体已读 SDK `cooperative_groups/details/reduce.h` 的 multi-warp tile collective：由 SDK 管理
归约的线程间同步与片上 scratch；不再自写 warp/shuffle 归约。精度、带宽和最终生命周期的完整 peer
比较仍属于 C1 bundle，不将当前承载决定或 correctness 冒充最快规则验收。

| FIX | 固定实施范围 | 数值 / P0 / 适用验收 |
|---|---|---|
| H01 | 既有 RmsNorm/LayerNorm 在 HYLE CUDA provider 的 cold proposal/prepare 与原生程序 | plain dense F32/F16/BF16，最后一轴，输出同输入；scale/bias 各自允许同输入 dtype 或 F32（遵守共同 schema）；仅 F32 accumulation；exact shape/output count≤INT32_MAX；zero output 不取 device pointer或launch；不改变共同 API/backend/default factory |
| H02 | SDK 256-thread tile 按行归约，FP32 affine 后一次 RNE | RMS 两遍读输入；LayerNorm 先 FP32 mean，再归约 (x-mean)²，三遍读输入；无整张量 FP32 展开、host payload copy、workspace 或 invocation 私有状态，每个非空 norm 一次 native launch；只有 input0 精确别名可原位，参数/部分重叠拒绝；dtype/kernel 准备绑定，capture/completion 保持 |
| H03 | 普通 Direct/Graph、独立数值与生命周期/文档 | widths0/1/127/128/129/2048/5120/6144/7168，FP64 两遍 oracle、half 位级转换、F32 参数不能先被窄化、零 epsilon/常量行/NaN/Inf、大均值小方差；两 slots/变更输入/保留结果、原生 capture、metadata OOM 回滚/accepted owner 保留/close；局部 correctness 后，完整三交付/P1 随固定 C1 bundle |

本次不接入整套 SGLang tensor/TVM，也不拆其已有 fused 门控。非连续 norm 显式使用普通 Copy
materialize，转换成本计入实际消费者；不声称新 norm 覆盖任意 layout。其余 Reduce/L2、Softmax、
TopK、Concat、checked index/Scan 仍待各自有限审计。有限实现与必要测试只改当前 CUDA owner 及
直接 consumer 测试/指南，公共治理和性能仪器不动；没有本轮适用 P1 verdict。

### C1-H 实施事实

- `3cfc1e71` 实现 mixed-F32 参数的 CUDA RMS/LayerNorm；`418b3217` 只给 norm CUDA TU
  添加 SDK collective 所需 MSVC 标准预处理器选项。前一次编译被 SDK 的传统预处理器检查拒绝，
  没有运行测试；固定后 W1 Release/WERROR 编译、provider 16 case、Direct 4 case、Graph 7 case
  共 **27 case PASS**。[命令、原始日志哈希与初次失败](../../../../evidence/correctness/compute-llm-moe-c1/h-windows-focused.json)。
- 实际数值包括三 dtype、F32 scale/bias 不能提前窄化、大均值小方差、空行/NaN/Inf/精确原位；
  BF16 Graph 的 F32 `1+gamma` → norm → F32 swish gate → BF16 两槽 capture、三次变化输入、
  另一槽旧结果保留和共享参数不变通过，热重放无 metadata allocation。cold OOM 回滚和在途
  norm owner 保留至完成实际通过；没有重写 backend 或已有门控。
- 统一 LOC self-check：Compute+Graph 实现 **38,923 行 / 139 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/h-implementation-loc.json)。无新 P1、三交付或其他
  环境 verdict；合成真实形状不是完整本地权重片段。矩阵仍 UNPROVEN。

### C1-I：CUDA Reduce 与 masked Softmax（AUDITED）

baseline `418b3217`，owner `/root` / `compute_provider_cuda`。F04.3 的 Qwen DeltaNet head128 L2
需要 F32 平方和；F04.5 与 router/attention 概率要求按轴 F32 累加、Bool/additive mask 和稳定
Softmax。已定向核对共同 schema、Native `reduction.cc` / `transformer.cc`，以及固定 ggml
`456172ec` 的 `src/ggml-cuda/sumrows.cu` / `softmax.cu`。后者的行归约、F32 片上行缓存有实际成本
依据，但其 F32-only、四维对象、sinks/alibi 和异常数值语义不能直接替代当前合同。

`BORROW` 现有 CUDA SDK Cooperative Groups collective、math/half 原语；HYLE 只承载薄的
dtype/axis/byte-stride/mask 公式和程序绑定。SDK less/greater 并不传播 NaN，Min/Max combine
必须显式保留共同 NaN 语义；不自写 warp/shuffle，不引入另一个 tensor runtime。

| FIX | 固定有限范围 | 数值 / P0 / 必需验证 |
|---|---|---|
| I01 | HYLE CUDA Reduce Sum/Mean/Min/Max，F32/F16/BF16，任意合法 axes/keepdim/byte stride | F32 累加，half 必须显式声明；输出 dense 同类型且不与输入重叠；每 output 一 SDK tile，dense suffix/all 直接索引，其余 O(N·rank)，一次 launch、零 caller workspace；空集合 Sum=0/Mean=NaN/Min=+Inf/Max=-Inf，任一 NaN 传播；rank0、显式空 axes 拒绝、负轴、多轴、奇数 stride 和真实宽度 |
| I02 | Softmax/LogSoftmax，同三 dtype，任意 axis、scale、Bool 或同 dtype additive 广播 mask | F32 max/exp/sum 后一次 RNE；Bool false 覆盖输入 NaN 为 -Inf，无有效值的行和非有限归一化产生 NaN；dense output，仅输入0与输出描述相同的精确 alias；≤8192 列的行使用最多32 KiB F32片上缓存，较长行三遍读取、无整张量浮点展开或 caller workspace；SDK 两次 collective、一 CTA/row、一次 launch |
| I03 | 默认 Direct/Graph 接线、生命周期、数值/成本与指南 | FP64独立 oracle，0/1/127/129/2048/8192/8193及长行，FP16/BF16、mask broadcast/全false/NaN/Inf/极端logits，任意轴/原位；BF16→F32 L2完整公式和masked概率的两槽capture/变更输入/保留结果；准备拒绝、OOM回滚、accepted owner/drain；完整三交付/等责P1仍在C1固定bundle |

input/output 非空 element count≤INT32_MAX；空归约有非空 output 时必须写真实 identity，不能
当作 no-op，也不能取得空输入的 device pointer。shape/dtype/kernel/cache 布局冷期绑定，执行
不重新搜规则、不计时；现有 gated/norm Prepared 不扩大。源码列表和 MSVC collective TU 选项
只在当前 owner 扩展。长行的三遍读、片上缓存的占用和所有 Graph Cast/临时结果均计入完整 P0/P1；
未比较不声称默认选择最快或已完成四路线优选。TopK、Concat、checked index/Scan 不属本子集。

I 的第一轮数值反例（`45edb0e4`，W1）：默认 Direct/Graph、NaN/Inf 与生命周期已通过；Reduce
测试在 `Axes::from({}).value()` 抛异常，因为共同合同本就拒绝显式空列表，纠正测试与上表即可，
不改公共 schema。BF16 LogSoftmax 对概率接近1的输出约 -4.34e-7，旧 `log(sum(exp))` 在 F32
先把 1+tail 舍入，变成约 -4.77e-7。固定本轮唯一数值修复为 LogSoftmax 的 SDK argmax（值+i32
列号，8字节）和排除一个最大项后的 F32 tail sum，再算 `log1p(tail)`；仍两次 collective、同一
cache/launch/workspace合同，普通 Softmax 路径不变，非有限最大值仍全行 NaN。原独立门限不放宽，
补近1概率与重复最大值回归；不采用 F64 常态求和或扩大到其他 owner。

### C1-I 实施事实

- `2ddc093d` 接入 CUDA Reduce/Softmax/LogSoftmax；`45edb0e4` 补齐该 TU 的 CUDA constants
  include；`c4a0f216` 完成上面的 LogSoftmax 定向修复和测试纠正。原门限不变，没有借 host
  payload fallback、F64 累加或增加额外 dispatch 通过数值门。
- `c4a0f216` W1 Release/WERROR 实际 provider 22 case、Direct 4 case、Graph 8 case 共
  **34 case PASS**。[三次原始运行、失败原因和最终日志哈希](../../../../evidence/correctness/compute-llm-moe-c1/i-windows-focused.json)。
  Reduce 的多轴/rank0/rank8/odd stride/空 identity/huge-empty、Softmax 三 dtype/任意轴/
  mask/原位/8192与8193边界/32769长行/非有限值与近1概率回归均实际运行。
- 默认 BF16→F32 平方和→epsilon→rsqrt→L2 与 BF16 masked probability Graph，M=1/17/23、
  head128，两槽原生 capture、三次变化输入和mask、保留另一槽旧结果均通过；Direct Reduce
  和 Softmax 复用同实现程序，结果与 Graph 一致；独立 FP64/位级 oracle 另签数学正确性。
  metadata OOM 回滚、已接纳操作的 owner 在关闭后保留至完成、热重放无 metadata 分配已通过。
- 统一 LOC self-check：Compute+Graph 实现 **39,552 行 / 143 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/i-implementation-loc.json)。SDK 的片上 scratch/
  row cache 与 driver code storage 如实保留成本；零 caller workspace 不等于零临时硬件资源。
- 当前只签 W1 局部正确性。完整 C1 权重片段、其他环境、三交付/公开头增量、默认优选与 P1
  仍未运行，矩阵保持 UNPROVEN；不把当前 34 case 扩成整包或四路线覆盖 verdict。

### C1-J：CUDA TopK 的 values/indices（AUDITED）

baseline `c4a0f216`，owner `/root` / `compute_provider_cuda`。F04.6 与冻结七模型路由实际要求
E=256/288/384/512/896，k=1/6/8/10/16/E，F32 scores 和 I32 indices；后续量化激活的同类型
片段也需 F16/BF16。普通入口沿用 `TopK`，同时返回 values/indices；不添加 MoE/model facade。
正交域为非连续和中轴数据选择、smallest、重复分数与完整排序，不把单一 E 做成公共形状。

定向源码：固定 ggml `456172ec` 的 `top-k.cu` / `argsort.cu`，固定 SGLang `06992c92` 的
`kernels/jit/csrc/elementwise/fast_topk.cuh`。ggml 的 CUB DeviceTopK 分支明确要求非确定/无序且
逐 row 提交，SGLang fast-topk 只输出无序 indices，均不直接满足稳定 tie、完整 values 和 NaN
合同；其 radix/SDK 承载可借鉴，不复制 peer runtime 或牺牲共同语义。CUDA Toolkit13.3 自带
CUB `300304` 的 BlockRadixSort 提供线程块内排序，现有 SDK collective 承载 k=1 选择。

| FIX | 固定范围 | 语义 / P0 / 必需验证 |
|---|---|---|
| J01 | 私有 CUDA TopK proposal/prepare/program，F32/F16/BF16 input、同 dtype values、I32 indices | 任意 axis/普通 byte stride，dense 两输出，无任何输入或输出相互重叠；NaN 总在有限值/Inf 后，同分含±0按小列号优先，largest/smallest，允许 sorted=false 仍返回排序结果；输出值保留源 storage bits；rank≥1，k合法、INT32_MAX非空input/output计数；非axis空域无launch |
| J02 | k=1 复用SDK带索引选择；k>1 的行长≤4096复用CUB BlockRadixSort | HYLE只编码数值顺序+反向列号的64位key（NaN/zero/padding显式处理），CUB排序后解码列号并精确读取选中值；一次CTA/row和native launch，零caller workspace/slot私有状态；256threads、每线程1/2/4/8/16项按容量冷绑定；kernel与dtype/axis资格执行期不再匹配；不重写SDK排序/归约 |
| J03 | 独立稳定标量oracle、默认Direct/Graph、生命周期与文档 | 三dtype、实际E/k、1/127/129/897/4096尾块、largest/smallest、sorted/stable flags、all equal/NaN/±Inf/±0、任意轴/odd stride/empty；values与indices都验，Softmax→TopK→Reduce等普通组合，两slots/capture/变化输入/保留结果、OOM/accepted owner/close；完整等责P1和三交付随C1固定bundle |

4096 是本子集 block-sort 容量边界，不能伪装为覆盖任意长序列的 sparse-attention selection。
后者按 C3 的真实大 N/大 k 片段定向审计接入成熟选择程序；当下普通 prepare 必须明确拒绝超界
k>1 请求。k=1 可按一次全行SDK选择处理更长行。CUB/SDK源码不复制，不新增公共依赖/runtime；
编译只在 CUDA owner source list/该TU的MSVC预处理选项。完整 P1 将计入 key 排序和读回 values
的成本，当前没有“已优于成熟库”的结论。Concat、checked index/Scan 与其他 owner 不由 J 授权。

J03 的默认接线发现当前同 owner 的具体反例：SGLang adapter 在判断 operation family 前把
`outputs.size()!=1` 当作 binding mismatch，合法的两输出 TopK 会提前 fatal，阻止后面的 HYLE
provider。固定修复仅令 proposal binding 数量与 `request.outputs.size()` 比较，随后沿现有
`select_shape` 对非 gated family 返回 unsupported；cuBLASLt/Marlin 已是这个正确顺序，不重开。
默认 Direct/Graph TopK 和原门控回归必须同时通过，不能只用显式 HYLE provider 绕过此缺陷。

### C1-J 实施事实

- `6feef13b` 完成 SDK k=1 选择与 CUB 块排序、原始 values/I32 indices、默认多输出 fallback；
  `32deca71` 只补独立排序 oracle 的 `<algorithm>` include。前一次产品 CUDA/C++ 编译成功，
  测试编译缺头，未运行测试；修正后 W1 Release/WERROR 的 provider 25 case、Direct 4 case、
  Graph 9 case 共 **38 case PASS**。[原始日志/哈希](../../../../evidence/correctness/compute-llm-moe-c1/j-windows-focused.json)。
- 三 dtype 的独立 stable scalar ordering 和原始位比较实际通过；包括实际 E/k、4096容量、
  k=1 的65537长行、rank8/中轴/odd与zero stride、全NaN/payload/±0/±Inf、双输出与输入之间
  的各类重叠拒绝。默认 F32 Softmax→TopK→选中和→归一化权重图覆盖 M=1/17/23、五组真实
  E/k、k=1，两槽 capture/三次变更输入/保留旧values和indices；普通 Direct 跑同一程序且两输出
  与图一致，热重放无 metadata 分配。TopK cold OOM 回滚和关闭后已接纳 owner 保留也通过。
- 统一 LOC self-check：Compute+Graph 实现 **39,959 行 / 147 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/j-implementation-loc.json)。无适用完整 P1、三交付/
  公开头增量或其他环境 verdict，C1 整包和矩阵仍未证明。C3 大 N/k 的选择域不会继承当前容量证据。

### C1-K：CUDA Concat（AUDITED）

baseline `32deca71`，owner `/root` / `compute_provider_cuda`。F01 的 slice/split/reassemble 和
状态窗口 append 需要真实张量拼接；共同 `Concat` 已有任意合法输入数量、正负 axis、同 dtype/
非axis维相同的 schema，Native 已按 exact storage bits 拼接。已读固定 ggml `456172ec` 的
`src/ggml-cuda/concat.cu`：dense 双输入一遍读取/写出，非连续显式 byte stride；HYLE 保持同样
成本结构，适配 rank8、变参输入和已有资源合同，不复制 ggml 的四维对象/PDL runtime。

| FIX | 固定实施范围 | P0 / 数值 / 适用验收 |
|---|---|---|
| K01 | HYLE CUDA Concat cold proposal/prepare，plain1/2/4/8字节同类型storage、dense输出、合法axis/rank与变参数量 | bit exact，无算术和格式窄化；output≤INT32_MAX非空元素，无输入输出重叠；每最多4输入一组私有已编译程序，常见双输入一次launch，任意N为ceil(N/4)组；不扩大已有tensor/gated元数据、不新增API/Provider |
| K02 | 原生分组拼接kernel，dense直接索引与general byte-safe索引 | 每组输出分片一次读/写，组内最多4个边界匹配，O(elements·rank)，无O(elements·input_count)扫描；所有shape/stride/axis前缀和kernel在冷期绑定，热路径只取得至多4个资源指针并launch；空组跳过指针/launch，零caller workspace/slot状态；group metadata按实际arity冷分配且如实计persistent bytes |
| K03 | Direct/default Graph、数值和生命周期/文档 | 1/2/3/4/5/9输入，F32/F16/BF16/I32与1/8字节位模式，真实head/hidden宽度、0/1/tail、rank8、中轴/odd/zero stride、部分为空/全空/共享输入；View→Concat与窗口拼接，两slots/capture/变更输入/保留结果，alias拒绝、cold OOM、已接纳prefix错误后drain；当前局部correctness后，完整三交付/P1仍随C1固定bundle |

分组数量与dispatch成本冷期计算；若某组已提交后后续指针/launch失败，必须报告已接纳prefix的
真实Completion并保留owner，不改成同步拒绝或重跑。所有组写不同output片段，不需要组间额外
barrier或独立Queue。仅 CUDA owner/source list 与直接consumer测试/指南可改；checked indices/
Scan 的异步算法错误回报不属于 K，也不以 unchecked 或 host payload fallback 冒充完成。

### C1-K 实施事实

- `529411e6` 完成 CUDA variadic Concat；W1 Release/WERROR 编译、provider 28 case、默认 Direct
  4 case、Graph 10 case 共 **42 case PASS**。[原始命令/日志哈希](../../../../evidence/correctness/compute-llm-moe-c1/k-windows-focused.json)。
  此次候选一次通过，未追加性能采样或改变既有算子门限。
- 1/2/3/4/5/9 输入、F32/F16/BF16/I32/U8/I64 原始位、rank8/任意轴/odd byte stride、zero-stride
  广播/共享输入、前后整组为空/全空/huge-empty 实际通过。默认 BF16 View→五输入 Concat 与双输入
  窗口 append 图覆盖 M=1/17/23、width128/129/2048，两槽 capture、三次变化输入、另一槽旧结果保留；
  普通 Direct 与独立 host 拼接 oracle 都逐位一致，热重放无 metadata 分配。
- 九输入程序/组数组/运行槽逐次 metadata OOM 回滚，已接纳九输入操作在 close 后保留 owner 至完成
  已通过。另在既有 prevalidated provider/Graph seam 定向注入第二组指针错误：第一组确实已提交，
  返回 pending 的真实 Completion，待其终态报告 InvalidArgument，仅第一组目标片段被写入；借用
  程序/槽保留至完成。没有通过非法 GPU 访问/device assertion 破坏 CUDA context。
- 统一 LOC self-check：Compute+Graph 实现 **40,385 行 / 151 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/k-implementation-loc.json)。完整 C1 权重片段、公开头/
  三交付增量、其他环境及等责 P1 尚未运行，矩阵保持 UNPROVEN。

### C1-L 实施事实：CUDA 异步算法错误依赖

[独立依赖卡](llm-moe-c1-cuda-device-status.md)已实现 prefix 错误字、完成边界解释、跨 Queue
保活与 first-error 合并、捕获独立状态，未修改共同 API 或其它路线。最终候选 `21ff9fe0` 的 W1
backend/provider/default Direct/Graph 共 **79 case PASS**（新增机制10项）；
[原始尝试与失败记录](../../../../evidence/correctness/compute-llm-moe-c1/l-windows-focused.json)。
一个冷期扩池/外部 capture 顺序缺陷已在两轮产品纠正预算内解决；并发测试移除 SDK 明确禁止的
host callback 对未来独立 GPU 工作的等待，保留原错误字、提前观察/丢弃与40次池重用的确定性验证。
这只是实际 checked Gather/Scatter/Scan 的基础机制，截至 L 记录三者尚未接入。后续在本 owner 卡继续固定
有限算子 FIX；完整 C1 权重片段、三交付/公开头增量、其它环境与 P1 没有新的 PASS。
统一 LOC 为 Compute+Graph **40,733行 / 154文件**，不含测试；
[统计](../../../../evidence/correctness/compute-llm-moe-c1/l-implementation-loc.json)。

### C1-M：CUDA checked Gather（AUDITED）

baseline `59190011`，当前 owner `/root` / `compute_provider_cuda`。F02.1、F02.4 的 Gather 部分与 F19 的真实输入是
本地 BF16 embedding table、每次变化的 token IDs，以及 routing/state 的重新索引。主路径为
Gather→mixed-F32 RMSNorm→residual；正交域为任意 axis、I64/scalar/多维 indices 与 byte strides。
现有共同 schema 与 Native 数学合同已满足，C1-L 提供设备异步错误来源；本次不重开共同层/backend。

已读本地固定 ggml `456172ec733a135778adcd32d00e576a58232e45` 的
`src/ggml-cuda/getrows.cu`：普通同类型读取按连续后缀、可用16-byte向量搬运；其直接设备索引、
四维/I32与可选类型转换不满足本轮任意axis/rank8/I64/逐位保留/可恢复checked错误合同。
`BORROW` 连续行搬运的成本结构，`OWN` 薄的描述适配和checked前置验证；不复制其runtime/量化路径。

| FIX | 固定范围 | P0 / 必需验证 |
|---|---|---|
| M01 | HYLE CUDA Gather cold proposal/prepare/program，plain1/2/4/8字节同类型输出、I32/I64 indices、rank≤8任意axis、dense输出 | 输出与所有输入无重叠；indices可为rank0或多维、source/indices可odd或zero byte stride；非空output/indices≤INT32_MAX，source索引/offset保留64位，不按表总元素数无故截断；空output不取指针/launch，非空output的空source axis为静态OutOfRange |
| M02 | 真实GPU全索引验证后Gather，复用L01–L03 | checked先一次O(indices·rank)验证kernel，再一次O(output·rank)读取；坏值atomic first-error且所有output不变，无payload回传/host同步/device assertion。unchecked仅调用方保证索引合法时省验证/状态命令。连续source后缀可整除16字节时冷绑定16-byte搬运单位，非对齐基址在同kernel内按byte安全读取；其余按原标量位宽，无数据转换。shape/dtype/kernel冷绑定，零caller workspace/槽私有状态；reset/4B回报/capture merge另外按L卡如实计费 |
| M03 | 普通Direct/default Graph、独立位级oracle、失败和资源验证/文档 | BF16/F16/F32及1/8字节payload、I32/I64负值/等于extent/巨大值、scalar/多维/任意axis/rank8/odd/zero strides/empty、宽度128/129/2048等尾块、16-byte非对齐地址；prepare不消费输入、alias拒绝、checked坏索引输出不变、failed槽退役和fresh槽恢复；两captured槽/多次变更索引/保留旧结果、混合norm/residual、OOM/accepted owner/close/abandon、无热metadata分配；完整C1真实权重/三交付/其它环境/等责P1仍随固定bundle |

只扩当前CUDA owner/source list与直接consumer测试/指南。checked比较对象必须承担同样的索引验证、
失败回报、同步和结果寿命，不能用unchecked裸getrows给它排名；当前没有性能胜出结论。Scatter
重复更新的顺序/半精度舍入和Scan全部prefix溢出另行固定有限FIX，不把Gather机制代签二者。

### C1-M 实施事实

- `caa7f33b` 完成真实 checked CUDA Gather、默认 Direct/Graph 接线和指南；`a60adc0b` 只把
  Graph 测试的 BF16 比较改为显式 storage bits 比较，因为该类型没有 `operator==`。首次产品
  CUDA/C++ 已编译，Graph 测试编译失败且未运行；修正后 W1 Release/WERROR 的 provider 32 case、
  Direct 4 case、Graph 12 case 共 **48 case PASS**，未改变产品算法或数值容差。
  [原始尝试、命令与日志哈希](../../../../evidence/correctness/compute-llm-moe-c1/m-windows-focused.json)。
- U8/F16/BF16/F32/I64 原始位、I32/I64 索引、scalar/多维/rank8、中轴/负 axis、odd/zero stride、
  width1/128/129/2048、16-byte 非对齐向量地址及输出保护区实际通过。source axis 为 `2^40`、
  索引超过 INT32_MAX 的 zero-stride 合法输入也保持完整 I64 地址域。257 索引尾块中的负值、
  等于 extent 及整数极值均报告 OutOfRange，整个输出不变；failed 槽退役，fresh 槽恢复，空输出
  不取 payload，静态空 source axis 与输入输出重叠拒绝均通过。
- 默认 BF16 Gather→mixed-F32 RMSNorm→residual 在 M=1/17/23/128、width128/129/2048 下，
  两个 native captured 槽、三次变化 indices、保留另一槽旧结果均通过。普通 Direct 同程序结果
  与 Graph 逐位一致，另由独立 host Gather/FP64 norm oracle 核对；热重放无 metadata 分配。
  坏索引存在时 capture prepare 不执行 payload；pending 错误在程序/槽释放、context close
  或 Completion 丢弃后仍保留 owner 至真实完成，cold metadata OOM 逐点回滚检查通过。
- 统一 LOC self-check：Compute+Graph 实现 **41,186 行 / 158 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/m-implementation-loc.json)。selector 的
  109 项 skipped reasons 已保留；当前仅完成 configure 与上述三个 factor。完整 C1 权重片段、
  公开头/三交付增量、WSL/macOS 与等责 P1 尚未运行，矩阵保持 UNPROVEN。Scatter/Scan 仍待实施。

### C1-N：CUDA 有序 Scatter（AUDITED）

baseline `5c27e6ea`，owner `/root` / `compute_provider_cuda`。F02.2/F02.4/F19 的实际消费是
token 状态写回、permutation 和重复目标的累加。共同 schema 与 Native `indexing.cc` 已固定
逻辑行主序：assign 最后项覆盖，add 每一步 F32 求值后舍入到 storage；精确 input0 alias 可选，
checked 坏索引在任何 output 写入前报告。此合同不能换成无序 atomicAdd。

已读固定 ggml `456172ec` 的 `src/ggml-cuda/set-rows.cu`：普通路径是每元素独立直接写入，
适合唯一目标，却不提供本轮重复 assign/add 的顺序和 checked 失败合同。另一具名来源为本机
CUDA 13.3 的 `cccl/cub/device/device_radix_sort.cuh`：SDK 明确提供 stable SortPairs 和显式
临时存储。`BORROW` CUB 稳定排序；`OWN` byte-stride 适配、全索引检查和每目标有序更新，
不重新实现排序。排序只处理 indices/ordinal，不排序或展开整份 updates/状态。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| N01 | CUDA Scatter proposal/prepare；assign plain1/2/4/8字节原始位，add F32/F16/BF16；I32/I64索引、任意axis/rank8/odd与zero stride、dense output | 非空output/updates/indices计数≤INT32_MAX，空output无payload/launch；仅同描述input0精确alias，其他重叠拒绝；空indices只保留source；checked坏值所有output不变、failed槽退役/fresh恢复 |
| N02 | 单索引冷绑定无排序路径，多索引CUB stable SortPairs＋按目标分组有序更新 | 单索引零workspace：checked验证后一次source→output/更新kernel；多索引先pack/验证，再SDK排序，再必要source复制和每组更新。每个目标单writer，add逐次F32并storage舍入、assign取最后ordinal；O(output+updates+SDK索引排序)，不扫描output×indices，不用axis大小的索引表；精确alias跳过整状态复制 |
| N03 | 显式caller workspace与每invocation固定的SDK排序程序 | workspace为四个I32索引/ordinal数组加SDK查询的scratch，256B对齐、checked size；冷create_invocation捕获并实例化排序，不读payload。普通Direct launch该制品；外层Graph capture用原生child graph和capture依赖接入，热执行不再调用CUB的选择/准备。两槽独立workspace/graph，opaque CUDA graph内存如实unknown；接纳后失败保留槽/图到真实完成，冷OOM回滚 |
| N04 | 独立顺序舍入oracle、默认Direct/Graph状态消费、生命周期/文档 | scalar/多维索引、首末/重复/全偏斜、0/1/17/128/129及跨CTA长索引、真实128/129/2048宽度、F16/BF16舍入反例、特殊值、strides/alias；Scatter→Gather与norm/residual消费、两captured槽/变更输入/保留旧状态、prepare不执行、无热metadata分配；排序scratch不足/错resource、OOM/close/abandon。完整权重片段/三交付/其它环境/等责P1随C1固定bundle |

多索引 native graph 的冷建图及 runtime 私有内存不是零成本；SDK sort 的内部 dispatch 数量保持
unknown，不用一次 graph launch 冒充一次 kernel。checked 的状态 reset/回报/merge 按 L 卡计费。
同合同性能对照必须包含顺序、舍入、检查及结果寿命；当前没有性能胜出结论。只改 CUDA owner
及直接 consumer/source list/指南；Scan、共同 API、通用 Graph/backend 与治理工具不在 N 授权内。

### C1-N 实施事实

- `47592ccc` 实现有序 CUDA Scatter；`8b2066ab` 完成一次产品纠正：空/单索引路径的零 workspace
  对齐事实与 proposal 一致，单索引精确原位更新只访问目标行，避免遍历整份状态。初次产品和测试
  编译成功，但上述对齐不一致被共同层拒绝，不能记 PASS。Graph 丢弃测试另修正调用方 resource
  寿命：按 [Graph caller-borrow 合同](../../../user/graph.md)保持 workspace resource，直到已接纳工作
  及内部 Invocation 的异步保活回收完成；未改 backend/Graph 合同。
  最终 W1 Release/WERROR provider 36、默认 Direct 4、Graph 14 共 **54 case PASS**；
  [原始尝试、命令与日志哈希](../../../../evidence/correctness/compute-llm-moe-c1/n-windows-focused.json)。
- Assign 的 U8/F16/BF16/F32/I64 原始位、两索引位宽、空/scalar/多维索引、rank8/中轴/负 axis、
  odd/zero stride、末项覆盖与精确 source alias 已过；4097 项索引实际跨 SDK 排序 tile。
  F32/F16/BF16 add 用独立逐次 F32→storage 舍入 oracle 核对全偏斜重复目标、取消/中点反例、
  NaN/Inf、1/17/128/129/4097 项及 128/129/2048 宽度。单/257 索引路径中的负值、等于 extent
  和 I64 极值均不改变整个 output；failed 槽退役、fresh 排序槽恢复、unchecked 合法 zero-stride、
  空输出、缺失/过短/错误 resource 的 workspace 与非法 alias 均通过。
- 默认 BF16 Scatter→Gather→mixed-F32 RMSNorm→residual 运行 M=1/17/23/128、width128/129/2048；
  两个 native captured 槽独立排序图和 workspace、三次变化输入、GPU 结果回传下一步状态输入，
  保留另一槽旧状态/结果。Direct 相同程序逐位一致，另由独立顺序状态和 FP64 norm oracle 核对。
  cold prepare/capture 不执行坏 payload，pending 错误在 executable/槽释放、close/abandon 后
  仍完成并释放内部 workspace；逐点 HYLE metadata OOM 回滚及热重放零新增 metadata 分配通过。
- 统一 LOC self-check：Compute+Graph 实现 **41,999 行 / 162 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/n-implementation-loc.json)。selector 保留
  109 项 skipped reasons；本子集只运行 configure 和上述三个 factor。完整 C1 权重片段、公开头/
  三交付增量、WSL/macOS 与等责 P1 仍未运行，不签四路线完成或性能胜出。CUDA Scan 仍待实施。

### C1-O：CUDA checked Scan（AUDITED）

baseline `c77c11e7`，owner `/root` / `compute_provider_cuda`。沿 D01 的共同合同完成 F02.4 的
expert counts→offsets 和 C3 已登记的 F32 DeltaNet chunk gate 前缀；不新增 schema/API，也不扩大
至未声明的分段/ragged scan。已核对 Native 的全部整数前缀检查和固定路线 F32 重结合合同。

具名来源为固定 ggml `456172ec` 的 `src/ggml-cuda/cumsum.cu`（CUB 块扫描＋跨 tile carry，
长连续行另有 DeviceScan 路线）和本机 CUDA 13.3 `cccl/cub/block/block_scan.cuh`。本轮真实
expert/chunk 的短行、任意轴/byte stride 与整数全量失败原子性，采用 SDK BlockScan 的行内
分块方式；长行同样分块线性处理，不按 benchmark 人为截断支持域，也不声称长行已胜过 DeviceScan。
`BORROW` SDK 扫描 collective 与 prefix callback；`OWN` 描述适配、I64 检查和输出提交边界。

| FIX | 固定实施范围 | P0 / 数值 / 必需验证 |
|---|---|---|
| O01 | CUDA Scan proposal/prepare/program，I32/F32、inclusive/exclusive、rank1–8正负axis/odd与zero byte stride、同shape/type dense输出 | 非空总元素≤INT32_MAX，空域不取payload/launch；仅同描述input0精确alias。冷绑定dtype/exclusive/layout/kernel；32/64/128/256 threads覆盖短行，≥1024行宽每thread4项、其余1项，跨tile carry；无caller workspace/槽私有存储 |
| O02 | SDK BlockScan 行扫描与 checked整数两阶段 | 一CTA/row，O(elements·rank)，只用固定CTA共享存储/寄存器；I32先I64 inclusive prefixes验证所有rows/全部prefix含最终exclusive总和，再单独kernel写结果，任何溢出atomic first-error并使整个output不变。给定计数边界I64求和无溢出；F32允许固定并行重结合，exclusive直接用SDK exclusive而非inclusive减当前值，保护Inf/NaN前缀。精确alias全tile读完再写，跨tile共享存储有barrier；复用L错误字/Completion，无host payload fallback |
| O03 | 普通Direct/default Graph与独立oracle、生命周期/文档 | 空/1/17/127/128/129/255/256/257/1023/1024/1025/8193及长行、rank8/中轴/stride/alias；I64/FP64独立prefix、F32非二进制输入和NaN/±Inf；正负溢出、临时溢出后总和恢复、仅最终exclusive总和溢出、后行失败全输出不变、failed/fresh槽。counts→Scan→Gather和F32 gate前缀，两captured槽/变更输入/保留旧结果/prepare不执行、metadata OOM/accepted owner/close/abandon。完整权重/三交付/其它环境/等责P1仍随C1固定bundle |

I32 两次完整读取与L状态reset/回报/merge如实计费；不拿unchecked模加或浮点单遍给它排名。
本子集只改 CUDA owner/source list 与直接 consumer/指南；无共同层、backend 或公共仪器修改。

### C1-O 实施事实

- `ac8fb8a0` 完成 SDK 块扫描、I32 全前缀检查与普通 Direct/Graph 消费；W1 Release/WERROR 的
  provider 40、默认 Direct 4、Graph 16 共 **60 case PASS**。本候选首次编译/运行通过，无产品
  纠正或数值容差调整；[原始命令、日志与哈希](../../../../evidence/correctness/compute-llm-moe-c1/o-windows-focused.json)。
- I32/F32 的 inclusive/exclusive 由独立 I64/FP64 前缀核对，覆盖 0/1/17/63/64/127/128/129/
  255/256/257/1023/1024/1025/8193/65537、中轴/负 axis/rank8、odd/zero byte stride、精确
  原位与输出保护区；非二进制 F32、NaN/±Inf 通过。整数后行正负溢出、临时溢出后总和恢复、
  仅最后 exclusive 总和溢出均保持所有 output 不变；failed 槽退役、fresh 槽恢复，huge-empty
  跳过 payload、部分 alias 接纳前拒绝与超出已声明计数域的准备诊断通过。
- 默认 counts→exclusive Scan→BF16 Gather 和 F32 Softplus→broadcast scale→time-axis Scan
  实际运行 experts1/17/128/257、tokens1/17/23/128/129；两原生 captured 槽、三次变化输入、
  保留另一槽旧结果，普通 Direct 同程序与独立 host 公式均通过。热重放没有新增 HYLE metadata
  分配；capture prepare 不执行坏输入，pending 整数错误在程序/槽释放、close/abandon 后保活
  至终态，F32 已接纳程序保活与逐点 cold metadata OOM 回滚通过。
- 统一 LOC self-check：Compute+Graph 实现 **42,390 行 / 166 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/o-implementation-loc.json)。selector 保留
  109 项 skipped reasons；本子集只运行 configure 与上述三个 factor。完整 C1 真实权重片段、
  公开头/三交付增量、WSL/macOS 与等责 P1 尚未运行，矩阵仍 UNPROVEN，不签四路线共同完成。

当前无 C1 整包冻结候选或整包 PASS。旧 CLOSED/BLOCKED 卡不授权本包改动。
