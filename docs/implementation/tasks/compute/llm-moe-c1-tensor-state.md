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
  `compute_route_apple` / `graph_route_apple`、`compute_route_vulkan` / `graph_route_vulkan`。
  `compute_route_cpu/cuda` 和 `graph_route_cpu/cuda` 默认工厂先复用；只有具体代码缺陷才定向纳入。
  oneDNN、cuBLASLt、Marlin、backend、Mem/Sched 本包没有已证缺陷，不自动重开。
- 状态：`compute_ops` / `compute_provider_native` 已完成下述 C1-A/B/C/D 的 AUDITED；`compute` 仅
  对 D01 的 typed prepare 接线进入 AUDITED；`compute_provider_cuda` 仅对 E01–E03 进入 AUDITED；其余五个 owner
  保持 SCOPED。AUDITED 只授权具名固定子集，不授权本卡其他尚未裁决的草案。
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

当前无 C1 candidate 或整包 PASS。旧 CLOSED/BLOCKED 卡不授权本包改动。
