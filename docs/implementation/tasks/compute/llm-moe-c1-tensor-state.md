# C1：四路线基本张量与状态边界

当前 CPU 成熟归一化/融合接入见 [C1-AA](llm-moe-c1-onednn-normalization.md)；
oneDNN 两个 owner 的具体状态以 registry 为准。

用户里程碑要求（2026-09-19）：C1全部适用验收完成后单独明确通知，汇总能力、性能取舍及限制，
供用户提问；局部实现或单项测试通过不能触发C1完成通知。

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
  `compute_backend_cuda` 只按下述 C1-L 卡施工，`compute_provider_ggml_vulkan` 按 C1-Y 独立卡施工。
  `compute_route_cpu/cuda` 和 `graph_route_cpu/cuda` 默认工厂先复用；只有具体代码缺陷才定向纳入。
  oneDNN 后续仅因具名累加精度准入遗漏进入 [C1-Z 定向修复](llm-moe-c1-cpu-matmul-precision.md)；
  cuBLASLt、Marlin、Mem/Sched 没有已证缺陷，不自动重开。CUDA backend 只因 checked
  操作缺少异步算法状态进入 [C1-L 定向依赖卡](llm-moe-c1-cuda-device-status.md)，其余 backend 不重开。
- 状态：`compute_ops` / `compute_provider_native` 已完成下述 C1-A/B/C/D 的 AUDITED；`compute` 仅
  对 D01 的 typed prepare 和 F01–F02 的普通准备诊断进入 AUDITED；`compute_provider_cuda` 对 E01–E03、G01–G04、H01–H03、I01–I03、J01–J03、K01–K03、M01–M03、N01–N04 和 O01–O03 进入 AUDITED；Apple 两个 owner 对下述 P01–P05、Q01–Q04、R01–R04、S01–S03
  进入 AUDITED；Vulkan 两个 owner 对下述 T01–T04、U01–U04、V01–V04、W01–W03、X01–X04 进入 AUDITED，`graph` 保持 SCOPED。CUDA backend 仅对 C1-L 的 L01–L04 进入 AUDITED。AUDITED 只授权具名固定子集，
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

## 当前施工裁决：先 Provider 覆盖，再按缺口补实现

2026-09-18，owner `/root`，本次重新定界 baseline `6695fba0`。用户指出实际推进仍偏向自写，
而非先让成熟 Provider 承载需求后比较选优；源码复核支持这一纠正。此前 A–X 已实现并局部验证
的事实保留，但其按算子扩域的顺序不再作为后续施工队列；下文初始方向表也不再授权继续自写。
本节与上位计划的 Provider 顺序共同约束后续有限 FIX，未改变的证据按身份复用。

当前接入事实是：CPU oneDNN 仍主要为 BRGEMM，ggml-CPU 仍 BLOCKED；CUDA 有 cuBLAS/Marlin/
SGLang 的具名窄域，C1 多项仍为自有 kernel/CUB 组件；Apple 已有较广的 MLX typed/region 接线；
Vulkan `a7ebf205` 刚完成 norm/gate 等首组 managed 普通入口的 W1 static/shared 局部验证。
这些事实不等于各 Provider 的全部 C1 能力已经绑定，细目见
[Provider 承载复核](llm-moe-c1-provider-reuse-audit.md#当前接入覆盖与后续施工纠正2026-09-18)。

### 现在按此顺序施工

1. **按原矩阵核对四路线的实际承载。** 对 C1 的 F01–F04/F19，分别注明上游现成实现、HYLE
   已绑定域、待绑定域、需要的语义/资源适配，以及确实缺失的域。以真实模型片段成组处理，
   不以 Provider 名称、target 数量或一个示例证明全面支持。调查仍受60分钟/两个来源预算限制。
2. **优先补实际接入。** 从 oneDNN/ggml、CUDA 已登记库、MLX、ggml-Vulkan 选具名成熟实现，
   将其完整算子或 region 接普通 Direct/Graph 与固定规则；复用现有执行/资源机制，第三方由
   CMake 的既有锁、feature、target、资源和 package 合同描述。新增依赖和 CLOSED owner 必须
   先完成自身 scope/audit，原 ggml-CPU blocker 不自动解除。
3. **先资格化，再比速度。** 本包数值、checked/alias、失败与资源/状态合同、实际输入片段和
   三交付适用门先完成。检查式入口、不同精度、不同完成点或不同输出责任不能混排。已有实现
   的具体缺陷插入该 owner 的有限 FIX；纯配置/静态事实在准备期检查，动态事实保留必要检查。
4. **用完整成本裁决路由。** API/算法固定后执行原预声明 C1 P1 bundle，计入适配/转换/同步/
   首次准备/驻留/融合损失。可按硬件/驱动/toolkit/shape/dtype 条件分别胜出；执行不搜索或计时。
   没有充分资格或结果 INCONCLUSIVE 时如实记录，不把未证明路径命名为默认优选。
5. **补缺与清理。** 只有成熟候选确实缺少所需实现，或 HYLE 有可验证完整优势，才固定自有
   算法 FIX；接口差异优先用受审适配解决。HYLE 已有路径也参与公平比较，完整替代后按域迁移
   和删除无价值重复代码；原支持能力与 SIMD 基础库保留，基于 SIMD 的算子路径仍参与淘汰。
   最后回填原矩阵和本包三交付/性能证据。

### 必要检查与已有缺陷

普通入口必须符合已准入合同：shape/dtype、容量与字节起点/对齐、合法 alias、整数溢出与索引
边界、数值特殊值、OOM/部分接纳、错误传播、关闭/取消/完成前保活。只检查本轮实际接入和被
替代的调用链；没有具体新事实不重开下层 CLOSED 模块。明确请求的 unchecked 专家域单独
绑定合同，不能偷换普通默认域或与 checked 路径按不同责任比较。

每个发现记录“反例 → 影响域 → owner → 有限 FIX → 回归/资格门”。不合格实现先从该域的候选
集合排除；已有合格路径可继续使用，没有时明确失败并记录待修复。性能预算不是保留缺陷的
豁免，正确性修复也不授权顺手扩算子。确认安全且不可变的元数据/权重资格可缓存，不能以
调用者一般会传对为由省略会随输入变化的必需检查。

本次先纠正施工顺序和事实表达；下一产品子集仍须在当前 owner 卡中提交有限 AUDITED FIX。
不把这次文档更新、已有测试数量或接入覆盖调查计为新能力完成。

2026-09-18 的 [C1-Z](llm-moe-c1-cpu-matmul-precision.md) 冻结 Native 的新增 Z02：MatMul
准入遗漏显式累加类型，必须在共享冷计划函数拒绝非 F32 域，避免 oneDNN 拒绝后静默 fallback。
只修资格并添加普通 Direct/Graph 工厂回归，已有 F32/SIMD 算法不动，不借此扩 C2 或重开路由 owner。
该定向修复的 `e1913515` 已通过 W1 static/shared 与 WSL：每套 Native108例、全部8个目标通过；
完整结果见 C1-Z 终局。Native 仍保持本包 AUDITED，真实权重、其余平台和本包性能资格未由此代签。

## 已完成的源码审计输入

当前承载选择按 [C1 Provider 复核](llm-moe-c1-provider-reuse-audit.md)纠正执行顺序：先接成熟
实现或完整 region，再补接入与明确缺口。下文历史子集中的 `BORROW` 有时只指 SDK 原语、HYLE
既有能力或算法结构，不能据此记为第三方算子接入。X 的既定验证与记录完成后，下一项先处理
成熟实现的接入边界；尚未定界的 TopK/index/Scan 不直接延续为自写 shader 队列。

2026-09-18 用户进一步明确：先完成主流 Provider 的实际接入和覆盖核对，再为剩余必要缺口
安排 HYLE 实现。已完成的 A–X 自有算子按同一标准参加完整等责比较；不胜出且没有可信优化
机制，或在既定有限纠正预算内仍无优势时，迁移消费者并删除已被成熟实现完整覆盖的重复路径。
必要未覆盖域和具名硬件/版本/shape 的优势域分别保留。源码与测试投入不构成保留理由；本次
确认只固定后续裁决规则，不凭未运行的比较立即删掉有效能力。
用户明确保留 SIMD 基础库：其通用用途超出推理，本轮不删除/裁剪 SIMD 模块和公共能力。
该保留不豁免 Compute/Graph 中基于 SIMD 的算子 Provider；其适配、候选注册和路由同样在
同合同比较落败、替代覆盖及迁移验证完成后删除，不连带删除底层基础设施或非推理消费者。

当前 [C1-Y](llm-moe-c1-ggml-vulkan-region.md) 已完成 `3c39428e` 的 W1 static/shared、Air/Mini
局部 correctness；`351ec13d` 同产品的 Windows/Air managed FFN 三交付、112项所选门全部通过。
两平台 static/shared 的真实2B归一化片段也通过；其 source consumer、其余真实权重片段、
TopK/index/Scan 等能力映射和性能优选尚未完成，不能据此签 C1 全部能力。下一步仍按上节
固定成熟实现的接入/适配批次，不延续初始自写队列；已有程序保留为比较与补缺候选。

2026-09-19，baseline `d27c14a3`，Vulkan 三个当前 owner 对 [C1-Y 的 Y21–Y24](llm-moe-c1-ggml-vulkan-region.md)
重新定界并 AUDITED：冷计划/Graph 多输出 SSA，原版 ggml Argsort 比较语义适配与 GetRows
组合，普通 TopK 值/索引双输出及生命周期、三交付。只覆盖 F04.6 的 F32/I32 已准入消费域，
不新建自有排序算法、共享执行器或公共 API，不重开其他 owner；产品实施和验收尚未完成。

2026-09-19，`a50b2ef4` 的新增高秩常量反例暴露原版 Argsort 的设备 descriptor 对齐约束。
原Y21–Y24两轮纠正预算已用尽，`compute_route_vulkan` 仅对 [C1-Y 的 Y25](llm-moe-c1-ggml-vulkan-region.md)
重新定界并 AUDITED：准备时区分排序输入与已有偏移感知算子的对齐要求，最多一次具名修正。
其余 owner、能力范围与 CPU 性能 blocker 不变，不延续为新的自写算子队列。

以下表格记录 C1 初始 baseline 的审计输入，**不是当前产品状态或继续自写的施工指令**。

| owner | 初始 baseline 事实 / 源码 | 当时记录的缺口与方向 |
|---|---|---|
| compute_ops | `operation.h` Unary 止于 SiLU；已有 View/Cast/Reduce/RmsNorm/LayerNorm/Gather/Scatter/TopK；operation.cc 完成共同 shape 检查 | softplus 与必要 scan 的 typed schema；保留既有数学/alias 语义，明确半精度舍入 |
| compute | operation.cc 已有 cold propose → prepare 与明确 decline/error；普通入口不能只查 schema 推断硬件可用 | 普通 capability query 与准备拒绝复用同一资格判断；不执行 payload，不变热路径 |
| Native | elementwise/normalization/indexing/reduction/transform 分文件；SIMD F32 优势与 CPU executor 已存在 | F16/BF16 所需路径和稳定数学；按实际消费融合、尾块/stride；不每元素分配或调用 vtable |
| CUDA | native_kernel.cu + SGLang SwiGLU 与 cuBLAS/Marlin 工厂组合；有自己的 native stream/capture | 成熟 CUDA kernel 承载必要 tensor/index/norm；HYLE 控制边界、失败和数值，不借 CPU 模型 fallback |
| Apple | apple_operations.cc 仅 F32 2-D GEMM/affine Q4/SiLU；apple_lowering.cc 拒绝非纯/非固定域 | MLX GPU ops/完整 region 的 typed 接线；functional state 输入输出、实际 dtype/shape；保持数组 owner |
| Vulkan | vulkan_operations.cc 的录制/packing/liveness 已完成；vulkan_lowering.cc 只接 pure fixed 图 | 扩 SPIR-V tensor/index/norm 与真实 visibility/state 边界；有界原生程序，不引入通用逐 kernel executor |
| Graph | 通用 Graph 已有 derived shape、资源 effects、alias、独立 invocation；两条原生 lowering 较窄 | 在路线本身落实所需 shape/参数/state；不因 native 不同复制一套逻辑模型图 |

后续审计以当前施工裁决、实际源码、已登记 peers 与 C0 输入为准。新依赖同时绑定
pin/license/三交付；上表的“扩 shader/Native”不再自动成为下一 FIX。

## 有限 FIX 草案（尚不授权产品实施）

| ID | 允许改动 | 必需验收 |
|---|---|---|
| C1-01 | 共同 softplus/scan 语义、必要数值/索引/alias 域及普通能力查询 | 真实请求同准备资格一致；invalid/overflow/unsupported 清楚；新增公共面回映需求 |
| C1-02 | CPU 成熟 Provider 的 C1 实际覆盖、普通接入与有限适配；Native 作为已有比较/补缺候选 | BF16 真实输入、尾块/stride、特殊值、重复索引、独立状态；同合同资格及完整成本，不因已有 Native 代码先判自写 |
| C1-03 | CUDA 成熟算子/融合程序的覆盖接入和条件规则，复用 stream/capture | 实际 GPU 成熟实现、同精度 reference、checked/准备失败与接纳后 drain；CUB 构件与完整 Provider 分别记录 |
| C1-04 | Apple MLX typed 普通入口、原生图 lowerings 与状态结果 | 同 runtime GPU region、旧结果保活、两个状态、所需 dtype/shape；无用户 custom adapter |
| C1-05 | Vulkan ggml managed region 的现成能力映射、语义/状态适配与固定规则 | 普通 Direct/Graph 默认路由、同步/alias/scratch、重复执行与失败；上游已有但未绑定单列，不直接改为自写 shader |
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

### C1-P：Apple typed 张量、门控与归一化（AUDITED）

baseline `a466b406`，owner `/root` / `compute_route_apple` 与 `graph_route_apple`。本地 BF16
embedding/norm/gate/residual 和 FP32 gate 参数必须沿普通 typed Direct/Graph 到达 MLX GPU。
实际缺口是 `apple_operations.cc` 仅接 F32 GEMM/Q4/SiLU，`apple_lowering.cc` 拒绝非 dense F32
描述；现有 Context、原生编译/结果 owner 与失败排空可复用，没有重写其执行器的证据。

本轮只读锁定 MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85` 的 `mlx/ops.h/.cpp`、
`mlx/fast.cpp` 与对应 Metal kernels/NN activation 公式。相邻研究 checkout 的 HEAD 不同，
已从上述精确 Git revision 读取；不升级依赖。`BORROW` MLX tensor/elementwise/fast norm 与
完整编译 region；`OWN` HYLE schema、数值提升/最终舍入和薄 typed 适配。归一化显式使用 F32
输入/参数调用原生 fast RMSNorm/LayerNorm，最后一次回到输入存储类型，避免半精度 affine 中间
舍入；必要转换及 MLX 私有存储不是零成本。F32/半精度 Unary/Gated 同样在 F32 求值后一次舍入。

| FIX | 固定改动与 owner | P0 / 必需验证 |
|---|---|---|
| P01 | Compute Apple 的 logical dtype/shape 检查扩至 F32/F16/BF16/I32/Bool、rank0–8/empty；普通 typed prepare 增加 InputSpec overload，复用既有冷 compile | 形状/精度/模式先按共同 schema 推导再资格检查；元数据准备不读用户 payload。MLX 自己持有物理布局，不引入 Mem buffer 导入或 HYLE shadow copy；原 F32 GEMM/affine Q4 域保持其资格 |
| P02 | Compute Apple 接 View/Copy/Fill/浮点 Cast/Unary/Binary/Select/Concat | View 的 reshape/slice/permute/broadcast 与原生 stride，Select Bool mask，Fill 原始 scalar bits；计算只准入三种浮点，I32/Bool 用于搬运/选择/形状。稳定 softplus、NaN 与负无穷 activation、fmin/fmax 的 NaN 规则；广播与任意 Concat arity。全部使用同 stream 的 MLX ops，由完整 region 融合；HYLE 只冷期映射，不增加逐节点热调度 |
| P03 | Compute Apple 接三种 GatedActivation、RMSNorm/LayerNorm 与 mixed-F32 参数 | packed/separate、F32 累加、空域、独立参数类型/epsilon 诊断；使用原生 fast norm，半精度在最终 affine/gate 后舍入一次；未知/其他 accumulation 域明确拒绝。不是逐元素自写 Metal，也不添加模型层 API |
| P04 | Graph Apple 接同一 dtype/shape 和 View 派生逻辑描述，继续共享 lower/单个 compiled region；报告反映实际精度域 | native array 按逻辑 shape/dtype 匹配，物理 strides/必要布局整理由 MLX 管理；纯计算的 functional 输入/新结果保活，旧结果不能被另一槽覆盖；不修改通用 Graph、effect/control 或通用执行层 |
| P05 | 两 owner 普通入口/独立数值与布局 oracle、生命周期及指南 | F32/F16/BF16、2048 hidden、0/1/127/128/129/257、rank8/scalar/原生转置和广播/各 View/5输入Concat/Bool mask/特殊值；独立舍入和 FP64 norm/gate，metadata-only prepare/拒绝、两槽三次变输入/保留旧结果/close、现有 OOM/取消/错误排空回归。先 Air 与 Mini 对应 factor/公开头；完整 C1 权重片段/三交付/其它路线/等责 P1 随固定 bundle |

已审成本是冷期 HYLE schema/MLX tracing，热期仍一次原生 compiled region 的 managed replay；
原生布局整理、F32 临时量、编译/常量/结果 owner 成本沿现有报告如实记账，不能声称零分配或
零拷贝适用于任意 native stride。空 tensor 不运行 norm 计算。P 不接 Reduce/Softmax/TopK 或
checked Gather/Scatter/Scan；它们的数值/异步错误合同继续在 C1 后续有限 FIX 内处理，不继承 P。
本次不启动 P1，不改 MLX pin、公共治理、Mem/Sched 或原生执行状态机。

### C1-P 实施事实

- `dfba32e9` 接通 typed InputSpec 准备、MLX 张量/门控/mixed-F32 norm 与共享 Graph lower；
  原生编译成功，新增测试诊断先因 Dtype 非 enum、后因局部变量遮蔽 helper 编译失败。
  `8627bb80` 修正测试诊断，两台 Mac 各 21 case 与两个公开头检查通过；这些失败没有 runtime PASS。
- 在固定 MLX 的 `mlx/backend/common/compiled.h` 发现浮点常量打印把 NaN 统一写为 `NAN`，
  据此补入原始位 Fill→Select 编译片段反例。`1708fd54` 在 Air 实测 BF16 `0xffc3` 变成
  `0x7fc0`，plain Fill 与 F32/F16 变体通过；不能只凭源码认定该打印函数是唯一内部原因。
  `c4a0db38` 将浮点 Select 通过同宽整数 view/原生 where 保留位模式，再 view 回原 dtype。
  仍由 MLX 执行完整 region；没有自写 host payload 路径或修改依赖，一次产品纠正、未放宽容差。
- 最终 `c4a0db38` 在 Air `Mac16,12` 与 Mini `Mac16,10` 的 Release/WERROR 下，各完成
  Direct **15**、Graph **7**，共 **22 case PASS**，以及两个独立公开头检查。Mini 只签
  correctness/portability。Air 使用 Windows Git bundle，Mini 使用整树 archive 与 SHA-256；
  两份均为可丢弃验证副本，build2/CTest1、真实 Metal GPU，无依赖下载。
  [全部尝试、命令、日志哈希与 selector](../../../../evidence/correctness/compute-llm-moe-c1/p-apple-focused.json)。
- 独立 raw-bit/layout 与 FP64/存储舍入 oracle 覆盖五类存储、rank0–8、空/尾块、真实 native
  转置、广播与 View、五输入 Concat、浮点 Cast/13 Unary/6 Binary/三 gate、mixed norm。
  NaN/Inf/负零、原始 scalar payload、metadata-only prepare 和精度/参数拒绝均通过。
  Graph 的 F16/BF16 norm→gate→residual→Select 与 slice→Concat 实际运行 rows0/1/17/23/129、
  width129/2048；两 invocation、三次变输入、保留旧结果、输入不变和 close，以及既有 OOM/
  取消/late failure/排空回归通过。同程序 Direct 逐位对照之外另有独立数值 oracle。
- 统一 LOC self-check：Compute+Graph 实现 **42,747 行 / 168 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/p-implementation-loc.json)。Windows/Air
  selector 分别保留 112/110 项 skipped reasons；本子集实际执行两台 Mac configure、两个 factor
  与两个头检查。完整 C1 真实权重、三交付/其他环境与等责 P1 尚未运行；Apple 剩余归约/索引/
  Scan、Vulkan 及整包验收继续按有限 FIX 推进，矩阵仍 UNPROVEN，不签四路线共同完成。

### C1-Q：Apple 归约、Softmax 与稳定 TopK（AUDITED）

baseline `2b411b37`，owner `/root` / `compute_route_apple` 与 `graph_route_apple`。F04 的归一化、
router score→TopK→选中权重归一化，以及 F10 的 L2 norm 组合需要这组能力；复用已有共同 schema，
不加入 MoE 策略或新公开对象。P 的一次产品纠正已记录；Q 是此前明确留下的不同算法机制。

已核对同一 MLX pin `1f8e74e` 的 `mlx/ops.cpp`、Metal `kernels/reduction/ops.h`、
`kernels/softmax.h`、`kernels/arg_reduce.metal`、`kernels/sort.h` 与 `sort.cpp`。
MLX GPU ArgPartition 当前也是全排序，不能将其声称为更低复杂度选择；ArgMin 对等值取小 index，
merge sort 的比较/合并在相等时保持原序。直接 float Min/Max 的标量比较未定义混合 ±0 优先级，
不能继承为 HYLE 的 signed-zero 合同。共同语义来自 C1-C/J 已绑定的真实 consumer 和独立公式，
本轮不改变它们。`BORROW` 原生归约/softmax/稳定排序与资源管理；`OWN` 精度、空域、排序 key
和最终输出的薄适配；不复制 vendor kernel、不自建逐 kernel 执行器。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| Q01 | Apple Reduce Sum/Mean/Min/Max，F32/F16/BF16、任意合法 axis 集/keep-dim/native layout；F32 累加，half 必须显式声明 | 空集合 Sum=0、Mean=NaN、Min=+Inf、Max=-Inf；空输出不启动归约。Sum/Mean 借原生 F32；Min/Max 将 IEEE F32 位映射为单调 U32 key，NaN 映至胜出 sentinel，借一次原生整数归约再逆映射，保护 NaN 传播和 ±0。O(N)，key/精度转换与 native 临时成本计费 |
| Q02 | Apple Softmax/LogSoftmax，三浮点、任意轴、scale、广播 Bool/同类型 additive mask、显式 F32 accumulation | Bool false 覆盖坏输入为 -Inf；先 F32 scale/mask，最终一次 storage 舍入。Softmax 借原生 precise；LogSoftmax 用减行最大值后的 exp/sum/log，避免大共同偏置抵消。任何未屏蔽 NaN/+Inf 或全 -Inf 行全输出 NaN；空域不归约。O(N)，融合/布局/中间量如实计费 |
| Q03 | Apple TopK 两输出，F32/F16/BF16、任意轴，values 保留原始位、I32 indices；largest/smallest、stable/sorted flags | 同分含 ±0 按小索引优先，NaN 总在最后；U32 ordered key 合并 ±0、统一 NaN 排序，k=1 原生 ArgMin O(N)，k>1 原生 stable merge argsort O(N log N) 后切片，原始整数 view 取值后还原 storage。允许 sorted=false 仍有序；空非axis域不排序。不把 MLX 私有 scratch 写为零，也不声称胜过未比较 peer |
| Q04 | 普通 Direct/Graph 两输出和组合、报告/指南、独立 oracle 与生命周期 | 三 dtype、多轴/负轴/rank8/native transpose/broadcast、空/scalar/1/127/129/2048；TopK 实际 E/k 与跨 2048 tile 长行、ties/NaN/±Inf/±0、原位 bits oracle；softmax 非有限/mask/大偏置、拒绝域；score→Softmax→TopK→Reduce→归一化，两个槽/变更输入/保留旧结果/close，既有 OOM/错误回归。Air/Mini 对应 factors；完整真实权重/三交付/其他环境/等责 P1 仍随 C1 固定 bundle |

本子集只改 Apple 两 owner 的私有 lowering/source list、自然 consumer/文档；多输出冷推导复用
现有 `infer_output_descriptions`。不接 checked Gather/Scatter/Scan，不变 Context 运行状态机、
共同 Graph/API、依赖 pin 或公共治理。本轮不启动 P1；实现稳定后运行具名正确性，不反复追加姿势。

### C1-Q 实施事实

- `fe5362e3` 完成 Q01–Q04：原生归约/Softmax/稳定 TopK，共同冷推导支持两个输出；Direct 和
  普通 Graph 共享 lowering。Air `Mac16,12`、Mini `Mac16,10` 的 Release/WERROR 首次均通过
  Direct **19**、Graph **8**，合计 **27 case**，另各两个独立公开头检查。没有产品纠正、
  依赖变更或容差调整；[命令、两机日志与哈希](../../../../evidence/correctness/compute-llm-moe-c1/q-apple-focused.json)。
- 三浮点 Reduce 的独立 FP64 oracle 覆盖多/负 axes、keep-dim、rank8、native 转置、scalar、
  空集合与空输出；Min/Max 的混合 ±0、NaN 与 Inf 通过。Softmax/LogSoftmax 实测
  width0/1/17/127/129/2048/8193、非末轴/转置、负 scale、广播 Bool/同类型 additive mask；
  masked NaN 被排除，未屏蔽 NaN/+Inf、全 -Inf 与全 false 行全 NaN，F32 大共同偏置不丢失
  `-log(4)`。独立公式与既定存储舍入容差通过。
- TopK 的独立 stable scalar ordering 和源存储逐位对照覆盖 k=1、全选、实际 E/k 与
  4097 列跨 native sort tile；largest/smallest、sorted/stable flags、任意轴/转置、NaN/±Inf/
  ±0、不同 NaN payload、空域和准备拒绝均通过。score→Softmax→TopK→Reduce→选中权重归一化
  的普通 Graph 实测 rows/experts/k 为 1/128/6、17/256/8、23/897/8、129/129/6、0/256/8；
  两 invocation、三次变化输入、源 Graph 释放、旧 values/indices 保活、输入不变和 close 通过。
  同程序 Direct 逐位对照之外使用独立数值/排序 oracle，既有 OOM/取消/late failure/排空回归通过。
- 统一 LOC self-check：Compute+Graph 实现 **42,916 行 / 170 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/q-implementation-loc.json)。Windows/Air
  selector 保留 112/110 项 skipped reasons；本批只运行两台 Mac configure、两个 factor 和两个
  头检查。完整 C1 真实权重、三交付/其他环境、等责 P1 和矩阵逐格验收仍待完成；没有性能胜出
  或四路线完成结论。下一具名缺口是 Apple checked Gather/Scatter/Scan，不扩大至新模型策略。

### C1-R：Apple 算法失败回报、checked Gather 与 Scan（AUDITED）

baseline `1936a2e6`，owner `/root` / `compute_route_apple` 与 `graph_route_apple`。F02 的真实
token IDs/embedding、counts→offsets 和 F10 gate 前缀需要索引与 Scan。具体基础缺口是 Apple
当前只将 native 异常写入 Completion，没有设备算法状态路径；因此本有限子集依法纳入同 owner
的 `apple_context.cc`，不重审 Context 线程/FIFO/close 架构或其他 backend。

具名源码仍为 MLX `1f8e74e`：`mlx/ops.cpp` 的 Take/Depends/CumSum、Metal
`kernels/indexing/gather.h`、`kernels/scan.h/.metal`、`scan.cpp` 与 `primitives.h`。
native Gather 允许负索引偏移且没有 HYLE 边界错误回报；native Scatter 使用原子更新，不能证明
重复浮点更新的规定顺序。故 R 先完成共享失败通路及 Gather/Scan；Scatter 保留给下一次具名
有序更新审计，不能由 native atomic scatter 冒签。MLX 已有 I64 GPU scan，exclusive 原生为移位
写出而非 inclusive-current；借用它的原生算法与数组资源，不下载用户 indices/payload 作 host 运算。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| R01 | Apple 私有冷 trace 收集 checked 表达式，将可达错误合并为一个隐藏 I32 原生输出；Completion 在现有 GPU synchronize/array.wait 后解读 | 冷期 thread-local RAII scope 只记录 typed lowering 构图，原生 Depends 为数据/检查建立唯一可达锚点；死节点检查不进入终态。每程序冷 metadata 保存是否存在检查，未含检查的程序不增加 native status output；热期无 collector、图遍历或 provider 搜索。有检查时一个逻辑4字节状态原生数组/最终读取，实际 MLX 分配如实计费。仅算法 OutOfRange 退役当前槽、不 poison Context；native 执行异常仍沿原失败排空。不公开新 runtime/type/虚表 |
| R02 | checked/unchecked Gather：F32/F16/BF16/I32/I64/Bool source、I32/I64 indices、任意合法轴/原生布局，原始存储位输出；必要 I64 storage/Graph logical 描述接线 | 同一 GPU 验证全部非空 indices，checked 执行使用安全 clamp，禁止负索引 wrap 成成功；零 source-axis 且非空输出以安全占位结果＋错误，不越界访问。失败不发布结果 lease，functional 输入与旧结果不变。空输出不读 payload；非空 output/indices 计数≤INT32_MAX、维度≤native int。低层 checked lower 只能在 managed Context tracing 内登记错误，外部调用明确拒绝，不能静默丢错误 |
| R03 | I32/F32 inclusive/exclusive Scan，任意轴/native stride/empty | I32 用原生 I64 inclusive scan，检查全部前缀含最终 exclusive 总和；计数≤INT32_MAX 保证 I64 累加不溢出。非法前缀 GPU 合并 overflow，结果不发布；合法输出转回 I32，exclusive 移位前缀，不用减当前值。F32 直接原生 scan，保留浮点重结合及 NaN/Inf；O(N)，I64 全尺寸临时、状态归约/转换和必要原生布局整理计费，无 host payload fallback |
| R04 | 自然 Direct/Graph 及失败/寿命/指南 | 独立位级 gather/I64或FP64 prefix，I32/I64极值坏索引、重复/scalar/多维/rank8/转置/广播/empty、尾块和非有限；临时整数溢出恢复、后行错误、仅最终exclusive溢出。prepare 不以隔离 warm 值裁决用户 payload；实际 checked 图、两槽/变输入/旧结果、失败槽退役/fresh恢复、close/丢弃 completion/owner保活、死检查、首错优先及已有异常/OOM回归。Air/Mini factors，完整权重/三交付/其他环境/等责P1随C1固定bundle |

本轮不改变公共 Operation、共同 Graph、MLX pin、公共仪器或其他模块 owner。新增冷 scope 必须
异常安全、按线程隔离且退出恢复；原生检查依赖留在同一个 MLX compiled region。payload 错误
不能借 warmup 提前拒绝或在失败后换 Provider 重算。本子集不运行 P1，不能由 correctness 宣称 SOTA。

### C1-R 实施事实

- `b3095468` 完成 R01–R04：同一 MLX compiled region 内合并可达算法检查，接通 checked/unchecked
  Gather、I32/F32 Scan、I64 storage 与普通 Graph。Air `Mac16,12`、Mini `Mac16,10` 的
  Release/WERROR 首次均通过 Direct **25**、Graph **9**，共 **34 case**，另各两个独立公开头
  检查。没有产品纠正、依赖变更或容差调整；Mini 只签 correctness/portability。
  [命令、两机日志与哈希](../../../../evidence/correctness/compute-llm-moe-c1/r-apple-focused.json)。
- 六种 source storage 与两种 index 宽度的 Gather 通过独立原始位 oracle；覆盖 scalar/多维
  indices、重复和边界、rank8/native 转置/广播、I64 极值存储、空源/空输出。负索引、extent
  与整数极值坏索引均回报 OutOfRange，失败不发布结果，输入和旧结果不变。Scan 覆盖
  width0/1/17/127/129/1025/8193/65537、中轴/负 axis/native stride、inclusive/exclusive；
  本测试域 I32 前缀使用可精确表示整数和的独立 FP64 oracle，F32 使用独立 FP64 参考。
  NaN/Inf、正负临时溢出后恢复、仅最终 exclusive 总和溢出均通过。
- warmup 不裁决用户 payload：捕获坏常量的程序可准备，实际调用才失败；不可达的坏表达式
  不影响成功，首错优先、冷 trace 异常回滚、非 managed tracing 拒绝均通过。失败槽退役、
  同 Context 新槽恢复；测试原生 checkpoint 持有真实已接纳 GPU 前缀，释放 Program/Invocation
  后仍安全完成错误回报。丢弃 Completion、旧结果跨 close 保活和既有 OOM/取消/native 异常
  排空测试通过；算法错误没有误 poison Context。
- 普通 Graph 实测 I64 IDs→F16/BF16 embedding Gather→mixed RMSNorm→residual，以及 counts→
  exclusive Scan、F32 Softplus→Scan；rows/width/experts 为 1/129/17、17/2048/128、
  23/129/257、128/2048/129。三个公开结果不暴露内部状态；同 lower Direct 逐位对照之外，
  使用独立索引/FP64/存储舍入参考。两槽、三次变化输入、源 Graph 释放、保留旧值、两类算法
  失败与新槽恢复通过。本批使用诊断权重，不冒称完整 C1 的真实模型权重消费。
- 统一 LOC self-check：Compute+Graph 实现 **43,133 行 / 173 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/r-implementation-loc.json)。Windows/Air
  selector 保留 112/110 项 skipped reasons；本子集实际运行两台 Mac configure、两个 factor
  与两个头检查。完整 C1 真实权重、三交付/其他环境、等责 P1 和矩阵逐格验收仍待完成；
  Apple 有序 Scatter 与 Vulkan 继续按有限 FIX 推进，不签四路线完成或性能胜出。

### C1-S：Apple 有序 Scatter（AUDITED）

baseline `babd0d36`，owner `/root` / `compute_route_apple` 与 `graph_route_apple`。F02.2/F02.4/F19
的状态写回与重复目标累加要求与 C1-N 相同的逻辑行主序：assign 最后项胜出，add 每次 F32 求值
后舍入到 storage。Apple 保持 functional 输入/新结果，接 R 的算法状态；不新增原位 Mem 导入。

具名来源仍为 MLX `1f8e74e`：Metal `kernels/indexing/scatter.h` 使用原子更新，不能继承有序
语义；`kernels/sort.h` 的 merge argsort 稳定。`fast.h`、`backend/common/metal_kernel.cpp`、
`backend/metal/custom_kernel.cpp` 提供同 stream 的 lazy CustomKernel、native 输入布局整理、
临时数组保活与 Safe math 编译模式。`BORROW` MLX 排序/编译/分配/执行，`OWN` 有序更新片段；
没有另建 Queue、逐 kernel vtable 或 host payload 路径，亦不改 vendor pin。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| S01 | Apple Scatter assign 接六种已有 storage，add 接 F32/F16/BF16；I32/I64 索引、任意轴/scalar/多维/native layout | 非空 output/indices/updates 元素数≤INT32_MAX；空 output 不读 payload，空 indices 保留原始 source。默认 checked 在同 GPU 检查全部索引，失败不发布结果，输入与旧状态不变；unchecked 合法域同结果。整数/Bool add 明确拒绝 |
| S02 | 多索引复用 native stable argsort；一个 MLX managed Metal 片段按目标有序更新，单索引省略排序 | 每个目标元素一个 writer；每个 threadgroup 一次二分得到稳定索引段，assign 取最后 ordinal，add 顺序读 updates、每步 F32→storage。输入/未更新/assign 使用原始整数位，禁止浮点 NaN 规范化。复杂度 O(K log K + N + U + G log K)，K为索引数、N为output、U为updates、G为输出threadgroup数；共享8字节段界，无axis大小索引表、无N×K全扫描。MLX 原生布局整理、索引排序、结果/临时量及 checked 状态均计费；不声称零拷贝或性能胜出 |
| S03 | 普通 Direct/Graph 状态消费、独立有序舍入 oracle、生命周期和指南 | 六storage/两index宽度、scalar/多维/中负轴/rank8/转置/广播、空/单/17/129/4097索引、真实128/129/2048宽度、全偏斜与分散重复、NaN/Inf/负零/原始payload；半精度逐次舍入和F32取消反例。Scatter→Gather→mixed norm/residual、两槽三次变化、结果作下次状态、旧状态保活、坏索引失败/fresh恢复/owner释放/close，现有OOM与native错误回归。Air/Mini两factors与头检查；完整真实权重/三交付/其他环境/等责P1随C1固定bundle |

只改两 Apple owner 的私有 lowering、source list、直接测试和文档。复用 R completion/失败槽机制，
不改公共 Operation、通用 Graph、Context 状态机、其他模块或公共仪器。本子集不启动 P1；未运行的
完整 C1 门继续未通过，排序原语或局部核通过不代替四路线共同验收。

### C1-S 实施事实

- `373bbeee` 完成 S01–S03：多索引 native stable argsort、单索引免排序、MLX managed Metal
  有序更新与普通 Direct/Graph；原始位 assign、每次 F32→storage 舍入的 add 和 R 算法状态接通。
  Air `Mac16,12`、Mini `Mac16,10` 的 Release/WERROR 首次均通过 Direct **29**、Graph **10**，
  共 **39 case**，另各两个独立头检查；没有产品纠正、容差放宽或依赖变更，Mini 只签
  correctness/portability。[命令、两机日志与哈希](../../../../evidence/correctness/compute-llm-moe-c1/s-apple-focused.json)。
- Assign 的六类 storage/两索引宽度、scalar/多维/rank8/中负轴/native 转置与广播、原始 NaN
  payload/±0/大 I64 位模式通过独立逻辑更新 oracle。Add 的三浮点、全偏斜与分散重复、
  count/width1/1、17/129、129/128、4097/3、17/2048 通过独立逐次 F32/数学 storage 舍入
  oracle；含 NaN/Inf、F32 取消和半精度末次才舍入会出错的反例。数值与零符号精确核对，NaN
  单独分类，不借宽松容差签有序语义。
- 单/257 索引的 checked assign/add 对负值、extent、I32/I64 极值均返回 OutOfRange；不发布
  成功结果，输入/updates/旧状态不变，failed 槽退役、fresh 槽恢复。空 indices 保留原始 source，
  空 output 忽略坏索引 payload；整数/Bool add 与错误 update shape 在准备时明确拒绝。
  已接纳 native prefix 生命周期测试同时覆盖 Gather/Scatter，释放 Program/Invocation 后错误
  仍安全完成；既有 OOM/取消/晚期 native 异常/独立 Context/排空回归通过。
- 普通 F16/BF16 Scatter-add→Gather→mixed-F32 RMSNorm→residual 实测 rows/width 为
  1/128、17/2048、23/129、128/2048；两槽、三次变化输入，真实 native 结果作为下次状态输入。
  同 lower Direct 逐位对照和独立有序状态/FP64 norm 参考通过；源 Graph 释放、旧状态与结果
  保活、坏索引失败、新槽恢复和 close 通过。本批仍为诊断权重，不算完整 C1 真实权重验收。
- 统一 LOC self-check：Compute+Graph 实现 **43,260 行 / 174 文件**，不含 tests/vendor；
  [统计](../../../../evidence/correctness/compute-llm-moe-c1/s-implementation-loc.json)。Windows/Air
  selector 保留 112/110 项 skipped reasons；Air 单独 selector 首次因默认 SSH PATH 没有 cmake
  失败，补用验证脚本同一工具 PATH 后通过，未重建或重跑测试。完整 C1 真实权重、三交付/
  其他环境、等责 P1 和矩阵逐格验收仍待完成；下一具名路线缺口为 Vulkan，不签 C1 整包 PASS。

### C1-T：Vulkan typed storage 与逐元素计算（AUDITED）

baseline `f3552f36`，owner `/root` / `compute_route_vulkan` 与 `graph_route_vulkan`。F01/F03 的
半精度残差、gate 参数变换、mask 选择和原始存储搬运，需要先解除三个具名缺口：prepare_plan
只接非空 F32，transfer 拒绝非四字节倍数，以及 optional alias 被当作必须原位而整体拒绝。
现有 native Program/Invocation、录制、barrier、结果保留和故障排空继续复用。

核对来源为固定 ggml `456172ec` 的 Vulkan `types.glsl`、`generic_unary_head.glsl`、
`unary.comp`：已登记原版 shader 的布局为 rank≤4，半精度 typed storage 需要额外 Vulkan
能力；其 Softplus 与特殊值处理不能直接签本轮合同。保留既有已资格化 F32 GEMM/SiLU 核，
本批 `OWN` rank≤8/byte-stride 与 word storage 适配，`BORROW` 同一原生 SPIR-V 工具链及成熟
逐元素公式（GELU-erf 沿该 pin 的 A&S 近似）。半精度 RNE 使用整数位转换，不要求设备启用
shaderFloat16/Int8/Int16/Int64；I64 在存储操作中保留两个 U32，不承诺本批整数算术。

[Vulkan buffer copy 合同](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyBuffer.html)
和 [VkBufferCopy](https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCopy.html) 允许非零
任意字节长度；原 `%4` 限制不是原生 copy 的要求。shader 的 U32 storage 访问另要求物理尾部
padding，不能把二者混为一谈。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| T01 | Vulkan 原生 allocation/borrow/transfer/descriptor 支持六 storage、空 tensor 与非4倍数长度 | logical byte span 与4字节向上取整、至少4字节的 native storage 分开；自有分配自动padding，borrow显式要求足够物理容量与max(4,native alignment)对齐，不读写caller范围外。上传下载只传逻辑bytes；空传输无copy命令，仍检查closed/device/错误。quota和PlanFacts报告真实capacity；空输出保留合法owner，零dispatch程序录制/完成与寿命正常，无假payload |
| T02 | 普通 Copy/Fill/Select 接F32/F16/BF16/I32/I64/Bool，Cast/13Unary/6Binary 接三浮点 | rank0–8、native byte stride/广播/empty，dense output，合法非空byte/count域匹配原生32位索引/设备range。按共同schema冷推导后固定SPIR-V specialization，每个output word单writer，半精度/Bool尾块无原子写冲突；原始位storage操作、浮点F32求值后RNE。稳定Softplus/Sigmoid、GELU/SiLU负Inf极限、NaN/fmin/fmax/±0；未知精度明确拒绝 |
| T03 | 普通 Graph 共享T02；optional alias允许选择独立输出，保持MustAlias View拒绝直到单独规划 | native recorded dispatch与固定scratch复用，不加逐节点热解释。空输出无dispatch但有结果owner；旧F32 GEMM/SiLU明确保留dtype资格，不能因扩大计划storage误将half交给F32核。公开指南、borrow padding和原生资源成本清楚；源码list只扩本owner，既有vendor snapshot/pin不变 |
| T04 | 三浮点独立数值/codec与六storage位级oracle、普通Direct/Graph及生命周期 | scalar/0/1/127/128/129/257、2048宽、rank8、odd/zero byte stride、广播与Bool选择、F16/BF16舍入中点/边界/NaN/Inf/±0；空和1/2/3byte传输、borrow容量/对齐拒绝、保护区与metadata-only准备。half residual/Softplus/mask组合、两槽变输入、旧结果保留、关闭/abandon/现有OOM/native失败回归。W1真实Vulkan GPU及Air/Mini MoltenVK对应factors/公开头；完整权重/三交付/其他适用环境/等责P1随C1固定bundle |

本批不接 View 的必须别名规划、Concat/gates/norm/reduce/Softmax/TopK/索引/Scan；这些仍是本包
下一批具名算法/资源机制，不能改成缺席或由consumer重复实现。当前不改通用Graph、Mem、依赖pin、
公共构建治理或性能工具，不运行P1；owned shader/source编译只在Vulkan模块CMake内接线。

### C1-T 实施事实

- `83971269` 实现 T01–T04：六 storage 的 native word buffer、空/尾字节传输与结果 owner，
  Copy/Fill/Select，以及三浮点 Cast、13 Unary、六 Binary 的原生 SPIR-V。输入支持 rank0–8、
  odd/zero byte stride 和广播，输出 dense；F32 求值、半精度 RNE，存储操作保留原始位。
  普通 prepare、Direct Recorder 和 Graph 共享 lower；optional alias 使用独立输出，MustAlias
  继续明确拒绝。旧 F32 GEMM/SiLU 保留具名 dtype 资格，未向半精度误放行。
- Windows RTX 4070 / driver 610.62、Air `Mac16,12`、Mini `Mac16,10` 各 **17 Direct + 6 Graph**
  用例和两个公开头检查通过。三浮点独立 FP64/codec 参考、六 storage 原始位与 NaN payload、
  scalar/empty/rank8、0/1/127/128/129/257/2048、舍入中点/溢出/次正规数/±0、borrow 对齐/容量、
  metadata-only 准备和现有 OOM/native 失败回归均有实际结果。
- 普通 half residual→Softplus→Bool mask 图以 0/129、1/128、17/2048、23/129 的 rows/width
  运行；同 lower 的 Direct 录制仅作正确性对照，同时使用独立数学参考。两槽、三次变化输入、
  两输出、源 Graph 销毁、输入不变、旧结果保留、abandon 和 close 后读取通过。
- 预候选 shader 初编译发现 glslang 不接受 specialization 构造的全局 const 数组，改为显式
  八轴地址步骤后通过。W1 首轮唯一测试失败是旧卡要求 F32 Neg 不支持；`690d3582` 将拒绝域改为
  仍不支持的 I32 Neg，仅重跑 Direct，保留同产品的 Graph/头证据。未修改数值容差或运行算法。
- `bc7b6062` 补齐预声明的外部输出保护区检查：六 storage 的借用 slice 在真实非零 native offset
  写入，padding 以外的前后区域逐字节保持不变。三台机器各只补跑该变化用例。以上后续提交均未
  改产品实现；Windows/Air validation-layer 日志无 VUID/同步错误。Mini 使用已有 Air 安装的
  loader/MoltenVK/glslang 本地验证工具包，散列核对后置于隔离目录；未安装系统工具或验证层。
- 见[结果与全部尝试](../../../../evidence/correctness/compute-llm-moe-c1/t-vulkan-focused.json)及
  [原始日志包](../../../../evidence/correctness/compute-llm-moe-c1/t-vulkan-focused-raw.zip)。selector
  在 W1/Air 保留 110 项 skipped reasons；delivery/P1 等整包门未以本批结果冒签。
  [独立 LOC 工具卡](../build/loc-glsl-accounting-v1.md)补计此前静默遗漏的 GLSL，统一脚本对同一
  不可变产品统计 **43,605 行 / 177 文件**，包含 186 行 shader，不含测试/bench/第三方，
  [self-check PASS](../../../../evidence/correctness/compute-llm-moe-c1/t-implementation-loc.json)。
- 本批为诊断数据，真实模型权重消费、四路线共同逐格验收、三交付/其余适用环境和等责 P1 仍待
  完整 C1 固定 bundle；下一步仅审计 Vulkan 尚缺的具名布局/门控/归约/索引机制，不扩模型策略。

### C1-U：Vulkan 零拷贝 View 与别名资源规划（AUDITED）

baseline `1d1ed881`，owner `/root` / `compute_route_vulkan`、`graph_route_vulkan`。F01 的
reshape/slice/permute/broadcast 为 MustAlias，不能以 materialize 冒签。现有 Buffer 只有 descriptor
对齐的起点、liveness 逐 value 分配、结果 owner 在冷期绑定，无法承载 slice 的 byte offset 或
输入别名输出。先完成这个原生资源机制；Concat/gates/norm/reduce/index 等算法仍按后续具名批次实施。

证据：共同 `infer_view` / `view_alias_offset` 已有正步长、empty 和 span 验证；固定 ggml
`456172ec` 的 `generic_unary_head.glsl` 通过对齐 descriptor 加 shader offset 访问子区域。
其 F32 GEMM 不承诺任意 byte origin。本批复用共同语义和现有 SPIR-V/native Program，不更改 vendor。
[Vulkan descriptor 范围](https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorBufferInfo.html)
及 [command buffer 重新录制条件](https://docs.vulkan.org/refpages/latest/refpages/source/vkBeginCommandBuffer.html)
约束本批实现：空闲槽、允许 reset 的原生 command pool，复用原 descriptor/pipeline/存储。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| U01 | 自然 `Tensor::view`、普通 View prepare/Graph；六 storage，四种共同 View，真实别名 owner | 只建 host metadata，原 native buffer 和逻辑 offset 可见，无 payload copy/dispatch。flatten alias root 与累计 byte offset；计算 root 的最后读取/发布寿命包含全部别名，constant/root owner 不因零 dispatch 被裁掉。empty 沿共同 offset=0 合同 |
| U02 | typed shader 的 descriptor 对齐与 byte origin；固定 invocation 的原生绑定/录制 | descriptor 向下对齐、range 包含必要 prefix 和 word padding，输入 origins 用 12B push 常量；已有数值公式不变。仅当绑定的 origin 余数变化时在 idle 槽重录命令，复用 descriptor sets/pipeline/buffer；同 origin 更换 buffer 仍只更新 descriptor，submit 不分配/重录/搜索。host OOM/native record 错误拒绝接纳，未成功重录不得提交 |
| U03 | 别名输出保活、rebind 与固定 Recorder 语义 | alias metadata 冷分配；input rebind 更新内部别名但导出的别名/owner token 在释放前阻止覆盖。旧 root、独立槽、close/abandon 保活。Recorder 的 View 输出必须实际指向已外部绑定输入的对应范围，不将 caller 独立输出变成假别名；内部 root 无法由 caller 输出表达时明确拒绝。typed Recorder 不向 View slice 写 padded 尾字，使用独立或具名 borrow_ready 输出。旧 F32 GEMM/SiLU/custom 的 descriptor 对齐域保留，未资格化视图需显式 Copy |
| U04 | 独立位级 View/地址与寿命验证、普通 Direct/Graph 和指南 | 六 storage、reshape/permute/broadcast/正步长 slice/嵌套/rank8/empty/非零和奇数 origin，实测原 buffer identity/offset 与无设备分配；View→Copy/算术、constant view、root 最后使用后仍发布的别名、两槽/变绑定/旧结果/native token/close。录制计数证明正常重放不重录、余数变化才重录，并注入重录失败。W1 与 Air/Mini 两 factors/公开头；整包真实权重、三交付/其余适用环境和等责P1仍随C1固定bundle |

本批不改共同 TensorDesc、通用 Graph/Queue、依赖 pin、公共构建或统计仪器，不运行 P1。
View 的 descriptor prefix、别名保活和必要重录是显式成本，不签零成本执行或性能胜出。

#### C1-U 实现与局部验证事实

- `9532a7b3` 实现零拷贝 View、flatten alias/root liveness、常量和输入别名保活、descriptor
  对齐余数与原生 idle 重录；`38793bad` 修正 WERROR 局部变量遮蔽并补输入别名链换绑验证。
- U01 的源码 P0 发现常量 View 丢失不可变资格，导致下游成熟 MatMul 每次执行转置权重。
  `d2a764d2` 保留该资格和既有冷 packing 去重：两个等价视图共享一份打包，仅两个 MatMul
  dispatch 重放；只在发布原始视图时保留原始权重。独立数值及真实 owner 释放检查通过。
- 固定验证候选 `d0f9ca8c` 的 W1、MacBook Air、Mac mini 均通过 **20 Direct + 10 Graph = 30**
  用例和两个公开头检查；Mini 仅 correctness/portability。首轮 Windows 遮蔽警告和新增测试
  TensorShape 构造错误均保留 FAIL 记录，后者修正不改产品；运行期未改数值算法或容差。
- 六 storage 的四类 View、嵌套/广播/rank8/empty/奇数起点、计算根发布寿命、两槽与旧结果、
  导出 token、close、固定 Recorder 真别名和保护区、原生 Begin/分配计数及重录失败已验证。
  W1/Air validation layer 无 VUID/同步错误；Mini 复用已校验本地工具包，不冒签 validation layer。
- [局部结果与原始日志](../../../../evidence/correctness/compute-llm-moe-c1/u-vulkan-focused.json)
  绑定候选、命令、SHA、失败尝试和 selector 的 skipped reasons。
  [统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/u-implementation-loc.json)
  为 **43,833 行 / 177 实现文件**，包含 GLSL，不含测试/bench/第三方，self-check PASS。
- 上述只签 U01–U04 局部事实；Concat/gates/norm/reduce/Softmax/TopK/index/Scan、真实模型权重
  消费、完整 C1 三交付/其它适用环境及等责 P1 仍待实现和验收，两个 owner 保持 AUDITED。

### C1-V：Vulkan 门控与混合参数归一化（AUDITED）

baseline `4a79e0b3`；owner `/root` / `compute_route_vulkan`、`graph_route_vulkan`。真实消费仍为
C0 固定 Qwen 的 BF16 hidden、F32 learned norm scale、Gemma `1+gamma`、三类 gate 与 residual；
不扩 GEMM、模型专用 schema 或通用执行器。当前 gate 仅 dense F32 SiLU，norm 尚未接线。

有界研究使用既有 Native/CUDA 的共同语义及 ggml 固定 `456172ec` 的 `rms_norm.comp`、
`norm.comp`、`swiglu.comp`/`glu_main.glsl`。ggml 的行内并行 sum/barrier 可复用；其直接半精度
storage 扩展和四维寻址不能覆盖本路线已承诺的无可选窄存储、rank8/奇数字节布局。
`norm.comp` 使用 E[x²]−E[x]²，不能替代本包已有中心化方差；`swiglu.comp` 的
`a/(1+exp(-a))` 在 a=−Inf 时产生 NaN，而共同 gate 边界要求 activation 为 +0。
因此保留成熟 F32 GLU 调度/寻址，采用薄的稳定 activation 适配；新域复用 HYLE U32 storage。

| FIX | 固定实施范围 | P0 / 适用验证 |
|---|---|---|
| V01 | 三种 GatedActivation，packed-halves/separate，F32/F16/BF16；自然 Direct/Graph | 新域复用 tensor specialization/byte origins/逐输出 word 唯一 writer；F32 求值后只舍入最终乘积。rank1..8、合法非负/零/奇数 byte stride、empty/tail、NaN/Inf；无中间整张量展开或 host payload |
| V02 | RMSNorm/LayerNorm，最后维归一化，F32 累加，输入类型或 F32 的独立 scale/bias | 128-lane 行内 shared reduction，LayerNorm 先均值再中心化平方；RMS 两遍、LayerNorm 三遍输入。每个 workgroup 一行，半精度奇数 hidden 时同组两行以保证 word 唯一 writer；一次 dispatch，固定小 shared storage，无额外 device scratch；epsilon 与精度拒绝沿共同合同 |
| V03 | 保留有资格的成熟 F32 SiLU 接线与固定规则、公开指南 | dense/rank≤4/512-lane 能力且输入原生对齐在准备期可证明时仍用 ggml GLU shell，薄适配稳定 sigmoid 和特殊值；其余用 V01。普通动态输入的未知起点不假设对齐，不在 bind/submit 搜索。shader 编译只改本 owner 的 build_shaders.py 清单，不改 pin、公共构建或统计核心 |
| V04 | 独立数学/舍入、默认入口、Graph 资源生命周期及回归 | 三 dtype/三 gate/两输入形式；norm 无参数/scale/scale+bias、各 mixed dtype、0/1/127/128/129/257/2048 hidden、单/双/奇数行、rank8、广播/奇数 stride/View offset、零 epsilon/常量行/NaN/Inf/大均值小方差。固定 Recorder 邻字保护，两个槽/变化输入/旧结果/共享常量/close，W1/Air/Mini 两 factors 与公开头；三交付、真实权重与等责 P1 仍随完整 C1 固定 bundle |

承载为 `BORROW` ggml GLU shell/既有 tensor 原生录制与 `OWN` 共同数值、位存储和行末写入接线。
不签性能胜出；必要 native pipeline、shared reduction、输入读取和输出舍入均计为真实成本。
Concat/reduce/Softmax/TopK/index/Scan 仍是后续必须实现的机制，不由本批代签。

#### C1-V 实现与局部验证事实

- `40f4cbe3` 完成三类门控及两种输入形式、RMSNorm/LayerNorm 的 F32/F16/BF16 与独立 mixed
  scale/bias。普通 prepare/Graph 共享冷期原生接线；norm 一次 dispatch、固定 shared storage，
  半精度奇数宽度成对处理行，无额外 device scratch，最终存储只舍入一次。
- mature F32 SiLU 保留 ggml GLU shell，对齐可由 captured native buffer 或内部计算根证明时
  冷期采用；HYLE 薄适配稳定 sigmoid/NaN/Inf。其他门控使用已受审 U32 tensor shader。
  LayerNorm 采用中心化方差；本批不声称新数值适配或 norm 的 P1 已合格。
- `c62247ba` 只修测试的显式 Operation/Result 值构造。W1 随后 23 Direct、11 Graph 和两个
  公开头通过，唯一失败是旧测试仍期待拒绝 F16 gate；`82c221fb` 将该拒绝域改为尚不支持的
  F64，仅该用例重跑通过，其余证据按不变产品/测试复用。两次 FAIL 原样保留，不称首轮全绿。
- Air/Mini 在 `82c221fb` 各一次完整通过 **24 Direct + 11 Graph = 35** 用例及两个公开头；
  从产品候选开始未改算法/容差。W1/Air 无 validation-layer VUID/同步错误；Mini 仅 portability。
  空/尾块/奇数字节/偏移/广播/rank8、0/1/127/128/129/257/2048 hidden、所有参数形式、精确
  F32 affine 相消、特殊值、中心化方差和单/双/奇数行写入邻区均通过。
- 普通 gate→RMS→residual→LayerNorm 图为四 dispatch/三个复用输出 block，共享 F32 参数、
  两槽/变输入/旧结果和 close 均通过。[结果与原始日志](../../../../evidence/correctness/compute-llm-moe-c1/v-vulkan-focused.json)
  绑定候选、工具链、命令/SHA、增量 W1 证据及 selector 的 skipped reasons。
  [统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/v-implementation-loc.json)
  为 **43,990 行 / 178 实现文件**，含 GLSL、不含测试/bench/第三方，self-check PASS。
- 同步修正 Graph Vulkan 公开头中仍写 F32/SiLU-only 和全面拒绝 alias 的过时整行注释。
  删除整行注释后前后代码逐行相同，且无行号敏感 token；证明及 blob SHA 已写入上述结果。
  selector 的保守公共头构建选择如实保留；本次按用户的文档不重建规则跳过，不标为 PASS，
  不修改 selector。当前 docs/closure 正常执行，完整 C1 后续仍需实际交付验收。
- 仍未完成 Concat/reduce/Softmax/TopK/index/Scan 和整包实际权重、三交付/其他适用环境及等责
  P1；只签 V01–V04 局部事实，两个 owner 保持 AUDITED，C1 整包尚未签收。

### C1-W：Vulkan 变参 Concat 与字边界所有权（AUDITED）

baseline `7f9848f6`；owner `/root` / `compute_route_vulkan`、`graph_route_vulkan`。F01 的
slice/split/reassemble 与状态窗口 append 要求任意合法参数数量和六 storage 的精确拼接。
共同 schema 已处理 rank、正负 axis、非 axis 维、同 dtype 和溢出；本批不加公共 API。

研究限于固定 ggml `456172ec` 的 `concat.comp`/`generic_binary_head.glsl` 与既有 C1-K：
成熟方案一次读/写本片段，按 byte stride 寻址。Vulkan 现有 pipeline 有三个输入 bindings，
U32 storage 能跨平台支持六 storage。采用最多三个输入一组的预编译 recipe，不依赖可选
descriptor indexing，不扫描整个输出来寻找本组，也不做逐次增长的中间拼接。
空组跳过；其他组使用已有 compute read/write barrier。只有拼接接缝的共享 word 需保留先前组
已写 bits；未来组位置归零，避免额外 memset 和读取未初始化的逻辑结果。

| FIX | 固定实施范围 | P0 / 适用验证 |
|---|---|---|
| W01 | 变参 Concat 冷 lower/普通 Direct/Graph，六 storage、rank1..8、合法轴/stride/View | 每最多三个输入一组，常见双输入一次 dispatch；组内固定三个边界，输出 dense/new owner。复用 tensor byte origins、类型/维度/stride specialization，可提取本 owner 的共同准备辅助函数；所有地址/前缀和组数准备期固定 |
| W02 | 原生输出 word 所有权、分组片段写入、边界保留 | 每组只枚举自己的片段，O(elements·rank + groups·outer)；outer 行共享 word 时只由最早相交行写一次，覆盖该 word 中本组全部行的字节。保留已完成组的 bits，其余清零；不同组依赖已有 barrier，不增设备 scratch、host payload 或热期搜索。Uint32 模运算的轴前缀抵消须保证所选输入的最终地址位于已验证 span |
| W03 | 位级独立拼接 oracle、默认入口/资源/边界、指南 | 1/2/3/4/5/9输入，六 storage 原始位，首/中/末与负轴、0/1/tail、rank8、odd/zero stride、非零 View、重复输入、整组/全体为空与huge-empty；半精度/Bool跨行共享字、固定 Recorder邻区、输入输出alias拒绝、两槽/变输入/旧结果/close、现有OOM/drain回归。W1/Air/Mini两factors和公开头；真实权重/整包三交付/适用环境与P1仍随完整C1候选 |

源码 P0 先在固定范围枚举 **86,976** 个布局：arity1..5、轴长0..3、outer/inner各1/2/3/7、
标量字节1/2/4/8，验证组内每word唯一writer、触及word集合准确、前组bits保留和最终逻辑字节。
记录 `C:\hyle-scratch\llm-moe-c1-20260917\w-concat-word-ownership.json`；该枚举只证明本算法
的边界分工，不是 GPU 运行证据，不算路线/需求矩阵完成。真实 shader 和资源验证仍须运行。

仅改本 owner shader/冷 lower/编译清单和直接消费者；不改公共构建、selector、统计工具、
第三方 pin 或通用 Graph。Reduce/Softmax/TopK/index/Scan 仍为后续必需机制，不由本批代签。

#### C1-W 实现与局部验证事实

- `4449b77a` 实现六 storage 的变参 Concat：每最多三个输入准备为一次原生 dispatch，空组跳过，
  各组只写最终输出片段；组内共享 word 单 writer，组间通过既有 barrier 保留先前字节。
  不增加中间设备缓冲、memset、每组完整输出扫描或执行期 Provider 搜索。
- 同一不可变候选在 Windows RTX 4070、Air `Mac16,12`、Mini `Mac16,10` 各首轮通过
  **27 Direct + 12 Graph = 39** 用例及两个公开头检查；未因运行结果修改产品、测试或容差。
  W1/Air validation-layer 日志无 VUID/同步错误；Mini 仅 correctness/portability。
- 独立逐行拼接参考验证六 storage 原始位、1/2/3/4/5/9 输入、正负轴、奇数字节 stride、rank8
  广播、重复输入、整组/全体为空和 huge-empty。固定 Recorder 验证外部输出邻区及输入输出
  alias 拒绝；普通 Graph 将五个真实 View 重排后拼接，仅两个 dispatch、一个结果存储块。
  1/129/2048 宽度、奇数 native origin、两槽、变绑定、旧结果及 close 后读取均通过。
- [局部结果](../../../../evidence/correctness/compute-llm-moe-c1/w-vulkan-focused.json)及
  [原始日志](../../../../evidence/correctness/compute-llm-moe-c1/w-vulkan-focused-raw.zip)
  绑定三机候选、命令/SHA、前置 P0 枚举和 selector skipped reasons。
  [统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/w-implementation-loc.json)
  为 **44,139 行 / 178 实现文件**，含 GLSL，不含测试/bench/第三方，self-check PASS。
- 本批只签 W01–W03 局部事实；Reduce/Softmax/LogSoftmax/TopK、checked indexing/Scatter/Scan，
  以及完整 C1 真实权重、三交付/其他适用环境和等责 P1 仍待实施或验收。两个 owner 保持 AUDITED，
  64 行/256 格共同矩阵仍未签收，不把本批算作 C1 整包完成。

### C1-X：Vulkan 归约与 masked Softmax（AUDITED）

baseline `5eab611a`；owner `/root` / `compute_route_vulkan`、`graph_route_vulkan`。F04 的 Qwen
head128 L2、masked attention 概率和 router 归一化要求 Reduce/Softmax/LogSoftmax；沿共同 schema，
不增加模型策略或公共 API。本批重新定界上述 owner 的算法子集，T/U/V/W 的既有事实不代签 X。

有界源码核对固定 ggml `456172ec` 的 Vulkan `sum_rows.comp` / `soft_max.comp` 和已资格化 C1-I
数值合同：复用行内 F32 shared reduction、max→exp→sum 和行缓存的成熟方案。ggml 的四维寻址、
typed narrow storage、附带 sinks/ALiBi 及尾部半精度中间舍入不能替代当前 rank8/byte-stride 与
最终一次舍入合同。`BORROW` 原生 workgroup/shared/barrier 和上述归约结构；`OWN` 薄的 U32
存储、任意轴、mask/NaN/±0、LogSoftmax 与 word 所有权适配，不引入另一个 tensor runtime。

| FIX | 固定实施范围 | P0 / 必需验证 |
|---|---|---|
| X01 | Reduce Sum/Mean/Min/Max，F32/F16/BF16，任意合法 axes/keepdim/stride/View | F32 累加，half 显式声明；空集合分别0/NaN/+Inf/-Inf，NaN传播，Min/Max保留混合±0优先级。每组128线程，half成对输出归约后一次写word；一次dispatch、固定shared、零device scratch，O(N·rank)。rank0/空输出/huge-empty不能误读输入或截断有效维度 |
| X02 | Softmax/LogSoftmax，三浮点、任意axis、有限scale、广播Bool/同类型additive mask | F32求值/最终一次RNE；Bool false先屏蔽坏输入，未屏蔽NaN/+Inf、全-Inf行全NaN。LogSoftmax排除一个最大项后求tail，再用非负域补偿log1p，保留接近1的概率和大共同偏置；两次shared collective。width≤2048且设备shared容量足够时缓存F32行，其他域三遍读；每组最多两行，cache最多16KiB加2KiB归约数据，不展开整张F32张量 |
| X03 | 半精度任意轴的输出word所有权、普通Direct/Graph及原生资源 | F32、inner=1或half inner为偶数时一次dispatch；half inner>1且为奇数时先写每数学行8B的max/normalizer，再一次逐word输出。固定scratch参与既有liveness/容量/quota/barrier，超设备容量冷拒绝；不使用竞态窄写、额外memset或逐元素原子。布局/cache/dispatch规则准备期固定，执行不搜索/计时 |
| X04 | 独立FP64/位级参考、普通组合、生命周期、指南及三机定向验证 | 三dtype/四Reduce，任意/多/负axis与keepdim、rank0/8、odd/zero stride/View、空identity/输出；Softmax两模式/三mask、0/1/127/129/2048/2049/8193/32769、inner1/2/3、近1概率/ties/大偏置/NaN/Inf。borrow邻区/alias拒绝；BF16→F32平方和→epsilon→rsqrt→L2及masked概率，独立槽/变输入/旧结果/close、原生计数/已有OOM/drain。W1/Air/Mini两factors和公开头；真实权重/整包三交付/适用环境/P1仍随完整C1候选 |

只改当前 Vulkan owner 的私有 lower/shader/编译清单和直接消费者，不改共同语义、Graph运行时、
第三方pin、公共构建/selector/性能工具。TopK的多输出、checked indexing/Scatter/Scan依旧是
后续必需机制，不由本批代签。当前不跑P1；缓存、半精度特殊布局的额外读/dispatch/scratch均记为
真实成本，不将未比较的方案称为性能胜出。产品纠正最多两轮，按具体反例处理，不改数值门限。

X 首轮候选 `94252bf7` 的产品与 SPIR-V 编译通过，测试的 Bool 参考范围转换被 MSVC WERROR
拒绝；`c93c3b9c` 只显式构造参考值。随后 W1 **31 Direct + 14 Graph** 和两个头检查通过，
唯一失败用例有两个数值反例：大共同偏置 LogSoftmax 输出0而不是−log(4)，以及近1概率的 F32
结果−1.9361265e−7，相对独立参考−1.7476731e−7超门；BF16同样不合格。

第一轮产品纠正限定为 X02：原 SPIR-V 仅 scale 乘法与 `1+tail` 有 NoContraction，两次最终减法
和其余补偿没有顺序约束。[GLSL precise 规则](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html)
允许显式固定这些运算次序；源码的重排风险与大偏置反例一致，不将其当作测试不公平。
给 LogSoftmax 的移位、最终减法和补偿加 precise；小 tail `<1/64` 改用五阶交错 log1p 多项式，
解析截断误差不超过 tail⁶/6，避免该域依赖原生 log(1+tail) 的近1误差。补两侧边界回归，保留
现有全部门限；不加F64、额外dispatch/scratch，不改共同公式或其他owner。原始 FAIL 保留。

当前无 C1 整包冻结候选或整包 PASS。旧 CLOSED/BLOCKED 卡不授权本包改动。

#### C1-X 实现与局部验证事实

- `6be8e6a7` 完成上述一次有限产品数值纠正；W1、Air、Mini 各通过 **32 Direct + 14 Graph = 46**
  用例及两个公开头检查。小 tail 的 F32/BF16、大共同偏置及新增的1/64两侧回归均通过；没有
  放宽数值门限。首次测试编译失败和随后数值失败保留于
  [结果及原始日志](../../../../evidence/correctness/compute-llm-moe-c1/x-vulkan-focused.json)。
- 三浮点归约、任意轴和 mask、缓存边界/长行、非有限值、奇数 View 起点与输出 word 所有权，
  独立 L2/概率 Graph、双槽/变输入/旧结果/关闭、保护区和原生分配/录制计数实际通过。
  W1/Air validation layer 无 VUID/同步错误；Mini 仍只签 correctness/portability。
- [统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/x-implementation-loc.json) self-check PASS：
  Compute+Graph **44,473 行 / 178 实现文件**，含 GLSL、不含测试/bench/第三方。
- X 是自有 shader 的局部正确性事实，不是新 ggml Provider 接入或优选胜出。接下来按
  [成熟 Provider 复核](llm-moe-c1-provider-reuse-audit.md)处理现成计算链的接入边界，不直接新增
  自写 TopK/index/Scan。完整 C1 的实际权重、三交付、其他适用环境及 P1 尚未完成，owner 保持 AUDITED。

### C1-Y 首批依赖接入（AUDITED）

2026-09-18，baseline `ba5f0363`，`/root` 重新定界 `compute_route_vulkan` 的有限产品改动：
仅在中央 component 图增加 PRIVATE `compute_provider_ggml_vulkan` 边，由 Vulkan 路线开关
控制其第三方构建。实际 FIX/风险/验证见 [C1-Y](llm-moe-c1-ggml-vulkan-region.md) Y01–Y04。
Graph owner/公开 route API/执行器本批不变；设备/队列/完成桥接定界后再施工，不借 T–X 授权。

### C1-Y 普通 Context 设备归属（AUDITED）

2026-09-18，baseline `94b40163`，owner `/root`：`compute_route_vulkan` 新增有限 Y12，实际
FIX/反例/成本与适用验证见 [C1-Y 的 Y4](llm-moe-c1-ggml-vulkan-region.md#y4-同设备普通入口前置-fixaudited2026-09-18)。
普通 Context 采用 Provider 的唯一 native device，原生队列访问共锁，删除重复设备初始化。
保留当前公开 API、资源/状态/完成行为；managed Program/Graph lowering 与完成接线不由本批
代签。Graph 是必须复验的直接消费者，尚未新增产品改动。

### C1-Y managed 普通入口（AUDITED）

2026-09-18，baseline `ae39a9e1`，owner `/root`：`compute_route_vulkan` 与
`graph_route_vulkan` 新增 [C1-Y](llm-moe-c1-ggml-vulkan-region.md) Y13–Y15 的有限 FIX。
Compute 固定规则、边界资源、Recorder 与 Run/Completion；Graph 保留合法 DAG 和片段边界，
更新真实编译事实。Provider 只新增 PRIVATE compute_ops 语义依赖；三者不形成循环。公开执行
成本说明随实际原版 region 修正，未验收性能不写胜出；其他 C1 MUST 及整包验收保持不变。

`a7ebf205` 的 W1 static/shared 已分别通过38 Direct、16/15 Graph及公开头；Provider各19例，
无skip或VUID，首轮数值失败经一次有限纠正通过。完整事实与未运行项见
[C1-Y普通入口记录](llm-moe-c1-ggml-vulkan-region.md#y13y15-普通入口的-w1-局部验证)。
跨平台、真实权重、完整三交付与P1仍待完成；本卡保持AUDITED，不签整包或共同矩阵完成。

### C1-Y 成熟逐元素族绑定（AUDITED）

当前有限增量Y27（baseline `6900c5db`）已完成F32 Scan的SCOPED/AUDITED：provider接原版
cumsum，route补轴/stride/exclusive及原厂ragged-row边界适配，Graph作直接消费验证。
固定FIX和预算见[C1-Y的Y27](llm-moe-c1-ggml-vulkan-region.md)。I32受检Scan与Scatter未由本节授权，
不增加payload shader，不把F32接入计为整个F02.4或C1完成。
Y27候选`198cccba`的W1 static/shared、Air/Mini共430项回归现已通过，无产品纠正或P1。
当前仍缺I32受检Scan、Scatter、完整C1实际片段/三交付/优选淘汰；统一实现LOC为47,759行/193文件。

当前增量：2026-09-19，baseline `04643d18`，三个 Vulkan owner 对
[C1-Y 的 Y26](llm-moe-c1-ggml-vulkan-region.md) 完成受检 Gather 的有限审计：原版 GetRows、
GPU 索引检查、逐位布局适配及真实完成后错误回报。仅授权 Y26-A–D；Scatter/Scan 另行定界，
Y26-A–C现已实现；F1的W1 static/shared、Air/Mini局部回归均通过。真实宽2048的BF16片段
随后暴露锁定ggml的I32 GetRows工作组分母错误；Y26-F2已受审并作单值纠正，W1宽行位级回归
从FAIL转为PASS。固定候选d54ddc5a的W1 static/shared、Air/Mini共406项回归及Windows/Air真实
BF16片段的六条source/static/shared消费均通过；证据见Y26-F2结果节。此组停止重复验证，
全局selector门、Scatter/Scan、完整C1片段与优选淘汰仍未签收，CPU性能blocker不变。

2026-09-19，baseline `dff6a509`，owner `/root`：`compute_route_vulkan` 按
[C1-Y 的 Y17–Y19](llm-moe-c1-ggml-vulkan-region.md) 重新定界普通入口的冷 managed span
选择；Provider 同批映射原版五个 unary 与 Sub/Div/Repeat。公开 API/执行资源不变；Graph 作为
直接消费者保留 published/fork、native 边界与失败生命周期验证。未审性能不预称默认赢家，
未完成跨平台、三交付、真实权重或整包覆盖不写 PASS。旧 CPU ggml 与 oneDNN 的 BLOCKED 均保持。

### C1-Y28：Vulkan I32 受检 Scan（SCOPED / AUDITED）

2026-09-19，baseline `044ca8f3dc403de0958a2b83a6af5e2b04a38439`，owner `/root`。
产品owner仅`compute_route_vulkan`；`graph_route_vulkan`为直接消费者。C1/F02.4的counts→offsets
与已公开I32 Scan要求任意轴inclusive/exclusive、全部前缀（含exclusive最终total）溢出检查，
失败不写output。现有普通Vulkan入口明确拒绝I32，是该具名consumer的实际缺口。

复用裁决基于[Y27已停止的两来源审计](llm-moe-c1-ggml-vulkan-region.md)：ggml只有F32，
GPUPrefixSums具名实现只有U32模加，不能完整承载受检I32。此域`OWN`标准workgroup树形宽前缀
及checked适配，`BORROW`既有Vulkan布局、资源规划、shader嵌入、status、条件Copy与完成处理。
不导入第二运行时或宣称新Provider接入；不复制研究库的shader源码，F32继续原版ggml。

| FIX | 固定产品范围 | 证明与成本 |
|---|---|---|
| Y28-A | 单个私有GLSL程序的local-prefix/propagate/check三个冷specialization，256元素workgroup；两U32字表达有符号I64 | low相加及carry合入high；总元素≤INT32_MAX使任何总和绝对值<2^62，不需shaderInt64/subgroup或全局自旋。分层block总和前缀最多4层，反向合入carry；仅最终global prefix判I32，不能把局部block溢出误判为全局失败 |
| Y28-B | 冷lowering准备I64前缀/总和与I32临时输出，复用已有status字和最后条件发布 | L层有L个local、L−1个propagate、check/convert和conditional Copy，最多9个计算dispatch，另计必要layout Copy；O(N)。主要临时量8N+4N加各层递减的8字节总和/前缀及逻辑4字节status；全部quota/Invocation冷分配，热期不分配或下载payload。失败退役槽、新槽可用，整个算子输出保持不变 |
| Y28-C | 沿现有Scan冷轴/stride适配支持I32、rank≤8、byte origin；最后恢复布局仍用受检Copy | 复用F32实际使用的冷布局代码，不改公共API或热执行状态；空域无payload，宽1 inclusive Copy/exclusive Fill。冷拒绝I64 scratch的buffer范围、256 lanes/2048共享字节、U32地址与grid上限。Recorder沿现合同拒绝input/output overlap，不放松它以制造原位测试 |
| Y28-D | 独立I64 oracle、普通Direct/Graph与失败/寿命回归、指南 | 空/1/127/128/129/255/256/257/8193/65535/65536/65537；中/负轴、rank8、奇数/零stride与起点；正负溢出、暂时越界后恢复、仅exclusive最终total越界、后行失败全输出不变、局部block越界但global合法。counts→Scan→Gather→norm，两槽/5次变化/旧结果、failed/fresh、metadata限制/alias/OOM/close/drain；F32原版路线回归 |

256个uvec2共享槽恰为2048字节；local阶段只生成精确I64，不提前判错；最后check对所有行的所有
真实前缀置错，GPU屏障后才conditional Copy。exclusive输出取前一个global prefix，首项写零，
但检查仍包含最终total。输出布局恢复与checked发布合成一次现有Copy，避免错误时覆盖用户输出。

调查限于以上已固定来源和现有具名接线，不再搜寻泛库清单。产品纠正最多两轮，出现具体失败
先查成本/源码并保留反例，预算到界后裁决。验证：selector的Direct/Graph factors，W1 static/
shared、Air/Mini；shader能力检查、既有OOM/错误/生命周期同候选回归。W1/Air source/static/
shared实际C1片段与全局package门在本包固定候选完成；不由旧Gather交付代签。不在本子集跑P1，
完整C1沿原等责性能预算；CPU oneDNN资格BLOCKED、Scatter和其他C1尾项不被本节解除或授权。

#### Y28 实现与四构建局部结果

固定候选`f0115f96`完成I32宽前缀/全域溢出检查及条件发布，普通Direct/Graph已接通。
[局部验收](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-integer-scan/README.md)：
W1 static/shared分别98/97项，Air/Mini分别98项，合计391项；先行9项不重复计入。
本批无产品纠正或P1，shader只需baseline Shader能力。W1/Air无validation错误，Mini只签
correctness/portability；已有F32 Scan、OOM/部分接纳/drain在所选两factor中同候选回归通过。

独立I64参考与临时越界/抵消、exclusive最终total、局部block越界但全局合法、后行失败全输出
保持、轴/stride/origin、两槽/旧结果、quota失败回滚均通过；counts→Scan→原版Gather→norm
在普通Graph真实消费。status复用既有字的bit2诊断I32前缀越界，不增加运行对象或热分配。
这是成熟实现未覆盖的checked域补齐，不是新增Provider；浮点Scan继续采用原版ggml。

统一Compute+Graph实现为47,955行/194文件，不含测试/bench/第三方，自检PASS。证据核验后清理
旧Mini源码与Windows输入共123,819,830逻辑字节。本组回归停止重复运行；所有owner保持AUDITED。
Scatter、Scan三交付实际消费、完整C1真实片段/固定优选/淘汰和global门尚未完成，CPU oneDNN
资格BLOCKED不变，64行/256格矩阵未签收。C1整包完成通知尚不触发。

### C1-Y29：Vulkan 受检有序 Scatter（SCOPED / AUDITED）

2026-09-19，baseline `6cefad33`，owner `/root`；产品owner仅`compute_route_vulkan`，
Graph为直接消费者。F02.3及状态张量更新需要已公开Scatter的assign/add、重复索引顺序与
checked失败不写输出；普通Vulkan入口目前拒绝该操作。Y28验收不授权本节之前的Scatter实现。

有界来源核对仅限已锁定ggml `456172ec`的SetRows、GetRowsBack与现有Argsort/GetRows接线：
`copy_to_quant.comp`的SetRows并发写每个update，没有duplicate winner或checked边界；
supports_op只准入F32/F16更新，不能直接作为全部storage逐位assign。`get_rows_back.comp`
从零开始求匹配项和，再加base会改变顺序；每输出遍历全部索引，也不能承载逐次half舍入。
因此不直接注册两者为合格Scatter，不因此另写排序或通用payload搬运算法。来源调查到此停止。

| FIX | 固定实现 | 成本与必需证明 |
|---|---|---|
| Y29-A | 沿已有index check生成安全I32索引，再编码为严格单调、无NaN/零/subnormal的F32 raw keys；以现有TopK私有recipe仅导出完整stable Argsort的order | 对k<2^30取bits=0xff7fffff−k，其余取0x00800000+k−2^30；覆盖全部非负I32，禁止数值cast丢失大索引精度。Provider保留原版排序/allocator/runtime，未消费的values分支不展开；index数组/keys/order与原厂sort scratch全部计费 |
| Y29-B | Assign把base/updates冷规整为[outer,extent/count,inner]，既有Concat组成表；每个target二分stable order找最后更新，无更新取base，有更新取extent+update；用已有ggml GetRows布局适配完成全部payload选择 | 支持现有六storage、I32/I64 indices、任意轴/rank8/stride/origin。O(sort(count)+extent·log(count)+base+updates)，计入整表拼接、可能的轴物化/row padding/原版Gather/最终发布；不得把SetRows和GetRows组合称为零拷贝或性能胜出 |
| Y29-C | Add复用同一排序；keys/order用既有Concat合入8·count字节metadata，单个私有tensor shader按输出word独占，二分目标的匹配范围后依次累加 | 仅F32/F16/BF16；base先参与，half每项在F32相加后立即RNE回storage，precise限制重结合。必要updates物化、checked临时输出和最后条件Copy分别计费；O(base·log(count)+updates)，不对每个输出遍历全部indices、不用unordered float atomic |
| Y29-D | checked共用status/完成/失败槽；空indices直接Copy，空payload仍检查非空indices；更新指南、独立参考与普通Graph | count/合并buffer/padding/原版grid受设备范围冷检查；Recorder保持现有alias拒绝。六storage位级assign、三浮点有序add，重复/不排序索引、scalar/多维indices、任意/负轴/rank8/奇数与零stride、0/1/127/128/129/1025/4097索引、±0/NaN/Inf/half逐次舍入；高位I64及负越界全输出不变、failed/fresh、双槽/5次变化/旧结果、quota/OOM/close/drain |

只增加route私有lowering、tensor shader specialization与对应编译清单，复用既有4-buffer/
12-byte原生dispatch、GGML region和storage liveness；不改Provider源码/pin、公共API、执行器、
Graph产品、公共构建/selector/性能仪器。源码与已登记两条原版payload路径核对后停止泛库调查。
两个原生入口的准入与实际成本在同一冷计划固定，执行不搜索/计时。Recorder不承诺整段事务回滚。

验证限selector的Direct/Graph因素与shader能力检查，W1 static/shared、Air/Mini，既有生命周期/
失败排空同候选回归；普通Graph执行Scan→Scatter→Gather→norm并实际使用两个Scatter模式。
新功能三交付和真实C1片段与Scan一起进入本包固定bundle，不能用旧Gather交付代签，也不能
留到C7首次接线。产品纠正最多两轮；先具体失败源码归因，不放宽数值/顺序/失败责任。当前不跑
P1，完整C1沿原等责预算；CPU oneDNN资格BLOCKED和其他未签收项保持。

首候选之前的P0接线核对补充：原版Argsort用`ceil(log2f(float(ncols)))`计算padding，超过
2^21的相邻整数存在log2f舍入到整数的风险。Y29输入先按整数bit_ceil补齐到精确2次幂，padding
keys用NaN放在全部有效key之后，只导出前count个order；不改原厂排序。冷检查该padding的8字节
原厂scratch与各阶段workgroup/grid；4·padding的keys也计费。现有普通TopK的相同大宽风险是
具名后续核对项，未由Y29悄悄改动或写成已修复。本次是首次候选的冷规划完整化，没有消耗产品
运行失败的纠正预算。

#### Y29 实现与四构建局部结果

`07e30f13` 完成有序 Scatter：assign 复用原版 stable Argsort/GetRows，add 使用同一顺序，
保留逐项 half 舍入、受检失败输出不变及既有资源/完成机制。[局部验收](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-scatter/README.md)
共427项：W1 static/shared为107/106，Air/Mini各107；先行9项不重复累加，W1的source/MASM/
公开头另行通过。Air首次测试编译遗漏`<algorithm>`，`4c7d51c8`只补该include，产品源码未变；
失败日志保留，未消耗产品纠正预算，无P1。普通Graph的Scan→两种Scatter→Gather→norm、
六storage/三浮点、轴/stride/origin、顺序/位级语义、失败/新槽/旧结果/quota/关闭回归通过。

统一实现LOC为48,255行/195文件，不含测试/bench/第三方。此组局部运行停止重复；Scan/Scatter
真实三交付、完整C1片段/固定优选/淘汰/global门仍未签收，CPU资格BLOCKED不变。普通TopK的
浮点padding风险已由Windows实际CRT计算复现（2^21+1被补到2^21），后续按具名有限FIX修复；
这不是Y29的运行失败或新的泛库审计。C1完成通知尚不触发。

### C1-Y30：原版 Argsort 的整数 padding 修复（SCOPED / AUDITED）

2026-09-19，baseline `a302c004`，owner `/root`；产品owner为`compute_provider_ggml_vulkan`
与`compute_route_vulkan`，Graph为直接消费者。Y29的具名后续核对现有确切反例：Windows CRT的
`log2f(float(2097153))`为21，锁定ggml据此只排序2097152项；不是计时或测试门限问题。
[独立host复现](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-padding/host-reproducer.json)
记录19–26位边界；尚未把host结果写成GPU运行结果。普通TopK公开域没有排除该合法宽度。

有限FIX：Y30-A只在既有精确patch中把Argsort的padding指数改为无浮点的整数位宽计算，并用
无符号移位；原版排序shader、比较、dispatch、scratch生命周期与upstream pin不变。该语句在
原版每次录制计算，最多31次host移位，无新增分配或GPU工作。更新准确结果hash、依赖树锁和NOTICE，
不修改原始cache或公共构建机制。Y29已有整数padding适配本轮保留，不借修错作无关清理。

Y30-B只补普通TopK的冷padded scratch与原版workgroup/grid准入，同时覆盖managed region和
axis/stride适配入口，超设备范围在native录制前明确拒绝。复用现有错误类型，不增加公开API、
运行对象、热查询或替代排序。原版small路径没有全局8字节scratch，不能套用large路径错误限缩。

Y30-C先在修复前运行单个普通Direct反例：2^21+1宽，唯一胜者位于最后一项，独立期望top2；
随后覆盖2^21−1/2^21/2^21+1、升降序和stable tie，并在普通Graph串接已有逐元素再发布两个
TopK输出。加受控设备limit反例验证冷拒绝；小宽、NaN/位级、轴/stride、Scatter复用回归沿既有
测试。W1 static/shared、Air/Mini只跑selector适用provider/route/Graph及source contract；
记录首次失败，最多两次产品纠正。新候选的完整C1真实三交付及global门仍在本包完成，不由旧
Gather交付代签；没有P1授权，也不解除CPU资格BLOCKED。不继续调查其他未具名算子或全库边界。

#### Y30 实现与定向裁决

修复前`1c5f1df6`的普通Direct真实复现漏项：2097153宽的末尾唯一最大值4被遗漏，返回0。
`55fe0b8a`将原版host padding改为整数计算，并补managed/axis入口的冷设备准入；排序shader、
pin与执行机制不变。[四构建回归](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-padding/README.md)
共514项，W1 static/shared分别129/128、Air/Mini分别129/128；先行3项不重复计入。
大小边界两侧、stable tie、升降序及普通Graph Abs→TopK通过；已有Scan/Scatter/生命周期回归通过。

首次候选37fc82dd漏更新NOTICE的独立license摘要，配置被正确拒绝；该构建纠正记1/2，
算法纠正0，原始失败保留。未运行P1。统一LOC为48,294行/195文件；核验归档后删除旧输入和
dead patched-source副本共517,759,160逻辑字节。此定向修复停止追加调查/重复局部运行。

依赖锁使selector扩展为112个selected门；目前只完成具名局部门，global余项继续随固定C1
bundle完成，未记PASS。下一动作是原已授权的真实C1片段与Scan/Scatter source/static/shared
消费、完整成本优选/淘汰和余项验收；CPU oneDNN资格BLOCKED保持，矩阵及C1完成通知未触发。

### C1-Y31：真实张量状态片段与三交付（有限验证）

2026-09-19，baseline `27cf8f41`，owner `/root`。本节执行已授权的C1实际consumer与交付，
不授权新增产品FIX或P1。候选产品固定为Y30后的树；外部consumer、输入、oracle和命令在首次
运行前分别锁hash。若暴露产品问题，先保留反例并重新定界有限FIX，不直接扩实现或放宽标准。

- 输入：重新核验本地Qwen3.5-2B的13个文件；只提取实际BF16 embedding的34行×2048列及
  第0层Gemma RMS的F32(1+BF16权重)。不下载，不把模型加载纳入HYLE。固定token序列来自C0。
- 形状：`[B,T] = [1,1]/[1,17]/[1,128]/[1,23]/[1,129]/[2,23]/[2,129]`，后四为holdout；
  batch2保持三维payload，不能展平成单batch。两个独立状态流各连续五步，共享不可变表。
- 主链：I32索引差分→inclusive Scan→Gather→Cast F32→Gemma RMS→separate SiLU gate/value→
  residual Add→Cast BF16→Scatter assign→Scatter add→Gather。前一步完整state作为下一步输入。
  另外发布I32 exclusive Scan，以及实际scale派生F32输入的inclusive/exclusive Scan。
  这是明确组合的基础库验证片段，不宣称该状态更新链是Qwen完整forward。
- 同一数据/责任分别走普通Direct和Graph。每状态流一个保留快照槽、两个工作槽；全部冷准备。
  每步把两个工作槽绑定到同一批真实新输入，释放旧input引用后交替提交；计入这些绑定和槽成本。
  不用额外payload复制掩盖结果寿命或原位更新问题，早期结果必须逐位保持至Context关闭之后。
- 独立oracle：FP64标量RMS/SiLU/residual，BF16最终RNE，update同时满足既有
  `0.008*abs(expected)+2e-7`和最多一个相邻BF16编码。状态的独立上下界从这个既定update区间
  按assign最后写入、add逐项F32相加并BF16 RNE传播；不能给整条链套单算子误差后临时放宽。
  另以已独立合格的实际update逐位验证Scatter顺序/索引语义，明确区分两层证明。
  I32精确；F32前缀用FP64参考，误差不超过`3e-6*max(1,sum(abs(included inputs)))`。
- 生命周期：固定512MiB quota，冷热用量记录、工作槽重复不增长、两个state相互独立、原表逐位
  不变、负索引失败槽退役而新槽可用、旧结果关闭后可读、最终quota归零。各算子的完整OOM/
  部分接纳/drain边界复用Y28–Y30已锁同产品证据，不重复扩测。
- 交付：Windows和Air各source/static/shared三条真实消费，源码包身份校验、安装后重定位、
  隐藏producer与源码、仅依赖公开头和`hyle::graph`。DLL邻exe、PATH隔离、WERROR、无插桩、
  validation layer；重型步骤串行、并行度2。每条consumer timeout300秒；每个平台两次producer
  构建（source构建兼作static producer），不制造第三个完整build。Mini局部回归已覆盖，不再复制。

首次失败日志保留；consumer接线纠正最多两轮，数值标准不变；产品失败另行审计。本节结束就
裁决该组，不追加无反例调查。全局selector余门、四路线完整成本优选/淘汰和CPU oneDNN资格
BLOCKED仍单独处理；本组通过不能触发C1完成通知，也不签64行/256格或整模型完成。

#### Y31 六交付局部裁决

固定候选`a03f6786`的产品与`55fe0b8a`相同。[真实状态片段](../../../../evidence/correctness/compute-llm-moe-c1/vulkan-real-state-delivery/README.md)
在Windows/Air的source/static/shared六条交付全部通过：84组Direct/Graph、840次双流状态更新，
各84个负索引失败/新槽恢复；独立数值/区间/顺序、batch2、旧结果关闭后可读、原表不变、预热
quota不增长及最终归零均通过，无validation错误。两平台各只构建两次producer，安装重定位并
隐藏原源码/producer后消费；Windows DLL邻置且PATH隔离。无产品或consumer纠正、无P1。

Air首次launcher的Python路径不存在，尚未构建即失败；原始记录保留，改用已有系统Python，
产品/consumer/oracle/verifier均未变。证据核验后删除两平台临时producer/源码共1,853,985,523
逻辑字节，最新SDK/consumer/证据保留。统一实现LOC仍为48,294行/195文件，自检PASS。

Y31停止重复验收；完成的是本组Vulkan真实消费与新增Scan/Scatter三交付，未代签C1整包。
下一项继续原卡的全局selector余门与四路线完整成本资格/优选淘汰；CPU oneDNN资格BLOCKED
不变，64行/256格未签收。C1全部适用验收完成后再通知用户，并在C2之前等待提问。

### C1-Y32：已选全局门的固定候选验收

2026-09-19，owner `/root`，当前baseline `25bd596f`。这是Y30/Y31已登记的global余门验收，
不新增产品FIX或重开其他模块。Vulkan最后产品为`55fe0b8a`，Y31真实消费/六交付已完成；
其后只有文档、证据及CPU模块benchmark记录修复，Vulkan产品没有变化。

以原Y30 selector产品baseline `4b3ca598e7dacb752923658b3ff17bdaa4281b1c`对本节提交后的
不可变HEAD重新投影：第三方产品patch触发组件/交付闭包，原计划共112项。保存实际当前plan，
逐项执行或保留未运行事实；不手动删选中项，也不把旧候选的receipt改签新tree。

按原已登记门分三批串行：Windows 96项maintenance/configure/原生correctness/headers，
1项component-contract，15项默认产品三交付。复用固定`active-vk/static`的构建缓存；嵌套
delivery使用短的任务专属目录，低并行2。执行既有selected命令，原有依赖cache/工具链与
离线source overrides明确记录，不重写selector或公共构建工具，不新建一套gate规则。
Windows CRT/vendor/HYLE DLL在启动前邻置，禁弹窗、不请求UAC。任何失败保留原始日志并停止
该批，不用反复换配置寻找PASS；若发现具体产品问题，先回相应owner定界FIX。

Y31的真实BF16状态片段、Air/Mini同产品局部回归按现有身份保留，不再重复。本节不启动P1，
不推翻CPU Q2的BLOCKED，也不能代签四路线完整成本优选/淘汰或64行/256格。每批核验日志后
清理自己的重复producer，保留所需SDK、当前构建缓存和原始证据，避免临时目录累计。

#### Y32 固定候选裁决

固定候选`c733aa93`的[Windows所选工程验收](../../../../evidence/correctness/compute-llm-moe-c1/vulkan-c1-global-validation/README.md)
112项全部通过：96项maintenance/configure/原生correctness/headers、1项component-contract、
15项默认产品三交付，共45条真实source/static/shared消费。原生批次不重复，Y31真实BF16
片段和Air/Mini同产品证据保留；本节没有产品改动、P1或新Provider接入。

component-contract首次离线夹具在cache初始化前放置archive，被产品正确拒绝；保留失败后
使用现有cache初始化函数纠正夹具，仅该门重跑一次通过。没有更改公共工具或产品合同。
全部重型步骤串行、并行度2，DLL邻置、PATH隔离、禁错误弹窗。

原始证据与SDK hash核验后删除45个重复producer目录，释放9,492,224,026逻辑字节；保留SDK、
独立消费者、当前缓存和归档。统一实现LOC仍为48,294行/195文件，不含测试/bench/第三方。

Y32工程门停止重复；四路线完整成本资格、固定优选与自有实现取舍仍待完成，CPU Q2的
BLOCKED没有解除。矩阵与C1整包尚未签收；完成全部C1适用验收后再通知用户，在C2前等待提问。

### C1-Y33：先补 LayerNorm 成熟接入，再冻结整包性能候选

baseline `5f61546c`，owner `/root`。当前源码核对确认 LayerNorm/归约/Softmax 仍由自实现
承载，原 Provider 复核已将成熟接入列为余项。因此不提前启动整包 P1；本批三个 Vulkan owner
仅对 [Y33-A–D](llm-moe-c1-vulkan-layernorm.md) 重新定界并 AUDITED：原版 GroupNorm 承载
稳定 LayerNorm，沿既有转换/alias/region 适配 dtype/layout、设备限制与输出 owner，完成实际
普通入口和适用验收。无新归一化算法、公共 API 或 vendor patch；其他接入余项与 CPU BLOCKED
保持。本批审计不计新增已验收能力，旧路径的性能取舍仍随完整 C1 bundle 裁决。

Y33候选`4d255a51`现已完成本卡适用验收：W1 static/shared分别132/131项，Air132项，
Mini131项通过、专用验证层用例1项明确跳过；验证层风险由W1/Air覆盖。selected 9门通过、
111项未选理由保留。W1/Air本实际LayerNorm的source/static/shared六条消费共360例、
1080次执行通过，使用本地2B输入、独立参考和原门限，保活/quota通过。
完整首败、环境/consumer纠正与界限见[Y33终局](llm-moe-c1-vulkan-layernorm.md)。
停止重复本批验证，无P1或默认性能赢家裁决；归约/Softmax等成熟接入、CUDA/Apple覆盖核对、
四路线完整成本优选/淘汰及CPU Q2资格BLOCKED仍待处理，C1整包与矩阵未签收。
统一Compute/Graph实现48,424行/196文件，不含测试；本批归档后清理4,207,211,641逻辑字节。
C1全部适用验收完成后才通知用户，并在C2之前等待提问。

### C1-Y34：成熟 Sum/Mean/Softmax 接入（SCOPED / AUDITED）

baseline `3cd981b3`，owner `/root`。三个Vulkan owner现对
[Y34-A–D](llm-moe-c1-vulkan-reduction.md)重新定界：普通region及任意轴/三浮点/mask适配、
SumRows尾行分片、两处具名长行Softmax累计max修正和真实消费资格。依赖pin及公共合同不变；
Min/Max/LogSoftmax原有合法能力保留，未按本卡比较不得删除。先完成成熟接入，不提前启动P1；
CPU Q2 BLOCKED、C1全部适用验收与C1完成通知条件保持。审计时产品尚未实施，验证未运行。

Y34产品`05581df8`现已通过W1 static/shared的136/135项；测试编译修正`11804be7`通过
Air136项、Mini135项（验证层专用用例1项明确跳过），W1受影响Direct因子91项增量通过。
没有产品/算法纠正或数值/生命周期失败，Clang测试警告和首次PowerShell环境失败分别保留。
详见[Y34原生资格证据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/README.md)。
只完成本批模块正确性：真实2B L2/概率三交付、当前selected工程余门、C1完整成本优选/淘汰
与CPU Q2资格仍未完成，不回填共同矩阵PASS、不进入C2。统一实现48,737行/197文件；
归档后清理本批过期验证源码191,054,277逻辑字节，其中Windows49,192,304字节。

Y34的`923895c5`已完成W1/Air实际归约/Softmax/L2六条source/static/shared消费：432例/
1296次执行、原容差、双槽变输入、published中间值、旧结果/close/quota均通过；三环境共6个
Vulkan公开头门通过。Direct L2沿既有Recorder组合，单算子/Graph沿Invocation；消费端两次
有限纠正（显式构造、结果保活）保留首败，没有产品或公式修改。
[具体资格](llm-moe-c1-vulkan-reduction.md#实际张量三交付及公开头已完成)不代签剩余selected
工程门或性能。此阶段另清理1,868,242,674逻辑字节；C1和共同矩阵仍未签收，完成通知条件不变。

### Y34 所选工程门：已完成

候选`ec240ac9`的[112项所选工程门](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/engineering/README.md)
全部资格化：96项普通工程检查、1项component-contract、15项默认三交付（45条消费）。
既有Y34三个C++ suite及两个公开头按源码、产物、163条未变编译命令与机器身份复用；
Provider的源码/MASM组合另行执行，未把旧版本门直接代签为本候选。8项未选理由保留。
本次没有产品、测试、公共工具改动或P1；W1/Air此前432例/1296次实际GPU三交付不重复。

归档核验后清理45个重复producer，共9,492,226,425逻辑字节。实现仍48,737行/197文件。
本批验收到此停止重复；C1其余Provider覆盖、完整成本比较/条件规则/存量淘汰与CPU Q2
BLOCKED仍待处理，C1及共同矩阵没有签收。C1全部适用验收完成后通知用户，在C2前等待提问。

### C1-Y35：成熟逐元素及门控接入（SCOPED / AUDITED）

baseline `1e43875a`，owner `/root`。Vulkan 三个当前 owner 仅按
[Y35-A–D](llm-moe-c1-vulkan-elementwise.md)重新定界：Log/Sqrt/Softplus/Rsqrt/packed SwiGLU
接入、必要稳定数学修正及三浮点/布局适配，沿普通 Direct/Graph 和原厂 region。源码事实与
有限验证先固定后实施；旧 Y34 验收停止重复。本批不跑 P1，CPU Q2 BLOCKED 和 C1 通知条件不变。

Y35已实现；`6af6bfe4`的W1 static/shared模块资格141/140项及source contract5项通过，
实现48,943行/198文件。Air首编失败于同宽`size_t/uint64_t`的`std::min`类型推导，尚无GPU
结果；仅按[Y35-F2有限审计](llm-moe-c1-vulkan-elementwise.md)修复这一处类型。Mac原生、
本地2B三交付、所选工程余门及C1完整成本/条件路由/淘汰仍未签收，C1完成通知尚不触发。
