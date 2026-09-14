# Compute Provider Native v1：reference + HYLE SIMD 算子因子审计

## Closure card

- component / owner：`compute_provider_native` / Compute；round=`compute-provider-native-v1-factor`；
- baseline commit：`ec83514fe9d082698eb05018680cfaeda1822a5a`；
- trigger 与证据：当前 RMSNorm/MatMul/SiluAndMul 的 validation、reference route、SIMD whole-row kernel 和
  通用 provider/runtime 混在 `compute`；扩展算子目录会让 GPU-only runtime 强制取得 Platform/SIMD 和大量
  CPU kernels，也无法独立对 HYLE winner route 做资格化；
- 正常 consumer 任务：用户把 `native::provider()` 放入 CPU Context 的显式 provider sequence，冷路径
  prepare 一批 typed operations，在固定 tensor/storage 合同上重复 submit；不在热路选 ISA、推 shape、
  分配或静默 materialize；
- 主产品纵切：decoder 的 norm/linear/RoPE/attention/gated-MLP/top-k CPU fallback；正交纵切为通用
  elementwise/reduction/transform pipeline；
- scope：迁移已有 native validation/route/kernels，给 `compute_ops` 首批 `MUST` 目录提供正确 reference f32
  execution，并仅在已有 SIMD whole-row 或固定 kernel 符合同合同时接入；逐 op 登记 dtype/layout/alias/
  precision/workspace/capability，reference 与 optimized route 必须可差分；
- non-goals：拥有 Tensor/Operation schema、backend context/queue、oneDNN/ggml、GPU kernels、模型/KV/
  sampling/serving、为 benchmark 特设 API 或未经证据的微型 HYLE route；
- 直接依赖 / 消费者：候选公开依赖 Compute runtime，私有依赖 `compute_ops`、Math、Mem 与 SIMD；Platform
  是 SIMD 的公开依赖，不重复登记为本 factor 的直接边；
  直接消费者为 direct CPU Compute、Graph fallback 和 provider qualification；
- 性能：`P0` 冻结每项复杂度、allocation/dispatch/copy/ISA 账单；只有真实高权重 op 进入 `P1`，与成熟
  peer 比较同 dtype/layout/threading/output/completion，5% 非劣门；reference 正确性 route 不冒充 SOTA；
- 验证：每个 operation 的 normal/zero/tail/shape/dtype/layout/stride/alias/overflow/special values，scalar
  oracle 与 SIMD differential，provider fallback、OOM、并发 Invocation、逐头、三平台、sanitizers、三交付、
  factor-only、P0/有限 P1、红队；
- 预算 / stop：每能力族至多两个强 peer、每个 FIX 两轮、每条 P1 一次资格化；若一个 op 需要 provider-
  specific packed artifact 或模型策略，留给对应 provider/引擎，不污染 native 通用面。

## SCOPED 待审计项

1. 当前三项 native implementation 的逐入口 `MOVE/KEEP/REPLACE/DELETE`；
2. 首批 operation family 哪些必须有 reference 才能对外宣称可执行，哪些明确推迟给 provider card；
3. generalized input/output arity 后 prepared/invocation metadata 的冷/热成本；
4. SIMD whole-row/fat-kernel 可直接复用的清单，以及没有 material 证据时保留 scalar/reference 的边界；
5. CPU product façade、可选 oneDNN 顺序与 source/static/shared 构建投影。

AUDITED 前不移动或增加 native kernel 产品代码。

## AUDITED 结论（2026-09-12）

### 入口与路线裁决

| ID | 当前形状 | 裁决 | 冻结合同 |
|---|---|---|---|
| N10 | native 是 Context 隐藏的最终 fallback | `REPLACE` | `native::provider()` 是普通显式 Provider；没有特殊枚举或无法禁用的隐藏 fallback |
| N11 | 三项 validation/kernel 位于 `operation.cc` | `MOVE` | schema validation 归 compute_ops；CPU capability/prepare/reference/optimized route 归本 factor |
| N12 | `PreparedState` 固定 2 input/1 output | `REPLACE` | prepare 冷路径保存精确 descriptor arrays；Invocation 预建 bindings，submit 不增长 metadata |
| N13 | provider execute 只收裸 host pointers | `REPLACE` | 先验证 CPU-accessible domain并显式 map/resolve；opaque/device storage 返回 Unimplemented 供其他 provider |
| N14 | RMSNorm/Silu whole-row SIMD | `KEEP/MOVE` | 每 row 一次现有 SIMD entry；不在 element loop 再 dispatch |
| N15 | MatMul 两条 HYLE route | `KEEP/MOVE` | transpose-weight dot 与 fixed-target row-major kernel保留；general batch/stride 由 prepare 选择 tiled/reference route |
| N16 | 没有普通 CPU journey | `FIX` | `hyle/compute/cpu.h` 聚合 CPU backend + native provider，并提供 `cpu::create_context()`；高级用户仍可显式排 provider sequence |
| N17 | `native::Options::executor` 被 Context 保存但未参与 prepare/submit | `FIX` | prepare 把借用 executor 固定进 immutable plan；只对有足够工作量的独立 row/tile 做一次同步 fan-out，嵌套执行安全退化，空 executor 保持 caller-thread 串行 |

### v1 family 实现账本

`compute_ops` v1 的 19 个 family 全部必须有 CPU reference 才能宣称进入 Compute 产品；优化只复用已经
关闭的 SIMD 能力，不为 benchmark 发明第二套 public shape：

- direct SIMD whole-row：Cast 的 f32/f16/bf16/fp8 路线，Unary 的 GELU/SiLU/Tanh 等已存在 batch 路线，
  Binary add/multiply，RMSNorm/LayerNorm，Softmax/LogSoftmax，TopK 与 dot-based MatMul；
- fixed-target Compute kernel：保留现有 row-major MatMul；batch/stride 在一次 prepare 后落到固定 callable；
- reference/tiled CPU：View、Copy/Fill、其余 Cast/Unary/Binary/Select/Reduce、Gather/Scatter/Concat、RoPE、
  SDPA 和 SIMD 暂无同合同入口的边界形状；reference 不是性能宣传，但提供 differential oracle 与 fallback；
- SDPA reference 采用分块 online-softmax，workspace 与 O(B*H*Q*K) 时间、O(tile) 额外空间显式；不得先
  materialize 完整 attention matrix 再把内存成本藏起来。TopK 复用 SIMD selection 时保持冻结 tie/index
  语义，否则走 reference。

公开依赖冻结为 `compute` + `compute_backend_cpu`；私有依赖 `compute_ops`、Math、Mem、SIMD。
`native::provider()` 是小型无 owner 配置值；`cpu::create_context()` 只在这个聚合 factor 中出现，因此 CPU
backend 不反向依赖算子 provider。oneDNN fallback 示例必须显式传 `{onednn, native}`，顺序在 Context
create 后固定。

### 性能任务冻结

P0 对每 family 登记 exact arity walk、mapping、workspace 与循环复杂度；prepare 可分配一次 descriptor/
artifact metadata，Invocation create 可分配一次 lane state，submit 不分配、不锁、不选 ISA/provider。
P1 不铺 19 行纳秒测试，只冻结七个真实高权重合同：RMSNorm、MatMul decode、MatMul short-prefill、RoPE、
gated MLP、softmax/top-k 与分块 SDPA 纵切；peer 优先 ggml/oneDNN 的同 dtype/layout/threading 路线。
没有 profile 权重的 reference family只做 codegen/P0 与正确性，不以“未测即 SOTA”关闭。

## Implementation checkpoint（2026-09-12）

- N10--N16 的结构迁移已完成：native 是普通显式 Provider，CPU convenience 显式组合它；原有 RMSNorm、
  rank-2 MatMul 与 packed SiLU gated-activation 的 validation、artifact、reference/SIMD 路线已从通用
  `operation.cc` 搬入本 factor；
- prepare 保存精确 input/output descriptor spans，submit 接收预绑定 TensorView spans 与具体 Queue；旧固定
  2-input/1-output runtime state、隐藏 native fallback 与裸 host-pointer provider vtable 已删除；
- Windows factor tests、oneDNN 显式 `{onednn, native}` 组合、Graph 兼容纵切，以及 source/static/shared
  Compute consumer 已通过；
- `View`、`Copy`、`Fill` 已作为第一批 transform 路线实现：View 是严格校验 inferred byte offset 的零 kernel；
  dense Copy 复用 Mem resource copy，plain strided Copy 做 logical materialization；Fill 写入全部 scalar dtype 的
  精确 bits。三者在 prepare 冷路径选定直接 submit function，不在热路径按 kind 搜索，也不增长 metadata；
  transform 实现已从 provider orchestration 拆到独立内部单元，并有 typed prepare/零输入 submit 旅程。
- `Cast` 已实现 f32/f16/bf16/E4M3/E5M2 的完整 5x5 storage graph：同 dtype 复用 Copy，dense 异类型一次
  whole-buffer SIMD convert，plain strided 输入用同一 scalar oracle materialize；strided load/store 使用 byte
  copy，合法的非自然对齐 stride 不触发 C++ UB。Cast 明确为 out-of-place；旧 schema 允许 input/output
  等范围 alias 会令 strided materialization 覆盖尚未读取的元素，现已收紧并有回归。整数/Bool/f64 转换的越界/NaN/overflow 语义没有在 v1
  card 冻结，因此继续返回 `Unimplemented`，不以实现语言的偶然规则冒充产品合同。
- `Unary` 的 12 个 f32 mode、`Binary` 的 6 个 f32 mode 与 `Select` 已取得 executable route。dense 等形
  arithmetic 每个 tensor 只调用一次现有 SIMD；NumPy-style broadcast 与任意 plain stride 走有界 rank 的
  scalar reference，不在 element loop 做 ISA/provider dispatch。Select 对全部 1/2/4/8-byte scalar branch
  原样选择 bits。dense 等形输入可精确 alias 输出，映射只打开一次；broadcast/strided alias 会覆盖尚未读取
  数据，因此 submit 明确拒绝。`Minimum/Maximum` 冻结为 `fmin/fmax` 的 NaN 与 signed-zero 语义。
- `Reduce` 的 Sum/Mean/Minimum/Maximum 已覆盖任意 axis 集、负 axis、keep-dim、dense output 与普通 byte-
  strided f32 input；dense reduce-all/最后一轴 Sum/Mean 直接复用整段 SIMD reduction，其余走同语义 scalar
  oracle。空 reduction 的 identity、Mean NaN、first NaN propagation 与 signed zero 已固定；submit 为
  O(input elements * bounded rank + output elements)、O(1)，不分配、不锁、不重新选择路线。
- `LayerNorm` 已覆盖 dense f32 `[..., hidden]` 的 non-affine、scale-only、scale+bias 与 input/output 精确
  in-place；每 row 复用已冻结 SIMD primitive，scale-only 由 normalization 后的整 row multiply 组合，不在
  element loop 做 dispatch。typed prepare 自动推导输出，参数 storage 不得 alias output；空 hidden 是 no-op。

本卡仍为 `AUDITED`，且当前 family 进度必须按 **12/19 executable** 计算。现有 12 个代表 family 已足够
支撑 Graph v3 的真实多节点纵切；因此暂停横向补齐 Native 目录，先完成 constant/general arity/effect/
region/partition/liveness/workspace/report 等 Graph 合同，之后再由真实图缺口驱动其余 7 个 family。剩余
reference、逐 op shape/dtype/layout/alias/error 测试、七条有限 P1、三平台与 sanitizer 尚未实现；schema
存在不能冒充 native provider 已支持执行。

### Apple portability checkpoint（2026-09-13）

Graph v5 的真实 Apple 回放在 `submit_copy` 中发现一个只被 MSVC/GCC 漏报的未使用局部变量；它不参与
strided copy 的地址计算，也不属于任何公开合同。该变量已删除，Windows 的 Native provider 与 Graph
定向测试通过。此修正只解除 AppleClang `-Werror` 编译阻塞，不增加 family、不改变 route、复杂度、布局
或 steady path；Apple 候选回放的最终结果仍由对应 Graph 记录签发。

### Indexing checkpoint（2026-09-14）

- `Gather`、`Scatter` 与 `Concat` 已取得 native executable route，当前 family 进度为 **15/19**；
- Gather 对 dense source 以 `prefix × index` 次连续 suffix-block copy 执行，覆盖 embedding/token row gather；
  plain strided source 保留 O(output-elements × bounded-rank) reference。checked 模式在写 output 前完整扫描
  indices，越界不产生部分写；unchecked 明确把范围作为调用方前置条件；
- Scatter 先 materialize input0，再按 updates 的确定逻辑顺序 assign 或 f32 add；exact dense input0/output 可
  原地执行，非一致 layout 不宣称 alias。Concat 对 variadic dense inputs 逐 prefix 复制连续 axis×suffix 块，
  strided inputs 走同语义 reference；三条 submit 路线均无 allocation、provider/ISA search 或隐藏线程；
- W1-WIN/MSVC 19.50 factor-only WERROR 构建、逐头与 Native 定向测试通过，覆盖 dense suffix gather、越界
  预检、Scatter assign/add/exact alias 和 middle-axis variadic Concat。完整 product delivery、三平台、
  sanitizer 与 P1 仍等 19/19 candidate 后按卡片一次执行。

### Transformer checkpoint 与下层复用门（2026-09-14）

- `RoPE`、`Softmax/LogSoftmax` 与 `TopK` 已取得 native executable route，目录进度为
  **18/19 executable**；只剩 `ScaledDotProductAttention`。RoPE 覆盖 interleaved/half-split、显式
  i32/i64 positions、dense fat-kernel 与 plain-strided fallback；Softmax 覆盖 axis、scale、boolean/additive
  broadcast mask、log mode 与 exact alias；TopK 覆盖任意 axis、largest/smallest、稳定 index、双输出和
  caller-owned workspace。TopK schema 先拒绝无法由 i32 输出表示的 axis；undefined all-masked softmax
  明确产生 NaN，而不是伪造概率；
- RoPE 的 sin/cos 与 pair rotation 是 tensor/layout 融合路线，使用 Compute 私有、prepare 时一次选择的
  Highway fat kernel；它不是把模型 layout 下沉为通用 SIMD API。TopK 复用现有 SIMD selection，Softmax
  的自然 dense row 复用现有 SIMD；common dense trailing broadcast 也按完整连续 suffix 复用一次
  SIMD whole-row route；
- Mem 继续拥有 storage/map/coherence/copy/workspace，Math 继续拥有 checked size arithmetic，Platform/SIMD
  继续拥有 ISA fat selection；Graph 的 growable metadata、workspace/lifetime 和 intrusive owner 分别复用
  Container、Mem 与 Util-core。固定 rank/小型 descriptor 的 `std::array`、compile 冷路径排序和已映射连续
  bytes 的 `memcpy` 有不同 layout/lifetime/owner 合同，不为形式统一替换；
- 当前真实 consumer 暴露了 SIMD 的四组候选通用缺口：row `rsqrt`、同语义 elementwise
  `fmin/fmax/select`、以及同时满足 Compute NaN propagation 与 signed-zero 合同的 extrema reduction。
  现有 `simd::min_value/max_value` 在 signed zero 上不等价，已通过定向差分排除直接复用。它们必须形成
  独立 SIMD reopen card、冻结公共语义后才能实现；不得在本 factor 私造第二套普通 row primitive；
- `native::Options::executor` 当前仍只被 Context 保存、未被 immutable plan/submit 消费，这是 N17 的真实
  API/性能缺口。先在本 factor 消费既有 `ParallelExecutorRef`；只有适配 Sched persistent workers 时证明
  缺少可复用机制，才以该 journey 重开 Sched。预留的 `compute_sched` 只拥有可选 bridge，不替 Native
  provider 隐藏创建线程池。

本 checkpoint 的 Windows factor-only WERROR 构建、逐头、schema/native 定向测试通过；新增验证覆盖
RoPE vector blocks/tail 与 strided materialization、additive/all-masked Softmax，以及 TopK i32 axis 上界。
三平台、sanitizer、三交付、七条 P1、N17 和 SDPA 仍待 candidate 阶段统一完成，因此本卡保持 `AUDITED`。

### Attention 与 executor checkpoint（2026-09-14）

- ScaledDotProductAttention 已补齐 Native executable route，19 个 public family 现均有 CPU 路线。当前路线
  接收 plain f32 rank-4 Q/K/V、dense f32 output、BSHD/BHSD、不同的 Q/K 与 V/output depth、Bool/additive
  broadcast mask、显式 scale 与 bottom-right causal；窄 storage、block/opaque encoding 和非 f32 accumulation
  继续诚实返回 `Unimplemented`；
- 实现采用 64-score tile 的 online softmax，不 materialize Q×K matrix。每个并行 task 独占
  `value_depth * sizeof(float) + 64 * sizeof(float)` 的 64-byte-aligned caller workspace；时间复杂度为
  O(B·H·Q·K·(Dq+Dv))，额外空间为 O(tasks·(Dv+64))；dense Q/K dot、V accumulation 与 score/accumulator
  row 运算复用已冻结 SIMD，strided plain 输入保留同语义 scalar oracle；
- N17 已实际消费：prepare 把借用 `ParallelExecutorRef` 固定进 immutable plan，根据总工作量和 concurrency
  冷选择不超过独立 row 数的 task count；RMSNorm、LayerNorm、MatMul、packed gated activation 与 SDPA 每次
  submit 最多做一次同步 fan-out，已知 nested episode 串行退化，默认空 executor 保持 caller-thread 路线。
  这使用 Compute 既有通用执行 seam，未发现需要重开 Sched 的能力缺口；
- Windows factor/Graph 定向测试已通过，覆盖两种布局、多 batch/head、Q/K 与 V depth 分离、decode causal、
  两种 mask、all-masked/empty-key NaN、strided reference、四 task 独占 workspace、nested executor 与 Graph
  planned workspace。三平台、sanitizer、三交付及七条限定 P1 仍未执行，因此本卡保持 `AUDITED`。

## Candidate（2026-09-14）

- immutable product commit：`ca59798143345ef54ac8fa293d2eec799e721aa5`；公开 API、19-family capability、
  NativePlan、路由算法、executor 分片与 SDPA online-softmax 已停止变化；
- `AUDITED -> CANDIDATE` 前态 revision：同一 product commit（registry 仍为 `AUDITED`）；
- P0 账单：prepare 做一次 provider/ISA/shape/task/workspace 选择；submit 无 metadata allocation、provider/ISA
  search、隐藏线程、JIT 或 packing。SDPA 为 O(B·H·Q·K·(Dq+Dv))，workspace 为
  O(tasks·(Dv+64))；其余 route 保持各自已登记线性/矩阵复杂度。每个有资格的调用最多一次同步 fan-out；
- Windows MSVC 19.50 Release/WERROR 的 `hyle_compute_provider_native_test`、`hyle_compute_ops_test`、
  `hyle_graph_test` 与 `hyle_graph_executable_test` 通过。候选后只运行预声明的三平台、sanitizer、三交付、
  七条 P1 与一遍红队；任何产品修正先退回 `AUDITED`。

## 首候选撤回：族内档位覆盖（2026-09-14）

终局红队把“family 有一条可执行路线”与本卡已冻结的“v1 最小完整语义”逐项对账，
确认 `ca59798143345ef54ac8fa293d2eec799e721aa5` 不能作为 Native 终局候选：

| ID | 可复现问题 | 有限修正 | 验收 |
|---|---|---|---|
| N18 | `MatMul` schema 公开承诺 `[...,M,K] * [...,K,N]` 与 NumPy batch broadcast，Native 却在 rank 不等于 2 时直接 `Unimplemented` | 在 prepare 冷路固定 batch broadcast 索引与 dense/transposed-weight route；submit 对每个逻辑 batch 调用已有 kernel，无 metadata allocation 或 route search | rank-3/rank-4、左/右/双边 broadcast、零 batch、转置权重、overflow 与 Graph 纵切 |
| N19 | `GatedActivation` schema 已接受 packed/separate 与 SiLU/GELU/GELU-tanh，Native 只接受 packed SiLU | 复用已冻结 SIMD activation/binary row primitives，在 prepare 固定六种 f32 route；不增加 `fast/light` API | 两种 input owner × 三种 activation、in-place/overlap 边界、tail/special-value 差分与 Graph 纵切 |
| N20 | 中央组件图把 Platform 登记为 Native 的直接私有依赖，但产品源码没有 Platform include/call；所需 ISA facts 已由直接依赖 SIMD 的公开边带入 | 删除冗余 direct edge，保留 Math/Mem/SIMD 的真实直接边；不改变所选组件闭包或运行时代码 | exact-factor 解析、source/static/shared 链接与依赖文档一致 |

首轮结论中 `compute_ops` schema/API 未发现问题，但随后 N18 的真实批量转置权重纵切发现 O18；
该 factor 已独立返回 `AUDITED`。Graph 产品算法未发现缺陷，但其候选依赖身份也需刷新。
`compute_provider_native` 保持 `AUDITED`，只实施 N18--N20。先前已通过的跨平台、sanitizer 与交付
回放是历史证据，新产品候选必须重跑受 N18/N19 影响的正确性、交付、红队和预声明 P1；N20 只影响
exact-factor 与三交付依赖闭包。
