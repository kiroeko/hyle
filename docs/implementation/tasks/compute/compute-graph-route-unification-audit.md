# Compute/Graph：第三方广度与 HYLE 全局路由统一审计

> 本页记录 2026-09-14 的架构需求协商、源码审计、foundation-v1 架构基线和当前总目标的工作顺序。它不是
> 单个 component 的 closure card，也不自行改变 `module-status.json`；产品修改前仍须让受影响 component
> 依法返回 `AUDITED` 并分别冻结有限改动集。

## 两条需求与统一结论

1. HYLE 不可能经济地在所有平台独立实现所有算子，需要通过成熟 Provider 获得算子、融合和设备广度；
2. HYLE 又必须拥有 Graph、内存/放置、调度和最终路由，才能逐步用已证明更好的 HYLE 实现替换局部路线，
   并对端到端性能负责。

统一结论是**层级调度**：HYLE 唯一拥有逻辑 DAG、partition、target/storage/lane、跨 region copy/sync、
liveness/workspace 和最终 Completion；入选 Provider 只拥有一个 region 内的 artifact、私有 layout、局部
workspace 与内部 kernel 顺序。外包粒度可以是单 op、融合 region、block 或完整 engine，但每个粒度都是
HYLE 冷路径的候选。HYLE 不需要调度第三方 artifact 内每个 kernel，也不能在不透明 artifact 内偷偷替换
私有 leaf；渐进替换发生在明确的 HYLE boundary。

默认裁决不是“HYLE 优先”或“第三方优先”，而是：完整计划上存在合格、material 的 HYLE 胜利才替换；
没有可比证据、测不准或差异不足时使用已配置的成熟路线。steady replay 只运行不可变计划，不支付搜索、
计时、JIT、建图或常量 packing。

## Foundation v1 架构基线

下表只锁设计，不把尚未完成的实现写成通过。任何后续 closure card 都不能无证据改变这些结论。

| 维度 | 锁定结论 |
|---|---|
| 公开产品 | 只有 Compute 与 Graph；`graph_plan` 是 Graph 私有实现职责，不新增第三个 Plan 产品 |
| 生命周期 | Compute 固定为 Device/Context/Queue/Completion、Operation/PreparedOperation/per-run Invocation；Graph 固定为 Builder/immutable Graph/shareable Executable/per-run Invocation |
| IR 与 shape | 有限 typed DAG；Executable 声明 exact 或 bounded/guarded shape domain，Invocation 在运行前绑定 concrete shape；bounded route 要求 schema/provider 对整个 subdomain 给出 proof，否则 exact-only；无界数据依赖输出和模型循环不进 v1，active set、KV/batching/sampling 仍在引擎 |
| 全局/局部 owner | HYLE 唯一拥有 region boundary、target/storage/lane、跨 region copy/sync、liveness/workspace 和最终 Completion；Provider 只拥有入选 region 内 artifact、私有 layout/workspace/顺序 |
| 资源模型 | execution target、exact storage resource/domain、execution lane 三轴分离；shared Queue 与 delegated island 都是一等路线，不伪造统一 stream/allocator |
| Provider 接入 | leaf 与 graph-hosting runtime 都经版本化内部 SPI 提案；第三方私有 kernel 不偷渡。CPU/CUDA/Metal/Vulkan carrier 按 runtime owner 拆 factor |
| 路由空间 | 每个 node 的 leaf 与可收缩 node-set region 进入同一有界 `RouteCandidate` 空间；连续区间只是 fast path；完整 boundary 成本决定计划，覆盖数/dispatch/ordinal 只作最后 tie-break |
| workload | compile 显式区分 steady latency/throughput、预期并发与可选 replay 摊销；默认长寿命、单 Invocation、steady latency；memory/host-wait/compile 分别受硬约束，不混成魔法分数 |
| 性能证据 | 只消费同 HYLE boundary、完整 identity 的 qualified route profile；Provider 私有相对 ns 不可跨 Provider 排序；无证据或不可分时按成熟 Provider 顺序稳定回退 |
| tuning | 普通 `compile()` 不用 runtime input/output 执行 graph；Provider 可在私有隔离 scratch 内做有预算的局部 tactic/JIT 选择，immutable constant 只按显式 policy 读取。跨 Provider `graph_tune` 未来由调用方显式提供 bindings；两者都不是隐藏热路径 |
| identity/cache | OperationId 是 schema+version，不是 vtable 地址；Executable 绑定 closed shape domain，可含 immutable 有限 exact variant 表；constant 显式选择 borrow 或 materialize；跨 Executable cache 由消费方拥有，无全局隐式 cache |
| capture | capture 与 compilation 正交；exact capture key 可与 bounded/general route 共存，业务 batch mode 由引擎选择，steady submit 不临时 capture/JIT |
| 分布式边界 | rank/topology/communicator lifecycle、全局 batching 与恢复归引擎；Graph 只调度带 external communicator 的本地 collective participation 与 Completion |
| failure | selected-route prepare/compile 事务化；`Unsupported` 和明确 candidate-local、完整回滚的资源拒绝可在硬预算内重选；work 一经接纳只经 Completion 报错，不静默跨 Provider 重跑 |
| 热路径 | warm replay 不搜索、推 shape、JIT、建第三方图、pack 常量、隐式 allocate/copy/sync；每 segment 只产生真实完成边界 |
| 交付 | Provider SPI 是 HYLE factor 间 C++ 合同，不是稳定动态 plugin/Native FFI；source/static/shared 语义一致，可选 SDK/provider 独立发现 |

因此大对象模型、依赖方向、资源模型和调度权不再开放讨论。允许在实现卡中局部选择的数据结构、beam/DP
细节、厂商 API 调用和 profile 存储表示，不得改变上表语义。架构基线要靠 CPU shared region、CUDA async+
Q4 route、Apple delegated region/Metal interop 三类真实实现探针签发 foundation implementation candidate；
在该 gate 前不横向填算子或平台大全。

### 必须由纵切决定、不能在纸面伪造的部分

下表是剩余实验变量，不是未定义的产品边界。探针失败只先修正对应 seam；只有结果证明上表的 owner 或
lifetime 根本不可实现，才构成整体架构 reopen fact。

| 实验变量 | 已固定的不变量 | 签发探针 |
|---|---|---|
| node-set 的物理表示与选择算法 | Provider 语义是可收缩 node set；搜索有界、确定性，不宣称一般 DAG 全局最优 | 单节点、连续和非 insertion-order oneDNN region；重叠/离开后重入/预算耗尽反例；再裁决 interval fast path、small vector/bitset 与 DP/beam |
| `ShapeDomain` 的 C++ 表示与 guard dispatch | exact 与 bounded/guarded 都能表达；schema/provider 必须证明完整 subdomain；submit 不 JIT | CPU 动态 token 维度的 general route + exact tuned variant，越界/倍数/overflow/workspace 反例；再裁决 guard/variant table 表示 |
| constant materialize 的 owner 数据结构 | caller 明确选 borrow/materialize；逐 constant release point 可查，不能静默延长或缩短 lifetime | mmap borrowed weight、host-to-device materialized weight、部分 Provider packing、compile failure rollback 与 Completion lifetime |
| global selector 的实现和默认预算 | leaf/region/placement/boundary 同池；结构硬约束优先，只有 qualified profile 可跨 Provider 比较 | native + oneDNN gated MLP，构造局部 kernel 赢但完整计划输、prepare decline/reselect、profile 缺失/不可分 |
| CUDA capture 与 Q4 路由的具体 carrier | capture 与 compile 正交；exact key；接纳后不跨 Provider 重跑；HYLE 掌握 residence/lane/Completion | CUDA Q4 MLP 在 HYLE/成熟/unsupported、HYLE/Provider-owned storage、general/exact capture、多 lane 与 device error 上回放 |
| Apple delegated/shared 边界 | MLX island 与 native Metal 都进入同一 HYLE boundary；不得伪造统一 allocator/stream | MLX compiled region + Metal custom kernel/interop，在两台 Apple 机器验证 copy/sync、Completion、workspace 与并发 Invocation |
| 公共抽象与三交付成本 | direct op 不被 Graph 强迫；prepared/Invocation 热路径不搜索、不分配；source/static/shared 语义一致 | CPU one-op 同合同抽象税、跨 shared custom `OperationId`、provider factor 单独发现/未安装路径 |

因此实验后仍要确定的是数据结构、默认预算、provider-specific carrier 和是否达到 material 性能门；不再
开放的是 Compute/Graph 两产品边界、全局/局部 owner、三轴资源模型、生命周期、shape/capture 基本语义、
模型策略归属与热路径禁止项。

## 真实引擎与成熟 runtime 反证

本轮读取的是本地源码快照，不用项目名或二手总结代替实现事实：ggml `e91ded11bdcd`、llama.cpp
`e71b80510c84`、vLLM `51da0ca66c80`、SGLang `dcebe8c4733a`、ds4 `f62ca29a3087`。成熟 runtime 另以
ONNX Runtime、oneDNN、TensorRT、PJRT/IREE 与 MLX 的官方设计/API 交叉检查。结论只进入 HYLE 的 owner、
lifetime 和 seam，不复制 peer 的模型 API。

| 来源 | 可复核事实 | 对 HYLE 的约束 |
|---|---|---|
| ggml `src/ggml-backend.cpp` | 先按既有 residence、支持度、输入局部性与 backend 顺序放置 node，再形成连续 backend execution splits，插入显式 copy/event，并独立 reserve graph allocation | 连续 segment 是优秀执行 fast path，但不能反推 Provider capability 只能用 begin/count；HYLE 仍须拥有跨 segment residence/copy/sync |
| llama.cpp `src/llama-context.cpp`、`src/llama-graph.*` | `sched_reserve()` 用 prefill/token-generation worst cases 预留；graph 只在 `llm_graph_params` 可复用时重用，否则重建；用户配置拥有 offload/split 与 batch 语义 | Graph 要能预报/预留 bounded workspace，shape/拓扑 identity 要显式；模型 offload、batch 与 reuse policy 留给引擎 |
| vLLM `docs/design/torch_compile.md`、`cuda_graphs.md` | 同时保留 general symbolic compile 与 `compile_sizes` exact tuned kernels；CUDA Graph 的 FULL/PIECEWISE/NONE 与 compilation 解耦，并由 runtime mode + `BatchDescriptor` 选已捕获路线 | Executable 不能只支持 exact；bounded/general 与 exact variants 要共存；capture 是正交 replay 能力，业务 batch class 不下沉到 Graph core |
| SGLang `tc_piecewise_cuda_graph_backend.py` | attention 周围切 piecewise graph；启动期对 capture shape 做 dummy/JIT warmup，共享 graph pool，再捕获/重放 | compile/capture/warmup 必须是显式冷路径，有独立内存 owner；steady replay 不能临时 JIT 或分配 |
| ds4 `ds4_cuda.cu` decode-island path | 只捕获 position-independent islands；首见 key 先 eager warm，次见 capture，之后 replay；key 含 layer/island/variant 与 pointer identity，失败在未接纳 work 时回到 eager | exact capture key、分段 capture、pre-acceptance fallback 和 work-accepted 边界必须进入合同；不能要求整图都可 capture |
| ONNX Runtime / oneDNN Graph | EP/partition 用 node index/ID 集合描述可支持 subgraph/partition，而不是拓扑区间；compiled artifact 与每次执行 scratch 分离 | Region SPI 使用可收缩 node set + boundary ports；compiled artifact 与 Invocation/workspace 生命周期继续分离 |
| TensorRT / StableHLO/PJRT | optimization profile 用 min/opt/max shape；engine 与 execution context 分离；builder 可 timing tactics，StableHLO 区分 bounded/unbounded dynamism | shape domain 同时容纳 exact 与 bounded/guarded；允许 Provider 私有隔离调优，但 Graph 不因此执行用户 graph |

主要官方入口：[ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)、
[ONNX Runtime high-level design](https://onnxruntime.ai/docs/reference/high-level-design.html)、
[oneDNN graph partition](https://uxlfoundation.github.io/oneDNN/v3.13/class_dnnl_graph_partition.html)、
[TensorRT dynamic shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)、
[TensorRT optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)、
[StableHLO dynamism](https://openxla.org/stablehlo/dynamism)、
[PJRT C++ API](https://openxla.org/xla/pjrt/cpp_api_overview) 和
[MLX compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)。

## 当前源码勘验

源码 baseline：`49d32deacf80fbc046e53be64de47042144edc5a`（架构文档提交未改变前一轮勘验的产品树）。

| 需要的机制 | 当前事实 | 裁决 |
|---|---|---|
| provider-neutral tensor/op | `TensorDesc` 已分离 logical dtype 与 blocked/opaque physical format；`Operation` 是 typed schema | `KEEP` |
| 冷 prepare 与热 artifact | Provider 先 `propose`，再对 exact resources `prepare`；`PreparedOperation`/Invocation 分离 | `KEEP` |
| shared/delegated runtime | 单 op 与 region 均声明 Queue ownership、workspace、threading、Completion endpoint | `KEEP` |
| 第三方 compiled region | versioned `RegionProviderVTable` 已能 propose/compile region，Executable 持有 artifact；但身份仍是连续 begin/count，不能表达成熟 peer 的 node-set partition | 生命周期 `KEEP`，proposal 表示 `FIX` |
| 全局 target/storage/lane/liveness | Graph 已显式规划三轴、copy、happens-before、workspace 与 report | `KEEP` |
| shape domain | Graph IR 有 `DimensionBounds`，但 `CompileOptions::shape` 要求每个 symbol exact-once，Executable/Invocation 只能固定一个 point | `FIX`：增加 closed exact 或 bounded/guarded domain；不在 submit JIT |
| capture 正交性 | 已有 `ReplayMode::kCommands/kPreferCapture/kRequireCapture` 与 hybrid route 骨架，但它只建立在 exact specialization 上，尚未证明 bounded/general route 与 exact capture variant 共存 | 骨架 `KEEP`，shape/capture 组合 `FIX` |
| constant lifetime | 当前 `ConstantBinding` 一律保守借到 Executable 与全部 Completion 结束，即使 Provider 已报告 packed copy；aggregate count 不能证明哪些 source 可释放 | `FIX`：显式 borrow/materialize，逐 constant 报 owner/release point |
| 单 op 与 region 同池竞争 | target placement 先因 `region_support` 布尔值偏向能提 region 的 target，operation fallback 再按 Context provider 顺序准备，region 随后独立做 interval selection | `FIX` |
| 完整成本选择 | selector 只使用 provider 自报 savings、覆盖数、dispatch 和 provider rank；已有 copy/sync/workspace/persistent/host-thread facts 主要只进 report，选中后得到的 exact artifact facts 不触发重选 | `FIX` |
| 可比成本来源 | `estimated_net_savings_ns` 相对各 Provider 自己的 fallback，不能跨 Provider 相加；没有资格化 profile identity | `FIX` |
| 单节点 graph-hosting Provider | region discovery 拒绝 `nodes.count < 2`；只能另走 delegated operation seam，不能统一比较 | `FIX` |
| route evidence seam | 没有显式 workload intent 或 qualified route profile；当前 compile 也不能安全取得代表性 payload 来调优 | `FIX`；compile 保持无执行副作用，未来 tuner 独立 |
| route 诊断 | CompileReport 描述胜出 step/fallback，但没有有界 shortlist、拒绝原因和所用 evidence identity | `FIX` |
| typed custom op 三交付身份 | `OperationId` 被保存但 `custom_as<T>()` 比较模板静态 vtable 地址；default-hidden shared factor 间地址不构成合同 | `FIX` |
| ggml backend 身份 | 当前无后缀 identity 实际只记录 CPU blocker；不能代表 CUDA/Metal/Vulkan | `FIX`（新 factor 时） |
| CUDA Q4 可执行能力 | CUDA backend runtime 已存在，但没有接受 Q4 `StorageFormatId` 的 CUDA operation/region Provider | `MISSING`，不是 runtime 缺陷 |

这说明现有 Compute/Graph 的大生命周期、storage、异步执行和 plan 骨架没有被反证；被反证的是**冷路径
route selection 层**、region/shape 表示上限和 custom-operation 的 shared 交付身份。继续填算子会把错误的
选择机制固化，整套推倒重写同样没有依据。

## 有限解决方案

### R1：修正 typed custom operation identity

`compute_ops` 以 `OperationId`（其中包含 schema/version identity）和 descriptor size 校验类型，不再以
inline vtable 地址判断。traits/infer 仍是冷路径函数指针，descriptor 仍保持有界、trivially-copyable、
heap-free。增加 source/static/shared 跨边界构造与 Provider 解码测试；不建立全局 registry 或 Native FFI。

### R2：把单 op 与 region 归一为冷路径 `RouteCandidate`

Graph 内部候选至少包含 canonical node set/fingerprint、显式 boundary ports、Provider/version、target、
resource/layout requirements、runtime mode、
workspace/persistent/packing、boundary copy/repack、sync/dispatch、host blocking/private threads、precision/
alias/failure/completion 与证据 identity。集合须无路径离开后重入，Graph 验证收缩后仍为 DAG；
普通 Provider proposal 产生单节点候选，region extension 产生一个或多个节点的显式 proposal。连续区间可
走 fast path，Graph 不枚举任意子集。只有选中后才构造重型 artifact，避免为所有候选 JIT/packing。

冷编译按有界闭环执行：先枚举 candidate skeleton 与资源要求；再为有限的 target/storage 组合计算 Graph
拥有的 boundary copy/repack/sync；随后选择候选计划并只 prepare 胜出 artifact。Provider proposal 对选择
所需结构成本给出 exact、保守上界或 unknown，artifact 必须验证该承诺。selected-route prepare 返回
结构化 success、无副作用 candidate decline 或真正 error；只有 `Unsupported` 或明确 candidate-local、
已完整回滚的预算 decline 才把候选排除并在预声明 retry budget 内重新选择。事实超过已接纳上界是合同
违约而不是普通 fallback；不能把不一致成本留到 report 后仍执行原计划，也不能预先 JIT 所有候选来换取
一个“完美”评分。

Direct Compute 继续使用显式 Context provider 顺序与 `Unimplemented` fallback，因为它没有全图边界信息；
Provider 应只在自己愿意按该顺序拥有的精确域内提案。Graph 则必须把这些 leaf proposal 与 region proposal
放入同一有界计划搜索。无需为此增加第三套公开 runtime 或 `Plan` 产品。

### R3：结构成本、资格化 profile 与显式 tuner 分层

1. 先按语义、精度、storage/import、host-wait、并发和内存预算排除不可行路线；
2. provider 只报告可审计结构事实，不提供可跨 Provider 比较的任意相对 ns；
3. 可比较执行估值来自绑定 provider/build、device/driver、dtype/format/layout、shape domain 与完成边界的
   evidence profile；差异不可分或不足 material margin 时选择已配置的成熟路线；
4. 普通 compile 不以 runtime input/output 执行 graph；Provider 可在私有隔离 scratch 中作有预算的局部
   tactic/JIT search，immutable constant 只按显式 policy 读取；但须报告开销、cache identity 和确定性政策，
   不能借此替 Graph 比较全局候选；
5. 未来独立 `graph_tune` 只有取得调用方代表性 bindings、允许执行的纯/确定性范围和独立预算后，才测量
   结构筛选后的少量**完整候选计划**并产出 profile；它不阻塞 foundation v1；
6. 一般 DAG 不宣称求全局最优。compile 有确定的 candidate/state/prepare-retry 预算，预算耗尽明确失败，
   不以静默截断改变计划，也不在 warm path 继续试探；未来 tuner 另有独立预算。CompileReport 只保留
   有界 shortlist 的身份、拒绝原因、成本来源和最终裁决，不复制无限 proposal 日志。

### R4：按 runtime owner 拆第三方 carrier

CPU、CUDA、Metal、Vulkan 的 ggml adapter 使用独立 component、feature、依赖和验证。CPU 的 abort/threadpool
blocker 继续只约束 CPU；CUDA route 必须另证 buffer import/ownership、stream/event、device error、capture
和三交付。调用私有 ggml kernel 不是普通 adapter；必要时用独立、版本锁定的 source-patch factor 承担维护
成本。

### R5：用 CUDA Q4 MLP 纵切验收，而不是用空 selector 测试

同一 Graph 至少产生 HYLE Q4 leaf、成熟 CUDA/ggml leaf 或 region、以及显式 unsupported 三类路径。测试要
覆盖权重位于 HYLE CUDA storage 与 Provider-owned storage 两种 residence，并证明：局部 HYLE kernel 胜出
但拆分成本更大时保留 region；完整成本 material 胜出时切到 HYLE；无合法 Provider 时返回
`Unimplemented`；CompileReport 能解释选择；warm replay 无 route search、建图、packing 和 metadata allocation。

### R6：shape domain、capture、constant 与 cache owner

一个 Executable 对应一个 closed shape domain，可以是 exact，也可以是 bounded/guarded。后者可包含真正
接受范围的 artifact，或 immutable 有限 exact variant 表与已编译 general fallback；所有 guards 必须有限、
互斥并完整覆盖声明 domain，schema/provider 无法证明 bounded subdomain 时明确 exact-only。Invocation 在用户工作
接纳前绑定 concrete shape，warm submit 不 JIT。CUDA Graph 等 capture 只绑定 exact shape/storage/Queue 和
Provider 声明的稳定 key；compile 与 capture 正交，Executable 只共享 immutable recipe，address/workspace/
run state 属于 Invocation，业务 batch class 的选择留给引擎。跨 Executable specialization cache 也留给
引擎。compile-time constant 显式选择 borrowed immutable 或 materialize：前者
source 活到 Executable/Completion 终态，后者要求 compile 成功前 artifact 已拥有 replay 所需 bytes，随后
source 可释放；report 按 constant 披露，不以 aggregate packed count 缩短 lifetime。可变权重建模为 input 或
重新 compile。Graph 不读写进程全局 route/plan cache，qualified profile 由调用方显式传入。OperationId
只标识 schema/version；custom instance 没有稳定 fingerprint 时不得进入持久 profile。

### R7：只有接纳前可以回退

candidate decline、prepare failure 或 submit 的明确 pre-acceptance failure 可以在有限、已冻结路线中回退。
一旦 Queue/Provider 已接纳用户工作，错误只经 Completion 传播；Graph 不在另一 Provider 静默重跑，避免
重复 side effect、资源写入或部分完成。

## 是否需要再次重构

需要，但只做**定向重构**：

- `compute_ops`：从 `CANDIDATE` 返回 `AUDITED`，只修复 custom descriptor 的跨 DSO identity。后续所有
  bounded-domain 支持均是 Graph domain 与具体 provider 对完整 subdomain 的承诺，不能把 provider-specific
  可执行能力误写成 `OperationTraits` 的静态布尔属性；
- `compute`：从 `CANDIDATE` 返回 `AUDITED`，因为现有 leaf proposal 缺少与 region 同池所需的结构事实和
  shape-subdomain guard、事务化 decline 分类；Device/Context/Queue/Completion 生命周期不重写；
- `graph`：从 `CANDIDATE` 返回 `AUDITED`，统一 leaf/region route candidate、workload/profile、成本来源和
  有界重选，并引入 exact + bounded/guarded shape domain；
- `compute_provider_native`：从 `CANDIDATE` 返回 `AUDITED`，只迁移新版 proposal seam 并参与探针；
- `graph_provider_onednn`：当前虽为 `AUDITED`，其活动卡尚未包含 node-set region；产品修改前必须绑定新
  baseline，把该迁移和非 insertion-order node-set 探针写入有限集合；
- `compute_provider_onednn`：Graph 的 node-set 表示本身不会重开它，但 Compute v8 会修改它直接实现的
  私有 proposal/outcome SPI，且前轮尚欠真实 region consumer，因此以 v5 新卡保持 `AUDITED` 并共同迁移；
- CPU/CUDA backend、Tensor storage、Graph 生命周期/liveness/workspace 骨架没有重写依据；
- backend-qualified ggml CUDA/Metal/Vulkan factor 在各自 closure card 成立后从 `QUEUED` 开始，不恢复旧
  adapter，也不让当前 CPU blocker 冒充它们的状态。

在上述状态迁移和有限卡片完成前，不继续横向填充算子，也不把现有 `RegionSelectionPolicy` 的最大覆盖
结果称为自动 SOTA 路由。

## 当前 Compute + Graph goal 的重排

当前总目标不变，但执行依赖改为下列顺序；前一步未给出合法 candidate 时不提前运行后一步的重型 P1/P2：

| 顺序 | 工作包 | 当前裁决 / 完成条件 |
|---:|---|---|
| 0 | Foundation v1 架构基线 | 本页和两份 architecture 固定上表全部系统决策；docs/closure contract 通过，不改 registry/product |
| 1 | 合法状态迁移与小卡 | `compute_ops`、`compute`、`graph`、`compute_provider_native` 分别 `CANDIDATE -> AUDITED`；`compute_provider_onednn` 与 `graph_provider_onednn` 绑定新卡，不能沿用旧 proposal/interval 范围 |
| 2 | Identity + proposal/shape substrate | R1 custom identity；node-set leaf/region 共用结构事实、decline 分类、workload/profile、provider 的 bounded-domain proof/exact-only、constant borrow/materialize seam；不先写 selector 大算法 |
| 3 | Graph route unification | leaf/region 同池、contractibility、target/storage/boundary 完整成本、有界 state search/reselect、可解释 shortlist/report |
| 4 | CPU foundation probe | direct/one-op tax + gated MLP；oneDNN shared-runtime direct/region 接受非 insertion-order 的合法 node set；同一 Graph 验 exact 与 bounded route/workspace |
| 5 | CUDA foundation probe | 同 storage 多 lane/dependency、exact capture 与 bounded/general fallback；Q4 MatMul/MLP 的 HYLE/vendor/第三方/unsupported 路由与 residence |
| 6 | Apple foundation probe | MLX delegated region + 原生 Metal custom-kernel/interop，反证 island boundary/Completion，不横向填 Apple 算子 |
| 7 | **Foundation implementation gate** | 上述三类探针、三平台基础 correctness、one-op P0/P1、source/static/shared、红队通过；Compute/Graph 基础 candidate 固定 |
| 8 | NVIDIA/Apple provider 广度 | cuBLASLt/cuDNN/HYLE 与 MLX/MPS/Metal 按真实热路径扩展；不再改基础对象模型 |
| 9 | Vulkan 保底 | Windows/Linux 原生 Vulkan 与 Apple MoltenVK portability；不冒充原生 Metal 性能 |
| 10 | Transformer prefill/decode | 多 region、引擎拥有的 specialization/cache 与 batch-class 选择、并发 Invocation、capture、cancel/drain 和长寿命计划纵切 |
| 11 | 系统性算子广度 | foundation lock 后按四个真实引擎热路径扩展，不用算子数量掩盖架构缺陷 |
| 12 | `compute_sched` convenience | 只有 coroutine/Sched 真实消费者形成 trigger 才做，不是 foundation blocker |
| 13 | 终局 closure/package | 各 factor 独立 CLOSED 或精确 BLOCKED；source/static/shared、依赖锁、许可证、user/architecture/docs 一致 |

原目标中“ggml/oneDNN region”不再视为一个并列动作：oneDNN 是 route 层的首个真实验证者；当前 CPU ggml
保持独立 blocker；CUDA/Metal/Vulkan ggml carrier 各自随对应平台纵切开卡。原先尚未执行的 foundation P1
后移到 route 实现稳定之后，并在第 7 项只运行一次，避免给即将改变的 selector 重复签证。第 7 项以前
所有“基础完成”百分比都必须按这张表计算；算子 family 数量不计入。
