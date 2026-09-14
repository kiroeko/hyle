# Compute/Graph：第三方广度与 HYLE 全局路由统一审计

> 本页记录 2026-09-14 的架构需求协商、源码审计与有限解决方案。它目前不是 closure card，也不改变
> `module-status.json`；产品修改前仍须让受影响 component 依法返回 `AUDITED` 并把有限改动集写入其当前
> record。

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

## 当前源码勘验

baseline：`e23b94addeec04b403f63e6553936570831a287b`。

| 需要的机制 | 当前事实 | 裁决 |
|---|---|---|
| provider-neutral tensor/op | `TensorDesc` 已分离 logical dtype 与 blocked/opaque physical format；`Operation` 是 typed schema | `KEEP` |
| 冷 prepare 与热 artifact | Provider 先 `propose`，再对 exact resources `prepare`；`PreparedOperation`/Invocation 分离 | `KEEP` |
| shared/delegated runtime | 单 op 与 region 均声明 Queue ownership、workspace、threading、Completion endpoint | `KEEP` |
| 第三方 compiled region | versioned `RegionProviderVTable` 已能 propose/compile 连续 region，Executable 持有 artifact | `KEEP` |
| 全局 target/storage/lane/liveness | Graph 已显式规划三轴、copy、happens-before、workspace 与 report | `KEEP` |
| 单 op 与 region 同池竞争 | target placement 先因 `region_support` 布尔值偏向能提 region 的 target，operation fallback 再按 Context provider 顺序准备，region 随后独立做 interval selection | `FIX` |
| 完整成本选择 | selector 只使用 provider 自报 savings、覆盖数、dispatch 和 provider rank；已有 copy/sync/workspace/persistent/host-thread facts 主要只进 report，选中后得到的 exact artifact facts 不触发重选 | `FIX` |
| 可比成本来源 | `estimated_net_savings_ns` 相对各 Provider 自己的 fallback，不能跨 Provider 相加；没有资格化 profile identity | `FIX` |
| 单节点 graph-hosting Provider | region discovery 拒绝 `nodes.count < 2`；只能另走 delegated operation seam，不能统一比较 | `FIX` |
| tune-once | 没有对少量完整候选计划的显式、可禁用 tune 与 identity-bound cache | `FIX` |
| route 诊断 | CompileReport 描述胜出 step/fallback，但没有有界 shortlist、拒绝原因和所用 evidence identity | `FIX` |
| typed custom op 三交付身份 | `OperationId` 被保存但 `custom_as<T>()` 比较模板静态 vtable 地址；default-hidden shared factor 间地址不构成合同 | `FIX` |
| ggml backend 身份 | 当前无后缀 identity 实际只记录 CPU blocker；不能代表 CUDA/Metal/Vulkan | `FIX`（新 factor 时） |
| CUDA Q4 可执行能力 | CUDA backend runtime 已存在，但没有接受 Q4 `StorageFormatId` 的 CUDA operation/region Provider | `MISSING`，不是 runtime 缺陷 |

这说明现有 Compute/Graph 的生命周期、storage、异步执行和 plan 骨架没有被反证；被反证的是**冷路径
route selection 层**和 custom-operation 的 shared 交付身份。继续填算子会把错误的选择机制固化，整套推倒
重写同样没有依据。

## 有限解决方案

### R1：修正 typed custom operation identity

`compute_ops` 以 `OperationId`（其中包含 schema/version identity）和 descriptor size 校验类型，不再以
inline vtable 地址判断。traits/infer 仍是冷路径函数指针，descriptor 仍保持有界、trivially-copyable、
heap-free。增加 source/static/shared 跨边界构造与 Provider 解码测试；不建立全局 registry 或 Native FFI。

### R2：把单 op 与 region 归一为冷路径 `RouteCandidate`

Graph 内部候选至少包含 node range、Provider/version、target、resource/layout requirements、runtime mode、
workspace/persistent/packing、boundary copy/repack、sync/dispatch、host blocking/private threads、precision/
alias/failure/completion 与证据 identity。普通 Provider proposal 产生长度一候选；region extension 产生长度
一或更长候选。只有选中后才构造重型 artifact，避免为所有候选 JIT/packing。

冷编译按有界闭环执行：先枚举 candidate skeleton 与资源要求；再为有限的 target/storage 组合计算 Graph
拥有的 boundary copy/repack/sync；随后选择候选计划并只 prepare 胜出 artifact。Provider proposal 对选择
所需结构成本给出 exact、保守上界或 unknown，artifact 必须验证该承诺。exact prepare 返回
`Unimplemented`、事实超过已接纳上界或破坏预算时，将该候选排除并在预声明 retry budget 内重新选择；
不能把不一致成本留到 report 后仍执行原计划，也不能预先 JIT 所有候选来换取一个“完美”评分。

Direct Compute 继续使用显式 Context provider 顺序与 `Unimplemented` fallback，因为它没有全图边界信息；
Provider 应只在自己愿意按该顺序拥有的精确域内提案。Graph 则必须把这些 leaf proposal 与 region proposal
放入同一有界计划搜索。无需为此增加第三套公开 runtime 或 `Plan` 产品。

### R3：结构成本、资格化 profile 与 tune-once 分层

1. 先按语义、精度、storage/import、host-wait、并发和内存预算排除不可行路线；
2. provider 只报告可审计结构事实，不提供可跨 Provider 比较的任意相对 ns；
3. 可比较执行估值来自绑定 provider/build、device/driver、dtype/format/layout、shape domain 与完成边界的
   evidence profile；差异不可分或不足 material margin 时选择已配置的成熟路线；
4. 长寿命推理 specialization 可显式启用 tune-once，只在结构筛选后的少量**完整候选计划**上同合同测量。
   缓存绑定完整 identity，任何相关身份变化都失效；默认确定性模式可以完全禁用 tuning；
5. 一般 DAG 不宣称求全局最优。搜索有确定的 candidate/placement/tuning 预算，预算耗尽明确失败或采用
   预声明的成熟 fallback，不在 warm path 继续试探；CompileReport 只保留有界 shortlist 的身份、拒绝原因、
   成本来源和最终裁决，不复制无限 proposal 日志。

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

## 是否需要再次重构

需要，但只做**定向重构**：

- `compute_ops`：从 `CANDIDATE` 返回 `AUDITED`，修复 custom descriptor 的跨 DSO identity；
- `graph`：从 `CANDIDATE` 返回 `AUDITED`，统一 leaf/region route candidate、成本来源和可选 tune-once；
- `compute` runtime、CPU/CUDA backend、Queue/Completion、Tensor storage、Graph 生命周期/liveness/workspace
  骨架没有重写依据；若 R2 的实现最终必须改变 Compute 的 factor 间 proposal seam，再以精确依赖事实让
  `compute` 返回 `AUDITED`，不能预先扩大范围；
- backend-qualified ggml CUDA/Metal/Vulkan factor 在各自 closure card 成立后从 `QUEUED` 开始，不恢复旧
  adapter，也不让当前 CPU blocker 冒充它们的状态。

在上述状态迁移和有限卡片完成前，不继续横向填充算子，也不把现有 `RegionSelectionPolicy` 的最大覆盖
结果称为自动 SOTA 路由。

## 当前 Compute + Graph goal 的重排

当前总目标不变，但执行依赖改为下列顺序；前一步未给出合法 candidate 时不提前运行后一步的重型 P1/P2：

| 顺序 | 工作包 | 当前裁决 / 完成条件 |
|---:|---|---|
| 0 | 本次需求、架构与源码差距基线 | 本页与 Compute/Graph architecture、current user limitation 一致；只跑 docs/closure contract |
| 1 | 生命周期状态迁移 | 为 `compute_ops` 与 `graph` 各建有限 closure card，保存 previous registry，依法 `CANDIDATE -> AUDITED` |
| 2 | R1 custom-operation identity | 数值 schema + size 跨 source/static/shared；correctness、逐头、红队通过 |
| 3 | R2/R3 Graph route unification | leaf/region 同池、placement/boundary 完整成本、有界重选、qualified profile、显式 tune-once、可解释 report |
| 4 | Foundation 候选回放 | 在新身份上回放 Compute/Native/Graph 正确性、one-op tax、三交付和终局红队；不沿用旧候选性能签 |
| 5 | oneDNN direct + region | 作为首个真实成熟 CPU Provider 关闭现有 `AUDITED` factors，并反证 route selector；CPU ggml blocker 不阻塞 |
| 6 | CUDA Q4 MatMul/MLP | backend-qualified HYLE/vendor/ggml 候选按可用公开 seam 分卡；完成规范例的 correctness、route 与 P1/P2 |
| 7 | Apple 主栈 | MLX delegated region + 原生 Metal kernel/interop；MPSGraph/MPS 按真实收益审计 |
| 8 | Vulkan 保底 | Windows/Linux 原生 Vulkan 与 Apple MoltenVK portability；不冒充原生 Metal 性能 |
| 9 | Transformer prefill/decode | 多 region、动态 specialization、并发 Invocation、capture、cancel/drain 和长寿命计划纵切 |
| 10 | `compute_sched` convenience | 只在 coroutine/Sched 真实消费者需要时完成，不反向污染 Compute core |
| 11 | 系统性算子广度 | runtime/route/三条设备纵切稳定后再按四个真实引擎热路径扩展，不用算子数量掩盖架构缺陷 |
| 12 | 终局 closure/package | 各 factor 独立 CLOSED 或精确 BLOCKED；source/static/shared、依赖锁、许可证、user/architecture/docs 一致 |

原目标中“ggml/oneDNN region”不再视为一个并列动作：oneDNN 是 route 层的首个真实验证者；当前 CPU ggml
保持独立 blocker；CUDA/Metal/Vulkan ggml carrier 各自随对应平台纵切开卡。原先尚未执行的 foundation P1
也后移到 R1--R3 之后，避免给即将改变的 selector 重复签证。
