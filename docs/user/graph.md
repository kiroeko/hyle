# HYLE Graph：编译并重复执行计算 DAG

> 本页描述 foundation 迁移中的**当前实现**，不是 foundation 已收口的声明。当前状态只以
> [`module-status.json`](../implementation/module-status.json) 为准；node-set region 已实现，已接受但尚未实现的
> exact + bounded/guarded shape domain、统一完整成本路由与 constant materialize 合同见
> [Compute/Graph 路由审计](../implementation/tasks/compute/compute-graph-route-unification-audit.md)，不得从本页
> 推断它们已经交付。当前已经实现 exact symbolic specialization，execution target / storage domain / execution lane
> 三轴 placement，真实跨 storage copy，partial-order workspace、显式的硬预算序列化取舍，shared Queue 与
> delegated provider island、segment 调度及显式 replay 准备。CUDA target/storage/lane、同 Context 双 stream
> event import、single-queue capture/hybrid replay 已真机资格化；生产级 GPU operation provider 仍是后续纵切。

当前 exact specialization 已支持有限派生维度与 shape guards，Provider replay 分类事实也可查询。
完整 bounded runtime/variants、资源副作用域、variant 内存共享和搜索受限返回可行解仍是
[架构合同](../architecture/graph-design.md)，不是已经交付的运行能力。

当程序要把一段模型无关计算冷编译一次，再用固定 shape specialization 和不同 storage 反复执行时使用
Graph。只执行一个已知 operation 或 native kernel 时直接使用 [Compute](compute.md)。Graph 不是 Sched
task graph，也不拥有模型循环、KV cache、batching、sampling 或 serving policy。

## 四个生命周期对象

```text
GraphBuilder -> immutable Graph
Graph + CompileEnvironment/CompileOptions -> Executable + CompileReport
Executable -> one Invocation per concurrent lane
Invocation.bind(...) -> prepare_replay(...) -> submit(...) -> Completion
```

- `GraphBuilder` 是单 owner、可失败的逻辑构造器。`ValueId`、`NodeId` 和 `DimensionId` 带 graph identity，
  异图 ID 会被拒绝；node 只能读取已经产生的 value，因此数据环不能被构造出来。
- `Graph` 是不可变、可共享且与具体设备无关的逻辑 DAG；它不持有 tensor payload、Context、Queue 或
  provider artifact。
- `Executable` 是不可变、可共享的单一 concrete specialization。它固定 provider artifact、placement、
  physical value、segment dependency 和 workspace layout。
- `Invocation` 是 move-only 的单次并发 lane，独占 bindings、workspace、Completion/capture slots 和 replay
  route。同一个 Executable 的并发请求必须使用不同 Invocation。

## 当前运行时能力

| 能力 | 当前合同 |
|---|---|
| DAG | typed、多输入/输出 node；共享 value、分支、多个 published output 与显式 control dependency |
| shape | concrete `TensorDesc`，或独立 min/max/multiple 维度、有限 sum/product/exact-quotient/broadcast 派生维度及 equal/less-equal guards；同一 `DimensionId` 共享值 |
| specialization | `compile()` 只要求每个独立维度恰好绑定一次，派生值由程序计算；一个 Executable 对应一个包含全部维度值的 collision-safe `SpecializationSignature` |
| execution target | `Context` 是执行者；node placement 只选择 target，不把 allocator 或 Queue 混入 identity |
| storage domain | exact `MemoryResourceRef` 加 workspace hard budget；value placement 与真实 copy 只由 storage 决定 |
| execution lane | shared lane 绑定 target、role 与 caller Queue；delegated provider 为每个 island 产生无需 QueueBinding 的 synthetic lane |
| placement | node/value 可 required 或 preferred；其余按输入 locality、边界数和稳定 target/storage 顺序确定性选择；prepare 尝试有硬预算 |
| region partition | provider 在冷路径提出非空 node-ID 集合，支持单节点、连续和非连续 region；Graph 验证独立及联合收缩无环，并只编译胜出 artifact；当前仍是显式 prefer-region 的结构策略，不是统一完整成本选择或 SOTA 性能证明 |
| boundary | canonical tensor 只有在真实 storage residence 改变时才产生 `Copy`；换 target 或 lane 本身不复制 |
| memory plan | 每个 `(logical value, storage)` 是一个 physical instance；默认只复用 happens-before 已证明有序的 lifetime，显式允许时才为硬预算增加可报告的顺序边 |
| scheduling | 冷路径先做确定性 lane-aware Kahn 调度，再建立 happens-before 和 workspace；同 shared lane 尽量连续录入以减少 Queue segment |
| replay | `prepare_replay()` 冷验证 bindings、transfer、Queue 与 delegated endpoint identity、dependency route 和 Completion 容量 |
| Provider replay 成本 | `CompileOptions::provider_replay` 在冷编译筛选 leaf/region；默认允许所有合法类别，严格模式只接受 prepared |
| capture | `kCommands`、`kPreferCapture`、`kRequireCapture`；capture 只适用于 shared Queue segment，delegated island 不能伪装成 Queue capture |
| report | `CompileReport` 分列 target/storage/lane/segment/runtime owner；`ReplayReport` 给出 exact endpoint 后的 ordered/imported/host-wait 事实 |

Provider-private layout 只能留在一个 region 内，公开 region boundary 当前必须回到 canonical `TensorDesc`。
因此 Graph 不伪造一个没有 layout producer/consumer 合同的通用 `Repack` step；将来某个真实 provider 需要
跨 region opaque layout handoff 时，必须先增加有版本的 boundary-layout 合同和可执行 repack owner。

对于 ggml、MLX 或 TensorRT 这类自己 host 图/engine 的 runtime，未来 adapter 会在 compile 冷路径把一个
HYLE region 翻译成 provider-native artifact；单节点 region 可以像函数一样重复提交，多节点 region 可以
保留融合。Graph 不会因为库已链接就自动发现这些能力，也不会进入第三方黑盒逐 kernel 替换。只有显式
安装且通过当前 backend/storage/completion 审计的 Provider 才能提案；当前尚无可用 ggml region provider。
第三方的 `compile` 不必然意味着每次调用零分配：Provider 分别声明 prepared/managed/unknown 成本，
严格使用者可在 compile 要求 prepared-only；它与 `ReplayMode` 的 command/capture 选择互相独立。

Region 由数据流而非插入顺序定义。例如 gate/up MatMul 与 gated activation/down MatMul 之间插入无关
node，或者先创建 up 再创建 gate，不应阻止 oneDNN 识别同一 MLP。反之，如果 data/control 路径离开该集合
后又回到集合，就不能把它收成一个 execution island；多个各自合法的集合也须通过联合无环检查。
普通用户不需要手工分区。实现 Provider 时可以用 `is_region_contractible()` 检查候选：合法但不能收缩返回
`false`，空集、重复或异图 ID 返回错误；Graph 在接收 proposal 后仍独立验证。连续集合保留快速路径。

leaf/region 的结构提案共用 `compute::ProviderRouteCostFacts`，workspace、persistent、copy/repack、sync 和
dispatch 都允许明确 unknown，不能写零补齐。当前 prefer-region 结构策略先看合法覆盖，只有参与本次
比较的候选都声明 dispatch 数时才比较该维度，否则按稳定 Provider 顺序继续；这不是推理性能排名。
旧的“相对本 Provider fallback 节省若干 ns”入口和 `kRequireEstimatedBenefit` 已移除，不保留别名。
要求跨 Provider 量化择优的 qualified full-plan profile 尚未交付，不能用各家的私有数字冒充；此期间可
显式安装 Provider/关闭 region，但不应声称已得到自动 SOTA 计划。

### 何时要求 prepared-only

若热路径不能接受 Provider 实例化执行 metadata/array 或内部 shape/cache 工作，在冷编译时明确要求：

```cpp
auto compiled = compile(
    graph, context,
    CompileOptions{.provider_replay = ProviderReplayPolicy::kRequirePrepared});
```

这个约束同时作用于 leaf 和 region：managed/unknown 候选不能影响 target/region 选择，也不会创建
artifact；仍可选择其他合格 Provider，没有合格路线则 compile 失败，不在 submit 时放宽。默认
`kAllowAny` 保留已安装 Provider 的合法路线，但不把 unknown 解释成零成本。prepared 也不等于 shared
Queue 或可 capture：已预备的 delegated artifact 同样可以合格。

实现 region Provider 时，在 `RegionProposal::replay_class` 提前声明类别。known 值必须与资源提案和
artifact 一致；unknown 只能在 allow-any 下细化，不能先以 unknown 进入严格计划再补声明。此策略只约束
Provider 成功 replay，不替调用方 executor/callback、View/Copy、失败诊断或 backend Completion 的独立
合同证明零开销，也不把声明本身当成性能资格化证据。

## 单 target/storage/lane：自然短路径

固定 shape、单 Context 的调用不需要显式写 environment 或 replay options：

```cpp
#include <array>
#include <hyle/graph/executable.h>

using namespace hyle::compute;
using namespace hyle::graph;

auto created = GraphBuilder::create();
if (!created) return created.error();
GraphBuilder builder = std::move(*created);

auto x = builder.input(x_desc);
auto weight = builder.constant(weight_desc);
if (!x || !weight) return hyle::Status::InvalidArgument("graph inputs");

auto normalized = builder.add(RmsNorm{}, *x);
if (!normalized) return normalized.error();
auto output = builder.add(MatMul{}, *normalized, *weight);
if (!output) return output.error();
if (auto status = builder.publish(*output); !status.ok()) return status;

auto graph = std::move(builder).finish();
if (!graph) return graph.error();
const std::array constants{ConstantBinding{*weight, weight_view}};
auto compiled = compile(*graph, context,
                        CompileOptions{.constants = constants});
if (!compiled) return compiled.error();

auto invocation = compiled->executable.create_invocation();
if (!invocation) return invocation.error();
if (auto status = invocation->bind(*x, input_view); !status.ok()) return status;
if (auto status = invocation->bind(*output, output_view); !status.ok())
  return status;

// 单 active shared lane 会自动走同一套 cold prepare；后续相同 Queue 可直接复用。
auto done = invocation->submit(queue);
if (!done) return done.error();
return done->wait();
```

`constant()` 只声明逻辑 value；`CompileOptions::constants` 提供 payload。当前 Executable 保守借用所有
constant source，即使 provider 另有 packed representation，源 storage 仍须活过 Executable 及其全部提交。

## 独立与派生维度

例如消费者按 batch × tokens 提供一块连续的 activation，用派生行数描述它，不要再创建第三个独立
symbol 然后要求调用者重复维护乘积：

```cpp
auto batch = builder.dimension({.minimum = 1, .maximum = 8});
if (!batch) return batch.error();
auto tokens = builder.dimension({.minimum = 1, .maximum = 4096});
if (!tokens) return tokens.error();
auto rows = builder.dimension_product(Dimension(*batch), Dimension(*tokens));
if (!rows) return rows.error();
if (auto s = builder.require_dimension_less_equal(
        Dimension(*rows), Dimension::fixed(4096)); !s.ok()) return s;
auto spec = TensorSpec::dense(DataType::kFloat32,
    std::array{Dimension(*rows), Dimension::fixed(128)});
if (!spec) return spec.error();
auto input = builder.input(*spec);
if (!input) return input.error();
auto output = builder.add(RmsNorm{}, *spec, *input);
if (!output) return output.error();
if (auto s = builder.publish(*output); !s.ok()) return s;
auto graph = std::move(builder).finish();
if (!graph) return graph.error();

const std::array shape{ShapeBinding{*batch, 2}, ShapeBinding{*tokens, 128}};
auto compiled = compile(*graph, context, CompileOptions{.shape = shape});
// 只绑定 batch/tokens；实际输入、输出 TensorView 是 [256, 128]。
```

- `dimension_sum` 适合 concat 输出轴；`dimension_product` 适合展平后的 extent；
  `dimension_quotient` 要求除数非零且整除，不向下取整；`dimension_broadcast` 要求两值相等或其中一个为 1。
  零尺寸合法，广播 `(0, 1)` 得到 0，但具体算子仍可能不支持空输入。
- 操作数只允许非负常量或同 Builder 已存在的 ID；表达式天然无环，不读取 payload。无递归表达式解释器、
  用户 callback 或通用符号求解器。固定矛盾在构造/finish 时拒绝，其余约束在 compile 的具体求值阶段检查。
- `dimension_bounds()` 给的是保守范围；guards 可以排除范围中的组合，各轴上界未必同时合法。过宽范围
  导致整数溢出时必须缩窄声明，不能依赖 guard 自动收紧。派生 bounds 由系统计算，不接受独立覆盖。
- `dimension_count()` 枚举全部 ID；`independent_dimension_count()` 和 `dimension_is_independent(id)` 用于
  识别调用者需绑定的部分。重复、缺失、外来或派生 ID 的 `ShapeBinding` 都会失败。signature 按 ID 顺序保存
  **全部**已解析值，不能把其 `binding_count()` 当成调用者的输入维度数。

这仍是冷编译生成一个 exact Executable：更换具体 shape 需要重新 compile，或由消费方缓存多个
Executable。它还不是“一个 Executable 跑整个 bounded domain”。当前 `View::reshape/broadcast_to` 的
operation attributes 仍是具体 `TensorShape`；派生 TensorSpec 不会偷偷改写这些固定参数，也不自动生成
reshape/copy。完整动态属性与 runtime variants 尚待实现。

## Symbol 与多 target/storage/lane

symbolic graph 在 compile 时产生 concrete specialization；Graph 本身不维护无界 cache。推理引擎可用
`SpecializationSignature` 管理 decode/prefill 等有限 Executable 集合：

```cpp
auto batch = builder.dimension({.minimum = 1, .maximum = 64,
                                .multiple_of = 1});
const std::array logical_shape{Dimension(*batch), Dimension::fixed(hidden)};
auto spec = TensorSpec::dense(DataType::kFloat16, logical_shape);
auto input = builder.input(*spec);
// ... add operations and finish the Graph ...

const std::array storage{
    StorageDomain{.resource = host_resource},
    StorageDomain{.resource = device_resource,
                  .workspace_budget_bytes = device_budget},
};
const std::array targets{
    ExecutionTarget{.context = &cpu_context,
                    .default_storage = StorageDomainId{0}},
    ExecutionTarget{.context = &cuda_context,
                    .default_storage = StorageDomainId{1}},
};
const std::array lanes{
    ExecutionLane{.target = ExecutionTargetId{0},
                  .role = ExecutionLaneRole::kCompute},
    ExecutionLane{.target = ExecutionTargetId{1},
                  .role = ExecutionLaneRole::kCompute},
    ExecutionLane{.target = ExecutionTargetId{1},
                  .role = ExecutionLaneRole::kTransfer},
};
const std::array shape{ShapeBinding{*batch, 1}};
const std::array resources{
    ExternalResourceContract{.value = *input,
                             .storage = StorageDomainId{1}},
};
auto compiled = compile(graph,
                        CompileEnvironment{.storage_domains = storage,
                                           .targets = targets,
                                           .lanes = lanes},
                        CompileOptions{.shape = shape,
                                       .external_resources = resources});
```

`NodePlacement` 选择 target，`ValuePlacement` 与 `ExternalResourceContract` 选择 storage；前两者可以表达
required 或 preferred。required route 不可执行时 compile
失败；preferred route 不可执行时按稳定候选顺序继续。只有 `Unimplemented` 表示候选不支持，OOM、损坏
状态或其它错误不会被吞掉后静默 fallback。`max_placement_attempts` 限制实际冷准备次数；当前实现若
在形成完整可行计划之前耗尽这个上限，仍然失败，不能返回部分图。

`max_region_proposals`、`max_region_proposals_per_provider` 和 `max_region_proposal_nodes` 限制候选发现，
最后一项限制复制保存的 node-ID 总数。默认 `search_policy = CompileSearchPolicy::kAllowLimited`：达到
容量时停止扩大相应发现，只用已接收的候选生成并验证完整计划；per-provider 上限不阻止还有总额时
探索下一家。成功报告的 `search().optimization_limited()` 明确说明搜索是否被截断，可再查看三个
`*_limit_reached` 字段、实际接收数和 prepare 次数。这不是全局最优或性能资格声明。

需要拒绝受限发现时使用 `CompileSearchPolicy::kRequireUnexhausted`；它不是要求无限搜索。恰好用满
容量但没有任何候选/provider 被省略，不标记受限。两种政策都不放宽 coverage、required placement、
workspace 等硬约束，也不吞真实 Provider/OOM 错误。

Provider 的 `RegionProposalWriter::try_add()` 返回 `Result<bool>`：true 表示已复制并排序，false 表示
容量不够，应结束当前发现并返回 `Status::Ok()`；unexpected 必须原样传播。没有部分接收一个 node set，
Provider 可以使用临时栈数组，不必把它们保活到 compile 结束；不能把自身 OOM 转换成 false。

非默认 storage 的 external value 应提供 `ExternalResourceContract`，让 compiler 能选择真实 transfer route；不能让
Graph 从偶然地址猜 host/device locality。普通 canonical boundary 使用 Compute byte transfer，不隐藏
pageable staging、repack 或同步 copy。

每个 `StorageDomain::workspace_budget_bytes` 都是硬上限。默认
`WorkspaceReusePolicy::kExistingOrderOnly` 保留原有并发；若现有 partial order 下的峰值超过预算，compile
返回 `ResourceExhausted`。只有调用者显式选择 `kAllowSerialization` 时，compiler 才会在有界、精确的
happens-before 分析内给独立 lane 加顺序边并重新规划 workspace：

```cpp
auto compiled = compile(
    graph, environment,
    CompileOptions{.workspace_reuse =
                       WorkspaceReusePolicy::kAllowSerialization});
if (!compiled) return compiled.error();

// 非零表示为了预算牺牲了多少条原有并发关系；实际 peak 仍看
// workspace_bytes() 和各 storage report。
const auto added_order = compiled->report.serialization_edge_count();
```

该选项不会保证任何预算都可满足：预算小于单条不可拆 lifetime、分析规模超过内部硬界或不存在无环排序时仍
失败。预算本来足够时不会加边，`serialization_edge_count()` 为零。

## 冷态 replay 路由

多 active shared lane 必须显式准备 Queue 集合；synthetic delegated lane 不接受 QueueBinding：

```cpp
const std::array queues{
    QueueBinding{.lane = ExecutionLaneId{0}, .queue = &cpu_queue},
    QueueBinding{.lane = ExecutionLaneId{1}, .queue = &cuda_compute_queue},
    QueueBinding{.lane = ExecutionLaneId{2}, .queue = &cuda_copy_queue},
};

auto status = invocation.prepare_replay(
    queues, ReplayOptions{.mode = ReplayMode::kPreferCapture,
                          .host_wait = HostWaitPolicy::kReject});
if (!status.ok()) return status;

auto replay = invocation.replay_report();
if (!replay) return replay.error();
auto done = invocation.submit(queues);
if (!done) return done.error();
return done->wait();
```

`prepare_replay()` 是冷态 commit point。在任何 segment 提交前，它会：

1. 验证全部 external bindings、exact resource 与 Queue/target identity；
2. 对每个真实 copy 查询 immediate/queued-direct/peer capability；
3. 取得 shared Queue 和 delegated provider 的 exact completion endpoint，把每条 dependency 固定成
   ordered endpoint、同步 source、非阻塞 imported wait 或 host wait；
4. 按物理 Queue 汇总并预留 Completion slots；
5. 按 replay mode 建立 command、capture 或 hybrid route。

默认 `HostWaitPolicy::kAllow` 允许不兼容 backend 的显式 host fallback；延迟敏感部署应选择 `kReject`，在
进入 warm path 前拒绝这种 topology。host wait 不会藏在 report 里，也不会在持有 destination Queue/provider
submission admission 时等待上游。`ReplayReport` 提供四类 dependency 数和三类 transfer 数；
`dependency_route(segment, index)` 可查询精确边。

`kPreferCapture` 在不支持、不安全或 delegated 的 segment 上保留普通 route，最终可报告 `kCommands`、
`kCaptured` 或 `kHybrid`；`kRequireCapture` 遇到任一 delegated 或不能 capture 的 command segment 就失败。
当前 RTX 4070 纵切已验证：同一 CUDA target 的 compute/transfer lane 共用一个 device storage identity，只有
实际产生 segment 的 lane 要求 Queue；兼容 CUDA stream 可用 event 非阻塞导入前缀，单 Queue 的兼容 copy
segment 可 cold capture 并 warm replay。它不代表尚未实现的 CUDA operation provider，也不承诺一般
multi-stream capture。

## 冷路径、热路径与调度取舍

- compile 做 shape concrete 化、provider proposal/exact prepare、region 选择、target/storage placement、
  boundary materialization、lane scheduling、happens-before、liveness/alias/workspace；复杂度有显式预算，不读取历史 benchmark 常数作为 placement
  oracle。`kAllowSerialization` 的额外搜索只在冷编译且实际超过 storage hard budget 时发生；warm path
  只消费已经固化的 segment dependency。
- `create_invocation()` 实例化 workspace、provider invocation、预定长 metadata，并固定 delegated endpoint；应按服务并发槽创建并复用。
- `prepare_replay()` 做 Queue/endpoint/transfer/dependency/capture 资格化；bind、reset 或稳定地址变化会使其失效。
- warm `submit()` 只遍历 emitted steps、segment dependencies 与 bindings：不遍历原始 DAG、不推 shape、
  不搜索 provider、不 JIT、不增长 Graph heap、不重新查询 transfer/dependency capability。backend
  finalization/event 数按 segment 增长，不按 node 增长。
- 调度器优先减少 host 录入与 event 边界，不声称求解一般 DAG 的全局最优设备并行调度。需要跨请求公平、
  CPU worker 或 cancellation policy 时，在消费层组合 Sched；Graph 自身不启动线程池。

## Report 怎么读

`CompileReport` 是可复制的冷态事实副本，不保活 heavyweight artifact。它可枚举：

- `search()` 的已接收候选/node 数、prepare 次数和是否因容量省略了候选/provider；
- specialization、target/device/backend、storage workspace/budget，以及每 lane runtime mode/role/Queue/capture 需求；
- physical value、external/constant/workspace/alias ownership；
- operation fallback、provider region 和真实 copy step；
- shared/delegated segment、dependency、为 workspace budget 新增的 serialization edge、potential
  synchronization 和 concurrent-invocation 能力；
- provider/version、实际配置实例 `provider_instance`、dispatch、workspace、persistent/packed/borrowed
  bytes、blocking/threading 与 replay 类别。

`persistent_bytes == nullopt` 表示 provider 无法报告，不是零。`delegated_segment_count()` 是 provider-owned
island 数，不是 hidden Queue 数。CompileReport 中的 `kQueueBindingDependent` 也不是已经发生 host wait；
具体 Queue/provider endpoint 固定后的真实路线只看 `ReplayReport`。

Provider operation/region 的 `step(...)->runtime.replay_class` 保留 Compute 声明的 prepared/managed/unknown
事实（选型见 [Compute](compute.md)）。它不从 region 大小、shared/delegated 或 capture 反推；unknown
不能视为 prepared。compile 验证 proposal/artifact 一致性，并按显式 prepared-only 政策筛选候选。
metadata-only view 可以由 Provider 在冷路径准备别名描述，但不因此录入 kernel；Graph copy 由 backend
负责，不用 Provider replay 字段替代各自的执行/后端成本合同。

`step(...)->provider_instance` 区分同一 Provider 家族的多个配置槽位：operation 来自 PreparedOperation，
region 来自实际绑定的 Provider extension，Graph copy 没有 Provider 实例。由 Provider 准备的 metadata
view 保留实例 ID，但这不表示发生了 dispatch。ID 本身不保活 Context/artifact；报告可留作诊断，不能用
旧 ID 授权提交或当成跨进程、跨构建的完整性能 key。详见 [Compute 的身份与成本说明](compute.md)。

`CompileStepReport::node_count` 是该 step 覆盖的逻辑节点数量；`node_begin` 是最小 ordinal，不总是一个
连续 range 的起点。只有 `nodes_contiguous` 为真时才能按 begin/count 展开；通用代码使用
`CompileReport::step_node(step_index, node_index)` 查询精确节点集合。真实 Copy step 的 node count 为零。

## 所有权与常见陷阱

- `ExecutionTarget` 的 Context 会被 plan 内部 artifact 保活；`StorageDomain` 中 `MemoryResourceRef` 的 provider 仍是 caller
  borrow，必须活过 Executable、Invocation 和全部 in-flight work。
- Tensor view 不拥有 payload。输入、输出、shared Queue、external resource 与 native kernel 参数都必须活到
  Completion 真终态；delegated provider invocation 由 Graph Completion 保活，丢弃 pending Completion 不会隐式等待。
- 普通 materialized published output 绑定 mutable `TensorView`。source alias 或 Invocation-owned output
  不接受重新绑定；成功完成后用 `Invocation::output()` 取得借用 view。
- `reset()` 保留已规划 workspace，但清 external bindings；in-flight 时失败。bind/reset 后 replay route
  失效，多 shared lane 必须重新 `prepare_replay()`。
- 不同 published outputs，以及 output 与仍存活 input，不得绕过 plan 形成危险 overlap。只在相同
  storage/resource、最后一次读取且 provider 明确允许时选择 in-place；其它 partial overlap 被拒绝。
- 外部副作用 node 必须用数据依赖或 `depends_on()` 排序。纯 node 不要为了“看起来顺序一致”手工串行化。
- Graph 可以保存合法 blocked/opaque TensorDesc，但没有理解该 encoding 的 provider 时 compile 会返回
  `Unimplemented`，不会把压缩 bytes 当普通 scalar storage。

source、headers + static、headers + shared 都选择 `hyle::graph`。三种交付是同一 C++23 surface，不是
Native FFI，也不承诺跨编译器/STL 的 C++ ABI。
