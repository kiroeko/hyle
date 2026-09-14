# HYLE Graph：编译并重复执行计算 DAG

> 本页描述当前 Graph v7 **实现基线**，不是 foundation 已收口的声明。当前状态只以
> [`module-status.json`](../implementation/module-status.json) 为准；已接受但尚未实现的 node-set region、
> exact + bounded/guarded shape domain 与 constant materialize 合同见
> [Compute/Graph 路由审计](../implementation/tasks/compute/compute-graph-route-unification-audit.md)，不得从本页
> 推断它们已经交付。当前已经实现 exact symbolic specialization，execution target / storage domain / execution lane
> 三轴 placement，真实跨 storage copy，partial-order workspace、显式的硬预算序列化取舍，shared Queue 与
> delegated provider island、segment 调度及显式 replay 准备。CUDA target/storage/lane、同 Context 双 stream
> event import、single-queue capture/hybrid replay 已真机资格化；生产级 GPU operation provider 仍是后续纵切。

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
| shape | concrete `TensorDesc`，或带 min/max/multiple 约束的 graph-local symbol；同一 `DimensionId` 表示相等约束 |
| specialization | `compile()` 要求每个 symbol 恰好绑定一次；一个 Executable 对应一个 collision-safe `SpecializationSignature` |
| execution target | `Context` 是执行者；node placement 只选择 target，不把 allocator 或 Queue 混入 identity |
| storage domain | exact `MemoryResourceRef` 加 workspace hard budget；value placement 与真实 copy 只由 storage 决定 |
| execution lane | shared lane 绑定 target、role 与 caller Queue；delegated provider 为每个 island 产生无需 QueueBinding 的 synthetic lane |
| placement | node/value 可 required 或 preferred；其余按输入 locality、边界数和稳定 target/storage 顺序确定性选择；prepare 尝试有硬预算 |
| region partition | provider 在冷路径提出连续 region；当前实现按覆盖、dispatch 数和 provider 顺序确定性选择并只编译胜出 artifact；这是现行结构策略，不是跨 provider 的 SOTA 性能证明 |
| boundary | canonical tensor 只有在真实 storage residence 改变时才产生 `Copy`；换 target 或 lane 本身不复制 |
| memory plan | 每个 `(logical value, storage)` 是一个 physical instance；默认只复用 happens-before 已证明有序的 lifetime，显式允许时才为硬预算增加可报告的顺序边 |
| scheduling | 冷路径先做确定性 lane-aware Kahn 调度，再建立 happens-before 和 workspace；同 shared lane 尽量连续录入以减少 Queue segment |
| replay | `prepare_replay()` 冷验证 bindings、transfer、Queue 与 delegated endpoint identity、dependency route 和 Completion 容量 |
| capture | `kCommands`、`kPreferCapture`、`kRequireCapture`；capture 只适用于 shared Queue segment，delegated island 不能伪装成 Queue capture |
| report | `CompileReport` 分列 target/storage/lane/segment/runtime owner；`ReplayReport` 给出 exact endpoint 后的 ordered/imported/host-wait 事实 |

Provider-private layout 只能留在一个 region 内，公开 region boundary 当前必须回到 canonical `TensorDesc`。
因此 Graph 不伪造一个没有 layout producer/consumer 合同的通用 `Repack` step；将来某个真实 provider 需要
跨 region opaque layout handoff 时，必须先增加有版本的 boundary-layout 合同和可执行 repack owner。

对于 ggml、MLX 或 TensorRT 这类自己 host 图/engine 的 runtime，未来 adapter 会在 compile 冷路径把一个
HYLE region 翻译成 provider-native artifact；单节点 region 可以像函数一样重复提交，多节点 region 可以
保留融合。Graph 不会因为库已链接就自动发现这些能力，也不会进入第三方黑盒逐 kernel 替换。只有显式
安装且通过当前 backend/storage/completion 审计的 Provider 才能提案；当前尚无可用 ggml region provider。

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
状态或其它错误不会被吞掉后静默 fallback。`max_placement_attempts`、`max_region_proposals` 和
`max_region_proposals_per_provider` 都是硬预算，耗尽会失败而不是改变计划。

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

- specialization、target/device/backend、storage workspace/budget，以及每 lane runtime mode/role/Queue/capture 需求；
- physical value、external/constant/workspace/alias ownership；
- operation fallback、provider region 和真实 copy step；
- shared/delegated segment、dependency、为 workspace budget 新增的 serialization edge、potential
  synchronization 和 concurrent-invocation 能力；
- provider/version、dispatch、workspace、persistent/packed/borrowed bytes 与 blocking/threading fact。

`persistent_bytes == nullopt` 表示 provider 无法报告，不是零。`delegated_segment_count()` 是 provider-owned
island 数，不是 hidden Queue 数。CompileReport 中的 `kQueueBindingDependent` 也不是已经发生 host wait；
具体 Queue/provider endpoint 固定后的真实路线只看 `ReplayReport`。

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
