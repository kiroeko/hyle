# HYLE Graph：编译并重复执行计算 DAG

> 本页描述 foundation 迁移中的**当前实现**，不是 foundation 已收口的声明。当前状态只以
> [`module-status.json`](../implementation/module-status.json) 为准；node-set region 已实现，已接受但尚未实现的
> exact + bounded/guarded shape domain、storage/lane 联合优化与跨 variant 常量共享/聚合预算合同见
> [Compute/Graph 路由审计](../implementation/tasks/compute/compute-graph-route-unification-audit.md)，不得从本页
> 推断它们已经交付。当前已经实现 exact symbolic specialization，execution target / storage domain / execution lane
> 三轴 placement，真实跨 storage copy，partial-order workspace、显式的硬预算序列化取舍，shared Queue 与
> delegated provider island、segment 调度及显式 replay 准备。CUDA target/storage/lane、同 Context 双 stream
> event import、single-queue capture/hybrid replay 已真机资格化；生产级 GPU operation provider 仍是后续纵切。

当前 exact specialization 已支持有限派生维度与 shape guards、具名资源副作用域，以及有限 region 发现/
重选、完整候选 profile 成本匹配与受限优化，以及显式常量物化后的 source release；Provider replay 分类事实也可查询。完整 bounded runtime/variants、variant 内存共享与 storage/lane 联合优化仍是
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

- `GraphBuilder` 是单 owner、可失败的逻辑构造器。`ValueId`、`NodeId`、`DimensionId` 和 `EffectResourceId` 带 graph identity，
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
| 外部资源 | graph-local `EffectResourceId`、按 schema 端口绑定、read/write 冲突验证与固定 backing footprint；独立域不加边，未知副作用保守排序 |
| shape | concrete `TensorDesc`，或独立 min/max/multiple 维度、有限 sum/product/exact-quotient/broadcast 派生维度及 equal/less-equal guards；同一 `DimensionId` 共享值 |
| specialization | `compile()` 只要求每个独立维度恰好绑定一次，派生值由程序计算；一个 Executable 对应一个包含全部维度值的 collision-safe `SpecializationSignature` |
| execution target | `Context` 是执行者；node placement 只选择 target，不把 allocator 或 Queue 混入 identity |
| storage domain | exact `MemoryResourceRef` 加 workspace hard budget；value placement 与真实 copy 只由 storage 决定 |
| execution lane | shared lane 绑定 target、role 与 caller Queue；delegated provider 为每个 island 产生无需 QueueBinding 的 synthetic lane |
| placement | node/value 可 required 或 preferred；其余按输入 locality、边界数和稳定 target/storage 顺序确定性选择；prepare 尝试有硬预算 |
| region partition | provider 在冷路径提出非空 node-ID 集合，支持单节点、连续和非连续 region；与 leaf 共用有界冷搜索，完整覆盖且联合收缩无环后才准备 artifact；无 profile 时按配置顺序，不是完整成本择优或 SOTA 证明 |
| boundary | canonical tensor 只有在真实 storage residence 改变时才产生 `Copy`；换 target 或 lane 本身不复制 |
| memory plan | 每个 `(logical value, storage)` 是一个 physical instance；默认只复用 happens-before 已证明有序的 lifetime，显式允许时才为硬预算增加可报告的顺序边 |
| scheduling | 冷路径先做确定性 lane-aware Kahn 调度，再建立 happens-before 和 workspace；同 shared lane 尽量连续录入以减少 Queue segment |
| replay | `prepare_replay()` 冷验证 bindings、transfer、Queue 与 delegated endpoint identity、dependency route 和 Completion 容量 |
| Provider replay 成本 | `CompileOptions::provider_replay` 在冷编译筛选 leaf/region；默认允许所有合法类别，严格模式只接受 prepared |
| Provider 准入 | 全局 `allowed_providers/denied_providers` 和逐节点 `node_providers` 硬约束同时筛选 leaf/region；可指定家族或精确 Context 配置槽位，不因融合或回退而放宽 |
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
dispatch 都允许明确 unknown，不能写零补齐。默认 `kAllowProviderRegions` 让 region 与 leaf 同池选择；
无 qualified profile 时，从最小未覆盖 node 开始，先看 preferred target，再按配置 target/Provider 顺序，
最后才以覆盖数与 dispatch 打破等价次序。只有本次候选都声明 dispatch 时才比较该维度；否则按稳定
候选顺序。它不会因为另一 Provider 的 region 更大或 dispatch 更少就绕过前排可用 leaf，也不是推理性能排名。
旧的“相对本 Provider fallback 节省若干 ns”入口和 `kRequireEstimatedBenefit` 已移除，不保留别名。
要求跨 Provider 量化择优的 qualified full-plan profile 尚未交付，不能用各家的私有数字冒充；此期间可
显式安装/筛选 Provider、约束具名节点或关闭 region，但不应声称已得到自动 SOTA 计划。

### 指定某个节点的实现，其它节点仍可外包

例如用户已经配置了一个自定义实现和成熟 Provider，希望某个 MatMul 节点只能由自定义实现准备，
其它节点仍参与正常选择。约束与 target/storage placement 独立，不用拆出第二张图：

```cpp
// context、graph、matmul_node 已创建；slot 是调用方配置的 Provider 槽位。
auto family = context.provider_identity(slot).value();
auto instance = context.provider_instance_id(slot).value();
const std::array required{
    NodeProviderConstraint{matmul_node, family.id, instance}};
auto compiled = compile(graph, context,
                        CompileOptions{.node_providers = required});
```

省略 `instance` 时，同一 `ProviderId` 家族的任何已配置实例均可；提供它则只允许该 Context 的那一个
槽位。重新创建 Context 会得到不同实例 ID，不要持久化 ID 或把它当作可跨进程的版本/性能身份。
`NodeProviderConstraint` 是必需条件，不是软偏好；不相容的 region 不会参与 placement 或创建 artifact。
若没有完整可执行路线，compile 返回错误，不能为凑齐计划换用其它实现。安全资源拒绝后的重选同样守约。
它约束的是准备逻辑 node/region 的 Provider；metadata-only View 的准备不因此变成 kernel，Graph 插入的
跨存储 copy/sync 仍由 backend 完成，不受 Provider allow/deny 冒充的“设备禁用”控制。

全局筛选用 `CompileOptions::allowed_providers/denied_providers`，元素都是家族 `ProviderId`：allow 空表示
不限制，deny 优先，重复 ID 无额外语义。全局被排除的 Provider 不会被 Graph 查询 extension、发现候选或
准备算子，但 Context 的配置/构造已经发生，筛选不是 library 卸载或安全隔离。未知 allow-only 家族、
不匹配的必需实例或与 placement 冲突的约束不会被默默忽略。没有候选满足约束时返回 `Unimplemented`；
已匹配路线的资源拒绝可以保留 `ResourceExhausted`，真实 Provider 错误继续按原错误返回。无效家族、
异图节点和重复节点声明返回 `InvalidArgument`。

这些 span 只借用至 `compile()` 返回，之后可以释放或修改。筛选/索引只存在于冷编译，Invocation 不保留
它们，也不在 submit/replay 再查策略。普通用户全部留空即可。逐节点软偏好与完整成本/profile 的统一
择优仍未交付；不要把硬约束或 Context 安装顺序当成自动性能排名。

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
Provider 成功 replay；逻辑 View/Copy 若经 Provider 准备，同样受筛选。它不替调用方 executor/callback、
编译器插入的 backend transfer、失败诊断或 Completion 的独立合同证明零开销，也不把声明本身当成
性能资格化证据。

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

`constant()` 只声明逻辑 value；`CompileOptions::constants` 提供 payload。默认 `ConstantPolicy::kBorrow`
不复制字节，源 storage 必须不可变，并活过 Executable、全部 Invocation 和 in-flight Completion。即使
Provider 另有 packed representation，也不因此缩短借用期。

模型加载后要释放 staging buffer 或 mapping 时，显式选择物化：

```cpp
const std::array constants{ConstantBinding{
    *weight, weight_view, ConstantPolicy::kMaterialize}};
auto compiled = compile(*graph, context, {.constants = constants});
if (!compiled) return compiled.error();
// 此后可释放 weight_view 的源 allocation/mapping；不可再通过旧 view 访问它。
// Executable 及其 Invocation/Completion 共享编译时创建的不可变副本。
```

物化把完整 TensorDesc 所需的字节复制一次，保留 dtype、encoding、shape、strides 和 padding；它不是
重新量化或 Provider packing。目标为 constant 的 external storage 域（默认 0，也可由
`ExternalResourceContract`/value placement 指定），无需与源相同。配置的某个 target 必须支持这条
Compute transfer；没有明确 copy route 就失败，不偷偷用 host staging。零字节常量不提交复制。

源必须在调用前 ready，并在整个 `compile()` 期间保持不变；冷编译等待自有复制到终态，不执行用户
Graph，也不替用户等待此前独立 Queue 的工作。最终 Provider prepare 只看到有效副本；路线回退不重复
复制。编译失败不会接管或修改源，已接纳复制也会等待后再清理。副本资源的 `MemoryResource` owner
仍由调用者保活；“可释放 source allocation”不等于“可销毁仍承载副本的 resource 对象”。

跨 variant 共享与 Provider opaque packing 的唯一 resident/冷峰值预算尚未交付；
`materialized_constant_bytes()` 只统计 canonical 副本，不代表整个 Executable 的内存总量。

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
失败；preferred route 不可执行时按稳定候选顺序继续。冷 proposal 的 Unimplemented 表示没有对应能力；
真正 prepare/region compile 使用共同 `compute::ProviderPrepareOutcome`，只有明确的 unsupported 或
candidate-local resource-budget decline 可在回收部分 artifact 后回退。kind=Error 即使 code 是
Unimplemented/ResourceExhausted 也立即失败，不能从错误码推断 rollback。非法 kind/status 同样失败。
当前编译分别发现一次 region 与普通算子候选；普通算子按允许的 node/target/Provider 配置查询，即使前排
Provider 或 region 最后胜出，后排候选也可以被发现。`propose` 只返回冷能力与资源要求，不准备重型制品；
明确 unsupported 被跳过，其它查询错误立即传播，不能因该 Provider 最后未选中而吞错。
region 的 boundary、资源要求、runtime 与效果声明也在发现阶段查询并缓存；required target 不允许的
候选不做资源查询。查询不代表后面一定会 compile 该候选，Provider 不应在这里创建重型制品。
资源要求非法或后排查询出现真实错误，同样在准备任何制品前失败；明确 unsupported 只排除该候选。
某个 region 的已选资源路线，或某个 leaf 的 prepare 被明确安全拒绝后，释放本次构造的制品并排除该
候选，再从原候选池重新放置/分区，可以改选另一 Provider、另一 node set 或另一个允许的 target。禁止
leaf 回退时仍可改选完整 region 覆盖；允许 leaf 时也可以使用已安装的 operation 路线。重选不重复发现或资源查询，
不重置任何预算。region 联合成环、某个完整组合无 storage/copy 路线时，非递归搜索回溯到下一组合，
不会仅因该组合失败就淘汰其中所有候选。当前这是 leaf/region/target 同池的确定性可行解搜索，
不是 qualified 完整成本优化，也不穷举所有 storage/lane 组合。
先构造完整 leaf/region/copy 的物理步骤和 storage 路线（包括最终输出 copy），之后才准备选中的制品；
因此后续已知无能力/无合法 storage/copy 承载 target 时，不会先给前面的步骤做 JIT/packing。最终 prepare
只使用已选路线，仍经 Compute 校验；不能原地换 Provider/target/storage，明确 decline 必须整图重规划。
这可能重新准备已成功的前缀，全部计入累计预算。不承诺提前证明任意 Provider 的真实 prepare 必定成功：
精确 workspace/persistent、lane/内存和实际 Queue transfer 仍需后续校验。只有完整可行计划才参与下述 profile 成本比较。
最终 lane/schedule/workspace 校验也在同一冷重选事务中：已准备路线缺可用 lane，或完整 workspace
无法满足显式 storage 预算/允许的有界序列化时，释放本次制品及其调度/别名/内存决定，再尝试下一组合。
这不永久排除组合中的每个 Provider；所有路线都无解仍失败。真实 OOM、Provider/alias 合同错误不会被
按 Status 名称猜成“可以换路”。胜出报告不保留被拒绝路线的 delegated lane、序列化边或实现键。
`max_placement_attempts` 限制跨所有构造尝试的实际冷准备次数；若
在形成完整可行计划之前耗尽这个上限，仍然失败，不能返回部分图。
`max_route_search_states`（默认 65,536）另行限制累计尝试的候选状态，包括重叠/已拒绝的候选；不在重选时
归零。尚未找到可行计划就用尽，返回 ResourceExhausted。恰好用满且没有省略必需搜索不算受限。
两者都必须非零。候选表容量、搜索工作量和真正准备次数不是同一个预算。
启用 profile 优化且已有完整可行方案时，后续 state/prepare 耗尽可在 `kAllowLimited` 下保留它；报告分别
置 `route_search_limit_reached` / `placement_limit_reached`。strict 政策仍拒绝，真实 OOM/Provider 错误
不因已有方案而被忽略。未完成的 trial 不会成为 Executable。

`View` 的 required 输出域会反向约束其 input 0 的物理副本：必要时先 copy 输入，再在目标域上建立零拷贝
View。报告明确显示 copy；若 alias 根在内部副本，输出由 Invocation 持有，用 `output(value)` 取得，
不能再绑定一个 caller 输出。只想把最终切片复制到 caller 存储时，用 `ExternalResourceContract`，
不必要求 View 本体换域。

`max_operation_proposals`（默认 16,384）限制本次编译接收的普通算子候选总数，独立于实际 prepare 次数；
禁止 operation fallback 时不发现 leaf，该上限也不使用。`max_region_proposals`、
`max_region_proposals_per_provider` 和 `max_region_proposal_nodes` 分别限制 region 总数、每配置实例数量与
复制保存的 node-ID 总数。默认 `search_policy = CompileSearchPolicy::kAllowLimited`：达到
容量时停止扩大相应发现，只用已接收的候选生成并验证完整计划；per-provider 上限不阻止还有总额时
探索下一家。成功报告的 `search().optimization_limited()` 明确说明发现是否被截断，可再查看
`*_limit_reached` 字段、实际接收数和 prepare 次数。这不是全局最优或性能资格声明。

需要拒绝受限发现时使用 `CompileSearchPolicy::kRequireUnexhausted`；它不是要求无限搜索。恰好用满
容量但没有任何候选/provider 被省略，不标记受限。两种政策都不放宽 coverage、required placement、
workspace 等硬约束，也不吞真实 Provider/OOM 错误。
候选上限不是“保证找到任何存在的可行解”：当前普通候选按 node/target/Provider 的稳定顺序接收，过小
的上限可能让后面的节点无可用候选，此时 compile 失败而非返回不完整 Executable。

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

## 按完整计划的性能资料选路

普通用户不需要 profile：默认取配置顺序下第一个通过全部硬约束的完整计划。已有合格测量资料的部署方/
profiler 可以提供 `CompileOptions::profile_context` 和 `profiles`，让完整 leaf/region/target 候选继续比较。
这不是自动调优：compile 不测量、不执行用户图、不读取磁盘 cache；查表和搜索均不进入 warm submit。

`CompileProfileContext` 分清两类事实：

- owner 给真实 HYLE/backend/Provider build、hardware/driver/runtime、ISA/executor/placement 配置、各
  storage class、假定的 Queue/capture/host-wait replay 配置，以及语义实例和 workload 内容/测量合同指纹。
  `targets` / `storage_domains` 按 CompileEnvironment 顺序；`ProfileProviderBinding` 的 target/index/instance
  必须与当前 Context 相符。不要从显示名称、版本字符串或指针猜这些值；缺事实保持零。
- Graph 自动编码当前内建属性、拓扑/副作用、原 shape program、全部 concrete TensorDesc，以及实际
  Provider 实现键、physical copies、target/storage/lane、segment/dependency、alias/workspace layout 和常量
  borrow/materialize 策略。自定义 descriptor 的完整实例语义仍由 owner 的 `semantics` 覆盖，不能只填 schema ID。

只有 context、不传 profiles 时，`CompileReport::profile_identity()` 导出已选计划的 key。返回 spans 借用
报告，不借用 Context/常量；inputs 顺序是 Graph inputs 后接 Graph constants，outputs 是 published outputs。
可以用 per-node Provider/placement 硬约束构造其它具体路线并测量。测量必须覆盖整个 Graph 的真实 terminal
Completion，包含适用的 copy/sync/managed 成本；不能用单 kernel 时间求和。示意如下，变量中的身份与
区间必须来自真实已资格化资料，而非照抄常数：

```cpp
auto baseline = hyle::graph::compile(
    graph, environment, {.profile_context = &current_profile_context});
if (!baseline) return baseline.error();

// 独立 profiler 对这条完整路线取得合格的不确定区间后，登记观测。
const hyle::compute::RouteProfile observation{
    .identity = baseline->report.profile_identity(),
    .replay = qualified_replay_interval,
    .compilation = qualified_total_compilation_interval};
// 将 observation 与其它路线的资料合并成 qualified_route_profiles。
// 各 RouteProfile 借用的 report/key 存储必须仍存活。
auto selected = hyle::graph::compile(
    graph, environment, {.profile_context = &current_profile_context,
                         .profiles = qualified_route_profiles});
```

只有第一个可行方案有完整匹配成本才继续扩搜；否则保留确定性选择，unknown 不是“无限慢”。后选方案的
成本区间上界必须严格低于当前方案下界，且达到 `minimum_relative_improvement`，才替换当前方案；默认
阈值为零也要求区间不重叠。重叠/失配/缺值继续保留原方案；重复匹配观测取保守包络。预计 replay 次数
已给出时比较 `replay + 完整 compilation / expected_replays`，缺 compilation 不能参与摊销比较。

`search()` 给出 `feasible_plans`、`profiled_plans`、`unprofiled_plans`、`profile_replacements` 和
`selected_profile_cost`；这表示本次有限搜索的事实，不证明全局最优或为数据签发科学资格。最多保留一个
已验证方案与一个 trial，二者共用唯一 canonical 常量副本；后选真实错误仍导致 compile 失败。
只读匹配的各类拒绝原因可用 [Compute matcher](compute.md#只读性能资料的匹配) 检查。

坑：改变后续 Queue topology、capture 模式、输入分布或并发负载后，原选路预测不再适用，即使执行仍然
合法。`concurrent_invocations > 1` 会排除必须串行的 Provider 计划；当前尚无完整 Provider 确定性证明合同，
`require_determinism=true` 明确返回 Unimplemented，而不是仅凭 profile 标签宣称确定性。
完整 storage/lane 联合优化、bounded variants 和自动生成所有外部 build/device/custom 指纹仍未交付。

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

- `search()` 的 `operation_proposals/region_proposals`、已接收 region node 数、累计 prepare 次数、
  `route_search_states` 尝试状态数、`route_plan_rejections` 不可行组合数、
  `region_route_rejections/operation_route_rejections` 候选拒绝数和是否因
  容量省略了候选/provider；正常拒绝不等于搜索容量耗尽；
- specialization、target/device/backend、storage workspace/budget，以及每 lane runtime mode/role/Queue/capture 需求；
- physical value、external/constant/workspace/alias ownership；
- `constant(index)` 按 Graph 常量顺序给出 value、borrow/materialize policy、storage、TensorDesc 和
  canonical owned bytes；`borrowed_constant_count()`、`materialized_constant_count()` 和
  `materialized_constant_bytes()` 给汇总，报告本身不保活这些 allocation；
- operation fallback、provider region 和真实 copy step；
- leaf 的 `kRegionDeclined` 表示本轮曾安全拒绝包含该 node 的 region，不是 Provider 执行出错后重跑；
- shared/delegated segment、dependency、为 workspace budget 新增的 serialization edge、potential
  synchronization 和 concurrent-invocation 能力；
- provider/version、实际配置实例 `provider_instance`、dispatch、workspace、persistent/packed/borrowed
  bytes、blocking/threading 与 replay 类别。

`step(...)->profile_key` 保留 Provider 自行声明的稳定实现键，与临时 prepare token、配置实例编号都
不同；unknown 不会被自动补成上述编号，Graph 插入的 copy 也没有 Provider key。它是完整性能资料匹配
所需的一项，不代表有合格计时，更不参与无 profile 的排序。完整候选的用法见下一节。

Provider 名称/版本文本由报告复制，释放 Context、factory 或 Executable 不会令报告悬空。
`provider()` / `step()` 返回值中的文本 view 借用报告共享状态，不能活过最后一个报告 handle；复制/移动
报告不复制执行资源，也不缩短这份共享文本的生命周期。

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

## 外部可变资源：声明访问与顺序

普通 tensor 的输入/输出依赖不需要资源 ID。custom operation 还访问外部状态时，先按
[Compute 资源 schema](compute.md#custom-operation-的外部资源访问)声明每个端口的 read/write，再把端口
绑定到 graph-local `EffectResourceId`。例如更新一个外部累积缓冲区后读取它：

```cpp
// update_op / observe_op 是已由 context 中的 Provider 支持的 custom Operation；
// 均有一个资源端口，分别声明 Write 和 Read，此例没有 tensor 参数。
using namespace hyle::graph;
auto state = builder.effect_resource().value();
auto update = builder.add_node(update_op, {}).value();
auto observe = builder.add_node(observe_op, {}).value();
const std::array ports{state};
HYLE_RETURN_IF_ERROR(builder.bind_resources(update, ports));
HYLE_RETURN_IF_ERROR(builder.bind_resources(observe, ports));
HYLE_RETURN_IF_ERROR(builder.depends_on(observe, update));
auto graph = std::move(builder).finish().value();
const std::array backing{EffectResourceBinding{
    state, EffectResourceFootprint::memory(state_buffer.view())}};
auto compiled = compile(graph, context, CompileOptions{.effect_bindings = backing});
```

`finish()` 不会按添加顺序暗中插入依赖。同域写→读、读→写、写→写必须已有 data/control path；多个
并行 readers 后的 writer 必须依赖全部 readers。read/read 和独立资源无需串行化。缺端口绑定、异图 ID
和冲突未排序都会失败；失败后可以在 Builder 更正绑定或补 `depends_on()` 再 finish。重复端口可指向同一
ID，`Graph::node_resource_count/node_resource/effect_resource_id` 可查询逻辑声明。未知外部副作用使用
`kExternalSideEffect`，与全部具名访问及其它未知副作用保持保守顺序。

`effect_bindings` 是**访问范围证明，不是执行参数传递或资源所有权**。custom 实现仍须自行取得对应资源，
且声明必须覆盖它实际访问的范围；此入口不自动安装 custom execution。地址化内存使用 Mem 的
`ConstBufferView`，按 resource/allocation/offset/size 检查；不同 ID 的范围不得重叠，即使两个声明都只读。
需要 alias 时复用同一 ID；同一 backing 的不重叠 subview 可以使用不同 ID。非地址化外部对象使用
`EffectResourceFootprint::object(canonical_owner)`；不能用两个包装对象的地址掩盖同一 owner，也不能用
object 掩盖本应与 tensor 检查重叠的内存。

这个检查依赖 Mem 的 canonical allocation identity，不探测 OS 物理页。若同一内存经多个 handle/resource
重复导入或映射，调用方必须统一声明它的 backing 身份，不能把不同包装当作独立性证明。

已使用的每个资源需恰好一个有效 backing；未使用的 ID 可不绑定。Graph tensor input/constant/output
不得与 effect memory 隐式重叠，compile 检查常量，`Invocation::bind()` 检查输入和输出，不因绑定顺序
不同而漏检。需要从可变资源产生数据时，用 custom operation 的显式输出建立 SSA value，不把相同内存
重新伪装成普通输入。

编译会复制声明而不保留传入的数组/span；footprint 本身仍借用，实际 backing、resource provider 或外部
owner 必须活过该 Executable、全部 Invocation 和 in-flight Completion。footprint 固定，不随 tensor
rebind/reset 改变。不同 Invocation 使用同一外部可变资源时，仍由调用方 Queue/Completion 排序；
`supports_concurrent_invocations()` 不代表用户资源天然无数据竞争。

Region 编译收到按域合并的访问集与 backing 声明。Provider 必须在资源提案与最终制品中一致承诺
`preserves_effects`；缺承诺的路线在创建制品前排除，前后矛盾是合同错误而非可忽略 fallback。资源顺序
沿已有 data/control 边进入 region 收缩和 lane/Completion，不增加全局 registry、锁或 submit 扫描。

## 所有权与常见陷阱

- `ExecutionTarget` 的 Context 会被 plan 内部 artifact 保活；`StorageDomain` 中 `MemoryResourceRef` 的 provider 仍是 caller
  borrow，必须活过 Executable、Invocation 和全部 in-flight work。
- Tensor view 不拥有 payload。输入、输出、shared Queue、external resource 与 native kernel 参数都必须活到
  Completion 真终态；delegated provider invocation 由 Graph Completion 保活，丢弃 pending Completion 不会隐式等待。
- `output_ownership()` 为 `kCaller` 的计算结果绑定 mutable `TensorView`；`kSourceAlias` 借用原输入/
  常量，`kInvocation` 使用本次运行的私有 storage，`kExecutable` 则是已物化常量或它的零拷贝 View。
  后三种不接受重新绑定；成功完成后用 `Invocation::output()` 取得借用 view，不得活过对应 owner。
- `reset()` 保留已规划 workspace，但清 external bindings；in-flight 时失败。bind/reset 后 replay route
  失效，多 shared lane 必须重新 `prepare_replay()`。
- 不同 published outputs，以及 output 与仍存活 input，不得绕过 plan 形成危险 overlap。只在相同
  storage/resource、最后一次读取且 provider 明确允许时选择 in-place；其它 partial overlap 被拒绝。
- 外部资源按上述域和 read/write 规则排序；纯 node、独立域与同域 read/read 不要为了“看起来顺序一致”
  手工串行化。固定 effect backing 与每次运行可重新绑定的 tensor 是不同生命周期。
- Graph 可以保存合法 blocked/opaque TensorDesc，但没有理解该 encoding 的 provider 时 compile 会返回
  `Unimplemented`，不会把压缩 bytes 当普通 scalar storage。

source、headers + static、headers + shared 都选择 `hyle::graph`。三种交付是同一 C++23 surface，不是
Native FFI，也不承诺跨编译器/STL 的 C++ ABI。
