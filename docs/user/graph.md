# HYLE Graph：何时编译可重复执行的计算 DAG

> 本页描述当前 Graph v3 CPU 实现检查点；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。Graph v3 的通用 IR、region partition、
> ordered submission、可枚举报告、策略预算和异步 Invocation 生命周期已经进入实现检查点，但生产
> multi-op provider 尚未签发；选择 oneDNN 时仍只执行受支持的单个 row-major f32 MatMul node。真实
> oneDNN/ggml multi-op region、GPU provider 与 dynamic shape 仍是后续资格项。

Graph 适合“先定义一段模型无关计算，冷路径编译一次，再为不同请求绑定 storage 并反复执行”。如果只
执行一个已知 operation 或工程 kernel，直接使用 [Compute](compute.md)。Graph 不是 Sched task graph，
也不拥有模型循环、KV cache、batching、sampling 或 serving。

## 生命周期和所有权

```text
GraphBuilder -> immutable Graph
Graph + Compute Context -> Executable + CompileReport
Executable -> one Invocation per concurrent run
Invocation.bind(...) -> submit(Queue) -> Compute Completion
```

- `GraphBuilder::create()` 是可失败的单-owner 构造入口。输入和 node metadata 使用可注入 allocator；失败
  的 mutator 不发布部分逻辑修改。
- `ValueId`/`NodeId` 带 graph generation。一个 builder/graph 的 ID 不能误用于另一个实例；Graph 的
  node 查询也接收 `NodeId` 而不是把裸下标当身份。node 只能读取已创建 value，因此循环不能被构造出来。
- `Graph` 是不可变、可复制、可跨线程共享的逻辑元数据；它不拥有 tensor payload、Context 或 Queue。
- `Executable` 是不可变、可共享的 prepared plan。它拥有获选 provider-region artifact、其余 node 的
  单-op `PreparedOperation` 和一份 Mem lifetime plan，不拥有每次请求的 bindings/workspace。
- `Invocation` 是 move-only、单-owner 的每次运行状态。并发请求从同一 Executable 创建不同 Invocation；
  `create_invocation()` 可以分配，稳定态 `bind/reset/submit` 不增长 metadata。一次 submit 到最终
  Completion 终态之间不可 rebind/reset/reuse；`in_flight()` 可诊断该状态。

## 当前能表达什么

| 能力 | 当前合同 |
|---|---|
| DAG | 任意数量的 typed node；共享输入、分支、显式 control dependency 和多个 published outputs |
| value | graph input、compile-time constant 或 node result；input/constant 本身也可 publish |
| node | 通用 span arity；`add_node()` + `result(node,index)` 表达零/多输出，单输出 typed `add()` 保留自然短路径 |
| op | Native 当前可执行 View、Copy、Fill、Cast、Unary、Binary、Select、Reduce、RMSNorm、LayerNorm、rank-2 MatMul 与 packed SiLU GatedActivation；其他合法 schema 在无 provider 时返回 `Unimplemented` |
| shape/layout | static `TensorDesc`；byte stride、block axis 与 opaque identity 随 value 原样保留；MatMul 的物理权重方向由 descriptor 表达 |
| compile | constant 一次绑定；provider 冷路径提出 region，Graph 确定性选不重叠区间并只编译获选 artifact；其余 node 单-op fallback；中间 tensor、scratchpad、view alias 与安全 in-place liveness 交给 Mem planner |
| run | caller 绑定所有 graph inputs 与普通 materialized outputs；发布的零拷贝 View 若根在私有 workspace，则由 Invocation 保活并通过 `output()` 取得；内部中间值、provider run state 与 scratchpad 均由该 Invocation 独占 |
| report | `CompileReport` 独立拥有轻量诊断副本；除聚合摘要外，`step(i)` 枚举 kind、回退原因、provider/version、node range、workspace、persistent/packed/borrowed bytes、copy/sync/dispatch 与线程、阻塞和并发 Invocation 事实 |

当前没有已签发的生产 multi-op region/fusion、symbolic dimensions、GPU copy/barrier、serialization/cache
schema 或通用 control flow。constant payload 默认由 Executable 借用；后续 provider packing
必须在 compile artifact 中显式拥有并写入 report，不能藏在第一次 submit。

## Region 选择不是运行时搜索

默认 `RegionSelectionPolicy::kConservative` 只考虑 provider 对当前静态 Graph 声明为正净收益的连续拓扑
区间；Graph 冷路径确定性选择不重叠组合。`max_region_proposals` 与
`max_region_proposals_per_provider` 分别是全局和单 provider 硬预算，耗尽会返回
`ResourceExhausted`，不会截断候选后悄悄生成另一份计划。需要严格逐算子调试时可选择 `kDisabled`；
必须取得完整 fused/captured artifact 的调用方可设
`OperationFallbackPolicy::kReject`，任何未被 region 覆盖的单-op 都使 compile 返回 `Unimplemented`。

region 内部临时 value 由 provider artifact 管理，Graph 只为边界 value 和 provider 声明的每次运行
workspace 规划 backing。当前没有生产 provider 因此默认行为仍是单-op fallback；存在 region 机制并不代表
oneDNN、ggml 或未来 GPU 路线已经通过正确性/性能资格。

## 自然的多节点路线

下面构造 inference 常见的 `RMSNorm -> MatMul -> SwiGLU` 片段。存储中的权重是 `[2*ffn, hidden]`；逻辑
MatMul 需要 `[hidden, 2*ffn]`，所以只转置 description，不搬数据。

```cpp
#include <hyle/graph/executable.h>

using namespace hyle::compute;
using namespace hyle::graph;

auto logical_weight_desc = stored_weight_desc.transposed();
if (!logical_weight_desc) return logical_weight_desc.error();

auto created = GraphBuilder::create();
if (!created) return created.error();
GraphBuilder builder = std::move(*created);

auto x = builder.input(x_desc);
auto norm_w = builder.constant(norm_weight_desc);
auto gate_up_w = builder.constant(*logical_weight_desc);
if (!x || !norm_w || !gate_up_w) return hyle::Status::InvalidArgument("input");

auto normalized = builder.add(RmsNorm{}, *x, *norm_w);
if (!normalized) return normalized.error();
auto gate_up = builder.add(MatMul{}, *normalized, *gate_up_w);
if (!gate_up) return gate_up.error();
auto activated = builder.add(GatedActivation{}, *gate_up);
if (!activated) return activated.error();
if (auto status = builder.publish(*activated); !status.ok()) return status;

auto graph = std::move(builder).finish();
if (!graph) return graph.error();
std::array constants{
    ConstantBinding{*norm_w, norm_weight_view},
    ConstantBinding{*gate_up_w, gate_up_weight_view},
};
auto compiled = compile(*graph, context, CompileOptions{.constants = constants});
if (!compiled) return compiled.error();

auto invocation = compiled->executable.create_invocation();
if (!invocation) return invocation.error();
if (auto s = invocation->bind(*x, x_view); !s.ok()) return s;
if (auto s = invocation->bind(*activated, output_view); !s.ok()) return s;
auto done = invocation->submit(queue);
if (!done) return done.error();
return done->wait();
```

## 成本与常见误用

- compile 是冷路径：shape 复核、每节点 prepare 和 lifetime planning 均在这里发生。shape/layout 改变时
  重新构图/编译；不要每次请求重新 compile。
- `constant()` 只声明逻辑 value；`compile()` 要求每项恰好一个 `ConstantBinding`，拒绝遗漏、重复、异图
  ID 或错误 descriptor。v3 保守地始终 borrow source，payload 必须活过 Executable 及其全部 completion；
  report 的 packed count 表示 provider 另有一份 owned representation，不缩短 source lifetime。不要把
  request input 假装成 constant，也不要在 source 失效后继续复用 Executable。
- Graph 可以保存合法的 block/opaque TensorDesc，但不会假装 CPU native 已能执行它。若当前 Context 没有
  理解该 encoding 的 provider，compile 返回 `Unimplemented`；不会静默解码、复制或把压缩 bytes 当作
  plain scalar storage。
- `create_invocation()` 实例化 workspace；应按并发槽复用 Invocation。`reset()` 只清外部 bindings，保留
  已规划 workspace，并在 work in flight 时明确失败，因此必须检查返回 Status。一次完整绑定的 alias
  校验最坏为 `O(inputs*outputs + outputs^2)`，不分配；随后重复 `submit()` 为
  `O(bindings + emitted steps)`。
- 普通 materialized published output 必须绑定 mutable `TensorView`；graph input 可绑定 const 或 mutable
  view；内部 value 不允许由 caller 覆盖绑定。`Executable::output_ownership()` 区分独立 caller destination、
  source alias 与 Invocation workspace。后两者都不接受 output bind：成功完成后用
  `Invocation::output(value)` 取得借用 view。published input/constant/View 直接复用已有 source binding；
  private intermediate 上的 View 由 Invocation 保活。若必须写入指定 destination，应在 Graph 中增加显式
  `Copy`，不要让 View 暗中复制。
- 不同 published outputs 不得无计划地重叠，output 与仍存活的 external input 也不得重叠。operation traits
  只给 alias 上界，真正选中的 PreparedOperation 还会按 provider/shape/layout 收窄；例如 dense Unary 可以
  原地，而广播 Binary 不会被误规划成原地。可物化算子的 optional in-place 只发生在输入最后一次读取且
  description 相同时；View 则按推导出的 offset 传播零物化 subrange。其他 partial overlap 一律拒绝。
- `depends_on(node, prerequisite)` 只接受更早的 node；纯 node 可不写控制边。有外部副作用的 custom node
  必须形成显式或数据依赖链，否则 `finish()` 失败。它不是用来把整个 DAG 手工串行化的顺序 API。
- 一次 submit 在同一 ordered Queue 中连续签发全部 materializing steps，不逐 step host wait，只返回代表
  已接受完整 queue prefix 的最终 `Completion`。同一 Queue 不允许两个 producer 同时录入 sequence；不同
  Queue 可并发。若后段在前缀已接纳后同步拒绝，最终 Completion 保留前缀并报告该错误。
- CPU 当前同步完成，但仍要检查 `Completion::wait()`。异步运行期间 Invocation、provider state 和 workspace
  由最终 Completion 保活；丢弃 pending Completion 不会隐藏等待，也不能提前释放状态，backend 必须在真实
  终态回调释放。Tensor payload/resource 与 Queue 仍是 caller borrow，必须自行活到终态。
- CompileReport 可区分 HYLE-native、oneDNN 与 mixed，并逐 step 披露 fallback 原因、copy/sync/dispatch、workspace 与并发限制等成本；
  report 副本不保活 heavyweight Executable artifact。当前 CPU 路线仍应为零 copy、零显式 sync；oneDNN
  node scratchpad 已计入同一 Mem workspace plan，不需要用户另行分配。
- Graph 不启动 worker，也不依赖 Sched。需要请求调度、取消和服务 drain 时，在消费层组合它们。

source、headers + static、headers + shared 都选择 `hyle::graph`。公开 `GraphBuilderOptions` 可注入 Mem
allocator，因此 Graph 的公开构建闭包包含 Compute、Error 和 Mem；Container/Util 只用于私有实现。
三种交付均为 C++23 surface，不是 Native FFI，也不承诺跨工具链 C++ ABI。
