# HYLE Graph：何时编译可重复执行的计算 DAG

> 本页描述当前 CPU 多节点 C++ surface；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。GPU provider、partition/fusion 与动态
> shape 尚未签发。

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
- `Executable` 是不可变、可共享的 prepared plan。它拥有每个 node 的 `PreparedOperation` 和一份 Mem
  lifetime plan，不拥有每次请求的 bindings/workspace。
- `Invocation` 是 move-only、单-owner 的每次运行状态。并发请求从同一 Executable 创建不同 Invocation；
  `create_invocation()` 可以分配，稳定态 `bind/reset/submit` 不增长 metadata。

## 当前能表达什么

| 能力 | 当前合同 |
|---|---|
| DAG | 任意数量的已支持 typed node；共享输入、分支和多个 published outputs |
| op | Compute 当前的 `RmsNorm`、rank-2 `MatMul`、packed `SiluAndMul` |
| shape/layout | static `TensorDesc`；MatMul 的物理权重方向由 stride 表达 |
| compile | 每个 node prepare 一次；CPU topological plan；中间 tensor liveness 交给 Mem planner |
| run | caller 绑定所有 graph inputs/outputs；内部中间值由每个 Invocation 独占 |
| report | backend/provider、node 数、workspace bytes/backings、copy/sync/fallback |

当前没有 graph-owned constants、symbolic dimensions、provider region partition、fusion、GPU copy/barrier、
serialization/cache schema 或 control flow。权重目前作为普通 graph input 绑定；后续 provider packing 必须
在 compile artifact 中显式拥有，不能藏在第一次 submit。

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
auto norm_w = builder.input(norm_weight_desc);
auto gate_up_w = builder.input(*logical_weight_desc);
if (!x || !norm_w || !gate_up_w) return hyle::Status::InvalidArgument("input");

auto normalized = builder.add(RmsNorm{}, *x, *norm_w);
if (!normalized) return normalized.error();
auto gate_up = builder.add(MatMul{}, *normalized, *gate_up_w);
if (!gate_up) return gate_up.error();
auto activated = builder.add(SiluAndMul{}, *gate_up);
if (!activated) return activated.error();
if (auto status = builder.output(*activated); !status.ok()) return status;

auto graph = std::move(builder).finish();
if (!graph) return graph.error();
auto compiled = compile(*graph, context);
if (!compiled) return compiled.error();

auto invocation = compiled->executable.create_invocation();
if (!invocation) return invocation.error();
if (auto s = invocation->bind(*x, x_view); !s.ok()) return s;
if (auto s = invocation->bind(*norm_w, norm_weight_view); !s.ok()) return s;
if (auto s = invocation->bind(*gate_up_w, gate_up_weight_view); !s.ok()) return s;
if (auto s = invocation->bind(*activated, output_view); !s.ok()) return s;
auto done = invocation->submit(queue);
if (!done) return done.error();
return done->wait();
```

## 成本与常见误用

- compile 是冷路径：shape 复核、每节点 prepare 和 lifetime planning 均在这里发生。shape/layout 改变时
  重新构图/编译；不要每次请求重新 compile。
- `create_invocation()` 实例化 workspace；应按并发槽复用 Invocation。`reset()` 只清外部 bindings，保留
  已规划 workspace。一次完整绑定的 alias 校验最坏为
  `O(inputs*outputs + outputs^2)`，不分配；随后重复 `submit()` 仍为 O(nodes+bindings)。
- published output 必须绑定 mutable `TensorView`；graph input 可绑定 const 或 mutable view；内部 value
  不允许由 caller 覆盖绑定。
- 不同 published outputs 不得重叠，output 与仍存活的 external input 也不得重叠。当前唯一允许的跨 value
  alias 是 RMSNorm output 与其已经到最后一次读取的第一输入**精确同一 byte range**；partial overlap
  即使地址接近也会拒绝。同一大 backing 中互不重叠的 weight/input subview 是正常用法。
- CPU 当前同步完成，但仍要检查 `Completion::wait()`。payload/resource/Queue 必须遵守 Compute 的终态
  生命周期；未来异步 backend 不会因 CPU 的偶然同步语义而放宽合同。
- CompileReport 当前 CPU 路线应报告零 copy、零显式 sync、无 fallback；不要忽略报告后猜测隐藏成本。
- Graph 不启动 worker，也不依赖 Sched。需要请求调度、取消和服务 drain 时，在消费层组合它们。

source、headers + static、headers + shared 都选择 `hyle::graph`。公开 `GraphBuilderOptions` 可注入 Mem
allocator，因此 Graph 的公开构建闭包包含 Compute、Error 和 Mem；Container/Util 只用于私有实现。
三种交付均为 C++23 surface，不是 Native FFI，也不承诺跨工具链 C++ ABI。
