# HYLE Graph：何时编译一个可重复执行的计算图

> 当前页面描述正在收口的 CPU one-op C++ candidate，不签发多节点 compiler。精确施工状态见
> [`module-status.json`](../implementation/module-status.json)。

Graph 适合“先构造逻辑计算、冷路径编译一次、再为多个请求反复绑定 storage 并执行”的任务。如果程序
只需要直接执行一个已知 operation 或工程 kernel，使用 [Compute](compute.md)，无需构图。Graph 不是
Sched task graph，也不拥有模型循环、KV cache、batching、sampling 或 serving。

## 四个生命周期对象

```text
GraphBuilder -> immutable Graph
Graph + Compute Context -> Executable + CompileReport
Executable -> Invocation
Invocation.bind(...) -> submit(Queue) -> Compute Completion
```

- `GraphBuilder` 是单 owner 的构造器；失败的 mutator 不改变它。`finish()` 后得到自含逻辑元数据的 Graph。
- `Executable` 是不可变、可复制和可跨线程共享的已编译 artifact。`CompileReport` 明示 backend/provider、
  workspace、copy、同步和 fallback，而不是隐藏成本。
- `Invocation` 是独占、move-only 的每次调用 binding 集；并发请求从同一个 Executable 各建一个
  Invocation。异步 Completion 终态前不得 reset、rebind 或 resubmit。
- `ValueId`/`NodeId` 只属于创建它们的 graph generation；跨 builder 使用会返回错误，而不是偶然按下标
  命中另一个值。

## 当前 one-op 路线

```cpp
#include "hyle/graph/executable.h"

hyle::graph::GraphBuilder builder;
auto input_id = builder.input(desc);
if (!input_id) return input_id.error();
auto output_id = builder.add(hyle::compute::RmsNorm{}, *input_id);
if (!output_id) return output_id.error();
if (auto status = builder.output(*output_id); !status.ok()) return status;

auto graph = std::move(builder).finish();
if (!graph) return graph.error();
auto compiled = hyle::graph::compile(*graph, context);
if (!compiled) return compiled.error();

auto invocation = compiled->executable.create_invocation();
if (!invocation) return invocation.error();
if (auto status = invocation->bind(*input_id, input_view); !status.ok()) {
  return status;
}
if (auto status = invocation->bind(*output_id, output_view); !status.ok()) {
  return status;
}
auto completion = invocation->submit(queue);
if (!completion) return completion.error();
return completion->wait();
```

当前 Graph 只接受一个 dense rank-1 f32 RMSNorm node，可选一个 weight input，并产生一个 output。compile
准备一次 Compute operation；steady submit 只读取固定 binding slots，不重新遍历 Graph、选择 provider、
分配或 JIT。当前 report 的 workspace/copy/synchronization 为零且不使用 fallback。

## 常见误用

- 不要把 one-op API 当成完整 DAG compiler。fusion、partition、liveness、alias/workspace packing、GPU
  capture 和 control flow 尚未实现。
- 不要为单个直接 operation 强制构图；Graph 的价值来自未来多节点规划与重复执行，不来自包装层本身。
- Graph/Executable 拥有元数据和 prepared artifact，不拥有绑定的 Mem payload。storage、resource、Context
  和 Queue 必须满足 Compute Completion 的生命周期合同。
- `reset()` 只清 binding，不能改变 Executable specialization。shape/layout 变化要重新构图或编译。
- CPU queue 当前同步完成，但不要依赖“submit 返回就永远可复用”的偶然事实；异步 backend 会严格要求
  Completion 终态。

source、headers + static 和 headers + shared 都链接 `hyle::graph`；它会传递公开 Compute 合同，而不会
把 Sched、通用 Container 或第三方 graph runtime 暴露给用户。三种交付都是 C++23 surface，不是
Native FFI。
