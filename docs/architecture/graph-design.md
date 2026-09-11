# HYLE Graph：计算 DAG、编译与可复用执行计划

> 决策状态：`ACCEPTED`。本文定义 Graph 的长期产品合同；当前实现状态只查
> [`module-status.json`](../implementation/module-status.json)。当前 Graph 仍为 `AUDITED`，多节点 CPU
> 纵切只证明本轮登记的 DAG/Executable/Invocation 合同。旧 `TaskGraph`、parallel/data-parallel API 与
> benchmark 已退出产品树；被删除的旧字节只可从 Git 历史取得调查事实，历史 task/evidence 不是当前
> 设计输入、实现模板或用户承诺。本页不签发用户可用性，也不要求兼容旧头。

## 1. Graph 解决什么问题

Graph 把**模型无关的逻辑计算 DAG**编译成可重复、可检查、低开销提交的执行物。它拥有：

- logical tensor/value 与 operation node 的拓扑；
- shape/layout/resource/side-effect 约束及图合法性；
- provider/backend capability matching、region partition 和 placement；
- fusion、repack、copy、同步、liveness、alias 与 workspace 规划；
- compiled artifact 的组合、重复执行和每次调用状态分离。

Graph 不是通用 task executor，也不是模型 runtime。Sched 拥有 task/fiber/coroutine、取消、等待和线程；
Compute 拥有 device/context/queue/kernel/provider 与单个 region 的执行；Mem 拥有 storage 与通用 lifetime
packing。推理引擎仍拥有模型构图、KV/prefix-cache、batch/admission、sampling、循环停止和请求策略。

## 2. 三个概念层，两个公开产品

系统需要 Compute、Graph 和 compiled plan 三层概念，但不新增一个含义模糊的 `HYLE::plan` 产品：

```text
HYLE::compute                         直接 op/kernel 与 backend/provider runtime
HYLE::graph                           逻辑图、编译和执行 façade
  graph_ir                            逻辑计算 DAG
  graph_compile                       冷路径编译/partition/placement/planning
  graph_plan                          内部执行物与 invocation runtime
```

`graph_plan` 是内部 factor/实现职责；对 C++ 用户公开的结果类型是 `graph::Executable`。这样用户不会在
“graph 还是 plan”之间选择，构建系统仍能把冷编译依赖与热执行闭包分开。

## 3. 四个生命周期对象

概念上的正常路径固定为：

```text
GraphBuilder -> Graph
Graph + CompileEnvironment/Options -> Executable + CompileReport
Executable -> Invocation
Invocation.bind(...) -> submit(...) -> compute::Completion
```

### `GraphBuilder`

- 单 owner、可修改；添加输入、常量声明、operation、输出与显式依赖。
- 返回 typed `ValueId/NodeId`，不把裸下标、地址或第三方 node 泄漏为身份。
- `finish()` 做结构验证并产生不可变 Graph；失败不留下半可用图。

### `Graph`

- 不可变、可共享、与设备/provider 无关；持有逻辑描述而不持有运行时 payload。
- value 至少描述 dtype、rank/shape constraint、stride/layout、storage class 和可变/常量意图。
- node 使用 Compute 的公共 operation schema；side effect、resource write 和 control dependency 必须显式。
- hash/fingerprint 只在完整 canonical contract 成立时提供，不能把进程地址或不稳定类型布局当缓存键。

### `Executable`

- 编译后不可变、可共享；绑定 graph fingerprint、shape specialization、device/backend/provider identity
  和 compile options。
- 拥有 provider compiled artifacts、预打包常量及不可变 plan metadata；清楚报告其 storage owner。
- 暴露输入/输出合同、workspace 要求、所选 regions、copies/sync、fallback 与并发能力。
- 不承诺跨 provider/version/device 可移植，也不在首轮承诺序列化格式。

### `Invocation`

- 每次并发执行独占的可变对象；绑定 inputs/outputs、动态 dimension values、workspace 和 provider state。
- 默认非线程安全，但可在上次 Completion 结束后 reset/reuse；并发请求各用一个 Invocation。
- submit 后所有 borrow 和 workspace 活到 Completion 终态。取消不允许提前复用。
- Provider 若需要串行 context，Graph 必须在 report 中声明并由 queue/resource admission 表达，不能藏在
  process-global mutex 中。

把 Executable 与 Invocation 分开不是 API 装饰：TensorRT 的 engine/context、PJRT 的 executable/buffer、
oneDNN 的 compiled partition/scratchpad 和真实推理引擎的 graph capture/replay 都说明“可共享编译物”与
“每次运行状态”有不同并发生命周期。

## 4. IR 的最小完整语义

HYLE Graph 是有限计算 DAG，不是任意控制流 IR。完整产品合同必须能表达：

- 多输入/输出 operation，显式常量与 external binding；
- static shape 和带约束的 symbolic dimension；编译结果按明确 shape signature specialization；
- dense/strided/layout-tagged 与量化 storage 描述，但不绑定模型文件格式；
- in-place/alias 候选与禁止 alias 的输出；最终合法性由 operation/provider 合同决定；
- copy、repack 和 host/device boundary，而不是把它们藏进 provider；
- 有副作用或外部资源访问的 node 及其顺序边；纯 node 默认可重排；
- typed custom operation extension，供 HYLE/消费者 kernel 接入，不使用字符串属性袋或裸 callable。

首轮不提供任意 `if/while`、递归图或模型脚本解释器。recurrent-depth、speculative 和 adaptive compute
先由引擎重复提交一个或多个 Executable，并负责 active-set、停止与重聚批。Graph 必须让这种 replay
低成本且 state binding 显式；只有第二个通用消费者证明 region/control-flow primitive 必要时才扩 IR。

## 5. 编译流水线

`graph_compile` 按可检查阶段工作：

1. **validate/canonicalize**：检查拓扑、shape、dtype、alias、side effect 和输出；
2. **infer/normalize**：传播 shape/layout 约束，把等价 operation 变成稳定形式；
3. **capability match**：询问每个 Compute provider 对 op 或 region 的真实支持与成本；
4. **partition/place**：选择成本有效的 regions 与 devices，不机械追求节点数最大的 partition；
5. **materialize boundaries**：插入明确的 copy、repack、barrier 和 provider handoff；
6. **liveness/alias/workspace**：把 tensor facts 转成 Mem `LifetimePlanner` 能理解的 interval/resource 请求；
7. **lower/compile**：取得 provider artifact、native command/capture 或 HYLE kernel route；
8. **emit plan/report**：生成不可变 graph plan 与可诊断 compile report。

Partition 成本必须覆盖跨 region copy/sync、layout conversion、thread handoff、workspace、首次编译和常量
packing。Provider 的 `supports(op)` 不是“选它一定更快”。无法覆盖的 node 必须回到已登记 fallback 或使
编译失败；不能在运行时突然下载依赖、切 backend 或用 host scalar 静默求值。

ONNX Runtime 的 execution-provider partition、oneDNN 的 partition/compiled partition、TensorRT 的
network/engine/context 与 vLLM/SGLang 的 shape specialization/capture 都作为该流水线的独立参照；HYLE
不复制它们的 public API，也不把其中任何一个强行定为唯一执行引擎。

## 6. Plan 的热路径合同

内部 graph plan 是已经排好序的 region/copy/sync/workspace 程序。每个 step 持有预绑定 callable 或
backend command；Invocation 只填充运行时 slots。

- prepare 完成后，steady submit 不遍历原始 DAG、不重新推 shape、不搜索 provider、不 JIT、不打包常量；
- workspace layout 对固定 specialization 不变；允许 caller/pool 预留，正常 replay 零 heap allocation；
- per-region 的一次预绑定 dispatch 可以存在，per-node/per-element 的通用虚调用或字符串 lookup 不可接受；
- host wait 只在合同要求时出现；可由 queue/event 表达的 dependency 不退化成同步 round-trip；
- plan 必须可枚举 step、device、provider、copy bytes、workspace 与 serialization constraint，便于归因。

复杂度基线：图结构验证、拓扑和 liveness 应为 `O(V + E)`；在 `P` 个已登记 provider 上的 capability
matching 应有明确上界，不得随运行次数重复。更复杂的 partition/cost search 若采用启发式，必须登记
最坏预算与 deterministic fallback。

## 7. Compute、Mem、Sched 与 Obs 的边界

```text
graph_ir      -> compute_ops
graph_compile -> graph_ir + compute provider contract + mem
graph_plan    -> compute_core + mem
graph_sched   -> graph_plan + compute_sched + sched       # 可选组合
```

- Graph 向 Mem 交付 lifetime/alias/resource facts；Mem 不理解 Tensor、Node 或 provider policy。
- Graph 向 Compute 交付 region 和 bindings；Compute 不反向读取全图做偷偷融合。
- Sched 决定 CPU work、blocking provider、请求取消和公平；Graph 只声明 dependency/resource readiness。
- Obs 接收 compile/submit/region/completion seam；Obs OFF 时不改变 plan layout 或热路径分支。
- Error 保存 node/region/provider/backend context；不能另造 graph 私有 TLS error。

Compute 不依赖 Graph。直接 kernel/op 用户因此不取得 graph compiler、container-heavy IR 或 Sched runtime。

## 8. 旧 Graph 资产的处置

当前源码中的三族能力不能因目录叫 `graph` 就继续留在新产品：

| 旧能力 | 新裁决 |
|---|---|
| incremental dirty/cached `TaskGraph` | `DEFER` 为未来 `graph_incremental` 候选；只有 CAD/build 等真实消费者出现才恢复并重新审计 |
| `parallel_for/reduce/scan/sort` | 不属于计算 DAG；若 Sched 原语不足且有独立用户，另审 `parallel` factor/product |
| data-parallel partition helper | 作为 Graph compiler 的私有启发式或未来 `parallel` 候选逐入口裁决，不保留旧公开面 |

迁移时逐入口标记 KEEP、MOVE、REPLACE、DELETE；没有消费者和自然 owner 的旧 API 直接删除，Git 历史
负责恢复。不能为了兼容旧 benchmark，把 callable task DAG 与 tensor compute DAG 焊成一个万能类型。

## 9. 性能与正确性资格

Graph 至少要用三种不同层级验证：

1. **one-op 税**：Graph Executable 对同一 PreparedOperation 的额外 steady 成本；同 buffer/layout/queue；
2. **multi-op 收益**：真实 region 的 submission、fusion、workspace reuse、copy/sync 与 direct sequence 对比；
3. **消费者纵切**：量化 transformer block 的 prefill/decode，以及 Vulkan compute 的资源流水线。

每项都记录 compile/cold 与 execution/warm 两种边界，不能把 compile amortize 后与 peer 每次 prepare 对比，
也不能把 HYLE external storage 与 peer 内部 cache 混为同合同。正确性覆盖 shape/stride/zero-size、alias、
unsupported op、OOM、provider failure、device loss、取消、迟到 completion、并发 Invocation、重复 reset、
partial compile failure 和销毁顺序。

Graph 的价值不要求每个 node 都由 HYLE kernel 获胜。它必须证明选择和组合没有 material 税，并在至少
一个真实 multi-op workload 上通过减少 submission、copy、repack、allocation 或同步形成可解释收益。

## 10. 首轮纵切和停止点

开工与 [Compute 的顺序](compute-design.md#10-开工顺序)共同执行：

1. 用 CPU reference provider 建 `GraphBuilder -> Graph -> Executable -> Invocation -> Completion` 的 one-op
   完整旅程；
2. 加 HYLE SIMD/native provider，证明同一 Graph 无需改写即可选择另一实现；
3. 扩为 normalization → matmul → gated activation，接 Mem liveness/workspace；
4. 接 ggml/oneDNN region，验证 provider-owned allocation/threading 的诚实边界；
5. 优先落到两条主平台栈：NVIDIA 的 CUDA backend/providers；Apple 先资格化 MLX compiled region，
   同时保留原生 Metal custom-kernel/interop 和 MPSGraph/MPS 候选；Core ML 留在模型级 routed runtime
   边界；
6. 再落到正式支持的 Vulkan SPIR-V，验证消费级设备与 renderer 的 external memory/semaphore handoff；
7. 补齐 HIP/ROCm track，再扩 attention、quantized layout 和重复 decode。

每一步都有独立 closure card、固定 consumer slice 和停止预算。one-op lifecycle、P0 成本或三种 C++ 交付
未通过时，不得用“后面完整 graph 会摊薄”进入下一步；provider region 不成立时也不得扩大 operator
catalog 掩盖根因。
