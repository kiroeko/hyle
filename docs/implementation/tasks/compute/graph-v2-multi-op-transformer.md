# Graph v2：多节点 CPU Transformer 片段与执行计划

## Closure card

- component / owner：`graph` / Graph；round=`graph-v2-multi-op-transformer`；
- baseline：`1b32fb8e792567b23d2aae15b1174ac52e0f8ffb`；previous terminal：
  [`graph-v1-clean-dag`](graph-v1-clean-dag.md)；
- 合法 trigger：v1 terminal 明确只签发 one-op；已接受的
  [`graph-design`](../../../architecture/graph-design.md)与四个真实推理引擎共同要求 normalization →
  matmul → gated activation 的可复用多节点计划；
- 正常用户任务：构建一个模型无关的多节点 DAG，冷路径 compile 一次，取得公开 CompileReport，创建
  多个互不共享 mutable run state 的 Invocation，绑定 caller input/output 并重复执行；
- 正交反偏置任务：工程用户构造分支/共享中间值的短 DAG，证明 API 不是 decoder 顺序表或固定三节点
  benchmark façade；
- scope：把固定 one-op storage 改为可扩展 typed DAG；支持本片段需要的多 input/node/output；在构图期
  保证 ID origin 与拓扑合法；compile 生成不可变 Executable，完成 shape/layout propagation、liveness、
  alias 与 CPU workspace plan；Invocation 拥有或显式借用每次运行 workspace，并顺序提交已经 prepared 的
  artifacts；CompileReport 如实列出 workspace/copy/sync/fallback；
- non-goals：通用 task graph、模型控制流、动态 batching/KV/sampling、外部 provider partition/fusion、
  GPU placement/copy、serialization/cache schema、Graph 自有 scheduler 或 Native FFI；
- 直接依赖：公开 `core/error/compute/mem`（公开 Builder options 暴露 `AllocatorRef`）；内部使用 `mem`
  lifetime planning 与 `container`/`util_core` metadata primitive，不能复制 allocator/container；Compute
  永远不反向依赖 Graph；
- 直接消费者：Compute v2 CPU op catalog、独立 multi-op package sample、后续 ggml/oneDNN region 卡；
- P0：compile 允许有界 allocation；记录 O(V+E) validate/plan、workspace packing 与 metadata；steady submit
  必须 O(V+bindings)、零 route search/JIT/packing/metadata allocation，且每节点至多一次预绑定调用；
- P1：候选后比较 Graph Invocation 与完全相同 PreparedOperation、Queue、TensorView、workspace 和完成边界
  的手工 direct sequence；compile/create/bind 在测量外，5% material 门，方向如实登记；
- 机器与验证：Windows、WSL、macOS correctness/header；WSL sanitizers；branch/shared-value、invalid/cross
  builder ID、shape mismatch、cycle-by-construction、OOM/rollback、workspace shortage/alignment、alias/liveness、
  concurrent independent Invocation、reset/resubmit/teardown、direct consumer、source/static/shared、red-team；
- 调查预算：真实引擎/成熟 graph compiler 各一遍；API/plan 纠正最多两轮，性能纠正最多两轮，每个正式
  workload 最多一次 `n=15` qualification；
- stop：若多节点 metadata owner 需要新的独立质因数、Compute v2 合同不足、workspace owner 无法与 Mem
  合法组合，或 steady path 出现无法解释的 material 税，则停止并形成精确 blocker/reopen trigger。

## SCOPED 边界

本状态只授权 DAG/compiler/plan 审计，不授权产品修改。AUDITED 前必须交付：

1. Graph IR、compile artifact、Invocation/workspace 的 owner 与失败矩阵；
2. 对 ggml graph、oneDNN graph/partition、ONNX Runtime execution plan 及四个真实引擎的形状对账；
3. 现有 one-op 入口向多节点形状的逐项迁移裁决；
4. 有限 `KEEP/FIX/BORROW/DEFER/REJECT` ledger 和自然 C++ journey。

## AUDITED IR、owner 与执行计划裁决（2026-09-12）

### 对账输入

- ggml `e91ded11bdcd` 将 logical graph、backend split、graph allocator 和 CPU work buffer 分开；
  `ggml_gallocr_reserve/alloc_graph` 证明中间值 lifetime planning 是 compile/prepare 事实，不应在每次 node
  submit 临时分配；
- llama.cpp `e71b80510c84` 的 `llm_graph_context::build_ffn` 与模型实现反复产生共享输入、分支 MatMul、
  gated activation 和下游节点，否决固定 decoder 顺序表；
- oneDNN `6b7f1fbbb18e` gated-MLP 以 logical tensors/ops 构图、finalize/partition/compile 后重复 execute，
  combined weight 仍由 logical stride/offset 表达；
- vLLM `51da0ca66c80`、SGLang `dcebe8c4733a` 与 ds4 `f62ca29a3087` 的执行层都区分 immutable
  kernel/model artifacts 与 per-request buffers/streams；模型特有 scheduler、KV 与 batch 状态不属于 Graph；
- ONNX Runtime 的 immutable execution-plan/per-run frame 模式只作 owner 交叉验证；本轮不借其 schema、
  allocator 或 provider ABI。

### 迁移与有限 ledger

| ID | v1/候选形状 | 裁决 | v2 合同 |
|---|---|---|---|
| G20 | fixed two inputs/one node/one output arrays | `REPLACE` | allocator-owned growable value/node/edge/output metadata；不保留 fixed fast façade |
| G21 | `ValueKind::kOutput` 同时表示 node result | `FIX` | ID 表示 graph input 或 node output；“published output”是独立 graph role，不制造第二个 value |
| G22 | builder accepts only RMSNorm | `FIX` | generic typed `Operation` + typed conveniences；每次 add 调 Compute 唯一 shape inference |
| G23 | one-op `Graph` inline value | `FIX` | immutable、copyable shared metadata state；不拥有 tensor payload/provider artifact |
| G24 | one prepared artifact plan | `FIX` | compile 按 builder topological order prepare all nodes，形成 immutable `Executable` |
| G25 | no intermediate storage | `BORROW` | 使用 Mem `LifetimePlanner/LifetimePlan/PlannedBuffers`；每个 Invocation 独占实例，不复制 arena 算法 |
| G26 | fixed Invocation bindings | `FIX` | create 时分配完整 binding/workspace metadata；bind 做不分配的 external alias 校验，reset/submit 不增长、不查 hash、不分配 |
| G27 | branch/shared value DAG | `FIX` | node 只能读取此前创建的 ValueId，因此 cycle 不可表示；共享值和多 published output 合法 |
| G28 | public `graph_plan`/scheduler | `REJECT` | plan 是 Executable 内部 factor；Graph 不创建线程池、task graph 或 model scheduler |
| G29 | provider partition/fusion/serialization | `DEFER` | CPU multi-op 先闭环；后续 region/provider 卡扩 CompileReport，不改变 owner 分层 |

### Owner 与失败矩阵

| 对象 | 拥有 | 借用/不拥有 | 失败和线程合同 |
|---|---|---|---|
| `GraphBuilder` | 可增长 metadata 与 allocator ref | input payload、Context | single-owner；mutator OOM/invalid ID/shape 失败保持逻辑内容不变 |
| `Graph` | immutable value/node/edge/output metadata | payload、Context/provider | copyable read-only；跨线程共享；查询 out-of-range 可恢复 |
| `compile()` | 临时 validate/liveness requests | Graph + Context | O(V+E)；任一 prepare/planner/OOM 失败不发布半个 Executable |
| `Executable` | prepared operations、binding schema、immutable lifetime plan、Context retention | per-run payload/workspace | copyable read-only；跨线程共享；报告精确 workspace/copy/sync/fallback |
| `Invocation` | 每次运行 binding slots 与 `PlannedBuffers` | caller input/output views、Queue | move-only/single-owner；不同 Invocation 可并发；create 可 OOM，steady bind/reset/submit 不分配 |
| `Completion` | Compute/provider terminal state | bound payload 至终态 | CPU v2 同步；未来异步不得削弱 no-reset-before-terminal 合同 |

中间 value 的 lifetime 为 producer node 到最后 consumer（半开区间）；published graph output 永远由 caller
绑定，不放入内部 workspace。v2 操作均禁止任意 partial alias，因此 planner 只做 non-alias reusable slots；
external bindings 在 bind 时按真实 backing/offset 检查，只允许到达最后一次读取的 RMSNorm 第一输入与其
output 精确同址。未来原生 in-place op 必须显式产生 alias request，不能靠地址偶合。

### 自然 C++ journey

```cpp
auto builder = GraphBuilder::create().value();
auto x = builder.input(x_desc).value();
auto norm_w = builder.input(norm_weight_desc).value();
auto gate_up_w = builder.input(gate_up_weight_desc).value();
auto norm = builder.add(RmsNorm{.epsilon = 1e-5F}, x, norm_w).value();
auto gate_up = builder.add(MatMul{}, norm, gate_up_w).value();
auto activated = builder.add(SiluAndMul{}, gate_up).value();
builder.output(activated).value();
auto graph = std::move(builder).finish().value();
auto compiled = compile(graph, context).value();
auto invocation = compiled.executable.create_invocation().value();
invocation.bind(x, x_view).value();
invocation.bind(norm_w, norm_weight_view).value();
invocation.bind(gate_up_w, gate_up_weight_view).value();
invocation.bind(activated, output_view).value();
auto done = invocation.submit(queue).value();
```

正交测试构造一份共享 input 的两分支 DAG 并发布两个 output；这保证 IR 不是为三节点 benchmark 定制。

### P0/P1 preflight

- compile 逐 node 恰好一次 shape validation/prepare，liveness 与 plan 为 O(V+E)，可在冷路径有界分配；
- create-invocation 允许 workspace/binding allocation；bind 的 typed ordinal lookup 为 O(1)，为保证跨节点
  alias 正确性还会与已绑定 external output/input 做比较，完整绑定最坏
  O(inputs*outputs+outputs^2) 且零 allocation；submit 为 O(V)，每节点一次预绑定调用且零
  route search/JIT/packing/metadata allocation；
- P1 使用完全相同的 PreparedOperation 类型、TensorView 描述、Queue、算术路线与 completion boundary，
  对比 Graph Invocation 和手写 direct sequence；Graph 依法拥有隐藏的 per-Invocation `PlannedBuffers`，
  direct consumer 依法拥有等价的两个预分配中间 Buffer，因此不伪称同一地址/公开 owner，而是把分配、
  首次触碰、compile/create/bind 全部移到测量外并交叉 allocation order，禁止苹果/橘子对比；
- Graph overhead 与算子 leaf throughput 分开报告，不能用大型 MatMul 淹没 per-node tax。

## Audit verdict

G20--G27 可由已关闭的 Compute v2 合同、Mem lifetime planner、Container SmallVector 与 Util intrusive
refcount 直接实现；这些是已有模块的正确复用，不需要新公开模块。允许 Graph 进入 `AUDITED`。若实现需要
公开 graph_plan、隐藏 scheduler、per-submit allocation 或 provider-specific IR，立即停止并另立依赖卡。

## 实现 checkpoint（2026-09-12）

- 固定 one-op arrays 已替换为 allocator-owned value/node/edge/input/output records；Graph copy 共享不可变
  intrusive state，ID generation 与拓扑约束在 builder 边界执行；
- `compile()` 按拓扑逐 node prepare，并把未发布的 node outputs 转为 Mem `LifetimeRequest`；Executable
  持有 immutable prepared/lifetime plan，每个 Invocation 独占 `PlannedBuffers` 与预定长 bindings；
- steady `bind/reset/submit` 不增长 metadata；submit 不读取 logical Graph、不重新推 shape/选 route，并对
  每个 node 恰好调用一次 PreparedOperation；
- Windows selected Graph library、public-header self-containment 与 Graph tests 已通过；覆盖真实三节点片段、
  分支/共享值/多输出、跨 builder ID、shape failure recovery、allocator OOM/finish retry、workspace/report、
  reset/reuse、并发 Invocation、owner lifetime、零尺寸 workspace、发布后继续消费的中间值、跨 Context Queue
  和 steady metadata allocation；
- Windows `graph` selected-product smoke 的 source embedding、重定位 installed shared、重定位 installed static
  consumer 均编译并实际运行；shared 扫描 7 个 HYLE libraries / 754 exports 通过；产品 API/算法候选固定为
  `39bbb3be235317f4b0e70736a4d1f0112f2c359a`；
- v1 fixed-shape benchmark target 已退出活动 bench，冻结合同迁到历史 evidence。v2 P1、Linux/macOS、
  sanitizer 与终局 red-team 尚未执行，因此本 checkpoint 不构成 `CLOSED`。

### 红队纠正 checkpoint（2026-09-12）

- 公开 node 查询已从裸 `size_t` 统一为 generation-aware `NodeId`，并补 `producer(ValueId)`；避免一边声明
  typed identity、一边仍让跨 Graph 下标成为隐式身份；
- 补齐整图 external alias 检查：逐 op 检查不足以阻止 output 提前覆盖仍被后续 node 读取的 graph input。
  现在只放行 lifetime 已结束的 RMSNorm 精确原地复用，partial/output-output/仍存活 input overlap 均拒绝；
- Windows/Linux/两台 Apple Silicon 的 Compute/Graph public-header 与功能测试已通过；WSL ASan、UBSan、
  TSan 的定向 Compute/Graph 测试已通过。上述红队产品纠正发生在最终 candidate 之前，需对修正后的字节
  重新跑定向候选门；P1 仍未执行。

### Provider run-state follow-up（2026-09-12）

[`compute-provider-onednn-v1`](compute-provider-onednn-v1.md) 证明第三方 prepared artifact 可共享，但
stream/memory handles/scratchpad 不能放在它或 `Executable` 中。Graph 的既有 owner 分层不变：plan node
保存 shared `PreparedOperation`；每个 Graph `Invocation` 创建对应的 Compute `OperationInvocation`，并把
其 scratchpad request 与 tensor intermediates 一起交给 Mem LifetimePlanner。steady submit 仍为每 node
一次已裁决调用，不搜索 provider、不创建 descriptor、不分配；CompileReport 对混合 provider如实报告。

### Identity exhaustion follow-up（2026-09-12）

红队复核发现 node ordinal 与 value ordinal 原先共用 `UINT32_MAX` 上界判定，但 node producer 字段把
`UINT32_MAX` 保留为“没有 producer”的哨兵。现在二者使用独立判定：value identity 仍可使用完整
`uint32_t` 空间，node identity 在产生保留值之前返回 `ResourceExhausted`；编译期边界断言无需构造数十亿
节点即可锁住该不变量。修正不改变正常图的布局、复杂度或常态执行成本。

## Candidate checkpoint（2026-09-12）

红队纠正后的不可变 Graph v2 产品候选为
`df6981ad942c7e9f3620785af2a730079b6b1433`。它固定 G20--G27 的 dynamic typed DAG、generation-aware
ValueId/NodeId、immutable shared Graph/Executable、per-run Invocation、完整 external alias 检查、Mem
lifetime workspace 与 O(V+E) compile/submit 结构。当前 HEAD 相对该候选的 `include/hyle/graph` 与
`src/graph` 差异为空；后续 Compute provider、测试、证据、状态和文档没有改写 Graph 产品。

候选 LOC 自检为全库 `563,481` 行、自实现不含测试 `99,014` 行、Graph 实现 `1,414` 行。这只允许
进入 `CANDIDATE`：修正后候选的三平台 correctness/header、WSL sanitizers、source/static/shared、三节点
Graph 对同一 PreparedOperation direct sequence 的 P1 composition-tax、终局七问与红队仍须逐项签发；
旧 one-op v1 receipt 不继承到本轮。

## Multi-op P1 preflight（2026-09-12）

正式合同为 [`graph-multi-op-tax-p1-v1`](../../../../bench/contracts/graph-multi-op-tax-p1-v1.json)，
consumer shape provenance 固定在
[`consumer_shapes_v2.json`](../../../../bench/graph/consumer_shapes_v2.json)。两条产品 workload 是五个已审
推理来源共同拥有的 RMSNorm → transpose-view MatMul → packed SiluAndMul；`1x256x512` 保留 decode
形状，`4x64x128` 是让固定 Graph bookkeeping 不被大矩阵掩盖的正交 small-batch 形状。

### 同合同与时序账单

| 阶段 | Graph candidate | direct Compute peer | 计时 |
|---|---|---|---|
| Context/Queue、Buffer 初始化/首次触碰 | caller 创建并 warm | 同一 caller 创建并 warm | 不计 |
| prepare/Graph compile | 三个相同种类的 HYLE-native `PreparedOperation` 固定在 immutable plan | 三个相同种类的 `PreparedOperation` 固定在 direct route | 不计 |
| workspace/output | Invocation 创建时实例化 Mem lifetime workspace，caller output 独立持有 | caller 预分配两个等价中间 Buffer 和独立 output | 不计；交叉分配顺序 |
| bind | 四个 external value 一次绑定 | 三个 `OperationInvocation` 已持有，参数在 submit 传入 | 不计 |
| steady execute | `Invocation::submit` 扫描 3 inputs/1 output，顺序提交并等待 3 个 node，返回 terminal Completion | 按同一依赖顺序提交并等待同三个 operation | `CHARGED` |

`src/graph/executable.cc::Invocation::submit` 是 candidate 边界：只读取固定 binding arrays 和三个
`PlanNode`，每个 node 构造最多两个栈上 input view，调用一次已经创建的 `OperationInvocation::submit` 并
等待同一同步 Completion。`bench/graph/multi_op_tax_bench.cc::Fixture::run_direct` 是 peer 边界：对相同
Queue/input/weight descriptions 顺序调用 RMSNorm、MatMul、SiluAndMul 并逐次等待。两侧都在第三个
operation 完成后返回；没有把 direct 的错误、mapping 或 completion 成本漏到计时外。

### 静态成本账单

| 成本 | Graph 增量 | direct peer | 结论 |
|---|---|---|---|
| 复杂度 | `O(bindings + nodes)`，本合同 4+3 | `O(nodes)`，本合同 3 | Graph 承诺本身 |
| allocation / free | 0 | 0 | setup/warmup 后 steady path 均为零 |
| route search / JIT / packing | 0 | 0 | prepare/compile 已裁决 HYLE-native 路线 |
| atomic / lock / wake / syscall | 除三次相同 Compute submit 外为 0 | 三次相同 Compute submit | 无 Graph 新增同步机制 |
| copy | 最多两个 input `TensorView` 的栈上值复制/node | typed arguments 的栈上传递 | 不复制 tensor bytes |
| 常驻 metadata | immutable plan、binding mask、per-node invocation | 三个 direct invocation 与 caller workspace owner | 都在测量前固定 |

### 极短 diagnostic

Windows W1 P-core、MSVC Release/static/portable、LTO off 的 primary target 已成功编译并完成两条 workload
的 correctness warmup。candidate-first 快照中 Graph/direct 的 op/s ratio 约为 `1.049/1.055`；换成
peer-first 后约为 `0.978/1.000`。单次 nanobench 快照存在明显顺序噪声，但没有发现固定超过 5% 的反向
机制差异；因此不依据该点值改产品，并进入预声明的 8 strata、`n=15` 正式测量。placement 与 allocation
order 必须在正式合同中同时覆盖，若 instrument controls 或 5% non-inferiority 门失败则停止，不追加姿势
寻找 PASS。
