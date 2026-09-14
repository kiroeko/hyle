# Graph v9：foundation route、shape 与 ownership 闭环

## Closure card

- component / owner：`graph` / Graph；round=`graph-v9-foundation-routing`；
- baseline commit：`49d32deacf80fbc046e53be64de47042144edc5a`；前轮记录为
  [graph-v8](graph-v8-explicit-allocator-sort.md)；
- trigger 与证据：[路由审计](compute-graph-route-unification-audit.md)证明当前 Graph 只编译 exact point、region
  只能表达连续 insertion-order range、leaf/region 分池选择、常量一律长期借用，无法承载已接受的推理
  foundation；
- 正常 consumer 任务：引擎编译一个 exact 或 bounded/guarded DAG，在运行前绑定 concrete shape；Graph 在
  target/storage/lane、leaf/region、copy/sync/workspace 的完整边界上选择一次并稳定重放；
- 主产品纵切：动态 token 的 normalization→matmul→gated activation，Native leaf 与 oneDNN region 竞争；
  反偏置为 exact 单节点 CPU Graph；
- 跨模块 journey：Compute/provider 拥有 leaf/region artifact，Mem 拥有 storage/workspace，Graph 唯一拥有
  node-set boundary、placement/liveness/selection/final Completion；模型 cache/batch/KV 策略仍由引擎拥有；
- scope：可收缩 node-set region、exact+bounded/guarded shape domain、leaf/region 同池有界选择、完整结构
  成本与 qualified profile、事务化 reselect、逐 constant borrow/materialize、可解释 shortlist/report，以及
  bounded general route 与 exact capture variant 共存；
- non-goals：第三个 public Plan、模型/采样/KV/serving、跨 provider graph tuner、任意 DAG 全局最优、横向
  填算子或平台、submit 时 JIT/search/capture；
- 直接依赖 / 消费者：依赖 Core/Error/Container/Mem/Compute/ComputeOps；推理/渲染/数值 C++ engine 消费；
- 性能 tier：`P0` 固定复杂度/成本结构，CPU one-op 与 gated MLP 再做一次 P1；
- 调查与纠正预算：node-set、shape-domain、selector/ownership 各一轮表示实验和一轮修正；一次终局红队；
- 预声明验证：见 G90--G97；Windows/Linux/两台 Apple、sanitizer、source/static/shared、P0、CPU P1、docs；
- stop：任何表示若要求执行用户输入来 compile、热路径搜索/JIT/metadata allocation、隐式全局 cache，或
  provider 接纳工作后静默 rerun，立即退回 `SCOPED/BLOCKED`。

## AUDITED 结论与有限改动集

| ID | 问题 | 允许改动与验收 |
|---|---|---|
| G90 | region 是 begin/count interval | 改为 nonempty unique node-ID set + explicit boundary；验证“离开后重入”、重叠、收缩成环与预算反例；interval 仅 fast path |
| G91 | Executable 只表示 exact point | `CompileOptions` 表达 closed exact/bounded domain；Invocation 在 work acceptance 前绑定 concrete shape；越界/multiple/overflow 拒绝，submit 不 JIT |
| G92 | bounded/general 与 exact tuned/capture 无组合 | immutable Executable 可含有限互斥 exact variants + 完整 general fallback；capture key 精确绑定 shape/storage/queue，recipe 与 Invocation state 分离 |
| G93 | leaf、region、placement 分池且成本不全 | 建立有界统一候选/计划选择：先硬约束，再完整 boundary cost/profile，再稳定 tie-break；不宣称一般 DAG 最优 |
| G94 | prepare 后事实变化无法合法重选 | 只对 unsupported 或完整 rollback 的 local decline 在硬 retry budget 内重选；合同上界破坏和真实错误终止 compile |
| G95 | 常量总是借到 Executable 终态 | 调用方逐 constant 选 borrow/materialize；artifact 成功拥有 bytes 后 materialized source 可释放；逐项 report 与 OOM rollback |
| G96 | report 只描述胜出项 | 有界记录 shortlist identity、拒绝原因、成本/evidence 来源、copy/sync/workspace；不复制无限日志 |
| G97 | 旧 exact/liveness/replay 可能漂移 | 保留三轴资源、happens-before、alias/workspace、Completion 与 exact specialization identity；单节点和现有 Graph tests 全回放 |

### API / usability freeze

表示实验只可裁决 node-set 存储（range/small-vector/bitset）、shape guard/variant table 和 bounded search 的内部
结构及默认硬预算；不能改变上述 owner/lifetime。普通用户只需声明 domain、常量策略和可选 placement/profile，
然后创建 Invocation、绑定 concrete tensors/shape、显式 prepare replay 并 submit。未提供 profile 时按成熟
provider 顺序稳定回退，不用微基准魔法分数。

## 七问与当前裁决

第三方只经独立 provider；四个真实引擎及 ONNX Runtime/oneDNN/TensorRT 给出准入；复用全部已关闭下层；
复杂度按候选/预算显式有界，P1 只在接口固定后运行；FFI N/A；三交付必须验证 provider factor；guide 必须
给 exact、bounded、borrow/materialize、fallback/capture recipe。当前为 `AUDITED`，只授权 G90--G97。

