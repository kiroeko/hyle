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
- 直接依赖：公开 `core/error/compute`；内部只在审计证明语义吻合时使用 `mem` lifetime planning、
  `container`/`util_core` metadata primitive，不能复制 allocator/container；Compute 永远不反向依赖 Graph；
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
