# Graph v3：常量、多输出与 provider-neutral region 审计

## Closure card

- component / owner：`graph` / Graph；round=`graph-v3-constants-regions`；
- baseline commit：`97af06e66bc0f47ac64a32d165cd8f36cef28c4e`；上一轮已关闭的 CPU 多节点
  合同与证据保存在 [`graph-v2-multi-op-transformer`](graph-v2-multi-op-transformer.md)，本轮不回写
  该历史记录；
- trigger 与证据：v2 公开 IR 只能表示 graph input 和单一 node output，`GraphBuilder::add()` 只返回一个
  `ValueId`，图内没有 immutable constant、alias/effect 或 provider region。新登记的 `compute_ops` 必须
  表示 Top-K 等多输出 operation；ggml `e91ded11` 的 graph/backend split、llama.cpp `e71b8051` 的常量
  权重与 backend scheduler，以及 vLLM `51da0ca6`、SGLang `dcebe8c4`、ds4 `f62ca29a` 的真实推理图
  都要求这些合同。这是新 consumer 事实证明的能力缺口，不是 benchmark 或“再审一遍”；
- 正常 consumer 任务：C++ 用户把 caller-owned inputs、immutable weights/constants 和 typed operations
  构造成多节点 DAG，冷路径 compile 为 provider regions、copy/sync 与 workspace 明确的 immutable
  `Executable`，为每次请求创建独占 `Invocation`，绑定动态输入/输出并重复执行；steady path 不重新分区、
  不隐藏 JIT/packing/metadata allocation；
- 主产品纵切 / 具名来源：量化 decoder 中常量权重参与的 norm → QKV linear → RoPE → attention →
  residual → gated MLP，至少跨一个 HYLE-native/oneDNN 或后续 ggml region，并包含 Top-K 的 values +
  indices 多输出；
- 正交反偏置纵切：图像/数值 DAG 使用常量 kernel、分支、concat/reduction 和一个 resource-writing custom
  node，证明 Graph 不是 decoder 顺序表；renderer 的 raster/present 与资源生命周期仍留在消费方；
- 跨模块 public journey 与 owner：`compute_ops` 拥有 Tensor/Operation schema、output arity、alias/effect；
  Compute 拥有 backend/provider、artifact、queue/completion/storage；Graph 拥有 value topology、constant role、
  region partition、placement/liveness/workspace/copy/sync plan 和每次运行 bindings；模型、KV、batching、
  sampling、serving 与通用 task scheduling 不下沉；
- scope：审计并冻结 immutable constant、零/多输出 node、alias/effect/control dependency、provider-neutral
  region/partition 与 CompileReport 合同；对现有 Graph/GraphBuilder/Executable/Invocation 逐入口做
  `KEEP/MOVE/REPLACE/DELETE`；审计 Graph IR、plan 与 compiler 是否需要独立内部 factor，只实现审计后有限
  `FIX` 集；
- non-goals：模型文件/权重加载、KV/page table、sampling policy、dynamic batching、任意控制流、训练/autograd、
  distributed collectives、具体 ggml/CUDA/Metal/Vulkan/HIP provider、Graph 自建线程池、Native FFI、未有
  consumer 的通用 IR serialization；
- 直接依赖 / 直接消费者：候选 Graph 公开消费 Core/Error/`compute_ops`/Compute/Mem；Container 与
  Util-core 只在真实 metadata 实现需要时作为私有边登记。直接消费者为 inference engine、direct Graph、
  ggml/oneDNN region 和后续 GPU provider；Compute 不反向依赖 Graph；
- 性能 tier 与原因：先做 `P0`，compile/partition 允许显式冷路径 allocation，目标复杂度为 O(V+E+R)；
  steady invocation 为 O(bindings+regions+copies)，不按节点 route search/JIT/packing/metadata allocation。
  immutable candidate 才用 `P1` 比较完全相同 prepared artifacts、queue/storage/completion 的 direct sequence；
- 适用机器 / 仪器 / material margin：Windows、WSL Linux、两台 Apple 先验证 CPU fallback、IR/owner、
  headers 与三交付；具体 GPU 由 backend/provider 卡资格化。P1 使用统一仪器、双顺序与固定 5% 非劣门；
- 调查与纠正预算：四类推理来源各一次定向 graph/region 扫描；ggml backend scheduler、oneDNN Graph、
  一个成熟 execution-plan runtime 各一次合同扫描；IR/API 与 partition 各最多两轮；每个冻结 FIX 最多两轮；
- 预声明验证：constant ownership/lifetime、zero/multi-output、branch/shared output、cross-builder/invalid IDs、
  alias/effect ordering、shape/dtype/layout mismatch、unsupported partition/fallback、copy/sync edges、workspace
  overflow/OOM/rollback、concurrent Invocation、close/drain/late completion、逐头、sanitizers、四机 CPU、
  source/static/shared、factor selection、P0 账单、必要 P1 与终局红队；
- stop / blocker 条件：若 output/alias/effect 无法由 `compute_ops` 提供，先完成其 schema 而不在 Graph 复制；
  若 region artifact/queue/storage 合同必须由 Compute runtime 提供，先完成 Compute v4；若自然边界确需
  独立内部 factor，先登记 identity；不得用单节点 enum stub、隐藏 host copy 或全局 provider registry 冒充完成。

## SCOPED 待审计项

1. constant 是借用、shared owner 还是 compile-time packed artifact，以及生命周期和 fingerprint 的边界；
2. 一个 node 的 output-count 查询、ValueId 创建、published output 与零输出 effect node 的自然 C++ 形状；
3. view alias、in-place、resource write 和普通 data edge 如何共同形成无隐含副作用的执行顺序；
4. provider capability、region proposal/compile、fallback、copy/sync 与 CompileReport 的唯一 owner；
5. dynamic dimensions 是 v3 必需合同还是应以具名 trigger 留到后续，不得无 consumer 地引入符号代数；
6. Graph IR、compiler 与 immutable execution plan 是否满足独立质因数标准；公开 façade 是否仍只叫 Graph；
7. 现有入口逐项 `KEEP/MOVE/REPLACE/DELETE`，以及 source/static/shared 与 factor-only 构建投影。

AUDITED 前不改 Graph 产品代码、公开头、依赖、benchmark 或 provider 实现。
