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

## AUDITED 结论（2026-09-12）

### 来源与边界

ggml `e91ded11bdcd` 的 logical cgraph/backend split/gallocr、oneDNN `6b7f1fbbb18e` 的 partition + compiled
partition，以及 llama.cpp/vLLM/SGLang/ds4 的 immutable model artifacts 与 per-request state 共同确认：
Graph 必须拥有 logical constant、多输出、provider region 和显式 plan step，但不应吸收模型 scheduler、
KV policy 或 backend native handles。v3 冻结为**单 Context/Device、一个或多个 provider、一个 ordered Queue**；
跨设备 placement/transfer 在首个 GPU backend 有真资源后另卡审计，不能用 host-only伪接口预演。

### 入口与 IR ledger

| ID | 当前形状 | 裁决 | v3 冻结合同 |
|---|---|---|---|
| G30 | `ValueKind` 只有 input/node output | `FIX` | 增加 logical constant role；Graph 仍不持有 payload |
| G31 | `GraphBuilder::add()` 返回单 ValueId | `REPLACE` | general `add_node()` 返回 NodeId，随后 O(1) `result(node,index)`；single-output typed convenience 返回 ValueId |
| G32 | node 固定最多 2 input/1 output | `REPLACE` | allocator-owned edge/output ranges；arity 来自 compute_ops，compile 后固定且 steady path不增长 |
| G33 | `output(ValueId)` 只接受 node output | `RENAME/FIX` | `publish(ValueId)` 可发布合法 input/constant/node result；避免与 node result 概念混名 |
| G34 | 无 constant payload seam | `FIX` | Builder 只声明 `constant(TensorDesc)`；CompileOptions 传精确 ConstantBinding，Executable 按 report 明示 borrow 或 packed owner |
| G35 | alias 由 RMSNorm 特判 | `REPLACE` | 使用 operation traits + liveness；view alias/in-place candidate/禁止 overlap 统一规划，provider 只能收窄 |
| G36 | 无 side-effect/control edge | `FIX` | pure 为默认；effect/resource-write node 必须 `depends_on()` 或数据依赖，缺少顺序即 finish/compile 失败 |
| G37 | 每 node 独立 prepare | `FIX` | 先查询 provider region proposals，再以确定性 cost 选择不重叠 regions；剩余 node 回落单-op artifact |
| G38 | submit 每节点 host wait | `DELETE` | 同一 ordered Queue 连续提交 plan steps，只返回代表整个 sequence 的最终 Completion |
| G39 | CompileReport 只有一个 ProviderKind | `REPLACE` | copyable report view 枚举 region/provider/version、node range、workspace、packed/borrowed constants、copy/sync/fallback |
| G40 | compile 只接 Context，无选项 | `FIX` | `CompileOptions` 包含 constant bindings、fallback/optimization policy 和显式预算；provider 来源仍是 Context 冻结序列 |
| G41 | Executable/Invocation owner 分离 | `KEEP/FIX` | Executable 保活 artifacts/packed constants；Invocation 独占 mutable bindings/workspace并禁止 in-flight reset/reuse |

ConstantBinding 默认是显式 borrow，source storage 必须活过 Executable 与其全部 in-flight completions；若
provider 打包/复制，report 必须给出 bytes、owner 和 source borrow 是否解除。Graph 不用 `shared_ptr<void>`
掩盖所有权，也不把权重伪装成每次 Invocation input。

### Region 与 plan 合同

- Compute Provider 保持单一显式对象。Graph 通过该对象的局部、具名 region-extension query 取得编译接口；
  不使用进程注册表，Compute core 不包含 Graph 类型，也不反向依赖 Graph；
- provider 对 immutable Graph view 提交候选 region、支持约束与可比较 cost facts；Graph 决定 partition，
  provider 只编译获选 region。`supports()` 不等于必选，copy/repack/sync/packing/JIT 均计入冷/热成本；
- v3 计划 step 只有 `Region`、`Copy/Repack`、`Barrier` 三类。每个 Region 是一个预绑定 submit callable；
  steady path 不遍历原始 DAG、不查 provider、不推 shape、不 JIT；
- deterministic fallback 是拓扑顺序单-op regions。compile/partition 目标 O(P*(V+E)+Q log Q)，其中 P 是
  显式 provider 数、Q 是有限 proposals；每 provider 和全局 proposal 数都有 CompileOptions 预算；
- Graph v3 不拆 `graph_ir/graph_compile/graph_plan` 独立 component：目前只有同一 Graph product 消费，代码/
  依赖隔离收益尚不足以抵消三个状态 owner。职责仍在源码目录内分层；首次重型 compiler 依赖或独立
  plan consumer 是新的 factorization trigger。

### 正确性、性能和后续边界

v3 FIX 为 G30--G41；symbolic dimension、跨 Device QueueSet、serialization/cache ABI 与通用 control flow
明确不进入本候选。P0：validate/infer/liveness O(V+E)，constant binding O(C)，Invocation bind O(B) 总表后
每次 O(1)，submit O(R+X)（regions + boundary steps）且零 metadata allocation。候选 P1 只比较：同一
prepared artifacts 的 direct sequence、单-op fallback plan、以及首个真实 ggml/oneDNN multi-op region；
compile/create/bind 均在 steady timing 外，同 storage/layout/threading/completion，不做苹果对橘子。

## Runtime compatibility checkpoint（2026-09-12）

Graph v2 的现有单节点 executable 已迁移到开放 ProviderId、显式 CPU Context 与 `mixed_providers` report
语义；Windows Graph/oneDNN consumer 和 source package smoke 通过。这只是解除 Compute v4 的编译阻塞，
没有实现 G30--G41：constant/multi-output/general arity、effect/control edge、region partition、完整 report、
workspace/liveness 与最终 Completion sequence 仍是本卡后续工作。因此 Graph 保持 `AUDITED`，不得把兼容
迁移或旧单节点测试记作 v3 进度完成。

## IR、constant 与 alias implementation checkpoint（2026-09-13）

- G30--G34 已落地：Graph 有独立 constant role、通用 `add_node()`/`result()` 零至多输出形状、任意合法
  input arity、可发布任意 value，以及 compile-time 精确 constant binding。当前 artifact 明确借用 payload，
  report 的 packed 数为零；Invocation reset 不清 constant；
- G36 的结构合同已落地：`depends_on()` 只建立指向更早 node 的显式控制边，finish 以 O(V+E) 分组去重；
  external-side-effect nodes 必须经 data/control dependency 形成链，缺失顺序不能靠 builder 插入次序蒙混；
- G35 已移除 RMSNorm 特判。Graph 先用 operation traits 给 alias 上界，再查询具体 PreparedOperation 对当前
  provider/shape/layout 的 refinement；Native 广播/strided elementwise 不再被误判为可原地，dense Unary、
  normalization 等仍可在最后一次读取时复用 Mem lifetime backing。View 携带 byte offset，在外部 storage
  或内部 workspace 上零物化传播；当前公开 view 若根在不可见的 private workspace，compile 明确返回
  `Unimplemented`，等待 G41 统一裁决 runtime-owned output，而不是隐藏 copy；
- Windows 的 Compute/Graph/Native 11 项定向测试已覆盖 constant lifetime、TopK 两输出、零输出 effect、
  分支、内部 exact in-place、slice subrange 及广播 alias 反例。G37 region partition、G38 最终 sequence
  completion、G39 完整逐 region report、G40 policy/budget 与 G41 in-flight/runtime-owned lifetime 尚未完成。

这仍是 `AUDITED` 中间候选，不能签发 Graph v3 `CLOSED`；下一实现边界是 G37，不继续用 CPU 算子数量掩盖
Graph runtime/plan 缺口。

## G37 provider-region implementation checkpoint（2026-09-13）

- Graph 现在从 Context 的显式 provider 序列查询精确版本的、对象局部 region extension。provider 对
  immutable Graph 提交**连续拓扑区间**候选；Graph 以 `estimated_net_savings_ns`、覆盖 node 数和稳定
  provider/route 顺序做确定性的 weighted-interval selection，复杂度为 O(P*(V+E)+Q log Q)；
- `CompileOptions::max_region_proposals` 是全局硬预算。预算耗尽、非法区间、零 dispatch 或 malformed
  artifact 都使 compile 显式失败，不静默截断或改变 fallback；保守策略只接受 provider 能声明正净收益的
  proposal，disabled 策略完全不查询 region；每 provider 的独立预算仍归 G40；
- 获选区间只编译一次 `ProviderRegionPrepared`，未获选 node 继续走明确的单-op fallback。内部 plan 已从
  node 数组改成 operation/region step；region 内部 value 不由 Graph 重复分配，边界 value 与每 Invocation
  scratchpad 仍统一进入 Mem lifetime plan；
- compiled region 的 provider Context owner、prepared artifact、per-Invocation state 与 workspace 已分离。
  红队发现并修正 workspace 早于 invocation state 析构的问题；provider compile/create 的部分构造失败均
  对称 rollback；Invocation 即使在调用方释放 Executable handle 后仍保活 plan/artifact；
- Windows shared 纵切以两个 fused negation node 验证单 region、region+Native fallback、重叠候选选择、
  policy disabled、proposal budget exhaustion、失败回滚与 owner lifetime；现有 oneDNN 单-op Graph 路线也
  通过回归；Graph 的 source embedding、installed/relocated shared 与 static consumer 也全部通过。该测试
  provider 只证明中立合同，不冒充真实 oneDNN/ggml multi-op region 的性能资格。

G37 的中立 partition/artifact 机制已实现，但 Graph 仍是 `AUDITED`：G38 ordered sequence Completion、G39
逐 step/region report、G40 policy/per-provider budget，以及 G41 in-flight/reuse/runtime-owned output 尚未完成；
首个真实 oneDNN/ggml region 和公平 P1 也仍是终局必要证据。

## G38--G41 runtime/diagnostic implementation checkpoint（2026-09-13）

- G38：Invocation 先解析全部 view/alias 与 step binding，再取得同一 Queue 的一个连续 submission
  sequence；所有 materializing step 只做 non-blocking poll 以捕获同步终态，不再逐节点 wait，最终只返回
  覆盖完整已接纳 prefix 的 Completion。后段同步拒绝会附着到已接纳前缀，不能把异步工作伪装成普通
  `Result` 失败后提前释放；
- G39：`CompileReport` 现在是独立 copyable owner，不借用也不保活重型 Executable artifact。O(1)
  `step(index)` 返回 kind、provider/version、node range、workspace、persistent bytes、packed/borrowed
  constants、copy/sync、fallback reason、dispatch count 与 host/private-thread/concurrent-invocation facts；
  聚合统计仍为 O(1)，非物化 View 如实报告零 dispatch；
- G40：constant bindings、conservative/disabled region policy、allow/reject operation fallback，以及全局和
  per-provider proposal 硬预算均进入 `CompileOptions`。任一 proposal writer 超限显式
  `ResourceExhausted`，reject-fallback 下未被 region 覆盖的节点显式 `Unimplemented`；
- G41：Invocation 有跨 terminal 的 in-flight 状态，bind/reset/resubmit 在运行中失败；最终 Completion
  保活 Invocation、workspace 与 provider run state，pending handle 被丢弃也延迟到 backend 真终态释放。
  发布的 required View 若根在 private workspace，不再 compile 失败：该 root 以 Mem `kOutput` placement
  保留，`output_ownership()` 披露 Invocation owner，成功终态后由 `output()` 借出；若根是 input/constant，
  则披露 source alias 并直接复用已有 source binding，不要求用户重复绑定同一切片。调用方要求自有目的地
  时必须在图中显式增加 Copy。实现过程中同时修正 non-zero slice alias 被错误强制 64-byte 对齐的问题；
- Windows tests 覆盖两段模拟 async region 无 host wait、abandoned completion、tail reject、两级预算、
  reject fallback、report 独立 lifetime/逐 step facts、source-alias/运行时-owned slice output 和 reset/reuse。
  Graph 的 source embedding、installed/relocated shared 与 static package consumer 全部通过；shared export
  hygiene 扫描 11 个 HYLE libraries/913 exports 通过。

G30--G41 的通用 CPU/runtime 形状至此全部进入实现候选，但 Graph 继续保持 `AUDITED`。终局仍需首个真实
oneDNN 或 ggml multi-op region、公平 direct-sequence P1、Linux/两台 Apple、sanitizer、逐头与最后红队；
在这些完成前不得用本 checkpoint 冒充 `CLOSED`，也不扩大 CPU 算子目录。
