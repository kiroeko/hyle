# Graph v6：动态 specialization、多设备 placement 与异步 plan 主干

## Closure card

- component / owner：`graph` / Graph；round=`graph-v6-runtime-specialization-placement`；
- baseline commit：`aca4fe7f043944dd12dad90eace92693b300b67c`；上一轮合法 terminal 为
  [`graph-v5-region-admission`](graph-v5-region-admission.md)；
- trigger 与证据：v5 terminal 明确没有签发真实异步 backend、跨设备 placement/copy/sync 或 symbolic
  specialization，并要求下一轮先走 Runtime/Graph 纵切。当前公开 `GraphBuilder` 只接收具体
  `compute::TensorDesc`，`compile()` 只借一个 `compute::Context`，`Invocation::submit()` 只接一个
  `compute::Queue`；`kCopy/kRepack/kBarrier` 只是 report 枚举，产品没有生成这些 step。W1-WIN 已确认
  RTX 4070、CUDA 13.3 与可执行 `nvcc`，因此可以用真异步设备合同验证，不再以 CPU fake provider 预演；
- 正常 consumer 任务：推理引擎把一个模型无关 transformer fragment 以 bounded symbolic batch/token
  dimensions 构图，按 decode/prefill shape signature 冷编译或命中 specialization，跨 CPU/GPU storage
  domain 显式 placement/copy/repack，在一组 caller-owned queues 上重复异步 replay；多个 Invocation 并发，
  每个 borrow、workspace、event 和 capture backing 活到最终 Completion 终态；
- 主产品纵切 / 具名来源：llama.cpp/ggml `e71b80510c84` 的 backend scheduler、reserve/alloc/reset、async
  tensor copy 与 event wait/record；vLLM `51da0ca66c80` 的 `BatchDescriptor`、full/piecewise CUDA graph、
  graph pool 和稳定地址；SGLang `dcebe8c4733a` 的 decode/extend capture/replay 与预分配 pool；ds4
  `f62ca29a3087` 的 stream-owned pool、capture-safe allocation 和 CUDA/Metal resource lifecycle；
- 正交反偏置纵切：renderer/codec 把 caller-owned CPU staging 与 GPU storage 交给同一 Graph，在两个
  device domains 之间执行 upload -> typed custom compute -> download，并把最终 Completion 交还原资源
  管线。它验证通用 resource/placement/copy/event 语义，而不引入模型、KV 或 batch policy；
- 跨模块 public journey 与每段 owner：`compute_ops` 拥有 concrete operation/Tensor schema；Compute
  backend 拥有 Device/Context/Queue、device storage、copy/event/native interop 和真实 Completion；Mem
  拥有 host/unified storage 与通用 lifetime packing；Graph 拥有 symbolic logical value、specialization、
  partition/placement、boundary materialization、per-domain liveness/workspace、Executable/Invocation 与
  CompileReport；Sched/Obs 仅经后续可选 bridge 组合等待、取消和观测；
- scope：审计并冻结 bounded symbolic dimension 与 concrete specialization 的自然 C++ 形状；多 Context/
  Device compile environment；node/value placement constraints 与确定性有界 partition；显式 copy/repack/
  barrier/event plan；per-domain liveness/alias/workspace；stable-address capture/create-invocation/replay
  生命周期；多 queue submit、late completion、cancel/drain；CompileReport 真值；现有 Graph 入口逐项
  `KEEP/MOVE/REPLACE/DELETE`；以及 `graph_ir/graph_compile/graph_plan` 是否满足质因数拆分条件；
- non-goals：批量填充 CPU/GPU 算子目录；模型文件、tokenizer、KV/prefix-cache、continuous batching、
  sampling、serving、renderer raster/present；任意 if/while/递归控制流；跨版本可移植 Executable 序列化；
  HIP/ROCm；Native FFI；用计时常数驱动 placement 或为 oneDNN microbenchmark 加特判；
- 直接依赖 / 直接消费者：当前 Graph 直接公开依赖 Core/Error/Compute/compute_ops/Mem，私有依赖
  Container/Util-core。候选可增加已合法登记的具体 backend factor 作为测试/产品 façade root，但 Graph
  通用 factor 不得直接依赖 CUDA/Metal/Vulkan/MLX 或具体 provider。直接消费者为 direct Graph 用户、
  CUDA/Apple/Vulkan graph slice 及未来推理引擎 runtime；
- 性能 tier 与原因：先做 `P0`，冻结 compile `O(V+E+provider/placement budget)`、specialization lookup、
  steady replay、allocation/atomic/lock/copy/sync/JIT/metadata 账单。只有 immutable candidate 与至少一条
  真异步 backend 成立后才做一次 `P1` 抽象非劣和 `P2` 同任务 replay/copy/sync 端到端；固定 5% 门，
  不把裸 kernel 与完整 Graph 比较；
- 适用机器 / 仪器 / material margin：Windows W1-WIN 负责 CUDA 真机与工程性能；WSL Linux 负责
  portable correctness/sanitizer/package；MacBook Air 与 Mac mini 负责 Apple portability，并在对应 backend
  卡就绪后签 Metal/MLX 真机。Vulkan/MoltenVK 和 NVIDIA/Apple provider 各由独立 factor card 资格化；
- 调查与纠正预算：四个推理来源各一次定向源码扫描；ggml scheduler、CUDA graph/event、Metal command
  buffer、Vulkan timeline、ONNX Runtime/TensorRT 或同级 graph runtime 各一次合同扫描；IR/API/owner 最多
  两轮；factor 拆分一次；每个冻结 FIX 最多两轮；candidate 后每个平台/交付/P1/P2 各一次；
- 预声明验证：symbol constraint satisfiable/overflow/zero/boundary、specialization identity/cache collision、
  placement required/preferred/unsupported、multi-device copy/repack/barrier/event ordering、partial compile
  rollback、per-domain workspace/OOM/alignment、constant pack/borrow、stable capture address、两个并发
  Invocation、in-flight reset/rebind/replay、late completion abandonment、cancel/device loss/close/drain、
  cross-context/queue mismatch、report truth、逐头、ASan/UBSan/TSan、四机、source/static/shared、精确 factor/
  product/offline/license、P0 及必要 P1/P2、终局红队；
- stop / blocker 条件：若某项语义必须由尚未登记的 backend/provider 拥有，先建立该独立 factor，不把
  vendor handle 塞进 Graph；若 symbolic/capture API 只有单个框架私有 consumer，先保持 provider-private；
  若跨设备 correctness 不能在真设备上验证，精确阻塞该 backend slice 而不伪造 Graph PASS；若有限审计
  无法得到自然 owner/API 或确定性有界算法，退回 `SCOPED/BLOCKED`，不得靠补算子或扩 benchmark 掩盖。

## SCOPED 待审计项

1. logical symbolic description 应由 Graph 单独拥有，还是需要 provider-neutral `compute_ops` prime factor；
   concrete `TensorDesc` 必须继续作为每个 compiled specialization 的物理合同，不能把未知维塞进 runtime view；
2. shape symbol、范围/整除/相等约束、binding-time value 与 specialization signature 怎样做到 typed、可哈希、
   overflow-safe，且不把模型 batch policy 下沉；
3. 普通 C++ 用户如何提供多个 Context/Device/Queue：`CompileEnvironment`、placement constraints 与 submit
   bindings 的 owner/lifetime 应是什么，不能使用 process-global backend registry；
4. provider capability、resource domain、layout、copy/repack/sync 与 compile/persistent/workspace 成本如何
   输入 deterministic bounded placement；unknown 成本如何 fail honest，而不是默认为免费；
5. copy/repack/barrier/event 是 Graph IR 显式 operation、compiler-emitted step，还是 backend command；每类
   failure、completion prefix、device-loss 和 report 事实由谁拥有；
6. 多设备 liveness 与 Mem `LifetimePlanner` 的边界：每个 resource domain 是否独立 packing，跨域 value
   何时产生新 storage root，Invocation/caller/pool 各自拥有哪些 backing；
7. `Executable` 绑定一个 concrete specialization 还是管理有界 specialization set；若 cache 留给消费方，
   HYLE 仍须提供哪些 stable signature/report API，才能自然承载 vLLM/SGLang batch dispatch；
8. CUDA/Metal/Vulkan capture 需要稳定地址和 per-invocation mutable state；capture 应在 compile、
   create-invocation 还是首次显式 prepare/capture 发生，如何避免 warm submit JIT/allocation；
9. 单 queue `Invocation::submit(queue)` 如何替换为不割裂 CPU 用户的多 queue 形状；ordered prefix、跨 queue
   event wait、cancel/drain 与 abandoned Completion 如何组合；
10. `graph_ir`、`graph_compile`、`graph_plan` 是否存在独立消费者、不同依赖或可并行维护价值。只有满足质因数
    判据才登记内部 component；公开产品仍只有 `hyle::graph::Graph/Executable/Invocation`；
11. 当前 `CompileReport` 缺哪些 specialization/device/domain/copy/event/capture/concurrency facts；复制 report
    不能保活重型 backend artifact，也不能把 unknown bytes/time 写成零；
12. 哪一组最小 probe 足以验证主干：已有 gated MLP region、一条 CPU<->CUDA copy + typed kernel + copy、
    一个 decode/prefill specialization replay；除此之外不得横向增加 operator catalog。

本卡只授权审计。进入 `AUDITED` 并冻结能力表、入口 ledger、API/usability 和有限 FIX 集前，不修改
`include/hyle/graph`、`src/graph`、Compute/backend 产品代码、公开构建投影或 benchmark。

## AUDITED 结论（2026-09-13）

### 真实 runtime 对账

- ggml `e91ded11bdcd` 的 scheduler 显式保存 backend 顺序、split、copy slots 与
  `events[backend][copy]`，reserve/alloc 与 compute_async 分离；它证明 Graph 必须拥有 placement、boundary
  storage 和 event plan，但具体 stream/event 仍由 backend 拥有；
- llama.cpp `e71b80510c84` 的模型 runtime 重复构图/执行并使用 ggml backend scheduler；vLLM
  `51da0ca66c80` 的 `BatchDescriptor`、full/piecewise CUDA graph 与 graph pool，以及 SGLang
  `dcebe8c4733a` 的 decode/extend capture/replay，都把 shape signature、稳定地址与每 lane mutable state
  分开；一个无界、进程全局 specialization cache 不是 Graph 基础能力；
- ds4 `f62ca29a3087` 的显式 stream、分层 device placement、stream-owned scratch 与 capture-safe allocation
  证明 workspace owner、queue affinity 和 capture 时点必须可见；不能在首次 warm submit 才偷偷分配；
- renderer/codec 纵切同样需要 external host/device binding、upload/compute/download、跨 queue dependency 和
  native interop，因此以下 API 不是 transformer 专用形状。

### 能力与形状审计

| 能力族 | 层级 | 冻结语义/边界 | 当前 | 动作 |
|---|---|---|---|---|
| bounded shape symbols | 原子机制 | graph-local `DimensionId`，positive min/max/multiple；复用同 ID 表达相等，只支持常量或单 symbol extent | MISSING | FIX |
| logical TensorSpec | 原子机制 | symbolic dense/blocked-dense；concrete TensorDesc 仍支持任意已签 layout；opaque/任意 symbolic stride 必须 provider-typed | MISSING | FIX |
| concrete specialization | 复合功能 | compile 时 exact-once 绑定全部 symbol并 overflow-safe 生成 TensorDesc/稳定 signature；一个 Executable 只绑定一个 signature | MISSING | FIX |
| explicit execution domains | 原子机制 | borrowed Context + workspace MemoryResourceRef 的有序 span；无全局 registry，Executable 通过 artifacts/extensions 保活 Context | MISSING | FIX |
| external resource contract | 原子机制 | input/output/constant 冷路径声明 domain/resource；Invocation 可换 allocation，不可换不兼容 resource domain | MISSING | FIX |
| deterministic placement | 复合功能 | required/preferred constraint、input locality、environment order；逐候选 prepare，Unimplemented 才 fallback，硬预算内确定 | MISSING | FIX |
| provider region partition | 复合功能 | 在 concrete specialization 与单 domain 连续区间上复用 v5 deterministic admission；不跨 domain 猜 vendor region | COMPLETE | KEEP/FIX |
| explicit boundary plan | 复合功能 | copy/repack/device-wait/host-barrier 都是 report/plan step；copy 用 Compute byte transfer，repack 用明确 operation/provider | MISSING | FIX |
| per-domain liveness/alias | 复合功能 | 每 `(logical value, domain)` 是物理 instance；各 domain 独立 Mem LifetimePlan，alias 不跨 domain/resource | PARTIAL | FIX |
| immutable Executable / exclusive Invocation | 复合功能 | Executable 固定 signature/domains/artifacts；Invocation 独占 bindings、workspace、events/capture state | PARTIAL | FIX |
| multi-queue replay | 复合功能 | submit 接 domain->Queue bindings；每 queue segment 一次 prefix Completion，兼容 device wait，显式 host barrier fallback | MISSING | FIX |
| stable-address capture | 复合功能 | bindings/workspace/queues 完整后显式 prepare replay；prefer/require capture 可区分，地址变化需重 prepare | MISSING | FIX |
| terminal aggregation | 复合功能 | 最终 Completion 保活全部 segment prefix/Invocation，聚合 late error/cancel；abandon 不 host-wait | MISSING | FIX |
| arbitrary control flow/cache | 应用策略 | recurrent/speculative/continuous batching 由引擎选择并重复 Executable；无 process-global specialization cache | OUT_OF_SCOPE | REJECT |

这组能力已覆盖主纵切和正交纵切；算子 breadth、KV manager、batch admission、sampling 和模型循环没有
被伪装成 Graph 缺口。当前 `kCopy/kRepack/kBarrier` 只占枚举不算实现，必须由 G60 的真实 emitted step
替换；当前 concrete/single-context/single-queue overload 只能作为自然 convenience，不能冒充完整路线。

### 冻结公开形状

```cpp
auto batch = HYLE_TRY(builder.dimension({.minimum = 1, .maximum = 64,
                                         .multiple_of = 1}));
auto tokens = HYLE_TRY(builder.dimension({.minimum = 1, .maximum = 4096,
                                          .multiple_of = 1}));
auto input = HYLE_TRY(builder.input(
    TensorSpec::dense(DataType::kFloat16, {Dimension(batch), Dimension(tokens),
                                           Dimension::fixed(hidden)})));

ExecutionDomain domains[]{
    {.context = &cpu_context, .workspace_resource = host_resource},
    {.context = &cuda_context, .workspace_resource = device_resource},
};
ShapeBinding shape[]{ {batch, 1}, {tokens, decode_tokens} };
ExternalResourceContract external[]{
    {.value = input, .domain = ExecutionDomainId{1},
     .resource = device_resource},
};
auto compiled = HYLE_TRY(graph::compile(
    graph, CompileEnvironment{domains},
    CompileOptions{.shape = shape, .external_resources = external}));

auto invocation = HYLE_TRY(compiled.executable.create_invocation());
HYLE_TRY(invocation.bind(input, input_view));
QueueBinding queues[]{ {ExecutionDomainId{0}, &cpu_queue},
                       {ExecutionDomainId{1}, &cuda_queue} };
HYLE_TRY(invocation.prepare_replay(
    queues, ReplayOptions{.mode = ReplayMode::kPreferCapture}));
auto done = HYLE_TRY(invocation.submit(queues));
```

名字和语义冻结如下：

- `DimensionId/DimensionBounds/Dimension/TensorSpec/ShapeBinding` 属 Graph IR；相同 ID 表达相等，不引入
  任意 affine solver。batch、token、active-token 等调用方可直接定义为独立 symbol；复杂模型关系由构图者
  展开或按另一 signature 编译；
- `ExecutionDomainId` 是本次 CompileEnvironment 的有序 ordinal；`ExecutionDomain` 借用 Context 和稳定
  workspace resource。环境只在 compile 调用期间借用 span，Executable 内部保活已选 artifact/extension，
  但 resource provider 仍须先于 Executable/Invocation 及其 Buffer；
- `NodePlacement` 与 `ValuePlacement` 分别表达 required/preferred domain；external resource contract 必须
  可唯一解析。没有约束时按输入 locality、最少 direct boundary、环境顺序稳定裁决，不使用旧 benchmark ns；
- 每个 Executable 只持有一个 concrete `SpecializationSignature`。Graph 不拥有无界 cache；vLLM/SGLang
  风格的 batch descriptor -> executable map 留给引擎，HYLE 提供可比较/哈希的自含 signature；
- `QueueBinding` 在 submit/prepare 期间借用 Queue，必须与 domain Context 精确匹配。单 domain overload
  `compile(graph, context, ...)` / `submit(queue)` 保留为静态 CPU/简单设备 convenience，并路由同一实现；
- `prepare_replay()` 是显式冷路径：预留 completion slots、冻结地址/queue identity，并按
  `kCommands/kPreferCapture/kRequireCapture` 准备 backend capture。普通 command replay 不需要 capture；
  require 不支持时明确失败。reset/rebind 若改变稳定地址会使 capture 失效，不能静默回退；
- `Invocation::submit()` 不重新 specialization、placement、prepare provider、分配 workspace/event 或 JIT。

### placement、boundary 与 plan 算法

1. validate/canonicalize 为 `O(V+E)`；shape binding 按 symbol ordinal 建 fixed array，检查 missing/duplicate、
   bounds、multiple、乘法/stride/bytes overflow；
2. 外部 resource 与 required constraints 先固定。每个 topo node 对 D 个 domain 按 required、preferred、
   已在该域的输入数、boundary 数、环境 ordinal 排稳定候选；调用该 Context 的 prepare/region capability，
   只把 `Unimplemented` 当不支持，其它错误终止。总 prepare 尝试硬上界 `V*D`，region proposal 继续受现有
   global/per-provider budget；
3. 一条 edge 跨 domain 时产生目标域物理 instance和显式 Copy；若布局不同则必须有明确 Repack op/provider，
   不能把转换藏在 copy。compatible queue wait 由 Compute backend contract 录入；否则计划标记 HostBarrier；
4. liveness 在每个 domain 对其物理 instances 单独运行 Mem LifetimePlanner，复杂度总计 `O(V+E)`；alias
   仅在相同 resource/domain、provider 明确支持且 last-use 成立时选择；
5. plan 按 domain/queue segment 预绑定 command；同 queue 连续 step 只 finish 一次。多 queue completion
   state 和 dependency slots 在 create/prepare replay 冷路径一次分配，最终 Completion 聚合 O(Q+S)，不按
   node 分配 event；
6. placement 是有界确定性启发式，不声称求解一般 DAG 全局最优。CompileReport 必须披露 constraint、
   fallback、boundary 和 host barrier，使消费方可选择更强 provider/显式 placement。

### 有限入口与实现 ledger

| ID | 当前形状 | 裁决 | 冻结改动 |
|---|---|---|---|
| G55 | Graph value 只存 concrete TensorDesc | REPLACE | 增加 symbol/TensorSpec；concrete overload 路由 fixed spec |
| G56 | compile 没有 specialization identity | FIX | exact ShapeBinding -> concrete desc + self-contained SpecializationSignature；Executable 单 signature |
| G57 | compile 只借一个 Context | REPLACE | CompileEnvironment 的显式有序 domains；保留单 domain convenience |
| G58 | 无 placement constraint/算法 | FIX | Node/Value required/preferred + deterministic bounded placement，拒绝 timing oracle |
| G59 | external bind 没有 resource contract | FIX | compile-time ExternalResourceContract；Invocation bind 校验 exact compatible domain/resource |
| G60 | Copy/Repack/Barrier 只有 report enum | FIX | 真 plan step、目标域物理 instance、copy bytes、device wait/host barrier；无隐藏 staging |
| G61 | 单一 workspace LifetimePlan | REPLACE | per-domain plan/backing/report；跨域 instance 不 alias |
| G62 | Invocation 只接一个 Queue | REPLACE | domain QueueBinding span、segment prefix 和最终聚合；单 queue overload 路由同一路径 |
| G63 | 无稳定 replay/capture 阶段 | FIX | 显式 `prepare_replay` + ReplayMode；cold reserve/capture，warm submit 不分配/JIT |
| G64 | CompileReport 只有单 backend/provider 汇总 | FIX | signature、domain、placement、physical instance、transfer/sync/barrier/capture/concurrency facts |
| G65 | Graph 覆盖每-step Completion | REPLACE | 消费 Compute v5 EnqueueResult/QueueSequence finish；event 数按 segment |
| G66 | graph_ir/compile/plan 立即拆 target | DEFER | 概念职责明确，但当前 public symbol/owner/consumer 同轮强耦合；候选前只在源码形成边界，不为目录制造身份 |
| G67 | wide operator catalog | REJECT | 只使用 gated MLP、byte transfer、typed no-op 与 decode/prefill signature 验证主干 |

实现限 G55--G65；G66 只有当源码形成不同直接依赖、测试 owner 与独立 consumer 后才登记 component，不能
为了与架构图同名先造空 target。G67 不产生算子代码。

### CompileReport 真值扩展

每个 report 必须可查询：specialization signature；domain count 和每域 Device/Backend、workspace/persistent
bytes、queue/capture 要求；每 step 的 node range、kind、domain、provider、输入/输出物理 instance、copy
bytes、source/destination domain、transfer mode、device wait/host barrier、dispatch、blocking/threading 和
concurrent-invocation 能力。未知 estimate 仍用 optional，不写 0；复制 report 不保活 Context/provider
artifact/resource owner。

### P0 与公平性能边界

- compile：`O(V+E+V*D*provider-attempt + Q log Q)`，所有预算显式有界；允许冷 allocation/JIT/packing；
- create/prepare replay：O(values+steps+domains)，一次 workspace/event/capture allocation；
- warm submit：O(bindings+steps+dependency edges+segments)，零 Graph heap/provider search/shape inference/JIT/
  hidden copy/hidden host wait；event/finalization O(segments)，不是 O(nodes)；
- P1 比较同一 TensorDesc/storage/queue/completion 边界的 direct sequence；P2 比较同一 placement/copy/sync/
  completion 的 transformer fragment。CPU 裸函数、CUDA kernel-only 或不同 pageable/pinned host 合同不得
  与完整 Graph 比较。

### 七问

1. **第三方路径**：Graph 本体不直接依赖 CUDA/oneDNN/ggml；具体 factor 经 Compute extension 提供。四个
   consumer revision 固定，fallback 与版本进入 report。
2. **功能完整**：bounded specialization、显式 domains/placement/boundary、多域 workspace、多 queue replay/
   capture 和聚合覆盖两条纵切；模型循环/cache/policy 明确留给引擎。
3. **内部复用**：Error、Mem LifetimePlanner/Buffer、Compute v5 queue/completion/transfer/provider 全部复用；
   不私造 event、allocator、scheduler 或 vendor registry。
4. **性能**：冷/热复杂度、event 数和预算已固定；candidate 后只对真 backend 做同合同 P1/P2，不用算子
   差异掩盖 Graph 税。
5. **交付边界**：Native FFI N/A，旧 callable TaskGraph/C ABI 不恢复；公开仅 C++ Graph façade。
6. **构建交付**：Graph 通用 factor 不链接 vendor backend；source/static/shared 和选择 product 必须从相同
   component graph取得可选实现闭包。G66 当前不制造空 factor。
7. **易用与文档**：静态单域保留短路径；动态/异构用户显式看到 shape、resource、domain、queue、capture
   和 lifetime，不需要理解内部 plan。guide 必须分别给出两条 recipe 与 host-barrier/capture 坑。

### AUDITED 有限 FIX 与停止点

1. 先消费 Compute v5，完成 G65 和现有单域 direct/region 回归；
2. 实现 G55--G61 的 IR/compile/per-domain plan，并只用 CPU+mock second domain 验证全部错误/rollback；
3. 实现 G62--G64 的多 queue completion、prepare replay 与 report；mock 不能签真实 async/capture；
4. 以真 CUDA backend 跑 CPU<->CUDA copy + typed command + copy，以及 decode/prefill 两个 signature；
5. 冻结 candidate 后才运行三交付、四机通用门、sanitizer、P0和一次适用 P1/P2。

任何新 operator、无界 cache、控制流、模型/KV/batching 或 benchmark 专用 placement 都越出本轮。若 CUDA
factor 尚未可用，只精确阻塞真异步/capture slice，不缩成 fake backend 后宣称 Graph 完整。审计完成，
`graph` 可以进入 `AUDITED`，但产品修改仍须等待 Compute v5 的基础 SPI 同轮可编译迁移。

## 实现检查点（2026-09-13）

本节记录 `AUDITED` 后有限 FIX 的当前事实，不把候选写成 `CLOSED`，也不启动 G67 算子扩张。

### G55--G65 落地状态

- G55/G56：`DimensionId/DimensionBounds/Dimension/TensorSpec/ShapeBinding` 已进入 Graph IR；compile exact-once
  concrete 化并生成比较时不只依赖 hash 的 `SpecializationSignature`。一个 Executable 仍只绑定一个
  specialization，无进程全局 cache；
- G57--G59：`CompileEnvironment` 接有序 execution domains，node/value required/preferred placement 与
  external resource contract 已落地。placement 对每个 node/domain 的 prepare 尝试有硬预算，只有
  `Unimplemented` 继续候选；其它错误终止并回滚；
- G60 设计修正：跨 domain canonical tensor 产生真实 `Copy` step；同步由 segment dependency 表示，删除
  只占枚举而无执行语义的 synthetic Barrier。当前 region boundary 必须是 canonical TensorDesc，provider
  private layout 留在 region 内；在没有版本化 boundary-layout producer/consumer 合同前不制造 generic
  Repack。真实 provider 出现该需求时，它构成独立 reopen trigger，而不是本轮伪实现；
- G61：每个 `(logical value, domain)` 建 physical instance；各 domain 独立生成 Mem LifetimePlan，workspace
  backing、alias 和 invocation-owned output 均不跨 resource/domain；
- G62/G65：plan 已生成 Queue segments 与依赖 DAG。同一 segment 用一次 `QueueSequence::finish()` 覆盖全部
  accepted commands；最终 Completion 只聚合每个 physical Queue 的最新 terminal prefix，并保活 Invocation、
  workspace 与 provider state；
- G63：`prepare_replay(queues, ReplayOptions)` 已成为 cold commit point，完成 exact Queue、bindings、实际
  transfer route、dependency route、Completion capacity 与 capture 准备。`kCommands/kPreferCapture/
  kRequireCapture` 已实现，CUDA 真机可形成 command/captured hybrid；bind/reset 会使 route 失效；
- G64：CompileReport 已枚举 specialization、domain、physical value、step、segment、dependency、workspace、
  copy/capture/concurrency facts；新增 ReplayReport 披露绑定具体 Queue 后的 immediate/queued/peer transfer
  数以及 ordered/synchronous-source/device-wait/host-wait 数。host fallback 可由
  `HostWaitPolicy::kReject` 在任何提交和 slot reserve 前拒绝。

### 调度与 steady route

编译器在所有 operation/region/copy 已 prepare 后，对 step dependency 图运行确定性 Kahn scheduler。每个
domain 使用稳定 ready FIFO；当前 domain 仍有 ready step 时继续该 domain，否则选择原拓扑 ordinal 最早的
ready step 所在 domain。该有界启发式消除交错独立分支造成的可避免 segment，不读取计时 oracle，也不
声称求解全局最优异构调度。

`prepare_replay` 把每条 dependency 固定为：同 physical Queue 的 ordered route、同步 source、backend
确认的 device wait 或显式 host wait。warm submit 不再做 capability query，也不会把运行期
`Unimplemented` 静默降为 host wait；若已承诺 device wait 的 backend 在提交时拒绝，按内部合同破坏报告。
host wait 在取得 destination Queue single-producer admission 前执行，避免慢上游长期占住目标 Queue。
共享 physical Queue 的 completion capacity 在冷路径按所有 logical domain segments 求和并只 reserve 一次。

### 当前验证事实与剩余资格

- W1-WIN Release/shared、CUDA 13.3：`hyle_graph_test` 14/14、`hyle_graph_executable_test` 63/63、
  `hyle_compute_backend_cpu_test` 18/18；其中 RTX 4070 真 CUDA capture/hybrid replay 通过；
- 已覆盖 bounded symbol、跨域真实 copy、interleaved branch scheduling、共享 Queue、不可用 transfer 冷拒绝、
  host-wait 披露/拒绝、同步 source、device-wait mock、并发 Invocation、late error、abandon 与 metadata OOM；
- P0 结构目标已经由代码和 allocation/capability-query probes 支持：warm 路径无 Graph heap、DAG walk、
  shape inference、provider search、JIT 或 route query，finalization 数随 segment 而非 node 增长；
- W1-WIN 的 `compute` 与 `graph` 均已通过直接源码、shared SDK、static SDK 及 relocation 后 installed
  consumer；Graph consumer 同时覆盖 native 三算子执行与 oneDNN gated-MLP region，安装树没有泄漏 private
  provider SPI；
- WSL GCC 14 的 CUDA/oneDNN-OFF Release、Compute/Graph 逐头及直接 owner/consumer 测试通过；ASan、UBSan、
  TSan 四组直接目标均通过。首次回放发现 optional-absent CUDA placeholder target 被测试误认为真实实现，已将
  `cuda_graph_test.cc` 接线改为要求 `HYLE_COMPUTE_CUDA_ENABLED`，随后 CUDA-OFF 构建通过；
- 本轮尚未形成完整跨平台 immutable candidate，也未执行 macOS 与必要 P1/P2，因此 registry 继续保持
  `AUDITED`。这些门应在基础实现不再变化后一次运行，不因文档或状态维护重复启动。
