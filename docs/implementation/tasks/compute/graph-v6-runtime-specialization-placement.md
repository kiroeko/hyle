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
