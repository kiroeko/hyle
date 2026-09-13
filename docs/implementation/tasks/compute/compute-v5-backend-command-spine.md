# Compute v5：真实 backend command 与 queue-prefix runtime 主干

## Closure card

- component / owner：`compute` / Compute；round=`compute-v5-backend-command-spine`；
- baseline commit：`b32bef0486e642416c02eb835b68697db9be439c`；上一轮 v4 的已实现检查点和精确
  阻塞事实保存在 [`compute-v4-runtime-extensibility`](compute-v4-runtime-extensibility.md)，本轮不回写或
  扩张其 R40--R51 历史账本；
- trigger 与证据：CUDA v1/Graph v6 审计发现，现有 provider/region SPI 为每个已录入命令返回 terminal
  `Completion`，真实异步 backend 只能每 step 记录 event 或谎报终态；同时 `BackendVTable` 没有对象局部
  command extension、sequence finalization、async transfer 或 device-side dependency wait。ggml
  `e91ded11bdcd` 的 scheduler 在 split/依赖边界预留 event 并 record/wait，llama.cpp
  `e71b80510c84`、vLLM `51da0ca66c80`、SGLang `dcebe8c4733a` 与 ds4 `f62ca29a3087` 的实际 GPU 路线也都
  把 stream command recording 与用户终态分开；
- 正常 consumer 任务：调用方在一个显式 Context 的 Queue 上连续录入 prepared op、backend-native command
  和 transfer，只在 direct call 或 Graph queue segment 的边界取得一个可 poll/wait 的 Completion；另一个
  Queue 可在设备侧等待兼容的前缀，不兼容路线明确返回 unsupported 或由 Graph 记为显式 host barrier；
- 主产品纵切 / 具名来源：CPU prepared op 与 CUDA H2D -> typed no-op/kernel -> D2H 共享相同 direct/Graph
  queue-sequence 合同；Graph 对 CUDA segment 只产生边界 event，不随 node 数增长；decode/prefill replay
  不在 steady path 搜索 extension、分配 wrapper 或逐 step host-wait；
- 正交反偏置纵切：renderer/codec 在调用方已有 CUDA/Vulkan/Metal queue 与 storage 上录入 upload、typed
  native command、download，并把最终 Completion 接回资源管线；证明该 seam 是通用 backend runtime，
  不是 transformer 专用 API；
- 跨模块 public journey 与每段 owner：Mem 继续拥有通用 Buffer/View 与同步 resource 操作；Compute 拥有
  Context/Queue admission、命令序列、backend-neutral transfer/dependency SPI 和 terminal Completion；具体
  backend 拥有 stream/event/native handle/resource implementation；Provider 只录入一个 op/region，不创建
  Graph 每-step 用户 event；Graph 拥有跨 domain segment、显式 copy/sync、host-barrier fallback 和最终聚合；
- scope：冻结并实现 queue sequence 的 accepted-prefix/finalize 合同；把 provider 与 region admitted submit
  改为同步 enqueue `Status`；增加有版本的对象局部 backend extension；冻结 backend-neutral async copy/fill、
  compatible device-wait、queue-prefix Completion 和 typed native command 的 factor bridge；裁决 async resource
  owner；迁移现有 CPU/native/oneDNN/Graph consumers；逐入口 `KEEP/MOVE/REPLACE/DELETE`；
- non-goals：增加公共算子、CUDA/Metal/Vulkan 的具体实现、公开万能 `Module/Kernel/CommandList`、Graph 的
  symbolic IR/placement/partition、跨 backend 自动 interop、Sched/coroutine bridge、模型/KV/batching/sampling/
  serving、Native FFI、为 oneDNN benchmark 保留旧 per-step Completion；
- 直接依赖 / 直接消费者：Compute 继续公开依赖 Core/Error/Mem，私有使用 Util-core；不新增 CUDA 或 Graph
  反向依赖。直接消费者是 `compute_backend_cpu`、各 provider SPI、Graph v6 与已登记的 CUDA/Apple/Vulkan
  backend factors；受 SPI 变更影响的 factor 必须各自形成合法 reopen/当前轮修正，不在本卡越权写产品；
- 性能 tier 与原因：先 `P0` 证明 direct submit 为一次 admission + 一次 provider enqueue + 一次 backend
  finalize，Graph 同 Queue 的 event/finalize 数只随 segment/queue 数增长而不随 node 数增长；无 steady
  extension search、HYLE heap、JIT、copy 或 host wait。只有真 CUDA candidate 后才对同 stream/event 合同做
  一次 `P1`，固定 5% non-inferiority；Graph 端到端留在 Graph `P2`；
- 适用机器 / 仪器 / material margin：W1-WIN 做 CPU 回归、三交付和 RTX 4070 真 CUDA seam；WSL Linux 与
  两台 Apple 回放通用 runtime、headers、sanitizer/交付，并只对未选择 CUDA 的构建语义签字。P1 只在真实
  backend immutable candidate 后启动，固定 5%；
- 调查与纠正预算：ggml scheduler/event 一次、CUDA Runtime/Driver event/stream 一次、Metal command
  buffer/shared event 一次、Vulkan timeline 一次；owner/API 最多两轮；有限 FIX 每项最多两轮；candidate 后
  各适用验证一次；
- 预声明验证：empty/one/multi-command sequence、同步拒绝前/后已有 prefix、一个 segment 只 finalizes 一次、
  same/cross Queue dependency、unsupported cross backend、immediate/pending/error Completion、abandon without
  hidden wait、late terminal、close/drain race、wrong Context/Device/resource、zero/range/overlap transfer、
  provider/region migration、concurrent queue producers、OOM/error injection、逐头、ASan/UBSan/TSan、四机、
  source/static/shared、选择 factor/offline/license、P0及必要 P1、终局红队；
- stop / blocker 条件：若 sequence 边界无法在不使所有普通 Completion/Buffer 支付常态 owner 成本的前提下
  保活 async work，精确阻塞；若 Graph 必须知道 vendor handle 才能表达 copy/wait，退回 owner 设计；若
  cross-backend dependency 只能 host wait，允许作为显式可报告 fallback，但不得隐藏或冒充 device wait；若
  provider SPI 迁移要求修改非当前合法状态 factor，先登记该 factor 的 reopen trigger。

## SCOPED 待审计项

1. `QueueSequence` 是否应记录 accepted command 数并由 `finish(Status tail)` 唯一生成 prefix Completion；空
   prefix、同步 tail rejection 和析构未 finish 各自是什么语义；
2. provider 和 region SPI 的 admitted 路线是返回 `Status` 还是更窄的 enqueue result；怎样保证“错误即本
   command 未接纳”，同时保留已接纳 prefix 的最终错误；
3. `BackendVTable` 哪些能力应为所有 backend 的必需 base（queue-prefix finalize），哪些应在有版本的
   object-local extension（copy/fill/wait/native）；不能为可选 CUDA 能力扩大每个 Device/Queue 对象；
4. backend extension 的 ID、query、retained Context owner、Queue belonging/admission 与预绑定方式；typed
   factor 怎样在不让普通 Compute 头包含 vendor header 的情况下提供自然 C++ façade；
5. Graph 同 queue 多 step 如何只记录一个 event；多 queue 如何保存每域 prefix、enqueue compatible wait、
   显式 host barrier，并把所有 terminal error/lifetime 聚合为最终 Completion；
6. Completion 需要暴露何种仅供 factor 使用的 dependency identity/state；不能使 public Completion 可复制，
   也不能让 dependency 在 source event 真终态前悬空；
7. async copy/fill 的标准合同应接 `BufferView` 还是 `TensorView`；resource/device/domain/range/overlap 验证
   分别由 Compute bridge 与 backend 谁承担；
8. CUDA device/pinned/managed resource 是稳定地址的 move-only pool owner 还是 Context 内借用 resource；
   `MemoryResourceRef`/Buffer 不保活 provider，因此 owner 必须自然地先于 Buffer 且能 retained Context；
9. 同步 Mem `read/write/copy/fill` 与 Queue async transfer 的命名和完成边界，不能让 Graph steady path 调用
   同步 Mem fallback；
10. typed native command 是 inline callback-on-stream、typed handle scope 还是预编译 command artifact；普通
    direct 用户和 Graph compiled command 都不能在 steady path字符串查询/JIT/分配；
11. close/drain、Completion abandonment、event pool/callback 与 device loss 的 exact owner；不支持设备抢占时
    cancel 只请求停止后续 admission，不能提前回收 storage；
12. 现有 Provider/Region/Graph/CPU 入口及注释逐项 KEEP/MOVE/REPLACE/DELETE；识别所有因当前测试纵切形成、
    但在真实 backend 上会逐 node 付费的形状。

本卡只授权审计。完成能力表、自然调用草图、P0 账单、七问和有限 FIX ledger 并进入 `AUDITED` 前，不改
Compute、Provider、Graph 或 backend 产品代码。
