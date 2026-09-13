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

## AUDITED 结论（2026-09-13）

### 来源对账与边界裁决

- ggml `e91ded11bdcd` 的 backend 基础面包含 async set/get/copy、device event create/record/wait/synchronize，
  scheduler 在 split 边界复用 event；其 CUDA route 在 source stream copy 后 record event，再令 destination
  stream wait。它证明 transfer/event 是 backend runtime，不是 operator provider，也不应按 node 产出 event；
- llama.cpp `e71b80510c84` 使用该 backend scheduler；vLLM `51da0ca66c80` 和 SGLang
  `dcebe8c4733a` 把 graph pool、稳定地址与 capture/replay state 放在 invocation/runner；ds4
  `f62ca29a3087` 把显式 stream、stream-owned scratch/pool 与 capture-safe allocation 分开。共同稳定形状是
  “冷路径准备资源和 event，热路径向既定 queue 录命令，只在 segment/依赖边界终态化”；
- CUDA Runtime/Driver、Metal command buffer/shared event 与 Vulkan timeline 的 vendor 对象不能进入普通
  Compute 头。共同可上提的只有：ordered queue sequence、byte transfer、compatible dependency wait、prefix
  completion 与版本化对象局部 extension。module/kernel/pipeline 参数仍由 typed backend factor 拥有；
- 本轮不拆 `compute_command`/`compute_event` component：这些机制与 Context/Queue/Completion 共用相同依赖、
  owner、测试和所有直接消费者，独立 target 只会制造环与公开心智。`compute` 仍是该 runtime 质因数。

### 能力与形状审计

| 能力族 | 层级 | 准入旅程/来源 | 冻结语义与生命周期 | 当前 | 动作 |
|---|---|---|---|---|---|
| queue sequence | 原子机制 | direct op、Graph segment、ggml split | begin 先取得 admission/单 producer 并预留终态 slot；accepted work 只在 `finish` 生成一个 prefix Completion | PARTIAL | FIX |
| provider/region enqueue | 原子机制 | CPU/CUDA op、oneDNN region | 返回 `EnqueueResult{accepted,status}`；rejected 表示本 command 未录入，accepted+error 表示已有部分 prefix 必须终态化 | MISSING | REPLACE |
| prefix finalization | 原子机制 | CUDA event、Metal command buffer、timeline value | 每 direct submit 或 Graph queue segment 一次；空/rejected-only sequence 直接 inline terminal；accepted sequence 必须返回有效 Completion | MISSING | FIX |
| byte transfer | 复合功能 | model staging、renderer upload/download、ggml async copy | `copy_async`/`fill_async` 使用 BufferView；不解释 Tensor format、不隐藏 pageable staging，range/domain/overlap 在接纳前验证 | MISSING | FIX |
| transfer capability | 原子机制 | Graph placement/materialize boundary | 冷查询说明 immediate 或 queued-direct；unsupported 显式失败，unknown 不写成免费/异步 | MISSING | FIX |
| compatible queue wait | 原子机制 | multi-stream Graph、ggml event wait | 只借用仍存活的 backend prefix Completion；兼容 route 设备侧录 wait，不兼容返回 Unimplemented 供 Graph 显式 host barrier | MISSING | FIX |
| backend extension | 原子机制 | CUDA/Metal/Vulkan typed interop | 精确 ID/version、Context-retained、对象局部 query；compile/typed setup 可缓存，普通 op submit 不查询 | MISSING | FIX |
| typed native command | 复合功能 | custom CUDA kernel、renderer compute pass | vendor header/factor 提供 typed stream/encoder callback 或 artifact；callable 只在 host enqueue 期间借用，不塞进通用 ABI | ROUTED | BORROW |
| async storage owner | 复合功能 | device/pinned/managed allocation | Mem Buffer/View 不改 layout；typed backend 稳定 resource owner 保活 backend context，并明确要求 owner 先于 Buffer/in-flight work | ROUTED | BORROW |
| public generic Event/CommandList | 复合功能 | 无第二个独立自然 consumer | Completion 已是用户终态；内部 sequence/event 足够 Graph，公开万能 vendor 抽象会丢语义 | OUT_OF_SCOPE | REJECT |
| callback/coroutine continuation | 复合功能 | Sched bridge | 不强制 Compute 依赖 Sched；后续 `compute_sched` 组合同一 Completion | ROUTED | DEFER |

全部公开入口回映如下：`Backend/Device/Context/Queue` 回指 runtime identity/admission；`Completion` 回指
sequence boundary；`OperationInvocation::submit` 保持自然 public 形状，但内部改为 begin -> enqueue -> finish；
新增 `transfer_properties/copy_async/fill_async` 回指 byte-transfer 族。Provider/backend extension、
`QueueSequence`、`EnqueueResult` 和 completion dependency 均只供 factor 间使用，不成为普通用户第二套路。

### 冻结自然 C++ 形状

```cpp
auto queue = HYLE_TRY(context.create_queue());
HYLE_TRY(queue.reserve_completions(concurrent_lanes)); // cold, idempotent

auto uploaded = HYLE_TRY(compute::copy_async(queue, device_view, host_view));
HYLE_TRY(uploaded.wait());

auto done = HYLE_TRY(invocation.submit(queue, inputs, outputs));
HYLE_TRY(done.wait());

// Vendor header owns the native type and callback; ordinary Compute headers do not.
auto custom = HYLE_TRY(compute::cuda::submit(queue, [&](cudaStream_t stream) noexcept {
  return launch_my_kernel(stream, args);
}));
```

`reserve_completions()` 只增长 backend 的可并发终态槽，不能 shrink、drain 或改变 queue 顺序；CPU 为 O(1)
no-op。普通单次 direct submit 仍产生一次 Completion。Graph 使用内部 sequence 把同一 queue segment 的多个
op/region/copy 录完再 finish，因而只产生一次 backend completion；不要求用户管理内部 sequence。

`TransferProperties` 只报告 `kImmediate` 或 `kQueuedDirect` 以及是否支持 peer route。HYLE 不在该 API 内
偷偷分配 staging；普通 pageable host <-> CUDA 若不能保证 queued-direct，就明确不具备该路线，Graph 可
显式 materialize pinned staging，普通用户也可选择同步 Mem 操作。

### 有限入口与实现 ledger

| ID | 当前入口/机制 | 裁决 | 冻结改动 |
|---|---|---|---|
| C55 | provider/region submit 返回每-step Completion | REPLACE | 内部 `EnqueueResult` 明确 accepted 与同步 status；public direct submit 仍返回 Completion |
| C56 | QueueSequence 只做 admission guard | FIX | 增加 backend begin token、accepted/tail 累积、显式 finish 与安全 abandon；同一 sequence 只 finalizes 一次 |
| C57 | BackendVTable 只有 create/destroy/drain | FIX | 增加 begin/finish sequence、transfer query/enqueue、compatible wait 与 query_extension；CPU 完整实现 immediate route |
| C58 | 只有 ProviderExtensionRef | FIX | 增加同形状的 versioned `BackendExtensionRef/Access`，保活 exact Context 并验证 Queue belonging/admission |
| C59 | Completion 无 factor dependency identity | FIX | 增加仅内部借用的 exact backend/vtable/state view；源 Completion 必须保活到 consumer prefix 终态，public 不可复制 |
| C60 | 无 direct async byte transfer | FIX | 增加 `transfer_properties`、`copy_async`、`fill_async`；BufferView 合同，不把 repack/tensor fill 混进来 |
| C61 | Queue 无 completion slot 预留 | FIX | 增加 cold、幂等 `reserve_completions(minimum)`；Graph 在 replay 前按 report/lanes 预留，steady exhaustion 明确失败 |
| C62 | native custom command 只有长期概念 | BORROW | core 只交付 backend extension/sequence bridge；CUDA 等 typed factor 交付自然 callback/artifact API |
| C63 | Graph 保存并覆盖每-step Completion | REPLACE | Graph v6 改为每 queue segment finish；多 queue 保存前缀并显式 wait/host barrier/聚合，不按 node record event |
| C64 | CPU/native/oneDNN 与 region SPI consumers | FIX | 逐 factor 迁移到 EnqueueResult；同步实现不分配、不造 async state，错误与 partial acceptance 保真 |
| C65 | QueueOptions 空类型 | KEEP | 通用 options 暂不塞 vendor priority/flags；completion 容量用可增长 reserve，typed options 留给 backend façade |
| C66 | 通用 Module/Kernel/CommandList | REJECT | 删除/不恢复占位；只有第二个稳定跨 backend consumer 证明共同 typed 形状时再 scope |

实现只限 C55--C64；C65--C66 不产生产品字节。因 C63 会改变当前 Graph，Graph v6 必须先独立进入
`AUDITED`；因 graph_provider_onednn 当前 `BLOCKED`，只有其 resume condition 在 common submit path 实际
变化后才恢复并迁移，不能由本卡越权修改。

### 生命周期、失败与并发矩阵

- begin 在任何命令接纳前取得 Context/Queue admission、single-producer guard 和 backend completion slot；
  reserve/OOM/closed 失败不会留下设备工作。accepted 后即使 tail 同步失败也必须 finish，最终 Completion
  覆盖已录 prefix 并保留首个 tail error；
- provider/region 可返回 `accepted=false,error`，也可在多命令 region 已录部分工作后返回
  `accepted=true,error`。后者禁止继续录入，但不能丢失前缀、workspace 或错误；成功 semantic no-op 可
  `accepted=false,OK`；
- sequence destructor 只作为异常防线：无 accepted work 时归还预留 slot；有 accepted work 时以内部错误
  finalizes 并立即走 backend 的安全 abandonment，绝不 host-wait。正常产品路径必须显式 finish；
- compatible device wait 借用 source Completion，Graph 保活它直到 destination prefix 终态；backend 不得
  缓存裸状态超过该边界。跨 backend 不兼容时不隐式 wait，Graph 选择显式、可报告的 host barrier 或拒绝；
- Completion abandonment 仍不等待。具体 backend 必须让 event/stream state 和 attached Invocation owner
  活到真实终态；设备不支持抢占时 cancel 返回 Unimplemented，不能提前释放 Buffer/workspace；
- typed backend resource owner 的稳定实现地址先于所有 MemoryResourceRef/Buffer/View/in-flight work；
  Compute 不给所有 Buffer 增加 shared owner，也不允许从临时 resource owner 取得引用。

### P0 成本与复杂度账单

| 路径 | 冷路径 | steady host 成本 | event/finalize 上界 |
|---|---|---|---|
| CPU direct op | prepare/create invocation；reserve no-op | 两级 admission + single-producer guard + 预绑定 provider call；无 heap/search/JIT | inline Completion，一次 finish |
| CUDA direct op/copy/custom | context/queue/event slots/resource/kernel prepare | admission + 一次 typed enqueue + 一次 event record；无 HYLE heap/search/host wait | 每 public direct submit 一个 event |
| Graph single queue, N steps | compile/bind/workspace/event reserve | O(bindings + N enqueue)，不重新遍历原始 DAG、不搜索 provider | **每 queue segment 一个**，不是 N 个 |
| Graph Q queues/S segments | placement、dependency、per-lane slots | O(bindings + steps + dependency edges)；兼容边设备 wait，不兼容边显式 barrier | O(S)，最终聚合 O(Q) |

Context/provider 枚举仍是冷路径 O(P)；transfer query O(1) vtable/capability call；copy/fill 为 O(bytes) device
work、O(1) host enqueue；dependency enqueue O(1)。`reserve_completions(n)` 允许 O(n) vendor resource 创建，只
在冷路径。地址、顺序和亚纳秒诊断不得改变这些公开形状。

### 七问

1. **第三方路径**：Compute core 无新增第三方依赖；CUDA/Metal/Vulkan 库、版本、许可证和 fallback 均留给
   各 typed factor。四个 consumer revision 与 ggml revision 已固定在本卡。
2. **功能完整**：本轮覆盖 direct op、byte transfer、typed custom command、single/multi-queue Graph 所需
   最小 backend runtime；宽算子、模型策略和 coroutine continuation 有明确 owner，不以“上层处理”掩盖
   通用 glue。
3. **内部复用**：继续复用 Core/Error、Mem Buffer/View、Util-core refcount；不复制 scheduler、allocator、
   status 或 Graph planner。资源 owner 不反向修改 Mem 常态对象。
4. **性能**：复杂度和 event 数按 public submit/Graph segment 而非 node；extension 查询、reserve 和
   capability match 都在冷路径；P1 等真实 CUDA immutable candidate 后一次执行。
5. **交付边界**：Native FFI N/A；不恢复历史 C ABI。vendor handle 只出现在选择对应 typed C++ header 的
   consumer TU。
6. **构建交付**：Compute core 不链接 CUDA；optional backend factor 负责 toolkit discovery/export/install，
   AUTO/OFF 不探测。base vtable/extension bridge 必须在 source/static/shared 和精确 factor 选择一致。
7. **易用与文档**：普通 C++ 用户只需 Context/Queue、BufferView、direct submit/copy 和 Completion；复杂
   sequence 对 Graph 隐藏，custom kernel 用户显式选择 typed backend header，所有 borrow/close/drain 坑须进 guide。

### AUDITED 有限 FIX 集与停止点

1. 实现 C55--C61 的 core API/SPI、CPU immediate 路线与单元/错误注入测试；
2. 在其各自合法状态内迁移 Native/oneDNN provider；Graph v6 AUDITED 后实现 C63；
3. 实现 backend extension 的一个无 vendor mock，证明 exact owner、wrong Context、close 与预绑定零搜索；
4. 更新 architecture/user/developer 文档和三交付；完成 P0 codegen/成本检查；
5. 随后由 CUDA v1 实现真实 event/stream/resource/copy/native seam，才签异步生命周期与必要 P1。

不在上述五项内增加算子、公共 Event/CommandList、callback/coroutine 或 vendor 实现。发现新的能力必须
重新 scope；每项最多两轮根因纠正。由此，本轮审计完成，`compute` 可以进入 `AUDITED` 并开始有限实现。
