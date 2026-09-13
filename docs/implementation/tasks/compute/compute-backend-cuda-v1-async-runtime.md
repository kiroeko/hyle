# Compute CUDA backend v1：真异步 storage、queue、completion 与 interop 审计

## Closure card

- component / owner：`compute_backend_cuda` / Compute；round=`compute-backend-cuda-v1-async-runtime`；
- baseline commit：`c9382c4008e616e816e62de30ee5c2d39b2e327b`；component 已在该提交以空
  `QUEUED` identity 登记，没有源码、依赖、公开头、选项或产品投影；
- trigger 与证据：Compute v4 已实现 async-capable `Completion`、Context/Queue close/drain、late-lifetime
  attachment 与开放 backend identity，但只用 CPU immediate/fake async 验证，因缺真实异步 backend 保持
  `AUDITED`。Graph v6 又需要真实 device storage、copy/event 与 multi-queue prefix。W1-WIN 有 NVIDIA
  GeForce RTX 4070（compute capability 8.9）、driver 610.62、CUDA Toolkit/nvcc 13.3，因此不存在无设备
  伪资格化理由；
- 正常 consumer 任务：C++ 用户显式取得 CUDA Backend/Device，建立 Context 与 non-blocking Queue，创建或
  导入 pinned/device/managed storage，在 queue 上提交 async H2D/D2D/D2H copy 与 typed native command，
  取得可 poll/wait/cancel-query 的 Completion；close 停止接纳，drain 等待已接纳 prefix，丢弃 pending
  Completion 不提前释放 event、stream、storage 或 Context；
- 主产品纵切 / 具名来源：llama.cpp/ggml `e71b80510c84` 的 CUDA backend buffer、stream/event 与 scheduler
  async copy；vLLM `51da0ca66c80`、SGLang `dcebe8c4733a` 的 CUDA graph stream/pool/capture replay；ds4
  `f62ca29a3087` 的 explicit `cudaStream_t`、stream-owned async pool 与 capture-safe allocation；本机 CUDA
  13.3 的 stream/event/mempool/external-memory API；
- 正交反偏置纵切：renderer/codec 导入已有 CUDA-compatible allocation/stream 或显式导出 native handle，
  执行 upload -> typed custom kernel -> download，并把 Completion 接回自己的同步链；不引入模型、KV、
  cuBLAS 算子或 Graph 才有的 placement 策略；
- 跨模块 public journey 与每段 owner：Mem 提供通用 `Buffer/View/MemoryResource` owner 形状但不理解 CUDA
  stream；Compute core 拥有 Backend/Device/Context/Queue/Completion 和对象局部 extension 生命周期；本
  factor 拥有 CUDA device/context/stream/event、device/pinned/managed resource、async copy/fill、native
  interop 与错误映射；Graph 只消费 backend-neutral capability，不反向拥有 CUDA；具体 cuBLAS/CUTLASS/
  HYLE kernels 属后续独立 provider factor；
- scope：审计 CUDA Driver API 与 Runtime API 的唯一 owner；device enumeration/properties；primary/runtime
  context 与 thread-current 规则；non-blocking stream queue；event/pool-backed Completion；device/pinned/
  managed resources及 import/adopt；async copy/fill/event wait；typed stream/context/native command interop；
  close/drain/device reset/loss；capture compatibility；错误/OOM；现有 Compute/Mem seam 是否足够；公开入口
  与构建交付逐项 `KEEP/MOVE/REPLACE/DELETE`；审计后只实现冻结的最小真异步纵切；
- non-goals：cuBLAS/cuDNN/CUTLASS/ggml provider、宽算子目录、Graph partition/placement、模型/KV/
  batching/sampling/serving、分布式 NCCL、训练、HIP/ROCm、Vulkan/Metal/MLX、默认全局 device/stream、
  process registry、Native FFI；
- 直接依赖 / 直接消费者：SCOPED 时仍为空 factor。候选预计公开依赖 Core/Error/Compute/Mem，私有依赖
  Platform/Util-core 与 CUDA Toolkit；若 Mem 通用资源无需修改，不增加反向依赖。直接消费者为 direct
  CUDA C++ 用户、Graph v6 的真异步纵切及后续 CUDA provider；普通 `hyle::compute` 产品是否默认包含本
  factor 必须在 AUDITED 时根据可选依赖/发现语义裁决；
- 性能 tier 与原因：`P0` 先冻结每次 submit/copy 的 allocation、event、atomic、lock、driver call、host
  wait、hidden copy 与最坏复杂度。candidate 若 wrapper 相对同一 CUDA stream/event/copy 合同存在未解释
  边界成本，再用一次 `P1`；Graph 端到端收益属于 Graph/backend 共同 `P2`，不在本 factor 用裸 kernel
  伪造；固定 5% 工程门；
- 适用机器 / 仪器 / material margin：W1-WIN RTX 4070 为 CUDA correctness/lifetime/P0/P1 资格机；WSL
  仅在 GPU passthrough 与 toolkit 身份可证时回放，不以 Windows 同物理卡冒充独立硬件；两台 Apple 对本
  可选 factor 只验证 AUTO/OFF 和显式 ON 的清晰诊断，不签 CUDA 能力；
- 调查与纠正预算：CUDA 13.3 stream/event/memory/capture API 一次；ggml、vLLM、SGLang、ds4 各一次定向
  owner 扫描；API/owner 最多两轮；必要 Compute backend extension 一轮；每项 FIX 最多两轮；candidate
  后 correctness/sanitizer/三交付/P0/P1 各一次；
- 预声明验证：0/N device、invalid ordinal、multi-device identity、properties、context/thread-current、两个
  queue、ordered prefix、event pending/poll/wait、pending handle abandonment、late error、close/drain、
  context/queue/storage mismatch、H2D/D2D/D2H、zero/unaligned/range/overlap、device/pinned/managed OOM、
  imported owner teardown、capture-safe/unsafe operation、concurrent invocations、device loss、错误注入、
  逐头、Windows真机、Apple OFF/ON diagnostic、source/static/shared、精确 factor/product、toolkit discovery/
  offline/package/license、P0及必要 P1、终局红队；
- stop / blocker 条件：若现有 Compute/Mem 不能在不破坏已关闭合同下表达 async resource/copy/event，先把
  精确上游缺口形成合法 reopen trigger，不在 CUDA factor 私造同义 runtime；若 Toolkit/driver API 的
  failure 或 context ownership 无法转成 HYLE Error，精确阻塞；若 pending Completion 析构必须隐藏 wait
  或常态 heap allocation才能安全，停止；若真机不可用只阻塞 CUDA factor，不用 fake backend 冒充。

## SCOPED 待审计项

1. Driver API、Runtime API 或两者组合中，谁拥有 device/context/stream lifetime；是否使用 primary context，
   怎样避免线程 current-context 隐式状态污染普通调用；
2. `BackendVTable` 目前只有 create/destroy/drain，Graph 需要的 async copy/event/native command 能否由一个
   versioned object-local backend extension表达；该机制应留在 Compute core 还是 CUDA typed façade；
3. device、pinned 和 managed storage 应由 Context-owned resource、独立 retained resource owner，还是
   typed pool 拥有；`MemoryResourceRef` 非 owning，如何保证 Buffer 比临时 Context 活得久时安全失败而非
   UAF；
4. 同步 Mem `read/write/copy/fill` 与 Compute queue async copy 怎样并存；不能把隐式 stream synchronize
   藏入 Graph steady path，也不能让普通 Buffer 操作假装异步；
5. 每次提交是否创建 `cudaEvent_t`；若使用 event pool，pool、slot、generation 与 abandoned Completion
   谁持有，如何实现 warm zero-HYLE-allocation 且 device error 可终态化；
6. `Completion::request_cancel()` 在不可抢占 CUDA work 上应返回 `Unimplemented` 还是只停止后续 Graph
   admission；无论哪种都不能提前复用 storage/workspace；
7. close/destroy stream、context、resource 与 late host callback/event 的顺序；`cudaDeviceReset` 不得作为
   普通析构或错误恢复副作用；
8. native interop 暴露哪些 typed handles 和 import/export ownership；普通 Compute 头不得包含 CUDA 头，
   shared/static/source 的 ABI 与编译选项如何一致；
9. graph capture 期间 event query、host callback、async allocation 的限制怎样报告；哪些准备动作必须移到
   create-invocation/capture 冷路径，warm replay 只能保留什么；
10. CUDA Toolkit 采用 `find_package(CUDAToolkit)` 还是其它机制；OFF 不探测，ON 缺失须明确失败；版本、
    redistributable runtime、driver compatibility、许可证和 offline source/shared/static packaging 如何表达；
11. 最小 architecture probe 是 async H2D -> typed no-op/copy kernel -> D2H，还是仅 copy/event 已足以签
    runtime；任何 kernel 只验证 native command seam，不扩算子 provider；
12. CUDA factor 暴露后，Compute v4 的 R48--R50、late-lifetime 和 Context/Queue contracts 需要哪些真实
    修正；发现上游缺陷必须先更新其合法状态，不能在本 factor 绕过。

本卡只授权审计。进入 `AUDITED` 并冻结能力表、API/usability、依赖、有限 FIX 和上游触发前，不添加
CUDA 源码、公开头、CMake option、Toolkit discovery、测试或产品 façade 投影。

## AUDITED 结论（2026-09-13）

### 来源对账与版本裁决

- CUDA Runtime 13.3 的 execution-context 合同明确区分两类 API：device-wide 的 allocation/event 创建仍是
  current-device 冷操作；`cudaLaunchKernel`、`cudaMemcpyAsync`、`cudaMemsetAsync` 等 context resource
  操作从**非空显式 stream**推导 context。`cudaDeviceGetExecutionCtx` 与
  `cudaExecutionCtxStreamCreate` 因而可以让 steady enqueue 不调用 `cudaSetDevice`，也不临时改写调用线程的
  current context；
- HYLE v1 选择 **CUDA Toolkit 13.3 minimum**。ggml/llama.cpp 的宽版本兼容通过反复
  `cudaGetDevice/cudaSetDevice` 取得，但 HYLE 是会嵌入其它 runtime/plugin 的基础库，不能把 TLS context
  污染或每次 push/pop 成本藏进热路径。若未来真实 consumer 必须支持 12.x，应另开明确 compatibility
  route，不降低本轮默认合同；
- primary execution context 由 Runtime 取得，HYLE 不创建隔离 `CUcontext`，也不调用
  `cudaExecutionCtxDestroy` 或 `cudaDeviceReset`。为了验证 borrowed stream 的 exact context，并让少数冷
  Driver 操作精确恢复调用线程状态，Context 同时 retain 对应 Driver primary `CUcontext`；两种 handle 不相互
  转型；
- ggml `e91ded11bdcd` / llama.cpp `e71b80510c84` 证明 device buffer、async copy、stream/event 与
  scheduler boundary 是 backend runtime；vLLM `51da0ca66c80`、SGLang `dcebe8c4733a` 证明 capture pool、
  stable address 与 replay lane 是 Graph/Invocation 冷状态；ds4 `f62ca29a3087` 证明 explicit stream、
  stream-owned scratch 和 capture-safe preallocation。CUDA factor 不吸收模型/KV/provider policy。

### 十二项审计答案

1. **context owner**：`cuda::backend()` 只返回静态 factory identity；每个 HYLE Context 取得一个 primary
   execution-context handle，并 retain 同 device Driver primary context。HYLE-owned Queue 用
   `cudaExecutionCtxStreamCreate(..., cudaStreamNonBlocking, priority=0)`；全部热命令传非空 stream。只有
   allocation/event-slot 创建、pointer/context validation 等冷调用允许 Driver push/pop，且精确恢复原
   `CUcontext`；
2. **通用/typed seam**：Compute v5 的 base vtable 增加每个 backend 都必须如实兑现的 sequence
   begin/finish、transfer properties、copy/fill 与 compatible wait；不支持路线明确返回
   `Unimplemented`，不能用 optional extension 让 Graph 猜基础顺序能力。CUDA stream/context/pointer、
   import/adopt、capture 与 native callback 仍只在 typed/object-local extension 及
   `<hyle/compute/backends/cuda.h>`；
3. **storage owner**：一个 move-only、pimpl-stable `cuda::MemoryResource` 拥有 device/pinned-host/managed 三种
   resource implementation并 retained exact Context。它返回普通 `mem::MemoryResourceRef`，Buffer/View layout
   不变；resource owner 必须晚于 Buffer 与 in-flight work 销毁，live allocation 违反顺序时 fail-fast；
4. **sync/async 分界**：Mem `read/write/copy/fill` 保持同步显式合同；Graph/direct async 只走 Compute
   `copy_async/fill_async`。pageable host memory 不偷偷 staging；CUDA H2D/D2H 只接受本 factor 的 pinned 或
   managed resource，D2D 要求同 device，peer route 必须明确报告；
5. **Completion/event**：Queue 冷路径按稳定 block 创建 `cudaEventDisableTiming` slots；
   `reserve_completions(n)` 保证至少 n 个同时 outstanding，未预留的普通 direct call 可在接纳任何 work 前
   按需增长。finish 每个 direct submit/Graph segment 只 record 一个 event。slot 在 Completion settle、后续
   acquire 扫描或 drain 时回收，generation 不在 slot 未终态/仍被句柄持有时复用；
6. **cancel**：pending CUDA work 不可抢占，`request_cancel()` 返回 `Unimplemented`；Graph cancellation 只
   停止尚未 admission 的 segment。event、storage、workspace 在真实终态前都不得复用；
7. **abandon/teardown**：丢弃 pending Completion 先做一次 non-blocking event query；仍 pending 时把其已
   预分配 intrusive node、terminal hint、lifetime 与 Queue retain 交给 factor-wide reaper。reaper 是一个用
   `platform_thread::Thread/Parker` 和 `util_core::SpinLock` 实现的延迟终态服务，不枚举/选择 backend，因而
   不是 process registry；普通保留句柄路径没有 callback/reaper traffic。Queue/Context 析构只 close，不
   host-wait；动态卸载 CUDA factor 前必须 drain，库终止阶段才停止并 join reaper；
8. **interop**：typed façade 提供 exact device ordinal、execution context、stream、validated byte pointer，
   owned/borrowed stream Queue 及 borrowed/adopted external allocation。borrowed owner 必须覆盖 Queue/
   Completion；adopt 只在全部校验成功后转移。default/null stream、另一 primary/green context 和错 device
   明确拒绝；普通 Compute 头不包含 CUDA；
9. **capture**：普通 `cuda::submit` 在 callback 前拒绝外部 active capture。Graph 的 CUDA extension 在
   `prepare_replay` 冷路径完成 begin/end/instantiate，且 bindings、workspace、event slots 已稳定；warm
   submit 只 `cudaGraphLaunch` 后 record segment event。v1 的 multi-queue capture 为 unsupported，prefer
   显式回退 commands，require 明确失败；capture 内不 query event、不加 host callback、不 allocation；
10. **build/package**：选中 Compute/Graph 产品或该 factor 时才声明
    `HYLE_COMPUTE_CUDA=OFF|ON|AUTO`，默认 OFF；OFF 不探测，ON 缺 13.3 Toolkit 硬失败，AUTO 找不到则
    `OPTIONAL_ABSENT`，且只供开发机探索，不签正式制品。使用 `find_package(CUDAToolkit 13.3)`，不
    FetchContent、不进入共享源码 cache，也不要求 HYLE 本体启用 CUDA language；只有 `.cu` probe/consumer
    自己启用。enabled target public 传递 `CUDA::cudart` 和 headers、private 使用 driver；安装包必须
    `find_dependency(CUDAToolkit 13.3)`。NVIDIA SDK/runtime/driver 由机器提供，HYLE 不复制进离线 source
    bundle，也不重新分发其许可；
11. **最小真机 probe**：copy-only 不能证明 typed native command seam。必须是 pinned host -> device async
    copy -> 一个 consumer `.cu` no-op/copy kernel -> pinned host async copy，校验数据和 pending/poll/wait/
    abandon；kernel 只验证 seam，不成为算子 provider；
12. **上游修正**：Compute v5 C55--C61/C64 是前置；其中 C59 还必须让 Completion abandon 把已保存的
    synchronous tail terminal hint 交给 backend，否则 partial-accepted 序列会丢错误。Mem 现有
    `MemoryResource/Buffer::adopt` 足够，不重开；`platform_thread`/`util_core` 只被复用，不重开。安装包缺少
    non-static/system `find_dependency` 登记是独立 build/package trigger，已由
    `build-cuda-system-sdk-projection-v1` 持有。

### 冻结能力与自然 C++ 形状

| 能力族 | 层级 | 冻结语义 | 当前 | 动作 |
|---|---|---|---|---|
| CUDA backend/device/context | 原子机制 | 显式多 device、primary execution context、无全局默认选择 | MISSING | FIX |
| ordered async Queue | 原子机制 | owned non-blocking stream；borrow/adopt exact primary-context stream | MISSING | FIX |
| event Completion | 原子机制 | cold/pregrown disable-timing slot、每 segment 一 event、late error | MISSING | FIX |
| abandoned-state retirement | 原子机制 | 无析构等待/host callback/逐项 heap；factor-wide reaper 只处理 abandon | MISSING | FIX |
| device/pinned/managed resource | 原子机制 | stable owner -> Mem resource；同步 resource op 与 async queue op 分离 | MISSING | FIX |
| async copy/fill/wait | 复合功能 | H2D/D2D/D2H/peer capability 可查询，无 pageable staging/host wait | MISSING | FIX |
| typed native submit | 复合功能 | callback 在 host enqueue 期间借用 stream，调用后即失效；无 type erasure allocation | MISSING | FIX |
| external storage/stream interop | 复合功能 | borrow/adopt 两条显式 ownership；pointer/context/device 校验 | MISSING | FIX |
| single-queue capture/replay | 复合功能 | Graph cold capture、warm launch，prefer/require 可报告 | MISSING | FIX |
| cuBLAS/CUTLASS/ggml kernels | provider | 独立 provider factor 选择/prepare | OUT_OF_SCOPE | ROUTE |
| green context/MPS/NCCL | 专项 runtime | 需要独立资源/进程/通信 policy | OUT_OF_SCOPE | DEFER |

公开形状冻结为：

```cpp
auto device = HYLE_TRY(cuda::device(0));
auto context = HYLE_TRY(Context::create(device));
auto queue = HYLE_TRY(context.create_queue());
HYLE_TRY(queue.reserve_completions(4));

auto device_memory = HYLE_TRY(
    cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice));
auto pinned_memory = HYLE_TRY(
    cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost));
auto input = HYLE_TRY(mem::Buffer::create(pinned_memory.ref(), bytes));
auto resident = HYLE_TRY(mem::Buffer::create(device_memory.ref(), bytes));

auto uploaded = HYLE_TRY(compute::copy_async(queue, resident.view(), input.view()));
HYLE_TRY(uploaded.wait());
auto resident_pointer = HYLE_TRY(cuda::pointer<float>(resident.view()));
auto launched = HYLE_TRY(cuda::submit(queue, [&](cudaStream_t stream) noexcept {
  return cuda::status(launch_my_kernel(stream, resident_pointer));
}));
HYLE_TRY(launched.wait());

auto borrowed = HYLE_TRY(cuda::borrow_stream(context, external_stream));
auto external = HYLE_TRY(cuda::ExternalBuffer::borrow(
    context, external_pointer, bytes));
```

`MemoryResource` move 只移动一个 pimpl pointer，已发出的 `MemoryResourceRef` 地址稳定。
`ExternalBuffer` 自带一份稳定 no-op/adopting resource 与普通 Mem Buffer，因此无需给 Mem 增加公开的
unsafe borrowed-view constructor。最终逐头测试与 guide 必须使用上述可编译 recipe，而不是伪代码逃逸。

### 有限实现 ledger

| ID | owner | 冻结改动 |
|---|---|---|
| CU10 | build/CUDA factor | 建立 optional factor 目录/target/header ownership，加入 Compute/Graph product roots；默认 OFF、显式 ON、开发用 AUTO 与 13.3+ discovery |
| CU11 | build/package 独立任务 | package registry 增加所有 linkage 适用的 `FIND_DEPENDENCIES`，installed CUDA feature 找到 Toolkit 后再载入 targets |
| CU12 | CUDA | Backend/Device properties、0/N/ordinal/multi-device、Runtime/Driver 错误映射与 primary context owner |
| CU13 | CUDA | owned non-blocking Queue、borrow/adopt stream、exact context/device/flags/owner validation |
| CU14 | Compute v5 + CUDA | sequence begin/finish、completion reserve/growth、disable-timing event state和每 segment 一次 record |
| CU15 | Compute v5 + CUDA | abandoned terminal hint handoff、intrusive reaper、late error、cancel/close/drain/device-loss teardown |
| CU16 | CUDA/Mem bridge | stable device/pinned/managed `MemoryResource`，sync resource operations、stats/trim truthful fallback |
| CU17 | CUDA | async copy/fill/compatible wait，range/domain/overlap/peer/pageable-host failure matrix |
| CU18 | CUDA | typed context/stream/pointer queries与 zero-allocation `cuda::submit` callback bridge |
| CU19 | CUDA | `ExternalBuffer` borrow/adopt 与 ownership/error injection/teardown tests |
| CU20 | CUDA/Graph | single-queue capture extension、cold instantiate、warm replay、prefer/require/report；multi-queue unsupported |
| CU21 | tests | 真机 H2D -> `.cu` kernel -> D2H、two queue wait、pending abandon、late failure、OOM/race/capture probes |
| CU22 | delivery/docs | source/static/shared、exact factor/product、OFF/ON diagnostics、installed SDK/driver/runtime/license/guide |
| CU23 | performance/red team | P0 bill、同 stream/event/copy 合同的必要 P1、终局 ownership/context/capture/build 红队 |

只授权 CU10--CU23。green context、NCCL、宽 kernel/provider、通用公开 Event/CommandList、默认 device/stream、
兼容 CUDA 12.x 和 multi-queue capture 不得顺手进入本轮。

### 生命周期、并发与 P0 账单

- Queue 是单 producer recording domain，但 Completion poll/wait 与 reaper 可并发；slot free-list 用既有
  `util_core::SpinLock`，每个 slot 从 free -> recording -> outstanding -> terminal/free 单向转换，复用前
  event 必须终态且旧 Completion 已 settle/abandon；
- Context drain 对**自己创建/导入的 Queue prefix**逐 queue 记录/等待内部 drain boundary，不调用会等待同
  device 其它库和 green context 的 `cudaExecutionCtxSynchronize`。borrowed stream 要求调用方在 HYLE
  recording interval 外部同步；HYLE 不拥有其上其它 producer 的顺序；
- `destroy_queue` 只在 QueueState 最后一引用时销毁 owned/adopted stream；pending Completion/reaper 会保活。
  borrowed stream 从不销毁。primary ref 在所有 Queue/resource state 之后释放；
- direct steady submit 的 host 成本为一次通用 admission、一次 extension call、consumer CUDA calls、一次
  `cudaEventRecord`；无 `cudaSetDevice`、context push/pop、HYLE heap、provider search、host callback 或
  host wait。Graph N commands 同一 segment 仍只有一次 record；reaper thread 在没有 abandoned pending
  work 时无限期 park，不按 timer 空醒；
- `reserve_completions(n)` 与 resource allocation 是冷路径，允许 O(n) event/metadata 与 vendor calls；copy/
  fill host enqueue O(1)，device work O(bytes)；reaper 成本只由用户主动 abandon pending handle 触发；
- P1 只比较相同 non-blocking stream、相同 pinned/device buffers、相同 event boundary 的 direct CUDA 调用与
  HYLE wrapper，固定 5% 非劣门。pageable vs pinned、kernel-only vs terminal Completion、默认 stream vs
  non-blocking stream或不同 capture 状态都是苹果/橘子，禁止裁决。

### 七问与停止点

1. 第三方：只有系统 CUDA Toolkit/driver；版本、缺席、传播与许可已明确，vendor 类型只在 CUDA header；
2. 功能：真异步 runtime、storage、interop、single-queue replay 已覆盖两条纵切；算子/模型/通信有明确 owner；
3. 内部复用：Core/Error/Compute/Mem/Util-core/Platform-thread 全部复用，无私造 Status/Buffer/Thread；
4. 性能：热路径账单与同合同 P1 已固定，不以裸 kernel 或旧 benchmark 签字；
5. 交付：纯 C++；Native FFI N/A；source/static/shared 都用同 typed façade；
6. build：可选 factor 与 system-SDK package gap 分 owner，OFF 不探测、不造 stub DLL；
7. 易用性：普通 Compute 用户不见 CUDA；CUDA 用户有短路径、明确 owner 和可编译 recipe，限制进入 guide。

CUDA factor 可进入 `AUDITED`。实施顺序必须是：先完成 Compute v5 C55--C61/C64 和 CU10/CU11，使全部
factor 同一提交序列可构造；再 CU12--CU19 真异步纵切；随后 Graph v6/CU20；candidate 后只运行一次
CU21--CU23。若 reaper 无法在 abandon 路径做到零 allocation/零 wait，或 installed package 不能自然传播
Toolkit，立即 `BLOCKED`，不得以 leaked state、hidden sync 或 opaque `void*` façade 绕过。

### 实施进度（非 terminal verdict）

- CU10--CU11 已形成候选：CUDA 是用户安装的 system SDK；`OFF` 不探测，`ON` 要求 13.3+，`AUTO`
  仅在开发配置中探测并清晰报告版本/路径或 optional-absent；package registry 可为所有 linkage 投影
  `find_dependency(CUDAToolkit 13.3)`；
- CU12--CU19 已形成 Windows 真机候选：显式 device/context、owned/borrowed/adopted stream、event
  Completion、idle-park/abandon-poll reaper、device/pinned/managed resource、同步与异步 transfer、typed
  submit/pointer、ExternalBuffer borrow/adopt 均已实现；
- CU20 与 CU21 的当前真机纵切已通过 RTX 4070/CUDA 13.3：`.cu` kernel、三种 resource、stream
  ownership、active external capture 拒绝、pending Completion abandon、Graph-owned single-queue
  capture/hybrid replay，以及同 Context 双 stream 的 event import。双 stream probe 不做 host wait：延迟
  producer store 后，consumer stream 通过 imported prefix 再 copy 并观察到正确值；
- CUDA target 的 compute/transfer lane 也已在 Graph 使用同一 device storage 资格化，未使用 lane 不产生假
  Queue 要求。一般 multi-stream capture 仍明确 unsupported；CU21 剩余 loss/race、CU22 三交付、CU23
  P0/P1/终局红队尚未完成，因此组件保持 `AUDITED`。
