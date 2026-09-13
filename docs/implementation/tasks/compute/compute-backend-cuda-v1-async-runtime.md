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
