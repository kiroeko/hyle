# HYLE compute/ 异构计算选型

> 本文档定下 HYLE `compute/` 模块（GPU/异构算力）的设计。上游约束见 [`hyle-scope.md`](./hyle-scope.md)（不特化、贴硬件）、调度对接见 `sched/`。GPU 现状盘点见 [`clay-asset-liability.md`](./clay-asset-liability.md) 第五节。

## 调查事实（真实数据）

1. **infra 本体零自有 GPU 计算**：`.cu/.cuh = 0`，唯一碰 CUDA 的是 `host_device.h`，纯宏门面（`__host__ __device__` 修饰符 + launch-bounds 宏 + `CUDA_KERNEL_ASSERT`，`host_device.h:5-75`），不含 kernel、不调 cuda runtime API。
2. **ggml 是跨多后端的 GPU 计算抽象**，统一在 `ggml-backend.h`。iron 里那份（stable-diffusion.cpp 依赖）含 `ggml-cuda`/`ggml-hip`/`ggml-metal`/`ggml-sycl`/`ggml-vulkan`/`ggml-kompute`/`ggml-blas`/`ggml-cann`/`ggml-musa`/`ggml-rpc`/`ggml-amx`，已有 backend-registry 架构。
3. **⚠️ 但 iron 里那份是旧快照（约 2024末–2025初）**：git 仅 `99d1082 update`，`ggml-aarch64.c` 仍在 src 顶层（新版已并入 `ggml-cpu`），**无 `ggml-webgpu`**。它是 sd.cpp 的冻结依赖、跟 sd.cpp 走，**不会为 HYLE 更新、会一直变老**。
4. **上游 ggml 已显著演进**：新增 WebGPU 后端（浏览器/跨平台，JIT shader）、OpenCL 后端（Adreno 等移动 GPU）。设备内存语义靠 `ggml_backend_dev_memory(dev,&free,&total)`（只给字节数）+ `enum ggml_backend_dev_type{CPU,GPU,IGPU,ACCEL,META}`（IGPU=集显走 host 内存=统一内存信号）+ `dev_caps{async,host_buffer,buffer_from_host_ptr,events}` + `ggml_backend_buffer_is_host()` 共同表达——**不是单一"统一 vs 独立"开关**（核对真实 `ggml-backend.h`）。
5. **slate 自带 `vulkan` 库**（+ opengl + embree）——shader compute 底子现成。
6. **2026 格局**：CUDA 仍主导但锁死 NVIDIA；Vulkan compute 在 llama.cpp 里与 CUDA/ROCm 正面竞争（FOSDEM 2026）；chipStar 把 CUDA/HIP 编译到 OpenCL/SPIR-V；SYCL 推统一后端。跨厂商 GPGPU 正从"只有 CUDA"转向"Vulkan compute / SYCL"。

## 核心判断

### 1. 借 ggml 拿全平台后端 —— 但对接上游、用 adapter 隔离，不焊死在 vendored 快照上
ggml-backend 已跑通 CUDA/HIP/Metal/Vulkan/SYCL，HYLE 不重造这个轮子。但 iron 里那份是 sd.cpp 的冻结旧快照（会一直变老、接口已演进），**不能直接依赖**。正确依赖方向：
- HYLE `compute/` 上层抽象（device/stream/buffer/kernel）**自有且稳定，不暴露任何 ggml 类型**。
- ggml 降级为"**可替换的算子库供应商之一**"，只在一个 `ggml_bridge` adapter 文件里对接，且对接**上游 pin 的较新版本**（自己 vendor/submodule），不是 iron 那份。
- ggml 升级、`ggml-backend.h` 接口变动（已在变），**只动 adapter 一个文件，不动 HYLE 地基**。
- iron 那份只当"现在能立刻跑通验证"的参照物，非长期依赖。

教训：任何 vendored 快照都会变老（这份已老），所以降低耦合、把 ggml 当可换后端，而非把 HYLE 焊在它身上。

### 2. 但 ggml-backend 是"张量算子图"专用，HYLE 要更底层
ggml 的调度单元是 `ggml_tensor` + 固定算子集，不是通用 GPU 计算。HYLE 要 **device/stream/buffer/kernel** 抽象，能跑任意 kernel。所以 `compute/` 分两层：

```
compute/
├─ 上层（HYLE 自有，平台无关）
│   device   — 枚举/选择 GPU/CPU 设备
│   stream   — 异步执行队列（= 可调度资源）
│   buffer   — 设备内存 + host/device 传输 + pinned/unified
│   kernel   — 任意 kernel 的提交/启动抽象
└─ 下层后端（复用 ggml 实现作起点 + 可直调原生 API）
    cuda / hip / metal / vulkan(compute) / sycl / cpu(simd fallback)
```
上层之上再挂一个 `ggml-bridge`：把 ggml-backend 当成"算子库"接进来，需要现成 AI 算子时走 ggml，需要自定义 kernel 时走 HYLE 原生 stream/kernel。

> **校正（对照真实 ggml-backend.h）**：ggml-backend **已经有** device/stream/buffer/buffer_type/event/sched 全套抽象。所以 HYLE 真正"净新"的只有两处：**①任意 kernel 启动**（ggml 是固定算子集 + `graph_compute(cgraph)`，无 raw launch）；**②host 侧 completion→fiber 唤醒**（ggml 的 `synchronize` 是阻塞式，event 仅 GPU-GPU 跨后端同步，无"算完回调唤醒 host"）。device/buffer 枚举应**薄包 ggml-backend**，避免双层抽象税；只在 kernel-launch + fiber-completion 处真正自造。可直接照镜 ggml 的 `dev_caps{async,host_buffer,buffer_from_host_ptr,events}` 能力查询结构。
>
> **深读补充（buffer_type 是 device buffer 的范本，归 compute/ 不归 mem/）**：ggml 的 `buffer_type_i{get_alignment/get_max_size/get_alloc_size/is_host}` 是 HYLE **compute/ device buffer** 该照镜的 trait——且 ggml 里 **host/device/pinned/split 不是枚举开关，而是各立一个 buffer_type 实例**（`is_host()` 是唯一 host 布尔判据）。**一种内存模型 = 一个 buffer_type 实例**(不是 `switch(MemKind)`),这条洞见落在 **compute/ 的 device buffer**(自有所有权 + stream 绑定),**不是 mem/**:device 内存走显式 copy/command-submit,不经 mem/ 的 CPU allocator 抽象(mem/ 只管 host RAM,见下"内存模型"节 + mem-design)。非标能力（`set_n_threads`、`split_buffer_type` 等）经 `get_proc_address` 探测取得（不破 vtable）——HYLE `ggml_bridge` 拿这些扩展走同一机制。
> **张力收口**："buffer 薄包 ggml"与"不暴露 ggml 类型"有张力：**枚举/能力查询薄包**（照镜 buft trait），**内存所有权 + stream 绑定自造**，bridge 只在算子边界做 tensor↔buffer 翻译。另注意 ggml sched 把 INPUT 强制落 CPU、末位 backend 必须是 CPU（`ggml-backend.cpp:903/1736` ASSERT）——这是 ggml 的兜底耦合，**HYLE 自有 stream 调度不要继承**。

### 3. shader 计算是一等公民，不只绑 CUDA
**只押 CUDA = 锁死 NVIDIA，违背 HYLE 通用定位。** Vulkan compute（SPIR-V）跨厂商——AMD/Intel/Apple(MoltenVK)/NVIDIA 全能跑，`ggml-kompute` 已用 `.comp` 示范。CUDA 与 Vulkan compute **平级**：CUDA 走成熟生态/性能上限，Vulkan compute 走"一份 SPIR-V 全平台"。

### 4. GPU 完成 = 又一种 completion source（与 io/ 同构）
这是 `compute/` 与 `sched/`/`io/` 统一的关键：
- `cudaEvent` / Vulkan `fence`·`semaphore` / Metal command-buffer completion handler —— 全是**完成信号**。
- 和 io_uring/IOCP 的 CQ 一样，接进 fiber 调度器作为 completion source。
- fiber `co_await gpu_kernel(...)`：提交到 GPU stream → 挂起 → GPU 算完经 event/fence 唤醒。与 `co_await async_read()` 完全同构。
- **`sched/` 统一调度异构资源的真义**：CPU 线程、GPU stream、网络 IO 都是产生 completion 的可调度资源，一套挂起/唤醒机制。

> **第二接缝（诚实承认，校正自"只通过 adapter / 不暴露 ggml 类型"）**：fiber 唤醒需要原生 handle（`cudaStreamAddCallback`/`cudaLaunchHostFunc`、Vulkan fence），而 ggml-backend 恰好把这些藏在不透明指针后、且只提供阻塞式 `synchronize`。所以 **GPU→fiber 唤醒这条路必然绕过 ggml-backend、直连原生**——它是 `ggml_bridge` adapter **管不住的第二接缝**。结论：算子/设备枚举走 adapter；completion 唤醒走原生 handle。这不矛盾，是两条接缝各司其职。
>
> **深读 ggml 源码加固此判断（铁证）**：ggml-cuda 里唯一的 `cudaLaunchHostFunc` 被 `#if 0` 注释掉、标 "untested"，活路径是 `GGML_ABORT("fatal error")`（`ggml-cuda.cu:4539-4548`）。`ggml_backend_event_*` 只做 GPU↔GPU 跨 stream 同步（`cudaEventRecord`/`cudaStreamWaitEvent`），`event_synchronize` 是 host 阻塞等、**无回调**。即 **ggml 故意没接 host completion callback，连半成品都没有**——HYLE 的 completion 脊在 GPU 侧是 100% 自造，不是"补 ggml 的缺"，是 ggml 结构性不提供。这反而印证：completion 自造是刚需、非镀金。

## 跨平台后端矩阵

| 后端 | 厂商/平台 | kernel/shader 语言 | HYLE 怎么拿 | 定位 |
|---|---|---|---|---|
| **CUDA** | NVIDIA | CUDA C / PTX | 原生 or 复用 `ggml-cuda` | 性能上限、生态最成熟 |
| **HIP/ROCm** | AMD | HIP（近似 CUDA）| 复用 `ggml-hip` | AMD 卡主路 |
| **Metal** | Apple | MSL | 复用 `ggml-metal` | mac/iOS 唯一原生路 |
| **Vulkan compute** | 全厂商 | SPIR-V（GLSL `.comp`）| slate 自带 vulkan + 参考 `ggml-kompute` | **跨厂商一等公民，一份 shader 全平台** |
| **WebGPU** | 浏览器/跨平台 | WGSL/SPIR-V | 对接上游 ggml-webgpu（iron 那份没有）| 端侧/浏览器，2026 新增 |
| **SYCL** | Intel/跨 | SYCL C++ | 复用 `ggml-sycl`（可选）| Intel GPU / 实验 |
| **CPU fallback** | 全 | HYLE `simd/` | 自有 | 无 GPU 时退化 |

## CUDA vs shader：怎么取舍

- **要极限性能 / 已知跑 N 卡**（训练、大模型推理）→ CUDA（或 HIP 对 AMD）。tensor core、cuBLAS/cuDNN、生态算子最全。
- **要"一次写、全平台跑" / 不想锁厂商**（agent 端侧、异构集群、消费级显卡）→ Vulkan compute（SPIR-V）。一份 `.comp` 编译成 SPIR-V，N/AMD/Intel/Apple 都吃。
- **HYLE 策略**：上层抽象屏蔽差异，两条后端都建。选择走**编译期优先、运行期一次冷跳兜底**的统一 dispatch 模型（详见 [`coding-conventions.md` → 能力 dispatch](./coding-conventions.md)）——构建期可只编目标后端（llama.cpp 式 `-DGGML_CUDA=ON` 按机器定死，零 dispatch）；发布二进制则启动探测一次、冷跳进可用后端（CPU(simd) → Vulkan compute → CUDA/HIP/Metal 原生）。**后端选择是冷路径，kernel 内部满速无间接。**

## 借 vs 造

| 层 | 借/造 | 说明 |
|---|---|---|
| AI 算子集（matmul/attention/conv…）| ✅ 借 ggml（上游版）| 各后端算子，不重写；**对接上游 pin 版，非 iron 旧快照** |
| 跨后端"算子图"调度 | ✅ 借 ggml-backend | 现成跑通，封在 `ggml_bridge` adapter |
| **CPU 算子并行注入** | 🔧 注入 HYLE 线程池 | **`ggml_backend_cpu_set_threadpool`（`ggml-cpu.h:135`）把 HYLE 调度器喂给 ggml**，不让 ggml 自起线程——避免双调度器（同理 oneDNN）。这是"借算子不借运行时"的落地点 |
| **ggml 版本/接口隔离** | 🔧 自造 adapter | `ggml_bridge` 是唯一耦合点；ggml 升级/接口变只动这一文件 |
| 通用 device/stream/buffer/kernel 抽象 | 🔧 自造 | ggml 没有的通用层，绑 HYLE sched，**不暴露 ggml 类型** |
| GPU completion → fiber 唤醒 | 🔧 自造 | event/fence 接 sched，io/ 同款机制 |
| Vulkan compute 自定义 kernel | 🔧 自造（参考 kompute）| 一等公民路径，slate vulkan 打底 |
| 设备内存分配 | 🔧 compute/ 自有 | device/pinned/unified buffer 各立 ggml 式 buffer_type 实例,所有权+stream 绑定在 compute/;**不走 mem/ 的 CPU allocator 抽象**(host↔device 是显式 copy/command-submit)|

**原则**：算子和跨后端调度借 ggml（别重造）、但**只通过 `ggml_bridge` adapter 对接上游 pin 版、不焊死 vendored 快照**；通用抽象层 + 与 sched 的 completion 统一自造（ggml 给不了），且 HYLE 抽象不暴露 ggml 类型。

## 模块形状

```
compute/
├─ device.h        设备枚举/能力查询/选择
├─ stream.h        异步执行队列（可调度资源，产生 completion）
├─ buffer.h        设备 buffer:自管 device/pinned/unified 内存 + 绑 stream(ggml buft 式:枚举/能力查询薄包,所有权自造)。host↔device = 显式 copy(src,dst,stream),同 coherent 内存硬件无需拷贝。**不走 mem/ 的 allocator 抽象**(mem/ 只管 host RAM,见 mem-design「device 内存」节)
├─ kernel.h        任意 kernel 提交/启动抽象
├─ backend/
│  ├─ cuda.cc      原生 or 桥 ggml-cuda
│  ├─ hip.cc       桥 ggml-hip
│  ├─ metal.cc     桥 ggml-metal
│  ├─ vulkan.cc    Vulkan compute（SPIR-V），自定义 kernel 主路
│  ├─ sycl.cc      桥 ggml-sycl（可选）
│  └─ cpu.cc       simd/ fallback
├─ ggml_bridge.h   把 ggml-backend 当算子库接入
        ↓ completion（event/fence）绑定
     sched/         GPU stream 完成 → 唤醒 fiber（与 io/ 同机制）
```

## compute-01 开工前研究结论（2026-07-05,CUDA 先行）

> 破土前先研究 SOTA 定形接口(同 io proactor 的纪律),把下面几条从"待深挖"提为已决。

### 1. 环境现实:CUDA 13.3 双通道就绪(Windows MSVC + WSL,均真编真跑 GPU)

实测本机(**2026-07-06,全链路统一到 CUDA 13.3**——老 12.x + 13.2 均已清,toolkit/cudart/驱动上限/cuDNN 构建档/TRT 构建档全一致):
- **驱动**:610.62(WDDM),`nvidia-smi` CUDA UMD 上限 **13.3**;GPU **RTX 4070 / sm_89 / 12.9GB / 46 SM**。
- **WSL(✅ 已完全就绪,优先在此开发,原生 ext4 干净)**:`nvcc 13.3.73` 在 `/usr/local/cuda-13.3`(软链 `/usr/local/cuda` 已重指、在 PATH),cudart 13030,driver 直通 `/usr/lib/wsl/lib/libcuda.so`。**cuDNN 9.24 + TensorRT 11.1.0.106(+cuda13.3)已装**(apt `cuda/repos/ubuntu2404` 源;apt pin 只钉驱动/内核模块防毁直通,不钉 toolkit)。
- **Windows**:cuDNN 装到独立 `C:\Program Files\NVIDIA\CUDNN\v9.24`(用其 `13.3/` 变体)、TensorRT zip 解压到 `C:\public\TensorRT-11.1.0.106`(include/lib/bin);toolkit 正从 13.2 换 13.3。CMake 走 `-I`/`-L` 指 cuDNN/TRT,不依赖 PATH;运行时 dll 才需 PATH。⚠️ 某 shell 看不到 nvcc/CUDA_PATH = 安装器前启动的 stale-env 假象,新开 shell 即正常。
- **端到端已验证(2026-07-06,13.3)**:`nvcc -arch=sm_89` 编 saxpy + **event-poll completion**(`cudaEventRecord`+`cudaEventQuery` 轮询,正是下面 §2 的 reap 模型)+ 链接 cuDNN(v92400)/TRT(v110100)在 WSL RTX 4070 真跑通(cudart 13030)。**整条工具链 + 计划的完成机制 + cuDNN/TRT 链接均已真机去险。**
- **cuDNN/TensorRT 是加速档**:compute-01~03 的 device/stream/completion 脊 + raw kernel launch **不需要**它们(cuDNN=卷积/attention 算子、TRT=推理图优化器,更上层;借 ggml 算子即可)。装齐是为后续算子/推理路径备好,不阻塞开工。

**结论**:CUDA 后端可 Windows/MSVC + WSL/nvcc **双通道真编真测**(不再是 Windows-only)。跨平台"一份代码"仍由上层抽象保证;CPU fallback(simd)双平台;mac 无 NVIDIA GPU → CUDA 后端在 mac `#if` 门控编不进(Metal/Vulkan 后端才是 mac 路,后续)。**WSL TSan 覆盖 compute host 侧逻辑(stream/completion 记账),GPU kernel 本身不进 TSan。**

### 2. GPU completion→fiber:走 **event 轮询(reap 模型)**,不走 callback

**决策:`cudaEventRecord` + `cudaEventQuery` 轮询,由 shard worker 收割,而非 `cudaLaunchHostFunc`/`cudaStreamAddCallback` 回调。** 理由(研究铁证):

- **两种 callback 都禁止在回调里调任何 CUDA API**(deadlock:回调可能等自己的完成;NVIDIA 论坛多例 host-func 死锁报告)。虽然我们的 `Waker::wake()` 不碰 CUDA API(只重注入 fiber),回调仍跑在 **CUDA 内部线程**上,把跨线程 wake 的时机交给驱动、不可控。
- **ggml-cuda 的铁证**:唯一的 `cudaLaunchHostFunc` 被 `#if 0` 注释标 "untested",活路径是 `GGML_ABORT`——即成熟库也不用 callback 接 host completion。
- **轮询 = io_uring reap 模型的同构**:提交 kernel 后 `cudaEventRecord(ev, stream)`;worker 在**已有的 poll hook**(`Scheduler::set_io_poll` seam,scheduler.cc:527 抽干热路径 + :587 park 边界带 timeout)里 `cudaEventQuery(ev)`,`cudaSuccess` 即 `waker.wake()`。**wake 跑在自己的 worker 上(同 io),不在 CUDA 线程**;GPU 成为 worker 循环收割的**第三种 completion source**(CPU/io/GPU 一套挂起-唤醒),正是 sched-design §3 与 compute-design §4 的论点落地。
- **避免每次 poll 空转**:沿用 io 的 park 边界策略——有在途 GPU op 才轮询;`cudaEventQuery` 是轻量非阻塞查询。真实节奏调优(轮询间隔 vs 延迟)留实测,不预设。

### 3. completion awaitable 形状 = 照镜 io 的 `IoOperation`

`co_await stream.launch(...)`/`co_await event` 的 awaitable **完全照镜 net.h 的 IoOperation 模式**(net.h:66-90):栈 slot 存 `{Waker, result}` 活在挂起 fiber 栈、`await_suspend` 捕获 `current_waker()` + 记 pending event + fiber-yield、worker poll 到 event done 后填 result + `wake()`、`await_resume` 返回。**同一套 stack-slot 生命周期契约**(fiber 栈冻结到 wake,base op 靠 drain-to-completion)。shard-per-core:stream 绑 owner worker,submit/poll/resume 同核,免锁(同 io plan A)。

### 4. device/buffer 枚举薄包、kernel-launch + completion 自造(§2/§4 已定,不重复)

## compute-01 已落地（2026-07-06，device/stream/event 脊 + event-poll 原语）

> 破土第一步落地纪要（细节以代码为准：`include/hyle/compute/{types,device,stream}.h` + `src/compute/{compute.cc,backend.h,host_backend.cc,cuda_backend.cu}`）。

- **device/stream/event 抽象 + event 轮询原语**（§2 reap 模型的最小兑现）：`Event::query()→kReady/kNotReady` 直接是 `cudaEventQuery` 的 `cudaSuccess`/`cudaErrorNotReady`（照镜 CCCL `event_ref::is_done`）。双平台真机验证：Win MSVC + WSL nvcc 各 10/10，`GpuEventPollsToReady` 在 RTX 4070（sm_89）真跑 NotReady→Ready 轮询 + 显存 H2D/D2H 往返数据校验（非 skip）；WSL TSan host-only 零 race。
- **手动虚表分派 + 组合优于包含**（conventions §4/§10）：`ComputeBackend` 是纯 fn-ptr 结构（同 `error_handler_t`），每后端一个独立 TU 填一张 vtable（`cuda_vtable()`/`host_vtable()`），**非**上帝类 `switch(backend)`。加一个后端（HIP/Metal/Vulkan）= 加一个 TU + 一张表，0 调用点改动。device 全局表在 `compute.cc` 冷路径构建（GPU 优先、host 恒在末位保证非空）。
- **无 pimpl / 无堆分配**（性能）：`Stream`/`Event` 内联 `{const ComputeBackend* vt, void* native, int ordinal}` 三字；公开头靠前向声明 `struct ComputeBackend` 不泄漏 `cudaStream_t`。`query()` 热路径（compute-03 worker 每轮 poll）= 一次 vtable 间接（后端多态的不可约成本），无 impl_ 解引用。
- **CUDA 可选依赖 + C++ 方言现实**：`find_package(CUDAToolkit)`→`HYLE_COMPUTE_CUDA` 门控（仿 OpenSSL/tracy 可选能力模式；无 CUDA 只编 host、零依赖，mac/CI 可配）。**nvcc 13.3 转发 `-std=c++23` 给 host 编译器时忽略之**（实测 warning "flag not supported with the configured host compiler"；与 CMake 版本无关——CMake 4.1 已支持 CUDA23 方言，是 nvcc 前端硬限制）。故拆出轻量 `types.h`（`DeviceType`/`DeviceCaps`/`EventStatus`，不含 `error/result.h`），让 `cuda_backend.cu` 编译时不碰 `std::expected`（C++23 库类型），CUDA TU 用 C++20 方言、host C++ TU 仍 C++23。
- **host fallback = compute-03 的确定性测试后端**：inline 执行、event 恒 Ready。它验证 completion 契约在无 GPU 时也编译可测（同 io epoll fallback 的角色）。

**compute-01 明确不做**（留后续里程碑，但接口已按下面想明白的方案对齐——不是"以后再说"）：GPU completion 接 sched worker loop（`co_await stream`）留 **compute-03**。device buffer 留 compute-02，kernel launch 留 compute-04。

## compute-03 GPU completion 接 sched：方案已定（2026-07-06 深调研 ggml/CCCL/seastar 源码）

> compute 是新支柱，接口对不对取决于 completion 怎么接 sched——这个必须**开工前想透**，不能拖到 compute-03 才发现 compute-01 的 Event 形状错了。深读源码后定案如下（compute-01 的 `Event::query()` 轮询 + `Event::sync()` 阻塞两个原语正是为这两条路准备的，接口已验证够用）。

**completion 通知只能走轮询/阻塞，不能走 callback（铁证）**：ggml 全库对 GPU 完成**只有阻塞 `cudaStreamSynchronize`/`cudaEventSynchronize`**、**零 `cudaEventQuery`**（它是同步执行模型，无 fiber 要唤醒）；跨端等待里唯一的 `cudaLaunchHostFunc` 被 `#if 0` 注掉标 untested，旁边直接 `GGML_ABORT`（ggml-cuda.cu:4539）。CCCL 把 host callback 契约标为"回调内禁调任何 CUDA API、禁依赖阻塞在异步 work 的线程"（host_launch.h:72），CUDASTF 为让 callback 落到指定线程不得不起专用 progress 线程 + GPU 侧 busy-spin kernel。**结论：callback 无法安全唤醒 fiber（唤醒涉及锁/futex，与驱动内部锁易死锁），路线否决。** CCCL 只提供 `event_ref::is_done()`（= `cudaEventQuery`）作非阻塞 poll 一等公民，且 **CUDA 无 `EventQueryMany` 批量原语**——只能逐 event 查。

**两条落地路线（compute-03 首版走 A，B 是后续优化）**：
- **方案 A（首版，零改 sched，风险最低）**：GPU 完成走**独立 poller 线程**阻塞 `cudaEventSynchronize`（或轻量轮询一组 in-flight event），完成时 `Waker::wake()` 把 fiber 重新注入 + `set_io_wake` 已注册的 wake 打断 worker 的 park。GPU **不占** worker 的 park 边界，和任何外部 completion 一样纯靠 Waker 从外部注入——**完全复用现有 Waker / io 的 `IoOperation`+`complete()` 模式 / `net_timeout` 的 refcount 堆 slot（GPU op 可能永不完成→需 timed/heap slot 非栈 slot）/ JoinHandle / spawn_blocking（卸载阻塞 sync）**，sched 一行不改。GPU event awaitable 照抄 `io::RecvOp`（`CaptureWaker`→arm event→`SuspendFiber`）。
- **方案 B（后续，省 poller 线程，需改 sched）**：worker 在 park 时**同时**轮询 GPU event + 阻塞 io ring。需把 `set_io_poll`/`set_io_wake` 的**单槽泛化成多 completion source 列表**（park 前依次 poll、取最小 timeout），并抄 **seastar 的 `try_enter_interrupt_mode` veto 协议**（reactor.cc:3477/2937）：GPU source 有在途 event 就**否决 worker 深睡**、继续和 io busy-poll（此时轮询几乎免费，反正没睡）；无在途 GPU 工作才放行 io_uring `GETEVENTS` 深睡。这样**只在既有 io 又有 GPU 在途时才轮询，纯 io 场景零浪费**——正是"可阻塞源 io + 只能轮询源 GPU 共存于一个 park 边界"的教科书解。

**为何 compute-01 接口无需改**：方案 A 用 `Event::sync()`（poller 线程阻塞），方案 B 用 `Event::query()`（worker 轮询）——两个原语 compute-01 都已提供并双平台真机验证。选 A 还是 B 是 compute-03 的调度集成细节，不回头改 Event/Stream 形状。

## 待深挖（下一步）

1. **kernel 表示的统一**——device/stream/kernel 接口怎么同时容纳"原生 CUDA kernel"(`cuLaunchKernel` + PTX/cubin)和"SPIR-V compute shader"两种 kernel 表示(compute-04/06;compute-01 只做 stream/completion 脊,先不碰 launch 抽象)。
2. ~~**GPU completion 接 sched**~~ **✅ 决策已定**（见上"compute-03 GPU completion 接 sched"节：方案 A 零改 sched 复用 Waker/IoOperation/net_timeout；方案 B seastar veto 协议为后续）+ compute-01 已把 event-poll 原语落地。
3. **buffer 与 mem/ 的边界**——host/device 传输、pinned、unified memory;§2 已定"device 内存归 compute/ 走显式 copy、不进 mem/ allocator",compute-02 落具体 buffer_type 实例形状。**注**：compute-01 的 `Stream::device_alloc`/`memcpy_async` 是脊验证的**最小**辅助（host 用 `std::malloc`、CUDA 用 `cudaMalloc`），compute-02 的 buffer 抽象取代它——届时 pinned **host staging**（`cudaMallocHost` 分配的是 host RAM）可包成 `mem::Allocator` 复用（`mem::MemoryArena` over `cudaMallocHost` 的 slab = pinned 块池，agent 调研已确认机制就位）；显存 `cudaMalloc` 仍 compute 自有、**不**走 mem。
4. **Vulkan compute 自定义 kernel 的工程链**——GLSL/`.comp` → SPIR-V 编译、运行时加载、与 CUDA kernel 的统一启动 API(compute-06)。**参考库 = kompute**(`C:\public\project-hyle\kompute`,见下"compute-06 kompute 映射"节已定该抄什么)。

## compute-06 kompute 映射：Vulkan 后端该抄什么/丢什么（2026-07-06 深读 kompute 源码）

> compute-06 的 Vulkan compute 后端以 **kompute** 为骨架参考。kompute 是理想参考——**纯 Vulkan、零隐藏运行时**（全 `src/` grep 无 `std::thread`/事件循环/condition_variable，唯一同步是可选的 submit mutex + 每 Sequence 一个 `vk::Fence`），不违反 HYLE"不引入第二个调度器"铁律。整体照抄其 Vulkan 样板，**唯一实质改动是把 fence 的阻塞 await 换成非阻塞轮询接 fiber**。

**该抄的三块不可省样板**（映射到 HYLE compute 抽象）：
1. **Vulkan 初始化链 → HYLE `Device`（Vulkan 后端的 vtable 创建路径）**：instance → 物理设备枚举 → 找 compute queue family（遍历 `eCompute`）→ logical device + queue → 能力/扩展查询。`Manager.cpp:174-496`。**保留 Apple/MoltenVK portability 分支**（`Manager.cpp:197-204`——跨厂商一等公民的前提，一份 SPIR-V 全平台跑）。
2. **SPIR-V → pipeline 全套 → HYLE `Kernel`（compute-04 的 Vulkan 实现）**：shader module 加载 SPIR-V → descriptor pool/layout/set 三件套 + `updateDescriptorSets` → pipeline layout（含 push constant range）+ compute pipeline + pipeline cache → 录制期 `bindPipeline`/`bindDescriptorSets`/`pushConstants`/`dispatch(x,y,z)`。`Algorithm.cpp:128-380`。
3. **device buffer + memory-type 选择 + staging 双缓冲 + barrier 编排 → HYLE `Buffer`（compute-02 的 Vulkan 实现）**：device-local primary buffer + host-visible staging buffer 双缓冲；手动扫 `memoryTypeBits` 找 memory type（`Tensor.cpp:396-438`，必抄样板）；传输不是独立 API 而是**录进 command buffer 的 op**（copy staging↔device 前后夹 `pipelineBarrier` buffer memory barrier，`Tensor.cpp:139-251`）；host 访问 `mapMemory` + coherent 内存免 flush。

**completion 机制——抄构造、换等待（接 fiber 的关键）**：
- kompute submit：`evalAsync`（`Sequence.cpp:112-141`）`resetFences` → `queue.submit(submitInfo, mFence)` 立即返回 = HYLE `Stream` submit 产生 completion。**照抄。**
- kompute await：`evalAwait`（`Sequence.cpp:163-187`）`waitForFences(...UINT64_MAX)` **阻塞** = **必须换掉**。换成 `device.getFenceStatus(fence)`（非阻塞，返 `eSuccess`/`eNotReady`）——**这正是 HYLE `Event::query()` 的语义,`vk::Fence` 就是 Vulkan 后端 `Event` 的底层 native handle**。轮询挂进 fiber 调度器（compute-03 的方案 A poller 线程 / 方案 B worker veto 轮询）。
- **关键印证**：`cudaEventQuery`（CUDA）和 `vkGetFenceStatus`（Vulkan）是**同一个 Ready/NotReady 轮询原语**——compute-01 的 `Event::query()` 抽象对两个后端都成立，后端多态在这一层是对的（compute-01 的设计经此验证）。

**该丢的（kompute 作为独立库的包袱，HYLE 不要）**：
- 阻塞 `waitForFences` → 换非阻塞 `getFenceStatus`（上述）。
- `shared_ptr`/`mManaged*` weak_ptr 所有权全家桶（`Manager.cpp:110-141`）→ HYLE 用手动虚表 + 自己的 Buffer/生命周期。
- spdlog/fmt logger、Python 绑定、debug report callback、timestamp query pool（除非要 profiling）。
- `OpBase` 虚基类 op 对象化（`OpBase.hpp` + `operations/*`）→ HYLE 已有 `Kernel`/`Stream` 抽象，不需要这层 op 对象；kompute 的三段式 record/preEval/postEval 语义可参考（barrier→dispatch→回读），但不抄它的 op 类层次。
- begin/end 录制状态机（`Sequence.cpp:40-94`）可简化为按需一次录制。

**SPIR-V 工程链（compute-06 需自己解决，kompute 未覆盖的）**：`.comp` GLSL → SPIR-V 的编译（`glslangValidator`/`shaderc` 离线编译进二进制,还是运行时编译）+ SPIR-V 加载。这是 compute-06 相对 kompute 要补的一环（kompute 假定调用方给编好的 SPIR-V）。
5. ~~**`cudaSetDevice` 的 per-thread 上下文 vs fiber 跨 worker 迁移**~~ **✅ 已解决（2026-07-06 深调研 ggml/CCCL 源码，推翻原顾虑）**：CUDA stream/event 的**操作**（record/query/synchronize/wait/free）**不依赖调用线程的 current device**——handle 内在绑定创建时的 context。铁证：ggml 的 event record/wait/sync 前**都不** set device（ggml-cuda.cu:4527/4533/5510），CCCL 刻意用 driver API 做 stream 操作正是为绕开 runtime API "context 栈空时误 push dev0"（stream_ref.h:189 注释）。current device 只影响**创建**（`cudaMalloc`/`StreamCreate`/`EventCreate`）与 **kernel launch**。**故 fiber 跨 worker 迁移后操作同一 stream 安全，不需要 shard-per-core 绑核。** compute-01 后端据此修正：`set_device()` helper（照 ggml-cuda.cu:109 的 get-if-differ 规避冗余 set）**只在创建站点调用**，reap/poll/操作路径 set-free。绑核降为**可选性能优化**（NUMA/context 局部性），非正确性必需。

## 参考

- CUDA vs ROCm vs Vulkan vs Metal: GPU Compute in 2026 — orchestrator.dev
- AMD GPU Acceleration: ROCm, HIP, Vulkan, OpenCL Guide (2025)
- IWOCL 2026: chipStar — CUDA/HIP on OpenCL/SPIR-V
- FOSDEM 2026: Vulkan API for Machine Learning? Competing with CUDA and ROCm in llama.cpp
- AdaptiveCpp (hipSYCL) single-pass SYCL
