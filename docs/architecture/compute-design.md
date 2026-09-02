# HYLE compute/ 异构计算选型

> **V2 状态：`INCOMPLETE / NOT_REVIEWED`。** 本页是半成品阶段的调查与设计历史，不是冻结的
> 产品架构或性能证书。真实推理消费者和前置模块可能改变其类型；只在 compute 轮次重写。

> **Storage 边界更新（2026-09-02，Mem v2）：**下文 2026-07 的 Compute-owned `Buffer`/
> `BufferType` 方案已被实际迁移取代。storage ownership、resource/property、`Buffer`/view、同步
> range operation 与 transient reuse 统一归 `hyle::mem`；Compute 只提供 device/host-pinned/
> managed `MemoryResourceRef` provider、Device identity、Stream/Event ordering 和 typed kernel/async
> adapter。当前代码入口是 `hyle/compute/memory.h` + `hyle/mem/buffer.h`，不存在旧
> `hyle/compute/buffer.h`。本页后续历史调查若仍提到“Compute Buffer”，以本段为准。

> 本文档定下 HYLE `compute/` 模块（GPU/异构算力）的设计。上游约束见 [`hyle-scope.md`](hyle-scope.md)（不特化、贴硬件）、调度对接见 `sched/`。GPU 现状盘点见 [`clay-asset-liability.md`](clay-asset-liability.md) 第五节。

> **当前收口原则（2026-09-02 更新）：**公开形状由可复述的消费者场景和跨后端合同决定，benchmark 只验证某个实现选择，不反向定义 API。Compute 的主轴是 `Device/Stream/Event/Kernel`，storage 参数统一使用 Mem `Buffer/View`；合批服务多算子链，record/replay 服务固定 bindings 的迭代负载，timing event 服务 profiling。一次 tiny-saxpy 的 `ns/op` 不能证明这些形状，更不能单独推出新抽象。当前 Vulkan 多个逻辑 Stream 仍共享一个 `VkQueue`，正确性成立但会保守跨 Stream 排队；真正独立并发需消费者工况 + timeline semaphore/多队列方案及端到端数据后再承诺。

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
- HYLE `compute/` 上层抽象（device/stream/event/kernel + Mem storage）**自有且稳定，不暴露任何 ggml 类型**。
- ggml 降级为"**可替换的算子库供应商之一**"，只在一个 `ggml_bridge` adapter 文件里对接，且对接**上游 pin 的较新版本**（自己 vendor/submodule），不是 iron 那份。
- ggml 升级、`ggml-backend.h` 接口变动（已在变），**只动 adapter 一个文件，不动 HYLE 地基**。
- iron 那份只当"现在能立刻跑通验证"的参照物，非长期依赖。

教训：任何 vendored 快照都会变老（这份已老），所以降低耦合、把 ggml 当可换后端，而非把 HYLE 焊在它身上。

### 2. 但 ggml-backend 是"张量算子图"专用，HYLE 要更底层
ggml 的调度单元是 `ggml_tensor` + 固定算子集，不是通用 GPU 计算。HYLE 要
**device/stream/event/kernel + provider-neutral Mem storage**，能跑任意 kernel。所以边界分两层：

```
compute/
├─ 上层（HYLE 自有，平台无关）
│   device   — 枚举/选择 GPU/CPU 设备
│   stream   — 异步执行队列（= 可调度资源）
│   memory   — 向 Mem 提供 device/pinned/managed resource；所有权/view 仍是 Mem 类型
│   kernel   — 任意 kernel 的提交/启动抽象
└─ 下层后端（复用 ggml 实现作起点 + 可直调原生 API）
    cuda / hip / metal / vulkan(compute) / sycl / cpu(simd fallback)
```
上层之上再挂一个 `ggml-bridge`：把 ggml-backend 当成"算子库"接进来，需要现成 AI 算子时走 ggml，需要自定义 kernel 时走 HYLE 原生 stream/kernel。

> **校正（对照真实 ggml-backend.h）**：ggml-backend **已经有** device/stream/buffer/buffer_type/event/sched 全套抽象。所以 HYLE 真正"净新"的只有两处：**①任意 kernel 启动**（ggml 是固定算子集 + `graph_compute(cgraph)`，无 raw launch）；**②host 侧 completion→fiber 唤醒**（ggml 的 `synchronize` 是阻塞式，event 仅 GPU-GPU 跨后端同步，无"算完回调唤醒 host"）。device/buffer 枚举应**薄包 ggml-backend**，避免双层抽象税；只在 kernel-launch + fiber-completion 处真正自造。可直接照镜 ggml 的 `dev_caps{async,host_buffer,buffer_from_host_ptr,events}` 能力查询结构。
>
> **Mem v2 校正：**ggml 的 `buffer_type_i{get_alignment/get_max_size/get_alloc_size/is_host}`
> 证明了“每个 storage provider 自报属性/上限”比闭合 kind enum 更自然。HYLE 把这条洞见落在
> Mem `MemoryResource` 开放 provider 上：Compute 实现 backend provider，Mem `Buffer/View` 保留
> ownership、domain 与 opaque handle；Stream/Event 仍由 Compute 负责 ordering。ggml 非标能力
> 经 bridge 探测，不能把 ggml 类型泄漏到 Mem 或 Compute 公开头。

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
- **HYLE 策略**：上层抽象屏蔽差异，两条后端都建。选择走**编译期优先、运行期一次冷跳兜底**的统一 dispatch 模型（详见 [`coding-conventions.md` → 能力 dispatch](../developer/standards/coding-conventions.md)）——构建期可只编目标后端（llama.cpp 式 `-DGGML_CUDA=ON` 按机器定死，零 dispatch）；发布二进制则启动探测一次、冷跳进可用后端（CPU(simd) → Vulkan compute → CUDA/HIP/Metal 原生）。**后端选择是冷路径，kernel 内部满速无间接。**

## 借 vs 造

| 层 | 借/造 | 说明 |
|---|---|---|
| AI 算子集（matmul/attention/conv…）| ✅ 借 ggml（上游版）| 各后端算子，不重写；**对接上游 pin 版，非 iron 旧快照** |
| 跨后端"算子图"调度 | ✅ 借 ggml-backend | 现成跑通，封在 `ggml_bridge` adapter |
| **CPU 算子执行所有权** | 🔧 adapter 分流 | `ggml_backend_cpu_set_threadpool` 只接受 ggml 自建 pool，不能注入 HYLE scheduler。不可合批小任务用 `n_threads=1` + HYLE 外层并发；合批大 graph 才用 ggml 自有并行，并从 HYLE blocking pool 隔离提交 |
| **ggml 版本/接口隔离** | 🔧 自造 adapter | `ggml_bridge` 是唯一耦合点；ggml 升级/接口变只动这一文件 |
| 通用 device/stream/event/kernel + Mem storage seam | 🔧 自造 | ggml 没有的通用层，绑 HYLE sched；storage 统一用 Mem resource/Buffer，**不暴露 ggml 类型** |
| GPU completion → fiber 唤醒 | 🔧 自造 | event/fence 接 sched，io/ 同款机制 |
| Vulkan compute 自定义 kernel | 🔧 自造（参考 kompute）| 一等公民路径，slate vulkan 打底 |
| 设备内存分配 | 🔧 Compute provider + Mem owner | device/pinned/managed 各由 Compute 实现一个开放 `MemoryResource`；Mem `Buffer/View` 统一所有权与 range，Compute Stream/Event 保留 async ordering |

**原则**：算子和跨后端调度借 ggml（别重造）、但**只通过 `ggml_bridge` adapter 对接上游 pin 版、不焊死 vendored 快照**；通用抽象层 + 与 sched 的 completion 统一自造（ggml 给不了），且 HYLE 抽象不暴露 ggml 类型。

## 模块形状

```
compute/
├─ device.h        设备枚举/能力查询/选择
├─ stream.h        异步执行队列（可调度资源，产生 completion）
├─ memory.h        为 Device 暴露 device/host-pinned/managed Mem resource provider
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
- **手动虚表分派 + 组合优于包含**（conventions §4/§10）：`ComputeBackend` 是纯 fn-ptr 结构，每后端一个独立 TU 填一张 vtable（`cuda_vtable()`/`host_vtable()`），**非**上帝类 `switch(backend)`。加一个后端（HIP/Metal/Vulkan）= 加一个 TU + 一张表，0 调用点改动。device 全局表在 `compute.cc` 冷路径构建（GPU 优先、host 恒在末位保证非空）；公开 `ordinal()` 是该表的全局不透明句柄，分派时再映射成 backend-local ordinal。默认构造的 `Device` 仅为无效输出占位（`valid()==false`），避免多后端时把字段中的 CPU 与全局 0 号 GPU 混成矛盾对象。
- **无 pimpl / 无堆分配**（性能）：`Stream`/`Event` 内联 `{const ComputeBackend* vt, void* native, int ordinal}` 三字；公开头靠前向声明 `struct ComputeBackend` 不泄漏 `cudaStream_t`。`query()` 热路径（compute-03 worker 每轮 poll）= 一次 vtable 间接（后端多态的不可约成本），无 impl_ 解引用。
- **CUDA 可选依赖 + C++ 方言现实**：`HYLE_COMPUTE_CUDA=AUTO|ON|OFF` 控制 Toolkit 探测；ON 是硬能力契约，OFF 不探测且只编 host，AUTO 才允许按机器发现。最终启用结果门控 CUDA TU 与公开 feature manifest。**nvcc 13.3 转发 `-std=c++23` 给 host 编译器时忽略之**（实测 warning "flag not supported with the configured host compiler"；与 CMake 版本无关——CMake 4.1 已支持 CUDA23 方言，是 nvcc 前端硬限制）。故拆出轻量 `types.h`（`DeviceType`/`DeviceCaps`/`EventStatus`，不含 `error/result.h`），让 `cuda_backend.cu` 编译时不碰 `std::expected`（C++23 库类型），CUDA TU 用 C++20 方言、host C++ TU 仍 C++23。
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
3. ~~**buffer 与 mem/ 的边界**~~ **✅ Mem v2 已统一**——storage ownership、resource identity、
   `Buffer/View`、host mapping/coherence 与同步 range operation 归 Mem；device/pinned/managed allocation
   artifact 由 Compute provider 实现，Stream/Event 保留 async ordering。非 ReBAR Vulkan 的同步
   provider 路线可复用内部 staging，但不能把它冒充 async ring；未来真实异步上传仍需由 Compute
   用 pinned Mem Buffer + Device identity + Event/backpressure 组成 `StagingRing`。所有 in-flight
   kernel/copy 完成前 owner/view 不得失效。
4. **Vulkan compute 自定义 kernel 的工程链**——GLSL/`.comp` → SPIR-V 编译、运行时加载、与 CUDA kernel 的统一启动 API(compute-06)。**参考库 = kompute**(`C:\public\project-hyle\kompute`,见下"compute-06 kompute 映射"节已定该抄什么)。

## compute-05 借 ggml 算力：调研发现汇总（2026-07-06 深调研 ggml/llama.cpp/iron/CCCL 定案）

> compute-05 = 借 ggml 算子。动手前把所有关键发现集中在此（文档线性，不散）。分两块：**内存零拷贝（好消息，能拿最高性能）** + **completion 阻塞（ggml 结构性限制，接受它，非缺陷）**。

### 发现 1：iron 是"整库拼装"的反面教材，不是"借算子"的范本
iron（clay 宿主）**不直接用 ggml**，而是 `add_subdirectory(stable-diffusion.cpp)` 把整个 sd.cpp 应用 + 它冻结的 ggml 快照吞进来（`iron/thirdparty/CMakeLists.txt:1`），业务层 `flux_api.cc` 调 sd.cpp 高层门面 `new_sd_ctx`/`txt2img`（20 参数），几乎不碰 ggml。这是 **L2 层"借整个推理应用" + clay/oatpp/sd.cpp 三套 runtime 各自为政**——正是 HYLE 动机（motivation §24）要消除的"东拼西凑"。**HYLE compute-05 要借的是"算子 kernel"粒度（借 ggml 不借 sd.cpp）、pin 上游版（不绑应用冻结快照）、接进统一 completion 脊（iron 里 clay 和 ggml 完全不通）。** iron 反证：不 fork、不借整库、不绑冻结快照。

### 发现 2：显存归属边界——host 归 HYLE / device 归 ggml（外部显存注入在 CUDA 上结构性死路，非零拷贝）
> **⚠️ 2026-07-07 两阶段真跑 + gdb 彻底修正**：本发现初版愿景是"HYLE 自己 cudaMalloc 的显存零拷贝喂给 ggml 算子算"。**被 RTX 4070 真跑 + WSL gdb 逐行证伪**——先是 `->buffer=NULL` 单算子段错误，再是 dummy-buffer 合批 varied 形状崩（`ggml-cuda.cu:1924`，共享 size-0 dummy 的 context 被 gallocr 弄丢）。根因不是"没找对 API"：**CUDA 后端 `buffer_from_host_ptr=false`（`ggml-cuda.cu:5032`），结构性不接受外部显存指针**。全过程 gdb 坐实见 [`../debug/ggml-zerocopy-null-buffer-hunt.md`](../developer/debugging/ggml-zerocopy-null-buffer-hunt.md)，正式边界见 [`borrowing-ggml-memory-model.md`](borrowing-ggml-memory-model.md)。

**正确边界：借 ggml 算子 = host 内存归 HYLE（mem 喂入/取出），device 显存归 ggml（gallocr 自管），就像 llama.cpp。**
- **device 侧归 ggml（CUDA 上必须）**：ggml 算子计算用的 tensor buffer（输入/中间/结果）由 `ggml_gallocr` 分配管理。CUDA 无公开 API 注册外部显存（`buffer_from_host_ptr=false`）；且 CUDA matmul 内核依赖 buffer 的 `context->device`/`buft` 元数据、图内复用、padding 一整套，是算子正确性的一部分。**不注入外部 `->data`、不造假 dummy buffer**（那在多算子 varied 形状必崩）。
- **host 侧全程归 HYLE**：数据进出 ggml 显存的入口 `ggml_backend_tensor_set/get` 对 host 源指针**零约束**——当前可使用普通 HYLE `mem::AllocatorRef` 数据，或由 Compute host/pinned provider 创建并 map 的 Mem Buffer。Mem 已分离 native handle 与 host mapping 并显式提供 flush/invalidate，但 ggml bridge 没有 async slot 生命周期，故不宣称 pinned overlap；StagingRing 待真实 async overlap 消费者。
- **落地形态**：`ggml_init({.no_alloc=true})` 建 tensor → `ggml_set_input`(输入)/`ggml_set_output`(结果) → `ggml_mul_mat` → `ggml_gallocr_alloc_graph`（device buffer 全归 ggml）→ `ggml_backend_tensor_set`(HYLE host 输入拷入) → `graph_compute` → `ggml_backend_tensor_get`(结果拷回 HYLE host)。合批同理（一图多算子、gallocr 一次分配）。
- **权重常驻**：反复用同一批权重（推理）时，权重拷进 ggml 显存**一次**、常驻复用（llama.cpp 模式），非每次推理都拷。初始化成本，非热路径。
- **HYLE 显存自管没名存实亡——只是划清了范围**：它在 **compute-04/06 自造 kernel**（真零拷贝跑在 HYLE Buffer 上，已三平台验证——服务一切非 AI 的通用 GPU 计算：普通 CUDA 数值 kernel + ② CAD/CAE 专用核）和其 **数据**（普通 compute buffer + KV-cache/网格/BVH/刚度矩阵）上完整成立；只在"借 ggml 算 AI"这条路上，device 侧让位给 ggml。这是**借 vs 造**的本质分工（呼应发现 5/7/9）。

### 发现 3：ggml stream 私有，completion 只能阻塞 synchronize（接受它）
- **ggml CUDA stream 是 backend 私有的**（`common.cuh:1391` `streams[][]`，`stream()` 懒创建），**没有任何公开 API 让外部注入 HYLE 的 stream**。想让 ggml 跑在 HYLE stream 上不可行（除非改 ggml 源码）。
- ggml 提交异步（`graph_compute` 只 launch 不同步，`ggml-cuda.cu:4581`），但**对外完成语义只有阻塞 `cudaStreamSynchronize`**（`ggml_backend_cuda_synchronize`，`:3276`）。ggml 的 `event_record`/`event_wait` **仅用于多 backend/多 stream 内部依赖排序，不用于唤醒 host 异步循环**（`:4640`）。
- 结论：**HYLE 的 GpuPoller（compute-03，cudaEventQuery 轮询）poll 不到 ggml 私有 stream 上的活**——借 ggml 算子这条路，completion **接不进 GpuPoller**。

### 发现 4：llama.cpp 实证的"一般做法"= 单线程阻塞 synchronize + 连续批处理（合批）
- llama.cpp server 是**单线程主循环**（`server-queue.cpp:125` while(true) update_slots），并发靠**连续批处理**：把所有活跃 slot 的 token 合并进一个 `llama_batch` 一次 decode（`server-context.cpp:2816`）。
- decode 内最终 `synchronize()` **阻塞等这一个 batch 算完**（`llama-context.cpp:684`）。**GPU 一次只算一个 batch，host 主线程就阻塞在 cudaStreamSynchronize**——不是"去处理别的请求"，请求早已合批进当前 decode。**真并发 = 合批，不是多 stream。**
- ggml 官方 examples 无一例外用阻塞 `ggml_backend_graph_compute`。业界"自管调度接 CUDA 异步"（CCCL STF 的 `cudaLaunchHostFunc` 回调，`host_launch_scope.cuh:339`）**需要控制 stream，ggml 不给，故对 ggml 只剩阻塞一条路**。

### 发现 5：三层 × 两条 kernel 来路矩阵（澄清"ggml 也有 vulkan 岂不重复"）
```
                AI 算子(matmul/attn)        任意 GPU 计算(普通 CUDA: vecadd/reduce/scan; ② 专用: 几何核/stencil/BVH/SpMV)
                ── 借 ggml ──                ── HYLE 自造 ──
CUDA(N卡)    ggml-cuda (compute-05 桥)     compute-04 NVRTC ✅
Vulkan(全厂商) ggml-vulkan (compute-05 桥)   compute-06 SPIR-V (待,消费者=CAD/CAE)
```
HYLE device/stream/event/buffer/kernel 抽象在最上层，下挂两类后端：**借 ggml 的（AI 算子，全平台，含 ggml-vulkan）** + **HYLE 自造的（任意 GPU 计算：CUDA=04 NVRTC / Vulkan=06 SPIR-V）**。**不重复**：ggml-vulkan 给"AI 算子的跨厂商实现"（借），compute-06 给"HYLE 自定义 SPIR-V kernel 的跨厂商实现"（造，ggml 没有的普通计算 + 几何/网格/求解 kernel）。同 CUDA 侧 05 借 ggml-cuda、04 自造 NVRTC。**右列不只是"② CAD/CAE 专用核"——它是"一切 ggml 那 ~90 个 AI 算子之外的 GPU 计算"，含最普通的 vecadd/reduce（ggml 也不给）。**

### 发现 6：三个"graph"同名不同物（澄清"graph 消费者是 vulkan 吗"）
- **HYLE graph-01** = 任务依赖 DAG 增量重算（clay taskgraph；改参数→重算下游节点，节点=任务、边=依赖）。消费者 = **北极星② CAD/CAE 参数化建模**，跑 sched 上，**与 GPU/Vulkan 无关**。
- **ggml cgraph** = 算子计算图（matmul→softmax…，节点=算子、边=张量数据流）。ggml 内部，compute-05 借。
- **ggml-vulkan** = ggml 的一个后端（用 Vulkan 实现算子）。
三者不同层，别混。"graph 的消费者"是 CAD/CAE，不是 vulkan。

### 发现 7：自造 kernel（04/06）的正当性来自"通用 GPU 计算"这个大类，不只② CAD/CAE
- **推理（北极星①）**：AI 算子几乎全在 ggml（含 ggml-vulkan）。HYLE 对推理的价值是**统一底座**（io/sched/mem/completion），AI 算子**借 ggml 就够**——compute-04 的自造 kernel 对纯推理用处不大（推理不写自定义 kernel）。
- **但"自造 kernel"服务的是"通用 GPU 计算"整个大类，不止② 的专用核**：ggml 只是"AI 推理算子多"，它**不是通用 GPU 计算框架**。**普通 CUDA 并行计算**——vecadd/reduce/scan/前缀和/逐元素变换/任意自定义数值 kernel——ggml 一概不提供（发现 9）。这些**通用计算 HYLE 必须自己做**（compute-04 已验证：端到端跑的正是 `vecadd` 这种最普通的 CUDA 计算，非几何核）。
- **② CAD/CAE 是通用计算里最深的消费者**：几何核/网格 stencil/BVH/**CAE 稀疏求解器 SpMV** ggml **完全没有**——这是自造 kernel（04 NVRTC / 06 SPIR-V）的**最重量级**真实消费者（motivation §26：CAE 把 compute/simd 压到"稀疏线代/迭代求解器"深度），但不是唯一：任何"不是 ggml 那 ~90 个 AI 算子"的 GPU 计算都归自造路径。
- 故：**compute-05（借 ggml）服务① 推理的 AI 算子 + ②CAE 能借的部分；compute-04/06（自造 kernel）服务一切非 AI 的通用 GPU 计算——普通 CUDA 数值 kernel + ② CAD/CAE 专用几何/数值核。** compute-06（Vulkan 自造）消费者（CAD/CAE 上层）尚未开工 → 按 §0.2 可等消费者驱动，不急。

### 发现 8：⚠️ 核心张力——§4"co_await gpu_kernel 同构"愿景 vs ggml 阻塞（诚实厘清）
§4 的"fiber `co_await gpu_kernel()` → 挂起 → event 唤醒、与 co_await async_read 同构"**成立于 HYLE 自造 kernel 路径**（compute-03/04：HYLE 自己的 stream + event + GpuPoller，已三平台真跑）。但**借 ggml 算子路径（compute-05）不适用**——ggml stream 私有（发现 3），completion 只能阻塞 synchronize（发现 4）。**两条路的 completion 模型不同**：
- **HYLE 自造 kernel（04/06）**：`co_await stream` 真挂起、GpuPoller 唤醒（§4 愿景，已兑现）。
- **借 ggml 算子（05）**：`ggml graph_compute` 阻塞 → **走 `spawn_blocking` 卸到线程，fiber 挂起等该线程**（llama.cpp 模式 A）。fiber 仍不阻塞 worker（spawn_blocking 卸载），但**不是 GpuPoller 那种"event 轮询唤醒"，而是"阻塞线程完成唤醒"**。
- **这不违背"fiber 统一"**：fiber 挂起等 completion 的**语义**一致（都经 sched Waker 唤醒）；只是 ggml 路径的 completion source 是"阻塞卸载线程"而非"GpuPoller 轮询的 event"。sched-11 的 spawn_blocking 本就是为"借入的阻塞第三方"设计的。**§4 需补注**：GPU completion 有两种 source——自造 kernel 走 GpuPoller（零阻塞轮询），借 ggml 走 spawn_blocking（阻塞卸载），都归一到 fiber Waker 唤醒。

### 发现 9：⚠️ ggml GPU 后端只有固定 AI 算子，连普通 CUDA 计算都不给（坐实 04/06 是"通用 GPU 计算"的独占刚需）
- **ggml 是"AI 张量计算库"，不是"通用 GPU 计算框架"**：它的算子集是**编译期固定的 ~90 个 AI 算子枚举**（`GGML_OP_MUL_MAT`/`SOFTMAX`/`ROPE`/`CONV`… 见 `ggml.h` 的 `ggml_op`），每个后端把这些枚举 switch-case 映射到自己的 kernel（`ggml-cuda.cu:2140` 的 `ggml_cuda_compute_forward` 巨型 switch）。**ggml 只是"AI 推理算子多"——不是"能跑任意 GPU 计算"。**
- **自定义 op 是 CPU-only**：ggml 有 `ggml_map_custom1/2/3`（用户塞函数指针），但**其 CUDA 后端零实现**——`GGML_OP_CUSTOM`/`MAP_CUSTOM` 在 `ggml-cuda.cu` 的 switch 里**零匹配**（grep 实证），落到 default 分支 return false = "不支持"，sched 会把该节点回退到 CPU 后端跑（`ith/nth` 线程分片，`ggml.c` custom op 执行体）。**即：想在 GPU 上跑一个 ggml 没有的算子，ggml 结构上做不到。**
- **推论（范围比"缺 CAD/CAE 核"更广）**：借 ggml（compute-05）只能拿到它**已有的那 ~90 个 AI 算子**在 GPU 上的实现。**连最普通的通用 CUDA 计算——vecadd/reduce/scan/前缀和/逐元素变换/任意自定义数值循环——ggml 都不提供**（它们不在那 ~90 个 AI 枚举里，且 GPU 无 custom-op 路径）。② 的几何核/网格 stencil/BVH/CAE SpMV 更是一个都没有。**ggml 给不了的不止"② 的专用核"，是"一切非 AI 的 GPU 计算"。**
- **故 compute-04（CUDA NVRTC）/ compute-06（Vulkan SPIR-V）不是可有可无的"造轮子"，而是 ggml 结构性提供不了的能力**：把**任意** kernel 源码跑上 GPU。这是 HYLE **通用 GPU 计算**（普通 CUDA 数值 kernel + ② CAD/CAE 专用核）的**唯一**路径（① 推理的 AI 算子 ggml 全有、不需自造）。发现 7 的"04/06 正当性"由此从"消费者视角"坐实到"ggml 能力边界"视角：**不是① 用不上，而是 ggml 根本不是通用 GPU 计算框架、非 AI 算子一概给不了**。compute-04 端到端验收跑的 `vecadd` 就是这条"普通 CUDA 计算 ggml 不给、HYLE 自造"的最小证明。

### 发现 10：⚠️ ggml CPU 并行是"线程池+算子"焊死的一体，借不了核内并行；线程调度权必须归 HYLE（引出执行模型专章）
> **历史研究基线**：下面的行号铁证来自 ggml `v0.15.3`，它推翻了 project 记忆里
> "`ggml_backend_cpu_set_threadpool` 喂 HYLE 调度器给 ggml"的假设。当前产品已锁到 `v0.21.0`；
> adapter/构建测试必须复证结论，旧行号不能冒充当前源码定位。结论不是"n_threads=1 是普世答案"（那是
> 一度的过度简化），真正答案仍是"线程数按合批性二分"，见下方执行模型专章。
- **铁证 A：`ggml_threadpool` 不可插拔**（`ggml-cpu.c:471`）：字段全是 ggml 私有的 `mutex/cond/n_barrier/workers[]/stop/pause`；`ggml_threadpool_params`（`ggml.h:2844`）**无任何 fn-ptr/callback**，只有 `cpumask/n_threads/prio/poll/strict_cpu/paused`。**没有让调用方注入"work 派发函数"的接缝** → `ggml_backend_cpu_set_threadpool` 只能喂一个 *ggml 自建的* threadpool（`ggml_threadpool_new` 造的、自起 pthread `:3286` 或 OpenMP `:3338`）。在 HYLE 底下 = **第二个调度器**。
- **铁证 B：连"借算子分块、换掉线程"这条中间路也走不通——barrier 是焊点**：算子函数只收 `{ith, nth}`（`ggml-cpu-impl.h:18` 的 `ggml_compute_params`）自分块，看似可借；**但算子内部有自旋 barrier**（`ggml_barrier`，`ggml-cpu.c:576` = `while(...) ggml_thread_cpu_relax()` 忙等），mul_mat 里 5 处（`:1355` 等）。**自旋 barrier 要求 nth 个执行流真并行推进**——若把 nth 个分片当 nth 个 fiber 铺到 worker，只要一个 fiber 没被调度到、卡在 barrier 自旋，其余 fiber 空转等它、而它在同 worker 队列里排队等 CPU → **协作式 fiber 与自旋 barrier 根本对立，死锁/活锁**。故"借 ggml 核内并行但用 HYLE 线程"**物理上不可能**——它的核内并行和线程池是一体，barrier 是焊点。要核内并行就得吞它的线程池（第二调度器），没有中间道路。
- **铁证 C：但算子分两类（决定后续能优化的边界）**：**elementwise 类**（mul/add/激活/norm…）`binary-ops.cpp`/`unary-ops.cpp` **零 barrier**——每输出元素独立，**HYLE 可自己按行切块、喂多 fiber、无需 barrier**；**mul_mat/attention 类**有 barrier（chunk 协调 + 归约）——核内并行**借不了**，只能整块单核或整块卸载。
- **结论（引出专章）**：ggml CPU 后端对 HYLE 只有两种正当用法——**(a) `n_threads=1` 当纯计算叶子**（`n_threads==1` 时 ggml 零 spawn、直接在调用线程 `ggml_graph_compute_thread(workers[0])`，`ggml-cpu.c:3356-3358` 铁证），并行由 HYLE 铺；**(b) 合批成单个大 graph 后放它全核**（此时无并发、不打架，见专章）。**选 a 还是 b 取决于"能不能合批"**——这不是 07 才评的细节，是 compute-05 的执行模型骨架，下方专章定死。

### compute-05 定案（基于上述）
- **显存归属边界（发现 2，2026-07-07 gdb 定论）**：**host 归 HYLE / device 归 ggml，就像 llama.cpp**。ggml 算子用 `gallocr` 分配它的 device buffer（CUDA 上 `buffer_from_host_ptr=false`，外部显存注入结构性死路）；HYLE 的 host 数据经 `tensor_set`(拷入,源可 pinned)/`tensor_get`(拷出) 进出（host 源零约束）。HYLE 显存自管归 compute-04/06 自造 kernel + ② CAD/CAE 数据。**不注入外部显存、不造假 dummy buffer。** 专文 [`borrowing-ggml-memory-model.md`](borrowing-ggml-memory-model.md)。
- **completion 走 spawn_blocking**（发现 3/4/8）：`ggml graph_compute` 卸到 sched-11 blocking pool，fiber 挂起、完成经 JoinHandle/Waker 唤醒。**不接 GpuPoller**（ggml stream 私有）。
- **CPU 线程模型 = 合批性二分，`n_threads` 可配（非硬编码 1）**（发现 10 + 执行模型专章，照 llama.cpp `:2425-2429` 铁证）：能合批的大 graph 放 ggml 全核（无并发不打架）；合不了批的高并发独立小推理走 `n_threads=1` + HYLE 铺 N 个。**详见下方「compute-05 CPU 执行模型专章」。**
- **性能杠杆是合批**（发现 4，llama.cpp 真并发手段）：HYLE 侧合并多算力请求成一个 ggml graph 一次提交 > 多 fiber 各自卸载。**且合批不只是性能杠杆——它还是决定线程数的开关**（见专章）。
- **adapter 单点隔离 + pin 上游版**：ggml 类型全封 `ggml_bridge.cc`，对外只 HYLE 抽象；ggml 升级只动一文件 + 跑对拍测试。
- **当前 ggml 产品合同**：由中央 lock 固定 `v0.21.0` commit
  `8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`，FetchContent 用官方 archive + SHA-256；不在
  `third_party/` vendored。HYLE 只构建私有 PIC 静态 CPU + 可选 CUDA 闭包，关闭 tests/examples、
  OpenMP、BLAS/Accelerate、Metal/Vulkan/RPC 和动态 backend；所有 ggml 类型继续封在
  `ggml_bridge.cc`，升级必须复跑 adapter、package/install 和执行模型测试。

## compute-05 CPU 执行模型专章：线程调度权归属（合批性二分，照 llama.cpp 铁证）

> **为什么单开一章**：发现 4（合批）与发现 10（n_threads=1）初看互斥、实则是同一条轴的两端。搞错这条轴 = compute-05 的执行模型骨架错，返工伤筋动骨。此章把它一次定死，不留到 07。**核心押注**：没人比 llama.cpp 更懂怎么调 ggml，直接照抄它的线程模型 + 补 HYLE 独有的第三格。

### llama.cpp 的铁证：它显式按 `batched` 二分线程模型，连线程池都备两个

```cpp
// llama-context.cpp:2425-2429
ggml_status llama_context::graph_compute(ggml_cgraph* gf, bool batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;
    ...
    set_threadpool_fn(backend_cpu, tp);   // :2435 注入它自建的 threadpool
```

- **`batched=true`（prompt 预填充 / 多序列合批）**：用 `n_threads_batch`，默认**全核**（`common.cpp:275` `common_cpu_get_num_math()` = 物理核数）。大 matmul 计算密集，吃满核。
- **`batched=false`（单 token 解码）**：用 `n_threads`，`common.cpp:133` 故意 `n<=4 ? n : n/2`——**不吃满核**。单 token decode 是**访存瓶颈**（逐 token 流式读权重），线程再多只是抢内存带宽 + 增加 barrier 同步开销，反而更慢。

即：**"用几个核"由算子的计算/访存特征决定，llama.cpp 把它固化成 batched 二分 + 双 threadpool。**

### 关键澄清：发现 10 与 llama.cpp 做法不矛盾——差在"谁当核仲裁者"

`:2431-2436` 里 llama.cpp **正是用发现 10 判定"HYLE 不能用"的那个 `ggml_backend_cpu_set_threadpool` 注入自建 threadpool**。这不矛盾，差在架构位置：
- **llama.cpp 是纯推理进程，它自己就是顶层、唯一的核仲裁者**——没有别的用户态调度器和 ggml 抢核，让 ggml threadpool 吃满核天经地义。
- **HYLE 底下有 fiber work-stealing 调度器**，它才是核仲裁者。再让 ggml 起一个 threadpool 才叫"第二调度器打架"（超订、cache 局部性被毁、两套 barrier/park 决策互相拖）。

**故同一个 API，llama.cpp 用得对、HYLE 用则错——不是 API 错，是"谁说了算用核"不同。** HYLE 让 fiber 调度器当唯一仲裁者，ggml 永远是被它调度的叶子。

### 执行模型决策轴（compute-05 骨架，现在定死）

| 场景 | 同时几个 graph | 打不打架 | 线程模型 | 出处 |
|---|---|---|---|---|
| **合批大 graph**（prefill/多序列）| 1 个大 graph | **不打架**（此时 HYLE worker 无活、本该 park，让出核）| **ggml 全核** | llama.cpp `:2428` batched 全核 |
| **单请求 decode** | 1 个 | 不打架 | 中等/半核（访存瓶颈，多核无益）| llama.cpp `:133` n/2 |
| **高并发独立小推理**（HYLE 独有：embedding 服务/多租户小模型/CAE 代理模型）| N 个小 graph（**合不了批**：权重不同/延迟敏感/散发）| **打架**（N×核 抢 M 核）| **`n_threads=1` + HYLE 调度 N 个**（每小推理独占一核，小 matmul 本不吃核内多线程）| 发现 10 (a) |
| **有 GPU（任意）** | CPU 侧 1 根 sync 线程 | 不打架 | n_threads 无所谓（CPU 只 spawn_blocking 等 `cudaStreamSynchronize`）| 发现 3/8 |

**合批性是那个开关**：能合批 → 单 graph → ggml 全核（照 llama.cpp）；合不了批的高并发独立请求 → `n_threads=1` + HYLE 铺并发（llama.cpp 没有这一格，因为它只服务一个模型，而 HYLE 的 embedding/多租户/代理模型场景才有）。

### 落到 `GgmlBackend` 的设计决定（焊进 compute-05）

1. **`n_threads` 可配，非硬编码 1**：`GgmlBackend` 暴露线程数（或"合批/独立"模式），由调用方按合批性选——照抄 llama.cpp 的 batched 二分，不写死。
2. **合批是一等能力**：bridge 提供"把多请求合进一个 cgraph"的入口（发现 4）。合批既降 launch 开销，又切到"全核不打架"格。
3. **HYLE 始终是核仲裁者**：无论哪格，ggml 都不自作主张吃核——全核那格是 HYLE"授权"它吃（因为此刻 worker 本该 park）。这与 GPU 路径（spawn_blocking 卸载）语义一致：**ggml 永远是被 HYLE 调度的叶子。**
4. **elementwise 自切块留作 07 优化**（发现 10 铁证 C）：零 barrier 的算子 HYLE 可按行切喂多 fiber——但这是边角优化，不影响 compute-05 骨架（骨架 = 上面的合批性二分）。
5. **诚实边界**：`n_threads=1` 那格放弃了 ggml 单 graph 内的跨核 barrier 并行（一个 graph 一个核）——但那一格本就是"高并发独立小推理"，小 matmul 不吃核内并行，是净赚不是净亏。真正的大 matmul 走合批格、吃全核，不经过这一格。

## compute-06 kompute 映射：Vulkan 后端该抄什么/丢什么（2026-07-06 深读 kompute 源码）

> compute-06 的 Vulkan compute 后端以 **kompute** 为骨架参考。kompute 是理想参考——**纯 Vulkan、零隐藏运行时**（全 `src/` grep 无 `std::thread`/事件循环/condition_variable，唯一同步是可选的 submit mutex + 每 Sequence 一个 `vk::Fence`），不违反 HYLE"不引入第二个调度器"铁律。整体照抄其 Vulkan 样板，**唯一实质改动是把 fence 的阻塞 await 换成非阻塞轮询接 fiber**。

**该抄的三块不可省样板**（映射到 HYLE compute 抽象）：
1. **Vulkan 初始化链 → HYLE `Device`（Vulkan 后端的 vtable 创建路径）**：instance → 物理设备枚举 → 找 compute queue family（遍历 `eCompute`）→ logical device + queue → 能力/扩展查询。`Manager.cpp:174-496`。**保留 Apple/MoltenVK portability 分支**（`Manager.cpp:197-204`——跨厂商一等公民的前提，一份 SPIR-V 全平台跑）。
2. **SPIR-V → pipeline 全套 → HYLE `Kernel`（compute-04 的 Vulkan 实现）**：shader module 加载 SPIR-V → descriptor pool/layout/set 三件套 + `updateDescriptorSets` → pipeline layout（含 push constant range）+ compute pipeline + pipeline cache → 录制期 `bindPipeline`/`bindDescriptorSets`/`pushConstants`/`dispatch(x,y,z)`。`Algorithm.cpp:128-380`。
3. **device buffer + memory-type 选择 + barrier 编排 → HYLE `Buffer`（compute-02 的 Vulkan 实现）**：device-local primary buffer；ReBAR 直接持久映射，非 ReBAR 的同步 `set/get` 懒建并复用 transfer-sized host-visible staging slice；手动扫 `memoryTypeBits` 找 memory type（`Tensor.cpp:396-438`）；copy staging↔device 前后夹 `pipelineBarrier`，host coherent 内存免 flush。异步 overlap 不复用这条同步 convenience，必须另建 Event-owned backpressure ring。

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

## compute-07 CPU 算子后端：接 hyle simd，不是借 ggml（2026-07-09 定案 + 纠错）

> **⚠️ 一次方向纠错**：设计过程中一度误判"compute-07 也借 ggml CPU 后端"。**错**——那和 compute-05 重复（05 已实现 `GgmlBackend::create(CPU, n_threads)` 借 ggml CPU 算子）。翻 README:175 + backlog + scope 后端矩阵定论：**compute-07 是另一条正交线——compute-04 `Kernel`/`Stream` 自造抽象的 CPU 后端,kernel 走 hyle `simd/` 数值核,completion 恒 ready。** 两条 CPU 路并存不重复。

**两条 CPU 计算路的分工（正交,不是二选一）**：

| | compute-05（借 ggml） | compute-07（接 hyle simd） |
|---|---|---|
| 借谁 | **ggml** AI 算子（matmul/softmax/rope…）| **hyle simd** 数值核（dot/saxpy/exp_batch…）|
| 属哪套抽象 | `ggml_bridge`（算子库供应商,独立入口）| **compute-04 `Kernel::launch` 抽象的 CPU 后端**（CUDA NVRTC / Vulkan SPIR-V 的 CPU 孪生）|
| completion | spawn_blocking 卸载（ggml stream 私有,阻塞 synchronize）| **恒 ready**（CPU 同步内联,`host_event_query` 已恒 kReady）|
| 定位 | AI 推理算子（北极星①）| ①无 GPU 功能基线 ②**GPU 后端对拍 oracle**（CUDA/Vulkan 结果 vs CPU 逐元素比对）|
| 消费者 | 推理引擎 | compute-04 kernel 抽象本身 + 通用/CAE 数值计算 |

**为什么 CPU-AI 借 ggml、而 compute-07 接 simd 不矛盾**：
- **AI 推理的 CPU 路（matmul 主导,占算力 >90%）借 ggml**——ggml CPU 后端自带各架构 SIMD + 量化 GEMM,hyle simd **没有 matmul**(只有 BLAS-1 dot/saxpy + 逐元素超越),拿 dot 拼 GEMM 会慢数十倍。这条路是 compute-05。
- **compute-07 不是 AI-GEMM 路**——它是 compute-04 那套"提交任意 kernel"抽象的 CPU 落地:一个 host kernel(saxpy/vecadd/逐元素 map)用 hyle simd 实现,主要价值是**无 GPU 时抽象仍可跑 + GPU 结果的 CPU 对拍 oracle**,不是跟 ggml 抢 AI 推理。
- **hyle simd 的消费入口就是 07**：simd 支柱(含刚铺全的 f64)一直缺 compute 侧消费者,07 正是它——host kernel 包 simd 核。CAE/科学计算的 CPU 数值路也从这里长出(simd 有 f64 全套超越,ggml 没有)。

**实现形态（compute-04 hook 已就位,07 是"用起来"非"重建"）**：
- compute-04 已建好 host 路径:`Kernel::HostFn = void(*)(const LaunchConfig&, const KernelArgs&)`、`load_host(device, fn)`、`host_kernel_launch` 同步调用、`host_event_query` 恒 kReady。**07 = 把 simd 核包成 HostFn 注册进这套已有抽象,零新运行时、零 sched 改动。**
- host kernel 从 `KernelArgs` 解包 buffers[]/scalar blob（照 `host_vecadd` 模式,kernel_test.cc:44）→ 转调 `simd::saxpy_f32`/`dot_f32`/`exp_f32_batch`。
- **数据并行铺多核不在后端自起线程**（scope 铁律"不引第二运行时"）——靠 HYLE sched 的 fork-join（graph-02 parallel_for）,host_kernel_launch 保持单线程内联。
- **对拍 oracle**：同输入喂 CPU 后端(simd)与 GPU 后端(CUDA vecadd / Vulkan saxpy),逐元素比对——这是 07 双用途②,也是 compute 的正确性回归护栏。

**明确不做（防蔓延）**：不重造 GEMM(那借 ggml,compute-05);不给 07 堆 90 个 host 算子门面(消费者驱动);不让 CPU 后端自起线程池(数据并行归 graph-02)。

## 参考

- CUDA vs ROCm vs Vulkan vs Metal: GPU Compute in 2026 — orchestrator.dev
- AMD GPU Acceleration: ROCm, HIP, Vulkan, OpenCL Guide (2025)
- IWOCL 2026: chipStar — CUDA/HIP on OpenCL/SPIR-V
- FOSDEM 2026: Vulkan API for Machine Learning? Competing with CUDA and ROCm in llama.cpp
- AdaptiveCpp (hipSYCL) single-pass SYCL
