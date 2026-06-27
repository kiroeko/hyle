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
> **深读补充（buffer_type 即 MemorySpace 的范本）**：ggml 的 `buffer_type_i{get_alignment/get_max_size/get_alloc_size/is_host}` 正是 HYLE MemorySpace 该照镜的 trait——且 ggml 里 **host/device/pinned/split 不是枚举开关，而是各立一个 buffer_type 实例**（`is_host()` 是唯一 host 布尔判据）。这印证方案 D：**一种内存模型 = 一个 space/buft 实例**，不是 `switch(MemKind)`。非标能力（`set_n_threads`、`split_buffer_type` 等）经 `get_proc_address` 探测取得（不破 vtable）——HYLE `ggml_bridge` 拿这些扩展走同一机制。
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
| **ggml 版本/接口隔离** | 🔧 自造 adapter | `ggml_bridge` 是唯一耦合点；ggml 升级/接口变只动这一文件 |
| 通用 device/stream/buffer/kernel 抽象 | 🔧 自造 | ggml 没有的通用层，绑 HYLE sched，**不暴露 ggml 类型** |
| GPU completion → fiber 唤醒 | 🔧 自造 | event/fence 接 sched，io/ 同款机制 |
| Vulkan compute 自定义 kernel | 🔧 自造（参考 kompute）| 一等公民路径，slate vulkan 打底 |
| 设备内存分配 | 🔧 改造 mem/ | 扩 host/device/pinned/unified（呼应 ggml 的 device-memory-aware API）|

**原则**：算子和跨后端调度借 ggml（别重造）、但**只通过 `ggml_bridge` adapter 对接上游 pin 版、不焊死 vendored 快照**；通用抽象层 + 与 sched 的 completion 统一自造（ggml 给不了），且 HYLE 抽象不暴露 ggml 类型。

## 模块形状

```
compute/
├─ device.h        设备枚举/能力查询/选择
├─ stream.h        异步执行队列（可调度资源，产生 completion）
├─ mem_space.cc    实现并注册 device/pinned/unified 的 mem/ MemorySpace（device 内存方案 D）+ copy(src,dst,stream)，同 coherent space 时塌缩为 no-op
├─ buffer.h        基于注册的 MemorySpace 的设备 buffer（分配走 mem/ 的 Arena<Space>，绑 stream）
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

## 待深挖（下一步）

1. **device/stream/kernel 接口的具体形状**——提交/同步/事件/错误模型，怎么同时容纳"原生 CUDA kernel"和"SPIR-V compute shader"两种 kernel 表示。
2. **GPU completion 接 sched 的细节**——CUDA stream callback / Vulkan fence 怎么映射到唤醒哪个 fiber，避免每次 poll。
3. **buffer 与 mem/ 的边界**——host/device 传输、pinned、unified memory 怎么纳入 HYLE 分配器家族。
4. **Vulkan compute 自定义 kernel 的工程链**——GLSL/`.comp` → SPIR-V 编译、运行时加载、与 CUDA kernel 的统一启动 API。

## 参考

- CUDA vs ROCm vs Vulkan vs Metal: GPU Compute in 2026 — orchestrator.dev
- AMD GPU Acceleration: ROCm, HIP, Vulkan, OpenCL Guide (2025)
- IWOCL 2026: chipStar — CUDA/HIP on OpenCL/SPIR-V
- FOSDEM 2026: Vulkan API for Machine Learning? Competing with CUDA and ROCm in llama.cpp
- AdaptiveCpp (hipSYCL) single-pass SYCL
