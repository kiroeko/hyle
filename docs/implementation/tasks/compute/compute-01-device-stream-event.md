# compute-01：device / stream / event —— 异构计算的 completion 原语

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | sched-04（Waker completion 脊，已✅）|
| **状态** | ✅ 完成（2026-07-06，compute 支柱破土第一步；双平台 RTX 4070 真机 10/10 + WSL TSan）|

> **范围**：compute-design「模块形状」里 `device.h` / `stream.h` 的那一层 + event completion 原语——**异步执行队列 + 完成信号的最薄抽象**，双后端（CUDA sm_89 真机 + host fallback）。
>
> **这是 compute 版的 io-01**：io-01 立的是「proactor 收割接缝 + 双后端 + 心跳验证」，**不含**接 sched worker loop（那是 io-04）。compute-01 对称地立「device/stream/event 抽象 + 双后端 + event 轮询完成验证」，**不含** fiber `co_await`——GPU event 接进 worker poll 边界是 **compute-03** 的核心难题（见下「明确不做」）。
>
> **借 vs 造**：device/能力查询**薄包**（照镜 ggml `dev_caps`）；stream/event 生命周期 + **completion 轮询原语自造**（ggml 结构性不提供 host completion，见 compute-design §4 铁证）。CUDA 后端直调 runtime API（不经 ggml——ggml 的 `synchronize` 是阻塞式，给不了 poll）。

## 为什么是它（compute 支柱的地基）

compute-design §4 的论点：**GPU 完成 = 又一种 completion source，与 io/ 同构**。要兑现这句话，先得有一个「提交异步 GPU 工作 → 拿到一个可**轮询**的完成信号」的最小对象。这就是 stream（异步队列）+ event（完成信号）。

completion 机制在写接口前已在真机去险（reference_cuda_env / compute-design §1）：`nvcc -arch=sm_89` 编 saxpy + `cudaEventRecord`/`cudaEventQuery` 轮询在 RTX 4070 跑通。**接口要承载的物理机制先验证过，再定形**——同 io-01 的纪律。

## 核心决策

### 1. completion = event 轮询（reap 模型），不走 callback
compute-design §2 已定案，此处落地。`stream.record(event)` 后，完成信号靠 `event.query()` 轮询：

- `query()` → `Ready` / `NotReady`（照镜 CCCL `event_ref::is_done()`：`cudaEventQuery` 返 `cudaSuccess`/`cudaErrorNotReady`）。
- **为什么不用 `cudaLaunchHostFunc`/`cudaStreamAddCallback`**：两种 callback 都禁止在回调里调任何 CUDA API（死锁风险），且回调跑在 CUDA 内部线程上、wake 时机不可控。ggml-cuda 的铁证——唯一的 `cudaLaunchHostFunc` 被 `#if 0` 注释标 "untested"，活路径是 `GGML_ABORT`。
- **轮询 = io_uring reap 模型的同构**：worker 在**已有的 poll 边界**（scheduler.cc 的 `set_io_poll` seam）里 `query()`，Ready 即 `waker.wake()`。wake 跑在自己的 worker 上、不在 CUDA 线程。**——但这一步是 compute-03，见下。** compute-01 只把 `query()` 原语做出来并验证它真能观察到 GPU 完成。

### 2. 双后端：CUDA（真机）+ host fallback（无 GPU 可编可测）
- **CUDA 后端**：直调 `cudaStreamCreate`/`cudaEventCreate`/`cudaEventRecord`/`cudaEventQuery`/`cudaStreamSynchronize`。`sm_89`（RTX 4070，Ada）。WSL nvcc 13.3 + Windows MSVC 双通道。
- **host fallback 后端**：无 GPU（或 CUDA 未编入，如 mac/CI）时，stream = 立即执行的顺序队列，event = 一个已完成标志。**目的不是性能**，是让 completion 脊的抽象在**任何平台可编可测**——同 io 的 epoll fallback「保住无 io_uring 也能跑」。它也是 compute-03 接 worker loop 时的确定性测试后端（无需真 GPU 就能验证 fiber 唤醒逻辑）。
- **后端选择是冷路径**（compute-design §CUDA vs shader）：启动探测一次，选可用后端。compute-01 只需 CUDA + host 两个；Vulkan/HIP/Metal 是后续里程碑。

### 3. 不暴露后端类型（compute-design §2 张力收口）
HYLE `Device`/`Stream`/`Event` 抽象**不暴露** `cudaStream_t`/`ggml_backend_t`。pimpl + 后端在 .cc 内分流（同 proactor.cc 的 `#if HYLE_OS_*` 结构，这里是 `#if HYLE_COMPUTE_CUDA`）。device 枚举/能力查询照镜 ggml `dev_caps{async,host_buffer,buffer_from_host_ptr,events}` + `dev_type{CPU,GPU,IGPU,ACCEL,META}`，但用 HYLE 自有 struct，不引 ggml 头。

### 4. 失败模型（契约优先 vs 外部边界）
- **外部边界**（`cudaMalloc`/`cudaStreamCreate` 失败、无 GPU）→ 返 `Status`/`expected`（运营失败，可恢复：回退 host 后端）。
- **调用方契约违反**（对已销毁 stream record、query 一个未 record 的 event）→ `HYLE_FATAL_ASSERT` fail-fast，不静默兜底（同 mem `is_pow2`、conventions 契约优先纪律）。
- CUDA runtime 错误码翻译成 HYLE `Status`，**不外泄 `cudaError_t`**。

## 明确不做（留后续里程碑，防范围蔓延）

| 不做的事 | 去向 | 为什么现在不做 |
|---|---|---|
| GPU event 接进 sched worker poll loop（`co_await stream`）| **compute-03** | `set_io_poll` 是单槽 fn-ptr、已被 io 独占；GPU 要接同一 worker 收割边界需扩展 seam（多 completion source 复用），是独立的架构难题。io 支柱也是 io-01 立 seam、io-04 才接 worker loop |
| device buffer（显存分配 + host↔device copy）| **compute-02** | ggml buffer_type 式实例，所有权 + stream 绑定 |
| kernel 启动抽象（`cuLaunchKernel` / SPIR-V）| **compute-04** | compute-01 只做 stream/event 脊，不碰 launch。测试用现成 kernel（saxpy）或纯 event record 验证完成信号 |
| ggml_bridge（借算子）| **compute-05** | — |
| Vulkan compute 后端 | **compute-06** | — |

## 交付物（实际落地）
- `include/hyle/compute/types.h`（`DeviceType`/`DeviceCaps`/`EventStatus` POD，**不含 result.h**——让 CUDA TU 不碰 std::expected，见下 CUDA 方言纪要）
- `include/hyle/compute/device.h`（`Device` 值类型 + `enumerate`/`default_device` + `device_type_name`）
- `include/hyle/compute/stream.h`（`Stream` 异步队列 + `Event` 完成信号；内联 `{vt,native,ordinal}` 无 pimpl）
- `src/compute/backend.h`（内部：`ComputeBackend` 手动虚表 + `host_vtable()`/`cuda_vtable()`）
- `src/compute/compute.cc`（分派层：device 全局表 + Stream/Event 转发；纯 C++ 无 CUDA）
- `src/compute/host_backend.cc`（host fallback：inline 执行、event 恒 Ready、sentinel handle 无分配）
- `src/compute/cuda_backend.cu`（nvcc：cudaStream/Event/Malloc/MemcpyAsync + cudaEventQuery poll）
- `tests/compute/device_test.cc`（10 测试，GPU 测试运行时探测、无 GPU 自 skip）
- `src/compute/CMakeLists.txt` + 根 CMake（`find_package(CUDAToolkit)`→`HYLE_COMPUTE_CUDA` 门控 + `CUDA_STANDARD 20` + `CUDA_ARCHITECTURES native`）

**cloc 实测**：生产 **538**（头 141 + C++ 240 + CUDA 143）+ 测试 139。**538 « 2-3k 预测**=脊精瘦（device/stream/event + poll 原语，重活 buffer/kernel/ggml/vulkan 在 compute-02~06），**非 stub**——双平台 CUDA 真机 10/10 + 显存往返 + TSan 为证。

## 实测结果（2026-07-06，三平台真机去险）
- **WSL（GCC14 + nvcc 13.3）**：`ninja` 编过（CUDA TU + host TU），`hyle_compute_test` **10/10 PASS**；`GpuEventPollsToReady` 在 RTX 4070（sm_89）**真跑 event NotReady→Ready 轮询**（非 skip，实测 26~217ms 随 GPU 忙闲）+ `AsyncCopyRoundTrip` 显存 H2D/D2H 往返数据逐元素校验。
- **WSL TSan（host-only，`-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON`）**：`setarch -R` **9/10 + 1 skip**（GPU 测试无 CUDA 自跳过）**零 data race**——host 侧脊（device 全局表 Meyers singleton + vtable 分派）并发安全。
- **Windows（MSVC 19.50 + nvcc 13.3）**：vcvars64 + Ninja 编过（`cuda_backend.cu.obj` nvcc + MSVC host），`hyle_compute_test.exe` **10/10 PASS**，GPU 测试同在 RTX 4070 真跑（25ms，非 skip）。
- **macOS（Apple clang，arm64/Apple Silicon，无 CUDA）**：`cmake -G "Unix Makefiles"` 配置时 `find_package(CUDAToolkit)` **找不到→compute 只编 host fallback**（真机验证无 CUDA 时的可移植性，非 `-DCMAKE_DISABLE_FIND_PACKAGE` 模拟）；`hyle_compute_test` **9/10 + 1 skip**（GPU 自跳过）。验证 host 后端的 device_alloc/memcpy 完成契约在第三个平台/编译器/架构（arm64 Apple clang）真跑通。

## 实现纪要 / 实测纠错（文档=当前真相）
- **手动虚表非虚继承**（用户纠偏 + conventions §4/§10）：`ComputeBackend` 纯 fn-ptr 结构；每后端独立 TU 一张表（组合优于包含），非上帝类 switch。早期 `error_handler_t` 曾采用同类机制，但该无人消费的 error handler 已在 error-03 删除。
- **去 pimpl 内联 3 字**（用户"性能优先、时间复杂度低"纠偏）：初版 Stream/Event 用 pimpl `Impl*`（每对象一次堆分配 + query() 多一次间接）；改为内联 `{vt,native void*,ordinal}`，公开头前向声明 `struct ComputeBackend` 仍不泄漏后端类型，query() 热路径省一次解引用。
- **CUDA C++ 方言纠错**：初判可给 CUDA TU 传 C++23，实测 **nvcc 13.3 转发 `-std=c++23` 给 host 编译器时忽略之**（warning "flag not supported"），`std::expected` 不可用；**与 CMake 版本无关**（CMake 4.1 已支持 CUDA23 方言，是 nvcc 前端硬限制）。解法=拆 `types.h`（POD 无 result.h）让 `.cu` 编译时不碰 `std::expected`，CUDA TU 用 C++20、host TU 仍 C++23。
- **显存单独、mem 只管 host RAM**（用户明确边界）：`cuda_backend` 的 `device_alloc`=`cudaMalloc`（真显存,**不**包 `mem::Allocator`——host 不能解引用显存）；host fallback 的 `device_alloc`=`std::malloc`（host "device"=host RAM，脊验证最小辅助，compute-02 buffer 抽象取代，届时 pinned host staging 才走 mem）。
- **error 是 compute 的 PUBLIC_DEPS**（同 platform 模式）：公开头 `Stream::create` 等返回 `Result<T>` 暴露 Status，消费者/测试链 hyle_compute 需 error DSO 符号。
- **`cudaSetDevice` 语义想透并修正（2026-07-06 深调研，越早越好）**：自审曾担心"fiber 跨 worker 迁移 vs current-device per-thread"是待调研隐患——**深读 ggml/CCCL 源码后解决且推翻了顾虑**。CUDA stream/event 的**操作**（record/query/sync/free）**不依赖 current device**（handle 内在绑 context）；ggml event 操作前都不 set device（ggml-cuda.cu:4527/4533/5510），CCCL 用 driver API 绕开 runtime "栈空 push dev0"（stream_ref.h:189）。**故 fiber 迁移安全、不需绑核。** 后端 `set_device()` helper 照 ggml-cuda.cu:109 的 get-if-differ 规避冗余 set，**只在创建站点调用**（stream/event/malloc），操作路径 set-free。双平台 GPU 重验 10/10。
- **compute-03 completion 接 sched 方案已定（不滞后到 compute-03 才想）**：深调研 ggml/CCCL/seastar 源码定案（见 compute-design "compute-03 GPU completion 接 sched"节）——callback 路线否决（ggml `#if 0`+CCCL 标高危）；**方案 A（首版零改 sched）**=独立 poller 线程 `cudaEventSynchronize` 完成→`Waker::wake()`+`set_io_wake` 打断 park，复用现有 Waker/`IoOperation`/`net_timeout` refcount slot/JoinHandle/spawn_blocking；**方案 B（后续）**=seastar `try_enter_interrupt_mode` veto 协议 + `set_io_poll` 泛化多 source。**compute-01 的 `query()`（轮询,方案 B）+ `sync()`（阻塞,方案 A）两原语已覆盖两条路,接口经此验证正确、无需回改。**

## clay 吸纳查漏
clay `compute/` = **0 个 .cu**，唯一碰 CUDA 的是 `host_device.h`（纯宏门面 `__host__ __device__`，无 kernel、无 runtime API 调用）。coverage-map：**compute 完全 net-new，无 clay 能力可吸纳**。故本里程碑无 clay diff 义务，纯自造。

## 验收门槛
- Win MSVC + WSL nvcc 双通道真编真跑；host 逻辑（记账/生命周期）过 WSL TSan。
- host fallback 后端：无 GPU 也能编能测，event 完成语义确定性可断言。
- CUDA 后端：RTX 4070 真机 `record → query 轮询 → Ready` 观察到真实 GPU 完成（非 stub）。
- cloc 验真回填；对抗式自审（正确性 / 契约边界 / 对照 compute-design §2 completion 决策）。

## 参考
- compute-design §1（环境去险）、§2（event 轮询决策 + reap 模型）、§4（GPU completion 与 io 同构）、「模块形状」。
- reference_cuda_env（本机 CUDA 13.3 双通道 + sm_89 + 端到端已验证）。
- CCCL `event_ref::is_done()`（`cudaEventQuery` poll 原语印证）；ggml `ggml-backend.h`（`dev_caps`/`dev_type` 能力查询照镜，借设计不引头）。
- io-01（同支柱首个里程碑的「立 seam、不接 worker loop」范式先例）。
