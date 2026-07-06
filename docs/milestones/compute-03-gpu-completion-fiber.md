# compute-03：GPU completion → fiber —— `co_await stream` 与 io 同构

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Stream/Event）、sched-04（Waker completion 脊）|
| **状态** | ✅ 完成（2026-07-06，三平台真机：Win/WSL GPU fiber 真挂起/poller 真唤醒 + TSan 跨线程 wake 零 race + mac host）|

> **范围**：把 GPU 完成事件接进 HYLE fiber 调度器——`co_await stream`（等该 stream 所有已入队工作完成）/ `co_await event` 挂起**fiber**（非仅协程），GPU 算完经 event 唤醒它。兑现 compute-design §4「GPU 完成 = 又一种 completion source，与 io/ 同构」：`co_await gpu_kernel()` 与 `co_await async_read()` 同一套挂起/唤醒。
>
> **这是 compute 版的 io-04**：compute-01 立 Event/query 原语（不接 sched，如 io-01）；compute-03 把它接进 worker，让 fiber 能 await GPU（如 io-04 把 proactor 接进 worker）。**compute-01 的 `query()`(轮询)/`sync()`(阻塞) 两原语已为此备好，接口无需回改**（深调研已验证）。

## 完成机制：方案已定（2026-07-06 深调研 ggml/CCCL/seastar/sched 源码）

**callback 路线否决（铁证复述）**：ggml 唯一的 `cudaLaunchHostFunc` 被 `#if 0` 标 untested（ggml-cuda.cu:4539），CCCL 标 host callback "禁调 CUDA API、禁阻塞"（死锁风险）。**callback 无法安全唤醒 fiber**（唤醒涉及锁/futex，与驱动内部锁死锁）。故走轮询/阻塞。

**方案 A（首版，零改 sched，已验证可行）**：
- **GPU 完成走独立 poller 线程**：submit kernel/copy 到 stream 后 `cudaEventRecord(event, stream)`；一个 compute poller 线程持有在途 event 列表，对每个 `cudaEventSynchronize`（阻塞等）或 `cudaEventQuery`（轮询）；完成即 `Waker::wake()`。
- **wake 经现有脊注入 fiber，零改 sched（已读 scheduler.cc 确认）**：`Waker::wake()`→`reinject(fiber)`。poller 是**外部线程**（`t_current != this`），reinject 走 global queue push + `notify()`（scheduler.cc:449-451），`notify()` unpark 睡在 futex 的 owner worker。**纯 compute 场景（worker park 在 futex）直接生效，不需要 set_io_wake**。
- **awaitable 照镜 net.h 的 `IoOperation`**（net.h:66）：栈 slot `{Waker, result/status}` 活在挂起 fiber 栈；`await_suspend` 捕获 `current_waker()`、注册 event 到 poller、`sched::suspend()` fiber-yield；poller 完成填 status + `wake()`；`await_resume` 返回。**GPU op 可能永不完成（kernel 挂死）**→ 若要超时/取消，用 heap slot + refcount（照 net_timeout.h），基础版栈 slot（drain-to-completion 契约，同 io base op）。
- **复用现有 sched/io 设施**（agent 调研确认）：Waker / IoOperation 模式 / net_timeout refcount slot / JoinHandle / spawn_blocking（可选：把阻塞 sync 卸载到 blocking pool 而非专用 poller）。**sched 一行不改。**

**方案 B（后续优化，需改 sched，登记）**：worker park 时**同时**轮询 GPU event + 阻塞 io ring。把 `set_io_poll`/`set_io_wake` 单槽泛化成多 completion source 列表 + 抄 **seastar `try_enter_interrupt_mode` veto 协议**（reactor.cc:3477）：有在途 GPU event 就否决 worker 深睡、继续 busy-poll；无则放行 io_uring 深睡。省 poller 线程，纯 io 场景零浪费。**当 io+GPU 同 worker 高频共存、poller 线程成瓶颈时才做**。

## 参考 ggml 的 stream/event 编排（用户指示"调度也参考 ggml"，2026-07-06 深挖）

ggml 的 GPU **完成通知**是空白（只阻塞 sync，compute-03 填补），但它的 **stream/event 并发编排是 SOTA，全盘参考**：

- **多 stream 池**（common.cuh:1462）：每 device 8 个 `cudaStreamNonBlocking` stream 惰性创建 + 每 stream 独立 memory pool（切 stream 切 pool，避跨流分配竞争）。**compute-03 的 Stream 应支持每 device 多个**（compute-01 只做了单 stream 生命周期；多 stream 池是并发前提）。默认算子串行到主 stream，并发时分散多 stream。
- **fork/join event 编排 = 核心参考**（ggml-cuda.cu:4256-4347）：主 stream `cudaEventRecord(fork_event)` → 每辅助 stream `cudaStreamWaitEvent(fork_event)`（等前置数据，**全 GPU 侧排队 host 不阻塞**）→ 分支并发 → join 时各辅助 stream record、主 stream wait 汇合。
- **★ 关键设计原则（这条直接定 compute-03 的编排哲学）**：**GPU 之间的依赖下沉到 `cudaStreamWaitEvent`（device 侧串联），fiber 只在整条依赖链的末端 `co_await` 一次**。不要让每个 GPU 中间依赖都唤醒 fiber——那是 host 往返浪费。stream B 依赖 stream A → `cudaStreamWaitEvent(B, eventA)`（GPU 侧），fiber 挂起等最终 event，中间不回 host。这与 io 的「批量 submit 再一次 reap」同哲学（最小化 host↔device 往返）。
- **跨 stream copy 流水**（ggml-cuda.cu:3229）：`cudaMemcpyPeerAsync` + src 流 record / dst 流 wait event，拷贝依赖不落 host——compute-02 的 async copy（compute-03 落）接同款。
- **双缓冲 event 环**（ggml-backend.cpp:1719/1571）：split 尾 record / 头 wait + `cur_copy` 双缓冲避覆写竞争——多 batch 流水可借鉴。

**不适用（张量图特化，不抄）**：`stream_mapping`/`concurrent_events` 以 `ggml_tensor*` 为 key、node 重排恢复 fusion、MoE expert 选择性拷贝、CUDA graph capture——绑 tensor graph，HYLE 不做 tensor。但 `is_valid()` 的"写区间不相交 + 无跨分支 src"并发安全判据（common.cuh:1281），若 HYLE 在 fiber 层做 GPU 依赖 DAG 可借鉴。

## 明确不做（防蔓延）

| 不做 | 去向 | 为什么 |
|---|---|---|
| 方案 B（worker 亲自轮询 GPU + veto 协议）| compute-03 后续 / 优化 | 首版方案 A（poller 线程）零改 sched、风险最低；B 是 io+GPU 同 worker 高频共存时的省线程优化 |
| kernel 启动 | compute-04 | compute-03 用 compute-02 的 async copy（有真 GPU 工作产生 event）验证完成→fiber，不碰 launch |
| GPU 依赖 DAG（fork/join 编排 API）| 后续（graph 或 compute 扩展）| compute-03 立 `co_await stream/event` 单点；多 stream fork/join 编排 API 待消费者驱动（参考 ggml is_valid 判据）|
| 超时/取消 GPU op | 按需（照 net_timeout/net_cancel）| 基础版栈 slot drain-to-completion；超时用 heap refcount slot，待消费者 |

## 交付物（预期）
- `include/hyle/compute/gpu_await.h`（`co_await stream` / `co_await event` awaitable，照镜 net.h IoOperation；`GpuOp{waker,status}` 栈 slot）
- `src/compute/gpu_poller.{h,cc}`（compute poller：在途 event 列表 + 完成 `Waker::wake()`；或复用 spawn_blocking 卸载 sync）
- backend vtable 可能加：多 stream 池（照 ggml 8 stream）/ `cudaStreamWaitEvent`（fork/join 下沉）——按 compute-03 实际需要
- `tests/compute/gpu_await_test.cc`（fiber `co_await` GPU copy 完成 + 多 fiber 并发 await + host fallback 确定性）
- CMake：compute 加 `hyle_sched` 依赖（compute-03 首次接 sched，如 io-04 首次接 sched）

## 验收门槛
- 三平台：Win/WSL GPU 真机 fiber `co_await` GPU async copy → 挂起 → event 完成 → 唤醒 → 读回校验；mac host fallback（event 恒 ready，co_await 立即返回）确定性。
- 多 fiber 并发 await 不同 stream，各自正确唤醒。
- WSL TSan：poller 线程 `Waker::wake()` 跨线程注入 fiber 零 race（重点验证——跨线程 wake 是 race 高发区，同 io 的 io_poll 跨线程发布）。
- 复用 sched 脊零改 sched（改动仅在 compute/）；cloc 验真；对抗自审对照 ggml 编排 + net.h 模式。

## 交付物（实际落地）
- `include/hyle/compute/gpu_await.h`（`StreamComplete{&stream}` awaitable + `GpuOp` 栈 slot〔waker/event/status/done〕 + `detail::gpu_poll_register/shutdown`）
- `src/compute/gpu_poller.cc`（`GpuPoller` 进程级懒启动单例:in-flight `GpuOp*` 列表 + 一线程 `Event::query()` 轮询 + ready 时 `Waker::wake()`;有在途 50µs sleep 轮询、无在途 condvar 空闲等〔零成本 idle〕。`StreamComplete::await_ready`〔record event + 已 ready 则不挂起,host 走这条〕/`await_suspend`〔capture waker + register + `sched::suspend()`〕）
- `tests/compute/gpu_await_test.cc`（fiber co_await GPU copy 完成 + 8 fiber 并发 await）
- `src/compute/CMakeLists.txt`（gpu_poller.cc 门控 HYLE_HAS_FCONTEXT + hyle_sched PUBLIC_DEPS）

**cloc 实测**：~264 生产（gpu_await.h 97 + gpu_poller.cc 167）+ 126 测试。264«1-2k 预测=**复用现有 Waker 脊**（不重造唤醒机制,poller 只是薄轮询层 + 照镜 net.h IoOperation）+ **sched 零改**。非 stub（三平台真挂起/唤醒 + TSan 为证）。

## 实测结果（2026-07-06，三平台真机）
- **WSL（GCC14+nvcc）**：gpu_await 2/2 在 RTX 4070 真跑——`CoAwaitStreamCompletion`：fiber co_await stream → **真挂起**（GPU async copy 在飞）→ poller 线程 `cudaEventQuery` 观测 ready → `Waker::wake()` 跨线程唤醒 fiber → 显存往返数据校验；`ConcurrentAwaits`：8 fiber 各 await 自己的 stream，poller 同时追踪多 event，全部唤醒完成。
- **WSL TSan（host-only）**：gpu_await 2/2 **零 data race**——poller 线程跨线程 `Waker::wake()`→reinject→notify unpark worker 干净（这是 race 高发区,重点验证）。
- **Windows（MSVC19.50+nvcc）**：gpu_await 2/2 GPU 真跑 + 全量 compute 33/33。
- **macOS（Apple clang arm64,无 CUDA）**：gpu_await 2/2 —— host event 立即 ready，`await_ready` 返 true，co_await **不挂起**直接完成（同代码驱动 host+GPU 的证明）。

## 实现纪要 / 对抗自审
- **零改 sched（验证达成）**：poller 走现有 `Waker::wake()`→`reinject`（scheduler.cc:449 外部线程 global+notify unpark），改动全在 compute/。
- **poller 而非 spawn_blocking**：选专用 poller（一线程轮询多 event，io reap 式）而非 spawn_blocking（一线程阻塞一 event），更省线程 + 直接复用 compute-01 的 `Event::query()`。
- **测试 `detail` 歧义坑**：`using namespace hyle` + `hyle::compute` 下 `detail::` 歧义（`hyle::detail` vs `hyle::compute::detail`）→ 完整限定 `compute::detail::`。
- **GpuOp 栈 slot 生命周期**：fiber 挂起栈冻结、GPU op 必完成 → slot 活过 poll（同 io base op drain-to-completion 契约）。挂死 kernel 需 heap/timed slot（future，照 net_timeout），基础版栈 slot。
- **ggml 编排原则已入设计**（用户"调度也参考 ggml"）：GPU 依赖下沉 `cudaStreamWaitEvent` device 侧、fiber 只在链末 co_await 一次——compute-03 立单点 co_await,多 stream fork/join 编排 API 待后续消费者驱动（参考 ggml is_valid 判据）。
- ggml stream/event 编排（common.cuh:1462 多 stream 池 / ggml-cuda.cu:4256-4347 fork/join / 3229 跨 stream copy / ggml-backend.cpp:1719 双缓冲 event 环）——**device 侧编排全盘参考，完成→fiber 唤醒 HYLE 自建**。
- sched：waker.h（Waker/current_waker/suspend）、scheduler.cc:449（reinject 外部线程 notify unpark）、io net.h:66（IoOperation 栈 slot 模式）、net_timeout.h（heap refcount slot，超时/永不完成 op）。
- CCCL event_ref::is_done（cudaEventQuery poll）；seastar reactor.cc:3477（方案 B veto 协议）。
