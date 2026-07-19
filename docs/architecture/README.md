# HYLE 设计文档

> **HYLE = HYbrid LEvels**。一个通用高性能 C++ infra——给 AI 推理引擎 / CAD·BIM·CAE 工程计算管线 / agent 共用的、贴近硬件的混合计算基底。是对 `slate-render` 中 clay infra 层的重构与扩大。
>
> 本目录是 HYLE **设计/背景/决策的信源**：设计推理、背景、命名、源码位置、重构路线都集中在此。**注意：代码本身才是实现的唯一信源**——文档描述与代码冲突时以代码为准（性能结论以 `performance.md` 实测为准）。

## HYLE 是什么

**一个 C++ 基础设施库**（编译成带 C-ABI 的共享库），给上层（AI 推理引擎、CAD/BIM/CAE 工程计算管线、训练框架、agent）当地基。把四样东西做在**同一套底座**上：调度（线程/进程/纤程/协程一个调度器）、内存（分配器/arena/cache 对齐/NUMA）、计算（CPU SIMD + GPU CUDA/Vulkan，算子借 ggml）、IO（网络 proactor + 文件 + 跨进程共享内存）。

**差异点是"同一套底座"**：GPU 算完、网卡收包、fiber 让出，都回到同一个调度器、唤醒同一个 fiber，而非四个子系统各开线程池、各跑事件循环互相打架。HYLE 不是"又一个 hashmap/日志库"，而是让计算/并发/内存/IO 无阻抗拼在一起的那层地基。

- **地基（继承 clay，约占 1/3）**：内存分配器 + 容器（flat hashmap/carray/cstring）+ 跨平台原子（catomics）+ 并行编排骨架（taskgraph）。
- **支柱（自造，约占 2/3）**：统一调度（线程/进程/纤程/协程）、fiber、网络 IO、SIMD、异构计算。
- **体量（2026-07-17）**：自有生产（`src`+`include`）已 **~53k**（其中 **simd ~34k ≈64%**）；纪律收口终态预期 **~65–80k** 生产，做满守界约 **~80–100k**。含测/文档的自有工程 ~105k；git 全跟踪含 third_party ~187k；再加 ggml 等足迹 ~200–350k+。旧「完成 ~28–40k」投影已作废。详见 [`hyle-scope.md`](./hyle-scope.md) 规模节与 [`../milestones/midterm-report-2026-07-scale-positioning.md`](../milestones/midterm-report-2026-07-scale-positioning.md)。

## 模块划分

```
HYLE infra（通用、不特化、贴硬件）
├─ mem/       极致内存管理     ← 继承 clay 分配器+arena，修 5 bug，加 NUMA/大页/cache-line
├─ container/ 超越 std 容器    ← 继承 hashmap/carray/cstring，加 SwissTable SIMD/SmallVector/intrusive/lock-free
├─ simd/      SIMD 抽象        ← 推倒重写：可移植 vector + 运行时 dispatch + AVX512 + bf16
├─ sched/     统一调度【核心】   ← 线程/进程/纤程/协程一张表：fiber 上下文 + per-core 无锁 run-queue + work-stealing
├─ io/        网络 IO          ← proactor（io_uring/IOCP/kqueue/epoll），与 sched 一体（fiber 遇 IO 让出）
├─ compute/   异构计算          ← device/stream/buffer/kernel 抽象，借上游 ggml 作算子供应商
└─ graph/     并行任务编排       ← 北极星② 编排：继承 clay taskgraph ~1k 骨架(剥离特化)+ 从零造增量重算引擎(脏标记/缓存,clay 零资产)+ 数据并行层;跑在 sched+compute 之上
```

## 贯穿性设计原则

1. **不引入第二个调度器**。HYLE 自带统一调度器（`sched/`），任何自带 event loop / executor 的库（asio、libuv、seastar）都会与之冲突——一律不整体引入，只借 syscall 薄封装层。
2. **三支柱底层统一为 completion**。CPU 线程、GPU stream、网络 IO 都是"产生 completion 的可调度资源"：io_uring/IOCP 的 CQ、`cudaEvent`/Vulkan `fence`、都接进同一套 fiber 挂起/唤醒。`co_await async_read()` 与 `co_await gpu_kernel()` 同构。
3. **跨平台代价讲在明处**。真 proactor 只有 Linux(io_uring)+Windows(IOCP)，mac/BSD 付模拟税；真 GPU 后端按厂商分裂，**shader 计算（Vulkan compute/SPIR-V）作跨厂商一等公民，不只绑 CUDA**（绑 CUDA = 锁死 NVIDIA）。
4. **第三方当可换供应商，降低耦合**。典型是 ggml：持续开源更新，但 iron 里 vendored 那份是旧快照、会一直变老。所以 HYLE 抽象自有且稳定、不暴露第三方类型，只在 adapter（如 `ggml_bridge`）里对接**上游 pin 版**，升级/接口变只动一个文件。
5. **回到 C、贴硬件、把性能做到极致**（用户称"回到原初权能"）。焦点：SIMD、CUDA/shader 混合计算、L1/L2 cache 命中、极致内存管理、超越 std 的容器。

## 文档索引

| 文档 | 内容 |
|---|---|
| ⭐ [`implementation-cheatsheet.md`](./implementation-cheatsheet.md) | **实现速查表（写代码前只读这一张）**：聚合每次实现都要用的硬规则——std 红线/失败模型/分层判据/C-ABI/调用边界/里程碑 CHECKLIST/构建环境。深层 nuance 按需深读下面各 design 文档 |
| [`motivation.md`](./motivation.md) | **动机**：为什么是现在/为谁做——四样全压负载、双北极星(推理引擎 ＋ CAD/BIM/CAE 工程管线)、比 Seastar 多异构维度、极致=吃尽硬件独家加速 |
| [`engineering-charter.md`](./engineering-charter.md) | **工程章程**：动手前最终依据——C++23 标准/错误模型/命名宏、开发范式、测试方案、边界、对外 FFI 决策 |
| [`coding-conventions.md`](./coding-conventions.md) | **编码规范**：每一行怎么写才不破范式——std 分层/include/宏纪律、C-ABI 范式、错误处理、能力 dispatch 三层模型、clay 独特写法三张表 |
| [`c-abi-design.md`](./c-abi-design.md) | **C++ 库如何全面导出 C ABI（why）**：类型擦除（编译期 `<T>`→运行时 size/enum）、三层架构（实现/C ABI/可选 C++ 便利层，ggml/oneDNN 实证）、版本协商删除决策（场景驱动非标配，clay 零/ggml 仅插件）、有状态调试类留 C++、错误跨边界、两条消费者线路的易用性 |
| [`c-abi-authoring-guide.md`](./c-abi-authoring-guide.md) | **给 HYLE 开发者的 C ABI 施工参考（how）**：收口某模块做 C ABI 照这走——导出面归类判据、六件套文件与接线、头/桥接骨架、桥接三铁律（opaque 句柄/零 reinterpret/fail-fast）、易用性 header-inline 层、纯 C smoke、三平台真测、收口 checklist。样板=mem |
| [`util-design.md`](./util-design.md) | **util 架构与维护边界**：七个能力族、原生 C++/类型擦除 C ABI 双面、依赖/性能边界和机械维护闸门。 |
| [`clay-asset-liability.md`](./clay-asset-liability.md) | clay 资产负债表：六子系统逐个盘点（内存/DataModel/SIMD/容器/并发/taskgraph），含 5 个 bug、空位 grep 证据、继承/重写判断 |
| [`hyle-scope.md`](./hyle-scope.md) | scope 定义：定位、两条主轴、7 模块划分、规模（生产已 ~53k / 终态纪律 ~65–80k / 足迹含 ggml ~200–350k+）、边界外明确不做的东西 |
| [`io-design.md`](./io-design.md) | 网络库选型：proactor 模型 + io_uring/IOCP/kqueue/epoll 四后端 + 只借 syscall 封装、调度绑 fiber |
| [`sched-design.md`](./sched-design.md) | 统一调度器（4 节完整）：fcontext 上下文切换 + per-core 无锁 run-queue/work-stealing + Waker completion 脊 + 协程/fiber 统一 |
| [`lifecycle-design.md`](./lifecycle-design.md) | **失败边界**：fail-fast 绝不 try-catch + 四类失败分类 + 取消(结构化并发/await点) + 背压 + 优雅关闭 + 故障隔离=进程边界 + 跨 C-ABI 错误约定 |
| [`compute-design.md`](./compute-design.md) | 异构计算选型：device/stream/kernel 两层抽象 + 借上游 ggml（adapter 隔离）+ shader 一等公民 + GPU completion 接 sched |
| [`borrowing-ggml-memory-model.md`](./borrowing-ggml-memory-model.md) | **借 ggml 的显存归属边界**：host 归 HYLE（mem 喂入/取出）、device 归 ggml（gallocr 自管），就像 llama.cpp——CUDA 结构性否决外部显存注入（gdb 坐实），HYLE 显存自管是 compute-04/06 自造 kernel 的领域 |
| [`simd-design.md`](./simd-design.md) | SIMD 总设计：D-tag/七后端、自有 math/dtype/算法核、Runtime/native/exact 双轨及历史决策 |
| [`simd-dispatch-guide.md`](./simd-dispatch-guide.md) | SIMD 当前部署与维护规范：ASCII 架构图、未知 CPU Runtime、整库 exact、用户 TU native、自定义 fat-kernel 与 C ABI 边界 |
| [`time-design.md`](./time-design.md) | time 依赖位置、时钟分层、双 timer source、所有权数据流、C++/C ABI 双面与维护硬门 |
| [`obs-design.md`](./obs-design.md) | obs 双通路：同步结构化日志的 quiescent sink 生命周期、C ABI trampoline，以及默认零成本/开启时单 Tracy client 的 trace 架构 |
| [`simd-vendor-surface.md`](./simd-vendor-surface.md) | 产品零 Highway/xsimd 的边界、外部 benchmark 对照和可选 simdutf 薄借 |
| [`mem-design.md`](./mem-design.md) | 内存设计：通用 malloc 借 mimalloc + 自造专用分配器（arena/pool/stack/fiber栈/pinned）+ 修 5 bug + NUMA/大页/64B 对齐契约 |
| [`container-design.md`](./container-design.md) | 容器设计：唯一"造不借"的 T0 + 补 SwissTable SIMD 探测 + SmallVector/intrusive/lock-free 队列（供 sched）|
| [`graph-design.md`](./graph-design.md) | **graph 支柱选型**：北极星② CAD/BIM/CAE 编排——依赖 DAG 增量重算(graph-01,clay taskgraph 零增量资产、从零造脏标记/缓存)+ 数据并行层(graph-02 parallel_for/reduce,归属拍板归 graph)；clay 继承面 ~800-1.5k 骨架 vs 从零 3-4k 增量引擎；纯编排层零自有执行资源(跑 sched+compute 之上)|
| [`graph-02-design.md`](./graph-02-design.md) | **graph-02 数据并行层设计**：parallel_for/reduce/scan/sort over range，跑 HYLE sched(不借 oneTBB 运行时,只借其设计)。五决策：分块=递归 split+grainsize(work-stealing 友好非静态均分)/reduce=fast+deterministic 双档(浮点结合性)/kernel=Body 收 subrange(接 simd 紧循环)/scan v1 就做/复用 graph-01 sched 基建。分块靠 graph-02、块内靠 simd 的零阻抗接缝 |
| [`roadmap.md`](./roadmap.md) | 实现路线图：模块依赖图（无环）+ 依赖层 P0-P6 拓扑序 + 关键路径 + 每阶段验收门槛 + 完整决策清单（已决+待决）|
| [`../performance/`](../performance/README.md) | 性能预期 vs std / vs SOTA + 实测对标(已独立成 `docs/performance/` 目录:总账 ledger + 按模块分档的 benchmarks) |
| [`../milestones/midterm-report-2026-07-scale-positioning.md`](../milestones/midterm-report-2026-07-scale-positioning.md) | **中期报告（2026-07-17）**：体量口径、终态剧本、~100k 时的规模定位 |
| [`references.md`](./references.md) | 参考库映射：clone 的 20+ 库 → HYLE 模块，借/参考标注、开工优先级（含 Intel oneMath/oneDNN）|
| [`usage-contract.md`](../guide/usage-contract.md) | **调用安全边界**：为 SOTA 性能保留的特定调用约束（fiber 勿跨 yield 持有 thread_local；分配器跨线程 free 安全/勿 static+LTO 内联进 fiber 体），每条带机制+安全模式+后果 |

## 进度

进度**不在此维护**（避免与里程碑表 drift）——逐里程碑的真实状态 + cloc 实测 LOC + clay API diff 见 [`../milestones/README.md`](../milestones/README.md)，那是**唯一进度信源**。本目录只管设计、背景、决策这些慢变内容。

一句话现状（粗粒度）：地基层（error/platform/mem/container/time/util/obs）立住;**sched 支柱 01~11 全 ✅ 收口**（work-stealing + Waker 脊 + conditional-notify 防惊群 + CV/Mutex/WaitGroup/Latch/Channel,对标 marl 全线追平/反超 + Task/co_await→fiber 让出 + spawn_on 绑核 shard-per-core 基座 + JoinHandle/when_all·when_any + spawn_blocking + 并发原语补全 + 优雅关闭 + **结构化取消 CancelToken 树(shard-per-core 无锁,借 tokio-util 拓扑+folly callback)**;栈 guard page/分级待）;**simd 支柱完成内部双轨收口**（387 个公开/selector 寄存器名 + scalar/SSE4.2/AVX2/AVX512/NEON/SVE/RVV 七后端，内部 Runtime、整库 exact、源码 TU native、自定义 fat-kernel；math/dtype/sort/bitpack/RNG 产品核全 HYLE，外部 Highway/xsimd 仅 benchmark 对照；Windows/WSL/macOS/SDE/SVE/RVV 多 VL 全绿；发布给用户的唯一入口仍等待 close-01 纯 C shim）;**io 支柱大面积落地**（proactor 三后端全真 op 无 stub + co_await socket/tcp/udp + shard-per-core + 带超时 io + multishot + TLS + 异步 DNS/signal + **HTTP/1.1 server+client**,双平台+TSan;file ✅ / shmem 待 / HTTP2 后置）;**compute 支柱破土**（compute-01 ✅：device/stream/event 脊 + event-poll completion 原语 + CUDA/host 双后端,双平台 RTX 4070 真机 10/10 + WSL TSan;device buffer/kernel launch/接 sched worker loop 留 compute-02~06）;**graph 支柱破土**（graph-01/02 数据并行层已接 sched/simd，细节见 milestones）。

---

## 附：项目背景与命名

### 命名

**HYLE = HYbrid LEvels**，拼字取自亚里士多德哲学概念「原初质料 hyle（ὕλη）」——万物成形前那团未分化的底料。

- 结构对仗前身 **CLAY = Core LAYers**（2+2 拆分，拼成有神话分量的真词）。
- `HY = Hybrid` 编码混合异构计算；`LE = Levels` 接住 clay 的「层」DNA 并升维（Layers → Levels）。
- 延续 slate / iron / clay / cycles 的「短、硬、材料/工业词」家族基调。

### 命名哲学

回到 C、抛弃过度抽象、贴近底层硬件、混合异构计算、把性能做到极致（用户称之为「回到原初权能」）。

### 源码位置

- **clay 源码（重构对象）**：`C:\public\project-hyle\slate-render\slate\infra\` 与 `C:\public\project-hyle\slate-render\iron\infra\`（两份镜像拷贝，命名空间 `clay::`）。
- **HYLE 工作目录**：`C:\public\project-hyle\hyle`（本目录）。
- **相关 TS 项目**：`C:\public\workspace\axiomate`。

### clay 是什么

跨平台 C++ 运行时库 / 共享库（默认 SHARED/DLL，slate 线实际静态链），命名空间 `clay::`。提供 infra 原语：`sys/`（allocator/memory_arena/catomics/log/os/time/flat hashmap/murmur_hash）、`parallel/`（thread_pool/worker_thread/async_value + plugins/taskgraph 并行 DAG）、`math/`（half/float16/SIMD/consts）、`utils/`（result/status/object_pool/ref_counted/registry/type_traits）。

为跨 DLL 边界做过专门规避：`make_XXX/destroy_XXX` 接口、`carray`/`c-hash` 等 C-API 对象、flat hashmap 替代 `std::unordered_map`。

### clay 现状（一句话）

19 年项目、自有代码 ≈2.9 万行、已停更。好地基（内存/SIMD/线程池/容器/原子）是真的，但 README_Clay 美化过现状——fiber 调度器零实现、自有 GPU 计算层是空门面、DataModel 不存在。这就是「scope 不足以支撑野心」的代码证据，也是 HYLE 要补的方向。详细逐项盘点见 [`clay-asset-liability.md`](./clay-asset-liability.md)。
