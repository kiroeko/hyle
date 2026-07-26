# HYLE 设计文档

> **HYLE = HYbrid LEvels**。一个通用高性能 C++ infra——给 AI 推理引擎 / CAD·BIM·CAE 工程计算管线 / agent 共用的、贴近硬件的混合计算基底。是对 `slate-render` 中 clay infra 层的重构与扩大。
>
> 本目录是 HYLE **当前架构、边界和长期设计决策的信源**。代码决定实现事实；性能结论只以
> [`../performance/`](../performance/README.md) 的绑定 commit 实测为准；任务状态只在
> [`../milestones/`](../milestones/README.md) 维护。完整分层规则见 [`../README.md`](../README.md)。

## HYLE 是什么

**一个 C++ 基础设施库**（可组装成带 C ABI 的共享/静态库，也可按授权交付完整源码），给上层
（AI 推理引擎、CAD/BIM/CAE 工程计算管线、训练框架、agent）当地基。把四样东西做在**同一套底座**上：调度（线程/进程/纤程/协程一个调度器）、内存（分配器/arena/cache 对齐/NUMA）、计算（CPU SIMD + GPU CUDA/Vulkan，算子借 ggml）、IO（网络 proactor + 文件 + 跨进程共享内存）。

**差异点是"同一套底座"**：GPU 算完、网卡收包、fiber 让出，都回到同一个调度器、唤醒同一个 fiber，而非四个子系统各开线程池、各跑事件循环互相打架。HYLE 不是"又一个 hashmap/日志库"，而是让计算/并发/内存/IO 无阻抗拼在一起的那层地基。

- **地基（继承 clay 并重新收口）**：内存分配器 + 容器（flat hashmap/carray/cstring）+ 跨平台原子（catomics）+ 并行编排骨架（taskgraph）。
- **支柱（自造）**：统一调度（线程/进程/纤程/协程）、fiber、网络 IO、SIMD、异构计算。二者是设计来源分类，不是 LOC 比例预算。
- **范围与体量口径**：见 [`hyle-scope.md`](./hyle-scope.md)；当前任务状态只在
  milestone 记录中维护，架构入口不复制施工流水。

## 模块划分

```
HYLE infra（通用、不特化、贴硬件）
├─ 基础基质
│  ├─ core/      宏、基础类型和零依赖 intrusive/atomic 原语
│  ├─ error/     统一失败值与跨 ABI 诊断
│  ├─ platform/  CPU/thread/wait/VM/fs/dynlib 跨 OS 机制
│  ├─ time/      typed clocks、deadline、thread sleep 与 timer queues
│  ├─ util/      无更合适 owner 的窄通用工具
│  └─ obs/       日志、trace 与 instrumentation hook
├─ mem/          极致 host 内存管理：allocator/region/pool/NUMA/大页/fiber stack
├─ container/    超越 std 容器：SwissTable/SmallVector/intrusive/lock-free
├─ simd/         SIMD 抽象：可移植 vector + 运行时 dispatch + dtype/math/算法核
├─ math/         标量数值类型、常量与基础数学能力
├─ sched/        统一调度【核心】：fiber + per-core run-queue + work-stealing + coroutine
├─ io/           proactor IO：io_uring/IOCP/kqueue/epoll，与 sched completion 一体
├─ compute/      异构计算：device/stream/buffer/kernel，隔离 ggml 与原生 backend
└─ graph/        并行任务编排：增量重算 + 数据并行，运行在 sched/compute 之上
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
| [`c-abi-design.md`](./c-abi-design.md) | **C++ 原生 API 与独立 C/FFI 产品面（why）**：能力映射而非机械镜像、类型擦除、错误边界、跨编译器 ABI 与两条消费者线路的易用性 |
| [`c-abi-authoring-guide.md`](./c-abi-authoring-guide.md) | **给 HYLE 开发者的 C ABI 施工参考（how）**：导出面归类判据、文件与接线、opaque control、零 reinterpret、FFI 数据校验、时序误用 fail-fast、纯 C smoke 与跨平台验证；mem 是已审查样板 |
| [`core-design.md`](./core-design.md) | **core 零依赖根**：版本单源、静态 cache layout 与 runtime platform 分工、宏/可见性边界、Folly/Boost-shaped auto-unlink intrusive primitive 及 C/C++ 双面 |
| [`build-system.md`](./build-system.md) | **项目组织的可执行描述**：组件依赖闭包、惰性第三方、static/shared、源码嵌入、runtime staging、测试归属及 install/package 待闭环边界 |
| [`licensing.md`](./licensing.md) | **许可与第三方边界**：HYLE 暂时闭源的 proprietary notice、SPDX `LicenseRef`、vendored/ported/fetched 许可证归属、安装分发与将来开源切换规则 |
| [`util-design.md`](./util-design.md) | **util 架构与维护边界**：七个能力族、原生 C++/类型擦除 C ABI 双面、依赖/性能边界和机械维护闸门。 |
| [`clay-asset-liability.md`](./clay-asset-liability.md) | clay 资产负债表：六子系统逐个盘点（内存/DataModel/SIMD/容器/并发/taskgraph），含 5 个 bug、空位 grep 证据、继承/重写判断 |
| [`hyle-scope.md`](./hyle-scope.md) | scope 定义：定位、模块分层、四项 `cloc` 口径、第三方边界、规模纪律与明确不做的东西 |
| [`io-design.md`](./io-design.md) | 网络库选型：proactor 模型 + io_uring/IOCP/kqueue/epoll 四后端 + 只借 syscall 封装、调度绑 fiber |
| [`sched-design.md`](./sched-design.md) | 统一调度器（4 节完整）：fcontext 上下文切换 + per-core 无锁 run-queue/work-stealing + Waker completion 脊 + 协程/fiber 统一 |
| [`lifecycle-design.md`](./lifecycle-design.md) | **失败边界**：fail-fast 绝不 try-catch + 四类失败分类 + 取消(结构化并发/await点) + 背压 + 优雅关闭 + 故障隔离=进程边界 + 跨 C-ABI 错误约定 |
| [`error-design.md`](./error-design.md) | **error 产品形状**：8 字节 Status、`std::expected` Result、分层上下文、首错聚合/格式化工具、C/FFI 所有权，以及 error/obs/fatal 的唯一职责边界 |
| [`platform-design.md`](./platform-design.md) | **platform 跨 OS 机制层**：CPU 系统/进程视图区分、Thread/Parker、VM/NUMA、fs/mmap/dynlib，及与 mem/sched/io/compute 的所有权边界 |
| [`compute-design.md`](./compute-design.md) | 异构计算选型：device/stream/kernel 两层抽象 + 借上游 ggml（adapter 隔离）+ shader 一等公民 + GPU completion 接 sched |
| [`borrowing-ggml-memory-model.md`](./borrowing-ggml-memory-model.md) | **借 ggml 的显存归属边界**：host 归 HYLE（mem 喂入/取出）、device 归 ggml（gallocr 自管），就像 llama.cpp——CUDA 结构性否决外部显存注入（gdb 坐实），HYLE 显存自管是 compute-04/06 自造 kernel 的领域 |
| [`simd-design.md`](./simd-design.md) | SIMD 总设计：D-tag/七后端、自有 math/dtype/算法核、Runtime/native/exact 双轨及历史决策 |
| [`simd-highway-capability-ledger.md`](./simd-highway-capability-ledger.md) | SIMD × Highway 实际能力账本：类型/内存/算法/数学/dispatch、LOC 差因、P0/P1 与明确不做项 |
| [`simd-dispatch-guide.md`](./simd-dispatch-guide.md) | SIMD 当前部署与维护规范：ASCII 架构图、未知 CPU Runtime、整库 exact、用户 TU native、自定义 fat-kernel 与 C ABI 边界 |
| [`time-design.md`](./time-design.md) | time 依赖位置、时钟/算术/sleep 分层、radix/common queues、所有权数据流、C++/C ABI 双面与维护硬门 |
| [`obs-design.md`](./obs-design.md) | obs 双通路：同步结构化日志的 quiescent sink 生命周期、C ABI trampoline，以及默认零成本/开启时单 Tracy client 的 trace 架构 |
| [`simd-vendor-surface.md`](./simd-vendor-surface.md) | 产品零 Highway/xsimd 的边界、外部 benchmark 对照和可选 simdutf 薄借 |
| [`mem-design.md`](./mem-design.md) | mem 当前整体设计：host-RAM 边界、模块化 allocator/region/pool/stack/diagnostics、C++/C 能力映射、所有权与失败合同 |
| [`container-design.md`](./container-design.md) | 容器设计：唯一"造不借"的 T0 + 补 SwissTable SIMD 探测 + SmallVector/intrusive/lock-free 队列（供 sched）|
| [`graph-design.md`](./graph-design.md) | **graph 支柱选型**：北极星② CAD/BIM/CAE 编排——依赖 DAG 增量重算(graph-01,clay taskgraph 零增量资产、从零造脏标记/缓存)+ 数据并行层(graph-02 parallel_for/reduce,归属拍板归 graph)；clay 继承面 ~800-1.5k 骨架 vs 从零 3-4k 增量引擎；纯编排层零自有执行资源(跑 sched+compute 之上)|
| [`graph-02-design.md`](./graph-02-design.md) | **graph-02 数据并行层设计**：parallel_for/reduce/scan/sort over range，跑 HYLE sched(不借 oneTBB 运行时,只借其设计)。五决策：分块=递归 split+grainsize(work-stealing 友好非静态均分)/reduce=fast+deterministic 双档(浮点结合性)/kernel=Body 收 subrange(接 simd 紧循环)/scan v1 就做/复用 graph-01 sched 基建。分块靠 graph-02、块内靠 simd 的零阻抗接缝 |
| [`roadmap.md`](./roadmap.md) | 实现路线图：完整 target/header/cross-cut 依赖、P0-P6 构建拓扑、R0-R6 模块严格收口顺序、验收门槛与慢变决策 |
| [`../performance/`](../performance/README.md) | 性能预期 vs std / vs SOTA + 实测对标(已独立成 `docs/performance/` 目录:总账 ledger + 按模块分档的 benchmarks) |
| [`../milestones/midterm-report-2026-07-scale-positioning.md`](../milestones/midterm-report-2026-07-scale-positioning.md) | **中期报告（2026-07-17）**：体量口径、终态剧本、~100k 时的规模定位 |
| [`references.md`](./references.md) | 参考库映射：clone 的 20+ 库 → HYLE 模块，借/参考标注、开工优先级（含 Intel oneMath/oneDNN）|
| [`usage-contract.md`](../guide/usage-contract.md) | **调用安全边界**：为 SOTA 性能保留的特定调用约束（fiber 勿跨 yield 持有 thread_local；分配器跨线程 free 安全/勿 static+LTO 内联进 fiber 体），每条带机制+安全模式+后果 |

## 文档职责

本目录不维护完成百分比、测试计数、某次 benchmark 数字或下一步排期。模块设计稳定后应
合并到这里；施工过程留在 milestones，性能证据留在 performance，用户 recipe 留在 guide。

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
