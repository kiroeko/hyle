# HYLE 设计文档

> **HYLE = HYbrid LEvels**。一个通用高性能 C++ infra——给 AI agent / 推理 / 训练框架共用的、贴近硬件的混合计算基底。是对 `slate-render` 中 clay infra 层的重构与扩大。
>
> 本目录是 HYLE 的**唯一信源（single source of truth）**：设计、背景、命名、源码位置、重构路线都集中在此。

## HYLE 是什么

**一个 C++ 基础设施库**（编译成带 C-ABI 的共享库），给上层（AI agent、训练/推理框架）当地基。把四样东西做在**同一套底座**上：调度（线程/进程/纤程/协程一个调度器）、内存（分配器/arena/cache 对齐/NUMA）、计算（CPU SIMD + GPU CUDA/Vulkan，算子借 ggml）、IO（网络 proactor + 文件 + 跨进程共享内存）。

**差异点是"同一套底座"**：GPU 算完、网卡收包、fiber 让出，都回到同一个调度器、唤醒同一个 fiber，而非四个子系统各开线程池、各跑事件循环互相打架。HYLE 不是"又一个 hashmap/日志库"，而是让计算/并发/内存/IO 无阻抗拼在一起的那层地基。

- **地基（继承 clay，约占 1/3）**：内存分配器 + 容器（flat hashmap/carray/cstring）+ 跨平台原子（catomics）+ 并行编排骨架（taskgraph）。
- **支柱（自造，约占 2/3，30–50k 行）**：统一调度（线程/进程/纤程/协程）、fiber、网络 IO、SIMD、异构计算。
- **体量**：HYLE 粗估 50–70k 行级，是 clay 的 2–3 倍。不是"重构那 2 万行"，是"以 clay 为地基的全新 infra"。

## 模块划分

```
HYLE infra（通用、不特化、贴硬件）
├─ mem/       极致内存管理     ← 继承 clay 分配器+arena，修 5 bug，加 NUMA/大页/cache-line
├─ container/ 超越 std 容器    ← 继承 hashmap/carray/cstring，加 SwissTable SIMD/SmallVector/intrusive/lock-free
├─ simd/      SIMD 抽象        ← 推倒重写：可移植 vector + 运行时 dispatch + AVX512 + bf16
├─ sched/     统一调度【核心】   ← 线程/进程/纤程/协程一张表：fiber 上下文 + per-core 无锁 run-queue + work-stealing
├─ io/        网络 IO          ← proactor（io_uring/IOCP/kqueue/epoll），与 sched 一体（fiber 遇 IO 让出）
├─ compute/   异构计算          ← device/stream/buffer/kernel 抽象，借上游 ggml 作算子供应商
└─ graph/     并行任务编排       ← 继承 clay taskgraph，剥离特化，跑在 sched 之上
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
| [`motivation.md`](./motivation.md) | **动机**：为什么是现在/为谁做——AI-native 四压负载、蜂群 agent 北极星、比 Seastar 多异构维度、极致=吃尽硬件独家加速 |
| [`engineering-charter.md`](./engineering-charter.md) | **工程章程**：动手前最终依据——C++23 标准/错误模型/命名宏、开发范式、测试方案、边界、对外 FFI 决策 |
| [`coding-conventions.md`](./coding-conventions.md) | **编码规范**：每一行怎么写才不破范式——std 分层/include/宏纪律、C-ABI 范式、错误处理、能力 dispatch 三层模型、clay 独特写法三张表 |
| [`clay-asset-liability.md`](./clay-asset-liability.md) | clay 资产负债表：六子系统逐个盘点（内存/DataModel/SIMD/容器/并发/taskgraph），含 5 个 bug、空位 grep 证据、继承/重写判断 |
| [`hyle-scope.md`](./hyle-scope.md) | scope 定义：定位、两条主轴、7 模块划分、50–70k 行规模、边界外明确不做的东西 |
| [`io-design.md`](./io-design.md) | 网络库选型：proactor 模型 + io_uring/IOCP/kqueue/epoll 四后端 + 只借 syscall 封装、调度绑 fiber |
| [`sched-design.md`](./sched-design.md) | 统一调度器（4 节完整）：fcontext 上下文切换 + per-core 无锁 run-queue/work-stealing + Waker completion 脊 + 协程/fiber 统一 |
| [`lifecycle-design.md`](./lifecycle-design.md) | **失败边界**：fail-fast 绝不 try-catch + 四类失败分类 + 取消(结构化并发/await点) + 背压 + 优雅关闭 + 故障隔离=进程边界 + 跨 C-ABI 错误约定 |
| [`compute-design.md`](./compute-design.md) | 异构计算选型：device/stream/kernel 两层抽象 + 借上游 ggml（adapter 隔离）+ shader 一等公民 + GPU completion 接 sched |
| [`simd-design.md`](./simd-design.md) | SIMD 选型：精简可移植 vector + 运行时 dispatch（clay 最大缺口）+ dtype(fp16/bf16) + 借 sleef 数学核 + container 探测 tier 澄清 |
| [`mem-design.md`](./mem-design.md) | 内存设计：通用 malloc 借 mimalloc + 自造专用分配器（arena/pool/stack/fiber栈/pinned）+ 修 5 bug + NUMA/大页/64B 对齐契约 |
| [`container-design.md`](./container-design.md) | 容器设计：唯一"造不借"的 T0 + 补 SwissTable SIMD 探测 + SmallVector/intrusive/lock-free 队列（供 sched）|
| [`roadmap.md`](./roadmap.md) | 实现路线图：模块依赖图（无环）+ 6 批施工序 + 关键路径 + 每批验收门槛 + 完整决策清单（已决+待决）|
| [`performance.md`](./performance.md) | 性能预期 vs std：时间/空间分列、drop-in 可比 vs std 没有的能力两桶、Amdahl 整体观、benchmark gate 验证 |
| [`references.md`](./references.md) | 参考库映射：clone 的 20+ 库 → HYLE 模块，借/参考标注、开工优先级（含 Intel oneMath/oneDNN）|
| [`usage-contract.md`](./usage-contract.md) | **调用安全边界**：为 SOTA 性能保留的特定调用约束（fiber 勿跨 yield 持有 thread_local；分配器跨线程 free 安全/勿 static+LTO 内联进 fiber 体），每条带机制+安全模式+后果 |

## 状态与下一步

- ✅ 资产负债表已出，scope 已定，io/compute 两根支柱已展开。
- ⏭️ **建议下一步深挖 `sched/`**——它是 `io/` 和 `compute/` completion 机制的共同地基，动它最值。关键设计点：fiber 上下文切换方案、协程与 fiber 如何统一、io/gpu completion 如何映射到唤醒哪个 fiber、per-core 无锁 run-queue + work-stealing。
- 其余支柱（`simd/` 选型、`mem/` 加固、`container/` 扩展）依赖序靠后。

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
