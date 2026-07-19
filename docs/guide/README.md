# HYLE 用户文档（docs/guide）

> 这个文件夹是**面向库使用者**的文档——把 HYLE 当依赖集成进自己工程的人看这里。
> 想读"为什么这么设计"的内幕（面向贡献者）去 [`../architecture/`](../architecture/)。
> 想看可运行的示例工程去仓库最外层 [`../../samples/`](../../samples/)。

## 本目录文档

| 文档 | 给谁 / 讲什么 |
|---|---|
| [`usage-guide.md`](./usage-guide.md) | **怎么用**：模块就绪度、构建选项全集、SIMD 编译变体模型、各模块最小用法、C-ABI 边界。**集成第一站。** |
| [`c-error.md`](./c-error.md) | **error 纯 C ABI**：`hyle_status` 所有权、工厂/查询、诊断 handler、自由函数导出清单。**FFI / 跨编译器第一站。** |
| [`c-platform.md`](./c-platform.md) | **platform 纯 C ABI**：Parker/Thread/DynLib/VM/mapped_file/fs/cpu/numa 不透明句柄与自由函数、named enum、所有权与易用示例。 |
| [`c-mem.md`](./c-mem.md) | **mem 纯 C ABI**：`hyle_allocator` vtable、system 分配器、MemoryArena/StackAllocator/Arena/GrowablePool/HugeRegion 不透明句柄、typed 池便利宏、进程级大页/NUMA 策略。 |
| [`c-time.md`](./c-time.md) | **time 纯 C ABI**：单调/墙钟、饱和 deadline、sleep、TimerWheel/CommonTimeoutQueue opaque handle、visitor 与生命周期合同。 |
| [`c-obs.md`](./c-obs.md) | **obs 纯 C ABI**：结构化 record、quiescent sink 生命周期、默认 sink，以及 C/FFI runtime trace 与原生零成本 trace 的边界。 |
| [`c-simd.md`](./c-simd.md) | **SIMD 纯 C ABI**：247 个 whole-buffer 入口、低精度/排序 POD、RNG opaque handle、运行时 ISA 诊断、调用合同和 FFI 绑定方式。 |
| [`c-core.md`](./c-core.md) | **core 纯 C ABI**：`hyle_version`（packed）+ `hyle_version_string`，跨语言诊断版本查询；纯 `extern "C"`，无 api 表/版本协商。 |
| [`simd-usage.md`](./simd-usage.md) | **SIMD 怎么选和怎么调**：普通动态派发、整库固定 ISA、固定 kernel、常用 batch 能力与调用合同；并标明 close-01 C ABI 迁移边界。 |
| [`usage-contract.md`](./usage-contract.md) | **怎么调才不出 UB**：为 SOTA 性能保留的调用安全红线（fiber 勿跨 `yield()` 持有 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离）。每条带机制+安全模式+后果。**热路径集成前必读。** |

## 三份文档的分工（别混）

- **guide（这里）**：正常怎么用 + 怎么调才安全 —— 使用者视角。
- **[`architecture/`](../architecture/)**：为什么这么造、各支柱选型、clay 资产负债 —— 贡献者视角。
- **[`samples/`](../../samples/)**：照着抄就能跑的示例工程 —— 上手最快。
