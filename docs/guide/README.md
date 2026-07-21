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
| [`c-mem.md`](./c-mem.md) | **mem 纯 C ABI**：`hyle_allocator` vtable、system 分配器、硬容量 SharedPool、软容量 SharedPoolCached、StackAllocator/Arena/GrowablePool/HugeRegion 不透明句柄、hard/cached typed 池宏、进程级大页/NUMA 策略。 |
| [`c-time.md`](./c-time.md) | **time 纯 C ABI**：单调/墙钟、饱和 deadline、sleep、TimerWheel/CommonTimeoutQueue opaque handle、visitor 与生命周期合同。 |
| [`c-obs.md`](./c-obs.md) | **obs 纯 C ABI**：结构化 record、quiescent sink 生命周期、默认 sink，以及 C/FFI runtime trace 与原生零成本 trace 的边界。 |
| [`c-util.md`](./c-util.md) | **util 纯 C ABI**：RNG POD、Ref/ScopeGuard/Registry/SpinLock 类型擦除句柄、demangle buffer-fill、所有权和并发合同。 |
| [`c-container.md`](./c-container.md) | **container 纯 C ABI**：字符串/路径、StringBuf、运行时擦除 flat Swiss map/vector、四类队列与 append-only vector；所有权、失效和并发合同。 |
| [`c-simd.md`](./c-simd.md) | **SIMD 纯 C ABI**：247 个 whole-buffer 入口、低精度/排序 POD、RNG opaque handle、运行时 ISA 诊断、调用合同和 FFI 绑定方式。 |
| [`c-sched.md`](./c-sched.md) | **sched 纯 C ABI**：worker-stack/fiber 双腿、Waker、优雅关闭、同步/消息/取消/blocking pool 的类型擦除入口、生命周期和外语绑定合同。 |
| [`c-io.md`](./c-io.md) | **io 纯 C ABI**：proactor/completion、网络/文件提交、后端诊断、multishot/fixed-file、共享内存，以及 C 与 sched 的组合合同。 |
| [`io.md`](./io.md) | **IO C++ 使用指南**：per-worker proactor、TCP/UDP deadline/cancel、自动文件卸载、HTTP/HTTPS transport、DNS/signal 与专属快腿。 |
| [`c-core.md`](./c-core.md) | **core 纯 C ABI**：`hyle_version`（packed）+ `hyle_version_string`，跨语言诊断版本查询；纯 `extern "C"`，无 api 表/版本协商。 |
| [`simd-usage.md`](./simd-usage.md) | **SIMD 怎么选和怎么调**：普通动态派发、整库固定 ISA、固定 kernel、常用 batch 能力与调用合同；并标明 close-01 C ABI 迁移边界。 |
| [`usage-contract.md`](./usage-contract.md) | **怎么调才不出 UB**：为 SOTA 性能保留的调用安全红线（fiber 勿跨 `yield()` 持有 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离）。每条带机制+安全模式+后果。**热路径集成前必读。** |

各 `c-*.md` 不只列导出符号，也必须解释原生 C++ 的模板/RAII/协程形状在 C/FFI
里怎样组合出来。找不到稳定组合通路就是 C ABI 缺口，不能用“C 没有模板”略过；
`c-sched` §9 是非平凡组合 recipe 的样板。

## 三份文档的分工（别混）

- **guide（这里）**：正常怎么用 + 怎么调才安全 —— 使用者视角。
- **[`architecture/`](../architecture/)**：为什么这么造、各支柱选型、clay 资产负债 —— 贡献者视角。
- **[`samples/`](../../samples/)**：照着抄就能跑的示例工程 —— 上手最快。
