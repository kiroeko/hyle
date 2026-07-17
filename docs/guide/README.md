# HYLE 用户文档（docs/guide）

> 这个文件夹是**面向库使用者**的文档——把 HYLE 当依赖集成进自己工程的人看这里。
> 想读"为什么这么设计"的内幕（面向贡献者）去 [`../architecture/`](../architecture/)。
> 想看可运行的示例工程去仓库最外层 [`../../samples/`](../../samples/)。

## 本目录文档

| 文档 | 给谁 / 讲什么 |
|---|---|
| [`usage-guide.md`](./usage-guide.md) | **怎么用**：模块就绪度、构建选项全集、SIMD 编译变体模型、各模块最小用法、C-ABI 边界。**集成第一站。** |
| [`c-error.md`](./c-error.md) | **error 纯 C ABI**：`hyle_status` 所有权、工厂/查询、诊断 handler、`hyle_error_api` 表、导出清单与版本。**FFI / 跨编译器第一站。** |
| [`c-platform.md`](./c-platform.md) | **platform 纯 C ABI**：Parker/Thread/DynLib/VM/fs/cpu、`hyle_platform_api` 表、所有权与易用示例。 |
| [`simd-usage.md`](./simd-usage.md) | **SIMD 怎么选和怎么调**：普通动态派发、整库固定 ISA、固定 kernel、常用 batch 能力与调用合同；并标明 close-01 C ABI 迁移边界。 |
| [`usage-contract.md`](./usage-contract.md) | **怎么调才不出 UB**：为 SOTA 性能保留的调用安全红线（fiber 勿跨 `yield()` 持有 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离）。每条带机制+安全模式+后果。**热路径集成前必读。** |

## 三份文档的分工（别混）

- **guide（这里）**：正常怎么用 + 怎么调才安全 —— 使用者视角。
- **[`architecture/`](../architecture/)**：为什么这么造、各支柱选型、clay 资产负债 —— 贡献者视角。
- **[`samples/`](../../samples/)**：照着抄就能跑的示例工程 —— 上手最快。
