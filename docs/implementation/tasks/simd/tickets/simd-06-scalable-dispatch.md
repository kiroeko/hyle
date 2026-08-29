# SIMD-06：Scalable backend、dispatch 与交付

- ticket state：`QUEUED`
- kind：product review/refactor
- depends on：SIMD-05
- budget：1 工程日；性能预算 60 分钟

审查 SVE/RVV sizeless 合同、Runtime 表、exact/native/fat-kernel、Platform CPU facts、初始化/并发和
source/static/shared 组件交付。真机只对现有机器签性能；SVE/RVV emulator 只签 correctness。runtime
比较必须计入双方等价 dispatch 成本，source-primary 入口必须独立可用。

不得新增 per-ISA DLL、稳定跨工具链 C++ ABI 或 Native FFI；不得把 internal selector 暴露为普通用户面。
