# SIMD-06：Scalable backend、dispatch 与交付

- ticket state：`QUEUED`
- kind：product review/refactor
- depends on：SIMD-05
- budget：1 工程日；性能预算 60 分钟

审查 SVE/RVV sizeless 合同、Runtime 表、exact/native/fat-kernel、Platform CPU facts、初始化/并发和
source/static/shared 组件交付。当前 public `Runtime` 函数指针表改为 private implementation table；普通
公开面只保留实际 selected target/features 的小型只读诊断，不再从 `hyle::simd` 转发 Platform detector。
Highway 是 SIMD kernel target 的唯一 decision authority，Platform 只拥有通用 CPU/OS facts。

真机只对现有机器签性能；SVE/RVV emulator 只签 correctness。runtime 比较必须计入双方等价 dispatch
成本，source-primary 入口必须独立可用。初始化一次构造整张表，不允许每个 op 分别探测或缓存
未初始化的 Highway trampoline。

不得新增 per-ISA DLL、稳定跨工具链 C++ ABI 或 Native FFI；不得把 internal selector 暴露为普通用户面。
