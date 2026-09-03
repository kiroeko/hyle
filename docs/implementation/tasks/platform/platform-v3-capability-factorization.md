# Platform v3：推理 host 能力补全与质因数迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / Platform /
  `platform-v3-capability-factorization` / `REOPEN`；
- baseline：`6bf023c90b6ec3d62e9892028e172a385671de1e`；前轮：`platform-v2`；
- trigger：`PLATFORM-FACTORIZATION-2026-09-03`；
- `PLAT3-R1`：已接受的全局能力账本把 physical-core/SMT/cache-domain topology
  (`PLAT-A-05`)、host memory budget (`PLAT-A-09`) 与 caller affinity save/restore (`PLAT-A-17`)
  判为当前 llama.cpp/ggml 推理旅程的 `MISSING/MUST`；旧 platform-v2 明确没有审计这些能力；
- `PLAT3-R2`：`PLAT-A-02` 已准入 scalable-vector width 与 per-thread AMX/SME enablement。当前只有
  coarse feature bool，不能完成 ggml kernel 选择、线程进入和恢复的完整旅程；
- `PLAT3-R3`：Util-core 的统计 RNG 明示不安全，而两个成熟基础库都提供 OS entropy；全局 owner
  裁决已把 secure seed mechanism (`PLAT-A-33`) 放入 Platform host factor；
- `PLAT3-R4`：接受的三因子边界证明 host facts/raw VM/NUMA/DynLib、native thread/wait 与同步
  filesystem 具有不同直接消费者和依赖成本。现有单一 factor 让只需 CPU facts 的 SIMD/Container
  传递取得 Time/fs/thread artifact，已满足 component dependency/delivery 变化的重开条件；
- `PLAT3-R5`：既有 MappedFile/residency 与 SharedMemory storage 最终 owner 已裁决为未来 `mem_vm`。
  本轮只冻结迁移 seam，不在 Mem 未依法重开前修改其产品 ownership；不能为赶进度留下双 owner；
- 2026-09-03 clean reference identities：llama.cpp
  `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`（随仓 ggml tree
  `b91bd1c0eadf24e7d56143f36942e1a838c1dd8b`）、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`、Folly
  `eeab4246827eae9a11c0fa22529bff1a258aa83a`、oneTBB
  `843162db8cf4b2531a4a6147b6ea3fd860d44668`、Boost
  `cb94f33fcb13e90922905c2d8105508e71fa6db4`；
- 初步受影响面：Platform capability/API audit；`platform`、`platform_thread`、`platform_fs` 的头、
  source、tests、direct edges 与 product façade；SIMD/Mem/Sched/IO 的直接消费；source/static/shared；
  correctness、P1、architecture/user/performance 文档；
- 初步 tier：`P1`。CPU facts 是初始化/重配控制面，thread park 可处于 runtime 热路，filesystem 是
  loader 控制面；API freeze 后按语义族分别使用结构成本或已有正常 consumer，不把 syscall 微点值
  混成一个“Platform 更快”结论；
- 暂定非目标：在 Mem round 前搬迁 mapping/shared-storage owner；async/direct/scatter I/O、scheduler
  placement policy、device facts/kernel dispatch、JIT/process/plugin policy、Native FFI/C API、稳定
  C++ ABI、benchmark-driven 入口、公共仪器改造与 Formal certification。

当前只登记足以击穿旧结论的新事实和新 component ID。closure card、完整双向能力审计、自然 C++
API freeze 与有限 `FIX` 集尚未完成，因此不授权产品实现。
