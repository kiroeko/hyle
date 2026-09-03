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

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform` / Platform /
  `platform-v3-capability-factorization` / `REOPEN`；baseline 与 trigger 见上；
- 真实消费者：SIMD/Container 的 CPU facts，Mem 的 raw VM/NUMA 与 loader storage，Sched 的 worker
  placement，IO 的 host budget/backend loading；北极星是最新锁定 llama.cpp/ggml 的 physical-core
  sizing、NUMA allocation、host-memory planning、dynamic backend 与 model-loading journey；
- 本 factor 范围：CPU/ISA/cache/topology、host memory、NUMA、raw anonymous VM、DynLib、process
  environment 与 secure OS entropy；现有 `platform` component ID 保留为该 host factor，避免伪兼容
  target 或删除 registry 历史；
- 产品级范围：与 `platform_thread`、`platform_fs` 协同完成三因子迁移，最终
  `hyle::platform` façade 聚合三者；MappedFile/SharedMemory 的 storage artifact 只在 Mem factor
  round 依法 `SCOPED/AUDITED` 后迁走，本轮不得留下双 owner；
- 非目标：device inventory/kernel dispatch、scheduler placement/stealing、async/direct I/O、lexical
  path、JIT/process/sandbox/plugin ABI、Native FFI/C API、稳定 C++ ABI 与 Formal；
- 预期直接依赖：host factor 只 `PUBLIC core,error`；OS API/compiler builtin 是机制合同。预期直接
  consumer 是 `platform_thread`、`platform_fs`、Mem VM/pool、SIMD、Container、Sched 与 IO；
- tier / normal performance：`P1`。primary 是推理初始化中的 process-available topology + host-memory
  snapshot，对同语义 direct-OS/locked peer route；holdout 是已缓存 CPU/ISA facts 的重复读取。5%
  non-inferiority；不得把多 syscall 的完整 snapshot 与单个 cached scalar 作苹果橘子对比；
- 常态成本：稳定 facts 一次探测后 O(1)，动态 availability/topology 为 O(available CPUs + reported
  relations)，buffer-fill core 不分配；host-memory snapshot 为 O(1) 有界 OS/file queries；secure entropy
  为 O(bytes/native chunk)，无 TLS PRNG state；raw VM/NUMA 保持既有 complexity；
- 总预算：三个 Platform factor 协同调查与实现最多六小时，性能取证合计九十分钟；每个 family 最多
  一次机制诊断和两次产品修正。达到预算仍有适用 gate 未裁决即 `BLOCKED`，不扩建公共仪器；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 primary 仅 W1-WIN +
  `closure-perf-v1`，Formal `PENDING/formal-release-v1`；
- strong peers：Folly `eeab4246827eae9a11c0fa22529bff1a258aa83a` 的 portability/Random/
  MemoryMapping，oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668` 的 info/NUMA；CPU/VM/
  entropy 还受各 OS 官方 ABI 约束；
- correctness gates：normal、empty/zero、capacity/overflow、unsupported/permission/OOM/system failure、
  sparse IDs、snapshot churn、move/destroy、concurrency/cache init、property/stress、ASan/UBSan/TSan、
  可用的真 ISA/NUMA；不适用项必须逐项说明；
- delivery gates：三个 factor 精确构建和 header ownership；Platform product source embedding、relocated
  static/shared；SIMD/Mem/Container/Sched/IO direct consumers；三平台；旧 C/Native FFI 负向扫描；
- 停止条件：双向审计冻结有限 `FIX`，固定 API/guide 后实现；新 process/JIT/device/async journey 只
  路由。正常 P1 无资格则停止为 `BLOCKED`，不依据微观点值改变类型或常态 metadata。

`SCOPED` 只冻结问题、预算、机器、性能等级和交付边界；完整 capability/public-surface trace、所有
`MUST` 的 owner 与有限 `FIX` 集完成前，仍不授权产品实现。
