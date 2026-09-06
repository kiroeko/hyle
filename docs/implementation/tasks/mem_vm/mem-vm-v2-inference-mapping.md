# Mem VM v2：推理 mapping 与 OS storage 合同重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> Closure card 已冻结，当前状态为 `SCOPED`。进入 `AUDITED` 并冻结 API、成本和有限 `FIX` 集以前，
> 本页仍不授权修改 Mem-VM 产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem_vm` / Mem mapping and residency semantics /
  `mem-vm-v2-inference-mapping` / `REOPEN`；baseline
  `5aaeaa770be72cd7a58465659b2abc7d98a2d89b`，tree
  `2035afb59e4a658526bc3f39e458b305a36f1a1d`；
- trigger IDs：`MEMVM-INFERENCE-SPARSE-MAP-01`、`MEMVM-PLATFORM-V8-REPLAY-02`；
- 当前锁定 llama.cpp `74a7c897f049c17e7080423aa2111776eff6ebbf` 的 model loader 直接使用
  lazy file ranges，并在 POSIX 上通过 `llama_mmap::unmap_fragment(first,last)` 释放已经不再需要的
  weight pages；同一 owner 还按 lazy range 分别施加 sequential/random advice。`mem-vm-v1` 明确把
  holey/segmented mapping 排除并要求用户创建多个 owner，不能自然表达这条已命名北极星旅程，命中
  “正式准入的共同 storage journey 不能由当前 owner 表达”的 reopen trigger；
- Mem-VM v1 候选后，直接依赖 Platform 的 NUMA implementation 与 VM range error contract 已进入
  `platform-v8` 候选 `c39df6f94c636ab8fa94cb700dd263763a42d680`。现有 `VirtualMapping`、
  `HugeMapping` 与 guarded stack 路线需要重新验证唯一 decision authority、错误传播、平台差异和
  source/static/shared 导入；
- 初步 red-team 输入还包括：`VirtualMapping::advise` 目前依赖两个 enum 的数值偶然一致；
  `VirtualMapping`/`HugeMapping` 与统一 Buffer/View 组合是否自然；shared mapping 的 read-only、size、
  alignment 与 name lifecycle 是否形成半项能力；file mapping 的 offset、prefetch/populate/lock/flush/
  partial-release 在 Windows、Linux、macOS 上是否诚实使用系统接口。这些只是待裁决 finding，未在
  `AUDITED` 前预判为全部要实现；
- strong inputs 固定为当前 llama.cpp、standalone ggml、Folly
  `65749da412d92fe27d87ae45a8ef670720712858` 的 `MemoryMapping` 与 Boost Interprocess 当前本地
  checkout。peer 用于发现稳定任务与语义档，不按 symbol 数量复制；
- `CLOSED -> REOPENED` 前 registry 快照位于 canonical checkout 外：
  `C:\public\project-hyle\_validation\registry-before-mem-v6-mem-vm-v2-reopened-5aaeaa77.json`，
  SHA-256 `036760D948DFF33F9600A7B926F9F7E7CCB3304A328842EB2C7706E9383E9D6B`。

## 初步停止边界

- 本轮只拥有 anonymous/file/shared mapping、range release/advice/residency、large/huge-page placement
  与 guarded stack storage；base allocator/pool/planner 归协同 `mem-v6`，reclamation 无新事实不重开；
- async/direct IO、backend/device mapping、Tensor/KV/model policy、IPC synchronization protocol、
  Native FFI/C API、稳定 C++ ABI、公共性能仪器和 Compute/Graph 均不进入；
- 非便携 OS 能力可以形成 capability query、明确 `Unimplemented` 或具名 platform-specific route，
  不能静默假装成功，也不能为了表面统一复制 Platform 已拥有的机制；
- API freeze 后只允许内部 OS adapter、owner metadata/algorithm 与 build wiring 修正。性能只比较相同
  mapping/residency/release 语义和相同 charged pages/syscalls；苹果对橘子只保留 natural-route 或
  diagnostic 结论。

## Closure card（`SCOPED`）

- component / owner / round / kind：`mem_vm` / Mem mapping and residency semantics /
  `mem-vm-v2-inference-mapping` / `REOPEN`；baseline
  `5aaeaa770be72cd7a58465659b2abc7d98a2d89b`；
- demands：`MEMVM-INFERENCE-SPARSE-MAP-01` 与 `MEMVM-PLATFORM-V8-REPLAY-02`。主 journey 是模型
  loader 对同一文件建立长期只读 mapping，按 lazy tensor ranges 提示访问、逐步锁定可选前缀，并在
  POSIX 上释放不再需要的页区间，同时 owner 仍正确清理剩余 fragments；
- 范围：anonymous reserve/commit/decommit/protect/advice/lock；file RO/RW/COW mapping、offset/
  length、range advice/prefetch/populate/residency/flush 与 partial release；named shared-memory
  create/open/access/size/name/unlink/flush；large/huge-page hint 与 NUMA placement；guarded fiber stack
  及 bounded cache；与 Mem Buffer/View 的安全组合；
- 非目标：base allocator/pool/planner、reclamation、async/direct IO、device/backend mapping、Tensor/KV/
  model policy、IPC synchronization protocol、C/Native FFI、稳定 C++ ABI、公共性能仪器和 Compute/Graph；
- 直接依赖：`core`、`error`、`mem`、`platform`、`platform_fs`、`util_core`；直接消费者为 Sched 的
  fiber stack、模型加载/IO/Compute 的 mapping journey，以及 Mem product facade；
- 待复用设施：Platform `VmRegion`、NUMA/host facts、`FileHandle` 与 OS error translation，Mem
  Buffer/View，Util-core `SpinLock`；direct OS file/shared mapping 只在 Platform 没有同合同 owner 时保留，
  但错误、range、page/alignment 和 decision authority 不能复制或依赖 enum 数值偶合；
- tier：`P1`。主要 workload 是 `NORTH_STAR_JOURNEY` 的 model-loader RO map + deterministic lazy
  ranges + range advice + POSIX partial release + remaining-owner teardown；在 `W1-WSL` 原生 ext4 上与
  当前 llama.cpp 同语义机制 adapter 作 `SAME_CONTRACT`，计费相同 mmap/advice/munmap/剩余 unmap/close
  及实际 mapped pages/syscalls。W1-WIN 的 whole-map lifecycle/progressive residency 仅作一个冻结
  holdout，不与 POSIX partial-unmap 数字池化；
- 5% material/non-inferiority margin；无相同 OS 语义的平台只签 correctness/fallback，禁止跨 OS
  排名或以 Windows no-op partial release 冒充 POSIX 同合同；
- 总调查预算：6 小时；其中性能最多 90 分钟。最多两轮产品修正；性能异常只允许一次反序复现、
  一次 syscall/codegen/profile 机制检查、一个产品修改和一次重跑；
- correctness machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；主要 P1 machine：`W1-WSL`；instrument：
  `closure-perf-v1`，P0 辅助为 `closure-correctness-v1`；WSL 只签明确的 Linux guest engineering
  contract，不冒充 native-Linux 或独立硬件 Formal；
- strong inputs：llama.cpp `74a7c897f049c17e7080423aa2111776eff6ebbf` 的 `llama_mmap`/
  `llama_mlock`，Folly `65749da412d92fe27d87ae45a8ef670720712858` 的 `MemoryMapping`；shared
  memory family 以当前本地 Boost.Interprocess `cb94f33fcb13e90922905c2d8105508e71fa6db4` 取代
  Folly，仍保证每 family 最多两个；
- API 语义档：reserve-only 与 committed anonymous owner 分明；file mapping access/lifetime 显式；
  partial release 后不得继续声称完整连续 `bytes()/view()`；shared access 与 payload size/alignment 明确；
  THP hint、explicit large page、NUMA bind/interleave 不混为一种保证；guarded stack cache 只缓存完整 owner；
- 适用门：全部 public surface reverse trace；正常、空/零、page/range/offset/overflow、OS failure/
  invalid、move/destroy/partial teardown、共享 name 生命周期、并发 cache、property/stress、ASan/UBSan/
  TSan；Windows/WSL/macOS 系统交互，source/static/shared delivery，直接消费者、FFI 清洁、文档/易用性、
  内部复用、红队和 P1 工程性能；真实 ISA/backend 仅在确实不适用后记 `NOT_APPLICABLE`；
- 输出：能力 family map、七问、冻结 API、有限 `FIX` ledger、candidate、验证 manifest、终局 receipt
  和同候选 LOC；
- 停止：所有 `MUST` 为 `COMPLETE/ROUTED`、适用门通过且无 blocker 即 `CLOSED`；平台缺失而无法
  诚实实现的路线必须明确 capability/`Unimplemented` 并裁决，固定预算内 P1 或 correctness 未决则
  `BLOCKED`，不得扩建仪器或不断换 benchmark。
