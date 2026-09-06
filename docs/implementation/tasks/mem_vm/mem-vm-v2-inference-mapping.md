# Mem VM v2：推理 mapping 与 OS storage 合同重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前只登记 `REOPENED` 新事实。进入 `SCOPED/AUDITED` 并冻结 API、成本和有限 `FIX` 集以前，
> 本页不授权修改 Mem-VM 产品源码。

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
