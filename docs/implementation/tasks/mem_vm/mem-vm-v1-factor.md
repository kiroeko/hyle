# Mem VM v1：mapping、residency 与 guarded storage factor

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Registration（`QUEUED`）

`mem_vm` 是全局能力因子化已接受的内部 factor ID，预期唯一拥有 anonymous/file/shared mapping、
residency/huge-page 与 guarded-stack storage。普通用户不直接选择该名字，而通过 `hyle::mem` product
facade 使用同一 `hyle::mem` namespace。当前仅登记，尚无 buildability、API 或性能结论。

## Closure card（`SCOPED`）

- component / owner / round / kind：`mem_vm` / Mem / `mem-vm-v1-factor` / `NEW`；baseline
  `5fe4f72d6dff8b307a1271a1f51e7e5b6fc0fed2`；
- demand：`PLAT3-X01` 与全局 `B-F12`；consumer/范围/预算/machines/peers/gates 与 Mem v3 协同卡一致；
- 本 factor 唯一拥有 `VirtualMapping/FileMapping/SharedMapping`、mapping residency、HugeMapping 与
  FiberStack public storage semantics；Platform 只保留 raw VM/NUMA/FileHandle/OS mechanism；
- 预期 public deps：`core,error,mem,platform,platform_fs,util_core`；`util_core` 只承载既有 FiberStackPool
  的 SpinLock；不允许依赖 platform product façade、SIMD、
  Container、Sched、IO、Compute 或 Graph；
- 常态成本：owner inline native state，无 wrapper pimpl allocation；view O(1)；mapping setup O(1) native
  calls，populate/lock/advice 与覆盖页/range 成正比；
- API、完整 capability trace 与有限 FIX 集必须在 `AUDITED` 后才可实现。

## Audit verdict（`AUDITED`）

public/source reverse trace 证明当前 Platform `MappedFile/SharedMemoryRegion` 只被 Mem owner 与 Platform
自身 tests/sample 使用；普通下游没有必须保留的独立 Platform storage journey。`VmRegion`、NUMA、
FileHandle 则是可复用 raw mechanisms，继续由 Platform factors 拥有。最终边界与有限 FIX 直接采用
Mem v3 的 `MEM3-VM01..06`：

- `mem_vm` public headers 为 `mapping.h`、`hugepage.h`、`fiber_stack.h`，compiled source 直接实现
  File/Shared mapping native state；
- public deps 冻结为 `core,error,mem,platform,platform_fs,util_core`；普通用户仍只链接
  `hyle::mem` facade；
- `FileMapping::populate` 是唯一新增 public operation；fragment 用多个 mapping owner，不建洞状 owner；
- API freeze 为 `FROZEN`（2026-09-03）：path/FileHandle mapping、advise/populate/lock、named shared、
  anonymous NUMA 与 guarded-stack journeys、公开头自含及 Windows/WSL 精确 factor 均已通过；
- 自身无第三方 product dependency、无 C/Native FFI、无稳定 C++ ABI；source/static/shared 和三平台
  交付在 candidate 前都不能写 `PASS`。

该结论授权实现，不授权修改其他 Mem-v2 family 或公共性能仪器。

## 候选前实现检查点

- VM01--VM04 与 VM06 已落地：三个 public headers、直接 native owners、同步 populate、唯一 ownership、
  精确 factor 与 Mem façade/documentation；未触及 allocator/pool/resource/buffer/planner；
- W1-WIN 精确根下 Mem `223/223`、mem_vm `60/60`；GCC 14 `-Werror` 同为 `223/223` 与 `60/60`，公开头
  自含通过；
- 在该检查点 VM05 的正式 P1、M1-MAC、sanitizer、三种产品交付、direct consumers 与红队尚未
  裁决；后续进度以紧接的 Candidate 节为准。

## Candidate（`CANDIDATE`）

product commit/tree 是 `e9b9a43d4782c60d84badc02b1a41fbe21207a00` /
`c6429316279aaac1e0f5d7a7a71872a60cf967b6`。VM01--VM06、公开 owner/layout 与 mapping/residency normal
workloads 已冻结；W1-WIN、GCC 14 `-Werror`、ASan/UBSan、公开头自含、Windows exact factor 及 Mem
source/relocated shared/static journeys 已通过。M1-MAC、direct consumers、正式 P1 与最终红队仍待签署。

首个共同 candidate 因 Platform host 的 Apple `getentropy` 声明缺失退回 `AUDITED`；Mem VM 本身没有
授权产品改动，修复依赖后重放同一验证。

## 第二候选（`CANDIDATE`）

共同 product commit/tree 是 `b1a526d6fdb05b6bb42c4cb041788959e74511de` /
`f808df6fcbd0b5549777e0c6939240a3feab9512`。M1-MAC 的 `mem_vm` 公开头与实现测试已随五组件共同重放
通过；VM01--VM06、owner、layout 与 mapping/residency workloads 未改变。sanitizer、direct consumers、
正式 P1 与最终红队仍待裁决。

## 最终验证与六问（`VERIFIED`）

本 factor 与 Mem v3 共用 immutable product candidate
`b1a526d6fdb05b6bb42c4cb041788959e74511de` / tree
`f808df6fcbd0b5549777e0c6939240a3feab9512`，完整验证矩阵、红队与前态 snapshot 见
[Mem v3 主记录](../mem/mem-v3-vm-factorization.md)。W1-WIN 实现/公开头、W1-WSL Release/ASan/UBSan/
TSan、M1-MAC Release/ASan、Mem source/static/shared、exact factor、Sched/IO/Container direct slices 与旧
FFI/owner 负向扫描全部 `PASS`。

[Mem VM P1 报告](../../evidence/performance/benchmarks-mem-vm.md)中完整 file mapping 4/4 strata 的
median ratio 为 0.977834--1.005194、最低 confidence lower 0.963487；progressive residency 8/8 strata
的 median ratio 为 0.991597--1.001353、最低 lower 0.965027。两族均为 `PARITY` 且 5% NI `PASS`，只
支持冻结 loader 旅程，不外推 partial unmap、异步 I/O、多 NUMA socket 带宽或端到端 token latency。

Owner 六问：自身无第三方 product implementation，base Mem 的 mimalloc 不由本 factor 重导出；VM
storage 超集与自然 fragment route 完整；inline native owner 与 O(1)/O(pages) 结构成本及两项同语义 P1
通过；无公开 C/Native FFI；factor 精确构建但用户只选 `hyle::mem`；guide 已覆盖 truncate、residency
quota、共享名、NUMA fallback、guard page 与外部同步陷阱。VM01--VM06 全部 `COMPLETE`，七项 aggregate
gate 全为 `PASS`，无 blocker；当前进入 `VERIFIED`。
