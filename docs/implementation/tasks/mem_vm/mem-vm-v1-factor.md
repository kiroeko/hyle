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
