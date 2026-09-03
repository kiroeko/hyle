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
- 预期 public deps：`core,error,mem,platform,platform_fs`；不允许依赖 platform product façade、SIMD、
  Container、Sched、IO、Compute 或 Graph；
- 常态成本：owner inline native state，无 wrapper pimpl allocation；view O(1)；mapping setup O(1) native
  calls，populate/lock/advice 与覆盖页/range 成正比；
- API、完整 capability trace 与有限 FIX 集必须在 `AUDITED` 后才可实现。
