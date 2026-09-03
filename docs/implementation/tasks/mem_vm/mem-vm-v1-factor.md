# Mem VM v1：mapping、residency 与 guarded storage factor

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Registration（`QUEUED`）

`mem_vm` 是全局能力因子化已接受的内部 factor ID，预期唯一拥有 anonymous/file/shared mapping、
residency/huge-page 与 guarded-stack storage。普通用户不直接选择该名字，而通过 `hyle::mem` product
facade 使用同一 `hyle::mem` namespace。当前仅登记，尚无 buildability、API 或性能结论。
