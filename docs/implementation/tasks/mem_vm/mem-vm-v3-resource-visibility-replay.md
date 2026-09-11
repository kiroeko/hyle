# Mem-vm v3：公开资源基类交付回放

## Closure card

- component / owner / kind：`mem_vm` / Mem-vm / `REOPEN`；round=`mem-vm-v3-resource-visibility-replay`；
- baseline：`5eaf7af55f9844e3175edbd958507eb60fb1208c`；前一 terminal record 是
  [`mem-vm-v2-inference-mapping`](mem-vm-v2-inference-mapping.md)；
- trigger：MacBook Air default-hidden shared build 在链接本模块的 `BorrowedMappedResource` 时无法取得下层
  `MemoryResource` RTTI；这是可复现的跨 factor 交付失败；
- consumer/纵切：`mapped_resource()` 的只读/可写 provider 经 Mem 公共基类进入 Mapping，随后由 Sched
  source/static/shared consumer 链接；
- scope：不改 Mem-vm 产品源码；审计下层类型可见性修复后，重放本模块三交付和直接 Sched consumer；
- non-goals：新增 VM 能力、改 mapping 生命周期/API/算法、绕过 RTTI、合并 factor 或运行 P1；
- 依赖：现有 Core/Error/Mem/Platform/Platform-fs/Util-core 不变；
- 性能：`P0`，本模块无运行时代码变化；
- 验证/stop：三平台 link、Mem-vm correctness、selected Mem/Sched 交付和 export scan；若仍需改 Mem-vm
  源码则停止并重新审计有限改动。

## AUDITED 裁决

源码检查确认 `BorrowedMappedResource` 的继承、对象生命周期和 `mapped_resource()` 路线正确；链接失败
完全由下层 public type visibility 解释，本模块没有第二个 FIX。进入 `AUDITED`，只继承 Mem v8 候选并
重放验证；不允许复制基类、关闭 RTTI 或修改 Mem-vm 产品源码来掩盖错误边界。
