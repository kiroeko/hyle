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

## Candidate

- immutable dependency-closure candidate：`ad5bff895125fb0d95281b36559010435c3d7892`；
- 本模块产品源码、公开 API、layout、mapping 生命周期、算法和常态成本没有变化；
- Windows Mem-vm correctness 通过；Windows、Linux GCC 14、MacBook Air arm64 的 Sched
  source/static/shared relocated consumer 均成功跨过本模块；
- default-hidden Mac shared 构建已证明 `BorrowedMappedResource` 可以沿公开 Mem 基类完成 RTTI/vtable
  链接；
- candidate 后只允许补 terminal 证据、文档与状态。

## 验证

| Gate | 适用风险与命令/证据 | Verdict |
|---|---|---|
| scope | 本模块产品源码与公开头零变化，只回放下层 Mem 修复 | PASS |
| architecture | `BorrowedMappedResource` 继续作为私有 adapter，公开抽象仍由 Mem 单点拥有 | PASS |
| correctness | Windows Mem-vm correctness 通过；既有 mapping 生命周期与故障测试未受影响 | PASS |
| delivery | Windows、Linux GCC 14、MacBook Air arm64 的 Sched source/static/shared 闭包均跨过 Mem-vm | PASS |
| usability_docs | Mapping、FileMapping、SharedMapping 与 LargeMapping 的用户选择和 API 均未改变 | PASS |
| red_team | 没有复制基类、关闭 RTTI、增加 trampoline 或修改本模块可见性来掩盖下层缺陷 | PASS |
| engineering_performance | P0：本模块无代码变化；v2 的同合同 P1 证据继续有效 | PASS |

## 红队与裁决

- 反证：default-hidden Mac shared 链接是最强跨 factor 检查；下层修复后原样 Mem-vm 源码即可通过，证明
  owner 与依赖方向正确；
- 未决限制：平台 VM 能力差异和共享 C++ 工具链要求沿用 v2 已披露合同；
- reopen triggers：新的真实平台/consumer 暴露 mapping 正确性、生命周期、交付或 material 性能问题；
- LOC（candidate）：全库 560,631；自实现不含测试 100,233；Mem-vm compiled 实现 1,318；
- 最终状态与理由：`CLOSED`。本模块没有独立产品缺陷，所依赖的公开类型身份已由 owner 修复并经三平台
  完整交付回放确认。
