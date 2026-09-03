# Mem VM factor migration：登记阶段

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 目标与边界

Platform v3 的双向审计证明，现有公开 `platform::MappedFile` / `SharedMemoryRegion` 与
`mem::FileMapping` / `SharedMapping` 形成重复 storage owner；同时单体 Platform target 无法在不产生
错误依赖环的情况下拆出 filesystem。全局能力因子化已经把 mapping/residency/guarded stack 的唯一
产品 owner 裁决为 `mem_vm`，所以这是 Mem 既有 dependency/delivery shape 改变的明确 reopen trigger。

本登记任务只增加 `mem_vm` component ID，并把 Mem 从 `CLOSED` 改为 `REOPENED`。它不建立 target、
不移动源文件、不改变 `hyle::mem` product roots，也不把 QUEUED 写成可构建。普通用户最终仍只选择
`hyle::mem`；`mem_vm` 是内部独立开发和精确依赖 factor。

修改前 registry 保存为
`C:\public\project-hyle\_validation\mem-vm-v1\module-status-before-registration.json`，SHA-256
`481A4F2D851EB06AEB12DB4089319FA4F1D7272FB236BCD75D1CB0F9C2CDF29A`。

## 结果合同

`check-module-closure.py --previous` 必须证明 `mem` 只走 `CLOSED -> REOPENED`，新 ID 只以
`QUEUED` 出现，既有 terminal receipt 不被回写。后续完成 SCOPED/AUDITED 前不得修改 Mem/Platform
storage 产品代码；迁移必须删除重复公开 owner，不能留下兼容 alias。
