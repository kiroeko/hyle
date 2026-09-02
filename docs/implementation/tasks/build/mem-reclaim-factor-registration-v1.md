# Mem reclamation factor：登记阶段

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 目标与边界

全库能力因子化把 reader-safe reclamation 从 intrusive membership、并发容器和 Obs 的私有实现中抽出，
归入 Mem 产品内部的 `mem_reclaim` 质因数。Build/package v7 要求新 factor 在写产品代码前先同时进入中央
component manifest 与 `module-status.json`，并以 `QUEUED` 起步。

本登记任务只建立这个合法起点：不移动头或源码，不改变 `hyle::mem` 产品门面，不修改旧 `mem` factor，
也不赋予新 factor 可构建、可交付、功能完整或性能通过的结论。修改前 registry 的明确快照保存在
`C:\hyle-scratch\mem-reclaim-pre-registration.json`，SHA-256 为
`5C1D3C7A6AB959559BA65BF7623A497581DD721606C1731D91ACED7C2168FF09`。

## 登记合同

- component ID：`mem_reclaim`；语义 owner 是 Mem，初始直接依赖只允许 `core`；
- registry 初始状态：`QUEUED`；后续必须另建 closure card 进入 `SCOPED`，完成能力/API 审计后才能实现；
- `hyle::mem` 仍只选择当前 `mem` factor，普通 source/static/shared 用户看不到未完成 factor；
- 当前 `mem` 的 `AUDITED` 状态、公开 API、实现、性能证据与消费者均不由本登记继承或改写；
- 后续迁移只能让并发容器和 Obs 消费唯一 reclamation owner，不把 intrusive membership、容器策略或日志
  策略塞进本 factor。

## 验证与裁决

`check-module-closure.py --previous C:\hyle-scratch\mem-reclaim-pre-registration.json` 必须证明新 ID 只以
`QUEUED` 出现、既有 component 没有删除、重命名、重排或跳级；普通 closure/docs 检查和 diff hygiene
同时通过。满足这些条件后本登记单元关闭，`mem_reclaim` 本身仍保持 `QUEUED`。
