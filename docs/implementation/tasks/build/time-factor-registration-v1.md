# Time factor migration：登记阶段

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 目标与边界

Stage C 已把旧 Time 目录分解为 host-clock `time` 与 timer-node/data-structure `timer_queue`，而 Build/package
v7 要求新增 factor 先取得中央 ID 和 `QUEUED` 状态，再由各自 closure round 冻结语义、依赖、源码与
产品 roots。本登记任务只建立这个合法起点，不移动头/源码、不改变 `hyle::time` façade，也不赋予新
factor 虚假的可构建或交付结论。

修改前 registry 保存为 `C:\hyle-scratch\time-v3-pre-registration.json`。中央 manifest 在 `time` 后登记
无依赖的 `timer_queue` ID；registry 同序增加 `QUEUED` 记录，并把既有 Time 依其公开依赖/交付变化和
缺失 manual-clock 新事实定向改为 `REOPENED`。既有 component 的顺序与状态历史不被重写。

## 结果

`check-module-closure.py --previous` 必须同时证明：新 ID 只以 `QUEUED` 出现，Time 只走
`CLOSED -> REOPENED`，没有删除、重命名或重排。当前 `hyle::time` 仍选择旧 `time` factor，因此登记
提交不改变源码、静态或共享消费者。后续只有 Time v3 与 timer_queue 各自完成 `SCOPED/AUDITED` 后，
才能迁移 headers、target edges 与 product façade roots。
