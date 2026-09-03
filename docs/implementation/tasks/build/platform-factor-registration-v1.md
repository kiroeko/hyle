# Platform factor migration：登记阶段

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 目标与边界

已接受的全库因子化把旧 Platform 目录分成 host facts/mechanisms、native thread/wait 与同步 filesystem
三个独立成本和依赖闭环。Build/package v7 要求新增 factor 先取得中央 ID 和 `QUEUED` 状态，再由产品
round 完成 `SCOPED/AUDITED` 后迁移源码、头、依赖和 product roots。

本登记任务保留既有 `platform` ID 作为 host factor，新增 `platform_thread` 与 `platform_fs`。这与 Time
保留 `time`、新增 `timer_queue` 的迁移方式一致：不制造兼容 target，也不要求 closure registry 支持
删除或重命名已存在 component。面向用户的产品名始终是 `hyle::platform`；内部 factor 名不进入
`HYLE_COMPONENTS` 或 `find_package(... COMPONENTS ...)`。

修改前 registry 保存为
`C:\public\project-hyle\_validation\platform-v3\module-status-before-registration.json`，SHA-256
`E5D68A0B3A11B385843E39D6147D0BBA7482A78CE4FBECFF6A29E9458EC3E759`。本提交只在中央 manifest
登记两个无依赖、未被 façade 选择的 ID，在 registry 同序增加 `QUEUED` 记录，并把既有 Platform 按
已接受的新事实改为 `REOPENED`；它不移动产品源码、改变当前 `hyle::platform` façade 或赋予新 factor
虚假的可构建和交付结论。

## 结果合同

`check-module-closure.py --previous` 必须证明新 ID 只以 `QUEUED` 出现、Platform 只走
`CLOSED -> REOPENED`，且没有删除、重命名或重排既有 component。后续只有三个 factor 都完成
`SCOPED/AUDITED`，才能迁移公开头、compiled artifact、真实 direct edge 和 Platform product roots。
