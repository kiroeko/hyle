# 当前施工状态

> 更新时间：2026-08-26。这里是模块重审状态的唯一入口；历史任务中的绿色勾或“已收口”不再
> 代表新标准下的结论。

HYLE 正以真实推理消费者为首要需求锚，从底层开始重新建立模块证书。Core v2 已完成首个严格
签署；本轮为修复其能力归属与直接依赖，把侵入链表迁为独立轻量组件，但这不等于提前签署
container 能力。

| 范围 | 状态 | 说明 |
|---|---|---|
| core | `SIGNED` | `b241a22`：header-only compiler/invariant 与 cache primitive；版本归 build/package，Core FFI 为 N/A；三平台、codegen/symbol、sanitizer、三种交付与独立红队证据已由 `b923105` 绑定 |
| container_intrusive | `CANDIDATE` | auto/safe 双生命周期、通用 O(1) relink 边界、time 消费与 sched 特化裁决已冻结；Windows/WSL/macOS correctness、sanitizer、component/codegen dirty 预检已通过。旧 n=31 性能轮因仪器继续收紧而 superseded；等待当前 harness 完整 qualification、clean candidate 三平台性能/evidence 与最终红队后签署 |
| error / platform / mem / time / obs / util | `NOT_REVIEWED_V2` | 有多轮旧实现与证据，可作输入，不能继承旧证书 |
| build/package | `NOT_REVIEWED_V2` | 已有组件化与三种 C++ 交付形状，需随每个模块持续做闭包回归 |
| container / simd / math / sched / io | `NOT_REVIEWED_V2` | 已实现程度不同，等前置模块稳定后逐一审查 |
| compute / graph | `INCOMPLETE` | 半成品；依赖与产品形状仍可能重做，不做冻结式优化 |
| Native FFI | `N/A` | 没有已命名外语消费者；旧 schema/bridge/fixture 是待逐模块删除的技术债，不承诺兼容 |

状态流为 `OPEN -> AUDITED -> CANDIDATE -> EVIDENCE_COMPLETE -> SIGNED`。发现新的真实消费者缺口、
正确性问题、依赖契约变化或同合同的实质性能落后时进入 `REOPENED`，随后重新走门。

当前 Core 工作项及迁移记录见 [core rebuild](tasks/core/core-rebuild.md)；当前归属以
[Core 架构](../architecture/core-design.md) 和本页为准。审查与停止条件见
[模块审查方法](../developer/review/module-closure-methodology.md)。
