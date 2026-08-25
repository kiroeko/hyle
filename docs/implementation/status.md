# 当前施工状态

> 更新时间：2026-08-26。这里是模块重审状态的唯一入口；历史任务中的绿色勾或“已收口”不再
> 代表新标准下的结论。

HYLE 正以真实推理消费者为首要需求锚，从底层开始重新建立模块证书。当前只审 core；除非为
修复直接依赖或验证消费者，不同时重构后续模块。

| 范围 | 状态 | 说明 |
|---|---|---|
| core | `OPEN` | 新方法的试点；功能归属、C++ 形状、真实消费者和性能仪器均需重新签署 |
| error / platform / mem / time / obs / util | `NOT_REVIEWED_V2` | 有多轮旧实现与证据，可作输入，不能继承旧证书 |
| build/package | `NOT_REVIEWED_V2` | 已有组件化与三种 C++ 交付形状，需随每个模块持续做闭包回归 |
| container / simd / math / sched / io | `NOT_REVIEWED_V2` | 已实现程度不同，等前置模块稳定后逐一审查 |
| compute / graph | `INCOMPLETE` | 半成品；依赖与产品形状仍可能重做，不做冻结式优化 |
| Native FFI | `N/A` | 没有已命名外语消费者；旧 schema/bridge/fixture 是待逐模块删除的技术债，不承诺兼容 |

状态流为 `OPEN -> AUDITED -> CANDIDATE -> EVIDENCE_COMPLETE -> SIGNED`。发现新的真实消费者缺口、
正确性问题、依赖契约变化或同合同的实质性能落后时进入 `REOPENED`，随后重新走门。

当前 Core 工作项见 [core rebuild](tasks/core/core-rebuild.md)。审查与停止条件见
[模块审查方法](../developer/review/module-closure-methodology.md)。
