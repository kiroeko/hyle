# R6 候选独立阶段复核

产品 `8e45b104`，测量快照 `3894b72e`。本遍在实现/API 冻结之后进行，限 R6 V1–V7；
不重新审计底层 CLOSED 模块。此处记录源码复核结果，运行门仍各自引用实际 receipts。

后续 Apple 测量快照 `a41d43d5` 只修正 benchmark 冷 loader 路径：两侧使用同一公开
ContextOptions，不改变产品、热路径、数值或公共仪器。Windows 原快照的证据保持独立。

| 问题 | 核对与裁决 |
|---|---|
| 普通入口与责任 | Direct prepare/Invocation 无 Graph 依赖；Graph typed DAG 只冷降为原生 Program。Recorder 是明确固定 caller buffers 的专家组合；不冒充 owning output。 |
| 成熟实现/规则 | 原样 ggml shader pin 与数值合同可追溯；subgroup/limits/型号在冷期使用，热提交无规则搜索、计时或逐 kernel Provider 裁决。FP16/cooperative 未混入 strict F32。 |
| F32 资源账单 | 实际 shader BK32；共享内存按 32 计算。aligned load 只准入完整 tile 和足够 offset alignment。128 tile 的 Air 失败及两平台修复验收均保留。 |
| 常量与 scratch | 完整 lowering/packing 去重之后才释放没有最终 dispatch 引用的 dense owner；仍被引用的原常量保活。没有引用的 value 不进入 storage blocks。liveness 必须 last < first 才复用，published output 延长到程序末尾。 |
| 提交/失败/取消 | 成功 QueueSubmit 才链接 pending owner；submit OOM 不等待未接纳 fence。观察失败保留资源，原生 fence/idle 证实终态后才交付原错误。取消不重置 pending command/fence。 |
| 并发与回收 | 对象局部 queue mutex；阻塞 fence wait 放锁，observers 阻止复用；外部 owner 的析构在 queue 锁外。每槽独立 command/descriptor/fence，ResultSlot 和 Tensor/native token 都阻止输出覆盖。 |
| native interop/transfer | borrow_ready 的设备/queue-family/可见性是显式调用者合同，检查 range/alignment/overflow 并保活 token；upload/download 保留 staging，按真实 memory flags flush/invalidate。没有隐式 CPU 模型 fallback。 |
| 比较公平性 | peer 从独立 pin 编译原样 SPIR-V，数学 recipe 不调用 typed lower/Graph compile；两侧共用的生命周期 adapter 明确且完整计费。裸 submit/fence 缺少取消/quarantine/lease，单列 NOT_COMPARABLE，不排名。 |
| 制品与文档 | ON/OFF、public target、源码/静态/动态与 relocation 使用原有交付机制；最终 artifact 验收独立执行。发现总入口的“Apple 尚未交付”旧文案及未限定的 Queue 描述，已在 `20a029ad` 修正并链接路线指南，未改 API。 |

本遍未发现需退回 AUDITED 的新产品缺陷。适用的最终 ASan、Air P1、原生边界诊断、最终三交付和
中央 selector 必须各自完成，不能由源码 review 替代。既有 UBSan/TSan 的具体适用性随最终记录说明；
未知 AMD/Intel、其它 dtype、实际非一致 memory type 和故意 device-loss/hang 不伪造实机资格。
