# G02 NJ-01 三 port 页图节点阶段

三个 owner 沿 `b2b282eca` AUDITED。本阶段只签 graph_batch descriptor/factory 的普通
GraphBuilder source 消费：三 inputs，active5/capacity11 的 packed 输出，logical95>
physical64 的重复/乱序页，foreign 同 capacity schema 拒绝、空输出与伪造 root 拒绝。
原 materialized Graph 的三个 shape/recipe-owner/冷逐分配 OOM 负空间也 PASS，共五项。

原单 CPU source 树、四 jobs；两次配置20.375/19.187秒、构建22.203/22.219秒、最终
run0.047秒，累计84.031秒。首 fixture 错用 Operation(descriptor) 编译 FAIL；修为原
Operation::custom，未因此修改产品算法/API。失败日志保留；归档中明确标记的首版
fixture 是按唯一修正反向重建，不能冒充此前保存的原输入快照。

receipt/raw 保存命令、退出码/elapsed、最终 canonical SHA、consumer、两次日志与
selector READY 的 selected/skipped reasons。尚未签 SDK/逐头/宿主风险或 actual CPU+
Apple/Vulkan 数学；common schema 不提供 native qualification 或 arena read permission。
NJ02–06 的 owned layout/原 fragments/精确 frame/实际退休仍待实现，三 owner AUDITED。
