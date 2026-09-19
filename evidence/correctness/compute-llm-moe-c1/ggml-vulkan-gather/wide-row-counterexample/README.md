# Y26-F2：真实 BF16 宽行反例

产品 `807bc22bb1b50cc6eaf9bc03d7f8a72be417a597` 的实际2B embedding子表首次source消费失败。
完整数值/命令记录见 [result.json](result.json)，原始日志与诊断consumer见 [raw.zip](raw.zip)。
归档只有输入身份和oracle来源，不含模型权重字节。

Direct读取一个2048宽BF16行时，元素1024起漏写；首次差异为`0000`对`bc11`。
锁定ggml的I32 GetRows shader每组512 lane，pipeline却按1024计算工作组数。
独立普通入口的宽行位级回归进一步复现该问题，包括checked/unchecked与六种原始存储类型。
小尺寸既有PASS不能覆盖这个失败域。

归档也保留此前外部consumer遗漏既有MSVC C4324对齐padding豁免的编译失败；该编译失败与
本产品缺陷分别记录。F2修复范围及剩余预算见
[当前owner卡](../../../../../docs/implementation/tasks/compute/llm-moe-c1-ggml-vulkan-region.md)。
本记录不签修复后结果、完整C1、整模型或性能PASS。
