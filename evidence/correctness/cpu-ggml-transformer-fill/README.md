# CPU 成熟接线：Softmax、非因果 Attention、精确 Fill

产品候选 `5ebd167aa29a5f2bb022c71ddfda6d96f3da6f47`；Windows canonical，Linux 单向验证副本。
[TF1–4 当前卡](../../../docs/implementation/tasks/compute/cpu-ggml-transformer-fill-admission.md)。

## 实现与普通消费

- Softmax：dense F32/F16/BF16，最后轴正负写法、有限 scale、同类型 dense additive mask 的 NumPy 广播。
  使用原 F32 cast/scale/add/Softmax/output cast；存在 NaN/+Inf 或全无有效有限值的行按公共合同输出 NaN。
  原核前安全行替换与原核后恢复在 region 内完成，Softmax→Copy 实际消费证明下游获得修复结果。
- 非因果 SDPA：原 score→Softmax→value 整程序，BSHD、三 dtype、GQA 与 query 长于 key 的消费。
  原 causal F16 flash 保持；causal F32/BF16 score 程序只在 causal 时施加原 mask，并采用同一安全 Softmax。
- Fill：无输入、普通 Direct 与 Fill→Copy Graph、标量/空/多维；冷存原生 scalar bits，经原 `repeat_4d`。
  F32/F16/BF16 signed zero、Inf 和 quiet NaN payload 精确保留，没有新 HYLE Fill 数学/循环。
- 普通用户只选 CPU；准备时绑定，Graph 保持成熟 region，执行不重建、搜索或计时裁决。
  同一独立 C++ consumer 只链接 `hyle::compute`/`hyle::graph`，安装消费只给 HYLE prefix。

## 实际资格

| 环境 | 正确性与普通入口 | C++ 交付 |
|---|---|---|
| Windows MSVC | Provider52/52、CPU12通过/1旧fixture跳过、Graph15/15；三dtype普通Direct/Graph与变化输入通过 | 源码、installed static、installed shared 同一消费者通过；DLL按既有helper邻置 |
| Ubuntu24.04 GCC14.2 | 同样52/52、12通过/1旧fixture跳过、15/15及三dtype普通消费 | 源码、installed static、installed shared 同一消费者通过 |

Provider 新 witness 覆盖 rank1/2/4、非平凡 leading/row mask 广播、scale、NaN/+Inf/all−Inf/all-mask、
零scale×Inf、输入变化后恢复、原位拒绝、metadata 热无分配与 close 平衡；非因果 attention 的
多batch/GQA/奇数depth/query>key 与独立double oracle通过；原有因果、OOM、生命周期测试同批通过。
跳过的 `D5RealQwenProjectionUsesExactProfileAndReplays` 未提供旧模型 fixture，不计 PASS。
完整模型权重/GPU/新增ISA/P1未运行，本轮不为它们签资格。

首次GCC编译将测试中的vector循环复制视为WERROR；两行改引用后续跑原批，产品数学不变。
失败和append receipt保留，没有放宽警告或用新测试改变合同。

## 成本、终审与剩余域

P0：一个F32安全score payload、两次O(n)行检查；scale/add各为原节点，包含其临时值和mask布局
materialization，不能与未经适配的 fused 原调用比较同成本。无新增exp/求和/归一化数学。
冷arena/原pool/独立invocation规划并复用；不新增热分配、Provider搜索或syscall。
Fill只有冷标量位与原repeat，无完整常驻输入副本。性能胜负和Native清退仍冻结，不能由接入签WIN。
分离的终审核对数值修复顺序、原位/未接域拒绝、资源owner/回收、原库边界与普通交付，TF1–4七门通过。

其它axis、Bool mask、LogSoftmax、非dense Softmax、外部alias、其它attention mask/layout/depth，以及
非floating Fill保留精确拒绝/其它已登记carrier；拒绝不等于成熟库无能力或Native UNIQUE。
其它activation/reduce/index/layout候选继续按全族ledger批量处理，CPU及goal未完成。
Compute + Graph 为 **61,133 实现行 / 248 文件**，统一脚本，不含测试与vendor。

[收据](receipt.json)与[工件](qualification.zip)保存实际命令/失败/XML、同一consumer、1,582项
单向产品输入身份、selected/skipped reasons、驱动和LOC；不归档SDK/build大副本。
关闭仅改状态/文档，按selector不再构建或跑P1。重开由本轮数值、布局、资源或交付的具体反例触发。
