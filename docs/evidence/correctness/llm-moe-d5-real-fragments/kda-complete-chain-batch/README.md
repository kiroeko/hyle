# D5 六画像完整 KDA 链

候选 `c800b76e` 的完整 SHA 见[机器收据](receipt.json)。CPU、CUDA、Apple、Vulkan 的
普通 Direct/Graph **32/32 测试组通过，零跳过**。其中六画像完整 KDA 的四路线八入口形成
**48个 R06 region 验收项**，已写入现有 ledger 的 `partial_receipts`；完整画像仍为72/176。

## 真实边界

| 模型 | hidden / heads / depth | 投影存储 | 参数及状态 |
|---|---|---|---|
| Kimi K3 官方 | 7168 / 96 / 128 | BF16 | F32 conv/norm；原 A_log[128] 只取96 heads |
| Kimi 主流 GGUF | 同上 | Q8_0，beta 为 F32 | F32，folded negative-A |
| Kimi 高保真 GGUF | 同上 | BF16 | F32，folded negative-A |
| GLM Flash 官方 | 4096 / 64 / 128 | BF16，未被全局 FP8 配置覆盖 | BF16 conv/norm 实际 Cast，F32 A_log/dt/state |
| GLM Flash 两个 GGUF | 同上 | Q8_0 | F32 conv/norm、folded negative-A/state |

每个画像接齐 q/k/v 投影、K4 causal conv、Q/K L2、两级 decay gate、beta、GDN、输出 norm/gate
和 o 投影。Kimi 输出门为全秩，GLM 为 g_a/g_b。八次变化输入均把**路线自己的卷积历史和
recurrent 输出状态**回送给下一次执行，独立 oracle 自行推进；没有用 oracle 状态覆盖实际反馈。
BF16 投影使用既有 `BFloatLinearF32` 的 BF16 输入/权重、F32 输出；Q8_0 保留原生块及点积。
完整链权重和输入是格式合法的合成非零值，保留真实内宽，不冒充巨型 checkpoint 内容验证。

## 参数纠正与证据边界

[原始配置摘录](config-facts.json)来自既有官方 metadata，包含原文件哈希及固定模型 revision。
两个模型的 RMSNorm epsilon 都为 **1e-5**。固定 Kimi source 的 Q/K L2 也使用1e-5，GLM 的
L2 独立固定1e-6；两个 gate lower bound 均为-5。Kimi latent MoE norm 同步改为其配置1e-5。
旧共享核心、旧 MoE 和初始 `9bb6be90` 使用1e-6的相关结果保留，但撤回其对上述真实参数的资格。
本批没有扩大数值容差，没有修改产品算子、Provider、shader 或数学实现。

32组包含完整 KDA、既有 KDA 核心、官方 Kimi 混合存储 MoE、八个 GGUF MoE。共享多状态 helper
变更还在 `9bb6be90` 通过四路线九画像 Qwen 单状态回归；随后仅修改 Kimi/GLM 夹具与 oracle，
复用该证据。CPU 初始 Qwen 因未设置 fixture 环境变量而 SKIP，后续单独配置真实本地 fixture 的
Direct/Graph 日志均 PASS；初始 SKIP 没有被改写或计为通过。初始 BF16 类型编译错误日志也保留。

本批复用现有构建树，低并行、重型串行。selector 保留所选门和跳过理由；本批为 region 定向
验收，完整 D5 factor、剩余 region、三种外部交付与 P1 未由它代签。常量准备转换和 native workspace
的完整成本仍须纳入稳定路线 P1。统一实现统计 **58,588行／232文件**，不含测试和第三方源码。
