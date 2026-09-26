# D5：Kimi / GLM Flash 六画像投影批次

候选 `40699b3d7`；精确完整 SHA 见 [收据](receipt.json)。Windows canonical 的同一候选同步至
MacBook Air。仅修改测试消费者，无产品数学、Provider 或路由策略修改。

四路线 **Direct/Graph 八入口全部 PASS**，每入口一个批量测试遍历30种 profile/dtype/shape 签名，
每种两次变化输入。零失败、零 SKIP。30种签名对应51个真实权重角色，同签名保留全部原始名字，
不为 q/k/v 等等价 MatMul 重复建立测试驱动。

- 六画像：Kimi K3、GLM-5.3-Flash，各自官方、主要 GGUF 和高保真 GGUF。
- 原生存储：官方 BF16；Kimi 主要 GGUF 的 beta 为 F32，其它所列投影 Q8_0；Kimi 高保真为 BF16；
  GLM Flash 两个 GGUF 的所列投影为 Q8_0。不能由 artifact 的 Q4/Q8 标签替代逐权重事实。
- 保留 H7168/4096、投影宽12288/8192、低秩128、beta宽96/64及输出投影方向。
- BF16 输入、权重和输出均保留 BF16 舍入；Q8_0 使用原生 block 和独立解码 oracle；
  F32 权重保持实际转置物理布局。非零稀疏内容仍保留完整物理矩阵与全部输出行。

来源 [tensor 事实](../../../../tests/compute_models/d5_kda_projection_facts.json) 固定原文件 SHA、
tensor 名、dtype、shape 和 shard。生成器直接读取此前保存的官方/GGUF headers，没有重新下载。
其中另记录 Kimi A_log[128] 的前96项加载规则，以及 GLM 官方 BF16 conv/norm 与 GGUF F32 的差异。

**边界：** 本批只签 R03 所列投影，不代签参数转换、完整 KDA/MLA/MoE、三交付或 P1。
此前 [共用核心批次](../moe-shared-core-batch/README.md) 的资格继续保留；完整画像仍72/176。
selector plan/跳过理由与定向运行日志均保留，不宣称完成整个 selected-factor suite。
输入工具13项、closure、docs、diff 检查通过。复用四条构建树、重型验证串行。

实现代码相对统一 [LOC 报告](../moe-shared-core-batch/loc.json) 未变化：**58,255行／228文件**，
不含测试与 vendor。本批未创建额外完整 build/vendor 副本，传输 bundle 用毕清理。
