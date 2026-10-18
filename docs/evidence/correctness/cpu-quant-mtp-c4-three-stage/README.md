# C4 完整三阶段 DSpark 与实际 sidecar 角色

Windows 固定候选 `6d0d2db37af58d4df69bb77d9103d30c042d1c93`，Linux后续候选
`025354fe6d426b3036e8d5454332bc445c479a5c`（仅补测试可移植初始化），普通 Direct/Graph 均通过。
原始命令、两入口日志与 JUnit 位于 windows 子目录。

- 五 token 批次，共享 BF16 token embedding；三个完整 HC→attention→HC→FP4专家/FP8共享MoE 阶段。
- 三套独立 cache，实际输出反馈、回绕与 reset；每阶段只持有一份批量权重，cold wo_a 准备合并但保留原资格。
- 最终 HC collapse、RMSNorm、F32 confidence；全 129280 行的主 head 与 Markov bias 以 resident shard 资格化。
  每个 shard 的同一 LM 参数同时服务 main 与 draft；采样/接受策略在范围外。
- 批量分数索引使用已准入的两个成熟 Gather，T=5 只产生150个分数标量；不添加整数数学核。

测试构造保留全词表和真实内宽，因此该测试进程的 payload/reference 峰值约8GB；不是模型加载器，
没有下载/落盘全模型，不将此测试峰值冒充 Provider 私有内存边界。其 managed 资源合同仍需包末复核。

[sidecar角色](sidecar-roles.json)保存18/81/81个实际tensor、完整格式/shape与metadata。
Qwen命名Q4_0的MTP实际混合Q3_K/Q4_K/Q6_K/F32；Flash两份sidecar均包含MXFP4专家和BF16
router/Markov/confidence，不能用文件档位名替代tensor编码。该角色表本身不是执行PASS。

Flash sidecar 完整程序、C4 资源/交付/成本和 C5 全矩阵尚未由本目录签完成。
