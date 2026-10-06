# C4 首批实际新版计算消费者

精确 Windows/Linux 候选、实际命令与范围见 [summary](summary.json) 和平台收据。
本目录只签列出的完整片段，不签三个 DSpark stage、完整模型或整个 goal 完成。

- Pro-0813 H7168/R512、Flash-0731 H4096/R256，沿 C1 实际 header/config/revision。
- 三个目标层的 HC mean→concat→原 FP8 main projection→norm；原 BF16-source/F32 confidence。
- BF16 Markov embedding、所有 129280 词表行的 F32 head 与 logit bias，保留五个 draft row。
  head 用 resident shard 控制峰值，不缩小词表或模型内宽。
- 原 target KV prefill128/129、窗口回绕、五 token 非因果 draft sink attention、inverse RoPE、
  分组 wo_a/wo_b。wkv 只绑定一次；wo_a 的原 FP8→BF16 准备为 cold 操作。
- prefill 的真实结果进入 draft；后续重放反馈实际 cache，独立参考推进；reset 重新绑定原 cache。

## inverse RoPE 的输入对账

[诊断收据](inverse-rotation-diagnostic/receipt.json)保留最初失败。两入口 Pro 的六次重放，inverse RoPE
相对于**实际 attention 输出**的独立原公式都逐值相等（max_abs=0），原 pointwise 相对门却将另一条
参考链的 BF16 输入产生的小量相减结果与它比较。这不是相同输入的算子比较。

修正仅在测试消费者中：该中间端口按实际前驱执行独立公式逐值检查；前驱 attention、cache、后续
分组投影和最终输出仍按独立完整链比较。没有改产品数学、放宽全链容差或把未运行项写为 PASS。
Windows/Linux 首批四片段通过。

## 新版主干后续

[后续收据](main-followup/summary.json)验证新版Pro/Flash的ratio0/4/128 attention、
压缩/索引/cache反馈、普通及hash路由的完整FP4专家/FP8共享MoE，两平台Direct/Graph通过。
旧test recipe的preview对象指针限制已按实际语义/shape替换；其原公式与产品Provider未改。

完整三阶段 HC/MoE/共享主 head、Qwen/Flash sidecar、C4 资源/三交付/P0
以及 C5 全部准入画像仍未完成。没有新自研核或性能赢家裁决。
