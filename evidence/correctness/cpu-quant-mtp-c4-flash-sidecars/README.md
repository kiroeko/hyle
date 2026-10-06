# Flash-0731 两份实际混合 DSpark sidecar

固定 Windows/Linux 候选与命令见 [summary](summary.json) 和平台收据。
两份 sidecar 的普通 CPU Direct/Graph 完整程序通过，不签 C4/C5 全部验收完成。

- 按已登记 Unsloth `dflash.cpp` 的 F32 图组合；普通投影 Q8_0/BF16，专家始终 MXFP4。
  router、Markov、confidence 保留实际 BF16，HC/norm/bias/sink 保留 F32。
- 129-token target projection 与三套 KV prefill，实际结果进入 draft；三阶段 HC/MLA/MoE、
  五 token batch、cache 反馈/wrap/reset、逆RoPE与分组输出均执行。
- 原 target Q8_0/BF16 embedding 实际行输出进入 decoder；同一 LM 参数服务 main 与 draft。
  完整 129280 行主 head＋BF16 Markov bias 分片验证；confidence 按该原程序输出 sigmoid 概率。
  外部提供 Markov 前一 token，采样/接受策略属于消费者。
- 参考包含原 CPU activation codec/vec-dot。第一轮全零参考来自直接 vec-dot 前未调用
  `ggml_cpu_init()`，CPU half-scale 表尚未初始化；修正原库使用顺序后通过，产品数学未变。

状态缓存在本消费资格中声明为 F32；不据此代签其它 cache dtype 或未测重排。
尚余 C4 安装交付/资源/成本和 C5 完整准入画像，不把文件名 BF16/Q8 当作整个模型的统一 dtype。
