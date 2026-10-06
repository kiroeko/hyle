# C5 完整Qwen2.4T NextN

候选 `61155eed2a739ddbd08d8e3bfc79d0f067132ae6`，Windows/Linux普通Direct/Graph各一个测试，
每个测试执行Q8_0和UD-IQ1_M两份完整程序，合计8次profile/entry/platform旅程通过，没有skip。
[Windows](windows/receipt.json)、[Linux](linux/receipt.json)保存命令、原日志、XML及作用域。

原Unsloth llama `46cbf0e95786` qwen35moe.cpp SHA
`e449e79acb469d122addd9a9aaa2886fc99c13c3b5826877f84e9c6ab38b25bd`的完整NextN顺序已核对：
embedding-first双norm/eh_proj，GQA/cache/gate/output/residual，post norm，routed+gated shared MoE，
FFN residual，shared-head norm，h_nextn与完整共享head。

- H8192、Q64/KV4/D256、E512/top10/I2048/shared2048、V248320，所有内宽和词表跨度保留。
- 原block角色：eh_proj Q8；Q8或Q5 attention；Q8或Q2 experts；Q8或Q5/Q6 shared；router/norm F32。
  embedding/head是Q8或Q4_K，不按UD-IQ1_M文件名把所有tensor误当IQ1。
- 实际产品embedding输出进入decoder；carry与两份KV用每轮产品输出反馈，独立oracle自己推进。
  四次changed replay；产品h_nextn进入全部248,320行head，4096行驻留分片，不缺行。
- 原activation codecs/vec-dot是reference；norm1e-6、HalfSplit64/theta1e7、sigmoid gate及
  softmax/top10/normalize、SiLU专家和shared gate全程保留。
- 驻留专家减少到8类，显式I32表将真实512-way选择映射到合成重复权重；router512、top10与实际K/N不减少。
  这是计划允许的验证容量安排，额外映射是真实消费者成本，不声称产品实现专家offload引擎。
- 产品数学/API和数值门没有变化；只批量接线测试消费者。合成合法非零权重不是全模型实权重推理。

该资格补全NextN完整组合，不由输入/projection/carry边界代签。其它档位的全部实际格式合同仍引用F1矩阵。
