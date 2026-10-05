# CPU oneDNN-labelled FFN：混合接缝诊断

已安装TC静态SDK产品`c2df0b8973a80b7140e69983c8e861668483a97c`；消费者HEAD9134c530，产品未改。
三个F32 explicit-accum、三个constant weights的普通四节点FFN实际为一项onednn-graph-region。
输入1x8，gate/up 8x8，down 8x8、每行平均输出；不发布中间值。
−80与−90最终结果匹配，−100/value1e38实得−0.000378350553，稳定F32 activation参考−0.000372044742，
差约1.7%。八个最终输出逐个检查，未计时、未做自研性能比较。

根因接缝源码是graph_provider_onednn的ActivationTask调用`simd::swiglu_f32_batch`；oneDNN承担
矩阵乘法，并非整个FFN数学都来自oneDNN。不能把这个结果记成原oneDNN logistic kernel失败，
也不能因Provider名称就把HYLE SwiGLU记成熟替代。初次down为8x1不匹配旧FFN资格，正确走
四个leaf且三点通过；outside-ffn-*保留这段独立正确事实，不能冒充完整region验证。

[diagnostic.zip](diagnostic.zip)保留两种消费形状、原源码、命令/日志与runtime/product身份。
这只签具体反例，不签产品修复、三交付、性能WIN或CPU完整；当前产品处于独立施工卡管理。
