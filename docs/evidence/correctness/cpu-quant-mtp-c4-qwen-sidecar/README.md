# Qwen 命名 Q4_0 的真实混合 MTP sidecar

固定候选和制品身份见 [summary](summary.json)。Windows/Linux 普通 CPU Direct/Graph 均通过。

- Q3_K 全词表描述的 embedding，选中行的实际 GetRows 输出进入 decoder。
- 原 `qwen35::graph_mtp` 顺序：enorm→hnorm 拼接，Q4_K eh projection、Q6_K 按 head 拆 Q/G、
  Q/K norm、64维 text MRoPE、GQA/cache、gate/output、残差、Q4_K FFN 与 shared-head norm。
- 真实输出 feedback/reset；全248320行Q3_K head用resident shard验证，没有缩小内宽/词表。
- 对照包括原CPU activation编码与vec-dot；不拿未量化F32 projection代签。

本资格限于相同text position的MRoPE退化路线，不声明视觉/音频多轴position的新资格。
产品源码没有变化；C4资源/三交付/成本与C5总矩阵仍待收口，不将这个测试等同整个模型推理引擎。
