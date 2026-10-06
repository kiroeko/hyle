# C5 参数角色精确对账

[实际角色表](actual-roles.json)重新读取固定 C1 完整 headers，覆盖104档位的全部小参数，
不只使用需求表每类前三个示例。30类shape/storage细分为80类真实role/shape合同，
共1,354个profile关联、47,653张量。它们不是80个待自研算子。

用途包括 norm、Q/K norm、卷积、门控/时间偏置、router bias、shared expert gate、
残差评分、mHC base/scale、indexer及nextn参数；每条保留原角色、物理GGUF形状、profile关联和region。
本表只签 `DEMAND_ROLE_MAPPED_EXECUTION_RECONCILIATION_PENDING`，不签执行PASS。

六类卷积的GGUF最快轴在前：Qwen形状反转后为[channel,4]；Kimi/GLM三个Q/K/V参数分别带
singleton维度，reshape为[channel,4]再沿channel合并。原数值字节不需要转置或重新计算。
既有完整recurrent消费者已使用归一化权重；C5新增有限消费者从这六个原物理形状接入原完整
GDN/KDA程序，保留独立oracle和状态反馈，补验实际storage-to-port接缝。两系统普通两入口已通过。

剩余工作：核对80条role合同的完整消费和精确身份，并汇总required-region。

另一明确缺口是新版DeepSeek GGUF主干：官方主干收据不能代签GGUF的F32 state/math。
固定Pro-0813/Flash-0731各UD-Q4_K_XL/UD-Q8_K_XL四个精确身份，共用原MXFP4专家和BF16 router，
ordinary/shared投影为Q8_0或BF16。已一次接入原GGUF完整attention/compression、ordinary/hash MoE、
mHC输出后全词表head消费者，原完整oracle/cache反馈保留。两系统普通两入口已通过。
本轮有限P0成本与性能适用边界见[成本审计](cost-audit.md)。

## 实际执行

[Windows](windows/receipt.json)候选 `ed23f301facc564396f5ed34b78f98137464fcd8`，
[Linux](linux/receipt.json)候选 `f53bf27dc35b3e4aa116b50d4f7bab15fe9a825b`。
两者只差docs/evidence，产品与测试source相同；两系统每入口10项、共40项通过，无失败/skip。

- 六类卷积用实际GGUF最快轴/单例形状；三Q/K/V参数逐一View后Concat进入原完整KDA。
  原GDN/KDA独立oracle、实际heads/depth、changed replay与state反馈没有改动。
- 四个新版GGUF精确主干身份保留MXFP4专家、BF16 router和Q8/BF16 ordinary/shared投影。
  ratio0/4/128 attention/cache、indexer compressor、ordinary/hash MoE全部执行；
  mHC产品输出送入全129,280行head，reference来自独立完整mHC oracle和原GGUF codec/dot。
  Head驻留4096行分片，不缺行、不以产品输出自己充当完整reference。
- Linux只前向校验Windows canonical source，batch build并行2、重型串行。没有重跑已资格的594项/入口。
- 未增产品kernel、容差或优化。合成非零合法payload仍不是完整实权重推理。

上述收据签本批六类存储与四个主干消费者，不把全部80条角色或104个完整混合模型改成RUN_PASS。
全部角色/required-region的最终关联与聚合仍由当前C5卡负责。

[80条参数的收据索引](parameter-witness-map.json)包含按模型族的源码anchor/SHA与两系统两入口testcase。
它是可追踪索引，保留生成时的待复核状态；当前源码/合同复核结论见下方复核表，
测试名匹配本身不能改变verdict。
此复核已发现并补验Qwen实际RoPE/epsilon/MTP输入边界，见[准确资格](../cpu-quant-mtp-c5-qwen-actual/README.md)。

上述有限源码/合同复核现已完成，见[逐类结论与复用边界](contract-review.md)。
索引不是1,354次完整模型运行；required-region最终聚合与七门终审仍未关闭。
