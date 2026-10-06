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
GDN/KDA程序，保留独立oracle和状态反馈，补验实际storage-to-port接缝。尚未运行，不计PASS。

剩余工作：核对80条role合同的完整消费和精确身份、运行六类卷积接缝，并汇总required-region。

另一明确缺口是新版DeepSeek GGUF主干：官方主干收据不能代签GGUF的F32 state/math。
固定Pro-0813/Flash-0731各UD-Q4_K_XL/UD-Q8_K_XL四个精确身份，共用原MXFP4专家和BF16 router，
ordinary/shared投影为Q8_0或BF16。已一次接入原GGUF完整attention/compression、ordinary/hash MoE、
mHC输出后全词表head消费者，原完整oracle/cache反馈保留。这四项尚未运行，不签PASS。
本轮有限P0成本与性能适用边界见[成本审计](cost-audit.md)。
