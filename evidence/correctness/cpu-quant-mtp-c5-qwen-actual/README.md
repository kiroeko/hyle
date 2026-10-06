# C5 Qwen实际参数与消费边界

候选 `a68f8ed7a5ead29cd418c4642388a7a02a2f6009`，两系统普通Direct/Graph各3项，共12项通过，
没有skip。产品未改变，本批是使用成熟操作的CPU-only消费测试；没有修改GPU测试或旧资格。
[Windows收据](windows/receipt.json)、[Linux收据](linux/receipt.json)保留构建、普通入口、日志及XML。

## 有依据的有限纠正

固定C1 Qwen2.4T GGUF第一shard的metadata：qwen35moe、Q64/KV4/D256、rotary64、theta10000000、
rms_norm_eps1e-6、MRoPE sections11/11/10/0。旧交错RoPE和默认1e-5 norm不能代签此域。
已登记[Transformers固定源码](https://github.com/huggingface/transformers/blob/c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
的rotate_half/apply_rotary_pos_emb使用前后半区；同源码attention使用sigmoid gate。
本地已登记llama qwen35程序也执行MRoPE和sigmoid；本批只验文本三轴position相同的域。

- Q8_0/UD-IQ1_M两份精确C1身份：F32 Q/K norm、HalfSplit64、实际频率、两行KV历史、
  四次changed replay/实际反馈。输出同时核对norm、旋转、cache、attention和gate。
- H2048/5120/8192三个真实模型宽度的F32 norm与残差，明确epsilon1e-6，保持严格F32门。
- 2.4T两份身份的原Q8_0 eh_proj K16384/N8192：embedding-first双norm、Concat、MatMul、
  后norm以及四次carry反馈；原权重与activation codec/vec-dot作为reference。

最后一项是NextN输入/投影/carry边界，不能据此冒称完整NextN decoder已验。
合成合法非零payload也不是全模型实权重推理。投影所有实际格式仍由F1矩阵独立资格化。

## 首轮失败保留

`911efc5c7e10`时两项norm/RoPE通过，MTP反馈失败；新夹具漏标`.quantized=true`，
把Q8反馈链误按非量化F32片段检查。原`sem::mtp`和既有完整GGUF链使用已冻结量化片段门。
修正分类后两系统通过，没有新设阈值、改产品数学或换成未量化F32 oracle；独立norm仍严格检查。

[原失败收据](first-failure/receipt.json)未改写；原log/XML无损gzip归档，解压SHA已核验，
[原始大小与SHA](first-failure/lossless-archive.json)保留。删除已核验的重复原文件325,473,374 bytes，
compressed归档11,233,414 bytes保留。
