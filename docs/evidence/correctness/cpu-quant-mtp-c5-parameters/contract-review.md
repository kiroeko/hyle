# C5 参数合同复核

本页复核[80条索引](parameter-witness-map.json)的语义，不以testcase名称单独判定。
固定C1全部参数storage均F32；相同模型族/角色/shape/精度/算法的参数在不同量化档位间复用，
主矩阵编码另由559实际合同资格化。这里证明参数合同，不声称1,354次整模型运行。

| 实际参数 | 必须相同的合同与源码核对 | 运行作用域 |
|---|---|---|
| hidden norm：2048/4096/5120/6144/7168/8192 | F32 gamma、末轴RMS；Qwen/DeepSeek epsilon1e-6，Kimi/GLM1e-5，与C1首shard metadata一致 | Qwen三真实宽度严格F32；Kimi/GLM完整FFN/MLA；新版DeepSeek完整mHC后norm/head |
| Q/K norm256 | F32、仅head末轴；Qwen D256但只RoPE64，theta1e7，HalfSplit；三实际Q/KV几何 | 当前expanded两系统两入口，旧full256/default-epsilon夹具不代签 |
| MLA q-rank1024/1536/2048、KV512 | 对应各模型rank/latent，RMS gamma长度匹配，不把head数当rank | Kimi/GLM完整absorbed MLA/cache及新版DeepSeek ratio0/4/128完整attention |
| GDN a/bias16/48/128与norm128 | GGUF a已fold为-exp(A_log)；bias按head、F32 state；norm后SiLU output gate | 九画像实际state反馈、六原卷积形状补验；与官方head重排分开 |
| KDA a96/64、bias12288/8192、norm128 | a原head数，经连续view broadcast；bias为head*depth；per-key受限decay、原L2 epsilon；F32 state | 原完整KDA带projection/history/state；新三Q/K/V原storage形状接入，不能用简化GDN代签 |
| 六卷积shape/十二role | GGUF最快轴反转，singleton去除；Qwen[channel,4]，Kimi/GLM三参数分别reshape后concat | 六实际storage class、完整GDN/KDA反馈，两系统普通两入口 |
| router bias896/288/384/256 | F32 correction只影响选择，原score另归一化；sqrt-softplus与sigmoid合同分开 | 原完整Kimi/GLM MoE及新版DeepSeek ordinary/hash MoE |
| routed norm3584、shared gate8192 | Kimi latent降维/专家combine后norm再升维；Qwen shared gate是F32 vector→逻辑MatMul→sigmoid | 完整latent MoE；新完整NextN保留gated shared expert，不把一维gate当普通norm |
| attn/ffn/output res_score7168 | 对bank/current作RMS与learned scoring、softmax mix；不是普通逐元素残差 | Kimi普通/banking完整AttnRes，actual bank反馈 |
| mHC base24/scale3、head base4/scale1 | F32原base/scale、Sinkhorn20及pre/post combine；head gate与final norm独立 | GLM Flash与新版DeepSeek完整mHC；产品输出送完整vocab head |
| indexer k_norm/bias128、compressor norm128/512 | 原LayerNorm1e-6（GLM固定源码k_norm）；learned pooling与压缩RMS、position/selection/state分别保留 | GLM完整DSA/pooled-tail及新版DeepSeek indexer compressor/cache，不以APE MatMul代签norm |
| nextn enorm/hnorm/shared_head_norm4096/5120/6144/8192 | embedding-first，独立gamma，模型epsilon；最终head norm在decoder之后；carry/cache实际反馈 | GLM完整NextN；Qwen27B完整Q4 sidecar；Qwen2.4T完整mixed NextN及全vocab，输入边界单独收据不足 |

## 复用边界

- 80条逐模型族关联的源码anchor/SHA、两系统Direct/Graph收据都在索引。源码核对的是实际
  dtype、长度、使用位置、epsilon、广播/布局及state；不是仅匹配文件名或format标签。
- 参数F32 bytes不因主矩阵量化改变；其连续view与原port元素顺序一致。任何不同shape/角色仍保留独立行。
  不把旧BF16、交错/full256 RoPE或默认epsilon结果当本轮Qwen实际F32资格。
- 104精确制品画像已各执行实际projection；全部矩阵格式/宽度/group/lookup合同另有F1/F2收据。
  参数复用不免除这些条件，也不自动授予整模型引擎或全部混合DAG资格。
- 当前core/patch/ISA与原软件身份相同；CPU-only新版Native旧WIN不启用。成熟Provider的
  原model数学与普通入口实际运行已核对，未将未知GPU状态或另一版本结果代签。

结论：这80条参数角色/shape合同的有限复核完成。全准入required-region最终聚合、资源/交付
证据关联与七门终审仍须由C5完成；本页不提前关闭goal。
