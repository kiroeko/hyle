# 模型、量化与算子地图

本页是**局部重点：新增模型或制品时查需求与计算边界**。它不是支持白名单，也不是逐实验进度表；运行范围看[引擎指南](../user/inference.md)，当前任务看[看板](../developer/inference-progress.md)。

## 先分清三种事实

| 要查什么 | 唯一来源 | 不能由它推出什么 |
|---|---|---|
| 固定模型／制品身份 | [需求目录D](tasks/compute/llm-moe-provider-demand-matrix.json)的`current_scope`、`profile_catalog`与`added_releases` | 收录不代表已经支持或完成整模型验证 |
| 计算、量化与状态需求 | 本页及D的`demand_units`；需要追溯时查[历史需求C](tasks/compute/llm-moe-coverage-matrix.json)和[原生映射P](tasks/compute/llm-moe-provider-map.json) | 历史编号／数量不是能力上限，也不是当前实现状态 |
| 实际实现与运行结果 | 当前源码、所属测试／证据；当前完整运行入口见引擎指南 | reader、单个原角色或生成fixture通过不等于完整模型通过 |

固定需求目录覆盖三个Qwen、Kimi-K3、GLM-5.3／Flash及两个正式DeepSeek V4，共8个模型身份；旧DeepSeek preview另行排除。24个原profile及后续派生制品只是固定调查范围，不限制新增真实发布格式。完整文本入口仍以Qwen为主，其它家族组件结果保留各自上游输入／初始历史／专家与设备边界。

## 模型到计算与融合子图

公共项包括布局/物化/转换、embedding/head、norm/逐点/残差、线性/FFN与状态资源复用（D：R01–R03、R15）；量化另查 R14。R 编号只是区域索引，不表示所有 dtype、role 或 fusion 已接入。

| 模型 | 主要计算与待核融合子图 | D / C 索引 |
| --- | --- | --- |
| Qwen3.5-2B / Qwen3.8-27B | Q/K/V→norm/RoPE→GQA→output gate→projection；causal conv→gate/decay→GDN→state；dense FFN；按profile核MTP | R04/R05/R13；F03.4、F04.3/4、F07–F10、F14.3 |
| Qwen3.8-2.4T-A95B | 上述attention/GDN；router→TopK→dispatch→grouped/shared experts→combine；MTP | R04/R05/R10/R11/R13；F07–F10、F12.1/2、F13.1/2、F14.3 |
| Kimi-K3 | KDA三路conv→gate/decay→recurrent；MLA；latent expert→SiTU/再量化→combine；AttnRes | R06/R07/R10–R12/R14；F09、F11–F13、F14.2、F15、F18.3 |
| GLM-5.3 | MLA/YaRN；index projection→score/TopK→sparse gather/attention；跨层选择复用；MoE；MTP index carry | R07–R11/R13/R14；F08.2、F12/13、F14.5、F15、F16.1/2、F18.1/5 |
| GLM-5.3-Flash | KDA/MLA；pooled index+未池化tail；跨层选择复用；MoE；mHC；MTP | R06–R14相应分支；F11–F16、F18.1/5 |
| Pro-0813 / Flash-0731 GA | MLA/CSA/HCA；H64 index；hash / sqrt-softplus router；mHC；FP4 expert+FP8 activation；DSpark | 对照R07–R14、旧F08/F12–F18；DSpark另列新合同 |

## 续算状态与多会话接入

同一快照需要覆盖模型的全部续算状态；请求目录、预算、COW、退休与缓存机制共享。新的状态数学或物理格式需要真实原语，不能仅靠目录或模型名称宣称支持。MTP接受前缀、修复、pending与RNG须对应同一次完整发布。

| 模型／状态族 | 同一快照须保留的内容 |
|---|---|
| 三个Qwen | 各attention层KV、各线性层conv历史／GDN状态、有效长度与位置 |
| Kimi-K3 | MLA latent／位置历史、Q/K/V短conv历史、逐通道KDA状态与位置 |
| GLM-5.3 | CKV512＋旋转KPE64、非池化E4／F32scale index缓存、绝对位置／2048选中ID、待追赶journal及hidden／pending |
| GLM-5.3-Flash | KDA／conv、NoPE MLA、index keys／gates、已pool缓存和未pool尾窗、偏移／位置 |
| 两个正式DeepSeek V4 | SWA环、ratio4／128 compressor KV及score窗口、独立BF16-QAT index历史、环位置／压缩相位／长度／映射 |

状态与资源的公共合同见[整体架构](../architecture/inference-design.md)及[状态版本](../user/state-versions.md)，具体有限角色结果按需查历史明细。

## 模型与量化身份

新增制品先确认真实仓库、固定revision、base model、逐角色编码与数值合同。名称相同或scale形状相同，不足以合并reader；不能用完整profile的格式集合替代每个role。

| 格式类别 | 接入时必须分清 |
|---|---|
| F32／F16／BF16 | 权重、激活、状态和累加精度分别声明，保留中间舍入 |
| GGUF混合codec | 实际tensor type、packing、stride与角色；不能从Q4／Q8文件名推导全部数学 |
| FP8 | block128与channel／token、scale编码、激活策略及原计算精度 |
| FP4 | ModelOpt NVFP4、local-dynamic16、MXFP4／E8M0及DSpark分别核实；即使group相同也不能归并 |
| INT4 | codes、zero／bias、group、axis、orientation和scale类型；GPTQ与CT等源合同不能互相套用 |

固定身份逐项明细直接查D；[历史制品表](../evidence/history/model-operators-2026-10-31.md#模型与量化身份)用于追溯当时调查，不与D重复维护。

## 当前融合与量化接线缺口

当前待做的执行／融合／带宽任务只在看板维护。核候选时依次看实际consumer、admission／lowering、精度／中间值、资源alias与状态顺序、原生dispatch及完整消费者收益。

Graph region、批量capture和物理kernel融合是不同事实；kernel数量、Copy减少或局部带宽改善不能直接换算整引擎提速。已有源合同和真实消费者仍在时，不能因旧preview退出就删除共享实现。

## 按需细节与历史索引

以下只在查具体格式、原角色或旧编号时阅读，日常开发不顺读、不逐片更新。

| 细节 | 查哪里 |
|---|---|
| 原角色／状态及有限组件结果 | [历史状态与接入表](../evidence/history/model-operators-2026-10-31.md#续算状态与多会话接入) |
| DeepSeek正式版、DSpark与数值区别 | [角色、存储与数值边界](../evidence/history/model-operators-2026-10-31.md#正式版-deepseek角色存储与数值边界) |
| 派生制品和源格式组合 | [扩展量化制品](../evidence/history/model-operators-2026-10-31.md#扩展量化制品) |
| 已调查的融合／接线候选 | [原调查表](../evidence/history/model-operators-2026-10-31.md#当前融合与量化接线缺口)；其中“下一步”不是当前任务 |
| 实际使用的算子与图数量 | [有限Qwen数量示例](../evidence/history/model-operators-2026-10-31.md#单一模型工况的数量示例)，不外推其它模型／配置 |
| 历史F编号与descriptor反向关系 | [需求C](tasks/compute/llm-moe-coverage-matrix.json)、[映射P](tasks/compute/llm-moe-provider-map.json)，按下面入口追溯 |

## 单一模型工况的数量示例

见上方有限数量记录；它描述固定图，不代表公开算子数、region数与物理kernel数可以互换。

## 正式版 DeepSeek：角色、存储与数值边界

见[固定角色参考](../evidence/history/model-operators-2026-10-31.md#正式版-deepseek角色存储与数值边界)。当前接入仍须核对应GA角色，不能继承preview、其它sidecar或整模型通过结论。

## 扩展量化制品

见D的身份记录及上方历史表；reader、冷绑定、原payload、完整模型和性能分别判断，不设“每模型几个profile”的上限。

## 64 条历史需求的反向索引

按需查[原64项映射](../evidence/history/model-operators-2026-10-31.md#64-条历史需求的反向索引)。编号只用于来源追溯，不作为当前任务表。

## 25 个 custom descriptor 与实际反向关系

按需查[原descriptor映射](../evidence/history/model-operators-2026-10-31.md#25-个-custom-descriptor-与实际反向关系)，实际定义与消费者以当前源码为准。

## 继续细化与融合审查边界

本页只在模型、格式或真实合同变化时更新。新增实验日志直接归所属任务；有长期复用价值的结论更新对应合同或当前性能结果，不向此页追加运行流水。
