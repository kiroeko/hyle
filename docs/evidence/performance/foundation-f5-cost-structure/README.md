# F5固定candidate成本结构P0

产品25e，按原六FIX与真实消费作结构审计，不运行P1或用纳秒/地址次序改API。
N为physical max_query_rows，H为query_heads，D为head_dim，L为冷max_kv_per_request，
R为实际request数，P为page refs，M为schema固定metadata bytes。F16/BF16均每元素2字节。

| 阶段 | 有界成本与事实 | 验收边界 |
|---|---|---|
| schema/version | 固定容量checked乘法；版本校验/active span复制O(R+P)，固定metadata backing清零O(M)，owned host snapshot/generation | 不从payload地址猜版本；host spans不跨调用借用；不把capacity清零漏计 |
| PagedArena create/read | backing=pages×page_size×kv_heads×D×2+sentinel；create O(max_pages)，read两集合排序/去重O(P log P)、O(P)只读guards | 精确物理页、原F1 independent blocks；不是重建KV payload或cache/eviction策略 |
| native Direct cold slot | 一次固定输出/metadata mirror/float/int workspace、固定use/retained completion capacity；output=2NHD，Graph外部输出不另分配Direct输出 | 用户显式容量；native计划私有host/driver分配另为UNKNOWN |
| native prepare/update | M字节原F2 H2D；原PrefillPlan显式plan；完整query/kv indptr key相同才reuse，page IDs/tails经新同版本mirror供run | plan不放在submit；paged尾增长但page indptr未变可复用，跨页/拓扑变化需plan；报告host waits通常reuse1/new plan2 |
| native run | 既有FI scheduler/kernel+实际lease/read权限和retirement；paged permission按distinct pages接线 | HYLE不额外gather整历史KV；原数学/native occupancy/dispatch成本不声称0或capture |
| reference metadata | indices=4NHL，mask=4NL；构造O(NHL)，同版本host backing | capacity包含padding，不把padding当有效KV；inactive gather只读实际页footprint |
| reference payload | gathered K/V逻辑bytes=4NHLD；原gather/GQA展开/F32 cast/MatMul/softmax/Copy | 额外O(NHLD)数据移动是显式参考档费用，不能称所有路线native paged/同吞吐 |
| reference output/workspace | logical_intermediate_bytes是所有views/copies的保守总和，含alias重复；真实native scratch/driver另外UNKNOWN | 不是resident upper bound；严格未知预算可拒绝，未知不能写0 |
| Apple/Vulkan native mirrors | 一staging+两device bank：host bytes B、device2B、alignment余量378，总3B+378；原F2每layout upload B | 实测样本B=1920/backing6138；Apple waits2、Vulkan waits1，仍非GPU私有总成本 |
| CPU fixed controllers | 原11个已准备stage；Direct每stage显式同步，原Graph固定recipe，独立output backing | 明确wait/copy价；不是auto native fallback或吞吐WIN |

来源是固定src/compute_batch/{batch_layout,batch_execution,batch_materialize,paged_arena}.cc、
原FI update/plan adapter与detail/materialized_native_controller.h；字段checked byte值来自
已签实际consumer日志。七profile ragged append/F4实测dirty clone2560+append512+tail-init
1024=4096，full backing4608；这是有限ragged重排/初始化账，不把它写成任意paged token
append的零复制，也不自动证明所选模型KV编码。原native quota统计不计私有driver heap。

[普通Graph字段/codegen](../foundation-f5-ordinary-graph-p0/README.md)证明本匹配配置无新增
默认字段/warm指令税；opt-in callback/metadata/frame owner费用留在本新对象。
[产品LOC](../../architecture/foundation-f5-loc-inventory/README.md)分别记录代码/依赖/owner范围。
当前结构没有依benchmark行新增算法/API、无无限bucket/每run JIT、未发现需自动P1的
同合同热点。本结论只P0结构与已声明成本边界，不签单机引擎吞吐/延迟或各平台capture对称。

若目标decode每步额外materialize全历史KV不可接受，native/equivalent resident consumption
成为下一轮具名基础缺口；本goal的显式参考资格不能代签该高性能诉求。模型选择、位置/
page/cache/batching策略仍在引擎，最后工作量评估必须保留这项依赖风险。
