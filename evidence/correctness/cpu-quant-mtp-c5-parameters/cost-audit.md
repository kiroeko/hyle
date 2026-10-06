# C5 有限差异的 P0 成本

产品与原数值消费者为 `46a55325a465a94da3d79d2e5e92b0cb6b2cfb62`；新增完整消费者为
`9c9bbfd9d`，只改tests/docs/evidence。沿用[C4冷热审计](../cpu-quant-mtp-c4-installed-sdk/cost-audit.md)，
本页只列本批差异，不签未完成的correctness/required-region gate。

## 实际成本和边界

- 大Q8_0 embedding只扩CPU-only新版Gather第一个encoded输入的总字节准入。
  `ggml_cpu_provider.cc::representable/supports`保留维度、元素数、stride、输出和索引边界；
  准入检查在prepare。既有GetRows执行按原row/stride定位，不增加模型查询、计时或路由搜索。
  2.16GB调用方权重及原invocation backing真实存在，不能因只选几行宣称零驻留成本。
- F1原F16权重明确cold转F32并持有；原F16 activation舍入在每次调用发生，再经成熟F32 MatMul。
  冷存储至少包含原2*N*K与转换后的4*N*K字节，另计Provider必要临时及ownership；
  不把这条路径称为原packed零转换程序或完整性能优势。BF16走原bfloat-linear-F32合同。
- NF4/nested/AWQ/CT显式cold decode后复用constant；MLX按原current K group与selected expert执行。
  `torch_cpu_program.cc`调用固定借入的bitsandbytes/AutoAWQ/compressed-tensors/MLX程序，
  没有新增数学核或全浮点专家库热展开。ATen临时、scale转换、索引复制及结果copy仍是真实managed成本。
- 六类卷积实际GGUF shape反转后只含连续singleton维度；公共View恢复[channel,4]。
  Kimi/GLM的三个Q/K/V源参数需Concat；该合并及可能的route materialization必须计入普通执行成本，
  本批只证明完整程序可消费，没有承诺这段cold整理由编译器自动移出热调用。
- 新版DeepSeek GGUF主干使用原F32 state/math及MXFP4专家，不能套用官方BF16状态边界。
  mHC产品输出进入实际Q8/BF16 head，完整vocab按4096行驻留分片验证；分片是消费测试安排，
  不宣称HYLE提供模型loader、全模型驻留或分片推理策略。

## 性能适用范围

本批新编码/显式decode及MLX合同没有新增同合同成熟竞争者；没有新winner规则。
普通浮点MatMul、norm等沿用已存在的成熟默认顺序；Native旧187个WIN要求原peer软件身份，
对新版CPU-only ggml不生效。本批数值结果不成为这些浮点域的新版胜负证据。

没有以本批新增的同合同重叠或机制反例触发P1；不跑逐档位计时或自研竞赛。
若最终对账出现具名完整接入成本风险，仍须按总计划两代表族/一预算处理，不能由本页提前免验。
