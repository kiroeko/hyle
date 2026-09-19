# C1-Y35 Vulkan 逐元素及门控资格

范围和有限纠正见[Y35施工卡](../../../../docs/implementation/tasks/compute/llm-moe-c1-vulkan-elementwise.md)。

| 候选 / 验证 | 结果 | 原始证据 |
|---|---|---|
| `aa6690c6`，W1 static/shared原生模块 | 141/140项PASS，各含身份复用的Provider19项 | [收据](f2-windows-native.json)、[逐字日志](f2-windows-native-raw.zip) |
| `aa6690c6`，Air原生模块 | 141项PASS，source contract5项PASS | [收据](y35-f2-air-focused.json)、[逐字日志](y35-f2-air-focused.zip) |
| `aa6690c6`，Mini原生模块 | 140项PASS、1项预声明layer测试显式跳过，source contract5项PASS | [收据](y35-f2-mini-focused.json)、[逐字日志](y35-f2-mini-focused.zip) |
| `6af6bfe4`，产品`117b1c13`，W1 static/shared原生模块 | 141/140项PASS；source contract5项PASS | [收据](windows-native.json)、[逐字日志](windows-native-raw.zip) |
| `6af6bfe4`，Air首次构建 | FAIL：size_t/uint64_t的std::min类型推导；GPU尚未执行 | [收据](first-air-failure.json)、[逐字日志](first-air-failure-raw.zip) |
| 首个W1测试构建 / Softplus尾值回归 | 首败保留；已在具名纠正中处理 | [收据](first-failures.json)、[逐字日志](first-failures-raw.zip) |

W1 static的Provider19/Direct95按未变产品和EXE身份复用，Graph27在测试结构断言迁移后通过；
shared实际为Provider19/Direct95/Graph26。私有runner预期数量纠正及原始停止记录均保留。
F2只改route冷期设备限制的一处整数常量类型。Windows受影响Direct/Graph已重编执行，
Provider与source contract依未变输入/产物身份复用；Air/Mini在该候选首次通过本批原生资格。
上述原生记录本身不签实际三交付或工程余门；后续资格如下。

## 实际权重片段三交付与工程资格

`f58d2609`（产品仍`aa6690c6`）的[固定输入与汇总](delivery-inputs.json)、[输入/命令归档](delivery-inputs-raw.zip)
保留实际本地2B权重来源、独立FP64参考、36例/108次每交付的冻结合同。没有consumer纠正。

| 环境 | 结果 | 证据 |
|---|---|---|
| Windows | source/static/shared三条全部通过 | [收据](windows-delivery.json)、[日志](windows-delivery.zip) |
| Air Mac16,12 | source/static/shared三条全部通过 | [收据](air-delivery.json)、[日志](air-delivery.zip) |
| 所选工程门 | 112项资格确认：5项执行、107项按身份复用，8项未选理由保留 | [收据](engineering.json)、[日志及复用证明](engineering-raw.zip) |

六交付合计216例、648次实际执行，含published输出、两槽变输入、旧结果、close后读取及quota
归零。模型权重只用于生成固定片段输入，没有完整attention/state或整模型forward声明。
工程复用核验前批归档、76个未变测试产物、30份SDK清单和45条未启用Vulkan的默认消费。
163条编译命令仅覆盖原厂；另核对完整685项CMake缓存，唯一差异为已知原厂patch副本路径。
未重建45个未变默认producer，没有把复用写成新运行。

[统一实现统计](current-loc.json)：48,943行/198文件，不含测试/第三方；[清理](input-cleanup.json)
四个失效输入压缩包共95,384,616字节。适用原生、实际交付和工程资格完成后停止重复；C1完整
成本优选/默认规则/自实现淘汰、CPU Q2 BLOCKED及共同矩阵保持未签收。
