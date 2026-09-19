# C1-Y35 Vulkan 逐元素及门控资格

范围和有限纠正见[Y35施工卡](../../../../docs/implementation/tasks/compute/llm-moe-c1-vulkan-elementwise.md)。

| 候选 / 验证 | 结果 | 原始证据 |
|---|---|---|
| `6af6bfe4`，产品`117b1c13`，W1 static/shared原生模块 | 141/140项PASS；source contract5项PASS | [收据](windows-native.json)、[逐字日志](windows-native-raw.zip) |
| `6af6bfe4`，Air首次构建 | FAIL：size_t/uint64_t的std::min类型推导；GPU尚未执行 | [收据](first-air-failure.json)、[逐字日志](first-air-failure-raw.zip) |
| 首个W1测试构建 / Softplus尾值回归 | 首败保留；已在具名纠正中处理 | [收据](first-failures.json)、[逐字日志](first-failures-raw.zip) |

W1 static的Provider19/Direct95按未变产品和EXE身份复用，Graph27在测试结构断言迁移后通过；
shared实际为Provider19/Direct95/Graph26。私有runner预期数量纠正及原始停止记录均保留。
这不是本批Mac、真实2B三交付、全部工程门或性能资格；不签收C1或共同矩阵。
