# CPU ggml 剩余成熟准入诊断

baseline `7053e854783c0d9ec42d0b8e80a79a7c02251b14`。这是无计时、未改产品的实际诊断；
不签 Native WIN，不声称完整能力资格。已资格化 AVX2 主机上的当前生产
`hyle_ggml_cpu_avx2_456172ec`，通过同 revision 私有 CPU program table 执行原版 graph，3 threads。
原 GGML base 与 tensor/plan ABI 来自同一 producer；此探针不作为跨机器 loader 或新 ISA 资格。

| 实际输入 | 原版结果 | 结论 |
|---|---|---|
| F32 rank4 activation [2,2,3,7]，物理 weights [5,7] 或 [2,2,5,7] | 60 个输出与独立 double oracle 最大差 0 | 原执行已能承载，当前 rank2 准入是 adapter 限制；尚非普通入口 PASS |
| ReLU NaN/±Inf/±0/有限大值 | NaN→0，其余列出的情况正确 | 需要 NaN 边界，不能直接签等价 |
| SiLU -Inf，有限 -90/-100 | -Inf→NaN，尾值→0；oracle 约 -7.37e-38/-3.72e-42 | -Inf 接缝之外还有尾部资格需查；值很小不自动证明 gate 大倍率后可忽略 |
| GELU-erf -Inf，有限 -5 | NaN；-1.490116e-6，double oracle -1.433258e-6 | F32 公式/尾部误差资格需单列，未接入 |
| GELU-tanh F32 x=3 | 2.99609375，double oracle 2.99636261 | 原版默认 FP16 查表；未改配置，不能签通用精度 |

`raw-probe.log`保留三行输入、四种 unary/三种 separate gate 的特殊分类/有限误差和两种 batch。
探针的 special_mismatches 是诊断计数，不是完整产品 pass/fail gate；未检查全部 dtype/layout/alias。
普通精确域和新验收由[批量卡](../../../docs/implementation/tasks/compute/cpu-ggml-mature-admission-batch.md)持有。
