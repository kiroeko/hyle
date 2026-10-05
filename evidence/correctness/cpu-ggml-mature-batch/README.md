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

## GB1–3 产品资格与终审

不可变候选 `ef9defea07bbc576a4c6bff1ce0d07e4e53acd19`。Windows 与单向同步的 Linux/GCC14.2
各通过 ggml Provider **48/48**、普通 CPU **12通过/1个旧 fixture 跳过**、普通 Graph **15/15**；跳过不计 PASS。
两个平台的源码、installed static、installed shared 使用同一个独立普通 C++ 消费者，三 dtype 的
Direct ReLU→MatMul 与完整混合广播 Graph region、变化输入重放和真实 completion 均通过。
没有重新运行完整模型权重、GPU、ISA loader 或 Native P1，不能由此签 CPU 全域完成。

首次 Windows 测试误期待外部原位成功，两项收到既有 unsafe-overlap 拒绝；修正仅限测试预期，
产品仍拒绝外部 alias，保留原失败及 append receipt。内部 inplace NaN view 不扩大公共原位能力。
终审核对原数学、下游前 NaN 修复、dtype cast、广播拒绝、线程 chunk、slot/arena 生命周期、
安装消费者独立性及文档边界，GB1–3 未发现未解决问题。七门在此有限范围通过。

P0：ReLU 增加 O(n) 检查和一个 view node，无额外完整 payload；MatMul 只放宽冷准入，
原执行/转换/资源成本不变。无热计时或重新搜 Provider。不是 Native 胜出证据。
实际命令、XML、原始失败、同一消费者、selector selected/skipped、1,581 项单向源码身份和
统一 LOC 在 [归档](qualification.zip)；摘要/hash 在 [收据](qualification-receipt.json)。
Compute + Graph 为 **61,050 实现行 / 248 文件**，不含测试与第三方源码。
状态/归档收口不重建、不跑 P1；重开仅由本轮具名数值、广播、生命周期或交付反例触发。
