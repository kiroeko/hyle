# R3 CPU：用户批准的具名验收例外

2026-09-16，用户明确选择“记录验收例外，继续 R4（推荐）”。
[终局卡](../../../docs/implementation/tasks/compute/cpu-r3-acceptance-exception.md)限定为
`decode-fused-boundary-1x256x512`：工程动作 ACCEPT_TRADEOFF；原 5% 非劣门仍 INCONCLUSIVE，
严格方向仍 LOSS。没有新增构建、测试或性能采样，不能把批准写成统计 PASS。

- `decision.json`：批准原话、具名 workload、工程动作与保留的统计结论。
- `product-identity.json`：三个 R3 候选组至 `42ae3005` 的产品路径一致性；后续验收状态提交只改文档。
- `loc-current.json`：统一脚本 Compute + Graph 实现 31,034 行，不含测试，self-check PASS。
- `sha256.json`：上述文件身份。

普通验证与四环境三交付沿用 [R3 原包](../compute-cpu-r3-foundation/README.md)和
[Graph 修正包](../graph-cpu-replay-cost/README.md)。性能保留
[Direct 四项](../../performance/compute-cpu-route-p1-v3-d9712e68-w1-win/README.md)与
[Graph 十二项](../../performance/graph-cpu-route-p1-v3-7fa6dc0a-w1-win/README.md)的完整原始证据。
统计不重算，单项例外不扩展为未来路线的性能豁免。
