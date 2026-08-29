# SIMD-07：Dtype、narrow 与普通 row ops

- ticket state：`QUEUED`
- kind：product capability/usability
- depends on：SIMD-06
- budget：1 工程日；性能预算 60 分钟

以 Qwen/ggml CPU row 与通用连续 buffer 为输入，审查 f16/bf16/fp8 conversion、mixed dot、f32/f64
add/subtract/multiply/divide/scale/add-scalar/sum/abs/negate/square/sqrt。普通入口、native/exact/runtime
必须共享明确 alias、zero/tail、rounding 和资源合同；copy/fill 留给标准库，tensor stride/fusion/quant block
留给 compute。

P1 只选基础 unary/binary/reduction、conversion 和 mixed-dot 代表行；不为每个 spelling 生成胜负声明。
