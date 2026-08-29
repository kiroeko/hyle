# SIMD-07：Dtype、narrow 与普通 row ops

- ticket state：`QUEUED`
- kind：product capability/usability
- depends on：SIMD-06
- budget：1 工程日；性能预算 60 分钟

以 Qwen/ggml CPU row 与通用连续 buffer 为输入，完成 f32 与 f16/bf16/e4m3/e5m2 以及常用
storage-to-storage 的无整段临时转换路线、mixed dot、f32/f64 add/subtract/multiply/divide/scale/add-scalar/
affine/sum/abs/negate/square/sqrt。普通入口、native/exact/runtime 必须共享明确 alias、zero/tail、rounding
和资源合同；现有单向 f16→fp8 特例不能继续冒充完整转换图。copy/fill 留给标准库，tensor stride/fusion/
quant scale/block 留给 compute。

P1 只选基础 unary/binary/reduction、conversion 和 mixed-dot 代表行；不为每个 spelling 生成胜负声明。
