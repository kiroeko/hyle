# SIMD-04：xsimd/transcendental port 刷新

- ticket state：`QUEUED`
- kind：conditional product
- depends on：SIMD-03
- budget：1 工程日；性能预算 60 分钟

只处理 SIMD-02 冻结的 transcendental backend。先资格化 function/domain/ULP/special-value 合同，再比较
xsimd、SLEEF 或 HYLE/Highway 路线；不以一个输入域的点值决定全函数 backend。输出 upstream/local-delta/
license manifest、受影响 accuracy property、代表性 exp/tanh/erf sentinel 和最终 `WRAP/PORT/DROP` 裁决。
