# SIMD-03：Highway integrated port 刷新

- ticket state：`QUEUED`
- kind：conditional product
- depends on：SIMD-02 / current round `AUDITED`
- budget：1 工程日；性能预算 60 分钟

只执行 SIMD-02 冻结的 Highway `PORT/WRAP/DROP` 集。保持 HYLE 公共类型不泄漏 Highway 类型；逐路径
记录 upstream revision、local delta 和许可证，机械重算 content hash。正确性覆盖受影响 ISA/tail/alias；
性能只跑 register、bitpack、ordering 中受影响的预选 sentinel，不跑完整 344-op 表。

停止：升级造成语义不兼容、无法解释的 material regression 或超过一次机制诊断时 `BLOCKED`，不得靠
新增 API/layout/常态 metadata 迎合 benchmark。
