# SIMD-02：Vendor ownership 与升级决策

- ticket state：`QUEUED`
- kind：architecture/capability audit
- depends on：SIMD-01
- budget：4 小时；性能预算 0

## 范围

绑定 Highway、xsimd、SLEEF 及 inference peer 的当前官方 identity；对每个 integrated subtree 给出
`OWN/WRAP/PORT/DROP`，列出上游差异、HYLE 修改、许可证、支持 ISA、accuracy/dispatch/layout 要求与
升级触发器。最多为一个 capability 选两个 strong peer。

## 验收

- vendor source-to-path manifest 和直接/间接产品依赖六问；
- 不以“某 benchmark 赢”选择逐操作 backend；
- 冻结 SIMD-03/04 的有限升级集合，允许结论为保持现状或移除 port；
- 更新后的 capability ledger 足以让新 closure round 进入 `AUDITED`；
- 不写产品代码、不修改依赖锁内容。
