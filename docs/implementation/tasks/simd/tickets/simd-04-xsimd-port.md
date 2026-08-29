# SIMD-04：数学 vendor 合并与 port 消减

- ticket state：`ACTIVE`
- kind：conditional product
- depends on：SIMD-03
- budget：1 工程日；性能预算 60 分钟

- input commit：`9d006769b6ccf3ff67291bab1ad6d4a43b446bfb`

以 SIMD-03 的 Highway 直接承载结果为默认，先资格化 function/domain/ULP/special-value 合同，再确认
xsimd 或 SLEEF 是否仍填补真实缺口。能由 Highway 等价承载的 xsimd/Highway checked-in math port 删除；
只有明确 accuracy/capability gap 才保留最小 fallback。输出 upstream/local-delta/license manifest、受影响
accuracy property、代表性 exp/tanh/erf sentinel 和最终 `DIRECT/DROP/MINIMAL_PORT` 裁决；不以一个输入域
的点值决定全函数 backend。
