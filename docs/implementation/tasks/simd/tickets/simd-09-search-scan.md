# SIMD-09：Search 与 scan

- ticket state：`QUEUED`
- kind：product capability/correctness
- depends on：SIMD-08
- budget：1 工程日；性能预算 60 分钟

归属当前 extrema-index 草稿。确认 count/find/value extrema/index extrema/minmax/is-sorted 与 inclusive/
exclusive scan 的类型矩阵、first-tie、empty、NaN、overflow、tail、alias 和 Runtime 版本合同。scan 必须
同时提供 in-place 与 `const input -> output` 路线，exclusive 支持显式 init；custom operation 继续使用标准库
或 expert source，不增加阻断内联的 runtime callback。extrema index 若不能在所有支持 backend 上正确保持
first tie/first NaN，则重写或删除草稿，不能只保留 scalar 假完成。

P1 仅使用 f32 count/argmax 与 u32 scan 的 cache/streaming consumer 行；其余由 property/differential 覆盖。
