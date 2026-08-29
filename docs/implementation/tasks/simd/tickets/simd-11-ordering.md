# SIMD-11：Ordering、radix 与 top-k

- ticket state：`QUEUED`
- kind：product capability/correctness/performance
- depends on：SIMD-10
- budget：1 工程日；性能预算 60 分钟

归属当前 typed-scratch radix/argsort 草稿。审查 full/partial/top-k/select、stable radix、argsort、整数和
f32/f64 order、signed zero、NaN stable-last、first-equal stability、workspace/overlap、OOM/exception 与
32-bit index boundary。普通 `sort()`/`partial_sort()`/`select()` 必须无隐藏 heap；移除当前 `n>=4096`
自动分配 radix scratch 的 benchmark-derived 策略，或改为调用者显式提供 policy/workspace。small-k streaming
路径以 adversarial distribution 和复杂度资格化，不能只凭随机 benchmark 保留。hot stable route 使用
caller-owned scratch；allocating convenience 不得在 OOM 时静默降为不稳定算法。integrated Highway/VQSort
hash 在最终内容确定后机械重算。

P1 限定 f32 full sort、partial_sort(4096,10)、scratch u32 radix、f32 argsort；不同稳定性/NaN 合同不比较。
