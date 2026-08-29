# SIMD-05：Register API 与定宽 backend

- ticket state：`QUEUED`
- kind：product review/refactor
- depends on：SIMD-04
- budget：1 工程日；性能预算 60 分钟

审查 `vec.h`、scalar/SSE4/AVX2/AVX-512/NEON backend 的类型、mask、load/store、conversion、shuffle、
gather/compress 和 arithmetic 语义。387 个名字是 inventory，不是 387 个独立需求。缺口必须来自真实
kernel、两个成熟来源或更优复杂度；能自然组合的 convenience 不机械复制。

正确性使用生成/属性矩阵与目标隔离；性能只用同 TU transform、reduction、gather/compress、widen-dot
代表行。任何公共 surface 或 backend 生成策略变化同步架构与维护文档。
