# SIMD-05：Register API 与定宽 backend 迁移

- ticket state：`READY`
- kind：product review/refactor
- depends on：SIMD-04
- budget：1 工程日；性能预算 60 分钟

依据 SIMD-03 proof，将 `vec.h`、scalar/SSE4/AVX2/AVX-512/NEON backend 中可由 Highway 直接承载的
类型、mask、load/store、conversion、shuffle、gather/compress 和 arithmetic 路径迁移或删除。387 个名字
是迁移 inventory，不是必须保留的 387 个 HYLE spellings；普通用户面保持 HYLE buffer 合同，expert 面
优先 direct Highway 或薄 adapter。只有真实 HYLE semantic/consumer gap 才保留自实现。

正确性使用生成/属性矩阵与目标隔离；性能只用同 TU transform、reduction、gather/compress、widen-dot
代表行。任何公共 surface 或 backend 生成策略变化同步架构与维护文档。
