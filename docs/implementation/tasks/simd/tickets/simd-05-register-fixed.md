# SIMD-05：Register API 与定宽 backend 迁移

- ticket state：`ACTIVE`
- kind：product review/refactor
- depends on：SIMD-04
- budget：1 工程日；性能预算 60 分钟
- input commit：`a7a62e54b9f2fe021ebaf5776837cc1af186be76`

依据 SIMD-03 proof，将 `vec.h`、scalar/SSE4/AVX2/AVX-512/NEON backend 中可由 Highway 直接承载的
类型、mask、load/store、conversion、shuffle、gather/compress 和 arithmetic 路径迁移或删除。387 个名字
是迁移 inventory，不是必须保留的 387 个 HYLE spellings；普通用户面保持 HYLE buffer 合同，expert 面
优先 direct Highway 或薄 adapter。只有真实 HYLE semantic/consumer gap 才保留自实现。

正确性使用生成/属性矩阵与目标隔离；性能只用同 TU transform、reduction、gather/compress、widen-dot
代表行。任何公共 surface 或 backend 生成策略变化同步架构与维护文档。

## 资格化审计

- 当前 `vec.h` 加 scalar/SSE4/AVX2/AVX-512/NEON/SVE/RVV backend 共 24,084 行；机械抽取为
  387 个去重 operation 名。锁定 Highway `f503c485...` 源树中有 364 个同名 operation；其余 23 个
  主要是 HYLE 内部 selector、Newton/bitwise-reduction/complex 等可组合 gap。同名搜索不是语义证明，
  但已经否定“需要继续维护七套平行 backend 才能获得能力面”的前提。
- 仓库内 `simd` 之外只有 container 的 bitset/string 两个 build-private kernel body 直接消费 `vec.h`；
  其余直接消费者是 SIMD 自己的 family body、测试/benchmark 与旧示例。普通已安装用户使用 pointer/
  count batch API，并不需要寄存器类型。
- 已安装 SDK 不能无意把 vendor header/type 变成稳定 HYLE ABI；首席 source consumer 则可以安全地在
  明确 target TU 中直接使用中央锁定 Highway。这两种 journey 必须分开，不能用 387 个 HYLE spelling
  伪装成“更稳定”。

## 冻结裁决与有限 FIX 集

1. `hyle_target_simd()` 变成 source-only expert seam：绑定锁定 Highway include、单目标编译策略与 HYLE
   ISA flags；经典 cl.exe 的 Highway AVX3 blocklist 必须显式报错而非静默冒充 AVX-512。
2. `samples/02_simd` 的 expert kernel 改为直接 Highway；普通 runtime batch 路径仍只见 HYLE API。
   installed static/shared sample 不承诺寄存器层，源码嵌入 sample 才允许打开 expert ISA。
3. `vec.h`/`native/*` 从“推荐产品 API”降为迁移期 private legacy。现有 SIMD family 与 container
   build-private seam 在各自后续 ticket 迁走前仍保留并持续编译；禁止为维持旧 spelling 新写 387 层
   wrapper。SIMD-07～11 迁移各 family，SIMD-12 删除无消费者的 legacy backend。
4. Highway 不存在的 23 个名字不是自动保留清单：内部 selector 删除；只有后续真实 consumer 需要的
   semantic composition 才在所属 family 保留最小 HYLE 实现。
5. 本票验证 Windows/WSL source expert 的 AVX2 与 macOS NEON 编译/运行；性能只重放 transform、
   reduction、gather/compress、widen-dot 五个同 TU 同 ISA 代表机制。小于 5% 或顺序翻转不驱动 API。

本票不迁移 dtype、activation、search/scan、bitpack/random、ordering 的产品 body，也不修改公共 closure/
性能仪器。它们已有顺序 owner；本票的停止线是 expert seam 与迁移边界成立，而不是提前清空全部 24k 行。
