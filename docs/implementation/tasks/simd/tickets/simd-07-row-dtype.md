# SIMD-07：Dtype、narrow 与普通 row ops

- ticket state：`ACTIVE / AUDITED`
- kind：product capability/usability
- depends on：SIMD-06
- budget：1 工程日；性能预算 60 分钟
- input commit：`22d000f809e1690958d446d0a4ca5a1dba82f544`

以 Qwen/ggml CPU row 与通用连续 buffer 为输入，完成 f32 与 f16/bf16/e4m3/e5m2 以及常用
storage-to-storage 的无整段临时转换路线、mixed dot、f32/f64 add/subtract/multiply/divide/scale/add-scalar/
affine/sum/abs/negate/square/sqrt。普通入口、native/exact/runtime 必须共享明确 alias、zero/tail、rounding
和资源合同；现有单向 f16→fp8 特例不能继续冒充完整转换图。copy/fill 留给标准库，tensor stride/fusion/
quant scale/block 留给 compute。

P1 只选基础 unary/binary/reduction、conversion 和 mixed-dot 代表行；不为每个 spelling 生成胜负声明。

## 资格化审计

- 真实 Compute consumer 当前只调用 `add/dot/saxpy`；但锁定的 llama.cpp/ggml CPU row 路线实际消费
  add/add-scalar/subtract/multiply/divide/scale/square/sqrt/abs/negate/sum，并在归一化、attention、loss 与
  backward 中组合使用。这证明普通连续 row algebra 是推理基础能力，而不是 benchmark 造出的函数表。
- `ops.h` 只有 add/dot/saxpy；`dtype_ops.h` 只有 f32 与四种 storage 的双向转换，加两个 f16→fp8
  特例。在 f32/f16/bf16/e4m3/e5m2 五种公开 storage 间，共 20 条不同类型的有向边，当前缺 10 条；
  窄类型都可无损提升到 f32，因此缺边可在一次遍历中完成，不需要整段中间数组，也不会引入二次舍入。
- 锁定 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 已拥有 fixed/scalable target、f16/bf16
  lane、Promote/Demote、普通 row arithmetic/reduction 及 f32/bf16 dot。锁定 xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b` 没有提供更完整的 fp8 或本票所需的产品机制，故继续只作
  benchmark peer；不得为了“同时支持两个 vendor”给热循环再套一层运行时抽象。
- 两个 peer 都不拥有 HYLE 的 OCP fp8 value/rounding/NaN/saturation 合同，也不覆盖 f16 扩展为 f32
  累加及 i8/i16 modulo-2^64 dot；这些是最小 HYLE semantic/mechanism gap，不得因 vendor-first 删除。
- `native/ops.h`、`narrow_float.h` 与 `narrow_algorithm.h` 的公开 D-tag/register surface 没有仓库外语法
  consumer，仓库内产品仅由本模块旧 kernel 间接消费；它们复制 Highway 寄存器 DSL，与 SIMD-05 已冻结的
  direct-Highway source expert 路线冲突。普通 pointer/count narrow semantics 有独立价值，仍由
  `narrow_ops.h` 保留。
- 现有 dtype 文档仍称实现刻意避开 Highway，用户指南后半又推荐迁移期 D-tag，均与 SIMD-05/06 的
  vendor-first 决策矛盾；`mixed_dot_bench` 还依赖已私有化的 Runtime table，不能直接充当本票性能证据。

## 冻结合同与有限 FIX 集

1. 普通 C++ row 面补齐 f32/f64 `subtract/multiply/divide/scale/add_scalar/affine/sum/abs/negate/
   square/sqrt`，保留 add/dot/saxpy。`n == 0` 可传空指针；输出可与任一完整输入精确同址，其他部分重叠
   未定义；O(n)、O(1) extra storage、无 heap/lock/整段 copy。reduction 的结合顺序可随 target 变化，
   saxpy/affine 允许 FMA 导致最后舍入不同。copy/fill 留给标准库，stride/fusion/quant block 留给 Compute。
2. dtype 公共面收敛为可发现的 `convert(src, dst, n)` overload graph，覆盖五种 storage 间全部 20 条
   不同类型有向边；同类型复制留给标准库。相同宽度允许 exact alias，不同宽度要求 disjoint，部分重叠
   未定义；保持既有 RNE、saturation、NaN 与无整段临时合同。旧的十个方向命名不是兼容合同，项目尚未
   发布，本票直接移除而不保留两套 spelling。
3. ops 与 narrow 的 generated target TU 改用 `hyle_add_dispatched_kernel(HIGHWAY ...)`。固定目标必须由
   宏选定唯一 `HWY_STATIC_TARGET` 并让模板 body 在同一 TU 单态化；source-fat 与普通预编译入口每个整段
   buffer 最多一次函数指针选择，元素循环内不得有 HYLE wrapper、callback 或第二次 Highway dispatch。
   f32/f64 与可承载的 f16/bf16 路线直接用 Highway；fp8、f16 widen-dot、整数 modulo dot 和精确 feature
   specialization 只保留最小 HYLE body。xsimd 没有已证明的能力/性能胜出前不进入产品依赖。
4. 删除 installed `native/ops.h`、`narrow_float.h`、`narrow_algorithm.h` 及其 umbrella/export/test/bench
   依赖；source expert 直接 include Highway。保留普通 `narrow_ops.h`，其 numeric/bit equality、find、sum
   与 IEEE totalOrder body 改为 direct Highway/raw-bit composition，不改变公共语义。
5. 同步 runtime private table、exact symbols、source/static/shared component、测试和用户/架构文档。P1 只
   测代表性 4096/streaming row：row add/multiply/affine/sum、f16/bf16 conversion 与同合同 mixed dot；
   candidate 与 direct Highway/成熟 peer 必须同 TU、同 ISA、同 alias/rounding/dispatch/resource 计费，
   正反顺序且通过 A/A。5% 是 material margin；亚纳秒点值、地址/顺序翻转不授权改 API/layout/metadata。

本票停止线是上述五项及 Windows/WSL/macOS、适用 SVE/RVV correctness 与三种交付门完成。activation、
search/scan、bitpack/random、ordering 和 legacy backend 总删除分别属于 SIMD-08～12；不得在本票扩张。
