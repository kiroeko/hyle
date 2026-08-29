# SIMD-07：Dtype、narrow 与普通 row ops

- ticket state：`DONE`
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

## 实现与验证结果

- 普通 C++ row API 已补齐 f32/f64 add/subtract/multiply/divide/scale/add-scalar/affine/sum/abs/negate/
  square/sqrt，并保留 dot/saxpy。五种公开 storage 间的 20 条不同类型有向转换统一为可发现的
  `convert(src, dst, n)` overload；alias、zero、tail、RNE、saturation 与 NaN 合同由公共测试锁定。
- common ops、dtype 和 narrow target TU 直接实例化锁定 Highway fixed/scalable templates；只保留 fp8、
  f16 widen-dot、整数 modulo dot、F16C/BF16/VNNI 等具名 HYLE semantic/mechanism gap。重复 Highway
  register DSL 的 installed `native/ops.h`、`narrow_float.h`、`narrow_algorithm.h` 已删除。
- 固定 ISA/source expert 路线没有运行时分派；ordinary/fat 路线每个整段 buffer 只做一次缓存选择和一次
  间接调用。`HWY_HEADER_ONLY` 避免引入第二个 Highway runtime ABI；static export 不再泄漏内部
  `hyle_simd_source_expert` target。
- Windows MSVC 19.50：fat shared SIMD 387/388 通过（1 个合法 AVX512BF16 skip），SSE4 301/301、AVX2
  335/335、SSE4 AES 232/232；scalar/SSE4/AVX2 exact 的本票 filter 各 40/41 通过加 1 个合法 skip，
  exact artifact 检查全部通过。
- WSL GCC 14：Release/WERROR 的 fat、SSE4、AVX2、AES 结果与 Windows 对应门一致；exact-scalar ASan
  本票 filter 40/41 通过加 1 个合法 skip且无 sanitizer finding。fat ASan 全 SIMD 因 SIMD-11 的旧
  AVX512 VQSort object 编译超过十分钟而按票边界停止，未伪写 PASS，也不构成本票产品缺陷。
- 同一 scalable binary 在 QEMU SVE VL 128/256/384/512/2048 与 RVV VLEN 128/256/512/1024 的本票
  43 项 filter 均为 42 pass 加 1 个合法 x86-only skip。macOS M4 AppleClang 21 的 ordinary SIMD
  378/378、NEON 304/304 通过；三平台 source、relocated shared、relocated static selected-component
  consumer 均通过。

## 有界性能 verdict

本票签 `PASS / P1_ENGINEERING / 5% / NO MATERIAL LOSE`，不签“整个 SIMD SOTA”。W1-WIN、MSVC
19.50、AVX2 的同 TU candidate/direct-Highway 对照通过 A/A、反序、双 layout 与 certification：工具对
含 calibration anchor 的 15 行报告 7 `TIE`、8 `INCONCLUSIVE`、0 `LOSE`，paired center 均约在
0–2%，没有 5% material regression。add/multiply 两边的测量主体各有 126 条完全相同的指令序列且无
helper call；固定公开入口只尾调用 target body，ordinary/fat 只有一次 buffer-boundary 间接调用。
完整合同、原始日志、hash 与未签范围见
[SIMD-07 performance evidence](../../../evidence/performance/simd-07-row-dtype.md)。

## 终止 receipt

- product output commit：`63a1e614c9a137c140216d1f744b9e0a7074edad`；48 个文件，2,599 additions、
  3,030 deletions。
- 四项 LOC 由该 immutable commit 以 `cloc 2.08`、
  `python scripts/loc-report.py --module simd --commit 63a1e614 --self-check --json` 复算且
  `self-check=pass`：全库代码 **310,043**；自实现包含测试 **120,715**；自实现不包含测试
  **79,229**；当前 SIMD 实现 **33,546** 行。
- known limits：scalable ISA 未签真硅性能；classic cl.exe direct Highway AVX-512 仍受上游 blocklist
  限制；tensor stride/fusion/quant 属于 Compute；activation/search/scan/bitpack/random/ordering 与
  legacy 总删除按 SIMD-08～12 继续。8 行 `INCONCLUSIVE` 只表示 exact tie/win 未裁决，不能写成胜利。
- reopen：公开 row/dtype 合同出现可复现正确性缺陷，热循环出现第二次 dispatch/整段临时转换，三种交付
  失效，或同合同真实 consumer 在资格化真机上复现超过 5% 的 material regression。
