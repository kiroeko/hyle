# SIMD-14：多 provider 静态路由底座

- ticket state：`DONE`
- kind：product architecture
- budget：1 工程日；P1 30 分钟
- depends on：SIMD-13 / module `AUDITED`

把 Highway 与 xsimd 都登记为锁定 product header dependencies，建立 target-bound provider adapter 与生成的
compile-time route manifest。ordinary ABI 只见 HYLE pointer/POD；fixed/exact 无 runtime dispatch，fat/runtime
只在 whole-buffer 入口选择一次。用代表性 arithmetic、mask、load/store、math 与 fused row 做 A/A、codegen
和 provider-coherence proof；无法证明零成本的跨 vendor register bridge 禁止进入元素循环。

## 结果

- Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 与 xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b` 已作为独立锁定、校验 archive/tree/license 的
  header-only product candidates 进入中央依赖合同；两者只在 owning implementation/source-expert target
  可见，不进入 installed SDK 的公开依赖或 ABI。
- `scripts/simd-op-routing.json` 的 387/387 operation 现在机械生成 source-only
  `operation_manifest.h`；inventory 数量和 SHA
  `5e330e8e91fc6d835a75db1298c3bdf82a95cf69b633ea6574a4f234f391b401` 不匹配即失败。该 digest 已随
  SIMD-15 的 additive parser erratum 重生成；operation 集和本票的 15/15 零开销 codegen 证明均未变化。
- `route.h` 用 operation/type/target/compiler/version 模板特化选择 provider；primary 永远是 `PENDING`，
  未资格化 route 在编译期失败。调用宏直接展开为 `Candidate<provider>::run(...)`，没有可被 outline 的路由函数，
  也没有循环内函数指针、分支或 vendor-register bridge。
- W1-WIN / MSVC / AVX2 的最终 object code 审计覆盖 HYLE、Highway、xsimd × arithmetic、load/store、mask、
  math、fused-row 共 15 对 direct/routed body：15/15 指令形状相同，额外 instruction/call/stack reference 均为
  0。收据为
  [`simd-14-provider-route-codegen-w1-win-msvc-avx2.json`](../../../../../evidence/performance/simd-14-provider-route-codegen-w1-win-msvc-avx2.json)。
- `hyle_simd_provider_route_test` 与 `hyle_simd_provider_codegen_test` 通过；source parent、relocated shared SDK、
  relocated static SDK 三条 `simd` selected-component journey 均编译并运行通过，私有 provider/native 头未安装，
  Highway/xsimd 许可证随包交付。严格 component/offline dependency contract 通过。

本票的性能门是“编译期路由本身零开销”的结构证明，不用计时噪声代替 codegen 事实，也不裁决 provider
赢家；逐 operation 的语义、正确性和 P1 winner 属于 SIMD-15。terminal verdict：`DONE`。387 个产品 route
仍全部为 `PENDING`，没有旧实现被删除。
