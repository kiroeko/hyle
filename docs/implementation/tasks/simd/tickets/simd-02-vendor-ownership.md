# SIMD-02：Vendor ownership 与升级决策

- ticket state：`DONE`
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

## 绑定身份（2026-08-29）

官方 default-branch identity 由只读 `git ls-remote --symref <official-url> HEAD` 验证：

| role | repository | frozen identity | current HYLE relation |
|---|---|---|---|
| portable register/math/bitpack/sort peer | Google Highway | `f503c485cec6b4f2f596fa60d83484797c9eeb5c` | product port origin 仍为 `83eff39371e5e6494b305b253f8a3c06f831faf4` |
| fixed-width transcendental peer | xsimd | `3b3151ff665514a00a437cc985f47b7de99b772b` | product port origin 为独立 snapshot `5d2490f515cf773f984da9ec0a1aec103d00d238` |
| accuracy/mechanism peer | SLEEF | `7623d6cfa2712462880fa63a4d0f0b5f775d1a83` | peer only |
| inference row/task source | llama.cpp | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` | peer only；本地旧审计输入为 `7584430716ee229751771ed0d6bbcb780d105eeb` |
| CPU primitive/task source | ggml | `36da57138425487184aa1da2eee2cde155909c6f` | peer only |
| general CPU primitive peer | oneDNN | `a244fe249f408a8f6ae61ad800e28fa64a8fb074` | peer only |

“最新”只绑定上述日期和 commit，不写浮动 tag/branch。后续升级票若开始日期官方 HEAD 已变化，只能登记
为下一次升级输入；不得在实施中无界追逐 moving target。

## Ownership verdict

| surface | class | semantic owner | mechanism/artifact owner | frozen action |
|---|---|---|---|---|
| ordinary buffer API 与用户合同 | HYLE product | HYLE | HYLE façade + selected vendor | `OWN CONTRACT`；不泄漏 vendor type/macro |
| expert register/kernel surface | migration boundary | HYLE owns routing/usability | Highway preferred | `DIRECT_OR_THIN_ADAPTER`；不默认保留 387-name 平行 DSL |
| Runtime/exact/native/fat implementation | HYLE product | HYLE batch contract | Highway preferred + HYLE glue | `WRAP/DISPLACE`，只有缺口保留 HYLE backend |
| `math_hyle-inl.h` 与 Highway transcendental TUs | integrated source debt | HYLE accuracy/API | Highway origin + HYLE delta | `WRAP_OR_MINIMAL_PORT`；直接使用可行即删除 port |
| bitpack HYLE files | integrated source debt | HYLE stable wire layout | Highway origin + HYLE fixed-layout/u16 delta | `WRAP` 若格式相同；否则仅保留 layout delta |
| `vqsort_hyle/**` 与 target TUs | integrated source debt | HYLE sort/API | Highway VQSort origin + HYLE target/dtype delta | `WRAP_OR_DROP_PORT`；不默认继续手工同步 fork |
| `math_xsimd_hyle-inl.h` 与 xsimd TUs | integrated source debt | HYLE accuracy/API | xsimd origin + HYLE D-tag delta | `DROP_OR_MINIMAL_FALLBACK`；Highway 先资格化 |
| Highway/xsimd/SLEEF benchmark source | peer | upstream | upstream artifact | `PEER`，不进入 installed product |

集成源码虽然没有外部 link/runtime dependency，仍是第三方产品路径，必须保留 revision、local delta、许可证
和内容 hash。HYLE 是 public semantic、选择和 delivery owner；上游是被移植算法的 mechanism origin。

## Upgrade delta 与有限 FIX

- Highway origin→冻结 HEAD 在直接相关文件包含约 471+18 行 `math-inl` 变化、新增 250 行 f16 math、
  bit-pack 26/17、VQSort 49/48 和新的 find/minmax family；SIMD-03 只审这些 origin family，不导入 btree、
  hash、base64 等无关 contrib。
- xsimd port origin 与官方 HEAD 不是可假定线性 fast-forward；SIMD-04 对 common math/trigo 做 snapshot
  语义 diff，保留本地 delta 只能靠当前 HYLE consumer/accuracy 证明。
- 不把普通 HYLE buffer API 改成 Highway API；但 expert D-tag/backend 并非既定保留资产。SIMD-03 必须先
  证明“普通 HYLE façade + expert direct/thin Highway + compiled Highway kernels”的 source/static/shared、
  native/exact/fat、SVE/RVV、编译成本与零额外 hot-loop 开销。证明成立的 surface 在 v2 优先迁移删除。
- SLEEF 保持 peer。只有 xsimd/Highway 两条路线都无法满足冻结 accuracy/resource contract 时才允许另行
  rescope 为产品依赖，不能在 SIMD-04 临时追加。

verdict revision 2：`DONE`。用户明确 vendor-first 是本次重构的主要目的；新 round 必须等待 SIMD-03
隔离 proof 冻结可迁移边界，不能先假设保留自有 backend 再机械刷新 port。本票未修改产品或锁。
output commit：`2953e0598ff7007570de41fb82f968bbfe8f1226`。
