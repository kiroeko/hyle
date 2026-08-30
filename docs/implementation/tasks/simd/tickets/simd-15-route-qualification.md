# SIMD-15：全部 operation 路由资格化

- ticket state：`DONE`
- kind：correctness / P1 route audit
- budget：2 工程日；串行机器 8 小时
- depends on：SIMD-14

使用现有统一 parity engine，按冻结 implementation-equivalence groups 在 W1-WIN、W1-WSL、M1-MAC 上完成
当前 Highway/xsimd/HYLE 同合同裁决。314 个 source-benched operation 必须被 current report union 或完全相同
codegen group 覆盖；其余 73 项逐项提供结构、composition、selector 或 workload proof。先把所有 material
HYLE/xsimd winner 提取到具名 owner，再把每项 ledger 写为
`KEEP_HYLE/ROUTE_HIGHWAY/ROUTE_XSIMD/REMOVE_REDUNDANT`；零 `PENDING` 才完成。不得在本票删除整套旧 backend。

2026-08-30 owner 收紧了停止规则：Highway 是同合同标准 primitive 的维护性默认值。最终 codegen 等价、
合格 `TIE`、低于 5% material margin、顺序/编译器反转，或冻结的有限实验后仍无法分胜负，均终结为
`ROUTE_HIGHWAY`；不得为了寻找潜在赢家继续增加样本、shape 或 peer。只有已有资格化证据证明某个可表达的
type × ISA/platform scope 上 HYLE/xsimd 明显胜出才保留例外；Highway 不可用或合同不同则保留 HYLE。
逐项 ledger 继续保证能力不漏失，但不再把逐项登记误作逐项论文级性能证明。
有限批次校准失败、没有对应真硅或结果仍不可分时，不继续扩建实验，也不把 HYLE/xsimd 宣称为赢家：只要
三方合同、正确性与 Highway 可用性已经闭合，该 scope 直接采用 Highway。只有这些产品前提本身无法确认时
才进入 `BLOCKED`。

## 当前进度（2026-08-30）

- route ledger 已升级为 `hyle.simd.operation-routing/v2`。冻结 inventory 为 387 operation / 1,797
  definition sites；每个 route scope 现在必须分别给出 HYLE、Highway、xsimd 的合同结论与证据。源码 token
  discovery 不再能够冒充语义等价；terminal verifier 当前按预期拒绝尚未裁决的 `PENDING` scope。
- variant 审计发现旧 inline 正则漏掉返回类型含括号的 23 个真实 definition site。受约束的 additive
  refresh 保持 387 个 operation、零删除，并且只扩充 7 个尚未裁决的 `legacy-all` scope；新 digest 为
  `5e330e8e91fc6d835a75db1298c3bdf82a95cf69b633ea6574a4f234f391b401`。精确勘误见
  `evidence/simd-15-inventory-parser-erratum.json`。
- W1-WIN / MSVC 19.50 / AVX2 的 HYLE↔Highway current report union 已通过 calibration 与 certification，
  机械映射 318/318 个 source-observed operation、350 个 shape rows。receipt：
  `evidence/performance/simd-15-w1-win-msvc-avx2-highway.json`；44 `WIN`、129 `TIE`、1 `LOSE`、
  169 `INCONCLUSIVE`、7 `INFO`。这些是 route 输入，不是 387 项的批量 winner。
- 唯一 `LOSE` 为 `MaskedLoad i32`；最终 object 显示 HYLE 为 full load + compare + and，Highway 为
  `vpmaskmovd`。二者还有 masked-off fault-suppression 合同差异，必须按 AVX2 32/64-bit scope 单独裁决，
  不能把整个 `MaskedLoad` 名字全局切换。
- `MaskedLoad` 已按 W1-WIN/MSVC-19.50/AVX2/i32 拆 scope；新版 xsimd 的
  `batch::load(ptr, runtime_mask)` 已通过全部运行时 mask correctness 检查和同一 replicated parity engine。
  四个等价 anchor 均为 `TIE`；HYLE/xsimd 为 `0.923 [0.887, 0.941]`、`LOSE`。receipt：
  `evidence/performance/simd-15-w1-win-msvc-avx2-xsimd-masked-load.json`。
- 最终 object 中 Highway 与 xsimd 的 masked-load body 指令形状完全相同，且二者都不同于旧 HYLE 的 full
  load + mask body。因此两份 vendor timing 的小差别不再作为算法差异；按 exact-codegen tie 的
  provider-coherence 规则，该 scope 已签为 `ROUTE_HIGHWAY`。该步完成时 388 个 route scope 中 1 个 terminal；
  `MaskedLoad` 的其他 type/target/compiler scope 仍保持 `PENDING`，没有扩大结论或删除旧实现。
- xsimd 全头扫描有 104 个 token-hit operation，其中只有 97 个命中 public API/batch operator surface；二者都
  不是合同 verdict。下一步逐项去除同名异义和 internal-only 命中，unsupported/不同合同者写成具证据
  `NOT_AVAILABLE/NOT_COMPARABLE`，其余才进入同一 parity engine。
- 第一批 alias 红队已证明不能按名字批签：HYLE `Rol/Ror<k>(v)` 是 compile-time per-lane bit rotate，
  Highway 同名入口接受 per-lane runtime shift vector；HYLE `ZipLower/Upper` 是同 lane-width 的
  `InterleaveLower/Upper` 别名，而 Highway `Zip*` 返回 double-width lanes，xsimd AVX2 `zip_*` 又按整个
  batch 的 lower/upper half 组织结果。候选发现器已改用 xsimd 的精确 `rotl/rotr/zip_lo/zip_hi` token，
  但四项 route 仍须按 type/target contract 拆 scope，不能由 token hit 直接终态化。
- 同一轮 source-signature 与消费者扫描已把 6 个无产品消费者的 façade forwarding spelling 签为
  `REMOVE_REDUNDANT/COMPOSITION`：`Rol`、`Ror` 以及四个 `PromoteInRange*To`。证据
  `evidence/simd-15-alias-facade-forwarders.json` 明确记录 HYLE canonical route、Highway 同名异义和
  xsimd adapter/缺失；真正删除仍留给 SIMD-16。
- 剩余 4 个 legacy alias-classified 名称也已逐项终态，但没有错误地批量删除：`ZipLower/ZipUpper` 是
  HYLE 对 `InterleaveLower/Upper` 的真 forwarding alias，签为 `REMOVE_REDUNDANT`；
  `CompressBlocksNot` 是 vqsort 的 128-bit block contract，签为 provider-coherent `KEEP_HYLE`；
  `InterleaveUpperBlocks` 在宽于两个 128-bit block 时不等于 odd-block，签为 `ROUTE_HIGHWAY`。旧的
  全向量 CompressNot 对照测试已替换为 block-uniform mask / defined-prefix oracle，W1-WIN scalar、SSE4、
  AVX2 三个 test artifact 为 3/3 PASS。证据：
  `evidence/simd-15-alias-contract-corrections.json`。
- 21 个 `COMPOSITION` operation 已完成 P0 合同和成本结构审计。12 个有完整同合同 Highway primitive，
  签为 `ROUTE_HIGHWAY`；9 个依赖 trusted-mask、vqsort block 语义或 in-range 专用快路径，签为
  `KEEP_HYLE`。本轮不靠纳秒点值改变 API 或 metadata，逐项公式和 provider 证据见
  `evidence/simd-15-composition-routes.json`。
- 13 个 `SELECTOR` operation 已完成定义点和全仓消费者扫描。它们都是对应 umbrella capability 的
  native/emulated 实现选择器；11 个只由 umbrella 调用，另外两个 `*Emulated` 只多出旧 A/B benchmark
  消费者，没有独立产品需求。全部签为 `REMOVE_REDUNDANT/COMPOSITION`，表示 SIMD-16 应内部化 spelling
  而不是删除仍被 umbrella 选中的机制；证据见 `evidence/simd-15-selector-internalization.json`。
- 26 个 `STRUCTURAL` operation 已完成 type/width/target 与成本结构审计。25 个由 locked Highway 直接
  覆盖，签为 `ROUTE_HIGHWAY`；无产品消费者且与当前 Highway 同名异义的 HYLE
  `InsertIntoUpper(D, hi, v)` 归并为 `Combine + LowerHalf`。审计另记录 HYLE `Blocks` 的 scalable
  runtime-width 缺口和 `ZeroExtendResizeBitCast` 丢 source-tag 的 scalable widening 缺口，均由所选 Highway
  route 覆盖；证据见 `evidence/simd-15-structural-routes.json`。
- 3 个 `ALGORITHM_WORKLOAD` operation 已完成真实合同和 workload 审计。`SafeCopyN/SafeFillN` 保留 HYLE
  的 target-independent clamp，通过 `min(num, Lanes(d))` 适配 locked Highway；W1-WIN/MSVC/AVX2 最终
  object 分别为完全相同的 34/14 条指令形状，签为 `ROUTE_HIGHWAY`。`Stream` 的 256 MiB x86 工作集与
  Highway exact-codegen 打平；Mac M4/NEON 真机则发现 Highway 每个 store 多出的 `prfm pstl1strm` 使其仅
  5.1–12.0 GiB/s，HYLE plain-store 为 32.6–34.9 GiB/s，因此全 target provider-coherent 地签为
  `KEEP_HYLE`。SIMD-16 必须同时补齐已有 `Stream` 的 `FlushStream` 可见性生命周期和对齐/整 cache-line/
  跨核发布文档，不能只留下一个不可安全完成的 store。证据见
  `evidence/simd-15-algorithm-routes.json` 与 `evidence/performance/simd-15-stream-workload.json`。
- 在 owner hard-stop 前，388 个 route scope 中 74 个 terminal、314 个仍为 `PENDING`；10 个 `ALIAS`、
  21 个 `COMPOSITION`、13 个 `SELECTOR`、26 个 `STRUCTURAL` 与 3 个 `ALGORITHM_WORKLOAD`
  operation 已全部拥有逐项结论。qualification receipt 通过 `--apply-qualification` 只更新 ledger 的可变
  route 字段，并在写回前重验冻结 387/1,797 inventory，避免继续手改超大 JSON。
- xsimd 有界审计把 387 项分为 277 `UNAVAILABLE`、97 adapter、9 composition、2 direct 与 2 个经核实的
  rotate adapter；78 个含 SVE/RVV 的待裁决 scope 不由 fixed-width xsimd 完整表达，13 个 fixed/façade
  候选完成合同核对。没有现成 material 证据的 primitive 按 Highway 默认终止，不再追加 xsimd matrix；
  逐项依据见 `evidence/simd-15-xsimd-bounded-screen.json`。
- 两个独立 AVX2 编译器的合格收据只在同一 title 都越过 5% 门时保留 HYLE，得到 28 个 operation 的精确
  MSVC-19.50/GCC-14.2 AVX2 例外。单编译器 winner、跨编译器反转、tie 与 inconclusive 均回到 Highway；
  见 `evidence/simd-15-x86-material-exceptions.json`。
- M1-MAC/AppleClang-21/NEON 严格执行一次冻结批次：parity+widen 的 117 个合格 row 中 5 个精确 HYLE
  material winner；full 的 210 row 因一个 1.00 anchor 越过校准门全部压制，记为 `INCONCLUSIVE`，没有
  重跑或产生例外。证据分别是
  `evidence/performance/simd-15-m1-mac-clang21-neon-highway.json`、
  `evidence/simd-15-neon-material-exceptions.json` 与
  `evidence/performance/simd-15-m1-mac-clang21-neon-full-unqualified.json`。
- 旧 register inventory 之外的 whole-buffer `tanh_f32_batch/gelu_tanh_f32_batch` 使用同一 reporter 重放
  既有四个正反序批次，校准/认证通过；4096 与 1048576 的四个 row 全部证明 xsimd material winner，且
  xsimd 最大 1 ULP、Highway 最大 2 ULP。该具名 family 签为 `ROUTE_XSIMD`，不外推到其他 primitive；见
  `evidence/performance/simd-15-w1-win-msvc-avx2-xsimd-tanh.json`。
- 最终 policy receipt 机械消费全部 314 个 pending operation，拆出 30 个 operation 的 33 个机器可读
  `type × target/compiler` HYLE 例外，并保留默认 remainder。SIMD-15 当时的 ledger 为 421/421 scope
  terminal：355 `ROUTE_HIGHWAY`、44 `KEEP_HYLE`、22 `REMOVE_REDUNDANT`，零 `PENDING`；冻结清单仍是
  387 operation /
  1,797 site。正式写回前的 ledger 快照为
  `C:\hyle-scratch\simd-v3\simd-op-routing-before-final-qualification.json`，SHA-256
  `BF1192C482A407F295543EB4AA7844C424ED02EE90A2BE32137035911EEC6909`；终态依据见
  `evidence/simd-15-route-policy-qualification.json`。

terminal verdict：`DONE`。本票没有删除旧 backend；SIMD-16 只可提取上述有限赢家、生成零成本静态
specialization、迁移真实 consumer，并在逐项引用归零后清算冗余源。

## SIMD-16 ownership correction

迁移 VQSort 后，10 个只为旧 register owner 保留的 composition/facade 已由 locked Highway 同合同机制
直接承接；`Stream` 则从全局 HYLE 保留收紧为仅 AppleClang 21 NEON 的 material 例外，其他 scope 按
tie-to-Highway 策略路由。该变更不重新解释本票的历史裁决，精确依据见
`evidence/simd-16-migration-route-corrections.json`。迁移终态 ledger 因一个有限 `Stream` selector 拆分为
422/422 scope：366 `ROUTE_HIGHWAY`、34 `KEEP_HYLE`、22 `REMOVE_REDUNDANT`、零 `PENDING`。
