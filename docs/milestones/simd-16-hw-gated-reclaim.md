# simd-16：翻案硬件门控项（RVV Zvbb/Zvbc + fp8）

| 项 | 内容 |
|---|---|
| 状态 | ✅ 完成（2026-07-11，grok 实现 / opus 四查通过） |
| 前置 | simd-15（P5） |
| 起因 | 2026-07-11 qemu 探针推翻"无硬件不做"——Zvbb/Zvbc qemu 可真跑（`feedback_infra_capability_completeness`） |
| 范围 | `backend/rvv.h` + `rvv_test.cc` + 新 qemu tier；fp8 走 SDE（后议，见 §4） |

## 0. 协作分工 + 这份卡怎么用

- **grok（实现方）**：按 §2 逐 op 卡填 `rvv.h`，加测试，跑新 qemu tier。intrinsic 全名、坑、验收命令都已钉死，**不需要查名，照抄卡里的全名**。
- **opus（校验方）**：完工后四查（编译过 / 多 VLEN 全绿 / 对拍 scalar oracle / 手法无降级）+ 复核门控翻案是否正确。

> ⚠️ **opus 已探明的工具链事实（grok 直接采信，别重新试名）**：
> - GCC-14 `riscv64-linux-gnu-g++-14` 的 `riscv_vector.h` 里，Zvbb per-lane intrinsic **用全名**（`-E` 预处理搜不到，是 pragma 生成，但全名编得过）。
> - `vclz_v_u{8,16,32,64}m1` / `vctz_v_*` / `vbrev_v_*` / `vror_vx_*` / `vrol_vx_*` **全部可用**（已逐个编译确认）。
> - ❌ **per-lane popcount 在 GCC-14 这版头里不存在**（`vcpop` 短名被 mask-popcount 占用，`vcpopv`/`vcpop_v_*` 都编不过）。→ **PopulationCount 本轮 RVV 不做**，保持 Zvbb-gated 登记，等工具链更新。这不是找借口，是逐个穷举拼写坐实的工具链缺口。
> - `vclmul_vv_u64m1` / `vclmulh_vv_u64m1`（Zvbc）**可用**。
> - qemu 参数：Zvbb 用 `-cpu rv64,v=true,zvbb=true,vlen=<bits>`；Zvbc 用 `zvbc=true`。

## 1. 本轮范围（实测可做可验的翻案项）

| op | intrinsic（全名，已验证） | oracle | 门控翻案理由 |
|---|---|---|---|
| `LeadingZeroCount` | `vclz_v_u{8,16,32,64}m1` | scalar.h `LeadingZeroCount`（std::countl_zero） | Zvbb qemu 实测跑通 |
| `TrailingZeroCount` | `vctz_v_u{8,16,32,64}m1` | scalar.h（std::countr_zero） | **P5 时误标"不做",实为软件面唯一剩项** |
| `ReverseBits` | `vbrev_v_u{8,16,32,64}m1` | scalar.h `ReverseBits` | Zvbb 原生 |
| `RotateRight<k>` / `RotateLeft<k>` | `vror_vx_u*` / `vrol_vx_u*` | 已有泛型 RotateRight（shift-or 合成）作对拍 | Zvbb 原生比 shift-or 省指令 |
| ~~`PopulationCount`~~ | — | — | ❌ **不做**：GCC-14 头无 per-lane vcpop intrinsic（穷举坐实）。保持登记 |
| `CLMulLower` / `CLMulUpper` | `vclmul_vv_u64m1` / `vclmulh_vv_u64m1` | scalar.h `CLMulLower/Upper`（detail::clmul64） | Zvbc qemu 实测跑通 |

**op 数：5（clz/ctz/brev/rotate×2 记 1）+ 2（clmul）= 净新增约 6-7 个 RVV op**。TZCnt 补上后主闭集软件面才真 100%。

## 2. 逐 op 施工卡

### 前置：新建 Zvbb/Zvbc tier（gate 用）

现有 RVV op 在 base `rv64gcv`。Zvbb/Zvbc op 需要更高 `-march`，且**不能污染 base 变体**（base qemu 不带 zvbb 会 SIGILL）。做法：

- 这些 op 在 `rvv.h` 里用 `#if defined(__riscv_zvbb)` / `#if defined(__riscv_zvbc)` 门控（编译器在 `-march=..._zvbb` 时定义这些宏）。base 变体不编入 → 不破坏现有 `rv64gcv` tier。
- 测试：`rvv_test.cc` 里对应 TEST 也用同样 `#if defined(__riscv_zvbb)` 包裹，base 跑时整个 TEST 编译期消失。
- **grok 先确认宏名**：`echo | riscv64-linux-gnu-g++-14 -march=rv64gcv_zvbb -dM -E - | grep zvbb` 看 `__riscv_zvbb` 的确切拼写（可能是 `__riscv_zvbb` 值为版本号）。

### C1. LeadingZeroCount / TrailingZeroCount / ReverseBits（Zvbb，一组）
- **签名**：`template<class V> V LeadingZeroCount(V v)` 等，integer-lane（加 `static_assert(std::is_integral_v<TFromV<V>>)`）。
- **实现**：按 `sizeof(T)` 四分支调 `vclz_v_u{8,16,32,64}m1(bits, vl)`。输入若 signed 先 BitCast 到 unsigned（intrinsic 是 u 版），结果 BitCast 回 T。参照 rvv.h 已有 `PopulationCount` 被 defer 的注释位置，替换成真实现并去掉 `#if` 外的 defer 注释。
- **vl**：用 `detail::vlmax<TFromV<V>>()`（同 rvv.h 其它 elementwise op）。
- **坑**：`vctz(0)` = lane 位宽（scalar countr_zero(0)=位宽，一致）；`vclz(0)` = 位宽。对拍时 0 输入要覆盖。
- **门控**：整段 `#if defined(__riscv_zvbb)`。

### C2. RotateRight\<k\> / RotateLeft\<k\>（Zvbb，覆盖已有 shift-or 合成）
- **现状**：rvv.h 已有 `RotateRight<kBits>`（shift-or 合成，base 可用）。Zvbb 版是**性能升级**（一条 vror 替 shift-or-or）。
- **决策**：用 `#if defined(__riscv_zvbb)` 提供 `vror_vx`/`vrol_vx` 原生版，`#else` 保留现有 shift-or。**两版都要过同一测试**（已有 ScalableGenericP0 里测了 Rol/Ror）。
- **签名**：保持 `template<int kBits, class V>`，`vror_vx_u*(v, kBits, vl)`。注意 vror 是"右旋 k"，vrol 是"左旋 k"，与 HYLE Ror/Rol 语义直接对应。

### C3. CLMulLower / CLMulUpper（Zvbc）
- **签名**：`template<class V> V CLMulLower(V a, V b)`，u64 lane（`static_assert(sizeof(TFromV<V>)==8)`）。oracle scalar.h:1467。
- **语义**：CLMulLower = 每 128-bit pair 的低 64 位无进位乘 → 128 位积。**但 RVV vclmul_vv 是逐 64-bit lane 的 carryless mul 低 64 位，vclmulh 是高 64 位**。HYLE 定宽 CLMulLower 语义是"取每 128-bit block 的低 64 位 lane 相乘得 128 位积放回该 block"。
- **⚠️ 语义对齐关键（grok 必读，opus 会重点校验这条）**：HYLE/x86 的 CLMul 在 128-bit block 内操作，RVV 是逐 lane。要先读 scalar.h:1467-1481 的确切语义（它对 u64[2] 输入 `clmul64(av[0],bv[0])` 产出 o[2]=128位积），再决定 RVV 怎么拼：vclmul 出低 64、vclmulh 出高 64，两者交织成 128 位结果。**这条不是直通，是本轮唯一有语义编排的 op**——若拿不准，标记出来等 opus 校验，别硬猜。
- **门控**：`#if defined(__riscv_zvbc)`。

## 3. 验收

新增两条 qemu 命令（grok 加进 scripts/rvv-qemu-test.sh 或单独脚本）：
```bash
# Zvbb tier
riscv64-linux-gnu-g++-14 -std=c++23 -O2 -march=rv64gcv_zvbb -DHYLE_SIMD_TARGET_RVV ... rvv_test.cc
qemu-riscv64-static -cpu rv64,v=true,zvbb=true,vlen=128/256/512/1024 ...
# Zvbc tier
-march=rv64gcv_zvbc + -cpu ...zvbc=true
```
- 每个新 op 在 `rvv_test.cc` 加对拍 scalar oracle 的用例，`#if defined(__riscv_zvbb/zvbc)` 包裹。
- **base rv64gcv tier 必须仍全绿**（门控 op 编译期消失，不影响现有 14+ tests）。
- 四 VLEN（128/256/512/1024）全过。

### grok 实测（2026-07-11）

`bash scripts/rvv-qemu-test.sh all`：

| tier | march | tests | VLEN×4 |
|---|---|---|---|
| base | rv64gcv | 15 | ✅ |
| zvbb | rv64gcv_zvbb | 16（+`Simd16Zvbb`） | ✅ |
| zvbc | rv64gcv_zvbc | 16（+`Simd16ZvbcClmul`） | ✅ |

CLMul 编排：`DupEven/DupOdd` + `vclmul`/`vclmulh` + `OddEven(hi,lo)` → 128-bit block `[lo,hi]`，对拍软件 clmul64（与 `detail::clmul64` 同算法）。
PopulationCount 未做（工具链缺口）。宏名 `__riscv_zvbb`/`__riscv_zvbc`=1000000。

### opus 四查（2026-07-11）

| 查项 | 结果 |
|---|---|
| ①编译过 | ✅ base/zvbb/zvbc 真 qemu 全编过 |
| ②多 VLEN 全绿 | ✅ 128/256/512/1024 × 三档 |
| ③对拍 oracle | ✅ clmul64_soft ≡ crypto_soft.h:156（已被 x86 PCLMULQDQ 验证）；clz/ctz/brev 对 std |
| ④手法无降级 | ✅ objdump：zvbb 见 vclz/vctz/vbrev/vror/vrol；zvbc 见 vclmul/vclmulh；base 0 条 |

CLMul 语义编排手工推演正确（DupEven→vclmul/h→OddEven = 128-bit block）。

## 4. fp8 AVX10.2（本轮不含，单列）

fp8 硬件 convert（E4M3/E5M2）走 Intel SDE `-future`（同 AVX512 SDE 先例，正确性可信/性能不可信）。这是 x86 路径，与 RVV 本轮正交，且 SDE AVX10.2 支持需先探针确认。**建议 simd-16 只做 RVV Zvbb/Zvbc，fp8 留 simd-17**，避免一轮跨两个 ISA + 两个模拟器。

## 5. 明确不做（本轮翻案后仍门控）

| 项 | 理由 |
|---|---|
| RVV `PopulationCount` | ~~GCC-14 头无 per-lane vcpop~~ → **已翻案做掉(2026-07-12 b0ad404)**:inline-asm `vcpop.v`,qemu 8.2.2 就正确执行(本条当时的"no-op"结论是 vmv.x.s+exit-code harness bug) |
| SVE2 `AES*`/`CLMul` | ~~qemu Property not found 只能编译验~~ → **已翻案做掉(2026-07-12 8c466f8)**:`-cpu max`(非 `sve2-aes=on`)qemu 8.2.2 就真跑,8 op native svaese/svpmull 全 4 VL 验 |
| scalable ASan | qemu-user + `-fsanitize=address` shadow-memory 段错误（实测），真边界 |
| mask-bits 大 VL | VL 2048-bit i8=256 lane 超 uint64 位域，语义受限 |

## 6. clay 覆盖

clay 无可移植 vector API，本轮是 RVV 后端对齐 Zvbb/Zvbc 硬件能力的净新增，不改 clay 结论。
