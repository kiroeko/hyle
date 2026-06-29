# simd-03：ARM64 NEON + AVX-512 后端（写+编+objdump 验，未跑存疑）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | ✅ NEON + AVX512 均完成（写+编+objdump 指令验）；**两者均未在目标硬件实跑，存疑待真验**（无 ARM 机/qemu、无 AVX512 机/SDE） |

## 前置依赖

- 硬前置：`simd-01`（vec D-tag 契约）+ `simd-02`（后端结构 + op 契约模板）。
- 决策依据：[`simd-design.md`](../architecture/simd-design.md) 落地清单 #2 + AVX512 延期理由；`reference_wsl_linux_tsan.md`（aarch64 交叉编译验证法）。

## 目的

补齐 simd 的 **ARM 半**：aarch64 NEON 后端，让同一份 D-tag 源码在 ARM 上也吃满 128 位向量。AVX512 原在本里程碑，但因不可真跑 + mask 模型分歧**理由化延期**（见下）。

## 具体事务

- [x] `include/hyle/simd/backend/neon.h`：128 位 NEON 后端，lane f32/f64/i32/u32，op 全集匹配 scalar/sse4/avx2 契约。intrinsic 对照 Highway arm_neon-inl.h：`vdupq/vld1q/vst1q`、`vaddq/vmulq/vdivq(aarch64)/vminq/vmaxq`、`vfmaq`(FMA)/`vfmsq`、`vbicq`(AndNot 需 swap 成 `~a&b`)、`vceqq/vcgtq/...`(返 mask 向量)、`vbslq`(IfThenElse)、`vaddvq/vminvq/vmaxvq`(横向 reduce)、mask pack `{1,2,4,8}+vaddvq`+popcnt/ctz。mask=全1/全0 unsigned 向量(同 SSE/AVX2 契约)。
- [x] vec.h 的 `HYLE_SIMD_TARGET_NEON` ladder hook 已就位（simd-01 留），无需改。
- [x] **AVX512 延期登记**（simd-design 落地清单末 + 本文档）。

## 验收标准

- **编译**：✅ `aarch64-linux-gnu-g++-14 -std=c++23 -O3 -c` 干净（零 error/warning）。**真实捕到 bug**：`typename RawNeonT<T>`（别名模板 id 前误加 typename）交叉编译即报错并修掉——证明交叉编译是真验证非走过场。
- **指令验证（objdump）**：✅ `aarch64-linux-gnu-objdump -d` 确认真发 NEON：dot kernel → `fmla v31.4s`(FMA)+`faddp v31.4s`(横向和)；relu(Gt+IfThenElse) → `fcmgt v30.4s,#0.0`；int sum → `addv s31,v30.4s`(横向整数和)，全在 `v` 寄存器。
- **未跑（诚实边界）**：无 ARM 机 / 无 qemu-aarch64,**不能执行**。等同 container-04b GroupNeon 口径："写+编过+指令验,仅未跑"。NEON 结构与 SSE/AVX2 同构,且 vec_test 是宽度无关的——将来在 aarch64 机/qemu 上用 `-DHYLE_SIMD_TARGET_NEON` 编 vec_test 即可直接跑对拍。**登记待真跑。**
- **x64 无影响**：neon.h 仅在 `HYLE_SIMD_TARGET_NEON` 下 include,x64 构建根本不编它——全量 ctest 不受影响。

## AVX-512 后端（已实现，写+编+objdump 验，未跑存疑）

> **纠"延期"误判（用户纠偏）**：本里程碑一度把 AVX512 延期,理由①不可真跑②mask 需 vec.h 改造。**用户指出:没机器不该不做,平台优先不知道写没写对,应实现后存疑——否则永远落不了地。** 照做后第②条被证伪。详见 [`simd-design.md` "AVX512 后端：已实现"]。

- [x] `include/hyle/simd/backend/avx512.h`：512 位 AVX-512(F+BW+VL+DQ),lane f32/f64/i32/u32(16/8/16 lane),op 全集。**自定义 `Mask512` 包 `__mmask16/8`（k 寄存器）**——比较直接写 k(`_mm512_cmp_*_mask`/`vpcmp*`)、`IfThenElse`→`_mm512_mask_blend_*`、`CountTrue`→k 本就是位掩码 popcnt。unsigned 比较有原生 epu32 mask(无需 SSE/AVX2 的 xor-MSB)。
- **验证**：`g++-14 -mavx512f -mavx512bw -mavx512vl -mavx512dq -c` 干净(任何 x64 可编)；objdump 实证真发 zmm/k-reg——dot→`vfmadd231ps %zmm`+`vextractf64x4`;relu→`vcmpgt_oqps → %k1`+`vmovaps %zmm{%k1}{z}`(k 掩码路径通);unsigned count→`vpcmpnleud → %k0`+`kmovw`+`popcnt`。
- **设计结论(重要)**：写 AVX512 **证明了 vec.h 的 per-backend mask 抽象够用**——`Mask` 本就每后端自定义(scalar=bool[]/SSE=向量/AVX512=k-reg),op 签名不变,vec.h 和测试零改动。先前"需 mask 改造"是误判。
- **存疑(诚实)**：未在 AVX512 硬件/SDE **执行**,正确性=指令级验证 + 与低 ISA 同构推断,非实跑对拍。dispatch 仍把 AVX512 主机降级跑 AVX2(已验证安全),avx512.h 是"已就绪待启用",**未接进 ops/dtype 运行期 selected_tier**——真跑通过再接,不拿未跑代码上生产热路径。**待真验**:装 Intel SDE 或得 AVX512 机,`-DHYLE_SIMD_TARGET_AVX512` 编 vec_test 跑对拍(宽度无关测试直接可用)。

## 参考

- Highway `hwy/ops/arm_neon-inl.h`（intrinsic 逐个对照）。
- `container/probe.h` GroupNeon（mask pack 折位 idiom 来源 + "写+编+指令验未跑"先例）。
- simd-design 落地清单 + AVX512 延期节。
