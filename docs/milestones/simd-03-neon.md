# simd-03：ARM64 NEON + AVX-512 后端（NEON ops 已 Apple-Silicon 真机验证；AVX-512 仍存疑）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | 🔶→✅ **NEON 全路径 Apple-Silicon 真机验证**（2026-07-05，M4 Air/macOS）：**ops**(dot/saxpy)+ **dtype**(fp16 原生 vcvt / bf16 整数)+ **math**(exp/log/tanh 借 Highway NEON 核)三件套全部 NEON 真跑全绿——simd_test 53/53(`active_variant()==kNeon` 坐实、dtype 全 65536 fp16 位模式 + bf16 位精确 vs scalar、math vs libm ~3.5ULP)+ SwissTable GroupNeon(container_test 158/158)。双平台无回归(WSL x86 56/56)。**AVX512 仍写+编+objdump 未跑存疑**（无 AVX512 机/SDE） |

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
- **✅ 真机已跑（2026-07-05，Apple-Silicon M4 Air / macOS / Apple clang 21）**：NEON 结构与 SSE/AVX2 同构、vec_test 宽度无关的推断被真机证实——`hyle_simd_test` **49/49 全绿**(vec 对拍 + dot/saxpy NEON kernel),关键铁证：`ActiveVariantMatchesBestCompiledTier` 断言 `active_variant()==kNeon` 通过=dispatch 真选了 NEON kernel(非 scalar 蒙混);`hyle_container_test` **158/158**(SwissTable GroupNeon 真跑)。**双平台无回归**:同批 CMake 改动下 WSL x86 simd 56/56 + container 160/160。
- **真机暴露的真缺口(交叉编译从没碰到)**：NEON 此前只**单编 neon.h 一个头**,从没走完整 CMake 构建——真机一编发现 **simd/container 的 CMake 是 x86 写死**(无条件编 x86 TU + 无条件 `-mavx2`,arm64 clang 报 "unsupported option")。补:①`hyle_isa_flags` 加 NEON tier(ARMv8 baseline 空 flag);②simd/container CMakeLists 按 `CMAKE_SYSTEM_PROCESSOR` 分支(x86 编 x86 TU、arm64 编 neon TU/走 baseline);③`kernels_neon.cc` 薄 TU(复用 neon.h)+ `kernels.h`/`dispatch_ops.cc` 加 `HYLE_ARCH_ARM64`→kNeon 分支。另修一个 macOS 真 bug:`pthread_t` 是 opaque 指针(glibc 是整数),`thread.cc` 的 static_cast 非法→改 memcpy。**教训:"交叉编译过"≠"该架构能构建"——只编了一个后端头,不是编了模块。**
- **✅ dtype + math NEON 已补全(2026-07-05,真机)**:dtype fp16↔f32 用 Apple Silicon 原生 ARMv8.2 `vcvt`(finite 位精确、NaN by-class,同 x86 F16C 契约)、bf16↔f32 用 NEON 整数位运算(位精确匹配 scalar,NaN payload 含);math exp/log/tanh 借 Highway contrib/math 核(探针证 lower 到 NEON、match libm,arm64 用一个核跑全三 op)。dtype_test 5 个 cross-check + math 3 个 libm 对拍从"non-x86 skip"改真验证。**xsimd-NEON per-op 速度调优 + AVX512 无机/SDE 仍未跑**,是仅剩的 simd 存疑项。

## AVX-512 后端（已实现，写+编+objdump 验，未跑存疑）

> **纠"延期"误判（用户纠偏）**：本里程碑一度把 AVX512 延期,理由①不可真跑②mask 需 vec.h 改造。**用户指出:没机器不该不做,平台优先不知道写没写对,应实现后存疑——否则永远落不了地。** 照做后第②条被证伪。详见 [`simd-design.md` "AVX512 后端：已实现"]。

- [x] `include/hyle/simd/backend/avx512.h`：512 位 AVX-512(F+BW+VL+DQ),lane f32/f64/i32/u32(16/8/16 lane),op 全集。**自定义 `Mask512` 包 `__mmask16/8`（k 寄存器）**——比较直接写 k(`_mm512_cmp_*_mask`/`vpcmp*`)、`IfThenElse`→`_mm512_mask_blend_*`、`CountTrue`→k 本就是位掩码 popcnt。unsigned 比较有原生 epu32 mask(无需 SSE/AVX2 的 xor-MSB)。
- **验证**：`g++-14 -mavx512f -mavx512bw -mavx512vl -mavx512dq -c` 干净(任何 x64 可编)；objdump 实证真发 zmm/k-reg——dot→`vfmadd231ps %zmm`+`vextractf64x4`;relu→`vcmpgt_oqps → %k1`+`vmovaps %zmm{%k1}{z}`(k 掩码路径通);unsigned count→`vpcmpnleud → %k0`+`kmovw`+`popcnt`。
- **设计结论(重要)**：写 AVX512 **证明了 vec.h 的 per-backend mask 抽象够用**——`Mask` 本就每后端自定义(scalar=bool[]/SSE=向量/AVX512=k-reg),op 签名不变,vec.h 和测试零改动。先前"需 mask 改造"是误判。
- **存疑(诚实)**：未在 AVX512 硬件/SDE **执行**,正确性=指令级验证 + 与低 ISA 同构推断,非实跑对拍。dispatch 仍把 AVX512 主机降级跑 AVX2(已验证安全),avx512.h 是"已就绪待启用",**未接进 ops/dtype 运行期 selected_tier**——真跑通过再接,不拿未跑代码上生产热路径。**待真验**:装 Intel SDE 或得 AVX512 机,`-DHYLE_SIMD_TARGET_AVX512` 编 vec_test 跑对拍(宽度无关测试直接可用)。
- **批量红队补审（2026-06，整支柱批次，objdump 验）**：揪修 AVX512 两 bug。① `Ne(float/double)` 同 AVX2 用有序谓词 `_CMP_NEQ_OQ`(NaN 返 false)→改 `_CMP_NEQ_UQ`(objdump 确认立即数 `0x04`,无序,NaN 返 true)。② `Neg(float/double)` 用 `0-a` 得 `Neg(+0.0)=+0.0`,与 SSE4/AVX2 的 xor-符号位(`-0.0`)发散→改 `_mm512_xor_ps/pd`(objdump 确认 `vxorps/vxorpd`)。**仍未实跑**(objdump 只证编码对,NaN/符号零语义靠与已实跑的 AVX2 同构推断),并入上面的存疑账。

## 参考

- Highway `hwy/ops/arm_neon-inl.h`（intrinsic 逐个对照）。
- `container/probe.h` GroupNeon（mask pack 折位 idiom 来源 + "写+编+指令验未跑"先例）。
- simd-design 落地清单 + AVX512 延期节。
