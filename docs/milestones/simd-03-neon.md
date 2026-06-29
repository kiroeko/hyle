# simd-03：ARM64 NEON 后端（+ AVX512 理由化延期）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | ✅ NEON 完成（交叉编译+objdump 验，未跑）；AVX512 显式延期（→ simd-03b） |

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

## AVX512 为何延期（不是遗漏，是有意）

详见 [`simd-design.md` 落地清单末 "AVX512 后端的理由化延期"]。两条硬理由：① **不可真跑**（本机 AVX2 级 + 无 SDE 模拟器,盲写跑不了=假完成；AVX512 不同构于 SSE/AVX2,objdump 不足以背书）；② **mask 模型根本不同**（`__mmask` k 寄存器 vs vec.h 的 mask=向量契约,需先给 mask 抽象加 k-reg 路径,是设计工作非照抄 intrinsic）。**当前安全**：AVX512 主机经 dispatcher 降级跑 AVX2 变体(合法安全,永不 SIGILL),只是没吃 512 位红利。**前置满足才开 simd-03b**：有 AVX512 机/SDE + vec.h mask 抽象扩 k-reg 路径。

## 参考

- Highway `hwy/ops/arm_neon-inl.h`（intrinsic 逐个对照）。
- `container/probe.h` GroupNeon（mask pack 折位 idiom 来源 + "写+编+指令验未跑"先例）。
- simd-design 落地清单 + AVX512 延期节。
