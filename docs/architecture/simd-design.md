# HYLE simd/ 设计

> SIMD 抽象层（Tier 1，性能焦点 #1）。clay 这块基本空白，HYLE 净新 ~10-16k 行。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`engineering-charter.md`](./engineering-charter.md)。

## clay 现状：近乎空白

- `simd.h` 全文 **43 行**，只做编译期指令集探测（`__SSE2__`→emmintrin、`__SSE4_2__`→nmmintrin、ARM→sse2neon），**只定义 `CLAY_HAVE_SSE2/SSE4` 两个宏，无 vector 类型、无 AVX/AVX512、无运行时 dispatch**。
- `half.h` 是纯软件查表转换（512+2048 项表），无 F16C 硬件路径。
- 结论：**等于从零造**，但可继承 `math/utils.h` 的标量工具、`host_device.h` 宏、`CLAY_ALIGNMENT 64`。

## 2026 格局

| 库 | 性质 | 对 HYLE |
|---|---|---|
| **std::simd（C++26 P1928）** | 定宽、无运行时 dispatch | ❌ 不能当核心——"表达不了 90% 真实 SIMD 代码"（gather/scatter/mask/swizzle 弱）；仅可作平凡循环的薄互操作 |
| **Highway（Google）** | 可移植 + 运行时 dispatch + scalable vector（AVX512/NEON/SVE/RVV/WASM）| 金标准，但 C++ idiom 重、头文件庞大，整体 vendor 与"贴硬件/不过度抽象"有张力 |
| **xsimd** | 定宽 + 优化数学函数（exp/log/trig）| 数学核可借 |
| **sleef** | 高精度 SIMD transcendental 核 | exp/tanh 等的多项式核可借（别手搓，易错） |

## 核心决策：自造精简抽象 + 运行时 dispatch + 借数学核

与 io/（借 liburing 不借 asio）、compute/（借 ggml 算子不借框架）同构——**借难且已解决的、造贴自己需求的**：

| 部件 | 造/借 | 理由 |
|---|---|---|
| 可移植 vector 抽象 `vec<T,N>` + 多 ISA 后端 | 🔧 **自造（精简）** | 焦点 #1；不 vendor 整个 Highway（idiom 重、违背贴硬件）；参考其 scalable-vector + dispatch 理念造精简版 |
| 运行时 dispatch（CPUID + 多版本）| 🔧 **自造** | clay 完全没有，是核心缺口 |
| transcendental 数学核（exp/log/sin）| ✅ **借 sleef-style 多项式核** | 手推多项式逼近易错，已解决问题 |
| dtype 转换（fp16/bf16/int8/fp8）| 🔧 自造（硬件路径 + 软件 fallback）| 替换 clay 纯软件 half 表 |
| AI 数学函数（gelu/softmax/tanh）| 🔧 自造（组合数学核）| HYLE 特定，AI 热路径 |
| std::simd 互操作 | ⚠️ 薄封装 | 平凡循环可转发给它，非核心 |

## 分层结构

```
simd/
├─ vec.h          可移植 vector 类型 vec<T, Width>（load/store/算术/fma/compare/mask/shuffle）
├─ backend/       per-ISA 实现（编译多版本）
│  ├─ sse42.h  avx2.h  avx512.h     (x86)
│  ├─ neon.h   sve.h                 (arm)
│  └─ scalar.h                       (无 SIMD fallback)
├─ dispatch.h     运行时 CPUID 探测 + 函数指针表选最优后端
├─ dtype/         fp16/bf16/int8/fp8 ↔ fp32（F16C/AVX512-FP16/BF16 硬件 + 软件 fallback）
├─ math/          exp/log/tanh/gelu/softmax/erf（借 sleef 核 + 组合）
└─ ops/           供 container/compute 用的 SIMD 原语（如 SwissTable 探测、reduce、memcpy）
```

## dispatch：编译期消变体，运行期最多冷跳一次（三层）

clay 是编译期固定指令集且只到 SSE2——**跑在 AVX512 机器上只用 SSE2**，这是真缺口。修正方向不是"运行时函数指针表每次查"（那是热路径间接、违背极致），而是三层模型，**热路径任何一层都零间接**。

| 层 | 时机 | C++ 机制 | 热路径开销 |
|---|---|---|---|
| **构建期** | 为已知硬件编 | 宏/feature flag + 每 ISA 独立 TU + `-march=native`（llama.cpp `GGML_NATIVE` 本地自编默认）| **零**（错误路径根本不编出来）|
| **编译期** | 变体内部 | `if constexpr` + concepts + 模板单态化 | **零**（死分支消失、全内联）|
| **运行期** | 一个二进制发到未知 CPU | 同源码按 ISA 编 N 份（**每变体独立 DLL**），启动 CPUID **打分选最高分**变体 + dlopen，**一次冷跳**进去 | **进去后零间接**（满速跑该变体，非每调用查表）|

**两者都是一等公民、按部署选（校正自真实 llama.cpp）**：自编/已知硬件 → 构建期、零 dispatch（极致）；**分发单一二进制 → 运行期多变体，这是 llama.cpp 官方分发的常态**（`GGML_CPU_ALL_VARIANTS`+`GGML_BACKEND_DL`，~13 个 x86 变体各编独立 DLL，启动 `ggml_backend_score()` 打分选优）。**不是"运行期罕见/只是便利"，也不是 IFUNC，而是"每变体 DLL + load 时 score"**。

> **运行期 dispatch 有两种模型，别混（深读 highway 校正）**：
> - **变体 DLL 模型（llama.cpp）**：每 ISA 编独立 DLL，启动 dlopen 选一个，**之后该库内全调用零间接**——但 DLL 边界本身有成本。
> - **函数指针表模型（Highway `HWY_DYNAMIC_DISPATCH`）**：单二进制内一张按 target 优先级排的函数指针表，**每次跨变体调用过一次指针间接**（非"进去后零间接"）。
> 真正"热路径零间接"的只有**构建期 `-march=native` 静态路径**（或 highway `HWY_STATIC_TARGET`）。HYLE 若要"进去后零间接"须走变体 DLL 路线，不是 highway 的表。两条都可选，但要标清各自的间接成本。

**探测器隔离（llama.cpp 实证的坑）**：CPUID 探测代码必须编在 baseline ISA、独立 TU、**禁 LTO**——否则 LTO 把高级指令内联进探测函数，老 CPU 上探测自身先 SIGILL。`dispatch.h` 照此。

**组合优于包含**：每个 ISA kernel 是独立组合单元（一个 TU / 一个 tag / 一个变体 DLL），**不是**上帝类 `Vec` 内部 `switch(caps)`/vtable 分支。加一个厂商特化（AMX/VNNI）= 加一个组合单元、改 1 文件、0 调用点、0 热路径开销。

> 详见横切原则 [`coding-conventions.md` → 能力 dispatch](./coding-conventions.md)。`compute/` 的后端选择是同一模型，粒度从"指令集"换成"设备/扩展"。

## dtype 层（AI 数值刚需，scope 完整性评估补的缺口）

| 类型 | 硬件路径 | fallback |
|---|---|---|
| fp16 | F16C（`_mm_cvtph_ps`）/ AVX512-FP16 | 软件查表（继承 clay half）|
| bf16 | AVX512-BF16 / 截断转换 | 移位截断（bf16 就是 fp32 高 16 位）|
| int8/fp8 | AVX512-VNNI（int8 点积）| 标量 |

bf16 尤其重要（AI 训练/推理主力格式）；clay 完全没有。

## 与 mem/container 的耦合 + 一个 tier 倒置的澄清

- **对齐**：vector load/store 要求 `CLAY_ALIGNMENT=64` 对齐，与 `mem/` 分配器对齐契约一致（AVX512 一条 512-bit load 要 64B 对齐）。
- **SwissTable SIMD 探测的 tier 问题**：`container/`（Tier 0）的 SwissTable 要 16 路并行探测 control word，看似依赖 `simd/`（Tier 1）——**tier 倒置**。澄清：探测只需一小段自包含的 SSE2/NEON 片段（16 字节比较 + movemask），**container/ 内联这一小段、不依赖整个 simd/ 模块**。保持 T0 不依赖 T1。`simd/` 成熟后 container 可选择复用其 `ops/`，但不强耦合。

### probe.h 的去留裁决（2026-06，simd/ 已成熟后回看，用户提出）

> 背景：`container/probe.h` 是 simd/ 还不存在时、为做高性能 hash 自己内联的 SwissTable 控制字节探测（GroupSse2/GroupNeon/GroupPortable + `#ifdef __SSSE3__` 编译门控）。如今 simd/ 已有 vec 抽象 + 运行期 dispatch，问：probe.h 要不要改成复用 simd/？

**裁决：不改，保持现状。** 理由：

1. **改了就违背本节自己定的 tier 规则**。container 是 T0、simd 是 T1；让 probe.h `#include` simd/ 就是 T1→T0 倒置（依赖图 `container → simd` 是单向的，反向即架构倒置，roadmap 明令禁）。
2. **probe 不是 vec 抽象的合适用例**。它要的是"16 字节控制字节比较 + movemask + BitMask 迭代"这一段**专用语义**，不是 `vec<T,N>` 的类型化算术。硬塞进 vec 接口反而别扭、且把 SwissTable 的探测细节泄进通用 SIMD 层。
3. **没有正确性缺口**。SSE2 基线恒在且正确；SSSE3 快路径是 `#ifdef` 门控的纯加速（performance.md 实测：match_empty 的 SSSE3 路径是绝对小增益，**不**是 find-miss 差距来源）。NEON 已交叉编译验证。
4. **唯一真实缺口是"分发型二进制吃不到 SSSE3"**——纯编译门控,没开 `-mssse3` 就走 SSE2 兜底,即便 CPU 支持。但这是**小加速**,不值得为它倒挂依赖。

**真要做运行期 dispatch 的正确解(若将来 profile 证明 baseline SwissTable 是分发型二进制的真瓶颈)**：**不是**让 container 依赖 simd,而是**把 CPUID 探测器下沉到 `platform/`（比 container 更低层）**,让 container 和 simd 都能查同一个探测结果——这样 container 可在 hashmap 内部做"SSE2 vs SSSE3 group"的运行期分发,仍不碰 simd/。当前不做(收益小、增复杂度)。

**复审触发条件（写明,免得凭感觉重提）**：① profile 显示分发型(非 `-march`)二进制的 SwissTable 探测是实测热点 **且** ② SSSE3+ 路径能显著缩差距。两条同时成立才动,且按上一段方案(下沉 CPUID 到 platform),非 container→simd 依赖。

## 落地清单

1. ✅ `vec<T,N>` 抽象 + scalar fallback 后端（先能跑、再加速）。**simd-01。**
2. 🔶 SSE4.2 / AVX2 / AVX512 / NEON 后端 + 运行时 dispatch。**SSE4.2+AVX2=simd-02(实跑)、运行期 dispatch=simd-04、NEON+AVX512=simd-03(写+编+objdump 验,未跑存疑)。** 各后端覆盖,只剩 NEON/AVX512 待目标硬件实跑。
3. ✅ dtype 转换（先 fp16/bf16 硬件路径）。**simd-05。**
4. 📋 数学函数（借 sleef 核：exp/tanh → 组合出 gelu/softmax）。**simd-06。**
5. 📋 `ops/`：reduce、对齐 memcpy、SwissTable 探测片段。
6. 🔶 单测：每后端对拍 scalar 参考值、跨 ISA CI、数学函数精度 ULP 测试、benchmark 对标 Highway/手写 intrinsics。**scalar/SSE4.2/AVX2 对拍 + Highway bench 已做；NEON objdump 验；ULP 测试随 simd-06。**

### AVX512 后端：已实现（写+原生编+objdump 验，未跑存疑）——纠"延期"误判

> **历史更正（2026-06，用户纠偏）**：我一度把 AVX512"理由化延期",理由是①不可真跑②mask 模型需 vec.h 改造。**两条都被推翻了。** 用户指出:没机器不该不做——平台优先永远不知道写没写对,这种地方应当**实现然后存疑**,否则可能永远落不了地。照做后发现第②条根本是误判。本节记录真相,免得再犯。

**已实现**（`backend/avx512.h`）：512 位 AVX-512 后端,lane f32/f64/i32/u32(16/8/16 lane),op 全集匹配契约。验证标准与 NEON 一致——**写 + 原生编译(`-mavx512f -mavx512bw -mavx512vl -mavx512dq`,任何 x64 可编)+ objdump 验真发 zmm/k-reg 指令 + 未跑(无 AVX512 机/SDE)= 存疑待真跑**。

**objdump 实证**（铁证,非空谈）：dot→`vfmadd231ps %zmm`+`vextractf64x4` reduce；relu→`vcmpgt_oqps → %k1`+`vmovaps %zmm{%k1}{z}`(**k 寄存器掩码路径通**)；unsigned count→`vpcmpnleud → %k0`(原生无符号比较,无 xor-MSB)+`kmovw`+`popcnt`。

**第②条误判的更正(重要设计结论)**：我曾说"AVX512 的 k-reg mask 需要改造 vec.h 的 mask 抽象"。**错。** vec.h 的 `Mask` 本就是**每后端自定义**(scalar 用 `bool[]`、SSE/AVX2/NEON 用向量)——avx512.h 只是再定义自己的 `Mask512` 包 `__mmask16/8`,op 签名不变,vec.h 和测试**零改动**。比较直接写 k reg(`_mm512_cmp_*_mask`)、`IfThenElse`→`_mm512_mask_blend_*`、`CountTrue`→k 本就是位掩码 popcnt。**写它恰恰证明了抽象够用**——这正是用户"实现以验证设计"论点的兑现。附带红利:unsigned 比较有原生 epu32 mask 指令,比 SSE/AVX2 的 xor-MSB 技巧更干净。

**仍存疑的部分（诚实）**：未在 AVX512 硬件/SDE 上**执行**,故正确性是"指令级验证 + 与低 ISA 后端同构推断",非"实跑对拍"。**待真验**:装 Intel SDE 或得 AVX512 机后,用 `-DHYLE_SIMD_TARGET_AVX512` 编 vec_test 跑对拍(宽度无关测试直接可用)。在此之前 dispatch 仍把 AVX512 主机**降级跑 AVX2 变体**(已验证安全),故 avx512.h 是"已就绪待启用",未接进 ops/dtype 的运行期 selected_tier——等真跑通过再接,避免拿未跑代码上生产热路径。

## 待决（需主拍板）

1. **vector 抽象用定宽 `vec<T,N>` 还是 scalable（length-agnostic）**？**✅ 已决：scalable-friendly 接口 + 先实现定宽后端**——接口按 length-agnostic 设计（参考 Highway，将来加 SVE/RVV 不改 API），但首版只实现定宽后端（x86 SSE4.2/AVX2/AVX512 + ARM NEON，它们本就定宽）。对冲：不堵未来又不先背复杂度。
   > **深读 highway 落地约束**：照搬 highway 的 **D-tag 接口契约**——传零大小标签 `D=Simd<T,N,kPow2>`，op 形如 `VFromD<D> Set(D,T)`，`Lanes(d)` 取宽度。**关键：`Lanes(d)` 必须设计成可运行期求值**（定宽后端是 constexpr、SVE 后端是 `svcntw()` 运行期），否则将来加 SVE 要改 API。highway 已证同源码 x86/SVE 共存可行。**只照搬接口契约，不 vendor 整个 highway**（其 ops/ 全平台万行级、重宏元编程，违背"贴硬件/不过度抽象"）。给 `vec.h` 设硬上限（核心 op < ~2k 行/后端），超了就是丧失了不 vendor 的理由。
2. **数学核 vendor sleef 还是自抄多项式**？**✅ 倾向 vendor sleef**（Apache/BSL，精度经考验，主力 `_u35`(3.5ULP)/`_u10`(1ULP) 档）；adapter 几乎零成本（sleef 吃裸 intrinsic 向量，HYLE `vec.raw` 直接喂）。每 ISA 后端各封一份。**xsimd 不入依赖**，仅作 exp/log 多项式的第二参考（其 `common/*math*.hpp` 是纯 C++ 模板、可读性高于 sleef 的宏生成 C）。sleef 原生支持 SVE，不堵未来 scalable。
3. **要不要 SVE/RVV 进首版**？建议 v1 只做 x86(SSE4.2/AVX2/AVX512) + ARM NEON，SVE/RVV 留接口、缓实现。
4. **dtype 层借谁**？**✅ 自造，但参照 highway 的 `PromoteTo/DemoteTo` intrinsic 用法**（F16C `_mm_cvtph_ps`、AVX512-BF16 `cvtne2ps`、NEON-BF16）——照抄 intrinsic 不照抄抽象。sleef/xsimd 都无 fp16/bf16，唯 highway 当原生 lane 类型且硬件路径全；但 dtype 逻辑不复杂（bf16=fp32 高 16 位移位），自造避开 vendor 整个 highway。

## 参考

- [C++26 Shipped a SIMD Library Nobody Asked For（std::simd 表达力批判）](https://lucisqr.substack.com/p/c26-shipped-a-simd-library-nobody)
- [Highway：可移植 + 运行时 dispatch SIMD](https://google.github.io/highway)
- [xsimd：SIMD wrapper + 优化数学函数](https://github.com/xtensor-stack/xsimd)
- [Evaluation of C++ SIMD Libraries（AVX2/AVX512/SVE/NEON 基准）](https://www.mnm-team.org/pub/Fopras/rock23/PDF-Version/rock23.pdf)
