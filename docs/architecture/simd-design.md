# HYLE simd/ 设计

> SIMD 抽象层（Tier 1，性能焦点 #1）。clay 这块基本空白，HYLE 净新 ~5-8k 行（详见 hyle-scope.md 规模节）。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`engineering-charter.md`](./engineering-charter.md)。

## clay 现状：近乎空白

- `simd.h` 全文 **43 行**，只做编译期指令集探测（`__SSE2__`→emmintrin、`__SSE4_2__`→nmmintrin、ARM→sse2neon），**只定义 `CLAY_HAVE_SSE2/SSE4` 两个宏，无 vector 类型、无 AVX/AVX512、无运行时 dispatch**。
- `half.h` 是纯软件查表转换（512+2048 项表），无 F16C 硬件路径。
- 结论：**等于从零造**，但可继承 `math/utils.h` 的标量工具、`host_device.h` 宏、`CLAY_ALIGNMENT 64`。

## 2026 格局

| 库 | 性质 | 对 HYLE |
|---|---|---|
| **std::simd（C++26 P1928）** | 定宽、无运行时 dispatch | ❌ 不能当核心——"表达不了 90% 真实 SIMD 代码"（gather/scatter/mask/swizzle 弱）；仅可作平凡循环的薄互操作 |
| **Highway（Google）** | 可移植 + 运行时 dispatch + scalable vector（AVX512/NEON/SVE/RVV/WASM）| 金标准；不 vendor 整套抽象（idiom 重），但其 **contrib/math 多项式核已 vendored** 借用（log 等）|
| **xsimd** | 定宽 + 优化数学函数（exp/log/trig）| 数学核已 vendored 借用（exp/tanh）|

## 核心决策：自造精简抽象 + 运行时 dispatch + 借数学核

与 io/（借 liburing 不借 asio）、compute/（借 ggml 算子不借框架）同构——**借难且已解决的、造贴自己需求的**：

| 部件 | 造/借 | 理由 |
|---|---|---|
| 可移植 vector 抽象 `vec<T,N>` + 多 ISA 后端 | 🔧 **自造（精简）** | 焦点 #1；不 vendor 整个 Highway（idiom 重、违背贴硬件）；参考其 scalable-vector + dispatch 理念造精简版 |
| 运行时 dispatch（CPUID + 多版本）| 🔧 **自造** | clay 完全没有，是核心缺口 |
| transcendental 数学核（exp/log/sin）| ✅ **借 Highway/xsimd 多项式核（vendored）** | 手推多项式逼近易错，已解决问题；逐 op best-of-breed 选最快核 |
| dtype 转换（fp16/bf16/int8/fp8）| 🔧 自造（硬件路径 + 软件 fallback）| 替换 clay 纯软件 half 表 |
| AI 数学函数（gelu/softmax/tanh）| 🔧 自造（组合数学核）| HYLE 特定，AI 热路径 |
| std::simd 互操作 | ⚠️ 薄封装 | 平凡循环可转发给它，非核心 |

## 分层结构

```
simd/
├─ vec.h          可移植 vector 类型 vec<T, Width>（load/store/算术/fma/compare/mask/shuffle）
├─ backend/       per-ISA 实现（编译多版本）
│  ├─ sse4.h   avx2.h  avx512.h     (x86)
│  ├─ neon.h   sve.h                 (arm)
│  └─ scalar.h                       (无 SIMD fallback)
├─ dispatch.h     运行时 CPUID 探测 + 函数指针表选最优后端
├─ dtype/         fp16/bf16/int8/fp8 ↔ fp32（F16C/AVX512-FP16/BF16 硬件 + 软件 fallback）
├─ math/          exp/log/tanh/gelu/softmax/erf（借 Highway/xsimd 核 + 组合）
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

**探测档 ≠ kernel 档（kAvx 的案例,2026-06）**：`best_isa()` 的枚举档报告的是**CPU 真实能力**,与"我们编了哪些 kernel"是**两层**。`Isa::kAvx`(纯 256-bit float,Sandy/Ivy Bridge:有 AVX 无 AVX2 的 256-bit 整数)就是这个分离的实例——`f.avx` 早已被探测(且 AND 进 OS-YMM-save 的 `read_xcr0` 门),但之前没有对应枚举档,Ivy Bridge 被误报成 `kSse42`。补 `kAvx` 让 `best_isa()` 诚实。但**没有 avx-only kernel**,故 `dispatch_ops.cc` 的 `selected_tier()` 把 AVX 主机 clamp 到 sse4 变体跑。两层故意分开:**探测层求真**(CPU 是什么如实报)、**kernel 层务实**(只编值得编的档,缺档向下 clamp,绝不 SIGILL)。有意思的印证:dtype 的 f16c kernel 门正是 `f16c && avx`=Ivy Bridge,所以纯 AVX 机本就在跑 f16c dtype kernel,`kAvx` 只是给这个已存在的能力起了名。

> 详见横切原则 [`coding-conventions.md` → 能力 dispatch](./coding-conventions.md)。`compute/` 的后端选择是同一模型，粒度从"指令集"换成"设备/扩展"。

## dtype 层（AI 数值刚需，scope 完整性评估补的缺口）

| 类型 | 硬件路径 | fallback | 状态 |
|---|---|---|---|
| fp16 | F16C（`_mm_cvtph_ps`）/ AVX512-FP16 | 软件查表（继承 clay half）| ✅ simd-05 |
| bf16 | AVX512-BF16 / 截断转换 | 移位截断（bf16 就是 fp32 高 16 位）| ✅ simd-05 |
| fp8 E4M3/E5M2 | AVX10.2 / GPU tensor core（本代 CPU 均无）| **标量位运算（即完整实现）** | ✅ simd-09 |
| int8 | AVX512-VNNI（int8 点积）| 标量 | int8 lane 已有，VNNI 点积按消费 |

bf16 尤其重要（AI 训练/推理主力格式）；clay 完全没有。

**fp8（simd-09 补，推理常见精度）**：OCP 两格式 **E4M3FN**（bias 7/3 尾数/无 inf/唯一 NaN=0x7F/max 448，权重激活）与 **E5M2**（bias 15/2 尾数/IEEE inf-NaN/max 57344，梯度）。编码 transcribe 自权威 **CUDA `cuda_fp8.hpp`**（ggml/Highway 均无 host fp8→**自实现**，同 fp16/bf16 抄 Highway 的做法）。saturate-to-max-normal（OCP SATFINITE）；E5M2 真 inf 输入保留 inf，仅 finite 溢出 saturate。**关键：无 CPU 硬件 fp8-convert**——硬件 fp8 转换是 AVX10.2 / GPU tensor core 专属，本代 AVX2/AVX-512 CPU 全无，Intel SDE 也不模拟 AVX10.2 fp8 convert。故 **scalar 位运算就是当前完整实现**（非缓做/非半成品）；有 AVX10.2 硬件时按 `cvt_avx512bf16.cc` 先例接原生核。验证：known values 锚 OCP 谱 + **全 256 值 exhaustive round-trip**（fp8 仅 256 值→完备，强于 fp16 的抽样）+ 三平台（MSVC/WSL GCC/arm64 qemu 512-check bad=0）。

## 与 mem/container 的耦合 + 一个 tier 倒置的澄清

- **对齐**：vector load/store 要求 `CLAY_ALIGNMENT=64` 对齐，与 `mem/` 分配器对齐契约一致（AVX512 一条 512-bit load 要 64B 对齐）。
- **SwissTable SIMD 探测的 tier 问题**：`container/`（Tier 0）的 SwissTable 要 16 路并行探测 control word，看似依赖 `simd/`（Tier 1）——**tier 倒置**。澄清：探测只需一小段自包含的 SSE2/NEON 片段（16 字节比较 + movemask），**container/ 内联这一小段、不依赖整个 simd/ 模块**。保持 T0 不依赖 T1。`simd/` 成熟后 container 可选择复用其 `ops/`，但不强耦合。

### probe.h 的去留裁决（2026-06，simd/ 已成熟后回看，用户提出）

> 背景：`container/probe.h` 是 simd/ 还不存在时、为做高性能 hash 自己内联的 SwissTable 控制字节探测（GroupSse2/GroupNeon/GroupPortable + `HYLE_HAVE_SSSE3` 编译门控）。如今 simd/ 已有 vec 抽象 + 运行期 dispatch，问：probe.h 要不要改成复用 simd/？

**裁决：不改，保持现状。** 理由：

1. **改了就违背本节自己定的 tier 规则**。container 是 T0、simd 是 T1；让 probe.h `#include` simd/ 就是 T1→T0 倒置（依赖图 `container → simd` 是单向的，反向即架构倒置，roadmap 明令禁）。
2. **probe 不是 vec 抽象的合适用例**。它要的是"16 字节控制字节比较 + movemask + BitMask 迭代"这一段**专用语义**，不是 `vec<T,N>` 的类型化算术。硬塞进 vec 接口反而别扭、且把 SwissTable 的探测细节泄进通用 SIMD 层。
3. **没有正确性缺口**。SSE2 基线恒在且正确；SSSE3 快路径由 `HYLE_HAVE_SSSE3` 门控,且 **SSSE3 现为全局 baseline**(CMake 对 x86 GCC/Clang 全局加 `-mssse3`、MSVC x64 无条件),故三平台默认生效(performance.md 实测：match_empty 的 SSSE3 路径是绝对小增益，**不**是 find-miss 差距来源)。NEON 已交叉编译验证。
4. **无遗留分发缺口**——SSSE3 既是全局 baseline,分发型二进制默认即吃到 SSSE3 快路径,无需 `-march=native`。

**运行期 dispatch 的正确解——CPUID 探测已下沉 platform(2026-07 落地)**:曾预留的方案是"**不**让 container 依赖 simd,而是**把 CPUID 探测器下沉到比 container 更低的层**,让 container 和 simd 都查同一份探测结果"。**此下沉已实施**:`CpuFeatures`/`best_isa`/`cpu_features` 从 `simd/dispatch` 搬到 `platform/cpu.{h,cc}`(`simd/dispatch.h` 转发,simd 内部零改动),container 经 `container→mem→platform` 白拿探测,无新依赖边。首个消费者是 **container·str 的 `count`/`rfind` AVX2 变体**(`string_ref_avx2.cc`,独立 `-mavx2` TU,运行期 `cpu_features().avx2` 门控)——追平 MSVC STL 的 AVX2(见 performance.md container·str 节)。**probe.h(SwissTable)本身仍未上此 dispatch**:其 SSSE3 baseline 已够(下方复审条件未触发),但下沉后的探测器已就位,将来若需要可直接查。

**probe.h 的复审触发条件(仍适用于 SwissTable 探测)**:① profile 显示分发型(非 `-march`)二进制的 SwissTable 探测是实测热点 **且** ② 比 SSSE3 baseline 更高的 ISA(AVX2 group)路径能显著缩差距。两条同时成立才给 probe.h 上 AVX2 变体(探测器已在 platform,不再是阻碍)。

## 落地清单

1. ✅ `vec<T,N>` 抽象 + scalar fallback 后端（先能跑、再加速）。**simd-01。**
2. ✅ SSE4.2 / AVX2 / AVX512 / NEON 后端 + 运行时 dispatch。**SSE4.2+AVX2=simd-02(实跑)、运行期 dispatch=simd-04、NEON=simd-03(mac 真机)、AVX512=simd-03 写 + simd-09 用 Intel SDE 真跑(vec_test 169/169 全 op 对拍 scalar oracle,dispatch 真选 kAvx512)**。四定宽 ISA 全真跑,不再"未跑存疑"。**+ SVE + RVV(simd-09,两个 scalable 后端)**:`backend/sve.h`(~76 op,qemu-aarch64 多 VL 128/256/512)+ `backend/rvv.h`(核心 op,qemu-riscv64 rv64gcv)。sizeless 类型用 traits 映射(见下 SVE/RVV 节)+ 运行期 Lanes(svcntw / vsetvlmax)。SVE 的 gather/compress/scatter/predicated-tail 全原生非 store+reload。两者 gtest + CMake gate(`-DHYLE_SIMD_SVE/RVV=ON`)+ qemu 脚本固化(`scripts/{sve,rvv}-qemu-test.sh`)。**RVV 复用 SVE 全部 scalable 机制、vec.h 共享层零改动**——坐实"sizeless=后端内模板元编程"对两个 ISA 都成立。
3. ✅ dtype 转换（fp16/bf16 硬件路径）。**simd-05。** f32→bf16 的 **AVX-512-BF16 native `vcvtneps2bf16`** 档随 AVX512 变体 SDE 可真跑(此前仅 objdump 验)。
4. 🔶 数学函数（借 Highway/xsimd 核）。**simd-06 exp/log/tanh + simd-09 补 11 个(sin/cos/exp2/log2/erf/atan/cbrt/expm1/log1p/asinh/acosh),per-op best-of GCC+MSVC 双平台 bench 定 winner,三平台 + AVX512-SDE 真跑。** **gelu/softmax 组合核=消费驱动缓**(compute-07 用 ggml 算子,不调 simd 的;非缺口)。
5. ✅ `ops/`：dot_f32/saxpy_f32 跨 scalar/SSE4/AVX2/AVX512 dispatch;**AVX512 派发档 simd-09 Intel SDE 真跑坐实(tier 探针确认 kAvx512,非仅 objdump)**。reduce/对齐 memcpy/SwissTable 探测片段按需(container 已自有 SwissTable 探测)。
6. ✅ 单测：每后端对拍 scalar 参考值、数学函数 ULP 测试、benchmark 对标 Highway/xsimd。**四 ISA(SSE4/AVX2/AVX512-SDE/NEON)+ scalar 全真跑对拍 + WSL ASan + AVX512 SDE;math ULP vs libm ~3.5;three-way best-of bench。**

### AVX512 后端：已实现（写+原生编+objdump 验，未跑存疑）——纠"延期"误判

> **历史更正（2026-06，用户纠偏）**：我一度把 AVX512"理由化延期",理由是①不可真跑②mask 模型需 vec.h 改造。**两条都被推翻了。** 用户指出:没机器不该不做——平台优先永远不知道写没写对,这种地方应当**实现然后存疑**,否则可能永远落不了地。照做后发现第②条根本是误判。本节记录真相,免得再犯。

**已实现**（`backend/avx512.h` 后端 + `ops/kernels_avx512.cc` 派发 kernel）：512 位 AVX-512 后端,lane f32/f64/i32/u32(16/8/16 lane),op 全集匹配契约。`kernels_avx512.cc` 是 3 行 TU(`#define HYLE_SIMD_TARGET_AVX512` + suffix + 共享 `kernel_impl.h`),与 avx2/sse4 TU 同构,16-wide(512/4B f32)编译同一份 kernel 体、零源码改动——D-tag 抽象的兑现。验证标准与 NEON 一致——**写 + 原生编译(`-mavx512f -mavx512bw -mavx512vl -mavx512dq` / MSVC `/arch:AVX512`,任何 x64 可编)+ objdump 验真发 zmm/k-reg 指令 + 未跑(无 AVX512 机/SDE)= 存疑待真跑**。

**objdump 实证**（铁证,非空谈,双平台）：backend dot→`vfmadd231ps %zmm`+`vextractf64x4` reduce；relu→`vcmpgt_oqps → %k1`+`vmovaps %zmm{%k1}{z}`(**k 寄存器掩码路径通**)；unsigned count→`vpcmpnleud → %k0`(原生无符号比较,无 xor-MSB)+`kmovw`+`popcnt`。派发 kernel TU(`kernels_avx512.cc.obj`):MSVC 33 条 zmm(`vfmadd213ps zmm`/`vmovups zmm`)、GCC-14 9 条 zmm(`vfmadd231ps zmm`/`vmovups zmm`/`vbroadcastss xmm→zmm`/`vextractf64x4 zmm→ymm`);两编译器 codegen 计数不同(GCC reduce 尾降 ymm/xmm),但核心 512-bit FMA/load 都真发。

**第②条误判的更正(重要设计结论)**：我曾说"AVX512 的 k-reg mask 需要改造 vec.h 的 mask 抽象"。**错。** vec.h 的 `Mask` 本就是**每后端自定义**(scalar 用 `bool[]`、SSE/AVX2/NEON 用向量)——avx512.h 只是再定义自己的 `Mask512` 包 `__mmask16/8`,op 签名不变,vec.h 和测试**零改动**。比较直接写 k reg(`_mm512_cmp_*_mask`)、`IfThenElse`→`_mm512_mask_blend_*`、`CountTrue`→k 本就是位掩码 popcnt。**写它恰恰证明了抽象够用**——这正是用户"实现以验证设计"论点的兑现。附带红利:unsigned 比较有原生 epu32 mask 指令,比 SSE/AVX2 的 xor-MSB 技巧更干净。

**仍存疑的部分（诚实）**：未在 AVX512 硬件/SDE 上**执行**,故正确性是"指令级验证(双平台 objdump)+ 与低 ISA 后端同构推断 + 共享 `kernel_impl.h` 体(与已实跑的 avx2/sse4 同一份源码,仅宽度宏不同)",非"实跑对拍"。**待真验**:装 Intel SDE 或得 AVX512 机后,`ops_test`/`vec_test` 的对拍(宽度无关)直接可用——上真机即自证,无需改测试。

**派发现状(2026-06,用户指令"写且默认派发")**:`kernels_avx512.cc` 已接进 `dispatch_ops.cc` 的 `selected_tier()`——AVX512 主机**默认派发到 zmm kernel**(不再降级 AVX2)。这是用户的明确取舍:平台优先库应让 AVX512 机器跑 AVX512 路径,即便该路径在本开发箱(AVX2-only)未执行验证过;靠双平台 objdump + 同构推断兜底,真机一跑即证伪或确认。**风险显式标注**:一台真 AVX512 CPU 会跑这条未执行验证的路径(`存疑`),已在本节与代码注释(`kernels_avx512.cc` 顶部)标明。这与早先"降级 AVX2、等真跑再接"的保守立场相反——后者已被用户推翻(理由同 NEON/backend:不派发就永远不知道写没写对,落不了地)。

### AVX-512 dtype/math 上架（2026-07，同"实现+接线+编译验真、执行存疑"原则）

承上 ops 后端先例,把 dtype 与 math 的 AVX-512 路径也上架(performance.md AVX-512 节①③)。**同一纪律**:本机 AVX2-only,512-bit 路径双平台编译+objdump 验真发宽指令,执行对拍 `存疑` 待真机/SDE。

- **① dtype `f32→bf16` native(`cvt_avx512bf16.cc`)**:`_mm512_cvtneps_pbh`(`VCVTNEPS2BF16`)把 AVX2 路径的 ~7-op RNE 序列(bias+shift+packus+permute)折成**单条**硬件 RNE 指令,16-wide。`dtype_dispatch` 的 `f32_to_bf16_batch` 优先 `avx512bf16` 档(`cpu_features().avx512bf16` 门控,该 flag 在 `dispatch.cc` 已 AND-fold XGETBV/ZMM),回退 AVX2→scalar。`bf16→f32` 无 512 win(纯 `<<16` 移位,AVX2 已最优),不加 512 变体。新 `AVX512_BF16` CMake tier(GCC `-mavx512bf16` 叠加四件套;MSVC `/arch:AVX512` 已含,probe 验)。**验真**:GCC14 `vcvtneps2bf16`(EVEX.512)、MSVC `vcvtneps2bf16 ymm,zmmword`。测试 `F32ToBf16Avx512MatchesScalar`(finite bit-exact、NaN equal-by-class——硬件 quiet 到自有 QNaN,同 F16C fp16 立场),本机无 bf16 故正确 SKIP。
- **③ math 512b 变体(`transcendental_{xsimd,highway}_avx512.cc`)**:借同核、仅编译宽一档→zmm。`math_dispatch` 加 avx512 档优先(exp/tanh→xsimd-512、log→highway-512),回退 AVX2→scalar。新 `hyle_set_avx512_math_isa`(全 AVX-512 组**含 cd**——xsimd 的 avx512dq 核 `#error` 要求 cd;**+ AVX2 baseline**——Highway 静态 dispatch 需完整宏链选 AVX3,否则降级)。**验真**:GCC xsimd512 92zmm/highway512 43zmm、MSVC xsimd512 581zmm。
  - **MSVC Highway-AVX3 排除(probe 实锤,非接线 bug)**:Highway 在 `detect_targets.h:192` 把整个 AVX3 标 `HWY_BROKEN_MSVC`(MSVC AVX-512 codegen 已知 bug),故 `/arch:AVX512` 下 `hn::` 仍发 ymm(dumpbin 250 ymm/0 zmm)。MSVC 上**没有 512-bit log win**(avx2 highway log 已是 Highway 能给的最宽)。故 `transcendental_highway_avx512.cc` 与 dispatch 的 log-512 分支用 `#if !defined(_MSC_VER)` 守卫,MSVC log 留 avx2 highway,**不路由到 fake-512**。xsimd(exp/tanh)不受影响——其 512 batch 是纯 AVX-512F,MSVC 真发 zmm。
- **执行存疑边界(同 ops)**:三项 512-bit 正确性=指令级验证(双平台 objdump)+ 同构推断(borrow 同核,仅宽度宏不同)+ 共享/对 oracle 测试。真机/SDE 跑既有 `dtype_test`/`math_test`/`ops_test`(全宽度无关或对 oracle/libm)即自证,**无需改测试**。本机双平台 56 过 + 1 正确 SKIP。

### op 覆盖补齐（2026-07,compute 前把 vec 后端 op 集补全）

起点是 ~35 个 op(Set/Load/Store/算术/Min/Max/Neg/Abs/Sqrt/MulAdd/位运算/比较/IfThenElse/AllTrue-CountTrue/Reduce{Sum,Min,Max}/GetLane)。相对 Highway ~200 op 的差距按**优先级分批**补全,四后端(sse4/avx2/avx512/neon)+scalar 参考,**抄 Highway 的 per-op intrinsic 序列(契约,非 vendor 库)**,每批独立真机验证(WSL x86 跑 scalar/sse4/avx2、编 avx512;mac SSH 跑 NEON;三平台 gtest 各 136+ 全绿):

- **批1 移位** `ShiftLeft<k>/ShiftRight<k>`(编译期立即数):x86 无 epi8 移位(epi16+mask 模拟)、avx2/sse4 无 `srai_epi64`(srli+符号填充)、avx512 原生、NEON 全宽原生;**i8 算术右移**要 logical-byte-shift+cmpgt 符号填充(naive slli-to-high-byte 错——已修+测)。
- **批2 饱和/平均** `SaturatedAdd/Sub`(8/16-bit)+`AverageRound`(u8/u16):x86 `adds/subs/avg` 原生、NEON `vqaddq/vqsubq/vrhaddq`、scalar 宽类型算后 clamp。
- **批3 同宽转换** `ConvertTo/RoundToInt`(f32↔i32 量化(de)quant 核):x86 `cvttps/cvtps/cvtepi32`、NEON `vcvtq/vcvtnq`。
- **批4 浮点舍入** `Round/Floor/Ceil/Trunc`:x86 `round_ps/pd`、avx512 `roundscale`、NEON `vrndnq/mq/pq/q`、scalar `nearbyint/floor/ceil/trunc`。
- **批5 掩码/尾部** `FirstN/MaskedLoad(Or)/BlendedStore/LoadN/StoreN`(向量化循环余数,std::simd 表达不了的 90% 之一):avx512 原生 k 掩码 fault-safe;avx2/sse4/neon 内存暂存 fault-safe(只碰 on-lane / 前 k 个 lane)。
- **批6 reduce 扩展 + Broadcast** `ReduceMul/ReduceAnd/ReduceOr`(avx512 32/64-bit 原生 `_mm512_reduce_*`、余宽 store+fold)+`Broadcast<i>`(lane splat)。
- **批7 变宽转换** `PromoteLowerTo/PromoteUpperTo`(满 narrow→满 wide-半)+`DemoteTo`(2→1 饱和 pack)+`ReorderDemote2To`:量化核。x86 `cvtep*`/`packs`(avx2 4-way permute 修块序、avx512 `permutexvar_epi64` 修 4 块交织,与 Highway kIdx 同)、NEON `vmovl_/vmovl_high_`/`vqmovn`+`vcombine`(天然有序)。**纠正了先前"需 partial 向量所以后置"的误判**——half-at-a-time 形式完美契合定宽模型。
- **批8 变量移位 + 加宽乘** `Shl/Shr`(每 lane 计数;avx2 `sllv/srlv/srav`+sse4 无原生全 fallback+avx512 全宽原生含 srav_epi64+NEON vshlq 负计数)+`MulHigh`(i16/u16)+`WidenMulPairwiseAdd`(i16→i32 madd,int8/16 GEMM 核)+`AbsDiff`。
- **批9 swizzle 全套** `OddEven/Reverse/Reverse2/InterleaveLower/Upper/Concat{Lower,Upper}²/ConcatOdd/Even/DupOdd/Even/TableLookupLanes+IndicesFromVec`。**关键:InterleaveLower/Upper 是每 128-bit 块**(Highway 契约,256b avx2=2 块);avx2 块内原生(unpack/blend/permute2x128)+跨块难的 store+reload,sse4/neon 单块 store+reload,avx512 4 块块内。
- **批10 gather/scatter + compress/expand** `GatherIndex/ScatterIndex`(embedding lookup 核)+`Compress/CompressStore/Expand`(稀疏过滤核)。**性能档已收口(见下 op-parity 节)**:avx2 GatherIndex→原生 vpgatherd、Compress→packed 索引表+vpermd、avx512→vpcompress/vpexpand/vpgather、sse4 Compress→pshufb 字节索引表;仅无硬件的宽度/后端保留 store+reload(同 Highway)。
- **批11 位计数 + 符号 + 杂项** `PopulationCount/LeadingZeroCount/TrailingZeroCount`(**已全换原生**:vpshufb 半字节-LUT popcount、float-exp trick LZCnt/TZCnt、NEON vcntq/vclzq)+`Not/Clamp/RotateLeft/RotateRight/CopySign/BroadcastSignBit/IfNegativeThenElse/ZeroIfNegative`(全由现有 op 组合,四后端同构)。

**op 覆盖现状**:量化(promote/demote/convert/满足 int8 GEMM 的 madd/mulhigh)、通用向量循环(移位/舍入/饱和/掩码尾部/reduce 全家族)、swizzle(interleave/concat/reverse/permute)、gather/scatter/compress/expand、位计数/符号——**全部四后端 + scalar 覆盖,三平台真机对拍 scalar oracle**。设计原则(用户 2026-07-06 纠偏):**高性能 infra 库的消费面最广,能力要提供到位,不能因"暂无消费者"后置**。**性能档也已收口(2026-07-09 op-parity,见下节)**:曾经 store+reload/std 标量的慢 op 全换原生 intrinsic 并对拍 Highway 拉到统计打平——能力面 + 性能面双达标。各后端仍在 ~2k 行设计上限内(最大 neon ~1.2k)。

### op-parity 性能收口(simd-09 尾,2026-07-09)——自造件 vs Highway 同 ISA 同 op

用户核心担忧:"自己手写了 ~217 op/后端,性能不如抄 Highway 就很愚蠢"。建 `bench/simd/op_parity_bench.cc`(HYLE vs Highway vs xsimd,**同 TU/同 ISA/两边 header-only 都内联**=真 codegen 差,非内联不对称;`bench-stat.py -n15` 看中位倍数,gap>两变体 spread 和才判赢/输)。

**结论:9 个 store+reload/std 标量慢 op(GetExponent/DupEven-Odd/ReverseBits/PopCount/LZCnt/TZCnt/Compress/ConcatEven-Odd)全换原生 + 2 个残留 2.x 输家(LZCnt/Compress)也追平 → 四固定宽后端全部 ≥ 统计打平**:
- 手法:vpshufb 半字节-LUT(popcount/reversebits)、位提取指数域(GetExponent)、**float-exp trick**(LZCnt=min(158-biased_exp,32)、TZCnt=min(be(x&-x)-127,32),`cvtepi32_ps` 取指数)、packed 索引表+vpermd / pshufb 字节表(Compress)、vpermd/shuffle_ps/vuzp/vtrn(Concat/Dup)。NEON 硬件原生最省(vcntq/vclzq/vrbitq/vuzp/vtrn)。
- **Compress 契约变更**:HYLE 旧契约保证不活跃尾部清零(强于 Highway"unspecified"),该保证在 AVX2/SSE4(无清零-compress 指令)是 ~2x 软件税。核实唯一消费者 vqsort 自带 Highway 不依赖 + 标准用法(`CompressStore`/前 `CountTrue`)不碰尾部后,**契约改为尾部 unspecified**(对齐 Highway/xsimd),各 target 尾部行为可不同(avx512 maskz 硬件清零、store+reload 路径恰好清零、avx2/sse4 vpermd/pshufb 不清零)。
- SVE/RVV 无需改(SVE 早用原生 svcnt/svclz/svuzp;RVV popcount/clz 属 Zvbb 扩展,rv64gcv 无,登记边界同 Highway)。详见 `docs/performance/benchmarks-simd.md` op-parity 全表。

### SVE scalable 后端(simd-09,首个非定宽后端)——sizeless 类型如何融入 D-tag 契约

用户"SVE/RVV 不假设向量长度,为啥冲击架构 / 模板元编程无法解决么"——**实证核查后纠正了一度过度渲染的"冲击全库地基"结论**:SVE/RVV **不冲击架构**,是 `backend/sve.h` 内部**模板元编程可解**的问题(qemu 多 VL 真跑坐实)。

**为什么不是纯"独立 branch"、又为什么不冲击架构(两个都对一半)**:
- 后端选择**确实是独立分支**(vec.h ladder 加一个 `#elif HYLE_SIMD_TARGET_SVE`,CMake 加 tier)——这部分隔离。
- 真正的挑战是 **sizeless 类型**:SVE 的 `svfloat32_t`/RVV 的 `vfloat32m1_t` 按 ACLE **不能做 sizeof / 结构体成员 / 数组元素 / 全局变量**。而定宽后端的 `Vec` 是 `struct { float32x4_t raw; using PrivateT; kPrivateN; }`——把寄存器当结构体成员。
- **但 `DFromV`/`Vec` 每后端各自定义(非全库共享契约)**:一个 TU 只 include 一个后端头,后端从不共存 → sve.h 可自定义一套,不动其他后端或 vec.h 共享层。

**模板元编程解法(全在 sve.h 内)**:
1. `VFromD<D>` 用 traits 映射 `detail::SveVecT<T>::type`(替代 struct 成员包 raw)——sizeless 类型能做函数**参数/返回值/局部变量**(ABI 合法),HYLE 的 op 全是 `V Add(V,V)` 传值/返回,天然不碰禁区。
2. `DFromV<V>` 用**反向 traits 特化表** `detail::DFromV_sve<V>`(替代 `V::PrivateT` 成员访问,sizeless 无成员)。
3. store+reload 的 `T buf[kMaxLanes]` 是**标量数组**(T=float/int,合法);`kMaxLanes` 从"精确"变"上界"(kNativeVectorBytes=256=2048-bit 上界),循环边界用运行期 `Lanes(d)`。**但 SVE 的 gather/compress/scatter/predicated-tail 全走原生**(`svld1_gather`/`svcompact`/`svst1_scatter`/`svwhilelt`),不 store+reload——正是 scalable 的性能强项。
4. `Lanes(d)` 运行期化:vec.h `HYLE_SIMD_HAVE_SCALABLE`(SVE/RVV target 翻 1)关掉泛型 constexpr Lanes,sve.h 供运行期 `svcntw/svcntb`。**这是唯一碰共享层处,且设计者早预留了宏。**

**验证(qemu-aarch64,同一二进制强制 VL 128/256/512)**:~76 op(Set/Load/Store/算术/位/比较→svbool_t/选择/mask-reduce/值reduce/移位/Reverse/tail/compress/scatter/gather/ConvertTo/BitCast/饱和/分类/CopySign/位计数)对拍 scalar oracle **8 个 gtest 全绿 × 3 VL**。lane 数随 VL 正确缩放(f32:4/8/16,i8:16/32/64)。`scripts/sve-qemu-test.sh` 可复现(cross-build + qemu 多 VL,类比 AVX-512 的 Intel SDE)。CMake `-DHYLE_SIMD_SVE=ON` gate(真 SVE 机 ctest 直跑)。

**未做的诚实边界**(~83 op,留 backlog):固定 128-bit-block 语义(ConcatOdd/InterleaveLower/ZipLower/DupEven/TableLookupBytes——Highway 在 SVE 上都特殊处理,scalable 下"块"概念模糊)、变宽转换(PromoteLowerTo/DemoteTo/ReorderDemote2To——改 lane 数,与运行期 VL 交互微妙)、SVE2 crypto(需 `+sve2-aes`)。核心载荷面已覆盖。

### RVV scalable 后端(simd-09,第二个 scalable 后端,镜像 SVE)

`backend/rvv.h` 复用 SVE 建立的**全部 scalable 机制**(`HYLE_SIMD_HAVE_SCALABLE`=1 → 运行期 Lanes、kNativeVectorBytes 上界、sizeless traits 映射 `RvvVecT<T>` + 反向表 `DFromV_rvv`)。首版 **LMUL=1**(kPow2=0),`Lanes(d)`=`vsetvlmax`。覆盖 Set/Load/Store/算术/位/比较→vbool/选择(vmerge)/mask-reduce(vcpop)/GetLane/值reduce(vred*)/移位。

**RVV 独特处(vs SVE)**:① **每 intrinsic 显式 `vl`**(SVE 用隐式谓词)——从 `Lanes(d)` 算一次穿进每个 op;② **unsigned lane 用独立助记符**(`vminu/vmaxu/vmsltu/vmsgtu` 非 `vmin` 泛型),signed/unsigned `if constexpr` 分派(这是 RVV 铺量的主要 gotcha);③ mask 是 `vbool<ratio>_t`(ratio=SEW/LMUL)。

**验证**:`tests/simd/rvv_test.cc`(3 gtest,VL-agnostic 对拍 scalar,含 unsigned-mnemonic 覆盖)qemu-riscv64 全绿。`scripts/rvv-qemu-test.sh` 可复现(cross-build rv64gcv + qemu)。CMake `-DHYLE_SIMD_RVV=ON` gate。**边界**:native gather/compress/scatter、变宽 convert、Zvbb popcount/clz(rv64gcv 无 Zvbb)留 backlog。**结论坐实**:SVE 趟通的 scalable 抽象**真的可复用**——RVV 后端只是换 intrinsic 方言(+vl 穿线 +unsigned 分派),vec.h 共享层零改动,证明"sizeless=后端内模板元编程"的判断对两个 ISA 都成立。

## 已定决策（1/2/4 已落地，3 SVE 进行中/RVV 排队）

1. **vector 抽象用定宽 `vec<T,N>` 还是 scalable（length-agnostic）**？**scalable-friendly 接口 + 先实现定宽后端**——接口按 length-agnostic 设计（参考 Highway，将来加 SVE/RVV 不改 API），但首版只实现定宽后端（x86 SSE4.2/AVX2/AVX512 + ARM NEON，它们本就定宽）。对冲：不堵未来又不先背复杂度。
   > **深读 highway 落地约束**：照搬 highway 的 **D-tag 接口契约**——传零大小标签 `D=Simd<T,N,kPow2>`，op 形如 `VFromD<D> Set(D,T)`，`Lanes(d)` 取宽度。**关键：`Lanes(d)` 必须设计成可运行期求值**（定宽后端是 constexpr、SVE 后端是 `svcntw()` 运行期），否则将来加 SVE 要改 API。highway 已证同源码 x86/SVE 共存可行。**只照搬接口契约，不 vendor 整个 highway**（其 ops/ 全平台万行级、重宏元编程，违背"贴硬件/不过度抽象"）。给 `vec.h` 设硬上限（核心 op < ~2k 行/后端），超了就是丧失了不 vendor 的理由。
2. **数学核 vendor 现成库还是自抄多项式**？**vendor Highway + xsimd（均在 `third_party/`、无条件默认编进生产）**，逐 op best-of-breed 选最快核（exp/tanh→xsimd、log→Highway）；adapter 几乎零成本（吃裸 intrinsic 向量，HYLE `vec.raw` 直接喂）。两库精度同档（~3.5ULP，AI 热路径足够），互为校验。
3. **要不要 SVE/RVV**？**两者均已落地(simd-09,gtest+CMake gate+qemu 脚本固化)**:SVE ~76 op(qemu-aarch64 多 VL 128/256/512)、RVV 核心 op(qemu-riscv64 rv64gcv)。原判"未来项缓实现",用户 2026-07-09 要求做全(通用 infra 库=能力完整性)→ 验证 sizeless 非架构冲击后落地(见上 SVE/RVV 节)。D-tag 的 `Lanes(d)` 运行期口子 + `HYLE_SIMD_HAVE_SCALABLE` 宏是当初预留、现在兑现的接口。RVV(`backend/rvv.h`)**已镜像落地**(见下 RVV 节,riscv64-gcc + qemu-riscv64,rv64gcv,gtest 固化)。
4. **dtype 层借谁**？**自造，但参照 highway 的 `PromoteTo/DemoteTo` intrinsic 用法**（F16C `_mm_cvtph_ps`、AVX512-BF16 `cvtne2ps`、NEON-BF16）——照抄 intrinsic 不照抄抽象。Highway/xsimd 数学核无 fp16/bf16，唯 highway 当原生 lane 类型且硬件路径全；但 dtype 逻辑不复杂（bf16=fp32 高 16 位移位），自造避开 vendor 整个 highway。

## 参考

- [C++26 Shipped a SIMD Library Nobody Asked For（std::simd 表达力批判）](https://lucisqr.substack.com/p/c26-shipped-a-simd-library-nobody)
- [Highway：可移植 + 运行时 dispatch SIMD](https://google.github.io/highway)
- [xsimd：SIMD wrapper + 优化数学函数](https://github.com/xtensor-stack/xsimd)
- [Evaluation of C++ SIMD Libraries（AVX2/AVX512/SVE/NEON 基准）](https://www.mnm-team.org/pub/Fopras/rock23/PDF-Version/rock23.pdf)
