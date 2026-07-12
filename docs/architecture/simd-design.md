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

> **现状与决策（2026-07-12 评估）——HYLE 出厂用函数指针模型，变体-DLL 有意不做**：
> - **出厂 = 函数指针模型**（`dispatch_ops.cc`：`static const Fn fn = resolve_*(); return fn(...)`，首次 CPUID 选一次并缓存，之后每次调用一次可预测间接调用）。这与 Highway `HWY_DYNAMIC_DISPATCH` 同档——**runtime dispatch 维度 HYLE 平 Highway，不领先也不欠**。
> - **变体-DLL 评估结论：不做（收益≈0、成本中高、属错配复杂度）**。决定性事实：**全部 139 个公开 simd op 都是 whole-buffer 批量 op**（`dot_f32(x,y,n)`/`exp_f32_batch(src,dst,n)`/…，内部循环处理 `n` 元素）。那"一次间接调用"是**每 batch 一次**，被数千 cycle 的 kernel 彻底摊销——实测 dot 42 G op/s、saxpy 39 G 平 Highway，间接调用从不是瓶颈。变体-DLL 只对**细粒度、被内层循环密集调用的 op**（如 `Vec::Add` 每寄存器一调）有收益，而 HYLE 的细粒度 op 全是 `vec.h` header-only 模板（编译期单态化、随消费者 TU 内联、零间接、**根本不过 dispatch**）。过 dispatch 的只有批量 op → 要消除的间接不在热路径上。
> - **成本侧**：拆 N 份独立 DLL（每 ISA 全套 kernel）+ 启动 score/dlopen 层 + 139 符号跨平台（`LoadLibrary`/`GetProcAddress` vs `dlopen`/`dlsym`）绑定 + 测试矩阵翻倍；且 **DLL 边界过不了内联**，若将来导出细粒度 op 反而可能比函数指针更慢。
> - **触发条件（两者皆须满足，HYLE 当前都不满足）**：① 分发单一预编译二进制给未知 CPU（不许消费者自编）；② 消费者热循环密集调用细粒度 op。真要"进去后零间接"，更省的是**构建期 `-march=native` 静态路径**（已支持，平 `HWY_STATIC_TARGET`），非变体-DLL。
> - **一句话**：这不是"欠 Highway 一个变体-DLL"，而是"当前函数指针模型对批量 op 面已最优，变体-DLL 是错配复杂度，有意不做"。若 op 面将来新增细粒度导出 op 或启动二进制分发需求，再重启此评估。

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
2. ✅ SSE4.2 / AVX2 / AVX512 / NEON / SVE / RVV 后端 + 运行时 dispatch。四定宽 ISA 均真跑；AVX512 由 Intel SDE 跑 `vec_test` 全绿并验证 dispatch 选中 kAvx512。SVE/RVV 用 sizeless traits + 运行期 `Lanes`，P0-P4 共享泛型已补齐；SVE 19 tests×VL 128/256/384/512，RVV 18 tests×VLEN 128/256/512/1024。partial tag 的 `Rebind+kPow2+cap` 在 vec.h 共享层表达，后端用 predicate/`vl` 执行。
3. ✅ dtype 转换（fp16/bf16 硬件路径）。**simd-05。** f32→bf16 的 **AVX-512-BF16 native `vcvtneps2bf16`** 档随 AVX512 变体 SDE 可真跑(此前仅 objdump 验)。
4. ✅ 数学函数（借 Highway/xsimd 核）。simd-06 的 exp/log/tanh、simd-09 的 11 个 best-of 函数，以及 GELU/GELU-tanh/softmax/log-softmax 组合核均已落地并对拍 libm/标量参考。`hypot_f32_batch` 保留 Highway exponent-scaling 的完整语义；另显式提供 `hypot_in_range_f32_batch` 借 xsimd direct-formula 核，由调用方承担输入域契约，AVX2/NEON 真机均约快 2.0×。f64 direct-formula 未测出可判定收益，故不暴露 fast 变体。
5. ✅ `ops/`：dot_f32/saxpy_f32 跨 scalar/SSE4/AVX2/AVX512 dispatch;**AVX512 派发档 simd-09 Intel SDE 真跑坐实(tier 探针确认 kAvx512,非仅 objdump)**。reduce/对齐 memcpy/SwissTable 探测片段按需(container 已自有 SwissTable 探测)。
6. ✅ 单测：每后端对拍 scalar 参考值、数学函数 ULP 测试、benchmark 对标 Highway/xsimd。SSE4/AVX2/NEON 有真机性能数据；AVX512-SDE 与 SVE/RVV QEMU 只作正确性门，不作性能证据。math ULP vs libm 约 3.5，best-of 由真机 bench 决策。

### AVX512 后端：已实现并由 Intel SDE 真跑

> AVX512 使用后端自有 `Mask512` 包装 k-register，公开 D-tag/op 签名不变；无须改造 vec.h mask 模型。无 AVX512 真机不降低能力标准，Intel SDE 是执行闸门。

**已实现**（`backend/avx512.h` + `ops/kernels_avx512.cc`）：512 位 AVX-512 后端覆盖全部 lane/op 契约。验证包括 GCC/MSVC 原生编译、objdump 的 zmm/k-register 指令，以及 Intel SDE `vec_test` 全绿；运行时 tier 探针确认选择 kAvx512。

**objdump 实证**（铁证,非空谈,双平台）：backend dot→`vfmadd231ps %zmm`+`vextractf64x4` reduce；relu→`vcmpgt_oqps → %k1`+`vmovaps %zmm{%k1}{z}`(**k 寄存器掩码路径通**)；unsigned count→`vpcmpnleud → %k0`(原生无符号比较,无 xor-MSB)+`kmovw`+`popcnt`。派发 kernel TU(`kernels_avx512.cc.obj`):MSVC 33 条 zmm(`vfmadd213ps zmm`/`vmovups zmm`)、GCC-14 9 条 zmm(`vfmadd231ps zmm`/`vmovups zmm`/`vbroadcastss xmm→zmm`/`vextractf64x4 zmm→ymm`);两编译器 codegen 计数不同(GCC reduce 尾降 ymm/xmm),但核心 512-bit FMA/load 都真发。

**Mask 模型**：vec.h 的 `Mask` 由每个后端自定义；AVX512 的 `Mask512` 直接包 `__mmask16/8`，op 签名不变。比较产出 k-register，`IfThenElse` 用 mask blend，`CountTrue` 对位掩码 popcount；unsigned 比较直接使用原生 epu32 mask 指令。

**执行边界**：SDE 提供正确性与 dispatch 验证，不用于性能结论；AVX512 性能仍只接受真机 benchmark。

**派发现状**：`kernels_avx512.cc` 已接入 `dispatch_ops.cc::selected_tier()`；具备完整 ZMM OS-state 与 ISA 特性的主机默认派发到 AVX512 kernel，否则回落 AVX2/scalar。SDE 双向探针已验证 tier 选择。

### AVX-512 dtype/math 上架

dtype 与 math 的 AVX-512 路径已上架；本机虽是 AVX2-only，512-bit 代码除双平台编译/objdump 外，还由 Intel SDE 执行正确性与 tier 选择测试。

- **① dtype `f32→bf16` native(`cvt_avx512bf16.cc`)**:`_mm512_cvtneps_pbh`(`VCVTNEPS2BF16`)把 AVX2 路径的 ~7-op RNE 序列(bias+shift+packus+permute)折成**单条**硬件 RNE 指令,16-wide。`dtype_dispatch` 的 `f32_to_bf16_batch` 优先 `avx512bf16` 档(`cpu_features().avx512bf16` 门控,该 flag 在 `dispatch.cc` 已 AND-fold XGETBV/ZMM),回退 AVX2→scalar。`bf16→f32` 无 512 win(纯 `<<16` 移位,AVX2 已最优),不加 512 变体。新 `AVX512_BF16` CMake tier(GCC `-mavx512bf16` 叠加四件套;MSVC `/arch:AVX512` 已含,probe 验)。**验真**:GCC14 `vcvtneps2bf16`(EVEX.512)、MSVC `vcvtneps2bf16 ymm,zmmword`。测试 `F32ToBf16Avx512MatchesScalar`(finite bit-exact、NaN equal-by-class——硬件 quiet 到自有 QNaN,同 F16C fp16 立场),本机无 bf16 故正确 SKIP。
- **③ math 512b 变体(`transcendental_{xsimd,highway}_avx512.cc`)**:借同核、仅编译宽一档→zmm。`math_dispatch` 加 avx512 档优先(exp/tanh→xsimd-512、log→highway-512),回退 AVX2→scalar。新 `hyle_set_avx512_math_isa`(全 AVX-512 组**含 cd**——xsimd 的 avx512dq 核 `#error` 要求 cd;**+ AVX2 baseline**——Highway 静态 dispatch 需完整宏链选 AVX3,否则降级)。**验真**:GCC xsimd512 92zmm/highway512 43zmm、MSVC xsimd512 581zmm。
  - **MSVC Highway-AVX3 排除(probe 实锤,非接线 bug)**:Highway 在 `detect_targets.h:192` 把整个 AVX3 标 `HWY_BROKEN_MSVC`(MSVC AVX-512 codegen 已知 bug),故 `/arch:AVX512` 下 `hn::` 仍发 ymm(dumpbin 250 ymm/0 zmm)。MSVC 上**没有 512-bit log win**(avx2 highway log 已是 Highway 能给的最宽)。故 `transcendental_highway_avx512.cc` 与 dispatch 的 log-512 分支用 `#if !defined(_MSC_VER)` 守卫,MSVC log 留 avx2 highway,**不路由到 fake-512**。xsimd(exp/tanh)不受影响——其 512 batch 是纯 AVX-512F,MSVC 真发 zmm。
- **执行验证**：dtype/math/ops 的 AVX512 路径均经 Intel SDE 正确性门；SDE 不作为性能背书，性能数据仍要求真机。

### op 覆盖补齐（2026-07,compute 前把 vec 后端 op 集补全）

起点是 ~35 个 op(Set/Load/Store/算术/Min/Max/Neg/Abs/Sqrt/MulAdd/位运算/比较/IfThenElse/AllTrue-CountTrue/Reduce{Sum,Min,Max}/GetLane)。相对 Highway ~200 op 的差距按**优先级分批**补全,四后端(sse4/avx2/avx512/neon)+scalar 参考,**抄 Highway 的 per-op intrinsic 序列(契约,非 vendor 库)**,每批独立真机验证(WSL x86 跑 scalar/sse4/avx2、编 avx512;mac SSH 跑 NEON;三平台 gtest 各 136+ 全绿):

- **批1 移位** `ShiftLeft<k>/ShiftRight<k>`(编译期立即数):x86 无 epi8 移位(epi16+mask 模拟)、avx2/sse4 无 `srai_epi64`(srli+符号填充)、avx512 原生、NEON 全宽原生;**i8 算术右移**要 logical-byte-shift+cmpgt 符号填充(naive slli-to-high-byte 错——已修+测)。
- **批2 饱和/平均** `SaturatedAdd/Sub`(8/16-bit)+`AverageRound`(u8/u16):x86 `adds/subs/avg` 原生、NEON `vqaddq/vqsubq/vrhaddq`、scalar 宽类型算后 clamp。
- **批3 同宽转换** `ConvertTo/RoundToInt`(f32↔i32 量化(de)quant 核):x86 `cvttps/cvtps/cvtepi32`、NEON `vcvtq/vcvtnq`。
- **批4 浮点舍入** `Round/Floor/Ceil/Trunc`:x86 `round_ps/pd`、avx512 `roundscale`、NEON `vrndnq/mq/pq/q`、scalar `nearbyint/floor/ceil/trunc`。
- **批5 掩码/尾部** `FirstN/MaskedLoad(Or)/BlendedStore/LoadN/StoreN`(向量化循环余数,std::simd 表达不了的 90% 之一):avx512 原生 k 掩码 fault-safe;avx2/sse4/neon 内存暂存 fault-safe(只碰 on-lane / 前 k 个 lane)。
- **批6 reduce 扩展 + Broadcast** `ReduceMul/ReduceAnd/ReduceOr`(avx512 32/64-bit 原生 `_mm512_reduce_*`、余宽 store+fold)+`Broadcast<i>`(lane splat)。
- **批7 变宽转换** `PromoteLowerTo/PromoteUpperTo`(满 narrow→满 wide-半)+`DemoteTo`(2→1 饱和 pack)+`ReorderDemote2To`:量化核。x86 `cvtep*`/`packs`(avx2 4-way permute 修块序、avx512 `permutexvar_epi64` 修 4 块交织,与 Highway kIdx 同)、NEON `vmovl_/vmovl_high_`/`vqmovn`+`vcombine`(天然有序)。**纠正了先前"需 partial 向量所以后置"的误判**——half-at-a-time 形式完美契合定宽模型,是 HYLE 原生首选。**Highway 的单向量便利形**(`PromoteTo(dw,v)`/`DemoteTo(dn,v)`,Rebind-tag、同 lane 数半寄存器)后来也补成薄封装(2026-07-12 d488813:`DemoteTo(dn,v)=LowerHalf(DemoteTo(Twice<dn>,v,v))`、`PromoteTo(dw,v)=PromoteLowerTo(ZeroExtendVector(v))`,加 `ShiftRightAndDemoteTo<k>`);返回 partial 向量,读回用 `DFromV`。
- **批8 变量移位 + 加宽乘** `Shl/Shr`(每 lane 计数;avx2 `sllv/srlv/srav`+sse4 无原生全 fallback+avx512 全宽原生含 srav_epi64+NEON vshlq 负计数)+`MulHigh`(i16/u16)+`WidenMulPairwiseAdd`(i16→i32 madd,int8/16 GEMM 核)+`AbsDiff`。
- **批9 swizzle 全套** `OddEven/Reverse/Reverse2/InterleaveLower/Upper/Concat{Lower,Upper}²/ConcatOdd/Even/DupOdd/Even/TableLookupLanes+IndicesFromVec`。**关键:InterleaveLower/Upper 是每 128-bit 块**(Highway 契约,256b avx2=2 块);avx2 块内原生(unpack/blend/permute2x128)+跨块难的 store+reload,sse4/neon 单块 store+reload,avx512 4 块块内。
- **批10 gather/scatter + compress/expand** `GatherIndex/ScatterIndex`(embedding lookup 核)+`Compress/CompressStore/Expand`(稀疏过滤核)。**性能档已收口(见下 op-parity 节)**:avx2 GatherIndex→原生 vpgatherd、Compress→packed 索引表+vpermd、avx512→vpcompress/vpexpand/vpgather、sse4 Compress→pshufb 字节索引表;仅无硬件的宽度/后端保留 store+reload(同 Highway)。
- **批11 位计数 + 符号 + 杂项** `PopulationCount/LeadingZeroCount/TrailingZeroCount`(**已全换原生**:vpshufb 半字节-LUT popcount、float-exp trick LZCnt/TZCnt、NEON vcntq/vclzq)+`Not/Clamp/RotateLeft/RotateRight/CopySign/BroadcastSignBit/IfNegativeThenElse/ZeroIfNegative`(全由现有 op 组合,四后端同构)。

**op 覆盖现状**:量化(promote/demote/convert/满足 int8 GEMM 的 madd/mulhigh)、通用向量循环(移位/舍入/饱和/掩码尾部/reduce 全家族)、swizzle(interleave/concat/reverse/permute)、gather/scatter/compress/expand、位计数/符号——**全部四后端 + scalar 覆盖,三平台真机对拍 scalar oracle**。设计原则(用户 2026-07-06 纠偏):**高性能 infra 库的消费面最广,能力要提供到位,不能因"暂无消费者"后置**。**性能档也已收口(2026-07-09 op-parity,见下节)**:曾经 store+reload/std 标量的慢 op 全换原生 intrinsic 并对拍 Highway 拉到统计打平——能力面 + 性能面双达标。各后端实测 LOC(2026-07-12):scalar 2058 / sse4 2617 / avx512 2913 / avx2 2943 / neon 3008。**原始 ~2k 硬上限已被"做全 Highway op 面"的能力完整性指令突破**(op ~228/后端 + Crypto/complex/best-of 数学)——权衡是明确接受的:即便 3k 仍远小于 Highway 每-target ops/(万行级重宏元编程),不 vendor 的理由(贴硬件、可读、可控)依然成立。上限从"硬阈值"降级为"警戒线":单后端逼近 ~3.5-4k 时应重新审视是否有该拆的泛型层组合。

### op-parity 性能收口(simd-09 尾,2026-07-09)——自造件 vs Highway 同 ISA 同 op

用户核心担忧:"自己手写了 ~228 op/后端,性能不如抄 Highway 就很愚蠢"。建 `bench/simd/op_parity_bench.cc`(HYLE vs Highway vs xsimd,**同 TU/同 ISA/两边 header-only 都内联**=真 codegen 差,非内联不对称;`bench-stat.py -n15` 看中位倍数,gap>两变体 spread 和才判赢/输)。

**结论:9 个 store+reload/std 标量慢 op(GetExponent/DupEven-Odd/ReverseBits/PopCount/LZCnt/TZCnt/Compress/ConcatEven-Odd)全换原生 + 2 个残留 2.x 输家(LZCnt/Compress)也追平 → 四固定宽后端全部 ≥ 统计打平**:
- 手法:vpshufb 半字节-LUT(popcount/reversebits)、位提取指数域(GetExponent)、**float-exp trick**(LZCnt=min(158-biased_exp,32)、TZCnt=min(be(x&-x)-127,32),`cvtepi32_ps` 取指数)、packed 索引表+vpermd / pshufb 字节表(Compress)、vpermd/shuffle_ps/vuzp/vtrn(Concat/Dup)。NEON 硬件原生最省(vcntq/vclzq/vrbitq/vuzp/vtrn)。
- **Compress 契约变更**:HYLE 旧契约保证不活跃尾部清零(强于 Highway"unspecified"),该保证在 AVX2/SSE4(无清零-compress 指令)是 ~2x 软件税。核实唯一消费者 vqsort 自带 Highway 不依赖 + 标准用法(`CompressStore`/前 `CountTrue`)不碰尾部后,**契约改为尾部 unspecified**(对齐 Highway/xsimd),各 target 尾部行为可不同(avx512 maskz 硬件清零、store+reload 路径恰好清零、avx2/sse4 vpermd/pshufb 不清零)。
- SVE/RVV 无需改(SVE 早用原生 svcnt/svclz/svuzp;RVV popcount/clz 属 Zvbb 扩展,rv64gcv 无,登记边界同 Highway)。详见 `docs/performance/benchmarks-simd.md` op-parity 全表。

### P3 定宽 Masked* 补全与 complex 直接公式

泛型层新增24个可独立落地的 Masked* 名字，覆盖 variable/immediate shift及Or形、masked reduce、approximate reciprocal/rsqrt、convert/LZCnt、Q15、widen pairwise multiply-add、5个complex形态及Sat短名alias。非Or形false lane为0，Or形取`no`；compute-family维持HYLE既有`(mask, no, ...)`参数顺序。`MaskedModOr`已在 P4 复用 i32/u32 invariant `Divisor` 的 `Mod` 落地，仍不以逐lane标量循环补名。

性能闸发现旧`MulComplexConj(a,b)=MulComplex(a,ComplexConj(b))`在Apple NEON令`MaskedMulComplexConjAdd`慢Highway 18.5%。底层`MulComplex/MulComplexConj`改为`DupEven/DupOdd + MulAdd/Sub + OddEven`直接公式，减少conj/reverse/addsub中间重排并允许FMA；P3 24/24公开名完整复测全部统计打平，该项为0.204 vs 0.204 ns/op。该底层变更同时通过SSE4/AVX2/AVX512-SDE/NEON、SVE/RVV多VL和ASan回归。

### P4 定宽 Mul / Mod / Magnitude

P4 新增12个公开名：四个 FMA 符号/交错变体、`MinMagnitude/MaxMagnitude`、`MulEven/MulOdd`、packed-bf16 的`MulEvenAdd/MulOddAdd`和 i32/u32 invariant `Mod/MaskedModOr`。浮点 FMA 变体复用 `MulAdd` 保留单次舍入，整数通过无符号位运算保持模语义；signed magnitude 不会在`INT_MIN`取绝对值时溢出。浮点 equal-magnitude tie 依赖目标 ISA 的 `Min/Max`，并逐 ISA 对齐 Highway：AArch64 NEON 用`vminnmq/vmaxnmq`，异号零分别为`-0/+0`。

`MulEven/MulOdd` 泛型路径是交替 lane promote 加 multiply；i/u32→i/u64 由固定后端原生 widening multiply 接管。`Mod` 复用已有 Granlund-Montgomery reciprocal multiply，预计算`Divisor`不进入批量循环，余数为`a-q*d`；`INT_MIN/-1`按 HYLE 模 quotient 约定给出零余数。NEON 的 signed high-product 采用 Highway 同形`vmull[_high]+vuzp2`，消除多余输入重排与窄化。

所有P4批量路径保持 Θ(n) 读写下界。Windows AVX2 三组独立`N=15`对照的`Mod`、`MaskedModOr`、`MulEven`、`MulOdd`均为统计TIE；Apple NEON三组独立`N=15`稳定复现前两项分别1.073x/1.058x WIN、两个 widening multiply TIE。P4 后主施工闭集为77/92，剩余软件项仅为 scalable P5。

### simd-12 热点核与小调用路由

- **bit-pack 11/13**：保留tier-dependent 8+3/8+5 block layout，自有核修正Highway 13 pack的`0x7C00<<3`截断错误、13 inverse错源与11/13 rawF先移后掩码。独立标量layout oracle + 全值域对拍；大N相对scalar约14~32x。`count<128`在baseline直接执行相同scalar tail，消除无向量工作时的跨TU税。raw/packed范围不允许重叠。
- **SoA `csin/ccos`**：AVX2/NEON走`real sincos + expm1稳定sinh/cosh`。tiny imaginary不消减；±0、NaN/Inf与有限overflow只逐lane调用`std::complex`，regular邻居继续向量化；先保存异常lane结果再写输出，支持精确原位alias。Windows AVX2 random大N约1.28x胜xsimd；WSL与Apple NEON对xsimd统计打平，NEON相对原生产scalar约2.2x。
- **small-K sort**：VQSort已有small-N network，不重造。整数/KV `partial_sort(k<=16)`走实测更快的std核，`k>16`回VQPartialSort；浮点不分流。scalar浮点比较器与VQSort统一为“有限值按方向，NaN始终末尾”的严格弱序。失败的有序前缀候选只保留benchmark负结论。

三套scaling bench均把public/direct固定税与candidate/vendor算法差分开，支持N/K/分布/方向筛选；生产只保留正确且跨平台没有确认回退的路径。

### simd-13：继续竞争 vendor complex 核

- **`ctanh`**：普通 lane 计算一次 `sincos(2y)`，用 `expm1(|2x|)`推导小实部的双曲对；整块安全 large-real lane直接写出极限值，混合 block 的`|x|>20` lane用`q=exp(-2|x|)`渐近式，避免双曲溢出。real/imaginary axis、subnormal、NaN/Inf、近极点及会让`2y`溢出的有限虚部逐 lane 回退`std::complex`，不会连带正常邻 lane。异常结果先保存、vector store后覆盖，所以只支持的精确原位 alias 仍正确。
- **`cpow`**：对 regular lane 计算一次`rho=hypot(z)`、`L=log(rho)`、`phi=atan2(z)`，再融合`A=w_re*L-w_im*phi`、`B=w_re*phi+w_im*L`，只做一次`exp(A)`和一次`sincos(B)`。base/exponent 的轴、subnormal、非有限、过大值、危险指数窗口与 branch-cut lane 均回退`std::pow`。
- `ctanh/cpow`的 xsimd core 仍只保留在同 TU benchmark 中作公平对照；生产 AVX2/NEON 已走 HYLE SoA。`cexp`已同构无重写价值，`clog/csqrt`需要完整缩放/相位体系而没有可证结构性收益，继续保留 xsimd；VQSort、VQPartialSort、VQSelect 和除u16 11/13外的 bit-pack 同样继续保留 Highway。完整裁决见[`simd-13`](../milestones/simd-13-vendor-displacement-2.md)。

### SVE scalable 后端(simd-09,首个非定宽后端)——sizeless 类型如何融入 D-tag 契约

用户"SVE/RVV 不假设向量长度,为啥冲击架构 / 模板元编程无法解决么"——**实证核查后纠正了一度过度渲染的"冲击全库地基"结论**:SVE/RVV **不冲击架构**,是 `backend/sve.h` 内部**模板元编程可解**的问题(qemu 多 VL 真跑坐实)。

**为什么不是纯"独立 branch"、又为什么不冲击架构(两个都对一半)**:
- 后端选择**确实是独立分支**(vec.h ladder 加一个 `#elif HYLE_SIMD_TARGET_SVE`,CMake 加 tier)——这部分隔离。
- 真正的挑战是 **sizeless 类型**:SVE 的 `svfloat32_t`/RVV 的 `vfloat32m1_t` 按 ACLE **不能做 sizeof / 结构体成员 / 数组元素 / 全局变量**。而定宽后端的 `Vec` 是 `struct { float32x4_t raw; using PrivateT; kPrivateN; }`——把寄存器当结构体成员。
- **但 `DFromV`/`Vec` 每后端各自定义(非全库共享契约)**:一个 TU 只 include 一个后端头,后端从不共存 → sve.h 可自定义一套,不动其他后端或 vec.h 共享层。

**模板元编程解法**:
1. `VFromD<D>` 用 traits 映射 `detail::SveVecT<T>::type`(替代 struct 成员包 raw)——sizeless 类型能做函数**参数/返回值/局部变量**(ABI 合法),HYLE 的 op 全是 `V Add(V,V)` 传值/返回,天然不碰禁区。
2. `DFromV<V>` 用**反向 traits 特化表** `detail::DFromV_sve<V>`(替代 `V::PrivateT` 成员访问,sizeless 无成员)。
3. store+reload 的 `T buf[kMaxLanes]` 是**标量数组**(T=float/int,合法);`kMaxLanes` 从"精确"变"上界"(kNativeVectorBytes=256=2048-bit 上界),循环边界用运行期 `Lanes(d)`。SVE 的 gather/compress/scatter/predicated-tail 走原生 `svld1_gather`/`svcompact`/`svst1_scatter`/`svwhilelt`。
4. `Lanes(d)` 在 scalable target 运行期求值，并按 `kPow2` 缩放后受 `kMaxLanes` 截断；`Rebind<T,D>` 同步调整 `kPow2`，保证窄化后逻辑 lane 数不变。`FixedTag<T,N>` 因而仍是 `min(硬件 lanes,N)`；接受 D-tag 的 load/store、masked/gather/scatter、reduce、reverse、slide、whole-vector swizzle 与 widen/demote 只观察逻辑 active lanes。只接收 native `V` 的重载按 full-vector 契约运行，因为 sizeless `V` 本身不携带 partial tag。正 `kPow2` 只有在最小 128-bit 原生向量容不下目标 lane 数时才表示 grouped vector；SVE/RVV 当前单寄存器后端对此 fail-fast。

**验证(qemu-aarch64,同一二进制强制 VL 128/256/384/512)**：`tests/simd/sve_test.cc` 的 **19 个 gtest × 4 VL 全绿**；首项用 `PR_SVE_SET_VL` 设置并回读实际 VL，避免 CPU model 参数与进程当前 VL 脱节。覆盖十种 lane、P0 泛型消费者、P1 swizzle/分段 load-store、P2 widen/quantize/reduce、P4 FMA/magnitude/widen/bf16/mod，以及 partial tag 的 masked/gather/scatter、widen/demote 与跨 signedness/Q15 极值。另在 VL 2048 真正设置后复验 P2/partial；384-bit 门覆盖非 2 次幂 lane 数。`scripts/sve-qemu-test.sh` 可复现正式四宽门。

**scalable P0-P4 当前能力面**:SVE/RVV 已具备泛型 complex/128-cmp/Newton/rotate 所需原语、whole-vector swizzle/table/slide、原生 2/3/4 路分段 load-store，以及 PromoteEven/Odd、Truncate/U8FromU32、跨 signedness Demote2、widen-dot、SumsOf2/4/8、任意 lane 数 ReduceMul、Q15 与 saturated abs/neg。量化/点积与结构 op 已和 HYLE 定宽后端对称；P4新增的FMA/magnitude、even/odd widen、bf16 accumulate和i32/u32 Mod也由共享泛型层直接实例化。剩余软件项仅为 gap inventory 的 P5。硬边界仍是 SVE2 crypto、RVV Zvbb 与超过 `uint64_t` 容量的 mask-bits 位域接口。

### RVV scalable 后端(simd-09,第二个 scalable 后端,镜像 SVE)

`backend/rvv.h` 复用 SVE 建立的 scalable 机制(`HYLE_SIMD_HAVE_SCALABLE`、运行期 Lanes、sizeless traits `RvvVecT<T>` + 反向表 `DFromV_rvv`)。原生类型保持 **LMUL=1**；逻辑 partial tag 用非正 `kPow2` + `vl` 表达，`Lanes(d)=min(vsetvlmax >> -kPow2,kMaxLanes)`，无需把 fractional-LMUL sizeless 类型扩散到公开 API。覆盖 Set/Load/Store/算术/位/比较→vbool/选择/mask/value reduce/移位及 P0-P4 的结构、量化与 Mul/Mod/Magnitude op。

**RVV 独特处(vs SVE)**:① **每 intrinsic 显式 `vl`**(SVE 用隐式谓词)——从 `Lanes(d)` 算一次穿进每个 op;② **unsigned lane 用独立助记符**(`vminu/vmaxu/vmsltu/vmsgtu` 非 `vmin` 泛型),signed/unsigned `if constexpr` 分派(这是 RVV 铺量的主要 gotcha);③ mask 是 `vbool<ratio>_t`(ratio=SEW/LMUL)。

**验证**:`tests/simd/rvv_test.cc` **13 个 gtest × VLEN 128/256/512/1024 全绿**；`P4MulModMagnitude`直接实例化共享FMA/magnitude/widen/bf16/mod层，1024-bit 门同时覆盖 unsigned `ReduceMin/Max` 助记符分派、tail-zero、oversize StoreN 截断与 swizzle。`scripts/rvv-qemu-test.sh` 固化同一二进制四宽门。**边界**:QEMU 当前最高 1024-bit，代码对 2048-bit u8 的 FirstN/FindLastTrue 使用不截断的 unsigned 逻辑；更宽 RVV 由 2048-bit HYLE 上界 fail-fast。Zvbb 位计数仍按硬件 gate 登记。

## 已定决策（1/2/3/4 均已落地）

1. **vector 抽象用定宽 `vec<T,N>` 还是 scalable（length-agnostic）**？**scalable-friendly 接口 + 先实现定宽后端**——接口按 length-agnostic 设计（参考 Highway，将来加 SVE/RVV 不改 API），但首版只实现定宽后端（x86 SSE4.2/AVX2/AVX512 + ARM NEON，它们本就定宽）。对冲：不堵未来又不先背复杂度。
   > **深读 highway 落地约束**：照搬 highway 的 **D-tag 接口契约**——传零大小标签 `D=Simd<T,N,kPow2>`，op 形如 `VFromD<D> Set(D,T)`，`Lanes(d)` 取宽度。**关键：`Lanes(d)` 必须设计成可运行期求值**（定宽后端是 constexpr、SVE 后端是 `svcntw()` 运行期），否则将来加 SVE 要改 API。highway 已证同源码 x86/SVE 共存可行。**只照搬接口契约，不 vendor 整个 highway**（其 ops/ 全平台万行级、重宏元编程，违背"贴硬件/不过度抽象"）。给 `vec.h` 设行数警戒线（初版 ~2k/后端；随做全 op 面已增至 2-3k，见上节实测），逼近 Highway 每-target 量级就是丧失了不 vendor 的理由。
2. **数学核 vendor 现成库还是自抄多项式**？**vendor Highway + xsimd（均在 `third_party/`、无条件默认编进生产）**，逐 op best-of-breed 选最快核（exp/tanh→xsimd、log→Highway）；adapter 几乎零成本（吃裸 intrinsic 向量，HYLE `vec.raw` 直接喂）。两库精度同档（~3.5ULP，AI 热路径足够），互为校验。
3. **要不要 SVE/RVV**？**两者均已落地并完成 scalable P0-P4 的共享泛型实例化**：SVE qemu-aarch64 VL 128/256/384/512，RVV qemu-riscv64 VLEN 128/256/512/1024；同一二进制逐宽对拍 scalar。D-tag 的运行期 `Lanes(d)`、partial `kPow2`/cap、sizeless traits 与原生 predication/`vl` 共同维持公开 API 不变。定宽 P3 Masked* 与 P4 Mul/Mod/Magnitude 已完成；剩余软件项为 P5，硬件 gate 以 gap inventory 为准。
4. **dtype 层借谁**？**自造，但参照 highway 的 `PromoteTo/DemoteTo` intrinsic 用法**（F16C `_mm_cvtph_ps`、AVX512-BF16 `cvtne2ps`、NEON-BF16）——照抄 intrinsic 不照抄抽象。Highway/xsimd 数学核无 fp16/bf16，唯 highway 当原生 lane 类型且硬件路径全；但 dtype 逻辑不复杂（bf16=fp32 高 16 位移位），自造避开 vendor 整个 highway。

## 参考

- [C++26 Shipped a SIMD Library Nobody Asked For（std::simd 表达力批判）](https://lucisqr.substack.com/p/c26-shipped-a-simd-library-nobody)
- [Highway：可移植 + 运行时 dispatch SIMD](https://google.github.io/highway)
- [xsimd：SIMD wrapper + 优化数学函数](https://github.com/xtensor-stack/xsimd)
- [Evaluation of C++ SIMD Libraries（AVX2/AVX512/SVE/NEON 基准）](https://www.mnm-team.org/pub/Fopras/rock23/PDF-Version/rock23.pdf)
