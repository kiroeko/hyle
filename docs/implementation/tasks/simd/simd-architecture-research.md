# SIMD 架构调研：什么架构最利于"覆盖面广 + 时间复杂度 SOTA"

> **目标（用户定，2026-07-11）**：HYLE simd 终极目标 = **覆盖面广** + **时间复杂度性能 SOTA**。本文调研 Highway / xsimd / std::simd 架构，对照 HYLE 现状，判定**根本解法**并给出演进方案。
>
> **修订说明（2026-07-11 二稿）**：初稿曾推荐"方案②partial-tag 全宽存储"。经深入核实 Highway 源码 + HYLE 冲突面，**推翻该推荐**——方案② 是绕过问题的 workaround，留永久语义债。**根本解法是方案①：分层 include 多宽度类型 + 宽度-SFINAE**（Highway 原型）。本稿据此重写。
>
> **终态说明（2026-07-16）**：本文是 simd-19 之前针对 partial-vector 表示的研究记录，不是当前部署
> 文档。实际 NEON 落地采用 `RawNeonNT<T,N>` D-band + 常绿垫片/热点 native-D，x86 采用低半消费指令与
> canary，而非机械照搬全文提议的所有 layered include。多版本派发也已由 simd-22 统一为 Runtime/native/
> exact；当前规范见 [`simd-dispatch-guide.md`](../../../developer/maintenance/simd-dispatch-guide.md)，实际表示层裁决见
> [`simd-arch-review-2026-07.md`](simd-arch-review-2026-07.md)。

## 0. 结论先行（根本解法）

- **根本解法 = 定宽后端分层 include（avx2.h→sse4.h→...）让 `Vec128/Vec256/Vec512` 在宽 target 里共存 + `Set/Zero` 等 D-dispatched 函数改宽度-SFINAE**。这是 Highway 的确切模型，`LowerHalf/Combine/Concat*/PromoteTo/宽度转换` 全族返回**真实体**，广度天花板 = Highway。
- **性能 SOTA 与架构正交**：单 kernel 速度靠每 ISA 独立 codegen、全内联固定目标 body 与算法 best-of。动态模式只在 whole-buffer 边界间接一次；exact/native 模式直连，**分层 include 不影响内核 codegen**。
- **HYLE 当时架构 = "单后端-per-TU + 单一 native 宽度"是早期图快的选择**（`feedback_no_early_shortcut_as_excuse`），是当时广度天花板的根因。批4 半向量族撞墙就是它。
- **代价 = 一次性有界重构**（~24 个 D-dispatched 函数改 SFINAE + detail 去重 + 分层 include），**非永久债**。不推翻变体-TU/dispatch/scalable 任何地基。

## 1. 三库架构对比

### 1.1 Highway：分层多宽度类型 + foreach_target（根本解法原型）

**类型系统**（shared-inl.h:218）：`Simd<Lane, N, kPow2>` 一个模板表达所有宽度；`kPow2` 是缩放因子（half=`kPow2-1`）。**两条正交的宽度轴**：
- **轴A 宽度实体**：`Vec128`/`Vec256`/`Vec512` 是**不同类型**，通过**分层 include** 共存一 TU：`x86_512-inl.h:84 #include x86_256-inl.h`，`x86_256:84 #include x86_128`（实测）。
- **轴B 宽度内 partial**：`Vec128<T, N/2>`（同寄存器少 lane，sub-128）。

**为什么合法共存一 TU**：AVX512 机器合法执行 SSE/AVX 的 `_mm_*`/`_mm256_*` 指令。所以宽 target 编译时窄类型的 intrinsic 直接可用（`-mavx512` 下 `_mm_add` → VEX/EVEX-128，零惩罚）。

**半向量返回真实体**（x86_256:4739）：`LowerHalf(Vec256<float>)→Vec128<float>{_mm256_castps256_ps128(v.raw)}`——`Vec128` 就在同一 TU（layered），直接返回**真 128 位窄向量**，不是打 tag 的全宽壳。

**D-dispatched 冲突如何解**：`Set(D)/Zero(D)` 同名同参 `(D)`，靠**宽度-SFINAE** 区分——`HWY_IF_V_SIZE_D(D,16)` 约束的 `Zero`→Vec128，`,32`→Vec256，两者**共存不冲突**（constrained overload，非返回类型重载）。`VFromD<D>=decltype(Zero(D()))` 自然按 D 宽度解析到对应 Vec。

**多 ISA 编译**（foreach_target.h）：同源 re-include 多次编各 ISA namespace。**运行期 dispatch**（highway.h）：`HWY_DYNAMIC_DISPATCH` fn-ptr 表选最优 ISA。

**代价**：类型系统重（Vec128/256/512 × partial × 12 类型），但**广度天花板最高**，任意 op 任意宽度/半宽都干净表达。

### 1.2 xsimd：`batch<T, Arch>`

`batch<T, A>` = lane 类型 + arch，宽度由 (T,A) 隐式固定。**半向量非一等公民**（靠 batch_cast/slice 有限支持）。数学库定位，结构 op 弱。

### 1.3 std::simd（P1928）

`simd<T, Abi>`，ABI 参数化，`split`/`concat` 有半向量但 op 面最窄（无 gather/compress/complex），广度不足以对标 Highway。

### 1.4 HYLE 当时现状

**抄了 Highway 的 tag 层但没抄 Vec 存储层**：
- tag `Simd<T,N,kPow2>` 含 `Half/Twice`（vec.h:59）✅ ——**类型 tag 层能算出半宽**。
- 但 Vec 存储是**单后端单宽度**：avx2 TU 只有 `Vec256{__m256 raw}` + `static_assert(kNativeWidth)`（avx2.h:101）。**sse4.h 不在 avx2 TU**（vec.h:225-239 `#elif` 择一），`Vec128` 不共存 → `LowerHalf` 无处返回。
- **裂缝**：tag 说"有半宽"，Vec 说"存不下"。这就是广度天花板。

**当时已 SOTA、后来继续保留的部分**：每 ISA 独立 codegen、whole-buffer 函数表派发、scalable kMaxLanes + 运行期 `Lanes(d)`、per-op best-of。simd-22 在其上补了 exact/native 入口，没有改写表示层结论。

## 2. HYLE 分层 include 的冲突面（实测，有界）

核实三定宽后端（sse4/avx2/avx512）层叠后的冲突，**分三类**：

| 类 | 内容 | 冲突？ | 解法 |
|---|---|---|---|
| **Vec-typed ops** | `Add(Vec128)` vs `Add(Vec256)` 等 162 个（sse4）| ❌ 天然重载安全 | 不动——不同 Vec 参数即不同重载 |
| **T-only detail 助手** | `kSupported/kI8/kI16/kI32/kI64/kSigned`（只依赖 T，各后端**定义相同**）| ⚠️ 同名同义→ODR 重定义 | 提取到共享头 `backend/x86_detail.h`，三后端 include 一次 |
| **width detail** | `kNativeWidth<D>`（sse4==16/avx2==32/avx512==64）| ⚠️ 同名异义 | **删除**——被宽度-SFINAE 取代 |
| **Raw/V/M 别名** | `Raw128/Raw256`、`V128/V256`、`Mask128/Mask256` | ❌ 名字不同 | 不动 |
| **D-dispatched** | `Set(D)/Zero(D)` ~24 个，同参 `(D)` 异返回 | ⚠️ 硬冲突（不能按返回重载）| 改**宽度-SFINAE**（`enable_if D::MaxBytes()==16/32/64`）|
| **类型别名** | `VFromD/DFromV/TFromV` | ⚠️ 重定义 | 定义一次——`DFromV/TFromV` 已是 width-agnostic（作用于 `V::PrivateT`）；`VFromD=decltype(Zero(D()))` SFINAE 后自然解析 |

**结论**：冲突集中在 ~24 D-dispatched 函数 + detail 去重，**有界可解**，正是 Highway 已解过的同一问题。HYLE 的 `D::MaxBytes()`（vec.h:55）已就位，宽度-SFINAE 直接可用。

## 3. 性能：分层 include 零影响单 kernel SOTA

- 单 kernel 靠①每 ISA -march 编译 ②全内联零封装 ③intrinsic 直发 ④算法 best-of——**分层 include 不碰任何一条**。
- 变体-TU 正交性核实：当前由 `hyle_add_dispatched_kernel` 在构建目录生成薄 AVX2 TU，只定义 target/suffix 后包含同一 `kernel_impl.h`。层叠让 AVX2 TU 多见 Vec128 不改变 Runtime 选 ISA。
- 半向量 op 的 permute 成本（如 `UpperHalf` 的 extracti128）与 Highway 同——非 HYLE 独有劣势，且返回真实体后**下游 op 全宽度正确**，无"逻辑半/物理全"错位的隐藏成本。

## 4. 演进方案（根本解法 vs 被否决的 workaround）

### 方案①（根本解法）★采纳：分层 include 多宽度类型 + 宽度-SFINAE
- avx2.h `#include "sse4.h"`；avx512.h `#include "avx2.h"`（`#pragma once` 已在，additive 安全）。NEON 独立（aarch64），半宽用 `Vec64`。
- `Set/Zero/D-dispatched` 从 `static_assert(kNativeWidth)` 改宽度-SFINAE overload（Highway `HWY_IF_V_SIZE_D` 模型），三宽度共存 D 选之。
- T-only detail 提共享头去重；`kNativeWidth` 删除；`VFromD/DFromV/TFromV` 定义一次。
- 半向量族 `LowerHalf/Combine/...` 返回**真实体 Vec128/Vec64**。
- **利**：广度天花板 = Highway，类型系统自洽，接口对齐，无永久债。**弊**：一次性重构 ~24 函数/后端 + detail 去重。**这是根本解。**

### 方案②（否决）：partial-tag 全宽存储
半宽 = 全宽物理 + 逻辑半（tag 说"当我一半"）。
- **否决理由**：留**永久语义债**——每个消费半宽的 op 要记"上半是垃圾"，`Store` 要 partial，`Combine` 要 permute 拼装，未来每个宽度转换 op 都在"物理全/逻辑半"错位里绕。是**假半宽**，不是 Highway 的真实体。初稿误推荐，此稿否决。

### 方案③（否决）：memory-oriented 签名
`StoreLowerHalf` 等绕开半宽类型。偏离 Highway 契约，消费者不能互换 = **假广度**。否决。

## 5. 落地顺序（grok 执行到底，opus 验证）

1. **批0（根本重构，最先，是使能器）**：分层 include + 宽度-SFINAE + detail 去重。**不加新 op**，只重构类型地基。**正确性判据 = 现有 vec_test 全后端 100% 绿（零回归）**。这是风险最高的一步，opus 重点验。
2. **批1/2/3/5（广度，与批0 正交但落在其上）**：Shuffle/字节移位、*Same/Pairwise、Concat scalable、复数 ARM+偏门。全宽 Vec→Vec / scalable，天然重载安全。
3. **批4（半向量族，批0 之上变平凡）**：`LowerHalf/Combine/ZeroExtendVector/...` 返回真 Vec128/Vec64——批0 铺好后就是照抄 Highway `_mm256_castsi256_si128`/`set_m128i`。
4. **不动**：变体-TU、whole-buffer dispatch、best-of、crypto/数学算法与 scalable kMaxLanes；simd-20 只摘除产品 vendor 依赖，simd-22 只统一部署入口。

## 6. 一句话总结

**广度靠类型系统的宽度参数化（Highway 分层多宽度实体 + 宽度-SFINAE），性能靠编译隔离+算法（HYLE 已 SOTA）。根本解法是让 HYLE 定宽后端分层 include 使 Vec128/256/512 共存、D-dispatched 函数改宽度-SFINAE，放开早期"单一 native 宽度"约束——半向量/partial/宽度转换全族返回真实体、接口对齐 Highway，达成真覆盖面且零伤性能。方案②partial-tag 是留永久债的 workaround，已否决。**

## 参考
- Highway `hwy/ops/shared-inl.h`（Simd 类型系统 + `HWY_IF_V_SIZE_D` 宽度-SFINAE）、`x86_{128,256,512}-inl.h`（分层 include，:84 行）、`foreach_target.h`、`highway.h`。
- HYLE `include/hyle/simd/vec.h`（tag 层已抄，Vec 层未跟上；`D::MaxBytes()` :55 就位）、`backend/{sse4,avx2,avx512}.h`（冲突面 §2）、`src/simd/CMakeLists.txt`（变体 TU 正交）。
- 施工：`simd-17-breadth-batch0-architecture.md`（根本重构）→ `batch{1,2,3,5}` → `batch4`（半向量，批0 之上）。
