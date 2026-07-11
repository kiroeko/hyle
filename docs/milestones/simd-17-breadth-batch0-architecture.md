# simd-17 批0：根本架构重构 —— 分层多宽度类型 + 宽度-SFINAE

| 项 | 内容 |
|---|---|
| 状态 | 🟡 WSL 核心完成（opus 亲自执行，2026-07-11）；待 Win MSVC + AVX512 SDE 运行 + mac NEON 复验 |
| 前置 | simd-16 |
| 范围 | `backend/{sse4,avx2,avx512}.h` 类型地基重构 + 新建 `backend/x86_detail.h` + `vec.h` 后端选择块。**不加任何新 op**，只改类型系统 |
| 目标 | 让 `Vec128/Vec256/Vec512` 在宽 target 里**分层共存**（Highway 模型），使能批4 半向量族 + 未来宽度转换全族 |
| 判据 | **现有 vec_test 全后端 100% 绿（零回归）**——这是纯重构，不许任何测试数变化 |

## 0. 为什么先做这个（读 `docs/architecture/simd-architecture-research.md`）

HYLE 定宽后端当前"单后端-per-TU + 单一 native 宽度"（`static_assert(kNativeWidth)`）是早期图快的选择，是广度天花板的根因。批4 半向量族 `LowerHalf(Vec256)→Vec128` 撞墙——avx2 TU 里没有 `Vec128`。

**根本解 = 抄 Highway 分层 include**：`x86_512→x86_256→x86_128`（Highway 实测 :84 行）让三宽度类型共存一 TU。AVX512 机器合法跑 SSE/AVX 指令，故宽 target 下窄 intrinsic 零惩罚可用。

**这一批不加 op，只重构地基。批1-5 都落在它上面。**风险最高（动核心类型系统），所以单独一批、判据是零回归。

## 1. 冲突面（opus 已实测，grok 照解）

层叠 sse4/avx2/avx512 后的冲突分类（研究文档 §2）：

| 类 | 例 | 解法 |
|---|---|---|
| Vec-typed ops | `Add(Vec128)` / `Add(Vec256)` 各 160+ | **不动**，不同 Vec 参数=天然重载 |
| T-only detail | `kSupported/kI8/kI16/kI32/kI64/kSigned` | 提共享头去重 |
| width detail | `kNativeWidth<D>` | **删除**，宽度-SFINAE 取代 |
| Raw/V/M 别名 | `Raw128/Raw256`、`V128/V256` | 不动（名字已不同）|
| D-dispatched | `Set(D)/Zero(D)` 等 ~24 | 改宽度-SFINAE |
| 类型别名 | `VFromD/DFromV/TFromV` | 定义一次 |

## 2. 施工步骤（严格按序，每步跑测试）

### 步骤1：抽 T-only detail 到共享头
新建 `include/hyle/simd/backend/x86_detail.h`：
```cpp
#pragma once
#include <type_traits>
#include <cstdint>
namespace hyle::simd::detail {
template <typename T> inline constexpr bool kSupported =
    std::is_same_v<T,float> || std::is_same_v<T,double> ||
    std::is_same_v<T,std::int32_t> || ... ;  // 照搬现有（sse4/avx2 完全相同）
template <typename T> inline constexpr bool kI8  = ...;
template <typename T> inline constexpr bool kI16 = ...;
template <typename T> inline constexpr bool kI32 = ...;
template <typename T> inline constexpr bool kI64 = ...;
template <typename T> inline constexpr bool kSigned = std::is_signed_v<T>;
}  // namespace
```
- sse4.h/avx2.h/avx512.h 删各自这些定义，改 `#include "hyle/simd/backend/x86_detail.h"`。
- **先只做这步，跑 vec_test 三后端**（此时还没层叠，纯去重，应全绿）。

### 步骤2：删 kNativeWidth，Set/Zero 改宽度-SFINAE
每后端的 D-dispatched 函数（`Set/Zero` 及所有 `(D)` 起手的 ~24 个）：
- **删** `static_assert(detail::kNativeWidth<D>, ...)`。
- 加宽度约束模板参数（HYLE 无 HWY_IF_V_SIZE_D 宏，用 `std::enable_if`/concept）：
```cpp
// sse4.h
template <class D, std::enable_if_t<D::MaxBytes() == 16, int> = 0>
HYLE_FORCE_INLINE detail::V128<D> Set(D, TFromD<D> t) noexcept { ... }
template <class D, std::enable_if_t<D::MaxBytes() == 16, int> = 0>
HYLE_FORCE_INLINE detail::V128<D> Zero(D) noexcept { ... }
// avx2.h: == 32 ; avx512.h: == 64
```
- **`D::MaxBytes()` 已在 vec.h:55**（`kMaxLanes*sizeof(Lane)`），直接可用。
- `VFromD = decltype(Zero(D()))` 不改——SFINAE 后按 D 宽度解析到对应 Zero。
- **仍单后端时**（未层叠）每后端只有自己那档 SFINAE，行为等同原 assert。**先只做这步 + 步骤1，跑三后端全绿**（证明 SFINAE 改造本身零回归）。

> ⚠️ **哪些函数要改**：所有第一参数是 `D`（tag）且返回宽度相关类型的。grok 用 `grep -nE "HYLE_FORCE_INLINE.*\(D[ ,)]" backend/sse4.h` 列出（~24 个），逐个加 SFINAE。**纯 Vec-typed（`Add(Vec128 a, Vec128 b)`）不加**——它们靠参数类型天然重载。边界情形：`Load(D,ptr)`/`Store(v,D,ptr)`/`BitCast(D,v)`/`MaskedLoad` 等带 D 的都要 SFINAE。

### 步骤3：类型别名去重
- `DFromV`/`TFromV`（作用于 `V::PrivateT`，width-agnostic）：三后端定义**逐字相同**，层叠后重定义。移到 `x86_detail.h` 或用 `#ifndef` 守卫定义一次。
- `VFromD = decltype(Zero(D()))`：依赖 Zero 重载，层叠后 Zero 三档都在→SFINAE 解析。定义一次即可（三后端相同）。
- **风险点**：`DFromV` 用了 `Simd<V::PrivateT, V::kPrivateN, 0>`——三后端相同，去重安全。

### 步骤4：分层 include
- avx2.h 顶部（`#include <immintrin.h>` 后）加 `#include "hyle/simd/backend/sse4.h"`。
- avx512.h 顶部加 `#include "hyle/simd/backend/avx2.h"`（传递拉入 sse4）。
- `#pragma once` 已在三头，重复 include 安全。
- **sse4 不 include 谁**（它是最窄，被别人 include）。NEON 独立不参与 x86 层叠。
- **此时 avx2 TU 里 Vec128 + Vec256 共存**。跑 vec_test **avx2/avx512 后端全绿**（关键验证：层叠后 D=256 tag 仍选 Vec256 op，D=128 选 Vec128 op，无歧义）。

### 步骤5：vec.h 后端选择块调整
vec.h:225-239 现在是 `#elif` 择一 include。层叠后 avx512 会传递拉入 avx2+sse4——**这正是要的**。但确认：
- `-DHYLE_SIMD_TARGET_AVX512` → include avx512.h → 传递 avx2.h → sse4.h。三档类型全在。
- `-DHYLE_SIMD_TARGET_AVX2` → include avx2.h → sse4.h。Vec128+Vec256 在。
- `-DHYLE_SIMD_TARGET_SSE4` → 只 sse4.h。仅 Vec128（正确，SSE 机器无 256）。
- **kNativeVectorBytes（vec.h:195）语义不变**——它仍是"本 target 的最宽"（avx2 TU=32），ScalableTag/默认 D 仍用它。层叠只是让更窄类型**也可用**，不改默认宽度。

## 3. 关键陷阱（grok 必读，opus 会重点查）

1. **kNativeVectorBytes 仍 per-TU 单值**：层叠后 avx2 TU 的 `kNativeVectorBytes` 仍是 32（不因见到 Vec128 变 16）。默认 `ScalableTag<T>` 仍全宽。**只是 `Simd<T,16/sizeof(T),0>` 这种窄 tag 现在能实例化出 Vec128 op**。
2. **VFromD 歧义风险**：层叠后 `Zero(D())` 有三档 SFINAE 重载，D 的 MaxBytes 必须**唯一匹配一档**。检查 `Simd<float,4,0>::MaxBytes()==16`→sse4 档，`Simd<float,8,0>==32`→avx2 档。若某 D 匹配 0 或 ≥2 档就是 bug。
3. **ODR**：`x86_detail.h` 的 `inline constexpr` 变量三 TU 一致（同定义）=ODR 安全。`DFromV` 等别名同理。
4. **不碰 scalar/neon/sve/rvv**：本批只 x86 三档层叠。NEON 的半宽（Vec64）留批4 单独处理（aarch64 无 x86 层叠问题，但 Vec128→Vec64 是同轴，批4 一并做）。
5. **测试数必须不变**：这是纯重构。vec_test 的 TEST 数、断言数、通过数**逐一对比重构前后**。任何变化=引入了行为改动=bug。

## 4. 验收（零回归铁律）

- **每步骤后跑 vec_test**（步骤1→2→4 各一次），不是只最后跑——定位回归到具体步骤。
- 双平台 Win MSVC + WSL g++（sse4/avx2 后端）+ **AVX512 Intel SDE**（avx512 后端）+ **mac NEON**（确认 NEON 不受 x86 层叠影响）。
- **测试数逐一对比**：重构前 `vec_test` 各后端 X 个 TEST 全绿 → 重构后必须**同样 X 个全绿**，一个不多不少。
- ASan（WSL fixed-backend）零错误——层叠不应引入任何内存问题（纯类型改动）。
- objdump 抽验：重构后 avx2 kernel 的热路径指令序列**与重构前逐条相同**（证明纯类型改动零代码影响）。这是"手法无降级"的铁证。

## 5. 明确不做

- 不加任何新 op（那是批1-5）。
- 不碰 dispatch/变体-TU/CMake（正交）。
- 不碰 scalable（SVE/RVV 已是 kMaxLanes 模型，无需层叠）。
- NEON Vec64 半宽留批4（同"半向量"主题一起）。

## 6. clay 覆盖

纯内部类型系统重构，无对外 API 变化（现有 op 签名/语义全不变），不改 clay 结论。使能后续广度批次。

## 7. 实施结果（opus 亲自执行，2026-07-11）

**实际改动**（比蓝图更精准，实测冲突面驱动）：
1. **新建 `backend/x86_detail.h`**：共享 `kSupported/kI8/kI16/kI32/kI64/kSigned`（lane-type-only，三后端逐字相同）+ `RedOp/RedOp2` 枚举（reduce fallback 用，三后端相同）+ **宽度-SFINAE 宏**：
   ```cpp
   #define HYLE_SIMD_BYTES_D(D) (D::kMaxLanes * sizeof(typename D::T))
   #define HYLE_SIMD_IF_W(D, LO, HI) \
     std::enable_if_t<(HYLE_SIMD_BYTES_D(D) > (LO) && HYLE_SIMD_BYTES_D(D) <= (HI)), int> = 0
   ```
   **关键：用字节 BAND 不是等值**——sse4 owns `(0,16]`、avx2 `(16,32]`、avx512 `(32,64]`。因为后端内部用 partial tag（TruncateTo 的 `Load(Simd<u8,4,0>(),buf16)` 是 4 字节 tag 但意为"128 位寄存器"），等值 `==16` 会误拒，band `(0,16]` 正确收纳所有 sub-128 tag 归 sse4。
2. **删每后端 `kNativeWidth` + `static_assert(native-width-only)`**（被 SFINAE 取代；NEON 保留自己的，不层叠）。
3. **宽度-SFINAE 加在纯 tag 函数**（无 Vec/Mask 参数的）：`Set/Zero/Load/LoadU/LoadN/FirstN/LoadInterleaved2/3/4/LoadMaskBits/StoreInterleaved2/3/4`。带 Vec/Mask 参数的（`Store/Add/Reverse/Gather/MaskedLoad`...）**不加**——参数类型天然重载。sse4 9 处 / avx2 13 处 / avx512 11 处。
4. **分层 include**：avx2.h `#include "sse4.h"`；avx512.h `#include "avx2.h"`（传递拉 sse4）。
5. **TruncateTo/U8FromU32 返回类型** `Vec256/512<DstT,N>` → `VFromD<Simd<DstT,N,0>>`：narrowing 结果 N 个 DstT lane 若 <native 宽，经 width-banded Load 自然返回窄 Vec（消除旧"partial 塞满宽寄存器"的 wart）。Promote* 不变（总是 widen 回全宽）。

**验收实测**（WSL g++-14）：
| 后端 | 结果 |
|---|---|
| sse4 (`-msse4.2`) | ✅ 184/184（层叠后仍全绿）|
| avx2 (`-mavx2 -mfma -mf16c`，层叠 sse4) | ✅ 184/184 |
| avx512 (`-mavx512f...+gfni`，层叠 avx2+sse4) | ✅ 编译通过（SDE 运行待 Win 侧）|
| scalar（未层叠，验未波及）| ✅ 184/184 |
| neon (aarch64 cross，未层叠，验未波及)| ✅ 编译通过（真跑在 mac）|

**零回归铁证（objdump）**：avx2 vec_test 重构后的指令直方图与 git 基线（`git stash` 三后端头恢复原状后编译）**逐条相同 = IDENTICAL** —— 证明纯类型改动，现有 op 零码变化，手法无降级。测试数 184→184 一个不差。

**待其他平台复验**（非 WSL 可完成）：Win MSVC 三后端 + AVX512 Intel SDE 真跑 + mac NEON 真机。这些应在有对应环境时跑，机制与平台无关（SFINAE/层叠是纯 C++），预期通过。
