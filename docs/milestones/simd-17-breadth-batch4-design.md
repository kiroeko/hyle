# simd-17 批4：半向量合并族 —— 设计文档（需拍板路线）

| 项 | 内容 |
|---|---|
| 状态 | 🟠 设计待拍板（opus 出设计 / 用户或 opus 定路线 / 再出施工卡给 grok） |
| 前置 | simd-16；批1-3 无关可并行 |
| 目标 | `LowerHalf` `UpperHalf` `Combine` `ZeroExtendVector` `InsertIntoUpper` `ResizeBitCast` `ZeroExtendResizeBitCast`（~7 op） |
| 为何单独出设计 | 撞 HYLE"单后端-per-TU + 单一 native 宽度"架构，不是加函数，是**类型系统决策** |

## 0. 问题陈述（为什么不能照抄 Highway）

Highway 的半向量族返回**半宽向量类型**：`LowerHalf(Vec256<T>) → Vec128<T>`。这依赖 Highway 所有宽度类型（Vec128/256/512）**在同一 TU 共存**。

**HYLE 架构不同**（vec.h:216-239 实测）：
- **一个 TU 只编一个后端**：`-DHYLE_SIMD_TARGET_AVX2` 只 include `avx2.h`，`kNativeVectorBytes=32`。sse4.h **不在这个 TU**。
- 所以 avx2 TU 里**没有 `Vec128` 类型**——`Vec256` 的 `LowerHalf` 无处可返回。
- `Simd` tag 已有 `Half`（vec.h:59 `Simd<T, N/2, kPow2-1>`），**类型 tag 层没问题**，问题是**Vec 存储类型**：avx2 的 `Vec256{__m256 raw}` 没有半宽变体。

**这是早期"单一 native 宽度图方便"的直接后果**（`feedback_no_early_shortcut_as_excuse`）。做广度就该正面解决，不绕开。

## 1. 三条候选路线

### 路线 A：后端内引入半宽 Vec 类型（Highway 模型移植）
每个定宽后端**额外定义半宽类型**并让全宽 op 能产出它：
- avx2.h 加 `Vec128<T,N>{__m128i/__m128/__m128d raw}`（就是把 sse4 的 Vec128 搬一份进 avx2 TU）。
- `LowerHalf(Vec256)→Vec128` = `_mm256_castsi256_si128`；`Combine(Vec128,Vec128)→Vec256` = `_mm256_set_m128i`。
- avx512 加 `Vec256`（半）；sse4 加 `Vec64`（半的半，8 字节）。
- **代价**：每后端多一套半宽类型 + 半宽上的基本 op（否则半宽向量拿到手不能用）。类型爆炸——Highway 正是为此有 Vec128/256/512 全套 + partial N。**这是最"正确"但最重的路线**（往 Highway 全类型系统靠）。

### 路线 B：半宽用 partial-tag 全宽存储（HYLE scalable 已用的 kMaxLanes 思路）
半宽向量**物理上仍是全宽寄存器**，逻辑宽度由 tag 的 `N/kMaxLanes` 表达（只用低半 lane，高半忽略）：
- `LowerHalf(d_half, Vec256 v) → Vec256`（同物理类型，但配 `Half<D>` tag，Lanes()返回 n/2）。低半就位，直接返回 v。
- `UpperHalf(d_half, v)` = 把上半 slide/permute 到低半，返回全宽存储。
- `Combine(d_full, hi_half, lo_half)` = 两个"只用低半的全宽向量"拼成真全宽：`lo` 低半就位 + `hi` 低半搬到上半 + blend。
- `ZeroExtendVector(d_full, lo_half)` = lo 低半 + 上半清零。
- **代价**：半宽向量的"物理全宽"要求所有消费它的 op 都认 partial tag（Lanes(d) 限制循环）。HYLE **定宽后端目前无 partial 语义**（`static_assert(kNativeWidth)`）——要放开这个 assert，让定宽后端接受 `N < kMaxLanes` 的 tag。这也是架构改动，但**复用 scalable 已验证的 kMaxLanes+Lanes(d) 模式**，比路线 A 的类型爆炸轻。
- **风险**：全宽存储的"上半是垃圾"——`Store(half_vec, d_half, p)` 只能写 n/2 个（用 StoreN 或 partial store）。已有 partial-tag 测试（PartialTagSemantics）说明 HYLE 部分支持 partial，需确认覆盖面。

### 路线 C：半宽 op 只在 scalar + 提供"到内存"的等价，不返回半宽向量类型
承认定宽后端的半向量族**签名改为 memory-oriented**：
- `StoreLowerHalf(v, d, p)` / `StoreUpperHalf` 直接写低/上半到内存（绕开半宽向量类型）。
- `Combine(d, hi_ptr, lo_ptr)` 从两段内存拼。
- **代价**：偏离 Highway 契约（返回值 vs 内存），消费者代码不能与 Highway 互换。**违背"追平广度"初衷**——这是"能力做了但接口不对齐"，最不推荐。

## 2. opus 推荐：路线 B（partial-tag 全宽存储）

**理由**：
1. **复用已验证机制**：HYLE scalable 后端已跑 kMaxLanes + 运行期 Lanes(d) + PartialTagSemantics 测试。路线 B 把同一思路用到定宽——半宽 = `Half<D>` tag（Lanes 返回 n/2）+ 全宽物理存储。**不引入新类型系统**，只放开定宽的 `kNativeWidth` assert 到"接受 partial tag"。
2. **比路线 A 轻**：不用给每后端造 Vec128/Vec64/Vec256-half 全套 + 半宽上的重复 op。
3. **接口对齐 Highway**：`LowerHalf(d,v)`/`Combine(d,hi,lo)` 签名一致，返回向量（虽物理全宽），消费者可与 Highway 互换。

**路线 B 的具体设计**（待拍板后展开为施工卡）：
- 定宽后端 `Vec256<T,N>` 的 `N` 已是模板参数——**放开 Set/Load 等的 `static_assert(kNativeWidth)`**，允许 `N = kMaxLanes/2` 的 partial tag（物理仍 `__m256`，逻辑 n/2 lane）。
- `LowerHalf(Half<D> dh, VFromD<Twice<Half<D>>> v)`：低半就位 → 返回同 raw（配 dh tag）。
- `UpperHalf(dh, v)`：avx2 `_mm256_permute2x128_si256` 把上 128 搬到下 128（或 `_mm256_extracti128` 存低）→ 全宽 raw。
- `Combine(D d, half hi, half lo)`：avx2 `_mm256_permute2x128` 组合 lo 低半 + hi 低半。
- `ZeroExtendVector(D d, half lo)`：lo 低半 + 上半 `_mm256_zextsi128_si256` 清零。
- `InsertIntoUpper(D d, half hi, full v)`：把 hi 放进 v 的上半。
- `ResizeBitCast` / `ZeroExtendResizeBitCast`：位宽重解释（同 raw 不同 tag）。

**scalable（SVE/RVV）的半向量族**：Highway 也实现了（rvv `LowerHalf`=Trunc 半 LMUL；sve slide）。HYLE scalable 已是 kMaxLanes 模型，路线 B 天然契合——`LowerHalf` 返回同向量配 `Half` tag（Lanes 减半）。**路线 B 让定宽和 scalable 用同一 partial-tag 语义，统一。**

## 3. 待拍板问题（需用户/opus 决策）

1. **路线 A / B / C 选哪个？**（opus 推荐 B）
2. 若 B：**放开定宽 `static_assert(kNativeWidth)` 到接受 partial tag** 的影响面——现有多少 op 依赖"全宽"假设？需先审计（PartialTagSemantics 测试覆盖了哪些）。
3. 半宽向量的 `Store` 语义：partial store 写 n/2 个（用现有 StoreN？还是新 StoreLowerHalf？）。
4. 是否借此**统一定宽+scalable 的 partial 语义**（更大重构，但消除"定宽无 partial / scalable 有"的分裂）？

## 4. 不在本批

- 一旦路线定了，出施工卡给 grok（B 路线的施工卡会比批1-3 长，因涉及放开 assert + 每后端 permute 手法）。
- `LowerHalfOfMask`/`UpperHalfOfMask`（mask 的半）跟随向量半一起，但可能延到批4b。

## 5. clay 覆盖

半向量族是窄化管线能力，clay 无对应；net 新增，不改 clay 结论。本文档是**设计决策记录**，实现待路线拍板。
