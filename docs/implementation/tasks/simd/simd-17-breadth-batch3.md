# simd-17 批3：Concat* scalable 补全（SVE/RVV）

| 项 | 内容 |
|---|---|
| 状态 | 🟢 qemu 多 VL 已实现+验过（grok）：SVE 128/256/384/512 + RVV 128/256/512/1024 全绿 |
| 前置 | simd-16 |
| 范围 | `backend/sve.h` + `backend/rvv.h` + `sve_test.cc`/`rvv_test.cc`。**不动**定宽（已全有）、不动 vec.h |
| 目标 | 翻案 sve.h:696 "ConcatLowerLower etc. have no scalable meaning" 的误判 —— Highway 对 SVE/RVV **都实现了整向量半-concat**，HYLE 补齐 |

## 0. 分工 + 翻案依据

- grok 实现 / opus 四查（含多 VL qemu）。
- **翻案依据**：sve.h:696 注释 "ConcatLowerLower etc. have no scalable meaning" **是错的**。Highway `arm_sve-inl.h:3845` / `rvv-inl.h:3833` 都实现了——语义是**整向量半-concat**（与 128-bit block 无关）：把两个全宽向量各自的"低半 / 上半"（按 `Lanes(d)/2` 分）拼成一个新全宽向量。这在 scalable 完全成立。
- **grok 先删掉 sve.h:696 那句错误注释**，换成正确实现。

## 1. 语义（务必对齐 Highway，scalar-oracle 式定义）

设 `n = Lanes(d)`，`h = n/2`。输入 `hi`、`lo` 两个全宽向量：

| op | 结果（lane 0..n-1） |
|---|---|
| `ConcatLowerLower(d,hi,lo)` | `[lo[0..h), hi[0..h)]` = lo 的低半 ++ hi 的低半 |
| `ConcatUpperUpper(d,hi,lo)` | `[lo[h..n), hi[h..n)]` = lo 的上半 ++ hi 的上半 |
| `ConcatLowerUpper(d,hi,lo)` | `[lo[h..n), hi[0..h)]` = lo 的上半 ++ hi 的低半 |
| `ConcatUpperLower(d,hi,lo)` | `[lo[0..h), hi[h..n)]` = lo 的低半 ++ hi 的上半 |

**注意参数序（Highway 惯例）**：`hi` 在前但填结果的**高半**，`lo` 填低半。定宽 scalar.h 已是这个序（`ConcatLowerLower(hi,lo) = [lo.lower, hi.lower]`，scalar.h:1160），scalable 必须一致。

**验证锚**：定宽 4 后端已实现且测过，用同一 oracle。scalable 只是把"半"从固定 128-block 改成运行期 `Lanes(d)/2`。

## 2. SVE 施工卡

HYLE sve.h 已有原语：`SlideUpLanes/SlideDownLanes`（svtbl 实现，sve.h:461+）、`IfThenElse`(svsel)、`FirstN`（mask）、`Lanes(d)` 运行期。

**手法（Highway Splice + MaskLowerHalf 的 HYLE 等价）**：Highway 用 `detail::Splice(hi,lo,mask)` + `MaskLowerHalf/UpperHalf`。HYLE 无 Splice，用 **slide + select** 合成。逐 op：

### ConcatLowerLower = [lo低半, hi低半]
- lo 的低半已在 lane [0,h)，就位。
- hi 的低半要搬到 lane [h,n)：`SlideUpLanes(d, hi, h)` 把 hi[0..h) 移到 [h..n)（低位补 0）。
- 合并：`IfThenElse(FirstN(d,h), lo, slid_hi)`——前 h 取 lo，后 h 取 slid_hi。
```cpp
const std::size_t h = Lanes(d) / 2;
return IfThenElse(FirstN(d, h), lo, SlideUpLanes(d, hi, h));
```

### ConcatUpperUpper = [lo上半, hi上半]
- lo 的上半 lane[h,n) → 搬到 [0,h)：`SlideDownLanes(d, lo, h)`。
- hi 的上半 lane[h,n) → 留在原位 [h,n)。
- 合并：`IfThenElse(FirstN(d,h), slid_lo, hi)`。
```cpp
return IfThenElse(FirstN(d, h), SlideDownLanes(d, lo, h), hi);
```

### ConcatLowerUpper = [lo上半, hi低半]
- lo 上半 → [0,h)：`SlideDownLanes(d, lo, h)`。
- hi 低半 → [h,n)：`SlideUpLanes(d, hi, h)`。
- 合并：`IfThenElse(FirstN(d,h), slid_lo, slid_hi)`。

### ConcatUpperLower = [lo低半, hi上半]
- lo 低半已在 [0,h) 就位；hi 上半已在 [h,n) 就位。**最简，直接 select**：
```cpp
return IfThenElse(FirstN(d, h), lo, hi);
```

> **⚠️ SlideDownLanes 补零语义**：HYLE SlideDownLanes 移出的高位补 0（sve.h 实现）。ConcatUpperUpper/LowerUpper 用它把上半搬到低位，[h,n) 会补 0——但那些位随后被 `IfThenElse` 用 hi 覆盖，**只要 mask 边界精确 h 就正确**。grok 验证时重点查 h 奇偶（VL 使 n/2 非整？scalable n 总是 2 的幂×基数，h 整除，但测试覆盖 VL128/256/384/512 确认）。

## 3. RVV 施工卡

HYLE rvv.h 已有：`SlideUpLanes/SlideDownLanes`（vslideup/down，rvv.h:563+）、`IfThenElse`(vmerge)、`FirstN`、`Lanes(d)`。

**手法与 SVE 完全同构**（slide + FirstN select）——RVV 不用 Highway 的 `Get<0>/Set<1>`（那是 LMUL 段操作，HYLE 是 LMUL=1 单寄存器，用 slide 更直接）：

```cpp
// ConcatLowerLower
const std::size_t h = Lanes(d) / 2;
return IfThenElse(FirstN(d, h), lo, SlideUpLanes(d, hi, h));
// ConcatUpperUpper
return IfThenElse(FirstN(d, h), SlideDownLanes(d, lo, h), hi);
// ConcatLowerUpper
return IfThenElse(FirstN(d, h), SlideDownLanes(d, lo, h), SlideUpLanes(d, hi, h));
// ConcatUpperLower
return IfThenElse(FirstN(d, h), lo, hi);
```

**四个 op 的 SVE/RVV 实现代码几乎逐字相同**（都是 slide+FirstN+IfThenElse），因为两后端的 SlideUpLanes/SlideDownLanes/FirstN/IfThenElse 语义已统一。grok 可两边对照写，降低出错。

## 4. 验收

- **删 sve.h:696 错误注释**（"no scalable meaning"）。
- `sve_test.cc` / `rvv_test.cc` 各加一个 `ConcatHalves` 测试组：
  - 构造 hi、lo 使 lane 值 = 全局可辨（如 lo[i]=i, hi[i]=100+i）。
  - 四个 Concat 各对拍 §1 表的 oracle，逐 lane `EXPECT_EQ`。
  - **多 VL 全覆盖**：SVE VL 128/256/384/512（h 各不同）、RVV VLEN 128/256/512/1024。这是关键——h=Lanes/2 随 VL 变，单 VL 过 ≠ 正确。
- oracle 直接用定宽 scalar.h:1160 的同款逻辑（按 h=n/2 分半）。
- 验收命令：`bash scripts/sve-qemu-test.sh` + `bash scripts/rvv-qemu-test.sh`（base tier 即可，非 Zvbb）。

## 5. 边界 / 风险

- **ConcatUpperLower 最简**（直接 IfThenElse 无 slide），先做它建立信心。
- SlideDownLanes 的补零不影响结果（被 mask 覆盖），但 grok 要在测试里确认——若 HYLE SlideDownLanes 语义与假设不符（如补的不是 0 而是环绕），ConcatUpperUpper 会错。**先单独验 SlideDownLanes 行为再组装**。
- 不涉及 Combine/LowerHalf（那是半宽返回，归批4）。这里四个 Concat 都是**全宽→全宽**，无类型系统问题。

## 6. clay 覆盖

翻案补齐 scalable Concat，纠正历史误判（sve.h:696），不改 clay 结论。
