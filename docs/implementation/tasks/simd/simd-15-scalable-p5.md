# simd-15：P5 变宽（SVE/RVV）平凡合成 + mask 操作

| 项 | 内容 |
|---|---|
| 状态 | ✅ 完成（2026-07-11，A+B 次模型 / C 校对方） |
| 前置 | simd-14（P4） |
| 后续 | `simd-gap-inventory.md` 收口 |
| 范围 | **仅** `include/hyle/simd/backend/sve.h` 与 `backend/rvv.h`。不动 vec.h、不动定宽后端、不动 dispatch |
| 公开 op 计数 | A+B 12 个 + C（SVE 3 / RVV 2）；RVV `TrailingZeroCount` 归 Zvbb 门控不做 |

## 0. 这份文档怎么用（协作分工）

本轮采用**双模型协作**：强模型出蓝图 + 校对，另一模型按卡实现。

- **批 A、批 B（12 个 op）→ 实现方按卡填写**。每张卡给了签名、要用的 intrinsic、oracle 位置、陷阱、验收命令。语义已由定宽后端锁死（逐 lane 对拍），实现方**不需要设计语义，只需把 oracle 翻译成 scalable intrinsic**。
- **批 C（3 个 op）→ 留给校对方（强模型）**。SVE `Expand`（无 prefix-sum 原语，要 Highway 风格迭代合成）、RVV `TrailingZeroCount`（Zvbb 门控决策）、RVV `Expand` 边界，难度和决策成分高，不外包。
- **每个 op 完工即校对**，不攒批。校对四查：①编译过（真 qemu，不是 `-fsyntax-only`）②多 VL 全绿 ③对拍 scalar oracle ④手法无降级。

> ⚠️ **核实纪律**：本文档基于 2026-07-11 源码（HEAD `c967f23`）核实——15 个 op 当前在 sve.h/rvv.h **全部真缺失**，定宽后端全部已有作 oracle，vec.h 泛型层无一个。实现方开工前对将做的每一项**再 grep 一次**当前源码确认没被后续 commit 补上。

## 0.5 架构决策：为什么在后端层做，不在 vec.h 泛型层做

批 A 里 6 个（BroadcastSignBit/ExclusiveNeither/IsEitherNaN/MulRound/GetExponent/BroadcastLane）理论上可提到 vec.h 泛型层一份覆盖所有后端。**本轮不这么做**，因为定宽后端已各自有 native 实现（avx2 `GetExponent` 是 bit-extract、`BroadcastSignBit` 是 native asr）；若 vec.h 再定义泛型同名会与定宽 native 重载冲突。P5 的目标是**让 SVE/RVV 追平定宽**，所以在这两个后端头里各自实现，与 P0-P2 同构。"把纯组合 op 上提泛型层去重"是独立的"回看去重"项，登记 backlog，不夹进 P5 主线（避免一次动多目标）。

## 1. 后端写法速查（实现方必读）

先读这两个文件的已有 op 找模板，别凭空造风格：

- **SVE**（`backend/sve.h`）：mask 类型是 `svbool_t`（**不是 vector**）；per-lane-width 分支用 `if constexpr (sizeof(T) == 1/2/4/8)`；全宽谓词用 `detail::active(d)` 或 `detail::ptrue<T>()`；已有参照 op：`FindFirstTrue`(sve.h:376，用 `svbrkb_z`+`svcntp`)、`FindLastTrue`(sve.h:381，用 `svsel`+`svindex`+`svmaxv`)、`Broadcast<kLane>`(sve.h:846，`svdup_lane`)、`IsNaN`(sve.h:872，`svcmpuo`)、`ShiftRight<k>`(sve.h:427，`svasr`/`svlsr`)、`Round`(sve.h:573，`svrintn_x`)。
- **RVV**（`backend/rvv.h`）：mask 类型是模板 `M`；VL 用 `Lanes(d)` 取；已有参照 op：`FindFirstTrue`(rvv.h:433，`__riscv_vfirst`)、`FindLastTrue`(rvv.h:436，`vid`+`vmerge`+`vredmaxu`——**批 B 的模板就在这**)、`Broadcast<kLane>`(rvv.h:1085，`vrgather` 常量索引)、`IsNaN`(rvv.h:747)、`Round`(rvv.h:652)、`ShiftRight`(rvv.h:488)。

**三条铁律（记忆里踩过的坑，违反必翻车）**：
1. **mask 不是 vector**。想把两个 compare 结果 mask 组合，**不能用 `hs::And`（那是 vector op，`DFromV<mask>` 会 incomplete 编译失败）**；SVE 用谓词 op（`svnor_z`/`svbic_z`/`svnot_z`），RVV 用 mask op（`vmnor`/`vmor`/`vmand`）。
2. **必须真 qemu 编译跑，不能 `-fsyntax-only`**。模板 op 不被实例化就不报错，单跑语法检查会漏过真 bug。测试里必须**直接实例化**（见 §4）。
3. **多 VL 一致**。SVE 一个二进制要在 VL 128/256/384/512 全过；RVV VLEN 128/256/512/1024 全过。单 VL 过 ≠ 正确（P2 就有 VL2048 分歧）。任何依赖 lane 数的常量都要从 `Lanes(d)` 取，不能硬编码。

---

## 2. 批 A — 直通 / 组合（6 op，最易，SVE+RVV 原语都齐）

每个都是"组合已有 op"或"一条原生 intrinsic"，抄定宽 oracle 手法即可。

### A1. `BroadcastSignBit`
- **签名**：`V BroadcastSignBit(V v)`，signed-int only（加 `static_assert(std::is_integral_v<T> && std::is_signed_v<T>)`）。
- **语义**：每 lane 变成"全 1（若原值<0）或全 0"。= 算术右移 (bits-1) 位。oracle：avx2.h:2197。
- **SVE**：`ShiftRight<sizeof(T)*8 - 1>(v)`（signed ShiftRight 已是 `svasr`，sve.h:427）。
- **RVV**：同 `ShiftRight<sizeof(T)*8 - 1>(v)`（`vsra`）。
- **验收**：正数→0、负数→全1(-1)、0→0，i8/i16/i32/i64。

### A2. `ExclusiveNeither`（mask op）
- **签名**：`M ExclusiveNeither(M a, M b)`。语义：`~(a|b)`（a、b 都为假的 lane 为真）。oracle：avx2.h:1383。
- **SVE**：`svnor_z(detail::active(d_for_T), a, b)`——注意 nor 需要 governing predicate，用全宽 ptrue。**别用 vector And/Or**。
- **RVV**：`__riscv_vmnor(a, b, Lanes(d))`。
- **验收**：真值表四组合 (FF→T, FT→F, TF→F, TT→F)。

### A3. `IsEitherNaN`（返回 mask）
- **签名**：`mask IsEitherNaN(V a, V b)`，float only。语义：a 或 b 是 NaN。oracle：avx2.h:1305（unordered 比较）。
- **SVE**：`svcmpuo(detail::ptrue<T>(), a, b)`（unordered 比较，一条；参照 `IsNaN` sve.h:872 的 `svcmpuo` 自比）。
- **RVV**：`vmor(vmfne(a,a), vmfne(b,b))`——`x!=x` 即 NaN，两个 mask 用 `vmor` 合。**别用 vector op 合**。
- **验收**：(NaN,x)→T、(x,NaN)→T、(finite,finite)→F、含 ±Inf→F。

### A4. `MulRound`
- **签名**：`V MulRound(V a, V b)`，float only。语义：`Round(Mul(a,b))` 就近偶数。oracle：avx2.h:431。
- **SVE / RVV**：直接 `Round(Mul(a, b))`——两 op 都有（SVE `svrintn_x`、RVV round path）。一行。
- **验收**：2.4*1=2、2.5*1=2(偶)、3.5*1=4，f32/f64。

### A5. `GetExponent`
- **签名**：`V GetExponent(V v)`，float only，返回 float。语义：`floor(log2(|v|))` = 提取偏置指数域再去偏置。oracle：avx2.h:372（f32：`(bits>>23 & 0xFF) - 127`；f64：`(bits>>52 & 0x7FF) - 1023`）。
- **SVE / RVV**：BitCast 到同宽 uint → `ShiftRight<23或52>` → `And` 掩码 → `Sub` 偏置 → `ConvertTo` 回 float。全是已有 op。**f32 和 f64 分支的移位量/掩码/偏置不同**，用 `if constexpr (std::is_same_v<T,float>)` 分。f64 的 i64→f64 转换 SVE/RVV 有原生 `ConvertTo`（比 avx2 的 store+reload 干净）。
- **验收**：GetExponent(1.0)=0、(8.0)=3、(0.5)=-1，f32/f64；对拍 `std::logb`（normal 输入）。

### A6. `BroadcastLane<kLane>`
- **签名**：`template<int kLane> V BroadcastLane(V v)`。语义：把 lane kLane 广播到全部 lane。oracle：avx2.h:1829（= `Broadcast<kLane>`）。
- **SVE / RVV**：**两后端都已有 `Broadcast<kLane>`**（SVE `svdup_lane` sve.h:846，RVV `vrgather` 常量 rvv.h:1085）。`BroadcastLane` 就是它的别名：`return Broadcast<kLane>(v);`。一行。
- **验收**：kLane=0/1/最后一个 lane，各类型；VL 变化时 kLane 仍指同一逻辑 lane。

---

## 3. 批 B — mask 前缀族（6 op，SVE 近原生 break 指令，RVV vfirst+iota 合成）

**抄写源**：Highway `hwy/ops/arm_sve-inl.h` 的 `SetBeforeFirst/SetOnlyFirst/SetAtOrBeforeFirst/SetAtOrAfterFirst`（SVE 用 break 指令原生实现，直接抄）。RVV 无 break 指令，用 `FindLastTrue`(rvv.h:436) 同款 `vfirst`+`vid`+compare 合成。

**四个 Set\* 的语义 + no-true 行为**（oracle avx2.h:1503-1529，务必对齐）：

| op | 有 true 时 | **无 true 时** |
|---|---|---|
| `SetBeforeFirst(m)` | 第一个 true **之前**的 lane（不含） | **全 true** |
| `SetAtOrBeforeFirst(m)` | 第一个 true 及**之前**（含） | **全 true** |
| `SetOnlyFirst(m)` | 只有第一个 true 那一个 lane | **全 false** |
| `SetAtOrAfterFirst(m)` | 第一个 true 及**之后**（含）= `~SetBeforeFirst` | **全 false** |

### B1-B4. Set* 四个（mask→mask）
- **SVE**（抄 Highway arm_sve）：
  - `SetBeforeFirst` = `svbrkb_z(ptrue, m)`（break-before 已在 FindFirstTrue 用，sve.h:367）。
  - `SetAtOrBeforeFirst` = `svbrka_z(ptrue, m)`（break-after 含首 true）。
  - `SetOnlyFirst` = AtOrBefore 去掉 Before = `svbic_z(ptrue, svbrka_z(...), svbrkb_z(...))`（谓词 bic = a AND NOT b）。
  - `SetAtOrAfterFirst` = `svnot_z(ptrue, svbrkb_z(...))`。
  - no-true 行为：`svbrkb_z`/`svbrka_z` 对全假输入天然返回全真 → Before/AtOrBefore 自动全真 ✓；Only 的 bic(全真,全真)=全假 ✓；AtOrAfter 的 not(全真)=全假 ✓。**无需特判，但要在测试里验 no-true 分支**。
- **RVV**（`vfirst`+`vid`+compare，参照 rvv.h:436）：
  - `const std::ptrdiff_t f = __riscv_vfirst(m, vl);` `auto iota = __riscv_vid_v_uXm1(vl);`
  - `f < 0`（无 true）分支：Before/AtOrBefore = 全真 mask，Only/AtOrAfter = 全假 mask。
  - `f >= 0` 分支：Before = `vmsltu(iota, f)`；AtOrBefore = `vmsleu(iota, f)`；Only = `vmseq(iota, f)`；AtOrAfter = `vmsgeu(iota, f)`。（比较对象 f 用 `vmsltu(iota, (uint)f, vl)` 标量形式）。
  - **陷阱**：iota 的元素宽度要匹配 mask 对应的 lane 宽（SEW），四种 sizeof(T) 各一套 `vid_v_uXm1`，照 FindLastTrue 的 `if constexpr(sizeof(T))` 四分支写。

### B5-B6. `FindKnownFirstTrue` / `FindKnownLastTrue`
- **签名**：`std::size_t FindKnownFirstTrue(D d, mask m)` / `...LastTrue(...)`。语义：与 `FindFirstTrue`/`FindLastTrue` **相同**，但调用方保证 mask 非全假（"Known"=省掉 -1 检查）。返回 `size_t` 不是 `ptrdiff_t`。oracle：avx2.h:1531/1535。
- **SVE / RVV**：**直接复用已有 `FindFirstTrue`/`FindLastTrue` 的核**，去掉 `AllFalse`/`vfirst<0` 的 -1 分支，`static_cast<std::size_t>`。SVE：`svcntp(ptrue, svbrkb_z(...))`；RVV first：`(size_t)__riscv_vfirst(m, vl)`，last：复用 rvv.h:436 的 iota-max 但去 -1 检查。
- **验收**：first-true 在 lane 0/中间/最后一个，各类型、各 VL。（不测 no-true——契约是调用方保证有 true。）

---

## 4. 批 C — 校对方完成（✅ 2026-07-11，opus）

| op | 后端 | 落地手法 | 备注 |
|---|---|---|---|
| `MulByPow2` | SVE | 原生 `svscale_x`（FSCALE，一条，IEEE ldexp 语义） | 校对方裁决：不做指数域合成——SVE 有硬件 FSCALE，一行即最优 |
| `MulByPow2` | RVV | store+reload `std::ldexp`（scratch 按 kMaxLanes） | RVV base V 无 vfscale，与 avx2/sse4/neon oracle 同款；不投机构造指数域（边界 overflow/subnormal 由 libm 保证） |
| `Expand` | SVE | store+reload scalar（`dst[i]=m[i]?src[k++]:0`） | **纠正蓝图**：不抄 Highway 8-lane 查表。gather/compress 类在无硬件后端本就是 store+reload（simd-08/09 定论），Highway SVE base 也走 ExpandLoop 兜底；store+reload 更简洁且 VL-agnostic |
| `Expand` | RVV | `viota_m`（exclusive 前缀和）+ `vrgather` + `vmerge` 0 | 原生前缀和，无 store+reload；`viota` 无 overloaded 短名 → 按 SEW 全名分支 |
| `TrailingZeroCount` | SVE | `LeadingZeroCount(svrbit_x(v))`（Highway arm_sve 恒等式） | svrbit 原生位反转；复用已有 svclz |
| `TrailingZeroCount` | RVV | **登记不做**（Zvbb 门控） | 与 rvv.h Pop/LeadingZeroCount defer 一致；base rv64gcv 无 vctz/vclz/vcpop，不用 float-exp workaround 破坏一致性 |

**校对方对 A+B 的复核**：12 op 全部对拍 oracle、双平台 4 VL 全绿，手法无降级。一处非阻塞观察（未改，warning-only-if-real）：SVE `Set*First` 用 `ptrue<T>()` 而非 `active(d)`，对 sub-max partial tag 不严格；P5 范围是 full scalable，测试全绿，符合规范不动。

> `Expand` oracle 语义（avx2.h:2021 / scalar.h）：把**紧凑排列**的低 CountTrue 个输入 lane 散布到 mask 为真处，假处填 0（Compress 的逆）。RVV `viota(m)[i]` = 严格小于 i 的 active lane 数 = 该真 lane 的源索引，正是所需前缀和。

---

## 5. 验收（每 op 完工即跑，全绿才算完）

**SVE**（WSL，需 aarch64 交叉工具链 + qemu，已装）：
```bash
bash scripts/sve-qemu-test.sh    # 自动跑 VL 128/256/384/512
```
**RVV**（WSL，需 riscv64 交叉工具链 + qemu）：
```bash
bash scripts/rvv-qemu-test.sh    # 自动跑 VLEN 128/256/512/1024
```

- 测试文件：`tests/simd/sve_test.cc` / `rvv_test.cc`。P5 加一个 `P5TrivialAndMask` 测试组，**直接实例化每个 op 的每个 lane 类型**（不能只写模板不调用——见 §1 铁律 2）。对拍标量 oracle（同 vec_test 风格：算 scalar 期望值逐 lane `EXPECT_EQ`）。
- **对拍参照**：直接用定宽后端的同名 op 逻辑手算期望，或复用 scalar 后端。mask op 用 `Store`+逐 lane 比对布尔。
- no-true / 全 true / 单 lane true / lane 0 / 最后 lane：Set* 家族这些边界都要覆盖。
- 完工后 `cloc --diff` 量 sve.h/rvv.h/测试 增长，写回本文档 §6。

## 6. 实际代码量

A+B+C 相对 P4 HEAD：`+782` 插入（sve.h +116 / rvv.h +182 / sve_test +253 / rvv_test +231）。
- **SVE**：18 tests × VL 128/256/384/512 全绿（`P5TrivialAndMask` 12 op + `P5BatchC` MulByPow2/Expand/TZCnt）
- **RVV**：15 tests × VLEN 128/256/512/1024 全绿（`P5BatchC` MulByPow2/Expand；TZCnt Zvbb 门控不做）

## 7. clay 覆盖

clay 无可移植 vector API，这些是 HYLE `simd/` scalable 后端的净新增对称能力，不改变 clay 覆盖结论。

## 参考
- 定宽 oracle：`include/hyle/simd/backend/avx2.h`（GetExponent:372 / MulByPow2:395 / MulRound:431 / ExclusiveNeither:1383 / IsEitherNaN:1305 / Set*First:1503-1529 / FindKnown*:1531 / Expand:2021 / TZCnt:2132 / BroadcastLane:1829 / BroadcastSignBit:2197）。
- 后端模板：`backend/sve.h`、`backend/rvv.h`（§1 列的行号）。
- 抄写源：Highway `hwy/ops/arm_sve-inl.h`（Set*First、Expand）。
- memory `project_simd_int_lanes`（op 补全全史 + scalable 踩坑）、`project_simd_sve_rvv_feasibility`。
