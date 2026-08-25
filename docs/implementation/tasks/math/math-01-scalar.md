# math-01：scalar 标量数学助手

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | math/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；新增 math 套件 6 test）|

> **范围**：吸纳 clay `math/consts.h`(105) + `math/utils.h`(353)。结论是**绝大多数被 C++20/23 标准库取代**(同 `result.h`→`expected` 的现代化红利),只有 std 真没有的极少数增量自写,落进 `math/scalar.h`。
> **cloc**：**生产 28 行**(scalar.h) + **测试 51 行**。生产远小于 clay 458 行=现代化红利(下表),非 stub。

## clay API diff：逐项映射(非略过)

| clay 物 | 行数级 | HYLE 去向 |
|---|---|---|
| `numbers::*`(e/pi/ln2/sqrt2…) | consts 大头 | ♻️ C++20 `<numbers>`(clay 头自注"will be replaced when we switch")|
| `CLAY_PI/EPS/INF/MAX/DEG2RAD…` | consts 余 | ♻️ `<numbers>` + `std::numeric_limits` + 消费端自定义 |
| `is_power_of_2/next_pow2/log2_floor/log2_ceil/ilog2/get_largest_pot/round_up_to_power_of_two/count_leading_zero_bits/log_u32` | utils 大头 | ♻️ C++20 `<bit>`(`has_single_bit`/`bit_ceil`/`bit_floor`/`bit_width`/`countl_zero`)——硬件 LZCNT,比 clay 循环快 |
| `clamp/min3/max3/min4/max4/minn/maxn/between` | utils | ♻️ `std::clamp` / `std::max({...})` / `std::min({...})` |
| `pow/pow<I>/is_normal/abnormal/abs/sqr/fract/hypot2` | utils | ♻️ `<cmath>`(C++23 constexpr) + 平凡内联 |
| `align_up` | utils | ♻️ `hyle::mem::align_up`(mem/alignment.h,2 幂位掩码更快)|
| `deg2rad/rad2deg/radians/degrees` | utils | ⛔ 图形域,通用 infra 不做(归消费端)|
| `div_ceil` | 4 重载 | ✅ **HYLE 自写**:std 只有 floor 除,无 ceil |
| `roundup`(任意倍数) | utils | ✅ **HYLE `round_up_to_multiple`**:非 2 幂倍数对齐 |
| `sgn`(带 eps 死区) | utils | ✅ **HYLE 自写**:std 无死区符号 |
| `equal_adjusted_eps` | utils | ✅ **HYLE `approx_eq`**:相对 eps 浮点等值 |

## 改进 / 设计

- **div_ceil 改溢出安全式**：clay `(a+b-1)/b` 在 a 近类型上界时**溢出回绕**;HYLE `a/b + (a%b!=0)` 永不溢出(`div_ceil(uint64_max, 1)==uint64_max` 真测),编译器仍折叠成单 `div`。concept `std::unsigned_integral` 约束(clay SFINAE `enable_if`)。
- **approx_eq 相对 eps**：固定绝对 eps 跨标度错(1e15 vs 1e15+1 应等、1e-9 vs 2e-9 应不等);按 `min(|a|,|b|)*tol` 缩放。保留 clay 语义(对零比较退化为绝对零,与 clay 同,文档标注)。

## 对抗式自审（六类）

- **① 正确性**：div_ceil 0/整除/非整除/近上界溢出四边界真测,**测试初版抓到我自己的错误预期**(默认 tol≈1ulp,1e15+1 差多 ulp 应 false,我误写 true)——修正后演示相对 eps 真义(同绝对差大标度等/小标度不等);全 constexpr static_assert。
- **② 性能**：全 `constexpr` inline 模板,编译期可折叠;无分配/间接。被取代项指向 `<bit>` 硬件指令(比 clay 循环 log2 快)。
- **③ 规范**：纯头 INTERFACE 库;无 std 拥有型;无导出宏需求(全 inline);concepts 替 SFINAE(§8)。
- **④ clay 覆盖**：consts+utils 每个符号上表逐项映射(♻️/⛔/✅),无默默略过。
- **⑤ 残留脚手架**：无。
- **⑥ 过度复杂**：只留 4 个 std 真缺的;拒绝把能 ♻️ 的搬过来凑行数。
- **"比 std 快"**：div_ceil/sgn/approx_eq 是 std **没有**的能力(非对标项);被取代的 bit 操作指向 `<bit>` 本就是硬件最优,HYLE 不重造。

## 交付物

- `include/hyle/math/scalar.h`（div_ceil / round_up_to_multiple / sgn / approx_eq）
- `tests/math/scalar_test.cc`（6）
- CMake：新增 `src/math`(INTERFACE) + 主 CMakeLists 挂载 + math 测试目标。

## 参考

- clay `math/consts.h`/`math/utils.h`。C++20 `<bit>`/`<numbers>`、C++23 constexpr `<cmath>`。conventions §8(concepts 替 SFINAE)。
