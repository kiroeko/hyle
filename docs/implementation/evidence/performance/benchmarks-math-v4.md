# Math v4 P0 结构性能证据

> **当前工程 verdict：`PENDING`。** 本页只覆盖 checked scalar `try_lcm` 与 `try_bit_ceil`。绑定
> immutable candidate 后才记录三工具链 code shape 与交付结果；不运行纳秒 benchmark，也不声明
> Math、SIMD 或推理引擎的普遍 SOTA。

设计边界见 [scalar math architecture](../../../architecture/math-design.md)，用户选择见
[Math guide](../../../user/math.md)，本轮事实见
[Math v4 task](../../tasks/math/math-v4-checked-discrete-sizing.md)。

## 冻结合同

- `try_bit_ceil` 是 `constexpr noexcept`、O(1)、零 allocation/lock/syscall/HYLE runtime state or dispatch；
  representability guard 必须先于标准 primitive，优化目标应保留 native leading-zero/bit-count 形状，
  不能退化成与输入数值相关的循环；portable 标准库可以通过有界 helper 选择目标已有的 scalar bit
  instruction；
- `try_lcm` 是 `constexpr noexcept`、O(log(min(|lhs|,|rhs|))) 的标准 GCD，先除后 checked multiply、
  零动态资源；
  它的数据相关 scalar recurrence 不是 SIMD workload；
- 两者沿用 v3 的 exact-or-failure `optional<T>` policy，不增加返回值/success state 之外的 metadata，
  并保持 Core-only header component；
- comparison class 是 `STRUCTURAL`；timing peer comparison 为 `NOT_APPLICABLE`。标准 C++ 定义 trusted
  primitive，但没有同合同 checked failure route；
- test-only `extern "C"` probe 只提供稳定反汇编名称，不进入安装 ABI，也不是 Native FFI。

## Candidate 与三工具链结果

Candidate identity、object hash 与代表性 lowering 在产品提交不可变后回填。

| environment | expected representative lowering | forbidden cost | verdict |
|---|---|---|---|
| W1-WIN / MSVC x64 | guarded standard bit scan；unsigned magnitude；standard GCD；checked multiply boundary | heap、lock、syscall、HYLE backend dispatch、input-proportional bit-ceil loop | `PENDING` |
| W1-WSL / GCC x86-64 | 相同语义，使用 x86 native bit-count/branch | 同上 | `PENDING` |
| M1-MAC / AppleClang arm64 | 相同语义，使用 arm64 count-leading-zero/branch | 同上 | `PENDING` |

## Correctness 与交付旁证

候选必须通过 signed/unsigned/character-integral compilation、8-bit exhaustive LCM/bit-ceil oracle、
zero/one/min/max、constexpr、WSL UBSan、逐头，以及三平台 source/搬迁后 static/shared `hyle::math`
consumer。结果不预先标 PASS。

## Claim 边界

表格仍为 pending 时不签性能结论。P0 完成后也只允许声称：冻结的 scalar 合同达到准入任务的最优
复杂度档，且在已验证编译器上没有意外动态机制。不得改写成 timing、全平台或端到端推理结论。
