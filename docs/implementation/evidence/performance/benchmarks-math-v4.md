# Math v4 P0 结构性能证据

<!-- PERFORMANCE EVIDENCE: CURRENT -->

> **当前工程 verdict：`PASS`。** 本页只覆盖 candidate
> `686ec1c0215fa7a808ee470fe103ac62a1050759` / tree
> `1cb4bb81451ba016e09aecce563b3b591648d7a0` 的 checked scalar `try_lcm` 与
> `try_bit_ceil`；不运行纳秒 benchmark，也不声明
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

Probe source `tests/math/codegen_probe.cc` 的 SHA-256 是
`321BF646D52B1D7B48A6E559AD302C994FCB6DEB6EF66BB18BB94E37733AFF59`。从 candidate 到验证 HEAD，
Math public header、tests、probe、CMake 与 package sample 的 path diff 为空。

| environment / object SHA-256 | representative lowering | forbidden cost | verdict |
|---|---|---|---|
| W1-WIN；MSVC 19.50 x64；`F524BEEE609975E06D130EC20C76245824FB9C532C85924BF6451AB87994949F` | representability guard 后进入 bounded standard bit helper，其目标体包含 `lzcnt/bsr`；LCM 使用 standard GCD、`div` 与 full-width `mul`/high-half 判界 | 无 heap、lock、syscall、HYLE backend dispatch 或 input-proportional bit-ceil loop | `PASS` |
| W1-WSL；GCC 14.2 x86-64；`CAE89783BFE56F74172B70F77120E0343DC6939540EAB9F3819747D3BD359B48` | bit-ceil 直接为 guard + `bsr`/shift；LCM inline binary-GCD (`tzcnt`/subtract) 后 `div` + overflow-signalling `mul` | 同上；无 indirect call 或 generic arithmetic backend | `PASS` |
| M1-MAC；AppleClang 21 arm64；`40DB7278DFECB91C475B3B24B54C755A68ED481C1CF50F068F74E3188A32EAC9` | bit-ceil 为 guard + `clz`/shift；LCM inline standard GCD 后 `udiv` + `umulh/mul` 判界 | 同上；无 helper seam 或 runtime dispatch | `PASS` |

## Correctness 与交付旁证

候选在 MSVC 19.50、GCC 14.2 和 AppleClang 21 上通过 Math unit 与逐头测试；GCC exact-candidate 另以
UBSan 通过。测试覆盖 signed/unsigned/character-integral compilation、signed/unsigned 8-bit exhaustive
LCM oracle、8-bit bit-ceil oracle、zero/one/min/max、signed minimum、全部 representability boundary 与
constexpr evaluation。

三平台 `HYLE_COMPONENTS=math` selected smoke 均通过 source embedding、搬迁后的 installed shared 与
installed static consumer；package sample 直接调用两个 v4 API。输出闭包都是 `core,math`，安装物只含
header 与 INTERFACE target，没有 Math archive、第三方运行时、C header 或 Native FFI export。

## Claim 边界

P0 聚合 verdict 为 **`PASS`**：bit-ceil 是 O(1) native scalar bit-count shape，LCM 是标准 GCD 的
O(log(min magnitude)) 加一次 checked product；没有意外动态资源或 HYLE dispatch。这个结论只覆盖
绑定 candidate 与三种工具链/交付，不得改写成 timing、所有平台、所有 Math 操作或端到端推理 SOTA。
