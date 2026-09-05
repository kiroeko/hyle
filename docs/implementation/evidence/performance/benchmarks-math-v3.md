# Math v3 P0 结构性能证据

> **当前工程 verdict：`PASS`。** 本页只签发 Math v3 新增整数策略在候选
> `ff20319408372013fec6818c35f4ba7a7c79440c` / tree
> `3aba9d8c9eb06c927f7d29228b925668150bf0a6` 上的结构成本；不运行纳秒 benchmark，也不声明
> Math、SIMD 或推理引擎的普遍 SOTA。

设计边界见 [scalar math architecture](../../../architecture/math-design.md)，用户选择见
[math guide](../../../user/math.md)，本轮事实见
[Math v3 task](../../tasks/math/math-v3-integer-policy-superset.md)。v1/v2 页面是各自候选的历史证据，
不能代替本轮 candidate identity。

## 冻结合同

- scalar checked conversion/divide/remainder/compound 与 saturating operation 必须是 `constexpr noexcept`、
  O(1)、零 allocation/lock/syscall/state/runtime dispatch；
- `try_sum` / `try_product` 必须 O(n)、单遍、短路、零分配且不保留 caller span；
- runtime 路线优先编译器 overflow primitive 或目标原生 full-width multiply，不能引入通用数值后端；
- comparison class 为 `STRUCTURAL`。timing peer comparison 是 `NOT_APPLICABLE`，不能从点值推导 API；
- `extern "C"` probe 只提供稳定反汇编符号，不进入安装物，也不是 Native FFI。

probe source `tests/math/codegen_probe.cc` 的 SHA-256 为
`3B49F216CF71225E5DF43EB38E42C35B1253C59846B23BD89972C0896C3E0F97`。

## 三工具链结果

| environment / object SHA-256 | representative lowering | forbidden cost | verdict |
|---|---|---|---|
| W1-WIN；MSVC 19.50 x64；`96F76A3F8F005A7D8CDBCC29A24F60B837981532EBF26F54B9EDB8676AB4CCD3` | range compare；flag/high-half checked arithmetic；guarded hardware divide；conditional-select saturation | 无 heap、lock、runtime dispatch 或通用 arithmetic fallback | `PASS` |
| W1-WSL；GCC 14.2 x86-64；`240FFB803E0CC2E75AD19FF7FD70E0476BF29664827BE5DB41A121F8235A3E17` | cast 为 compare/branch；add/sub/multiply 使用 flag；divide 只有非法边界 guard 后的一次硬件 divide | 无 allocation、indirect call 或额外 divide | `PASS` |
| M1-MAC；AppleClang 21 arm64；`32D14815F1DEB37F654A50E84447FD1FE2A971F099A38284A588A0FE92CEE887` | cast 为 shift/branch；add saturation 为 `adds+csel`；multiply saturation 为 `mul+smulh+csel`；guarded `sdiv` | 无 heap、lock、runtime dispatch 或 helper seam | `PASS` |

不同 ISA 的 full-width multiply 指令数不同：AArch64 需要 low/high 两个结果，x86 可从 multiply
flags/high half 判界。这是目标机制，不是 HYLE optional/policy wrapper 的隐藏动态成本。

span aggregate 的循环按元素调用同一 checked primitive，首次失败立即返回；代码与 API 都没有缓存、
常态 metadata 或为 benchmark 固定的长度。其 Ω(n) 输入读取下达到 O(n) 最优复杂度档。

## Correctness 与交付旁证

- MSVC 19.50 Release/WERROR、GCC 14.2 Release/WERROR+UBSan、AppleClang 21 Release/WERROR 的候选精确
  worktree 均通过 Math unit test 与逐头 self-contained test；
- 边界覆盖 signed/unsigned/character integer conversion、8-bit exhaustive oracle、各宽度 min/max、
  zero divisor、signed min/`-1`、compound intermediate overflow、empty aggregate identities、saturation
  direction、constexpr 与 death precondition；
- W1-WIN、W1-WSL、M1-MAC 的 `hyle::math` source embedding、搬迁后 installed shared 和 installed static
  consumer 均通过。Math 保持 Core-only INTERFACE factor，没有虚构 archive 或 C/Native FFI symbol。

## 结论与限制

P0 聚合 verdict 为 **`PASS`**：scalar 是 O(1)，aggregate 是不可再改善复杂度档的 O(n)，正常路径没有
意外动态资源、分发或常态元数据，三工具链均保留目标原生形状。

本证据不比较任意 peer 纳秒点值，不覆盖 floating、SIMD、tensor、quantization、模型算子或下游端到端
KPI，也不把结构 PASS 改写成“平或更快”。支持工具链、公开 API、复杂度、组件边或正常 code shape
变化时必须重新取证。
