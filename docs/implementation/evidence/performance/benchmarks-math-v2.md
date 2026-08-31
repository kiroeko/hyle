# Math-v2 P0 结构性能证据

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> 本页承载 `math-v2-checked-arithmetic` 的 P0 结构合同。它不运行纳秒 benchmark，也不声明整个 Math、
> SIMD 或推理引擎全面 SOTA。当前产品状态只看
> [`module-status.json`](../../module-status.json)。

## 冻结合同

- workload：observable output 的 signed/unsigned 64-bit `try_add`、`try_subtract`、`try_multiply`；
- target：每个 helper O(1)，零 allocation、lock、syscall、常态 metadata 与 indirect dispatch；正常
  runtime 只有一个目标 arithmetic operation 和 overflow predicate；
- tools：W1-WIN MSVC Release x64、W1-WSL GCC 14 Release x86-64、M1-MAC AppleClang Release arm64；
- probe：`tests/math/codegen_probe.cc`，不可安装且 IPO 关闭；`extern "C"` 只固定反汇编符号名，不是
  产品 C ABI；
- disallowed verdict source：点值 timing、函数地址顺序、单次 assembly 文本 hash 或只编译未运行的
  package fixture。

## 当前结果

产品候选已冻结为 `1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` / tree
`010323120bc6eb07b5a07e7d3e6de1725968d314`；candidate 后只有 warning-clean test fixture 与证据
提交，产品路径为零 diff。probe source SHA-256 为
`FDC60F286C7934144AB6F6C6CAA56318651C336015DC048A44DFE0D28233F998`。

| Environment / object SHA-256 | add/sub 64-bit | multiply 64-bit | calls / dynamic resources | Verdict |
|---|---|---|---|---|
| W1-WIN, MSVC 19.50 x64, `026659C5E954751E686A2956DD8E686A549300CB4EE41B0CB4A087B6EBAB5973` | one `lea/sub` plus carry/sign predicate | one `mul/imul` plus high-half/sign predicate | no call/divide/indirect；zero heap/lock | `PASS` |
| W1-WSL, GCC 14.2 x86-64, `081f8ef9cd41bb601f03a21a0f7736651a898fdf38ac44f82d2af85fee433057` | one `add/sub` plus `jb/jo` | one `mul/imul` plus `jo` | no call/divide/indirect；zero heap/lock | `PASS` |
| M1-MAC, AppleClang 21 arm64, `e303735fb738805230c950074a811642c0552c3dbd3673352e5da2a9f2263f38` | one `adds/subs` and flag predicate | unsigned `umulh+mul`；signed `mul+smulh`，再比较 sign extension | no call/divide/indirect；zero heap/lock | `PASS` |

AArch64 没有单指令同时返回 64-bit product 的 low/high halves；low+high 两条 multiply 是 full-width
overflow detection 的原生最短形状，不是 optional、HYLE wrapper 或 runtime route 的成本。x86 的
`mul/imul` 同时提供所需 halves/overflow flag。三个工具链都把 public optional API 内联进 observable
caller，没有 material helper seam。

聚合 P0 verdict 为 **`PASS`**：新增函数保持 O(1)、零动态资源和工具链原生 overflow lowering；没有
运行 timing benchmark，也不声明跨 workload 的纳秒优胜或整个 Math/SIMD SOTA。math-v1 的 ceil/round
结构结论继续由
[`benchmarks-math`](benchmarks-math.md)承担，但不能替代本轮新增函数的证据。
