# Math-v2 P0 结构性能证据

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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
`010323120bc6eb07b5a07e7d3e6de1725968d314`。W1-WIN 的 MSVC 19.50 optimized object 已确认 add/sub
各一次目标运算，signed/unsigned multiply 各一次 `imul/mul`，六个 probe 均无 call、divide 或
indirect dispatch；W1-WSL 与 M1-MAC 尚未完成，因此聚合 verdict 仍为 `PLANNED`。未运行项不得改写为
`PASS`。math-v1 的 ceil/round 结构结论继续由
[`benchmarks-math`](benchmarks-math.md)承担，但不能替代本轮新增函数的证据。
