# Math P0 结构性能证据

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> 本页记录 `math-v1` 冻结候选的可复现 P0 结构证据。它证明的是公开 scalar integer
> primitive 的复杂度与支持工具链代码形状，不是纳秒 benchmark，也不声明整个 math 或 SIMD
> “全面 SOTA”。当前模块状态只看
> [`module-status.json`](../../module-status.json)。

## 合同与身份

- product candidate：`753ad5f304f52727c3ba925c9aac0a30f941502e`，tree
  `93f7bf1a996d6beabebe2b42801b086c483a18e6`；
- test-only probe commit：`897263755564e3b5ef39024901d20bee9f54ac56`；
- probe：`tests/math/codegen_probe.cc`，SHA-256
  `c208c35425cf6cdbbf7dc8b167eb796c7c1c1e435088d7a99fd48aa18488ca4f`；
- workload：`uint64_t` normal `ceil_div`、normal `round_up_to_multiple` 与 checked
  `try_round_up_to_multiple`，输入来自调用者，结果可观察；probe 是不可安装的独立 object consumer，
  IPO 关闭；
- gate：每个入口 O(1)、零分配/锁/间接分派，正常数值路径最多一次整数除法。普通入口只允许
  无效输入进入 Core fail-fast；checked 入口不得调用失败 sink。

一次除法是 quotient/remainder 合同的结构下界；受测 ISA 的整数除法同时产生 quotient 与 remainder，
不应为两个表达式发出两次除法。CCCL 的 `ceil_div/round_up` 用于 API 与语义对照；本轮没有可归因的
独立热点或正常 consumer trace，因此按 P0 规则不制造点值竞速和模块级优胜声明。

## 三工具链结果

| Environment | Compiler / flags | Assembly SHA-256 | `ceil` | `round` | `try_round` | call/resource verdict |
|---|---|---|---:|---:|---:|---|
| W1-WIN x64 Release | MSVC 19.50.35728.0, `/O2 /DNDEBUG /FAs`, IPO off | `d09b81573fc1f9e0a660b70e7e7e9d070ef54be0aa902959eb3e89de5ce3f7d1` | 1 `div` | 1 `div` | 1 `div` | checked 0 call；normal 仅 direct fatal sink；0 heap/lock/indirect |
| W1-WSL x86-64 Release | GCC 14.2.0, `-O3 -DNDEBUG -S -masm=intel` | `ad976f8e238def64242adefde90ec366b8d44e972296bd4c44b94f81e9ab3c49` | 1 `div` | 1 `div` | 1 `div` | checked 0 call；normal failure split to `.text.unlikely`；0 heap/lock/indirect |
| M1-MAC arm64 Release | AppleClang 21.0.0, `-O3 -DNDEBUG -S` | `fe7346895e8ad9a3303ba44f359f69f9bb2b127a8560aa88965c29eb41a8d9dc` | 1 `udiv` | 1 `udiv` | 1 `udiv` | checked 0 call；normal 仅 direct fatal sink；0 heap/lock/indirect |

MSVC 为可能调用失败 sink 的普通 wrapper 保留了 ABI 要求的 40-byte shadow-space frame；checked
wrapper 没有 frame/call。它不是 allocation 或常态 metadata，也不能在没有 consumer profile 时用
亚纳秒 benchmark 驱动 API 删除。若真实 shape/kernel profile 证明普通 fail-fast wrapper 的分支或
frame 达到 material contribution，应在该 consumer 合同下重开，而不是改变本轮结构结论。

## 正确性与可重放

同一 probe 已进入 `hyle_math_tests` 的构建依赖；Windows MSVC Release/Werror、WSL GCC 14.2
UBSan/Werror 和 macOS AppleClang 21 Release/Werror 均成功编译。三平台 math unit test 与
public-header self-contained test 均为 2/2 `PASS`。

可从干净 checkout 复现核心 codegen（路径按机器替换）：

```text
cl /std:c++latest /O2 /DNDEBUG /EHsc /I<repo>/include /FAs /c <repo>/tests/math/codegen_probe.cc
g++-14 -std=c++23 -O3 -DNDEBUG -I<repo>/include -S -masm=intel <repo>/tests/math/codegen_probe.cc
xcrun clang++ -std=c++23 -O3 -DNDEBUG -I<repo>/include -S <repo>/tests/math/codegen_probe.cc
```

结构结论为 `PASS`：四个公开 template 均保持 O(1)、零动态资源；三个冻结消费形状在三工具链没有
重复除法或常态 out-of-line dispatch。后续 compiler/algorithm/API 变化时重新生成，而不把具体 opcode
文本固化成产品 API。
