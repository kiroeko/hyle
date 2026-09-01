# Core 工程性能资格与证据边界

> 当前机械状态只见 [`module-status.json`](../../module-status.json)。本页绑定 Core v3 候选
> `cdd6b1c612c8e318220972fa0936f23e28045447` 的 P0 结构证据；完整命令边界、三平台结果与限制见
> [verification manifest](../../../../evidence/core/manifest-cdd6b1c.md)，施工与六问见
> [Core v3 closure card](../../tasks/core/core-v3-foundation-completeness.md)。这不是纳秒榜，也不宣称
> “整个 Core 全局 SOTA”或脱离消费者的普适加速倍数。

## Core v3 当前有限结论

本轮新增的是 failure-only 与 compile-site 能力，因此 timing peer comparison 为
`NOT_APPLICABLE`，工程 verdict 是 `P0 STRUCTURAL PASS`：

- always-on assertion 成功路线仍是一个预测冷分支；context field construction、formatter、observer
  atomic load 与终止逻辑都只在失败块中；
- Release 禁用的 debug assertion 不求值 condition、message 或 fields，也不留下 runtime 指令；
- environment facts、compiler annotations、版本与 layout 常量没有 runtime registry、metadata 或
  Core artifact；
- failure formatter 每个 text segment 最多输出 256 bytes、context 最多 8 fields；这是 failure
  resource bound，不是把 `abort()` I/O 拿来计时；
- MSVC、clang-cl、GCC 与 AppleClang 的 optimized codegen、symbol visibility、sanitizer、直接消费者和
  source/static/shared consumer 均按冻结候选通过。三平台与精确 test count 见 manifest。

可公开的有限说法是：**冻结工具链与交付形状上的常态结构成本符合上述合同。** Prefetch 或 hint 是否
改善某个 kernel，仍由拥有访问模式和资源计费的 Container/Mem/Sched/SIMD workload 签署。不得把
`P0 PASS` 改写为“比 Highway/Abseil/Folly 更快”或“所有平台 SOTA”。

## 结构证据模型

Core 没有统一可计时的“操作”。不同能力需要不同证据，有限 claim 如下：

| 能力 | 复杂度/常态成本目标 | 合格证据 | 不合格证据 |
|---|---|---|---|
| version 常量 | 常态 constant-evaluated use 无 runtime call；无独立 runtime artifact | 编译、正常 package consumer 与 artifact inspection | getter 纳秒竞速；声称 ODR-use 也绝不物化 inline symbol |
| compiler hint | 不支持 target 安全退化；支持 target 形成预期 code shape | MSVC/GCC/AppleClang 的 optimized codegen、negative/debug/sanitizer compile，加 owning consumer workload | 只比较 macro 展开文本或单次纳秒点值 |
| fatal/assertion context | 成功路径 O(1) branch；debug-disabled 零 runtime；失败路径 bounded terminal diagnostic | hot/cold codegen、no-evaluation/release test、death/reentrancy/concurrency test、sink 不进入动态 export | benchmark `abort()`；声称进程内所有 DSO 共享唯一 observer |
| compile environment / annotations | compile-site facts/attributes；零 runtime registry/metadata | truth-table compile、三前端 WERROR、真实 Mem/Sched/Container consumer | 用 runtime CPU benchmark 证明 compile fact；把 hint 当保证 |
| layout 常量 | compile-time；不增加 runtime metadata | `sizeof/alignof` contract，加 consumer footprint/contention | 独立读取常量 benchmark |
| prefetch | O(1)，至多一个 target hint 或 no-op | instruction/no-call codegen，加拥有访问模式的 container/mem/sched/SIMD workload | 单独计时一条 prefetch |

失败诊断的输出成本随表达式/路径文本和 I/O 增长，但它终止进程，不属于 steady-state throughput。
`HYLE_ASSUME` 在 optimized Release 不应制造 runtime validation branch；Debug 借 fatal assertion 诊断
违约。`HYLE_UNROLL` 在 O0/sanitizer 下允许退化，正确性和 `-Werror` 构建优先于强行施加 pragma。

## 仪器与验收方法

Core 本轮不恢复旧的 module-wide nanobenchmark。资格由两层构成：

1. **结构证据**：在每个受支持 compiler 上编译固定 probe，确认 ordinary version value use 无
   linked getter/runtime artifact，force-inline 可消除可见 call，noinline 保留独立 body，prefetch
   不产生 library call，fatal 成功路径不复制 `fprintf/abort` body 且 detail sink 不进入 shared
   export surface，Debug/Release/sanitizer 退化可编译且语义正确；
2. **消费证据**：prefetch、padding、branch/loop hint 的得失回到拥有访问模式的模块 workload。
   consumer 先声明真实分布、资源计费边界和 material margin，再做 A/A、known-delta、order、placement
   与 holdout；没有第二把 consumer key 时，不因 microbench 修改 Core API、layout 或常态 metadata。

具体 opcode 会随 ISA、compiler 和 flags 合理变化，因此永久测试应断言结构合同，不把某一版汇编
拼写当 API。性能结论必须绑定 `(capability, consumer contract, hardware, toolchain, delivery,
candidate)`；同复杂度 non-inferior 不能外推为全局 SOTA。

## Core v3 证据登记

| 项目 | Candidate / environment | Verdict |
|---|---|---|
| Windows Core | `cdd6b1c6`；MSVC 19.50 Release/Debug，clang-cl 20.1.8 Release，WERROR | Release `18/18`、Debug Core `6/6`、clang-cl `18/18`，`PASS` |
| WSL Core / sanitizer | GCC 14.2 Release WERROR；ASan/UBSan Debug；TSan + `setarch -R` | Release `18/18`；ASan/UBSan 各 `5/5`；TSan Core `14/14` + release assertion `1/1`，`PASS` |
| macOS Core | Apple M4；AppleClang 21 arm64 Release WERROR | `18/18 PASS` |
| source/static/shared | Windows、WSL、macOS selected-component relocated consumers | 每平台三种 journey 各 `1/1 PASS`；Core binary artifact = 0 |
| 直接消费者 | Mem/Container/Sched；Windows/macOS WERROR，WSL normal warning policy | 每平台 `38/38 PASS`；只证明 Core 合同没有破坏其旅程 |

WSL 消费者的 GCC WERROR 重放在 Container 自身 `SmallVector` test-only `-Warray-bounds` 诊断停止；三个
产品 target 已在该轮编译完成，正常 warning policy 的完整 suite 为 `38/38 PASS`。它不改变 Core 自身
GCC WERROR `18/18`，也不被本页伪装成 Container 已重新签署。TSan 首轮的 WSL ASLR 映射冲突同样不
冒充产品 verdict；只签项目规定的 `setarch x86_64 -R` 重放。

## Core v2 历史证据登记

下表绑定已退休的 Core v2 candidate；完整命令、环境和 SHA-256 见历史 manifest。

| 项目 | Candidate / environment | Raw evidence | Verdict |
|---|---|---|---|
| Windows MSVC optimized/debug Core probe | `b241a22`; MSVC 19.50 x64 + ARM64 cross；clang-cl 20.1.8 x64 | `core-v2-ctest-*msvc*`、`*clangcl*`、`core-v2-codegen-msvc*`、`core-v2-symbol-visibility-msvc*` | `PASS` |
| Linux GCC optimized + ASan/UBSan probe | `b241a22`; Ubuntu 24.04 WSL；GCC 14.2 | `core-v2-ctest-release-gcc*`、`core-v2-codegen-gcc*`、`core-v2-symbol-visibility-gcc*`、`core-v2-{a,u}bsan-gcc*` | `PASS` |
| macOS AppleClang optimized/debug probe | `b241a22`; Apple M4；AppleClang 21 | `core-v2-ctest*-appleclang*`、`core-v2-codegen-appleclang*`、`core-v2-symbol-visibility-appleclang*` | `PASS` |
| source/static/shared isolated consumer | `b241a22`; Windows + macOS relocated contracts | `core-v2-component-contract-{msvc,appleclang}*` | `PASS` |
| 受影响 direct-consumer regression | `b241a22`; Windows Release/Werror | `core-v2-direct-consumers-msvc*`：66/66 | `PASS`；不等于下游模块签署 |

当前性能 verdict 是：**Core v2 的复杂度、常态成本和支持工具链上的代码形状已签署。** `prefetch`、
padding、branch/loop hint 是否改善吞吐仍必须由拥有访问分布的 container/mem/sched/SIMD 正常 workload
签署；没有该第二把钥匙时，Core 不单独声称收益，也不因纳秒 microbench 修改 API 或常态元数据。

## 历史证据如何处理

旧 Core benchmark 混入 linked version、Core FFI 和后来迁给 `container_intrusive` 的链表能力，不能
证明当前 Core。侵入链表的证据归
[`benchmarks-container-intrusive`](benchmarks-container-intrusive.md)；旧 run 只保留在 Git 历史作
回归调查输入，不继承为当前绿色结论。

稳定的 0.5 ns、顺序互换输赢或地址 placement 敏感结果，首先说明仪器/code shape 需要调查。只有
存在可解释的算法/codegen 机制，并在正常 consumer workload 达到预声明 material threshold，才
构成产品修改理由。
