# Core 性能资格与证据边界

> **状态：`UNQUALIFIED / EVIDENCE PENDING`。** 当前 Core 设计已经收敛为 header-only compiler/invariant
> 与 cache primitive，但最终 candidate commit、三平台 codegen 和 consumer workload 证据尚未在
> 本页绑定，因此不宣称性能已签署或“整个 Core SOTA”。施工状态见
> [`core-rebuild`](../../tasks/core/core-rebuild.md)。

## 可以证明什么

Core 没有统一可计时的“操作”。不同能力需要不同证据，有限 claim 如下：

| 能力 | 复杂度/常态成本目标 | 合格证据 | 不合格证据 |
|---|---|---|---|
| version 常量 | 常态 constant-evaluated use 无 runtime call；无独立 runtime artifact | 编译、正常 package consumer 与 artifact inspection | getter 纳秒竞速；声称 ODR-use 也绝不物化 inline symbol |
| compiler hint | 不支持 target 安全退化；支持 target 形成预期 code shape | MSVC/GCC/AppleClang 的 optimized codegen、negative/debug/sanitizer compile，加 owning consumer workload | 只比较 macro 展开文本或单次纳秒点值 |
| fatal assertion | 成功路径 O(1) branch；失败路径 terminal diagnostic | hot/cold codegen、death test、sink 不可抢占且不进入动态 export | benchmark `abort()`；声称进程内所有 DSO 共享唯一 sink |
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

## 本轮证据登记

下表由最终执行者在签署前绑定同一 candidate；空缺表示未签，不表示默认通过。

| 项目 | Candidate / environment | Raw evidence | Verdict |
|---|---|---|---|
| Windows MSVC optimized/debug Core probe | 待绑定 | 待绑定 | `PENDING` |
| Linux GCC optimized + ASan/UBSan probe | 待绑定 | 待绑定 | `PENDING` |
| macOS AppleClang optimized/debug probe | 待绑定 | 待绑定 | `PENDING` |
| source/static/shared isolated consumer | 待绑定 | 待绑定 | `PENDING` |
| 受影响 direct-consumer regression | 待绑定 | 待绑定 | `PENDING` |

在这些行具备可复现 command、compiler/flags、source hash 和 raw output 前，当前唯一性能 verdict 是：
**Core 的算法复杂度与预期常态成本已经限定，但 v2 性能资格尚未完成。**

## 历史证据如何处理

旧 Core benchmark 混入 linked version、Core FFI 和后来迁给 `container_intrusive` 的链表能力，不能
证明当前 Core。侵入链表的证据归
[`benchmarks-container-intrusive`](benchmarks-container-intrusive.md)；旧 run 只保留在 Git 历史作
回归调查输入，不继承为当前绿色结论。

稳定的 0.5 ns、顺序互换输赢或地址 placement 敏感结果，首先说明仪器/code shape 需要调查。只有
存在可解释的算法/codegen 机制，并在正常 consumer workload 达到预声明 material threshold，才
构成产品修改理由。
