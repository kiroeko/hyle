# sched-v1 当前源码重构与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮是 `sched` 第一次按当前模块收口协议审查。2026-07 的 `sched-01`--`sched-11`、旧
> `sched-module-closure.md`、架构叙事和 benchmark 只作历史输入，不继承功能、性能、交付或 API
> verdict。当前机械状态只看 [`module-status.json`](../../module-status.json)。进入 `AUDITED` 并冻结有限
> `FIX` 集前，不修改 sched 产品实现。

## Closure card（`SCOPED`）

- component / owner / round ID：`sched` / scheduling-runtime owner / `sched-v1`；
- task kind：`REFACTOR`；baseline commit：
  `63406dd0bf37ed2eb78baef8008820c64294fd81`；
- 进入原因与 demand ID：`SCHED-V1-NORTHSTAR-01`。HYLE 的首席消费者将是直接源码集成的推理引擎；
  当前 IO completion、Compute GPU/ggml bridge 与 Graph parallel 已真实暴露 sched 类型，但现有 sched 从未
  按当前 consumer-shaped 六问、能力完整性和交付方法签署，旧实现/测试可能由 benchmark 形状驱动；
- 真实 consumer / normal workload：
  1. 线程池上执行短 CPU callback 与可派生任务，包含 fork/join、work stealing、placement 和 shutdown；
  2. IO/timer/GPU completion 挂起并唤醒 fiber/coroutine，包含 timeout 与 cooperative cancellation；
  3. 推理后端或阻塞系统/库调用显式卸载到 blocking pool，再以 typed result/join 回到 scheduler；
  4. fiber 与外部线程之间使用 bounded channel、notification、semaphore 和结构化等待；
- 范围：当前 `include/hyle/sched`、`src/sched` 与 vendored fcontext 产品面；所有公开类型/函数、算法、
  ownership/lifetime/failure/concurrency、直接 HYLE 依赖与 IO/Compute/Graph consumer edge；C++23
  source/static/shared 交付、文档、旧 C/Native FFI 清洁度和当前 build/package 接线；
- 非目标：收口 IO/Compute/Graph；把 graph bulk/parallel algorithm、IO reactor、GPU stream policy 或模型
  执行策略下沉 sched；distributed/actor/process scheduler、抢占式 kill、自动 sysmon、稳定跨工具链 C++
  ABI、Native FFI/C API、全操作/全平台 SOTA、P2 端到端声明和 Formal 证书；
- 直接依赖：中央图 `PUBLIC core,error,platform,mem,container,util_core`，`PRIVATE time`；编译实现还使用
  compile-time Obs trace seam，必须在审计中裁决是否由 trace infrastructure 完整接线。产品第三方输入是
  vendored Boost.Context fcontext `cc1b9e02cd530d6607becc984e1e5bfa4f7b3600`；
- 直接消费者：`io`、`compute`、`graph`，以及 `samples/03_sched_fibers`；这些未收口模块只提供真实
  consumer slice，不借此取得自身 verdict；
- tier 与理由：`P1`。ready dispatch、spawn/join、suspend/wake、work stealing、sync 与 timer/cancel 可能
  高频，但尚无完成的 HYLE 推理引擎 profile 证明端到端贡献，故不得升为 `P2`；
- material margin / resource budget：默认 5% 同合同 non-inferiority；yield/wake/frozen-ready 常态不得新增
  allocation、syscall、全局锁或 per-operation metadata；work-stealing/queue/registration 的复杂度与 bounded
  footprint 在 audit ledger 逐项冻结；blocking/timer 只计其语义要求的 OS wait/clock/ownership 成本；
- 总预算：能力与 public trace 150 分钟；一个冻结产品批次最多 180 分钟；三平台 correctness/sanitizer、
  三交付和文档 150 分钟；P1 资格化与采样最多 150 分钟；红队 45 分钟。总工程不超过一个工作日；超出
  则按能力族拆出新 task，不扩写本轮；
- 异常预算：每个 P1 workload 最多一次反序复现、一次 codegen/profile 机制诊断、一个产品修正和一次
  完整重跑；不为亚纳秒、地址/顺序翻转或未资格化点值改变 API/layout；
- machine profile：`W1-WIN` 为 P1 锁定性能机；`W1-WSL` 为 GCC/sanitizer/ELF；`M1-MAC` 为
  AppleClang/AArch64/Mach-O correctness、交付和结构回放；
- instrument profile：P1 `closure-perf-v1`；correctness/delivery 使用 `closure-correctness-v1`；
- strong peers / current clean identities（2026-08-31 remote checked）：
  - stackful scheduler/fiber/sync：Marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e`；
  - async runtime/wake/blocking/sync：Tokio `ea91b33ca57ff0581b38e735cc108f831bccbdaa`；
  - CPU task/work-stealing/bulk demand：Taskflow `83f90a279aaab33fd4ce1deed00ca07819cbc8f8` 与 oneTBB
    `a8216b95202f3c55f9749870f4f8cfa4df0b32e3`；
  - north-star consumer：llama.cpp `9723942adc518b43c4b95dc4dce6906903eb5e09`、standalone ggml
    `36da57138425487184aa1da2eee2cde155909c6f`；随仓 ggml identity 在 audit 中单列；
- 无同合同 peer 时：以锁定 source consumer 的 throughput、tail latency、allocation/syscall/worker
  utilization 和 shutdown/cancel correctness 预算裁决；禁止把 stackful 与 stackless、caller-participating 与
  asynchronous、弱 failure/lifetime 合同混成胜负；
- 适用 gates：完整 capability-family map 与 public trace；normal/empty/boundary/OOM/move-destroy/
  concurrency/reentrancy/cancel-close/stress；ASan/UBSan/TSan；真实 x86-64 与 AArch64 fcontext；直接
  IO/Compute/Graph consumer slice；source/static/shared selected-component；C++ API/guide；C/FFI/symbol
  清洁度；有限 P1 workload 与独立红队；
- 预期输出：六问直答、能力/缺口 ledger、有限 `FIX` 集、单一 candidate identity、三平台/三交付与
  sanitizer 证据、资格化 P1 结论、architecture/user/performance/implementation 文档、限制/reopen trigger、
  terminal receipt 与四项 LOC；
- 停止条件：所有 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，全部适用门有真实 verdict 且 P1 聚合为
  `PASS` 才可 `CLOSED`；否则以具名 blocker 进入 `BLOCKED`。不把 actor/distributed/preemption、下游模块
  未完成或 Formal `NONE/PENDING` 误写成本轮产品缺口。

`QUEUED -> SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v1\module-status-before-scoped.json`，SHA-256
`F6B7F6D035EC94E693E361B837EC89963E228EE1BFCDCBBBC575637594BD8A9B`。

## Capability-family coverage map

等待当前源码、全部公开入口、真实 consumer、当前 llama.cpp/ggml 与 mature peers 的完整审计；本节完成并
冻结有限 `FIX` 集前，状态不得进入 `AUDITED`。

## Owner 六问直答

等待审计。旧任务中的“完整”“SOTA”“C ABI 完成”等文字不是本轮 verdict。

## Candidate

尚未产生；`SCOPED` 阶段禁止修改产品实现。

## Verification manifest

尚未进入候选验证；未运行项不得写为 `PASS`。

## 独立红队

等待冻结 candidate。

## Closure decision

当前状态为 `SCOPED`；下一合法状态是形成完整能力地图、public trace 与有限 FIX ledger 后进入
`AUDITED`，或以具名 blocker 停止。
