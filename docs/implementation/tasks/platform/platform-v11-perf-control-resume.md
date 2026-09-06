# Platform v11：NUMA memory candidate 性能恢复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `platform` 从 v10 仪器 blocker 恢复的独立 `RESUME` 轮。NUMA node memory API、三平台实现、
> correctness、delivery、usability 和 red-team 已绑定 product candidate
> `3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c`；本轮不修改产品，只续签冻结的 host-planning P1。

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform` / host facts / `platform-v11-perf-control-resume` /
  `RESUME`；baseline `aac877695b04b76b68eeff69c9bccbbafc55f411`；
- 前一 round / blocker：[`platform-v10-numa-node-memory-facts`](platform-v10-numa-node-memory-facts.md)；
  产品 4/4 NI 数值通过，但 n=15 的 5% known-delta 一个 order CI lower 未证明大于 1，runner 没有把
  precision-only positive-control uncertainty 送入合同已声明的 n=31；
- resume key：独立
  [`closure-perf-v1 control task`](../performance/closure-perf-v1-control-precision-escalation.md)
  已预冻结并实现通用规则；49 个 live/replay/control canary 通过。它不使用 Platform product ratio
  调阈值，也不改变 5% margin、统计量或 n=31 终裁；
- 复用产品 candidate/tree：`3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c` /
  `63b58d1051d86c4aecb1bf736ae84265e359a5f5`；v10 已完成三平台和三交付，本轮先机械核对产品路径无漂移；
- scope：保持 `platform-host-p1-v3`、oneTBB peer、W1-WIN、5% margin、四个 strata 和 same-contract matrix
  不变，在 clean immutable candidate checkout 与 fresh build/output 上执行恰好一次 qualification，随后
  deep verify、记录 fixed-direction 与 terminal state；
- non-goals：任何 Platform API/实现/测试、benchmark/adapter/contract/peer/margin、公共仪器再修改、
  预检、第二包、挑顺序/样本、拼接 v10 raw、Platform-thread/fs、Mem、FFI 或 Formal；
- tier / machine / instrument：`P1` / `W1-WIN` / `closure-perf-v1`；正确性机器仍为
  `W1-WIN/W1-WSL/M1-MAC`，只继承 exact candidate evidence；
- 预算/停止：一次 fresh qualification；统一 runner 可按预声明 `[15,31]` 自动完整替换一次。最终 bundle
  全部 required controls 与 product NI 为 `PASS` 才进入 `VERIFIED/CLOSED`；否则再次 `BLOCKED`，不继续
  改仪器或 Platform；
- build/package：核对 candidate 到当前 source 的 Platform-owned path、中央 edge、contract/bench path
  均无漂移；新 instrument commit 只属于 tooling，不进入 product artifact。

`BLOCKED -> SCOPED` 前态 registry snapshot：
`C:\public\project-hyle\_validation\platform-v11-perf-control-resume\transitions\module-status-before-scoped.json`，
SHA-256 `2116DE531E3C83CF0E0FA4143A2631BCF192ED0BD29102578DEC402AF047F072`。
