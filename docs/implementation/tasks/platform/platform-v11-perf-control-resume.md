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

## Audit（`AUDITED`）

- `git diff 3c1ef239..46f4892e -- include/hyle/platform src/platform bench/platform
  bench/contracts/platform-host-p1-v3.json CMakeLists.txt cmake` 为空；NUMA memory 产品、全部 Platform normal
  path、benchmark/adapter/contract 与中央 build edge 均未漂移；
- oneTBB checkout clean，identity 精确为合同锁定的
  `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；comparison class、四 strata、ownership/failure、charged
  phases 与 5% margin 沿用 v10 的 same-contract 审计；
- `aac87769` 只修改公共 runner/control 判断、tooling canary 和说明；它不会链接进
  `hyle_platform`/benchmark product path。live runner 与 manifest verifier 共享同一个纯 stop-rule；
- v10 三平台 correctness、source/static/shared、API/docs 与 red-team 全部绑定不变 candidate，继续有效；
  本轮没有新产品 finding。冻结 FIX ledger 为空，允许操作只有 clean qualification、deep verify 和记录；
- 当前唯一风险是 n=31 仍可能不能资格化；这是合法终裁，不授权再次运行。

`SCOPED -> AUDITED` 前态 registry snapshot：
`C:\public\project-hyle\_validation\platform-v11-perf-control-resume\transitions\module-status-before-audited.json`，
SHA-256 `227D8562BFE74706B7C85657A2DF5FCE1F91524178188A120D427CAD42DE7583`。

## Candidate（`CANDIDATE`）

- product commit/tree 重新绑定为未改的 `3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c` /
  `63b58d1051d86c4aecb1bf736ae84265e359a5f5`；没有生成替代产品 candidate；
- immutable validation source 必须从当前 harness commit 建立 clean worktree，并用上述空 diff 证明其
  Platform/benchmark/contract bytes 等于 candidate；manifest 同时记录实际 source/tool identity；
- 公开 API、node-memory OS 语义、复杂度、布局、依赖、三平台 correctness、三交付、文档和红队保持
  v10 冻结结果；本轮不能用性能结果反向改变它们；
- `AUDITED -> CANDIDATE` 前态 registry snapshot：
  `C:\public\project-hyle\_validation\platform-v11-perf-control-resume\transitions\module-status-before-candidate.json`，
  SHA-256 `CE7586A0B8B896593F2445981122482330784709B626189A8444FDF9D3A4988A`。

## Verification（`VERIFIED`）

- clean harness/source identity：`85c666626f6e83517450ee36d26a2a0e5011c16f`；产品与 benchmark/contract
  path 相对 product candidate 无 diff；oneTBB clean/exact；fresh build/output 均在本 round 的 validation
  目录；
- 唯一 `qualify-platform` 在 n=15 已通过全部 controls，因此按规则没有不必要地进入 n=31。A/A、5%/
  10% known-delta、timer floor、placement、artifact/build、host/input stability 与 consumer provenance
  全部 `PASS`；qualification 为 `QUALIFIED`；
- 四个 same-contract host-planning strata 的 5% NI 均 `PASS`。median advantage ratio 为
  `1.015725 / 1.025350 / 1.024989 / 1.037168`，四个点值都偏向 HYLE；由于 99.26% CI 穿过 exact
  equivalence 上界或同时覆盖 1，exact verdict 均为 `INCONCLUSIVE`，这不影响预冻结的 non-inferiority
  工程门，也不宣称 material WIN；
- `hyle-perf.py verify`：`paths_verified=true`、qualification `QUALIFIED`、environment policy
  `CURRENT`、provenance `EXACT`、certification policy eligible；live 与 replay verdict 一致；
- raw bundle：`C:\public\project-hyle\_validation\platform-v11-perf-control-resume\perf-evidence`；
  contract-lock / manifest / summary SHA-256：
  `2714905D56EA9CCED7132A5EA55FE44C2925AACEF45B14A774EEDE4EF987C420` /
  `3BA2589D00AC079B6A9366F7D12F49CAF1DC95E6199CB33895533EE5A9E8F0F8` /
  `353ED5BFF2C576DC6E35B019ABAB863C4992509781DA8729F79473B5E93D8764`；
- v10 的三平台 correctness、source/static/shared、API/docs、dependency 与 red-team gates 因 product
  identity 未漂移继续 `PASS`。本轮全部适用 gate 已通过；
- `CANDIDATE -> VERIFIED` 前态 registry snapshot：
  `C:\public\project-hyle\_validation\platform-v11-perf-control-resume\transitions\module-status-before-verified.json`，
  SHA-256 `9D43237076C767C0AEFDA15C02579668AF1701B07EBC8F16FD2D07C76FB293EA`。
