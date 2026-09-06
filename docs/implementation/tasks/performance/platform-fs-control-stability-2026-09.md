# Platform-fs W1-WIN 控制项稳定性

<!-- TOOL TASK: QUEUED -->

## Finding

Platform v8 第二候选 `c39df6f94c636ab8fa94cb700dd263763a42d680`、tree
`37889346564f3a5cdc895f26c9d610883076ba02` 的 `platform-fs-p1-v2` 正式资格运行连续两次没有得到
合格仪器：

- `C:\public\project-hyle\_validation\platform-v8-perf-fs-evidence-r3` 是 exact candidate、exact
  consumer provenance。8/8 产品 strata 都为 `PARITY`，non-inferiority 都为 `PASS`；A/A、10%
  known-delta、timer-floor、placement 通过，但 5% known-delta 的一个 order 置信区间跨过 1，bundle
  因而只能是 `DIAGNOSTIC`；
- 按冻结预算进行的唯一一次独立重跑
  `C:\public\project-hyle\_validation\platform-v8-perf-fs-evidence-r4` 仍为 exact candidate/provenance，
  但 A/A 两个 order 都跨出等价判定边界。产品点值仍围绕 1，置信区间功效不足，聚合结果为
  `INCONCLUSIVE / UNRESOLVED`；
- 首候选 `e2a0ce9ddeb39d319591e5231f88112c06c75ef2` 的 r2 bundle 曾在 31 个独立样本下完整
  `QUALIFIED / PARITY / PASS`。它说明相同 workload 没有已知结构性劣化，却不能代替第二候选的 exact
  engineering receipt。

这不是一个已证明的 Platform-fs 性能回归：当前 exact r3 的全部产品行都通过 5% non-inferiority，失败
发生在已知差值控制项；r4 又由 A/A 先否定当前机器/运行时段的裁决资格。反过来，也不能利用这些产品
点值宣称当前候选性能通过。

## Owner 与边界

- owner：公共性能仪器 / `closure-perf-v1` 的 W1-WIN 文件系统运行环境；
- blocked consumer：`platform_fs` 的 `platform-v8-os-resource-contract` P1 gate；
- 不修改 Platform/Platform-fs 产品、公开 API、5% margin、contract row、peer adapter 或现有 manifest；
- 不在活动模块任务中增加第三次“跑到通过”、改变统计边界或把 `DIAGNOSTIC` 改写为 `QUALIFIED`；
- 不因一次受干扰时段就扩建整套仪器。先证明这是可重复的环境问题还是公共工具缺陷。

## 恢复条件

满足下列任一条后，创建新的 Platform-fs resume round，重新绑定同一候选并运行一次预声明的 exact P1：

1. W1-WIN 有一个独立、无并发 build/扫描/更新负载的登记时段，预跑能同时稳定通过 A/A、5%/10%
   known-delta、timer-floor 与 placement controls；或
2. 独立 tool task 用跨 workload 回放证明现有控制项存在可复现实现缺陷，修复并版本化 instrument
   contract，且旧/新 canary 都通过。

恢复运行必须重新产生完整产品 strata 与 controls；不得拼接 r3/r4 样本。若 controls 合格而产品仍
material fail，再按 Platform-fs 产品性能缺陷处理；若仍不合格，本任务继续保持具名 blocker，而不是
返回模块内无限调参。
