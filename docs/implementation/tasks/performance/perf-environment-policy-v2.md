# 性能子进程环境策略版本化

<!-- TOOL TASK: COMPLETE -->

## Finding

2026-08-29 回放 util-core v1 engineering bundle 时，当前 verifier 以
`child environment policy differs at 'cleared_names'` 停止。旧 manifest 记录的是最初的
`hyle.perf.child-environment-policy/v1`；Tracy localhost 控制任务后来把
`TRACY_ONLY_LOCALHOST` 原地加入同一个 v1 blocklist。产品、contract、benchmark、consumer provenance、
artifact 与原始日志均未漂移，且 util-core frozen runner cache 明确 `HYLE_TRACE=OFF`。

问题属于公共工具身份设计：verifier 用今天生成的策略逐字段替代历史策略，而不是验证证据当时记录的
不可变策略。它既不能安全回放旧证据，也不能区分“已知且与 artifact 无关的策略差异”和真正弱化。

## Bounded fix

- 新采样只生成不可变 `hyle.perf.child-environment-policy/v2`；未来策略变化必须新增 schema；
- verifier 保留两个已经真实存在的 v1 精确指纹：原始 v1，以及加入 Tracy 清理后的过渡 v1；
- 原始 v1 只有在 frozen runner cache 明确 `HYLE_TRACE=OFF`，且 controlled environment/全部 strata
  均未使用 `TRACY_ONLY_LOCALHOST` 时才兼容，报告
  `LEGACY_COMPATIBLE_TRACE_DISABLED`；
- 过渡 v1 报告 `LEGACY_SCHEMA_CURRENT_POLICY`，v2 报告 `CURRENT`；
- 其他删项、fixed value/prefix 改动、未知 schema 或无法证明 Tracy 未进入 artifact 的旧合同继续
  fail closed；不修改任何旧 manifest、hash、raw log、统计门、margin 或产品代码。

## Verification

- `python tests/tooling/hyle_perf_runner_test.py`：28/28 通过，覆盖 v2 generation、两个已知 v1 指纹、
  blocklist/fixed-value 弱化、未知 schema、Trace ON/OFF 兼容边界；
- `python tests/tooling/hyle_perf_evidence_runner_test.py`：7/7 通过，含 legacy v1 + Trace OFF 的
  end-to-end manifest replay；
- retained util-core manifest 原文件 SHA-256 仍为
  `7027954068A7B5C1DDFC1ED2D83A3646F951694B90A19939FFD445ADF88FFEFB`；当前 CLI 深回放返回
  `paths_verified=true`、`qualification=QUALIFIED` 与
  `environment_policy_compatibility=LEGACY_COMPATIBLE_TRACE_DISABLED`；
- error、platform、mem owner、mem handoff 与 container-intrusive 的 retained bundle 也逐一完成当前
  CLI 深回放并得到同一 compatibility；前四者保持当前 consumer-policy eligible，container-intrusive
  仍因历史合同 `product_gate_policy=null` 只作原合同机械重放；
- Obs trace bundle（过渡 v1，已经包含 Tracy 清理）深回放返回
  `environment_policy_compatibility=LEGACY_SCHEMA_CURRENT_POLICY`、`paths_verified=true` 与
  `qualification=QUALIFIED`，证明 Trace ON 证据没有借 Trace OFF 例外通过；
- Time bundle 仍在环境策略检查前因 scheduler consumer provenance hash 漂移而 fail closed；本任务不
  掩盖或修复该独立产品证据身份问题；
- 本任务不续签 Formal，不改变 util-core 产品状态或性能点值。
