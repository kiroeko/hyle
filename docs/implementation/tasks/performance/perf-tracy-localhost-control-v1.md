# Tracy benchmark 本地监听控制

<!-- TOOL TASK: COMPLETE -->

## Finding

2026-08-28 的 obs `trace-native-seam` 资格运行在 Windows 上为不同 build path 的 Tracy-enabled
benchmark 反复触发网络访问授权界面。Windows Firewall operational log 的 event 2099 将这些授权
直接绑定到 `hyle_obs_tracy_bench.exe` 与 placement variant；可执行文件 manifest 为 `asInvoker`，
runner 也没有 `runas`/elevation 路径，因此不是 priority/affinity 提权。

Tracy 0.14.1 的 listener 默认绑定 wildcard interface，并提供 `TRACY_ONLY_LOCALHOST=1` runtime
配置。当前统一仪器只允许 HYLE namespace 环境变量，无法既清除 ambient profiler tuning、又显式登记
这一安全/交互边界。

## Bounded fix

- child sanitizer 总是清除 ambient `TRACY_ONLY_LOCALHOST`；
- nanobench runner 只额外接受精确 name `TRACY_ONLY_LOCALHOST` 与精确 value `1`；
- obs 合同的每个 trace stratum 显式声明该值；
- 其他 Tracy runtime/compile switches 保持拒绝，统计、sample schedule、margin、产品 workload 与
  candidate 均不改变。

## Verification

- `python tests/tooling/hyle_perf_runner_test.py`：27/27 通过，覆盖 exact allow、错误 value 与未审查
  Tracy switch；
- 第一版 localhost-only obs contract validate 通过，contract SHA-256 为
  `81f9ca4c1e0304e285f8949f888a8159b908a377699530bc1d8a5bbf4cfe8e4b`；
- fresh-path 单样本 probe 位于
  `C:\hyle-scratch\obs-v1\perf-localhost-probe-20260828a-output`，manifest 深重放通过且
  `paths_verified=true`；n=1 按协议为 `DIAGNOSTIC/INCONCLUSIVE`，不冒充性能裁决；
- Windows Firewall event 2099 按该 fresh build path 查询为 0，采样锁正常释放；可以恢复全新资格轮。

首次 n=15 资格采样随后暴露第二个同源生命周期问题：每个未连接 viewer 的 Tracy process 在
`main` 完成后约花 13 秒清空已计时但无人消费的 event queue。完整 n=15 controls 全部通过；唯一未决
产品 stratum 是 counter `primary/peer-first`，99.26% CI 下界 `0.95068` 只略越过 5% NI 边界
`0.95238`，所以冻结规则正确请求 n=31。但按原退出行为完整升级预计使本轮超过 45 分钟预算，runner
已在 n=31 初期停止，部分 n=31 样本作废。

补充的有界修复只作用于 benchmark process lifecycle：合同显式设置
`HYLE_BENCH_SKIP_TRACE_DRAIN=1`；全部 nanobench row 与 telemetry flush 后，trace benchmark 用
`_Exit(0)` 跳过不在 charged phases 内的静态 Tracy queue drain。普通手工运行不设置该变量，仍验证
真实 backend shutdown。该修复不得改变 event count、epoch、sample schedule、margin 或产品代码；
短探针必须证明 retained output 完整、wall time 恢复且防火墙事件仍为 0，才可重启资格轮。

最终验证：单 executable 机制探针从约 13 秒恢复为 0.95 秒，exit code 0，telemetry 与三组
candidate/peer rows 全部保留；统一 fresh n=1 probe 的完整 build + 4 个产品进程 + 2 个 controls 约
39 秒，4/4 产品日志均记录 `trace_shutdown=skip-drain`。manifest 深重放通过、
`paths_verified=true`、Windows Firewall event 2099 为 0、采样锁正常释放。最终 contract SHA-256 为
`cd1e2b1657ef4aacff7988741086667f94be08814757e8359344bb06a18ae75e`。工具任务结案；n=1 仍按协议只
是 `DIAGNOSTIC/INCONCLUSIVE`，正式性能 verdict 必须来自新的 clean qualification。
