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
- obs contract validate 通过，contract SHA-256 为
  `81f9ca4c1e0304e285f8949f888a8159b908a377699530bc1d8a5bbf4cfe8e4b`；
- fresh-path 单样本 probe 位于
  `C:\hyle-scratch\obs-v1\perf-localhost-probe-20260828a-output`，manifest 深重放通过且
  `paths_verified=true`；n=1 按协议为 `DIAGNOSTIC/INCONCLUSIVE`，不冒充性能裁决；
- Windows Firewall event 2099 按该 fresh build path 查询为 0，采样锁正常释放；可以恢复全新资格轮。
