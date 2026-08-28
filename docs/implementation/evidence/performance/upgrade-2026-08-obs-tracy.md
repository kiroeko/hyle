# Tracy 0.14.1：obs trace-on 升级证据

> **HISTORICAL UPGRADE EVIDENCE：**本页保留依赖升级当天 dirty focused replay 的事实，不是当前
> 性能证书或复现入口。当前 clean qualification、identity、controls 与 verdict 只看
> [`benchmarks-obs`](benchmarks-obs.md)。

本页当时只验证 Tracy 0.14.1 升级触及的 native trace 热路，不是 obs 模块收口。产品 API
保持 `HYLE_TRACE_*`；direct Tracy 宏只出现在验证 target `hyle_obs_tracy_bench` 中。

## 同合同与 codegen

runner 对两种常见事件逐一同合同比较：

- `HYLE_TRACE_COUNTER(name, int64)` 对 `TracyPlot(name, int64)`；
- `HYLE_TRACE_ZONE_NAMED(name)` 对 `ZoneScopedN(name)`。

这是强于跨库 microbenchmark 的证据：HYLE 的 zone 宏在 trace-on Tracy build 中直接展开为
`ZoneScopedN`；counter 只增加一个编译期可消失的 `static_cast<int64_t>`。MSVC 19.50 Release
对象反汇编确认：

- 两个 counter lambda 的指令和 offset 完全一致，唯一不同是 plot-name relocation；
- direct/HYLE named-zone 函数从 `0x000` 到 `0x102` 的指令和 offset 完全一致，唯一不同是
  source-location relocation；
- 因此两条 HYLE seam 都与 Tracy 0.14.1 原生宏同为 O(1) event enqueue，没有额外扫描、
  allocation、lock、metadata 或 adapter call。

Windows `dumpbin /disasm:nobytes tracy_bench.obj` 可重复该核对；不同编译器只需归一化静态
字符串/source-location 符号，不应把 relocation 地址误判成指令差异。

## Windows focused 结果

环境：Windows 11、MSVC 19.50、Release、shared trace-on SDK、锁定的 Tracy
`v0.14.1` / `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`、nanobench 4.6.0，
15 个独立进程。当前工作区为 dirty 施工态，因此不能冒充三平台发布证书。

| 同合同事件 | direct Tracy | HYLE | HYLE/direct throughput | 判定 |
|---|---:|---:|---:|---|
| counter / plot | 11.062 ns | 10.966 ns | 1.007 | tie |
| named zone begin+end | 46.498 ns | 45.474 ns | 1.028 | codegen 同构；数值差只作顺序/噪声审计 |

zone 行的 paired sign test 会把一致的 2.8% 方向标为 win，但两函数实际指令序列同构，不能宣称
HYLE 比它直接展开的 Tracy 更快。正确结论是：升级后 HYLE seam 没有可测 adapter tax。

## 复现

`HYLE_TRACY_DIR` 必须通过中央 lock 的 source identity 校验：

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset bench --fresh `
  -B C:/hyle-scratch/upgrade-perf-tracy-win `
  -DHYLE_COMPONENTS=obs -DHYLE_TRACE=ON `
  -DHYLE_TRACY_DIR=C:/path/to/exact-tracy-0.14.1 -DHYLE_WERROR=ON
cmake --build C:/hyle-scratch/upgrade-perf-tracy-win --config Release `
  --target hyle_obs_tracy_bench
python scripts/bench-stat.py `
  C:/hyle-scratch/upgrade-perf-tracy-win/bin/Release/hyle_obs_tracy_bench.exe `
  -n 15 `
  --pair "direct TracyPlot(int64_t)" "HYLE_TRACE_COUNTER(int64_t)" `
  --pair "direct ZoneScopedN" "HYLE_TRACE_ZONE_NAMED"
```

后续 obs-v1 已取得 clean W1-WIN `closure-perf-v1` trace qualification；它不会追溯改变本页当时
dirty/diagnostic 的性质。其他平台承担适用 correctness/交付验证，不再要求把三平台点值池化成一张
发布表。
