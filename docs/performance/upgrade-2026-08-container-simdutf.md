# simdutf 9.1：container UTF-8 升级证据

本页只验证 simdutf 9.1 升级触及的 `utf8_validate` / `utf8_length`，不是 container
模块收口。验证没有增加 benchmark 专用产品 API；`hyle_container_bench` 的 direct peer
在 benchmark executable 内另编同一份锁定 `simdutf.cpp`，产品仍从 `hyle_container` 调用
正常公开形状。

## 输入与合同

旧 runner 只有约 2 KiB mixed valid 输入，无法发现短串固定开销、纯 ASCII fast path、
invalid early-exit 与尾部残缺。当前矩阵为：

| case | 代表场景 | 结果合同 |
|---|---|---|
| ASCII short，14 B | method/path/小 key | valid |
| ASCII long，4 KiB | protocol/text payload | valid |
| mixed UTF-8 short | UI label/文件名 | valid |
| mixed UTF-8 long，约 4 KiB | 多语言 text payload | valid |
| invalid byte in middle，4 KiB | 扫描半程后失败 | invalid |
| truncated multibyte tail，4 KiB | streaming/chunk boundary | invalid |

每个 case 都测两个严格 pair：

- validate-only：HYLE bool 对 `simdutf::validate_utf8` bool；
- validate+count：HYLE code-point count / `-1` invalid，对 direct simdutf 先 validate、成功再
  count、失败同样返回 `-1`。

`count_utf8` 单独调用假设输入有效，不再放进同合同排名。

## Windows focused n=15

环境：Windows 11、MSVC 19.50、Release、shared HYLE、simdutf 9.1.0、nanobench
4.6.0。单位为 ns/input byte；invalid 行会 early-exit，所以其绝对 ns/B 只用于同一 case
的 pair，不能跨 case 排名。工作区尚未形成 clean checkpoint，当前数字是施工证据。

| case | validate+count direct / HYLE | HYLE/direct throughput | validate direct / HYLE | HYLE/direct throughput |
|---|---:|---:|---:|---:|
| ASCII short | 1.140 / 1.160 | 0.984 | 0.882 / 0.890 | 0.988 |
| ASCII long | 0.016 / 0.016 | 1.038 | 0.009 / 0.009 | 1.004 |
| mixed short | 0.742 / 0.746 | 0.992 | 0.549 / 0.552 | 0.992 |
| mixed long | 0.198 / 0.199 | 0.996 | ≈0.194 / ≈0.194 | 0.997 |
| invalid middle | 0.012 / 0.012 | 0.984 | 0.012 / 0.012 | 0.991 |
| truncated tail | 0.010 / 0.010 | 0.995 | 0.010 / 0.010 | 0.999 |

全部 gap ≤ 3.8%；10/12 项 ≤ 1.6%。短输入 1.2–1.6% 的一致方向是 product wrapper
多一次函数边界的亚纳秒固定税，算法仍是同一 simdutf 9.1 dispatched core；长 ASCII 方向
反转。static 交叉核对同样只在短输入见 1.5–2.8% 固定税，排除了 shared import 特有问题。
按绝对 spread 全部为 tie，没有升级造成的算法/复杂度缺口。

## 复现

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset bench --fresh `
  -B C:/hyle-scratch/upgrade-perf-container-win `
  -DHYLE_COMPONENTS=container -DHYLE_BENCH_SIMDUTF=ON -DHYLE_WERROR=ON
cmake --build C:/hyle-scratch/upgrade-perf-container-win --config Release `
  --target hyle_container_bench
$env:HYLE_CONTAINER_BENCH_FILTER = 'string-algo'
python scripts/bench-stat.py `
  C:/hyle-scratch/upgrade-perf-container-win/bin/Release/hyle_container_bench.exe `
  -n 15 `
  --pair "simdutf validate+count -> -1 invalid (same contract)" `
         "HYLE utf8_length (simdutf 9.1 product path)" `
  --pair "simdutf validate_utf8 (same contract)" `
         "HYLE utf8_validate (simdutf 9.1 product path)"
```

最终发布重签必须在 clean 的同一提交上复跑 Windows、WSL、macOS。当前页只界定 simdutf
升级的 UTF-8 受影响边并保存中间诊断；三平台结果写入前不宣称该边已经关闭，也不重跑或
重签 container 其他容器的历史性能表。
