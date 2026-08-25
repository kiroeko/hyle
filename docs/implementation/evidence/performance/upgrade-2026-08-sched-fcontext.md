# Boost.Context 1.92：sched fcontext 升级证据

本页只验证依赖升级触及的 fiber switch 边，不是 sched 模块收口，也不覆盖 scheduler、
同步原语或吞吐的完整 SOTA 矩阵。runner 是
`hyle_sched_fcontext_bench`；它只使用当前 `hyle_sched` target 已选择的那组三份
`third_party/fcontext` 汇编，因此 peer 与产品不会漂到不同 Boost.Context 源码。

## 合同分层

| 层 | 做了什么 | 能否用于正常消费形状的升级回归 |
|---|---|---|
| raw primitive floor | 一次进入 peer 后在内部完成全部 `jump_fcontext` 往返 | 否；它隐藏了每次 consumer `resume/yield` 调用，只用于拆税 |
| raw consumer seam + explicit handle | 每轮从 consumer 调 `resume(handle)`，fiber 内调 `yield(handle)` | 否；缺少公开 static-yield/current 语义，只隔离调用边界 |
| raw seam + static yield/TLS current | 每轮与上行相同，另维护可嵌套的 thread-local current context | **是**；与 `Fiber::resume/Fiber::yield` 的正常消费形状相同 |
| HYLE Fiber | guarded `FiberStack`、static yield、`Fiber::current()`、嵌套 resume 恢复、完成态 | **是** |

所有行每个 epoch 都分配同档 guarded `FiberStack`、调用同一
`make_fcontext`，并执行 1000 次 caller → context → caller 往返。raw seam context 与公开
`Fiber` 对象都由 caller 在栈上持有，不夹带 benchmark 自己的 `new/delete`；创建/销毁成本同样
被 1000 次摊薄。正式判定只比较最后两层，不能再把 raw primitive floor 写成“同合同 rival”。

## Windows focused 结果

环境：Windows 11、MSVC 19.50、Release、trace/test instrumentation OFF、nanobench
4.6.0，15 个独立进程。工作区尚未形成 clean checkpoint，所以这是升级施工证据，不能冒充
三平台发布证书。

| linkage / 行 | median ns/roundtrip | spread | 相对同合同 raw seam |
|---|---:|---:|---:|
| shared raw primitive floor | 15.667 | 3.6% | 仅拆税 |
| shared explicit-handle seam | 25.774 | 2.9% | 非完整合同 |
| shared static-yield/TLS seam | 28.306 | 2.7% | 1.000 |
| shared HYLE Fiber | 27.696 | 1.8% | throughput 1.022；spread TIE，15/15 sign WIN |
| static raw primitive floor | 15.596 | 3.1% | 仅拆税 |
| static explicit-handle seam | 25.625 | 12.8% | 非完整合同 |
| static static-yield/TLS seam | 28.375 | 3.4% | 1.000 |
| static HYLE Fiber | 27.739 | 11.4% | throughput 1.024；spread/sign TIE |

旧 WSL focused 数字来自 raw context 尚含额外 heap allocation 的 runner，已作废；caller-owned
runner 必须在最终 clean hash 上重新采样，不能把旧结果换算后沿用。

Windows 中间结论：约 15.6 → 27.7 ns 的差距不是 HYLE 多做了一层算法。仅恢复“每次由
consumer 调用 resume/yield”的真实形状已经到 25.6–25.8 ns；再加入 static yield 所必需的
TLS current 后与完整 Fiber 同档。当前没有为了赢 benchmark 可以安全删除的元数据或查找；
这个自造 seam 只是同一 fcontext primitive 的机制下界，不能单独签署 sched SOTA 或模块收口；
跨平台升级回归结论仍以 final clean hash 的重跑为准。

MSVC 反汇编交叉核对了这项解释。trace/TSan OFF 后：

- `resume` 只剩 release-surviving `!finished_` 前置条件、保存/设置可嵌套
  `t_current`、一次 `jump_fcontext`、恢复 current，并保存新 suspended context；
- `yield` 只剩一次 TLS current 读取与非空前置条件、一次 `jump_fcontext`，并保存新的 caller
  context；
- 没有容器查找、heap、锁、原子 RMW、trace 或 TSan 调用。

TLS 不能在不削弱真实合同的情况下删除：`Fiber::yield()` 可在任意调用深度静态调用，
`Fiber::current()` 是公开语义，且一个 fiber 可以直接 resume 另一个 fiber；返回时必须恢复
外层 current。raw primitive floor 不提供这些能力。

## 复现

共享构建：

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset bench --fresh `
  -B C:/hyle-scratch/upgrade-perf-evidence-win `
  -DHYLE_COMPONENTS=sched -DHYLE_WERROR=ON
cmake --build C:/hyle-scratch/upgrade-perf-evidence-win --config Release `
  --target hyle_sched_fcontext_bench
python scripts/bench-stat.py `
  C:/hyle-scratch/upgrade-perf-evidence-win/bin/Release/hyle_sched_fcontext_bench.exe `
  -n 15 `
  --pair "raw consumer seam + static yield/TLS current" `
         "HYLE Fiber::resume/Fiber::yield"
```

静态交叉核对只需另加 `-DHYLE_STATIC=ON` 并使用独立 build directory。上述 Windows 数字仍是
dirty 施工证据；最终发布重签必须在 clean 的同一提交上复跑 Windows、WSL 与 macOS。raw
floor 始终只报告为税分解，不进入胜负表。
