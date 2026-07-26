# time 性能证书

> 本页是 time 性能数字的唯一信源。实现与选择理由分别见
> [`time-design`](../architecture/time-design.md) 和 [`guide/time`](../guide/time.md)。
> 当前证书：2026-07-25，提交 `14ee67e`；Windows/Apple n=15，WSL timer n=31，
> 每个样本是独立进程，判定以 `bench-stat.py` paired sign test（p < 0.01）为主。

## 仪器与对手

| 被测面 | 实际入口 | 对手 / 防伪 |
|---|---|---|
| clock read | `bench/misc/misc_bench.cc` | 同工作量 `std::chrono`，不是空函数 |
| arbitrary timer | `bench/time/timer_queue_bench.cc` | faithful Seastar `timer_set` radix；另列 indexed eager-erase heap |
| one-duration timer | 同上 | libevent-shaped intrusive TAILQ；另列 owner/order/link 同合同版本 |

所有 timer 节点在 epoch 外预分配，双方都携带 48-byte 的 link/deadline/data/callback/owner
持久状态。Seastar 线按本地成熟库源码使用 circular sentinel 和 Boost intrusive 默认
`constant_time_size=true`，不是早期错误的 null-ended 缩小版。两侧都完成 arm、取消或交付、
清 owner、drain ready list 的完整生命周期。

每个通用 timer 负载还注册一组 HYLE-first/rival-second 的 order-swap。Apple 无公开 P/E
硬绑核 API，因此正式跑显式设为 user-interactive QoS，并用 utility QoS 复核；两者都只表示
调度策略，不冒充物理 core pinning。

## 1. Clock

单位 ns/read，数值为中位数。

| 操作 | Windows x64 | WSL GCC 14.2 | Apple M4 | 结论 |
|---|---:|---:|---:|---|
| monotonic / `steady_clock` | 约同档（ratio 1.002） | 12.574 / 12.627 | 14.118 / 14.151 | 三平台统计平；默认路径达到 OS/标准库同档 |
| profiling clock / default | 约 1.77× throughput | 6.813 / 12.574（1.84×） | 13.789 / 14.118 | x86 TSC 可赢；M4 availability=false，仅回退 |
| realtime / `system_clock` | 统计平 | 12.697 / 12.624 | 14.714 / 14.667 | 三平台同档 |
| coarse realtime / default | 约 5.19× throughput | 3.821 / 12.697（3.32×） | 14.119 / 14.714 | Win/Linux coarse 可赢；mac availability=false |

默认钟本来就复用标准库/OS 原语，因此目标是“不因 HYLE 抽象退化”，不是声称比同一个
primitive 更快。M4 fallback 即设计成功，不能把约 1.02–1.04 的同函数采样差写成 fast path。

## 2. TimerQueue 对同算法 radix

N=4096；单位 ns/node。`arm+expire` 与 churn 的 run-to-run 绝对 spread 较大，所以表中
WIN/TIE/LOSE 使用同进程成对符号检验，而不是只看两个独立中位数。

| 平台 / 注册顺序 | arm+cancel H / Seastar | arm+expire H / Seastar | churn 70% cancel H / Seastar |
|---|---:|---:|---:|
| Windows，正常 | 4.035 / 4.755（WIN 15/15） | 4.836 / 5.532（WIN 14/15） | 5.728 / 6.324（WIN 15/15） |
| Windows，反序 | 4.104 / 5.007（WIN 15/15） | — | 5.883 / 6.264（WIN 15/15） |
| WSL，正常 | 2.597 / 3.308（WIN 31/31） | 4.358 / 5.254（WIN 26/31） | 5.345 / 5.355（TIE，22/31，p=.029） |
| WSL，反序 | 2.585 / 3.300（WIN 31/31） | — | 5.428 / 5.053（LOSE 0/31） |
| M4 interactive，正常 | 3.729 / 4.162（WIN 14/15） | 5.036 / 7.348（WIN 15/15） | 4.057 / 4.759（WIN 15/15） |
| M4 interactive，反序 | 3.754 / 4.218（WIN 15/15） | — | 3.779 / 4.776（WIN 15/15） |

M4 utility 复核仍同向：normal arm/cancel 3.777/4.147、expire 5.180/7.412、churn
3.777/4.766；反序 arm/cancel 3.764/4.151 为统计 TIE，反序 churn 3.785/4.797 为 WIN。
因此先前担心的“大/小核或 SSH background QoS 导致 M4 假胜”没有出现。

诚实结论：HYLE 与 Seastar 处于同一 SOTA radix 算法与复杂度档；Windows/M4 稳定更快，WSL
arm/cancel 与 expire 更快，但 churn 对代码/注册布局敏感——正常顺序平，反序 HYLE 慢约
7%。这不是可以宣称“三平台所有形状全赢”的结果，也没有证据支持为了这一处 GCC
code-layout 差异增加常态 metadata 或改变 API。性能门应继续保留双顺序。

## 3. TimerQueue 对 indexed heap

indexed heap 使用真实 id→index 和 eager erase，不使用 tombstone 假对手。它代表
libevent/libuv/Go 一类另一算法路线；cancel 为 O(log n)，radix 为 O(1)。

| 平台 | arm+cancel speedup | arm+expire speedup | churn speedup |
|---|---:|---:|---:|
| Windows n=15 | 2.31× | 9.83× | 3.33× |
| WSL n=31 | 约 2.50× | 约 9.14× | 约 2.80× |
| Apple M4 n=15 | 约 2.25× | 约 4.68× | 约 2.76× |

这里证明的是大量可取消 timer 的服务器形状。小 N、极少 cancel、必须全序交付的 workload
仍可能偏好 heap；HYLE 没有为了 benchmark 把 general API 锁死成 heap 或 wheel。

## 4. CommonTimeoutQueue

单位 ns/node。checked TAILQ 是相同安全合同的主基线；naked TAILQ 是去掉 owner/order/link
验证后的性能下界，不是可互换产品。

| 平台 | arm+cancel H / naked | arm+expire H / naked | arm+expire H / checked |
|---|---:|---:|---:|
| Windows n=15 | 2.468 / 3.059（WIN） | 3.174 / 4.108（WIN） | 3.174 / 3.616（WIN） |
| WSL | 约 1.835 / 2.569（WIN） | 约 2.989 / 3.960（WIN） | 约 2.989 / 3.110（同档） |
| M4 interactive | 3.034 / 3.118（TIE） | 3.531 / 2.836（LOSE） | 3.531 / 4.279（WIN） |

Windows 的 naked-first order-swap 仍为 3.177 / 4.044（HYLE WIN）。M4 utility 与
interactive 结论一致：对 checked 合同赢，对 naked expire 下界约慢 22%。这部分差异是
可检查 ownership、deadline 顺序、可恢复 C Status 和输出生命周期的合同税；删掉合同只为
数字变绿不属于优化。Common queue 尚无生产 consumer，接入真实 fixed-idle-timeout 路由时
必须补 application-level 证书。

## 5. 当前性能边界

| 项 | 状态 |
|---|---|
| 默认 clock | SOTA 同档：与 `std::chrono` 同 primitive，无额外抽象损耗 |
| opt-in profiling/coarse clocks | x86/支持 coarse 的平台有真实收益；只能用于合同允许的路径 |
| TimerQueue 算法 | SOTA radix，同 Seastar 复杂度；跨平台总体强，WSL reverse churn 约 7% 缺口公开保留 |
| Common vs 同合同链 | 三平台不输；M4 对 naked expire 下界有约 22% 安全税 |
| timer 精度 | exact ns key；实际唤醒延迟仍受 OS/event-loop 调度影响 |
| cross-thread queue | 不提供；single-owner 是消除热路锁的架构合同 |
| C ABI | arm/cancel/advance 无分配；尚未单列 wrapper-overhead benchmark，因为正常使用在队列批处理层 |

## 6. 复跑

```bash
scripts/bench-stat.py <build>/bin/hyle_misc_bench -n 15 \
  --pair "std::chrono::steady_clock::now" "HYLE monotonic_ns" \
  --pair "HYLE monotonic_ns" "HYLE profiling_clock_ns (rdtsc)" \
  --pair "std::chrono::system_clock::now" "HYLE realtime_ns"

scripts/bench-stat.py <build>/bin/hyle_timer_queue_bench -n 15 \
  --pair "indexed min-heap (libevent/libuv)" "HYLE TimerQueue (radix)" \
  --pair "radix timer_set (Seastar shape)" "HYLE TimerQueue (radix)" \
  --pair "radix timer_set second" "HYLE TimerQueue first" \
  --pair "radix timer_set churn second" "HYLE TimerQueue churn first" \
  --pair "checked intrusive TAILQ (HYLE contracts)" "HYLE CommonTimeoutQueue"
```

Apple 低优先级复核在命令前设置 `HYLE_BENCH_APPLE_QOS=utility`；默认是
user-interactive。Linux 正式性能必须在原生 ext4 worktree 跑，不能用 `/mnt/c` 9p。
