# time 性能证书

> 本页是 time 性能数字的唯一信源。实现与选择理由分别见
> [`time-design`](../architecture/time-design.md) 和 [`guide/time`](../guide/time.md)。
> 当前最终实现形状固定在 `9129001`：Windows MSVC 19.50 与 WSL GCC 14.2 native-ext4
> Release 已完成 n=15；Apple M4 待独立真机复证。每个样本是独立进程，
> 判定以 `bench-stat.py` paired sign test（p < 0.01，且中位差 >1%）为主。
> TimerQueue 算法表采于 `bd29d57`；其后的 `9129001` 只补 bool duration 与 typed facade
> inline，不改变算法表使用的 raw queue 路径。normal-use 表已在 `9129001` 重采。

## 仪器与对手

| 被测面 | 实际入口 | 对手 / 防伪 |
|---|---|---|
| clock read | `bench/misc/misc_bench.cc` | 同工作量 `std::chrono`，不是空函数 |
| arbitrary timer 算法 | `bench/time/timer_queue_bench.cc` | faithful Seastar `timer_set` radix；另列 indexed eager-erase heap |
| 正常公开入口 | `bench/time/normal_usage_bench.cc` | 小 N、散布节点、增量循环、真实读钟、C ABI、对象生命周期 |
| thread sleep | 同上 | HYLE 与 `std::this_thread::sleep_for` 的 warm overshoot 分布 |
| one-duration timer | `timer_queue_bench.cc` | libevent-shaped intrusive TAILQ；另列 owner/order/link 同合同版本 |

timer 节点在 epoch 外预分配。Seastar 线按成熟库形状使用 circular sentinel 和
`constant_time_size=true`；indexed heap 维护真实 id→index 并 eager erase，不是 tombstone 假对手。
所有对手都完成 arm、取消或交付、清 owner 和 drain ready list 的完整生命周期。

## 1. Clock 与校准冷路

默认 clock header-inline 到与标准库相同的 OS primitive；目标是没有抽象退化，不是声称比同一个
primitive 更快。上一轮三平台证书（`14ee67e`）记录：默认 monotonic/realtime 与对应 chrono
同档；x86 profiling TSC 在 Windows/WSL 分别约 1.77×/1.84× throughput；Windows/Linux coarse
wall 分别约 5.19×/3.32×；M4 两个 opt-in availability 均为 false 并正确回退。名称重构没有
改变热读实现，但这组旧提交数字只作回归参照，最终三平台复证完成前不冒充当前点值。

TSC 无 exact CPUID frequency 时，fallback 现在用 1/2/4/... ms sleep-based sampling，不再
busy-spin。fresh-process 强制 fallback 的 15 次结果：

| 平台 | 首次调用 wall median（min–max） | thread CPU median（min–max） | 结论 |
|---|---:|---:|---|
| Windows | 3.703 ms（3.150–4.041） | 计时粒度下为 0 | 不再占满 CPU |
| WSL GCC 14.2 | 3.118 ms（3.084–3.215） | 0.030 ms（0.023–0.050） | CPU / wall 约 1% |

该测试只验证冷启动延迟/CPU 合同；已校准后的 profiling read 仍是独立热路。

## 2. TimerQueue：正常 cardinality

单位 ns/node，Windows n=15。H/S/heap 分别是 HYLE radix、同算法 Seastar-shape radix 和
indexed eager-erase heap；speedup 是 HYLE throughput / 对手 throughput。

| 操作 | N | H / S / heap | H vs S | H vs heap |
|---|---:|---:|---:|---:|
| arm+cancel | 1 | 3.141 / 4.195 / 2.286 | 1.336× WIN | 0.723× LOSE |
| arm+cancel | 8 | 4.165 / 4.501 / 5.675 | 1.074× WIN | 1.370× WIN |
| arm+cancel | 64 | 3.325 / 4.762 / 10.205 | 1.461× WIN | 3.083× WIN |
| arm+expire | 1 | 8.885 / 21.977 / 4.200 | 2.465× WIN | 0.475× LOSE |
| arm+expire | 8 | 4.740 / 9.095 / 9.755 | 1.921× WIN | 2.065× WIN |
| arm+expire | 64 | 4.818 / 6.757 / 11.351 | 1.402× WIN | 2.323× WIN |

N=1 indexed heap 的固定成本更低，这是真边界，不隐藏。HYLE 从 N=8 起胜出，且 N=1/8/64
均胜同复杂度 Seastar 形状；这支持 per-worker 多 timeout 的 radix 默认，不支持为了单 timer
微测增加第二套常态 metadata 或扭曲公开 API。

WSL GCC 14.2 n=15：

| 操作 | N | H / S / heap | H vs S | H vs heap |
|---|---:|---:|---:|---:|
| arm+cancel | 1 | 2.977 / 3.860 / 1.312 | 1.292× WIN | 0.439× LOSE |
| arm+cancel | 8 | 2.386 / 2.282 / 4.180 | 0.976× TIE | 1.794× WIN |
| arm+cancel | 64 | 2.547 / 2.645 / 6.314 | 1.033× WIN（sign） | 2.465× WIN |
| arm+expire | 1 | 6.278 / 19.103 / 1.839 | 3.044× WIN | 0.296× LOSE |
| arm+expire | 8 | 5.258 / 9.144 / 3.722 | 1.731× WIN | 0.716× LOSE |
| arm+expire | 64 | 3.769 / 6.205 / 6.302 | 1.644× WIN | 1.656× WIN |

所以“从 N=8 起均胜 heap”只在 Windows 成立；GCC 的 N=8 expire 仍由 heap 胜。更稳定的架构
结论是：radix 在 N=1 固定成本高，在几十个及大批可取消 timer 上占优；它不保证每个小 N 行
都赢，也不需要为这些交叉点新增公开类型。

## 3. TimerQueue：N=4096 算法门

Windows n=15，单位 ns/node：

| 操作 | HYLE | Seastar radix | indexed heap | 结论 |
|---|---:|---:|---:|---|
| arm+cancel | 3.401 | 5.037 | 15.395 | H/S 1.463×；H/heap 4.541×，15/15 WIN |
| arm+expire | 4.846 | 5.479 | 46.745 | H/S 1.141×，14/15 WIN；H/heap 9.642× |
| churn，70% cancel | 5.590 | 6.333 | 19.743 | H/S 1.132×；H/heap 3.556×，15/15 WIN |

WSL native-ext4 n=15，单位 ns/node：

| 操作 | HYLE | Seastar radix | indexed heap | 结论 |
|---|---:|---:|---:|---|
| arm+cancel | 2.623 | 3.372 | 6.540 | H/S 1.287×；H/heap 2.442×，15/15 WIN |
| arm+expire | 4.152 | 5.139 | 40.231 | H/S 1.238×，14/15 WIN；H/heap 9.549× |
| churn，70% cancel | 5.553 | 5.499 | 15.422 | H/S TIE；H/heap 2.823× WIN |
| reverse churn | 5.364 | 5.059 | — | H/S 0.937×，LOSE 15/15 |

WSL 复现了旧证书的 code-layout 边界：normal churn 平，reverse churn HYLE 慢约 6.3%。这不
支持给所有节点增加常态 metadata。上一轮 `14ee67e` 的 M4 两档 QoS 仅作方向参照；最终 M4
点值 pending。

## 4. 正常 event-loop 与 schedule_after

单位 ns/step 或 ns/op，均 n=15：

| 场景 | Windows | WSL | 评价 |
|---|---:|---:|---|
| 64-step incremental，连续节点 | 12.637 | 10.542 | — |
| 64-step incremental，对象内嵌散布节点 | 12.620 | 10.687 | 两平台与连续布局 TIE |
| absolute schedule + cancel | 3.618 | 3.293 | queue primitive |
| raw schedule_after，预采 now | 3.779 | 3.034 | queue primitive + saturating add |
| chrono schedule_after，预采 now | 3.745 | 4.079 | Windows TIE；WSL 0.738× LOSE |
| raw schedule_after，真实读 monotonic | 18.037 | 19.761 | 正常端到端路径 |
| chrono schedule_after，真实读 clock | 18.055 | 19.718 | 两平台与 raw TIE |

真实 `schedule_after` 的大头是读钟，不是 3–4ns queue primitive。GCC 的显式预采样 typed
路径多约 1ns；反汇编确认最终代码已完整 inline，差异来自 signed time-point clamp/代码放置，
而默认真实读钟路径统计平。不能为这条微路径删除负 time-point 的正确 clamp。散布
stable-address 节点没有系统性退化；因此连续 N=4096 数字不是靠 benchmark-only layout 得到的。

## 5. C ABI 成本

经真实 shared library、opaque handle 和共享 `hyle_status` 合同，均 n=15：

| 场景 | Windows | WSL |
|---|---:|---:|
| C++ schedule+cancel | 3.618 ns | 3.144 ns |
| C ABI schedule+cancel+Status | 11.996 ns | 11.024 ns |
| C ABI schedule_after clock+cancel+Status | 20.234 ns | 20.964 ns |
| C ABI schedule+advance+visitor，fire one | 16.450 ns | — |
| C++ new/delete `TimerEntry` | 21.349 ns | 7.349 ns |
| C opaque timer create/destroy+Status | 26.005 ns | 15.663 ns |

这不是“零成本 C wrapper”：可恢复错误、DLL 接缝和 opaque 生命周期有真实常数税。但 queue
操作仍 O(1)、无分配，opaque timer 只在创建/销毁时分配。没有大规模 C timer consumer 前，
不为表变绿削弱 Status 合同，也不凭空加入 allocator/bulk API；有消费证据时再评估该 P1。

## 6. Thread sleep overshoot

Windows n=15；每进程 warm 采样，表中是跨进程分位数的中位数，单位 ms；early 为 0。

| request | HYLE p50 / p95 / p99 | `std::sleep_for` p50 / p95 / p99 | 结论 |
|---|---:|---:|---|
| 100 us | 0.413 / 0.463 / 0.612 | 15.426 / 16.008 / 16.394 | HYLE 高分辨率 waitable timer 明显更准 |
| 1 ms | 0.517 / 0.632 / 0.912 | 14.523 / 15.095 / 15.504 | 同上 |
| 10 ms | 0.256 / 0.578 / 0.578 | 5.514 / 11.869 / 11.869 | 同上 |

WSL native Linux 的 HYLE/std overshoot（ms）同档：

| request | HYLE p50 / p95 / p99 | `std::sleep_for` p50 / p95 / p99 |
|---|---:|---:|
| 100 us | 0.058 / 0.076 / 0.106 | 0.058 / 0.071 / 0.100 |
| 1 ms | 0.065 / 0.103 / 0.138 | 0.063 / 0.102 / 0.139 |
| 10 ms | 0.076 / 0.126 / 0.126 | 0.077 / 0.129 / 0.129 |

两平台所有样本 early=0。这里衡量 OS thread blocking wake quality，不是 scheduler/fiber timer；
接口保证不主动早醒，不承诺实时调度。Apple overshoot 分布 pending。

## 7. CommonTimeoutQueue

最终实现未改 Common 核；上一轮 `14ee67e` 的三平台结构证据是：Windows 对 checked/naked
TAILQ 均胜；WSL 对 checked 同档、对 naked 胜；M4 对 checked 胜，但 naked expire 下界快约
22%。owner/order/link 检查与可恢复 C Status 是合同，不为裸链下界删除。该结构尚无生产
fixed-idle-timeout consumer，接入时仍需 application-level 证书。最终形状的三平台复跑 pending。

## 8. 当前结论与未完成证据

| 项 | 状态 |
|---|---|
| exact duration conversion | 整数路径精确饱和并覆盖 bool Rep；Windows/WSL 正确性门已通过 |
| 默认/opt-in clock | 语义形状已定；旧热读证据强，最终点值待 Apple 复跑 |
| TimerQueue 算法 | SOTA-class radix、O(1) arm/cancel；Windows/WSL 小 N 和 N=4096 已复证 |
| 正常 event-loop | Windows/WSL 连续/散布同档；真实 clock 成本已单列 |
| C ABI | 能力覆盖完整，正常操作常数税已公开；不把 DLL/Status 成本伪装为零 |
| thread sleep | Windows 高分辨率路径明显优于 std；Linux 与 std 同档；Apple pending |
| strict 性能收口 | **未最终关闭**：只欠 Apple M4 在固定提交上的独立复证 |

## 9. 复跑

```bash
scripts/bench-stat.py <build>/bin/hyle_timer_queue_bench -n 15 \
  --pair "Seastar-shape radix N=1" "HYLE radix N=1" \
  --pair "indexed heap N=1" "HYLE radix N=1" \
  --pair "radix timer_set (Seastar shape)" "HYLE TimerQueue (radix)" \
  --pair "indexed min-heap (libevent/libuv)" "HYLE TimerQueue (radix)"

scripts/bench-stat.py <build>/bin/hyle_time_normal_bench -n 15 \
  --pair "HYLE contiguous TimerEntry" "HYLE heap-embedded scattered TimerEntry" \
  --pair "schedule_after presampled + cancel" "chrono schedule_after presampled + cancel" \
  --pair "C++ schedule+cancel" "C ABI schedule+cancel+Status" \
  --pair "C++ new/delete TimerEntry" "C opaque timer create/destroy+Status"
```

Linux 正式性能必须在原生 ext4 worktree 跑，不能用 `/mnt/c` 9p。Windows、WSL 同物理机必须
错开。Apple 默认 user-interactive QoS，并用 `HYLE_BENCH_APPLE_QOS=utility` 做低优先级复核。
