# time 性能证书

> **2026-08-25 freshness gate：**本页点值绑定下述历史提交；time 的产品收口状态未改变，但
> 依赖升级和 delivery/lifetime 修复使受影响性能边必须重签。clock 热读、raw queue 与未改路径可作为回归输入，但不是
> 当前 working tree 的自动性能证书；尤其 C `schedule+advance+visitor` 已改为同批 callback
> 可安全 cancel/reschedule/destroy peer 的 mutation-safe 交付，旧 16.450 ns 只能解释旧实现。
> 最终三平台 n=15 重采完成前，不据本页旧数字宣称当前依赖组合已有新鲜 SOTA 点值。
> core 的 2026-08 直接消费者复验再次观察到 M4 N=8 cancel 的已知差距；该行还包含 HYLE
> `size/next-deadline/owner` 合同而非纯 list unlink，不能归作 core 同合同结论。本轮只给
> small-cardinality harness 补 `HYLE_BENCH_RIVAL_FIRST` 正反序守门；点值和 time 模块裁决仍待
> time 自己的完整复审，不能由 core 证书代签。

### 2026-08 core 受影响边诊断（不是 time 证书）

M4 用修正后的 small-cardinality 顺序守门跑了 interactive/utility × HYLE-first/peer-first，
每格 n=15、3% margin。下表 ratio 为 Seastar-shape op/s ÷ HYLE op/s；runner 先按数值合同输出，
随后产品审查再裁决语义资格：

| QoS / 顺序 | N=8 cancel ratio / CI | runner exact / gate |
|---|---|---|
| interactive / HYLE-first | 1.017 / `[1.010,1.025]` | `EQUIVALENT/PASS` |
| interactive / peer-first | 1.095 / `[1.085,1.121]` | `LOSE/FAIL` |
| utility / HYLE-first | 1.021 / `[1.015,1.025]` | `EQUIVALENT/PASS` |
| utility / peer-first | 1.099 / `[1.095,1.110]` | `LOSE/FAIL` |

HYLE 四格稳定在 2.636--2.637 ns/node；Seastar-shape 在其后运行约 2.584--2.588 ns，先运行则
约 2.399--2.405 ns，顺序效应几乎全部落在 peer。更重要的是，HYLE 同时维护公开的全局
`size()`、空转移 `next_deadline` 复位和 owner/linked 合同，而该 dependency-free peer 没有等强
语义。故产品级资格是 `NOT_COMPARABLE/N/A`，数值合成只说明
`INCONCLUSIVE/placement-sensitive`，不能签 N=8 SOTA，也不能把它归咎于 core list。N=1/N=64
cancel 及 N=1/8/64 expire 的四格诊断均为 HYLE WIN；它们同样只供 time 后续完整复审归因。

> 本页是 time 性能数字的唯一信源。实现与选择理由分别见
> [`time-design`](../architecture/time-design.md) 和 [`guide/time`](../guide/time.md)。
> 当前 clock 产品实现固定在 `e58d3f5`，x86-only test 修正在 `e16a927`，Apple QoS 仪器固定
> 在 `3b811e8`：Windows MSVC 19.50、WSL GCC 14.2 native-ext4、Apple M4 AppleClang 21
> Release 均完成 n=15。
> M4 同时跑 user-interactive 与 utility QoS；每个样本是独立进程，
> 判定以 `bench-stat.py` paired sign test（p < 0.01，且中位差 >1%）为主。
> Windows/WSL TimerQueue 算法表采于 `bd29d57`；其后的 clock/ABI 增量不改变算法表使用的
> raw queue 路径。normal-use 表已在
> `9129001` 重采；M4 两表使用最终 Apple 仪器 `ba61eb5`。caller-storage 行是本轮最终候选在
> 同一 Windows 工具链上的新增 n=15；未改写其它平台的历史列。

## 仪器与对手

| 被测面 | 实际入口 | 对手 / 防伪 |
|---|---|---|
| clock read / profiling boundary | `bench/time/clock_bench.cc` | 同工作量 `std::chrono`；raw sampling 与 ordered boundary 分表，不跨合同判胜负 |
| arbitrary timer 算法 | `bench/time/timer_queue_bench.cc` | faithful Seastar `timer_set` radix；另列 indexed eager-erase heap |
| 正常公开入口 | `bench/time/normal_usage_bench.cc` | 小 N、散布节点、增量循环、真实读钟、C ABI、对象生命周期 |
| thread sleep | 同上 | HYLE 与 `std::this_thread::sleep_for` 的 warm overshoot 分布 |
| one-duration timer | `timer_queue_bench.cc` | libevent-shaped intrusive TAILQ；另列 owner/order/link 同合同版本 |

timer 节点在 epoch 外预分配。Seastar 线按成熟库形状使用 circular sentinel 和
`constant_time_size=true`；indexed heap 维护真实 id→index 并 eager erase，不是 tombstone 假对手。
所有对手都完成 arm、取消或交付、清 owner 和 drain ready list 的完整生命周期。

## 1. Clock 与校准冷路

默认 clock header-inline 到与标准库相同的 OS primitive；目标是没有抽象退化，不是声称比同一个
primitive 更快。`e58d3f5` 把 ARM64 architected virtual counter 纳入 profiling，并把 raw sampling
与 ordered measurement 分开。下表单位 ns/read（pair 行为 ns/pair），均为 n=15 中位数：

| 平台 / QoS | std steady | HYLE monotonic | raw profiling | raw / std throughput | std steady pair | ordered pair | std system / HYLE realtime / coarse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows | 12.754 | 12.796 | **7.188** | **1.77× WIN** | 24.770 | 30.764 | 15.137 / 15.176 / **2.913** |
| WSL / CPU2 | 12.447 | 12.520 | **6.769** | **1.84× WIN** | 25.019 | 28.241 | 12.475 / 12.432 / **3.745** |
| M4 interactive | 12.929 | 12.930 | **1.132** | **11.42× WIN** | 24.517 | 33.088 | 13.555 / 13.561 / 12.917 |
| M4 utility | 12.939 | 12.939 | **1.132** | **11.42× WIN** | 24.513 | 33.107 | 13.546 / 13.670 / 12.947 |

默认 monotonic/realtime 三平台与 std 同档。Windows/Linux coarse 是真正 native coarse source，
分别约 5.20×/3.33×；macOS availability=false，表中 `coarse` 只是正确 fallback，约 1.05× 的
点值不能宣称原生 coarse 能力。M4 的 CNTFRQ 为 1GHz，counter tick 已是 ns，所以 raw sampling
无需 fixed-point scale；interactive/utility 同向，不能据此声称硬 P/E 核绑定。

ordered pair 比 std 双读慢 11--26%，但二者不是同合同：std 双读不阻止待测代码跨读钟重排；
HYLE x86 pair 使用 LFENCE/RDTSC/RDTSCP，ARM64 pair 使用 ISB，购买的是精确区域边界。Folly/
Abseil 的 x86 measurement pair 采用同一序列；不存在同能力 std 入口，因此这列只报告绝对成本，
不把与无序 pair 的数值差写成 SOTA LOSE，也不借强合同掩盖成本。

x86 TSC 无 exact CPUID frequency 时，生产 fallback 用 1/2/4/... ms sleep-based sampling，不再
busy-spin。test-only forced path 从 32ms 窗口开始，让 Windows 15.625ms thread-accounting 粒度
能可靠区分 sleep 与 busy work；Windows/WSL fresh-process 门均约 96ms wall 且满足低 CPU 合同。
这 96ms 不是 shipping 首调延迟。ARM64 无 TSC calibration，M4 不再注册该 x86-only CTest；它由
counter availability/nondecreasing 真机用例覆盖。

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

Apple M4 user-interactive n=15：

| 操作 | N | H / S / heap | H vs S | H vs heap |
|---|---:|---:|---:|---:|
| arm+cancel | 1 | 3.267 / 3.329 / 3.648 | TIE | TIE |
| arm+cancel | 8 | 2.695 / 2.352 / 7.550 | 0.878× LOSE | 2.798× WIN |
| arm+cancel | 64 | 2.577 / 3.075 / 12.261 | 1.188× WIN（sign） | 4.671× WIN |
| arm+expire | 1 | 7.014 / 12.859 / 3.826 | 1.844× WIN | 0.545× LOSE |
| arm+expire | 8 | 3.426 / 5.811 / 6.443 | 1.686× WIN | 1.855× WIN |
| arm+expire | 64 | 3.228 / 6.342 / 11.464 | 1.957× WIN | 3.540× WIN |

utility 的关键交叉点同向：N=8 arm+cancel 为 2.699 / 2.501 / 7.483，HYLE 对 Seastar
0.928× LOSE、对 heap 2.771× WIN。N=1 的进程间 spread 仍大，故只判 TIE；不把 QoS 或可能的
P/E 调度当作物理 affinity。

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
支持给所有节点增加常态 metadata。

Apple M4 两档 QoS，单位 ns/node：

| QoS / 操作 | HYLE | Seastar radix | indexed heap | 结论 |
|---|---:|---:|---:|---|
| interactive arm+cancel | 3.771 | 4.170 | 8.336 | H/S 1.096× WIN；H/heap 2.199× WIN |
| interactive arm+expire | 5.229 | 7.400 | 23.419 | H/S 1.430×；H/heap 4.479× WIN |
| interactive churn | 4.049 | 4.710 | 10.998 | H/S 1.158×；H/heap 2.724× WIN |
| interactive reverse churn | 4.107 | 4.683 | — | H/S 1.153× WIN |
| utility arm+cancel | 4.150 | 4.378 | 8.339 | H/S 1.065× WIN；H/heap 2.012× WIN |
| utility arm+expire | 5.212 | 7.362 | 23.765 | H/S 1.412×；H/heap 4.509× WIN |
| utility churn | 3.778 | 4.717 | 11.250 | H/S 1.239×；H/heap 2.885× WIN |
| utility reverse churn | 3.796 | 4.780 | — | H/S 1.260× WIN |

## 4. 正常 event-loop 与 schedule_after

单位 ns/step 或 ns/op，均 n=15：

| 场景 | Windows | WSL | M4 interactive | 评价 |
|---|---:|---:|---:|---|
| 64-step incremental，连续节点 | 12.637 | 10.542 | 15.389 | — |
| 64-step incremental，对象内嵌散布节点 | 12.620 | 10.687 | 9.405 | x86 TIE；M4 scattered 1.629× WIN |
| absolute schedule + cancel | 3.618 | 3.293 | 3.290 | queue primitive |
| raw schedule_after，预采 now | 3.779 | 3.034 | 3.119 | queue primitive + saturating add |
| chrono schedule_after，预采 now | 3.745 | 4.079 | 4.579 | Win TIE；WSL/M4 0.738×/0.690× LOSE |
| raw schedule_after，真实读 monotonic | 18.037 | 19.761 | 22.971 | 正常端到端路径 |
| chrono schedule_after，真实读 clock | 18.055 | 19.718 | 23.398 | 三平台与 raw TIE |

真实 `schedule_after` 的大头是读钟，不是 3–4ns queue primitive。GCC 的显式预采样 typed
路径多约 1ns；反汇编确认最终代码已完整 inline，差异来自 signed time-point clamp/代码放置，
而默认真实读钟路径统计平。不能为这条微路径删除负 time-point 的正确 clamp。散布
stable-address 节点没有系统性退化；因此连续 N=4096 数字不是靠 benchmark-only layout 得到的。
M4 的 scattered 对象反而更快，说明真实 object-embedded layout 不是弱项；packed 48-byte entry
跨 128-byte cache-line 的布局才是其边界。utility 复核仍为 15.390 / 9.479，同向。

## 5. C ABI 成本

经真实 shared library、opaque handle 和共享 `hyle_status` 合同，均 n=15：

| 场景 | Windows | WSL | M4 interactive |
|---|---:|---:|---:|
| C++ schedule+cancel | 3.618 ns | 3.144 ns | 3.263 ns |
| C ABI schedule+cancel+Status | 11.996 ns | 11.024 ns | 7.915 ns |
| C ABI schedule_after clock+cancel+Status | 20.234 ns | 20.964 ns | 22.624 ns |
| C ABI schedule+advance+visitor，fire one（旧交付实现，待重测） | 16.450 ns | — | — |
| C++ new/delete `TimerEntry` | 21.349 ns | 7.349 ns | 15.449 ns |
| C opaque timer create/destroy+Status | 26.005 ns | 15.663 ns | 18.999 ns |
| C caller-storage timer init/deinit+Status | **9.156 ns** | **3.314 ns** | — |

这不是“零成本 C wrapper”：可恢复错误、DLL 接缝和 opaque 生命周期有真实常数税。但 queue
操作仍 O(1)、无分配。默认 opaque timer 只在创建/销毁时分配；新增的 caller-storage 形状参考
libevent `event_assign`，让 arena/slab 用户保持布局 opaque 而不逐 timer 分配。相对默认
create/destroy，Windows 同机 n=15 为 **2.925× WIN**（15/15，范围 2.733–3.051），WSL GCC14
native-ext4 为 **4.532× WIN**（15/15，范围 4.328–4.827）；这不是删掉 Status 得到的 benchmark
特供入口，两条路径有相同的生命周期错误合同。

本轮还试做过 queue-level single-timer fast slot。它能消除 N=1 radix 固定成本，但同一最终二进制
让 N=8 与 N=4096 churn 相对 Seastar 的主 workload 退化，故实现已撤回。N=1 对 heap 的 LOSE
继续作为算法交叉点保留，不为一行表格牺牲默认 server workload。

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
接口保证不主动早醒，不承诺实时调度。

M4 overshoot（ms），两档 QoS 均 early=0：

| QoS / request | HYLE p50 / p95 / p99 | `std::sleep_for` p50 / p95 / p99 | 结论 |
|---|---:|---:|---|
| interactive 100 us | 0.052 / 0.056 / 0.058 | 0.052 / 0.055 / 0.058 | TIE |
| interactive 1 ms | 0.515 / 0.519 / 0.521 | 0.515 / 0.521 / 0.539 | TIE |
| interactive 10 ms | 5.031 / 5.041 / 5.041 | 5.033 / 5.043 / 5.043 | TIE |
| utility 100 us | 0.810 / 0.818 / 0.820 | 0.812 / 0.817 / 0.825 | TIE |
| utility 1 ms | 8.017 / 8.041 / 8.046 | 8.016 / 8.040 / 8.046 | TIE |
| utility 10 ms | 70.259 / 80.042 / 80.042 | 74.286 / 80.041 / 80.041 | TIE |

utility 的强 coalescing 是 macOS 低优先级节能策略，HYLE 与 std 同行为；需要低延迟 blocking
sleep 的线程应选择交互/高优先级策略，fiber 仍应使用 sched 而不是阻塞 worker。

## 7. CommonTimeoutQueue

最终实现未改 Common 核；Windows 对 checked/naked TAILQ 均胜，WSL 对 checked 同档、对
naked 胜。M4 interactive 为 3.473ns，对 checked 4.287ns 为 1.236× WIN，对 naked 2.804ns
则约慢 24%；utility 为 3.515 / 4.293 / 2.822，同向。owner/order/link 检查与可恢复 C Status
是合同，不为裸链下界删除。该结构尚无生产 fixed-idle-timeout consumer，接入时仍需
application-level 证书。

## 8. 当前结论与证据边界

| 项 | 状态 |
|---|---|
| exact duration conversion | 整数路径精确饱和并覆盖 bool Rep；Windows/WSL/Apple 正确性门已通过 |
| 默认/opt-in clock | 语义形状已定；默认路径同 OS primitive，opt-in availability 合同不变 |
| TimerQueue 算法 | SOTA-class radix、O(1) arm/cancel；旧三平台证据是回归输入，当前五态重签待 time 复审 |
| 正常 event-loop | 三平台真实 clock 成本已单列；M4 scattered 优于 packed contiguous |
| C ABI | 能力覆盖完整，正常操作常数税已公开；不把 DLL/Status 成本伪装为零 |
| thread sleep | Windows 高分辨率路径明显优于 std；Linux/M4 与 std 同档；三平台 early=0 |
| strict 性能收口 | 🟡 待 time 完整复审；WSL reverse churn 是历史差距，M4 N=8 新证据因不同合同与注册顺序敏感降为 `NOT_COMPARABLE` 诊断 |

## 9. 复跑

```bash
scripts/bench-stat.py <build>/bin/hyle_timer_queue_bench -n 31 \
  --require-clean --equivalence-margin-pct 3 \
  --pair "HYLE radix N=1" "Seastar-shape radix N=1" \
  --pair "HYLE radix N=1" "indexed heap N=1" \
  --pair "HYLE TimerQueue (radix)" "radix timer_set (Seastar shape)" \
  --pair "HYLE TimerQueue (radix)" "indexed min-heap (libevent/libuv)"

HYLE_BENCH_RIVAL_FIRST=1 scripts/bench-stat.py \
  <build>/bin/hyle_timer_queue_bench -n 31 \
  --require-clean --equivalence-margin-pct 3 \
  --pair "HYLE radix N=1" "Seastar-shape radix N=1" \
  --pair "HYLE radix N=1" "indexed heap N=1" \
  --pair "HYLE TimerQueue (radix)" "radix timer_set (Seastar shape)" \
  --pair "HYLE TimerQueue (radix)" "indexed min-heap (libevent/libuv)"

scripts/bench-stat.py <build>/bin/hyle_time_normal_bench -n 15 \
  --pair "HYLE contiguous TimerEntry" "HYLE heap-embedded scattered TimerEntry" \
  --pair "schedule_after presampled + cancel" "chrono schedule_after presampled + cancel" \
  --pair "C++ schedule+cancel" "C ABI schedule+cancel+Status" \
  --pair "C++ new/delete TimerEntry" "C opaque timer create/destroy+Status" \
  --pair "C opaque timer create/destroy+Status" "C caller-storage timer init/deinit+Status"
```

Linux 正式性能必须在原生 ext4 worktree 跑，不能用 `/mnt/c` 9p。Windows、WSL 同物理机必须
错开。Apple 默认 user-interactive QoS，并用 `HYLE_BENCH_APPLE_QOS=utility` 做低优先级复核。
