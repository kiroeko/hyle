# time 实测铁证

> 本页是 [performance 总账](./README.md) 的 **time 模块**详节。  
> 源：`bench/misc/misc_bench.cc`（clock）+ `bench/time/timer_wheel_bench.cc`（wheel / common queue）。  
> **权威收口**：2026-07-19 三平台 `bench-stat -n15`（见 [`../milestones/time-module-closure.md`](../milestones/time-module-closure.md)）。  
> 源码时间戳与证书同日；本轮整理以**拆文档 + 当前真相**为主，未因结构问题重跑。  
> 旧「塞在 mem-container 次要模块」的 clock 双平台片段作废，以本页与总账为准。

## 模块面

| 能力 | 头 | 标尺 |
|---|---|---|
| `monotonic_ns` / `realtime_ns` | `clock.h` | 同工作量 `std::chrono` |
| `fast_monotonic_ns`（opt-in TSC） | 同上 | 相对 HYLE monotonic（非默认） |
| `fast_realtime_ns`（opt-in coarse） | 同上 | 相对 HYLE realtime |
| `TimerWheel` 8×64 | `timer_wheel.h` | indexed eager-erase heap（libevent/libuv 形） |
| `CommonTimeoutQueue` 固定时长 | `common_timeout_queue.h` | 裸 TAILQ / checked TAILQ |

## 1. Clock（读时钟）

| 操作 | Win / Lin / M4 (ns) | 对照 | 判定 |
|---|---|---|---|
| **monotonic_ns** | 13.22 / 13.07 / 12.93 | steady_clock 13.30 / 13.07 / 12.88 | ➖ **三平台统计平**（同 OS 原语） |
| **realtime_ns** | ~15 / ~13 / ~14 档 | system_clock 同档 | ➖ **平** |
| **fast_monotonic_ns** | 7.37 / 7.13 / 12.93 | vs monotonic 1.79x / 1.84x / 1.0x | ✅ x86 ~1.8x；**M4 无 invariant TSC → 透明回退** |
| **fast_realtime_ns** | 3.20 / 3.90 / 13.64 | vs realtime 4.94x / 3.36x / 1.0x | ✅ Win/Lin 粗粒度钟；**macOS 无 coarse → 回退** |

**要点**：

1. 默认路径目标是 **与 std 同档**，不是「比 OS 更快」——标尺就是同一 syscall。  
2. Win 曾输 steady_clock **1.48x**（无条件两次 64 位除法）；已抄 MSVC 频率特判追平（objdump/dumpbin 坐实）。  
3. fast_* 是 **opt-in**：可跳变 / 粗粒度，只给 profiling 或可接受误差的路径，不是默认契约。

## 2. TimerWheel（N=4096，n=15）

对手 = **indexed min-heap + 真实 keyed eager erase**（无墓碑假对手）。

| 负载 | Win HYLE vs heap | Lin | M4 | 判定 |
|---|---|---|---|---|
| arm+cancel | 6.86 / 10.42（**1.52x**） | 5.95 / 6.73（**1.13x**） | 4.91 / 8.36（**1.70x**） | ✅ 三平台赢 |
| arm+expire | 21.34 / 49.80（**2.33x**） | 22.05 / 40.64（**1.84x**） | 11.83 / 23.74（**2.01x**） | ✅ 三平台赢 |
| churn 70% cancel 再 expire | 12.87 / 20.85（**1.62x**） | 13.63 / 16.01（**1.17x**） | 7.62 / 11.03（**1.45x**） | ✅ 三平台赢 |

**复杂度**：arm/cancel **O(1)**（轮 + 桶）；cascade 按触发推进。相对堆的 O(log n) keyed erase，这是 **结构阶优势**，不是测噪声。

## 3. CommonTimeoutQueue（固定时长，N=4096，n=15）

| 负载 | Win | Lin | M4 | 判定 |
|---|---|---|---|---|
| arm+expire / **裸** TAILQ | 3.28 / 4.22（1.29x） | 3.14 / 3.95（1.29x） | 3.52 / 2.73（**0.77x**） | ⚠️ **M4 输裸下界** |
| arm+cancel / 裸 TAILQ | 2.70 / 3.13（1.14x） | 1.99 / 2.59（1.30x） | 3.06 / 3.06（平） | ✅ 不输 |
| arm+expire / **checked** TAILQ（同合同） | 3.28 / 3.61（1.10x） | 3.14 / 3.21（平） | 3.52 / 4.21（**1.16x**） | ✅ **同语义三平台不输** |

**唯一公开非 SOTA 叙事**：

- 对 **同 owner/order/link 检查** 的 TAILQ：三平台不输（M4 还 1.16x）。  
- 对 **什么检查都不做的裸 TAILQ**：M4 expire **0.77x**——这是 **安全语义成本**（owner、顺序、lifecycle），不是「还能无代价再快一档」。  
- 曾试 null-ended / membership tag 链表：M4 expire 无改善、cancel 变差 → **撤回**。  
- Common queue **尚未接生产 io**；接入前应用真实 fixed-timeout 负载再验。

## 4. 非 SOTA / 边界清单

| 项 | 状态 | 说明 |
|---|---|---|
| 默认 clock vs std | ➖ 平 | 目标达成 |
| TimerWheel vs heap | ✅ 赢 | 阶优势 |
| Common vs 同合同 TAILQ | ✅ 不输 | — |
| Common vs 裸 TAILQ @M4 expire | ⚠️ 0.77x | 有意安全税；勿写成可随意优化掉 |
| fast_* @M4 | 回退同档 | 无硬件原语 |
| 亚毫秒唤醒 | 不做承诺 | wheel 公开 ns，执行粒度 1ms；不早触发 |
| 跨线程 timer 容器 | 不是 | owner-thread only |

## 5. 复跑

```bash
# clock
scripts/bench-stat.py <build>/bin/hyle_misc_bench -n 15 \
  --pair "std::chrono::steady_clock::now" "HYLE monotonic_ns" \
  --pair "std::chrono::system_clock::now" "HYLE realtime_ns" \
  --pair "HYLE monotonic_ns" "HYLE fast_monotonic_ns (rdtsc)"

# wheel + common
scripts/bench-stat.py <build>/bin/hyle_timer_wheel_bench -n 15 \
  --pair "indexed min-heap (libevent/libuv)" "HYLE timing wheel" \
  --pair "intrusive TAILQ (libevent common_timeout shape)" "HYLE CommonTimeoutQueue" \
  --pair "checked intrusive TAILQ (HYLE contracts)" "HYLE CommonTimeoutQueue"
```

仪器细节、order-swap 防伪、预分配节点等见 `time-module-closure.md` §5。
