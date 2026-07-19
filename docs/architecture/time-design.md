# time 模块架构与维护指南

> 当前实现依据：`include/hyle/time/`、`src/time/`、`include/hyle/c/time.h`。
> 功能里程碑见 [`time-01`](../milestones/time-01-clock.md) /
> [`time-02`](../milestones/time-02-timer-wheel.md)，收口证据见
> [`time-module-closure`](../milestones/time-module-closure.md)。

## 1. 依赖位置

```text
core/
  └─ intrusive_list.h + macros
          │
          ▼
time/  ──────────────────────────────┐
  clocks                             │
  TimerEntry / TimerSource           │
  TimerWheel / CommonTimeoutQueue    │
          │                          │
          ├──────────────┐           │
          ▼              ▼           ▼
sched/ timers       io/ timed ops   C ABI
                                      include/hyle/c/time.h
```

`time` 没有编译期 HYLE 模块依赖。侵入链原语必须留在 `core/`；把它放回
`container/` 会制造 `time → container`，与
`core/error → platform → {mem,time,simd}` 的 DAG 冲突。

## 2. 两个公开面

```text
同工具链 / 源码用户                    稳定 ABI / FFI 用户
include/hyle/time/*.h                  include/hyle/c/time.h
        │                                      │
        │ header-inline 标准钟                 │ extern "C" wrapper
        │ C++ 类、concept、侵入节点             │ uint64/POD/callback/opaque handle
        └──────────────────┬───────────────────┘
                           ▼
                    src/time/clock.cc
                    src/time/c_time.cc
```

- C++ 面追求内联、静态 concept 和零分配；不承诺跨编译器 ABI。
- C 面是发布接缝。不能暴露 `TimerEntry` 布局、STL、模板、异常或 C++ 函数类型。
- C ABI 的 timer handle 创建时分配一次；schedule/cancel/advance 不分配。

## 3. 时钟分层

| 入口 | 语义 | 可用于 deadline | 实现 |
|---|---|---:|---|
| `monotonic_ns` | 稳态单调钟 | 是 | C++ header-inline `steady_clock` |
| `realtime_ns` | Unix epoch 墙钟 | 否 | C++ header-inline `system_clock` |
| `fast_monotonic_ns` | profiling 快照 | 否 | x86 invariant TSC；否则回退 |
| `fast_realtime_ns` | coarse 日志时间戳 | 否 | Win shared data / Linux coarse；mac 回退 |

deadline 算术只走 `saturating_add_ns`、`deadline_after_ns`、
`remaining_ns`。禁止在下游重新写 `monotonic_ns() + timeout`。

## 4. timer 数据流

```text
caller owns stable TimerEntry
        │ schedule(deadline)
        ▼
exactly one TimerSource owns it
        │
        ├─ cancel ───────────────► caller owns unlinked entry
        │
        └─ advance(now, fired)
                 │ source clears owner before hand-off
                 ▼
           caller-owned TimerList
                 │ pop_front
                 ▼
         wake / callback / re-arm / destroy
```

硬不变量：

1. `TimerEntry` 不可 copy/move，地址从 arm 到摘链必须稳定。
2. 同时最多属于一个 source；owner 是精确指针，不是 bool。
3. source 析构必须为空；entry 析构必须未调度且未挂在 fired list。
4. source 只交付节点，不执行用户代码；回调属于 sched/io/C adapter。
5. 所有 source 单线程拥有。跨线程操作由上层 completion/command spine 投递。

## 5. 两种 source

| 条件 | 选择 | 关键结构 |
|---|---|---|
| 任意/混合时长 | `TimerWheel`（默认） | 8×64、1ms tick、occupied bitmap |
| 大量相同固定时长 | `CommonTimeoutQueue` | deadline 非递减的侵入链 |

`TimerWheel` 的 8 层不是装饰：6×64 只能覆盖约 2.18 年，无法兑现公开
`uint64_t ns` 域的“永不早触发”。8×64 覆盖完整域；deadline 向上取整，
advance 的 now 向下取整，最多晚 1ms。

`CommonTimeoutQueue` 只适用于 append-order 成立的负载。显式 deadline 用
`try_schedule`，拒绝时必须保留旧 arm。Apple M4 上，完整安全 API 对不检查
owner/order/link 的裸 TAILQ 下界仍为约 0.77x，但对相同合同的 checked TAILQ
为约 1.16x；维护时必须同时报告“裸下界”和“同语义实现”，不能删合同换数字。

## 6. 修改时的硬门

- 改 clock：三平台对 `std::chrono` 同工作量 benchmark；不能比较不同单位/
  转换工作。
- 改 wheel：覆盖 former horizon、`UINT64_MAX`、cascade、FIFO、wrong-owner。
- 改链表/entry：跑 time + container + sched + io timeout，并跑 ASan/UBSan。
- 改 C ABI：纯 C 编译测试；机械核对头/实现/测试/guide；核对动态导出符号。
- 改 benchmark：节点预分配、两侧同生命周期、arm/consume 之间保留 compiler
  barrier；heap 必须是 indexed eager erase；common 同时保留裸 TAILQ 下界、
  checked TAILQ 同语义线和 M4 order-swap。
- 性能结论逐平台写。一个 ISA 赢不能外推到另一个 ISA。
