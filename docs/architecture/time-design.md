# time 模块架构

> **V2 状态：`NOT_REVIEWED`。** 本页暂作既有设计输入，不代表当前功能、性能或 API 证书。
> 当前只提供 C++23 source/static/shared，Native FFI 为 `N/A`。BUILD-03 已删除旧 C schema、
> bridge、动态导出和 fixture；正文的其他验收叙事仍待 time 轮次重裁决。

> 这里维护当前整体设计，不记录逐轮施工流水，也不复制性能点值。任务记录见
> [`time-module-closure`](../implementation/tasks/time/time-module-closure.md)，正式数字只见
> [`benchmarks-time`](../implementation/evidence/performance/benchmarks-time.md)，用户选型见
> [`guide/time`](../user/time.md)。

## 定位与依赖

`time` 是独立底层模块：拥有“现在是什么时间、如何形成 deadline、如何保存大量
deadline”这些产品语义。原生 C++ timer 通过独立 header-only 目标
[`hyle::container_intrusive`](container-intrusive-design.md) 使用
`<hyle/container/intrusive_list.h>`。time 的公开头直接使用 Core compiler/invariant contract，因此
中央图记录 `time -> PUBLIC {core, container_intrusive}`。整个模块不依赖 `error`、`platform` 二进制模块、
完整 `container`、`sched` 或 `io`；Core 与 container_intrusive 都是有真实 target 边的
header-only INTERFACE 组件，不是 target-neutral include 例外。

```text
core + container_intrusive ──► native time ── clock / deadline / sleep
  (header-only)               └─ TimerEntry + TimerQueue + CommonTimeoutQueue
native time ──────────────────────────► sched（fiber sleep、timed wait）
native time ──────────────────────────► io（经 sched 做 timer-vs-I/O 仲裁）
```

`TimerEntry` 的 private safe hook 和公开 `TimerList` 别名都使用 `hyle::container` 类型，所以这条边必须
是 `PUBLIC`，不能依靠安装时复制整棵 include tree 偶然通过。完整 `hyle::container` 不聚合该
轻量 target；依赖方向也不是 `time -> container`，这正是拆 target 所保护的底层闭包。

`platform::Parker` 的 timed wait 是 OS 阻塞机制，不拥有时钟；但 `platform::FileInfo` 的公开头直接
使用 `time::UnixTimestamp`，所以中央图诚实记录 `platform -> PUBLIC time`，同时不取得
clock/deadline/timer 策略。`sched` 组合两者：从 timer queue 取得下一 deadline，再把剩余时长交给
Parker。这个公开 header ownership 边不能因值类型没有额外调用成本而省略。

## 能力分层

| 层 | 公开能力 | 设计理由 |
|---|---|---|
| typed clock | `monotonic_now`、`realtime_now`、`Duration`、time point | C++ 正常代码避免把墙钟、duration、deadline 混成整数 |
| compact clock | `monotonic_ns`、`realtime_ns` | 热路径和系统边界使用一个明确的 ns 表示 |
| opt-in clock | `profiling_clock_ns`、有序 measurement pair、`coarse_realtime_ns` + availability | 低成本采样与精确区间边界分开；绝不偷偷改变默认钟 |
| Unix interchange | signed `UnixTimestamp{seconds,nanoseconds}` + checked OS/chrono conversion | 文件交换保留 pre-epoch、亚秒精度和宽范围 |
| arithmetic | 饱和 duration 转换、`deadline_after`、`remaining` | 消除 `now + timeout` 溢出和向下舍入导致的提前超时 |
| blocking sleep | 相对/绝对、整数/chrono | OS 线程休眠；fiber 必须使用 `sched`，不能阻塞 worker |
| elapsed | `Instant` | oneTBB `tick_count` / `std::chrono` 形状；差值有符号，便利方法显式 clamp |
| arbitrary timers | `TimerEntry` + `TimerQueue` | 大量 arm/cancel、混合 deadline、无分配、精确 ns |
| one-duration timers | `CommonTimeoutQueue` | 相同 duration 天然按 deadline 递增，用侵入链省去无意义排序 |

日历、时区、格式化与解析交给 C++ 标准库/专用库；支持工具链的 chrono parse 能力并不一致，
协议入口不得假定标准库均可解析。周期 timer 的 missed-tick、catch-up、
drift 与取消策略属于 executor/event loop；底层只提供一次性节点和安全 re-arm，不能替用户
猜策略。

## 时钟合同

`monotonic` 只用于 elapsed/deadline，零点不公开；`realtime` 是 Unix epoch 墙钟，只用于
日志、协议和文件时间。两者不能相减或互换。

默认 C++ 钟 header-inline 到 `std::chrono`，避免无意义的 DLL 跳转。
`profiling_clock` 在 x86 invariant TSC 和 GCC/Clang ARM64 architected virtual counter
(`CNTVCT_EL0/CNTFRQ_EL0`) 上启用。x86 可能受 suspend/VM migration/跨核偏差影响，因此名称
不承诺 deadline 所需的严格 monotonic；任何平台都禁止把它接入 timeout。普通
`profiling_clock_ns` 是最低成本、未序列化的采样入口；精确测量代码区间使用
`profiling_measurement_start_ns/stop_ns`，x86 采用 capability-gated
LFENCE/RDTSC/RDTSCP，ARM64 采用 ISB 包围 counter read，阻止编译器/CPU 把区域指令移过边界。
该 pair 不等待先前 store 全局可见，不能冒充持久化/设备 I/O fence。

`coarse_realtime` 仅在系统存在真正 coarse 源时报告 available；否则透明回退。availability
是策略输入，不是正确性前置条件。x86 TSC 频率优先读 CPUID 0x15；不可用时用倍增 sleep
window 对 OS monotonic 实际 elapsed 校准，不靠 busy-spin 保持 core 活跃。校准只影响首次
调用，热读仍是 rdtsc + fixed-point scale。ARM64 直接读取系统登记的 counter frequency；
Apple Silicon 当前为 1GHz，tick 已是 ns，不需要缩放。
CPUID.15 缺 crystal 时，CPUID.16 的 base MHz fallback 以 checked wide multiply/divide
派生，并拒绝 0 或超出 32-bit crystal 字段的结果；异常 hypervisor tuple、非有限频率以及
落在 `[1, UINT64_MAX]` 之外的 32.32 multiplier 都 fail closed 到 OS monotonic。

相对 timeout 转绝对 deadline 必须饱和并向上取整。integral duration 使用精确整数 ratio
算术，不能经浮点而在 2^53 以上丢失 ns；floating duration 单独处理 NaN、无穷和 ceil。
`Instant::elapsed/until` 在比较后以 checked subtraction 形成非负 duration：反向 clamp 0，
跨 `MonotonicTimePoint::min/max` 的正向不可表示差值饱和 `Duration::max`。raw `Instant`
subtraction 刻意保持 `std::chrono` 的有符号语义与 representability 前置条件。
绝对 sleep 在 Windows 使用高分辨率
waitable timer，在 Linux 使用绝对 `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`；
`EINTR` 才重试。零时长是 no-op，yield 属于调度 API。

## TimerQueue

`TimerQueue` 采用 Seastar `timer_set` 同类的 65 桶 radix 结构。前 64 桶按
`countl_zero(deadline ^ origin)` 分类，最后一桶保存已到期 deadline。

- arm：O(1)，一次 bucket 选择和侵入链尾插；
- cancel：O(1)，从可恢复 bucket 直接摘链；
- advance：O(触碰的 bucket + 交付/重分桶节点)；
- next：O(1)；取消当前最小项后允许保守地早醒一次，避免 cancel 热路扫描。

它相对 indexed heap 消除了 cancel 的 O(log n)，相对固定 tick wheel 消除了量化误差和大
bucket-head 常驻空间。`advance(now, out)` 只把 due 节点拼到调用方 `TimerList`，不运行回调；
这样 sched/io 可以在自己的 completion spine 中决定唤醒、仲裁和重入。

## CommonTimeoutQueue

一个实例声明一个 duration。相同 duration 在单调钟上依次 arm 时，绝对 deadline 天然非
递减，所以 O(1) tail append、O(1) cancel、O(到期数) prefix delivery 已是最紧形状。这对应
libevent `common_timeout` 的实际需求。

显式 deadline 必须非递减。C++ `try_schedule` 会在违反顺序时拒绝，并在
reschedule 被拒时保留旧 arm。多种固定 duration 应由消费模块建立多个 queue 并路由；通用
层不猜 duration 数量、路由表和租户策略。当前生产 sched/io 使用任意 deadline，故仍用
`TimerQueue`；Common queue 是公开的特化能力，不能冒充已接入生产。

## 生命周期与并发不变量

```text
caller owns stable TimerEntry
        │ schedule
        ▼
exactly one queue owns the linked node
        ├─ cancel ─────────► unlinked caller-owned node
        └─ advance ────────► caller-owned TimerList
                                   │ drain/pop
                                   ▼
                             re-arm / callback / destroy
```

1. `TimerEntry` 不可 copy/move；从 arm 到摘链地址稳定。
2. 一个节点同时最多属于一个 queue。错误 owner 是合同 bug。
3. queue 析构必须为空；entry 析构必须既未调度也未留在输出 list。
4. fired/cancelled 输出必须 drain 后才能复用或销毁节点。
5. 两种 queue 都是 single-owner 数据结构。跨线程操作投递到 owner event loop，不在热路加锁。
6. 同一 radix bucket 保持 FIFO；跨 deadline/bucket 不承诺全序。需要全序的消费者对批次排序。
## C++23 交付与 Native FFI 边界

C++ 模块和同工具链用户直接使用 typed chrono 与侵入节点，不经过 ABI bridge。time 当前只交付
source/static/shared；BUILD-03 已删除早期 C-syntax timer/queue schema、bridge、动态导出和 fixture。
Native FFI 为 `N/A`，没有 C ABI layout、handle 或 callback 兼容承诺。未来只有已命名的
foreign-runtime timer 任务通过全局 [Native FFI 准入策略](native-ffi-policy.md) 后，才另行设计窄的
runtime 类型擦除边界。

C++ 的 `advance(now,out)` 保持零分配批量 splice：输出节点在 caller-owned `TimerList` 中时，
caller 必须先 pop/drain 再复用或销毁。

## 修改门禁

- clock：Windows/WSL/macOS 与同工作量 `std::chrono` 对照；opt-in 路必须单独报告
  availability；raw sampling 与 ordered measurement 不跨语义判胜负；
- timer：随机 oracle、完整 `uint64_t` 边界、wrong-owner、输出链生命周期、回调 re-arm；
- 性能：预分配相同节点；同时对 indexed heap 和 faithful Seastar radix；固定顺序与反序注册；
  另测 N=1/8/64、散布节点、增量 event-loop、真实读钟的 schedule_after；
- 交付：time-only source/static/shared installed C++23 consumer 都必须通过；Native FFI 保持 `N/A`，
  除非新的已命名任务先通过全局准入策略；
- 消费回归：`time`、`container`、`sched`、`io_timeout`，并按能力跑 ASan/UBSan/TSan；
- 任何性能结论只进入 performance 文档，架构与 guide 不复制易腐点值。
