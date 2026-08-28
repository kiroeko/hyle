# time 模块架构

`time` 统一 HYLE 内部和 C++ 用户的时钟域、deadline 算术、Unix 时间互操作、阻塞式线程等待，
以及 event loop 所需的无分配 timer 容器。它不拥有 executor、线程间路由、周期任务策略、日历、
时区或文本协议。

施工事实见 [`time-v1 closure`](../implementation/tasks/time/time-v1-closure.md)，当前工程性能证据见
[`benchmarks-time`](../implementation/evidence/performance/benchmarks-time.md)，用户选型见
[`guide/time`](../user/time.md)。本页只保存稳定设计与决策。

## 边界与依赖

`time` 的直接公开依赖为 `core` 和 header-only `container_intrusive`：

```text
core + container_intrusive ──► time ──► platform（文件时间值）
                                  ├──► sched（deadline 与 TimerQueue）
                                  └──► io（Unix 时间与 timeout 仲裁）
```

- `TimerEntry` 的 private safe hook 和公开 `TimerList` 使 `container_intrusive` 成为真实 `PUBLIC`
  target edge；不能依赖安装树偶然包含头文件。
- `platform` 拥有 OS wait/resource mechanism，`time` 拥有 clock/deadline 语义。`platform::FileInfo`
  公开 `UnixTimestamp`，所以 `platform -> PUBLIC time` 是合法的值类型边。
- `sched`/`io` 拥有 callback 执行、取消仲裁、跨线程命令和 virtual/periodic policy。time 容器只返回
  caller-owned 节点，不造第二个 event loop。
- time 不依赖 `error`。公开操作是无失败值读取、checked value conversion，或程序员合同；前两者分别
  用值/`std::optional`，合同错误 fail-fast。给时钟热读强加 `Status` 会改变错误模型和成本。

模块只提供 C++23 source/static/shared 三种交付。Native FFI 当前为 `N/A`，没有 C layout、opaque
handle、callback ABI 或稳定 C++ ABI 承诺；未来只有具名 foreign-runtime consumer 通过
[`Native FFI policy`](native-ffi-policy.md) 后，才设计窄的 runtime 类型擦除边界。

## 能力族

| 能力族 | 公开形状 | 正常用途 |
|---|---|---|
| precise monotonic/realtime | typed chrono 与 compact ns | elapsed/deadline；墙钟 timestamp |
| coarse monotonic/realtime | typed/compact、独立 availability | log/rate limit/event-loop 可接受粗分辨率的读钟 |
| profiling counter | raw sample、有序 start/stop、availability | 局部热区诊断，不参与 timeout |
| CPU accounting | `ThreadCpuClock`、`ProcessCpuClock` 与 compact ns | 区分执行时间和等待时间 |
| duration/deadline | 精确向上取整、饱和 add、remaining、`Instant` | 安全 timeout 和 elapsed |
| Unix interchange | canonical signed seconds/nanoseconds + checked conversion | 文件、OS、持久化与协议边界 |
| blocking sleep | relative/absolute、raw/chrono | 不在 scheduler 上的 OS 线程等待 |
| arbitrary timers | `TimerEntry`、`TimerList`、`TimerQueue` | 混合 deadline、大量 cancel 的 event loop |
| one-duration timers | `CommonTimeoutQueue` | 海量相同 duration timeout |

日历、时区、locale、RFC 文本解析/格式化由 C++ 标准库或具名协议 owner 承担。周期 timer 的 skip、
catch-up、bounded catch-up、fixed-delay 和 drift 也由 executor API 明示，不能藏在一个含糊的
`repeat=true` 中。

## 时钟域合同

| Domain | 保证 | 禁止 |
|---|---|---|
| monotonic | 稳定递增，零点不公开 | 当墙钟或持久化值 |
| realtime | Unix epoch 墙钟，可能因校时跳变 | elapsed、deadline、timeout |
| coarse monotonic | 与 precise monotonic 同 domain/epoch；可能多个 tick 不变 | 需要精细排序的 deadline |
| coarse realtime | 与 realtime 同 wall domain；可能多个 tick 不变 | elapsed 或精确事件排序 |
| profiling | opt-in 低成本局部 counter；不可用时回退 monotonic | timeout、跨机时间、resume 后连续性 |
| thread/process CPU | 主体实际消耗的 CPU 时间，等待时基本停止 | wall elapsed、deadline、外部线程计费 |

默认 typed monotonic/realtime 直接 header-inline 到 `std::chrono`，compact 入口只是同一 domain 的
明确 ns 表示。coarse availability 是策略信息，不是正确性条件：没有原生 coarse source 时保持 domain
语义并回退 precise clock。

profiling 在 x86 invariant TSC 和支持的 ARM64 architected counter 上启用。raw 采样不序列化周围
指令；有序 pair 阻止编译器/CPU 把普通指令或 load 穿过测量边界，但不代替设备 I/O、持久化或跨核
同步 fence。x86 首次校准优先用 CPUID 精确频率，否则用 sleep-based OS-clock sampling；失败关闭到
monotonic，不 busy-spin，也不给默认 clock 添加状态。

`ThreadCpuClock` 与 `ProcessCpuClock` 是独立 chrono clock type，避免与 wall/monotonic time point
混算。实现使用当前线程或进程的原生 accounting clock；平台合同不能成立时 fail-fast，绝不伪装成
wall elapsed。

## Duration、deadline 与 sleep

相对 timeout 转 ns 必须向上取整并饱和，避免请求时间被缩短。标准 whole-nanosecond units 走常数
乘除；任意 integral ratio 使用精确整数算术，不能在 `2^53` 后经浮点丢精度。浮点 duration 将
NaN/非正值规范为 0，将正无穷/超范围值饱和。

`deadline_after*` 用饱和加法，`remaining*` 对 past 值返回 0。`Instant::elapsed/until` 是非负便利
入口，反向 clamp 0、不可表示正差饱和；直接 time-point subtraction 保留 chrono 的有符号合同。

blocking sleep 的绝对域始终是 monotonic。Windows 使用高分辨率 waitable timer，Linux 使用绝对
`clock_nanosleep` 并只对 `EINTR` 重试；0/past 立即返回。fiber/coroutine 必须路由 `sched`，不能阻塞
worker。

## Unix 时间互操作

`UnixTimestamp{seconds, nanoseconds}` 是 canonical value：`seconds` 有符号，`nanoseconds` 必须在
`[0, 1e9)`，因此能表示 pre-epoch 值。它具有自然默认值和 comparison，并与 system-clock
time point、`time_t`、POSIX `timespec`、Windows FILETIME ticks 双向 checked 转换。

FILETIME 只有 100 ns 分辨率。timestamp 转 FILETIME 对纳秒部分向下取整到最近的 100 ns；这保留
“不晚于原 timestamp”的明确策略。100 ns 对齐值精确 round-trip；无效或超范围输入返回
`std::nullopt`，不能 wrap。`realtime_ns()` 只适合当前非负 epoch 的 compact 热路，不是通用存储格式。

## TimerQueue

`TimerQueue` 是与 Seastar `timer_set` 同类的 65-bucket radix queue。前 64 桶按
`countl_zero(deadline ^ origin)` 分类，最后一桶保存已到期项：

- arm/cancel/next：O(1)，正常路径零分配；
- advance：O(触碰桶 + fired/rebucketed 节点)；
- 当前 minimum 被 cancel 后允许一次保守 early wake，以保留 cancel O(1)；
- 同一 bucket FIFO，跨 bucket/deadline 不承诺全序。

相对 indexed heap，它为高取消率负载去掉 O(log n) cancel；相对固定 tick wheel，它没有量化误差和
大 bucket-head 常驻空间。`advance(now, out)` 只把 due 节点 splice 到 `TimerList`，不执行 callback，
从而让 scheduler/IO completion spine 保持唯一的执行与重入 owner。

## CommonTimeoutQueue

一个实例声明一个 duration。相同 duration 在 monotonic clock 上依次 arm 时，deadline 天然非递减，
因此 tail append、cancel、next 均为 O(1)，advance 为 O(fired)。显式 deadline 的
`try_schedule` 在顺序不成立时返回 false；reschedule 被拒时保留旧 arm。

多种固定 duration 使用多个 queue 加消费方路由；混合/乱序 deadline 使用 `TimerQueue`。该类型是参考
libevent common-timeout 形成的公开通用能力，目前没有 HYLE 产品消费者，因此不进入 scheduler 的
P1 性能聚合。

## 所有权、析构与并发

```text
caller-owned stable TimerEntry
       │ schedule
       ▼
one queue owns linked membership
       ├─ cancel ───────► idle caller-owned entry
       └─ advance ──────► caller-owned TimerList ── drain/pop ─► re-arm/callback/destroy
```

1. `TimerEntry` 不可 copy/move；linked 期间地址稳定，同时最多属于一个 queue。
2. 两种 queue 都是 single-owner；跨线程操作必须投递到 owner event loop，热路不加锁。
3. queue-first teardown 是正常操作：析构 O(n) detach 所有仍 armed 的 entry，不执行 callback；entry
   随后为 idle，可复用或销毁。
4. entry-first teardown 仍是程序员错误：不能让 armed entry 或仍在 delivery list 的 entry 先析构。
5. 需要取得取消节点时显式 `cancel_all(out)` 并 drain；只想放弃整批时可让 queue 析构自动 detach。
6. wrong-owner、monotonic advance 倒退和未 drain 的 re-arm/destroy 都 fail-fast。

## 性能与变更门禁

- API 先由具名用户旅程和成熟实现确定；benchmark 不得发明产品类型、节点 metadata 或特殊入口。
- scheduler timer 产品门使用 stable scattered nodes、真实 arm/cancel/advance/drain、约 70% cancel 的
  deterministic replay；faithful Seastar peer 与 HYLE 保持相同 ownership 和计费边界。
- 工程性能裁决固定 W1-WIN、5% non-inferiority、独立进程、双执行/分配顺序和 primary/placement；
  A/A、5%/10% known delta、timer floor、artifact/build/source identity 全部是前置门。
- clock、sleep、Common queue 和极端 cardinality 可作 mechanism diagnostic，不能混成“time 总分”。
- 任何算法、公开 API、layout、直接依赖、source/static/shared delivery 或首席 consumer 改变都重开；
  单次点值、亚纳秒摆动或某一地址姿势翻转本身不能改变产品形状。
