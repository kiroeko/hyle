# Time：时钟域、转换与无分配 timer

Time 统一 monotonic deadline、wall timestamp、CPU accounting、时间转换与阻塞线程等待；timer
容器只维护 caller-owned 节点，不执行 callback。使用方法见 [Time 指南](../user/time.md)。

## 两种内部职责，一个产品

host-clock `time` 只依赖 Core；header-only `timer_queue` 依赖 Core、Time 与 Container intrusive。
`hyle::time` 聚合两者，普通用户不需要了解内部 target。仅用 timestamp 的 Platform/Obs 代码不因此
取得 timer-node 闭包，Sched/IO 则直接复用 timer 数据结构。

clock/deadline 语义归 Time；OS wait/resource 归 Platform；callback 执行、跨线程命令、取消仲裁与
periodic policy 归 Sched/IO。Time 没有第二个 event loop，也不拥有 executor、日历、时区或文本协议。
当前 public failure 形状用值、checked optional 或程序员合同，不给每次读钟增加 Status 分配/传播。

## Clock domain 必须保留

| Domain | 用途与保证 | 易误用处 |
|---|---|---|
| monotonic | elapsed/deadline；零点不公开 | 不能当墙钟或持久化 timestamp |
| realtime | Unix epoch 墙钟，允许校时跳变 | 不能计算 timeout 或执行耗时 |
| coarse monotonic/realtime | 与对应 precise clock 同 domain/epoch，允许多次读数相等 | 不能假装精细排序；availability 与精度是策略事实 |
| profiling | opt-in 局部低成本 counter，缺少可信路线时回退 monotonic | 不能用于 timeout、跨机或设备同步 |
| thread/process CPU | 主体消耗的 CPU time，等待时基本停止 | 不能代替 wall elapsed 或计费其他线程 |

typed monotonic/realtime 直接使用标准 chrono domain，compact entry 给出明确 ns 表示。coarse
source 不存在时回退对应 precise clock，availability 不决定正确性。ThreadCpuClock 与 ProcessCpuClock
是独立 type，防止与墙钟混算；无法履行 accounting 合同时不会伪装成 wall clock。

profiling raw sample 不序列化周围指令；measurement start/stop 保留对应有序边界，仍不等于设备 I/O、
持久化或跨核同步 fence。硬件 counter 的可用性和校准是可选冷路径，不能污染默认时钟热读。

## 可注入时钟为何不统一成虚表

确定性 scheduler/timeout/retry 测试可保存满足 MonotonicClockSource 的具体 provider，模板路线允许
完全内联。只有必须在运行时选择 real/manual 时才使用两字借用 MonotonicClockRef 并支付间接调用。
borrow ref 不延长 source 寿命，拒绝 temporary source。

ManualClock 是 noncopyable、forward-only、饱和的线程安全值。relaxed atomic 只发布时刻，不同步用户
payload；advance 不拥有 waiter，也不自动 poll queue 或唤醒线程。这些仍归实际 runtime。

## Duration、deadline 与 sleep

相对 timeout 转 ns 向上取整并饱和，避免缩短调用方请求。标准整数单位走常数乘除，任意 integral
ratio 保持精确整数语义，不在 2^53 后经浮点丢精度。浮点 duration 的 NaN/非正值规范为 0，正无穷或
超范围饱和。

deadline_after 使用饱和加法，remaining 对过去返回 0。Instant 的 elapsed/until 是非负便利面，反向
clamp 0、过大正差饱和；直接 chrono subtraction 仍保持自己的有符号语义。

blocking sleep 的绝对域是 monotonic，0/past 立即返回。Linux absolute clock_nanosleep 对 EINTR
续等，Windows 使用对应 timer；它们阻塞的是 OS thread。fiber/coroutine 必须使用 Sched 的等待，不能
用 sleep 阻塞 worker 后声称支持协作调度。

## Unix timestamp 与 native interop

`UnixTimestamp{seconds, nanoseconds}` 是 canonical wide value：seconds 有符号，nanoseconds 在
`[0,1e9)`，所以 Unix -1ns 是 `{-1,999999999}`。它比单个 signed ns 的范围宽；提取 signed s/ms/us/ns
可能溢出，返回 optional，不截断。absolute 向较低精度转换向负无穷取整，-1ns 不能错误变成 epoch 0。

system_clock、time_t、POSIX timespec/timeval、FILETIME 与 signed Unix units 使用 checked 转换。
native 结构放在 native_interop 头，避免基本 clock/value 头强迫用户 include OS headers。值布局是 C++
合同，不是自动冻结的 wire/FFI 格式；序列化由所属协议决定。

## TimerQueue 为何使用 radix buckets

真实 event-loop 常有大量 arm、绝大多数在到期前 cancel。indexed heap 每次 insertion/cancel 都要维护
O(log n) 顺序；固定 tick wheel 引入精度与额外 bucket footprint。TimerQueue 使用 intrusive radix
buckets，把排序工作延迟到 expiry，不分配 node，也不量化 deadline。

| 操作 | 合同 |
|---|---|
| schedule | O(1) bucket 选择；过去 deadline 在下次 advance 交付，不现场调用用户代码 |
| cancel | O(1) 摘链；idle no-op，错误 owner 是程序合同违反 |
| advance | 与触及 buckets、交付/重分配节点数有关 |
| next | O(1)；取消最早 timer 后可保守地提前唤醒 |

advance 将 due nodes 放到 caller-owned delivery list，不执行 callback，也不保证不同 radix buckets
之间的严格总 deadline 顺序。需要排序时由 caller 显式对交付批次排序。保守 early wake 不等于 timer
提前触发，外层仍检查实际 due 条件。

TimerEntry 地址需稳定，不能在 queue 或 delivery list 仍持有它时析构/移动。queue 单 owner，跨线程
arm/cancel 通过真正拥有 event loop 的命令队列完成，不为本地路径内置 lock。节点 delivery 与 callback
完成是不同生命周期。

## CommonTimeoutQueue 为何单独存在

同一 duration 加 monotonic arm time 自然产生不降的 deadline，队列无需比较或分桶：tail append、
head expire、known-node cancel、next 均 O(1)。它适合大量连接共享 idle timeout 的情形，复用 intrusive
safe list，不增加通用 timer 的 metadata。

一个 queue 只表示一个 duration；多个 duration 的路由属于 consumer。显式 deadline 入口要求非递减，
try_schedule 在违反顺序时拒绝并保留先前 arm。混合/无序 deadline 使用 TimerQueue，不能静默破坏
排序不变量。两者都不内建 repeat、skip/catch-up、周期漂移或 callback policy。

## 维护重点

时钟 domain、overflow/rounding、pre-epoch、伪唤醒、owner 和 node 失效比几个 ns 的点值更重要。
clock/sleep/timer workload 不混成一个总分，coarse、profiling 与 precise 路线不能按不同精度比较。
接口与复杂度见[公开头](../../include/hyle/time/)；跨模块边界见[运行时架构](runtime-design.md)。
