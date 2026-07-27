# time 模块收口任务（2026-07-25）

> 本页是 milestone/Jira 式任务记录，只说明改了什么、如何验收。当前整体设计见
> [`time-design`](../architecture/time-design.md)，用户用法见 [`guide/time`](../guide/time.md) /
> [`guide/c-time`](../guide/c-time.md)，全部性能数字只见
> [`benchmarks-time`](../performance/benchmarks-time.md)。

## 目标与完成定义

- [x] 按真实基础库用户形状复审能力，不按已有 benchmark 反推 API；
- [x] C++ typed API 与紧凑热路径并存，C ABI 覆盖全部语言无关能力；
- [x] arbitrary timer 达到成熟库 SOTA 算法与 O(1) arm/cancel；
- [x] sched/io 消费统一，旧 API 和 open-coded deadline 清零；
- [x] 2026-07-26 最终形状的 Windows、WSL、Apple 真机正确性/性能，以及 WSL
  GCC14 ASan/UBSan；Release 五门和 sanitizer 五门均通过；
- [x] architecture / guide / performance / milestone 文档各归其位，真实 LOSE 与边界不冒充绿色。

## 本轮变更

1. 用 65 桶 exact-ns radix `TimerQueue` 替换 8×64、1ms tick 的固定 wheel。形状直接参考
   Seastar `timer_set`：取消 O(1)，排序成本延迟到交付，常驻 bucket heads 约 1 KiB。
2. 删除只为两个实现做静态同形包装的 `TimerSource` concept。`TimerQueue` 与
   `CommonTimeoutQueue` 的真实前置条件不同，不再伪装可互换。
3. `TimerEntry` 保留 stable-address/data/callback/exact-owner；source 只交付 list，不执行用户
   代码。补 `cancel_all`、chrono overload、typed next deadline 和完整生命周期门。
4. clock 补 typed chrono facade、signed `Instant` subtraction、absolute sleep；Windows 使用
   high-resolution waitable timer，Linux 使用 absolute monotonic sleep。
5. C ABI 从未经产品审查的旧 wheel 面重做为 42 个动态符号 + 4 个 header-inline convenience：
   opaque timer/general/common queue、Status、同步 visitor、callback re-arm、完整 shutdown。
6. sched 的 fiber sleep/timed I/O 迁到 per-worker `TimerQueue`；deadline 统一走 time 饱和算术。
7. benchmark 新增 faithful Seastar radix、真实 indexed eager heap、checked/naked TAILQ、
   两组 order-swap 与 Apple QoS 报告；纠正早期缩小 rival 节点/链形的失真。
8. 2026-07-26 strict 重审把 integral duration 转换改为 exact integer ratio，两个 opt-in clock
   改为 `profiling_clock` / `coarse_realtime` 语义名，TSC fallback 改为 sleep-based calibration。
9. 正常使用性能面新增 N=1/8/64、散布节点、incremental event loop、真实 schedule_after、
   C ABI lifecycle/Status 与 warm sleep overshoot；没有为 benchmark 增加产品 metadata。
10. C caller-storage timer 参考 libevent `event_get_struct_event_size + event_assign` 落地：runtime
    size/alignment + init/deinit 允许一个 arena/slab 承载海量 opaque controls，不增加 TimerEntry
    或 wrapper 大小、不引入 time→mem 依赖。single-timer fast slot 因伤害默认 N=8/4096 workload
    被实验后撤回。
11. platform/io 消费回审补 canonical signed Unix `{seconds,nanoseconds}` 与 chrono/time_t/
    timespec/FILETIME checked conversion，修复 pre-epoch wrap 和 platform 丢亚秒精度；C current
    timestamp 入口令同一值形状可由 C/FFI 直接取得。

提交：

- `ffff65f time: replace fixed wheel with radix timer queue`
- `2c1bab2 time: tighten intrusive timer hot paths`
- `14ee67e time: certify timer queues against faithful peers`
- `52595a6 time: make integral duration conversion exact`
- `5af0fe9 time: name opt-in clocks by semantics`
- `0423671 time: calibrate profiling clock without busy spin`
- `bd29d57 bench(time): cover normal timer usage shapes`
- `9129001 time: complete and inline integral duration facade`
- `ef8e066 bench(time): audit Apple normal-use QoS`
- `ba61eb5 bench(time): stabilize Apple timer epochs`
- `16c91b4 time: compile calibration gate only on TSC targets`

## 功能审计结论

公开面覆盖 clock、deadline arithmetic、thread sleep、elapsed snapshot、任意 deadline timer、
单一 duration timer、C/FFI。成熟库对照包括 oneTBB `tick_count`、Seastar `timer_set`、
libevent common timeout、folly TimeoutQueue/HHWheelTimer。

明确不加入底层的能力：calendar/timezone/format parse（标准库/专用库）、executor callback
调度（sched/io）、隐式 repeating policy、内部加锁的 cross-thread queue。它们不是“缺功能”，
而是避免底层类型替消费场景猜策略。

全库生产审计只发现两个正确消费者：sched 和 io。它们都直接走 C++ time API；HYLE 内部没有
绕 C ABI。`sched::BlockingPool` 的 condition-variable keepalive 与 `platform::touch` 的文件墙钟
属于各自 OS 资源语义，不应制造反向 time 依赖。

## 正确性与 ABI 验收

- C++ time：25 clock + 23 TimerQueue + 16 CommonTimeoutQueue + 1 header test；
- 两组 million-operation randomized oracle、完整 uint64 deadline、wrong-owner、death/lifecycle；
- 纯 C consumer 覆盖全部 handle、跨 source Status、顺序拒绝、visitor re-arm、NULL 默认；
- 当前 42/42 动态符号 surface manifest 在 Windows 通过，C11/C++23 自包含头通过；新增
  timestamp/API 与 platform/io 消费边已在 Windows、WSL GCC14、Apple M4 最终复证；
- Windows Release、WSL GCC14 native-ext4、AppleClang 21 最终 time 五门均通过；
- 最终 `16c91b4` 上 WSL GCC14 ASan 五门 5/5、UBSan 五门 5/5；time-only 构建不需要
  mimalloc ODR workaround，分别使用 `detect_leaks=1:halt_on_error=1` 与
  `halt_on_error=1:print_stacktrace=1` 的正常严格选项。
- 2026-07-27 timestamp 增量的 WSL ASan/UBSan time+platform 各 13/13，三平台 io file/stat
  消费组各 3/3；pre-epoch、亚秒精度与跨 OS conversion 均在最终形状通过。

Windows scheduler 全组曾有一次既有 `Blocking.RetiredThreadHandlesStayBoundedAcrossBursts`
瞬态失败，隔离重跑 10/10 通过；与 timer 迁移无稳定关联。

## 性能验收结论

详细表不在 milestone 复制。摘要：

- 上轮默认 clocks 三平台与同工作量 `std::chrono` 同档；x86 opt-in paths 有收益，M4 正确回退；
- 最终形状 Windows 已证：对同算法 radix 在 N=1/8/64 与 N=4096 均胜；对 indexed heap 只有
  N=1 输，从 N=8 起胜；连续/散布 incremental event-loop 统计平；
- Windows C ABI、真实读钟 schedule_after 和 sleep overshoot 已单列合同税/OS 行为；
- caller-storage init/deinit 对默认 create/destroy n=15 为 Windows 2.925×、WSL 4.532× WIN，hot
  queue Status 合同不变；
- WSL 最终形状复现 normal churn TIE / reverse churn LOSE 边界；M4 N=8 arm+cancel 对
  Seastar 慢 7–12%，但大 N 两档 QoS 均胜；真实 typed/raw schedule_after 三平台 TIE，显式
  预采样 typed 在 GCC/M4 有 signed clamp 税。三平台性能证据闭环，不把真实 LOSE 改绿。

因此“性能收口”含义是：选用成熟的 SOTA 算法、复杂度不落后、正常跨平台证据完整，并诚实
保留编译器/合同差异；不等于要求每个 microbenchmark 在每个注册顺序都必须数值第一。

## 最终边界

- queue single-owner；跨线程投递由 event loop/scheduler 负责；
- callback 不在 `advance` 内执行；输出 list 必须 drain；
- Common 只接受单一 duration 形成的非递减 deadline，且尚无生产 consumer；
- `next_deadline` 在取消 minimum 后可能保守早醒一次，不会晚醒/漏发；
- 发布前允许继续改 C ABI，但任何变化必须同步 header、实现、manifest、纯 C test 与 guide。
