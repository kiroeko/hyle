# time-02：分层时间轮（hierarchical timing wheel）

| 字段 | 值 |
|---|---|
| **阶段** | P2 |
| **模块** | time/ |
| **前置** | time-01（`monotonic_ns` 单调钟）、container（侵入式链表 hook）、platform-01（`wait_on_address` per-wait timeout） |
| **状态** | ✅ 完成。6 层 × 64 槽分层时间轮（tokio 移植），O(1) 摊还 add / O(1) 摘链 cancel / 不空转到期。platform 加 `park_timeout` 执行底座。双平台 15 功能测试绿 + platform 9 测试（含 4 timed）+ WSL TSan 净。**解锁 sched-04b**。 |

> **范围**：海量短超时的定时器数据结构。`schedule(deadline, entry)` O(1) 摊还、`cancel` O(1)、`advance()` 批量到期；到期动作 = **把 fiber 塞回 run-queue**（与 Waker completion 脊同一唤醒路径，非执行回调）。基准用 `monotonic_ns`（deadline 绝不用墙钟）。
> **交付**：`include/hyle/time/timer_wheel.h`（167 LOC 头，纯数据结构不读时钟——由调用方喂 `now`，便于测试）+ `platform/wait.h` 的 `wait_on_address_for`/`Parker::park_timeout`（90 LOC）+ `tests/time/timer_wheel_test.cc`（213 LOC，15 用例 + 百万级 oracle 对拍）+ `bench/time/timer_wheel_bench.cc`（对 min-heap）。
> **为什么单列 time/ 而非塞进 sched**：时间轮有独立设计分量，io/ 的 timeout 也直接用它；但它是 sched 的 T2 仆从，不反向依赖 sched（sched-design §2.1）。

## 为什么是它（北极星驱动 + 卡住 sched 后半段）

北极星"百万 IO-bound agent"里，每个 agent 频繁挂 timeout（等 LLM 响应、io 超时、fiber sleep）——**add/cancel 是热路径，且 add/cancel ≫ 真到期**（响应先到，timeout 在触发前就被取消）。这正是需要一个**中央定时器结构统一管理海量到期**的场景：不可能给百万 fiber 每个占一个 OS 线程去 park+timeout。

它卡着 sched 的整条定时支线：
```
time-02 时间轮（本里程碑）
   ↓ 前置
sched-04b（Waker × timer，deadline 到期 wake）= wait_for / wait_until
   ↓ 前置
io/compute 的 timeout / deadline 路径
```
sched 的**无限等待**部分（01/02/03/04a/05）已完整；卡住的只有**定时**这条支线。

## 选型：分层时间轮是性能正解（SOTA 调研，非教科书泛论）

作为性能 SOTA 库，选型对标真实产品级实现。调研结论**单向指向分层时间轮**：

| 运行时 | 结构 | 关键参数 |
|---|---|---|
| **tokio** | 分层 hashed timing wheel | 6 层 × 64 槽，L0=1ms/槽，不空转（算精确下次唤醒） |
| **Linux 内核** timer | 分层 timer wheel | 9 层 × 64 桶；精确定时另走 hrtimer 红黑树 |
| **folly** HHWheelTimer | hashed hierarchical wheel | tick 10ms，256 槽 × 4 层，不空转 |
| nginx | 红黑树 | 连接数中等、需精确超时（与我们负载不同） |
| Go / libev / libuv | 4 叉最小堆 | 定时器少 + 需精确到期 + 取消少（与我们相反） |

**时间轮 vs 堆，针对"add/cancel ≫ expire"的判断**：

| 操作 | 分层时间轮 | 最小堆(4叉) |
|---|---|---|
| add | **O(1)** 摊还 | O(log n) |
| cancel | **O(1)**（摘侵入式链表节点） | O(log n)，或惰性标记 → 堆膨胀 |
| expire-tick | O(到期数) + 级联降级 | O(k log n) |

关键：agent 负载 = add/cancel 极多、真到期极少——这恰是 **Linux 内核 timer 的负载画像**，也是时间轮压倒堆的分水岭。堆的 cancel 若走惰性标记，取消不真删 → 百万 add-then-cancel 下堆塞满死节点、内存涨、expire 反复弹死节点（硬伤）。堆唯一优势"O(1) 拿确切最近到期"对我们不重要（单调钟批量 tick，不需精确 min）。

## 与 futex timeout 的关系：协同，不是二选一

澄清一个曾经的表述混淆：**时间轮和 `wait_on_address`/`Parker` 的 per-wait timeout 不是竞争方案，是分工**——
- **中央时间轮决定"何时醒"**：算出最近非空槽的绝对纳秒。
- **futex per-wait timeout 是执行层**：把那个 deadline 喂给底层 `wait_on_address`/`WaitOnAddress`，OS 只 park 一个 worker 线程到该时刻。

底层 futex/WaitOnAddress 本就支持 timeout（`FUTEX_WAIT` 有 timespec、`WaitOnAddress` 有 dwMilliseconds）。本里程碑已给 `platform/wait.h` 加了执行底座：`detail::wait_on_address_for(addr, expected, timeout_ns)`（相对 timeout：Linux `FUTEX_WAIT` 相对 timespec / Windows `WaitOnAddress` dwMilliseconds 向上取整 / macOS `os_sync_*_with_timeout`）+ `Parker::park_timeout(timeout_ns)`（单发，语义仿 Rust std `park_timeout`：不内部循环剩余时间，调用方重查条件 + 用新 deadline 重 park，返回是否拿到 token）。现代补充：futex per-wait timeout 就是最好的底层机制，**无需 timerfd**（那是 epoll 世界的产物，对 futex 模型是累赘）。

## 参数（tokio + folly 合流，非内核空转版）——已实现

- **层级/槽：6 层 × 64 槽，L0 = 1ms/槽**（tokio 参数）。1ms 精度对"等 LLM/io 超时"绰绰有余；6 层覆盖 2^36 tick ≈ 2.18 年，超过任何真实 fiber 超时。
- **不空转（deadline-driven）**：`next_deadline_ns()` 用每层 `occupied` u64 位图 + `countr_zero`（`std::rotr` 后数尾零）算最近非空槽绝对纳秒——不逐 tick 空转。这个纳秒喂给 `Parker::park_timeout`（本里程碑加的 platform 底座），OS 只 park 一个 worker 到该时刻。
- **cancel O(1)**：timer 节点内嵌 `IntrusiveListHook`，取消 = 摘链 + 从存储的 `deadline_tick` 反算 (level, slot)，**绝不用惰性标记**（堆膨胀陷阱）。
- **单调钟兼容**：轮基准 = 构造时传入的 `start_ns`（单调纳秒），级联降级（cascade）时按新 `elapsed_` 重算槽位。schedule 向上取整到 tick、advance 向下取整——保证"至少等到 deadline，绝不早触发"。
- **分片是 sched 集成决策，不在轮内**：本里程碑交付的是**单个轮数据结构**（不含锁、不跨核）。"per-worker 分片 + 跨核取消走 MPSC 命令队列"是 **sched-04b 如何使用**这个轮的问题（每 worker 持一个轮），非轮本身的分量——下面的分歧点①因此下沉到 sched-04b 实测。

## bench 实测（对 min-heap，双平台 N=15 统计，scripts/bench-stat.py）

SOTA 对手 = min-heap（Go/libev/libuv 的定时器选择），**同职算法**对照。folly HHWheelTimer 焊死在 EventBase/libevent 事件循环 + glog + boost 上，链进 microbench 测的是事件循环不是轮，故对标算法本身（设计文档争论的正是"轮 vs 堆"）。堆按其真实行为建模：无 O(1) keyed 删除 → **惰性 cancel（墓碑标记）**，expire 时跳墓碑——这正是文档预言的"add-then-cancel churn 下堆膨胀"。

| 工作负载 | Windows MSVC | Linux GCC14 | 判定 |
|---|---|---|---|
| **churn（arm N + 取消 70% + expire，真实 agent 画像）** | 轮 **赢 2.09x** | 轮 **赢 3.27x** | ✅ 双平台铁赢 |
| arm+expire（全触发） | 统计平局(1.09) | 轮 **赢 2.25x** | 平台依赖 |
| arm+cancel（纯取消，无 expire） | 统计平局(1.00) | 轮 **输 0.53x** | 平台依赖 |

**关键诚实判断（对齐 mem 红队方法论）**：`arm+cancel` 纯取消微基准 Linux 上轮**输** 2x——但这是**微基准假象**：每 op 丢弃一个全新堆，堆的墓碑 `vector<bool>` 位标记比轮的摘链便宜，**但这个 bench 在墓碑必须被走之前就把膨胀的堆扔了**。真实负载（churn，墓碑要活着穿过 expire）双平台轮都赢 2-3x。这与堆的 O(1) "拿最近到期"优势对我们不重要（单调钟批量 tick、不需精确 min）一致——文档"轮压倒堆"的论断在**真实负载**成立，纯取消微基准是不公平切片，如实记账不藏。

## 遗留到 sched-04b 的分歧点（需真实负载才能定）

1. **单中央轮 vs per-worker 分片**：分片省锁，但跨核取消要走命令队列（agent 常 A 核挂 timeout、B 核收响应取消）。轮数据结构已就绪，分片是 sched-04b 用几个轮 + 命令队列的问题——测真实"取消发起核 vs timer 归属核"分布再定。
2. **槽粒度 1ms vs 250µs**：若 fiber sleep 有亚毫秒需求，L0 变细但级联更频繁。当前 1ms（`kTickNs`）是编译期常量，改动一处。
3. **级联降级成本**：短超时集中在 L0 时几乎无级联；churn bench 已含 L0-L2 分布未见降级热点。若超时长尾，测降级是否成热点，必要时超长超时旁路。

## 交付物（已落地）

- `include/hyle/time/timer_wheel.h`（`TimerWheel` + `TimerEntry` 侵入式节点 + O(1) schedule/cancel + advance 批量到期 + next_deadline_ns 不空转；167 LOC）
- `platform/wait.h`/`.cc`：`detail::wait_on_address_for`（相对 timeout：Linux `FUTEX_WAIT` timespec / Windows `WaitOnAddress` dwMilliseconds / macOS `os_sync_*_with_timeout`）+ `Parker::park_timeout`（单发，语义仿 Rust std park_timeout）
- `tests/time/timer_wheel_test.cc`（15 用例：到期顺序、FIFO 同 tick、O(1) cancel 无泄漏、cascade 精确不早触发、cancel-after-cascade、reschedule、horizon clamp、百万 arm/cancel oracle 对拍）+ `tests/platform/wait_test.cc` 加 4 个 park_timeout 用例
- `bench/time/timer_wheel_bench.cc`（对 min-heap 三工作负载）

## 参考

- sched-design §2.1（time/ 最小接口：单调钟 + 时间轮 + 到期塞回 run-queue）、§3（Waker 脊，timer 是三源之一）。sched-04b（本里程碑的直接消费者）。
- [Hashed and Hierarchical Timing Wheels (SOSP'87)](https://www.cs.columbia.edu/~nahum/w6998/papers/sosp87-timing-wheels.pdf)、[tokio timer 设计](https://tokio.rs/blog/2018-03-timers)、[folly HHWheelTimer](https://github.com/facebook/folly/blob/main/folly/io/async/HHWheelTimer.h)、[Linux kernel/time/timer.c](https://codebrowser.dev/linux/linux/kernel/time/timer.c.html)。
- 反例（为何不选堆）：Go runtime per-P 4 叉堆、libev 4-ary heap——定时器少 + 需精确到期 + 取消少的场景，与百万 agent 负载相反。
