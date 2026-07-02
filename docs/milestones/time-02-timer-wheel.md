# time-02：分层时间轮（hierarchical timing wheel）

| 字段 | 值 |
|---|---|
| **阶段** | P2 |
| **模块** | time/ |
| **前置** | time-01（`monotonic_ns` 单调钟）、container（侵入式链表 hook）、platform-01（`wait_on_address` per-wait timeout） |
| **状态** | 📋 规划（选型已定 = 分层时间轮，见下 SOTA 调研；实现待排期。是解锁 sched-04b 定时唤醒 + io/compute timeout 的关键前置） |

> **范围**：海量短超时的定时器数据结构。`add(deadline, fiber)` O(1) 摊还、`cancel` O(1)、`tick()` 批量到期；到期动作 = **把 fiber 塞回 run-queue**（与 Waker completion 脊同一唤醒路径，非执行回调）。基准用 `monotonic_ns`（deadline 绝不用墙钟）。
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

底层 futex/WaitOnAddress 本就支持 timeout（`FUTEX_WAIT` 有 timespec、`WaitOnAddress` 有 dwMilliseconds），但当前 `platform/wait.h` 的 `wait_on_address`/`Parker::park()` **未暴露 timeout 参数**——本里程碑需给它们加 timeout 重载（platform 小改），作为时间轮的执行底座。现代补充：futex per-wait timeout 就是最好的底层机制，**无需 timerfd**（那是 epoll 世界的产物，对 futex 模型是累赘）。

## 参数起点（tokio + folly 合流，非内核空转版）

- **分片：per-worker（per-runqueue）一个轮**。百万 fiber 下避免跨核锁竞争是第一性能维；timer 与被唤醒 fiber 同核，wake 即本地入 run-queue，cache 亲和。跨核取消走 MPSC 命令队列投递，不加锁。
- **层级/槽：6 层 × 64 槽，L0 = 1ms/槽**。1ms 精度对"等 LLM/io 超时"绰绰有余；6 层覆盖 ~2 年。
- **不空转（deadline-driven）**：算最近非空槽绝对纳秒作 park deadline，喂 futex timeout——不逐 tick 空转。
- **cancel O(1)**：timer 节点内嵌侵入式双向链表指针，取消 = 摘链，**绝不用惰性标记**（堆膨胀陷阱）。
- **单调钟兼容**：轮基准 = 启动单调纳秒，级联降级时重算槽位。

## 待 bench 验证的分歧点（性能库不拍脑袋）

1. **头号必测：单中央轮 vs per-worker 分片**。分片省锁，但跨核取消要走命令队列（agent 常在 A 核挂 timeout、B 核收响应取消）。若跨核取消占比高，分片的队列开销可能吃掉省下的锁——**测真实的"取消发起核 vs timer 归属核"分布**再定。
2. **槽粒度 1ms vs 250µs**：若 fiber sleep 有亚毫秒需求，L0 变细但级联更频繁。
3. **级联降级成本**：短超时集中在 L0 时几乎无级联；若超时分布长尾，测降级是否成热点，必要时对超长超时退化到旁路红黑树（混合方案）。

## 交付物（规划）

- `include/hyle/time/timer_wheel.h`（分层时间轮 + 侵入式 timer 节点 + O(1) add/cancel + tick）
- `platform/wait.h` 扩展：`wait_on_address` / `Parker::park` 加 timeout 重载（时间轮的执行底座）
- `tests/time/timer_wheel_test.cc`（到期顺序、O(1) cancel 无泄漏、级联降级正确性、单调钟基准）
- benchmark：对标 tokio/folly 量级 + 上述三个分歧维度的 A/B

## 参考

- sched-design §2.1（time/ 最小接口：单调钟 + 时间轮 + 到期塞回 run-queue）、§3（Waker 脊，timer 是三源之一）。sched-04b（本里程碑的直接消费者）。
- [Hashed and Hierarchical Timing Wheels (SOSP'87)](https://www.cs.columbia.edu/~nahum/w6998/papers/sosp87-timing-wheels.pdf)、[tokio timer 设计](https://tokio.rs/blog/2018-03-timers)、[folly HHWheelTimer](https://github.com/facebook/folly/blob/main/folly/io/async/HHWheelTimer.h)、[Linux kernel/time/timer.c](https://codebrowser.dev/linux/linux/kernel/time/timer.c.html)。
- 反例（为何不选堆）：Go runtime per-P 4 叉堆、libev 4-ary heap——定时器少 + 需精确到期 + 取消少的场景，与百万 agent 负载相反。
