# sched-09：并发原语补全（Semaphore / RwLock / oneshot / watch / broadcast）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | 能力 C7 驱动 |
| **状态** | ✅ 完成（2026-07-03；双平台 73/73 sched 测试 + WSL TSan 零 race，sync_test 新增 14）|

> **范围**：对照 SOTA（tokio/seastar/folly 审计）补齐 sched-05 之外仍缺的真原语（能力 C7）。tokio 语义抄全 + **建在现有 fiber-aware CV 地基上**（marl 教训：一套唤醒机制，非照抄 tokio lock-free future）。
> **cloc**：生产 ~1.5k（sync.h 内五组原语）+ 测试（sync_test 新增 14）。

## 为什么是它（sched-05 补齐 SOTA 仍缺的件）

2026-07-03 参考库审计（tokio/seastar/folly 对照）发现 sched-05 补齐了 CV/Mutex/WaitGroup/Latch/Channel，但对照 SOTA 仍缺几件异步标准件：计数信号量（限流）、读写锁、单值回传、最新值传播、多消费者 fan-out。全建在 CV 地基上成本低（sched-design §2.4 补记登记为 C7 驱动）。

## 核心决策：全部层在 CV 上（marl 收敛）

不为每个原语写一套唤醒机制——延续 sched-05 的 marl 收敛策略，五组原语全部建在 fiber-aware `ConditionVariable` 上。存储走注入 allocator（charter：无隐藏堆），refcounted 共享态最后一 drop 释放。

## 五组原语

- **`Semaphore`**：fiber-aware 计数信号量。**严格 FIFO 队头不跳过**（tokio batch_semaphore 语义）——想要 n permit 的 waiter 坐队头跨 release 累积 permit，release **绝不**跳过它去满足更小的后来者。这防大请求饥饿（也正是 RwLock 的 writer——想要全部 permit——不会被读者流饿死的原因）。每个 waiter 自持栈上 CV，release 恰好 1:1 唤醒队头（无惊群 recheck 共享谓词）。RAII `Permit` + `forget()`（永久移除 permit 缩容）。
- **`RwLock`**：write-preferring（无写饥饿）。**直接建在 Semaphore(kMaxReaders) 上**（tokio 设计）：read 拿 1 permit / write 拿全部。因 Semaphore 不跳 FIFO 头，排队的 writer（要全部 permit）让后续 readers 排它后面，**公平白得**，无手写读写计数器。RAII ReadGuard/WriteGuard + `WriteGuard::downgrade()`（原子降级：留 1 permit 使锁在转换中从不全空，队列 writer 无法插入）。
- **`oneshot`**：move-only SPSC 单值。send/recv 各消费一次；Sender drop 未 send → recv 返 Cancelled；Receiver drop → send 把值退回错误；`Sender::closed()` async 取消钩子（producer await 到 rx 走了就放弃昂贵计算）。两个独立 CV（sender 等 "rx gone"、receiver 等 "value/tx gone"，tokio 两 slot）。
- **`watch`**：版本号最新值广播。Receiver 可复制（订阅多个），只见**最近**值（无累积）；`changed()` 阻塞到 version 变；`borrow_and_update`/`send_modify`/`send_if_modified`/`send_replace`/`receiver_count`。配置热载 / 关停信号原语——晚订阅者立即见当前值。
- **`broadcast`**：环形缓冲多消费者 fan-out——每 Receiver 观测**每个**值（不同于 watch 只留最新）。每 receiver 独立游标；落后超 capacity 得 `Lagged` 快进不阻塞 sender（tokio overwrite-oldest：sender **永不**因慢读者阻塞）。一个 SpinLock 守整个 ring（非 tokio per-slot mutex：正确且简单，per-slot 吞吐精化待真实高 fanout 消费者需要，charter 无投机复杂度）。

## 核心难点：close flag 必须 mu-guarded（防 lost wakeup）

oneshot/watch/broadcast 的 close 语义都有同一陷阱：`closed`/`tx_alive` 等 flag **必须在 mu 下翻转**，不能从 lock-free 的 refcount/sender 计数推断。因为 waiter 在 mu 下 re-check predicate——若通过无锁计数翻转，notify 会 race ahead of the flag，receiver 永久 park（lost wakeup → 所有 sender 走后永久 hang）。代码里每处 close 都显式在 mu 下置 flag、再 notify。

## 对抗式自审（挑适用类）

- **① 正确性**：双平台 73/73 sched 测试（sync_test 新增 14）——Semaphore FIFO 不饥饿、RwLock 写不饿、oneshot 双向 drop、watch 版本、broadcast Lagged 快进。
- **③ 规范**：全建在 CV 上（一套唤醒），存储走注入 allocator。
- **④ TSan**：WSL TSan 零 race（含 close flag mu-guarded 的 lost-wakeup 防护）。
- **⑥ 过度复杂**：broadcast 用一个 SpinLock 守 ring（非 tokio per-slot lock），per-slot 精化待真实高 fanout 消费者——不投机。

## 交付物

- `include/hyle/sched/sync.h`（Semaphore + RwLock + OneshotSender/Receiver + WatchSender/Receiver + BroadcastSender/Receiver + make_oneshot/make_watch/make_broadcast）
- `src/sched/sync.cc`（Semaphore acquire/release/close FIFO 队列 out-of-line）
- `tests/sched/sync_test.cc`（新增 14）

## 参考

- sched-design §2.4（同步/结构化并发缺口补记：Semaphore/RwLock/oneshot/broadcast·watch 登记 C7）。tokio batch_semaphore（FIFO 不跳头）、tokio RwLock（Semaphore(kMaxReaders) + downgrade）、tokio oneshot/watch/broadcast 语义。sched-05（fiber-aware CV 地基 + marl 收敛）。
