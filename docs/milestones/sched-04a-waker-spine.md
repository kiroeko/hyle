# sched-04a：Waker completion 脊（suspend / wake，race-free）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台 4/4 Waker；**WSL TSan 8/8 零 race**）|

> **范围**：sched-04 的**纯 Waker 脊**半边——fiber `suspend()` 离开运行循环、外部 `Waker::wake()` 从任意线程把它重新注入。**deadline/timer 接入(sched-04b)依赖 time-02(占位),按"不依赖后面功能"原则拆出**。
> **cloc**：生产 ~114（waker.h 26 + fiber.h +34 + scheduler.cc +50）+ 测试 96。

## 为什么是它(架构拱顶)

sched-design §3 原话:"统一的本质是唤醒网(completion 脊)"。HYLE 区别于"又一个库"的核心论点——GPU 算完 / 网卡收包 / timer 到期都回到**同一个调度器、唤醒同一个 fiber**——全靠这一个 `Waker` 接缝。io(io_uring/IOCP CQ 收割)、compute(GPU event)两支柱都只通过 `Waker::wake()` 与 sched 耦合,互不知细节。本步把这个接缝做实。

## API

- `Waker{Scheduler*, Fiber*}`(trivially copyable,任意线程 `wake()` 一次/episode)。
- `current_waker()`:在 fiber 内拿到命名"本调度器+本 fiber"的 Waker。
- `suspend()`:把当前 fiber **移出运行循环**(区别于 `yield()` 立即重排),直到 `wake()` 重新注入。
- `Fiber::current()`:复用 fiber 已有的 running-fiber thread-local。

## 核心难点:wake-before-suspend 竞态(对称 rendezvous)

最微妙的 race:`wake()` 在 fiber **还没完成挂起**时就从另一线程到达。解法是 fiber 上一个原子 `wake_state_` 的**对称握手**——worker 的 dispatch(在 `resume()` 返回后、见到 wait 请求时)与 `Waker::wake()` **都**调 `wake_rendezvous()`:`exchange(1)`,谁看到对方已置 1 谁做那唯一一次重新注入。

**关键正确性**:dispatch 只在 `resume()` 返回**之后**才握手 → fiber 必已完全挂起(fcontext 已存、控制回到 worker 栈)→ **重新注入永不会在 fiber 仍在栈上时发生**,绝无两线程同时 resume 同一 fiber。`acq_rel` exchange 双侧保证重注入侧看见挂起完成。握手完回 0,下个 episode 干净。

- wake 先到:worker dispatch 后握手得 1 → worker 重注入(post-resume,安全)。
- worker 先挂起:wake 握手得 1 → wake 重注入(fiber 早已挂起,安全)。

`outstanding` 计数:挂起 fiber **仍计数**(never decremented while parked),`drain()` 正确等它;`reinject` 不动计数,走 global queue + notify(任意线程可调)。

## 对抗式自审(六类)

- **① 正确性**:4 测——前台 suspend→外部线程 wake→resume;**500 fiber wake-before-suspend 竞态**(wake 在 waker_ready 后立即打,赶在 suspend 完成前;丢唤醒→drain 挂→ctest 超时=硬失败);wake 从 worker fiber 内打(io-completion 常态);current_waker 离 worker 返 invalid。
- **② 性能**:rendezvous 单原子 exchange;热路径无新锁(reinject 复用既有 global+notify)。
- **③ 规范**:waker.h 公开头零 std 容器;Waker 是 {ptr,ptr} POD;`friend class Waker` 仅为 wake() 触 impl_->reinject。
- **④ TSan**:**WSL TSan 8/8 零 race**(Waker + Scheduler 一起),lock-free 脊硬闸门过。
- **⑤ 残留**:无。deadline 半边显式拆 sched-04b(依赖 time-02),非略过。
- **⑥ 过度复杂**:只做脊本身,未投机做 channel/CV(sched-05)、deadline(04b)。

## 交付物

- `include/hyle/sched/waker.h`(Waker + current_waker + suspend)
- `include/hyle/sched/fiber.h`/`.cc`(current() + wait_requested + wake_rendezvous)
- `src/sched/scheduler.cc`(dispatch wait 处理 + reinject + tls_owner)
- `tests/sched/waker_test.cc`(4)

## 参考

- sched-design §3(Waker 统一接缝、completion→唤醒哪个 fiber)。tokio `Notify`/folly wake 竞态处理。sched-03(run-queue + notify 复用)、platform-01(Parker)。
