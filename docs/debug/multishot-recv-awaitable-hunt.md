# multishot recv awaitable debug（单槽 clobber + 三次假 bench 结果 —— 已定位并修复）

> 调查日志（已结案）。关联:[`../performance/benchmarks-io.md`](../performance/benchmarks-io.md)("multishot recv awaitable"节)、`net_multishot.h`、commit `e19312d`(awaitable)+ `740630e`(bench)。
> 结论先行:① 正确性——multishot 是**完成流**,一次 reap 批可交付多个完成;初版用单个 `result` 字段当落点,后一个(EOF)覆盖前一个(6 字节数据)→ 静默丢数据。修:slot 内建完成队列 + 只在 empty→non-empty 转移 wake。② 性能——三次"惊人加速"全是 bench bug 造的假象,逐个消除后诚实结论是 **vs one-shot 统计平局**。

## 背景
proactor 早已实现 `submit_recv_multishot` + provided-buffer-ring,但 net.h 只暴露 one-shot RecvOp——那零拷贝多发路径是 co_await 够不着的死代码。#50 把它提升为 `MultishotRecvStream` + `co_await stream.next()`。

## 第一部分:正确性 —— 单槽 clobber

### 现象
回归测试 `MultishotRecvAwaitable`(DGRAM socketpair 预塞 3 段 + peer close):`chunks=0`,fiber 立刻拿到终止空 chunk,数据"aabbcc"没收到。

### 探针
在 `complete_multishot`(router)打每个完成的 result/flags:
```
[ms] complete result=6 flags=0x3 bid=0   ← 数据:6 字节 "aabbcc"(kMore|kBuffer)
[ms] complete result=0 flags=0 bid=0     ← 终止:EOF(kMore clear)
```
**两个完成在同一 reap 批到达**,`complete_multishot` 都写同一个 `slot->io.result`——第二个(result=0 EOF)**覆盖**了第一个(result=6 数据)。fiber 恢复时读到 result=0 → 当终止,数据丢了。

### 根因
one-shot awaitable 的模型是"单 IoOperation slot,一次完成写一次读一次"。**multishot 是完成流**:一次 `submit_recv_multishot` 交付 N 个完成(kMore 置位续流),且**多个可在一次 reap 批同时到达**。单个 result 字段装不下流——后到的 clobber 前面未被消费的。

### 修法
`MultishotSlot` 内建一个**完成队列**(单线程单生产者[reaper]单消费者[fiber],无锁环):`complete_multishot` push、`next()` pop。关键的 wake 纪律:**只在 empty→non-empty 转移时 wake**——fiber 醒后 `await_ready` 持续为真直到队列抽干才再挂起,故一次挂起周期恰一次 `wake_rendezvous`(否则第二次 wake 泄漏 wake_state_,同 net_timeout 的双-waker 陷阱)。取消安全用堆槽 refcount{kernel,fiber},kernel-ref 撑到 kMore-clear 终止(stream 提前析构则 submit_cancel + 迟到 -ECANCELED 落在仍存活的槽)。

## 第二部分:性能 —— 三次假结果

用户要求"测性能",第一次汇报我只做了正确性、性能含糊带过——这是没做完。补做时撞了三次假象,全靠"太好的比值先疑自己 + 完成计数核对工况对称"揪出。

| # | 假结果 | 真相(探针/计数) |
|---|---|---|
| ① | multishot "63x 更快" | 我给每流各注册一个 buf ring,但 **proactor 每核只支持一个 ring**(`buf_bgid>=0 return false`)→ 第 2 条注册失败 → multishot run 提前 bail,`chunks=0`,那 34ns/msg 只是空转。 |
| ② | 挂死 | 预塞式 feed:64KB flood 塞满 socket 缓冲,而 drain fiber 还没被调度跑 → 阻塞写死锁。 |
| ③ | multishot "178x 更快" | 后台 feeder 线程注入**跨线程唤醒延迟**(WSL2 ~12µs 地板)主导 one-shot 行(174µs/recv);且 STREAM 合并 64B 写,致两行**完成数不等**(`chunks=12.8万` vs `recvs=24.4万`)——根本不是同工况。 |

### 干净版设计(逐个消除混淆)
- 单 ring 共享(所有流用同 bgid)——绕开每核一 ring 限制。
- **同 worker 内** sender fiber + receiver fiber——无跨线程唤醒地板。
- **DGRAM** socketpair——保消息边界,N send = N recv 完成,**两行完成数严格相等**(铁证:都收 2000)。
- receiver 收满确定 N 停——DGRAM 无可靠 EOF terminal,确定计数最干净。
- 唯一变量 = receiver 收方式。

### 诚实结论(n=15)
multishot 165,679ns vs one-shot 166,985ns/datagram,ratio 0.998、gap 0.2% « spread 13.4% = **统计平局**。**符合预期**:lockstep 同核 sender↔receiver 只 ~1 datagram 在途,~166µs 由两行都付的 fiber 往返主导;multishot 拿不到"一批完成同时到"来摊薄省下的 submit(那才是它的价值)。它的优势需突发/高扇出负载,单 worker socketpair 造不出来。

## 教训
1. **one-shot 的单结果槽模型不能套到 multishot**。凡"一次提交多次完成",先想清楚完成如何**排队**(单槽会 clobber)、wake 如何**恰好一次**。这是任何 multishot awaitable 的通用陷阱。照抄参考库(tokio-uring Lifecycle / proxy.c 每完成独立处理)本就该队列化。
2. **"太好的比值"必先疑自己**。63x/178x 两次假象都靠**完成计数**(两行必须相等)当铁证核对工况对称揪出——裸 ns/op 比会骗人。
3. **性能是交付的一半,别只测正确性**。用户明确"做了要测性能",我第一次漏了。平局也是发言权:现在能说"multishot 在 lockstep 负载不快于 one-shot 且知道为什么",而非猜它快。
4. **bench 的测量陷阱和被测代码一样需要严谨**:预塞死锁、跨线程唤醒地板、STREAM 合并——每个都能把结论带偏。

## 相关 commit
- `e19312d` — multishot recv awaitable(net_multishot.h,含单槽→队列修正 + wake 纪律 + 堆槽 refcount)。
- `740630e` — 干净对照 bench(同核 DGRAM,统计平局)+ 三次假结果的方法学记录。
- `0f208bc` — 标注"已就位待承重"定位 + 回填 io-design。
