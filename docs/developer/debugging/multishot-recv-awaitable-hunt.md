# multishot recv awaitable debug（单槽 clobber + 三次假 bench 结果 —— 已定位并修复）

> 调查日志（已结案）。关联:[`../performance/benchmarks-io.md`](../../implementation/evidence/performance/benchmarks-io.md)("multishot recv awaitable"节)、`net_multishot.h`、commit `e19312d`(awaitable)+ `740630e`(bench)。
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

## 第三部分:承重尝试 —— HTTP body 不适配(先疑自己排除)

平局后要让 #50 从"待承重"变"承重",第一想法是**拿 multishot 改 HTTP body 流式读**(body 正是"多段陆续到")。动手前先疑自己,读 http.cc body 路径,发现**架构不兼容**:

- HTTP body 读建立在 `read_more` → `s.read(buf.data+buf.size, ...)`——读进**连接自己的 GrowBuf 的指定 offset**,因为 head/body 必须在该 buffer 里**连续**(phr 重解析、chunked in-place 解码、pipelined 剩余留在 buffer 头)。`HttpRequest::body()` 返回单个连续 `string_ref`。
- multishot 的模型**恰相反**:**内核**从 ring 选 buffer,你拿到指向 ring 内存的 `RecvBuf`,必须 `buf_ring_return`。**无法**让 multishot"读进我的 GrowBuf 的 offset X"。

**结论**:强套 multishot 进 HTTP body,得把每个 RecvBuf 从 ring **拷进 GrowBuf**——**加一次拷贝、抵消零拷贝收益**,而"免 per-recv submit"对单个 body 读又微乎其微。**这是 pessimization 伪装成"承重"**。

**教训(第 5 条)**:一个原语"待承重"不代表随便找个消费者塞进去就行。**消费者的数据形状必须匹配原语的模型**——multishot 要"按 chunk 消费、不需连续、每块可独立转发"(proxy/echo/SSE-consume);HTTP 现有"整个 body 连续物化"的 API 结构上就吃不到它。**先疑自己 + 读消费者代码**,在写第一行前就排除了这个错误方向,省了一次"实现完才发现更慢"的返工。真正适配的承重路径是 **zero-copy echo/proxy**(multishot 收 chunk → 直接从 `buf_ring_addr` submit_send 转发 → 发完再 return,全程不需连续缓冲,正是 liburing proxy.c 模型)。

## 第四部分:zero-copy echo 也平局 —— 根因是 lockstep 掩盖 io 机制

按第三部分结论,写了 zero-copy echo bench(`multishot_echo_bench.cc`):multishot 收 chunk → 直接从 ring buffer `submit_send` 回显(零拷贝)→ 发完 RecvBuf dtor 归还。vs one-shot recv-into-buffer + send。

**又平局**:multishot 349,226ns vs one-shot 350,758ns/echo(0.4%,噪声),两行都 echo 2000。这是**第三个** multishot bench 平局(recv-drain、echo),它钉死了根因:

- 这个 echo **仍是 lockstep**(client 等每个 echo 回来才发下一个)→ ~1 op 在途 → ~349µs/echo 由**fiber 往返**主导(client send→server recv→server send→client recv,每跳一次 suspend/resume)。
- 零拷贝 send-from-ring 确实省了 64 字节拷贝,但那拷贝在 µs 级 fiber 调度前**微不足道**。

**根因(务必内化)**:**单 worker + lockstep 往返下,任何 io 机制优化(multishot / 零拷贝 / fixed-file)都不显——fiber 调度成本(~µs)碾压 io 机制节省(~几十 ns)。** multishot 的吞吐优势需要**真并发/突发**(数据到得比单消费者 drain 快),而单 worker lockstep bench 结构上造不出来。这也解释了本 session 所有 io 机制优化在 loopback 上的平局(fixed-file +2% 是唯一勉强可测的,因它省的 fget/fput 在内核侧不经 fiber)。

## 第五部分:#52 DEFER 默认打开 → 悄悄压垮 multishot(regression,已修)

做 echo bench 时 `multishot echoed=-1`(空转 skip)——`register_buf_ring` 返回 false。**先疑自己**查到:#52 让 IoContext **默认** DEFER(R_DISABLED),而 R_DISABLED ring **不能在构造线程 register buf-ring**(enable 在 worker 首 reap)。于是 #52 悄悄把所有 multishot buf-ring 路径打成 skip——连 `MultishotRecvAwaitable` 回归测试都从 PASS 变 SKIP(测试数不变但内容空转,正是"捏造铁证"警示的形态)。**修**:IoContext 的 `defer_taskrun` 改**opt-in 默认 false**(COOP)——恢复 multishot;DEFER 仍接着、留给无 buf-ring 的纯 echo/proxy 上下文显式开。教训:**一个默认开关的副作用会跨特性传染;改默认要跑另一特性的测试**(而且要看测试是真跑还是变 SKIP)。

## 相关 commit
- `e19312d` — multishot recv awaitable(net_multishot.h,含单槽→队列修正 + wake 纪律 + 堆槽 refcount)。
- `740630e` — 干净对照 bench(同核 DGRAM,统计平局)+ 三次假结果的方法学记录。
- `0f208bc` — 标注"已就位待承重"定位 + 回填 io-design。
- (承重)zero-copy echo 路径 —— 见 commit(HTTP body 不适配,改走 echo/proxy)。

