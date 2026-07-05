# io 实测铁证

> 本页是 [performance 总账](./README.md) 的 io 详节:proactor 三后端机制 + echo 吞吐/扩展性对标。环境:Linux GCC14.2 Release(原生 ext4,32 核 i9-13900K)+ Win MSVC19.50 Release,nanobench + `scripts/bench-stat.py` n=15 统计口径。bench 源:`bench/io/echo_bench.cc`(吞吐+worker 扩展)、`bench/io/io_scaling_bench.cc`(连接数 + payload 两条复杂度曲线)、`bench/io/accept_bench.cc`(建连速率 O(1)/accept)。

## 为什么 io 的对标要等到有真实并发连接负载

io-01/02/03 阶段**故意不出性能数**(见总账 io 行的"缺口登记")。理由:孤立测单个 `read`/`write` 是在测**内核的 io_uring/OS**,不是测 HYLE 的薄封装——那个数字对"HYLE 是否 SOTA"零信息量。真正有意义的对标必须是**真实并发连接负载**:HYLE echo server vs 传统就绪式 epoll server,在多连接、多 round-trip 下比吞吐与扩展。io-04 落地(co_await socket 层 + shard-per-core)后,这个负载才存在,故现在补。

## echo 吞吐 + shard-per-core 扩展(`bench/io/echo_bench.cc`)

**负载**:64 个连接,每个连接 64 次 echo round-trip(client send→server recv→server echo→client recv = 1 round-trip),payload 64B。连接按 owner 核轮询分片(shard-per-core),每连接的 client/server fiber 都 `spawn_on` 钉在其 owner 核。报 round-trip/秒。

**Linux io_uring(单跑中位,err% 1-3%,多次复现)**:

| 配置 | echo/s(批量后) | 对裸 epoll 比 | 判定 |
|---|---|---|---|
| HYLE shard-per-core 1 worker | **730~800K**(跑间中位波动) | ~1.0x | ➖平/微赢裸 epoll |
| HYLE shard-per-core 2 workers | 1.1~1.2M | ~1.5x | ✅赢 |
| HYLE shard-per-core 4 workers | 1.5~1.9M | ~2x | ✅赢 |
| 裸 epoll 1 线程(就绪循环) | ~730-790K | 1.0x | 基线 |

**关键转折(2026-07-03,submit 批量落地后)**:单 worker 从**曾经的 440K(输裸 epoll 0.61x)提到 ~730-800K(追平/微赢)**——约 1.7x 提升。机制铁证:`HYLE_IO_BENCH_STATS` 打出 **`avg_batch=13~61`**(每次 `io_uring_enter` 提交 13-61 个 SQE),对比批量前恒为 1.0。这直接砍掉了曾经主导的"每 op 一次 syscall"开销。**shard-per-core 扩展仍在**(1w→4w ~2x)。

## vs 裸 liburing:抽象的代价,与扩展性怎么赢回来(`-DHYLE_BENCH_LIBURING`)

raw epoll 对照答的是"vs 旧就绪模型";要答"vs **同样的 io_uring**、社区标准封装手写的紧凑循环",echo_bench 增了一个裸 liburing 对照行(gated `-DHYLE_BENCH_LIBURING`,链接 liburing.a)。同样 64conn×64rt×64B socketpair ping-pong,单线程单 ring 状态机(`user_data` 编码 conn<<2|stage,`submit_and_wait`+`for_each_cqe`),无 fiber、无协程、无调度器。与 HYLE 行在**同一次 nanobench run** 并排出相对比(同环境,最公平)。

**Linux io_uring 实测(同次 run,relative 列可信;绝对值受当次机器负载影响)**:

| 配置 | ns/echo | echo/s | 对 HYLE-1w |
|---|---|---|---|
| HYLE shard-per-core 1 worker | 1,869 | 535K | 1.0x |
| HYLE shard-per-core 4 workers | **709** | **1.41M** | 2.6x |
| **裸 liburing 1 thread(手写)** | **810** | **1.23M** | 2.3x |
| 裸 epoll 1 thread | 1,283 | 779K | 1.5x |

**诚实解读(这不是"HYLE 全面赢"的故事,是个更真实的故事)**:
1. **单线程比,裸 liburing 快 HYLE 1-worker ~2.3x**(810 vs 1869ns)。这是**协程/fiber 抽象的固定代价**,符合预期、非 bug:HYLE 每 op 完成要"唤醒 waker → fcontext 切换 → 协程恢复 → co_await 下一 op → 再挂起",每 round-trip 4 op 各摊一次 fiber 往返(~20ns 切换 + 协程胶水);裸 liburing 收割一个 CQE 后**直接在同循环 prep 下一个**,零上下文切换。抽象换的是**可组合性**(co_await 顺序代码 vs 手写状态机),代价是这段固定切换开销。
2. **HYLE 用 shard-per-core 把单核劣势赢回来**:4 worker(709ns)**反超**单线程裸 liburing(810ns)。裸 liburing 那个循环是单线程的,要多核得自己手写 sharding + 跨核协调;HYLE 的 `spawn_on` + 每核 ring 让连接分片到核是**免费的结构收益**。所以"SOTA"的正确表述:**单核吞吐输给零抽象的手写循环(抽象代价),但可组合性 + 开箱即用的 shard-per-core 扩展让它以多核反超**——这才是一个协程运行时相对裸 API 的真实定位。
3. 本次 HYLE-1w(535K)低于上文记的 730-800K 区间,是当次机器负载差异(我并发开着多个 WSL 会话跑对标);**同次 run 内的 relative 比不受影响**,故对标结论以 relative 列为准。

## vs asio / libuv:抽象 vs 抽象——跟成熟异步框架打平(`-DHYLE_BENCH_ASIO` / `-DHYLE_BENCH_LIBUV`)

裸 liburing 答的是"vs 零抽象手写循环"(HYLE 输 2.3x,抽象的固定代价)。但那不是 HYLE 的真实竞品——真实竞品是**同样提供抽象的成熟异步框架**:C++ 的 **asio**(io_context + completion handler,THE 标准异步库)和 C 的 **libuv**(uv_loop + 回调,Node.js 引擎、最实战的异步运行时)。它们**同样付调度/抽象成本**,所以这才是"HYLE 在抽象层里输不输"的公平对照。echo_bench 增两个 gated 对照行(asio 走 io_uring 后端 `ASIO_HAS_IO_URING`+链 liburing;libuv 走 epoll),同 64conn×64rt×64B ping-pong,与 HYLE 行**同一次 nanobench run**,`bench-stat.py` n=15 统计判定。

**Linux 实测(n=15 跨run中位,判定规则:gap>合并spread 才算赢/输,否则统计平局)**:

| 对照 | HYLE-1w vs 它 | gap vs spread | 判定 |
|---|---|---|---|
| **vs asio**(io_uring,同后端) | 1868 vs 1695 ns | 8.9% < 23.0% | **统计平局** |
| **vs libuv**(epoll) | 1868 vs 2070 ns | 10.3% < 26.3% | **统计平局** |
| vs 裸 epoll(就绪循环) | — | 25.9% > 23.2% | 输(旧模型,已知) |
| **HYLE-4w vs asio-1w** | 789 vs 1695 ns | 109% > 37.6% | **赢 2.09x** |

**诚实解读**:
1. **单核 vs asio/libuv:统计平局**——HYLE 的 co_await + fiber + shard-per-core 抽象跟这两个业界最成熟的异步框架**在同后端上打平**(gap 都 < 测量 spread,不构成赢/输)。这是关键结论:**抽象没让我们输给成熟框架**。对比 vs 裸 liburing 的 2.3x 落后——那是"零抽象手写状态机"的代价;asio/libuv 也是抽象框架、同付调度成本,所以我们跟它们平。**抽象的固定代价是这一类运行时的行业共性,不是 HYLE 独有的短板。**
2. **多核 HYLE 2.09x 赢 asio**:asio 单 io_context 是单线程的,要吃多核得自己起线程池 + 手工把连接分到不同 io_context + 跨核协调;HYLE 的 `spawn_on` + 每核 ring 让 shard-per-core 是**开箱即免费**。这是 HYLE 相对单 loop 框架的结构优势。
3. **libuv 是跨模型对照**(它 epoll reactor,HYLE io_uring proactor),故最该对标的是 HYLE 的 epoll 降级路径/裸 epoll 行;单核与 HYLE-io_uring 平已说明 co_await 泵没引入额外成本。
4. **方法学诚实纠错(2026-07-05)**:两个对照都不是一次写对的——① asio 首版用 `shared_ptr`+`asio::post` 保活 client 状态机,posted lambda 跑一次即释放 → 悬垂;且 server 永久 re-arm 读、客户端读完不关连接 → `ctx.run()` 永不返回。改:值语义 functor(conns 向量提供稳定存储)+ 末轮 close 触发 server EOF。② libuv 更曲折:一开始 `uv_pipe_open` 裸 socketpair fd(单核环境下读一次就停),换 `uv_socketpair`+`uv_tcp_open` 仍每连接只做 1 round-trip。**用独立探针程序**(2conn×3rt、每回调打点)才定位到真根因:**libuv 契约里 `nread==0` 是"暂无数据"(类 EAGAIN)不是 EOF,只有 `nread<0`(UV_EOF)才是**;我的 `if(nread<=0) close` 在良性 n==0 上误关了连接,一次交换后就 EPIPE。改 `nread<0` 后 `rt_left=0` 全通。教训:**对照库的语义必须照抄准**(nread==0 vs <0 是 libuv 的经典坑),别拿 HYLE/POSIX 的"0=EOF"直觉套。

## 两处热路径优化:reinject 同核短路 + io_uring setup flags(2026-07-05,commit 93b1e23)

上面 vs-asio/libuv 的"单核统计平局"促成一轮 io 优化调研(复用 sched 机制 + 参考 asio/libuv/liburing 本地源码)。落地两处、都过 144 测试 + 双平台 TSan:

1. **reinject 同核 unpark 短路**(`src/sched/scheduler.cc`):io 完成被 worker N reap 时唤醒 owner=N 的 pinned fiber——shard-per-core 常态(submit+reap 同核,echo 每往返 4 次唤醒全落此)。原 `reinject` 无条件 `unpark_specific(owner)`,即向**正在跑 reap 的自己**发一次跨线程唤醒 syscall(io_uring/epoll eventfd write / IOCP PostQueuedCompletionStatus)+ parker 原子——纯浪费(worker 显然醒着)。加 `owner_idx == t_worker_idx` 短路跳过(镜像 `reschedule` 早有的同款逻辑);fiber 已在 pinned 队列,reap 返回后 `find_task` 立即取回,无丢唤醒。

2. **io_uring COOP_TASKRUN + SUBMIT_ALL**(`src/io/proactor.cc`):原 setup flags 全 0。接 `COOP_TASKRUN`(≥5.19,完成 task_work 不强制 IPI,靠 owner 线程常态 enter 自然收割)+ `SUBMIT_ALL`(≥5.18,批量提交坏 SQE 不中止整批),带 `-EINVAL` 内核版本回退阶梯(coop+submit_all → submit_all → 无 flag)。

**实测(`bench-stat.py` n=15,同 run 对照隔离归属)**:

| 配置 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| HYLE-1w echo | 1,868 ns | **1,255 ns** | **-33%** |
| HYLE-4w echo | 789 ns | **581 ns** | **-26%** |
| asio(同 run 对照) | 1,695 ns | 1,583 ns | ~持平 |
| raw epoll(同 run 对照) | 1,389 ns | 1,346 ns | ~持平 |

单核 -33% 后 **HYLE-1w 反超 asio(1255 vs 1583)与裸 epoll(1255 vs 1346)**——把之前 vs-asio 的"单核平局"变成"单核赢"。同 run 里 asio/epoll(无本改动)基本未动,隔离出收益确归本两处改动、非机器波动。

**诚实纠错 —— DEFER_TASKRUN 撤下(登记 follow-up)**:调研最看好的其实是 `DEFER_TASKRUN`(+SINGLE_ISSUER,≥6.1),它把完成 task_work 全延到 GETEVENTS enter 才跑,理论上最契合"每核一 ring、单消费者"。但**实测挂死基本 send/recv 往返**——深挖(探针 errno=17 EEXIST)定位真根因是 **SINGLE_ISSUER 线程绑定**:HYLE 在 IoContext 构造函数(主线程)建所有 ring,却在各 owner worker 线程 submit+reap,worker 首次 enter 就 EEXIST(次要因是 reap 路径 submit-only enter 不带 GETEVENTS,已修好、dormant 就位)。真正接入需 ring 生命周期改造(`R_DISABLED` 建 + 各 worker 首触发在自己线程 `ENABLE_RINGS`),登记为 follow-up(#52)。COOP_TASKRUN 拿到大部分收益(task_work 批处理)且无此约束。教训同 P0 hang:**优化契合直觉 ≠ 能跑,必须过它自己的 benchmark/测试**。

## multishot recv awaitable vs one-shot RecvOp:同核对称负载下统计平局(`bench/io/multishot_bench.cc`)

把 proactor 的 multishot recv + provided-buffer-ring 提升为 co_await awaitable(io-07b,commit e19312d)后,必须回答"它比 one-shot RecvOp 快么"。理论收益=免 per-recv submit + 零拷贝 ring 缓冲。设计一个**干净可信**的对照:同 worker 内 sender fiber + receiver fiber(无跨线程唤醒地板)、DGRAM socketpair(保消息边界,两行完成数严格相等)、receiver 收满确定 N 个后停(DGRAM 无可靠 EOF terminal)、唯一变量=receiver 收方式。

**Linux io_uring 实测(`bench-stat.py` n=15)**:

| 配置 | ns/datagram | 判定 |
|---|---|---|
| HYLE multishot(1 arm,ring 缓冲,无 per-recv submit) | 165,679 | — |
| HYLE one-shot RecvOp(每 datagram 一次 submit) | 166,985 | — |
| **比值** | **0.998,gap 0.2% « spread 13.4%** | **统计平局** |

**诚实解读**:multishot 在此负载**不快于** one-shot——这是**符合预期**的结果,非 bug。这个同核 sender↔receiver **lockstep** 只让 ~1 个 datagram 同时在途(sender 每 send 都 co_await、receiver 每 recv 都 co_await),~166µs/datagram 由**两行都付的 fiber 往返**主导;multishot 永远拿不到"一批完成同时到"的场景来摊薄它省下的 submit——那才是它的价值所在。multishot 的优势需要**突发生产者**(大量 datagram 在 receiver 消费前排队),lockstep sender 造不出来。故此 bench 证的是 **multishot 不是回归(在不利负载上与 one-shot 打平)**;要显出它的优势需要单 worker socketpair 造不出的高扇出/突发负载(登记为未来工作)。

**方法学血泪(三次假结果才到诚实平局)**:① 首版给每流各注册 buf ring,但 proactor 每核只支持一个 ring→第 2 条注册失败→multishot 行空转 bail,得"63x 更快"**假象**;② 预塞式 feed 死锁(flood 塞满 socket 缓冲而 drain fiber 还没跑);③ 后台 feeder 线程注入跨线程唤醒延迟(WSL2 ~12µs 地板)主导 one-shot 行(174µs/recv)、且 STREAM 合并致两行完成数不等,又是"178x"**假象**。逐个消除(单 ring 共享 / 同核 fiber 无跨线程 / DGRAM 保边界 / 确定计数)才得到可信的平局。**教训:"太好的比值"必先疑自己**——用完成计数(两行须相等)当铁证核对工况对称,别信裸 ns/op 比。

## pipelined / 高扇出(512 流)+ 一个被它揪出的 P0 hang

ping-pong(每流稳态 ≤1 op 在途)填不满 SQ,也测不到"海量 op 同时在途"的收割路径。故加一个 pipelined 负载:**512 条独立 echo 流共享 worker,每流 16 round-trip**——512 » ring 深度 256,一次 reap 唤醒的一批 fiber 各自 submit,几百个 op 同时在途。

**Linux io_uring(单跑中位,修复后)**:

| 配置 | echo/s | 相对 1w | 判定 |
|---|---|---|---|
| HYLE pipelined 1 worker | 328K | 1.0x | 基线 |
| HYLE pipelined 2 workers | 1.02M | 3.1x | ✅超线性(摊 syscall)|
| HYLE pipelined 4 workers | 1.63M | 5.0x | ✅ |
| HYLE pipelined 8 workers | 1.71M | 5.2x | ✅(err% 高,需增迭代)|

**这个负载一上来就 100% 挂死(P0)——根因是收割节奏的结构错**:批量落地时把 reap 挪到"run-queue 抽干"处,worker 空闲则走 futex parker(1ms 上限重 poll)。但那个"重 poll"是 `reap(timeout=0)`,**空 CQ 时只做 submit-only enter(不带 `IORING_ENTER_GETEVENTS`)**。io_uring 的完成 task-work **只在本线程带 GETEVENTS 进内核那一下才跑**——所以一旦 run-queue 抽干、几百 op 在途,那些 completion 永不 surface,worker 空转 `reap(0)==0`、`drain()` 死等 `outstanding!=0`。ping-pong 侥幸不挂:socketpair 的 send 在 submit enter 内联完成,CQ 稳态非空,纯用户态收割即可,从不需要 GETEVENTS。

**修法(正确性,非延迟优化)**:worker 的 park 边界改为——io hook 注册且**有在途 op**时,把阻塞交给 proactor 一次 **GETEVENTS `io_uring_enter`**,而非 futex park。hook 返回"是否有在途 op":有→在环里收割完循环重查队列;无→照常 futex park(可被跨线程 `unpark()` 立即打断,而阻塞 reap 不能——所以"无 io 在途"时仍是即时唤醒)。`Proactor::in_flight()` 三后端各自实现(io_uring 计 submit-not-completed、IOCP 查在途 OVERLAPPED 链、epoll 查 pending/posted)。阻塞 reap 用完整 timeout(timer deadline 或 INFINITE)——**跨线程唤醒经 eventfd 即时打断,见下"park 合并"节**(消除了这个正确性修复引入的空闲 busy-poll)。**诚实纠错**:此前一版按"CQ overflow"方向加了 `IORING_SQ_CQ_OVERFLOW` 恢复循环——实测 `overflow=0`,方向错了,真修到位后已撤(真病根是压根没做 GETEVENTS enter,不是 CQ 溢出)。

**方法学教训(再记一条)**:"能跑的负载"(ping-pong)绿 ≠ 机制对。是"做一个符合该机制的 benchmark"(pipelined 高扇出)才把这个结构性 hang 逼出来——**优化/机制落地后必须配一个真能压到它的负载去测**,否则潜藏的 P0 要到生产才炸。

## 曾经的"单 worker 输" + 怎么修好的(完整过程,诚实)

**初版(每 op 一次 `io_uring_enter`)单 worker 输裸 epoll 0.61x**。根因诊断:HYLE 每 round-trip 4 op,每 op = submit(一次 syscall)+ fiber 挂起 + reap + Waker 唤醒 + 协程续跑;裸 epoll 是紧凑 `epoll_wait→read→write` 零上下文切换 + 批量就绪。量级核对(fiber 切换 ~21ns×8≈170ns « 900ns 差)证明**主导项是 4 次 submit syscall**,非 fiber 切换。

**两次失败的批量尝试(2026-07-02)**:先试"填 SQE 攒到 reap flush",但**没动 worker 循环节奏**——循环每轮顶部都 `reap(0)` 且每轮只跑一个 fiber(≤1 submit),SQE 在任何 reap 点都 ≤1 个,**批量结构上不可能**;推迟 flush 反而每 op 加一个调度往返,1w 掉到 124K。又试 park 合并(阻塞 reap + eventfd 唤醒),仍慢于基线。**当时误判为"这俩优化对本负载无用"并全回退**——但那个结论是错的,真正的病根没找到。

**照抄 tokio-uring 后找到真病根(2026-07-03)**:读本地 `tokio-uring` 源码(`runtime/mod.rs` + `driver/mod.rs`)——它**不在每个 task 之间 reap**,而是跑完一批 ready task 才在 tick/park 边界 flush(`on_thread_park`)+ SQ 满时 flush(`submit_op` 里 `while push().is_err() { submit() }`)。HYLE 病根 = **worker 每轮迭代都 reap**,SQE 攒不起来。修法两步:
1. **reap 挪到 run-queue 抽干时**(不是每轮):`find_task` 有活就跑(各自 queue SQE 不 enter),抽干才 reap → 一批 fiber 的 submit 攒成一次 enter。tokio 的 tick 模型。
2. **submit_* 填 SQE 不 enter**,drain reap 折 `to_submit=sq_pending` 一次提交;submit_one 在 SQ 满时自 flush(tokio 的 SQ-full 触发)。

这两步**配套**才成立——单独任一个都无效(这是前两次失败的教训)。落地后 avg_batch 13-61、1w 追平裸 epoll。**这就是"按规模自适应":低并发 SQ 不满→实际每 op 提交(低延迟);高并发→自动攒批一次 enter(摊薄 syscall),同一份代码无需显式判断规模。**

**方法学教训(记牢)**:① 优化必须过它自己的 benchmark;"机制更优"≠"这个负载更快"。② 失败先挖到真病根再下结论——我第一次误判"优化无用",其实是循环节奏这个更深的结构问题没看到,照抄 SOTA(tokio)完整机制才挖出来。③ 参考库拉到本地读源码,比凭记忆/上网翻快且准。

## 机制层为何"方向对"(补充)

除吞吐数外,proactor 机制本身借的就是各平台 SOTA、封装零额外结构开销:
- reap 时 CQ 非空**纯用户态**收割零 syscall;批量 N 完成 1 次 release-store(io_uring for_each_cqe 形状)。
- **submit 批量**:一批 fiber 的 SQE 一次 `io_uring_enter` 提交(avg_batch 13-61 实测),SQ 满自 flush = tokio-uring 同款。
- timeout 走 `IORING_ENTER_EXT_ARG` 不额外消耗 SQE。
- submit 挂 token 无拷贝(`user_data` = `IoOperation*`,完成时 verbatim 返回)。
- 后端本身 = io_uring(Linux)/ IOCP(Windows)/ epoll(降级),各平台 proactor/reactor SOTA,非自造。

## park 合并:空闲 worker 从 ~1kHz busy-poll 到真睡眠(2026-07-03)

P0 hang 修复时,空闲 io-worker 走"阻塞 GETEVENTS reap,但上限 `kIoPollNs`=1ms"——正确(完成能收割),但**每个空闲 worker 每秒醒来重 poll ~1000 次**。诚实量化(探针:1 个 op 永久在途、worker 空转 500ms,`HYLE_IO_BENCH_STATS` 计 `io_uring_enter`):

| 配置 | 500ms 空闲的 `io_uring_enter` 次数 |
|---|---|
| 1ms 上限(修复前)| **421**(≈1kHz busy-poll)|
| eventfd 唤醒(park 合并后)| **3**(init 武装 + 唤醒 + 收尾;真睡眠)|

**~140x 空闲 syscall 削减。** 机制:worker 有 io 在途时以**完整 timeout(timer deadline 或 INFINITE)**阻塞 reap,不再 1ms 轮询;跨线程 spawn/steal 经 `Scheduler::wake_worker` → `Proactor::wake()` 即时打断——io_uring 在专用 eventfd 上常驻一个 `IORING_OP_READ`(`wake()` 从任意线程 `write` 8 字节,**不碰单生产者 SQ**,故无 TSan race——这正是此前 NOP-SQE 版被撤的原因),IOCP 用线程安全的 `PostQueuedCompletionStatus`,epoll 用常驻 level-triggered eventfd。wake 完成带保留 `kWakeToken`:io_uring/epoll 在 reap 内丢弃并重武装、**不计入 in-flight**(否则无真 io 的 worker 永不 park futex),IOCP 经 io_context 过滤。**吞吐无变化**(ping-pong 1w 734K、pipelined 1w 323K,与合并前持平——负载下 worker 从不空闲,优化只作用于空闲态);双平台 + WSL TSan(net/tcp/udp/sched)净。

## 两条复杂度曲线:连接数 scaling + payload scaling(`bench/io/io_scaling_bench.cc`)

echo_bench 在**固定** 64-conn/64B 负载证吞吐 + worker 扩展;但"SOTA proactor"主张还依赖两条**复杂度曲线**——单点 ratio 证不了。故补 io_scaling_bench 专测形状。

### 1. 连接数 scaling:proactor 侧 O(1)/op(用户态有工作集残余)

**设计**:固定**每连接** round-trips(kRt=64,非固定总量),扫 N=64/256/1024 连接、单 worker(隔离出 proactor 成本,不掺 work-stealing)。因 R 按连接固定,batch(N×R)与 O(N) 建连成本同步增长,建连落成 ns/round-trip 的**常数偏置**而非随 N 膨胀项。

**Linux io_uring 实测(单跑中位)**:

| conns | ns/round-trip | round-trip/s | 对 64 档 |
|---|---|---|---|
| 64 | 1,875 | 533K | 1.0x |
| 256 | 1,954 | 512K | +4% |
| 1024 | 3,843 | 260K | +105% |

曲线**不平坦**——我一开始假设是"SQ 深度 256 被在途 op 打满、批量提交退化",但**被探针推翻**。`HYLE_IO_BENCH_STATS` 按档位聚合:

| conns | avg_batch | sqes | enters |
|---|---|---|---|
| 64 | 55.7 | 16,554 | 338 |
| 256 | 62.1 | 65,730 | 1,066 |
| 1024 | 52.2 | 262,291 | 5,028 |

**avg_batch 三档恒定(52-62)、sqes 随 N 完美线性**——SQ 深度 256 根本不是瓶颈,批量提交一路健康,**内核/proactor 完成路径是干净 O(1)/op**(io_uring/IOCP 完成模型不像 select/poll 重扫 fd 集)。

那 ns 上升来自哪?**栈大小 A/B 定位**:把 fiber 栈 64KB→8KB(活跃工作集 128MB→16MB)重跑——256 档 2190→1875ns(**64→256 那步被抹平**),但 1024 档 3827→3527ns(仍高 88%)。结论:64→256 的上升是 **fiber 栈 + Conn 的 cache 工作集效应**;1024 档剩余上升是更大规模的用户态/内核态常数因子(2048 fd 单 ring 注册、Conn 足迹),**均非 proactor 的算法复杂度**。这是所有 M:N 运行时的共性(活跃工作集越大 cache 越差),如实登记而非粉饰成"完美平坦"。

### 2. payload scaling:bytes/s 干净 plateau = 零额外拷贝

**设计**:固定 64 conn × 32 round-trip,payload 64B→64KB,报 bytes/s。小 payload 受 syscall 限,大 payload 逼近内存带宽;**无超线性拐点**才证 co_await 路径无额外 memcpy/逐字节遍历(awaitable 直接把调用方缓冲交给内核)。

**Linux io_uring 实测(单跑中位)**:

| payload | ns/byte | bytes/s |
|---|---|---|
| 64B | 30.60 | 32.7 MB/s |
| 1KB | 2.05 | 488 MB/s |
| 16KB | 0.25 | 4.00 GB/s |
| 64KB | 0.23 | 4.32 GB/s |

ns/byte 单调降、每档改善递减,16KB→64KB 仅差 0.02 = **已到内存带宽地板、干净 plateau,零拷贝证实**。(sweep 停在 64KB:256KB 单迭代搬 512MB、分钟级拖垮 bench,且 16KB 已见地板,无额外形状。)

**方法学**:此段是"先疑自己的假设"的活例——SQ-满退化是个看似合理的初判,靠探针(avg_batch/sqes)+ 栈 A/B 两步实验推翻并定位到真因,而非凭直觉写进文档。Windows IOCP 侧仅验证能编能跑 + 低档平坦(loopback TCP 模拟 socketpair 在 1024 连接反复建拆触发 TIME_WAIT 端口堆积,非 proactor 问题);真 scaling 数取自 io 主平台 io_uring。

## accept scaling:建连速率 O(1)/accept(`bench/io/accept_bench.cc`)

echo/scaling bench 都在计时区**外**预建 socket,从不测 accept。补第三条腿专测**建连路径**:一个 `TcpListener` 在 worker 0 循环 accept N 个连接,N 个 client fiber round-robin 分布到 4 worker 并发 connect,新连接经 `accept(assign_owner)` round-robin 分派到各核(seastar/HYLE 式"新连接分片到核"),accept 后立即 drop(close)只计建连。报 accepts/s、扫 N。负载模型参考 liburing `test/accept-mshot-stress.c`。

**Linux io_uring 实测(单跑中位)**:

| conns | ns/accept | accepts/s | err% |
|---|---|---|---|
| 64 | 268,178 | 3,729 | 0.5% |
| 256 | 261,944 | 3,818 | 0.3% |
| 1024 | 259,690 | 3,851 | 0.4% |

accepts/s 三档恒定(甚至微升,batch 摊薄)、err% <0.5% = **accept 路径 O(1)/accept 证实**,建连速率不随连接规模退化。

**绝对值(~260μs/accept)的诚实解读**:这不是内核 accept 本身的耗时,而是**端到端建连**摊到吞吐——每次含完整 TCP 三次握手(loopback 仍多次内核往返)+ client fiber 与 accept 分处不同核的 fiber 调度 + 跨核 wake 往返。测的是"建连吞吐是否随规模保持"(是),不是"单次 accept 有多快"。

**又一个"先疑自己"的活例(bench bug,非 proactor)**:冒烟首跑 64→256 每 accept 慢 **16x**(242 accepts/s、单档 36s),像 accept 路径崩了。诊断:一次性 spawn N 个 client 并发 connect,但默认 **backlog=128** 撑不住 256+ 涌入的 SYN → 丢弃 → **TCP SYN 重传(1s 退避)**。把 backlog 定为 conns 后曲线立刻平坦。教训:**测建连吞吐必须给足 listen backlog**,否则测的是 TCP 重传退避,不是 proactor。

## TLS 握手 + 加密 echo:vs 裸 OpenSSL(`bench/io/tls_bench.cc`,`-DHYLE_IO_TLS`/OpenSSL)

TLS 吞吐有**两块成本**:①OpenSSL 加密(握手非对称 + 记录对称,**不可避**,任何 OpenSSL 用户都付)②HYLE 的 memory-BIO 泵 + fiber 调度开销。单一吞吐数分不开,故 SOTA 基线=**裸 OpenSSL over 阻塞 socket**(同 cert/cipher/payload,server 独立线程,无 HYLE)——两者付相同加密地板,**比值即 HYLE co_await 抽象的净开销**。15 遍 `bench-stat.py` 双平台。

| 项 | HYLE(Win/Lin) | 裸 OpenSSL(Win/Lin) | 判定 |
|---|---|---|---|
| **加密 echo**(64×256B 记录往返,握手后稳态) | 46.1µs / 396µs | 47.4µs / 696µs | **Win 统计平局**(gap 3%<spread)/ **Lin 反超 1.76x** |
| **TLS 握手**(connect+handshake+close) | 2.03ms / 2.22ms | 1.56ms / 1.01ms | **~1.3x(Win)/ ~2.2x(Lin)慢**(gap>spread,真差) |

- **稳态 echo:泵零税(Win 平),异步重叠反超(Lin 1.76x)**。握手后每记录 read/write,加密成本主导;HYLE 的 co_await 泵在 Win 上与裸阻塞 OpenSSL **统计平局**(泵不加可测开销)。**Lin 上 HYLE 反超 1.76x**——WSL2 每 syscall 唤醒地板高时,阻塞 OpenSSL 每往返串行等,HYLE 的 proactor 异步能重叠 submit/reap,地板越高重叠收益越大。
- **握手:HYLE 慢 ~1.3-2.2x(诚实的真开销)**。TLS 握手是固定 ~4-6 flight 的 ping-pong;HYLE 每 flight 付一次 fiber suspend/resume + proactor 往返,裸 OpenSSL 是紧循环阻塞读写。这是抽象在**连接建立**(非稳态)上的真代价——对长连接 serving(握手一次、echo 无数次)影响小,对握手风暴敏感场景需知晓。**未来可优化**:握手期的 flight 批量/减少 yield。
- **⚠️ WSL2 地板(同 sched-11,必须诚实框定)**:同一 echo Win 46µs、WSL2 396µs=**~9x**——但**裸 OpenSSL 也 47µs→696µs(~15x)**,两者同被 WSL2 虚拟化的跨线程/syscall 唤醒地板放大。**故跨平台绝对值不可比**,只有**同平台内 HYLE vs OpenSSL 的比值**有意义(Win 平 / Lin HYLE 赢)。裸 Linux 上绝对值应回落。

## HTTP req/resp 吞吐:parse/framing 层开销 vs 裸 TCP 传输(`bench/io/http_bench.cc`)

TLS bench 隔出的是 co_await 泵的开销;HTTP 要隔的是**协议层**(picohttpparser parse + GrowBuf head/body 拷贝 + Request/Response 构建 + 状态行/头部格式化)相对**已被 echo_bench 测过的 io 传输**的净开销。故基线=**同一 co_await TcpStream 传输上的裸 TCP echo**(同 socketpair、同 fiber、同 proactor,搬运**逐字节相同**的请求+响应字节量,零 HTTP 工作)——两者传输成本相同,比值即纯 parse/framing 的 CPU 代价。负载:64 条 keep-alive 连接、每条 64 次 `GET /`→`hello-hyle` 请求/响应往返,shard-per-core 分片。socketpair 传输(同 echo_bench),无 accept、无 TIME_WAIT。

**Linux io_uring 实测(`bench-stat.py` n=15,跨run中位)**:

| 配置 | ns/req | req/s | 相对 1w | 判定 |
|---|---|---|---|---|
| HYLE HTTP 1 worker | 2,626 | ~381K | 1.0x | 基线 |
| HYLE HTTP 2 workers | 1,926 | ~519K | 1.36x | ✅ shard 扩展 |
| HYLE HTTP 4 workers | 1,093 | ~915K | 2.40x | ✅ |
| **裸 TCP echo 1 worker(同字节,无 HTTP)** | **1,827** | **~547K** | (传输地板) | — |

**诚实解读**:
1. **parse/framing 净开销 ~1.44x**(HTTP-1w 2626ns vs 裸 TCP-1w 1827ns;`bench-stat.py` 判 gap 33.3% > 合并 spread 26.5% = **真 LOSE 非噪声**)。这是**符合预期的真代价**:每请求要 phr_parse_request 扫头、head 拷进稳定 buffer 再重解析(修 realloc-dangle 的必要拷贝)、状态行+头部 put 进 out buffer、body framing——裸 echo 全省。~800ns/req 的绝对开销对小响应可见,对大 body/流式(传输主导)摊薄。
2. **shard-per-core 扩展仍在**:1w→4w 2.40x(2626→1093ns),证 parse/framing 层不破坏 shard 分片——每连接的解析全在 owner 核串行,无跨核共享。
3. **Windows(loopback TCP,self-timed,非 nanobench)**:

| 配置 | ns/req | req/s | 相对 1w |
|---|---|---|---|
| HYLE HTTP 1 worker | 11,597 | ~86K | 1.0x |
| HYLE HTTP 2 workers | 6,439 | ~155K | 1.80x |
| HYLE HTTP 4 workers | 3,571 | ~280K | 3.25x |
| 裸 TCP echo 1 worker | 9,827 | ~102K | (传输地板) |

parse/framing 净开销 **~1.18x**(11597 vs 9827),shard 扩展 1w→4w **3.25x** 更陡(Windows 单核基线更慢,并行摊得更开)。**跨平台绝对值不可比**(Windows loopback TCP + IOCP 唤醒地板 vs WSL socketpair,同 TLS bench 记的地板差),只同平台内 HYLE-vs-裸 的比值有意义:两平台都是 parse/framing 加一层可测但温和的开销(WSL 1.44x / Win 1.18x)。
4. **诚实纠错(方法学,2026-07-05)**:此 bench 首版我误判"Windows 挂死、loopback 限制未出数"。**真根因**:nanobench 的**自适应 epoch**在 Windows loopback 的高单次方差下**无界重试**逼近目标精度——进程**没挂**(内存平、TIME_WAIT 仅 5、端口充足),只是**每行跑分钟级**,我 3min kill 太早、又因 nanobench 只在末尾一次性打表而误读成"零输出=挂死"。**先疑自己**:探针分阶段(单 run→各 worker 档→重复 40 次→自计时)逐步排除,证单次 `run()` 40-48ms 正常、40 次重复正常完成,定位到 nanobench epoch 机制而非传输/我的 bench。自计时(固定 run 数)绕开该机制即出数。教训:**nanobench 不适合单次 op 达毫秒级、方差大的负载**(它为亚 µs 稳定核设计),这类负载用固定次数自计时。

## 热路径分配消除:TLS 泵 + HTTP 流式 chunk 缓冲复用(2026-07-05)

两处此前登记的"协程帧/循环体分配"优化点,改法都是**把缓冲挪出热路径复用**,收益是**确定性的分配次数下降**(结构性,不靠计时——~几十 ns 的 alloc 在 µs 级 syscall 噪声里测不准,故按 alloc 计数论证,同 sched 亚 µs 数字的方法学)。

| 优化 | 改前(每操作分配) | 改后 | 论证 |
|---|---|---|---|
| **TLS 泵 scratch**(tls.cc) | `flush_wbio`/`feed_rbio` 各一个 16KB 局部数组;因是 `sched::Task` 自由协程→落在**每次调用的协程帧**(Allocator operator new)。每个 `tls_read`/`tls_write` 至少一次 16KB 帧 alloc+free | 16KB scratch 移进 `TlsStream`,**懒分配一次、全连接复用、dtor 释放**;协程帧缩到几十字节 | N 次读写 = **N 次 16KB 分配 → 1 次**。单 fiber 独占 + flush/feed 串行(每个 `co_await` 完成才下一个)→ 一块共享 scratch 安全 |
| **HTTP 流式 chunk**(http.cc) | 流式响应每 chunk 一个 `GrowBuf frame(alloc)` = 每 chunk 一次 alloc+free | 复用 head 的 `out`(每轮 `size=0`),容量保留 | M chunk = **M 次分配 → 0 次**(首块后容量够就零 realloc)。`write_all` 每次全排空才返回→重置 size 前无在途写 |

**正确性验证**:TLS `HandshakeAndEncryptedEcho`(覆盖 handshake/read/write/shutdown 四条 scratch 路径)+ HTTP `GetKeepAlive`(含 3-chunk 流式)双平台(Win IOCP / Linux io_uring+epoll)+ **WSL setarch TSan 零 race**。scratch 复用未引入跨线程共享(单 fiber 串行),TSan 净证实。

> **为何不上计时 bench**:单次 TLS 读写的 16KB 帧分配 ~几十 ns,淹没在 loopback TLS 握手/加解密的 µs~ms 里;HTTP chunk 同理。分配**次数**是确定的、可从代码结构证明的,比一个被 syscall 噪声主导的计时数字更诚实。若要计时铁证,需计数分配器 microbench(登记,见下)。

## io 系统性红队审查（2026-07-05，对标 sched 那轮）

两个并行红队 agent 审全 io（下层 proactor 三后端 + net/tcp/udp + io_context;上层 tls/http/file/timeout/cancel），逐条人工核实。**审出 11 项全部处理:10 修(含 2 个 P1 + 1 个安全) + U5 调研判非 bug——零遗留,正确性审查彻底收口。**

**已修（commit ca317db，双向验证 + 双平台 + TSan）**：
- **L2 (P1)** epoll multishot accept:re-arm 失败仍标 kMore → 调用方永久挂起。改:只在 epoll_ctl 重挂成功才标 kMore（镜像 IOCP）。
- **L3 (P1, commit 2dcd6ce)** epoll 全双工同 fd 并发:`pending` 单槽 map<fd,op>,同 fd 并发 recv+send 时第二个覆盖第一个 → 首个 fiber 永久孤儿。改:拆 pending_in(EPOLLIN)/pending_out(EPOLLOUT) 双槽,reap 按方向分别服务 + re-arm 另一侧,事件掩码取并集。仅 epoll fallback 受影响(io_uring/IOCP 无此问题)。回归测试 `FullDuplexSameFdEpoll` 双向验证(pre-fix 12s 超时挂起 / post-fix 1ms PASS);forced-epoll + io_uring 双后端 × 7 套件 TSan 全绿。
- **U1 (P2 安全)** HTTP 请求走私(RFC 7230 §3.3.3):拒绝 CL+TE 并存 / TE 非 bare chunked / 重复冲突 CL。探针证 pre-fix 三 payload 全被接受(=真漏洞)。回归测试 `RejectsSmuggling`。
- **U2 (P2)** HTTP 客户端 HEAD/204/304/1xx 未抑制 body → 按 CL 读不存在的 body 挂起/desync。改:这些响应短路空 body。
- **U4 (P3 latent)** `GrowBuf::reserve` 增长 `>` 应为 `>=`,2-4GB u32 溢出死循环(当前不可达)。
- **U3 (P2, commit 694bb65)** HTTP 客户端 read-until-close body:无 CL 无 chunked 的响应(HTTP/1.0 或 Connection: close 流式)原只拷已缓冲的 head 包尾字节 → body 在后续包到达时静默截断(status 200 但 body 空/残)。改:read-until-EOF 循环(read_more 追加至对端关闭,带 kHttpMaxBodyBytes 上限)。回归测试 `ReadsCloseDelimitedBody`(裸 server 分离 head/body + 20ms 暂停)双向验证:pre-fix body_len=0 截断 / post-fix 全量;双后端 TSan 全绿。
- **L1 (P2/诊断, commit b033295)** submit-failure 哨兵 `-1` 撞 `-EPERM`(errno 1):net.h/net_timeout/net_cancel 的 submit 失败/OOM 用 -1 作 result,诊断时无法区分"从未提交"与真 EPERM。改:加 `kSubmitFailed=-1000`(errno 范围外,区别 kCancelled=-125/kTimedOut=-110),替换全部 9 处。调用方只测 result<0 不受影响,仅让值可辨识。双平台 11/3 io 测试全绿。
- **L4 (P2, commit 29da665)** epoll 伪就绪 EAGAIN 透传:epoll 报就绪后内联 read/recv/accept 若遇 -EAGAIN(伪就绪:数据/连接已不在)原样透传给 fiber(io_uring/IOCP 永不吐 EAGAIN),fiber 误当读失败关连接。改:EAGAIN 不投递 completion,重新 arm 该方向 watch 继续等真就绪(!err 守卫防 HUP/ERR 下自旋)。仅 epoll fallback。**与 L2/L3 同族**(epoll reactor→proactor 模拟保真度,非 L5/U5 的跨后端生命周期不变量)。伪就绪不可确定性构造→结构验证 + forced-epoll 7 套件非回归 + 双后端 TSan 全绿。
- **L6 (P3, commit 1ae247d)** IOCP `SO_UPDATE_ACCEPT_CONTEXT`/`SO_UPDATE_CONNECT_CONTEXT` setsockopt 返回值未查:失败会交出半配置 socket(getpeername/getsockname/shutdown/继承选项不工作,MSDN)。改:检查返回值,accept 失败关 socket + 返回 neg_err,connect 失败返回 neg_err,而非交坏 socket / 假 connected。IOCP-only,Win 全绿。

**红队核实为安全/契约化（清账）**:TLS scratch 复用(本 session 改)、net_timeout io-vs-timer 竞态仲裁 + 迟到 CQE refcount、net_cancel 栈 slot 单唤醒、parse_u64 溢出、HTTP head/body 视图跨 realloc 稳定性、IOCP teardown 顺序、buf_ring_return 无竞态、RecvFromOp/SendToOp scratch 生命周期、io_uring multishot inflight 计数。
- **L5 (P2, 契约化 → 彻底封死 commit 88f989a)** base net.h op 栈 slot 无迟到 CQE 保护:分析证正常 **drain-to-completion 下安全**——base op 不取消、只等自己的 completion、阻塞 fiber 计入 outstanding、shutdown drain 等它完成、IoContext dtor 在 scheduler 停后跑(此时无在途 base op)。唯一窄窗=`drain_for` 超时放弃 + base op 永不完成(且 `~Scheduler` 无条件 drain 本就挡住常规触发)。**与 U5 同源但不同**:U5 是"kernel ref 谁 drop"(refcount 保命,够);L5 是"栈帧活到 completion 没"(drain 契约保命)。**彻底封死(方案 A,零热路径成本)**:`~IoContext` orphan drain 对 base op 迟到 completion **只排空丢弃、不 route 到 complete()**(那会 wake 已销毁 fiber 的栈 slot),timed op 仍走 complete_timed 释放堆 ref。只改冷拆除路径,base op 运行时照样栈上零分配(不像"base 也改堆+refcount"会给每次 recv/send 上税)。回归测试 `BaseOpCompletionRoutingIntact` + 全 io 套件双后端 TSan 兜底。

**U5 (agent 报 P3 leak，调研后判定非 bug，2026-07-05)**:agent 称"timed-op 若 submit_cancel 失败(ring 满)+ 静默 peer → TimedSlot 泄漏"。**两个前提在 HYLE 上都不成立**:①**io_uring `submit_one` 永不因 SQ 满失败**(proactor.cc:1376——满时当场 flush 再入队,总返回 true);②**内核契约保证每个被消费的 SQE 必产生一个 CQE**(liburing `io_uring_prep_cancel.3`:"a CQE is always generated"),cancel 返回 false / -ENOENT 只意味着原 op 已完成、CQE 已在路上,`complete_timed` 仍会 drop kernel ref。**cancel 只是让完成来得更快的优化,不是 free slot 的必要条件**——tokio-uring `Lifecycle::Ignored` 只在 CQE 到达时 free;seastar 干脆不发内核 cancel,纯靠"每 SQE 必有完成"。HYLE 同款,+ `IoContext::~IoContext` orphan-drain 兜底(与 L5 共用同一机制)。唯一真漏点=**原 op submit 就失败**(SQE 从未被内核消费),已由 `on_submit_failed` 同步 drop kernel ref 堵住。结论已写进 net_timeout.h `resume_timed` 注释,防再次误报。**这也是 L5 的同族问题**:两者都靠"submitted SQE 必有完成 + orphan-drain 兜底"这一不变量,故 L5 亦安全。

**小结**:红队审出的 11 项**全部处理完毕**——10 修(chunked keep-alive + L1/L2/L3/L4/L5/L6 + U1/U2/U3/U4)+ U5 调研判非 bug。零遗留未决项。io 正确性审查彻底收口。

## 未做(登记)

- ✅ **HTTP chunked keep-alive bug(2026-07-05 发现并修复,fce90a6)**:症状——同一 keep-alive 连接上,一个 chunked 响应**之后**紧跟的下一个请求/响应解析失败(status 0、body 空)。**系统排查(先隔离再下结论,非臆断)**:①纯 `phr_decode_chunked` 循环探针(无 socket)证 HYLE 解码循环逻辑正确(3/50 chunk 单/逐字节全 OK)→排除解码器;②网络探针 `/big` 作为**唯一**请求时 3/8/16/50 chunk 全 OK→排除流式发送;③双 chunked keep-alive 请求探针复现:req1 OK、req2 失败→定位到"chunked 之后再 keep-alive"。**真根因**(推翻我最初"多次 -2 跨读"的猜测):`phr_decode_chunked` 默认 `consume_trailer=0` 时在 last-chunk `0\r\n` 后立即 Complete,把 chunked body 的**终止 CRLF**(RFC 7230 §4.1)当作"未解码尾字节"(rv=2)返回;HYLE 把这 2 字节当"下个 pipelined 消息"留 buf,污染后续 `phr_parse_*`。**修**:`decode_chunked` 设 `dec.consume_trailer=1`,解码器吃掉 trailer+终止 CRLF。server(chunked POST body)+ client(chunked 响应)共用同一函数两侧同修。回归测试 `GetKeepAlive` 加第 5 请求(chunked-after-chunked)双向验证(pre-fix FAILED / post-fix PASSED);Win + WSL io_uring + forced-epoll + TSan 全绿。
- **计数分配器 microbench**:给上面两处热路径分配消除补一个 alloc-count 铁证(计数 Allocator 跑 N 次 TLS 读写 / M chunk 流式,断言分配次数不随 N/M 增长)。scratchpad 探针已验证分配计数下降(structural),permanent 版可后补。
- ✅ **裸 liburing 对标**(已做,见上"vs 裸 liburing"节):liburing 本地 clone `../liburing`,`-DHYLE_BENCH_LIBURING=ON` 自动 configure+make 链接。结论:单核输 2.3x(抽象代价)、4 核反超。
- ✅ **TLS vs 裸 OpenSSL**(已做,见上"TLS 握手 + 加密 echo"节):稳态 echo Win 平/Lin 反超 1.76x,handshake ~1.3-2.2x 慢(真开销)。
- ✅ **HTTP bench**(已做,见上"HTTP req/resp 吞吐"节):64conn×64rt keep-alive `GET /` vs 同传输裸 TCP echo,parse/framing 净开销 WSL 1.44x / Win 1.18x(真 LOSE,符合预期),shard 扩展 1w→4w 2.4x(Lin)/3.25x(Win)。双平台出数;Windows 用自计时(nanobench 自适应 epoch 在 loopback 高方差下无界重试,不适合毫秒级 op)。
- **file io bench**:open/read/write/stat 吞吐(io_uring 真异步 vs IOCP/epoll 同步合成的代价差)。未做。
- ✅ **vs asio / libuv**(已做,见上"vs asio / libuv"节):单核与两者统计平局(抽象没输给成熟框架),多核 2.09x 赢 asio(shard-per-core 免费)。
- ✅ **io_uring setup flags**(已做,见上"两处热路径优化"节):COOP_TASKRUN+SUBMIT_ALL 已接(带内核回退),echo-1w -33%。
- **DEFER_TASKRUN + SINGLE_ISSUER**(登记 follow-up):最契合每核 ring 但实测挂死基本 send/recv(drain-time reap 的 submit-only enter 不带 GETEVENTS,defer 下完成永不 surface);需 reap 路径保证"有 op 在途必带 GETEVENTS enter"才能安全接。
- **registered files / multishot-recv awaitable**(调研中,agent 出设计):registered files(IOSQE_FIXED_FILE 省每 op fd 查表)+ 把 proactor 已实现的 multishot recv/provided-buf-ring 提升为 net.h awaitable。
- **SQPOLL**:内核侧 submit,连批量 enter 都免;但每核一 ring→每核多一个 kernel poller 烧 idle CPU,且与 DEFER_TASKRUN 互斥。**未做,也未预留接口**(不留死代码——真要做时在 `try_init_uring` 的 flag 阶梯加一档即可)。调研判 LOW,不建议现在做。
- Windows IOCP 的批量对应物(IOCP 无 submit 攒批概念,ReadFile/WriteFile 即时下发)+ n=15 统计版。
