# io 实测铁证

> 本页是 [performance 总账](./README.md) 的 io 详节:proactor 三后端机制 + echo 吞吐/扩展性对标。环境:Linux GCC14.2 Release(原生 ext4,32 核)+ Win MSVC19.50 Release,nanobench + `scripts/bench-stat.py` n=15 统计口径。bench 源:`bench/io/echo_bench.cc`。

## 为什么 io 的对标要等到有真实并发连接负载

io-01/02/03 阶段**故意不出性能数**(见总账 io 行的"缺口登记")。理由:孤立测单个 `read`/`write` 是在测**内核的 io_uring/OS**,不是测 HYLE 的薄封装——那个数字对"HYLE 是否 SOTA"零信息量。真正有意义的对标必须是**真实并发连接负载**:HYLE echo server vs 传统就绪式 epoll server,在多连接、多 round-trip 下比吞吐与扩展。io-04 落地(co_await socket 层 + shard-per-core)后,这个负载才存在,故现在补。

## echo 吞吐 + shard-per-core 扩展(`bench/io/echo_bench.cc`)

**负载**:64 个连接,每个连接 64 次 echo round-trip(client send→server recv→server echo→client recv = 1 round-trip),payload 64B。连接按 owner 核轮询分片(shard-per-core),每连接的 client/server fiber 都 `spawn_on` 钉在其 owner 核。报 round-trip/秒。

**Linux io_uring(n=15 统计,权威)**:

| 配置 | 中位 ns/echo | echo/s | 对裸 epoll 比 | 判定 |
|---|---|---|---|---|
| HYLE shard-per-core 1 worker | 2271 | 440K | 0.61x | ❌输(见下根因) |
| HYLE shard-per-core 2 workers | 1229 | 814K | 1.10x | 越过基线 |
| HYLE shard-per-core 4 workers | 742 | 1.35M | **1.81x** | ✅赢 |
| HYLE shard-per-core 8 workers | 672 | 1.49M | 2.02x | ✅赢(vs 1w **3.27x 扩展**)|
| 裸 epoll 1 线程(就绪循环) | 1359 | 736K | 1.0x | 基线 |

统计判定(bench-stat.py,仅当 gap > 合并 spread 才判输赢):
- 1 worker / 裸 epoll:ratio 0.61,gap 39% » spread 22% → **确输**(非噪声)。
- 4 workers / 裸 epoll:ratio 1.81,gap 81% » spread 25% → **确赢**。
- 8 workers / 1 worker:ratio 3.27,gap 227% » spread 24% → **确赢**(近线性扩展到 4 核后渐平,64 连接 /8 核 = 8 连接/核,socketpair 内核争用显现)。

**Windows IOCP(首轮指示值,loopback TCP)**:1 worker 93K → 4 workers 200K echo/s,**~2.1x 扩展**。绝对值低于 Linux——Windows 无 socketpair,用 loopback TCP 模拟(比 Unix domain socket 重),且 IOCP 每 op 一个 OVERLAPPED 堆分配。扩展趋势与 Linux 一致(shard-per-core 生效)。

## 诚实的发现:单 worker 输裸 epoll 0.61x —— 根因诊断,非 bug

这是必须直面而非粉饰的结果。**根因是 fiber/协程模型的固有 per-op 开销 + 当前每 submit 一次 syscall**,不是缺陷:

- **HYLE 每 round-trip 走 4 个 op**(client send/recv + server recv/send),每个 op 的路径 = `submit`(当前实现每次一个 `io_uring_enter` 系统调用)→ fiber 挂起(fcontext 切换)→ worker reap → Waker 唤醒 → fiber 恢复 → 协程续跑。**≈4 次 `io_uring_enter` + 8 次 fiber/协程上下文切换 / round-trip**。
- **裸 epoll 每 round-trip** = 紧凑的 `epoll_wait → read → write`,**零上下文切换**,且 `epoll_wait` 一次批量返回多个就绪 fd(低并发下摊薄 syscall)。
- 量级核对:fiber 切换实测 ~21ns(见 [sched 详节](./benchmarks-sched.md)),8 次 ≈ 170ns;但 1w 与 epoll 的差 ≈ 900ns/round-trip,故**主导项是 4 次 submit 的 `io_uring_enter` syscall**,不是 fiber 切换。

**这是 fiber 模型换取的代价**:per-op 开销换来 ① co_await 人体工学(遇 IO 让出,任意调用深度)② 越过单线程天花板的透明扩展(4 worker 就反超裸 epoll、8 worker 3.27x)。裸 epoll 的紧凑循环卡在单核,加核要手写分片 + 无栈状态机。

**优化轴(已知,接口已预留,本阶段未做)**:
- **批量 submit**:攒多个 SQE 再一次 `io_uring_enter`(liburing 默认行为)。当前每 submit 一次 enter 是最朴素形态。这一项能直接砍掉主导的 syscall 开销。
- **SQPOLL**:内核轮询 SQ,submit 完全免 syscall(`Proactor` 接口按 liburing v2.15 研究过,不挡此升级)。
- **multishot recv**:一次 submit 多次完成,省重复 arm。

补齐这些后单 worker 有望追平/接近裸 epoll;但**shard-per-core 的扩展性(核心卖点)已实证**,是 io serving 北极星负载的正解。

## 试过并回退的优化(2026-07-02,诚实登记失败)

两个针对"单 worker 输裸 epoll"的优化被实现、实测、**回退**——它们在本 ping-pong echo 负载上是**净退步**,不该上架一个实测更慢的东西:

1. **submit 批量**(填 SQE 不 enter,攒到 reap 一次 flush):本负载是**严格 ping-pong**,每连接同时只有一个 op 在飞,**没有可批量的并发**;把 flush 推迟到阻塞 reap 反而**每 op 多加一个调度往返延迟**,1 worker 从 440K 掉到 124K。
2. **park 合并**(worker 空闲阻塞在 reap 等 completion + eventfd 跨线程唤醒,替代 parker.park_timeout(1ms)):即便 submit 立即 flush,1 worker(334K)仍慢于旧的 parker+1ms(440K);仅 4 worker 反超裸 epoll。本延迟受限负载净退步。

**根因**:这两个都是**吞吐导向**优化,需要**流水线/多 op 在飞**的负载才能兑现价值;当前 ping-pong echo bench 每连接单 op 在飞,批量无从批起、park 合并的延迟收益被阻塞 reap 的开销吃掉。**教训**:优化必须过它自己的 benchmark;"机制上更优"不等于"这个负载上更快"。真要提速单 worker,方向是:①**流水线 echo bench**(每连接多个 outstanding op)才能让批量发挥;或 ②**SQPOLL**(内核侧 submit,零 per-op enter)——它对延迟受限负载也有效。均未做,登记。

代码已回退到 a922baa 基线(submit_one 每 op 一 enter + parker 1ms park),即本页上表的数。

## 机制层为何"方向对"(补充)

除吞吐数外,proactor 机制本身借的就是各平台 SOTA、封装零额外结构开销:
- reap 时 CQ 非空**纯用户态**收割零 syscall;批量 N 完成 1 次 release-store(io_uring for_each_cqe 形状)。
- timeout 走 `IORING_ENTER_EXT_ARG` 不额外消耗 SQE。
- submit 挂 token 无拷贝(`user_data` = `IoOperation*`,完成时 verbatim 返回)。
- 后端本身 = io_uring(Linux)/ IOCP(Windows)/ epoll(降级),各平台 proactor/reactor SOTA,非自造。

## 未做(登记)

- **裸 liburing 对标**:证 HYLE 相对 liburing echo 的 0 封装开销。WSL 缺 `liburing-dev`;HYLE 本用裸 syscall,可后续在 bench 内手写等价 liburing echo 或装库补。
- **vs asio / libuv**:证不输成熟异步框架。
- **流水线 echo bench + SQPOLL**:让 submit 批量真正发挥(见上"试过并回退");当前 ping-pong bench 无法暴露其价值。
- Windows IOCP 的 n=15 统计版(首轮为 3→20 epoch 指示值)。
