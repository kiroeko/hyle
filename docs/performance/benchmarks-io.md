# io 实测铁证

> 本页是 [performance 总账](./README.md) 的 io 详节:proactor 三后端机制 + echo 吞吐/扩展性对标。环境:Linux GCC14.2 Release(原生 ext4,32 核)+ Win MSVC19.50 Release,nanobench + `scripts/bench-stat.py` n=15 统计口径。bench 源:`bench/io/echo_bench.cc`。

## 为什么 io 的对标要等到有真实并发连接负载

io-01/02/03 阶段**故意不出性能数**(见总账 io 行的"缺口登记")。理由:孤立测单个 `read`/`write` 是在测**内核的 io_uring/OS**,不是测 HYLE 的薄封装——那个数字对"HYLE 是否 SOTA"零信息量。真正有意义的对标必须是**真实并发连接负载**:HYLE echo server vs 传统就绪式 epoll server,在多连接、多 round-trip 下比吞吐与扩展。io-04 落地(co_await socket 层 + shard-per-core)后,这个负载才存在,故现在补。

## echo 吞吐 + shard-per-core 扩展(`bench/io/echo_bench.cc`)

**负载**:64 个连接,每个连接 64 次 echo round-trip(client send→server recv→server echo→client recv = 1 round-trip),payload 64B。连接按 owner 核轮询分片(shard-per-core),每连接的 client/server fiber 都 `spawn_on` 钉在其 owner 核。报 round-trip/秒。

**Linux io_uring(单跑中位,err% 1-3%,多次复现)**:

| 配置 | echo/s(批量后) | 对裸 epoll 比 | 判定 |
|---|---|---|---|
| HYLE shard-per-core 1 worker | **728K** | ~1.0x | ➖平/微赢裸 epoll |
| HYLE shard-per-core 2 workers | 1.10M | 1.51x | ✅赢 |
| HYLE shard-per-core 4 workers | 1.49M | 2.05x | ✅赢 |
| 裸 epoll 1 线程(就绪循环) | 729K | 1.0x | 基线 |

**关键转折(2026-07-03,submit 批量落地后)**:单 worker 从**曾经的 440K(输裸 epoll 0.61x)提到 728K(追平/微赢)**——1.65x 提升。机制铁证:`HYLE_IO_BENCH_STATS` 打出 **`avg_batch=13~61`**(每次 `io_uring_enter` 提交 13-61 个 SQE),对比批量前恒为 1.0。这直接砍掉了曾经主导的"每 op 一次 syscall"开销。**shard-per-core 扩展仍在**(1w→4w ~2x)。

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

## 未做(登记)

- **裸 liburing 对标**:证 HYLE 相对 liburing echo 的 0 封装开销。WSL 缺 `liburing-dev`;HYLE 本用裸 syscall,可后续在 bench 内手写等价 liburing echo 或装库补。
- **vs asio / libuv**:证不输成熟异步框架。
- **SQPOLL**:内核侧 submit,连批量 enter 都免;对延迟受限负载也有效。接口已预留(setup flag),未接。
- **park 合并**:worker 空闲仍走 parker 1ms park(非阻塞 reap);消这 1ms 延迟需 park 合并(碰防惊群,之前一版含 eventfd 唤醒方案,已随批量回退时一起撤,可复用思路重做)。
- Windows IOCP 的批量对应物(IOCP 无 submit 攒批概念,ReadFile/WriteFile 即时下发)+ n=15 统计版。
