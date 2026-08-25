# io-01：proactor 收割接缝（IOCP + io_uring 双后端收割层）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | sched-04（Waker completion 脊）|
| **状态** | ✅ 完成（Win IOCP + Linux io_uring 双后端真测 4/4 + WSL TSan 净）|

> **范围**：io-design「模块形状」的 `proactor.h` 那一层——平台无关的**完成式收割接缝**。per-worker `Proactor`，`reap` 批量收割 completion、把 token 交回 sched wake，`post` 注入合成唤醒。**两后端的收割层都在此**：IOCP 与 io_uring。**这是收割接缝，不含真 IO op**——submit read/write/accept 是 io-02。
> **cloc**：生产 90（`proactor.h`）+ 245（后端 `proactor.cc`）+ 测试 130（`proactor_test.cc`）。

## 为什么是它
io/ 与 sched/ 的物理接缝就在这里。sched-design §3.2/§3.3 早已论证：一个 async op 需要两样东西回来——**wake**（把 fiber 重新注入）和 **result**（字节数 / 负 errno）。proactor 是承载这个映射的最薄层：每个后端都带一个 64 位 `user_data` 随提交的 op 走（io_uring SQE user_data / IOCP completion key + OVERLAPPED），完成时原样回来。`sched/` 拥有 Waker，`io/` 只搬 token。这个映射在写接口之前先在真机上验证过，是 io 支柱的地基。

## 核心决策
- **接口先研究 SOTA 再定形，不挡任何一个机制**：`Completion{user_data,result,flags,buffer_id}` 的 `flags` 不是装饰——`kMore`（multishot 一提交多完成、token 不早回收）+ `kBuffer`（零拷贝 provided buffer id）从第一天就在接口里，照 liburing v2.15 核对。一个「1 提交 = 1 完成」的抽象会腐蚀 multishot，所以这两个 flag 必须原生存在（真正补完是 io-07a/07b）。
- **API 宏放方法不放类**：后端持 OS handle / std 成员，方法级导出把 layout 留在后端侧、接缝干净（同 Parker/Scheduler）。
- **per-worker，不跨线程加锁**：每个 worker 独占一个 Proactor（自己的 IOCP port / io_uring ring），在自己线程 submit + reap——per-core 哲学，与 run-queue 同构。跨线程提交是后端在它出现的地方处理的事，不靠锁这个对象。

## 收割机制（两后端）
- **`reap(out, max, timeout_ns)`**：批量取完成写进调用方数组，`timeout_ns=0` 轮询即返、`UINT64_MAX` 阻塞至少一个完成。这是 `GetQueuedCompletionStatusEx` / io_uring cq peek 收在一个签名后。
- **io_uring**：裸 syscall setup / mmap ring / reap-CQ / post-NOP，**连 liburing 都不链**——只用 3 个 io_uring syscall + `linux/io_uring.h` uapi 头 + 自写内联环算术，barrier 精确复刻 liburing v2.15 的 fast-path 点。零第三方链依赖。
- **IOCP**：`GetQueuedCompletionStatusEx` 收割 + `PostQueuedCompletionStatus` 注入。
- **`post(user_data, result)`**：手动注入一个合成完成——驱动无真 fd 的测试，也（io-04）为同步完成的 op 合成一个 completion。io_uring 上填 NOP SQE、单生产者故须在 ring owner 线程调。

## 对抗式自审（正确性 / 残留 / 性能对标缺口）
- **① 正确性**：4 测双平台绿——两后端各自心跳验证 reap / post 环路 + WSL TSan 净。
- **⑤ 残留（诚实登记）**：这是**收割接缝**，真 IO op（submit read/write/accept）明确留 io-02；epoll/kqueue 后端留 io-03。
- **② 性能对标缺口（显式登记）**：io-01/02/03 目前只有功能正确性测试，无性能对标——孤立测单个 read/write 是在测内核 io_uring/OS 而非 HYLE 薄封装，无意义。真正的对标（HYLE echo vs 裸 liburing echo 证 0 封装开销 / vs epoll 证降级代价 / vs asio·libuv）**必须等 io-04 net 有真实并发连接负载**才有意义，届时补。「方向对」当前靠：接口按 liburing v2.15 SOTA 机制先研究再定形 + 借的就是各平台 proactor SOTA 本身（io_uring/IOCP），非自造。

## 交付物
- `include/hyle/io/proactor.h`（`Proactor{init/reap/post/wake}` + `Completion{user_data,result,flags,buffer_id}` + `kMore/kBuffer` flag）
- `src/io/proactor.cc`（IOCP 后端 + io_uring 裸 syscall 后端，pimpl `Impl`）
- `tests/io/proactor_test.cc`

## 参考
- io-design「编程模型：proactor（完成式）」「模块形状」「已落地(io-01/02 实现纪要)」；sched-design §3.2/§3.3（completion→fiber 物理机制）。liburing v2.15（SOTA 机制蓝本 + barrier 复刻，但不链）；IOCP（OS 原生）。相关里程碑：sched-04（Waker 脊）、io-02（真 op）、io-03（epoll/kqueue）。
