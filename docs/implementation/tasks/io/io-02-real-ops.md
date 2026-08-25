# io-02：io_uring / IOCP 真 IO 操作（read/write）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-01（收割接缝）|
| **状态** | ✅ 完成（两后端 read/write 真 op 全通：Windows 8/8 + Linux 11/11 + TSan 净）|

> **范围**：在 io-01 收割接缝上填第一批**真 IO op**——`submit_read` / `submit_write`。io_uring 填 `IORING_OP_READ/WRITE` SQE；IOCP 用 `ReadFile/WriteFile` + per-op `OVERLAPPED`。socket / accept / connect op 留 io-04。
> **cloc**：+80（io_uring）+ ~200（IOCP）。

## 为什么是它
io-01 只证明了「token 进得去、完成收得回」（NOP 心跳）。io-02 让接缝搬真字节：一个 `submit_read`/`submit_write` 携带 fd/buf/len/offset + `user_data` token，完成时 `Completion.result` 是字节数（>=0）或负 errno。这是 walking-skeleton 从「空跑」到「真 IO」的一步。

## 核心决策
- **fd 用 `intptr_t`**：同一签名既载 POSIX `int` fd 又载 Windows `HANDLE`（指针宽），头里不需要 per-OS 类型。
- **submit 失败 vs IO 错误分离**：`submit_*` 返 false 只代表「排不进队」（ring 满 / 坏 handle）；真正的 IO 错误（EPIPE、短读）走 reap 时的**负 `Completion.result`**——io_uring CQE `res` 约定，IOCP 把 `GetLastError` 映射上去。
- **buf 生命周期契约**：`buf` 调用方所有，**必须活到匹配的完成到达**——内核直接往里写（io_uring 真零拷贝；IOCP OVERLAPPED read）。

## IOCP 的 OVERLAPPED 自持（本步的硬点）
IOCP 每个 op 需要一个 `OVERLAPPED`，其生命周期必须跨过 async 完成。proactor 用**侵入式在途链表自持** OVERLAPPED：
- OVERLAPPED 是 `IoOp` 的**首成员**，reap 时把 `lpOverlapped` 直接 cast 回 `IoOp`，`GetOverlappedResult` 取字节/errno。
- `post()` 保持 `lpOverlapped == nullptr`，让 reap 区分**合成唤醒** vs **真 op**。
- dtor 遇在途 op：先 `CancelIoEx` 再排空端口回收再释放（防关机 UAF）。

**「IOCP 需与 net/ 一起定形」是过度谨慎**——OVERLAPPED 纯 IOCP 内部细节由 proactor 自持，net/ 不可见，故此处即可落地。

## 对抗式自审（正确性 / TSan）
- **① 正确性**：io_uring WSL 真测 write→read 回读字节校验 + bad-fd→-EBADF；IOCP Windows 真测文件往返 + bad-handle 同步失败（submit false）+ 在途 op 析构干净。
- **④ TSan**：双平台 TSan 净。
- **性能对标**：同 io-01——单 op microbench 无意义，等 io-04 真并发负载。

## 交付物
- `include/hyle/io/proactor.h`（`submit_read` / `submit_write` 签名 + buf 生命周期契约注释）
- `src/io/proactor.cc`（io_uring SQE 填充 + IOCP ReadFile/WriteFile + OVERLAPPED 自持在途链表 + dtor CancelIoEx 排空）
- `tests/io/proactor_test.cc`

## 参考
- io-design「已落地(io-01/02 实现纪要)」§后端进度（诚实分层）。liburing v2.15（SQE 字段）；IOCP OVERLAPPED 模型。相关里程碑：io-01（收割接缝）、io-03（epoll 降级）、io-04（socket op）。
