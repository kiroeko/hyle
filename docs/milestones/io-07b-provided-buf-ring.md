# io-07b：provided buffer ring + multishot recv + co_await 流

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-07a（multishot accept）|
| **状态** | ✅ kBuffer 半成品补完（io_uring only）。真测 ProvidedBufRingMultishotRecv io_uring + TSan 净 |

> **范围**：补完 io-01 在 `Completion.kBuffer` 上预留的另一半——`register_buf_ring` + `submit_recv_multishot` + `buf_ring_addr`/`buf_ring_return`，内核选 buffer 免预投递 = 零拷贝 recv。**至此 multishot 全家（accept + recv）+ kMore/kBuffer 两 flag 全补完**。再上一层 `net_multishot.h` 的 co_await 流式 API。
>
> **2026-07-20 API 收口**：这些原语现为 `IoUringExtension` 的
> `register_buffer_ring/submit_recv_multishot/buffer_address/return_buffer`；
> `MultishotRecvStream` 也必须显式持有该扩展。非 io_uring 后端不再暴露
> 恒定失败的同名 `Proactor` 方法。
> **cloc**：+130 proactor + `net_multishot.h` 304（+ bench）。

## 为什么是它
代理/echo 服务器想要的读路径（liburing proxy.c recv_multishot）：一次 `submit_recv_multishot` 持续投递每个到达的段，内核每次从注册的 buffer ring 选一个 buffer（`kBuffer` + buffer_id），**无 caller buffer、无 per-recv submit**——普通 `RecvOp` 每次调用付的两笔开销都省掉。这是 io-01 `kBuffer` flag 的兑现。

## 核心决策
- **裸 io_uring register**：`register_buf_ring` 自写 `IORING_REGISTER_PBUF_RING` + `sys_io_uring_register` wrapper；`IOU_PBUF_RING_MMAP` 内核分配环、mmap 于 `IORING_OFF_PBUF_RING | bgid<<shift`；自持 backing pool。`submit_recv_multishot` 用 `IORING_RECV_MULTISHOT` ioprio + `IOSQE_BUFFER_SELECT` + buf_group。`buf_ring_addr`（读字节）/ `buf_ring_return`（add + release-store tail 还内核）。
- **IOCP/epoll 无对应 = 诚实 return false**，net 侧回退预投递 recv。

## co_await 层（`net_multishot.h`）——两个硬点（同 net_timeout 族）
### ① 一次 arm、多 completion、一个 fiber
co_await 只 resume fiber 一次，但 multishot 是一个流。建模成 `MultishotRecvStream`，用 `co_await stream.next()` 驱动：第一次 next 才 arm，之后每个 completion 唤醒 fiber、读 buffer、再 co_await next **不重新提交**（multishot 内核侧仍 armed）。`kMore` CLEAR 是终结（peer 关 / error / ring 耗尽）→ next 返空终结 chunk。
- **单 token 多完成需完成队列**：一批 reap 可能落好几个 completion（内核有多段排队）在 fiber resume 消费第一个之前——单 result 字段会让 N+1 clobber N（那个让 6 字节数据段消失在后随 EOF 后的 bug）。故 slot 持一个小 `Pending` 队列（`kMaxPending=64`），`complete_multishot` push、`next()` pop。
- **只在 empty→non-empty 转移 wake**：fiber 一旦醒就 drain 整个队列（`await_ready` 队列非空时为 true，不重挂），故一批多完成只第一个 wake，恰**一次 wake_rendezvous** per suspend episode。

### ② 迟到 completion 生命周期（取消安全陷阱）
multishot 在终结（kMore-clear）CQE 前一直有在途 completion。fiber 早弃流（drop / 上层取消）时 proactor cancel 是异步的，在途 + 最终 -ECANCELED CQE 仍落。栈 slot 会 UAF。故 stream state **heap + refcount {kernel, fiber}**，与 `TimedSlot` 完全同形——但 kernel ref **不是每 completion 花**，而是**只在 kMore-CLEAR 终结 completion 花**（那才是内核真正用完 token 时）。`RecvBuf` 是非拥有 view，**析构自动 `buf_ring_return`**（RAII 防消费者忘记归还饿死 ring）。

### token tag
net_timeout 用 bit0=1 表 timed slot；multishot 用 `0b11`（bit0=1 让 io_context 的 is_timed_token 抓住，bit1=1 让 complete_timed 转发到 complete_multishot）。kCancelToken/kWakeToken 保持 bit1 clear，永不撞。

## 对抗式自审（正确性 / 定位 / TSan）
- **① 正确性**：ProvidedBufRingMultishotRecv（DGRAM socketpair，一次 arm 收 3 包、每包 kBuffer、buf_ring_addr 字节校验、归还复用）io_uring + TSan 净。
- **② 性能 / 定位「已就位待承重」**：正确性 / 不回归已验（vs one-shot 同核对称负载统计平局），**暂无真实消费者**。兑现下一步 = 拿它改 HTTP body 流式读 / proxy 转发。
- **④ TSan**：净。

## 交付物
- `include/hyle/io/net_multishot.h`（RecvBuf RAII + MultishotSlot + complete_multishot + MultishotRecvStream/NextAwaitable）
- `include/hyle/io/proactor.h` + `src/io/proactor.cc`（register_buf_ring / submit_recv_multishot / buf_ring_addr / buf_ring_return）
- `include/hyle/io/tcp.h`（`TcpStream::read_multishot(bgid, alloc)` 暴露）
- `tests/io/proactor_test.cc`

## 参考
- io-design「② 参考库审计真缺口」io-07b 节；io-01 的 `Completion::kBuffer` 接口预留。liburing proxy.c recv_multishot；tokio-uring Lifecycle。相关里程碑：io-07a（kMore/multishot accept）、io-06（heap slot + refcount 同族）、io-07c（registered files）。详见 [performance/benchmarks-io.md](../performance/benchmarks-io.md)。
