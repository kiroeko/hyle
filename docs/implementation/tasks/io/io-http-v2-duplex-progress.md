# IO HTTP v2：协议 transport 双向进展修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round ID：`io_http` / IO maintainers /
  `io-http-v2-duplex-progress`；task kind：`REOPEN`；baseline：
  `720c055b4e4bcf98556641acb77f82ad689c10cc`；
- trigger：`IOHTTP-REOPEN-H2-DUPLEX-01`。冻结在
  [`io-http2-p1-v1`](../../../../bench/contracts/io-http2-p1-v1.json) 的同合同
  4 KiB socket-buffer、双向 HTTP/2 flow-control holdout，在
  `HYLE_IO_FORCE_EPOLL=1 HYLE_BENCH_SMOKE=1` 下可重复停止进展；系统调用轨迹证明
  双端各有待写数据时，现有协议驱动只等待可写而没有继续吸收已到达输入；
- affected surface：`HttpTransport` 给上层协议使用的 transport-neutral 组合缝，以及它对
  `io_core`/可选 `io_tls` 的即时进展表达。HTTP/1 framing、message/body API、parser、路由和
  应用策略不在受影响范围；
- evidence：候选 `3bd9c9391efcacf63185c5f1c73d4d9351f51fdf` 加冻结 fixture
  `720c055b4e4bcf98556641acb77f82ad689c10cc`；W1-WSL 默认 io_uring 正确完成，强制 epoll
  在 holdout 稳定挂起，末端轨迹为两端 `EPOLLOUT|EPOLLONESHOT`，而对应 peer receive queue
  已有未消费字节。这满足旧 `io_http` 登记的“HTTP2 consumer 复现 backpressure/teardown
  defect” reopen trigger；
- scope：为协议因子增加一个 internal、transport-neutral 的 immediate-read attempt。它一次最多
  吸收一个已经到达的 bounded chunk，以独立 `would_block` 状态区分“现在无数据”和 EOF/error；
  不等待、不分配、不吞掉 deadline/cancel，也不公开 native/provider handle。`io_http2` 在准备阻塞
  写入前用它维持 full-duplex progress；plain TCP owner 在 `io_core`，TLS provider owner 若由确定性
  复现证明受影响则依法单独重开；
- non-goals：改变 HTTP/1 parser/framing/message/body API，给 ordinary 用户公开 reactor syscall，
  新建后台 scheduler/线程、无界读写队列，修改 nghttp2/flow-control 语义，或增加应用 server 策略；
- tier / gates：P0 先证明 nonblocking、EOF/error、单 reader/owner 与 no-allocation；correctness 固定覆盖
  4 KiB socket buffer + 双向 flow-control、默认 io_uring 与强制 epoll，并补 h2c/TLS 分类。P1 只回放
  已冻结 HTTP/1 两条 workload 与 HTTP/2 normal/holdout，不改变 peer、socket/window/body/order；
- budget：3 engineering hours；只允许一次 transport progress 产品机制纠正和一次受影响 provider
  纠正；性能最多一次机制检查。超过后按真实剩余问题 CLOSED 或 BLOCKED；
- machines / delivery：W1-WIN、W1-WSL、M1-MAC；source/static/shared。sanitizer 只跑受影响的
  io_core/io_tls/io_http/io_http2 exact factors；
- strong references：Boost.Asio、libuv/nginx 的 optimistic nonblocking receive + readiness fallback
  只作为机制参照；nghttp2 仍是协议 authority。不能把直接 busy polling 或 blocking raw loop 当成
  HYLE scheduler 路线；
- stop：确定性双向进展与旧 HTTP/1 语义/P1 回放全部通过后关闭；若 TLS 需要无法在本轮边界内表达的
  provider engine 重构，`io_http` 不冒充完成，登记精确 io_tls blocker/trigger。

## Decision invariants

- immediate attempt 只在 transport owner worker、且没有另一 read operation 在途时执行；上层仍保持
  每方向最多一项 native operation；
- `would_block` 不是错误，EOF 与部分 bytes + terminal status 不得丢失；任何已吸收 bytes 必须在本次
  调用返回前进入协议 parser；
- optimistic path 与普通 awaitable path 使用同一 native error 映射；Windows 不假设 adopted socket
  已处于 nonblocking mode；
- HTTP/2 只调用组合缝，不复制 POSIX/WinSock/OpenSSL 分支。API/layout/普通 HTTP/1 路线不因 benchmark
  增加 metadata。

## Historical boundary

`io-http-v1` 的 HTTP/1 能力、三平台/三交付和 P1 receipt 仍是历史证据，不因重开而改写。
新一轮不得借机重做 parser、增加 server framework、公开 native provider handle，或把 HTTP/2
frame/flow-control 语义下沉到 `io_http`。
