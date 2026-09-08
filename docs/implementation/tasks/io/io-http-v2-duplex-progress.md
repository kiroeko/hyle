# IO HTTP v2：协议 transport 双向进展修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen record（`REOPENED`）

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
- immediate decision：依法移除旧 FROZEN confirmation，先进入 `REOPENED`。在 `SCOPED` 前
  不修改产品；随后只冻结一个有界 transport progress primitive、回归矩阵与受影响回放。

## Historical boundary

`io-http-v1` 的 HTTP/1 能力、三平台/三交付和 P1 receipt 仍是历史证据，不因重开而改写。
新一轮不得借机重做 parser、增加 server framework、公开 native provider handle，或把 HTTP/2
frame/flow-control 语义下沉到 `io_http`。
