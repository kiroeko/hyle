# IO 任务索引

当前多轮导航见 [`io-v1-rebuild-epic.md`](io-v1-rebuild-epic.md)。模块状态只见
[`module-status.json`](../../module-status.json)；本目录其他记录是旧实现阶段的事实输入，不是当前授权。

## 当前规划

- [`io-v1-rebuild-epic.md`](io-v1-rebuild-epic.md)：按 IO core、TLS、HTTP/1、HTTP/2、transfer 五个内部
  因子依次建立能力超集、自然 API、依赖和验证。

## 历史实现记录

- [`io-01-proactor-seam.md`](io-01-proactor-seam.md)
- [`io-02-real-ops.md`](io-02-real-ops.md)
- [`io-03-epoll-kqueue-backend.md`](io-03-epoll-kqueue-backend.md)
- [`io-04-net-coawait.md`](io-04-net-coawait.md)
- [`io-05-file-io.md`](io-05-file-io.md)
- [`io-06-timed-io.md`](io-06-timed-io.md)
- [`io-07a-multishot-accept.md`](io-07a-multishot-accept.md)
- [`io-07b-provided-buf-ring.md`](io-07b-provided-buf-ring.md)
- [`io-07c-registered-files.md`](io-07c-registered-files.md)
- [`io-08-tls.md`](io-08-tls.md)
- [`io-09-dns-signal.md`](io-09-dns-signal.md)
- [`io-10-http1.md`](io-10-http1.md)
- [`io-11-http2.md`](io-11-http2.md)
- [`io-capability-fill-list.md`](io-capability-fill-list.md)
- [`io-module-closure.md`](io-module-closure.md)

这些记录中的 C ABI、阶段状态、性能结果和“下一步”只代表当时事实；本轮采用 C++ source/static/shared
产品与 Native FFI demand gate，不能直接继承旧结论。
