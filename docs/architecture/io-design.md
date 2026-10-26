# IO：Scheduler 上的完成式 I/O

IO 为文件、网络和协议提供 Scheduler-native 的异步操作。用户通过 RAII、typed Result 与
`co_await` 组合工作，不必为 HTTP/TLS 再引入另一套线程或事件循环。接口、用法与平台限制见
[IO 指南](../user/io.md)，共同进展与关闭规则见 [运行时总览](runtime-design.md)。

```text
TCP / UDP / Local / File / DNS / Signal / Watch
TLS / HTTP / HTTP2 / WebSocket / Transfer
                     |
                IoContext shard
                     |
      Windows IOCP / Linux io_uring 或 epoll / macOS kqueue
                     |
            completion -> Sched Waker
```

IOCP/io_uring 直接提供 completion；epoll/kqueue 把 readiness 转成相同的操作完成语义。
公开 API 不暴露 SQE/CQE、OVERLAPPED、fd readiness 或供用户驱动的第二事件循环。

## 内部组合与公开产品

| factor | 拥有的机制 | 留给上层 |
|---|---|---|
| `io_core` | context、buffer、socket/local/file、file-region 发送、DNS、signal、typed Linux extension | 协议与服务政策 |
| `io_tls` | provider-neutral config/context/session/stream、握手与 close-notify | HTTP 与连接业务 |
| `io_http` | HTTP/1 framing、head/body/connection 生命周期、Upgrade 交接 | listener、route、auth、MIME/Range/cache 政策 |
| `io_http2` | session/stream、HPACK/frame、flow control、RST/GOAWAY | transport runtime、gRPC/业务调度 |
| `io_transfer` | 有界 copy/download、resume、校验与原子发布 | 仓库认证、模型格式、archive、分片政策 |
| `io_watch` | direct/exact/tree 变化流、overflow/rescan、关闭 | debounce、重读、验证与新对象发布 |
| `io_websocket` | RFC 6455 握手、message/frame/control/close 与 limits | route/auth、schema、reconnect、rooms |

这些内部 factor 聚合为 `hyle::io`，不要求普通用户逐项配线。TLS 与 HTTP/2 的编译可用性仍取决于
所选特性与依赖，不能仅凭安装了公开头就认定可用；初始化/构建入口见 [构建架构](build-system.md)。
中央依赖边见 [HyleComponents.cmake](../../cmake/HyleComponents.cmake)，不在本页复制第二份表。

## Owner、借用与真实完成

socket/file 绑定一个 IoContext worker，操作按其 owner affinity 提交。连接可以有协议允许的一读
一写，但不能并发发起同方向的第二操作，也不能把 native handle release 当作跨 backend 迁移。
尤其 Windows socket 在 native lifetime 内仍关联原 IOCP，再次 adopt 不会解除这一事实。

buffer、取消 token 与 operation state 必须活到其合同允许的完成点。base、deadline 与 cancel
共用一个状态机，竞争唯一结果；取消不能跳过 kernel 仍持有的 iovec、参数块或 payload。错误与
partial bytes 可以同时存在，已经提交的字节不会因 Result 失败自动回滚。

上下文关闭应先停止新接纳，再请求取消/完成，继续收割并排空，最后销毁资源。析构只是防泄漏
后备，不替代异步 close 或协议 shutdown。公开错误保留 typed Status，不让用户解释后端负整数。

## 文件：异步能力不由 OS 名字推断

File 优先使用实际 backend 的原生异步能力，其它操作卸载到 IoContext 的 BlockingPool。direct /
non-buffered 是显式请求：不满足地址、offset、length alignment 时失败，不能偷偷改为 buffered。
普通同步 FS 与原子 replace 归 Platform，命名共享 storage 归 Mem，IO 组合异步 workflow。

文件 data 的 deadline/cancel 有明确限制：非原生异步 fallback 无法安全打断已进入 kernel 的
pread/pwrite，因此带逐操作预算时返回 Unimplemented。需要该保证的消费者应查询
`IoCapability::kAsyncFileData`，不能靠“运行在 Linux”猜测已取得 io_uring。无预算时仍可卸载并等
实际读写结束；不能返回后就让调用方回收仍在使用的 buffer。

DNS 的 getaddrinfo 是另一种合同：查询状态拥有后台工作所需存储，因此 deadline/cancel 可让调用方
先返回，后台自行完成与释放。不能把这项能力泛化为所有 borrowed-buffer 文件操作也能立即返回。
主机名连接用同一总预算进行有界错峰尝试，成功前后都排空 loser，避免遗留连接或无限并发。

FileRegion 是已打开文件中的 offset/length 值。file-to-TCP 要求双方共享 IoContext/owner，返回
partial progress：Windows 用 overlapped TransmitFile，Linux/macOS 用 nonblocking sendfile 和
writable wait；机制不适用时使用一次 64 KiB scratch 做 positional-read/write，空间 O(1)，但不
宣称 zero-copy。direct alignment 与短 EOF 的错误仍保留，不能为了快路径重开另一种文件。

## 文件变化是提示，不是无损日志

exact file 观察 parent，才能跨 inode replacement；direct directory 只观察一层；recursive tree
返回 root-relative path。Windows/macOS 使用原生 subtree stream，Linux tree 模式才建立有界的
O(directory count) watch set，普通 direct/exact 不为递归能力支付全树 metadata。

overflow、rename/topology 丢失或粗粒度事件要求显式 rescan，消费者重读 authoritative state。
watch 不承诺每次 write 都有一个精确事件，也不跟随 symlink/junction。解析、debounce、验证、
reload 与发布新配置/模型由上层 owner 承担，watcher 只提供有界变化流和取消/关闭。

## TLS、HTTP 与 Upgrade 的交接

TLS provider 类型不进入公开 API。长期 config/context 拥有证书与验证政策，session/stream 拥有
连接状态，ALPN、mTLS、session ticket 与 close-notify 有明确选项和错误。

plain 与 TLS 复用 HTTP framing。大 body 的默认形状是增量 reader/writer；有界 read-all 是显式
便利。writer 等上一块完成才取得下一块，不建立用户看不见的无界发送队列。file-body 复用同一
writer 状态机：plain 可委托 kernel route，TLS 使用有界 plaintext scratch；HEAD/bodyless 不触碰
文件，post-header 失败后不能把连接当成可复用的完整消息边界。

HTTP/2 借 nghttp2 的 frame/HPACK/state machine，不引入它的 transport 或应用 runtime。流控同时
受连接与 stream 窗口约束，RST/GOAWAY 和 body 完成是不同事件，不能因某个 stream 失败就错误
销毁仍可服务其它流的资源。

HTTP/1 Upgrade 是唯一的 transport ownership 交接点。server 先完成 request body 才发送 101，
client 先检查普通 response 再接管。成功时 HttpUpgradeStream 必须一起移动 parser 已预读 bytes：
head 与首帧可能同包到达，丢掉 prefix 就会破坏新协议。源 HTTP connection 与 body borrow 同时
失效，后续读取先排空 prefix。普通 HTTP 路径不为可选 Upgrade 增加长期状态。

WebSocket 在该 seam 上拥有 message 语义：text/binary 类型、跨帧 UTF-8、message limit、有界
fragmentation、ping/pong 与 close 握手。client 必须 mask，可复用有界 scratch；server 可用 vector
write。掩码/握手的私有密码机制不变成通用 crypto 产品。extension、Origin/auth、route、heartbeat
与 reconnect 属于上层，不因库支持 WebSocket 就获得完整服务框架。

## 可靠制品传输

Transfer 组合已有 file、HTTP/TLS 和 Platform FS，不复制协议。跨 suspend 的 plan/checkpoint/report
必须 owning，工作 buffer 不随制品大小增长。resume 必须验证 source identity/validator，不能盲信
已有 partial file；redirect、restart、keep-partial 与 cleanup 也要显式决定。

只有校验、flush/close 与原子 publication 成功后才替换旧目标。网络完成、文件可见和持久化不是
同一个条件。repo client、token storage、模型解析与 archive extraction 留给引擎或其它消费者。

## 性能与扩展边界

优先解释 syscall、copy、allocation、wake 与 contention 数量；批量收割完成并复用有限的 operation
state。readiness backend 需要 fd→pending 映射，IOCP/io_uring 不为统一外观额外做同样的查表。
公开 extension 必须能真实取得能力，例如 Linux 的 provided-buffer/multishot，而非在其它系统
暴露恒失败的伪 portable 入口。

第三方只提供成熟协议或 kernel 机制，Scheduler、allocator、资源退休和用户合同仍由 HYLE 拥有。
不同错误、取消、持久化与完成边界的 peer 不能直接按单个吞吐点值排名。IO 当前交付 C++ SDK，
没有单独的 C/FFI 或服务路由产品；推理 server 应在其上组合自己的接纳与应用协议。
