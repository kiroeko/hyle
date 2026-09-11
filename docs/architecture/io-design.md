# IO 架构：Scheduler 上的完成式 I/O

本页只说明当前长期设计。实际 C++ 用法见 [IO 用户指南](../user/io.md)，组件状态和当前性能证据见
[`module-status.json`](../implementation/module-status.json)。实现过程、旧测试数字和被否决路线由当前
record 或 Git 历史追溯，不复制到这里。

## 设计目标

HYLE IO 为推理服务和普通系统程序提供一套 scheduler-native C++23 I/O：socket、文件、文件变化流、
DNS、signal、TLS、HTTP/1.1、HTTP/2 与可靠制品传输。应用不应手写 socket/event loop，也不应为了使用协议层再引入
一套线程或 coroutine runtime。

WebSocket 已进入[基础能力扩展范围](foundation-capability-expansion.md)，但在自己的 closure round 完成前
不属于当前公开 API。file-region streaming/HTTP file body、TLS credential source 与 recursive tree watch
已经按各自 factor 交付；终态仍以 registry 为准。file-region streaming 归 `io_core + io_http`，而不是
制品发布语义不同的 `io_transfer`。

```text
Tcp/Udp/Local/File/DNS/Signal/TLS/HTTP/Transfer
              │ co_await + deadline/cancel + RAII
              ▼
IoContext（每 Scheduler worker 一个 owner completion engine）
              │
              ├─ Windows: IOCP
              ├─ Linux: io_uring，运行时不可用时 epoll
              └─ macOS: kqueue
                         │
                         └─ completion -> sched::Waker
```

IOCP/io_uring 原生是 completion 模型；epoll/kqueue 在 private backend 内把 readiness 转成相同的
completion 语义。公开层不暴露 fd readiness、OVERLAPPED、SQE/CQE 或 backend event loop。

## 组件边界

| factor | 责任 | 不拥有 |
|---|---|---|
| `io_core` | context、operation、buffer、TCP/UDP/local、file、file-region→TCP、DNS、signal、backend 与 Linux typed extension | TLS/HTTP 状态机、服务路由 |
| `io_tls` | provider-neutral config/context/session/stream 与 handshake/read/write/shutdown | socket runtime、HTTP |
| `io_http` | HTTP/1 framing、head/body/file-body lifecycle、client/server connection | listener、path/MIME/Range policy、routing、auth、batching |
| `io_http2` | HTTP/2 session/stream、HPACK/frame provider、flow control、RST/GOAWAY | transport event loop、gRPC policy |
| `io_transfer` | local/HTTP(S) copy/download、resume、verify、temporary file 与 atomic publication | repository/model schema、credentials、archive |
| `io_watch` | bounded exact/direct/tree change stream、root-relative path、coalescing、overflow/rescan、cancel/close | snapshot、debounce、content diff、TLS/config/model/shader reload policy |

六个 factor 对用户聚合为 `hyle::io` 产品并共享 `hyle::io` namespace；拆分只服务内部无环依赖、独立
构建和交付。中央依赖图是 [`HyleComponents.cmake`](../../cmake/HyleComponents.cmake)，本页不复制边。
未来 `io_websocket` 只有在登记中央 identity 和 registry 后才成为第七个 factor；架构决策本身不预造 target。

## Ownership 与进度

- 一个 socket/file/session 由一个 worker 拥有，提交、完成和销毁在 owner 上发生；跨 worker 只通过
  明确迁移或消息，不并发操作同一 stream。
- buffer、取消 token 和 callback state 必须活到 await 完成。析构不是取消协议；shutdown 先停止
  admission，再 cancel，最后 drain completion。
- deadline 使用 Time 的 absolute monotonic 形状；取消使用 Sched 的 token；失败统一返回 Error 的
  typed `Status/Result`，backend 原生错误在边界转换。
- base、deadline 与 cancellation 路线共享一个 operation state，不为便利复制三套 I/O 状态机。
- partial read/write/progress 与错误可以同时有意义；结果不能假装事务性回滚已经完成的字节。
- context、stream 和协议 driver 的内部状态使用 Mem/Container；IO 不复制 allocator、hash table、
  timer queue 或 filesystem 原语。

## 文件与平台能力

普通 `File` 根据 backend capability 选择原生异步或 `IoContext` 的 blocking offload；用户不按 OS 名字
猜路线。direct/non-buffered I/O 是显式请求，alignment 合同不可满足时诚实失败，不静默变成 buffered。
named shared storage 属于 Mem，原子 replace/同步文件机制属于 Platform FS，IO 只组合异步 workflow。

`FileRegion` 是 caller-owned open file 内的稳定 offset/length value。`send_file_region` 要求 File 与
TcpStream 共享 IoContext/owner，返回 partial progress；Windows 走 overlapped `TransmitFile`，Linux/macOS
走 nonblocking `sendfile` + owner completion engine 的 writable wait。机制不适用时只有一个 64 KiB
positional-read/write fallback，故空间 O(1)，但不冒充 zero-copy。deadline/cancel、short EOF、peer close
和 direct alignment 都保留 typed 语义；默认 File/TcpStream layout 与普通 read/write 路线不增加 metadata。

文件变化使用 parent/root 的原生 stream（Windows IOCP + `ReadDirectoryChangesW`、Linux inotify、macOS
FSEvents），对外统一为 bounded borrowed batch。exact file 也观察 parent，才能跨 inode atomic replace；
direct directory 与 recursive tree 是两个显式 target 语义，后者返回 root-relative path。Windows/macOS
使用原生 subtree stream；Linux 只在选择 tree 时维护有界 O(directory count) watch set，topology/loss 后
发 rescan 并恢复覆盖。普通 direct/exact 路径不建立递归 metadata。overflow 或粗粒度事件只承诺显式
rescan，不伪装成完整 event log。watcher 复用 Scheduler 的 owner、取消、deadline 和唤醒，不建立 polling
loop 或永久私有线程，也不跟随 symlink/junction。reload/验证/发布策略属于调用方或计划中的独立
`hyle-service`，不能反向进入 IO factor。

Linux 专属的 provided-buffer/multishot receive 经成功取得的 `IoUringExtension` 使用；portable
one-shot 永远存在。backend-specific 快腿必须用 typed capability/extension 显式取得，不能在其他平台
暴露第一次调用才恒定失败的伪 portable API。

## TLS 与 HTTP

TLS provider 类型不进入公开 API。config/context 拥有长期 provider policy，stream/session 拥有连接
状态；证书验证、ALPN、mTLS、session ticket 与 close-notify 都使用明确选项和 typed failure。
plain TCP 与已经握手的 TLS 通过同一 transport abstraction 进入 HTTP framing，避免复制协议解析器。

HTTP API 只负责 connection、framing、body 和 flow-control lifecycle。增量 reader/writer 是默认大消息
形状；有界 `read_all` 是显式便利。writer 每次 await 完成后才拉下一块，不能建立不可见的无界发送
队列。HTTP/1 file-body overload 复用同一个 writer 状态机：plain transport 委托 io_core 的 kernel route，
TLS 使用一次 64 KiB plaintext scratch；HEAD/bodyless status 不触碰 file，任何 post-header failure 都废弃
connection reuse。path traversal、MIME、Range/conditional/cache 是服务 policy，不进入这个原子组合。
HTTP/2 可借 nghttp2 处理 HPACK/frame/state machine，但 transport、Scheduler、allocator 和 application
policy 仍由 HYLE/应用拥有。

listener ownership、routing、middleware、authentication、request admission、dynamic batching、模型
schema、graceful service policy 和 metrics policy 都属于上层 server/推理引擎；基础库不伪装成框架。

## 可靠制品传输

Transfer 是对现有 file/HTTP/TLS/Platform FS 的有界组合，不复制协议。它必须：

- 把跨 suspend 的 plan/checkpoint/report 状态变成 owning value；
- 对制品大小保持 O(1) 工作 buffer；
- 用 source identity/validator 验证 resume，不能盲信 partial file；
- 在校验、flush/close 和 atomic publication 成功前保留旧目标；
- 明确 redirect、restart、keep-partial、cancel 与 cleanup policy。

repository client、token storage、archive extraction、模型解析和分片调度留给推理引擎。

## 性能与实现约束

- 热路径目标先看 syscall、copy、allocation、wake 和 contention 数量；不同持久化、错误、完成或
  lifecycle 合同不能直接排名。
- completion 应批量收割；owner-local free list/operation state 可以复用，但不得改变外部 lifetime。
- readiness backend 才需要 fd→pending 映射；IOCP/io_uring 不为统一外观增加无用 hash lookup。
- provider/library 只借其成熟机制，不引入第二 scheduler。依赖、版本与许可证由 build system 一处管理。
- 当前只交付 C++ source/static/shared；没有 C API 或 Native FFI。未来跨语言 server 使用 HTTP/协议，
  只有具名 foreign runtime 需求才按 [Native FFI 策略](native-ffi-policy.md)设计窄边界。

任何新 portable 能力都必须在支持平台真实实现或有语义等价 fallback；`Unimplemented` 可以诚实描述
运行时 capability 缺失，但恒定失败声明不能冒充功能完整。
