# IO stub / no-op 与能力补全清单

> 审计日期：2026-07-20。范围：`include/hyle/io/`、`include/hyle/c/io.h`、
> `src/io/`、`tests/io/` 与 IO guide。benchmark 在不支持平台打印 `SKIP` 的
> stub main 不属于库能力。

## 原则

HYLE 不接受用 stub 假装功能已经完成：

```text
普通用户合理需要的产品能力
  └─ 每个支持平台必须有真实实现、等价模拟或 blocking offload

纯后端加速机制（例如 io_uring fixed files）
  └─ 不属于 portable 必备面；通过 capability/extension 显式选择

可选构建组件（例如 HTTP/2）
  └─ availability probe + StatusCode::Unimplemented
     只表示“这个构建没带该组件”，绝不能冒充功能成功

错误调用
  └─ error 系统报告 InvalidArgument/Unimplemented 等，属于契约防线
```

`StatusCode::Unimplemented` 是“这个实现/平台不支持调用者选择的操作”的错误
提示，不是 stub 的许可证，更不是一种实现策略。它不能成为本来应该实现的产品
路径。`HYLE_NOT_IMPLEMENTED()` 是 fatal/unreachable 诊断宏，也不应用在可恢复
的 capability 分支；IO 产品代码应返回 Status。本轮正是因为旧 `Proactor` 把
portable 语义、后端加速和可选组件混在同一个 `bool` 面里，才出现“编译通过、
调用恒定失败”的假完成感；该混层现已拆除。

## 审计结论

机械扫描和逐项读实现后，当前有：

- **原有 4 组公开契约假成功已于 2026-07-20 修复**：readiness 后端保留
  socket flags（含 OOB readiness selector）、四后端 one-shot accept 回填
  peer address、非原生后端拒绝非零 fallocate mode、Windows shared-memory
  unlink 返回 typed Unimplemented。
- **11 个 io_uring 专属方法已移出 portable `Proactor`**：只有成功取得
  `IoUringExtension` 才能调用；C ABI 的扩展获取在其他后端返回 owning
  `HYLE_STATUS_UNIMPLEMENTED`，原有 `void` release no-op 已删除。
- **伪 portable multishot facade 已删除**：普通跨平台连续接收就是循环
  one-shot `read/submit_recv`（天然逐块 backpressure）；只有“一次提交多完成 +
  内核选 registered buffer”保留为 io_uring 扩展，不伪造同名慢腿。
- **Linux metadata 已补真实 `IORING_OP_STATX`**：稳定 native storage 活到
  CQE，reap 前转换为 C/C++ 共用 POD；epoll/kqueue/IOCP 的 C++ `File` facade
  继续经 blocking pool 卸载。
- **HTTP/2 feature-off stub** 是显式可选组件：`available()==0`，其余调用设置
  `last_error` 后失败。它不报告成功，属于允许保留的 ABI 错误腿。

因此 IO 的主干能运行，但 capability closure 尚未完成，不能标成“没有已知
行为缺口”的预发布冻结。

## 后端能力矩阵

| 能力 | IOCP | io_uring | epoll | kqueue | 判定 |
|---|---|---|---|---|---|
| TCP/UDP one-shot | 原生完成 | 原生完成 | readiness 模拟 | readiness 模拟 | 完整 |
| `recv/send(flags)` | 支持 | 支持 | `recv/send` + `EPOLLPRI` | `recv/send` + `EVFILT_EXCEPT` | 完整 |
| one-shot accept peer address | `getpeername` 回填 | 支持 | `accept4` 回填 | `accept` 回填 | 完整 |
| multishot accept | AcceptEx 重挂 | 原生 | 重挂模拟 | 重挂模拟 | 完整 |
| 循环 buffered receive | one-shot 循环 | one-shot 循环 | one-shot 循环 | one-shot 循环 | 完整 |
| provided-buffer multishot recv | 无扩展 | 原生扩展 | 无扩展 | 无扩展 | 后端扩展 |
| fixed files / direct accept | 无扩展 | 原生扩展 | 无扩展 | 无扩展 | 后端扩展 |
| `register_handle` | 真绑定 IOCP | 成功 identity | 成功 identity | 成功 identity | 合理 |
| async file data | 原生 | 原生 | C++ facade offload | C++ facade offload | 高层完整 |
| async file metadata | File facade offload | 原生 STATX | File facade offload | File facade offload | 完整 |
| `fallocate(mode=0)` | 等价扩文件 | 原生 | `posix_fallocate` | `F_PREALLOCATE` | 可用 |
| `fallocate(mode!=0)` | 明确失败 | 原生 extension | 明确失败 | 明确失败 | 完整 |
| shared-memory unlink | typed Unimplemented | `shm_unlink` | `shm_unlink` | `shm_unlink` | 完整 |
| HTTP/2 disabled | 显式 unavailable | 同左 | 同左 | 同左 | 合格 error 腿 |

`register_handle` 在 POSIX 返回成功不是假实现：POSIX fd 本来无需预绑定，调用后
“该 fd 可用于这个 Proactor”的后置条件已经成立。`datasync` 在 Windows/macOS
执行更强的完整 flush，也不是 no-op。

## P0：必须修复的公开契约

### IO-CAP-01：readiness 后端保留 socket flags ✅

`submit_recv/submit_send` 已公开 flags。IOCP/io_uring 传给 OS，epoll/kqueue
却降成 `read/write`，静默按 flags=0 执行。

`PendingOp` 已保存 flags，readiness `perform` 使用
`recv/send(..., flags)`。OOB 还要求 selector 对齐语义：epoll 监听
`EPOLLPRI`、kqueue 监听 `EVFILT_EXCEPT`，不能只改最终 syscall。TCP urgent
data 回归分别独立证明 send flag 与 recv flag；Windows、io_uring、
forced-epoll、M4 kqueue 真跑通过。

### IO-CAP-02：one-shot accept 填写 peer address ✅

C++ 头和 C ABI 都承诺填写 `address/address_length`，但 IOCP、epoll、kqueue
忽略参数。

- IOCP 用 `GetAcceptExSockaddrs` 或 accepted socket 的 `getpeername`。
- epoll/kqueue 将 address/length 存进 pending op，执行 accept 时传入。
- IPv4、IPv6、null-address 都要跨后端测试 family、port、length。

multishot accept 因一个 submit 对应多个连接，没有独立 caller buffer，可以继续
明确要求调用方对 accepted fd 使用 `getpeername`；不能反过来削弱 one-shot
契约。

实现与 IPv4 family/port 回归已在 Windows、io_uring、forced-epoll、M4
kqueue 真跑；null-address 仍由既有网络测试覆盖。IPv6 专项留在后续网络能力
补厚，不再是“参数被忽略”的 stub。

### IO-CAP-03：非零 allocation mode 不得静默成功 ✅

只有 io_uring 真正透传 fallocate mode；其余后端和 C++ blocking fallback
按 mode 0 执行后返回成功。portable API 应只保证 mode 0；其他 mode 若后端
不能实现，必须返回 `StatusCode::Unimplemented`，不能改变语义后报成功。Linux 原生
mode 放入 extension capability。

现状：io_uring 继续原样透传原生 mode；IOCP/epoll/kqueue 与 C++ blocking
offload 明确返回负 unsupported，跨平台 File 回归证明不会改变文件后报成功。

### IO-CAP-04：Windows shared-memory unlink 不得成功 no-op ✅

POSIX unlink 会立即移除名字，现有 mapping 继续存活；Windows named mapping
没有等价操作，名字要等最后 handle 关闭才消失。当前 Windows 只检查名字非空
便返回 true，会误导调用方。

在没有真实等价实现前，C++ 返回 `StatusCode::Unimplemented`，C ABI 返回对应
稳定状态码并提供准确错误上下文。guide 应把“最后 handle 关闭后自动消失”写成
平台生命周期差异，而不是成功 unlink。

现状：C++ 返回 `Status`；C ABI 返回 owning `hyle_status`，Windows 回归检查
`HYLE_STATUS_UNIMPLEMENTED`，POSIX 检查 OK。旧 bool ABI 已在开发期直接删除。

## P1：消灭通用 API 中的恒定 stub

### IO-CAP-05：增加 capability POD 与 typed extension status ✅

旧 capability 只有 backend kind 和两个 file bool；失败又只有
“unsupported or registration failed”。需要 C/C++ 同源的 capability bitset，
至少覆盖：

- native/emulated multishot accept；
- provided buffers；
- fixed files；
- direct accept；
- async file data/metadata；
- HTTP/2、TLS；
- 后续 vectored IO、zero-copy send。

现状：C/C++ 共用 `hyle_io_capabilities` POD；backend、native/emulated
multishot accept、provided buffer、fixed file、direct accept、异步
data/metadata、TLS/HTTP2 均可观察。`hyle_io_uring_extension_get` 用 owning
`hyle_status` 区分 InvalidArgument/Unimplemented。低层 submit 的
queue-full/OS-error 细分仍沿用 bool + completion/last_error，属于后续 C ABI
错误厚度优化，不再造成“平台不支持却像普通提交失败”的歧义。

### IO-CAP-06：收窄连续 receive 契约，不伪造 multishot ✅

旧 `TcpStream::read_multishot()` 看似普通 facade，实际非 io_uring/旧内核上首个
`next()` 就返回 submit failure；头文件曾写“net fallback to pre-posted recv”，
实现并不存在。

最终形状：

```text
TcpStream::read() 循环                    portable 产品能力
  ├─ IOCP / io_uring 原生 one-shot completion
  └─ epoll / kqueue readiness→completion 模拟

IoUringExtension + MultishotRecvStream    Linux 专属零提交/选 buffer 快腿
```

普通 `read()` 已统一 chunk、EOF、deadline/cancel 与逐次消费 backpressure；
再包一层 one-shot 循环不会增加能力。旧 `TcpStream::read_multishot(bgid,alloc)`
改为必须显式传入成功取得的 `IoUringExtension`，`MultishotRecvStream` 自身也
只接受该 handle。非 io_uring 后端不再有“首次 next 才失败”的入口。

### IO-CAP-07：11 个 io_uring 方法移出 portable Proactor ✅

旧版非 io_uring 后端曾以 stub 定义下列方法：

1. `register_buf_ring`
2. `register_file_table`
3. `assign_fixed_slot`
4. `release_fixed_slot`
5. `submit_recv_fixed`
6. `submit_send_fixed`
7. `submit_accept_direct`
8. `submit_close_direct`
9. `submit_recv_multishot`
10. `buf_ring_addr`
11. `buf_ring_return`

buffered receive 按 IO-CAP-06 做 portable 产品腿；fixed file、direct accept、
provided-buffer registration 是内核机制，不应在其他 OS 上伪实现。开发期应
直接分层：

```text
Proactor                         portable completion API
ProactorCapabilities             runtime POD capability
IoUringExtension                 get_if_supported() 后才能调用
```

C ABI 同样通过 capability + borrowed extension handle 调用；所有符号改为
`hyle_io_uring_*`。旧 `hyle_io_release_fixed_file(void)` 已删除，新
`hyle_io_uring_release_fixed_file` 返回可检查结果。Windows/macOS 不再编译
11 个恒定失败的 `Proactor` backend stub 定义。

### IO-CAP-08：补真实 async statx ✅

旧头文件声称 Linux `submit_statx` 使用 `IORING_OP_STATX`，实现却在所有 Linux
后端同步 `fstat` 再排 software completion，`has_async_file_metadata()` 也返回
false。普通 `File::stat` 已 offload，不堵 worker；直接 Proactor/C 调用仍可能
在 submit 线程阻塞。

已实现真实 io_uring STATX：每个在途查询有稳定的 native `struct statx`
storage，CQE 收割时转换成 `hyle_io_file_stat` 后释放；duplicate token 与
提交失败不会泄漏。测试证明 submit 返回时输出仍为 sentinel、reap 后才填入。
其他后端由产品 facade 继续 offload，capability 与实现一致。

## P2：没有写成 stub、但仍需补的产品能力

- HTTP/1.1、TLS、DNS、signal 的同等级易用 C facade。
- HTTP whole-request deadline/cancel/disconnect 传播。
- HTTP/2 流式/无界 body writer；当前 fixed body cap 为 8 MiB。
- TLS session reuse/tickets 与通用 ALPN API。
- vectored read/write、sendfile/splice/TransmitFile 等 bulk transfer。
- DNS timeout/cancel/happy-eyeballs；当前是 blocking `getaddrinfo` offload。
- 更完整的 socket options；目前主要是 nodelay、keepalive、shutdown-write。

这些能力若进入公开接口，就应随真实实现和测试一起落地，不得先放空函数占位。

## 可以保留的显式错误腿与合法 no-op

- `http2_stub.cc`：可选依赖关闭时保持当前预发布 FFI 候选的统一符号集合，先报告
  `available()==0`，业务调用明确失败并写 error。它不假装成功。
- destroy/reset 接受 NULL、零长度 feed/write：幂等资源管理和零长度语义。
- `await_ready()==false`：coroutine 总要进入 submit/suspend 仲裁，不是失败。
- `register_handle` 的 POSIX identity、非 io_uring 忽略 setup hint：不改变
  产品语义。
- timeout 竞态中 cancel 返回“目标已完成”：最终 completion/refcount 仍负责
  回收，不是漏实现。

## 验收门与施工顺序

新增测试至少覆盖：

1. ✅ accept peer address：Windows、io_uring、forced epoll、kqueue IPv4；
   IPv6 随 family-aware 网络专项补。
2. ✅ readiness 非零 recv/send flags，含 epoll/kqueue OOB selector。
3. ✅ 非零 fallocate mode 在不支持后端明确失败。
4. ✅ Windows shared-memory unlink 返回 typed Unimplemented。
5. ✅ 普通 receive 循环四后端一致，native `kMore/kBuffer` 另验。
6. ✅ capability bit、extension 获取与真实 backend 一致。
7. 🔶 C ABI 已机械区分 extension unsupported/invalid-argument；queue-full
   与更细 OS 错误仍待未来 typed submit result。
8. ✅ file metadata 的阻塞/异步报告与实现一致。

2026-07-20 最终复验：

- Windows IOCP：完整 `^hyle_io` CTest 14/14；
- WSL ext4 本地工作树：完整 `^hyle_io` CTest 13/13；Proactor 专项
  19 passed + 1 skipped，skip 是当前内核 direct accept CQE `-EINVAL`，
  provided buffer、fixed file、小环深度 flush 与原生 STATX 均真跑；
- M4 kqueue：完整 `^hyle_io` CTest 13/13；
- 顺带修复 TLS on / HTTP2 off 时 `tls_test` 无条件引用 HTTP/2 类型的配置洞。

推荐顺序：

1. ✅ P0 契约包。
2. ✅ capability POD + extension typed status。
3. ✅ 收窄 receive 契约，删除伪 portable multishot。
4. ✅ io_uring extension 分层，删除通用面恒定 stub。
5. ✅ async statx 与 C 产品 facade。
6. P2 能力与专项性能收口。

性能工作（kqueue changelist batching、同核 pinned-ready、TLS pump 扁平化）
见 [`benchmarks-io.md`](../performance/benchmarks-io.md)。性能优化不能替代本页
的语义补全：先保证 API 不撒谎，再优化真实实现。
