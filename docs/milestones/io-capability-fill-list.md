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
的 capability 分支；IO 产品代码应返回 Status。当前 `Proactor` 把 portable
语义、后端加速和可选组件混在同一个 `bool` 面里，才会出现“编译通过、调用
恒定失败”的假完成感。

## 审计结论

机械扫描和逐项读实现后，当前有：

- **4 组公开契约存在静默忽略或成功 no-op**：readiness 后端的 socket flags、
  非 io_uring 的 accept peer-address、非 io_uring 的非零 fallocate mode、
  Windows shared-memory unlink。
- **11 个 io_uring 专属方法**在 IOCP/kqueue 上恒定
  `false/-1/nullptr/void no-op`；Linux 运行时退到 epoll 后也不可用。
- **portable multishot receive facade 没有承诺过的 fallback**：非 io_uring
  后端首个 `next()` 直接失败。
- **HTTP/2 feature-off stub** 是显式可选组件：`available()==0`，其余调用设置
  `last_error` 后失败。它不报告成功，属于允许保留的 ABI 错误腿。

因此 IO 的主干能运行，但 capability closure 尚未完成，不能标成“没有已知
行为缺口”的预发布冻结。

## 后端能力矩阵

| 能力 | IOCP | io_uring | epoll | kqueue | 判定 |
|---|---|---|---|---|---|
| TCP/UDP one-shot | 原生完成 | 原生完成 | readiness 模拟 | readiness 模拟 | 完整 |
| `recv/send(flags)` | 支持 | 支持 | **flags 被忽略** | **flags 被忽略** | **P0** |
| one-shot accept peer address | **忽略** | 支持 | **忽略** | **忽略** | **P0** |
| multishot accept | AcceptEx 重挂 | 原生 | 重挂模拟 | 重挂模拟 | 完整 |
| portable buffered receive stream | 缺失 | 原生快腿 | 缺失 | 缺失 | **P1 产品洞** |
| provided-buffer multishot recv | 恒定失败 | 原生 | 恒定失败 | 恒定失败 | 后端扩展 |
| fixed files / direct accept | 恒定失败/no-op | 原生 | 恒定失败/no-op | 恒定失败/no-op | 后端扩展 |
| `register_handle` | 真绑定 IOCP | 成功 identity | 成功 identity | 成功 identity | 合理 |
| async file data | 原生 | 原生 | C++ facade offload | C++ facade offload | 高层完整 |
| async file metadata | 否 | **当前也否** | 否 | 否 | **P1** |
| `fallocate(mode=0)` | 等价扩文件 | 原生 | `posix_fallocate` | `F_PREALLOCATE` | 可用 |
| `fallocate(mode!=0)` | **忽略** | 原生 | **忽略** | **忽略** | **P0** |
| shared-memory unlink | **成功 no-op** | `shm_unlink` | `shm_unlink` | `shm_unlink` | **P0** |
| HTTP/2 disabled | 显式 unavailable | 同左 | 同左 | 同左 | 合格 error 腿 |

`register_handle` 在 POSIX 返回成功不是假实现：POSIX fd 本来无需预绑定，调用后
“该 fd 可用于这个 Proactor”的后置条件已经成立。`datasync` 在 Windows/macOS
执行更强的完整 flush，也不是 no-op。

## P0：必须修复的公开契约

### IO-CAP-01：readiness 后端保留 socket flags

`submit_recv/submit_send` 已公开 flags。IOCP/io_uring 传给 OS，epoll/kqueue
却降成 `read/write`，静默按 flags=0 执行。

必须让 `PendingOp` 保存 flags，readiness `perform` 使用
`recv/send(..., flags)`，并增加非零 flag 的 epoll/kqueue/C ABI 真 socket
测试。“高层目前只传 0”不是保留契约洞的理由。

### IO-CAP-02：one-shot accept 填写 peer address

C++ 头和 C ABI 都承诺填写 `address/address_length`，但 IOCP、epoll、kqueue
忽略参数。

- IOCP 用 `GetAcceptExSockaddrs` 或 accepted socket 的 `getpeername`。
- epoll/kqueue 将 address/length 存进 pending op，执行 accept 时传入。
- IPv4、IPv6、null-address 都要跨后端测试 family、port、length。

multishot accept 因一个 submit 对应多个连接，没有独立 caller buffer，可以继续
明确要求调用方对 accepted fd 使用 `getpeername`；不能反过来削弱 one-shot
契约。

### IO-CAP-03：非零 allocation mode 不得静默成功

只有 io_uring 真正透传 fallocate mode；其余后端和 C++ blocking fallback
按 mode 0 执行后返回成功。portable API 应只保证 mode 0；其他 mode 若后端
不能实现，必须返回 `StatusCode::Unimplemented`，不能改变语义后报成功。Linux 原生
mode 放入 extension capability。

### IO-CAP-04：Windows shared-memory unlink 不得成功 no-op

POSIX unlink 会立即移除名字，现有 mapping 继续存活；Windows named mapping
没有等价操作，名字要等最后 handle 关闭才消失。当前 Windows 只检查名字非空
便返回 true，会误导调用方。

在没有真实等价实现前，C++ 返回 `StatusCode::Unimplemented`，C ABI 返回对应
稳定状态码并提供准确错误上下文。guide 应把“最后 handle 关闭后自动消失”写成
平台生命周期差异，而不是成功 unlink。

## P1：消灭通用 API 中的恒定 stub

### IO-CAP-05：增加 capability POD 与 typed status

当前 capability 只有 backend kind 和两个 file bool；失败又只有
“unsupported or registration failed”。需要 C/C++ 同源的 capability bitset，
至少覆盖：

- native/emulated multishot accept；
- portable buffered receive；
- provided buffers；
- fixed files；
- direct accept；
- async file data/metadata；
- HTTP/2、TLS；
- 后续 vectored IO、zero-copy send。

提交失败必须可区分 `Unimplemented`、`InvalidArgument`、
`QueueFull`、`ResourceExhausted` 和 OS error。C/FFI 不能靠解析
`hyle_io_last_error()` 英文做程序分支。

### IO-CAP-06：实现跨平台连续 receive 产品腿

`TcpStream::read_multishot()` 看似普通 facade，实际非 io_uring/旧内核上首个
`next()` 就返回 submit failure；头文件曾写“net fallback to pre-posted recv”，
实现并不存在。

应形成：

```text
TcpStream::read_stream()                 portable 产品能力
  ├─ io_uring provided-buffer multishot  原生快腿
  ├─ IOCP 预投递 WSARecv buffer pool      等价模拟
  └─ epoll/kqueue one-shot recv 重挂       等价模拟

IoUringExtension::provided_buffer_ring()  Linux 专属加速机制
```

portable 腿必须统一连续 chunk、EOF、取消、背压和 RAII buffer 归还。只有
“kernel-selected registered buffer/减少 submit”是可选加速。如果暂时不实现
模拟，现有 API 必须改成明确的 io_uring extension 名字，不能继续伪装 portable。

### IO-CAP-07：11 个 io_uring 方法移出 portable Proactor

非 io_uring 后端当前为 stub 的方法：

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

C ABI 同样通过 capability + extension handle 调用。尤其
`hyle_io_release_fixed_file` 目前返回 void，在不支持平台完全静默，必须改成
可检查结果或只允许由成功取得的 extension handle 调用。

### IO-CAP-08：补真实 async statx 或诚实收窄低层契约

头文件声称 Linux `submit_statx` 使用 `IORING_OP_STATX`，实现却在所有 Linux
后端同步 `fstat` 再排 software completion，`has_async_file_metadata()` 也返回
false。普通 `File::stat` 已 offload，不堵 worker；直接 Proactor/C 调用仍可能
在 submit 线程阻塞。

优先实现真实 io_uring STATX，并用稳定 op storage 承载原生结果；其他后端由
产品 facade 继续 offload。实现前必须修正文档，不能把同步代码描述成异步 SQE。

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

- `http2_stub.cc`：可选依赖关闭时保持稳定 C ABI，先报告
  `available()==0`，业务调用明确失败并写 error。它不假装成功。
- destroy/reset 接受 NULL、零长度 feed/write：幂等资源管理和零长度语义。
- `await_ready()==false`：coroutine 总要进入 submit/suspend 仲裁，不是失败。
- `register_handle` 的 POSIX identity、非 io_uring 忽略 setup hint：不改变
  产品语义。
- timeout 竞态中 cancel 返回“目标已完成”：最终 completion/refcount 仍负责
  回收，不是漏实现。

## 验收门与施工顺序

新增测试至少覆盖：

1. accept peer address：Windows、io_uring、forced epoll、kqueue，IPv4/IPv6。
2. readiness 非零 recv/send flags。
3. 非零 fallocate mode 在不支持后端明确失败。
4. Windows shared-memory unlink 不报告已删除。
5. portable receive stream 四后端一致，native `kMore/kBuffer` 另验。
6. capability bit 与真实调用结果一致。
7. C ABI 可机械区分 unsupported、queue-full、invalid-argument。
8. file metadata 的阻塞/异步报告与实现一致。

推荐顺序：

1. P0 契约包。
2. capability POD + typed status/error。
3. portable receive stream。
4. io_uring extension 分层，删除通用面恒定 stub。
5. async statx 与 C 产品 facade。
6. P2 能力与专项性能收口。

性能工作（kqueue changelist batching、同核 pinned-ready、TLS pump 扁平化）
见 [`benchmarks-io.md`](../performance/benchmarks-io.md)。性能优化不能替代本页
的语义补全：先保证 API 不撒谎，再优化真实实现。
