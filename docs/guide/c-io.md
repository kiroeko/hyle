# io 纯 C ABI 使用指南

`hyle/c/io.h` 是 IO 模块的稳定语言边界。它暴露完成式 proactor、POD
completion、文件/网络提交原语、io_uring 可选加速和命名共享内存；不把
C++ coroutine、`TcpStream`、OpenSSL 对象或 STL 类型放进 ABI。

链接 `hyle_io`：

```c
#include "hyle/c/io.h"
```

## 1. 基本完成循环

每个成功的 `hyle_io_submit_*` 最终产生一个带原 `user_data` 的
`hyle_io_completion`。buffer、路径、sockaddr、scratch 和 stat 出参必须活到
对应 completion 被 reap。同一时刻的在途操作必须使用唯一 `user_data`；
multishot 是一个操作用同一 token 产生多个 completion 的特例。`UINT64_MAX`
是内部 wake token，应用不能使用。

```c
hyle_io_proactor_options opts = hyle_io_proactor_options_default();
hyle_io_proactor* p = hyle_io_proactor_create(&opts);
if (!p) fail(hyle_io_last_error());

if (!hyle_io_submit_read(p, fd, buf, sizeof(buf), 0, 42))
  fail(hyle_io_last_error());

hyle_io_completion c[64];
size_t n = hyle_io_reap(p, c, 64, UINT64_MAX);
for (size_t i = 0; i < n; ++i) {
  if (c[i].user_data == 42) consume_result(c[i].result);
}
hyle_io_proactor_destroy(p);
```

`hyle_io_proactor_options` 的 ABI 头固定为两个 32-bit word：
`uint32_t struct_size` 和必须为 0 的 `uint32_t reserved`。全部业务字段都有
默认值，因此 `HYLE_IO_PROACTOR_OPTIONS_REQUIRED_SIZE` 是 8；普通源码仍应使用
`HYLE_IO_PROACTOR_OPTIONS_INIT` 或 `hyle_io_proactor_options_default()`，而短前缀
主要供旧 binding/FFI 使用。

`struct_size` 表示调用方提供的可读前缀，不要求等于当前 `sizeof`。库先建立完整
默认 options，再只用调用方完整提供的字段覆盖：缺失的 `entries`、
`defer_taskrun` 和 `backing` 保留默认值，落在字段中间的前缀把该字段视为缺失。
`hyle_allocator` 是一个不可拆分的 vtable+state 单元，只有整个字段都可读时才会
覆盖默认 allocator。大于当前结构的未知尾部会被忽略，库不会读取它。小于
required prefix 或 `reserved != 0` 时 create 返回 NULL，
`hyle_io_last_error()` 报告 invalid argument。`defer_taskrun` 是固定宽度
`uint32_t` 布尔值，只接受 0 或 1。

proactor 是单 owner 线程对象；只有 `hyle_io_wake` 可从其他线程调用。
`hyle_io_post` 仍是 owner-thread 注入。C 用户若要和 HYLE scheduler 组合，常见
形状是：每个 worker 一个 proactor，`hyle_sched_spawn_on` 把该 shard 的完成循环
钉在同一 worker；其他线程发布任务后调用 `hyle_io_wake`。

## 2. 后端必须可观察

`hyle_io_proactor_backend` / `hyle_io_backend_name` 返回实际选中的
IOCP、io_uring、epoll 或 kqueue。Linux 可能因内核、seccomp 或 sysctl 从
io_uring 降为 epoll，benchmark 和依赖特定能力的程序必须查询，不能靠平台名
猜测。`hyle_io_proactor_capabilities()` 一次返回 backend + bitset；
`hyle_io_has_async_file_data/metadata` 是相同信息的便利查询。当前 io_uring
的 data 与 STATX metadata 都是真异步；IOCP 只有 data read/write 是
overlapped，epoll/kqueue 的文件 facade 走 blocking pool。

`hyle_io_backend` 是固定 `uint32_t` typedef，命名值是 integer constants；FFI 不应
绑定成本语言默认 enum，`-fshort-enums` 也不会改变返回值或 snapshot 字段宽度。
`hyle_io_capabilities` 是按值返回的小型 **frozen v1 snapshot**：当前 64-bit ABI 固定
size 16、backend offset 4、bits offset 8。这里的 `struct_size` 只描述库返回的完整值，
不是调用方 capacity，也不是允许未来直接追加字段的 extensible handshake；需要新字段时
新增 query/type，现有 bit 能力继续在 `bits` 内演进。

capability bit 表示已选 backend 暴露相应机制，不是对任意内核版本、容器
策略和当前资源的“调用必成功”保证。例如拿到 io_uring extension 后，旧内核
仍可能拒绝 provided-buffer 或 direct accept；调用方必须检查注册/提交结果及
completion。它与 stub 的区别是：非 io_uring 在扩展获取阶段就返回 typed
Unimplemented，而不是把“不支持”伪装成一次普通资源失败。

## 3. 网络

`submit_recv/send/accept/connect/recvfrom/sendto` 接受原生 socket handle。
C ABI 不复制 sockaddr，也不冻结平台 socket 结构：调用方用本平台
`sockaddr_in`/`sockaddr_in6`，并把 `sizeof` 作为长度传入。Windows socket 必须
用 overlapped 模式创建，然后先调用 `hyle_io_register_handle`；POSIX socket
必须是 nonblocking。这样没有第二套 event loop，也没有隐藏 socket 所有权。

datagram 操作的 `scratch` 至少为 `HYLE_IO_DATAGRAM_SCRATCH_SIZE`，并与其他
参数一起活到 completion。C++ 的 `TcpStream/UdpSocket` 只是这些原语之上的
RAII + co_await 便利层；C/FFI 用 opaque/native handle + completion loop 表达
相同行为。

## 4. 文件

`submit_open/close/read/write/stat/sync/allocate` 覆盖 C++ `File` 的全部底层
能力；flags 使用 `HYLE_IO_OPEN_*`。completion 的负 `result` 是平台错误，
非负结果分别是 handle、字节数或 0。

io_uring 对主要文件操作是真异步；IOCP 的 data read/write 是 overlapped，
部分 metadata，以及 epoll/kqueue 的 regular-file 操作，可能是同步模拟。
延迟不可控的文件工作应先查 capability，必要时经
`hyle_sched_blocking_submit` 卸载，不能堵住 scheduler worker。

## 5. 显式 io_uring 扩展

provided-buffer multishot、fixed file 和 direct accept 是 Linux 内核机制，
不属于 portable `Proactor`。必须先取得借用扩展句柄：

```c
hyle_io_uring_extension uring = HYLE_IO_URING_EXTENSION_INIT;
hyle_status status = hyle_io_uring_extension_get(p, &uring);
if (!hyle_status_is_ok(status)) {
  /* IOCP/epoll/kqueue: HYLE_STATUS_UNIMPLEMENTED，继续用普通 submit_recv/send。 */
  hyle_status_free(&status);
  use_portable_path();
} else {
  hyle_status_free(&status);
  if (!hyle_io_uring_register_buffer_ring(uring, 7, 256, 4096))
    fail(hyle_io_last_error());
}
```

成功取得句柄后才可使用：

- buffer ring：`hyle_io_uring_register_buffer_ring` →
  `hyle_io_uring_submit_recv_multishot` →
  `hyle_io_uring_buffer_address` → 处理 →
  `hyle_io_uring_buffer_return`；
- fixed file：`hyle_io_uring_register_file_table` →
  `hyle_io_uring_assign_fixed_file` →
  `hyle_io_uring_submit_recv_fixed/submit_send_fixed` →
  `hyle_io_uring_release_fixed_file`；
- direct accept/close：`hyle_io_uring_submit_accept_direct/close_direct`。

扩展不拥有 proactor，`hyle_io_proactor_destroy()` 后立即失效。普通连续接收
就是重复提交 portable `hyle_io_submit_recv`；这已具有完整 backpressure 和
EOF/error 语义。io_uring multishot 的额外价值是“一次 submit 多次完成 +
内核从 registered ring 选 buffer”，不在其他后端伪造一个同名慢路径。

带 `HYLE_IO_COMPLETION_MORE` 的 token 不能回收；直到收到 MORE 清零的终止
completion。带 `HYLE_IO_COMPLETION_BUFFER` 时 `buffer_id` 有效，每个取得的
buffer 必须恰好归还一次。

## 6. 命名共享内存

```c
const char* name = "my_process_pair";
hyle_io_shared_memory* owner =
    hyle_io_shared_memory_create(name, 1u << 20);
hyle_io_shared_memory* peer = hyle_io_shared_memory_open(name);

memcpy(hyle_io_shared_memory_data(owner), payload, payload_len);
hyle_io_shared_memory_flush(owner);

hyle_io_shared_memory_destroy(peer);
hyle_io_shared_memory_destroy(owner);
hyle_status unlink_status = hyle_io_shared_memory_unlink(name);
if (!hyle_status_is_ok(unlink_status)) {
    /* Windows: HYLE_STATUS_UNIMPLEMENTED; see lifecycle note below. */
}
hyle_status_free(&unlink_status);
```

create 是排他的；open 映射完整对象。POSIX 的 unlink 删除名字但不破坏已有
mapping；Windows 没有同等的“立即移除名字”操作，对象在最后一个 handle
关闭后自动消失。因此 `hyle_io_shared_memory_unlink()` 返回 owning
`hyle_status`：POSIX 成功为 OK，Windows 稳定返回
`HYLE_STATUS_UNIMPLEMENTED`，不再以成功 no-op 误导调用方。所有返回的 status
都要调用 `hyle_status_free()`（OK 不分配，但统一释放最省心）。能力边界见
[`io-capability-fill-list.md`](../milestones/io-capability-fill-list.md)。
指针只在本进程有效，跨进程协议必须存 offset/POD，不能存裸指针。
默认工厂用 `hyle_system_allocator()` 保存 opaque handle；需要由宿主 allocator
统一记账时使用 `hyle_io_shared_memory_create_with_allocator` /
`hyle_io_shared_memory_open_with_allocator`。allocator 必须活到 destroy，且
alloc/free/realloc 三个函数要么全部提供，要么整个结构清零以选默认 allocator。

## 7. C++ 能力到 C/FFI 的组合

| C++ 形状 | C/FFI 通路 |
|---|---|
| `Proactor` / `Completion` | opaque `hyle_io_proactor` + POD completion |
| `RecvOp/SendOp/File*Op` | submit + user token + reap |
| `TcpStream/UdpSocket/File` RAII | native handle 由调用方持有；submit；最终 close |
| `IoContext` 每核 ring | 每 worker 一个 proactor + sched `spawn_on`/wake |
| timed/cancellable op | timer/cancel winner 由调用方状态机仲裁；`submit_cancel`，并等最终 CQE 后回收参数 |
| multishot RAII `RecvBuf` | MORE/BUFFER flags + 显式 `buffer_return` |
| `SharedMemory` | opaque create/open/destroy/unlink |
| HTTP/2 | opaque transport-neutral session：feed/next_output/message |
| DNS | `hyle_sched_blocking_submit` 中调用系统 resolver；结果写调用方 POD |
| signal | platform C ABI 的信号/线程设施或宿主语言事件入口，再 `hyle_io_wake` |

HTTP/2 已有 `hyle_io_http2_session` codec facade。它不拥有 socket：先把所有
`hyle_io_http2_next_output` span 通过 proactor/TLS 发出，再把收到的 bytes
交给 `hyle_io_http2_feed`；用 `next_message` 轮询完整 stream。这样可直接接
C、C#、Java 或自有 event loop，而不冻结 coroutine/TLS ABI。大 body 受 peer
window 阻塞时，`next_output` 暂时返回 0；继续 feed 对端的 WINDOW_UPDATE 后
再取。所有返回 span/message/header 均为 borrowed view，下一次 session 调用前
必须消费或复制。

HTTP/2 是可选构建能力，但 C ABI 的符号集合保持稳定。先调用
`hyle_io_http2_available()`：返回 0 时，session 工厂返回 NULL、其他 codec
操作失败并设置 `hyle_io_last_error()`；调用方不需要按构建变体做动态符号
探测。返回 1 才创建 session。session 工厂的全零 `hyle_allocator` 选择系统
allocator；非零 allocator 必须完整提供 alloc/free/realloc。HTTP/2 开启与
关闭的纯 C smoke 都受 CI/收口测试覆盖。`hyle_io_http2_role` 同样是固定
`uint32_t` typedef，只接受 `HYLE_IO_HTTP2_CLIENT` / `HYLE_IO_HTTP2_SERVER`；不要把
宿主语言或编译器的默认 enum 宽度带过 ABI。
HTTP/2 transport 依赖 sched/fcontext：支持目标默认启用，不支持目标默认关闭且显式
`HYLE_IO_HTTP2=ON` 在 configure 阶段失败。后一种形状仍导出上述稳定 stub/availability=0，安装包也不会
把 `io_http2` 写进 feature manifest。TLS 同理：无 fcontext 时 AUTO 不探测 OpenSSL，ON 直接失败。

这个无 fcontext 的 `io` 产品只包含 standalone proactor、shared memory 和稳定 HTTP/2
C probe/stub；安装树仍保留统一公共头文件集，头文件存在不代表 coroutine transport
已编入。CMake 用户可查看 `sched` 是否在 `Hyle_AVAILABLE_COMPONENTS` 中；链接
`hyle::io` 时还会收到 `HYLE_IO_HAS_COROUTINE_TRANSPORT=0|1`。值为 0 时包含
`hyle/io/net.h` 或任何依赖它的 tcp/udp/file/dns/signal/http C++ 头会给出明确
编译期错误，而不是在最后链接时才出现缺失符号。

HTTP/1.1 与 TLS convenience 仍没有同等级 C facade；C/FFI 可完整使用 proactor
网络机制并接宿主 HTTP/TLS stack。这是余下的易用性缺口，不用把 C++ 对象强转
过边界。

## 8. 错误与所有权

工厂失败返回 NULL，submit 失败返回 0；随后读取线程局部
`hyle_io_last_error()`。成功会清理 stale error。异步 OS 错误在 completion 的
负 `result` 中返回。destroy 接受 NULL。不得让外语异常跨 callback/C 边界。
