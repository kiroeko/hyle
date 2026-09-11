# IO Core v4：异步文件区域流送

<!-- MODULE CLOSURE RECORD: AUDITED -->

## Closure card

- component / owner / round：`io_core` / IO core maintainers / `io-core-v4-file-region-streaming`；
  kind=`REOPEN`；canonical checkout=`C:\public\project-hyle\hyle`；baseline=
  `694cde21d5608e4569fb87a1b12d7c68ee133ea0`；前轮 terminal record 是
  [`io-core-v3-recursive-directory-submit`](io-core-v3-recursive-directory-submit.md)。
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)已准入 G06：
  现有 `File::read_some_at` 与 `TcpStream::write_some` 是两个原子，用户仍须重复实现 offset/progress、
  short read/write、deadline/cancel、peer close 和 backend drain，并且无法自然取得 OS file-to-socket fast
  path。该缺口同时出现在推理制品/静态资源服务与渲染资产流送。
- 正常 consumer 任务：从一个已打开、稳定生命周期的 file 中，把显式 `[offset, offset+length)` 区域异步
  发到同 owner 的 connected byte stream；返回已提交字节和 typed terminal 状态，调用方可从剩余区域重试。
- 主产品纵切 / 具名来源：推理服务发送 tokenizer、adapter、模型 sidecar 或下载 range；审计
  ggml/llama.cpp、ds4、vLLM/SGLang 的制品服务调用形状，并以 nginx/Seastar/libuv/Asio-Beast 中最多两个
  同合同 strong peer 的源码确定 fast-path/fallback 语义。
- 正交反偏置纵切：渲染/资源 runtime 从 pack file 流送一个纹理或 shader region，防止接口被 HTTP path、
  MIME 或模型格式塑形。
- 跨模块 public journey 与 owner：Platform-fs 打开/描述 native file；io_core 拥有 `File`、`TcpStream`、
  async progress/cancel 与 OS fast path；Mem 提供 fallback scratch；HTTP 仅组合 framing。应用拥有 path
  authorization、range policy、cache validators 和资源 schema。
- scope：定义自然的 borrowed File + explicit offset/length C++ 形状；支持 partial progress、EOF、
  deadline/cancel、peer close 和 owner teardown；Windows/Linux/macOS 在语义成立时使用 native
  file-to-socket 机制，否则使用单块有界 read/write fallback；fast path 与 fallback 差分一致。
- non-goals：路径打开/授权、整个文件 convenience、HTTP header/range 解析、TLS、压缩、mmap cache、
  upload、文件复制/atomic publish、第二 event loop、Native FFI、G03/G04、Compute/Graph，或为 benchmark
  暴露 native operation。
- 直接依赖 / 直接消费者：先审计现有 `core,error,time,platform,sched,mem,container_text,util_core` 边；
  不预设新增 factor。直接 consumer 是 `io_http` G06；不得令 io_core 反向依赖 HTTP/TLS/Transfer。
- 性能 tier 与原因：先 `P0` 审 syscall/copy/allocation/wake/完成边界；bulk plain-socket route 预期 `P1`，
  只有自然 API、同语义 peer 与实际源码 preflight 全部通过才采样。固定 material non-inferiority margin 5%。
- 适用机器 / 仪器：Windows MSVC、WSL GCC 14、macOS AppleClang 做 correctness；ASan/UBSan 及适用 TSan；
  source/static/shared selected IO consumer。P1 若适用只用统一仪器、release artifact、相同 file/socket/
  backpressure/completion 合同、workload-matched A/A 与 A/B、B/A。
- 调查与纠正预算：一次 consumer/peer/API 源码审计；AUDITED 前冻结一个 API/backend 方案和至多三个
  `FIX` 批次（公共合同、portable path、native paths）。完整 P1 最多一次；任何固定失败先查源码/汇编/
  syscall 与完成边界，不重复采样寻找 PASS。
- 预声明验证：zero/empty、offset/length overflow、区域越过 EOF、partial progress、短写/EAGAIN、cancel/
  deadline、peer close、file/stream close、wrong owner/context、OOM、fast-path unsupported/fallback；三平台、
  sanitizer、逐头、直接 HTTP consumer、三交付、红队、P0 与适用 P1。
- stop / blocker：若三平台不能在不复制 runtime、不隐藏 blocking worker 或不破坏既有 File/TcpStream owner
  合同下提供一致 progress/cancel/teardown，则 `BLOCKED`；若 native 机制语义不等价，明确走 fallback，
  不降低公共合同或用 raw syscall 对完整 façade 做苹果对橘子排名。

## 能力与形状审计（`AUDITED`）

| 能力族 | 层级 | 真实任务 / peer | 当前能力 | 裁决 |
|---|---|---|---|---|
| borrowed file region | 原子机制 | Seastar `make_file_data_source(file, offset, len)`；pack-file asset | `File::read_some_at`，调用者自己维护 progress | `FIX`：一个显式 value region，不接管 file |
| file-to-TCP progress | 复合原子 | nginx `sendfile` chain；Boost.Beast Win32 file body | 只能手写 read/write loop | `FIX`：完整发送或保留 partial+typed terminal |
| native fast path | backend 机制 | nginx Linux/Darwin；Beast Win32 `TransmitFile` | 无 | `BORROW` 机制与边界，不复制 peer runtime |
| bounded fallback | backend 机制 | libuv 64 KiB emulation、TLS/不支持 native 的文件 | 用户自己分配和循环 | `FIX`：一次有界分配，逐块 positional read/write |
| cancel/deadline/teardown | 公共合同 | 推理制品服务、渲染 asset stream | 两个原子分别有合同，组合易漏 | `KEEP + FIX`：复用一个 absolute budget，等待原 op retirement |
| path/range/MIME/cache | 应用策略 | llama.cpp mount、SGLang `FileResponse` | 不属于 byte transfer | `REJECT`：留给 HTTP/service owner |

全部现有公开面已反向追踪到 `File`、`TcpStream`、`OperationOptions`、`TransferResult`、完整 read/write helpers
和 Linux expert extension。没有旧 file-region 入口可兼容，也不新增 whole-file convenience 或 native handle
overload。自然 C++ 形状冻结为：

```cpp
struct FileRegion {
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
};

sched::Task<TransferResult>
send_file_region(mem::AllocatorRef, TcpStream&, File&, FileRegion,
                 OperationOptions = {}) noexcept;
```

目标 stream 在 source file 之前，和 `write_all(allocator, stream, bytes)` 及 OS `sendfile(out, in, ...)`
一致。`FileRegion` 按值进入 coroutine frame，调用者只需保持 `File`、`TcpStream` 活到 task 完成。两者必须
有效、绑定同一 owner worker，且调用 task 的 fiber 必须在该 owner；范围须能由 native signed file offset
及 `size_t` progress 表示。零长度仍校验 owner/对象后成功。区域短于文件以外时返回 `DataLoss`、
`eof=true` 与已发送字节；任意错误保留已提交 progress，调用者可从 `offset + bytes` 显式恢复。

### consumer 与 peer 源码锚点

- llama.cpp `e71b80510c84` 的 `tools/server/server-http.cpp:335` 使用 static mount；SGLang
  `dcebe8c4733a` 的 image/video/mesh endpoints 返回 `FileResponse`。这些证明已打开制品到 response 的任务，
  但 path authorization、MIME 和 route 仍由服务层拥有；vLLM 未增加更低层 MUST，ds4 无独立形状。
- nginx `5c812c034e6c` 的 `ngx_linux_sendfile_chain.c:177-291` 与
  `ngx_darwin_sendfile_chain.c:135-185` 保留 partial，`EAGAIN` 后等可写事件，零进度解释为截断，并把单次
  Linux 调用限制为 bounded chunk。该 source timeline 是 POSIX strong peer。
- Boost `cb94f33fcb13` 的 Beast `file_body_win32.hpp:416-511` 把 file offset 写入 `OVERLAPPED`，每次
  `TransmitFile` 限定到 serializer limit/`INT_MAX-1`，最终 completion 才推进 body position；这是 Win32
  strong peer。
- libuv `3e606bc30a84` 的 Unix `fs.c:1048-1155` 有 sendfile/fallback，但 `POST` 把 FS request 投到
  `UV__WORK_FAST_IO`；Windows `fs.c:2488-2531` 更是 64 KiB `_read/_write` emulation。它不是 HYLE owner
  fiber/nonblocking completion 的同合同性能 peer，只借 fallback 边界，禁止直接排名。
- Seastar `78f64e2317b1` 的 `fstream.cc:209-218,343-348` 验证显式 offset/length 是自然 region 形状；它的
  DMA input stream 生命周期不同，不作为 file-to-socket 性能 peer。

## 三 OS 时序、资源与 P0 成本账单

| 路线 | submit / wait / complete | allocation / copy / lock / wake | 复杂度与停止裁决 |
|---|---|---|---|
| Windows plain TCP | 每个至多 2 MiB region chunk 用 overlapped `TransmitFile`；IOCP completion 后推进 offset | 复用 Proactor 预分配 `IoOp` slab；零 payload copy、无新锁、每 chunk 至多一次 wake；bounded 档只用既有 timed slot | O(bytes/chunk)，取消走 `CancelIoEx(socket, ov)` 并等 terminal completion；支持 |
| Linux plain TCP | 非阻塞 `sendfile`；有 progress 立即提交，`EAGAIN` 用同 ring `POLL_ADD(POLLOUT)`/epoll 后重试 | 零 payload copy；poll 用既有 ring/map admission，普通档不分配；每 2 MiB cooperative yield 防止垄断 worker | O(bytes/chunk)，无 busy poll；支持 |
| macOS plain TCP | 非阻塞 Darwin `sendfile`，读取 `len` 中的 partial；`EAGAIN` 用 kqueue write filter 后重试 | 与 Linux 相同；无 payload copy；文件页由 kernel 路由 | O(bytes/chunk)，无 busy poll；支持 |
| portable fallback | 一次分配 64 KiB scratch；每块 positional read，再把同一块完整写出 | O(1) bounded storage、每字节一次 userspace copy；无逐块 scratch 分配；file offload 沿既有 pool | O(n)，取消/deadline 在块边界检查且 socket wait 使用原 budget；支持 |

native 不支持的已知 errno 才切 fallback；Windows `TransmitFile` 对越过 EOF/部分 provider 限制返回
`WSAEINVAL`，该值同样退到 positional-read 路径，由后者区分短文件与真实 file/socket 错误。其他真实
错误不伪装为“不支持”。direct/unbuffered file 若
native 机制拒绝且 region 不满足 direct alignment，则 typed `Unimplemented/InvalidArgument`，不能偷偷改成
buffered IO。一次 native success 或 poll completion都必须先核对 reported bytes 不超过请求。默认
`File`/`TcpStream` layout 不变；新增 detail submit 只扩 backend seam，不给用户暴露 syscall。

## 冻结有限改动集

| ID | 已证问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `G06-C01` | 两个原子不能表达 region owner/progress | `FileRegion` + `send_file_region`，完整 validation/partial/EOF | zero/overflow/short file/wrong owner/peer close |
| `G06-C02` | 无 OS fast path 和背压等待；Linux `sendfile` 没有 `MSG_NOSIGNAL` 参数，不能继承普通 socket write 的逐调用抑制 | IOCP `TransmitFile`；POSIX `sendfile` + private writable wait；Linux 每次同步 syscall 在当前线程短暂 block/drain 自身新产生的 `SIGPIPE` 后立即恢复 mask；一次 scratch fallback | 无 busy loop、无进程级 signal policy、cancel/deadline retirement、native/fallback 差分 |
| `G06-C03` | 新公共 journey 尚无交付与说明 | 定向 unit/header/source-static-shared、三平台/sanitizer、user/architecture 文档 | 预声明矩阵诚实裁决 |

不允许第四个产品批次；不得加入 path API、MIME/Range、TLS、file cache、第二 event loop 或 benchmark-only
knob。源码账单已经给出算法差异，在实现正确性稳定前不运行 P1；候选时仅当能构造完整同合同 peer 才运行
一次。审计结论为 **PASS**，允许进入上述有限实现；全部 correctness/交付/性能结果仍未运行，不能提前写
成通过。

## 候选实现 checkpoint（`WORKTREE`）

- `G06-C01/C02` 已落地：公开 `FileRegion` 与 `send_file_region`；Windows 使用运行时
  `WSAIoctl(SIO_GET_EXTENSION_FUNCTION_POINTER)` 取得的 overlapped `TransmitFile`，Linux/macOS 使用
  nonblocking `sendfile`，不可用时只分配一块 64 KiB scratch。所有路线保留 partial progress、typed EOF、
  cancel/deadline 和 owner/context 校验。
- 源码红队修正了三项具体风险：Linux `sendfile` 不再继承进程级 `SIGPIPE` policy；io_uring poll 使用
  portable-width `poll32_events`；`File` 与 `TcpStream` 除 worker number 外还必须共享同一 `IoContext`。
- Windows/MSVC 19.50 shared Release 的定向 `file_region` 与 HTTP 组合测试通过；Linux/GCC 14 shared
  Release 的同组测试在默认 io_uring 和强制 epoll 下均通过。覆盖 zero/overflow、预取消/过期、短文件、
  peer reset、in-flight cancel、跨 context 预提交拒绝与 Linux `SIGPIPE` handler 不触发。
- macOS、sanitizer、逐头和 source/static/shared 仍待 immutable candidate 后完成；因此本节不是 terminal
  verdict，registry 继续保持 `AUDITED`。源码 preflight 没有发现完整同合同 strong peer，当前 P1 为
  `NOT_APPLICABLE`，不得用 raw syscall 与完整 typed coroutine 排名。
