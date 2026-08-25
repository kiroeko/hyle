# IO 模块收口审计（2026-07-20）

## 结论

IO 已完成本轮实现与红队收口：后端、普通 C++ 产品腿、低层 C ABI、协议安全和
benchmark 计量正确性均已补齐。Windows IOCP、Linux io_uring+forced epoll、
macOS kqueue 各 14/14 IO test 全绿；当前不再有已知 P0 内存安全/挂死阻断，
2026-07-20 的 stub/capability P0 也已收口。尚不能宣称
“跨平台 SOTA”：新 harness 的三平台证书已经完成，但证实 M4 TLS/kqueue
多 worker、三平台 TLS handshake 仍有性能尾，而且 file/UDP/DNS/cancel/timeout
尚缺专项 peer benchmark。
HTTP/TLS 高层 C facade 与 whole-request deadline 属于已明确登记的产品尾项，
不影响低层机制完整，但发布承诺必须继续诚实标注。

后续 stub/no-op 审计发现的问题已经修复：socket flags、accept peer address、
allocation mode、Windows shmem unlink 均不再静默改变语义；11 个 io_uring
机制已移到显式 extension，Linux STATX 也已成为真实异步 SQE。逐项证据和仍
允许保留的显式 error 腿见
[`io-capability-fill-list.md`](io-capability-fill-list.md)。

## 本轮已修

- POSIX TCP/UDP 与 accepted socket 强制 nonblocking/CLOEXEC。
- TLS verifying client 强制 server identity，设置 SNI 与 DNS/IP hostname
  verification；trust store 失败不再返回伪有效 context；shutdown 不再做无
  timeout 的 transport read。
- HTTP request/response 拒绝 TE+CL、冲突 CL、非 bare chunked；chunked cap
  不可被 final packet 绕过；出站 CRLF/header injection 被拒绝。
- multishot 正常结束和 early-drop 的双引用、旧 Waker、buffer 归还与 overflow
  cancel 全部补齐；IoContext 析构先识别 multishot tag。
- SignalSet move-assignment lifetime UB、宿主旧 handler 恢复、单 process owner、
  Windows stale bits 修复。
- IOCP teardown 不再提前释放内核仍持有的 OVERLAPPED；steady-state IoOp 改成
  owner-worker free-list，消除每 op `new/delete`。
- IoContext worker 数必须与 Scheduler 完全一致，owner 索引加 fatal invariant。
- IPv4/IPv6 literal、family-aware TCP/UDP 与 DNS 结果打通；null/非法超长 host
  不再静默截断。
- Proactor/C ABI 暴露实际 backend 与 regular-file async capability。
- `SharedMemory` 补齐：Windows named file mapping、POSIX `shm_open`，C++ RAII
  与纯 C opaque ABI 均有 create/open/data/size/flush/unlink。
- C ABI 纯 C smoke 覆盖低层导出、post/reap 和同名 shared-memory 双 mapping；
  guide 给出 C+sched 组合与 multishot 生命周期 recipe。
- 普通 C++ `File` 会按实际 backend capability 自动选择原生异步或
  `IoContext` blocking pool；IOCP metadata 与 epoll/kqueue regular-file 不再
  堵 scheduler worker。
- TCP read/write/connect/accept 与 UDP recv/send 的 deadline、CancelToken
  便利腿已经铺齐。
- HTTP transport 可同时拥有 plain TCP 与已握手 TLS，client/server 复用同一
  framing 核；TLS 测试包含真实 HTTPS loopback。
- HTTP head 不再二次 parse；出站 method/path/host/header 做 token/control
  验证，header 容量溢出显式返回失败。
- epoll/kqueue readiness pending 表改用 HYLE `SwissTable<int, PendingOp>`：
  flat storage + SIMD control-byte probe，消除 STL node churn，也形成
  container→io 的真实内部消费证据；扩容 OOM 可返回 submit failure，同一
  fd+方向的第二个 pending op 会被拒绝，不再覆盖并孤儿化首个等待者。
- 全部 8 个 IO benchmark 已校验实际完成量；不支持的 multishot 显式 SKIP，
  不再把少完成/立即失败计成“极快”。
- io-11 HTTP/2 已落地：固定版本 nghttp2 核 + 自有 coroutine transport、
  allocator/caps/stream lifecycle；C++ h2c/TLS 与 transport-neutral C codec。
  三平台验证多 stream、HPACK、跨 64 KiB flow-control 和真实 h2-over-TLS；
  feature-on/off 符号集合稳定，C codec 失败统一维护 thread-local last-error。
- Linux io_uring `post(token,result)` 不再用固定 `res=0` 的 NOP CQE丢弃
  result，改为 owner-thread software completion；macOS TCP 设置
  `SO_NOSIGPIPE`，H2 graceful shutdown 不再受 EOF/SIGPIPE 竞态影响。

## 性能与 SIMD 判断

目前不能声称整个 IO SOTA。2026-07-20 已完成现有 8 个 benchmark 的三平台
重跑：Windows IOCP echo 4w/1w 2.24x，WSL io_uring 1.34x，M4 kqueue 则
0.48x；Linux 单 worker 比 raw epoll 慢约 10%。TLS steady 为 Windows 平、
Linux 赢、M4 输，三平台 handshake 都输 raw OpenSSL。HTTP 对 raw TCP 的
1.24x/1.66x/1.23x 是协议层成本，不冒充同功能 peer 对标。完整数据和口径见
`docs/implementation/evidence/performance/benchmarks-io.md`。

优先级：

1. 狩猎 M4 kqueue 4/8 worker、M4 TLS 与三平台 handshake 的已确认落后；
   重跑 asio/libuv 同功能 peer，区分抽象成本和实现回退。
2. IOCP allocation-count、SwissTable pending churn、file queue depth、
   burst multishot、UDP pps、cancel storm、DNS pool saturation。
3. HTTP parser-only/transport benchmark 判断长 header delimiter scan 与
   consume memmove 是否值得局部 SIMD/环形 buffer；不得凭直觉全目标开 ISA flag。

SIMD 不适合 syscall、socket/file copy、TLS crypto、DNS 或短 header compare。
唯一当前候选是长 HTTP header delimiter scan；必须采用 parser 粒度的一次动态
派发，不能给整个 io 目标加 SSE4.2 编译 flag。

## 发布尾项

- **证据门**：三平台正确性与现有 8 benchmark 新证书均已生成；证书明确暴露
  的 lose 和未覆盖专项必须保留，不能因此声称整个 IO SOTA。
- **产品尾**：HTTP whole-request 统一 deadline/cancel/disconnect；HTTP/1.1/TLS
  C facade；HTTP/2 无界 chunk writer。HTTP/2 已提供 transport-neutral C codec，
  不再属于 C ABI 空洞。
- **能力尾**：TLS ALPN/session reuse、vectored IO、socket options/half-close、
  burst multishot 与 file/UDP/cancel/DNS 专项 benchmark。
- **可选优化**：HTTP consume memmove/输出 buffer 复用；cancel 在 readiness
  后端目前按 token 扫 pending table（O(n)，取消为冷路径），若 cancel storm
  证实成瓶颈再加 token→fd 第二索引。

当前主干实现、C ABI 红队正确性与三平台证书已经完成，但
`io-capability-fill-list.md` 的 P0/P1 尚未清零。IO 暂不进入“无行为缺口”的
预发布冻结；产品/能力尾项须保留在公开 guide 与 roadmap，不能用“模块已收口”
掩掉。
