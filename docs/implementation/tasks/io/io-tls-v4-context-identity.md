# IO TLS v4：组合层 context identity

<!-- MODULE CLOSURE RECORD: AUDITED -->

## Closure card

- component / owner / round：`io_tls` / IO / `io-tls-v4-context-identity`；kind=`REOPEN`；
  canonical checkout=`C:\public\project-hyle\hyle`；baseline=`b834aee4`；前轮 terminal record 为
  [`io-tls-v3-credential-sources`](io-tls-v3-credential-sources.md)。
- trigger 与证据：G06 `io_http` file-body 把一个已打开 `File` 与 `HttpTransport` 组合。plain transport 可以在
  发 response head 前比较 `TcpStream`/`File` 的 `IoContext` identity；`TlsStream` 目前只公开 worker index。
  两个不同 Scheduler 都可以拥有编号为 0 的 worker，因此仅比较 owner number 会接受错误组合，随后把
  `File` operation 提交到没有由当前 worker 驱动的另一个 proactor，可能永久等待。这是新的具体直接
  consumer 缺陷，满足 CLOSED reopen trigger，不是“再审一遍”。
- 正常 consumer 任务：HTTP/TLS response 使用同一 `IoContext` 上的 open File；错误 context 在任何 wire
  output/file IO 之前返回 `InvalidArgument`。
- 主产品纵切 / 正交纵切：推理服务通过 HTTPS 发送 model sidecar；渲染 asset server 通过 HTTPS 发送
  pack-file region。二者都只需要 transport identity，不需要 OpenSSL/native handle。
- scope：增加一个只供 HYLE 组合层使用的 header-inline `detail::io_context_identity(const TlsStream&)`，委托
  内含 `TcpStream` 的既有 identity；`io_http` 用它与 File 比较。不得新增 export symbol、字段、allocation、
  provider call、branch 或公开支持承诺。
- non-goals：公开 native socket/TcpStream accessor、改变 TLS config/session/handshake/read/write/shutdown、
  OpenSSL provider、证书、ABI/FFI、另一个 runtime 或修改任何非 `io_tls/io_http` 产品面。
- 直接依赖 / 消费者：既有 `io_core` identity seam；直接消费者仅本轮 `io_http` file-body。中央 component
  graph 不变。
- 性能 tier：`P0`。新增 accessor 必须完全 inline，只在 file-response 冷 validation 使用；普通 TLS/HTTP
  路线的 layout、代码与 runtime 成本为零变化，前轮 P1 不重跑。
- 验证：TLS ON/OFF header；same/different-IoContext HTTPS file response；Windows/Linux/macOS correctness；
  source/static/shared IO consumer；candidate 后红队、docs/closure/diff。不同 context 用例必须证明 pre-wire
  typed rejection，不能用 timeout 当成功。
- 调查/纠正预算：一个 API 方案、一个产品 patch；没有 P1 sampling。若不能零字段/零符号取得 identity，
  本轮 `BLOCKED`，不得泄漏 provider/native state。

## 形状与成本审计（`AUDITED`）

`TlsStream` 已经以值成员拥有 `TcpStream socket_`，而 `tls.h` 已公开 include `tcp.h`。因此唯一最小形状是：

```cpp
namespace detail {
inline const IoContext *io_context_identity(const TlsStream &stream) noexcept {
  return io_context_identity(stream.socket_);
}
}
```

它需要一个 forward declaration、一个 friend 和一个 inline overload；不增加 object bytes、export symbol、
provider access、allocation 或 runtime dispatch。公开 `context()`/native socket accessor 会错误扩大用户产品面；
在 `HttpTransport` 复制一份 pointer 会让每条普通 HTTP connection 永久付费；继续只比较 owner index 则无法
保证进度。三者均拒绝。`io_http` 只在 file overload 的 pre-wire validation 调用 detail seam；其它 TLS/HTTP
代码不引用它。

审计结论为 **PASS**，授权上述一个 header seam 与一个 io_http 调用点；correctness、交付与候选红队尚未
运行，不能提前写 terminal verdict。
