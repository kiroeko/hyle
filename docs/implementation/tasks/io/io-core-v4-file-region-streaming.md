# IO Core v4：异步文件区域流送

<!-- MODULE CLOSURE RECORD: SCOPED -->

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

## 审计待办

进入 `AUDITED` 前必须完成 capability-family map、全部现有公开入口回映、三 OS 时序/资源账单、peer
源码锚点、自然 C++ 草图、依赖与三交付影响，并冻结有限 `KEEP/FIX/BORROW/DEFER/REJECT` ledger。
