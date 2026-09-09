# IO Transfer v1：可靠模型/制品传输工作流

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

- component / owner / round ID：`io_transfer` / IO maintainers / `io-transfer-v1`；task kind：`NEW`；
  baseline commit：`e1499bb404e1a69881fb1c4d91c60aa183d1d1a7`；
- demand：`IOTRANSFER-MODEL-01`、`IOTRANSFER-RESUME-02`、`IOTRANSFER-ATOMIC-03`。
  llama.cpp 与推理部署普遍需要下载/复制大型模型；当前 HYLE 只有 file/socket primitive，用户必须
  自写 redirect/range/resume/checksum/temp-file/atomic publish 和失败清理；
- normal journey：在总 deadline/cancel 和 byte budget 下，把 HTTP(S) 或本地 file source 可靠地
  传到临时文件，支持安全 resume、进度/速率 backpressure、校验，成功后原子发布；失败不损坏旧文件；
- scope：typed source/destination、bounded copy pipeline、HTTP range/redirect policy、resume validator、
  checksum hook、temp-file + fsync + atomic replace、progress observation、cancel/cleanup；
- non-goals：Hugging Face/model repository schema、authentication UI/token storage、archive extraction、
  model parsing/loading、P2P/RDMA、object-store SDK、route/server、Native FFI/C API；
- dependencies / consumers：实际公开边为 `core,error,io_core,io_tls,mem,sched,container_text`，实现边为
  `container,io_http,platform_fs,time`；公开 `io_tls` 边来自 borrowed custom `TlsContext`，HTTPS route
  only when provider capability is available；
- tier：`P1`。primary 为 256 MiB sparse/generated source→temp→atomic publish，holdout 为 interrupted
  range resume；吞吐预算以 OS/file/network 限制为主，但用户态只允许有界 buffers、无全文件 copy、
  O(1) memory in file size；
- budget：总 8 engineering hours，审计/API 3 小时，实现/correctness 3 小时，P1 2 小时；一次 candidate
  bundle，最多一次机制诊断；
- machines / instruments：`W1-WIN/W1-WSL/M1-MAC`，`closure-correctness-v1` +
  `closure-perf-v1`；
- strong peers：libcurl transfer/resume semantics 与 Boost.Beast composed HTTP operations；它们用于
  workflow/contract 对照，不自动成为相同 scheduler/allocator 的 same-contract peer；
- provenance：llama.cpp model download/remote media 是真实 reference；vLLM/SGLang/ds4 的 model
  deployment 说明制品获取是通用 journey，但仓库/auth policy留在上层；
- API freeze 前明确 partial file identity、resume validator、redirect/security、checksum ownership、
  progress callback lifetime/reentrancy、cancel/close、atomic replacement 和 direct/buffered file policy；
- gates：empty/large/short IO、disconnect/resume/range mismatch/redirect loop/checksum/OOM/disk full/
  cancel/old-file preservation、三平台、source/static/shared、P1、guide、红队；
- stop：若 HTTP factor 尚未提供完整 streaming body/cancel contract，本卡可保持 `AUDITED/BLOCKED`，
  不复制 socket/parser；不把模型 schema、auth 或 batch policy下沉。

## Completed audit inputs

- 追踪 llama.cpp download 与四组引擎部署路径，区分库机制与应用 repository policy；
- 对 libcurl/Beast 的 redirect/range/resume/temp-file/checksum 生命周期建立能力图；
- 冻结与 `platform_fs`/Mem/HTTP/TLS 的唯一 ownership 和有限 FIX 集；
- 没有 existing public surface；新 API 必须先通过 normal C++ sketch，再允许实现。

## Capability-family coverage map（`AUDITED`）

| family | admitted journey / lifecycle | baseline | coverage / action |
|---|---|---|---|
| local source | regular file range、known/virtual-unknown size、short read/source mutation | raw async file primitive only | `IMPLEMENTED / VERIFIED` composition |
| HTTP(S) source | streaming body、length/range、redirect、validators | no transfer workflow | `IMPLEMENTED / VERIFIED` |
| destination transaction | sibling temp、flush + atomic replace | primitives exist across io_core/platform_fs | `IMPLEMENTED / VERIFIED` owner |
| resume | exact partial owner、Range/If-Range、206/200/416 decisions | absent | `IMPLEMENTED / VERIFIED` |
| integrity | byte count and caller-supplied incremental whole-artifact verifier | absent | `IMPLEMENTED / VERIFIED` hook |
| resource budget | total deadline/cancel、max bytes、bounded buffers、natural sink backpressure | per-operation primitive only | `IMPLEMENTED / VERIFIED` |
| progress | monotonic observation without reentrant ownership transfer | absent | `IMPLEMENTED / VERIFIED` |
| failure cleanup | preserve old target、retain/remove/keep partial by policy、typed outcome | absent | `IMPLEMENTED / VERIFIED` |
| repository/auth/model semantics | token store、Hub paths、manifest/archive/model parsing | application-specific | `OUT_OF_SCOPE / REJECT` |

llama.cpp provides the defining direct consumer: remote artifacts use redirects, range/resume validators, temporary files and
integrity checks before publication. ds4/vLLM/SGLang confirm deployment demand but do not admit their repository/auth/model
policy. libcurl is a semantic peer only; HYLE composes its already-owned HTTP, file, cancellation and scheduler mechanisms.

### Public/source reverse trace and owner decision

- no current public transfer surface constrains names or layout. The API starts from an owning `TransferPlan`/typed source and
  destination, not from benchmark buffer sizes or callback context pointers;
- resume is valid only when local metadata and an exact source/partial prefix comparison agree, or the remote validator agrees. A range ignored as `200`, mismatched ETag/
  Last-Modified, or inconsistent total length restarts or fails according to explicit policy; bytes are never blindly appended;
- publication creates a sibling temporary file, flushes file data, uses the platform's strongest admitted atomic replace and
  preserves the previous destination on failure. Directory durability is reported where the platform contract can express it;
- checksum is a caller-owned incremental type-erased hook with no algorithm mandate in IO. Model format, expected digest lookup
  and authentication remain consumers' responsibility;
- `io_transfer` owns orchestration; `io_http`/`io_tls` own protocol bytes, `io_core` owns async files, Platform FS owns atomic
  filesystem mechanisms, Error/Time/Sched own failure/budget/cancellation. A reproduced missing lower contract lawfully reopens
  its owner through the registry; this factor must not fork a replacement primitive.

## Frozen finite FIX set（`AUDITED`）

| ID | exact work | required gate |
|---|---|---|
| `IOTRANSFER-F01` | 建立 `io_transfer` target/header/source/test owner、独立 export macro、产品 façade/root 与精确依赖 | exact factor、source/static/shared、selected component/header scan |
| `IOTRANSFER-F02` | typed local/HTTP(S) source、destination transaction、limits/resume/redirect policy 与 typed report | compile-only local/HTTP/HTTPS natural journeys; invalid option tests |
| `IOTRANSFER-F03` | bounded copy pipeline + whole-transfer OperationOptions/max bytes/progress/checksum；O(1) memory in file size | empty/short/large/OOM/cancel/deadline/slow source/sink |
| `IOTRANSFER-F04` | HTTP redirect/range/If-Range/validator state machine 与安全 restart/retain decision | 200/206/416, mismatch, redirect loop/cross-origin security |
| `IOTRANSFER-F05` | sibling temp、flush、atomic replace、old-target preservation 与 policy-driven partial cleanup | disk full/rename failure/crash-boundary approximation/old file intact |
| `IOTRANSFER-F06` | guide/architecture、三平台 sanitizer/correctness、三交付、256 MiB+resume P1 与红队 | no apple-vs-orange、terminal receipt |

不实现 repository client、auth/token、archive/model parsing 或 server route；不保留失败后损坏目标的 convenience。
若 io_http 的 streaming/cancel 合同尚未成为 immutable candidate，本 factor 等待其 owner，不复制 parser/socket。

## Implemented API / usability candidate

```cpp
TransferOptions options;
options.limits.max_bytes = model_budget;
auto plan = TransferPlan::http(
    alloc, HttpTransferSource{url}, AtomicFileDestination{path}, options);

TransferExecution execution;
execution.owner_worker = owner;
execution.operation = operation;
execution.verifier = TransferVerifier::from(checksum_hook);
execution.progress = TransferProgressObserver::from(progress_observer);
execution.tls_context = optional_custom_tls_context;
auto report = co_await transfer(alloc, io, std::move(*plan), execution);

auto retry = TransferPlan::resume(std::move(report.checkpoint));
```

Plans own configuration strings/paths; source body buffers are borrowed only during one pipeline step. Execution callbacks,
cancel state, the coroutine allocator and an optional custom TLS context are borrowed through task completion. The plan
factory allocator must outlive every plan, checkpoint and kept-path value derived from it. Progress observers run
on the owner fiber, cannot re-enter the same transfer, and may request cancellation through the normal token. The report
distinguishes published, resumed, restarted and retained-partial outcomes. Concrete names may tighten before API freeze, but
the transaction and resume safety contract cannot be weakened.

## Owner 七问（AUDITED）

1. Third-party: no new product dependency is admitted; fixed curl 8.22.0 is linked only into the opt-in P1 peer executable,
   while libcurl/Beast and inference engines remain semantic/reference evidence for the shipped factor.
2. Capability: eight general transfer families are admitted and absent/partial; F02--F05 are the complete finite repair set.
3. Reuse: all wire, TLS policy/context, file, filesystem, allocation, timing, cancellation and scheduling primitives come from lower factors;
   only the transfer state machine is local. Concrete lower-factor gaps trigger lawful reopen rather than duplication.
4. Performance: P1 256 MiB bounded transfer + interrupted resume; compare equal durability/checksum/buffer ownership or use
   OS-limited consumer budgets. Raw copy without flush/atomic publish is not a peer. A pre-candidate mechanism replay found
   that the old 256 KiB default paid one io_uring completion per block and trailed the equal-task POSIX cost floor by
   10--18%; a bounded 1 MiB default reduced both execution orders to about 2--3% without changing ownership or complexity.
   The HTTP holdout uses libcurl's public easy API rather than a benchmark-local raw parser.
5. FFI: `N/A`; no C exports, schema or managed-language wrapper.
6. Delivery: target/header/source and exact-factor graph are `PASS`; source/static/shared relocated product smoke remains in F06.
7. C++ usability/docs: owning plan/checkpoint/report and the user guide are implemented. Red-team removed the final
   buffer-fill escape hatch: `keep_partial()` now returns a move-only owning path value without allocating or copying.

## Bounded red-team receipt

- a resumed HTTP attempt now starts from the original stable source URL, not a transient or signed redirect target;
- 206/416 resume responses must repeat the checkpoint ETag/Last-Modified; an omitted validator is a mismatch, not consent;
- unsupported absolute redirect schemes are rejected instead of being interpreted as same-origin relative paths;
- `keep_partial()` closes and disarms cleanup before transferring its existing path storage into
  `KeptTransferPartial`; close failure leaves the checkpoint owning the artifact;
- arithmetic, byte budgets, old-target preservation, callback borrowing and teardown were replayed against the finite
  F02--F05 set. No benchmark-only metadata, compatibility overload, pseudo C ABI or second protocol/filesystem primitive
  remains. Windows/MSVC, WSL/GCC 14 `-Werror` and macOS/AppleClang targeted correctness plus header isolation pass.

## Candidate readiness（`CANDIDATE`）

有限 F01--F06 产品形状在 `18bf62f81a91975e0776439d88fca3891f10e560` 完整，tree 为
`98632557e0237bac0bed3ca5521e59420d89176e`；精确 Git archive SHA-256 为
`329DA82D0438AA24ACB90A36DE397475A16A8FABAB65E6040F3DC7DABDE4E3DB`。`io_transfer` 自身最后一次
产品变化在 `18d6d6b9`；后续候选仅吸收同轮冻结的 HTTP/TLS transport 修复。当前 HEAD 相对候选的
`include/src/cmake/third_party/samples` 差异为空，文档状态提交没有改变产品。

| profile / shape | result |
|---|---|
| W1-WIN / MSVC 19.50 / exact-product shared | transfer correctness 与逐头 `PASS`；缺失 Windows OpenSSL SDK 时 TLS `ON` 被配置期明确拒绝，合法 `OFF` 路线通过 |
| W1-WIN / exact candidate | `hyle::io` 源码嵌入、relocated shared、relocated static 消费者均实际构造 `TransferPlan` 并 `PASS` |
| W1-WSL / GCC 14.2 / exact archive | provider-on OpenSSL、transfer correctness 与逐头 `PASS` |
| W1-WSL / exact archive | ASan、UBSan、TSan（`setarch x86_64 -R`）下 transfer correctness 与逐头均 `PASS` |
| MacBook Air `Mac16,12` / AppleClang 21 | clean exact commit、OpenSSL 3.6.1、transfer correctness 与逐头 `PASS` |
| Mac mini `Mac16,10` / AppleClang 21 | exact archive、OpenSSL 3.6.3、transfer correctness 与逐头 `PASS` |

两台 Mac 的 OpenSSL deployment-target linker warning 是已知环境差异；真实 transfer 运行通过。远程/WSL
副本只用于验证，没有反向覆盖 canonical Windows checkout。构建、sanitizer 与三交付完成后，候选 API、
algorithm、buffer/durability contract 和 [`io-transfer-p1-v1`](../../../../bench/contracts/io-transfer-p1-v1.json)
从此不可变。

## Current decision

`VERIFIED`。F01--F06 的产品、P0、三平台 correctness、exact sanitizer、source/static/shared、逐头、
组件图、易用性/文档与 bounded red-team 已通过。唯一 P1 在 W1-WSL clean EXT4 上使用 source-pinned
libcurl 8.22.0 完成：256 MiB verified atomic local publish 为一个顺序 `WIN`、一个顺序
`INCONCLUSIVE`，64 MiB interrupted HTTP Range resume 双顺序 `WIN`；四行固定 5% 非劣门及全部 controls
均为 `PASS`。严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、`provenance=EXACT`；完整 receipt
见 [`benchmarks-io.md`](../../evidence/performance/benchmarks-io.md) 和
[`io-transfer-p1-v1-18bf62f8`](../../../../evidence/performance/io-transfer-p1-v1-18bf62f8/)。七门已有可复核
证据，下一法定动作仅为终局限制、重开条件、LOC 与冻结确认，不再运行产品测试或 P1。
