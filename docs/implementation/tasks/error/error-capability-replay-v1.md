# Error 能力完整性存量回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)回放到已经 `CLOSED` 的
> `error-v4`。这是 product-non-mutating 的存量审计，不把新版方法伪造成旧轮当时已经执行过，也不
> 替代 [`module-status.json`](../../module-status.json) 的当前状态。

## 回放合同

- component / replay：`error` / `error-capability-replay-v1`；
- reviewed state / round：`CLOSED` / `error-v4`；
- closure task：[`error-v4-resume`](error-v4-resume.md)；
- candidate / tree / end / registry anchor：
  `9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9` /
  `a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d` /
  `26679f1b2a1987b39cdf302eb95ed3e5fefeb168` /
  `26679f1b2a1987b39cdf302eb95ed3e5fefeb168`；
- replay base：`c47af6965ae83139d4cdebf71f6b70ee93d04daf`；reviewer：Codex；日期：
  2026-08-28；
- normal consumer：HYLE native C++ 模块和 source/static/shared SDK 用户，把可恢复运营失败作为值在
  函数、线程和模块间构造、传播、增加语义上下文、观察并最终销毁；
- tier / instrument：`P1 / closure-perf-v1 / 5%`。产品、source-consumer contract、peer 和 artifact
  路径均未漂移，本回放 timing budget 为 0；性能只继承同一候选已签署的四个 workload，不制造新
  benchmark 姿势；
- 调查预算：30 分钟；一个 fresh Windows Error-only build、一次当前 source/static/shared selected-
  component smoke、每族最多两个预钉死 strong peers；
- strong peers：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 的
  `Status/StatusOr/StatusBuilder`，Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的
  `Expected/result/rich-error` 家族；C++23 `std::expected` 是语言合同，不充当第三个性能 peer；
- 范围：自顶向下能力族、全部现有 public surface、正常失败旅程、真实消费者、复杂度/资源、依赖、
  三种交付、易用性和负空间；
- 非目标：修改 Error 产品、重跑纳秒性能、替 IO/Compute/Obs 等未收口 owner 签结论、恢复 C/Native
  FFI，或因参考库 API 数量扩面。

候选到 replay commit 的 `include/hyle/error/**` 与 `src/error/**` 为零 diff。当前 architecture、user
guide 和 performance evidence 已准确描述 error-v4；更早 task 中的 C handle、旧性能合同和旧状态均
有 `historical/superseded` 标记，不构成现行产品提示。

## 自顶向下 capability-family map

| Family | 已准入用户旅程与来源 | 必须覆盖的变体、生命周期和边界 | 当前路线 | Coverage / action | 缺失边或负空间 |
|---|---|---|---|---|---|
| canonical recoverable failure value | Platform/Mem/Obs/Sched/Graph/Compute 等真实 consumer；Abseil/gRPC taxonomy | OK、16 个 canonical error、code-only/owning message、分类/比较/复制/移动/销毁；无失败 value 的操作返回 `Status` | `StatusCode`、one-word `Status`、factory/predicate/name | `COMPLETE / KEEP` | 正常 empty/pending/closed 用 optional/enum；程序不变量走 Core fatal assert，不把所有分支错误化 |
| value-or-error result | 35 个 production 文件的 value-returning journey；C++23 `expected`、Abseil/Folly | value/error、`void`、default/value/error construction、observers、monadic composition、move-only value、非 OK error-arm invariant | `Result<T> = std::expected<T, ResultError>` 与 `make_unexpected` | `COMPLETE / KEEP` | `Result<T>{}` 是 value-initialized success；不存在第三个 empty 状态 |
| semantic diagnostic context | 多层 Compute/Graph/Sched/Mem 传播；Abseil Builder/context practice | code-only/message leaf、空 context、lvalue COW、unique rvalue chain、alias、OOM、native detail 保真 | `Status::WithContext` 的扁平渐进表示 | `COMPLETE / KEEP` | unique chain 总 O(final bytes)；刻意保留每层 lvalue snapshot 可 O(k²)，guide 已明确 |
| machine-readable native failure | errno/Win32/HRESULT 和 Vulkan/CUDA/NVRTC 真实边界；Abseil errno、Boost.System domain/value | portable code + 至多一个稳定 domain/int64；立即捕获、跨 host 分类、UTF-8/fallback、OOM；owner 可覆盖 operation-specific 语义 | 三个公共 OS factory/classifier + `Status::FromNative` + owning backend mapper | `COMPLETE / KEEP`（当前已准入 domain） | custom category 留在 owner typed error 或边界转换；WinSock 是 IO-owned future trigger，不能猜成 Win32 |
| propagation and composition | Platform/Mem/Graph/Compute 真实调用帧；标准 expected 与成熟 return macros | Status/`Result<void>` 早退、`Result<T>` 取值早退、表达式一次求值、lvalue copy/rvalue move、Status/Result 外层返回 | 两个 `HYLE_*_OR_RETURN` 宏、标准 monadic API、显式 `WithContext` | `COMPLETE / KEEP` | 声明型 assign 宏是多语句，if/loop 必须加 braces；不自动记录 source/log/context |
| first-error, all-error and discard policy | Platform teardown、Mem cleanup；基础批处理/并行失败旅程 | 保留首错、显式忽略、收集全部错误时的顺序/上限/OOM/ownership 由调用任务决定 | `Update`、`IgnoreError`；all-error 使用 owner 的 caller-owned collection | `COMPLETE + ROUTED / KEEP` | 不提供无界公共 `MultiError`；出现两个模块共享同一 bounded aggregation 合同时重开 |
| human-readable observation | 用户边界、测试与未来 Obs adapter；Abseil formatting | code/message 分片、caller-buffer size query/truncation/NUL、UTF-8 byte boundary、opt-in owning string | `message`/name、`ToString`、`hyle/error/format.h::to_string` | `COMPLETE / KEEP` | Status 不自动 log；structured fields 交给 Obs；截断可能切开 UTF-8 code point，guide 已提示 |
| lifetime, concurrency and allocation failure | cross-thread handoff、shared SDK、OOM 基础故障路径；Abseil intrusive rep | 8B carrier、OK/code-only 零分配、immutable shared rep、retain/release、unique final destroy、同对象写同步、OOM 不伪装 OK | tagged inline word + one immutable heap block | `COMPLETE / KEEP` | 不承诺跨版本/工具链稳定 C++ ABI；matching header/artifact 和完整 rebuild 是交付前提 |
| borrowed success value | 基础库 view/reference journey；Abseil `StatusOr<T&>` 与 C++23 restriction | non-null borrow、nullable pointer、range/text view、owner lifetime 和 monadic use 都需显式 | `Result<reference_wrapper<T>>` / pointer / span / string_view | `ROUTED / KEEP route` | 当前没有专用 `Result<T&>`；多个真实 API 证明 wrapper 破坏 lifetime/易用性时才重开 |
| typed/rich failure, cause and provenance | model parser/backend 等 owner-specific future journey；Abseil payload/source、Folly rich error | schema、多个 typed fields、cause traversal、serialization、DSO lifetime、OOM、source/backtrace policy | owner typed error 在模块内保真，边界按需转 canonical/native；Obs/crash 负责诊断 provenance | `ROUTED / OUT_OF_SCOPE` | 不把 payload map、cause list、source location、backtrace、exception hierarchy 或自动 logging 常驻每个 Status |
| protocol state, cancellation machine and foreign runtime | IO/Sched 正常协议；当前无具名 managed consumer | pending/ready/closed、cancel owner、callback/handle lifetime、FFI unload 与版本合同 | owner enum/state machine；Native FFI 按全局准入策略为 `N/A` | `OUT_OF_SCOPE / REJECT from Error` | `Cancelled` 只是 canonical failure code；没有 C 产品或预埋动态符号 |

所有当前 `MUST` 都为 `COMPLETE` 或有具名且可发现的完整 `ROUTED` 路线，不存在
`PARTIAL + KEEP`。routed 只签署 Error 消费者能走到正确 owner，不替对应 owner 的实现收口。

### WinSock/IO 边界裁决

Windows IOCP completion 当前故意使用与 io_uring 对齐的紧凑“成功值或负 native code”协议；把每个
completion 包成 owning Status 会改变热协议形状。IO 的 cold factory/control seam 仍有 bool/sentinel
采用债务，已在旧 error ledger 登记为 `IO-ADOPT-01`，应由 IO 收口裁决其公开 Result/Status 旅程。
若该旅程需要 WSA code 的机器恢复，届时触发独立 `WinSock` domain/factory；当前不得把
`WSAGetLastError` 猜成 `GetLastError`，也不得为未冻结的 IO API 提前扩 Error。

## 自底向上 public-surface trace

生产反向计数是对 `include/hyle`/`src` 排除 Error 自身、tests、bench 和 third-party 后的 lexical
scan；它用于证明真实采用面，不冒充动态 profile。

| 当前入口 | Capability / demand | 生产反向证据 | 裁决 |
|---|---|---|---|
| `StatusCode`、code name、factory、predicate | canonical failure value | `Status` 共 1133 处 / 57 个 production 文件；canonical taxonomy 全表有 round-trip/factory tests | `KEEP`；标准完整 taxonomy 不按单个 code 的调用数裁剪 |
| `Status` ownership、message、native、comparison/swap | recoverable carrier/lifetime | Platform/Mem/Obs/Sched/Graph/Compute 等跨模块使用；copy/move/thread/OOM/8B tests | `KEEP` |
| `NativeError[Domain]`、`FromNative` | backend machine recovery | 10 处 / 5 个 production 文件；CUDA/Vulkan 实际 mapper | `KEEP` |
| errno / Win32 / HRESULT classifier+factory | OS boundary | 分别 9/7、7/7、1/1 处/文件；Platform 已复用，cross-host/OOM/UTF-8 tests | `KEEP` |
| `WithContext` | semantic propagation | 29 处 / 10 个 production 文件；Graph/Compute/Sched/Mem/Platform | `KEEP` |
| `Result<T>` / `Result<void>` / `make_unexpected` | value-or-error | `Result` 184 处 / 35 个 production 文件；helper 2 处 / 2 文件，另有标准 `std::unexpected` 直接写法 | `KEEP` |
| `HYLE_RETURN_IF_ERROR` | value-less early return | 9 处 / 3 个 production 文件；Status 与 Result 外层、lvalue/rvalue tests | `KEEP` |
| `HYLE_ASSIGN_OR_RETURN` | value early return | 24 处 / 6 个 production 文件；声明、已有 lhs、嵌套与 value-category tests | `KEEP` |
| `Update` / `IgnoreError` | first-error / explicit discard | 11/2 与 8/4 处/文件；Platform teardown、Mem/Compute cleanup | `KEEP` |
| `ToString` / opt-in `to_string` | human output boundary | 当前无非 Error production 调用；76 个 Error test cases 中覆盖 caller buffer、long/NUL/UTF-8/owning 路径，user guide 给出选型 | `KEEP`；这是成熟且零常态 metadata 的完整输出边，不是 benchmark-only API |
| `HYLE_FOREACH_*_STATUS_CODE`、`HYLE_ERROR_API`、`detail::ResultError/FailReturn` | header/binary implementation seam | Error TU、tests 和 inline Result 实现使用；未作为用户任务文档化 | 保持 installed implementation seam；不把它们宣称为独立用户能力 |

没有发现 Error 自身的 TLS last-error、全局 handler、隐式日志、异常传播、动态 category registry 或
benchmark-only metadata。中央 component graph 与真实 target 均保持 `error -> PUBLIC core`；
`result.h` 对 Core compiler/preprocessor seam 的公开依赖因此可由独立 component closure 正确交付。

## 当前验证

| Gate | 当前结果 | 证据边界 |
|---|---|---|
| product identity | `PASS` | candidate→`c47af696...` 的 Error product path 零 diff；candidate tree 与 registry 一致 |
| fresh Windows Error-only Release/Werror | `PASS`，18/18 | VS 18 2026 / MSVC 19.50；配置只解析 `core,error`，含 76-case Error binary、header self-contained 与公共 tooling gates |
| source consumer | `PASS`，1/1 | generated parent `add_subdirectory()`，只请求/链接 `hyle::error` |
| relocated shared SDK consumer | `PASS`，1/1 | exact-version package，Status heap owner 在 `hyle_error` 内完成 retain/final release |
| relocated static SDK consumer | `PASS`，1/1 | exact-version package，安装闭包只含 Core + Error headers/artifact |
| architecture/user/performance docs | `PASS` | 当前三页与 error-v4 对齐；历史页明确是追溯事实 |
| P1 engineering performance | inherited `PASS` | 同一 candidate/contract/peer；四个 source-primary workload 的四 strata 全过 5% NI，见 [performance evidence](../../evidence/performance/benchmarks-error.md) |
| new nanobenchmark | `N/A` | replay timing budget 为 0；没有 identity/contract drift 或产品 trigger |

本回放没有重跑 Linux/macOS，也不把候选历史三平台测试升级成新的 Formal certificate；本轮只复验当前
Windows 产品和三种当前交付。Formal 仍独立 `PENDING`。

## 裁决与下一动作

- 产品 verdict：`CONFIRMED`；`error` 保持 `CLOSED`，实现改动为 0，没有 `PRODUCT_TRIGGER`；
- 方法 verdict：旧 v3 ledger 已经很宽，v4 又纠正并恢复了真实 consumer 性能，但旧 `KEEP/DEFER`
  没有独立 coverage 轴。本页把取得、正常操作、组合、失败、teardown、边界与负空间重新签成
  `COMPLETE/ROUTED/OUT_OF_SCOPE`，不倒写历史过程；
- 功能 verdict：作为全库共享的最小 recoverable-failure spine，当前功能完整；“最小”不是缺少全部
  typed error 能力，而是 owner typed error 能无损工作、到公共边界有自然 route；
- 性能 verdict：保留 error-v4 的 P1 engineering `PASS`，只覆盖四个冻结 consumer workload；不宣称
  all-operation/cross-platform Formal SOTA。旧 fixed-backing copy 约 8% 慢仍是无 material consumer 的
  mechanism diagnostic，不改变产品；
- 下一动作：项目采用清单进入 `util_core`。IO 收口必须处理 cold bool/sentinel 与 WinSock route，
  Compute/Obs 收口分别裁决 backend translator 和 structured observation；这些都不据此重开 Error；
- 重开条件：真实 consumer 无法用 canonical/native/context 或 owner typed route 完成恢复；公开
  Status/Result/组件/交付合同改变；supported compiler 破坏 tagged/lifetime codegen；或 strong peer 在
  同语义正常 source consumer 上证明更好复杂度档/material 优势。peer API 并集、LOC、单个新 backend、
  零调用互补入口或 isolated microbench 不单独触发重开。
