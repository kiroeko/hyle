# Error v5 消费者能力补全

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本轮不是无新事实复查。Error v4 关闭后，全局能力展开与真实 IOCP 消费路径共同暴露了一个已存在的
> 原生错误类别缺口；成熟 peer 与普通系统集成 journey 还暴露了三项需要有限裁决的 API 缺口。本轮只
> 重开这些受影响能力，不重做已经签署的 Status 表示、传播宏或全部性能研究。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`error` / Error / `error-v5-capability-completion` / `REOPEN`；
- baseline：`afdb6b81ab4e8e3ad9d790be87a80dc9242c1cec`；前轮：`error-v4`；
- trigger：`ERROR-CAPABILITY-2026-09-01`；
- `ERR5-R1`：Windows IOCP 实现反复读取 `WSAGetLastError()`，但 Error 只有 POSIX errno、Win32、
  HRESULT 与若干计算后端 domain。现有 IO 代码只能把 WinSock 数值压成无类别负整数；这满足“真实
  consumer 暴露 missing-capability defect”的重开条件；
- `ERR5-R2`：普通 registry/cache/parser journey 需要“成功时非空借用、失败时传播 Error”的形状。
  当前 guide 只能要求用户写 `Result<std::reference_wrapper<T>>`，公开类型没有直接表达 borrow 的
  ownership 与 `value()/operator*()/operator->()` 语义；
- `ERR5-R3`：边界适配器需要把运行时诊断参数附到 Status，同时维持 HYLE 默认不抛异常。当前只有
  literal/string 路径，没有统一的 no-throw 动态格式化入口；
- `ERR5-R4`：标准库与第三方 adapter 普遍交付 `std::error_code`。HYLE 文档要求每个调用者自己比较
  category 并映射，形成重复、容易丢失类别的冷边界代码；
- capability provenance：长期准入与归属规则见[能力展开与组件质因数分解](../../../architecture/capability-factorization.md)；
  本卡的 `ERR5-R1..R4` 是该轮完整有限证据。真实 WinSock 消费位于 `src/io/proactor.cc`，其下游迁移仍由 IO owner
  在自己的轮次完成；
- 初步受影响面：Error 的 borrowed-result、native-domain/translator、冷路径动态诊断与标准错误适配，
  以及 architecture/user guide、correctness、三种 C++ 交付和定向 P1 证据；
- 初步依赖结论：Error 保持只依赖 Core，不引入 platform、obs、fmt 或其他第三方产品依赖；OS 文本
  查询与 C++ 标准库 adapter 是 Error 自己的边界机制；
- 暂定 tier：`P1`。现有 Status/Result 热路径仍复放 Error v4 的正常 consumer 合同；新的 borrowed
  result 属可能高频 primitive，WinSock、`std::error_code` 与动态诊断属于冷边界并以复杂度、allocation
  和异常遏制为主；
- 暂定 non-goals：异常框架、任意 payload/cause/source-location、MultiError、运行时 domain registry、
  日志、Native FFI/C API、稳定 C++ ABI、IO 模块迁移、相邻模块收口、公共性能仪器改造和 Formal 认证。

当前仅登记 `REOPENED` 新事实；closure card、API freeze 与有限 `FIX` ledger 尚未冻结，不授权产品修改。

## Closure card（`SCOPED`）

- component / owner / round / kind：`error` / Error / `error-v5-capability-completion` / `REOPEN`；
- baseline / trigger：`afdb6b81ab4e8e3ad9d790be87a80dc9242c1cec` /
  `ERROR-CAPABILITY-2026-09-01`；
- real journeys：IOCP 在失败点立即捕获并分类 `WSAGetLastError()`；registry/backend lookup 返回非空
  mutable/const borrow 或 Status；模型/缓存/设备诊断把运行时路径、尺寸和设备号放进 failure text；
  filesystem/第三方边界把 `std::error_code` 单向降为共享 Status；既有 graph/sched Status/Result owning
  handoff 保持原语义和代码形状；
- scope：补齐固定 WinSock domain、纯 O(1) classifier 与 no-throw factory；让 `Result<T&>` 直接表达
  非空 borrow；提供无日志/无 source-policy 的有界 `format_status`；提供 generic/system/custom
  `std::error_code` 冷适配与 caller-owned canonical override；更新 Error architecture/user/performance 与
  package journey；
- non-goals：异常框架、任意 payload/cause/source-location、MultiError、运行时 domain registry、反向
  `Status -> std::error_code` 伪 round-trip、日志、IO 产品迁移、相邻模块收口、Native FFI/C API、稳定
  C++ ABI、公共仪器重写和 Formal；
- dependencies / consumers：产品依赖仍为 `PUBLIC core` 且无第三方产品依赖。Abseil/Folly 只作 peer，
  `<format>`/`<system_error>` 是 opt-in C++ 标准头；直接消费者是 Platform/Mem/Sched/IO 与未来 registry/
  backend map，当前只回放 Error public journey 和已冻结的 source-consumer owning handoff；
- tier / margin：`P1` / 5%。`Result<T&>` 是可能高频 primitive；既有 owning carrier 重放当前四个正常
  consumer workload。WinSock、动态诊断和 standard-error adapter 是冷边界，分别要求 O(1) classifier、
  有界栈工作集、至多一次 Status backing allocation、无隐式日志/锁/registry；
- normal metadata：owning `Result<T>` 的类型、layout 与 codegen 必须不变；borrow carrier 只为选择它的
  返回类型支付一个 pointer、一个 Status error arm 和 discriminator，不分配、不加全局状态；code-only/
  OK Status 不增加字段；格式化固定使用 1 KiB caller-side stack scratch，溢出带显式 truncation marker；
- budgets：能力/API/实现 4 小时，正确性/交付/文档 2 小时，性能资格 2 小时，红队 30 分钟；最多一次
  frozen workload 复现、一次 codegen/profile 机制检查、一次性能导向产品修正和一次重跑；仪器若不能
  给出预声明 verdict，记录独立 blocker，不在 Error 轮扩建统计系统；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 `closure-perf-v1`，P0 结构检查沿用
  `closure-correctness-v1`；
- strong peers：Abseil `5e12b490a2009cb223dd75a70fd0213847b82f9c` 的 `Status/StatusOr<T&>` 与 errno
  taxonomy；Folly `176c088840724715b2c3c6211b2b347611676002` 的 `result<T&>` deep-const/lifetime
  设计。性能同合同 peer 只用 Abseil；Folly 不因 rich-error 语义差异进入同合同 timing；
- inference provenance：llama.cpp `9d817213a0975020775efe6c458822616826f376`、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`，均为 2026-09-01 clean fast-forward reference；
- prototype provenance：WinSock 为 `REAL_CONSUMER`；borrow 为跨 Container/Util/Platform 后确认的
  `NORTH_STAR_JOURNEY + PEER_CONSENSUS`；动态诊断和 standard-error 为 llama/ggml 与普通 C++ 系统集成
  `REAL_CONSUMER`，不是 benchmark 需求；
- performance workloads：更新 Abseil identity 后重放 v4 四个 source-primary workload；新增一个
  fixed-table registry lookup（命中返回非空 borrow、缺失返回 code-only Status）的
  `SAME_CONTRACT` consumer，双方共享 lookup 算法、命中分布、lifetime 与 observable，5% 非劣；固定
  backing copy 只由既有 owning handoff 的 material 结果解释，仍不得单独改变产品；
- correctness gates：正常/错误、零码/未知码、WinSock taxonomy、UTF-8/numeric fallback、format 成功/
  truncation/custom formatter failure/OOM、borrow mutable/const/deep-const/temporary rejection/转换/copy/move/
  monadic/传播宏、custom error category/throwing message、溢出与 Result 非 OK invariant；并发只验证共享
  Status immutable handoff，cancel/close 与 real ISA 不属于本轮；
- delivery gates：公开头 self-contained，Error component-only source test；source embedding、installed
  static/shared C++ consumer 与 relocation/runtime staging；中央 factor/façade/include-owner 合同；
- output：有限 capability ledger、冻结 API、实现/测试/guide/architecture/performance evidence、候选与
  manifest、独立红队、四项 LOC 和 terminal receipt；
- stop：四项 `MUST` 均 `COMPLETE` 或完整 `ROUTED`，owning Result 无漂移，正确性/三交付/docs/red-team
  全通过且所有 P1 workload 通过 5% 门即 `CLOSED`；任何 correctness、交付或稳定 material P1 failure
  在一次机制检查后仍存在即 `BLOCKED`，不新增第五种测试姿势。

当前为 `SCOPED`。上述范围、预算、peer、成本与验证合同冻结；产品代码仍须等待 public-surface 反向
追踪、capability ledger 和 API freeze 进入 `AUDITED`。

## Capability-family coverage map（`AUDITED`）

| Family | admitted journey / evidence | required variants | current route | state | missing edge / decision |
|---|---|---|---|---|---|
| canonical Status value | portable recovery、owning diagnostic、跨线程 handoff | full taxonomy、code/message/native、copy/move/OOM | one-word `Status` + immutable backing | `COMPLETE` | `KEEP`，不重做表示 |
| owning Result | value/move-only/void/error、短路与 monadic composition | default、observer、non-OK error invariant | exact `std::expected<T, ResultError>` | `COMPLETE` | `KEEP`，类型与 codegen 必须不变 |
| fallible non-null borrow | registry/backend/map lookup | mutable/const、temporary rejection、deep-const、转换、monadic/宏传播 | 只能手写 `reference_wrapper` | `PARTIAL` | `FIX` 为 `Result<T&>`；nullable 仍用 pointer，range/text 仍用 view |
| semantic context / dynamic diagnostic | 路径、尺寸、设备号进入失败文本 | UTF-8/NUL、truncation、formatter failure、OOM | `WithContext` 完整；动态值无 no-throw route | `PARTIAL` | `FIX` 一个无日志副作用的 bounded `format_status` |
| native/system translation | errno、Win32、WinSock、HRESULT、backend 与 standard error | zero/unknown、跨 host、category override、立即捕获 | errno/Win32/HRESULT 完整 | `PARTIAL` | `FIX` WinSock 与单向 `std::error_code` adapter；不保留 custom category pointer |
| propagation / aggregation / human output | 多层 return、首错、discard、最终文本 | lvalue/rvalue、Status/Result、buffer/owning string | 两个宏、`Update`/`IgnoreError`、`ToString`/`to_string` | `COMPLETE` | `KEEP` |
| async protocol / typed rich error / fatal / FFI | pending/cancel、owner schema、invariant、foreign runtime | lifecycle 与 policy 由 owner 决定 | Sched/IO、owner typed error、Core、future named FFI | `ROUTED` / `OUT_OF_SCOPE` | 不把协议状态、payload map、fatal 或伪 C ABI 塞入 Error |

### Capability / gap ledger

| ID | class | exact demand / source | normal shape | owner / failure / lifetime | complexity / cost | state | action | verification |
|---|---|---|---|---|---|---|---|---|
| `ERR5-01` | `MUST` | IOCP `WSAGetLastError`；`ERROR-A-07` | typed classifier + factory | Error；立即复制 int，Status owning | classifier O(1)；失败冷路一个 Status backing | `MISSING` | `FIX` | table differential、zero/unknown、三平台文本/native |
| `ERR5-02` | `MUST` | foundation registry/backend lookup；Abseil/Folly；`ERROR-A-04` | `Result<T&>` | caller owns referent；carrier borrow；deep const | O(1)、零 allocation/global state | `PARTIAL` | `FIX` | type traits、lifetime diagnostics、macro/monadic、same-contract P1 |
| `ERR5-03` | `MUST` | llama/ggml runtime diagnostics；`ERROR-A-06` | `format_status(code, fmt, args...)` | Status owns bounded text；formatter is adapter boundary | 1 KiB stack、至多一个 Status backing、无 log | `PARTIAL` | `FIX` | exact/NUL/truncation/throw/OOM/codegen structure |
| `ERR5-04` | `MUST` | filesystem/SDK `std::error_code`；`ERROR-A-08` | generic route + explicit canonical override | category borrowed only during call；Status owns result | cold O(message bytes)；custom `message()` may allocate but never escapes exception | `PARTIAL` | `FIX` | generic/system/custom/throwing/override/native retention |
| `ERR5-05` | `MUST` | existing graph/sched Error consumers | unchanged Status/owning Result | existing ownership contracts | existing complexity/layout | `COMPLETE` | `KEEP` | source replay + static asserts |
| `ERR5-06` | `GENERAL` | arbitrary dynamic domains/payloads/MultiError | runtime registry or rich schema | would add lifetime/registry policy | normal metadata/indirection not justified | `OUT_OF_SCOPE` | `REJECT` | reopen only with named shared schema/consumer |

- existing public-surface reverse trace：`status.h` owns taxonomy, native pair, one-word Status, errno/Win32/HRESULT
  translation and context; `result.h` owns checked expected alias, unexpected helper and two propagation macros;
  `format.h` owns opt-in C++ owning string conversion；`status.cc` is the only compiled representation seam；there
  are no public C headers, C bridges or unadmitted exports；
- syntax/semantic duplicate scan：IO currently duplicates WinSock negative integers but remains an IO migration ticket；
  generic/custom `std::error_code` mapping otherwise exists only as guide prose；Obs formatting has logging-specific
  truncation/record policy and cannot be called by Error without a cycle，so Error owns only Status construction；
- public/internal decision：all four entries solve ordinary downstream tasks and are public；storage helpers, UTF-8
  validation and native description stay source-internal；no new component or dependency edge；
- frozen `FIX` set：`ERR5-01..04` plus their tests、guide、architecture、delivery and declared performance replay；
  no other capability may enter this round；
- adjacent backlog：IO later replaces raw negative WSA outcomes at its public ownership boundary；other modules may
  adopt `Result<T&>` only in their own rounds，so Error does not generate a repository-wide migration diff。

## API / usability freeze

- status：`FROZEN`；
- WinSock：append `NativeErrorDomain::WinSock`，add
  `status_code_from_winsock(int) noexcept` and
  `status_from_winsock(int, std::string_view context = {}) noexcept`。Zero is OK；unknown stays Unknown；the raw
  integer is retained under the WinSock domain。Future fixed domains require a named consumer and Error review；no
  runtime registration or category pointer is admitted；
- borrowed result：`Result<T>` is one public class-template family：owning/void specializations retain the
  `std::expected<T, ResultError>` representation and operations，while `Result<T&>` selects a header-only borrow
  carrier。The family must preserve ordinary template-argument deduction such as
  `template<class T> f(const Result<T>&)`；it must not route through a non-deducible selector alias。A borrow
  constructs only from a compatible lvalue or unexpected error；an
  explicit default is code-only Unknown，temporaries/null are rejected。Mutable carriers expose `T&`，const carriers
  expose deep-const；a mutable borrow cannot be copied back out of a const carrier。Provide expected-style
  `has_value`/bool、`*`/`->`/`value`/`error`、swap/rebind assignment、compatible const/base conversion、`value_or`
  and `and_then`/`transform`/`or_else`，plus both propagation macros。`transform_error` is not mirrored：changing the
  canonical error type exits HYLE Result，and `WithContext` is the supported in-domain operation；
- dynamic diagnostic：`<hyle/error/format.h>` adds
  `format_status(StatusCode, std::format_string<Args...>, Args&&...) noexcept`。It formats into a 1 KiB caller stack
  buffer，marks truncation explicitly，catches custom formatter exceptions when the toolchain supports exceptions，
  and degrades formatting failure or Status OOM without throwing；it never logs or attaches source location；
- standard error：new opt-in `<hyle/error/system_error.h>` adds
  `status_code_from_error_code(const std::error_code&) noexcept`、
  `status_from_error_code(const std::error_code&, context = {}) noexcept` and
  `status_from_error_code_as(StatusCode, const std::error_code&, context = {}) noexcept`。Generic category maps as
  errno；system maps as Win32 on Windows and errno elsewhere；custom category defaults to Unknown，while `_as` lets
  the owner supply operation-specific canonical semantics。Only generic/system retain stable native numeric detail；
  custom categories are formatted during the call and never stored by pointer；
- ownership/default/misuse：Status owns all retained text；borrow lifetime remains entirely caller-owned and is
  annotated lifetime-bound；pointer Result remains nullable by design and is not silently changed；OK ignores native/
  diagnostic detail；nonzero error + `_as(StatusCode::Ok, ...)` fails closed to Unknown；guide recipes show when to
  choose value、borrow、nullable pointer、view、native adapter and formatted diagnostic；
- delivery/discovery：the new standard-error header is selected and installed as Error-owned public API；WinSock
  symbols and category-neutral error-code part helpers carry `HYLE_ERROR_API`；public `std::error_code` adapters
  classify the caller's standard-library category inline before the DLL seam，so no cross-DSO singleton address is
  treated as stable identity；format and borrow stay header-only and use the same source/static/shared C++ spelling；
- freeze identity / reviewer：本 task 的 `AUDITED` transition / primary owner；
- allowed implementation freedom：private storage/helpers、switch layout、stack appender and inlining may change，but
  no public name/semantic、owning Result representation/cost、dependency or frozen workload may change without
  returning to `AUDITED`。

### AUDITED freeze correction：generic deduction

首次编译冻结的 v4 source-consumer replay 时发现：原拟实现把 `Result<T>` 改成
`ResultSelector<T>::type` alias；虽然 non-reference instantiation 仍与 `std::expected` 类型相同，C++ 的
template deduction 无法反解 nested dependent type，导致普通 `f(const Result<T>&)` 和既有 carrier trait
无法推导 `T`。这是 public C++ usability regression，不是 benchmark 胜负，也不是兼容性要求。

因此候选前修正上述 freeze：`Result<T>` 改为统一 public class template；owning/void route 以零附加成员的
`std::expected` 表示实现，reference route 以 pointer + `ResultError` discriminator 实现。Owning 的 layout、
allocation、observer、monadic semantics 与 generated hot path 必须保持；exact type identity 不再作为产品
需求。新增 deduction/monadic/comparison compile-time tests，并继续用四个既有 owning consumer workload
裁决成本。该修正没有新增能力、workload 或依赖，`FIX` 集仍为 `ERR5-01..04`，状态保持 `AUDITED`。

同一 delivery smoke 随后证明 MSVC consumer 与 HYLE DLL 可各持一个不同地址的 standard category
singleton；把 category comparison 放在 `status.cc` 会把 consumer 的 generic error 错判为 custom。
API spelling 不变，冻结实现边界修正为 caller-side inline classification + category-neutral compiled parts；
custom pointer 仍只在调用期间借用且不进入 Status。这是 shared-delivery correctness 修复，不改变能力集。

## Owner 六问（AUDITED 直答）

1. **第三方产品路径：**没有。Error 只依赖 Core；standard library/OS APIs are mechanisms。Abseil/Folly/llama/
   ggml are clean locked references only and do not enter targets、package or public types beyond standard C++ types。
2. **推理引擎能力：**对已准入 failure journey，`ERR5-01..04` 完成后不存在未处置 `MUST`；model/token/
   backend policy、protocol pending、typed backend schema and exception-owning boundaries remain correctly routed。
3. **拟真性能：**当前 verdict 仍 `PLANNED`。旧 v4 only proves its exact candidate/four workloads；v5 must replay
   them and qualify the new registry-borrow SAME_CONTRACT row on W1-WIN at 5%。No all-operation/cross-platform SOTA
   claim is permitted。
4. **C ABI / Native FFI：**current policy is no admitted slice；no public C headers、bridge source、managed sample、
   option or undecorated C exports。Private OS C ABI does not create a product surface；expected verdict `CLEAN`。
5. **Source/static/shared：**the previous shape passed，but new headers and compiled symbols make all three journeys
   pending on the v5 candidate；Error remains one compiled factor behind the same `hyle::error` façade。
6. **C++ 易用性与文档：**the frozen API answers value vs borrow vs nullable/view，native vs standard adapter，and
   literal/context vs dynamic text without exposing benchmark/internal seams。Current guide/architecture are stale on
   WinSock and reference-wrapper advice，so verdict remains pending until `ERR5-01..04` documentation lands。

当前为 `AUDITED`：四项有限 `FIX`、公开 API、依赖、常态成本、正常 workload 与停止线全部冻结；现在才
授权实现产品代码。

## Implementation and candidate freeze

- `ERR5-01 COMPLETE`：新增固定 `WinSock` native domain、O(1) classifier 与立即捕获 raw code 的
  no-throw Status factory；零码、已知 taxonomy、未知码、非 Windows numeric fallback 均有测试；
- `ERR5-02 COMPLETE`：`Result<T>` 成为可正常模板推导的统一公开 class-template family；owning/void
  仍以 `std::expected<T, ResultError>` 表示且无附加状态，`Result<T&>` 交付非空 borrow、deep const、
  lvalue-only construction、expected-style observers/monads/comparison 和两种传播宏；
- `ERR5-03 COMPLETE`：`format_status` 使用固定 1 KiB caller stack scratch，有明确 truncation marker，
  formatter exception 与 Status OOM 均遏制为 no-throw failure，且不记录日志或 source policy；
- `ERR5-04 COMPLETE`：新增 generic/system/custom `std::error_code` 单向 adapter 与显式 canonical override。
  标准 category 在 consumer 侧 inline 分类，跨 DLL 只传 category-neutral parts，避免把标准库 singleton
  地址误当稳定 ABI identity；custom category pointer 不进入 Status；
- `ERR5-05 KEEP`：既有 owning Status/Result consumer 形状、one-word Status、Core-only 依赖、无 C/Native
  FFI 产品面保持；v4 四个 owning workload 纳入 v2 replay；
- public guide、architecture、component header inventory、source consumer sample 和 performance contract 已与
  最终 API 同步；候选前 diagnostic probe 仅确认仪器可运行，不作为 qualification verdict。

候选身份：

- product commit：`d61f0710ec0e465b31d458257e2587fdfe37ed52`；
- product tree：`305b2cb5947627dfb5b9697c939205b6c26cc6b7`；
- `AUDITED -> CANDIDATE` previous registry：
  `C:\hyle-scratch\error-v5\module-status-audited.json`；
- candidate 前三平台 Error Release/WERROR shipping tests 均为 `19/19 PASS`；W1-WIN、W1-WSL、M1-MAC
  的 component-only source、relocated installed shared 与 relocated installed static consumer 均 `PASS`；
- 此 identity 后只允许补充验证证据、task/registry/receipt 和事实文档。任何产品 API、表示、算法、依赖、
  workload verdict 语义或交付合同改动都退回 `AUDITED` 并产生新 candidate。

当前为 `CANDIDATE`。下一步只从 clean candidate checkout 执行冻结的 W1-WIN P1 qualification、独立红队、
四项 LOC 与终态门；不能再增加能力或第五种 benchmark 姿势。

## Candidate verification

### Correctness、delivery 与结构成本

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| Release/WERROR Error shipping tests | `19/19` | `19/19` | `19/19` | `PASS` |
| public-header/self-contained compile | `PASS` | `PASS` | `PASS` | `PASS` |
| component-only source consumer | `PASS` | `PASS` | `PASS` | `PASS` |
| relocated installed shared consumer | `PASS` | `PASS` | `PASS` | `PASS` |
| relocated installed static consumer | `PASS` | `PASS` | `PASS` | `PASS` |

三平台均绑定 candidate `d61f0710` 的相同 Error product tree；WSL/macOS 是 disposable native validation
copy，不是产品改动来源。测试覆盖冻结卡所列 normal/error、zero/unknown、WinSock taxonomy/text fallback、
formatted exact/truncated/throw/OOM、borrow mutable/const/deep-const/temporary rejection/conversion/monads/macros、
custom/throwing `std::error_category`、Result invariant、copy/move 和 immutable shared Status handoff。取消/关闭、
real ISA/backend 与 runtime domain registry 在本模块语义上 `NOT_APPLICABLE`。冷边界结构检查确认 classifier
O(1)、format scratch bounded、无隐式 log/lock/global registry，owning Result 无附加字段，borrow 无分配。

### W1-WIN P1 qualification

- clean candidate：`d61f0710ec0e465b31d458257e2587fdfe37ed52` / tree
  `305b2cb5947627dfb5b9697c939205b6c26cc6b7`；clean Abseil：
  `5e12b490a2009cb223dd75a70fd0213847b82f9c` / tree
  `82ccc3a39ed6f9829391f361063d8177a9039a17`；
- contract `error-source-consumer-v2` SHA-256：
  `61993b4c19750f6d19b9906c06216f4453de10335e8788e13397a23683a526e0`；fresh MSVC 19.50 Release/static、
  portable ISA、LTO off、n=15；全部 required validators/controls `PASS`，未触发 n=31；
- 五个 workload × 四个 order/placement strata 的 5% non-inferiority 全 `PASS`。TaskGraph success combined
  `PARITY`，其最低 CI lower 0.9673；三个 owning failure handoff 与 `Result<T&>` registry borrow combined
  `WIN`，最低 CI lower 分别为 1.1814、1.3534、1.4904、1.6850；
- retained manifest：`C:/hyle-scratch/error-v5/qualify-v2-output/manifest.json`，SHA-256
  `12685d8f398d4f55c9f05c71d54a3f5054ee3b99f21147778dfe91644f8b7eff`；summary/contract-lock SHA-256 为
  `3303032a19b5cf036733ac2627626631f8460147b26432453cb60e5a6421cd45` /
  `04b41774b914124c3189083dee962a46819e8312256bcd5ad51cb2fb5ca10612`；deep verifier 为
  `QUALIFIED`、paths verified、exact provenance、current-policy eligible；
- owning-failure peer 数字有明显 execution-order 摆动；红队因此只接受所有 strata 的保守 NI/最低 CI
  结论，不宣传最大倍数。成功路径亚百分比变化没有触发产品修改。

### Hostile red team 与 owner 六问终答

1. **第三方：`PASS`。**产品闭包仍只有 Core；Abseil/Folly/llama/ggml 只是锁定 peer/provenance，标准库与
   OS API 是机制。没有新增 FetchContent、package dependency 或第三方 public type。
2. **功能：`PASS`。**`ERR5-01..04` 已由真实 IOCP、普通 registry/adapter 与推理诊断 journey 完成，
   `ERR5-05` 保持；未处置 `MUST` 为 0。payload/MultiError/runtime registry、async protocol、fatal、logging
   仍分别拒绝或路由给 owner，没有用 API 数量扩张 Error。
3. **性能：`PASS`。**冷能力通过复杂度/allocation/exception 结构门；可能高频的 owning/reborrow 五个
   同合同 consumer 全部通过冻结的 W1-WIN 5% P1 非劣门。只签该范围，不声称跨平台或全操作 SOTA。
4. **旧 C ABI / Native FFI：`CLEAN`。**产品树无 C header/schema/bridge/managed sample/option；Windows
   candidate DLL 的 22 个 exports 全是 MSVC decorated C++ symbols，没有 undecorated C export。`detail`
   exports 是 header-inline C++ adapter 的 compiled seam，不是 Native FFI。
5. **构建与导入：`PASS`。**中央图真实声明 `error PUBLIC_DEPS core`；selected header 安装与三平台
   source/static/shared relocation consumer 全通过。shared smoke 真实捕获并修复了 category singleton
   跨 image 地址不稳定缺陷，而非只验证 producer build。
6. **C++ API/文档：`PASS`。**公开形状按 value/non-null borrow/nullable pointer/view/native/standard adapter/
   dynamic diagnostic 的正常任务选择；ownership、deep const、temporary、custom category/unload、truncation、
   exception-disabled formatter 陷阱均在 guide。公开 symbol 无 benchmark seam，architecture/user/task/
   performance 各自承担长期设计、当前用法、事实和证据。

红队进一步比较 baseline→candidate 产品 diff、依赖闭包、公开 header、DLL exports、consumer semantic matrix、
四种 placement/order 与 retained bundle；候选后产品 diff 为零，剩余 medium-or-higher finding 为 0。

### Candidate LOC

`python scripts/loc-report.py --module error --commit d61f0710ec0e465b31d458257e2587fdfe37ed52
--self-check --json` 从不可变 tree 复算并通过 cloc 2.08 self-check：

| Metric | code |
|---|---:|
| 全库代码 | 417,715 |
| 自实现包含测试 | 101,020 |
| 自实现不包含测试 | 62,970 |
| Error 实现 | 2,178 |

### Aggregate verdict

| Gate | Verdict | Boundary |
|---|---|---|
| scope | `PASS` | 仅冻结 `ERR5-01..04` 与 `ERR5-05 KEEP`；candidate 后产品未改 |
| architecture | `PASS` | Error 保持 Core-only prime、无第三方产品路径或依赖环 |
| correctness | `PASS` | 三平台冻结风险矩阵与 header contract |
| delivery | `PASS` | 三平台 source/relocated shared/relocated static consumer |
| usability/docs | `PASS` | task-shaped C++ API、选型 recipe 与陷阱完整 |
| red team | `PASS` | 依赖/API/FFI/DLL seam/比较合同/产品 diff；MEDIUM+ 为 0 |
| engineering performance | `PASS` | W1-WIN P1 资格门、五个 workload 全 strata 5% NI |

七项 aggregate gate 全部 `PASS`，五个 `MUST` 均 `COMPLETE`，没有 blocker。Formal 仍为独立 `PENDING`，
不属于工程关闭条件。Registry 使用保存于
`C:\hyle-scratch\error-v5\module-status-candidate.json` 的明确前态执行 `CANDIDATE -> VERIFIED`。

当前为 `VERIFIED`；下一步只写接受限制、重开条件和 terminal receipt，不再改变 candidate product。

## 接受限制与重开条件

接受限制：

- P1 certificate 只覆盖 W1-WIN、冻结的五个 normal consumer、Abseil identity、MSVC 19.50 与 source/static
  benchmark 交付；跨平台性能、全部 Error 操作、端到端推理 KPI、Formal 和通用 SOTA 均未签署；
- `Result<T&>` 是 caller-owned non-null borrow；它不延长 referent lifetime，deep const 也刻意不提供浅 const
  mutable escape。nullable、range/text 和 shared ownership 应分别使用 pointer、view/span 与 owning handle；
- `format_status` 最多保留 1023 字节并显式标记截断；异常关闭构建无法遏制一个违反合同、主动抛异常的
  custom formatter，用户必须提供 non-throw formatter；
- custom `std::error_category` 的 pointer 不保留；category owner 负责 canonical override，且定义该 category
  的 DSO 至少要存活到 adapter call 返回。standard category 只在 caller image 以地址分类；
- HYLE 仍不承诺稳定 C++ ABI 或 Native FFI。header、consumer 和 binary 必须匹配；IO 对 WinSock adapter 的
  adoption 由 IO 自己的轮次完成；
- retained P1 bundle 位于本机 scratch，不是发布 artifact；Formal 继续作为独立 `PENDING` 工作。

只有以下新事实重开受影响能力族：

1. 真实消费者复现 Error correctness/security、ownership/lifetime、OOM/no-throw、API 易用性或缺失能力缺陷；
2. 支持 compiler/stdlib/OS 改变 `std::expected` 表示、reference lifetime diagnostics、standard-category
   image identity、format 或 native translator 语义；
3. public API、Status/Result layout/normal cost、直接依赖、组件闭包、selected header 或交付形状改变；
4. 仍要续签工程声明时，冻结 consumer provenance/ownership、peer、machine、instrument、compiler 或
   retained evidence identity 发生 material 改变；
5. mature same-contract peer 证明更优复杂度档，或正常 consumer/profile 证明超过 5% 的可复现 material
   regression/advantage。

新 benchmark、API/LOC 数、命名偏好、单次点值、order/address 摆动、亚纳秒差异、Formal 待办或无新事实的
“再看一遍”不构成产品重开理由。

## Closure decision

本轮裁决为 `CLOSED`：`ERR5-01..04` 已完成，`ERR5-05` 未回归，未处置 `MUST` 与 blocker 均为 0。
不可变候选在三个支持平台通过 correctness、source/static/shared delivery、C++ usability/docs 与红队，
并在冻结的 W1-WIN 五个同合同 consumer 上通过 5% P1 non-inferiority。候选保持
`d61f0710ec0e465b31d458257e2587fdfe37ed52` / `305b2cb5947627dfb5b9697c939205b6c26cc6b7`；
本 terminal receipt 不改变产品 identity。
