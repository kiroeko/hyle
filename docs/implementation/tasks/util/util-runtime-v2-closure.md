# util-runtime v2：owning demangle 易用性切片

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本轮由
> [`util runtime capability replay`](util-runtime-capability-replay-v1.md) 的
> `UR-REPLAY-OWNING-TEXT-01` 触发。它不是再次重构 Registry，也不把 demangle 变成性能热点；只补齐
> 两个成熟来源共同证明的完整 owning-text 用户旅程，并校准相关使用边界。当前机械状态只看
> [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

- component / owner / round ID：`util` / util-runtime maintainers / `util-runtime-v2`；
- task kind：`REOPEN`；baseline commit：
  `c9d8e9e60a2897ab672194e20f21665e2a26e2e7`；trigger：
  `UR-REPLAY-OWNING-TEXT-01`；
- 进入原因：Folly `Demangle` 与 Boost `core::demangle` 都提供一次取得完整 owning demangled text 的
  C++ 形状；HYLE 只有 caller-buffer API，完整文本需要先 sizing、分配，再第二次执行平台 demangler，
  还要求用户正确处理 `std::string` capacity/NUL。这是重复成熟基础需求和实际易用性缺口；
- normal consumer / north-star journey：错误日志、异常诊断和 backtrace UI 需要保留完整 symbol 文本；
  调用方写 `std::string text = hyle::util::demangle(name)` 或对 `type_info` 调用同形 overload。固定大小、
  不希望 API 层抛 allocation exception 的日志字段继续使用现有 buffer overload；
- scope：
  - `std::string demangle(const char*)` 与 `std::string demangle(const std::type_info&)`；
  - 每次 public 调用至多执行一次平台 demangler，null/plain/failure 都回显既有语义；
  - owning overload 的 allocation/exception/lifetime 合同，buffer input/output overlap 与
    async-signal-safety 陷阱；
  - 受影响 tests、package sample、architecture/user 文档、公开头/符号、source/static/shared 交付；
  - 把 `FrozenView` 从“kernel/op 高频 dispatch”误导校准为按名 service lookup，并把不存在的 HYLE
    `ModuleManager` 改写为调用方/上层 module owner；
- non-goals：Registry API/算法/布局；dynamic module discover/load/unload/quiescence；新增 HYLE string
  类型或 allocator-aware owning string；async-signal-safe parser；util target 拆分；util_core/mem/
  container 收口；Native FFI/C API；timing benchmark、端到端推理或 SOTA 声明；
- direct dependencies：保持 `core/error/mem/container`；direct consumer 仍是 package sample，推理引擎
  是已声明 north-star，不冒充已完成产品调用；
- tier / instrument：**P0 / `closure-correctness-v1`**。demangle 是诊断冷路径，timing budget 为 0；
  只验证 O(input+output)、一次平台 demangle、无常态 metadata，以及公开的 allocation/lock 事实；
- resource target：owning overload 返回独立 `std::string`，可因标准字符串分配抛异常；它必须是
  header-inline convenience，通过窄 internal sink seam 在 caller CRT 中构造字符串，不能从 DLL 返回
  DLL-owned STL。Itanium `__cxa_demangle` 仍可有一次内部 malloc，随后复制到 standard string；
  Windows 短 symbol 使用栈 temporary，超长 symbol 的现有有界重试保持。不得先调用 buffer API
  sizing 再调用第二次；
- total budget：45 分钟；结构/实现检查 10 分钟；红队 15 分钟；最多一个实现批次和一次受影响重跑，
  不扩建性能仪器；
- machine profiles：W1-WIN、W1-WSL、M1-MAC；instrument profile：
  `closure-correctness-v1`；Formal state：`NONE`；
- strong peers：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` `Demangle`；Boost
  `df8eeb90846648464e3c8e378e312f0cec772e83` `core::demangle`；Registry 沿用 v1 已签结论，不在本轮
  再引入 peer；
- correctness/delivery gates：null/empty/plain/real mangled、type_info、long/truncated、zero cap、
  in-place/overlap、platform failure fallback、exception cleanup、public-header self-containment、current
  util-only build、source/static/shared installed consumer、导出符号与依赖闭包、文档选择/陷阱；
- expected output：有限 ledger/FIX、一个 candidate、逐门 verdict、当前限制/reopen triggers、结束
  receipt 和四项 LOC；
- stop condition：owning 与 buffer 两条 admitted journey 都 `COMPLETE`，Registry 回放结论未被本轮
  改动，所有适用 P0/correctness/delivery/docs/red-team 门通过；否则用具名 blocker 停止。

## Capability / gap ledger（`AUDITED`）

回放已给出完整 util-runtime capability-family map；本轮只重新审计触发族、其 ABI/交付边和直接相邻
文档。Registry 的实现与 public contract 不进入 FIX。

| ID | demand / current coverage | exact source / normal shape | semantic / artifact / delivery owner | lifecycle、failure 与复杂度 | action / severity / verification / negative space |
|---|---|---|---|---|---|
| U2-DM-01 | `MUST` owning runtime symbol text；`MISSING` | Folly/Boost 两份 pinned peer；`std::string demangle(name/type_info)` | util / `demangle.h` inline + `hyle_util` internal sink / util component | 返回独立 string；平台失败回显；caller-side allocation 可抛；O(input+output) | `FIX / high`；null/plain/real symbol/type_info/long tests；不新增 allocator-aware HYLE string |
| U2-DM-02 | `MUST` 一次 public 调用只执行一次 platform demangler；`MISSING` 于 owning 路线 | north-star 完整诊断文本；不能把 buffer sizing+second call 冒充自然组合 | util / `demangle.cc` common visit / util component | platform temporary 在 sink 返回/异常前有效；RAII cleanup；一次 platform call（Windows 只有容量重试） | `FIX / high`；结构审查与 callback counter；不做 timing benchmark |
| U2-ABI-01 | `MUST` shared C++ SDK 的 owning convenience；直接导出 STL 为错误 route | build/package C++ source/static/shared；Error 已明确不让 DLL-owned STL 跨 CRT | util semantics / header-inline `std::string` + exported `detail` sink / build-package smoke | sink 在 caller TU 构造 string；catch/rethrow 发生在 caller，异常不穿过 compiled seam；无 STL layout/heap ownership跨 DLL | `FIX / high`；relocated shared/static/source sample；internal seam 不是 Native FFI |
| U2-DM-03 | 现有 bounded buffer 的 input/output overlap 边界未定义；`PARTIAL` | caller 可能原地清理 backtrace symbol；支持只需 overlap-safe copy | util / `fill` / util component | 先取得完整长度，再 `memmove` 并 NUL；O(n)，无新增 allocation/metadata | `FIX / medium`；exact/partial overlap tests；不承诺 arbitrary non-NUL input |
| U2-DM-04 | signal-handler demangle；无跨平台路线 | Folly 仅可选 libiberty buffer path，Boost/平台 ABI 不共同支持 | crash/symbolizer owner | 需要 async-signal-safe parser；当前 Itanium allocation、Windows mutex 明确不满足 | `DEFER / OUT_OF_SCOPE`；user trap 必须直说；不假装 caller buffer 等于 signal-safe |
| U2-REG-01 | Registry 六族均 `COMPLETE` | v1 closure + capability replay；推理 backend startup/discovery | util / `registry.h` / util component | 合同、布局、算法、依赖均未漂移 | `KEEP / no product change`；只把“高频 dispatch”改成按名 service lookup，kernel/op 初始化后缓存 typed handle |
| U2-MOD-01 | dynamic module lifecycle 不属于 Registry | ggml owner-specific manager；HYLE 当前只有 `platform::DynLib` | caller/module owner；platform 只拥有 loader handle | drain objects/readers → destroy registry → close library | `REJECT from Registry / DOC FIX`；不得把不存在的 HYLE `ModuleManager` 写成现有完整 route |
| U2-BUILD-01 | util component 与 symbol closure | current central graph/public includes | util + build/package | direct deps 保持 core/error/mem/container；compiled symbol 从一个变两个，不拆 target | `KEEP + VERIFY`；只有 material demangle-only consumer 才考虑 component split |

所有本轮 `MUST` 在候选前必须由 `MISSING/PARTIAL -> COMPLETE`；没有 `PARTIAL + KEEP`。冻结 FIX 只有：

1. public header 增加 owning string/type-info inline overload和 internal sink 声明；sink callback 捕获并在
   caller 侧重抛 allocation/length exception，compiled seam 保持 no-throw；
2. `demangle.cc` 抽取一次 platform-demangle/visit 路径；所有 temporary 使用 RAII，Windows mutex 只
   覆盖 DbgHelp 调用，不覆盖 string construction/copy；buffer fallback 改为 overlap-safe copy；
3. tests/package sample 覆盖 owning、type_info、原地/部分 overlap、现有 bounded contract 与三种交付；
4. architecture/user 文档写清 owning vs fixed-buffer 选择、异常、内部 allocation/lock、非
   async-signal-safe、Registry 不用于 kernel/op string dispatch，以及 caller-owned module lifecycle。

其外发现进入下轮 trigger，不在 v2 扩张 Registry、string/container、module manager 或性能仪器。

## Public-surface trace

| 入口 | capability / 当前状态 | 本轮裁决 |
|---|---|---|
| `demangle(const char*, char*, size_t) noexcept` | bounded full-length/truncation/pass-through；`COMPLETE`，overlap edge `PARTIAL` | 保留 ABI；共用 single-visit 实现并补 overlap |
| buffer `type_info` / fixed-array overload | 自然固定存储入口；`COMPLETE` | `KEEP` |
| owning `demangle(const char*)` | 完整 owning text；`MISSING` | header-inline `std::string` `FIX` |
| owning `demangle(const type_info&)` | runtime type convenience；`MISSING` | header-inline forwarder `FIX` |
| `detail::demangle_visit` | source/static/shared 不跨 STL ownership 的 implementation seam；`MISSING` | exported internal ABI `FIX`；用户不直接调用 |
| `RegistryPriority`、`Registry`、`EntryView/FrozenView` 及全部 register/query/enumerate 入口 | startup/discovery 六族；`COMPLETE` | 产品零改动；仅校准选型文字 |

仓库生产扫描仍没有第二套 demangler 或通用 Registry，也没有 util runtime 的内部产品调用；tests/bench/
sample 不冒充需求。current central dependency graph 与 public include trace 一致，不需要新增依赖。

## Candidate

- product commit / tree：
  `f3b341cb2ae81154f81bc45052a651e773c5e2b3` /
  `c3141ccfd2902a450eb271f867a2ee8aacc5cbb9`；
- public C++ contract：新增 header-inline `std::string demangle(const char*)` 与 `type_info` overload；
  buffer overload ABI 保持且明确支持 overlap。owning overload 抛出 caller-side standard string 的
  allocation/length exception，平台 demangle failure 仍回显输入；
- compiled ABI：新增 exported `detail::demangle_visit_internal` sink；MSVC shared DLL 的导出检查只见
  buffer symbol 与该 sink，没有导出 STL-returning symbol。sink callback 为 `noexcept`，在 caller TU
  捕获 exception，compiled seam 自身不传播异常；
- algorithm / resources：同一 internal visit 执行一次 platform demangle；Itanium temporary 由 RAII
  `free`，Windows 每次 DbgHelp 调用单独持有全局 mutex，allocation/copy/string construction 均在锁外。
  buffer fallback 使用 `memmove`；复杂度保持 O(input+output)，无常态 metadata；
- Registry：公开声明、算法、layout 和 tests 之外的实现均为零改动；只校准用户指南的 lookup/module
  lifecycle 边界；
- direct consumer：package sample 同时执行 raw/fallible Registry、buffer demangle 与 owning
  demangle。W1-WIN source/shared/static relocated consumer 各 1/1；
- candidate correctness：W1-WIN Release/Werror build 通过，`hyle_util_test` 32/32，其中 demangle
  12 项；util public-header self-contained 1/1；完整跨平台和最终门见 verification manifest。

## Verification manifest

验证始终绑定 product commit
`f3b341cb2ae81154f81bc45052a651e773c5e2b3` / tree
`c3141ccfd2902a450eb271f867a2ee8aacc5cbb9`；后续 task/registry 文档提交不改变候选产品树。

### Correctness risk matrix

| Risk | verdict | 命令 / 证据 | 限制 |
|---|---|---|---|
| normal | `PASS` | W1-WIN、W1-WSL、M1-MAC 的 Release `hyle_util_test` 均 32/32；plain、真实平台 mangled symbol、`type_info`、owning/buffer 两形状均通过 | 仓内直接消费者是 package sample，不冒充完整 inference engine |
| empty / zero | `PASS` | null、empty、zero-capacity 与 buffer NUL 合同继续通过；owning null/plain 回显通过 | owning empty string 仍遵循标准库自己的 empty-string 实现策略 |
| capacity / boundary / overflow | `PASS` | required-size、truncation、long symbol、exact/partial input-output overlap；fallback 改用 `memmove` | Windows decorated symbol 重试上限保持 1 MiB，超限按合同回显原名 |
| OOM / system failure / invalid input | `PASS` | Itanium temporary 由 RAII `free`；平台失败回显；caller sink 捕获 `std::string` exception 后在 caller 侧重抛；WSL ASan/UBSan 通过 | 未对标准库 string allocator 或系统 demangler 做强制 fault injection；owning allocation failure 按标准异常合同处理 |
| move / destroy / lifetime | `PASS` | owning result 独立于 platform temporary；sink 只在 temporary 有效期内同步调用；Windows/macOS source/shared/static relocated consumer 通过 | `detail` sink 是 implementation seam，不是用户可保存的异步 callback |
| concurrency / reentrancy | `PASS` | Registry 产品零 diff；现有并发 Registry tests 随三平台 32/32 通过；Windows mutex 仅包住每次 HYLE DbgHelp 调用 | HYLE 的 mutex 不替直接调用 DbgHelp 的外部代码协调；应用混用时必须统一串行化 |
| cancellation / close | `NOT_APPLICABLE` | 同步 demangle 与本轮未改的 startup Registry 没有异步 operation | dynamic unload/quiescence 仍由调用方 module owner 负责 |
| property / differential / fuzz / stress | `PASS` | 两个真实 ABI symbol、plain fallback、边界/overlap 与既有 Registry stress 构成本轮风险覆盖 | 冷诊断 helper 没有单独建立 fuzz campaign |
| ASan / UBSan | `PASS` | W1-WSL GCC 14.2 独立 Debug sanitizer build，`hyle_util_test` 各 32/32 | sanitizer 在 WSL native ext4 候选 worktree 运行 |
| TSan | `NOT_APPLICABLE` | v2 没有改变 Registry 或新增共享 product state；Windows DbgHelp serialization 由平台分支 mutex 结构审查覆盖 | Registry v1 已有 TSan 证据；本轮不重复运行并冒充新增证据 |
| real ISA / backend | `NOT_APPLICABLE` | MSVC DbgHelp、GCC Itanium ABI、AppleClang Itanium ABI 均以真实 symbol 测试；没有 ISA dispatch | 不产生 GPU/backend 功能或性能声明 |

### Aggregate gates

| Aggregate gate | verdict | 证据 | 限制 |
|---|---|---|---|
| scope | `PASS` | 冻结四项 FIX 全部落地；Registry 产品零改动；没有扩张 string/module-manager/perf 仪器 | 新需求按 reopen trigger 独立立项 |
| architecture | `PASS` | owning string 在 caller TU 构造；compiled seam 只传 transient bytes；buffer 与 owning 共用 single-visit 路径 | internal sink 是 shared C++ SDK 实现边，不是 Native FFI |
| correctness | `PASS` | 三平台 Release 32/32；W1-WSL ASan/UBSan 各 32/32；public header self-contained 1/1 | TSan 按上表 N/A，不改写为已运行 |
| delivery | `PASS` | W1-WIN 与 M1-MAC selected source、relocated shared、relocated static consumer 各 1/1；三平台 util component build 通过 | macOS 按项目合同使用 deployment target 14.4 |
| usability / docs | `PASS` | owning/buffer 选择、exception、overlap、signal-handler、Registry typed-handle 与 caller module owner 陷阱均写入 architecture/user guide | 没有 C API/Native FFI |
| red team | `PASS` | 见下节；导出、ownership、异常、overlap、锁和误导文档逐项攻击 | 只裁决本轮冻结范围 |
| engineering performance | `PASS` | P0：O(input+output)、一次 platform demangler、无常态 metadata；锁/temporary/allocation 事实公开；三平台动态符号均只有 buffer + sink | 未运行 timing benchmark，不声明纳秒或跨平台 SOTA |

### Reproduction summary

- W1-WIN：Visual Studio 18 2026 / MSVC 19.50，Release、warnings-as-errors，selected `util`；
  `hyle_util_test` 32/32，public-header 1/1。selected source/shared/static relocated sample 各 1/1；
  `dumpbin /exports` 只见 buffer demangle 与 internal sink，没有 STL-returning symbol。
- W1-WSL：Ubuntu 24.04 native-ext4 candidate worktree，GCC 14.2 Release/Werror 32/32，public-header
  1/1；ASan 与 UBSan 独立 build 各 32/32。`nm -D --defined-only | c++filt` 只见 buffer demangle
  与 internal sink 两个 util demangle symbol。
- M1-MAC：AppleClang 21、Unix Makefiles、deployment target 14.4，Release 32/32，public-header 1/1；
  selected source/shared/static relocated sample 各 1/1。`nm -gU | c++filt` 只见上述两个符号，
  `otool -L` 的实际 closure 与 util 声明依赖一致。
- P0 不运行 timing；完整当前解释集中在
  [`benchmarks-util`](../../evidence/performance/benchmarks-util.md)，旧 2026-07 点值不进入本轮 verdict。

## 独立红队

| 攻击 | severity | verdict | 证据 / 裁决 |
|---|---|---|---|
| 从 shared library 直接返回 DLL-owned `std::string` | high | `fixed` | owning overload header-inline，compiled sink 只传指针/长度；三平台符号检查无 STL-returning export |
| owning convenience 先 sizing、再第二次调用 platform demangler | high | `fixed` | common visit 一次取得结果并同步送入 caller sink；Windows 只有 API 无 required-size 时的有界容量重试 |
| string allocation exception 穿过 `noexcept` compiled seam，或 callback 保存临时指针 | high | `disproved` | caller callback 捕获 `exception_ptr`，sink 返回后 caller 重抛；callback 同步且不公开给用户 |
| fallback 在 input/output 重叠时 `memcpy` 触发 UB | medium | `fixed` | 改为 `memmove`，exact/partial overlap tests 三平台通过 |
| caller buffer 被误读为 allocation-free、lock-free 或 signal-safe | medium | `fixed` | public/user/architecture 明写 Itanium allocation、Windows serialization 和受限上下文禁用 |
| HYLE mutex 能自动协调应用直接调用的所有 DbgHelp | medium | `accepted` | HYLE 只序列化自己的 DbgHelp 调用；用户指南要求混用方统一串行化，不增加进程级 hook |
| Registry 被推荐为每 token/kernel 字符串 dispatch，或文档冒充已有 `ModuleManager` | medium | `fixed` | 指南改为启动期 service discovery/按名查询，热路缓存 typed handle；动态生命周期归调用方具名 owner |
| 两个冷 binary symbols 值得拆成独立 component | low | `accepted` | 当前直接 consumer 同时用 Registry/demangle，拆分只增加交付选择；material demangle-only consumer 才重开 |

红队未发现未处理 HIGH；没有为 P0 冷路径增加 benchmark-only API、常态 metadata 或 workload 特化。

## Verification decision

所有当前 `MUST` 已从 `MISSING/PARTIAL` 变为 `COMPLETE`，Registry 的 v1 能力结论未漂移；scope、
architecture、correctness、delivery、usability/docs、red-team 与 P0 engineering performance 七门均
`PASS`，Formal 为 `NONE`，无 blocker。因此候选满足 `VERIFIED`，并可在 terminal receipt 与中央
registry 锚定后进入 `CLOSED`。

已知限制：

- demangle 是诊断冷路，不是 async-signal-safe；Itanium 可能内部 malloc/free，Windows DbgHelp
  串行且超长 symbol 有 1 MiB 重试上限，失败回显原名；
- owning convenience 返回 `std::string` 并可抛标准分配/长度异常；buffer convenience 为
  `noexcept`，但不保证平台内部零分配或无锁；
- Registry 仍为 case-sensitive、枚举顺序未指定、`FrozenView` owner-bounded borrow，且不负责
  unregister、动态 unload 或 reader quiescence；shared util component 仍带 Registry 的依赖闭包；
- 本轮只有 P0 结构签证，没有端到端 inference profile、timing SOTA 或 Formal 发布证书。

重新打开触发器：

1. 真实消费者暴露本边界内 correctness、ownership、异常、API 易用性或常用能力缺口；
2. demangle/Registry 的公开合同、算法、布局、直接依赖、compiler/platform ABI 或交付形状变化；
3. 成熟同合同 peer 证明更好复杂度档，或 profile 证明 demangle/按名 Registry lookup 成为 material 热点；
4. 真实 demangle-only consumer 证明当前 util component closure 成本 material；
5. 需要 async-signal-safe symbolization 或动态 module unload/quiescence 时，由相应 owner 另立能力单元。

候选的 canonical LOC self-check（snapshot
`f3b341cb2ae81154f81bc45052a651e773c5e2b3`）：全库代码 **298,478**；自实现包含测试
**122,321**；自实现不包含测试 **82,176**；当前 `util` 实现 **1,244** 行。这里的模块边界按
canonical `include/hyle/util + src/util`，包含同目录下已独立签结的 util-core 公开头；LOC 只用于规模
复盘，不作为完成度证据。

## Terminal receipt

- terminal verdict：`CLOSED`；product candidate
  `f3b341cb2ae81154f81bc45052a651e773c5e2b3` / tree
  `c3141ccfd2902a450eb271f867a2ee8aacc5cbb9` 的全部适用门为 `PASS`，无 blocker；
- terminal end receipt commit：`25cff0db8478e9adf59b7dbd8a85e7bff684d947`；
- registry anchor commit：`2a47768f8a038590d81f33a218efb2d83b86e2d5`，这是首个确实保存本
  `CLOSED` state/candidate 的中央 registry 快照。提交不能在自身内容中预写自身 hash，故紧随其后的
  evidence-only 提交只校准 `registry_anchor_commit` 字段；产品、状态、候选和 verdict 均未变化；
- raw local evidence：W1-WIN `C:\public\project-hyle\_validation\util-v2-win`；W1-WSL
  `/root/hyle-scratch/util-v2-{f3b341cb,build-release-f3b341cb,build-asan-f3b341cb,build-ubsan-f3b341cb}`；
  M1-MAC `/Users/kiro/hyle-scratch/{util-v2-f3b341cb,util-v2-delivery-f3b341cb}`。这些是本地可重放
  artifact，不冒充长期签名 Formal bundle；
- LOC snapshot：`f3b341cb2ae81154f81bc45052a651e773c5e2b3`，`self-check=pass`；四项数值见上节；
- 后续只按本页 reopen triggers 重开，旧 v1 与 2026-07 task/evidence 不再承担当前状态。
