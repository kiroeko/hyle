# util-runtime v2：owning demangle 易用性切片

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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
- candidate correctness：W1-WIN Release/Werror build 通过，`hyle_util_test` 31/31，其中 demangle
  11 项；util public-header self-contained 1/1；完整跨平台和最终门见 verification manifest。
