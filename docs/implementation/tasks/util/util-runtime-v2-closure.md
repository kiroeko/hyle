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
- resource target：owning overload 返回独立 `std::string`，可因标准字符串分配抛异常；Itanium
  `__cxa_demangle` 仍可有一次内部 malloc，随后复制到 standard string；Windows 短 symbol 使用栈
  temporary，超长 symbol 的现有有界重试保持。不得先调用 buffer API sizing 再调用第二次；
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

## Capability / gap ledger（待 `AUDITED` 冻结）

本节将在 public-surface、实现和两份 peer 的有限审计完成后冻结；`SCOPED` 阶段不预写产品结论。

