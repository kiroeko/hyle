# util runtime 能力完整性存量回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)回放到已经 `CLOSED` 的
> `util-runtime-v1`。这是 product-non-mutating 的存量审计：本页只裁决当前产品事实和重开条件，
> 不在回放中顺手修改公开 API、算法或布局。

## 回放合同

- component / replay：`util` / `util-runtime-capability-replay-v1`；
- reviewed state / round：`CLOSED` / `util-runtime-v1`；
- closure task：[`util-runtime-v1 closure`](util-runtime-v1-closure.md)；
- candidate / tree / end / registry anchor：
  `469356394a185b5201675fd76ff1a10a4050528b` /
  `eabcba1fe1cffb484b20c377af0423028e2f9062` /
  `b2a07d8a60f8687ea8b305eda004e7bf2a003707` /
  `b2a07d8a60f8687ea8b305eda004e7bf2a003707`；
- replay base / evidence commit：
  `5ca422c9f8041a613f6af7e7d47f56c30dc81c80` /
  `c9d8e9e60a2897ab672194e20f21665e2a26e2e7`；reviewer：Codex；日期：2026-08-28；
- normal consumer：C++23 source/static/shared 用户在启动期登记和冻结少量 backend factory，在服务期
  按名查询/创建；日志、异常诊断和 backtrace 把运行期 mangled name 变成人可读文本；
- north-star journey：量化推理引擎显式登记 CPU/GPU backend，初始化时解析一次实现并保留 typed
  handle；诊断路径既要能写入固定栈 buffer，也要能在需要完整文本时一次取得 owning string；
- tier / instrument：`P0 / closure-correctness-v1`，timing budget 为 0。Registry 是 startup/
  discovery，demangle 是诊断冷路；本回放不把旧 N=256 lookup benchmark 当需求或 SOTA 证书；
- 调查预算：30 分钟；每族最多两个已钉死 strong peers；
- strong peers：Registry 使用 llama.cpp
  `7584430716ee229751771ed0d6bbcb780d105eeb` 的 `ggml-backend-reg.cpp` 与 Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f` flat hash；demangle 使用 Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` `Demangle` 与 Boost
  `df8eeb90846648464e3c8e378e312f0cec772e83` `core::demangle`；
- 范围：Registry/demangle 两族的完整用户旅程、全部 public surface、consumer/重复实现、依赖、
  复杂度/资源、易用性和 source/static/shared 路线；
- 非目标：修改产品、重跑纳秒 benchmark、替未来推理 backend/module manager 设计策略、动态库安全
  卸载、捕获型任意 callable、op/kernel dispatch、Native FFI 或复审 util_core/mem/container。

候选到 replay base 的 runtime 产品路径 `include/hyle/util/{registry,demangle}.h` 与
`src/util/demangle.cc` 为零 diff。候选后的 util 文档变化来自 util_core 回放，不改变 Registry/
demangle 声明、算法、对象布局、依赖或二进制符号。

## 自顶向下 capability-family map

| Family | 已准入用户旅程与来源 | 必须覆盖的变体、生命周期和边界 | 当前路线 | Coverage / action | 缺失边或负空间 |
|---|---|---|---|---|---|
| named factory registration and composition | 推理 backend startup；ggml 显式登记 | raw nullable/`Result<T>`、owned key/help、duplicate/priority、OOM 原子性、显式 startup error | `Registry<Value, Args...>::register_creator/reserve` | `COMPLETE / KEEP` | captured callable 状态通过显式 `Args...`/owner；不恢复静态初始化宏或 default-`new` |
| factory lookup and failure propagation | 按名创建、missing、creator 运营失败 | mutable/frozen、creator 重入、missing sentinel/`NotFound`、move-only args | `create/has/help/size` 与 `RegistryResultTraits` | `COMPLETE / KEEP` | per-op/kernel dispatch 应在初始化时解析为 typed handle/function，不在热 kernel 中重复字符串查找 |
| mutable startup to immutable service | 多模块组合后并发只读；成熟 flat index | 单向/幂等 freeze、publication race、view borrow、destroy/quiescence | 一表两阶段 + `FrozenView` | `COMPLETE / KEEP` | Registry 必须活过 view/in-flight call；没有 owning/shared view |
| inspection and enumeration | backend help/list UI | mutable 时间点快照、frozen direct、callback 重入/early stop、顺序未指定、OOM | `for_each/for_each_key/EntryView` | `COMPLETE / KEEP` | mutable 枚举 O(n) 临时空间；frozen O(1) extra space |
| dynamic module lifetime | 可选 backend DSO load/register/drain/unload；ggml | loader handle、registration entry、created object、reader quiescence、卸载顺序 | `platform::DynLib` 只承载 loader；完整 lifecycle 由消费方的具名 module owner 设计 | `OUT_OF_SCOPE / REJECT from Registry` | HYLE 当前没有名为 `ModuleManager` 的产品类型；文档必须写“调用方/上层 owner”，不能把未来 owner 冒充现有完整 route |
| bounded runtime demangling | 固定日志字段、栈上诊断 buffer；Folly | null/plain/mangled、required size、NUL/truncation、platform fallback、内部 allocation/lock | `demangle(name/type_info, out, cap)` 与数组 overload | `COMPLETE / KEEP` | 不是 async-signal-safe；Itanium 仍 malloc/free，Windows DbgHelp 串行且 temporary 有 1 MiB 上限 |
| complete owning demangled text | 完整异常/诊断/backtrace 文本；Folly 与 Boost 重复成熟形状 | null/plain/mangled、一次逻辑调用、allocation failure、`type_info` convenience、返回值 lifetime | 用户只能先 size、再自行分配、再调用一次 buffer API | **`MISSING / FIX`** | 当前组合会重复执行平台 demangler，且要求用户手写易错的 string capacity/NUL 处理；项目其他 C++ convenience 已自然使用 `std::string` |
| signal-handler demangling | crash signal handler | zero allocation、zero lock、平台可用 parser | 无跨平台路线 | `OUT_OF_SCOPE / DEFER` | 只有 Folly 的可选 libiberty buffer path 提供该窄合同；Boost 和 HYLE 平台 ABI 不共同证明它是当前 MUST |
| component delivery | C++23 source/static/shared selected component | public header closure、compiled symbol、relocated consumer | `hyle::util -> core,error,mem,container` | `COMPLETE / KEEP` | 单个 demangle symbol 与 Registry template 共 target；demangle-only closure 只有 material consumer 才触发拆分 |

`complete owning demangled text` 由两个独立 strong peers 反复提供，问题边界稳定，不给 Registry 或其他
热路径增加 metadata/lock/dynamic dispatch；它因此按能力准入规则从 `GENERAL` 提升为 `MUST`。当前路线
是 `MISSING`，旧 closure 的“不增加 owning string”裁决不能继续写成完整。其余 `MUST` 都为
`COMPLETE`；动态装卸和 signal-handler parser 是明确的 `OUT_OF_SCOPE`，不是半项 Registry/demangle。

## 自底向上 public-surface trace

| 当前入口 | Capability / journey | 生产反向证据 | 裁决 |
|---|---|---|---|
| `RegistryPriority` | deterministic duplicate composition | 当前仓内仅 tests/sample/bench；north-star backend startup 与 ggml 旅程支撑 | `KEEP` |
| `Registry::Creator` / `EntryView` | typed factory 与只读枚举 | package sample、20 个 Registry test case；function pointer `noexcept` 编译期约束 | `KEEP` |
| constructor/destructor、deleted copy/move | allocator-owned one-shot registry lifetime | default system allocator 与 injected/failing allocator 测试 | `KEEP`；invalid allocator 是 mem precondition，不另造 nullable Registry |
| `register_creator/reserve` | mutable startup | dynamic strings、priority、collision、每个 allocation failure 与 retry | `KEEP` |
| `freeze/frozen/FrozenView` | immutable publication和借用 view | freeze race、并发读、idempotent/zero-allocation 测试 | `KEEP`；不是 op/kernel dispatch policy |
| mutable/frozen `create/has/help/size` | lookup/create/failure | raw、`Result<T>`、intrusive owner、move-only args、reentrant creator | `KEEP` |
| mutable/frozen `for_each`、`for_each_key` | inspection/snapshot | reentrant mutation、early-stop/void visitor、allocation failure | `KEEP` |
| `demangle(name, out, cap)` | bounded diagnostic text | package sample 与 null/plain/type/truncation tests | `KEEP` |
| `demangle(type_info, ...)` 与数组 overload | natural fixed-buffer C++ entry | typeid/array tests | `KEEP` |
| owning `demangle(name/type_info)` | complete diagnostic text | **不存在**；Folly/Boost 都有，用户需重复 demangle + 手工 string storage | `MISSING / PRODUCT_TRIGGER` |

语法扫描确认 HYLE production 中没有 Registry/demangle 调用，也没有第二套 `__cxa_demangle`、
`UnDecorateSymbolName` 或通用 named-factory registry。当前直接消费者仍是独立 package sample；bench 只
是历史测量输入，不承担需求。`compute` 的半成品 backend table 是 owner-specific device dispatch，不能
反向证明通用 Registry 已被端到端采用，也不在本回放越界迁移。

## 复杂度、资源、依赖与易用性裁决

- Registry register 保持 O(|key|+|help|)+expected-amortized O(1)，lookup 为 O(|key|)+expected
  O(1)，freeze O(1)/零 allocation，枚举 O(n)；frozen 读无锁/无 allocation。creator 在 Swiss value
  内联、冷 metadata 在 owned Entry，属于 read-heavy registry 的明确时空取舍，不是 benchmark case
  专用字段。
- demangle buffer API 为 O(input+output)，但 Itanium 内部 allocation、Windows 全局锁/有界 temporary
  是平台事实；不得把 caller-buffer 误写成实现 zero-allocation 或 async-signal-safe。
- 用户指南当前把 `FrozenView` 描述为“高频 dispatch”容易诱导 kernel/op 级字符串查找；真实推理
  journey 应初始化时解析 backend/handle，只有确需反复按名创建/查询时缓存 FrozenView。此项是文档
  校准，不要求删除自然的 immutable-view API。
- 文档写“由上层 module manager 完成”会让用户误以为 HYLE 已提供该类型。当前完整事实是
  `platform::DynLib` 只负责 RAII loader handle；对象排空、reader quiescence 和 registry 销毁由消费方
  的 module owner 负责。
- central graph 的 `util -> PUBLIC core,error,mem,container` 与公开 include 完全一致；util 不依赖
  util_core。source/static/shared 形状没有产品或依赖漂移。

## 裁决与下一动作

- 产品 verdict：**`PRODUCT_TRIGGER`**，trigger ID `UR-REPLAY-OWNING-TEXT-01`；`util` 从
  `CLOSED` 进入 `REOPENED`；
- trigger evidence：两个 pinned strong peers 都提供一次取得 owning demangled text 的公开 C++ 形状，
  HYLE 现有 buffer-only 组合要求两次平台 demangle 和易错的 sizing/NUL boilerplate；这击中旧轮明确
  的 API-usability/missing-capability reopen 条件；
- 未发现 Registry correctness、复杂度、ownership、dependency 或 delivery trigger；不因仓内尚无
  production consumer 删除其成熟 startup/discovery 能力；
- 下一轮只 scope `std::string` owning demangle（含 `type_info` convenience、OOM/exception contract、
  平台一次 demangle 实现、测试和文档），并校准 FrozenView/module-owner 两处用户陷阱；不重构
  Registry，不进入 timing benchmark，不扩到动态 module manager；
- Formal 仍为 `NONE`。P0 新轮只验证复杂度、allocation/lock、正确性和三种交付，不产生 SOTA timing
  声明。
