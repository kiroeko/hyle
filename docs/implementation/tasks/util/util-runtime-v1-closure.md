# util-runtime v1 严格收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `util-runtime-v1` 当前轮的冻结卡与施工事实。旧 util task、2026-07 点值和 C adapter
> 只作审计输入；当前状态只看 [`module-status.json`](../../module-status.json)。

## Closure card（`AUDITED`）

- component / owner / round ID：`util` / util-runtime maintainers / `util-runtime-v1`。
- task kind：`REFACTOR`；baseline commit：
  `d6ff8735f81d891050ddef90d4eee10600f43ab2`。
- 进入原因与 demand/defect ID：
  - `UR-V1-INFERENCE-01`：本地推理引擎需要模型无关的 backend/factory 启动注册、枚举与按名选择；
    当前 llama.cpp/ggml backend registry 提供直接消费者形状，但 HYLE 尚无可执行 engine consumer；
  - `UR-V1-LEGACY-01`：现有 Registry/demangle 继承 2026-07 面向 benchmark/C adapter 的结论，必须
    重新裁决功能、C++ 易用性、依赖、复杂度和文档；
  - `UR-V1-CLOSURE-01`：Registry 是重模板、demangle 是唯一 binary symbol，两者共用 `hyle::util`
    是否造成不合理安装闭包尚未审查；
  - `UR-V1-DEMANGLE-01`：demangle 文档把“API 不转移 allocation ownership”写成近似“无分配”，
    与 Itanium `__cxa_demangle` 的真实成本不一致。
- 真实 consumer task / normal workload：
  - 推理引擎启动时登记约 4--32 个 CPU/GPU/backend factory，名字通常 4--32 bytes；处理 duplicate、
    unavailable/OOM，完成后按名选择/创建并可枚举帮助信息；动态库装卸由上层 module manager 拥有；
  - 日志、失败诊断与 backtrace 将运行期 C++ symbol 写入 caller buffer，普通符号不超过 2 KiB；
    失败时安全回显、截断可检测。demangle 不进入 token/kernel 热路。
- 当前 consumer 证据限制：仓内只有 package sample、tests 与 benchmark，没有 HYLE 产品调用点；所以
  本轮不得由 benchmark 自证新增公开能力，也不签“推理引擎端到端已验证”。保留的 Registry 形状
  必须同时由普通用户任务与成熟实现的重复形状支持。
- 范围：`registry.h`、`demangle.h/.cc`、`hyle::util` component 的公开/internal 边界、C++ API、
  ownership/lifetime/concurrency/failure/复杂度、source/static/shared 交付、直接依赖、用户/架构/任务/
  性能文档，以及只用于验证冻结合同的测试/结构证据。
- 非目标：重开 `util_core`；关闭或重构 error/mem/container；模型/设备语义；动态库 discover/load/
  unload/quiescence；unregister；捕获型任意 callable；op/kernel dispatch policy；Native FFI/C API；
  Formal 发布证书；为旧 N=256 benchmark 保留或增加产品字段。
- 直接依赖：core、error、mem、container、util_core。直接消费者：当前只有独立 C++ package sample；
  未来 compute/推理引擎是需求锚但不是已通过的 consumer。用户明确选择本轮先审 util-runtime，虽然
  mem/container 在中央状态仍为 `QUEUED`；本轮不借此宣称它们已收口，且后续依赖语义变化属于
  util 的 reopen trigger。
- tier 与理由：**P0**。Registry 的当前真实形状是 init/discovery，demangle 是诊断冷路；没有 profile
  证明任一项影响端到端 KPI。旧 frozen lookup 微基准只作历史输入，不足以把本轮提升为 P1。
- 复杂度 / 资源预算：register 为 O(|key|+|help|)+expected-amortized O(1)，允许启动期 allocation；
  frozen lookup 为 O(|key|)+expected O(1)，不得 allocation/lock；freeze 必须 O(1)、零 allocation；
  enumeration O(n)。demangle 为 O(input+output)，允许平台 ABI 所需的有界临时 allocation/serialization，
  但不得隐瞒或进入普通计算热路。新增常态 metadata 预算为 0，除非 correctness/ownership 必需。
- 总调查预算：两小时；结构性能/复杂度预算 20 分钟；独立红队 30 分钟。最多一次机制确认、一个
  产品修复批次和一次受影响重跑；不修改统一统计仪器，不发明第五种 benchmark 姿势。
- machine profile：W1-WIN（MSVC/Windows ABI）、W1-WSL（GCC/sanitizer）、M1-MAC
  （AppleClang/Itanium ABI）仅作 correctness、code structure 与交付。
- instrument profile：`closure-correctness-v1`；Formal state 为 `NONE`。
- strong peers（固定本地源码）：
  - Registry：llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb` 的
    `ggml/src/ggml-backend-reg.cpp`（真实 inference lifecycle）；Abseil
    `81c993af4fb88b3a51829a796df02251fa27102f` flat hash（索引/复杂度下界，不冒充完整 registry）；
  - demangle：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` `Demangle`；Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83` `core::demangle`。
- 适用 correctness / delivery / performance gates：normal、empty/zero、长度/容量/overflow、OOM/
  invalid input、duplicate/priority、move/destroy/lifetime、mutable/freeze race、creator/visitor reentrancy、
  stress、ASan/UBSan/TSan、MSVC/GCC/AppleClang、公开头 self-containment、组件独立 source 与 relocated
  static/shared consumer、依赖图、P0 complexity/allocation/lock/symbol structure、用户选择表/坑、红队。
  cancellation 不适用，因为该同步 startup registry 与 pure demangle 没有异步 operation。
- 预期输出：有限 capability/gap ledger、公开/internal/组件裁决、冻结 FIX、candidate commit/tree、
  逐风险 verdict、三平台与交付证据、P0 cost-structure verdict、文档、限制/reopen triggers、四项 LOC。
- 停止条件 / 可接受限制：全部适用门通过则 `CLOSED`；correctness/ownership/交付 blocker 或 P0 成本
  结构无法裁决则 `BLOCKED`。可接受限制包括 case-sensitive key、未指定枚举顺序、caller-owned
  FrozenView 生命周期、无 dynamic unload、demangle 可能回显原名/内部临时分配，以及没有端到端性能证书。

## Capability / gap ledger

审计基线为 `585a64959f2d4106b0c5984a07c70edaacf96065`。W1-WIN 的独立
`HYLE_COMPONENTS=util` source build 完成 121 个 build steps，37/37 tests 通过；这只是现状
基线，不是候选签证。源码调用扫描确认 Registry 与静态注册宏只有 tests/benchmark/package sample，
尚无 HYLE 产品调用点；demangle 也只在 tests/sample。因此下表用已命名的 inference 启动任务、
强 peer 和工程章程裁决能力，不允许测试自己成为需求。

| ID | 能力/缺口 | consumer / peer / standard | 裁决 | 本轮边界与验证 |
|---|---|---|---|---|
| UR-REG-01 | 显式按名登记 function-pointer factory；nullable 与 `Result<T>` 两种失败合同 | inference backend startup；ggml 显式 backend registration；HYLE error contract | `KEEP + FIX` | 保留一张 typed Registry 和 `Status` 注册；creator 改为编译期 `noexcept`，可失败 factory 必须返回 `Result`，不让用户异常越过 HYLE API |
| UR-REG-02 | mutable startup → immutable service；`freeze()` 与 `FrozenView` | inference init/read lifecycle；Abseil flat index 下界 | `KEEP` | freeze O(1)、零 allocation、幂等；frozen lookup O(|key|)+expected O(1)、无锁/无 allocation；view 是 borrow |
| UR-REG-03 | Registry 拥有 key/help，registration OOM 保持状态 | 普通动态配置字符串；HYLE recoverable-OOM rule | `FIX` | 当前 absent insert 先消耗两个 monotonic Region、最后 table 仍可 OOM，失败会留下不可达 arena 空间；改为先准备 table capacity，再以一次 arena allocation 提交 Entry+key+help，失败时 size/旧 entry 不变 |
| UR-REG-04 | duplicate 与 fallback/default/preferred resolution | 多 backend composition；链接顺序不得偷偷决定 winner | `KEEP` | 高 priority 覆盖、低 priority 成功 no-op、同级 `AlreadyExists`；旧 help view 在 owner 析构前稳定 |
| UR-REG-05 | mutable snapshot / frozen direct enumeration；callback 可重入 | backend list/help UI；C++ callback contract | `KEEP` | mutable O(n) temporary、锁外 callback；frozen O(1) extra space；顺序不指定，visitor 不持内部锁 |
| UR-REG-06 | `DECLARE/DEFINE/REGISTER_CLASS` 宏、`Registerer::default_creator` 与静态初始化 fail-fast | 仅旧 Clay 与 HYLE tests；ggml 真实 registry 采用显式登记 | `REMOVE` | 静态 archive TU 可被 dead-strip，宏隐藏 startup 顺序/错误路径；默认 `new/make_ref` 绕过 allocator 与 `Result` 失败合同。用户显式创建 Registry、显式登记 factory |
| UR-REG-07 | arbitrary captured callable | 无当前 consumer；function pointer 是 ggml/HYLE backend 最小形状 | `REJECT` | 不为假想消费引入 type-erasure allocation、对象状态或 lifetime；状态通过显式 factory 参数/owner 表达，出现具名消费者才 reopen |
| UR-REG-08 | unregister、DSO discover/load/unload 与 quiescence | ggml module manager 分层；函数指针/对象/reader 生命周期 | `DEFER / HIGHER LAYER` | Registry 只拥有名字表；动态模块 owner 必须排空对象与在途调用后整体销毁 Registry |
| UR-DEM-01 | runtime symbol → caller buffer，required-size/truncation/pass-through | logging/backtrace；Folly/Boost demangle | `KEEP` | O(input+output)，冷路径；失败回显原名，`cap>0` 总 NUL；不增加 owning `std::string`/allocator policy |
| UR-DEM-02 | C++ `type_info` 与固定数组易用入口 | `typeid(x)` 是普通使用；Folly 提供 type-info convenience | `FIX` | 添加 `type_info` 与 array overload，避免用户重复 `.name()`/`sizeof`；仍汇聚到一个 compiled symbol |
| UR-DEM-03 | 实际平台资源成本 | Itanium `__cxa_demangle`；Windows DbgHelp contract | `FIX DOC` | 明确 Itanium 内部 malloc/free、Windows 全局 serialization 与超长符号有界 temporary；“caller 不拥有 allocation”不等于“没有 allocation” |
| UR-BUILD-01 | runtime component 闭包 | public include owner + central component graph | `FIX` | 删除旧宏/default creator 后 Registry 不再使用 util_core；`util` 的真实 PUBLIC deps 缩为 core/error/mem/container，并复验 source/static/shared relocated consumer |
| UR-BUILD-02 | Registry template 与单个 demangle binary 是否再拆 target | 当前用户同时选择 runtime discovery/diagnostics；一个 binary symbol | `KEEP` | 本轮不为一个冷符号增加 component；若 demangle-only consumer 证明 util 闭包 material，再独立立项 |

### 冻结的产品修复批次

本轮只允许以下一次产品批次；其外发现进入 backlog/reopen trigger，不临场扩大范围：

1. `registry.h` 删除静态注册宏、默认 allocation factory 和 util_core/preprocessor 依赖；`Creator`
   类型强制 `noexcept`，并更新显式注册 sample/tests。
2. absent registration 先完成 Swiss capacity 准备，再用单次 Region allocation 构造
   `Entry + owned key + owned help`；补逐 allocation fault injection、retry、replacement 保留旧值及
   freeze 后零 allocation 结构门。
3. demangle 增加 `std::type_info`/固定数组 convenience overload；不新增 owning string API，修正
   平台 allocation/serialization 和 truncation/lifetime 文档与测试。
4. 中央 component manifest 删除 `util -> util_core`，同步 architecture/user/performance 当前结论，
   并跑 public-header、component-only、source/static/shared installed consumer。

明确不运行 timing benchmark：本轮是 P0 startup/diagnostic 能力，性能门只裁决复杂度、allocation、
lock、常态 metadata、binary dependency 与 code structure。旧 N=256 数字保留为历史原始输入，
不得写成当前候选 SOTA 证书。

## Candidate

`AUDITED` 冻结有限 FIX 并实现后填写。

## Verification manifest

候选冻结后逐风险填写；未运行项不得写 `PASS`。

## 独立红队

候选冻结后在 30 分钟预算内执行。

## Closure decision

当前为 `SCOPED`，尚无 terminal verdict。
