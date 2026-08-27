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

- product candidate commit：`469356394a185b5201675fd76ff1a10a4050528b`；tree：
  `eabcba1fe1cffb484b20c377af0423028e2f9062`。
- 该候选完整实现冻结批次：
  - Registry 只保留显式、可恢复的 startup registration；creator 类型强制 `noexcept`，旧
    self-registration macros、Registerer/default creator 与 util_core 直接依赖删除；
  - absent registration 先准备 Swiss capacity，再以一个 Region allocation 提交
    Entry/key/help；replacement help 失败保留旧 creator/priority/help；
  - demangle 增加 `type_info`/固定数组 C++ overload，平台内部 allocation/serialization 成本在
    public header、architecture、user 与 performance 文档一致；
  - package sample、tests、历史 benchmark compile surface 和中央 component graph 已迁移。
- freeze 前 sanity：W1-WIN 独立 util closure 37/37、public-header self-containment、util benchmark
  compile，以及 selected util source/relocated shared/relocated static consumer 均通过；docs
  267 Markdown / 757 links、module registry contract 通过。这些会在 verification manifest 按
  candidate identity 记最终 verdict，不借预跑跳过三平台/红队。
- 候选后若需产品代码变化，状态退回 `AUDITED`；verification 期间只允许补证据、task/registry
  事实和非产品性命令修正。

## Verification manifest

验证对象始终是 product commit
`469356394a185b5201675fd76ff1a10a4050528b` / tree
`eabcba1fe1cffb484b20c377af0423028e2f9062`；后续 task/registry 提交只记录证据，未改变产品树。

### Correctness risk matrix

| Risk | verdict | 命令 / 证据 | 限制 |
|---|---|---|---|
| normal | `PASS` | W1-WIN、W1-WSL、M1-MAC 的 `HYLE_COMPONENTS=util` Release closure 均为 37/37；三平台 `hyle_util_test` 均通过 | 仓内消费者仍是 package sample，不冒充完整 inference engine |
| empty / zero | `PASS` | Registry empty/missing、empty help、freeze；demangle null/empty/zero-capacity 测试 | nullable factory 的 missing 与 creator-null 按公开合同都返回 null |
| capacity / boundary / overflow | `PASS` | reserve/freeze-after-reserve、demangle truncation/required-size、`MaximumStringLengthWidensBeforeNulByteAllocation` | `string_ref` 的公开长度上限仍由 container 拥有 |
| OOM / system failure / invalid input | `PASS` | fault injector 穷举 initial backing call；replacement OOM、retry、null key/help/creator、priority 和 platform demangle fallback | DbgHelp/Itanium 不提供可恢复的 demangler allocation error，按合同回显原名 |
| move / destroy / lifetime | `PASS` | move-only factory argument、owned dynamic key/help、replacement 后旧 help view、三 sanitizer | `FrozenView` 是 borrow，owner 析构前必须由调用方排空 reader |
| concurrency / reentrancy | `PASS` | concurrent registration/read、creator reentry、snapshot visitor reentry、freeze/read race；W1-WSL TSan | registry 析构与 operation 并发不受支持；已在用户文档明确 |
| cancellation / close | `NOT_APPLICABLE` | 同步 startup registry 与 pure diagnostic demangle 没有异步 operation | `freeze()` 是单向 publication，不是 cancellation/close protocol |
| property / differential / fuzz / stress | `PASS` | allocation-boundary sweep、multi-thread registration/read stress 与三平台完整测试 | 本轮风险由穷举 fault boundary/stress 覆盖，未单独运行 fuzz |
| ASan / UBSan / TSan | `PASS` | W1-WSL GCC 14.2 三个独立 build；`hyle_util_test` 各 1/1；TSan 用 `setarch $(uname -m) -R` 关闭 ASLR 后运行 | sanitizer 由 WSL native-ext4 承担，Mac 不重复冒充 sanitizer 证据 |
| real ISA / backend | `NOT_APPLICABLE` | Registry 无 ISA/backend 分支；demangle 的 MSVC ABI、GCC Itanium ABI 与 AppleClang Itanium ABI 已由三平台 correctness 覆盖 | 不产生真实 GPU/backend 功能声明 |

### Aggregate gates

| Aggregate gate | verdict | 命令 / 证据 | 限制 |
|---|---|---|---|
| correctness / sanitizer | `PASS` | W1-WIN MSVC 14.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21；各平台 37/37，WSL ASan/UBSan/TSan 各 1/1 | 不把下游未关闭模块的正确性并入本结论 |
| component / source / static / shared | `PASS` | 三平台 selected-component smoke；源码嵌入、relocated shared SDK、relocated static SDK consumer 全通过 | Mac consumer 按项目合同显式设 `MACOSX_DEPLOYMENT_TARGET=14.4`；缺少 Ninja 时用系统 Makefiles |
| direct consumer | `PASS` | `samples/package/util/cpp` 同时走 fallible Registry、freeze/create/help 与 demangle；三种交付运行通过 | 这是独立 C++ package journey，不是端到端 inference profile |
| P0 engineering performance | `PASS` | fault injection/zero-allocation assertions、源码结构审查；M1 `nm -gU` 仅导出一个 demangle symbol，`otool -L` 与声明闭包一致 | 不运行 timing benchmark，不声明全平台纳秒 SOTA |
| usability / docs | `PASS` | public-header self-containment；`docs/user/util.md` 选择表、完整 journey、ownership/failure/concurrency/坑；`check-docs.py` 267 Markdown / 757 relative links | Native FFI/C API 明确 N/A |
| independent red team | `PASS` | 下节冻结合同攻击；module checker、consumer scan、symbol/load-command 检查 | 只攻击本轮范围，未重开 util_core 或依赖模块 |

### Reproduction summary

- W1-WIN：Ninja Release、MSVC 14.50、`HYLE_COMPONENTS=util`，37/37；public-header、
  `hyle_util_bench` compile-only，以及 selected source/shared/static consumer 通过。
- W1-WSL：GCC 14.2 Release 37/37；ASan、UBSan、TSan 的 `hyle_util_test` 各 1/1；selected
  source/shared/static consumer 通过。系统默认 GCC 13.3 不满足项目编译器合同后，显式使用
  `CC=gcc-14 CXX=g++-14`，这是环境纠正而非产品改动。
- M1-MAC：AppleClang 21 / CMake 4.3.4 / Unix Makefiles，Release 37/37；设置项目规定的
  `MACOSX_DEPLOYMENT_TARGET=14.4` 后 selected source/shared/static consumer 全通过。Mac 缺少 Ninja
  的首次 configure 未产生产品 verdict，改用系统 Makefiles 的独立目录后通过。
- P0 symbol/dependency structure：M1 `nm -gU` 的 `libhyle_util.dylib` 只有
  `hyle::util::demangle(char const*, char*, unsigned long)` 一个公开 binary symbol；`otool -L`
  显示 core-header + error/mem/container 的实际 transitively loaded closure，未出现 util 反向依赖。

## 独立红队

- 审查者 / 日期 / 时间预算：独立冻结候选复检 / 2026-08-28 / 30 分钟上限。
- capability / contract / complexity：逐项反向追踪 public Registry/demangle；无 benchmark-only API；
  registration、lookup、freeze、enumeration 与 demangle 均在冻结复杂度档内。
- correctness / lifetime / concurrency：检查 absent/replacement OOM commit 点、key/help ownership、
  mutable/frozen publication、creator/visitor 锁外重入和 owner/view 生命周期；sanitizer 与压力测试支持结论。
- dependency / build / delivery：扫描当前调用点和旧静态宏；核对中央 component edge、公开头、真实 target、
  source/static/shared relocation，以及 Mach-O export/load commands。
- engineering performance validity：P0 只签复杂度、allocation/lock/metadata/symbol structure；拒绝沿用旧
  N=256 timing 数字或以它签 SOTA。

| 问题 | severity | verdict | 证据 |
|---|---|---|---|
| 静态注册 TU 在 archive 中可被 dead-strip，且 init 失败无自然接收方 | high | `fixed` | 旧宏/Registerer/default-new 已删除；sample/tests 走显式、可检查的 registration |
| absent registration 的多段 monotonic allocation 会在末段 OOM 后留下不可达 bytes | high | `fixed` | table capacity 先准备，随后 Entry+key+help 单 allocation；逐 backing call fault sweep + retry 通过 |
| freeze/read 或 callback 重入可能持锁调用用户代码 | high | `disproved` | creator 与 visitor 锁外；release/acquire publish；race/reentry/TSan 均通过 |
| demangle 文档可能把 caller-owned allocation-free 误写成实现零分配 | medium | `fixed` | public/user/architecture/performance 均明确 Itanium malloc/free 与 Windows serialization/temporary |
| 一个 demangle symbol 仍使 shared `hyle_util` 带上 Registry 的 public dependency closure | low | `accepted` | `otool -L` 实证；当前 sample 同时消费两项且没有 material demangle-only consumer，出现该消费者即 reopen |
| Mac selected smoke 首次未声明 14.4 deployment target | low | `disproved` | 这是已公开的父工程契约；设置 `MACOSX_DEPLOYMENT_TARGET=14.4` 后三种 journey 全通过 |

## Closure decision

- 各适用门最终 verdict：scope、architecture、correctness、delivery、usability/docs、red-team、P0
  engineering performance 全部 `PASS`；Formal 为 `NONE`。
- blocker：无。候选没有未裁决的 correctness/security/ownership/交付或 P0 cost-structure blocker。
- 已知限制：case-sensitive key；枚举顺序未指定；`FrozenView`/返回 view 是 owner-bounded borrow；无
  unregister/dynamic unload/quiescence；demangle 可能回显原名并在平台内部 allocation/serialization；
  util shared component 当前仍带完整 Registry 依赖闭包；没有端到端 inference 性能证书。
- reopen triggers：真实消费者暴露本边界内 correctness/ownership/API 缺口；Registry/demangle 算法、
  公开合同、直接依赖或交付形状改变；依赖模块改变实际语义；成熟同合同 peer 证明更好复杂度档；
  profile 证明此处成为 material 热点；demangle-only consumer 证明当前 component closure 成本 material。
- terminal receipt：尚未写入；本提交只建议机械状态进入 `VERIFIED`。
- 建议机械状态：`VERIFIED`。
