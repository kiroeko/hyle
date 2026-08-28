# util_core 能力完整性存量回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)回放到已经 `CLOSED` 的
> `util-core-v1`。这是 product-non-mutating 的存量审计：它不把新版方法倒写成旧轮事实，也不替代
> [`module-status.json`](../../module-status.json) 的当前状态。

## 回放合同

- component / replay：`util_core` / `util-core-capability-replay-v1`；
- reviewed state / round：`CLOSED` / `util-core-v1`；
- closure task：[`util-core-v1 closure`](util-core-v1-closure.md)；
- candidate / tree / end / registry anchor：
  `27d73a610b13e6c07a3c3a32f491f8af32252ee1` /
  `6f84e0b4c78f3cbc9a27781ff2d6e9d6aac75e95` /
  `8d32916d2e29a2f914d8862f88538c110b5b229f` /
  `8d32916d2e29a2f914d8862f88538c110b5b229f`；
- replay base：`1ff605ef45c2c6c7994c442da900646e0cab5235`；reviewer：Codex；日期：
  2026-08-28；
- normal consumer：HYLE native C++ 模块和 source/static/shared SDK 用户；北极星推理引擎会需要
  非安全采样随机流、热对象的单指针强所有权、临时 rollback、极短状态锁和编译期诊断标签；
- tier / instrument：`P1 / closure-perf-v1 / 5%`。五个实现 header、冻结 peer、contract 和 retained
  evidence 均未漂移，本回放 timing budget 为 0；不为没有产品触发器的 primitive 制造新 benchmark；
- 调查预算：30 分钟；一个 fresh Windows `util_core`-only build、一次当前 source/static/shared
  selected-component smoke，每族最多两个已钉死 strong peers；
- strong peers：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 xoshiro/ScopeGuard/
  MicroSpinLock，oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 的 spin mutex/backoff，
  Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 的 Cleanup，Boost
  `df8eeb90846648464e3c8e378e312f0cec772e83` 的 intrusive pointer/type-name；标准 URBG、smart
  pointer、Lockable 和 C++23 `<scope>` 是语言合同，不再充当额外性能 peer；
- 范围：五个公开 primitive 的 capability families、全部 public surface、生命周期/失败/边界、真实
  consumer、复杂度/资源、依赖、三种交付、易用性与负空间；
- 非目标：修改 API/算法/布局，重跑纳秒性能，替 Container/Sched/Graph 等 owner 收口，恢复 C 或
  Native FFI，或把成熟库 API 求并集。

candidate 到 replay base 的五个实现 header 为零 diff。回放只校准 installed comment/user 文档中三项
过强措辞，并把“随机得到 32 个不同首值”的概率测试换成确定性的 TLS 状态隔离测试：
`std::random_device` 不保证 OS 熵，默认 TLS 流不保证跨线程/DSO 唯一或不重叠，`type_hash` 允许碰撞，
`make_ref`/自定义 teardown 的异常边界必须显式。公开声明、算法、对象布局、依赖和运行时代码均未改变。

## 自顶向下 capability-family map

| Family | 已准入用户旅程与来源 | 必须覆盖的变体、生命周期和边界 | 当前路线 | Coverage / action | 缺失边或负空间 |
|---|---|---|---|---|---|
| deterministic statistical bit stream | 推理采样、仿真、测试；标准 URBG、Folly xoshiro | 显式 seed、64/32/bool/float draw、复制 state、全零非法状态、零常态 allocation/lock | 32B `Rng` / xoshiro256++ / SplitMix64 | `COMPLETE / KEEP` | 非密码学；安全材料转 OS/vetted crypto |
| unbiased range and distribution composition | bounded index、shuffle、标准分布；Lemire/URBG | u32/u64 全闭区间、全宽/单点、finite double 半开区间、随机访问 shuffle、标准 distribution 互操作 | Lemire bounded draw、iterator/callback shuffle、URBG `operator()` | `COMPLETE + ROUTED / KEEP` | Gaussian/discrete 等分布由 `<random>` 完整承载，不复制 API；token top-k/top-p 属推理 owner |
| replay and parallel stream partition | 可复现实验/worker 分流；xoshiro published jump | get/set state、discard、固定 jump/long-jump、非法 state、owner-per-thread | `RngState`、`discard`、2^128/2^192 jump | `COMPLETE / KEEP` | 不把默认 TLS convenience 当可复现或数学不重叠合同 |
| default thread-local statistical stream | 普通无需复现的 per-thread 随机；成熟 TLS PRNG | 首次播种失败、线程隔离、稳态无锁、DSO/entropy/security 边界 | `thread_rng()` + best-effort random_device/salt/sequence | `COMPLETE / KEEP` | 不保证 OS entropy、唯一 stream ID 或不重叠；需保证时走显式 seed+jump |
| intrusive strong ownership | 热对象/单指针 owner；Boost/Folly intrusive pointer | initial +1、adopt/retain、copy/move/reset/release、const/upcast/casts、failed dynamic rvalue、final destroy、并发 handoff | `RefCounted<T>` + one-pointer `Ref<T>` | `COMPLETE / KEEP` | weak、cycle、aliasing/non-intrusive owner 完整路由 `std::shared_ptr/weak_ptr` 或领域 handle |
| allocation and custom intrusive teardown | 普通 `new`、allocator-owned object、OOM/constructor failure | throwing make、fallible owner factory、same-allocator destroy、no-throw release | `make_ref`；owner factory + `adopt_ref` + custom `destroy` | `COMPLETE + ROUTED / KEEP route` | util_core 不可反向依赖 mem/error；若两个 owner 重复同一 factory 合同，在 mem/owner 立项 |
| lexical once-cleanup | acquisition rollback、多早退、显式 commit；Abseil/Folly/C++23 scope | once、dismiss/invoke、move、storage/transfer failure、throwing callback、逆序 teardown | `ScopeGuard` / `HYLE_DEFER` | `COMPLETE + ROUTED / KEEP` | failure-only 用 guard+dismiss 或标准 scope_fail；typed resource/unique_resource 由 owning RAII 承载 |
| compact short critical-state lock | Mem/Sched/Graph 真实状态槽；oneTBB/Folly | Lockable、try/unlock、TTAS/backoff、1B、thread blocking、公平/递归/诊断边界、supported ISA pause | `SpinLock` / `cpu_relax` | `COMPLETE / KEEP` | 长/高争用走 mutex，fiber wait 走 Sched；timed/fair/RW 不是本 primitive 的半项能力 |
| compiler-local type diagnostics | compile-time logging/labels；Boost/compiler pretty function | qualifiers/arrays/functions/incomplete、static borrow、unsupported compiler、hash collision、TU/plugin/build 边界 | `type_name<T>` / FNV-1a `type_hash<T>` | `COMPLETE + ROUTED / KEEP` | correctness/persistent ID 用 schema/UUID；动态 RTTI spelling/demangle 在 util-runtime |
| component delivery and adoption | 独立低层组合、三个 SDK 形态；当前真实 consumer | self-contained headers、direct edge、no runtime artifact、source/static/shared discoverability | `hyle::util_core -> core` INTERFACE component | `COMPLETE / KEEP` | Native FFI 无具名 consumer，按全局策略 `OUT_OF_SCOPE` |

所有当前 `MUST` 都为 `COMPLETE` 或有用户可发现的完整 `ROUTED` 路线，不存在 `PARTIAL + KEEP`。
路由只签署 util_core 用户能走到正确 owner，不替标准库、Mem、Sched、Container 或推理层收口。

### 成熟实现差异裁决

- Folly/Abseil/标准库更宽的 secure random、distribution、scope-fail/success、unique-resource、weak/
  aliasing ownership、blocking/fair synchronization 并不是五个 primitive 各自“缺一半”；它们分别由
  crypto、`<random>`、显式 commit/`<scope>`、typed RAII、`shared_ptr` 和 Sched/标准同步完整承载；
- 当前 oneTBB AArch64 `machine_pause` 使用 `isb sy`，当前 Folly AArch64 pause 在 Armv8 使用 `isb`；
  HYLE 的 ARM hint 与两个钉死 peer 同族，不是为旧 benchmark 独创的 workload 特化；
- `make_ref` 不吸收 allocator/error 依赖。侵入式 pointer 负责 ownership，不负责所有 owner 的对象
  construction policy；需要 no-throw OOM 的模块在自己的 Result factory 中完成 allocation/construct/adopt；
- compile-time hash 是诊断 label，不是 type registry 的绝对 identity。碰撞处理或显式 schema 属消费
  protocol，不能通过改一个 hash 算法假装消除。

## 自底向上 public-surface trace

生产反向计数扫描 `include/hyle`/`src`，排除 util_core 自身、tests、bench 和 third-party；它只说明
当前仓内采用，不把“零内部调用”伪装成删除面向 SDK 用户的定义性能力理由。

| 当前入口 | Capability / journey | 生产反向证据 | 裁决 |
|---|---|---|---|
| `RngState`、`Rng` seed/state/draw/range/shuffle/discard/jump、`thread_rng` | statistical stream 全生命周期 | 当前内部 0；north-star sampling journey、64 个 tests 中 23 个 RNG case、installed sample 覆盖 | `KEEP`；标准 URBG 让普通分布无需 HYLE wrapper |
| `RefCounted`、`Ref`、make/adopt/retain、casts、comparison/hash/swap | intrusive strong owner 全生命周期 | 当前内部 0；16 个 Ref case、installed unordered owner sample 覆盖 | `KEEP`；没有为了 benchmark 添加 control block/metadata |
| `ScopeGuard`、`HYLE_DEFER` | lexical rollback/commit/teardown | 当前内部 0；10 个 failure/move/early-return case、installed sample 覆盖 | `KEEP`；优先 typed RAII 的选择边界已文档化 |
| `SpinLock` | short thread lock | 15 处 / 5 个 production 文件：Mem 1、Graph 3、Sched 11 | `KEEP`；各临界区长期适配性由 owner 回审 |
| `cpu_relax` | atomic polling hint | Sched 1 个真实调用；Container 两个 private adaptive hint 拼写是既有相邻 backlog | `KEEP`；Container park/wait 算法不在本轮越界改依赖 |
| `type_name/type_hash` | compile-time diagnostic label | 当前内部实际调用 0；10 个 compile-time/shape case、installed sample 覆盖 | `KEEP`；runtime demangle 与持久 identity 均已有明确 route |

中央 graph 与真实 target 保持 `util_core -> PUBLIC core`；Mem/Sched 是 PUBLIC consumer edge，Graph 是
PRIVATE edge。util-runtime 已删除旧的 util_core 采用并正确移除该边，因此旧 closure 中“util PUBLIC
util_core”的事实不再冒充当前依赖。没有发现 util_core 内部调用 Error/Platform/Mem 的反向依赖，
也没有 C bridge、动态导出、全局 registry 或 benchmark-only 元数据。

相邻债务仍按 owner 保留：Container 的两个 adaptive lock 拥有自己的 park/wait 状态机，是否复用
`cpu_relax` 在 Container 轮裁决；Sched 中锁内构造/移动 generic `T` 的路径在 Sched 轮裁决；Graph
首错槽与其他未关闭模块的 SpinLock 选择不能反向污染 util_core verdict。

## 当前验证

| Gate | 当前结果 | 证据边界 |
|---|---|---|
| implementation identity | `PASS` | candidate→replay base 的五个实现 header 零 diff；回放后仅 comment/test/docs 校准，声明/算法/布局/运行码不变 |
| fresh Windows util_core-only Release/Werror | `PASS`，19/19 | VS 18 2026 / MSVC 19.50；只解析 `core,util_core`，含 64-case binary、header self-contained 与公共 tooling gates |
| source consumer | `PASS`，1/1 | generated parent `add_subdirectory()`，只请求/链接 `hyle::util_core` |
| relocated shared SDK consumer | `PASS`，1/1 | exact-version package；header-only util_core 不制造伪 binary artifact |
| relocated static SDK consumer | `PASS`，1/1 | exact-version package；安装闭包只有 Core + 五个 util_core headers |
| peer/ISA shape | `PASS` | pinned Folly/oneTBB/Abseil/Boost commits仍与 v1 相同；ARM pause 注释已由当前 peer source 核实 |
| P1 engineering performance | inherited `PASS` | candidate、contract、peer、toolchain和 retained bundle 未漂移；Ref handoff/short SpinLock 全 strata 过 W1 5% NI |
| retained evidence identity | `PASS` | manifest 仍存在且 SHA-256 为 `7027954068A7B5C1DDFC1ED2D83A3646F951694B90A19939FFD445ADF88FFEFB` |
| new nanobenchmark | `N/A` | timing budget 为 0；无运行实现、consumer contract 或性能 identity 漂移 |

本回放没有重跑 Linux/macOS，也不把 v1 的历史三平台 correctness/交付升级成新 Formal 证书。
Formal 仍独立 `PENDING`；RNG 只继承成熟算法档、correctness 和历史 diagnostic，不宣称全部操作或
全部平台 SOTA。

## 裁决与下一动作

- 产品 verdict：`CONFIRMED`；`util_core` 保持 `CLOSED`，没有 `PRODUCT_TRIGGER`；
- 方法 verdict：旧 v1 ledger 已覆盖五族、peer、consumer 和复杂度，但没有把每族的取得、组合、失败、
  teardown 与 route 独立签为 coverage。本页补齐该证据，不倒写旧过程；
- 功能 verdict：五个窄 primitive 各自完整，未发现应加入 util_core 的常用能力族；更宽能力都有自然
  标准库或 owning-module route，不存在 `PARTIAL + KEEP`；
- 性能 verdict：Ref owning handoff 与短 SpinLock 保留 W1-WIN 5% P1 engineering `PASS`；Rng/Guard/
  type-name 只签成熟算法/复杂度/结构，不宣称 all-operation/cross-platform Formal SOTA；
- 易用性 verdict：公开形状保持自然 C++23；本回放明确补上 `make_ref` throwing、no-throw teardown、TLS
  seed 和 type-hash collision 四个陷阱，并删除一个把概率当合同的测试；
- 下一动作：存量采用清单进入独立的 `util` runtime（Registry/demangle）回放；Container/Sched 债务
  由其 owner 正常轮处理，不据此重开 util_core；
- 重开条件：真实 consumer 无法通过五个 primitive 或记录的 route 完成常用旅程；公开 API、算法、
  布局、direct dependency、supported compiler/ISA 或交付变化；同合同 strong peer 证明更好复杂度档
  或普通 consumer workload 上可复现 material 优势；或 retained claim identity 丢失而项目仍要续签。
  peer API 并集、LOC、零内部调用、单点微基准或相邻 owner 未收口不能单独触发重开。
