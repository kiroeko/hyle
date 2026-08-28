# container_intrusive 能力完整性存量回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)回放到已经 `CLOSED` 的
> `container-intrusive-v3`。这是 product-non-mutating 的存量审计：它不把新版方法倒写成旧轮事实，
> 不改变产品 candidate，也不替代 [`module-status.json`](../../module-status.json) 的当前状态。

## 回放合同

- component / replay：`container_intrusive` / `container-intrusive-capability-replay-v1`；
- reviewed state / round：`CLOSED` / `container-intrusive-v3`；
- closure task：[`container-intrusive-v3`](container-intrusive-v3-resume.md)；
- candidate / tree / end / registry anchor：
  `b5816b810d536f2041520e4b370c1e17d3ac7a4a` /
  `3bf043c40dbb5e4b49aba563f23f71b3c2d7caf3` /
  `178112c7723540a032d3ef14d2cb26742138e642` /
  `178112c7723540a032d3ef14d2cb26742138e642`；
- replay base：`b045b28c96b522bd6dec4c5f8e590ef91c13cfb1`；reviewer：Codex；日期：
  2026-08-28；
- normal consumer：`time::TimerEntry` 与 C++ SDK 用户把 stable-address、caller-owned timer/task/
  completion node 放入零分配双向链，执行 FIFO、已知节点 O(1) cancel、遍历、连续 range/整链 handoff
  和显式或自动 retirement；
- north-star journey：推理 runtime 的 timer、completion 和 backend-control node 在 owner storage 中
  保持地址稳定，初始化/服务期按 owner lock 组合链并批量交付；kernel/token 数据本身不按字符串或
  intrusive node 逐项分派，普通 owning sequence 继续选标准/HYLE owning container；
- tier / instrument：`P1 / closure-perf-v1 / 5%`。本回放 timing budget 为 0，不生成新纳秒样本；只
  复核复杂度、当前 codegen/correctness、已有 bundle 可重放性以及产品/peer/contract identity；
- 调查预算：30 分钟；一个 fresh Windows selected-component Release/Werror build、一次当前
  source/static/shared selected smoke、每族最多两个已钉死 strong peers；
- strong peers：Boost.Intrusive
  `df8eeb90846648464e3c8e378e312f0cec772e83` 的 list/member-hook/link-mode/算法策略；Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 `IntrusiveList`、`SafeIntrusiveList`、
  `CountedIntrusiveList` alias 与实际 waiter consumer；标准 `std::list` 只帮助区分普通 sequence
  任务，不充当同 ownership 性能 peer；
- 范围：caller-owned 双向 member-hook 的能力族、全部 public surface、当前/北极星 consumer、
  lifecycle/边界、复杂度/资源、依赖/交付、C++ 易用性和更宽 intrusive framework 的 route；
- 非目标：修改公开 API/算法/layout，重跑 P1 timing，替 time/sched/io/container 收口，新增 owning/
  concurrent container、恢复 C/Native FFI，或按 Boost API 数量扩面。

候选到 replay base 的 `include/hyle/container/intrusive_list.h` blob 均为
`9d13f4fbb7fbe3aeea3431a9072f3694b2ab3e8d`，tests、sample、直接 Core 语义与 time consumer 产品路径
也为零 diff。中央 component manifest 唯一相关范围内的文本 diff是 util 删除已失真的 `util_core`
依赖；`container_intrusive -> PUBLIC core` 与 `time -> PUBLIC {core,container_intrusive}` 均未改变。

## 自顶向下 capability-family map

| Family | 已准入用户旅程与来源 | 必须覆盖的变体、生命周期和边界 | 当前路线 | Coverage / action | 缺失边或负空间 |
|---|---|---|---|---|---|
| embedded membership construction | timer/task/completion stable node；Boost/Folly member hook | direct member、非首成员/non-standard-layout、multi-membership、default/copy/move、supported PMD ABI、地址稳定 | 两 pointer auto/safe hook + typed `List<T,&T::hook>` | `COMPLETE / KEEP` | base/function/fancy pointer 由 Boost 完整承载；未知 PMD representation 编译期 fail closed |
| auto retirement | owner list 未知时撤销、node 先析构；Folly auto list | `linked`、幂等 `unlink`、node/list 任意先后析构、copy/move 不偷 membership、外部同步 | `IntrusiveListHook` / `IntrusiveList` | `COMPLETE / KEEP` | auto unlink 不是并发 reclamation；storage lifetime 仍归 caller |
| owner-coordinated safe retirement | `time::TimerEntry` 与 waiter state machine；Folly safe list | 只经 list 摘链、linked-node destruction fail-fast、list 先析构、owner mirror 一致性 | `SafeIntrusiveListHook` / `SafeIntrusiveList` | `COMPLETE / KEEP` | safe hook 不暴露任意 unlink/swap；owner token/quiescence 属消费方 |
| sequence observation and traversal | FIFO 两端、诊断/批量扫描；标准 bidirectional range | empty、front/back null、mutable/const/reverse iterator、end identity、exception-preserving read traversal | pointer endpoints、C++23 bidirectional range、`for_each` | `COMPLETE / KEEP` | exact owner/contains 需要 scan 或 owner metadata，不向所有 hook 加第三个 pointer |
| single-node insertion and retirement | enqueue、已知 timer cancel、iterator erase；time/Boost/Folly | null/already-linked 可恢复形状、unchecked owner 前置条件、front/back/position、successor、空 pop | push/try/insert、pop、erase/remove、`iterator_to` | `COMPLETE / KEEP` | `try_*` 不冒充 owner/range validator；untrusted token 由 owner 验证 |
| zero-copy membership transfer | ready prefix、bucket/completion batch handoff；time/Boost | single/whole/range、same/cross-list、empty/boundary、iterator/end invalidation、owner mirrors | 三种 `splice`、`append`、list move/swap | `COMPLETE / KEEP` | arbitrary predicate 选择属于 owner；range position-inside-range 是前置条件违反 |
| detach-and-handoff lifecycle | completion drain、owner teardown；Boost disposer family、Folly free-list | callback 前摘当前 node、destroy/relink current、exception boundary、全部清空、不隐式 delete | `consume_all(noexcept fn)`、`clear`、`for_each` | `COMPLETE / KEEP` | allocator/delete policy 由 callback owner；单项 dispose 可自然 `erase` 后调用 owner disposer |
| typed membership substitution | movable auto holder；Boost hook swap/Folly holder shape | linked/unlinked、same/cross-list、adjacent/self、iterator invalidation、不同 member contract 不可混用 | auto `List::swap_nodes(T*,T*)` | `COMPLETE / KEEP` | safe replacement 必须同步 owner state；无具名重复 consumer，保持 `GENERAL / DEFER` |
| cardinality and backpressure | 冷诊断 count；Folly/Boost counted waiter；owner-specific queue state | linear/O(1)、auto/safe、range splice count、owner publication、move/swap/clear | 冷 count 用 `std::ranges::distance`；业务 count 留给 owner；通用 O(1) counted 直接选 Boost `constant_time_size` | `COMPLETE + ROUTED / KEEP route` | Folly counted 是 Boost alias，不是第二独立实现；其真实用例把 count 接入 waiter/state word。出现独立通用 consumer 后再设计单独 counted type，不给默认 list 加写税 |
| ordering/filtering algorithms | 需要 intrusive sort/merge/reverse/unique/remove-if 的通用序列；Boost/standard | comparator exception、stable ordering、disposal、复杂度和 ownership | 完整 intrusive 算法选 Boost；owning sequence 选标准/HYLE owning container | `ROUTED / REJECT from primitive` | HYLE hook 不承诺与 Boost hook 互操作；用户必须按任务在类型定义时选 route |
| alternate hooks/link structures | shared-memory/base hook、slist/hlist、normal-link；Boost | pointer model、unlink complexity、lifecycle、reclamation、ABI | Boost 或拥有状态机的 sched/io/mem 专用结构 | `ROUTED / OUT_OF_SCOPE` | 单库/单状态机形状不扩成公共类型；专用 storage reuse 不是重复实现缺陷 |
| synchronization and reclamation | cross-thread queues/waiters；Sched/Container owner | lock/atomic、wake/progress、cancel、ABA/UAF、epoch/hazard policy | 外部同步；并发 owner 使用 Sched/Container 机制 | `OUT_OF_SCOPE / REJECT` | auto destructor 绝不授权与遍历并发 |
| component delivery | 最小 source/static/shared C++ SDK；time direct edge | self-contained header、selected root、0 binary、relocation、无 FFI symbols | `hyle::container_intrusive -> core` INTERFACE target | `COMPLETE / KEEP` | Native FFI `N/A`；完整 container 不聚合该 target |

所有当前 `MUST` 都为 `COMPLETE` 或有用户可发现的完整 `ROUTED` 路线，不存在
`PARTIAL + KEEP`。`GENERAL` 的 counted/safe-replace/alternate-policy 能力均有 owner、触发条件和默认
成本解释，不以“当前没调用”作为唯一拒绝理由。

### Counted 与更宽算法的对抗裁决

- Folly 的三个 list 类型都直接 alias Boost.Intrusive；它证明成熟 C++ 消费形状，但不能与 Boost
  计作两个独立算法来源。Folly 的 counted 实际调用位于 `ThrottledLifoSem`、`TimedMutex/RWMutex`，
  count 被同一 owner lock/state word 用于唤醒或 reader 数协议，并非一个可脱离 owner 语义的免费查询；
- HYLE 的 time count 跨 radix buckets，Sched waiter count 还承担发布/fast-reject 协议；把 list-local
  count 取代它们会错误改变语义。为所有 list 加 count 会增加一个 word 与每次 mutation 写；单独 counted
  type 又必须裁决 range splice 是 O(k) scan 还是接受 caller-supplied count。当前明确 route 比半实现
  一个 `.size()` 更完整；
- `std::ranges::distance` 是 linear count 的自然、可发现路线，且不会让类型被误识别为
  `sized_range`。真正需要通用 O(1) intrusive size 的用户可从一开始选择 Boost
  `constant_time_size<true>`；出现不愿引入 Boost 的具名 HYLE consumer 时，才触发独立 modular type；
- sort/merge/reverse/unique/remove-if 的成熟实现涉及异常、stable order 和 disposer policy。它们不补
  本 primitive 的复杂度缺口；指南已要求该任务从类型定义时选择 Boost，普通 owning 数据选择标准/
  HYLE container。这里是完整 route，不是让用户手写易错平台或生命周期代码。

## 自底向上 public-surface trace

| 当前入口 | Capability / journey | 生产/验证证据 | 裁决 |
|---|---|---|---|
| 两种 hook 的 ctor/copy/move/assignment/destructor、`linked`、auto `unlink` | membership formation 与 auto/safe retirement | time 使用 safe；lifetime/copy/move/death tests | `KEEP` |
| list aliases、iterator typedef、begin/end/cbegin/cend/rbegin/rend | sequence traversal 与 iterator identity | range-for/sample、const/reverse/iterator tests | `KEEP` |
| `empty/front/back` | bounded observation | time 与 sample；empty/non-empty tests | `KEEP` |
| push/try-push/insert/try-insert | single-node insertion | time push；null/already-linked/position tests | `KEEP` |
| pop/erase/remove/clear | explicit retirement | time cancel/list teardown；normal/empty/lifetime tests | `KEEP` |
| `iterator_to` | O(1) known-node position | timer/pinned peer shape、direct tests | `KEEP`；不增加 owner scan |
| three `splice` overloads、`append` | O(1) single/whole/range handoff | TimerQueue/CommonTimeoutQueue；same/cross-list differential | `KEEP` |
| list move/swap 与 auto `swap_nodes` | list/membership transfer | deterministic + 4000-step model differential | `KEEP`；safe variant按 owner route |
| `for_each` / `consume_all` | read traversal / detached handoff | TimerQueue direct consumer；exception/noexcept/destruction tests | `KEEP` |

公开头没有 benchmark-only API、test flag、allocator、lock、owner/count metadata 或动态 dispatcher。完整
public surface 全部能反向映射到已准入 family；`detail::IntrusiveListLink/Impl` 只是 header implementation
seam，不是独立用户能力。

## 消费者、依赖与交付回审

- 唯一当前通用直接消费者仍是 `time`；它的 `TimerEntry`、TimerQueue 与 CommonTimeoutQueue 产品路径
  自候选为零 diff。safe hook、remove、range splice、append、consume_all 与 for_each 都是实际路线；
- Sched 的 ConditionVariable/Semaphore waiter 要在摘链后复用 storage 作 wake chain，并把计数接入
  publication/fast path；Cancellation hlist 与 IOCP/free-list 也有不同 link/storage 状态机。保留特化
  避免 footprint、锁内写和错误 lifetime，不构成“该用公共模块却没用”的漏采用；
- 中央 graph 与真实 target 均为 `container_intrusive -> PUBLIC core`，`time -> PUBLIC
  container_intrusive`；完整 container 不聚合。候选后的 build/package 变更没有改变这两条边；
- 当前 Windows selected-component smoke 的 source、relocated shared、relocated static consumer 各
  1/1。INTERFACE 安装只含 Core closure 与 intrusive header，没有 archive/DLL 或导出符号。

## 当前验证与性能边界

| Gate | 当前结果 | 证据边界 |
|---|---|---|
| implementation identity | `PASS` | candidate 与 replay base 的产品 header blob相同；Core 直接语义和 time consumer 路径零 diff |
| fresh Windows Release/Werror | `PASS` | VS 18 2026 / MSVC 19.50；selected `core,container_intrusive`，5/5 ctest；30/30 unit cases，header/codegen/parser 全通过 |
| source/shared/static delivery | `PASS`，各 1/1 | 当前树 selected-component source、relocated shared/static sample；header-only 0 binary artifact |
| historical cross-platform correctness | retained | v3 candidate 的 MSVC/clang-cl、GCC 14 ASan/UBSan、AppleClang gates；本回放没有冒充新三平台证书 |
| complexity/resource | `PASS` | 两 pointer hook/list；热 mutation O(1)，range relink O(1)，linear traversal/drain O(n)；零 allocation/lock/syscall/owner/count metadata |
| v3 P1 bundle integrity | `PASS` for deep replay | manifest 仍存在且 SHA-256 为 `3f081670022973a1b07ea6b4c04ab93b8d8fcbad61feef8bc6c584f30a10bd70`；当前 verifier 为 `QUALIFIED`、`paths_verified=true` |
| current product-policy renewal | `N/A` | post-v3 consumer-key policy 使旧 manifest 显示 `product_gate_policy=null` / `INCONCLUSIVE`；性能页已记录，它不倒撤当时 closure，也不能续签/Formal |
| new nanobenchmark | `N/A` | timing budget 为 0；没有产品/peer/复杂度或正常 workload trigger |

回放没有把旧 P1 点值重命名为当前 Formal/SOTA。若项目要求在新 consumer-key policy 下续签，必须使用
新 contract ID、真实 consumer provenance/ownership/material contribution 重新采样；这是 evidence
refresh trigger，不是当前链表算法变慢或 API 缺失的证据。

## 裁决与下一动作

- 产品 verdict：`CONFIRMED`；`container_intrusive` 保持 `CLOSED`，实现改动为 0，没有
  `PRODUCT_TRIGGER`；
- 方法 verdict：v3 已经审过 consumer、peer、negative space 和 public surface，但没有用
  `COMPLETE/ROUTED/OUT_OF_SCOPE` 独立表达覆盖。本页补齐两轴，并明确 counted/算法 route，不倒写
  旧过程；
- 功能 verdict：当前 auto/safe caller-owned 双向链 primitive 的取得、操作、组合、retirement、边界
  与交付完整；counted、完整算法、alternate hook 和并发 reclamation 都有明确 route/owner/trigger；
- 性能 verdict：复杂度与资源档保持 SOTA 同族；v3 的历史 W1-WIN 5% P1 工程非劣事实和完整 bundle
  保留，但不能按后来的 consumer-key policy 续签，不发布跨平台/全部姿势 SOTA；
- API/易用性 verdict：pointer mutation、nullable pop、auto/safe 选择、iterator invalidation、callback、
  private-hook friend adapter 和 unsupported task route 均可由 C++23 用户自然发现；Native FFI 为 `N/A`；
- 下一动作：项目采用清单进入 build/package development-unit replay；`time` 保持自己的 `AUDITED`
  round，不因本回放被修改或提前签结论；
- 重开条件：具名 consumer 不能用当前 primitive 或已记录 route 完成常用 journey；公开 API、算法、
  layout、PMD/compiler ABI、Core 语义、直接 edge 或交付形状改变；独立 consumer 证明 counted/safe
  replacement 等 GENERAL 能力应准入；同合同 peer 证明更优复杂度或正常 consumer 上 material 优势；
  或项目需要按新 policy 续签而旧 bundle 无法使用。API 数、LOC、单点微基准、Folly 对 Boost 的 alias
  或专用单链状态机不能单独触发重开。

回放 base 的 canonical LOC self-check（snapshot
`b045b28c96b522bd6dec4c5f8e590ef91c13cfb1`）：全库代码 **298,646**；自实现包含测试
**122,321**；自实现不包含测试 **82,176**；当前 `container_intrusive` 实现 **564** 行。LOC 只用于
规模复盘，不作为完整性或优先级证据。
