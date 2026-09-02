# container_intrusive v4 能力扩展

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`VERIFIED`。本页已冻结本轮用户任务、能力账本、预算与验证边界；当前不可变候选为
> `58431ddbef6f54f9d4275b0fea2d317cba00b9f3` / tree
> `8df2b7d7806918d03b5bb5c31611adb196f30c7f`。权威状态是
> [`module-status.json`](../../module-status.json)。

## Reopen trigger

- component / owner / round：`container_intrusive` / Container owner /
  `container-intrusive-v4-capability-extension`；
- trigger ID：`INTR-CAPABILITY-EXTENSION-2026-09-02`；
- baseline：`b4915574020cc7da857d2871c545a3489c75288b`；
- 新事实：owner 明确要求把 v3 已记录的模块化常用负空间重新作为产品能力裁决；全局能力账本的
  `INTR-A-06/07/10`、推理 watchlist 的 `INTR-WATCH-01/05`、ggml 的一指针 object chain，以及当前
  Folly/Boost/oneTBB 的 counted/slist/list-algorithm 实际消费共同证明这些不是 benchmark-only symbol；
- 旧结论受影响范围：v3 对 counted list、单向链/hlist 和完整非 owning sequence 算法的 `DEFER`，以及
  “只有双向 member-hook primitive”这一公开能力边界；v3 已签的现有双链 correctness 不被无界重审，
  但默认 layout/codegen 必须作为回归门。

锁定 peer identity（本轮审计可从中为每个 family 最多选择两个）：

- Boost `78505eab9b203e3e26e83893d0685f160133fa0f`：Intrusive list/slist、constant-time-size 与
  sequence algorithms；
- Folly `176c088840724715b2c3c6211b2b347611676002`：counted waiter、Arena slist、atomic publish 和
  runtime 消费形状；
- oneTBB `b03a694ee0fe4b188bc77daa8a567551e3e3ebfc`：runtime counted intrusive list；
- ggml `d4716378882593333721eb33f153144b6885caf2` 与 llama.cpp
  `b96806d96061049a5b574269b049bf6241d63d46`：北极星基础/真实消费差异输入。

## Closure card（`SCOPED`）

- component / owner / round ID：`container_intrusive` / Container owner /
  `container-intrusive-v4-capability-extension`；task kind：`REOPEN`；
- baseline：`b4915574020cc7da857d2871c545a3489c75288b`；trigger：
  `INTR-CAPABILITY-EXTENSION-2026-09-02`；
- 使用原型：
  - `NORTH_STAR_JOURNEY`：ggml arena/object/deferred chain 在 caller-owned stable node 上 append、遍历和
    bulk release，每节点只应支付一个 link；
  - `PROVISIONAL_ARCHETYPE`：wait-set/LRU/cache membership 高频 mutation 后 O(1) 查询节点数；
  - `REAL_CONSUMER`：Sched cancellation callback 在外部递归锁下注册、任意注销、detach-before-fire；
  - `PROVISIONAL_ARCHETYPE`：预分配节点作为非 owning sequence 做稳定 sort/merge/filter/dedupe；
- 范围：counted safe-list、一指针 forward-list、hlist，以及双链/forward-list 的完整常用非 owning
  sequence algorithms；现有 auto/safe 双链只接受保持旧 layout/codegen/语义的必要内部泛化；新增头
  owner、Container façade、直接消费者和三种 C++ 交付；correctness、易用性、文档和适用 P1 证据；
- 非目标：owning storage/allocator、内部同步、atomic publish、lock-free reclamation、intrusive
  tree/hash/heap、base/function/value traits、fancy/offset pointer、Sched/IO 执行策略、Native FFI/C API、
  Formal 证书和公共性能仪器重写；
- 直接依赖：保持 `PUBLIC core`。候选直接消费者为 `timer_queue`（旧双链回归）和 Sched cancellation
  callback（hlist）；Mem forward-chain 只做语义扫描，不跨其活动轮强迁移；
- tier / margin：`P1` / W1-WIN 5%。W1-WSL、M1-MAC 只承担 correctness、sanitizer、ABI 与交付；
- machine / instrument：`W1-WIN` + `closure-perf-v1`；correctness 用 `closure-correctness-v1` 的适用门；
- strong peers：每族从本页锁定的 Boost/Folly/oneTBB 中最多选两个；hlist 的正常门使用 HYLE 当前
  cancellation callback consumer baseline，不把没有同合同 peer 的结果写成 SOTA；
- 资源/复杂度合同：默认旧 hook/list 仍各两个 pointer；counted 只给选择它的 list 加一个 count word；
  forward node 一个 pointer、list 两个 pointer；hlist node 两个 pointer、list 一个 pointer；所有 hot
  mutation 零分配、无锁、无 syscall。known-node 双链/hlist mutation O(1)，forward predecessor mutation
  O(1)，counted single/whole transfer O(1)，sort O(n log n)，其他 traversal/filter O(n)；
- API 档位：按 `CountedIntrusiveList`、`IntrusiveForwardList`、auto/safe `IntrusiveHList` 的语义命名，
  不建立 fast/light 入口。counted range transfer 必须提供普通 O(k) 计数路线和显式 verified-count O(1)
  路线；forward-list 缓存 tail 以自然支持 arena append/whole splice，不承诺 arbitrary node O(1) unlink；
  hlist 不伪装可排序 sequence；
- 调查/性能预算：设计、实现和验证总计一个工作日内拆成独立提交；性能人工归因最多 120 分钟，每个
  形状最多一次复现、一次 codegen/profile 机制检查、一个产品修改和一次重跑；不发明第五种姿势；
- correctness gates：normal/empty、copy/move/destroy/link retirement、same/cross-list splice、count
  invariant、iterator/range、throwing predicate/comparator 的结构有效性、fixed-seed differential、Debug
  misuse、ASan/UBSan、支持编译器 PMD ABI、公开头 self-containment、直接消费者；并发/ABA/reclamation
  明确 `NOT_APPLICABLE`；
- delivery gates：精确 factor source build；Container 产品 source/static/shared relocated consumer；新增
  split-header owner 与 direct Sched edge；header-only factor 不制造 binary；
- 性能 workload：counted wait-set mutation+size 对 Boost counted；arena append/traverse/drain 对 Boost
  cached-tail slist；hlist callback register/unregister/fire 对当前 faithful Sched baseline；稳定
  sort/merge 对 Boost intrusive sequence。先做语义/计费矩阵，旧 v3 workloads 只作回归；
- 停止条件：全部 `MUST` 完整或自然 routed、API/成本冻结、适用门有 verdict 后 `CLOSED`；material
  correctness/delivery/P1 未决写一个具名 blocker 后 `BLOCKED`，不得无限扩建仪器。

## 有界能力审计输入

待冻结的四个能力族如下；名称和操作面仍须由普通 C++ journey、生命周期与成本合同验证，不能从 peer
symbol 或旧 benchmark 直接复制：

1. **可选 O(1) membership count**：独立 counted safe-list，不给现有 uncounted list/hook 增字段；
   single/whole-list mutation O(1)，range splice 明确区分扫描计数与 caller-provided count 的 O(1)
   路线。
2. **一指针 forward membership**：arena object chain、单 owner free/deferred-work/FIFO chain；节点只付
   一个 pointer，显式 retirement，不能虚假承诺 arbitrary-known-node O(1) unlink 或 auto-unlink。
3. **hlist membership**：`next + prev_next`、head 一指针、已知节点 O(1) unlink/replace；与完整双向 list
   的区别必须来自 bucket/head footprint 与状态机，不以“更快”命名。
4. **非 owning sequence algorithms**：`reverse/remove_if/unique/merge/sort` 等只有在稳定性、异常策略、
   iterator/lifetime 和无分配复杂度完整时才进入 public surface；不复制 base/value/fancy-pointer 等
   Boost policy matrix。

审计同时扫描 Mem/Sched/IO 的私有链结构。较高 factor 可直接增加
`PUBLIC/PRIVATE container_intrusive` 边；不得因依赖顺序复制同义 primitive。以下特化先作为反例保留：

- freed slot 上覆盖 next 的 pool free-list，不能让 live object 为公共 hook 付费；
- waiter 摘链后复用同一 storage 的锁外 wake chain；
- Win32 kernel completion/inflight/free-state 复用的 IOCP link。

cancellation hlist、arena/deferred-work forward chain 和普通 counted membership 必须在新类型成立后重新
裁决，能自然迁移才增加真实直接边；不能为了证明新 API 有消费者而强行迁移状态机。

## Capability-family coverage map（`AUDITED`）

| Family | admitted journey / evidence | lifecycle / semantic variants | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| 双向 member membership | TimerQueue、FIFO/LRU、known-node cancel；v3 + Boost/Folly | auto/safe、same/cross-list transfer、node/list-first retirement | 两种现有 list | `COMPLETE` | 只做回归，不改默认 layout |
| 可选 membership count | wait-set/LRU/cache 的 mutation 后 O(1) count；Folly counted + oneTBB runtime | safe-link；single/all/range transfer；count overflow 不可发生于可表示对象数 | 无公共路线 | `MISSING` | 默认 uncounted 不付 count word/write；auto-unlink 不可与隐藏 count 组合 |
| 一指针 forward sequence | ggml object chain、Folly Arena cached-tail slist | explicit retirement；front/back、predecessor mutation、whole/range handoff、list/node teardown | 多个 owner 私有 next chain | `MISSING` | 不承诺 arbitrary-node O(1) unlink、auto-unlink、内部同步或 reclamation |
| hlist membership | HYLE CancelCallback register/unregister/fire；成熟 kernel-style `next + prev_next` | auto/safe；head/known-node insert/remove/replace；detach-before-callback | Sched 手写 record links | `MISSING` | 不是 bidirectional/sorted sequence；不缓存 tail/count |
| 非 owning sequence algorithms | 预分配节点稳定 sort/merge/filter/dedupe；Boost list/slist | stable sort/merge；throwing callback 后结构/ownership 有效；双链与 forward | `for_each/consume_all`，其余缺失 | `PARTIAL` | hlist 明确不取得这套算法；advanced disposer/policy route Boost |
| atomic publish / concurrent reclamation | Folly atomic list、Sched remote publish | producer/consumer、ordering、ABA、retirement | owning concurrent owner | `ROUTED` | 本轮不准入普通 hook；未来由 Container/Sched/Mem-reclaim 独立类型拥有 |
| base/value/fancy pointer 与 intrusive tree/hash/heap | Boost policy matrix | shared-memory pointer model、index/key/allocator policy | 直接使用成熟依赖或相邻 owner | `ROUTED` | 不为 API 数量复制 framework |
| factor/product delivery | 低层 factor 独立构建，用户只选 Container | split header owner、source/static/shared、无 binary/FFI | 旧单头完整 | `PARTIAL` | 新头与 Sched direct edge 待实现 |

### Public-surface 与语义重复反向追踪

- 现有 `IntrusiveListHook`、`SafeIntrusiveListHook`、双向 iterator/mutation/splice/consume/typed swap 全部
  归入第一族；没有 benchmark-only public 入口。
- `TimerEntry` 继续使用 safe 双链；其跨 bucket count 是 timer owner 的业务计数，不迁移 counted list。
- `ConditionVariable` 的 atomic waiter count 是无 waiter 时跳过锁的发布协议；waiter link 摘链后又复用为
  锁外 wake chain，因此不是普通 counted membership，也不迁移。
- `mem::GrowablePool` 把 next 覆盖到已释放 slot；`SharedPool` 使用 parallel atomic index；两者若改用
  public hook 都会增加 live object/slot 成本，合法保留。
- `CancelCallbackRecord` 的 `next + previous_link` 与新 hlist 完全同合同；Sched 可以依赖更低的
  `container_intrusive`，本轮把该迁移列为直接消费者验证。
- `CancelNode` sibling tree、pending-delete、Region block、IOCP inflight/free-state 同时带 parent、allocator、
  kernel 或 storage-reuse policy；本轮只登记后续 owner 回审，不为了增加调用点改写。
- ggml/llama.cpp 没有 counted/hlist 公共底座，但 ggml object chain 和推理 runtime 的稳定 node/callback
  journey验证 forward/hlist 不含模型语义；Folly/oneTBB/Boost 用于通用性与算法档位，不做 symbol 并集。

## Capability / gap ledger 与冻结 `FIX`

| ID | class | source / normal shape | owner / contract | complexity / cost | current/action | verification / negative space |
|---|---|---|---|---|---|---|
| `INTR4-BIDI-01` | `MUST` | v3 TimerQueue 与普通双向 membership | `container_intrusive`；caller-owned、external sync | 旧 hook/list 各 2 pointer；已知 mutation O(1) | `COMPLETE/KEEP` | 旧 tests/codegen/P1 replay 回归；禁止默认 count |
| `INTR4-COUNT-01` | `MUST` | Folly/oneTBB counted waiter/list | 独立 `CountedIntrusiveList<T,&T::safe_hook>` | list 额外 1 word；single/all O(1)；普通 range O(k)，verified-count overload O(1) | `MISSING/FIX` | count differential、move/swap/splice；不支持 auto hook |
| `INTR4-FWD-01` | `MUST` | ggml object chain + Folly Arena | `IntrusiveForwardListHook/List`；一指针 safe explicit-retirement node；list 缓存 tail | node 1 pointer、list 2 pointer；front/back/known-predecessor/all-splice O(1) | `MISSING/FIX` | std::forward_list/Boost differential；arbitrary node remove 仅 O(n) 或不暴露 |
| `INTR4-HLIST-01` | `MUST` | HYLE cancellation callback + stable hlist task | auto/safe hook/list；head owns membership，callback 前摘链 | node 2 pointer、head 1 pointer；known-node unlink/replace O(1) | `MISSING/FIX` | topology/random differential + Sched callback migration；无 tail/size/sort |
| `INTR4-ALG-01` | `MUST` | Boost list/slist 与普通 non-owning sequence | 双链和 forward 提供 `reverse/remove/remove_if/unique/merge/sort` | filter/reverse/merge O(n)，stable sort O(n log n)，零分配 | `PARTIAL/FIX` | std/Boost oracle；用户 predicate exception 后全部 node 仍恰属一个有效 list，顺序可部分改变 |
| `INTR4-ACCESS-01` | `MUST` | 三类 direct-member hook 共用 node recovery | factor-private helper；支持的 PMD ABI fail closed | compile-time/inline；无字段、分支或动态 dispatch | `PARTIAL/FIX` | 四编译器 header/codegen；不得扩成 public traits matrix |
| `INTR4-SCHED-01` | `MUST` | `CancelCallbackRecord` register/unregister/fire | semantic owner Sched；artifact owner factor hook；delivery edge Sched | footprint 不增加；锁、reentrancy、fire ordering 不变 | `MISSING/FIX` | sched cancellation correctness/stress；只迁 callback list |
| `INTR4-DELIVERY-01` | `MUST` | Container façade 三交付 | build/package delivery owner | header-only factor、无 binary | `PARTIAL/FIX` | split headers、exact factor、Container+Sched source/static/shared |
| `INTR4-PERF-01` | `MUST` | closure card 四个 normal workload | module + performance evidence | 5% peer NI 或预冻结 hlist consumer budget | `MISSING/FIX` | v2 contract；先语义等价，禁止从点值改 API/layout |
| `INTR4-ATOMIC-01` | `GENERAL` | Folly atomic publish | Container/Sched/Mem-reclaim future owner | 独立 atomic/reclamation cost | `ROUTED/DEFER` | 真实 producer/reclamation task 才重开 |
| `INTR4-POLICY-01` | `GENERAL` | Boost advanced policy/index | Boost or future domain owner | 不给普通 hook 付费 | `ROUTED/REJECT` | 只有两独立 HYLE journeys 且依赖路线不足才准入 |

本轮冻结 `FIX` 仅为 `COUNT/FWD/HLIST/ALG/ACCESS/SCHED/DELIVERY/PERF` 八项；任何 atomic、reclamation、
heap/tree/hash、allocator 或 traits 需求进入下一轮，不得顺手实现。

## API / usability freeze

- 状态：`FROZEN`；freeze identity：本 task 的 `AUDITED` 提交。
- 现有双链名称、两 pointer layout、pointer-based normal mutation、auto/safe 生命周期和 list move/swap 不变。
- counted 复用 `SafeIntrusiveListHook`，避免第三种 node hook；`.size()` O(1)。range splice 的普通重载扫描
  `[first,last)` 更新 count，另有显式 `count` 参数的 O(1) 重载，参数错误是前置条件违反；类型不满足
  `auto_unlink`，也不让 uncounted list 成为 `sized_range`。
- forward hook 的 unlinked state 是 self-next，使 tail 的 null-next 仍可区分 linked；hook 析构要求已显式
  退休。list 公开 `before_begin/begin/end/front/back`、front/back push、`insert_after/erase_after/pop_front`、
  whole/single/range `splice_after`、append/swap/clear/consume 和 sequence algorithms；缓存 tail 只增加 list
  一个 word，不增加 node 成本。
- hlist 提供 auto/safe 两种 hook/list，公开 forward iteration、front、push-front、before/after insert、
  known-node erase/remove/replace、pop-front、clear/consume/swap；head 只有一个 pointer。没有 back、reverse
  iterator、range splice、count、sort/merge；需要这些语义就选双链/forward-list。
- 双链/forward sequence algorithms 稳定地保留未删除节点和引用/iterator（只失效被删节点）；比较器/
  predicate 抛出的异常原样传播，结构与 membership 保持有效，已发生的 filter/merge 顺序变化不回滚；
  sort 必须在异常路径重新汇集所有临时链，不能让 node 静默变成 unlinked。
- 默认路径无 allocation、lock、atomic、syscall、owner pointer 或 dynamic dispatch。所有类型要求 direct
  member hook、64-bit 支持 ABI、linked 期间 stable address、外部同步；错误 owner/range/count 是程序合同，
  nullable/already-linked 只由具名 `try_*` 入口处理。
- 新公共头为 `<hyle/container/intrusive_forward_list.h>` 与
  `<hyle/container/intrusive_hlist.h>`；counted 留在 `<hyle/container/intrusive_list.h>`。普通用户仍只
  链接 `hyle::container`；维护者可精确构建 `container_intrusive` factor。
- 冻结后只允许内部算法、无可观察 cost 的 member-access 去重和 provider-free codegen 修正；正常
  consumer 的 material 证据若证明合同本身错误，必须退回 `AUDITED` 并更新有限 `FIX`。

## 已冻结的施工边界

- 范围：上述四族的 capability map、自然 C++ API、header-only 实现、复杂度/layout、correctness、
  直接消费者回审、Container façade/source/static/shared 交付、user/architecture 文档和适用 P1 证据；
- 非目标：owning storage、allocator、内部同步、lock-free publication/reclamation、intrusive
  tree/hash/heap、base/function/value traits、fancy/offset pointer、Native FFI/C API、Sched/IO policy、
  Formal 证书与公共性能仪器重写；
- 依赖目标：仍只 `PUBLIC core`；新增 header 继续由 `container_intrusive` factor 拥有并由
  `hyle::container` façade 聚合；
- tier：`P1`；热 mutation/遍历/批处理需保持零分配、无锁、无 syscall，并在同语义成熟路线存在时用
  `closure-perf-v1` 的 5% material margin；没有同合同 peer 的 hlist 只签预冻结 consumer/resource
  budget，不制造 SOTA 声明；
- 验证平台：W1-WIN 为工程性能机，W1-WSL/M1-MAC 为 correctness、sanitizer、ABI 和可移植性门；
- 停止规则：审计后只允许一个有限 `FIX` 集；每族最多两个 strong peer；P1 异常只允许协议规定的一次
  复现、一次机制诊断和一个产品修改。公共仪器缺口单独阻塞，不在本模块扩建仪器。

## 当前裁决

`VERIFIED`。能力 map、public-surface trace、owner、API/cost 与八项有限 `FIX` 已冻结；八项产品改动、
AppleClang 有界修复和性能 factor 选择器修复已经按冻结边界实现并绑定 candidate 3。旧 v3 性能证据
不能覆盖新增能力或变化后的产品 identity。candidate 3 的新增能力资格已经完成，全部适用门为
`PASS`；只剩 terminal receipt、限制/reopen trigger 和不可变 candidate LOC。

### 撤回的 candidate 1

- commit/tree：`d15b196f64dea5fda1671312a9b91adec77bd73e` /
  `0e1c82ab4b6b7e2897a68e0f79e06d40ee94e8f1`；
- Windows MSVC 19.50、WSL GCC 14.2 与 Mac AppleClang 21 的精确 factor correctness 都通过；
- AppleClang 21 对 `HYLE_ASSUME(function_call())` 发出 `-Wassume` 并忽略 optimizer assumption。受影响的
  是 forward hook retirement、counted invariant 与 hlist 前置条件；这不改变 Debug fail-fast 或本次
  correctness 结果，却不满足冻结的 Release codegen 合同；
- 前态 registry 快照：`%TEMP%/hyle-module-status-intrusive-v4-candidate-1.json`，SHA-256
  `F0FB6A532346129EF79E6EEEEDAA86D7F0FBBD30BC71087E398D31FC7CD2F3D4`；
- `CANDIDATE -> AUDITED` 后只允许把函数调用结果保存为局部值再传给 `HYLE_ASSUME`，并消除本轮新增测试的
  两个 unused equality warning。API、layout、算法、依赖、workload 与公共 Core 宏均不得改变。

### Candidate 2 identity

- commit/tree：`c63490f5b3508242d07168b6c65b4028aad6c76a` /
  `5e240cc3dc90fff1068f9d9580a4f9abba3e40e5`；
- `AUDITED -> CANDIDATE` 前态 registry：
  `%TEMP%/hyle-module-status-intrusive-v4-audited-2.json`，SHA-256
  `453694CFD8C392FE2DE39694DC7C82A4E82B613E25F506069BCCC8207A5C265F`；
- candidate 1 的结果不自动继承。candidate 2 已在冻结前重放 MSVC 19.50、GCC 14.2 与 AppleClang 21
  精确 factor；AppleClang 构建不再产生本轮新增 warning。候选后产品/API/CMake/workload 改动必须再次
  退回 `AUDITED`。

### 撤回的 candidate 2

- W1-WIN P1 在 configure、任何采样之前拒绝合同：`HYLE_COMPONENTS=container_intrusive` 把内部 factor
  当成公开产品；合法选择器是 `HYLE_DEV_FACTORS=container_intrusive`；
- 产品代码、API、算法、workload、peer 和统计门没有失败，也不得借此改变；
- 前态 registry 快照：`%TEMP%/hyle-module-status-intrusive-v4-candidate-2.json`，SHA-256
  `B5A2456F2D4434EF414D7C0DF03F6CA35271892494D8FA1ABCD66649F8EC2A1A`；
- `CANDIDATE -> AUDITED` 后只允许替换该 CMake cache key、刷新合同 lock hash 并建立新 candidate。

### Candidate 3 identity

- commit/tree：`58431ddbef6f54f9d4275b0fea2d317cba00b9f3` /
  `8df2b7d7806918d03b5bb5c31611adb196f30c7f`；
- `AUDITED -> CANDIDATE` 前态 registry：
  `%TEMP%/hyle-module-status-intrusive-v4-audited-3.json`，SHA-256
  `453694CFD8C392FE2DE39694DC7C82A4E82B613E25F506069BCCC8207A5C265F`；
- candidate 3 只相对 candidate 2 修正合同的 CMake factor selector；产品/API/算法/workload/peer、三平台
  correctness 与三种交付 identity 不变。性能 qualification 必须从新 clean checkout 与空 build/output
  开始，不能复用 candidate 2 的 configure 目录。

候选准备阶段已经完成以下非性能门：

- Windows MSVC 19.50 精确 factor 的 header、unit/differential、异常保证、no-exception consumer 与旧
  codegen gate 全部通过；Sched cancellation 真实消费者的 header/runtime 测试通过；
- WSL GCC 14.2 Release 精确 factor 的同一组测试通过，ASan/UBSan 已在本轮实现 identity 上分别通过；
- 中央 component contract 与 Container source/static/shared relocated consumer 通过，新头 owner、Sched
  direct edge 和 installed header closure 得到验证；
- `container-intrusive-p1-v2` 合同通过 schema/provenance 校验，lock hash 为
  `c10b5e062cb944f06ccb880a7ac1ff454f810cf28906d2b3afad84a0a3efe415`。

### Candidate 3 verification

- clean candidate checkout：
  `C:\public\project-hyle\_validation\container-intrusive-v4-candidate3-58431ddb`；fresh performance build：
  `C:\public\project-hyle\_validation\container-intrusive-v4-perf-build-58431ddb`；
- correctness：W1-WIN MSVC 19.50 精确 factor 6/6、no-exception、exception guarantee、codegen、
  differential/header 与 Sched cancellation 2/2 通过；W1-WSL GCC 14.2 精确 factor、ASan/UBSan 通过；
  M1-MAC AppleClang 21 精确 factor 6/6 且无本轮新增 warning；
- delivery：中央 component contract、Container source/static/shared relocated consumer，以及 M1-MAC
  source/static/shared 产品 smoke 通过；`container_intrusive` 保持 header-only，不制造虚构 binary；
- P1：`container-intrusive-p1-v2` 从空 build/output 执行；n=15 只因统计功效升级为完整 fresh n=31，
  四项 workload × 八个 strata 全部通过 5% non-inferiority。counted combined exact
  `INCONCLUSIVE`，forward `PARITY`，hlist 与 sequence 分别 `WIN`，但胜出只在合同 peer/consumer
  边界内解释；
- controls：A/A 双顺序、5%/10% known-delta、timer floor、placement、artifact identity 与 build
  contract 全部 `PASS`；最终 `performance_gate=PASS`、`qualification_gate=PASS`；
- verifier：`qualification=QUALIFIED`、`paths_verified=true`、
  `product_gate_policy=consumer-evidence-v1`、`provenance_compatibility=EXACT`、
  `current_certification_policy_eligible=true`；
- retained manifest：
  `C:\public\project-hyle\_validation\container-intrusive-v4-perf-evidence-58431ddb\manifest.json`，
  SHA-256 `EB3BF0A4BFED3CB39BF61A239487963FE6D889B13721CC5D996E7057F8DD4283`；992 observations、
  460 raw logs、6 artifacts；完整数表与边界见
  [性能证据](../../evidence/performance/benchmarks-container-intrusive.md)。
- final red-team：按冻结预算复核默认双链成本、counted/forward/hlist 生命周期与 owner 前置条件、
  throwing sort gather、Sched detach-before-fire、split-header owner、公开/内部依赖边和用户选择指南；未发现
  material correctness、复杂度、交付或 benchmark-shaped API finding。`check-module-closure.py`、
  `check-docs.py`、performance contract validation 与 `git diff --check` 全部通过。

剩余动作只允许生成 candidate LOC、写 terminal receipt 与限制/reopen trigger，并按状态协议迁移
`VERIFIED -> CLOSED`；不得借收尾重开产品代码或 API。
