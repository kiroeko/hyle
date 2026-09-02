# container_intrusive v4 能力扩展

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`SCOPED`。本页先保存击穿 v3 negative space 的新事实，并冻结本轮用户任务、预算与
> 验证边界；完成两遍能力审计和有限 `FIX` 集以前仍不写产品实现、不开始计时。权威状态是
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

`SCOPED`。允许的下一步只有：完成四族自顶向下 journey、现有 public surface 反向追踪、私有重复/
直接消费者扫描，冻结 API/cost sketch、`FIX` 集、测试和性能 workload。尚无 candidate，旧 v3 性能
证据不能覆盖新增能力或变化后的产品 identity。
