# Container v2 当前源码重构与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮是从 `container-v1` 的 `BLOCKED` 状态恢复的全新审查。旧轮的 capability、API、correctness、
> performance 与 delivery verdict 全部只作历史线索，不能继承为本轮结论。原因不是单纯重复 review：
> 当前产品已在旧 candidate 之后改变了 Highway dispatch、Container 的 per-ISA 实现、依赖锁和构建/
> package 合同，旧 candidate identity 已经失效。机械状态只看
> [`module-status.json`](../../module-status.json)。进入 `AUDITED` 并冻结有限 `FIX` 集前，不修改
> Container 产品实现。

## Closure card（`SCOPED`）

- component / owner / round ID：`container` / data-structure owner / `container-v2`
- task kind：`RESUME`
- `RESUME` 前一 round / blocker / resume condition：
  - 前一轮：`container-v1`；
  - blocker：冻结的 P1 范围大于实际资格化的五个 workload，未运行的适用轴使聚合性能结论为
    `INCONCLUSIVE`；
  - 原恢复条件：在未变化 candidate 上补齐 strict/non-trivial sequence、string/sorted/ordered
    associative、并发 map/vector/queue、active set/ring、hash/text/interner 的有限 P1 verdict；
  - 当前裁决：原 candidate 不再“未变化”。从 `19a1e9b515522ad74872fb963a3b3260cf915bab`
    到本轮 baseline，Container 的 Highway dispatcher/kernel、`probe.h`、直接外部输入和三种交付机器
    已发生变化；因此本轮重新绑定当前产品，并从零审计，而不是续写旧性能结论。
- baseline commit / tree：`5cebdddf8dd40f48218b5eb15c03dc8d9910b8ef` /
  `8babbc6c1d54ffcc1fcfd8eefd5605011cf50e8f`
- 进入原因与 demand/defect ID：
  - `CONTAINER-V2-IDENTITY-01`：旧 candidate 之后，Container 的 Highway 生成式 dispatch、字符串/
    bitset kernel TU、Highway 锁定输入、component/package 合同发生产品级变化，旧验证不可继承；
  - `CONTAINER-V2-CAP-01`：以本地量化推理的 metadata、registry、handoff、active-set 和 loader staging
    任务重新枚举能力族，不把旧类名或 benchmark case 当成需求；
  - `CONTAINER-V2-SURFACE-01`：逐项反查当前约 11k 行非侵入式 Container 公开面，查找半项生命周期、
    错误 owner、benchmark seam、重复标准/第三方能力和不自然失败形状；
  - `CONTAINER-V2-PERF-01`：用当前统一仪器为实际热机制建立有限而完整的 P1 证据；冷操作只做 P0
    复杂度、allocation/copy/lock/syscall/metadata 与 codegen 门，不再次承诺无界 benchmark 矩阵；
  - `CONTAINER-V2-DELIVERY-01`：重新核对 `core/mem/platform`、Highway build-only seam、xxhash/wyhash/
    simdutf owner，以及 source/static/shared 的 selected-component 交付。
- 真实 consumer task / normal workload：
  - 推理引擎用短连续 sequence 组装 shape/batch/dependency，flat/string index 保存 token、tensor、operator
    metadata，stable/concurrent index 保存跨 grow 仍需 identity 的 registry；
  - scheduler worker 用 SPSC/MPMC/sharded/work-stealing 容器传递 typed payload，控制面用 dense/sparse
    active set；模型与协议 loader 用 byte/object ring、borrowed text、hash 与 UTF-8 primitive；
  - 当前仓库 `util`、`sched`、`io`、`graph` 的真实调用序列作为直接 consumer replay；下游未收口不
    影响其对 Container 合同的证据价值；
  - 工程计算以 read-mostly sorted index、持续 mutation 的 ordered range index、sparse active IDs 与
    streaming ring 作为泛化留出，防止接口被单一模型或 benchmark 过拟合。
- 范围：`include/hyle/container/**` 中除 `intrusive_list.h` 外的全部公开 C++ 面，`src/container/**`，
  Container-owned third-party adapters、直接依赖与直接消费者，source/static/shared 交付，architecture、
  user、implementation 与 performance 文档，以及有限 correctness/P1/red-team 证据。
- 非目标：已独立 `CLOSED` 的 `container_intrusive`；模型格式、tensor/KV、batching/serving policy；
  scheduler 的 wait/cancel/backpressure service；通用 cache/LRU/persistence；device memory；复制 std、
  Boost 或单个 peer 的全部 feature；Native FFI/C API、稳定 C++ ABI；下游整体收口与 Formal 认证。
- 直接依赖 / 直接消费者：中央产品边为 `PUBLIC core mem`、`PRIVATE platform`；Container 的目标绑定
  TU 通过 build-only `hyle_highway_headers` 使用锁定 Highway。外部产品输入为 xxhash、wyhash、
  simdutf 与 Highway；直接 HYLE 消费者为 `util`、`sched`、`io`、`graph`，并核对 package/sample。
- tier 与理由：`P1`。lookup、append、queue、active-set、ring、hash/text 会在推理 worker 或 loader
  正常路径重复；bulk build、reserve/rehash/grow、ordered rebuild、path normalize 和 teardown 等冷操作
  先按 P0 结构门，只有审计已绑定正常重复 journey 的代表操作进入计时。
- material margin / resource budget：W1-WIN 同合同 peer 或 immutable baseline 的 non-inferiority
  margin 为 5%。吞吐不能抵偿更弱 lifetime/progress/failure、额外 allocation/copy/lock/hash scan、
  更大的常态 per-entry metadata 或 material resident-memory 增长；不为 benchmark 新增公共 API、对象
  padding、动态分发或缓存字段。
- 总调查预算 / 性能预算：从 `SCOPED` 起最多 6 小时，其中统一性能取证最多 2 小时，独立红队最多
  30 分钟。若审计冻结的产品修复预计超过一个工作日，拆成可独立验证的有限批次；到预算必须
  `CLOSED`、具名 `BLOCKED` 或由 owner 显式重 scope。
- 最多复现、机制诊断、产品修改次数：每个 material 异常最多一次反序/复现、一次 codegen/profile/
  allocation/contention 机制检查、一个同时满足机制与 consumer 两把钥匙的产品改动及一次重跑；本轮
  不修改中央统计规则或发明新的公共仪器。
- machine profile：`W1-WIN` 负责 P1 工程裁决；`W1-WSL` 与 `M1-MAC` 负责适用 correctness、sanitizer、
  ISA fallback 与交付复核，不跨机器池化点值。Mac 只使用登记 QoS 合同。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 与当前 component/package contracts。
- strong peers（每族最多两个，均为当前本地 clean source）：
  - inline/general sequence：Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7`、Boost
    `78505eab9b203e3e26e83893d0685f160133fa0f`；strict fixed-capacity 以 CCCL
    `b7aaea69a2b07e50f09e67f2962da0243e0b7c5d` 替代 Abseil；
  - flat/sorted/ordered associative：Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7`、Boost
    `78505eab9b203e3e26e83893d0685f160133fa0f`；
  - concurrent associative/vector：oneTBB `a8216b95202f3c55f9749870f4f8cfa4df0b32e3`、Folly
    `23ebefdf96c116700d3e134f04d95a2b60a509de`；
  - bounded handoff/work sharing：Folly `23ebefdf96c116700d3e134f04d95a2b60a509de` 与
    rigtorp MPMCQueue `b9808ede08f26fa9df4df4e081d19cace8f6c6ea`；语义不等价时不强行比较；
  - dense/sparse membership 与 rings：Boost `78505eab9b203e3e26e83893d0685f160133fa0f`、Folly
    `23ebefdf96c116700d3e134f04d95a2b60a509de` 只用于能力、复杂度和同合同机制；
  - byte hash / UTF-8：锁定 xxhash `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53` 与 simdutf
    `2d4e249dcfd978b39dd567b156f20a43db97377b`，各自只裁决对应算法合同；
  - 北极星差异输入而非 API parity peer：独立 ggml
    `36da57138425487184aa1da2eee2cde155909c6f`、llama.cpp
    `9723942adc518b43c4b95dc4dce6906903eb5e09`；二者于 2026-08-31 对各自 `origin/master`
    执行 `fetch --prune` 与 clean fast-forward 检查。llama.cpp 随仓 `ggml/` 另作为该引擎实际使用的
    foundation snapshot，不用独立 ggml 的最新能力反向虚构 llama.cpp 已消费它。
- 无同合同 peer 时的 consumer budget 及禁止声明：使用本轮 immutable baseline，在相同 public
  journey、allocator、容量、线程角色和计费边界下通过 5% NI，并守住渐进复杂度、常态 allocation/
  metadata/lock 与 resident-resource 不恶化。只能声明具名 workload 的工程 non-inferiority，不能
  声称“Container 全面 SOTA”、跨平台固定纳秒或从不同语义 peer 外推胜负。
- 适用 correctness gates：normal、empty/zero、capacity/wrap/overflow、OOM/fault、invalid input、
  hash/equality/comparator、copy/move/destroy/self-alias/iterator/reference invalidation、non-trivial/
  move-only lifetime、并发 progress/quiescence、property/differential/random stress、ASan/UBSan/TSan、
  x86 scalar/SSE/AVX 与 arm64 scalar/NEON dispatch、public-header self-containment。
- 适用 delivery gates：中央 component/include-owner 对账；Highway/xxhash/wyhash/simdutf pin、offline、
  license 与 package projection；container-only source embedding、relocated static/shared C++23 consumer；
  无旧 C schema/bridge/fixture/Native FFI symbol。
- 有限 performance gates：candidate 同时冻结的 [`container-p1-v2.json`](../../../../bench/contracts/container-p1-v2.json)
  只把五个已有真实 provenance 的重复任务纳入 P1：4096-element short inline/spill lifecycle、32768-entry
  metadata flat lookup、4096-item bounded MPMC handoff、4096-block dense mask count、varied 64-byte table
  hash。ring/text/interner/ordered 等没有当前 profile 证明为热点的能力只做 P0 复杂度、allocation/copy/
  lock/syscall/metadata 与 correctness 门，不以“模块整体 P1”为由制造逐 overload 微基准。这个边界在
  `336793d63dd7cf08fde9c1be74c6439472335381` 中先于正式采样冻结，不是看结果后的降级。
- 预期输出：全新 capability-family map、完整 public-surface trace、owner/复杂度/资源表、有限 `FIX`
  集、依赖与消费者裁决、correctness/sanitizer/ISA、资格化 P1 bundle、三种交付、四类文档、owner
  六问、独立红队和四项 LOC。
- 停止条件 / 可接受限制：全部已准入 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，五组适用工程门有合格
  verdict，且无 correctness/security/delivery blocker 即停止。业务 cache/eviction、blocking service、
  ordered multi、node extraction/merge、compressed bitmap/rank-select、Native FFI 与 Formal 可以保持
  具名负空间；material `MUST` 或任何适用门无法在预算内裁决时进入具名 `BLOCKED`，不无限研究。

## Scope identity / inherited-evidence rule

- 当前非侵入式产品实现为 11,031 行；目录级 `loc-report --module container` 的 11,595 行还包含独立
  `container_intrusive` 的 564 行，本轮 terminal LOC 仍按统一工具报告并另注净值。
- 旧 `container-v1`、旧 architecture/user/performance 文档、当前 tests 与 benchmarks 都是待审输入；
  “曾通过”“已有测试”“已有 benchmark”不形成 verdict。
- 本轮开始时旧 candidate 到 baseline 的 Container/build 相关差异为 21 个文件、约 +896/-231 行，
  包括 Highway 生成式 kernel、`probe.h`、P1 harness、component graph 与 dependency lock。
- 当前 public API、实现、测试和文档只有在下面两遍审计完成后才会得到 `KEEP/FIX/BORROW/REJECT`
  裁决；本页不会预填旧轮的 `COMPLETE`。

## Capability-family coverage map（`AUDITED`）

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| 短/通用/固定容量 sequence | shape、batch、dependency、小型 metadata 与 loader staging；Abseil/Boost/CCCL | inline+grow、strict fixed、external allocator、POD bulk | empty/full/OOM、non-trivial/move-only、iterator/reference invalidation | `SmallVector`、`StaticVector`/`InplaceVector`、`PodVector`；普通大 sequence 走 `std::vector` | `COMPLETE` | 不复制 std 的 general vector/list/deque 全部表面；特殊 tensor storage 属上层 |
| object/byte ring | 流式 staging、parser bytes、llama.cpp sampler/repetition history | growable deque、bounded reject、exact-capacity overwrite、two-segment access | wrap、0/1/non-pow2 capacity、shrink/grow、non-trivial lifetime | `RingDeque`、`ByteRingBuffer`；新增独立 `CircularBuffer` 保持普通 deque 无额外 metadata | `PARTIAL` | `RingDeque` 会把 60 扩到 64 且 full 时只能 grow/reject，不能自然表达 llama.cpp 的 exact overwrite history |
| flat associative | token/tensor/op metadata、registry、ggml graph pointer state | map/set、heterogeneous lookup、insert/assign/emplace、iteration/filter erase | OOM、tombstone/churn、move-only value、key immutability | `FlatHashMap`/`FlatHashSet` | `PARTIAL` | key erase 完整；缺 iterator/range erase 与 `erase_if`，过滤循环不自然 |
| sorted/ordered associative | read-mostly index、range query、position/order metadata | flat sorted map/set、B-tree map/set、heterogeneous lookup | bulk build、mutation、iterator/reference lifetime、allocator | `FlatMap/FlatSet`、`BTreeMap/BTreeSet` | `COMPLETE` | multimap/multiset、node handle/merge 无当前 MUST；走 std/Boost 或未来 consumer 重开 |
| concurrent registry/append | engine registry、parallel metadata handoff/append | flat stable-within-lock view、node-stable map、segmented append | lock/progress、grow publication、quiescence、reference guard | `ConcurrentFlatHashMap`、`ConcurrentNodeHashMap`、`ConcurrentVector` | `COMPLETE` | 不承诺 lock-free associative；transactional snapshot 与 reclamation policy 属上层 |
| worker queues | scheduler typed payload handoff/work sharing | SPSC、bounded MPMC、sharded、owner/thief deque、FIFO steal | full/empty、close/wait owner、memory order、move-only | `SpscQueue`、`BoundedQueue`、`ShardedQueue`、`WorkStealingDeque`、`WsFifoQueue` | `COMPLETE` | blocking/cancel/timeout/backpressure service 由 `sched`持有；不塞入底层容器 |
| dense/sparse membership | active IDs、mask algebra、set-bit traversal | runtime dense bitset、sparse-set O(1) membership | growth/bounds、bulk ops、iteration invalidation | `DynamicBitSet`、`SparseSet`；compile-time mask 走 `std::bitset` | `COMPLETE` | compressed bitmap/rank-select 无已准入消费；不为 API parity 添加 |
| borrowed text/path | model/protocol metadata、tokenizer/parser views、path normalization | standard-shaped borrowed view、known-NUL interop、search/split/case/UTF-8 | null/empty/tail、embedded NUL、<=4 GiB explicit bound、borrow lifetime | 当前 `string_ref` + alias `StringView`、`PathView/PathBuffer`、UTF-8 helpers | `PARTIAL` | 真实类名、public fields 和 lowercase header 不符合自然 C++ 使用；empty tail 丢失已知 NUL 与 tail pointer |
| semantic hashing | integral/pointer/string/composite metadata keys；llama.cpp BPE pair key、bitset key | fast bytes、process-seeded string、hardened string、semantic custom/composite/std bridge | Eq-compatible、padding/NaN/zero、secret/seed owner | `Hash`、`BytewiseHash`、`StringHash`、`HardenedStringHash` | `PARTIAL` | pair/tuple/array 与 valid `std::hash` type无自然 route；`StringId` 不可直接做 HYLE key |
| string identity | token/op/name dedup、stable registry key | permanent、ref-counted、batch、lookup by generation id | OOM prefix、stale ID、stable view、sharded concurrency | `StringInterner` | `PARTIAL` | read-only `lookup` 非 const；`StringId` hash 缺口与上行合并 |
| inference-specific graph/text/cache algorithms | ggml side-array graph hash、llama trie/Aho、BPE heap、backend pool/LRU | stable slot/SoA、alphabet/matcher policy、decrease-key/move-pop、eviction/device lifetime | 强 domain semantics | graph state 走 map value；heap 走 std heap algorithms；trie/cache 属 tokenizer/runtime owner | `ROUTED` | 不把 C 实现的 slot ID、tokenizer algorithm 或 device-cache policy 伪装为通用容器 |

## Capability / gap ledger（`AUDITED`）

| ID | requirement class / source | normal shape | owners / lifetime / failure / concurrency | complexity / resource target | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|
| `CONTAINER-V2-RING-01` | MUST / llama.cpp sampler、Boost `circular_buffer` | 固定最近 N 个 token，push 满时丢最旧 | Container owns storage/API；allocator failure only on capacity change；non-concurrent | steady push/pop/index O(1)，容量是用户给定精确 N，无 steady allocation | `PARTIAL` | `FIX` / material | 新 `CircularBuffer<T>`；0/1/60、wrap、overwrite、non-trivial lifetime、allocation fault；不改 `RingDeque` 常态布局 |
| `CONTAINER-V2-HASH-01` | MUST / llama.cpp BPE pair + batch bitset、Abseil Hash | pair/tuple/array/custom standard-hashable metadata key | semantic hasher owns Eq compatibility；no retained borrow | field count O(1)，each field once；no allocation/dynamic dispatch | `PARTIAL` | `FIX` / material | public combine route + composite specializations + noexcept `std::hash` fallback + `StringId`；padding-only semantic types remain rejected |
| `CONTAINER-V2-FLAT-01` | MUST / normal filtering、Abseil/Boost/std associative conventions | erase current/range and predicate filtering during mutable maintenance | iterator invalidation follows open-addressed table；no concurrent mutation | erase current amortized O(1)；range/erase_if O(n)；no allocation | `PARTIAL` | `FIX` / moderate | map/set iterator/range erase + non-member `erase_if`；do not expose unstable bucket/slot ID |
| `CONTAINER-V2-TEXT-01` | MUST / C++ consumer usability、std::string_view/Abseil StringView | construct/pass/query borrowed text naturally; explicit known-NUL check | caller owns bytes；view is 16-byte trivial value；4 GiB limit explicit | trivial copy; search retains current qualified implementation; no ownership/allocation | `PARTIAL` | `FIX` / material | actual `StringView` type/header/method shape, std-view interop, known-NUL query, correct empty-tail semantics; remove public lowercase legacy spelling |
| `CONTAINER-V2-INTERN-01` | MUST / read-only registry consumer | lookup through const interner and use IDs as map keys | interner synchronizes; returned view borrows until matching release | lookup expected O(1), one shard lock, no allocation | `PARTIAL` | `FIX` / moderate | const `lookup`; `Hash<StringId>` is covered by HASH-01 |
| `CONTAINER-V2-GGML-01` | MUST / ggml graph/scheduler/allocator | pointer key -> graph state, reset between graph passes | graph owns values/lifetime; HYLE map owns table | expected O(1), `clear` O(capacity) preserving allocation | `COMPLETE` | `KEEP` / none | `FlatHashMap<tensor*, State>` + `clear/reserve`; reject public stable-slot imitation because ggml slot indexes its C side arrays |
| `CONTAINER-V2-STDTYPES-01` | GENERAL / llama.cpp+ggml source census | ordinary vector/array/map/set/deque/list/priority queue/fixed bitset | standard library owner | standard contract | `ROUTED` | `KEEP` / none | users compose std containers; HYLE adds a type only for a distinct acceleration/lifetime/failure contract |
| `CONTAINER-V2-TEXTALG-01` | GENERAL / llama.cpp trie/Aho/tokenizer | prefix matching and multi-pattern search | future text/tokenizer algorithm owner, alphabet and automaton policy | workload dependent | `ROUTED` | `DEFER` / none | record inference watchlist trigger; not a generic Container MUST in this round |
| `CONTAINER-V2-HEAP-01` | GENERAL / llama.cpp SPM/BPE、ggml backend | priority scheduling and move-pop | algorithm/consumer owns comparison/update policy | std heap O(log n) | `ROUTED` | `REJECT` / none | use `std::push_heap/pop_heap` over chosen sequence; no duplicate adapter absent decrease-key/indexed-heap demand |
| `CONTAINER-V2-CACHE-01` | GENERAL / llama.cpp backend/KV caches | device buffer reuse、LRU/eviction | runtime/backend owns budgets and device lifetime | policy dependent | `OUT_OF_SCOPE` | `DEFER` / none | only reopen Container for a reusable storage mechanism proven independent of cache policy |
| `CONTAINER-V2-DOC-01` | MUST / delivery and maintainability | one current architecture contract + choice-oriented guide + qualified evidence | docs reflect shipped headers/source/static/shared | no stale performance/general SOTA claims | `PARTIAL` | `FIX` / material | rewrite architecture/user/performance/status surfaces after candidate; add inference watchlist limits |

- 现有公开入口反向追踪：35 个 public header 按 sequence、associative、concurrent、queue、active-set/ring、hash/text/path/interner 八组全部归属；`intrusive_list.h` 只是单独 `container_intrusive` 的 include owner，不纳入本轮产品修改。
- 已有能力完整性检查：sequence/ordered/concurrent/queue/active-set 的 normal、empty/full/OOM、lifetime、invalidation 和 ownership 变体无新 material 缺口；ring/hash/text/interner 的五项缺口已全部绑定上表。
- 语法消费者 / 语义重复扫描：HYLE 直接消费者为 util/sched/io/graph；llama.cpp/ggml 另见六问。未发现与 `mem`、`platform` 重复的 allocator/ISA owner；Container 只消费它们的 public contract。
- public/internal/higher-layer 裁决：容器语义、storage 与 hash/text primitive 为 public；Highway/xxhash/wyhash/simdutf 是 private mechanism/build-only artifact；tokenizer trie、graph/KV/cache/scheduler service 保持 higher-layer。
- 本轮冻结 `FIX` 集：`RING-01`、`HASH-01`、`FLAT-01`、`TEXT-01`、`INTERN-01`、`DOC-01`。进入 `AUDITED` 后只实施这六项；新想法进 backlog/reopen trigger，不边测边扩项。
- 相邻模块 backlog：text/tokenizer 所有者评估 trie/Aho；runtime/backend 所有者评估 indexed heap 与 cache policy；`sched` 所有者承担 wait/cancel/backpressure。

## Owner 六问（审计阶段直答）

### 1. 第三方产品路径

| Dependency / locked identity | direct / indirect | role | semantic / artifact / delivery owner | public leak / upgrade trigger |
|---|---|---|---|---|
| Highway `f503c485d990a6f6adb5b220c4e8a47533110e66` | direct build-only | product mechanism | Container 持有 string/bitset 合同，Highway 持有 register lowering；build/package 持有 source projection | 无 public type；pin 或 generated-dispatch 变化触发 correctness/codegen replay |
| xxHash `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53` | direct private | product mechanism | Container 持有 public hash contract，xxHash 持有指定 byte kernel | 无 public type；pin/algorithm behavior 变化触发 hash evidence replay |
| wyhash `e4764a0aa99a62691a7f5d35cc7d627b9ad94d28` | direct private | product mechanism | 同上，用于锁定长度区间 | 无 public type；pin 变化触发 |
| simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b` (v9.1.0) | direct private compiled source | product mechanism | Container 持有 UTF-8 API，simdutf 持有 validate/count implementation；package 安装 license | 无 public type；新 stable release 或 ISA dispatcher 变化触发 |
| `core` / `mem` / `platform` | direct HYLE | product | core=compiler/cache contract，mem=allocator，platform=runtime ISA/topology | 只公开 HYLE-owned C++ types；dependency-edge 变化触发 |

- 标准库 / builtin / OS：`std::vector/string/string_view/bitset/heap algorithms`、atomics/locks 等按 C++23 实现合同；seed/topology 通过已归属的 platform/core seam，Container 不自建 OS 探测。
- 反证：CMake public edges 只有 `core mem`，private edge 为 `platform`；vendor include/source 均为 PRIVATE/build-only，package 只投影 license，不投影 vendor header/type。

### 2. 推理引擎能力完整度

- llama.cpp reference / clean locked commit / last remote check：`C:/public/project-hyle/llama.cpp` / `9723942adc518b43c4b95dc4dce6906903eb5e09` / 2026-08-31 `fetch --prune` + clean fast-forward。
- standalone ggml reference / clean locked commit / last remote check：`C:/public/project-hyle/ggml` / `36da57138425487184aa1da2eee2cde155909c6f` / 2026-08-31 `fetch --prune` + clean fast-forward。
- llama.cpp 随仓 ggml identity：`scripts/sync-ggml.last` 为 `36da57138425487184aa1da2eee2cde155909c6f`，与本轮 standalone ggml 一致。

| llama.cpp real journey / ggml foundation difference | admitted demand | HYLE route / owner | coverage | evidence / remaining trigger |
|---|---|---|---|---|
| sampler/repetition 保留 exact N recent tokens，full push overwrites oldest | yes | Container `CircularBuffer` | `MISSING` -> `RING-01` | `src/llama-sampler.cpp`、`common/sampling.cpp`；Boost circular-buffer 同语义 |
| BPE rank 的 pair<string,string> key，batch/KV 的 fixed bitset key | yes | Container semantic `Hash` + flat map；fixed mask 仍走 std::bitset | `MISSING` -> `HASH-01` | `src/llama-vocab.cpp`与 batch/KV map 查找 |
| ggml graph visited pointer set + 与 slot 对齐的 side arrays | yes，但不准入 C slot API | `FlatHashMap<tensor*, State>`/graph owner | `ROUTED` | `ggml-impl.h`、`ggml.c`、backend/allocator；map value 取代 C SoA slot identity |
| token/name registries、vectors/arrays、ordered position sets/maps、fixed masks | yes | HYLE sequence/flat/sorted/B-tree/bitset routes or std route | `COMPLETE/ROUTED` | llama.cpp/ggml source census；不按标准容器类名求并集 |
| tokenizer trie/Aho、SPM/BPE priority heap | yes，owner 不属通用 Container | text/tokenizer owner；std heap algorithms | `OUT_OF_SCOPE/ROUTED` | `common/trie.*`、`llama-vocab.cpp`；有通用 decrease-key 或 matcher 合同时重开 |
| backend buffer pool、KV/LRU/eviction | yes，为 domain policy | backend/runtime | `OUT_OF_SCOPE` | 预算、device lifetime、eviction 不由通用容器定义 |

- 当前证据下是否存在未处置 `MUST`：`NO`；全部 material 缺口已进入冻结 FIX，但 candidate 实现前 coverage 仍是 `MISSING/PARTIAL`。
- 结论边界：只回答本轮锁定引擎的实际数据结构形状，不承诺 ggml/llama.cpp 今后每次新增的 domain algorithm 自动属于 Container。

### 3–6. 当前阶段裁决

- 性能：`P1`；旧 bundle 不可继承，当前为 `PLANNED`。Candidate 后只跑 closure card 的五组拟真 journey，5% non-inferiority；不声称“Container 全面 SOTA”。
- C ABI / Native FFI：本轮无已准入 slice；Container 是 C++ 模板/所有权库，无 public C header/schema/bridge/sample/option/export；private C/vendor/ISA seam 不是客户 ABI。
- C++ API / 三种交付：修复集专门收束 Ring/Hash/Flat/Text 的不自然形状；source/static/shared 将在 Candidate 后分别验证 selected-component consumer，不承诺稳定跨版 C++ ABI。
- 文档：architecture 只保留长期 owner/边界，user guide 按“什么情况选什么”，implementation task 保留本轮事实，performance 只发布资格化有限 claim；当前旧文档为 `DOC-01` 待修。

## Candidate

当前产品候选固定为：

- commit：`336793d63dd7cf08fde9c1be74c6439472335381`
- tree：`968d169bafe0eab270d397ab1614f80c1a98079f`

它只实施审计阶段冻结的六项 FIX：

1. 新增 `CircularBuffer<T>`，用独立 exact logical capacity 表达“最近 N 项”；满时端点覆盖、steady
   push/pop/index 为 O(1) 且零分配，调整容量为 O(n)，缩小保留最新项。`RingDeque` 的 grow/bounded
   合同与常态布局未被 benchmark 改写。
2. `Hash` 新增 equality-compatible 的 `noexcept std::hash` bridge、pair/tuple/array 逐字段组合、公开
   `hash_combine` 和 `StringId` 语义 hash；不推断 padding 或任意 object representation。
3. `FlatHashMap`/`FlatHashSet` 新增 iterator/range erase 与 `erase_if`；单项 erase 不重新 hash，
   amortized O(1)，predicate/range 遍历 O(n)，无 allocation 或新 per-entry metadata。
4. 删除 lowercase `string_ref` 公开类型/header，交付真实 `StringView`：私有 16-byte trivial layout、
   method-shaped API、`std::string_view` interop、显式 known-NUL 与正确 empty-tail pointer/terminator 语义。
   util/io 等直接消费者已经整体迁移，不保留伪兼容 alias。
5. `StringInterner::lookup` 成为 const query，`StringId` 可自然作为 HYLE flat key；锁、generation 和
   lifetime 算法不变。
6. architecture/user/build sample/bench contract 和直接消费者同步到上述形状；未新增 C schema、Native
   FFI 或稳定 C++ ABI 声明。

候选前置验证不作为最终 manifest：受控 MSVC Release 下 Container 336/336 单元测试与 public-header
self-containment 通过；util、HTTP、HTTP/2 直接消费者编译/测试通过；container-only source、relocated
shared 与 relocated static selected-component smoke 通过。一次 dirty/single-sample probe 只用于确认
五个既有代表机制能运行并排除明显回退，明确不形成 P1 verdict。最终 correctness、sanitizer/ISA、
delivery、qualified P1 和独立红队只能绑定上述冻结候选及后续纯证据快照。

## Verification manifest

产品 candidate 始终是 `336793d63dd7cf08fde9c1be74c6439472335381` /
`968d169bafe0eab270d397ab1614f80c1a98079f`。资格化性能从 clean
`0ce6168c94ccdf008dfce7337354f437fe82f67b` 构建；correctness/ISA/交付的最终测试快照为
`1199a9edbffef46a1c76bdee2f2e722f76371532`。从 candidate 到测试快照，Container 产品、P1 contract、
bench adapter 和 package sample 的路径 diff 为空；仅 closure 状态/任务与测试证据改变，因此没有
candidate drift。

### Correctness risk matrix

| Risk | Verdict | 命令 / 证据 | 限制 |
|---|---|---|---|
| normal | `PASS` | W1-WIN MSVC、W1-WSL GCC 14.2、M1-MAC AppleClang 21 的 Release `hyle_container_test` 均通过 336 cases；public-header self-containment 三平台通过 | `container_intrusive` 是独立组件，不并入本轮 test count |
| empty / zero | `PASS` | sequence/map/set/queue/bitset/ring/text/interner 的 empty/zero tests；`CircularBuffer` 覆盖 capacity 0/1 | empty `StringView` 保留原 tail pointer 与 known-NUL 语义 |
| capacity / boundary / overflow | `PASS` | `circular_buffer_test.cc` 覆盖 0/1/60、wrap、full overwrite 与 shrink/grow；vector/ring/hash/text 覆盖边界和 32-bit view 上限 | `StringView` 明确拒绝超过 `uint32_t` 的长度 |
| OOM / system failure / invalid input | `PASS` | failing allocator、reserve/grow failure、invalid UTF-8/path/hash/interner stale-ID tests | 容器层没有 OS I/O；可恢复 allocation failure 保持原 owner 有效 |
| move / destroy / lifetime | `PASS` | non-trivial/move-only/self-alias sequence、ring、queue、map 与 interner generation tests；ASan/UBSan | borrowed views 不延长 backing lifetime |
| concurrency / reentrancy | `PASS` | concurrent map/vector、SPSC/MPMC/sharded/work-steal stress；W1-WSL TSan 全 Container suite | 非并发容器仍是 caller-synchronized |
| cancellation / close | `NOT_APPLICABLE` | Container 只提供 nonblocking primitive；wait/cancel/timeout/backpressure 归 `sched` | 未把缺少 scheduler service 写成容器 PASS |
| property / differential / fuzz / stress | `PASS` | randomized wrap/churn/hash/equality/bitset tests；scalar↔ISA differential；并发 stress | 本轮没有独立 coverage-guided fuzz campaign，不把它伪称已运行 |
| ASan / UBSan / TSan | `PASS` | W1-WSL fresh GCC 14 builds：ASan、UBSan、TSan 各完整 336 cases；TSan 用 `setarch x86_64 -R` | mimalloc aligned-allocation 与 GTest fork warning 无 sanitizer report |
| real ISA / backend | `PASS` | W1-WIN static test 直接执行 scalar/SSE4/AVX2；M1-MAC static test 直接执行 scalar/NEON，逐 boundary 比较 string/bitset kernel | 主机不支持的 AVX-512、SVE、RVV 未写成 PASS；Container 没有 device backend |

W1-WIN 最终显式运行 `hyle_container_header_self_contained_test`、`hyle_container_test` 与
`hyle_container_hash_seed_process_test` 3/3，以及 static `hyle_container_isa_variants_test` 1/1。
一次 `ctest -L container` 同时选中独立 `container_intrusive` 的两个未构建 executable 而非零；它只说明
label 横跨两个组件，不能作为本轮结果，故按 closure card 的精确目标重跑并记录。W1-WSL 的 util、sched、
IO HTTP/HTTP2/TLS、graph/parallel/data-parallel 八个直接消费者 8/8 通过；TLS test 的未使用参数只做了
test-only 可移植性修正，没有改变 IO 或 Container 产品。

### Aggregate gates

| Gate | Verdict | 证据 | 限制 |
|---|---|---|---|
| scope | `PASS` | capability-family map、全部 public header trace、冻结六项 FIX；candidate 后无产品扩项 | 结论只绑定当前 llama.cpp/ggml 与已准入 common journeys |
| architecture | `PASS` | `PUBLIC core mem`、`PRIVATE platform`；Highway build-only seam 和 xxhash/wyhash/simdutf private mechanism 均由现行组件/package 合同投影 | tokenizer/cache/scheduler policy 不下沉 |
| correctness | `PASS` | 上述三平台、sanitizer、direct ISA 与 risk matrix | 未运行 ISA 明确保留限制 |
| delivery | `PASS` | W1-WIN、W1-WSL、M1-MAC 的 selected `container` source、relocated shared、relocated static consumer 全部通过 | C++23，同 headers/artifact/toolchain 合同；不承诺稳定跨工具链 C++ ABI |
| usability/docs | `PASS` | architecture、choice-oriented user guide、package sample、public-header test、旧 API 扫描 | 不提供兼容 alias/header |
| red team | `PASS` | 下节有界攻击均 fixed/disproved/accepted，无 blocker | 只攻击冻结合同，不扩大 peer/API 并集 |
| engineering performance | `PASS` | W1-WIN `container-p1-v2` QUALIFIED；五项全部通过 5% NI；详见 performance evidence | 只覆盖五个具名 normal tasks，Formal 为 `NONE` |

P1 bundle：合同 SHA-256
`50323f331be34fa1528dceb2d2ffbf07058a3f5593e121580d1f6ed65e532941`，manifest SHA-256
`44a4d2e9aaa3726e74282df16987c5f407ecfffa7fbf7ded5ed4c04625f2ca94`，deep verifier 返回
`QUALIFIED`、`paths_verified=true`、provenance `EXACT`、current policy eligible。五个 task 的八个 strata
均通过 5% NI；sequence、flat lookup、MPMC、dense count 为 exact `WIN`，64-byte hash 为 exact
`INCONCLUSIVE`，没有把 margin 内约 3% 点值写成胜出。完整边界见
[`benchmarks-container.md`](../../evidence/performance/benchmarks-container.md)。

## 终止阶段 owner 六问

1. **第三方。**产品路径直接使用 Highway
   `f503c485d990a6f6adb5b220c4e8a47533110e66`、xxHash
   `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53`、wyhash
   `e4764a0aa99a62691a7f5d35cc7d627b9ad94d28`、simdutf
   `2d4e249dcfd978b39dd567b156f20a43db97377b`，并经 `mem` 间接使用其已审机制。Container 是
   semantic owner；第三方只拥有 private/build-only mechanism，public API 不泄漏 vendor 类型；
   build/package 拥有 pin、cache/offline、license 和三种交付。升级算法、dispatcher 或 codegen 时定向
   重开受影响能力族。
2. **功能。**2026-08-31 终止前再次对 llama.cpp 与 standalone ggml 执行 `fetch --prune`；两者 clean
   `origin/master` 仍分别是 `9723942adc518b43c4b95dc4dce6906903eb5e09` 与
   `36da57138425487184aa1da2eee2cde155909c6f`，llama 随仓 `scripts/sync-ggml.last` 也是后者。exact recent
   history、semantic composite key、flat filtering、borrowed text 与 const interner 已是
   `COMPLETE`；graph side-array、标准 heap/container、tokenizer/cache policy 有完整 `ROUTED` 或
   `OUT_OF_SCOPE` owner。当前没有未处置 `MUST`，但不承诺未来模型永不提出新需求。
3. **性能。**模块级 P1 的五个真实重复任务在 source-primary、同语义/ownership/resource 计费、锁定
   peer、W1-WIN 与 5% margin 下 `NON_INFERIOR`，四项还为 exact `WIN`；其余能力只有 P0 structural
   `PASS`。禁止外推“Container 全面 SOTA”、跨平台固定点值或未运行 operation/ISA。
4. **旧 C ABI / Native FFI。**策略为 `N/A`；Container 产品树没有 public C header/schema、bridge、
   undecorated export、C/managed sample、option/target/package metadata 或为 FFI 支付的 layout。第三方 C
   implementation seam 是 private，不是客户 ABI。verdict：`CLEAN`。
5. **构建与导入。**Windows、WSL、macOS 的 source embedding、installed relocated static、installed
   relocated shared journey 都为 `PASS`；selected headers 包含 `circular_buffer.h`、`string_view.h`，不再
   安装 `string_ref.h`。中央 closure、static private link、shared loader/RPATH 与 exact package discovery
   都由真实 consumer 证明，不以 producer build 替代。
6. **C++ API 与文档。**用户按任务选择 sequence、flat/sorted/B-tree、concurrent registry、queue、
   active set/ring、hash/text/interner；`StringView` 是自然 method-shaped borrowed value，
   `CircularBuffer` 表达 exact overwrite，flat filtering 与 composite hash 不再需要 internal seam。
   ownership、failure、concurrency、invalidations、OOM 和选择陷阱见 user guide；architecture 只保留长期
   边界，implementation/performance 只保留事实。无旧 `string_ref`/`_sr` 活动引用，也无兼容层。

## 独立红队

- 审查者 / 日期 / 预算：当前 closure reviewer / 2026-08-31 / 30 分钟；只攻击冻结 candidate、依赖、
  correctness、交付和 P1 validity。

| 问题 | Severity | 裁决 | 证据 |
|---|---|---|---|
| 旧 lowercase text API 可能通过 alias/header/consumer 残留 | material | `fixed` | 删除类型/头/源/测试/bench，活动树 `string_ref`、旧头、`_sr` 扫描为空；三种 SDK 不安装旧头 |
| 新 `StringView` 为 generic range 隐藏 throwing/oversize 或 empty-tail 语义 | material | `fixed` | generic route 只接受 noexcept `data/size`，超过 u32 fail-fast；empty-tail/NUL/bounds tests 通过 |
| exact-capacity ring 在 full/self-alias/shrink/OOM 时破坏最新项或 non-trivial lifetime | material | `fixed` | 0/1/60、wrap、两端 overwrite、self-alias、move/destroy、failing allocator 与 sanitizer 通过 |
| composite hash 读取 padding、与 Eq 不兼容或无界桥接用户 `std::hash` | material | `fixed` | 逐字段 combine；只桥接 valid noexcept `std::hash`；pair/tuple/array/StringId differential tests |
| iterator erase 跳项、range 返回错误或引入 per-entry metadata | material | `fixed` | map/set erase/erase_if churn tests；算法仍 amortized O(1)/O(n)，layout 无新增字段 |
| exact-target 配置被误当成 Container 多 ISA execution proof | material | `fixed` | 新 test-only static gate 直接调用本机 supported internal kernels；Windows scalar/SSE4/AVX2、Mac scalar/NEON 真差分 |
| vendor 类型/target 或私有 link 从 shared/static package 泄漏 | material | `disproved` | selected component source/static/shared relocated consumers + installed header/license census；public target 只暴露 HYLE owner |
| 旧 v1 五行或当前五项被外推为全模块 SOTA | material | `fixed` | performance 文档重写；P1 applicability 与 P0 negative space 明写，hash exact 保持 `INCONCLUSIVE` |
| llama.cpp/ggml reference 在本轮期间漂移 | material | `disproved` | terminal `fetch --prune` 后 clean HEAD 与 origin/master 未变，embedded ggml identity 一致 |

没有 finding 需要修改冻结产品；red-team 产生的 ISA test 与 TLS test portability 修正只改变测试，已单独
提交为 `1199a9edbffef46a1c76bdee2f2e722f76371532`。

## Closure decision（等待机械终止提交）

- 所有当前 `MUST` 已为 `COMPLETE` 或完整 `ROUTED`，不存在 `PARTIAL/MISSING + KEEP`。
- scope、architecture、correctness、delivery、usability/docs、red-team、engineering performance 全部
  `PASS`；没有 correctness/security/ownership/delivery/P1 blocker。
- 已知限制：P1 只签 W1-WIN 五项 normal task；Formal 为 `NONE`；未在真硬件执行 AVX-512/SVE/RVV；
  `StringView` 是 caller-owned <=4 GiB borrow；普通容器不自动线程安全；blocking/cancel/cache/tokenizer/
  model/device policy 不属于本模块；C++ shared SDK 不承诺跨工具链稳定 ABI，也无 Native FFI。
- reopen triggers：真实消费者暴露缺失 common journey、correctness/security/OOM/lifetime/concurrency/交付
  缺陷；公共 API/layout/算法/组件边变化；Highway/xxhash/wyhash/simdutf 或直接 HYLE 依赖改变实际产品
  semantics/codegen；锁定正常 workload 出现 material regression；新 ISA/platform 或更新 llama.cpp/ggml
  提供可准入、当前 route 无法完整承载的通用能力。
- raw evidence：Windows
  `C:\hyle-scratch\container-v2-perf-qualified-336793d6-evidence`；WSL
  `/root/hyle-scratch/container-v2-1199a9ed-delivery`；Mac
  `/Users/kiro/hyle-scratch/container-v2-1199a9ed-delivery`。后两者是可丢弃验证副本，不是产品来源。
- 四项 LOC snapshot commit：`1199a9edbffef46a1c76bdee2f2e722f76371532`；全库代码 404,562；自实现
  包含测试 96,940；自实现不含测试 60,339；Container 目录实现 11,871，其中独立已关闭的
  `container_intrusive` 为 564，本轮非侵入式实现净值 11,307。
- 建议机械状态：`VERIFIED`；保存 terminal receipt 后再以合法相邻 transition 进入 `CLOSED`。
