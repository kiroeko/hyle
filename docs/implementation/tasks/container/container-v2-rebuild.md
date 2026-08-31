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
- 有限 performance gates：
  1. `metadata-sequence`：短 append/iterate、spill/grow 与 strict/non-trivial fixed sequence；
  2. `metadata-index`：integral/string flat hit/miss/build/erase，read-mostly sorted lookup/bulk 与 ordered
     range mutation；
  3. `registry-and-handoff`：read-heavy/mixed concurrent map、parallel append，以及 SPSC/MPMC/sharded/
     work-steal 的正常角色 journey；
  4. `active-set-and-stream`：bitset count/algebra/set-bit iteration、SparseSet lifecycle、object/byte ring
     batch staging；
  5. `hash-text-intern`：short/64-byte/long hash、UTF-8 validate/count、borrowed search/case 与 interner
     acquire/find/release。每组只以代表 workload 裁决共享算法/成本机制，不为每个 overload 建一项跑分。
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

## Candidate、Verification、红队与终止裁决

这些章节在对应状态到达时填写。未运行项保持未裁决，不从 `container-v1` 复制 `PASS`。
