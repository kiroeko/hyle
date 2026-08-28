# container-v1 重构与收口记录

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->

> 本页是完整 `container` 组件的当前 closure card。既有
> [`container-module-closure.md`](container-module-closure.md)、architecture、guide 与 benchmark
> 只作审计输入；其中旧 C ABI、旧点值和“最终证书”不能继承为本轮结论。独立的
> `container_intrusive` 已有自己的产品组件和冻结记录，不属于本轮产品面。机械状态只看
> [`module-status.json`](../../module-status.json)。在 capability map、public-surface trace 与有限
> `FIX` 集进入 `AUDITED` 前，不修改 container 产品实现。

## Closure card（`BLOCKED` from `CANDIDATE`）

- component / owner / round ID：`container` / data-structure owner / `container-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`2866c7df73c6bd36cc4808226d370ff0ea338da2` /
  `5e31a8cabed513c1f5d6c4c7328ef00ed4f67612`
- 进入原因与 demand/defect ID：
  - `CONTAINER-CAP-01`：按当前能力完整性标准，从真实任务而不是已有类名或 benchmark 重新枚举完整
    container 能力族，并反查全部公开入口是否只有半项生命周期或错误 owner；
  - `CONTAINER-NORTHSTAR-01`：本地量化推理需要 token/metadata 索引、小批量连续 storage、并发 registry、
    stage handoff、工作窃取、活跃集合与 byte/text staging；工程计算留出使用 ordered/range index、
    sparse active IDs 与同样的 queue/ring 机制；
  - `CONTAINER-CONSUMER-01`：当前 `util` registry、`sched` queues、`io` text/hash 与 `graph` small sequence
    已直接消费 container，需要核对语义、依赖和失败路线；
  - `CONTAINER-API-01`：旧 C 产品已经删除，但现有 `carray`、`string_ref`、buffer-fill 与若干 pointer/out
    形状仍需证明是自然 C++ primitive 或重塑，而不是保留旧 ABI 外壳；
  - `CONTAINER-PERF-01`：2026-07 的性能总账早于当前 `closure-perf-v1`，混有不同合同、平台点值与
    benchmark-oriented 叙事，不能证明当前 source-primary 产品候选；
  - `CONTAINER-DEP-01`：当前组件依赖 `core/mem/platform`、build-private SIMD header seam 以及 vendored
    xxhash/wyhash/simdutf；升级后需重新核对算法、ISA fallback、license 与三种交付；
  - `CONTAINER-DOC-01`：architecture 仍标 `NOT_REVIEWED`，旧 task/user/performance 文档仍包含已删除
    Native C surface 和过期证书，需要按四类文档职责重新收束。
- 真实 consumer task / normal workload：
  - 推理引擎用 flat hash 保存 token/张量/算子 metadata，用 small/fixed sequence 组装短 shape、batch 与
    dependency list，用 stable-node/concurrent map 保存跨 grow 仍需 identity 的 registry；
  - worker pipeline 使用有界 SPSC/MPMC/sharded/work-stealing 机制传递 typed payload，控制面使用 bitset/
    sparse active set，模型与 I/O loader 使用 byte ring、borrowed text、hash/UTF-8/path primitive；
  - 当前 HYLE `util/sched/io/graph` 的直接调用序列作为仓库内 consumer replay；未关闭下游只验证它们
    实际依赖的 container 合同，不借此宣称下游模块已经收口；
  - 工程计算留出以 read-mostly sorted index、持续 mutation 的 ordered/range index、active-ID set 与
    streaming ring 验证 API 没有被模型、token 或单一 benchmark 过拟合。
- 范围：`include/hyle/container/**` 中除 `intrusive_list.h` 外的完整公开 C++ 面、`src/container/**`、
  container-owned external adapters、全部 public-surface trace、实际依赖和直接消费者；source/static/shared
  交付；architecture/user/performance 文档；有限的 correctness、P1 consumer replay 与独立红队。
- 非目标：已关闭的 `container_intrusive`；模型格式、tensor/KV layout、batching/serving policy；scheduler
  wait/cancel/backpressure service；通用 cache/LRU、阻塞或持久化队列；device memory；复制 std/Boost 全部
  容器或单库独有 feature；Native FFI/C API、稳定 C++ ABI；下游模块整体收口与 Formal 发布证书。
- 直接依赖 / 直接消费者：产品边为 `PUBLIC core mem`、`PRIVATE platform`，实现 header seam 为
  `PRIVATE_BUILD simd_vec_headers`；外部源码 owner 为 xxhash、wyhash、simdutf。中央图的直接消费者为
  `util`、`sched`、`io`、`graph`；当前具体语法消费者还包括 package/sample。
- tier 与理由：`P1`。lookup、append、queue、bitset、ring、hash/text 等可在推理 worker 热路径重复发生；
  reserve/rehash/grow、bulk build、clear、path normalization 和 teardown 等控制/冷路径先按 P0 结构门审查，
  只有能传播到正常 journey 的操作进入计时。
- material margin / resource budget：W1-WIN 同合同 peer 或 immutable baseline 的工程 NI margin 为 5%。每族先
  达到成熟复杂度档；不得为点值增加无消费者承担的常态 metadata、padding、allocation、lock、dynamic
  dispatch 或重复 hash/copy。队列容量、引用稳定性、allocator/OOM 与 synchronization 语义必须等价后才
  比较；吞吐不能抵偿丢失 lifetime、错误 progress 或额外驻留内存。
- 总调查预算 / 性能预算：从 `SCOPED` 起能力/实现/依赖调查最多 6 小时，其中统一性能取证最多 2.5 小时、
  最终独立红队最多 30 分钟。预计产品修复超过一个工作日时，先把冻结 gap 拆为可独立验证的有限批次；
  到预算只能按协议裁决、具名 `BLOCKED` 或显式重新 scope，不能继续发明实验。
- 最多复现、机制诊断、产品修改次数：每个 material 异常最多一次反序/复现、一次 codegen/profile/
  allocation/contention 机制检查、一个同时满足机制与 consumer 两把钥匙的产品修改及一次重跑；不在本轮
  改中央统计规则或扩建公共仪器。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用 correctness、
  sanitizer、ISA fallback 和交付复核，不跨不同机器池化点值。Mac 大小核只使用登记的 QoS 合同。
- instrument profile：P1 为 `closure-perf-v1`；correctness/delivery 为 `closure-correctness-v1` 与当前
  component/package contracts。
- strong peers（当前本地 clean source，按能力族最多两个）：
  - general/inline sequence：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 与 Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83`；strict fixed-capacity 另以 CCCL
    `132670eba92ca5bbddb2d479e76045184ed885a5` 代替 Abseil；
  - flat/sorted/ordered associative：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 与 Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83`；
  - concurrent associative/vector：oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 与 Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`；
  - bounded FIFO/ring/work sharing：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`，并按具体
    MPMC 合同使用 rigtorp `b9808ede08f26fa9df4df4e081d19cace8f6c6ea` 或 moodycamel
    `c227e283a9edd9d15393733223235527af6101e9` 二选一，不把不同 progress/capacity 合同池化；
  - bitset/sparse-set family：Boost `df8eeb90846648464e3c8e378e312f0cec772e83` 与 Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 只用于能力/复杂度审计；
  - byte hash 与 UTF-8：xxHash `e573d4d2aaeaba0f3e5a0a9a54144a1f2b4b56e7` 与 simdutf
    `282107c23196ddc6c7b8ef06567f81bbd056fc3c` 各自只比较对应算法合同。
- 无同合同 peer 时的 consumer budget 及禁止声明：StringInterner、path/view、SparseSet、byte staging 等
  对 frozen baseline 做 5% source-primary NI，并守住零额外常态 allocation/hash scan/lock/syscall 与既定
  渐进复杂度；若没有合格可比 peer，只能声明该冻结 workload 无 material regression，不能写成
  “container 全面 SOTA”、跨平台固定纳秒或由另一种语义的 peer 推导胜负。
- correctness gates：normal；empty/zero/N=0；capacity/wrap/overflow；allocator OOM/fault；invalid input、
  hash/equality/comparator contracts；copy/move/destroy/self-alias/iterator/reference invalidation；non-trivial/
  move-only lifetime；producer/consumer/accessor/rehash/quiescence progress；property/differential/random stress；
  ASan/UBSan/TSan；x86 scalar/SSE/AVX 与 arm64 scalar/NEON runtime dispatch；public-header self-containment。
- delivery gates：中央依赖/include-owner 对账；xxhash/wyhash/simdutf pin、license、offline 与 package
  projection；container-only source embedding、relocated static/shared C++23 consumer；传递安装的独立
  `container_intrusive` target/header 不得被文档描述为 `hyle::container` ownership；无旧 C
  schema/bridge/fixture/Native FFI symbol。
- performance gates：冻结五组正常 family，不把不同语义池化：
  1. `sequence-and-inline-batch`：短 append/iterate、spill/grow、strict bounded batch 与非平凡 lifetime；
  2. `associative-index`：integral/string flat hit/miss/build/erase、read-mostly sorted lookup/bulk、ordered range
     mutation/expiry；
  3. `concurrent-handoff-and-registry`：SPSC/MPMC/sharded/work-steal、parallel append、flat/stable-node map 的
     read/mixed/rebuild，并同时报告 progress、tail 与驻留资源；
  4. `active-set-and-stream`：bitset algebra/count/iteration、sparse active-ID lifecycle、object/byte ring staging；
  5. `hash-text-intern`：短/中/长 byte hash、UTF-8 validate/count、borrowed search/case、interner
     acquire/find/release 的单线程与正常并发 journey。
- 预期输出：capability-family map、全部 public-surface trace、有限 `FIX` 集、owner/复杂度/资源表、直接
  consumer 与依赖裁决、correctness/sanitizer/ISA 证据、统一 P1 bundle、三种交付、四类文档校准、独立
  红队与四项 LOC。
- 停止条件 / 可接受限制：全部 `MUST` 为 `COMPLETE` 或完整 `ROUTED`、五组适用工程门 `PASS` 且无
  correctness/security/delivery blocker 即停止。业务 cache/eviction、blocking service、ordered multi、
  node extraction/merge、Native FFI 与 Formal 可在证据不足时保持具名负空间；任何 material `MUST`、
  correctness、交付或统一性能门不能在预算内裁决时进入具名 `BLOCKED`，不无限研究。

## Capability-family coverage map（`CANDIDATE`）

本表先从推理与工程计算任务、标准问题域和 pinned peers 自顶向下枚举能力，再反向追踪现有公开
入口。`container_intrusive` 是独立组件；这里不会为凑“完整 container”重新聚合它。

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| inline / fixed / POD sequence | shape、短 dependency/batch、trivial staging；Abseil/Boost/CCCL | 可增长 SBO、严格不分配、trivial relocation、普通长序列路由 | copy/move/range/insert/erase/reserve/OOM/teardown；连续 storage 与失效规则 | `SmallVector`、`StaticVector`/`InplaceVector`、`PodVector`；普通长寿命序列用 `std::vector` + mem PMR | `COMPLETE` | `N==0` empty move 编译期不形成 byte copy；`PodVector` allocation 只含 elements |
| flat hash index | token/metadata/registry hit/miss/build/erase；Abseil Swiss table | map/set、透明 lookup、fast/hardened string policy、可恢复 OOM | owning values；grow/erase 失效；expected O(1)；tombstone-aware reserve | `FlatHashMap`/`FlatHashSet`，实现核为 `SwissTable` | `COMPLETE` | `try_reserve` 可恢复、`reserve` fail-fast；稳定引用路由 node map |
| sorted / ordered unique index | read-mostly sorted metadata、range expiry；Boost flat/tree | flat SoA unique 与 mutation-friendly B-tree unique | bulk sorted build、range lookup/erase、OOM、iterator/reference invalidation | `FlatMap/FlatSet`、`BTreeMap/BTreeSet` | `COMPLETE` | flat 空对象不嵌 K/V；multi/node-handle 未准入 |
| concurrent associative index | cache/counter 与稳定 object registry；oneTBB/Folly | flat accessor 与 node-stable accessor | shard/bucket lock、grow/clear/quiescence、OOM、accessor lifetime | `ConcurrentFlatHashMap`、`ConcurrentNodeHashMap` | `COMPLETE` | pending writer 阻止新 reader；两类 reserve 失败形状一致 |
| concurrent append sequence | 多 writer 构建 metadata，join 后遍历；oneTBB `concurrent_vector` | stable segmented address、fallible/fail-fast append、quiescent reuse | reservation count 不是 completion fence；OOM 不留洞；destroy/clear 只在 quiescence | `ConcurrentVector` | `COMPLETE` | 三项 embedded table、按需扩展并支持 quiescent `clear()` |
| active membership | dirty mask、frontier、active ID；Boost/Folly | dense bits 与 sparse bounded universe | resize/OOM、algebra/count/set-bit iteration、O(1) sparse insert/erase/clear | `DynamicBitSet`、`SparseSet`、`FlatHashSet` | `COMPLETE` | value equality 按逻辑 size/blocks；rank/select/compressed bitmap 未准入 |
| single-thread ring / byte staging | parser/I/O staging、两端 deque；Boost/std problem domain | growable/bounded object ring、zero-copy byte spans | wrap、partial read/write、reserve、non-trivial lifetime、clear | `RingDeque`、`ByteRingBuffer` | `COMPLETE` | persistent/multithreaded byte pipe 归 io/sched，不给 ring 塞 wait/close policy |
| concurrent handoff / work sharing | scheduler stage handoff、owner/thief；Folly/rigtorp/crossbeam family | SPSC、strict MPMC、sharded non-global FIFO、LIFO/FIFO stealing | bounded allocation、full/empty/retry、role/progress、quiescent destroy | `SpscQueue`、`BoundedQueue`、`ShardedQueue`、`WorkStealingDeque`、`WsFifoQueue` | `COMPLETE` | MPMC/sharded 支持 placement lifetime 与 move-only payload；stealing payload 限制保持具名 |
| borrowed text / path | HTTP、registry key、loader/path preprocessing；standard string-view problem domain | borrowed view、search/case/UTF-8、lexical path、caller-owned output | borrow invalidation、embedded NUL、NUL requirement、OOM、batch spans | `StringView`（16-byte `string_ref` implementation）、`string_algo`、`path` | `COMPLETE` | borrowed view 不再伪装 owned allocation；producing path 写 caller span |
| hash / interning | metadata hash、防碰撞策略、稳定 deduplicated strings；xxHash/simdutf | primitive/byte/string hash，fast/secret policy，refcounted/intern-only ids | seed/key ownership、collision equality、acquire/find/release/lookup lifetime、batch | `Hash`/`StringHash`/`HardenedStringHash`、`StringInterner` | `COMPLETE` | interner batch 使用 span；不扩成 Unicode/持久化服务 |
| intrusive membership | timer/task 已有稳定 owner 的零分配挂链 | auto/safe retirement、multi-hook、splice | non-owning、外部同步、node/list 双向 lifetime | 独立 `container_intrusive` target | `ROUTED` | 已 CLOSED；完整 container 不重新拥有该 header 或状态 |
| cache/eviction、blocking/unbounded service、ordered multi | 模型 cache、异步 channel 等上层需求 | policy、quota、wait/cancel/close 与 duplicate-key 语义 | owner 必须是具体 runtime/service，而非裸机制 | std/Boost 或未来 sched/engine owner | `OUT_OF_SCOPE` | 没有具名 policy consumer 前不复制 LRU、priority queue service、multimap/multiset 全面 API |

### Public-surface trace

| public surface | family / audit decision |
|---|---|
| `small_vector.h`、`static_vector.h`、`inplace_vector.h` | inline/fixed sequence `KEEP`；`N==0` empty move 编译期排除 null-pointer memcpy，N>0 用户合同与代码路径不变 |
| `pod_vector.h` | typed trivial sequence `KEEP`；旧 `carray.h`、`CArrayHeader/carray_*` 与 allocation prefix 已删除 |
| `flat_hash_map.h`、`flat_hash_set.h`、`flat_map.h`、`flat_set.h`、`btree_map.h`、`btree_set.h` | 三种 associative cost model `KEEP`；自然入口转发同一 family core，reserve failure 与 flat normal metadata 已统一 |
| `concurrent_flat_hash_map.h`、`concurrent_hash_map.h`、`concurrent_node_hash_map.h` | accessor ownership与两种引用稳定性 `KEEP`；writer preference 与 reserve naming 已修复 |
| `concurrent_vector.h` | stable segmented append `KEEP`；compact segment table 与 quiescent reuse 已实现 |
| `dynamic_bitset.h`、`sparse_set.h` | dense/sparse membership `KEEP`；bitset value equality 已实现 |
| `ring_deque.h`、`byte_ring_buffer.h` | object/byte ring `KEEP` |
| `spsc_queue.h`、`bounded_queue.h`、`sharded_queue.h`、`work_stealing_deque.h`、`ws_fifo_queue.h` | role-specific queues `KEEP`；MPMC non-trivial lifetime 已补，不把 stealing queue 扩成宽原子 payload |
| `string_view.h`、`string_ref.h`、`string_algo.h`、`path.h` | borrowed text/search/path mechanism `KEEP`；owner/view 与 batch/buffer 形状已分离；16-byte view 与 <=4 GiB text contract 保留，超大 binary 用 byte span/mapping |
| `hash.h`、`string_interner.h` | hash/security/interner algorithm `KEEP`；batch facade 使用 span |
| `probe.h`、`detail/adaptive_*` | container implementation mechanism；不作为独立用户能力；AArch64 relax 与 pending race 已修复 |
| `intrusive_list.h` | 本轮 `OUT_OF_SCOPE`；安装/组件 owner 只属于 `container_intrusive` |

语法与语义消费扫描结论：`util::Registry` 真实消费 `StringView`、`FlatHashMap` 与可恢复 reserve；`sched`
真实消费 pointer/POD 的 MPMC、sharded 与 work-sharing 队列并在构造后检查 `valid()`；`io` 真实消费
borrowed HTTP/TLS views 与 flat hash，且在初始化路径传播 reserve OOM；`graph` 真实消费
`SmallVector`。这些调用都属于正确 owner；下游局部 `std::vector/std::string` 是普通 owning value，不能为
“统一使用 HYLE”强制替换。container 经 public `core/mem`、private runtime `platform` 与 build-private
SIMD header seam 的依赖方向正确，xxhash/wyhash/simdutf 仍由 container artifact 私有拥有。

## Capability / gap ledger（`CANDIDATE`）

| ID | requirement / exact evidence | normal owner、failure / concurrency | complexity / normal resource | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|
| `CONTAINER-SEQ-01` | MUST；shape/batch consumer + Abseil/Boost/CCCL | owning sequence；显式 allocator；fallible `try_*` / ordinary fail-fast | append amortized O(1)，fixed hot path O(1)/零 heap | `COMPLETE` | `KEEP` | non-trivial long sequence路由 std::vector；N>0 codegen/生命周期回归 |
| `CONTAINER-CORR-01` | MUST；rejected candidate 的 GCC 14 fortified/WERROR 可复现诊断 | `SmallVector<T,0>` empty inline owner；move/swap 不向 byte primitive 传 null pointer | O(1)、零分配；`N==0` 编译期无 memcpy，N>0 codegen 不变 | `COMPLETE` | `KEEP` | 原失败 GCC 14 WERROR 与 Windows 完整 container 已通过；三平台/sanitizer 随 replacement candidate 验证 |
| `CONTAINER-POD-01` | MUST；旧 C product 已删除，但 `CArrayHeader` 重复 allocator/size/capacity 并同步写 header | `PodVector` 唯一 owner；直接 block reallocate/free；size type 使用 `size_t` | 同样 amortized O(1)；去掉每 allocation header/padding 与 size sync | `COMPLETE` | `KEEP` | OOM、over-align、alias/range/copy/move 已覆盖；交付/性能随 candidate 验证 |
| `CONTAINER-HASHMAP-01` | MUST；util/io consumer；Abseil Swiss family | map/set owning；`try_reserve` 可恢复，`reserve` fail-fast；grow/erase 失效 | expected O(1)，7/8 load，零额外常态 metadata | `COMPLETE` | `KEEP` | tombstone/OOM/property、util/io 迁移与自然头已覆盖；不做 node stability |
| `CONTAINER-FLAT-01` | MUST；read-mostly index + Boost default vector storage | flat SoA owner；`SmallVector<T,0>` internal sequence；普通/try failure 保持 | lookup O(log n)，insert O(n)；空对象不按 sizeof(K/V) 增长 | `COMPLETE` | `KEEP` | object-size、non-trivial/OOM/bulk/range 已覆盖；正常 workload 随 P1 验证 |
| `CONTAINER-BTREE-01` | MUST；ordered mutation/range + Boost | unique owning tree；fail-fast/InsertResult OOM；外部同步 | O(log n) lookup/mutation，O(n) copy/range | `COMPLETE` | `KEEP` | multi、stable ref、node handle/rank-select 不准入；property/differential即可 |
| `CONTAINER-CMAP-01` | MUST；registry/cache + oneTBB/Folly；可复现 writer-pending窗口 | accessor owns lock；flat/node stable variants；quiescent management | expected O(1)；无 reader 在 pending writer 后新进入；AArch64 spin 用 `yield` | `COMPLETE` | `KEEP` | pending race与两 map 形状已覆盖；TSan/AArch64/runtime replay 随 candidate 验证 |
| `CONTAINER-CVECTOR-01` | MUST；parallel append + oneTBB 3-entry embedded table/clear | stable segmented owner；append completion contract不变；clear只允许 writers quiesced | append amortized O(1)；空对象 O(embedded table)，extended table只在 grow时分配 | `COMPLETE` | `KEEP` | clear/reuse/non-trivial/OOM/stress 已覆盖；资源/NI 随 candidate 验证 |
| `CONTAINER-MPMC-01` | MUST；typed stage handoff；rigtorp `try_emplace` placement lifecycle | bounded MPMC owns live slots；nothrow construct/destruct；move/callback pop；quiescent destroy | O(1) lock-free；trivial payload codegen/slot metadata不回退 | `COMPLETE` | `KEEP` | move-only/nonassignable/destructor/OOM/stress 与 Sharded 传播已覆盖；TSan/peer 随 candidate 验证 |
| `CONTAINER-ROLEQUEUE-01` | MUST；sched consumer + mature owner/thief algorithms | SPSC 与两种 stealing role显式；full/empty/retry；quiescent teardown | O(1)，预分配，SPSC无 CAS | `COMPLETE` | `KEEP` | wide/non-lock-free stealing payload传稳定 pointer；blocking/cancel在 sched |
| `CONTAINER-ACTIVE-01` | MUST；mask/frontier/active IDs + Boost dynamic_bitset | dense/sparse owners；size match是调用合同；补 value equality | O(words) algebra/equality、O(1) sparse mutation | `COMPLETE` | `KEEP` | tail padding/equality 已覆盖，ISA differential 随 candidate；compressed bitmap/rank-select defer |
| `CONTAINER-RING-01` | MUST；stream staging/deque | single-thread owner；bounded/grow、partial I/O、span invalidation | both-end O(1)，grow O(n)，byte path零中间 copy | `COMPLETE` | `KEEP` | current correctness + consumer replay；不加入 wait/close metadata |
| `CONTAINER-TEXT-01` | MUST；HTTP/registry/path + standard owner/view rule | `StringView` 永远 non-owning；拥有结果由 caller storage/std::pmr::string；span batch/buffer | search/hash O(n)，path lexical O(n)；不隐藏 allocation/free | `COMPLETE` | `KEEP` | owned helper 已删除，path/interner span 与 borrow/NUL tests 已覆盖；sanitizer/consumer 随 candidate |
| `CONTAINER-HASH-01` | MUST；table/interner + xxHash/simdutf pins | fast process seed / deterministic seed / caller-secret hardened policy | bytes O(n)，integral O(1)，无 hidden allocation | `COMPLETE` | `KEEP` | source-primary code-size、short/long hash与 UTF-8 differential；不声称 cryptographic fast hash |
| `CONTAINER-API-01` | MUST；公开 discoverability 与当前 C++ failure convention | natural type-named headers；普通 `reserve` fail-fast、`try_reserve` recoverable；无 C ABI | forwarding header零运行成本；不增加 duplicate implementation | `COMPLETE` | `KEEP` | header self-containment、sample/consumer 编译与 stale API scan 已通过；package journey 随 delivery |
| `CONTAINER-DELIVERY-01` | MUST；central graph/package contract | C++23 source/static/shared；private vendors；无 Native FFI/C API | exact selected closure；独立 target ownership 不被传递安装混淆 | `COMPLETE` | `KEEP` | source embedding、relocated static/shared 各 1/1；offline/pin/license、17 个 direct-consumer replay 与 target ownership 对账通过 |
| `CONTAINER-PERF-01` | MUST；旧性能页不满足当前统一仪器 | 五个 frozen family分别资格化；peer语义/allocator/layout等价 | W1-WIN 5% NI；同时记录 object/slot/resident bytes 与 progress | `PARTIAL` | `BLOCKER` | 五项具体 slice 全部 8/8 strata NI PASS；但冻结卡还要求的 sorted/ordered/concurrent/ring/text/interner 等轴未进统一仪器，未运行不能写 PASS |
| `CONTAINER-DOC-01` | MUST；architecture/user/performance仍 provisional/旧 C 叙事 | architecture=边界；user=选型/陷阱；performance=当前证据；task=事实 | 不复制易腐点值、旧 milestone 或 benchmark 形状 | `COMPLETE` | `KEEP` | provisional 标记、旧性能总账与 intrusive 交付歧义已移除；current 边界、选型/陷阱、证据与 blocker 对账并通过 docs checker |

### `GENERAL` / negative space

- **owning string、普通长序列、单线程 node-stable map：`ROUTED`。** 分别使用 `std::string`/
  `std::pmr::string`、`std::vector`/mem PMR 与 `std::map/std::unordered_map`。HYLE 只有 profile 证明标准
  shape 的算法或 allocator seam 不够时才另立能力，不能让 non-owning `StringView` 冒充 owner。
- **ordered multi、node extraction/merge、rank/select、compressed bitmap：`DEFER`。** 单个成熟库存在不足以
  准入；需要第二个独立来源或具名推理/工程消费者，并在同轮交付完整生命周期和成本模型。
- **LRU/cache/eviction、persistent containers：`REJECT` 本模块。** 它们需要 quota、I/O、serialization、
  invalidation 与 service policy，owner 是 engine/io/上层库。
- **blocking/unbounded/priority queue service：`ROUTED` 到 sched。** container 只提供预分配机制；等待、
  shutdown、cancel、backpressure 和 callback owner 必须同 runtime 一起设计。
- **完整 Unicode transform/normalization/regex：`ROUTED` 到专用 text/上层 adapter。** container 保留
  byte/ASCII 与 UTF-8 validate/count/decode mechanism；不复制 ICU。
- **Native FFI/C API：`REJECT`。** 当前无已准入 foreign-runtime consumer；source/static/shared 都是
  typed C++ 产品，不恢复旧 carray schema、handle 或动态符号。

### 冻结施工边界

本轮产品修改只允许以下七个批次：

1. `CONTAINER-POD-01`：去掉 C-era allocation header，保留 typed `PodVector`；
2. `CONTAINER-FLAT-01`：合法的 zero-inline sequence core 与 flat normal-object layout；
3. `CONTAINER-MPMC-01`：non-trivial/move-only placement lifecycle，并传播到 sharded 组合；
4. `CONTAINER-CVECTOR-01`：小 embedded table、按需 extended table 与 quiescent `clear()`；
5. `CONTAINER-CMAP-01`：writer-pending race、AArch64 relax 与 concurrent reserve failure形状；
6. `CONTAINER-TEXT-01`：owner/view 分离和 span-oriented path/interner facade；
7. `CONTAINER-ACTIVE-01` + `CONTAINER-HASHMAP-01` + `CONTAINER-API-01`：bitset equality、reserve
   一致性与自然 include 入口。

随后只允许 `CONTAINER-DELIVERY-01`、`CONTAINER-PERF-01`、`CONTAINER-DOC-01` 的测试、仪器、交付和
current 文档工作。新的 correctness/security blocker 必须先回到 ledger；命名偏好、单个 peer feature、
孤立 benchmark 点值和未具名上层 policy 不得扩面。审计预算内没有发现必须新增另一种核心 container
family 的证据；功能缺口集中在已有 family 的不完整生命周期与错误成本形状。

### Implementation progress（`CANDIDATE`）

| batch | current fact | remaining verdict |
|---|---|---|
| `CONTAINER-POD-01` | 已删除 C-era `carray.h`/header-before-data；`PodVector` allocation 只含 elements，typed owner 保存 allocator/size/capacity；Windows 定向与完整 container tests 已通过 | correctness/sanitizer/三交付 PASS；仍随聚合 P1 blocker 保持本轮未关闭 |
| `CONTAINER-FLAT-01` | `SmallVector<T,0>` 具有无 dummy element 的合法空 storage；`FlatMap/Set` 空对象不再内嵌 K/V，增量 mutation 使用几何容量；Windows 定向、对象资源断言与完整 container tests 已通过 | correctness/资源与现有 sequence replay PASS；sorted/ordered P1 轴仍属 blocker |
| `CONTAINER-MPMC-01` | Bounded/Sharded MPMC 以既有 sequence 作为 lifetime state，新增 noexcept placement `try_emplace`、move pop 与 callback consume；move-only stress/lifetime tests通过；MSVC 19.50.35728 `/O2` trivial-u64 probe 对比 parent `3d094e84`：queue object 均为 `0x180`，外部 push/pop 的指令数与控制流不变；实际 `sched` target 编译及 21 项 RunQueue/Scheduler replay 通过 | WSL TSan 与 rigtorp 同合同 W1-WIN 8/8 strata NI PASS；其他 role queue 性能轴仍属 blocker |
| `CONTAINER-CVECTOR-01` | 采用 3-pointer embedded + on-demand extended table；table promotion OOM 保持旧 capacity/value；quiescent `clear()` 销毁 live T 并保留 segment capacity 复用；Windows 14 项定向测试通过 | sanitizer/跨平台 PASS；append/index/iterate P1 轴仍属 blocker |
| `CONTAINER-CMAP-01` | reader admission 的 fetch-add 线性化点同时拒绝 writer/pending，关闭 writer 发布 pending 后的 reader barging；AArch64 spin hint 从 pipeline-serializing `isb` 改为 `yield`；两种并发 map 统一为 fallible `try_reserve` / fail-fast `reserve` | WSL TSan、M4 AArch64 runtime 与 direct consumer PASS；concurrent map P1 轴仍属 blocker |
| `CONTAINER-TEXT-01` | 删除把 raw allocation 装进 borrowed `StringView` 的 helper；普通 owner 路由标准 string，path producing ops 写 caller-owned spans，normalize 按 view length 返回借用 output 的 optional view；interner batch改为 spans并以 `prewarm` 取代 nullable output；Windows 64 项 text/path/interner、完整 container、header 与更新后的 benchmark source build 通过 | OOM/sanitizer/direct consumer/package PASS；text/interner P1 轴仍属 blocker |
| `CONTAINER-ACTIVE-01` + `CONTAINER-HASHMAP-01` + `CONTAINER-API-01` | `DynamicBitSet` value equality 比较逻辑 size 与 canonical blocks；Swiss map/set 统一 `try_reserve` 可恢复、`reserve` fail-fast，并迁移 util/io；九个类型名入口头由现有 family header 零实现转发；Windows 完整 container/header、util consumer 与 IO 编译闭包、benchmark source build 通过 | bitset count 与 flat integral lookup slice PASS；其余冻结 family P1 轴仍属 blocker |
| `CONTAINER-CORR-01` | `SmallVector<T,0>::move_from` 以 `if constexpr` 排除 inline byte copy；新增 trivial empty move/swap regression；原失败 WSL GCC 14.2 Release + WERROR 与 Windows 完整 container 各 3/3 通过 | replacement candidate 三平台与 sanitizer 已重验；PASS |

## Candidate

- product commit / tree：`19a1e9b515522ad74872fb963a3b3260cf915bab` /
  `68d456382352d2d81fdd41e4d1c725440424336a`。
- 公开合同与 user journey 变化：旧 C-era PodVector 外壳删除；flat/sorted/ordered/concurrent 类型有自然
  include；哈希 reserve 失败语义与其他 HYLE 容器一致；typed MPMC、quiescent concurrent-vector reuse、
  bitset equality、borrowed text + caller-owned output/span batch 已形成完整 C++ journey。
- ownership / lifetime / concurrency / failure 变化：PodVector allocation 只含 elements；StringView 永远
  borrowed；path producing operations 写 caller buffer；MPMC 管理 non-trivial slot lifetime；writer pending
  后拒绝新 reader；普通容量入口 fail-fast、`try_*` 保留可恢复 OOM。
- 算法 / 复杂度 / 常态 metadata 与 layout 变化：已有渐进复杂度档不变；flat storage 空 owner 不再嵌
  任意 K/V；ConcurrentVector 由 3 个 embedded segment slots 按需提升到 extended table；没有为
  benchmark 添加 per-entry metadata、额外 lookup/hash、常态 lock 或 dynamic dispatch。
- 直接消费者迁移：sched 的 RunQueue/Scheduler 已消费 typed MPMC；`util::Registry` 与 IO readiness/HTTP2
  表使用可恢复 reserve；package/sample 使用自然 flat-hash 入口。下游模块只验证这些直接合同，不继承
  container 的闭包结论。
- candidate tests：W1-WIN、W1-WSL GCC 14 与 M1-MAC 的精确 container 三项 CTest 均 3/3；WSL
  ASan/UBSan/TSan 共 324/324。公开头 self-contained、util/sched/io/graph 17/17 direct-consumer replay、
  container-only offline configure/build 及 source/shared/static relocated consumer 全部通过。五项 P1
  slice 的 bundle 通过，但不能替代冻结卡其余未运行性能轴。
- harness / tool identity：correctness/delivery 使用 `closure-correctness-v1` 与中央 component/package
  contracts；P1 使用 `closure-perf-v1`、`W1-WIN` 和本卡冻结五个 family。harness/evidence commit 在各门
  运行时单独记录，不改变上述 product identity。
- rejected predecessor：`8c5d036e5ff7819974a93eae86efc8f536ab38fc` / tree
  `b7407252b0b093b6b2a5549427b6b3173c9e254c` 因 `CONTAINER-CORR-01` 被拒绝；任何 gate verdict 与性能
  evidence 均不得从该 identity 继承。

## Verification manifest

| gate | evidence / verdict |
|---|---|
| Windows correctness/header | MSVC 19.50 Release + WERROR；container 精确 3/3 `PASS` |
| W1-WSL correctness/sanitizer | GCC 14 Release + WERROR 3/3；ASan/UBSan/TSan 324/324 `PASS` |
| M1-MAC correctness/ISA | AppleClang 21 Release + WERROR 3/3；AArch64 scalar/NEON TUs build/runtime `PASS` |
| direct consumers | util/sched/io/graph 17/17 `PASS`；只验证其实际 container 合同 |
| delivery | selected container source embedding、relocated shared SDK、relocated static SDK 各 1/1；offline configure/build、pin/license projection `PASS` |
| dependency/ownership | `PUBLIC core mem`、`PRIVATE platform`、`PRIVATE_BUILD simd_vec_headers` 对账；selected SDK 因传递闭包安装独立 `container_intrusive`，但 target ownership 不混淆；`PASS` |
| selected P1 slice | clean harness `9f3fa43a`，contract SHA-256 `f46e4e4a...f3a4`；五项各 8/8 strata NI `PASS`，全部 controls/validators与 deep replay `PASS` |
| aggregate P1 | 冻结卡的其余正常轴未运行；`INCONCLUSIVE` / blocker |
| current documents | architecture/user/performance/task 职责对账；旧总账与 provisional 叙事移除；repository docs checker `PASS` |

## 独立红队

审查者 / 日期 / 预算：post-candidate reviewer / 2026-08-29 / 30 分钟上限。只攻击冻结的 capability、
public surface、复杂度、ownership/lifetime、依赖/交付、文档与工程性能门；发现硬停止后不扩建公共
仪器、不新增产品功能或第六种 benchmark 姿势。

| finding | severity | decision | evidence |
|---|---|---|---|
| user/architecture 仍带 provisional/NOT_REVIEWED，且把“独立 target”误写成选择完整组件时不会随 SDK 取得 | documentation/usability | `fixed` | current guide/architecture 区分 target ownership 与中央传递安装；docs checker PASS |
| 旧 C array/schema/Native FFI 或隐藏 product drift 仍可能存在 | blocker if true | `disproved` | `carray_`/`CArrayHeader`/旧 bridge scan 无产品命中；candidate→harness 的 `include/hyle/container`、`src/container` diff 为空；三交付 smoke 无旧符号需求 |
| 五项 qualified replay 能否代签冻结卡的完整五个 family | performance/blocker | `accepted blocker` | contract 只有 sequence mixed、integral flat lookup、MPMC、bitset count、64B hash；冻结卡另要求 sorted/ordered/concurrent/ring/text/interner 等正常轴及无 peer baseline NI，未运行不得 PASS |
| 第一版 MPMC 的 31-sample 双峰是否证明产品慢 | instrument | `fixed/disproved` | adapter 把 wait/notify/yield 放入计时；改为同一常驻 worker + busy-pause 后 8 个 diagnostic strata 稳定，clean n=15 的 8/8 strata NI PASS；产品实现未改 |
| public header family、direct dependency 与 consumer owner 是否遗漏 | architecture | `disproved` | 36 个 header/public-surface trace 对账；intrusive 只有 time 与独立 sample 直接 include；util/sched/io/graph 17/17 replay 与中央 edge 对账 PASS |

未发现新的产品 correctness/security、OOM、lifetime、复杂度档、常态 metadata、API ownership 或
source/static/shared 交付缺陷。红队唯一未解决项是冻结 P1 覆盖不足；因此 red-team 流程本身完成，
但不能把聚合 engineering-performance gate 写成通过。

## Closure decision

本轮从 `CANDIDATE` 终止为 **`BLOCKED`**，而不是 `VERIFIED/CLOSED`。scope、architecture、
correctness、delivery、usability/docs 与 red-team 执行均为 `PASS`；全部 `MUST` 产品能力为
`COMPLETE` 或完整 `ROUTED`，产品 candidate 没有新 blocker。唯一 blocker 是冻结卡的 P1 正常轴只完成
五项具体 qualification，剩余轴未运行，故聚合 engineering performance 必须为 `INCONCLUSIVE`。

blocker owner：`container/container-v1-p1-coverage`。恢复条件：按 `BLOCKED -> SCOPED` 创建新的
`RESUME` round，复用未变化的 product candidate，并在 `closure-perf-v1` / W1-WIN、5% margin 下为以下
有限集合取得合格 verdict：strict fixed/non-trivial sequence；string/sorted/ordered associative；
SPSC/sharded/work-steal/concurrent vector/concurrent maps；bitset algebra/SparseSet/rings；short/long hash、
UTF-8/search/case/interner。无同合同 peer 的轴使用 immutable candidate baseline。恢复轮不得继承旧
2026-07 点值，不得缩窄 ownership/concurrency 语义，也不得因为结果不利修改公开 API/layout/常态 metadata。

已知限制：

- 当前只能声明已列五项 consumer slice 在 W1-WIN 通过 5% NI，不能声明 container 聚合 P1、全面
  SOTA、跨平台固定点值或 Formal；
- `SmallVector` 在该 slice 快于 Abseil，但 owner object 为 88 B 对 72 B，资源 trade-off 必须保留；
- ordinary long sequence/owning string/node-stable single-thread map 继续路由标准/PMR；cache、blocking
  service、ordered multi、node extraction/merge、compressed bitmap/rank-select 与 Native FFI 仍是具名负空间；
- independent `container_intrusive` 不属于本产品面；selected SDK 的传递安装不改变 target ownership。

产品重开 trigger：真实 consumer 暴露本边界 correctness/security/OOM/lifetime/usability 或缺失能力；
container 算法、public API/layout、直接依赖、vendor/ISA 语义或 source/static/shared 交付改变；强 peer
证明更优复杂度档或正常 consumer 上 material advantage。当前 performance blocker 满足 resume
condition 时是恢复本轮证据，不是无新事实重开产品的许可。

四项 LOC 从 immutable product candidate
`19a1e9b515522ad74872fb963a3b3260cf915bab` 运行
`python scripts/loc-report.py --module container --commit <candidate> --self-check --json`，self-check `PASS`：

1. 全库代码：305,616；
2. 自实现包含测试：124,208；
3. 自实现不包含测试：83,088；
4. container 实现：11,591（`include/hyle/container` + `src/container`，不含独立
   `container_intrusive`）。

clean selected-slice performance bundle、hash 与复现命令见
[`benchmarks-container.md`](../../evidence/performance/benchmarks-container.md)；correctness、delivery 与
consumer verdict 见上方 verification manifest。terminal end/registry anchor 的完整 commit identity 由
后续 registry transition 记录。
