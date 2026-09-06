# Container v5：真实推理能力与内部因子化

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由当前全产品只读回放触发。它不继承 v4 的窄 CPU-selector 结论，也不把历史能力候选直接当作
> 实现授权。机械状态只以 [`module-status.json`](../../module-status.json) 为准。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`container` / Container /
  `container-v5-product-factorization` / `REOPEN`；
- baseline：`a9b5b6523867601dd2c708a2bedc8fc4d81469f2` / tree
  `fadf478497ce9a873a31ea984ed4ba165765530f`；前轮：
  `container-v4-cpu-feature-migration`；
- trigger：`CONTAINER-INFERENCE-SUPERSET-AND-DELIVERY-2026-09-07`；完整只读证据见
  [`Container 当前全产品能力回放`](container-current-product-replay-2026-09.md)，terminal verdict 为
  `PRODUCT_TRIGGER`；
- `CONT5-R1`：public Swiss probe 宣称 SSE2 baseline，但 MSVC x64 无条件编译 SSSE3 指令，target 又将
  SSSE3 option 以 `PUBLIC` 传播给 consumer TU；该形状存在合法旧 x86-64 上的 illegal-instruction 风险，
  并污染任意消费者的 compile ISA floor；
- `CONT5-R2`：vendored simdutf 直接编入 shared Container，Unix 没有 hidden visibility，私有 provider
  namespace 可能进入动态导出表；
- `CONT5-R3`：ggml/llama.cpp、ds4、vLLM、SGLang 的当前消费与成熟 peer 共同证明 generic eviction
  storage 等能力候选必须重新裁决；v4 明确排除了能力/API，不能代表当前完整产品；
- `CONT5-R4`：真实 target 仍把 sequential、concurrent、queue 与 text 编为单一 Container factor；owner
  已要求内部独立/并行开发和最小依赖闭包，因此长期四因子边界需要在本轮正式 admit/reject 并实施，而
  不能继续只留在历史架构记录；
- 初步受影响面：非 intrusive Container public surface、compiled algorithms/providers、内部 factor 与
  product façade、直接消费者依赖、source/static/shared 交付、用户/架构/性能文档；
- 初步 tier：`P1`。容器 lookup、queue handoff、text codec 与 cache touch 属可能高频 primitive；能力和
  API 未冻结前不开始计时，任何 peer 行先完成 comparison-class 与语义等价；
- 只读依赖：`core`、`error`、`container_intrusive`、`util_core`、`time`、`platform`、`mem`、
  `mem_reclaim` 等已 FROZEN factor。只有具体事实击中其自身 trigger 才另开相邻轮；本轮不得顺手修改；
- 非目标：Compute/Graph、tokenizer 或 KV-cache policy、scheduler wait/cancel/service lifecycle、IO protocol、
  Native FFI/C API、稳定跨工具链 C++ ABI、复制标准库/peer 全表面、公共 closure/performance 仪器重写与
  Formal SOTA；
- 下一状态：先冻结产品 suite 的 scope、预算、机器、factor reservation 和 gate 后进入 `SCOPED`；再完成
  top-down/bottom-up audit、普通 C++ 调用草图与有限 FIX ledger 后进入 `AUDITED`。此前不修改产品代码。

## Closure card（`SCOPED`）

### 产品、内部因子与边界

- consumer-facing 产品仍只有 `hyle::container` / `HYLE_COMPONENTS=container`；内部 factor 不制造用户
  必须理解的新产品，也不提供单独稳定 ABI；
- 现有 `container` component 收窄为 sequential storage/algorithms；登记
  `container_concurrent`、`container_queue`、`text` 三个 `QUEUED` identity。它们只有完成各自
  `SCOPED/AUDITED`、公开头 owner 和真实 target 后才能进入 Container façade；
- 预期语义边界：
  - `container`：small/fixed/POD/ring sequences、flat/sorted/B-tree/hash、bitset/sparse set，以及审计
    准入的 heap/cache storage；
  - `container_concurrent`：concurrent map/set/vector、accessor/publication 与 reclamation；
  - `container_queue`：try-only SPSC/MPMC/sharded/work-steal/unbounded/priority storage；不拥有
    wait/deadline/cancel/close/backpressure service；
  - `text`：borrowed text、ASCII/Unicode codec、lexical path 与 interning；不拥有 filesystem I/O、
    tokenizer/model automaton 或 protocol；
- 公开 include 路径和 C++ namespace 本轮先保持 `hyle/container/**` 与 `hyle::container`，因为内部
  factorization 是依赖/交付机制而不是要求用户理解的产品拆分。审计若证明 namespace 本身造成 owner
  冲突，必须在 `AUDITED` 前显式裁决，不能实现期偷偷迁移；
- `container_intrusive` 保持只读 FROZEN factor，并继续由同一产品 façade 聚合。新通用 LRU 若需要
  caller-owned membership，将优先复用它；不能复制第二套 list hook。

### 冻结输入与能力审计

- inference demand identities：llama.cpp
  `3ad1ba7336986d98592d3e28cafd1a406715351f` / embedded ggml tree
  `7674bdaa9279cfc46abde8ce489652f508c0db09`，standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`，ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`，vLLM
  `52358e6e192aeae73dd3764046c98fc4b156ea83`，SGLang
  `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`；这些只证明 demand，不自动成为 timing peer；
- mature peer identities：Abseil `38f8c1535de9ebc12c6340e5c1c8ebd72afde560`、Boost
  `cb94f33fcb13e90922905c2d8105508e71fa6db4`、Folly
  `65749da412d92fe27d87ae45a8ef670720712858`、oneTBB
  `e00c68dcac0e83947c9b54f6d5304c0616c9639f`、simdutf
  `b91b70d7d7bf8609eec2612971c01f3c92499a4b`。每个 family 最多选两个强 peer；
- direct/build-only mechanisms：vendored xxHash
  `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53`、wyhash
  `e4764a0b637d34d3421a7760affada9288b625a8`、simdutf v9.1.0 与 centrally locked Highway；必须逐项
  裁决 `BORROW/LOCAL`、隐藏符号、license、offline pin 和 source/static/shared closure；
- top-down map 沿用已冻结的 `CONT-A-01..37` 作为历史输入，但必须用上述五个当前 inference identity
  重新确认；bottom-up 必须枚举全部非 intrusive public entry、编译 symbol、tests/bench-only seam 和
  内部消费者；
- 必须在 `AUDITED` 前对以下候选给出唯一结论和普通 C++ 调用草图：generic entry/weighted LRU storage、
  concurrent flat/node set、unbounded typed queue、Unicode error-offset/UTF-8/16/32 conversion、
  mutable/indexed heap、concurrent priority queue、trie/radix/matcher route、throwing-move sequence support，
  以及 associative `try_emplace` 与 recoverable OOM 的命名；
- 已有 public 类型只有映射到需求并完成 lifecycle/failure/concurrency/complexity 才保留；不以“已经写了”
  或 benchmark 仍能编译作为理由。标准 `vector/string/deque/node-map/ordinary heap` 能自然完成的任务
  继续 `ROUTED`，不复制表面。

### 依赖、成本与性能

- 目标依赖只在 `AUDITED` 后冻结。候选最小图为：sequential -> `core,mem`（若 LRU 复用 hook，再
  PUBLIC `container_intrusive`）且 compiled selector PRIVATE `platform`；
  concurrent -> `core,mem,mem_reclaim,container`；queue -> `core,mem`；text ->
  `core,mem,container` 且 compiled CPU facts PRIVATE `platform`。不得为了依赖数量复制 allocator、
  hash、reclamation、wait 或 ISA authority；
- `CONT5-R1` 必须让 public inline x86 baseline 真正保持 SSE2，不能以 PUBLIC compiler option 扩大
  consumer TU；显式 higher-ISA consumer 可由它自己的 compile target 选择；
- `CONT5-R2` 必须让 private provider symbols 在 Unix shared artifact 中 hidden，同时保持所有
  `HYLE_*_API` product symbols 可导入；
- P1 / 5%，instrument `closure-perf-v1`，资格机 `W1-WIN`；W1-WSL 与 M1-MAC 负责 correctness、
  sanitizer、真实 ISA 和三交付，不外推跨平台 SOTA；
- API/usability freeze 前不得计时。审计后从下列风险族中每个 factor 只选一个 normal primary 和至多
  一个不同粒度 holdout：短字符串 metadata hash lookup、small spill sequence、cache hit/insert/evict、
  concurrent registry membership、bounded/unbounded handoff、UTF validate/transcode。未改且未被真实
  profile 指为热点的 family 只做 P0 complexity/codegen 回放；
- 有 peer 的每行运行前固定 `SAME_CONTRACT` 或 `NATURAL_ROUTE`，对齐 result、ownership、OOM、
  ordering、capacity/backpressure、thread progress、allocator 和 setup/teardown。单侧 safety/service、
  raw provider 对 product seam、地址/顺序翻转和亚纳秒点值不得签实现 non-inferiority；
- 每个同合同 row 的 material margin 为 5%。若固定方向输但仍在门内，只做一次两钥匙复核；没有真实
  workload materiality 和机制差异，不改 API/layout/常态 metadata。性能最多允许一次产品导向修正和
  一次完整重放，不增加第三种姿势。

### Gate、预算与停止

- correctness：normal、empty/zero、capacity/wrap/overflow、OOM、alias/self-insert、copy/move/destroy、
  iterator/reference invalidation、malformed text/error offset/output-too-small、eviction callback、并发
  publication/erase/quiescence/progress、property/differential/fuzz、stress、ASan/UBSan 与适用真实 ISA；
- build/delivery：每个 factor exact developer build、Container source embedding、installed static/shared
  relocated consumer、selected header owner、central direct edge、Windows runtime staging、Unix private
  vendor symbol scan、offline dependency/license、无公开 C/Native FFI；
- usability/docs：正常构造/操作/失败/teardown compile journey；guide 按“何时用标准库、何时用哪个
  HYLE family”说明 ownership、OOM、iterator、thread role 和陷阱；architecture 写稳定 owner/因子，
  implementation/performance 只留事实；
- 独立终态红队必须攻击 factor owner、API benchmark 污染、exception/OOM、并发 lifetime、comparison
  class 与 dynamic symbol surface；当前只读红队是 trigger evidence，不替代 candidate 红队；
- 总预算：capability/API/factor audit 3 小时，实现 7 小时，correctness/build/docs 4 小时，P1 2 小时，
  candidate 红队 30 分钟。到达任一预算就按已冻结 gate 裁决，不继续扩仪器或能力矩阵；
- `SCOPED -> AUDITED` 的唯一输出是完整 capability map、public trace、factor/header/source owner、
  普通调用草图、第三方/内部复用 ledger、有限 FIX 集与性能合同。此前没有产品实现授权；
- terminal：所有准入 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，四个 factor 与 façade、correctness、
  三交付、docs、红队和 P1 全部通过才可各自 `CLOSED/FROZEN`。适用项未运行或结果不可裁决为
  `BLOCKED/INCONCLUSIVE`；“继续研究”不是状态。

## Audit verdict（`AUDITED`）

### 完整 public trace 与 owner

本轮逐项反查了全部非 intrusive Container headers、compiled source、tests、bench seam 和直接消费者。
公开产品仍只有 `hyle::container`；下表的 factor 名只用于内部 target、最小依赖闭包和并行维护：

| 内部 owner | 当前公开面 | 审计结论 |
|---|---|---|
| `container` | small/static/inplace/POD vector，ring/circular sequence，flat hash/sorted/B-tree，dynamic bitset、sparse set，semantic hash | 保留自然任务；新增通用 LRU storage；清除 `SwissTable`/public probe 的实现名泄漏，并把 throwing-move 与 fallible insert 约束放到真正需要它们的操作 |
| `container_concurrent` | concurrent flat/node map、concurrent vector | 保留两种 map 的 compact/stable-node 取舍；补 flat/node set，不能要求用户伪造 mapped payload；复用 `container` 内同时服务 text interner 的 adaptive-lock mechanism，不复制第三套锁 |
| `container_queue` | byte ring、SPSC、bounded/sharded MPMC、work-stealing deque、owner/thief FIFO | 保留不同 producer/consumer/progress 合同；补 try-only unbounded typed handoff；等待、关闭、取消与 deadline 留给 Sched/IO |
| `text` | `StringView`、ASCII/string algorithms、lexical path、interner/ID、UTF provider | 保留 borrowed/identity 路线；补 typed UTF diagnostics 与 UTF-8/16/32 caller-buffer conversion；provider/ISA 仍是 private mechanism |
| `container_intrusive` | list/forward-list/hlist hooks | 已 FROZEN，只由 façade 聚合并供 LRU 复用；本轮不修改它 |

`probe.h`、`swiss_table.h`、adaptive locks 与 generated/provider headers 是实现所需安装 header，不是第二套
consumer product API。它们必须进入对应 factor 的 header ownership，但 guide 不把这些 seam 推荐为用户入口。

### 能力超集裁决

| 能力族 | 来源与真实任务 | 当前覆盖 | 动作 / owner |
|---|---|---|---|
| small/fixed/POD sequence | 四组引擎的短 metadata、token/result staging；LLVM/Abseil/Boost | `PARTIAL`：能力齐，但 SmallVector/RingDeque class-wide 排除 throwing move | `FIX`：只在 relocation/shift 操作表达实际 exception contract，不让整个类型失去可用性 |
| exact ring / byte ring | streaming token、log/transport staging；Boost | `COMPLETE` | `KEEP`；分别保留 overwrite、deque 和 raw-byte 合同 |
| flat hash map/set | graph/metadata/registry；Abseil/Folly | `PARTIAL`：任务完整，公开实现名和 OOM vocabulary 混乱 | `FIX`：用户入口只叫 `FlatHashMap/Set`；fallible 与 std-shaped insert 语义可见且不冲突 |
| flat sorted / B-tree | 小型 read-mostly index、有序 range；Boost/Abseil | `PARTIAL`：任务完整，fallible `try_emplace` 命名同样混乱 | `FIX`：统一插入结果语义；不补 multi/node-handle 全表面 |
| dense/sparse membership | active masks、dense IDs；ggml/llama.cpp | `COMPLETE` | `KEEP`；compressed rank/select 无当前通用需求 |
| bounded recency/eviction storage | ggml CANN cache、vLLM LRU/LoRA、SGLang expert/KV cache；Folly `EvictingCacheMap` | `MISSING` | `FIX` / `container`：entry/weight budget、peek/touch/get、erase/clear、逐出 callback；device/KV/in-flight/pinning policy 留调用者 |
| concurrent registry map | backend/model registry；oneTBB/Folly | `COMPLETE` | `KEEP`；flat 偏密集，node 给 guard 内 stable address |
| concurrent membership set | 并发 active/registered membership；oneTBB/Folly | `MISSING` | `FIX` / `container_concurrent`：flat/node set，无 mapped payload，继承相应 guard/reclamation 合同 |
| parallel append | request/result publication；oneTBB | `COMPLETE` | `KEEP`；不把 random erase 冒充 concurrent vector 能力 |
| bounded/SPSC/work-sharing queue | engine stage handoff、worker run queue；rigtorp/oneTBB | `COMPLETE` | `KEEP`；每种类型继续显式 role、capacity、progress 和 quiescence |
| unbounded typed handoff | llama server burst deque、vLLM/SGLang runtime queues；Folly/oneTBB | `MISSING` | `FIX` / `container_queue`：try-only MPMC storage、recoverable OOM、approximate size；不吸收 blocking service |
| priority/mutable heap | vLLM arbitrary request removal、llama BPE/普通 priority；Boost heap | `ROUTED` | 普通 heap 走 `std`；indexed/decrease-key 保持 watchlist，第二个独立通用 consumer 或 material profile 才准入 |
| concurrent priority queue | oneTBB peer，但四组 consumer 的 policy 由单 scheduler owner 串行维护 | `OUT_OF_SCOPE` | `DEFER` 到真实 Sched/engine demand；不让普通用户支付并发 heap 成本 |
| borrowed text/path/interning | model metadata、token text、Windows path boundary | `COMPLETE` | `KEEP`；filesystem I/O 仍由 Platform FS，owning string 用标准/PMR |
| UTF diagnostics/transcode | ggml Windows/backend boundary、llama token text、simdutf | `PARTIAL`：仅 bool validate/count | `FIX` / `text`：typed code+input offset+output count 与 UTF-8/16/32 caller-owned spans |
| trie/radix/Aho/token automaton | llama trie、SGLang radix/ngram | `ROUTED` | tokenizer/matcher/KV owner；alphabet、match、eviction 和 serialization 不具备一个诚实通用合同 |
| ordinary vector/string/deque/node map、stable file checksum | 全部引擎 | `ROUTED` | 分别用标准/PMR、IO/format owner；不复制 API 数量 |

该表覆盖旧 `CONT-A-01..37` 的全部已有公开族，并以当前锁定的 ggml+llama.cpp、ds4、vLLM、SGLang
identity 重新验证差异。没有现存公开类型因只服务 benchmark 而删除；被清除的是实现 seam 和含混命名，
不是实际任务。每项最多使用两个成熟 peer；Python/GPU policy 只证明需求，不作为 C++ 点值性能对手。

### 冻结普通调用形状

以下是语义草图，不冻结具体拼写以外的实现细节；candidate 必须让这些调用无需 internal seam：

```cpp
LruCache<Key, Value> cache({.max_entries = 1024, .max_weight = bytes});
auto put = cache.try_insert_or_assign(key, value, weight, on_evict);
if (Value* value = cache.get(key)) { /* hit also touches recency */ }
if (const Value* value = cache.peek(key)) { /* no recency mutation */ }

ConcurrentFlatHashSet<Key> active;
auto inserted = active.try_insert(key);       // inserted/already-present/OOM
bool present = active.contains(key);          // snapshot result

UnboundedQueue<Job> ready;
if (!ready.try_emplace(args...)) { /* allocation failure */ }
ready.try_pop([&](Job&& job) { consume(std::move(job)); });

UtfResult result = utf8_to_utf16(input, output);
// result exposes status, input_offset and output_count; output-too-small is explicit.
```

LRU 的 weight 由调用者在插入/更新时给出，不保存 type-erased callback；eviction callback 是单次模板参数，
默认路线无常驻函数指针。返回的 value pointer 只在下一次可能 erase/evict 的 mutation 前有效。并发 set 的
node variant 只在 accessor 生命周期内给 stable key；unbounded queue 的 `try_pop` callback 让 nonassignable
payload 无需额外默认构造或 assignment。Unicode API 不分配、不拥有 output，也不暴露 simdutf 类型。

### 第三方、内部复用与 dependency verdict

| 能力 | semantic owner | mechanism / decision authority | delivery owner |
|---|---|---|---|
| allocation | 各 Container factor | FROZEN `mem` allocator | central component/package graph |
| LRU membership | `container` | FROZEN `container_intrusive` hook + `FlatHashSet` index | Container façade；新增真实 PUBLIC edge |
| concurrent table retirement | `container_concurrent` | FROZEN `mem_reclaim` | `container_concurrent` target |
| CPU/OS facts | owning compiled kernel | FROZEN `platform` facts；每个 kernel owner 自己作唯一选择 | corresponding factor target |
| byte hash | `container` | private xxHash/wyhash/local short path | dependency lock + `container` target |
| UTF codec | `text` | private simdutf runtime dispatch | dependency lock + `text` target，Unix hidden symbols |
| string/bitset kernels | owning factor | private generated Highway TU；不依赖 `hyle::simd` register API | owning target |

冻结的最小边为：`container -> PUBLIC core,mem,container_intrusive; PRIVATE platform`；
`container_concurrent -> PUBLIC core,mem,mem_reclaim,container`；`container_queue -> PUBLIC core,mem`；
`text -> PUBLIC core,mem,container; PRIVATE platform`。Sched 的 queue 使用改为 direct private
`container_queue`；Util/IO 按其公开类型分别直连 `text`/`container`。不存在为了“多引用”而增加的依赖，
也不复制 FROZEN allocator、reclamation、hook、CPU facts 或 error 通道。

### 有限 FIX ledger

本轮产品实现只授权下列项目；新能力或新 benchmark 姿势必须另行退回 `AUDITED`：

| ID | owner | 冻结修复与验收 |
|---|---|---|
| `CONT5-F01` | all four factors/build | 建立四个真实 target、逐 header/source/test owner、Container façade 和 direct consumer edges；四个 factor 可独立 developer build，三种产品交付保持一个用户心智 |
| `CONT5-F02` | `container` | public inline x86 真实保持 SSE2 baseline，移除传播给 consumer TU 的 PUBLIC SSSE3 option；显式 higher-ISA 仅属于对应 compiled variant |
| `CONT5-F03` | `container` | 新增 bounded entry/weight `LruCache`，expected O(1) lookup/touch/erase/evict，复用 intrusive hook，无默认动态分派/无 domain policy |
| `CONT5-F04` | `container` | 清理 `SwissTable`/probe 实现入口与 stale benchmark/ABI comment；统一 associative fallible insert 与 std-shaped `emplace` 心智，不让 `try_emplace` 在 OOM 时 fatal |
| `CONT5-F05` | `container` | 将 SmallVector/RingDeque 的 class-wide throwing-move 禁令收窄到实际需要的操作，提供成熟容器同档 exception/basic-guarantee 路线且保留 trivial/nothrow fast path |
| `CONCC-F01` | `container_concurrent` | 独立 factor + flat/node concurrent set；静态证明无 mapped payload，覆盖 OOM、accessor、erase/rehash/reclamation/stress |
| `CONQ-F01` | `container_queue` | 独立 factor + try-only unbounded MPMC typed queue；明确 linearization、progress、OOM、approximate size、quiescent destruction，性能只与同合同 peer 比 |
| `TEXT-F01` | `text` | 独立 factor + typed UTF validation/error offset 与 UTF-8/16/32 caller-buffer conversion；malformed/output-too-small/empty/alias 规则完整 |
| `TEXT-F02` | `text`/delivery | Unix shared artifact 隐藏 simdutf/private provider symbols，Windows/static/source 路线不泄漏第三方类型或第二套 ABI |
| `CONT5-F06` | suite | 迁移直接消费者与 tests，补 choice-oriented guide、architecture、qualified P1 evidence、public header self-containment、source/static/shared/offline/license/symbol gate |

性能冻结为四条 primary：LRU hit/insert+evict、concurrent membership、typed handoff、UTF transcode；
每个 factor 最多一条 normal primary 和一条 holdout；这里的列举是风险池，不要求
把同一 factor 全部做成 P1。每行先签 `SAME_CONTRACT/NATURAL_ROUTE`，5% material margin；只有固定方向
material regression 且可定位机制时才允许一次产品修正。其余现有容器做 P0 复杂度、metadata、allocation、
codegen 与 correctness 回放，不复活旧 benchmark 全矩阵。
