# HYLE container 架构

> **V2 状态：`NOT_REVIEWED`。** 本页暂作既有设计输入，不代表当前功能、性能或 API 证书。
> 当前只提供 C++23 source/static/shared，Native FFI 为 `N/A`。BUILD-03 已删除旧 C schema、
> bridge、动态导出和 fixture；正文的其他验收叙事仍待 container 轮次重裁决。

container 是 Tier 0 数据结构层：提供侵入式链接、平坦 hash、连续数组、字符串/路径、驻留池和
调度器所需并发容器。侵入式链接单独形成轻量 header-only target；其余编译型能力由完整
container target 承载。性能核心自行维护；abseil、boost、TBB、rigtorp 和
`std::*` 只作为 benchmark/oracle，不进入产品依赖。

## 分层与依赖

```text
C++23 source/static/shared
          |
          +-----------------------+
          |                       |
          v                       v
C++ typed / source surface      compiled cold cores
  PodVector<T> / SmallVector /  hash_bytes, PodVector grow,
  StaticVector<T,N>
  FlatHashMap<K,V> /            Swiss resize/free,
  FlatHashSet<K>
  FlatMap<K,V> / FlatSet<K>
  BTreeMap<K,V> / BTreeSet<K>   erased ordered POD B-tree
  ConcurrentFlatHashMap /
  ConcurrentNodeHashMap
  DynamicBitSet / SparseSet    bitset bulk ISA kernels
  RingDeque<T> / ByteRingBuffer
  string_ref / StringInterner   StringInterner/path/string algorithms
  typed queues                  runtime ISA dispatcher
          |                       |
          +-----------+-----------+
                      |
                      v
Tier 0 dependencies: core cache geometry + mem allocator
                     + private platform CPU discovery
Build-only seam:     simd_vec_headers register DSL
```

- 完整 `container -> PUBLIC {core, mem}`、`PRIVATE {platform}`；platform 只服务
  编译实现的 CPU/OS dispatch。per-ISA TU 对寄存器 DSL 的借用另由 build-private
  `simd_vec_headers` target seam 登记，不扩大产品闭包。
- `container_intrusive` 是只 `PUBLIC core` 的轻量 `INTERFACE` 组件；它不拉入完整 container、mem、
  error 或 platform，完整 container 也不聚合它。Core 同样是 INTERFACE，所以这条公开源码依赖
  没有 binary artifact。
- 完整 container 对 `<hyle/core/cache.h>`/compiler contract 的使用由直接 `PUBLIC core` 边表达；
  中央图与公开头扫描会拒绝靠安装 include tree 隐藏这条依赖。
- 字符串批处理的每 ISA TU 复用 `hyle/simd/vec.h` 的 header-only 寄存器 DSL；
  当前不调用 SIMD runtime，也不会把 SIMD 公共 API 带给 container 消费者；中央 manifest 用直接
  `PRIVATE_BUILD_DEPS simd_vec_headers` 投影真实 build-interface target 边，避免共享 include root
  隐藏耦合，同时避免为 container 单独编译完整 simd runtime。
- Swiss control group 仍在 `probe.h` 自包含实现：x86 16-byte group、NEON/portable
  8-byte group。它必须可被 typed `SwissTable` hot path 内联。

## 数据结构面

| 能力 | C++23 产品面 | 关键合同 |
|---|---|---|
| intrusive list | [`IntrusiveListHook` / `IntrusiveList<T, &T::hook>`（auto）与 `SafeIntrusiveListHook` / `SafeIntrusiveList<T, &T::hook>`（explicit retirement）](container-intrusive-design.md) | caller-owned 稳定节点；零分配；node→iterator、单节点/整段 O(1) 重链；外部同步；独立 `container_intrusive` target |
| flat map | `FlatHashMap<K,V>`（实现名 `SwissTable`） | non-trivial ownership 与异构查找；ctrl+连续 slots；SIMD group probe |
| flat set | `FlatHashSet<K>` | 真 set API/slot；canonical-key find；只读 key 迭代 |
| sorted flat map / set | `FlatMap<K,V>` / `FlatSet<K>` | lookup-heavy/mutation-light；unique-key；连续有序迭代；`sorted_unique` O(N) bulk；位置 mutation O(N) |
| ordered map / set | `BTreeMap<K,V>` / `BTreeSet<K>` | unique-key；双向有序迭代；lower/upper/range visit/range erase；成功 mutation 后引用与 iterator 失效 |
| concurrent flat map | `ConcurrentFlatHashMap<K,V>`（短名 `ConcurrentHashMap`） | 独立 cache-line shard + Swiss table；shared/exclusive accessor；紧凑、高吞吐；rehash 后地址不稳定 |
| concurrent node map | `ConcurrentNodeHashMap<K,V>` | bucket RW accessor；node 地址跨 rehash 稳定；表增长保留旧 bucket array，销毁前要求 quiescent |
| POD dynamic array | `PodVector<T>` | 默认可用或注入 allocator；copy/move 值语义；allocation 只含 elements，owner 保存 allocator/size/capacity |
| inline array | `SmallVector<T,N>` | 通用 owning value semantics；POD 保留 memcpy relocation；N 是编译期类型 |
| strict fixed-capacity sequence | `StaticVector<T,N>`（`InplaceVector` alias） | storage 全内嵌、永不分配、data 地址固定；容量失败可恢复；N=0 可用；紧凑 size metadata |
| dynamic bitset | `DynamicBitSet` | allocator-aware 值类型；canonical zero padding；set algebra/shift/popcount/set-bit iteration；长 mask runtime ISA dispatch |
| bounded sparse IDs | `SparseSet<Id>` | dense+sparse 双数组；固定 universe；insert/erase/contains/clear O(1)；dense iteration O(active)；swap erase 不保序 |
| single-thread ring | `RingDeque<T>` / `ByteRingBuffer` | 前者是可增长/显式 bounded 的 owning 双端序列；后者为流式字节 staging 提供 copy 与至多两段 zero-copy spans；均非线程安全 |
| bounded SPSC FIFO | `SpscQueue<T>` | exact usable capacity；单 producer / 单 consumer；无 CAS/逐 slot atomic |
| string view | `StringView`（实现名 `string_ref`） | ptr+u32 length+NUL bit；借用、不拥有 |
| interner | `StringInterner` | generation-tagged direct ID；内容相等才同一条目 |
| path/text algorithms | typed view/buffer-fill | 借用输入与 caller-owned 输出边界明确 |
| bounded/sharded queues | typed payload | container 不拥有 payload |
| work steal/share | typed payload | owner/thief 契约不同，不能混用 |
| concurrent vector | typed append-only + random-access iterator | 多 writer 并行领取 index；地址稳定；quiesce 后读取；`size()` 不是 in-flight 完成栅栏；fallible path 分配失败不留洞 |

上表全部能力的 Native FFI 当前均为 `N/A`。

## 独立侵入式容器组件

侵入式链接由独立的 `hyle::container_intrusive` header-only target 承载，完整
`hyle::container` 不聚合它。它不拥有节点、不分配、不保存 owner/count，也不提供
内部同步；time 的稳定 timer node 是 safe-link 真实消费者，sched 的 wake-link 状态复用则是保留
专用结构的反证。

Auto/safe member hook、`iterator_to`、range splice、生命周期/iterator 失效矩阵、复杂度、明确
负空间和 Native FFI=`N/A` 的长期合同统一见
[`container_intrusive` 架构](container-intrusive-design.md)。本页不复制独立组件的公开 API 或当前
状态；是否已满足工程门只以 [`module-status.json`](../implementation/module-status.json) 为准。

### 成熟库能力对账：仍不冒充“容器全集”

实现行数只能反映维护成本，不能证明能力完整。container 已包含
多套非平凡并发算法，但同本地 Abseil、Folly、oneTBB、Boost 的公开能力轴对账后，覆盖面
仍明显偏窄。现状与真实场景如下；“缺”不是立即复制一个类型，而是收口前必须设计或明确
负空间：

| 数据结构轴 | 当前 | 常用场景与判断 |
|---|---|---|
| 连续 sequence / inline sequence | `PodVector`、`SmallVector`、`StaticVector` | 分开覆盖 POD dynamic、可 spill 小 owning sequence 与严格 no-allocation fixed-capacity；普通非平凡长序列继续用 `std::vector` |
| flat associative | `FlatHashMap/Set` | 已覆盖缓存友好的通用 lookup/membership |
| ordered/range associative | `FlatMap/Set` + `BTreeMap/Set` | 前者覆盖批量装载后的 read-mostly 连续索引，后者覆盖持续 mutation、时间/调度索引；均为 unique-key |
| node/reference-stable associative | `ConcurrentNodeHashMap` 覆盖并发 registry | accessor 生命周期内引用稳定，node 地址跨 grow 稳定；单线程 node extraction/merge 仍不复制 std API |
| concurrent associative | `ConcurrentFlatHashMap` + `ConcurrentNodeHashMap` | compact/rebuild 与 stable-reference/read-heavy 两种常用形状分开，不靠 policy flag 隐藏不同成本 |
| 单线程 ring/deque | `RingDeque<T>` / `ByteRingBuffer` | 流式窗口、音视频/网络 staging 与外部同步 ring 已覆盖；不让并发原子合同向单线程用户征税 |
| 单 producer / 单 consumer FIFO | `SpscQueue<T>` | 独立于 MPMC/stealing：一个 acquire/release index pair，无 CAS/逐 slot atomic；覆盖线程/流水线 stage 间的常见 bounded handoff |
| dynamic bitset / sparse set | `DynamicBitSet` / `SparseSet<Id>` | graph frontier、mask/set algebra 与稠密 ID active set 已覆盖；前者长 mask SIMD，后者以 O(1) clear + O(active) iteration 换取 2*U*Id bytes |
| priority queue / heap owner | 由标准库覆盖 | 单线程 comparator heap 先用 `std::priority_queue`；只有并发 priority 或 decrease-key 真实需求出现才新增形状 |
| cache/LRU policy | 不在 container | eviction 是策略，先由 util/消费者组合；不能因 Folly 有同名类型就把策略塞进 Tier 0 |
| blocking/unbounded queue | sched 负责等待/背压 | container 保持非阻塞机制，避免另造一套 shutdown/cancel/wait 语义 |

因此当前结论不是“行数已经等价主流全集”，也不是按对手 LOC 补代码。dynamic bitset/
sparse set、单线程 ring、ordered/range 与 concurrent associative 已通过 C++ 正常入口、
复杂度与实际 graph/active-ID/stream-staging/time-index/shared-registry workload。
明确负空间仍是 single-thread node extraction/merge 与 ordered multi；候选能力轴还包括
带完整策略的并发 priority/unbounded
机制。它们都需要真实消费者触发，不能用薄 wrapper 追成熟库类型数量。

### StaticVector 的严格容量形状

`StaticVector<T,N>` 补的是 `SmallVector` 无法表达的协议/实时合同，不是另一个小 N push
benchmark fixture。设计对照本地 CCCL `cuda::std::inplace_vector` 的 capacity-sized
metadata 与成熟 static-vector 共同面：一个紧凑整数记录 size，T-aligned storage
直接位于对象内；inactive union 的真实 `T[N]` 成员保留合法 array pointer 边界，逐项
placement construction 启停 lifetime，空 data 不对尚不存在的 T 使用 `launder`。没有
allocator、capacity pointer 或 heap fallback。`N==0` 返回空/null
range；raw storage 不做按 capacity 清零，故默认构造保持 O(1)，而不是随 N 扫描整块对象。
generic code 不需要为零上限另造 wrapper。`InplaceVector` 只是 alias，避免两套
实现和两个语义相近的公开类型漂移。

普通入口容量越界 fail-fast；所有 `try_*` 先验证容量，在 false 路径不构造参数、不移动
现有元素。input/self range 的 try assign/insert 先在第二个 fixed-capacity 临时对象中单遍
materialize，才能同时满足 single-pass、alias-safe 和“失败不改原值”；这是冷 bulk mutation
的明确成本，不向常态元素增加 metadata。没有 class-wide movable 约束：不可移动 T 仍可
原位 emplace_back/pop/clear；只有必须 relocation 的位置 insert/erase 要求 nothrow move
construction。整容器 throwing move 提供 basic guarantee，构造失败会销毁完成的目标前缀。
非平凡位置 mutation 不额外要求 move assignment；POD copy/move 和 suffix relocation 分别走
`memcpy`/`memmove`。push/pop O(1)，insert/erase O(suffix+inserted)，已经是连续 storage 的
下界。data 本身永不迁移；位置 mutation 前的地址保持，从位置起的元素 identity 视为失效。

N、T 构造/析构和对象内布局是 C++ source contract；它没有 runtime 类型擦除产品面。

### 排序连续关联容器的实际使用形状

`FlatMap/Set` 对照 Boost.Container 与 C++23 flat associative 的共同形状：unique key、
binary search 点查，不为 node、rank/finger 或 benchmark 保存常态 metadata。`FlatMap`
采用成熟的 split storage（连续 keys + 连续 values），只查 key 或只扫 mapped value 时不会
把另一列拖进 cache；`FlatSet` 使用独立的单数组 core，不为 map proxy/values 支付结构税。
异构 lookup 只要求 comparator 能对 K/Q 双向比较；comparator 抛出的异常会在 mutation
前传播，不会留下半排序状态。

两列都复用已审查的 `SmallVector<...,0>` allocator-aware sequence engine；空 map/set 不按
`sizeof(K/V)` 嵌入元素 storage，首次插入才分配，后续增量插入按几何容量增长。这里的
zero-inline 是 storage policy，不是另一个 vector，也没有为 singleton benchmark 保存特殊
layout。map iterator 是 random-access proxy，提供标准 `.first/.second`，
同时保留 `key()/value()` 便利访问；key 只读、value 可写。`keys()` 与 `values()` 暴露各自
真正连续的 span，但 map 的 `value_type` 逻辑上仍是 `pair<K,V>`，不能把 iterator 当成
contiguous pair pointer。已知大表可先 reserve，分配与释放仍走 mem 的 sized-free/alignment
合同。

普通 lookup O(log N)，iteration O(N)，位置 insert/erase O(N)。`sorted_unique` bulk
construction/assign 信任调用方“严格有序、无等价 key”的显式前提，预留一次后单遍构造，
所以为 O(N)；库不再做一遍 comparator validation。allocation-recoverable mutation 使用
`FlatInsertResult` / `try_reserve` / `try_assign`，standard-shaped `try_emplace` 只承诺命中
不构造 mapped value。成功位置 mutation 从该点起失效，reallocation 全失效。

bulk 不经公开的逐项 `try_emplace_back`：窄 internal access seam 让 fresh key/value vectors
对 sized range O(1) 取长度、一次 reserve、在未初始化尾部原位构造，整批完成后一次发布
size；任一列异常/OOM 都销毁已完成前缀。这个 seam 不进入 SmallVector 公共 API，也不允许
用户任意修改 size/uninitialized storage。map 与 set 的 binary-search core 分开，避免让
直接存 K 的 set 缴 map proxy/projection 税。

这里不增加 runtime 类型擦除 handle：它会复制动态能力，却仍无法表达任意 C++ K/V
构造析构。typed shape 保持 source surface，内部 allocator 继续遵守 mem 契约。

### 有序 B-tree 的实际使用形状

`BTreeMap/Set` 服务 flat hash 无法表达的 lower/upper bound、确定性遍历、时间窗口与范围
淘汰。设计对照本地 Abseil B-tree 的 cache-packed family shape：entry 同时位于 leaf 和
internal node，separator 不重复存 key；leaf 目标约 256 bytes。普通 interior mutation
保持密集节点；单调 append/prepend 的满页分裂按插入位置偏置，避免每次只留下半页；连续
边界删除允许最外侧非空页短暂低于半满，避免把马上要过期的 entry 从兄弟页搬来再删除。
空页仍立即 merge/unlink，树高和单项操作仍为 O(log N)。实现不为单条测项保存 subtree
size、rank、finger 或预取 metadata。短数值 key 在 node 内线性搜索；自定义 comparator
走二分。该选择由小节点 cache locality 决定，不向用户公开节点容量或搜索 policy。

单项 find/insert/erase 为 O(log N)，有序遍历为 O(N)。copy 递归克隆现有树形，O(N) 且
不调用 comparator；它不是逐项重新 insert 的 O(N log N)。iterator range erase 对 leaf 中
连续 run 做一次后缀搬移；完全落在删除范围内的 page 可直接卸下，避免先 merge 整页再逐项
销毁。满页先尝试把 entry 向有余量的兄弟页平衡；边界写入会填满旧页、把新页留给后续有序
数据。分裂前会预留必要 sibling/root，OOM 时结构保持不变。trivially-copyable entry 使用
`memmove`，非平凡 entry 执行 nothrow move construction/destruction；两条路径共享同一公开
合同，不是 benchmark 特化。

成功 mutation 可能搬移一个 node 内的 slot，因而统一失效既有 iterator/reference/pointer。
这是 cache-packed storage 的必要代价；需要 grow 后稳定地址的用户选 node-based 结构。
map iterator 保持 key 只读、value 可写，set iterator 为 `const K&`。公开面只做 unique-key；
multimap/multiset、node handle、extract/merge、stable reference 与 rank/select 都是明确负空间。

### 两种并发关联容器的实际使用形状

并发 map 没有一个同时最优的 storage 形状。`ConcurrentFlatHashMap` 以 power-of-two
shard 把 key 分流到独立 cache-line-aligned RW lock 与 Swiss table：适合 memo、cache、
计数/状态表和 clear 后并行重建。单 key 只 hash 一次，先用高位选 shard，再把同一个
hash 送入 Swiss H1/H2 probe；不会因长 key 在 wrapper 和 table 各扫一次。查找/插入/删除
expected O(1)，`size/reserve/clear/for_each` 为 O(shards) 加实际表工作。accessor 持有整个
shard 的读锁或写锁，因此其引用不会在 accessor 活着时被 erase/rehash，但释放后没有
地址稳定承诺。

`ConcurrentNodeHashMap` 面向 read-heavy registry、需要较长 accessor 或 grow 后对象 identity
稳定的场景。每个 bucket 有独立 RW lock，node 单独由注入 allocator 分配；rehash 只 relink
node，不移动 key/value。普通查找/插入/删除 expected O(1)，`size()` 是 O(1) atomic，rehash
为 O(N+B)。增长是冷 stop-the-table 阶段：按固定顺序排空旧 bucket locks、发布新 table，
旧 bucket arrays 退休到容器销毁；几何增长使退休 bucket 总量小于当前 bucket 数，换取 hot
lookup 不写全局 hazard/refcount。已存在 key 的 insert/assign 在判断命中后直接返回，不会
仅因装载率触发无意义 rehash。

两者共享一个 compact、writer-preferring adaptive RW lock。状态布局参考 oneTBB
`spin_rw_mutex`，短竞争 bounded spin，flat rehash 等长等待则用 C++20 `atomic::wait` park。
reader-waiting hint 只在真 waiter 存在时 notify，避免 libstdc++ shared waiter pool 把互不
相关的 shard 串成全局热点。这是正常并发使用的机制，不是 benchmark-only metadata。

accessor 是并发安全合同本身：`ConstAccessor` 持共享锁，`Accessor` 持独占锁；复用同一
accessor 前 find/insert 会先 release 旧绑定。每线程对同一 map 最多保留一个 accessor；
flat 调用 `size/reserve/clear/for_each`、node 调用 `reserve/clear/for_each` 前也必须释放它，
callback 不得 re-enter 同一 map。Hash/Eq/allocator 必须能被
多线程安全调用，析构必须等待所有调用和 accessor quiescent。

### RingDeque / ByteRingBuffer 的实际使用形状

`RingDeque<T>` 对照 Folly `DynamicRingQueue` / `FixedCapacityRingQueue` 的单调无符号
head/tail 与 power-of-two storage 形状，同时补上基础库用户真正需要的 owning 值语义、
双端 mutation、随机访问 iterator 和逻辑顺序的至多两段 readable spans。普通
`try_emplace/push_*` 满时增长并报告 OOM；显式 `try_*_bounded` 在 reserve 后永不分配，
满就是 backpressure。两者是同一个清楚的容器合同，而不是为 benchmark 复制一个简化
queue。元素要求 nothrow move construction/destruction，使增长失败能保持原状态。

head/tail 按成熟 ring 形状单调推进；无符号差值给出 size，power-of-two mask 给出物理
slot，hot pop 不在每次变空时重置两个游标。后者曾令 GCC 常态 FIFO 多一条分支和双写；
删除冗余后恢复到成熟 portable ring 的成本档，而不是增加常态 metadata 或专用 benchmark API。
push/pop 两端与随机访问均为 O(1)，增长为 O(N)，已达到该 storage 形状的渐进下界；具体
平台点值只见 performance 文档。

`ByteRingBuffer` 不是 `RingDeque<std::byte>` 的别名：它面向 socket/file/audio staging，
以 partial `write/read` 提供方便 copy 路径，并以 `writable/readable` 返回至多两段 span，
配合 `commit_write/consume` 直接接 scatter/gather I/O。reserve 会线性化并保持字节序；
普通搬运只做至多两次 memcpy，没有逐 byte 元素生命周期税。两个类型都只负责单线程
机制；跨线程使用必须由调用方外部同步，真正 SPSC/MPMC/owner-thief 选择专用并发队列。

### SpscQueue 的实际使用形状

`SpscQueue<T>` 对照 Folly `ProducerConsumerQueue` 与成熟 bounded SPSC ring 的共同形状：
producer 独占 write index，consumer 独占 read index，两个发布原子分离到 destructive
interference cache line。每个成功操作只需读取对端 index、访问一个 slot，再 release 自己的
index；不支付 MPMC 的 CAS、sequence number 或逐 slot atomic。requested capacity 是 exact
usable capacity，内部只多一个 sentinel slot 来区分满/空，所以任意正数容量都不会被偷偷
取整。push/pop 为 O(1)，构造时唯一一次 allocation 为 O(capacity)。

typed 面在 producer 发布前原位构造 `T`，consumer 释放 slot 前销毁它，因此支持非平凡与
move-only 值；不可 move-assign 的值通过 callback 消费。producer/consumer 角色各自只能有一个
线程，析构前必须 quiesce；`size_approx` 只用于 telemetry，不是跨线程完成栅栏。
这是一种常用并发架构能力，不是为 benchmark 保存的旁路或常态 metadata。

### DynamicBitSet / SparseSet 的实际使用形状

`DynamicBitSet` 对照 Boost dynamic_bitset 的常用面与 Folly 内部 frontier bitset 形状，保留
runtime size、resize/reserve、单 bit mutation、bulk set/reset/flip、AND/OR/XOR/difference、
shift、count/any/all、find-first/find-next 和 set-bit range。没有照抄 string conversion、
stream operator 等非基础加速职责。storage 是 allocator-owned `uint64_t` blocks；最后一块
的非逻辑 bit 永远清零，所有 bulk kernel 因此不需要额外 mask 每个 lane，只在 mutation
结束 sanitize 一次。大块集合运算通过 platform `best_isa()` 选独立 variant TU，container
通过 `PRIVATE platform` 与 build-private `simd_vec_headers` 登记实现依赖；当前只使用
`hyle/simd/vec.h` 的编译期 DSL，
不调用 `hyle_simd` 的 runtime 函数，也不把 SIMD 类型暴露给 container 消费者。

`SparseSet<Id>` 采用 Folly `SparseByteSet` 与经典 dense+sparse set 的共同算法：sparse[id]
记录 dense position，membership 再验证 `dense[position] == id`，因此 `clear()` 只需 size=0，
旧 sparse 值不会产生 false positive；erase 以 dense 尾项回填并修复其 sparse position。
HYLE 的通用 allocator-backed 版本在 universe 建立时初始化 sparse 数组，不读取 C++
indeterminate storage；这比 Folly 的 byte-only 栈上特化多一次冷 O(U) 初始化，但避免依赖
MSan suppression/未初始化 byte 语义。两个数组在 reset 时一次性预留完整 universe，换取
hot insert 不分配且为严格 O(1)，而不是 amortized O(1)。

两个类型不能互相替代：DynamicBitSet 为 U/8 内存并擅长全 universe 集合代数；SparseSet
约为 `2*U*sizeof(Id)`，但 clear 是 O(1)、iteration 是 O(active) 且直接给出 ID。纯
membership 的 dense byte array 只需一次 indexed load，预期快于 SparseSet 的反查；性能
证书必须保留这个 lose，不能为 benchmark 给 SparseSet 增加另一条 bitmap 旁路。

### SmallVector 的成熟形状门

SmallVector 的接口不再由“小 N push benchmark”定义。本轮同时对照本地
Abseil `InlinedVector`、Folly `small_vector` 与 Boost `small_vector`：三者共同证明
inline storage 只是存储策略，用户仍需要普通 owning sequence 的值语义。HYLE 当前
共同面包括 copy/move、count/range/initializer-list construction、连续/反向迭代、
reserve、resize、任意位置 emplace/count/range insert、assign、单项/范围 erase、clear、
swap、shrink-to-fit 和值比较；自别名参数即使触发 inline→heap 迁移也安全。
位置插入和删除只搬移受影响后缀，复杂度为 O(size-position+inserted-count)，已达到
连续存储容器的渐进下界；trivially-copyable T 使用 memmove，其他 T 只要求 nothrow
move construction，不额外要求 move assignment。forward range 在写入前一次 reserve；
single-pass input range 只消费一次。range/count insert 先建立 replacement sequence，
换取自别名和抛异常时不破坏原容器的明确合同；它不在逐项插入时反复移动原后缀。

`try_shrink_to_fit()` 能把短序列从 heap 收回 inline storage，也能报告 heap 精确收缩的
OOM 且保持原状态；普通 `shrink_to_fit()` 沿用本容器无错误返回入口的 fatal OOM 合同。
`N==0` 是合法的 zero-inline storage policy：空对象不携带 dummy `T`，首次插入才分配；
它主要供需要同一成熟 sequence engine、但不应把任意 `T` 嵌入 owner 的组合容器复用。
`at()` 没有机械照抄：HYLE 构建可关闭异常，必须先决定是 Status/nullable 访问还是
fatal checked access，不能用一个异常形状破坏全库合同。新增普通序列接口没有借旧
push/read microbenchmark 数字宣称新的性能结论。

### PodVector 的成熟形状门

旧 `CArray<T>` 把 clay 实现名和窄 benchmark 形状一起暴露给了用户。当前公开主类型改为
`PodVector<T>`，并按 `std::vector` 与 Folly `fbvector` 的共同连续序列习惯提供默认构造、
count/range/initializer-list 构造、copy/move、front/back、正反迭代、reserve、push_back、
resize、count/range insert、assign、单项/范围 erase、clear、swap、shrink-to-fit 与值比较；
`try_reserve/try_push_back/try_resize/try_shrink_to_fit` 保留 HYLE allocator 的可恢复 OOM
通路。它刻意只接受 trivially-copyable T：这是明确的 byte-relocation 合同，
不是第二个通用 `std::vector`。非平凡值用 SmallVector 或标准 vector。byte relocation
不额外要求 T 的 C++ copy constructor；增长前的别名值以对齐 byte snapshot 保存。

allocation 只保存 elements；allocator、size 与 capacity 只由 typed owner 保存，不在每块
allocation 前重复一份 C-era header，也不要求 `data()` 写回 shadow size。type-erased
reallocate/free core 仍只编译一次，indexed read 仍是单次寻址，增长仍可利用 backing realloc。
contiguous range 直接批量 memcpy；single-pass input range 只消费一次。range insert 先
物化输入，再以一次 memmove 搬后缀和一次 memcpy 写入；复杂度
O(size-position+inserted-count)，达到连续存储下界。缩短 resize 为 O(1)，不会构造无用
fill value；shrink O(size)，失败保留原 block。旧 growth/read 点值不能替新增 range/mutation API
宣称 SOTA；若真实消费者把位置 mutation 变成热点再按同合同单独建证书，具体数字只进
performance 文档。

## SwissTable

SwissTable 不是 clay 线性探测 map 的换名版，而是完整的 flat Swiss 形状：

```text
ctrl:  [H2][H2][EMPTY][DELETED]...
slots: [K,V][K,V][-----][-----]...

hash -> H1 起始 group + H2 七位指纹
     -> SIMD 一次比较整个 ctrl group
     -> 只对候选 lane 执行 key equality
     -> triangular group probing
```

erase 在安全时把 tombstone 恢复成 empty，否则保留 deleted。resize/free 是
type-erased cold core；typed find/insert/erase 保持 header-inline，避免廉价 key
每 probe 一次函数指针。C map/set 使用同一 ctrl/group/resize 机制，slot layout
分别在运行时计算为 `[stored hash][aligned key][aligned value]` 与
`[stored hash][aligned key]`；不是每元素分配的链表，也不让 set 用户用 map +
dummy value 模拟 membership。
`FlatHashSet` 共享这一 probing/resize core，但 empty mapped policy 不占 slot 空间；
map/set iterator 都用 control-group full mask 跳转 occupied lane。map entry 的 key
只读、value 可写，set iterator 只读 key，因此不会经 iterator 破坏 hash 不变量。
map 的 mutation 语义按成熟容器拆开：`insert` 不覆盖，`try_emplace` 命中不构造
mapped value，`insert_or_assign` 才覆盖，`operator[]` 只为 default-constructible value
提供熟悉入口；它们都复用单 hash、单 fused probe 的热路径。

Owning 容器语义也不再因早期 benchmark 壳而残缺：copy 在 K/V 可复制时启用，copy
assignment 保留目标 allocator，range/initializer-list construction 与 insert、swap 和
内容相等均可直接使用；move-only map 仍在类型层面不可复制。forward range 先按输入
上界一次 reserve，避免逐级整表 rehash；single-pass input range 只消费一次。copy/range
构造若元素复制抛异常会销毁已完成 slots 并释放 ctrl/storage。相等比较是 expected O(N)
次 hash lookup，不依赖未指定的 slot/迭代顺序。这些冷/批量能力没有改变单 key fused
probe 的指令路径。

边界原则：

- 所有容量乘法、对齐和 power-of-two 扩张先做 overflow 检查。
- 增长失败保留旧状态并返回失败。
- insert 输入若别名当前 slots，增长前复制到 allocator-backed scratch。
- C callback 不得抛异常或在遍历中修改同一 map。

## StringInterner 身份与生命周期

StringInterner 有两个正交用法：

```text
acquire/acquire_many -> StringId(generation, shard, slot) -> retain/release -> refs=0 时释放
intern / intern_many                -> permanent until interner destruction
```

内容 hash 只负责定位，不是身份。`by_string` 继续做完整内容 equality；ID 编码
`generation + shard + slot`，generation 防止 slot 重用后 stale ID 命中新字符串。
构造时按硬件并行度一次选择 16/32/64 shard，单次调用只锁内容或 ID 对应的 shard。
shard 取 hash 中间位，SwissTable 仍直接使用原 hash 的低位 H1 与高位 H2，避免
固定 index/tag 熵或每次 lookup 再做一次 remix。永久 `intern` 默认不创建外部不可见
的 ID slot；第一次 `acquire` 同内容时才 lazy materialize。

Entry 缓存 table hash：新输入不可避免地是 Ω(L) 读字节，但 Swiss 扩容只移动
16-byte `{Entry*, hash}` 索引槽、删除按缓存 hash 定位，不再反复扫描已驻留长串。索引
不再重复保存 `string_ref + Entry*`；字符地址恒为 `Entry+1`，Entry 本身也不保存冗余 data
pointer。≤256B 的 Entry+字符使用
每 shard 小块 slab，release 后回到池内复用；大字符串立即归还注入 allocator，全部
slab 在 StringInterner 销毁时归还。地址在有效生命周期内稳定。

每个 shard 的短临界区使用 `AdaptiveMutex`：无竞争是 `atomic_flag` fast path，有限自旋后
用 `atomic::wait` park，waiter count 避免无竞争 unlock 的通知和 lost wakeup；表增长仍由
冷 `growth_mu` 串行化 allocator copy。它不是无界 spin lock，分配/rehash 阻塞时不会持续
烧核。

SwissTable 的 `reserve(final-size)` 会同时处理正常 grow 与 tombstone 耗尽的 growth
budget：容量足够时做同容量 rehash，而不是虚假返回成功。`try_prepare_insert()` 由此
合同推导下一项所需 final-size，先把可能失败的 allocation 做完，随后 `insert()` 才是
allocation-free；StringInterner 因而能在 noexcept/batch API 中返回短前缀，而不是让冷路径
OOM 落进 fatal assert。

| 操作 | 当前预期复杂度 | 下界/相对旧形状 |
|---|---:|---|
| 新内容 `intern/acquire` | O(L) amortized | 必须读输入，达到 Ω(L) 下界 |
| 已驻留内容命中 | O(L) expected | 必须确认调用方字节；hash 只算一次 |
| `lookup/retain/release(id)` | O(1) expected | direct generation slot，达到 Ω(1) 下界 |
| table rehash | O(N) | 只搬定宽 slot；旧形状会重扫 resident bytes，最坏 O(ΣL) |
| 最后一份 ID 删除 | O(1) expected | cached hash + pointer-equality erase，不重扫字符串 |
| batch | O(ΣL + N + U) expected | 每块 `U` 为实际涉及的 shard 数，`U <= min(N,64)` |
| `size()` | O(S)，硬上界 O(1) | 不扫描条目 |
| 短串重复 churn | O(N) | 达到高水位后 slab/ID slot 复用，无逐条 allocator 调用 |

吞吐/生命周期用户应使用 `intern_many`、`acquire_many`、`retain_many`、
`release_many`。每个有界块先算一次 hash，只按升序获取实际涉及的 shard lock；
既避免全池冻结也避免重复 hash。性能结论见
[`benchmarks-container.md`](../implementation/evidence/performance/benchmarks-container.md)。

## 字符串 SIMD 策略

动态调用链如下：

```text
string_ref/string_algo public call
          |
          v
baseline dispatcher (cached platform::best_isa())
          |
   +------+------+------+-------+
   | scalar | SSE4 | AVX2 | AVX512 | NEON
   +------+------+------+-------+
        independently compiled target TUs
```

`count`、`rfind(char)`、ASCII lower/upper 走上述变体。调用方不需要 CPU 宏，也不会
在旧 CPU 执行宽指令。substring 与 character-set 搜索在 `.cc` 内转发
`std::string_view`，因为当前 MSVC 标准库实现实测更强；公开头不会因此暴露 STL。
短路径组件继续用内联标量反向扫描，避免 dispatcher seam。

UTF-8 validate/count 使用内化的 simdutf source，并由 simdutf 自己派发；validate-only
是一遍常用路径，count 合同在合法输入上明确为 validate + count 两遍。这不是
container 对外依赖。hex encode/decode 当前是无查表分支的标量/编译器向量化实现，
只证明胜过等契约 scalar reference，未宣称战胜专门 SIMD codec。

SIMD 只进入可批量处理的字节/lane 工作，不按模块名强套：

| 路径 | 当前向量化 | 决策 |
|---|---|---|
| Swiss control-byte probe | x86 SSE2 16 lanes；AArch64 NEON 8 lanes | 已是成熟 Swiss 基线形状；每次 probe 再动态派发 AVX2 的固定税通常不值 |
| count/rfind/ASCII case | SSE4/AVX2/AVX512/NEON runtime dispatch | 已进入产品；长串摊薄 dispatcher，短串走标量 |
| UTF-8 validate/count | simdutf 自派发 | 一遍 validate-only 与严格 validate+count 分开；借成熟核，不重写 |
| table hash | 17–128B inline multiply stream；x86 AVX2 / AArch64 NEON XXH3；baseline wyhash | varied-input 产品 API 证据后进入产品；Windows/M4 从 1KiB、Linux/GCC14 从 512B，CPU/OS gate 在 platform |
| vector relocation | compiler/libc `memcpy/memmove` | 不重复实现平台已有宽拷贝与 non-temporal 策略 |
| lock-free queue / atomics | 不做 | 瓶颈是依赖链、cache line ownership 和 CAS，不是 lane arithmetic |
| DynamicBitSet | SSE4/AVX2/AVX512/NEON runtime dispatch | 位运算与 x86 count 复用 header-only vec DSL；NEON count 用 byte CNT+四路 UDOT，避免通用 u64-lane reduction 的已测回退；小于 8/16 words 分别走本地 scalar |
| BTree node search | 数值 key 线性比较；通用 comparator 二分 | 典型 node 很小且 comparator/entry layout 泛型；当前三平台实际 workload 不支持为每个 node 增加 SIMD metadata 或动态派发税 |

长 hash 的 x86 宽核放在独立 AVX2 TU，baseline dispatcher 只有在 CPUID+XGETBV 安全门
通过后调用；AArch64 使用 baseline NEON XXH3。varied-input 产品 sweep 分别给出
Windows/MSVC 与 M4/AppleClang 1KiB、Linux/GCC14 512B 的保守交叉点。
`hash_bytes` 是进程内 table hash，值可随
HYLE 版本、构建与 ISA 选择变化，不得持久化或作为 wire fingerprint。若用户需要稳定内容
指纹，应由独立命名/独立合同提供，不能暗中冻结 table hasher 的当前 bit pattern。
默认 `StringHash` / `Hash<string/view>` 在 policy 构造时缓存一个进程 seed，并把它作为输入
注入 short/medium/wyhash/XXH3 core；只在结果末端 XOR 或 mix 无法拆开 raw core 的完整
碰撞类，因此不采用。所有默认 policy 共享进程 seed；每个 hasher/table policy 缓存一个
`uint64_t`，不增加 per-entry metadata，也避免每次 lookup 读取进程全局状态。这与 Abseil 的
per-table/process salt 目标相同但粒度更粗，只保证跨独立启动破坏离线
预计算集合，不保证密码学不可预测或抗可观察的自适应 hash flood。进程首次使用后 `fork`
会继承 seed，`exec` 才重新派生；显式 `StringHash(seed)` 保留确定性复现。

不可信字符串另有显式 `HardenedStringHash`：SipHash-2-4 持有调用方注入的 128-bit secret，
且无默认构造。HYLE 的 ASLR/clock process seed 不
冒充 CSPRNG，也不自动生成 hardened key；应用必须从自己的安全随机源/secret manager 获取、
保密并按隔离边界轮换。hardened policy 仍只在调用期间借 view，与 `StringEqual` 保持同一
owning/heterogeneous 等价关系。正常短串和整表 hit/miss workload 分开记录 fast/salted 与
keyed SipHash 成本，不能用 raw byte microbench 隐藏安全策略代价。
默认 typed `Hash<T>` 还必须满足 `Eq(a,b) => Hash(a)==Hash(b)`：整数/枚举/指针走语义明确
的 mix，浮点使用 equality-compatible value hash 并归一 signed zero；所有未特化的用户
自定义类型一律不提供默认 Hash。`BytewiseHash<T>` 只允许调用方显式 opt-in，并要求
`has_unique_object_representations`，但这个 trait 仍无法证明自定义 Eq 比较所有 byte。
因此带 padding、alternate encoding 或业务 equality 的 struct 必须提供逐字段 hasher，不能
把“trivially-copyable/unique representation”误当成“对象表示就是相等语义”。负向编译
测试同时锁住 padded key，以及 padding-free 但 Eq 忽略字段的 key；整表测试锁住
`+0.0/-0.0` 查找与去重。

标准 owning string 是单独审定的内建语义类型：`Hash<std::string>` 覆盖同型默认查找；
公开透明 `StringHash` / `StringEqual` 统一 `std::string`、`string_view`、`string_ref` 和
C literal 的即时 byte view，使 owning key 的异构 find/contains/erase 不要求消费者
复制测试私有 boilerplate，也不保存借用指针。

## 交付与 Native FFI 边界

container 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 C-syntax 容器 schema、
runtime-erased bridge、动态导出、symbol manifest 和 C/managed fixture；Native FFI 为 `N/A`，
这些旧函数与布局没有兼容义务。HYLE 模块和同工具链用户直接使用 typed API。

未来只有已命名的 foreign-runtime consumer 通过全局
[Native FFI 准入策略](native-ffi-policy.md)后，才按该任务另行设计窄的 runtime 类型擦除边界；
不会因为 container 模板很多，就预先恢复 erased vector/map/queue parity surface。

旧 C surface 的逐项裁决与命名演进只保留在
[`container-module-closure.md`](../implementation/tasks/container/container-module-closure.md) 作为历史证据，
不再混入当前架构合同，也不能作为 v2 证书。

## 下游利用

- sched 使用 ShardedQueue、WsFifoQueue、WorkStealingDeque 等调度原语。
- graph/sched 使用 ConcurrentVector 类 append-only 发布结构。
- time 直接通过 `hyle::container_intrusive` 使用 safe-link 原语，不经完整 container 取得该能力；
  sched 因 detached wake-link 复用保留专用链，裁决见
  [`container-intrusive-design`](container-intrusive-design.md)。
- 其他模块可使用 `hash_bytes`，但 hash 模块不依赖 `hyle_simd` runtime；≤16B typed key
  走 inline mix，17–128B 使用公开头内的紧凑 multiply stream，较长字节串走库内
  wyhash 与 platform-gated 宽 XXH3。公开头不暴露 vendor 实现头；中等输入内联是产品
  C++ API 的固定形状，不是 source-only benchmark 旁路。

## 性能验收边界

“SOTA”必须逐契约、逐操作成立，不能写成整个 container 的无条件标签：

- flat Swiss map/set 分开验收 hit、miss、insert、erase 与 iteration，不以其中一项替代全表；
- ordered B-tree 分开验收点查、范围、确定性遍历、build/copy 与 mutation，不为某个 miss 分布
  增加常态元数据或 benchmark API；
- PodVector、SmallVector、bitset、sparse set 与各类队列只和相同 ownership/capacity/concurrency
  合同的 peer 比较；
- 独立 `container_intrusive` 的性能合同与真实消费者验收不由完整 container 的结果代签，见
  [`container-intrusive-design`](container-intrusive-design.md)；
- path 的零分配结构优势不能外推为同形 string-view 扫描优势；
- StringInterner 的逐条、batch、长串增长、可回收生命周期和并发分片必须分别验收；高吞吐
  导入优先 batch，但不能用 batch 隐藏逐条入口的真实代价；
- 字符扫描必须同时看多种长度、执行顺序与方差，不能用单一长度包装输赢。

具体点值、平台翻转、对手 pin 和 freshness 只由
[`benchmarks-container`](../implementation/evidence/performance/benchmarks-container.md) 维护；本架构页只固定比较合同与
不允许为 benchmark 扭曲产品形状的原则。

测试矩阵包括 typed 单测、allocator 故障注入、三平台构建/运行、source/static/shared C++23
consumer、ASan/UBSan/TSan，以及独立 SOTA benchmark。

## 维护红线

1. 新分配路径必须保存并使用原 allocator/alignment，且有 OOM 状态保持测试。
2. 不得在普通 container 改动中恢复旧 C schema；Native FFI 只能由已准入的独立消费任务建立。
3. 修改共享 per-ISA header 后，所有 variant TU 都必须重编译；Windows 还要验证
   import library，而不只运行库内部测试。
4. 不为复用 `hyle_simd` runtime 制造 Tier 倒挂；批量 kernel 可复用 header DSL，
   但 dispatcher 属 container 自己。
5. 任何性能结论注明对手合同、平台、输入形状和不成立的边界。

## 参考

- [abseil Swiss Tables design](https://abseil.io/about/design/swisstables)
- [SwissTable control-byte overview](https://faultlore.com/blah/hashbrown-tldr)
- 本地 `abseil-cpp/absl/container/internal/btree.h`（cache-packed B-tree 对照）
- [Native FFI 准入策略](native-ffi-policy.md)
