# HYLE container 架构

container 是 Tier 0 数据结构层：提供平坦 hash、连续数组、字符串/路径、驻留池和
调度器所需并发容器。其性能核心自行维护；abseil、boost、TBB、rigtorp 和
`std::*` 只作为 benchmark/oracle，不进入产品依赖。

## 分层与依赖

```text
稳定发布面
  include/hyle/c/container.h
    POD + pointer/count + buffer-fill + opaque handles
                      |
                      v
C ABI 类型擦除层
  src/container/c_container.cc + c_btree.cc
    vector/map erased metadata, allocator-preserving handles,
    pointer-payload concurrent facades, shared hyle_status cold failures
                      |
          +-----------+-----------+
          |                       |
          v                       v
C++ typed / source surface      compiled cold cores
  PodVector<T> / SmallVector    hash_bytes, carray grow,
  FlatHashMap<K,V> /            Swiss resize/free,
  FlatHashSet<K>
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
Tier 0 dependencies: core cache geometry + error + mem allocator +
                     platform CPU discovery
```

- `container -> {error, mem, platform}` 是允许的基础依赖；error 是 C ABI 冷失败
  的直接合同，不再藏在 mem 的传递依赖或模块私有 TLS 中。
- container 不链接 `hyle_simd` 模块，避免 Tier 0→Tier 1 倒挂。
- 字符串批处理的每 ISA TU 复用 `hyle/simd/vec.h` 的 header-only 寄存器 DSL；
  这不会调用 SIMD runtime，也不会把 SIMD 公共 ABI带入 container。
- Swiss control group 仍在 `probe.h` 自包含实现：x86 16-byte group、NEON/portable
  8-byte group。它必须可被 typed `SwissTable` hot path 内联。

## 数据结构面

| 能力 | C++ 源码面 | C/FFI 动态面 | 关键合同 |
|---|---|---|---|
| flat map | `FlatHashMap<K,V>`（实现名 `SwissTable`） | erased POD hash map | C++ 支持 non-trivial ownership 与异构查找；C 面按字节存 POD；ctrl+连续 slots；SIMD group probe |
| flat set | `FlatHashSet<K>` | erased POD hash set | 真 set API/slot；canonical-key find；只读 key 迭代；C 面不要求 dummy mapped value |
| ordered map / set | `BTreeMap<K,V>` / `BTreeSet<K>` | erased POD B-tree map/set | unique-key；双向有序迭代；lower/upper/range visit/range erase；成功 mutation 后引用与 iterator 失效 |
| concurrent flat map | `ConcurrentFlatHashMap<K,V>`（短名 `ConcurrentHashMap`） | sharded erased POD concurrent map | 独立 cache-line shard + Swiss table；shared/exclusive accessor；紧凑、高吞吐；rehash 后地址不稳定 |
| concurrent node map | `ConcurrentNodeHashMap<K,V>` | 不机械导出 node 指针 | bucket RW accessor；node 地址跨 rehash 稳定；表增长保留旧 bucket array，销毁前要求 quiescent |
| POD dynamic array | `PodVector<T>` | erased POD vector | 默认可用或注入 allocator；copy/move 值语义；header-before-data；allocator/alignment 随 allocation 记录 |
| inline array | `SmallVector<T,N>` | 不单独导出 | 通用 owning value semantics；POD 保留 memcpy relocation；N 是编译期类型，C 动态需求由 erased vector 覆盖 |
| dynamic bitset | `DynamicBitSet` | opaque dynamic bitset | allocator-aware 值类型；canonical zero padding；set algebra/shift/popcount/set-bit iteration；长 mask runtime ISA dispatch |
| bounded sparse IDs | `SparseSet<Id>` | opaque sparse u32 set | dense+sparse 双数组；固定 universe；insert/erase/contains/clear O(1)；dense iteration O(active)；swap erase 不保序 |
| single-thread ring | `RingDeque<T>` / `ByteRingBuffer` | opaque byte ring | 前者是可增长/显式 bounded 的 owning 双端序列；后者为流式字节 staging 提供 copy 与至多两段 zero-copy spans；均非线程安全 |
| bounded SPSC FIFO | `SpscQueue<T>` | opaque `void*` SPSC handle | exact usable capacity；单 producer / 单 consumer；非平凡 typed 值或 caller-owned C pointer；无 CAS/逐 slot atomic |
| string view | `StringView`（实现名 `string_ref`） | `hyle_container_string_view` | ptr+u32 length+NUL bit；借用、不拥有 |
| interner | `StringInterner` | opaque StringInterner | generation-tagged direct ID；内容相等才同一条目 |
| path/text algorithms | typed view/buffer-fill | pointer/count/buffer-fill | 无 STL 所有权跨边界 |
| bounded/sharded queues | typed payload | `void*` payload opaque handle | HYLE 不拥有 payload |
| work steal/share | typed payload | `void*` payload opaque handle | owner/thief 契约不同，不能混用 |
| concurrent vector | typed append-only + random-access iterator | named `concurrent_ptr_vector` | caller-owned `void*` payload；多 writer 并行领取 index；地址稳定；quiesce 后读取；`size()` 不是 in-flight 完成栅栏；fallible path 分配失败不留洞 |

IntrusiveList 已下沉 core，因为 error/time/sched 都需要它；它不再属于 container
库或 container C ABI。

### 成熟库能力对账：仍不冒充“容器全集”

实现行数只能反映维护成本，不能证明能力完整。container 已包含
多套非平凡并发算法，但同本地 Abseil、Folly、oneTBB、Boost 的公开能力轴对账后，覆盖面
仍明显偏窄。现状与真实场景如下；“缺”不是立即复制一个类型，而是收口前必须设计或明确
负空间：

| 数据结构轴 | 当前 | 常用场景与判断 |
|---|---|---|
| 连续 sequence / inline sequence | `PodVector`、`SmallVector` | 已覆盖性能敏感 POD 与小 owning sequence；普通非平凡长序列继续用 `std::vector` |
| flat associative | `FlatHashMap/Set` | 已覆盖缓存友好的通用 lookup/membership |
| ordered/range associative | `BTreeMap/Set` | range query、确定性排序遍历、时间/调度索引；cache-packed unique-key B-tree，C++ typed 与 C erased POD 两面均覆盖 |
| node/reference-stable associative | `ConcurrentNodeHashMap` 覆盖并发 registry | accessor 生命周期内引用稳定，node 地址跨 grow 稳定；单线程 node extraction/merge 仍不复制 std API |
| concurrent associative | `ConcurrentFlatHashMap` + `ConcurrentNodeHashMap` | compact/rebuild 与 stable-reference/read-heavy 两种常用形状分开，不靠 policy flag 隐藏不同成本 |
| 单线程 ring/deque | `RingDeque<T>` / `ByteRingBuffer` | 流式窗口、音视频/网络 staging 与外部同步 ring 已覆盖；不让并发原子合同向单线程用户征税 |
| 单 producer / 单 consumer FIFO | `SpscQueue<T>` | 独立于 MPMC/stealing：一个 acquire/release index pair，无 CAS/逐 slot atomic；覆盖线程/流水线 stage 间的常见 bounded handoff |
| dynamic bitset / sparse set | `DynamicBitSet` / `SparseSet<Id>` | graph frontier、mask/set algebra 与稠密 ID active set 已覆盖；前者长 mask SIMD，后者以 O(1) clear + O(active) iteration 换取 2*U*Id bytes |
| priority queue / heap owner | 由标准库覆盖 | 单线程 comparator heap 先用 `std::priority_queue`；只有并发 priority 或 decrease-key 真实需求出现才新增形状 |
| cache/LRU policy | 不在 container | eviction 是策略，先由 util/消费者组合；不能因 Folly 有同名类型就把策略塞进 Tier 0 |
| blocking/unbounded queue | sched 负责等待/背压 | container 保持非阻塞机制，避免另造一套 shutdown/cancel/wait 语义 |
| strict inline-capacity sequence / sorted flat associative | 仍为候选轴 | 成熟库有真实形状，但必须由 no-allocation 或 read-mostly 消费合同驱动，不能按类型数量补齐 |

因此当前结论不是“行数已经等价主流全集”，也不是按对手 LOC 补代码。dynamic bitset/
sparse set、单线程 ring、ordered/range 与 concurrent associative 已通过 C++ 正常入口、
C ABI、复杂度与实际 graph/active-ID/stream-staging/time-index/shared-registry workload。
明确负空间仍是 single-thread node extraction/merge 与 ordered multi；候选能力轴还包括
strict inline-capacity sequence、sorted flat associative 与带完整策略的并发 priority/unbounded
机制。它们都需要真实消费者触发，不能用薄 wrapper 追成熟库类型数量。

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
C ABI 用 runtime size/alignment/comparator/allocator 擦除 POD，并提供 trivial cursor、range
visitor 与 `[lower,upper)` erase；不冻结 typed C++ node 布局。

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
多线程安全调用，析构必须等待所有调用和 accessor quiescent。C ABI 不返回无法携带 RAII
锁的裸 value pointer，而提供锁内 `find_copy` 与 `update(callback)`；这完整覆盖安全的 POD
读写，同时不冻结 C++ node/accessor 布局。

### RingDeque / ByteRingBuffer 的实际使用形状

`RingDeque<T>` 对照 Folly `DynamicRingQueue` / `FixedCapacityRingQueue` 的单调无符号
head/tail 与 power-of-two storage 形状，同时补上基础库用户真正需要的 owning 值语义、
双端 mutation、随机访问 iterator 和逻辑顺序的至多两段 readable spans。普通
`try_emplace/push_*` 满时增长并报告 OOM；显式 `try_*_bounded` 在 reserve 后永不分配，
满就是 backpressure。两者是同一个清楚的容器合同，而不是为 benchmark 复制一个简化
queue。元素要求 nothrow move construction/destruction，使增长失败能保持原状态。

head/tail 按成熟 ring 形状单调推进；无符号差值给出 size，power-of-two mask 给出物理
slot，hot pop 不在每次变空时重置两个游标。后者曾令 GCC 常态 FIFO 多一条分支和双写；
删除冗余后与 portable Folly-shaped floor 在 WSL 统计同档，并在 Windows/M4 快约
6%/15%，而不是增加常态 metadata 或专用 benchmark API。push/pop 两端与随机访问均为 O(1)，增长
为 O(N)，已达到该 storage 形状的渐进下界。

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
move-only 值；不可 move-assign 的值通过 callback 消费。C 面固定承载 caller-owned `void*`，
不冻结 element stride 或 C++ ctor/dtor callback。两个面都要求 producer/consumer 角色各自
只有一个线程，析构前 quiesce；`size_approx` 只用于 telemetry，不是跨线程完成栅栏。
这是一种常用并发架构能力，不是为 benchmark 保存的旁路或常态 metadata。

### DynamicBitSet / SparseSet 的实际使用形状

`DynamicBitSet` 对照 Boost dynamic_bitset 的常用面与 Folly 内部 frontier bitset 形状，保留
runtime size、resize/reserve、单 bit mutation、bulk set/reset/flip、AND/OR/XOR/difference、
shift、count/any/all、find-first/find-next 和 set-bit range。没有照抄 string conversion、
stream operator 等非基础加速职责。storage 是 allocator-owned `uint64_t` blocks；最后一块
的非逻辑 bit 永远清零，所有 bulk kernel 因此不需要额外 mask 每个 lane，只在 mutation
结束 sanitize 一次。大块集合运算通过 platform `best_isa()` 选独立 variant TU，container
仍只链接 platform/mem/error，不依赖 `hyle_simd` runtime。

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
`at()` 没有机械照抄：HYLE 构建可关闭异常，必须先决定是 Status/nullable 访问还是
fatal checked access，不能用一个异常形状破坏全库合同。新增普通序列接口没有借旧
push/read microbenchmark 数字宣称新的性能结论。

### PodVector 的成熟形状门

旧 `CArray<T>` 把 clay 实现名和窄 benchmark 形状一起暴露给了用户。当前公开主类型改为
`PodVector<T>`，并按 `std::vector` 与 Folly `fbvector` 的共同连续序列习惯提供默认构造、
count/range/initializer-list 构造、copy/move、front/back、正反迭代、reserve、push_back、
resize、count/range insert、assign、单项/范围 erase、clear、swap、shrink-to-fit 与值比较；
`try_reserve/try_push_back/try_resize/try_shrink_to_fit` 保留 HYLE allocator 的可恢复 OOM
通路。它刻意只接受 trivially-copyable T：这是 C/FFI-friendly byte relocation 合同，
不是第二个通用 `std::vector`。非平凡值用 SmallVector 或标准 vector。byte relocation
不额外要求 T 的 C++ copy constructor；增长前的别名值以对齐 byte snapshot 保存。

header-before-data、recorded allocator/alignment 和 type-erased reallocate core 仍是内部
实现，新增接口没有改变 indexed read 的单次寻址或增长时利用 backing realloc 的能力。
contiguous range 直接批量 memcpy；single-pass input range 只消费一次。range insert 先
物化输入，再以一次 memmove 搬后缀和一次 memcpy 写入；复杂度
O(size-position+inserted-count)，达到连续存储下界。缩短 resize 为 O(1)，不会构造无用
fill value；shrink O(size)，失败保留原 block。首轮 Windows 同合同 growth/read benchmark
仍分别约 1.48x/1.0x `std::vector`；新增 API 本身不能借旧 push/read 数字宣称 SOTA，若
真实消费者把位置 mutation 变成热点再单独建证书。

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
定长 key、删除按缓存 hash 定位，不再反复扫描已驻留长串。≤256B 的 Entry+字符使用
每 shard 小块 slab，release 后回到池内复用；大字符串立即归还注入 allocator，全部
slab 在 StringInterner 销毁时归还。地址在有效生命周期内稳定。

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
[`benchmarks-mem-container.md`](../performance/benchmarks-mem-container.md)。

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

UTF-8 validate/count 使用内化的 simdutf source，并由 simdutf 自己派发；这不是
container 对外依赖。hex encode/decode 当前是无查表分支的标量/编译器向量化实现，
只证明胜过等契约 scalar reference，未宣称战胜专门 SIMD codec。

SIMD 只进入可批量处理的字节/lane 工作，不按模块名强套：

| 路径 | 当前向量化 | 决策 |
|---|---|---|
| Swiss control-byte probe | x86 SSE2 16 lanes；AArch64 NEON 8 lanes | 已是成熟 Swiss 基线形状；每次 probe 再动态派发 AVX2 的固定税通常不值 |
| count/rfind/ASCII case | SSE4/AVX2/AVX512/NEON runtime dispatch | 已进入产品；长串摊薄 dispatcher，短串走标量 |
| UTF-8 validate/count | simdutf 自派发 | 借成熟核，不重写 |
| 长字节 table hash | x86 AVX2 XXH3 / baseline wyhash | varied-input 证据后进入产品；Windows/MSVC 从 1KiB、Linux/GCC14 从 512B，CPU/OS gate 在 platform，非 x86 不退化 |
| vector relocation | compiler/libc `memcpy/memmove` | 不重复实现平台已有宽拷贝与 non-temporal 策略 |
| lock-free queue / atomics | 不做 | 瓶颈是依赖链、cache line ownership 和 CAS，不是 lane arithmetic |
| DynamicBitSet | SSE4/AVX2/AVX512/NEON runtime dispatch | 位运算与 x86 count 复用 header-only vec DSL；NEON count 用 byte CNT+四路 UDOT，修复 M4 通用 u64-lane reduction 的 12% 缺口；小于 8/16 words 分别走本地 scalar |
| BTree node search | 数值 key 线性比较；通用 comparator 二分 | 典型 node 很小且 comparator/entry layout 泛型；当前三平台实际 workload 不支持为每个 node 增加 SIMD metadata 或动态派发税 |

长 hash 的宽核放在独立 AVX2 TU，baseline dispatcher 只有在 CPUID+XGETBV 安全门通过后
调用；varied-input sweep 分别给出 Windows/MSVC 1KiB 与 Linux/GCC14 512B 的保守交叉点。
`hash_bytes` 是进程内 table hash，值可随
HYLE 版本、构建与 ISA 选择变化，不得持久化或作为 wire fingerprint。若用户需要稳定内容
指纹，应由独立命名/独立合同提供，不能暗中冻结 table hasher 的当前 bit pattern。

## C ABI 边界

`include/hyle/c/container.h` 当前遵守统一 C ABI 机械合同，但它仍包含早期按实现逐项
导出的宽接口；HYLE 尚未发布、C ABI 尚未冻结。typed 容器能力稳定后将从真实 FFI
场景重新审查保留、合并或删除哪些入口，而不是为了兼容继续扩张。当前原则是：

| C ABI 能力组 | 当前判定 | 重审方向 |
|---|---|---|
| string view/hash/search/path buffer-fill | 保留能力 | 都是 FFI-native POD/无状态入口；合并重复命名与错误哨兵，不按 C++ 每个成员机械导出 |
| StringInterner | 保留 opaque 能力 | permanent 与 refcounted lifecycle 都有真实跨语言用途；batch/result 形状已按实际分片重审 |
| erased POD vector | 保留能力 | C 没有模板，size/alignment + allocator 是合理动态形状；不加入 C++ ctor/dtor callback |
| erased POD flat map/set | 保留能力、形状已重塑 | map 拆分 insert/insert_or_assign；set 用独立最小 handle，共享 value_size=0 raw policy，但不暴露伪 value |
| erased POD B-tree map/set | 保留能力、形状已审定 | C/FFI 确有 runtime ordered index；comparator+ctx、trivial cursor、range visit/erase 覆盖动态消费，不导出 node 布局或 C++ ctor/dtor callback |
| erased POD concurrent map | 保留能力、形状已审定 | sharded Swiss storage；只返回锁内 copy 或执行 exclusive update callback，不导出无法携带锁的裸 pointer/node accessor |
| 五类 pointer queue | 保留能力、角色已审定 | C 热路径统一 `try_*`；pointer 是跨 FFI 的 caller-owned work handle，不增加 runtime stride/copy callback；SPSC 明示唯一 producer/consumer，Chase-Lev 明示 owner bottom/thief top，WsFifo 明示 back/front |
| concurrent pointer vector | 保留能力、形状已重塑 | 明确命名 `concurrent_ptr_vector`；reserve 预分配 stable segments；`try_get(out)` 区分越界与合法 NULL payload；不冒充 erased POD vector |
| dynamic bitset / sparse u32 set | 保留 opaque 能力 | 两者是语言无关的 runtime-universe 数据结构；C 面覆盖 resize/algebra/iteration 与 dense active IDs，不暴露 C++ 布局 |

“保留能力”不等于冻结当前函数列表；“暂定”也不等于为了缩 ABI 直接删。最终 C 面只为
跨边界仍成立的动态能力服务，HYLE 内部 C++ 模块继续直接使用 typed API。

2026-07-25 的消费审计确认：仓内生产代码只直接使用 typed queue（sched），没有任何
模块通过 container C queue/ConcurrentVector facade 回调自己。因而早期含糊的
`push/pop` 没有兼容理由：C queue 已统一为非阻塞 `try_*`，并把 owner/thief 与
front/back 角色写进函数名。能力是否长期保留仍由真实 FFI 消费决定，不能因已重命名就
自动冻结。

2026-07-26 对照本地 rigtorp/Vyukov、oneTBB、crossbeam 与 moodycamel 后，pointer
queue 能力转为保留：work item/foreign object 本来就由调用方拥有，固定 `void*` 能让
lock-free hot path 保持定宽搬运，不引入运行时 element stride、copy callback 或逐项
分配。五类并非同一 benchmark 的重复品，而是 strict SPSC FIFO、strict MPMC FIFO、
sharded relaxed FIFO、owner-LIFO/thief-FIFO 与 owner-FIFO/batch-share 五种不同合同。

同一轮也把 typed work-stealing 面从早期算法名收束为公开合同：所有可能失败的热操作
统一 `try_*`，方法名编码 bottom/top、back/front；pop/steal 通过 out 参数只在成功时
写值。Chase-Lev 的第三态命名为 `kRetry`，与 Crossbeam 的竞态语义一致，不再用容易被
理解为取消/终止的 `abort`。这只重塑调用面，底层内存序、定宽 slot 和复杂度不变。
`BoundedQueue` 的值合同只要求 trivially-copyable，slot 只保存对象表示，不需要先默认
构造 T。两种 stealing ring 的 slot 则必须是 atomic，避免环回写与旧 thief read 构成
C++ data race；因此明确继承预分配 `std::atomic<T>` 的默认构造及 copy/move
constructible/assignable 限制，并要求 always lock-free，宽 task 通过 pointer/word-sized
handle 间接承载。

C concurrent facade 则确有旧形状问题：它只存 pointer 却叫通用 concurrent vector，且
旧 `at()` 无法区分越界与合法 NULL payload。现改名 `concurrent_ptr_vector`，以
`try_get(out)` 返回 presence，并补 typed/oneTBB 都有的 cold `reserve`。create 的
`first_segment_capacity` 只决定首段粒度、不偷偷分配；真实预热必须显式 reserve。

2026-07-26 又按本地 Abseil/Folly/simdutf 的字符串边界回审早期 Clay facade：
`capitalize(C-ident -> UI title)` 是应用展示策略，已从 C++/C 公共面删除；旧
`utf8_icmp` 只折叠 ASCII，却用名称暗示 Unicode case folding，也已删除。保留的
UTF-8 基元只承诺 validate/count；C escaping 则改为明确的 `string_c_escape`，对 NUL、
控制字节和非 UTF-8 byte 都产生完整转义。这里删的是错误抽象，不是为了缩符号数。
C hash traversal 同样不再冒充标准 `for_each`：提前停止的 callback 形状改名为
`map/set_visit`，返回 1 表示完整遍历、0 表示 visitor 停止或参数无效。

同一命名规则也覆盖线性容器：erased POD vector 使用 `try_push_back/try_pop_back`，并以
pointer/count 的 `try_append/try_insert/try_resize/erase/try_shrink_to_fit` 覆盖真正的 C
批量序列需求；concurrent pointer vector 使用 `try_push_back` 并返回稳定 index。不再让
一个裸 `push` 掩盖尾部方向或可失败的 allocation，也不要求 FFI 用户逐项跨 ABI。

- options 带 `struct_size`；allocator 按值捕获但 state 生命周期由调用方保证。
- 对象只经 create/destroy；不导出 C++ 类布局、STL、模板或异常。
- create/reserve/insert/intern 等冷失败走共享 `hyle_status + out`；队列满空、map miss、
  stale ID 等正常热控制流不构造错误对象。
- String/path 产出使用借用 view 或 buffer-fill。
- PodVector/FlatHashMap 通过运行时 item size/alignment/hash/equal 类型擦除；erased vector
  的 batch mutation 是 O(搬移尾部+输入字节)，自别名 insert 仅在必要时申请 scratch，
  OOM/越界不修改旧 size/data/capacity。
- C flat map 与 typed mutation 语义一致：insert 不覆盖，insert_or_assign 才覆盖；
  三态结果明确区分 inserted/already-present/assigned。
- C B-tree 与 typed unique-key/range 语义一致；NULL bound 表示无界，成功 mutation 统一
  使 cursor/pointer 失效，OOM 不改变旧结构。
- 并发 queue/vector 的稳定最小共同面使用 pointer payload；并发 map 则按 options 擦除
  POD key/value，并用 `find_copy/update` 保住锁的生命周期。

完整使用和失效规则见 [`c-container.md`](../guide/c-container.md)；收口证据见
[`container-module-closure.md`](../milestones/container-module-closure.md)。

## 下游利用

- sched 使用 ShardedQueue、WsFifoQueue、WorkStealingDeque 等调度原语。
- graph/sched 使用 ConcurrentVector 类 append-only 发布结构。
- error/time 等共享 intrusive 基础设施已归 core，不制造反向 container 依赖。
- 其他模块可使用 `hash_bytes`，但 hash 模块不依赖 `hyle_simd` runtime；≤16B typed key
  走 inline mix，较长字节串走库内 XXH3/wyhash 与 platform-gated AVX2 kernel。公开头不暴露 vendor 实现头；
  共享库调用 seam 是交付合同的一部分，不再为微基准提供 source-only inline 旁路。

## 性能验收边界

“SOTA”必须逐契约、逐操作成立，不能写成整个 container 的无条件标签：

- flat Swiss map 的 Windows 已测核心操作领先 Abseil；WSL 除 miss 约慢 3% 外领先，
  set miss 约慢 6%，不作无条件 SOTA 宣称；
- ordered B-tree 对 Abseil 的三平台实际矩阵中，copy/点查/构建/mutation 整体同档或领先；
  边界偏置 split/delete 已把原 M4 time-index 0.872x 提升到 1.027x。剩余 Windows full
  iteration/range、WSL above-max/expiry 与 M4 domain-miss/build/churn 仍公开登记，不用
  miss 分布特化、常态元数据或 benchmark API 隐藏；
- PodVector/SmallVector/队列按各自对等形状判断；
- 路径的大倍数主要来自零分配结构，对同形 string-view 扫描通常是平局；
- StringInterner 长串增长、重复 churn、并发分片和 batch 分别验收；单调用仍有 DLL seam；
- 4KB 字符扫描的 0.8x–0.95x 点值应结合 64KB/1MB、方差和调用形状，不包装成赢。

测试矩阵包括 typed 单测、纯 C 调用覆盖、C++ C-ABI 并发测试、allocator
故障注入、三平台构建/运行、ASan/UBSan/TSan，以及独立 SOTA benchmark。

## 维护红线

1. 新分配路径必须保存并使用原 allocator/alignment，且有 OOM 状态保持测试。
2. 新 C API 必须进入纯 C test、指南和动态导出机械核对。
3. 修改共享 per-ISA header 后，所有 variant TU 都必须重编译；Windows 还要验证
   import library，而不只运行库内部测试。
4. 不为复用 `hyle_simd` runtime 制造 Tier 倒挂；批量 kernel 可复用 header DSL，
   但 dispatcher 属 container 自己。
5. 任何性能结论注明对手合同、平台、输入形状和不成立的边界。

## 参考

- [abseil Swiss Tables design](https://abseil.io/about/design/swisstables)
- [SwissTable control-byte overview](https://faultlore.com/blah/hashbrown-tldr)
- 本地 `abseil-cpp/absl/container/internal/btree.h`（cache-packed B-tree 对照）
- [C ABI design](./c-abi-design.md)
- [C ABI authoring guide](./c-abi-authoring-guide.md)
