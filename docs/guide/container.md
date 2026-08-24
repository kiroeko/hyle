# HYLE container：什么情况选择什么

container 提供性能敏感的基础数据结构、借用字符串/路径算法和调度队列。它不是为了
替换全部 STL，也不因为某个 microbenchmark 赢就要求用户改类型：先按 ownership、元素
生命周期、并发角色、顺序保证和容量上限选形状，再看性能证书。

本文是语言无关的选型入口，示例使用自然的 C++ typed API。纯 C/FFI 用户先在这里选
能力，再到 [`c-container.md`](./c-container.md) 查看 Status/out、opaque handle 和 POD
类型擦除映射。

## 一页选择表

| 需求 | 首选形状 | 不变量与代价 |
|---|---|---|
| 元素通常很少，希望避免首个 heap allocation | `SmallVector<T,N>` | owning、连续、支持非平凡/ move-only T；超过 N 后整体迁移 |
| 最大容量是协议/实时合同，整个生命周期禁止分配 | `StaticVector<T,N>`（别名 `InplaceVector`） | storage 全在对象内，data 地址不变；满时 `try_*` 返回 false，普通入口 fail-fast |
| trivially-copyable 数据、C-friendly layout | `PodVector<T>` | 只做字节 relocation；默认 system allocator，也可注入运行时 allocator |
| 普通长寿命非平凡动态序列 | `std::vector` + 可选 mem PMR adapter | HYLE 不造第二个通用 vector |
| 高频 key→value lookup | `FlatHashMap<K,V>`（实现名 `SwissTable`） | flat slot、expected O(1)、引用在 grow/erase 后失效 |
| 高频 membership / 去重 | `FlatHashSet<K>` | 真正 set slot，不为 empty mapped value 付空间；expected O(1) |
| 查多改少、批量装载后要有序点查/遍历 | `FlatMap<K,V>` / `FlatSet<K>` | 排序连续 storage；查找 O(log N)、遍历 cache-friendly，位置 mutation O(N) |
| 持续 mutation 的有序索引、range query | `BTreeMap<K,V>` / `BTreeSet<K>` | cache-packed B-tree；查找/mutation O(log N)，成功 mutation 后 iterator/reference 失效 |
| 多线程共享 map，优先紧凑 storage/重建吞吐 | `ConcurrentFlatHashMap<K,V>` | 分片 Swiss table；accessor 锁住 shard；释放后引用可因 grow/erase 失效 |
| 多线程 registry，grow 后对象地址必须稳定 | `ConcurrentNodeHashMap<K,V>` | bucket accessor；node 地址跨 rehash 稳定；每元素一次分配、间接访问 |
| 稠密编号上的 mask、frontier、集合代数 | `DynamicBitSet` | U/8 bytes；单 bit O(1)，bulk O(U/word) 且长 mask 按 ISA 派发 |
| 已知有界 ID，活跃项远少于 universe，频繁 clear/遍历 | `SparseSet<Id>` | 约 `2*U*sizeof(Id)`；insert/erase/clear O(1)，dense 迭代 O(active)，erase 不保序 |
| 单线程双端队列、滑动窗口或外部同步 ring | `RingDeque<T>` | owning、随机访问、两段 readable spans；普通入口可增长，`*_bounded` 明确固定容量背压 |
| 网络/文件/音视频字节 staging | `ByteRingBuffer` | partial copy API + 至多两段 zero-copy readable/writable spans；不逐 byte 构造元素 |
| 非拥有字符串切片并使用 HYLE 搜索/路径算法 | `StringView`（实现名 `string_ref`） | ptr + u32 byte length + NUL bit；底层字节必须长寿 |
| 内容去重并取得稳定字符串地址 | `StringInterner` | 分片线程安全；永久视图或 generation ID 两种生命周期 |
| 恰好一个 producer 与一个 consumer 的跨线程 FIFO | `SpscQueue<T>` | exact bounded capacity；每端只写自己的 index，无 CAS/逐 slot atomic |
| 严格容量、多生产者多消费者 | `BoundedQueue<T>` | bounded MPMC；满/空是正常控制流，不分配 |
| 极高 MPMC 争用，可放弃全局 FIFO | `ShardedQueue<T>` | producer hint 分片；每 consumer 保留自己的 cursor |
| 单 owner 做新任务、多个 thief 偷旧任务 | `WorkStealingDeque<T>` | owner bottom LIFO，thief top；固定容量 |
| owner 需要 FIFO，thief 批量接走约一半 | `WsFifoQueue<T>` | 调度专用；两个队列必须容量相同 |
| 多 writer append，quiesce 后按索引遍历 | `ConcurrentVector<T>` | 分段稳定地址；append 并行，实时 `size()` 不是元素完成栅栏 |
| 进程内 table hash | `hash_bytes` / `Hash<T>` | 内建标量/指针/view 有默认；自定义 key 提供语义 hasher；值不可持久化/上 wire |
| 词法路径切分/拼接/归一化 | `path.h` view/buffer API | 不访问文件系统，不解析 symlink/case/权限 |

### 当前没有提供的形状

不要把现有并发队列或 flat hash 硬拗成另一种合同：

| 需求 | 当前选择 | 原因/后续 |
|---|---|---|
| 单线程 grow 后引用稳定、node extraction/merge | 暂用 `std::unordered_map` 等 node-based 容器 | HYLE node map 的公开合同是并发 accessor，不为补 std API 数量复制 extraction/merge |
| duplicate-key ordered multi-index | 暂用 `std::multimap` / `std::multiset` | HYLE 当前只提供 unique-key index；没有真实消费者证明 multi 语义值得复制一套 API |
| 并发 priority / 通用 unbounded queue | 由 sched 的等待、关闭与背压策略组合 | container 不应只复制机制而遗漏 shutdown/cancel/ownership 合同 |

前两项是明确负空间；并发 priority/unbounded 仍是候选轴。它只有在真实基础库需求把
所有权、失败、等待和关闭合同说清后才会进入公开面，不能只因成熟库存在就复制。

## 连续容器

### 小序列用 SmallVector

```cpp
#include "hyle/container/small_vector.h"

hyle::container::SmallVector<std::string, 8> names;
names.emplace_back("alpha");
names.push_back(names.front());  // 即使这次触发 spill，别名也安全
for (const std::string& name : names) consume(name);
```

`SmallVector` 的常用值语义与 `std::vector` 相同：copy 是深拷贝，move-only 元素可用；
可按 count、iterator range 或 initializer-list 构造，`assign` 可整体替换，count/range
`insert`、单项/范围 `erase` 只移动受影响后缀。range 接口支持 single-pass input
iterator；插入本容器自己的 range 即使 spill 也安全。`swap`、值比较和
`try_shrink_to_fit`/`shrink_to_fit` 可用于普通容器生命周期，其中短序列可从 heap 收回
inline storage。这些位置操作与成熟 inline vector 一样是 O(尾部长度+插入数量)，不是
为 benchmark 增加的旁路。为了让 allocator OOM 后原 vector 仍可恢复，`T` 必须
nothrow move-construct/destruct；POD 仍走 memcpy relocation。`try_reserve`、
`try_emplace_back`、`try_push_back` 和 `try_shrink_to_fit` 报告 allocation failure；
无 `try_` 的熟悉入口以及 count/range mutation 把 OOM 当 fatal。当前不提供 `at()`：
项目还没有选择跨异常/无异常构建一致的越界返回策略；需要检查时先比较 `size()`。
若序列经常远超 N，inline bytes 只会放大每个 owner，对它使用普通 `std::vector`。

### 严格不分配的小序列用 StaticVector

```cpp
#include "hyle/container/static_vector.h"

hyle::container::StaticVector<Packet, 32> batch;
if (!batch.try_emplace_back(header, payload)) {
  flush(batch);                         // 满是正常背压，不是 OOM
  batch.clear();
}
```

`StaticVector<T,N>` 对应 C++26 `std::inplace_vector`、CCCL `cuda::std::inplace_vector`
和成熟 `static_vector` 的实际合同：capacity 是类型的一部分，元素 storage 完全嵌在对象内，
任何操作都不会访问 heap，`data()` 在对象生命周期内不会改变。它适合 packet batch、固定
协议字段、实时/嵌入式 scratch、小型 command list，以及“超过 N 必须背压/拒绝”而不是
自动增长的边界。只想减少通常的小序列 allocation、但合法输入可能超过 N 时仍用
`SmallVector`；动态上限或巨大 N 不应搬进 stack/object，改用 reserve 后的动态容器。

公开面包括 count/range/initializer-list 和 copy/move 值语义、连续及反向迭代、resize/
assign、任意位置 emplace/insert/erase、swap 和比较。类型本身没有 class-wide move
约束：不可移动的 mutex/handle 一类值仍可用尾部原位 emplace、pop/clear 与访问；会搬动
后缀的位置 insert/erase 要求 nothrow move construction，throwing-move 的整容器 move/swap
则给 basic exception guarantee。常见 non-trivial、move-only、无 move-assignment 元素可用，
trivially-copyable 后缀走 `memmove`。`N==0` 是可正常传递的空
容器。普通 std-shaped 入口在容量溢出时沿用 HYLE fail-fast；`try_emplace_back`、
`try_push_back`、`try_emplace`、`try_insert`、`try_resize`、`try_assign` 在容量不足时返回
false 且原值不变。range try 操作会先在对象内临时 materialize 输入，以同时保证
single-pass、self-range 和失败不改值；因此很大的 N 也会带来相应的 stack/object 成本。
这里的“不改值”指容量不足返回 false；若 T 自己的 copy/move construction 抛异常，尾部
emplace/resize 会回滚未完成后缀，整容器 assignment/move 保持可析构的 basic guarantee。

尾部 push/pop 为 O(1)，位置 insert/erase 为 O(受影响后缀+插入数)，没有比连续存储更低的
渐进复杂度。push 不使任何 pointer/reference 失效；位置 mutation 前方的引用与地址保持，
从 mutation 点开始的元素身份/值发生移动，应视为失效。和 `SmallVector` 一样不机械提供
异常 `at()`；先检查 `size()`。`InplaceVector<T,N>` 只是同一实现的熟悉别名。模板的元素
生命周期与 N 无法形成有意义的稳定 erased C ABI，因此 C 面不复制这一类型。

### POD 与通用 vector 不要混用

`PodVector<T>` 明确表示按字节 relocation 的容器，适合 packet descriptors、数值索引、
C ABI 数据和自定义 HYLE allocator。它默认即可构造，也能显式注入 allocator；拥有
copy/move 值语义、count/range 构造与 assign/insert、范围 erase、resize、shrink、swap、
比较和连续/反向迭代。连续 range 批量复制；single-pass input range 只读取一次；插入
自己的 range 即使增长也安全。它不适合 `std::string`、智能指针或自引用对象；这些类型
用 `SmallVector` 或 `std::vector`。任何增长、shrink 或位置 mutation 都可能使旧 data/
element pointer 失效。普通操作把 OOM 当 fatal；需要恢复时使用 `try_reserve`、
`try_push_back`、`try_resize` 或 `try_shrink_to_fit`。和 SmallVector 一样，当前没有在全库
越界策略确定前机械加入 `at()`。

## FlatHashMap 与 FlatHashSet

```cpp
#include "hyle/container/swiss_table.h"

hyle::container::FlatHashMap<std::string, std::unique_ptr<Node>,
                              hyle::container::StringHash,
                              hyle::container::StringEqual>
    nodes;
nodes.insert("root", std::make_unique<Node>());
if (auto* node = nodes.find(std::string_view("root"))) use(**node);

hyle::container::FlatHashSet<std::string, hyle::container::StringHash,
                              hyle::container::StringEqual>
    enabled;
enabled.insert("avx2");
if (const std::string* name = enabled.find(std::string_view("avx2"))) {
  use_canonical_name(*name);
}

// Internet-facing or otherwise attacker-controlled keys: obtain these two
// words from the application's CSPRNG/secret manager and keep them private.
using HardenedMap = hyle::container::FlatHashMap<
    std::string, Request, hyle::container::HardenedStringHash,
    hyle::container::StringEqual>;
HardenedMap requests(
    hyle::mem::SystemAllocator{},
    hyle::container::HardenedStringHash{secret_word_0, secret_word_1},
    hyle::container::StringEqual{});
```

map 支持 owning non-trivial key/value、move-only value 和 compatible Hash/Eq 的异构
find/contains/erase。`Hash<std::string>` 已覆盖同型 owning-string lookup；需要用
`string_view`、`string_ref` 或 C literal 无分配查询时，选公开透明
`StringHash` + `StringEqual`，两者只在调用期间借用字符且都是 `noexcept`。C 指针和
数组按 NUL 结尾字符串解释；含嵌入 NUL 的 key 必须用带显式长度的 `string_view` 或
`string_ref`。

默认 `StringHash` 在构造 hasher 时缓存同一进程的 best-effort seed，并把 seed 喂入 byte
core；所有默认实例在同一进程内一致，所以 copy/swap/rehash 保持正确，又能使离线预计算的
固定碰撞集合不能直接跨独立启动复用。每个 hasher/table policy 缓存一个 `uint64_t`，不增加
per-entry metadata，也避免每次 lookup 读取进程全局状态；它不宣称抗自适应攻击或提供
密码学随机性；父进程首次使用后再 `fork`，子进程会继承该 seed，只有 `exec` 才是新的启动。
测试/可复现工作可显式构造 `StringHash{seed}`。来自网络或其他不可信主体、攻击者可观察
table 行为的字符串应使用 `HardenedStringHash`：它是 SipHash-2-4，要求调用方注入并管理
不可预测的 128-bit secret，HYLE 不会把 ASLR/clock seed 冒充秘密。fast 与 hardened 都遵守
同一透明 lookup 与 `Eq(a,b) => same hash` 合同；代价选择必须按真实威胁模型，而不是为了
微基准统一成一个策略。和
Abseil/Folly/std 的熟悉语义一致：`insert` 不覆盖已有 value；
`try_emplace` 只在 key 不存在时构造 value，并返回 `{value_pointer, inserted}`；
`insert_or_assign` 明确表示覆盖；默认构造 value 的常用计数/缓存可用 `operator[]`。
这些普通入口把 OOM 当 fatal。需要恢复时用 `try_insert_or_assign`，或先
`try_prepare_insert` 把可能的 grow/tombstone rehash 移到冷路径。`reserve(final_size)`
本身也遵守熟悉的后续插入保证：若 tombstone 已吃光 growth budget，它会同容量 rehash，
不会只看 slot capacity 就虚假成功。Hash/Eq、元素 move/destroy 以及 `try_emplace` 的
value 构造必须 noexcept，保证失败不留下半迁移 table。

slot 是 flat storage；grow、rehash、erase 都会使相关 pointer/reference 失效。不要在
`for_each` 回调中修改同一个 map/set。map 可 range-for；每个 entry 用只读
`entry.key()` 与可写 `entry.value()`，避免用户破坏 key 的 probe 位置。set 的 iterator
只暴露 `const K&`。两者都是标准 forward iterator，并按 control group 批量跳过空槽，
不是每个 capacity slot 都走完整查找。

`FlatHashSet` 复用相同 ctrl/probe/resize 核心，但提供真正 set 接口；编译期约束保证
每个 occupied slot 的 `value_type` 与 `K` 等大，不把 `map<K, Empty>` 的空间浪费暴露给
用户。`find` 返回容器内 canonical key，适合用 `string_view` 查到 owning string。
`try_insert` 区分 inserted、already-present 与 OOM；普通 `insert` 把 OOM 视为 fatal。
纯 C/FFI 对应使用独立的 `hyle_container_hash_set_*` erased POD handle，同样不需要
dummy mapped value。

map/set 都是 owning 值容器：K/V 可复制时支持深 copy/copy assignment，也可从 iterator
range 或 initializer-list 构造/批量 insert，并支持 `swap` 与忽略迭代顺序的内容相等。
forward range 会先 reserve 一次；input iterator 只读取一次。move-only value 的 map
自然仍不可复制，但 move、lookup、insert/erase 不受影响。批量插入遇到重复 key 沿用
单项 `insert` 语义：保留先到的 mapped value，不暗中覆盖。

## FlatMap 与 FlatSet

查多改少、数据能按 key 批量生成，且需要确定性顺序或 binary-search range 时，使用
`FlatMap` / `FlatSet`。它们不是 `FlatHash*` 的改名：storage 始终按 comparator 排序，点查
为 O(log N)，完整遍历连续，位置 insert/erase 则要移动后缀 O(N)。长期不断插入、按时间
淘汰或维护大范围索引时应改用 `BTreeMap/Set`。

```cpp
#include "hyle/container/flat_sorted.h"

std::vector<std::pair<std::uint64_t, Route>> rows = load_sorted_routes();
hyle::container::FlatMap<std::uint64_t, Route> routes(
    hyle::container::sorted_unique, rows.begin(), rows.end());

if (auto it = routes.find(route_id); it != routes.end())
  dispatch(it->second);
for (auto it = routes.lower_bound(first); it != routes.upper_bound(last); ++it)
  visit(it->first, it->second);
```

两个类型为 allocator-aware、owning、unique-key 容器。`FlatMap` 采用 C++23 同类的
split storage：key 与 mapped value 各自连续；`keys()`/`values()` 可在只需一列时避免读入
另一列。`FlatSet` 是独立单数组 core。两者复用 `SmallVector<...,1>`，singleton 不分配，
也没有另造第二个通用 vector。公开面包括
异构 `find/contains/count/lower_bound/upper_bound/equal_range`、insert/erase、map 的
`try_emplace`/`insert_or_assign`/`operator[]`、reserve/shrink、copy/move 和有序迭代；map
iterator 是 random-access proxy，标准 `.first` key 只读、`.second` value 可写，亦可用
`key()/value()`；它不是 contiguous `pair*`。set iterator 始终只读。`sorted_unique` 构造/assign 是
O(N) bulk 路径；tag 是调用方对“严格有序且无等价 key”的承诺，库不会再逐项验证或偷偷
排序。sized range 只计算一次长度并 reserve 一次，随后直接构造连续尾部；失败会销毁完成
前缀且保持原容器。普通无 tag range 按 unique insert 处理，复杂度最坏 O(N²)。

`try_emplace` 中的 `try` 沿用标准容器含义：命中时不构造 mapped value，**不表示 OOM
可恢复**。需要恢复 allocation failure 时用返回 `FlatInsertResult` 的 `try_insert` /
`try_insert_or_assign`，或先 `try_reserve`；`try_assign(sorted_unique, ...)` 会在独立
replacement 中构造，失败时保持原容器。普通入口把 OOM 视为 fail-fast。成功插入/删除使
mutation 点及其后的 iterator/reference 失效；若发生 reallocation 则全部失效。mapped value
原地赋值不改变排序，也不使 iterator 失效。

这两个 typed 模板不新增 C ABI：C 面已有 erased POD B-tree 覆盖动态 comparator/range
需求，也可由调用方管理排序 POD buffer。为 flat sorted 再冻结一套 runtime entry layout、
iterator 与 bulk-tag ABI，只会复制能力而不能表达非平凡 C++ 生命周期。

## BTreeMap 与 BTreeSet

当用户需要 `lower_bound`、有序范围、确定性遍历，或按时间/序号删除一段索引时，用
`BTreeMap` / `BTreeSet`；只有点查且不要求顺序时仍优先 `FlatHashMap` / `FlatHashSet`。
这不是为了赢某条有序查找 benchmark 新增的别名：flat hash 根本无法表达 range query，
而 `std::map` 的每元素 node 形状会为短 key/value 支付更多 allocation 与 pointer chase。

```cpp
#include "hyle/container/btree.h"

hyle::container::BTreeMap<std::uint64_t, Job> deadline_index;
deadline_index.try_emplace(deadline_ns, Job{id});

const std::uint64_t now = monotonic_now();
for (auto it = deadline_index.begin();
     it != deadline_index.lower_bound(now);) {
  run(it->value());
  it = deadline_index.erase(it);
}
```

map iterator 暴露只读 `entry.key()` 与可写 `entry.value()`；set iterator 只暴露
`const K&`。两个类型都是 unique-key、owning、双向有序容器，提供熟悉的
find/contains/count、lower/upper/equal_range、insert/try_emplace/insert_or_assign、
`operator[]`（map）、iterator/range erase、clear/swap、range/initializer-list 构造和
copy/move。copy 直接克隆树形，复杂度 O(N)，不会按元素重新比较成 O(N log N)。

节点以约 256-byte leaf 为目标，值同时存于 leaf/internal node；普通查找、插入和删除
为 O(log N)，完整遍历为 O(N)。范围删除按 leaf run 批量搬移，并能直接卸下完全过期页，
适合 time index、prefix eviction 和窗口淘汰，但仍按实际删掉的节点工作。单调 append/
prepend 会按插入边界偏置分裂；连续从 begin/end 淘汰时不会先从兄弟页搬入马上又要删除的
entry。这些是日志/时间索引的常用结构优化，不改变 API，也没有为测项保留常态 rank、
subtree-size、finger 或预取元数据。

成功 mutation 可能在同一节点内搬移多个 slot，因此所有既有 iterator、pointer 和
reference 都视为失效；如果 grow 后对象地址必须稳定，单线程用 `std::map`，共享 registry
则考虑 `ConcurrentNodeHashMap`。类型非线程安全。默认走 system allocator，也可注入 HYLE
allocator；需要从 OOM 恢复时用返回 `InsertResult::kOutOfMemory` 的
`try_insert` / `try_insert_or_assign`；标准语义的 `try_emplace` 只表示“命中时不构造 value”，
和普通 insert 一样沿用 fail-fast OOM 合同。key/entry 必须 nothrow move/destruct，
comparator 必须 noexcept。
当前不提供 multimap/multiset、node handle、extract/merge、稳定引用或 rank/select；这些
能力不能只为追 API 数量塞进基础类型。

## ConcurrentFlatHashMap 与 ConcurrentNodeHashMap

先按引用稳定性和内存形状选，不按一条吞吐数字选：

- cache、memo、共享计数/状态表，值短小且允许释放 accessor 后地址变化：优先
  `ConcurrentFlatHashMap`。它是 `ConcurrentHashMap` 的显式别名；新代码在需要和 node
  版区分时优先写长名。
- service/device/graph object registry，调用期间要持有稳定引用，或 reserve/自动 grow 后
  object identity 仍必须稳定：用 `ConcurrentNodeHashMap`。代价是每 node 一次 allocation、
  pointer chase 和退休 bucket arrays。
- 根本不需要并发：仍用 `FlatHashMap`；并发类型的锁和 accessor 不是免费保险。

```cpp
#include "hyle/container/concurrent_hash_map.h"
#include "hyle/container/concurrent_node_hash_map.h"

using Counters =
    hyle::container::ConcurrentFlatHashMap<std::uint64_t, std::uint64_t>;
Counters counters(256);  // power-of-two shards；高争用服务可显式调大
Counters::Accessor counter;
counters.try_emplace(counter, request_id, 0);
++*counter;              // 独占保护这个 shard
counter.release();       // 尽早释放，缩短同 shard 阻塞

using Registry =
    hyle::container::ConcurrentNodeHashMap<std::uint64_t, DeviceState>;
Registry devices(256);   // 初始 bucket 数；可先 reserve 已知规模
Registry::ConstAccessor device;
if (devices.find(device, device_id)) use(*device);  // 引用在 accessor 内稳定
device.release();
```

`ConstAccessor` 持共享锁，`Accessor` 持独占锁；同一 accessor 传入下一次 find/insert 前会自动
release 旧绑定，但显式尽早 `release()` 更容易看出临界区。每线程对同一 map 最多保留一个
live accessor；不要拿着一个 accessor 再用另一个访问同 map。flat 的 `size/reserve/clear/
for_each` 与 node 的 `reserve/clear/for_each` 也要求先 release；不要从 callback re-enter 同一
map，否则会等待自己。析构前必须 join/quiesce 所有调用者；自定义 Hash/Eq 与 allocator
也必须允许不同线程并发调用。

两者的 `contains()` 只是瞬时快照：返回后其他线程可以立即 erase。需要读出并继续使用值，
应持有 `ConstAccessor`；需要 read-modify-write，直接用 `Accessor`，不要先 contains 再 find。
自动增长或 `reserve` 可能 OOM：可恢复路径使用返回 `InsertResult::kOutOfMemory` 的 `try_*`，
普通 `emplace/insert_or_assign` 沿用 HYLE fail-fast OOM 合同。flat 的 `size()` 要逐 shard 取
快照，node 的 `size()` 是 O(1) atomic；两者都不是跨 key 的事务快照。已知最终规模时先
reserve，既把 cold allocation 移出服务热路径，也缩短写锁持有时间。

## DynamicBitSet、SparseSet 与 FlatHashSet 怎么选

这三个类型都能回答 membership，但成本模型完全不同：

- universe 稠密、需要 AND/OR/XOR/difference、popcount、shift 或按 set bit 扫描时，用
  `DynamicBitSet`。典型是 graph frontier、feature mask、资源槽位和批量过滤。
- ID 有已知上界、active 集合稀疏、每帧/每轮频繁 `clear()` 后重建并遍历 active ID 时，
  用 `SparseSet<Id>`。典型是 ECS/graph visited worklist、稠密 handle membership。
- key 没有小而稳定的 universe，或 universe 很大但 active 极少、不能接受按 universe
  预留内存时，用 `FlatHashSet`。

```cpp
#include "hyle/container/dynamic_bitset.h"
#include "hyle/container/sparse_set.h"

hyle::container::DynamicBitSet ready(node_count);
hyle::container::DynamicBitSet allowed(node_count, true);
ready.set(3).set(80);
ready &= allowed;
for (std::size_t node : ready.set_bits()) schedule(node);

hyle::container::SparseSet<std::uint32_t> dirty(node_count);
dirty.insert(3);
dirty.insert(80);
for (std::uint32_t node : dirty) rebuild(node);
dirty.clear();  // O(1)，适合下一帧复用
```

`DynamicBitSet` 的普通 constructor/resize/reserve/push 把 OOM 当 fatal；可恢复路径使用
`try_resize`、`try_reserve`、`try_push_back`。`test/set/reset/flip` 的越界 index 和二元
运算尺寸不匹配是调用者 bug，C++ 面会 fatal；性能敏感的 `operator[]` 与 vector 一样要求
调用者先保证 index 有效。C ABI 会返回中性值或 `INVALID_ARGUMENT`。padding bit 始终为零，因此
`count/all/set_bits` 不会把尾部垃圾当成员。大 mask 的 AND/OR/XOR/and-not/count 使用
container 自己的 SSE4/AVX2/AVX512/NEON dispatcher；小 mask 避免间接调用税。它仍是普通
值类型，支持 copy/move、resize、push/pop、shift 和同尺寸集合代数。

`SparseSet` 构造/`reset_universe` 一次性分配 dense+sparse 两个数组；之后 insert/erase/
contains/clear 都不分配。`clear()` 只把 active size 归零，不擦整个 universe；contains
通过 dense 反查验证旧 sparse slot，因此复用安全。erase 用最后一项覆盖被删位置，所以
iterator/reference 与迭代顺序都不稳定。只做 membership 且愿意付 O(U) clear 时，单字节
dense bitmap 会有更少的 load；SparseSet 的价值在 O(1) clear、O(active) 迭代和同时保存
active IDs。性能表明确保留了这个 trade-off。

## 单线程 RingDeque 与字节 staging

需要拥有任意 `T`、从两端增删、随机访问或窗口遍历时用 `RingDeque<T>`；需要把 socket、
文件、codec 的 bytes 直接填入/取出 ring 时用 `ByteRingBuffer`。不要因为元素恰好是 byte
就机械选 `RingDeque<std::byte>`：后者仍逐元素表达生命周期，前者提供批量 memcpy 和
scatter/gather spans。

```cpp
#include "hyle/container/byte_ring_buffer.h"
#include "hyle/container/ring_deque.h"

hyle::container::RingDeque<Job> pending;
pending.reserve(1024);
if (!pending.try_push_back_bounded(make_job())) apply_backpressure();
consume(pending.front());
pending.pop_front();

hyle::container::ByteRingBuffer input(64 * 1024);
auto writable = input.writable();
std::size_t received = recv_into(writable.first, writable.second);
input.commit_write(received);  // received 不得超过刚才两段的总长度
parse(input.readable().first, input.readable().second);
input.consume(parsed_bytes);
```

`RingDeque` 的普通 `try_emplace/push_front/back` 在满时增长，OOM 返回 false；熟悉的
`emplace/push_*` 把 OOM 当 fatal。只有显式 `try_*_bounded` 保证 hot path 不分配，并在满
时返回 false。capacity 会向上取 power-of-two；增长/reserve 使所有地址失效，pop 使被
删除元素的地址失效；不增长的 push 保留既有元素地址。任何 mutation 后都应重新取得
iterator/segments，因为逻辑 index 与分段可能改变。`T` 必须 nothrow
move-construct/destruct。该类型不是线程安全容器；外部同步后可以跨线程使用，但真正的
lock-free 1P/1C 角色应选后文 `SpscQueue`。

`ByteRingBuffer::write/read` 是 partial transfer：返回实际搬运 bytes，空间不足/数据不足
不是错误。zero-copy 路径必须只在刚取得的 `writable/readable` spans 内访问，并在同一
逻辑步骤用不超过可用量的 `commit_write/consume`；reserve 或任何后续 mutation 后重新
取得 spans。reserve 会保留顺序并线性化已有数据。它不会自行增长 write：先 reserve 是
调用方明确的容量/背压决定。

## 字符串、驻留与路径

`StringView` 不拥有字节。`null_terminated` 只说明 `data[size]` 确实可读且为 NUL，不能
因为原始来源“看起来像字符串”就手填。默认 view 和空切片可安全查询；所有长度按 bytes，
只有 UTF-8 入口按 code point 解释。

协议/parser 边界只需判断输入是否合法时调用 `utf8_validate(view)`：它只做一遍 validate。
确实需要 code-point 数量时才调用 `utf8_length(view)`；非法返回 -1，合法输入需要 validate
再 count 两遍。两者都走 vendored simdutf 9.1 的 runtime ISA dispatch，不要求消费者选择
backend。

`StringInterner` 有两种不要混用的生命周期：

- `intern` / `intern_many` 返回稳定借用 view，活到 interner 销毁；适合进程/模型/图级符号。
- `acquire` 返回 generation-tagged `StringId`，用 `retain`/`release` 管引用；最后一份释放后内容可
  回收，旧 ID 不会命中新复用的 slot。

高吞吐导入优先 `acquire_many` / `intern_many`；实现按有界块只锁该块实际涉及的
shard，并按 shard 序加锁，不会因单 shard 批量工作冻结整个 interner。返回 view 都由
interner 拥有，不得 free。interner 必须比 view/ID 的所有使用者长寿。

性能选择也与合同一致：共享/回收需求明确时选 interner，高吞吐导入优先 batch；单线程
一次性导入用 `intern_many`，完全不需要跨线程共享时则直接用普通 flat set/map。分片会让
逐条入口的局部性表现随平台变化，不要为单个平台点值把并发合同降级为全局锁。当前证据
新鲜度、具体对手和点值只见
[`benchmarks-container`](../performance/benchmarks-container.md)。

`path.h` 只做词法操作。`normalize("a/../b")` 不查询磁盘，所以不能证明结果与真实文件、
symlink 或 Windows case rules 等价；需要真实 filesystem 行为时使用 platform 模块。

## 并发容器按角色选择

不要把五个 queue 当作同一 benchmark 的五个候选：

- 恰好一个 producer、一个 consumer 的有界 FIFO：`SpscQueue`。
- 任意 producer/consumer 且必须 hard bound：`BoundedQueue`。
- 同一 ring 已被 cache-line 争用压住，业务允许跨 shard 重排：`ShardedQueue`。
- scheduler 的 worker 自己 push/pop、别人 steal：`WorkStealingDeque`。
- owner 公平 FIFO，并希望空闲 worker 批量接工作：`WsFifoQueue`。

这些 queue 构造时完成 storage allocation，热 push/pop/steal 不分配。容量与
owner/thief 角色是合同；违反单 owner 约束不是“稍慢”，而是错误。

`SpscQueue<T>` 接受任意正数的 exact usable capacity；内部多一个 sentinel slot 区分
满/空，不把用户容量偷偷向 power-of-two 取整。producer 只调用 `try_emplace/try_push`，
consumer 只调用 `try_front/try_pop_front/try_pop/try_consume`。两端分别发布自己的 index，
每次成功操作最多 acquire 对端一次、release 自己一次，没有 CAS 或逐 slot atomic。
因此它可以拥有非平凡、move-only 元素；不能 move-assign 的元素通过 `try_consume` 在 live
slot 上处理。析构前必须 join/quiesce 两端。

所有可能因满、空或竞争失败的 typed 热入口也统一使用 `try_*`。Chase-Lev 使用
`try_push_bottom`、`try_pop_bottom(out)`、`try_steal_top(out)`；最后一项返回
`StealStatus::kOk/kEmpty/kRetry`，其中 retry 只是输掉本次竞争，不是取消。
WsFifo 使用 `try_push_back`、`try_pop_front(out)`、`try_steal_into(target, out)`。
失败时 out 保持原值；方向和 owner/thief 角色直接写在名称中，不再依靠记住实现细节。
`BoundedQueue` 的 payload 只需 trivially-copyable，slot 保存对象表示，不额外要求默认
构造或 copy assignment。两种 stealing queue 为保证环回 slot 的并发访问无数据竞争而
使用 `std::atomic<T>`，并要求它 always lock-free；常见 pointer、整数和 word-sized handle
满足，宽 task struct 请放在稳定 storage 中并传指针。它们的预分配 atomic slot 还要求 T
可默认构造并满足标准 atomic 的 copy/move constructible/assignable 合同。

`ConcurrentVector` 与 oneTBB 一样允许 writers 原子领取不同索引并并行构造，只有跨
segment allocation 进入冷锁。安全模式是：writers join/quiesce 后遍历；append 返回后
使用它返回的 index；或在 append 期间只读事先完成的 prefix。`size()` 是 reservation
count，不能在 writers 活跃时把 `[0,size())` 当作全部构造完成。元素构造必须 noexcept。
可恢复的 `try_emplace_back/try_push_back` 会先完成容量分配，OOM 不领取 index、不会
留下永久洞；普通 `emplace_back/push_back` 把 OOM 视为 fatal，因而用单次原子取号走
oneTBB 同级的常用快路径。writers quiesce 后可直接 range-for；iterator 是标准随机访问
iterator，并在 `++` 时缓存当前 segment 指针。不要在 writers 活跃时创建全量 iterator
range，因为它对 reservation count 取快照，并不额外发布仍在构造的 tail。

## 常见坑

- allocator 必须活到容器销毁；storage 必须由创建它的 allocator 释放。
- `StringView`、flat map/set find 返回值和 vector data 都是借用地址，分别受来源/grow/erase 失效。
- flat map/set 平均 O(1) 不等于 adversarial hash 安全；业务 Hash 必须提供足够熵且与 Eq 一致。
  默认 `Hash<T>` 对所有未特化的用户自定义类型都不可用；请提供逐字段语义 hasher。浮点零
  已归一，因此默认 `Hash<double>` 与 `std::equal_to<>` 对 `+0.0/-0.0` 保持同一 equality
  class。`BytewiseHash<T>` 是调用方显式 opt-in，并只接受 unique-representation type；该
  trait 仍不能证明自定义 Eq 比较每个 byte，故调用方必须另行保证
  `Eq(a,b) => bytes(a)==bytes(b)`。
- `ShardedQueue` 不保证 global FIFO；work-stealing 两种队列也不提供 MPMC 的任意角色语义。
- `SpscQueue` 的 producer/consumer 角色不能各自增加第二个线程；`size_approx()` 只是监控
  快照，不是发布 fence，也不能代替 push/pop 成功值做控制流。
- `RingDeque` / `ByteRingBuffer` 没有原子同步；不能用“只有一个 producer/consumer”作为
  无数据竞争的充分条件，除非外部同步也明确建立 happens-before。
- `ConcurrentVector::size()` 不是 live publication fence；并发写时读取新 tail 需要外部
  完成同步，或让 `T` 自己承载 atomic/future 状态。
- `hash_bytes` 的 17–128B 路径仍在公开头内内联；只有 >128B 才跨库进入长输入
  dispatcher。足够长的输入会按 shipping 平台的产品 API 交叉点自动进入宽 XXH3。
  它是进程内 table hash，不承诺跨版本/构建/CPU bit-identical；持久化 fingerprint、文件
  格式、网络协议和安全 MAC 都不能用它。极端热点仍优先缓存 hash 或批处理；HYLE 不公开
  依赖私有 vendor 头、且只适用于部分交付形状的 inline 旁路。
- 性能数字、平台、对手合同和仍未 SOTA 的条目只见
  [`benchmarks-container.md`](../performance/benchmarks-container.md)，本文不复制点值。
