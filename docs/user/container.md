# Container 使用指南

Container 给 C++23 用户提供 HYLE 的高频数据结构、borrowed text、hash 和字符串 identity 原语。
先按语义选类型，再看 benchmark；如果标准库已经自然满足需求，直接使用标准库通常是正确答案。

构建与三种交付方式见[交付与集成](distribution.md)，长期边界见
[Container 架构](../architecture/container-design.md)。

## 一分钟选择表

| 你的需求 | 选择 |
|---|---|
| caller-owned 稳定节点，需要零分配 O(1) 摘链或整段 handoff | `IntrusiveList` / `SafeIntrusiveList`（见[专门指南](container-intrusive.md)） |
| 元素很少，通常不分配，但允许偶尔增长 | `SmallVector<T, N>` |
| 编译期固定最大容量，绝不能增长 | `StaticVector<T, N>` / `InplaceVector<T, N>` |
| trivial/POD 连续数据、bulk 操作、显式 runtime allocator | `PodVector<T>` |
| 普通大连续数组 | `std::vector<T>` |
| 双端 push/pop、随机访问，并允许 grow | `RingDeque<T>` |
| 固定“最近 N 项”，满时自动丢最旧 | `CircularBuffer<T>` |
| byte stream 的 wrap/segments | `ByteRingBuffer` |
| 高频无序 metadata map/set | `FlatHashMap` / `FlatHashSet` |
| 构建后读多写少、需要有序遍历 | `FlatMap` / `FlatSet` |
| 持续插删并需要 ordered/range query | `BTreeMap` / `BTreeSet` |
| 普通 ordered map/set，无 HYLE 特殊需求 | `std::map` / `std::set` |
| dense active mask 与位运算 | `DynamicBitSet` |
| ID 稀疏但当前 active 数较少 | `SparseSet<Id>` |
| 单 producer / 单 consumer handoff | `SpscQueue<T>` |
| 固定容量 MPMC handoff | `BoundedQueue<T>` |
| 已有稳定 producer hint、需要分片降争用 | `ShardedQueue<T>` |
| 一个 owner、其他 worker steal | `WorkStealingDeque<T>` / `WsFifoQueue<T>` |
| 并发 registry，value 地址不要求跨 grow 稳定 | `ConcurrentFlatHashMap` |
| 并发 registry，需要 guard 内稳定 node | `ConcurrentNodeHashMap` |
| 多线程只 append、之后按 index 读 | `ConcurrentVector<T>` |
| 不拥有字符串、并需要 known-NUL/search/path helper | `StringView` |
| 重复字符串只存一份并获得 generation ID | `StringInterner` |
| 普通 borrowed text，不需要 HYLE 额外事实/算法 | `std::string_view` |

`intrusive_list.h` 与 owning/concurrent/queue/text 能力都通过 `hyle::container` 产品交付。直接使用
任何 Container 能力都选择并链接同一个 `container` 产品；其他产品公开类型所需的传递闭包由 CMake
自动处理。

## Sequence 和 ring

### 小序列

```cpp
#include <hyle/container/small_vector.h>

hyle::container::SmallVector<std::uint32_t, 16> shape;
if (!shape.try_push_back(128)) {
  // spill allocation failed; shape is still valid
}
for (std::uint32_t dimension : shape) {
  // ...
}
```

`SmallVector` 在前 N 项使用对象内 storage，超过 N 才分配。`StaticVector` 的 N 是严格上限；
满时 `try_*` 返回 false。不要用 `StaticVector` 表达“当前 reserve N、以后还可 grow”。

`PodVector` 只用于满足其 trivial-type 约束的批量连续数据。普通对象、异常安全或标准 allocator
生态优先 `SmallVector` / `std::vector`。

### growable deque 与最近 N 项不是同一种语义

```cpp
#include <hyle/container/circular_buffer.h>
#include <hyle/container/ring_deque.h>

hyle::container::RingDeque<Job> ready;
ready.push_back(next_job);              // full 时可 grow

hyle::container::CircularBuffer<int> recent_tokens(60);
for (int token : generated) {
  recent_tokens.push_back(token);       // 第 61 个会覆盖最旧 token
}
```

`CircularBuffer::capacity()` 精确返回用户给的 60，不会暴露 backing 的 2 次幂取整。steady push/pop/index
是 O(1) 且不分配；`try_set_capacity` 可能分配，缩小时保留最新元素。容量 0 时 `try_push_*` 返回
false。满时 push 会销毁被覆盖元素，因此不要保存指向该端元素的引用。

`RingDeque` 需要 grow-or-reject 语义；预先 `reserve` 后使用 bounded 操作可表达背压。两者都要求
元素满足头文件列出的 nothrow move/destruction 约束。

## Associative containers

### Flat hash

```cpp
#include <hyle/container/flat_hash_map.h>

hyle::container::FlatHashMap<std::uint64_t, TensorMeta> metadata;
metadata.reserve(expected);
metadata.insert_or_assign(id, meta);

if (TensorMeta* value = metadata.find(id)) {
  consume(*value);
}

for (auto it = metadata.begin(); it != metadata.end();) {
  if (expired(*it)) {
    it = metadata.erase(it);
  } else {
    ++it;
  }
}
const std::size_t removed =
    hyle::container::erase_if(metadata, [](const auto& entry) {
      return entry.value().unused();
    });
```

`find` 返回 value pointer，不返回 `std::map` iterator。iterator 暴露只读 key 和可写 value；
rehash 会使 slot 的 pointer/reference/iterator 失效。`erase(iterator)` 不会重新 hash key，
`erase_if` 是 O(n) 且不分配。

字符串表要显式决定威胁模型：

```cpp
using Trusted = hyle::container::FlatHashMap<
    std::string, Value,
    hyle::container::StringHash,
    hyle::container::StringEqual>;

using Hostile = hyle::container::FlatHashMap<
    std::string, Value,
    hyle::container::HardenedStringHash,
    hyle::container::StringEqual>;
```

`StringHash/StringEqual` 支持 `std::string`、`std::string_view`、`StringView` 和 C literal 的
无分配 heterogeneous lookup。攻击者可控 key 使用 caller-secret 的 hardened hasher。

### Composite 与用户 key

`Hash` 已支持 integral/enum/pointer/floating point、string、`pair`、`tuple`、`array`，并桥接
有效且 noexcept 的 `std::hash<T>`。用户结构请按 equality 字段组合：

```cpp
struct Key { std::uint32_t layer; std::uint32_t expert; };
struct KeyHash {
  std::uint64_t operator()(const Key& key) const noexcept {
    auto state = hyle::container::Hash<std::uint32_t>{}(key.layer);
    return hyle::container::hash_combine(
        state, hyle::container::Hash<std::uint32_t>{}(key.expert));
  }
};
```

不要默认按对象 bytes hash 一个带 padding 或自定义 equality 的结构。`BytewiseHash<T>` 只在
equality 真的是完整对象表示时显式选择。`hash_bytes` 的输出不能写入模型文件、数据库或 wire
protocol；升级或 ISA 路由后它可以改变。

### Sorted 与 B-tree

- 已有批量数据、构建后 read-mostly：`FlatMap/FlatSet`，lookup O(log n)，中间 mutation O(n)。
- 持续 mutation、range/lower_bound：`BTreeMap/BTreeSet`，主要操作 O(log n)。
- 需要 multi、node handle/merge 或标准 allocator 生态：直接选标准库/Boost。

## 并发容器和 queue

先写出线程角色，再选类型：

- `SpscQueue` 只允许一个 producer 和一个 consumer。
- `BoundedQueue` 是 strict bounded MPMC；满/空由 `try_push/try_pop` 返回，不会替你 park。
- `ShardedQueue` 的 hint 必须是 caller 稳定的 producer 身份，不是随机数。
- work-stealing 类型区分 owner 端和 thief 端；不要让多个线程调用 owner-only 操作。
- concurrent map 的回调/guard 决定引用有效期；不要把内部 value pointer 带出保护范围。
- `ConcurrentFlatHashMap` 适合不要求 value 地址跨 grow 稳定的紧凑 registry；
  `ConcurrentNodeHashMap` 的 node 在 accessor guard 内跨 rehash 保持稳定，但不提供无 guard 的永久引用。
- `ConcurrentNodeHashMap` 用析构前保留旧 bucket arrays 换取 lookup 不进入全局 reclamation domain。
  bucket 容量按二次幂增长，retired bucket storage 的几何和小于当前 array；强内存预算场景应在并发阶段前
  `reserve()` 到预计容量。持续增长且必须及时归还旧 table 的应用目前没有 shipping HYLE 路线，不要
  把 `mem_reclaim` 开发 factor 当作已交付能力。
- `reserve/clear/for_each` 是 management operation；不要在同一线程仍持有该 map accessor 时调用，
  map 析构前必须保证全部访问已静止。
- `ConcurrentVector` 适合并行 append，不是可并行 erase/insert 的 `std::vector`。

timeout、cancel、blocking wait、close 和 backpressure policy 属于 scheduler/runtime。需要这些语义时，
在 Container queue 外组合事件/调度原语，不要忙等一个没有该服务合同的 API。

## Dense 与 sparse membership

`DynamicBitSet` 适合 ID 范围较密、需要 count/and/or/xor/set-bit iteration 的 active mask；成本按
machine-word block 线性。编译期固定 mask 直接用 `std::bitset`。

`SparseSet<Id>` 适合 active 数少、但 ID 来自受控范围的 O(1) membership/insert/erase。它不是压缩
bitmap；极大、不受控 ID 可能让 sparse 索引增长，应在 owner 边界先验证 ID。

## StringView、text 与 path

```cpp
#include <hyle/container/string_view.h>

hyle::container::StringView view = "model.gguf";
std::string_view standard = view.as_std();

auto suffix = view.substr(6);
if (suffix.is_null_terminated()) {
  call_c_api(suffix.c_str());
}
```

`StringView` 永不拥有字节，来源必须活得比 view 久。它的长度是 `std::uint32_t`，输入超过 4 GiB
会 fail-fast。普通构造自 `std::string`/`std::string_view` 时不会猜末尾 NUL；从 C literal/C string
构造时记录 known-NUL。suffix 如果仍到原字符串尾部会保留该事实，prefix/middle 不会。

常见坑：

- `is_null_terminated()` 只证明 `data()[size()] == '\0'`；逻辑内容仍可能有 embedded NUL。
- `c_str()` 只在 known-NUL 时可调用；否则复制到 caller-owned buffer/`std::string`。
- `as_std()` 仍是借用，不延长 owner lifetime。
- 空 tail 可以有非 null 的 `data()`；不要用 `data()==nullptr` 判断 empty。
- HYLE 的 `_sv` 位于 `hyle::container::literals`；标准库 suffix 是 `sv`（没有前导下划线）。
  不想导入 literal namespace 时，直接写 `StringView{"text"}`。

`string_algo.h` 提供 byte/ASCII、hex/escape、UTF-8 validate/count/decode 等基础操作；path helper 处理
跨平台 lexical path。它们不访问文件系统，不替代 `std::filesystem` 的 IO/normalization 语义。
需要输出的 helper 优先写 caller buffer；按返回合同处理容量不足，不能假设一定 NUL-terminate。

## StringInterner

```cpp
#include <hyle/container/string_interner.h>

hyle::container::StringInterner names;
const auto id = names.acquire("attention.q_proj");
if (id.empty()) {
  // 非空输入返回 zero 表示 allocation failure
}
const hyle::container::StringInterner& read_only = names;
hyle::container::StringView name = read_only.lookup(id);

// matching lifetime
names.release(id);
```

`acquire/release` 是引用计数路线；最后一次 release 后旧 view 失效，generation-tagged `StringId`
也不会误指向复用 slot。`intern` 是永久路线，直到 interner 析构。两种路线不要混成“永久字符串也
release”。`StringId{0}` 表示空字符串，同时也是非空 acquire 的失败值；先看原输入是否为空。

`StringId` 可直接作为 HYLE hash key。ID 只在产生它的 pool 内有意义，不是持久 hash 或跨进程 ID。

## AllocatorRef、OOM 与异常

不同模板按 hot-path 需要接受 `mem::AllocatorRef` 或静态 allocator policy；以该头文件构造签名为准。
通用规则：

- 需要恢复 OOM 时调用 `try_*`，检查返回值；
- fail-fast 便利 API 只适合“失败即 invariant violation”的位置；
- reserve/grow 失败必须保留原容器；不要在失败后假设 requested capacity 已生效；
- injected allocator 的 lifetime 必须覆盖容器和所有 live allocation；
- 不要跨 CRT/模块用错误 allocator 释放 storage。

## 交付和头文件

源码、静态库和动态库用户都 include 同一套 `<hyle/container/...>` 头并链接
`hyle::container`。模板逻辑通常直接实例化在消费者 TU；compiled hash/text/interner 路径通过组件
target 正确链接。不要直接 include vendor header 或链接私有 Highway/xxHash/simdutf target。

Container 当前只面向 C++ 消费者，没有 C ABI/Native FFI。Java/C# 等 foreign runtime 不能把模板
容器当作稳定 ABI；只有出现完整 create/call/error/destroy consumer 后才会设计窄 runtime type-erased
边界。

## 排错清单

- link error：确认链接 `hyle::container`，而不是只添加 include 目录。
- shared build 下 vendor symbol/type 泄漏：这是 package 缺陷，请报告；不要在应用侧补私有依赖。
- hash lookup 找不到“相等”key：检查 hasher 是否与 equality 同字段、signed zero/embedded NUL 是否一致。
- 随机崩溃：优先查 borrowed `StringView`、rehash/grow 后旧 pointer、被覆盖 circular-buffer 元素。
- queue 卡住：确认 producer/consumer 角色、容量、shutdown 与外部 wait policy，没有把 nonblocking
  `try_*` 当 blocking service。
- 跨版本 hash 不一致：这是允许的；不要持久化 table hash。
