# Container 架构

Container 是 HYLE 的通用数据结构与低层文本原语组件。它服务推理引擎的 metadata、registry、
worker handoff、active set 与 loader staging，也服务普通 C++ 消费者。它不是“把标准库再抄一遍”，
也不接受只为了 benchmark 出现的类型或常态元数据。

当前可依赖的用法见[用户指南](../user/container.md)，当前状态见
[`module-status.json`](../implementation/module-status.json)，每轮审查事实见对应 closure task。

## 设计目标

Container 只在至少满足一项时提供新类型：

- 标准库缺少所需的复杂度、并发进展、固定容量或分配失败合同；
- HYLE 能提供经过验证的、更适合高频数据路径的布局或算法；
- caller 必须显式选择 allocator、borrowed lifetime、稳定 identity 或 bounded backpressure；
- 多个真实 consumer 反复需要同一通用机制。

单个模型的 tokenizer、KV eviction、device buffer、graph side array 或 scheduler cancel policy，不会因
“里面也有数据结构”而下沉到 Container。标准容器已经自然满足的普通大 vector/map/set/deque、
fixed bitset 与 heap algorithm 继续直接使用标准库。

## 组件边界和依赖

```text
core ───────────────┐
mem ────────────────┼──> container ──> util / sched / io / graph / future engine
platform (private) ─┘
                         └─ build-only/private: Highway, xxHash, wyhash, simdutf
```

- `core` 提供编译器、断言与 visibility 合同。
- `mem` 提供 allocator；Container 不自建第二套系统分配器。
- `platform` 只提供运行时 ISA/CPU 事实，Container 自己拥有容器语义和 fallback。
- Highway、xxHash、wyhash、simdutf 只是私有实现机制，公开头不泄漏它们的类型。
- `intrusive_list.h`、`intrusive_forward_list.h` 与 `intrusive_hlist.h` 的 include owner 是独立的
  `container_intrusive` factor，不属于 compiled container factor 的实现边界；用户侧
  `hyle::container` façade 聚合两者。

产品源码是 C++23。Container 没有 C API、Native FFI schema 或稳定跨版本 C++ ABI；source、static 和
shared 三种交付暴露同一套 HYLE C++ 源码接口。

## 能力族和选择原则

| 需求族 | HYLE 路线 | 为什么存在 |
|---|---|---|
| 小序列，通常不分配、偶尔增长 | `SmallVector<T, N>` | inline storage + spill |
| 严格固定容量对象序列 | `StaticVector<T, N>` / `InplaceVector` | 容量是类型合同，绝不增长 |
| trivial 连续数据与显式 allocator | `PodVector<T>` | bulk、raw-storage 友好 |
| growable 双端 ring | `RingDeque<T>` | O(1) 双端与随机访问，按需增长或 bounded reject |
| 精确最近 N 项 | `CircularBuffer<T>` | 满时覆盖最旧项，逻辑容量不做 2 次幂取整 |
| byte stream ring | `ByteRingBuffer` | byte-oriented reserve/read/write/segments |
| 热 metadata hash index | `FlatHashMap/FlatHashSet` | open addressing、连续 control/slot |
| read-mostly sorted index | `FlatMap/FlatSet` | 连续存储与二分查找 |
| 持续 mutation、ordered/range index | `BTreeMap/BTreeSet` | 有序树与范围遍历 |
| 并发 registry | `ConcurrentFlatHashMap` / `ConcurrentNodeHashMap` | 前者偏密集，后者给被 guard 保护的稳定 node identity |
| 并行 append | `ConcurrentVector` | segmented growth，已发布元素地址稳定 |
| worker handoff | `SpscQueue` / `BoundedQueue` / `ShardedQueue` | 明确 producer/consumer 与 bounded 语义 |
| work sharing | `WorkStealingDeque` / `WsFifoQueue` | owner/thief 角色合同 |
| active membership | `DynamicBitSet` / `SparseSet` | dense algebra 或稀疏 O(1) membership |
| borrowed text/path | `StringView`、string/path algorithms | 显式借用、长度和 known-NUL 事实 |
| string identity | `StringInterner` / `StringId` | stable bytes、generation handle、可回收/永久两种生命周期 |
| table hash | `Hash`、`StringHash`、`HardenedStringHash` | semantic composition、trusted 与 hostile key 分流 |

## 不变量、失败和所有权

### 分配与失败

分配型容器必须允许注入 HYLE allocator，并提供不会把 OOM 伪装成异常成功的 `try_*` 路线。
便利构造/`push_*` 可以 fail-fast，但不得让失败后的对象处于半提交状态。grow、reserve、rehash、
capacity change 是允许分配的冷路径；已声明 bounded 或 steady-state no-allocation 的操作不得暗中分配。

容器不统一吞掉 `T` 的构造异常。每个模板在头文件中明确约束，例如 ring 类型要求 nothrow move，
并把该约束作为复杂度和失败合同的一部分。

### lifetime 与 invalidation

- contiguous grow 会使旧地址失效；不要跨 grow 保存 `data()`、iterator 或 reference。
- open-addressed table 的 rehash 会使 slot 地址失效；key 不可通过 iterator 修改。
- `ConcurrentNodeHashMap` 的 node 稳定性仍受 guard/lifetime 合同保护，不等于无锁永久引用。
- `ConcurrentNodeHashMap` 默认保留旧 table 到析构，以服务期旧表内存换取零 read-side reclamation tax；
  显式 `ReclaimingTableConcurrentNodeHashMap` 才通过 `mem_reclaim` 保护 table load，取得并确认 bucket
  lock 后立即退出 read region。publish 后必须先释放全部旧 bucket lock，再 retire 旧表，避免同步
  callback 销毁仍持有的 lock storage。
- `StringView` 永不拥有字节；`StringInterner::lookup` 返回的 view 受对应 ID 的生命周期约束。
- `CircularBuffer` 满时 push 会销毁一端元素；指向被覆盖元素的引用立即失效。

### 并发

“线程安全”按类型和操作声明，不是命名空间级保证。普通 sequence、ring、flat/sorted/B-tree 容器需要
外部同步。queue/work-stealing/concurrent map/vector 只承诺各自列出的角色、progress 与 quiescence；
它们不提供 timeout、cancel、wait service 或 shutdown policy，这些属于 scheduler/runtime。

## 复杂度与常态资源

| 操作 | 目标 |
|---|---|
| Small/Static/Pod append | amortized O(1)；strict fixed 满时 O(1) reject |
| RingDeque push/pop/index | amortized O(1)；bounded push O(1)，不分配 |
| CircularBuffer steady push/pop/index | O(1)，不分配；resize O(n) |
| Flat hash lookup/insert/erase-current | expected amortized O(1) |
| Flat sorted lookup | O(log n)；中间 insert/erase O(n) |
| B-tree lookup/insert/erase | O(log n) |
| DynamicBitSet algebra/count | O(number of machine-word blocks) |
| SparseSet contains/insert/erase | O(1)，内存与最大 admitted ID 及 dense size 有关 |
| StringView copy | O(1)；search 按输入长度，且不分配 |
| StringInterner acquire/lookup/release | expected O(1)，按 shard 同步；bytes 只保存一份 |

复杂度门先于点值 benchmark。任何优化都不得通过为全部对象增加无需求字段、改变 lifetime、减少
正确性检查、削弱进展保证或偷掉被计费阶段来换分。

## Hash 与文本决策

`Hash<T>` 是 semantic hash，不把任意 trivially-copyable 对象按 padding bytes 猜成 key。
integral、enum、pointer、floating point、string、pair、tuple、array 和有效的 noexcept `std::hash`
类型有自然路线；用户类型应提供 HYLE specialization/自定义 hasher，只有 equality 确实等于对象表示时
才显式选 `BytewiseHash<T>`。

`hash_bytes` 与默认 string hash 是 process-local table hash：可在一个进程内重复，不是文件格式、
wire fingerprint 或 cryptographic MAC。攻击者可控 key 使用 caller-secret 的
`HardenedStringHash`。持久 identity 必须保存原值或使用单独协议。

`StringView` 额外保存 `is_null_terminated()` 事实。它不保证逻辑内容中没有更早的 NUL，也不延长
来源生命周期；把它交给只接受 C string 的 API 前必须同时检查合同。4 GiB 上限是公开选择，不是
`std::string_view` 的无损替代。

## 外部实现与升级

依赖版本由 build/package lock 管理。升级 Highway、xxHash、wyhash 或 simdutf 时：

1. 先记录新 pin 和许可证；
2. 复跑对应 correctness、fallback/ISA 与 package projection；
3. 只有受影响的 hash/text/bitset 性能合同需要重取证；
4. 不把 vendor header/type 加入 public surface；
5. vendor 更新不能单方面改变 HYLE 的语义、seed、failure 或 delivery owner。

独立 ggml + llama.cpp、ds4、vLLM 与 SGLang 是能力差异和真实消费参考，不是产品依赖。每轮审查
必须先更新或确认 clean checkout 并记录准确 commit；不能用独立 ggml 的最新能力假装当前
llama.cpp 已消费，也不能把 vLLM/SGLang 的 GPU、Triton 或 tensor-fusion 表面机械复制成 Container API。

## 明确负空间和重开条件

下列内容在没有新 consumer 证据时不进入 Container：

- 通用 std/Boost API 全量 parity、node handle/merge、ordered multi；
- tokenizer trie/Aho、indexed/decrease-key heap；
- LRU/KV/device cache、eviction、persistence；
- blocking/cancel/backpressure service；
- compressed bitmap、rank/select；
- C ABI、Native FFI 或稳定 C++ ABI。

只有具名真实 journey、可复现缺陷、标准变化或成熟 peer 的通用解决方案同时证明现有 route 不足，才可
形成 reopen trigger。未来推理能力先登记到
[能力观察清单](inference-engine-capability-watchlist.md)，不能直接改 CLOSED 产品。

## 维护与验证

模块修改遵守[收口方法](../developer/review/module-closure-methodology.md)。最低要求包括：

- public-header self-containment；
- normal/empty/full/OOM/invalid/lifetime/invalidation 和适用并发 stress；
- x86/arm64 fallback 与 sanitizer；
- selected `container` 的 source、relocated static、relocated shared 消费；
- 资格化性能只用统一仪器与锁定 peer；历史 breadth benchmark 只是诊断；
- `python scripts/check-module-closure.py` 与 `python scripts/check-docs.py`。

当前性能事实只写在
[Container 性能证据](../implementation/evidence/performance/benchmarks-container.md)，不回填到本架构文档。
