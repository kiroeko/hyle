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
| trivially-copyable 数据、运行时 allocator、C-friendly layout | `PodVector<T>`（实现名 `CArray<T>`） | 只做字节 relocation；不运行元素 ctor/dtor |
| 普通长寿命非平凡动态序列 | `std::vector` + 可选 mem PMR adapter | HYLE 不造第二个通用 vector |
| 高频 key→value lookup | `FlatHashMap<K,V>`（实现名 `SwissTable`） | flat slot、expected O(1)、引用在 grow/erase 后失效 |
| 非拥有字符串切片并使用 HYLE 搜索/路径算法 | `StringView`（实现名 `string_ref`） | ptr + u32 byte length + NUL bit；底层字节必须长寿 |
| 内容去重并取得稳定字符串地址 | `StringInterner`（实现名 `StringBuf`） | 分片线程安全；永久视图或 generation ID 两种生命周期 |
| 严格容量、多生产者多消费者 | `BoundedQueue<T>` | bounded MPMC；满/空是正常控制流，不分配 |
| 极高 MPMC 争用，可放弃全局 FIFO | `ShardedQueue<T>` | producer hint 分片；每 consumer 保留自己的 cursor |
| 单 owner 做新任务、多个 thief 偷旧任务 | `WorkStealingDeque<T>` | owner bottom LIFO，thief top；固定容量 |
| owner 需要 FIFO，thief 批量接走约一半 | `WsFifoQueue<T>` | 调度专用；两个队列必须容量相同 |
| 多 writer append，quiesce 后按索引遍历 | `ConcurrentVector<T>` | 分段稳定地址；append 并行，实时 `size()` 不是元素完成栅栏 |
| 字节 hash | `hash_bytes` / `Hash<T>` | 小 typed key 内联；字节范围统一走稳定库入口 |
| 词法路径切分/拼接/归一化 | `path.h` view/buffer API | 不访问文件系统，不解析 symlink/case/权限 |

## 连续容器

### 小序列用 SmallVector

```cpp
#include "hyle/container/small_vector.h"

hyle::container::SmallVector<std::string, 8> names;
names.emplace_back("alpha");
names.push_back(names.front());  // 即使这次触发 spill，别名也安全
for (const std::string& name : names) consume(name);
```

`SmallVector` 的常用值语义与 `std::vector` 相同：copy 是深拷贝，move-only 元素可用，
clear/erase/pop 会析构元素。为了让 allocator OOM 后原 vector 仍可恢复，`T` 必须
nothrow move-construct/destruct；POD 仍走 memcpy relocation。`try_reserve`、
`try_emplace_back`、`try_push_back` 报告 allocation failure；无 `try_` 的熟悉入口把
OOM 当 fatal。若序列经常远超 N，inline bytes 只会放大每个 owner，对它使用普通
`std::vector`。

### POD 与通用 vector 不要混用

`PodVector<T>` 明确表示按字节 relocation 的容器，适合 packet descriptors、数值索引、
C ABI 数据和自定义 HYLE allocator。它不适合 `std::string`、智能指针或自引用对象；这些
类型用 `SmallVector` 或 `std::vector`。任何增长都会使旧 data/element pointer 失效。

## FlatHashMap

```cpp
#include "hyle/container/swiss_table.h"

hyle::container::FlatHashMap<std::string, std::unique_ptr<Node>,
                             TransparentStringHash, TransparentStringEq>
    nodes;
nodes.insert("root", std::make_unique<Node>());
if (auto* node = nodes.find(std::string_view("root"))) use(**node);
```

map 支持 owning non-trivial key/value、move-only value 和 compatible Hash/Eq 的异构
find/contains/erase。默认 `insert` 是 insert-or-assign：返回 true 表示新 key，false 表示
更新；OOM 是 fatal。需要恢复时用 `try_insert_or_assign`，或先
`try_prepare_insert` 把可能的 grow/tombstone rehash 移到冷路径。Hash/Eq、元素 move/
destroy 必须 noexcept，保证失败不留下半迁移 table。

slot 是 flat storage；grow、rehash、erase 都会使相关 pointer/reference 失效。不要在
`for_each` 回调中修改同一个 map。当前 typed 面只提供 map，没有把 map<key,empty> 冒充
空间最优 set；需要大量 set 的代码在 FlatHashSet 完成前使用成熟 flat set。

## 字符串、驻留与路径

`StringView` 不拥有字节。`null_terminated` 只说明 `data[size]` 确实可读且为 NUL，不能
因为原始来源“看起来像字符串”就手填。默认 view 和空切片可安全查询；所有长度按 bytes，
只有 UTF-8 入口按 code point 解释。

`StringInterner` 有两种不要混用的生命周期：

- `intern` / `intern_many` 返回稳定借用 view，活到 interner 销毁；适合进程/模型/图级符号。
- `add` 返回 generation-tagged `StringId`，用 retain/remove 管引用；最后一份删除后内容可
  回收，旧 ID 不会命中新复用的 slot。

高吞吐导入优先 batch；它一次获取 shard lock set。返回 view 都由 interner 拥有，不得
free。interner 必须比 view/ID 的所有使用者长寿。

`path.h` 只做词法操作。`normalize("a/../b")` 不查询磁盘，所以不能证明结果与真实文件、
symlink 或 Windows case rules 等价；需要真实 filesystem 行为时使用 platform 模块。

## 并发容器按角色选择

不要把四个 queue 当作同一 benchmark 的四个候选：

- 任意 producer/consumer 且必须 hard bound：`BoundedQueue`。
- 同一 ring 已被 cache-line 争用压住，业务允许跨 shard 重排：`ShardedQueue`。
- scheduler 的 worker 自己 push/pop、别人 steal：`WorkStealingDeque`。
- owner 公平 FIFO，并希望空闲 worker 批量接工作：`WsFifoQueue`。

这些 queue 构造时完成 storage allocation，热 push/pop/steal 不分配。容量与
owner/thief 角色是合同；违反单 owner 约束不是“稍慢”，而是错误。

`ConcurrentVector` 与 oneTBB 一样允许 writers 原子领取不同索引并并行构造，只有跨
segment allocation 进入冷锁。安全模式是：writers join/quiesce 后遍历；append 返回后
使用它返回的 index；或在 append 期间只读事先完成的 prefix。`size()` 是 reservation
count，不能在 writers 活跃时把 `[0,size())` 当作全部构造完成。元素构造必须 noexcept；
OOM 在领取 index 前返回，不会留下永久洞。

## 常见坑

- allocator 必须活到容器销毁；storage 必须由创建它的 allocator 释放。
- `StringView`、map find 返回值和 vector data 都是借用地址，分别受来源/grow/erase 失效。
- flat map 平均 O(1) 不等于 adversarial hash 安全；业务 Hash 必须提供足够熵且与 Eq 一致。
- `ShardedQueue` 不保证 global FIFO；work-stealing 两种队列也不提供 MPMC 的任意角色语义。
- `ConcurrentVector::size()` 不是 live publication fence；并发写时读取新 tail 需要外部
  完成同步，或让 `T` 自己承载 atomic/future 状态。
- `hash_bytes` 的 >16B 路径会跨库调用。若它确实成为极端热点，应先批处理或让上层缓存
  hash；HYLE 不公开依赖私有 vendor 头、且只适用于部分交付形状的 inline 旁路。
- 性能数字、平台、对手合同和仍未 SOTA 的条目只见
  [`benchmarks-container.md`](../performance/benchmarks-container.md)，本文不复制点值。
