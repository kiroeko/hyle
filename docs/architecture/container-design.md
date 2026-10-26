# Container：按布局、寿命与进展选择数据结构

Container 为 Hyle 与独立 C++ 用户提供确有布局、容量、allocator、稳定 identity 或并发需求的容器。
它不复刻整个标准库，也不替推理引擎决定 KV 淘汰、请求背压或 tokenizer 策略。操作示例和失效规则
见 [Container 指南](../user/container.md)；节点 membership 单独见[侵入式容器](container-intrusive-design.md)。

## 产品与实现依赖

用户链接 `hyle::container`，内部按 intrusive、基础容器、concurrent、queue 与 text 分开依赖，
不让 timer 或 scheduler 为一个 hook 引入完整文本/哈希实现。公开 header 使用的 allocator/资源类型
形成真实 Mem 边；ISA 探测与第三方 codec 留在实现内部，vendor 类型不成为用户必须理解的配置。
当前构建组织见[组件图](../../cmake/HyleComponents.cmake)。

只有真实 storage/algorithm 语义不同才增加类型，不能按 fast/light 为同一种合同发明层级。

| 需求 | 类型族 | 取舍 |
|---|---|---|
| inline storage，溢出可增长 | SmallVector | 小尺寸不分配，grow 使原地址失效 |
| 严格固定容量 sequence | StaticVector/InplaceVector | 满即拒绝，不自动 spill |
| trivial bulk storage | PodVector | 显式 allocator 与 raw-storage 路线 |
| 双端与随机访问 | RingDeque | growable 或 bounded reject，不覆盖最旧值 |
| 精确最近 N 项 | CircularBuffer | 满时销毁/覆盖最旧项，容量不偷换成 2 次幂 |
| byte stream | ByteRingBuffer | byte segments 与 reserve/read/write 协议 |
| 热 metadata index | FlatHashMap/Set | 密集 control/slot，rehash 使 slot 地址失效 |
| read-mostly ordered index | FlatMap/Set | 二分查找，中间 mutation 搬移元素 |
| ordered/range mutation | BTreeMap/Set | 树节点/范围遍历成本 |
| bounded recency | LruCache | entry/weight 预算与 O(1) touch，不拥有 device 状态 |
| 并发 registry | ConcurrentFlatHashMap/Set、ConcurrentNodeHashMap/Set | 密集布局或受 guard 保护的稳定 node |
| 并行 append | ConcurrentVector | segmented growth，已发布元素地址稳定 |
| 有界 handoff | SpscQueue、BoundedQueue、ShardedQueue | producer/consumer、满/空与容量合同明确 |
| bursty handoff | UnboundedQueue | 分块增长、try-only OOM，不自带等待策略 |
| work sharing | WorkStealingDeque、WsFifoQueue | owner/thief 角色与进展不同 |
| dense/sparse membership | DynamicBitSet、SparseSet | word algebra 或按 ID 索引 |
| borrowed text/path | StringView 与算法 | 只借用字节，known-NUL 事实单独表达 |
| 文本/二进制 codec | UTF、Base64 | caller buffer、精确容量与失败合同 |
| stable string identity | StringInterner/StringId | bytes 与 generation 生命周期由 interner 持有 |

## 分配与失败不能隐藏

分配型容器接受 Hyle allocator，并为 OOM 提供对应 `try_*` 路线；便利入口可按已声明合同 fail-fast，
不能将失败伪装成成功。grow/reserve/rehash 是允许分配的 management path；bounded steady 操作不
暗中分配。allocator backing 必须覆盖全部 payload 与 metadata，不能只看 wrapper 自己是否还活着。

payload 的构造异常不由所有容器统一吞掉。平凡/nothrow 类型保留 relocation fast path；SmallVector、
RingDeque 等对 throwing move/copy 的保证以具体操作为准。并发 FIFO 已发布位置不能任意回滚，
因此 UnboundedQueue 的构造、提取 callback 与析构要求 nothrow，这属于算法约束。

## Stable identity 不等于永久引用

contiguous grow、flat hash rehash、erase/evict、ring 覆盖都有各自失效点。LruCache 的 get 改 recency，
peek 不改；返回 pointer 不能跨可能逐出的 mutation 保留。StringView 不拥有字节，StringId 的 generation
能检查失效，但不能延长来源寿命。

ConcurrentNodeHashMap 的 guard 保护节点使用，不能把稳定 node 地址当成无锁永久引用。默认路线将
旧 bucket tables 保留到析构，以生命周期内的有界额外表内存换取零 read-side reclamation tax。
显式 ReclaimingTableConcurrentNodeHashMap 才复用 Mem reclamation domain：取得并确认 bucket lock
后退出 table-read region，避免不必要地拉长 grace period；publish 后先释放旧 bucket locks，才退休
旧表，避免同步 callback 销毁仍被持有的 lock storage。

这是空间与读取成本的显式选择，不应把所有用户强制放到同一种回收策略。

## 并发保证属于类型和角色

普通 sequence、ring、flat/sorted/B-tree/LRU 需要外部同步。queue/work-stealing/concurrent map/set/vector
只承诺各自的 producer/consumer、owner/thief、growth 和 quiescence 条件，不把命名空间整体称为
thread-safe。cancel、timeout、wait、shutdown 与 service backpressure 由拥有 runtime 的上层组合。

自定义 allocator/hasher/equality policy 也在并发合同内：不同 shard 可并发调用同一个 policy，
producer allocation 可与 consumer deallocation 重叠。无状态默认实现满足该要求，不表示任意外部
policy 都安全。预分配 queue 不在 steady path 触达 allocator，不能为统一形状增加多余 lock。

## Hash 与文本的特别边界

Hash 是语义 hash，不按任意 trivially-copyable 类型的 padding bytes 猜 equality。自定义类型提供
specialization/hasher；只有对象表示确实等于 equality 时才选 BytewiseHash。默认 hash_bytes/string
hash 是 process-local table hash，不是稳定文件 fingerprint 或 cryptographic MAC。外部可控 key
使用带 caller secret 的 HardenedStringHash；持久 identity 由协议单独定义。

StringView 保存 known-NUL 事实，但不保证逻辑内容中没有更早 NUL，也不延长 backing 寿命。其 4 GiB
上限是公开设计选择，不是 std::string_view 的无损替代。

UTF conversion 的 input/output 由调用方持有，输出容量不足不做部分写入，并报告 required count。
Base64 的 alphabet、padding 与 whitespace 是正交策略：默认 standard/canonical padded/reject whitespace，
不能让第三方 forgiving default 绕过 Hyle 语义。快速 decode 与精确 first-invalid diagnosis 分工，
失败输出不作为成功结果提交。公开头不暴露 simdutf 的 option bitmask。

## 复杂度与完整成本

| 类型/操作 | 复杂度与注意事项 |
|---|---|
| sequence append、ring 双端 | amortized O(1)；固定满拒绝 O(1) |
| CircularBuffer steady push/pop/index | O(1)，resize O(n)，覆盖涉及析构 |
| flat hash lookup/insert/erase | expected amortized O(1)，rehash 另计 |
| flat ordered lookup / 中间 mutation | O(log n) / O(n) |
| B-tree lookup/insert/erase | O(log n) |
| LRU get/touch/erase/单项 evict | expected O(1) |
| LRU insert/assign、set_limits | 含逐出 k 项的 O(k)；insert/assign 另含索引摊销成本 |
| LRU clear | O(n) |
| bounded queue steady handoff | O(1)，受具体 contention/progress 合同约束 |
| unbounded queue | steady O(1)，block 边界发生分配/回收 |
| bitset algebra/count | O(machine-word blocks) |
| SparseSet membership | O(1)，空间还受最大 admitted ID 约束 |
| text/UTF/Base64 bulk | 与输入/输出长度成正比，caller-buffer 路线不隐式拥有分配 |

并发算法的 expected/steady 成本不等于每次调用都 wait-free。完整比较包括分配、rehash、回收、
payload lifetime、失败与峰值；不能改变这些合同只比较一个便宜步骤。

Highway、xxHash、wyhash、simdutf 等实际依赖由锁定版本管理；升级不能单方面改变 seed、编码、
fallback、failure 或交付 owner。新类型根据真实布局/复杂度缺口选择，普通标准容器足够时继续使用。
代码见[公开头](../../include/hyle/container/)，使用和安装方式由用户指南维护。
