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
  src/container/c_container.cc
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
  string_ref / StringInterner   StringInterner/path/string algorithms
  typed queues                  runtime ISA dispatcher
          |                       |
          +-----------+-----------+
                      |
                      v
Tier 0 dependencies: mem allocator + platform CPU discovery
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
| POD dynamic array | `PodVector<T>` | erased POD vector | 默认可用或注入 allocator；copy/move 值语义；header-before-data；allocator/alignment 随 allocation 记录 |
| inline array | `SmallVector<T,N>` | 不单独导出 | 通用 owning value semantics；POD 保留 memcpy relocation；N 是编译期类型，C 动态需求由 erased vector 覆盖 |
| string view | `StringView`（实现名 `string_ref`） | `hyle_container_string_view` | ptr+u32 length+NUL bit；借用、不拥有 |
| interner | `StringInterner` | opaque StringInterner | generation-tagged direct ID；内容相等才同一条目 |
| path/text algorithms | typed view/buffer-fill | pointer/count/buffer-fill | 无 STL 所有权跨边界 |
| bounded/sharded queues | typed payload | `void*` payload opaque handle | HYLE 不拥有 payload |
| work steal/share | typed payload | `void*` payload opaque handle | owner/thief 契约不同，不能混用 |
| concurrent vector | typed append-only + random-access iterator | named `concurrent_ptr_vector` | caller-owned `void*` payload；多 writer 并行领取 index；地址稳定；quiesce 后读取；`size()` 不是 in-flight 完成栅栏；fallible path 分配失败不留洞 |

IntrusiveList 已下沉 core，因为 error/time/sched 都需要它；它不再属于 container
库或 container C ABI。

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
| 四类 pointer queue | 保留能力、角色已审定 | C 热路径统一 `try_*`；pointer 是跨 FFI 的 caller-owned work handle，不增加 runtime stride/copy callback；Chase-Lev 明示 owner bottom/thief top，WsFifo 明示 back/front |
| concurrent pointer vector | 保留能力、形状已重塑 | 明确命名 `concurrent_ptr_vector`；reserve 预分配 stable segments；`try_get(out)` 区分越界与合法 NULL payload；不冒充 erased POD vector |

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
分配。四类并非同一 benchmark 的重复品，而是 strict MPMC FIFO、sharded relaxed FIFO、
owner-LIFO/thief-FIFO 与 owner-FIFO/batch-share 四种不同调度合同。

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
- 并发容器的稳定最小共同面使用 pointer payload。

完整使用和失效规则见 [`c-container.md`](../guide/c-container.md)；收口证据见
[`container-module-closure.md`](../milestones/container-module-closure.md)。

## 下游利用

- sched 使用 ShardedQueue、WsFifoQueue、WorkStealingDeque 等调度原语。
- graph/sched 使用 ConcurrentVector 类 append-only 发布结构。
- error/time 等共享 intrusive 基础设施已归 core，不制造反向 container 依赖。
- 其他模块可使用 `hash_bytes`，但 hash 模块不依赖 SIMD runtime；≤16B typed key
  走 inline mix，较长字节串走库内 XXH3/wyhash kernel。公开头不暴露 vendor 实现头；
  共享库调用 seam 是交付合同的一部分，不再为微基准提供 source-only inline 旁路。

## 性能验收边界

“SOTA”必须逐契约、逐操作成立，不能写成整个 container 的无条件标签：

- flat Swiss map 的 Windows 已测核心操作领先 Abseil；WSL 除 miss 约慢 3% 外领先，
  set miss 约慢 6%，不作无条件 SOTA 宣称；
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
- [C ABI design](./c-abi-design.md)
- [C ABI authoring guide](./c-abi-authoring-guide.md)
