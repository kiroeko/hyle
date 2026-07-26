# container 纯 C ABI 使用指南

`#include "hyle/c/container.h"` 是 container 的纯 C/FFI 发布面。它只暴露
定宽 POD、buffer-fill 函数、allocator vtable 和 opaque handle；不暴露
STL、模板或 C++ 对象布局。C++ 源码用户若与 HYLE 使用同一工具链，也可直接使用
`hyle/container/*.h`，而且通常应优先使用 typed C++ 面；它不承诺跨编译器 ABI。

## 共同合同

- `hyle_container_string_view` 不拥有字节。`hyle_container_view` 构造普通视图；
  只有确实带尾零时才把 `null_terminated` 设为 1。
- 搜索失败返回 `HYLE_CONTAINER_NPOS`。所有长度都按字节计，UTF-8 函数除外。
- buffer-fill 函数返回完整所需长度；容量大于零时尽量写入并追加 `NUL`。先用
  `out=NULL, capacity=0` 查询长度是受支持的。输入与输出不要重叠，除非函数明确
  是原地操作（`hyle_container_path_normalize`）。
- 创建、扩容、插入和驻留这类可能失败的冷路径统一返回共享 `hyle_status`，并把
  结果写到首个 `out` 参数；进入函数时会先清空输出。调用者用
  `hyle_status_is_ok` / `hyle_status_get_code` 检查，并始终用 `hyle_status_free`
  释放返回状态。container 不再维护私有 TLS `last_error`。
- 满队列、空队列、未找到 key、stale interner ID 都是正常控制流，不创建 Status；
  仍由 `int`、空 view 或 `NULL` 表达。热路径因此没有错误字符串和分配成本。
- 无状态查询要求 view 满足 `size == 0 || data != NULL`。非法 view 返回该函数的
  中性/失败哨兵；这类编程错误不会偷偷写某个线程局部槽。
- `*_create_system` 是 header-inline 的 system allocator 便利入口。自定义
  `hyle_allocator` 必须活到对应 handle 销毁；handle、内部存储和释放始终走同一
  allocator。
- opaque handle 必须用对应 `*_destroy` 释放。队列只搬运 `void*`，从不拥有或
  释放 payload。

最小例子：

```c
#include "hyle/c/container.h"

int main(void) {
  hyle_container_vector_options opt =
      HYLE_CONTAINER_VECTOR_OPTIONS_FOR(int, 16);
  hyle_container_vector* v = NULL;
  int x = 42;
  hyle_status st = hyle_container_vector_create(&v, &opt);
  if (!hyle_status_is_ok(st)) {
    hyle_status_free(&st);
    return 1;
  }
  hyle_status_free(&st);
  st = hyle_container_vector_try_push_back(v, &x);
  if (!hyle_status_is_ok(st)) {
    hyle_status_free(&st);
    hyle_container_vector_destroy(v);
    return 1;
  }
  hyle_status_free(&st);
  hyle_container_vector_destroy(v);
  return 0;
}
```

## 字符串、路径与匹配

ASCII 单字节便利函数全部 header-inline：
`hyle_container_ascii_is_space`、`hyle_container_ascii_is_digit`、
`hyle_container_ascii_is_print`、`hyle_container_ascii_is_hex_digit`、
`hyle_container_ascii_hex_value`、`hyle_container_ascii_lower`、
`hyle_container_ascii_upper`、`hyle_container_path_is_separator`。

视图与查询：
`hyle_container_view`、`hyle_container_hash_bytes`、
`hyle_container_string_compare`、`hyle_container_string_equal_ignore_case`、
`hyle_container_string_find`、`hyle_container_string_rfind`、
`hyle_container_string_find_char`、`hyle_container_string_rfind_char`、
`hyle_container_string_count_char`、
`hyle_container_string_find_first_of`、
`hyle_container_string_find_first_not_of`、
`hyle_container_string_find_last_of`、
`hyle_container_string_find_last_not_of`、
`hyle_container_string_starts_with`、`hyle_container_string_ends_with`、
`hyle_container_string_contains`、`hyle_container_string_slice`、
`hyle_container_string_split`、`hyle_container_string_rsplit`、
`hyle_container_string_trim`。

写缓冲区与文本算法：
`hyle_container_string_copy`、`hyle_container_string_concat`、
`hyle_container_string_lower`、`hyle_container_string_upper`、
`hyle_container_hex_encode`、`hyle_container_hex_decode`、
`hyle_container_string_c_escape`、`hyle_container_utf8_length`。
`hyle_container_utf8_length` 对非法 UTF-8 返回 -1。quick-open fuzzy ranking 是
应用搜索策略，不是容器/FFI 基元；严格复审已从本模块删除。
`string_c_escape` 对控制字符使用熟悉的 `\\n`/`\\t` 等形式，其余非打印字节（包括
内嵌 NUL）使用固定三位八进制转义；它处理任意 byte string，不会把输出提前截断成
伪 C 字符串。

`hyle_container_hash_bytes` 是非加密、进程内 table hash：同一进程内对同一输入/seed
可重复，但值可随 HYLE 版本、构建或 CPU ISA 变化。它适合 C erased map callback 和
临时 lookup，不适合作为落盘 fingerprint、协议字段或不可信输入的 MAC。

路径视图与 buffer-fill：
`hyle_container_path_extension`、`hyle_container_path_strip_extension`、
`hyle_container_path_base`、`hyle_container_path_directory`、
`hyle_container_path_is_absolute`、`hyle_container_path_is_unc`、
`hyle_container_path_pop_first`、`hyle_container_path_pop_last`、
`hyle_container_path_join`、`hyle_container_path_join_array`、
`hyle_container_path_split_all`、
`hyle_container_path_replace_extension`、
`hyle_container_path_normalize`。`split_all` 返回完整分量数，即使输出数组较小；
分量仍只是原输入的借用视图。

## StringInterner

`hyle_container_string_interner_create` /
`hyle_container_string_interner_create_system` 通过 Status + out 创建线程安全驻留池，
`hyle_container_string_interner_destroy` 销毁。

- 可回收生命周期：`hyle_container_string_interner_acquire` 把 pool-local、带 generation
  的 `hyle_container_string_id` 写入 out；用 `hyle_container_string_interner_retain`、
  `hyle_container_string_interner_release`、`hyle_container_string_interner_lookup` 管理。
  删除后的旧 ID 永远不会指向后来复用该 slot 的字符串。
- 批量生命周期：`hyle_container_string_interner_acquire_many`、
  `hyle_container_string_interner_retain_many`、
  `hyle_container_string_interner_release_many` 处理批量生命周期。分配型 batch 通过
  `out_completed` 保留已完成前缀；每 64 项只锁实际涉及的 shard。非零 count 要求
  输入/输出数组有效。
- 永久驻留：`hyle_container_string_interner_intern` 和
  `hyle_container_string_interner_intern_many` 返回稳定、带尾零的借用视图，存活到池
  销毁。吞吐场景优先 batch。
- `hyle_container_string_interner_size` 返回当前不同字符串数。所有返回视图都由池拥有，
  不得释放。

## DynamicBitSet 与 sparse u32 set

两者都是 opaque allocator-owning handle，但服务不同的数据形状：

- `hyle_container_dynamic_bitset` 用于 graph frontier、mask 和按 bit 的集合代数，空间约
  为 universe/8；大块 AND/OR/XOR/and-not/count 会按 CPU ISA 派发。
- `hyle_container_sparse_u32_set` 用于有界 `uint32_t` ID 的稀疏 active 集合，空间约
  为 8*universe bytes；构造时预分配，之后 insert/erase/contains/clear 都不分配，clear
  是 O(1)，遍历 `data()[0..size)` 是 O(active)。erase 使用 swap-remove，不保序。

bitset 用 `hyle_container_dynamic_bitset_create[_system]` 创建，用 `destroy` 销毁；
`reserve/resize` 是可能 OOM 的 Status 冷路径。单 bit `test/set/flip` 对 NULL 或越界返回 0，
其中 `set/flip` 返回 1 表示 index 有效，不表示值发生了变化。`set_all/reset_all/flip_all`、
`shift_left/shift_right` 原地操作；`count/any/all` 查询。按 set bit 遍历时从 `find_first`
开始，反复调用 `find_next`，结束哨兵是 `HYLE_CONTAINER_BITSET_NPOS`。二元
`and/or/xor/and_not` 要求尺寸相同；NULL/尺寸不同返回 `INVALID_ARGUMENT` 且不修改目标。
`intersects/is_subset` 对非法或不同尺寸输入返回 0。

```c
hyle_container_dynamic_bitset* ready = NULL;
hyle_status st =
    hyle_container_dynamic_bitset_create_system(&ready, node_count, 0);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return 1; }
hyle_status_free(&st);
hyle_container_dynamic_bitset_set(ready, 3, 1);
for (size_t i = hyle_container_dynamic_bitset_find_first(ready);
     i != HYLE_CONTAINER_BITSET_NPOS;
     i = hyle_container_dynamic_bitset_find_next(ready, i)) {
  schedule_node(i);
}
hyle_container_dynamic_bitset_destroy(ready);
```

sparse set 用 `hyle_container_sparse_u32_set_create[_system]` 创建；`reset_universe` 会清空
集合并以 strong guarantee 更换 universe，失败保持旧集合。`insert` 仅在新 ID 成功加入时
返回 1；重复、越界或 NULL 返回 0。`erase` 仅在原来存在时返回 1；`contains` 是普通查询。
`data` 是借用的 contiguous active-ID 数组，在下一次 insert/erase/clear/reset/destroy 后
不得继续依赖内容或顺序。若只做 membership、不需要 O(1) clear/O(active) 遍历，且可接受
每轮清空整个 universe，dense byte bitmap 的单次查询更快；若 ID 没有合理上界，用 erased
flat hash set，不要为巨大 universe 预留 sparse set。

## 字节 RingBuffer

`hyle_container_byte_ring_*` 是单线程流式 staging 容器，不是原子 SPSC/MPMC queue。
create/reserve 使用调用方 allocator，capacity 向上取 power-of-two；`write/read` 可部分
完成并返回实际 byte 数，满/空返回 0 是正常背压，不通过 Status 报错。

```c
hyle_container_byte_ring* ring = NULL;
hyle_status st = hyle_container_byte_ring_create_system(&ring, 65536);
if (!hyle_status_is_ok(st)) return st;

hyle_container_mutable_byte_span dst[2];
size_t span_count = hyle_container_byte_ring_writable(ring, dst);
size_t n = recv_scatter(dst, span_count);
if (!hyle_container_byte_ring_commit_write(ring, n)) abort();

hyle_container_const_byte_span src[2];
span_count = hyle_container_byte_ring_readable(ring, src);
size_t used = parse_scatter(src, span_count);
if (!hyle_container_byte_ring_consume(ring, used)) abort();
hyle_container_byte_ring_destroy(ring);
```

`readable/writable` 的 out array 必须容纳两个 span，返回非空 span 数。span pointer 由
ring 借出，不得 free；reserve/destroy 会使地址失效，任何 write/read/commit/consume/
clear 后逻辑范围都改变，应重新查询。`commit_write` 不得超过当前 free space，`consume`
不得超过当前 size；C 面越界返回 0 且不修改状态。copy API 的非零 size 要求 input/output
非 NULL。需要拥有 typed 元素、双端操作或随机访问时用 C++ `RingDeque<T>`；C ABI 不
机械导出任意 C++ 构造/析构 callback，字节 staging 才是稳定且完整的动态能力。

## 运行时擦除 vector

用 `HYLE_CONTAINER_VECTOR_OPTIONS_INIT` 或
`HYLE_CONTAINER_VECTOR_OPTIONS_FOR` 初始化带 `struct_size` 的 options。
元素必须可按字节复制；C++ 非平凡类型不能放入该容器。

入口：
`hyle_container_vector_create`、`hyle_container_vector_destroy`、
`hyle_container_vector_size`、`hyle_container_vector_capacity`、
`hyle_container_vector_data`、`hyle_container_vector_data_const`、
`hyle_container_vector_at`、`hyle_container_vector_reserve`、
`hyle_container_vector_try_push_back`、
`hyle_container_vector_try_append`、
`hyle_container_vector_try_resize`、
`hyle_container_vector_try_insert`、
`hyle_container_vector_try_pop_back`、
`hyle_container_vector_erase`、
`hyle_container_vector_try_shrink_to_fit`、
`hyle_container_vector_clear`。

其中分配或检查 range 的 mutation 返回 Status；`try_pop_back` 的 0 只是容器为空。
batch API 的 `count` 是元素数，不是字节数；非零 count 要求 `items` 指向连续的
`count * item_size` 字节。`try_resize` 缩短时忽略 `fill_item`，增长时用它的一个元素
重复填充，故增长时不得为 NULL。`try_insert` 接受 `index == size`，`erase` 要求整个
`[index,index+count)` 位于 live range。空 vector 的 shrink 会释放 backing storage。

任何增长、shrink、insert 或 erase 都可能使先前由 `data`/`at` 返回的地址失效。
push/append/resize fill/insert 的输入都可来自当前 vector 的 live data；即使增长或
移动后缀也不会读悬空/被覆盖地址。批量 append/insert 只跨一次 ABI，正常外部输入不
分配额外临时块；只有自别名 insert 需要 scratch。OOM/越界返回时旧状态保持不变。

## 运行时擦除 flat hash map

`HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(K,V)` 创建 trivially-copyable POD map。
默认按字节 hash/equal；有 padding、指针成员或自定义相等语义的 key 必须提供
`hyle_container_hash_fn` 和 `hyle_container_equal_fn`。回调不得抛异常、递归修改
同一 map，且 context 必须活到销毁。

入口：
`hyle_container_hash_map_create`、`hyle_container_hash_map_destroy`、
`hyle_container_hash_map_size`、`hyle_container_hash_map_capacity`、
`hyle_container_hash_map_reserve`、`hyle_container_hash_map_clear`、
`hyle_container_hash_map_insert`、`hyle_container_hash_map_find`、
`hyle_container_hash_map_insert_or_assign`、
`hyle_container_hash_map_find_const`、`hyle_container_hash_map_contains`、
`hyle_container_hash_map_erase`、`hyle_container_hash_map_visit`。

这是 control-byte + SIMD group probing 的平坦 Swiss map，不是链式 C facade。
`capacity` 是 slot 数，最大装载量较小；增长/erase 会使返回的 key/value 地址失效。
`create` / `reserve` / 两种 insert 返回 Status。和 typed 面一致，`insert` 在 key 已
存在时保持原 value，并返回 `HYLE_CONTAINER_ALREADY_PRESENT`；只有
`insert_or_assign` 会覆盖，并返回 `HYLE_CONTAINER_ASSIGNED`。新 key 都返回
`HYLE_CONTAINER_INSERTED`。两者都支持 key/value 指向当前 map slot，growth 前会先
保存借用字节。
`reserve(final_size)` 同时检查 slot capacity 与 tombstone growth budget；必要时会
同容量 rehash，因此成功返回后不会因为旧 tombstone 立刻再次分配。set 复用相同合同。
`visit` 的 visitor 返回 0 可提前停止；此时 `visit` 也返回 0，完整遍历返回 1。
遍历期间不得修改 map。

## 运行时擦除有序 B-tree map / set

需要 lower/upper bound、按 key 顺序遍历、范围访问或时间窗口淘汰时，使用
`hyle_container_btree_map_*` / `hyle_container_btree_set_*`。只有点查且不要求顺序时，
flat hash 的 expected O(1) 更合适。C 面按值保存 trivially-copyable key/value；options
记录 size/alignment、comparator/context 和 allocator，不暴露 C++ 模板或 node 布局。

```c
typedef struct job_record {
  uint64_t id;
} job_record;

hyle_container_btree_options opt = HYLE_CONTAINER_BTREE_MAP_OPTIONS_FOR(
    uint64_t, job_record, hyle_container_compare_u64);
hyle_container_btree_map* deadlines = NULL;
hyle_status st = hyle_container_btree_map_create(&deadlines, &opt);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return 1; }
hyle_status_free(&st);

uint64_t deadline = 100;
job_record job = {0};
hyle_container_insert_result result;
st = hyle_container_btree_map_insert(&result, deadlines, &deadline, &job);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return 1; }
hyle_status_free(&st);

uint64_t now = 101;
(void)hyle_container_btree_map_erase_range(deadlines, NULL, &now);
hyle_container_btree_map_destroy(deadlines);
```

内置 comparator 有 i32/u32/i64/u64 和逐字节顺序；它们通过 memcpy 读取数值，不要求调用
方传入地址天然对齐。结构体 key、字符串内容或业务排序应提供返回 `<0 / 0 / >0` 的 strict
weak-order callback；context 必须活到 handle 销毁，callback 不得修改同一容器。

map 的 insert 不覆盖已有值，`insert_or_assign` 才覆盖；两者都用共享
`hyle_container_insert_result` 区分 inserted/already-present/assigned，并用 Status 报 OOM。
set 只存 key，不需要 dummy value。find 返回容器拥有的借用地址；contains/erase 是普通控制
流。成功 mutation 会使此前所有 find pointer 和 cursor 失效，失败或未命中的操作不失效。

`begin/lower_bound/upper_bound` 写出 trivial opaque cursor；`cursor_key/value` 取得当前元素，
`cursor_next/previous` 双向移动，其中 previous(end) 可到最后一个元素。cursor 只能交回创建
它的同一个 map/set，并应先用 `cursor_valid` 检查。`visit_range(lower, upper, ...)` 与
`erase_range(lower, upper)` 都使用半开区间 `[lower, upper)`；任一 bound 为 NULL 表示该侧
无界。visitor 返回 0 可提前停止，遍历期间不得修改容器或保留借用地址。

查找/插入/删除为 O(log N)，有序全遍历为 O(N)；长范围删除会按 node 中连续 run 批量搬移，
不是反复从 root 对每个 key 做一次全查找。它是单线程 unique-key index，不提供 multi、
node extraction/merge、稳定引用或并发锁。共享有序索引应由上层确定锁/快照合同，不能把
单线程 cursor 当成并发 accessor。

## 运行时擦除 concurrent hash map

多个线程共享 POD key/value 时，用 `hyle_container_concurrent_hash_map_*`，不要给单线程
`hyle_container_hash_map` 外包一把全局 mutex。它复用相同的
`hyle_container_hash_map_options`、insert result 与 Swiss probing，但把 key 分到
power-of-two 个独立 shard。options 宏已经带 system allocator，所以没有重复的
`create_system`；自定义 allocator/hash/equal/context 的生命周期与单线程 map 相同，且
这些 callback 必须允许多线程并发调用。

```c
static void increment(void* value, void* context) {
  *(uint64_t*)value += *(const uint64_t*)context;
}

hyle_container_hash_map_options opt =
    HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(uint64_t, uint64_t);
hyle_container_concurrent_hash_map* counts = NULL;
hyle_status st =
    hyle_container_concurrent_hash_map_create(&counts, &opt, 64);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return 1; }
hyle_status_free(&st);

uint64_t key = 7, value = 0, one = 1;
hyle_container_insert_result inserted;
st = hyle_container_concurrent_hash_map_insert(
    &inserted, counts, &key, &value);
if (!hyle_status_is_ok(st)) {
  hyle_status_free(&st);
  hyle_container_concurrent_hash_map_destroy(counts);
  return 1;
}
hyle_status_free(&st);
hyle_container_concurrent_hash_map_update(counts, &key, increment, &one);
hyle_container_concurrent_hash_map_find_copy(&value, counts, &key);
hyle_container_concurrent_hash_map_destroy(counts);
```

并发面刻意不返回 value pointer：跨 C ABI 无法让裸指针携带 RAII read lock，返回后另一个
线程就可能 erase/rehash。`find_copy` 在 shared lock 内复制恰好 `value_size` bytes；未找到
时返回 0 且保持 out 不变。原子 read-modify-write 用 `update`，callback 在目标 shard 的
exclusive lock 内执行；不得 re-enter 同一 map，也不应做长时间 blocking I/O。`contains`
只是瞬时快照。`visit` 对每个 shard 分别持 shared lock，跨 shard weakly consistent；callback
期间当前 key/value 稳定，但不能把地址保存到返回之后。

`insert`/`insert_or_assign` 与单线程 map 一样可能 grow 并通过 Status 报 OOM；已知规模先
`reserve(final_size)`。`size/capacity/clear/visit/reserve` 都逐 shard 工作，不是全表原子
snapshot；不要在这些管理操作的 callback 中重入。shard count 必须是非零 power-of-two，
默认从 64 开始通常合理；争用高且 key 分布均匀时再增加，过大只会增加空表和管理成本。
destroy 前必须让所有线程与 callback quiescent。

## 运行时擦除 flat hash set

只做 membership、去重或需要取得 canonical key 时，用
`HYLE_CONTAINER_HASH_SET_OPTIONS_FOR(K)` 和独立的
`hyle_container_hash_set_*`，不要建立 `map<K, char>`。它与 map 共享 flat Swiss
control/probe 核心，但 occupied slot 只存 hash 和 key，没有 dummy mapped-value 字段。

入口：
`hyle_container_hash_set_create`、`hyle_container_hash_set_destroy`、
`hyle_container_hash_set_size`、`hyle_container_hash_set_capacity`、
`hyle_container_hash_set_reserve`、`hyle_container_hash_set_clear`、
`hyle_container_hash_set_insert`、`hyle_container_hash_set_find`、
`hyle_container_hash_set_contains`、`hyle_container_hash_set_erase`、
`hyle_container_hash_set_visit`。

key 的 POD、hash/equality callback、context 生命周期和地址失效规则与 C map 相同。
`insert` 用 `HYLE_CONTAINER_INSERTED` / `HYLE_CONTAINER_ALREADY_PRESENT` 区分新成员与
重复成员；`find` 返回 set 内只读 canonical key，未找到返回 `NULL`。遍历只暴露
`const void* key`；visitor 返回 0 时 `visit` 返回 0，完整遍历返回 1。遍历期间不得
修改 set。

## 并发容器

下列 queue 的所有 `try_*` 都是无分配、非阻塞热路径：返回 0 表示该次操作未取得元素/空间，out item
仅在返回成功时写入。名称刻意不再使用早期含糊的 `push/pop`：bounded/sharded 明示
“可能因满/空失败”，work-stealing 明示 owner 的 bottom 与 thief 的 top，WsFifo 明示
owner 的 back/front。调用者若要等待，应在 sched/platform 层组合 park/wake，而不是
把容器误解成 blocking queue。

`hyle_container_spsc_queue_create`、
`hyle_container_spsc_queue_create_system`、
`hyle_container_spsc_queue_destroy`、
`hyle_container_spsc_queue_try_push`、
`hyle_container_spsc_queue_try_pop`、
`hyle_container_spsc_queue_size_approx`、
`hyle_container_spsc_queue_capacity` 是 exact-capacity 的单 producer / 单 consumer FIFO。
producer 线程只能 push，consumer 线程只能 pop；各角色增加第二个调用线程是合同错误。
`size_approx` 是可过期的监控快照，不是发布栅栏。payload 可为 NULL，HYLE 不拥有它。

`hyle_container_bounded_queue_create`、
`hyle_container_bounded_queue_create_system`、
`hyle_container_bounded_queue_destroy`、
`hyle_container_bounded_queue_try_push`、
`hyle_container_bounded_queue_try_pop`、
`hyle_container_bounded_queue_capacity` 是有界 MPMC ring。

`hyle_container_sharded_queue_create`、
`hyle_container_sharded_queue_create_system`、
`hyle_container_sharded_queue_destroy`、
`hyle_container_sharded_queue_try_push`、
`hyle_container_sharded_queue_try_pop`、
`hyle_container_sharded_queue_capacity`、
`hyle_container_sharded_queue_shard_count` 是高争用 MPMC 组合；producer hint 选分片，
consumer cursor 应由各 consumer 私有并在多次 pop 间保留。

`hyle_container_work_stealing_deque_create`、
`hyle_container_work_stealing_deque_create_system`、
`hyle_container_work_stealing_deque_destroy`、
`hyle_container_work_stealing_deque_try_push_bottom`、
`hyle_container_work_stealing_deque_try_pop_bottom`、
`hyle_container_work_stealing_deque_try_steal_top`、
`hyle_container_work_stealing_deque_capacity` 是单 owner 在 bottom push/pop、多个
thief 从 top steal
的 Chase-Lev 风格 deque。steal 返回
`HYLE_CONTAINER_STEAL_OK/EMPTY/RETRY`；RETRY 表示输掉本次竞争，调用方可按自己的
退避策略重试；它不是取消或终止。

`hyle_container_ws_fifo_queue_create`、
`hyle_container_ws_fifo_queue_create_system`、
`hyle_container_ws_fifo_queue_destroy`、
`hyle_container_ws_fifo_queue_try_push_back`、
`hyle_container_ws_fifo_queue_try_pop_front`、
`hyle_container_ws_fifo_queue_try_steal_into`、
`hyle_container_ws_fifo_queue_capacity` 是 owner FIFO + 批量转移队列；
owner 在 back push、front pop；`try_steal_into` 要求两个不同且等容量的队列。

`hyle_container_concurrent_ptr_vector_create`、
`hyle_container_concurrent_ptr_vector_create_system`、
`hyle_container_concurrent_ptr_vector_destroy`、
`hyle_container_concurrent_ptr_vector_reserve`、
`hyle_container_concurrent_ptr_vector_try_push_back`、
`hyle_container_concurrent_ptr_vector_size`、
`hyle_container_concurrent_ptr_vector_capacity`、
`hyle_container_concurrent_ptr_vector_try_get` 是 append-only pointer vector。
`try_push_back` 返回
Status 并写 `out_index`；已经发布的 index 稳定，HYLE 不拥有或移动 payload 指向的
对象，调用者仍负责其生命周期。
多个 `push` 可真正并行；`size` 返回已领取 index 数，不是 in-flight push 的完成栅栏。
读取新 tail 前必须等待对应 push 返回或等待所有 writers quiesce；并发 append 期间可读
事先完成的 prefix。
create 的 `first_segment_capacity` 只选择首段及后续倍增粒度，不执行预分配；已知批量
大小时应先调用 `reserve`，把 OOM 和 segment allocation 移出 writer hot path。
`try_get` 以返回值表示 index 是否存在，并通过 out 写 payload；因此合法 NULL payload
不会再和越界混淆，失败时 out 保持不变。

所有并发 handle 的 `create` 都是 Status + out。queue 创建完成后，满/空/竞争走无分配
热路径返回值；concurrent map 的 insert/grow 仍用 Status 报 OOM。queue 的 `void*`
payload 可为 NULL，HYLE 不解释也不拥有它。

## C++ 形状在 C 中怎么取得

不导出模板实例或 `std::vector`，也不冻结内部类布局；按能力使用下列动态形状：

| C++ 形状 | C/FFI 通路 |
|---|---|
| `SmallVector<T,N>` / `PodVector<T>` | `hyle_container_vector_*`：options 传 `item_size`/`item_alignment`（通常用 `HYLE_CONTAINER_VECTOR_OPTIONS_FOR`），按值存储 trivially-copyable 元素。C ABI 不接收 C++ ctor/dtor callback；inline `N` 也不是稳定跨 ABI 属性 |
| `FlatHashMap<K,V>` | `hyle_container_hash_map_*`：C++ map 可拥有 non-trivial 值；C/FFI map 则按 options 中的 size/alignment 擦除 POD，hash/equality 由 callback+ctx 注入 |
| `BTreeMap<K,V>` / `BTreeSet<K>` | `hyle_container_btree_map_*` / `hyle_container_btree_set_*`：按值 POD ordered index；comparator+ctx 决定顺序，cursor/range visit/range erase 完整覆盖动态有序消费 |
| `ConcurrentFlatHashMap<K,V>` | `hyle_container_concurrent_hash_map_*`：sharded erased POD map；`find_copy`/`update` 保证锁内访问，不跨 ABI 泄漏无保护 pointer |
| `ConcurrentNodeHashMap<K,V>` | 并发 POD 读写仍走上述 concurrent map；C 不暴露 node/accessor 布局。若必须长期持有 identity，保存 caller-owned pointer/handle 作为 value，并由业务管理其生命周期 |
| `FlatHashSet<K>` | `hyle_container_hash_set_*`：按 options 中的 key size/alignment 擦除 POD；独立 set slot，不用 map + dummy value |
| `ByteRingBuffer` | `hyle_container_byte_ring_*`：opaque 单线程字节 ring，copy 与 two-span zero-copy 两条路径都可达 |
| `RingDeque<T>` | 无直接模板 ABI；字节流用 byte ring，trivially-copyable 批量序列用 erased vector。若未来出现真实 typed deque FFI 需求，再设计 element-size/alignment 形状，不冻结 ctor/dtor callback |
| `SpscQueue<T>` / `BoundedQueue<T>` / `ShardedQueue<T>` / `WorkStealingDeque<T>` / `WsFifoQueue<T>` | 对应 `hyle_container_*queue*` / `*_deque_*` opaque 句柄接受 `void*` payload；payload 生命周期归调用方；SPSC 额外要求每种角色恰好一个线程 |
| `ConcurrentVector<T>` | `hyle_container_concurrent_ptr_vector_*` append caller-owned `void*`；名称明确它不是 arbitrary POD 类型擦除 |
| `Hash<T>` / `hash_bytes` | C 侧对实际字节调用进程内 `hyle_container_hash_bytes`，或向 flat map 注入业务 hash callback；不暴露 vendor 实现头，也不承诺持久 bit pattern |

这张表表达的是“C++ 静态形状不跨 ABI、运行时能力仍可达”。如果未来需要
按值 POD queue，应该新增 `element_size`/`alignment` 的类型擦除接口，而不是
导出某个模板实例；按值 vector 已由 `hyle_container_vector_*` 覆盖。
