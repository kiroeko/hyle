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
  是原地操作（`hyle_container_path_normalize`、
  `hyle_container_string_capitalize`）。
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
  st = hyle_container_vector_push(v, &x);
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
`hyle_container_string_escape`、`hyle_container_string_capitalize`、
`hyle_container_utf8_length`、`hyle_container_utf8_icmp`。
`hyle_container_utf8_length` 对非法 UTF-8 返回 -1。quick-open fuzzy ranking 是
应用搜索策略，不是容器/FFI 基元；严格复审已从本模块删除。

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

## StringBuf

`hyle_container_string_buf_create` /
`hyle_container_string_buf_create_system` 通过 Status + out 创建线程安全驻留池，
`hyle_container_string_buf_destroy` 销毁。

- 可回收生命周期：`hyle_container_string_buf_add` 把 pool-local、带 generation
  的 `hyle_container_intern_id` 写入 out；用 `hyle_container_string_buf_retain`、
  `hyle_container_string_buf_remove`、`hyle_container_string_buf_lookup` 管理。
  删除后的旧 ID 永远不会指向后来复用该 slot 的字符串。
- 批量生命周期：`hyle_container_string_buf_add_many`、
  `hyle_container_string_buf_retain_many`、
  `hyle_container_string_buf_remove_many` 处理批量生命周期。分配型 batch 通过
  `out_completed` 保留已完成前缀；C shim 以最多 64 项为一块，把 shard 锁和动态
  ABI 接缝摊薄。非零 count 要求输入/输出数组有效。
- 永久驻留：`hyle_container_string_buf_intern` 和
  `hyle_container_string_buf_intern_many` 返回稳定、带尾零的借用视图，存活到池
  销毁。吞吐场景优先 batch，它只获取一次 shard lock set。
- `hyle_container_string_buf_size` 返回当前不同字符串数。所有返回视图都由池拥有，
  不得释放。

## 运行时擦除 vector

用 `HYLE_CONTAINER_VECTOR_OPTIONS_INIT` 或
`HYLE_CONTAINER_VECTOR_OPTIONS_FOR` 初始化带 `struct_size` 的 options。
元素必须可按字节复制；C++ 非平凡类型不能放入该容器。

入口：
`hyle_container_vector_create`、`hyle_container_vector_destroy`、
`hyle_container_vector_size`、`hyle_container_vector_capacity`、
`hyle_container_vector_data`、`hyle_container_vector_data_const`、
`hyle_container_vector_at`、`hyle_container_vector_reserve`、
`hyle_container_vector_push`、`hyle_container_vector_pop`、
`hyle_container_vector_clear`。

其中 `create` / `reserve` / `push` 返回 Status；`pop` 的 0 只是容器为空。

任何可能增长的操作都可能使先前由 `data`/`at` 返回的地址失效。`push` 支持 item
来自当前 vector 内部，即使增长也不会读悬空地址。

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
`hyle_container_hash_map_erase`、`hyle_container_hash_map_for_each`。

这是 control-byte + SIMD group probing 的平坦 Swiss map，不是链式 C facade。
`capacity` 是 slot 数，最大装载量较小；增长/erase 会使返回的 key/value 地址失效。
`create` / `reserve` / 两种 insert 返回 Status。和 typed 面一致，`insert` 在 key 已
存在时保持原 value，并返回 `HYLE_CONTAINER_ALREADY_PRESENT`；只有
`insert_or_assign` 会覆盖，并返回 `HYLE_CONTAINER_ASSIGNED`。新 key 都返回
`HYLE_CONTAINER_INSERTED`。两者都支持 key/value 指向当前 map slot，growth 前会先
保存借用字节。
`for_each` 的 visitor 返回 0 可提前停止；遍历期间不得修改 map。

## 并发容器

`hyle_container_bounded_queue_create`、
`hyle_container_bounded_queue_create_system`、
`hyle_container_bounded_queue_destroy`、
`hyle_container_bounded_queue_push`、
`hyle_container_bounded_queue_pop`、
`hyle_container_bounded_queue_capacity` 是有界 MPMC ring。

`hyle_container_sharded_queue_create`、
`hyle_container_sharded_queue_create_system`、
`hyle_container_sharded_queue_destroy`、
`hyle_container_sharded_queue_push`、
`hyle_container_sharded_queue_pop`、
`hyle_container_sharded_queue_capacity`、
`hyle_container_sharded_queue_shard_count` 是高争用 MPMC 组合；producer hint 选分片，
consumer cursor 应由各 consumer 私有并在多次 pop 间保留。

`hyle_container_work_stealing_deque_create`、
`hyle_container_work_stealing_deque_create_system`、
`hyle_container_work_stealing_deque_destroy`、
`hyle_container_work_stealing_deque_push`、
`hyle_container_work_stealing_deque_pop`、
`hyle_container_work_stealing_deque_steal`、
`hyle_container_work_stealing_deque_capacity` 是单 owner push/pop、多个 thief steal
的 Chase-Lev 风格 deque。steal 返回
`HYLE_CONTAINER_STEAL_OK/EMPTY/ABORT`；ABORT 表示竞争，应重试。

`hyle_container_ws_fifo_queue_create`、
`hyle_container_ws_fifo_queue_create_system`、
`hyle_container_ws_fifo_queue_destroy`、
`hyle_container_ws_fifo_queue_push`、
`hyle_container_ws_fifo_queue_pop`、
`hyle_container_ws_fifo_queue_steal_into`、
`hyle_container_ws_fifo_queue_capacity` 是 owner FIFO + 批量转移队列；
`steal_into` 要求两个不同且等容量的队列。

`hyle_container_concurrent_vector_create`、
`hyle_container_concurrent_vector_create_system`、
`hyle_container_concurrent_vector_destroy`、
`hyle_container_concurrent_vector_push`、
`hyle_container_concurrent_vector_size`、
`hyle_container_concurrent_vector_capacity`、
`hyle_container_concurrent_vector_at` 是 append-only `void*` vector。`push` 返回
Status 并写 `out_index`；已经发布的元素地址不移动，调用者仍负责 payload 生命周期。
多个 `push` 可真正并行；`size` 返回已领取 index 数，不是 in-flight push 的完成栅栏。
读取新 tail 前必须等待对应 push 返回或等待所有 writers quiesce；并发 append 期间可读
事先完成的 prefix。

所有并发容器的 `create` 都是 Status + out；创建完成后，满/空/竞争仍走原来的
无分配热路径返回值。`void*` payload 可为 NULL，HYLE 不解释也不拥有它。

## C++ 形状在 C 中怎么取得

不导出模板实例或 `std::vector`，也不冻结内部类布局；按能力使用下列动态形状：

| C++ 形状 | C/FFI 通路 |
|---|---|
| `SmallVector<T,N>` / `PodVector<T>` | `hyle_container_vector_*`：options 传 `element_size`/`alignment`，按值存储 trivially-copyable 元素。C ABI 不接收 C++ ctor/dtor callback；inline `N` 也不是稳定跨 ABI 属性 |
| `FlatHashMap<K,V>` | `hyle_container_hash_map_*`：C++ map 可拥有 non-trivial 值；C/FFI map 则按 options 中的 size/alignment 擦除 POD，hash/equality 由 callback+ctx 注入 |
| `BoundedQueue<T>` / `ShardedQueue<T>` / `WorkStealingDeque<T>` / `WsFifoQueue<T>` | 对应 `hyle_container_*queue*` / `*_deque_*` opaque 句柄接受 `void*` payload；payload 生命周期归调用方 |
| `ConcurrentVector<T>` | `hyle_container_concurrent_vector_*` append `void*`；调用方保存 typed payload |
| `Hash<T>` / `hash_bytes` | C 侧对实际字节调用 `hyle_container_hash_bytes`，或向 flat map 注入业务 hash callback；不暴露 vendor 实现头 |

这张表表达的是“C++ 静态形状不跨 ABI、运行时能力仍可达”。如果未来需要
按值 POD queue，应该新增 `element_size`/`alignment` 的类型擦除接口，而不是
导出某个模板实例；按值 vector 已由 `hyle_container_vector_*` 覆盖。
