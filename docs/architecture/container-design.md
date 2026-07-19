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
    pointer-payload concurrent facades, TLS last_error
                      |
          +-----------+-----------+
          |                       |
          v                       v
C++ typed / source surface      compiled cold cores
  CArray<T> / SmallVector       hash_bytes, carray grow,
  SwissTable<K,V>               Swiss resize/free,
  string_ref / StringBuf        StringBuf/path/string algorithms
  typed queues                  runtime ISA dispatcher
          |                       |
          +-----------+-----------+
                      |
                      v
Tier 0 dependencies: mem allocator + platform CPU discovery
```

- `container -> mem -> platform` 是允许的基础依赖。
- container 不链接 `hyle_simd` 模块，避免 Tier 0→Tier 1 倒挂。
- 字符串批处理的每 ISA TU 复用 `hyle/simd/vec.h` 的 header-only 寄存器 DSL；
  这不会调用 SIMD runtime，也不会把 SIMD 公共 ABI带入 container。
- Swiss control group 仍在 `probe.h` 自包含实现：x86 16-byte group、NEON/portable
  8-byte group。它必须可被 typed `SwissTable` hot path 内联。

## 数据结构面

| 能力 | C++ 源码面 | 稳定 C ABI | 关键合同 |
|---|---|---|---|
| flat map | `SwissTable<K,V>` | erased POD hash map | trivially-copyable key/value；ctrl+连续 slots；SIMD group probe |
| dynamic array | `CArray<T>` | erased POD vector | header-before-data；allocator/alignment 随 allocation 记录 |
| inline array | `SmallVector<T,N>` | 不单独导出 | N 是编译期类型；C 动态需求由 erased vector 覆盖 |
| string view | `string_ref` | `hyle_container_string_view` | ptr+u32 length+NUL bit；借用、不拥有 |
| interner | `StringBuf` | opaque StringBuf | generation-tagged direct ID；内容相等才同一条目 |
| path/text algorithms | typed view/buffer-fill | pointer/count/buffer-fill | 无 STL 所有权跨边界 |
| bounded/sharded queues | typed payload | `void*` payload opaque handle | HYLE 不拥有 payload |
| work steal/share | typed payload | `void*` payload opaque handle | owner/thief 契约不同，不能混用 |
| concurrent vector | typed append-only | `void*` append-only | 发布后地址稳定 |

IntrusiveList 已下沉 core，因为 error/time/sched 都需要它；它不再属于 container
库或 container C ABI。

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
每 probe 一次函数指针。C map 使用同一 ctrl/group/resize 机制，但 slot layout
运行时计算为 `[stored hash][aligned key][aligned value]`；不是每元素分配的链表。

边界原则：

- 所有容量乘法、对齐和 power-of-two 扩张先做 overflow 检查。
- 增长失败保留旧状态并返回失败。
- insert 输入若别名当前 slots，增长前复制到 allocator-backed scratch。
- C callback 不得抛异常或在遍历中修改同一 map。

## StringBuf 身份与生命周期

StringBuf 有两个正交用法：

```text
add -> InternId(generation, slot) -> retain/remove -> refs=0 时释放
intern / intern_many             -> permanent until pool destruction
```

内容 hash 只负责定位，不是身份。`by_string` 必须继续做完整内容 equality；ID 通过
`by_id` 直接索引，generation 防止 slot 重用后 stale ID 命中新字符串。每个 Entry
与字符数据在一个 allocator allocation 中，地址稳定。pool、索引、entry、C handle
都使用注入的同一个 allocator。

单次 API 为线程安全每次取锁；吞吐用户应使用 `intern_many`。单次生命周期 churn
不是该模块的 SOTA 宣传点，性能结论见
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

`include/hyle/c/container.h` 遵守 close-01：

- 106 个 `extern "C"` 动态符号；15 个 header-inline 便利入口。
- options 带 `struct_size`；allocator 按值捕获但 state 生命周期由调用方保证。
- 对象只经 create/destroy；不导出 C++ 类布局、STL、模板或异常。
- String/path 产出使用借用 view 或 buffer-fill。
- CArray/SwissTable 通过运行时 item size/alignment/hash/equal 类型擦除。
- 并发容器的稳定最小共同面使用 pointer payload。

完整使用和失效规则见 [`c-container.md`](../guide/c-container.md)；收口证据见
[`container-module-closure.md`](../milestones/container-module-closure.md)。

## 下游利用

- sched 使用 ShardedQueue、WsFifoQueue、WorkStealingDeque 等调度原语。
- graph/sched 使用 ConcurrentVector 类 append-only 发布结构。
- error/time 等共享 intrusive 基础设施已归 core，不制造反向 container 依赖。
- 其他模块可使用 `hash_bytes`，但 hash 模块不依赖 SIMD runtime；≤16B typed key
  走 inline mix，热短字节串可显式选择 `hash_inline.h` 消掉 DLL seam。

## 性能验收边界

“SOTA”必须逐契约、逐操作成立，不能写成整个 container 的无条件标签：

- flat Swiss map 对 std 和 absl 的已测核心操作成立；
- CArray/SmallVector/队列按各自对等形状判断；
- 路径的大倍数主要来自零分配结构，对同形 string-view 扫描通常是平局；
- StringBuf batch throughput 强，单调用/生命周期 churn 有锁和 DLL seam 成本；
- 4KB 字符扫描的 0.8x–0.95x 点值应结合 64KB/1MB、方差和调用形状，不包装成赢。

测试矩阵包括 typed 单测、纯 C 121/121 调用覆盖、C++ C-ABI 并发测试、allocator
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
