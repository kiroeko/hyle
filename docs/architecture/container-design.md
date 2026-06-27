# HYLE container/ 设计

> 超越 std 的高性能容器（Tier 0，焦点 #5）。继承 clay flat hashmap/carray/string_ref，补 SwissTable SIMD 探测，加 SmallVector/intrusive/lock-free 队列。上游见 [`hyle-scope.md`](./hyle-scope.md)、[`coding-conventions.md`](./coding-conventions.md)。

## clay 现状：真地基，成色好

| 资产 | 行数 | 性质 | 处置 |
|---|---|---|---|
| flat `hashmap`（C 宏泛型 open-addressing）| 715 | power-of-two 位与取模、线性探测+tombstone、keys/values 连续。**规划了 SwissTable SIMD 探测但未实现**（`hashmap.h:109-131`）| 改后继承 + **补 SwissTable** |
| `carray`（header-before-data 侵入式数组）| 423 | 指针即数据、cache 友好、可传 C API | 改后继承 + RAII 壳 |
| `cstring`/`string_ref` | 1003 | ptr+len+null_terminated 位、零拷贝、interning | 直接继承 |
| `string_buf`（引用计数驻留池）| — | murmur hash + make/destroy 跨 DLL | 直接继承 |
| `concurrent_vector` | — | 读优化、写加锁 | 改后继承（lock-free 化）|

## 核心决策：唯一真"造不借"的 Tier 0

mem/simd/io/compute 都借一个通用 substrate（mimalloc/sleef/liburing/ggml）。**container/ 不借——自己造**。三条理由：

1. **clay 有真地基**：flat hashmap + carray + string_ref 已是 absl/folly 雏形，改造成本 < 引入外部依赖。
2. **C-ABI 边界过不去**：abseil `flat_hash_map`、folly F14 是 C++ 类型、带 C++ ABI，**跨 DLL/FFI 过不去**——这正是 clay 当年造 carray/c-hash 的原因（也是 mem-design 里 mimalloc 要封在 vtable 后的同款约束）。
3. **是差异焦点 #5**："超越 std 的容器"是卖点，外包卖点说不过去。

所以 container/ = **继承 clay + 补 clay 规划未做的 + 填缺的类型**，不引入外部容器库。

## 2026 格局（指导 SwissTable 实现）

- **SwissTable（abseil）**：密集 metadata 数组，每槽 1 控制字节 = 7 位 hash 片段 + 空/删标志；查找时 SIMD 一次比 16 个控制字节（`_mm_cmpeq_epi8`+movemask）定位候选，再比 key。cache 友好 + 并行探测。
- **F14（folly）**：分 chunk，同样 SIMD。
- clay 的 hashmap 已是 open-addressing flat，**补上 control-byte + SIMD 探测就到 SwissTable 水平**——不是推倒，是完成。

## 分层结构

```
container/
├─ hashmap.h       flat hashmap：改 C 宏泛型 → C++23 concept 模板，补 SwissTable 控制字节 + SIMD 探测
├─ carray.h        侵入式动态数组（继承 header-before-data）+ RAII/迭代器薄壳
├─ small_vector.h  inline storage（小 N 不堆分配）—— clay 没有，新增
├─ string.h        string_ref（继承，null_terminated 位）+ string_buf 驻留池
├─ intrusive/      侵入式 list/slist（节点内嵌 hook）—— 给 sched run-queue/waiter
├─ lockfree/       MPMC/SPSC 队列、bounded ring —— 给 sched global queue/per-core queue
└─ probe.h         SwissTable 的 16 路 SIMD 探测片段（自包含 SSE2/NEON，见 tier 澄清）
```

## SwissTable SIMD 探测（完成 clay 的规划）

- **控制字节数组**：与 slot 平行的 `int8_t ctrl[]`，每字节：`0x80`=空、`0xFE`=删、`0x00-0x7F`=hash 高 7 位。
- **探测**：取 key hash 低位定 group（16 字节对齐），SIMD 加载 16 控制字节，`_mm_cmpeq_epi8(group, broadcast(h7))` + `movemask` → 候选位图，逐候选比 key。一条 SIMD 扫 16 槽，cache 命中率高。
- **tier 倒置澄清（重申 simd-design）**：探测只需自包含的 SSE2/NEON 片段（16 字节比较 + movemask + 标量 NEON 等价），**内联在 `probe.h`、不依赖整个 simd/（Tier 1）**。保持 container（T0）不依赖 simd（T1）。simd/ 成熟后可选复用，不强耦合。

## 新增类型（clay 缺、HYLE 焦点要的）

| 类型 | 用途 | 谁要 |
|---|---|---|
| **SmallVector<T,N>** | inline storage，小集合零堆分配 | 通用热路径（参数列表、临时集合）|
| **intrusive list/slist** | 节点内嵌 hook，零额外分配、O(1) 摘除 | **sched 的 waiter 链、run-queue** |
| **lock-free MPMC 队列** | 多生产者多消费者无锁 | **sched 的 global 注入队列** |
| **bounded SPSC/MPMC ring** | 有界无锁环 | **sched 的 per-core run-queue**（sched-design 2.2）|

## 与其他模块的耦合（container 是 sched 的原语供应商）

- **→ sched/（关键）**：sched 第二节的 per-core 有界 ring、global MPMC 注入队列、fiber waiter 链，**全是 container 提供的数据结构原语**。依赖故事干净：**container（T0）提供原语 → sched（T0）组合成调度器**。
- **→ mem/**：所有容器显式注入 `Allocator*`（继承 clay 铁律），分配走 mem/。
- **→ simd/**：仅 SwissTable 探测用自包含片段，不依赖整个 simd/（见上 tier 澄清）。
- **string_buf 跨 DLL**：make/destroy + 引用计数，是 C-ABI 边界的字符串载体。

## 落地清单

1. hashmap：C 宏泛型 → concept 模板重写（去掉 clay 的 `temp` 字段/哨兵/编译期 sizeof 校验技巧），补 SwissTable 控制字节 + SIMD 探测。
2. carray RAII 壳 + 迭代器（保留 header-before-data 布局）。
3. SmallVector、intrusive list/slist、lock-free MPMC/SPSC ring（sched 依赖，优先）。
4. string_ref/string_buf 继承。
5. 单测：hashmap 对拍 std::unordered_map 正确性 + benchmark 对标 absl/F14；lock-free 队列 TSan + 线性化测试；SmallVector 的 inline→heap 切换。

## 待决（需主拍板）

1. **hashmap 保持 C 宏泛型还是 C++23 concept 模板重写**？**✅ 已决：concept 模板写内部实现 + 需过边界的特定类型封薄 C-ABI shim**——去掉 C 宏的 temp/哨兵/sizeof 校验包袱，干净类型安全；hashmap 绝大多数是内部 C++ 用，过边界的少数（如 string 映射）再封 shim。即 charter「内部 C++23 / 外部 C-ABI」在容器上的落地。
2. **SwissTable 自实现还是抄 abseil 内部**？建议**自实现**（C-ABI 边界 + 差异焦点 + clay 已规划）——这是唯一不外包的 Tier 0，参考 abseil 设计笔记但代码自写。
3. **lock-free 队列放 container/ 还是 sched/**？建议放 **container/**（它是通用数据结构，sched 是消费者），保持"container 供原语、sched 做组合"的干净分层。

## 参考

- [abseil Swiss Tables 设计笔记](https://abseil.io/about/design/swisstables)
- [abseil flat_hash_map 实现](https://github.com/abseil/abseil-cpp/blob/master/absl/container/flat_hash_map.h)
- [folly F14 vs absl flat_hash_set 基准](https://groups.google.com/g/hashtable-benchmarks/c/-nT-3fESXq8)
- [SwissTable 快速描述（控制字节 + SIMD 探测）](https://faultlore.com/blah/hashbrown-tldr)
