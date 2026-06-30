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

mem/simd/io/compute 都借一个通用 substrate（mimalloc/Highway+xsimd/liburing/ggml）。**container/ 不借——自己造**。三条理由：

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
container/                  (实际文件名,扁平布局)
├─ swiss_table.h   flat hashmap(SwissTable:concept 模板 + 控制字节 + SIMD 探测)
├─ carray.h        侵入式动态数组(header-before-data)+ RAII/迭代器薄壳
├─ small_vector.h  inline storage(小 N 不堆分配)—— clay 没有,新增
├─ string_ref.h    string_ref(null_terminated 位)+ string_algo
├─ string_buf.h    string_buf 驻留池(C-ABI 字符串载体)
├─ intrusive_list.h 侵入式 list(节点内嵌 hook)—— 给 sched run-queue/waiter
├─ bounded_queue.h / sharded_queue.h / ws_fifo_queue.h / work_stealing_deque.h / concurrent_vector.h
│                   无锁队列簇 —— 给 sched global/per-core queue + work-stealing
└─ probe.h         SwissTable 的 16 路 SIMD 探测片段(自包含 SSE2/SSSE3/NEON,见 tier 澄清)
```

## SwissTable SIMD 探测（探测层重写，非"补字节"）

> **深读 abseil 校正**：clay 是**线性探测**，SwissTable 是**三角数二次探测的 group 跳跃 + 控制字节 + 特定 tombstone 降级逻辑**——这不是"补 control-byte 就完成"，是**换探测算法 + 重写 resize/erase**。工作量接近"从零造探测核"，落地清单须单列。

- **控制字节数组**：与 slot 平行的 `int8_t ctrl[]`，每字节四种值（照 abseil `hashtable_control_bytes.h:184-219`，**这些值是为 SIMD 精选的，别自改**）：`0x80`(-128)=空、`0xFE`(-2)=删、`0xFF`(-1)=哨兵（kSentinel，`pcmpeqd` 免加载）、`0hhhhhhh`(MSB=0)=full 槽存 hash 高 7 位(H2)。H1=全宽做 group 索引，H2=hash>>57 取高 7 位。
- **探测**：三角数二次探测 `offset += (index += Width)`（**非线性探测**）。每 group：SIMD 加载 16 控制字节，`_mm_set1_epi8(h2)` 广播 + `_mm_cmpeq_epi8` + `movemask` → 16 位候选位图，逐候选比 key。空槽终止探测；erase 时能证明不在任何探测链的删除位可降级为 empty（避墓碑堆积）。
- **三套 group 实现（probe.h 必须全带）**：x86 SSE2 group 宽 16；**ARM NEON group 宽 8**（无 movemask，用 `vceq_u8` + 64 位 mask 方案，复杂度比 x86 高）；**标量 SWAR 兜底**（abseil `GroupPortableImpl`，非 x86/ARM 平台必需，否则编不过）。
- **可借鉴 F14 的溢出计数**（folly）替代裸 tombstone——稳态 insert/erase 自动回收墓碑、免周期 rehash 抖动，对长生命周期 agent 表友好。主体抄 abseil 平坦布局 + 借 F14 溢出计数思想。
- **tier 倒置澄清（重申 simd-design）**：探测片段自包含（三套 group 直接写 intrinsic）、**内联在 `probe.h`、不依赖整个 simd/（Tier 1）**。保持 container（T0）不依赖 simd（T1）。

## 新增类型（clay 缺、HYLE 焦点要的）

| 类型 | 用途 | 谁要 | 参考/来源 |
|---|---|---|---|
| **SmallVector<T,N>** | inline storage，小集合零堆分配 | 通用热路径（参数列表、临时集合）| 抄 **folly `small_vector`** 设计（MaxInline 策略 + size 压进 inline buffer 省头部，比 absl::InlinedVector 更抠字节），不抄代码 |
| **intrusive list/slist** | 节点内嵌 hook，零额外分配、O(1) 摘除 | **sched 的 waiter 链、run-queue** | 参考 **folly `IntrusiveList`**（boost-free，比 boost.intrusive 轻）|
| **lock-free bounded ring（SPSC/MPMC）** | 有界无锁环 | **sched 的 per-core run-queue + global 注入** | **vendoring 成熟实现**（Vyukov bounded MPMC：数组+每槽 seq；rigtorp SPSC），**不自造**——见下张力 |

> **lock-free 队列造借结构（按实现校正）**：分两类。**基础有界 ring（SPSC/MPMC）借/vendoring**——出名难写对（ABA、memory order、TSan），且无 C-ABI 理由必须自造，故 vendor Vyukov MPMC / rigtorp SPSC（皆单头可内联，container-01），自造预算集中到差异焦点 hashmap。**但 work-stealing 变体自写**——sched/graph 要的 owner-LIFO + steal-half + packed-head CAS 语义现成件不提供，故 container-07（Chase-Lev deque，供 graph fork-join）、container-08（tokio 式 FIFO ring，供 sched 本地队列）移植自写。即 container "自造"边界 = hashmap + carray + string + work-stealing 队列；基础 ring 借。

## 与其他模块的耦合（container 是 sched 的原语供应商）

- **→ sched/（关键）**：sched 第二节的 per-core 有界 ring、global MPMC 注入队列、fiber waiter 链，**全是 container 提供的数据结构原语**。依赖故事干净：**container（T0）提供原语 → sched（T0）组合成调度器**。
- **→ mem/**：所有容器显式注入 `Allocator*`（继承 clay 铁律），分配走 mem/。
- **→ simd/**：仅 SwissTable 探测用自包含片段，不依赖整个 simd/（见上 tier 澄清）。
- **string_buf 跨 DLL**：make/destroy + 引用计数，是 C-ABI 边界的字符串载体。

## 落地状态(已实现)

1. **hashmap**:`SwissTable`(concept 模板 + 控制字节 + SIMD 探测,去 clay 的 `temp`/哨兵)——✅ 对拍 std + 对标 absl(find/insert/erase 平或反超)。
2. **carray**:`CArray<T>` RAII 壳(header-before-data 布局保留)——✅(迭代器 begin/end 已具)。
3. **SmallVector**(inline→heap 切换)、**intrusive list**、**lock-free 队列**(`BoundedQueue` Vyukov MPMC / `ShardedQueue` / `WsFifoQueue` / `WorkStealingDeque` / `ConcurrentVector`)——✅ 全部实现 + WSL TSan 零 race。
4. **string_ref/string_buf**——✅ 继承实现。
5. **单测**:hashmap 对拍 std + 对标 absl bench;lock-free 队列 TSan;SmallVector inline→heap 切换——✅ 均已覆盖。

## 待决（需主拍板）

1. **hashmap 保持 C 宏泛型还是 C++23 concept 模板重写**？**✅ 已决：concept 模板写内部实现 + 需过边界的特定类型封薄 C-ABI shim**——去掉 C 宏的 temp/哨兵/sizeof 校验包袱，干净类型安全；hashmap 绝大多数是内部 C++ 用，过边界的少数（如 string 映射）再封 shim。即 charter「内部 C++23 / 外部 C-ABI」在容器上的落地。
2. **SwissTable 自实现还是抄 abseil 内部**？建议**自实现**（C-ABI 边界 + 差异焦点 + clay 已规划）——这是唯一不外包的 Tier 0，参考 abseil 设计笔记但代码自写。
3. **lock-free 队列放 container/ 还是 sched/**？建议放 **container/**（它是通用数据结构，sched 是消费者），保持"container 供原语、sched 做组合"的干净分层。

## 参考

- [abseil Swiss Tables 设计笔记](https://abseil.io/about/design/swisstables)
- [abseil flat_hash_map 实现](https://github.com/abseil/abseil-cpp/blob/master/absl/container/flat_hash_map.h)
- [folly F14 vs absl flat_hash_set 基准](https://groups.google.com/g/hashtable-benchmarks/c/-nT-3fESXq8)
- [SwissTable 快速描述（控制字节 + SIMD 探测）](https://faultlore.com/blah/hashbrown-tldr)
