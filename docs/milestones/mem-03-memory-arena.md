# mem-03：MemoryArena 无锁定长块池（clay 最亮资产，packed tagged CAS）

| 字段 | 值 |
|---|---|
| **阶段** | P2 |
| **模块** | mem/ |
| **Tier** | 0 |
| **LOC 估** | 2-3k |
| **骨架?** | |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，14 mem case 全过；**TSan 零 data race**）|

> **cloc 留档（`cloc memory_arena.h memory_arena_test.cc`）**：**209 行** = 91 头 + 118 测试。header-only(热路径内联,符合 A 范式,非模板无需 .cc),非 stub。
> **实现/自审**：① **packed 64 位 tagged head**(index<<32|version)纯 64 位 CAS,**去掉 clay 的 16 字节 atomic + cmpxchg16b/-mcx16 依赖**(mem-design 附注隐患);version 防 ABA。② **修 clay bug #5**:block_count 构造参数,实测 1000 块远超 64 上限。③ freelist `next` 用**并行索引数组**(非 clay 的块内嵌入)——避免 std::byte slab 上 atomic_ref 的生命周期/对齐纠葛,TSan 干净,代价 4 字节/块(定长池可忽略)。④ allocate/deallocate CAS 用 acq_rel/acquire(保守正确:release 分量发布 next 写、acquire 见前次 free 的发布)。
> **8 线程×10 万 alloc-写marker-free 压力**:无两线程持同块(marker 独占校验)+ 无丢块(终局全 256 块可回收),Win 263ms,TSan 零 race。
> **顺序决策**：本步排在 mem-02(mimalloc)**之前**——先吸纳 clay 自有代码(MemoryArena 在任何 backing 上工作、不依赖 mimalloc),外部依赖后置。catomics **不 port**(C++23 std::atomic 取代,charter 已定)。

## 前置依赖

- 硬前置：`mem-01`（slab 走 `Allocator` 注入）。**不依赖 mem-02**（mimalloc）——MemoryArena 在任何 backing 上都工作。
- 决策依据：`mem-design.md`（MemoryArena 无锁块池=最亮资产、5 bug 之 #5=64 槽上限、#附注=16字节 atomic 需 cmcx16）；`clay-asset-liability.md` 行30/43/45。

## 目的（为什么做这一步）

clay 最亮的资产：**无锁定长块池**（tagged-pointer CAS 防 ABA）。定长分配（fiber 栈、固定结构体、池化对象）比通用 malloc 快得多——这是 HYLE"极致内存"的发力点之一。本步把它吸纳进来并**修两个 bug**（64 槽上限 + 16 字节 atomic 依赖），用 WSL TSan 验无锁正确性。

## 在架构中的位置

mem/（T0）的定长分配器，sched fiber 栈池、ObjectPool 等的底座。**排在 mem-02（mimalloc）之前**：先吸纳 clay 自有代码，外部依赖后置（维护者决策，符合"mimalloc 留后"）。

## 关键设计决策

### D1：packed 64 位 tagged head，不用 16 字节 atomic（改进 clay）
- clay 用 `std::atomic<Head{ptr,tag}>`（16 字节），依赖 `cmpxchg16b`/`-mcx16`，否则**静默退化加锁**（mem-design 附注点名的隐患）。
- HYLE 改：freelist head = **一个 `atomic<uint64_t>` = `(index << 32) | version`**。块用**索引**（非指针）串 freelist（块内嵌 `next` 索引）。纯 64 位 CAS，**lock-free 处处成立、无需 -mcx16**。version 每次 push 递增防 ABA。
- 代价：块数 ≤ 2^32（远超实际）；块靠 slab base + index*stride 寻址。

### D2：修 clay bug #5——64 槽上限
- clay `make_arena_allocator` 从固定 64 槽 static 池取对象，超 64 个并发 arena 返回 null。HYLE：`MemoryArena(alloc, block_size, block_count)` 构造参数,块数由调用方定,slab 一次性从注入 Allocator 取 `block_size*block_count`。

### D3：Treiber stack freelist，块内嵌 next
- 初始化：slab 切成 block_count 个块,每块头 4 字节存 `next` 索引,串成 0→1→...→空。head 指 0。
- `allocate()`：CAS pop 栈顶(读 head.index → 该块的 next → CAS head 到 next，version+1)。耗尽返 null。
- `deallocate(p)`：CAS push(块的 next = 当前 head.index → CAS head 指向该块，version+1)。
- 块对齐:slab 按 block 对齐分配,block_size 向上取整到对齐。

## 具体事务

- [ ] `include/hyle/mem/memory_arena.h`：`MemoryArena`（构造 `(Allocator, block_size, block_count, alignment)`、`allocate()->void*`、`deallocate(void*)`、`block_size()`/`capacity()`）。non-copyable/non-movable。
- [ ] **不做**：Arena bump 分配器（同文件可后补或 mem 另一步）、ObjectPool/StackAllocator 重写（mem-04）、mimalloc（mem-02）、NUMA/大页（mem-06）。本步聚焦无锁块池。

## 交付物

- memory_arena.h + 测试（含 TSan）。

## 验收标准

- **编译**：Win x64（MSVC）+ Linux x64（GCC14.2）。
- **测试**：① 单线程 alloc/free 往返、耗尽返 null、free 后复用；② 对齐(返回的块满足 alignment)；③ 块不重叠；④ **并发压力**：N 线程各反复 alloc-写marker-free，验**无两线程同时拿到同一块**(marker 独占) + 无丢块(最终全部可回收)。
- **正确性闸门**：**WSL TSan 零 data race**（纯 64 位 CAS,无 fence 盲区,TSan 覆盖可靠）。
- **硬指标**：8 线程各 10 万次 alloc/free，无重叠无丢块，TSan 干净；block_count 远超 64（验 bug #5 已修）。

## 参考

- `mem-design.md`（MemoryArena、bug #5、cmpxchg16b 附注）。
- clay `sys/memory_arena.cc`（tagged CAS 思路，存储换 packed index+version）。
- Treiber stack；ABA via version tag。
