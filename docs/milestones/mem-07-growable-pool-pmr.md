# mem-07：GrowablePool + pmr adapter

| 字段 | 值 |
|---|---|
| **阶段** | P2 |
| **模块** | mem/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；mem +10 test）|

> **范围**：吸纳 clay `utils/memory_pool.h`(170) + `utils/stl_allocator.h`(196) + `utils/generic_allocator.h`(62)。
> **cloc**：**生产 145 行**(pmr.h + growable_pool.h) + **测试 116 行**。

> **后续替代说明（2026-07-26）：**本页记录 mem-07 当时的选择，不再是完整标准 adapter 真源。
> 当前产品另外提供 empty/always-equal 的 `SystemStdAllocator<T>` 与运行时注入的
> `StdAllocator<T>`，PMR 也分为默认 `system_memory_resource()` 和注入 `PmrResource`；见
> [`mem-design`](../architecture/mem-design.md) 与 [`guide/mem`](../guide/mem.md)。本页的
> `GrowablePool<T>` 是**固定大小 typed slot** 的 growable pool，不是 arbitrary-size heap。

## clay 三文件 → HYLE 去向(逐项)

| clay 文件 | 行 | HYLE 去向 |
|---|---|---|
| `utils/generic_allocator.h` | 62 | ⛔ **无对应物**:HYLE `Allocator` 值句柄(`{fn,state}` type-erased 转发)**已经就是**它。无新代码 |
| `utils/stl_allocator.h` | 196 | ✅ **`mem::PmrResource`**:`std::pmr::memory_resource` 包 Allocator。一个 resource 服务**所有** T(pmr 擦除元素类型),取代 clay 每类型 `stl_allocator<T>` 的 rebind 样板 |
| `utils/memory_pool.h` | 170 | ✅ **`mem::GrowablePool<T>`**:growable + single-threaded 池(填补 ObjectPool 固定+lock-free 之外的空位)|

## GrowablePool 设计 + 修 clay bug

- **niche**:ObjectPool(固定/lock-free,已知上界+多线程共享)vs **GrowablePool(无界/单线程无原子,更快,per-thread/fiber 未知规模)** vs Arena(bump,批量释放)。三者不重叠。
- **块设计**:几何增长块(O(log n) 次分配),侵入式 free-list(空闲槽在自身存储里存 next-free 指针,零 per-slot 开销),块链表供析构回收,每块记 byte 大小(几何增长块大小不一)。
- **修 clay 真 bug**:clay move-assign(memory_pool.h:61)swap 块链表但**从不转移 allocator_** → move-assigned 池用陈旧/null allocator 释放块。HYLE move 转移**全部**字段(真测覆盖)。
- **生命周期契约**:池跟踪 live **计数**而非**哪些槽** live,故析构不能跑剩余对象析构。非平凡 T 必须 destroy() 所有 create() 的;泄漏触发 fail-fast assert(跳过的析构是 bug)。平凡 T 可留。

## PmrResource 设计

- **边界 throw(conventions §0/§5 认可例外)**:`do_allocate` 失败 throw `std::bad_alloc`——这是 memory_resource 文档契约,std::pmr 容器依赖它报 OOM。是 std 适配边界(std 期待 throw),非 HYLE 内部错误路径。
- **do_is_equal**:两 resource 可互换 iff 同 `{fn,state}`(一个分配的能被另一个释放)。

## 对抗式自审（六类）

- **① 正确性**:GrowablePool 跨多块增长指针稳定真测、free-list 复用(同地址)真测、非平凡析构每次真测、move 转移所有权(修 clay bug)真测、交错 churn live 一致真测;PmrResource 服务 pmr::vector/多类型/equality 真测。
- **② 性能**:GrowablePool 单线程无原子(比 lock-free ObjectPool 快,uncontended);free-list O(1);几何增长摊还分配。PmrResource 纯转发。
- **③ 规范**:头-only 模板(消费端实例化,不出 DLL ABI);Allocator 注入(无隐藏堆);GrowablePool 无 std 拥有容器;PmrResource throw 是认可的 std 适配边界例外(已文档)。
- **④ clay 覆盖**:三文件逐项映射(⛔/✅/✅),memory_pool 的 STL-allocator 接口意图由 PmrResource 现代化覆盖。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:GrowablePool 块+free-list 是标准池设计;PmrResource 4 方法极简。
- **"比 std 快"**:GrowablePool vs `std::pmr::unsynchronized_pool_resource`——HYLE 无原子单线程路径 + 类型化 create 更轻;对标待 bench-02。PmrResource 是**让 std 容器用 HYLE 分配器**的桥,非对标项。

## 交付物

- `include/hyle/mem/pmr.h`（PmrResource:std::pmr::memory_resource 包 Allocator）
- `include/hyle/mem/growable_pool.h`（GrowablePool<T>:几何块 + free-list,修 move-assign bug）
- `tests/mem/pmr_test.cc`（4）+ `tests/mem/growable_pool_test.cc`（6）

## 参考

- clay `utils/memory_pool.h`/`stl_allocator.h`/`generic_allocator.h`。C++17 `std::pmr`。conventions §1(禁 std 垃圾容器)、§6(Allocator 注入)。mem-03/04(ObjectPool/Arena 对比)。
