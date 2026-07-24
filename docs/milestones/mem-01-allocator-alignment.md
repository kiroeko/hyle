# mem-01：Allocator vtable + 修 alignment

| 字段 | 值 |
|---|---|
| **阶段** | P2 |
| **模块** | mem/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | ✅ |
| **状态** | ✅ 完成（MSVC 19.50 Debug+Release，9 case 全过；**Linux/GCC14.2 posix_memalign 路径也已真验**）|

> **cloc 留档（`cloc include/hyle/mem src/mem tests/mem`）**：**182 行** = 41 头 + 140 cc（system.cc 裸 backing + allocator_test）+ 1 CMake。
> **远低于估 1-2k，非 stub**——骨架版只做"vtable + 修对齐 + 裸 backing"三件事，故意精瘦：mem/ 的重头（arena/pool/stack/NUMA/大页/fiber栈/stats）全在 **mem-02..06**。本步 backing ~70 行、句柄 inline ~40 行、测试 ~140 行。对齐修复是**真生效**（16/32/64/128/256/4096 六档 `(uintptr_t)p%align==0` 全过），realloc 保数据已验。
> **实现注记**：backing 用裸 `_aligned_malloc`/`posix_memalign`（用户拍板，mimalloc 留 mem-02 换进同一 vtable）。realloc=alloc+copy+free（可移植，mem-02 换 mi_realloc_aligned）。`Allocator` 句柄方法全 inline（纯转发零开销），公开表面只有函数指针+void*。该里程碑的分配热路失败返 nullptr，因而当时没有调用 error API；这只是实现范围，不是依赖禁令。error 是 P0 根，后续 mem 可直接依赖。

## 前置依赖

- 硬前置：`00-scaffolding`（宏/导出）。骨架顺序上接 `platform-01`，但裸 malloc 版**无代码级依赖** platform。
- 决策依据：`mem-design.md`（Allocator 单函数指针 vtable、5 bug 之首=alignment 丢弃、通用 malloc 借/专用造）；`coding-conventions.md` §4（C-ABI vtable）、§6（显式 allocator 注入）。

## 目的（为什么做这一步）

万物分配都走这个注入接缝。clay 的 `Allocator` 是最成熟资产（单函数指针 vtable、C-ABI 稳），但有个**硬 bug**：`allocate(size, alignment)` 静默丢弃 alignment。simd/ 的 AVX512 load/store 要 64B 对齐——不修这个，整条 SIMD 对齐契约（mem-design §90）立不住。这步把注入接缝立起来 + 修对齐。

## 在架构中的位置

`mem/`（T0）是 container/sched/simd/compute 的共同分配地基。本步是 walking skeleton 第四块（`...→ platform-01 → mem-01 → container-01 → sched-01`），给后面的有界 ring、fiber 栈池提供 `Allocator`。arena/pool/NUMA 等是后续 mem-02..06。

## 关键设计决策

### D1：统一 realloc 单函数指针 vtable（继承 clay，签名加 alignment）
- 一个函数承载 alloc/free/realloc：`void* (*)(void* state, void* ptr, size_t old, size_t new, size_t align)`。
  - `ptr==null, new>0` → allocate(new, align)
  - `ptr!=null, new==0` → free(ptr)（返 null）
  - `ptr!=null, new>0` → reallocate
- **修 bug**：clay 签名没带 alignment（`allocator.h:37-40` 丢弃）。HYLE 全程透传 align 到 backing。
- `Allocator` 句柄 = `{fn, state}`，**方法 inline**（纯转发，零开销，热路径不进 DLL）；只有 `system_allocator()` 与 backing 在 DLL。公开表面只有函数指针 + void*，ABI 干净。

### D2：backing 裸对齐 malloc（mimalloc 留 mem-02）—— 偏离文档的刻意选择
- 里程碑索引原写"骨架版含最小 mimalloc 包装"；**用户拍板改为裸 malloc 先行**：先验通 vtable 抽象 + 对齐契约，不引 mimalloc 的 CMake/FetchContent 复杂度。
- backing：Windows `_aligned_malloc`/`_aligned_free`，POSIX `posix_memalign`/`free`。**始终走对齐路径**，free 不依赖 align 值（规避"必须记住对齐"陷阱）。
- mem-02 把 mimalloc 换进**同一个 vtable 后面**，调用方零改动——这正是 vtable 注入接缝的价值。

### D3：失败返回 nullptr（非 Status）
- 分配器是热路径，OOM 返回 nullptr（调用方查），不用 Status（省体积）。因此 mem-01
  的这条热路**不调用 error API**；不能据此推出 mem 模块不得依赖 P0 error。

## 具体事务

- [ ] `include/hyle/mem/allocator.h`：`ReallocFn` 类型、`Allocator` 句柄（inline allocate/deallocate/reallocate）、`kDefaultAlignment`、`HYLE_MEM_API system_allocator()`。
- [ ] `src/mem/system.cc`：裸对齐 backing realloc + `system_allocator()`。
- [ ] **不做**：arena/pool/stack（mem-03/04）、mimalloc（mem-02）、NUMA/大页（mem-06）、pmr adapter、stats。

## 交付物

- allocator.h + system.cc + 测试。最小演示：用 `system_allocator()` 分配 64B 对齐块、写读、释放。

## 验收标准

- **编译**：Windows 真编（MSVC 19.50）；POSIX 走查标未验证。
- **测试**（T0 高要求）：alloc/free/realloc 往返；**64B/128B/256B 对齐真生效**（`(uintptr_t)p % align == 0`）——这是修 bug 的硬证据；realloc 保留数据；size0；多次分配不重叠；压力分配/释放无泄漏（debug）。
- **硬指标**：`allocate(n, 64)` 返回的指针必 64 字节对齐（clay 的 bug 会让它退化到 16B）。

## 参考

- `mem-design.md`（clay `allocator.h:24` vtable、`:37-40` alignment bug、落地清单 1-2）。
- Windows `_aligned_malloc`；POSIX `posix_memalign`。
