# container-03：carray header-before-data + RAII 壳

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **真实 LOC** | 224（106 头 + 118 核+测试） |
| **骨架?** | 否 |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，container 27→36 case）|

## 前置依赖

- 硬前置：`mem-02`（mimalloc backing 的 `Allocator`——carray 显式注入分配器）。

## 目的（为什么做这一步）

carray 是 clay 的 C-ABI 可传连续数组：handle 直接指向元素、`arr[i]` 零间接，能直传 C API（不像 `std::vector` 的不透明类型过不了 C 边界）。是 container T0 里 string_buf 等的底层载体。本步吸纳并**修正 clay 的过对齐 bug**：clay header 固定 `max_align_t` 对齐，64B cacheline / AVX512 元素会错位——这正是 mem/simd 热路径要的对齐保证的前提。

## 在架构中的位置

T0 容器。采 conventions「type-erased core + typed shell」：核 `carray_set_capacity_internal(void*, item_size, align, alloc)` 只编一份进 DLL（防 per-T 膨胀），`CArray<T>` 是 header-only 类型壳。

## 具体事务（做什么）

- **吸纳 clay `sys/carray.h`(436)+`.cc`(65)**：header-before-data 布局——handle 指向元素，`CArrayHeader{capacity,size,header_offset,align,alloc}` 紧贴其前。
- **修过对齐 bug**：用 `header_offset`（向上取整到 `max(align, alignof(header))`）→ 元素指针精确按 align 对齐，而非 clay 的固定 `max_align_t`。
- **typed 壳 `CArray<T>`**：push_back/pop_back/reserve/clear/operator[]/迭代器/move/初始化列表。T 限 trivially-copyable（memcpy 增长，同 clay）。OOM 返 null（调用方保留旧 data）。

## 交付物

- `include/hyle/container/carray.h`（106 行类型壳）+ `src/container/carray.cc`（type-erased 核）。
- `tests/container/carray_test.cc`（含 `OverAlignedElement` 64B 对齐验证）。

## 验收标准

- **编译**：✅ 双平台。
- **正确性**：✅ container 27→36 case。`OverAlignedElement` 测 64B 对齐过；1000 次 regrow 数据完整。
- **诚实边界**：单线程结构，无并发，不需 TSan。

## 对抗式自审

move 转移 handle 不 realloc（测试验地址不变）；1000 次 regrow 后数据完整（memcpy 保留）；clear 留容量。over-align 修正前 clay 布局会让 64B 元素错位，已用专项测试锁住。

## 参考

- clay `sys/carray.h`/`.cc`（吸纳对象）。
- conventions「type-erased core + typed shell」；`container-design.md`（carray header-before-data 决策）。
