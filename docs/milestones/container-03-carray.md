# container-03：carray header-before-data + RAII 壳

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，container 27→36 case）|

> **cloc 留档（`cloc carray.h carray.cc carray_test.cc`）**：**224 行** = 106 头 + 118(carray.cc 类型擦除核 + 测试)。type-erased 核只编一份,非 stub。
> **吸纳 clay `sys/carray.h`(436)+`.cc`(65)**：header-before-data 布局(handle 指向元素,`CArrayHeader{capacity,size,header_offset,align,alloc}` 紧贴其前)——`arr[i]` 零间接、可直传 C API。type-erased 核 `carray_set_capacity_internal(void*, item_size, align, alloc)` 只编一份(conventions「type-erased core + typed shell」,防 per-T 膨胀)。
> **改进 clay**：clay header 固定 `max_align_t` 对齐,**over-aligned 元素(如 64B cacheline / AVX512)会错位**;HYLE 用 `header_offset`(向上取整到 max(align,alignof(header)))→ 元素指针精确按 align 对齐,`OverAlignedElement` 测试验 64B 对齐过。
> **typed 壳 CArray<T>**:push_back/pop_back/reserve/clear/operator[]/迭代器/move/初始化列表;T 限 trivially-copyable(memcpy 增长,同 clay)。OOM 返 null(调用方保留旧 data)。
> **自审**:move 转移 handle 不 realloc(测试验地址不变);1000 次 regrow 后数据完整(memcpy 保留);clear 留容量。无并发(carray 单线程,不需 TSan)。