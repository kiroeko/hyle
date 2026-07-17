# mem-06c：NUMA 就近放置

| 字段 | 值 |
|---|---|
| **阶段** | P2 / mem-06 收口 |
| **模块** | platform/ + mem/ |
| **状态** | ✅ 完成（Win MSVC + Linux WSL；单 node 平凡正确）|
| **不含** | hwloc 全机拓扑编排（可选运维层，非 mem 义务）|

## 目标

标准库能力面完整（**不**等消费者）：大区就近 / 硬绑 / 交错 + 1GiB 预留封装 + 大 `MemoryArena` slab 自动 HugeRegion。

## 分层

1. **platform** `NumaPolicy`：Preferred / Bind / Interleave；`vm_bind_numa` / `vm_bind_numa_interleave` / `vm_reserve_numa`
2. **HugeRegion**：`allocate` / `allocate_local` / `allocate_interleaved` + `kNumaBind`/`kNumaInterleave`
3. **`enable_allocator_numa`** + **`reserve_huge_1g_pages`**（mi 1GiB，失败 soft）
4. **MemoryArena**：slab ≥ threshold → `HugeRegion::allocate_local` + THP

## 契约

- Preferred 默认；Bind/Interleave 显式  
- 单 node 平凡成功；无特权时 1GiB 预留 false 不崩  
- 无 libnuma/hwloc  

## 验收

- platform/mem Numa* 全绿；大 MemoryArena 可分配释放  
- 双平台 mem/platform 回归不破  

