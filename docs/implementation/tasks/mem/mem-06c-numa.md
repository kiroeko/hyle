# mem-06c：NUMA 就近放置

> **历史任务记录。**本文中的 `MemoryArena`、模块 API 表和“Stats 仅 C++”均已被后续设计
> 替代。当前 mem 设计见 [`../architecture/mem-design.md`](../../../architecture/mem-design.md)，
> 当前 C API 见 [`../guide/c-mem.md`](../../../architecture/native-ffi-policy.md)，VM/NUMA 错误接口追记见
> [`platform-error-convergence.md`](../platform/platform-error-convergence.md)。后文保留当时验收语境。

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
3. **`reserve_huge_1g_pages`**（mi 1GiB，失败 soft）；早期的
   `enable_allocator_numa` 后经 mimalloc 3.5 loader 时序红队证明无法从用户代码可靠生效，已移除。
   system allocator 的进程启动 node cap 使用启动前环境，运行期 placement 使用 HugeRegion/platform。
4. **MemoryArena**：slab ≥ threshold → `HugeRegion::allocate_local` + THP

## 契约

- Preferred 默认（软；永不失败进程）；Bind/Interleave 显式  
- **诚实失败**：单 node 上任何 per-node 策略是真 no-op → true；**多 node 但平台无 API**（Windows post-hoc 硬 Bind / 非 Linux Interleave）→ **false + `vm_set_last_error`**，不再假装成功。`vm_reserve_numa` 的 preferred 绑定失败**保留映射**（区可用），仅写 `vm_last_error` 供观测  
- 单 node 平凡成功；无特权时 1GiB 预留 false 不崩  
- 无 libnuma/hwloc  

## C ABI（close-01，v3）

NUMA/大页原语当时随 platform C ABI v3 导出（`include/hyle/c/platform.h`，`hyle_platform_api` 表尾追加，保 v2 前缀布局）：`hyle_numa_node_count` / `hyle_numa_current_node` / `hyle_vm_bind_numa` / `hyle_vm_bind_numa_interleave` / `hyle_vm_reserve_numa` + hugepage `hyle_large_page_size` / `hyle_vm_advise_hugepage(_no)` + `hyle_vm_set_last_error`。stats（`StatsAllocator`/`LeakTracker`）是 mem 有状态类，仅 C++。该 C surface 与旧指南现在都是待删历史债。

## 验收

- platform/mem Numa* 全绿；大 MemoryArena 可分配释放  
- 双平台 mem/platform 回归不破；C ABI `hyle_platform_c_test` v3 smoke 过  
- Bind 诚实失败：多 node Windows 硬 Bind/Interleave 返 false（`numa_test.cc` 按平台断言）  
