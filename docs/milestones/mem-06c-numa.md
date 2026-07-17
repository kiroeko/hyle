# mem-06c：NUMA 就近放置

| 字段 | 值 |
|---|---|
| **阶段** | P2 / mem-06 收口 |
| **模块** | platform/ + mem/ |
| **状态** | ✅ 完成（Win MSVC + Linux WSL；单 node 平凡正确）|
| **不含** | hard-bind、hwloc、libnuma、1GB+NUMA 交错预留 |

## 目标

大、长寿命 host 缓冲落在**调用线程 home node**（preferred），跨 node 远程访存 2–3× 劣化可避免。  
与 hugepage 正交可叠加。小对象走 mimalloc 自带 NUMA；不污染热路径。

## 分层

1. **platform** `numa.h`：`numa_node_count` / `numa_current_node` / `vm_bind_numa` / `vm_reserve_numa`  
   - Linux：sysfs 枚举 + `getcpu` + `mbind(MPOL_PREFERRED)`  
   - Windows：`GetNuma*` + `VirtualAlloc2` NumaNode 参数（动态解析）
2. **HugeRegion::allocate(..., node)** / `allocate_local()`  
3. **`enable_allocator_numa(max_nodes)`** → `mi_option_use_numa_nodes`

## 契约

- soft preferred，失败不 FATAL；`kNumaAny` 无约束  
- 单 node：count=1, current=0，API 成功  
- 推荐：worker 钉核后在该线程 `allocate_local` 大区  

## 验收

- platform Numa* 测全绿；mem Numa* 全绿  
- 双平台 mem / platform 回归不破  
