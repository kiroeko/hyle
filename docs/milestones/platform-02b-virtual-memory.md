# platform-02b：virtual_memory（host 虚拟内存）

| 字段 | 值 |
|---|---|
| **阶段** | P1 |
| **模块** | platform/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；VirtualMemory 9/9）|
| **cloc** | **生产 356 行**（`virtual_memory.h` 61 + `virtual_memory.cc` 295）+ **测试 120 行** |

> **范围**：host 端 `mmap`/`VirtualAlloc` 薄封装——**保留/提交分离 + protect/guard page**。  
> **真实消费者**：fiber 栈 overflow 保护（`mem-design` `fiber_stack` / `sched-design` §1）。  
> **不是**：device/VRAM、通用 malloc（mimalloc）、跨进程 shmem（io-05 后续可共用原语）。  
> **clay**：无 virtual_memory API → **net-new**（coverage-map 已登记）。

## 设计决策

1. **自由函数 + RAII `VmRegion`**：`vm_reserve/commit/decommit/release/protect` 是薄 OS 包装；`VmRegion` 拥有整段 reservation（move-only，析构 release），对齐 `DynLib` 风格。
2. **`Protect::kNone` = 永久不可访问**（`PAGE_NOACCESS` / `PROT_NONE`），**不用** Windows 一次性 `PAGE_GUARD`——fiber 栈要持续 trap，不是单次 STATUS_GUARD_PAGE。
3. **`reserve_guarded(usable)`**：低地址 1 页 guard + 高地址可用区（栈向下增长友好）。只 commit 可用区；guard 保持 no-access。
4. **页对齐契约**：size 向上取整到 `page_size()`；非空指针必须页对齐（`HYLE_FATAL_ASSERT`，调用方 bug）。
5. **错误模型**：失败返 `nullptr`/`false` + 线程局部 `vm_last_error()`（与 dynlib 同款，零分配）。
6. **分层**：纯 platform，**零** mem/container 依赖；fiber 栈池后续在 mem/sched 组合本 API。

## 交付物

- `include/hyle/platform/virtual_memory.h`
- `src/platform/virtual_memory.cc`
- `tests/platform/virtual_memory_test.cc`（9）
- CMake：`hyle_platform` 源 + `hyle_platform_test` 源

## 验收

- **Win MSVC**：强制重编 `virtual_memory.cc.obj` → `hyle_platform_test` VirtualMemory **9/9**；全 platform suite **49/49**。
- **Linux WSL ext4**（`/root/hyle`，离线 deps）：`hyle_platform_test --gtest_filter=VirtualMemory*` **9/9**。
- 覆盖：page_size、reserve/commit/write/release、protect None→RW、decommit/recommit、size 上取整、`VmRegion` move、`reserve_guarded` 布局、越界 commit 失败、null release。

## 对抗式自审

- **① 正确性**：Win `MEM_RELEASE` 必须从原始 base 整段释放（`VmRegion` 持有 base+size）；POSIX decommit = `MADV_DONTNEED` + `mprotect(PROT_NONE)`，避免 decommit 后仍可静默写。guard 页不在测试里故意触碰（会 SIGSEGV/AV），布局与 protect 返回值验过。
- **② 性能**：冷路径 OS syscall，非分配热路径；无额外间接。
- **③ 规范**：OS 头只在 `.cc`；公开头无 std 拥有型；`HYLE_PLATFORM_API` 导出；契约 fail-fast 对齐。
- **④ clay 覆盖**：clay 无此项 = net-new，非漏移植。
- **⑤ 不做**：NUMA/大页策略（mem-06）、跨进程 shmem（io-05）、fiber 栈池接线（下一步消费者）、PAGE_GUARD 一次性语义。

## 下游

- **下一步消费者**：sched/mem 的 `fiber_stack`（guard + 分级尺寸 + 池化）应调用 `VmRegion::reserve_guarded` / `vm_*`。
- mem-06 大页可另开 `madvise(MADV_HUGEPAGE)` / `mi_option`，不必阻塞在本里程碑。

## 参考

- backlog-map A 类 platform-02b 行；`mem-design.md` fiber_stack；`sched-design.md` §1 固定栈 + guard page。  
- 现有 ad-hoc 模式：`tests/simd/vec_partial_boundary_test.cc`（本 API 落地后可逐步改用 `VmRegion`，非本步义务）。
