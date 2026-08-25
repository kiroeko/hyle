# mem-06b：大页 / THP hint（大 arena · 张量 TLB）

> **历史任务记录。**当前 HugeRegion/NUMA/THP 能力与 platform 边界见
> [`mem-design`](../architecture/mem-design.md) 和 [`guide/mem`](../guide/mem.md)；下文保留该切片
> 当时的平台范围与验收数字。

| 字段 | 值 |
|---|---|
| **阶段** | P2 / mem-06 切片 |
| **模块** | platform/ + mem/ |
| **状态** | ✅ 完成（Win MSVC + Linux WSL）|
| **不含** | NUMA 绑 node、1GB 硬预留、hugetlbfs 特权路径（后置）|

## 目标

大工作集访问 TLB 项数 \(\sim N/\text{page}\) → 对齐 2MiB + THP hint 后 \(\sim N/2\text{MiB}\)。  
分配仍 O(1)；**不**污染小对象 / fiber 栈热路径。

## 分层

1. **platform**：`large_page_size` / `large_page_align_up` / `vm_advise_hugepage` / `vm_advise_no_hugepage`  
   - Linux：`madvise(MADV_HUGEPAGE|NOHUGEPAGE)`  
   - Win/mac：诚实 `false`（无匿名 THP madvise）
2. **mem::HugeRegion**：large-page 对齐 reserve+commit + 可选 THP + prefault  
3. 早期曾封装 `enable_allocator_hugepages`；mimalloc 3.5 loader 时序红队证明 Windows 会在用户代码
   前缓存 large-page capability，而 Linux 只对未来 mmap 动态读 option，故该跨平台伪 setter 已移除。
   system allocator 用启动前环境配置，逐 span 策略用 HugeRegion。

## 交付

- `include/hyle/platform/virtual_memory.h` + `src/platform/virtual_memory.cc`
- `include/hyle/mem/hugepage.h`（header HugeRegion）+ `system.cc` 策略
- `tests/platform/virtual_memory_test.cc`（LargePage* / AdviseHugepage*）
- `tests/mem/hugepage_test.cc`

## 验收

- large_page_size 为 0 或 ≥ page_size 的 2 幂  
- HugeRegion ≥ threshold 可写；move/reset  
- 旧 setter 的“双调用不崩”测试已作废：它不能证明策略生效
- 双平台 mem + platform 相关测全绿  
