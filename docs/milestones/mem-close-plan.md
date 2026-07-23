# mem-final：预发布冻结任务

| 字段 | 值 |
|---|---|
| **状态** | 🔶 封口候选，尚未冻结 |
| **基线** | `024f379`（2026-07-23） |
| **范围** | 只补 API parity、性能证据和最终认证；不新增 allocator 类型 |
| **设计真源** | [`../architecture/mem-design.md`](../architecture/mem-design.md) |
| **性能真源** | [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md) |
| **用户入口** | [`../guide/mem.md`](../guide/mem.md)、[`../guide/c-mem.md`](../guide/c-mem.md) |

这是 Jira 式剩余 work item，不再复制完整能力表、设计理由或 benchmark 点值。mem 的稳定
功能面与“何时算完成”在 architecture；性能结果与输项在 performance；这里仅列未完成任务、
验收和最终提交。

## 已完成到基线

- 功能族已封闭：allocator/diagnostics、owned/external/borrowed region、hard/soft/growable pool、
  object lifetime、fiber stack、THP/NUMA 均有明确产品形状或模块归属。
- C ABI 已重审为 C 用户形状：StatsAllocator snapshot/leak copy、三字值对象 FixedRegion、
  FixedPool 显式 control allocator、system collect；header/纯 C consumer/DLL·ELF manifest 为
  101 个动态导出同步。
- Windows 完整 mem 174/174、纯 C/C surface 通过；WSL 相关 C++、纯 C 与 ELF 101/101 通过。

上述是 `024f379` 的任务证据，不自动等于最终候选 commit 的认证。

## 剩余 work items

### MEM-FINAL-1：API closure

- [ ] 裁决并一次补齐 C 用户需要的只读观察量：`QuotaAllocator::remaining()`、
  `FixedRegion::remaining()`、`FixedPool::peak()`、`GrowablePoolCore::slot_stride()`。
- [ ] C 与 C++ 各写/编译三份最短 recipe：request region、typed cached pool、quota+stats；
  验证默认路径、注入路径、ownership 与销毁顺序自然。
- [ ] 在 C guide 明确 FiberStack 是 scheduler-owned specialization；没有独立 C consumer 时
  不导出 raw stack control。

### MEM-FINAL-2：performance closure

- [ ] 当前 harness 重签 M4 `PooledRegion`/`StackAllocator`，interactive 与 utility 分开报告。
- [ ] 给 realloc 增加 direct `SystemAllocator::reallocate` 行，把默认 direct path 与运行时 vtable
  税分开。
- [ ] 测 counters-only Stats 与 Quota+Stats 正常组合；leak map 只标诊断慢路。
- [ ] 用 sched 的真实 fiber create/reuse workload 验证 `FiberStackPool`，不造孤立纳秒 API。

### MEM-FINAL-3：release-candidate certification

- [ ] 最终 API/perf commit 上跑 Windows x64、Linux GCC/Clang、macOS arm64 完整 mem 与纯 C。
- [ ] 跑适用的 ASan/UBSan/TSan、DLL/ELF/Mach-O 导出清单和 header self-contained 检查。
- [ ] 更新 performance 的 raw/result metadata；检查 architecture/guide/performance/milestone
  没有互相复制的陈旧现状。

## 停止点

三个 work item 全部通过后，本任务关闭，mem 标记“预发布冻结”并移出主动重构队列。
冻结和重开条件只认 [`mem-design.md` §12](../architecture/mem-design.md#12-收口定义与重开边界)，
不再新增“再观察几轮”或“所有数字都必须赢较弱合同对手”等不可验收条件。
