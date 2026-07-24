# mem-final：预发布冻结任务

| 字段 | 值 |
|---|---|
| **状态** | ✅ 完成；mem 进入预发布冻结（2026-07-23） |
| **起始基线** | `024f379`（2026-07-23） |
| **范围** | 只补 API parity、性能证据和最终认证；不新增 allocator 类型 |
| **设计真源** | [`../architecture/mem-design.md`](../architecture/mem-design.md) |
| **性能真源** | [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md) |
| **用户入口** | [`../guide/mem.md`](../guide/mem.md)、[`../guide/c-mem.md`](../guide/c-mem.md) |

> **当前替代（2026-07-24）：**下文 105 项是最终候选 `551be0b` 的历史证书；后续共享 error
> 收敛删除 2 个模块私有错误入口，当前产品面为 **103/103**，已由通用 C surface 门禁在
> Windows/ELF/Mach-O 重新验证。能力面未因此减少。随后 `1de939d` 完成依赖后回审：公共
> alignment 溢出、HugeRegion placement flags、失效 cache allocator 语义已关闭；SharedPool
> empty-proof 冷路与 FixedPool retained-span realloc 已重塑并重签，未新增 allocator 类型或
> C ABI symbol。

这是 Jira 式剩余 work item，不再复制完整能力表、设计理由或 benchmark 点值。mem 的稳定
功能面与“何时算完成”在 architecture；性能结果与输项在 performance；这里仅列未完成任务、
验收和最终提交。

## 已完成证据

- 功能族已封闭：allocator/diagnostics、owned/external/borrowed region、hard/soft/growable pool、
  object lifetime、fiber stack、THP/NUMA 均有明确产品形状或模块归属。
- C ABI 已重审为 C 用户形状：StatsAllocator snapshot/leak copy、三字值对象 FixedRegion、
  FixedPool 显式 control allocator、system collect；header/纯 C consumer/DLL·ELF manifest 为
  API closure 后 105 个动态导出同步。
- Windows 178/178 C++ mem、纯 C 与 DLL 105/105 surface 通过；Linux 178/178 C++ mem、
  纯 C 与 ELF 105/105 surface 通过。
- 最终候选 `551be0b`：公开头逐个独立编译时发现并修复 `hugepage.h` 对 `<utility>` 的
  间接 include 依赖；Windows ASan DLL 的 alias 导出行也纳入 manifest 校验。
- 依赖后回审 `1de939d`：Windows 188/188 + 4 个 mem gate、Linux 188/188、M4 187 PASS +
  1 个无 affinity API 的预期 skip；Linux TSan 14 个共享并发组合 + 纯 C 零报告。新增
  FixedPool realloc 对 oneTBB 在 Win/WSL/M4 双 QoS 全部 sign WIN；SharedPool/cache 重签见
  performance，不把 hard-capacity 与 owner-local 合同混成一张胜负表。

`024f379` 是重审起点；本页勾选只记录后续任务验收，不自动等于最终候选 commit 的认证。

## 剩余 work items

### MEM-FINAL-1：API closure

- [x] 裁决并一次补齐 C 用户需要的只读观察量：`QuotaAllocator::remaining()`、
  `FixedRegion::remaining()`、`FixedPool::peak()`、`GrowablePoolCore::slot_stride()`。
- [x] C 与 C++ 各写/编译三份最短 recipe：request region、typed cached pool、quota+stats；
  验证默认路径、注入路径、ownership 与销毁顺序自然。
- [x] 在 C guide 明确 FiberStack 是 scheduler-owned specialization；没有独立 C consumer 时
  不导出 raw stack control。

### MEM-FINAL-2：performance closure

- [x] 当前 harness 重签 M4 `PooledRegion`/`StackAllocator`，interactive 与 utility 分开报告。
- [x] 给 realloc 增加 direct `SystemAllocator::reallocate` 行，把默认 direct path 与运行时 vtable
  税分开。
- [x] 测 counters-only Stats 与 Quota+Stats 正常组合；leak map 只标诊断慢路。
- [x] 用 sched 的真实 fiber create/reuse workload 验证 `FiberStackPool`，不造孤立纳秒 API。

### MEM-FINAL-3：release-candidate certification

- [x] 最终 API/perf commit 上跑 Windows x64、Linux GCC/Clang、macOS arm64 完整 mem 与纯 C。
- [x] 跑适用的 ASan/UBSan/TSan、DLL/ELF/Mach-O 导出清单和 header self-contained 检查。
- [x] 更新 performance 的 raw/result metadata；检查 architecture/guide/performance/milestone
  没有互相复制的陈旧现状。

最终候选认证矩阵（commit `551be0b`）：

| 平台 / 工具链 | 完整 C++ mem + 纯 C | 动态 C surface | 独立头单元 | sanitizer |
|---|---|---|---|---|
| Windows x64 / MSVC 19.50 | 178/178 + PASS | DLL 105/105 | 16 C++ + 1 C | MSVC ASan PASS |
| Linux x64 / GCC 14.2 | 178/178 + PASS | ELF 105/105 | 16 C++ + 1 C | GCC TSan PASS（`setarch -R`） |
| Linux x64 / Clang 20.1 | 178/178 + PASS | ELF 105/105 | 16 C++ + 1 C | Clang ASan/UBSan PASS |
| macOS arm64 / AppleClang 21 | 178/178 + PASS | Mach-O 105/105 | 16 C++ + 1 C | —；三类 sanitizer 已由 Win/Linux 覆盖 |

performance raw 不重写：最终性能产品提交仍为 clean `2cbc307`；`551be0b` 相对它的产品差异
只有 `hugepage.h` 补直接 `<utility>` include，不改变实现、ABI 或生成路径，其余差异均为
文档/认证工具。因此原始 CSV 和统计结果仍绑定实际产生数据的 commit，而不是伪造新 metadata。

## 停止点

三个 work item 已全部通过，本任务关闭；mem 标记“预发布冻结”并移出主动重构队列。
冻结和重开条件只认 [`mem-design.md` §12](../architecture/mem-design.md#12-收口定义与重开边界)，
不再新增“再观察几轮”或“所有数字都必须赢较弱合同对手”等不可验收条件。
