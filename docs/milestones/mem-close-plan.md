# mem-final：预发布冻结任务

| 字段 | 值 |
|---|---|
| **状态** | ✅ 完成；2026-07-26 clean `c640072` 严格复证后重新冻结 |
| **起始基线** | `024f379`（2026-07-23） |
| **范围** | 只补 API parity、性能证据和最终认证；不新增 allocator 类型 |
| **设计真源** | [`../architecture/mem-design.md`](../architecture/mem-design.md) |
| **性能真源** | [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md) |
| **用户入口** | [`../guide/mem.md`](../guide/mem.md)、[`../guide/c-mem.md`](../guide/c-mem.md) |

> **2026-07-26 后续状态：**本页保留上一轮 final 的历史验收；当前状态只见
> [`pre-release-readiness.md`](./pre-release-readiness.md)。后续历史复审输入见
> [`mem-mature-gap-audit.md`](./mem-mature-gap-audit.md)。成熟库红队发现的 alignment、并发
> peak、snapshot/lifetime P0 已修，随后补齐 `SystemStdAllocator<T>` 与
> `system_memory_resource()` 默认静态 adapter；clean `c640072` 已完成 Win/WSL/M4 双 QoS shared
> same-link allocator/adapter/realloc n=15 与受影响正确性/ABI/sanitizer 复签。`FixedPool` 已经提供
> external fixed-capacity 的 variable-size alloc/free/realloc + bulk
> reset；未上架的只是 HYLE-owned growable 版本，不得把它写成一般“可变大小内存”缺口。

> **最终状态补记：**selected-node/chunk owner 已把 mem C surface 扩为 **109/109**；后续复杂度
> 复审允许重复 node 表达 weighted placement，并用 topology membership 位图与 Linux 有界 VMA
> policy 关闭极细条带风险。稳定结论仍以 `mem-mature-gap-audit.md` 为准，下文 105/107 数字只属于
> 对应历史候选，不能当当前 surface。

> **当前替代（2026-07-25）：**下文 105 项是候选 `551be0b` 的历史证书；共享 error
> 收敛曾把产品面降为 103 项，当前 `6213115` 在成熟库/API/复杂度回审后为 **107/107**。
> 新增的是 FixedPool runtime control layout 与 LeakTracker incomplete-inventory 观察量，不是
> 内部 C ABI；HYLE 模块间仍只用 C++ `Allocator/Status/Result`。`1de939d` 完成依赖后回审：公共
> alignment 溢出、HugeRegion placement flags、失效 cache allocator 语义已关闭；SharedPool
> empty-proof 冷路与 FixedPool retained-span realloc 已重塑并重签，未新增 allocator 类型或
> C ABI symbol。`20f3676` 随后按成熟 TLSF/oneTBB/mimalloc 的索引形状回审 FixedPool：公开
> 类型不增长，内部从最坏 O(free blocks) 的同 bin 扫描改为 O(1) 两级位图。`6213115` 又把
> Region 长寿命复用从潜在 O(blocks²) 收口为每 epoch O(blocks)，补齐 aligned PMR、array
> resize 与诊断降级语义，并以较长三平台证书重签。

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
- 最终回审 `6213115`：Windows/Linux/macOS 的 197-test 普通构建、纯 C、107/107 动态 surface、
  16 C++ + 1 C 独立头均通过；MSVC ASan 为 198/198，Linux TSan 33 个相关/并发测试与纯 C
  零报告。shipping benchmark 明确 `HYLE_TESTING=OFF`，不把诊断构建数据签成产品性能。
  scenario 证书脚本已机械拒绝 `testing=1` 的 benchmark，防止该边界再靠人工记忆。

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

### MEM-FINAL-4：FixedPool bounded-lookup reclosure

- [x] 对照 TLSF two-level mapping、oneTBB IndexedBins、mimalloc span queues 与 Boost
  `rbtree_best_fit`，选择保留 `FixedPool` 产品类型、内部采用 8-way FL/SL bitmap。
- [x] 删除同 size bin 的线性 fit scan；allocate/free/coalesce 索引 O(1)，搬移 realloc 仅保留
  不可避免的 O(min(old,new)) copy；4096-byte 长寿命 control 成本写入 C/C++ guide。
- [x] 以 96-hole arena 结构测试证明至多 cursor + 一个索引头，并通过随机 alloc/realloc、
  大 alignment/overflow、Windows/WSL/M4 全门禁及 Linux ASan。
- [x] `20f3676` 的 mixed-size、request percentile、fragmentation 在 Windows clean checkout 与
  WSL/M4 同源码镜像重签；同形成熟参照无 LOSE，M4 interactive/utility 结论一致。

### MEM-FINAL-5：成熟能力、C 边界与长寿命 Region reclosure

- [x] 逐项处置 oneTBB scalable/fixed/cache-aligned/memory-pool、Folly Arena/ThreadCachedArena、
  mimalloc 与 marl：新增 `AlignedPmrResource` 和 overflow-safe array resize；通用 growable heap、
  hidden TLS arena 与 GPU/graph 生命周期明确由已有 backing 或拥有语义的模块负责。
- [x] C++ 内部依赖审计确认普通 mem 实现不包含 `hyle/c/*`、不使用 `hyle_status`；C ABI 只在
  `src/mem/c_mem.cc` outward shim。C 侧补 runtime FixedPool control layout、calloc/realloc-array
  和 LeakTracker dropped/complete，107 个声明/纯 C 调用/动态导出同步。
- [x] Region clear 后使用单调 retained cursor，结构测试以 512 blocks×64 epochs 证明每 block
  每 epoch 至多考察一次；peak 在边界延迟折叠、查询仍精确，正常 request 热路无回归。
- [x] 三平台 shipping 构建重签：scaling 为 31 process×10,001 epochs/size；normal request 为
  Win/Linux 15、M4 双 QoS 各 31 process×10,001 samples，Region 同合同 percentile 无 LOSE。

最终候选认证矩阵（commit `551be0b`）：

| 平台 / 工具链 | 完整 C++ mem + 纯 C | 动态 C surface | 独立头单元 | sanitizer |
|---|---|---|---|---|
| Windows x64 / MSVC 19.50 | 178/178 + PASS | DLL 105/105 | 16 C++ + 1 C | MSVC ASan PASS |
| Linux x64 / GCC 14.2 | 178/178 + PASS | ELF 105/105 | 16 C++ + 1 C | GCC TSan PASS（`setarch -R`） |
| Linux x64 / Clang 20.1 | 178/178 + PASS | ELF 105/105 | 16 C++ + 1 C | Clang ASan/UBSan PASS |
| macOS arm64 / AppleClang 21 | 178/178 + PASS | Mach-O 105/105 | 16 C++ + 1 C | —；三类 sanitizer 已由 Win/Linux 覆盖 |

当前产品认证（commit `6213115`；镜像按实际旧 HEAD/dirty 记录，但同步同一产品源码）：

| 平台 / 工具链 | 完整 C++ mem + 纯 C | 动态 C surface | 独立头单元 | sanitizer |
|---|---|---|---|---|
| Windows x64 / MSVC 19.50 | 197/197 + PASS | DLL 107/107 | 16 C++ + 1 C | MSVC ASan 198/198 + C PASS |
| Linux x64 / GCC 14.2 | 197/197 + PASS | ELF 107/107 | 16 C++ + 1 C | GCC TSan 33 relevant + C PASS |
| macOS arm64 / AppleClang 21 | 197/197 + PASS | Mach-O 107/107 | 16 C++ + 1 C | —；ASan/TSan 由 Win/Linux 当前源码覆盖 |

严格 follow-up 认证（产品 commit `c640072`）：

| 平台 / 工具链 | 受影响正确性 | 动态 C surface | sanitizer / 性能 |
|---|---|---|---|
| Windows x64 / MSVC 19.50 | mem 4/4（含 header 与纯 C） | DLL 107/107 | shared same-link n=15；默认形状无 material LOSE |
| WSL / GCC 14.2 | mem 非 death 191/191 + sched 142/142 + 两项纯 C | ELF 107/107 | GCC TSan 同组零报告；CPU 2 same-link n=15 |
| macOS arm64 / AppleClang 21 | mem/sched 完整 6/6（death test 由本组覆盖） | Mach-O 107/107 | interactive + utility 各 n=15；无硬 P/E affinity 宣称 |

本 follow-up 只改 shared backend 的库内/库外调用边界和证书 harness；`6213115` 的完整
197-test/独立头/ASan 基线仍有效。新证书额外发现并修复 ELF DSO 内二次 exported-seam PLT，
不是为了 benchmark 增加 allocator 类型或常态 metadata。

历史 performance raw 不重写；各节继续绑定实际产生数据的 commit。FixedPool 的新 raw 另以
`20f3676` 后缀保存：Windows 行是 clean 权威 checkout，WSL/M4 行如实保留副本旧 HEAD 与
`dirty=1`，并以相同实现 hash 证明被测产品源码，不伪造镜像 Git metadata。

## 停止点

五个 work item 已全部通过，本任务关闭；mem 标记“预发布冻结”并移出主动重构队列。
冻结和重开条件只认 [`mem-design.md` §12](../architecture/mem-design.md#12-收口定义与重开边界)，
不再新增“再观察几轮”或“所有数字都必须赢较弱合同对手”等不可验收条件。
