# mem-08：fiber_stack（guard + 分级 + 池化）

> **历史任务记录。**当前 FiberStackPool 的 capacity、trim/clear、scheduler-owned 生命周期与
> 性能证书见 [`mem-design`](../architecture/mem-design.md)、[`guide/mem`](../guide/mem.md) 和
> [`benchmarks-mem`](../performance/benchmarks-mem.md)；下文保留初次接线记录。

| 字段 | 值 |
|---|---|
| **阶段** | P2 / mem P0 收口 |
| **模块** | mem/ + sched/ 接线 |
| **状态** | ✅ 完成（Win MSVC + Linux WSL + WSL TSan；池化接线 2026-07-17，删兼容构造）|
| **cloc** | 相关 TU ~518 code（`fiber_stack.h` + fiber 接线 + 两份测试；fiber.h 体量大头为既有 API 注释）|
| **依赖** | platform-02b `VmRegion::reserve_guarded` |

> **范围**：fiber 栈 = 定长 usable + 低 1 页永久 guard（`Protect::kNone`）+ 分级尺寸 + 可选 `FiberStackPool` 回收。  
> **真实消费者**：`sched::Fiber`（默认路径全部走 guarded 栈）。  
> **不是**：NUMA/大页/stats（mem-06）、device/pinned。

## 设计决策

1. **`mem::FiberStack`**：拥有 `platform::VmRegion`（`reserve_guarded`）；`base/size/top/guard`。
2. **分级**：`kFiberStackSmall=16KiB` / `Default=64KiB` / `Large=256KiB`（usable；guard 另加一页）。`sched` 侧 `kSmallStackSize` / `kDefaultStackSize` / `kLargeStackSize` 别名。
3. **`FiberStackPool`**：同 tier 定长 freelist，SpinLock，cache cap 默认 64；错尺寸 release 直接销毁。
4. **`Fiber` 接线**：栈永远是 `FiberStack`；两个构造 `Fiber(size, entry, arg)` 与 `Fiber(FiberStack, entry, arg, pool*)`（可选归还池）。**删除**了曾为源码兼容保留的 `Fiber(Allocator, size, …)`（HYLE 未发布、无外部调用方，静默吞 allocator 是陷阱）——全部调用点（测试 + `spawn_task`）改直传 size 或 `FiberStack`。
5. **池化接线**：`spawn_task` 从 `Scheduler` 的 per-tier 共享 `FiberStackPool`（Small/Default/Large）`acquire()` 栈，`Fiber` 析构时 `release` 回同池（在 worker dispatch 的 finish 钩子里，栈已离栈执行——安全归还点）。acquire 可能在 foreign 线程、release 在 worker，故用**共享**池（SpinLock 往返）而非 per-worker。非 tier 尺寸回退一次性 `FiberStack::allocate`（pool==nullptr）。池归 `Scheduler::Impl`，生命周期 ≥ 所有 driver（shutdown 先 drain 全部 fiber）。
6. **不测 guard 触碰 fault**（会 SIGSEGV/AV）；布局 + 可写 usable + fiber 跑通 + 池回收指针相等。

## 交付物

- `include/hyle/mem/fiber_stack.h`
- `include/hyle/sched/fiber.h` / `src/sched/fiber.cc`（删兼容构造）
- `include/hyle/sched/scheduler.h` / `src/sched/scheduler.cc`（per-tier `FiberStackPool` + `fiber_stack_pool()`）
- `include/hyle/sched/join.h`（`spawn_task` acquire/归还池）
- `tests/mem/fiber_stack_test.cc`、`tests/sched/fiber_stack_guard_test.cc`
- 全仓库 Fiber 调用点（io/graph/compute/sched 测试）改直传 size
- `docs/architecture/mem-design.md`（fiber_stack 状态）

## 验收

- **Linux WSL**：`hyle_mem_test` 36/36（stats+numa+hugepage+fiber 切片）、`hyle_sched_test` 107/107；**TSan 107/107 零竞争**（池跨线程 acquire/release 经 SpinLock）
- **Win MSVC**：`hyle_platform_test` 28/28、`hyle_mem_test` 36/36、`hyle_sched_test`（fiber+join+pool+scheduler+sync）58/58、`hyle_simd_sse4_test` PartialBoundary 46/46
- 删兼容构造后既有 fiber/sched/io/graph/compute 回归全绿  
