# mem-08：fiber_stack（guard + 分级 + 池化）

| 字段 | 值 |
|---|---|
| **阶段** | P2 / mem P0 收口 |
| **模块** | mem/ + sched/ 接线 |
| **状态** | ✅ 完成（Win MSVC + Linux WSL：`hyle_mem_test` 56/56，`hyle_sched_test` 108/108）|
| **cloc** | 相关 TU ~518 code（`fiber_stack.h` + fiber 接线 + 两份测试；fiber.h 体量大头为既有 API 注释）|
| **依赖** | platform-02b `VmRegion::reserve_guarded` |

> **范围**：fiber 栈 = 定长 usable + 低 1 页永久 guard（`Protect::kNone`）+ 分级尺寸 + 可选 `FiberStackPool` 回收。  
> **真实消费者**：`sched::Fiber`（默认路径全部走 guarded 栈）。  
> **不是**：NUMA/大页/stats（mem-06）、device/pinned。

## 设计决策

1. **`mem::FiberStack`**：拥有 `platform::VmRegion`（`reserve_guarded`）；`base/size/top/guard`。
2. **分级**：`kFiberStackSmall=16KiB` / `Default=64KiB` / `Large=256KiB`（usable；guard 另加一页）。`sched` 侧 `kSmallStackSize` / `kDefaultStackSize` / `kLargeStackSize` 别名。
3. **`FiberStackPool`**：同 tier 定长 freelist，SpinLock，cache cap 默认 64；错尺寸 release 直接销毁。
4. **`Fiber` 接线**：栈永远是 `FiberStack`；旧 `Fiber(Allocator, size, entry, arg)` **保留**但 **忽略 Allocator**（源码兼容海量调用点）；新增 `Fiber(size, …)` 与 `Fiber(FiberStack, …, pool*)`（可选归还池）。
5. **不测 guard 触碰 fault**（会 SIGSEGV/AV）；布局 + 可写 usable + fiber 跑通 + 池回收指针相等。

## 交付物

- `include/hyle/mem/fiber_stack.h`
- `include/hyle/sched/fiber.h` / `src/sched/fiber.cc`（改）
- `tests/mem/fiber_stack_test.cc`
- `tests/sched/fiber_stack_guard_test.cc`
- `docs/architecture/mem-design.md`（fiber_stack 状态）

## 验收

- `hyle_mem_test`：FiberStack* 全过  
- `hyle_sched_test`：Fiber* + FiberGuard* 全过；既有 fiber/sched 回归不破  
