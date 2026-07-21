# 调用安全边界（Usage Contract）

> **HYLE 的核心目标是性能做到 SOTA**（见 [`performance/`](../performance/README.md)）。为此某些热路径**只保证特定调用方式安全**，或在文档里讲明注意事项——本文件就是这份"怎么调才对"的契约。**讲明边界优先于牺牲性能去做万能防御**：HYLE 是贴硬件的 infra，调用者被假定读过本契约。
>
> 每条边界都附 **为什么**（机制）+ **怎么调**（安全模式）+ **违反的后果**。带 ⚡ 的是为性能刻意保留的约束。

---

## 1. fiber：不得跨 `yield()`/`suspend()` 持有 thread_local 派生值 ⚡

**契约**：fiber 体（跑在 `Fiber` 栈上的用户代码）中，**任何 `thread_local` 变量的地址、或其读出值缓存到局部，都不得跨 `Fiber::yield()` / `sched::suspend()` 复用**。yield 之后要用，必须**重新读**。

**为什么**：HYLE 调度器是 work-stealing 的——fiber 在 worker A 上 `yield()` 后，可能被 worker B 偷走、在 **B 线程**上恢复（迁移）。底层 `jump_fcontext` 切换栈与通用寄存器，但**不切换 `%fs`**（TLS 基址寄存器），也不是编译器可见的"线程可能改变"信号。`-O3` 下编译器把 `thread_local` 的 `%fs` 相对地址当作"跨调用稳定"，缓存进 callee-saved 寄存器跨 yield 复用——迁移后该地址仍指向**原线程 A** 的 TLS。实测：强制持有时 **899 万次 / 900 万次迁移全部读到错误线程的 TLS**（见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md) E15）。

**怎么调（安全）**：
```cpp
// 危险:cached/addr 跨 yield 复用 -> 迁移后指向旧线程 TLS
thread_local Ctx g_ctx;
Ctx* p = &g_ctx;          // 在 worker A 解析地址
Fiber::yield();            // 迁移到 worker B
p->use();                  // ✗ 读到了 A 的 g_ctx

// 安全:yield 之后重新访问
Fiber::yield();
g_ctx.use();               // ✓ 在当前线程(B)重新解析
```
- 需要"当前 worker / 当前线程"的身份，**在 yield 之后**再取。
- 把 per-thread 状态换成 **per-fiber 状态**（放进 fiber 自己的栈/上下文，随迁移一起走），或经**显式注入**的对象传递。

**后果**：静默读/写错线程的 TLS → 数据损坏 / 崩溃，且只在真发生迁移（重负载/work-stealing 命中）时出现，极难复现。

> 这不是 HYLE 独有的——**所有有栈协程库**（Boost.Context、ucontext、早期 Go）都有这条铁律。HYLE 选择"讲明 + 让热路径自然安全"，而非在每个 yield 插屏障（屏障也救不了：`%fs` 缓存被 CSE 到函数入口，memory clobber 与"重取地址"都无效，见 E16）。

## 2. 分配器：跨线程 free 安全；mimalloc 常开、无 off-switch

**契约**：经 `mem::system_allocator()`（mimalloc 后端）分配的内存，**可以在任意线程释放**——fiber 在 worker A 分配、迁移后在 worker B 释放是**安全且支持**的核心用例。

**为什么安全**：分配器封在 `Allocator` vtable（函数指针）后、且实现跨 `.so` 边界（`hyle_mem`）。fiber 体调 `deallocate()` 是**不透明调用**——`-O3` 无法把 `mi_free` 内部的 TLS heap 读提升/缓存到 fiber 体的 yield 之上，每次在**当前线程**重新解析。即第 1 条的危险模式在分配器路径上**天然不成立**。实测：强制迁移 **687 万次跨线程 free，rc=0 净**（mimalloc v2.3.2，含段完整性校验，见 E13/E17）。

**⚡ 性能注意事项**：
- **对齐**：默认对齐（≤16B，`alignof(max_align_t)`）走 `mi_malloc` 快路径；**仅在真需要过对齐**（如 AVX512 的 64B）才传更大 alignment——过对齐路径有额外簿记，紧凑 alloc/free 循环里会变慢。
- **不要 static-link + LTO 把 `mi_*` 内联进 fiber 体** ⚡：那会打破上面的"不透明边界"，使 `mi_free` 的 TLS heap 读被缓存跨 yield，**复活第 1 条的危险**。HYLE 默认 **shared lib + 无 LTO**，安全;若为极致性能改 static+LTO 发布，须 pin fiber 到创建 worker（禁迁移）或保持分配器 TU 不可内联。

**无 off-switch**：`HYLE_USE_MIMALLOC` 开关与 OS 分配器后备**已移除**（提交 92f8ec8）。理由:后备是陷阱——会让构建静默绕过 mimalloc 依赖问题、被误判为"无事"。mimalloc 是出厂分配器,出问题就正面查。

### 2.1 共享固定池：硬容量与软容量必须按合同选

**硬容量默认**：`SharedPool` / `ObjectPool<T>` 不把块藏在 TLS。一次 `deallocate()` 完成后，该块立即对所有线程可见；`allocate()` 通过稳定双快照排除分片扫描期间的假耗尽。容量紧贴工作集、跨线程 handoff 或固定 quota 必须用这组。

**软容量 opt-in**：`SharedPoolCached` / `ObjectPoolCached<T>` 使用每线程 **128-slot TLS cache**，并以 **32-block magazine** 与 shared core 批量交换。同线程 free→alloc 是零原子快路，但 A 线程 TLS 中已 free 的块，在 A 调 `publish_thread_cache()`、cache overflow 或线程退出前，B 看不到。因此软容量窗口最多为 `128 ×` 活跃线程数，B 可以在 shared core 真空时拿到 `nullptr`。

**怎么调**：
- 要 completed free 立即共享：直接用 `SharedPool` / `ObjectPool<T>`，无需 publish；
- 明确选择 cached 版本后要跨线程交接容量：由 free 方调用 `publish_thread_cache()`，或 join free 线程；
- `reset()` 要求无 live block，并先静默所有并发 alloc/free。cached 版本会清注册 cache 再重建 shard。

两组池共用 sharded core；正常走 32-bit tagged CAS，tag 饱和的 shard 永久切入 serialized fallback，绝不回绕。不要把 cached 池的吞吐数据或软容量语义写成硬池合同。

## 3. SIMD：CPUID 探测器 TU 必须 LTO-free（分发型构建）

**契约**：分发给未知 CPU 的二进制（非 `-march=native` 自用），**不要对 `simd/` 模块开全局 LTO/IPO**。

**为什么**：`dispatch.cc` 是 CPUID 探测 TU——它在一台**尚未探明的 CPU** 上**最先**运行，去发现这台 CPU 能跑什么。若 LTO 把某个高 ISA 变体 kernel（AVX2/AVX512）的指令内联进探测器或 baseline dispatcher，探测器会在老 CPU 上**还没来得及报告"这是老 CPU"就先 SIGILL**（llama.cpp 踩过的硬伤，conventions §10）。

**安全模式**：HYLE 的 `src/simd/CMakeLists.txt` 已**强制整个 simd 模块 IPO/LTO off**（防御性，即便未来全局开 LTO 也不波及），探测器 + baseline dispatcher + 各变体 kernel TU 互不内联。**你只要别在消费侧用全局 LTO 重新内联跨 TU 边界**即可；自用 `-march=native` 构建无此约束（无变体、无运行期探测）。**后果**：违反 = 老 CPU 上启动即 SIGILL。

## 4. 通用纪律

- **跨平台代价讲在明处**：HYLE 的所有权/生命周期约定见 [`lifecycle-design.md`](../architecture/lifecycle-design.md)；`string_ref`/`CArray` 等非拥有型视图的生命周期由调用者保证。
- **线程/迁移模型**：除非文档另述,假定 fiber 可在任意 worker 间迁移;不要假设"同一 fiber 总在同一 OS 线程"。
- **本契约随实现演进**：新增热路径若引入"为性能而限制调用方式"的约束,必须在此登记(规则 + 为什么 + 安全模式 + 后果)。

---

> 一句话:**HYLE 把热路径顶到 SOTA,代价是调用者遵守少数几条明确边界**。这些边界都来自真实验证(见 ki-1-crash-hunt.md 的 E1–E17),不是臆测的防御。
