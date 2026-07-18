# mem 模块审查处置清单 (followup)

> 2026-07-18 对 mem 做了一次全面体检(5 维度:红队实现 / C ABI 完整性+易用 / error 使用 / 功能完善 / 测试覆盖+性能,含三平台)。4 个红队子代理 + 主进程核实。本文件是处置清单 + handoff,给接续的 context。

## 体检总体结论(核实后)

mem 实现**根本可靠**:最高风险的无锁分片 `MemoryArena`(ABA / 内存序 / 无丢块 / 跨线程路由)经红队逐行 + mac arm64 25 遍并发压测双向确认**正确**;C 桥接层无 reinterpret 跨 ABI 布局;system(mimalloc)/ObjectPool/fiber_stack/pmr/stats 都经得起推敲。真问题集中在下面处置清单。

## 红队误报(已核实剔除,勿重查)

- **"GrowablePool 多次 grow 泄漏"= 假**:`chunks_` 是完整单链表,`grow()` 串入(growable_pool.h:93-94),`release_all()` 遍历全部释放(:109-119),析构调它。无泄漏。
- **"HugeRegion prefault SIGBUS"= 假**:失败软降级返 invalid region,不 abort。
- **"system good_size(0) 返非0"= 假**:system.cc:77 有 `size != 0 ? mi_good_size(size) : 0`,返 0 符合文档。

## 处置清单

### #1 alignment==0 野指针 [代码完成 + Windows 验证过;三平台待补]
- `StackAllocator::allocate`(stack_allocator.h:48)、`Arena::allocate`(memory_arena.h:328)的对齐 guard 从 `(a & (a-1)) == 0`(**漏判 0**:`0 & SIZE_MAX == 0` 通过 → 后续野指针)换成 `is_pow2(a)`(alignment.h:22,正确 reject 0 和非2幂)。两文件已加 `#include "hyle/mem/alignment.h"`。
- 无效对齐从"静默野指针"→ clean trap(abort)。`MemoryArena` ctor 本就 clamp alignment 到 16,不受影响。
- **状态**:代码改完,Windows mem 测试全绿。**待**:WSL/mac 三平台复验(和后续改动一起做)。

### #2 池耗尽/OOM graceful + last-error 字符串 [未做;档 A/B 决策待用户拍板]
依据:c-abi-authoring-guide §3 铁律 3 明文"真边界故障(`new` OOM / base==NULL / backing 缺指针)→ **返 NULL**;契约违反 → abort"。当前 slab-OOM / 池耗尽走 abort,**违反了本库自己的铁律**。用户诉求:"保证 C ABI 也能拿到失败的字符串"(类似 error 模块的 `hyle_status_message_cstr`)。
- **档 A(安全,建议先做)**:
  - `Arena` 借块耗尽(memory_arena.h:317 `HYLE_FATAL_ASSERT(block_ != nullptr)`)改可失败:ctor early-return 置 invalid + 加 `valid()`。
  - `c_mem.cc` 的 `hyle_arena_create` 检查 `valid()` → delete + 设 last-error `"pool exhausted"` + 返 NULL。
  - 无效 backing(已 `valid_backing`→NULL)补设 last-error 字符串。
  - `MemoryArena::allocate` 池满**已返 NULL**(:131),无需动。
  - `Arena` dtor 是 `pool_->deallocate(block_)`,`deallocate(nullptr)` 是 no-op → null-safe 几乎免费。
  - last-error 复用 platform `vm_set_last_error`(`hyle_mem_last_error` 已转发 `vm_last_error`)。
- **档 B(高风险,建议隔离 worktree 子代理做,或 fresh session)**:`MemoryArena`/`StackAllocator` 构造时 slab-OOM graceful。难点:`MemoryArena` 三次 alloc(slab memory_arena.h:64 / heads :68 / next :71),placement-new 在分配**之后**(:75 heads、:83 next),而 dtor(:103-115)当前**无条件析构 + 零 null-check**。改可失败要重构成三态(未分配 / 已分配未构造 / 已构造)逆序 null-safe 清理——改错即崩溃/UAF/泄漏。场景罕见(slab 通常几KB–MB,真 OOM=进程内存耗尽)。

### #3 GrowablePool 类型擦除导出 C ABI [未做]
`GrowablePool<T>` 的 `<T>` 只用于 `sizeof`/`alignof` + 裸槽(`acquire()` 返回未构造 storage,`release()` 不调 `~T`),和 `ObjectPool` 同构 → **能类型擦除**。c-mem.md §3.8 把它归"泛型不可达"是**错误归因**(同样论点对 ObjectPool 也成立,而 ObjectPool 明明擦除导出了)。
- 重构 `GrowablePool` 为"非模板运行时核心(slot_size / slot_align / initial_cap)+ `GrowablePool<T>` 薄包装",对齐 `MemoryArena`(非模板)↔ `ObjectPool<T>`(包装)架构。stride 用 `align_up(max(sizeof(void*), slot_size), max(alignof(void*), slot_align))` 运行时算。
- C ABI 六件套:`hyle_growable_pool_o` + `hyle_growable_pool_create(backing, slot_size, slot_align, initial_cap)` / `acquire` / `release` / `reset` / `capacity` + `HYLE_GROWABLE_POOL_CREATE(backing, T, cap)` typed 宏(套 `HYLE_OBJECT_POOL_CREATE`)。

### #4 测试补齐 [未做]
- **mock 失败分配器**(backing alloc 返 nullptr):验 #2 的 last-error/NULL 路径。全仓 tests/mem **目前无任何失败分配器**,OOM 路径零覆盖。
- `alignment==0` 的 death test(`EXPECT_DEATH`):锁死 #1 修复。
- GrowablePool 的 C 测试(配合 #3)。
- (增强,非缺陷)线程扩展曲线 bench(现 bench 写死 8 线程)、耗尽×并发紧张池。

### #5 hardening [未做,低危]
- `MemoryArena` 的 `block_size` 上界校验,防 `stride * count` 溢出(64位不实际/32位可能)。
- `Allocator::valid()`(allocator.h:58)查全三指针(现只查 `alloc_fn`,null free/realloc 会 deref)。

### 文档同步 [未做]
- **c-abi-design.md**:§1 类型擦除表加 GrowablePool 行(和 ObjectPool 并列);§5 补 mem 分配器层错误策略(资源失败 nullptr + last-error / 契约违反 abort,明确区别于 error 模块的 Status——mem 刻意不依赖 error 模块是对的)。
- **c-abi-authoring-guide.md**:§3/§4 补 "资源失败返 NULL 时设 last-error 字符串让 C 用户查因" 的模式。
- **c-mem.md**:GrowablePool 导出、error last-error 通路、arena 不再 abort、纠正 §3.8 "泛型不可达" 误述。

### 三平台验证 + commit [未做]
所有代码改动收尾必须三平台真编真测:
- Windows MSVC:`build/`,vcvars + `cmake --build`。
- WSL GCC-14:github 不通,用 **copy 源码方案**(`FETCHCONTENT_SOURCE_DIR_*` 喂 Windows `.deps` 源码)+ `CUDAToolkit_ROOT=/usr/local/cuda`;详见 .axiomate 的 `reference_wsl_linux_tsan`。
- mac clang:git 同步(`ssh kiro@172.22.10.198`,`git fetch + reset --hard origin/main`),**脚本别用裸 `===`**(zsh EQUALS 展开报错);详见 `reference_mac_ssh`。

## 给下个 context 的提示
- **工具在长 session 偶发污染**(幻觉 Read / Grep 返 "Prj0" 乱码 / TaskUpdate "not found"):改代码前**重读核实**,别信单次可疑输出。
- **#2 的档 A/B 决策待用户拍板**(上一轮问了,未答)。
- 之前已落库:`ad2a05f`(档 B 错误码单一真相源 + UNAUTHENTICATED)、`dec5b36`(thread_spawn_opts + hyle_version + dns/avx512 修复)。
- 三平台环境细节在 .axiomate memory:`reference_wsl_linux_tsan` / `reference_mac_ssh` / `reference_windows_msvc_build`。
