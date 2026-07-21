# mem 模块审查处置清单 (followup)

> **历史名称边界：**本文保留审计当时的 `MemoryArena` 名称与数字；当前固定块池 API 是硬容量 `SharedPool` 和软容量 `SharedPoolCached`，旧 `memory_arena.h` / `hyle_memory_arena_*` 已删除并由现 API 取代。

> 2026-07-18 对 mem 做了一次全面体检(5 维度:红队实现 / C ABI 完整性+易用 / error 使用 / 功能完善 / 测试覆盖+性能,含三平台)。4 个红队子代理 + 主进程核实。本文件是处置清单 + handoff,给接续的 context。

## 体检总体结论(核实后)

mem 实现**根本可靠**:最高风险的无锁分片 `MemoryArena`(ABA / 内存序 / 无丢块 / 跨线程路由)经红队逐行 + mac arm64 25 遍并发压测双向确认**正确**;C 桥接层无 reinterpret 跨 ABI 布局;system(mimalloc)/ObjectPool/fiber_stack/pmr/stats 都经得起推敲。真问题集中在下面处置清单。

## 红队误报(已核实剔除,勿重查)

- **"GrowablePool 多次 grow 泄漏"= 假**:`chunks_` 是完整单链表,`grow()` 串入(growable_pool.h:93-94),`release_all()` 遍历全部释放(:109-119),析构调它。无泄漏。
- **"HugeRegion prefault SIGBUS"= 假**:失败软降级返 invalid region,不 abort。
- **"system good_size(0) 返非0"= 假**:system.cc:77 有 `size != 0 ? mi_good_size(size) : 0`,返 0 符合文档。

## 处置清单

### #1 alignment==0 野指针 [已做 + 三平台验证 + death test 锁死]
- `StackAllocator::allocate`(stack_allocator.h:48)、`Arena::allocate`(memory_arena.h:328)的对齐 guard 从 `(a & (a-1)) == 0`(**漏判 0**:`0 & SIZE_MAX == 0` 通过 → 后续野指针)换成 `is_pow2(a)`(alignment.h:22,正确 reject 0 和非2幂)。两文件已加 `#include "hyle/mem/alignment.h"`。
- 无效对齐从"静默野指针"→ clean trap(abort)。`MemoryArena` ctor 本就 clamp alignment 到 16,不受影响。
- **状态**:代码改完 + `alignment==0` death test 锁死(见 #4);**三平台全绿**(Windows MSVC / mac clang / WSL GCC-14)。

### #2 池耗尽/OOM graceful + last-error 字符串 [档 A + 档 B 均已做]
依据:c-abi-authoring-guide §3 铁律 3 明文"真边界故障(`new` OOM / base==NULL / backing 缺指针)→ **返 NULL**;契约违反 → abort"。当前 slab-OOM / 池耗尽走 abort,**违反了本库自己的铁律**。用户诉求:"保证 C ABI 也能拿到失败的字符串"(类似 error 模块的 `hyle_status_message_cstr`)。
- **档 A(安全,建议先做)**:
  - `Arena` 借块耗尽(memory_arena.h:317 `HYLE_FATAL_ASSERT(block_ != nullptr)`)改可失败:ctor early-return 置 invalid + 加 `valid()`。
  - `c_mem.cc` 的 `hyle_arena_create` 检查 `valid()` → delete + 设 last-error `"pool exhausted"` + 返 NULL。
  - 无效 backing(已 `valid_backing`→NULL)补设 last-error 字符串。
  - `MemoryArena::allocate` 池满**已返 NULL**(:131),无需动。
  - `Arena` dtor 是 `pool_->deallocate(block_)`,`deallocate(nullptr)` 是 no-op → null-safe 几乎免费。
  - last-error 复用 platform `vm_set_last_error`(`hyle_mem_last_error` 已转发 `vm_last_error`)。
- **档 B [已做]**:`MemoryArena`/`StackAllocator` 构造 slab-OOM + `GrowablePool` grow-OOM 全部 graceful(不再 abort):
  - `MemoryArena` ctor 三次 alloc(slab/next/heads)任一失败 → 提前 return 置 invalid;加 `valid_`/`valid()`;dtor 三态 null-safe(按 `valid_` 判元素析构、按指针 null 判 deallocate);allocate/deallocate/reset 加 `valid_` 检查(invalid arena 安全返 null/no-op)。
  - `StackAllocator` ctor slab-OOM → invalid + `valid()`;dtor null-safe(allocate 天然返 null,end_==null)。
  - `GrowablePoolCore::grow()` 改返 bool,OOM → `acquire()` 返 null → `GrowablePool<T>::create()` 返 null(同 `ObjectPool`)。
  - `c_mem.cc`:`hyle_memory_arena_create`/`hyle_stack_allocator_create` 检查 `valid()`,invalid → delete + last-error "slab allocation failed" + NULL;mem.h 注释同步(slab OOM 不再 abort)。
  - 测试:c_mem_test.c mock OOM backing 段(三分配器 create/acquire graceful NULL + 非空 last-error),已验 Windows 绿。保留契约违反(block_size==0 等)fail-fast abort。
  - (可选增强,未做)C++ 层单元 mock 测试直接验 `MemoryArena::valid()==false`/`GrowablePool::create()==null`——C 测试已端到端覆盖 graceful,此为补充。

- **[last-error 覆盖收尾,已做]** 三个 create 的 wrapper OOM(`new(nothrow)` 失败)原先返 NULL 但**没设 last-error**(C 用户会读到上一次的过时串)。已补 `hyle_memory_arena_create`/`hyle_stack_allocator_create`/`hyle_arena_create` 的 wrapper OOM + `arena_create` 的 `pool==NULL`,都设 last-error;mem.h 对应注释同步。现三个 create 的每条 NULL 返回都有原因串。(档 B slab-OOM graceful 落地时再补 "slab allocation failed" 串。)
- **[可选增强,未做]** `MemoryArena::free_count()`/`has_free()` + `hyle_memory_arena_free_count()` C 导出。上个会话 C 子代理指出 C ABI 无空闲块数查询、无 try_create,FFI 借前无法预检容量。`valid()`+NULL 已够防御,此为额外可观测性 API,非必须。

### #3 GrowablePool 类型擦除导出 C ABI [已做]
**认知修正**:旧描述"`GrowablePool<T>` 裸槽、acquire 返回未构造 storage、release 不调 ~T"**不准确**——实际 `GrowablePool<T>` 是 `create()`(跑 `::new T`)/`destroy()`(跑 `~T`),和 `ObjectPool<T>` 同语义。裸槽是**擦除后 C 核心**的行为,不是 C++ 模板的。擦除依然成立(同 `ObjectPool<T>`↔`MemoryArena` 架构)。
- **[已做]** `detail::GrowablePoolCore`(非模板:slot_size/slot_align/initial_slots,裸槽 acquire/release/clear,几何增长 + intrusive free-list,stride=`align_up(max(sizeof(void*),slot_size), max(alignof(void*),slot_align))`)+ `GrowablePool<T>` 薄包装(create/destroy 跑 ctor/dtor)。C++ 7 测试行为不变。
- **[已做]** C ABI:`hyle_growable_pool_create(backing,slot_size,slot_align,initial_slots)` / `destroy` / `acquire`(裸槽) / `release` / `reset`(trim) / `capacity` / `live` + `HYLE_GROWABLE_POOL_CREATE/ACQUIRE/RELEASE` 宏。裸槽模型,C 用户自 init/cleanup(同 object pool)。
- **[已做]** C 测试(c_mem_test.c):grow/指针稳定/free-list 复用/reset trim/invalid backing+last-error/typed 宏。
- 注:grow OOM 仍 abort(留 #2 档 B 统一 graceful);**c-mem.md §3.8 "泛型不可达"误述 → ⑤文档同步修**。

### #4 测试补齐 [alignment death + GrowablePool C 测试 + arena 耗尽已补;mock 失败分配器归档 B]
- **[已补]** `hyle_arena_create` 池耗尽 → NULL + 非空 last-error 的 C 测试(c_mem_test.c,配合 #2 档 A)。
- **[已补]** `alignment==0` death test:`StackAllocatorDeathTest.ZeroAlignmentTraps` + `ArenaDeathTest.ZeroAlignmentTraps`(EXPECT_DEATH),锁死 #1。**已验 Release 下 HYLE_FATAL_ASSERT 确实 trap**(96 测试含 2 death 全绿)。
- **[已补]** GrowablePool 的 C 测试(配合 #3):grow/指针稳定/free-list 复用/reset trim/invalid backing last-error/typed 宏。
- **[归档 B]** **mock 失败分配器**(backing alloc 返 nullptr)验 slab-OOM NULL+last-error——依赖档 B 的 graceful(现 slab-OOM 仍 abort),和档 B 一起做。tests/mem 目前无失败分配器。
- (增强,非缺陷)线程扩展曲线 bench(现写死 8 线程)、耗尽×并发紧张池。

### #5 hardening [全部已做]
- **[#5b 已做]** `MemoryArena` ctor(memory_arena.h:48-54)加溢出 guard:round-up `block_size_ + (alignment_-1)` 不 wrap + slab `stride_ * count_` 不超 size_t;超大 block_size/count 从"静默 under-alloc"→ fail-fast(契约违反)。
- **[#5a 已做 d1f7930]** `Allocator::valid()`(allocator.h:58-63)查全三指针(原只查 `alloc_fn`,null free/realloc 会 deref)。

### 文档同步 [已做]
- **[已做] c-abi-design.md**:§1 类型擦除表加 GrowablePool 行(和 ObjectPool 并列);§5 补 mem 分配器层错误策略(资源失败 nullptr + last-error / 契约违反 abort;mem 刻意不依赖 error 模块 Status——最底层基元,依赖会倒置)。
- **[已做] c-abi-authoring-guide.md**:§3 铁律 3 补"池耗尽"+ "句柄返回型 API 返 NULL 时配 thread-local last-error 串"模式(呼应 hyle_status_message_cstr)。
- **[已做] c-mem.md**:新增 §3.5 GrowablePool 章节(七件套 + 宏 + 裸槽模型,同 object pool);§3.9 纠正"泛型不可达"误述(→泛型语法过不了 C ABI 但能力经宏可达,同 ObjectPool<T>);3 处 create 资源失败改 graceful;last-error 覆盖扩面;章节顺延 §3.5-3.8→3.6-3.9(引用同步)。
- **[附带,已做]** mem.h growable grow-OOM 注释修正(档 B 遗漏:原说 abort,实际 acquire 返 NULL graceful);README.md 索引修正(补 GrowablePool 句柄 + 去 stale "hyle_mem_api 表"——mem 无 api 表)。
- (可选后续,未做) c-mem.md §4 未加 GrowablePool 代码示例(其他池都有,一致性 nice-to-have);guide/README 其他潜在 stale 未逐一核查。

### 三平台验证 [已做——三平台全绿]
mem 收口(#1/#2A/#2档B/#3/#4/#5b + 文档)三平台真编真测,`hyle_mem_test` 96 + `hyle_mem_c_test` 全绿:
- **[✓] Windows MSVC**:`build/`(Ninja + cl.exe 14.50),vcvars64 + `cmake --build`。主力开发环境。
- **[✓] mac clang**:git 同步(`ssh kiro@172.22.10.198`,`cd ~/hyle && git fetch + reset --hard origin/main`,tracked clean),cmake 在 `/opt/homebrew/bin`,generator = Unix Makefiles。
- **[✓] WSL GCC-14**:**WSL 的 SSH 连 github 认证失败**(fetch 报 access rights —— 这才是"github 不通"的真相),改用 **rsync 增量**从 `/mnt/c/public/project-hyle/hyle/` 同步源码到 `/root/hyle`(exclude .git/build/.deps),复用其 Ninja build(.deps 在 /mnt/c);**坑**:rsync 保留 Windows 旧 mtime → ninja "no work to do" 假象,需 `touch` mem 源刷新才重编。
- 注:WSL `/root/hyle` 的 git HEAD 停在 `74be0ae`(main 老 commit,simd 早期阶段),工作区靠 rsync 更新;`git status` 大量 M 是 rsync 痕迹(工作区新/HEAD 旧),**非开发工作**。

## 给下个 context 的提示
- **工具在长 session 偶发污染**(幻觉 Read / Grep 返 "Prj0" 乱码 / TaskUpdate "not found"):改代码前**重读核实**,别信单次可疑输出。
- **用户已拍板"收口 mem = 全做"(2026-07-18)**。批次顺序:①#5b + last-error 覆盖(**已做**,本 session) → ②#3 GrowablePool 擦除 → ③#4 测试补齐 → ④#2 档 B slab-OOM(高风险,隔离 worktree 子代理) → ⑤文档同步 + 三平台复验。纪律:每档编译验证 + 提交,不攒大摊子。
- 之前已落库:`ad2a05f`(档 B 错误码单一真相源 + UNAUTHENTICATED)、`dec5b36`(thread_spawn_opts + hyle_version + dns/avx512 修复)。
- 三平台环境细节在 .axiomate memory:`reference_wsl_linux_tsan` / `reference_mac_ssh` / `reference_windows_msvc_build`。
