# platform 模块审查处置清单 (followup)

> **后续追记（2026-07-23）：**本文保留审计当时的错误接口结论；DynLib、MappedFile、
> VM/NUMA 已迁到共享 `Status`/`Result` 与 `hyle_status` + out。见
> [`platform-error-convergence.md`](platform-error-convergence.md)。

> **Thread 追记（2026-07-24）：**旧实现把 worker ordinal 当 CPU id 且吞掉配置失败。
> 当前 sched 消费 `available_cpu_ids()` 的真实稀疏 id；Thread 提供 best-effort/required
> 双合同、单 CPU/CPU 集合 affinity 与返回 Status 的 `this_thread::set_*`。当前合同见 guide，
> 本任务单只作历史记录。

> 2026-07-18 对 platform 做四维度体检(① 功能完整+下游利用 / ② 红队实现 / ③ 测试+性能 / ④ C ABI+guide+close)。4 个子代理并行审 + 主进程逐条对代码核实红队与功能关键项。本文件是处置清单 + handoff。

## 体检总体结论

platform **功能完整、C ABI 接近满分**:8 子系统(wait/cpu/numa/virtual_memory/thread/mapped_file/dynlib/fs)能力全有 C 通路、命名/句柄/错误/named-enum/FFI 友好度全面对齐 mem/error 规范、桥接层基本零 reinterpret、close-01 合规。**问题集中在红队实现维**——尤其 1 个 HIGH 级数据丢失。红队 5 条真 bug **零误报**(比 mem 那次干净,mem 有 3 误报)。**收口硬卡点 = #1**。

## 处置进度(2026-07-18 本轮 — 三平台复验已过)
- **[✓ 三平台绿] P0 #1/#2** 递归删除跟随 junction/symlink 数据丢失(4b2e69c):remove_symlink primitive + rm_visit/根 is_symlink 防护 + `RemoveRecursiveDoesNotFollowDirLink` 回归测试。
- **[✓ 三平台绿] P1 #3/#4**(c51b514):join joinable 守卫 + affinity 越界 guard + `JoinNonJoinableAborts` death test。
- **[✓ 三平台绿] P2 #5/copy_file/getpwuid_r**(743b085):dynlib 宽字符(+ 顺带 open(nullptr) 防护) + 自拷贝守卫 + 线程安全。
- **[✓] P3 文档**(e23a06a):api 表 stale / alloc_rw / §3.8 补全。
- **[✓ 三平台绿] P4 核心**(0bb8cd8):#3 join death test;覆盖增强降后续。
- **[P0–P4] 三平台复验**:Windows MSVC 121 / mac clang 124 / WSL GCC-14 124 全绿 + 三平台 c 测试通过。

## 处置进度(2026-07-18 续 — 后续批次 #16/#17/#18 三平台已过)
- **[✓ 三平台绿] #16 边际 LOW guard**(6bd0ed7):page_align_up/large_page_align_up 溢出 / get_env_var Windows 探长度竞态 / rm 深递归上限(kRmMaxDepth=512)/ vm_region 4 个 create wrapper OOM last-error。dynlib/mapped_file/parker wrapper OOM 无公开 set 通路,接受。
- **[✓ 三平台绿] #17 P4 覆盖增强**(234d0a1,子代理补):park_timeout timeout-vs-unpark 竞态压测 / C ABI 薄转发(wait 四导出、vm_protect/decommit、VmRegion decommit-protect、mapped_file 数据路径)/ mapped_file 非对齐 offset+move / VmRegion::adopt 单测 / bench 三热路径。
- **[✓ 三平台绿] #18 fs 通用文件读写**(600ea1d,真缺口):open_file(path,OpenFlags)->OwnedFd + OwnedFd::read/write + read_raw/write_raw 三平台 + C ABI hyle_fs_open/read/write(HYLE_OPEN_* enum)+ 测试 + guide。
- **三平台复验**:Windows MSVC 132 / mac clang 134 / WSL GCC-14 134 全绿 + 三平台 c 测试通过。(WSL 复验坑:`wsl bash -lc '内联'` 会吞 shell 变量致 rsync 退化 `// /`,改用脚本文件;裸 `/mnt/...` 参数被 Git Bash MSYS 转换,改用 PowerShell 调 wsl。)
- **后续(handoff — 本 session 续做)**:**P5 下游错配(sched 不绑核 / dynlib 动机)**、真缺口(numa CPU→node / cpu cache 拓扑)。

## 红队核实结果(主进程逐行复核,非盲信子代理)

真 bug 5 个全部坐实;明确无 bug 项也已复核:
- **[HIGH] #1** Windows 递归删除跟随目录 junction/symlink(见 P0)
- **[MEDIUM] #2/#3/#4** / **[LOW] #5**(见处置清单)
- **无 bug(复核确认)**:cpu.cc CPUID/XGETBV 门控正确、virtual_memory.cc `range_ok` 溢出检查、`VmRegion` move/guarded 清理无泄漏、mapped_file 句柄默认值 -1。
- **待真机(红队自标 ~45%)**:mac `os_sync_wait_on_address_with_timeout` 的 `OS_CLOCK_MACH_ABSOLUTE_TIME` 是相对 ns 还是绝对/mach-tick(wait.cc:124-128)——arm64 时基非 ns,若理解错则 park_timeout 每次立即超时退化忙等。

## 处置清单(按优先级)

### P0 数据安全 [HIGH — 收口 blocker]
- **#1 Windows 递归删除跟随目录 junction/symlink → 删树外数据**。`directory_iterate`(fs.cc:355-356)只把纯 `FILE_ATTRIBUTE_DIRECTORY` 位当 `dir` 传给 visitor,不带 reparse 信息;`rm_visit`(:894) `is_dir ? remove_directory_recursive : remove_file` 照着递归;`remove_directory_recursive`(:901)的 `is_directory` 又跟随 reparse。POSIX 侧(:788-794)有 lstat 防护(子项 symlink 报 dir=false),Windows 侧完全没有 → 删掉链接**目标目录(树外)**内容。**修法**:Windows `directory_iterate` 的 entry 带 `is_symlink`(查 `FILE_ATTRIBUTE_REPARSE_POINT`),删除路径对"目录+reparse"当非目录处理(删链接本身不递归);配 death/行为测试(建 junction/symlink 验不跟随)。
- **#2 POSIX 根路径为 symlink-to-dir 时同样跟随删目标**(fs.cc:901 `is_directory` 跟随、根未 lstat)。与 #1 同一 reparse 防护缺口,一起修:`remove_directory_recursive` 开头用 lstat/`is_symlink` 判根,是链接则 remove_file 不递归。

### P1 正确性 [MEDIUM]
- **#3 `Thread::join()` 无 `joinable()` 守卫**(thread.cc:114-117 Win / :241-242 POSIX):对 non-joinable 线程 join,POSIX `pthread_join(0)` 退化 UB、Win 静默语义错。同文件 dtor(:306)/move(:296)都有 `HYLE_FATAL_ASSERT(!joinable())`,join 漏了。**修法**:join 开头加 `HYLE_FATAL_ASSERT(joinable())`(C 桥 hyle_thread_join 已有守卫,仅裸 C++ API 暴露)。
- **#4 亲和性越界**:Windows(thread.cc:76)`DWORD_PTR(1) << s->cpu`,cpu≥64 移位 UB(x86 掩码→静默绑 CPU 0),且 >64 核需 processor group 寻址;Linux(:167)`CPU_SET(cpu)` cpu≥1024 越界写。**修法**:边界检查 + Windows processor group(或至少 reject/文档化 ≥64)。

### P2 一致性/轻微 [LOW — #5/copy_file/getpwuid_r 已做;其余边际记录]
- **#5 dynlib 非 UTF-8 [已做]**:open/get 转 `LoadLibraryW`/`GetModuleHandleW` + `to_wide`(MultiByteToWideChar),current 转 `GetModuleHandleW(nullptr)`;顺带修 open(nullptr) 未防护(to_wide null 检查)。Windows DynLib 14 测试绿。
- **copy_file(x,x) 自拷贝截断 [已做]**:POSIX 先 open/fstat 验证 regular source，再以同路径 no-op；不同名字仍以 dev+inode 防 hard-link alias，既避免 O_TRUNC 截空，也不把 `copy_file(missing, missing)` 误报成功。Windows CopyFileW 本就失败安全。Windows/Linux 已复验，macOS 待门禁。
- **getpwuid→getpwuid_r [已做]**:home_directory 改线程安全的 getpwuid_r。POSIX 段待三平台复验。
- **[边际 LOW — 已处置(后续批次)]**:
  - **[已做]** page_align_up/large_page_align_up 溢出 guard(vm.h:`n > SIZE_MAX-(p-1)` → 0)。
  - **[已做]** get_env_var Windows 探长度竞态守卫(`got >= need` 即并发变长 → 返空,不越界读)。
  - **[已做]** remove_directory_recursive 改 resumable native-iterator postorder DFS，移除递归调用栈与固定 512 深度限制；每层一个 OS iterator、所有层共享一个 path buffer，普通文件/link 立即删且不排队同层兄弟，辅助内存 `O(depth + max_path)`。300 层与 2048-file 宽目录回归已过 Windows/Linux；frame/path OOM 或 OS handle/fd 耗尽返回 ResourceExhausted。
  - **[部分已做]** wrapper OOM last-error:vm_region 4 个 create 已设(vm_set_last_error);dynlib(内部 g_error 无公开 set)/mapped_file(无独立 last-error)/parker(无 error 通道)的 wrapper OOM 无 set 通路,wrapper 极小、OOM 罕见,**接受现状**。

### P3 文档 stale
- milestone platform-03b:83 还写"导出到 `hyle_platform_api` 表"(design §3 已删表、实物无表)。
- README.md:13 **c-platform 索引行**仍写"`hyle_platform_api` 表"(platform 无此表,与 milestone:83 同一处 stale)。〔扫描核实:子代理原报"README:16 mem 串 platform.h"是**误报**——:16 是 simd-usage 行、:14 mem 行已正确;真 stale 在 :13〕
- **[已做]** c-platform.md:31 `alloc_rw` 从"VM 自由函数"移除(它只是 C ABI 便利 `hyle_vm_alloc_rw`,C++ header 无此自由函数)。
- **[已做,子代理误报修正]** c-platform.md §3.8"有意不导出"表**本就存在**(子代理"缺表"是误报——表里已有 Unparker/path/模板/测试钩子);已补全遗漏项:`symbol_as` / `OwnedFd` / `VmRegion::release`·`MappedFile::reset`。
- 〔以上 milestone:83 / README:13 亦已做:分别加勘误注、去 `hyle_platform_api` 表〕

### P4 测试/性能补齐 [#3 join death test 已做;其余覆盖增强记后续]
- **[已做]** #3 join-non-joinable death test(`ThreadDeathTest.JoinNonJoinableAborts`)锁死 P1 join 守卫。**本轮所有修复均有测试覆盖/锁死**:#1/#2=`RemoveRecursiveDoesNotFollowDirLink`(P0)、#3=此 death test、#4=现有 `AffinityOutOfRangeDoesNotCrash`、#5=DynLib 测试。
- **[后续覆盖增强,非 bug]**(原有代码覆盖不足,与本轮修复无关;核心修复已验证故降后续):
  - park_timeout timeout-vs-unpark 并发压测(功能测试已有 6 个,仅缺并发竞态一项)。
  - C ABI 薄转发零覆盖:低层 wait 四导出、vm_protect/vm_decommit、VmRegion decommit/protect、mapped_file 数据路径(valid/data/size/flush)。
  - mapped_file 非页对齐 offset + move-assign;thread spawn 失败(EINVAL);VmRegion::adopt C++ 单测;fs copy/rename/change_dir 失败路径。
  - bench:park_timeout / vm_protect / mapped_file 三热路径无 bench。

### P5 下游利用错配 [✓ 已接上 — #21 fd9ea83]
- **[✓ 已修正] 调度器 worker 绑核/命名**：最初版本传 `affinity_cpu=i`；后续回审发现 Linux
  稀疏 cpuset 下 ordinal 不是 OS id。当前按 `available_cpu_ids()` 顺序绑定，超额 worker
  保持可迁移；`this_thread::set_affinity` 也已提供 Status-returning 的显式重绑机制。
- **[✓ 已登记] dynlib 动机落空**:dynlib.h 头注登记为 FFI-only 铺垫(SIMD 实为进程内 TU + cpu_features 表派发,非 dlopen)。保留因其为干净、已测、有真实 FFI 用途的 platform 能力。
- 其余仅-C ABI 暴露(thread stack/priority、numa bind/interleave、mapped_file/fs 整套、vm_free_rw)多为面向 FFI 嵌入者的合理超前铺垫,非错配。

### 真缺口 [✓ 全部补齐]
- **[✓ 已做 #18 600ea1d]** fs 通用文件读写:open_file + OwnedFd::read/write + C ABI。
- **[✓ 已做 #19 473053a]** numa 逻辑 CPU→节点查询:numa_node_of_cpu + C ABI。
- **[✓ 已做 #20 8822a3e]** cpu 运行时 cache-line:cache_line_size + C ABI。〔cache-size L1/L2/L3 分级、P/E 核区分**未做**——边际价值,未纳入本轮;cache-line 是 false-sharing padding 的核心需求,已满足。〕

### close-02 [搁置]
用户澄清 close-02 = "继续做 C ABI",暂不作为独立验收门(close-01 已合规,验收 = c-abi-authoring-guide §7 checklist)。

## 批次规划(建议处置顺序 — 待用户拍板)

纪律沿用 mem 收口:小步、每批编译验证 + 提交落库、高风险隔离、收尾三平台复验。

1. **批次1 [P0 数据安全]**:#1 + #2 递归删除 reparse 防护。**最高风险**(改删除逻辑,改错=数据丢失),必配行为测试(建 junction/symlink 验不跟随)+ 三平台格外仔细。**最优先**。
2. **批次2 [P1]**:#3 join joinable 守卫 + #4 affinity 越界。
3. **批次3 [P2]**:#5 dynlib 宽字符 + wrapper OOM last-error + 选定 LOW guard。
4. **批次4 [P4]**:测试/bench 补齐(尤其 P0/P1 的回归测试)。
5. **批次5 [P3]**:文档 stale 同步。
6. **批次6**:三平台复验(方法见 [mem-audit-followup.md](mem-audit-followup.md) 三平台节:Windows MSVC / mac clang ssh git 同步 / WSL GCC-14 rsync 增量)。
- **[P5 下游错配]** sched 绑核+命名:单独议(动 sched 模块,非 platform 收口范围)。
- **[真缺口增强]** fs read/write、numa CPU→node、cpu cache 拓扑:登记,非收口必须。

## 给下个 context 的提示
- **工具偶发污染**:维度1 子代理首次 Read thread.h/vm.h/numa.h 返回**旧版本内容**(旧设计:std::function/detach、alloc_rw 自由函数、numa_alloc_on_node),grep 交叉核对后校正——改前重读核实,别信单次可疑 Read。
- **#1 是数据安全 blocker**,platform 收口前必须修。
- 三平台环境:WSL github SSH 认证失败(非网络),走 rsync 增量 + touch mtime;详见 mem-audit-followup.md。
