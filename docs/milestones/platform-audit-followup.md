# platform 模块审查处置清单 (followup)

> 2026-07-18 对 platform 做四维度体检(① 功能完整+下游利用 / ② 红队实现 / ③ 测试+性能 / ④ C ABI+guide+close)。4 个子代理并行审 + 主进程逐条对代码核实红队与功能关键项。本文件是处置清单 + handoff。

## 体检总体结论

platform **功能完整、C ABI 接近满分**:8 子系统(wait/cpu/numa/virtual_memory/thread/mapped_file/dynlib/fs)能力全有 C 通路、命名/句柄/错误/named-enum/FFI 友好度全面对齐 mem/error 规范、桥接层基本零 reinterpret、close-01 合规。**问题集中在红队实现维**——尤其 1 个 HIGH 级数据丢失。红队 5 条真 bug **零误报**(比 mem 那次干净,mem 有 3 误报)。**收口硬卡点 = #1**。

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
- **copy_file(x,x) 自拷贝截断 [已做]**:POSIX copy_file 开头 `strcmp(from,to)==0` 守卫(避免 O_TRUNC 截空);Windows CopyFileW 本就失败安全。POSIX 段待三平台复验。
- **getpwuid→getpwuid_r [已做]**:home_directory 改线程安全的 getpwuid_r。POSIX 段待三平台复验。
- **[边际 LOW,本轮未做,低优先]**(已核实真,但价值/成本比低):
  - wrapper OOM 未设 last-error:实害极小(wrapper 结构体极小、OOM 罕见)+ 各模块 last-error 机制不一(dynlib g_error 内部无公开 set / vm 有 vm_set_last_error / parker 无)——补全需先统一机制。
  - page_align_up 溢出(vm.h:38-49):触发需 ~16 EiB,不现实。
  - get_env_var Windows 探长度竞态(fs.cc:286-300):需并发改同一 env 变量,罕见。
  - 深递归无深度上限:需重构递归加深度参数;深目录树(Windows ~170 层)才触发。

### P3 文档 stale
- milestone platform-03b:83 还写"导出到 `hyle_platform_api` 表"(design §3 已删表、实物无表)。
- README.md:13 **c-platform 索引行**仍写"`hyle_platform_api` 表"(platform 无此表,与 milestone:83 同一处 stale)。〔扫描核实:子代理原报"README:16 mem 串 platform.h"是**误报**——:16 是 simd-usage 行、:14 mem 行已正确;真 stale 在 :13〕
- c-platform.md:31 把 `alloc_rw` 列作"VM 自由函数"(实际只 C ABI `hyle_vm_alloc_rw`,C++ header 无此自由函数)。
- c-platform.md 缺"有意不导出"表(authoring §7 要求登记 Unparker/symbol_as/OwnedFd/VmRegion::release 等仅-C++ 项)。

### P4 测试/性能补齐
- **错误/失败路径**:thread spawn 失败(EINVAL)+ 析构/move joinable 的 death test、fs copy_file/rename/change_directory 失败、dynlib open(nullptr)(实现未防护)、VmRegion::adopt 的 C++ 单测、mapped_file 非页对齐 offset + move-assign。〔核实修正:thread 的 affinity 读回(Linux)/stack_size/affinity_cpu=4096 边界**已有测**(thread_test.cc:50-88/137/168),子代理"affinity/stack 完全未覆盖"**不准**〕
- **并发竞态**:park_timeout 的 **timeout-vs-unpark 并发压测**缺。〔核实修正:park_timeout **功能测试其实有 6 个**(wait_test.cc:102-132/259),子代理"仅单次"**不准**;真缺的只是并发竞态一项〕
- **C ABI 薄转发**:低层 wait 四导出(wait_on_address/_for/wake_one/wake_all,含 as_atomic reinterpret)、mapped_file 数据路径(valid/data/size/flush)、VmRegion decommit/protect/commit_usable、vm_decommit/vm_protect 零覆盖。
- **bench 缺口**:park_timeout、vm_protect、mapped_file 三个热路径无 bench。
- (可选)Thread 析构/move fail-fast death test、alignment guard 等。

### P5 下游利用错配 [跟进 — 非 platform 缺陷,是"造了没接上"]
- **调度器 worker 不绑核/命名**:scheduler.cc:232 `Thread::spawn(&worker_main, &ctxs[i])` 用默认 `ThreadOpts{}`,未设 affinity_cpu/name;而 affinity_cpu(thread.cc:48 接线完好)按 platform-01 D3 是"sched per-core pinning 刚需"。**platform 侧功能就绪,需 sched 侧接上**(spawn 传 affinity_cpu + name);注意当前无"当前线程重绑"API,只能 spawn 时设。
- **dynlib 动机落空**:设计为 dlopen CPUID 选中的 SIMD 后端(dynlib.h:14-18),但实际派发是进程内 TU + cpu_features 运行时表(runtime.cc),无任何 dlopen。接上或登记为 FFI-only 铺垫。
- 其余仅-C ABI 暴露(thread stack/priority、numa bind/interleave、mapped_file/fs 整套、vm_free_rw)多为面向 FFI 嵌入者的合理超前铺垫,非错配。

### 真缺口 [功能增强 — 非收口必须]
- fs 无通用文件读写(OwnedFd 无 read/write;读文件只能 mmap)。
- numa 无"逻辑 CPU→节点"查询(旧 `numa_node_of_cpu` 已删)。
- cpu 无运行时 cache-line/cache-size、无 P/E 核区分。

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
