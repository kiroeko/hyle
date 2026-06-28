# clay → HYLE 覆盖映射表（吸纳完整性的持续标尺）

> **目的**：防"以为吸纳完了、其实漏文件/漏 API 层"。clay 源在 `slate-render/iron/infra/{include,src}/clay/`。
> **铁律（对抗审查纪律）**：每个吸纳 clay 的里程碑闭环前，必须 `diff` clay 同名文件的 API 表面——缺的要么补、要么在本表显式登记 `📋待办`，**不许默默略过**。本表是唯一权威进度标尺。
> **状态图例**：✅ 完成 / 🔶 部分(有登记的缺口) / 📋 计划 / ♻️ 被 C++23/std 取代 / 🔻 借第三方(不自写) / ⛔ 不吸纳。
> **行数**：clay 列为该文件总行数(`grep -c`)；HYLE 列为对应物生产行数。clay 自有(去 thirdparty/plugins)≈ **19,000**；HYLE 当前生产 ≈ **2,885**(container-06 +363 + util-01..05 +441 + time-01 +103 + platform-02/dynlib +158 + mem-02 增量 + mem-aux/alignment +36;mimalloc 第三方不计)。差距大部分是"未做"(进度)，小部分是已修正的"缩水"(见 🔶)。

## sys/ — 系统基座（allocator/arena/os/path/string/hash/log/time）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `sys/allocator.h` + `.cc` | 485+375 | mem-01 + mem-01b + mem-02 | ✅ | vtable+裸 allocate + typed 层(mem-01b) + **backing 换 mimalloc(mem-02,封 vtable 后不外露,Win 4–12x>默认堆/Linux 真实模式 5.2x)**;CUSTOM_MEMORY_FUNCTIONS 仍 📋(次要) |
| `sys/memory_arena.h` + `.cc` | 65+258 | mem-03 + mem-03b | ✅ | `MemoryArena` 块池(packed CAS) + **`Arena` bump 区域分配器 + as_allocator() 适配已补(mem-03b)**;`make_arena_allocator` 等价为 `Arena::as_allocator()` |
| `sys/carray.h` + `.cc` | 436+65 | container-03 | ✅ | header-before-data 侵入数组 + type-erased 核 + RAII 壳 CArray<T>;**改进:over-align 元素(clay 固定 max_align_t 会错位)** |
| `sys/cstring.h` + `.cc` | 1033+374 | container-06 | ✅ | string_ref(ptr+len+null 位)+零拷贝算法(find/rfind/find_first/last_of/not_of/substr/trim/starts/ends_with/contains/count/split/rsplit/slice/concat/dup/c_str/equal_ignore_case)+ operator<=> + Hash<string_ref> + `_sr` 字面量。**类型核心 API diff 过 clay**;clay `capitalize`/UTF-8/hex/escape 真实现归 `string_algo.h`(container-06b✅);`score_similarity`/`format_si` 等 **clay 只声明无实现**→container-06c(净新) |
| `sys/string_buf.h` + `.cc` | 141+363 | container-06 | ✅ | 引用计数驻留池 `StringBuf`(pimpl 藏 SwissTable 反查+mutex)+ make/destroy 工厂;每唯一串单独分配(地址稳定)。**改进**:hash 命中 memcmp 校验(XXH3 碰撞 fail-fast 而非静默别名);复用 container-04 SwissTable→.cc 仅 83 行(非 stub,是复用红利)。**全局单例 intern/usprintf 📋待补**(见下) |
| `sys/hashmap.h` | 809 | container-04 | ✅ | concept 模板重写 + **实现 clay 只 TODO 的 SwissTable**(控制字节+SIMD探测+三角探测+tombstone);type-erased core 延迟 container-04b |
| `sys/murmur_hash.h` | 294 | container-02 | 🔶 | **运行期哈希→XXH3 已接(container-02✅,借 xxHash 替 murmur)**;编译期 type 哈希→**FNV-1a constexpr 已做(util-02 type_hash)**;murmur 本体未移植(无剩余用途) |
| `sys/error.h` + `.cc` | 222+143 | **error-02** | ✅ | error_handler vtable(跨 DLL 诊断 sink)。改进:非变参 vtable(ABI 安全,clay 变参函数指针隐患)、干净 fatal(HYLE_TRAP vs clay rand+空写+exit)、补 charter §5 `HYLE_ASSERT`、record handler 无分配。NOT_IMPLEMENTED 等便利宏 gap 登记 |
| `sys/os.h`+`os.linux.cc`+`os.shared.cc` | 795+830+210 | platform-02/03b | 🔶 | **dynlib ✅(platform-02)**;**path/fs/env/os_info→platform-03b**(file_system_api/getcwd/absolute/temp_dir/app_folder/directory_entries 等)。**virtual_memory(reserve/commit)clay 无此 API**(os.h 只有 mmap_file=映射文件,io 域)→ HYLE-new,随 mem-05 MemorySpace/mem-06 大页落地,非 clay 吸纳项;mmap_file→io-05 |
| `sys/path.h` + `.cc` | 202+1372 | platform-03a + platform-03b | 🔶 | **vtable `path_api_t`(16 函数)纯字符串半 ✅(platform-03a,落 container/path.{h,cc},232 行)**:extension(splitext)/strip/base/directory/parent_dir/pop_first/pop_last/split_all/join/join_array/normalize(native 折叠)/is_absolute(升级自 clay blend `is_relative`)/is_unc。修 clay directory 的 `//x` OOB。**vtable 外 ~1100 行**(path_rel blend⛔/normalize_unc+宽字符📋/make_safe_filename📋/sequence_decode 渲染域⛔/path_contains+cmp_normalized📋/Win 盘符+native sep📋)→ platform-03b |
| `sys/log.h` + `.cc` | 150+178 | obs-01 / platform | 📋 | 日志 |
| `sys/random.h` + `.cc` | 112+143 | util-04 | ✅ | `Rng`(xoshiro256++ 升级 clay xoroshiro128+)+ **无偏 range(Lemire,修 clay `%` 偏差)** + SplitMix64 seed + jump + thread_rng。header-only |
| `sys/time.h` + `.cc` | 47+112 | time-01 | ✅ | **单调钟(monotonic_ns,clay 缺)** + 墙钟(realtime_ns) + sleep_for_ns + Instant;ns 精度替 clay float us/ms |
| `sys/sprintf.h` + `.cc` | 96+69 | ♻️ std::format | ♻️ | clay sprintf=`stb_sprintf` 薄封装;头广告的 "UTF-8 安全截断" **实为注释掉未实现**(.cc 行42/54,clay 美化现状一例)。HYLE:格式化→`std::format`/`std::format_to_n`(类型安全 `{}`+编译期查错=no-throw+buffer-fill 截断等同 snprintf,conventions §8);printf-`%`/va_list 转发→`std::snprintf`(C 标准已有)。不投机 vendor stb_sprintf(快 printf 属实测决定);真 UTF-8 安全截断归 container-06b string_utils |
| `sys/scope_guard.h` | 74 | util-01 | ✅ | `ScopeGuard` CTAD(省 clay scope_guard/finally 工厂) + `HYLE_DEFER`;单向 dismiss(enable 重启用未做,非缺口) |
| `sys/alignment.h` | 37 | mem-aux | ✅ | `align_up/align_down/is_aligned`(size + `_ptr` 变体)+ is_pow2 + kMaxSimdAlignment;升级 clay 只有 align_size(独立头已建,mem/alignment.h) |
| `sys/demangle.h` + `.cc` | 28+45 | container/util type_name | ✅ | **util-06**：运行期 demangle(`__cxa_demangle`/`UnDecorateSymbolName`)。改进:buffer-fill 无分配(clay 用 Allocator+std::function deleter)、MSVC 真反修饰(clay 只回显)。`demangle_type<T>` gap→`type_name<T>`(util-02)更优替代(无 RTTI) |
| `sys/murmur_hash.h` | (见上) | | | |
| `sys/assert.cc` | 31 | macros.h | ✅ | HYLE_FATAL_ASSERT 已覆盖等价 |
| `sys/types.h`/`decls.h`/`version.h` | 67/72/55 | 00-scaffolding | 🔶 | platform.h/visibility.h/version.h 覆盖思路；decls.h include 纪律待坐实 |

## utils/ — 工具层（status/result/pool/类型设施/registry）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `utils/status.h` + `.cc` | 299+182 | error-01 | ✅ | Status(8B 指针标签,已现代化) |
| `utils/result.h` + `.cc` | 470+18 | error-01 | ♻️ | `std::expected` 取代手写 Result(charter 定) |
| `utils/object_pool.h` | 205 | mem-04 | ✅ | 修 bug#3(坐 MemoryArena) |
| `utils/stack_allocator.h` | 122 | mem-04 | ✅ | 修 bug#4(禁拷贝、无堆回退) |
| `utils/memory_pool.h` | 170 | mem-07 | ✅ | **`GrowablePool<T>`**:growable+single-threaded 池(无原子,填 ObjectPool 固定+lock-free 之外的空位)。几何块+侵入 free-list。**修 clay move-assign bug**(从不转移 allocator_) |
| `utils/ref_counted.h` | 317 | util-01 | ✅ | 侵入式 `RefCounted<Sub>` CRTP + `Ref<T>` + make/adopt/retain_ref。**改进**:修 clay destroy() 不 CRTP-dispatch bug、decref release+末次acquire-fence(性能)、删隐式 operator T*/全局计数器、concepts 替 SFINAE |
| `utils/registry.h` | 340 | util-03 | ✅ | 自注册 + DECLARE/DEFINE 宏。**改进**:SwissTable 替 3 个 unordered_map、Creator 函数指针(Entry 内联,非 std::function)、同优先级 fail-fast 替 std::exit、Meyers 替 leaky new、三套(typed/ref/shared)收一套泛型 |
| `utils/type_traits.h` | 240 | (编译期) | 🔶♻️ | 多数被 C++23 concepts 取代；保留底层兼容部分待评 |
| `utils/type_name.h` | 146 | util-02 | ✅ | 编译期 type_name<T>()(probe 派生偏移,比 clay 硬编码签名矩阵 robust) + type_hash<T>()(FNV-1a constexpr) |
| `utils/func_traits.h`/`meta_funcs.h`/`C++17.h`/`comparable.h` | 105/80/253/53 | (编译期) | ♻️ | 多数被 C++23(concepts/`<=>`)取代,按需保留 |
| `utils/string_utils.h` + `.cc` | 673+449 | container-06b | ✅(真实现部分) | 字节串算法吸纳 `string_algo.h`:hex_encode/escape/capitalize/UTF-8(decode/length/icmp)/ASCII case+分类器(全 buffer-fill 无 std 流)。**重大发现**:clay `format_si/find_path_matches/cscore_path_match/cstr_score_similarity` **只声明无实现**(src/ 全无),与 sprintf 假截断同款美化→无可吸纳,净新设计登记 container-06c。str/join/to_string→std::format♻️;stoi/stod→std::from_chars♻️;to_utf8/from_utf8(wchar)→platform-03 |
| `utils/generic_allocator.h`/`stl_allocator.h` | 62/196 | mem-07 | ✅⛔ | **stl_allocator→`mem::PmrResource`**(std::pmr::memory_resource 包 Allocator,一个 resource 服务所有 T,取代每类型 rebind 样板)。**generic_allocator→⛔**:HYLE `Allocator` 值句柄已是它(type-erased {fn,state} 转发) |
| `utils/singleton.h` | 133 | (util 待排) | 🔶♻️ | Meyers 单例(C++11 起 std 保证),薄封装待评 |
| `utils/spin_lock.h` | 43 | util-05 | ✅ | `SpinLock`(TTAS + cpu_relax pause + cache-line padding,升级 clay 裸 TAS)满足 Lockable 配 std::lock_guard |
| `utils/vector_utils.h`/`device_utils.h`/`float16.h` | 98/125/52 | 各相关 | 📋 | 杂项工具 |

## parallel/ — 并发（thread_pool/async_value/concurrent_vector）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `parallel/thread_pool.h` + `.cc` | 130+167 | sched/ | ♻️ | 单全局锁池 → **被 sched-03 work-stealing 调度器取代(重写非移植)** |
| `parallel/worker_thread.h` + `.cc` | 88+109 | sched/ | ♻️ | 同上,被 sched worker 取代 |
| `parallel/async_value.h` + `.cc` | 465+42 | sched-04/05 | 📋 | promise/future + waiter → completion 脊/同步原语 |
| `parallel/concurrent_vector.h` | 145 | container-09 | ✅ | 读优化 append-only 并发 vector(读 lock-free/写 SpinLock 串行)。**分段存储**修 clay 五问题:外层 reserve(65)→定长 C 数组(结构性不变量)、copy-on-grow 2x 内存→零拷贝 1x、T copyable→move-only 可、每读边界检查→去除、std::vector→Allocator。**WSL TSan 零 race**。归 container(不依赖 sched)|
| `parallel/barrier.h` | 46 | sched 原语 | 📋 | 屏障 |

## math/ — 数学（half/utils/consts）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `math/half.h` | 6923 | simd-05 | 🔻♻️ | fp16 软件查表(巨)→ **硬件 F16C/AVX512-FP16 取代,不 1:1 移植**;只留必要接口 |
| `math/utils.h` | 353 | math-01 | ✅♻️ | **绝大多数被 C++20/23 取代**(现代化红利):bit 操作→`<bit>`、clamp/minmax→`<algorithm>`、pow/abs→`<cmath>`、align_up→mem/alignment。std 真缺的 4 个自写进 `math/scalar.h`:`div_ceil`(改溢出安全)、`round_up_to_multiple`、`sgn`(死区)、`approx_eq`(相对eps);deg2rad 等图形域⛔ |
| `math/consts.h` | 105 | math-01 | ♻️ | clay 头自注"same as C++20 `<numbers>`, will be replaced"——`<numbers>` + `std::numeric_limits` 取代 |

## 已修正的"完成"状态（红队核账发现的缩水）

- **mem-03**：从"✅完成"改"🔶块池完成、Arena bump 待补(mem-03b)"。
- **mem-01**：标注 typed 便利层待补(mem-01b)。
- **error-01**：标注 error_handler vtable 待补(error-02)。

## 新登记的补漏里程碑

- **mem-01b**：allocator typed 便利层(`allocate<T>/construct<T>/make_unique<T>/destroy<T>` + pmr 钩子)。
- **mem-03b**：`Arena` bump 分配器 + `ArenaAllocator` 适配 + `make_arena_allocator`。
- **error-02**：error_handler vtable(跨 DLL 错误上报)。
- **container-06b**：`string_utils` 高层字符串工具(score_similarity/path_match/format_si/capitalize/replace/find_path_matches)+ `StringBuf` 全局单例 intern/usprintf。container-06 只做 string_ref 类型核心 + StringBuf 实例,这些高层/全局便利层显式登记待补,**非略过**。

## "clay 完全吸纳"在 P 几（见本文件末尾结论由 milestones/README 给出）

## HYLE vs clay 的计划性差异（关键：吸纳 ≠ 追平 LOC）

> **"完全吸纳 clay" 不等于 "HYLE LOC 追平 clay"。** 标尺只对"该吸纳的那部分"成立——对那些组件，HYLE 明显小于 clay = 缩水嫌疑(要查)。但整体 HYLE 会**超过** clay，因为三类计划性差异：

### ① 砍掉 / 替换（HYLE 比 clay 少的部分——合理，不是缩水）
- `utils/result.h`(470)→ `std::expected`；`comparable.h`(53)→ `operator<=>`；`C++17.h`/`meta_funcs`/`func_traits` 大半 → C++23 concepts；`singleton.h` → Meyers；`type_traits` 大半 → concepts。**clay 是 C++17 手搓 SFINAE，HYLE 是 C++23，这些"消失"是现代化红利,不是漏做。**
- `parallel/thread_pool`+`worker_thread`(约 494 行)→ **废弃**，被 sched-03 work-stealing 调度器**重写取代**(单全局锁长不出统一调度器)。
- `math/half.h`(6923 软件查表)→ **不 1:1 移植**，硬件 F16C/AVX512-FP16 取代,只留接口。
- catomics(clay sys/details,未在上表)→ **不 port**，C++23 std::atomic 取代。
> 合计:clay 里约 **8,000+ 行**属于"HYLE 计划性不吸纳/大幅替换"——所以拿 19k 当目标分母是错的，**真实"应吸纳"分母约 10-11k**。

### ② 新增（clay 根本没有——HYLE 最终远超 clay 的原因）
- **fiber 有栈上下文切换**(fcontext)、**M:N work-stealing 调度器**(per-core 队列+偷取+park)——clay 零 fiber、只有单锁池。
- **lock-free 容器簇**:Vyukov MPMC、Chase-Lev deque、tokio FIFO ring——clay 只有读优化加锁 concurrent_vector。
- **io/**:proactor + io_uring/IOCP/epoll/kqueue + net + 跨进程共享内存——clay 完全没有。
- **compute/**:device/stream + ggml bridge + GPU completion→fiber + Vulkan kernel——clay 完全没有(0 个 .cu)。
- **simd/**:D-tag 抽象 + 多 ISA 后端 + 运行期 dispatch——clay 无 SIMD 抽象。
- **取消/结构化并发/优雅关闭**、**C-ABI 纯 C 头 shim**、**FFI 绑定(TS/Python/Rust)**——clay 无。
> 这些是 hyle-scope/roadmap 的 P4-P6 主体，**估 30-50k 行净新增**，clay 无对应物。

### ③ 升级（同一组件,HYLE 比 clay 更强——LOC 可能持平或更多）
- `hashmap`:clay 线性探测 → HYLE 补 SwissTable SIMD 三角探测(更复杂)。
- `MemoryArena`:clay 16字节 atomic+cmcx16 → HYLE packed 64位 CAS(更稳)。
- device 内存:clay 无 → HYLE MemorySpace 组合抽象。

### 结论：正确的"吸纳完整性"判据
- **不是** "HYLE 总 LOC ≥ clay 19k"。
- **而是** 对每个 ✅/🔶 组件，HYLE 覆盖了 clay 同名文件该吸纳的 API 表面(diff 核对)；clay 中 ♻️/⛔/🔻 的部分按计划替换/不吸纳并登记在本表。
- **吸纳完成** = 本表 sys/utils/parallel/(math 按需) 中**所有非 ♻️/⛔ 行变 ✅**，且每行经过 clay 同名文件 API diff。当前:`status/object_pool/stack_allocator/carray ✅`、`allocator/memory_arena ✅(补漏后)`、`murmur 🔶(运行期 XXH3 已接)`,其余 📋。
- **P3 = 真·吸纳,不是"略过缺的就算实现"（用户 2026-06 铁律）**：到 P3 末必须是"**该实现的都实现了 + 该替换的脚手架都换了 + 比较完整**",而**不是把缺的部分都略过就宣布完成**。每个标 ✅ 的组件,缺的子功能要么真做、要么显式登记为补漏里程碑(如 mem-03b/04b),**绝不默默略过当完成**(mem-03 曾犯,已纠)。残留脚手架(std::string_view 顶 string 等)在等价物落地后必须替换,红队每里程碑扫一遍。

## "clay 完全吸纳"在 P 几（结论）

clay 各组件落到 HYLE 依赖层(P0-P6):

| clay 模块 | HYLE 的 P 层 | 吸纳完点 |
|---|---|---|
| `utils/status`+`result`+`sys/error` | P0 (error-01✅ / error-02) | **P0 收尾** |
| `sys/allocator`+`memory_arena`+`utils/object_pool`+`stack_allocator`+`memory_pool` | P2 (mem-01..06 + 01b/03b) | **P2 末** |
| `sys/os`+`path`+`time`+`log`+`random` | P1/P2 (platform-02/03, time-01) | **P2 末** |
| `sys/carray`+`cstring`+`string_buf`+`hashmap`+`murmur`、`utils/string_utils`+`ref_counted`+`registry`+`type_name` | P3 (container-02..06 + util) | **P3 末** |
| `parallel/thread_pool`+`worker_thread` | P4 (sched,**重写取代**) | 已取代(sched-03✅) |
| `parallel/async_value`+`barrier`+`concurrent_vector` | P4-P5 | **P5 初** |
| `math/utils`+`consts` | 后期 math 簇 | 排期后 |
| `math/half` | P3 simd-05(硬件取代🔻) | P3(留接口) |

**结论(直接回答):**
- clay 的 **sys + utils 主体**(分配器家族 + 全部容器 + string + hashmap + registry + ref-counted + 类型设施)→ **到 `P3 末` 完全吸纳**。这是 clay 地基核心的大头、最该对照行数核账处。
- clay **parallel/** → **P4 被 work-stealing 调度器重写取代**(thread_pool,已 sched-03✅)+ **P5 初收掉 async_value**(→ completion 脊)。
- clay **math/** → half 在 **P3 由硬件 fp16 取代**(只留接口);utils+consts 排后期 math 簇。
- **一句话:盯 `P3 末`(container-02..06 + mem 全部 + 01b/03b)——跑完且每文件 diff 过 clay,clay 主体即吸纳完毕;`P5 初` 补完 async_value 为完整收口。** P6(graph/C-ABI/FFI)是 HYLE 净新增,与 clay 吸纳无关。
- **当前位置:P0-P4 部分推进(12 里程碑),clay 吸纳实际刚起步——离 `P3 末` 还有 container 全簇 + mem-05/06 + 01b/03b + platform-02/03 + time 等一大段。"吸纳完 clay"远未到,之前我表述过于乐观,据此修正。**
