# clay → HYLE 覆盖映射表（吸纳完整性的持续标尺）

> **目的**：防"以为吸纳完了、其实漏文件/漏 API 层"。clay 源在 `slate-render/iron/infra/{include,src}/clay/`。
> **铁律（对抗审查纪律）**：每个吸纳 clay 的里程碑闭环前，必须 `diff` clay 同名文件的 API 表面——缺的要么补、要么在本表显式登记 `📋待办`，**不许默默略过**。本表是唯一权威进度标尺。
> **状态图例**：✅ 完成 / 🔶 部分(有登记的缺口) / 📋 计划 / ♻️ 被 C++23/std 取代 / 🔻 借第三方(不自写) / ⛔ 不吸纳。
> **行数(cloc 实测)**：clay 自有 ≈ **19,005**(sys 7917 + math 5831 + utils 3469 + parallel 811 + 顶层头 973;去 thirdparty)。HYLE 生产(src+include)≈ **8,130**（2026-06-29 cloc 复核;含 sched-05 同步原语 + conditional-notify + TSan fcontext 注解;旧值 8,048/7,757 已 stale）。mimalloc/xxhash/nanobench/fcontext/sleef 第三方不计。
>
> **⚠️ 进度口径(2026-06-29 重核，旧 tally 已 stale)**：按**实际表行**重数(52 个 clay 文件行,非旧"94 符号"细粒度估):
> - **含 ✅(写了 HYLE 等价物)= 28/52 ≈ 54%**(旧称 50%,里程碑推进后多行翻 ✅,旧 tally 偏低)。
> - **已决议(✅ + ♻️std取代 + 🔻硬件取代 + ⛔不吸纳)≈ 42/52 ≈ 81%**——含"取代/借/不要",不等于"写了代码"。
> - **剩余未决:仅 4 行 🔶(有登记缺口)+ 5 行 📋**,且 2026-06-29 逐文件复核判定**这些基本是终态或归别处**:murmur(XXH3 已替,无剩用途)、type_traits/singleton(C++23 concepts/Meyers 取代)、float16/device_utils(CUDA 域→compute)、vector_utils(concepts 取代)、async_value/barrier(归 P5 sched)。**真·clay 吸纳欠账≈零。**
> - **符号无行级权重**:`math/half.h` 一行 🔻♻️ = clay 6923 行(软件 fp16→硬件 F16C/AVX512 取代,不移植,且 simd-05 dtype 已落可移植 fp16/bf16);`parallel/thread_pool+worker`(494 行)→ sched 重写取代。HYLE 生产远小于 clay 的主因=**有意删/std取代/硬件取代**,非漏做。
> - **进度真相**:clay 地基(sys/utils/container 主体)**吸纳基本收口**;距完成(自有 ~30-45k)的差距=**净新增支柱**(simd 支柱基本完成,SSE4.2/AVX2 实跑、NEON/AVX512 编+objdump 验未实跑;io/compute/graph 未开工),那不是 clay 吸纳范畴。**clay 吸纳权重 ≈ 32%**——注意这是"占被重构旧 clay 库的比例",**不是项目对完整愿景的进度**(后者按自有 LOC + 能力加权两法交叉 = **≈22%**,含 io/compute/graph 三大未开工支柱 + simd 全广度;不含借入的 ggml。算式见 [`milestones/README.md` 整体进度节](../milestones/README.md))。
>
> **替换/延期核验结论(2026-06-29 逐项查 clay 源,回答"std/第三方替换的有没有暗漏 clay 逻辑、defer 的能否现在做")**:
> - **std/第三方替换 = 真终态,无暗漏**:`result→std::expected`(ASSIGN_OR_RAISE 宏保留)、`sprintf→std::format`(clay 的"UTF-8 安全截断"本就是**注释掉的假功能**,HYLE 真 UTF-8 已落 string_algo:utf8_decode/length/icmp)、`func_traits/comparable→concepts/<=>`、`consts→<numbers>`、`thread_pool/worker→sched work-stealing`(重写超越)。逐个查过:**没有"std 其实没覆盖却被当替换"的坑**。
> - **被当"待补"实为已做或故意**:`string_buf intern` **已实现**(等内容返同指针;"待补"只是全局单例便利层,次要);`CUSTOM_MEMORY_FUNCTIONS` **故意不做**(mimalloc 无 off-switch 是用户指令,非漏)。
> - **`async_value`+`barrier` 地基已吸纳(sched-05,2026-06-29)**:fiber-aware 同步原语(CV + Mutex/WaitGroup/Latch/Channel)落地——这是 async_value"等待值就绪"+ barrier 倒数的**真机制**。API diff 揪出 clay async_value **不是** promise/future+waiter 而是值状态机 cell(`.cc` 42 行纯类型注册,无阻塞逻辑);故剩余 = 类型化 future 值封装(CV 上薄层,io/compute 需要时建)+ 可重置 Barrier(薄层),均**净新非移植**。`decls.h` include 纪律仍是个小真空待坐实。**真·clay 吸纳欠账 ≈ 零。**

## sys/ — 系统基座（allocator/arena/os/path/string/hash/log/time）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `sys/allocator.h` + `.cc` | 485+375 | mem-01 + mem-01b + mem-02 | ✅ | vtable+裸 allocate + typed 层(mem-01b) + **backing 换 mimalloc(mem-02,封 vtable 后不外露,Win 默认堆 2.5–12x 随负载/Linux 真实模式 5.2x)**;CUSTOM_MEMORY_FUNCTIONS 故意不做(mimalloc 无 off-switch,用户指令非漏,见上) |
| `sys/memory_arena.h` + `.cc` | 65+258 | mem-03 + mem-03b | ✅ | `MemoryArena` 分片无锁定长块池(N 头各占 cache line,地址路由 free,修单 head_ 的 O(threads) 争用 278x→1.94x) + **`Arena` bump 区域分配器(去原子) + as_allocator() 适配(mem-03b)**;`make_arena_allocator` 等价为 `Arena::as_allocator()` |
| `sys/carray.h` + `.cc` | 436+65 | container-03 | ✅ | header-before-data 侵入数组 + type-erased 核 + RAII 壳 CArray<T>;**改进:over-align 元素(clay 固定 max_align_t 会错位)** |
| `sys/cstring.h` + `.cc` | 1033+374 | container-06 | ✅ | string_ref(ptr+len+null 位)+零拷贝算法(find/rfind/find_first/last_of/not_of/substr/trim/starts/ends_with/contains/count/split/rsplit/slice/concat/dup/c_str/equal_ignore_case)+ operator<=> + Hash<string_ref> + `_sr` 字面量。**类型核心 API diff 过 clay**;clay `capitalize`/UTF-8/hex/escape 真实现归 `string_algo.h`(container-06b✅);`score_similarity`/`format_si` 等 **clay 只声明无实现**→container-06c(净新) |
| `sys/string_buf.h` + `.cc` | 141+363 | container-06 | ✅ | 引用计数驻留池 `StringBuf`(pimpl 藏 SwissTable 反查+mutex)+ make/destroy 工厂;每唯一串单独分配(地址稳定)。**改进**:hash 命中 memcmp 校验(XXH3 碰撞 fail-fast 而非静默别名);复用 container-04 SwissTable→.cc 仅 83 行(非 stub,是复用红利)。**全局单例 intern/usprintf 📋待补**(见下) |
| `sys/hashmap.h` | 809 | container-04 + container-04b | ✅ | concept 模板重写 + **实现 clay 只 TODO 的 SwissTable**(控制字节+SIMD探测+三角探测+tombstone)；**type-erased 冷核(resize/free 进 DLL,防 per-T 膨胀)+ erase abseil reclaim-to-empty + NEON group(aarch64 编过未跑)已补(container-04b✅)** |
| `sys/murmur_hash.h` | 294 | container-02 | 🔶 | **运行期哈希→XXH3 已接(container-02✅,借 xxHash 替 murmur)**;编译期 type 哈希→**FNV-1a constexpr 已做(util-02 type_hash)**;murmur 本体未移植(无剩余用途) |
| `sys/error.h` + `.cc` | 222+143 | **error-02** | ✅ | error_handler vtable(跨 DLL 诊断 sink)。改进:非变参 vtable(ABI 安全,clay 变参函数指针隐患)、干净 fatal(HYLE_TRAP vs clay rand+空写+exit)、补 charter §5 `HYLE_ASSERT`、record handler 无分配。NOT_IMPLEMENTED 等便利宏 gap 登记 |
| `sys/os.h`+`os.linux.cc`+`os.shared.cc` | 795+830+210 | platform-02/03b | ✅ | **dynlib ✅(platform-02)**;**fs/env/os_info ✅(platform-03b)**:file_system_api 13 fn(stat/mkdir+recursive/remove+recursive/rename/copy/iterate/cwd/chdir/temp/home)+ touch + get_env_var,**buffer-fill+visitor 替 clay 分配式 strings_t/std::function,保 fs 在 P1**;改进 mkdir 幂等/temp 认环境变量/copy 可移植替 sendfile。`file_io`(open/read/write)+`mmap_*`→io-05;`socket_*`→io-04;`num_logical_processors`→♻️`std::thread::hardware_concurrency`;**`avx_support`→✅ simd-01 dispatch 取代并超越**(clay 单 `__builtin_cpu_supports("avx")`→HYLE `CpuFeatures` 全特性位 CPUID+XGETBV 探测);`absolute`/`ensure_parent`=组合登记。**virtual_memory clay 无**→platform-02b(net-new,未排期) |
| `sys/path.h` + `.cc` | 202+1372 | platform-03a (+platform-03b/io) | ✅ | **vtable `path_api_t`(16 函数)纯字符串半 ✅(platform-03a,落 container/path.{h,cc},232 行)**:extension(splitext)/strip/base/directory/parent_dir/pop_first/pop_last/split_all/join/join_array/normalize(native 折叠)/is_absolute(升级自 clay blend `is_relative`)/is_unc。修 clay directory 的 `//x` OOB。**vtable 外 ~1100 行**:path_contains/cmp_normalized=`stat`+`normalize` 组合(platform-03b 基元已备);make_safe_filename Win 域(随用补);path_rel blend⛔/sequence_decode 渲染域⛔;normalize_unc+宽字符↔UTF-8→宽字符随 io/Win 路径补 |
| `sys/log.h` + `.cc` | 150+178 | obs-01 | ✅ | **日志产出(obs-01)**:level 门控 + LogSink vtable 注册表 + 默认 stderr ANSI sink + log_fmt。改进:**非变参 sink**(clay printf 变参 fn-ptr ABI 坑,同 error-02)、caller-side `std::format` 替 va_list、atomic level 无锁门 + mutex 快照(clay 全局裸数组多线程 race)。instrumentation hook(计数器/trace)=obs 后续净新增步 |
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
| `utils/vector_utils.h`/`device_utils.h`/`float16.h` | 98/125/52 | ♻️ / compute | ♻️⛔ | **2026-06-29 逐文件判定终态**:`vector_utils`(has_resize SFINAE 探测器等)→**♻️ C++23 concepts 取代**;`device_utils`(identity 函子=`std::identity` C++20 + host/device)→**♻️ + compute 域**;`float16`(`using float16=__half` CUDA 别名 + is_half traits,依赖 cuda_fp16.h)→**compute 域**(注意:simd-05 已落可移植 fp16/bf16 存储+转换,clay 这个是 CUDA 类型别名,非同物)。三者均**非 P3 clay 吸纳欠账** |

## parallel/ — 并发（thread_pool/async_value/concurrent_vector）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `parallel/thread_pool.h` + `.cc` | 130+167 | sched/ | ♻️ | 单全局锁池 → **被 sched-03 work-stealing 调度器取代(重写非移植)** |
| `parallel/worker_thread.h` + `.cc` | 88+109 | sched/ | ♻️ | 同上,被 sched worker 取代 |
| `parallel/async_value.h` + `.cc` | 465+42 | sched-05(地基)/ 后续(值封装) | 🔶 | **API diff 重大发现(2026-06-29)**:clay async_value **不是** promise/future+waiter——它是 **refcounted 类型擦除值状态机**(kUnconstructed→kConstructed→kConcrete→kError + 16bit type-id 注册),`.cc` 42 行**纯类型注册表**,**无 waiter 链表/无阻塞/无 notify**(同 container-06b 揪出的 format_si "声明无实现"一类:头部 promise/future 是 TFRT 味期望,clay 这份只有值 cell)。真正的"等待值就绪"=**fiber-aware 阻塞**,sched-05 的 `ConditionVariable` 已落地这个**真地基**。剩"类型化 future 值封装"(在 CV 上包一层 ready 谓词)=薄层,待 io/compute 需要 promise/future 时建(净新,非 clay 移植) |
| `parallel/concurrent_vector.h` | 145 | container-09 | ✅ | 读优化 append-only 并发 vector(读 lock-free/写 SpinLock 串行)。**分段存储**修 clay 五问题:外层 reserve(65)→定长 C 数组(结构性不变量)、copy-on-grow 2x 内存→零拷贝 1x、T copyable→move-only 可、每读边界检查→去除、std::vector→Allocator。**WSL TSan 零 race**。归 container(不依赖 sched)|
| `parallel/barrier.h` | 46 | sched-05 | 🔶 | clay Barrier=可重置屏障(std::mutex+cv+count)。sched-05 落地 fiber-aware **`Latch`**(一次性倒数,std::latch 形)+ **`WaitGroup`**(动态计数,Go 形)覆盖屏障的倒数语义;clay 的**可重置**(`wait(reset=true)`/`Break()`)变体=在同一 CV 上加 reset 的薄封装,需要时补(用户 sched-05 scope 列的是 CV+mutex/waitgroup/channel/latch,Barrier 未列,故登记薄层) |

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

## 红队核账登记（2026-06-28，4 agent 对抗审计本批 path/swiss-04b/fs/log）

红队验证架构核心扎实（type-erase 热路径零回归成立、NEON 逻辑等价、erase reclaim 不变量正确、无 double-free/泄漏、零 std 流违规、无玩具空壳）。**当轮已修 7 处硬伤**：log.h 美化注释("trailing marker"→静默截断)、Win `current_directory`/`temp_directory` 超 MAX_PATH 读未初始化缓冲、POSIX `directory_iterate` DT_UNKNOWN 用 `lstat` 防递归删跟随符号链接(+ WSL 符号链接安全测试)、`copy_file` EINTR 重试 + O_CLOEXEC、log 双 OutputDebugString 合一、`path::join` uint32 溢出守卫、log sink 生命周期硬契约文档。

**显式登记待办（红队挖出、本轮未修的大件，非略过）**：
- ~~【高】platform-03c：Windows UTF-8 路径~~ **✅ 已做(platform-03c)**:fs.cc Win 分支全转 `...W` API + `WidePath`(UTF-8→UTF-16,栈+堆 RAII)/`fill_w`(UTF-16→UTF-8 buffer-fill),platform 内自包含无倒挂。**红队 UTF-8 测试还揪出 `touch()` 双重 ANSI 漏洞**(std::fopen + CreateFileA 都绕过转换)→ Win 改 CreateFileW(OPEN_ALWAYS)。双平台 UTF-8 路径(CJK+emoji)往返测试过。
- ~~【中】container-04c：SwissTable `find_or_prepare_insert` 融合~~ **✅ 已做(container-04c)**:insert 新键改单趟探测(walk 同时查 key 命中 + 记首个 empty/deleted 槽),哈希算一次(原 2× 哈希 + 2× 探测)。**附带修掉 growth_left 复用墓碑重复扣减 bug**(只在插 empty 槽时扣)。双平台 11/11 含 oracle 对拍(50k/200k)。算法级 op 削减,未跑 before/after microbench。
- **【中】obs log 无锁快照** — **尝试后撤销(诚实记录)**:试了 seqlock 让并发 logger 无锁 fan-out,但读端 memcpy 非原子 g_sinks 与写端并发 = C++ 内存模型 data race/UB,**WSL TSan 正确报 race**(log.cc add_sink)。标准洁净的无锁替代(atomic<shared_ptr> 快照、逐字段原子)开销≈mutex 的原子 RMW 且更复杂,而 mutex 临界区只是 tiny memcpy、被 sink I/O 淹没。**遵 CHECKLIST"lock-free 必过 TSan"撤回 seqlock,保留 mutex**。净收获:新增并发压力测试(6 logger 并行 + churn add/remove)且**过 TSan**,验证 mutex 注册表 race-free。
- ~~【中】platform fs copy_file 内核零拷贝~~ **✅ 已做**:Linux 走 `copy_file_range`(内核内拷贝,btrfs/xfs 可 reflink),仅首调失败(EXDEV/ENOSYS/EINVAL,未写字节)才回退用户态 read/write。`_GNU_SOURCE` 暴露声明。macOS/其他 POSIX 留用户态(无 fcopyfile 测试机)。300 KiB 字节图样往返测试过。
- ~~【中】platform-03c：Win 长路径 + UNC~~ **🔶 部分已做**:`directory_iterate` 的 MAX_PATH 截断已在 03c 消除(堆分配宽 pattern)。**`\\?\` 扩展前缀已做+测**:WidePath 对 ≥248 的绝对路径自动加 `\\?\`(drive `C:\`→`\\?\C:\`,slash 归一为 backslash),320+ 字符深树 make/touch/iterate/recursive-remove 双平台测过(Win 无前缀会 MAX_PATH 失败)。**UNC `\\?\UNC\` 代码带上但未测**(无网络共享,写不验,登记)。`make_directory_recursive` 的 UNC 前缀跳过仍待(需共享测)。
- **【低】Win-ARM64 NEON MSVC 编译门**:GroupNeon 仅 aarch64 GCC 交叉编译验证,`_M_ARM64`(MSVC)路径未编译验证;若 Win-ARM64 入目标需补门或排除。
- **【低】swiss erase 复用墓碑 growth_left 重复扣减**(自愈、纯空间)、log default sink 非 tty/重定向时 ANSI 污染、`stat` exists 语义 Win(任意属性)vs POSIX(文件||目录)分叉。

## HYLE vs clay 的计划性差异（关键：吸纳 ≠ 追平 LOC）


> **"完全吸纳 clay" 不等于 "HYLE LOC 追平 clay"。** 标尺只对"该吸纳的那部分"成立——对那些组件，HYLE 明显小于 clay = 缩水嫌疑(要查)。但整体 HYLE 会**超过** clay，因为三类计划性差异：

### ① 砍掉 / 替换（HYLE 比 clay 少的部分——合理，不是缩水）
- `utils/result.h`(470)→ `std::expected`；`comparable.h`(53)→ `operator<=>`；`C++17.h`/`meta_funcs`/`func_traits` 大半 → C++23 concepts；`singleton.h` → Meyers；`type_traits` 大半 → concepts。**clay 是 C++17 手搓 SFINAE，HYLE 是 C++23，这些"消失"是现代化红利,不是漏做。**
- `parallel/thread_pool`+`worker_thread`(约 494 行)→ **废弃**，被 sched-03 work-stealing 调度器**重写取代**(单全局锁长不出统一调度器)。
- `math/half.h`(6923 软件查表)→ **不 1:1 移植**，硬件 F16C/AVX512-FP16 取代,只留接口。
- catomics(clay sys/details,未在上表)→ **不 port**，C++23 std::atomic 取代。
> 合计:clay 里约 **8,000+ 行**属于"HYLE 计划性不吸纳/大幅替换"——所以拿 19k 当目标分母是错的，**真实"应吸纳"分母约 10-11k**。

#### ①′ std 取代的性能裁决（2026-06-28，回答"用 std 真比 clay 快吗"）

> **核心问题**：HYLE 核心目标是性能 SOTA。用 std/C++23 取代 clay 手写,**会不会反而更慢、把核心目标搞砸**？逐条裁决(分 4 类),结论:**没有一个 std 取代落在"会输给 clay 的热路径"上**。

| std 取代 | 类别 | 裁决依据 |
|---|---|---|
| bit ops → `<bit>`(popcount/countl_zero/bit_ceil…) | **intrinsic 背书** | **实测 objdump**(`-O3 -march=native`):`std::popcount`→`popcnt`、`countl_zero`→`lzcnt`、`countr_zero`→`tzcnt` **单条硬件指令**,与手写 intrinsic 同。温路径(hashmap probe/对齐)安全。 |
| clamp/min/max → `<algorithm>` | intrinsic/同码 | 平凡内联,与手写同 codegen。 |
| pow/abs/is_normal → `<cmath>` | 同 libm | 同 libm/编译器 intrinsic;C++23 起 constexpr。 |
| `math/half.h` → 硬件 F16C/AVX512-FP16 | **已知大赢** | 硬件 fp16 指令 vs clay 6923 行**软件查表**——数量级更快,正是不移植的理由。 |
| consts → `<numbers>`/`numeric_limits` | **零运行时** | 编译期常量,clay 头自注"will be replaced"。 |
| type_traits/concepts/func_traits/singleton | **零运行时** | 纯编译期/init-once,无热路径开销。 |
| `result.h` → `std::expected` | **零开销设计** | value/error union 无分配;charter 错误模型。比手写 Result 不慢。 |
| stoi/stod → `std::from_chars` | **已知赢** | from_chars 无 locale/无分配,普遍快于 stoi/strtod。 |
| `thread_pool`+`worker` → sched work-stealing | **已实测赢 SOTA** | sched bench:spawn 比 **marl(真 SOTA 调度器)快 2.17x**(141 vs 306ns;非 std::async 弱基线);work-stealing 重写取代 clay 单全局锁池。详见 performance.md。 |
| `num_logical_processors` → `hardware_concurrency` | **冷(启动一次)** | 启动期调一次,零热路径。 |
| sprintf → `std::format`/`snprintf`;to_string → format | **冷路径** | 唯一理论风险(某些 libstdc++ 版 format 慢于 printf),但格式化是**冷路径**(日志/错误/调试),clay sprintf 同样冷用途——即便略慢也不碰 hashmap/alloc/fiber 热循环。**已留实测口子**:若 profile 发现 format 上热路径,vendor stb_sprintf(覆盖图 sys/sprintf 行已记)。 |

**裁决结论**:① intrinsic 背书的(bit/clamp/pow)= **同指令,零损失**(已 objdump 证);② 编译期/零运行时的(consts/concepts/expected)= **无热路径开销**;③ 已实测赢的(work-stealing/from_chars/F16C)= **更快**;④ 唯一冷路径风险(std::format)= **不在热路径,且留了实测翻案口**。**故"用 std 取代"未违背 SOTA 目标**;真正决定 SOTA 的是热路径自造件(SwissTable/分配器/fiber/SIMD),那些**不**用 std,要对标 absl/mimalloc/Highway 实测(见 performance.md gate)。


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
- device 内存:clay 无 → HYLE **compute/ 自有 stream-bound buffer**(ggml buffer_type 式,所有权+stream 绑定在 compute/;不走 mem/ allocator 抽象,见 mem-design「device 内存」节)。

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
| `parallel/async_value`+`barrier`+`concurrent_vector` | P4-P5 | **sched-05 地基✅** |
| `math/utils`+`consts` | 后期 math 簇 | 排期后 |
| `math/half` | P3 simd-05(硬件取代🔻) | P3(留接口) |

**结论(直接回答):**
- clay 的 **sys + utils 主体**(分配器家族 + 全部容器 + string + hashmap + registry + ref-counted + 类型设施)→ **到 `P3 末` 完全吸纳**。这是 clay 地基核心的大头、最该对照行数核账处。
- clay **parallel/** → **P4 被 work-stealing 调度器重写取代**(thread_pool,已 sched-03✅)+ **P5 初收掉 async_value**(→ completion 脊)。
- clay **math/** → half 在 **P3 由硬件 fp16 取代**(只留接口);utils+consts 排后期 math 簇。
- **一句话:盯 `P3 末`(container-02..06 + mem 全部 + 01b/03b)——跑完且每文件 diff 过 clay,clay 主体即吸纳完毕;`P5 初` 补完 async_value 为完整收口。** P6(graph/C-ABI/FFI)是 HYLE 净新增,与 clay 吸纳无关。
- **当前位置(2026-06-29 重核,纠旧"刚起步"陈述):clay 地基吸纳基本收口。** sys/utils/container 主体已 ✅(allocator 家族 + arena/pool/stack + 全部容器 + string/hashmap/registry/ref-counted/类型设施 + path/fs/log/time/random),container-02..06 全簇 + mem-01..04/07 + platform-02/03 + time-01 + util-01..06 均完成。剩余 clay 行(4 🔶 + 5 📋)经逐文件判定**基本是终态(std/硬件取代)或归别处(compute/P5 sched)**,真·clay 吸纳欠账 ≈ 零。**未到完全 ✅ 的尾巴**:`decls.h` include 纪律坐实、NUMA(净新增非 clay)、async_value/barrier(P5 sched)。**结论:clay 吸纳从"刚起步"修正为"主体收口、尾巴是净新增/别模块",剩下的工程量是 io/compute/graph 三大净新增支柱,与 clay 无关。**
