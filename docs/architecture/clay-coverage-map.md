# clay → HYLE 覆盖映射表（吸纳完整性的持续标尺）

> **目的**：防"以为吸纳完了、其实漏文件/漏 API 层"。clay 源在 `slate-render/iron/infra/{include,src}/clay/`。
> **铁律（对抗审查纪律）**：每个吸纳 clay 的里程碑闭环前，必须 `diff` clay 同名文件的 API 表面——缺的要么补、要么在本表显式登记 `📋待办`，**不许默默略过**。本表是唯一权威进度标尺。
> **状态图例**：✅ 完成 / 🔶 部分(有登记的缺口) / 📋 计划 / ♻️ 被 C++23/std 取代 / 🔻 借第三方(不自写) / ⛔ 不吸纳。
> **行数**：clay 列为该文件总行数(`grep -c`)；HYLE 列为对应物生产行数。clay 自有(去 thirdparty/plugins)≈ **19,000**；HYLE 当前生产 ≈ **1,784**。差距大部分是"未做"(进度)，小部分是已修正的"缩水"(见 🔶)。

## sys/ — 系统基座（allocator/arena/os/path/string/hash/log/time）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `sys/allocator.h` + `.cc` | 485+375 | mem-01 + mem-01b | ✅ | vtable+裸 allocate + **typed 层 allocate_n<T>/callocate/construct/destroy/release/make_unique 已补(mem-01b)**;CUSTOM_MEMORY_FUNCTIONS/allocator-printf 仍 📋(次要) |
| `sys/memory_arena.h` + `.cc` | 65+258 | mem-03 + mem-03b | ✅ | `MemoryArena` 块池(packed CAS) + **`Arena` bump 区域分配器 + as_allocator() 适配已补(mem-03b)**;`make_arena_allocator` 等价为 `Arena::as_allocator()` |
| `sys/carray.h` + `.cc` | 436+65 | container-03 | ✅ | header-before-data 侵入数组 + type-erased 核 + RAII 壳 CArray<T>;**改进:over-align 元素(clay 固定 max_align_t 会错位)** |
| `sys/cstring.h` + `.cc` | 1033+374 | container-06 | 📋 | string_ref(ptr+len+null 位)+零拷贝算法 |
| `sys/string_buf.h` + `.cc` | 141+363 | container-06 | 📋 | 引用计数驻留池 |
| `sys/hashmap.h` | 809 | container-04 | 📋 | flat open-addressing → 补 SwissTable SIMD 探测【大】 |
| `sys/murmur_hash.h` | 294 | container-02 | 🔶 | **运行期哈希→XXH3 已接(container-02✅,借 xxHash 替 murmur)**;murmur 仅留编译期 type_name 哈希(→ container-06/type_name 时做) |
| `sys/error.h` + `.cc` | 222+143 | **error-02** | 📋 | error_handler vtable(跨 DLL 错误上报)——error-01 只做进程内 Status，**这层未做 → error-02** |
| `sys/os.h`+`os.linux.cc`+`os.shared.cc` | 795+830+210 | platform-02/03 | 📋 | 虚拟内存/dynlib/env/os_info |
| `sys/path.h` + `.cc` | 202+1372 | platform-03 | 📋 | 路径操作(1372 行大头) |
| `sys/log.h` + `.cc` | 150+178 | obs-01 / platform | 📋 | 日志 |
| `sys/random.h` + `.cc` | 112+143 | (util 待排) | 📋 | RNG |
| `sys/time.h` + `.cc` | 47+112 | time-01 | 📋 | 单调钟 |
| `sys/sprintf.h` + `.cc` | 96+69 | (util 待排) | 📋 | 安全 sprintf |
| `sys/scope_guard.h` | 74 | (util 待排) | 📋 | ScopeGuard/DEFER |
| `sys/alignment.h` | 37 | macros/mem | 🔶 | 对齐工具——部分进了 mem-01 的 round_up；独立头未建 |
| `sys/demangle.h` + `.cc` | 28+45 | container/util type_name | 📋 | 符号 demangle |
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
| `utils/memory_pool.h` | 170 | (mem 待排) | 📋 | 另一种池变体 |
| `utils/ref_counted.h` | 317 | container/util | 📋 | 侵入式 RefCounted + Ref<T> |
| `utils/registry.h` | 340 | (util 待排) | 📋 | 自注册 + DECLARE/DEFINE |
| `utils/type_traits.h` | 240 | (编译期) | 🔶♻️ | 多数被 C++23 concepts 取代；保留底层兼容部分待评 |
| `utils/type_name.h` | 146 | (util 待排) | 📋 | 编译期 type_name + 哈希 |
| `utils/func_traits.h`/`meta_funcs.h`/`C++17.h`/`comparable.h` | 105/80/253/53 | (编译期) | ♻️ | 多数被 C++23(concepts/`<=>`)取代,按需保留 |
| `utils/string_utils.h` + `.cc` | 673+449 | container-06 | 📋 | 字符串算法 |
| `utils/generic_allocator.h`/`stl_allocator.h` | 62/196 | mem pmr adapter | 📋 | std::pmr 互操作 |
| `utils/singleton.h` | 133 | (util 待排) | 🔶♻️ | Meyers 单例(C++11 起 std 保证),薄封装待评 |
| `utils/spin_lock.h` | 43 | (并发原语) | 📋♻️ | 可能被 std::atomic 自旋取代 |
| `utils/vector_utils.h`/`device_utils.h`/`float16.h` | 98/125/52 | 各相关 | 📋 | 杂项工具 |

## parallel/ — 并发（thread_pool/async_value/concurrent_vector）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `parallel/thread_pool.h` + `.cc` | 130+167 | sched/ | ♻️ | 单全局锁池 → **被 sched-03 work-stealing 调度器取代(重写非移植)** |
| `parallel/worker_thread.h` + `.cc` | 88+109 | sched/ | ♻️ | 同上,被 sched worker 取代 |
| `parallel/async_value.h` + `.cc` | 465+42 | sched-04/05 | 📋 | promise/future + waiter → completion 脊/同步原语 |
| `parallel/concurrent_vector.h` | 145 | container | 📋 | 读优化并发 vector(lock-free 化) |
| `parallel/barrier.h` | 46 | sched 原语 | 📋 | 屏障 |

## math/ — 数学（half/utils/consts）

| clay 文件 | 行 | HYLE 目标 | 状态 | 备注 |
|---|---|---|---|---|
| `math/half.h` | 6923 | simd-05 | 🔻♻️ | fp16 软件查表(巨)→ **硬件 F16C/AVX512-FP16 取代,不 1:1 移植**;只留必要接口 |
| `math/utils.h` | 353 | (math 待排) | 📋 | 数学工具 |
| `math/consts.h` | 105 | (math 待排) | 📋 | 常量 |

## 已修正的"完成"状态（红队核账发现的缩水）

- **mem-03**：从"✅完成"改"🔶块池完成、Arena bump 待补(mem-03b)"。
- **mem-01**：标注 typed 便利层待补(mem-01b)。
- **error-01**：标注 error_handler vtable 待补(error-02)。

## 新登记的补漏里程碑

- **mem-01b**：allocator typed 便利层(`allocate<T>/construct<T>/make_unique<T>/destroy<T>` + pmr 钩子)。
- **mem-03b**：`Arena` bump 分配器 + `ArenaAllocator` 适配 + `make_arena_allocator`。
- **error-02**：error_handler vtable(跨 DLL 错误上报)。

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
- **吸纳完成** = 本表 sys/utils/parallel/(math 按需) 中**所有非 ♻️/⛔ 行变 ✅**，且每行经过 clay 同名文件 API diff。当前:`status/object_pool/stack_allocator ✅`、`allocator/memory_arena 🔶(已登记补漏)`,其余 📋。

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
