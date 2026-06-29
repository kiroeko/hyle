# clay 资产负债表

> 本文档是 HYLE 重构的依据：把 clay infra 层逐子系统盘点为「直接继承 / 改后继承 / 推倒重写 / 从零造」，每条结论附代码证据（`file_path:line`）。
>
> 盘点对象：`C:\public\project-hyle\slate-render\slate\infra\`（与 `iron\infra\` 为镜像拷贝，命名空间 `clay::`）。
> clay 自有代码 ≈ 28,667 行（不含 thirdparty 67,544 行）。

## 体量分布（自有代码，不含 thirdparty）

| 子系统 | 行数 | 性质 |
|---|---|---|
| `sys/`（allocator/arena/catomics/log/os/time/hashmap/murmur/cstring）| 11,953 | 地基核心 |
| `plugins/taskgraph/`（并行 DAG 执行器）| 4,890 | 计算图编排 |
| `utils/`（result/status/object_pool/registry/type_traits）| 4,711 | 工具层 |
| `math/`（half/float16/SIMD/consts）| 4,363 | CPU 数学原语 |
| 顶层 headers（host_device/simd/platform/macros）| 1,520 | 门面/宏 |
| `parallel/`（thread_pool/worker_thread/async_value）| 1,220 | 并发原语 |
| **合计** | **28,667** | |

---

## 一、内存分配器家族 — 真实现，地基扎实，5 处缺陷

核心设计：**单函数指针 vtable**。`Allocator` 只有一个 `realloc_func(inst, ptr, old, new)` 原语，allocate/free/realloc 全归一到它（`sys/allocator.h:24-79`），为跨 DLL / C-ABI 故意做的收敛，上层叠 placement-new 模板糖（`make_unique`/`construct`/`SYS_*`）。

| 模块 | 文件 | 处置 | 说明 |
|---|---|---|---|
| Allocator 接口 + SYS_* 家族 | `sys/allocator.h` | ✅ 直接继承 | C-ABI 友好，接口层留用 |
| SystemAllocator / AlignedAllocator | `sys/allocator.cc` | 🔧 改后继承 | 含计数 bug |
| MemoryArena（无锁定长块池）| `sys/memory_arena.cc` | ✅ 直接继承 | 最亮资产，tagged-pointer CAS 防 ABA |
| Arena（bump 分配器）| `sys/memory_arena.cc:76` | ✅ 直接继承 | 无锁 bump |
| MemoryPool（STL 兼容）| `utils/memory_pool.h` | ✅ 继承 | 经典第三方实现 |
| generic_allocator / alignment | `utils/generic_allocator.h`、`sys/alignment.h` | ✅ 继承 | 薄封装，正确 |
| ObjectPool | `utils/object_pool.h` | 🔧 推倒重写 | alloc/free 不匹配 |
| StackAllocator | `utils/stack_allocator.h` | 🔧 推倒重写 | 拷贝构造丢 allocator |

### 已确认的 5 处缺陷（重写依据）

1. **`Allocator::allocate(size, alignment)` 静默丢弃 alignment**（`sys/allocator.h:37-40`）：直接 `realloc_func(inst, nullptr, 0, size)`，对齐参数没传下去。对齐只在底层 realloc 恰好对齐时才成立。AVX512 场景（`CLAY_ALIGNMENT=64`）是隐患。
2. **统计计数 32/64 位下溢**（`sys/allocator.cc:277`）：`uint64_t dif = (new_size?1u:0u) - (old_size?1u:0u)`。free 时 `0u-1u` 在 32 位 unsigned 里是 `0xFFFFFFFF`，提升到 uint64 后 `fetch_add` 加的是 `0xFFFFFFFF` 而非 -1，`system_allocation_count` 越加越大。
3. **ObjectPool<T,false> 分配/释放不匹配**（`utils/object_pool.h:66` vs `:79`）：buffer 用 `allocator_->allocate<slot_t>()` 拿，却用全局 `delete pool` 还，绕过自己的 Allocator，UB。
4. **StackAllocator 拷贝构造丢 allocator_**（`utils/stack_allocator.h:28,30`，字段 `:108`）：拷贝只初始化 `pointer_(0)`，`allocator_` 无默认初值，拷贝后是不定值；溢出走堆回退路径时解引用野指针。STL 容器拷贝 allocator 是常规操作，必中。
5. **arena allocator 静态 64 槽上限**（`sys/memory_arena.cc:211-218`）：`make_arena_allocator` 从两个固定 64 槽 static 池取对象，最多 64 个并发 arena allocator，超了返回 null。

> 附注：`std::atomic<Head>`（16 字节，`memory_arena.cc:64`、`object_pool.h:155`）依赖 cmpxchg16b，需 `-mcx16`，否则退化加锁——「无锁」前提要在 HYLE 构建配置里坐实。

---

## 二、DataModel — 确认空位，README 纯吹

全仓 glob `*data*model*`、符号 grep `[Dd]ata_?[Mm]odel` **均零命中**。README 吹的 undo / 序列化 / 协作在 clay 里**连头文件都没有**。HYLE 若要 DataModel 是**从零造**，无任何可继承资产。

> 注：`taskgraph` 里有 `graph_io`/`schema` 的序列化雏形，但那是计算图 IO，不是文档 DataModel。

---

## 三、SIMD / 数学原语 — 空门面，基本从零造

- **`simd.h` 全文仅 43 行**，只做指令集探测：`__SSE2__`→`emmintrin.h`、`__SSE4_2__`→`nmmintrin.h`、ARM 走 `sse2neon`（`simd.h:10-42`）。只定义 `CLAY_HAVE_SSE2/SSE4` 两个宏，**无 AVX/AVX2/AVX512 探测、无 `_mm_*` 封装、无 vector 类型、无运行时 dispatch**。
- **`half.h`** 是 Christian Rau IEEE-754 半精度库 v1.11.0（`half.h:1-23`），**纯标量软件模拟**（查表 `base_table`/`shift_table` 等，`half.h:503-561`），无 F16C/SSE。仅 `__CUDA_ARCH__` 时走硬件 `__half2float`（`half.h:1362`）。
- **`float16.h`** 只是 alias（device→`__half`，host→`half`，`float16.h:16-21`），无 bf16。
- **`math/utils.h`、`consts.h`** 是扎实但普通的标量工具（clamp/next_pow2/div_ceil/log2，含 `CLAY_HOST_DEVICE` 注解，`_BitScanReverse`/`__builtin_clz` 封装到位 `utils.h:279-322`）。
- **`math/details.cc` 是空文件**（0 行有效代码）。
- **`host_device.h`** 是质量较高的 CUDA launch-bounds / `__host__ __device__` 宏门面（PyTorch 风格），仅约定无计算。

**对标**：远低于 Highway/xsimd（后者有可移植 vector 抽象 + 运行时 dispatch + AVX512/SVE/bf16）。

| 文件 | 处置 | 理由 |
|---|---|---|
| `simd.h` | 🔧 推倒重写 | 仅 SSE2/4 探测，需换 Highway 式可移植层 |
| `math/half.h` | 边界外/改后继承 | 第三方软件库，保留作 fallback，硬件路径另建 |
| `utils/float16.h` | 🔧 改后继承 | alias 可用，需补 bf16/fp8 |
| `math/utils.h`、`consts.h` | ✅ 直接继承 | 标量工具扎实 |
| `math/details.cc` | 边界外 | 空文件 |
| `host_device.h` | ✅ 直接继承 | 高质量 CUDA 宏门面 |

---

## 四、容器库 — 真地基，成色好

| 文件 | 行数 | 性质 | 处置 |
|---|---|---|---|
| `sys/hashmap.h` | 715 | C 宏泛型 flat open-addressing：power-of-two 位与取模（`:517-521`）、线性探测+tombstone、keys/values 单次连续分配（`:640-647`）、70% 负载扩容（`:606`）。注释规划了 SwissTable control-word + AVX 16 路并行探测但**明确未实现**（`:109-131`）。 | 🔧 改后继承（核心资产，补 SIMD 探测即顶级）|
| `sys/carray.h` | 423 | header-before-data 侵入式动态数组（`:14-30`），几何增长、POD memcpy 语义，裸指针可当数组，cache 友好。 | 🔧 改后继承（补类型安全/RAII）|
| `sys/cstring.h` | 1003 | `string_ref`（ptr+len+null 标志）+ 零拷贝算法、interning、相似度评分。 | ✅ 直接继承 |
| `sys/string_buf.h` | — | 引用计数 + murmur hash 字符串驻留池，纯虚接口 + make/destroy 跨 DLL 边界。 | ✅ 直接继承 |
| `parallel/concurrent_vector.h` | — | 读优化并发容器，写加锁、读单次 atomic load，状态打包进 `atomic<uint64_t>`（`:122-135`），扩容不释放旧 buffer（`:11-16`）。 | 🔧 改后继承（写仍全局锁，需 lock-free 化）|
| `utils/vector_utils.h` | — | std::vector traits/convert 辅助，薄。 | ✅ 直接继承 |

**对标**：hashmap/carray 接近 absl/folly 设计意图（flat、连续、避免 std 节点分配），但 hashmap 缺 SwissTable SIMD 探测，carray 缺类型安全/RAII，concurrent_vector 不如 folly/TBB 真 lock-free 版。

**缺口**：SmallVector/inline storage、intrusive list/map、lock-free MPMC 队列、hashmap SIMD 探测、非 64-bit key 的通用 hash。

---

## 五、并发子系统 — 只完成「线程」一格 + 一块优质原子层

| 模块 | 处置 | 说明 |
|---|---|---|
| `sys/details/catomics_*.h`（跨平台原子）| ✅ 直接继承 | **最硬的地基**。MSVC（Interlocked 全家桶）+ unix（`__sync_`/`__atomic_` 含锁回退）双实现，覆盖 8/16/32/64 位、CAS、pointer exchange、size_t 泛化。足以支撑无锁队列/调度器。|
| `utils/spin_lock.h` | ✅ 直接继承 | test_and_set 自旋锁 |
| `parallel/async_value.h/.cc` | 🔧 改后继承 | **TFRT 式 async value**（非 future/promise）：refcount + 4 态机（`async_value.h:89-104`）+ 类型擦除 payload。但状态机**非线程安全**（自承 `:213`），且**无 waiter 链表/回调**（无 `and_then`/`enqueue_waiter`）。能当 await 值容器，但还不是 await 原语，须补 waiter 队列才能驱动 fiber。|
| `parallel/thread_pool.{h,cc}` | ✅ 留作「长任务池」；统一调度器另起炉灶 | 单队列 + 全局锁经典模型：`std::priority_queue`（`thread_pool.cc:130`）+ 单 `std::mutex`（`thread_pool.h:122-124`）+ affinity（`thread_pool.cc:114`）。**无 work-stealing、无 per-thread 队列、无 lock-free**。`enqueue`/出队全程持全局锁。与「每核 run-queue + work-stealing + fiber 上下文切换」根本不兼容，强行改造等于重写背包袱。|
| `parallel/worker_thread.{h,cc}` | ✅ 留作长任务池 | thread_pool 的单线程版 |
| `parallel/barrier.h`、`decls.h` | ✅ 直接继承 | 小工具 |

### fiber / 协程 / 进程 / 网络 IO 四块现状（grep 证据）

- **fiber / 协程**：❌ **零实现**。全树仅 `thread_pool.h:18` 一句注释（"CLay is moving to fiber-based task system"）；无 ucontext/makecontext/setjmp（`os.linux.cc:59` 只有 `HAVE_SETJMP_H` 探测宏）、无 co_await/coroutine。
- **进程**：❌ **零实现**。无 fork/CreateProcess/posix_spawn；唯一命中是 `os.linux.cc:568 pthread_create`（线程，非进程）。
- **网络 IO**：⚠️ **仅声明无实现**。`os.h:407-513` 有 socket 句柄/地址/错误枚举，`decls.h:47 struct socket_api_t` 仅前向声明，但 `os_api_t`（`os.h:788`）只挂 file_io/file_system/dynlib/thread/time/info——**socket_api_t 从未定义、未实现、未接入**。（taskgraph 里的 "socket" 是 DAG 节点端口，与网络无关。）
- **epoll / io_uring / kqueue**：❌ 零命中。

**结论**：相对「线程/进程/纤程/协程统一调度 + 网络IO」的野心，clay 提供约 **20-25% 地基**——能复用原子层和长任务池，统一调度器核心（fiber 上下文切换、per-core 无锁调度、IO proactor、进程管理）全部从零造。但 catomics 质量足以支撑从头长出来。

---

## 六、taskgraph — 意外通用的并行 DAG 编排器（资产）

**架构**：DAG = node/socket/edge + schema。`GraphNode`（`graph_node.h:22`）持 `ancestors_/descendants_` 邻接表；连接用字符串 URL socket（`socket.h:16`，`<node>/<port>`）；`task_schema_t`（`schema.h:18`）从 kernel 函数签名自动推导类型（`schema.h:449`）；图从 JSON 加载（`graph_io.h:99`），支持子图嵌套。

**调度模型**：**数据流驱动的拓扑序**。`FrameBuffer` 为每节点维护原子入度（`framebuffer.cc:72`），执行器从 source 出发（`graph_executor.cc:41`），节点完成后 `decrease_counter` 降后继入度，归零即就绪（`graph_executor.cc:107`），就绪后继 `async` 投递到 `TaskQueue`（`graph_executor.cc:66`）。

**实现深度**：**能跑的真执行器**，非骨架。完整闭环（JSON→build→serial_init→并行执行→sink 收尾 `framebuffer.cc:145`）、promise/future 等待、arena 池化（`framebuffer.cc:22`）、abort 传播、动态库插件加载（`task_graph.cc:145`）、4 种 visitor。

**特化度（关键结论）**：**几乎无领域特化，是通用 DAG**。`framebuffer.h` 名字误导——不是图形帧缓冲，而是"一次执行的数据流值容器 + allocator"（`framebuffer.cc:56`）。`data_type.h:14` 是纯通用标量/向量/string，无 tensor/image/纹理；唯一外向耦合是 `DATATYPE_PYTHON_OBJECT`（void*）。无 .cu、无 GPU、无图形 API 依赖。

**处置**：🔧 **改后继承**。
- 保留：DAG 建模、数据流拓扑执行器、kernel 反射注册、JSON IO、arena 化 framebuffer。
- 剥离/改造：(a) `TaskQueue` 后端抽象化，对接 HYLE fiber 调度而非裸 `std::thread`；(b) `DATATYPE_PYTHON_OBJECT`、动态库 `load_plugin` 等应用层耦合移出核心；(c) `data_type` 扩 tensor/device-buffer 承接异构；(d) 改 `FrameBuffer` 命名避免与图形帧混淆。

---

## 总账

**继承面（约占 HYLE 1/3，16-18k 行可留用）**：catomics 原子层、flat hashmap + carray + cstring + string_buf 容器、内存分配器家族 + MemoryArena、math/utils + consts + host_device 宏、thread_pool（长任务池）、async_value（补 waiter）、taskgraph（剥离特化）。

**负债面 / 从零造（HYLE 真正"加的东西"，30-50k 行级）**：

1. fiber 上下文切换
2. 统一调度器（线程/进程/纤程/协程一张表，per-core 无锁 run-queue + work-stealing）
3. 协程集成（co_await）
4. 进程管理（fork/spawn）
5. 网络 IO（io_uring/IOCP/epoll proactor）
6. SIMD 抽象（可移植 vector + dispatch + AVX512 + bf16）
7. 自有异构计算原语（device/stream/memory 抽象）
8. 新容器（SmallVector/intrusive/lock-free 队列）

**HYLE 总体量**：HYLE 自有 **~30-45k 生产 / ~48-72k 含测试**；含第三方 vendored 总足迹 **~200-300k+**（ggml 主导，详见 hyle-scope.md 规模节）。这不是"重构 clay"，是"以 clay 为地基的全新 infra"。
