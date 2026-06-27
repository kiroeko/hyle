# HYLE 实现路线图

> 动手前的收口：把 12 份设计里散落的硬依赖收成一张无环图，排出施工序与并行批次，给每批定验收门槛，并汇总全部决策（已决 + 待决）。这既是施工计划，也是对全部设计的最后一次端到端校验。

## 一、模块依赖图（验证无环）

箭头 = "依赖/被阻塞于"。只画硬依赖（编译期或接口契约级）。

```
error/  ────────────────────────────────────┐ (status + std::expected，零上游)
  │                                          │
  ▼                                          │
platform/ (thread/vm/dynlib/path/fs/park)    │
  │                                          │
  ├──────────────┐                           │
  ▼              ▼                            │
mem/          time/                          │  obs/ hook 接口
(mimalloc      (单调钟+                       │  (最小埋点 API，
 backing,       时间轮)                       │   批0 先定、空实现，
 修 alignment)   │                            │   各模块热路径嵌)
  │              │                            │
  ├────────┐     │                            │
  ▼        ▼     │                            │
container/ simd/ │   (container 内联自包含 SIMD 探测片段，不依赖整个 simd/)
(ring/MPMC/(可移植│
 intrusive/ vector+│
 SwissTable)dispatch)│
  │        │     │
  └────┬───┴─────┘
       ▼
   ┌─────────────────────────┐
   │  sched/  【关键路径瓶颈】 │  ← 依赖 container(队列原语)+time(deadline)
   │  fcontext+work-stealing  │     +platform(park)+mem(fiber栈)
   │  +Waker completion 脊    │
   │  +fiber-aware sync       │
   └───────────┬─────────────┘
       ┌───────┼───────┐
       ▼       ▼       ▼
     io/    compute/  (graph/)
   (proactor (device/   跑在 sched 上
    +liburing) stream/
       │       kernel
       │       +ggml)←── simd/ 喂 kernel
       │       │
       └───┬───┘
           ▼
   C-ABI shim（各模块稳定后逐个封）+ ffi 绑定（build-time）
```

**无环校验**：error → platform → {mem,time} → {container,simd} → sched → {io,compute} → graph。单向，无回边。`simd/` 只依赖 `mem/`（对齐契约），**不在通往 sched 的关键路径上**，可与 container 并行、晚点喂 compute。

## 二、施工批次（批内并行，批间串行）

| 批 | 模块 | 依赖 | 为什么这批 |
|---|---|---|---|
| **批0 基础** | `error/` + `obs/` hook 接口（空实现）| 无 | 万物用 error；埋点 API 先定，后续热路径直接嵌、免回头改 |
| **批1** | `platform/` | error | OS 抽象，mem/time/sched 都要 |
| **批2** | `mem/`（**先修 alignment**）+ `time/` | platform | 并行；mem 的 alignment 修复阻塞 simd，最优先 |
| **批3** | `container/`（ring/MPMC/intrusive 优先）+ `simd/` | mem | 并行；container 的队列原语是 sched 的前置 |
| **批4** | `sched/` 核心（fcontext+run-queue+work-stealing+Waker+fiber-sync）| container,time,platform,mem | **关键路径瓶颈**，四方汇聚 |
| **批5** | `io/` + `compute/` | sched | 并行，都挂 Waker/completion；simd 在此喂 compute kernel |
| **批6** | `graph/` + `obs/` 重机器 + T3 便利 + C-ABI shim 收口 | sched/io/compute | 上层便利 + 对外封口 |

**关键路径**（最长链，决定最早可用时间）：`error → platform → mem → container → sched`。**sched 是瓶颈**——它要等 container 的队列原语 + time + platform 的 park 全部就位。所以**批2/批3 的 container/time 要尽早动**，它们一卡，sched 就卡。

## 三、每批验收门槛（立住才进下一批）

| 批 | 门槛 |
|---|---|
| 批0 | error 跨三编译器编译；`std::expected` 往返；埋点 API 编译期零开销验证 |
| 批1 | platform 三平台（Linux/Win/mac）编译；dynlib 加载冒烟；park 原语单测 |
| 批2 | **mem：alignment 保证测试（64B 对齐必须真生效）**、并发分配 TSan、对标裸 mimalloc 不退化；time：单调性、时间轮到期精度 |
| 批3 | container：hashmap 对拍 std + 基准对标 absl/F14、lock-free 队列 TSan + 线性化测试；simd：每后端对拍 scalar、数学函数 ULP 精度 |
| 批4 | **sched：百万次 fiber 切换基准、work-stealing 公平性、fiber-sync 正确性（TSan）、防惊群验证** |
| 批5 | io：echo/吞吐基准、proactor 四后端对齐；compute：device 枚举、CPU+GPU 后端对拍、completion 唤醒 fiber 链路通 |
| 批6 | graph：DAG 执行正确性；C-ABI shim：FFI 冒烟（TS/Python 各调一个）；全栈 sanitizer CI |

## 四、决策清单

### 已决（带依据/出处）

| 决策 | 内容 | 出处 / 谁拍板 |
|---|---|---|
| C++ 标准 | **C++23**（GCC14+/Clang18+/MSVC19.43+）| charter；**用户拍板**（看编译器支持后定 23）|
| 错误模型 | 零异常 + `std::expected`，保留 ASSIGN_OR_RAISE 宏 | charter |
| 命名/宏 | 继承 clay，前缀 `HYLE_` | charter |
| 协程 | `co_await` 驱动 fiber | charter/sched |
| executors | NVIDIA/stdexec，不等 C++26 STL | charter |
| modules | 暂缓 | charter |
| 单测 | **GoogleTest** | charter；**用户拍板** |
| 基准/查错 | nanobench + TSan/ASan/UBSan 分构建 + libFuzzer | charter |
| 对外沟通 | **纯 extern "C" C-ABI shim（ggml 式）** + 内部 C++23 | charter；**用户拍板**（选方案2 保 ABI 稳定）|
| io 模型 | proactor + io_uring/IOCP/kqueue/epoll 四后端，借 liburing 不借 asio | io-design |
| compute | device/stream/kernel 两层抽象 + 借上游 ggml（adapter 隔离）+ Vulkan compute 一等公民 + GPU completion 接 sched | compute-design；**用户确认**（ggml 解耦）|
| sched 切换 | boost.context **fcontext 裸原语**（不要 boost.fiber），ucontext 出局 | sched-design |
| sched 调度 | per-core 有界 ring + LIFO slot + global，spin-then-park + unpark 接力 | sched-design |
| sched 统一 | fiber=单元、thread=引擎、coroutine=挂起点、process=联邦（不共享队列）| sched-design |
| mem backing | 通用 malloc 借（不自造），专用分配器自造，修 5 bug | mem-design |
| container | **唯一造不借的 T0**，补 SwissTable SIMD 探测 | container-design |
| 模块分层 | Tier 0-3 + 双轴判据（地基性 × 差异性）| scope；**用户参与**（container 提到 T0）|
| 边界外 | 负空间 + L0/L1/L2 分层（协议/框架在 L1 另一项目）| scope |
| tier 倒置 | container(T0) 内联自包含探测片段，不依赖 simd(T1) | simd/container-design |
| device 内存 | **方案 D（MemorySpace 组合式）**：mem/ 定义 MemorySpace 接口+通用分配器+SystemRAM；compute/ 注册 device/pinned/unified space；分配器组合在 space 上；统一内存(DGX Spark/Mac) copy 塌缩为 no-op | mem/compute-design；**用户拍板**（组合优于继承 + 统一内存零损失）|
| simd 向量 | **scalable-friendly 接口 + 先实现定宽后端**（x86 SSE4.2/AVX2/AVX512 + ARM NEON）；将来加 SVE/RVV 不改 API | simd-design；**用户拍板** |
| hashmap 写法 | **concept 模板写内部 + 过边界特定类型封 C-ABI shim**；去掉 C 宏 temp/哨兵包袱 | container-design；**用户拍板** |

### 待决（有建议，多数不阻塞、可采纳默认后实现期微调）

### 待决（剩余均非阻塞，有安全默认、实现期可微调）

| # | 待决项 | 建议（默认采纳）| 阻塞? |
|---|---|---|---|
| 1 | mem backing 选哪个 | **mimalloc**（现代/THP/MIT/v3 2026）| 否 |
| 2 | mimalloc 怎么集成 | 封 `Allocator` vtable 后、不外露类型 | 否 |
| 5 | simd 数学核 | vendor sleef、封 adapter | 否 |
| 6 | SVE/RVV 进首版? | v1 只 x86(SSE4.2/AVX2/AVX512)+ARM NEON，留接口 | 否 |
| 8 | SwissTable 自实现 vs 抄 abseil | 自实现（边界+差异焦点）| 否 |
| 9 | lock-free 队列归属 | container/（供原语），sched 消费 | 否 |

> 三个"接口/边界形状"待决（#3 device 内存、#4 simd 向量、#7 hashmap 写法）均已拍板，见上「已决」表。**剩余 6 项全部非阻塞**——采纳默认即可开工，实现期按需微调。

**剩 4、7 两个"接口/边界形状"待拍**（影响多模块接口签名）；其余有安全默认，采纳即可、实现期可调。

### 实现期选择（非设计空白，落地时定策略）

阻塞 syscall 的 blocking 池 + sysmon 补线程；LIFO slot 轮询预算上限；steal 的 backoff 曲线；GPU completion 选回调还是轮询线程。这些在 sched-design 已标注，是落地参数、非设计悬念。

## 五、收口

设计阶段到此完整：**定位 → 范式（charter/conventions）→ 六支柱深挖（io/sched/compute/simd/mem/container）→ 分层/规模/负空间（scope）→ 依赖图/施工序（本文）**。`docs/architecture/` 共 12 份，互相交叉引用，无环可施工。

动手第一刀：**批0 的 `error/` + obs hook 接口**。三个接口形状决策（device 内存 D / simd scalable-friendly / hashmap concept 模板）已全部拍定，剩余待决项均非阻塞，可干净开工。
