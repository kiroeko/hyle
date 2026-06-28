# HYLE 实现路线图

> 动手前的收口：把设计里散落的硬依赖收成一张无环图，分解为**依赖层（阶段）× 里程碑（约 35-40 个，每个 1-4k 行可独立编/测）**两个轴，给每阶段定验收门槛，并汇总全部决策（已决 + 待决）。这既是施工计划，也是对全部设计的最后一次端到端校验。
>
> **注意区分**：依赖层（P0-P6）是"谁先能编译"的拓扑序、6 层；里程碑才是真正的实现步骤、约 35-40 个。别把 6 层当 6 步。

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

## 二、施工计划：两个轴别混

**澄清（之前的错）**：上一版"6 批"是**依赖层**（谁先谁后的拓扑序），**不是 6 个实现步骤**。65-100k 行分 6 步 = 每步上万行，不可落地。正确做法分两个轴：

- **轴 A — 依赖层（阶段 P0-P6）**：哪层先能编译，下面表的拓扑序。粗粒度。
- **轴 B — 里程碑（M）**：每模块拆 2-6 个里程碑，**每个 1-4k 行、各自可编可测可基准**。全项目约 **35-40 个里程碑**，这才是真正的"步骤"。

### 先走骨架（walking skeleton）——纵切，不要逐模块做完再下一个

第一目标**不是**把 error/ 做到 100% 再做 platform/。而是**每层只取最小一片，纵向打通一条能跑的脊**：

> **骨架里程碑 SK**：`error` 最小（Status+expected）→ `platform` 最小（thread+park）→ `mem` 最小（mimalloc 包一层，不修 bug 不加 space）→ `container` 最小（一个有界 ring，vendor 的）→ `sched` 最小（fcontext 切换 + 单 worker 单队列跑一个 fiber，**先不 work-stealing**）。

打通后 = "**一个 fiber 能在一个 worker 上跑起来、让出、被唤醒**"。这验证了最高风险的 fcontext+调度脊，之后每层只是"加厚"。**骨架 ≈ 5-8k 行、是整个项目的风险试金石**，必须最先做。

### 依赖层 × 里程碑分解（轴 A × 轴 B）

| 阶段 | 模块 | 里程碑（每个 1-4k 行，可独立编/测）| 生产 LOC |
|---|---|---|---|
| **P0** | error / obs-hook | M:Status+expected+ASSIGN_OR_RAISE宏 / M:obs 埋点 API（空实现）| 3-5k |
| **P1** | platform | M:thread+park(futex/WaitOnAddress) / M:vm+dynlib / M:path+fs+env | 5-8k |
| **P2** | mem | M:Allocator vtable+修alignment / M:mimalloc backing / M:arena+修64槽 / M:pool+stack重写 / M:MemorySpace+xcpu_freelist / M:NUMA+大页 | 6-10k |
| **P2** | time | M:单调钟 / M:分层时间轮 | 1.5-2.5k |
| **P3** | container | M:vendor 有界 ring/MPMC / M:XXH3 接入 / M:carray+RAII / M:SwissTable 探测层重写 / M:SmallVector+intrusive / M:string_ref+buf | 7-11k |
| **P3** | simd | M:vec D-tag 抽象+scalar后端 / M:SSE4.2+AVX2 / M:AVX512+NEON / M:运行期变体dispatch / M:dtype(fp16/bf16) / M:sleef 数学核 adapter | 10-16k |
| **P4** | sched | M:fcontext 6平台+栈池 / M:per-core run-queue / M:work-stealing+防惊群 / M:Waker completion 脊 / M:fiber-aware CV+同步原语 / M:协程胶水 | 10-15k |
| **P5** | io | M:proactor 抽象 / M:io_uring 后端 / M:epoll+kqueue+IOCP / M:net(tcp/udp) / M:file+shmem | 9-14k |
| **P5** | compute | M:device/stream 薄包ggml / M:MemorySpace 注册+copy塌缩 / M:GPU completion→fiber / M:kernel 启动 / M:ggml_bridge+注入线程池 / M:Vulkan compute kernel | 10-16k |
| **P6** | graph + 收口 | M:taskgraph 剥特化绑sched / M:C-ABI shim / M:serialize/hash/T3 / M:FFI 绑定 | 11-17k |

**约 35-40 个里程碑**，每个是一次"写完→编过→单测→基准→提交"的闭环。这才是能照着干的步骤表，而非 6 个万行黑箱。

**关键路径**（最长链）：`error → platform → mem → container → sched`。sched 是瓶颈，等 container 队列 + time + platform park 全就位。所以 **P2/P3 的 container/time 要尽早动**。simd 不在关键路径（只依赖 mem 对齐），可与 container 并行、晚点喂 compute。

### 当前实现状态（2026-06-28 快照，cloc 实测 HYLE 生产 5,383 行 / 目标 50–70k）

> 口径:整体 ≈ **25–30%**(模块权重)。**地基层基本立住,四大支柱(simd/io/compute/graph)未动**。这是 5,383 生产行的真实分布,不是"覆盖 clay 85%"(那是符号计数误读,见 clay-coverage-map 头部口径)。

| 阶段 | 模块 | 状态 | 说明 |
|---|---|---|---|
| P0 | error / obs | ✅ 基本完成 | error-02(vtable sink)、obs-01(log) |
| P1 | platform | ✅ 基本完成 | thread/park、dynlib、path/fs/env/os_info(platform-02/03) |
| P2 | mem | ✅ 主力完成 | Allocator vtable、mimalloc(v2.3.2,强制常开)、arena;pool/MemorySpace/NUMA/大页 📋未做 |
| P2 | time | 🔶 部分 | 单调钟(time-01)✅;分层时间轮 📋未做 |
| P3 | container | ✅ 主力完成 | BoundedQueue/WsFifoQueue、XXH3、CArray、SwissTable、string_ref/buf、concurrent_vector;SmallVector/intrusive 🔶部分 |
| P3 | **simd** | 🔶 起步 | simd-01 已落:运行期 CPUID dispatch 骨架(探测器隔离禁 LTO)+ vec D-tag 抽象(照搬 Highway 契约)+ scalar 地板后端,双平台 22/22。各 ISA intrinsic 后端(SSE2/AVX2/AVX512/NEON)/变体 DLL dispatch/dtype/数学核 📋未做 |
| P4 | sched | 🔶 核心done | fiber(fcontext)、work-stealing 调度器、Waker completion 脊(sched-03/04a)✅;fiber-aware CV/同步原语、协程胶水 📋未做 |
| P5 | **io** | ⛔ **未动** | 整条支柱未开工(proactor/io_uring/epoll/IOCP/net/file) |
| P5 | **compute** | ⛔ **未动** | 整条支柱未开工(device/stream/ggml_bridge/GPU completion/Vulkan) |
| P6 | **graph** | ⛔ **未动** | taskgraph/C-ABI shim/FFI 未开工 |

**已闭环里程碑**(真编真测双平台):error-02、obs-01、platform-02/03、mem-01/01b/02/03/03b、time-01、container-02/03/04/04b/04c/06/06b/06c/09、util-01..05、sched-03/04a、**simd-01**。**下一步候选**:① 续 simd 支柱(simd-02 SSE2/AVX2 实后端对拍 scalar——dispatch 骨架+对拍地板已就位);② 补 sched 后段(CV/同步原语/协程胶水);③ 补 mem 后段(pool/MemorySpace)。



| 阶段 | 门槛 |
|---|---|
| SK 骨架 | **一个 fiber 在单 worker 上跑起→让出→被唤醒**（fcontext+最小脊打通，最高风险先验）|
| P0 | error 跨三编译器编译；`std::expected` 往返；埋点 API 编译期零开销验证 |
| P1 | platform 三平台（Linux/Win/mac）编译；dynlib 加载冒烟；park 原语单测 |
| P2 | **mem：alignment 保证测试（64B 对齐必须真生效）**、并发分配 TSan、对标裸 mimalloc 不退化；time：单调性、时间轮到期精度 |
| P3 | container：hashmap 对拍 std + 基准对标 absl/F14、lock-free 队列 TSan + 线性化测试；simd：每后端对拍 scalar、数学函数 ULP 精度 |
| P4 | **sched：百万次 fiber 切换基准、work-stealing 公平性、fiber-sync 正确性（TSan）、防惊群验证** |
| P5 | io：echo/吞吐基准、proactor 四后端对齐；compute：device 枚举、CPU+GPU 后端对拍、completion 唤醒 fiber 链路通 |
| P6 | graph：DAG 执行正确性；C-ABI shim：FFI 冒烟（TS/Python 各调一个）；全栈 sanitizer CI |

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

### 待决（剩余均非阻塞，有安全默认、实现期可微调）

| # | 待决项 | 建议（默认采纳）| 阻塞? |
|---|---|---|---|
| 1 | mem backing 选哪个 | **mimalloc**（现代/THP/MIT/v3 2026）| 否 |
| 2 | mimalloc 怎么集成 | 封 `Allocator` vtable 后、不外露类型 | 否 |
| 5 | simd 数学核 | vendor sleef、封 adapter | 否 |
| 6 | SVE/RVV 进首版? | v1 只 x86(SSE4.2/AVX2/AVX512)+ARM NEON，留接口 | 否 |
| 8 | SwissTable 自实现 vs 抄 abseil | 自实现（边界+差异焦点）| 否 |
| 9 | lock-free 队列归属/造借 | **vendoring 成熟实现**（Vyukov bounded MPMC / rigtorp SPSC，单头可内联），归 container/ 供 sched 消费；不自造（无 C-ABI 理由、难写对）。自造预算集中到 hashmap | 否 |

> 三个"接口/边界形状"待决（#3 device 内存、#4 simd 向量、#7 hashmap 写法）均已拍板，见上「已决」表。**剩余 6 项全部非阻塞**——采纳默认即可开工，实现期按需微调。

### 实现期选择（非设计空白，落地时定策略）

阻塞 syscall 的 blocking 池 + sysmon 补线程；LIFO slot 轮询预算上限；steal 的 backoff 曲线；GPU completion 选回调还是轮询线程。这些在 sched-design 已标注，是落地参数、非设计悬念。

## 五、收口

设计阶段完整：**动机 → 定位 → 范式（charter/conventions）→ 六支柱深挖（io/sched/compute/simd/mem/container）→ 分层/规模/负空间（scope）→ 性能预期 → 参考库映射 → 依赖图/里程碑（本文）**。`docs/architecture/` 共 15 份，互相交叉引用，无环可施工。经 4 波参考库实证校验。

**实现是约 35-40 个里程碑（每个 1-4k 行、可编可测可基准），不是 6 个万行黑箱。** 动手第一刀：**走骨架 SK**——纵切 error/platform/mem/container/sched 各取最小一片，打通"一个 fiber 跑起→让出→唤醒"，先验最高风险的 fcontext+调度脊。骨架立住后，按依赖层逐里程碑加厚。所有接口形状决策已拍定、待决项非阻塞，可干净开工。
