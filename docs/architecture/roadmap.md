# HYLE 实现路线图

> 动手前的收口：把设计里散落的硬依赖收成一张无环图，分解为**依赖层（阶段）× 里程碑（约 58 个，每个 1-4k 行可独立编/测）**两个轴，给每阶段定验收门槛，并汇总全部决策（已决 + 待决）。这既是施工计划，也是对全部设计的最后一次端到端校验。
>
> **注意区分**：依赖层（P0-P6）是"谁先能编译"的拓扑序、6 层；里程碑才是真正的实现步骤、约 58 个（milestones/README 为准）。别把 6 层当 6 步。

## 一、模块依赖图（验证无环）

箭头 = "依赖/被阻塞于"。只画硬依赖（编译期或接口契约级）。

```
error/  ────────────────────────────────────┐ (status + std::expected，零上游)
  │                                          │
  ▼                                          │
platform/ (thread/vm/dynlib/path/fs/park)    │
  │                                          │
  ├──────────────┐                           │
  ▼              ▼                            │  obs/ hook 接口
mem/          time/                          │  (最小埋点 API，
(mimalloc      (单调钟+                       │   P0 先定、空实现，
 backing,       时间轮)                       │   各模块热路径嵌)
 修 alignment)   │                            │
  │              │                            │   simd/  ← 依赖 platform(借 CPUID
  ▼              │                            │   (可移植       探测,2026-07 下沉);不 include
container/       │                            │    vector+      mem;container 自带内联 SSE
(ring/MPMC/      │                            │    dispatch)    探测片段,不依赖 simd/,但同样
 intrusive/      │                            │    │            借 platform 的 CPUID 上 AVX2 变体)
 SwissTable)     │                            │    │  喂 compute kernel
  │              │                            │    │
  └────────┬─────┴────────────────────────────────┘
           ▼
   ┌─────────────────────────┐
   │  sched/  【关键路径瓶颈】 │  ← 依赖 container(队列原语)+time(deadline)
   │  fcontext+work-stealing  │     +platform(park)+mem(fiber栈)
   │  +Waker completion 脊    │     （不依赖 simd）
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

**无环校验**：error → platform → {mem,time,simd} → container → sched → {io,compute} → graph。单向，无回边。`simd/` **依赖 platform**（2026-07：CPUID 探测 `CpuFeatures`/`best_isa` 下沉 platform，simd 借它做运行期 ISA dispatch；此前 simd 是"零 HYLE 依赖根"、只用 `platform.h` 宏伞，下沉后升级为依赖 platform 模块本身）——仍可与 mem/container/sched 链**大体并行**，最终只喂 compute 的 kernel，**不在通往 sched 的关键路径上**。CPUID 下沉 platform 让 container 也能查同一探测、安全上 AVX2 变体（如 container·str 的 count/rfind），不必依赖 simd。（曾计划 simd 依赖 mem 的对齐契约，但 kernel 走 LoadU/StoreU 非对齐 load，该边未落地。）

## 二、施工计划：两个轴别混

**澄清**：依赖层（阶段 P0-P6）是**拓扑序**（谁先能编译），**不是实现步骤**。把总量分成 6 大步 = 每步上万行，不可落地。正确做法分两个轴：

- **轴 A — 依赖层（阶段 P0-P6）**：哪层先能编译，下面表的拓扑序。粗粒度。
- **轴 B — 里程碑（M）**：每模块拆 2-6 个里程碑，**每个 1-4k 行、各自可编可测可基准**。全项目约 **58 个里程碑**（milestones/README 为准），这才是真正的"步骤"。
- **轴 C — 交付能力（C）**：用户/消费者视角"这个库能做什么"。轴 B 是**机制**（收割接缝、时间轮、work-stealing），轴 C 是**能力**（带超时的网络 io、优雅关闭、取消）。见下节。

### 轴 C：交付能力清单（消费者视角）——机制的反向拉动者

> **为什么需要这一轴（2026-07-03 补，源于一次真实教训）**：计划最初只有轴 A/B，两者都是**自底向上的机制**。结果做到 sched-04d（fiber 定时等待 + 跨核 cancel 队列）时发现**没有消费者**——因为它服务的能力"带超时的网络 io"从没被列出来。机制清单会堆出**悬空机制**：某块底层机制做完了却没人用，因为顶层能力没被登记来反向拉动它。轴 C 修这个：先列北极星能力，每个能力**驱动**（consumer-drives）一组机制里程碑；**没有能力驱动的机制里程碑不做**（等它，避免拍错接口——这正是 sched-04d 当初被推迟的理由）。
>
> **北极星**：能跑一个**真实的网络服务器**（不是 echo 玩具）——多连接、带超时、可取消、能优雅关闭。这是 HYLE io/sched 支柱的验收标尺。

| 能力 | 一句话 | 驱动的机制里程碑 | 状态 |
|---|---|---|---|
| **C1 并发 io 服务** | 多连接 co_await recv/send/accept，shard-per-core | io-01/02/03/04 ✅；**io-07a**（multishot accept）✅；**io-07b**（multishot recv + provided buffer ring）✅ | ✅ **半成品全补完**：`Completion` 当初照 liburing 预留的 `kMore`(multishot)/`kBuffer`(零拷贝 buffer ring) 两 flag,**kMore 随 io-07a multishot accept、kBuffer 随 io-07b buffer ring + multishot recv 全接上**(io_uring 原生;accept 在 IOCP/epoll 有 emulation,buffer ring io_uring only)。留 flag 不接的"没抄全"已清 |
| **C2 带超时的 io** | `recv/accept/connect` 带 deadline，超时取消在途 op | **io-06**（io_uring `ASYNC_CANCEL`/IOCP `CancelIoEx` 取消在途 op）**驱动** sched-04d（fiber 定时等待 + 跨核 timer cancel 队列）；底层需 **C7 的 select**（等 timer/io 先完成者）| 📋 未排期。**几乎所有网络库的生存能力**（tokio `timeout()`/Go `SetReadDeadline`/asio `cancel_after`/seastar `with_timeout`）——无它，慢/死连接永久占 fiber+fd+buffer。sched-04d 的真实消费者 |
| **C3 结构化取消** | CancelToken 树 + await 取消点，父取消传播子 | sched-07 | 📋 未排期。C2 的 io 取消是它的一个特例；先做 C2 定形 io 侧 cancel，再泛化到 C3 |
| **C4 优雅关闭** | 两阶段 drain：停收新连接 → 排空在途 → 逆序清理 | sched-08（依赖 sched-07）；触发靠 **C10 的 signal 处理**（SIGTERM/SIGINT）| 📋 未排期。真服务器停机/热重启必需 |
| **C5 文件/共享内存 io** | 异步文件读写（openat/statx/fsync/fallocate）+ 跨进程 shmem | io-05 | 📋 未排期。**独立能力**，不依赖 C2（文件 io 基本不用超时取消）。审计补：io-05 原设计只写"复用 read/write"，真异步文件服务还需 async open/stat/fsync/fallocate（io_uring 均有 op）|
| **C6 可观测性** | 结构化日志 + trace/counter 埋点（对标 tracy）| obs-01 ✅ + **obs-02**（instrumentation/trace hook）| 🔶 日志已交付。**缺口**：所谓"埋点 hook"目前**只存在于一句 CMake 注释**,无真 seam；tracy 有 zone/plot/lock/mem + **fiber-aware 宏**(TracyFiberEnter/Leave),恰配我们的 fiber 调度器。obs-02 定义 hook + 把 tracy 接为编译期后端 |
| **C7 并发原语补全** | Semaphore / RwLock / oneshot / broadcast·watch / JoinHandle+select·when_all | **sched-09**（Semaphore/RwLock/oneshot/broadcast）+ **sched-10**（JoinHandle + select/when_all 结构化并发）| 📋 未排期。审计发现 tokio/seastar/folly 全有、HYLE 缺:**Semaphore**(异步限流标准件,限在途连接/buffer 池)、**JoinHandle**(spawn 现返 void,拿不到结果/不能 abort)、**select/when_all**(等 N 中第一个并取消其余=超时/可取消 io 的底层)。sched-10 的 select 是 C2 的前置 |
| **C8 阻塞卸载** | `spawn_blocking`：把阻塞 syscall/CPU 活扔到卸载池,不拖死 shard worker | **sched-11**（阻塞卸载线程池 + sysmon）| 📋 未排期。sched-design §2.4 **自己承认**"一个阻塞 syscall 拖死整 worker/core"却无 API。tokio/marl/seastar 都有。真服务器调用阻塞库(DB 驱动/文件/getaddrinfo)必需 |
| **C9 TLS** | 加密 socket（over co_await stream）| **io-08**（借 BoringSSL/OpenSSL 做 shim over 现有 stream）| 📋 未排期。asio/libevent 都当核心。**任何面向公网的服务器必需**,roadmap 原本完全没有=真漏 |
| **C10 异步 DNS + 信号** | 异步域名解析 + signal_set（SIGTERM/SIGINT）| **io-09** | 📋 未排期。审计发现 `resolve_addr` 是同步 `getaddrinfo`,**阻塞整个 shard worker**(非一个 fiber),违背 share-nothing——真客户端必需。signal 处理是 C4 优雅关闭的触发器 |
| **C11 HTTP serving/client** | HTTP/1.1 server+client over 自有 io_uring/fiber（借纯 parser + 自造集成）| **io-10**（HTTP/1.1，借 picohttpparser + 集成层）+ **io-11**（HTTP/2，借 nghttp2，后置）| 📋 未排期,依赖 C2·C9·C10(超时/TLS/DNS)。**类 llama.cpp 推理引擎必需 REST**=北极星一等能力。定案:三层分工(IO 自有 / 解析借纯 parser / 集成自写 / 路由业务留上层);**不结合完整库**(asio/beast/httplib C++ 里结合不干净,详见 scope HTTP 专项)。性能靠"数据不离开 io_uring/fiber、无双重 poll/无阻塞线程池"。**验收标尺**:比 nginx 的正确战场=高并发动态+AI serving(与 compute 同进程零阻抗),非纯静态伺服 |

> **读法**：轴 C 决定"下一个做什么"（选一个未交付能力 → 做它驱动的机制）；轴 B 决定"这个能力怎么拆成可编可测的步骤"。**机制里程碑若在轴 C 找不到驱动它的能力，就是投机，不做**（例：孤立的 kqueue/SQPOLL/sched-04d 都曾是这类——SQPOLL 由"延迟受限 io"驱动、sched-04d 由 C2 驱动，能力未排期前都不做）。
>
> **C7-C10 的来源（2026-07-03 参考库审计）**：拿 tokio/seastar/folly/liburing/asio/libevent/tracy/oneTBB 当能力清单逐条对照 HYLE,把"它们有、我们设计没有"的真缺口登记为 C7-C10 + io-05/06/07 的补完。**这些是"待驱动"占位——按轴 C 规矩,能力未被选中前不动手,避免又造悬空机制。** 另有两项**需你拍板的 scope 澄清**（非直接排期）：① **序列化**——close-02 把"serialize/hash/T3"糊成占位、点名 flatbuffers/simdjson 却无设计;scope 内部打架("二进制序列化原语"IN vs "IDL/RPC"OUT)。待决:借 flatbuffers builder/reader 当原语,还是把 wire-format 全推 L1。② **数据并行层**（parallel_for/reduce/scan over range,oneTBB/oneDPL）——不同于 graph 支柱(DAG 编排);是 sched+simd 的天然消费者(大 buffer 上 compute kernel),不被 scope 排除,可作为 compute 支柱的一部分登记。

### 先走骨架（walking skeleton）——纵切，不要逐模块做完再下一个

第一目标**不是**把 error/ 做到 100% 再做 platform/。而是**每层只取最小一片，纵向打通一条能跑的脊**：

> **骨架里程碑 SK**：`error` 最小（Status+expected）→ `platform` 最小（thread+park）→ `mem` 最小（mimalloc 包一层，不修 bug 不加 space）→ `container` 最小（一个有界 ring，vendor 的）→ `sched` 最小（fcontext 切换 + 单 worker 单队列跑一个 fiber，**先不 work-stealing**）。

打通后 = "**一个 fiber 能在一个 worker 上跑起来、让出、被唤醒**"。这验证了最高风险的 fcontext+调度脊，之后每层只是"加厚"。**骨架 ≈ 5-8k 行、是整个项目的风险试金石**，必须最先做。

### 依赖层 × 里程碑分解（轴 A × 轴 B）

> **生产 LOC 列已 2026-06-29 用实测下修**(右列 = 再投影)。旧列锚"对标参考库体量"偏高 ~2x(foundation 5-6x、pillar ~2x);**已实现的非玩具**(33 真 op/零 stub/红队揪真 bug),下修一半因 std/C++23 取代 + 借第三方真精简、一半因部分广度仍待铺(会长回)。完整拆解 + 第三方(ggml 主导 ~150-250k)见 [`hyle-scope.md` 规模节](./hyle-scope.md)。

| 阶段 | 模块 | 里程碑（每个可独立编/测）| 旧预测 | 再投影(实测口径) |
|---|---|---|---|---|
| **P0** | error / obs-hook | Status+expected+ASSIGN_OR_RAISE / obs 埋点 | 3-5k | ~0.8k ✅ |
| **P1** | platform | thread+park / vm+dynlib / path+fs+env | 5-8k | ~1.2k ✅(vm 待) |
| **P2** | mem | Allocator+alignment / mimalloc / arena / pool+stack / NUMA | 6-10k | ~0.9k ✅主力(arena/pool/stack/growable 净新) |
| **P2** | time | 单调钟 / 分层时间轮 | 1.5-2.5k | ~0.3k 🔶(时间轮待) |
| **P3** | container | ring/MPMC / XXH3 / carray / SwissTable / SmallVector / string | 7-11k | ~2k ✅主力 |
| **P3** | simd | vec+scalar / SSE4.2+AVX2 / AVX512+NEON / 变体dispatch / dtype / 数学核(Highway+xsimd) | 10-16k | ~2.3k 已做 + ~3-5k 广度补全 = **~5-8k** |
| **P4** | sched | fcontext+栈池 / run-queue / work-stealing / Waker / CV+同步 / 协程胶水 | 10-15k | ~1.3k 核心+同步+协程胶水+fiber亲和✅(fcontext/work-stealing/Waker/conditional-notify/CV+Mutex+WaitGroup+Latch+Channel 对标 marl 追平反超 + Task/co_await + spawn_on 绑核) + ~2-4k 后段(取消 07/优雅关闭 08/fiber-mode 定时 04d/**并发原语补全 09**/**JoinHandle+select 10**/**阻塞卸载 11**)——**均由轴 C 能力驱动,未排期前不做**(04d/10←C2·C7,07←C3,08←C4,09·10←C7,11←C8) |
| **P5** | io | proactor / io_uring / epoll+kqueue+IOCP / net / file+shmem / HTTP | 9-14k | ~1.8k 已做(proactor 三后端全真 op + co_await + shard-per-core + Tcp/Udp 封装 + echo/pipelined bench + **multishot accept/recv + provided buffer ring 07a/b**)✅ + ~5-9k 后段(file/shmem 05·带超时 06·**TLS 08**·**异步DNS+signal 09**·**HTTP/1.1 10**·**HTTP/2 11**;kqueue)——由 C2/C5/C9/C10/C11 驱动 |
| **P5** | compute | device/stream包ggml / device buffer(compute 自有) / GPU completion / kernel / ggml_bridge / Vulkan | 10-16k | **~5-8k**(借 ggml 写 bridge) |
| **P6** | graph + 收口 | taskgraph绑sched / C-ABI shim / serialize/hash/T3 / FFI | 11-17k | **~6-11k**(多借 simdjson 等) |

**HYLE 自有合计:~28-40k 生产 / ~52-68k 含测试**(实测 test/prod≈0.65;2026-07-03 参考库审计后上修——补登 sched-09/10/11 + io-06/07/08/09/**10/11(HTTP)** + obs-02 等真缺口机制,分母回涨 ~6-10k;HTTP serving+client 是推理引擎北极星的一等能力)。**约 58 个里程碑**(milestones/README 为准),每个"写完→编过→单测→基准→提交"闭环。

> **⚠️ scope 负空间待复核（2026-07-03）**：审计里被判"有意不做(L1/运维栈 out-of-scope)"的项(如 TLS 一度、log 聚合、序列化边界、数据并行归属等)——其中**部分可能判错了**,`hyle-scope.md` 的负空间是项目早期写的,本身可能"漏点"。这些**不当已决**,留作后续讨论;TLS 已从"排除"改判为 C9 真能力就是一例修正。序列化 IN/OUT 边界、数据并行层归属尤其待拍。

**关键路径**（最长链）：`error → platform → mem → container → sched`。sched 是瓶颈，等 container 队列 + time + platform park 全就位。所以 **P2/P3 的 container/time 要尽早动**。simd 不在关键路径（依赖 platform 借 CPUID 探测，其余零 HYLE 依赖），可与整条链大体并行、晚点喂 compute。

### 当前实现状态 → 见 milestones/README（唯一进度信源）

> **进度不在这里维护**（避免两处状态表互相 drift）。逐里程碑的真实状态 + cloc 实测 LOC +
> clay API diff 是 [`../milestones/README.md`](../milestones/README.md) 的"全部里程碑"表，**那是唯一信源**。
> 本文件只管**拓扑序、施工计划、决策、验收门槛**这些慢变内容。
>
> 一句话现状（粗粒度，细节看 milestones）：**地基层（error/platform/mem/container/time/util/obs）基本立住；
> sched 核心 + fiber 同步原语 done（work-stealing + Waker 脊 + conditional-notify 防惊群 + CV/Mutex/WaitGroup/Latch/Channel,对标 marl 全线追平/反超）+ 协程胶水（Task + co_await→fiber 让出）+ fiber 亲和（spawn_on 绑核，shard-per-core 基座），后段（取消 sched-07/优雅关闭 sched-08/fiber-mode 定时 04d）**待其轴 C 能力驱动**（见轴 C：04d←C2 带超时 io、07←C3 结构化取消、08←C4 优雅关闭——孤立做=投机）；**simd 支柱基本完成**（dispatch + vec 抽象 +
> scalar/SSE4.2/AVX2 实跑 + 运行期变体派发 + fp16/bf16 dtype + 数学核(Highway+xsimd vendored),对标 Highway 追平；NEON/AVX512 写+编+objdump 验未跑存疑）；**io 支柱破土并接线到 sched**（proactor 三后端 io_uring/IOCP/epoll 全真 op 无 stub + co_await socket 层 + shard-per-core 集成 + TcpListener/Stream + UdpSocket(recvfrom/sendto 三后端)，真 loopback TCP/UDP echo 三后端+TSan；echo+pipelined benchmark 已补，worker 空闲已阻塞 GETEVENTS reap=修了 pipelined 揪出的 P0 hang；**io-04 net 完成 ✅**（含 park 合并：空闲 worker 真睡眠、eventfd 唤醒，空闲 syscall 421→3）+ **io-07a/b multishot accept/recv + provided buffer ring ✅**（补完 kMore/kBuffer 两半成品 flag）；file/shmem io-05 + 带超时/TLS/DNS/HTTP 待）；**compute/graph 两支柱未开工**（异构混合极致性能的头牌本体）。**
> 整体 ≈ 33%（LOC 口径,2026-07-03 参考库审计 + HTTP 定案后分母上修到 ~28-40k;见 milestones/README 三分母表 + "审计上修"注；compute/graph + io HTTP 是最大剩余块）。



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
| device 内存 | **compute/ 自有 stream-bound buffer（不走 mem/ allocator 抽象）**：device/pinned/unified buffer 各立一个 ggml 式 buffer_type 实例，所有权 + stream 绑定在 compute/;host↔device 边界是显式 copy/command-submit。统一内存(DGX Spark/Mac)无"互传"概念,本就无拷贝可省 | mem/compute-design；**异构内存不强行统一**（device 走 drawcall/提交、统一内存无互传,故 mem/ 只管 host RAM）|
| simd 向量 | **scalable-friendly 接口 + 先实现定宽后端**（x86 SSE4.2/AVX2/AVX512 + ARM NEON）；将来加 SVE/RVV 不改 API | simd-design；**用户拍板** |
| hashmap 写法 | **concept 模板写内部 + 过边界特定类型封 C-ABI shim**；去掉 C 宏 temp/哨兵包袱 | container-design；**用户拍板** |

### 待决（剩余均非阻塞，有安全默认、实现期可微调）

| # | 待决项 | 建议（默认采纳）| 阻塞? |
|---|---|---|---|
| 1 | mem backing 选哪个 | **mimalloc**（现代/THP/MIT/v3 2026）| 否 |
| 2 | mimalloc 怎么集成 | 封 `Allocator` vtable 后、不外露类型 | 否 |
| 5 | simd 数学核 | vendor Highway+xsimd、封 adapter | 否 |
| 6 | SVE/RVV 进首版? | v1 只 x86(SSE4.2/AVX2/AVX512)+ARM NEON，留接口 | 否 |
| 8 | SwissTable 自实现 vs 抄 abseil | 自实现（边界+差异焦点）| 否 |
| 9 | lock-free 队列归属/造借 | **vendoring 成熟实现**（Vyukov bounded MPMC / rigtorp SPSC，单头可内联），归 container/ 供 sched 消费；不自造（无 C-ABI 理由、难写对）。自造预算集中到 hashmap | 否 |

> 三个"接口/边界形状"待决（#3 device 内存、#4 simd 向量、#7 hashmap 写法）均已拍板，见上「已决」表。**剩余 6 项全部非阻塞**——采纳默认即可开工，实现期按需微调。

### 实现期选择（非设计空白，落地时定策略）

LIFO slot 轮询预算上限；steal 的 backoff 曲线；GPU completion 选回调还是轮询线程。这些在 sched-design 已标注，是落地参数、非设计悬念。（**注**：原列此处的"阻塞 syscall 的 blocking 池 + sysmon"已在 2026-07-03 审计中**升级为真能力缺口** `spawn_blocking`——见轴 C 的 C8 / sched-11,不再当"实现期参数"糊过去,因为它是 tokio/seastar/marl 都有的公开 API。）

## 五、收口

设计阶段完整：**动机 → 定位 → 范式（charter/conventions）→ 六支柱深挖（io/sched/compute/simd/mem/container）→ 分层/规模/负空间（scope）→ 性能预期 → 参考库映射 → 依赖图/里程碑（本文）**。`docs/architecture/` 共 19 份，互相交叉引用，无环可施工。经 4 波参考库实证校验。

**实现是约 58 个里程碑（每个 1-4k 行、可编可测可基准），不是 6 个万行黑箱。** 动手第一刀：**走骨架 SK**——纵切 error/platform/mem/container/sched 各取最小一片，打通"一个 fiber 跑起→让出→唤醒"，先验最高风险的 fcontext+调度脊。骨架立住后，按依赖层逐里程碑加厚。所有接口形状决策已拍定、待决项非阻塞，可干净开工。
