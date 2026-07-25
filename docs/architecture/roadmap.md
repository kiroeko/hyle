# HYLE 实现路线图

> 本页把设计里散落的硬依赖收成无环图，分解为**依赖层 × 里程碑 × 交付能力 × 模块收口波次**四个轴，给每层定验收门槛，并维护慢变的关键决策。它既是施工顺序，也是对整体设计的端到端校验。
>
> **注意区分**：依赖层（P0-P6）是“谁先能编译”的拓扑序；里程碑是真正的
> 实现步骤；收口波次则是“谁先接受实际使用导向的发布复审”。三者不是一张表。

## 一、模块依赖图（验证无环）

本节维护**当前产品构建与公开头的真实依赖**，不是按目录名猜层级，也不是只画 scheduler
关键路径。图中 `A ──▶ B` 表示 **B 依赖 A**；`A ··▶ B` 是 header-only 编译依赖；
`A - -▶ B` 是可选或跨切构建边。精确的 PUBLIC/PRIVATE/条件边以紧随其后的表为准。

```text
主调度链：error ──▶ platform ──▶ mem ──▶ container ──▶ sched ──▶ io
时间分支：core/header ··▶ time-native；error ──▶ time ──▶ sched
计算分支：platform ──▶ simd ──▶ compute；sched ──▶ compute
图分支：  container ──▶ graph；sched ──▶ graph
util 产品：container ──▶ util-runtime

util-core ··▶ mem / sched / graph
hyle_trace - -▶ every hyle_add_module target - -▶ one optional Tracy client
error ──▶ obs-runtime ──▶ application/FFI sinks
math ──▶ users（当前无生产下游）
```

### 构建目标真相表

下表列直接产品模块边，不重复每个 `hyle_add_module` 都自动 PUBLIC 连接的
`hyle_trace` / `hyle_testing` gate；前者在图中单列，后者只是测试构建的 ODR 配置面，不是发布模块。

| 模块/目标 | PUBLIC target 依赖 | PRIVATE / 条件 target 依赖 | header-only / 跨切面 |
|---|---|---|---|
| `core` | — | — | 基础宏、intrusive 原语供底层头使用 |
| `error` | — | — | 全库可恢复失败值根 |
| `obs-runtime` (`hyle_obs`) | error | — | log/C ABI 产品；共享 Status；与 trace build seam 分开 |
| `math` | —（INTERFACE） | — | 独立公开标量助手，当前无生产模块消费者 |
| `platform` | error | — | OS/thread/VM/fs/CPU 机制；`hyle/platform.h` 宏头本身零链接 |
| `mem` | error、platform | mimalloc | `util::SpinLock`、obs trace hook |
| `time` | error | — | native timer 借 `core/intrusive_list`；不链接 platform/container |
| `container` | mem | platform | 当前无 util-core 边；ConcurrentVector 的非短写临界区使用 mutex |
| `simd` | platform | — | 运行期 ISA 能力由 platform 提供 |
| `util-runtime` (`hyle_util`) | container | — | registry 同时借 mem；其余 util headers 不依赖该二进制目标 |
| `sched` | error、mem、container、platform | time | `util::SpinLock`、obs trace hook |
| `io` | — | error、mem、container；有 fcontext 时 sched | obs trace hook |
| `compute` | error、mem；有 fcontext 时 sched | platform、simd；可选 GPU/ggml 供应商 | obs trace hook |
| `graph` | error、mem、container；有 fcontext 时 sched | — | `util::SpinLock` |

### 两个不能压成单节点的模块

`util` 必须分层阅读。`SpinLock/cpu_relax`、`Ref`、`ScopeGuard`、RNG 与 type-name 是独立
header-only 能力，可被低层模块直接使用；`Registry` 和 type-erased C ABI 才依赖
container/mem，因而编译出的 `hyle_util` 位于 container 之后。container 不 include
util-core 且不链接 `hyle_util`，所以 target graph 没有 `container ↔ util` 环。新增
util 能力时必须先判断属于哪一层，禁止让 util-core 反向 include container。
`add_subdirectory` 在根 CMake 中的文本先后不是依赖层；这里只以 target link 与公开头合同判定顺序。

`obs` 同样有两条线：`hyle_obs` 是独立的同步日志/C ABI 产品；`hyle_trace` 是
`HyleModule.cmake` 自动 PUBLIC 传播给每个模块的 INTERFACE gate。默认 OFF 时它为空，开启时
所有模块汇入同一个 `hyle_tracy_client`。产品代码 include `obs/trace.h` 不等于链接日志 DLL，
依赖图必须用跨切虚线而不是伪造 `hyle_obs → 全库` 的硬边。

### 无环结论与已知收束边

实际 target graph 无环。最长基础链仍是
`error → platform → mem → container → {sched, util-runtime}`；time、simd 分别从 error、
platform 分叉，sched 再汇入 io/compute/graph。`math` 是当前没有生产下游的合法用户叶；它的
存在不应因此从完整模块图消失。

部分旧 C shim（util/simd/sched/io）仍使用模块私有 `last_error`，是 error adoption
审计尚未完成的显式欠账。obs 与 container 已在 2026-07-25 复审中迁到共享
`hyle_status`；container 的 C 冷失败现在形成直接 `error → container` 单向边，不会
导致 error 反向依赖或形成环。

## 二、施工计划：四个轴别混

**澄清**：依赖层（阶段 P0-P6）是**拓扑序**（谁先能编译），**不是实现步骤或当前复审队列**。把总量只分成 7 个拓扑层仍不能当施工清单；真正的施工粒度在 milestone。正确做法分四个轴：

- **轴 A — 依赖层（阶段 P0-P6）**：哪层先能编译，下面表的拓扑序。粗粒度。
- **轴 B — 里程碑（M）**：每个 work item 自成“实现→正确性→性能→文档”闭环。数量、实测 LOC 和当前状态只在 milestones/README 维护。
- **轴 C — 交付能力（C）**：用户/消费者视角"这个库能做什么"。轴 B 是**机制**（收割接缝、timer queue、work-stealing），轴 C 是**能力**（带超时的网络 io、优雅关闭、取消）。见下节。
- **轴 D — 模块严格收口（R）**：已有代码按“真实场景→成熟库→API/复杂度→C/C++→下游采用→文档”重新认证。它按接口爆炸半径排序，不按旧 benchmark 或目录顺序。

### 轴 C：交付能力清单（消费者视角）——机制的反向拉动者

> **为什么需要这一轴（2026-07-03 补，源于一次真实教训）**：计划最初只有轴 A/B，两者都是**自底向上的机制**。结果做到 sched-04d（fiber 定时等待 + 跨核 cancel 队列）时发现**没有消费者**——因为它服务的能力"带超时的网络 io"从没被列出来。机制清单会堆出**悬空机制**：某块底层机制做完了却没人用，因为顶层能力没被登记来反向拉动它。轴 C 修这个：先列北极星能力，每个能力**驱动**（consumer-drives）一组机制里程碑；**没有能力驱动的机制里程碑不做**（等它，避免拍错接口——这正是 sched-04d 当初被推迟的理由）。
>
> **北极星**（两个四样全压的具体 workload，见 motivation）：**① 类 llama.cpp 推理引擎**——能跑一个真实的网络服务器（不是 echo 玩具，多连接/带超时/可取消/优雅关闭）+ serving 与 compute 同进程零阻抗，验收 io/sched/mem/compute 四支柱；**② CAD/BIM/CAE 工程计算管线**——参数化/仿真依赖 DAG 增量重算 + 几何/BVH compute kernel + CAE 的稀疏线代/迭代求解器，验收 graph 支柱（推理引擎不碰的那根）+ compute/simd/mem（CAE 把 compute/simd 压到求解器深度）。两锚合起来是 HYLE 全部五支柱的验收标尺。

| 能力 | 一句话 | 驱动的机制里程碑 | 状态 |
|---|---|---|---|
| **C1 并发 io 服务** | 多连接 co_await recv/send/accept，shard-per-core | io-01/02/03/04 ✅；**io-07a**（multishot accept）✅；**io-07b**（multishot recv + provided buffer ring）✅ | ✅ **半成品全补完**：`Completion` 当初照 liburing 预留的 `kMore`(multishot)/`kBuffer`(零拷贝 buffer ring) 两 flag,**kMore 随 io-07a multishot accept、kBuffer 随 io-07b buffer ring + multishot recv 全接上**(io_uring 原生;accept 在 IOCP/epoll 有 emulation,buffer ring io_uring only)。留 flag 不接的"没抄全"已清 |
| **C2 带超时的 io** | TCP read/write/connect/accept 与 UDP recv/send 带 deadline，超时取消在途 op | **io-06**（io_uring `ASYNC_CANCEL`/IOCP `CancelIoEx` 取消在途 op）**驱动** sched-04d（fiber 定时等待 + 跨核 timer cancel 队列）；底层需 **C7 的 select**（等 timer/io 先完成者）| ✅ **完成(2026-07-20)**。timed 网络便利面已铺齐；timer vs io 竞态**先到者胜**。**shard-per-core 单线程串行=非原子仲裁**(io completion + timer fire + resume 全在 owner worker,seastar 式免原子);对称 gate 每 episode 恰 1 次 wake_rendezvous。timer 赢→`submit_cancel` 四后端(io_uring ASYNC_CANCEL/IOCP CancelIoEx/epoll+kqueue 删除 watch 并合成);**迟到 CQE 续命**(heap TimedSlot refcount,普通 op 栈 slot 零成本不变)防 UAF。**顺带落地 sched-04d**——但 shard-per-core **免去其最硬的跨核 cancel 队列**。红队审 + 计数分配器验漏 + 三后端测试 |
| **C3 结构化取消** | CancelToken 树 + await 取消点，父取消传播子 | sched-07 | ✅ sched CancelToken 树与 io TCP/UDP cancellable convenience 已接通；HTTP whole-request 统一 cancel/deadline 仍是产品层尾项 |
| **C4 优雅关闭** | 两阶段 drain：停收新连接 → 排空在途 → 逆序清理 | sched-08；触发靠 **C10 的 signal 处理**（SIGTERM/SIGINT）| ✅ **完成(2026-07-05)**。调研 seastar gate/tokio TaskTracker 后落地:关闭只需**平的关闭信号广播 + 在途计数归零,不需 sched-07 取消树**(与 C3 解耦)。sched-08:`begin_shutdown()` 置 accepting=false 拒新 spawn + 广播全 worker;`drain_for(timeout)` 等 outstanding 归零(超时返 false);`shutdown_requested()` 长命 fiber 轮询退出。有栈 fiber 超时不强杀(协作退/detach)。signal_set(C10 ✅)作触发器。双平台 sched 85/85 + WSL TSan 净 |
| **C5 文件/共享内存 io** | 异步文件读写（openat/statx/fsync/fallocate）+ 跨进程 shmem | io-05 | ✅ 文件原语 + `File` RAII + **命名 shared memory**（Win file mapping / POSIX `shm_open`，C++ RAII + C opaque ABI）。普通 C++ `File` 按 capability 自动选原生异步或 `IoContext` blocking pool：io_uring 数据/metadata 快腿、IOCP data overlapped，IOCP metadata 与 epoll/kqueue regular-file 自动卸载，不堵 worker；低层 Proactor 保留精确机制语义 |
| **C6 可观测性** | 结构化日志 + trace/counter 埋点（对标 Tracy）| obs-01/02 ✅；2026-07-25 严格收口 | ✅ 同步 borrowed record + immutable sink snapshot + quiescent unregister/flush/递归抑制 + 共享 Status C ABI；默认零成本 native trace seam 覆盖 zone/plot/counter/fiber/frame/message/memory/thread-name，已接 mem/sched/io/compute。共享构建只加载一个 Tracy client。架构与验收见 [`obs-design.md`](./obs-design.md) / [`../milestones/obs-module-closure.md`](../milestones/obs-module-closure.md) |
| **C7 并发原语补全** | Semaphore / RwLock / oneshot / broadcast·watch / JoinHandle+select·when_all | **sched-09**（Semaphore/RwLock/oneshot/broadcast/watch）+ **sched-10**（JoinHandle + when_all/when_any(select) 结构化并发）| ✅ **完成**（sched-10 JoinHandle+spawn_task+when_all/when_any;sched-09 Semaphore/RwLock/oneshot/watch/broadcast 2026-07-03 落地,tokio 语义抄全建在 CV 地基,双平台 73/73 + WSL TSan 净;**select=C2 前置已就位**） |
| **C8 阻塞卸载** | `spawn_blocking`：把阻塞 syscall/CPU 活扔到卸载池,不拖死 shard worker | **sched-11**（阻塞卸载线程池） | ✅ **完成**（2026-07-03,sched-11:懒创建池上限512+10s keep-alive,结果经现有 JoinHandle 回传,双平台82/82+WSL TSan 净）。sysmon 不做(tokio 也无,卸载显式)。真服务器调阻塞库(DB驱动/文件/getaddrinfo)可用 |
| **C9 TLS** | 加密 socket（over co_await stream）| **io-08**（借 OpenSSL memory-BIO shim over 现有 stream）| ✅ **完成(三平台,2026-07-05)**。借 OpenSSL memory-BIO 泵:SSL 引擎只做握手+加解密(纯内存),密文收发由 co_await TcpStream 驱动(SSL_get_error WANT_READ/WRITE→flush_wbio/feed_rbio)。TlsContext/TlsStream + 自由函数 tls_handshake/read/write/shutdown。踩坑全抄(MOVING_WRITE_BUFFER/错误队列/EOF 三分/BIO 所有权)。红队审三大面净+修 2 隐患。loopback TLS 握手+加密 echo:Linux io_uring+epoll+TSan 净 / **Windows IOCP 净**(vcpkg OpenSSL 3.6.3,机制与平台无关一次编过) |
| **C10 异步 DNS + 信号** | 异步域名解析 + signal_set（SIGTERM/SIGINT）| **io-09** | ✅ **完成**。`dns.h/cc` `resolve()` 经 sched-11 spawn_blocking 卸载 getaddrinfo(抄 tokio,非自造 DNS 协议),值拷贝+freeaddrinfo 不泄指针,JoinHandle 回传;修了 `resolve_addr` 只 inet_pton(连 getaddrinfo 都没有,比"同步阻塞"更糟)。`signal.h/cc` SignalSet co_await wait():POSIX self-pipe+sigaction 进程全局单例,Windows SetConsoleCtrlHandler(CTRL_CLOSE handler 阻塞防提前杀);单 waiter 模型。三后端真测 + TSan 净。signal 是 **C4 优雅关闭的触发器**(已就位) |
| **C11 HTTP serving/client** | HTTP/1.1 server+client over 自有 proactor/fiber（借纯 parser + 自造集成）| **io-10**（HTTP/1.1，借 picohttpparser + 集成层）+ **io-11**（HTTP/2，借 nghttp2）| ✅ **io-10 HTTP/1.1 + HTTPS；io-11 HTTP/2 三平台真后端全绿**。H2 有 h2c/TLS ALPN、HPACK、多 stream、RST/GOAWAY、跨 64 KiB window 流控、graceful shutdown、C++ convenience 与 transport-neutral C codec。尾项：whole-request deadline/cancel 与 H2 无界 chunk writer |
| **C12 参数化/仿真依赖 DAG 增量重算** | 节点图：改一个输入只重算下游脏节点，重算子图并行跑在 work-stealing sched 上、几何/数值/求解 kernel 走 compute | **graph-01**（taskgraph 绑 sched：脏标记传播 + 拓扑并行调度 + 节点结果缓存）+ **graph-02**（数据并行 parallel_for/reduce over range，与 graph 编排区分）| 📋 未排期。**北极星② CAD/BIM/CAE 一等能力**（clay taskgraph 老本行：Task/schema/socket/kernel/graph_io）。推理引擎不碰 graph，故这是**独占拉满 graph 支柱**的驱动者（CAD 参数化重算、CAE 仿真流水/求解迭代同构）。把 graph 从"P6 纯收口件"提为能力驱动的支柱；依赖 sched（节点跑 fiber/work-stealing）+ compute（几何/求解 kernel）。**验收标尺**：一个真实参数化/仿真模型（改参数→增量重算，对比全量重算的加速比 + 重算可中途取消，接 C3）|

> **读法**：轴 C 决定"下一个做什么"（选一个未交付能力 → 做它驱动的机制）；轴 B 决定"这个能力怎么拆成可编可测的步骤"。**机制里程碑若在轴 C 找不到驱动它的能力，就是投机，不做**（例：孤立的 kqueue/SQPOLL/sched-04d 都曾是这类——SQPOLL 由"延迟受限 io"驱动、sched-04d 由 C2 驱动，能力未排期前都不做）。
>
> **C7-C10 的来源（2026-07-03 参考库审计）**：拿 tokio/seastar/folly/liburing/asio/libevent/tracy/oneTBB 当能力清单逐条对照 HYLE,把"它们有、我们设计没有"的真缺口登记为 C7-C10 + io-05/06/07 的补完。**这些是"待驱动"占位——按轴 C 规矩,能力未被选中前不动手,避免又造悬空机制。** 另有两项**需你拍板的 scope 澄清**（非直接排期）：① **序列化**——close-02 把"serialize/hash/T3"糊成占位、点名 flatbuffers/simdjson 却无设计;scope 内部打架("二进制序列化原语"IN vs "IDL/RPC"OUT)。待决:借 flatbuffers builder/reader 当原语,还是把 wire-format 全推 L1。② **数据并行层**（parallel_for/reduce/scan over range,oneTBB/oneDPL）——不同于 graph 支柱(DAG 编排);是 sched+simd 的天然消费者(大 buffer 上 compute kernel),不被 scope 排除,可作为 compute 支柱的一部分登记。

### 先走骨架（walking skeleton）——纵切，不要逐模块做完再下一个

第一目标**不是**把 error/ 做到 100% 再做 platform/。而是**每层只取最小一片，纵向打通一条能跑的脊**：

> **骨架里程碑 SK**：`error` 最小（Status+expected）→ `platform` 最小（thread+park）→ `mem` 最小（mimalloc 包一层，不修 bug 不加 space）→ `container` 最小（一个有界 ring，vendor 的）→ `sched` 最小（fcontext 切换 + 单 worker 单队列跑一个 fiber，**先不 work-stealing**）。

打通后 = "**一个 fiber 能在一个 worker 上跑起来、让出、被唤醒**"。这验证了最高风险的 fcontext+调度脊，之后每层只是"加厚"。**骨架 ≈ 5-8k 行、是整个项目的风险试金石**，必须最先做。

### 依赖层 × 里程碑分解（轴 A × 轴 B）

P0–P6 只表示最早可编译的拓扑层。不在这里复制 LOC、冻结日期或完成度；它们分别属于
[`hyle-scope.md`](./hyle-scope.md) 和 [`milestones/README.md`](../milestones/README.md)。

| 拓扑层 | 本层可建立的目标 | 为什么在这一层 |
|---|---|---|
| **P0** | core、error、obs-runtime、math、util-core headers、`hyle_trace` gate | obs-runtime 只依赖同层 error；其余零 HYLE 模块上游；先冻结基础值、宏、header 合同与跨切 gate |
| **P1** | platform、time | 两者只需 P0；time 不需等 platform |
| **P2** | mem、simd | 都需 platform；mem 另直接需 error |
| **P3** | container | 公开依赖 mem，实现依赖 platform |
| **P4** | sched、util-runtime/registry | 两者都要 container；sched 另要 time/platform/mem/error |
| **P5** | io、compute、graph | 完整产品形状都消费 sched；compute 另消费 simd，graph/io 消费 container |
| **P6** | install/export、跨编译器 ABI/FFI 冒烟、发布 bundle | 不是新模块；在各模块 C adapter 就地收口后做整库交付验收 |

**关键路径**：到 scheduler 的最长链是
`error → platform → mem → container → sched`；独立的 `error → time → sched` 分支必须在
P4 前汇合。util-runtime 与 sched 在 container 之后并行，但 util-core headers 从 P0 就可用。
simd 不进 sched 链，只在 compute 汇合。obs-runtime 是用户产品根，trace gate 则横切全库；
二者都不应被“不在最长链”误删。

### 轴 D：模块严格收口顺序

“实现完成”“旧 closure 任务完成”和“按当前标准通过预发布认证”是三件事。收口按接口
爆炸半径走：先稳定根、header seam 与跨切面，再审其直接消费者；一个模块被修改后，只对
已经认证的直接依赖/消费者做**受影响项回验**，不把整库重新打开。

| 波次 | 审查单元 | 排序理由 |
|---|---|---|
| **R0 基线** | error、platform、mem、time | 已按当前实际使用标准形成严格证书；后续只有触发器或依赖受影响项可重开 |
| **R1 根、构建与接缝** | core ✅ → build/package system ✅ → util-core seam ✅ → obs ✅ | 根、装配、低层 util 头与可观测横切 seam 已按当前合同冻结；下一单元进入 container |
| **R2 容器层** | container → util-runtime/完整 util | container 是 sched/io/graph 和 Registry 的地基；先稳表/字符串/并发容器，再冻结依赖它的 Registry/C ABI |
| **R3 运行时链** | sched → io | sched 汇合 mem/time/container/platform，io 再消费 sched；反序会反复改 timeout/cancel/completion 形状 |
| **R4 计算基础** | math↔simd 边界审查 → simd | math 是 clay math 被 std/SIMD 吸收后的残余，不先假定它应独立；先裁决 scalar/std/SIMD 归属，再冻结 SIMD dispatch、类型、算法与正常 batch API |
| **R5 未完支柱** | compute → graph | 这两者不是“复审已完成实现”，而是先补真实能力再按同一合同收口；graph 最后汇合 sched/compute/simd |
| **R6 发布** | install/export → 全库 C/FFI 冒烟 → bundle | 每模块 C ABI 随本模块收口，不把接口设计推到最后；本层只做跨模块交付验证 |

R1 的 build/package system 是独立审查单元：CMake target 图必须准确表达模块源码、公开头、
依赖、C ABI、测试和 benchmark 归属；整库/模块子集、static/shared、CPU-only/可选后端可组合；
source embed、`add_subdirectory`、install/`find_package`、CI/交叉编译有一致迁移路径；preset、
toolchain 和脚本执行环境契约，文档只解释。完成它之后，R1 的 util 才审
`SpinLock/cpu_relax`、`Ref`、`ScopeGuard`、RNG/type-name 等低层公开头的
依赖与形状；Registry 和整个 `hyle_util` 产品必须等 container 后在 R2 完成。这是两个真实
构建层的门，不是对同一模块做两轮无边界复审。当前严格状态与唯一“下一模块”只见
[`pre-release-readiness.md`](../milestones/pre-release-readiness.md)。

math 不按“28 行小模块”单独快速封口。它不是从用户场景独立设计出来的产品，而是 clay math
迁移后，大部分能力被标准库或 hyle::simd 承载所剩的集合。R4 必须逐项回答“为什么是
`hyle::math` 而不是 std、simd 或消费域”，再决定保留命名空间、迁移能力、合并目标或删除；
在边界裁决前不为当前偶然 surface 机械扩 C ABI。

每个 R 波次都使用同一合同：

1. 从真实用户/HYLE 内部消费者场景列能力与负空间，再与本地成熟库逐项对账；参考库是查漏与算法/形状证据，不是照抄功能数量。
2. 先审 C++ source API 的 ownership、lifetime、concurrency、failure 和组合方式；不得保留只为 benchmark 存在的类型、元数据或入口。
3. 证明复杂度档与同合同正常用法性能；微基准只解释机制，不能反向塑造产品 API。点值、方法和 win/loss 只写 performance。
4. C ABI 覆盖全部语言无关能力，使用共享 error 合同；C++ 模块内部不绕经 C ABI。纯 C 头、导出、生命周期和 FFI 组合必须机械验证。
5. 同轮检查依赖的两面：本模块是否正确消费已冻结地基，以及直接下游是否仍自造重复机制或错误调用；只回验实际受影响边。
6. 交付 architecture、语言无关/C++ 选型 guide、`c-<module>` 精确映射、坑点、测试和性能证书；milestone 只保存施工记录。

## 三、验收门槛与状态信源

> **进度不在这里维护**（避免两处状态表互相 drift）。逐里程碑的真实状态 + cloc 实测 LOC +
> clay API diff 是 [`../milestones/README.md`](../milestones/README.md) 的"全部里程碑"表，**那是唯一信源**。
> 本文件只管**拓扑序、施工计划、决策、验收门槛**这些慢变内容。
>
> 本节有意不再复制“一句话现状”、模块冻结表、LOC 或完成百分比：这些快变信息曾多次与
> milestones 漂移。roadmap 只回答依赖顺序和验收门；当前状态沿上面的唯一信源链接读取。



| 阶段 | 门槛 |
|---|---|
| SK 骨架 | **一个 fiber 在单 worker 上跑起→让出→被唤醒**（fcontext+最小脊打通，最高风险先验）|
| P0 | core/error/math/util-core 公开头自包含；Status/Result 所有权；obs log 生命周期；trace OFF 参数不求值、ON 只有一个 backend client |
| P1 | platform 三平台编译与 OS 机制冒烟；time 单调/墙钟分离、饱和 deadline、exact-ns radix/common queue oracle |
| P2 | mem 对齐/NUMA/pool/allocator 正确性与 mimalloc peer；simd 每后端对拍 scalar、ULP 与运行期 ISA 安全 |
| P3 | container 的 hash/table/string/队列分层对标；lock-free 线性化与 TSan；公开头的 util-core include 边不得引入 `hyle_util` 链接环 |
| P4 | sched 的 fiber/work-stealing/sync/timer 汇合；util-runtime 的 registry/ref/RNG/C ABI；两者都跑 TSan/ASan/UBSan 和对应热路 peer |
| P5 | io 四后端与真网络/file；compute CPU/GPU 对拍与 completion；graph 增量 DAG/数据并行在 sched 上的真消费 |
| P6 | install/export/package；纯 C 与跨编译器/FFI 冒烟；全库 sanitizer/ABI surface 和发布 bundle 验收 |

## 四、决策清单

### 已决（带依据/出处）

| 决策 | 内容 | 出处 / 谁拍板 |
|---|---|---|
| C++ 标准 | **C++23**（GCC14+/Clang18+/MSVC19.43+）| charter；**用户拍板**（看编译器支持后定 23）|
| 错误模型 | 零异常 + `std::expected`，保留 `ASSIGN_OR_RETURN` / `RETURN_IF_ERROR` 宏 | charter + error-design |
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

### 历史选型（均已落地，保留决策来源）

| # | 当时的选型 | 落地决定 | 当前性质 |
|---|---|---|---|
| 1 | mem backing 选哪个 | **mimalloc**，封在 `Allocator` 后 | 已落地；当前设计见 mem-design |
| 2 | mimalloc 怎么集成 | 不外露 mimalloc 类型 | 已落地 |
| 5 | simd 数学核 | 成熟核与 HYLE adapter/自有算法面分层 | 已落地；当前供应商见 simd-design |
| 6 | SVE/RVV 进首版? | 先实现定宽 x86/NEON，API 保留 scalable 扩展面 | 已落地 |
| 8 | SwissTable 自实现 vs 抄 abseil | 自实现控制组/表结构，对标成熟 peer | 已落地 |
| 9 | lock-free 队列归属/造借 | 归 container 供 sched 消费，按成熟算法形状实现/验证 | 已落地 |

> 三个“接口/边界形状”决策（#3 device 内存、#4 simd 向量、#7 hashmap 写法）见上「已决」表。本表不是 backlog；新的未决项必须进 milestones，不在 roadmap 里假装为“安全默认”。

### 实现期选择（非设计空白，落地时定策略）

steal backoff 曲线是可根据真负载调整的参数，不是架构空白。LIFO poll 上限、GPU
completion poller 和 `spawn_blocking` 均已落地；其当前机制见 sched/compute design，当时的实施记录见 milestones。

## 五、维护门

- `src/*/CMakeLists.txt` 的 target 依赖变化时，同一提交更新第一节真相表。
- 公开头新增跨模块 include 时，必须判断是 PUBLIC target 边还是可证明自包含的 header-only 边。
- util/obs 变更必须分别检查两层；禁止把 util-core 与 util-runtime、obs-runtime 与 trace gate 再次压成一个假节点。
- 实时 LOC/状态只写 hyle-scope/milestones，性能点值只写 performance，本文只维护拓扑、能力顺序和慢变决策。
- CI 应增加 CMake target graph 与公开头 include 扫描；人工 ASCII 只做可读摘要，真相表必须可由构建配置复核。
