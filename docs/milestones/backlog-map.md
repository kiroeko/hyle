# HYLE 欠账地图与排期(backlog map)

> **这份文档回答一个问题:所有"还没做"的里程碑,分别因为什么没做、前置齐了没、能不能现在开工。**
> 目的是**排期决策**——避免"能力驱动"沦为"永远等驱动、永远不做"。逐里程碑的完成状态与 LOC 仍以 [`README.md`](./README.md) 为唯一信源;本文件只管**慢变的"欠账原因分类 + 依赖就绪度 + 开工顺序"**,不重复维护状态(避免两处 drift)。
>
> 更新纪律:一个欠账里程碑落地后,从下面对应类目移除;新登记的缺口按类目归入。

## 一、三类欠账(原因不同,处理方式不同)

"没实现"不是一个原因。区分清楚才能排期:

| 类 | 含义 | 处理原则 |
|---|---|---|
| **A 模块广度延期** | 地基/支柱模块内部,当时优先级低或缺硬件,**无下游能力在等** | 按价值/成本自行排期,不需要"消费者驱动"。该做就做 |
| **B 能力驱动的半成品机制** | sched/io 里有前置依赖、孤立做会拍错接口的机制,**挂在轴 C 能力上等驱动** | 由其能力里程碑落地时定形。**关键:一旦前置齐 + 能力被选中,就必须开工,不能无限等** |
| **C 依赖序未到的主体支柱** | compute/graph 两大支柱,不在"等驱动"逻辑里——它们本身是北极星驱动的支柱,只是**关键路径还没走到** | 走到即开工。是最大的剩余工作量 |

> **对"一直不做"的自省(2026-07-04)**:B 类的"等消费者驱动"是防投机的纪律(sched-04d 曾因此推迟是对的——当时接口会拍错)。但它**不该变成拖延的借口**:当一个能力的前置全部就绪时,它就该进入排期,而不是继续挂着。下面的"就绪度"列就是防这个——**标 🟢 就绪的必须排期,不能再等**。

## 二、A 类:模块广度延期(无下游等待,按价值排期)

| 里程碑 | 缺什么 | 就绪度 | 建议 |
|---|---|---|---|
| platform-02b | virtual_memory(host 端 mmap/VirtualAlloc 封装:保留/提交分离 + guard page) | 🟢 就绪 | **host 内存管理,与显存无关**。真实消费者=**fiber 栈 guard page 溢出保护**(mem-design fiber_stack.h)。**纠正旧记:compute device buffer 走 ggml buffer_type、显式 DMA,不用它**——离散 GPU 在 VRAM 里算,host 不可解引用。无紧迫消费者,按需 |
| mem-06 | NUMA + 大页 + stats 下溢修复 | 🟢 就绪 | 优先级低(单机未压 NUMA);北极星未要求,**可缓** |
| simd-03 尾 | NEON / AVX512 真机跑 | 🔴 **缺硬件**(无 ARM 机 / 无 AVX512 机) | 已写+编+objdump 验;**硬阻塞**,待真机/QEMU/SDE。非优先级问题 |
| bench-02 | 对标 SOTA 套(absl/F14/moody/oneTBB/highway/marl) | 🟢 就绪 | 分散已做(各模块审查时对标过);整合成套可缓 |
| **compute-06 收尾②·性能收尾** | kompute 核对表标"抄/改"但验收未要的性能项:#2 spec constants、#1 pipeline cache 落盘、#14/#16 rerecord/replay、#11/#12 barrier 精确化(现一刀切 global)、#6 HOST_CACHED 档。(**#25 timestamp query 已 2026-07-08 完成=compute-06b**,记录见 compute-06 里程碑状态行,已从本行移除) | 🟢 就绪 | **A 类低优先**:原语完整可用后的加速。barrier 精确化最该先(核对表点名 SOTA 关键)。逐项见 compute-06 里程碑核对表 |

> container-06c(模糊路径匹配)**2026-07-05 ✅ 完成**,已从本表移除(抄 fzf V1 两趟算法,benchmark 证 O(N),container 160/160)。
> **compute-06 内部欠账(2026-07-08 登记)**:compute-06 按验收(三平台 vecadd+合批+co_await 全绿)已 ✅,但 kompute 25 点核对表"应做(抄/改)"21 项里做了 12、欠 9。**里程碑"明确不做"表(compute-06-vulkan-spirv.md:191)已接住多队列/运行时GLSL/Image/spirv-opt(B类消费者驱动,合理等);上面两行接住剩下无归属的**——① 是正确性硬伤(基础库义务,必做)、② 是性能收尾。避免这些散落在核对表/memory 里无排期。

**A 类结论**:全部"就绪但低优先级",不阻塞北极星。simd-03 硬件硬阻塞。**platform-02b 的真实用途是 host 端 fiber 栈 guard page + 保留/提交型分配(与显存无关)**,唯一现实消费者是 sched 的栈溢出保护,不阻塞 compute。剩余 A 类(compute-06 收尾②性能/mem-06/bench-02)按需——性能收尾里 **barrier 精确化(#11/#12)最该先做**(核对表点名 SOTA 关键)。(compute-06 收尾① Vulkan staging、收尾②-a #25 timestamp query[=compute-06b] 均已 2026-07-08 完成,记录见 compute-06 里程碑文档状态行,已从本表移除。)

## 三、B 类:能力驱动的半成品机制(前置就绪度是排期关键)

这曾是"集中在轴 C"的大头。**2026-07-05:C2/C3/C4/C9/C11(HTTP/1.1)全部 ✅ 落地,B 类基本清空**,仅剩两个小尾(io-05 shmem / io-11 HTTP/2)。

| 能力 | 机制里程碑 | 前置 | 就绪度 | 说明 |
|---|---|---|---|---|
| **C2 带超时 io** | io-06 + sched-04d | io-04 ✅ / time-02 ✅ / sched-10 select ✅ | ✅ **完成(2026-07-05)** | io-06 落地:timer vs io 竞态(shard-per-core 单线程免原子仲裁)+ submit_cancel 三后端 + refcount slot 续命防迟到 CQE UAF + IoContext orphan drain。**顺带落地 sched-04d**(shard-per-core 免去其最硬的跨核 cancel 队列)。TimedRecv/Accept/ConnectOp;红队审 + 计数分配器验漏;三后端双平台 + TSan 净 |
| **C3 结构化取消** | sched-07 | 无硬前置(C2 已定形 io cancel;io-10 HTTP 消费者已到位) | ✅ **完成(2026-07-05)** | shard-per-core 无锁取消树落地:唯一原子是 per-node cancelled 标志(兼容 is_cancel_requested seam),树/回调链/派生/传播/析构过继全非原子。借 tokio-util tree_node 拓扑+析构过继+迭代传播(去 Mutex)+ folly CancellationCallback(去原子握手)。CancelToken/CancelScope/CancelCallback/HYLE_CHECK_CANCEL + io/net_cancel.h 可取消 io(栈 slot 无 heap,经 tokio-uring 核实 oneshot cancel 恰一个 CQE)。消费者=HTTP 请求 fan-out(io-10)。红队审揪修 fire_callbacks 相邻注销 UAF。Win+WSL(io_uring+epoll)+TSan 全绿。**sched 支柱 01~11 全 ✅** |
| **C4 优雅关闭** | sched-08 | signal ✅(C10);**不依赖 C3**(调研推翻) | ✅ **完成(2026-07-05)** | 调研发现关闭只需平信号广播不需取消树→与 C3 **解耦**。sched-08 落地:accepting 接纳位两阶段(begin_shutdown 拒新+广播 / drain_for 超时)+ shutdown_requested() 长命 fiber 轮询退出。有栈 fiber 超时不强杀(协作退/detach)。signal_set(C10)作触发器。双平台 sched 85/85 + TSan 净 |
| **C5 文件/共享内存** | io-05 | io-02 ✅ | 🔶 **文件 io ✅ / shmem 待** | 文件 io 三后端(open/close/statx/fsync/fallocate + co_await read/write)已 ✅。**shmem(mmap 跨进程共享内存)未做**——无当前消费者(compute device buffer 走显式 copy,不是 shmem;IPC 场景北极星未要求),🟢 就绪但**按需**,与 platform-02b(virtual_memory)可一并做 |
| **C9 TLS** | io-08 | io-04 ✅(over co_await stream) | ✅ **完成(三平台,2026-07-05)** | OpenSSL memory-BIO 泵 over co_await TcpStream:TlsContext/TlsStream + tls_handshake/read/write/shutdown。红队审净+修 2 隐患。Linux io_uring+epoll+TSan 净 + Windows IOCP 净(vcpkg OpenSSL 3.6.3)。可选依赖(不装 OpenSSL 零依赖) |
| **C11 HTTP** | io-10(HTTP/1.1)+ io-11(HTTP/2) | 超时 C2 ✅ + TLS C9 ✅ + DNS C10 ✅ | 🔶 **io-10 ✅ 完成(2026-07-05)** / io-11 后置 | 北极星① REST 门面。io-10 落地:借 vendored picohttpparser(596 行 C)+ 自写集成层(co_await TcpStream→parser→Request/Response→write)。两级 API + client + chunked 流式(SSE)。红队审修 3 真 bug + chunked 跨平台 bug。三后端×常规+TSan 全绿。**驱动 sched-07**(请求 fan-out 取消)待做。HTTP/2(io-11)借 nghttp2 后置 |

**B 类结论——2026-07-05 基本清空**:
1. ~~io-06 / C2(带超时 io)~~ ✅ **完成**。顺带落地 sched-04d。
2. ~~sched-08 / C4(优雅关闭)~~ ✅ **完成**。调研解耦了 C3。
3. ~~io-08 / C9(TLS)~~ ✅ **完成(三平台)**。OpenSSL memory-BIO,红队审净。
4. ~~io-10 / C11(HTTP/1.1)~~ ✅ **完成(三平台)**。借 picohttpparser + 自写集成层。**北极星① 服务器骨架最后一块**。
5. ~~sched-07 / C3(取消)~~ ✅ **完成**。io-10 HTTP fan-out 驱动。**sched 支柱 01~11 全 ✅ 真收口**。
6. **仅剩两个小尾**:**io-05 shmem**(C5,🟢 就绪无消费者,按需/随 platform-02b 一并做)+ **io-11 HTTP/2**(⏸️ 借 nghttp2,等 HTTP/1.1 生产验证后再动)。两者都不阻塞北极星。
7. **下一大块 = compute 支柱**(C 类,0 行、前置就绪、北极星① 算力本体)。

## 四、C 类:依赖序未到的主体支柱(最大剩余块)

| 支柱 | 里程碑 | 前置 | 就绪度 | 说明 |
|---|---|---|---|---|
| **compute** | compute-01 ✅ / 02~08 | sched-04 ✅ | 🟢 **就绪**(01 已破土) | 主干 01~06:device/stream/event(✅)→ device buffer → GPU completion→fiber → kernel 启动 → ggml_bridge → Vulkan。**缺口补登(2026-07-06)**:07 CPU 算子后端(接 simd,对拍 oracle + 无 GPU 基线)、08 HIP/Metal(优先借 ggml 覆盖、原生后置、硬件阻塞)。借 ggml 不自造算子。**异构混合极致性能的头牌本体**,北极星① 推理引擎的算力侧 |
| **graph** | graph-01 + graph-02? | sched-03 ✅ + compute(kernel) | 🟡 **部分**(graph-01 可先,substrate Chase-Lev 已备) | 北极星② CAD/BIM/CAE 驱动:脏标记增量重算 + 拓扑并行。graph-02(数据并行 parallel_for)归属待拍(见收口待决) |

**C 类结论**:compute **已破土(compute-01 ✅)**——它是北极星① 的算力支柱本体,**排期上是一个大阶段**(主干 01~06 借 ggml ~5-8k + 缺口 07 CPU 后端 / 08 HIP·Metal)。**缺口补登(2026-07-06)**:07 CPU 算子后端(scope §CPU fallback + roadmap 验收"CPU+GPU 对拍"要求的真后端,非 compute-01 的 host inline 兜底)、08 AMD·Apple 后端(scope 后端矩阵列了但主干未覆盖,借 ggml-hip/metal 优先)。graph 依赖 compute 的 kernel,排在 compute 之后。

## 五、收口层待决(需你拍板,非纯排期)

| 里程碑 | 状态 | 待决 |
|---|---|---|
| close-01 C-ABI shim | 📋 承诺锁定待排期 | 各模块稳定后做;发布给 FFI 的唯一稳定面 |
| close-02 serialize | ❓ scope 待澄清 | 借 flatbuffers/capnproto builder+reader 当原语 vs 全推 L1?JSON 借 simdjson 已定 |
| close-03 FFI 绑定 | 占位 | 依赖 close-01 |
| graph-02 数据并行层 | ❓ 归属待定 | 归 graph 还是 compute?与序列化一起在 scope 复核时拍 |

## 六、一句话排期建议

**北极星① 服务器骨架已完整(2026-07-05)**:C1 并发 io / C2 超时 / C4 优雅关闭 / C9 TLS / C10 DNS·signal / **C11 HTTP/1.1 全部落地**,sched 支柱收口(01~11 全 ✅ + 三复杂度优化)。**serving 侧基本齐了**——从 socket 到 HTTP 请求/响应、超时、TLS、优雅关闭、请求 fan-out 取消,一条 co_await 链跑通,三后端(io_uring/IOCP/epoll)+ TSan 全绿。**下一大块 = compute 支柱**(0 行、前置 sched-04 就绪、北极星① 算力本体):device/stream 薄包 ggml-backend → device buffer → GPU completion→fiber → kernel 启动 → ggml_bridge → Vulkan,借 ggml 不自造算子。推理引擎 = serving(✅ io/HTTP)+ 算力(compute,待做),两条线合起来才是完整北极星①。**io-05 shmem / io-11 HTTP/2 / A 类广度**按需穿插,simd-03 硬件阻塞除外。
