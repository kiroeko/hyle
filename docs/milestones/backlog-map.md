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
| platform-02b | virtual_memory(mmap/VirtualAlloc 封装) | 🟢 就绪 | 小件,compute device buffer 可能用到,可随手补 |
| mem-06 | NUMA + 大页 + stats 下溢修复 | 🟢 就绪 | 优先级低(单机未压 NUMA);北极星未要求,**可缓** |
| container-06c | 模糊路径匹配(clay 只声明无实现=净新) | 🟢 就绪 | 无消费者,**需要时再做**(非吸纳欠账) |
| simd-03 尾 | NEON / AVX512 真机跑 | 🔴 **缺硬件**(无 ARM 机 / 无 AVX512 机) | 已写+编+objdump 验;**硬阻塞**,待真机/QEMU/SDE。非优先级问题 |
| bench-02 | 对标 SOTA 套(absl/F14/moody/oneTBB/highway/marl) | 🟢 就绪 | 分散已做(各模块审查时对标过);整合成套可缓 |

**A 类结论**:除 simd-03(硬件硬阻塞)外全部"就绪但低优先级"。**platform-02b 值得随 compute 开工时一并补**(device buffer 要 mmap)。其余不阻塞北极星,按需。

## 三、B 类:能力驱动的半成品机制(前置就绪度是排期关键)

这是你问的"集中在轴 C"的部分。核心排期杠杆 = **C2(带超时 io)已 ✅**,下一杠杆 = **C3(取消)+ C9(TLS)**。

| 能力 | 机制里程碑 | 前置 | 就绪度 | 说明 |
|---|---|---|---|---|
| **C2 带超时 io** | io-06 + sched-04d | io-04 ✅ / time-02 ✅ / sched-10 select ✅ | ✅ **完成(2026-07-05)** | io-06 落地:timer vs io 竞态(shard-per-core 单线程免原子仲裁)+ submit_cancel 三后端 + refcount slot 续命防迟到 CQE UAF + IoContext orphan drain。**顺带落地 sched-04d**(shard-per-core 免去其最硬的跨核 cancel 队列)。TimedRecv/Accept/ConnectOp;红队审 + 计数分配器验漏;三后端双平台 + TSan 净 |
| **C3 结构化取消** | sched-07 | 无硬前置(C2 已定形 io cancel;但**无消费者**) | 🟡 **等消费者(HTTP fan-out)** | **重新定位(2026-07-05)**:sched-08 已证明优雅关闭**不需要**取消树(用平信号即可,已落地)。sched-07 的树价值="取消一请求→连带取消其派生的子 task",真实消费者是 HTTP 请求 fan-out(io-10)。现有 cancel_req/JoinHandle::abort(sched-10)+ io-06 submit_cancel 已覆盖单 task/单 io 取消。孤立做树=投机,等 io-10 驱动。最小形态已调研定案 |
| **C4 优雅关闭** | sched-08 | signal ✅(C10);**不依赖 C3**(调研推翻) | ✅ **完成(2026-07-05)** | 调研发现关闭只需平信号广播不需取消树→与 C3 **解耦**。sched-08 落地:accepting 接纳位两阶段(begin_shutdown 拒新+广播 / drain_for 超时)+ shutdown_requested() 长命 fiber 轮询退出。有栈 fiber 超时不强杀(协作退/detach)。signal_set(C10)作触发器。双平台 sched 85/85 + TSan 净 |
| **C9 TLS** | io-08 | io-04 ✅(over co_await stream) | 🟢 **就绪** | **不依赖 C2/C3**!借 BoringSSL/OpenSSL shim 套在现有 TcpStream 上。**当前最该开工(前置全齐 + 北极星刚需)** |
| **C11 HTTP** | io-10(HTTP/1.1)+ io-11(HTTP/2) | C2 超时 ✅ + C9 TLS + C10 DNS ✅ | 🟡 **等 C9** | 北极星① 刚需。超时/DNS 已就位;只等 TLS。HTTP/1.1 先,HTTP/2 借 nghttp2 后置。**io-10 会驱动 C3(请求 fan-out 取消)** |

**B 类结论——明确的开工顺序**:
1. ~~io-06 / C2(带超时 io)~~ ✅ **完成**。顺带落地 sched-04d。
2. ~~sched-08 / C4(优雅关闭)~~ ✅ **完成(2026-07-05)**。调研解耦了 C3——关闭只需平信号。**sched 支柱由此收口**(01~11 全 ✅,仅 07 取消树等 HTTP 消费者)。
3. **io-08 / C9(TLS)** 🟢 — **下一顺位**,前置全齐(不依赖 C2/C3/C4),北极星① 刚需。借 BoringSSL/OpenSSL。
4. **io-10 / C11(HTTP/1.1)** — C9 齐后(超时+DNS 已就位),北极星① REST 门面;它会驱动 C3(sched-07 取消树)。
5. **sched-07 / C3(取消)** — 由 io-10 HTTP fan-out 驱动,不再孤立做。
6. **compute 支柱**(C 类)可与 io HTTP 线并行推进(0 行、前置就绪、北极星① 算力本体)。

## 四、C 类:依赖序未到的主体支柱(最大剩余块)

| 支柱 | 里程碑 | 前置 | 就绪度 | 说明 |
|---|---|---|---|---|
| **compute** | compute-01~06 | sched-04 ✅ | 🟢 **就绪**(0 行) | device/stream 薄包 ggml-backend → device buffer → GPU completion→fiber → kernel 启动 → ggml_bridge → Vulkan。借 ggml 不自造算子。**异构混合极致性能的头牌本体**,北极星① 推理引擎的算力侧 |
| **graph** | graph-01 + graph-02? | sched-03 ✅ + compute(kernel) | 🟡 **部分**(graph-01 可先,substrate Chase-Lev 已备) | 北极星② CAD/BIM/CAE 驱动:脏标记增量重算 + 拓扑并行。graph-02(数据并行 parallel_for)归属待拍(见收口待决) |

**C 类结论**:compute **前置就绪、0 行、是最大缺口**。它不属"等驱动"——它是北极星① 的算力支柱本体,**排期上应视为一个大阶段**(6 个里程碑,借 ggml 但仍 ~5-8k)。graph 依赖 compute 的 kernel,排在 compute 之后。

## 五、收口层待决(需你拍板,非纯排期)

| 里程碑 | 状态 | 待决 |
|---|---|---|
| close-01 C-ABI shim | 📋 承诺锁定待排期 | 各模块稳定后做;发布给 FFI 的唯一稳定面 |
| close-02 serialize | ❓ scope 待澄清 | 借 flatbuffers/capnproto builder+reader 当原语 vs 全推 L1?JSON 借 simdjson 已定 |
| close-03 FFI 绑定 | 占位 | 依赖 close-01 |
| graph-02 数据并行层 | ❓ 归属待定 | 归 graph 还是 compute?与序列化一起在 scope 复核时拍 |

## 六、一句话排期建议

**sched 支柱已收口(2026-07-05)**:C2 带超时 io + C4 优雅关闭双双落地,sched-01~11 全 ✅(仅 sched-07 结构化取消树留待 HTTP fan-out 消费者驱动,非卡住)。调研关键收获:优雅关闭只需平信号广播,**与 C3 取消树解耦**,不必先做树。**下一步:io-08(C9 TLS)**——前置全齐(不依赖 C2/C3/C4),北极星① 公网服务器刚需,借 BoringSSL/OpenSSL shim 套现有 TcpStream。之后 io-10(C11 HTTP/1.1,超时+DNS 已就位、等 TLS),它会驱动 sched-07 取消树。**compute 支柱**(0 行、前置就绪、北极星① 算力本体)作为独立大阶段并行推进——它是最大剩余愿景块。A 类广度延期按需穿插,simd-03 硬件阻塞除外。
