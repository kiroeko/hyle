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
| ~~platform-02b~~ | ~~virtual_memory~~ | ✅ **完成(2026-07-17)** | 见 [platform-02b-virtual-memory.md](./platform-02b-virtual-memory.md)。**下游 mem-08 ✅** |
| ~~mem-08~~ | ~~fiber_stack guard+分级+池~~ | ✅ **完成(2026-07-17)** | 见 [mem-08-fiber-stack.md](./mem-08-fiber-stack.md)。`Fiber` 默认 guarded |
| ~~mem-06a~~ | ~~stats 下溢 + 泄漏追踪~~ | ✅ **完成(2026-07-17)** | 见 [mem-06a-stats.md](./mem-06a-stats.md) |
| ~~mem-06b~~ | ~~大页/THP hint + HugeRegion~~ | ✅ **完成(2026-07-17)** | 见 [mem-06b-hugepage.md](./mem-06b-hugepage.md) |
| ~~mem-06c~~ | ~~NUMA preferred + HugeRegion node~~ | ✅ **完成(2026-07-17)** | 见 [mem-06c-numa.md](./mem-06c-numa.md) |
| ~~mem-06~~ | ~~NUMA+大页+stats~~ | ✅ **切片全收(06a/b/c)** | Preferred/Bind/Interleave + 1GiB API + 大 Arena HugeRegion 已在库内；hwloc 全机编排非 mem 义务 |
| ~~simd-03 尾 NEON/AVX512 真机跑~~ | ~~缺硬件~~ | ✅ **已解(2026-07-09)** | NEON=mac 真机(simd-07 起);AVX512=simd-09 用 **Intel SDE 9.48.0**(WSL /root,`sde64 -future`)真跑——vec_test avx512 变体 173/173 全 op 对拍 + math/sort/bitpack tier 探针确认 dispatch 真选 kAvx512。四 ISA 全真跑,不再硬阻塞。**教训:"无硬件→理由化不验"是陷阱,SDE 是 AVX-512 无机时的真验闸门** |
| bench-02 | 对标 SOTA 套(absl/F14/moody/oneTBB/highway/marl) | 🟢 就绪 | 分散已做(各模块审查时对标过);整合成套可缓 |
| [simd-19](./simd-19-partial-vector-repr.md) | 定宽后端 partial-vector 表示(部分标签用窄 D 寄存器) | 🟢 就绪(有设计) | NEON perf 收口留的唯一"可修"真输 `PromoteMaskTo i16→i32` 0.575:定宽后端把每个 `Vec<T,N>` 当满宽寄存器,partial tag 满宽物化再掰回;scalable 已用 `kPow2`+`vl` 解,定宽待跟上。方案 A 根治(翻倍实现面)/ B 热点窄手术/ C 不做,建议 B。**COLD op(~0.5ns 绝对),不阻塞北极星**,按"只能平不能输"排期 |
| ~~[simd-12](./simd-12-vendor-displacement.md)~~ | ~~自研超过部分Highway/xsimd + 确认小数组调用税~~ | ✅ **完成(2026-07-11)** | u16 11/13正确SIMD核取代scalar fallback；SoA csin/ccos自研编排在AVX2胜xsimd、NEON胜原scalar；small-K自研候选未稳定赢std而撤销，整数/KV `k<=16`采用实测更快的std核。三套scaling bench、独立oracle、特殊值/alias/ASan闭环。 |
| ~~[simd-13](./simd-13-vendor-displacement-2.md)~~ | ~~继续竞争直接借用的 complex SIMD 核~~ | ✅ **完成(2026-07-11)** | HYLE SoA `ctanh/cpow` 在 AVX2/NEON 上架：常规 lane 分别消除复三角/双曲重复计算和 complex-pow 的重复 `log/exp`，全域异常 lane 保留 `std::complex`。Windows、WSL 与 Apple 真机统计证明常规或 large 输入无确认回退，Windows/WSL有明确随机胜利；`cexp/clog/csqrt`、VQSort 和其余 bit-pack 宽度保留供应商。P4 已由[`simd-14`](./simd-14-fixed-width-p4.md)完成，下一步仅P5。 |
| ~~simd-08 尾 → simd-09~~ | best-of 审计 + 数学函数/算法级能力 | ✅ **simd-09 全完成(2026-07-09)** | 线① 11 数学函数 best-of(Highway 9/xsimd 2)+ 线② 6 件(intdiv/cmp128/RNG 自实现 + vqsort/bit_pack/KV-top-k vendor 薄封装)。四 ISA + SDE + mac 全真跑。详见 simd-09 里程碑 |
| ~~simd SVE/RVV scalable 后端~~ | ~~原判"未来项缓实现"~~ | ✅ **scalable P0-P4 + fixed P3完成** | sizeless native type由后端traits承载；共享层以`Rebind+kPow2+cap`表达partial tag与lane-width rebind。P0解锁泛型complex/128-cmp/Newton/rotate，P1补scalable swizzle/interleaved I/O，P2补Promote/Truncate/Demote2/widen-dot/Sums/ReduceMul/Q15/SatAbs-Neg，P3补24个定宽Masked*公开名并完成性能闸，P4直接实例化FMA/magnitude/even-odd widen/bf16/mod，P5平凡合成+mask前缀(simd-15)。SVE 19 tests×4 VL、RVV 18 tests×4 VLEN全绿；主闭集P0-P5软件面收口，剩硬件门控(qemu≥9.1)，见`simd-gap-inventory.md` |
| ~~simd native-intrinsic 性能档~~ | ~~gather/compress 现 store+reload → native vpgather/vpcompress~~ | ✅ **已做+实测(2026-07-09,#66/#69)** | avx2 GatherIndex→`_mm256_i32gather`(**本机实测 1.31x 赢,保留**);avx512 Compress/Expand/CompressStore→`vpcompress/vpexpand`、GatherIndex→`vpgather`(SDE 正确性验,**性能待真机**——SDE 是功能模拟器,复杂指令模拟开销高,perf 不可信)。留 `GatherIndexEmulated/CompressEmulated` store+reload 参照供 A/B bench。`gather_bench.cc` 逐 ISA 对拍,performance.md 已记数据+SDE 工具边界。swizzle 全是硬件 shuffle 直通(无 emulation),不在此列 |
| ~~build-asan hwy::Abort 链接~~ | ~~undefined reference to hwy::Abort~~ | ✅ **已修(2026-07-09,0a64455)** | 根因:Highway 借核经 HWY_DASSERT 调 hwy::Abort,Release -O2 DCE 掉、ASan -O2-g 没 DCE→残留未定义(HYLE 从不编 Highway abort.cc)。修:把 `third_party/highway/hwy/abort.cc`(baseline C++)编进 hyle_simd 源列表→所有构建配置都链干净。build-asan 整库 ASan 实测 sse4/avx2 173/173 零错误(不再需 standalone 绕过) |
| ~~simd op-parity 自造件 vs Highway 全平~~ | ~~担忧"手写 ~228 op 性能不如抄 Highway"~~ | ✅ **可真机 ISA 已收口** | 同 TU/ISA + `bench-stat -n15`。AVX2/SSE4已追平历史9个store+reload/std慢op；Apple-Silicon NEON累计测71个op（含P3 24/24），逐宽修掉u8/i16/i32/i64 Compress/Expand回退，并在P3揪出`MaskedMulComplexConjAdd`慢18.5%，改FMA友好直接公式后0.204 vs 0.203 ns/op统计打平。可真机已测项全部≥Highway统计打平；P4 的Windows AVX2四条热路径三组独立N=15均TIE，Apple NEON三组均复现`Mod/MaskedModOr` 1.073x/1.058x WIN、widen multiply TIE。AVX512 SDE与SVE/RVV QEMU只作正确性证据。详见benchmarks-simd.md |

> ### best-of 判据(2026-07-09 三方审计定,固化防再犯)
> "三者都有就选最快"只在**三条件同时成立**时有意义:① op 有**真实算法内容**(多项式逼近 / 多指令组合 / 软件 emulation),② 三家实现**有差异**,③ **不贴单一硬件指令**(否则三家编译成同一条,比了也一样)。
> **⚠️ best-of 的 correctness 闸按公开契约判断**：同一契约内先过 libm/oracle，再比速度；精度不达标或契约内错误的实现没资格入选。若完整语义和显著加速不可兼得，则提供显式双入口而非替用户静默取舍：`hypot_f32_batch` 借 Highway exponent scaling 保持 `std::hypot` 全域，`hypot_in_range_f32_batch` 借 xsimd direct formula，在文档限定输入域内约快2倍且不加逐 lane 检查。同理，xsimd lgamma lane-mixing 与 Highway Pack16<11/13> 是各自契约内错误，只能退回正确实现——**借不等于信**。按此分类 simd op:
> - **无 best-of 空间(最多)**:纯 intrinsic 直通——add/sub/min/max/sqrt/BitCast/Mask 域/swizzle 硬件 shuffle/饱和/移位/位计数。三家同一条指令。
> - **有算法但借库借不到便宜**:gather/scatter/compress/expand 的 store+reload emulation——**Highway/xsimd 在 SSE4/NEON 上也是 store+reload**(它们也无硬件指令),借=借同款 emulation。真正加速是**自己上 AVX2/AVX512 native**(vpcompress/vpgather),那是**自造 profiling 优化,不是 best-of**。
> - **数学函数=best-of 借库主场**:多项式逼近算法最重、最易错、三家投入最大、不贴单一指令——"借难且已解决的"最典型。simd-06 只做了 exp/log/tanh(borrow 主场却只覆盖三个),sin/cos/erf/pow/exp2/log2/atan… 是同类金矿 → simd-09 ①。
> - **少数结构 op 有真 best-of/超越空间**:Complex(HYLE 自组合 vs xsimd `batch<complex>`+`fmas`)、reduce(shuffle-cascade vs 蝶形 transpose);rsqrt/rcp 三家都无 Newton 精修 → **HYLE 加精修可反超两家**(自造超越,非借)。
> - **算法级能力=借库补缺**(非 best-of,只 Highway 有):vqsort 向量排序 / 向量 RNG / intdiv 向量整除 / bit_pack / Lt128-Min128 → simd-09 ②。
| **compute-06 边角项(按需备选,非欠账)** | #19 push_descriptor(实测 update 便宜,收益有限)、#22 timeline semaphore(多序列依赖才需)、#24 批量 createComputePipelines(kompute 自留 TODO)、#23 多队列 ownership transfer(上 async compute 才需)、Image 路径(做 image kernel 才核)、红队 E(event marker cb 长命 stream 累积)、红队 G(build 部分失败无回滚)。#10 coherent flush 存疑已随 #6 解决 | 🟢 就绪 | **明确消费者驱动**:各有触发条件,无当前消费者→登记不做,别散落核对表/memory |

> container-06c(模糊路径匹配)**2026-07-05 ✅ 完成**,已从本表移除(抄 fzf V1 两趟算法,benchmark 证 O(N),container 160/160)。
> **compute-06 内部欠账(2026-07-08 登记)**:compute-06 按验收(三平台 vecadd+合批+co_await 全绿)已 ✅,但 kompute 25 点核对表"应做(抄/改)"21 项里做了 12、欠 9。**里程碑"明确不做"表(compute-06-vulkan-spirv.md:191)已接住多队列/运行时GLSL/Image/spirv-opt(B类消费者驱动,合理等);上面两行接住剩下无归属的**——① 是正确性硬伤(基础库义务,必做)、② 是性能收尾。避免这些散落在核对表/memory 里无排期。
> **simd-09 线② 全 5 件完成(2026-07-09)**:通用原语 intdiv/cmp128/RNG **自实现泛型层**;重型件 vqsort/bit_pack **vendor 薄封装 Highway**(用户"vqsort 先做、bit_pack 也要,不能用 Highway 么"→修正了之前"无 flat 入口无法薄借"的过度结论:公开层吃裸指针不 include highway、借核 TU 内部用 Highway D-tag,与数学核借法同构;VQSortStatic/-DVQSORT_ONLY_STATIC 与 Pack<kBits> 都 header-only)。x86 全绿(MSVC+WSL GCC)。**借核过对拍闸门揪出 Highway vendor bug**:Pack16<11>/<13> 满位宽 round-trip 失败(三处独立探针坐实算法 bug),回退 HYLE bitstream codec 绕过,见 docs/debug/highway-pack16-roundtrip-hunt.md。剩:NEON 待 mac(#63)、K32V32 top-k 键值对留消费驱动。

**A 类结论**:全部"就绪但低优先级",不阻塞北极星。simd-03 硬件硬阻塞已解。**platform-02b + mem-08 ✅**——VM 原语与 fiber 栈 guard/分级/池已接线。剩余 A 类(mem-06/bench-02)按需。**compute-06 收尾②性能收尾 2026-07-08 全清**——核对表 25 项该做的(#25 计时/#11-12 barrier/#1 cache 落盘/#6 HOST_CACHED/#2-3 spec constants/tools+load_spirv/#14-16 record-replay)全完成=compute-06b/c/d,记录见 compute-06 里程碑状态行;仅剩消费者驱动的边角项(上表,无当前消费者按需)。(收尾① staging 亦已完成,均从本表移除。)

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
| **compute** | compute-01~07 ✅ / 09 ✅(step-1+2)/ 08 | sched-04 ✅ | 🟢 **主干+CPU 后端+算子面扩展(含 FFN 权重常驻)就绪** | 主干 01~06:device/stream/event(✅)→ device buffer → GPU completion→fiber → kernel 启动 → ggml_bridge → Vulkan。**07 CPU 算子后端 ✅**:compute-04 Kernel 抽象的 CPU 后端,host_ops 接 hyle simd(接 simd 非 ggml,与 05 正交)。**09 算子面扩展**:step-1 ✅(2026-07-09)通用图执行器 + rms_norm/silu/add/mul(纠正"消费者驱动才加算子"误用——推理链本身是消费者);**step-2 ✅(2026-07-10)**=FFN block + 权重常驻(GgmlBackend 无状态→FfnBlock 持独立 weight buffer `ggml_backend_alloc_ctx_tensors_from_buft`,构造拷入一次,forward 只喂激活;WSL GPU+CPU 21/21 + MSVC 9/9,红队 3 项);**step-3 backlog(#102)**=attention 半 block(rope/softmax/permute,形状杂耍风险高,权重常驻机制可复用)。**剩 08 HIP/Metal**(优先借 ggml,硬件阻塞)。借 ggml 不自造算子;hyle simd 是 07 消费者。**异构混合极致性能的头牌本体**,北极星① 推理引擎的算力侧 |
| **graph** | graph-01 ✅ 全 + graph-02(step-1 ✅ / step-2·3 待)| sched-03 ✅ + compute(kernel) ✅ + simd ✅ | 🟢 **graph-01 完成,graph-02 step-1 完成**(设计 [`graph-design.md`](../architecture/graph-design.md) + [`graph-02-design.md`](../architecture/graph-02-design.md))| 北极星② CAD/BIM/CAE 驱动:脏标记增量重算(graph-01 全完成:串行核心+并行 fork-join+demand+去中心化 wavefront,五轮红队/校准)+ 数据并行层(graph-02)。**graph-02 step-1 ✅(2026-07-10)**:`parallel_for`/`parallel_reduce`(递归二分 fork-join+grainsize+auto,借 oneTBB 设计不借运行时,Body 收 subrange 接 simd),四配置 14/14 + 红队(揪 kAutoGrain sentinel hang)。**下一个 = graph-02 step-2**(parallel_scan 两趟 + reduce_deterministic)/ step-3(parallel_sort 借 simd vqsort) |

**C 类结论**:compute **已破土(compute-01 ✅)**——它是北极星① 的算力支柱本体,**排期上是一个大阶段**(主干 01~06 借 ggml ~5-8k + 缺口 07 CPU 后端 / 08 HIP·Metal)。**缺口补登(2026-07-06)**:07 CPU 算子后端(scope §CPU fallback + roadmap 验收"CPU+GPU 对拍"要求的真后端,非 compute-01 的 host inline 兜底)、08 AMD·Apple 后端(scope 后端矩阵列了但主干未覆盖,借 ggml-hip/metal 优先)。graph 依赖 compute 的 kernel,排在 compute 之后。

> **graph 后续能力(2026-07-10 taskflow 校准登记,等消费者驱动,非投机)**:① **动态/条件任务**(taskflow subflow/condition task):CAE 定点迭代求解/自适应网格靠它表达,HYLE 静态 DAG(插入序=拓扑序、无环)现表达不了——CAE 迭代求解器消费者到位时做,可能要放松 DAG 约束。② **pipeline/data_pipeline**(有序阶段流水并行):taskflow 有、HYLE 零覆盖;对应北极星① 推理 token 流水 + CAE 流式求解,是独立并行范式(候选 graph-03)。③ wavefront **continuation cache**(完成 task 直接跑下一就绪后继零入队):sched 未暴露"当前 fiber 转跑另一 fiber",评估后按需。校准全账见 memory `project_graph_taskflow_calibration`。

## 五、收口层待决(需你拍板,非纯排期)

| 里程碑 | 状态 | 待决 |
|---|---|---|
| close-01 C-ABI shim | 📋 承诺锁定待排期 | 各模块稳定后做;发布给 FFI 的唯一稳定面 |
| close-02 serialize | ❓ scope 待澄清 | 借 flatbuffers/capnproto builder+reader 当原语 vs 全推 L1?JSON 借 simdjson 已定 |
| close-03 FFI 绑定 | 占位 | 依赖 close-01 |
| graph-02 数据并行层 | ✅ **归属已定(2026-07-10)** | **归 graph**(graph-design.md §2):compute 铁律把铺多核推给 graph-02 fork-join,归 compute 会破其"不起线程池"边界;消费者与 graph-01 同源。序列化(close-02)仍待 scope 复核 |

## 六、一句话排期建议

**北极星① 服务器骨架已完整(2026-07-05)**:C1 并发 io / C2 超时 / C4 优雅关闭 / C9 TLS / C10 DNS·signal / **C11 HTTP/1.1 全部落地**,sched 支柱收口(01~11 全 ✅ + 三复杂度优化)。**serving 侧基本齐了**——从 socket 到 HTTP 请求/响应、超时、TLS、优雅关闭、请求 fan-out 取消,一条 co_await 链跑通,三后端(io_uring/IOCP/epoll)+ TSan 全绿。**下一大块 = compute 支柱**(0 行、前置 sched-04 就绪、北极星① 算力本体):device/stream 薄包 ggml-backend → device buffer → GPU completion→fiber → kernel 启动 → ggml_bridge → Vulkan,借 ggml 不自造算子。推理引擎 = serving(✅ io/HTTP)+ 算力(compute,待做),两条线合起来才是完整北极星①。**io-05 shmem / io-11 HTTP/2 / A 类广度**按需穿插,simd-03 硬件阻塞除外。
