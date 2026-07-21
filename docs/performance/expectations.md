# HYLE 性能预期与边界

> 本页是 [performance 总账](./README.md) 的预期/边界部分:AVX-512 上架状态、主表(时间/空间分列)、Amdahl 整体观、诚实边界、验证 gate。实测数据见同目录的 benchmarks-*.md。

## AVX-512 上架状态（①②③ 已接线并经 Intel SDE 执行对拍；性能待真机；④ pointer-bump 已完成）

> 逐模块性能审查(error→simd→math→dtype→platform→mem→time)后，把“已赢或追平 SOTA、但本机硬件够不到、或仍差一条指令”的优化显式记账于此。本开发机 i9-13900K 的 P-core 不提供 AVX-512；①②③ 已完成代码、dispatch、CMake tier、双平台编译/objdump，并由 Intel SDE 通过正确性与派发执行门。SDE 不模拟真实硬件时序，性能结论仍待 AVX-512 真机 benchmark。④ 与硬件无关，已完成双平台验证。

| # | 方向 | 状态(2026-07 已上架) | 验真证据 | 性能边界 |
|---|---|---|---|---|
| ① | **dtype `f32→bf16` native `VCVTNEPS2BF16`** | ✅上架:`cvt_avx512bf16.cc`(`_mm512_cvtneps_pbh` 16-wide 硬件 RNE,把 AVX2 ~7-op RNE 折成单条),`dtype_dispatch` f32→bf16 优先 `avx512bf16` 档(`cpu_features().avx512bf16` 门控)回退 AVX2/scalar;新 `AVX512_BF16` CMake tier | 双工具链编译+指令验:GCC14 `vcvtneps2bf16`(EVEX.512)、MSVC `/arch:AVX512` `vcvtneps2bf16 ymm,zmmword`。测试 `F32ToBf16Avx512MatchesScalar`(NaN equal-by-class、finite bit-exact)本机正确 SKIP | Intel SDE 正确性门已通过；吞吐/延迟仍待 AVX-512-BF16 真机 |
| ② | **simd ops AVX-512 派发档** | ✅已接线(此前里程碑):构建生成共享 `kernel_impl.h` 的 AVX-512 variant + `dispatch_ops.cc` 默认路由 AVX-512 host;本轮复核接线完整 | objdump:GCC 7 zmm/MSVC 33 zmm(含 `vfmadd213ps zmm`)。`ops_test` 宽度无关对拍 oracle，`runtime().ops.isa` 在对应机器报告 kAvx512 | Intel SDE `vec_test` 169/169 且 tier=kAvx512；性能仍待真机 |
| ③ | **math exp/log/tanh 512b 变体** | ✅上架:`transcendental_{xsimd,highway}_avx512.cc`(借同核 512b/16-lane),`math_dispatch` 加 avx512 档优先(exp/tanh→xsimd-512、log→highway-512)回退 AVX2;新 `hyle_set_avx512_math_isa`(全 AVX-512 组+cd) | objdump:GCC xsimd512 92zmm/highway512 43zmm、MSVC xsimd512 581zmm。`Math.*MatchesLibm` 对 libm 对拍,本机走 AVX2 路径过 | Intel SDE 正确性门已通过；性能仍待真机。**MSVC log-512 已排除**(Highway AVX3=`HWY_BROKEN_MSVC` 退 ymm,故 highway512 仅 GCC/Clang;MSVC log 留 avx2 highway,同速不假装512) |
| ④ | **Arena/StackAllocator pointer-style bump**(硬件无关) | ✅完成:offset-based(存 `base_`+`top_`/`offset_`,每 `allocate` 重算 `base+offset`→对齐→减回 offset)改 canonical `cur_`/`end_` 双指针纯指针 bump,直接对齐活指针。Marker 从 size_t 偏移改 byte*(API 不变) | objdump(GCC -O2 inlined):offset 版 ~15 指令含 3 load + base+offset 往返(对齐一次、overflow 记账再算一次);pointer 版 ~10 指令 2 load 无 +base/-base。**bench-stat 15 遍实测(Linux):Arena 0.68→0.34ns、Stack 0.68→0.34ns、mark/rewind 0.63→0.43ns=~2x**(gap 50%»spread ~12%,真赢非平局) | — |

**诚实定性**:①②③ 已从"留账非即做"转为**已上架+双平台编译 objdump 验真**(用户指令:写完接线才能在真机验证,否则永远落不了地);三项的 512-bit 执行正确性是 `存疑`,待 Intel SDE/真 AVX-512 机跑既有测试(全部宽度无关/对 oracle,真机一跑即自证,无需新码)。**关键发现**:Highway 在 MSVC 把整个 AVX3 标 `HWY_BROKEN`(已知 codegen bug),故 ③ 的 highway-512 log 仅 GCC/Clang 上是真 512,MSVC 退 ymm 已显式排除——不假装 512。④ 已完成(pointer-style bump);**实测意外——不是预期的"亚-ns 平局"而是 ~2x 真赢**(Arena/Stack 0.68→0.34ns、mark/rewind 0.63→0.43ns,15 遍 gap 50%»spread 12%):offset 版每 alloc 多一次 `base_` load + base±offset 往返算两遍(对齐一次、overflow 记账再算一次),objdump ~15 指令 vs pointer 版 ~10 指令;在 0.68ns(仅 ~3-4 指令成本)的量级上省 5 条指令直接砍半。**"亚-ns 天花板不值得做"的预判被实测推翻**——bump 成本本身太小,任何多余指令占比都放大。**没有其它"本机低成本高收益"项**——time/`monotonic_ns` 那种(频率特判换 1.48x)的机会,逐模块扫描后未再发现(④ 这个 2x 是本轮附带发现的第二个)。

## 主表：时间 / 空间分列

### 桶 A：drop-in 可比

| 组件 vs std | 时间（speedup + 复杂度）| 空间（vs std）|
|---|---|---|
| **flat hashmap** vs `unordered_map` | 查找/插入 **2-4×**；均摊 O(1) 同，但常数小（连续布局 cache 命中 + SIMD 16 路探测）| **通常更省**：无 node 的指针 + 每节点分配头；代价是留负载因子（有空槽 + 1 控制字节/槽）。最坏（刚扩容）峰值略高 |
| **专用分配器** 单线程 bump/pool vs new/delete | 单线程用 `StackAllocator`/`Arena` pointer bump 或 `GrowablePool`；多线程固定池须按合同分开：`SharedPool` 是硬容量，`SharedPoolCached` 以 TLS 128 + magazine 32 换软容量吞吐。最终对标数字见 mem 证书，不把 cached 快路外推给硬池 | **更费**（以空间换时间）：预留 slab、block 尾部与 cached TLS 窗口；硬池不隐藏已 free block |
| **通用分配器** mimalloc vs malloc | 多线程 **1.5-3×** 吞吐、p99 更好；单线程仅 10-30% | 碎片通常 ≤ glibc；大页会增驻留 |
| **SmallVector** vs `vector` | 小 N 省 malloc（大赢）；大 N 持平 | **更费**（以空间换时间）：inline 存 N 元素，sizeof 变大（没用满也占）；超 N 退化到堆 |
| **intrusive list/map** vs `list`/`map` | O(1) 摘除、无分配；查找同 | **更省**：hook 内嵌节点，无单独分配 + 无 node 包装 |
| `vector`/`string` 本身 | **≈ 持平** | ≈ 持平。std 这俩已很好，不白捡 |

### 桶 B：std 没有的能力（"倍数"仅作量级参考）

| 组件 vs std 最接近物 | 时间 | 空间 |
|---|---|---|
| **fiber 切换** vs `std::thread` | **~100×**（10-20ns vs 1-2μs）；仅海量任务兑现 | **更费**（以空间换时间）：每 fiber 独立栈（64KB-1MB × M）；vs 无栈协程只占 frame。故 mem/ 做栈池化 + 分级栈大小 |
| **fiber-aware mutex** vs `std::mutex` | 高竞争吞吐高数倍（不阻塞 OS 线程）；无竞争持平 | ≈ 持平（等待队列用 intrusive，零额外分配）|
| **async IO** io_uring/proactor vs 阻塞 IO | 并发连接 **10-100×**；批量提交省 syscall | **远省**：SQ/CQ ring 固定开销 vs thread-per-conn 的每线程栈 |
| **SIMD 数值核** vs 标量 | **4-16×**（AVX2=8/AVX512=16 宽）；vs 自动向量化仅 1.5-4×（transcendental 这类是大头）| 轻微增：对齐 padding + dtype 转换 scratch |
| **device buffer**(compute/ 自有) vs host/device 双份镜像 | 异构(独显)本就要显式 copy/drawcall 过 PCIe——无"零拷贝"可言,优化点是**批量提交 + 复用 buffer 省往返**;统一内存(DGX/Mac)host/device 同物理 RAM,**本就无互传概念**,不存在"省一次 DMA"的红利 | **更省**:统一内存下不需 host+device 双份镜像(物理同址) |

## 整体系统层面（Amdahl，别期待"全面快 N 倍"）

局部再快也受非热路径拖累：

- **瓶颈不在这些点** → 整体可能仅 **1.2×**。
- **并发 / IO 密集** → **10×+**。
- **架构连贯性红利**（不打架的双调度器、统一内存无双份镜像）不体现为微基准倍数，而是真实混合负载的系统级延迟/吞吐——Seastar/ScyllaDB 那种 shard-per-core 重写报过 **~2× 系统吞吐**，是"架构做对"的现实数字。

## 诚实边界

**很多情况下 HYLE 对 std 几乎零提升**：代码本就 cache 友好、分配少、单线程、低并发时，std 对常见场景真的好，HYLE 白搭。**赢点恰好集中在 AI infra 热路径**——海量并发（fiber）、高 IOPS（io_uring）、hash 密集（flat map）、分配 churn（arena）、数值核（SIMD）。这不是巧合，HYLE 就是为这个 workload 设计的。

## 一句话

不是"处处比 std 快"。是：**hashmap 稳赢 2-4×、专用分配器对的模式 5-10×、SIMD 数值核 4-16×、并发/IO 是数量级（仅高负载兑现），平凡单线程代码 ≈ 持平**；且 arena/fiber/SmallVector 的时间优势**以空间为代价**。HYLE 的价值是把 AI infra 那几条热路径顶到 std 摸不到的天花板，并让它们在同一套底座上不互相拖累。

## 验证（不是空头支票）

每个数字到 [`../architecture/roadmap.md`](../architecture/roadmap.md) 的 benchmark gate 实测对标：
- P2 mem：对标裸 mimalloc 不退化
- P3 container：hashmap 对标 absl/F14；simd 数学函数 ULP 精度
- P4 sched：百万次 fiber 切换基准
- P5 io：echo/吞吐；compute：CPU+GPU 后端对拍

是骡子是马，gate 上遛。
