# HYLE 性能预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 实测铁证（持续回填,真跑非推算）

> 环境:Linux x86_64 GCC14.2 Release,nanobench。100k uint64 键。`bench/container/`。

### SwissTable hashmap

| 操作 | vs `std::unordered_map` | vs `absl::flat_hash_map`(**SOTA 标尺**) | 结论 |
|---|---|---|---|
| find-hit | **2.42x** 赢 | **~1.10x 赢**(12 遍 avg 110.1%;prefetch+mix64+H1,曾 0.76x) | **反超 SOTA** |
| find-miss | **2.26x** 赢 | **≈1.00x 统计平局**(12 遍 avg 97.5%,spread 6.3% > 差距,2/12 遍赢) | **追平 SOTA** |
| insert | **5.3x** 赢 | **~1.40x 赢** | **超 SOTA** |
| erase | **3.5x** 赢 | **~1.39x 赢** | **超 SOTA** |

> **2026-06-28 原生 ext4 复测(每步跑 3 遍,非噪声)——两刀优化,学自 absl**:
> 1. **probe 预取**(`HYLE_PREFETCH` slot,镜像 absl `find_large` 的 `PrefetchToLocalCache`):find-hit 0.76x→**0.88x**(slot load 藏进 ctrl SIMD match);find-miss 不变(miss 不碰 slot)。
> 2. **mix64 换 absl 式 128 位乘折叠**(`(x^k)*k → hi^lo`,单乘短依赖链 vs splitmix64 的 2 串行乘):find-hit **0.88x→1.10x 反超**、find-miss **0.78x→0.92x**(hash 占 miss 成本大头,短链直接砍)。代价:整数 hash 从双射(零碰撞)降为统计级(~1/2⁶⁴),correctness 由 eq_ 兜底;insert/erase **未退化**(1.35x/1.32x)证明碰撞质量没崩。
> - 双平台真编真测:Linux 133/133(含雪崩/分布质量测试)+ Win MSVC `_umul128` 分支编译干净 + 测试过。`__extension__` 消 `__int128` 的 -Wpedantic。
> 3. **match_empty 加 SSSE3 快路径**(`_mm_sign_epi8(c,c)` 1 指令 vs SSE2 重建 kEmpty 广播 4 指令,`#ifdef __SSSE3__` gate + SSE2 兜底):**绝对增益/与 absl ISA 对等**,但**未缩 find-miss 相对差距**——`-mssse3` 下 absl 也用 `psignb`,两边同步提速,find-miss 仍 ~0.89x。**证伪了"match_empty 是 find-miss 差距来源"**。双 ISA 输出穷举验证一致(psignb 确认)+ SSSE3 激活下 134/134。注:默认 SSE2 ship 构建走兜底(no-op),仅 `-march`/`-mssse3` 生效。**[已登记]最终的"编译/运行期自动辨识硬件规格选 ISA 变体"与 simd 支柱统一做**(conventions §10 baseline+dispatch);现阶段 gated 着 + 文档留痕即可,认真部署可先 `-march=native`/提基线 `x86-64-v2`。
> - **find-miss 剩 ~10% 已排除三因**:非 hash(mix64 对齐)、非 prefetch(miss 不碰 slot)、非 match_empty(同 ISA 仍差)。**根源在别处**(probe_seq codegen / H1 的 `>>7` / BitMask 迭代),需逐指令 profile,留作 profile 引导任务。
> 4. **H1=identity**(objdump 逐指令对比发现:absl 用 hash 低位直接当 index、top 7 位当 H2,省掉我们的 `hash>>7` 一条移位):find/insert/erase 内循环各省 1 指令。find-miss **0.89x→~0.96x**、insert/erase 升到 ~1.40x。正确性:mix64 雪崩后低位(index)与高 7 位(H2)独立,探测链不变;SSE2 基线 134/134 + Win MSVC 通过。
> - **12 遍统计收口(回答"4% 差在哪")**:find-hit avg **110.1%**(min 102.5/max 119.4)、find-miss avg **97.5%**(min 94.9/max 101.2,**spread 6.3% > 差距 2.5%,12 遍有 2 遍赢**)、insert **146.6%**、erase **140.0%**。**find-miss 是统计平局**——那"4%"是测量噪声(微基准 ~2-4ns,find-hit 自身 spread 都 16.9%),非结构缺陷,无单点杠杆。**四项全部 ≥ 或追平 absl,SOTA 反例消除。**

**这就是"对标 SOTA"的意义**:只对标 std 会得出"全面 2-5x 碾压"的虚假安心;对标 absl 才暴露 **find 有 ~25% 真实差距**。**已排除三个错误候选**(读码查实):① `eq_` 对 uint64 在 -O3 完全内联成裸 `==`,**无间接可去**;② `Hash<uint64>` 已是内联 `mix64`(splitmix64),**非 XXH3**,hash 不慢;③ **控制字节尾部镜像已实现**(`swiss_table.cc` core_set_ctrl 镜像前 kGroupWidth 字节 + 分配 `cap+kGroupWidth` + memset 尾),group near-end 加载已无回绕掩码。**find 循环结构已几乎等价 absl**——25% 来源**无法从源码结构看出**,需 **profile(perf/objdump 对比两个 find 内循环 codegen)** 定位真因;唯一已知结构差异是 absl 在 probe 起点 **预取槽数组**(100k 表常驻 cache 时收益小,待测)。**结论:这是一个需要 profile-guided 的真优化,非猜改**——留作下一步 container 性能待办。insert/erase 反超 absl 证明设计本身到位。

### hash 哈希（`hash_bytes`/`mix64` vs 裸 XXH3 + wyhash + std，`hash_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。三组:HYLE / 裸 XXH3+wyhash(SOTA) / std::hash。XXH3 vendored 无条件,wyhash gated `-DHYLE_BENCH_WYHASH`。此前 hash 只在 SwissTable 内部跑过,从未单独对标(README 宣称 XXH3 默认 hasher)。

| 字节哈希 | HYLE hash_bytes | 裸 XXH3(SOTA) | wyhash | std::hash | 结论 |
|---|---|---|---|---|---|
| 8B | 0.34 ns | **0.18**(HYLE 慢 1.85x) | 0.27 | 0.26 | 小输入 HYLE 全场最慢 |
| 16B | 0.15 ns | 0.10(慢 1.5x) | 0.14 | 0.29 | 赢 std,输裸 XXH3 |
| 64B | 0.05 ns | 0.04(慢 1.38x) | 0.05 | 0.44 | 赢 std 大,输裸 XXH3 |
| 256B | 0.08 ns | 0.08(平) | **0.03**(HYLE 输 3x) | 0.62 | 中串 wyhash 碾 XXH3 |

| 整数哈希(4096 key) | HYLE mix64 | wyhash64 | std::hash | 结论 |
|---|---|---|---|---|
| ns/op | **0.45** | 0.57(HYLE 赢 1.27x) | 1.52(赢 3.4x) | ✅ mix64 真好 |

**结论(一个包装税 + 一个 SOTA 观察 + 一个真赢)**:① **`hash_bytes` 比裸 XXH3 慢 1.05-1.85x,8B 小输入最重**:它是导出函数(`HYLE_CONTAINER_API` 不可内联,藏 7490 行 xxhash.h),裸 XXH3 用 `XXH_INLINE_ALL` 全内联——同 allocator vtable / dtype batch 的**导出包装税**,小工作量下调用开销占比大。② **wyhash 在 256B 比 XXH3 快 3x**(SOTA 观察:中等串 wyhash > XXH3;HYLE 选 XXH3 是质量/分布而非纯速度,设计权衡非缺陷)。③ **`mix64` 整数哈希真赢**:赢 wyhash64 1.27x、赢 std 3.4x(内联无包装税)——这正是 SwissTable 整数键的热路径,印证 find/insert 反超 absl 的哈希基础扎实。



> 环境:WSL Ubuntu24.04 原生 ext4(非 9p)、GCC14.2 Release、nanobench。4096 个 f32(L1 常驻,隔离计算吞吐非内存带宽)。`bench/simd/`。**Highway = SOTA 标尺**(同 -mavx2 静态 dispatch,与 HYLE avx2 同 ISA、同 kernel 形状,只差抽象层)。**数字 = 4 遍均值(2026-06-29 复测,沿 hashmap 那节的多遍统计纪律,非单遍)。**

| kernel | scalar(auto-vec SSE2) | HYLE sse4(128b) | HYLE avx2(256b+FMA) | Highway avx2(**SOTA**) | 结论 |
|---|---|---|---|---|---|
| **dot product** | 2.75 G op/s (1.0x) | 11.2 G (4.07x) | **12.4 G (4.5x)** | 11.2 G (4.07x) | **追平/微超 Highway**(~1.10x,4 遍均赢);4.5x over auto-vec |
| **saxpy** y=a*x+y | 17.3 G (1.0x) | 17.5 G (1.01x) | **25.1 G (1.45x)** | 20.7 G (1.20x) | avx2 **~1.21x 微超** Highway;saxpy 内存带宽受限故倍数低 |

**结论(回答"比 SOTA 有下降么")**:**没有——HYLE 向量后端追平且小幅微超 Highway**。dot 上 HYLE avx2 ~1.10x、saxpy ~1.21x,**4 遍复测 HYLE avx2 两 kernel 均稳定领先**(非单遍噪声);但领先幅度小(dot 0.08 ns/op 量级,两边内循环都编成同一条 `vfmadd231ps`,objdump 可证),诚实表述是**追平/微超 SOTA**,非"碾压"。compute-bound 的 dot 上 **4.5x over 自动向量化标量**,落在性能文档"SIMD 4-16x / vs auto-vec 1.5-4x"区间(dot 含 reduction,略超 4x 合理)。saxpy 是 3 流内存操作(读 x/读 y/写 y),标量 auto-vec 已近饱和带宽,故 SIMD 倍数天然低——AVX2 靠更宽 load 仍取 1.45x。**SSE4 saxpy ~1.0x(与标量持平)**诚实留账:128 位宽度下该 kernel 已带宽受限,SIMD 无加速空间,非 bug。
>
> **意义**:这验证了"自造精简抽象不牺牲性能"——HYLE 的 vec.h+后端(~340+360 行)在热路径上与 vendor 整个 Highway(单后端 ~3k 行)同速或微超。抽象层零成本(全内联到 intrinsic),没有为"可移植"付性能税。
>
> **范围诚实**:此对标只覆盖 **dot/saxpy 两个 compute/memory-bound 数值核**(AI 热路径主力)。Highway 还有 ~200 op(swizzle/gather/scatter/compress/transcendental)+ 6 ISA,HYLE 现做 ~25 op + 2 ISA 实跑——**"追平"指已实现的核上追平,非功能对等**。广度补全(int 量化类型/更多 op/AVX512 启用)见 milestones simd 延期登记。**正确性**:Ne(NaN)/Neg(±0) 等边界经红队批量审计揪修(2026-06,见 milestones simd-02/03),已加 FloatEdgeCases 覆盖。

### simd 数学核（exp/log/tanh vs 标量 libm，`math_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,4096 f32(L1 常驻)。**SLEEF IS the transcendental SOTA**(HYLE 借其验证过的核,不手搓多项式),故对标 = SLEEF AVX2 8-wide vs 标量 `std::exp/log/tanh` 循环,测的是**向量化加速本身**。`active_math_variant()` 打印当前层。

| kernel | 标量 libm | HYLE(SLEEF AVX2) | 加速 | 结论 |
|---|---|---|---|---|
| **exp f32** | 2.04 ns | **0.31 ns** | **6.5x** | 超 8-wide 理论的多项式部分 |
| **log f32** | 2.71 ns | **0.85 ns** | **3.2x** | 落在 transcendental 区间 |
| **tanh f32** | 6.26 ns | **1.56 ns** | **4.0x** | — |

**核心诚实结论(两个真相,一好一须澄清)**:

1. **"SIMD 数值核 transcendental 4-16x"声明坐实——但仅 SLEEF opt-in**:启用 `-DHYLE_SIMD_SLEEF=ON` 后 exp 6.5x/log 3.2x/tanh 4.0x 真兑现(`active_math_variant()` 报 `AVX2 (SLEEF)`)。**默认构建无 SLEEF → exp/log/tanh 走标量 libm,加速 = 1x(零向量化)**。README/状态表已据实标"SLEEF 为 opt-in,默认标量"。

2. **真编暴露并修了一个 Windows 链接 bug**:SLEEF opt-in 路径**此前从未在 Windows 实际构建过**(零 bench = 零真编)。本次首编即撞:`sleef.h` 在 MSVC 下默认 `__declspec(dllimport)`,消费**静态** sleef.lib 必须定义 `SLEEF_STATIC_LIBS`,否则 `__imp_Sleef_*` 全部未解析。已在 `src/simd/CMakeLists.txt` 补该定义(跨平台无害)。**这正是"benchmark 覆盖 = 真编真测"价值的活例**:不真跑,这条 opt-in 路径会一直"声称可用"实则链接不过。

### simd dtype 转换（fp16/bf16 <-> f32，`dtype_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,4096 f32。三组:HYLE F16C 硬件 dispatch / HYLE scalar 参考 / Highway(gated SOTA)。`active_dtype_variant()` 报 `AVX2/F16C`(硬件路径真启用)。std 无 fp16 批量等价物。

| 转换 | HYLE F16C | HYLE scalar | Highway(SOTA) | 结论 |
|---|---|---|---|---|
| **f16→f32** | **0.02 ns** | 0.65(慢 26x) | 0.05 | **赢 Highway 1.9x** |
| **f32→f16** | **0.03 ns** | 2.05(慢 71x) | 0.05 | **赢 Highway 1.6x** |
| bf16→f32(无 HW) | 0.12 ns | **0.10**(scalar 反快 1.11x) | — | bf16 包装税,见下 |

**结论**:① **fp16 F16C 真兑现且超 SOTA**——比 scalar 26-71x,**还赢 Highway 1.6-1.9x**,"fp16 dtype(F16C)"声明坐实。② **bf16 批量导出函数(0.12ns)比内联 scalar(0.10ns)还慢 1.11x**:bf16 无 AVX2 硬件转换(header 诚实标 deferred,只 auto-vec 移位),导出 `bf16_to_f32_batch` 付不可内联调用开销却无硬件收益摊薄——同 allocator vtable / hash_bytes 的**导出包装税**。微小,但印证模式:无硬件加速的批量导出 op 不如直接内联 scalar。

### simd 逐 op 扩展（reduce_max / scale，`simd_bench.cc`）

> dot/saxpy 之外补两个 distinct pattern,验证 vec.h 抽象在非 FMA 模式下是否退化。环境同上,4 ISA 隔离 TU。

| op | scalar(auto-vec) | HYLE sse4 | HYLE avx2 | Highway(SOTA) | 结论 |
|---|---|---|---|---|---|
| **reduce_max**(非 sum 归约) | 0.74 ns | 0.17(4.3x) | **0.08(9.7x)** | 0.08 | **平 Highway** |
| **scale** y=a*x(纯乘) | 0.03 ns | 0.05 | **0.03(1.16x)** | 0.03 | **平 Highway**(带宽受限) |

**结论(干净,无欺骗)**:**4 个 pattern(dot/saxpy/reduce_max/scale)HYLE avx2 全平 Highway**——vec.h 抽象在 sum-归约之外的 max-归约、纯 element-wise map 上同样零成本,无隐藏退化。reduce_max 9.7x 赢 scalar;scale 是 1 读 1 乘 1 写的内存受限 op,scalar auto-vec(SSE2)已近饱和带宽,故 avx2 仅 ~1.16x(正常,非缺陷)。sse4 在 scale/saxpy 这类带宽受限 op ~1x(128b 与 scalar SSE2 同宽,无更宽 load 优势)。



| 项 | 实测(Linux GCC14) | 参考 |
|---|---|---|
| fiber 上下文切换(resume/yield 对) | **~21-23 ns**(本 session 复测 21.0ns) | fcontext 量级,= boost.context 原语本身(零封装开销) |
| spawn 2000 任务 + drain | vs `std::async` 快 144x(**弱基线**)→ **真 SOTA 见下 sched-05 节:vs marl 快 2.17x** | std::async 每任务起 OS 线程(稻草人);真标尺是 marl 调度器 |

> **注**:spawn 的权威数字在下方 sched-05 节(对标 marl,非 std::async)。此处保留 std::async 行仅作"赢最低门槛"的历史参照。

> sched bench(`bench/sched/scheduler_bench.cc`)**现已编入默认 build**(提交后解封)。
> 它曾被认为触发一个 P0 崩溃,但一轮穷尽调查(见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)
> + [`known-issues.md`](./known-issues.md) KI-1)**排除了 HYLE 逻辑缺陷**:TSan 无源码级
> race、MI_DEBUG=3 全量堆校验无损坏、原始崩溃器本体在 Linux 原生 + Windows MSVC 本机
> 重负载下 + 强制迁移 687 万次跨线程 free 均**不复现**。证据(SIGILL = 代码页问题 +
> 全部 instrumented 净 + 原始崩溃全在 9p)指向 **9p/WSL demand-paging 环境假象**,非
> HYLE 缺陷。fiber 切换 ~21-23ns 数值有效(mimalloc v2.3.2 + 本 session 复测一致);spawn 的权威对标见下 sched-05 节(vs marl 2.17x,非此处历史的 vs std::async)。

### sched-05 fiber 同步原语(`sync_bench.cc` + `sync_marl_bench.cc`)

> 环境:Linux x86_64 GCC14.2 Release(原生 ext4)+ Win MSVC19.50 Release,nanobench。**SOTA 标尺 = marl**(本设计收敛自的 fiber-sync 库;gated `-DHYLE_BENCH_MARL=ON`)。

| 项 | HYLE | SOTA(marl / std) | 结论 |
|---|---|---|---|
| **争用 Mutex**(8 任务/4 worker,lock+inc+unlock) | **30.7 ns** | marl 32.1 ns(=std::mutex) | ✅ **打平 marl**(±噪声,本轮略胜) |
| **无争用 Mutex**(多线程进程=真实场景) | **6.5 ns** | std::mutex 9.1 ns | ✅ **HYLE 快 1.40x**(单线程 microbench 的"2.9x 慢"是 glibc 假象,见下) |
| **spawn 2000 noop**(纯派发) | **141 ns/task** | marl 306 ns | ✅ **HYLE 快 2.17x**(对标真 SOTA,非 std::async) |
| Channel 收发(cap 16,1p1c) | **63 ns/item(15.9M/s)** | (marl 无 channel) | HYLE 自有 |
| **WaitGroup fork-join**(1024 任务) | **205 ns/批**(池化栈 179) | marl 306 ns | ✅ **反超 marl 1.49x**(池化 1.71x) |

**Mutex = futex(Drepper 3 态),不是 fiber-aware——这是学 marl 的 SOTA 决策**:最初把 Mutex 做成 fiber-aware(争用让出 fiber),实测**慢 marl 5.8x**(176ns vs 30ns)——短临界区让出 fiber(suspend+重入 ~100ns)远贵于就地 park worker 几 ns。改写成建在 `platform::wait_on_address`(futex/WaitOnAddress/ulock)上的 Drepper 3 态锁(无争用单 CAS、争用 park worker)后**追平 marl**。marl 同理:它的 mutex 就是 `std::mutex`(futex),只把 fiber-awareness 留给 ConditionVariable。**HYLE 对齐:Mutex=futex(本节),CV=fiber-aware 双模(地基),各司其职。** 双平台 TSan 8/8 零 race。

**无争用 Mutex "慢 std" 是 microbench 假象——已实测揭穿**:HYLE futex Mutex 与 std::mutex 算法同级(各 1 locked RMW)。单线程 bench 里 std::mutex 假性快到 2.2ns,因 glibc 用 `__libc_single_threaded` **跳过原子**;HYLE 不做单线程特化,恒付。**铁证**:加一个后台线程让进程变多线程(`sync_bench.cc` MT 变体),std::mutex 立刻 2.2→**9.1ns**(不能再跳原子),HYLE 纹丝不动 **6.5ns**——**HYLE 反超 std::mutex 1.40x**。在 mutex 唯一真正被用的多线程场景,HYLE 的 Drepper 3 态比 pthread_mutex_t 更轻、更快;marl mutex=std::mutex,故也赢 marl。**结论:不是性能问题,是单线程 microbench 被 glibc 优化骗了。**

**spawn vs SOTA:对标 marl(非 std::async 弱基线)HYLE 快 2.17x**:scheduler_bench 的"spawn 2000 vs std::async 144x"是赢 thread-per-task 稻草人。直接对标真调度器 marl(`sync_marl_bench.cc` noop spawn-2000):marl 306ns/task vs **HYLE 141ns/task = 2.17x 反超**。spawn/dispatch 是真 SOTA,非弱基线侥幸。

**WaitGroup fork-join 反超 marl 的真根因 = 调度器唤醒策略,不是栈池(实测纠错)**:fork-join 起初慢 marl **17x**(5.4µs vs 0.30µs)。第一反应怀疑"没 fiber 栈池",但**诊断推翻**:调高 worker 泊车前自旋后立刻反超,单上栈池只改善 ~5%。**真根因 = 调度器急泊车 + 每次入队无条件 unpark 的 futex 系统调用风暴**。**治本方案 = §2.3 ②③ conditional-notify(已落地)**:维护 `num_searching_` 原子计数,producer 入队后仅当无 worker 在搜才 unpark(否则搜索者会捡到),从源头消除风暴;lost-wakeup 安全靠 notify 与 worker 离开搜索的 **seq_cst StoreLoad Dekker** 握手(双平台 28/28 + TSan + NoLostWakeup 100× 零超时验证)。结果:fork-join **205-241 ns/批,反超 marl 1.49x**;栈池(`MemoryArena::as_allocator`)再加 ~13%→**1.71x**(次要增益)。**教训:17x 一度被错误归因到栈池(连架构审计 agent 都照旧文档复述),实为唤醒策略——"登记延后掩盖真瓶颈"的活例;先做的 spin-then-park 是缓解,conditional-notify 才是根治(去自旋空烧 + 去魔法常数 + 源头治本)。** CV 的 `waiters_` 门让 Channel/WaitGroup 的 notify 无 waiter 时零锁。

> **TSan × fcontext(本步另一修复)**:调度器唤醒优化让 fiber 跨 worker 迁移更频繁,撞出 TSan 对 fcontext 栈切换的盲区(SEGV in libtsan,非逻辑 bug——无 TSan 50/50 过)。已给 `fiber.cc` 加 `__tsan_switch_to_fiber` 注解(每次切换告知 TSan 当前逻辑上下文,迁移时按 resume 重新捕获 resumer 句柄),**release 零成本**(非 TSan 构建宏全展开为空)。修复后双平台 28/28 + TSan 28/28 零 race,lock-free 闸门恢复。




### lock-free 队列簇（`queue_sota_bench.cc` + crossbeam Rust 对照，task #9 / bench-02b）

> 环境:Win MSVC19.50 Release + Linux x86_64 GCC14.2 Release(原生 ext4),nanobench。cap=1024、50000 item/生产者。**SOTA 标尺**:moodycamel ConcurrentQueue(单头,gated `-DHYLE_BENCH_MOODYCAMEL`)、oneTBB concurrent_bounded_queue/concurrent_vector(gated `-DHYLE_BENCH_TBB`,apples-to-apples 有界)、crossbeam-deque(Rust,work-stealing 同族,`bench/rust/`)。**此前这簇零对标("覆盖低就默认 SOTA"),本节是真对标揪出的结果——赢/平/输都登记。**
>
> **对标依赖 `project-hyle/` sibling 布局**:gated 库默认在仓库 sibling(`../moodycamel-ConcurrentQueue`/`../oneTBB`),crossbeam Rust crate path 依赖 `../../../../crossbeam-deque`(本地 0.8.3 clone)。换机器时:C++ 侧 `-DHYLE_MOODYCAMEL_DIR=/path` 覆盖;Rust 侧改 Cargo.toml path 或换 `crossbeam-deque="0.8"` 走 crates.io。

| 场景 | HYLE | moodycamel(SOTA) | oneTBB | 结论 |
|---|---|---|---|---|
| **单线程 fill+drain**(BoundedQueue) | 7.2(Lin)/7.8(Win) | 7.7 / 9.4 | 12.6(Win) | **追平~微超 moody、赢 tbb 1.6x** |
| **MPMC 2P/2C**(BoundedQueue 单环) | 37.7(Lin)/41.8(Win) | 47.0 / 62.1 | 137(Win) | Lin **赢 moody 1.25x**、Win 打平;赢 tbb 远 |
| **MPMC 4P/4C**(BoundedQueue 单环) | 95(Lin)/99(Win) | 55 / 62 | 224(Win) | **输 moody 1.7x**(争用)、赢 tbb 2x |
| **MPMC 2P/2C**(**ShardedQueue**) | **18.1(Lin)/24.9(Win)** | 47.0 / 62.1 | — | **反超 moody 2.6x** |
| **MPMC 4P/4C**(**ShardedQueue**) | **21.0(Lin)/20.4(Win)** | 55 / 62 | — | **反超 moody 2.6-3.0x** |
| **work-steal owner LIFO**(WorkStealingDeque) | 2.8(Lin)/3.1(Win) | (无同物) | — | crossbeam LIFO 3.6(Win) 参照,HYLE 略快 |
| **work-steal owner FIFO**(WsFifoQueue) | 3.6(Lin)/3.9(Win) | — | — | crossbeam FIFO 4.4(Win) 参照,HYLE 略快 |
| **ConcurrentVector** push_back/read | 3.9/0.8(Lin) | — | 6.1/1.3(Win) | **赢 tbb push 1.3x / read 1.1x** |

**核心诚实结论 + 它驱动的真优化**:

1. **单环 BoundedQueue 在高竞争全局 MPMC(4P)输 moodycamel 1.7x——这是揪出的真"自以为 SOTA"**。诊断(cap 1024→64k 解背压)证实**根因=单 `enqueue_pos_`/`dequeue_pos_` 的 CAS 单点 serialize**(解压后差距反扩 1.47→1.73x),**非有界税**。moodycamel 用 per-producer 子队列(放弃严格全局 FIFO + 无界 + 消费者 O(producers) 扫描)换生产者侧零争用——在多生产者高吞吐场景合理。

2. **backoff(`util::cpu_relax` 加在 CAS 失败重试)**:零约束代价(FIFO+有界全留),**2P 反超 moody 1.55x**(Win 62→43);4P 无效(单点 serialize 是硬墙,pause 只错开不增吞吐)。

3. **新 `ShardedQueue`(N 个 Vyukov 子环,生产者 hash 到各自子环)根治**:消除单点争用,**4P 反超 moody 2.6-3.0x、2P 反超 2.6x,且线性扩展**(2P→4P 几乎不恶化:Lin 18→21、Win 25→20,而单环 38→95/42→99 急剧恶化)。**保留有界**(每子环有界),**放弃全局 FIFO**——这对其唯一生产用途(sched `global` 注入可运行 fiber 指针,work-stealing 本就无全局序)**是零代价**:约束服务用途,严格全局 FIFO 在此无用途支撑=可弃的假约束。需 FIFO 的场景仍用 BoundedQueue(单环)。双平台 6/6 测试 + **WSL TSan 零 race**(含 100k×4 MPMC 守恒)。**已接入 sched `global` 注入队列**(worker idx 作 push hint、per-worker pop cursor、外部注入 round-robin;28/28 sched 测试 + WSL TSan 零 race,fiber 切换 24.96ns 无退化)——价值兑现,不止 benchmark 数字。

> **诚实边界**:① ShardedQueue 的 ns/op 因太快采样不足,nanobench 标 unstable(err 6-11%),但反超幅度(2.6-3x)远超误差带,方向稳健。② **跨语言对照(crossbeam Rust)只能下"同量级/有竞争力"**——亚 ns、~1.2x 差落在 rustc-LLVM vs MSVC + 手写计时 vs nanobench 的噪声内,不下细比 verdict。③ 单环 vs moodycamel 是**有界严格-FIFO vs 无界弱序**两类设计,4P 的"输"是约束选择非实现缺陷(单环已是该约束下 SOTA,rigtorp/folly 同款)。④ work-stealing owner 路径单线程无争用(两侧同样),steal 路径未并发 owner=测无争用 steal。

### mem 分配器（`Allocator` vtable vs 裸 mimalloc vs std::malloc，`allocator_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,200k/50k 小块(16..511B)。**SOTA 标尺 = 裸 `mi_malloc`/`mi_free`**(设为相对基线,HYLE 的 % 直读 = vtable 接缝开销)。`std::malloc` = 弱基线(MSVC CRT)。此前这层零对标——本节真测揪出接缝成本。

| 场景 | 裸 mi(SOTA) | HYLE vtable | std::malloc | 结论 |
|---|---|---|---|---|
| **churn 空循环**(alloc+free) | 7.79 ns | 9.45 ns(**+21%**) | 26.07 ns | 接缝固定成本在空循环被放大 |
| **alloc+touch+free**(真实) | 7.82 ns | 9.23 ns(**+18%**) | 25.23 ns | **赢 std 2.7x**;接缝仍 +18% |
| **alloc-all/free-all**(批) | 10.41 ns | 10.94 ns(**+5%**) | ~65 ns | 分母大,接缝摊薄到 5% |
| **8 线程并发** | 1.86 ns/op | 2.03 ns/op(**+9%**) | 5.64 ns/op | **赢 std 2.78x** |

**诚实结论**:`Allocator` vtable(注入接缝)相对裸 mimalloc 有 **~1.4ns/对的固定开销**(churn +21%、真实负载 +18%、批量摊到 +5%、并发 +9%),但**比 std::malloc 快 2.7-2.8x**。开销来源:**单 `ReallocFn` 间接调用不可内联** + Win64 ABI 下 5 参签名**第 5 参 `alignment` 压栈** + `system_realloc` 热路径分支(`new_size==0`/`normalize_alignment`/对齐判断)。

**这是接缝设计代价,非 rival-loss**:与 BoundedQueue 输给 moodycamel(真竞争缺陷,已分片修)不同,system_allocator 自身就用 mimalloc(SOTA backing),~1.4ns 是"显式注入 + 跨 DLL C-ABI + 可换 backing"(charter 铁律:单函数指针 vtable)买来的接缝。**真实负载下 alloc+touch 只 +18%,且 touch 越重越摊薄**。可削的肥肉:5 参单函数若拆 3 指针 vtable 可免 Win64 压栈(~0.7ns),但这是 charter 级 ABI 表面改动,留作显式权衡决策(单函数收敛的跨 DLL 价值 vs 0.7ns 微基准)。

### mem 专用分配器簇（arena/pool/stack vs `std::pmr` + 裸 mimalloc，`arena_pool_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,8192×64B。**SOTA 标尺 = `std::pmr`**(标准库自带的 arena=`monotonic_buffer_resource`、pool=`unsynchronized_pool_resource`,完美 drop-in,无需 clone)+ 裸 `mi_malloc`(=`new/delete` 的实际 backing,即"5-10x vs new/delete"声明的对照基线)。**此前这簇零对标——本节真测揪出无锁变体的争用灾难。**

| bump 场景(alloc N + 批量释放) | ns/op | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE Arena**(无锁 bump,池块) | 6.77 | **0.91x(还慢!)** | 原子 CAS 税 |
| **HYLE StackAllocator**(单线程自有 slab) | **0.65** | **9.5x** | ✅ 真兑现"5-10x" |
| std::pmr::monotonic_buffer(SOTA) | 1.46 | 4.2x | Arena 慢它 4.6x |
| 裸 mi_malloc(general,无批量) | 6.32 | 1.0x | — |

| pool churn 场景(同尺寸 alloc/free) | ns/op | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE MemoryArena**(无锁池) | 13.59 | **0.35x(慢 2.88x!)** | 原子 CAS 税 |
| **HYLE GrowablePool**(单线程) | **1.83** | **2.57x** | ✅ 真兑现 |
| std::pmr::unsync_pool(SOTA) | 5.99 | 0.79x | — |
| 裸 mi_malloc/free(general) | 4.72 | 1.0x | — |
| **8 线程共享 MemoryArena**(无锁设计点) | **344.72** | **0.004x(慢 278x!!)** | ⛔ 单点 CAS 争用灾难 |
| └ 裸 mi(8 线程 per-thread heap) | 1.24 | — | mimalloc 零争用 |

**核心诚实结论(揪出的第三个"自以为 SOTA")**:

1. **"专用分配器 5-10x vs new/delete"只对单线程变体成立**:`StackAllocator`(纯指针 bump,0.65ns)**9.5x 赢裸 mi、2.25x 赢 std::pmr arena**;`GrowablePool`(单线程 free-list,1.83ns)2.57x 赢裸 mi、3.3x 赢 std::pmr pool。这俩是真 SOTA。

2. **两个"无锁旗舰"(`Arena`/`MemoryArena`)是全场最慢**:`Arena` 无锁 bump 比裸 malloc 还慢、慢 std::pmr arena **4.6x**;`MemoryArena` 无锁池单线程慢裸 malloc **2.88x**,8 线程共享(**它存在的理由**)慢 mimalloc per-thread heap **278x**。**根因=单 `offset_`/`head_` 原子,每次 alloc/dealloc CAS 同一 cache line**(BoundedQueue 单点 CAS 同病,因 churn 更密而惨烈得多)。mem-design 称 `MemoryArena` "最亮的资产"——benchmark 证伪:**无锁≠快,它是一个被争用的单原子**。

3. **诚实边界 + 为何暂不重写**:① 278x 是 alloc-立即-free 的 tight-churn 最坏争用;真实用途(fiber 栈池:借栈→跑 μs 级→还栈,低频借还)争用远小于此——但作为**通用并发池**,它被 per-thread 分配(mimalloc 的设计)架构性压制。② `Arena`/`MemoryArena`/`StackAllocator`/`GrowablePool`/`ObjectPool` 当前 **src/ 零内部用户**(io/compute/graph 未开工,fiber 栈直接走注入 allocator 未接池),故这是**潜在缺陷非活跃 bug**。不投机重写无人用的代码(避免为假想需求加复杂度);**修复方向已明确**(若将来接 fiber 栈池且需并发:per-core 分片 free-list,照 mimalloc/ShardedQueue),留待真实消费者出现时按其访问模式定。单线程消费者直接用 `StackAllocator`/`GrowablePool` 即可,无需碰无锁变体。

### container 杂项（SmallVector / IntrusiveList vs std，`container_misc_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。两个有性能声明却此前零 bench 的容器。标尺 = std 等价物(`std::vector`/`std::list`,无外部依赖)。

| 场景 | HYLE | std | vs std | 结论 |
|---|---|---|---|---|
| **小 N churn**(建+填 8 int+析构) | SmallVector<int,16> **0.50 ns** | vector 16.57 ns / vector+reserve 3.06 ns | **33x / 6.1x** | ✅ "小 N 省 malloc 大赢"坐实 |
| **list churn**(push 4096 + pop 全) | IntrusiveList **2.55 ns** | list 24.07 ns | **9.4x** | ✅ "O(1) 摘除、无分配"坐实 |

**诚实结论(这两个是真兑现,非欺骗)**:SmallVector<int,16> 全程 inline 零堆分配,小集合 churn **33x 赢 std::vector**(连给 std::vector 预 reserve 也赢 6.1x——省的是构造期那次 malloc);IntrusiveList 链接内嵌节点,push/pop 纯指针操作零分配,**9.4x 赢 std::list**(后者每 push 分配节点)。这俩声明无水分,与无锁分配器的反例对照鲜明:**单线程、无共享、零原子的设计才是 HYLE 真正快的地方**。






- **桶 A — 同操作更省开销（真 drop-in 可比）**：hashmap、分配器、容器。给"倍数"有意义。
- **桶 B — std 根本没有的能力**：fiber、async IO。"倍数"是错的框（无可比物），是「从不能到能」。

## 主表：时间 / 空间分列

### 桶 A：drop-in 可比

| 组件 vs std | 时间（speedup + 复杂度）| 空间（vs std）|
|---|---|---|
| **flat hashmap** vs `unordered_map` | 查找/插入 **2-4×**；均摊 O(1) 同，但常数小（连续布局 cache 命中 + SIMD 16 路探测）| **通常更省**：无 node 的指针 + 每节点分配头；代价是留负载因子（有空槽 + 1 控制字节/槽）。最坏（刚扩容）峰值略高 |
| **专用分配器** 单线程 bump/pool vs new/delete | 分配 **2.5-9.5×**(实测:StackAllocator 9.5×、GrowablePool 2.57×)；O(1) 指针自增 vs free-list+锁。**注:无锁 Arena/MemoryArena 反而慢**(单点 CAS 税,见上节实测)| **更费**（以空间换时间）：块尾浪费 + 不单独释放，生命周期不齐时占用更久、峰值更高 |
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
| **统一内存 MemorySpace** vs host/device 拷贝 | DGX/Mac 上 copy 塌缩 no-op（省一次 DMA）| **更省**：不需 host+device 双份镜像 |

## 整体系统层面（Amdahl，别期待"全面快 N 倍"）

局部再快也受非热路径拖累：

- **瓶颈不在这些点** → 整体可能仅 **1.2×**。
- **并发 / IO 密集** → **10×+**。
- **架构连贯性红利**（不打架的双调度器、统一内存零拷贝）不体现为微基准倍数，而是真实混合负载的系统级延迟/吞吐——Seastar/ScyllaDB 那种 shard-per-core 重写报过 **~2× 系统吞吐**，是"架构做对"的现实数字。

## 诚实边界

**很多情况下 HYLE 对 std 几乎零提升**：代码本就 cache 友好、分配少、单线程、低并发时，std 对常见场景真的好，HYLE 白搭。**赢点恰好集中在 AI infra 热路径**——海量并发（fiber）、高 IOPS（io_uring）、hash 密集（flat map）、分配 churn（arena）、数值核（SIMD）。这不是巧合，HYLE 就是为这个 workload 设计的。

## 一句话

不是"处处比 std 快"。是：**hashmap 稳赢 2-4×、专用分配器对的模式 5-10×、SIMD 数值核 4-16×、并发/IO 是数量级（仅高负载兑现），平凡单线程代码 ≈ 持平**；且 arena/fiber/SmallVector 的时间优势**以空间为代价**。HYLE 的价值是把 AI infra 那几条热路径顶到 std 摸不到的天花板，并让它们在同一套底座上不互相拖累。

## 验证（不是空头支票）

每个数字到 [`roadmap.md`](./roadmap.md) 的 benchmark gate 实测对标：
- P2 mem：对标裸 mimalloc 不退化
- P3 container：hashmap 对标 absl/F14；simd 数学函数 ULP 精度
- P4 sched：百万次 fiber 切换基准
- P5 io：echo/吞吐；compute：CPU+GPU 后端对拍

是骡子是马，gate 上遛。
