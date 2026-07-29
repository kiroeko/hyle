# HYLE 性能：预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 本目录结构（原单文件已拆分，按模块分档）

| 文档 | 内容 |
|---|---|
| 本页（对标总账） | 全模块 win/loss ledger 一览 + 总账小结（最常看的一张表） |
| [error 实测](./benchmarks-error.md) | **error 2026-07-29 两类消费者信息合同三平台 n=15 证书**（无诊断 / 带诊断严格分表；同能力 Abseil 无 LOSE，较弱 std 下界的真实输赢单列） |
| [core 实测](./benchmarks-core.md) | **core 当前证书**：IntrusiveList 对 Boost auto_unlink 同 ownership 的 FIFO/remove/batch-discard；cache/prefetch 只在真实消费者承重 |
| [build/package 实测](./benchmarks-build.md) | **构建系统三平台正常路径证书**：clean/reconfigure/no-op、测试依赖获取分层、组件隔离、production-shape 测试与生成器数据库健康 |
| [platform 实测](./benchmarks-platform.md) | **platform 2026-07-25 当前证书**（既有性能表不变；MappedFile advice/residency 按 OS-call complexity 守门） |
| [time 实测](./benchmarks-time.md) | **time 2026-07-25 当前三平台证书**（clock / exact-ns radix TimerQueue / CommonTimeoutQueue；含 Seastar faithful baseline、order-swap 与 M4 QoS） |
| [mem 实测](./benchmarks-mem.md) | **mem 严格证书**：既有三平台 Region/FixedPool/pool 证书有效；clean `c640072` 补齐 shared allocator/standard-adapter/realloc，`8c85331` 又局部复证 PMR zero-size 修复后的正常路径；Win/WSL/M4 双 QoS n=15 无新增 material LOSE |
| [util-core 实测](./benchmarks-util.md) | **util-core 2026-07-29 三平台 n=15 证书**：Rng/Ref/SpinLock 实际形状；oneTBB 正反序；M4 双 QoS 与 acquire-only RMW 真实回退 A/B；mutex 反例 |
| [obs 实测](./benchmarks-obs.md) | **obs-runtime 2026-07-29 `06cec5a` 三平台 n=15 证书**：compile/runtime gate、exact-byte record/C ABI、immutable sink snapshot、fan-out 与多 producer；成熟库只作同机制对照，不作跨合同倍数宣称 |
| [container 实测](./benchmarks-container.md) | **container 唯一当前证书**：三平台详表、新鲜度、非 SOTA、选型与复跑命令 |
| [simd 产品 kernel](./benchmarks-simd.md) | dot/saxpy、math、dtype、gather、vendor 竞争…；**op 全量不在此** |
| [simd op 对拍证书](./benchmarks-simd-ops.md) | **逐 op vs Highway**：三平台 LOSE=0 摘要 + 覆盖树；全表 → [orderswap 矩阵](../milestones/simd-orderswap-matrix.md)（最新最全） |
| [simd op 对拍方法论](./simd-op-benchmark-methodology.md) | op_parity 规程：形状、双序、锚点闸门、LOSE 证据阶梯 |
| [sched 实测](./benchmarks-sched.md) | fiber 切换、sched 同步（marl 对标）、调度器层；**platform 原语已拆出** → [benchmarks-platform.md](./benchmarks-platform.md) |
| [io 实测](./benchmarks-io.md) | echo 吞吐 + shard-per-core 扩展 + vs 裸 epoll/liburing/asio/libuv + 连接生命周期 + TLS/HTTP + multishot/DEFER/fixed-files/direct-accept 机制项 |
| [预期与边界](./expectations.md) | AVX-512 上架状态、主表（时间/空间分列）、Amdahl 整体观、诚实边界、验证 gate |

## 对标总账（win/loss ledger，全部实测 vs 真 SOTA，非 std）

> **一眼看尽输赢**。**两把标尺各拆成「组件名 + ns」两列**:`std组件`/`std ns`(基线下界——赢它是最低门槛)和 `SOTA组件`/`SOTA ns`(真门槛:absl/F14、mimalloc、moodycamel/tokio/rigtorp、Highway/xsimd、marl、boost、simdutf)。`比std`/`比SOTA`:>1=HYLE 快几倍。**判定**针对 SOTA:✅赢 / ➖平(噪声内) / ❌输(均附根因:by-design 接缝 / 约束选择 / 选错工具 / scalar 待优化,无一为未处理的真缺陷) / ⚠️平台翻转 / ⏳待补。`n/a`=该层无对等物(非未测:如 bounded-MPMC 队列无 std 等价物);`同std`=该项 std 本身即领域 SOTA,无独立第三方更快。**error 行来自 2026-07-29 `6d42add` 的 Win/GCC14/M4、n=15 消费者合同证书**；其余模块日期不一，未刷新前勿当当前证书。

| 模块 · 操作 | HYLE ns | std 组件 | std ns | 比std | SOTA 组件 | SOTA ns | 比SOTA | 判定 |
|---|---|---|---|---|---|---|---|---|
| **error · 无诊断** Status::Ok 构造+检查 | 0.547/0.057/0.226(Win/Lin/M4) | n/a | n/a | n/a | absl::OkStatus | 0.547/0.054/0.226 | ~1.0x | ➖三平台平 |
| **error · 无诊断** code-only Status | 0.547/0.060/0.224 | std::error_code | 0.374/0.062/0.224 | Win **0.68x**；Lin/M4 平 | absl code-only | 1.694/0.916/0.958 | **3.10/15.58/4.28x** | ✅三平台赢同能力 Abseil；诚实保留 Win std 下界 LOSE |
| **error · 无诊断** Result OK 四帧+消费 | 2.585/2.478/3.058 | expected&lt;int,ec&gt; | 2.248/2.624/3.328 | **0.87/1.07/1.09x** | absl::StatusOr | 3.181/2.915/3.415 | **1.216/1.191/1.113x** | ✅三平台赢同能力 Abseil；Win std 下界更快 |
| **error · 无诊断** Result code error 四帧 | 2.380/1.103/1.443 | expected&lt;int,ec&gt; | 4.836/4.413/2.166 | **2.03/4.00/1.50x** | absl::StatusOr | 4.240/2.258/2.455 | **1.773/2.006/1.681x** | ✅三平台赢 |
| **error · 带诊断** Status message 构造/拷贝 | 27.744·10.238/9.876·8.850/17.412·3.181 | n/a | n/a | n/a | absl::Status | 47.071·10.580/20.017·9.262/31.054·4.180 | 构造 **1.70/2.04/1.88x**；拷贝 **1.04/1.05/1.31x** | ✅三平台赢 |
| **error · 带诊断** 一层/三层语义 context | 44.537·120.361/20.948·56.547/19.170·62.786 | n/a | n/a | n/a | absl::StatusBuilder | 262.126·771.647/91.302·278.287/118.366·342.946 | 一层 **5.88/4.32/6.28x**；三层 **6.39/4.90/5.56x** | ✅三平台赢 |
| **error · 带诊断** message Result error | 9.709/9.221/3.814 | n/a | n/a | n/a | absl::StatusOr | 21.157/21.248/7.202 | **2.198/2.292/1.860x** | ✅三平台赢 |
| **error · 带诊断** owning 格式化 / 首错聚合 | 31.484·9.221/19.741·8.794/23.787·3.181 | n/a | n/a | n/a | absl ToString / Update | 40.831·10.537/22.874·9.188/30.729·4.184 | 两项三平台均赢 | ✅详见 error 分表 |
| **error · 带诊断** errno+native / native 构造 / 查询 | 39.732·35.018·2.017/51.330·12.122·0.642/26.238·16.477·0.962 | n/a | n/a | n/a | absl errno+payload / payload | 168.032·87.575·5.683/74.727·37.709·4.636/116.480·53.482·4.422 | 三项三平台均赢 | ✅同一个机器可读原生值；不外推任意 payload |
| **simd·ops** 可对拍 op 全量(矩阵 v4) | — | (无 std) | n/a | n/a | Highway 同 ISA 同 op | — | **LOSE=0** | ✅**Win/WSL/NEON 明确 LOSE 均为 0**(320/320/308 case)；?≠TIE；全表 [benchmarks-simd-ops.md](./benchmarks-simd-ops.md)→[orderswap-matrix](../milestones/simd-orderswap-matrix.md) |
| **simd** dot product(4096 f32) | 0.024 | scalar autovec | 0.37 | 15x | Highway contrib/dot(真 SOTA 4-acc) | 0.024 | 1.00x | ✅**4 累加器展开后 0.079→0.024(3.3x)平真 SOTA**;旧"平 Highway 0.079"是两个都单累加器的假平局(对照用了 Highway 教学 kernel 非 contrib/dot),见 [benchmarks-simd.md](./benchmarks-simd.md) |
| **simd** saxpy(4096 f32) | 0.026 | scalar autovec | 0.057 | 2.2x | Highway avx2 | 0.025 | 1.01x | ✅赢 std、平 Highway(element-wise 带宽受限,多累加器无益,未改) |
| **simd** reduce_max(4096 f32) | 0.077 | scalar autovec | 0.73 | 9.5x | Highway avx2 | 0.077 | 1.00x | ➖平 Highway(bench-only,非公开 op;单累加器,若上架公开 op 可同 dot 展开) |
| **simd** scale(4096 f32) | 0.025 | scalar autovec | 0.044 | 1.8x | Highway avx2 | 0.025 | 1.01x | ➖平 Highway(bench-only 非公开 op;带宽受限) |
| **simd** dot_f32 公开分发入口 | 0.024 | scalar autovec | 0.37 | 15x | Highway contrib/dot(真 SOTA) | 0.024 | 1.00x | ✅出厂 dispatch 入口 4 累加器,平真 SOTA(分发税大 batch 摊薄≈0) |
| **simd** saxpy_f32 公开分发入口 | 0.026 | scalar autovec | 0.057 | 2.2x | 直调 avx2(无间接) | 0.025 | 1.01x | ➖同上(大 batch 摊薄≈0,小 n 未测) |
| **dtype** f16→f32 | 0.025 | scalar ref | 0.66 | 26x | Highway PromoteTo | 0.048 | Win 1.88x/Lin 平 | ✅Win 赢 Highway/Linux 统计平局(平台差异) |
| **dtype** f32→f16 | 0.030 | scalar ref | 2.07 | 25–80x | Highway DemoteTo | 0.048 | Win 1.61x/Lin 1.90x | ✅赢 Highway 双平台(唯一双平台稳赢) |
| **dtype** bf16→f32 | 0.030 | scalar ref | 0.10 | Win 3.5x/Lin 平 | Highway PromoteTo | 0.050 | Win 1.87x/Lin 平 | ✅Win 赢 Highway/Linux 平(纯位运算 GCC 也向量化) |
| **dtype** f32→bf16 | 0.095(Lin)/0.098(Win) | scalar ref | 0.18/0.70 | 1.9–7x | Highway OrderedDemote2To(真 SOTA 2-vec) | 0.097/0.099 | 1.02x/1.01x | ➖平真 SOTA(改 2-input 生产形状后 Lin 0.104→0.095 ~1.09x,平 OrderedDemote2To;旧对标的 DemoteTo 便利版是 8→8 朴素形状,HYLE 现反超它 0.107/0.117);**AVX-512-BF16 native `vcvtneps2bf16` 已上架**(待真机,见 expectations AVX-512 节①) |
| **simd native range** transform/count/find/copy-if | Win AVX2 0.075/0.065/0.064/0.085 | pinned Highway 同形 | 0.075/0.064/0.064/0.099 | 1.004/0.954/1.002/1.165 | Highway contrib/algo | 同左 | 前三项 TIE；copy-if WIN 15/15 | ⚠️新 expert 算法层无结构性税；仅 Windows AVX2，三平台/变宽证书未齐，见 [simd](./benchmarks-simd.md) |
| **math** exp(best-of-breed→xsimd) | 0.23 | libm scalar | 2.06 | 9x | Highway / xsimd | 0.30/0.22 | 1.30x/1.0x | ✅平 xsimd 最快、反超 Highway(双平台一致) |
| **math** log(best-of-breed→Highway) | 0.27 | libm scalar | 2.0–2.6 | 4–8x | Highway / xsimd | 0.27/0.37 | 1.0x/1.45x | ✅平 Highway、反超 xsimd(修产品核 128b bug 后 Linux 同样成立) |
| **math** tanh(best-of-breed→xsimd) | 0.44 | libm scalar | 6.3–8.7 | 14–20x | Highway / xsimd | 0.54/0.45 | 1.22x/1.0x | ✅平 xsimd、反超 Highway(双平台一致) |
| **platform** Parker ping-pong(真阻塞) | 341/22.5k/2053(Win/Lin/M4) | std::condition_variable | 638/23.5k/2130 | **1.75x/1.05x/1.05x** | (std CV 即实用标尺) | 同左 | 同左 | ⚠️**平台翻转**(Win 大赢;Lin/M4 平或微赢)。旧 2.99x 作废。见 [platform](./benchmarks-platform.md) |
| **platform** Thread spawn+join | 43.4k/33.6k/10.6k | std::thread | 44.4k/33.6k/10.7k | ~1.0x 三平台 | std::thread(即 SOTA) | 同左 | ~1.0x | ➖平(OS 主导,证零封装税) |
| **platform** Thread owning callable | 40.529k/32.787k/10.510k | raw `entry+ctx` core | 40.081k/33.511k/10.538k | ~1.0x 三平台 | std::thread | 40.494k/33.686k/10.553k | ~1.0x 三平台 | ➖n=15 三平台平；易用入口无可测 OS-thread 创建税 |
| **platform** directory_iterate visitor | 40.23k/7.05k/16.46k | std::filesystem | 39.76k/14.97k/18.25k | **0.981/2.135/1.109x** | filesystem | 同左 | 同左 | ⚠️当前三平台证书：Win TIE，Linux/M4 WIN |
| **platform** vm_allocate_rw vs raw 一步 | 885/860.9/309.8 | raw 一步分配 | 880/853.8/305.0 | **0.987/0.978/0.982x** | raw 一步 | 同左 | 同左 | ➖三平台 TIE，普通 VM 形状近 OS 地板 |
| **platform** vm_reserve+commit vs raw 一步 | 1144/1086.3/720.5 | raw 一步分配 | 877/853.8/305.0 | **0.77/0.779/0.421x** | raw 一步 | 同左 | 同左 | ❌**by-design 两步**输一步 raw；购买 staged commit，非漏优化 |
| **platform** MappedFile 64KiB vs ifstream | 45.05k/11.48k/13.31k | ifstream 拷贝 | 79.42k/**8.72k**/20.11k | **1.76/0.76/1.51x** | 使用形状标尺 | 同左 | 同左 | ⚠️Win/M4 赢；Linux 小顺扫应选 buffered read |
| **platform** MappedFile vs 同合同 raw mmap | 45.05k/11.48k/13.31k | raw validated mmap | 44.77k/11.22k/13.15k | **0.992/0.982/0.992x** | raw 地板 | 同左 | ≤1.8% | ➖Win/M4 TIE；WSL sign 输 1.8%，完整 Result/RAII 接近地板 |
| **platform** FileHandle vs raw open/read/close | 34.41k/7.64k/11.90k | raw OS | 35.01k/7.49k/11.62k | **1.006/0.976/0.978x** | raw 地板 | 同左 | ≤2.4% | ➖Win TIE；POSIX 安全便利路径付 2.2–2.4%，`get()` 保留 hard escape |
| **platform** persistent MappedFile 随机页访问 | 2.309/2.532/0.647 | ifstream seek/read | 2585.7/420.5/1527.9 | **1125/168/2355x** | page-cache random access | 同左 | 同左 | ✅真实 mapping 甜点三平台大赢 |
| **platform** Parker token / `park_timeout(0)` | 9.215·1.690 / 9.818·0.564 / 1.850·1.346 ns | (无公平对手) | n/a | n/a | atomic floor 非对等 | n/a | n/a | ✅zero-timeout 纯 load；旧 µs 级路径作废 |
| **time** realtime/monotonic | 三平台约12–15 | `std::chrono` 对应钟 | 同档 | ~1.0x | std 本身即同 OS primitive | 同左 | ~1.0x | ➖ header-inline 默认钟无抽象退化 |
| **time** profiling/coarse clocks | raw profiling 7.19/6.77/1.13ns（Win/Lin/M4）；native coarse 2.91/3.75ns | HYLE 默认钟 | 12.8/12.5/12.9ns | raw **1.77/1.84/11.42x** | Folly/Abseil hardware-counter 形状 | 同算法档 | ordered pair 单列强合同成本 | ✅ raw profiling 三平台真收益；M4 ARM counter 可用；macOS coarse unavailable 正确回退 |
| **time** TimerQueue arm/cancel/expire | 见模块证书 | (std 无) | n/a | n/a | faithful Seastar radix | 同算法同档 | Win/M4 多数 1.05–1.43x；WSL churn 顺序敏感 | ⚠️ SOTA radix；WSL reverse churn 约慢7%，禁止写全胜 |
| **time** TimerQueue vs indexed eager heap | 见模块证书 | (std 无) | n/a | n/a | libevent/libuv shape heap | 见详表 | 约2.25–9.83x | ✅ 三平台三负载胜，cancel 复杂度 O(1) vs O(log n) |
| **time** CommonTimeoutQueue | 见模块证书 | (std 无) | n/a | n/a | checked / naked TAILQ | 见详表 | 同合同不输；M4 naked expire 约0.78x | ⚠️ 同合同 SOTA，公开安全税 |
| **util-core** Rng next_u64 | 1.550/1.385/2.264 (Win/Lin/M4) | mt19937_64 | 2.243/1.455/2.509 | **1.451/~1.05/1.106x** | xoshiro256++ canonical/Folly 同族 | 同算法档 | 同左 | ✅ 三平台 sign WIN |
| **util-core** Rng varying range_u64 | 2.002/1.474/2.202 | std distribution | 2.750/2.031/7.763 | **1.372/~1.38/3.520x** | Lemire mapping | canonical | 同左 | ✅ 三平台 WIN |
| **util-core** iterator shuffle | 1.137/1.260/0.810 | std::shuffle(同一 Rng) | 5.597/1.345/3.677 | **4.858/~1.06/~4.54x** | 同 URBG Fisher–Yates | 同左 | 同左 | ✅ 三平台 sign WIN |
| **util-core** SpinLock 无争用 | 3.758/4.570/0.904 | std::mutex | 10.991/2.379/9.190 | Win/M4 大赢、Linux mutex 更快 | oneTBB spin_mutex | 3.739/4.573/0.897 | 1.000/1.000/0.993x | ➖正反序同档；acquire-only M4 0.888x 回退已由 `3c32be6` 修复 |
| **util-core** SpinLock 2/8 threads | 8.379·61.382/7.490·50.047/2.405·5.545 | mutex（选择边界） | 13.970·46.306/33.731·38.457/高方差 | 平台翻转 | oneTBB spin_mutex | 8.419·59.219/7.593·51.146/2.316·5.518 | 正反序约同档 | ➖M4 双 QoS 同档；高争用不据此选 spin |
| **util-core** Ref fan-out 64 | 7.956/8.055/3.190 | shared_ptr | 8.020/10.773/3.211 | **~1.00/~1.34/~1.01x** | shared_ptr ownership 原子基线 | 同左 | 同左 | ✅ Win/M4 TIE、Linux WIN |
| **obs** gate / preformatted view / structured / 8-producer | 2.160·42.070·38.450·93.690 / 0.910·38.970·34.500·62.950 / 1.100·23.910·23.560·124.390 (Win/Lin/M4) | (无 std log) | n/a | n/a | 同步 sink registry 机制参照 Abseil/Folly | 非同合同 | n/a | ✅ `06cec5a`：编译裁剪 O(0)、runtime gate O(1)、exact view/emit O(S)，NUL convenience O(M+S)；热路径无分配/无 mutation mutex。无同 metadata/ownership peer，故不作跨合同胜负。见 [obs](./benchmarks-obs.md) |
| **mem** SystemAllocator churn(static/source direct) | 7.93/8.04/7.77 | malloc OS | 25/4.7/20 | 大赢 OS | 裸 mimalloc | 7.86/8.04/7.91 | **~1.00x 三平台** | ➖合并形状**平 mi SOTA**(2026-07-20 n=15) |
| **mem** SystemAllocator churn(shared same-link) | 8.17/7.81/7.98 | malloc OS | 37.22/—/— | 平台相关 | 同一 DSO backend seam | 8.03/7.84/7.95 | **0.980/1.009/0.996** | ➖ clean `c640072` n=15：三平台默认 shared 形状无 material LOSE；M4 utility 同为 0.996 |
| **mem** SystemStdAllocator same-link | 8.18/7.86/7.98 | SystemAllocator same-link | 8.17/7.81/7.98 | **0.995/1.000/1.000** | runtime StdAllocator | 8.53/7.95/7.64 | 更强 runtime 合同另列 | ➖ empty/always-equal 默认 adapter 三平台零 material 税；M4 utility 1.004 |
| **mem** system_memory_resource same-link | 8.62/8.15/7.82 | direct-backend PMR | 8.65/8.47/8.48 | **1.004/1.039/1.083** | runtime PmrResource | 9.43/8.54/8.16 | 默认更快 1.094/1.048/1.043x | ✅默认单层 PMR 三平台 TIE/WIN；M4 utility 1.086；runtime Windows 双动态层成本显式保留 |
| **mem** runtime Allocator same-link | 8.35/7.90/7.64 | direct backend seam | 8.03/7.84/7.95 | **0.964/0.999/1.039** | runtime PMR（更重合同） | 9.43/8.54/8.16 | 更快 | ➖显式 policy 注入：Win 约 3.6%，WSL TIE、M4 WIN；不混入默认路径 |
| **mem** PooledRegion/Stack bump | 0.568·0.571/0.345·0.342/1.176·1.175 | pmr monotonic | 1.483/0.648/1.204 | **2.59·2.64/1.81·1.85/1.023·1.023x** | 裸 mi | 6.17/3.84/4.50 | 10–11x/3.8x | ✅ clean `2cbc307`：Win/Linux 明赢；M4 interactive/utility 均 sign WIN、同档，无硬 P/E 绑定宣称 |
| **mem** Region owned request scratch p50（4096 alloc，ns/op） | 1.318/1.389/1.343 | pmr monotonic release | 1.855/2.015/2.431 | **1.407/1.451/1.803x** | Folly Arena（shape/复杂度参考，未同 harness 实测） | n/a | n/a | ✅ clean `6213115` shipping：Win/Linux 15/15、M4 31/31 sign WIN；p95/p99 同样全胜，M4 utility 1.811x，无硬核心亲和 |
| **mem** Region clear/reuse 2048/32 blocks 单位成本比 | 1.266/0.977/1.804（Win/Lin/M4） | O(N²) restart scan 预期约 64x | ~64x | — | Folly monotonic block cursor | 同形 | 线性总工作 | ✅ 31 process×10,001 epochs；结构门禁另证每 block/epoch 至多一次；M4 utility 同为 1.804x |
| **mem** GrowablePool 单线程 | 2.76/1.38/2.46 | pmr unsync | 5.57/11.3/3.47 | **2.0/8.3/1.4x** | 裸 mi | 4.75/4.31/5.80 | **1.7/3.1/2.4x** | ✅三平台赢公平 ST 标尺；固定共享池不是这条单线程合同的对手 |
| **mem** GrowablePool 8t per-thread | 0.93/0.61/0.60 | mi 8t | 1.35/1.28/1.42 | **1.49/2.07/2.38x** | mi per-thread | 同左 | 同左 | ✅三平台 sign 赢 |
| **mem** SharedPool request batch=64（硬容量） | 6.556/6.182/8.260 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 2.244/1.703/2.108 | **0.354/0.270/0.258x** | ℹ️较弱合同成本参照，不作 WIN/LOSE；completed-free 立即共享与精确 quota 不能冒充本地 churn |
| **mem** SharedPool pipelined handoff（硬容量） | 4.046/3.266/9.348 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 8.894/7.305/4.994 | **2.176/2.132/0.546x** | ℹ️较弱合同成本参照，不作 WIN/LOSE；绝对成本与平台差异保留在详表 |
| **mem** SharedPoolCache request batch=64（显式 owner cache） | **1.131/1.058/0.897** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 2.244/1.703/2.108 | **1.951x/1.593x/2.348x** | ✅ clean `1de939d`：Win/WSL 15/15、M4 interactive+utility 31/31 sign WIN |
| **mem** SharedPoolCache pipelined handoff（显式 owner cache） | **1.387/1.336/1.945** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 8.894/7.305/4.994 | **6.419x/5.493x/2.601x** | ✅ clean `1de939d`：三平台 spread/sign WIN；M4 utility 2.593x、31/31 |
| **mem** FixedPool external variable-size alloc/free | 6.835/6.322/6.012 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 10.101/7.845/9.406 | **1.491/1.293/1.563x** | ✅ `20f3676`：O(1) two-level fit；Win、M4 双 QoS spread/sign WIN，WSL 绝对 spread TIE 但 15/15 sign WIN；utility 1.564x；fixed 指 backing 容量，不指请求大小 |
| **mem** FixedPool retained-span realloc | **1.286/1.207/1.267** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool realloc | 1.649/1.687/2.334 | **1.280/1.394/1.829x** | ✅ `20f3676`：Win/WSL 15/15、M4 interactive+utility 31/31 spread/sign WIN；utility 1.657x |
| **mem** FixedPool whole-request p50（Win/Lin/M4，µs） | 28.600/22.814/24.292 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 42.900/33.374/40.791 | **1.500/1.474/1.679x** | ✅ Win clean `20f3676`、Linux/M4 `6213115` shipping：p50/p95/p99 全 sign WIN；M4 utility 1.666x，无硬 P/E 绑定宣称 |
| **mem** Region 128MiB request reservation / payload（Win/Lin/M4） | **1.031/1.031/1.031** | pmr monotonic | 1.443/1.302/2.000 | **更低 28.6%/20.8%/48.5%** | 同 backing live RSS Δ (MiB) | 129.11/138.05/128.67 vs 128.59/144.09/128.72 | — | ✅ clean `02ec81e`：exact fragmentation 三平台更低；RSS Win +0.4% / Linux **-4.2%** / M4 -0.04%，三平台同档且 Linux 缺口关闭；full release backing 均为 0 |
| **mem** FixedPool 75% mixed-fill 后 fragmented availability（Win/Lin/M4） | **0.6858/0.6858/0.6858** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 0.4843/0.4816/0.4816 | **1.416/1.424/1.424x** | ✅ `20f3676`：此 selective-free workload 下最大连续对象约 1.42x；两者 full recovery=1.000；不外推到任意分布 |
| **mem** SystemAllocator reallocate(static/source direct) | 66.98/79.28/81.64 | std::realloc | 196/42/187 | 平台相关 | mi_realloc | 67.69/79.00/83.58 | **1.005/0.989/1.018** | ✅ clean `2cbc307`：Win/Linux TIE，M4 sign WIN；合并形状同档 |
| **mem** SystemAllocator reallocate(shared same-link) | 72.77/68.23/82.07 | std::realloc | 196.38/42/187 | 平台相关 | 同一 DSO backend realloc | 71.14/67.84/81.91 | **0.974/0.997/1.003** | ➖ clean `c640072`：默认 shared realloc 最大约 2.6%，三平台无 material LOSE；M4 utility 0.983 |
| **mem** runtime Allocator reallocate(shared same-link) | 73.79/66.47/79.48 | std::realloc | 196/42/187 | 平台相关 | 同一 DSO backend realloc | 71.14/67.84/81.91 | **0.965/1.018/1.023** | ➖Win 动态策略约 3.5%，WSL/M4 WIN；M4 utility 1.037，不混入默认路径结论 |
| **mem** scheduler-owned FiberStackPool batch p50 | 7.60/10.19/13.25 µs | （无 std fiber stack cache） | n/a | n/a | direct guarded OS VM | 151.80/182.62/80.67 µs | **20.03/17.89/6.07x** | ✅ p50/p95/p99 三平台全 sign WIN；M4 utility 也全 WIN，详见 mem §5 raw CSV |
| **container·hash** Hash&lt;T&gt; 16B POD 内联 | 0.43 | std::hash | ~0.9 | ~2x | wyhash | 0.94 | 2.2x | ✅赢(2026-07 改 abseil 单 128 位乘 mix64_2:0.59→0.43,vs wyhash 1.6x→2.2x) |
| **container·hash** mix64 整数 | 0.44 | std::hash | 1.50 | 3.4x | wyhash64 | 0.57 | 1.27x | ✅赢(已 SOTA=abseil 同款单乘折叠) |
| **container·hash** hash_bytes 8B(inline mix,非 XXH3) | 0.23/B | std::hash&lt;string&gt; | 0.29 | 1.3x | 裸 XXH3-small / wyhash | 0.21/0.30 | ~平/赢 wyhash | ➖**≤16B 走 inline mix 非 XXH3**(两算法比,非"输 XXH3");赢 wyhash-small、近平 XXH3-small |
| **container·hash** hash_bytes 16B(inline mix,非 XXH3) | 0.15/B | std::hash&lt;string&gt; | 0.35 | 2.3x | 裸 XXH3-small / wyhash | 0.12/0.15 | ~平 wyhash | ➖同上(inline mix vs XXH3-small,不同算法;热路径 Hash&lt;T&gt; 见上行赢 wyhash 2.2x) |
| **container·hash** hash_bytes 64B(Win/Lin/M4) | **21.94/26.71/29.03 GB/s** | std::hash&lt;string&gt; | 明显较慢 | — | raw XXH3/wyhash 最快者 | 24.21/26.46/31.21 GB/s | **三平台同档** | ➖ 17–128B 已是公开头内产品 multiply stream；combined spread 内同档，不再有旧 DLL seam |
| **container·hash** hash_bytes 256B(M4) | **40.06 GB/s** | std::hash&lt;string&gt; | 明显较慢 | — | raw wyhash | 43.77 GB/s | 0.915x | ⚠️ 小型 long-input product seam；没有为一个长度扩大 public header streaming core |
| **container·hash** hash_bytes 1KiB(Win/Lin/M4) | **45.70/61.09/40.92 GB/s** | std::hash&lt;string&gt; | 明显较慢 | — | raw 最快 vendor core | 47.43/58.59/42.72 GB/s | **同档/赢/同档** | ✅ 三平台 n=15；Win/M4 1KiB、Linux 512B 是产品 API crossover，不取裸算法交叉点 |
| **container·hash** hash_bytes 4KiB(Win/Lin/M4) | **63.78/68.93/47.13 GB/s** | std::hash&lt;string&gt; | 明显较慢 | — | raw 最快 vendor core | 63.29/68.66/47.35 GB/s | **三平台同档** | ✅ runtime CPU+OS gate；M4 NEON 证书已完成 |
| **container·map** find-hit(100k u64, Win) | 2.464 | std::unordered_map（历史） | 10.47 | 4.25x | absl flat_hash_map | 3.010 | 1.21x | ✅ Win/WSL/M4 分别 1.21/1.15/1.05x |
| **container·map** find-miss(100k, Win) | 4.499 | std::unordered_map（历史） | 11.57 | 2.57x | absl flat_hash_map | 4.801 | 1.07x | ⚠️ Win/M4 赢；WSL **0.97x**，Absl 13/15 |
| **container·map** insert(100k fresh, Win) | 11.742 | std::unordered_map（历史） | 80.07 | 6.82x | absl flat_hash_map | 24.672 | 2.07x | ✅ Win/WSL 赢；M4 0.975x，统计同档 |
| **container·map** erase(100k build+erase, Win) | 19.396 | std::unordered_map（历史） | 93.10 | 4.80x | absl flat_hash_map | 34.748 | 1.80x | ✅ Win/WSL/M4 分别 1.80/1.37/1.19x |
| **container·map** iterate(100k visit, Win) | 0.980 | std::unordered_map（历史） | 8.37 | 8.54x | absl flat_hash_map | 1.857 | 1.90x | ✅ Win/WSL/M4 分别 1.90/2.30/2.27x |
| **container·set** find-hit(100k u64, Win) | 1.891 | （无当前 std 证书） | — | — | absl flat_hash_set | 3.136 | 1.66x | ✅ Win/M4 赢；WSL TIE 8/15 |
| **container·set** find-miss(100k, Win) | 4.419 | （无当前 std 证书） | — | — | absl flat_hash_set | 5.951 | 1.35x | ⚠️ Win/M4 赢；WSL **0.94x**，Absl 14/15 |
| **container·set** insert(100k fresh, Win) | 10.159 | （无当前 std 证书） | — | — | absl flat_hash_set | 19.475 | 1.92x | ✅ Win/WSL 赢；M4 0.976x，统计同档 |
| **container·set** iterate(100k visit, Win) | 1.004 | （无当前 std 证书） | — | — | absl flat_hash_set | 1.916 | 1.91x | ✅ Win/WSL/M4 分别 1.91/2.20/2.69x |
| **container·btree** find-hit 64Ki(Win/Lin/M4) | **44.906/47.329/41.585** | （无 std 当前证书） | — | — | absl::btree_map | 49.989/51.633/44.820 | **1.107/1.092/1.108x** | ✅ 三平台 paired WIN |
| **container·btree** in-domain miss(Win/Lin/M4) | **46.298/49.539**/45.730 | （无 std 当前证书） | — | — | absl::btree_map | 49.782/53.048/**41.348** | **1.078/1.065x**/**0.887x** | ⚠️ x86 WIN，M4 LOSE；真实 sparse-key 域内 miss |
| **container·btree** above-max miss 边界(Win/Lin/M4) | **15.562**/18.164/**17.067** | （边界分布） | — | — | absl::btree_map | 21.379/**16.996**/19.426 | **1.380x**/**0.932x**/**1.137x** | ⚠️ Win/M4 WIN，Linux LOSE；不能用单一分布塑形整个类型 |
| **container·btree-set** membership(Win/Lin/M4) | 47.085/47.503/**36.360** | （无 std 当前证书） | — | — | absl::btree_set | 48.095/47.444/39.370 | TIE/TIE/**1.073x** | ✅ x86 同档，M4 WIN |
| **container·btree** lower_bound+scan16(Win/Lin/M4) | 74.341/**69.977**/60.342 | （无 std 当前证书） | — | — | absl::btree_map | **70.934**/74.755/62.839 | **0.956x**/**1.079x**/TIE | ⚠️ Windows LOSE，Linux WIN，M4 TIE |
| **container·btree** deterministic iteration(Win/Lin/M4) | 1.151/1.021/**0.820** | （无 std 当前证书） | — | — | absl::btree_map | **1.020**/1.021/1.055 | **0.890x**/TIE/**1.287x** | ⚠️ Windows LOSE，Linux TIE，M4 WIN |
| **container·btree** shuffled build(Win/Lin/M4) | **65.937/64.511**/59.660 | （无 std 当前证书） | — | — | absl::btree_map | 73.561/69.103/**56.916** | **1.120/1.062x**/**0.941x** | ⚠️ x86 WIN，M4 LOSE |
| **container·btree** O(N) copy(Win/Lin/M4) | **2.936/2.803/1.305** | （无 std 当前证书） | — | — | absl::btree_map | 11.715/4.330/4.563 | **3.927/1.523/3.490x** | ✅ 三平台 WIN；树形克隆，无 comparator |
| **container·btree** erase+insert churn(Win/Lin/M4) | **48.064/45.455**/41.890 | （无 std 当前证书） | — | — | absl::btree_map | 55.442/49.487/**41.196** | **1.159/1.096x**/**0.983x** | ⚠️ x86 WIN，M4 仅慢 1.7% |
| **container·btree** time expiry+refill(Win/Lin/M4) | **31.641**/32.198/**25.463** | （无 std 当前证书） | — | — | absl::btree_map | 33.293/**31.497**/26.107 | **1.051x**/**0.978x**/**1.027x** | ⚠️ Win/M4 WIN，Linux 仅慢 2.2%；边界偏置 split/delete，无测项元数据 |
| **container·concurrent-node** protected/8R/mixed（Win/Lin/M4） | 10.239·3.379·4.609 / 8.910·3.133·4.258 / 7.229·5.347·5.576 | （无 std concurrent map） | n/a | n/a | oneTBB concurrent_hash_map 同 accessor | 16.617·7.048·7.120 / 15.749·5.857·6.434 / 10.616·11.834·10.165 | **1.45–2.22x** | ✅ 三平台三路径均 15/15；稳定 node address |
| **container·concurrent-node** clear+8W rebuild（Win/Lin/M4） | 76.309/62.862/**61.529** | （无 std concurrent map） | n/a | n/a | oneTBB concurrent_hash_map | 90.416/76.056/79.669 | 1.174/1.210/**1.298x** | ⚠️ Win/Linux paired TIE（9/15、12/15）；M4 15/15 WIN |
| **container·concurrent-flat** protected/8R/mixed（Win/Lin/M4） | 10.926·3.746·5.515 / 8.327·3.528·5.054 / 5.381·6.669·7.034 | （无 std concurrent map） | n/a | n/a | oneTBB concurrent_hash_map 同 accessor | 16.617·7.048·7.120 / 15.749·5.857·6.434 / 10.616·11.834·10.165 | **1.25–1.96x** | ✅ 三平台三路径均 15/15；compact Swiss shards |
| **container·concurrent-flat** clear+8W rebuild（Win/Lin/M4） | **12.115/10.624/12.690** | （无 std concurrent map） | n/a | n/a | oneTBB concurrent_hash_map | 90.416/76.056/79.669 | **6.250/7.052/6.275x** | ✅ 三平台 15/15 WIN；高频 registry rebuild 首选 |
| **container·bitset** graph frontier OR+count(1M bits, Win/Lin/M4) | **0.004/0.003/0.004 per bit** | std::bitset&lt;1M&gt; | 0.005/0.018/0.004 | 1.40x/5.18x/TIE | portable fused uint64 blocks | 0.019/0.016/**0.003** | 5.21x/4.52x/**0.65x** | ✅ 三平台 n=15；M4 与同接口 std 同档，诚实保留一遍扫描 floor 快 1.53x；⏳ Boost peer |
| **container·sparse-set** frame clear+insert(4096/1M, Win/Lin/M4) | **1.145/0.959/1.402 per ID** | dense byte bitmap | 3.705/3.454/2.604 | 3.29x/3.61x/1.86x | std::unordered_set | 32.439/10.814/16.039 | 28.6x/11.6x/11.4x | ✅ 三平台 n=15；O(1) clear + O(active) iteration 形状 |
| **container·sparse-set** mixed membership(Win/Lin/M4) | 1.380/1.327/0.693 per query | dense byte bitmap | **0.363/0.275/0.477** | **0.264x/0.211x/0.688x** | std::unordered_set | 5.560/2.895/0.945 | 4.03x/2.17x/1.36x | ⚠️ dense 一次 load 胜 3.77x/4.74x/1.45x；这是选型边界，不加 bitmap 元数据隐藏 |
| **container·ring** fixed-capacity FIFO(Win/Lin/M4) | **1.307/0.610/0.524** | std::deque | 2.101/0.783/4.823 | **1.61x/1.28x/9.20x** | portable Folly fixed-ring shape | 1.390/0.622/0.600 | **1.06x/TIE/1.15x** | ✅ 三平台 n=15；Win/M4 胜 floor，Linux 同档，均胜 std |
| **container·byte-ring** 32KiB backlog/4KiB staging(Win/Lin/M4) | **0.009/0.009/0.010 per B** | std::deque&lt;byte&gt; | 1.024/0.419/1.216 | **约118x/48.2x/121x** | linear buffer+memmove | 0.016/0.017/0.043 | **1.82x/1.94x/4.26x** | ✅ copy API 三平台 15/15；另有 two-span zero-copy 正常入口 |
| **container·vec** SmallVector 小 N build+consume(Win/Lin/M4) | 0.982/0.868/0.702 | std::vector | 16.698/3.939/— | 大幅赢 | absl::InlinedVector | 1.237/0.697/1.747 | **1.225x/0.803x/2.49x** | ✅ Win/M4 WIN；Linux combined spread 46.1%，TIE；旧 data-only DCE 测项作废 |
| **container·vec** SmallVector spill build+consume(Win/Lin/M4) | 1.320/0.632/0.609 | std::vector | 4.825/—/— | 赢 | absl::InlinedVector | 2.795/0.770/1.626 | **2.124/1.218/2.67x** | ✅ 三平台赢 |
| **container·vec** PodVector growth/read(Win/Lin/M4) | 0.509/0.184；0.277/0.091；0.342/0.035 | std::vector(即 SOTA) | 0.787/0.184；0.249/0.091；0.362/0.035 | Win 1.545x；Lin TIE；M4 1.06x | 同 std | 同左 | 同左 | ✅ growth 同档或赢；read 三平台同档 |
| **container·str** StringInterner intern(Win/Lin/M4) | 21.413/21.149/11.302 | （无锁 floor 仅诊断） | — | — | thread-safe map+mutex peer | 27.265/18.909/17.325 | **1.273/0.894/1.533x** | ⚠️ 仅 WSL 单条入口落后；batch 赢 1.36x |
| **container·str** StringInterner 1KiB unique growth(Win/Lin/M4) | 95.852/96.568/76.656 | （无锁） | — | — | absl::flat_hash_set floor | 579.192/149.046/98.901 | **5.934/1.543/1.290x** | ✅ cached resident hash；三平台赢 |
| **container·str** StringInterner 8-thread adaptive shards(Win/Lin/M4) | 12.837/14.821/14.678 | sharded std+mutex | 55.722/—/— | Win 4.34x | sharded Absl+mutex | 22.661/21.167/19.679 | **1.782/1.428/1.341x** | ✅ 三平台赢；M4 128B core spacing 受影响边已复证 |
| **container·str** StringInterner acquire/retain/release(Win/Lin/M4) | 11.897/9.637/7.678 | direct-ID+mutex 同契约 peer | 18.463/12.267/12.345 | **1.559/1.273/1.608x** | 同左 | 同左 | 同左 | ✅ 三平台赢 |
| **container·str** find(char) 全扫 miss(4KB) | 0.02/B | std::string_view::find | 0.01 | ~1.0x | string_view(即 SOTA,libc memchr) | 0.01 | ~1.0x | ➖平(切 memchr;原 scalar 输 ~20x) |
| **container·str** find(needle) 6B 子串 | 0.08/B | std::string_view::find | 0.08 | ~1.0x | string_view(即 SOTA,two-way 向量化) | 0.08 | ~1.0x | ➖平(转发 std;自研 memchr 候选曾输 2.5x) |
| **container·str** find_first_of(4 字符集) | 0.05/B | std::string_view | 0.05 | ~1.0x | string_view(即 SOTA) | 0.05 | ~1.0x | ➖平(转发 std;自研 memchr-min 曾 0.62x) |
| **container·str** starts_with(16B 前缀) | 1.79 | std::string_view | 1.77 | ~1.0x | string_view(即 SOTA,memcmp) | 1.77 | ~1.0x | ➖平(切 memcmp;原 字节循环输 2x) |
| **container·str** split 全文按空格(4KB) | 0.44/B | string_view find+remove_prefix | 0.54 | 1.22x | string_view(即 SOTA) | 0.54 | 1.22x | ✅赢(无分配子视图) |
| **container·str** trim 两端(4KB) | 13.6 | string_view 两侧 trim | 16.85 | 1.24x | string_view(即 SOTA) | 16.85 | 1.24x | ✅赢(组合 find_first/last_not_of) |
| **container·str** rfind(char) 后向扫 miss | 0.012/B | std::string_view::rfind | 0.012 | ~1.0x | string_view(即 SOTA) | 0.012 | ~1.0x | ➖平(AVX2 后向扫,统计平局 gap2%«spread11%;原 scalar 输 ~20x) |
| **container·str** find_first_not_of 全字母 | 0.06/B | std::string_view | 0.06 | ~1.0x | string_view(即 SOTA) | 0.06 | ~1.0x | ➖平(转发 std;自研 bool 表/PCMPESTRI 曾输 4.5x/0.31x) |
| **container·str** find_last_of(digits) miss | 0.06/B | std::string_view | 0.06 | ~1.0x | string_view(即 SOTA) | 0.06 | ~1.0x | ➖平(转发 std;自研曾输 4.4x) |
| **container·str** rfind(needle) 6B(4KB) | 0.12/B | std::string_view::rfind | 0.12 | ~1.0x | string_view(即 SOTA,two-way 向量化) | 0.12 | ~1.0x | ➖平(转发 std;自研 region_eq 曾输 ~4x) |
| **container·str** count(char) 全扫 | 0.008/B | std::count | 0.008 | ~1.0x | std::count(即 SOTA,自动向量化) | 0.008 | ~1.0x | ➖平(AVX2 4累加器 psadbw,统计平局 gap7%«spread10%;原 scalar 输 ~27x) |
| **container·str** utf8 validate+count(2KB) | 0.41(scalar)/0.21(simdutf) | (无 std utf8 count) | n/a | n/a | simdutf validate+count | 0.21 | 2.0x输/1.0x | ➖scalar 参考 2x;simdutf 核(vendored 无条件默认)追平 |
| **container·str** hex_encode 1KB→2KB | 0.45/B | naive 逐字节 | 0.63 | 1.41x | (无单一 SOTA hex 库) | n/a | n/a | ✅赢参考实现(表查找) |
| **container·path** extension / base / directory | 2.3/4.6/10(Win)·1.8/4.3/4.7(Lin) | std::string_view 手写反向扫(纯算法基线,零分配) | 双平台各项 | 双平台**平**算法基线 | std::filesystem 词法 152/169/172(Win) | — | 66/37/17x(Win) | ✅**双平台平算法基线**(证扫描已最优,非更巧算法)+ 碾 filesystem(结构差:零分配 vs 构造 path);extension 2026-07 正向全扫→反向扫遇分隔符停 40→2.3(18x);base Win"1.84x"是MSVC的string_view构造税,Lin平 |
| **container·path** normalize 词法折叠 | 69 | (无零分配 std 词法等价) | n/a | n/a | std::filesystem lexically_normal | 401 | 5.8x | ✅赢(就地折叠 vs 堆 path 对象;PRODUCE 型无 string_view 零分配对手) |
| **container·path** join_array 6 部件(无 alloc) | 29 | (无零分配 std 等价) | n/a | n/a | std::filesystem operator/= | 618 | 21x | ✅赢(buffer-fill 折叠 vs N 次 path 分配;PRODUCE 型无 string_view 对手) |
| **queue** 单线程 fill+drain | 7.77/7.79 | (std 无 bounded-MPMC) | n/a | n/a | rigtorp/moody | 11.54·11.84/9.48·8.41 | 1.5/1.2x·平 | ✅赢 rigtorp 双平台1.5x;moody Win赢1.23x·Lin平 |
| **queue** MPMC 2P 单环 BoundedQueue | 56.08/55.97 | (std 无) | n/a | n/a | rigtorp/moody | 61.04·57.88/73.90·61.69 | gap3.5-7.5% | ➖双平台平 rigtorp+moody(gap«spread26-29%,15遍) |
| **queue** MPMC 4P 单环 BoundedQueue | 69.26/69.10 | (std 无) | n/a | n/a | rigtorp/moody | 78.41·77.89/68.49·67.01 | gap12-15% | ➖双平台平 rigtorp+moody(原输1.33x,cache-pad+去backoff消除落后;gap«spread23-29%,不判反超) |
| **queue** MPMC 4P ShardedQueue | 25.31/14.60 | (std 无) | n/a | n/a | rigtorp | 78.41/77.89 | 3.27x/5.3x | ✅赢(生产走这条;spread21-43%大=采样不足,取量级) |
| **queue** work-steal owner FIFO/LIFO | 3.93/3.06 | (std 无) | n/a | n/a | crossbeam(Rust) | ~3.6/4.4 | ~1.2x/1.3x | ✅赢(跨语言,谨慎) |
| **queue** work-steal steal | 8.42 | (std 无) | n/a | n/a | crossbeam steal | (跨语言) | n/a | ➖HYLE WsFifo(本族 WSDeque 10.5) |
| **queue** ConcurrentVector push/读 | 4.51/0.93 | (std 无并发 vec) | n/a | n/a | oneTBB | 5.99/1.29 | 1.33x/1.39x | ✅赢 |
| **queue** ConcurrentVector 并发读(4读1写) | 0.37 | (std 无) | n/a | n/a | oneTBB | 0.82 | 2.2x | ✅赢(删冗余 acquire) |
| **sched** 争用 Mutex(15遍双平台) | 15.4(Win)/31.3(Lin) | marl=std::mutex | 19.0/31.2 | — | marl | 19.0/31.2 | Win1.23x/Lin平 | ✅**Win 反超**(SRWLOCK 修,gap22%>spread18%)/➖Lin 平 |
| **sched** 无争用 Mutex(MT) | 6.5(Lin) | std::mutex | 9.1 | 1.40x | marl(=std::mutex) | 9.1 | 1.40x | ✅赢(**单平台单遍,待双平台统计**) |
| **sched** schedule 2000 noop（worker-stack，15遍三平台） | 192.60(Win)/218.44(Lin)/252.12(M4) | marl enqueue（严格4 worker） | 297.53/327.03/252.72 | 1.53/1.48/1.007x | marl | 同左 | 同左 | ✅ Win/Lin 明确赢；M4 统计平局。旧 stackful 混比与 marl caller bind 形状已作废；ARM 全局完成计数争用已修 |
| **sched** WaitGroup fork-join（worker-stack，15遍三平台） | 192.51(Win)/220.77(Lin)/253.98(M4) | marl | 297.79/337.42/252.75 | 1.55/1.51/1.003x | marl | 同左 | 同左 | ✅ Win/Lin 明确赢；M4 统计平局；conditional-notify + per-worker completion accounting |
| **sched** Latch fork-join(64) | 5059(fiber) | std::latch(64线程) | 21975 | 4.3x | marl(无 latch) | n/a | n/a | ✅赢 std(fiber 不阻塞 worker) |
| **sched** CV notify_one 快路径 | 1.70 | std::condition_variable | 2.59 | 1.52x | marl(无独立 CV) | n/a | n/a | ✅赢 std |
| **sched** CV wait+notify handoff | 280 | (无 std fiber 等价) | n/a | n/a | (HYLE 自有) | n/a | n/a | ➖HYLE 自有(阻塞 handoff,TSan 零race) |
| **sched** fiber 切换 | 21–23 | std::thread 切换 | ~1000+ | ~50x | fcontext 原语 | 21–23 | 1.0x | ➖平 fcontext(=原语零封装) |
| **sched** suspend/wake 完成脊柱 | 163(Win)/121(Lin) | (无 std 等价) | n/a | n/a | (HYLE 自有脊柱) | n/a | n/a | ➖HYLE 自有(GPU/NIC/timer 汇流);**LIFO fast-slot 后 Win 240→163(−32%)/Lin 186→121(−35%)**——同 worker 唤醒走 cache-hot 慢槽,跳过 global CAS+notify futex(见 sched 详节) |
| **sched** Channel 收发(cap16,1p1c) | 63/item | (无 std 等价) | n/a | n/a | marl(无 channel) | n/a | n/a | ➖HYLE 自有 |
| **sched** RunQueue fifo/lifo push+next | 8.3/2.6 | (内部结构) | n/a | n/a | (无外部对手) | n/a | n/a | ➖HYLE 自有(完成唤醒热路径) |
| **sched** remove_sleeper(park 退出) | O(1) | (内部结构) | n/a | n/a | tokio idle.rs(容忍 O(n)扫描) | O(n) | 复杂度赢 | ➖复杂度机会②:per-worker position 索引 swap-remove 消 O(worker数)扫描,缩短 idle_mu 临界区(HYLE 每次 park 退出都调,故值得;tokio 只罕见路径扫故容忍) |
| **sched** global-queue 反饥饿 | 修 bug | (正确性) | n/a | n/a | tokio global_queue_interval=61 | 同款 | 修真 bug | ➖复杂度机会③:探针实证本地 firehose 饿死全局注入真实存在(yield-hog 1700万次而 victim 从不运行),每 61 tick 先查 global 修复(回归测试双向验证 pre-fail/post-49ms) |
| **sched** fiber sleep(sched-04b) | — | (std 无 M:N fiber sleep) | n/a | n/a | Seastar/Tokio time driver（概念对照） | n/a | n/a | ➖SOTA 成本模型：无 sleeper 时 queue.empty() 单比较；arm O(1) 无分配；park 到 exact next deadline |
| **sched** CV wait_for notify-before-deadline round-trip | 580(Win)/~(Lin) | std::condition_variable::wait_for | 1165(Win) | 2.0x | (std 是直接 SOTA) | n/a | n/a | ✅**Win 赢 2.0x**(gap99.8%»spread47%,15遍)/ **Lin 平**(glibc cv 已薄 futex 封装,gap3.4%«spread21%)——又一单平台数陷阱,同 sched-05 Parker |
| **sched** CV wait_for 纯超时(1ms) | 15.58ms | std::condition_variable::wait_for | 15.58ms | ~1.0x | (OS 定时器地板) | n/a | n/a | ➖双平台平(gap0.1%«spread):HYLE 超时路径零额外开销,两者同被 OS ~15ms 定时器粒度地板 |
| **sched** spawn_task+join 每任务(sched-10,批量摊销) | 390(Win)/423(Lin) | 裸 Fiber+WaitGroup(1)+slot | 338/415 | 0.87x/1.02x | (糖 vs 自有裸机制) | n/a | n/a | ➖**层开销 Win ~52ns/Lin 平局**:糖多 2 次堆分配换调用方零生命周期管理;初测单任务 ~1.3x/~2x 是 OS 唤醒抖动非层开销,批量摊销后现原形 |
| **sched** when_all 64 任务(sched-10) | 372(Win)/467(Lin) ns/任务 | (= 逐个 join) | — | ~1.0x | (HYLE 自有) | n/a | n/a | ➖群等待 vs 逐 join 零额外;when_any(4) ~2x 快于 when_all(4)(取第一即返) |
| **io** echo shard-per-core 4w（2026-07-20 收口） | Win 3,937 / WSL 1,138 / M4 6,151 ns | (无 std 异步 IO) | n/a | n/a | 自身 1w | 8,811 / 1,515 / 3,082 ns | **2.24x / 1.34x / 0.50x** | ⚠️ **Win/Linux 扩展赢，M4 4w 真输**。旧 2.2x headline 只属于历史 Linux run；当前三平台证书见 [IO 详节](./benchmarks-io.md) |
| **io** echo 1w（2026-07-20 收口） | WSL 1,515 ns | (无 std 异步 IO) | n/a | n/a | raw epoll | 1,375 ns | **0.90x** | ❌ **当前 Linux 单 worker 比 raw epoll 慢约 10%**；Win/M4 raw peer 未同形重跑，不沿用旧的 asio/libuv headline |
| **io** pipelined 高扇出（2026-07-20 收口） | WSL 1w 3,567 → 4w 1,754 ns | (无 std 异步 IO) | n/a | n/a | 自身扩展性 | n/a | **2.06x** | ✅ 实际完成数校验后的当前 Linux 扩展证据；旧 5.2x 留在详节作优化历史，不作发布证书 |
| **io** 连接生命周期 accept+serve+close(真实短连接) | ~240µs/conn(4162 conn/s) | (无 std 异步 IO) | n/a | n/a | (自身:ephemeral→persistent 谱系) | n/a | n/a | ➖**每连接固定成本 ~240µs(accept+握手+close)主导**,R=1→16 round-trip 仅 +~2.6µs/轮=echo 稳态往返(填 accept_bench 即弃/echo_bench 预建之间的真实使用缺口;正是 keep-alive/连接池要摊薄的那块)。双平台(Win IOCP ~2ms)+ TSan 净 |
| **io** multishot recv awaitable vs one-shot(io-07b) | 165,679 ns/dgram | (无 std) | n/a | n/a | HYLE one-shot RecvOp(自身对照) | 166,985 | 0.998 | ➖**统计平局**(gap0.2%«spread13.4%,n=15):同核 lockstep ~1 op 在途,fiber 往返主导、multishot 拿不到"一批完成同到"来摊薄省下的 submit。**非回归、待承重**(需突发/高扇出负载显优势,登记未来;正确性+TSan+回归已验) |
| **io** DEFER_TASKRUN vs COOP_TASKRUN(io setup flag) | 1.35-1.43µs(1w) | (无 std) | n/a | n/a | COOP_TASKRUN(自身对照) | 1255ns(1w) | ~同量级 | ➖**同量级无显著提升**:DEFER 收益=省完成 task_work 的 IPI,单 worker loopback 本就每次 drain 都 enter ring、几乎无 IPI 可省。**正确接入(R_DISABLED 生命周期修 + kCancelToken latent bug)、不回归、机制就位待高 syscall 压力/多核争用显收益**(opt-in 默认 COOP) |
| **io** registered/fixed files(IOSQE_FIXED_FILE,io-07c) | +0.8-3.3% | (无 std) | n/a | n/a | 裸 fd(自身对照) | 基线 | 1.008-1.033x | ➖**边际正收益**(socketpair ping-pong 3 跑一致为正、淹在 1-3% err% 边缘):省内核每 op fget/fput 是真的,但 loopback 上 fd 查表只占往返极小一块 |
| **io** direct-accept vs normal accept(io-07c) | 合并慢~1%/accept 相快~10-30ns | (无 std) | n/a | n/a | normal accept(自身,拆相分离) | — | accept 段 direct 快 | ➖**非回归**(拆相追根 0.7%问题):**accept 原语本身 direct 反而快 ~10-30ns/conn**(省 fget/fput,合理);合并 bench 那~1%全是 teardown 不对称伪影(direct close_direct+reap+FILES_UPDATE 2 syscall 其一冗余 vs normal ::close 1 syscall),生产不即建即关故不出现。primitives-only 不改 TcpStream |

**历史总账小结（2026-06-30 快照，不作为当前证书）**：当时按两把标尺统计约
36 win / 16 tie，并推动了 allocator、hashmap、SIMD、queue、string-ref 等多轮
优化。此后接口、对手和三平台仪器都已升级，不能继续用这组数量证明当前
“全线赢”。尤其 StringInterner 的旧 1.17–1.24x 证书已被本页当前 direct-ID peer
三平台表替代。发布判断以各模块最新 closure 文档和本页逐行当前数据为准，不再
维护一个会把不同日期、不同合同混在一起的总 win 数。

**io 段补记(2026-07-05,reinject+COOP 优化后 n=15 重测)**:HYLE echo-1w **单核反超成熟抽象框架 asio 1.25x / libuv 1.52x**(优化前是统计平局,-33% 推成真赢)、**平裸 epoll**(零抽象就绪循环,追平即好);多核 shard-per-core 开箱大赢。新增**真实连接生命周期**基准(accept+serve+close):每连接固定成本 ~240µs 主导、每 round-trip 仅 +~2.6µs——量化了 keep-alive/连接池的价值。io 机制项(multishot awaitable / DEFER_TASKRUN / registered-fixed-files / direct-accept)均为**正确接入、loopback 上统计平局或边际**、"就位待承重"——它们的收益需 loopback lockstep 造不出的突发/高扇出/高建连率负载,已诚实登记为待承重(非回归、非缺陷);direct-accept 的"慢 1%"经拆相追根证为 teardown microbench 伪影(accept 原语本身 direct 更快)。
