# error / hash 实测铁证

> 本页是 [performance 总账](./README.md) 的 error + hash + SwissTable 详节。环境:Win MSVC19.50 / Linux GCC14.2 Release,nanobench。`bench/container/` + `bench/error/`。

## error Status/Result（`status_bench.cc`，对标 absl::Status/StatusOr + std::expected）

> 环境:Win MSVC19.50 Release,nanobench。**两个标尺**:① **同款 SOTA = `absl::Status`/`absl::StatusOr`**(HYLE 这套 tagged-pointer + 规范 gRPC 码表正是抄它;gated `-DHYLE_BENCH_ABSL`)——这才是 apples-to-apples"带消息错误载体"这一类的真门槛;② `std::expected<int,std::error_code>` = 标准给的、但 E 平凡(俩 int,**带不了消息**)的下界。此前 error 是零依赖根却唯一无 bench 的模块。

| 操作 | HYLE | absl(同款 SOTA) | std | 结论 |
|---|---|---|---|---|
| **Status::Ok 构造+ok()** | **0.37** | 0.37 | std::expected 0.56 | ✅ 平 absl、赢 std 1.5x(`rep_==0` 单字) |
| **Status(code) 无消息** | **0.37** | 1.64(必堆 rep) | std::unexpected 0.55 | ✅ **赢 absl 4.6x**(HYLE inline 码零堆,absl code-only 也 malloc) |
| **Status(msg) 带消息→堆** | **28.2** | 54.3 | (无等价) | ✅ **赢 absl 1.92x**(单 malloc(rep+len+1),消息内联尾随) |
| **Status 拷贝(heap,refcount)** | **9.4** | 10.6 | (无等价) | ✅ 赢 absl 1.13x(3 指针 vtable refcount) |
| **Status 拷贝(OK)** | 0.59 | 0.57 | (无等价) | ➖ 平 absl(word copy,无 retain) |
| **Result OK 传播 4 层(裸)** | 0.73–0.94 | 0.91–0.94 | std::expected 0.39–0.55 | ➖ 平 absl;输 std=microbench 假象(见下) |
| **Result OK 传播(解包消费,真实)** | **1.42** | ~1.4 | std::expected 1.40 | ✅ **平 std**(真实用法,假象消失) |
| **Result ERROR 上抛 3 层** | **10.2** | 21.5 | (errc 无堆消息不可比) | ✅ **赢 absl 2.11x** |

**核心诚实结论(error 是 SOTA——对标真同侪 absl 全线平或赢,空间最省)**:

1. **"带消息错误载体"这一类里 HYLE ≥ absl 全项**:OK 构造平、inline 码 **赢 4.6x**、带消息构造 **赢 1.92x**、heap 拷贝赢 1.13x、ERROR 上抛 **赢 2.11x**。absl 是 HYLE 这套设计的来源(tagged ptr + gRPC 码表),逐项对标证明 HYLE 实现更精:**inline 错误码零堆**(absl 即便 code-only 也 malloc 一个 rep)是最大结构优势,直接兑现 status.h 的三态编码设计。

2. **Result OK 传播"输 std 1.7x"是 microbench 假象,已用消费变体证伪**:裸 microbench(构造 Result 立即在循环里析构、不取值)HYLE 0.73–0.94ns vs std::expected 0.39–0.55ns。**三条铁证它不是缺陷**:① objdump 看 `hyle_chain_ok` 函数体 = **7 条指令、全内联、零析构调用**(OK 路径本身零开销);② **隔离探针** `expected<int,TrivialErr>`(平凡析构错误类型)穿同样传播 idiom **精确追平 std(0.39=0.39)**——证明差距来自 `Status` 的非平凡析构(它要能 OWN 堆消息 → 析构带 `is_heap()` 分支;`std::error_code` 平凡因它带不了消息),不是宏/idiom;③ **同款 SOTA absl::StatusOr 付一模一样的税(0.91–0.94,与 HYLE 并排)**——这是"可携带堆消息的错误类型"整类的固有代价,SOTA 同付。**改用真实用法(`ASSIGN_OR_RAISE` 解包 + 消费 int)后 HYLE 1.42ns 平 std 1.40ns**——紧循环里反复构造-析构临时量才会把那一个析构分支放大,真实代码取值用掉后分支与后续工作交织即消失(同 util/Ref 条目记录的同类 microbench 假象)。验证过 static 构建(`status_release` 同 TU 可内联)仍 0.95ns,**证伪了"DLL 接缝阻止内联"假设**——确认是 std::expected 对非平凡 E 的语言规则,非接缝。

3. **空间维度(时间/空间冲突时优先时间,此处无冲突反而双赢)**:`sizeof(Status)=8B`(单 tagged word,= absl::Status 的 8B)、`Result<int>=Result<void>=16B` **< std::expected<int,errc> 24B、< absl::StatusOr<int> 24B**。HYLE 错误载体在两个维度都 ≤ SOTA:Status 与 absl 同宽且功能等价,Result 比 std/absl 省 1/3(8B Status vs std 的 16B error_code / absl 的内嵌 Status+对齐)。

## container·map —— SwissTable hashmap

| 操作 | vs `std::unordered_map` | vs `absl::flat_hash_map`(**SOTA 标尺**) | 结论 |
|---|---|---|---|
| find-hit | **2.42x** 赢 | **~1.10x 赢**(12 遍 avg 110.1%;prefetch+mix64+H1,曾 0.76x) | **反超 SOTA** |
| find-miss | **2.26x** 赢 | **≈1.0x 平/微胜**(最新 15 遍 4.41 vs 4.86,1.12x,gap«spread) | **追平 SOTA(见下 profile 收口)** |
| insert | **5.3x** 赢 | **~1.40x 赢** | **超 SOTA** |
| erase | **3.5x** 赢 | **~1.39x 赢** | **超 SOTA** |
| iterate(for_each) | **8.7x** 赢 | **2.0x 赢**(0.96 vs 1.94) | **反超 SOTA(2026-07 组扫)** |

> **2026-07 补 iterate 对标 + 组扫优化(揪出真输并反转)**:此前 iterate 只对 std(赢 3.6x)、absl 标"未测此项"——补 absl 对标**揪出旧 for_each 逐字节扫(load ctrl_[i]+每 slot 分支)输 absl 迭代器 1.2x**(2.32 vs 1.94)。改**组扫**:每 kGroup 字节一次 SIMD `match_full`(SSE2 `movemask^0xFFFF`)得占用 lane 位掩码,只迭代 set bit(无每 slot 分支),= absl 迭代器形状。**iterate 2.32→0.96ns=2.4x,反超 absl 2.0x、赢 std 8.7x**——组扫 + `for_each(fn)` 模板无迭代器对象/校验开销双重优势(absl range-for 每步构造迭代器)。加 `Group::match_full()` + `ForEachVisitsExactlyLiveSet` 测试(空/插/删 tombstone/growth,守卫组扫;此前 for_each 零直接测试)。双平台 container 148/148。**教训同 dtype/dot:"只赢 std 未对 SOTA"会掩盖真输——补对标才暴露,补上就反超。**
> **find-miss profile 待办收口(2026-07)**:历史留的"find-miss ~10% 待 profile"已过时——最新 15 遍 SwissTable 4.41 vs absl 4.86 = **HYLE 微胜 1.12x(gap«spread,统计平局偏赢)**,那 ~10% 差距在后续 mix64/H1/组 baseline 优化后已消失,HYLE 现不落后 find-miss,profile 待办关闭。
> **x86 SOTA 复核重跑(2026-07-12,WSL GCC-14 + 本地 abseil-cpp)**:重新配 `-DHYLE_BENCH_ABSL` 真跑 3 遍坐实上表仍成立——find-hit 108-118%、find-miss 99-103%(统计平)、insert 148-151%、erase 148%、iterate 212%。**x86 SIMD 探测(16-wide SSE2 `_mm_cmpeq_epi8`+movemask)= abseil 同款且 find-hit 反超**(edge 来自 identity-H1 免 shift + prefetch + 全内联 match,非探测指令差异)。probe.h 三套 group(SSE2-16/NEON-8/SWAR-8)、H2=hash>>57、控制字节值、三角探测、tombstone reclaim、7/8 load 全对齐 abseil,是忠实 SOTA 移植。
> **✅ NEON pack 向量化(2026-07-12 f152354,复核揪出的 ARM 缺口已修)**:原 `GroupNeon::pack()` 用 **8 次标量循环**把 `vceq_u8` 的 8 字节 mask 折成 8 位(为让三套 ISA 共用一个 1-bit-per-lane `BitMask`),是 probe 热路径每-match 开销——abseil `GroupAArch64Impl` 则用 Shift=3 字节-mask 零打包。**已改**为 HYLE simd `neon.h` 同款向量化 movemask(AND 权重 `{1,2,4..128}` + 一次 `vaddv_u8` 水平加,2 条 SIMD op 替标量循环),保持单一 BitMask 契约不变(零消费者改动),而非引入 Shift=3 变体。**Mac Apple Silicon 实测:新旧输出逐位相同(等价确认),向量化 match 比旧标量 pack 快 3.9-5.8x(0.26-0.50 vs 1.0-2.9 ns/match)**——直接摊进 ARM find/insert/erase。container_test 158/158 真机过;x86 arch-gated 不受影响。(ARM 仍无 absl clone 做端到端 vs-absl,但 x86 已坐实同款探测 SOTA,ARM 侧此修消除了唯一已知结构落后点。)
> **hash 层提升是否反哺 map(2026-07 评估)**:SwissTable 默认 `Hash<K>`,所以 hash 层优化按 **key 类型**自动流入。① **整数/指针 key**(如本表 uint64 bench):走 `mix64`,本轮**未动**(它早是 abseil 单乘折叠、已 SOTA)——故 uint64 map 数字不变,符合预期,非遗漏。② **9-16B POD 复合 key**(如 `{u64,u64}`):走 `hash_small`,本轮 `mix64_2`(单乘替两串行乘)使其 hash 1.38x,map 用此类 key 时 find/insert 直接受益。③ **长 string key**(>128B):走 best-of-breed wyhash,map 用长串 key 时受益。**结论**:反哺是"按 key 类型条件生效"的——最热的整数 key 本就 SOTA 无可反哺,复合/长串 key 才吃到本轮红利;无需额外改 map 代码(hash 是模板参数,自动继承)。

> **2026-06-28 原生 ext4 复测(每步跑 3 遍,非噪声)——两刀优化,学自 absl**:
> 1. **probe 预取**(`HYLE_PREFETCH` slot,镜像 absl `find_large` 的 `PrefetchToLocalCache`):find-hit 0.76x→**0.88x**(slot load 藏进 ctrl SIMD match);find-miss 不变(miss 不碰 slot)。
> 2. **mix64 换 absl 式 128 位乘折叠**(`(x^k)*k → hi^lo`,单乘短依赖链 vs splitmix64 的 2 串行乘):find-hit **0.88x→1.10x 反超**、find-miss **0.78x→0.92x**(hash 占 miss 成本大头,短链直接砍)。代价:整数 hash 从双射(零碰撞)降为统计级(~1/2⁶⁴),correctness 由 eq_ 兜底;insert/erase **未退化**(1.35x/1.32x)证明碰撞质量没崩。
> - 双平台真编真测:Linux 133/133(含雪崩/分布质量测试)+ Win MSVC `_umul128` 分支编译干净 + 测试过。`__extension__` 消 `__int128` 的 -Wpedantic。
> 3. **match_empty 加 SSSE3 快路径**(`_mm_sign_epi8(c,c)` 1 指令 vs SSE2 重建 kEmpty 广播 4 指令,`HYLE_HAVE_SSSE3` gate + SSE2 兜底):**绝对增益/与 absl ISA 对等**,但**未缩 find-miss 相对差距**——`-mssse3` 下 absl 也用 `psignb`,两边同步提速,find-miss 仍 ~0.89x。**证伪了"match_empty 是 find-miss 差距来源"**。双 ISA 输出穷举验证一致(psignb 确认)+ SSSE3 激活下 134/134。注:SSSE3 现为全局 baseline(CMake 对 x86 GCC/Clang 全局加 `-mssse3`、MSVC x64 无条件),`HYLE_HAVE_SSSE3` 默认成立、快路径默认生效。
> - **find-miss 剩 ~10% 已排除三因**:非 hash(mix64 对齐)、非 prefetch(miss 不碰 slot)、非 match_empty(同 ISA 仍差)。**根源在别处**(probe_seq codegen / H1 的 `>>7` / BitMask 迭代),需逐指令 profile,留作 profile 引导任务。
> 4. **H1=identity**(objdump 逐指令对比发现:absl 用 hash 低位直接当 index、top 7 位当 H2,省掉我们的 `hash>>7` 一条移位):find/insert/erase 内循环各省 1 指令。find-miss **0.89x→~0.96x**、insert/erase 升到 ~1.40x。正确性:mix64 雪崩后低位(index)与高 7 位(H2)独立,探测链不变;SSE2 基线 134/134 + Win MSVC 通过。
> - **12 遍统计收口(回答"4% 差在哪")**:find-hit avg **110.1%**(min 102.5/max 119.4)、find-miss avg **97.5%**(min 94.9/max 101.2,**spread 6.3% > 差距 2.5%,12 遍有 2 遍赢**)、insert **146.6%**、erase **140.0%**。**find-miss 是统计平局**——那"4%"是测量噪声(微基准 ~2-4ns,find-hit 自身 spread 都 16.9%),非结构缺陷,无单点杠杆。**四项全部 ≥ 或追平 absl,SOTA 反例消除。**

**这就是"对标 SOTA"的意义**:只对标 std 会得出"全面 2-5x 碾压"的虚假安心;对标 absl 才暴露 **find 有 ~25% 真实差距**。**已排除三个错误候选**(读码查实):① `eq_` 对 uint64 在 -O3 完全内联成裸 `==`,**无间接可去**;② `Hash<uint64>` 已是内联 `mix64`(splitmix64),**非 XXH3**,hash 不慢;③ **控制字节尾部镜像已实现**(`swiss_table.cc` core_set_ctrl 镜像前 kGroupWidth 字节 + 分配 `cap+kGroupWidth` + memset 尾),group near-end 加载已无回绕掩码。**find 循环结构已几乎等价 absl**——25% 来源**无法从源码结构看出**,需 **profile(perf/objdump 对比两个 find 内循环 codegen)** 定位真因;唯一已知结构差异是 absl 在 probe 起点 **预取槽数组**(100k 表常驻 cache 时收益小,待测)。**结论:这是一个需要 profile-guided 的真优化,非猜改**——留作下一步 container 性能待办。insert/erase 反超 absl 证明设计本身到位。

## container·map —— hash 哈希（`hash_bytes`/`mix64` vs 裸 XXH3 + wyhash + std，`hash_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,15 遍中位数。三组:HYLE / 裸 XXH3+wyhash(SOTA) / std::hash。XXH3 vendored 无条件,wyhash gated `-DHYLE_BENCH_WYHASH`。**2026-07 红队修 bench 公平性 + 改 abseil 单乘**——数字与结论已重做,见下。
>
> **⚠️ bench 公平性更正(2026-07,红队 hash)**:旧 bench/ledger 有三处误导,已修:① **≤16B 行标签 "XXH3 wrapper" 错**——`hash_bytes(len≤16)` 是 `HYLE_FORCE_INLINE` 且短路走 **inline mix(mix64 折叠)根本不调 XXH3、不跨导出 seam**;所以 8B/16B 是 HYLE-inline-mix vs **完整 XXH3**(两个不同算法),旧结论"输裸 XXH3 0.86x/0.67x"是拿 inline-mix 比 XXH3-small,非真输。**真正的 seam/wrapper 税只在 64B/256B**(那里才是同算法 XXH3 vs XXH3)。② 旧 bench **单固定串重复 hash 百万次**→数据常驻 L1、长度分派分支完美预测,小串读出亚周期(0.03ns 假地板);已改**每长度轮转 256 个随机串**(smhasher 式),8B 从假地板 0.03→真实 0.23ns。③ `batch(len)`=ns/byte,小串固定开销主导,per-byte 低≠算法快。

| 字节哈希(ns/op per byte,15 遍) | HYLE | 裸 XXH3 | wyhash | std | 结论(算法归属) |
|---|---|---|---|---|---|
| 8B | 0.23 | 0.21 | 0.30 | 0.29 | **inline mix**(非 XXH3):赢 wyhash-small、近平 XXH3-small |
| 16B | 0.15 | 0.12 | 0.15 | 0.35 | **inline mix**(非 XXH3):平 wyhash、近平 XXH3-small |
| 64B | 0.055 | 0.043 | 0.052 | 0.45 | **真 XXH3 wrapper**:同算法 vs 裸 XXH3 ~1.28x 接缝税(出线核不可内联) |
| 256B | 0.082 | 0.075 | **0.029** | 0.63 | 平裸 XXH3(接缝);**wyhash 碾 XXH3 2.6x=SOTA 观察** |

| 整数哈希(4096 key) | HYLE mix64 | wyhash64 | std::hash | 结论 |
|---|---|---|---|---|
| ns/op | **0.44** | 0.57(赢 1.27x) | 1.50(赢 3.4x) | ✅ mix64 已 SOTA(= abseil 同款单 128 位乘折叠) |

| 小 POD key(16B 结构,4096,真热路径) | HYLE Hash<T> | wyhash | 结论 |
|---|---|---|---|
| ns/op(改 mix64_2 前→后) | 0.595 **→ 0.43** | 0.94 | ✅ **改 abseil 单乘后 1.38x,vs wyhash 从 1.6x→2.2x** |

**结论(bench 已诚实化 + 9-16B 改 abseil 单乘 + SOTA 观察)**:
1. **9-16B key 改 abseil 单 128 位乘(mix64_2)——16B POD key 热路径 1.38x**:HYLE 原对 9-16B 用**两次串行 mix64**(各一个 128 位乘);abseil `CombineContiguousImpl9to16` 只用**一次** `Mix(a,b)=fold(a*b)`(两 word 进单乘的两操作数)。加 `detail::mix64_2` 照此改,16B POD key **0.595→0.43ns(1.38x)、vs wyhash 赢 2.2x**(原 1.6x)。短依赖链(1 乘替 2 串行乘)的真收益;质量经 `SmallStructInlineQuality`(40000 distinct + 每 bit 雪崩≥12)+ StringView 10k 无碰撞门(10/10 Hash 测试双平台过)。已知 abseil 文档的 1/2^64 corner(操作数恰 0,仅 16B 可达,XOR 常数+重叠字节 practically 不可达)。
2. **≤16B "输 XXH3" 是伪命题(两算法比)**:纠旧结论——`hash_bytes(≤16)` 走 inline mix 非 XXH3,拿它比完整 XXH3 不是同物;正确对手是 wyhash-small(HYLE 赢/平)。旧文档"接缝代价"归因也错:≤16B 全内联无 seam,**真接缝税只在 >16B 的 XXH3 出线核**。**64B 那 0.78x 已实测隔离出根因(2026-07)**:① `XXH3_64bits_withSeed(64B,0)` vs plain `XXH3_64bits(64B)` 同 TU 全内联时 **1.01x 同速**——排除"withSeed 是不同算法"这个候选;② 同一 XXH3 经 HYLE 导出核 `hash_bytes_xxh3`(`HYLE_CONTAINER_API` 不可内联,DLL/`.so` hop):inline 1.9ns → 导出 2.9ns = **+1ns 纯调用边界税**(函数调用 + 跨 seam 不内联 + 共享库 PLT/import 间接)。这是**故意的设计取舍**——出线核让 7490 行 xxhash.h 不进任何 consumer TU(编译期 + ABI 洁净),换约 1ns 固定开销;map 定长 key 走 ≤16B inline 路径不付此税。**这 ~1ns 是固定量,随串长摊薄**:64B(单 hash ~2.5ns)时占比大→可见 ~1.28x;**256B(单 hash ~12ns)时已摊薄到统计平局**(Linux 15 遍 ratio1.02 gap2%«spread13%,导出核甚至因独立 TU 的 codegen 偶尔反超)。旧 ledger "256B 0.91x 输" 是 Win 单遍噪声,实为平局。**短串 seam 逃生口已提供(2026-07,应用户要求)**:opt-in header `hyle/container/hash_inline.h` 的 `hash_bytes_inline()` 把 hasher 内联进调用者 TU(无出线调用=无 seam),bit-identical `hash_bytes`(`InlineVariantBitIdenticalToOutOfLine` 测试验全长度带+多 seed)。**代价=该 TU 吃 xxhash.h/wyhash.h 的编译期开销**,故 opt-in per-TU、默认 `hash.h` 绝不 include 它(隔离不破)。只对短串(17~128B)热点有意义:≤16B 默认已内联无 seam、长串已摊平。**两个设计决策(记以防复问)**:① **不违反 C-ABI 分层**——`hash_bytes_inline` 是 header-inline、非导出符号(无 `HYLE_CONTAINER_API`),不参与 C-ABI;默认 `hash_bytes` 仍是导出/DLL 可调/隔离 xxhash.h 的稳定符号。inline 变体要求**源码内嵌/静态链接**(把 xxhash.h 编进调用 TU),纯 DLL/C-FFI 消费者用默认 `hash_bytes`。这是分层(默认 C-ABI 洁净 + opt-in 源码内嵌换极致),非破例。② **只做一个全长度函数,不按长度拆对称 API**——≤16B inline 变体与默认字节级同码(零差别)、长串 seam 已摊平(省 ~1ns 噪声级),真收益只在 17-128B;为"对称"给每长度带配 inline 入口=制造与默认相同的冗余 API + 选择困惑,违"不为对称加复杂度"。用户按"要不要消 seam + 能否源码内嵌"决策,非按长度。usage-guide §3.2 讲清何时用。
3. **mix64 整数哈希已 SOTA**:与 abseil `Mix` 逐字同款(单乘折叠),赢 wyhash64 1.27x(后者两次 `_wymum` 更长链)、赢 std 3.4x。SwissTable 整数键热路径的哈希基础扎实。
4. **best-of-breed byte hasher:按长度选 XXH3/wyhash(2026-07 落地,256B 2.8x)**:长度谱实测(15 遍 median,per-hash)定出交叉点——**XXH3 短中串(17-128B)赢 wyhash 1.1-1.26x,wyhash 长串(≥192B)赢 1.45-2.6x**(XXH3 在 256B 附近有 midsize-path 劣化谷)。既然两者质量同档(都过 SMHasher,wyhash 是 Go1.17+/Zig/Nim 默认),就"谁快用谁":vendor wyhash(`third_party/wyhash`,单头 295 行,public domain)出线核 `hash_bytes_wyhash`,阈值 `kWyhashThreshold=128`(保守,tie 点归 XXH3 侧,wyhash 只接管明显赢的区间)。**两条分派各用其所**:① `hash_bytes(ptr,len)` 变长——`len` 运行时才知,用**运行时分支**(≤16 inline / ≤128 XXH3 / >128 wyhash,一条可预测 if,真"热切换");② `Hash<T>` 定长——`sizeof(T)` 是编译期常量,用 **`if constexpr` 三档全展开**,每 `Hash<T>` 实例化只编译进一个 hasher、**零运行时分支**。**实测 256B 0.082→0.029ns=2.8x**,精确追平 raw wyhash;64B 仍走 XXH3(0.060,不误切)。按长度切换是确定性的(hash 不跨进程持久化,同 build 内一致即可)。质量:新增测试覆盖 wyhash 长串路径(10k 无碰撞 + seed 敏感)+ 128/129 边界确定性;container 双平台 146/146。**真实收益边界**:HYLE 内部 >16B hash 的调用者是 string 驻留(string_buf)+ string_ref map key,多为短中串(XXH3 侧),长串是冷门——best-of-breed 主要是"谁快用谁"的正确性与未来长串负载兜底,当前热路径受益有限。
5. **未被 mem/simd/util 加速(确认)**:hash 不分配(mem 无关);批量 SIMD hash 需 64×64→128 高位乘,AVX2 无(要 AVX-512 `vpmullq`),本机不划算=新特性非漏接;util 的 `mul128_hi_lo` 与 hash 的 128 位乘语义略异(hi+lo out-param vs hi^lo 折叠),可后续收敛为单一 helper(纯代码复用,非性能)。
