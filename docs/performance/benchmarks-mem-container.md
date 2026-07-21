# container 实测详表（+ 历史夹带）

> **导航 / 新鲜度 / 非 SOTA 清单 / 选型** → **[benchmarks-container.md](./benchmarks-container.md)**（performance 入口）。  
> **mem 分配器** → [benchmarks-mem.md](./benchmarks-mem.md)。  
> 本页保留 **分面详表 + 优化史**；与入口摘要冲突时：新鲜度以入口 §0 为准，数字以更新的实测为准。  
> runtime hasher 在 `hyle/container/hash.h`（不在 util）。

## container·hash —— `hash_bytes` / `mix64` / `Hash&lt;T&gt;`（`hash_bench.cc`）

> 归属：`include/hyle/container/hash.h` + `src/container/hash.cc`。  
> 环境:Win MSVC19.50 Release,nanobench,15 遍中位数（**⏳ 非 2026-07-20 三平台口径，待本轮 container 刷新**）。三组:HYLE / 裸 XXH3+wyhash(SOTA) / std::hash。XXH3 vendored 无条件,wyhash gated `-DHYLE_BENCH_WYHASH`。

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

| 小 POD key(16B 结构,4096,真热路径) | HYLE Hash&lt;T&gt; | wyhash | 结论 |
|---|---|---|---|
| ns/op(改 mix64_2 前→后) | 0.595 **→ 0.43** | 0.94 | ✅ **改 abseil 单乘后 1.38x,vs wyhash 从 1.6x→2.2x** |

**结论(历史快照摘要)**:① 9-16B 改 abseil 单乘 `mix64_2` 热路径 1.38x；② ≤16B “输 XXH3”是伪命题；真 seam 税在 &gt;16B 出线核，opt-in `hash_inline.h` 可消；③ mix64 整数哈希已 SOTA；④ best-of-breed 按长度 XXH3/wyhash。完整推演见 git 历史中旧 `benchmarks-error-hash.md` 长注；**刷新时以三平台 bench-stat 重写本表**。

## container·map —— SwissTable（`swiss_table_bench.cc` / `swiss_vs_absl_bench.cc`）

> 归属：`include/hyle/container/swiss_table.h`。对标 `absl::flat_hash_map`（`-DHYLE_BENCH_ABSL`）。**⏳ 下表多为 2026-07 前后 Win/WSL 快照，非本轮三平台证书。**

| 操作 | vs `std::unordered_map` | vs `absl::flat_hash_map`(**SOTA 标尺**) | 结论 |
|---|---|---|---|
| find-hit | **2.42x** 赢 | **~1.10x 赢**(12 遍 avg 110.1%;prefetch+mix64+H1,曾 0.76x) | **反超 SOTA** |
| find-miss | **2.26x** 赢 | **≈1.0x 平/微胜**(最新 15 遍 4.41 vs 4.86,1.12x,gap«spread) | **追平 SOTA** |
| insert | **5.3x** 赢 | **~1.40x 赢** | **超 SOTA** |
| erase | **3.5x** 赢 | **~1.39x 赢** | **超 SOTA** |
| iterate(for_each) | **8.7x** 赢 | **2.0x 赢**(0.96 vs 1.94) | **反超 SOTA(2026-07 组扫)** |

> 优化史要点：probe 预取、mix64 单乘折叠、H1=identity、SSSE3 match_empty、for_each 组扫、NEON pack 向量化。find-miss 旧 “~10% 待 profile” 已关。hash 层按 key 类型反哺 map（整数 key 本就 SOTA；复合/长串 key 吃 mix64_2 / wyhash）。

## container·vec / ·list —— SmallVector / IntrusiveList / CArray（vs std + 真 SOTA，`container_misc_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。声明承重容器。**标尺升级**:不止 std——SmallVector→`absl::InlinedVector`(small-vector SOTA,gated `-DHYLE_BENCH_ABSL`)、IntrusiveList→`boost::intrusive::list`(侵入容器 canonical SOTA,gated `-DHYLE_BENCH_BOOST`)、CArray→`std::vector`(动态数组的 drop-in SOTA,本身已是高度调优,always-on)。

| 场景 | HYLE | std | 真 SOTA(gated) | 结论 |
|---|---|---|---|---|
| **小 N churn**(建+填 8 int+析构) | SmallVector<int,16> **0.41 ns** | vector 16.8 / vector+reserve 3.06 | **absl::InlinedVector 0.84–1.0** | ✅ **赢 absl 2.0–2.4x**(absl 侧 spread 大)、赢 std 41x |
| **list churn**(push 4096 + pop 全) | IntrusiveList **3.15 ns** | list 25.0 | boost::intrusive **2.77**(平) | **8x 赢 std、平 boost SOTA**(2026-07 实测) |
| **动态数组 growth**(push_back 4096 int,无 reserve) | CArray<int> **0.51 ns** | vector **0.75** | (std::vector 即 SOTA) | ✅ **赢 std::vector 1.48x**(grow 用 reallocate) |
| **动态数组 read**(顺序求和 4096) | CArray<int> **0.18 ns** | vector 0.18 | (std::vector 即 SOTA) | ➖ **平 std::vector**(arr[i] 单 load 兑现) |

**诚实结论**:① **SmallVector 反超 absl::InlinedVector 2.0–2.4x**——不是"只赢 std":absl::InlinedVector 是 small-vector 的 SOTA(同 inline 存储设计),HYLE 0.41ns vs absl 0.84–1.0ns(absl 侧 nanobench 标 unstable、spread 大,故取范围),小 N churn 真赢同款 SOTA(也赢 std::vector 41x);溢出路径同样赢 absl 2.7x。② **IntrusiveList 8x 赢 std::list + 平 boost::intrusive SOTA(2026-07 实测,补 ⏳待补)**:零节点分配。checkout boost submodule(intrusive+config/assert/core/move/static_assert 依赖闭包)后实测:push/pop churn HYLE 3.15 vs boost 2.77(ratio 0.88,gap 12%«spread 44%=**统计平局**)、任意位置 O(1) remove HYLE 3.13 vs boost 3.15(平)。两者同 **circular-sentinel 双向链**形状(canonical intrusive),无抄漏——HYLE `unlink` 额外清 prev/next=nullptr 是 unlink-安全特性(支持 `linked()` 查询 + 安全重复 unlink),非性能差。list 项无可反哺的前轮优化(纯指针操作,不涉分配/hash/SIMD),测试项已覆盖 push/pop/front/back/任意 remove/empty 六种 + 两 bench 场景(churn + 乱序 unlink)。③ **CArray 反超 std::vector ~1.48x growth、平 read**:growth 赢的**真根因(2026-07 纠错)**——旧文档称"经 `reallocate()` mimalloc 原地扩展、无 copy"是**双重错**:① 当时代码根本没用 reallocate(是 `allocate`+`memcpy`+`free`,与 std::vector 同形);② 几何增长(4→8→16…)常跨 mimalloc size-class,**即便用 reallocate 也多是 relocate+copy,in-place 很少触发**。真正赢 std 的因是 **mimalloc 分配器质量**(比 std::vector 的 `operator new` 快)+ header-before-data 零索引开销,不是"原地无 copy"。现已把 grow **真改用 `mem::reallocate`**(兑现文档、且是更简 API：≤ 手动路径,偶尔省一次 copy),实测 growth 0.53→0.51ns、1.34x→1.48x(边际,证实 in-place 少触发);read 路径 arr[i] 单 load 与 vector 持平(header-before-data 布局零索引开销)。

## container·str —— string_ref / StringBuf / string_algo（`string_ref_bench.cc` / `string_buf_bench.cc` / `string_algo_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,~4KB 文本(L1 常驻),历史 15 遍统计(`bench-stat.py`)；2026-07-19 container 收口又跑一遍完整当前二进制。**标尺 = `std::string_view`**(string_ref 是其 drop-in 替代,conventions §7:同 16 字节 POD + null_terminated 位,同契约对照公平)+ `std::count` / simdutf。旧标题“container·str 无真输、全线追平或反超”已经撤回：当前 4KB 点值里 split/rfind/count 有 **0.83–0.96x**，而 64KB/1MB 回到约 1.0–1.14x；短输入的 DLL/dispatcher seam 是真实产品边界，方差只能解释一部分。分两条腿走(2026-07 定策):HYLE **自研** find(char)/rfind(char)/count 和 ASCII lower/upper 的 scalar/SSE4/AVX2/AVX512/NEON 运行期 kernel；对 MSVC STL 手写向量化汇编实测更强的 substring/成员族，**转发 `std::string_view`**。公开头不引 std string 头。

### 2026-07-19 收口复测（当前接口形状）

比值大于 1 表示 HYLE 更快。三平台均为 Release、nanobench 3 repetitions
aggregate；macOS 未装 Abseil 的行使用同分片 std peer。

| 场景 | Windows | WSL GCC 14 | Apple M4 | 判定 |
|---|---:|---:|---:|---|
| 单次 intern / thread-safe std peer | **0.994x** | **0.772x** | **0.934x** | Win 平；WSL/M4 明确输 |
| `intern_many` / unlocked std floor | **0.990x** | **0.732x** | **1.045x** | Win 平、M4 小赢；WSL 输 |
| 1KiB unique growth / flat-set floor | **5.36x Absl** | **1.71x Absl** | **1.67x std** | cached hash 的复杂度优势三平台兑现 |
| 8-thread adaptive shards / strongest peer | **0.948x Absl** | **0.958x Absl** | **0.956x std** | 距强 peer 约 4–5%，不是无条件赢 |
| add/retain/remove / direct-ID+mutex peer | **0.990x** | **0.813x** | **0.785x** | Windows 平；Linux/M4 generation+slab 常数可见 |
| 16-cycle churn / retained peer | **0.998x** | **0.861x** | **0.808x** | 高水位复用成立，但非三平台常数 SOTA |
| find/substring/set-search/starts_with，4KB | 0.93–1.00x | ≈1.0x | ≈1.0x | 统计近似平；不宣传赢 |
| split/rfind/count，4KB | 0.83–0.96x | ≈1.0x | ≈1.0x | 短输入 DLL/dispatcher seam 可见 |
| count/rfind，64KB–1MB | 0.995–1.14x | ≈1.0x | ≈1.0x | 长输入摊平 |
| UTF-8 validate+count / simdutf | **1.004x** | — | — | 平 |
| ASCII lower / scalar branch | **1.52x** | — | — | 赢 scalar reference |

这轮不是只换报告口径：StringBuf 改成 16/32/64 自适应 shard、cached resident
hash、generation+shard+slot ID、短 entry slab、永久 intern 的 lazy ID、四类 batch
API；hash 用中间位选 shard，保留 Swiss H1/H2 熵且省掉第二次 avalanche。
`intern/add` 为预期 O(|input|)，growth/erase 不重扫 resident string；handle
lookup/retain/remove 为预期 O(1)，`size()` 只扫硬上限 64 shard。旧表保留为历史
优化证据，不再作为当前无条件证书；Linux/M4 lifecycle 的 14–22% 落后是已知冻结
边界，只有同契约真实 workload 再次证明它是瓶颈才重开。

| op | HYLE | std::string_view | vs SOTA | 结论 |
|---|---|---|---|---|
| **find(char) 全扫 miss** | 0.02/B | memchr 0.01 | ~1.0x | ➖平(自研 memchr;原 scalar 输 ~20x) |
| **starts_with(16B)** | 1.79 | memcmp 1.77 | ~1.0x | ➖平(自研 memcmp) |
| **split 全文按空格** | 0.44/B | find+remove_prefix 0.54 | 1.22x | ✅赢(自研,无分配子视图) |
| **trim 两端** | 13.6 | 16.85 | 1.24x | ✅赢(自研,组合 find_first/last_not_of) |
| **rfind(char) 后向扫** | 0.012/B | 0.012 | ~1.0x | ➖平 std(自研 vec.h 后向扫 FindLastTrue;**arm NEON 8x 赢原 scalar**;原 scalar 输 ~20x) |
| **count(char) 全扫** | 0.008/B | std::count 0.008 | ~1.0x | ➖平 std(自研 vec.h 4累加器 SumsOf8;**arm NEON 3.7x 赢原 scalar**;原 scalar 输 ~27x) |
| **find(needle) 6B 子串** | 0.08/B | 0.08 | ~1.0x | ➖平(转发 std;自研 memchr 候选曾输 2.5x) |
| **rfind(needle) 6B** | 0.12/B | 0.12 | ~1.0x | ➖平(转发 std;自研 region_eq 曾输 4x) |
| **find_first_of(4 字符集)** | 0.05/B | 0.05 | ~1.0x | ➖平(转发 std;自研 memchr-min 曾 0.62x、PCMPESTRI 0.76x) |
| **find_first_not_of 全字母** | 0.06/B | 0.06 | ~1.0x | ➖平(转发 std;自研 bool 表曾输 4.5x) |
| **find_last_of(digits)** | 0.06/B | 0.06 | ~1.0x | ➖平(转发 std;自研曾输 4.4x) |
| **StringBuf intern**(100k/2k 唯一) | 28.86(Win) | map+mutex 同契约 peer 28.70 | 0.994x | ➖Win 平；WSL 0.772x、M4 0.934x，见上方当前三平台表 |
| **StringBuf add/retain/remove churn** | 18.62(Win) | direct-ID+mutex 同契约 peer 18.44 | 0.990x | ➖Win 平；WSL 0.813x、M4 0.785x，旧 1.24x 证书撤回 |

**历史优化路径（当前结论以上述收口复测为准）**:
- **自研追平的项(HYLE 自己的 SIMD)**:`find(char)`/`starts_with` 早已切 memchr/memcmp;`rfind(char)`/`count` 走运行期 ISA dispatch。**2026-07 统一到 `simd/vec.h`**:count/rfind 的核从手写裸 intrinsic(`string_ref_avx2.cc` 256b + inline SSE2,**x86-only**)改写为 `str_search_{scalar,sse4,avx2,avx512,neon}.cc`(共享体 `str_search_impl.h`,写在 D-tag 抽象上),`string_ref.cc` 按 `best_isa()` 经缓存 fn 指针选最宽核——与 `simd/ops` 同变体-TU 派发模型。收益:**一份源码覆盖全 ISA,arm64 NEON + AVX-512 白得**(此前 arm 落 scalar)。为保住手写版的 psadbw 4 累加器计数技巧,给 vec.h 补了三原语 `SumsOf8`(SAD)/`VecFromMask`/`FindLastTrue`。**回归实测(WSL,apples-to-apples 两边都过 .so)**:新 vec.h 核 vs 旧手写 AVX2 核 **打平**(count 96-101%、rfind 99-100%,gap«spread;早先看到的 3-11% 是"手写核可内联进 bench TU、.so 调用不可内联"的测量不对称,非真回退——`__attribute__((noinline))` 对同一份核复现出同样 gap 坐实)。**arm 真机(Apple Silicon,vs 重构前 arm 落的 scalar 字节循环)**:count **3.7x**、rfind **8.0x**(4KB/64KB/1MB 全稳)。`split`/`trim` 靠零分配子视图**反超** std。
- **转发 std 的项(借 SOTA,不硬造)**:`find(needle)`/`rfind(needle)`/`find_first_of`/`find_first_not_of`/`find_last_of`/`find_last_not_of`——MSVC STL 的 two-way 向量化子串搜索 + PCMPESTRI 字符集扫描是实测 SOTA,**自研 PCMPESTRI 尝试只到 0.76-0.78x 追不动**(PCMPESTRI 延迟高),遂转发 `std::string_view`。转发局限在 `string_ref.cc`(公开头不引 `<string_view>`),不破 C-ABI。**代价**:string_ref 在这些项上不是"纯自研替代",而是"借 std 的最优实现"——这是刻意取舍(不为纯粹性造打不过微软手写汇编的轮子),非缺陷。**结果:5 项从输 2.5–4.5x 全部统计平局(ratio 0.98–1.02,gap«spread)。**
- **前身 clay 的做法参照**:clay 的 `string_ref` 搜索族**全部**转发 `std::string_view`(且其头 `#include <iostream>`/`<string>`/`<functional>`)。HYLE 只转发 std 真正更强的几项、且只从 .cc 转发、公开头零 std string 头——比 clay 更克制。
- **StringBuf 的旧 1.24x churn 证书已撤回**：旧 peer 每次按 key re-find，
  仍弱于当前 generation-tagged direct-ID+mutex 对手。新形状下 Windows 0.990x
  统计平，WSL 0.813x、M4 0.785x 明确输；另一方面 cached hash 的 1KiB growth
  三平台赢 strongest available floor（Win/WSL 对 Absl 5.36/1.71x，M4 对 std
  1.67x），8-thread 距 Absl 约 4–5%。因此产品价值落在复杂度、稳定 identity、
  并发接近强 peer 和 batch，而不是虚构“所有 lifecycle 常数都赢”。

## container·path —— 纯词法路径操作（`path_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4,各 15 遍 `bench-stat.py` 中位数+spread**。`path.h` 是 HYLE 的纯字符串路径层(无 syscall),`extension`/`base`/`directory`/`normalize`/`split` 在 string_ref 视图上原地走分量、**零分配**。**三方对标(2026-07 补公平性)**:① `std::filesystem::path` 词法操作——drop-in std SOTA,但每次操作**构造一个 path 对象(堆分配 + 存字符串副本)**;② **`std::string_view` 手写反向扫**——同零分配视图 + 同 back-to-front 扫描形状、仅 API 不同,是**纯算法基线**(隔离"算法" vs "结构");③ HYLE。**HYLE≈② 证扫描算法已最优;②③一起碾 ① 是"零分配 vs 必分配"的结构差,非更巧的算法。**

| op | HYLE(Win/Lin) | std::string_view 手写(算法基线,Win/Lin) | std::filesystem(Win/Lin) | vs 算法基线 | vs filesystem | 结论 |
|---|---|---|---|---|---|---|
| **extension** | **2.26/1.77** | 2.13/1.60 | 152/129 | 双平台**平**(Win 1.01/Lin 0.90,gap«spread) | 66/73x | ✅平算法基线(证扫描最优)+ 碾 fs(结构差) |
| **base** | **4.59/4.25** | 8.41/4.07 | 169/123 | **Lin 平**(0.96)/Win 数字 1.84x=MSVC 侧 string_view 构造税,非算法优势 | 37/29x | ✅平算法基线(Lin 为准)+ 碾 fs |
| **directory** | **10.08/4.66** | 9.94/4.71 | 172/249 | 双平台**平**(Win 0.99/Lin 1.01) | 17/53x | ✅平算法基线 + 碾 fs |
| **normalize 词法折叠** | 69 ns(Win) | (无零分配对手) | 401 | — | **5.8x** | ✅赢 fs(就地折叠 vs 堆 path;PRODUCE 型无 string_view 对手) |
| **join_array 6 部件** | 29 ns(Win) | (无零分配对手) | 618 | — | **21x** | ✅赢 fs(buffer-fill vs N 次 path 分配) |

**诚实结论(2026-07 re-audit:补算法基线 + extension 真优化 + 反哺检查,双平台)**:

1. **对 `std::filesystem` 的 17–73x 领先是结构差,不是巧妙算法**——补上 `std::string_view` 手写反向扫这个**零分配算法基线**后诚实分离:extension/base/directory **双平台均与算法基线统计平局**,证明 HYLE 的扫描算法本身 = 纯零分配标量扫的最优形状;对 fs 的巨大倍数纯粹来自"HYLE 在视图上原地扫、零分配" vs "fs 每次构造 path 对象(堆分配 + 拷贝字符串)"。**⚠️单平台数字陷阱(方法学)**:Win 上 base 看似"反超算法基线 1.84x"(4.59 vs 8.41),但 **Linux 平手(4.25 vs 4.07)**——那 1.84x 是 **MSVC 对 `std::string_view` 构造/返回的额外开销**,不是 HYLE 算法优势;双平台一跑就现原形(同 dtype/time 那几次"单平台数当无条件结论"的教训)。诚实表述:**扫描算法与零分配基线平手,不反超**。normalize/join_array 是 PRODUCE 型(产出新路径),std 无零分配对手,只两方对标 fs。

2. **extension 真优化:正向全扫 → 反向扫遇分隔符停,40.4→2.26ns(~18x)**——re-audit 揪出的真算法低效:原 `extension()` **正向全扫整条路径**(深路径 54 字符每次全扫),而 base/directory 早已**反向扫遇分隔符即停**(只碰最后一个 component ~12 字符)。这精确解释了旧账本"extension 仅 3.4x、远低于 base/directory 15x+"的异常。改成同样的反向扫(latch 最后一个点,遇第一个非点字符即返回——正是 Python splitext 语义),extension 从 40.4ns 追平到 2.26ns,与算法基线 string_view 手写版统计平局。新增 6 个反向扫边界用例(`..foo`/`...`/`a..b`/`x.`/深路径/点在分隔符前)守卫语义,18/18 过。

3. **反哺检查(全表 error→path,无可反哺项)**——① **string_ref 的 SIMD 反向扫核**(`str_rfind_char` SSE2、`str_find_last_of` PCMPISTRI)对 path **不适用**:path 组件短(basename ~12-20 字符,远低于 SIMD 收益阈值,账本"string_ref SIMD 仅 multi-KB 扫才划算"),且这些核是**导出符号**(有 ~1ns call seam)、会破坏 extension/base/directory 的 `constexpr`。实测 base/directory 平手 string_view **标量**反向扫已印证——短组件标量扫即最优,无 SIMD 空间。② **string_algo/hash/mem/simd** 与 path 零分配视图扫**正交**(path 不分配、不 hash、不做数值核)。结论:path 的赢点(零分配结构 + 最优标量扫)自成一体,总表其他模块的优化无一能反哺。

## 次要模块（util/random、time/clock、obs/log、SwissTable iterate，`misc_bench.cc` + `swiss_table_bench.cc`）

> 环境:Win MSVC19.50 + Linux GCC14.2 + Apple M4 AppleClang（2026-07-19
> util/time/obs 收口复测）。nanobench。标尺 = std 等价物(同 OS 原语/同算法类),
> 或 HYLE 自有(无 std 对等物时诚实标注)。三平台数字按 Win/Linux/M4 排列；
> 高争用行只判断适用边界，不把高方差点值宣传成 SOTA。

| 场景 | HYLE | 标尺 | vs 标尺 | 结论 |
|---|---|---|---|---|
| **Rng next_u64**(xoshiro256++) | 1.59/1.38/2.26 ns | std::mt19937_64 2.32/1.44/3.47* | 1.46x/平/1.53x | Win/M4 赢，Linux 平；M4 std 行方差高 |
| **Rng runtime-varying range_u64** | 2.09/1.47/2.17 | std distribution 2.86/2.08/7.68 | 1.37/1.41/3.54x | ✅ 三平台赢且无偏 |
| **Rng C batch fill** | 4.99/0.67/0.65 | native C++ 5.16/0.68/0.65 | 全部平 | 类型擦除 ABI 被 batch 摊薄 |
| **SpinLock 无争用** | 3.71/4.79/1.01 | mutex 10.94/2.48/4.19 | 2.95x/0.52x/4.15x | 平台翻转，只限短低争用 |
| **SpinLock 8 threads** | 50.17/60.44/271.88 | mutex 50.59/39.08/20.98 | 平/0.65x/0.08x | ❌ 高争用选 mutex |
| **Ref fan-out** | 10.13/8.33/3.19 | shared_ptr 8.09/11.12/3.22 | 0.80x/1.34x/平 | 平台翻转；所有权形状而非普遍提速 |
| **SwissTable iterate**(for_each) | **2.48 ns** | std::unordered_map 8.73 | **3.5x** | ✅ 赢(扁平 ctrl 扫 vs 节点指针链) |
| **time clock / wheel / common** | → | → | → | **已迁出** [benchmarks-time.md](./benchmarks-time.md)（2026-07-19 三平台；勿用本表旧双平台 clock 行） |
| **log gated-out** | 2.03–2.11(Win)/1.05–1.09(WSL)/0.91–0.92(M4) | (无 std 对照) | — | relaxed level gate + branch；参数不求值、不格式化 |
| **log preformatted C++**(1 sink,无 I/O) | 22.48/13.07/8.85 ns | (无 std 对照) | — | mutex snapshot + per-slot active refs + callback；换取 quiescent unregister |
| **log preformatted C ABI**(1 C sink,无 I/O) | 23.93/16.03/10.68 ns | native C++ 上行 | +1.45/+2.96/+1.83 ns | POD bridge + opaque trampoline 的真实 ABI 成本很小 |
| **log formatted emit**(1 sink,无 I/O) | 81.9–84.6/69.5–71.4/50.9–53.5 ns | (无 std 对照) | — | `std::format_to_n` + safe fan-out；真实 I/O sink 通常远大于此 |

**诚实结论**:① Rng 的 runtime-varying range 三平台赢；`next_u64` 在
MSVC/AppleClang 赢而 GCC 平，不能再写成无条件 1.54x。真 Lemire 用一次 128 位乘
和罕见 threshold 修正；基准必须让 bound 在运行时变化，避免编译器把常量除数折成
魔数。C ABI 单值调用有 0.09–0.95ns seam，但 fill batch 三平台与 C++ 平。
② SpinLock 的结果按平台甚至反向，高争用时 Linux/M4 均输 mutex；它是低争用
短临界区工具，不是通用锁。Ref 也随工具链翻转，其价值是侵入式共享所有权和省控制
块，而非原子 RMW 必然更快。完整 util 红队、ABI 和 sanitizer 证据见
[`../milestones/util-module-closure.md`](../milestones/util-module-closure.md)。
③ **SwissTable iterate 赢 std::unordered_map 3.5x**——扁平 ctrl 字节扫 +
连续 slot(cache 友好)碾压节点指针链；`for_each` 是函数模板(内联、零分配)。
④ **time** 见 [benchmarks-time.md](./benchmarks-time.md) + time-module-closure。  
⑤ **obs 收口后三层性能**：宏 gate 0.91–2.11ns；
preformatted native 8.85–22.48ns；C ABI trampoline 再加 1.45–2.96ns。
完整红队见 [`../milestones/obs-module-closure.md`](../milestones/obs-module-closure.md)。

## 模糊路径匹配（container-06c，`fuzzy_bench.cc`）

> 环境:Linux GCC14.2 Release,nanobench。净新设计,抄 fzf `src/algo/algo.go` V1 两趟算法(forward 找子序列结束位 → backward 收紧起点 → 收紧窗口评分)。**两件事要证明**(非单点 ratio):时间复杂度 + bonus 开销。

| 场景 | ns/candidate | candidate/s | 结论 |
|---|---|---|---|
| **find_path_matches N=1,000** | 45.8 | 21.9M | ✅ 基准点 |
| **find_path_matches N=10,000** | 47.9 | 20.9M | ✅ **O(N):跨 10× 持平** |
| **find_path_matches N=100,000** | 47.8 | 20.9M | ✅ **O(N):跨 100× 仅 +4%(cache)** |
| **path_match_score(完整评分)** | 44.8 | 22.3M | vs floor 2.35x |
| **std 子序列 floor(无评分)** | 19.0 | 52.6M | 算法下限(纯 membership 扫) |

**诚实结论**:① **O(N) 时间复杂度实证**——ns/candidate 在 N=1k/10k/100k = 45.8/47.9/47.8,跨 **100× 规模仅 +4%**(cache 效应,非算法项),线性 scaling 铁证(两趟均单遍扫,无隐藏 rescore)。这正是选贪心两趟而非 fzf Smith-Waterman DP 的理由:DP 每候选 O(pattern×len) + 分配 H/C 表,对"10万路径每键全量重评分"不可接受。② **bonus 成本 2.35x vs 子序列 floor(44.8 vs 19.0ns)——诚实标价**:floor 只做"是否有序子序列"(每个 fuzzy 匹配器的不可约最小工作),HYLE 多付的是 fzf 级排序质量(charClass 分类 + boundary/camel/consecutive bonus + backward 收紧 pass)。这 2.35x 买的是"能排序"而非"仅能过滤"——quick-open UI 的刚需。③ **绝对吞吐 2200 万候选/秒**:10万路径全量重评分 ~4.5ms,交互式按键延迟无感。**find_path_matches 零堆分配**(定长最小堆,scores 存调用方缓冲,O(N log K))——契约优先,K=20 时 100k 候选约 4.3N 次比较,远优于全排序 O(N log N)。
