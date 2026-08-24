# util 历史性能证据（2026-07-30）

> **2026-08-25 freshness gate：**本页固定于下述 2026-07 commit。util 已因 mem/container
> 下游变化与本轮总复审重新打开；相同 source identity、peer pin、编译合同与三平台重跑完成前，
> 这些点值不是当前发布证书。现行状态只看
> [`pre-release-readiness`](../milestones/pre-release-readiness.md)。

本页记录当时对 util-core 与 util-runtime 的签证。core 产品实现固定在 `3c32be6`，runtime 的错误传播
能力固定在 `1b148ef`，C nullable callback 定稿固定在 `70fa4e8`；Windows MSVC 19.50、WSL/native-ext4
GCC 14.2、Apple M4 AppleClang 21 均为
Release、15 个独立进程。core 保留 2026-07-29 的 Rng/Ref/SpinLock 证据；runtime 新增真实
Registry N=256 与 C RNG batch 消费形状，不以测试专用 API 或常态 benchmark 元数据换结果。

## util-core：方法与对手

- runner：`hyle_util_core_bench`，`bench-stat.py -n 15`；同时读取 absolute spread 与逐 run
  paired sign test。小于 1% 或正反执行顺序翻转的差异不写成产品输赢。
- PRNG：标准 `mt19937_64`、`uniform_int_distribution` 与 `std::shuffle` 代表普通 C++ 组合；
  xoshiro256++ 与 Lemire mapping 另按本地 Folly/标准同族源码核对。
- lock：oneTBB `spin_mutex` 是同合同成熟 peer，HYLE/oneTBB 正反顺序都跑；`std::mutex`
  只证明选择边界，不是要击败的 spin peer。
- ownership：`shared_ptr` fan-out 只对拍 retain/release 原子热路。它有 control block/weak/aliasing，
  与单指针 intrusive `Ref` 不是完整同语义类型。
- M4 以明确 user-interactive 与 utility QoS 各跑一组；Darwin 无公开硬 P/E affinity，
  两档都不冒充物理核心绑定。

表中 ns/op 顺序为 **Windows / Linux / M4 interactive**；ratio 是 HYLE/peer throughput 的
paired median，约值只用于阅读，判定看完整 n=15。

## util-core 2026-07 历史结果

| 场景 | HYLE ns | peer ns | ratio | 判定 |
|---|---:|---:|---:|---|
| `Rng::next_u64` | 1.550 / 1.385 / 2.264 | mt19937 2.243 / 1.455 / 2.509 | 1.451 / ~1.05 / 1.106 | ✅ 三平台 sign WIN |
| varying `range_u64` | 2.002 / 1.474 / 2.202 | std distribution 2.750 / 2.031 / 7.763 | 1.372 / ~1.38 / 3.520 | ✅ 三平台 WIN |
| iterator shuffle | 1.137 / 1.260 / 0.810 | `std::shuffle(Rng)` 5.597 / 1.345 / 3.677 | 4.858 / ~1.06 / ~4.54 | ✅ 三平台 sign WIN；同一 HYLE URBG |
| intrusive `Ref` fan-out 64 | 7.956 / 8.055 / 3.190 | shared_ptr 8.020 / 10.773 / 3.211 | ~1.00 / ~1.34 / ~1.01 | ✅ Win/M4 TIE，Linux WIN |
| SpinLock 无争用 | 3.758 / 4.570 / 0.904 | oneTBB 3.739 / 4.573 / 0.897 | 1.000 / 1.000 / 0.993 | ➖ 正反序同档 |
| SpinLock 2 threads | 8.379 / 7.490 / 2.405 | oneTBB 8.419 / 7.593 / 2.316 | 0.992 / 0.991 / 0.980 | ➖ 正反序合看同档；M4 正序小差在反序消失 |
| SpinLock 8 threads | 61.382 / 50.047 / 5.545 | oneTBB 59.219 / 51.146 / 5.518 | 0.995 / 1.021 / 1.000 | ➖ 正反序同档 |

M4 utility 的 SpinLock HYLE/oneTBB paired ratio 为无争用 `0.976`（反序 `0.993`）、2-thread
`0.991`（反序 `0.996`）、8-thread `0.979`（反序 `0.973`），均在 combined spread 内；
Rng/range/shuffle 仍分别 `1.117/3.502/4.517x`，Ref 仍 TIE。布局在三工具链保持
`SpinLock=1 byte`、`Ref<T>=1 pointer`、`Rng=32 bytes`。

## util-core 抓出的真实 ARM 缺陷

曾把成功的 `atomic<bool>::exchange` 从 acquire-release 削弱成 acquire-only，理由是语言层锁
获取只需 acquire。x86 指令未变，普通正确性/TSan 也全绿，但 M4 user-interactive 的 8-thread
handoff 从 oneTBB 同档跌到约 `0.478x`，正反序均 15/15 LOSE；无争用也约慢 11%。候选 A/B
只恢复 RMW release ordering 后，8-thread 回到 `1.000/0.999`，无争用回到同档，Windows/Linux
不退化。最终 `3c32be6` 保留 acquire-release。

这是 microarchitecture/codegen 约束的算法修复，不是为 benchmark 增加常态元数据或线程数
特化。它也说明“内存序越弱一定越快”不是可接受的性能论证：低层原语必须在真实 ISA 上与
成熟 peer 对拍。

## util-core 选择边界

Linux 无争用 `std::mutex` 约 2.38 ns，快于 SpinLock 4.57 ns；8-thread 高争用下 mutex 约
38.5 ns，也快于两种 spin lock 的约 50 ns。Windows/M4 可能反向。结论不是按平台硬编码，
而是只有“几个指令、低争用、不能睡眠”同时满足才用 SpinLock；否则用 mutex 或 sched 的
fiber-aware 原语。mem FiberStackPool/graph 首错槽保留短 spin；会分配、rehash 或执行未知
构造的路径不得使用它。sched 内 move/construct 泛型 `T` 的路径留 sched 严格审查裁决。

## util-core 复杂度与复跑

- `ScopeGuard` 构造/dismiss/invoke/析构、`Ref` copy/drop/cast/identity compare 都是 O(1)；
  hash 是指针 hash。`dynamic_pointer_cast` 的 RTTI 成本仍为实现定义的常数级类型查询。
- `Rng next_*` 期望 O(1)，`discard(n)` O(n)，`jump/long_jump` 固定 O(256)，shuffle O(n)。
- `type_name<T>()` 在常量求值中切片；`type_hash<T>()` 是 O(类型名长度) constexpr FNV-1a。

```bash
cmake --preset bench -DHYLE_COMPONENTS=util_core -DHYLE_BENCH_TBB=ON
cmake --build --preset bench --target hyle_util_core_bench
scripts/bench-stat.py build-bench/bin/hyle_util_core_bench -n 15 \
  --pair "std::mt19937_64" "HYLE Rng xoshiro256++" \
  --pair "std::uniform_int_distribution" "HYLE Rng::range_u64" \
  --pair "std::shuffle(Rng URBG)" "Rng::shuffle(iterators)" \
  --pair "oneTBB spin_mutex" "HYLE SpinLock" \
  --pair "std::shared_ptr" "HYLE intrusive Ref"
```

Apple 默认显式 user-interactive；utility 复跑在命令前加
`HYLE_BENCH_APPLE_QOS=utility`。因此该轮 util-core 在同合同成熟 peer 上没有未解释的
material LOSE。

## util-runtime：测试合同

- runner：`hyle_util_bench`，每行 15 个独立进程；Windows/Linux 固定到 performance-first CPU
  并请求 high priority，M4 分 user-interactive/utility 两组，Darwin 不伪称硬 P/E affinity。
- Registry 注册 256 个 11-byte backend 名，creator 返回一个真实对象指针。mutable peer 是
  `mutex + std::unordered_map`；frozen peer 是 Abseil `flat_hash_map`，另保留 std map 给 C
  adapter 的普通基线。
- fallible Registry 行使用 `Registry<Result<void*>>`；std/Abseil peer 同样执行 missing-to-NotFound
  分支和 `Result` factory，因此对比的是同一错误合同，不是假定必命中的弱 peer。
- Abseil immutable map 不含 owned help、priority、factory lifecycle 或并发 freeze，合同比 HYLE
  更弱，因而是严格的 lookup 下界；结果只有打平/胜出才支持热路径结论。
- `FrozenView` 是初始化结束后高频 dispatch 的推荐产品形状；普通 `Registry::create` 保留一次
  lifecycle acquire，以允许同一 API 安全跨越 mutable→frozen 阶段。
- C checked create 与 nullable hot create 语义不同。creator 直接返回对象；checked 传入已初始化
  的 OK `hyle_status` 以区分 missing/failure/successful-NULL，nullable 传 NULL 让 creator 跳过
  诊断。两者只用于量化诊断成本，不把 checked 入口慢于 hot 入口写成算法缺陷。
- 独立 11-byte hash 行只诊断 Registry 的组成原语，不是 util 公开能力，也不能取代整表查找。

下表 ns/op 为 **Windows / Linux / M4 interactive**；speedup 是逐 run paired median 的
peer/HYLE throughput 比，判定同时看 sign test 与 absolute spread。

## util-runtime 2026-07 历史结果

| 场景 | HYLE ns | peer ns | paired speedup | 判定 |
|---|---:|---:|---:|---|
| C RNG `fill_u64(1024)` | 4.955 / 0.608 / 0.697 | C++ loop 5.181 / 0.749 / 2.458 | 1.046 / 1.235 / 3.508 | ✅ 三平台 WIN |
| Registry mutable lookup+call | 16.865 / 8.909 / 10.686 | mutex+std 18.251 / 14.076 / 15.058 | 1.096 / 1.587 / 1.438 | ✅ 三平台 sign WIN |
| cached `FrozenView` lookup+call | 7.960 / 3.958 / 4.521 | Abseil 8.472 / 4.750 / 4.502 | 1.061 / 1.183 / 0.995 | ✅ Win/Linux WIN；M4 TIE |
| ordinary frozen Registry | 7.983 / 4.088 / 4.717 | Abseil 8.472 / 4.750 / 4.502 | 1.059 / 1.145 / 0.955 | ✅ Win/Linux WIN；M4 TIE |
| cached `FrozenView`, 8 readers | 1.703 / 1.044 / 1.022 | Abseil 1.720 / 1.223 / 1.045 | 1.002 / 1.178 / 0.986 | ✅ Linux WIN；Win/M4 TIE |
| ordinary Registry, 8 readers | 1.577 / 1.031 / 1.011 | Abseil 1.720 / 1.223 / 1.045 | 1.085 / 1.142 / 0.998 | ✅ Win/Linux WIN；M4 TIE |
| fallible `Result` FrozenView | 7.846 / 4.049 / 4.761 | exact-contract Abseil 8.645 / 5.035 / 4.765 | 1.103 / 1.243 / 1.001 | ✅ Win/Linux WIN；M4 TIE |
| C nullable frozen create | 7.824 / 5.583 / 5.733 | immutable std 10.901 / 10.300 / 6.122 | 1.388 / 1.814 / 1.068 | ✅ 三平台 sign WIN |
| C nullable create, 8 readers | 1.715 / 1.453 / 1.357 | immutable std 2.250 / 2.089 / 1.783 | 1.318 / 1.435 / 1.316 | ✅ 三平台 sign WIN |
| C checked create / nullable create | 11.436 / 7.575 / 6.518 | hot 7.824 / 5.583 / 5.733 | 0.687 / 0.745 / 0.879 | ℹ️ 共享 Status 诊断合同成本 |
| 11-byte `hash_bytes` | 0.939 / 0.872 / 0.666 | Abseil 1.506 / 0.749 / 0.743 | 1.603 / 0.858 / 1.115 | ℹ️ Win/M4 WIN；Linux 原语 LOSE，整表仍 WIN |

M4 utility 复跑中，推荐的 `FrozenView` 单线程为 4.524 vs Abseil 4.516 ns（TIE），8-reader
为 1.065 vs 1.042 ns（TIE）。普通 Registry 单线程 4.730 vs 4.516 ns，paired sign 判 LOSE
（仅 2/15 run 更快）、约慢 4.5%，absolute 仍 TIE；这正是生命周期 atomic 的可见成本，dispatch-heavy 用户
缓存 `FrozenView` 即消除。fallible Result 行 4.772 vs 4.769 ns（TIE）；mutable Registry 仍
1.395x WIN；C nullable 单/多线程仍 1.072/1.318x WIN；C RNG batch 仍 3.530x WIN。

## util-runtime 复杂度与结论

Registry lookup/create 是 O(|key|) hash + expected O(1) probes；register 另复制
O(|key|+|help|)，扩容摊销 expected O(1)；enumeration O(n)。mutable 枚举需要 O(n) 时间点快照，
callback 在锁外；frozen 枚举 O(1) 额外空间。freeze 只是 release-publish 同一张 Swiss index，
时间与额外空间均 O(1)、零分配。Registry 不按 N=256、线程数或 CPU 型号特化。

三平台 intended hot shapes 没有 material LOSE：raw/Result `FrozenView` 对成熟 immutable peer
为 WIN/TIE，C nullable hot path 对 std 基线全胜。Linux raw short-hash 落后 Abseil 约 14% 是保留的诊断事实，
但 Registry 的完整查找/所有权/调用路径仍胜出，不能为了赢一条孤立微基准扭曲公开 API。普通
Registry 在 M4 utility 付一次 lifecycle atomic；易用入口保留它，明确高频入口使用 FrozenView。

```bash
cmake --preset bench -DHYLE_COMPONENTS=util -DHYLE_BENCH_ABSL=ON
cmake --build --preset bench --target hyle_util_bench
scripts/bench-stat.py build-bench/bin/hyle_util_bench -n 15
```
