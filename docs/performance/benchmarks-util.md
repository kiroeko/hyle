# util-core 性能证书（2026-07-29）

本页只给 `Rng`、`Ref`、`SpinLock` 以及无需运行期计时的 `ScopeGuard/type_name` 签证。
产品实现固定在 `3c32be6`，Apple QoS 仪器固定在 `fde9b41`；Windows MSVC 19.50、
WSL/native-ext4 GCC 14.2、Apple M4 AppleClang 21 均为 Release、15 个独立进程。
Registry/demangle/C adapter 属于 util-runtime，不能继承本页结论。

## 方法与对手

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

## 当前结果

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

## 本轮抓出的真实 ARM 缺陷

曾把成功的 `atomic<bool>::exchange` 从 acquire-release 削弱成 acquire-only，理由是语言层锁
获取只需 acquire。x86 指令未变，普通正确性/TSan 也全绿，但 M4 user-interactive 的 8-thread
handoff 从 oneTBB 同档跌到约 `0.478x`，正反序均 15/15 LOSE；无争用也约慢 11%。候选 A/B
只恢复 RMW release ordering 后，8-thread 回到 `1.000/0.999`，无争用回到同档，Windows/Linux
不退化。最终 `3c32be6` 保留 acquire-release。

这是 microarchitecture/codegen 约束的算法修复，不是为 benchmark 增加常态元数据或线程数
特化。它也说明“内存序越弱一定越快”不是可接受的性能论证：低层原语必须在真实 ISA 上与
成熟 peer 对拍。

## 选择边界

Linux 无争用 `std::mutex` 约 2.38 ns，快于 SpinLock 4.57 ns；8-thread 高争用下 mutex 约
38.5 ns，也快于两种 spin lock 的约 50 ns。Windows/M4 可能反向。结论不是按平台硬编码，
而是只有“几个指令、低争用、不能睡眠”同时满足才用 SpinLock；否则用 mutex 或 sched 的
fiber-aware 原语。mem FiberStackPool/graph 首错槽保留短 spin；会分配、rehash 或执行未知
构造的路径不得使用它。sched 内 move/construct 泛型 `T` 的路径留 sched 严格审查裁决。

## 复杂度与复跑

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
`HYLE_BENCH_APPLE_QOS=utility`。因此 util-core 当前在同合同成熟 peer 上没有未解释的
material LOSE；util-runtime 尚未取得这份结论。
