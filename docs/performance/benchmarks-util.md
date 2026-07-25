# util-core 性能证书（2026-07-25）

本页只给当前 `util-core` 头能力签证：`Rng`、`Ref`、`SpinLock` 以及无需运行期计时的
`ScopeGuard/type_name`。依赖 container 的 Registry、demangle target 和完整 C ABI 属于
util-runtime，等 container 严格复审后在本页续签，不能沿用 2026-07-19 的旧 mixed 表。

## 方法与标尺

- runner：`hyle_util_core_bench`，Release，nanobench；`bench-stat.py -n 15` 跨 15 个独立
  process 取中位数，并同时看 spread 与逐 run 配对 sign test。
- 平台：Windows x64 / MSVC 19.50，WSL2 x64 / GCC 14.2，Apple M4 / AppleClang 21。
- PRNG：`std::mt19937_64`、`std::uniform_int_distribution` 和 `std::shuffle` 验证正常组合成本；
  xoshiro256++ 状态转移与 Lemire bounded mapping 另按本地 Folly、Abseil/Rust 同族源码核对。
- lock：oneTBB `spin_mutex` 是同合同强 peer；`std::mutex` 只用来证明选择边界。oneTBB/HYLE
  顺序和 HYLE/oneTBB 反序都跑，避免把 1–2% 的频率/热身次序写成输赢。
- ownership：`std::shared_ptr` fan-out 与单指针 intrusive `Ref` 对拍 retain/release 正常路径；
  两者语义不同，结果只证明原子热路，不把 control-block 能力差异藏掉。

表中均为 ns/op，顺序固定为 **Win / Linux / M4**；ratio 是 HYLE/peer 的吞吐比。

## 当前结果

| 场景 | HYLE ns | peer ns | ratio | 当前判定 |
|---|---:|---:|---:|---|
| `Rng::next_u64` | 1.556 / 1.384 / 2.264 | mt19937_64 2.282 / 1.443 / 2.561 | 1.463 / 1.045 / 1.131 | ✅ 三平台 sign WIN |
| varying `range_u64` | 2.021 / 1.460 / 2.230 | std distribution 2.832 / 2.018 / 7.784 | 1.390 / 1.382 / 3.494 | ✅ 三平台 WIN |
| iterator shuffle | 1.159 / 1.255 / 0.824 | `std::shuffle(Rng)` 5.747 / 1.331 / 3.731 | 4.917 / 1.060 / 4.525 | ✅ 三平台 sign WIN；同一 HYLE URBG |
| intrusive `Ref` fan-out 64 | 8.312 / 7.976 / 3.202 | shared_ptr 8.248 / 10.779 / 3.211 | 1.006 / 1.353 / 1.003 | ✅ Win/M4 TIE，Linux WIN |
| SpinLock 无争用 | 3.830 / 4.574 / 0.935 | oneTBB 3.823 / 4.595 / 0.907 | 0.997 / 1.010 / 0.980 | ➖ 同档；反序仍为 0.997 / 0.994 / 0.995 附近 |
| SpinLock 2 threads | 9.834 / 7.800 / 1.793 | oneTBB 9.766 / 7.688 / 1.772 | 0.989 / 0.995 / 0.990 | ➖ 正反序均同档 |
| SpinLock 8 threads | 59.535 / 48.593 / 4.297 | oneTBB 61.646 / 48.451 / 4.187 | 1.038 / 1.022 / 0.982 | ➖ 正反序均无 material LOSE |

`SpinLock` 的 `sizeof` 在 MSVC/GCC/AppleClang 均为 1 byte；`Ref<T>` 为一个指针，`Rng`
为 32 bytes。旧实现给每把 SpinLock 无条件 pad 到 cache line，MSVC 对象为 64 bytes；当前
紧凑默认与 oneTBB/Folly 一致，只有 owner 经 profile 证明伪共享后才显式 padding。

## 结果怎样改变了实现

1. 旧 SpinLock 在进入 `lock()` 时先做 RMW，所有新 waiter 都会写已经被占用的 cache line；
   Apple M4 8-thread 曾出现约 272 ns 的灾难信号。当前从第一次 probe 起完整 TTAS，指数
   backoff 对齐 oneTBB，并在 AArch64 使用 oneTBB/Folly 同类 instruction barrier。当前 M4
   8-thread 为 4.297 ns，对 oneTBB 4.187 ns 统计同档；这是真实 cache-coherence 修复，不是
   为某一 benchmark 添加状态或特化线程数。
2. Ref 的单份 retain 原先为了极端 32-bit count overflow 总要消费 fetch-add 的旧值，使 MSVC
   发出 value-returning `lock xadd`；旧快照比 shared_ptr 慢约 20%。现在普通 `Ref` 单份热路
   采用成熟 intrusive-pointer 的 ownership 前置条件，显式批量 overload 继续做零/上下溢检查；
   Windows 当前回到 1.006x TIE，Linux 1.353x WIN，M4 TIE。
3. `Rng` 现在是标准 URBG，并提供自然 iterator shuffle；公开形状来自普通 C++ 组合需求。
   专用 iterator 路径仍复用 Lemire bounded draw，所以 Windows/M4 不必为标准库不同实现多付
   4–5 倍；用户仍可直接传给标准 distribution/shuffle。

## 选择边界，不隐藏失败样本

`std::mutex` 不是 SpinLock 的同合同 SOTA peer，但它揭示何时根本不该自旋：Linux 无争用
mutex 约 2.378 ns，明显快于 SpinLock 4.574 ns；Linux 8-thread 高争用下 mutex 约 38.9 ns，
SpinLock 约 48.6 ns 且方差很大。Windows/M4 的极短路径又可能反向。结论不是按平台硬编码
选锁，而是只有“几个指令、低争用、不能睡眠”同时满足才用 SpinLock；否则用 mutex 或 sched
的 fiber-aware 原语。`mem::LeakTracker` 和 `ConcurrentVector` 正因锁内可能分配/rehash/执行
用户构造函数，已改回 mutex。

## 无需微基准的复杂度项

- `ScopeGuard` 构造、dismiss/invoke 和析构均 O(1)，只保存 callable+active bit；没有 heap
  是 callable 自己的属性，不能由 guard 假承诺。
- `type_name<T>()` 在常量求值中切片；`type_hash<T>()` 是 O(类型名长度) 的 constexpr FNV-1a，
  不进入运行期热路径，也不承诺跨编译器稳定。
- `Rng::discard(n)` 明确 O(n)；`jump/long_jump` 是固定 O(256)，shuffle 是 O(n)。
- `Ref` copy/drop 是 O(1) 原子操作；没有 weak/control block 是能力边界，不是漏测的优化。

因此 util-core 当前性能结论是：同合同成熟 peer 上没有未解释的 material LOSE，算法复杂度与
布局达到同类成熟形状；mutex 反例已进入用户选型合同。util-runtime 尚未取得这份结论。
