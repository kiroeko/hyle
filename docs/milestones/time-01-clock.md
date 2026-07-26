# time-01：单调钟、墙钟、deadline 与 sleep

| 字段 | 值 |
|---|---|
| 阶段 | P2 |
| 模块 | time/ |
| 依赖 | core 宏/头；无编译期 HYLE 模块依赖 |
| 状态 | 🟡 2026-07-26 strict 重审实现完成；Windows/WSL 已复证，Apple 最终形状待真机复证 |

## 交付能力

- `monotonic_ns()`：唯一可用于 elapsed/deadline/timeout 的稳态时钟。
- `realtime_ns()`：Unix epoch ns 墙钟，只用于日志/时间戳。
- `profiling_clock_ns()` + `profiling_clock_available()`：x86 invariant-TSC
  profiling 快路径；不支持时回退，绝不用于 deadline。
- `coarse_realtime_ns()`：Windows shared-data / Linux
  `CLOCK_REALTIME_COARSE` 的低分辨率日志时间戳。
- `sleep_for_ns/us/ms()`、`sleep_until_ns()`。
- `saturating_add_ns()`、`deadline_after_ns()`、`remaining_ns()`：统一消除
  `now + timeout` 溢出和过期减法下溢。
- chrono typed facade：`Duration`、monotonic/realtime time point、duration/deadline/sleep overload。
- `Instant`：C++ 轻量 elapsed 快照；便利方法钳到非负，两个 instant 相减保留有符号方向。

纯 C/FFI 对应入口在 `include/hyle/c/time.h`，指南见
[`docs/guide/c-time.md`](../guide/c-time.md)。

## 收口修整

1. C++ `monotonic_ns`/`realtime_ns` 改为 header-inline 的 `std::chrono`
   标准钟。此前共享构建中调用路径是 caller → `hyle_time.so` →
   libc/libstdc++，Linux 15-run 配对测出约 6–7.5% 真接缝损失；内联后只剩
   标准钟本身的一跳。C ABI 仍由 DLL 导出稳定 wrapper。
2. deadline 与单位乘法全部饱和；`Instant::elapsed_ns` 不再在未来 instant
   上无符号下溢。integral chrono duration 改为精确 ratio 算术，不再经 MSVC
   double-precision `long double` 而在 2^53 以上丢 ns；浮点路径仍单独 ceil/saturate。
3. Windows sleep 使用高分辨率 waitable timer；Linux absolute sleep 使用
   `clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME)`，只对 `EINTR` 重试。
4. 相对 sleep 委托绝对 deadline，早醒后仍以 monotonic deadline 为准。
5. coarse realtime 的 epoch 乘法饱和，POSIX clock 调用失败不再静默使用
   未初始化 timespec。
6. TSC timing fallback 从最多累计 255ms 的 busy-spin 改为 sleep-based sampling；按实际
   monotonic elapsed 算频率，首次校准不再占满一个 core，热读路径不变。

## 正确性边界

- `profiling_clock_ns` 无全局 max 钳制，跨核/恢复/VM 迁移可能产生毛刺；
  它的价值是高频 profiling，而不是“更快的 deadline 时钟”。
- `coarse_realtime_ns` 值的单位仍是 ns，但分辨率通常只有 1–15ms。
- OS 调度器仍可能让线程晚醒；接口保证不主动向下取整、不早返回，不承诺实时调度。
- 所有 deadline 均使用 absolute monotonic ns；墙钟跳变不得进入调度路径。

## 验收

- 25 个 clock 测试：单调性、epoch、typed facade、sleep 下限、coarse/TSC availability、
  deadline 饱和、2^53±/signed/unsigned/非整 ns 精确转换、过去/未来 sleep-until、
  bool duration、signed Instant subtraction 与 fresh-process 低 CPU 校准。
- 纯 C smoke 覆盖每个时钟与便利入口。
- sched/io 已改用 `deadline_after_ns`/`remaining_ns`，不再各自手写易溢出算术。
- clock benchmark 与 `std::chrono` 比较完全相同的“转换为非负 uint64 ns”
  工作，而不是拿 MSVC system_clock 的原生 100ns count 充 ns 基线。

## 参考

- `std::chrono::steady_clock/system_clock`
- MSVC `__msvc_chrono.hpp` QPC 频率特判
- Abseil TSC 频率配对校准思路
- Tracy invariant-TSC CPUID 探测

当前设计、用法和数字分别见 [`time-design`](../architecture/time-design.md)、
[`guide/time`](../guide/time.md) 与 [`benchmarks-time`](../performance/benchmarks-time.md)。
